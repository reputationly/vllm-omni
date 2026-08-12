# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request generation progress, reported out of the worker process.

A pipeline calls :func:`report_phase` as it moves through prepare / encode /
denoise / decode; the worker turns those calls into ``AsyncOutputKind.PROGRESS``
messages on the existing ``result_mq``, and the executor's result pump parks the
latest one per request so the API server can answer ``GET
/v1/tasks/{id}/status`` with a live phase instead of an opaque "processing".

Module-level state, mirroring ``forward_context``: the reporting sites are deep
inside pipeline code that has no handle on the worker, and threading one through
every call would touch far more than it is worth.

Three invariants keep this off the critical path:

* **No sink, no cost.** Offline inference, unit tests and any direct pipeline
  call have no sink registered, so ``report_phase`` returns immediately.
* **Throttled.** ``result_mq`` is a fixed-size shared-memory ring; a per-step
  message from every rank is pure noise. One message per phase change, and
  within a phase at most one per _MIN_INTERVAL_S once it has advanced
  _MIN_DELTA points — and only from the primary rank of each replica.
* **Never fatal.** Any sink failure is swallowed. Progress is a nicety; a
  generation that dies because a status message could not be enqueued is not.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)

# Controlled phase vocabulary shared with the GPUStack facade's weight table.
# A name outside this set is dropped by the facade, so keep them in sync (see
# gpustack docs/视频任务进度上报-统一契约设计.md).
PHASE_PREPARE = "prepare"
PHASE_ENCODE = "encode"
PHASE_DENOISE = "denoise"
PHASE_DECODE = "decode"
PHASE_SAVE = "save"

_MIN_INTERVAL_S = 0.5
_MIN_DELTA = 1.0


@dataclass(frozen=True)
class ProgressEvent:
    """One progress observation. Crosses the process boundary as the payload of
    an ``AsyncDiffusionOutput``, so keep it small and picklable."""

    request_id: str
    phase: str
    phase_progress: float


@dataclass
class _Scope:
    request_id: str
    phase: str | None = None
    last_progress: float = -1.0
    last_emit: float = 0.0


_sink: Callable[[ProgressEvent], None] | None = None
_scope: _Scope | None = None
_primary_rank: bool | None = None


def set_progress_sink(sink: Callable[[ProgressEvent], None] | None) -> None:
    """Install the process-wide consumer of progress events (the worker's
    ``result_mq`` writer). ``None`` disables reporting."""
    global _sink
    _sink = sink


@contextmanager
def progress_scope(request_id: str | None) -> Iterator[None]:
    """Attribute everything reported inside the block to ``request_id``.

    ``None`` disables reporting for the duration — used for multi-request
    batches, where a single "which phase" answer would be a lie.
    """
    global _scope
    previous = _scope
    _scope = _Scope(request_id=request_id) if request_id else None
    try:
        yield
    finally:
        _scope = previous


def _is_primary_rank() -> bool:
    """True on the one rank per replica that should report.

    Same predicate the worker uses to decide which rank replies to an RPC: under
    SP/TP/CFG/PP every rank runs the *same* request, so without this the queue
    gets N copies of every event. Under DP the ranks run *different* requests and
    each is primary in its own replica, which is exactly what we want. Resolved
    lazily and cached — the parallel state is not initialized when the sink is
    installed.
    """
    global _primary_rank
    if _primary_rank is not None:
        return _primary_rank
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        from vllm_omni.diffusion.distributed.parallel_state import (
            get_classifier_free_guidance_rank,
            get_pipeline_parallel_rank,
            get_sequence_parallel_rank,
        )

        _primary_rank = (
            get_sequence_parallel_rank() == 0
            and get_classifier_free_guidance_rank() == 0
            and get_tensor_model_parallel_rank() == 0
            and get_pipeline_parallel_rank() == 0
        )
    except Exception:
        # Not distributed (or state not up yet): this process is the only one
        # that could report, so let it.
        _primary_rank = True
    return _primary_rank


def _should_emit(scope: _Scope, phase: str, progress: float, now: float) -> bool:
    """Phase changes always go out; within a phase both gates must pass.

    Requiring both bounds the rate no matter the pipeline: a 50-step video job
    emits per step (they are seconds apart), while a fast image pipeline running
    the same loop in two seconds emits a handful, not fifty.
    """
    if phase != scope.phase:
        return True
    return progress - scope.last_progress >= _MIN_DELTA and now - scope.last_emit >= _MIN_INTERVAL_S


def report_phase(phase: str, completed: float | None = None, total: float | None = None) -> None:
    """Report position within ``phase``.

    Without counters this means "entering this phase" (0% through it); with them
    it means ``completed``/``total`` of the way through — the natural call inside
    a denoise loop. The global percentage is NOT computed here: the facade owns
    the stage cost model, so an engine never has to guess how expensive its own
    VAE decode is relative to sampling.
    """
    sink, scope = _sink, _scope
    if sink is None or scope is None:
        return
    if total:
        progress = max(0.0, min(100.0, float(completed or 0.0) * 100.0 / float(total)))
    else:
        progress = 0.0

    now = time.monotonic()
    if not _should_emit(scope, phase, progress, now):
        return
    if not _is_primary_rank():
        return

    scope.phase = phase
    scope.last_progress = progress
    scope.last_emit = now
    try:
        sink(ProgressEvent(request_id=scope.request_id, phase=phase, phase_progress=progress))
    except Exception:
        logger.debug("Progress report dropped for request %s", scope.request_id, exc_info=True)
