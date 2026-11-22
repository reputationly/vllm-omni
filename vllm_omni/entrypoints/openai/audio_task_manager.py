# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FIFO task manager + backpressure for the async audio task API.

The generic in-memory store/registry (``stores.py``) has no queue bound; the
GPUStack async contract requires FIFO admission with 503 backpressure. This
mirrors LightX2V ``server/task_manager.py``: admission raises ``RuntimeError``
when the queue is full (the route turns that into HTTP 503).

Task state lives in ``AUDIO_TASK_STORE``; the running ``asyncio.Task`` lives in
``AUDIO_TASKS``. This manager only owns admission control, queue accounting,
and cancellation — the background job itself is started by the route (which has
the speech handler and app state).
"""

import asyncio
import os
import time
from collections.abc import Callable
from pathlib import Path

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio_tasks import (
    AudioTaskResponse,
    AudioTaskStatus,
)
from vllm_omni.entrypoints.openai.stores import AUDIO_TASK_STORE, AUDIO_TASKS

logger = init_logger(__name__)

_TERMINAL = (
    AudioTaskStatus.COMPLETED,
    AudioTaskStatus.FAILED,
    AudioTaskStatus.CANCELLED,
)
_ACTIVE = (AudioTaskStatus.PENDING, AudioTaskStatus.PROCESSING)


def resolve_save_path(save_result_path: str, task_id: str, output_root: str, default_ext: str = ".wav") -> str:
    """Resolve the output path (semantics per LightX2V ``file_service``).

    Absolute -> used verbatim; relative -> under ``output_root``; empty ->
    ``task_id``; missing suffix -> ``default_ext`` appended.

    ``default_ext`` only applies when the caller gave no suffix at all. When a
    caller DOES pass one, it owns it — see :func:`check_save_path_matches_format`,
    which refuses a path whose suffix contradicts ``response_format``.
    """
    raw = save_result_path or task_id
    path = Path(raw)
    if not path.is_absolute():
        path = Path(output_root) / raw
    if not path.suffix:
        path = path.with_suffix(default_ext)
    return str(path)


#: ``response_format`` -> the one suffix the artifact may carry.
FORMAT_SUFFIXES: dict[str, str] = {
    "wav": ".wav",
    "mp3": ".mp3",
    "flac": ".flac",
    "opus": ".opus",
    "pcm": ".pcm",
}


def check_save_path_matches_format(save_result_path: str, response_format: str | None) -> str | None:
    """Return an error message when the artifact's name contradicts its bytes.

    The facade (GPUStack) names the artifact; the engine writes the container the
    request asked for. Both are correct on their own, and the combination is
    corrupt: mp3 bytes in a ``.wav`` file, which every downstream hop then labels
    ``audio/wav``. The failure is silent and expensive to trace, so it is checked
    here — the one place where both facts are visible before any audio is written.

    Returns an error for a caller to surface as 400 (fail closed: a 400 costs one
    round trip, a mislabelled artifact costs a support ticket).
    """
    requested = (response_format or "").strip().lower()
    expected = FORMAT_SUFFIXES.get(requested)
    if expected is None:
        return None
    actual = Path(save_result_path).suffix.lower()
    if not actual or actual == expected:
        return None
    return (
        f"save_result_path ends with '{actual}' but response_format='{requested}' "
        f"produces {expected} bytes; the caller naming the artifact must use "
        f"the same container it asked the engine for"
    )


def visible_task_status(status: AudioTaskStatus, executing: bool | None) -> AudioTaskStatus:
    """The status to report for a task the engine has accepted.

    This route admits several jobs at once and starts a coroutine for each, so
    PROCESSING really means "accepted" — the diffusion scheduler still runs them
    one at a time. A job waiting its turn is reported as pending, matching
    LightX2V, whose FIFO worker only flips to processing when it picks the task
    up. It matters downstream: the GPUStack facade starts an elapsed-time
    progress estimate the moment a task looks like it is running, and under
    backpressure that clock would tick through the whole queue wait — parking the
    bar near its ceiling before the job had done any work.

    ``executing`` is tri-state and None must NOT be read as "queued": it means
    the scheduler has no state for the request — not submitted yet, already
    finished, or a request that never goes through a diffusion stage at all
    (plain TTS). Only an explicit False demotes.

    Both stage clients answer it: the inline one reads the scheduler directly,
    the out-of-process one reads the value its subprocess pumps over the
    response socket (at most one pump interval stale, which only widens the
    window in which a just-queued job still reports processing). A client with
    no such probe at all reports None for every request and therefore never
    demotes — ``AsyncOmni.supports_live_progress`` exists to tell that apart
    from "nothing to say about this one request" so the gap can be logged
    rather than silently degrading into the queue-wait estimate above.
    """
    if status == AudioTaskStatus.PROCESSING and executing is False:
        return AudioTaskStatus.PENDING
    return status


class AudioTaskManager:
    def __init__(self, max_queue_size: int = 8, *, store=None, tasks=None) -> None:
        # store/tasks default to the module-level singletons; injectable for tests.
        self._store = store if store is not None else AUDIO_TASK_STORE
        self._tasks = tasks if tasks is not None else AUDIO_TASKS
        self._lock = asyncio.Lock()
        self._max = max_queue_size

    async def reserve(self, task_id: str, save_result_path: str) -> AudioTaskResponse:
        """Admit a task: reject duplicates and enforce the queue bound, then
        create the PENDING record. Raises ``RuntimeError`` when full or on a
        duplicate id (the route maps these to HTTP 503)."""
        async with self._lock:
            if await self._store.get(task_id) is not None:
                raise RuntimeError(f"Task ID {task_id} already exists")
            items = await self._store.list_values()
            active = sum(1 for t in items if t.status in _ACTIVE)
            if active >= self._max:
                raise RuntimeError(f"Task queue is full (max {self._max} tasks)")
            resp = AudioTaskResponse(
                task_id=task_id,
                status=AudioTaskStatus.PENDING,
                save_result_path=save_result_path,
            )
            await self._store.upsert(task_id, resp)
            return resp

    async def get_status(self, task_id: str) -> AudioTaskResponse | None:
        return await self._store.get(task_id)

    async def list_tasks(self) -> list[AudioTaskResponse]:
        return await self._store.list_values()

    async def queue_status(self, executing: Callable[[str], bool | None] | None = None) -> dict:
        """Queue-wide view, bucketed by the SAME rule as the per-task endpoint.

        ``executing`` is the engine's execution probe. Without it this reports
        raw stored status, which contradicts ``GET /v1/tasks/{id}/status``: a
        backpressured job shows there as pending while still holding
        ``current_task`` here. ``active_count`` (and therefore admission) is
        unaffected either way — pending and processing both count as active.
        """
        items = await self._store.list_values()
        visible = [(t, visible_task_status(t.status, executing(t.task_id) if executing else None)) for t in items]
        pending = sum(1 for _, status in visible if status == AudioTaskStatus.PENDING)
        processing = [t for t, status in visible if status == AudioTaskStatus.PROCESSING]
        active = pending + len(processing)
        return {
            "is_processing": len(processing) > 0,
            "current_task": processing[0].task_id if processing else None,
            "pending_count": pending,
            "active_count": active,
            "queue_size": self._max,
            "queue_available": max(0, self._max - active),
        }

    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending/processing task. Returns False if unknown or already
        terminal."""
        job = await self._store.get(task_id)
        if job is None or job.status in _TERMINAL:
            return False
        task = await self._tasks.get(task_id)
        if task is not None:
            task.cancel()
        await self._store.update_fields(task_id, {"status": AudioTaskStatus.CANCELLED, "end_time": time.time()})
        return True


AUDIO_TASK_MANAGER = AudioTaskManager(max_queue_size=int(os.environ.get("VLLM_OMNI_AUDIO_MAX_QUEUE", "8")))
