# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for mid-forward progress reporting.

The properties locked here are the ones that keep a status nicety from becoming
a liability on the generation path: silence when nobody is listening, a bounded
message rate whatever the pipeline's step cadence, one reporter per replica, and
failures that stay inside the reporter.
"""

import time
from unittest.mock import MagicMock

import pytest

from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind
from vllm_omni.diffusion.progress import (
    PHASE_DENOISE,
    PHASE_ENCODE,
    PHASE_PREPARE,
    ProgressEvent,
    progress_scope,
    report_phase,
    set_progress_sink,
)
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def sink():
    """Collect events, and always tear the sink down — it is module state, and a
    leaked one would report into another test's assertions."""
    events: list[ProgressEvent] = []
    set_progress_sink(events.append)
    try:
        yield events
    finally:
        set_progress_sink(None)


class TestReportPhase:
    def test_no_sink_is_a_no_op(self):
        # Offline inference and direct pipeline calls take this path.
        set_progress_sink(None)
        with progress_scope("req-1"):
            report_phase(PHASE_DENOISE, 1, 10)  # must not raise

    def test_no_scope_is_a_no_op(self, sink):
        report_phase(PHASE_DENOISE, 1, 10)
        assert sink == []

    def test_phase_changes_always_emit(self, sink):
        with progress_scope("req-1"):
            report_phase(PHASE_PREPARE)
            report_phase(PHASE_ENCODE)
        assert [(e.phase, e.phase_progress) for e in sink] == [
            (PHASE_PREPARE, 0.0),
            (PHASE_ENCODE, 0.0),
        ]

    def test_counters_become_a_percentage_of_the_phase(self, sink):
        with progress_scope("req-1"):
            report_phase(PHASE_DENOISE, 16, 32)
        assert sink[-1] == ProgressEvent(request_id="req-1", phase=PHASE_DENOISE, phase_progress=50.0)

    def test_events_carry_the_scoped_request_id(self, sink):
        with progress_scope("req-a"):
            report_phase(PHASE_DENOISE, 1, 10)
        with progress_scope("req-b"):
            report_phase(PHASE_DENOISE, 1, 10)
        assert [e.request_id for e in sink] == ["req-a", "req-b"]

    def test_batch_scope_reports_nothing(self, sink):
        # A multi-request batch has no single answer to "which phase are you in".
        with progress_scope(None):
            report_phase(PHASE_DENOISE, 5, 10)
        assert sink == []

    def test_scope_is_restored_after_the_block(self, sink):
        with progress_scope("outer"):
            with progress_scope("inner"):
                report_phase(PHASE_DENOISE, 1, 10)
            report_phase(PHASE_ENCODE)
        assert [e.request_id for e in sink] == ["inner", "outer"]

    def test_a_fast_loop_is_rate_limited(self, sink):
        # 50 steps in microseconds: without the time gate this floods the shm
        # ring with a message per step per rank.
        with progress_scope("req-1"):
            for step in range(1, 51):
                report_phase(PHASE_DENOISE, step, 50)
        assert len(sink) <= 2

    def test_a_slow_loop_reports_every_step(self, sink):
        with progress_scope("req-1"):
            for step in range(1, 4):
                report_phase(PHASE_DENOISE, step, 3)
                time.sleep(0.6)
        assert len(sink) == 3

    def test_a_failing_sink_never_reaches_the_caller(self):
        def boom(_event):
            raise RuntimeError("queue is full")

        set_progress_sink(boom)
        try:
            with progress_scope("req-1"):
                report_phase(PHASE_DENOISE, 1, 10)  # must not raise
        finally:
            set_progress_sink(None)

    def test_zero_total_does_not_divide_by_zero(self, sink):
        with progress_scope("req-1"):
            report_phase(PHASE_DENOISE, 1, 0)
        assert sink[-1].phase_progress == 0.0


class TestWorkerEnqueue:
    def _worker(self, result_mq):
        proc = object.__new__(WorkerProc)
        proc.result_mq = result_mq
        return proc

    def test_event_is_wrapped_in_a_progress_envelope(self):
        result_mq = MagicMock()
        event = ProgressEvent(request_id="req-1", phase=PHASE_DENOISE, phase_progress=25.0)

        self._worker(result_mq)._enqueue_progress(event)

        (msg,), _ = result_mq.enqueue.call_args
        assert isinstance(msg, AsyncDiffusionOutput)
        assert msg.kind == AsyncOutputKind.PROGRESS
        assert msg.result is event
        # No rpc_id / async_output_id: this message resolves no future, and the
        # result pump must not try to complete one with it.
        assert msg.rpc_id is None and msg.async_output_id is None

    def test_ranks_without_a_result_queue_stay_silent(self):
        worker = self._worker(None)
        worker._enqueue_progress(ProgressEvent(request_id="r", phase=PHASE_DENOISE, phase_progress=1.0))

    def test_a_broken_queue_does_not_break_the_forward(self):
        result_mq = MagicMock()
        result_mq.enqueue.side_effect = RuntimeError("shm closed")
        worker = self._worker(result_mq)
        worker._enqueue_progress(ProgressEvent(request_id="r", phase=PHASE_DENOISE, phase_progress=1.0))
