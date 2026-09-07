# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for ExclusiveResidency (CPU-only; engines are fakes).

These cover the four hard constraints transcribed from the HunyuanImage-3.0 A100
report §11.1 into vllm_omni/entrypoints/openai/stage_residency.py, and the
regression item from that report's §12.5 checklist: "异常/取消后不会留下两个
awake engine".
"""

import asyncio
from dataclasses import dataclass, field

import pytest

from vllm_omni.entrypoints.openai.stage_residency import (
    ExclusiveResidency,
    ResidencyError,
    ResidencyGroup,
)

# Level + hardware marks are what Buildkite collects on (tests/helpers/mark.py);
# cpu because every engine here is a fake and nothing touches an accelerator.
pytestmark = [pytest.mark.asyncio, pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class FakeAck:
    status: str = "SUCCESS"
    stage_id: int | None = 0
    rank: int | None = 0
    freed_bytes: int = 1 << 30
    error_msg: str | None = None


@dataclass
class FakeEngine:
    """Minimal AsyncOmni residency surface, recording the transition order."""

    name: str
    log: list[str] = field(default_factory=list)
    wake_error: Exception | None = None
    wake_ack_status: str = "SUCCESS"
    sleep_ack_status: str = "SUCCESS"
    sleep_freed_bytes: int = 1 << 30
    transition_delay_s: float = 0.0
    awake: bool = False

    async def wake_up(self, stage_ids=None, tags=None):
        if self.transition_delay_s:
            await asyncio.sleep(self.transition_delay_s)
        if self.wake_error is not None:
            self.log.append(f"{self.name}:wake_raise")
            raise self.wake_error
        self.awake = True
        self.log.append(f"{self.name}:wake")
        return [FakeAck(status=self.wake_ack_status)]

    async def sleep(self, stage_ids=None, level=2, mode="abort"):
        if self.transition_delay_s:
            await asyncio.sleep(self.transition_delay_s)
        self.awake = False
        self.log.append(f"{self.name}:sleep(level={level})")
        return [FakeAck(status=self.sleep_ack_status, freed_bytes=self.sleep_freed_bytes)]


def _residency(*engines: FakeEngine, **kwargs) -> ExclusiveResidency:
    return ExclusiveResidency([ResidencyGroup(label=e.name, engine=e) for e in engines], **kwargs)


async def test_prepare_sleeps_every_group_at_level_1():
    ar, dit = FakeEngine("ar"), FakeEngine("dit")
    residency = _residency(ar, dit)

    await residency.prepare()

    # Level 1 is load-bearing: level 2 discards weights and cannot be woken.
    assert ar.log == ["ar:sleep(level=1)"]
    assert dit.log == ["dit:sleep(level=1)"]


async def test_sequential_phases_never_overlap():
    """Constraint 1: AR must be asleep again before DiT wakes."""
    shared: list[str] = []
    ar, dit = FakeEngine("ar", log=shared), FakeEngine("dit", log=shared)
    residency = _residency(ar, dit)
    await residency.prepare()
    shared.clear()

    async with residency.session(request_id="r1") as sess:
        async with sess.awake("ar"):
            assert ar.awake and not dit.awake
        async with sess.awake("dit"):
            assert dit.awake and not ar.awake

    assert shared == [
        "ar:wake",
        "ar:sleep(level=1)",
        "dit:wake",
        "dit:sleep(level=1)",
    ]
    assert not ar.awake and not dit.awake


async def test_nested_awake_is_refused():
    """Constraint 1: a second concurrent wake inside one session is an error."""
    ar, dit = FakeEngine("ar"), FakeEngine("dit")
    residency = _residency(ar, dit)
    await residency.prepare()

    with pytest.raises(ResidencyError, match="still awake"):
        async with residency.session(request_id="r1") as sess:
            async with sess.awake("ar"):
                async with sess.awake("dit"):
                    pass

    # The outer group is still returned to sleep on the way out.
    assert not ar.awake and not dit.awake


async def test_wake_failure_does_not_wake_the_other_group():
    """Constraint 2: after a failed wake, nothing else gets woken."""
    ar = FakeEngine("ar", wake_error=RuntimeError("cumem restore failed"))
    dit = FakeEngine("dit")
    residency = _residency(ar, dit)
    await residency.prepare()
    ar.log.clear()
    dit.log.clear()

    with pytest.raises(ResidencyError, match="wake_up\\(ar\\) raised"):
        async with residency.session(request_id="r1") as sess:
            async with sess.awake("ar"):
                pytest.fail("body must not run when wake fails")

    assert dit.log == []  # DiT never touched
    assert not dit.awake


async def test_partial_wake_ack_failure_is_still_put_back_to_sleep():
    """A non-SUCCESS ACK means partially warm; it must not linger untracked."""
    ar = FakeEngine("ar", wake_ack_status="ERROR")
    residency = _residency(ar)
    await residency.prepare()
    ar.log.clear()

    with pytest.raises(ResidencyError, match="wake_up\\(ar\\) failed"):
        async with residency.session(request_id="r1") as sess:
            async with sess.awake("ar"):
                pass

    # Swept by session()'s forced sleep so the next request starts clean.
    assert "ar:sleep(level=1)" in ar.log
    assert not ar.awake


async def test_sleep_reporting_zero_freed_bytes_warns_but_proceeds():
    """Constraint 3 is advisory, not fatal.

    AR/LLM workers under-report freed_bytes: the allocator can log "sleep freed
    23.77 GiB" while the worker's own accounting logs "Freed 0.00 GiB" in the same
    instant. Raising here blocked a deployment whose sleep was working.
    """
    ar = FakeEngine("ar", sleep_freed_bytes=0)
    residency = _residency(ar)

    await residency.prepare()

    assert ar.log == ["ar:sleep(level=1)"]
    assert not ar.awake
    # The group must still be usable afterwards, not stuck resident.
    async with residency.session(request_id="r1") as sess:
        async with sess.awake("ar"):
            assert ar.awake


async def test_expect_freed_bytes_opt_out():
    ar = FakeEngine("ar", sleep_freed_bytes=0)
    residency = ExclusiveResidency([ResidencyGroup("ar", ar, expect_freed_bytes=False)])
    await residency.prepare()
    assert ar.log == ["ar:sleep(level=1)"]


async def test_body_exception_still_sleeps():
    ar = FakeEngine("ar")
    residency = _residency(ar)
    await residency.prepare()
    ar.log.clear()

    with pytest.raises(ValueError, match="boom"):
        async with residency.session(request_id="r1") as sess:
            async with sess.awake("ar"):
                raise ValueError("boom")

    assert ar.log == ["ar:wake", "ar:sleep(level=1)"]
    assert not ar.awake


async def test_cancellation_still_restores_residency():
    """Constraint 4 / report §12.5: cancel must not leave a group awake.

    The sleep transition is deliberately slow so a naive `await` inside `finally`
    would be cancelled before it completed.
    """
    ar = FakeEngine("ar", transition_delay_s=0.05)
    residency = _residency(ar)
    await residency.prepare()
    ar.log.clear()
    entered = asyncio.Event()

    async def one_request():
        async with residency.session(request_id="r1") as sess:
            async with sess.awake("ar"):
                entered.set()
                await asyncio.sleep(10)

    task = asyncio.create_task(one_request())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert ar.log == ["ar:wake", "ar:sleep(level=1)"], "sleep must complete despite cancellation"
    assert not ar.awake


async def test_cancelled_request_releases_mutex_for_the_next_one():
    """The mutex must not be stranded, or the model wedges after one cancel."""
    ar = FakeEngine("ar", transition_delay_s=0.05)
    residency = _residency(ar)
    await residency.prepare()
    entered = asyncio.Event()

    async def cancelled_request():
        async with residency.session(request_id="r1") as sess:
            async with sess.awake("ar"):
                entered.set()
                await asyncio.sleep(10)

    task = asyncio.create_task(cancelled_request())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async with residency.session(request_id="r2") as sess:
        async with sess.awake("ar"):
            assert ar.awake


async def test_requests_are_serialized():
    """Two concurrent requests must not interleave their awake windows."""
    ar = FakeEngine("ar")
    residency = _residency(ar)
    await residency.prepare()
    ar.log.clear()
    concurrent = 0
    peak = 0

    async def one_request(rid: str):
        nonlocal concurrent, peak
        async with residency.session(request_id=rid) as sess:
            async with sess.awake("ar"):
                concurrent += 1
                peak = max(peak, concurrent)
                await asyncio.sleep(0.01)
                concurrent -= 1

    await asyncio.gather(*(one_request(f"r{i}") for i in range(4)))

    assert peak == 1
    assert ar.log.count("ar:wake") == 4
    assert ar.log.count("ar:sleep(level=1)") == 4


async def test_acquire_timeout_surfaces_as_timeout_error():
    """Callers that must reject rather than queue map this to a busy response."""
    ar = FakeEngine("ar")
    residency = _residency(ar, acquire_timeout_s=0.01)
    await residency.prepare()
    holding = asyncio.Event()

    async def hog():
        async with residency.session(request_id="hog"):
            holding.set()
            await asyncio.sleep(0.5)

    task = asyncio.create_task(hog())
    await holding.wait()
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        async with residency.session(request_id="loser"):
            pass
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_level_2_is_rejected_at_construction():
    ar = FakeEngine("ar")
    with pytest.raises(ValueError, match="sleep_level must be 1"):
        _residency(ar, sleep_level=2)


async def test_unknown_label_and_duplicate_labels():
    ar = FakeEngine("ar")
    residency = _residency(ar)
    await residency.prepare()

    with pytest.raises(ResidencyError, match="unknown residency group"):
        async with residency.session(request_id="r1") as sess:
            async with sess.awake("nope"):
                pass

    with pytest.raises(ValueError, match="duplicate ResidencyGroup labels"):
        ExclusiveResidency([ResidencyGroup("dup", ar), ResidencyGroup("dup", FakeEngine("other"))])


async def test_engine_without_residency_surface_is_rejected():
    class NotAnEngine:
        pass

    # sleep() is probed first, so that is the name reported.
    with pytest.raises(TypeError, match="no callable sleep"):
        ResidencyGroup("bad", NotAnEngine())


async def test_stage_ids_are_forwarded():
    """A group may own a stage subset of one engine."""
    seen: list = []

    class Recorder(FakeEngine):
        async def sleep(self, stage_ids=None, level=2, mode="abort"):
            seen.append(("sleep", stage_ids))
            return await super().sleep(stage_ids=stage_ids, level=level, mode=mode)

        async def wake_up(self, stage_ids=None, tags=None):
            seen.append(("wake", stage_ids))
            return await super().wake_up(stage_ids=stage_ids, tags=tags)

    engine = Recorder("multi")
    residency = ExclusiveResidency([ResidencyGroup("s0", engine, stage_ids=(0,))])
    await residency.prepare()
    async with residency.session(request_id="r1") as sess:
        async with sess.awake("s0"):
            pass

    assert seen == [("sleep", [0]), ("wake", [0]), ("sleep", [0])]


@dataclass
class FakeAREngine(FakeEngine):
    """FakeEngine that reproduces AsyncOmni's AR admission gate.

    Modelled on the real semantics, not on what the fix happens to call:
    ``sleep`` on an AR/mixed stage sets ``_hold_admission_until_resume``, and
    ``wake_up`` deliberately does NOT clear ``_paused`` while that flag is set
    (async_omni.py, wake_up docstring — the flow it was built for is the trainer
    order pause -> abort -> sleep -> train -> wake -> resume). Only
    ``resume_generation`` re-admits.

    Without that last call the engine is awake, reports SUCCESS, and then blocks
    every generate() forever on ``_pause_cond``: no error, no GPU work. The plain
    FakeEngine cannot catch it because it has no admission concept at all.
    """

    admitting: bool = True
    hold_admission_until_resume: bool = False
    resume_error: Exception | None = None

    async def sleep(self, stage_ids=None, level=2, mode="abort"):
        self.admitting = False
        self.hold_admission_until_resume = True
        return await super().sleep(stage_ids=stage_ids, level=level, mode=mode)

    async def wake_up(self, stage_ids=None, tags=None):
        acks = await super().wake_up(stage_ids=stage_ids, tags=tags)
        if not self.hold_admission_until_resume:
            self.admitting = True
        return acks

    async def resume_generation(self, stage_ids=None):
        if self.resume_error is not None:
            self.log.append(f"{self.name}:resume_raise")
            raise self.resume_error
        self.admitting = True
        self.hold_admission_until_resume = False
        self.log.append(f"{self.name}:resume")


async def test_waking_an_ar_group_leaves_it_admitting_requests():
    """Regression: wake alone leaves an AR engine awake but permanently paused.

    Upstream #6084 added the AR admission gate for trainers; residency is an
    inference flow with no resume step, so it hung on the first request that ever
    woke AR (task stuck in "processing", all workers idle, GPU 0%).
    """
    ar = FakeAREngine("ar")
    residency = _residency(ar)
    await residency.prepare()
    assert not ar.admitting, "sleep should have closed admission (fixture sanity)"

    async with residency.session(request_id="r1") as sess:
        async with sess.awake("ar"):
            assert ar.awake
            assert ar.admitting, "AR woke but is still refusing requests — generate() would hang forever"

    assert ar.log == ["ar:sleep(level=1)", "ar:wake", "ar:resume", "ar:sleep(level=1)"]


async def test_diffusion_group_without_resume_surface_still_wakes():
    """resume_generation is optional: diffusion-only engines never set the gate."""
    dit = FakeEngine("dit")
    assert not hasattr(dit, "resume_generation")
    residency = _residency(dit)
    await residency.prepare()
    async with residency.session(request_id="r1") as sess:
        async with sess.awake("dit"):
            assert dit.awake
    assert dit.log == ["dit:sleep(level=1)", "dit:wake", "dit:sleep(level=1)"]


async def test_failed_resume_is_reported_and_group_is_swept_back_to_sleep():
    """A warm-but-unusable engine must not keep holding the cards."""
    ar = FakeAREngine("ar", resume_error=RuntimeError("engine core gone"))
    residency = _residency(ar)
    await residency.prepare()

    with pytest.raises(ResidencyError, match="resume_generation"):
        async with residency.session(request_id="r1") as sess:
            async with sess.awake("ar"):
                pytest.fail("body must not run when resume failed")

    assert ar.log[-1] == "ar:sleep(level=1)", "safety sweep must put it back to sleep"
    assert not ar.awake
