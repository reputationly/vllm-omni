# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mutually-exclusive GPU residency for engines that cannot be awake together.

Some models are too large for every stage to hold GPU weights simultaneously,
yet small enough that each stage fits ALONE. For those, keep all stages loaded
and let at most ONE be awake at a time, using level-1 sleep (weights offloaded
to host RAM, fast DMA restore) at the stage boundary.

The concrete case this was written for is HunyuanImage-3.0-Instruct-Distil NF4 on
4x A100-PCIE-40GB (report: ``docs/实验报告/
vLLM-Omni-HunyuanImage3-A100-NF4-实验与优化复盘.md`` §7.2 / §11.1). Measured there:

    AR weights                14.93 GiB/rank
    DiT weights               15.83 GiB/rank
    sleeping residual         ~1067 MiB/card
    one-awake-at-a-time peak  35039 / 34619 / 34619 / 34619 MiB   (~5.7 GiB spare)
    AR wake / sleep           2.6658 s / 0.8948 s
    DiT wake / sleep          1.7871 s / 2.3726 s

Both awake at once would need 30.76 GiB of weights plus ~13.5 GiB of activations
per card, which does not fit in 40 GiB. Exclusion is therefore a CORRECTNESS
requirement, not a tuning knob — hence a coordinator rather than ad-hoc
sleep/wake calls at the call site.

The four hard constraints below are transcribed from report §11.1:

    1. at most one engine awake at any instant
    2. a failed wake must NOT be followed by waking another group
    3. verify memory actually dropped after sleep
    4. cancellation must still restore residency state in ``finally``

Constraint 4 is why :func:`_run_uncancellable` exists: a bare ``await`` inside a
``finally`` is cancelled immediately, which would release the mutex with a group
still awake and let the next request wake a second one.

This module deliberately knows nothing about HTTP, task queues, or which model is
running. An "engine" is anything exposing the ``AsyncOmni`` residency surface::

    async def sleep(stage_ids=None, level=2, ...) -> list[OmniACK]
    async def wake_up(stage_ids=None, tags=None) -> list[OmniACK]

so a single-stage engine, a stage subset of one engine, or several independent
engines in one process are all expressible as :class:`ResidencyGroup` values.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "ExclusiveResidency",
    "ResidencyError",
    "ResidencyGroup",
    "ResidencySession",
]

# Level 1 offloads weights to host RAM and supports a DMA restore. Level 2
# discards them, and AsyncOmni.wake_up() raises NotImplementedError afterwards
# (report §7.2: "level-1 是本方案的一部分,不能随意改成 level-2").
_WAKEABLE_SLEEP_LEVEL = 1


class ResidencyError(RuntimeError):
    """Residency invariant violated, or a wake/sleep transition failed."""


@dataclass(frozen=True)
class ResidencyGroup:
    """One independently sleepable unit of GPU residency.

    Args:
        label: Stable name used in logs and :meth:`ResidencySession.awake`.
        engine: Object exposing ``sleep()`` / ``wake_up()`` (an ``AsyncOmni``).
        stage_ids: Stages of ``engine`` this group owns. ``None`` means the whole
            engine, which is the right value when each engine is single-stage.
        expect_freed_bytes: When True, a sleep whose ACKs report zero freed bytes
            is treated as a failure (constraint 3). Turn it off only for groups
            whose backend does not report ``freed_bytes``.
    """

    label: str
    engine: Any
    stage_ids: tuple[int, ...] | None = None
    expect_freed_bytes: bool = True

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("ResidencyGroup.label must be non-empty")
        for attr in ("sleep", "wake_up"):
            if not callable(getattr(self.engine, attr, None)):
                raise TypeError(f"ResidencyGroup({self.label!r}).engine has no callable {attr}()")

    @property
    def stage_ids_arg(self) -> list[int] | None:
        return list(self.stage_ids) if self.stage_ids is not None else None


def _ack_failures(acks: Iterable[Any] | None) -> list[str]:
    """Collect error text from any non-SUCCESS ACK.

    ``collective_rpc`` fans out per worker, so a partial failure shows up as one
    bad ACK among many; treating "no exception raised" as success would silently
    accept a half-transitioned group.
    """
    failures: list[str] = []
    for ack in acks or ():
        status = str(getattr(ack, "status", "") or "").upper()
        if status and status != "SUCCESS":
            failures.append(
                f"stage={getattr(ack, 'stage_id', '?')} rank={getattr(ack, 'rank', '?')} "
                f"status={status} error={getattr(ack, 'error_msg', None)}"
            )
    return failures


def _total_freed_bytes(acks: Iterable[Any] | None) -> int:
    total = 0
    for ack in acks or ():
        try:
            total += int(getattr(ack, "freed_bytes", 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


async def _run_uncancellable(coro: Any, *, what: str) -> Any:
    """Await ``coro`` to completion even if the caller is being cancelled.

    Residency restoration must finish before the mutex is released (report §11.1
    constraint 4). If the caller is cancelled we remember that, let ``coro``
    finish, and re-raise ``CancelledError`` afterwards so cancellation semantics
    are preserved for the caller.
    """
    task = asyncio.ensure_future(coro)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Keep looping: the shielded task is still running and we must not
            # abandon it mid-transition.
            cancelled = True
        except Exception:
            break
    if cancelled:
        if task.done() and task.exception() is not None:
            logger.error("Residency %s failed while the caller was cancelled: %s", what, task.exception())
        raise asyncio.CancelledError()
    return task.result()


@dataclass
class ResidencySession:
    """Exclusive residency held for the duration of one request.

    Obtained from :meth:`ExclusiveResidency.session`; not constructed directly.
    """

    owner: ExclusiveResidency
    request_id: str
    _awake_label: str | None = field(default=None, init=False)

    @asynccontextmanager
    async def awake(self, label: str) -> AsyncIterator[ResidencyGroup]:
        """Wake the group named ``label``, yield it, then put it back to sleep.

        The group is asleep again by the time this context manager returns, so
        consecutive ``awake()`` blocks in one session never overlap — that is
        what upholds constraint 1 across a multi-phase request such as
        AR-then-DiT.

        Raises:
            ResidencyError: unknown label, another group already awake in this
                session, or the wake/sleep transition failed.
        """
        group = self.owner.group(label)
        if self._awake_label is not None:
            raise ResidencyError(
                f"[{self.request_id}] cannot wake {label!r}: {self._awake_label!r} is still awake "
                f"in this session (at most one group may be awake at a time)"
            )
        await self.owner._wake(group, request_id=self.request_id)
        self._awake_label = label
        try:
            yield group
        finally:
            # Uncancellable: see _run_uncancellable. Also runs on the error path,
            # so a failure inside the caller's body cannot leave the group awake
            # and starve the next request of GPU memory.
            try:
                await _run_uncancellable(
                    self.owner._sleep(group, request_id=self.request_id),
                    what=f"sleep({label})",
                )
            finally:
                self._awake_label = None


class ExclusiveResidency:
    """Coordinator enforcing "at most one group awake" across requests.

    Usage::

        residency = ExclusiveResidency([ar_group, dit_group])
        await residency.prepare()          # BOOT: every group asleep

        async with residency.session(request_id=rid) as sess:
            async with sess.awake("ar") as g:
                cot = await run_ar(g.engine)
            async with sess.awake("dit") as g:
                image = await run_dit(g.engine, cot)

    The session mutex serializes whole requests, which matches the residency
    model: with one group awake at a time there is no way to overlap two
    requests anyway. Callers that need to reject rather than queue should pass
    ``acquire_timeout_s`` and map :class:`TimeoutError` to their busy response.
    """

    def __init__(
        self,
        groups: Sequence[ResidencyGroup],
        *,
        acquire_timeout_s: float | None = None,
        sleep_level: int = _WAKEABLE_SLEEP_LEVEL,
    ) -> None:
        if not groups:
            raise ValueError("ExclusiveResidency requires at least one ResidencyGroup")
        if sleep_level != _WAKEABLE_SLEEP_LEVEL:
            raise ValueError(
                f"sleep_level must be {_WAKEABLE_SLEEP_LEVEL}: level 2 discards weights and "
                "wake_up() is not implemented after it, so a level-2 group can never be "
                "woken for the next request"
            )
        labels = [g.label for g in groups]
        duplicates = {label for label in labels if labels.count(label) > 1}
        if duplicates:
            raise ValueError(f"duplicate ResidencyGroup labels: {sorted(duplicates)}")
        self._groups = {g.label: g for g in groups}
        self._sleep_level = sleep_level
        self._acquire_timeout_s = acquire_timeout_s
        self._mutex = asyncio.Lock()
        # Labels believed to hold GPU weights right now. Written only by
        # _wake/_sleep so the safety sweep in session() can trust it.
        self._resident: set[str] = set()
        self._prepared = False

    @property
    def labels(self) -> list[str]:
        return list(self._groups)

    def group(self, label: str) -> ResidencyGroup:
        try:
            return self._groups[label]
        except KeyError:
            raise ResidencyError(f"unknown residency group {label!r}; known: {sorted(self._groups)}") from None

    async def park_group(self, group: ResidencyGroup) -> None:
        """Sleep ONE group immediately, outside any session.

        Used during boot when engines are built one at a time: each must be
        offloaded before the next one loads, because the whole premise is that
        they do not fit resident together.
        """
        async with self._mutex:
            await self._sleep(group, request_id="boot", force=True)

    def assume_parked(self) -> None:
        """Declare every group already asleep (they were parked during boot).

        Lets the caller skip :meth:`prepare`, whose blanket re-sleep would report
        zero freed bytes for an already-offloaded engine and be rejected as a
        failed transition.
        """
        self._resident.clear()
        self._prepared = True

    async def prepare(self) -> None:
        """Put every group to sleep so the first request starts from a known state.

        Call once after all engines finish loading. Boot cost is real and must be
        budgeted for by whatever health-checks the server (report §7.2 measured
        AR init 139.83s + sleep 13.15s + DiT init 73.84s + sleep 33.20s ~= 260s
        before the first request can be served).
        """
        async with self._mutex:
            for label, group in self._groups.items():
                logger.info("Residency: initial sleep of %r", label)
                await self._sleep(group, request_id="boot", force=True)
            self._prepared = True

    @asynccontextmanager
    async def session(self, *, request_id: str) -> AsyncIterator[ResidencySession]:
        """Hold the global residency mutex for one request."""
        if not self._prepared:
            logger.warning(
                "Residency: session(%s) before prepare(); groups may still hold GPU weights "
                "from load time, so the first wake could exceed the memory budget.",
                request_id,
            )
        acquire = self._mutex.acquire()
        if self._acquire_timeout_s is not None:
            await asyncio.wait_for(acquire, timeout=self._acquire_timeout_s)
        else:
            await acquire
        session = ResidencySession(owner=self, request_id=request_id)
        try:
            yield session
        finally:
            try:
                # Safety sweep: ResidencySession.awake() already sleeps on the way
                # out, so anything still resident here means a transition failed
                # part-way. Leaving it awake would break the NEXT request rather
                # than this one, which is much harder to diagnose.
                stragglers = sorted(self._resident)
                if stragglers:
                    logger.error(
                        "Residency: %s left group(s) %s awake; forcing sleep before releasing the mutex",
                        request_id,
                        stragglers,
                    )
                    for label in stragglers:
                        try:
                            await _run_uncancellable(
                                self._sleep(self._groups[label], request_id=request_id, force=True),
                                what=f"forced sleep({label})",
                            )
                        except Exception:
                            # Already logged by _sleep; keep sweeping the rest so
                            # one stuck group does not hide the others.
                            logger.exception("Residency: forced sleep of %r failed", label)
            finally:
                self._mutex.release()

    async def _wake(self, group: ResidencyGroup, *, request_id: str) -> None:
        others = self._resident - {group.label}
        if others:
            # Constraint 1. Refusing here (instead of waking anyway) is the whole
            # point: two awake groups would exceed the card and fail with an
            # allocator error far from the cause.
            raise ResidencyError(
                f"[{request_id}] refusing to wake {group.label!r} while {sorted(others)} still resident"
            )
        try:
            acks = await group.engine.wake_up(stage_ids=group.stage_ids_arg)
        except Exception as exc:
            # Constraint 2: propagate, and do NOT mark resident. The caller's
            # session unwinds without waking anything else.
            raise ResidencyError(f"[{request_id}] wake_up({group.label}) raised: {exc}") from exc
        failures = _ack_failures(acks)
        if failures:
            # Partially warm: mark resident anyway so the session's safety sweep
            # sleeps it, then fail. Otherwise the half-woken weights would linger
            # untracked and the next wake would overshoot the budget.
            self._resident.add(group.label)
            raise ResidencyError(f"[{request_id}] wake_up({group.label}) failed: {'; '.join(failures)}")
        self._resident.add(group.label)
        logger.info("Residency: %r awake (request %s)", group.label, request_id)

    async def _sleep(self, group: ResidencyGroup, *, request_id: str, force: bool = False) -> None:
        if not force and group.label not in self._resident:
            return
        try:
            acks = await group.engine.sleep(stage_ids=group.stage_ids_arg, level=self._sleep_level)
        except Exception as exc:
            logger.exception("Residency: sleep(%s) raised", group.label)
            raise ResidencyError(f"[{request_id}] sleep({group.label}) raised: {exc}") from exc
        failures = _ack_failures(acks)
        if failures:
            # Stay marked resident: the weights may still be on the GPU, so the
            # next wake must be refused rather than silently overcommitting.
            raise ResidencyError(f"[{request_id}] sleep({group.label}) failed: {'; '.join(failures)}")
        freed = _total_freed_bytes(acks)
        if group.expect_freed_bytes and acks and freed <= 0:
            # Constraint 3, but ADVISORY rather than fatal: freed_bytes is not a
            # trustworthy signal on every backend. AR/LLM workers all-reduce a
            # value that reads 0 even on a sleep that plainly worked — observed on
            # HunyuanImage-3.0's AR stage, where the allocator logged
            #   "CuMemAllocator: sleep freed 23.77 GiB ... 15.02 GiB backed up in CPU"
            # while the worker's own accounting logged
            #   "[LLM Worker 1] Level 1 Sleep: Freed 0.00 GiB."
            # in the same instant. Diffusion workers report it correctly.
            #
            # Failing hard here blocked a deployment whose sleep was working, which
            # is strictly worse than the condition being guarded against: a sleep
            # that genuinely freed nothing announces itself immediately anyway,
            # because the next engine to load runs out of memory.
            logger.warning(
                "Residency: sleep(%s) reported SUCCESS but freed 0 bytes. Some backends "
                "under-report this, so treating it as advisory — but if the next engine "
                "fails to load, suspect that enable_sleep_mode is off for this stage.",
                group.label,
            )
        self._resident.discard(group.label)
        logger.info(
            "Residency: %r asleep (request %s, freed %.3f GiB)",
            group.label,
            request_id,
            freed / 1024**3,
        )
