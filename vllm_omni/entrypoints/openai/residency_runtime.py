# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construction and lifetime of a co-located, exclusively-resident engine group.

Ties together the declaration (:mod:`residency_config`) and the mutual-exclusion
coordinator (:mod:`stage_residency`): builds one ``AsyncOmni`` per declared
engine, puts them all to sleep, and hands back a bundle the server keeps on
app state.

Kept out of ``api_server`` so the multi-engine path is reviewable on its own and
the single-engine path stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.residency_config import (
    ResidencyDeployment,
    load_residency_config,
)
from vllm_omni.entrypoints.openai.stage_residency import (
    ExclusiveResidency,
    ResidencyGroup,
)

logger = init_logger(__name__)

__all__ = ["ResidencyBundle", "build_residency_bundle"]


@dataclass
class ResidencyBundle:
    """Every engine in a residency group, plus the coordinator gating them."""

    deployment: ResidencyDeployment
    engines: dict[str, AsyncOmni]
    residency: ExclusiveResidency
    _shutdown_done: bool = field(default=False, init=False)

    @property
    def primary(self) -> AsyncOmni:
        """The terminal engine, which the server uses as its ``engine_client``.

        Making the output-producing engine primary keeps every endpoint pointed at
        the engine that actually renders the result. It does NOT make them work by
        itself: every engine is parked asleep at boot, so any route reaching
        ``engine_client.generate()`` directly is refused by AsyncOmni's
        sleeping-stage guard until something wakes it. Routes therefore go through
        ``_terminal_engine_awake`` (api_server), and only the image-task job runs
        the full multi-phase AR-then-diffusion sequence.
        """
        return self.engines[self.deployment.terminal.label]

    def label_of(self, role: str) -> str | None:
        labels = self.deployment.labels_by_role(role)
        return labels[0] if labels else None

    def shutdown(self) -> None:
        """Shut every engine down, even if one raises.

        A partial shutdown would strand worker processes holding GPU memory, so
        failures are logged and the sweep continues rather than propagating out
        of the server's ``finally``.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        for label, engine in self.engines.items():
            try:
                engine.shutdown()
            except Exception:
                logger.exception("Residency: shutdown of engine %r failed", label)


async def build_residency_bundle(*, model: str, base_kwargs: dict[str, Any], path: str) -> ResidencyBundle:
    """Build every declared engine, then park them all asleep.

    Engines are built STRICTLY SEQUENTIALLY. Each engine's stage-registration
    server tracks handed-out ports in its own ``_allocated_ports`` set
    (``engine/stage_engine_startup.py``), and ``get_open_ports_list`` only
    guarantees uniqueness within a single call — so two engines coming up
    concurrently could draw the same ephemeral port and the second to ``bind()``
    would die with "Address already in use". Building one fully before starting
    the next keeps each engine's ports bound before the next draw.

    ``enable_sleep_mode`` is forced on: without it the CuMem allocator never
    takes ownership of the weights, ``sleep()`` frees nothing, and the whole
    scheme silently degrades into "every engine resident" — which does not fit
    on the hardware this exists for. ``stage_residency`` also treats a sleep that
    frees zero bytes as an error, so the failure would surface at boot.
    """
    deployment = load_residency_config(path)

    engines: dict[str, AsyncOmni] = {}
    groups: list[ResidencyGroup] = []
    residency: ExclusiveResidency | None = None
    try:
        for spec in deployment.engines:
            kwargs = dict(base_kwargs)
            kwargs["deploy_config"] = spec.deploy_config
            kwargs["enable_sleep_mode"] = True
            logger.info(
                "Residency: building engine %r (role=%s) from %s",
                spec.label,
                spec.role,
                spec.deploy_config,
            )
            engine = AsyncOmni(model=model, **kwargs)
            engines[spec.label] = engine
            group = ResidencyGroup(label=spec.label, engine=engine)
            groups.append(group)

            # Sleep THIS engine before building the next one. Sleeping only after
            # the whole loop would leave every finished engine resident while the
            # next one loads, and the point of exclusive residency is that they do
            # not fit together: on 4x A100-40G the AR engine alone leaves ~25 GiB
            # per card, and the DiT engine needs ~16 GiB of weights plus loading
            # headroom, so the second build dies at 40417/40960 MiB. Report §11.1
            # states the boot order as "init AR -> sleep, init DiT -> sleep", and
            # §7.2's timeline confirms it: AR sleep (13.15s) precedes DiT init.
            residency = ExclusiveResidency([group], sleep_level=deployment.sleep_level)
            await residency.park_group(group)
    except Exception:
        # Roll back the engines already built; leaving them up would hold GPU
        # memory for a server that is about to fail startup anyway.
        for label, engine in engines.items():
            try:
                engine.shutdown()
            except Exception:
                logger.exception("Residency: rollback shutdown of engine %r failed", label)
        raise

    # The per-engine coordinators above existed only to park each engine as it
    # finished loading. Build the real one over ALL groups now, and tell it they
    # are already parked so it does not re-sleep engines that are already
    # offloaded (which would free zero bytes and be rejected as a failed sleep).
    residency = ExclusiveResidency(groups, sleep_level=deployment.sleep_level)
    residency.assume_parked()
    bundle = ResidencyBundle(deployment=deployment, engines=engines, residency=residency)

    logger.info(
        "Residency ready: engines=%s primary=%s (all asleep)",
        list(engines),
        deployment.terminal.label,
    )
    return bundle
