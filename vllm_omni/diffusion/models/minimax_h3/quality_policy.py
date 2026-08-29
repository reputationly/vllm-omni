# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model-owned request quality policy for MiniMax H3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.cachedit import CacheDiTRequestSpec
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.errors import OmniClientError

logger = init_logger(__name__)

MINIMAX_H3_GENERIC_CACHE_KEY = "minimax_h3.generic"
MINIMAX_H3_HIGH_CACHE_KEY = "minimax_h3.high"

# ``quality="high"`` is accepted but treated as "no cache" on MiniMax H3.
#
# Measured 2026-08-29 on 4x A100-40G, 832x480 / 124 frames, same seed, against
# the uncached result:
#   Turbo8 (8 steps)   36.07s -> 36.58s, outputs byte-identical. The profile's
#                      ``max_warmup_steps=4`` eats half of an 8-step schedule and
#                      ``max_continuous_cached_steps=1`` blocks the rest, so not a
#                      single step was skipped -- the 1.4% is pure hook overhead.
#   Base (20 steps)    65.65s -> 67.79s. Steps *were* skipped and the sample moved,
#                      but with no discernible quality gain (per-file sharpness went
#                      two down / one up -- divergence, not degradation) and it is
#                      3.3% slower. Note PSNR/SSIM between two diffusion samples
#                      measures divergence, not quality loss; do not read those as
#                      degradation.
# So the shipped profile costs time and buys nothing at either step count.
#
# Gating it here rather than at the gateway is deliberate: instance ports are
# reachable directly, so a gateway-side strip leaves the path open. Gating it here
# rather than by removing the protocol field is also deliberate: ``quality`` is a
# public OpenAI-compatible field and H3 is the only model that consumes it today.
#
# Flip this to True to re-enable once the profile has been re-tuned (Fn/Bn/
# residual_diff_threshold/max_warmup_steps are all adjustable); the paths below
# are otherwise unchanged.
MINIMAX_H3_CACHE_DIT_ENABLED = False
MINIMAX_H3_FORCE_REFRESH_STEP_HINT_ARG = "force_refresh_step_hint"
MINIMAX_H3_FORCE_REFRESH_STEP_POLICY_ARG = "force_refresh_step_policy"
MINIMAX_H3_FORCE_REFRESH_POLICIES = ("once", "repeat")


def _high_quality_cache_config() -> DiffusionCacheConfig:
    return DiffusionCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=4,
        residual_diff_threshold=0.04,
        max_continuous_cached_steps=1,
        enable_taylorseer=False,
        scm_steps_mask_policy=None,
    )


def _resolve_force_refresh(
    extra_args: Mapping[str, Any] | None,
    *,
    num_inference_steps: int,
) -> tuple[int, str] | None:
    """Resolve H3's request-scoped Cache-DiT refresh hint.

    These are model-specific ``extra_args`` rather than global sampling fields,
    keeping other diffusion models unchanged.
    """

    if not extra_args:
        return None

    raw_hint = extra_args.get(MINIMAX_H3_FORCE_REFRESH_STEP_HINT_ARG)
    raw_policy = extra_args.get(MINIMAX_H3_FORCE_REFRESH_STEP_POLICY_ARG)

    if raw_hint is None:
        if raw_policy is not None:
            raise OmniClientError("MiniMax H3 force_refresh_step_policy requires force_refresh_step_hint")
        return None
    if isinstance(raw_hint, bool) or not isinstance(raw_hint, Integral):
        raise OmniClientError("MiniMax H3 force_refresh_step_hint must be a positive integer")

    hint = int(raw_hint)
    if not 1 <= hint <= num_inference_steps:
        raise OmniClientError(
            "MiniMax H3 force_refresh_step_hint must be between 1 and "
            f"num_inference_steps ({num_inference_steps}), got {hint}"
        )

    policy = "once" if raw_policy is None else raw_policy
    if not isinstance(policy, str) or policy not in MINIMAX_H3_FORCE_REFRESH_POLICIES:
        raise OmniClientError(
            "MiniMax H3 force_refresh_step_policy must be one of "
            f"{list(MINIMAX_H3_FORCE_REFRESH_POLICIES)}, got {policy!r}"
        )
    return hint, policy


def _has_force_refresh_args(extra_args: Mapping[str, Any] | None) -> bool:
    if not extra_args:
        return False
    return any(
        extra_args.get(name) is not None
        for name in (
            MINIMAX_H3_FORCE_REFRESH_STEP_HINT_ARG,
            MINIMAX_H3_FORCE_REFRESH_STEP_POLICY_ARG,
        )
    )


def _with_force_refresh(
    cache_config: DiffusionCacheConfig,
    force_refresh: tuple[int, str] | None,
) -> DiffusionCacheConfig:
    if force_refresh is None:
        return cache_config
    hint, policy = force_refresh
    return replace(
        cache_config,
        force_refresh_step_hint=hint,
        force_refresh_step_policy=policy,
    )


def _cache_installation_key(base_key: str, force_refresh: tuple[int, str] | None) -> str:
    """Make refresh-policy transitions explicit to the request runtime.

    Cache-DiT cannot clear an existing ``force_refresh_step_hint`` through its
    incremental context update API because ``None`` is treated as "keep the
    old value".  Including the request hint in the installation key therefore
    makes a hint change perform a safe hook reinstall, while repeated requests
    with the same hint still only refresh the context.
    """

    if force_refresh is None:
        return base_key
    hint, policy = force_refresh
    return f"{base_key}:force_refresh={hint}:{policy}"


@dataclass(frozen=True)
class MiniMaxH3QualityPlan:
    """Resolved execution choices for one MiniMax H3 request."""

    cache_dit: CacheDiTRequestSpec | None


class MiniMaxH3QualityPolicy:
    """Resolve H3 request quality into a declarative Cache-DiT target.

    When the server starts with Cache-DiT, omitted quality selects the
    server-configured profile. Otherwise, omitted quality selects no cache.
    In either case, ``lossless`` selects no cache and ``high`` selects H3's
    high-quality profile. The policy therefore owns whether a request installs
    Cache-DiT; the startup backend only controls the omitted-quality default.
    H3-specific Cache-DiT refresh hints are read from request ``extra_args``;
    they do not change the global diffusion request contract.
    The pipeline owns applying the resulting target at the request boundary.
    """

    def __init__(self, od_config: Any) -> None:
        self._od_config = od_config
        self._configured_backend = str(getattr(od_config, "cache_backend", "none") or "none").lower()

    def resolve(
        self,
        *,
        quality: str | None,
        num_inference_steps: int,
        extra_args: Mapping[str, Any] | None = None,
    ) -> MiniMaxH3QualityPlan:
        if quality == "high" and MINIMAX_H3_CACHE_DIT_ENABLED:
            base_key = MINIMAX_H3_HIGH_CACHE_KEY
            base_config = _high_quality_cache_config()
        elif self._configured_backend == "cache_dit" and quality is None:
            base_key = MINIMAX_H3_GENERIC_CACHE_KEY
            base_config = self._od_config.cache_config
        else:
            # Only the request-selected ``high`` profile is gated. A server that
            # deliberately starts with ``cache_backend: cache_dit`` still gets its
            # configured profile above -- that is a deployment choice, not something
            # a caller can flip, and gating it here would silently ignore a config.
            if quality == "high":
                # Log rather than drop silently: an operator needs to see why a
                # requested knob did nothing. Not a 400 either, because ``quality``
                # is documented as advisory ("exact behavior is model-specific").
                logger.warning(
                    "MiniMax H3 ignores quality='high': its Cache-DiT profile measured slower at both "
                    "8 and 20 steps with no quality gain, so it is disabled (see "
                    "MINIMAX_H3_CACHE_DIT_ENABLED). Serving this request without cache."
                )
            if _has_force_refresh_args(extra_args):
                raise OmniClientError("MiniMax H3 force-refresh arguments require an active Cache-DiT request target")
            return MiniMaxH3QualityPlan(cache_dit=None)

        force_refresh = _resolve_force_refresh(
            extra_args,
            num_inference_steps=num_inference_steps,
        )
        return MiniMaxH3QualityPlan(
            cache_dit=CacheDiTRequestSpec(
                installation_key=_cache_installation_key(base_key, force_refresh),
                cache_config=_with_force_refresh(base_config, force_refresh),
                num_inference_steps=num_inference_steps,
            ),
        )


__all__ = [
    "MINIMAX_H3_GENERIC_CACHE_KEY",
    "MINIMAX_H3_HIGH_CACHE_KEY",
    "MiniMaxH3QualityPlan",
    "MiniMaxH3QualityPolicy",
]
