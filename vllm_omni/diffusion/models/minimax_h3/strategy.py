# SPDX-License-Identifier: Apache-2.0
"""Which MiniMax-H3 contract an instance serves, resolved once at startup.

Two orthogonal dimensions, deliberately kept apart:

``inference_contract``
    What the *model* does with what it was given — RNG, reference order, default
    frame count, geometry, normalization, truncation. ``legacy`` is what
    vLLM-Omni has always done; ``official_diffusers_v1`` is the pinned Diffusers
    ``MiniMaxH3ModularPipeline``.

``admission_policy``
    What the *deployment* is willing to accept — upload size, container and
    codec whitelists, resource budgets. Production may refuse media the official
    in-memory interface would happily take, but refusing it is an admission
    decision and must never change the model semantics of media already
    accepted.

Conflating the two is how "official-compatible" claims go wrong in both
directions: a narrower upload gate gets mistaken for a model contract, and a
model-contract change gets shipped as if it were a validation tweak.

The choice is startup-level. A request may not pick, because switching the
inference contract changes the output of every seed — including ``t2va``, which
conditions on nothing. See ``request_noise.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .reference_image_geometry import (
    MINIMAX_H3_REFERENCE_IMAGE_DEFAULT_TARGET,
    MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV,
    MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV,
    MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV,
    parse_reference_image_max_pixels,
    parse_reference_image_no_upscale,
    parse_reference_image_short_edge,
)
from .request_noise import MINIMAX_H3_RNG_LEGACY, MINIMAX_H3_RNG_MODES, MINIMAX_H3_RNG_OFFICIAL_V1
from .time_request import minimax_h3_align_frame_count

MINIMAX_H3_CONTRACT_LEGACY = "legacy"
MINIMAX_H3_CONTRACT_OFFICIAL_V1 = "official_diffusers_v1"
MINIMAX_H3_CONTRACTS = frozenset({MINIMAX_H3_CONTRACT_LEGACY, MINIMAX_H3_CONTRACT_OFFICIAL_V1})

MINIMAX_H3_ADMISSION_PRODUCTION = "production_safe_v1"
MINIMAX_H3_ADMISSION_PARITY = "parity_fixture_v1"
MINIMAX_H3_ADMISSIONS = frozenset({MINIMAX_H3_ADMISSION_PRODUCTION, MINIMAX_H3_ADMISSION_PARITY})

# Startup-level selection for deployments that cannot pass the fields through
# their config yet. A deploy-config field takes precedence when present.
MINIMAX_H3_CONTRACT_ENV = "VLLM_OMNI_H3_INFERENCE_CONTRACT"
MINIMAX_H3_ADMISSION_ENV = "VLLM_OMNI_H3_ADMISSION_POLICY"

# The reference-image geometry, selectable on its own. Two reasons it is not
# simply folded into the contract:
#
# 1. As an experiment, `official` changes the RNG stream *and* the reference
#    geometry at once, so an observed quality difference cannot be attributed to
#    either. Isolating the geometry needs a legacy-RNG + fixed-geometry run.
# 2. As a product, the canvas pre-stretch is a confirmed visible defect — a
#    reference whose aspect ratio differs from the canvas reaches the model
#    squashed. Fixing only that is a much smaller change than adopting the whole
#    official contract, which alters every seed's output.
MINIMAX_H3_GEOMETRY_ENV = "VLLM_OMNI_H3_REF_IMAGE_GEOMETRY"
MINIMAX_H3_GEOMETRY_MODES = frozenset({"official_short_edge", "legacy_canvas_prestretch"})

# The two noise axes, selectable on their own, for the same reason the geometry
# is: `official` moves both at once, so a quality difference between contracts
# cannot be attributed to either.
#
# This stopped being hypothetical on 2026-08-17. With the reference-image
# geometry isolated, hair length tracked the *contract* axis exactly across five
# arms — all three legacy arms reproduced the reference's short hair, both
# official arms invented long hair — and did not track geometry at all. Since
# that request carried one image and no audio or video reference, only these two
# axes could be responsible. Separating them needs one run with the official RNG
# and the legacy condition-noise shape, and one with the reverse.
MINIMAX_H3_RNG_ENV = "VLLM_OMNI_H3_RNG_MODE"
MINIMAX_H3_CONDITION_NOISE_ENV = "VLLM_OMNI_H3_CONDITION_NOISE_SHAPE"
MINIMAX_H3_CONDITION_NOISE_MODES = frozenset({"condition_shape", "legacy_oversized_slice"})

# The official workflow default, which every task shares. vLLM currently
# defaults ref2va to the same 124 but t2va/fl2va to 209.
MINIMAX_H3_OFFICIAL_DEFAULT_NUM_FRAMES = 124
MINIMAX_H3_LEGACY_DEFAULT_NUM_FRAMES = {"t2va": 209, "fl2va": 209, "ref2va": 124}

# WHEN the duration ceiling is applied, relative to the `17 * n + 5` alignment
# every request goes through. Not a formula difference — both modes align the
# same way — but an ordering one, which is exactly the class of bug that reading
# only the formula misses.
#
#   requested_frames  check what the caller asked for, then align. What vLLM
#                     has always done, and what production runs.
#   aligned_frames    check what actually gets generated. Official
#                     (`before_denoise.py`, `before_encoder.py`): "the duration
#                     the request generates is the one of the *aligned* frame
#                     count, so that is what the ceiling has to hold for".
MINIMAX_H3_DURATION_REQUESTED = "requested_frames"
MINIMAX_H3_DURATION_ALIGNED = "aligned_frames"
MINIMAX_H3_DURATION_MODES = frozenset({MINIMAX_H3_DURATION_REQUESTED, MINIMAX_H3_DURATION_ALIGNED})


@dataclass(frozen=True)
class MiniMaxH3InferenceStrategy:
    """The resolved inference contract of one instance.

    Every field is a decided value, never a knob to re-read per request: a
    request that could still consult an environment variable is a request whose
    contract is unknowable from the logs.
    """

    name: str

    # Request RNG: draw order, and whether a visual condition is drawn at its
    # own shape or at the oversized legacy shape that is then sliced.
    rng_mode: str
    visual_condition_noise_shape_mode: str  # "condition_shape" | "legacy_oversized_slice"

    # fl2va keyframes: the follower is cover-cropped officially, stretched today.
    fl2va_keyframe_resize_mode: str  # "official_cover_crop" | "legacy_stretch"

    # ref2va reference images: official encodes at a short edge of their own and
    # never binds the generated canvas. See P-C1 in the problem log — today the
    # serving layer pre-stretches them onto the canvas.
    reference_image_geometry_mode: str  # "official_short_edge" | "legacy_canvas_prestretch"

    # Whether an ordered heterogeneous reference list is carried end to end, or
    # rebuilt from modality buckets in a fixed image/video/audio order.
    reference_order_mode: str  # "ordered_references" | "legacy_bucket_canonicalization"

    # ref2va video: official decodes RGB24 from the source container; today the
    # path goes through a lossy H.264 intermediate.
    reference_video_decode_mode: str  # "official_lossless_frames" | "legacy_h264_intermediate"
    reference_video_target_truncation: bool  # official truncates to the target num_frames

    # ref2va video: what reaches the VAE when a reference is *shorter* than the
    # generated clip and therefore does not carry a `17 * n + 5` frame count.
    # Official snaps down to the largest valid count; legacy hands every decoded
    # frame over and lets the VAE pad. Separate from the truncation above, which
    # only ever fires on a reference that is *longer*.
    reference_video_vae_frame_snap_mode: str  # "official_vae_chunk" | "legacy_no_snap"

    # ref2va audio: official truncates at the native rate and resamples once.
    reference_audio_resample_mode: str  # "official_single_resample" | "legacy_double_resample"
    reference_audio_target_truncation: bool  # official truncates to num_frames / fps

    default_num_frames_by_task: dict[str, int]

    # Model-level input semantics. The oracle generates 5..15 s and accepts
    # reference ratios from 1:4 to 4:1; vLLM's legacy entry widened the first
    # and narrowed the second. File size, container and codec limits are NOT
    # here — those are admission, and stay whatever the policy says.
    output_duration_seconds: tuple[float, float]
    duration_validation_mode: str
    reference_image_aspect_ratio_range: tuple[float, float]

    # The short edge a reference image is encoded at. 2048 is the released
    # checkpoint's, and under a production admission policy the official
    # contract refuses to let anything move it. A `parity_fixture_v1` instance
    # may, because an oracle that cannot run is not a stricter test than one
    # that runs at a smaller geometry — but the value is recorded, so a dump
    # taken at another short edge can never be read as the released one.
    reference_image_short_edge: int = MINIMAX_H3_REFERENCE_IMAGE_DEFAULT_TARGET

    # Legacy-compatible resource controls. They are resolved once alongside
    # the rest of the strategy rather than re-read from ``os.environ`` for each
    # request: a stage-scoped env reaches this process as config, not as a
    # process mutation. The official contract refuses both below.
    reference_image_no_upscale: bool = False
    reference_image_max_pixels: int = 0

    admission_policy: str = MINIMAX_H3_ADMISSION_PRODUCTION

    def __post_init__(self) -> None:
        if self.name not in MINIMAX_H3_CONTRACTS:
            raise ValueError(f"inference_contract must be one of {sorted(MINIMAX_H3_CONTRACTS)}, got {self.name!r}")
        if self.rng_mode not in MINIMAX_H3_RNG_MODES:
            raise ValueError(f"rng_mode must be one of {sorted(MINIMAX_H3_RNG_MODES)}, got {self.rng_mode!r}")
        if self.admission_policy not in MINIMAX_H3_ADMISSIONS:
            raise ValueError(
                f"admission_policy must be one of {sorted(MINIMAX_H3_ADMISSIONS)}, got {self.admission_policy!r}"
            )
        if self.duration_validation_mode not in MINIMAX_H3_DURATION_MODES:
            raise ValueError(
                f"duration_validation_mode must be one of {sorted(MINIMAX_H3_DURATION_MODES)}, "
                f"got {self.duration_validation_mode!r}"
            )

    @property
    def is_official(self) -> bool:
        return self.name == MINIMAX_H3_CONTRACT_OFFICIAL_V1

    @property
    def model_validation_semantics(self) -> str:
        """Which validation envelope the *model* applies: ``official`` or ``legacy``.

        Derived rather than stored. As a settable field it was never read
        anywhere — the envelope is actually carried by
        ``output_duration_seconds`` and ``reference_image_aspect_ratio_range``,
        which are — so a builder could have set it to disagree with them and
        nothing would have noticed. Upload gates live in the admission policy,
        not here.
        """
        return "official" if self.is_official else "legacy"

    def default_num_frames(self, task: str) -> int:
        return self.default_num_frames_by_task[task]

    def requested_frame_window(self, fps: int) -> tuple[int, int]:
        """Inclusive ``num_frames`` window this contract admits, before alignment.

        One place decides, because the bug this replaces was two places deciding:
        the range check ran on the requested count and the alignment ran after
        it, so the number that was validated was never the number that was
        generated. Returning a *frame* window rather than a seconds window is
        what makes that impossible to reintroduce — alignment is a fact about
        frames, and a seconds bound cannot express it.

        ``aligned_frames`` widens the window to the alignment lattice at both
        ends, and the two ends are not symmetric:

        * Low end — every count in ``(124 - 17, 124]`` aligns to 124, so 108
          frames generate 5.167 s and are admissible. Rejecting them was strictly
          *stricter* than official for no reason: what runs is a legal clip.
        * High end — 360 frames (``duration=15``) align to 362, i.e. 15.083 s,
          which official rejects. We keep accepting it. That is a deliberate
          superset of one alignment step, decided on 2026-08-18: the ceiling is
          a product limit, ``duration=15`` is the most common request there is,
          and generating 0.083 s more than asked is not a contract violation any
          caller can observe as one. Form parity is about what the model does
          with the input, not about refusing inputs the model handles fine.
        """
        min_seconds, max_seconds = self.output_duration_seconds
        low = int(math.ceil(min_seconds * fps))
        high = int(max_seconds * fps)
        if self.duration_validation_mode == MINIMAX_H3_DURATION_ALIGNED:
            # 17 apart on the lattice, so anything above `aligned - 17` lands on
            # `aligned`; `+ 1` makes the window inclusive at its own low end.
            low = minimax_h3_align_frame_count(low) - 16
            high = minimax_h3_align_frame_count(high)
        return low, high

    def describe(self) -> dict[str, Any]:
        """The resolved contract, for the startup log and result metadata.

        Emitted rather than inferred so an operator can read which contract an
        instance served off the run itself, which is what makes a parity result
        auditable after the fact.
        """
        return {
            "inference_contract": self.name,
            "admission_policy": self.admission_policy,
            "rng_mode": self.rng_mode,
            "visual_condition_noise_shape_mode": self.visual_condition_noise_shape_mode,
            "fl2va_keyframe_resize_mode": self.fl2va_keyframe_resize_mode,
            "reference_image_geometry_mode": self.reference_image_geometry_mode,
            "reference_order_mode": self.reference_order_mode,
            "reference_video_decode_mode": self.reference_video_decode_mode,
            "reference_video_target_truncation": self.reference_video_target_truncation,
            "reference_video_vae_frame_snap_mode": self.reference_video_vae_frame_snap_mode,
            "reference_audio_resample_mode": self.reference_audio_resample_mode,
            "reference_audio_target_truncation": self.reference_audio_target_truncation,
            "default_num_frames_by_task": dict(self.default_num_frames_by_task),
            "output_duration_seconds": list(self.output_duration_seconds),
            "duration_validation_mode": self.duration_validation_mode,
            "reference_image_aspect_ratio_range": list(self.reference_image_aspect_ratio_range),
            "reference_image_short_edge": self.reference_image_short_edge,
            "reference_image_no_upscale": self.reference_image_no_upscale,
            "reference_image_max_pixels": self.reference_image_max_pixels,
            "model_validation_semantics": self.model_validation_semantics,
        }


def legacy_strategy(admission_policy: str = MINIMAX_H3_ADMISSION_PRODUCTION) -> MiniMaxH3InferenceStrategy:
    """Exactly what production runs today. Changing any value here is a regression."""
    return MiniMaxH3InferenceStrategy(
        name=MINIMAX_H3_CONTRACT_LEGACY,
        rng_mode=MINIMAX_H3_RNG_LEGACY,
        visual_condition_noise_shape_mode="legacy_oversized_slice",
        fl2va_keyframe_resize_mode="legacy_stretch",
        reference_image_geometry_mode="legacy_canvas_prestretch",
        reference_order_mode="legacy_bucket_canonicalization",
        reference_video_decode_mode="legacy_h264_intermediate",
        reference_video_target_truncation=False,
        reference_video_vae_frame_snap_mode="legacy_no_snap",
        reference_audio_resample_mode="legacy_double_resample",
        reference_audio_target_truncation=False,
        default_num_frames_by_task=dict(MINIMAX_H3_LEGACY_DEFAULT_NUM_FRAMES),
        output_duration_seconds=(2.0, 16.0),
        duration_validation_mode=MINIMAX_H3_DURATION_REQUESTED,
        reference_image_aspect_ratio_range=(0.4, 2.5),
        admission_policy=admission_policy,
    )


def official_diffusers_v1_strategy(
    admission_policy: str = MINIMAX_H3_ADMISSION_PRODUCTION,
) -> MiniMaxH3InferenceStrategy:
    """The pinned Diffusers ``MiniMaxH3ModularPipeline`` contract."""
    return MiniMaxH3InferenceStrategy(
        name=MINIMAX_H3_CONTRACT_OFFICIAL_V1,
        rng_mode=MINIMAX_H3_RNG_OFFICIAL_V1,
        visual_condition_noise_shape_mode="condition_shape",
        fl2va_keyframe_resize_mode="official_cover_crop",
        reference_image_geometry_mode="official_short_edge",
        reference_order_mode="ordered_references",
        reference_video_decode_mode="official_lossless_frames",
        reference_video_target_truncation=True,
        reference_video_vae_frame_snap_mode="official_vae_chunk",
        reference_audio_resample_mode="official_single_resample",
        reference_audio_target_truncation=True,
        default_num_frames_by_task=dict.fromkeys(("t2va", "fl2va", "ref2va"), MINIMAX_H3_OFFICIAL_DEFAULT_NUM_FRAMES),
        output_duration_seconds=(5.0, 15.0),
        duration_validation_mode=MINIMAX_H3_DURATION_ALIGNED,
        reference_image_aspect_ratio_range=(0.25, 4.0),
        admission_policy=admission_policy,
    )


_BUILDERS = {
    MINIMAX_H3_CONTRACT_LEGACY: legacy_strategy,
    MINIMAX_H3_CONTRACT_OFFICIAL_V1: official_diffusers_v1_strategy,
}

# The reference-image resource switches predate this strategy and stay usable
# under legacy, but they describe a geometry the official contract fixes, so an
# instance may not claim official while any of them is set. Startup fails loudly
# rather than serving a result that is neither contract.
_CONFLICTING_REFERENCE_IMAGE_ENVS = (
    MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV,
    MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV,
    MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV,
)


def _parse_short_edge(raw: str) -> int:
    """The reference-image short edge, resolved the same way for every contract.

    Delegated rather than reimplemented. This function briefly had rules of its
    own — raise on a non-integer, raise on anything that is not a multiple of
    32, and no upper bound at all — which broke legacy deployments carrying a
    perfectly usable value like ``1000`` and, worse, silently *admitted*
    ``10240``: a target the pipeline's own parser has always refused because it
    packs ~25x the rows of the default. See reference_image_geometry.py.
    """
    return parse_reference_image_short_edge(raw)


def contract_environ(od_config: Any = None) -> Any:
    """The environment ``resolve_strategy`` must read for a given config.

    Process environment, with the diffusion stage's own ``runtime.env`` laid on
    top — the precedence ``stage_runtime_env`` applies when it sets them.

    The overlay exists because a stage-scoped variable is applied while that
    stage starts and restored immediately afterwards. The stage process keeps
    it; nobody else ever had it. So the serving layer, which resolves the same
    contract to answer capability questions before a request reaches the worker,
    reads a process environment where the operator's selection simply is not
    present, and answers ``legacy`` for a worker running ``official`` — the same
    accident as a config field that never learned to cross a hop, arriving by a
    different road. Config *fields* cross by declaration; this mapping is how
    the knobs that exist only as environment variables cross.

    Any ``Mapping`` will do, deliberately. A YAML deployment's stage config is
    an OmegaConf ``DictConfig``, so the mapping arrives here wrapped — and a
    ``DictConfig`` is a ``MutableMapping`` but *not* a ``dict``. Testing for
    ``dict`` therefore dropped the overlay on exactly the deployment shape the
    overlay exists for, which is worse than not carrying it at all: the value
    made it across all four hops and was discarded by its reader, silently, so
    the serving layer answered ``legacy`` for a worker running ``official``
    while every test — all of which hand-build plain dicts — passed.

    Args:
        od_config: The diffusion config, when the caller has one.

    Returns:
        A mapping to hand ``resolve_strategy`` as ``environ``.
    """
    import os
    from collections.abc import Mapping

    stage_environ = getattr(od_config, "diffusion_runtime_environ", None)
    if isinstance(stage_environ, Mapping) and stage_environ:
        return {**os.environ, **{str(key): str(value) for key, value in stage_environ.items()}}
    return os.environ


def resolve_strategy(
    *,
    inference_contract: str | None,
    admission_policy: str | None,
    environ: Any = None,
) -> MiniMaxH3InferenceStrategy:
    """Resolve the startup contract, or fail with a reason.

    Args:
        inference_contract: ``legacy`` (default) or ``official_diffusers_v1``.
        admission_policy: ``production_safe_v1`` (default) or ``parity_fixture_v1``.
        environ: Environment mapping, for testing. Defaults to ``os.environ``.

    Raises:
        ValueError: On an unknown value, or on an official contract that would
            silently be reshaped by a reference-image resource switch.
    """
    import os

    environ = os.environ if environ is None else environ
    # Startup-level env fallbacks, for deployments whose config surface does not
    # carry the fields yet. Read once, here, and never again — the point of the
    # resolved strategy is that a request cannot consult the environment.
    name = (inference_contract or environ.get(MINIMAX_H3_CONTRACT_ENV) or MINIMAX_H3_CONTRACT_LEGACY).strip()
    policy = (admission_policy or environ.get(MINIMAX_H3_ADMISSION_ENV) or MINIMAX_H3_ADMISSION_PRODUCTION).strip()
    if name not in _BUILDERS:
        raise ValueError(f"MiniMax-H3 inference_contract must be one of {sorted(MINIMAX_H3_CONTRACTS)}, got {name!r}")

    strategy = _BUILDERS[name](admission_policy=policy)

    geometry = str(environ.get(MINIMAX_H3_GEOMETRY_ENV, "")).strip()
    if geometry:
        if geometry not in MINIMAX_H3_GEOMETRY_MODES:
            raise ValueError(
                f"{MINIMAX_H3_GEOMETRY_ENV} must be one of {sorted(MINIMAX_H3_GEOMETRY_MODES)}, got {geometry!r}"
            )
        strategy = replace(strategy, reference_image_geometry_mode=geometry)
        # The admission envelope describes what the *validator* sees, and that
        # depends on the geometry mode rather than on the contract name. Legacy's
        # [0.4, 2.5] only ever held because the canvas pre-stretch rewrote every
        # reference to the output ratio before validation: a 1664x656 portrait
        # (2.54) reached the validator as 1344x768 (1.75) and passed. Drop the
        # pre-stretch and the raw ratio arrives instead, so the legacy envelope
        # would hard-reject images that production accepts today. The envelope
        # gates admission only — widening it cannot change any result that was
        # already admitted — so it follows the geometry it is validating.
        if geometry == "official_short_edge":
            strategy = replace(
                strategy,
                reference_image_aspect_ratio_range=(
                    official_diffusers_v1_strategy().reference_image_aspect_ratio_range
                ),
            )

    rng_mode = str(environ.get(MINIMAX_H3_RNG_ENV, "")).strip()
    if rng_mode:
        if rng_mode not in MINIMAX_H3_RNG_MODES:
            raise ValueError(f"{MINIMAX_H3_RNG_ENV} must be one of {sorted(MINIMAX_H3_RNG_MODES)}, got {rng_mode!r}")
        strategy = replace(strategy, rng_mode=rng_mode)

    condition_noise = str(environ.get(MINIMAX_H3_CONDITION_NOISE_ENV, "")).strip()
    if condition_noise:
        if condition_noise not in MINIMAX_H3_CONDITION_NOISE_MODES:
            raise ValueError(
                f"{MINIMAX_H3_CONDITION_NOISE_ENV} must be one of "
                f"{sorted(MINIMAX_H3_CONDITION_NOISE_MODES)}, got {condition_noise!r}"
            )
        strategy = replace(strategy, visual_condition_noise_shape_mode=condition_noise)

    if not strategy.is_official:
        # The short edge predates this strategy and has always been usable under
        # legacy. It stopped working the moment the strategy began supplying an
        # explicit value, because the pipeline resolves it as
        # `short_edge or _reference_image_short_edge()` and the strategy's 2048
        # is truthy — so the env was shadowed rather than overridden, a dead
        # knob that still reads as live in a deploy file. Parse it here so the
        # resolved strategy carries the operator's value.
        raw = str(environ.get(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV, "")).strip()
        if raw:
            strategy = replace(strategy, reference_image_short_edge=_parse_short_edge(raw))
        return replace(
            strategy,
            reference_image_no_upscale=parse_reference_image_no_upscale(
                environ.get(MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV, "")
            ),
            reference_image_max_pixels=parse_reference_image_max_pixels(
                environ.get(MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV, "")
            ),
        )

    conflicting = [key for key in _CONFLICTING_REFERENCE_IMAGE_ENVS if str(environ.get(key, "")).strip()]
    if not conflicting:
        return strategy
    # The short edge is a legitimate production knob *when the aspect ratio is
    # preserved*: a smaller short edge trades detail for tokens, while a canvas
    # pre-stretch trades the reference's proportions for them, and only the
    # second is a defect the product refuses to ship. The other two switches
    # still move geometry in ways an oracle cannot mirror, so they stay
    # parity-only.
    if strategy.reference_image_geometry_mode == "official_short_edge" and conflicting == [
        MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV
    ]:
        pass
    elif policy != MINIMAX_H3_ADMISSION_PARITY:
        raise ValueError(
            f"inference_contract={MINIMAX_H3_CONTRACT_OFFICIAL_V1} fixes the reference-image geometry at "
            "short edge 2048, upscaling on and no area cap, but "
            f"{', '.join(sorted(conflicting))} is set. Unset it, run the legacy contract, or — for an "
            f"oracle comparison only — admission_policy={MINIMAX_H3_ADMISSION_PARITY}."
        )

    # Parity fixture: the override is allowed, but only the short edge, and the
    # resolved value is carried on the strategy so `describe()` reports it.
    # Nothing else about the official geometry moves.
    unsupported = [key for key in conflicting if key != MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV]
    if unsupported:
        raise ValueError(
            f"admission_policy={MINIMAX_H3_ADMISSION_PARITY} relaxes only the reference-image short edge; "
            f"{', '.join(sorted(unsupported))} would change the geometry in ways an oracle cannot mirror."
        )
    raw = str(environ[MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV]).strip()
    return replace(strategy, reference_image_short_edge=_parse_short_edge(raw))


__all__ = [
    "MINIMAX_H3_ADMISSIONS",
    "MINIMAX_H3_ADMISSION_PARITY",
    "MINIMAX_H3_ADMISSION_PRODUCTION",
    "MINIMAX_H3_ADMISSION_ENV",
    "MINIMAX_H3_CONTRACTS",
    "MINIMAX_H3_CONTRACT_ENV",
    "MINIMAX_H3_CONDITION_NOISE_ENV",
    "MINIMAX_H3_CONDITION_NOISE_MODES",
    "MINIMAX_H3_GEOMETRY_ENV",
    "MINIMAX_H3_GEOMETRY_MODES",
    "MINIMAX_H3_RNG_ENV",
    "MINIMAX_H3_CONTRACT_LEGACY",
    "MINIMAX_H3_CONTRACT_OFFICIAL_V1",
    "MINIMAX_H3_LEGACY_DEFAULT_NUM_FRAMES",
    "MINIMAX_H3_OFFICIAL_DEFAULT_NUM_FRAMES",
    "MiniMaxH3InferenceStrategy",
    "legacy_strategy",
    "official_diffusers_v1_strategy",
    "resolve_strategy",
]
