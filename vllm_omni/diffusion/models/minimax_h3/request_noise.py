# SPDX-License-Identifier: Apache-2.0
"""The random draws of one MiniMax-H3 request, in one testable place.

A request draws three kinds of noise — one per visual condition, then the target
video latent, then the target audio rows — and *the order and shape of those
draws is part of the request contract*, not an implementation detail: two runs
that consume the same generator differently produce different videos from the
same seed.

Two contracts live here, chosen at startup and never mixed within a request:

``legacy``
    What vLLM-Omni has always done. Every draw builds its own CPU generator:
    ``manual_seed(seed)`` for the video latent, ``manual_seed(seed)`` *again*
    for the audio rows (so audio restarts the same stream the video used rather
    than continuing it), ``manual_seed(seed)`` per visual condition and
    ``manual_seed(seed + 1)`` per audio condition. A visual condition is also
    drawn at ``max(target_latent_t + num_conditions, latent_t)`` frames and
    sliced back to its own length, so even the values at a given position are
    not the condition-shaped draw.

``official_diffusers_v1``
    What the pinned Diffusers ``MiniMaxH3ModularPipeline`` does: **one** CPU
    generator per request, consumed in block order — every visual condition at
    its own exact shape, then the video latent, then the channel-major audio
    rows.

Each contract pins two things at once — which generator draws, and what shape a
visual condition is drawn at — and ``condition_shape_mode`` lets an attribution
run move the second without the first. Production never mixes them; the
isolation knob exists so "the RNG changed the picture" and "the draw shape
changed the picture" can be told apart, which they cannot be while one switch
moves both.

Switching contracts changes the output of every seed, including ``t2va``, which
draws no conditioning at all: video and audio stop being two draws off the same
stream start and become one continuous stream. That is why this is a startup
choice with ``legacy`` as the default, not a bug fix.

The VAE posterior's fixed ``encode_seed = 42`` is a separate RNG and never
touches the request generator; see ``vae.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .packed_tokens import minimax_h3_patchify_video_latent

MINIMAX_H3_RNG_LEGACY = "legacy"
MINIMAX_H3_RNG_OFFICIAL_V1 = "official_diffusers_v1"
MINIMAX_H3_RNG_MODES = frozenset({MINIMAX_H3_RNG_LEGACY, MINIMAX_H3_RNG_OFFICIAL_V1})

# *How much* noise a visual condition draws is a second axis, separate from
# *which generator* draws it. Each contract pins both, but an attribution run
# has to be able to move one and hold the other — otherwise "the RNG changed
# the output" and "the draw shape changed the output" are indistinguishable.
MINIMAX_H3_CONDITION_SHAPE_EXACT = "condition_shape"
MINIMAX_H3_CONDITION_SHAPE_LEGACY = "legacy_oversized_slice"
MINIMAX_H3_CONDITION_SHAPE_MODES = frozenset({MINIMAX_H3_CONDITION_SHAPE_EXACT, MINIMAX_H3_CONDITION_SHAPE_LEGACY})
_DEFAULT_CONDITION_SHAPE_BY_RNG = {
    MINIMAX_H3_RNG_LEGACY: MINIMAX_H3_CONDITION_SHAPE_LEGACY,
    MINIMAX_H3_RNG_OFFICIAL_V1: MINIMAX_H3_CONDITION_SHAPE_EXACT,
}

# Checkpoint contract: 24 video latent channels packed with a [1, 2, 2] patch,
# 32 audio latent channels carried as two channel-major blocks.
MINIMAX_H3_VIDEO_LATENT_CHANNELS = 24
MINIMAX_H3_AUDIO_LATENT_CHANNELS = 32
MINIMAX_H3_AUDIO_CHANNELS = 2
MINIMAX_H3_PATCH_SIZE = (1, 2, 2)


class MiniMaxH3RequestNoisePlan:
    """The ordered noise draws of one request.

    Args:
        rng_mode: ``legacy`` or ``official_diffusers_v1``. Decides *which*
            generator every draw comes from.
        seed: The request seed.
        condition_shape_mode: ``condition_shape`` or ``legacy_oversized_slice``
            — decides the *shape* a visual condition is drawn at, independently
            of ``rng_mode``. ``None`` takes the contract's own pairing, so a
            caller that only knows the RNG mode keeps today's behaviour.

    Under ``official_diffusers_v1`` the three ``draw_*`` methods must be called
    in contract order — conditions, then video, then audio — because they share
    one generator and the order *is* the contract. Calling them out of order
    raises rather than silently producing a different request; under ``legacy``
    every draw is independent, so the order is not enforced.
    """

    _ORDER = ("visual_condition", "video", "audio")

    def __init__(
        self,
        *,
        rng_mode: str,
        seed: int,
        condition_shape_mode: str | None = None,
    ) -> None:
        if rng_mode not in MINIMAX_H3_RNG_MODES:
            raise ValueError(f"rng_mode must be one of {sorted(MINIMAX_H3_RNG_MODES)}, got {rng_mode!r}")
        if condition_shape_mode is None:
            condition_shape_mode = _DEFAULT_CONDITION_SHAPE_BY_RNG[rng_mode]
        if condition_shape_mode not in MINIMAX_H3_CONDITION_SHAPE_MODES:
            raise ValueError(
                f"condition_shape_mode must be one of {sorted(MINIMAX_H3_CONDITION_SHAPE_MODES)}, "
                f"got {condition_shape_mode!r}"
            )
        self.rng_mode = rng_mode
        self.condition_shape_mode = condition_shape_mode
        self.seed = int(seed)
        self._stage = 0
        self._generator: torch.Generator | None = None
        if rng_mode == MINIMAX_H3_RNG_OFFICIAL_V1:
            self._generator = torch.Generator(device="cpu").manual_seed(self.seed)

    @property
    def is_official(self) -> bool:
        return self.rng_mode == MINIMAX_H3_RNG_OFFICIAL_V1

    @property
    def draws_exact_condition_shape(self) -> bool:
        return self.condition_shape_mode == MINIMAX_H3_CONDITION_SHAPE_EXACT

    def _advance_to(self, stage_name: str) -> None:
        """Guard the draw order of the shared official generator."""
        if not self.is_official:
            return
        stage = self._ORDER.index(stage_name)
        if stage < self._stage:
            raise RuntimeError(
                f"{MINIMAX_H3_RNG_OFFICIAL_V1} draws {' -> '.join(self._ORDER)} in that order; "
                f"'{stage_name}' was requested after '{self._ORDER[self._stage]}'."
            )
        self._stage = stage

    def _fresh(self, offset: int = 0) -> torch.Generator:
        """A legacy per-draw generator."""
        return torch.Generator(device="cpu").manual_seed(self.seed + offset)

    def generator_state(self) -> torch.Tensor | None:
        """The shared generator's state, or None under ``legacy``.

        Exposed so a parity test can assert *how much* of the stream a request
        consumed, not merely that the tensors it produced happen to match.
        """
        return None if self._generator is None else self._generator.get_state()

    def draw_visual_condition_noise(
        self,
        condition_shapes: Sequence[Sequence[int]],
        *,
        target_latent_t: int,
    ) -> list[torch.Tensor]:
        """One noise tensor per visual condition, in packed order.

        Args:
            condition_shapes: ``(latent_t, latent_h, latent_w)`` per condition.
            target_latent_t: Latent frames of the generated video. Only
                ``legacy`` reads it — its draw is sized from the target rather
                than from the condition.

        Returns:
            One ``(1, 24, latent_t, latent_h, latent_w)`` float32 CPU tensor per
            condition.
        """
        shapes = [tuple(int(value) for value in shape) for shape in condition_shapes]
        for shape in shapes:
            if len(shape) != 3 or any(value <= 0 for value in shape):
                raise ValueError(f"condition shape must be a positive (latent_t, latent_h, latent_w), got {shape}")
        self._advance_to("visual_condition")

        drawn: list[torch.Tensor] = []
        for latent_t, latent_h, latent_w in shapes:
            # Two independent choices. The generator comes from ``rng_mode``:
            # one shared stream (official) or one rebuilt per condition
            # (legacy). The number of frames drawn comes from
            # ``condition_shape_mode``: the condition's own (official) or the
            # oversized draw legacy slices a prefix out of. Mixing them is the
            # whole point — it is what makes the two axes attributable.
            generator = self._generator if self.is_official else self._fresh()
            if self.draws_exact_condition_shape:
                noise = torch.randn(
                    (1, MINIMAX_H3_VIDEO_LATENT_CHANNELS, latent_t, latent_h, latent_w),
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                )
            else:
                full_t = max(int(target_latent_t) + len(shapes), latent_t)
                noise = torch.randn(
                    (1, MINIMAX_H3_VIDEO_LATENT_CHANNELS, full_t, latent_h, latent_w),
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                )[:, :, :latent_t]
            drawn.append(noise)
        return drawn

    def draw_video_noise(self, *, latent_t: int, latent_h: int, latent_w: int) -> torch.Tensor:
        """The target video latent noise, as a ``(1, 24, T, H, W)`` tensor."""
        self._advance_to("video")
        generator = self._generator if self.is_official else self._fresh()
        return torch.randn(
            (1, MINIMAX_H3_VIDEO_LATENT_CHANNELS, int(latent_t), int(latent_h), int(latent_w)),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )

    def draw_audio_noise(self, *, audio_t: int) -> torch.Tensor:
        """The target audio noise, drawn directly in channel-major row layout."""
        self._advance_to("audio")
        generator = self._generator if self.is_official else self._fresh()
        return torch.randn(
            (int(audio_t) * MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_LATENT_CHANNELS),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )

    def initial_rows(
        self,
        *,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The generated rows: patchified video rows and channel-major audio rows.

        This is the shape ``pipeline_minimax_h3._initial_noise`` consumed, kept
        so the pipeline reads one call rather than open-coding the draw order.
        """
        video = self.draw_video_noise(latent_t=latent_t, latent_h=latent_h, latent_w=latent_w)
        video_rows = minimax_h3_patchify_video_latent(video, patch_size=MINIMAX_H3_PATCH_SIZE)
        audio_rows = self.draw_audio_noise(audio_t=audio_t)
        return video_rows, audio_rows


__all__ = [
    "MINIMAX_H3_AUDIO_CHANNELS",
    "MINIMAX_H3_AUDIO_LATENT_CHANNELS",
    "MINIMAX_H3_CONDITION_SHAPE_EXACT",
    "MINIMAX_H3_CONDITION_SHAPE_LEGACY",
    "MINIMAX_H3_CONDITION_SHAPE_MODES",
    "MINIMAX_H3_RNG_LEGACY",
    "MINIMAX_H3_RNG_MODES",
    "MINIMAX_H3_RNG_OFFICIAL_V1",
    "MINIMAX_H3_VIDEO_LATENT_CHANNELS",
    "MiniMaxH3RequestNoisePlan",
]
