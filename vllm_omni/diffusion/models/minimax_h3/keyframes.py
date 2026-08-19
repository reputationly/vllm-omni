# SPDX-License-Identifier: Apache-2.0
"""Putting the `fl2va` keyframes onto the target canvas.

The two keyframes are not treated alike. The first one is the *geometry anchor*:
the canvas is derived from its aspect ratio, so stretching it onto that canvas
is a no-op in shape and the released model was conditioned that way. The second
one is a follower that has to land on a canvas it did not choose, and the
released model was conditioned on it **cover-scaled and centre-cropped**, not
stretched — a follower whose aspect ratio differs from the canvas comes out
distorted otherwise, and the conditioning latents move off the reference
implementation.

The crop arithmetic is the released model's own and is reproduced literally:
``round`` for the scaled size and ``(resized - target) // 2`` for the offset.
Diffusers' ``VaeImageProcessor(resize_mode="crop")`` is deliberately *not* used:
it floor-divides the size and centres with ``w // 2 - src_w // 2``, which agrees
on some aspect ratios and differs by a pixel on others (106 of 218 sampled in
the upstream port), and a one-pixel shift is a different condition.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

MINIMAX_H3_KEYFRAME_RESIZE_OFFICIAL = "official_cover_crop"
MINIMAX_H3_KEYFRAME_RESIZE_LEGACY = "legacy_stretch"
MINIMAX_H3_KEYFRAME_RESIZE_MODES = frozenset({MINIMAX_H3_KEYFRAME_RESIZE_OFFICIAL, MINIMAX_H3_KEYFRAME_RESIZE_LEGACY})


def cover_crop_geometry(
    source_width: int,
    source_height: int,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The follower keyframe's scaled size and crop offset.

    Split out from the resize so the integers can be asserted against the oracle
    on their own — a pixel comparison that fails tells you the pixels differ, not
    which of the two steps moved.

    Args:
        source_width, source_height: The follower's own size.
        width, height: The target canvas.

    Returns:
        ``((resized_width, resized_height), (left, top))``.
    """
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError(
            f"keyframe geometry needs positive sizes, got source {source_width}x{source_height} canvas {width}x{height}"
        )
    scale = max(width / source_width, height / source_height)
    resized = (max(width, round(source_width * scale)), max(height, round(source_height * scale)))
    offset = (max(0, (resized[0] - width) // 2), max(0, (resized[1] - height) // 2))
    return resized, offset


def prepare_fl2va_keyframes(
    images: Sequence[Image.Image],
    *,
    width: int,
    height: int,
    mode: str,
) -> list[Image.Image]:
    """Put the keyframes onto the target canvas, in packed order.

    Args:
        images: The keyframes, first-frame anchor first.
        width, height: The resolved target canvas.
        mode: ``official_cover_crop`` or ``legacy_stretch``.

    Returns:
        One image per input, each exactly ``(width, height)``.
    """
    if mode not in MINIMAX_H3_KEYFRAME_RESIZE_MODES:
        raise ValueError(f"mode must be one of {sorted(MINIMAX_H3_KEYFRAME_RESIZE_MODES)}, got {mode!r}")

    prepared: list[Image.Image] = []
    for index, image in enumerate(images):
        if image.size == (width, height):
            # Already on the canvas: both contracts leave it untouched rather
            # than round-tripping it through a resample.
            prepared.append(image)
        elif mode == MINIMAX_H3_KEYFRAME_RESIZE_LEGACY or index == 0:
            prepared.append(image.resize((width, height), Image.Resampling.LANCZOS))
        else:
            resized_size, (left, top) = cover_crop_geometry(image.size[0], image.size[1], width, height)
            resized = image.resize(resized_size, Image.Resampling.LANCZOS)
            prepared.append(resized.crop((left, top, left + width, top + height)))
    return prepared


__all__ = [
    "MINIMAX_H3_KEYFRAME_RESIZE_LEGACY",
    "MINIMAX_H3_KEYFRAME_RESIZE_MODES",
    "MINIMAX_H3_KEYFRAME_RESIZE_OFFICIAL",
    "cover_crop_geometry",
    "prepare_fl2va_keyframes",
]
