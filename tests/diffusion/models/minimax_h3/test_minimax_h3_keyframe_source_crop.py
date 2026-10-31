# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pika-CLI ``keyframe_fit="crop"`` parity tests for the FL2VA anchor path.

Pika's released CLI centre-crops every keyframe in source space to the canvas
aspect, then LANCZOS-resizes it — both anchors included. The
``source_center_crop`` mode reproduces that arithmetic verbatim; these tests
pin the geometry against a copy of the upstream function and the pixel outcome
against a painted-border fixture.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _pika_center_crop_box(source_width: int, source_height: int, width: int, height: int):
    """Verbatim copy of Pika-H3 ``core/conditioning.py`` ``center_crop_box``."""
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError("source and target dimensions must be positive")
    aspect = width / height
    if source_width / source_height > aspect:
        crop_height = source_height
        crop_width = max(1, min(source_width, round(source_height * aspect)))
    else:
        crop_width = source_width
        crop_height = max(1, min(source_height, round(source_width / aspect)))
    left, top = (source_width - crop_width) // 2, (source_height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


_GEOMETRY_CASES = [
    # name, source, canvas — square, portrait, 4:3, wider-than-canvas, exact, off-by-one
    ("square_to_16x9", (1024, 1024), (1344, 736)),
    ("portrait_9x16", (768, 1344), (1344, 736)),
    ("four_by_three", (1024, 768), (1344, 736)),
    ("wider_than_canvas_21x9", (2016, 864), (1344, 736)),
    ("exact_canvas", (1344, 736), (1344, 736)),
    ("off_by_one", (1345, 737), (1344, 736)),
    ("off_by_one_narrow", (1343, 735), (1344, 736)),
]


@pytest.mark.parametrize("name,source,canvas", _GEOMETRY_CASES)
def test_center_crop_box_matches_pika(name: str, source: tuple[int, int], canvas: tuple[int, int]) -> None:
    from vllm_omni.diffusion.models.minimax_h3.keyframes import center_crop_box

    assert center_crop_box(*source, *canvas) == _pika_center_crop_box(*source, *canvas)


def _bordered_image(width: int, height: int, border: int = 40) -> Image.Image:
    """Grey field with a bright border the centre crop must cut away."""
    image = Image.new("RGB", (width, height), (64, 64, 64))
    for y in range(height):
        for x in range(width):
            if x < border or y < border or x >= width - border or y >= height - border:
                image.putpixel((x, y), (255, 0, 0))
    return image


def test_source_center_crop_crops_both_anchors() -> None:
    """Off-aspect borders must be cut on both keyframes, not just index 1."""
    from vllm_omni.diffusion.models.minimax_h3.keyframes import prepare_fl2va_keyframes

    width, height = 1344, 736
    # Square source: the crop keeps full width and cuts top/bottom, so the
    # surviving border pixels are only at the sides. The horizontal centre of
    # the output's top and bottom edges must not be border-red.
    anchor = _bordered_image(1024, 1024)
    follower = _bordered_image(1024, 1024)
    prepared = prepare_fl2va_keyframes([anchor, follower], width=width, height=height, mode="source_center_crop")
    assert [p.size for p in prepared] == [(width, height)] * 2
    for keyframe in prepared:
        assert keyframe.getpixel((width // 2, 0)) != (255, 0, 0)
        assert keyframe.getpixel((width // 2, height - 1)) != (255, 0, 0)
    # Wider-than-canvas source: the crop cuts the sides instead.
    wide = _bordered_image(2016, 864)
    prepared = prepare_fl2va_keyframes([wide, wide], width=width, height=height, mode="source_center_crop")
    for keyframe in prepared:
        assert keyframe.getpixel((0, height // 2)) != (255, 0, 0)
        assert keyframe.getpixel((width - 1, height // 2)) != (255, 0, 0)


def test_source_center_crop_same_size_untouched() -> None:
    from vllm_omni.diffusion.models.minimax_h3.keyframes import prepare_fl2va_keyframes

    width, height = 1344, 736
    anchor = _bordered_image(width, height)
    prepared = prepare_fl2va_keyframes([anchor], width=width, height=height, mode="source_center_crop")
    assert prepared[0] is anchor


def test_other_modes_byte_identical() -> None:
    """legacy_stretch / official_cover_crop outputs must not move."""
    from vllm_omni.diffusion.models.minimax_h3.keyframes import prepare_fl2va_keyframes

    width, height = 1344, 736
    anchor = _bordered_image(1024, 1024)
    follower = _bordered_image(864, 480)
    for mode in ("legacy_stretch", "official_cover_crop"):
        first = prepare_fl2va_keyframes([anchor, follower], width=width, height=height, mode=mode)
        second = prepare_fl2va_keyframes(
            [_bordered_image(1024, 1024), _bordered_image(864, 480)], width=width, height=height, mode=mode
        )
        for a, b in zip(first, second):
            assert list(a.getdata()) == list(b.getdata())
        # anchor stays a plain stretch under both modes
        if mode == "official_cover_crop":
            expected = anchor.resize((width, height), Image.Resampling.LANCZOS)
            assert list(first[0].getdata()) == list(expected.getdata())


def test_model_metadata_no_prestretch_for_source_crop(monkeypatch: Any) -> None:
    from vllm_omni.diffusion import model_metadata

    strategy = SimpleNamespace(
        fl2va_keyframe_resize_mode="source_center_crop",
        reference_image_geometry_mode="legacy_canvas_prestretch",
    )
    monkeypatch.setattr(model_metadata, "_minimax_h3_strategy", lambda cfg: strategy)
    monkeypatch.setattr(
        model_metadata,
        "get_diffusion_model_metadata",
        lambda name: SimpleNamespace(reference_images_bind_output_canvas=True),
    )
    result = model_metadata.reference_images_bind_output_canvas("MiniMaxH3Pipeline", object(), task_type="fl2va")
    assert result is False


def test_env_mode_constant_accepted() -> None:
    from vllm_omni.diffusion.models.minimax_h3.keyframes import (
        MINIMAX_H3_KEYFRAME_RESIZE_MODES,
        MINIMAX_H3_KEYFRAME_RESIZE_SOURCE_CROP,
    )

    assert MINIMAX_H3_KEYFRAME_RESIZE_SOURCE_CROP in MINIMAX_H3_KEYFRAME_RESIZE_MODES
