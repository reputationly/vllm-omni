# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ImageTaskRequest size/input resolution (CPU-only).

Covers the GPUStack facade contract: ``target_shape`` is [height, width] and
wins over ``aspect_ratio``; a bare ratio must resolve to concrete dimensions
because nothing downstream consumes ``aspect_ratio``; ``image_path`` arrives
comma-joined for multi-image edit.
"""

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.image_tasks import (
    _RATIO_BASE_AREA,
    _SIZE_ALIGNMENT,
    ImageTaskRequest,
)


def _req(**kwargs) -> ImageTaskRequest:
    kwargs.setdefault("prompt", "a cat")
    return ImageTaskRequest(**kwargs)


# --------------------------------------------------------------------- size


def test_no_size_hint_is_auto():
    """i2i without an explicit size must stay auto so HunyuanImage-3.0's
    AR-predicted <img_ratio_*> keeps deciding the output shape."""
    assert _req().output_size() == (None, None)


def test_target_shape_is_height_width():
    # facade sends [h, w]; output_size returns (w, h)
    assert _req(target_shape=[720, 1280]).output_size() == (1280, 720)


@pytest.mark.parametrize(
    "shape",
    [None, [], [1024], [1024, 1024, 3], [0, 1024], [1024, -1]],
)
def test_malformed_target_shape_falls_through_to_auto(shape):
    """Wrong arity or non-positive values degrade to auto rather than erroring."""
    assert _req(target_shape=shape).output_size() == (None, None)


def test_non_integer_target_shape_is_rejected_at_validation():
    """Non-numeric entries never reach output_size(): pydantic's list[int] rejects
    them, so a facade sending garbage gets a 422 at submit instead of a task that
    fails later."""
    with pytest.raises(ValidationError):
        _req(target_shape=["a", "b"])


def test_target_shape_wins_over_aspect_ratio():
    """Facade precedence: '有 target_shape 时引擎会优先用后者'."""
    req = _req(target_shape=[720, 1280], aspect_ratio="1:1")
    assert req.output_size() == (1280, 720)


@pytest.mark.parametrize(
    "ratio,expected_wh_order",
    [
        ("1:1", "square"),
        ("16:9", "landscape"),
        ("9:16", "portrait"),
        (" 16 : 9 ", "landscape"),  # direct callers may not normalize
    ],
)
def test_aspect_ratio_resolves_to_concrete_size(ratio, expected_wh_order):
    width, height = _req(aspect_ratio=ratio).output_size()
    assert width is not None and height is not None
    if expected_wh_order == "square":
        assert width == height
    elif expected_wh_order == "landscape":
        assert width > height
    else:
        assert height > width


def test_aspect_ratio_result_is_alignment_safe():
    """Latent grids must divide evenly or the pipeline errors."""
    for ratio in ("1:1", "16:9", "9:16", "4:3", "3:2", "21:9"):
        width, height = _req(aspect_ratio=ratio).output_size()
        assert width % _SIZE_ALIGNMENT == 0, ratio
        assert height % _SIZE_ALIGNMENT == 0, ratio


def test_aspect_ratio_roughly_preserves_ratio_and_area():
    width, height = _req(aspect_ratio="16:9").output_size()
    # Alignment rounding perturbs both slightly; allow a loose tolerance.
    assert abs((width / height) - (16 / 9)) < 0.05
    assert 0.8 < (width * height) / _RATIO_BASE_AREA < 1.25


def test_square_ratio_lands_on_the_base_resolution():
    assert _req(aspect_ratio="1:1").output_size() == (1024, 1024)


@pytest.mark.parametrize("ratio", ["", "16", "16:", ":9", "16:0", "0:9", "-16:9", "16:9:3", "abc", "16x9"])
def test_malformed_aspect_ratio_is_auto_not_an_error(ratio):
    """A junk ratio must not raise: it degrades to auto, same as sending none."""
    assert _req(aspect_ratio=ratio).output_size() == (None, None)


def test_max_pixels_caps_the_ratio_budget():
    """The area is OUR choice for a ratio request, so it must fit the operator's
    limit rather than 400-ing every ratio request on a tightly capped server."""
    capped_w, capped_h = _req(aspect_ratio="1:1").output_size(max_pixels=512 * 512)
    assert capped_w * capped_h <= 512 * 512
    uncapped_w, uncapped_h = _req(aspect_ratio="1:1").output_size()
    assert capped_w < uncapped_w


@pytest.mark.parametrize("ratio", ["1:1", "16:9", "9:16", "4:3", "3:2", "21:9", "2:1", "1:2"])
@pytest.mark.parametrize("cap", [512 * 512, 768 * 768, 640 * 360, 1024 * 1024, 2048 * 2048])
def test_capped_ratio_never_exceeds_the_cap(ratio, cap):
    """Regression: nearest-multiple rounding used to round UP past the cap, so a
    capped ratio request got a 400 from the very check this branch exists to
    satisfy (16:9 under 512x512 produced 704x384 = 270,336 > 262,144)."""
    width, height = _req(aspect_ratio=ratio).output_size(max_pixels=cap)
    assert width * height <= cap, f"{ratio} @ {cap} -> {width}x{height}"
    assert width % _SIZE_ALIGNMENT == 0 and height % _SIZE_ALIGNMENT == 0


def test_the_specific_regression_case():
    assert _req(aspect_ratio="16:9").output_size(max_pixels=512 * 512) == (640, 384)


def test_cap_above_the_base_area_does_not_inflate():
    """A generous cap must not scale the image UP past the base budget."""
    assert _req(aspect_ratio="1:1").output_size(max_pixels=4096 * 4096) == (1024, 1024)


def test_max_pixels_does_not_shrink_an_explicit_target_shape():
    """An explicit oversized size is the caller's error; the route's size check
    rejects it with 400 instead of being silently shrunk here."""
    assert _req(target_shape=[2048, 2048]).output_size(max_pixels=512 * 512) == (2048, 2048)


@pytest.mark.parametrize("max_pixels", [None, 0, -1])
def test_absent_or_nonpositive_max_pixels_uses_the_default_budget(max_pixels):
    assert _req(aspect_ratio="1:1").output_size(max_pixels=max_pixels) == (1024, 1024)


# ------------------------------------------------------------------- inputs


def test_image_path_splits_on_comma():
    req = _req(image_path="/nfs/a.png,/nfs/b.png,/nfs/c.png")
    assert req.input_image_paths() == ["/nfs/a.png", "/nfs/b.png", "/nfs/c.png"]


@pytest.mark.parametrize("value", [None, "", "   ", ",", " , "])
def test_blank_image_path_means_text_to_image(value):
    assert _req(image_path=value).input_image_paths() == []


def test_image_path_tolerates_padding_and_trailing_comma():
    req = _req(image_path=" /nfs/a.png , /nfs/b.png , ")
    assert req.input_image_paths() == ["/nfs/a.png", "/nfs/b.png"]


def test_mask_image_path_is_exposed():
    """The facade maps its "image_mask" input to image_mask_path; the field must
    be reachable, or an inpainting request silently runs unmasked."""
    assert _req(image_mask_path="/nfs/mask.png").mask_image_path() == "/nfs/mask.png"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_blank_mask_is_none(value):
    assert _req(image_mask_path=value).mask_image_path() is None


# ------------------------------------------------------------- extra_body


def test_prompt_aliases():
    for key in ("prompt", "input", "text"):
        assert ImageTaskRequest(**{key: "hello"}).prompt == "hello"


def test_extra_body_carries_typed_and_unknown_params():
    req = _req(
        negative_prompt="blurry",
        num_inference_steps=8,
        bot_task="image",
        some_future_knob=1.5,
    )
    extra = req.diffusion_extra_body()
    assert extra["negative_prompt"] == "blurry"
    assert extra["num_inference_steps"] == 8
    assert extra["bot_task"] == "image"
    # extra="allow" forward-compat: unknown pipeline params must not be dropped.
    assert extra["some_future_knob"] == 1.5


def test_bot_task_vocabulary():
    """The residency config documents a fast path; the value it names must exist.

    "image" is the upstream HuggingFace spelling and is NOT valid here — this
    stack calls the same thing "vanilla". An unknown value is worse than useless
    in a residency deployment: requires_ar_generation() defaults unknowns to True,
    so the AR engine is woken before anything notices the name is wrong.
    """
    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        available_bot_tasks,
        requires_ar_generation,
    )

    valid = available_bot_tasks()
    assert "image" not in valid
    for name in ("vanilla", "think", "recaption", "think_recaption"):
        assert name in valid, name
    assert None in valid

    # Fast path: no AR. Quality path: AR.
    assert requires_ar_generation(None) is False
    assert requires_ar_generation("vanilla") is False
    assert requires_ar_generation("recaption") is True
    assert requires_ar_generation("think") is True
    assert requires_ar_generation("think_recaption") is True
    # Unknown errs toward running AR rather than silently dropping a caller's
    # requested reasoning; the route rejects it at submit before that matters.
    assert requires_ar_generation("image") is True


def test_extra_body_forwards_sys_type():
    """sys_type must survive to the diffusion layer.

    The DiT pipeline reads extra_args["use_system_prompt"]; the route maps
    sys_type onto it. Dropping it here would silently ignore the caller's
    requested system prompt.
    """
    extra = _req(sys_type="en_vanilla").diffusion_extra_body()
    assert extra["sys_type"] == "en_vanilla"


def test_layered_fields_reach_the_diffusion_layer():
    """Regression: giving layers/resolution types silently unplugged them.

    While undeclared they rode model_extra into extra_body. Declaring them (to
    validate them) removed them from model_extra, and they were not added to the
    typed forwarding list — so the route validated a value it then discarded.
    Declaring a field must never be what stops it from being delivered.
    """
    extra = _req(layers=4, resolution=640).diffusion_extra_body()
    assert extra["layers"] == 4
    assert extra["resolution"] == 640


def test_extra_body_omits_route_owned_and_unset_keys():
    req = _req(target_shape=[720, 1280], aspect_ratio="16:9", image_path="/nfs/a.png", n=2)
    extra = req.diffusion_extra_body()
    # Consumed structurally by output_size()/the route, not forwarded.
    for key in ("target_shape", "aspect_ratio", "image_path", "n", "prompt", "model"):
        assert key not in extra
    # Unset optional params must not appear as None and override model defaults.
    assert "guidance_scale" not in extra
    assert "seed" not in extra
