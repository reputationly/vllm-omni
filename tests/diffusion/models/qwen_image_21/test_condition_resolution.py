# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import calculate_dimensions
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import (
    CONDITION_BUDGET_ENV,
    OUTPUT_RESOLUTION,
    _allocate_condition_areas,
    get_qwen_image_21_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

FLOOR = OUTPUT_RESOLUTION * OUTPUT_RESOLUTION
POSTER = (1696, 2528)


@pytest.fixture
def preprocess(tmp_path):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps({"temperal_downsample": [False] * 4}))
    return get_qwen_image_21_pre_process_func(SimpleNamespace(model=str(tmp_path)))


def condition_sizes(preprocess, sizes, width, height, **sampling):
    prompt = {"prompt": "Keep every word", "multi_modal_data": {"image": [Image.new("RGB", s) for s in sizes]}}
    if "negative_prompt" in sampling:
        prompt["negative_prompt"] = sampling.pop("negative_prompt")
    request = OmniDiffusionRequest(
        request_id="r",
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(
            height=height, width=width, num_inference_steps=2, seed=42, **sampling
        ),
    )
    return preprocess(request).prompt["additional_information"]["input_image_sizes"]


def test_1024_class_output_keeps_the_historical_size(preprocess):
    # Large reference, 1024-class output: must resize exactly as the fixed path did.
    sizes = condition_sizes(preprocess, [POSTER], 1024, 1024)
    assert sizes == [calculate_dimensions(FLOOR, POSTER[0] / POSTER[1])]


def test_small_references_stay_at_the_floor_even_for_2k_output(preprocess):
    sizes = condition_sizes(preprocess, [(600, 600)], 2048, 2048)
    assert sizes == [calculate_dimensions(FLOOR, 1.0)]


def test_2k_output_lifts_a_large_reference_to_the_output_class(preprocess):
    ((width, height),) = condition_sizes(preprocess, [POSTER], 1696, 2528)
    assert width * height > 3.5 * FLOOR


def test_explicit_condition_resolution_wins(preprocess):
    sizes = condition_sizes(preprocess, [POSTER], 1696, 2528, condition_resolution=1024)
    assert sizes == [calculate_dimensions(FLOOR, POSTER[0] / POSTER[1])]


def test_budget_env_bounds_the_sum(preprocess, monkeypatch):
    monkeypatch.setenv(CONDITION_BUDGET_ENV, "6")
    sizes = condition_sizes(preprocess, [(2048, 2048)] * 3, 2048, 2048)
    total = sum(w * h for w, h in sizes)
    assert total <= 6e6 * 1.03  # 32-pixel rounding
    assert len({w for w, _ in sizes}) == 1  # equal share


def test_cfg_with_negative_prompt_halves_the_budget(preprocess):
    plain = condition_sizes(preprocess, [(2048, 2048)] * 4, 2048, 2048)
    guided = condition_sizes(preprocess, [(2048, 2048)] * 4, 2048, 2048, true_cfg_scale=4.0, negative_prompt="blurry")
    assert sum(w * h for w, h in guided) < sum(w * h for w, h in plain)


def test_invalid_budget_env_raises(preprocess, monkeypatch):
    monkeypatch.setenv(CONDITION_BUDGET_ENV, "lots")
    with pytest.raises(ValueError, match=CONDITION_BUDGET_ENV):
        condition_sizes(preprocess, [POSTER], 1696, 2528)


@pytest.mark.parametrize(
    "caps_px,budget,expected",
    [
        pytest.param([4e6], 10e6, [4e6], id="under-budget"),
        pytest.param([4e6, 4e6, 4e6], 9e6, [3e6, 3e6, 3e6], id="equal-share"),
        pytest.param([FLOOR, 4e6, 4e6], 7e6, [FLOOR, (7e6 - FLOOR) / 2, (7e6 - FLOOR) / 2], id="small-hands-over"),
        pytest.param([4e6] * 10, 5e6, [FLOOR] * 10, id="never-below-floor"),
    ],
)
def test_water_filling(caps_px, budget, expected):
    # Square images whose pixel count equals the intended cap; output area never binds.
    sizes = [(round(c**0.5), round(c**0.5)) for c in caps_px]
    areas = _allocate_condition_areas(sizes, output_area=1e9, budget=budget)
    assert areas == pytest.approx(expected, rel=1e-3)
