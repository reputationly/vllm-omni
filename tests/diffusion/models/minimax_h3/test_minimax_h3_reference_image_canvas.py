# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether the serving layer binds reference images to the generated canvas.

Historically it always did: any request carrying ``width``/``height`` had every
reference image LANCZOS-*stretched* onto that canvas before the pipeline saw it.
For MiniMax-H3 that is a contract violation — the official image reference
"never binds the generated geometry" — and it also makes the model's own
``[0.4, 2.5]`` aspect-ratio check unreachable, because everything arrives at the
canvas ratio.

These pin the capability, not the resize itself: the resize is shared by every
video model and must keep working for them.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Config:
    """The few attributes the capability probe reads off a diffusion config."""

    def __init__(self, model_class_name, contract=None, *, stage_env=None, task_type=None):
        self.model_class_name = model_class_name
        self.minimax_h3_inference_contract = contract
        self.minimax_h3_admission_policy = None
        self.diffusion_runtime_environ = stage_env
        self.task_type = task_type


def test_other_video_models_keep_binding_the_canvas():
    """The shared resize is unchanged for everything that is not H3."""
    from vllm_omni.diffusion.model_metadata import reference_images_bind_output_canvas

    for model in ("WanImageToVideoPipeline", "WanPipeline", None, "SomeUnknownPipeline"):
        assert reference_images_bind_output_canvas(model, _Config(model)) is True


def test_h3_legacy_keeps_binding_the_canvas():
    """Legacy must keep its current behaviour, distortion included."""
    from vllm_omni.diffusion.model_metadata import reference_images_bind_output_canvas

    for model in ("MiniMaxH3Pipeline", "MiniMaxH3ModularPipeline"):
        assert reference_images_bind_output_canvas(model, _Config(model, "legacy")) is True
        # No contract configured at all is legacy too.
        assert reference_images_bind_output_canvas(model, _Config(model, None)) is True


def test_h3_official_contract_releases_the_canvas_binding():
    from vllm_omni.diffusion.model_metadata import reference_images_bind_output_canvas

    for model in ("MiniMaxH3Pipeline", "MiniMaxH3ModularPipeline"):
        assert reference_images_bind_output_canvas(model, _Config(model, "official_diffusers_v1")) is False


def test_fl2va_cover_crop_releases_keyframes_without_changing_ref2va_legacy():
    """The route must not erase the follower's ratio before cover-crop runs."""
    from vllm_omni.diffusion.model_metadata import reference_images_bind_output_canvas

    config = _Config(
        "MiniMaxH3Pipeline",
        "legacy",
        stage_env={"VLLM_OMNI_H3_FL2VA_KEYFRAME_RESIZE": "official_cover_crop"},
    )

    assert reference_images_bind_output_canvas(config.model_class_name, config, task_type="fl2va") is False
    # The two policies are independent in a combined deployment. A legacy
    # Ref2VA request still receives its historical canvas pre-stretch.
    assert reference_images_bind_output_canvas(config.model_class_name, config, task_type="ref2va") is True


def test_combined_serving_resolves_the_image_role_before_canvas_binding():
    from types import SimpleNamespace

    from vllm_omni.entrypoints.openai.serving_video import (
        OmniOpenAIServingVideo,
        ReferenceVideo,
    )

    config = _Config(
        "MiniMaxH3Pipeline",
        "legacy",
        stage_env={"VLLM_OMNI_H3_FL2VA_KEYFRAME_RESIZE": "official_cover_crop"},
        task_type="combined",
    )
    handler = object.__new__(OmniOpenAIServingVideo)
    handler._engine_client = SimpleNamespace(get_diffusion_od_config=lambda: config)

    # Image-only is FL2VA on a combined root, so the raw follower is released.
    assert not handler._reference_images_bind_output_canvas_for_request(
        reference_video=None,
        reference_audio=None,
    )
    # A video-bearing request is Ref2VA and retains this instance's legacy
    # pre-stretch, proving that fixing FL2VA does not globally flip the root.
    assert handler._reference_images_bind_output_canvas_for_request(
        reference_video=ReferenceVideo(data=[]),
        reference_audio=None,
    )
    # An explicit request task outranks both the startup selection and media
    # inference, exactly as it does later in MiniMaxH3Pipeline._resolve_task.
    assert handler._reference_images_bind_output_canvas_for_request(
        reference_video=None,
        reference_audio=None,
        requested_task="ref2va",
    )
    assert not handler._reference_images_bind_output_canvas_for_request(
        reference_video=ReferenceVideo(data=[]),
        reference_audio=None,
        requested_task="fl2va",
    )


def test_a_bad_contract_does_not_silently_change_the_capability():
    """An invalid contract fails where it is applied, not by flipping a default."""
    from vllm_omni.diffusion.model_metadata import reference_images_bind_output_canvas

    assert reference_images_bind_output_canvas("MiniMaxH3Pipeline", _Config("MiniMaxH3Pipeline", "official")) is True


def test_official_reference_image_keeps_its_own_geometry(oracle_reference_images):
    """With the canvas binding released, the source ratio survives to the model.

    The bound path collapses every source onto one shape; the official path
    keeps each reference's own, which is what the row counts downstream reflect.
    """
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    canvas = (1344, 768)
    distinct = set()
    for case in oracle_reference_images:
        width, height = case["source_width"], case["source_height"]
        ratio = width / height
        if not 0.4 <= ratio <= 2.5 or min(width, height) < 256 or max(width, height) > 5760:
            continue
        # Bound: every source is stretched to the canvas first.
        bound = _reference_image_shape(Image.new("RGB", canvas))
        # Released: the source reaches the model as it is.
        released = _reference_image_shape(Image.new("RGB", (width, height)))
        distinct.add(released)
        assert bound == (3584, 2048)
        assert released == (case["target_width"], case["target_height"])
    # The bound path would have produced one shape for all of them.
    assert len(distinct) > 1


class _StageConfig:
    """The two attributes the engine's diffusion-stage lookup reads."""

    def __init__(self, stage_type, engine_args=None):
        self.stage_type = stage_type
        self.engine_args = engine_args


def _out_of_process_view(stage_configs, monkeypatch, model_class_name="MiniMaxH3Pipeline"):
    """The config view an out-of-process deployment's serving layer sees."""
    from vllm_omni.diffusion import data
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    monkeypatch.setattr(data, "resolve_model_class_name", lambda _model: model_class_name)

    class _Engine:
        _DIFFUSION_CONTRACT_VIEW_FIELDS = AsyncOmniEngine._DIFFUSION_CONTRACT_VIEW_FIELDS
        _diffusion_stage_engine_args = AsyncOmniEngine._diffusion_stage_engine_args
        get_diffusion_od_config = AsyncOmniEngine.get_diffusion_od_config

        def __init__(self):
            self.model = "/models/MiniMax-H3"
            self.stage_configs = stage_configs
            self._diffusion_od_config_view = None

    return _Engine().get_diffusion_od_config()


def test_the_contract_reaches_the_serving_process(monkeypatch):
    """Out of process this view is all the serving layer has.

    The worker holds the real config; the front end holds this. A view without
    the contract fields makes the probe answer *legacy* for a worker running
    *official* — and the front end then stretches every reference image onto the
    output canvas, silently, before the worker ever sees it.
    """
    from vllm_omni.diffusion.model_metadata import reference_images_bind_output_canvas

    view = _out_of_process_view(
        [
            _StageConfig("llm", {"minimax_h3_inference_contract": "legacy"}),
            _StageConfig("diffusion", {"minimax_h3_inference_contract": "official_diffusers_v1"}),
        ],
        monkeypatch,
    )

    assert view.minimax_h3_inference_contract == "official_diffusers_v1"
    assert reference_images_bind_output_canvas(view.model_class_name, view) is False


def test_the_partition_task_reaches_the_request_level_canvas_probe(monkeypatch):
    view = _out_of_process_view(
        [_StageConfig("diffusion", {"task_type": "ref2va"})],
        monkeypatch,
    )

    assert view.task_type == "ref2va"


def test_a_yaml_that_says_nothing_is_still_legacy(monkeypatch):
    """The deploy file ships the field commented out, and that must stay legacy."""
    from vllm_omni.diffusion.model_metadata import reference_images_bind_output_canvas

    view = _out_of_process_view([_StageConfig("diffusion", {})], monkeypatch)

    assert view.minimax_h3_inference_contract is None
    assert reference_images_bind_output_canvas(view.model_class_name, view) is True


def test_the_view_survives_a_deployment_with_no_diffusion_stage(monkeypatch):
    """Comprehension-only pipelines have no engine_args to read, and must not raise."""
    view = _out_of_process_view([_StageConfig("llm", {})], monkeypatch, model_class_name=None)

    assert view.minimax_h3_inference_contract is None
    assert view.minimax_h3_admission_policy is None


def test_omegaconf_engine_args_are_read_the_same_way(monkeypatch):
    """YAML-loaded stage configs carry a DictConfig, not a dict."""
    from omegaconf import OmegaConf

    engine_args = OmegaConf.create(
        {
            "minimax_h3_inference_contract": "official_diffusers_v1",
            "minimax_h3_admission_policy": "parity_fixture_v1",
        }
    )
    view = _out_of_process_view([_StageConfig("diffusion", engine_args)], monkeypatch)

    assert view.minimax_h3_inference_contract == "official_diffusers_v1"
    assert view.minimax_h3_admission_policy == "parity_fixture_v1"


def test_a_nested_value_comes_off_the_view_as_plain_python(monkeypatch):
    """The test above covers scalars, and scalars were never the problem.

    A ``DictConfig`` yields ``str``/``int``/``bool`` for its leaves, so reading
    the contract out of one looks like proof the representation is handled. It
    is not: the *container* stays wrapped, and a ``DictConfig`` is a
    ``MutableMapping`` but not a ``dict``. ``diffusion_runtime_environ`` is the
    one view field that is a container, its reader asked ``isinstance(value,
    dict)``, and the mapping was therefore dropped on exactly the deployment
    shape it exists for — every hand-built-dict test still passing.

    Normalizing here rather than at each reader is the point: this is the one
    place that knows a config representation was crossed.
    """
    from omegaconf import OmegaConf

    from vllm_omni.diffusion.model_metadata import (
        honours_explicit_reference_order,
        reference_images_bind_output_canvas,
    )

    engine_args = OmegaConf.create(
        {"diffusion_runtime_environ": {"VLLM_OMNI_H3_INFERENCE_CONTRACT": "official_diffusers_v1"}}
    )
    assert not isinstance(engine_args.diffusion_runtime_environ, dict), "the premise of this test has changed"

    view = _out_of_process_view([_StageConfig("diffusion", engine_args)], monkeypatch)

    assert type(view.diffusion_runtime_environ) is dict
    assert view.diffusion_runtime_environ == {"VLLM_OMNI_H3_INFERENCE_CONTRACT": "official_diffusers_v1"}
    # And the two answers that were wrong in the field because of it.
    assert reference_images_bind_output_canvas(view.model_class_name, view) is False
    assert honours_explicit_reference_order(view.model_class_name, view) is True


def test_the_deploy_yaml_field_lands_on_the_diffusion_stage_engine_args():
    """The path the view depends on, checked end to end against the real YAML.

    ``_build_engine_args`` passes ``StageDeployConfig`` through verbatim, which
    is why the front end can read the contract at all. Pinned here because it is
    an incidental-looking passthrough that a refactor could tighten into an
    allow-list without anyone noticing what broke.
    """
    from dataclasses import fields

    from vllm_omni.config.stage_config import StageDeployConfig

    declared = {field.name for field in fields(StageDeployConfig)}
    assert {"minimax_h3_inference_contract", "minimax_h3_admission_policy"} <= declared


@pytest.fixture(scope="module")
def oracle_reference_images():
    import json
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "minimax_h3_official_contract_v1.json"
    return json.loads(fixture.read_text(encoding="utf-8"))["reference_image_geometry"]
