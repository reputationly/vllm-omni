# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-H3 deploy-only pipeline registration.

Covers the three things that can silently break H3 deployment:
  * the registry entry stays *deploy-only* (no auto-detection regression);
  * a bare H3 checkpoint path still resolves to no pipeline;
  * the shipped YAMLs merge into a 4-card diffusion stage with the parallel
    knobs actually reaching ``parallel_config`` (flat ones would be dropped).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    StageExecutionType,
    load_deploy_config,
    merge_pipeline_deploy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PIPELINE_KEY = "minimax_h3_dit"
_MODEL_INDEX = {"_class_name": "MiniMaxH3Pipeline"}
# The production checkpoint path. Normalized it becomes "minimaxh3fl2vaint8",
# which *contains* "minimaxh3" — hence the key is not spelled "minimax_h3".
_PROD_MODEL_PATH = "/nfs-data/models/MiniMax-H3-FL2VA-INT8"
_REF2VA_MODEL_PATH = "/nfs-data/models/MiniMax-H3/Ref2VA"
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clear_factory_caches():
    yield
    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()


def _h3_checkpoint_files(filename, _model, revision=None):
    """H3 ships model_index.json only — no root config.json."""
    del revision
    return _MODEL_INDEX if filename == "model_index.json" else None


def test_registry_entry_is_deploy_only():
    assert _PIPELINE_KEY in OMNI_PIPELINES
    # Registering the bare name would make it a substring match against real
    # checkpoint directory names; see the pipeline module docstring.
    assert "minimax_h3" not in OMNI_PIPELINES

    pipeline = OMNI_PIPELINES[_PIPELINE_KEY]
    assert pipeline.hf_architectures == ()
    assert pipeline.diffusers_class_name is None
    assert pipeline.deploy_only is True
    assert pipeline.model_arch == "MiniMaxH3Pipeline"

    (stage,) = pipeline.stages
    assert stage.execution_type is StageExecutionType.DIFFUSION
    assert stage.final_output is True
    assert stage.final_output_type == "video"


def test_bare_h3_checkpoint_resolves_to_no_pipeline():
    """Launching without --deploy-config must behave as it did before #57."""
    with (
        patch.object(StageConfigFactory, "get_hf_config", return_value=None),
        patch(
            "vllm_omni.config.config_factory.get_hf_file_to_dict",
            side_effect=_h3_checkpoint_files,
        ),
    ):
        pipeline = StageConfigFactory.get_pipeline_config(
            model=_PROD_MODEL_PATH,
            trust_remote_code=True,
        )

    assert pipeline is None


def test_key_shaped_path_does_not_auto_select():
    """``deploy_only`` must beat the path-substring fallback.

    The key normalizes to "minimaxh3dit", so a checkpoint directory named like
    this one would otherwise be matched by the last resort in
    ``try_infer_model_type`` — and a hit is not harmless: it makes a bare
    ``vllm serve`` load the bundled 4-card deploy defaults.
    """
    with (
        patch.object(StageConfigFactory, "get_hf_config", return_value=None),
        patch(
            "vllm_omni.config.config_factory.get_hf_file_to_dict",
            side_effect=_h3_checkpoint_files,
        ),
    ):
        pipeline = StageConfigFactory.get_pipeline_config(
            model="/nfs-data/models/MiniMax-H3-DiT-INT8",
            trust_remote_code=True,
        )

    assert pipeline is None


def test_deploy_yaml_pipeline_field_selects_h3():
    """The only way in: the YAML names the pipeline explicitly."""
    deploy = load_deploy_config(Path(get_deploy_config_path("minimax_h3_dit.yaml")))
    assert deploy.pipeline == _PIPELINE_KEY

    with (
        patch.object(StageConfigFactory, "get_hf_config", return_value=None),
        patch(
            "vllm_omni.config.config_factory.get_hf_file_to_dict",
            side_effect=_h3_checkpoint_files,
        ),
    ):
        pipeline = StageConfigFactory.get_pipeline_config(
            model=_PROD_MODEL_PATH,
            trust_remote_code=True,
            user_deploy_config=deploy,
        )

    assert pipeline is not None
    assert pipeline.model_type == _PIPELINE_KEY


@pytest.mark.parametrize(
    ("deploy_path", "text_encoder_tp_size", "vae_patch_parallel_size"),
    [
        (Path(get_deploy_config_path("minimax_h3_dit.yaml")), 1, 1),
        (_REPO_ROOT / "deploy-configs" / "minimax_h3_a100_40g.yaml", 4, 4),
        (_REPO_ROOT / "deploy-configs" / "minimax_h3_fl2va_bf16_a100_40g.yaml", 4, 4),
        (_REPO_ROOT / "deploy-configs" / "minimax_h3_ref2va_bf16_a100_40g.yaml", 4, 4),
        (_REPO_ROOT / "deploy-configs" / "minimax_h3_ref2va_w8a8_a100_40g.yaml", 4, 4),
    ],
)
def test_shipped_deploy_configs_merge_to_four_cards(
    deploy_path: Path,
    text_encoder_tp_size: int,
    vae_patch_parallel_size: int,
):
    deploy = load_deploy_config(deploy_path)
    assert deploy.pipeline == _PIPELINE_KEY
    assert deploy.trust_remote_code is True

    (stage,) = merge_pipeline_deploy(OMNI_PIPELINES[_PIPELINE_KEY], deploy)
    assert stage.final_output is True
    assert stage.final_output_type == "video"

    engine_args = stage.yaml_engine_args
    assert engine_args["model_class_name"] == "MiniMaxH3Pipeline"

    # The nested block is what survives; a flat tensor_parallel_size would be
    # filtered out by OmniDiffusionConfig.from_kwargs and the stage would come
    # up single-card.
    parallel_config = engine_args["parallel_config"]
    assert parallel_config["tensor_parallel_size"] == 4
    assert parallel_config["ulysses_degree"] == 1
    assert parallel_config["ring_degree"] == 1
    assert parallel_config["text_encoder_tp_size"] == text_encoder_tp_size
    assert parallel_config["vae_patch_parallel_size"] == vae_patch_parallel_size
    assert parallel_config["vae_parallel_mode"] == "tile"


def test_a100_profile_carries_measured_memory_settings():
    deploy_path = _REPO_ROOT / "deploy-configs" / "minimax_h3_a100_40g.yaml"
    deploy = load_deploy_config(deploy_path)
    (stage,) = deploy.stages
    assert stage.devices == "0,1,2,3"
    assert stage.max_num_seqs == 1
    # 40 GB cards: swap in/out plus tiled VAE decode are both mandatory.
    assert stage.enable_cpu_offload is True
    assert stage.vae_use_tiling is True
    # Multi-threaded loading is what got the worker OOM-killed mid-load.
    assert stage.enable_multithread_weight_load is False
    assert stage.diffusion_attention_backend == "FLASH_ATTN"
    assert stage.diffusion_compile_granularity == "regional"
    assert stage.default_sampling_params == {"num_inference_steps": 20}


def test_ref2va_a100_profile_pins_partition_and_bf16_runtime():
    deploy_path = _REPO_ROOT / "deploy-configs" / "minimax_h3_ref2va_bf16_a100_40g.yaml"
    deploy = load_deploy_config(deploy_path)
    (stage,) = deploy.stages
    (merged_stage,) = merge_pipeline_deploy(OMNI_PIPELINES[_PIPELINE_KEY], deploy)

    assert deploy.pipeline == _PIPELINE_KEY
    assert deploy.quantization is None
    assert stage.engine_extras["task_type"] == "ref2va"
    assert merged_stage.yaml_engine_args["task_type"] == "ref2va"
    assert merged_stage.yaml_engine_args["parallel_config"]["tensor_parallel_size"] == 4
    assert stage.enable_cpu_offload is True
    assert stage.vae_use_tiling is True
    assert stage.max_num_seqs == 1
    assert stage.default_sampling_params is None
    assert stage.env == {
        "VLLM_OMNI_H3_INFERENCE_CONTRACT": "legacy",
        "VLLM_OMNI_H3_REF_IMAGE_GEOMETRY": "official_short_edge",
        "VLLM_OMNI_H3_REF_IMAGE_NO_UPSCALE": "1",
        "VLLM_OMNI_H3_REF_IMAGE_MAX_PIXELS": "1032192",
    }
    assert merged_stage.yaml_engine_args["diffusion_runtime_environ"] == stage.env

    # Transport is not enough: these values used to arrive here and then die
    # because the shape helpers re-read the worker process's bare os.environ.
    # Resolve the actual carried mapping and prove that the production profile
    # changes the geometry it was written to bound.
    from types import SimpleNamespace

    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape
    from vllm_omni.diffusion.models.minimax_h3.strategy import contract_environ, resolve_strategy

    od_config = SimpleNamespace(diffusion_runtime_environ=merged_stage.yaml_engine_args["diffusion_runtime_environ"])
    strategy = resolve_strategy(
        inference_contract=None,
        admission_policy=None,
        environ=contract_environ(od_config),
    )
    assert strategy.reference_image_no_upscale is True
    assert strategy.reference_image_max_pixels == 768 * 1344
    assert _reference_image_shape(
        Image.new("RGB", (1024, 1024)),
        aspect_ratio_range=strategy.reference_image_aspect_ratio_range,
        short_edge=strategy.reference_image_short_edge,
        no_upscale=strategy.reference_image_no_upscale,
        max_pixels=strategy.reference_image_max_pixels,
    ) == (992, 992)

    with (
        patch.object(StageConfigFactory, "get_hf_config", return_value=None),
        patch(
            "vllm_omni.config.config_factory.get_hf_file_to_dict",
            side_effect=_h3_checkpoint_files,
        ),
    ):
        pipeline = StageConfigFactory.get_pipeline_config(
            model=_REF2VA_MODEL_PATH,
            trust_remote_code=True,
            user_deploy_config=deploy,
        )

    assert pipeline is not None
    assert pipeline.model_type == _PIPELINE_KEY


def test_fl2va_bf16_profile_uses_partition_metadata_for_base_and_turbo_defaults():
    deploy_path = _REPO_ROOT / "deploy-configs" / "minimax_h3_fl2va_bf16_a100_40g.yaml"
    deploy = load_deploy_config(deploy_path)
    (stage,) = deploy.stages
    (merged_stage,) = merge_pipeline_deploy(OMNI_PIPELINES[_PIPELINE_KEY], deploy)

    assert stage.engine_extras["task_type"] == "fl2va"
    assert stage.default_sampling_params is None
    assert merged_stage.yaml_engine_args["task_type"] == "fl2va"
    assert merged_stage.yaml_engine_args["parallel_config"]["tensor_parallel_size"] == 4


def test_ref2va_w8a8_profile_uses_checkpoint_quantization_only():
    deploy_path = _REPO_ROOT / "deploy-configs" / "minimax_h3_ref2va_w8a8_a100_40g.yaml"
    deploy = load_deploy_config(deploy_path)
    (stage,) = deploy.stages
    (merged_stage,) = merge_pipeline_deploy(OMNI_PIPELINES[_PIPELINE_KEY], deploy)

    # The serialized transformer/config.json is authoritative.  A deploy-time
    # quantization flag would select the online path and re-quantize it.
    assert deploy.quantization is None
    assert stage.diffusion_quantization_config is None
    assert merged_stage.yaml_engine_args["task_type"] == "ref2va"
    assert merged_stage.yaml_engine_args["parallel_config"]["tensor_parallel_size"] == 4
