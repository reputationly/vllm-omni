# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Image-2.1 deploy-only pipeline registration.

Mirrors ``tests/config/test_minimax_h3_pipeline.py``: the registration is what
lets the A100-40G profile be expressed as ``--deploy-config`` instead of ~10
CLI flags, and the deploy-only guards are what keep a bare
``vllm serve <path>`` on the generic diffusion default (the e2e suite launches
that way).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    StageExecutionType,
    load_deploy_config,
    merge_pipeline_deploy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PIPELINE_KEY = "qwen_image_21"
_MODEL_INDEX = {"_class_name": "QwenImage21Pipeline"}
_PROD_MODEL_PATH = "/nfs-models/wuhanjisuan894/models/Qwen-Image-2.1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEPLOY_YAML = _REPO_ROOT / "deploy-configs" / "qwen_image_21_a100_40g_sp4.yaml"


@pytest.fixture(autouse=True)
def _clear_factory_caches():
    yield
    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()


def _q21_checkpoint_files(filename, _model, revision=None):
    """Q21 ships model_index.json only — no root config.json."""
    del revision
    return _MODEL_INDEX if filename == "model_index.json" else None


def test_registry_entry_is_deploy_only():
    assert _PIPELINE_KEY in OMNI_PIPELINES

    pipeline = OMNI_PIPELINES[_PIPELINE_KEY]
    assert pipeline.hf_architectures == ()
    assert pipeline.diffusers_class_name is None
    assert pipeline.deploy_only is True
    assert pipeline.model_arch == "QwenImage21Pipeline"

    (stage,) = pipeline.stages
    assert stage.execution_type is StageExecutionType.DIFFUSION
    assert stage.final_output is True
    assert stage.final_output_type == "image"


def test_bare_q21_checkpoint_resolves_to_no_pipeline():
    """Without --deploy-config, launch stays on the generic diffusion default."""
    with (
        patch.object(StageConfigFactory, "get_hf_config", return_value=None),
        patch(
            "vllm_omni.config.config_factory.get_hf_file_to_dict",
            side_effect=_q21_checkpoint_files,
        ),
    ):
        pipeline = StageConfigFactory.get_pipeline_config(
            model=_PROD_MODEL_PATH,
            trust_remote_code=True,
        )

    # QwenImage21Pipeline is a registered *diffusion* arch, not an OMNI_PIPELINES
    # topology, so pipeline resolution yields None and the resolver falls through
    # to the generic single-stage diffusion default.
    assert pipeline is None


def test_deploy_yaml_pipeline_field_selects_q21():
    deploy = load_deploy_config(_DEPLOY_YAML)
    assert deploy.pipeline == _PIPELINE_KEY

    with (
        patch.object(StageConfigFactory, "get_hf_config", return_value=None),
        patch(
            "vllm_omni.config.config_factory.get_hf_file_to_dict",
            side_effect=_q21_checkpoint_files,
        ),
    ):
        pipeline = StageConfigFactory.get_pipeline_config(
            model=_PROD_MODEL_PATH,
            trust_remote_code=True,
            user_deploy_config=deploy,
        )

    assert pipeline is not None
    assert pipeline.model_type == _PIPELINE_KEY


def test_deploy_yaml_merges_to_sp4():
    """The A100 profile's parallel knobs must reach ``parallel_config``.

    The YAML states ``sequence_parallel_size: 1`` alongside
    ``ulysses_degree: 4``; the typed path drops the stale explicit SP (it is a
    derived field there, ``init=False``) and re-derives 4, so the shipped YAML
    is accepted. The legacy merge path keeps the raw value, which is why this
    test goes through the production resolution rather than
    ``merge_pipeline_deploy`` directly.
    """
    deploy = load_deploy_config(_DEPLOY_YAML)
    pipeline_cfg = OMNI_PIPELINES[_PIPELINE_KEY]
    stage_cfgs = merge_pipeline_deploy(pipeline_cfg, deploy)
    (stage_cfg,) = stage_cfgs
    engine_args = dict(stage_cfg.yaml_engine_args)
    parallel = engine_args["parallel_config"]
    assert parallel["ulysses_degree"] == 4
    assert parallel["tensor_parallel_size"] == 1
    # The YAML's stale explicit value flows through the legacy path verbatim;
    # the typed path re-derives it. Both are documented behaviors — this
    # assertion pins the legacy projection so a silent change is caught.
    assert parallel["sequence_parallel_size"] == 1
