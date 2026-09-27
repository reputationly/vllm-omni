# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Image-2.1 deploy-only pipeline topology.

Keep this topology outside the ``qwen_image_21`` package: the package
``__init__`` is lazy-by-necessity (see its docstring — eager imports there
create a circular import with the distributed VAE), and importing a submodule
executes that ``__init__``. Same placement discipline as
``pi0_pipeline_config.py``.

``qwen_image_21`` exists so a deploy YAML can name a topology with
``pipeline: qwen_image_21``. Without it, ``--deploy-config`` is refused
outright (``resolver.py`` raises "not registered to OMNI_PIPELINES"), and the
A100-40G profile — Ulysses SP4, W8A16, fp8 prefix KV, VAE tiling — has to be
repeated as ~10 CLI flags. Same rationale as ``minimax_h3_dit``; see
``vllm_omni/model_executor/models/minimax_h3/pipeline.py`` for the full
argument.

Why deploy-only, same shape as the H3/HunyuanImage3 entries:

* ``hf_architectures=()`` and no ``diffusers_class_name``, so neither the HF
  config path nor the ``model_index.json`` path in ``try_infer_model_type``
  can select it. The only way in is a deploy YAML's ``pipeline:`` field.
* ``deploy_only=True`` keeps the path-basename fallback from matching it, so a
  bare ``vllm serve <path>`` keeps resolving through the generic diffusion
  default exactly as before — the e2e suite (``test_qwen_image_21.py``)
  launches that way.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# The diffusers pipeline class, as registered in
# ``vllm_omni/diffusion/registry.py`` and named by the checkpoint's
# ``model_index.json._class_name``. ``_build_engine_args`` copies it into
# ``model_class_name`` for diffusion stages.
_QWEN_IMAGE_21_MODEL_ARCH = "QwenImage21Pipeline"

QWEN_IMAGE_21_PIPELINE = PipelineConfig(
    model_type="qwen_image_21",
    model_arch=_QWEN_IMAGE_21_MODEL_ARCH,
    hf_architectures=(),
    deploy_only=True,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            # t2i and image-conditioned editing are one deployment; whether a
            # request carries images decides the path, not the topology.
            final_output_type="image",
            requires_multimodal_data=True,
            model_arch=_QWEN_IMAGE_21_MODEL_ARCH,
        ),
    ),
)
