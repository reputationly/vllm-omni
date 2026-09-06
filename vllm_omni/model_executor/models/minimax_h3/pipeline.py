# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 pipeline topologies.

Two entries, for two different jobs:

* ``MINIMAX_H3_PIPELINE`` (``minimax_h3_disaggregated``) -- upstream's two-stage
  topology that splits the text encoder into its own AR stage.
* ``MINIMAX_H3_DIT_PIPELINE`` (``minimax_h3_dit``) -- a single-stage, deploy-only
  registration. Every deploy-config in production is single-stage, so the
  disaggregated topology cannot serve them; this entry is what lets those YAMLs
  be merged at all. Its rationale is preserved below verbatim.
"""

# MiniMax-H3 pipeline topology (deploy-only).
# 
# H3 ships as a pure diffusers repo: no root ``config.json``, only a
# ``model_index.json`` naming ``MiniMaxH3Pipeline``. Without a registered
# ``PipelineConfig`` a deploy YAML has no topology to merge into, so
# ``--deploy-config`` is refused outright (``entrypoints/utils.py``) and every
# multi-GPU knob has to be repeated as a CLI flag — ~15 of them for the 4-card
# A100-40G profile.
# 
# Why this is a *deploy-only* key, same shape as ``hunyuan_image3_ar`` /
# ``hunyuan_image3_dit``:
# 
# * ``hf_architectures=()`` and no ``diffusers_class_name``, so neither the HF
#   config path nor the ``model_index.json`` path in ``try_infer_model_type``
#   can select it. The only way in is naming it from a deploy YAML's
#   ``pipeline:`` field, which ``_get_deploy_override_pipe_config`` resolves at
#   the highest priority.
# * ``deploy_only=True`` closes the last resort. That fallback matches registry
#   keys as substrings of the whole normalized model path, and a hit is not
#   harmless: ``_create_legacy_from_registry`` would then load this pipeline's
#   ``default_deploy_config_name`` and silently impose the bundled 4-card
#   defaults on a bare ``vllm serve <path>``. The flag makes the fallback skip
#   the entry outright, so no directory name can pull it in.
# * The key is ``minimax_h3_dit``, not ``minimax_h3`` — belt and braces rather
#   than the primary guard. ``minimax_h3`` normalizes to ``minimaxh3``, which
#   *is* a substring of ``minimaxh3fl2vaint8``; the ``_dit`` suffix keeps the
#   normalized key out of the checkpoint names actually in use, so the naming
#   alone would already spare
#   ``vllm serve /nfs-data/models/MiniMax-H3-FL2VA-INT8``. It is only naming
#   discipline, though — a future ``MiniMax-H3-DiT-*`` directory would match —
#   which is exactly why the flag above, not this bullet, is what guarantees it.
# 
# Net effect: launching without ``--deploy-config`` behaves exactly as before.

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.minimax_h3"
_CHECKPOINT = "vllm_omni.model_executor.models.minimax_h3.checkpoint"
_DIFFUSION_PIPELINE = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3"

MINIMAX_H3_PIPELINE = PipelineConfig(
    model_type="minimax_h3_disaggregated",
    default_deploy_config_name="minimax_h3_disaggregated.yaml",
    stage_cli_aliases={"text_encoder_tp_size": (0, "tensor_parallel_size")},
    model_arch="MiniMaxH3TextEncoder",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="text_encoder",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            requires_multimodal_data=True,
            model_arch="MiniMaxH3TextEncoder",
            engine_output_type="latent",
            prompt_transform_func=f"{_PROCESSOR}.prepare_text_encoder_prompt",
            sampling_constraints={
                "max_tokens": 1,
                "temperature": 0.0,
                "detokenize": False,
            },
            model_path_resolver=f"{_CHECKPOINT}.resolve_minimax_h3_model_root",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="video",
            requires_multimodal_data=True,
            model_arch="MiniMaxH3Pipeline",
            custom_process_input_func=f"{_PROCESSOR}.text_encoder2diffusion",
            omni_kv_config={"need_recv_cache": False},
            model_path_resolver=f"{_DIFFUSION_PIPELINE}.resolve_minimax_h3_diffusion_model_path",
            inline_diffusion=True,
        ),
    ),
)


# The diffusers pipeline class, as registered in
# ``vllm_omni/diffusion/registry.py`` and named by the checkpoint's
# ``model_index.json._class_name``. ``_build_engine_args`` copies it into
# ``model_class_name`` for diffusion stages, so this resolves to the same
# class the auto-detected path would have picked.
_MINIMAX_H3_MODEL_ARCH = "MiniMaxH3Pipeline"


MINIMAX_H3_DIT_PIPELINE = PipelineConfig(
    model_type="minimax_h3_dit",
    default_deploy_config_name="minimax_h3_dit.yaml",
    model_arch=_MINIMAX_H3_MODEL_ARCH,
    hf_architectures=(),
    deploy_only=True,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            # H3 emits joint video+audio, muxed into one MP4 by the video
            # serving path; "video" is the modality the endpoint selects on.
            final_output_type="video",
            requires_multimodal_data=True,
            model_arch=_MINIMAX_H3_MODEL_ARCH,
        ),
    ),
)
