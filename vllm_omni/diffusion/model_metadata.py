# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiffusionModelMetadata:
    # Keep serving-facing capability metadata in a lightweight shared module so
    # config/model plumbing can read it without importing concrete pipelines.
    supports_multimodal_inputs: bool = False
    max_multimodal_image_inputs: int | None = None
    supports_mixed_reference_inputs: bool = False
    attention_mask_free: bool = False
    # Whether a reference image is put onto the generated canvas before it
    # reaches the pipeline. True is the historical serving behaviour and stays
    # the default; a model whose reference images carry a geometry of their own
    # sets it False, or the canvas silently rewrites the reference's aspect
    # ratio. See `reference_images_bind_output_canvas` below, which lets a
    # startup-level contract override it for models that have one.
    reference_images_bind_output_canvas: bool = True
    final_output_type: str | None = None


QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES = 4
# Upstream HunyuanImage-3.0 "Multi-Image Fusion" caps reference images at 3.
HUNYUAN_IMAGE3_MAX_INPUT_IMAGES = 3
# Boogu-Image editing (TI2I) supports a single reference image for now.
BOOGU_IMAGE_MAX_INPUT_IMAGES = 1
# SenseNova-U1 / U1.5 img2img takes several reference images: the pipeline emits
# the upstream ``Image-1:<image>`` ... prefix and splits a pixel budget across
# them (``_prepare_input_images``). Upstream states no maximum -- its reference
# harness takes ``--image`` with ``nargs="+"`` and never validates the count --
# so this is purely a serving-side admission limit. Nine leaves each reference
# ~1365x1365 of the budget, well above the 512x512 min_pixels floor; the largest
# example in upstream's u1.5 best-practices doc uses five.
SENSENOVA_U1_MAX_INPUT_IMAGES = 9


_DIFFUSION_MODEL_METADATA: dict[str, DiffusionModelMetadata] = {
    "QwenImageEditPlusPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES,
    ),
    "HunyuanImage3Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=HUNYUAN_IMAGE3_MAX_INPUT_IMAGES,
    ),
    # Shared by the Base (text-to-image) and Edit (TI2I) checkpoints, which use
    # the same ``BooguImagePipeline`` class. Text-to-image requests simply carry
    # no reference image.
    "BooguImagePipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=BOOGU_IMAGE_MAX_INPUT_IMAGES,
    ),
    "MiniMaxH3Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=9,
        supports_mixed_reference_inputs=True,
        final_output_type="video",
        # H3 represents alignment padding as a second packed sequence.  The
        # packed TRTLLM backend consumes cu_seqlens and isolates that padding.
        attention_mask_free=True,
    ),
    # The modular alias is served by MiniMaxH3Pipeline and has the same
    # Ref2VA request contract. Keep admission limits in sync with it.
    "MiniMaxH3ModularPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=9,
        supports_mixed_reference_inputs=True,
        final_output_type="video",
    ),
    # Shared by U1 and U1.5 (both resolve from ``model_type == "neo_chat"``).
    # Without an entry here the serving layer read the dataclass default as "one
    # input image maximum" and rejected every multi-reference edit at the HTTP
    # boundary, before the pipeline's multi-image path could run.
    "SenseNovaU1Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=SENSENOVA_U1_MAX_INPUT_IMAGES,
    ),
    "Magi2Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=1,
        final_output_type="video",
    ),
    "WanPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
    ),
    "WanImageToVideoPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
    ),
    "WanVACEPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
    ),
    "WanS2VPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
    ),
    "WanT2VDMD2Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "WanI2VDMD2Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "LTX2Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "LTX2DistilledPipeline": DiffusionModelMetadata(final_output_type="video"),
    "LTX2T2VDMD2Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "LTX2I2VDMD2Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "HeliosPipeline": DiffusionModelMetadata(final_output_type="video"),
    "HeliosPyramidPipeline": DiffusionModelMetadata(final_output_type="video"),
    "HunyuanVideo15Pipeline": DiffusionModelMetadata(final_output_type="video"),
    "HunyuanVideo15ImageToVideoPipeline": DiffusionModelMetadata(final_output_type="video"),
    "LingBotVideoPipeline": DiffusionModelMetadata(final_output_type="video"),
    "LongCatVideoAvatarPipeline": DiffusionModelMetadata(final_output_type="video"),
    "MagiHumanPipeline": DiffusionModelMetadata(final_output_type="video"),
    "DreamIDOmniPipeline": DiffusionModelMetadata(final_output_type="video"),
    "Cosmos3OmniDiffusersPipeline": DiffusionModelMetadata(final_output_type="video"),
    "Cosmos3OmniPipeline": DiffusionModelMetadata(final_output_type="video"),
    "SanaVideoPipeline": DiffusionModelMetadata(final_output_type="video"),
    "SanaImageToVideoPipeline": DiffusionModelMetadata(final_output_type="video"),
    "SanaWmPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=1,
    ),
}

_DIFFUSION_MODEL_METADATA_ALIASES = {
    "WanDMDPipeline": "WanPipeline",
    "LTX2TwoStagePipeline": "LTX2Pipeline",
    "LTX2DistilledOneStagePipeline": "LTX2DistilledPipeline",
    "LTX2DistilledTwoStagePipeline": "LTX2DistilledPipeline",
    "LingBotWorldCausalDMDPipeline": "LingBotVideoPipeline",
}


def get_diffusion_model_metadata(model_class_name: str | None) -> DiffusionModelMetadata:
    # Unknown models fall back to "no special multimodal capabilities" so new
    # pipelines do not accidentally inherit limits meant for other models.
    if model_class_name is None:
        return DiffusionModelMetadata()
    metadata = _DIFFUSION_MODEL_METADATA.get(model_class_name)
    if metadata is not None:
        return metadata
    canonical_name = _DIFFUSION_MODEL_METADATA_ALIASES.get(model_class_name)
    if canonical_name is not None:
        return _DIFFUSION_MODEL_METADATA[canonical_name]
    # Some checkpoints report the HF architecture name diff from internal pipeline class name
    # (e.g. HunyuanImage3ForCausalMM, WanVACEPipeline, OmniVoice ...).
    from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

    entry = _DIFFUSION_MODELS.get(model_class_name)
    if entry is not None:
        # Unpack instead of indexing so a future change to the registry tuple
        # shape fails loudly instead of silently reading the wrong element.
        _, _, pipeline_cls_name = entry
        # Note: the registry ``cls_name`` and the metadata keys are two separate
        # key spaces. Aliases whose pipeline class has no metadata entry (e.g.
        # Wan22VACEPipeline, OmniVoicePipeline) still fall back to the defaults;
        # that is not a regression, it just means no capability override.
        canonical_name = _DIFFUSION_MODEL_METADATA_ALIASES.get(pipeline_cls_name, pipeline_cls_name)
        return _DIFFUSION_MODEL_METADATA.get(canonical_name, DiffusionModelMetadata())
    return DiffusionModelMetadata()


_MINIMAX_H3_PIPELINES = ("MiniMaxH3Pipeline", "MiniMaxH3ModularPipeline")


def _minimax_h3_strategy(od_config: object | None) -> Any | None:
    """The contract this instance serves, as the serving layer can see it.

    ``None`` when the contract cannot be resolved. Every caller treats that as
    "answer what this model has always answered": an unusable contract fails
    loudly where it is *applied*, in the pipeline, with the pipeline's own
    message. Masking it here would replace a startup diagnostic with a 400 that
    blames the caller's references.

    Resolved from exactly what the pipeline resolves from — config fields plus
    ``contract_environ``, which is where the stage-scoped environment the
    serving process never had is laid back on top. Anything less makes this
    answer for a contract the worker is not running.
    """
    # Imported lazily: the serving layer must not pull a model package in just
    # to answer a capability question for a model it is not serving.
    from vllm_omni.diffusion.models.minimax_h3.strategy import contract_environ, resolve_strategy

    try:
        return resolve_strategy(
            inference_contract=getattr(od_config, "minimax_h3_inference_contract", None),
            admission_policy=getattr(od_config, "minimax_h3_admission_policy", None),
            environ=contract_environ(od_config),
        )
    except ValueError:
        return None


def reference_images_bind_output_canvas(
    model_class_name: str | None,
    od_config: object | None = None,
    *,
    task_type: str | None = None,
) -> bool:
    """Whether the serving layer should put reference images on the canvas.

    Default True, which is what every model has always done. MiniMax-H3 answers
    False for every aspect-preserving H3 policy: official short-edge, Base's
    fixed-area ceiling, and Turbo's ``match``. Stretching a reference onto the
    canvas would destroy any of those policies before the model applies it.

    The contract is resolved from the same startup inputs the pipeline reads, so
    this answers identically in an inline or an out-of-process deployment.

    Args:
        model_class_name: The resolved pipeline class name.
        od_config: The diffusion config, when the caller has one.
        task_type: The request's resolved image role when the serving layer can
            distinguish it. This matters for a combined H3 instance: FL2VA
            keyframes and Ref2VA references have independent resize policies.

    Returns:
        Whether reference images are bound to the generated canvas.
    """
    metadata = get_diffusion_model_metadata(model_class_name)
    if not metadata.reference_images_bind_output_canvas:
        return False
    if model_class_name not in _MINIMAX_H3_PIPELINES:
        return True

    strategy = _minimax_h3_strategy(od_config)
    if strategy is None:
        return True
    if str(task_type or "").lower() == "fl2va" and strategy.fl2va_keyframe_resize_mode == "official_cover_crop":
        # The follower must reach ``prepare_fl2va_keyframes`` with its source
        # aspect ratio intact. A route-level stretch makes it equal the output
        # canvas and turns the official cover-crop into a no-op.
        return False
    return strategy.reference_image_geometry_mode == "legacy_canvas_prestretch"


def honours_explicit_reference_order(model_class_name: str | None, od_config: object | None = None) -> bool:
    """Whether this instance can serve a caller-ordered heterogeneous reference list.

    The second capability probe of its kind, and it exists because the first one
    was not enough: an instance's contract is decided at startup, in the worker,
    while the decision to *build* an order is taken at the HTTP boundary, in a
    process that may not be the same one. ``/v1/tasks/video/`` derives
    ``reference_order`` from every non-empty ``references`` list, and a
    ``legacy`` pipeline rejects any explicit order outright — so without this,
    the most ordinary request on the DEFAULT deployment returns 200 PENDING,
    takes a queue slot, and fails inside the job.

    Default True so no other model is affected: the field only carries meaning
    where a pipeline reads it.

    Args:
        model_class_name: The resolved pipeline class name.
        od_config: The diffusion config, when the caller has one.

    Returns:
        Whether an explicit reference order will be honoured rather than rejected.
    """
    if model_class_name not in _MINIMAX_H3_PIPELINES:
        return True

    strategy = _minimax_h3_strategy(od_config)
    if strategy is None:
        return True
    return strategy.reference_order_mode == "ordered_references"


def reference_video_decode_frame_cap(
    model_class_name: str | None,
    od_config: object | None = None,
    *,
    num_frames: int | None,
) -> int | None:
    """How many frames of a reference video the HTTP layer may keep.

    The third capability probe of its kind, and the reason it is a probe rather
    than the ``reference_video_decode_spec`` classmethod next to it is that the
    classmethod is handed ``(num_frames, extra_args)`` — a *request*, with no
    config — so it cannot be contract-aware even in principle. A model class
    answers for the model; only a config answers for the instance.

    The number that matters is the one the pipeline will target, and that is the
    frame count *after* ``17 * n + 5`` alignment. Decoding to the requested
    count instead leaves the reference short of the target, and the official
    encoder then snaps its VAE input DOWN to the previous multiple: 120 frames
    kept for a 124-frame clip is encoded as 107, one whole 17-frame chunk of
    conditioning less than the reference actually carried.

    Legacy keeps the requested count, unchanged. It aligns too, but it fills the
    target by repeating frame slots rather than truncating, so the loss is a
    slot-mapping shift rather than a dropped chunk — and changing what legacy
    decodes would change what production generates, which this effort does not
    do.

    Args:
        model_class_name: The resolved pipeline class name.
        od_config: The diffusion config, when the caller has one.
        num_frames: The frame count the request asked for, or ``None``.

    Returns:
        The frame cap to decode with, or ``None`` for "keep everything".
    """
    if num_frames is None or model_class_name not in _MINIMAX_H3_PIPELINES:
        return num_frames

    strategy = _minimax_h3_strategy(od_config)
    if strategy is None or not strategy.is_official:
        return num_frames

    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_align_frame_count

    return minimax_h3_align_frame_count(int(num_frames))
