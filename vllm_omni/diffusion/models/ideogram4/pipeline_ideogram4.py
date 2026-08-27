# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ideogram 4 text-to-image pipeline for vllm-omni.

This pipeline reuses the diffusers Ideogram4 implementation for components
that do not require vllm-omni-specific integration (VAE, scheduler,
constants, latent denormalization, sigma schedule). The vllm-omni-specific
parts are:
  - the transformer, which uses vllm parallel layers for distributed inference
  - the pipeline, which plugs the transformer into vllm-omni's CFG-parallel /
    progress-bar / weight-loading infrastructure and exposes the
    `OmniDiffusionRequest` interface.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn as nn
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.transformers.transformer_ideogram4 import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
)
from transformers import AutoModel, AutoTokenizer
from transformers.masking_utils import create_causal_mask
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.ideogram4.ideogram4_transformer import (
    Ideogram4Config,
    Ideogram4Transformer,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)


# Hidden states of these Qwen3-VL decoder layers are concatenated to form the per-token
# text conditioning consumed by the Ideogram4 transformer. Mirrors
# `diffusers.pipelines.ideogram4.pipeline_ideogram4.QWEN3_VL_ACTIVATION_LAYERS`.
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
QWEN3_VL_ACTIVATION_LAYER_SET = frozenset(QWEN3_VL_ACTIVATION_LAYERS)
QWEN3_VL_LAST_ACTIVATION_LAYER = QWEN3_VL_ACTIVATION_LAYERS[-1]

# Default sampling preset used when none is supplied. The schedule is in FIRST-step
# order (index 0 corresponds to the step with the largest noise level), matching the
# diffusers convention: the first N - `POLISH_STEPS` steps use `GUIDANCE_HI` and the
# last `POLISH_STEPS` steps use `GUIDANCE_LO`. The schedule is rebuilt at call time to
# match the requested number of inference steps.
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_MU = 0.0
DEFAULT_STD = 1.5
DEFAULT_MAX_SEQUENCE_LENGTH = 2048
DEFAULT_GUIDANCE_HI = 7.0
DEFAULT_GUIDANCE_LO = 3.0
DEFAULT_POLISH_STEPS = 3

# Official named sampler presets, mirrored verbatim from the upstream harness's
# `ideogram4.sampler_configs.PRESETS` (guidance_schedule there is stored in
# LOOP-INDEX order: index 0 is the LAST/polish step. We store it here in normal
# chronological order — main steps first, polish steps last — since that is
# how this pipeline's sampling loop actually consumes it). Keyed by step count
# so a plain `num_inference_steps=20` (no explicit preset) still reproduces
# V4_DEFAULT_20 exactly instead of falling back to the generic HI/LO split
# below, which assumes DEFAULT_POLISH_STEPS=3 and would silently mismatch the
# official mu/std/polish-step-count for anything other than 48 steps.
SAMPLER_PRESETS: dict[str, dict[str, float | int | tuple[float, ...]]] = {
    "V4_QUALITY_48": {"num_steps": 48, "mu": 0.0, "std": 1.5, "polish_steps": 3},
    "V4_DEFAULT_20": {"num_steps": 20, "mu": 0.0, "std": 1.75, "polish_steps": 2},
    "V4_TURBO_12": {"num_steps": 12, "mu": 0.5, "std": 1.75, "polish_steps": 1},
}
_PRESET_BY_STEP_COUNT = {int(p["num_steps"]): name for name, p in SAMPLER_PRESETS.items()}

# Per-channel affine stats for converting the DiT's native latent scale to the VAE
# decoder's expected input scale, mirrored verbatim from the upstream harness's
# `ideogram4.latent_norm.LATENT_SHIFT`/`LATENT_SCALE`. These are NOT the same as the
# checkpoint's own `vae.bn.running_mean`/`running_var` buffer: the harness's own
# `AutoEncoder.bn` (a `BatchNorm2d(affine=False, track_running_stats=True)`) loads that
# buffer from the identical checkpoint but never reads it anywhere in `Encoder`/
# `Decoder.forward()` — grepping the harness source confirms zero other references to
# `self.bn`. The values genuinely differ (checkpoint `bn.running_mean[0]` is -0.0674 vs.
# `_LATENT_SHIFT[0]` here of +0.0198), so this is not just an alternate parameterization
# of the same statistics.
_LATENT_SHIFT: tuple[float, ...] = (
    0.01984364,
    0.10149707,
    0.29689495,
    0.27188619,
    -0.21445648,
    -0.15979549,
    0.05021099,
    -0.15083604,
    -0.15360136,
    -0.20131799,
    0.01922352,
    0.0622626,
    0.10140969,
    -0.06739428,
    0.3758261,
    -0.233712,
    0.35164491,
    -0.02590912,
    -0.0271935,
    -0.10833897,
    -0.1476848,
    -0.01130957,
    -0.2298372,
    0.23526423,
    -0.10893522,
    0.11957631,
    0.04047799,
    0.3134589,
    -0.17225064,
    -0.18646109,
    -0.34691978,
    -0.03571246,
    0.02583857,
    0.10190072,
    0.28402294,
    0.26952152,
    -0.21634675,
    -0.17938656,
    0.04358909,
    -0.15007621,
    -0.1548502,
    -0.18971131,
    0.02710861,
    0.05609494,
    0.10697846,
    -0.06854968,
    0.38167698,
    -0.24269937,
    0.35705471,
    -0.03063305,
    -0.02946109,
    -0.11244286,
    -0.14336038,
    -0.01362137,
    -0.21863696,
    0.23228983,
    -0.11739769,
    0.11693044,
    0.02563311,
    0.31356594,
    -0.17420591,
    -0.19006285,
    -0.34905377,
    -0.04025005,
    0.01924137,
    0.07652984,
    0.2995608,
    0.2628057,
    -0.22011674,
    -0.12715361,
    0.04879879,
    -0.14075719,
    -0.15935895,
    -0.2123584,
    0.01974813,
    0.05523547,
    0.10011992,
    -0.06428964,
    0.37781868,
    -0.21491644,
    0.34254215,
    -0.03153528,
    -0.0310082,
    -0.10761415,
    -0.14730405,
    -0.02475182,
    -0.2285588,
    0.2515081,
    -0.10445128,
    0.12446,
    0.07062869,
    0.30880162,
    -0.18016875,
    -0.18869164,
    -0.34533499,
    -0.0129177,
    0.02578168,
    0.07993659,
    0.28642181,
    0.26038408,
    -0.22459419,
    -0.14820155,
    0.04059549,
    -0.14043529,
    -0.16111187,
    -0.2020305,
    0.02602069,
    0.04852717,
    0.10432153,
    -0.06309942,
    0.38402443,
    -0.22397003,
    0.34814481,
    -0.03774432,
    -0.03381438,
    -0.11245691,
    -0.14128767,
    -0.02853208,
    -0.21752016,
    0.24872463,
    -0.11399775,
    0.1222687,
    0.05620835,
    0.309178,
    -0.18065738,
    -0.19401479,
    -0.34495114,
    -0.01760592,
)
_LATENT_SCALE: tuple[float, ...] = (
    1.63933691,
    1.70204478,
    1.73642566,
    1.90004803,
    1.6675316,
    1.69059584,
    1.56853198,
    1.62314944,
    1.89106626,
    1.58086668,
    1.60822129,
    1.60962993,
    1.63322129,
    1.56074359,
    1.73419528,
    1.7919265,
    1.64040632,
    1.66802808,
    1.60390303,
    1.75480492,
    1.63187587,
    1.64334594,
    1.61722884,
    1.60146046,
    1.63459219,
    1.55291476,
    1.68771497,
    1.68415657,
    1.78966054,
    1.66631641,
    1.65626686,
    1.65976433,
    1.63487607,
    1.69513249,
    1.72933756,
    1.91310663,
    1.67035057,
    1.72286863,
    1.56719251,
    1.61934825,
    1.88628859,
    1.56911539,
    1.59455129,
    1.60829869,
    1.62470611,
    1.56052853,
    1.73677003,
    1.77563606,
    1.63732541,
    1.66370527,
    1.59508952,
    1.75153949,
    1.63029275,
    1.64517667,
    1.61659342,
    1.59722044,
    1.64103121,
    1.5408531,
    1.68610394,
    1.67772755,
    1.78998563,
    1.66621713,
    1.65458955,
    1.66041308,
    1.64710857,
    1.68163503,
    1.74000294,
    1.92784786,
    1.67411194,
    1.67395548,
    1.57406532,
    1.62199356,
    1.87618195,
    1.5584375,
    1.57438785,
    1.61711053,
    1.63094305,
    1.55644029,
    1.73124302,
    1.80666627,
    1.6463621,
    1.65932006,
    1.60816188,
    1.75682671,
    1.64695873,
    1.63121722,
    1.61380832,
    1.60478651,
    1.63396035,
    1.53505068,
    1.65534289,
    1.67132281,
    1.80317197,
    1.6767314,
    1.65700938,
    1.68426259,
    1.65339716,
    1.67540638,
    1.73298504,
    1.94067348,
    1.67893609,
    1.70635117,
    1.5730906,
    1.61928553,
    1.87148809,
    1.56244866,
    1.56697152,
    1.61584394,
    1.62759496,
    1.55480378,
    1.73484107,
    1.79055143,
    1.64688773,
    1.66121492,
    1.60135887,
    1.75254572,
    1.64798332,
    1.62989921,
    1.61381592,
    1.60792883,
    1.63939668,
    1.53075757,
    1.65371318,
    1.66801185,
    1.80029087,
    1.67591476,
    1.65655173,
    1.68533454,
)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def get_ideogram4_post_process_func(od_config: OmniDiffusionConfig):
    """Build a post-process function that converts decoded VAE outputs to PIL images."""
    if od_config.output_type == "latent":
        return lambda x: x

    image_processor = VaeImageProcessor(vae_scale_factor=8 * 2)

    def post_process_func(output: torch.Tensor):
        if isinstance(output, torch.Tensor):
            images = image_processor.postprocess(output.float(), output_type="pil")
            return images
        return output

    return post_process_func


def _logit_normal_sigmas(
    num_inference_steps: int,
    mu: float,
    std: float = 1.0,
    logsnr_min: float = -15.0,
    logsnr_max: float = 18.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a length-`num_inference_steps` sigma schedule using the Ideogram4 logit-normal flow-matching schedule."""
    intervals = torch.linspace(0.0, 1.0, num_inference_steps + 1, dtype=torch.float64)
    z = torch.special.ndtri(intervals)
    y = mu + std * z
    t = 1.0 - torch.special.expit(y)
    t_min = 1.0 / (1.0 + math.exp(0.5 * logsnr_max))
    t_max = 1.0 / (1.0 + math.exp(0.5 * logsnr_min))
    t = t.clamp(t_min, t_max)
    sigmas = (1.0 - t).flip(0)
    sigmas = sigmas[:-1].to(dtype=torch.float32, device=device)
    return sigmas


_MIN_SIDE = 256
_MAX_SIDE = 2048
_MAX_ASPECT_RATIO = 6.0


def _validate_resolution(height: int, width: int, multiple: int) -> tuple[int, int]:
    """Round/clamp ``height``/``width`` into Ideogram4's documented envelope.

    Per the upstream harness's ``docs/inference.md`` "Supported Resolutions":
    multiples of 16, 256-2048 per side, aspect ratio up to 6:1. Values outside
    this envelope are not a hard error (mirrors Krea2Pipeline's own
    round-and-warn convention in this codebase, `pipeline_krea2.py`) because a
    caller passing e.g. a bare aspect-ratio-like string ("16x9" parsed as
    literal pixels) should get a small-but-valid image back, not a crash —
    before this validation existed, a too-small size rounded down to 0 and hit
    ``math.log(0)`` in ``_resolution_aware_mu`` with an opaque "math domain
    error". Rounds UP (not down, unlike the truncation this replaces) so a
    request never rounds all the way to 0.
    """
    orig_height, orig_width = height, width

    def _round_up(value: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    height = min(max(_round_up(height), _MIN_SIDE), _MAX_SIDE)
    width = min(max(_round_up(width), _MIN_SIDE), _MAX_SIDE)

    if height > width * _MAX_ASPECT_RATIO:
        width = min(_MAX_SIDE, _round_up(math.ceil(height / _MAX_ASPECT_RATIO)))
    elif width > height * _MAX_ASPECT_RATIO:
        height = min(_MAX_SIDE, _round_up(math.ceil(width / _MAX_ASPECT_RATIO)))

    if (height, width) != (orig_height, orig_width):
        logger.warning(
            "Requested size %dx%d is outside Ideogram4's supported envelope "
            "(multiples of %d, %d-%d per side, aspect ratio up to %g:1); "
            "adjusted to %dx%d.",
            orig_width,
            orig_height,
            multiple,
            _MIN_SIDE,
            _MAX_SIDE,
            _MAX_ASPECT_RATIO,
            width,
            height,
        )
    return height, width


def _resolution_aware_mu(
    height: int,
    width: int,
    base_mu: float,
    base_resolution: tuple[int, int] = (512, 512),
) -> float:
    """Shift the schedule mean as a function of image resolution."""
    num_pixels = height * width
    base_pixels = base_resolution[0] * base_resolution[1]
    return base_mu + 0.5 * math.log(num_pixels / base_pixels)


class Ideogram4Pipeline(
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    """Ideogram 4 text-to-image pipeline (vllm-omni).

    Uses asymmetric CFG with two separate transformer instances:
    - conditional_transformer: processes text + image tokens
    - unconditional_transformer: processes image tokens only (zero text features)
    """

    supports_step_execution: ClassVar[bool] = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()

        model = od_config.model
        local_files_only = os.path.isdir(model)
        # Use the configured runtime dtype for HF models that do not have a quant config.
        # Quantized models (e.g. fp8/nvfp4) are loaded in their native dtype by the weight
        # loader; in that case we let `from_pretrained` keep the checkpoint's dtype.
        runtime_dtype = od_config.dtype

        # Weight sources for the two transformers and the VAE.
        # The transformer weights are loaded by the vllm-omni loader (it maps diffusers
        # parameter names to our parallel-layer parameter names). VAE weights are loaded
        # directly via the diffusers API in `__init__`.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=None,
                prefix="conditional_transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="unconditional_transformer",
                revision=None,
                prefix="unconditional_transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="vae",
                revision=None,
                prefix="vae.",
                fall_back_to_pt=True,
            ),
        ]

        # NOTE: this pipeline's own `_preprocess_nf4_weights` (ideogram4_transformer.py)
        # already dequantizes the checkpoint's packed NF4 tensors to BF16 in Python
        # before `load_weights` ever sees them, one weight at a time. So the disk
        # checkpoint's own "already offline-quantized" declaration is irrelevant here
        # by the time a weight reaches a Linear layer's weight_loader — every incoming
        # tensor is plain BF16. What actually needs a quant_config is the ONLINE path
        # (`--quantization bitsandbytes` -> DiffusionBitsAndBytesConfig): it re-quantizes
        # that incoming BF16 tensor down to 4-bit at load time, so the layer's resident
        # GPU parameter ends up packed instead of full BF16. Without it (quant_config=None),
        # the layer just keeps the full BF16 tensor — an 18.6B-param dual-DiT in BF16 is
        # ~37GB and OOMs a 40G card before the text encoder/VAE even load.
        transformer_config = Ideogram4Config()
        self.conditional_transformer = Ideogram4Transformer(
            od_config=od_config,
            config=transformer_config,
            quant_config=od_config.quantization_config,
        )
        self.unconditional_transformer = Ideogram4Transformer(
            od_config=od_config,
            config=transformer_config,
            quant_config=od_config.quantization_config,
        )

        self.text_encoder = AutoModel.from_pretrained(
            model,
            subfolder="text_encoder",
            local_files_only=local_files_only,
            trust_remote_code=True,
            torch_dtype=runtime_dtype,
        )
        visual_owner = None
        if hasattr(self.text_encoder, "model") and hasattr(self.text_encoder.model, "visual"):
            visual_owner = self.text_encoder.model
        elif hasattr(self.text_encoder, "visual"):
            visual_owner = self.text_encoder
        if visual_owner is not None:
            del visual_owner.visual
        self.text_encoder = self.text_encoder.to(self.device)
        self.text_encoder.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )

        # VAE (diffusers). Latent denormalization is performed via its batch-norm running stats.
        # Use the configured runtime dtype for unquantized VAE to be consistent with the rest
        # of the pipeline.
        if od_config.quantization_config is None:
            self.vae = AutoencoderKLFlux2.from_pretrained(
                model,
                subfolder="vae",
                local_files_only=local_files_only,
                torch_dtype=runtime_dtype,
            ).to(self.device)
        else:
            self.vae = AutoencoderKLFlux2.from_pretrained(model, subfolder="vae", local_files_only=local_files_only).to(
                self.device
            )
        self.vae.eval()

        # Scheduler (diffusers). We override the sigma schedule every call with a
        # resolution-aware logit-normal schedule, so the per-component config is only used
        # for `num_train_timesteps`.
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        self.patch_size = 2
        self.ae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self.vae.config, "block_out_channels", None)
            else 8
        )
        self.vae_scale_factor = self.ae_scale_factor
        self.max_text_tokens = DEFAULT_MAX_SEQUENCE_LENGTH
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * self.patch_size)

        # Defer full device placement to the weight-loading / offload path in three cases:
        # 1. Quantization: vLLM linear layers live on the meta device until the weight
        #    loader materializes them. Calling .to(device) would fail on those meta tensors,
        #    so we skip it and let the weight loader handle device placement.
        # 2. Layerwise offload: modules should be initialized on CPU first, then
        #    selectively materialized/moved by the offloader.
        # 3. HSDP: weights should be loaded on CPU first and sharded afterwards,
        #    rather than eagerly placing the full model on one GPU.
        if od_config.quantization_config is None and not (
            od_config.enable_layerwise_offload or od_config.parallel_config.use_hsdp
        ):
            self.to(self.device)

        # Optional cache-dit / TeaCache / MagCache wiring. Returns ``None`` when
        # Ideogram4 has two separate transformers (conditional + unconditional), so
        # the standard single-transformer Cache-DiT enabler doesn't apply out of the
        # box. No dedicated dual-transformer enabler exists yet, so Cache-DiT is a
        # no-op here until one is written (this mirrors what the upstream PR's own
        # ``may_enable_cache_dit`` call would have returned for this pipeline anyway).
        self._cache_backend = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, int]:
        """Build chat-formatted token ids for a single prompt."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = encoded["input_ids"][0]
        num_text_tokens = int(token_ids.shape[0])

        if num_text_tokens > self.max_text_tokens:
            raise ValueError(f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={self.max_text_tokens}")
        return token_ids, num_text_tokens

    def _build_inputs(
        self,
        prompts: list[str],
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        """Build the packed sequence (text tokens + image tokens) for one batch."""
        tokenized = [self._tokenize(p) for p in prompts]
        batch_size = len(prompts)

        patch = self.patch_size * self.ae_scale_factor
        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"height/width must be divisible by patch_size*ae_scale_factor={patch}")
        grid_h = height // patch
        grid_w = width // patch
        num_image_tokens = grid_h * grid_w

        max_text_tokens = max(num_text for _, num_text in tokenized)
        total_seq_len = max_text_tokens + num_image_tokens

        h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        token_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
        text_position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
        indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

        for b, (toks, num_text) in enumerate(tokenized):
            pad_len = max_text_tokens - num_text
            total_unpadded = num_text + num_image_tokens

            offset = pad_len
            token_ids[b, offset : offset + num_text] = toks

            text_pos = torch.arange(num_text)
            text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
            text_position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset : offset + num_text] = text_pos_3d
            position_ids[b, offset + num_text :] = image_pos

            indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
            indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

            segment_ids[b, offset : offset + total_unpadded] = 1

        return {
            "token_ids": token_ids.to(self.device),
            "text_position_ids": text_position_ids.to(self.device),
            "position_ids": position_ids.to(self.device),
            "segment_ids": segment_ids.to(self.device),
            "indicator": indicator.to(self.device),
            "num_image_tokens": num_image_tokens,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "max_text_tokens": max_text_tokens,
        }

    def _get_qwen3_vl_embeddings(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_2d: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run Qwen3-VL and capture hidden states from activation layers."""
        language_model = self.text_encoder.language_model

        inputs_embeds = language_model.embed_tokens(token_ids)

        position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
        text_position_ids = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

        captured: list[torch.Tensor] = []
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in QWEN3_VL_ACTIVATION_LAYER_SET:
                captured.append(hidden_states)
            if layer_idx >= QWEN3_VL_LAST_ACTIVATION_LAYER:
                break

        return captured

    def _encode_text(
        self,
        token_ids: torch.Tensor,
        text_position_ids: torch.Tensor,
        indicator: torch.Tensor,
    ) -> torch.Tensor:
        """Run Qwen3-VL and stack hidden states from the activation layers.

        Returns a (B, L, hidden_size * num_layers) float32 tensor.
        """
        batch_size, seq_len = token_ids.shape

        attention_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)
        pos_2d = text_position_ids[..., 0].contiguous()

        with torch.inference_mode():
            selected = self._get_qwen3_vl_embeddings(token_ids, attention_mask, pos_2d)
        stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
        stacked = torch.permute(stacked, (1, 2, 3, 0))
        stacked = stacked.reshape(batch_size, seq_len, -1)

        text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
        stacked = stacked * text_mask
        return stacked.to(torch.float32)

    def _decode(
        self,
        z: torch.Tensor,
        *,
        grid_h: int,
        grid_w: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Denormalize latents (via the harness's fixed per-channel stats) and decode through the VAE."""
        # Latent denormalization: fixed per-channel shift/scale on the packed-channel
        # latent space (ae_channels * patch ** 2), matching the harness's
        # `Ideogram4Pipeline._decode` exactly. NOT derived from the checkpoint's own
        # `vae.bn` buffer — see the `_LATENT_SHIFT`/`_LATENT_SCALE` module docstring.
        latent_shift = torch.tensor(_LATENT_SHIFT, dtype=z.dtype, device=z.device).view(1, 1, -1)
        latent_scale = torch.tensor(_LATENT_SCALE, dtype=z.dtype, device=z.device).view(1, 1, -1)
        z = z * latent_scale + latent_shift

        patch = self.patch_size
        ae_channels = z.shape[-1] // (patch * patch)
        z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels)
        z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
        z = z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)

        z = z.to(self.vae.dtype)
        decoded = self.vae.decode(z, return_dict=False)[0]
        return decoded.float().clamp(-1.0, 1.0)

    def predict_noise(
        self,
        *,
        transformer: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        llm_features: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
        max_text_tokens: int = 0,
    ) -> torch.Tensor:
        """Run one CFG branch and return the image-only velocity tensor.

        ``predict_noise_maybe_with_cfg`` (from ``CFGParallelMixin``) calls this once
        per rank with either the positive or negative kwargs. Slicing off the text
        prefix happens here so the mixin can call the standard ``combine_cfg_noise``
        to produce a CFG-merged prediction of shape ``[B, num_image_tokens, latent]``.

        Args:
            transformer: The branch-specific transformer module
                (``conditional_transformer`` for positive, ``unconditional_transformer``
                for negative). This dispatch happens via kwargs so the same method works
                under both sequential and CFG-parallel execution.
            x: Full packed sequence (text + image for the positive branch, image-only
                for the negative branch).
            t: Model-time in [0, 1] broadcast to batch size.
            llm_features: Text features for the positive branch, zero image-shape
                tensor for the negative branch.
            position_ids/segment_ids/indicator: MRoPE position ids, segment ids, and
                token-type indicator aligned to ``x``.
            max_text_tokens: Length of the text prefix in ``x``. When ``> 0`` the
                returned velocity is sliced to image tokens only.
        """
        v = transformer(
            llm_features=llm_features,
            x=x,
            t=t,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
        )
        if max_text_tokens > 0:
            v = v[:, max_text_tokens:]
        return v

    def forward(
        self,
        req: OmniDiffusionRequest,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        guidance_schedule: tuple[float, ...] | list[float] | None = None,
        mu: float | None = None,
        std: float | None = None,
        output_type: str = "pil",
    ) -> DiffusionOutput:
        """Generate images for the given request.

        Args:
            req: The diffusion request containing prompts and sampling params.
            height: Image height (must be divisible by patch_size * ae_scale_factor).
            width: Image width.
            num_inference_steps: Number of denoising steps (default 50).
            guidance_scale: Constant CFG scale; if set, overrides `guidance_schedule`.
            seed: Random seed.
            guidance_schedule: Per-step guidance weight, in first-step order (index 0
                corresponds to the step with the largest noise level). If unset, the
                default schedule is used: 7.0 for the first ``num_steps - 3`` steps and
                3.0 for the last 3 "polish" steps.
            mu: Base mean of the logit-normal schedule (default 0.0).
            std: Std of the logit-normal schedule (default 1.5).
            output_type: Output type ("pil" or "latent").
        """
        prompts = req.prompts or []
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts]
        if not prompts:
            raise ValueError("No prompts provided")

        height = height or req.sampling_params.height or 1024
        width = width or req.sampling_params.width or 1024
        patch = self.patch_size * self.ae_scale_factor
        height, width = _validate_resolution(height, width, patch)

        # `sampler_preset` (e.g. "V4_TURBO_12") is not a declared SamplingParams
        # field, so — like `skip_first_n_steps` above — it only reaches us via
        # `extra_params` in the request body, not a top-level JSON key (see
        # memory: non-declared engine params get silently dropped by Pydantic
        # otherwise). Selects the official mu/std/polish-step-count combo so a
        # caller doesn't have to know those numbers themselves.
        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        preset_name = extra_args.get("sampler_preset")
        preset = SAMPLER_PRESETS.get(preset_name) if preset_name else None
        if preset_name and preset is None:
            raise ValueError(f"Unknown sampler_preset {preset_name!r}; choices: {sorted(SAMPLER_PRESETS)}")

        num_steps = (
            req.sampling_params.num_inference_steps
            if req.sampling_params.num_inference_steps is not None
            else num_inference_steps
        ) or (preset["num_steps"] if preset else DEFAULT_NUM_INFERENCE_STEPS)

        # Auto-detect a preset from the step count alone when the caller didn't
        # explicitly name one — e.g. `num_inference_steps=20` should reproduce
        # V4_DEFAULT_20's mu/std/polish split exactly, not silently fall back to
        # V4_QUALITY_48's numbers (DEFAULT_MU/STD/POLISH_STEPS) just because 20
        # isn't 48.
        if preset is None:
            preset = SAMPLER_PRESETS.get(_PRESET_BY_STEP_COUNT.get(num_steps, ""))

        mu = (preset["mu"] if preset else DEFAULT_MU) if mu is None else mu
        std = (preset["std"] if preset else DEFAULT_STD) if std is None else std

        # Resolve guidance schedule.
        if guidance_scale is not None:
            effective_schedule: tuple[float, ...] = (float(guidance_scale),) * num_steps
        elif guidance_schedule is not None:
            schedule = tuple(guidance_schedule)
            if len(schedule) != num_steps:
                # Adjust to match num_steps by truncating or extending with the last value.
                if num_steps < len(schedule):
                    schedule = schedule[:num_steps]
                else:
                    schedule = schedule + (schedule[-1],) * (num_steps - len(schedule))
            effective_schedule = schedule
        else:
            # Default: `DEFAULT_GUIDANCE_HI` for the first `num_steps - polish`
            # steps, then `DEFAULT_GUIDANCE_LO` for the last `polish` steps. Uses
            # the matched preset's own polish-step count when one applies,
            # instead of always assuming DEFAULT_POLISH_STEPS=3.
            polish = min(int(preset["polish_steps"]) if preset else DEFAULT_POLISH_STEPS, num_steps)
            effective_schedule = (DEFAULT_GUIDANCE_HI,) * (num_steps - polish) + (DEFAULT_GUIDANCE_LO,) * polish
        gw_per_step = torch.tensor(effective_schedule, dtype=torch.float32, device=self.device)

        inputs = self._build_inputs(prompts, height=height, width=width)

        batch_size = len(prompts)
        num_image_tokens = inputs["num_image_tokens"]
        grid_h, grid_w = inputs["grid_h"], inputs["grid_w"]
        max_text_tokens = inputs["max_text_tokens"]
        latent_dim = self.conditional_transformer.config.in_channels

        llm_features = self._encode_text(inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"])

        neg_position_ids = inputs["position_ids"][:, max_text_tokens:]
        neg_segment_ids = inputs["segment_ids"][:, max_text_tokens:]
        neg_indicator = inputs["indicator"][:, max_text_tokens:]

        neg_llm_features = torch.zeros(
            batch_size,
            num_image_tokens,
            llm_features.shape[-1],
            dtype=llm_features.dtype,
            device=self.device,
        )

        # `DiffusionEngine`/`DiffusionModelRunner` invokes `pipeline.forward(req)` with
        # only the request — none of this method's other keyword parameters (height,
        # seed, etc.) are ever populated by a caller, so a bare `if seed is not None`
        # check here was always false and every generation silently drew from the
        # process's global unseeded RNG regardless of what the caller requested.
        # `DiffusionModelRunner._initialize_generator` already builds
        # `req.sampling_params.generator` from `req.sampling_params.seed` before
        # `forward()` runs; every other pipeline in this repo reads one of those two
        # fields (see e.g. `HeliosPipeline.forward`) instead of relying on a `seed`
        # kwarg from the caller.
        generator = req.sampling_params.generator
        if generator is None:
            seed = seed if seed is not None else req.sampling_params.seed
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
        z = torch.randn(
            batch_size,
            num_image_tokens,
            latent_dim,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )

        text_z_padding = torch.zeros(
            batch_size,
            max_text_tokens,
            latent_dim,
            dtype=torch.float32,
            device=self.device,
        )

        # DIAGNOSTIC ONLY (not a real fix): skip this model's own forward pass for
        # the first `skip_first_n_steps` entries of a FINER schedule — z is left as
        # untouched Gaussian noise and fed directly into the first non-skipped
        # timestep. This is numerically wrong (that timestep expects a partially
        # denoised input, not raw noise) so image fidelity for those steps is not
        # representative; it exists purely to test the community claim that the
        # baked-in safety trigger fires specifically off this model's OWN first
        # forward call, independent of what produced the input latent. A real fix
        # would hand off from a second model's genuine step-1 output instead of
        # skipping the compute outright.
        #
        # The schedule is built at `num_steps + skip_first_n_steps` resolution and
        # then the leading entries are dropped, so the number of REAL transformer
        # calls still equals the caller's requested `num_inference_steps` — skipping
        # steps doesn't quietly shrink the sampling budget the caller asked for.
        skip_first_n_steps = int(extra_args.get("skip_first_n_steps", 0))

        # Build the resolution-aware logit-normal sigma schedule on the scheduler.
        schedule_mu = _resolution_aware_mu(height=height, width=width, base_mu=mu)
        sigmas = _logit_normal_sigmas(num_steps + skip_first_n_steps, schedule_mu, std=std, device=self.device)
        self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=self.device)
        timesteps = self.scheduler.timesteps[skip_first_n_steps:]
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        with self.progress_bar(total=num_steps) as pbar:
            for i, t in enumerate(timesteps):
                # Map sigma-domain timestep to model time `t_model` in [0, 1] (0 = noise, 1 = clean data).
                t_model = 1.0 - (t.float() / num_train_timesteps)
                t_model = t_model.expand(batch_size).to(
                    getattr(self.conditional_transformer.input_proj, "compute_dtype", None)
                    or self.conditional_transformer.input_proj.weight.dtype
                )

                positive_kwargs = {
                    "transformer": self.conditional_transformer,
                    "llm_features": llm_features,
                    "x": torch.cat([text_z_padding, z], dim=1),
                    "t": t_model,
                    "position_ids": inputs["position_ids"],
                    "segment_ids": inputs["segment_ids"],
                    "indicator": inputs["indicator"],
                    "max_text_tokens": max_text_tokens,
                }
                negative_kwargs = {
                    "transformer": self.unconditional_transformer,
                    "llm_features": neg_llm_features,
                    "x": z,
                    "t": t_model,
                    "position_ids": neg_position_ids,
                    "segment_ids": neg_segment_ids,
                    "indicator": neg_indicator,
                    "max_text_tokens": 0,
                }

                # Delegate to the standard CFG-parallel mixin. It dispatches:
                #   - cfg_parallel_size > 1: each rank runs one branch only
                #   - sequential: runs both branches, then combines with CFG
                # In both cases the returned tensor has shape
                # ``[B, num_image_tokens, latent]``, matching the scheduler's expectation.
                step_cfg = float(gw_per_step[i].item())
                v = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=not math.isclose(step_cfg, 1.0),
                    true_cfg_scale=step_cfg,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                # Scheduler step: the diffusers scheduler expects -velocity (see diffusers' `__call__`).
                z = self.scheduler.step(-v, t, z, return_dict=False)[0]

                pbar.update()

        decoded = self._decode(z, grid_h=grid_h, grid_w=grid_w, batch_size=batch_size)

        return DiffusionOutput(
            output=decoded,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
