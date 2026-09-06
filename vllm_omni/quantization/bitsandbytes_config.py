# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""BitsAndBytes 4-bit quantization config for diffusion transformers.

Supports online (dynamic) NF4/FP4 weight-only quantization from BF16/FP16
checkpoints on CUDA GPUs.
"""

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Module
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization._copy_missing_attrs import (
    copy_missing_attrs as _copy_missing_attrs,
)
from vllm_omni.quantization.int8_config import LazyWeightMixin

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


class DiffusionBitsAndBytesConfig(QuantizationConfig):
    """BitsAndBytes 4-bit weight-only config for diffusion transformers.

    Supports online (dynamic) quantization from BF16/FP16 checkpoints.
    Works on CUDA GPUs with the optional ``bitsandbytes`` package installed.
    """

    def __init__(
        self,
        quant_type: str = "nf4",
        # Nested absmax quantization (double quantization) is off by default
        # because it has been observed to destroy the output of wide
        # projections.  On MiniMax H3, quantizing with ``True`` makes
        # ``blocks.0.adaln_proj`` (96768 outputs) return values around 1e30
        # while ``F.linear`` on the dequantized weight returns 1.95, and the
        # request fails with "v must be finite"; flipping this flag to
        # ``False`` and changing nothing else makes the same model produce
        # correct output.  Narrower layers in the same model
        # (``final_layer.adaln_proj``, 10752 outputs) are unaffected.
        # The underlying defect has not been isolated -- it is not a device
        # mismatch (the offloader moves the nested state, see
        # ``sequential_backend._move_params``) and it is not an out-of-bounds
        # read in the fused kernel (``bnb.functional.gemv_4bit`` dequantizes
        # the nested absmax to a full-size fp32 tensor in Python before the
        # kernel ever sees it).  Until it is, the nesting is not worth its
        # price: it saves ~0.4% of the quantized weight bytes.  Pass
        # ``compress_statistics=true`` explicitly to opt in.
        compress_statistics: bool = False,
        ignored_layers: list[str] | None = None,
        ignored_layers_match: str = "exact",
    ) -> None:
        super().__init__()

        if quant_type not in ("nf4", "fp4"):
            raise ValueError(f"Unsupported quant_type {quant_type!r}; expected 'nf4' or 'fp4'")
        if ignored_layers_match not in ("exact", "substring"):
            raise ValueError(
                f"Unsupported ignored_layers_match {ignored_layers_match!r}; expected 'exact' or 'substring'"
            )
        self.quant_type = quant_type
        self.compress_statistics = compress_statistics
        self.ignored_layers = ignored_layers or []
        self.ignored_layers_match = ignored_layers_match

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "bitsandbytes"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DiffusionBitsAndBytesConfig":
        quant_type = cls.get_from_keys_or(config, ["quant_type"], "nf4")
        # Default off -- see ``__init__``.  A checkpoint that ships
        # ``compress_statistics: true`` in its quant config still gets it.
        compress_statistics = cls.get_from_keys_or(config, ["compress_statistics"], False)
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        ignored_layers_match = cls.get_from_keys_or(config, ["ignored_layers_match"], "exact")

        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(
            quant_type=quant_type,
            compress_statistics=compress_statistics,
            ignored_layers=ignored_layers,
            ignored_layers_match=ignored_layers_match,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            # ``is_layer_skipped`` defaults to whole-prefix equality, which is
            # what offline checkpoints need: their ``modules_to_not_convert``
            # lists every layer path in full. Naming a subtree instead --
            # ``["text_model"]`` to keep the whole text encoder in BF16, or
            # ``["o_proj", "down_proj"]`` to keep the accuracy-sensitive
            # projections of every block -- requires the substring matcher.
            # Opt-in, because switching the default would silently widen the
            # skip set for every checkpoint that ships an exact list.
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                # vLLM 0.28 replaced skip_with_substr with match_mode;
                # "substring" preserves skip_with_substr=True verbatim.
                match_mode="substring" if self.ignored_layers_match == "substring" else "exact",
            ):
                return UnquantizedLinearMethod()
            if current_omni_platform.is_cuda():
                return BnBOnlineLinearMethod(self)
            raise NotImplementedError("BitsAndBytes online quantization is only supported on CUDA.")
        return None


class BnBOnlineLinearMethod(LazyWeightMixin, LinearMethodBase):
    """Online BitsAndBytes 4-bit linear method.

    Loads BF16/FP16 checkpoint weights and quantizes them during loading.
    """

    def __init__(self, quant_config: DiffusionBitsAndBytesConfig):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        import bitsandbytes.functional as bnb_F

        weight = layer.weight.data.contiguous()
        if not weight.is_cuda:
            weight = weight.cuda()

        original_shape = tuple(weight.shape)
        qweight, quant_state = bnb_F.quantize_4bit(
            weight,
            quant_type=self.quant_config.quant_type,
            compress_statistics=self.quant_config.compress_statistics,
        )

        replace_parameter(
            layer,
            "weight",
            torch.nn.Parameter(qweight, requires_grad=False),
        )
        layer.quant_state = quant_state
        layer.bnb_shape = original_shape

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import bitsandbytes as bnb

        ori_shape = x.shape
        ori_dtype = x.dtype
        x_2d = x.reshape(-1, ori_shape[-1])

        out = bnb.matmul_4bit(
            x_2d,
            layer.weight.t(),
            quant_state=layer.quant_state,
            bias=bias,
        )
        return out.reshape(*ori_shape[:-1], -1).to(ori_dtype)
