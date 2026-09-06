# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch import nn
from vllm.config import ModelConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_world_size,
)
from vllm.lora.utils import is_moe_model
from vllm.model_executor.model_loader import register_model_loader

try:
    # vLLM 0.28 moved BitsAndBytes out of tree (vllm#43529) to get its branches
    # out of the shared weight-loading path; the code itself lives on in the
    # official vllm-project/vllm-bnb-plugin, which also absorbed ParamMapping.
    # Import from both so one image can be validated against either base.
    from vllm_bnb_plugin.bitsandbytes_loader import (
        BitsAndBytesModelLoader,
        ParamMapping,
    )
except ModuleNotFoundError:  # vLLM <= 0.26
    from vllm.model_executor.model_loader.bitsandbytes_loader import (
        BitsAndBytesModelLoader,
    )
    from vllm.model_executor.model_loader.utils import ParamMapping

from vllm.model_executor.models import is_pooling_model
from vllm.model_executor.utils import (
    get_moe_expert_mapping,
    get_packed_modules_mapping,
)


@register_model_loader("hunyuan_image3_bitsandbytes")
class HunyuanImage3BitsAndBytesModelLoader(BitsAndBytesModelLoader):
    """Load HunyuanImage-3.0 pre-quantized NF4 experts with TP+EP.

    The stock vLLM loader rejects every pre-quantized BitsAndBytes model
    under TP because a generic packed tensor cannot safely be split. With
    Hunyuan TP4+EP4, however, RoutedExperts owns 16 complete experts per
    rank. No packed expert is split: the loader filters global experts
    through ``expert_map`` and fuses only the local QuantStates.
    """

    _SUPPORTED_ARCHITECTURES = {
        "HunyuanImage3ForCausalMM",
        "HunyuanImage3ForConditionalGeneration",
    }

    def _verify_model_compatibility(
        self,
        model: nn.Module,
        model_config: ModelConfig,
    ) -> None:
        architectures = set(model_config.hf_config.architectures or ())
        if not architectures.intersection(self._SUPPORTED_ARCHITECTURES):
            raise ValueError(
                "hunyuan_image3_bitsandbytes is restricted to "
                f"HunyuanImage-3.0, got architectures={sorted(architectures)}"
            )
        if not hasattr(model, "load_weights"):
            raise AttributeError(f"{type(model).__name__} must define load_weights().")
        if not hasattr(model, "packed_modules_mapping"):
            raise AttributeError(f"{type(model).__name__} has no packed_modules_mapping.")

        quant_config = getattr(
            model_config.hf_config,
            "quantization_config",
            None,
        )
        quant_method = (quant_config or {}).get("quant_method")
        if quant_method != "bitsandbytes":
            raise ValueError(
                f"hunyuan_image3_bitsandbytes requires a pre-quantized BitsAndBytes checkpoint, got {quant_method!r}."
            )
        self.pre_quant = True
        self.load_8bit = bool(quant_config.get("load_in_8bit", False))
        if self.load_8bit:
            raise ValueError(
                "HunyuanImage-3.0 RoutedExperts currently supports only pre-quantized BitsAndBytes 4-bit checkpoints."
            )
        if get_tensor_model_parallel_world_size() > 1:
            if get_ep_group().world_size <= 1:
                raise ValueError(
                    "HunyuanImage-3.0 pre-quantized NF4 with TP>1 requires "
                    "enable_expert_parallel=True so packed experts remain "
                    "whole and are distributed by expert_map."
                )

    def _initialize_loader_state(
        self,
        model: nn.Module,
        model_config: ModelConfig,
    ) -> None:
        self.is_pool_model = is_pooling_model(model)
        self.modules_mapping = ParamMapping(get_packed_modules_mapping(model))

        if is_moe_model(model):
            raw_mapping = get_moe_expert_mapping(model)
            # HunyuanModel also returns a checkpoint remapping dictionary
            # used by its own load_weights(). The generic vLLM helper's type
            # contract is only the first list.
            if isinstance(raw_mapping, tuple) and len(raw_mapping) == 2 and isinstance(raw_mapping[0], list):
                self.expert_params_mapping = raw_mapping[0]
            else:
                self.expert_params_mapping = raw_mapping

        if hf_to_vllm_mapper := getattr(model, "hf_to_vllm_mapper", None):
            unstacked_mapper = hf_to_vllm_mapper.get_unstacked_mapper()
            self.weight_mapper = lambda name, mapper=unstacked_mapper: mapper._map_name(name)

        self._get_bnb_target_modules(model)
        self._classify_module_sharding(model)

    @staticmethod
    def _local_expert_pairs(module: nn.Module) -> list[tuple[int, int]]:
        expert_map = getattr(module, "expert_map", None)
        if expert_map is None:
            raise ValueError("HunyuanImage-3.0 NF4 TP loading requires RoutedExperts.expert_map.")
        pairs = [
            (int(local_id), global_id) for global_id, local_id in enumerate(expert_map.tolist()) if int(local_id) >= 0
        ]
        return sorted(pairs)

    @staticmethod
    def _swap_gate_up_absmax(quant_state: Any) -> Any:
        # Checkpoint gate_and_up_proj is logical [up, gate], while vLLM w13
        # is [gate, up]. Packed bytes are split/reordered by model.load_weights;
        # the per-block scales must be reordered identically.
        if quant_state.absmax.numel() % 2:
            raise ValueError(
                f"gate_and_up_proj absmax must contain two equal halves, got {quant_state.absmax.numel()} elements."
            )
        half = quant_state.absmax.numel() // 2
        quant_state.absmax = torch.cat((quant_state.absmax[half:], quant_state.absmax[:half]))
        return quant_state

    def _fuse_hunyuan_expert_quant_states(
        self,
        model: nn.Module,
        quant_states_dict: dict[str, Any],
    ) -> dict[str, Any]:
        from bitsandbytes.functional import QuantState
        from vllm.model_executor.layers.fused_moe import RoutedExperts

        fused: dict[str, Any] = {}
        for module_name, module in model.named_modules():
            if not isinstance(module, RoutedExperts):
                continue

            checkpoint_prefix = module_name.removesuffix(".routed_experts")
            local_gate_up: list[Any] = []
            local_down: list[Any] = []
            for expected_local_id, global_id in self._local_expert_pairs(module):
                if expected_local_id != len(local_gate_up):
                    raise ValueError(
                        f"Non-contiguous local expert map for {module_name}: "
                        f"expected {len(local_gate_up)}, got {expected_local_id}."
                    )
                gate_up_name = f"{checkpoint_prefix}.{global_id}.gate_and_up_proj.weight"
                down_name = f"{checkpoint_prefix}.{global_id}.down_proj.weight"
                try:
                    gate_up_state = quant_states_dict.pop(gate_up_name)
                    down_state = quant_states_dict.pop(down_name)
                except KeyError as exc:
                    raise KeyError(f"Missing local Hunyuan NF4 QuantState {exc.args[0]!r}.") from exc

                gate_up_state = self._dequantize_dq(gate_up_state)
                down_state = self._dequantize_dq(down_state)
                local_gate_up.append(self._swap_gate_up_absmax(gate_up_state))
                local_down.append(down_state)

            if not local_gate_up:
                raise ValueError(f"No local experts found for {module_name}.")

            gate_up0 = local_gate_up[0]
            down0 = local_down[0]
            fused[f"{module_name}.w13_weight"] = QuantState(
                absmax=torch.cat([state.absmax for state in local_gate_up]),
                shape=(
                    len(local_gate_up) * gate_up0.shape[0],
                    gate_up0.shape[1],
                ),
                code=gate_up0.code,
                blocksize=gate_up0.blocksize,
                quant_type="nf4",
                dtype=gate_up0.dtype,
            )
            fused[f"{module_name}.w2_weight"] = QuantState(
                absmax=torch.cat([state.absmax for state in local_down]),
                shape=(
                    len(local_down) * down0.shape[0],
                    down0.shape[1],
                ),
                code=down0.code,
                blocksize=down0.blocksize,
                quant_type="nf4",
                dtype=down0.dtype,
            )
        return fused

    def load_weights(
        self,
        model: nn.Module,
        model_config: ModelConfig,
    ) -> None:
        self._verify_model_compatibility(model, model_config)
        self._initialize_loader_state(model, model_config)

        qweight_iterator, quant_state_dict = self._get_quantized_weights_iterator(
            model_config.model,
            model_config.revision,
        )
        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(qweight_iterator)
        if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(f"Following weights were not initialized from checkpoint: {weights_not_loaded}")

        expert_states = self._fuse_hunyuan_expert_quant_states(
            model,
            quant_state_dict,
        )
        stacked_states = self._stack_quantization_states(
            model,
            quant_state_dict,
        )
        self._bind_quant_states_to_params(
            model,
            {**expert_states, **stacked_states},
        )
        torch.accelerator.empty_cache()
