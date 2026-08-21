# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_grouped_qkv_checkpoint_reorder():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        _reorder_grouped_qkv_to_qkv,
    )

    # Two groups with rows [q, k, v] become [q0, q1, k0, k1, v0, v1].
    grouped = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    reordered = _reorder_grouped_qkv_to_qkv(
        grouped,
        num_query_groups=2,
        heads_per_group=1,
        head_dim=1,
    )

    assert reordered[:, 0].tolist() == [0, 3, 1, 4, 2, 5]


def test_pruned_diffusers_config_aliases_and_timestep_interpolation():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3TimeEmbedder,
    )

    arch = MiniMaxH3DiTArchConfig.from_mapping(
        {
            "hidden_size": 8,
            "num_refiner_layers": 3,
            "ffn_dim": 16,
            "in_channels": 2,
            "audio_in_channels": 4,
            "freq_dim": 6,
            "time_embed_hidden_dim": 12,
            "rope_freq_dim": 5,
            "time_embed_dim": 10,
            "adaln_rank": 2,
            "time_table_size": 3,
        }
    )
    assert arch.token_refiner_num_layers == 3
    assert arch.ffn_hidden_size == 16
    assert arch.latents_dim == 2
    assert arch.audio_latents_dim == 4
    assert arch.timestep_input_dim == 6
    assert arch.time_embed_hidden_size == 12
    assert arch.rope_inv_freq_len == 5
    assert arch.adaln_out_features == 18 * 8
    assert arch.final_adaln_out_features == 2 * 8

    embedder = MiniMaxH3TimeEmbedder(arch, prefix="time_embedder")
    embedder.table.copy_(torch.tensor([[0.0, 2.0], [10.0, 12.0], [20.0, 22.0]]))
    actual = embedder(torch.tensor([-1.0, 0.25, 1.0, 2.0]))
    expected = torch.tensor([[0.0, 2.0], [5.0, 7.0], [20.0, 22.0], [20.0, 22.0]])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_diffusers_name_mapping_covers_pruned_adaln_and_split_qkv():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        _diffusers_qkv_target,
        _diffusers_to_partition_name,
    )

    assert _diffusers_to_partition_name("transformer_blocks.7.adaln_proj.folded_bias") == (
        "blocks.7.adaln_proj.folded_bias"
    )
    assert _diffusers_to_partition_name("norm_out.folded_bias") == "final_layer.adaln_proj.folded_bias"
    assert _diffusers_qkv_target("token_refiner.refiner_blocks.1.attn.to_k.weight") == (
        "token_refiner.blocks.1.attn.qkv_proj.weight",
        "k",
    )


def test_pruned_adaln_skips_silu_and_adds_folded_bias_in_fp32():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3AdalnProj

    class CaptureLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = None

        def forward(self, value):
            self.input = value.clone()
            return torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16), None

    projection = object.__new__(MiniMaxH3AdalnProj)
    nn.Module.__init__(projection)
    projection.expand_ratio = 2
    projection.modality_num = 1
    projection.hidden_size = 1
    projection.pruned = True
    projection.linear = CaptureLinear()
    projection.register_buffer("folded_bias", torch.tensor([0.0039, 0.0079], dtype=torch.float32))

    timestep_coordinates = torch.tensor([[2.0, -3.0]], dtype=torch.float32)
    shift, scale = projection(timestep_coordinates)

    # The table is already activated, so its coordinates reach the projection
    # directly rather than through SiLU.
    torch.testing.assert_close(
        projection.linear.input,
        timestep_coordinates.to(torch.bfloat16),
        atol=0,
        rtol=0,
    )
    expected = (torch.tensor([[1.0, 2.0]]).float() + projection.folded_bias).to(torch.bfloat16)
    torch.testing.assert_close(torch.cat((shift, scale), dim=-1), expected, atol=0, rtol=0)


def test_qkv_checkpoint_loader_reorders_serialized_channel_scales():
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    captured: dict[str, torch.Tensor] = {}

    def parameter_with_loader(name: str, shape: tuple[int, ...]) -> nn.Parameter:
        parameter = nn.Parameter(torch.empty(shape), requires_grad=False)

        def loader(_param, value):
            captured[name] = value.clone()

        parameter.weight_loader = loader
        return parameter

    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(num_attention_heads=2, attention_head_dim=1)
    block = nn.Module()
    block.attn = nn.Module()
    block.attn.qkv_proj = nn.Module()
    block.attn.qkv_proj.register_parameter("weight", parameter_with_loader("weight", (6, 2)))
    block.attn.qkv_proj.register_parameter("weight_scale", parameter_with_loader("scale", (6, 1)))
    model.blocks = nn.ModuleList([block])

    grouped_weight = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    grouped_scale = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    model.load_weights(
        [
            ("blocks.0.attn.qkv_proj.weight", grouped_weight),
            ("blocks.0.attn.qkv_proj.weight_scale", grouped_scale),
        ]
    )

    assert captured["weight"][:, 0].tolist() == [0, 6, 2, 8, 4, 10]
    assert captured["scale"][:, 0].tolist() == [0, 3, 1, 4, 2, 5]


def test_transformer_declares_cache_sp_layerwise_offload_and_hsdp():
    from cache_dit import ForwardPattern

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    assert MiniMaxH3DiTModel._repeated_blocks == ["MiniMaxH3DiTBlock"]
    assert MiniMaxH3DiTModel._layerwise_offload_blocks_attrs == ["blocks"]
    assert MiniMaxH3DiTModel._cache_dit_adapter_config.block_forward_patterns["blocks"] == ForwardPattern.Pattern_3
    assert not MiniMaxH3DiTModel._cache_dit_adapter_config.has_separate_cfg
    assert set(MiniMaxH3DiTModel._sp_plan) == {"sp_prepare", "sp_gather"}

    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.token_refiner = nn.Module()
    model.token_refiner.blocks = nn.ModuleList([nn.Linear(4, 4)])
    model.final_layer = nn.Linear(4, 4)

    matched = [
        name
        for name, module in model.named_modules()
        if any(condition(name, module) for condition in MiniMaxH3DiTModel._hsdp_shard_conditions)
    ]
    assert matched == ["blocks.0", "blocks.1"]


def test_packed_attention_is_a_regional_compile_boundary():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    assert getattr(MiniMaxH3Attention._run_packed_attention, "_torchdynamo_disable", False)


def test_h3_fused_rope_matches_reference_and_preserves_unrotated_dims():
    from vllm_omni.diffusion.layers.rope import RotaryEmbedding
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    attention = object.__new__(MiniMaxH3Attention)
    nn.Module.__init__(attention)
    attention.rope = RotaryEmbedding(is_neox_style=True, half_head_dim=False)
    attention.rope._forward_method = attention.rope.forward_native

    x = torch.randn(11, 3, 128, dtype=torch.bfloat16)
    freqs_half = torch.randn(11, 48)
    freqs = torch.cat((freqs_half, freqs_half), dim=-1)
    actual = attention._apply_rope(x, freqs)

    cos = torch.cos(freqs).to(x.dtype).unsqueeze(1)
    sin = torch.sin(freqs).to(x.dtype).unsqueeze(1)
    x_rot = x[..., :96]
    x1, x2 = x_rot.chunk(2, dim=-1)
    expected_rot = x_rot * cos + torch.cat((-x2, x1), dim=-1) * sin
    expected = torch.cat((expected_rot, x[..., 96:]), dim=-1)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[..., 96:], x[..., 96:], atol=0, rtol=0)


@pytest.mark.parametrize(
    ("tp_size", "message"),
    [
        (3, "num_attention_heads"),
        (5, "num_attention_heads"),
    ],
)
def test_tp_rejects_non_divisible_head_counts(tp_size, message):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    with pytest.raises(ValueError, match=message):
        model._validate_tp_config(
            arch=MiniMaxH3DiTArchConfig(),
            tp_size=tp_size,
        )


def test_tp_accepts_checkpoint_supported_sizes():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    arch = MiniMaxH3DiTArchConfig()
    for tp_size in (1, 2, 4, 7):
        model._validate_tp_config(arch=arch, tp_size=tp_size)


def test_rope_inv_freq_is_initialized_without_a_checkpoint():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Rope

    inv_freq_len = 16
    rope = MiniMaxH3Rope(inv_freq_len)

    # The buffer is persistent, so a released checkpoint overwrites it. A
    # checkpoint that omits it (a pruned export, say) must still get the
    # reference curve rather than whatever ``torch.empty`` left behind.
    rot_dim = inv_freq_len * 2
    expected = 1.0 / (10000.0 ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))
    torch.testing.assert_close(rope.inv_freq, expected, atol=0, rtol=0)
    assert rope.inv_freq.dtype is torch.float32
    assert torch.isfinite(rope.inv_freq).all()


def test_swiglu_half_order_keys_off_the_source_name():
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    def build_model() -> tuple[MiniMaxH3DiTModel, dict[int, torch.Tensor]]:
        captured: dict[int, torch.Tensor] = {}
        parameter = nn.Parameter(torch.empty((4, 1)), requires_grad=False)

        def loader(_param, value, shard_id):
            captured[shard_id] = value.clone()

        parameter.weight_loader = loader
        model = object.__new__(MiniMaxH3DiTModel)
        nn.Module.__init__(model)
        model.arch = SimpleNamespace(num_attention_heads=2, attention_head_dim=1, adaln_rank=None)
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.fc1 = nn.Module()
        block.mlp.fc1.register_parameter("weight", parameter)
        model.blocks = nn.ModuleList([block])
        return model, captured

    packed = torch.tensor([[0.0], [1.0], [2.0], [3.0]])

    # Released partition naming: rows are already [gate, up].
    model, captured = build_model()
    model.load_weights([("blocks.0.mlp.fc1.weight", packed)])
    assert captured[0][:, 0].tolist() == [0.0, 1.0]
    assert captured[1][:, 0].tolist() == [2.0, 3.0]

    # Diffusers naming: rows are [up, gate] and must be swapped. A wrong order
    # here loads and runs, so the decision may not rest on the rename table.
    model, captured = build_model()
    model.load_weights([("transformer_blocks.0.ff.net.0.proj.weight", packed)])
    assert captured[0][:, 0].tolist() == [2.0, 3.0]
    assert captured[1][:, 0].tolist() == [0.0, 1.0]


def test_pruned_buffer_check_only_runs_when_load_weights_did():
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(adaln_rank=8, num_layers=0)

    # Weights arrived by another route (mmap under DLO+AllGather skips
    # load_weights but still calls this hook): nothing to verify, no error.
    # Both shapes of "load_weights did not run here" must be tolerated: the
    # attribute left at its declared None, and an object built without
    # __init__ that never got the declaration at all.
    assert not hasattr(model, "_loaded_pruned_buffers")
    model.post_load_weights()
    model._loaded_pruned_buffers = None
    model.post_load_weights()

    # load_weights ran and the checkpoint was short: that is a real failure.
    model._loaded_pruned_buffers = {"time_embedder.table"}
    with pytest.raises(ValueError, match="missing required FP32 buffers"):
        model.post_load_weights()
