"""Int8 storage for the MiniMax-H3 Qwen3-VL text encoder.

The failure this guards against is silent: a scale placed by a different rule
than the rows it scales still produces a running model and plausible video, so
the checks here compare against the BF16 result rather than against "it ran".
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.quantization.tools.quantize_qwen3vl_encoder_int8 import (
    classify,
    quantize_per_output_channel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Group:
    """Encoder process-group stand-in; TP sharding is exercised by rank/size."""

    def __init__(self, rank: int = 0, world_size: int = 1) -> None:
        self.rank_in_group = rank
        self.world_size = world_size

    def all_reduce(self, tensor: torch.Tensor) -> None:  # pragma: no cover - single rank
        return None


def _encoder_module(name: str):
    from vllm_omni.diffusion.models.minimax_h3 import encoder as enc

    return getattr(enc, name)


def test_quantize_round_trip_is_close_and_exact_on_scale_rows():
    weight = torch.randn(8, 16, dtype=torch.float32)
    q, scale = quantize_per_output_channel(weight)
    assert q.dtype == torch.int8
    assert scale.shape == (8, 1) and scale.dtype == torch.float32
    restored = q.to(torch.float32) * scale
    # Per-row max error is half a step of that row's own scale.
    assert torch.all((restored - weight).abs() <= scale / 2 + 1e-6)


def test_quantize_survives_a_dead_output_channel():
    weight = torch.zeros(3, 4, dtype=torch.float32)
    weight[1] = 1.0
    q, scale = quantize_per_output_channel(weight)
    assert torch.all(scale > 0)  # no division by zero for the all-zero rows
    assert torch.equal((q.to(torch.float32) * scale)[0], torch.zeros(4))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.language_model.layers.0.self_attn.q_proj.weight", "quantize"),
        ("model.language_model.layers.49.mlp.down_proj.weight", "quantize"),
        ("model.language_model.layers.50.mlp.down_proj.weight", "drop"),
        ("model.language_model.layers.0.input_layernorm.weight", "copy"),
        ("model.language_model.layers.0.self_attn.q_norm.weight", "copy"),
        ("model.language_model.embed_tokens.weight", "copy"),
        ("model.visual.blocks.0.attn.qkv.weight", "copy"),
        ("lm_head.weight", "drop"),
    ],
)
def test_classify_matches_what_the_encoder_actually_builds(name, expected):
    assert classify(name, 50) == expected


def test_classify_keeps_everything_when_layers_are_not_dropped():
    assert classify("model.language_model.layers.63.mlp.up_proj.weight", None) == "quantize"
    assert classify("lm_head.weight", None) == "drop"  # still never built


def _load_int8(module, weight: torch.Tensor, shard_id=None) -> None:
    q, scale = quantize_per_output_channel(weight)
    module.enable_int8()
    module.weight.weight_loader(module.weight, q, shard_id)
    module.weight_scale.weight_loader(module.weight_scale, scale, shard_id)


def test_row_parallel_int8_matches_bf16_and_replicates_its_scale():
    cls = _encoder_module("MiniMaxH3Qwen3VLRowParallelLinear")
    torch.manual_seed(0)
    weight = torch.randn(6, 8, dtype=torch.float32)

    reference = cls(_Group(), input_size=8, output_size=6, dtype=torch.float32)
    reference.weight.data.copy_(weight)
    quantized = cls(_Group(), input_size=8, output_size=6, dtype=torch.float32)
    _load_int8(quantized, weight)

    assert quantized.weight.dtype == torch.int8
    assert quantized.weight_scale.shape == (6, 1)
    x = torch.randn(4, 8)
    assert torch.allclose(quantized(x), reference(x), atol=2e-2, rtol=2e-2)


def test_qkv_int8_scale_follows_the_rows_it_scales_under_tp():
    cls = _encoder_module("MiniMaxH3Qwen3VLQKVParallelLinear")
    torch.manual_seed(0)
    head_dim, num_heads, num_kv, hidden = 4, 4, 2, 8
    q_full = torch.randn(num_heads * head_dim, hidden)

    # Rank 1 of a 2-way split must end up with the *second* half of q's rows and
    # the second half of q's scales — mismatching them is the silent failure.
    module = cls(
        _Group(rank=1, world_size=2),
        hidden_size=hidden,
        num_heads=num_heads,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        dtype=torch.float32,
    )
    _load_int8(module, q_full, "q")

    q_local = module.local_num_heads * head_dim
    expected_q, expected_scale = quantize_per_output_channel(q_full)
    assert torch.equal(module.weight[:q_local], expected_q[q_local : 2 * q_local])
    assert torch.equal(module.weight_scale[:q_local], expected_scale[q_local : 2 * q_local])


def test_merged_column_int8_places_gate_and_up_in_their_own_halves():
    cls = _encoder_module("MiniMaxH3Qwen3VLMergedColumnParallelLinear")
    torch.manual_seed(0)
    module = cls(_Group(), input_size=8, intermediate_size=6, dtype=torch.float32)
    gate, up = torch.randn(6, 8), torch.randn(6, 8)
    module.enable_int8()
    for tensor, shard in ((gate, 0), (up, 1)):
        q, scale = quantize_per_output_channel(tensor)
        module.weight.weight_loader(module.weight, q, shard)
        module.weight_scale.weight_loader(module.weight_scale, scale, shard)

    _, gate_scale = quantize_per_output_channel(gate)
    _, up_scale = quantize_per_output_channel(up)
    assert torch.equal(module.weight_scale[:6], gate_scale)
    assert torch.equal(module.weight_scale[6:], up_scale)


def test_no_caller_slices_a_stored_weight_directly():
    """Attention and MLP reach past ``forward`` into the fused projections.

    They sliced ``.weight`` and called ``F.linear`` on it, which under Int8
    storage hands an int8 matrix to a BF16 GEMM — an error no test of the linear
    classes alone can see, because their ``forward`` is never called. Anything
    that slices must go through ``dense_weight()`` first.
    """
    import inspect

    from vllm_omni.diffusion.models.minimax_h3 import encoder as enc

    for name in ("MiniMaxH3Qwen3VLTextAttention", "MiniMaxH3Qwen3VLTextMLP"):
        source = inspect.getsource(getattr(enc, name).forward)
        assert "_proj.weight[" not in source, f"{name}.forward slices stored weights directly"
        assert "dense_weight()" in source, f"{name}.forward must dequantize before slicing"


def test_bf16_path_is_untouched_when_int8_is_not_enabled():
    cls = _encoder_module("MiniMaxH3Qwen3VLRowParallelLinear")
    module = cls(_Group(), input_size=8, output_size=6, dtype=torch.float32)
    module.weight.data.normal_()
    assert module.dense_weight() is module.weight  # no copy, no cast
    assert not hasattr(module, "weight_scale")


def test_weight_scale_names_route_like_their_weight():
    enc = _encoder_module("MiniMaxH3Qwen3VLEncoder")
    stub = enc.__new__(enc)
    for suffix, fused, shard in (
        ("q_proj", "qkv_proj", "q"),
        ("v_proj", "qkv_proj", "v"),
    ):
        weight = f"model.language_model.layers.3.self_attn.{suffix}.weight"
        assert stub._map_weight_name(weight) == (f"text_model.layers.3.self_attn.{fused}.weight", shard)
        assert stub._map_weight_name(weight.replace(".weight", ".weight_scale")) == (
            f"text_model.layers.3.self_attn.{fused}.weight_scale",
            shard,
        )
    # A dropped layer's scale is dropped with it, not mapped to layer 50's slot.
    assert stub._map_weight_name("model.language_model.layers.50.mlp.up_proj.weight_scale") is None


def test_a_fused_scale_missing_one_shard_is_rejected():
    """A half-filled fused scale must abort startup, not scale rows with garbage.

    ``weight_scale`` arrives in the same q/k/v and gate/up shards as its weight,
    so it can be partially filled the same way. The final missing-parameter
    check cannot see that: a sibling shard already marked the parameter loaded,
    and the rows the absent shard owned keep whatever ``torch.empty`` left there
    — applied as per-row scales on real output channels.
    """
    from vllm_omni.diffusion.models.minimax_h3.encoder import _verify_fused_shards_complete

    sources = (("q", "q"), ("k", "k"), ("v", "v"))
    expected = {"m.weight": sources, "m.weight_scale": sources}
    complete = {"m.weight": {"q", "k", "v"}, "m.weight_scale": {"q", "k", "v"}}
    _verify_fused_shards_complete(expected, complete)  # no raise

    partial = {"m.weight": {"q", "k", "v"}, "m.weight_scale": {"q", "k"}}
    with pytest.raises(RuntimeError, match=r"m\.weight_scale: \['v'\]"):
        _verify_fused_shards_complete(expected, partial)


def test_int8_modules_register_their_scale_for_shard_completeness():
    """The check above only fires if the scale is actually registered for it."""
    from vllm_omni.diffusion.models.minimax_h3.encoder import _fused_source_shards

    cls = _encoder_module("MiniMaxH3Qwen3VLQKVParallelLinear")
    module = cls(_Group(), hidden_size=8, num_heads=4, num_kv_heads=2, head_dim=4, dtype=torch.float32)
    assert _fused_source_shards(module) == (("q", "q"), ("k", "k"), ("v", "v"))
    assert not hasattr(module, "weight_scale")
    module.enable_int8()
    assert hasattr(module, "weight_scale"), "an Int8 module must expose the scale the seeding looks for"


def test_encoder_rejects_an_online_quantization_config(tmp_path, monkeypatch):
    enc = _encoder_module("MiniMaxH3Qwen3VLEncoder")
    stub = enc.__new__(enc)
    stub.text_model = SimpleNamespace(modules=lambda: iter(()))
    (tmp_path / "config.json").write_text('{"quantization_config": {"quant_method": "int8"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="is_checkpoint_int8_serialized"):
        stub._enable_int8_if_serialized(str(tmp_path))


def test_encoder_rejects_a_checkpoint_missing_retained_layers(tmp_path):
    enc = _encoder_module("MiniMaxH3Qwen3VLEncoder")
    stub = enc.__new__(enc)
    stub.text_model = SimpleNamespace(modules=lambda: iter(()))
    (tmp_path / "config.json").write_text(
        '{"quantization_config": {"quant_method": "int8", "is_checkpoint_int8_serialized": true,'
        ' "retained_layers": 8}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="retained only 8"):
        stub._enable_int8_if_serialized(str(tmp_path))


def test_encoder_leaves_a_bf16_checkpoint_alone(tmp_path):
    enc = _encoder_module("MiniMaxH3Qwen3VLEncoder")
    stub = enc.__new__(enc)
    stub.text_model = SimpleNamespace(modules=lambda: iter(()))
    (tmp_path / "config.json").write_text('{"architectures": ["Qwen3VLForConditionalGeneration"]}', encoding="utf-8")
    assert stub._enable_int8_if_serialized(str(tmp_path)) is False
