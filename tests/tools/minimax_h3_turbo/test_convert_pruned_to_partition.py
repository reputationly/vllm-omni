import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.minimax_h3_turbo.convert_pruned_to_partition import (
    NAME_RENAMES,
    build_plan,
    convert,
    convert_config,
    fuse_grouped_qkv,
    partition_name,
    swap_swiglu_halves,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    _DIFFUSERS_NAME_RENAMES,
    _reorder_grouped_qkv_to_qkv,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HEADS = 4
HEAD_DIM = 2
HIDDEN = HEADS * HEAD_DIM


def test_rename_table_matches_the_loader():
    # The converter writes names the loader's partition path must recognise;
    # a table that drifts would only surface as skipped weights at serve time.
    assert NAME_RENAMES == _DIFFUSERS_NAME_RENAMES


def test_fuse_grouped_qkv_inverts_the_loader_reorder():
    rows = HEADS * HEAD_DIM
    q, k, v = (torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) + offset for offset in (0, 100, 200))
    fused = fuse_grouped_qkv(q, k, v, num_heads=HEADS)
    assert fused.shape == (3 * rows, 3)
    restored = _reorder_grouped_qkv_to_qkv(fused, num_query_groups=HEADS, heads_per_group=1, head_dim=HEAD_DIM)
    assert torch.equal(restored, torch.cat([q, k, v], dim=0))
    # Head-major, not plain concatenation: the first head's q/k/v come first.
    assert torch.equal(fused[:HEAD_DIM], q[:HEAD_DIM])
    assert torch.equal(fused[HEAD_DIM : 2 * HEAD_DIM], k[:HEAD_DIM])


def test_fuse_grouped_qkv_rejects_mismatched_parts():
    q = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="share a shape"):
        fuse_grouped_qkv(q, torch.zeros(6, 3), q, num_heads=2)
    with pytest.raises(ValueError, match="do not split"):
        fuse_grouped_qkv(q, q, q, num_heads=3)


def test_swap_swiglu_halves():
    fused = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    assert torch.equal(swap_swiglu_halves(fused), torch.tensor([[3.0], [4.0], [1.0], [2.0]]))
    with pytest.raises(ValueError, match="split evenly"):
        swap_swiglu_halves(torch.zeros(3, 1))


def test_partition_name_covers_the_pruned_only_tensors():
    assert partition_name("norm_out.folded_bias") == "final_layer.adaln_proj.folded_bias"
    assert partition_name("transformer_blocks.7.adaln_proj.folded_bias") == "blocks.7.adaln_proj.folded_bias"
    assert partition_name("adaln_basis") == "adaln_basis"
    assert partition_name("token_refiner.refiner_blocks.1.ff.net.2.weight") == "token_refiner.blocks.1.mlp.fc2.weight"


def test_build_plan_rejects_a_half_present_attention():
    with pytest.raises(ValueError, match="incomplete q/k/v"):
        build_plan({"transformer_blocks.0.attn.to_q.weight": "a", "transformer_blocks.0.attn.to_k.weight": "a"})


def test_convert_config_canonicalises_and_keeps_pruning_fields():
    config = convert_config(
        {
            "_class_name": "MiniMaxH3PrunedTransformer3DModel",
            "_diffusers_version": "0.40.0.dev0",
            "auto_map": {"AutoModel": "modeling_minimax_h3_pruned.MiniMaxH3PrunedTransformer3DModel"},
            "adaln_rank": 8,
            "time_table_size": 1025,
            "attention_head_dim": 128,
            "audio_in_channels": 32,
            "ffn_dim": 14336,
            "final_norm_eps": 1e-05,
            "freq_dim": 256,
            "hidden_size": 5376,
            "in_channels": 24,
            "norm_eps": 1e-05,
            "num_attention_heads": 56,
            "num_layers": 50,
            "num_refiner_layers": 2,
            "patch_size": [1, 2, 2],
            "qk_norm_eps": 1e-05,
            "rope_freq_dim": 16,
            "rope_theta": 10000.0,
            "text_dim": 5120,
            "time_embed_dim": 2688,
            "time_embed_hidden_dim": 5376,
        }
    )
    assert config["_class_name"] == "MiniMaxH3DiTModel"
    assert "auto_map" not in config
    assert config["token_refiner_num_layers"] == 2
    assert config["ffn_hidden_size"] == 14336
    assert config["latents_dim"] == 24
    assert config["audio_latents_dim"] == 32
    assert config["timestep_input_dim"] == 256
    assert config["time_embed_hidden_size"] == 5376
    assert config["rope_inv_freq_len"] == 16
    assert config["adaln_out_features"] == 18 * 5376
    assert config["final_adaln_out_features"] == 2 * 5376
    assert config["adaln_rank"] == 8
    assert config["time_table_size"] == 1025
    assert "ffn_dim" not in config


def test_convert_config_rejects_a_released_checkpoint():
    with pytest.raises(ValueError, match="adaln_rank"):
        convert_config({"hidden_size": 5376})


def _pruned_checkpoint(root: Path, *, layers: int = 2) -> None:
    root.mkdir(parents=True)
    tensors: dict[str, torch.Tensor] = {
        "adaln_basis": torch.rand(8, 6, dtype=torch.float32),
        "adaln_mean": torch.rand(6, dtype=torch.float32),
        "time_embedder.table": torch.rand(9, 8, dtype=torch.float32),
        "norm_out.folded_bias": torch.rand(2 * HIDDEN, dtype=torch.float32),
        "norm_out.linear.weight": torch.rand(2 * HIDDEN, 8, dtype=torch.bfloat16),
        "norm_out.norm.weight": torch.rand(HIDDEN, dtype=torch.bfloat16),
        "proj_in.weight": torch.rand(HIDDEN, 4, dtype=torch.float32),
        "proj_in.bias": torch.rand(HIDDEN, dtype=torch.float32),
        "token_refiner.final_norm.weight": torch.rand(HIDDEN, dtype=torch.bfloat16),
        "token_refiner.refiner_blocks.0.attn.to_q.weight": torch.rand(HIDDEN, HIDDEN, dtype=torch.bfloat16),
        "token_refiner.refiner_blocks.0.attn.to_k.weight": torch.rand(HIDDEN, HIDDEN, dtype=torch.bfloat16),
        "token_refiner.refiner_blocks.0.attn.to_v.weight": torch.rand(HIDDEN, HIDDEN, dtype=torch.bfloat16),
    }
    for layer in range(layers):
        prefix = f"transformer_blocks.{layer}"
        tensors[f"{prefix}.adaln_proj.folded_bias"] = torch.rand(18 * HIDDEN, dtype=torch.float32)
        tensors[f"{prefix}.adaln_proj.linear.weight"] = torch.rand(18 * HIDDEN, 8, dtype=torch.bfloat16)
        for part in ("to_q", "to_k", "to_v"):
            tensors[f"{prefix}.attn.{part}.weight"] = torch.rand(HIDDEN, HIDDEN, dtype=torch.bfloat16)
        tensors[f"{prefix}.attn.to_out.0.weight"] = torch.rand(HIDDEN, HIDDEN, dtype=torch.bfloat16)
        tensors[f"{prefix}.ff.net.0.proj.weight"] = torch.rand(4 * HIDDEN, HIDDEN, dtype=torch.bfloat16)
        tensors[f"{prefix}.ff.net.2.weight"] = torch.rand(HIDDEN, 2 * HIDDEN, dtype=torch.bfloat16)

    # Two shards plus the separate affine file, mirroring the real checkpoint.
    affine = {name: tensors.pop(name) for name in ("adaln_basis", "adaln_mean")}
    names = sorted(tensors)
    first = {name: tensors[name] for name in names[: len(names) // 2]}
    second = {name: tensors[name] for name in names[len(names) // 2 :]}
    save_file(affine, str(root / "adaln_affine.safetensors"))
    save_file(first, str(root / "diffusion_pytorch_model-00001-of-00002.safetensors"))
    save_file(second, str(root / "diffusion_pytorch_model-00002-of-00002.safetensors"))
    weight_map = {name: "adaln_affine.safetensors" for name in affine}
    weight_map.update({name: "diffusion_pytorch_model-00001-of-00002.safetensors" for name in first})
    weight_map.update({name: "diffusion_pytorch_model-00002-of-00002.safetensors" for name in second})
    (root / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}), encoding="utf-8"
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3PrunedTransformer3DModel",
                "auto_map": {"AutoModel": "modeling_minimax_h3_pruned.MiniMaxH3PrunedTransformer3DModel"},
                "adaln_rank": 8,
                "time_table_size": 9,
                "attention_head_dim": HEAD_DIM,
                "audio_in_channels": 32,
                "ffn_dim": 2 * HIDDEN,
                "final_norm_eps": 1e-05,
                "freq_dim": 256,
                "hidden_size": HIDDEN,
                "in_channels": 24,
                "norm_eps": 1e-05,
                "num_attention_heads": HEADS,
                "num_layers": layers,
                "num_refiner_layers": 1,
                "patch_size": [1, 2, 2],
                "qk_norm_eps": 1e-05,
                "rope_freq_dim": 16,
                "text_dim": 5120,
                "time_embed_dim": 6,
                "time_embed_hidden_dim": HIDDEN,
            }
        ),
        encoding="utf-8",
    )
    (root / "fusion_provenance.json").write_text(
        json.dumps({"fusion_verification": {"max_abs_error": 0.0}}), encoding="utf-8"
    )


def _read(directory: Path, index_name: str) -> dict[str, torch.Tensor]:
    index = json.loads((directory / index_name).read_text(encoding="utf-8"))
    out: dict[str, torch.Tensor] = {}
    for name, shard in index["weight_map"].items():
        with safe_open(str(directory / shard), framework="pt") as handle:
            out[name] = handle.get_tensor(name)
    return out


def test_convert_rewrites_layout_without_touching_values(tmp_path):
    src = tmp_path / "pruned"
    _pruned_checkpoint(src)
    output = tmp_path / "partition" / "transformer"

    report = convert(src=src, output=output)

    source = _read(src, "diffusion_pytorch_model.safetensors.index.json")
    result = _read(output, "model.safetensors.index.json")
    # Three q/k/v triples collapse into one fused tensor each.
    assert report["source_tensors"] == len(source)
    assert report["partition_tensors"] == len(result) == len(source) - 3 * 2

    for name in ("adaln_basis", "adaln_mean", "time_embedder.table", "blocks.0.adaln_proj.folded_bias"):
        assert result[name].dtype == torch.float32
    assert torch.equal(result["final_layer.adaln_proj.folded_bias"], source["norm_out.folded_bias"])
    assert torch.equal(result["video_patch_proj.weight"], source["proj_in.weight"])
    assert torch.equal(result["blocks.1.attn.out_proj.weight"], source["transformer_blocks.1.attn.to_out.0.weight"])

    fused = result["blocks.0.attn.qkv_proj.weight"]
    assert fused.dtype == torch.bfloat16
    restored = _reorder_grouped_qkv_to_qkv(fused, num_query_groups=HEADS, heads_per_group=1, head_dim=HEAD_DIM)
    assert torch.equal(
        restored,
        torch.cat([source[f"transformer_blocks.0.attn.{part}.weight"] for part in ("to_q", "to_k", "to_v")], dim=0),
    )
    # The token refiner uses the same head count and the same packing.
    assert "token_refiner.blocks.0.attn.qkv_proj.weight" in result

    fc1 = result["blocks.0.mlp.fc1.weight"]
    up, gate = source["transformer_blocks.0.ff.net.0.proj.weight"].chunk(2, dim=0)
    assert torch.equal(fc1, torch.cat([gate, up], dim=0))

    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == sum(t.nelement() * t.element_size() for t in result.values())
    assert sorted(set(index["weight_map"].values())) == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    assert json.loads((output / "config.json").read_text(encoding="utf-8"))["adaln_rank"] == 8
    assert (output / "fusion_provenance.json").is_file()


def test_convert_refuses_to_overwrite(tmp_path):
    src = tmp_path / "pruned"
    _pruned_checkpoint(src)
    output = tmp_path / "partition"
    output.mkdir()
    with pytest.raises(FileExistsError):
        convert(src=src, output=output)


def test_convert_dry_run_writes_nothing(tmp_path):
    src = tmp_path / "pruned"
    _pruned_checkpoint(src)
    output = tmp_path / "partition"
    report = convert(src=src, output=output, dry_run=True)
    assert report["shards"] == 3
    assert not output.exists()
