import json
from pathlib import Path

import pytest

from tools.minimax_h3_turbo.assemble_pruned_partition import assemble

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_assemble_pruned_partition_records_source_and_uses_relative_links(tmp_path):
    base = tmp_path / "base"
    for component in ("audio_vae", "processor", "text_encoder", "tokenizer", "video_vae"):
        (base / component).mkdir(parents=True)
    _write_json(
        base / "model_index.json",
        {"_class_name": "MiniMaxH3Pipeline", "_minimax_h3": {"partition": "fl2va"}},
    )

    transformer = tmp_path / "pruned" / "transformer"
    _write_json(
        transformer / "config.json",
        {"num_layers": 2, "adaln_rank": 8, "time_table_size": 1025},
    )
    weight_map = {
        "time_embedder.table": "one.safetensors",
        "adaln_basis": "one.safetensors",
        "adaln_mean": "one.safetensors",
        "norm_out.folded_bias": "one.safetensors",
        "transformer_blocks.0.adaln_proj.folded_bias": "one.safetensors",
        "transformer_blocks.1.adaln_proj.folded_bias": "one.safetensors",
    }
    _write_json(transformer / "diffusion_pytorch_model.safetensors.index.json", {"weight_map": weight_map})

    output = tmp_path / "assembled"
    assemble(
        base_partition=base,
        pruned_transformer=transformer,
        output=output,
        source_model="owner/model",
        source_revision="abc123",
    )

    release = json.loads((output / "model_index.json").read_text())["_minimax_h3"]
    assert release["pruned"] == {
        "format": "adaln_affine_rank_reduction_v1",
        "source_model": "owner/model",
        "source_revision": "abc123",
        "adaln_rank": 8,
        "time_table_size": 1025,
        "transformer_index": "diffusion_pytorch_model.safetensors.index.json",
        "base_components": str(base.resolve()),
    }
    assert (output / "transformer").is_symlink()
    assert not (output / "transformer").readlink().is_absolute()
    assert (output / "transformer").resolve() == transformer.resolve()


def test_assemble_pruned_partition_rejects_incomplete_folded_biases(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    _write_json(base / "model_index.json", {"_minimax_h3": {"partition": "fl2va"}})
    transformer = tmp_path / "transformer"
    _write_json(transformer / "config.json", {"num_layers": 1, "adaln_rank": 8, "time_table_size": 1025})
    _write_json(
        transformer / "diffusion_pytorch_model.safetensors.index.json",
        {
            "weight_map": {
                "time_embedder.table": "one.safetensors",
                "adaln_basis": "one.safetensors",
                "adaln_mean": "one.safetensors",
                "norm_out.folded_bias": "one.safetensors",
            }
        },
    )

    with pytest.raises(ValueError, match="missing required AdaLN tensors"):
        assemble(
            base_partition=base,
            pruned_transformer=transformer,
            output=tmp_path / "assembled",
            source_model="owner/model",
            source_revision="abc123",
        )
