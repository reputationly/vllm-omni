import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tools.minimax_h3_turbo.assemble_distilled_partition import assemble, uniform_base_schedule

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _base(tmp_path: Path, partition: str = "fl2va") -> Path:
    base = tmp_path / "base"
    base.mkdir()
    for component in ("audio_vae", "processor", "text_encoder", "tokenizer", "video_vae"):
        (base / component).mkdir()
    (base / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3Pipeline",
                "_minimax_h3": {
                    "partition": partition,
                    "tasks": ["ref2va"] if partition == "ref2va" else ["t2va", "fl2va"],
                    "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
                },
            }
        )
    )
    return base


def _fusion(tmp_path: Path, *, alpha: str = "2", fused_scale: float | None = None):
    base = _base(tmp_path)
    base_transformer = base / "transformer"
    base_transformer.mkdir()
    (base_transformer / "config.json").write_text(
        json.dumps({"num_attention_heads": 1, "attention_head_dim": 1}),
        encoding="utf-8",
    )
    base_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3).to(torch.bfloat16) / 16
    shard = "model-00001-of-00001.safetensors"
    save_file({"blocks.0.attn.out_proj.weight": base_weight}, base_transformer / shard)
    (base_transformer / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"blocks.0.attn.out_proj.weight": shard}}),
        encoding="utf-8",
    )

    lora = tmp_path / "turbo.safetensors"
    a = torch.tensor([[0.125, -0.25, 0.5], [0.75, 0.25, -0.125]], dtype=torch.float32)
    b = torch.tensor([[0.5, 0.25], [-0.5, 0.75], [0.125, -0.25], [0.625, 0.5]], dtype=torch.float32)
    save_file(
        {
            "transformer_blocks.0.attn.to_out.0.lora_A.default.weight": a.to(torch.bfloat16),
            "transformer_blocks.0.attn.to_out.0.lora_B.default.weight": b.to(torch.bfloat16),
        },
        lora,
        metadata={"alpha": alpha},
    )

    expected_scale = float(alpha) / a.shape[0] if fused_scale is None else fused_scale
    fused_weight = (base_weight.float() + (b @ a) * expected_scale).to(torch.bfloat16)
    fused_transformer = tmp_path / "fused" / "transformer"
    fused_transformer.mkdir(parents=True)
    save_file({"blocks.0.attn.out_proj.weight": fused_weight}, fused_transformer / shard)
    (fused_transformer / "config.json").write_text(
        (base_transformer / "config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (fused_transformer / "model.safetensors.index.json").write_text(
        (base_transformer / "model.safetensors.index.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return base, fused_transformer, lora


def test_uniform_schedule_has_one_more_boundary_than_nfe():
    assert uniform_base_schedule(4) == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert len(uniform_base_schedule(8)) == 9


def test_assemble_writes_partition_scoped_schedule_and_relative_links(tmp_path):
    base, transformer, lora = _fusion(tmp_path, alpha="2.0")
    output = tmp_path / "turbo"

    assemble(
        base_partition=base,
        fused_transformer=transformer,
        output=output,
        num_inference_steps=4,
        video_shift=6.0,
        audio_shift=3.0,
        source_lora=lora.name,
        lora_checkpoint=lora,
    )

    release = json.loads((output / "model_index.json").read_text())["_minimax_h3"]
    assert release["base_schedule"] == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert release["sigma_shift_scales"] == {"video": 6.0, "audio": 3.0}
    distilled = release["distilled"]
    assert distilled["num_inference_steps"] == 4
    assert distilled["recommended_num_inference_steps"] == 4
    assert distilled["supports_num_inference_steps_override"] is True
    assert distilled["source_lora"] == "turbo.safetensors"
    assert distilled["source_lora_sha256"] == hashlib.sha256(lora.read_bytes()).hexdigest()
    assert distilled["lora_rank"] == 2
    assert distilled["lora_alpha"] == 2
    assert distilled["effective_lora_scale"] == 1.0
    verification = distilled["fusion_verification"]
    assert verification["method"] == "all_lora_targets_deterministic_rows_exact_v1"
    assert verification["verified_factor_pairs"] == 1
    assert verification["verified_target_tensors"] == 1
    assert verification["verified_values"] == 12
    assert verification["changed_values"] > 0
    assert verification["max_abs_error"] == 0.0
    assert (output / "transformer").is_symlink()
    assert not (output / "transformer").readlink().is_absolute()


def test_assemble_refuses_to_modify_an_existing_output(tmp_path):
    base = _base(tmp_path)
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError):
        assemble(
            base_partition=base,
            fused_transformer=transformer,
            output=output,
            num_inference_steps=4,
            video_shift=12.0,
            audio_shift=3.0,
            source_lora=None,
            lora_checkpoint=tmp_path / "missing.safetensors",
        )


def test_assemble_rejects_a_fused_transformer_with_the_wrong_scale(tmp_path):
    base, transformer, lora = _fusion(tmp_path, alpha="2", fused_scale=0.0625)
    output = tmp_path / "turbo"

    with pytest.raises(ValueError, match=r"does not equal BF16\(base \+ alpha/rank"):
        assemble(
            base_partition=base,
            fused_transformer=transformer,
            output=output,
            num_inference_steps=4,
            video_shift=6.0,
            audio_shift=3.0,
            source_lora=lora.name,
            lora_checkpoint=lora,
        )
    assert not output.exists()


def test_assemble_rejects_a_source_label_that_is_not_the_checkpoint(tmp_path):
    base, transformer, lora = _fusion(tmp_path)
    output = tmp_path / "turbo"

    with pytest.raises(ValueError, match="source-lora names"):
        assemble(
            base_partition=base,
            fused_transformer=transformer,
            output=output,
            num_inference_steps=4,
            video_shift=6.0,
            audio_shift=3.0,
            source_lora="another.safetensors",
            lora_checkpoint=lora,
        )
    assert not output.exists()


# ------------------------------------------------ the LoRA scale the fusion owed


def _lora_file(path: Path, *, alpha: str | None, rank: int = 128) -> Path:
    """A safetensors file with only the header the scale is read from."""
    import struct

    header: dict = {
        "transformer.blocks.0.attn.to_q.lora_A.default.weight": {
            "dtype": "BF16",
            "shape": [rank, 64],
            "data_offsets": [0, 0],
        },
        "transformer.blocks.0.attn.to_q.lora_B.default.weight": {
            "dtype": "BF16",
            "shape": [64, rank],
            "data_offsets": [0, 0],
        },
    }
    if alpha is not None:
        header["__metadata__"] = {"alpha": alpha, "format": "pt"}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)
    return path


def test_the_scale_comes_from_the_checkpoint_not_from_a_default(tmp_path):
    """The three released Turbo LoRAs are all rank 128 and do NOT share an alpha.

    ``fl2v_turbo_4step_v1.0_768p`` declares 128; the 8-step and ``ref2v`` v0.1
    declare 8. Since the merge scales by ``alpha / rank``, one default is right
    for two of them and applies the third at 1/16 strength — which produces no
    error and looks like an ordinary under-distilled sample.
    """
    from tools.minimax_h3_turbo.assemble_distilled_partition import lora_merge_scale

    assert lora_merge_scale(_lora_file(tmp_path / "a.safetensors", alpha="128")) == {
        "lora_alpha": 128,
        "lora_rank": 128,
        "effective_lora_scale": 1.0,
    }
    assert lora_merge_scale(_lora_file(tmp_path / "b.safetensors", alpha="8")) == {
        "lora_alpha": 8,
        "lora_rank": 128,
        "effective_lora_scale": 0.0625,
    }


def test_a_checkpoint_that_declares_no_alpha_is_refused(tmp_path):
    """Silence is the one answer this must not turn into a number."""
    from tools.minimax_h3_turbo.assemble_distilled_partition import lora_merge_scale

    with pytest.raises(ValueError, match="declares no LoRA alpha"):
        lora_merge_scale(_lora_file(tmp_path / "c.safetensors", alpha=None))


def test_the_assembled_partition_records_the_scale(tmp_path):
    """The recorded scale is evidence only after fused rows reproduce it."""
    from tools.minimax_h3_turbo.assemble_distilled_partition import assemble

    base, fused, lora = _fusion(tmp_path, alpha="128")
    output = tmp_path / "out"
    assemble(
        base_partition=base,
        fused_transformer=fused,
        output=output,
        num_inference_steps=4,
        video_shift=6.0,
        audio_shift=3.0,
        source_lora=lora.name,
        lora_checkpoint=lora,
    )

    distilled = json.loads((output / "model_index.json").read_text())["_minimax_h3"]["distilled"]
    assert distilled["lora_alpha"] == 128
    assert distilled["lora_rank"] == 2
    assert distilled["effective_lora_scale"] == 64.0
    assert distilled["fusion_verification"]["max_abs_error"] == 0.0


def test_verified_provenance_backfill_is_atomic_and_preserves_the_schedule(tmp_path):
    from tools.minimax_h3_turbo.lora_provenance import (
        fusion_provenance,
        update_distilled_model_index,
    )

    base, fused, lora = _fusion(tmp_path)
    model_index = tmp_path / "existing-model-index.json"
    model_index.write_text(
        json.dumps(
            {
                "_minimax_h3": {
                    "base_schedule": [1.0, 0.5, 0.0],
                    "distilled": {"num_inference_steps": 2, "source_lora": lora.name},
                }
            }
        ),
        encoding="utf-8",
    )
    provenance = fusion_provenance(
        base_transformer=base / "transformer",
        fused_transformer=fused,
        lora_checkpoint=lora,
    )
    update_distilled_model_index(model_index, provenance)

    release = json.loads(model_index.read_text(encoding="utf-8"))["_minimax_h3"]
    assert release["base_schedule"] == [1.0, 0.5, 0.0]
    assert release["distilled"]["num_inference_steps"] == 2
    assert release["distilled"]["source_lora_sha256"] == hashlib.sha256(lora.read_bytes()).hexdigest()
    assert not list(tmp_path.glob(".existing-model-index.json.*.tmp"))


def test_backfill_rejects_a_model_index_that_names_another_lora(tmp_path):
    from tools.minimax_h3_turbo.lora_provenance import update_distilled_model_index

    model_index = tmp_path / "model_index.json"
    original = json.dumps({"_minimax_h3": {"distilled": {"source_lora": "another.safetensors"}}})
    model_index.write_text(original, encoding="utf-8")
    provenance = {
        "source_lora": "verified.safetensors",
        "source_lora_sha256": "a" * 64,
        "lora_rank": 2,
        "lora_alpha": 2,
        "effective_lora_scale": 1.0,
        "fusion_verification": {"max_abs_error": 0.0},
    }
    with pytest.raises(ValueError, match="names source_lora"):
        update_distilled_model_index(model_index, provenance)
    assert model_index.read_text(encoding="utf-8") == original


def test_backfill_refuses_incomplete_or_nonzero_error_evidence(tmp_path):
    from tools.minimax_h3_turbo.lora_provenance import update_distilled_model_index

    model_index = tmp_path / "model_index.json"
    model_index.write_text(json.dumps({"_minimax_h3": {"distilled": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        update_distilled_model_index(model_index, {"source_lora": "turbo.safetensors"})
    incomplete = {
        "source_lora": "turbo.safetensors",
        "source_lora_sha256": "a" * 64,
        "lora_rank": 2,
        "lora_alpha": 2,
        "effective_lora_scale": 1.0,
        "fusion_verification": {"max_abs_error": 0.5},
    }
    with pytest.raises(ValueError, match="zero-error"):
        update_distilled_model_index(model_index, incomplete)


def test_verifier_samples_qkv_interleave_and_both_swiglu_halves(tmp_path):
    from safetensors import safe_open

    from tools.minimax_h3.bake_turbo_lora import reorder_qkv_to_interleaved, swap_swiglu_halves
    from tools.minimax_h3_turbo.lora_provenance import _sample_delta_rows

    rank, inputs, heads, head_dim = 2, 3, 2, 2
    factors = {}
    qkv_pairs = []
    for slot, offset in (("q", 1), ("k", 2), ("v", 3)):
        a_key = f"{slot}.lora_A.default.weight"
        b_key = f"{slot}.lora_B.default.weight"
        factors[a_key] = (torch.arange(rank * inputs).reshape(rank, inputs) + offset).to(torch.bfloat16)
        factors[b_key] = (torch.arange(heads * head_dim * rank).reshape(heads * head_dim, rank) - offset).to(
            torch.bfloat16
        )
        qkv_pairs.append(SimpleNamespace(slot=slot, a_key=a_key, b_key=b_key))
    swiglu_a = "swiglu.lora_A.default.weight"
    swiglu_b = "swiglu.lora_B.default.weight"
    factors[swiglu_a] = torch.arange(rank * inputs).reshape(rank, inputs).to(torch.bfloat16)
    factors[swiglu_b] = torch.arange(6 * rank).reshape(6, rank).to(torch.bfloat16)
    checkpoint = tmp_path / "factors.safetensors"
    save_file(factors, checkpoint)

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        rows, sampled = _sample_delta_rows(
            handle,
            qkv_pairs,
            (heads * 3 * head_dim, inputs),
            {"num_attention_heads": heads, "attention_head_dim": head_dim},
            "blocks.0.attn.qkv_proj.weight",
        )
        full = {
            pair.slot: handle.get_tensor(pair.b_key).float() @ handle.get_tensor(pair.a_key).float()
            for pair in qkv_pairs
        }
        expected = reorder_qkv_to_interleaved(full["q"], full["k"], full["v"], num_heads=heads, head_dim=head_dim)
        assert rows == list(range(heads * 3 * head_dim))
        assert torch.equal(sampled, expected)

        swiglu_pair = [SimpleNamespace(slot="swiglu", a_key=swiglu_a, b_key=swiglu_b)]
        rows, sampled = _sample_delta_rows(
            handle,
            swiglu_pair,
            (6, inputs),
            {},
            "blocks.0.mlp.fc1.weight",
        )
        expected = swap_swiglu_halves(handle.get_tensor(swiglu_b).float() @ handle.get_tensor(swiglu_a).float())
        assert rows == list(range(6))
        assert torch.equal(sampled, expected)
