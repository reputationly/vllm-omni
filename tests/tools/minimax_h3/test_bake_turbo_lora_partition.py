# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools.minimax_h3.bake_turbo_lora import assemble_partition, build_plan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _release(partition: str = "fl2va", *, distilled: bool = False, scale: float = 0.0625) -> dict:
    meta = {
        "schema_version": 1,
        "partition": partition,
        "tasks": ["ref2va"] if partition == "ref2va" else ["t2va", "fl2va"],
        "task_aliases": {},
        # The base release keeps the plain shift; distilled releases pin flow shift 6.
        "sigma_shift_scales": {"video": 6.0, "audio": 3.0} if distilled else {"video": 12.0, "audio": 3.0},
    }
    if distilled:
        meta["base_schedule"] = [1.0, 0.75, 0.5, 0.25, 0.0]
        meta["distilled"] = {"num_inference_steps": 4, "effective_lora_scale": scale}
    return meta


def _partition(tmp_path: Path, name: str, meta: dict) -> Path:
    part = tmp_path / name
    part.mkdir()
    for component in ("audio_vae", "processor", "text_encoder", "tokenizer", "video_vae"):
        (part / component).mkdir()
    (part / "transformer").mkdir()
    (part / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Pipeline", "_minimax_h3": meta}), encoding="utf-8"
    )
    return part


def test_partition_requires_a_contract_when_base_is_not_distilled(tmp_path: Path) -> None:
    base = _partition(tmp_path, "base", _release())
    baked = base / "transformer"

    with pytest.raises(SystemExit) as excinfo:
        assemble_partition(baked, baked, tmp_path / "out")

    message = str(excinfo.value)
    assert "distillation contract" in message
    assert "--contract-from" in message


def test_partition_copies_explicit_contract_and_leaves_base_untouched(tmp_path: Path) -> None:
    base = _partition(tmp_path, "base", _release())
    release = _partition(tmp_path, "release", _release(distilled=True, scale=0.0625))
    base_index = base / "model_index.json"
    before = hashlib.sha256(base_index.read_bytes()).hexdigest()

    out = tmp_path / "out"
    assemble_partition(
        base / "transformer",
        base / "transformer",
        out,
        contract_from=release / "model_index.json",
        fusion_scale=0.25,
    )

    index = out / "model_index.json"
    assert not index.is_symlink(), "servable metadata must be a real file"
    meta = json.loads(index.read_text(encoding="utf-8"))["_minimax_h3"]
    assert meta["sigma_shift_scales"] == {"video": 6.0, "audio": 3.0}
    assert meta["base_schedule"] == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert meta["distilled"]["effective_lora_scale"] == 0.25
    assert hashlib.sha256(base_index.read_bytes()).hexdigest() == before


def test_partition_never_writes_through_a_pre_existing_metadata_symlink(tmp_path: Path) -> None:
    """Regression: an earlier flow symlinked ``model_index.json`` to the base and a later
    metadata patch rewrote the base model in place."""
    base = _partition(tmp_path, "base", _release())
    release = _partition(tmp_path, "release", _release(distilled=True))
    base_index = base / "model_index.json"
    before = hashlib.sha256(base_index.read_bytes()).hexdigest()

    out = tmp_path / "out"
    out.mkdir()
    (out / "model_index.json").symlink_to(base_index)

    assemble_partition(
        base / "transformer",
        base / "transformer",
        out,
        contract_from=release / "model_index.json",
        fusion_scale=0.0625,
    )

    assert not (out / "model_index.json").is_symlink()
    assert hashlib.sha256(base_index.read_bytes()).hexdigest() == before


def test_partition_inherits_from_a_distilled_base_without_explicit_contract(tmp_path: Path) -> None:
    base = _partition(tmp_path, "base", _release(distilled=True, scale=0.5))
    out = tmp_path / "out"

    assemble_partition(base / "transformer", base / "transformer", out, fusion_scale=0.5)

    meta = json.loads((out / "model_index.json").read_text(encoding="utf-8"))["_minimax_h3"]
    assert meta["sigma_shift_scales"]["video"] == 6.0
    assert meta["distilled"]["effective_lora_scale"] == 0.5


def test_partition_copies_shared_components_by_default(tmp_path: Path) -> None:
    base = _partition(tmp_path, "base", _release(distilled=True))
    (base / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "out"

    assemble_partition(base / "transformer", base / "transformer", out, fusion_scale=0.0625)

    component = out / "text_encoder"
    assert not component.is_symlink(), "a served partition must not depend on the base model"
    assert (component / "config.json").is_file()


def test_partition_can_still_symlink_components_when_asked(tmp_path: Path) -> None:
    base = _partition(tmp_path, "base", _release(distilled=True))
    out = tmp_path / "out"

    assemble_partition(base / "transformer", base / "transformer", out, fusion_scale=0.0625, copy_components=False)

    assert (out / "text_encoder").is_symlink()


def _write_named_lora(tmp_path, a_suffix: str, b_suffix: str, alpha_key: str | None) -> Path:
    from safetensors.torch import save_file

    torch.manual_seed(7)
    a = torch.randn(16, 64)
    b = torch.randn(64, 16)
    tensors = {
        f"transformer_blocks.0.attn.to_q{a_suffix}": a.clone(),
        f"transformer_blocks.0.attn.to_q{b_suffix}": b.clone(),
        f"transformer_blocks.0.attn.to_k{a_suffix}": a.clone(),
        f"transformer_blocks.0.attn.to_k{b_suffix}": b.clone(),
        f"transformer_blocks.0.attn.to_v{a_suffix}": a.clone(),
        f"transformer_blocks.0.attn.to_v{b_suffix}": b.clone(),
    }
    meta = {alpha_key: "128"} if alpha_key else {}
    path = tmp_path / f"lora_{a_suffix.strip('.').replace('.', '_')}.safetensors"
    save_file(tensors, str(path), metadata=meta)
    return path


@pytest.mark.parametrize(
    "a_suffix,b_suffix",
    [
        (".lora_A.default.weight", ".lora_B.default.weight"),
        (".lora_A.weight", ".lora_B.weight"),
        (".lora.down.weight", ".lora.up.weight"),
    ],
)
def test_build_plan_accepts_all_lora_namings(tmp_path, a_suffix, b_suffix):
    plan = build_plan(_write_named_lora(tmp_path, a_suffix, b_suffix, "alpha"))
    assert set(plan.pairs_by_base) == {"blocks.0.attn.qkv_proj.weight"}
    assert sorted(p.slot for p in plan.pairs_by_base["blocks.0.attn.qkv_proj.weight"]) == ["k", "q", "v"]
    assert plan.rank == 16
    assert plan.alpha == 128.0


def test_build_plan_reads_lora_alpha_metadata(tmp_path):
    plan = build_plan(_write_named_lora(tmp_path, ".lora.down.weight", ".lora.up.weight", "lora_alpha"))
    assert plan.alpha == 128.0
