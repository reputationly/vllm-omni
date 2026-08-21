#!/usr/bin/env python3
"""Bake a PEFT LoRA into a sharded Diffusers transformer with strict audit.

This is the layout-preserving counterpart of ``bake_turbo_lora.py``.  It is
used for AdaLN-pruned H3 checkpoints, whose q/k/v and SwiGLU tensors already use
Diffusers names and orders.  Every target is fused as
``BF16(base + (alpha / rank) * B @ A)`` in FP32, then deterministic rows from
every target are independently re-read and checked exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from .lora_provenance import read_lora_checkpoint_metadata
except ImportError:
    from lora_provenance import read_lora_checkpoint_metadata


INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"


@dataclass(frozen=True)
class Pair:
    target: str
    a_key: str
    b_key: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(lora_path: Path) -> list[Pair]:
    with safe_open(str(lora_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    a_keys = sorted(key for key in keys if key.endswith(LORA_A_SUFFIX))
    recognized = set(a_keys)
    pairs = []
    for a_key in a_keys:
        stem = a_key[: -len(LORA_A_SUFFIX)]
        b_key = stem + LORA_B_SUFFIX
        if b_key not in keys:
            raise ValueError(f"LoRA tensor {a_key} has no matching {b_key}")
        recognized.add(b_key)
        pairs.append(Pair(target=stem + ".weight", a_key=a_key, b_key=b_key))
    unsupported = sorted(keys - recognized)
    if not pairs or unsupported:
        raise ValueError(f"LoRA must contain only complete PEFT A/B pairs; unsupported={unsupported[:4]}")
    return pairs


def _read_index(directory: Path) -> dict[str, str]:
    path = directory / INDEX_NAME
    document = json.loads(path.read_text(encoding="utf-8"))
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{path} has no non-empty weight_map")
    return weight_map


def _slice_rows(directory: Path, weight_map: dict[str, str], key: str, rows: list[int]) -> torch.Tensor:
    shard = directory / weight_map[key]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(key)
        return torch.cat([tensor_slice[row : row + 1, :] for row in rows], dim=0)


def _sample_positions(limit: int, *, salt: str, count: int = 4) -> list[int]:
    positions: set[int] = set()
    attempt = 0
    while len(positions) < min(count, limit):
        digest = hashlib.sha256(f"{salt}:{attempt}".encode()).digest()
        positions.add(int.from_bytes(digest[:8], "big") % limit)
        attempt += 1
    return sorted(positions)


def verify(
    *,
    base: Path,
    fused: Path,
    lora_path: Path,
    pairs: list[Pair],
    scale: float,
) -> dict[str, Any]:
    base_map = _read_index(base)
    fused_map = _read_index(fused)
    if base_map != fused_map:
        raise ValueError("base and fused Diffusers indices do not have the same weight map")

    verified_values = 0
    changed_values = 0
    max_abs_error = 0.0
    base_digest = hashlib.sha256()
    fused_digest = hashlib.sha256()
    with safe_open(str(lora_path), framework="pt", device="cpu") as lora:
        for pair in pairs:
            a_slice = lora.get_slice(pair.a_key)
            b_slice = lora.get_slice(pair.b_key)
            a_shape = tuple(a_slice.get_shape())
            b_shape = tuple(b_slice.get_shape())
            if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
                raise ValueError(f"invalid LoRA shapes for {pair.target}: A{a_shape}, B{b_shape}")
            rows = _sample_positions(b_shape[0], salt=pair.target)
            a = lora.get_tensor(pair.a_key).float()
            b = torch.cat([b_slice[row : row + 1, :] for row in rows], dim=0).float()
            delta = b @ a
            base_rows = _slice_rows(base, base_map, pair.target, rows)
            fused_rows = _slice_rows(fused, fused_map, pair.target, rows)
            if base_rows.dtype != torch.bfloat16 or fused_rows.dtype != base_rows.dtype:
                raise ValueError(f"fusion audit requires matching BF16 weights for {pair.target}")
            expected = (base_rows.float() + delta * scale).to(base_rows.dtype)
            error = float((fused_rows.float() - expected.float()).abs().max())
            max_abs_error = max(max_abs_error, error)
            if not torch.equal(fused_rows, expected):
                raise ValueError(
                    f"fused tensor {pair.target} fails deterministic-row audit {rows}; max_abs_error={error:g}"
                )
            changed_values += int((fused_rows != base_rows).sum())
            verified_values += fused_rows.numel()
            for digest, tensor in ((base_digest, base_rows), (fused_digest, fused_rows)):
                digest.update(pair.target.encode())
                digest.update(struct.pack(f"<{len(rows)}Q", *rows))
                digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())

    if changed_values <= 0:
        raise ValueError("fused checkpoint is identical to the base on every audited value")
    return {
        "method": "all_diffusers_lora_targets_deterministic_rows_exact_v1",
        "merge_formula": "BF16(base + (alpha / rank) * (B @ A))",
        "compute_dtype": "float32",
        "verified_factor_pairs": len(pairs),
        "verified_target_tensors": len(pairs),
        "verified_values": verified_values,
        "changed_values": changed_values,
        "max_abs_error": max_abs_error,
        "base_index_sha256": _sha256(base / INDEX_NAME),
        "fused_index_sha256": _sha256(fused / INDEX_NAME),
        "base_sample_sha256": base_digest.hexdigest(),
        "fused_sample_sha256": fused_digest.hexdigest(),
    }


def bake(*, base: Path, lora_path: Path, output: Path) -> dict[str, Any]:
    base = base.expanduser().resolve()
    lora_path = lora_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    metadata = read_lora_checkpoint_metadata(lora_path)
    pairs = _pairs(lora_path)
    weight_map = _read_index(base)
    targets = {pair.target for pair in pairs}
    missing = sorted(targets - set(weight_map))
    if missing:
        raise ValueError(f"LoRA targets are absent from the Diffusers base checkpoint: {missing[:6]}")

    by_shard: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_shard.setdefault(weight_map[pair.target], []).append(pair)
    shards = sorted(set(weight_map.values()))
    temporary = output.with_name(f".{output.name}.bake-{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        with safe_open(str(lora_path), framework="pt", device="cpu") as lora:
            for position, shard in enumerate(shards, start=1):
                shard_pairs = by_shard.get(shard, [])
                print(f"[{position}/{len(shards)}] {shard} ({len(shard_pairs)} targets)", flush=True)
                with safe_open(str(base / shard), framework="pt", device="cpu") as source:
                    tensors = {key: source.get_tensor(key) for key in source.keys()}
                for pair in shard_pairs:
                    weight = tensors[pair.target]
                    a = lora.get_tensor(pair.a_key).float()
                    b = lora.get_tensor(pair.b_key).float()
                    delta = b @ a
                    if delta.shape != weight.shape:
                        raise ValueError(
                            f"delta shape mismatch for {pair.target}: {tuple(delta.shape)} != {tuple(weight.shape)}"
                        )
                    tensors[pair.target] = (weight.float() + delta * metadata.effective_lora_scale).to(weight.dtype)
                save_file(tensors, str(temporary / shard), metadata={"format": "pt"})

        for entry in base.iterdir():
            if entry.name in shards or entry.is_dir():
                continue
            shutil.copy2(entry, temporary / entry.name)
        verification = verify(
            base=base,
            fused=temporary,
            lora_path=lora_path,
            pairs=pairs,
            scale=metadata.effective_lora_scale,
        )
        alpha: int | float = int(metadata.lora_alpha) if metadata.lora_alpha.is_integer() else metadata.lora_alpha
        provenance = {
            "source_lora": metadata.source_lora,
            "source_lora_sha256": metadata.source_lora_sha256,
            "lora_rank": metadata.lora_rank,
            "lora_alpha": alpha,
            "effective_lora_scale": metadata.effective_lora_scale,
            "fusion_verification": verification,
        }
        (temporary / "fusion_provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return provenance
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--lora", dest="lora_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    provenance = bake(**vars(parser.parse_args()))
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
