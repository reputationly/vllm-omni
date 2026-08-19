#!/usr/bin/env python3
"""Audit an offline MiniMax-H3 Turbo LoRA fusion.

The product serves a transformer with the adapter baked into BF16 weights, so
there is no runtime LoRA object left to inspect.  This module ties the three
inputs back together before a partition can be assembled:

* the LoRA file declares one rank and alpha;
* its basename and SHA-256 identify the source rather than a human label;
* deterministic rows from every LoRA target reproduce the fused BF16 tensor as
  ``BF16(base + (alpha / rank) * B @ A)`` exactly.

The verifier deliberately samples every target instead of hashing only the
already-fused directory.  A hash proves identity, not that the claimed formula
or scale produced that identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LORA_A_MARKER = ".lora_A."
_LORA_B_MARKER = ".lora_B."
_INDEX = "model.safetensors.index.json"


@dataclass(frozen=True)
class LoRACheckpointMetadata:
    source_lora: str
    source_lora_sha256: str
    lora_rank: int
    lora_alpha: float
    effective_lora_scale: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safetensors_header(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint is missing: {path}")
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length <= 0 or header_length > size - 8:
            raise ValueError(f"{path} has an invalid safetensors header length: {header_length}")
        try:
            return json.loads(handle.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path} has an invalid safetensors JSON header") from exc


def read_lora_checkpoint_metadata(path: Path, *, with_sha256: bool = True) -> LoRACheckpointMetadata:
    """Read and validate the scale declared by a PEFT safetensors checkpoint."""
    path = path.expanduser().resolve()
    header = _safetensors_header(path)
    metadata = header.get("__metadata__") or {}
    if "alpha" not in metadata:
        raise ValueError(f"{path} declares no LoRA alpha; refusing to guess a merge scale")
    try:
        alpha = float(metadata["alpha"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has a non-numeric LoRA alpha: {metadata['alpha']!r}") from exc
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"{path} has a non-positive or non-finite LoRA alpha: {alpha!r}")

    a_keys = sorted(key for key in header if key != "__metadata__" and _LORA_A_MARKER in key)
    if not a_keys:
        raise ValueError(f"{path} contains no PEFT LoRA A tensors")
    ranks: set[int] = set()
    for a_key in a_keys:
        b_key = a_key.replace(_LORA_A_MARKER, _LORA_B_MARKER, 1)
        if b_key not in header:
            raise ValueError(f"{path} has no B tensor paired with {a_key}")
        a_shape = header[a_key].get("shape")
        b_shape = header[b_key].get("shape")
        if not isinstance(a_shape, list) or len(a_shape) != 2 or not isinstance(b_shape, list) or len(b_shape) != 2:
            raise ValueError(f"{path} has non-matrix LoRA factors for {a_key}")
        rank = int(a_shape[0])
        if rank <= 0 or int(b_shape[1]) != rank:
            raise ValueError(f"{path} has incompatible LoRA factor shapes A{a_shape} B{b_shape}")
        ranks.add(rank)
    if len(ranks) != 1:
        raise ValueError(f"{path} has mixed LoRA ranks: {sorted(ranks)}")
    rank = ranks.pop()
    return LoRACheckpointMetadata(
        source_lora=path.name,
        source_lora_sha256=_sha256(path) if with_sha256 else "",
        lora_rank=rank,
        lora_alpha=alpha,
        effective_lora_scale=alpha / rank,
    )


def _tensor_bytes(tensor: Any) -> bytes:
    """Stable bytes for BF16/FP tensors, including dtypes NumPy cannot expose."""
    import torch

    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _sample_positions(limit: int, *, count: int, salt: str) -> list[int]:
    if limit <= 0:
        raise ValueError(f"cannot sample from a dimension of length {limit}")
    positions: set[int] = set()
    attempt = 0
    while len(positions) < min(count, limit):
        digest = hashlib.sha256(f"{salt}:{attempt}".encode()).digest()
        positions.add(int.from_bytes(digest[:8], "big") % limit)
        attempt += 1
    return sorted(positions)


def _slice_rows(tensor_slice: Any, rows: list[int]) -> Any:
    """Read arbitrary matrix rows through safetensors' supported slice API.

    ``PySafeSlice`` deliberately does not implement list/fancy indexing. Keep
    each read as a one-row contiguous slice so the verifier stays streaming and
    does not materialize a complete H3 projection merely to audit a few rows.
    Concatenating in caller order is important for the swapped SwiGLU halves.
    """
    import torch

    shape = tuple(tensor_slice.get_shape())
    if len(shape) != 2:
        raise ValueError(f"fusion audit can only sample matrix rows, got shape {shape}")
    if not rows:
        raise ValueError("fusion audit needs at least one sampled row")
    invalid = [row for row in rows if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < shape[0]]
    if invalid:
        raise ValueError(f"sampled rows {invalid} are outside matrix shape {shape}")
    return torch.cat([tensor_slice[row : row + 1, :] for row in rows], dim=0)


def _load_rows(directory: Path, weight_map: dict[str, str], key: str, rows: list[int]) -> Any:
    from safetensors import safe_open

    try:
        shard = directory / weight_map[key]
    except KeyError as exc:
        raise ValueError(f"{directory}/{_INDEX} does not map required tensor {key}") from exc
    if not shard.is_file():
        raise FileNotFoundError(f"checkpoint shard is missing: {shard}")
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise ValueError(f"{shard} does not contain indexed tensor {key}")
        tensor_slice = handle.get_slice(key)
        shape = tensor_slice.get_shape()
        return _slice_rows(tensor_slice, rows), tuple(shape)


def _factor_rows(handle: Any, pair: Any, rows: list[int]) -> Any:
    a = handle.get_tensor(pair.a_key).float()
    b = _slice_rows(handle.get_slice(pair.b_key), rows).float()
    return b @ a


def _sample_delta_rows(handle: Any, pairs: list[Any], shape: tuple[int, ...], config: dict[str, Any], key: str):
    """Return native output rows and their FP32 LoRA delta for every slot."""
    import torch

    by_slot = {pair.slot: pair for pair in pairs}
    if set(by_slot) == {"plain"}:
        rows = _sample_positions(shape[0], count=4, salt=key)
        return rows, _factor_rows(handle, by_slot["plain"], rows)

    if set(by_slot) == {"swiglu"}:
        if shape[0] % 2:
            raise ValueError(f"SwiGLU target has an odd output dimension: {key} {shape}")
        half = shape[0] // 2
        within = _sample_positions(half, count=4, salt=key)
        native_and_source = [(row, row + half) for row in within] + [(row + half, row) for row in within]
        native_and_source.sort()
        native_rows = [native for native, _source in native_and_source]
        source_rows = [source for _native, source in native_and_source]
        return native_rows, _factor_rows(handle, by_slot["swiglu"], source_rows)

    if set(by_slot) == {"q", "k", "v"}:
        heads = int(config["num_attention_heads"])
        head_dim = int(config["attention_head_dim"])
        if shape[0] != heads * 3 * head_dim:
            raise ValueError(f"QKV target shape disagrees with config: {key} {shape}, heads={heads}, dim={head_dim}")
        head_rows = _sample_positions(heads * head_dim, count=4, salt=key)
        entries = []
        for flat in head_rows:
            head, within = divmod(flat, head_dim)
            for slot_index, slot in enumerate(("q", "k", "v")):
                native = head * 3 * head_dim + slot_index * head_dim + within
                entries.append((native, slot, flat))
        entries.sort()
        deltas = [_factor_rows(handle, by_slot[slot], [source_row]) for _native, slot, source_row in entries]
        return [native for native, _slot, _source in entries], torch.cat(deltas, dim=0)

    raise ValueError(f"unsupported LoRA slot combination for {key}: {sorted(by_slot)}")


def verify_lora_fusion(
    *,
    base_transformer: Path,
    fused_transformer: Path,
    lora_checkpoint: Path,
    metadata: LoRACheckpointMetadata | None = None,
) -> dict[str, Any]:
    """Prove sampled rows from every target use the checkpoint's alpha/rank."""
    import sys

    import torch
    from safetensors import safe_open

    try:
        from tools.minimax_h3.bake_turbo_lora import build_plan
    except ModuleNotFoundError:  # direct execution from tools/minimax_h3_turbo
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tools.minimax_h3.bake_turbo_lora import build_plan

    base_transformer = base_transformer.expanduser().resolve()
    fused_transformer = fused_transformer.expanduser().resolve()
    lora_checkpoint = lora_checkpoint.expanduser().resolve()
    if base_transformer == fused_transformer:
        raise ValueError("base and fused transformer directories must be different")
    metadata = metadata or read_lora_checkpoint_metadata(lora_checkpoint)
    plan = build_plan(lora_checkpoint)
    if plan.rank != metadata.lora_rank or plan.alpha != metadata.lora_alpha:
        raise ValueError("LoRA tensor plan disagrees with its audited rank/alpha metadata")

    base_index_path = base_transformer / _INDEX
    fused_index_path = fused_transformer / _INDEX
    base_index = json.loads(base_index_path.read_text(encoding="utf-8"))
    fused_index = json.loads(fused_index_path.read_text(encoding="utf-8"))
    base_map = base_index.get("weight_map")
    fused_map = fused_index.get("weight_map")
    if not isinstance(base_map, dict) or not isinstance(fused_map, dict):
        raise ValueError("base and fused transformer indices must contain weight_map objects")
    config = json.loads((base_transformer / "config.json").read_text(encoding="utf-8"))

    base_digest = hashlib.sha256()
    fused_digest = hashlib.sha256()
    verified_values = 0
    changed_values = 0
    max_abs_error = 0.0
    with safe_open(str(lora_checkpoint), framework="pt", device="cpu") as lora:
        for key, pairs in sorted(plan.pairs_by_base.items()):
            # Read shape first, then choose rows that exercise every Q/K/V or
            # SwiGLU half represented by this native target.
            probe, shape = _load_rows(base_transformer, base_map, key, [0])
            del probe
            rows, delta = _sample_delta_rows(lora, pairs, shape, config, key)
            base, base_shape = _load_rows(base_transformer, base_map, key, rows)
            fused, fused_shape = _load_rows(fused_transformer, fused_map, key, rows)
            if base_shape != fused_shape or base.dtype != fused.dtype:
                raise ValueError(
                    f"base/fused tensor contract differs for {key}: "
                    f"shape {base_shape}/{fused_shape}, dtype {base.dtype}/{fused.dtype}"
                )
            if base.dtype != torch.bfloat16:
                raise ValueError(f"fusion audit requires BF16 base/fused weights, got {base.dtype} for {key}")
            expected = (base.float() + delta * metadata.effective_lora_scale).to(base.dtype)
            error = float((fused.float() - expected.float()).abs().max())
            max_abs_error = max(max_abs_error, error)
            if not fused.equal(expected):
                raise ValueError(
                    f"fused tensor {key} does not equal BF16(base + alpha/rank * B@A) "
                    f"on deterministic rows {rows}; max_abs_error={error:g}"
                )
            changed_values += int((fused != base).sum())
            verified_values += fused.numel()
            base_digest.update(key.encode())
            base_digest.update(struct.pack(f"<{len(rows)}Q", *rows))
            base_digest.update(_tensor_bytes(base))
            fused_digest.update(key.encode())
            fused_digest.update(struct.pack(f"<{len(rows)}Q", *rows))
            fused_digest.update(_tensor_bytes(fused))

    if changed_values <= 0:
        raise ValueError("fused checkpoint is identical to the base on every audited value")
    return {
        "method": "all_lora_targets_deterministic_rows_exact_v1",
        "merge_formula": "BF16(base + (alpha / rank) * (B @ A))",
        "compute_dtype": "float32",
        "verified_factor_pairs": plan.num_pairs,
        "verified_target_tensors": len(plan.pairs_by_base),
        "verified_values": verified_values,
        "changed_values": changed_values,
        "max_abs_error": max_abs_error,
        "base_index_sha256": _sha256(base_index_path),
        "fused_index_sha256": _sha256(fused_index_path),
        "base_sample_sha256": base_digest.hexdigest(),
        "fused_sample_sha256": fused_digest.hexdigest(),
    }


def fusion_provenance(*, base_transformer: Path, fused_transformer: Path, lora_checkpoint: Path) -> dict[str, Any]:
    metadata = read_lora_checkpoint_metadata(lora_checkpoint)
    verification = verify_lora_fusion(
        base_transformer=base_transformer,
        fused_transformer=fused_transformer,
        lora_checkpoint=lora_checkpoint,
        metadata=metadata,
    )
    alpha: int | float = int(metadata.lora_alpha) if metadata.lora_alpha.is_integer() else metadata.lora_alpha
    return {
        "source_lora": metadata.source_lora,
        "source_lora_sha256": metadata.source_lora_sha256,
        "lora_rank": metadata.lora_rank,
        "lora_alpha": alpha,
        "effective_lora_scale": metadata.effective_lora_scale,
        "fusion_verification": verification,
    }


def update_distilled_model_index(model_index_path: Path, provenance: dict[str, Any]) -> None:
    """Atomically add verified fusion evidence without changing schedule metadata."""
    required = {
        "source_lora",
        "source_lora_sha256",
        "lora_rank",
        "lora_alpha",
        "effective_lora_scale",
        "fusion_verification",
    }
    missing = sorted(required - provenance.keys())
    if missing:
        raise ValueError(f"fusion provenance is incomplete; missing {missing}")
    verification = provenance["fusion_verification"]
    if not isinstance(verification, dict) or verification.get("max_abs_error") != 0.0:
        raise ValueError("fusion provenance must contain a zero-error verification result")
    model_index_path = model_index_path.expanduser().resolve()
    document = json.loads(model_index_path.read_text(encoding="utf-8"))
    release = document.get("_minimax_h3")
    distilled = release.get("distilled") if isinstance(release, dict) else None
    if not isinstance(distilled, dict):
        raise ValueError(f"{model_index_path} has no _minimax_h3.distilled object")
    existing_source = distilled.get("source_lora")
    if existing_source is not None and Path(str(existing_source)).name != provenance["source_lora"]:
        raise ValueError(
            f"{model_index_path} names source_lora={existing_source!r}, "
            f"but verified checkpoint is {provenance['source_lora']!r}"
        )
    distilled.update(provenance)

    temporary = model_index_path.with_name(f".{model_index_path.name}.lora-audit-{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, model_index_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-transformer", type=Path, required=True)
    parser.add_argument("--fused-transformer", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--model-index", type=Path, help="Existing partition model_index.json to backfill")
    parser.add_argument(
        "--write-model-index",
        action="store_true",
        help="Atomically update --model-index after verification; otherwise only print evidence",
    )
    args = parser.parse_args()
    if args.write_model_index and args.model_index is None:
        parser.error("--write-model-index requires --model-index")
    provenance = fusion_provenance(
        base_transformer=args.base_transformer,
        fused_transformer=args.fused_transformer,
        lora_checkpoint=args.lora_checkpoint,
    )
    if args.write_model_index:
        update_distilled_model_index(args.model_index, provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))


__all__ = [
    "LoRACheckpointMetadata",
    "fusion_provenance",
    "read_lora_checkpoint_metadata",
    "update_distilled_model_index",
    "verify_lora_fusion",
]


if __name__ == "__main__":
    main()
