#!/usr/bin/env python3
"""Assemble an auditable vLLM MiniMax-H3 partition around pruned Diffusers weights.

The pruned repository contains only the transformer and a truncated text
encoder.  Product parity tests intentionally keep the released tokenizer,
processor, text encoder and VAEs, changing only the transformer.  This tool
creates that view with relative symlinks and records the exact pruning source
in the real ``model_index.json`` used by vLLM-Omni.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

COMPONENTS = ("audio_vae", "processor", "text_encoder", "tokenizer", "video_vae")
PRUNED_INDEX = "diffusion_pytorch_model.safetensors.index.json"
PARTITION_INDEX = "model.safetensors.index.json"


def _pruned_buffer_names(index_name: str, num_layers: int) -> set[str]:
    """The AdaLN tensors that make a checkpoint loadable, in its own naming.

    A pruned transformer reaches this tool in either layout: the Diffusers
    shards straight from the pruning run, or the partition rewrite that the
    offline Int8 pass consumes and re-emits.  Checking Diffusers names against a
    partition checkpoint would report every buffer as missing.
    """
    if index_name == PRUNED_INDEX:
        return {
            "time_embedder.table",
            "adaln_basis",
            "adaln_mean",
            "norm_out.folded_bias",
            *(f"transformer_blocks.{index}.adaln_proj.folded_bias" for index in range(num_layers)),
        }
    return {
        "time_embedder.table",
        "adaln_basis",
        "adaln_mean",
        "final_layer.adaln_proj.folded_bias",
        *(f"blocks.{index}.adaln_proj.folded_bias" for index in range(num_layers)),
    }


def _relative_symlink(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing to replace existing path: {link}")
    link.symlink_to(os.path.relpath(target.resolve(), start=link.parent.resolve()))


def assemble(
    *,
    base_partition: Path,
    pruned_transformer: Path,
    output: Path,
    source_model: str,
    source_revision: str,
    num_inference_steps: int | None = None,
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
    fusion_provenance: Path | None = None,
) -> None:
    base_partition = base_partition.expanduser().resolve()
    pruned_transformer = pruned_transformer.expanduser().resolve()
    index_path = base_partition / "model_index.json"
    transformer_config_path = pruned_transformer / "config.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"base model index is missing: {index_path}")
    if not transformer_config_path.is_file():
        raise FileNotFoundError(f"pruned transformer config is missing: {transformer_config_path}")
    transformer_index = next(
        (name for name in (PRUNED_INDEX, PARTITION_INDEX) if (pruned_transformer / name).is_file()),
        None,
    )
    if transformer_index is None:
        raise FileNotFoundError(
            f"pruned transformer index is missing: expected {PRUNED_INDEX} or {PARTITION_INDEX} in {pruned_transformer}"
        )
    transformer_index_path = pruned_transformer / transformer_index
    if output.exists():
        raise FileExistsError(f"refusing to modify existing output: {output}")

    model_index = json.loads(index_path.read_text(encoding="utf-8"))
    release = model_index.get("_minimax_h3")
    if not isinstance(release, dict) or release.get("partition") not in {"fl2va", "ref2va"}:
        raise ValueError(f"invalid MiniMax-H3 partition metadata in {index_path}")

    config = json.loads(transformer_config_path.read_text(encoding="utf-8"))
    rank = config.get("adaln_rank")
    table_size = config.get("time_table_size")
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"pruned config has invalid adaln_rank: {rank!r}")
    if not isinstance(table_size, int) or table_size < 2:
        raise ValueError(f"pruned config has invalid time_table_size: {table_size!r}")

    weight_map = json.loads(transformer_index_path.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid pruned weight map in {transformer_index_path}")
    required = _pruned_buffer_names(transformer_index, int(config["num_layers"]))
    missing = required - set(weight_map)
    if missing:
        raise ValueError(f"pruned checkpoint is missing required AdaLN tensors: {sorted(missing)}")

    release["pruned"] = {
        "format": "adaln_affine_rank_reduction_v1",
        "source_model": source_model,
        "source_revision": source_revision,
        "adaln_rank": rank,
        "time_table_size": table_size,
        "transformer_index": transformer_index,
        "base_components": str(base_partition),
    }
    if num_inference_steps is not None:
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")
        if video_shift <= 0 or audio_shift <= 0:
            raise ValueError("Turbo sigma shifts must be positive")
        if fusion_provenance is None or not fusion_provenance.is_file():
            raise FileNotFoundError("a verified --fusion-provenance file is required for a Turbo partition")
        provenance = json.loads(fusion_provenance.read_text(encoding="utf-8"))
        verification = provenance.get("fusion_verification")
        if not isinstance(verification, dict) or verification.get("max_abs_error") != 0.0:
            raise ValueError("fusion provenance must contain zero-error verification")
        release["sigma_shift_scales"] = {"video": float(video_shift), "audio": float(audio_shift)}
        release["base_schedule"] = [
            float(num_inference_steps - index) / num_inference_steps for index in range(num_inference_steps + 1)
        ]
        release["distilled"] = {
            "num_inference_steps": num_inference_steps,
            "recommended_num_inference_steps": num_inference_steps,
            "supports_num_inference_steps_override": True,
            "schedule_semantics": "recommended N+1 sigma boundaries for N transformer evaluations",
            **provenance,
        }
    elif fusion_provenance is not None:
        raise ValueError("--fusion-provenance requires --num-inference-steps")

    missing_components = [component for component in COMPONENTS if not (base_partition / component).exists()]
    if missing_components:
        raise FileNotFoundError(f"base components are missing: {missing_components}")
    output.mkdir(parents=True)
    for component in COMPONENTS:
        source = base_partition / component
        _relative_symlink(source, output / component)
    _relative_symlink(pruned_transformer, output / "transformer")
    (output / "model_index.json").write_text(
        json.dumps(model_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-partition", type=Path, required=True)
    parser.add_argument("--pruned-transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-model", default="multimodalart/MiniMax-H3-Pruned")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--video-shift", type=float, default=12.0)
    parser.add_argument("--audio-shift", type=float, default=3.0)
    parser.add_argument("--fusion-provenance", type=Path)
    assemble(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
