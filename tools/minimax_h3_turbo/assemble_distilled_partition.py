#!/usr/bin/env python3
"""Assemble a vLLM MiniMax-H3 Turbo partition without copying model weights.

The fused transformer, base components, and output must already live on the
same filesystem.  The output contains relative symlinks plus a real
``model_index.json`` whose recommended ``base_schedule`` has N+1 sigma
boundaries for N transformer evaluations. Requests may still select another
step count; the runtime then builds the corresponding uniform N+1 grid.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .lora_provenance import fusion_provenance, read_lora_checkpoint_metadata
except ImportError:  # direct `python tools/.../assemble_distilled_partition.py`
    from lora_provenance import fusion_provenance, read_lora_checkpoint_metadata

COMPONENTS = ("audio_vae", "processor", "text_encoder", "tokenizer", "video_vae")


def uniform_base_schedule(num_inference_steps: int) -> list[float]:
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least 1")
    return [float(num_inference_steps - index) / num_inference_steps for index in range(num_inference_steps + 1)]


def lora_merge_scale(path: Path) -> dict[str, float | int]:
    """Compatibility helper for callers that only need declared scale metadata."""
    metadata = read_lora_checkpoint_metadata(path, with_sha256=False)
    alpha: int | float = int(metadata.lora_alpha) if metadata.lora_alpha.is_integer() else metadata.lora_alpha
    return {
        "lora_alpha": alpha,
        "lora_rank": metadata.lora_rank,
        "effective_lora_scale": metadata.effective_lora_scale,
    }


def _relative_symlink(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing to replace existing path: {link}")
    link.symlink_to(os.path.relpath(target.resolve(), start=link.parent.resolve()))


def assemble(
    *,
    base_partition: Path,
    fused_transformer: Path,
    output: Path,
    num_inference_steps: int,
    video_shift: float,
    audio_shift: float,
    source_lora: str | None,
    lora_checkpoint: Path,
) -> None:
    index_path = base_partition / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"base model index is missing: {index_path}")
    if not fused_transformer.is_dir():
        raise FileNotFoundError(f"fused transformer directory is missing: {fused_transformer}")
    if output.exists():
        raise FileExistsError(f"refusing to modify existing output: {output}")

    provenance = fusion_provenance(
        base_transformer=base_partition / "transformer",
        fused_transformer=fused_transformer,
        lora_checkpoint=lora_checkpoint,
    )
    if source_lora is not None and Path(source_lora).name != provenance["source_lora"]:
        raise ValueError(f"--source-lora names {source_lora!r}, but --lora-checkpoint is {provenance['source_lora']!r}")

    model_index = json.loads(index_path.read_text(encoding="utf-8"))
    release = model_index.get("_minimax_h3")
    if not isinstance(release, dict) or release.get("partition") not in {"fl2va", "ref2va"}:
        raise ValueError(f"invalid MiniMax-H3 partition metadata in {index_path}")

    schedule = uniform_base_schedule(num_inference_steps)
    release["base_schedule"] = schedule
    release["sigma_shift_scales"] = {"video": float(video_shift), "audio": float(audio_shift)}
    release["distilled"] = {
        "num_inference_steps": num_inference_steps,
        "recommended_num_inference_steps": num_inference_steps,
        "supports_num_inference_steps_override": True,
        "source_lora": provenance["source_lora"],
        "schedule_semantics": "recommended N+1 sigma boundaries for N transformer evaluations",
    }
    release["distilled"].update({key: value for key, value in provenance.items() if key != "source_lora"})

    output.mkdir(parents=True)
    for component in COMPONENTS:
        source = base_partition / component
        if not source.exists():
            raise FileNotFoundError(f"base component is missing: {source}")
        _relative_symlink(source, output / component)
    _relative_symlink(fused_transformer, output / "transformer")
    (output / "model_index.json").write_text(
        json.dumps(model_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-partition", type=Path, required=True)
    parser.add_argument("--fused-transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--video-shift", type=float, required=True)
    parser.add_argument("--audio-shift", type=float, default=3.0)
    parser.add_argument("--source-lora")
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        required=True,
        help="The exact LoRA fused into the transformer; its SHA, scale and sampled tensor math are verified.",
    )
    args = parser.parse_args()
    assemble(**vars(args))


if __name__ == "__main__":
    main()
