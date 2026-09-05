#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare the effective delta (``scale * B @ A``) of two MiniMax-H3 Turbo LoRAs.

``bake_turbo_lora.py`` reports ``||delta|| / ||W||`` for a single adapter, which
answers "did the scale look sane" but not "is this release a continuation of the
one it replaces". Two adapters can carry the same task name, resolution and step
count and still be unrelated solutions.

That distinction changed how we read an A/B at least once. The 4-step v1.2 768p
release declares alpha 8 (scale 0.0625) where the v1.1 it replaces declares alpha
128 (scale 1.0); its applied delta came out ~28x smaller. This tool showed the
direction was near-orthogonal to v1.1 (cosine median +0.006) while matching the
8-step v1.0 768p line in magnitude (norm ratio 1.23) -- i.e. v1.2 branched off the
alpha-8 lineage rather than continuing v1.1. Cosine is scale-invariant, so that
conclusion holds regardless of whether the declared alpha is trustworthy.

For contrast, a genuine continuation looks like v1.0_768p -> v1.1_768p: norm ratio
1.160, cosine median +0.872 at the default sampling (+0.885 over all 312 factor
pairs, which is how that pair was originally measured -- the sample tracks it).

Usage:

  python tools/minimax_h3_turbo/compare_lora_deltas.py <lora_a> <lora_b>
  python tools/minimax_h3_turbo/compare_lora_deltas.py <lora_a> <lora_b> --every 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

_LORA_A = ".lora_A.default.weight"
_LORA_B = ".lora_B.default.weight"


def _open(path: Path) -> tuple[object, list[str], float, int, float]:
    handle = safe_open(str(path), framework="pt", device="cpu")
    metadata = handle.metadata() or {}
    modules = sorted({key[: -len(_LORA_A)] for key in handle.keys() if key.endswith(_LORA_A)})
    if not modules:
        raise ValueError(f"{path}: no PEFT lora_A tensors (is this a ComfyUI or kohya export?)")
    alpha = metadata.get("alpha") or metadata.get("lora_alpha")
    if alpha is None:
        raise ValueError(f"{path}: metadata has no alpha, so the fusion scale is unknown")
    rank = handle.get_slice(modules[0] + _LORA_A).get_shape()[0]
    return handle, modules, float(alpha) / rank, rank, float(alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", type=Path, help="baseline adapter (usually the one in production)")
    parser.add_argument("right", type=Path, help="candidate adapter")
    parser.add_argument(
        "--every",
        type=int,
        default=25,
        help="sample every Nth module; 25 gives ~13 of 312 layers, enough for the trend "
        "without materialising every delta twice (default: 25)",
    )
    args = parser.parse_args()

    left, left_modules, left_scale, left_rank, left_alpha = _open(args.left)
    right, right_modules, right_scale, right_rank, right_alpha = _open(args.right)
    print(f"L: {args.left.name}  rank={left_rank} alpha={left_alpha:g} scale={left_scale:g}")
    print(f"R: {args.right.name}  rank={right_rank} alpha={right_alpha:g} scale={right_scale:g}")
    if left_modules != right_modules:
        raise SystemExit("module sets differ -- these adapters do not target the same base")

    cosines: list[float] = []
    ratios: list[float] = []
    for module in left_modules[:: args.every]:
        left_delta = left_scale * (
            left.get_tensor(module + _LORA_B).float() @ left.get_tensor(module + _LORA_A).float()
        )
        right_delta = right_scale * (
            right.get_tensor(module + _LORA_B).float() @ right.get_tensor(module + _LORA_A).float()
        )
        left_norm = left_delta.norm().item()
        right_norm = right_delta.norm().item()
        # float64 for the reduction only: over ~10M elements a float32 dot product
        # accumulates enough error to report cosines above 1.0.
        cosine = torch.nn.functional.cosine_similarity(
            left_delta.flatten().double(), right_delta.flatten().double(), dim=0
        ).item()
        cosines.append(cosine)
        ratios.append(right_norm / left_norm)
        print(
            f"  {module:<58} ||L||={left_norm:9.4f} ||R||={right_norm:9.4f}  R/L={ratios[-1]:6.3f}  cos={cosine:+.4f}"
        )

    cosine_t = torch.tensor(cosines)
    ratio_t = torch.tensor(ratios)
    print(f"\ncos   median={cosine_t.median():+.4f}  min={cosine_t.min():+.4f}  max={cosine_t.max():+.4f}")
    print(f"R/L   median={ratio_t.median():.3f}  min={ratio_t.min():.3f}  max={ratio_t.max():.3f}")
    print(
        "\nread: cosine near 0 means the candidate is a different solution, not a refinement;\n"
        "      cosine ~0.9 with R/L ~1 is what continued training on the same recipe looks like."
    )


if __name__ == "__main__":
    main()
