# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-element statistics for two prompt-embedding dumps.

A 32-element sample cannot separate "one bf16 ULP everywhere", which is the
expected consequence of two different reduction orders, from "a subset of rows
is systematically different", which would be a defect. They are told apart by
where the difference sits, so the report breaks the tensor down by row modality
— MiniMax-H3 tags a vision block's rows as video and everything else as text —
and by percentile rather than by extremes alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _metrics(left: np.ndarray, right: np.ndarray) -> dict:
    """The four quantities the acceptance criteria ask for, plus the spread."""
    diff = np.abs(left - right)
    denom = np.abs(left)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, diff / denom, np.where(diff > 0, np.inf, 0.0))
    flat_left, flat_right = left.ravel(), right.ravel()
    cosine = float(np.dot(flat_left, flat_right) / (np.linalg.norm(flat_left) * np.linalg.norm(flat_right)))
    finite_rel = rel[np.isfinite(rel)]
    return {
        "count": int(diff.size),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(finite_rel.max()) if finite_rel.size else float("nan"),
        "cosine": cosine,
        "abs_percentiles": {str(p): float(np.percentile(diff, p)) for p in (50, 90, 99, 99.9)},
        "std_left": float(left.std()),
        "std_right": float(right.std()),
        "exactly_equal_fraction": float((diff == 0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True, help="official dump directory")
    parser.add_argument("--candidate", type=Path, required=True, help="vLLM dump directory")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    official = np.load(args.official / "prompt_embeds.npy").astype(np.float64)
    candidate = np.load(args.candidate / "prompt_embeds.npy").astype(np.float64)
    official = official.reshape(-1, official.shape[-1])
    candidate = candidate.reshape(-1, candidate.shape[-1])
    if official.shape != candidate.shape:
        raise SystemExit(f"shape mismatch: {official.shape} vs {candidate.shape}")

    report = {"shape": list(official.shape), "overall": _metrics(official, candidate)}

    # Split by row modality. A vision block's rows come through the vision tower
    # and the text rows do not, so a difference confined to one of them points
    # at a path rather than at arithmetic.
    tags_path = args.candidate / "prompt_embeds.json"
    if tags_path.is_file():
        tags = json.loads(tags_path.read_text(encoding="utf-8")).get("token_tags")
        if tags and len(tags) == official.shape[0]:
            tags = np.asarray(tags)
            for name, mask in (("vision_rows", tags == 0), ("text_rows", tags == 1)):
                if mask.any():
                    report[name] = {"rows": int(mask.sum()), **_metrics(official[mask], candidate[mask])}

    # Per-row worst case, to say whether the difference is a few rows or all.
    row_max = np.abs(official - candidate).max(axis=1)
    report["row_max_abs"] = {
        "max": float(row_max.max()),
        "median": float(np.median(row_max)),
        "rows_above_10x_median": int((row_max > 10 * max(np.median(row_max), 1e-12)).sum()),
        "worst_rows": [int(index) for index in np.argsort(row_max)[-5:][::-1]],
    }

    text = json.dumps(report, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
