# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare a MiniMax-H3 stage dump against the official one.

Two kinds of comparison, and conflating them is how a parity report becomes
useless:

**Exact.** Tokens, tags, indices, masks, shapes, frame counts, sigma and
timestep tensors, and any CPU tensor produced by the same arithmetic on both
sides. These are reported as equal or not equal; a tolerance here would hide a
contract difference behind a plausible-looking number.

**Tolerant.** Prompt embeddings, DiT output, VAE tensors, final latents —
anything whose value legitimately moves with BF16 rounding, the attention
backend or the parallel topology. These get ``max_abs`` / ``mean_abs`` /
``max_rel`` / cosine, and no verdict: the envelope is established from repeated
runs first, and only then does a threshold mean anything.

vLLM-Omni pads its packed sequence up to a multiple of 64 and the official
layout has none, so a packed tensor is compared over the canonical prefix and
the pad is reported separately rather than counted as a difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Stage names, in the order the doc requires them to be fixed. A later stage's
# numbers are meaningless while an earlier one differs, so the report says where
# the first divergence is rather than leaving that to the reader.
STAGE_ORDER = (
    "geometry",
    "tokens",
    "packing",
    "rng",
    "prompt_embeds",
    "dit_input_step0",
    "dit_velocity_step0",
    "scheduler_step0",
    "latents_mid",
    "latents_final",
    "decoded_media",
)

EXACT_STAGES = frozenset({"geometry", "tokens", "packing", "rng"})

# Fields the candidate reports about its own execution, which the oracle has no
# counterpart for by design — vLLM-Omni's 64-alignment pad accounting is the
# case this exists for. The task brief asks for that pad to be *reported and
# shown to be isolated*, not for it to be absent, so a one-sided field here is
# information rather than a divergence. The prefix is required so the exemption
# cannot be claimed accidentally by a field that should have matched.
CANDIDATE_ONLY_PREFIX = "vllm_"


def _as_float_list(values: Any) -> list[float]:
    if isinstance(values, dict) and "data" in values:
        return [float(value) for value in values["data"]]
    return [float(value) for value in values]


def compare_exact(official: Any, candidate: Any) -> dict[str, Any]:
    """Zero-tolerance comparison, for discrete contract values."""
    equal = official == candidate
    result: dict[str, Any] = {"kind": "exact", "equal": bool(equal)}
    if not equal and isinstance(official, list) and isinstance(candidate, list):
        result["official_len"] = len(official)
        result["candidate_len"] = len(candidate)
        first = next(
            (index for index, (left, right) in enumerate(zip(official, candidate)) if left != right),
            None,
        )
        if first is not None:
            result["first_difference"] = {
                "index": first,
                "official": official[first],
                "candidate": candidate[first],
            }
    return result


def compare_tolerant(official: Any, candidate: Any) -> dict[str, Any]:
    """Numeric distance, with no pass/fail verdict attached.

    Deliberately returns metrics only. A threshold before the error envelope
    exists is a guess, and a guess that ships as a gate is worse than no gate.
    """
    left = _as_float_list(official)
    right = _as_float_list(candidate)
    if len(left) != len(right):
        return {"kind": "tolerant", "error": f"length {len(left)} != {len(right)}"}
    if not left:
        return {"kind": "tolerant", "count": 0}

    diffs = [abs(a - b) for a, b in zip(left, right)]
    max_abs = max(diffs)
    mean_abs = sum(diffs) / len(diffs)
    max_rel = max((abs(a - b) / abs(a) if a != 0 else (0.0 if b == 0 else float("inf"))) for a, b in zip(left, right))
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    cosine = dot / (norm_left * norm_right) if norm_left and norm_right else float("nan")
    return {
        "kind": "tolerant",
        "count": len(left),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "cosine": cosine,
    }


def compare_stage(name: str, official: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare one stage's dump, field by field."""
    fields: dict[str, Any] = {}
    exact_stage = name in EXACT_STAGES
    for key in sorted(set(official) | set(candidate)):
        if key.startswith(CANDIDATE_ONLY_PREFIX) and key not in official:
            fields[key] = {"kind": "candidate_only", "value": candidate[key]}
            continue
        if key not in official or key not in candidate:
            fields[key] = {"kind": "missing", "in_official": key in official, "in_candidate": key in candidate}
            continue
        left, right = official[key], candidate[key]
        if exact_stage or isinstance(left, (str, bool, int)) or key.endswith(("_ids", "_tags", "_indices", "_shape")):
            fields[key] = compare_exact(left, right)
        else:
            fields[key] = compare_tolerant(left, right)
    problems = [
        key
        for key, result in fields.items()
        if result.get("kind") == "missing" or (result.get("kind") == "exact" and not result.get("equal"))
    ]
    return {"stage": name, "exact_stage": exact_stage, "fields": fields, "mismatched": problems}


def build_report(official: dict[str, Any], candidate: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """The whole comparison, plus where the first divergence is.

    "First divergence" is the actionable part: a mismatch at ``tokens`` makes
    every later number a consequence, not a finding, and chasing the final SSIM
    instead is exactly what the task brief forbids.
    """
    stages = []
    for name in STAGE_ORDER:
        if name not in official and name not in candidate:
            continue
        stages.append(compare_stage(name, official.get(name, {}), candidate.get(name, {})))

    first_divergence = next((stage["stage"] for stage in stages if stage["mismatched"]), None)
    return {
        "manifest": manifest,
        "stages": stages,
        "first_exact_divergence": first_divergence,
        "verdict": (
            "exact stages agree"
            if first_divergence is None
            else f"first exact divergence at stage '{first_divergence}'; later stages are consequences"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True, help="official/ stage dump directory or JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="vllm_omni/ stage dump directory or JSON")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True, help="where to write comparison.json")
    args = parser.parse_args()

    def _load(path: Path) -> dict[str, Any]:
        if path.is_dir():
            merged: dict[str, Any] = {}
            for entry in sorted(path.glob("*.json")):
                merged[entry.stem] = json.loads(entry.read_text(encoding="utf-8"))
            return merged
        return json.loads(path.read_text(encoding="utf-8"))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else {}
    report = build_report(_load(args.official), _load(args.candidate), manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(report["verdict"])
    return 0 if report["first_exact_divergence"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
