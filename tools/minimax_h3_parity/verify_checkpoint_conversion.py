# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Are the two MiniMax-H3 checkpoint layouts the same weights?

The repository ships the same DiT twice: a root-level Diffusers layout with
separate q/k/v projections (638 tensors) and a partition layout with them fused
into one ``qkv_proj`` (535). vLLM-Omni loads the partition and fuses/splits on
the way in; the oracle loads the root layout untouched. Every parity result so
far compares numbers produced *after* that conversion, so the conversion itself
has never been checked — a permuted head order there would show up as a plausible
quality difference rather than as an error.

This compares the checkpoints directly, before any model is built:

* tensors that exist on both sides under different names must be bit-identical;
* ``to_q``/``to_k``/``to_v`` must reconstruct ``qkv_proj`` under one of the
  candidate orders, and the script reports *which* order, rather than assuming
  plain concatenation;
* anything unmapped on either side is reported, not skipped.

Bit-identity is the right bar here: both files are the same dtype and neither
side has done arithmetic, so any difference at all is a layout difference.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

# Diffusers name -> partition name, for the tensors that map one to one.
# The top-level pairs are proposed from the names alone; every one of them is
# then checked bit-for-bit, so a wrong guess surfaces as a mismatch rather than
# as a silent pass. Anchored at ^ so `proj_in` cannot swallow `audio_proj_in`.
_RENAMES = (
    (r"^audio_proj_in\.", "audio_patch_proj."),
    (r"^audio_proj_out\.", "final_layer.audio_out."),
    (r"^context_embedder\.", "condition_proj."),
    # AdaLN-pruned checkpoints only: the folded bias sits beside norm_out rather
    # than inside its linear, so the linear rule below cannot reach it.
    (r"^norm_out\.folded_bias$", "final_layer.adaln_proj.folded_bias"),
    (r"^norm_out\.linear\.", "final_layer.adaln_proj.linear."),
    (r"^norm_out\.norm\.", "final_layer.norm."),
    (r"^proj_in\.", "video_patch_proj."),
    (r"^proj_out\.", "final_layer.video_out."),
    (r"^time_embedder\.linear_1\.", "time_embedder.proj_in."),
    (r"^time_embedder\.linear_2\.", "time_embedder.proj_out."),
    (r"^transformer_blocks\.(\d+)\.", r"blocks.\1."),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.", r"token_refiner.blocks.\1."),
    (r"\.attn\.norm_q\.", ".attn.q_norm."),
    (r"\.attn\.norm_k\.", ".attn.k_norm."),
    (r"\.attn\.to_out\.0\.", ".attn.out_proj."),
    (r"\.ff\.net\.0\.proj\.", ".mlp.fc1."),
    (r"\.ff\.net\.2\.", ".mlp.fc2."),
)

_QKV = re.compile(r"^(.*)\.attn\.to_([qkv])\.(weight|bias)$")


def _to_partition_name(name: str) -> str:
    for pattern, replacement in _RENAMES:
        name = re.sub(pattern, replacement, name)
    return name


class _Shards:
    """Random access to a sharded safetensors checkpoint, one file open at a time."""

    def __init__(self, directory: Path, index_name: str):
        self.directory = directory
        index = json.loads((directory / index_name).read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        self._open: dict[str, object] = {}

    def names(self) -> set[str]:
        return set(self.weight_map)

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        handle = self._open.get(shard)
        if handle is None:
            handle = safe_open(str(self.directory / shard), framework="pt")
            self._open[shard] = handle
        return handle.get_tensor(name)


def _identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(torch.equal(left, right))


def _qkv_candidates(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> dict:
    """Orders a fused qkv could plausibly be stored in.

    Plain concatenation and per-head interleaving produce the same total shape,
    so shape agreement proves nothing and the order has to be identified by
    value. Naming them here is what turns "the check passed" into "the checkpoint
    uses this layout".
    """
    candidates = {"concat_qkv": torch.cat([q, k, v], dim=0)}
    rows = q.shape[0]
    if heads and rows % heads == 0:
        head_dim = rows // heads
        trailing = q.shape[1:]
        grouped = torch.stack(
            [
                q.reshape(heads, head_dim, *trailing),
                k.reshape(heads, head_dim, *trailing),
                v.reshape(heads, head_dim, *trailing),
            ],
            dim=1,
        )
        candidates["grouped_per_head"] = grouped.reshape(3 * rows, *trailing)
    return candidates


def _swiglu_candidates(fused: torch.Tensor) -> dict:
    """Orders a SwiGLU gate/up pair could be stored in.

    Diffusers emits one `ff.net.0.proj` holding gate and up stacked; which half
    comes first is a convention, not a fact, and both halves have identical
    shape, so nothing about the tensor reveals the order. Getting it backwards
    swaps the gate with the value it gates — a model that still runs and still
    produces plausible video, which is exactly the kind of defect an
    after-the-fact quality comparison cannot catch.
    """
    rows = fused.shape[0]
    if rows % 2:
        return {}
    first, second = fused[: rows // 2], fused[rows // 2 :]
    candidates = {"gate_up": fused, "up_gate": torch.cat([second, first], dim=0)}
    trailing = fused.shape[1:]
    candidates["interleaved"] = torch.stack([first, second], dim=1).reshape(rows, *trailing)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusers", type=Path, required=True, help="root-level transformer dir")
    parser.add_argument("--partition", type=Path, required=True, help="FL2VA/ or Ref2VA/ transformer dir")
    parser.add_argument("--heads", type=int, default=0, help="attention heads, for the grouped order")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    left = _Shards(args.diffusers, "diffusion_pytorch_model.safetensors.index.json")
    right = _Shards(args.partition, "model.safetensors.index.json")

    heads = args.heads
    if not heads:
        config = args.diffusers / "config.json"
        if config.is_file():
            data = json.loads(config.read_text(encoding="utf-8"))
            heads = int(data.get("num_attention_heads") or data.get("attention_head_num") or 0)

    report: dict = {
        "diffusers_tensors": len(left.weight_map),
        "partition_tensors": len(right.weight_map),
        "heads": heads,
        "identical": 0,
        "mismatched": [],
        "qkv_groups": 0,
        "qkv_orders": {},
        "qkv_unreconstructed": [],
        "mlp_orders": {},
        "mlp_unreconstructed": [],
        "unmapped_diffusers": [],
        "unmapped_partition": [],
    }

    partition_names = right.names()
    consumed: set[str] = set()

    # Fused attention projections first, so their three sources are not then
    # reported as unmapped.
    qkv_prefixes: dict[str, set[str]] = {}
    for name in left.names():
        match = _QKV.match(name)
        if match:
            qkv_prefixes.setdefault(f"{match.group(1)}|{match.group(3)}", set()).add(match.group(2))

    for key, parts in sorted(qkv_prefixes.items()):
        prefix, suffix = key.split("|")
        if parts != {"q", "k", "v"}:
            report["qkv_unreconstructed"].append({"prefix": prefix, "reason": f"only {sorted(parts)}"})
            continue
        # Rename the whole name, not the bare prefix: the block patterns are
        # anchored on a trailing dot, which a prefix on its own does not have.
        fused_name = _to_partition_name(f"{prefix}.attn.qkv_proj.{suffix}")
        if fused_name not in partition_names:
            report["qkv_unreconstructed"].append({"prefix": prefix, "reason": f"no {fused_name}"})
            continue
        fused = right.get(fused_name)
        candidates = _qkv_candidates(
            left.get(f"{prefix}.attn.to_q.{suffix}"),
            left.get(f"{prefix}.attn.to_k.{suffix}"),
            left.get(f"{prefix}.attn.to_v.{suffix}"),
            heads,
        )
        matched = [order for order, tensor in candidates.items() if _identical(tensor, fused)]
        report["qkv_groups"] += 1
        if matched:
            for order in matched:
                report["qkv_orders"][order] = report["qkv_orders"].get(order, 0) + 1
        else:
            report["qkv_unreconstructed"].append(
                {
                    "prefix": prefix,
                    "reason": "no candidate order reproduces the fused tensor",
                    "fused_shape": list(fused.shape),
                    "candidate_shapes": {o: list(t.shape) for o, t in candidates.items()},
                }
            )
        consumed.update(f"{prefix}.attn.to_{part}.{suffix}" for part in "qkv")
        consumed.add(fused_name)

    for name in sorted(left.names() - consumed):
        mapped = _to_partition_name(name)
        if mapped not in partition_names:
            report["unmapped_diffusers"].append(name)
            continue
        consumed.add(mapped)
        source, target = left.get(name), right.get(mapped)
        if _identical(source, target):
            report["identical"] += 1
            continue
        if mapped.endswith("mlp.fc1.weight"):
            # A fused SwiGLU projection: identify which half-order it uses
            # rather than filing it as an unexplained difference.
            matched = [order for order, tensor in _swiglu_candidates(source).items() if _identical(tensor, target)]
            if matched:
                for order in matched:
                    report["mlp_orders"][order] = report["mlp_orders"].get(order, 0) + 1
                continue
            report["mlp_unreconstructed"].append({"diffusers": name, "partition": mapped})
            continue
        report["mismatched"].append({"diffusers": name, "partition": mapped})

    report["unmapped_partition"] = sorted(partition_names - consumed)

    report["verdict"] = (
        "checkpoints agree"
        if not (
            report["mismatched"]
            or report["qkv_unreconstructed"]
            or report["mlp_unreconstructed"]
            or report["unmapped_diffusers"]
        )
        else "differences found"
    )

    # Long name lists are evidence, but they drown the verdict; cap the echo and
    # keep the full list in the written report.
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    summary = dict(report)
    for key in (
        "unmapped_diffusers",
        "unmapped_partition",
        "mismatched",
        "qkv_unreconstructed",
        "mlp_unreconstructed",
    ):
        if len(summary[key]) > 8:
            summary[key] = summary[key][:8] + [f"... {len(report[key]) - 8} more"]
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
