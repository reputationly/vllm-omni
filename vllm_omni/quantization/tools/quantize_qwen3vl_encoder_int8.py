#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Quantize the MiniMax-H3 Qwen3-VL text encoder to Int8 weights.

Why the encoder and not just the DiT
------------------------------------
Per-phase memory sampling shows the H3 peak moves to the *encode* phase as soon
as the DiT is pruned or quantized: every light-DiT arm peaks at the identical
value, reached before denoising starts.  That plateau is this encoder — 62.1 GiB
BF16 on disk, 12.35 GiB resident per card at ``text_encoder_tp_size=4``.  While
it holds the peak, no DiT-side optimization can lower it.

What this writes
----------------
Weight-only Int8 (W8A16): the seven linear projections of each retained decoder
layer become ``int8`` plus a float32 ``[out, 1]`` scale, exactly the shape the
DiT's serialized Int8 path already uses, so the loader change is one dequantize
branch rather than a new format.  Activations stay BF16 — they are half the
encode footprint but they are not what this pass targets, and leaving them alone
keeps the numerics of every non-linear op bit-unchanged.

Deliberately left in BF16:

* the vision tower (1.1 GiB) — small, and outlier-prone in a way per-channel
  scaling handles badly;
* ``embed_tokens`` (1.4 GiB) — a lookup, not a matmul;
* every norm and bias.

Deliberately dropped
--------------------
``lm_head`` and decoder layers at or beyond the retained depth are never built
by ``MiniMaxH3Qwen3VLEncoder`` — ``_map_weight_name`` returns ``None`` for them.
They are 14.2 GiB of shards that today are read off NFS into host RAM and then
discarded on every engine start.  ``--keep-all-layers`` restores them for anyone
who wants a general-purpose checkpoint instead of an H3-shaped one.

Usage
-----
    python3 quantize_qwen3vl_encoder_int8.py \
        --src /nfs-models/.../MiniMax-H3/Ref2VA/text_encoder \
        --dst /nfs-models/.../Qwen3-VL-32B-H3Encoder-INT8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

# ``regex`` rather than stdlib ``re``: repo policy, enforced by the
# check-forbidden-imports hook. Kept inside the try so a bare venv gets the same
# readable dependency message as a missing torch, not a raw traceback.
try:
    import regex as re
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    sys.exit(f"missing dependency: {exc}. Run inside the vllm-omni image or a venv with torch+safetensors+regex.")

# Kept in step with ``MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER``: the encoder returns
# unnormalized layer-50 states, so layers past it never exist as parameters.
DEFAULT_KEEP_LAYERS = 50

TEXT_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."
LAYER_RE = re.compile(r"^layers\.(\d+)\.")

# The projections that carry the parameters worth compressing.  Named in full
# rather than matched by ``.weight`` so a future norm or bias sharing the suffix
# cannot be swept in silently.
QUANTIZED_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)

# Never loaded by the encoder; see the module docstring.
DROPPED_EXACT = ("lm_head.weight", "model.language_model.norm.weight")


def quantize_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel Int8, computed in float32.

    Identical to the DiT pass so both checkpoints dequantize the same way. The
    float32 accumulation matters more here than there: BF16's 8-bit mantissa
    biases the row maxima of a 5120-wide projection enough to shift a scale by a
    full quantization step.
    """
    w = weight.to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def classify(name: str, keep_layers: int | None) -> str:
    """``quantize`` | ``copy`` | ``drop`` for one checkpoint tensor."""
    if name in DROPPED_EXACT:
        return "drop"
    if name.startswith(VISION_PREFIX):
        return "copy"
    if not name.startswith(TEXT_PREFIX):
        return "copy"
    rest = name[len(TEXT_PREFIX) :]
    match = LAYER_RE.match(rest)
    if match is None:
        return "copy"  # embed_tokens and anything else outside a layer
    if keep_layers is not None and int(match.group(1)) >= keep_layers:
        return "drop"
    return "quantize" if rest.endswith(QUANTIZED_SUFFIXES) else "copy"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="BF16 Qwen3-VL encoder directory")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--keep-layers", type=int, default=DEFAULT_KEEP_LAYERS)
    ap.add_argument("--keep-all-layers", action="store_true", help="retain lm_head and every decoder layer")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    args = ap.parse_args()

    keep_layers = None if args.keep_all_layers else args.keep_layers
    src = os.path.realpath(args.src)
    index_path = os.path.join(src, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        sys.exit(f"no safetensors index at {index_path}")
    with open(index_path, encoding="utf-8") as fh:
        weight_map: dict[str, str] = json.load(fh)["weight_map"]

    plan = {name: classify(name, keep_layers) for name in weight_map}
    counts = {kind: sum(1 for v in plan.values() if v == kind) for kind in ("quantize", "copy", "drop")}
    print(f"source     : {src}")
    print(
        f"tensors    : {len(weight_map)}  ->  quantize {counts['quantize']}, "
        f"copy {counts['copy']}, drop {counts['drop']}"
    )
    if counts["quantize"] == 0:
        sys.exit("matched no projections — is this a Qwen3-VL encoder directory?")

    if not args.dry_run:
        os.makedirs(args.dst, exist_ok=True)

    new_weight_map: dict[str, str] = {}
    total_src = total_dst = 0
    shards = sorted(set(weight_map.values()))
    started = time.time()

    for position, shard in enumerate(shards, 1):
        out: dict[str, torch.Tensor] = {}
        with safe_open(os.path.join(src, shard), framework="pt", device="cpu") as fh:
            for name in fh.keys():
                tensor = fh.get_tensor(name)
                total_src += tensor.nelement() * tensor.element_size()
                kind = plan.get(name, "copy")
                if kind == "drop":
                    continue
                if kind == "quantize":
                    if tensor.dim() != 2:
                        sys.exit(f"{name} is {tensor.dim()}-D; per-output-channel scaling expects 2-D")
                    q, scale = quantize_per_output_channel(tensor)
                    out[name] = q
                    out[name[: -len(".weight")] + ".weight_scale"] = scale
                else:
                    out[name] = tensor
        if not out:
            print(f"  [{position}/{len(shards)}] {shard}: empty after drops, skipped")
            continue
        for name, tensor in out.items():
            new_weight_map[name] = shard
            total_dst += tensor.nelement() * tensor.element_size()
        if args.dry_run:
            print(f"  [{position}/{len(shards)}] {shard}: {len(out)} tensors (dry-run, not written)")
        else:
            save_file(out, os.path.join(args.dst, shard), metadata={"format": "pt"})
            print(f"  [{position}/{len(shards)}] {shard}: {len(out)} tensors written  ({time.time() - started:.0f}s)")
        del out

    print(f"\nsize: {total_src / 2**30:.1f} GiB -> {total_dst / 2**30:.1f} GiB")

    with open(os.path.join(src, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    # Read by the encoder to decide whether to build Int8 parameters.  The
    # retained depth travels with it because dropping layers makes this
    # checkpoint valid only for an encoder that stops at the same layer.
    config["quantization_config"] = {
        "quant_method": "int8",
        "is_checkpoint_int8_serialized": True,
        "activation_scheme": "none",
        "quantized_suffixes": list(QUANTIZED_SUFFIXES),
        "retained_layers": keep_layers,
    }

    if args.dry_run:
        print("\nquantization_config that would be written:")
        print(json.dumps(config["quantization_config"], indent=2))
        return 0

    with open(os.path.join(args.dst, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    index_metadata = {"total_size": total_dst}
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
        json.dump({"metadata": index_metadata, "weight_map": new_weight_map}, fh, indent=2)
    for extra in os.listdir(src):
        if extra.endswith(".safetensors") or extra in ("config.json", "model.safetensors.index.json"):
            continue
        source = os.path.join(src, extra)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(args.dst, extra))

    print(f"\nwrote {args.dst} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
