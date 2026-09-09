#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Rewrite a ComfyUI ConvRot-INT8 MiniMax-H3 checkpoint into a vLLM partition.

Community H3 fine-tunes are published as one ComfyUI file (e.g.
``WarmBloodAban/Minimax-h3_Singularity``), which differs from our partition in three
independent ways. All three are mechanical; none is a re-derivation.

1. **ConvRot.** Every quantised weight carries a ``comfy_quant`` blob reading
   ``{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}``. The
   weights were rotated before quantisation to spread outliers -- ``W_rot = W @ H^T``
   over groups of 256 input columns, where ``H`` is a *fixed* regular Hadamard matrix
   (size a power of 4, built by Kronecker powers of a 4x4 block, divided by sqrt(size);
   see ``comfy_kitchen/tensor/int8_utils.py::_build_hadamard``). No seed, no learned
   component: ``H`` is orthogonal to machine precision, so ``W = W_rot @ H`` inverts it
   exactly.

   Skipping this step is not a subtle error. Rotated weights compare to the base at
   relative L2 1.37 -- near-orthogonal, i.e. indistinguishable from a different model --
   while the singular values and row norms match, because an orthogonal transform
   preserves both. Un-rotated they land at ~1%, which is the fine-tune itself.

2. **Layout.** Names carry a ``model.diffusion_model.`` prefix, and ``attn.qkv_proj`` is
   stored plain ``[all q; all k; all v]`` where the partition wants it grouped per head
   (head g contributes ``[q_g, k_g, v_g]``).

3. **Pruned AdaLN spelling.** ComfyUI stores the curve table as ``adaln_t_table`` and the
   folded modulation as ``adaln_proj.linear.bias`` (fp16); the partition calls them
   ``time_embedder.table`` and ``adaln_proj.folded_bias``, and asserts fp32 for both
   after load.

   ComfyUI does not store ``adaln_basis``/``adaln_mean`` at all -- inference never reads
   them, but our loader requires them present. They are taken from ``--reference``, and
   only after its ``time_embedder.table`` is confirmed to match this checkpoint's
   ``adaln_t_table``: the same table means the same affine coordinates, so the basis
   describes this checkpoint too. A mismatched table means it is a different pruning run
   and borrowing would be silently wrong, so that case is refused.

Output is BF16 (the INT8 is undone, not preserved): quantising is a separate offline pass,
and re-quantising a de-rotated tensor with our per-channel scheme is not the same
arithmetic ComfyUI's rotated kernel does.

    python3 tools/minimax_h3/convert_comfy_convrot_to_partition.py \\
        --src /nfs-data/models/MiniMax-H3-Singularity/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors \\
        --reference /nfs-data/models/MiniMax-H3-FL2VA-Pruned-r8-BF16-vLLM \\
        --dst /nfs-data/models/MiniMax-H3-Singularity-partition
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

COMFY_PREFIX = "model.diffusion_model."
CURVE_SRC, CURVE_DST = "adaln_t_table", "time_embedder.table"
# Present in the partition, absent from ComfyUI: inference never reads them, our loader
# requires them. Taken from --reference once its curve table is confirmed identical.
BORROWED_BUFFERS = ("adaln_basis", "adaln_mean")
SIBLINGS = ("audio_vae", "processor", "text_encoder", "tokenizer", "video_vae")
SHARD_BYTES = 4 * 1024**3
# What a partition-named checkpoint declares itself as.
PARTITION_CLASS_NAME = "MiniMaxH3DiTModel"
# The partition keeps these fp32 and asserts it after load.
FP32_NAMES = {"rope.inv_freq", CURVE_DST, *BORROWED_BUFFERS}


def build_hadamard(size: int) -> torch.Tensor:
    """The fixed regular Hadamard ConvRot rotates by. Mirrors comfy_kitchen exactly."""
    if size < 4 or (size & (size - 1)) != 0 or size not in {4**k for k in range(1, 12)}:
        raise SystemExit(f"convrot group size must be a power of 4, got {size}")
    block = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
    matrix, current = block, 4
    while current < size:
        matrix = torch.kron(matrix, block)
        current *= 4
    return matrix / (size**0.5)


def unrotate(weight: torch.Tensor, hadamard: torch.Tensor, group: int) -> torch.Tensor:
    """``W_rot = W @ H^T`` per group of ``group`` input columns; H is orthogonal."""
    out_features, in_features = weight.shape
    if in_features % group != 0:
        raise SystemExit(f"in_features {in_features} is not divisible by convrot group {group}")
    return (weight.reshape(out_features, in_features // group, group) @ hadamard).reshape(out_features, in_features)


def plain_to_grouped(weight: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """``[all q; all k; all v]`` -> ``[q0,k0,v0, q1,k1,v1, ...]`` per head."""
    query, key, value = weight.chunk(3, dim=0)
    stacked = torch.stack([part.view(heads, head_dim, -1) for part in (query, key, value)], dim=1)
    return stacked.reshape(-1, weight.shape[1])


def read_reference(root: Path) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict]:
    """The reference partition's curve table, borrowed buffers and transformer config."""
    transformer = root / "transformer"
    wanted = {CURVE_DST, *BORROWED_BUFFERS}
    found: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(str(transformer / "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                # Diffusers-named references spell the table the same way; the two
                # borrowed buffers are top-level in both spellings.
                if key in wanted:
                    found[key] = handle.get_tensor(key)
    missing = sorted(wanted - set(found))
    if missing:
        raise SystemExit(f"{transformer} is missing {missing}; is it a curve-pruned partition?")
    config = json.loads((transformer / "config.json").read_text())
    return found[CURVE_DST], {k: found[k] for k in BORROWED_BUFFERS}, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="ComfyUI single-file .safetensors")
    parser.add_argument(
        "--reference",
        required=True,
        help="a curve-pruned partition to take adaln_basis/adaln_mean, config.json and siblings from",
    )
    parser.add_argument("--dst", required=True, help="output partition root")
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    args = parser.parse_args()

    src, reference, dst = Path(args.src), Path(args.reference), Path(args.dst)
    if not src.is_file():
        raise SystemExit(f"{src} is not a file")
    if dst.exists() and not args.dry_run:
        raise SystemExit(f"{dst} exists; refusing to overwrite an artifact")

    ref_table, borrowed, ref_config = read_reference(reference)

    with safe_open(src, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        table_key = COMFY_PREFIX + CURVE_SRC
        if table_key not in keys:
            raise SystemExit(f"{src} has no {CURVE_SRC}; it is not a curve-pruned checkpoint")
        their_table = handle.get_tensor(table_key).float()

        # Borrowing the basis is only sound if both describe the same curve. Same table
        # => same affine coordinates. Anything else is a different pruning run.
        if their_table.shape != ref_table.shape:
            raise SystemExit(f"curve table {tuple(their_table.shape)} != reference {tuple(ref_table.shape)}")
        drift = float((their_table.double() - ref_table.double()).norm() / ref_table.double().norm())
        print(f"curve table vs {reference.name}: relative {drift:.3e}")
        if drift > 1e-6:
            raise SystemExit(
                f"the reference's curve table differs by {drift:.3e}, so its adaln_basis does not "
                f"describe this checkpoint; point --reference at the partition this was pruned from"
            )

        quantised = sorted(k for k in keys if k.endswith(".comfy_quant"))
        if not quantised:
            raise SystemExit(f"{src} carries no comfy_quant blobs; nothing to de-rotate")
        conf = json.loads(bytes(handle.get_tensor(quantised[0]).numpy().tobytes()))
        print(f"quantisation: {conf}")
        if conf.get("format") != "int8_tensorwise" or not conf.get("convrot"):
            raise SystemExit(f"this tool implements int8_tensorwise+convrot, got {conf}")
        group = int(conf.get("convrot_groupsize", 256))
        hadamard = build_hadamard(group)
        eye = torch.eye(group, dtype=torch.float64)
        print(f"hadamard {group}x{group}: ||H H^T - I|| = {float((hadamard @ hadamard.T - eye).norm()):.3e}")

        out: dict[str, torch.Tensor] = {}
        unmapped: list[str] = []
        n_unrot = n_grouped = 0
        for key in keys:
            if key.endswith((".comfy_quant", ".weight_scale")):
                continue  # consumed with their weight
            if not key.startswith(COMFY_PREFIX):
                unmapped.append(key)
                continue
            name = key[len(COMFY_PREFIX) :]
            tensor = handle.get_tensor(key)

            if name == CURVE_SRC:
                out[CURVE_DST] = tensor.float()
                continue
            if name.endswith(".adaln_proj.linear.bias"):
                # The partition's folded_bias, and it must be fp32: the pruned forward
                # adds it in fp32 because it carries most of the modulation. ComfyUI
                # stored it fp16, so this widens a value that already lost precision --
                # it does not recover it.
                out[name.replace(".linear.bias", ".folded_bias")] = tensor.float()
                continue

            scale_key = key.replace(".weight", ".weight_scale")
            quantised_here = key.endswith(".weight") and scale_key in keys
            if quantised_here:
                dequantised = tensor.float().double() * handle.get_tensor(scale_key).float().double()
                tensor = unrotate(dequantised, hadamard, group)
                n_unrot += 1

            # Regrouping is a property of the NAME, not of whether the tensor happened to
            # be quantised. ComfyUI leaves the token refiner in BF16, so keying this off
            # the quantised branch silently skips it -- and the refiner is the text
            # pathway, so the result renders a perfectly good video that ignores the
            # prompt entirely. Nothing downstream reports that; it was caught by eye.
            if name.endswith(".attn.qkv_proj.weight"):
                tensor = plain_to_grouped(tensor.double(), args.heads, args.head_dim)
                n_grouped += 1

            if quantised_here or name.endswith(".attn.qkv_proj.weight"):
                tensor = tensor.to(torch.bfloat16)

            out[name] = tensor.float() if name in FP32_NAMES else tensor

    out.update(borrowed)

    # Self-check against the reference. A missed layout transform leaves a tensor
    # near-orthogonal to its counterpart (relative ~1.41), two orders of magnitude away
    # from the ~1-3% a fine-tune produces -- so the two are trivially separable, and
    # nothing else in this pipeline would notice. The token refiner's QKV was skipped
    # once precisely because it is BF16 and the regrouping was keyed off the quantised
    # branch; the model still rendered, and only ignored the prompt.
    suspects = []
    for name in sorted(out):
        if not name.endswith((".attn.qkv_proj.weight", ".attn.out_proj.weight", ".mlp.fc1.weight")):
            continue
        theirs = out[name]
        ours = None
        for shard in sorted(glob.glob(str(reference / "transformer" / "*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as handle:
                if name in set(handle.keys()):
                    ours = handle.get_tensor(name)
                    break
        if ours is None or ours.shape != theirs.shape:
            continue
        a, b = theirs.double(), ours.double()
        drift = float((a - b).norm() / b.norm())
        if drift > 0.5:
            suspects.append((name, drift))
    if suspects:
        listing = "\n  ".join(f"{n}: relative {d:.3f}" for n, d in suspects[:8])
        raise SystemExit(
            f"{len(suspects)} tensors are near-orthogonal to {reference.name}'s, which a "
            f"fine-tune cannot produce -- a layout transform was missed:\n  {listing}"
        )
    print(f"layout self-check: no tensor is near-orthogonal to {reference.name}")

    print(f"de-rotated {n_unrot} weights ({n_grouped} regrouped to per-head QKV)")
    print(f"borrowed from reference: {', '.join(BORROWED_BUFFERS)}")
    if unmapped:
        raise SystemExit(f"{len(unmapped)} keys outside {COMFY_PREFIX!r}: {unmapped[:5]}")

    total = sum(t.numel() * t.element_size() for t in out.values())
    print(f"{len(out)} tensors, {total / 1024**3:.1f} GiB BF16")
    if args.dry_run:
        return 0

    transformer = dst / "transformer"
    transformer.mkdir(parents=True)
    shards: list[dict[str, torch.Tensor]] = [{}]
    used = 0
    for name in sorted(out):
        size = out[name].numel() * out[name].element_size()
        if used and used + size > SHARD_BYTES:
            shards.append({})
            used = 0
        shards[-1][name] = out[name]
        used += size

    weight_map = {}
    for index, shard in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(transformer / filename))
        weight_map.update(dict.fromkeys(shard, filename))
        print(f"  [{index}/{len(shards)}] {filename}: {len(shard)} tensors")
    (transformer / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )
    # The reference's config describes the same pruned ARCHITECTURE, but if the
    # reference is diffusers-named its ``_class_name``/``auto_map`` announce a class that
    # expects diffusers tensor names -- and this output is partition-named. Copying it
    # verbatim ships a checkpoint whose config contradicts its own weights, so restate
    # those two fields the way ``convert_pruned_to_partition.py`` does and drop auto_map
    # (it points at modeling_minimax_h3_pruned.py, which reads the names just rewritten).
    ref_config["_class_name"] = PARTITION_CLASS_NAME
    ref_config.pop("auto_map", None)
    ref_config["_singularity_source"] = str(src)
    (transformer / "config.json").write_text(json.dumps(ref_config, indent=2) + "\n")

    shutil.copy2(reference / "model_index.json", dst / "model_index.json")
    for component in SIBLINGS:
        source = reference / component
        if source.exists():
            os.symlink(source.resolve(), dst / component)
    print(f"wrote {dst}")
    print(f"  siblings symlinked from {reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
