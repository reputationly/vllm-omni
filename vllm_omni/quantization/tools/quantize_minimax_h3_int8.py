#!/usr/bin/env python3
"""Quantize the MiniMax-H3 DiT to serialized Int8 for vLLM-Omni.

Reads the FL2VA (or Ref2VA) *original* checkpoint — the one vLLM-Omni loads,
not the diffusers-format copy in the repo root — and writes a sibling
directory holding the same tensors with the four large block projections
stored as Int8 plus a per-output-channel scale.

Why this shape
--------------
``vllm_omni.quantization.int8_config`` registers, for a serialized checkpoint::

    weight        int8      [out_features, in_features]     (same orientation as BF16)
    weight_scale  float32   [out_features, 1]               (ChannelQuantScaleParameter)

so the transform is per-output-channel symmetric quantization and nothing has
to be transposed or repacked.  H3 already stores ``qkv_proj`` and ``mlp.fc1``
fused, exactly as the model builds them, and fusing along the output dimension
is what makes per-output-channel scales safe here: every fused sub-projection
keeps its own rows and therefore its own scales.

Only ``blocks.*`` projections are quantized.  Modulation (``adaln_proj``),
output projections (``final_layer``), the patch/condition embedders and the
token refiner stay BF16 — they are the layers this codebase's quantization
guide calls sensitive, and together they are a rounding error next to the 50
blocks.

Usage
-----
    python3 quantize_minimax_h3_int8.py \
        --src /nfs-data/models/MiniMax-H3/FL2VA \
        --dst /nfs-data/models/MiniMax-H3-FL2VA-INT8

Verify without writing anything:

    python3 quantize_minimax_h3_int8.py --src ... --dst ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    sys.exit(f"missing dependency: {exc}. Run inside the vllm-omni image or a venv with torch+safetensors.")


# Suffixes of the parameters we quantize.  Everything else is copied as-is.
QUANTIZED_SUFFIXES = (
    ".attn.qkv_proj.weight",
    ".attn.out_proj.weight",
    ".mlp.fc1.weight",
    ".mlp.fc2.weight",
)

# Only the main DiT blocks.  ``token_refiner.blocks.*`` shares these suffixes,
# hence the explicit prefix test rather than a bare ``endswith``.
QUANTIZED_PREFIX = "blocks."

# ``ignored_layers`` cannot be written as module-name prefixes.  vLLM's
# ``is_layer_skipped`` defaults to ``prefix in ignored_layers`` — an exact
# match against a layer's full prefix — so a group name like "condition_proj"
# never matches the real "condition_proj.linear_1" and the layer is then built
# with an Int8 weight plus a weight_scale that the checkpoint does not carry,
# leaving its parameters on the meta device.  The list is therefore derived
# from the tensors themselves: every 2-D weight we did not quantize.  Names
# that turn out not to be Linear layers are harmless, they simply never match.

# Components that live beside transformer/ and are unchanged by quantization.
SIBLING_COMPONENTS = ("text_encoder", "video_vae", "audio_vae", "processor", "tokenizer")


def quantize_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel Int8 quantization.

    Returns ``(int8_weight, scale)`` with ``scale`` shaped ``[out, 1]`` float32,
    matching ``ChannelQuantScaleParameter``.  Computed in float32 regardless of
    the source dtype so BF16's 8-bit mantissa does not bias the maxima.
    """
    w = weight.to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True)
    # A dead output channel would otherwise divide by zero; its rows are all
    # zero anyway, so any positive scale reproduces them exactly.
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(w / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def should_quantize(name: str) -> bool:
    return name.startswith(QUANTIZED_PREFIX) and name.endswith(QUANTIZED_SUFFIXES)


def link_or_copy(src: str, dst: str) -> None:
    """Symlink a sibling component; the encoder alone is 63 GB."""
    if os.path.exists(dst) or os.path.islink(dst):
        return
    os.symlink(os.path.realpath(src), dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="FL2VA/Ref2VA partition root (contains transformer/)")
    ap.add_argument("--dst", required=True, help="output partition root")
    ap.add_argument("--dry-run", action="store_true", help="report the plan and sizes, write nothing")
    args = ap.parse_args()

    src_tf = os.path.join(args.src, "transformer")
    index_path = os.path.join(src_tf, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        sys.exit(f"no safetensors index at {index_path}")

    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    weight_map: dict[str, str] = index["weight_map"]

    shards = sorted(set(weight_map.values()))
    print(f"source     : {src_tf}")
    print(f"tensors    : {len(weight_map)} across {len(shards)} shards")
    n_target = sum(1 for name in weight_map if should_quantize(name))
    print(f"to quantize: {n_target} tensors  (suffixes: {', '.join(QUANTIZED_SUFFIXES)})")
    if n_target == 0:
        sys.exit("matched no tensors — is this the diffusers-format copy instead of FL2VA/Ref2VA?")

    dst_tf = os.path.join(args.dst, "transformer")
    if not args.dry_run:
        os.makedirs(dst_tf, exist_ok=True)

    new_weight_map: dict[str, str] = {}
    # The checkpoint keeps the Qwen text encoder in BF16.  The transformer's
    # disk quantization config is reused while the composite H3 pipeline builds
    # that encoder, so it must explicitly skip the whole text-model subtree.
    ignored_layers: list[str] = ["text_model"]
    total_src = total_dst = 0
    t0 = time.time()

    for i, shard in enumerate(shards, 1):
        src_path = os.path.join(src_tf, shard)
        out: dict[str, torch.Tensor] = {}
        with safe_open(src_path, framework="pt", device="cpu") as fh:
            for name in fh.keys():
                tensor = fh.get_tensor(name)
                total_src += tensor.nelement() * tensor.element_size()
                if should_quantize(name):
                    if tensor.dim() != 2:
                        sys.exit(f"{name} is {tensor.dim()}-D; per-output-channel scaling expects 2-D")
                    q, scale = quantize_per_output_channel(tensor)
                    out[name] = q
                    out[name[: -len(".weight")] + ".weight_scale"] = scale
                else:
                    out[name] = tensor
                    if name.endswith(".weight") and tensor.dim() == 2:
                        ignored_layers.append(name[: -len(".weight")])

        for name, tensor in out.items():
            new_weight_map[name] = shard
            total_dst += tensor.nelement() * tensor.element_size()

        if args.dry_run:
            print(f"  [{i}/{len(shards)}] {shard}: {len(out)} tensors (dry-run, not written)")
        else:
            save_file(out, os.path.join(dst_tf, shard), metadata={"format": "pt"})
            print(f"  [{i}/{len(shards)}] {shard}: {len(out)} tensors written  ({time.time() - t0:.0f}s)")
        del out

    print(f"\nsize: {total_src / 2**30:.1f} GiB -> {total_dst / 2**30:.1f} GiB")

    # config.json drives detection: TransformerConfig.from_dict() passes these
    # keys straight to DiffusionInt8Config's constructor, so the serialized
    # flag has to be stated here — same convention as
    # merge_mxfp8_checkpoint.py's is_checkpoint_mxfp8_serialized.  Leave it out
    # and the flag keeps its default False, which selects
    # Int8OnlineLinearMethod: it builds meta-device weights expecting to
    # quantize BF16 at load time, and additionally trips the guard that
    # forbids online quantization under DLO+AllGather.
    with open(os.path.join(src_tf, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    config["quantization_config"] = {
        "quant_method": "int8",
        "is_checkpoint_int8_serialized": True,
        "activation_scheme": "dynamic",
        "ignored_layers": sorted(ignored_layers),
        "ignored_layers_match": "substring",
    }

    if args.dry_run:
        print("\nquantization_config that would be written:")
        print(json.dumps(config["quantization_config"], indent=2))
        return 0

    with open(os.path.join(dst_tf, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    # The source index's total_size describes the BF16 payload.  Keeping that
    # stale value makes storage audits and download planners overstate the
    # serialized INT8 checkpoint by tens of GiB even though the loader itself
    # only consumes weight_map.  Preserve other metadata, but publish the real
    # byte count of every tensor written above.
    index_metadata = dict(index.get("metadata", {}))
    index_metadata["total_size"] = total_dst
    with open(os.path.join(dst_tf, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
        json.dump({"metadata": index_metadata, "weight_map": new_weight_map}, fh, indent=2)

    for extra in os.listdir(src_tf):
        if extra.endswith(".safetensors") or extra in ("config.json", "model.safetensors.index.json"):
            continue
        shutil.copy2(os.path.join(src_tf, extra), os.path.join(dst_tf, extra))

    for component in SIBLING_COMPONENTS:
        src_component = os.path.join(args.src, component)
        if os.path.exists(src_component):
            link_or_copy(src_component, os.path.join(args.dst, component))

    # Partition-root files, model_index.json above all: without it the entrypoint
    # cannot resolve the pipeline class and rejects the directory outright.
    for entry in os.listdir(args.src):
        src_entry = os.path.join(args.src, entry)
        if os.path.isfile(src_entry):
            shutil.copy2(src_entry, os.path.join(args.dst, entry))

    print(f"\nwrote {args.dst} in {time.time() - t0:.0f}s")
    # Do not pass --quantization: that builds an online config and would try to
    # quantize these already-Int8 tensors again.  The loader picks the method up
    # from transformer/config.json, which is also what marks the checkpoint as
    # serialized and therefore loadable straight onto the host under offload.
    deploy_cfg = "/deploy-configs/minimax_h3_a100_40g.yaml"
    model_index_path = os.path.join(args.src, "model_index.json")
    if os.path.isfile(model_index_path):
        try:
            with open(model_index_path, encoding="utf-8") as fh:
                model_index = json.load(fh)
            partition = model_index.get("_minimax_h3", {}).get("partition")
            if partition == "ref2va":
                deploy_cfg = "/deploy-configs/minimax_h3_ref2va_w8a8_a100_40g.yaml"
            elif partition == "fl2va":
                deploy_cfg = "/deploy-configs/minimax_h3_a100_40g.yaml"
        except (json.JSONDecodeError, OSError):
            # Keep a best-effort hint so users still get an actionable command.
            pass
    print(f"serve with: vllm serve {args.dst} --omni --deploy-config {deploy_cfg} (no --quantization flag)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
