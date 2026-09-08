#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bake VDN-H3's adapters into a MiniMax-H3 partition, so the result can be quantized.

VDN ships two LoRAs (``default`` and, on stage-dmd, ``turbo``) plus a linear branch.
Serving fuses the LoRAs into the weight stream at load time, which is fine for BF16 --
but an INT8 checkpoint stores int8 weights with per-output-channel scales, and adding a
BF16 delta to that is meaningless. So the order has to be: bake to BF16 first, quantize
the result second. This is the bake half; ``quantize_minimax_h3_int8.py`` is the other.

The branch is deliberately NOT baked. It names parameters the base does not have, it
stays BF16 (its scan carries the fp32-sensitive statistics), and at TP4 it is ~1.1 GB
per rank. Serving keeps loading it through ``--lora-path``; point that at a directory
holding only ``linear_branch/`` and ``model_spec.json`` once the adapters are baked, or
the adapters would be applied twice.

Target the NATIVE partition (``MiniMax-H3/FL2VA``), not the diffusers-format root copy:
that is what ``quantize_minimax_h3_int8.py`` reads and what vLLM-Omni serves. The two
are the same weights in different spellings -- verified bit-exact, grouped(root q,k,v)
== FL2VA fused, max abs diff 0.0 -- and the fusion places its delta into whichever
layout it is handed.

    python3 tools/minimax_h3/bake_vdn_adapters.py \\
        --src /nfs-data/models/MiniMax-H3/FL2VA \\
        --vdn /nfs-data/models/VDN-MiniMax-H3/stage-dmd-step-250 \\
        --dst /nfs-data/models/MiniMax-H3-FL2VA-VDN8-BF16

Shard-by-shard: one input file is read, fused and written before the next is opened, so
peak host memory is one shard (~5 GB) rather than the 62 GB checkpoint.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - environment, not logic
    sys.exit(f"missing dependency: {exc}. Run inside the vllm-omni image.")

# Everything beside transformer/ that the partition carries and the bake does not touch.
SIBLINGS = ("text_encoder", "video_vae", "audio_vae", "processor", "tokenizer", "model_index.json")


def _index_file(directory: Path) -> Path | None:
    for name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        if (directory / name).is_file():
            return directory / name
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="native H3 partition (contains transformer/)")
    parser.add_argument("--vdn", required=True, help="VDN checkpoint directory")
    parser.add_argument("--dst", required=True, help="output partition")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=50)
    parser.add_argument(
        "--link-siblings", action="store_true", help="symlink text_encoder/VAEs instead of copying (~20 GB saved)"
    )
    parser.add_argument("--dry-run", action="store_true", help="resolve and validate, write nothing")
    parser.add_argument(
        "--adapters",
        default=None,
        help=(
            "comma-separated subset to bake instead of all of them. Only for a base whose "
            "shapes cannot carry an adapter: an r8-pruned partition refactorises AdaLN to "
            "rank 8 and the `turbo` delta does not fit (99.8%% of it lies outside the pruned "
            "basis). The result is a DIFFERENT model from the released one and the stamp "
            "records the subset, so serving it needs its own few-step story."
        ),
    )
    args = parser.parse_args()

    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpoint

    src, dst = Path(args.src), Path(args.dst)
    src_t, dst_t = src / "transformer", dst / "transformer"
    if not src_t.is_dir():
        sys.exit(f"{src_t} is not a directory")

    only = [name.strip() for name in args.adapters.split(",")] if args.adapters else None
    checkpoint = VDNCheckpoint.from_path(
        args.vdn, head_dim=args.head_dim, num_blocks=args.num_blocks, only_adapters=only
    )
    if checkpoint is None:
        sys.exit(f"{args.vdn} is not a VDN checkpoint directory")
    print(f"adapters to bake: {', '.join(checkpoint.adapters)}")

    shards = sorted(src_t.glob("*.safetensors"))
    if not shards:
        sys.exit(f"no safetensors under {src_t}")
    print(f"{len(shards)} shards under {src_t}")

    if not args.dry_run:
        dst_t.mkdir(parents=True, exist_ok=True)

    # ONE stream over every shard, consumed shard-by-shard. Calling fuse_stream once per
    # shard would trip its already-fused guard -- and that guard is worth keeping: it is
    # what stops a second pass from fusing nothing and then passing its own completeness
    # check, which would write out base weights labelled as the student.
    counts = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            counts[shard] = len(handle.keys())

    def every_tensor():
        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    yield name, handle.get_tensor(name)

    # fuse_stream, not apply: the branch tensors are a separate artifact and must not be
    # written into the baked base.
    fused = checkpoint.fuse_stream(every_tensor())

    started = time.time()
    fused_total = 0
    for position, shard in enumerate(shards, 1):
        out = dict(itertools.islice(fused, counts[shard]))
        if len(out) != counts[shard]:
            sys.exit(f"{shard.name}: stream produced {len(out)} of {counts[shard]} tensors")
        fused_total += len(out)
        if args.dry_run:
            print(f"  [{position}/{len(shards)}] {shard.name}: {len(out)} tensors (dry run)")
            del out
            continue
        save_file(out, str(dst_t / shard.name), metadata={"format": "pt"})
        del out
        print(f"  [{position}/{len(shards)}] {shard.name}: written ({time.time() - started:.0f}s elapsed)")

    # Closes the bake: every adapter edit must have met a tensor. A delta that never
    # landed would produce a checkpoint that loads, serves, and is not the released model.
    checkpoint.validate_fully_applied(injections_expected=False)
    print(f"all adapter edits met a tensor ({fused_total} tensors streamed)")

    if args.dry_run:
        return 0

    for name in ("config.json", *(p.name for p in src_t.glob("*.index.json"))):
        if (src_t / name).is_file():
            shutil.copy2(src_t / name, dst_t / name)

    # Stamp what was baked, into the config the served model reads. This is what lets the
    # loader refuse the two silent misconfigurations: a baked base served with an artifact
    # that still has adapters/ (applied twice), and a raw base served with a branch-only
    # artifact (adapters never applied). The offline quantizer copies config.json through,
    # so the stamp survives INT8 conversion.
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDN_BAKED_STAMP_KEY

    config_path = dst_t / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    config[VDN_BAKED_STAMP_KEY] = {
        "adapters": list(checkpoint.adapters),
        "source": str(Path(args.vdn).resolve()),
        "base": str(src.resolve()),
        "baked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config_path.write_text(json.dumps(config, indent=2))
    print(f"stamped {VDN_BAKED_STAMP_KEY}: {list(checkpoint.adapters)}")
    for name in SIBLINGS:
        source = src / name
        target = dst / name
        if not source.exists() or target.exists():
            continue
        if args.link_siblings and source.is_dir():
            target.symlink_to(source)
        elif source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)

    index = _index_file(dst_t)
    print(f"\nbaked -> {dst}\n  index: {index.name if index else '(none, unsharded)'}")
    print("  next: quantize_minimax_h3_int8.py --src", dst, "--dst", str(dst) + "-INT8")
    print("  serve --lora-path must then point at a VDN dir with adapters/ REMOVED,")
    print("  or the adapters get applied a second time on top of the baked ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
