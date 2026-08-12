#!/usr/bin/env python3
"""
Bake a Diffusers/PEFT LoRA (e.g. the lightx2v MiniMax-H3 Turbo step-distillation
adapters) into a native MiniMax-H3 transformer checkpoint.

vLLM-Omni's H3 pipeline has no runtime LoRA hook, and the released Turbo LoRAs
are in Diffusers naming while the official checkpoint is in the reference
model's native layout. This tool does the structural conversion offline and
writes a drop-in replacement transformer directory.

Three transforms are involved. Getting any of them wrong produces a checkpoint
that loads fine and generates garbage, so each is asserted rather than assumed:

1. Key renaming
       transformer_blocks.N.*              -> blocks.N.*
       token_refiner.refiner_blocks.N.*    -> token_refiner.blocks.N.*

2. QKV fusion with per-head interleave
   The checkpoint stores rows as [head0: q k v, head1: q k v, ...], not as
   three contiguous blocks. Both diffusers' converter (reorder_interleaved_qkv
   in scripts/convert_minimax_h3_to_diffusers.py) and vLLM-Omni's loader
   (_reorder_grouped_qkv_to_qkv, with heads_per_group=1) reorder it at load
   time. The LoRA carries separate to_q / to_k / to_v, so their deltas must be
   re-interleaved into the on-disk layout before being added.

3. SwiGLU half order
   The reference stores mlp.fc1 as [gate; value] (vLLM-Omni computes
   `gate, up = hidden.chunk(2, dim=-1); silu(gate) * up`). Diffusers' SwiGLU
   reads [value; gate]. The LoRA follows diffusers, so the two halves of the
   ff.net.0.proj delta swap places.

The fusion scale is alpha/rank, read per file from the LoRA metadata. It is NOT
constant across the released weights: the 8-step v1.0 adapter is rank 128 /
alpha 8 -> 0.0625, while the 4-step v1.0 768p adapter is rank 128 / alpha 128
-> 1.0. Hard-coding either one silently ruins the other. (lightx2v's own
MiniMaxH3LoraAdapter computes the same `alpha / lora_down.shape[0]`.)

Typical use:

  python tools/minimax_h3/bake_turbo_lora.py \
    --base /nfs-data/models/MiniMax-H3/FL2VA/transformer \
    --lora /nfs-data/models/MiniMax-H3-Turbo-LoRA/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
    --output /nfs-data/models/MiniMax-H3-FL2VA-Turbo8-BF16/transformer \
    --partition-out /nfs-data/models/MiniMax-H3-FL2VA-Turbo8-BF16

  # validate the mapping and print the plan without reading any weights
  python tools/minimax_h3/bake_turbo_lora.py --base ... --lora ... --dry-run

  # verify the tensor transforms against the reference implementations
  python tools/minimax_h3/bake_turbo_lora.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# LoRA factor suffixes, in PEFT's "default" adapter naming.
_LORA_A = ".lora_A.default.weight"
_LORA_B = ".lora_B.default.weight"

# Diffusers module suffix -> (native module suffix, slot).
# `slot` selects the assembly rule applied to the delta before it is added.
_MODULE_MAP: dict[str, tuple[str, str]] = {
    ".attn.to_q": (".attn.qkv_proj.weight", "q"),
    ".attn.to_k": (".attn.qkv_proj.weight", "k"),
    ".attn.to_v": (".attn.qkv_proj.weight", "v"),
    ".attn.to_out.0": (".attn.out_proj.weight", "plain"),
    ".ff.net.0.proj": (".mlp.fc1.weight", "swiglu"),
    ".ff.net.2": (".mlp.fc2.weight", "plain"),
}

_PREFIX_MAP: tuple[tuple[str, str], ...] = (
    # Longest first: token_refiner also starts with a block-like path.
    ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ("transformer_blocks.", "blocks."),
)


# --------------------------------------------------------------------------
# tensor transforms
# --------------------------------------------------------------------------


def reorder_qkv_to_interleaved(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Assemble separate q/k/v into the checkpoint's per-head-interleaved fused layout.

    Inverse of diffusers' ``reorder_interleaved_qkv``: given three [heads*head_dim, in]
    matrices, produce [heads*3*head_dim, in] ordered [head0 q, head0 k, head0 v, head1 q, ...].
    """
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"q/k/v deltas must share a shape, got {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}")
    inner = num_heads * head_dim
    if q.shape[0] != inner:
        raise ValueError(f"q delta has {q.shape[0]} rows, expected {inner} = {num_heads} heads * {head_dim}")

    rest = q.shape[1:]
    parts = [t.reshape(num_heads, head_dim, *rest) for t in (q, k, v)]
    grouped = torch.cat(parts, dim=1)  # [heads, 3*head_dim, in]
    return grouped.reshape(num_heads * 3 * head_dim, *rest)


def reorder_interleaved_to_qkv(
    weight: torch.Tensor, *, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference direction, kept for --self-test round-tripping.

    Mirrors diffusers' ``reorder_interleaved_qkv`` followed by ``split_fused_qkv``.
    """
    expected = num_heads * 3 * head_dim
    if weight.shape[0] != expected:
        raise ValueError(f"fused qkv has {weight.shape[0]} rows, expected {expected}")
    rest = weight.shape[1:]
    grouped = weight.reshape(num_heads, 3 * head_dim, *rest)
    q, k, v = grouped.split(head_dim, dim=1)
    return tuple(t.reshape(num_heads * head_dim, *rest).contiguous() for t in (q, k, v))  # type: ignore[return-value]


def swap_swiglu_halves(delta: torch.Tensor) -> torch.Tensor:
    """Diffusers ``[value; gate]`` -> reference ``[gate; value]``.

    Applies to mlp.fc1 only. Silent failure mode if skipped: the model still
    runs, gate and value are exchanged, output is noise-like.
    """
    if delta.shape[0] % 2:
        raise ValueError(f"fused SwiGLU delta must have an even row count, got {delta.shape[0]}")
    value, gate = delta.chunk(2, dim=0)
    return torch.cat([gate, value], dim=0)


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


@dataclass
class LoRAPair:
    """One low-rank factor pair targeting one native tensor slot."""

    module: str  # diffusers module path, e.g. transformer_blocks.0.attn.to_q
    base_key: str  # native tensor key, e.g. blocks.0.attn.qkv_proj.weight
    slot: str  # q | k | v | plain | swiglu
    a_key: str
    b_key: str
    rank: int


@dataclass
class BakePlan:
    pairs_by_base: dict[str, list[LoRAPair]] = field(default_factory=dict)
    rank: int = 0
    alpha: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def num_pairs(self) -> int:
        return sum(len(v) for v in self.pairs_by_base.values())


def read_safetensors_header(path: Path) -> dict:
    """Read only the JSON header. Avoids pulling 1.4 GB into RAM to inspect keys."""
    with path.open("rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(length))


def map_module(module: str) -> tuple[str, str]:
    """Diffusers module path -> (native tensor key, slot)."""
    for suffix, (base_suffix, slot) in _MODULE_MAP.items():
        if module.endswith(suffix):
            prefix = module[: -len(suffix)]
            for src, dst in _PREFIX_MAP:
                if prefix.startswith(src):
                    return dst + prefix[len(src) :] + base_suffix, slot
            raise ValueError(f"LoRA module has an unmapped prefix: {module}")
    raise ValueError(f"LoRA module has an unmapped suffix: {module}")


def build_plan(lora_path: Path) -> BakePlan:
    header = read_safetensors_header(lora_path)
    metadata = header.get("__metadata__") or {}
    keys = [k for k in header if k != "__metadata__"]

    modules: dict[str, dict[str, str]] = {}
    for key in keys:
        if key.endswith(_LORA_A):
            modules.setdefault(key[: -len(_LORA_A)], {})["a"] = key
        elif key.endswith(_LORA_B):
            modules.setdefault(key[: -len(_LORA_B)], {})["b"] = key
        else:
            raise ValueError(f"unrecognised LoRA tensor (not a PEFT lora_A/lora_B pair): {key}")

    plan = BakePlan(metadata=metadata)
    ranks: set[int] = set()
    for module, factors in sorted(modules.items()):
        if "a" not in factors or "b" not in factors:
            raise ValueError(f"LoRA module {module} is missing its lora_A or lora_B half")
        base_key, slot = map_module(module)
        rank = header[factors["a"]]["shape"][0]
        ranks.add(rank)
        plan.pairs_by_base.setdefault(base_key, []).append(
            LoRAPair(module=module, base_key=base_key, slot=slot, a_key=factors["a"], b_key=factors["b"], rank=rank)
        )

    if len(ranks) != 1:
        raise ValueError(f"mixed LoRA ranks are not supported by this tool: {sorted(ranks)}")
    plan.rank = ranks.pop()
    if "alpha" in metadata:
        plan.alpha = float(metadata["alpha"])

    # A qkv target needs all three slots; two of three would shift the heads.
    for base_key, pairs in plan.pairs_by_base.items():
        slots = sorted(p.slot for p in pairs)
        if base_key.endswith(".attn.qkv_proj.weight"):
            if slots != ["k", "q", "v"]:
                raise ValueError(f"{base_key} expects q/k/v deltas, got {slots}")
        elif len(pairs) != 1:
            raise ValueError(f"{base_key} received {len(pairs)} deltas, expected 1")

    return plan


def resolve_scale(plan: BakePlan, override: float | None) -> float:
    if override is not None:
        return override
    if plan.alpha is None:
        raise SystemExit(
            "This LoRA has no `alpha` in its metadata, so the fusion scale cannot be derived.\n"
            "Confirm the intended scale with the publisher and pass it explicitly with --scale.\n"
            "(For reference: lightx2v's v1.0 adapters use alpha/rank; larryvrh's weights document\n"
            " `W_eff = W + lora_B @ lora_A`, i.e. --scale 1.0.)"
        )
    return plan.alpha / plan.rank


# --------------------------------------------------------------------------
# bake
# --------------------------------------------------------------------------


def load_arch(base_dir: Path) -> dict:
    config = json.loads((base_dir / "config.json").read_text())
    for required in ("num_attention_heads", "attention_head_dim", "hidden_size", "ffn_hidden_size"):
        if required not in config:
            raise SystemExit(f"{base_dir}/config.json is missing `{required}`")
    return config


def compute_delta(
    src: safe_open,
    pairs: list[LoRAPair],
    *,
    scale: float,
    num_heads: int,
    head_dim: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Build the native-layout delta for one base tensor."""
    slots: dict[str, torch.Tensor] = {}
    for pair in pairs:
        a = src.get_tensor(pair.a_key).to(compute_dtype)
        b = src.get_tensor(pair.b_key).to(compute_dtype)
        slots[pair.slot] = torch.mm(b, a)

    if "plain" in slots:
        return slots["plain"] * scale
    if "swiglu" in slots:
        return swap_swiglu_halves(slots["swiglu"]) * scale
    fused = reorder_qkv_to_interleaved(slots["q"], slots["k"], slots["v"], num_heads=num_heads, head_dim=head_dim)
    return fused * scale


def bake(
    *,
    base_dir: Path,
    lora_path: Path,
    out_dir: Path,
    plan: BakePlan,
    scale: float,
    compute_dtype: torch.dtype,
) -> list[tuple[str, float]]:
    config = load_arch(base_dir)
    num_heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]

    index_path = base_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"{index_path} not found; this tool expects a sharded native checkpoint")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    missing = sorted(k for k in plan.pairs_by_base if k not in weight_map)
    if missing:
        preview = "\n  ".join(missing[:6])
        raise SystemExit(
            f"{len(missing)} LoRA targets do not exist in the base checkpoint:\n  {preview}\n"
            "The LoRA and the base partition do not match (wrong partition, or a ComfyUI-format LoRA)."
        )

    by_shard: dict[str, list[str]] = {}
    for base_key in plan.pairs_by_base:
        by_shard.setdefault(weight_map[base_key], []).append(base_key)

    out_dir.mkdir(parents=True, exist_ok=True)
    ratios: list[tuple[str, float]] = []
    shards = sorted({*weight_map.values()})

    with safe_open(str(lora_path), framework="pt", device="cpu") as lora:
        for position, shard in enumerate(shards, start=1):
            targets = by_shard.get(shard, [])
            print(f"[{position}/{len(shards)}] {shard}  ({len(targets)} tensors to patch)", flush=True)

            tensors: dict[str, torch.Tensor] = {}
            with safe_open(str(base_dir / shard), framework="pt", device="cpu") as src:
                for key in src.keys():
                    tensors[key] = src.get_tensor(key)

            for base_key in targets:
                weight = tensors[base_key]
                delta = compute_delta(
                    lora,
                    plan.pairs_by_base[base_key],
                    scale=scale,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    compute_dtype=compute_dtype,
                )
                if delta.shape != weight.shape:
                    raise SystemExit(
                        f"delta shape mismatch for {base_key}: delta={tuple(delta.shape)} weight={tuple(weight.shape)}"
                    )
                promoted = weight.to(compute_dtype)
                ratio = (delta.norm() / promoted.norm().clamp_min(1e-12)).item()
                ratios.append((base_key, ratio))
                tensors[base_key] = (promoted + delta).to(weight.dtype)
                del delta, promoted

            # `save_file` writes tensors in alphabetical key order regardless of
            # insertion order (verified on safetensors 0.8.0), so the output
            # shards do not preserve the base checkpoint's physical layout —
            # e.g. `out_proj, k_norm, q_norm, qkv_proj` becomes
            # `k_norm, out_proj, q_norm, qkv_proj`. This is a property of the
            # writer, not of this tool.
            #
            # It does not affect correctness (safetensors resolves tensors by
            # name via data_offsets) and shows no inference cost: a baked 768p
            # checkpoint ran 338 s warm against the base's 331 s at identical
            # step counts. The only real cost is that binary diffs against the
            # base checkpoint are meaningless. Preserving the order would mean
            # hand-rolling the container format; not worth it.
            save_file(tensors, str(out_dir / shard), metadata={"format": "pt"})
            del tensors

    # Everything that is not a shard is copied verbatim. Shapes and dtypes are
    # unchanged, so the index (including total_size) stays valid.
    for entry in sorted(base_dir.iterdir()):
        if entry.name in shards or entry.is_dir():
            continue
        shutil.copy2(entry, out_dir / entry.name)

    return ratios


def assemble_partition(base_dir: Path, out_transformer: Path, partition_out: Path) -> None:
    """Build a servable partition dir: baked transformer + symlinks to shared components.

    The text encoder and VAEs are unchanged and large; symlinking avoids a second
    copy on NFS. The result is what `--model` should point at.
    """
    source_partition = base_dir.parent
    partition_out.mkdir(parents=True, exist_ok=True)

    for entry in sorted(source_partition.iterdir()):
        if entry.name == base_dir.name:
            continue
        link = partition_out / entry.name
        if link.is_symlink() or link.exists():
            continue
        os.symlink(entry.resolve(), link)

    target = partition_out / base_dir.name
    if target.resolve() != out_transformer.resolve():
        if target.is_symlink() or target.exists():
            raise SystemExit(f"{target} already exists and does not point at {out_transformer}")
        os.symlink(out_transformer.resolve(), target)


# --------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------


def self_test() -> int:
    heads, head_dim, in_features = 4, 8, 5
    inner = heads * head_dim

    fused = torch.randn(heads * 3 * head_dim, in_features)
    q, k, v = reorder_interleaved_to_qkv(fused, num_heads=heads, head_dim=head_dim)
    assert q.shape == (inner, in_features), q.shape
    rebuilt = reorder_qkv_to_interleaved(q, k, v, num_heads=heads, head_dim=head_dim)
    assert torch.equal(rebuilt, fused), "qkv interleave round-trip mismatch"

    # Row 0 of head 1 in the fused layout must be q of head 1, not a k/v row.
    assert torch.equal(fused[3 * head_dim], q[head_dim]), "interleave places heads incorrectly"

    ffn = 6
    delta = torch.randn(2 * ffn, in_features)
    swapped = swap_swiglu_halves(delta)
    assert torch.equal(swapped[:ffn], delta[ffn:]), "gate half not moved to the front"
    assert torch.equal(swapped[ffn:], delta[:ffn]), "value half not moved to the back"
    assert torch.equal(swap_swiglu_halves(swapped), delta), "swap is not an involution"

    assert map_module("transformer_blocks.7.attn.to_v") == ("blocks.7.attn.qkv_proj.weight", "v")
    assert map_module("transformer_blocks.7.attn.to_out.0") == ("blocks.7.attn.out_proj.weight", "plain")
    assert map_module("transformer_blocks.7.ff.net.0.proj") == ("blocks.7.mlp.fc1.weight", "swiglu")
    assert map_module("transformer_blocks.7.ff.net.2") == ("blocks.7.mlp.fc2.weight", "plain")
    assert map_module("token_refiner.refiner_blocks.1.attn.to_q") == (
        "token_refiner.blocks.1.attn.qkv_proj.weight",
        "q",
    )

    print("self-test OK: qkv interleave round-trip, SwiGLU swap, key mapping")
    return 0


# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bake a Diffusers/PEFT LoRA into a native MiniMax-H3 transformer checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base", type=Path, help="native transformer dir, e.g. .../MiniMax-H3/FL2VA/transformer")
    parser.add_argument("--lora", type=Path, help="Diffusers/PEFT LoRA .safetensors")
    parser.add_argument("--output", type=Path, help="output transformer dir")
    parser.add_argument(
        "--partition-out",
        type=Path,
        default=None,
        help="also assemble a servable partition dir here (symlinks the shared components)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="fusion scale; default reads alpha/rank from the LoRA metadata",
    )
    parser.add_argument("--strength", type=float, default=1.0, help="extra multiplier on top of the scale")
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="dtype for B@A and the addition; the stored dtype is unchanged",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan, touch no weights")
    parser.add_argument("--self-test", action="store_true", help="check the tensor transforms and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()

    if not args.base or not args.lora:
        raise SystemExit("--base and --lora are required (or use --self-test)")
    if not args.dry_run and not args.output:
        raise SystemExit("--output is required unless --dry-run is given")

    plan = build_plan(args.lora)
    scale = resolve_scale(plan, args.scale) * args.strength

    blocks = {k.split(".")[1] for k in plan.pairs_by_base if k.startswith("blocks.")}
    refiner = {k.split(".")[2] for k in plan.pairs_by_base if k.startswith("token_refiner.blocks.")}

    print(f"LoRA      : {args.lora}")
    print(f"metadata  : {plan.metadata}")
    print(f"rank      : {plan.rank}   alpha: {plan.alpha}")
    print(f"scale     : {scale:g}" + ("" if args.scale is None else "  (--scale override)"))
    print(f"pairs     : {plan.num_pairs} factor pairs -> {len(plan.pairs_by_base)} base tensors")
    print(f"coverage  : {len(blocks)} transformer blocks, {len(refiner)} token-refiner blocks")

    if args.dry_run:
        for base_key in sorted(plan.pairs_by_base)[:6]:
            slots = ",".join(sorted(p.slot for p in plan.pairs_by_base[base_key]))
            print(f"  {base_key}  <- [{slots}]")
        print("  ...")
        print("dry run: nothing written")
        return 0

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.compute_dtype]
    ratios = bake(
        base_dir=args.base,
        lora_path=args.lora,
        out_dir=args.output,
        plan=plan,
        scale=scale,
        compute_dtype=dtype,
    )

    values = sorted(r for _, r in ratios)
    worst = max(ratios, key=lambda item: item[1])
    print()
    print(f"patched {len(ratios)} tensors")
    print(f"||delta|| / ||W|| :  min={values[0]:.4f}  median={values[len(values) // 2]:.4f}  max={values[-1]:.4f}")
    print(f"  largest at {worst[0]}")
    print(
        "  A median far above ~1.0 usually means the scale is wrong "
        "(e.g. alpha/rank ignored for the 8-step adapter, which needs 0.0625)."
    )

    if args.partition_out:
        assemble_partition(args.base, args.output, args.partition_out)
        print(f"\npartition assembled: {args.partition_out}")
        print("point --model at that directory")

    return 0


if __name__ == "__main__":
    sys.exit(main())
