#!/usr/bin/env python3
"""Rewrite an AdaLN-pruned Diffusers transformer into the vLLM partition layout.

vLLM-Omni loads the pruned Diffusers shards directly — ``load_weights`` fuses
q/k/v and swaps the SwiGLU halves on the way in — so this conversion buys
nothing for BF16 serving.  It exists for the offline Int8 pass:
``vllm_omni/quantization/tools/quantize_minimax_h3_int8.py`` quantizes tensors
by their *partition* names (``blocks.N.attn.qkv_proj.weight``,
``blocks.N.mlp.fc1.weight``) and reads ``model.safetensors.index.json``, neither
of which a Diffusers checkpoint has.

The rewrite is pure layout: three renames-and-repacks, no arithmetic, no casts.

1. ``attn.to_q/to_k/to_v`` fuse into ``attn.qkv_proj`` in grouped per-head
   order — head g contributes ``[q_g, k_g, v_g]`` — the inverse of
   ``_reorder_grouped_qkv_to_qkv`` the loader applies to released partitions.
2. ``ff.net.0.proj`` stores ``[up, gate]``; the partition stores ``[gate, up]``,
   so its two halves are swapped.
3. Everything else is renamed only, keeping its bytes and dtype — including the
   FP32 pruning buffers (``time_embedder.table``, ``adaln_basis``,
   ``adaln_mean``, ``*.adaln_proj.folded_bias``), which the model asserts are
   still FP32 after load.

Because nothing is computed, the result is checkable bit-for-bit:

    python3 tools/minimax_h3_parity/verify_checkpoint_conversion.py \
        --diffusers <pruned diffusers transformer> \
        --partition <this tool's --output> --heads 56

Usage
-----
    python3 convert_pruned_to_partition.py \
        --src /nfs-models/.../MiniMax-H3-Ref2VA-Pruned-r8-Turbo4-BF16-transformer \
        --output /nfs-models/.../MiniMax-H3-Ref2VA-Pruned-r8-Turbo4-BF16-partition/transformer
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    sys.exit(f"missing dependency: {exc}. Run inside the vllm-omni image or a venv with torch+safetensors.")

DIFFUSERS_INDEX = "diffusion_pytorch_model.safetensors.index.json"
PARTITION_INDEX = "model.safetensors.index.json"

# Diffusers name -> partition name.  Kept byte-identical to
# ``minimax_h3_transformer._DIFFUSERS_NAME_RENAMES`` (a unit test asserts the
# two tables are equal) so a checkpoint written here is named exactly the way
# the loader's partition path expects.
NAME_RENAMES = (
    (r"^audio_proj_in\.", "audio_patch_proj."),
    (r"^audio_proj_out\.", "final_layer.audio_out."),
    (r"^context_embedder\.", "condition_proj."),
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

# Diffusers field name -> the canonical partition name for the same value.  The
# loader's ``MiniMaxH3DiTArchConfig.from_mapping`` accepts both spellings, but
# the released partitions are written in the canonical one and the Int8 tool
# copies this config verbatim, so publish what the other partitions publish.
CONFIG_ALIASES = {
    "num_refiner_layers": "token_refiner_num_layers",
    "ffn_dim": "ffn_hidden_size",
    "in_channels": "latents_dim",
    "audio_in_channels": "audio_latents_dim",
    "freq_dim": "timestep_input_dim",
    "time_embed_hidden_dim": "time_embed_hidden_size",
    "rope_freq_dim": "rope_inv_freq_len",
}

CONFIG_FIELDS = (
    "hidden_size",
    "num_layers",
    "token_refiner_num_layers",
    "num_attention_heads",
    "attention_head_dim",
    "ffn_hidden_size",
    "latents_dim",
    "audio_latents_dim",
    "patch_size",
    "text_dim",
    "timestep_input_dim",
    "time_embed_hidden_size",
    "time_embed_dim",
    "adaln_out_features",
    "final_adaln_out_features",
    "rope_inv_freq_len",
    "norm_eps",
    "qk_norm_eps",
    "final_norm_eps",
    # Present only on pruned checkpoints; dropping either one turns the
    # checkpoint back into an unloadable "released" one.
    "adaln_rank",
    "time_table_size",
)

# FP32 pruning buffers.  Named here only to assert they survive untouched: a
# silent BF16 cast would trip ``post_load_weights`` at serve time instead.
FP32_BUFFERS = ("time_embedder.table", "adaln_basis", "adaln_mean")


def partition_name(name: str) -> str:
    for pattern, replacement in NAME_RENAMES:
        name = re.sub(pattern, replacement, name)
    return name


def fuse_grouped_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
) -> torch.Tensor:
    """Pack q/k/v as the released partition does: head-major ``[q_g, k_g, v_g]``.

    The inverse of ``_reorder_grouped_qkv_to_qkv(..., heads_per_group=1)``.
    Plain concatenation produces the same shape, which is why the parity script
    identifies the order by value rather than trusting the dimensions.
    """
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"q/k/v must share a shape, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    rows = q.shape[0]
    if num_heads <= 0 or rows % num_heads:
        raise ValueError(f"{rows} rows do not split into {num_heads} heads")
    head_dim = rows // num_heads
    trailing = q.shape[1:]
    grouped = torch.stack(
        [
            q.reshape(num_heads, head_dim, *trailing),
            k.reshape(num_heads, head_dim, *trailing),
            v.reshape(num_heads, head_dim, *trailing),
        ],
        dim=1,
    )
    return grouped.reshape(3 * rows, *trailing)


def swap_swiglu_halves(fused: torch.Tensor) -> torch.Tensor:
    """``[up, gate]`` -> ``[gate, up]``.

    Both halves have identical shape, so getting this backwards swaps the gate
    with the value it gates: a model that still runs and still produces
    plausible video.  Only a bit-level comparison catches it.
    """
    rows = fused.shape[0]
    if rows % 2:
        raise ValueError(f"fc1 rows must split evenly into gate/up, got {tuple(fused.shape)}")
    half = rows // 2
    return torch.cat([fused[half:], fused[:half]], dim=0)


class _Shards:
    """Random access to a sharded safetensors checkpoint."""

    def __init__(self, directory: Path, weight_map: dict[str, str]):
        self._directory = directory
        self._weight_map = weight_map
        self._open: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._open.get(shard)
        if handle is None:
            handle = safe_open(str(self._directory / shard), framework="pt", device="cpu")
            self._open[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        self._open.clear()


def build_plan(weight_map: dict[str, str]) -> list[dict]:
    """One entry per output tensor: its name, its sources, and its transform.

    Built from names alone and then executed verbatim, so an unmapped or
    half-present group is an error here rather than a missing weight at load.
    """
    qkv_groups: dict[tuple[str, str], dict[str, str]] = {}
    plan: list[dict] = []

    for name in sorted(weight_map):
        match = _QKV.match(name)
        if match is None:
            continue
        prefix, part, suffix = match.groups()
        qkv_groups.setdefault((prefix, suffix), {})[part] = name

    for (prefix, suffix), parts in sorted(qkv_groups.items()):
        if set(parts) != {"q", "k", "v"}:
            raise ValueError(f"{prefix}.attn has an incomplete q/k/v set for .{suffix}: {sorted(parts)}")
        plan.append(
            {
                "target": partition_name(f"{prefix}.attn.qkv_proj.{suffix}"),
                "kind": "qkv",
                "sources": [parts["q"], parts["k"], parts["v"]],
            }
        )

    consumed = {name for group in qkv_groups.values() for name in group.values()}
    for name in sorted(set(weight_map) - consumed):
        target = partition_name(name)
        plan.append(
            {
                "target": target,
                "kind": "fc1" if target.endswith(".mlp.fc1.weight") else "copy",
                "sources": [name],
            }
        )

    targets = [entry["target"] for entry in plan]
    duplicates = sorted({target for target in targets if targets.count(target) > 1})
    if duplicates:
        raise ValueError(f"rename collision: {duplicates}")
    return plan


def convert_config(source: dict) -> dict:
    """Canonical partition config: aliases resolved, Diffusers plumbing dropped."""
    normalized = dict(source)
    for alias, canonical in CONFIG_ALIASES.items():
        if canonical not in normalized and alias in normalized:
            normalized[canonical] = normalized.pop(alias)
    hidden_size = int(normalized["hidden_size"])
    normalized.setdefault("adaln_out_features", 18 * hidden_size)
    normalized.setdefault("final_adaln_out_features", 2 * hidden_size)

    for required in ("adaln_rank", "time_table_size"):
        if required not in normalized:
            raise ValueError(f"pruned config is missing {required!r}; is this a released checkpoint?")

    config = {"_class_name": "MiniMaxH3DiTModel"}
    if "_diffusers_version" in source:
        config["_diffusers_version"] = source["_diffusers_version"]
    missing = [field for field in CONFIG_FIELDS if field not in normalized]
    if missing:
        raise ValueError(f"pruned config is missing fields: {missing}")
    config.update({field: normalized[field] for field in CONFIG_FIELDS})
    # ``auto_map`` points at modeling_minimax_h3_pruned.py, whose module expects
    # the Diffusers names this tool just rewrote.  Leaving it would advertise a
    # class that cannot load these shards.
    return config


def convert(*, src: Path, output: Path, num_heads: int | None = None, dry_run: bool = False) -> dict:
    src = src.expanduser().resolve()
    index_path = src / DIFFUSERS_INDEX
    config_path = src / "config.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"no Diffusers index at {index_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"no config at {config_path}")
    if output.exists() and not dry_run:
        raise FileExistsError(f"refusing to modify existing output: {output}")

    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = convert_config(source_config)
    heads = num_heads if num_heads is not None else int(config["num_attention_heads"])

    weight_map: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    if not weight_map:
        raise ValueError(f"empty weight map in {index_path}")
    plan = build_plan(weight_map)

    # Keep the source sharding: each output tensor lands in the renamed shard of
    # its first source, so shard sizes stay in the range the writer was sized for.
    source_shards = sorted(set(weight_map.values()))
    shard_names = {
        shard: f"model-{position:05d}-of-{len(source_shards):05d}.safetensors"
        for position, shard in enumerate(source_shards, 1)
    }

    per_shard: dict[str, list[dict]] = {name: [] for name in shard_names.values()}
    for entry in plan:
        per_shard[shard_names[weight_map[entry["sources"][0]]]].append(entry)

    shards = _Shards(src, weight_map)
    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    started = time.monotonic()
    if not dry_run:
        output.mkdir(parents=True)

    for position, shard in enumerate(sorted(per_shard), 1):
        tensors: dict[str, torch.Tensor] = {}
        for entry in per_shard[shard]:
            if entry["kind"] == "qkv":
                q, k, v = (shards.get(name) for name in entry["sources"])
                tensor = fuse_grouped_qkv(q, k, v, num_heads=heads)
            elif entry["kind"] == "fc1":
                tensor = swap_swiglu_halves(shards.get(entry["sources"][0]))
            else:
                tensor = shards.get(entry["sources"][0])
            if entry["target"] in FP32_BUFFERS or entry["target"].endswith(".adaln_proj.folded_bias"):
                if tensor.dtype != torch.float32:
                    raise ValueError(f"{entry['target']} must stay fp32, source is {tensor.dtype}")
            tensors[entry["target"]] = tensor
            new_weight_map[entry["target"]] = shard
            total_bytes += tensor.nelement() * tensor.element_size()
        if dry_run:
            print(f"  [{position}/{len(per_shard)}] {shard}: {len(tensors)} tensors (dry-run, not written)")
        else:
            save_file(tensors, str(output / shard), metadata={"format": "pt"})
            print(
                f"  [{position}/{len(per_shard)}] {shard}: {len(tensors)} tensors written "
                f"({time.monotonic() - started:.0f}s)"
            )
        shards.close()
        del tensors

    report = {
        "source_tensors": len(weight_map),
        "partition_tensors": len(new_weight_map),
        "shards": len(per_shard),
        "total_size": total_bytes,
    }
    if dry_run:
        print(json.dumps(report, indent=1))
        print("\nconfig.json that would be written:")
        print(json.dumps(config, indent=1))
        return report

    (output / PARTITION_INDEX).write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    # Pruning/fusion provenance travels with the weights; the Int8 pass copies
    # the same class of extras forward.
    for extra in sorted(src.iterdir()):
        if (
            extra.is_file()
            and extra.suffix not in (".safetensors",)
            and extra.name not in (DIFFUSERS_INDEX, "config.json")
        ):
            shutil.copy2(extra, output / extra.name)

    print(json.dumps(report, indent=1))
    print(f"\nwrote {output} in {time.monotonic() - started:.0f}s")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="pruned Diffusers transformer directory")
    parser.add_argument("--output", type=Path, required=True, help="partition transformer directory to create")
    parser.add_argument("--num-heads", type=int, default=None, help="override config's num_attention_heads")
    parser.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    convert(**vars(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
