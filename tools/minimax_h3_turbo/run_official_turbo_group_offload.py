#!/usr/bin/env python3
"""Run the upstream Turbo CLI with Diffusers group offload on one GPU.

The upstream FSDP2 path shards the text encoder and transformer, then replicates
all remaining modules on every rank.  MiniMax-H3 reaches ~39.47 GiB per A100-40G
while moving those modules and OOMs.  This wrapper changes only placement: it
keeps the upstream CLI, LoRA loader, pipeline call and result writer intact,
while applying the same public Diffusers group-offload hooks used by the
reproducible Base oracle.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
from diffusers import ComponentsManager
from diffusers.hooks import apply_group_offloading

try:
    from .lora_provenance import read_lora_checkpoint_metadata
except ImportError:  # direct `python tools/.../run_official_turbo_group_offload.py`
    from lora_provenance import read_lora_checkpoint_metadata

UPSTREAM_SCRIPT = Path(
    os.environ.get(
        "H3_TURBO_UPSTREAM_SCRIPT",
        "/nfs-output/h3_turbo_eval/inference_minimax_h3.py",
    )
)


def _load_upstream():
    spec = importlib.util.spec_from_file_location("minimax_h3_turbo_upstream", UPSTREAM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import upstream Turbo script: {UPSTREAM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    upstream = _load_upstream()

    # Keep components on CPU while the upstream loader constructs the exact
    # pipeline and fuses the exact LoRA.  Its ordinary component-level offload
    # cannot split the 62 GiB text encoder on an A100-40G.
    def _defer_component_offload(self, *args, **kwargs):  # noqa: ARG001
        return None

    ComponentsManager.enable_auto_cpu_offload = _defer_component_offload
    original_load_pipeline = upstream.load_pipeline

    def load_pipeline_with_group_offload(args, context, workflow, fsdp_mesh=None):
        if args.fsdp2 or args.no_cpu_offload:
            raise ValueError("group-offload oracle must run single-process with CPU offload enabled")
        pipe = original_load_pipeline(args, context, workflow, fsdp_mesh)
        _name, active_transformer = upstream.get_active_transformer(pipe, workflow)

        active_transformer.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        offload = {
            "onload_device": torch.device(args.device),
            "offload_device": torch.device("cpu"),
            "use_stream": True,
        }
        active_transformer.enable_group_offload(
            offload_type="block_level",
            num_blocks_per_group=1,
            **offload,
        )
        apply_group_offloading(
            pipe.text_encoder.model,
            offload_type="leaf_level",
            **offload,
        )
        pipe.vae.to(args.device)
        pipe.audio_vae.to(args.device)
        print(
            "Official Turbo oracle placement: text encoder leaf offload, "
            "active transformer block offload, VAE/audio VAE on device",
            flush=True,
        )
        return pipe

    upstream.load_pipeline = load_pipeline_with_group_offload
    sys.argv[1:] = resolve_lora_alpha_argv(sys.argv[1:])
    upstream.main()


def _option_values(argv: list[str], name: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(argv):
        if item == name:
            if index + 1 >= len(argv):
                raise ValueError(f"{name} requires a value")
            values.append(argv[index + 1])
        elif item.startswith(f"{name}="):
            values.append(item.split("=", 1)[1])
    return values


def resolve_lora_alpha_argv(argv: list[str]) -> list[str]:
    """Bind upstream ``--lora-alpha`` to the exact checkpoint metadata.

    The upstream CLI defaults this to 8 and applies it silently, while the
    released checkpoints do *not* agree on one value: at rank 128,
    ``fl2v_turbo_4step_v1.0_768p`` declares alpha 128 and the other two declare
    8. Since the merge is scaled by ``alpha / rank``, accepting the default for
    the 768p checkpoint applies its LoRA at 1/16 strength — no error, and the
    result looks like an ordinary under-distilled sample rather than a
    misconfiguration.

    An explicit value is accepted only when it agrees.  This wrapper is an
    oracle: silently overriding the checkpoint would make the comparison
    neither the released adapter nor a reproducible local variant.
    """
    paths = _option_values(argv, "--lora-path")
    if not paths:
        return list(argv)
    if len(paths) != 1:
        raise ValueError("the official Turbo oracle accepts exactly one --lora-path")
    resolved = Path(paths[0]).expanduser().resolve()
    metadata = read_lora_checkpoint_metadata(resolved, with_sha256=False)
    if not metadata.lora_alpha.is_integer():
        raise ValueError(
            f"upstream --lora-alpha accepts an integer, but {resolved.name} declares {metadata.lora_alpha!r}"
        )
    declared = int(metadata.lora_alpha)

    explicit = _option_values(argv, "--lora-alpha")
    if len(explicit) > 1:
        raise ValueError("the official Turbo oracle accepts at most one --lora-alpha")
    if explicit:
        try:
            requested = int(explicit[0])
        except ValueError as exc:
            raise ValueError(f"--lora-alpha must be an integer, got {explicit[0]!r}") from exc
        if requested != declared:
            raise ValueError(
                f"--lora-alpha={requested} disagrees with {resolved.name} metadata alpha={declared}; "
                "refusing a non-reproducible oracle run"
            )
        return list(argv)

    resolved_argv = [*argv, "--lora-alpha", str(declared)]
    print(f"Using LoRA alpha {declared} declared by {resolved.name}", flush=True)
    return resolved_argv


if __name__ == "__main__":
    main()
