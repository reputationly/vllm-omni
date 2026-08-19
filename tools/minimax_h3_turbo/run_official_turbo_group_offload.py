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
from pathlib import Path

import torch
from diffusers import ComponentsManager
from diffusers.hooks import apply_group_offloading

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
    upstream.main()


if __name__ == "__main__":
    main()
