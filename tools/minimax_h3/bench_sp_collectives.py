# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cost of the collectives sequence parallelism would add to the VDN branch.

Run BEFORE touching the model. The design in
``docs/MiniMax-H3-VDN分支序列并行设计-2026-09-19.md`` trades ~10 GiB of full-sequence
activations for two extra collectives per block per step:

* an all-gather of the per-frame delta-rule factors, ``[F, H, d, d]`` fp32 x2 -- the only
  cross-frame dependency, and the reason the whole thing is feasible: it is sized by the
  state space, not by the token count
* a halo exchange of ``SHORT_CONV_KERNEL // 2`` frames for the branch's temporal conv

This box has no NVLink, and a previous model measured TP4 spending 96% of a step in
communication, so the benefit is not assumed -- it is measured against the generation time
the collectives would be added to.

Launch (inside the serving container, 4 ranks)::

    torchrun --standalone --nproc_per_node=4 tools/minimax_h3/bench_sp_collectives.py
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

# 15s @ 1088p, TP4: 362 pixel frames -> 112 latent frames, 2040 spatial tokens per frame,
# 56 heads / 4 ranks, head_dim 128. SHORT_CONV_KERNEL is 5, so the halo is 2 frames.
LATENT_FRAMES = 112
SPATIAL_TOKENS = 2040
LOCAL_HEADS = 14
HEAD_DIM = 128
HALO_FRAMES = 2
BLOCKS = 50
STEPS = 8


def _sync() -> None:
    torch.accelerator.synchronize()
    dist.barrier()


def time_op(fn, iterations: int, warmup: int = 10) -> float:
    """Median-free mean wall clock per call, in milliseconds, after a warmup."""
    for _ in range(warmup):
        fn()
    _sync()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.accelerator.set_device_index(rank % torch.accelerator.device_count())
    device = torch.device("cuda")

    local_frames = LATENT_FRAMES // world
    calls = BLOCKS * STEPS

    # 1. delta-rule factors: each rank owns its frames, every rank needs all of them.
    shard = torch.randn(local_frames, LOCAL_HEADS, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32)
    gathered = [torch.empty_like(shard) for _ in range(world)]
    shard_mib = shard.numel() * shard.element_size() / 1024**2

    def all_gather_factors() -> None:
        dist.all_gather(gathered, shard)

    # Two such tensors per block (transitions and injections).
    factor_ms = 2 * time_op(all_gather_factors, args.iterations)

    # 2. temporal-conv halo: HALO_FRAMES of tokens to each neighbour.
    halo = torch.randn(HALO_FRAMES, SPATIAL_TOKENS, LOCAL_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    halo_recv = [torch.empty_like(halo) for _ in range(world)]
    halo_mib = halo.numel() * halo.element_size() / 1024**2

    def all_gather_halo() -> None:
        dist.all_gather(halo_recv, halo)

    halo_ms = time_op(all_gather_halo, args.iterations)

    # 3. Reference point: the all-reduce the model ALREADY pays once per block, so the new
    #    cost can be read as a multiple of a collective the current design accepts.
    existing = torch.randn(
        LATENT_FRAMES * SPATIAL_TOKENS // world, LOCAL_HEADS * HEAD_DIM * 3, device=device, dtype=torch.bfloat16
    )
    existing_mib = existing.numel() * existing.element_size() / 1024**2

    def all_reduce_existing() -> None:
        dist.all_reduce(existing)

    existing_ms = time_op(all_reduce_existing, args.iterations)

    if rank == 0:
        added = factor_ms + halo_ms
        print(f"world={world}  latent_frames={LATENT_FRAMES}  local_frames={local_frames}")
        print(f"{'collective':34s} {'payload/rank':>13s} {'ms/call':>9s} {'s over 400 calls':>17s}")
        print("-" * 78)
        print(
            f"{'factors all-gather (x2/block)':34s} {shard_mib:11.1f} MiB {factor_ms:9.3f} "
            f"{factor_ms * calls / 1000:17.2f}"
        )
        print(f"{'temporal halo all-gather':34s} {halo_mib:11.1f} MiB {halo_ms:9.3f} {halo_ms * calls / 1000:17.2f}")
        print("-" * 78)
        print(f"{'ADDED BY SP':34s} {'':15s} {added:9.3f} {added * calls / 1000:17.2f}")
        print(
            f"{'(existing per-block all-reduce)':34s} {existing_mib:11.1f} MiB {existing_ms:9.3f} "
            f"{existing_ms * calls / 1000:17.2f}"
        )
        print()
        print("Compare the ADDED seconds against the 15s@1088p generation time (708 s single-pass).")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    raise SystemExit(main())
