#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""What VDN-H3 is worth per DiT block under the production TP4 shard.

The single-card benchmarks measure a ratio that tensor parallelism preserves -- TP
divides the attention heads and the GEMM widths by the same factor. It does not
preserve the *comm*: a TP block pays two all-reduces of the full ``[rows, hidden]``
activation, VDN removes none of that, and a term that only the denominator grows
pulls the speedup toward 1. On A100 PCIE, without NVLink, that term is not small.

Run under torchrun on one node::

    torchrun --nproc_per_node=4 benchmarks/diffusion/vdn_h3/bench_tp4_block.py --frames 107

Every quantity is measured at this rank's shard width, so the printed block time is
what one rank actually spends, and the all-reduces are timed in the same collective
group the model would use.
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial

import torch
import torch.distributed as dist

from vllm_omni.platforms import current_omni_platform

HIDDEN = 5376
TOTAL_HEADS = 56
HEAD_DIM = 128
FFN = 14336
TOKENS_PER_FRAME = 24 * 42
TEXT_ROWS = 512
BF16 = torch.bfloat16


def audio_rows(latent_t: int) -> int:
    frames = 17 * ((latent_t - 2) // 5) + 5
    return int(round(frames / 24.0 * 40.0)) * 2


def timed(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    current_omni_platform.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        current_omni_platform.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="107")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    current_omni_platform.set_device(rank)
    local_heads = TOTAL_HEADS // world
    local_inner = local_heads * HEAD_DIM
    local_ffn = FFN // world

    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout
    from vllm_omni.diffusion.attention.backends.vdn_window_attn import VDNWindowAttentionImpl

    impl = VDNWindowAttentionImpl(
        num_heads=local_heads,
        head_size=HEAD_DIM,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        qkv_layout="BSND",
    )

    def say(*parts):
        if rank == 0:
            print(*parts, flush=True)

    say(f"world={world}  {current_omni_platform.get_device_name()}  local_heads={local_heads}\n")

    for latent_t in (int(value) for value in args.frames.split(",")):
        prefix = TEXT_ROWS + audio_rows(latent_t)
        used_len = prefix + latent_t * TOKENS_PER_FRAME
        cu = torch.tensor([0, used_len], dtype=torch.int32, device="cuda")
        metadata = AttentionMetadata(
            video_layout=VideoTokenLayout(prefix_len=prefix, latent_grid=(latent_t, 24, 42)),
            extra={
                "cu_seqlens_q": cu,
                "cu_seqlens_k": cu,
                "max_seqlen_q": used_len,
                "max_seqlen_k": used_len,
                "valid_kv_length": used_len,
            },
        )

        shape = (1, used_len, local_heads, HEAD_DIM)
        q, k, v = (torch.randn(shape, device="cuda", dtype=BF16) for _ in range(3))
        dense_attn = timed(partial(impl.dense_fallback.forward_cuda, q, k, v, metadata), args.iters)
        window_attn = timed(partial(impl.forward_cuda, q, k, v, metadata), args.iters)
        del q, k, v
        current_omni_platform.empty_cache()

        # The two collectives a TP block runs: after the attention out-projection and
        # after the MLP down-projection. Both are the full activation, not the shard.
        activation = torch.randn(used_len, HIDDEN, device="cuda", dtype=BF16)
        all_reduce = timed(partial(dist.all_reduce, activation), args.iters) * 2

        gemms = 0.0
        for rows, in_features, out_features in (
            (used_len, HIDDEN, 3 * local_inner),  # qkv, column-parallel
            (used_len, local_inner, HIDDEN),  # out, row-parallel
            (used_len, HIDDEN, 2 * local_ffn),  # fused gate/up, column-parallel
            (used_len, local_ffn, HIDDEN),  # down, row-parallel
        ):
            x = torch.randn(rows, in_features, device="cuda", dtype=BF16)
            w = torch.randn(in_features, out_features, device="cuda", dtype=BF16)
            gemms += timed(partial(torch.mm, x, w), args.iters)
            del x, w
            current_omni_platform.empty_cache()

        # The linear branch, at this rank's head shard. Its GEMMs dominate; the scan is
        # naive PyTorch where VDN ships Triton, so it is reported separately.
        branch = 0.0
        video = latent_t * TOKENS_PER_FRAME
        for rows, in_features, out_features in (
            (video, local_inner, HIDDEN),  # to_out_linear, row-parallel
            (video, HIDDEN, local_heads),  # beta_proj
            (video, HIDDEN, local_heads),  # softmax_gate
            (video, HIDDEN, HEAD_DIM),  # output_gate down (replicated)
            (video, HEAD_DIM, local_inner),  # output_gate up
        ):
            x = torch.randn(rows, in_features, device="cuda", dtype=BF16)
            w = torch.randn(in_features, out_features, device="cuda", dtype=BF16)
            branch += timed(partial(torch.mm, x, w), args.iters)
            del x, w
            current_omni_platform.empty_cache()

        today = dense_attn + gemms + all_reduce
        vdn = window_attn + gemms + all_reduce + branch
        say(f"=== F={latent_t}  ({used_len} rows, {used_len * HIDDEN * 2 / 1024**2:.0f} MiB activation)")
        say(
            f"    attention   dense {dense_attn:7.2f} ms   window {window_attn:7.2f} ms  "
            f"({dense_attn / window_attn:.2f}x)"
        )
        say(f"    gemms       {gemms:7.2f} ms")
        say(f"    all-reduce  {all_reduce:7.2f} ms  (2 per block, VDN removes none)")
        say(f"    branch adds {branch:7.2f} ms  (GEMMs only; scan excluded)")
        say(f"    per block   today {today:7.2f} ms -> VDN {vdn:7.2f} ms   SPEEDUP {today / vdn:.2f}x")
        say(f"    x50 blocks  today {today * 50 / 1e3:6.2f} -> VDN {vdn * 50 / 1e3:6.2f} s/NFE")
        say(f"    comm is {all_reduce / today:.1%} of a block today, {all_reduce / vdn:.1%} under VDN\n")
        del activation
        current_omni_platform.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    main()
