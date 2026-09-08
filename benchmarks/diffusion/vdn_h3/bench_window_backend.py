#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Time VDN_WINDOW_ATTN against the dense FlashAttention it replaces, at real shapes.

``bench_window_share.py`` estimated the win from a hand-written decomposition; this
runs the shipped backend, so it also pays for the plan lookup, the K/V gathers and the
variable-length batching. One card, one DiT block's worth of attention, random tensors.

    python benchmarks/diffusion/vdn_h3/bench_window_backend.py [--frames 37,47,107]
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import torch

from vllm_omni.platforms import current_omni_platform

HEADS, HEAD_DIM = 56, 128
TOKENS_PER_FRAME = 24 * 42  # 768p: latent 48x84 under patch (1, 2, 2)
TEXT_ROWS = 512


def audio_rows(latent_t: int) -> int:
    frames = 17 * ((latent_t - 2) // 5) + 5
    return int(round(frames / 24.0 * 40.0)) * 2


def timed(fn, iters: int) -> float:
    for _ in range(2):
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
    parser.add_argument("--frames", default="37,47,107")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    from vllm_omni.diffusion.attention.backends.abstract import (
        AttentionMetadata,
        VideoTokenLayout,
    )
    from vllm_omni.diffusion.attention.backends.vdn_window_attn import VDNWindowAttentionImpl

    print(f"device: {current_omni_platform.get_device_name()}  heads={HEADS} head_dim={HEAD_DIM}\n")
    impl = VDNWindowAttentionImpl(
        num_heads=HEADS,
        head_size=HEAD_DIM,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        qkv_layout="BSND",
    )

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
        shape = (1, used_len, HEADS, HEAD_DIM)
        q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))

        plan = impl._resolve_plan(q, metadata)
        dense = timed(partial(impl.dense_fallback.forward_cuda, q, k, v, metadata), args.iters)
        if plan is None:
            print(f"=== F={latent_t}: window covers the clip, backend stays dense ({dense:.1f} ms)\n")
        else:
            window = timed(partial(impl.forward_cuda, q, k, v, metadata), args.iters)
            peak = torch.accelerator.max_memory_allocated() / 1024**3
            print(f"=== F={latent_t} latent frames, {used_len} rows")
            print(f"    groups {len(plan.window_groups)}, {plan.kept_pair_fraction():.1%} of pairs kept")
            print(f"    dense  {dense:8.2f} ms   window {window:8.2f} ms   -> {dense / window:.2f}x")
            print(
                f"    x50 blocks {dense * 50 / 1e3:6.2f} -> {window * 50 / 1e3:6.2f} s/NFE"
                f"   (peak alloc {peak:.1f} GiB)\n"
            )
        del q, k, v
        current_omni_platform.empty_cache()
        torch.accelerator.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
