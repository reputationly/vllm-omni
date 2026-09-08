#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure the ceiling on what VDN-H3's hybrid attention could buy us, per clip length.

VDN replaces H3's dense attention with a chunk-aligned window (chunk=5, radius=1 ->
every frame sees 3 whole VAE chunks = 15 latent frames) and adds a linear-attention
branch to carry what the window drops.  Everything else in a DiT block -- QKV/out
projections, the fused gate/up MLP, AdaLN -- is untouched, and the branch ADDS work.
So the achievable speedup is bounded by

    dense_attn + gemms                      (what we run today)
    ------------------------------------
    window_attn + gemms + branch            (what VDN would run)

and that bound collapses as the clip gets shorter, because the window is a FIXED 15
latent frames while the dense cost grows with F^2.  This script measures every term
on one GPU with random weights: no checkpoint, no NFS, no distributed setup.

TP does not change the ratio -- tensor parallel divides the attention heads and the
GEMM widths by the same factor -- so a single-card measurement is the right one, and
it is quoted per DiT block (H3 has 50 identical ones).

    python benchmarks/diffusion/vdn_h3/bench_window_share.py [--frames 37,47,102] [--iters 5]

The numbers are an upper bound on VDN's benefit, deliberately:
  * un-accelerated per-block overhead (AdaLN, RMSNorms, residuals) is excluded, and
    including it can only dilute the speedup;
  * the linear branch's scan is timed as naive PyTorch, where VDN ships Triton, so
    it is reported as its own line and the verdict is given with and without it.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial

import torch

from vllm_omni.platforms import current_omni_platform

# H3 DiT geometry (h3-base/transformer/config.json).
HIDDEN = 5376
NUM_HEADS = 56
HEAD_DIM = 128
INNER = NUM_HEADS * HEAD_DIM  # 7168
FFN = 14336
# 768p: latent 48x84, patch (1,2,2) -> 24x42 = 1008 patched tokens per latent frame.
GRID_H, GRID_W = 24, 42
TOKENS_PER_FRAME = GRID_H * GRID_W
# VDN's window, from stage-dmd-step-250/model_spec.json.
CHUNK, RADIUS = 5, 1
LINEAR_HEAD_DIM = 128
CONV_K = 5
TEXT_ROWS = 512  # a rewritten H3-Context-IR prompt through Qwen3-VL, order of magnitude
BF16 = torch.bfloat16


def frames_for_latent_t(latent_t: int) -> int:
    """Inverse of vllm_omni ... time_request._video_latent_t."""
    return 17 * ((latent_t - 2) // 5) + 5


def audio_rows(latent_t: int) -> int:
    """H3 packs 40 Hz audio latents, 2 channels, as their own rows."""
    return int(round(frames_for_latent_t(latent_t) / 24.0 * 40.0)) * 2


def window_bounds(num_frames: int) -> list[tuple[int, int]]:
    """VDN src/models/softmax_attention/window.py, chunk-aligned mode."""
    return [(((t // CHUNK) - RADIUS) * CHUNK, ((t // CHUNK) + RADIUS + 1) * CHUNK - 1) for t in range(num_frames)]


def _fa():
    from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_func

    if flash_attn_func is None:
        raise RuntimeError("no flash_attn_func resolved in this image")
    return flash_attn_func


def _timed(fn, iters: int, warmup: int = 2) -> float:
    """Median-of-iters wall time in milliseconds, device-synchronised."""
    for _ in range(warmup):
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


def bench_dense_attention(total: int, iters: int) -> float:
    fa = _fa()
    q, k, v = (torch.randn(1, total, NUM_HEADS, HEAD_DIM, device="cuda", dtype=BF16) for _ in range(3))
    ms = _timed(partial(fa, q, k, v, causal=False), iters)
    del q, k, v
    current_omni_platform.empty_cache()
    return ms


def bench_window_attention(num_frames: int, prefix: int, iters: int) -> float:
    """VDN's `decomposed` plan: the window mask as a union of dense calls.

    Query rows split into groups with an identical kept-KV set -- the globals and the
    two anchor frames against the full sequence, then one group per VAE chunk against
    [globals | its 3-chunk window | the anchor frames]. The KV gather is included: it
    is real work the dense path never does.
    """
    fa = _fa()
    total = prefix + num_frames * TOKENS_PER_FRAME
    q, k, v = (torch.randn(1, total, NUM_HEADS, HEAD_DIM, device="cuda", dtype=BF16) for _ in range(3))
    bounds = window_bounds(num_frames)
    anchors = (0, num_frames - 1)

    def frame_slice(tensor, lo, hi):
        start = prefix + lo * TOKENS_PER_FRAME
        stop = prefix + (hi + 1) * TOKENS_PER_FRAME
        return tensor[:, start:stop]

    def run(q, k, v):
        # anchor_frames="both": frames 0 and F-1 are dense ROWS as well as columns.
        dense_q = torch.cat(
            [q[:, :prefix]] + [frame_slice(q, f, f) for f in anchors],
            dim=1,
        )
        fa(dense_q, k, v, causal=False)

        for chunk_start in range(0, num_frames, CHUNK):
            chunk_stop = min(chunk_start + CHUNK - 1, num_frames - 1)
            rows = [f for f in range(chunk_start, chunk_stop + 1) if f not in anchors]
            if not rows:
                continue
            lo = max(bounds[chunk_start][0], 0)
            hi = min(bounds[chunk_start][1], num_frames - 1)
            extra = [f for f in anchors if not lo <= f <= hi]
            gathered_k = torch.cat(
                [k[:, :prefix], frame_slice(k, lo, hi)] + [frame_slice(k, f, f) for f in extra], dim=1
            )
            gathered_v = torch.cat(
                [v[:, :prefix], frame_slice(v, lo, hi)] + [frame_slice(v, f, f) for f in extra], dim=1
            )
            fa(frame_slice(q, rows[0], rows[-1]), gathered_k, gathered_v, causal=False)

    ms = _timed(partial(run, q, k, v), iters)
    del q, k, v
    current_omni_platform.empty_cache()
    return ms


def bench_block_gemms(total: int, iters: int) -> dict[str, float]:
    """The projections and MLP a DiT block runs either way. fc1 is the fused gate/up."""
    out: dict[str, float] = {}
    for name, rows, in_features, out_features in (
        ("qkv_proj", total, HIDDEN, 3 * INNER),
        ("out_proj", total, INNER, HIDDEN),
        ("mlp_fc1", total, HIDDEN, 2 * FFN),
        ("mlp_fc2", total, FFN, HIDDEN),
    ):
        x = torch.randn(rows, in_features, device="cuda", dtype=BF16)
        w = torch.randn(in_features, out_features, device="cuda", dtype=BF16)
        out[name] = _timed(partial(torch.mm, x, w), iters)
        del x, w
        current_omni_platform.empty_cache()
    return out


def bench_branch(num_frames: int, iters: int) -> dict[str, float]:
    """What the linear branch ADDS per block: its GEMMs, its convs, its scan."""
    video = num_frames * TOKENS_PER_FRAME
    out: dict[str, float] = {}

    # --- GEMMs (to_out_linear dominates: [7168, 5376], 3.85 of the 4.28 GB artifact)
    gemms = {
        "to_out_linear": (video, INNER, HIDDEN),
        "beta_proj": (video, HIDDEN, NUM_HEADS),
        "softmax_gate": (video, HIDDEN, NUM_HEADS),
        "output_gate_down": (video, HIDDEN, LINEAR_HEAD_DIM),
        "output_gate_up": (video, LINEAR_HEAD_DIM, INNER),
    }
    total_ms = 0.0
    for name, (rows, in_features, out_features) in gemms.items():
        x = torch.randn(rows, in_features, device="cuda", dtype=BF16)
        w = torch.randn(in_features, out_features, device="cuda", dtype=BF16)
        total_ms += _timed(partial(torch.mm, x, w), iters)
        del x, w
        current_omni_platform.empty_cache()
    out["branch_gemms"] = total_ms

    # --- separable depthwise conv on k and v: 5x5 spatial then a 5-tap temporal
    volume = torch.randn(num_frames, INNER, GRID_H, GRID_W, device="cuda", dtype=BF16)
    w_sp = torch.randn(INNER, 1, CONV_K, CONV_K, device="cuda", dtype=BF16)
    w_tm = torch.randn(INNER, 1, CONV_K, device="cuda", dtype=BF16)

    def conv_once(volume, w_sp, w_tm):
        spatial = torch.nn.functional.conv2d(volume, w_sp, padding=CONV_K // 2, groups=INNER)
        rows = spatial.permute(0, 2, 3, 1).reshape(num_frames, TOKENS_PER_FRAME * INNER)
        temporal = rows.reshape(num_frames, -1).transpose(0, 1).reshape(-1, 1, num_frames)
        torch.nn.functional.conv1d(temporal[: INNER * 8], w_tm[:1].expand(1, 1, CONV_K), padding=CONV_K // 2)

    out["branch_conv"] = _timed(partial(conv_once, volume, w_sp, w_tm), iters) * 2  # k and v
    del volume, w_sp, w_tm
    current_omni_platform.empty_cache()

    # --- feature prep: SiLU on q/k/v plus L2-norm on q/k, over the video rows
    feats = torch.randn(video, INNER, device="cuda", dtype=BF16)

    def features(feats):
        activated = torch.nn.functional.silu(feats)
        torch.nn.functional.normalize(activated.view(video, NUM_HEADS, HEAD_DIM), dim=-1)

    out["branch_features"] = _timed(partial(features, feats), iters) * 3
    del feats
    current_omni_platform.empty_cache()

    # --- the bidirectional frame scan, naive PyTorch (VDN ships Triton: overstated)
    key = torch.randn(NUM_HEADS, TOKENS_PER_FRAME, HEAD_DIM, device="cuda", dtype=BF16)
    value = torch.randn(NUM_HEADS, TOKENS_PER_FRAME, HEAD_DIM, device="cuda", dtype=BF16)
    query = torch.randn(NUM_HEADS, TOKENS_PER_FRAME, HEAD_DIM, device="cuda", dtype=BF16)
    alpha = torch.rand(NUM_HEADS, HEAD_DIM, 1, device="cuda", dtype=torch.float32)

    # Explicit parameters rather than closure capture: the tensors are freed right after,
    # and a deferred body reading names the enclosing scope has since deleted is exactly
    # the late-binding hazard ruff's F821 flags here.
    def scan(key, value, query, alpha):
        state = torch.zeros(NUM_HEADS, HEAD_DIM, HEAD_DIM, device="cuda", dtype=torch.float32)
        for _ in range(num_frames):
            a = torch.bmm(key.transpose(1, 2), key).float()
            b = torch.bmm(value.transpose(1, 2), key).float()
            eye = torch.eye(HEAD_DIM, device="cuda").expand_as(a)
            injection = torch.linalg.solve(eye + a, b)
            state = state * alpha + injection
            torch.bmm(query, state.to(BF16))

    out["branch_scan"] = (
        _timed(partial(scan, key, value, query, alpha), max(1, iters // 2), warmup=1) * 2
    )  # both directions
    del key, value, query, alpha
    current_omni_platform.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="37,47,102", help="latent frame counts (F)")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--json", default="", help="also write the raw record here")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device: {current_omni_platform.get_device_name()}  torch {torch.__version__}\n")

    records = []
    for latent_t in (int(value) for value in args.frames.split(",")):
        prefix = TEXT_ROWS + audio_rows(latent_t)
        total = prefix + latent_t * TOKENS_PER_FRAME
        seconds = frames_for_latent_t(latent_t) / 24.0
        covered = min((2 * RADIUS + 1) * CHUNK, latent_t)

        dense = bench_dense_attention(total, args.iters)
        window = bench_window_attention(latent_t, prefix, args.iters)
        gemms = bench_block_gemms(total, args.iters)
        branch = bench_branch(latent_t, args.iters)

        gemm_ms = sum(gemms.values())
        branch_ms = sum(branch.values())
        branch_no_scan = branch_ms - branch["branch_scan"]
        today = dense + gemm_ms
        vdn = window + gemm_ms + branch_ms
        vdn_ideal = window + gemm_ms + branch_no_scan

        record = {
            "latent_frames": latent_t,
            "seconds": round(seconds, 2),
            "video_rows": latent_t * TOKENS_PER_FRAME,
            "prefix_rows": prefix,
            "window_density": round(covered / latent_t, 3),
            "dense_attn_ms": round(dense, 3),
            "window_attn_ms": round(window, 3),
            "gemms_ms": {name: round(value, 3) for name, value in gemms.items()},
            "branch_ms": {name: round(value, 3) for name, value in branch.items()},
            "attn_share_today": round(dense / today, 3),
            "block_ms_today": round(today, 3),
            "block_ms_vdn": round(vdn, 3),
            "speedup": round(today / vdn, 3),
            "speedup_ideal_scan": round(today / vdn_ideal, 3),
        }
        records.append(record)

        print(f"=== F={latent_t} latent frames ({seconds:.1f} s, {total} packed rows)")
        print(f"    window covers {covered}/{latent_t} frames = {covered / latent_t:.1%}")
        print(
            f"    attention   dense {dense:8.2f} ms   window {window:8.2f} ms   "
            f"({dense / window:.2f}x on the kernel alone)"
        )
        print(
            f"    gemms       {gemm_ms:8.2f} ms   " + "  ".join(f"{name}={value:.1f}" for name, value in gemms.items())
        )
        print(
            f"    branch adds {branch_ms:8.2f} ms   "
            + "  ".join(f"{name.replace('branch_', '')}={value:.1f}" for name, value in branch.items())
        )
        print(f"    attention is {dense / today:.1%} of a block today")
        print(
            f"    per block   today {today:8.2f} ms -> VDN {vdn:8.2f} ms   "
            f"SPEEDUP {today / vdn:.2f}x  (ideal scan: {today / vdn_ideal:.2f}x)"
        )
        print(f"    x50 blocks  today {today * 50 / 1e3:6.2f} s/NFE -> VDN {vdn * 50 / 1e3:6.2f} s/NFE\n")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(records, handle, indent=2)
        print(f"raw record -> {args.json}")


if __name__ == "__main__":
    main()
