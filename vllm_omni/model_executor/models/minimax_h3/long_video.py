# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared long-video request limits for encoder and diffusion entry points."""

from collections.abc import Mapping
from typing import Any

from vllm_omni.errors import OmniClientError

# The ceiling for an ordinary request, i.e. one that did not ask for
# ``long_video``. It is the product's duration limit, not a model limit: the
# released workflow emits 5..15 s and the released checkpoint has been measured
# at 0..15 s, but 4x A100-40G serves 20 s (481 frames) at 28-29 GiB/card with
# ~11 GiB headroom per card, so 20 s is what this deployment admits
# (2026-11-19, measured on the Pika BF16 partitions at 1344x768:
# 437-458 s wall, denoise 393-407 s, peak 28.3-29.1 GiB/card).
MINIMAX_H3_MAX_STANDARD_SECONDS = 20.0
MINIMAX_H3_MAX_FULL_SECONDS = 30.0
MINIMAX_H3_MAX_CONTINUATION_SECONDS = 300.0


def resolve_long_video_mode(extra: Mapping[str, Any], task: str) -> str:
    long_video = extra.get("long_video", False)
    if not isinstance(long_video, bool):
        raise OmniClientError("MiniMax H3 long_video must be a boolean")
    mode = extra.get("long_video_mode", "continuation" if long_video and task == "ref2va" else "full")
    if mode not in ("full", "continuation"):
        raise OmniClientError("MiniMax H3 long_video_mode must be full or continuation")
    if mode == "continuation" and task != "ref2va":
        raise OmniClientError("MiniMax H3 continuation requires Ref2VA request execution (not step execution)")
    return mode


def max_output_seconds(extra: Mapping[str, Any], task: str) -> float:
    mode = resolve_long_video_mode(extra, task)
    if not extra.get("long_video", False):
        return MINIMAX_H3_MAX_STANDARD_SECONDS
    return MINIMAX_H3_MAX_CONTINUATION_SECONDS if mode == "continuation" else MINIMAX_H3_MAX_FULL_SECONDS


def validate_encoded_frame_limit(extra: Mapping[str, Any], task: str, num_frames: int) -> None:
    """Apply the same cap to externally encoded requests, allowing grid rounding."""
    seconds = max_output_seconds(extra, task)
    requested_max = int(seconds * 24)
    aligned_max = ((requested_max - 5 + 16) // 17) * 17 + 5
    if num_frames > aligned_max:
        raise OmniClientError(
            f"MiniMax H3 output duration exceeds the {seconds:g}-second request limit "
            f"({aligned_max} frames after alignment); got {num_frames} frames"
        )
