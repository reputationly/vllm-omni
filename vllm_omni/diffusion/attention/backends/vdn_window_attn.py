# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VDN-H3's chunk-aligned window softmax, as a union of dense FlashAttention calls.

VDN-H3 (``OpenVDN/vdn-minimax-h3``) restricts video-to-video attention to a window of
whole VAE chunks -- with the released ``chunk=5, radius=1`` every latent frame sees 15
frames -- while text, audio and reference rows stay dense in both directions and the
two anchor frames stay dense as rows and columns. A linear-attention branch (loaded
from the VDN checkpoint) carries what the window drops.

The mask is not arbitrary sparsity, so it needs no sparse kernel: every kept pair lies
in one of a few dense rectangles, because a chunk-aligned window is identical for every
frame in a chunk. ``vdn_window.build_window_plan`` compiles it into query groups over
key rectangles and this backend runs each group as an ordinary variable-length
FlashAttention call. Each query's softmax still spans exactly its kept set in one pass,
so the result differs from a masked kernel only by bf16 reduction order.

Measured on one A100-PCIE-40G at the 15 s / 768p production shape (F=107 latent frames,
1008 tokens per frame, 109,574 packed rows), per DiT block: dense 1863 ms -> windowed
394 ms, 4.73x on the attention alone, where attention is 82.5% of a block. The win
scales with clip length because the window is a fixed 15 frames while the dense cost
grows with F^2 -- at 5 s (F=37) it is only 1.95x, which is not enough to pay for the
branch. See ``benchmarks/diffusion/vdn_h3/bench_window_share.py``.

This backend is only correct for weights trained under the same window. On a dense H3
checkpoint it is not an optimisation, it is a different model; the VDN checkpoint
loader is what selects it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.models.minimax_h3.vdn_window import (
    VDN_ANCHOR_FRAMES,
    VDN_CHUNK,
    VDN_RADIUS,
    VDNWindowPlan,
    build_window_plan,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# This backend reads the sequence off dim 1, which only ``BSND`` guarantees.
_INPUT_LAYOUT = "BSND"

# How many gathered key rows one variable-length call may hold, expressed as a byte
# budget so it self-adjusts to the head shard: under TP4 a rank carries a quarter of the
# heads and can afford four times the rows. Batching whole groups into one call cuts
# kernel launches; letting a batch grow without bound instead materialises many groups'
# gathered K/V at once.
#
# 512 MiB is the setting every timing number in this work was taken at. Lowering it was
# tried against the fl2va/ref2va OOM at 768p/15 s and changed nothing -- the two runs
# failed on byte-identical allocations (1.48 and 1.24 GiB) at 128 MiB as at 512 MiB,
# because the allocation that fails is a full-sequence activation (~110k rows x 5376
# bf16), not a gather. That shape simply does not fit on 40 GB cards with reference
# media; the production answer is INT8/pruned weights, not a smaller gather.
# Overridable so the trade (peak vs kernel launches) stays measurable.
_KV_GATHER_BUDGET_BYTES = int(os.environ.get("VLLM_OMNI_VDN_KV_GATHER_BUDGET_MB", "512")) << 20


@dataclass(frozen=True)
class VDNWindowConfig:
    """The window this layer runs, straight from the checkpoint's ``model_spec.json``.

    These are checkpoint semantics rather than tunables -- the branch was trained
    against exactly this partition of the sequence -- so the defaults are the released
    values and ``backend_kwargs`` only exists so a future checkpoint can state its own.
    """

    chunk: int = VDN_CHUNK
    radius: int = VDN_RADIUS
    anchor_frames: str = VDN_ANCHOR_FRAMES

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict | None) -> VDNWindowConfig:
        bk = backend_kwargs or {}
        return cls(
            chunk=int(bk.get("vdn_chunk", VDN_CHUNK)),
            radius=int(bk.get("vdn_radius", VDN_RADIUS)),
            anchor_frames=str(bk.get("vdn_anchor_frames", VDN_ANCHOR_FRAMES)),
        )


class VDNWindowAttentionBackend(AttentionBackend):
    supported_platforms: tuple[str, ...] = ("cuda",)

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        """The window plan reads the packed metadata and never looks at attn_mask.

        ``_resolve_plan`` takes the used length from ``max_seqlen_q``/``valid_kv_length``
        and builds groups only over rows below it; everything at or past that length is
        alignment padding, belongs to no group, and is written as zeros. Declaring this
        is what stops the producer from materialising a padding mask that this backend
        would then be handed and rejected for -- H3 raises "does not support attn_mask"
        rather than running masked, so without it every packed request with tail padding
        fails at serve time while every unit test passes.

        CUDA only, like the other packed-mask-free backends: the fallbacks on other
        platforms hand the tensors to SDPA, which reads attn_mask and nothing else, so
        the pad rows would be attended as real keys.
        """
        return current_omni_platform.is_cuda()

    @classmethod
    def validate_available(cls) -> None:
        from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func

        if flash_attn_varlen_func is None:
            raise ValueError(
                "VDN_WINDOW_ATTN runs its window groups as variable-length FlashAttention "
                "calls and no flash_attn_varlen_func is available in this environment."
            )

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "VDN_WINDOW_ATTN"

    @staticmethod
    def get_impl_cls() -> type[VDNWindowAttentionImpl]:
        return VDNWindowAttentionImpl


class VDNWindowAttentionImpl(AttentionImpl):
    """Chunk-aligned window attention over a packed multimodal sequence."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.qkv_layout = qkv_layout
        self.window = VDNWindowConfig.from_backend_kwargs(backend_kwargs)

        if causal:
            raise ValueError(
                "VDN_WINDOW_ATTN is a bidirectional video window and cannot express a causal "
                "mask. Select FLASH_ATTN for causal roles."
            )
        self._validate_parallel_config()

        self.dense_fallback = FlashAttentionBackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            qkv_layout=qkv_layout,
        )

    def _validate_parallel_config(self) -> None:
        config = get_current_diffusion_config_or_none()
        parallel_config = getattr(config, "parallel_config", None)
        if getattr(parallel_config, "ring_degree", 1) > 1:
            # Ring hands each rank a slice of the sequence, so the plan's row indices --
            # which are global offsets into the packed document -- would address the
            # wrong rows. Ulysses is fine: after its all-to-all every rank holds the
            # whole sequence for its head shard.
            raise ValueError(
                "VDN_WINDOW_ATTN is not compatible with ring sequence parallelism: the window "
                "plan indexes the global packed sequence, which a ring rank does not hold. "
                "Use Ulysses SP (ring_degree=1)."
            )

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        plan = self._resolve_plan(query, attn_metadata)
        if plan is None:
            return self.dense_fallback.forward_cuda(query, key, value, attn_metadata)
        return self._forward_window(query, key, value, plan)

    def _resolve_plan(
        self,
        query: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> VDNWindowPlan | None:
        """The window geometry for this forward, or None when it must stay dense."""
        if attn_metadata is None or attn_metadata.video_layout is None:
            return None
        if self.qkv_layout is None or self.qkv_layout.upper() != _INPUT_LAYOUT:
            # The windowed path reads the sequence off dim 1 and the dense fallback
            # resolves an absent layout its own way. Windowing on an assumption would
            # put the two paths on different axes.
            logger.warning_once(
                "VDN_WINDOW_ATTN staying dense: this layer declares qkv_layout=%r, but the "
                "window plan needs %s to locate the sequence axis.",
                self.qkv_layout,
                _INPUT_LAYOUT,
            )
            return None
        if query.shape[0] != 1:
            # H3 packs one request per document; a co-batched forward would need one
            # plan per document and the row ranges are per-document offsets.
            logger.warning_once(
                "VDN_WINDOW_ATTN staying dense: batch %d, but the window plan describes a single packed document.",
                query.shape[0],
            )
            return None

        geometry = _target_video_geometry(attn_metadata)
        if geometry is None:
            return None
        video_start, (frames, grid_h, grid_w) = geometry

        extra = attn_metadata.extra or {}
        used_len = int(extra.get("max_seqlen_q") or extra.get("valid_kv_length") or query.shape[1])
        used_len = min(used_len, query.shape[1])
        if used_len <= 0:
            return None

        plan = _cached_plan(
            used_len=used_len,
            video_start=video_start,
            num_frames=frames,
            tokens_per_frame=grid_h * grid_w,
            chunk=self.window.chunk,
            radius=self.window.radius,
            anchor_frames=self.window.anchor_frames,
        )
        if plan.is_dense:
            # A window wide enough to cover every frame IS the original attention. Going
            # through the dense kernel rather than reproducing it as one degenerate
            # group keeps that identity exact instead of merely close: two bf16
            # attention kernels over ~100k keys differ by more than this mask does.
            logger.info_once(
                "VDN_WINDOW_ATTN dense for this shape: a %d-frame window covers all %d latent "
                "frames, so the mask keeps every pair.",
                (2 * self.window.radius + 1) * max(self.window.chunk, 1),
                frames,
            )
            return None

        logger.info_once(
            "VDN_WINDOW_ATTN active: chunk=%d radius=%d anchors=%s -> %d frames of %d per query, "
            "%d groups, %.1f%% of pairs kept over %d rows.",
            self.window.chunk,
            self.window.radius,
            self.window.anchor_frames,
            (2 * self.window.radius + 1) * max(self.window.chunk, 1),
            frames,
            len(plan.window_groups),
            100.0 * plan.kept_pair_fraction(),
            used_len,
        )
        return plan

    def _forward_window(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: VDNWindowPlan,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func

        if flash_attn_varlen_func is None:
            # validate_available() checks this at backend selection, but nothing makes
            # that a precondition of reaching here.
            raise ImportError("VDN_WINDOW_ATTN needs flash_attn_varlen_func to run its window groups")
        varlen = flash_attn_varlen_func

        q, k, v = query[0], key[0], value[0]
        # Zeros, not empty: rows at or past ``used_len`` are packing alignment and belong
        # to no group, so nothing below writes them.
        out = torch.zeros_like(q)

        # The globals and the anchor rows all attend the same thing -- the whole used
        # sequence -- so they are one call, and its K/V is a slice rather than a gather.
        if plan.dense_q_ranges:
            q_dense = _gather_rows(q, plan.dense_q_ranges)
            dense_out = varlen(
                q=q_dense,
                k=k[: plan.used_len],
                v=v[: plan.used_len],
                cu_seqlens_q=_cu_seqlens([q_dense.shape[0]], q.device),
                cu_seqlens_k=_cu_seqlens([plan.used_len], q.device),
                max_seqlen_q=q_dense.shape[0],
                max_seqlen_k=plan.used_len,
                causal=False,
                softmax_scale=self.softmax_scale,
            )
            _scatter_rows(out, plan.dense_q_ranges, _unwrap(dense_out))

        row_bytes = self.num_heads * self.head_size * q.element_size()
        budget_rows = max(1, _KV_GATHER_BUDGET_BYTES // max(row_bytes, 1))
        for batch in _batch_by_kv_budget(plan.window_groups, budget_rows):
            q_lens = [group.num_q_rows for group in batch]
            kv_lens = [group.num_kv_rows for group in batch]
            q_batch = _gather_rows(q, tuple(group.q_range for group in batch))
            kv_ranges = tuple(rng for group in batch for rng in group.kv_ranges)
            batch_out = varlen(
                q=q_batch,
                k=_gather_rows(k, kv_ranges),
                v=_gather_rows(v, kv_ranges),
                cu_seqlens_q=_cu_seqlens(q_lens, q.device),
                cu_seqlens_k=_cu_seqlens(kv_lens, q.device),
                max_seqlen_q=max(q_lens),
                max_seqlen_k=max(kv_lens),
                causal=False,
                softmax_scale=self.softmax_scale,
            )
            _scatter_rows(out, tuple(group.q_range for group in batch), _unwrap(batch_out))

        return out.unsqueeze(0)

    def forward_npu(self, query, key, value, attn_metadata=None):
        return self.dense_fallback.forward_npu(query, key, value, attn_metadata)

    def forward_xpu(self, query, key, value, attn_metadata=None):
        return self.dense_fallback.forward_xpu(query, key, value, attn_metadata)


def _unwrap(out: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    # FA3 may return (out, lse); FA2 returns out.
    return out[0] if isinstance(out, tuple) else out


def _cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    return torch.tensor(bounds, dtype=torch.int32, device=device)


def _gather_rows(tensor: torch.Tensor, ranges: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """Concatenate row ranges of a ``[S, N, D]`` tensor, avoiding a copy when possible."""
    if len(ranges) == 1:
        start, stop = ranges[0]
        return tensor[start:stop]
    return torch.cat([tensor[start:stop] for start, stop in ranges])


def _scatter_rows(out: torch.Tensor, ranges: tuple[tuple[int, int], ...], values: torch.Tensor) -> None:
    offset = 0
    for start, stop in ranges:
        length = stop - start
        out[start:stop] = values[offset : offset + length]
        offset += length


def _batch_by_kv_budget(groups, budget_rows: int):
    """Group the window groups into variable-length calls under a gathered-row budget."""
    batch: list = []
    rows = 0
    for group in groups:
        if batch and rows + group.num_kv_rows > budget_rows:
            yield batch
            batch, rows = [], 0
        batch.append(group)
        rows += group.num_kv_rows
    if batch:
        yield batch


def _target_video_geometry(
    attn_metadata: AttentionMetadata,
) -> tuple[int, tuple[int, int, int]] | None:
    """Where the target video sits, in both layout spellings H3 publishes.

    ``prefix_len``/``latent_grid`` is the t2va one-tail packing; ``video_spans`` is the
    fl2va/ref2va one, where reference media and audio rows sit between videos and only
    the last ``target`` span is being denoised. Reference spans are deliberately NOT
    windowed -- they are conditioning the model reads densely.
    """
    layout = attn_metadata.video_layout
    if layout is None:
        return None
    if layout.video_spans:
        target = next((span for span in reversed(layout.video_spans) if span.role == "target"), None)
        if target is None:
            return None
        frames, height, width = (int(dim) for dim in target.latent_grid)
        return int(target.start), (frames, height, width)
    if layout.prefix_len is None or layout.latent_grid is None:
        return None
    frames, height, width = (int(dim) for dim in layout.latent_grid)
    return int(layout.prefix_len), (frames, height, width)


_PLAN_CACHE: dict[tuple, VDNWindowPlan] = {}
_MAX_CACHED_PLANS = 8


def _cached_plan(**geometry) -> VDNWindowPlan:
    """Plans are pure functions of a handful of ints, and every DiT block wants the same
    one at every denoising step -- 50 blocks x N steps rebuilds of the same object."""
    key = tuple(sorted(geometry.items()))
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        if len(_PLAN_CACHE) >= _MAX_CACHED_PLANS:
            _PLAN_CACHE.clear()
        plan = build_window_plan(**geometry)
        _PLAN_CACHE[key] = plan
    return plan


__all__ = ["VDNWindowAttentionBackend", "VDNWindowAttentionImpl", "VDNWindowConfig"]
