# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-sparse video attention via SLA (Sparse-Linear Attention) on CUDA.

SLA (https://github.com/thu-ml/SLA, arXiv 2509.24006) scores mean-pooled query
and key blocks, keeps the top ``1 - sparsity`` fraction per query block, and runs
a Triton kernel over that block map. It is meant for checkpoints distilled under
the same sparsity: running it on weights trained for dense attention trades
quality for speed, and running such weights densely wastes what they were
trained for. ``lightx2v/Minimax-h3-Turbo-SLA`` is the first released MiniMax-H3
adapter distilled this way, at an 85% sparsity ratio.

Two deliberate departures from the reference module:

1. ``SparseLinearAttention`` adds a linear-attention branch through a trainable
   ``proj_l``, which ``init_weights_`` zero-initialises. No released MiniMax-H3
   adapter carries ``proj_l`` (all 624 tensors are LoRA A/B pairs), so that
   branch would contribute exactly zero while costing two extra matmuls per
   layer. This backend therefore calls the block-sparse kernel directly, which
   is also what LightX2V's own ``dynamic_sparse_attn`` does.
2. Block selection runs over the whole packed sequence rather than exempting the
   prefix the way ``rainfusion_attn`` does for rf_v2. Upstream distilled and
   serves the adapter with uniform selection, and the runtime must match the
   training-time attention pattern or the adapter is off-distribution.

Everything the kernel cannot serve — dense warmup steps, exempt layers, a layer
that does not declare ``qkv_layout="BSND"``, sequences with no published video
segment, or segments too short for selection to pay off — falls back to
FlashAttention, so a model may select this backend unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)

# The Triton kernel asserts BLOCK_M in {64, 128} and BLOCK_N == 64.
_BLOCK_Q = 64
_BLOCK_K = 64

# Under this many key blocks the pooling, top-k and LUT build cost more than the
# QK work they remove, so stay dense.
_MIN_KEY_BLOCKS = 32

# This backend reads the sequence off dim 1, which only ``BSND`` guarantees.
_INPUT_LAYOUT = "BSND"

_MISSING_SLA = (
    "SLA_ATTN requires the `sparse_linear_attention` package (pure Triton, no CUDA build): "
    "pip install git+https://github.com/thu-ml/SLA.git . "
    "Otherwise use FlashAttention by setting DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN"
)


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        return extract_layer_index(prefix)
    except (AssertionError, ValueError):
        return None


@dataclass(frozen=True)
class SLAConfig:
    """Resolved SLA controls for one attention layer.

    ``sparsity`` is the fraction of key blocks dropped per query block; the
    kernel keeps ``round((1 - sparsity) * num_key_blocks)``, at least one.
    ``start_step`` and ``skip_layers`` are the accuracy knobs: early denoise
    steps and named DiT blocks stay dense.
    """

    sparsity: float = 0.0
    start_step: int = 0
    skip_layers: frozenset[int] = frozenset()

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict | None) -> SLAConfig:
        bk = backend_kwargs or {}
        return cls(
            sparsity=float(bk.get("sparsity", 0.0)),
            start_step=int(bk.get("start_step", 0)),
            skip_layers=frozenset(bk.get("skip_layers") or ()),
        )

    @property
    def enabled(self) -> bool:
        return self.sparsity > 0.0

    @property
    def topk_ratio(self) -> float:
        return 1.0 - self.sparsity


@dataclass(frozen=True)
class SLAPlan:
    """Per-forward geometry handed to the SLA kernel."""

    used_len: int
    key_blocks: int


class SLAAttentionBackend(AttentionBackend):
    supported_platforms: tuple[str, ...] = ("cuda",)

    @classmethod
    def validate_available(cls) -> None:
        from importlib.util import find_spec

        if find_spec("sparse_linear_attention") is None:
            raise ValueError(_MISSING_SLA)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # The kernel keeps the head dim in registers and picks its warp count
        # off it; these are the two widths it is written for.
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SLA_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SLAAttentionImpl]:
        return SLAAttentionImpl


class SLAAttentionImpl(AttentionImpl):
    """Top-k block-sparse attention over a packed multimodal sequence."""

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

        self.sla = SLAConfig.from_backend_kwargs(backend_kwargs)
        self.layer_idx = _try_extract_layer_index(prefix)

        if self.sla.enabled:
            self._validate_parallel_config()
            if causal:
                raise ValueError(
                    "SLA_ATTN does not support causal attention: block selection ranks keys by "
                    "pooled relevance and cannot express a causal mask. Select FLASH_ATTN for "
                    "causal roles."
                )
            if qkv_layout is not None and qkv_layout.upper() != _INPUT_LAYOUT:
                raise ValueError(
                    f"SLA_ATTN needs {_INPUT_LAYOUT} tensors to locate the sequence axis, but this "
                    f"layer declares qkv_layout={qkv_layout!r}. Select FLASH_ATTN for this role."
                )

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
        ring_degree = getattr(parallel_config, "ring_degree", 1)
        if ring_degree > 1:
            # Ring hands each rank a slice of the sequence, so pooled scoring
            # would rank only local keys and the layer bypasses this backend
            # entirely (see Attention._run_ring_attention).
            raise ValueError(
                "SLA_ATTN is not compatible with ring sequence parallelism "
                f"(ring_degree={ring_degree}): block selection needs the whole key sequence. "
                "Use Ulysses SP (ring_degree=1) instead."
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
        return self._forward_sparse(query, key, value, plan)

    def _resolve_plan(
        self,
        query: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> SLAPlan | None:
        """Return the kernel geometry, or None when this forward must stay dense."""
        sla = self.sla
        if not sla.enabled:
            return None
        if self.layer_idx is not None and self.layer_idx in sla.skip_layers:
            return None
        if is_forward_context_available():
            step_idx = get_forward_context().denoise_step_idx
            if step_idx is not None and step_idx < sla.start_step:
                return None
        if self.qkv_layout is None:
            # The sparse path reads the sequence off dim 1, which the tensors
            # alone do not establish, and the dense fallback resolves an absent
            # layout its own way. Sparsifying on an assumption would put the two
            # paths on different axes.
            logger.warning_once(
                "SLA_ATTN staying dense: this layer does not declare qkv_layout, and block "
                "selection needs %s to locate the sequence axis.",
                _INPUT_LAYOUT,
            )
            return None
        if attn_metadata is None:
            return None
        if query.dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "SLA_ATTN staying dense: the kernel is written for bf16/fp16, got %s.",
                query.dtype,
            )
            return None

        # Rows past packed document 0 are alignment padding. They must not take a
        # share of any softmax denominator, and the kernel has no mask input, so
        # the sparse path only ever sees the used prefix.
        max_seqlen_q = attn_metadata.extra.get("max_seqlen_q")
        valid_kv_length = attn_metadata.extra.get("valid_kv_length")
        used_len = int(max_seqlen_q or valid_kv_length or query.shape[1])
        used_len = min(used_len, query.shape[1])
        if used_len <= 0:
            return None

        key_blocks = (used_len + _BLOCK_K - 1) // _BLOCK_K
        if key_blocks < _MIN_KEY_BLOCKS:
            logger.warning_once(
                "SLA_ATTN staying dense: %d rows is %d key blocks, under the %d-block threshold "
                "where selection pays for its own pooling and top-k.",
                used_len,
                key_blocks,
                _MIN_KEY_BLOCKS,
            )
            return None

        logger.info_once(
            "SLA_ATTN active: sparsity=%.2f (keeps %d of %d key blocks), start_step=%d, "
            "exempt_layers=%d, rows=%d. Selection covers the whole packed sequence, matching how "
            "the adapter was distilled.",
            sla.sparsity,
            max(1, int(sla.topk_ratio * key_blocks)),
            key_blocks,
            sla.start_step,
            len(sla.skip_layers),
            used_len,
        )
        return SLAPlan(used_len=used_len, key_blocks=key_blocks)

    def _forward_sparse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: SLAPlan,
    ) -> torch.Tensor:
        try:
            from sparse_linear_attention.kernel import _attention
            from sparse_linear_attention.utils import get_block_map
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(_MISSING_SLA) from exc

        used = plan.used_len
        # [B, S, N, D] -> [B, N, S, D]: the kernel indexes (batch*head, seq, dim).
        q, k, v = (tensor[:, :used].transpose(1, 2).contiguous() for tensor in (query, key, value))

        sparse_map, lut, topk = get_block_map(q, k, topk_ratio=self.sla.topk_ratio, BLKQ=_BLOCK_Q, BLKK=_BLOCK_K)
        out = _attention.apply(q, k, v, sparse_map, lut, topk, _BLOCK_Q, _BLOCK_K, self.softmax_scale)
        out = out.transpose(1, 2)

        if used == query.shape[1]:
            return out.contiguous()
        padded = torch.zeros_like(query)
        padded[:, :used] = out
        return padded


__all__ = ["SLAAttentionBackend", "SLAAttentionImpl", "SLAConfig", "SLAPlan"]
