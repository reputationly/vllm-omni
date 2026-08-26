# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Block-sparse video attention via SLA block selection run on a SageAttention2
(SpargeAttn) INT8 kernel, on CUDA.

Block selection reuses the same ``sparse_linear_attention.utils.get_block_map``
(https://github.com/thu-ml/SLA, arXiv 2509.24006) that ``SLA_ATTN`` uses, so the
two backends select identical key blocks and differ only in how the selected
blocks are computed: ``SLA_ATTN`` runs a dense-float Triton kernel, this backend
runs SpargeAttn's INT8-quantized SageAttention2 kernel
(https://github.com/thu-ml/SpargeAttn), the ``operator: "sage2"`` path LightX2V's
own published reference config for ``lightx2v/Minimax-h3-Turbo-SLA`` uses.

``_block_map_to_incremental_lut`` and ``_sage2_block_sparse_attn`` below are a
direct port of LightX2V's ``lightx2v/common/ops/attn/utils/sparge_util.py``
(Apache-2.0, Copyright (c) 2025 by SpargeAttn team) — that glue code is not part
of the installable ``spas_sage_attn`` package's public API, so there is nothing
to import it from.

Two things to know before trusting this backend:

1. **Only the sm80/86/87 (Ampere) kernel branch has been run on real hardware**
   (an A100). The sm90 branch is ported from LightX2V unchanged but unverified
   here — treat it as unvalidated until it has been.
2. **On MiniMax-H3, this backend's non-sm90 block size (``BLKQ=128, BLKK=64``)
   corrupted audio when block selection covered the whole packed sequence —
   confirmed on the real SpargeAttn kernel, not extrapolated, then fixed.**
   A prior vllm-omni experiment (see
   ``docs/实验报告/MiniMax-H3-SLA稀疏注意力-接入实测与暂缓结论-2026-08-22.md`` §5)
   found that forcing ``SLA_ATTN``'s Triton kernel to ``BLKQ=128`` corrupted
   audio on MiniMax-H3, because a 128-row query block can straddle the
   text/keyframe/audio/video boundary in the packed sequence when audio is a
   small fraction of the total rows. Running this backend for real on
   MiniMax-H3 (boars10s case, sparsity=0.85, 2026-08-26) reproduced the same
   failure on the actual SpargeAttn kernel: LUFS -41.3 → -7.4, true peak
   -28.1 → +3.3 dBFS (clipping), LRA 2.7 → 26.3, against the same-content
   ``SLA_ATTN`` (Triton) baseline.

   The fix, applied below: when the model publishes ``AttentionMetadata.
   video_layout``, block selection only runs over the pure video segment
   (``video_layout.prefix_len:used_len``); the prefix (text, visual
   conditions, audio) always runs dense, so no query block can ever straddle
   the boundary. This mirrors ``RAINFUSION_ATTN``'s existing prefix-dense
   design (``rainfusion_attn.py``), not something invented for this backend.
   It is a deliberate departure from ``SLA_ATTN``'s choice to select over the
   whole sequence uniformly (see that backend's docstring) to match how the
   adapter was distilled — trade a theoretical off-distribution risk on the
   prefix for fixing a confirmed, 100%-reproducible correctness bug. Validate
   video quality, not just audio, after any change here: this prefix carve-out
   has not been checked against the training distribution the adapter expects.
   When no ``video_layout`` is published, this backend falls back to its
   original whole-sequence selection (unverified quality on such roles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.triton_utils import tl, triton

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)

# Under this many key blocks the pooling, top-k and LUT build cost more than the
# QK work they remove, so stay dense. Matches SLA_ATTN's threshold.
_MIN_KEY_BLOCKS = 32

# This backend reads the sequence off dim 1, which only ``BSND`` guarantees.
_INPUT_LAYOUT = "BSND"

_MISSING_SLA = (
    "SLA_SAGE2_ATTN requires the `sparse_linear_attention` package (pure Triton, no CUDA build) "
    "for block selection: pip install git+https://github.com/thu-ml/SLA.git . "
    "Otherwise use FlashAttention by setting DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN"
)
_MISSING_SPARGE = (
    "SLA_SAGE2_ATTN requires the `spas_sage_attn` package (SpargeAttn, CUDA build) for the "
    "sage2 kernel: git clone https://github.com/thu-ml/SpargeAttn.git && cd SpargeAttn && "
    "TORCH_CUDA_ARCH_LIST=<your arch, e.g. 8.0 for A100> pip install . --no-build-isolation. "
    "Otherwise use SLA_ATTN (pure Triton) or FlashAttention."
)


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        return extract_layer_index(prefix)
    except (AssertionError, ValueError):
        return None


def _get_cuda_arch(device: torch.device | int | None) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm{major}{minor}"


def _block_sizes_for_arch(arch: str) -> tuple[int, int]:
    """SpargeAttn's compiled kernels are tiled for one block shape per arch family."""
    if arch == "sm90":
        return 64, 128
    return 128, 64


# ---------------------------------------------------------------------------
# Ported from LightX2V's sparge_util.py (Apache-2.0, Copyright (c) 2025 by
# SpargeAttn team). Converts a boolean [B, H, Q, K] block-selection map into the
# incremental-offset LUT format spas_sage_attn's block-sparse kernels expect.
# ---------------------------------------------------------------------------


@triton.jit
def _block_map_to_incremental_lut_kernel(map_ptr, lut_ptr, valid_block_num_ptr, num_block_k):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    H, Q = tl.num_programs(1), tl.num_programs(2)

    map_ptr = map_ptr + b * H * Q * num_block_k + h * Q * num_block_k + q * num_block_k
    lut_ptr = lut_ptr + b * H * Q * num_block_k + h * Q * num_block_k + q * num_block_k
    valid_block_num_ptr = valid_block_num_ptr + b * H * Q + h * Q + q

    valid_block_num = 0
    prev_block = 0
    for i in range(num_block_k):
        cur_block = tl.load(map_ptr + i)
        if cur_block:
            tl.store(lut_ptr + valid_block_num, i - prev_block)
            valid_block_num += 1
            prev_block = i
    tl.store(valid_block_num_ptr, valid_block_num)


def _block_map_to_incremental_lut(block_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert block_map.dim() == 4
    block_map = block_map.contiguous()
    B, H, Q, K = block_map.shape
    lut = torch.zeros((B, H, Q, K), dtype=torch.int32, device=block_map.device)
    valid_block_num = torch.zeros((B, H, Q), dtype=torch.int32, device=block_map.device)
    _block_map_to_incremental_lut_kernel[(B, H, Q)](block_map, lut, valid_block_num, K)
    return lut, valid_block_num


def _sage2_block_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lut: torch.Tensor,
    valid_block_num: torch.Tensor,
    block_q: int,
    block_k: int,
    arch: str,
    scale: float,
) -> torch.Tensor:
    """Port of LightX2V's ``sage2_block_sparse_attn``. Only the sm80/86/87 branch
    has been exercised on real hardware here; sm90 is carried over unverified."""
    import spas_sage_attn._fused as fused
    import spas_sage_attn._qattn as qattn
    from spas_sage_attn.utils import get_vanilla_qk_quant

    head_dim = q.size(-1)
    if head_dim not in (64, 128):
        raise ValueError(f"SLA_SAGE2_ATTN requires head_dim in (64, 128), got {head_dim}.")

    km = k.mean(dim=-2, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, block_q, block_k)
    out = torch.empty_like(q)

    if arch in ("sm80", "sm86", "sm87"):
        pv_threshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
        v_fp16 = v.to(torch.float16)
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8, k_int8, v_fp16, out, lut, valid_block_num, pv_threshold, q_scale, k_scale, 1, False, 1, scale, 0
        )
        return out

    # sm90+: unverified on this codebase, ported as-is for future hardware.
    b, h_kv, kv_len, head_dim = v.shape
    padded_len = (kv_len + 127) // 128 * 128
    v_transposed_permuted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
    fused.transpose_pad_permute_cuda(v, v_transposed_permuted, 1)
    v_fp8 = torch.empty(v_transposed_permuted.shape, dtype=torch.float8_e4m3fn, device=v.device)
    v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
    fused.scale_fuse_quant_cuda(v_transposed_permuted, v_fp8, v_scale, kv_len, 2.25, 1)
    if arch == "sm90":
        qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90(
            q_int8, k_int8, v_fp8, out, lut, valid_block_num, q_scale, k_scale, v_scale, 1, False, 1, scale
        )
    else:
        pv_threshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
        qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
            q_int8,
            k_int8,
            v_fp8,
            out,
            lut,
            valid_block_num,
            pv_threshold,
            q_scale,
            k_scale,
            v_scale,
            1,
            False,
            1,
            scale,
            0,
        )
    return out


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SLASage2Config:
    """Resolved SLA controls for one attention layer. Same knobs as SLA_ATTN's
    ``SLAConfig``; kept as an independent copy per this file's self-contained
    convention (see module docstring)."""

    sparsity: float = 0.0
    start_step: int = 0
    skip_layers: frozenset[int] = frozenset()

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict | None) -> SLASage2Config:
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
class SLASage2Plan:
    used_len: int
    key_blocks: int
    # Rows [0, prefix_len) run dense; only [prefix_len, used_len) is
    # block-selected. 0 when the model publishes no video_layout, which
    # degenerates to selecting over the whole used_len as before.
    prefix_len: int = 0


class SLASage2AttentionBackend(AttentionBackend):
    supported_platforms: tuple[str, ...] = ("cuda",)

    @classmethod
    def validate_available(cls) -> None:
        from importlib.util import find_spec

        if find_spec("sparse_linear_attention") is None:
            raise ValueError(_MISSING_SLA)
        if find_spec("spas_sage_attn") is None:
            raise ValueError(_MISSING_SPARGE)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SLA_SAGE2_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SLASage2AttentionImpl]:
        return SLASage2AttentionImpl


class SLASage2AttentionImpl(AttentionImpl):
    """Top-k block-sparse attention over a packed multimodal sequence, computed
    with SpargeAttn's INT8 SageAttention2 kernel."""

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

        self.sla = SLASage2Config.from_backend_kwargs(backend_kwargs)
        self.layer_idx = _try_extract_layer_index(prefix)

        if self.sla.enabled:
            self._validate_parallel_config()
            if causal:
                raise ValueError(
                    "SLA_SAGE2_ATTN does not support causal attention: block selection ranks keys by "
                    "pooled relevance and cannot express a causal mask. Select FLASH_ATTN for causal roles."
                )
            if qkv_layout is not None and qkv_layout.upper() != _INPUT_LAYOUT:
                raise ValueError(
                    f"SLA_SAGE2_ATTN needs {_INPUT_LAYOUT} tensors to locate the sequence axis, but this "
                    f"layer declares qkv_layout={qkv_layout!r}. Select FLASH_ATTN for this role."
                )
        # Arch/block-size are resolved lazily (see the `arch` property below),
        # not here: sibling backends (SLA_ATTN, FlashAttentionImpl) never touch
        # torch.cuda.* in __init__, since model construction can happen before
        # the framework has moved weights to a device. Querying the device
        # eagerly for every one of ~50 attention layers, ahead of the
        # framework's own controlled device placement, is exactly the kind of
        # early/uncontrolled CUDA touch worth avoiding on principle even where
        # it isn't proven to be the direct cause of anything.
        self._arch: str | None = None

        self.dense_fallback = FlashAttentionBackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            qkv_layout=qkv_layout,
        )

    @property
    def arch(self) -> str:
        if self._arch is None:
            if torch.cuda.is_available():
                self._arch = _get_cuda_arch(torch.accelerator.current_device_index())
            else:
                self._arch = "unknown"
        return self._arch

    @property
    def block_q(self) -> int:
        return _block_sizes_for_arch(self.arch)[0]

    @property
    def block_k(self) -> int:
        return _block_sizes_for_arch(self.arch)[1]

    def _validate_parallel_config(self) -> None:
        config = get_current_diffusion_config_or_none()
        parallel_config = getattr(config, "parallel_config", None)
        ring_degree = getattr(parallel_config, "ring_degree", 1)
        if ring_degree > 1:
            raise ValueError(
                "SLA_SAGE2_ATTN is not compatible with ring sequence parallelism "
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
            return self.dense_fallback.forward_cuda(query, key, value, attn_metadata)  # type: ignore[arg-type]
        return self._forward_sparse(query, key, value, plan)

    def _resolve_plan(
        self,
        query: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> SLASage2Plan | None:
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
            logger.warning_once(
                "SLA_SAGE2_ATTN staying dense: this layer does not declare qkv_layout, and block "
                "selection needs %s to locate the sequence axis.",
                _INPUT_LAYOUT,
            )
            return None
        if attn_metadata is None:
            return None
        if query.dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "SLA_SAGE2_ATTN staying dense: the kernel is written for bf16/fp16, got %s.",
                query.dtype,
            )
            return None

        max_seqlen_q = attn_metadata.extra.get("max_seqlen_q")
        valid_kv_length = attn_metadata.extra.get("valid_kv_length")
        used_len = int(max_seqlen_q or valid_kv_length or query.shape[1])
        used_len = min(used_len, query.shape[1])
        if used_len <= 0:
            return None

        # Rows before video_layout.prefix_len (text, visual conditions, audio
        # on MiniMax-H3) always run dense: a query block pooled across this
        # boundary blends unrelated modalities into one relevance score. Only
        # the pure video tail is block-selected. No video_layout means the
        # model hasn't published where that boundary is, so the whole
        # used_len is treated as one region — the original, unfixed behavior.
        video_layout = attn_metadata.video_layout
        prefix_len = 0
        if video_layout is not None:
            prefix_len = min(max(int(video_layout.prefix_len), 0), used_len)
        video_len = used_len - prefix_len

        # key_blocks counts the K side, which stays the full used_len (prefix
        # rows remain valid keys for the video query blocks — only the video
        # query's own block pooling is protected from the prefix boundary).
        key_blocks = (used_len + self.block_k - 1) // self.block_k
        if video_len <= 0 or key_blocks < _MIN_KEY_BLOCKS:
            logger.warning_once(
                "SLA_SAGE2_ATTN staying dense: %d video rows to select from %d key blocks over %d total rows "
                "(prefix=%d), under the %d-block threshold where selection pays for its own pooling and top-k.",
                video_len,
                key_blocks,
                used_len,
                prefix_len,
                _MIN_KEY_BLOCKS,
            )
            return None

        logger.info_once(
            "SLA_SAGE2_ATTN active: sparsity=%.2f (keeps %d of %d key blocks), operator=sage2 "
            "arch=%s BLKQ=%d BLKK=%d, start_step=%d, exempt_layers=%d, rows=%d (prefix=%d dense, video=%d sparse).",
            sla.sparsity,
            max(1, int(sla.topk_ratio * key_blocks)),
            key_blocks,
            self.arch,
            self.block_q,
            self.block_k,
            sla.start_step,
            len(sla.skip_layers),
            used_len,
            prefix_len,
            video_len,
        )
        return SLASage2Plan(used_len=used_len, key_blocks=key_blocks, prefix_len=prefix_len)

    def _forward_sparse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: SLASage2Plan,
    ) -> torch.Tensor:
        try:
            from sparse_linear_attention.utils import get_block_map
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(_MISSING_SLA) from exc
        try:
            import spas_sage_attn  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(_MISSING_SPARGE) from exc

        used = plan.used_len
        prefix_len = plan.prefix_len
        # [B, S, N, D] -> [B, N, S, D]: the kernel indexes (batch*head, seq, dim).
        # K/V stay the full used range — prefix rows remain valid keys for the
        # video query; only the video query's own block pooling is protected
        # from ever blending with prefix content.
        k, v = (tensor[:, :used].transpose(1, 2).contiguous() for tensor in (key, value))
        q_video = query[:, prefix_len:used].transpose(1, 2).contiguous()

        # Same block-selection scoring SLA_ATTN uses; discard its own LUT/topk
        # outputs and rebuild the LUT in the layout sage2_block_sparse_attn wants.
        sparse_map, _sla_lut, _sla_topk = get_block_map(
            q_video, k, topk_ratio=self.sla.topk_ratio, BLKQ=self.block_q, BLKK=self.block_k
        )
        lut, valid_block_num = _block_map_to_incremental_lut(sparse_map)
        video_out = _sage2_block_sparse_attn(
            q_video, k, v, lut, valid_block_num, self.block_q, self.block_k, self.arch, self.softmax_scale
        )
        video_out = video_out.transpose(1, 2)  # [B, N, video_len, D] -> [B, video_len, N, D]

        if prefix_len > 0:
            # Exact dense attention, not the plan.enabled dense fallback path:
            # the prefix query attends the same full [0:used] key range a
            # non-sparse layer would, it just never enters block selection.
            prefix_out = self.dense_fallback.forward_cuda(
                query[:, :prefix_len], key[:, :used], value[:, :used], attn_metadata=None
            )  # type: ignore[arg-type]
            out = torch.cat((prefix_out, video_out), dim=1)
        else:
            out = video_out

        if used == query.shape[1]:
            return out.contiguous()
        padded = torch.zeros_like(query)
        padded[:, :used] = out
        return padded


__all__ = ["SLASage2AttentionBackend", "SLASage2AttentionImpl", "SLASage2Config", "SLASage2Plan"]
