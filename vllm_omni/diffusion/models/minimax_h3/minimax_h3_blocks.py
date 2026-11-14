# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared architecture and DiT blocks for MiniMax H3 and its control branch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionMetadata,
    PackedPaddingMetadata,
    VideoTokenLayout,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.attention.backends.vdnh3_attn import VDNLayout
from vllm_omni.diffusion.attention.layer import Attention, ScheduleCandidate
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.layers.activation import SiluAndMul
from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope
from vllm_omni.diffusion.layers.indexed_modulation import (
    indexed_gate,
    indexed_gate_rms_norm_scale_shift,
    rms_norm_indexed_scale_shift,
)
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.offloader.module_residency import BoundedAllocatorCache, PinnedModuleStager

from .adaln_cache import MiniMaxH3RuntimeAdalnCache
from .low_vram import attention_head_chunks, ffn_row_chunks
from .vdn_hybrid import REQUIRED_ATTENTION_BACKEND, VDNHybridAttention, combine_hybrid_output
from .vdnh3 import VDNConfig, VDNH3HybridAttention

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from .reference_kv_cache import MiniMaxH3ReferenceKVTier2State

_BF16_DTYPE = torch.bfloat16
_FP32_DTYPE = torch.float32


# AdaLN modality count: token tags carry -1 for padding and 0/1/2 for
# video/text/audio tokens (padding is clamped to 0 before the embedding
# lookup and masked out afterwards).
MINIMAX_H3_ADALN_MODALITY_NUM = 3


# Opt-in fp16-range protection for the NPU ascend_laser_attention kernel
# (consumed only via the "laser_input_scale" extra key; other backends and
# platforms ignore it). The kernel stores unscaled QK^T in an fp16 GM
# workspace, and H3's outlier activations (per-element amax in the hundreds)
# push dot products past fp16 max 65504, turning whole 128-row blocks NaN.
# 256 is a power of two, so pre-dividing q/k/v and the compensating
# kernel-scale/output multiplies are exact in floating point.
MINIMAX_H3_LASER_INPUT_SCALE = 256.0


# Packed multi-request forwards require the attention backend to actually
# consume cu_seqlens as a block-diagonal plan (not a padding-mask rebuild that
# spans the full packed row). The pipeline gates on this capability before
# packing, and ``_run_packed_attention`` re-checks it per forward; a name-only
# gate would let FLASH_ATTN's NPU/XPU code paths through even though those
# variants would silently attend across request boundaries.
def _ring_sequence_parallel_is_active(attention_layer: Attention) -> bool:
    """Match :meth:`Attention._get_active_parallel_strategy` for Ring."""
    if not getattr(attention_layer, "use_ring", False) or getattr(attention_layer, "skip_sequence_parallel", False):
        return False
    if is_forward_context_available() and not get_forward_context().sp_active:
        return False
    return True


def _attention_isolates_packed_requests(attention_layer: Any, backend: type[AttentionBackend] | None = None) -> bool:
    """True if this attention layer keeps N-document packed boundaries.

    Requires a backend advertising ``supports_multi_doc_packed_varlen`` *and*
    that the layer is not running under ring sequence parallelism (the ring
    kernel dispatches through its own attention that ignores the packed
    cu_seqlens regardless of the configured backend). ``backend`` defaults to
    the layer's baseline; a forward passes the backend it will run.
    """
    if backend is None:
        backend = getattr(attention_layer, "attn_backend", None)
    if backend is None or not backend.supports_multi_doc_packed_varlen():
        return False
    return not _ring_sequence_parallel_is_active(attention_layer)


def _candidate_excludes_packed_padding(record: ScheduleCandidate) -> str | None:
    """Schedule candidate check: every DiT forward pads its packed documents to the sequence alignment."""
    backend = record.backend_cls
    if (
        backend.supports_packed_mask_free()
        or backend.supports_prefix_kv_slicing
        or backend.supports_attention_mask(record.spec)
    ):
        return None
    return (
        "MiniMax H3 pads every packed document, and this backend supports neither mask-free packed "
        "padding, prefix K/V slicing nor an attention mask, so it cannot exclude the padding rows"
    )


def _fasth3_requires_vsa_candidate(record: ScheduleCandidate) -> str | None:
    """Schedule candidate check for FastH3 VSA students, registered once their gates are built."""
    if record.backend_cls.get_name() == "FASTVIDEO_VSA":
        return None
    return (
        "the FastH3 student was distilled for FASTVIDEO_VSA and its compression gates are read only by "
        "that backend, so every DiT profile must select FASTVIDEO_VSA"
    )


def _vdn_requires_vdnh3_candidate(record: ScheduleCandidate) -> str | None:
    """Schedule candidate check for VDN-H3 checkpoints, registered by MiniMaxH3Attention.enable_vdn."""
    if record.backend_cls.get_name() == "VDNH3_ATTN":
        return None
    return (
        "the VDN-H3 checkpoint pairs its learned attention half with the VDNH3_ATTN frame window, which "
        "every other backend ignores, so every DiT profile must select VDNH3_ATTN"
    )


def _vdn_window_requires_window_candidate(record: ScheduleCandidate) -> str | None:
    """Schedule candidate check registered by MiniMaxH3Attention.enable_vdn_branch.

    The hybrid splits the sequence between the chunk-aligned window and the linear
    branch, so the softmax half *is* the window. A profile that selected any other
    backend would run dense attention behind the branch's gate and add the dropped frames
    back a second time, which renders wrongly instead of failing.
    """
    if record.backend_cls.get_name() == REQUIRED_ATTENTION_BACKEND:
        return None
    return (
        "this attention carries a VDN-H3 window branch whose softmax half is the chunk-aligned window "
        f"only {REQUIRED_ATTENTION_BACKEND} implements, so every DiT profile must select "
        f"{REQUIRED_ATTENTION_BACKEND}"
    )


# Modulation tensors each AdaLN projection emits per hidden_size: 18 inside a
# DiT block (shift/scale/gate for the two norms across the three modalities),
# 2 in the final layer. Both the dataclass defaults and the config-derived
# fallbacks below are these ratios times hidden_size.
MINIMAX_H3_ADALN_EXPAND_RATIO = 18
MINIMAX_H3_FINAL_ADALN_EXPAND_RATIO = 2


@dataclass
class MiniMaxH3DiTArchConfig:
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    # Present only on AdaLN-pruned checkpoints.  The released checkpoint feeds
    # every AdaLN projection with ``time_embed_dim`` values; the pruned one
    # stores coordinates in a checkpoint-defined affine subspace instead.
    adaln_rank: int | None = None
    time_table_size: int | None = None
    adaln_out_features: int = MINIMAX_H3_ADALN_EXPAND_RATIO * 5376
    final_adaln_out_features: int = MINIMAX_H3_FINAL_ADALN_EXPAND_RATIO * 5376
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    pdd_steps: int | None = None
    # False for an image checkpoint without the audio input/output projections
    # (audio_patch_proj, final_layer.audio_out); it serves no audio rows.
    audio_enabled: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> MiniMaxH3DiTArchConfig:
        # The released partition and the Modular Diffusers checkpoint describe
        # the same architecture with different field names.  Accept both here
        # so the loader can consume the pruned Diffusers shards directly while
        # preserving the existing partition path byte-for-byte.
        aliases = {
            "num_refiner_layers": "token_refiner_num_layers",
            "ffn_dim": "ffn_hidden_size",
            "in_channels": "latents_dim",
            "audio_in_channels": "audio_latents_dim",
            "freq_dim": "timestep_input_dim",
            "time_embed_hidden_dim": "time_embed_hidden_size",
            "rope_freq_dim": "rope_inv_freq_len",
        }
        normalized = dict(config)
        for source, target in aliases.items():
            if target not in normalized and source in normalized:
                normalized[target] = normalized[source]
        hidden_size = int(normalized.get("hidden_size", cls.hidden_size))
        normalized.setdefault("adaln_out_features", MINIMAX_H3_ADALN_EXPAND_RATIO * hidden_size)
        normalized.setdefault("final_adaln_out_features", MINIMAX_H3_FINAL_ADALN_EXPAND_RATIO * hidden_size)
        fields = cls.__dataclass_fields__
        values = {name: normalized[name] for name in fields if name in normalized}
        if "patch_size" in values:
            values["patch_size"] = tuple(values["patch_size"])
        arch = cls(**values)
        if type(arch.audio_enabled) is not bool:
            raise ValueError(f"audio_enabled must be a bool, got {arch.audio_enabled!r}")
        if not arch.audio_enabled and arch.pdd_steps is not None:
            raise ValueError("an audio-free MiniMax H3 DiT cannot use the FastH3 pdd_steps heads")
        if len(arch.patch_size) != 3:
            raise ValueError(f"patch_size must contain three values, got {arch.patch_size!r}")
        if arch.adaln_rank is not None:
            if arch.adaln_rank <= 0:
                raise ValueError(f"adaln_rank must be positive, got {arch.adaln_rank}")
            if arch.time_table_size is None or arch.time_table_size < 2:
                raise ValueError(f"AdaLN-pruned checkpoints require time_table_size >= 2, got {arch.time_table_size!r}")
        return arch


def _norm(size: int, *, eps: float, dtype: torch.dtype = _BF16_DTYPE) -> RMSNorm:
    # RMSNorm uses fp32 accumulation with bf16 inputs and outputs.
    # torch.nn.RMSNorm upcasts reduced-precision inputs for the variance
    # reduction, matching that accumulation semantic.
    return RMSNorm(size, eps=eps, dtype=dtype)


class MiniMaxH3Attention(nn.Module):
    # Full sparse checkpoints pin a ratio; legacy adapters use backend top-k.
    vsa_sparsity: float | None = None
    vsa_tile_size: int = 64
    vsa_ref_policy: str | None = None
    vsa_ref_keep_rate: float | None = None

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
        role: str = "self",
        role_category: str | None = None,
        skip_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.total_num_heads = arch.num_attention_heads
        self.head_dim = arch.attention_head_dim
        inner_dim = self.total_num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=arch.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            return_bias=True,
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.rot_dim = 6 * arch.rope_inv_freq_len
        self.q_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.rope = RotaryEmbedding(is_neox_style=True, half_head_dim=False)
        self.out_proj = RowParallelLinear(
            inner_dim,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        # VSA compression gate. A FastH3 VSA artifact assigns this projection
        # with ``.set_weight``; the dense path never builds it, so the module is
        # created only once the loader knows a VSA artifact is coming.
        self.to_gate_compress: ColumnParallelLinear | None = None
        self._gate_hidden_size = arch.hidden_size
        self._gate_quant_config = quant_config
        self._gate_prefix = f"{prefix}.to_gate_compress"
        # VDN-H3 hybrid attention. Two incompatible releases share this slot:
        # ``enable_vdn`` builds the upstream VDNH3_ATTN half (``vdnh3.py``), and
        # ``enable_vdn_branch`` builds our VDN_WINDOW_ATTN half (``vdn_hybrid.py``).
        # A layer carries at most one, and the forward dispatches on which class it got.
        self.vdn: VDNH3HybridAttention | VDNHybridAttention | None = None
        self._prefix = prefix
        self._vdn_prefix = prefix
        self._vdn_arch = arch
        from .attention.fastvideo_h3 import MiniMaxH3VSAImpl

        self.attention = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            softmax_scale=self.softmax_scale,
            causal=False,
            # Packed rows reach the impl as [B, S, N, D].
            qkv_layout="BSND",
            role=role,
            role_category=role_category,
            skip_sequence_parallel=skip_sequence_parallel,
            prefix=prefix,
            impl_overrides={"FASTVIDEO_VSA": MiniMaxH3VSAImpl},
        )

    def enable_vsa_gate(self) -> None:
        """Build the VSA compression gate this attention would otherwise lack.

        Called before ``load_weights`` so the artifact's ``.set_weight`` tensor
        has a parameter to land on. Zero-initialized like the Wan VSA layers, so
        a gate that never receives weights degrades to sparse-only selection
        rather than to garbage.
        """
        if self.to_gate_compress is not None:
            return
        self.to_gate_compress = ColumnParallelLinear(
            self._gate_hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            params_dtype=_BF16_DTYPE,
            quant_config=self._gate_quant_config,
            prefix=self._gate_prefix,
        )
        if hasattr(self.to_gate_compress, "weight"):
            nn.init.zeros_(self.to_gate_compress.weight)
        self.attention.add_schedule_candidate_check(_fasth3_requires_vsa_candidate)

    def _vsa_candidate_needs_gate(self, record: ScheduleCandidate) -> str | None:
        """Schedule candidate check: MiniMaxH3VSAImpl routes by H3 segments only when a VSA gate exists.

        A gateless layer whose baseline is already FASTVIDEO_VSA runs the same dense fallback without
        a schedule, so the candidate changes nothing there. This covers transformers_ref in the
        combined partition under FastH3 VSA, which gates only self.transformer while the FastH3
        contract resolves every self role to FASTVIDEO_VSA, and base H3 served with that baseline.
        """
        if record.backend_cls.get_name() != "FASTVIDEO_VSA" or self.to_gate_compress is not None:
            return None
        if self.attention.attn_backend.get_name() == "FASTVIDEO_VSA":
            return None
        return (
            "this attention has no VSA compression gate, so it never passes vsa_h3_prefix_segments and "
            "FASTVIDEO_VSA would run dense SDPA at every scheduled step; serve a FastH3 VSA checkpoint or "
            "select another backend for the profile"
        )

    def enable_vdn_branch(self, **window_config: Any) -> None:
        """Build the VDN hybrid extras this attention would otherwise lack (our VDN_WINDOW_ATTN).

        Called before ``load_weights`` so the checkpoint's branch tensors have
        parameters to land on, exactly like ``enable_vsa_gate``. It also flips
        ``out_proj`` off its own all-reduce: from here the two row-parallel projections
        share one collective, which is worth about a third of a block at the 15 s shape
        (see ``vdn_hybrid``).
        """
        if isinstance(self.vdn, VDNHybridAttention):
            return
        if self.vdn is not None:
            raise ValueError(
                f"{self._prefix} already carries the upstream VDNH3_ATTN half; a layer runs one VDN "
                "dialect, so enable_vdn and enable_vdn_branch are mutually exclusive"
            )
        arch = self._vdn_arch
        self.vdn = VDNHybridAttention(
            hidden_size=arch.hidden_size,
            total_heads=self.total_num_heads,
            head_dim=self.head_dim,
            params_dtype=_BF16_DTYPE,
            quant_config=self._gate_quant_config,
            prefix=self._vdn_prefix,
            # The branch reads THIS attention's raw q/k/v, so it must use THIS
            # attention's head shard. Letting it derive one independently means two
            # views of the same split that can disagree: the branch's fallback asks the
            # diffusion parallel state, which can be uninitialised while vLLM's TP group
            # (the one qkv_proj actually sharded on) is live.
            head_shard=(get_tensor_model_parallel_rank() * self.num_heads, self.num_heads),
            **window_config,
        )
        self.out_proj.reduce_results = False
        self.attention.add_schedule_candidate_check(_vdn_window_requires_window_candidate)

    def enable_vdn(self, config: VDNConfig) -> None:
        """Attach the learned half of VDN-H3 hybrid attention (see ``vdnh3.py``)."""
        if isinstance(self.vdn, VDNHybridAttention):
            raise ValueError(
                f"{self._prefix} already carries the VDN_WINDOW_ATTN branch; a layer runs one VDN "
                "dialect, so enable_vdn and enable_vdn_branch are mutually exclusive"
            )
        backend = self.attention.attn_backend.get_name()
        if backend != "VDNH3_ATTN":
            raise ValueError(f"VDN-H3 attention needs the VDNH3_ATTN backend, but {self._prefix} resolved {backend}")
        self.vdn = VDNH3HybridAttention(
            self._gate_hidden_size,
            self.total_num_heads,
            self.head_dim,
            config,
            quant_config=self._gate_quant_config,
            prefix=f"{self._prefix}.vdn",
        )
        self.attention.add_schedule_candidate_check(_vdn_requires_vdnh3_candidate)

    def _apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Rotate the first rot_dim head dims; pass the rest through.

        x: [T, heads, head_dim]; freqs: [T, rot_dim]. In the unfused path, cos/sin
        are cast to the activation dtype before the elementwise math.
        """
        rot_dim = self.rot_dim
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        cos = torch.cos(freqs).to(x.dtype)  # [T, rot_dim]
        sin = torch.sin(freqs).to(x.dtype)
        x_rot = self.rope(x_rot, cos, sin)
        return torch.cat((x_rot, x_pass), dim=-1)

    @torch.compiler.disable
    def _run_packed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_total: int,
        num_requests: int = 1,
        video_layout: VideoTokenLayout | None = None,
        vsa_prefix_segments: tuple[int, ...] = (),
        gate_compress: torch.Tensor | None = None,
        reference_kv_state: MiniMaxH3ReferenceKVTier2State | None = None,
        reference_kv_layer_index: int | None = None,
        reference_kv_compact: bool = False,
        vdn_window: VDNLayout | None = None,
    ) -> torch.Tensor:
        """Run packed attention as a small eager island.

        On a compact reuse step Q contains target rows only. Tier2 captures
        reference K/V after Ulysses has exchanged sequence rows for head shards,
        then prepends the balanced cached head shard as a K/V-only prefix. This
        avoids both rebuilding reference QKV and redistributing cached K/V.
        """
        if not 0 < max_seqlen <= packed_total:
            raise ValueError(
                f"max_seqlen must be within the packed sequence, got {max_seqlen} for length {packed_total}"
            )
        if reference_kv_compact and num_requests != 1:
            raise ValueError("compacted reference KV currently supports one request")

        reference_rows = 0
        if reference_kv_state is not None:
            if reference_kv_layer_index is None:
                raise RuntimeError("reference-KV cache requires a main-DiT layer index")
            if reference_kv_compact:
                reference_rows = reference_kv_state.global_reference_rows

        kv_total = packed_total + reference_rows
        kv_max_seqlen = max_seqlen + reference_rows
        if reference_rows:
            kv_cu_seqlens = cu_seqlens.clone()
            kv_cu_seqlens[1:] += reference_rows
        else:
            kv_cu_seqlens = cu_seqlens

        attn_mask = None
        mask_free_packed_padding = False
        out_padding = 0
        use_ring = _ring_sequence_parallel_is_active(self.attention)
        # Read once per forward: during an active scheduled denoise step this is
        # the candidate the layer dispatches to, and the packed layout below
        # must match that backend rather than the baseline.
        _, backend, backend_spec = self.attention.effective_attention()
        if num_requests > 1:
            if not _attention_isolates_packed_requests(self.attention, backend):
                backend_name = backend.get_name()
                raise ValueError(
                    f"MiniMax H3 packed a {num_requests}-request batch, but the resolved "
                    f"attention ({backend_name}, use_ring={getattr(self.attention, 'use_ring', False)}) "
                    "does not isolate multi-document packed cu_seqlens. Run one request "
                    "per forward on this backend."
                )
            used_q = packed_total
            used_kv = kv_total
        else:
            used_q = min(max_seqlen, packed_total)
            used_kv = min(kv_max_seqlen, kv_total)
            mask_free_packed_padding = not use_ring and (
                backend.supports_packed_mask_free()
                or (
                    reference_kv_compact
                    and backend.get_name() == "RAINFUSION_ATTN"
                    and FlashAttentionBackend.supports_packed_mask_free()
                )
            )
            no_mask = not use_ring and (backend.supports_prefix_kv_slicing or mask_free_packed_padding)
            # Hybrid Ulysses reshards Q to one ring partition before the ring
            # kernel runs, so a global key mask cannot pass its query-length
            # check. Ring consumes valid_kv_length directly.
            if used_kv < kv_total and not no_mask and not use_ring:
                if backend.supports_attention_mask(backend_spec):
                    attn_mask = torch.arange(kv_total, device=q.device)[None] < used_kv
                else:
                    # Backends without mask support still honor cu_seqlens as
                    # document bounds, so the valid prefix alone is a legal
                    # input: trim to it, run mask-free, and zero-pad the output
                    # back to packed_total. Rows >= used are the tail pad
                    # document (packed_sequence.py), never unpacked into the
                    # text/cond/audio/video outputs. K/V are trimmed to their own
                    # valid length, which exceeds Q's whenever cached reference
                    # rows lead the prefix.
                    q, k, v = q[:used_q], k[:used_kv], v[:used_kv]
                    out_padding = packed_total - used_q

        extra: dict[str, Any] = {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": kv_cu_seqlens,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_k": kv_max_seqlen,
            "valid_kv_length": used_kv,
            "npu_attn_varlen": not use_ring,
            "laser_input_scale": MINIMAX_H3_LASER_INPUT_SCALE,
            **({"vsa_h3_sparsity": self.vsa_sparsity} if self.vsa_sparsity is not None else {}),
            "vsa_h3_tile_size": self.vsa_tile_size,
            "vsa_h3_ref_policy": self.vsa_ref_policy,
            "vsa_h3_ref_keep_rate": self.vsa_ref_keep_rate,
        }
        if gate_compress is not None:
            extra["gate_compress"] = gate_compress.unsqueeze(0)
            if video_layout is not None and video_layout.video_spans:
                extra["vsa_h3_prefix_segments"] = vsa_prefix_segments
        if vdn_window is not None:
            extra["vdn_window"] = vdn_window
        if reference_kv_compact:
            extra["minimax_h3_compact_reference_rows"] = reference_rows
        if reference_kv_state is not None:
            extra["minimax_h3_reference_kv_post_parallel"] = (
                reference_kv_state,
                reference_kv_layer_index,
                reference_kv_compact,
            )
            if reference_kv_compact:
                kv_video_layout = getattr(reference_kv_state, "_reference_kv_video_layout", None)
                if kv_video_layout is not None:
                    extra["minimax_h3_reference_kv_video_layout"] = kv_video_layout

        metadata = AttentionMetadata(
            attn_mask=attn_mask,
            packed_padding=(
                PackedPaddingMetadata(
                    q_length=used_q,
                    kv_length=used_kv,
                    cu_seqlens_q=cu_seqlens[:2],
                    cu_seqlens_k=kv_cu_seqlens[:2],
                )
                if mask_free_packed_padding
                else None
            ),
            extra=extra,
            video_layout=video_layout,
        )
        heads = int(q.shape[1])
        # Head grouping runs the layer once per group; the reference-KV cache records
        # captured K/V per layer, so it stays on the single-call path.
        groups = 1 if reference_kv_state is not None else attention_head_chunks(heads)
        # Grouping splits the head axis, so it is only defined when Q and K/V have the
        # same head count. Under GQA a KV head is shared by several Q heads and an even
        # split would pair the wrong ones; H3 is MHA today, and this keeps the knob from
        # silently doing that if it ever is not.
        if groups == 1 or heads != int(k.shape[1]):
            out = self.attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                metadata,
            ).squeeze(0)
            if out_padding:
                out = nn.functional.pad(out, (0, 0, 0, 0, 0, out_padding))
            return out

        step = (heads + groups - 1) // groups
        gate = metadata.extra.get("gate_compress")
        out = None
        for start in range(0, heads, step):
            stop = min(start + step, heads)
            group_metadata = metadata
            if gate is not None:
                # The VSA gate is per-head; a group must see its own slice, and the
                # metadata is copied rather than mutated so a backend that retains it
                # cannot observe another group's gate.
                group_metadata = replace(metadata, extra={**metadata.extra, "gate_compress": gate[:, :, start:stop]})
            part = self.attention(
                q[:, start:stop].unsqueeze(0),
                k[:, start:stop].unsqueeze(0),
                v[:, start:stop].unsqueeze(0),
                group_metadata,
            ).squeeze(0)
            if out is None:
                out = torch.empty((part.shape[0], heads, part.shape[-1]), dtype=part.dtype, device=part.device)
            out[:, start:stop] = part
            del part
        assert out is not None
        if out_padding:
            out = nn.functional.pad(out, (0, 0, 0, 0, 0, out_padding))
        return out

    def _project_qkv(
        self, x: torch.Tensor, rope_table: torch.Tensor | None, vdn_window: VDNLayout | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project/normalize/rotate QKV, retaining raw Q/K for the VDN branch."""
        total = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_kv_heads, self.head_dim)
        v = v.view(total, self.num_kv_heads, self.head_dim)
        # Only the upstream VDNH3 branch consumes `vdn_window`; our VDN_WINDOW_ATTN
        # branch derives its own geometry from `video_layout` (see
        # ``VDNHybridAttention.plan_geometry``), so requiring the window for both classes
        # would reject calls that our class handles correctly.
        if isinstance(self.vdn, VDNH3HybridAttention) and vdn_window is None:
            raise ValueError(f"{self._prefix} is VDN-H3 hybrid attention but received no VDN window")
        # The VDN linear branch reads the raw projections; norm/RoPE are out of place.
        q_raw, k_raw = q, k
        if rope_table is None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        else:
            q, k = fused_qk_norm_rope(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                rope_table,
                self.q_norm.variance_epsilon,
            )
        return q, k, v, q_raw, k_raw

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_table: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_total: int | None = None,
        num_requests: int = 1,
        sp_seq_lens: list[int] | None = None,
        video_layout: VideoTokenLayout | None = None,
        vsa_prefix_segments: tuple[int, ...] = (),
        reference_kv_state: MiniMaxH3ReferenceKVTier2State | None = None,
        reference_kv_layer_index: int | None = None,
        reference_kv_compact: bool = False,
        vdn_window: VDNLayout | None = None,
        vdn_text_span: tuple[int, int] | torch.Tensor | None = None,
        sp_row_span: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """x: [T, hidden] packed thd rows -> [T, hidden].

        Operation order: fused qkv projection -> per-head q/k RMSNorm -> RoPE
        on q/k -> variable-length non-causal flash attention -> output projection.

        With Ulysses sequence parallelism, x holds this rank's row shard;
        qkv/norm/RoPE run locally, an all-to-all trades sequence for heads.
        Each rank attends the full sequence with heads/world_size local heads,
        so cu_seqlens retains global packed-document semantics. The inverse
        all-to-all restores the row shard before the output projection.

        On a VDN checkpoint the attention is hybrid. The upstream VDNH3_ATTN half runs
        the window in the kernel and adds its learned branch after ``out_proj``; our
        VDN_WINDOW_ATTN half runs the window in the packed call, scales it back toward
        the softmax mass it actually captured, and adds the linear branch's partial
        before the single all-reduce. See ``vdnh3`` and ``vdn_hybrid``.
        """
        total = x.shape[0]
        if reference_kv_compact:
            if reference_kv_state is None or reference_kv_layer_index is None:
                raise RuntimeError("compacted reference KV requires state and layer index")
            reference_kv_state.record_projection_skip(
                reference_kv_layer_index,
                reference_kv_state.local_sequence_rows,
                projected_rows=total,
            )
        q, k, v, q_raw, k_raw = self._project_qkv(x, rope_table, vdn_window)
        # Our VDN_WINDOW_ATTN branch consumes the RAW projections -- pre-QK-norm, pre-RoPE
        # -- so it adds no projection cost and, under a merged LoRA, sees the adapted
        # weights for free. Holding the references is safe because the fused kernel
        # declares ``mutates_args=[]`` and the unfused path is out-of-place too. The
        # upstream VDNH3 branch takes its own (q_raw, k_raw) plus ``v`` below.
        qkv_raw = (q_raw, k_raw, v) if isinstance(self.vdn, VDNHybridAttention) else None

        # The gate is projected from the same local rows as Q. Pure Ulysses
        # reshards it alongside Q/K/V in UlyssesParallelAttention so each VSA
        # rank receives the full sequence for its local head shard.
        gate_compress = None
        if self.to_gate_compress is not None:
            gate_result = self.to_gate_compress(x)
            gate_compress = gate_result[0] if isinstance(gate_result, tuple) else gate_result
            gate_compress = gate_compress.view(total, self.num_heads, self.head_dim)

        # Each request contributes a document for its rows plus one for any
        # nonempty alignment padding. Local/Ulysses backends unpad it, while
        # Ring keeps aligned rows for fixed-size P2P buffers.
        out = self._run_packed_attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            # Before Ulysses, q contains only this rank's row shard. The
            # backend receives the global sequence after all-to-all, so carry
            # its Python length explicitly instead of inferring it from q.
            packed_total=packed_total if packed_total is not None else q.shape[0],
            num_requests=num_requests,
            video_layout=video_layout,
            vsa_prefix_segments=vsa_prefix_segments,
            gate_compress=gate_compress,
            reference_kv_state=reference_kv_state,
            reference_kv_layer_index=reference_kv_layer_index,
            reference_kv_compact=reference_kv_compact,
            vdn_window=vdn_window,
        )
        if self.vdn is None:
            out = out.reshape(total, self.num_heads * self.head_dim)
            out, _ = self.out_proj(out)
            return out

        if isinstance(self.vdn, VDNH3HybridAttention):
            # Upstream VDNH3_ATTN: gate the softmax half, then add the learned linear half.
            out = self.vdn.gate_softmax(out, x)
            out = out.reshape(total, self.num_heads * self.head_dim)
            out, _ = self.out_proj(out)
            out = self.vdn.add_linear(out, x, q_raw, k_raw, v, vdn_window)
            return out

        # Our VDN_WINDOW_ATTN path (``vdn_hybrid.VDNHybridAttention``).
        # `used_len` the same way _run_packed_attention and the window backend derive
        # it, so all three agree on where the packing's alignment padding starts.
        used_len = min(max_seqlen, packed_total if packed_total is not None else total)
        geometry = self.vdn.plan_geometry(video_layout, used_len)
        # out_proj no longer reduces (enable_vdn_branch), so this is a partial sum.
        out, _ = self.out_proj(self.vdn.gate_softmax(out, x))
        branch_partial = video_span = None
        if geometry is not None:
            # The branch reads x by row, so being told "not sharded" while holding a shard
            # is unrecoverable and silent: it would summarise the wrong frames and render.
            # sp_row_span is threaded from the one place that knows both lengths, and this
            # catches any future path that splits the rows without threading it -- a
            # non-strict ulysses_mode, or a sequence the even split cannot divide.
            if sp_row_span is None and packed_total is not None and total != packed_total:
                raise RuntimeError(
                    f"the VDN branch holds {total} of {packed_total} packed rows but was given no "
                    "sequence-parallel span. Some sharding path did not thread sp_row_span through; "
                    "running on would slice this shard by global video offsets."
                )
            # sp_row_span is set only when x really is a row shard -- the caller decides,
            # because only it sees both the local and the global length. The branch then
            # cannot slice the video by its global span; the sharded path returns LOCAL row
            # bounds, which is what combine_hybrid_output needs to place the partial anyway.
            result = (
                self.vdn.readout_row_shard(
                    x, qkv_raw, geometry=geometry, text_span=vdn_text_span, local_span=sp_row_span
                )
                if sp_row_span is not None
                else self.vdn.readout(x, qkv_raw, geometry=geometry, text_span=vdn_text_span)
            )
            if result is not None:
                branch_partial, start, stop = result
                video_span = (start, stop)
        return combine_hybrid_output(out, branch_partial, video_span)


class MiniMaxH3MLP(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.fc1 = MergedColumnParallelLinear(
            arch.hidden_size,
            [arch.ffn_hidden_size, arch.ffn_hidden_size],
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.act_fn = SiluAndMul()
        # Chunk the fused fc1 output as [gate, up], then compute
        # silu(gate) * up.
        self.fc2 = RowParallelLinear(
            arch.ffn_hidden_size,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = ffn_row_chunks(int(x.shape[0]))
        if chunks == 1:
            hidden, _ = self.fc1(x)
            hidden = self.act_fn(hidden)
            out, _ = self.fc2(hidden)
            return out

        # Chunk fc1 and the activation; run fc2 ONCE on the whole tensor.
        #
        # fc1 is column-parallel with gather_output=False, so it performs no collective
        # and a row slice of it is exact. fc2 is a RowParallelLinear that all-reduces
        # internally: calling it per chunk would turn one collective into N, and NCCL
        # selects its reduction algorithm by message size -- so the cross-rank summation
        # order, and its rounding, would change with the chunk count. Measured on 5s/1088p
        # with TP4: chunking through fc2 left the video bit-identical but moved the audio
        # by 3.7% of full scale. Keeping fc2 whole costs the activation buffer and makes
        # the result byte-for-byte the unchunked one.
        rows = int(x.shape[0])
        step = (rows + chunks - 1) // chunks
        activated = None
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            part, _ = self.fc1(x[start:stop])
            part = self.act_fn(part)
            if activated is None:
                activated = torch.empty((rows, part.shape[-1]), dtype=part.dtype, device=part.device)
            activated[start:stop] = part
            del part
        assert activated is not None
        out, _ = self.fc2(activated)
        return out


class MiniMaxH3AdalnProj(nn.Module):
    """SiLU + zero-init linear over unique condition embeddings.

    Per block, three modalities each produce six H-wide vectors:
    [M, t_dim] -> [M, 3*6H] -> view(M*3, 6H) -> chunk(6).
    The final layer uses one modality and produces two H-wide vectors:
    [M, t_dim] -> [M, 2H] -> chunk(2).
    """

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        out_features: int,
        quant_config: QuantizationConfig | None,
        *,
        expand_ratio: int,
        modality_num: int,
        prefix: str,
        adaln_cache: MiniMaxH3RuntimeAdalnCache | None = None,
    ) -> None:
        super().__init__()
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError(
                f"adaln out_features mismatch: {out_features} != {expand_ratio}*{arch.hidden_size}*{modality_num}"
            )
        self._adaln_cache = adaln_cache
        self._offload_weights = adaln_cache is not None and adaln_cache.offload_weights
        self._weight_stager: PinnedModuleStager | None = None
        self._host_signature: tuple[Any, ...] | None = None
        if self._offload_weights and quant_config is not None:
            raise ValueError("MiniMax H3 AdaLN offload requires unquantized BF16 weights")
        self._cache_name = prefix + ".linear"
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        # An AdaLN-pruned checkpoint replaces the released ``time_embed_dim`` input with
        # ``adaln_rank`` affine coordinates and moves the linear's bias into an fp32
        # ``folded_bias`` buffer that the forward adds before rounding.
        self.pruned = arch.adaln_rank is not None
        # Allocate host weights from the start: moving them after construction
        # would still require enough VRAM for the complete resident model.
        with torch.device("cpu") if self._offload_weights else nullcontext():
            self.linear = ColumnParallelLinear(
                arch.adaln_rank if self.pruned else arch.time_embed_dim,
                out_features,
                bias=not self.pruned,
                gather_output=True,
                params_dtype=_BF16_DTYPE,
                quant_config=quant_config,
                prefix=f"{prefix}.linear",
            )
        if self.pruned:
            self.register_buffer(
                "folded_bias",
                torch.zeros(out_features, dtype=_FP32_DTYPE),
                persistent=True,
            )

    def _apply(self, fn, recurse=True):
        self._weight_stager = None
        self._host_signature = None
        if self._adaln_cache is not None:
            self._adaln_cache.clear()
        if not self._offload_weights:
            return super()._apply(fn, recurse=recurse)

        def keep_on_host(tensor):
            result = fn(tensor)
            return result if result.is_meta else result.cpu()

        return super()._apply(keep_on_host, recurse=recurse)

    def _offload_signature(self) -> tuple[Any, ...] | None:
        assert self._adaln_cache is not None
        try:
            return self._adaln_cache._signature(self.linear)
        except RuntimeError:
            # Weights created in inference_mode have no version counter.
            # Re-snapshot on every miss instead of trusting an immutable master.
            return None

    def _prepare_weight_stager(self, device: torch.device) -> None:
        if not self._offload_weights:
            return
        if device.type != "cuda" or torch.is_grad_enabled() or torch.compiler.is_compiling():
            raise RuntimeError("MiniMax H3 AdaLN offload requires eager CUDA inference")
        assert self._adaln_cache is not None
        if self._adaln_cache._has_forward_hooks(self.linear):
            raise RuntimeError("MiniMax H3 AdaLN offload does not support projection forward hooks")
        signature = self._offload_signature()
        if (
            signature is None
            or self._weight_stager is None
            or self._weight_stager.device != device
            or self._host_signature != signature
        ):
            # Re-snapshot on parameter replacement, in-place edits or moves.
            # The CPU master remains the registered parameter storage between
            # calls, so normal cache version checks and weight loaders apply.
            # The whole projector is one staging group: `_apply`'s keep_on_host
            # holds the pruned `folded_bias` buffer on the host too, and the
            # pruned path adds it to the staged activation, so it has to move
            # with the linear.
            self._weight_stager = PinnedModuleStager(self, device, cache_retention=BoundedAllocatorCache(device))
            self._host_signature = self._offload_signature()

    def _project(self, t_emb: torch.Tensor) -> torch.Tensor:
        self._prepare_weight_stager(t_emb.device)

        # The pruned table contains the coordinates of silu(time_embedder(t)),
        # whereas the released path still needs the activation here.
        def staged(compute: Callable[[], torch.Tensor]) -> torch.Tensor:
            """Run ``compute`` with the projector on the device when offloaded.

            AdaLN offload keeps the whole projector -- the column-parallel linear
            and, on pruned checkpoints, the ``folded_bias`` buffer -- on a pinned
            CPU master. Both are read by one projection, so the staging envelope
            has to cover the bias add as well as the matmul; staging only around
            the linear leaves the bias on the host and the add fails.
            """
            if self._weight_stager is None:
                return compute()
            self._weight_stager.load()
            try:
                return compute()
            finally:
                self._weight_stager.offload()

        def project() -> torch.Tensor:
            if self.pruned:
                # This order is part of the checkpoint semantics: the folded bias
                # carries most of the modulation and must not be rounded to BF16
                # before addition.
                def pruned() -> torch.Tensor:
                    x = self.linear(t_emb.to(_BF16_DTYPE))[0]
                    return (x.float() + self.folded_bias).to(x.dtype)

                return staged(pruned)
            x = nn.functional.silu(t_emb)

            def released() -> torch.Tensor:
                return self.linear(x.to(_BF16_DTYPE))[0]

            return staged(released)

        return (
            project()
            if self._adaln_cache is None
            else self._adaln_cache.project(self._cache_name, self.linear, t_emb, project)
        )

    @torch.compiler.disable
    def _project_cached(self, t_emb: torch.Tensor) -> torch.Tensor:
        # Cache decisions, TP hit voting and CPU storage rebinding are host
        # control flow. Keep this small boundary eager while the surrounding
        # norms, attention projections and MLP remain eligible for compilation.
        return self._project(t_emb)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """t_emb: [M, t_dim] -> expand_ratio tensors of [M*modality_num, H]."""
        use_cache_boundary = self._adaln_cache is not None and (
            self._adaln_cache.max_bytes > 0 or self._offload_weights
        )
        x = self._project_cached(t_emb) if use_cache_boundary else self._project(t_emb)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
        layer_index: int,
        adaln_cache: MiniMaxH3RuntimeAdalnCache | None = None,
        attention_cls: type[MiniMaxH3Attention] = MiniMaxH3Attention,
    ) -> None:
        super().__init__()
        self.layer_index = int(layer_index)
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        # The prefix also carries the block index that block-sparse attention
        # backends match against their skip_layers selector.
        # ``attention_cls`` lets a derived model (for example the TaoMate-H3
        # streaming variant) swap in a subclass of ``MiniMaxH3Attention`` that
        # keeps the checkpoint layout but changes the attention kernel path.
        self.attn = attention_cls(
            arch,
            quant_config,
            prefix=f"{prefix}.attn",
        )
        # Startup checks for attention schedule candidates. Only DiT blocks see
        # padded packed documents and can carry a VSA gate. The token refiner's
        # text rows are unpadded, and FASTVIDEO_VSA falls back to dense SDPA
        # there for the baseline too, so the refiner registers neither check.
        self.attn.attention.add_schedule_candidate_check(_candidate_excludes_packed_padding)
        self.attn.attention.add_schedule_candidate_check(self.attn._vsa_candidate_needs_gate)
        self.mlp = MiniMaxH3MLP(
            arch,
            quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            quant_config,
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
            prefix=f"{prefix}.adaln_proj",
            adaln_cache=adaln_cache,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_table: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        packed_total: int,
        num_requests: int = 1,
        sp_seq_lens: list[int] | None = None,
        video_layout: VideoTokenLayout | None = None,
        vsa_prefix_segments: tuple[int, ...] = (),
        reference_kv_state: MiniMaxH3ReferenceKVTier2State | None = None,
        reference_kv_compact: bool = False,
        vdn_window: VDNLayout | None = None,
        vdn_text_span: tuple[int, int] | torch.Tensor | None = None,
        sp_row_span: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """x: [T, H]; t_emb: [M, t_dim]; combined_indices: [T]
        (= inverse_indices * modality_num + token_tags.clamp(min=0)).

        Each block computes AdaLN parameters once, then applies
        norm1 -> scale/shift -> attention -> gated residual, followed by
        norm2 -> scale/shift -> MLP -> gated residual.
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_proj(t_emb)

        residual = x
        h = rms_norm_indexed_scale_shift(
            x,
            self.norm1.weight,
            shift_msa,
            scale_msa,
            combined_indices,
            self.norm1.variance_epsilon,
        )
        h = self.attn(
            h,
            rope_table=rope_table,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            packed_total=packed_total,
            num_requests=num_requests,
            sp_seq_lens=sp_seq_lens,
            video_layout=video_layout,
            vsa_prefix_segments=vsa_prefix_segments,
            reference_kv_state=reference_kv_state,
            reference_kv_layer_index=self.layer_index,
            reference_kv_compact=reference_kv_compact,
            vdn_window=vdn_window,
            vdn_text_span=vdn_text_span,
            sp_row_span=sp_row_span,
        )
        x, h = indexed_gate_rms_norm_scale_shift(
            residual,
            gate_msa,
            h,
            self.norm2.weight,
            shift_mlp,
            scale_mlp,
            combined_indices,
            self.norm2.variance_epsilon,
        )
        residual = x
        h = self.mlp(h)
        return indexed_gate(residual, gate_mlp, h, combined_indices)
