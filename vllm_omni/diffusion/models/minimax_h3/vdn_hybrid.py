# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The VDN-specific half of a hybrid MiniMax-H3 attention layer.

``MiniMaxH3Attention`` keeps its own projections, QK-norm, RoPE and packed attention
call. This module adds what VDN-H3 puts around them::

    softmax_out = out_proj( softmax_gate(x) * window_attention(q, k, v) )
    linear_out  = to_out_linear( linear_branch(x, raw q/k/v) )
    out         = softmax_out;  out[video rows] += linear_out

The window lives in ``vdn_window_attn`` (the attention backend) and the branch in
``vdn_branch``; both read their geometry from ``vdn_window.build_window_plan``, so the
two halves cannot disagree about which frames the softmax already covered.

**The two projections share one all-reduce.** Under tensor parallelism ``out_proj`` and
``to_out_linear`` are both row-parallel, so the naive spelling reduces twice. Measured
on 4x A100-PCIE at the 15 s shape, one all-reduce of the ``[109574, 5376]`` activation
is 133 ms against a 481 ms block -- a second one would give back a third of everything
the window buys. Both are therefore built with ``reduce_results=False`` and their
partial sums are added before a single collective.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout
from vllm_omni.diffusion.models.minimax_h3.vdn_branch import (
    VDN_GATE_SHARD_AXES,
    VDNLinearBranch,
    VDNOutputGate,
    attach_head_shard_loaders,
)
from vllm_omni.diffusion.models.minimax_h3.vdn_window import (
    VDN_ANCHOR_FRAMES,
    VDN_CHUNK,
    VDN_RADIUS,
    build_window_plan,
    window_bounds,
)

logger = init_logger(__name__)

# The window backend this branch is the complement of. Pairing the branch with any
# other attention double-counts the frames the window already covered: the softmax side
# would see the whole clip AND the branch would summarise it again.
REQUIRED_ATTENTION_BACKEND = "VDN_WINDOW_ATTN"


class VDNHybridAttention(nn.Module):
    """The gate, the linear branch and the branch's output projection, for one block."""

    def __init__(
        self,
        *,
        hidden_size: int,
        total_heads: int,
        head_dim: int,
        params_dtype: torch.dtype,
        quant_config=None,
        prefix: str = "",
        head_shard: tuple[int, int] | None = None,
        chunk: int = VDN_CHUNK,
        radius: int = VDN_RADIUS,
        anchor_frames: str = VDN_ANCHOR_FRAMES,
        short_conv: tuple[str, ...] = ("k", "v"),
        enable_text_state: bool = True,
    ) -> None:
        super().__init__()
        self.chunk, self.radius, self.anchor_frames = chunk, radius, anchor_frames
        self.linear_attention = VDNLinearBranch(
            hidden_size,
            total_heads,
            head_dim,
            short_conv=short_conv,
            enable_text_state=enable_text_state,
            params_dtype=params_dtype,
            head_shard=head_shard,
        )
        local_heads = self.linear_attention.local_heads
        # Per head, direct rather than low rank: the windowed softmax renormalises to 1
        # no matter how little mass it saw, so this scales that branch back toward the
        # share it captured -- a property of a distribution.
        self.softmax_gate = VDNOutputGate(hidden_size, local_heads, head_dim=None, bias=True, dtype=params_dtype)
        attach_head_shard_loaders(
            self,
            VDN_GATE_SHARD_AXES,
            head_start=self.linear_attention.head_start,
            local_heads=local_heads,
            head_dim=head_dim,
        )
        # to_out_linear is row-parallel: it consumes the branch's head-sharded channels,
        # so vLLM's own loader already slices it on the matching axis.
        #
        # quant_config is deliberately NOT passed. The branch is BF16 by design, and on a
        # quantized base this is the difference between working and not: the offline
        # quantizer derives ``ignored_layers`` from the tensors in the SOURCE checkpoint,
        # and this projection is not in it -- it only exists once enable_vdn_branch runs.
        # Built with the base's Int8 config it would expect a weight_scale the VDN
        # artifact never carries and strand its parameters on the meta device.
        del quant_config
        self.to_out_linear = RowParallelLinear(
            total_heads * head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            # See the module docstring: this shares out_proj's collective.
            reduce_results=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.to_out_linear",
        )

    def gate_softmax(self, attn_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """[T, H, d] attention output -> [T, H*d], scaled by the per-head mass gate."""
        gate = self.softmax_gate(x)
        return (attn_out * gate.to(attn_out.dtype)).reshape(attn_out.shape[0], -1)

    def plan_geometry(
        self, video_layout: VideoTokenLayout | None, used_len: int
    ) -> tuple[int, int, int, tuple[int, int]] | None:
        """(video_start, num_frames, tokens_per_frame, frame_size), or None to stay off.

        None means this shape has no windowed frames -- a clip shorter than the window,
        or a layer with no video at all -- and then the softmax side IS the full
        attention, so the branch must contribute nothing rather than something small.
        """
        if video_layout is None:
            return None
        if video_layout.video_spans:
            target = next(
                (span for span in reversed(video_layout.video_spans) if span.role == "target"),
                None,
            )
            if target is None:
                return None
            video_start, grid = int(target.start), target.latent_grid
        else:
            if video_layout.prefix_len is None or video_layout.latent_grid is None:
                return None
            video_start, grid = int(video_layout.prefix_len), video_layout.latent_grid

        frames, grid_h, grid_w = (int(dim) for dim in grid)
        tokens_per_frame = grid_h * grid_w
        plan = build_window_plan(
            used_len=used_len,
            video_start=video_start,
            num_frames=frames,
            tokens_per_frame=tokens_per_frame,
            chunk=self.chunk,
            radius=self.radius,
            anchor_frames=self.anchor_frames,
        )
        if plan.is_dense:
            return None
        return video_start, frames, tokens_per_frame, (grid_h, grid_w)

    def readout(
        self,
        x: torch.Tensor,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        geometry: tuple[int, int, int, tuple[int, int]],
        text_span: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, int, int]:
        """The branch's contribution to the residual stream, and the rows it covers.

        Returns a PARTIAL sum: ``to_out_linear`` does not reduce, so the caller must add
        this into the attention's own partial and reduce once.
        """
        video_start, frames, tokens_per_frame, frame_size = geometry
        video_end = video_start + frames * tokens_per_frame

        text_x = text_qkv_raw = None
        if self.linear_attention.enable_text_state and text_span is not None:
            text_start, text_end = text_span
            text_x = x[text_start:text_end]
            text_qkv_raw = tuple(tensor[text_start:text_end] for tensor in qkv_raw)

        readout = self.linear_attention(
            x[video_start:video_end],
            tuple(tensor[video_start:video_end] for tensor in qkv_raw),
            num_frames=frames,
            tokens_per_frame=tokens_per_frame,
            frame_size=frame_size,
            bounds=window_bounds(frames, radius=self.radius, chunk=self.chunk),
            text_x=text_x,
            text_qkv_raw=text_qkv_raw,
            # The partner of anchor_frames="both": the softmax side makes frames 0 and
            # F-1 exact in both directions, so the branch drops them entirely.
            skip_ends=self.anchor_frames == "both",
        )
        projected, _ = self.to_out_linear(readout.type_as(x))
        return projected, video_start, video_end


def combine_hybrid_output(
    attention_partial: torch.Tensor,
    branch_partial: torch.Tensor | None,
    video_span: tuple[int, int] | None,
) -> torch.Tensor:
    """Add the branch into the attention's partial sum, then reduce once.

    Both operands are row-parallel partials, so the single collective at the end is what
    makes the sum correct across ranks -- and it is the same collective the dense model
    already paid for.
    """
    out = attention_partial
    if branch_partial is not None:
        if video_span is None:
            raise ValueError("a branch partial without the rows it covers cannot be placed")
        start, stop = video_span
        out[start:stop] += branch_partial
    if get_tensor_model_parallel_world_size() > 1:
        out = tensor_model_parallel_all_reduce(out)
    return out


def validate_hybrid_runtime(
    *, attention_backend: str, ulysses_degree: int, ring_degree: int, allgather_degree: int = 1
) -> None:
    """Refuse the combinations that would render a plausible but wrong video."""
    if attention_backend != REQUIRED_ATTENTION_BACKEND:
        raise ValueError(
            f"the VDN linear branch is the complement of {REQUIRED_ATTENTION_BACKEND}, but the "
            f"resolved diffusion attention backend is {attention_backend!r}. Running the branch "
            "beside a dense attention counts every frame outside the window twice; running the "
            "window without the branch drops them. Set diffusion_attention_backend: "
            f"{REQUIRED_ATTENTION_BACKEND}."
        )
    # AllGather-KV is the THIRD sequence-sharding mode and it is mutually exclusive with
    # the other two (omni_config: allgather_degree > 1 forbids ulysses/ring > 1), so a
    # check that only looked at those two saw 1 and 1 and passed. Each rank would then
    # hold a row shard while `readout` slices x and the raw q/k/v by global video spans.
    if ulysses_degree > 1 or ring_degree > 1 or allgather_degree > 1:
        raise ValueError(
            "the VDN linear branch needs every row of the target video on each rank: its scan "
            f"runs over frames. Got ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, "
            f"allgather_degree={allgather_degree}; "
            "shard with tensor parallelism instead, which divides the heads and leaves the "
            "sequence whole. (Upstream solves this with a branch-parallel Ulysses scheme that "
            "hands each rank the beta/gate/frame-mean its sequence owner computed; that is not "
            "ported.)"
        )


__all__ = [
    "REQUIRED_ATTENTION_BACKEND",
    "VDNHybridAttention",
    "combine_hybrid_output",
    "validate_hybrid_runtime",
]
