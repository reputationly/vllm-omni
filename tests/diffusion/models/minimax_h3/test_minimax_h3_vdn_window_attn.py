# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""VDN_WINDOW_ATTN must equal the masked softmax it is a decomposition of.

The backend never builds the mask: it splits the query rows into groups that share a
kept-key set and runs each as a dense FlashAttention call. That is only the same
arithmetic if every query's softmax spans exactly its kept set and no key is counted
twice, so these tests compare against an explicitly masked attention over the same
inputs. A duplicated key would inflate that key's share of the denominator by a factor
of two and still produce a perfectly plausible tensor.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.gpu, pytest.mark.diffusion]

HEADS, HEAD_DIM = 4, 64
TOKENS_PER_FRAME = 6
PREFIX, PAD = 5, 3


def _impl(**window_kwargs):
    from vllm_omni.diffusion.attention.backends.vdn_window_attn import VDNWindowAttentionImpl

    return VDNWindowAttentionImpl(
        num_heads=HEADS,
        head_size=HEAD_DIM,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        qkv_layout="BSND",
        backend_kwargs=window_kwargs or None,
    )


def _packed_extra(used_len):
    """The packed metadata MiniMaxH3Attention._run_packed_attention publishes.

    All four keys or none: the dense fallback rejects a partial set, so a fixture that
    supplied only ``max_seqlen_q`` would exercise a shape production never produces.
    """
    cu_seqlens = torch.tensor([0, used_len], dtype=torch.int32, device="cuda")
    return {
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
        "max_seqlen_q": used_len,
        "max_seqlen_k": used_len,
        "valid_kv_length": used_len,
    }


def _metadata(*, used_len, video_start, frames, spans: bool):
    from vllm_omni.diffusion.attention.backends.abstract import (
        AttentionMetadata,
        VideoTokenLayout,
        VideoTokenSpan,
    )

    grid = (frames, 2, 3)  # 2*3 == TOKENS_PER_FRAME
    if spans:
        layout = VideoTokenLayout(
            used_len=used_len,
            video_spans=(VideoTokenSpan(start=video_start, latent_grid=grid, role="target"),),
        )
    else:
        layout = VideoTokenLayout(prefix_len=video_start, latent_grid=grid)
    return AttentionMetadata(video_layout=layout, extra=_packed_extra(used_len))


def _masked_reference(q, k, v, mask, scale):
    """Attention with the mask applied explicitly, in fp32 so bf16 is the only error."""
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", weights, v.float())


def _mask_from_plan(plan, used_len, device):
    mask = torch.zeros(used_len, used_len, dtype=torch.bool, device=device)
    for start, stop in plan.dense_q_ranges:
        mask[start:stop] = True
    for group in plan.window_groups:
        q_start, q_stop = group.q_range
        for kv_start, kv_stop in group.kv_ranges:
            mask[q_start:q_stop, kv_start:kv_stop] = True
    return mask


@pytest.mark.parametrize("spans", [False, True], ids=["t2va_prefix_layout", "ref2va_span_layout"])
def test_window_output_matches_masked_attention(spans):
    if not torch.cuda.is_available():
        pytest.skip("VDN_WINDOW_ATTN is a CUDA backend")
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import build_window_plan

    frames = 17
    used_len = PREFIX + frames * TOKENS_PER_FRAME
    total = used_len + PAD
    torch.manual_seed(0)
    shape = (1, total, HEADS, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    impl = _impl()
    out = impl.forward_cuda(q, k, v, _metadata(used_len=used_len, video_start=PREFIX, frames=frames, spans=spans))

    plan = build_window_plan(
        used_len=used_len, video_start=PREFIX, num_frames=frames, tokens_per_frame=TOKENS_PER_FRAME
    )
    assert not plan.is_dense, "this shape is meant to exercise the windowed path"
    reference = _masked_reference(
        q[0, :used_len],
        k[0, :used_len],
        v[0, :used_len],
        _mask_from_plan(plan, used_len, q.device),
        impl.softmax_scale,
    )
    torch.testing.assert_close(out[0, :used_len].float(), reference, rtol=2e-2, atol=2e-2)


def test_alignment_padding_rows_stay_zero():
    """Rows past ``used_len`` belong to no group. They must not carry stale memory: the
    caller feeds this straight into the output projection and the residual stream."""
    if not torch.cuda.is_available():
        pytest.skip("VDN_WINDOW_ATTN is a CUDA backend")

    frames = 17
    used_len = PREFIX + frames * TOKENS_PER_FRAME
    torch.manual_seed(1)
    shape = (1, used_len + PAD, HEADS, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    out = _impl().forward_cuda(q, k, v, _metadata(used_len=used_len, video_start=PREFIX, frames=frames, spans=False))
    assert torch.count_nonzero(out[0, used_len:]) == 0


def test_short_clip_falls_back_to_dense_attention():
    """When the window covers the clip the result must be the dense kernel's own output,
    not a re-derivation of it: matching the kernel is what keeps that case exact."""
    if not torch.cuda.is_available():
        pytest.skip("VDN_WINDOW_ATTN is a CUDA backend")

    frames = 8  # a 15-frame window spans this clip
    used_len = PREFIX + frames * TOKENS_PER_FRAME
    torch.manual_seed(2)
    shape = (1, used_len, HEADS, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    impl = _impl()
    metadata = _metadata(used_len=used_len, video_start=PREFIX, frames=frames, spans=False)
    assert impl._resolve_plan(q, metadata) is None
    torch.testing.assert_close(
        impl.forward_cuda(q, k, v, metadata),
        impl.dense_fallback.forward_cuda(q, k, v, metadata),
    )


def test_missing_video_layout_falls_back_to_dense():
    """The token refiner attends over text only -- no frame axis, so no window."""
    if not torch.cuda.is_available():
        pytest.skip("VDN_WINDOW_ATTN is a CUDA backend")
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

    torch.manual_seed(3)
    shape = (1, 64, HEADS, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    impl = _impl()
    metadata = AttentionMetadata(extra=_packed_extra(64))
    assert impl._resolve_plan(q, metadata) is None
    torch.testing.assert_close(
        impl.forward_cuda(q, k, v, metadata),
        impl.dense_fallback.forward_cuda(q, k, v, metadata),
    )


def test_backend_declares_packed_mask_free():
    """Without this, every packed request with tail padding fails at serve time.

    MiniMax-H3 builds a padding ``attn_mask`` for any backend that does not advertise
    mask-free packed handling, and then the attention layer rejects a backend that
    cannot consume one — so the failure is a 500 on the first real request while every
    shape-level test still passes. The declaration is honest: ``_resolve_plan`` takes
    the used length from the packed metadata and writes the pad rows as zeros.
    """
    from vllm_omni.diffusion.attention.backends.vdn_window_attn import VDNWindowAttentionBackend

    if not torch.cuda.is_available():
        pytest.skip("the declaration is CUDA-only by design")
    assert VDNWindowAttentionBackend.supports_packed_mask_free() is True


def test_causal_role_is_refused_rather_than_silently_windowed():
    with pytest.raises(ValueError, match="causal"):
        from vllm_omni.diffusion.attention.backends.vdn_window_attn import VDNWindowAttentionImpl

        VDNWindowAttentionImpl(
            num_heads=HEADS,
            head_size=HEAD_DIM,
            softmax_scale=HEAD_DIM**-0.5,
            causal=True,
            qkv_layout="BSND",
        )
