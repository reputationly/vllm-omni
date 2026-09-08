# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Our window softmax against OpenVDN's own reference, not against our own definition.

``test_minimax_h3_vdn_window.py`` checks that ``build_window_plan`` compiles the mask
faithfully -- but it checks it against a mask this repository also wrote. If our reading
of the geometry differs from the released model's, both sides of that test are wrong
together and it stays green. The branch has had a cross-implementation oracle since the
port began (``test_..._branch_parity.py``); the softmax half had none, which left the
half that carries the local detail resting on our own restatement of the rules.

So the oracle here is ``window_softmax_reference`` from the released checkout: point
``VDN_REFERENCE_ROOT`` at it and require our backend's output to match theirs on the same
q/k/v. Anchor modes and a clip long enough to exercise several chunks are parametrised,
because the places the two definitions could drift apart are the clip ends and the
anchor columns.
"""

import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.gpu, pytest.mark.diffusion]

HEADS, HEAD_DIM = 4, 64
TOKENS_PER_FRAME = 6
PREFIX, SUFFIX = 5, 4  # globals on both sides: text before the video, audio after


def _reference():
    root = os.environ.get("VDN_REFERENCE_ROOT") or ""
    if not root or not os.path.isdir(os.path.join(root, "src", "models", "softmax_attention")):
        pytest.skip("set VDN_REFERENCE_ROOT to a checkout of OpenVDN/vdn-minimax-h3")
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.models.sequence_layout import SequenceLayout
    from src.models.softmax_attention import window_bounds, window_softmax_reference

    return SequenceLayout, window_bounds, window_softmax_reference


def _impl(**backend_kwargs):
    from vllm_omni.diffusion.attention.backends.vdn_window_attn import VDNWindowAttentionImpl

    return VDNWindowAttentionImpl(
        num_heads=HEADS,
        head_size=HEAD_DIM,
        softmax_scale=HEAD_DIM**-0.5,
        causal=False,
        qkv_layout="BSND",
        backend_kwargs=backend_kwargs or None,
    )


def _metadata(*, used_len, video_start, frames):
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout

    cu = torch.tensor([0, used_len], dtype=torch.int32, device="cuda")
    return AttentionMetadata(
        video_layout=VideoTokenLayout(prefix_len=video_start, latent_grid=(frames, 2, 3)),
        extra={
            "cu_seqlens_q": cu,
            "cu_seqlens_k": cu,
            "max_seqlen_q": used_len,
            "max_seqlen_k": used_len,
            "valid_kv_length": used_len,
        },
    )


# 17 frames over chunk=5 gives four chunks, two of them clipped by the clip ends; 22
# gives a fully interior chunk as well. Both are past the point where the window stops
# covering everything, which is where the two definitions can disagree at all.
@pytest.mark.parametrize("frames", [17, 22])
@pytest.mark.parametrize("anchor_frames", ["both", "columns", "rows", "none"])
def test_window_softmax_matches_the_released_reference(frames, anchor_frames):
    if not torch.cuda.is_available():
        pytest.skip("VDN_WINDOW_ATTN is a CUDA backend")
    SequenceLayout, window_bounds, window_softmax_reference = _reference()

    from vllm_omni.diffusion.models.minimax_h3.vdn_window import VDN_CHUNK, VDN_RADIUS

    used_len = PREFIX + frames * TOKENS_PER_FRAME + SUFFIX
    torch.manual_seed(7)
    shape = (1, used_len, HEADS, HEAD_DIM)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    # The keys the backend actually reads. Spelling them `chunk`/`anchor_frames` here
    # silently leaves the impl on its defaults, which turns every non-default case into
    # a comparison of our "both" against their other mode -- six red tests that say
    # nothing, and a green "both" that is real but no longer evidence of plumbing.
    impl = _impl(vdn_chunk=VDN_CHUNK, vdn_radius=VDN_RADIUS, vdn_anchor_frames=anchor_frames)
    assert impl.window.anchor_frames == anchor_frames
    ours = impl.forward_cuda(q, k, v, _metadata(used_len=used_len, video_start=PREFIX, frames=frames))

    layout = SequenceLayout(
        seq_len=used_len,
        video_start=PREFIX,
        num_frames=frames,
        tokens_per_frame=TOKENS_PER_FRAME,
        frame_height=2,
        frame_width=3,
    )
    theirs = window_softmax_reference(
        q[0],
        k[0],
        v[0],
        layout,
        window_bounds(frames, VDN_RADIUS, VDN_CHUNK),
        impl.softmax_scale,
        anchor_frames=anchor_frames,
    )

    # bf16 over a few hundred keys: the two group their keys differently (they run one
    # SDPA per frame, we run one varlen call per chunk), so the summation order differs
    # and exactness is not on offer. A geometry disagreement is not a rounding-scale
    # difference -- one dropped or duplicated frame moves a row by O(1/window).
    torch.testing.assert_close(ours[0].float(), theirs.float(), rtol=2e-2, atol=2e-2)
