# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Our VDN linear branch against OpenVDN's own implementation, tensor by tensor.

The branch has no reference output we can diff against a rendered frame -- a wrong
forget gate or a wrongly rebased window bound produces a video that merely looks slightly
worse. So the oracle is the released code itself: point ``VDN_REFERENCE_ROOT`` at a
checkout of ``github.com/OpenVDN/vdn-minimax-h3``, give both implementations the same
random weights and inputs, and require them to agree.

    VDN_REFERENCE_ROOT=/path/to/vdn-minimax-h3 pytest tests/.../test_..._parity.py

Skipped when that checkout is absent, because the reference is not vendored here.
"""

import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HIDDEN, HEADS, HEAD_DIM = 128, 4, 32
GRID_H, GRID_W = 3, 4
TOKENS_PER_FRAME = GRID_H * GRID_W
TEXT_LEN = 7


def _reference_branch_cls():
    root = os.environ.get("VDN_REFERENCE_ROOT") or ""
    if not root or not os.path.isdir(os.path.join(root, "src", "models", "linear_attention")):
        pytest.skip("set VDN_REFERENCE_ROOT to a checkout of OpenVDN/vdn-minimax-h3")
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.models.linear_attention import BidirectionalLinearBranch

    return BidirectionalLinearBranch


def _build_pair(device, dtype):
    from vllm_omni.diffusion.models.minimax_h3.vdn_branch import VDNLinearBranch

    reference = (
        _reference_branch_cls()(
            HIDDEN,
            HEADS,
            HEAD_DIM,
            delta_rule="vdn_solve",
            bridge="alpha",
            a_fp32=True,
            short_conv=("k", "v"),
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    ours = VDNLinearBranch(HIDDEN, HEADS, HEAD_DIM, params_dtype=dtype).to(device=device, dtype=dtype).eval()

    # The parameter names are the checkpoint's on both sides -- that is the point of the
    # port -- so a state dict transfers directly. Any drift shows up here as a key error
    # rather than as a silently untrained tensor at load time.
    missing, unexpected = ours.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected, f"our branch has no home for {sorted(unexpected)}"
    assert not missing, f"our branch has parameters the reference never sets: {sorted(missing)}"
    return reference, ours


def _inputs(num_frames, device, dtype, *, with_text):
    torch.manual_seed(7)
    rows = num_frames * TOKENS_PER_FRAME

    def make(n):
        return torch.randn(n, HEADS, HEAD_DIM, device=device, dtype=dtype)

    video_x = torch.randn(rows, HIDDEN, device=device, dtype=dtype)
    qkv = (make(rows), make(rows), make(rows))
    if not with_text:
        return video_x, qkv, None, None
    text_x = torch.randn(TEXT_LEN, HIDDEN, device=device, dtype=dtype)
    text_qkv = (make(TEXT_LEN), make(TEXT_LEN), make(TEXT_LEN))
    return video_x, qkv, text_x, text_qkv


@pytest.mark.parametrize("num_frames", [13, 17])
@pytest.mark.parametrize("with_text", [True, False], ids=["text_state", "no_text_state"])
@pytest.mark.parametrize("skip_ends", [True, False], ids=["skip_ends", "all_frames"])
def test_branch_matches_the_reference(num_frames, with_text, skip_ends):
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import window_bounds

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # isolate the algorithm; bf16 agreement is its own test below
    reference, ours = _build_pair(device, dtype)
    video_x, qkv, text_x, text_qkv = _inputs(num_frames, device, dtype, with_text=with_text)
    bounds = window_bounds(num_frames)

    with torch.no_grad():
        expected = reference(
            video_x,
            num_frames,
            TOKENS_PER_FRAME,
            bounds,
            qkv_raw=qkv,
            frame_size=(GRID_H, GRID_W),
            skip_ends=skip_ends,
            text_x=text_x,
            text_qkv_raw=text_qkv,
            inference=False,
        )
        actual = ours(
            video_x,
            qkv,
            num_frames=num_frames,
            tokens_per_frame=TOKENS_PER_FRAME,
            frame_size=(GRID_H, GRID_W),
            bounds=bounds,
            text_x=text_x,
            text_qkv_raw=text_qkv,
            skip_ends=skip_ends,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_bf16_agreement_stays_at_rounding_level():
    """At the real hidden size and head_dim, in the dtype that actually ships.

    bf16 cannot be asserted elementwise: our readout is a batched matmul where the
    reference's eager path is an einsum, so the two reassociate differently. What must
    hold is that the difference stays at bf16 rounding -- roughly 1e-3 relative, the
    same order as the reference's own eager-vs-fused gap. A regression that dropped the
    scan, the statistics or alpha to bf16 would land orders of magnitude above this,
    which is the failure this test exists to catch.
    """
    if not torch.cuda.is_available():
        pytest.skip("bf16 matmul agreement is a GPU property")
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import window_bounds

    global HIDDEN, HEAD_DIM, GRID_H, GRID_W, TOKENS_PER_FRAME
    hidden, head_dim, grid = 5376, 128, (24, 42)
    num_frames = 13
    saved = (HIDDEN, HEAD_DIM, GRID_H, GRID_W, TOKENS_PER_FRAME)
    HIDDEN, HEAD_DIM = hidden, head_dim
    GRID_H, GRID_W = grid
    TOKENS_PER_FRAME = grid[0] * grid[1]
    try:
        reference, ours = _build_pair("cuda", torch.bfloat16)
        video_x, qkv, text_x, text_qkv = _inputs(num_frames, "cuda", torch.bfloat16, with_text=True)
        bounds = window_bounds(num_frames)
        with torch.no_grad():
            expected = reference(
                video_x,
                num_frames,
                TOKENS_PER_FRAME,
                bounds,
                qkv_raw=qkv,
                frame_size=grid,
                skip_ends=True,
                text_x=text_x,
                text_qkv_raw=text_qkv,
                inference=False,
            )
            actual = ours(
                video_x,
                qkv,
                num_frames=num_frames,
                tokens_per_frame=TOKENS_PER_FRAME,
                frame_size=grid,
                bounds=bounds,
                text_x=text_x,
                text_qkv_raw=text_qkv,
                skip_ends=True,
            )
    finally:
        HIDDEN, HEAD_DIM, GRID_H, GRID_W, TOKENS_PER_FRAME = saved

    error = (actual.float() - expected.float()).abs()
    magnitude = expected.float().abs().mean().clamp_min(1e-9)
    assert torch.isfinite(actual).all()
    assert (error.mean() / magnitude).item() < 5e-3


def test_anchor_frame_rows_are_exactly_zero():
    """``skip_ends`` is what keeps the two branches an exact partition.

    Frames 0 and F-1 are dense on the softmax side, so the branch must contribute
    exactly nothing there -- not "small", which would double-count them.
    """
    from vllm_omni.diffusion.models.minimax_h3.vdn_branch import VDNLinearBranch
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import window_bounds

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_frames = 17
    branch = VDNLinearBranch(HIDDEN, HEADS, HEAD_DIM, params_dtype=torch.float32).to(device).eval()
    video_x, qkv, _, _ = _inputs(num_frames, device, torch.float32, with_text=False)

    with torch.no_grad():
        out = branch(
            video_x,
            qkv,
            num_frames=num_frames,
            tokens_per_frame=TOKENS_PER_FRAME,
            frame_size=(GRID_H, GRID_W),
            bounds=window_bounds(num_frames),
            skip_ends=True,
        )
    assert torch.count_nonzero(out[:TOKENS_PER_FRAME]) == 0
    assert torch.count_nonzero(out[(num_frames - 1) * TOKENS_PER_FRAME :]) == 0
    assert torch.count_nonzero(out[TOKENS_PER_FRAME : 2 * TOKENS_PER_FRAME]) > 0


def test_unreleased_delta_rules_are_refused():
    """Re-pointing the rule changes nothing observable except the render, so it is not
    something to accept and ignore."""
    from vllm_omni.diffusion.models.minimax_h3.vdn_branch import VDNLinearBranch

    with pytest.raises(ValueError, match="vdn_solve"):
        VDNLinearBranch(HIDDEN, HEADS, HEAD_DIM, delta_rule="sana_scaled")
