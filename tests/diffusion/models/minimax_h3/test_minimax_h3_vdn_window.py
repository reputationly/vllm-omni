# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The VDN window plan must be the same mask the reference semantics describe.

``build_window_plan`` compiles the mask into dense rectangles for speed. These tests
rebuild the mask row by row from the definition instead -- globals dense both ways,
video queries restricted to their chunk-aligned window, anchor frames dense -- and
require the two to agree exactly. Anything the compilation drops or double-counts shows
up as a differing boolean, and a duplicated key would silently double its share of a
softmax denominator.
"""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _reference_mask(
    *,
    used_len: int,
    video_start: int,
    num_frames: int,
    tokens_per_frame: int,
    chunk: int,
    radius: int,
    anchor_frames: str,
) -> list[list[bool]]:
    """The mask straight from the definition, one row at a time."""
    video_end = video_start + num_frames * tokens_per_frame

    def frame_of(row: int) -> int | None:
        if not video_start <= row < video_end:
            return None
        return (row - video_start) // tokens_per_frame

    anchors = (0, num_frames - 1) if anchor_frames != "none" and num_frames >= 2 else ()
    row_anchors = anchors if anchor_frames in ("rows", "both") else ()
    col_anchors = anchors if anchor_frames in ("columns", "both") else ()

    mask = [[False] * used_len for _ in range(used_len)]
    for q_row in range(used_len):
        q_frame = frame_of(q_row)
        if q_frame is None or q_frame in row_anchors:
            mask[q_row] = [True] * used_len
            continue
        if chunk > 0:
            lo = ((q_frame // chunk) - radius) * chunk
            hi = ((q_frame // chunk) + radius + 1) * chunk - 1
        else:
            lo, hi = q_frame - radius, q_frame + radius
        lo, hi = max(lo, 0), min(hi, num_frames - 1)
        for k_row in range(used_len):
            k_frame = frame_of(k_row)
            if k_frame is None:
                mask[q_row][k_row] = True  # globals are dense columns
            elif lo <= k_frame <= hi or k_frame in col_anchors:
                mask[q_row][k_row] = True
    return mask


def _plan_mask(plan) -> list[list[bool]]:
    mask = [[False] * plan.used_len for _ in range(plan.used_len)]
    for start, stop in plan.dense_q_ranges:
        for row in range(start, stop):
            mask[row] = [True] * plan.used_len
    for group in plan.window_groups:
        q_start, q_stop = group.q_range
        for row in range(q_start, q_stop):
            for kv_start, kv_stop in group.kv_ranges:
                for col in range(kv_start, kv_stop):
                    assert not mask[row][col], "a key appears in two ranges of one group"
                    mask[row][col] = True
    return mask


# A 3-chunk window over chunk=5 spans 15 frames, so the three regimes are: shorter than
# the window (every chunk dense), mixed (the middle chunk covers the clip, the outer two
# do not -- the case that dropped rows on the floor), and longer (every chunk windowed).
@pytest.mark.parametrize("num_frames", [7, 12, 17])
@pytest.mark.parametrize("anchor_frames", ["both", "columns", "rows", "none"])
def test_plan_reproduces_the_reference_mask(num_frames, anchor_frames):
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import build_window_plan

    prefix, tokens_per_frame, suffix = 3, 2, 2
    used_len = prefix + num_frames * tokens_per_frame + suffix
    geometry = dict(
        used_len=used_len,
        video_start=prefix,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        chunk=5,
        radius=1,
    )
    plan = build_window_plan(anchor_frames=anchor_frames, **geometry)
    mask = _reference_mask(anchor_frames=anchor_frames, **geometry)
    assert _plan_mask(plan) == mask
    # `is_dense` is what routes a forward back to the dense kernel, so it has to mean
    # exactly "this mask keeps everything" -- not merely "no group happened to be built".
    assert plan.is_dense == all(all(row) for row in mask)


def test_every_used_row_is_a_query_exactly_once():
    """A row missing from every group would silently produce zeros for that token."""
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import build_window_plan

    plan = build_window_plan(used_len=100, video_start=6, num_frames=22, tokens_per_frame=4, chunk=5, radius=1)
    seen: list[int] = []
    for start, stop in plan.dense_q_ranges:
        seen.extend(range(start, stop))
    for group in plan.window_groups:
        seen.extend(range(*group.q_range))
    assert sorted(seen) == list(range(plan.used_len))
    assert len(seen) == len(set(seen))


def test_clip_shorter_than_the_window_collapses_to_dense():
    """A 15-frame window over a 10-frame clip keeps every pair, so it IS dense attention.

    The caller must then run its dense kernel: matching the kernel is what keeps the
    full-cover case exactly equal to the released model rather than merely close.
    """
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import build_window_plan

    plan = build_window_plan(used_len=55, video_start=5, num_frames=10, tokens_per_frame=5, chunk=5, radius=1)
    assert plan.is_dense
    assert plan.window_groups == ()
    assert plan.kept_pair_fraction() == pytest.approx(1.0)


def test_a_chunk_covering_the_clip_becomes_a_dense_row_not_a_dropped_one():
    """13 frames: the middle chunk spans the clip, the outer two do not.

    Its rows must land in ``dense_q_ranges``. Dropping them instead leaves those tokens
    attending nothing -- zeros out of the attention for a fifth of the video, which no
    shape check would catch.
    """
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import build_window_plan

    plan = build_window_plan(used_len=70, video_start=5, num_frames=13, tokens_per_frame=5, chunk=5, radius=1)
    assert not plan.is_dense
    middle_chunk = (5 + 5 * 5, 5 + 10 * 5)  # frames 5..9
    assert middle_chunk in plan.dense_q_ranges


def test_production_density_matches_the_measured_window():
    """15 s at 768p: F=107, 1008 tokens/frame, a 15-frame window.

    The kept fraction is what the whole port is being bought for, so it is asserted
    rather than left to a benchmark: a regression that widens the window would keep
    every test above green while quietly giving back the speedup.
    """
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import build_window_plan

    tokens_per_frame, num_frames, prefix = 1008, 107, 1716
    plan = build_window_plan(
        used_len=prefix + num_frames * tokens_per_frame,
        video_start=prefix,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
    )
    assert not plan.is_dense
    assert len(plan.window_groups) == 22  # ceil(107 / 5)
    assert plan.kept_pair_fraction() == pytest.approx(0.20, abs=0.02)


def test_rejects_a_video_span_outside_the_used_sequence():
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import build_window_plan

    with pytest.raises(ValueError, match="do not fit the used sequence"):
        build_window_plan(used_len=50, video_start=10, num_frames=20, tokens_per_frame=4)


def test_rejects_an_unknown_anchor_mode():
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import build_window_plan

    with pytest.raises(ValueError, match="anchor_frames"):
        build_window_plan(used_len=50, video_start=2, num_frames=8, tokens_per_frame=4, anchor_frames="ends")
