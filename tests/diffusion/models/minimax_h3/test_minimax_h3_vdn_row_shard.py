# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``frame_statistics_row_shard`` must sum back to the whole-sequence statistics.

Sequence parallelism splits the packed rows evenly, so a rank routinely holds part of a
frame. The branch's per-frame statistics are sums along the token axis, which makes them
reconstructible from partial sums -- but only if a rank contributes exactly its own rows
and exactly zero elsewhere. That is the property these tests pin: an off-by-one in the
frame indexing, or padding that leaks a nonzero row, shows up as a silently wrong delta
rule rather than as an error.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.vdn_branch import (
    frame_statistics,
    frame_statistics_row_shard,
    frame_sum_row_shard,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

FRAMES = 7
TOKENS_PER_FRAME = 6
HEADS = 3
HEAD_DIM = 4


def _inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    rows = FRAMES * TOKENS_PER_FRAME
    key = torch.randn(rows, HEADS, HEAD_DIM, generator=generator, dtype=torch.float32)
    value = torch.randn(rows, HEADS, HEAD_DIM, generator=generator, dtype=torch.float32)
    beta = torch.rand(rows, HEADS, generator=generator, dtype=torch.float32)
    return key, value, beta


def _whole_sequence(key, value, beta):
    shape = (FRAMES, TOKENS_PER_FRAME, HEADS, HEAD_DIM)
    key_f = key.view(shape).permute(0, 2, 1, 3)
    value_f = value.view(shape).permute(0, 2, 1, 3)
    beta_f = beta.view(FRAMES, TOKENS_PER_FRAME, HEADS).permute(0, 2, 1)
    return frame_statistics(key_f, value_f, beta_f)


def _shard_bounds(rows: int, world: int) -> list[tuple[int, int]]:
    """The even split sequence parallelism actually performs, frame-unaligned on purpose."""
    chunk = rows // world
    return [(rank * chunk, chunk) for rank in range(world)]


@pytest.mark.parametrize("world", [1, 2, 3, 6])
def test_row_shards_sum_to_whole_sequence(world: int) -> None:
    key, value, beta = _inputs()
    want_a, want_b = _whole_sequence(key, value, beta)

    got_a = torch.zeros_like(want_a)
    got_b = torch.zeros_like(want_b)
    for start, count in _shard_bounds(FRAMES * TOKENS_PER_FRAME, world):
        part_a, part_b = frame_statistics_row_shard(
            key[start : start + count],
            value[start : start + count],
            beta[start : start + count],
            row_start=start,
            num_frames=FRAMES,
            tokens_per_frame=TOKENS_PER_FRAME,
        )
        got_a += part_a
        got_b += part_b

    torch.testing.assert_close(got_a, want_a, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_b, want_b, rtol=1e-5, atol=1e-5)


def test_untouched_frames_are_exactly_zero() -> None:
    """A rank must contribute nothing outside its rows, or the sum double counts."""
    key, value, beta = _inputs()
    start, count = TOKENS_PER_FRAME * 2, TOKENS_PER_FRAME
    part_a, part_b = frame_statistics_row_shard(
        key[start : start + count],
        value[start : start + count],
        beta[start : start + count],
        row_start=start,
        num_frames=FRAMES,
        tokens_per_frame=TOKENS_PER_FRAME,
    )
    touched = torch.zeros(FRAMES, dtype=torch.bool)
    touched[2] = True
    assert bool((part_a[~touched] == 0).all())
    assert bool((part_b[~touched] == 0).all())
    assert bool((part_a[touched] != 0).any())


def test_partial_frame_span_is_split_at_the_right_row() -> None:
    """A span starting and ending mid-frame still lands on the frames it covers."""
    key, value, beta = _inputs()
    start = TOKENS_PER_FRAME + TOKENS_PER_FRAME // 2  # halfway through frame 1
    count = TOKENS_PER_FRAME * 2  # ends halfway through frame 3
    part_a, _ = frame_statistics_row_shard(
        key[start : start + count],
        value[start : start + count],
        beta[start : start + count],
        row_start=start,
        num_frames=FRAMES,
        tokens_per_frame=TOKENS_PER_FRAME,
    )
    nonzero = [index for index in range(FRAMES) if bool((part_a[index] != 0).any())]
    assert nonzero == [1, 2, 3]


def test_empty_shard_contributes_zero() -> None:
    """More ranks than rows is legal; the empty ranks must not poison the sum."""
    key, value, beta = _inputs()
    part_a, part_b = frame_statistics_row_shard(
        key[:0],
        value[:0],
        beta[:0],
        row_start=0,
        num_frames=FRAMES,
        tokens_per_frame=TOKENS_PER_FRAME,
    )
    assert part_a.shape[0] == FRAMES
    assert bool((part_a == 0).all())
    assert bool((part_b == 0).all())


def test_span_past_the_last_frame_is_refused() -> None:
    key, value, beta = _inputs()
    with pytest.raises(ValueError, match="exceed"):
        frame_statistics_row_shard(
            key[:TOKENS_PER_FRAME],
            value[:TOKENS_PER_FRAME],
            beta[:TOKENS_PER_FRAME],
            row_start=FRAMES * TOKENS_PER_FRAME,
            num_frames=FRAMES,
            tokens_per_frame=TOKENS_PER_FRAME,
        )


@pytest.mark.parametrize("world", [1, 2, 3, 6])
def test_frame_sums_reconstruct_the_per_frame_mean(world: int) -> None:
    """alpha's input: shards contribute sums, the caller divides once after reducing."""
    generator = torch.Generator().manual_seed(1)
    channels = 5
    rows = FRAMES * TOKENS_PER_FRAME
    tokens = torch.randn(rows, channels, generator=generator, dtype=torch.float32)
    want = tokens.view(FRAMES, TOKENS_PER_FRAME, channels).mean(dim=1, dtype=torch.float32)

    total = torch.zeros(FRAMES, channels, dtype=torch.float32)
    for start, count in _shard_bounds(rows, world):
        total += frame_sum_row_shard(
            tokens[start : start + count],
            row_start=start,
            num_frames=FRAMES,
            tokens_per_frame=TOKENS_PER_FRAME,
        )
    torch.testing.assert_close(total / TOKENS_PER_FRAME, want, rtol=1e-5, atol=1e-5)


def test_frame_sum_leaves_untouched_frames_zero() -> None:
    generator = torch.Generator().manual_seed(2)
    tokens = torch.randn(TOKENS_PER_FRAME, 5, generator=generator, dtype=torch.float32)
    partial = frame_sum_row_shard(
        tokens, row_start=TOKENS_PER_FRAME * 3, num_frames=FRAMES, tokens_per_frame=TOKENS_PER_FRAME
    )
    touched = torch.zeros(FRAMES, dtype=torch.bool)
    touched[3] = True
    assert bool((partial[~touched] == 0).all())
    assert bool((partial[touched] != 0).any())
