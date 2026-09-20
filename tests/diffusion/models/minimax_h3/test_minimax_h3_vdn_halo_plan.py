# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The halo plan must be symmetric and must exactly fill each shard's span.

Sequence parallelism splits the whole packed sequence, so ranks own unequal numbers of
VIDEO rows and the halo cannot be a symmetric "send my first k rows left" rule. These
tests pin the two properties a wrong plan breaks silently:

* **symmetry** -- every range one rank expects to receive is a range its peer plans to
  send. A mismatch deadlocks or, worse, transfers the wrong rows.
* **completeness** -- received ranges plus owned rows tile the haloed span exactly once.
  A gap leaves the convolution reading uninitialised memory; an overlap double counts.
"""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.models.minimax_h3.vdn_branch import (
    halo_frame_span,
    halo_row_counts,
    plan_halo_exchange,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TOKENS_PER_FRAME = 6
FRAMES = 12
TOTAL_ROWS = FRAMES * TOKENS_PER_FRAME


def _even(world: int) -> list[tuple[int, int]]:
    chunk = TOTAL_ROWS // world
    return [(rank * chunk, chunk) for rank in range(world)]


def _skewed(world: int) -> list[tuple[int, int]]:
    """A prefix on rank 0 leaves it fewer video rows -- the realistic packed layout."""
    counts = [TOTAL_ROWS // world - 4] + [TOTAL_ROWS // world] * (world - 2)
    counts.append(TOTAL_ROWS - sum(counts))
    shards, start = [], 0
    for count in counts:
        shards.append((start, count))
        start += count
    return shards


def _plans(shards):
    return [
        plan_halo_exchange(shards, rank, num_frames=FRAMES, tokens_per_frame=TOKENS_PER_FRAME)
        for rank in range(len(shards))
    ]


@pytest.mark.parametrize("layout", [_even, _skewed])
@pytest.mark.parametrize("world", [2, 3, 4])
def test_every_receive_is_matched_by_a_send(layout, world: int) -> None:
    shards = layout(world)
    plans = _plans(shards)
    for rank, (recv, _) in enumerate(plans):
        for transfer in recv:
            peer_sends = plans[transfer.peer][1]
            assert any(
                sent.peer == rank and sent.start == transfer.start and sent.count == transfer.count
                for sent in peer_sends
            ), f"rank {rank} expects {transfer} but rank {transfer.peer} does not send it"


@pytest.mark.parametrize("layout", [_even, _skewed])
@pytest.mark.parametrize("world", [2, 3, 4])
def test_every_send_is_wanted_by_its_peer(layout, world: int) -> None:
    shards = layout(world)
    plans = _plans(shards)
    for rank, (_, send) in enumerate(plans):
        for transfer in send:
            peer_recv = plans[transfer.peer][0]
            assert any(
                got.peer == rank and got.start == transfer.start and got.count == transfer.count for got in peer_recv
            ), f"rank {rank} sends {transfer} that rank {transfer.peer} never asked for"


@pytest.mark.parametrize("layout", [_even, _skewed])
@pytest.mark.parametrize("world", [2, 3, 4])
def test_received_plus_owned_tiles_the_haloed_span(layout, world: int) -> None:
    shards = layout(world)
    for rank, (start, count) in enumerate(shards):
        recv, _ = plan_halo_exchange(shards, rank, num_frames=FRAMES, tokens_per_frame=TOKENS_PER_FRAME)
        first, last = halo_frame_span(start, count, num_frames=FRAMES, tokens_per_frame=TOKENS_PER_FRAME)
        covered = [False] * TOTAL_ROWS
        for row in range(start, start + count):
            covered[row] = True
        for transfer in recv:
            for row in range(transfer.start, transfer.start + transfer.count):
                assert not covered[row], f"row {row} covered twice on rank {rank}"
                covered[row] = True
        span = range(first * TOKENS_PER_FRAME, (last + 1) * TOKENS_PER_FRAME)
        assert all(covered[row] for row in span), f"rank {rank} has a gap in its haloed span"


def test_halo_row_counts_agree_with_the_span() -> None:
    start, count = TOKENS_PER_FRAME * 4 + 2, TOKENS_PER_FRAME * 3
    first, last = halo_frame_span(start, count, num_frames=FRAMES, tokens_per_frame=TOKENS_PER_FRAME)
    before, after = halo_row_counts(start, count, num_frames=FRAMES, tokens_per_frame=TOKENS_PER_FRAME)
    assert before == start - first * TOKENS_PER_FRAME
    assert after == (last + 1) * TOKENS_PER_FRAME - (start + count)


def test_clip_ends_need_no_halo_beyond_them() -> None:
    before, _ = halo_row_counts(0, TOKENS_PER_FRAME, num_frames=FRAMES, tokens_per_frame=TOKENS_PER_FRAME)
    assert before == 0
    _, after = halo_row_counts(
        TOTAL_ROWS - TOKENS_PER_FRAME, TOKENS_PER_FRAME, num_frames=FRAMES, tokens_per_frame=TOKENS_PER_FRAME
    )
    assert after == 0


def test_a_shard_must_own_rows() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        halo_frame_span(0, 0, num_frames=FRAMES, tokens_per_frame=TOKENS_PER_FRAME)
