# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``exchange_halo_rows`` must move exactly the rows the plan asks for.

The planner is covered by ``test_minimax_h3_vdn_halo_plan``; this covers the transfer
itself, which the planner cannot: an off-by-one in the destination offset, a send that
aliases a buffer the loop later overwrites, or a deadlock from posting sends and receives
separately. Those only appear with real ranks moving real bytes.

Rows are tagged with their own global index, so a misplaced row is identified rather than
merely detected. Gloo on CPU keeps this runnable without GPUs.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.models.minimax_h3.vdn_branch import (
    exchange_halo_rows,
    halo_frame_span,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TOKENS_PER_FRAME = 6
FRAMES = 12
TOTAL_ROWS = FRAMES * TOKENS_PER_FRAME
CHANNELS = 2
HEAD_DIM = 3


def _tagged_rows() -> torch.Tensor:
    """Row ``i`` is filled with ``i``, so a misplaced row names itself."""
    tags = torch.arange(TOTAL_ROWS, dtype=torch.float32)
    return tags.view(TOTAL_ROWS, 1, 1).expand(TOTAL_ROWS, CHANNELS, HEAD_DIM).contiguous()


def _shards(world: int, skew: int) -> list[tuple[int, int]]:
    """``skew`` rows taken off rank 0, as a packed prefix does in the real layout."""
    chunk = TOTAL_ROWS // world
    counts = [chunk - skew] + [chunk] * (world - 2)
    counts.append(TOTAL_ROWS - sum(counts))
    shards, start = [], 0
    for count in counts:
        shards.append((start, count))
        start += count
    return shards


def _worker(rank: int, world: int, skew: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        everything = _tagged_rows()
        shards = _shards(world, skew)
        row_start, num_rows = shards[rank]
        local = everything[row_start : row_start + num_rows].clone()

        haloed = exchange_halo_rows(
            [local, local * -1.0],
            shards=shards,
            rank=rank,
            num_frames=FRAMES,
            tokens_per_frame=TOKENS_PER_FRAME,
        )

        first, last = halo_frame_span(row_start, num_rows, num_frames=FRAMES, tokens_per_frame=TOKENS_PER_FRAME)
        span = slice(first * TOKENS_PER_FRAME, (last + 1) * TOKENS_PER_FRAME)
        torch.testing.assert_close(haloed[0], everything[span])
        # The second tensor is exchanged through the same plan; a plan applied to only the
        # first would pass the check above and still corrupt the branch's value stream.
        torch.testing.assert_close(haloed[1], everything[span] * -1.0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("skew", [0, 4])
def test_halo_exchange_places_every_row(world: int, skew: int) -> None:
    port = 29700 + world * 10 + skew
    mp.spawn(_worker, args=(world, skew, port), nprocs=world, join=True)


def _subgroup_worker(rank: int, world: int, port: int) -> None:
    """Four ranks, two sequence-parallel groups of two: group rank 0 is global rank 2."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        # Both groups must be constructed on every rank, in the same order.
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        group = groups[rank // 2]
        group_rank, group_world = rank % 2, 2

        everything = _tagged_rows()
        shards = _shards(group_world, skew=0)
        row_start, num_rows = shards[group_rank]

        haloed = exchange_halo_rows(
            [everything[row_start : row_start + num_rows].clone()],
            shards=shards,
            rank=group_rank,
            num_frames=FRAMES,
            tokens_per_frame=TOKENS_PER_FRAME,
            group=group,
        )
        first, last = halo_frame_span(row_start, num_rows, num_frames=FRAMES, tokens_per_frame=TOKENS_PER_FRAME)
        span = slice(first * TOKENS_PER_FRAME, (last + 1) * TOKENS_PER_FRAME)
        torch.testing.assert_close(haloed[0], everything[span])
    finally:
        dist.destroy_process_group()


def test_peers_are_resolved_within_the_group_not_globally() -> None:
    """``shards`` is indexed by rank inside the sequence-parallel group.

    Whenever sequence parallelism coexists with tensor parallelism the group is a subset
    of the world, so a group rank is not a global rank. Passing one where torch expects
    the other addresses a rank in the wrong group -- which hangs if nobody is listening
    and silently transfers the wrong frames if somebody is.
    """
    mp.spawn(_subgroup_worker, args=(4, 29760), nprocs=4, join=True)


def test_single_rank_needs_no_transfer() -> None:
    """World size one must not touch the process group, so it works uninitialised."""
    everything = _tagged_rows()
    shards = [(0, TOTAL_ROWS)]
    haloed = exchange_halo_rows(
        [everything.clone()],
        shards=shards,
        rank=0,
        num_frames=FRAMES,
        tokens_per_frame=TOKENS_PER_FRAME,
    )
    torch.testing.assert_close(haloed[0], everything)
