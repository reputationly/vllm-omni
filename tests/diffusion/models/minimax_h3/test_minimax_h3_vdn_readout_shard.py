# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``readout_row_shard`` concatenated across ranks must equal the whole-sequence readout.

This is the equivalence the sequence-parallel VDN branch rests on. Splitting the packed
rows evenly leaves ranks holding partial frames, and three separate things have to line up
for the result to be the same clip: the haloed features must reproduce the convolutions'
neighbourhood, the per-frame statistics must reduce to the whole-sequence ones, and each
shard must read out against the frames its own rows belong to.

A mistake in any of them renders a plausible video rather than raising, which is why this
compares the actual readout rather than the intermediates.

The all-reduce is simulated: the totals are computed up front and handed back in call
order, which is exactly what a real collective would produce and needs no process group.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.vdn_branch import VDNLinearBranch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HIDDEN = 32
TOTAL_HEADS = 2
HEAD_DIM = 8
FRAMES = 9
FRAME_H, FRAME_W = 2, 3
TOKENS_PER_FRAME = FRAME_H * FRAME_W
HALO = 2  # SHORT_CONV_KERNEL // 2


def _branch(enable_text_state: bool = False) -> VDNLinearBranch:
    torch.manual_seed(0)
    branch = VDNLinearBranch(
        HIDDEN,
        TOTAL_HEADS,
        HEAD_DIM,
        enable_text_state=enable_text_state,
        params_dtype=torch.float32,
        head_shard=(0, TOTAL_HEADS),
    )
    # Zero-initialised projections would make every shard agree trivially.
    for parameter in branch.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    return branch.eval()


def _inputs():
    generator = torch.Generator().manual_seed(1)
    rows = FRAMES * TOKENS_PER_FRAME
    video_x = torch.randn(rows, HIDDEN, generator=generator, dtype=torch.float32)
    qkv = tuple(torch.randn(rows, TOTAL_HEADS, HEAD_DIM, generator=generator, dtype=torch.float32) for _ in range(3))
    return video_x, qkv


def _bounds() -> list[tuple[int, int]]:
    """Chunk-aligned windows, the shape ``window_bounds`` produces for chunk=5, radius=1."""
    chunk, radius = 5, 1
    return [
        (((frame // chunk) - radius) * chunk, ((frame // chunk) + radius + 1) * chunk - 1) for frame in range(FRAMES)
    ]


def _even_shards(rows: int, world: int) -> list[tuple[int, int]]:
    chunk = rows // world
    return [(rank * chunk, chunk) for rank in range(world)]


def _text_shards(length: int, world: int) -> list[tuple[int, int]]:
    """The prompt is packed into the same sequence, so it is split across ranks too."""
    base, extra = divmod(length, world)
    shards, start = [], 0
    for rank in range(world):
        count = base + (1 if rank < extra else 0)
        shards.append((start, count))
        start += count
    return shards


def _capture(sink: list[torch.Tensor]):
    """An identity reduce that records what the code asked to reduce.

    Recording the real partials rather than recomputing them here is the point: a
    reimplementation would only ever test itself.
    """

    def reduce(tensor: torch.Tensor) -> torch.Tensor:
        sink.append(tensor)
        return tensor

    return reduce


def _text_piece(text, span: tuple[int, int]):
    """This rank's slice of the prompt, or ``None`` when the request carries no prompt."""
    if text is None:
        return None
    start, count = span
    return text[0][start : start + count], tuple(t[start : start + count] for t in text[1])


def _call_shard(branch, video_x, qkv, bounds, start, count, text, all_reduce):
    halo_first = max(0, start // TOKENS_PER_FRAME - HALO)
    halo_last = min(FRAMES - 1, (start + count - 1) // TOKENS_PER_FRAME + HALO)
    span = slice(halo_first * TOKENS_PER_FRAME, (halo_last + 1) * TOKENS_PER_FRAME)
    text_x, text_qkv = text if text is not None else (None, None)
    return branch.readout_row_shard(
        video_x[start : start + count],
        tuple(t[span] for t in qkv),
        halo_first_frame=halo_first,
        row_start=start,
        num_rows=count,
        num_frames=FRAMES,
        tokens_per_frame=TOKENS_PER_FRAME,
        frame_size=(FRAME_H, FRAME_W),
        bounds=bounds,
        text_x=text_x,
        text_qkv_raw=text_qkv,
        all_reduce=all_reduce,
    )


def _sharded_readout(branch, video_x, qkv, bounds, world: int, text=None) -> torch.Tensor:
    """Run every shard twice: once to accumulate the collectives, once to read out.

    The all-reduce is simulated by summing each rank's partials up front and replaying the
    totals in call order -- which is exactly what a real collective would produce, and needs
    no process group.
    """
    shards = _even_shards(FRAMES * TOKENS_PER_FRAME, world)
    # A placeholder span per rank when there is no prompt: _text_piece ignores it, and this
    # keeps the two loops a plain zip over the same pair of lists.
    text_shards: list[tuple[int, int]] = (
        _text_shards(int(text[0].shape[0]), world) if text is not None else [(0, 0)] * world
    )

    totals: list[torch.Tensor] = []
    for (start, count), text_span in zip(shards, text_shards, strict=True):
        captured: list[torch.Tensor] = []
        _call_shard(branch, video_x, qkv, bounds, start, count, _text_piece(text, text_span), _capture(captured))
        totals = captured if not totals else [a + b for a, b in zip(totals, captured, strict=True)]

    pieces = []
    for (start, count), text_span in zip(shards, text_shards, strict=True):
        replay = iter(totals)
        pieces.append(
            _call_shard(
                branch,
                video_x,
                qkv,
                bounds,
                start,
                count,
                _text_piece(text, text_span),
                lambda _tensor, replay=replay: next(replay),
            )
        )
    return torch.cat(pieces, dim=0)


@pytest.mark.parametrize("world", [1, 2, 3])
def test_shards_reconstruct_the_whole_sequence_readout(world: int) -> None:
    branch = _branch()
    video_x, qkv = _inputs()
    bounds = _bounds()

    with torch.no_grad():
        want = branch._readout(video_x, qkv, FRAMES, TOKENS_PER_FRAME, (FRAME_H, FRAME_W), bounds, None, None)
        got = _sharded_readout(branch, video_x, qkv, bounds, world)

    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("world", [1, 2, 3])
def test_a_split_prompt_reconstructs_the_text_state(world: int) -> None:
    """The prompt is packed into the same sequence, so sequence parallelism splits it too.

    Its state is one delta-rule chunk over all prompt rows, which no rank holds in full.
    The statistics are row sums, so partials reduce -- but only if the symmetrisation and
    the Cholesky happen AFTER the reduction. Doing either per rank yields a state that is
    close enough to render and wrong.
    """
    branch = _branch(enable_text_state=True)
    video_x, qkv = _inputs()
    bounds = _bounds()

    generator = torch.Generator().manual_seed(7)
    text_len = 11  # deliberately not divisible by any world size under test
    text_x = torch.randn(text_len, HIDDEN, generator=generator, dtype=torch.float32)
    text_qkv = tuple(
        torch.randn(text_len, TOTAL_HEADS, HEAD_DIM, generator=generator, dtype=torch.float32) for _ in range(3)
    )

    with torch.no_grad():
        want = branch._readout(video_x, qkv, FRAMES, TOKENS_PER_FRAME, (FRAME_H, FRAME_W), bounds, text_x, text_qkv)
        got = _sharded_readout(branch, video_x, qkv, bounds, world, text=(text_x, text_qkv))

    torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-4)


def test_a_rank_owning_no_prompt_rows_contributes_zero() -> None:
    """Empty is not None: the text slot is part of a collective every rank must join."""
    branch = _branch(enable_text_state=True)
    empty_x = torch.zeros(0, HIDDEN)
    empty_qkv = tuple(torch.zeros(0, TOTAL_HEADS, HEAD_DIM) for _ in range(3))
    with torch.no_grad():
        a_stat, b_stat = branch._text_statistics(empty_x, empty_qkv)
    assert a_stat.shape == (1, TOTAL_HEADS, HEAD_DIM, HEAD_DIM)
    assert bool((a_stat == 0).all())
    assert bool((b_stat == 0).all())


def test_haloed_span_must_be_whole_frames() -> None:
    branch = _branch()
    video_x, qkv = _inputs()
    with pytest.raises(ValueError, match="whole frames"):
        branch.readout_row_shard(
            video_x[:TOKENS_PER_FRAME],
            tuple(t[:-1] for t in qkv),
            halo_first_frame=0,
            row_start=0,
            num_rows=TOKENS_PER_FRAME,
            num_frames=FRAMES,
            tokens_per_frame=TOKENS_PER_FRAME,
            frame_size=(FRAME_H, FRAME_W),
            bounds=_bounds(),
            text_x=None,
            text_qkv_raw=None,
        )


def test_owned_rows_outside_the_halo_are_refused() -> None:
    branch = _branch()
    video_x, qkv = _inputs()
    span = slice(0, TOKENS_PER_FRAME * 2)
    with pytest.raises(ValueError, match="not inside the haloed span"):
        branch.readout_row_shard(
            video_x[:TOKENS_PER_FRAME],
            tuple(t[span] for t in qkv),
            halo_first_frame=0,
            row_start=TOKENS_PER_FRAME * 5,
            num_rows=TOKENS_PER_FRAME,
            num_frames=FRAMES,
            tokens_per_frame=TOKENS_PER_FRAME,
            frame_size=(FRAME_H, FRAME_W),
            bounds=_bounds(),
            text_x=None,
            text_qkv_raw=None,
        )
