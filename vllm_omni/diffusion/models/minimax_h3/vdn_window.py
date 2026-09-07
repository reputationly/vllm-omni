# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""VDN-H3 window geometry: which rows each query row is allowed to attend to.

VDN-H3 (``OpenVDN/vdn-minimax-h3``) replaces H3's dense attention with a hybrid: a
chunk-aligned frame window carries local detail exactly, and a linear-attention branch
summarises everything the window drops. This module owns the *window* half's geometry
and nothing else -- no kernels, no weights, no tensors. Two consumers must agree on it
exactly or the two branches stop being a partition of the sequence:

* ``vllm_omni.diffusion.attention.backends.vdn_window_attn`` runs the softmax side;
* the linear branch covers precisely the frames a query's window excludes.

Deriving it twice would be two chances to disagree, so it is derived here once.

The window is a property of the CHUNK, not of the frame. H3's video VAE codes every 5
latent frames as one unit (``17 * n + 5`` pixel frames -> ``5 * n + 2`` latent frames),
and a frame that sees only part of a neighbouring chunk sees a fragment of something
that was never coded separably. ``chunk=5, radius=1`` therefore gives every frame a
complete previous, current and next chunk -- 15 latent frames. A centred per-frame
window cannot express that: whatever its width, some frame straddles a chunk boundary.

``anchor_frames="both"`` makes latent frames 0 and F-1 dense in both directions (their
queries see everything, and everyone sees them). That is what lets the linear branch
drop them from its input entirely and keeps the softmax/linear split exact.

Values here mirror ``stage-dmd-step-250/model_spec.json`` field for field; they are
checkpoint semantics, not tunables. Changing them retrains the model, it does not
reconfigure it.
"""

from __future__ import annotations

from dataclasses import dataclass

# The mask can make the two anchor frames dense as softmax COLUMNS (every query sees
# them), as ROWS (they see every key), both, or neither. Only "both" makes the
# softmax/linear partition exact, which is why only "both" lets the branch skip them.
ANCHOR_FRAME_MODES = ("none", "columns", "rows", "both")

# VDN's released hybrid configuration.
VDN_CHUNK = 5
VDN_RADIUS = 1
VDN_ANCHOR_FRAMES = "both"

# A row range [start, stop) in the packed sequence.
RowRange = tuple[int, int]


def window_bounds(num_frames: int, radius: int = VDN_RADIUS, chunk: int = VDN_CHUNK) -> list[tuple[int, int]]:
    """Per-frame inclusive window ``[lo, hi]`` over latent frames, unclamped.

    ``chunk <= 0`` is the centred per-frame window ``|t_q - t_k| <= radius``; ``chunk =
    K`` is the chunk-aligned window this checkpoint uses, where frame ``t`` belongs to
    chunk ``t // K`` and sees whole chunks ``[c - radius, c + radius]``. Bounds are
    returned unclamped so callers can tell "the clip ended" from "the window ended";
    every consumer here clamps on use.
    """
    if chunk <= 0:
        return [(t - radius, t + radius) for t in range(num_frames)]
    return [(((t // chunk) - radius) * chunk, ((t // chunk) + radius + 1) * chunk - 1) for t in range(num_frames)]


@dataclass(frozen=True, slots=True)
class VDNWindowGroup:
    """One set of query rows whose kept-key set is identical.

    Because the window is chunk-aligned, every frame in a chunk has the same bounds, so
    a whole chunk's rows form one group and the mask becomes a handful of dense
    rectangles rather than arbitrary sparsity. ``kv_ranges`` are disjoint and ascending,
    so a consumer may gather them with a single concatenation without double-counting a
    key -- which would silently double that key's softmax mass.
    """

    q_range: RowRange
    kv_ranges: tuple[RowRange, ...]

    @property
    def num_q_rows(self) -> int:
        start, stop = self.q_range
        return stop - start

    @property
    def num_kv_rows(self) -> int:
        return sum(stop - start for start, stop in self.kv_ranges)


@dataclass(frozen=True, slots=True)
class VDNWindowPlan:
    """The whole mask, as query groups over dense key rectangles.

    ``dense_q_ranges`` are the rows that attend the *entire* used sequence: the globals
    (text, audio, any reference media), the anchor frames under a mode with dense rows,
    and any chunk whose own window already spans the clip. ``window_groups`` are the
    remaining video rows, one group per chunk. Together they cover every used row
    exactly once -- a row in neither would attend nothing and come back as zeros.
    """

    used_len: int
    video_start: int
    num_frames: int
    tokens_per_frame: int
    dense_q_ranges: tuple[RowRange, ...]
    window_groups: tuple[VDNWindowGroup, ...]

    @property
    def video_end(self) -> int:
        return self.video_start + self.num_frames * self.tokens_per_frame

    @property
    def is_dense(self) -> bool:
        """True when the window covers everything, so this plan is plain attention.

        Short clips reach it (a 3-chunk window over a <=3-chunk clip keeps every pair),
        and the caller should then run its dense kernel rather than this plan: a window
        that covers every frame IS the original attention, and going through the same
        dense kernel keeps that identity exact instead of merely close.
        """
        return not self.window_groups

    def kept_pair_fraction(self) -> float:
        """Fraction of query-key pairs the mask keeps, for logging the actual density."""
        total = self.used_len * self.used_len
        if total <= 0:
            return 0.0
        kept = sum((stop - start) * self.used_len for start, stop in self.dense_q_ranges)
        kept += sum(group.num_q_rows * group.num_kv_rows for group in self.window_groups)
        return kept / total


def _merge(ranges: list[RowRange]) -> tuple[RowRange, ...]:
    """Sort, drop empties and coalesce touching ranges.

    Coalescing is not cosmetic: adjacent ranges left separate would make a consumer
    issue two gathers where one contiguous slice would do, and on the H3 packing the
    globals plus the first window frames are frequently adjacent.
    """
    ordered = sorted((start, stop) for start, stop in ranges if stop > start)
    merged: list[RowRange] = []
    for start, stop in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


def build_window_plan(
    *,
    used_len: int,
    video_start: int,
    num_frames: int,
    tokens_per_frame: int,
    chunk: int = VDN_CHUNK,
    radius: int = VDN_RADIUS,
    anchor_frames: str = VDN_ANCHOR_FRAMES,
) -> VDNWindowPlan:
    """Compile the window mask into dense rectangles.

    ``video_start``/``num_frames``/``tokens_per_frame`` describe the TARGET video only.
    Everything else inside ``used_len`` -- prompt rows, audio rows, and on fl2va/ref2va
    the reference media rows -- is global and stays dense in both directions, exactly as
    it is in the released model. Rows at or past ``used_len`` are packing alignment and
    appear in no group at all.
    """
    if anchor_frames not in ANCHOR_FRAME_MODES:
        raise ValueError(f"anchor_frames={anchor_frames!r}; expected one of {ANCHOR_FRAME_MODES}")
    if tokens_per_frame <= 0 or num_frames <= 0:
        raise ValueError(f"empty video grid: {num_frames} frames x {tokens_per_frame} tokens")

    video_end = video_start + num_frames * tokens_per_frame
    if video_start < 0 or video_end > used_len:
        raise ValueError(
            f"target video rows [{video_start}, {video_end}) do not fit the used sequence "
            f"[0, {used_len}); the window plan cannot be built on a layout it cannot address"
        )

    def frame_rows(frame: int) -> RowRange:
        start = video_start + frame * tokens_per_frame
        return start, start + tokens_per_frame

    globals_ranges = [(0, video_start), (video_end, used_len)]
    anchors = (0, num_frames - 1) if anchor_frames != "none" and num_frames >= 2 else ()
    dense_row_anchors = anchors if anchor_frames in ("rows", "both") else ()
    dense_col_anchors = anchors if anchor_frames in ("columns", "both") else ()

    dense_q = list(globals_ranges) + [frame_rows(frame) for frame in dense_row_anchors]

    bounds = window_bounds(num_frames, radius=radius, chunk=chunk)
    groups: list[VDNWindowGroup] = []
    # Frames sharing a chunk share bounds, so iterate chunks, not frames. With chunk<=0
    # (the per-frame window) every frame is its own group, which this still expresses.
    stride = chunk if chunk > 0 else 1
    for chunk_start in range(0, num_frames, stride):
        chunk_stop = min(chunk_start + stride, num_frames)
        rows = [frame for frame in range(chunk_start, chunk_stop) if frame not in dense_row_anchors]
        if not rows:
            continue
        q_range = (frame_rows(rows[0])[0], frame_rows(rows[-1])[1])
        lo = max(bounds[chunk_start][0], 0)
        hi = min(bounds[chunk_start][1], num_frames - 1)
        if lo <= 0 and hi >= num_frames - 1:
            # This chunk's window already spans the clip, so its queries keep every pair
            # a dense kernel would -- they are dense rows, NOT rows to drop. Near a clip
            # end this happens to some chunks and not others (a 12-frame clip: the middle
            # chunk covers everything, the outer two do not), and dropping them would
            # leave those tokens attending nothing at all.
            dense_q.append(q_range)
            continue
        kv = list(globals_ranges)
        kv.append((frame_rows(lo)[0], frame_rows(hi)[1]))
        # Anchor COLUMNS the window does not already cover. Adding one it does cover
        # would duplicate those keys and double their share of the softmax denominator.
        kv.extend(frame_rows(frame) for frame in dense_col_anchors if not lo <= frame <= hi)
        groups.append(VDNWindowGroup(q_range=q_range, kv_ranges=_merge(kv)))

    return VDNWindowPlan(
        used_len=used_len,
        video_start=video_start,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        dense_q_ranges=_merge(dense_q),
        window_groups=tuple(groups),
    )


__all__ = [
    "ANCHOR_FRAME_MODES",
    "VDN_ANCHOR_FRAMES",
    "VDN_CHUNK",
    "VDN_RADIUS",
    "VDNWindowGroup",
    "VDNWindowPlan",
    "build_window_plan",
    "window_bounds",
]
