# SPDX-License-Identifier: Apache-2.0
"""Normalizing a `ref2va` reference video onto MiniMax-H3's own 24 fps grid.

Three steps, in this order, and the order is the contract:

1. **resample the frame rate**, by dropping and duplicating whole frames — no
   blending, which is what ``ffmpeg``'s ``fps`` filter does and what the
   released model was conditioned on;
2. **truncate to the generated frame count**;
3. **put the frames on the canvas the reference's own aspect ratio resolves to**
   — the same 768-short-edge rule the target follows, unlike an image
   reference, which gets a short edge of its own.

vLLM-Omni's legacy path instead re-encodes the reference to ``libx264 +
yuv420p`` and reads the result back, so the frames the VAE sees have been
through chroma subsampling and a lossy quantiser. It also never truncates to the
generated frame count, which changes the number of packed video rows — a
discrete contract value, not a numeric one — and it feeds the conditioner by
spawning one ``ffmpeg`` process per sampled frame, each seeking from the start
of the file.

The functions here are pure: frames in, frames out. Decoding belongs to the
caller, which is what keeps the resampling arithmetic testable without a codec.
"""

from __future__ import annotations

import math

import numpy as np

MINIMAX_H3_FPS = 24.0

# The video VAE's chunking, mirrored from the checkpoint config it is read off
# at runtime (``clip_length`` / ``tokens_chunk_size``). Named here so the frame
# arithmetic stays a pure function that a test can call without a checkpoint.
MINIMAX_H3_VAE_FRAMES_PER_CHUNK = 17
MINIMAX_H3_VAE_LATENTS_PER_CHUNK = 5


def frame_slot_repeats(num_frames: int, fps: float, target_fps: float = MINIMAX_H3_FPS) -> np.ndarray:
    """How many target slots each source frame occupies.

    Every frame is held until the slot of the next one, and the last one until
    the slot the stream's end rounds to. Reproduced with ``floor(x + 0.5)``
    rather than ``round`` because Python and numpy round halves to even, while
    the reference implementation rounds halves up — they disagree on exactly the
    frame indices a 25 or 30 fps source lands on.

    Args:
        num_frames: Number of source frames.
        fps: The rate the source carries.
        target_fps: MiniMax-H3's own rate.

    Returns:
        ``(num_frames,)`` int64 repeat counts, summing to the resampled length.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if fps <= 0:
        raise ValueError(f"a reference video must have a positive frame rate, got {fps}")

    scale = target_fps / fps
    slots = np.floor(np.arange(num_frames) * scale + 0.5).astype(np.int64)
    return np.diff(slots, append=math.floor(num_frames * scale + 0.5))


def resample_frame_indices(num_frames: int, fps: float, target_fps: float = MINIMAX_H3_FPS) -> np.ndarray:
    """The source frame index that lands in each target slot.

    The index view of :func:`frame_slot_repeats`, which is what a parity test
    compares — it says *which frame* is conditioned on at each 24 fps slot,
    independently of the pixels.
    """
    return np.repeat(np.arange(num_frames), frame_slot_repeats(num_frames, fps, target_fps))


def normalize_reference_video_frames(
    frames: np.ndarray,
    *,
    fps: float,
    num_frames: int | None,
    canvas_multiple: int,
    canvas_short_edge: int,
    canvas_max_pixels: int,
    resolve_canvas,
    target_fps: float = MINIMAX_H3_FPS,
) -> np.ndarray:
    """Resample, truncate and rescale a reference video, official order.

    Args:
        frames: ``(num_source_frames, height, width, 3)`` uint8 RGB.
        fps: The rate ``frames`` carries.
        num_frames: The generated frame count the reference is truncated to.
            ``None`` keeps every frame, which is the legacy behaviour: the
            reference then contributes more packed rows than the official
            contract gives it, so this is an attribution knob and not a
            production setting.
        canvas_multiple, canvas_short_edge, canvas_max_pixels: The canvas rule.
        resolve_canvas: ``(aspect_w, aspect_h, multiple, short_edge, max_pixels)
            -> (height, width)``; injected so this module stays independent of
            where that helper lives.
        target_fps: MiniMax-H3's own rate.

    Returns:
        ``(num_frames_out, height, width, 3)`` uint8 RGB.
    """
    from PIL import Image

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"a reference video must be (num_frames, height, width, 3) RGB, got {tuple(frames.shape)}")
    if frames.dtype != np.uint8:
        raise ValueError(f"a reference video must be uint8 after decoding, got {frames.dtype}")

    if fps != target_fps:
        frames = np.repeat(frames, frame_slot_repeats(frames.shape[0], fps, target_fps), axis=0)

    frames = frames[:num_frames]
    height, width = resolve_canvas(
        frames.shape[2], frames.shape[1], canvas_multiple, canvas_short_edge, canvas_max_pixels
    )
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
    )


def normalize_reference_video_stream(
    timed_frames,
    *,
    fps: float,
    num_frames: int | None,
    canvas_multiple: int,
    canvas_short_edge: int,
    canvas_max_pixels: int,
    resolve_canvas,
    target_fps: float = MINIMAX_H3_FPS,
) -> np.ndarray:
    """:func:`normalize_reference_video_frames`, one source frame at a time.

    Same three steps in the same order and the same bytes out — the difference
    is that nothing ever holds the source stream. Three properties make that
    exact rather than approximate:

    * The repeat schedule is prefix-computable. A source frame owns the output
      slots from where it appears to where its successor does: ``floor(t_next *
      target_fps + 0.5) - floor(t * target_fps + 0.5)``, and the last frame,
      having no successor, is ended by the one nominal interval ``1/fps`` the
      container advertises. So a frame's share is known one frame later, not at
      the end of the stream.
    * Resizing commutes with repeating. Every copy of a source frame resizes to
      the identical canvas frame, so resizing once and writing the result
      ``repeats`` times is byte-for-byte what resizing the repeated stack gives.
    * The canvas comes from the source geometry, which the first frame already
      carries — the resample and the truncation never change it.

    Which means the loop can also stop the moment ``num_frames`` slots are
    filled: everything after that would have been truncated away. Peak is then
    the output buffer plus two source frames, ~384 MiB for a 124-frame request
    on a 1344x768 canvas, against tens of GiB for the buffered path.

    The schedule reads *timestamps* and not frame indices, which for a
    constant-rate source is the same arithmetic — ``t_i = i/fps`` makes the
    boundary above ``floor((i+1) * target_fps/fps + 0.5)``, the index-based
    expression exactly — and for a variable-rate one is the difference between
    resampling a source and resampling a guess about it. An MP4 or MOV may be
    variable-rate and a screen recording routinely is; its soundtrack is sliced
    by exact sample time regardless, so an index-based schedule drifts the two
    conditions apart by an amount nothing in the container bounds.

    Args:
        timed_frames: An iterable of ``(timestamp_seconds, (height, width, 3)
            uint8 RGB)`` source frames, timestamps non-decreasing. They are read
            relative to the first, so an absolute container timestamp and one
            already rebased to the requested start behave identically. Consumed
            lazily, and abandoned early once the output is full.
        fps: The nominal rate, used to end the final frame.
        num_frames: The generated frame count the reference is truncated to.
            ``None`` keeps every frame — the legacy attribution knob — which
            gives up both the early stop and the preallocation, so that path
            peaks at twice the *canvas*-resolution output. Still bounded by the
            canvas rather than by the source, which is the point.
        canvas_multiple, canvas_short_edge, canvas_max_pixels: The canvas rule.
        resolve_canvas: ``(aspect_w, aspect_h, multiple, short_edge, max_pixels)
            -> (height, width)``.
        target_fps: MiniMax-H3's own rate.

    Returns:
        ``(num_frames_out, height, width, 3)`` uint8 RGB.
    """
    from PIL import Image

    if fps <= 0:
        raise ValueError(f"a reference video must have a positive frame rate, got {fps}")
    if num_frames is not None and num_frames <= 0:
        raise ValueError(f"num_frames must be positive when given, got {num_frames}")

    slot = 0
    written = 0
    canvas: tuple[int, int] | None = None
    buffer: np.ndarray | None = None
    chunks: list[np.ndarray] = []
    # The frame whose slot span is not known yet: it ends where the next one
    # starts, and the next one has not arrived. One frame of lookahead is the
    # whole cost of reading the schedule off timestamps instead of off indices.
    pending: np.ndarray | None = None
    pending_at = 0.0
    origin: float | None = None

    def place(frame: np.ndarray, boundary: int) -> np.ndarray | None:
        """Write ``frame`` into the slots up to ``boundary``; the result if full."""
        nonlocal slot, written
        repeats, slot = boundary - slot, max(boundary, slot)
        if repeats <= 0:
            # A source frame this rate drops. Skipping the resize is free.
            return None
        height, width = canvas
        if frame.shape[:2] != canvas:
            frame = np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))
        if buffer is not None:
            repeats = min(repeats, num_frames - written)
            buffer[written : written + repeats] = frame
            written += repeats
            return buffer if written == num_frames else None
        chunks.append(np.repeat(frame[None], repeats, axis=0))
        written += repeats
        return None

    for timestamp, frame in timed_frames:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"a reference video frame must be (height, width, 3) RGB, got {tuple(frame.shape)}")
        if frame.dtype != np.uint8:
            raise ValueError(f"a reference video must be uint8 after decoding, got {frame.dtype}")
        if canvas is None:
            canvas = resolve_canvas(
                frame.shape[1], frame.shape[0], canvas_multiple, canvas_short_edge, canvas_max_pixels
            )
            if num_frames is not None:
                buffer = np.empty((num_frames, *canvas, 3), dtype=np.uint8)
        if origin is None:
            origin = float(timestamp)
        elapsed = float(timestamp) - origin

        if pending is not None:
            full = place(pending, math.floor(elapsed * target_fps + 0.5))
            if full is not None:
                return full
        pending, pending_at = frame, elapsed

    if canvas is None:
        raise ValueError("a reference video decoded to no frames")
    if pending is not None:
        # The last frame has no successor to end it, so it gets the one nominal
        # interval the container advertises — which is exactly the boundary the
        # index-based schedule gave it.
        full = place(pending, math.floor((pending_at + 1.0 / fps) * target_fps + 0.5))
        if full is not None:
            return full
    if buffer is not None:
        # Short of the request: fewer slots than it generates, which the caller
        # handles (the reference simply stops conditioning part way through).
        # Copied rather than sliced so the unwritten tail is released instead of
        # being held alive by a view for as long as the reference is.
        return buffer[:written].copy()
    return np.concatenate(chunks) if chunks else np.empty((0, *canvas, 3), dtype=np.uint8)


def vae_chunk_frame_count(
    num_frames: int,
    *,
    frames_per_chunk: int = MINIMAX_H3_VAE_FRAMES_PER_CHUNK,
    latents_per_chunk: int = MINIMAX_H3_VAE_LATENTS_PER_CHUNK,
) -> int:
    """How many frames of a reference video the video VAE encodes.

    The VAE consumes ``17 * n + 5`` frames, and a generated clip's own frame
    count already has that form — so this only ever bites on a reference
    *shorter* than the clip, which keeps whatever the source ran to. A 2-second
    24 fps reference is 48 frames, and 48 is not ``17 * n + 5``; encoding it
    directly pads inside the VAE and yields a different latent temporal extent
    than the reference contract's, which changes the packed row count.

    Snapping *down* is what the official encoder does, and the arithmetic is
    reproduced from it rather than re-derived::

        max(1, (num_frames - latents_per_chunk) // frames_per_chunk) * frames_per_chunk + latents_per_chunk

    Including the ``max(1, ...)``, which makes the result exceed ``num_frames``
    below 22 frames. That is not a bug to clamp away: the caller slices, and a
    slice past the end simply keeps everything, so mirroring the expression
    exactly is what mirrors the behaviour. (Admission needs 2 seconds, i.e. 48
    frames at 24 fps, so the branch is unreachable in a served request — it is
    reachable from a unit test, and the two must not disagree there either.)

    Args:
        num_frames: Frames the normalized reference carries.
        frames_per_chunk: The VAE's ``clip_length``.
        latents_per_chunk: The VAE's ``tokens_chunk_size``.

    Returns:
        The frame count to encode, to be applied by slicing.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    chunks = max(1, (num_frames - latents_per_chunk) // frames_per_chunk)
    return chunks * frames_per_chunk + latents_per_chunk


def sample_conditioner_frames(
    num_frames: int,
    *,
    fps: float = MINIMAX_H3_FPS,
    sample_fps: float = 2.0,
    temporal_patch: int = 2,
) -> tuple[list[int], list[float]]:
    """Which normalized frames the Qwen3-VL conditioner reads, and their labels.

    The conditioner sees the reference at ``sample_fps``: every
    ``fps / sample_fps``-th frame, deduplicated. Qwen3-VL then merges the sampled
    frames in groups of ``temporal_patch`` — repeating the last one when the
    count does not divide — and each merged group is labelled with the mean of
    its timestamps. That label is rendered into the prompt as
    ``"<{t:.1f} seconds>"``, so it is a *token* contract, not a rendering detail:
    ``"{:.1f}"`` rounds half to even, which is why the first block of a 2 fps
    pair reads ``<0.2 seconds>`` rather than ``<0.3 seconds>``.

    Args:
        num_frames: Number of normalized (24 fps) frames.
        fps: The rate those frames are at.
        sample_fps: The rate the conditioner reads at.
        temporal_patch: Qwen3-VL's temporal patch, read off the processor.

    Returns:
        ``(frame_indices, block_timestamps)``.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    stride = fps / sample_fps
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < num_frames:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise ValueError(
            f"a reference video is read at {sample_fps:g} fps and merged in groups of {temporal_patch}, "
            f"so it must run at least {minimum} frames at {fps:g} fps, got {num_frames}"
        )

    timestamps = [index / sample_fps for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
    block_timestamps = [
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
        for index in range(0, len(timestamps), temporal_patch)
    ]
    return indices, block_timestamps


__all__ = [
    "MINIMAX_H3_FPS",
    "MINIMAX_H3_VAE_FRAMES_PER_CHUNK",
    "MINIMAX_H3_VAE_LATENTS_PER_CHUNK",
    "frame_slot_repeats",
    "normalize_reference_video_frames",
    "normalize_reference_video_stream",
    "resample_frame_indices",
    "sample_conditioner_frames",
    "vae_chunk_frame_count",
]
