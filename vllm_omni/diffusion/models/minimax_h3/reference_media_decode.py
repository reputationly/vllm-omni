# SPDX-License-Identifier: Apache-2.0
"""Decoding a `ref2va` reference straight from its source container.

The official path reads RGB24 frames and the real frame rate out of the source,
undoes the display-matrix rotation, and takes the soundtrack at the rate the
container carries it. Nothing is re-encoded on the way in.

vLLM-Omni's legacy path instead writes a ``libx264 + yuv420p`` intermediate and
reads *that* back, which costs a chroma subsampling and a lossy quantiser before
the VAE ever sees a pixel; the conditioner's frames are then pulled out of the
same intermediate one ``ffmpeg`` process per frame, each seeking from the start
of the file. Decoding once, losslessly, is both the official contract and a lot
less work — see the P-C4 entry in the problem log.

A non-square pixel aspect ratio is deliberately left alone: the reference
implementation resolved a reference's canvas from its *display* geometry, so a
stream carrying a sample aspect ratio is conditioned on at the wrong shape, and
"fixing" it here would be an untested divergence rather than an alignment.

Two ways in, and production takes the second one:

* :func:`decode_reference_video` materializes the whole stream at its native
  resolution. It is the straightforward reading of the contract and it is what
  the equivalence tests compare against, but its peak cost is the *source*
  resolution times the source frame count — for a 5760x2304 60 fps 15 s
  reference, which is a few tens of megabytes compressed and so sails through
  the 50 MiB admission gate, that is ~36 GiB of host RAM in the frame list and
  another ~36 GiB in the ``np.stack``. That is a server-wide OOM, not a failed
  request.
* :class:`ReferenceVideoReader` hands out one upright frame at a time, so a
  caller that resizes onto the generated canvas as it goes never holds more
  than a single source frame. See
  :func:`~.reference_video_frames.normalize_reference_video_stream`, which
  additionally stops pulling once it has the frames the request generates.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch


class Soundtrack(NamedTuple):
    """An audio stream as decoded, together with where on the clock it starts.

    ``start_seconds`` is the container timestamp of the first sample, and it is
    part of the value rather than something the caller reconstructs, because
    the waveform itself has already forgotten: decoding concatenates samples
    from wherever the stream begins, so index 0 means "the first sample" and
    not "time zero". A caller lining this up against video timestamps — which
    *are* absolute — has to subtract it, and a caller that does not know it
    exists subtracts the stream offset twice.

    Attributes:
        waveform: ``(channels, num_samples)`` float32, at ``sample_rate``.
        sample_rate: The rate the stream carries; no conversion happened.
        start_seconds: The container timestamp of sample 0. ``0.0`` for the
            ordinary file that starts at zero, and for a stream whose frames
            carry no usable timestamps.
    """

    waveform: torch.Tensor
    sample_rate: int
    start_seconds: float


@dataclass(frozen=True)
class DecodedReferenceVideo:
    """What one reference container yields, before any normalization.

    Attributes:
        frames: ``(num_frames, height, width, 3)`` uint8 RGB, upright.
        fps: The frame rate the container reports.
        audio: ``(channels, num_samples)`` float32, or None without an audio
            stream. Still at the container's own rate — resampling belongs to
            the normalization step, which does it exactly once.
        sample_rate: The rate ``audio`` carries, or None.
    """

    frames: np.ndarray
    fps: float
    audio: torch.Tensor | None
    sample_rate: int | None


def _import_av():
    try:
        import av
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError(
            "decoding a MiniMax-H3 reference from its source container needs PyAV (`pip install av`)."
        ) from error
    return av


def _decode_soundtrack(av, container, stream) -> Soundtrack | None:
    """An audio stream as ``(channels, num_samples)`` float32, at its own rate.

    The resampler is a *format* conversion only — planar float at the stream's
    own rate and layout — so no rate conversion happens here. A mono soundtrack
    stays mono; upmixing belongs to the normalization step.

    The first decoded frame's timestamp is read *before* resampling and carried
    out on the result. Resampling does not preserve it, and the concatenated
    waveform has no room for it, so it would otherwise be lost here — silently,
    and only on files that do not start at zero.

    Returns:
        A :class:`Soundtrack`, or ``None`` for a stream that is declared and
        decodes to nothing. Whether an empty track is an error is the *caller's*
        question and it has two different answers — a video's own soundtrack may
        legitimately be empty, a standalone audio reference may not — so it is
        not answered here.
    """
    sample_rate = int(stream.codec_context.sample_rate)
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
    chunks = []
    start_seconds = None
    for frame in container.decode(stream):
        if start_seconds is None:
            start_seconds = frame.time
        chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(frame)]
    chunks += [torch.from_numpy(resampled.to_ndarray()) for resampled in resampler.resample(None)]
    if not chunks:
        return None
    return Soundtrack(torch.cat(chunks, dim=-1).to(torch.float32), sample_rate, float(start_seconds or 0.0))


def apply_display_rotation(frames: np.ndarray, rotation: float) -> np.ndarray:
    """Undo the display matrix's counterclockwise rotation, as ffmpeg does.

    Snapped to the nearest quarter turn. Split out from the decode so it can be
    tested without a container carrying rotation metadata.
    """
    turns = round(rotation / 90.0) % 4
    if not turns:
        return frames
    return np.ascontiguousarray(np.rot90(frames, k=-turns, axes=(1, 2)))


class ReferenceVideoReader:
    """One opened container, read a frame at a time.

    The same decode as :func:`decode_reference_video` — same frames, same rate,
    same soundtrack — minus the buffering, so the caller decides what to keep.
    Nothing here holds a frame past the ``yield``.

    Two deliberate differences from the buffered decode, both invisible in
    practice:

    * The display rotation is taken from the *first* frame rather than the last.
      PyAV surfaces the stream's display matrix on every frame of it, so the two
      agree; the buffered version could only read the last one because it
      applied the rotation to the finished stack, and a reader that stops early
      never sees a last frame at all.
    * ``start_seconds`` drops leading frames by decoding past them instead of
      seeking. Seeking lands on a keyframe, which would silently shift the
      offset a request asked for by up to a GOP; decoding past them costs CPU
      and no memory, which is the trade this class exists to make.

    Every frame is handed over with its presentation timestamp, and that is not
    decoration. A container's frame rate is an *average*; an MP4 or MOV may be
    variable-rate, and a screen recording routinely is. Counting frames then
    locates a timestamp only when the two happen to coincide, so a consumer that
    resamples or offsets by frame index drifts against the same file's
    soundtrack — which is always sliced by exact sample time — by an amount the
    container never bounds. The timestamp is what both sides can agree on.
    """

    def __init__(self, av, container) -> None:
        self._av = av
        self._container = container
        if not container.streams.video:
            raise ValueError("media has no video stream to decode")
        self._stream = container.streams.video[0]
        self._frames: Generator[tuple[float, np.ndarray], None, None] | None = None

    @property
    def fps(self) -> float:
        """The frame rate the container reports."""
        return float(self._stream.average_rate or self._stream.guessed_rate)

    def iter_frames(self, *, start_seconds: float = 0.0) -> Iterator[tuple[float, np.ndarray]]:
        """``(timestamp, frame)`` in stream order, from ``start_seconds`` on.

        Args:
            start_seconds: Where the caller wants the stream to begin, relative
                to its first decoded video frame. The first frame handed over is
                the one *on screen* at that instant — the last whose relative
                timestamp is at or before it — so the requested moment is
                covered rather than skipped past. ``0`` starts at the first
                frame.

        Returns:
            An iterator of ``(seconds, (height, width, 3) uint8 RGB)`` pairs.
            The timestamp is the container's, absolute and unshifted, so a
            caller can align other streams of the same file against it. Each
            array is freshly allocated, so a caller may keep it, but nothing
            here does.
        """
        self.close_frames()
        self._frames = self._iter_frames(start_seconds=float(start_seconds))
        return self._frames

    def _iter_frames(self, *, start_seconds: float) -> Generator[tuple[float, np.ndarray], None, None]:
        rotation = 0.0
        # The most recent frame at or before relative `start_seconds`, held back
        # until we know whether a later one supersedes it. That makes the choice
        # "the frame on screen at `start_seconds`" rather than "the first frame
        # after it", and on a constant-rate stream it selects exactly the frame
        # a `floor(start * fps)` count used to.
        # Held as the decoded frame, not as an array: a skipped frame must not
        # pay for a conversion, which is most of the cost of decoding past an
        # offset in the first place.
        held = None
        held_at = 0.0
        started = False
        # ``start_seconds`` is relative to the first decoded video frame, while
        # PyAV's ``frame.time`` is an absolute container PTS.  Keep the latter
        # for A/V alignment, but compare on a rebased clock.  Without this, a
        # stream placed at PTS 5 s treats a requested 2 s offset as already
        # passed and silently starts from its first frame.
        origin: float | None = None

        def upright(frame) -> np.ndarray:
            array = frame.to_ndarray(format="rgb24")
            return array if not rotation else apply_display_rotation(array[None], rotation)[0]

        for position, frame in enumerate(self._container.decode(self._stream)):
            if not position:
                rotation = frame.rotation
            timestamp = frame.time
            # A stream with no usable presentation timestamps is the one case
            # where the frame index is all there is; fall back to the nominal
            # rate so such a file behaves exactly as it did before.
            if timestamp is not None:
                at = float(timestamp)
            else:
                # Preserve the absolute clock once its origin is known.  For a
                # wholly PTS-less stream ``origin`` starts at zero, reproducing
                # the old counted-frame fallback exactly.
                at = (origin if origin is not None else 0.0) + position / self.fps
            if origin is None:
                origin = at
            elapsed = at - origin
            # Rebasing a non-zero PTS can turn an exact boundary such as 0.08
            # into 0.08000000000000007.  A one-nanosecond allowance preserves
            # the same frame choice as an otherwise identical zero-based
            # stream without being large enough to cross any real frame gap.
            if not started and elapsed <= start_seconds + 1e-9:
                held, held_at = frame, at
                continue
            if not started and held is not None:
                yield held_at, upright(held)
                held = None
            started = True
            yield at, upright(frame)
        if not started and held is not None:
            # Everything sat at or before `start_seconds` — a last frame that
            # runs long. It is still the frame on screen there.
            yield held_at, upright(held)

    def close_frames(self) -> None:
        """End an unfinished :meth:`iter_frames` pass.

        A caller that stops early leaves a suspended demux sitting on the
        container; anything that moves the read position afterwards has to
        retire it first rather than leave two readers on one container.
        """
        if self._frames is not None:
            self._frames.close()
            self._frames = None

    def soundtrack(self) -> Soundtrack | None:
        """The soundtrack at its own rate, or None if there is no usable one.

        Retires the frame pass and seeks back to the start, so this is valid
        whether or not :meth:`iter_frames` ran to exhaustion.

        "No usable one" covers a stream that is *declared* and decodes to
        nothing — an empty or truncated audio track. That is a real file, and
        the path this replaced took it: the legacy route decided a reference had
        sound from the container *metadata* and shelled out to ffmpeg, which is
        happy to hand back silence. Raising here instead would reject a video
        that used to work, and reject it as a 500 rather than as anything the
        caller could act on. A track with no samples is, for every purpose
        downstream, a video without one.
        """
        self.close_frames()
        if not self._container.streams.audio:
            return None
        self._container.seek(0)
        return _decode_soundtrack(self._av, self._container, self._container.streams.audio[0])


@contextlib.contextmanager
def open_reference_video(path: str) -> Iterator[ReferenceVideoReader]:
    """A :class:`ReferenceVideoReader` over ``path``, closed on the way out."""
    av = _import_av()
    with av.open(str(path)) as container:
        yield ReferenceVideoReader(av, container)


def decode_reference_video(path: str, *, with_audio: bool = True) -> DecodedReferenceVideo:
    """Decode one reference video container losslessly, all at once.

    Peaks at twice the decoded source stream, so production reads through
    :func:`open_reference_video` instead; this stays as the plain statement of
    what the decode means and as the reference the streaming path is tested
    against.

    Args:
        path: A local path to the source container.
        with_audio: Whether to make the second pass for the soundtrack.

    Returns:
        `DecodedReferenceVideo`.
    """
    av = _import_av()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames, rotation = [], 0.0
        for frame in container.decode(stream):
            # The display matrix belongs to the stream; PyAV surfaces it on
            # every frame of it.
            rotation = frame.rotation
            frames.append(frame.to_ndarray(format="rgb24"))
        frame_rate = float(stream.average_rate or stream.guessed_rate)
        soundtrack = None
        if with_audio and container.streams.audio:
            # Decoding the video drained the container, so the soundtrack needs
            # a second pass over it.
            container.seek(0)
            soundtrack = _decode_soundtrack(av, container, container.streams.audio[0])

    if not frames:
        raise ValueError(f"no video frames to decode in {path}")
    stacked = apply_display_rotation(np.stack(frames), rotation)
    waveform, sample_rate = (soundtrack.waveform, soundtrack.sample_rate) if soundtrack else (None, None)
    return DecodedReferenceVideo(frames=stacked, fps=frame_rate, audio=waveform, sample_rate=sample_rate)


def decode_reference_audio(path: str) -> tuple[torch.Tensor, int]:
    """An audio container's waveform and its own sample rate.

    Empty is an error here and not in :meth:`ReferenceVideoReader.soundtrack`,
    because a standalone audio reference *is* the condition: there is nothing
    left of the request if it carries no samples. A video's soundtrack is one
    optional part of one, so an empty one is simply absent.
    """
    av = _import_av()
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream to decode in {path}")
        decoded = _decode_soundtrack(av, container, container.streams.audio[0])
    if decoded is None:
        raise ValueError(f"the audio stream decoded to no samples in {path}")
    # A standalone reference is its own clock: nothing is being aligned
    # against it, so where the container placed it does not survive here.
    return decoded.waveform, decoded.sample_rate


__all__ = [
    "DecodedReferenceVideo",
    "Soundtrack",
    "ReferenceVideoReader",
    "apply_display_rotation",
    "decode_reference_audio",
    "decode_reference_video",
    "open_reference_video",
]
