# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Video/audio muxing utilities using PyAV (no ffmpeg binary dependency)."""

from __future__ import annotations

import io
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

import av
import numpy as np


@dataclass(frozen=True)
class VideoDelivery:
    """Delivery geometry applied to decoded frames at encode time.

    The model generates at its own native canvas; this rescales the decoded
    frames to the resolution the caller asked to be delivered, *before* the one
    and only encode. Doing it here rather than as a separate super-resolution
    hop avoids an extra encode/decode generation and keeps the intermediate out
    of a lossy container.

    ``sharpen`` is an ``unsharp`` luma amount compensating the interpolation
    kernel's rolloff. It is deliberately content-dependent and small: measured
    on real material, 0.3 buys ~1.17x high-frequency amplitude for a 0.0125 drop
    in inter-frame high-frequency correlation, while 1.0 pushes dark-region
    amplitude to 1.75x -- which is what re-amplifies generator noise in dark
    scenes. 0 disables the filter entirely.
    """

    width: int
    height: int
    sharpen: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"delivery size must be positive, got {self.width}x{self.height}")
        if self.sharpen < 0:
            raise ValueError(f"delivery sharpen must be >= 0, got {self.sharpen}")

    def matches(self, width: int, height: int) -> bool:
        """True when this delivery is a no-op for a source of the given size."""
        return self.width == width and self.height == height and self.sharpen <= 0


MAX_DELIVERY_SHORT_EDGE = 2160
MAX_DELIVERY_UPSCALE = 4.0
DEFAULT_DELIVERY_SHARPEN = 0.3


def resolve_delivery(
    *,
    source_width: int,
    source_height: int,
    short_edge: int | None,
    sharpen: float | None = None,
) -> VideoDelivery | None:
    """Turn a requested delivery short edge into a concrete, validated geometry.

    Only the short edge is a caller-facing knob: the long edge is derived from
    the *generated* frame so the picture is never stretched. Asking for a long
    edge as well is what silently anamorphoses output whenever the generator's
    real canvas differs from the nominal tier label.

    Returns ``None`` when there is nothing to do. Raises ``ValueError`` for a
    request beyond the caps -- delivering a soft 8K is worse than telling the
    caller the model tops out where it does.
    """
    if short_edge is None:
        return None
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"invalid source geometry {source_width}x{source_height}")
    if short_edge > MAX_DELIVERY_SHORT_EDGE:
        raise ValueError(f"delivery_short_edge must be <= {MAX_DELIVERY_SHORT_EDGE}, got {short_edge}")

    source_short = min(source_width, source_height)
    ratio = short_edge / source_short
    if ratio > MAX_DELIVERY_UPSCALE:
        raise ValueError(
            f"delivery_short_edge {short_edge} upscales the {source_width}x{source_height} "
            f"source by {ratio:.2f}x, above the {MAX_DELIVERY_UPSCALE:g}x ceiling"
        )

    # Downscaling loses nothing, so sharpening it only amplifies whatever noise
    # the generator already put there. Upscaling is the only case that needs the
    # interpolation kernel's rolloff compensated.
    effective_sharpen = (DEFAULT_DELIVERY_SHARPEN if sharpen is None else sharpen) if ratio > 1.0 else 0.0

    def _even(value: float) -> int:
        return max(2, int(round(value / 2.0)) * 2)

    if source_width <= source_height:
        width, height = _even(short_edge), _even(source_height * ratio)
    else:
        width, height = _even(source_width * ratio), _even(short_edge)

    delivery = VideoDelivery(width=width, height=height, sharpen=effective_sharpen)
    return None if delivery.matches(source_width, source_height) else delivery


def _build_delivery_graph(template: av.VideoFrame, delivery: VideoDelivery) -> av.filter.Graph:
    graph = av.filter.Graph()
    nodes = [
        graph.add_buffer(
            width=template.width,
            height=template.height,
            format=template.format.name,
            time_base=Fraction(1, 1000000),
        ),
        # param0=5 widens the Lanczos window from the default a=3. Measured on a
        # ground-truth downscale/upscale round trip it is the best pure scaler
        # (40.73 dB vs 40.63 for a=3); the margin is small but free.
        graph.add("scale", f"w={delivery.width}:h={delivery.height}:flags=lanczos:param0=5"),
    ]
    if delivery.sharpen > 0:
        nodes.append(graph.add("unsharp", f"5:5:{delivery.sharpen}:5:5:0"))
    nodes.append(graph.add("format", template.format.name))
    nodes.append(graph.add("buffersink"))
    for upstream, downstream in itertools.pairwise(nodes):
        upstream.link_to(downstream)
    graph.configure()
    return graph


def _iter_delivered_frames(
    frames: Iterable[av.VideoFrame],
    delivery: VideoDelivery,
) -> Iterator[av.VideoFrame]:
    """Rescale (and optionally sharpen) frames through one libavfilter graph.

    Frames enter the graph with a synthetic monotonic pts because buffersrc
    rejects a stream that never advances, and leave with ``pts=None`` so the
    encoder keeps assigning timestamps exactly as it does on the unscaled path.
    """
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        return
    if delivery.matches(first.width, first.height):
        yield from itertools.chain([first], iterator)
        return

    graph = _build_delivery_graph(first, delivery)
    for index, frame in enumerate(itertools.chain([first], iterator)):
        frame.pts = index
        frame.time_base = Fraction(1, 1000000)
        graph.push(frame)
        while True:
            try:
                filtered = graph.pull()
            except (av.error.BlockingIOError, av.error.EOFError):
                break
            filtered.pts = None
            yield filtered


class FragmentedMP4Muxer:
    """Incrementally mux video frames into one fragmented MP4 byte stream."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float = 25.0,
        video_codec: str = "h264",
        crf: str = "18",
        video_codec_options: dict[str, str] | None = None,
    ) -> None:
        self._buf = io.BytesIO()
        self._closed = False
        self._container = av.open(
            self._buf,
            mode="w",
            format="mp4",
            options={"movflags": "+frag_every_frame+empty_moov+default_base_moof"},
        )

        self._stream: av.VideoStream = cast(
            av.VideoStream,
            self._container.add_stream(video_codec, rate=Fraction(fps).limit_denominator(10000)),
        )
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"

        options: dict[str, object] = {"crf": str(crf)}
        if video_codec_options:
            options.update(video_codec_options)
        self._stream.options = options

        try:
            self._stream.codec_context.max_b_frames = 0
        except AttributeError:
            pass

    def mux_video_frames(self, video_frames: np.ndarray) -> bytes:
        """Mux a batch of ``uint8`` RGB frames and return newly written MP4 bytes."""
        if self._closed:
            raise RuntimeError("Cannot mux frames after FragmentedMP4Muxer.close().")
        if video_frames.ndim != 4 or video_frames.shape[-1] != 3:
            raise ValueError("video_frames must have shape (T, H, W, 3).")
        if video_frames.dtype != np.uint8:
            raise ValueError("video_frames must be uint8.")
        if video_frames.shape[1] != self._stream.height or video_frames.shape[2] != self._stream.width:
            raise ValueError("All fragmented MP4 chunks in a session must use the same frame size.")

        for frame_data in video_frames:
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
        return self._read_new_bytes()

    def close(self) -> bytes:
        """Flush delayed encoder packets, close the container, and return final bytes."""
        if self._closed:
            return b""
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._closed = True
        return self._read_new_bytes()

    def _read_new_bytes(self) -> bytes:
        """Return newly muxed bytes in the current video container,
        then clear the buffer to prepare for the next chunk."""
        chunk = self._buf.getvalue()
        self._buf.seek(0)
        self._buf.truncate()
        return chunk


def finalize_streaming_video_bytes(
    video_bytes: bytes,
    *,
    input_format: str,
    fps: float = 25.0,
    video_codec_options: dict[str, str] | None = None,
) -> bytes:
    """Convert streamed video bytes into a progressive MP4 for local playback."""
    if not video_bytes:
        return video_bytes

    normalized_format = input_format.lower()
    if normalized_format == "m4s":
        demux_format = "mp4"
    else:
        raise ValueError(f"Unsupported streaming video format: {input_format}")

    try:
        with cast(Any, av.open(io.BytesIO(video_bytes), format=demux_format)) as container:
            stream = container.streams.video[0]
            frame_arrays = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    except Exception:
        return video_bytes

    if not frame_arrays:
        return video_bytes

    frames_u8 = np.ascontiguousarray(np.stack(frame_arrays, axis=0), dtype=np.uint8)
    return mux_video_audio_bytes(
        frames_u8,
        None,
        fps=float(fps),
        video_codec_options=video_codec_options,
    )


def mux_video_audio_bytes(
    video_frames: np.ndarray,
    audio_waveform: np.ndarray | None = None,
    *,
    fps: float = 25.0,
    audio_sample_rate: int = 44100,
    video_codec: str = "h264",
    audio_codec: str = "aac",
    crf: str = "18",
    video_codec_options: dict[str, str] | None = None,
    delivery: VideoDelivery | None = None,
) -> bytes:
    """Mux video frames and optional audio waveform into MP4 bytes.

    Args:
        video_frames: uint8 array of shape ``(T, H, W, 3)`` (RGB).
        audio_waveform: float32 array – mono ``(N,)`` or ``(N, C)`` / ``(C, N)``.
        fps: Video frame rate.
        audio_sample_rate: Audio sample rate in Hz.
        delivery: Optional delivery geometry; rescales frames before the encode.
        video_codec: Video codec name.
        audio_codec: Audio codec name.
        crf: Constant rate factor for the video encoder.

    Returns:
        Raw MP4 bytes ready to be written to disk or streamed.
    """
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")

    v_stream = cast(av.VideoStream, container.add_stream(video_codec, rate=Fraction(fps).limit_denominator(10000)))
    v_stream.width = delivery.width if delivery is not None else video_frames.shape[2]
    v_stream.height = delivery.height if delivery is not None else video_frames.shape[1]
    v_stream.pix_fmt = "yuv420p"

    options: dict[str, object] = {"crf": str(crf)}
    if video_codec_options:
        options.update(video_codec_options)
    v_stream.options = options

    a_stream: av.AudioStream | None = None
    samples: np.ndarray | None = None
    layout: str | None = None
    if audio_waveform is not None:
        samples = audio_waveform.astype(np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        elif samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
            samples = np.ascontiguousarray(samples.T)
        num_channels = samples.shape[0]
        layout = "stereo" if num_channels >= 2 else "mono"
        a_stream = cast(av.AudioStream, container.add_stream(audio_codec, rate=audio_sample_rate))
        a_stream.layout = layout

    source_frames: Iterator[av.VideoFrame] = (
        av.VideoFrame.from_ndarray(frame_data, format="rgb24") for frame_data in video_frames
    )
    if delivery is not None:
        source_frames = _iter_delivered_frames(source_frames, delivery)
    for frame in source_frames:
        for packet in v_stream.encode(frame):
            container.mux(packet)
    for packet in v_stream.encode():
        container.mux(packet)

    if a_stream is not None and audio_waveform is not None:
        if samples is None or layout is None:
            raise ValueError("Audio samples were not prepared for muxing.")
        audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
        audio_frame.sample_rate = audio_sample_rate
        # AAC has a one-frame encoder delay. Mark the input waveform as
        # starting at t=0 so the MP4 muxer writes the corresponding negative
        # priming timestamp instead of exposing the delay as leading silence.
        audio_frame.pts = 0
        audio_frame.time_base = Fraction(1, audio_sample_rate)
        for packet in a_stream.encode(audio_frame):
            container.mux(packet)
        for packet in a_stream.encode():
            container.mux(packet)

    container.close()
    return buf.getvalue()


def mux_av_video_audio_bytes(
    video_frames: Iterable[av.VideoFrame],
    width: int,
    height: int,
    audio_waveform: np.ndarray | None = None,
    *,
    fps: float = 25.0,
    audio_sample_rate: int = 44100,
    video_codec: str = "h264",
    audio_codec: str = "aac",
    crf: str = "18",
    video_codec_options: dict[str, str] | None = None,
    delivery: VideoDelivery | None = None,
) -> bytes:
    """Mux preconstructed video frames and optional audio into MP4 bytes."""
    buf = io.BytesIO()
    with cast(Any, av.open(buf, mode="w", format="mp4")) as container:
        v_stream = cast(
            av.VideoStream,
            container.add_stream(video_codec, rate=Fraction(fps).limit_denominator(10000)),
        )
        v_stream.width = delivery.width if delivery is not None else width
        v_stream.height = delivery.height if delivery is not None else height
        v_stream.pix_fmt = "yuv420p"

        options: dict[str, object] = {"crf": str(crf)}
        if video_codec_options:
            options.update(video_codec_options)
        v_stream.options = options

        a_stream: av.AudioStream | None = None
        samples: np.ndarray | None = None
        layout: str | None = None
        if audio_waveform is not None:
            samples = audio_waveform.astype(np.float32)
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            elif samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
                samples = np.ascontiguousarray(samples.T)
            num_channels = samples.shape[0]
            layout = "stereo" if num_channels >= 2 else "mono"
            a_stream = cast(av.AudioStream, container.add_stream(audio_codec, rate=audio_sample_rate))
            a_stream.layout = layout

        source_frames = _iter_delivered_frames(video_frames, delivery) if delivery is not None else video_frames
        for frame in source_frames:
            for packet in v_stream.encode(frame):
                container.mux(packet)
        for packet in v_stream.encode():
            container.mux(packet)

        if a_stream is not None and audio_waveform is not None:
            if samples is None or layout is None:
                raise ValueError("Audio samples were not prepared for muxing.")
            audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
            audio_frame.sample_rate = audio_sample_rate
            # AAC has a one-frame encoder delay. Mark the input waveform as
            # starting at t=0 so the MP4 muxer writes the corresponding negative
            # priming timestamp instead of exposing the delay as leading silence.
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, audio_sample_rate)
            for packet in a_stream.encode(audio_frame):
                container.mux(packet)
            for packet in a_stream.encode():
                container.mux(packet)

    return buf.getvalue()
