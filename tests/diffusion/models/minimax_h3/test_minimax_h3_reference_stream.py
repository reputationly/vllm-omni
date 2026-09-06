# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded-memory `ref2va` reference decoding, against the buffered path.

The buffered path is correct and unusable: admission gates the *compressed*
file at 50 MiB, and a compressed 4K 60 fps 15 s reference passes that while
expanding to tens of GiB of RGB in host RAM, which takes the server down rather
than failing the request. The streaming path exists to bound that by the
*generated* canvas instead.

Which makes two things worth asserting, and they pull in opposite directions:

* it produces the very same bytes as the buffered path — otherwise this is a
  contract change wearing a memory fix's clothes;
* it never holds more than one source frame, and it stops decoding once the
  request's frames are filled — otherwise it is the buffered path with extra
  steps.

Most of this needs no codec: the normalizer takes an iterable, so a plain list
of arrays exercises the arithmetic. The container-level tests synthesize a file
with ffmpeg and skip where ffmpeg or PyAV is missing.

No weights, no GPU.
"""

from __future__ import annotations

import gc
import shutil
import subprocess
import weakref
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_HAS_FFMPEG = shutil.which("ffmpeg") is not None

# The rates admission actually lets through, plus the target itself.
_SOURCE_RATES = (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 60.0)


def _resolve_canvas(aspect_width, aspect_height, multiple, short_edge, max_pixels):
    from vllm_omni.diffusion.models.minimax_h3.reference_video import _resolve_reference_canvas

    return _resolve_reference_canvas(aspect_width, aspect_height, multiple, short_edge, max_pixels)


def _identity_canvas(aspect_width, aspect_height, multiple, short_edge, max_pixels):
    """A canvas rule that leaves the source geometry alone.

    The real rule snaps to a multiple of 32 and so resizes almost anything,
    which is right for production and wrong for a test about *which frame lands
    in which slot*: LANCZOS would rewrite the very pixels that carry the frame
    index. The resize gets its own test, against the real rule.
    """
    return int(aspect_height), int(aspect_width)


def _source_frames(count: int, *, height: int, width: int) -> np.ndarray:
    """Frames whose pixels identify their own index, deterministically."""
    frames = np.zeros((count, height, width, 3), dtype=np.uint8)
    for index in range(count):
        frames[index] = index % 256
        # A little structure so LANCZOS has something to do beyond a constant.
        frames[index, : height // 2, : width // 2, 0] = (index * 7) % 256
    return frames


def _timed(frames, fps: float, *, origin: float = 0.0):
    """``frames`` at a constant ``fps``, in the ``(timestamp, frame)`` shape.

    Constant-rate is the case the streaming normalizer has to keep byte-exact,
    so every equivalence test below feeds it through here: an evenly spaced
    timestamp is the definition of the frame index the arithmetic used to read.
    """
    for index, frame in enumerate(frames):
        yield origin + index / fps, frame


def _normalize_both(
    frames: np.ndarray, *, fps: float, num_frames: int | None, short_edge: int = 768, resolve_canvas=_identity_canvas
):
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import (
        normalize_reference_video_frames,
        normalize_reference_video_stream,
    )

    kwargs = dict(
        fps=fps,
        num_frames=num_frames,
        canvas_multiple=32,
        canvas_short_edge=short_edge,
        canvas_max_pixels=768 * 1344,
        resolve_canvas=resolve_canvas,
    )
    return normalize_reference_video_frames(frames, **kwargs), normalize_reference_video_stream(
        _timed(frames, fps), **kwargs
    )


# --------------------------------------------------------------------------
# Same bytes as the buffered path
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fps", _SOURCE_RATES)
@pytest.mark.parametrize("num_frames", [None, 1, 24, 121])
def test_streaming_normalization_is_byte_identical(fps, num_frames):
    """Every admitted rate, truncated and not.

    On an identity canvas, so this compares the resample and the truncation on
    their own; the resize is compared separately below.
    """
    frames = _source_frames(150, height=48, width=64)

    buffered, streamed = _normalize_both(frames, fps=fps, num_frames=num_frames)

    assert streamed.shape == buffered.shape, f"{fps} fps -> {num_frames}: output frame count differs"
    assert streamed.dtype == buffered.dtype
    assert np.array_equal(streamed, buffered), f"{fps} fps -> {num_frames}: normalized frames differ"


@pytest.mark.parametrize("fps", [23.976, 25.0, 60.0])
def test_streaming_normalization_is_byte_identical_through_the_resize(fps):
    """The LANCZOS pass too, which is where "resize then repeat" has to commute.

    Resizing once per *source* frame and writing the result to every slot it
    occupies is only equal to resizing the repeated stack because the resize is
    deterministic and per-frame. If that ever stops holding — a filter with
    temporal state, say — this is the test that says so.
    """
    frames = _source_frames(40, height=90, width=160)

    buffered, streamed = _normalize_both(frames, fps=fps, num_frames=48, short_edge=768, resolve_canvas=_resolve_canvas)

    assert buffered.shape[1:3] != frames.shape[1:3], "the canvas rule was expected to resize these frames"
    assert np.array_equal(streamed, buffered), f"{fps} fps: normalized frames differ through the resize"


def test_streaming_normalization_carries_the_official_slot_mapping():
    """Which source frame lands in which slot, read straight off the pixels.

    The same property ``test_reference_video_resample_matches_official``
    pins for the buffered path, restated against the streaming one so the two
    are anchored to the contract rather than only to each other.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import (
        normalize_reference_video_stream,
        resample_frame_indices,
    )

    for fps, num_frames in ((25.0, 30), (30.0, 24), (60.0, 49), (23.976, 121)):
        frames = _source_frames(200, height=48, width=64)
        streamed = normalize_reference_video_stream(
            _timed(frames, fps),
            fps=fps,
            num_frames=num_frames,
            canvas_multiple=32,
            canvas_short_edge=48,
            canvas_max_pixels=768 * 1344,
            resolve_canvas=_identity_canvas,
        )
        expected = list(resample_frame_indices(200, fps)[:num_frames])
        assert [int(frame[-1, -1, 0]) for frame in streamed] == expected, f"{fps} fps: slot mapping differs"


def test_a_reference_shorter_than_the_request_is_not_padded():
    """Fewer slots than the request generates comes back short, as before.

    The preallocated buffer must not leak its uninitialized tail — the caller
    reads ``frames.shape[0]`` to decide how many packed rows the reference
    contributes, so a padded array would silently condition on garbage.
    """
    frames = _source_frames(10, height=48, width=64)

    buffered, streamed = _normalize_both(frames, fps=24.0, num_frames=121)

    assert streamed.shape[0] == 10
    assert np.array_equal(streamed, buffered)


def test_an_empty_stream_is_refused_rather_than_shaped():
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import normalize_reference_video_stream

    with pytest.raises(ValueError, match="no frames"):
        normalize_reference_video_stream(
            iter(()),
            fps=24.0,
            num_frames=121,
            canvas_multiple=32,
            canvas_short_edge=768,
            canvas_max_pixels=768 * 1344,
            resolve_canvas=_resolve_canvas,
        )


# --------------------------------------------------------------------------
# ...and bounded while doing it
# --------------------------------------------------------------------------


def test_decoding_stops_once_the_request_is_filled():
    """A 15 s 60 fps reference must not be decoded past the 5 s the request wants.

    Counted at the source: a 60 fps source fills 121 slots at 24 fps well before
    the 900 frames it carries, and the buffered path decoded every one of them
    only to throw the tail away.
    """
    import math

    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import normalize_reference_video_stream

    pulled = 0

    def counted():
        nonlocal pulled
        for index, frame in enumerate(_source_frames(900, height=48, width=64)):
            pulled += 1
            yield index / 60.0, frame

    streamed = normalize_reference_video_stream(
        counted(),
        fps=60.0,
        num_frames=121,
        canvas_multiple=32,
        canvas_short_edge=48,
        canvas_max_pixels=768 * 1344,
        resolve_canvas=_resolve_canvas,
    )

    # The first source frame whose slot boundary reaches 121, by the same
    # expression the normalizer uses — spelling the number out here would only
    # record a rounding convention twice. One more than that is pulled: a
    # frame's slot span ends where its successor begins, so the successor has to
    # arrive before the frame that fills the output can be written.
    enough = next(count for count in range(1, 901) if math.floor(count * 24.0 / 60.0 + 0.5) >= 121)
    assert streamed.shape[0] == 121
    assert pulled == enough + 1 < 900, f"decoded {pulled} source frames for 121 output frames"


def test_only_one_source_frame_is_alive_at_a_time():
    """The property the whole change exists for, asserted rather than argued.

    CPython frees a frame as soon as the normalizer stops referring to it, so a
    weak reference taken at ``yield`` time is dead a frame or two later. The
    bound is one — the ``for`` target and the one-frame lookahead the timestamp
    schedule needs are the *same* frame while ``next()`` runs — and not "a few":
    if a future edit collects the source frames into a list, which is the shape
    of the original bug, every earlier weakref stays alive and the count climbs
    with the stream.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import normalize_reference_video_stream

    handed_out: list[weakref.ref] = []
    still_alive: list[int] = []

    def watched():
        for index, frame in enumerate(_source_frames(120, height=90, width=160)):
            frame = frame.copy()  # a fresh array per frame, as a decoder gives
            still_alive.append(sum(reference() is not None for reference in handed_out))
            handed_out.append(weakref.ref(frame))
            yield index / 30.0, frame
            del frame

    normalize_reference_video_stream(
        watched(),
        fps=30.0,
        num_frames=96,
        canvas_multiple=32,
        canvas_short_edge=768,
        canvas_max_pixels=768 * 1344,
        resolve_canvas=_resolve_canvas,
    )

    gc.collect()
    assert len(handed_out) == 120, "the whole stream was expected to be consumed"
    assert max(still_alive) <= 1, f"up to {max(still_alive)} earlier source frames were still referenced"


# --------------------------------------------------------------------------
# The reader, over a container
# --------------------------------------------------------------------------


def _fake_container(
    frames: np.ndarray,
    *,
    rotation: float = 0.0,
    rate: float = 30.0,
    with_audio: bool = False,
    times: Sequence[float] | None = None,
):
    """Just enough of a PyAV container for the reader, minus the codec.

    ``times`` is the presentation timestamp per frame; the default lays them out
    at ``rate``, which is what a constant-rate container does. Passing an uneven
    list is how a variable-rate source is expressed, and ``None`` entries stand
    for a stream that carries no usable timestamps at all.
    """

    class _Frame:
        def __init__(self, array, time):
            self._array = array
            self.rotation = rotation
            self.time = time

        def to_ndarray(self, format):
            assert format == "rgb24"
            return self._array

    stamps = [index / rate for index in range(len(frames))] if times is None else list(times)
    stream = SimpleNamespace(average_rate=rate, guessed_rate=rate)
    container = SimpleNamespace(
        streams=SimpleNamespace(video=[stream], audio=[object()] if with_audio else []),
        decode=lambda stream: (_Frame(frame, stamp) for frame, stamp in zip(frames, stamps, strict=True)),
        seek=lambda offset: None,
    )
    return container, stream


def test_the_reader_skips_leading_frames_and_uprights_them():
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import ReferenceVideoReader

    frames = _source_frames(6, height=4, width=8)
    container, _ = _fake_container(frames, rotation=90.0, rate=25.0)

    reader = ReferenceVideoReader(object(), container)

    assert reader.fps == pytest.approx(25.0)
    # 2/25 s in, so on a constant-rate stream this is the frame the old
    # `skip=int(start * fps)` count landed on.
    read = list(reader.iter_frames(start_seconds=2 / 25.0))
    assert [stamp for stamp, _ in read] == pytest.approx([index / 25.0 for index in range(2, 6)])
    # Rotated per frame, which is the same quarter turn the buffered path
    # applies to the whole stack.
    assert read[0][1].shape == (8, 4, 3)
    assert np.array_equal(np.stack([frame for _, frame in read]), np.rot90(frames[2:], k=-1, axes=(1, 2)))


def test_the_reader_applies_offsets_relative_to_a_nonzero_container_pts():
    """The request offset is relative even though yielded timestamps are absolute."""
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import ReferenceVideoReader

    frames = _source_frames(6, height=4, width=8)
    origin = 5.0
    times = [origin + index / 25.0 for index in range(len(frames))]
    container, _ = _fake_container(frames, rate=25.0, times=times)

    read = list(ReferenceVideoReader(object(), container).iter_frames(start_seconds=2 / 25.0))

    assert [stamp for stamp, _ in read] == pytest.approx(times[2:])
    assert int(read[0][1][-1, -1, 0]) == 2


def test_the_reader_refuses_a_container_without_a_video_stream():
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import ReferenceVideoReader

    container = SimpleNamespace(streams=SimpleNamespace(video=[], audio=[]))
    with pytest.raises(ValueError, match="no video stream"):
        ReferenceVideoReader(object(), container)


def test_an_abandoned_frame_pass_is_retired_before_the_soundtrack():
    """Stopping early leaves a suspended demux; the audio pass has to retire it.

    Two live readers on one container is the kind of bug that shows up as a
    corrupt soundtrack on one file in a hundred, so it is closed explicitly and
    asserted here rather than left to the garbage collector.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import ReferenceVideoReader

    frames = _source_frames(6, height=4, width=8)
    container, _ = _fake_container(frames, with_audio=False)
    reader = ReferenceVideoReader(object(), container)

    stream = reader.iter_frames()
    assert next(stream) is not None
    assert reader.soundtrack() is None  # no audio stream, but it still retires the pass

    with pytest.raises(StopIteration):
        next(stream)


def test_a_declared_but_empty_audio_track_is_a_video_without_one():
    """A truncated soundtrack must not become a 500 on a file that used to work.

    The path this replaced decided a reference had sound from the container
    *metadata* (``bool(meta["audio_codecs"])``) and shelled out to ffmpeg, which
    hands back silence for an empty track. Raising here instead would reject a
    real file that legacy accepted — and reject it as a server error, which no
    caller can act on. For everything downstream, a track with no samples and no
    track at all are the same video.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import ReferenceVideoReader

    frames = _source_frames(4, height=4, width=8)
    container, _ = _fake_container(frames, with_audio=True)
    # A declared stream whose decode yields nothing: no packets, and the
    # resampler flush produces nothing either.
    empty = SimpleNamespace(resample=lambda frame: [])
    av = SimpleNamespace(audio=SimpleNamespace(resampler=SimpleNamespace(AudioResampler=lambda **_: empty)))
    container.streams.audio = [SimpleNamespace(codec_context=SimpleNamespace(sample_rate=44100), layout="stereo")]
    container.decode = lambda stream: iter(())

    assert ReferenceVideoReader(av, container).soundtrack() is None


def test_the_soundtrack_reports_where_on_the_clock_it_starts():
    """Because the waveform itself cannot: index 0 is "first sample", not "0 s".

    A container whose streams begin at 5 s hands back samples starting at that
    instant, with the 5 s nowhere in the array. A caller aligning them against
    video timestamps — which *are* absolute — that assumes otherwise subtracts
    the stream offset a second time, and on this file a request starting at zero
    loses five seconds of sound, or all of it.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import _decode_soundtrack

    class _AudioFrame:
        def __init__(self, time):
            self.time = time

        def to_ndarray(self):
            return np.zeros((1, 4), dtype=np.float32)

    frames = [_AudioFrame(5.0), _AudioFrame(5.1)]
    resampler = SimpleNamespace(resample=lambda frame: [] if frame is None else [frame])
    av = SimpleNamespace(audio=SimpleNamespace(resampler=SimpleNamespace(AudioResampler=lambda **_: resampler)))
    container = SimpleNamespace(decode=lambda stream: iter(frames))
    stream = SimpleNamespace(codec_context=SimpleNamespace(sample_rate=8000), layout="mono")

    decoded = _decode_soundtrack(av, container, stream)

    assert decoded.sample_rate == 8000
    assert decoded.waveform.shape == (1, 8)
    assert decoded.start_seconds == pytest.approx(5.0)


def test_a_stream_without_audio_timestamps_starts_at_zero():
    """The fallback has to be the ordinary answer, not None: it is subtracted."""
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import _decode_soundtrack

    frame = SimpleNamespace(time=None, to_ndarray=lambda: np.zeros((2, 3), dtype=np.float32))
    resampler = SimpleNamespace(resample=lambda f: [] if f is None else [f])
    av = SimpleNamespace(audio=SimpleNamespace(resampler=SimpleNamespace(AudioResampler=lambda **_: resampler)))
    container = SimpleNamespace(decode=lambda stream: iter([frame]))
    stream = SimpleNamespace(codec_context=SimpleNamespace(sample_rate=16000), layout="stereo")

    assert _decode_soundtrack(av, container, stream).start_seconds == 0.0


def test_an_empty_standalone_audio_reference_is_still_an_error(monkeypatch):
    """The other half: a standalone reference with no samples is not a reference.

    Same primitive, opposite answer, because the two callers ask different
    questions — one about an optional part of a condition, one about the whole
    condition. Leniency that reached here would turn an unusable request into a
    silent one.
    """
    import contextlib

    from vllm_omni.diffusion.models.minimax_h3 import reference_media_decode

    empty = SimpleNamespace(resample=lambda frame: [])
    stream = SimpleNamespace(codec_context=SimpleNamespace(sample_rate=44100), layout="stereo")
    container = SimpleNamespace(decode=lambda stream: iter(()), streams=SimpleNamespace(audio=[stream]))
    av = SimpleNamespace(
        audio=SimpleNamespace(resampler=SimpleNamespace(AudioResampler=lambda **_: empty)),
        open=lambda _path: contextlib.nullcontext(container),
    )
    monkeypatch.setattr(reference_media_decode, "_import_av", lambda: av)

    with pytest.raises(ValueError, match="decoded to no samples"):
        reference_media_decode.decode_reference_audio("/nowhere.wav")


# --------------------------------------------------------------------------
# End to end, with a real codec
# --------------------------------------------------------------------------


def _synthesize(path: Path, *, fps: int, seconds: float, width: int, height: int, with_audio: bool) -> None:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate={fps}:duration={seconds}",
    ]
    if with_audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        command += ["-c:a", "aac"]
    command += [str(path)]
    subprocess.run(command, check=True)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    if not _HAS_FFMPEG:
        pytest.skip("ffmpeg is not available")
    pytest.importorskip("av")
    path = tmp_path_factory.mktemp("h3_stream") / "reference.mp4"
    _synthesize(path, fps=30, seconds=2.0, width=64, height=48, with_audio=True)
    return path


@pytest.fixture(scope="module")
def admissible_video(tmp_path_factory) -> Path:
    """Inside the admission envelope, so ``prepare_reference_videos_official`` runs.

    Bigger and longer than ``sample_video`` for no reason other than the gates:
    at least 256 pixels a side, a ratio inside [0.4, 2.5], and at least two
    seconds.
    """
    if not _HAS_FFMPEG:
        pytest.skip("ffmpeg is not available")
    pytest.importorskip("av")
    path = tmp_path_factory.mktemp("h3_stream_admissible") / "reference.mp4"
    _synthesize(path, fps=30, seconds=3.0, width=512, height=288, with_audio=True)
    return path


@pytest.mark.parametrize("skip", [0, 7])
def test_the_reader_matches_the_buffered_decode_over_a_real_container(sample_video, skip):
    """Same frames, same rate, same soundtrack — read a frame at a time.

    Parametrized by frame count rather than by seconds because that is the claim
    worth pinning against a real codec: on a constant-rate container, asking for
    ``skip / fps`` seconds hands over exactly the frames the old count did.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import (
        decode_reference_video,
        open_reference_video,
    )

    buffered = decode_reference_video(str(sample_video))

    with open_reference_video(str(sample_video)) as reader:
        assert reader.fps == pytest.approx(buffered.fps)
        read = list(reader.iter_frames(start_seconds=skip / buffered.fps))
        streamed = np.stack([frame for _, frame in read])
        soundtrack = reader.soundtrack()

    assert np.array_equal(streamed, buffered.frames[skip:]), "streamed frames differ from the buffered decode"
    assert read[0][0] == pytest.approx(skip / buffered.fps, abs=1e-6), "the first frame is not at the asked-for time"
    assert soundtrack is not None
    waveform, sample_rate, audio_start = soundtrack
    assert sample_rate == buffered.sample_rate
    assert audio_start == pytest.approx(0.0, abs=1e-6), "an ordinary file starts at zero"

    assert np.array_equal(waveform.numpy(), buffered.audio.numpy()), "streamed soundtrack differs"


def test_the_soundtrack_survives_an_early_stop(sample_video):
    """The production shape: read a few frames, then take the whole soundtrack.

    Seeking back over a demux that was abandoned mid-stream is exactly what the
    early stop makes routine, so it is checked against the buffered decode.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import (
        decode_reference_video,
        open_reference_video,
    )

    buffered = decode_reference_video(str(sample_video))

    with open_reference_video(str(sample_video)) as reader:
        frames = reader.iter_frames()
        for _ in range(3):
            next(frames)
        waveform, sample_rate, _ = reader.soundtrack()

    assert sample_rate == buffered.sample_rate
    assert np.array_equal(waveform.numpy(), buffered.audio.numpy())


def test_the_prepared_reference_is_unchanged_by_the_streaming_switch(admissible_video):
    """``prepare_reference_videos_official`` end to end, against the old path.

    The buffered composition it used to run is reproduced here explicitly, so
    this fails if the switch changed the reference the pipeline conditions on —
    frames, canvas, duration or soundtrack.
    """
    import torch
    from vllm_omni.diffusion.models.minimax_h3.reference_video import (
        MINIMAX_H3_BASE_SHORT_EDGE,
        MINIMAX_H3_CANVAS_MULTIPLE,
        MINIMAX_H3_MAX_PIXELS,
        _resolve_reference_canvas,
        prepare_reference_videos_official,
    )

    from vllm_omni.diffusion.models.minimax_h3.reference_audio import normalize_reference_audio
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import decode_reference_video
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import normalize_reference_video_frames

    target_frame_count = 25

    prepared = prepare_reference_videos_official(
        [str(admissible_video)],
        target_frame_count=target_frame_count,
        audio_sample_rate=16000,
    )[0]

    decoded = decode_reference_video(str(admissible_video))
    expected_frames = normalize_reference_video_frames(
        decoded.frames,
        fps=decoded.fps,
        num_frames=target_frame_count,
        canvas_multiple=MINIMAX_H3_CANVAS_MULTIPLE,
        canvas_short_edge=MINIMAX_H3_BASE_SHORT_EDGE,
        canvas_max_pixels=MINIMAX_H3_MAX_PIXELS,
        resolve_canvas=_resolve_reference_canvas,
    )
    expected_audio = normalize_reference_audio(
        decoded.audio,
        int(decoded.sample_rate),
        target_sample_rate=16000,
        max_duration=target_frame_count / 24.0,
    )

    assert np.array_equal(prepared["frames"], expected_frames)
    assert (prepared["height"], prepared["width"]) == expected_frames.shape[1:3]
    assert prepared["duration_seconds"] == pytest.approx(expected_frames.shape[0] / 24.0)
    assert torch.equal(prepared["audio"], expected_audio)


@pytest.fixture(scope="module")
def offset_video(tmp_path_factory) -> Path:
    """The same reference, with both streams placed five seconds up the clock.

    ``-output_ts_offset`` is what a container recorded mid-session looks like,
    and it is the one shape where "the timestamp of the first frame" and "how
    far into the file that is" stop being the same number.
    """
    if not _HAS_FFMPEG:
        pytest.skip("ffmpeg is not available")
    pytest.importorskip("av")
    path = tmp_path_factory.mktemp("h3_stream_offset") / "reference.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=512x288:rate=30:duration=3.0",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=3.0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-output_ts_offset", "5.0", "-muxdelay", "0", "-muxpreload", "0",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


@pytest.fixture(scope="module")
def offset_video_with_audio_marker(tmp_path_factory) -> Path:
    """A shifted clock with a source-relative, content-visible audio marker.

    Both streams begin together five seconds up the container clock.  The
    soundtrack itself is silent for 0.9 seconds and then carries a tone, so a
    request beginning at relative 0.4 seconds must hear that transition about
    0.5 seconds into the prepared condition.  Unlike comparing against a slice
    of the decoded waveform, that fact comes from the fixture, not production's
    timestamp arithmetic.
    """
    if not _HAS_FFMPEG:
        pytest.skip("ffmpeg is not available")
    pytest.importorskip("av")
    path = tmp_path_factory.mktemp("h3_stream_offset_marker") / "reference.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=512x288:rate=30:duration=3.0",
            "-f", "lavfi", "-i",
            "aevalsrc=if(lt(t\\,0.9)\\,0\\,0.8*sin(2*PI*440*t)):s=48000:d=3.0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-output_ts_offset", "5.0", "-muxdelay", "0", "-muxpreload", "0",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def test_a_container_that_does_not_start_at_zero_keeps_its_whole_soundtrack(offset_video):
    """The cut is the difference of two clocks, not one of them.

    The video's timestamps are absolute, so the first frame of a file recorded
    five seconds into a session reads 5.0 — while the decoded waveform starts at
    its own first sample, with that 5.0 nowhere in the array. Trimming it by the
    video timestamp alone takes the offset off twice and throws away five
    seconds of sound, which on this three-second file is all of it.
    """
    import torch
    from vllm_omni.diffusion.models.minimax_h3.reference_video import prepare_reference_videos_official

    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import open_reference_video

    with open_reference_video(str(offset_video)) as reader:
        first_at, _ = next(reader.iter_frames())
        soundtrack = reader.soundtrack()
    assert first_at > 4.0, "the fixture was expected to start well past zero"
    assert soundtrack.start_seconds > 4.0, "and its soundtrack with it"

    prepared = prepare_reference_videos_official([str(offset_video)], target_frame_count=25, audio_sample_rate=16000)[0]

    assert prepared["input_has_audio"] is True
    audio = prepared["audio"]
    assert audio is not None and audio.shape[-1] > 0, "the soundtrack was cut away entirely"
    # A full second of the requested window, not a remnant: the request asked
    # from the start, so essentially nothing should have been trimmed.
    assert audio.shape[-1] >= 16000, f"only {audio.shape[-1]} samples survived"
    assert torch.count_nonzero(audio) > 0, "the surviving samples are silence"


def test_a_relative_start_cuts_both_streams_inside_a_nonzero_container_clock(offset_video):
    """A non-zero request offset must not be compared directly with absolute PTS."""
    import torch
    from vllm_omni.diffusion.models.minimax_h3.reference_video import prepare_reference_videos_official

    from vllm_omni.diffusion.models.minimax_h3.reference_audio import normalize_reference_audio
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import open_reference_video

    start = 0.37  # deliberately between two 30 fps presentation timestamps
    with open_reference_video(str(offset_video)) as reader:
        origin, _ = next(reader.iter_frames())
        soundtrack = reader.soundtrack()
    assert soundtrack is not None
    with open_reference_video(str(offset_video)) as reader:
        video_start, _ = next(reader.iter_frames(start_seconds=start))

    assert 0.0 < video_start - origin <= start
    assert start - (video_start - origin) < 1.0 / 30.0 + 1e-6

    prepared = prepare_reference_videos_official(
        [str(offset_video)],
        target_frame_count=25,
        start_time_seconds=start,
        audio_sample_rate=16000,
    )[0]
    trim = video_start - soundtrack.start_seconds
    expected = normalize_reference_audio(
        soundtrack.waveform[:, int(trim * soundtrack.sample_rate) :],
        soundtrack.sample_rate,
        target_sample_rate=16000,
        max_duration=25 / 24.0,
    )

    assert torch.equal(prepared["audio"], expected)


def test_a_relative_start_places_known_audio_content_without_reusing_the_trim_formula(
    offset_video_with_audio_marker,
):
    """The requested media-relative cut is observable in the audio content."""
    import torch
    from vllm_omni.diffusion.models.minimax_h3.reference_video import prepare_reference_videos_official

    sample_rate = 48000
    prepared = prepare_reference_videos_official(
        [str(offset_video_with_audio_marker)],
        target_frame_count=49,
        start_time_seconds=0.4,
        audio_sample_rate=sample_rate,
    )[0]
    audio = prepared["audio"][0]

    # Inspect 10 ms windows. AAC may smear the transition by a packet, but it
    # cannot move a marker authored at source-relative 0.9 s anywhere near the
    # 0.9 s position it would retain if the requested 0.4 s cut were ignored.
    window = sample_rate // 100
    rms = audio[: audio.numel() // window * window].reshape(-1, window).square().mean(dim=1).sqrt()
    audible = torch.nonzero(rms > 0.05).flatten()

    assert audible.numel() > 0, "the known tone disappeared"
    onset_seconds = int(audible[0]) * window / sample_rate
    assert 0.4 <= onset_seconds <= 0.6, f"the tone began at {onset_seconds:.3f}s instead of about 0.5s"


@pytest.fixture(scope="module")
def delayed_audio_video(tmp_path_factory) -> Path:
    """A container whose soundtrack starts a second after its picture."""
    if not _HAS_FFMPEG:
        pytest.skip("ffmpeg is not available")
    pytest.importorskip("av")
    path = tmp_path_factory.mktemp("h3_stream_delayed") / "reference.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=512x288:rate=30:duration=3.0",
            "-itsoffset", "1.0", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2.0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-muxdelay", "0", "-muxpreload", "0",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def test_a_soundtrack_that_starts_late_stays_late(delayed_audio_video):
    """Leaving a delayed soundtrack alone does not preserve the delay — it deletes it.

    Nothing downstream carries an offset: the waveform goes straight into
    ``encode_waveform``, where sample 0 *is* time 0 of the reference. So a
    soundtrack that begins a second after the picture, handed over unpadded,
    is conditioned on a second early. The leading silence is not invented — it
    is what the container has in that second.
    """
    import torch
    from vllm_omni.diffusion.models.minimax_h3.reference_video import prepare_reference_videos_official

    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import open_reference_video

    with open_reference_video(str(delayed_audio_video)) as reader:
        video_at, _ = next(reader.iter_frames())
        soundtrack = reader.soundtrack()
    delay = soundtrack.start_seconds - video_at
    assert delay > 0.5, f"the fixture was expected to delay its audio, got {delay:.3f}s"

    prepared = prepare_reference_videos_official(
        [str(delayed_audio_video)], target_frame_count=49, audio_sample_rate=16000
    )[0]
    audio = prepared["audio"]

    # The first `delay` seconds are silent, and the tone has not been pulled
    # forward into them.
    silent = int(delay * 16000 * 0.9)
    assert torch.count_nonzero(audio[:, :silent]) == 0, "the delayed tone was pulled forward"
    assert torch.count_nonzero(audio) > 0, "the tone is missing entirely"


# --------------------------------------------------------------------------
# Variable frame rate: where a frame index stops being a timestamp
# --------------------------------------------------------------------------


def test_a_variable_rate_source_resamples_like_its_constant_rate_expansion():
    """The oracle is the same stream written out at a constant rate.

    A container's ``average_rate`` is exactly that. This source runs 30 fps for
    its first half second and 10 fps afterwards, averaging 20 — so an index-based
    schedule believes every frame lasts 50 ms and puts the tail frames in slots
    they are nowhere near.

    Rather than restate the boundary arithmetic here (which would only record a
    rounding convention twice), the expectation is produced by expanding the very
    same content to a real 120 fps constant-rate stream — every VFR frame
    repeated for as long as it is on screen — and resampling *that*. 120 is a
    common multiple of 30, 10 and 24, so the expansion is exact rather than a
    second approximation, and the constant-rate path it goes through is the one
    already pinned byte-for-byte against the buffered decode above.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import normalize_reference_video_stream

    frames = _source_frames(20, height=48, width=64)
    times = [index / 30.0 for index in range(15)] + [0.5 + index / 10.0 for index in range(5)]
    average = 20.0  # what the container reports, and what neither half runs at
    kwargs = dict(
        num_frames=None,
        canvas_multiple=32,
        canvas_short_edge=48,
        canvas_max_pixels=768 * 1344,
        resolve_canvas=_identity_canvas,
    )

    streamed = normalize_reference_video_stream(zip(times, frames, strict=True), fps=average, **kwargs)

    # The same content at 120 fps: source frame j held until frame j+1 appears,
    # and the last one for the nominal interval the normalizer gives it.
    ends = times[1:] + [times[-1] + 1.0 / average]
    expanded = [
        frames[index]
        for index, (start, end) in enumerate(zip(times, ends, strict=True))
        for _ in range(round(end * 120) - round(start * 120))
    ]
    expected = normalize_reference_video_stream(_timed(expanded, 120.0), fps=120.0, **kwargs)

    assert np.array_equal(streamed, expected)
    # And the test discriminates: reading the schedule off frame indices at the
    # average rate is what used to happen, and it is not this answer.
    by_index = normalize_reference_video_stream(_timed(frames, average), fps=average, **kwargs)
    assert not np.array_equal(streamed, by_index), "the variable rate made no difference — check the fixture"


def test_a_constant_rate_source_resamples_to_the_same_bytes_as_before():
    """The index-based schedule, restated over timestamps, must not move a pixel.

    ``t_i = i/fps`` turns ``floor(t_next * target + 0.5)`` into
    ``floor((i+1) * target/fps + 0.5)`` — the old expression, character for
    character — and that is the whole argument that the parity fixtures still
    hold. Asserted against ``resample_frame_indices``, which is the contract's
    own statement of the mapping rather than this module's.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import (
        normalize_reference_video_stream,
        resample_frame_indices,
    )

    for fps in _SOURCE_RATES:
        frames = _source_frames(120, height=48, width=64)
        streamed = normalize_reference_video_stream(
            _timed(frames, fps),
            fps=fps,
            num_frames=None,
            canvas_multiple=32,
            canvas_short_edge=48,
            canvas_max_pixels=768 * 1344,
            resolve_canvas=_identity_canvas,
        )
        assert [int(frame[-1, -1, 0]) for frame in streamed] == list(resample_frame_indices(120, fps)), f"{fps} fps"


def test_the_reader_seeks_a_variable_rate_stream_by_timestamp():
    """``start_seconds`` is a time, and on a VFR stream a count cannot express it."""
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import ReferenceVideoReader

    frames = _source_frames(8, height=4, width=8)
    # 6 frames in the first 0.2 s, then one every 0.4 s: average_rate ~ 8 fps.
    times = [0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.6, 1.0]
    container, _ = _fake_container(frames, rate=8.0, times=times)

    read = list(ReferenceVideoReader(object(), container).iter_frames(start_seconds=0.7))

    # At 0.7 s frame 6 (t=0.6) is on screen; an `int(0.7 * 8) = 5` frame count
    # would have started at frame 5, which left the screen half a second before.
    assert [stamp for stamp, _ in read] == pytest.approx([0.6, 1.0])
    assert int(read[0][1][-1, -1, 0]) == 6


def test_the_first_frame_carries_the_time_the_soundtrack_is_cut_at():
    """Video and audio are cut at one instant, and it is the video's.

    ``start_seconds`` lands inside a frame's display interval; the reader hands
    over the frame covering it, whose own timestamp is at or before the request.
    Slicing the soundtrack at the *requested* second instead would offset the two
    conditions by that remainder — bounded by a frame on a constant-rate source,
    unbounded on a variable-rate one. Exposing the timestamp on the first frame
    is what lets the caller cut both at the same place.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import ReferenceVideoReader

    frames = _source_frames(8, height=4, width=8)
    times = [0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.6, 1.0]
    container, _ = _fake_container(frames, rate=8.0, times=times)

    first_at, _frame = next(ReferenceVideoReader(object(), container).iter_frames(start_seconds=0.85))

    assert first_at == pytest.approx(0.6), "the caller cannot align the soundtrack without the real start"


def test_a_stream_without_timestamps_still_behaves_as_a_counted_one():
    """PTS-less frames fall back to the nominal rate rather than collapsing.

    Nothing admitted should reach this, but a frame whose ``time`` is ``None``
    must not be read as ``t = 0`` — that would give every frame the same instant
    and hand the whole reference one output slot.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import ReferenceVideoReader

    frames = _source_frames(6, height=4, width=8)
    container, _ = _fake_container(frames, rate=25.0, times=[None] * 6)

    read = list(ReferenceVideoReader(object(), container).iter_frames(start_seconds=2 / 25.0))

    assert [stamp for stamp, _ in read] == pytest.approx([index / 25.0 for index in range(2, 6)])


# --------------------------------------------------------------------------
# The gates in front of all of it — and the one that must not come back
# --------------------------------------------------------------------------


def _admissible_meta(width, height, fps, duration):
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "frame_count": int(duration * fps),
        "format_names": ("mov", "mp4", "m4a"),
        "video_codec": "h264",
        "audio_codecs": ("aac",),
        "file_size": 40 * 1024 * 1024,
    }


@pytest.mark.parametrize(
    "width,height,fps",
    [
        (3840, 2160, 60.0),  # ordinary 4K/60
        (3840, 2160, 50.0),
        (5760, 2304, 60.0),  # every ceiling in this validator, taken at once
    ],
)
def test_a_reference_at_the_advertised_ceilings_is_admitted(width, height, fps):
    """No gate on *decoded* pixels: a validator must not refuse what it advertises.

    Dimensions up to 5760, rate up to 60 and duration up to 15 s are each stated
    as admissible a few lines above, and 5760 x 60 fps x 15 s is 6.7G pixels. A
    decoded-pixel budget was briefly added here and refused exactly these; it was
    removed because it narrowed *both* contracts — including legacy, which never
    buffered anything, it shells out to ffmpeg — against a buffering failure mode
    that `normalize_reference_video_stream` had already made impossible.

    Parametrized on the three shapes that budget rejected, so restoring it in any
    form turns this red rather than passing quietly on the 1080p case.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_video import _validate_reference_video_metadata

    _validate_reference_video_metadata(_admissible_meta(width, height, fps, 15.0), index=0, source="big.mp4")


def test_the_compressed_size_gate_is_the_one_that_stayed():
    """Removing the decoded budget must not have taken the 50 MiB gate with it."""
    from vllm_omni.diffusion.models.minimax_h3.reference_video import _validate_reference_video_metadata

    from vllm_omni.errors import OmniClientError

    meta = _admissible_meta(1920, 1080, 30.0, 15.0)
    _validate_reference_video_metadata(meta, index=0, source="ordinary.mp4")

    meta["file_size"] = 60 * 1024 * 1024
    with pytest.raises(OmniClientError, match="50 MiB"):
        _validate_reference_video_metadata(meta, index=0, source="fat.mp4")
