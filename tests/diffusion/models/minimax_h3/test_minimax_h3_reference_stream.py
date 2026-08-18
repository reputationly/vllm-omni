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
    return normalize_reference_video_frames(frames, **kwargs), normalize_reference_video_stream(iter(frames), **kwargs)


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
            iter(frames),
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
        for frame in _source_frames(900, height=48, width=64):
            pulled += 1
            yield frame

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
    # record a rounding convention twice.
    enough = next(count for count in range(1, 901) if math.floor(count * 24.0 / 60.0 + 0.5) >= 121)
    assert streamed.shape[0] == 121
    assert pulled == enough < 900, f"decoded {pulled} source frames for 121 output frames"


def test_only_one_source_frame_is_alive_at_a_time():
    """The property the whole change exists for, asserted rather than argued.

    CPython frees a frame as soon as the normalizer stops referring to it, so a
    weak reference taken at ``yield`` time is dead a frame or two later. The
    bound is one — the ``for`` target still holds the previous value while
    ``next()`` runs — and not "a few": if a future edit collects the source
    frames into a list, which is the shape of the original bug, every earlier
    weakref stays alive and the count climbs with the stream.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import normalize_reference_video_stream

    handed_out: list[weakref.ref] = []
    still_alive: list[int] = []

    def watched():
        for frame in _source_frames(120, height=90, width=160):
            frame = frame.copy()  # a fresh array per frame, as a decoder gives
            still_alive.append(sum(reference() is not None for reference in handed_out))
            handed_out.append(weakref.ref(frame))
            yield frame
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


def _fake_container(frames: np.ndarray, *, rotation: float = 0.0, rate: float = 30.0, with_audio: bool = False):
    """Just enough of a PyAV container for the reader, minus the codec."""

    class _Frame:
        def __init__(self, array):
            self._array = array
            self.rotation = rotation

        def to_ndarray(self, format):
            assert format == "rgb24"
            return self._array

    stream = SimpleNamespace(average_rate=rate, guessed_rate=rate)
    container = SimpleNamespace(
        streams=SimpleNamespace(video=[stream], audio=[object()] if with_audio else []),
        decode=lambda stream: (_Frame(frame) for frame in frames),
        seek=lambda offset: None,
    )
    return container, stream


def test_the_reader_skips_leading_frames_and_uprights_them():
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import ReferenceVideoReader

    frames = _source_frames(6, height=4, width=8)
    container, _ = _fake_container(frames, rotation=90.0, rate=25.0)

    reader = ReferenceVideoReader(object(), container)

    assert reader.fps == pytest.approx(25.0)
    read = list(reader.iter_frames(skip=2))
    assert len(read) == 4
    # Rotated per frame, which is the same quarter turn the buffered path
    # applies to the whole stack.
    assert read[0].shape == (8, 4, 3)
    assert np.array_equal(np.stack(read), np.rot90(frames[2:], k=-1, axes=(1, 2)))


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
    """Same frames, same rate, same soundtrack — read a frame at a time."""
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import (
        decode_reference_video,
        open_reference_video,
    )

    buffered = decode_reference_video(str(sample_video))

    with open_reference_video(str(sample_video)) as reader:
        assert reader.fps == pytest.approx(buffered.fps)
        streamed = np.stack(list(reader.iter_frames(skip=skip)))
        soundtrack = reader.soundtrack()

    assert np.array_equal(streamed, buffered.frames[skip:]), "streamed frames differ from the buffered decode"
    assert soundtrack is not None
    waveform, sample_rate = soundtrack
    assert sample_rate == buffered.sample_rate
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
        waveform, sample_rate = reader.soundtrack()

    assert sample_rate == buffered.sample_rate
    assert np.array_equal(waveform.numpy(), buffered.audio.numpy())


def test_the_prepared_reference_is_unchanged_by_the_streaming_switch(admissible_video):
    """``prepare_reference_videos_official`` end to end, against the old path.

    The buffered composition it used to run is reproduced here explicitly, so
    this fails if the switch changed the reference the pipeline conditions on —
    frames, canvas, duration or soundtrack.
    """
    import torch

    from vllm_omni.diffusion.models.minimax_h3.reference_audio import normalize_reference_audio
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import decode_reference_video
    from vllm_omni.diffusion.models.minimax_h3.reference_video import (
        MINIMAX_H3_BASE_SHORT_EDGE,
        MINIMAX_H3_CANVAS_MULTIPLE,
        MINIMAX_H3_MAX_PIXELS,
        _resolve_reference_canvas,
        prepare_reference_videos_official,
    )
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


# --------------------------------------------------------------------------
# The gate in front of all of it
# --------------------------------------------------------------------------


def test_admission_bounds_the_decoded_pixels_not_just_the_file_size():
    """The 50 MiB gate says nothing about what a file expands to.

    5760x2304 at 60 fps for 15 s is inside every other limit — dimensions,
    ratio, rate, duration — and compresses well under 50 MiB, yet decodes to
    ~36 GiB of RGB. Refused up front, with the two knobs that fix it named.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_video import (
        MINIMAX_H3_MAX_DECODED_PIXELS,
        _validate_reference_video_metadata,
    )
    from vllm_omni.errors import OmniClientError

    def meta(width, height, fps, duration):
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

    with pytest.raises(OmniClientError, match="lower resolution or frame rate"):
        _validate_reference_video_metadata(meta(5760, 2304, 60.0, 15.0), index=0, source="huge.mp4")

    # The ordinary 1080p reference the limit must not touch.
    _validate_reference_video_metadata(meta(1920, 1080, 30.0, 15.0), index=0, source="ordinary.mp4")
    assert 1920 * 1080 * 30 * 15 < MINIMAX_H3_MAX_DECODED_PIXELS
