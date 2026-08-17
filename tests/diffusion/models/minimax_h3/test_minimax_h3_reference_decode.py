# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless decoding of a `ref2va` reference, against the official decoder.

Unlike the rest of the contract suite these need a codec, so they synthesize a
small container with ffmpeg and skip cleanly where it or PyAV is missing. The
strongest assertion — decoding the same file with the pinned Diffusers oracle
and comparing frame for frame — is opt-in via ``H3_DIFFUSERS_SRC``, because the
ordinary run must not depend on a checkout being present.

No weights, no GPU.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _synthesize(path: Path, *, fps: int, seconds: float, width: int, height: int, with_audio: bool) -> None:
    """A deterministic test container: a moving pattern plus an optional tone."""
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
    path = tmp_path_factory.mktemp("h3_ref") / "reference.mp4"
    _synthesize(path, fps=30, seconds=1.0, width=64, height=48, with_audio=True)
    return path


def test_decode_reports_the_container_rate_and_frames(sample_video):
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import decode_reference_video

    decoded = decode_reference_video(str(sample_video))

    assert decoded.fps == pytest.approx(30.0)
    assert decoded.frames.dtype == np.uint8
    assert decoded.frames.shape[1:] == (48, 64, 3)
    assert decoded.frames.shape[0] == 30
    # The soundtrack comes back at the container's own rate, not the VAE's:
    # resampling it here would be the second conversion the official path avoids.
    assert decoded.sample_rate == 44100
    assert decoded.audio is not None and decoded.audio.shape[0] in (1, 2)


def test_decode_can_skip_the_soundtrack_pass(sample_video):
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import decode_reference_video

    decoded = decode_reference_video(str(sample_video), with_audio=False)
    assert decoded.audio is None and decoded.sample_rate is None


def test_display_rotation_is_undone_by_quarter_turns():
    """Pure: no container needs to carry rotation metadata for this."""
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import apply_display_rotation

    frames = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)

    assert np.array_equal(apply_display_rotation(frames, 0.0), frames)
    assert apply_display_rotation(frames, 90.0).shape == (2, 4, 3, 3)
    assert apply_display_rotation(frames, 180.0).shape == (2, 3, 4, 3)
    # ffmpeg undoes a counterclockwise display rotation, so 90 and -270 agree.
    assert np.array_equal(apply_display_rotation(frames, 90.0), apply_display_rotation(frames, -270.0))
    # Snapped to the nearest quarter turn.
    assert np.array_equal(apply_display_rotation(frames, 89.0), apply_display_rotation(frames, 90.0))


def test_decode_is_lossless_relative_to_a_reencode(sample_video, tmp_path):
    """The legacy H.264 intermediate is not a no-op, which is why it has to go.

    Asserted rather than argued: re-encoding the very same frames and decoding
    them back changes pixels, so any conditioning built on the intermediate is
    built on different pixels than the source carries.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import decode_reference_video

    direct = decode_reference_video(str(sample_video), with_audio=False).frames
    intermediate = tmp_path / "intermediate.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(sample_video),
            "-an",
            "-vf",
            "fps=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(intermediate),
        ],
        check=True,
    )
    round_tripped = decode_reference_video(str(intermediate), with_audio=False).frames

    assert direct.shape == round_tripped.shape
    assert not np.array_equal(direct, round_tripped)


_ORACLE_DECODE_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from diffusers.modular_pipelines.minimax_h3.references import _decode_video_file

frames, fps, audio, rate = _decode_video_file(sys.argv[2])
np.save(sys.argv[3] + "/frames.npy", frames)
np.save(sys.argv[3] + "/audio.npy", audio.numpy() if audio is not None else np.zeros(0))
json.dump({"fps": float(fps), "sample_rate": None if rate is None else int(rate)},
          open(sys.argv[3] + "/meta.json", "w"))
"""


@pytest.mark.skipif(not os.environ.get("H3_DIFFUSERS_SRC"), reason="set H3_DIFFUSERS_SRC for oracle parity")
def test_decode_matches_the_official_decoder(sample_video, tmp_path):
    """Frame-for-frame against the pinned oracle's own decoder.

    The oracle runs in a subprocess and writes its result out, for two reasons:
    ``diffusers`` is usually already imported by the time a test runs, so
    prepending the checkout to ``sys.path`` would silently have no effect; and
    the checkout is a newer Diffusers than the one vLLM-Omni is built against,
    which has no business shadowing it inside the test process.
    """
    import json

    from vllm_omni.diffusion.models.minimax_h3.reference_media_decode import decode_reference_video

    subprocess.run(
        [sys.executable, "-c", _ORACLE_DECODE_SCRIPT, os.environ["H3_DIFFUSERS_SRC"], str(sample_video), str(tmp_path)],
        check=True,
        capture_output=True,
    )
    official_frames = np.load(tmp_path / "frames.npy")
    official_audio = np.load(tmp_path / "audio.npy")
    official_meta = json.loads((tmp_path / "meta.json").read_text())

    ours = decode_reference_video(str(sample_video))

    assert ours.fps == official_meta["fps"]
    assert ours.frames.shape == official_frames.shape
    assert np.array_equal(ours.frames, official_frames), "decoded frames differ from the official decoder"
    assert ours.sample_rate == official_meta["sample_rate"]
    assert ours.audio is not None
    assert np.array_equal(ours.audio.numpy(), official_audio), "decoded soundtrack differs"
