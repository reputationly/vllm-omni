# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many frames of a `ref2va` reference video reach the video VAE.

The VAE consumes ``17 * n + 5`` frames. A generated clip's own frame count
already has that form, and a reference *longer* than the clip is truncated to
it — so the only references that can arrive at some other count are the ones
*shorter* than the clip, which keep whatever the source ran to. Admission lets
them in from 2 seconds up, and 2 seconds at 24 fps is 48 frames, which is not
``17 * n + 5``.

The official encoder snaps such a reference down before the VAE
(``modular_pipelines/minimax_h3/encoders.py``, "Snap *down* to `17 * n + 5` so
the VAE encodes without padding"); handing all 48 frames over instead pads
inside the VAE and yields a different latent temporal extent, i.e. a different
packed row count. Legacy has always handed everything over and must keep doing
so, so this is a contract axis rather than a repair applied everywhere.

The frame arithmetic is a pure function; no weights, no GPU, no codec.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _reference_snap(num_frames: int, frames_per_chunk: int = 17, latents_per_chunk: int = 5) -> int:
    """The official expression, transcribed independently of the implementation."""
    return max(1, (num_frames - latents_per_chunk) // frames_per_chunk) * frames_per_chunk + latents_per_chunk


def test_snapped_counts_are_what_the_vae_consumes():
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import vae_chunk_frame_count

    for num_frames in range(1, 400):
        snapped = vae_chunk_frame_count(num_frames)
        assert snapped == _reference_snap(num_frames)
        assert (snapped - 5) % 17 == 0


def test_counts_that_already_fit_are_left_alone():
    """Which is why a full-length reference is untouched by the snap."""
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import vae_chunk_frame_count

    for n in range(1, 25):
        aligned = 17 * n + 5
        assert vae_chunk_frame_count(aligned) == aligned
    # The two shipped clip lengths, and the one an admitted 2 s reference hits.
    assert vae_chunk_frame_count(124) == 124
    assert vae_chunk_frame_count(209) == 209
    assert vae_chunk_frame_count(48) == 39


def test_the_snap_never_invents_frames_a_reference_does_not_have():
    """Above the admission floor it only ever removes frames."""
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import vae_chunk_frame_count

    # 2 s at 24 fps is the shortest reference admission accepts.
    for num_frames in range(48, 400):
        assert vae_chunk_frame_count(num_frames) <= num_frames


def test_below_the_admission_floor_the_expression_overshoots_and_slicing_absorbs_it():
    """Not a clamp to add: the reference slices, and so must this.

    ``max(1, ...)`` makes the count exceed the input below 22 frames. Clamping
    it would be a divergence dressed up as defensiveness — unreachable from a
    served request, but reachable from here, and the two must agree.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import vae_chunk_frame_count

    assert vae_chunk_frame_count(10) == 22
    frames = np.zeros((10, 4, 4, 3), dtype=np.uint8)
    assert frames[: vae_chunk_frame_count(len(frames))].shape[0] == 10


def test_a_non_positive_count_is_rejected():
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import vae_chunk_frame_count

    for bad in (0, -1):
        with pytest.raises(ValueError):
            vae_chunk_frame_count(bad)


class _Pipeline:
    """Only the attribute ``_reference_video_vae_frames`` reads."""

    def __init__(self, strategy):
        self.strategy = strategy


def _select(strategy, item):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    return MiniMaxH3Pipeline._reference_video_vae_frames(_Pipeline(strategy), item)


def test_official_snaps_a_short_reference_before_the_vae():
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    strategy = resolve_strategy(inference_contract="official_diffusers_v1", admission_policy=None, environ={})
    frames = np.arange(48 * 2 * 2 * 3, dtype=np.uint8).reshape(48, 2, 2, 3)

    selected = _select(strategy, {"frames": frames, "prepared_path": None})

    assert selected.shape[0] == 39
    # Snapped from the *end*: the reference keeps its opening frames.
    assert np.array_equal(selected, frames[:39])


def test_legacy_still_hands_every_decoded_frame_over():
    """Changing this would change what production generates today."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    strategy = resolve_strategy(inference_contract=None, admission_policy=None, environ={})
    frames = np.zeros((48, 2, 2, 3), dtype=np.uint8)

    assert _select(strategy, {"frames": frames, "prepared_path": None}).shape[0] == 48


def test_a_full_length_reference_is_identical_under_both_contracts():
    """The snap is invisible to everything the truncation already handled."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    frames = np.zeros((124, 2, 2, 3), dtype=np.uint8)
    item = {"frames": frames, "prepared_path": None}

    official = resolve_strategy(inference_contract="official_diffusers_v1", admission_policy=None, environ={})
    legacy = resolve_strategy(inference_contract=None, admission_policy=None, environ={})

    assert _select(official, item).shape[0] == _select(legacy, item).shape[0] == 124


def test_the_legacy_intermediate_file_is_still_the_source_when_there_are_no_frames(monkeypatch):
    """The legacy path reads its re-encoded intermediate, and reads it unsnapped."""
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    loaded = []

    def fake_load(path):
        loaded.append(path)
        return np.zeros((48, 2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(pipeline_minimax_h3, "load_video_frames", fake_load)
    strategy = resolve_strategy(inference_contract=None, admission_policy=None, environ={})

    selected = _select(strategy, {"frames": None, "prepared_path": "/tmp/prepared.mp4"})

    assert loaded == ["/tmp/prepared.mp4"]
    assert selected.shape[0] == 48
