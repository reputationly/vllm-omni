# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A 5.1 soundtrack in a reference video, which every other leg already handles.

The official reference-video path demuxes the soundtrack itself and keeps the
source layout, so a 6-channel AAC track arrives as ``(6, T)``. The legacy path
never saw that shape because it shelled out to ffmpeg with ``-ac 2``, and
``vae.encode_waveform`` never sees it either because it takes ``[:2]`` of
whatever it is given. The normalizer between them was the one leg that rejected
it — a request that used to work, refused, on media the model can encode.

The narrowing rule is the one already used twice (``encode_waveform`` and the
standalone-audio normalizer): keep the first two channels. Not an ffmpeg-style
downmix matrix, which would be a third rule and would make the *stereo* result
differ from what the VAE would have produced from the same source.

Officially this is a deliberate superset: the oracle raises here too. It is safe
to be wider because the widening is a no-op on every input official accepts —
the same shape as the §2.8 duration ceiling.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _ramp(channels: int, num_samples: int = 640) -> torch.Tensor:
    """Channels that are trivially distinguishable, so a mix-up cannot pass."""
    base = torch.linspace(-1.0, 1.0, num_samples, dtype=torch.float32)
    return torch.stack([base + index for index in range(channels)])


@pytest.mark.parametrize("channels", [3, 6, 8])
def test_a_wider_than_stereo_soundtrack_is_accepted(channels):
    from vllm_omni.diffusion.models.minimax_h3.reference_audio import normalize_reference_audio

    waveform = _ramp(channels)
    produced = normalize_reference_audio(waveform, 32000, target_sample_rate=32000, max_duration=None)

    assert produced.shape == (2, 640)


def test_the_kept_channels_are_the_first_two_the_vae_would_have_kept():
    """The same rule as `encode_waveform`, checked against it rather than restated."""
    from vllm_omni.diffusion.models.minimax_h3.reference_audio import normalize_reference_audio

    waveform = _ramp(6)
    produced = normalize_reference_audio(waveform, 32000, target_sample_rate=32000, max_duration=None)

    assert torch.equal(produced, waveform[:2])


def test_narrowing_happens_before_truncation_and_resampling_change_nothing():
    """Selecting channels commutes with both, so a 5.1 track and the stereo one
    ffmpeg would have handed the legacy path produce the identical tensor."""
    from vllm_omni.diffusion.models.minimax_h3.reference_audio import normalize_reference_audio

    surround = _ramp(6, num_samples=44100 // 2)
    stereo = surround[:2].contiguous()

    from_surround = normalize_reference_audio(surround, 44100, target_sample_rate=32000, max_duration=0.25)
    from_stereo = normalize_reference_audio(stereo, 44100, target_sample_rate=32000, max_duration=0.25)

    assert torch.equal(from_surround, from_stereo)


def test_mono_and_stereo_are_untouched_by_the_widening():
    """The only shapes that ever reached here before must be bit-identical."""
    from vllm_omni.diffusion.models.minimax_h3.reference_audio import normalize_reference_audio

    mono = _ramp(1)
    stereo = _ramp(2)

    upmixed = normalize_reference_audio(mono, 32000, target_sample_rate=32000, max_duration=None)
    assert upmixed.shape == (2, 640)
    assert torch.equal(upmixed[0], mono[0]) and torch.equal(upmixed[1], mono[0])

    passed = normalize_reference_audio(stereo, 32000, target_sample_rate=32000, max_duration=None)
    assert torch.equal(passed, stereo)


def test_the_shapes_that_are_still_refused_are_the_ones_with_no_reading():
    """Widening the channel rule must not turn the guard into no guard at all."""
    from vllm_omni.diffusion.models.minimax_h3.reference_audio import normalize_reference_audio

    for bad in (torch.zeros(640), torch.zeros(0, 640), torch.zeros(1, 2, 640)):
        with pytest.raises(ValueError, match="channels, num_samples"):
            normalize_reference_audio(bad, 32000, target_sample_rate=32000, max_duration=None)


def test_a_standalone_audio_upload_narrows_by_the_same_rule():
    """It used to narrow on its own; now it delegates, so the two cannot drift."""
    from vllm_omni.diffusion.models.minimax_h3.reference_audio import (
        normalize_reference_audio,
        normalize_standalone_reference_audios,
    )

    waveform = _ramp(6)
    normalized = normalize_standalone_reference_audios([(waveform, 32000)], target_sample_rate=32000, max_duration=None)
    standalone, rate = normalized[0]
    direct = normalize_reference_audio(waveform, 32000, target_sample_rate=32000, max_duration=None)

    assert rate == 32000
    assert torch.equal(standalone, direct)


def test_the_demuxer_is_what_hands_over_the_wide_shape():
    """Pins the premise. The resampler is built with the *source* layout, so a
    5.1 track stays 5.1 and the normalizer is genuinely the leg that had to
    accept it. If this ever became a fixed stereo layout the widening above
    would be unreachable — still correct, but no longer load-bearing."""
    import inspect

    from vllm_omni.diffusion.models.minimax_h3 import reference_media_decode

    source = inspect.getsource(reference_media_decode._decode_soundtrack)
    assert "layout=stream.layout" in source
