# SPDX-License-Identifier: Apache-2.0
"""Normalizing a `ref2va` reference soundtrack onto the audio VAE's rate.

The official recipe is three steps in a fixed order, and the order is what makes
it lossless-ish:

1. truncate **at the source sample rate**, to the generated duration;
2. upmix mono to stereo by repeating the channel;
3. resample **once**, to the audio VAE's 32 kHz.

vLLM-Omni's legacy path instead lets ffmpeg force every soundtrack to 44.1 kHz
while extracting it, and the audio VAE then resamples 44.1 kHz to 32 kHz — two
rate conversions where the reference implementation does one, and none of them
at the source rate. It also never truncates to the generated duration, which
changes the *number of packed audio rows*, i.e. a discrete contract value rather
than a numeric one.

Truncating before resampling is not interchangeable with truncating after: the
resampler's filter sees a different tail, so the last samples differ even when
the sample counts agree.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

MINIMAX_H3_AUDIO_RESAMPLE_OFFICIAL = "official_single_resample"
MINIMAX_H3_AUDIO_RESAMPLE_LEGACY = "legacy_double_resample"
MINIMAX_H3_AUDIO_RESAMPLE_MODES = frozenset({MINIMAX_H3_AUDIO_RESAMPLE_OFFICIAL, MINIMAX_H3_AUDIO_RESAMPLE_LEGACY})

MINIMAX_H3_AUDIO_CHANNELS = 2

# The rate ffmpeg forces on an extracted soundtrack in the legacy path, before
# the audio VAE resamples it a second time to its own rate.
MINIMAX_H3_LEGACY_INTERMEDIATE_SAMPLE_RATE = 44100


def normalize_reference_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    target_sample_rate: int,
    max_duration: float | None,
    resample_mode: str = MINIMAX_H3_AUDIO_RESAMPLE_OFFICIAL,
) -> torch.Tensor:
    """The official normalization: truncate at source rate, upmix, resample once.

    Args:
        waveform: ``(channels, num_samples)``. Mono is upmixed by repeating the
            channel; anything wider than stereo keeps its first two channels,
            as ``vae.encode_waveform`` and
            ``normalize_standalone_reference_audios`` already do.
        sample_rate: The rate ``waveform`` carries.
        target_sample_rate: The audio VAE's rate, 32000 for the released model.
        max_duration: Seconds to truncate to — ``num_frames / fps`` for a
            request. ``None`` keeps the whole clip, which is the legacy
            behaviour and is *not* the official contract.
        resample_mode: ``official_single_resample`` converts straight to
            ``target_sample_rate``. ``legacy_double_resample`` reproduces the
            legacy chain — source to 44.1 kHz, then 44.1 kHz to the target —
            so a parity run can attribute a difference to the extra conversion
            rather than to the truncation it used to be bundled with.

    Returns:
        ``(2, num_samples)`` float32.
    """
    if resample_mode not in MINIMAX_H3_AUDIO_RESAMPLE_MODES:
        raise ValueError(
            f"resample_mode must be one of {sorted(MINIMAX_H3_AUDIO_RESAMPLE_MODES)}, got {resample_mode!r}"
        )
    if waveform.ndim != 2 or waveform.shape[0] < 1:
        raise ValueError(
            f"a reference soundtrack must be a (channels, num_samples) waveform, got {tuple(waveform.shape)}"
        )
    if int(sample_rate) <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    if waveform.shape[0] > MINIMAX_H3_AUDIO_CHANNELS:
        # 5.1 is ordinary in an MP4 and the demuxer hands the soundtrack over
        # with its layout intact, so this is a shape the decoder really produces.
        # Rejecting it here would be a regression against the legacy path, which
        # accepted the same media because ffmpeg's `-ac 2` had already mixed it
        # down. Keep the first two channels rather than downmix: it is what
        # `vae.encode_waveform` does to the very same tensor a few steps later,
        # and what the standalone-audio normalizer beside this one does, so the
        # three agree on one rule instead of three. Before the float32 cast so a
        # 6-channel track is never materialized at three times the width.
        waveform = waveform[:MINIMAX_H3_AUDIO_CHANNELS]
    waveform = waveform.to(torch.float32)
    if max_duration is not None:
        # At the source rate, before any resampling — see the module docstring.
        waveform = waveform[:, : int(max_duration * sample_rate)]
    if waveform.shape[0] != MINIMAX_H3_AUDIO_CHANNELS:
        waveform = waveform.expand(MINIMAX_H3_AUDIO_CHANNELS, -1).contiguous()

    rates = [int(target_sample_rate)]
    if resample_mode == MINIMAX_H3_AUDIO_RESAMPLE_LEGACY:
        rates.insert(0, MINIMAX_H3_LEGACY_INTERMEDIATE_SAMPLE_RATE)

    current = int(sample_rate)
    for rate in rates:
        if current == rate:
            continue
        try:
            import torchaudio
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ImportError(
                f"resampling a MiniMax-H3 reference soundtrack from {current} Hz to {rate} Hz needs torchaudio."
            ) from error
        waveform = torchaudio.transforms.Resample(current, rate)(waveform)
        current = rate
    return waveform


def normalize_standalone_reference_audios(
    audios: Sequence[tuple[torch.Tensor, int]],
    *,
    target_sample_rate: int,
    max_duration: float | None,
    resample_mode: str = MINIMAX_H3_AUDIO_RESAMPLE_OFFICIAL,
) -> list[tuple[torch.Tensor, int]]:
    """The same normalization, for audio references that arrive on their own.

    A soundtrack pulled out of a reference video and an audio file uploaded
    beside it are the same kind of condition and are packed the same way, so
    they have to be normalized the same way. Truncation in particular is not
    cosmetic: an untruncated clip contributes more packed audio rows, which is
    a discrete contract value.

    Channel handling mirrors ``vae.encode_waveform`` so this normalization
    never rejects an input that used to be accepted: a 1-D waveform becomes
    mono, and anything wider than stereo keeps its first two channels — the
    latter in ``normalize_reference_audio`` itself, so a soundtrack pulled out
    of a video and an uploaded file are narrowed by one rule and not two.

    Args:
        audios: ``(waveform, sample_rate)`` pairs as loaded from the request.
        target_sample_rate: The audio VAE's rate.
        max_duration: Seconds to truncate to, or ``None`` to keep the clip.
        resample_mode: As in ``normalize_reference_audio``.

    Returns:
        ``(waveform[2, T], target_sample_rate)`` pairs, in the input order.
    """
    normalized: list[tuple[torch.Tensor, int]] = []
    for waveform, sample_rate in audios:
        tensor = torch.as_tensor(waveform)
        if tensor.ndim == 1:
            tensor = tensor[None]
        if tensor.ndim != 2:
            raise ValueError(f"a reference audio must be a waveform of at most 2 dimensions, got {tuple(tensor.shape)}")
        normalized.append(
            (
                normalize_reference_audio(
                    tensor,
                    int(sample_rate),
                    target_sample_rate=target_sample_rate,
                    max_duration=max_duration,
                    resample_mode=resample_mode,
                ),
                int(target_sample_rate),
            )
        )
    return normalized


def reference_audio_max_duration(num_frames: int, fps: float) -> float:
    """The duration a reference soundtrack is truncated to: the generated one."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return num_frames / fps


__all__ = [
    "MINIMAX_H3_AUDIO_CHANNELS",
    "MINIMAX_H3_AUDIO_RESAMPLE_LEGACY",
    "MINIMAX_H3_AUDIO_RESAMPLE_MODES",
    "MINIMAX_H3_AUDIO_RESAMPLE_OFFICIAL",
    "normalize_reference_audio",
    "normalize_standalone_reference_audios",
    "reference_audio_max_duration",
]
