# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for the audio output-container capability probe.

``response_format`` used to be trusted blindly: on an image whose libsndfile lacks
MPEG support, ``response_format=mp3`` generated the entire waveform and only then
failed with a 500. These tests pin the replacement contract:

1. wav/pcm are always available (this repo writes them);
2. mp3/flac/opus appear only when libsndfile reports the matching encoder;
3. an adapter may narrow the set but can never widen it past the image;
4. an unproducible container yields a readable 400-style message, not a crash.
"""

import pytest

from vllm_omni.entrypoints.openai import audio_output_formats as aof
from vllm_omni.entrypoints.openai.protocol.audio import SUPPORTED_AUDIO_FORMATS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ALL_CODECS = {"WAV", "MP3", "FLAC", "OGG"}


class _FullAdapter:
    supported_response_formats = None


class _WavOnlyAdapter:
    supported_response_formats = frozenset({"wav"})


class _WavMp3Adapter:
    supported_response_formats = frozenset({"wav", "mp3"})


@pytest.fixture(autouse=True)
def _fresh_probe():
    aof.available_output_formats.cache_clear()
    yield
    aof.available_output_formats.cache_clear()


def _probe(monkeypatch, codecs: set[str]) -> None:
    monkeypatch.setattr(aof, "_codec_containers", lambda: set(codecs))


def test_wav_and_pcm_never_depend_on_a_codec(monkeypatch):
    _probe(monkeypatch, set())
    assert aof.available_output_formats() == frozenset({"wav", "pcm"})


def test_every_container_available_with_a_full_libsndfile(monkeypatch):
    _probe(monkeypatch, ALL_CODECS)
    assert aof.available_output_formats() == SUPPORTED_AUDIO_FORMATS


def test_missing_mp3_encoder_only_drops_mp3(monkeypatch):
    _probe(monkeypatch, {"WAV", "FLAC", "OGG"})
    assert "mp3" not in aof.available_output_formats()
    assert {"wav", "flac", "opus", "pcm"} <= aof.available_output_formats()


def test_adapter_narrowing_is_honoured(monkeypatch):
    _probe(monkeypatch, ALL_CODECS)
    assert aof.allowed_output_formats(_WavOnlyAdapter()) == frozenset({"wav"})
    assert aof.allowed_output_formats(_WavMp3Adapter()) == frozenset({"wav", "mp3"})
    assert aof.allowed_output_formats(_FullAdapter()) == SUPPORTED_AUDIO_FORMATS


def test_adapter_cannot_widen_past_the_image(monkeypatch):
    # Declaring mp3 does not conjure an encoder: the image still wins.
    _probe(monkeypatch, {"WAV"})
    assert aof.allowed_output_formats(_WavMp3Adapter()) == frozenset({"wav"})


def test_supported_format_passes(monkeypatch):
    _probe(monkeypatch, ALL_CODECS)
    for fmt in sorted(SUPPORTED_AUDIO_FORMATS):
        assert aof.check_output_format(fmt, _FullAdapter()) is None


def test_unsupported_format_reports_what_is_available(monkeypatch):
    _probe(monkeypatch, {"WAV"})
    message = aof.check_output_format("mp3", _FullAdapter())
    assert message is not None
    assert "mp3" in message
    # 文案必须列出真正可用的容器,否则客户端只能靠猜。
    assert "wav" in message and "pcm" in message


def test_unknown_container_is_rejected(monkeypatch):
    _probe(monkeypatch, ALL_CODECS)
    assert aof.check_output_format("aac", _FullAdapter()) is not None


def test_empty_format_defers_to_the_request_default(monkeypatch):
    _probe(monkeypatch, ALL_CODECS)
    assert aof.check_output_format("", _FullAdapter()) is None


def test_case_and_whitespace_tolerated(monkeypatch):
    _probe(monkeypatch, ALL_CODECS)
    assert aof.check_output_format("  MP3 ", _FullAdapter()) is None
