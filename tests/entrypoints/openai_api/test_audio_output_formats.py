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

from types import SimpleNamespace

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


def _probe(monkeypatch, codecs: set[str], *, opus: bool = True) -> None:
    monkeypatch.setattr(aof, "_codec_containers", lambda: set(codecs))
    # OGG 容器存在不代表能编 Opus(libsndfile 的 Ogg 里可能是 Vorbis),所以探测要
    # 走 subtype;测试里单独控制,才能覆盖"有 OGG 没 OPUS"这种真实镜像。
    monkeypatch.setattr(aof, "_opus_subtype_available", lambda: opus)


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


def test_ogg_without_opus_subtype_does_not_advertise_opus(monkeypatch):
    """Containers are not codecs: OGG present ≠ Opus encodable.

    An OGG-capable libsndfile without the OPUS subtype used to advertise opus, and
    the request then failed inside the encoder — after the waveform was generated.
    """
    _probe(monkeypatch, ALL_CODECS, opus=False)
    available = aof.available_output_formats()
    assert "ogg" not in available  # never an output container of ours
    assert "opus" not in available
    assert {"wav", "pcm", "mp3", "flac"} <= available
    assert aof.check_output_format("opus", _FullAdapter()) is not None


def test_opus_is_capped_by_sample_rate(monkeypatch):
    """Opus only accepts 8/12/16/24/48 kHz; 22.05 kHz (IndexTTS2) must be refused up front."""
    _probe(monkeypatch, ALL_CODECS)

    for rate in (8000, 12000, 16000, 24000, 48000):
        assert aof.check_output_format("opus", _FullAdapter(), rate) is None

    message = aof.check_output_format("opus", _FullAdapter(), 22050)
    assert message is not None
    assert "22050" in message
    # 文案要告诉调用方能用什么采样率,否则只能靠试。
    assert "24000" in message


def test_mp3_accepts_the_speech_rates(monkeypatch):
    _probe(monkeypatch, ALL_CODECS)
    for rate in (16000, 22050, 24000, 44100, 48000):
        assert aof.check_output_format("mp3", _FullAdapter(), rate) is None
    assert aof.check_output_format("mp3", _FullAdapter(), 4000) is not None


def test_rate_check_is_skipped_when_the_rate_is_unknown(monkeypatch):
    _probe(monkeypatch, ALL_CODECS)
    # 不知道采样率时不要乱拒:交给编码器,而不是猜。
    assert aof.check_output_format("opus", _FullAdapter(), None) is None


# --------------------------------------------------------------------------- #
# Which sample rate the container check reasons about.
# --------------------------------------------------------------------------- #
def test_effective_output_sample_rate_prefers_the_request():
    from vllm_omni.entrypoints.openai.tts_adapters.base import TTSModelAdapter, effective_output_sample_rate

    class _Adapter:
        supported_output_sample_rates = frozenset({24000})
        native_output_sample_rate = TTSModelAdapter.native_output_sample_rate

    class _Request:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate

    assert effective_output_sample_rate(_Adapter(), _Request(48000)) == 48000
    # 没有显式请求时,只输出一种采样率的模型是可知的
    assert effective_output_sample_rate(_Adapter(), _Request(None)) == 24000

    class _ManyRates:
        supported_output_sample_rates = frozenset({8000, 24000})
        native_output_sample_rate = TTSModelAdapter.native_output_sample_rate

    # 多采样率模型无法在请求前确定 → 不猜,交给编码器
    assert effective_output_sample_rate(_ManyRates(), _Request(None)) is None


# --------------------------------------------------------------------------- #
# Serving a stored artifact must use the same media type as the sync endpoint.
# --------------------------------------------------------------------------- #
def test_artifact_media_types_cover_every_container():
    """Starlette has no entry for .pcm/.opus/.flac — a ``.pcm`` download used to
    arrive as ``application/octet-stream`` while the sync endpoint said ``audio/pcm``."""
    from vllm_omni.entrypoints.openai.audio_output_formats import artifact_media_type

    expected = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".opus": "audio/opus",
        ".pcm": "audio/pcm",
    }
    for suffix, media_type in expected.items():
        assert artifact_media_type("task-1" + suffix) == media_type
    # 大写后缀同样认
    assert artifact_media_type("task-1.PCM") == "audio/pcm"
    # 认不出就交给框架猜,不要瞎标
    assert artifact_media_type("task-1.bin") is None


def test_fixed_rate_models_declare_their_rate():
    """Opus cannot encode 44.1 kHz, so a fixed-rate decoder must say which rate it has.

    Otherwise the rate is "unknown", the container check is skipped, and
    ``response_format=opus`` only fails inside the encoder — after generation.
    """
    from vllm_omni.entrypoints.openai.tts_adapters.breeze_tts_2 import BreezeTTS2Adapter
    from vllm_omni.entrypoints.openai.tts_adapters.fish_omni import FishOmniAdapter
    from vllm_omni.entrypoints.openai.tts_adapters.fish_speech import FishSpeechAdapter
    from vllm_omni.entrypoints.openai.tts_adapters.indextts2 import IndexTTS2Adapter

    assert BreezeTTS2Adapter.supported_output_sample_rates == frozenset({24000})
    assert IndexTTS2Adapter.supported_output_sample_rates == frozenset({22050})
    assert FishSpeechAdapter.supported_output_sample_rates == frozenset({44100})
    assert FishOmniAdapter.supported_output_sample_rates == FishSpeechAdapter.supported_output_sample_rates


@pytest.mark.parametrize(("config", "rate"), [({}, 44100), ({"codec_sample_rate": 24000}, 24000)])
def test_audio8_uses_the_decoder_sample_rate(config, rate):
    from vllm_omni.entrypoints.openai.tts_adapters.audio8_tts import Audio8TTSAdapter
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext, effective_output_sample_rate

    engine = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**config)))
    adapter = Audio8TTSAdapter(SpeechServingContext(server=None, engine_client=engine))
    assert effective_output_sample_rate(adapter, SimpleNamespace(sample_rate=None)) == rate


def test_audio8_prefers_the_codec_stage_config():
    from vllm_omni.entrypoints.openai.tts_adapters.audio8_tts import Audio8TTSAdapter
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext

    slow_ar = SimpleNamespace(model_stage="audio8_tts_slow_ar", hf_config=SimpleNamespace(codec_sample_rate=44100))
    decoder = SimpleNamespace(
        model_stage="audio8_tts_codec_decoder", hf_config=SimpleNamespace(codec_sample_rate=24000)
    )
    engine = SimpleNamespace(stage_vllm_configs=[SimpleNamespace(model_config=c) for c in (slow_ar, decoder)])
    adapter = Audio8TTSAdapter(
        SpeechServingContext(server=None, engine_client=SimpleNamespace(engine=engine, model_config=slow_ar))
    )
    assert adapter.native_output_sample_rate() == 24000
