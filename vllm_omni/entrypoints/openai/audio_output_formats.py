# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Which audio containers this server can actually produce.

``response_format`` has always been a request field, but nothing checked whether
the process could honour it: an image built against a libsndfile without MPEG
support would fail *mid-request* (500) when asked for mp3, long after the model
had already generated the audio. This module turns that into a declared,
startup-time capability:

* every container that does not depend on a codec (wav/pcm) is always available;
* mp3/flac/opus depend on ``libsndfile`` really shipping that encoder, so they are
  probed once through ``soundfile.available_formats()``;
* adapters may narrow the set further (``TTSModelAdapter.supported_response_formats``).

The gateway in front of this server (new-api) keeps its own list; it must stay a
superset of :data:`SUPPORTED_AUDIO_FORMATS` or requests die as pydantic 422s
before any of this code runs.
"""

from __future__ import annotations

import functools

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import SUPPORTED_AUDIO_FORMATS

logger = init_logger(__name__)

#: Containers that never need a probed encoder: wav is a header around PCM and
#: pcm IS the raw samples. Both are written by this repo, not by libsndfile.
_ALWAYS_AVAILABLE = frozenset({"wav", "pcm"})

#: Container -> the key that must appear in ``soundfile.available_formats()``.
_CODEC_PROBED = {
    "mp3": "MP3",
    "flac": "FLAC",
    "opus": "OGG",  # soundfile writes Opus into an Ogg container.
}


def _codec_containers() -> set[str]:
    """Container keys libsndfile reports as available (``soundfile.available_formats``).

    Split out so tests can substitute a table without pretending soundfile exists.
    """
    try:
        import soundfile
    except ImportError:  # pragma: no cover - soundfile is a hard requirement
        logger.warning("soundfile is not importable; only wav/pcm output will be available")
        return set()
    try:
        return set(soundfile.available_formats())
    except Exception as exc:  # pragma: no cover - defensive, libsndfile load errors
        logger.warning("Could not query soundfile formats (%s); assuming wav/pcm only", exc)
        return set()


@functools.lru_cache(maxsize=1)
def available_output_formats() -> frozenset[str]:
    """Containers this process can encode, probed once.

    Cached because the answer cannot change without a new interpreter: libsndfile
    is loaded at import time and its format table is static. Call
    ``available_output_formats.cache_clear()`` in tests that monkeypatch the probe.
    """
    available = set(_ALWAYS_AVAILABLE)
    codecs = _codec_containers()
    for container, codec in _CODEC_PROBED.items():
        if codec in codecs:
            available.add(container)
    missing = sorted(SUPPORTED_AUDIO_FORMATS - available)
    if missing:
        # Loud but not fatal: the server still serves the other containers, and the
        # request validator rejects these with a readable 400 instead of a 500.
        logger.warning(
            "Audio output containers unavailable in this image: %s "
            "(libsndfile without the matching encoder; requests asking for them will be rejected)",
            ", ".join(missing),
        )
    return frozenset(available & SUPPORTED_AUDIO_FORMATS)


def allowed_output_formats(adapter: object | None = None) -> frozenset[str]:
    """Formats a request may ask for, given the process capability and the adapter.

    ``adapter.supported_response_formats`` is opt-in narrowing: ``None`` means
    "whatever this process can encode". An adapter that declares a container the
    image cannot produce does not widen the set — the image wins, because the
    alternative is advertising a format that 500s.
    """
    available = available_output_formats()
    declared = getattr(adapter, "supported_response_formats", None) if adapter is not None else None
    if declared is None:
        return available
    return frozenset(declared) & available


def check_output_format(response_format: str, adapter: object | None = None) -> str | None:
    """Return an error message for an unproducible container, else ``None``."""
    requested = (response_format or "").strip().lower()
    if not requested:
        return None
    allowed = allowed_output_formats(adapter)
    if requested in allowed:
        return None
    return f"response_format='{requested}' is not available on this server; supported: {', '.join(sorted(allowed))}"
