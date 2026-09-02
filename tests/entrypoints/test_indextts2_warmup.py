# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""IndexTTS2 startup warmup (CPU-only).

IndexTTS compiles two Triton kernels on the first inference of every process
(``kernel_unified_attention`` and ``SnakeBeta._init_triton``). Measured on a
production instance those sat 93 seconds apart and made the first request take
96s against a 15s steady state — a 6x penalty paid by whichever user arrived
first after a restart, on every replica, after every upgrade.

The adapter inherited a no-op ``warmup()`` from the base class, so the hook
``init_app_state`` already calls before flipping ``/ready`` did nothing. These
tests pin the parts that are silent when broken: that a request is actually
issued, that it carries a cloneable reference voice (IndexTTS2 has no zero-shot
mode and would reject the request), and that a failure cannot take the server
down.
"""

from typing import Any

import pybase64 as base64
import pytest

from vllm_omni.entrypoints.openai.tts_adapters.indextts2 import (
    _WARMUP_REF_AUDIO,
    IndexTTS2Adapter,
    IndexTTS25Adapter,
    _warmup_ref_audio_uri,
)

# core_model/cpu are the Buildkite collection marks; asyncio drives the async cases.
pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


class _Server:
    """Minimal stand-in for the serving object the adapter talks to."""

    model_name = "indextts-2.5"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[Any, str | None]] = []
        self.uploaded_speakers: dict[str, Any] = {}

    # Mirrors the real serving object's checks so ``validate()`` below is a
    # meaningful assertion rather than a stub agreeing with itself.
    def _validate_ref_audio_format(self, ref_audio):
        if not isinstance(ref_audio, str) or not ref_audio.startswith(("http://", "https://", "data:", "file://")):
            return "ref_audio must be a URL (http/https), base64 data URL (data:...), or file URI (file://...)"
        return None

    def _get_uploaded_audio_data(self, name):
        return None

    async def _generate_audio_bytes(self, request, request_id=None):
        self.calls.append((request, request_id))
        if self.fail:
            raise RuntimeError("engine exploded")
        return b"RIFF", "audio/wav"


class _Ctx:
    def __init__(self, server):
        self.server = server


def _adapter(cls=IndexTTS2Adapter, fail: bool = False):
    adapter = cls.__new__(cls)  # skip __init__; warmup only needs ctx
    adapter.ctx = _Ctx(_Server(fail=fail))
    return adapter


# ------------------------------------------------------------------ asset


def test_reference_asset_ships_with_the_package():
    """A build that drops package data would make warmup silently no-op."""
    assert _WARMUP_REF_AUDIO.is_file(), f"missing warmup asset: {_WARMUP_REF_AUDIO}"
    assert _WARMUP_REF_AUDIO.stat().st_size > 1024


def test_reference_is_a_base64_data_uri():
    """Must be a data URI, not file://.

    Production launches with ``--allowed-local-media-path /nfs-output`` while
    this asset lives inside the installed package, so a file:// URI would be
    rejected by the path allowlist and warmup would no-op on exactly the
    deployment it exists for.
    """
    uri = _warmup_ref_audio_uri()
    assert uri.startswith("data:audio/wav;base64,")
    payload = base64.b64decode(uri.split(",", 1)[1])
    assert payload[:4] == b"RIFF", "reference audio is not a RIFF/WAV stream"


# ----------------------------------------------------------------- warmup


async def test_warmup_issues_a_request_with_a_reference_voice():
    """The whole point: an actual inference must run, and it must be one the
    model accepts. IndexTTS2 has no zero-shot mode — ``validate`` rejects a
    request with neither ``ref_audio`` nor an uploaded voice — so a warmup that
    forgot the reference would be silently swallowed by the except branch and
    the JIT cost would still land on the first user."""
    adapter = _adapter()
    await adapter.warmup()

    assert len(adapter.ctx.server.calls) == 1, "warmup did not run any inference"
    request, request_id = adapter.ctx.server.calls[0]
    assert request_id == "speech-warmup"
    assert request.ref_audio.startswith("data:audio/wav;base64,")
    assert request.input.strip(), "empty input is rejected by validate()"

    # Same guard the real request path applies; catches a warmup body that the
    # model would refuse.
    assert adapter.validate(request) is None


async def test_warmup_voice_name_matches_no_uploaded_speaker():
    """``voice`` is required by the schema but must not collide with a real
    uploaded speaker, or _build_params would clone that speaker instead of the
    bundled reference."""
    adapter = _adapter()
    await adapter.warmup()
    request, _ = adapter.ctx.server.calls[0]
    assert request.voice.lower() not in adapter.ctx.server.uploaded_speakers


async def test_warmup_failure_is_not_fatal():
    """A model that cannot warm up still serves — just slowly on the first
    request. Propagating here would turn a latency problem into an
    availability one, because this runs inside server startup."""
    adapter = _adapter(fail=True)
    await adapter.warmup()  # must not raise


async def test_2_5_inherits_the_warmup():
    """IndexTTS25Adapter subclasses IndexTTS2Adapter and must not re-introduce
    the base class's no-op."""
    adapter = _adapter(IndexTTS25Adapter)
    await adapter.warmup()
    assert len(adapter.ctx.server.calls) == 1

    request, _ = adapter.ctx.server.calls[0]
    # 2.5 adds a speed range check on top of 2's validation.
    assert adapter.validate(request) is None
