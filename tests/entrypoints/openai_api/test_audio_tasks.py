# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 unit tests for the async audio task API (GPUStack integration).

Covers the pure-logic contract of the P1 async pieces — no GPU/model needed:
- AudioTaskStatus string values (facade `_ENGINE_STATE_MAP` alignment)
- AudioTaskRequest text alias (input/text/prompt) + to_speech_request()
- resolve_save_path() four branches
- AudioTaskManager backpressure / duplicate / queue_status / cancel
- atomic_write_bytes() to an arbitrary absolute path + parent creation
"""

import asyncio

import pytest

from vllm_omni.entrypoints.openai.audio_task_manager import AudioTaskManager, resolve_save_path
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.protocol.audio_tasks import (
    AudioTaskRequest,
    AudioTaskResponse,
    AudioTaskStatus,
)
from vllm_omni.entrypoints.openai.storage import atomic_write_bytes
from vllm_omni.entrypoints.openai.stores import AsyncDictStore, TaskRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# --------------------------------------------------------------------------- #
# 1. Status strings — must match the facade contract exactly.
# --------------------------------------------------------------------------- #
def test_audio_task_status_values():
    assert AudioTaskStatus.PENDING.value == "pending"
    assert AudioTaskStatus.PROCESSING.value == "processing"
    assert AudioTaskStatus.COMPLETED.value == "completed"
    assert AudioTaskStatus.FAILED.value == "failed"
    # double-L "cancelled" (facade maps to CANCELED); NOT "canceled".
    assert AudioTaskStatus.CANCELLED.value == "cancelled"


def test_audio_task_response_fields_use_start_end_time():
    resp = AudioTaskResponse(task_id="t1", status=AudioTaskStatus.PENDING)
    dumped = resp.model_dump()
    # Contract fields (LightX2V task_manager), not created_at/completed_at.
    assert set(dumped) == {
        "task_id",
        "status",
        "start_time",
        "end_time",
        "error",
        "error_type",
        "save_result_path",
    }
    assert dumped["start_time"] is None and dumped["end_time"] is None
    assert dumped["error_type"] == ""


# --------------------------------------------------------------------------- #
# 2. AudioTaskRequest text alias + to_speech_request()
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["input", "text", "prompt"])
def test_audio_task_request_text_alias(key):
    req = AudioTaskRequest.model_validate({key: "hello", "voice": "vivian"})
    assert req.input == "hello"


def test_to_speech_request_strips_async_fields_and_disables_stream():
    req = AudioTaskRequest.model_validate(
        {
            "input": "hi",
            "voice": "vivian",
            "stream": True,
            "task_id": "abc",
            "save_result_path": "/nfs/out.wav",
        }
    )
    speech = req.to_speech_request()
    assert isinstance(speech, OpenAICreateSpeechRequest)
    # async-only fields must not leak onto the speech request
    assert not hasattr(speech, "task_id")
    assert not hasattr(speech, "save_result_path")
    # streaming forced off so the byte path (_generate_audio_bytes) is used
    assert speech.stream is False
    assert speech.stream_format is None
    # passthrough speech fields preserved
    assert speech.input == "hi"
    assert speech.voice == "vivian"


# --------------------------------------------------------------------------- #
# 3. resolve_save_path — four branches.
# --------------------------------------------------------------------------- #
def test_resolve_save_path_absolute_verbatim():
    assert resolve_save_path("/nfs/out.wav", "task1", "/root") == "/nfs/out.wav"


def test_resolve_save_path_relative_under_root():
    assert resolve_save_path("sub/out.wav", "task1", "/root") == "/root/sub/out.wav"


def test_resolve_save_path_empty_uses_task_id():
    assert resolve_save_path("", "task1", "/root") == "/root/task1.wav"


def test_resolve_save_path_appends_default_extension():
    # absolute without suffix -> .wav appended
    assert resolve_save_path("/nfs/out", "task1", "/root") == "/nfs/out.wav"
    # relative without suffix -> under root + .wav
    assert resolve_save_path("out", "task1", "/root") == "/root/out.wav"


# --------------------------------------------------------------------------- #
# 4. AudioTaskManager — backpressure / duplicate / queue_status / cancel.
# --------------------------------------------------------------------------- #
def _fresh_manager(max_queue_size: int = 8) -> AudioTaskManager:
    # Inject isolated store/registry so tests never share global state.
    return AudioTaskManager(
        max_queue_size=max_queue_size,
        store=AsyncDictStore(),
        tasks=TaskRegistry(),
    )


@pytest.mark.asyncio
async def test_reserve_creates_pending_record():
    mgr = _fresh_manager()
    resp = await mgr.reserve("t1", "/nfs/t1.wav")
    assert resp.task_id == "t1"
    assert resp.status == AudioTaskStatus.PENDING
    assert resp.save_result_path == "/nfs/t1.wav"
    assert (await mgr.get_status("t1")).status == AudioTaskStatus.PENDING


@pytest.mark.asyncio
async def test_reserve_duplicate_raises_runtime_error():
    mgr = _fresh_manager()
    await mgr.reserve("t1", "/nfs/t1.wav")
    with pytest.raises(RuntimeError, match="already exists"):
        await mgr.reserve("t1", "/nfs/t1.wav")


@pytest.mark.asyncio
async def test_reserve_queue_full_raises_runtime_error():
    mgr = _fresh_manager(max_queue_size=2)
    await mgr.reserve("t1", "/nfs/t1.wav")
    await mgr.reserve("t2", "/nfs/t2.wav")
    with pytest.raises(RuntimeError, match="queue is full"):
        await mgr.reserve("t3", "/nfs/t3.wav")


@pytest.mark.asyncio
async def test_queue_status_fields():
    mgr = _fresh_manager(max_queue_size=8)
    await mgr.reserve("t1", "/nfs/t1.wav")
    status = await mgr.queue_status()
    assert set(status) == {
        "is_processing",
        "current_task",
        "pending_count",
        "active_count",
        "queue_size",
        "queue_available",
    }
    assert status["pending_count"] == 1
    assert status["active_count"] == 1
    assert status["queue_size"] == 8
    assert status["queue_available"] == 7
    assert status["is_processing"] is False
    assert status["current_task"] is None


@pytest.mark.asyncio
async def test_cancel_marks_cancelled_and_cancels_task():
    mgr = _fresh_manager()
    await mgr.reserve("t1", "/nfs/t1.wav")

    # A never-completing dummy task registered as the running job.
    async def _never():
        await asyncio.sleep(3600)

    task = asyncio.create_task(_never())
    await mgr._tasks.upsert("t1", task)

    assert await mgr.cancel("t1") is True
    assert (await mgr.get_status("t1")).status == AudioTaskStatus.CANCELLED
    assert (await mgr.get_status("t1")).end_time is not None
    # underlying asyncio task was cancelled
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancel_unknown_returns_false():
    mgr = _fresh_manager()
    assert await mgr.cancel("nope") is False


@pytest.mark.asyncio
async def test_cancel_terminal_returns_false():
    mgr = _fresh_manager()
    await mgr.reserve("t1", "/nfs/t1.wav")
    await mgr._store.update_fields("t1", {"status": AudioTaskStatus.COMPLETED})
    assert await mgr.cancel("t1") is False


# --------------------------------------------------------------------------- #
# 5. atomic_write_bytes — arbitrary absolute path + parent creation.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_atomic_write_bytes_writes_and_creates_parent(tmp_path):
    dest = tmp_path / "nested" / "dir" / "out.wav"  # parent does not exist yet
    data = b"RIFFsmoke-bytes"
    written = await atomic_write_bytes(data, str(dest))
    assert written == str(dest)
    assert dest.read_bytes() == data


@pytest.mark.asyncio
async def test_atomic_write_bytes_overwrites_atomically(tmp_path):
    dest = tmp_path / "out.wav"
    await atomic_write_bytes(b"first", str(dest))
    await atomic_write_bytes(b"second", str(dest))
    assert dest.read_bytes() == b"second"
    # no leftover temp files in the directory
    assert [p.name for p in tmp_path.iterdir()] == ["out.wav"]
