# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol models for the asynchronous audio task API (GPUStack integration).

Mirrors the async task contract used by LightX2V / IndexTTS so the GPUStack
facade (``POST /v1/tasks/audio/`` + poll) can drive vllm-omni TTS the same way
it drives those engines. This reuses the existing synchronous speech request
shape (``OpenAICreateSpeechRequest``) so every TTS model keeps working
unchanged — we only add the async-contract fields (``task_id`` /
``save_result_path``) and the task status/response models.

Status strings and the status-response field names match LightX2V's
``task_manager`` so the facade ``_ENGINE_STATE_MAP`` maps them without change.
"""

import os
from enum import Enum
from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


def _local_path_to_file_uri(path: str) -> str:
    """Bridge a bare absolute filesystem path to a ``file://`` URI.

    The GPUStack facade materializes voice-clone / dialogue reference audio onto
    shared NFS and injects the ABSOLUTE path under ``ref_audio_path`` /
    ``ref_audio_2_path`` (mirroring IndexTTS's ``spk_audio_path`` contract). The
    speech handler's ``ref_audio`` / ``ref_audio_2`` fields, however, only accept
    ``http(s)`` / ``data:`` / ``file://`` URIs (see
    ``OmniOpenAIServingSpeech._validate_ref_audio_format``), so convert here.
    Values already carrying a scheme pass through untouched; a non-absolute value
    is returned as-is so the handler surfaces a clear validation error rather than
    silently mis-resolving it.
    """
    p = (path or "").strip()
    if not p:
        return p
    if p.startswith(("file://", "http://", "https://", "data:")):
        return p
    if os.path.isabs(p):
        return Path(p).as_uri()
    return p


class AudioTaskStatus(str, Enum):
    """Task lifecycle states. ``cancelled`` is double-L to match the facade
    ``_ENGINE_STATE_MAP`` (which maps it to CANCELED)."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioTaskRequest(OpenAICreateSpeechRequest):
    """Async TTS task request = OpenAI speech fields + async-contract fields.

    Inheriting ``OpenAICreateSpeechRequest`` means all current and future
    speech fields (voice, instructions, ref_audio, ref_text, ref_audio_2,
    language, task_type, speaker_embedding, ambient_sound, duration_seconds…)
    pass straight through to the shared handler — no per-model wiring.
    """

    # Accept input/text/prompt for the synthesis text (facade / new-api compat).
    input: str = Field(
        validation_alias=AliasChoices("input", "text", "prompt"),
        description="Text to synthesize (aliases: input, text, prompt).",
    )
    task_id: str | None = Field(
        default=None,
        description="Client/facade-supplied task id; auto-generated if absent.",
    )
    save_result_path: str = Field(
        default="",
        description=(
            "Where the engine writes the result. Absolute -> written verbatim "
            "(facade injects an NFS absolute path); relative -> resolved under the "
            "output root; empty -> defaults to task_id; extension auto-added (.wav)."
        ),
    )
    # Facade-materialized reference-audio paths. The GPUStack facade writes the
    # voice-clone / second-speaker reference onto shared NFS and injects the
    # absolute path here (its _INPUT_FIELDS map: ref_audio -> ref_audio_path,
    # ref_audio_2 -> ref_audio_2_path). to_speech_request folds them into the
    # handler's URI-only ref_audio / ref_audio_2 fields as file:// URIs.
    ref_audio_path: str | None = Field(
        default=None,
        description=(
            "Absolute filesystem path to the voice-clone reference audio "
            "(GPUStack facade materializes it onto shared NFS). Folded into "
            "'ref_audio' as a file:// URI when 'ref_audio' is not otherwise set."
        ),
    )
    ref_audio_2_path: str | None = Field(
        default=None,
        description=(
            "Absolute filesystem path to the second reference audio for "
            "two-speaker dialogue (MOSS-TTSD). Folded into 'ref_audio_2' as a "
            "file:// URI when 'ref_audio_2' is not otherwise set."
        ),
    )

    def to_speech_request(self) -> OpenAICreateSpeechRequest:
        """Build a clean synchronous speech request (async-only fields stripped,
        streaming forced off so the byte path is used)."""
        data = self.model_dump(
            exclude={
                "task_id",
                "save_result_path",
                "ref_audio_path",
                "ref_audio_2_path",
            }
        )
        # Bridge the facade's materialized NFS paths onto the handler's URI-only
        # ref_audio / ref_audio_2. An explicit ref_audio in the request wins (a
        # caller that already sent a URL/base64/file URI is never overridden).
        if self.ref_audio_path and not data.get("ref_audio"):
            data["ref_audio"] = _local_path_to_file_uri(self.ref_audio_path)
        if self.ref_audio_2_path and not data.get("ref_audio_2"):
            data["ref_audio_2"] = _local_path_to_file_uri(self.ref_audio_2_path)
        data["stream"] = False
        data["stream_format"] = None
        return OpenAICreateSpeechRequest.model_validate(data)


class AudioTaskResponse(BaseModel):
    """Stored task state and status-response body.

    Fields mirror LightX2V ``task_manager.get_task_status`` (``start_time`` /
    ``end_time``, not ``created_at`` / ``completed_at``) so the GPUStack facade
    parses it unchanged.
    """

    task_id: str
    status: AudioTaskStatus
    start_time: float | None = None
    end_time: float | None = None
    error: str | None = None
    error_type: str = ""
    save_result_path: str | None = None
