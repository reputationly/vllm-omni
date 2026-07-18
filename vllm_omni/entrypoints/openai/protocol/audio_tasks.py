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

from enum import Enum

from pydantic import AliasChoices, BaseModel, Field

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


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

    def to_speech_request(self) -> OpenAICreateSpeechRequest:
        """Build a clean synchronous speech request (async-only fields stripped,
        streaming forced off so the byte path is used)."""
        data = self.model_dump(exclude={"task_id", "save_result_path"})
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
