# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 unit tests for the async diffusion-audio task API (AudioX / SoulX-Singer).

Pure-logic coverage of the request -> chat mapping — no GPU / model / engine:
- AudioGenTaskRequest text alias (input/text/prompt)
- to_chat_request(): AudioX params flatten to top-level (model_extra); SoulX
  params nest under extra_args; modalities==["audio"]; messages text == input
- shared status/response contract (AudioTaskStatus / AudioTaskResponse) is the
  same object as the TTS async API (facade parses it unchanged)
"""

import pytest

from vllm_omni.entrypoints.openai.protocol.audiogen_tasks import (
    AudioGenTaskRequest,
    AudioTaskResponse,
    AudioTaskStatus,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _flat_params(chat_req) -> dict:
    """The diffusion path reads params from model_extra when there is no declared
    extra_body field, so the flattened diffusion params live there."""
    return chat_req.model_extra or {}


# --------------------------------------------------------------------------- #
# 1. Text alias (input / text / prompt).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["input", "text", "prompt"])
def test_audiogen_request_text_alias(key):
    req = AudioGenTaskRequest.model_validate({"model": "audiox", key: "a dog barking"})
    assert req.input == "a dog barking"


# --------------------------------------------------------------------------- #
# 2. Shared status/response contract is the SAME object as the TTS async API.
# --------------------------------------------------------------------------- #
def test_shares_status_and_response_contract():
    # Re-imported from audio_tasks, so string values / fields are identical.
    from vllm_omni.entrypoints.openai.protocol import audio_tasks as _at

    assert AudioTaskStatus is _at.AudioTaskStatus
    assert AudioTaskResponse is _at.AudioTaskResponse
    assert AudioTaskStatus.PENDING.value == "pending"
    assert AudioTaskStatus.PROCESSING.value == "processing"
    assert AudioTaskStatus.COMPLETED.value == "completed"
    assert AudioTaskStatus.FAILED.value == "failed"
    assert AudioTaskStatus.CANCELLED.value == "cancelled"


# --------------------------------------------------------------------------- #
# 3. to_chat_request() — messages, modalities, and the base shape.
# --------------------------------------------------------------------------- #
def test_to_chat_request_messages_and_modalities():
    req = AudioGenTaskRequest.model_validate({"model": "audiox", "input": "a cat meowing"})
    chat = req.to_chat_request()
    assert chat.model == "audiox"
    # single user message with a text content part carrying the input verbatim
    assert chat.messages == [{"role": "user", "content": [{"type": "text", "text": "a cat meowing"}]}]
    flat = _flat_params(chat)
    assert flat["modalities"] == ["audio"]
    # no diffusion params set -> only modalities present, no extra_args key
    assert "extra_args" not in flat


# --------------------------------------------------------------------------- #
# 4. AudioX params flatten to the top level (read via model_extra).
# --------------------------------------------------------------------------- #
def test_audiox_params_land_top_level():
    req = AudioGenTaskRequest.model_validate(
        {
            "model": "audiox",
            "input": "ocean waves",
            "audiox_task": "v2a",
            "seconds_total": 10.0,
            "seconds_start": 0.0,
            "sigma_min": 0.3,
            "sigma_max": 500.0,
            "cfg_rescale": 0.7,
            "num_inference_steps": 100,
            "guidance_scale": 7.0,
            "seed": 42,
            "video_path": "/nfs/clips/scene.mp4",
        }
    )
    flat = _flat_params(req.to_chat_request())
    assert flat["audiox_task"] == "v2a"
    assert flat["seconds_total"] == 10.0
    assert flat["seconds_start"] == 0.0
    assert flat["sigma_min"] == 0.3
    assert flat["sigma_max"] == 500.0
    assert flat["cfg_rescale"] == 0.7
    assert flat["num_inference_steps"] == 100
    assert flat["guidance_scale"] == 7.0
    assert flat["seed"] == 42
    assert flat["video_path"] == "/nfs/clips/scene.mp4"
    # AudioX params are top-level, never nested under extra_args
    assert "extra_args" not in flat


def test_extra_diffusion_params_are_forwarded():
    """Undeclared diffusion options accepted via extra="allow" (negative_prompt,
    num_outputs_per_prompt, size, lora, future model extras) must reach the chat
    request's model_extra — the diffusion path reads them there. Dropping them
    would silently fall back to defaults (Codex P2)."""
    req = AudioGenTaskRequest.model_validate(
        {
            "model": "audiox",
            "input": "ocean waves",
            "audiox_task": "t2a",
            "negative_prompt": "music, melody",
            "num_outputs_per_prompt": 2,
            "size": "1024x1024",
            "lora": {"name": "foo", "scale": 0.8},
        }
    )
    flat = _flat_params(req.to_chat_request())
    assert flat["negative_prompt"] == "music, melody"
    assert flat["num_outputs_per_prompt"] == 2
    assert flat["size"] == "1024x1024"
    assert flat["lora"] == {"name": "foo", "scale": 0.8}
    # typed field still present + modality forced
    assert flat["audiox_task"] == "t2a"
    assert flat["modalities"] == ["audio"]


def test_caller_extra_args_merged_with_soulx_fields():
    """A caller-supplied extra_args dict is merged with (not clobbered by) the
    SoulX-derived extra_args."""
    req = AudioGenTaskRequest.model_validate(
        {
            "model": "soulx-singer",
            "input": "soulx-singer",
            "language": "Mandarin",
            "extra_args": {"custom_knob": 1},
        }
    )
    flat = _flat_params(req.to_chat_request())
    assert flat["extra_args"]["custom_knob"] == 1
    assert flat["extra_args"]["language"] == "Mandarin"


# --------------------------------------------------------------------------- #
# 5. SoulX-Singer params nest under extra_body["extra_args"].
# --------------------------------------------------------------------------- #
def test_soulx_params_land_under_extra_args():
    req = AudioGenTaskRequest.model_validate(
        {
            "model": "soulx-singer",
            "input": "soulx-singer",
            "num_inference_steps": 32,
            "guidance_scale": 3.0,
            "seed": 7,
            "prompt_audio": "/nfs/prompt.wav",
            "target_audio": "/nfs/target.wav",
            "language": "Mandarin",
            "control": "melody",
            "auto_shift": True,
            "pitch_shift": 2,
            "vocal_sep": False,
        }
    )
    flat = _flat_params(req.to_chat_request())
    # shared diffusion params stay top-level
    assert flat["num_inference_steps"] == 32
    assert flat["guidance_scale"] == 3.0
    assert flat["seed"] == 7
    # SoulX-specific params are nested under extra_args
    extra_args = flat["extra_args"]
    assert extra_args["prompt_audio"] == "/nfs/prompt.wav"
    assert extra_args["target_audio"] == "/nfs/target.wav"
    assert extra_args["language"] == "Mandarin"
    assert extra_args["control"] == "melody"
    assert extra_args["auto_shift"] is True
    assert extra_args["pitch_shift"] == 2
    assert extra_args["vocal_sep"] is False
    # SoulX params must not also leak to top-level
    assert "prompt_audio" not in flat
    assert "language" not in flat


def test_soulx_precomputed_metadata_forward_compat():
    req = AudioGenTaskRequest.model_validate(
        {
            "model": "soulx-singer",
            "input": "soulx-singer",
            "prompt_metadata_path": "/nfs/prompt.meta",
            "target_metadata_path": "/nfs/target.meta",
        }
    )
    extra_args = _flat_params(req.to_chat_request())["extra_args"]
    assert extra_args["prompt_metadata_path"] == "/nfs/prompt.meta"
    assert extra_args["target_metadata_path"] == "/nfs/target.meta"


# --------------------------------------------------------------------------- #
# 6. Unset optional params are omitted (no null noise handed to the pipeline).
# --------------------------------------------------------------------------- #
def test_unset_params_are_omitted():
    req = AudioGenTaskRequest.model_validate({"model": "audiox", "input": "wind"})
    flat = _flat_params(req.to_chat_request())
    assert flat == {"modalities": ["audio"]}
