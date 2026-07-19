# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol models for the asynchronous diffusion-audio task API.

This is the diffusion-model sibling of ``audio_tasks.py``. Where that module
drives the TTS handler (``/v1/audio/speech``) via submit+poll, this one drives
the *diffusion* chat path (``/v1/chat/completions`` in diffusion mode) so the
GPUStack facade can run AudioX (``AudioXPipeline``) and SoulX-Singer
(``SoulXSingerPipeline``) through the identical async contract.

Both diffusion audio models are served through ``create_chat_completion`` with
``modalities=["audio"]``; the generated WAV is returned base64-encoded in
``choices[0].message.audio.data``. ``AudioGenTaskRequest.to_chat_request``
builds the equivalent ``ChatCompletionRequest`` so the async job runner can call
the chat handler directly (raw_request=None is safe in the diffusion branch).

The task status/response contract is intentionally shared with the TTS async
API: ``AudioTaskStatus`` / ``AudioTaskResponse`` are re-imported here so the
status (``GET /v1/tasks/{task_id}/status``), result and cancel endpoints are the
same global, task-id-keyed endpoints — a new submit endpoint only.

Model -> param contract (verified via POC on real hardware):

AudioX (extra_body top-level keys):
    audiox_task (t2a/v2a/v2m/tv2a/tv2m), seconds_total, seconds_start,
    sigma_min, sigma_max, cfg_rescale, num_inference_steps, guidance_scale,
    seed, video_path (a bare local mp4 path — AudioX loads it via av.open, no
    file:// scheme needed). The text prompt travels in the chat messages.

SoulX-Singer SVS (extra_body["extra_args"] dict):
    prompt_audio, target_audio (bare server paths, integrated preprocess),
    language (e.g. "Mandarin"), control ("melody"/"score"), auto_shift (bool),
    pitch_shift (int), vocal_sep (bool). The chat text is the literal
    "soulx-singer". SVC + precomputed-metadata is a later batch; the optional
    prompt_metadata_path / target_metadata_path / audio_path fields are carried
    here for forward-compat and routed into extra_args when present.
"""

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

# Re-export the shared status/response contract so the status/result/cancel
# endpoints (which are global and task-id-keyed) serve diffusion-audio tasks
# unchanged. Importers can pull these from either module.
from vllm_omni.entrypoints.openai.protocol.audio_tasks import (  # noqa: F401
    AudioTaskResponse,
    AudioTaskStatus,
)


class AudioGenTaskRequest(BaseModel):
    """Async diffusion-audio task request (AudioX / SoulX-Singer).

    ``extra="allow"`` keeps the request forward-compatible: any not-yet-typed
    pipeline param the facade sends is accepted and can be folded into the chat
    request's flattened params without a schema bump.
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(description="Served model name (AudioX / SoulX-Singer).")
    # Accept input/text/prompt for the generation text (facade / new-api compat).
    input: str = Field(
        validation_alias=AliasChoices("input", "text", "prompt"),
        description="Generation text (aliases: input, text, prompt).",
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

    # ------------------------------------------------------------------ AudioX
    # These land at the top level of the flattened chat params (model_extra),
    # exactly where _create_diffusion_chat_completion reads them.
    audiox_task: str | None = Field(
        default=None,
        description="AudioX task: t2a / v2a / v2m / tv2a / tv2m.",
    )
    seconds_total: float | None = Field(default=None, description="AudioX output duration (s).")
    seconds_start: float | None = Field(default=None, description="AudioX output start offset (s).")
    sigma_min: float | None = Field(default=None, description="AudioX sampler sigma_min.")
    sigma_max: float | None = Field(default=None, description="AudioX sampler sigma_max.")
    cfg_rescale: float | None = Field(default=None, description="AudioX CFG rescale factor.")
    video_path: str | None = Field(
        default=None,
        description="Bare local mp4 path for AudioX v2a/v2m/tv2a/tv2m (loaded via av.open).",
    )
    audio_path: str | None = Field(
        default=None,
        description="Bare local audio path for AudioX conditioning (forward-compat).",
    )

    # ------------------------------------------------- shared diffusion params
    num_inference_steps: int | None = Field(default=None, description="Diffusion steps.")
    guidance_scale: float | None = Field(default=None, description="CFG guidance scale.")
    seed: int | None = Field(default=None, description="Sampling seed.")

    # ------------------------------------------------------ SoulX-Singer (SVS)
    # These are collected into a nested extra_args dict (extra_body["extra_args"])
    # which the diffusion path merges into gen_params.extra_args.
    prompt_audio: str | None = Field(default=None, description="SoulX reference/prompt audio path.")
    target_audio: str | None = Field(default=None, description="SoulX target audio path.")
    language: str | None = Field(default=None, description='SoulX language (e.g. "Mandarin").')
    control: str | None = Field(default=None, description='SoulX control: "melody" / "score".')
    auto_shift: bool | None = Field(default=None, description="SoulX auto pitch shift.")
    pitch_shift: int | None = Field(default=None, description="SoulX manual pitch shift (semitones).")
    vocal_sep: bool | None = Field(default=None, description="SoulX vocal separation toggle.")
    # SVC + precomputed-metadata (later batch; forward-compat, optional).
    prompt_metadata_path: str | None = Field(
        default=None, description="SoulX precomputed prompt metadata path (forward-compat)."
    )
    target_metadata_path: str | None = Field(
        default=None, description="SoulX precomputed target metadata path (forward-compat)."
    )

    # AudioX top-level extra_body keys (bare, not nested in extra_args).
    _AUDIOX_KEYS = (
        "audiox_task",
        "seconds_total",
        "seconds_start",
        "sigma_min",
        "sigma_max",
        "cfg_rescale",
        "num_inference_steps",
        "guidance_scale",
        "seed",
        "video_path",
        "audio_path",
    )
    # SoulX-Singer keys collected under extra_body["extra_args"].
    _SOULX_EXTRA_ARGS_KEYS = (
        "prompt_audio",
        "target_audio",
        "language",
        "control",
        "auto_shift",
        "pitch_shift",
        "vocal_sep",
        "prompt_metadata_path",
        "target_metadata_path",
    )

    def to_chat_request(self) -> ChatCompletionRequest:
        """Build the equivalent diffusion ``ChatCompletionRequest``.

        The diffusion chat path reads its params from
        ``getattr(request, "extra_body", None) or request.model_extra``. vLLM's
        ``ChatCompletionRequest`` (base ``OpenAIBaseModel``, ``extra="allow"``)
        does NOT declare an ``extra_body`` field, so a nested ``extra_body=``
        kwarg would be buried as ``model_extra["extra_body"]`` and read one level
        too deep. Instead we FLATTEN the diffusion params to top-level kwargs so
        they land directly in ``model_extra`` — the exact shape the OpenAI client
        produces and the ``or request.model_extra`` fallback consumes. The nested
        SoulX ``extra_args`` dict is preserved as a single top-level key.
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": self.input}]}]

        # Flattened params handed to ChatCompletionRequest as **kwargs -> model_extra.
        flat_params: dict[str, Any] = {}

        # Forward caller-supplied extras accepted via extra="allow" (negative_prompt,
        # num_outputs_per_prompt, size/height/width, lora, and any future model
        # extra) — the diffusion chat path reads these from model_extra, so dropping
        # them would silently fall back to defaults and break the endpoint's
        # forward-compatible contract. Typed AudioX fields below override any stray
        # extra with the same name; a caller-supplied extra_args dict is merged with
        # (and takes lower precedence than) the SoulX fields.
        if self.model_extra:
            flat_params.update(self.model_extra)

        for key in self._AUDIOX_KEYS:
            value = getattr(self, key, None)
            if value is not None:
                flat_params[key] = value

        extra_args: dict[str, Any] = {}
        caller_extra_args = flat_params.get("extra_args")
        if isinstance(caller_extra_args, dict):
            extra_args.update(caller_extra_args)
        for key in self._SOULX_EXTRA_ARGS_KEYS:
            value = getattr(self, key, None)
            if value is not None:
                extra_args[key] = value
        if extra_args:
            flat_params["extra_args"] = extra_args

        # Audiogen always produces audio — force the modality last so a caller can't
        # override it via an extra.
        flat_params["modalities"] = ["audio"]

        return ChatCompletionRequest(model=self.model, messages=messages, **flat_params)
