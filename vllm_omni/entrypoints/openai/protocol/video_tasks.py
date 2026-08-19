# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol models for the asynchronous video task API (GPUStack integration).

Fourth sibling of ``audio_tasks.py`` (TTS), ``audiogen_tasks.py`` (diffusion
audio) and ``image_tasks.py`` (image generation / editing), and it exists for the
same structural reason image_tasks.py does: the GPUStack facade dispatches
strictly by engine kind, calling ``POST v1/tasks/{kind}/`` (gpustack
``routes/videos.py``), and every task_type that is not image/audio/music/audiogen
falls through to kind ``"video"`` — so a vllm-omni VIDEO model is reachable
through the facade only if the engine exposes ``/v1/tasks/video/``.

Why this cannot reuse ``POST /v1/videos``: that endpoint and ``/v1/videos/sync``
are multipart/form-data ONLY, and take their references as uploaded bytes or
URLs. The facade sends JSON, and its media inputs are ABSOLUTE paths on the
shared NFS mount it has already materialized (see gpustack
``docs/lightx2v-nfs-input-design.md``). This module is that JSON + server-path
shape; the multipart endpoints stay the direct-caller surface.

Facade -> engine body contract (gpustack ``routes/videos.py``). The facade
forwards the caller's body minus ``_CONTROL_KEYS`` (``model`` / ``task_type`` /
``user_id`` / ``input_refs``) and minus ``_ENGINE_OWNED_FIELDS``, then injects
the path fields it owns:

    image_path        comma-joined ABSOLUTE NFS paths (facade field "image")
    last_frame_path   absolute NFS path (facade field "last_frame")
    video_path        comma-joined ABSOLUTE NFS paths (facade field "video")
    audio_path        comma-joined ABSOLUTE NFS paths (facade field "audio")
    save_result_path  absolute NFS path the engine must write (.mp4)

Everything else the caller sent rides through untouched, which is how the model
knobs arrive — width / height / aspect_ratio / seconds / num_inference_steps /
seed / flow_shift / quality, and the nested ``extra_params`` object. They are NOT
re-declared here: ``extra="allow"`` collects them and
:meth:`VideoTaskRequest.to_video_request` hands them to
``VideoGenerationRequest``, which is the single place video params are typed and
range-checked. One schema, not two drifting copies.

MiniMax-H3 in particular selects its task through ``extra_params.task``
(``t2va`` / ``fl2va`` / ``ref2va``) plus ``extra_params.frame_indices``, which
the facade backfills from its own task_type — there is no top-level ``task``
field, and a top-level one would be silently dropped.

``model`` is stripped as a control key, so it is optional here and backfilled
from the server's served model name, exactly as the three sibling task requests
do.

The status / result / cancel endpoints are shared and task-id keyed, so
``AudioTaskStatus`` / ``AudioTaskResponse`` are re-exported rather than
duplicated — only the submit endpoint is new.
"""

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# Re-export the shared status/response contract: the global status / result /
# cancel endpoints serve video tasks unchanged.
from vllm_omni.entrypoints.openai.protocol.audio_tasks import (  # noqa: F401
    AudioTaskResponse,
    AudioTaskStatus,
)
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest


def _split_paths(raw: str | None) -> list[str]:
    """Split a comma-joined facade path field into individual paths.

    The facade joins list-valued inputs with "," (its ``_LIST_INPUT_FIELDS``
    handling), so this is the inverse. Blank segments are dropped so a trailing
    comma is harmless.
    """
    value = (raw or "").strip()
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


class OrderedReference(BaseModel):
    """One entry of an ordered heterogeneous reference list.

    MiniMax-H3 reads its references in order, and the order is semantic three
    times over: it numbers the ``<Picture i>`` / ``<Audio j>`` / ``<Video k>``
    labels of the prompt, it fixes the order the request generator is consumed
    in, and it places each reference block on the shared audio/video rotary
    clock. The same files in another order are a different request.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["image", "video", "audio"] = Field(
        description="Reference modality.",
    )
    path: str = Field(
        description="Facade-materialized ABSOLUTE server path, read under --allowed-local-media-path.",
    )


class VideoTaskRequest(BaseModel):
    """Async video generation task request.

    ``extra="allow"`` is load-bearing rather than merely forward-compatible: all
    generation params (geometry, steps, seed, ``extra_params``, ...) arrive as
    extras and are validated by ``VideoGenerationRequest`` in
    :meth:`to_video_request`.
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(
        default="",
        description="Served model name; optional (facade strips it), engine backfills when absent.",
    )
    # The facade sends "prompt"; accept the audio-task spellings too so a direct
    # caller can reuse the same body shape across kinds.
    prompt: str = Field(
        validation_alias=AliasChoices("prompt", "input", "text"),
        description="Generation instruction (aliases: prompt, input, text).",
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
            "output root; empty -> defaults to task_id; extension auto-added (.mp4)."
        ),
    )

    references: list[OrderedReference] | None = Field(
        default=None,
        description=(
            "References in the order the model should read them, for models where that order is "
            "semantic (MiniMax-H3). Mutually exclusive with image_path / last_frame_path / "
            "video_path / audio_path: supplying both is rejected rather than resolved, because a "
            "precedence rule is exactly the kind of convention a caller misremembers, and the cost "
            "of misremembering it is a silently different video. Absent, the bucketed fields are "
            "used and their documented canonicalization applies."
        ),
    )

    # ------------------------------------------------------------ media inputs
    # Facade-materialized server paths, never bytes or URLs. Read under
    # --allowed-local-media-path; see _resolve_task_image_path in api_server.
    image_path: str | None = Field(
        default=None,
        description=(
            "Facade-materialized reference image(s) as ABSOLUTE server paths, comma-separated. "
            "Empty/absent means text-to-video."
        ),
    )
    last_frame_path: str | None = Field(
        default=None,
        description=(
            "Facade-materialized LAST-frame image as an ABSOLUTE server path. Ordered after "
            "image_path, so first+last keyframe generation sends image_path + last_frame_path."
        ),
    )
    video_path: str | None = Field(
        default=None,
        description="Facade-materialized reference video(s) as ABSOLUTE server paths, comma-separated.",
    )
    audio_path: str | None = Field(
        default=None,
        description="Facade-materialized reference audio as ABSOLUTE server path(s), comma-separated.",
    )

    # Keys consumed structurally by the route rather than forwarded to
    # VideoGenerationRequest. The prompt aliases are listed because a caller that
    # sends BOTH "prompt" and "input" leaves the unmatched spelling in
    # model_extra.
    _ROUTE_OWNED_KEYS = (
        "references",
        "reference_order",
        "model",
        "prompt",
        "input",
        "text",
        "task_id",
        "save_result_path",
        "image_path",
        "last_frame_path",
        "video_path",
        "audio_path",
    )

    # Reference-carrying keys from the multipart surface that this endpoint
    # refuses: they mean uploaded bytes / URLs, and this route builds references
    # from server paths and never decodes them. Both halves end in the same
    # silent no-op — a COMPLETED text-to-video task instead of the
    # reference-driven one the caller asked for — so the route rejects instead.
    #
    #   image/video/audio_reference  typed VideoGenerationRequest fields, so they
    #                                validate fine and are then never decoded here
    #   input_reference(s)           multipart File params VideoGenerationRequest
    #                                does not declare at all, so pydantic's
    #                                default extra="ignore" drops them wordlessly
    #
    # This is the COMPLETE set: every other _parse_video_form param IS a
    # VideoGenerationRequest field and rides through as a normal knob.
    _UNSUPPORTED_REFERENCE_KEYS = (
        "image_reference",
        "video_reference",
        "audio_reference",
        "input_reference",
        "input_references",
    )

    def reference_image_paths(self) -> list[str]:
        """Reference images in TIMELINE order: ``image_path`` then ``last_frame_path``.

        Order is the contract, not a convenience: MiniMax-H3 FL2VA pairs the Nth
        image with the Nth entry of ``extra_params.frame_indices``, so the
        facade's ``[0, -1]`` for first+last only lines up if the first-frame image
        comes first.
        """
        if self.references is not None:
            return [entry.path for entry in self.references if entry.type == "image"]
        return _split_paths(self.image_path) + _split_paths(self.last_frame_path)

    def reference_video_paths(self) -> list[str]:
        if self.references is not None:
            return [entry.path for entry in self.references if entry.type == "video"]
        return _split_paths(self.video_path)

    def reference_audio_paths(self) -> list[str]:
        if self.references is not None:
            return [entry.path for entry in self.references if entry.type == "audio"]
        return _split_paths(self.audio_path)

    def reference_order(self) -> list[tuple[str, int]]:
        """``(kind, index-within-its-bucket)`` per reference, in request order.

        Empty when the request used the bucketed fields, which is how a caller
        says "I have no order to express" — the pipeline then applies its own
        documented canonicalization and records that it did.
        """
        if self.references is None:
            return []
        counts = {"image": 0, "video": 0, "audio": 0}
        order: list[tuple[str, int]] = []
        for entry in self.references:
            order.append((entry.type, counts[entry.type]))
            counts[entry.type] += 1
        return order

    def supplies_route_derived_order(self) -> bool:
        """Whether the caller hand-wrote ``reference_order``, which is route-owned.

        This route *derives* the order from ``references``; the field only
        exists on ``VideoGenerationRequest`` so the derived value can reach the
        generation side, which no bucketed field can carry. A hand-written one
        alongside ``image_path`` / ``video_path`` / ``audio_path`` therefore has
        nothing keeping it consistent with the media that actually arrived, and
        the mismatch is not detectable until the pipeline pairs them up — by
        which time ``reserve()`` has taken a queue slot and the caller gets a
        FAILED task rather than the 400 this route promises.

        Rejected rather than dropped: a silently ignored order produces a video
        built from the *canonical* bucket order, which is a different video, and
        that silent no-op is the same failure mode ``unsupported_reference_keys``
        exists to prevent.
        """
        return (self.model_extra or {}).get("reference_order") is not None

    def conflicting_reference_inputs(self) -> list[str]:
        """Bucketed media fields supplied alongside an ordered list, if any."""
        if self.references is None:
            return []
        return [
            name for name in ("image_path", "last_frame_path", "video_path", "audio_path") if getattr(self, name, None)
        ]

    def unsupported_reference_keys(self) -> list[str]:
        """Byte/URL reference keys present in the body, if any.

        Looks one level into ``extra_params`` as well. That dict is an opaque
        passthrough merged wholesale into ``gen_params.extra_args``, where every
        consumer reads by explicit key — so a reference nested there is read by
        nobody, and the caller gets a COMPLETED text-to-video task instead of the
        reference-driven one they asked for. That silent no-op is the exact
        failure this check exists to prevent; only the three
        declared-unsupported names are inspected, and only one level down, so
        genuinely opaque engine params keep riding through untouched.
        """
        extra = self.model_extra or {}
        nested = extra.get("extra_params")
        if not isinstance(nested, dict):
            nested = {}
        found = [key for key in self._UNSUPPORTED_REFERENCE_KEYS if extra.get(key) is not None]
        found += [f"extra_params.{key}" for key in self._UNSUPPORTED_REFERENCE_KEYS if nested.get(key) is not None]
        return found

    def to_video_request(self) -> VideoGenerationRequest:
        """Build the ``VideoGenerationRequest`` the video handler consumes.

        Raises ``pydantic.ValidationError`` for out-of-range or malformed params;
        the route turns that into a 400 rather than a task that fails later.

        Two properties matter and are both tested:

        * ``None`` values are dropped. ``OmniOpenAIServingVideo._run_and_extract``
          gates most knobs on ``model_fields_set``, so forwarding an explicit
          ``None`` would mark a field as "provided" and overwrite the engine's
          own default with nothing.
        * Route-owned keys are dropped, so a path field can never be re-read as a
          generation param.
        """
        payload: dict[str, Any] = {
            key: value
            for key, value in (self.model_extra or {}).items()
            if key not in self._ROUTE_OWNED_KEYS and key not in self._UNSUPPORTED_REFERENCE_KEYS and value is not None
        }
        payload["prompt"] = self.prompt
        if self.model:
            payload["model"] = self.model
        # `references` itself is route-owned and dropped above — the paths are
        # resolved and security-checked by the route. What the generation side
        # needs is the ORDER, which no bucketed field can carry.
        order = self.reference_order()
        if order:
            payload["reference_order"] = [{"type": kind, "index": index} for kind, index in order]
        return VideoGenerationRequest(**payload)
