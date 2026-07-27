# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol models for the asynchronous image task API (GPUStack integration).

Third sibling of ``audio_tasks.py`` (TTS) and ``audiogen_tasks.py`` (diffusion
audio). Where those drive the speech handler and the diffusion chat path, this
one drives IMAGE generation/editing so the GPUStack facade can run vllm-omni
image models through the same submit+poll contract it already uses for LightX2V.

Why an async task endpoint at all, when ``/v1/images/edits`` already serves the
same models synchronously: the facade dispatches strictly by engine kind, calling
``POST v1/tasks/{kind}/`` (gpustack ``routes/videos.py``), so an image model
reachable through the facade must expose ``/v1/tasks/image/``. It also matches
how slow models behave — a request that takes minutes is far better polled than
held open on a blocking HTTP connection.

Facade -> engine body contract (gpustack ``routes/videos.py``). The facade
forwards the caller's body minus ``_CONTROL_KEYS`` (``model`` / ``task_type`` /
``user_id`` / ``input_refs``) and minus ``_ENGINE_OWNED_FIELDS``, then injects the
fields it owns:

    image_path        comma-joined ABSOLUTE NFS paths, from _INPUT_FIELDS["image"]
                      ("image" is one of the list-valued fields, so multi-image
                      edit arrives as "a.png,b.png,c.png")
    save_result_path  absolute NFS path the engine must write
    prompt            caller text
    aspect_ratio      e.g. "16:9" (t2i; engine picks a discrete resolution)
    target_shape      [height, width] exact pixels, wins over aspect_ratio

``model`` is stripped as a control key, so it is optional here and backfilled
from the server's served model name, exactly as ``AudioTaskRequest`` /
``AudioGenTaskRequest`` do.

The status/result/cancel endpoints are shared and task-id keyed, so
``AudioTaskStatus`` / ``AudioTaskResponse`` are re-exported rather than
duplicated — only the submit endpoint is new.
"""

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# Re-export the shared status/response contract: the global status / result /
# cancel endpoints serve image tasks unchanged.
from vllm_omni.entrypoints.openai.protocol.audio_tasks import (  # noqa: F401
    AudioTaskResponse,
    AudioTaskStatus,
)

# Pixel budget used to turn a bare ``aspect_ratio`` into concrete dimensions.
# 1024x1024 is the native operating point of the models served through this
# endpoint (HunyuanImage-3.0's `<img_size_1024>` token, Qwen-Image, FLUX), so a
# ratio request lands on a familiar amount of work rather than an arbitrary one.
_RATIO_BASE_AREA = 1024 * 1024

# Round ratio-derived dimensions to this multiple. Diffusion pipelines require
# the latent grid to divide evenly: HunyuanImage-3.0 needs
# ``vae_downsample_factor * patch_size`` (16 or 32), Qwen-Image
# ``vae_scale_factor * 2`` (16), FLUX 16. 64 is a common multiple of all of
# them, so one value satisfies every model this endpoint serves.
_SIZE_ALIGNMENT = 64


def _align(value: float) -> int:
    """Round to the NEAREST ``_SIZE_ALIGNMENT`` multiple, never below one."""
    return max(_SIZE_ALIGNMENT, int(round(value / _SIZE_ALIGNMENT)) * _SIZE_ALIGNMENT)


def _align_down(value: float) -> int:
    """Round DOWN to a ``_SIZE_ALIGNMENT`` multiple, never below one.

    Used only to recover from nearest-rounding overshooting a pixel cap: since
    flooring can never grow a dimension, the product stays within the budget the
    scale was derived from.
    """
    return max(_SIZE_ALIGNMENT, int(value // _SIZE_ALIGNMENT) * _SIZE_ALIGNMENT)


class ImageTaskRequest(BaseModel):
    """Async image generation / editing task request.

    ``extra="allow"`` keeps this forward-compatible with pipeline params that are
    not typed here yet: unknown keys are forwarded to the diffusion path rather
    than dropped, so a new model knob works without a schema bump.
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
        description="Generation / edit instruction (aliases: prompt, input, text).",
    )
    negative_prompt: str | None = Field(default=None, description="Negative prompt.")
    task_id: str | None = Field(
        default=None,
        description="Client/facade-supplied task id; auto-generated if absent.",
    )
    save_result_path: str = Field(
        default="",
        description=(
            "Where the engine writes the result. Absolute -> written verbatim "
            "(facade injects an NFS absolute path); relative -> resolved under the "
            "output root; empty -> defaults to task_id; extension auto-added (.png)."
        ),
    )
    image_path: str | None = Field(
        default=None,
        description=(
            "Facade-materialized input image(s) as ABSOLUTE server paths, "
            "comma-separated for multi-image edit. Empty/absent means text-to-image."
        ),
    )
    image_mask_path: str | None = Field(
        default=None,
        description="Facade-materialized mask image (absolute server path); inpainting models only.",
    )

    # --------------------------------------------------------------- geometry
    target_shape: list[int] | None = Field(
        default=None,
        description="Exact output size as [height, width]; takes precedence over aspect_ratio.",
    )
    aspect_ratio: str | None = Field(
        default=None,
        description='Aspect ratio such as "16:9"; used when target_shape is absent.',
    )

    # ------------------------------------------- layered-model geometry (typed)
    # Declared rather than left to the untyped passthrough so the route can apply
    # the same validation the synchronous edit endpoint does: `resolution` must be
    # one of the supported layered values, `layers` has its own validator, the two
    # conflict with an explicit size, and `resolution` participates in the
    # max-generated-size check (which has a resolution-only branch that a request
    # setting only `resolution` would otherwise slip past).
    layers: int | None = Field(default=None, description="Layer count for layered models (Qwen-Image-Layered).")
    resolution: int | None = Field(
        default=None,
        description="Layered-model resolution; output size is resolution x resolution. Conflicts with target_shape.",
    )

    # ------------------------------------------------- shared diffusion params
    num_inference_steps: int | None = Field(default=None, description="Diffusion steps.")
    guidance_scale: float | None = Field(default=None, description="CFG guidance scale.")
    true_cfg_scale: float | None = Field(default=None, description="True CFG scale.")
    strength: float | None = Field(default=None, description="Img2img denoise strength.")
    seed: int | None = Field(default=None, description="Sampling seed.")
    n: int | None = Field(default=None, description="Images per prompt.")

    # ------------------------------------------------- HunyuanImage-3.0 prompt
    # Same knobs /v1/images/edits exposes. bot_task in particular is the lever
    # that decides whether the AR stage runs think/recaption at all, which
    # dominates end-to-end latency for that model.
    bot_task: str | None = Field(
        default=None,
        description="HunyuanImage3 prompt mode: none / think / recaption / think_recaption / vanilla.",
    )
    sys_type: str | None = Field(default=None, description="HunyuanImage3 system-prompt type override.")
    system_prompt: str | None = Field(default=None, description="HunyuanImage3 explicit system prompt.")

    # Keys consumed structurally by the route rather than forwarded verbatim.
    _ROUTE_OWNED_KEYS = (
        "model",
        "prompt",
        "task_id",
        "save_result_path",
        "image_path",
        "image_mask_path",
        "target_shape",
        "aspect_ratio",
        "n",
    )

    def input_image_paths(self) -> list[str]:
        """Split the comma-joined ``image_path`` into individual paths.

        The facade joins list-valued inputs with "," (its ``_LIST_INPUT_FIELDS``
        handling), so this is the inverse. Blank segments are dropped so a
        trailing comma is harmless.
        """
        raw = (self.image_path or "").strip()
        if not raw:
            return []
        return [part.strip() for part in raw.split(",") if part.strip()]

    def mask_image_path(self) -> str | None:
        value = (self.image_mask_path or "").strip()
        return value or None

    def parsed_aspect_ratio(self) -> tuple[int, int] | None:
        """Parse ``aspect_ratio`` ("W:H") into (w, h), or None if unusable.

        new-api normalizes whitespace before sending, but a direct caller may
        not, so parse defensively rather than trusting the format.
        """
        raw = (self.aspect_ratio or "").strip()
        if ":" not in raw:
            return None
        left, _, right = raw.partition(":")
        try:
            width_ratio, height_ratio = int(left.strip()), int(right.strip())
        except (TypeError, ValueError):
            return None
        if width_ratio <= 0 or height_ratio <= 0:
            return None
        return width_ratio, height_ratio

    def output_size(self, *, max_pixels: int | None = None) -> tuple[int | None, int | None]:
        """Resolve (width, height) from ``target_shape``, else from ``aspect_ratio``.

        ``target_shape`` is [height, width] — the facade builds it from new-api's
        ``targetShape = []int{h, w}`` — and wins when present, matching the
        facade's own documented precedence ("有 target_shape 时引擎会优先用后者").

        ``aspect_ratio`` has no consumer anywhere downstream: vllm-omni's image
        surface speaks only width/height, and the discrete resolution table the
        facade's comment refers to belongs to LightX2V. So the ratio has to be
        resolved to concrete dimensions HERE or it is silently lost — which is
        reachable in practice, because new-api sends ``aspect_ratio`` (and no
        ``target_shape``) whenever a text-to-image caller passes a bare ratio
        like "16:9", and HunyuanImage-3.0 serves t2i from the same deployment as
        i2i.

        ``max_pixels`` (the server's ``--max-generated-image-size``) caps the
        budget rather than triggering a rejection: the caller gave a shape, not
        an area, so the area is OUR choice and must fit the operator's limit.
        An explicit oversized ``target_shape`` is a different matter and is left
        to the route's size check to reject.

        Returning (None, None) means "auto": for editing that resolves to the
        input image's size, and for HunyuanImage-3.0 it keeps the AR-predicted
        ``<img_ratio_*>`` token in charge of the output shape.
        """
        shape = self.target_shape
        if shape and len(shape) == 2:
            try:
                height, width = int(shape[0]), int(shape[1])
            except (TypeError, ValueError):
                height = width = 0
            if height > 0 and width > 0:
                return width, height

        ratio = self.parsed_aspect_ratio()
        if ratio is None:
            return None, None
        width_ratio, height_ratio = ratio
        area = _RATIO_BASE_AREA
        if max_pixels is not None and max_pixels > 0:
            area = min(area, max_pixels)
        scale = (area / (width_ratio * height_ratio)) ** 0.5
        width, height = _align(width_ratio * scale), _align(height_ratio * scale)
        if max_pixels is not None and 0 < max_pixels < width * height:
            # Nearest-multiple rounding can round UP past the cap and defeat the
            # whole point of capping here: "16:9" under a 512x512 cap rounds to
            # 704x384 = 270,336 > 262,144, so the route's size check would 400 a
            # request this branch exists to make fit. Flooring only shrinks, so
            # width*height <= area <= max_pixels holds afterwards.
            #
            # The floor clamps at one _SIZE_ALIGNMENT step, so a pathological cap
            # below 64x64 = 4096 pixels still cannot be satisfied; the route's
            # 400 is the correct answer for that configuration.
            width, height = _align_down(width_ratio * scale), _align_down(height_ratio * scale)
        return width, height

    def diffusion_extra_body(self) -> dict[str, Any]:
        """Build the ``extra_body`` for the diffusion chat path.

        Typed fields win over same-named passthrough extras, so a stray extra
        cannot shadow a validated value.
        """
        extra: dict[str, Any] = {}
        # Forward unknown keys first (extra="allow" forward-compat), minus the
        # ones the route handles structurally.
        for key, value in (self.model_extra or {}).items():
            if key not in self._ROUTE_OWNED_KEYS:
                extra[key] = value
        for key in (
            "negative_prompt",
            "num_inference_steps",
            "guidance_scale",
            "true_cfg_scale",
            "strength",
            "seed",
            "bot_task",
            "sys_type",
            "system_prompt",
            # Typed, but NOT route-owned: the route only validates them, the
            # diffusion pipeline consumes them. Declaring a field moves it out of
            # model_extra, so anything left off this list is silently dropped —
            # which is exactly what happened when layers/resolution were first
            # given types.
            "layers",
            "resolution",
        ):
            value = getattr(self, key, None)
            if value is not None:
                extra[key] = value
        return extra
