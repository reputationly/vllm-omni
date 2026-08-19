# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VideoTaskRequest path/param resolution (CPU-only).

Covers the GPUStack facade contract for ``POST /v1/tasks/video/``: media inputs
arrive as comma-joined ABSOLUTE server paths (never bytes or URLs), keyframe
images are ordered ``image_path`` then ``last_frame_path``, and every generation
knob rides through untyped extras into ``VideoGenerationRequest`` — which is
where they get range-checked, so a bad one is a submit-time error rather than a
task that fails on the GPU.
"""

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.video_tasks import VideoTaskRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _req(**kwargs) -> VideoTaskRequest:
    kwargs.setdefault("prompt", "a cat on a skateboard")
    return VideoTaskRequest(**kwargs)


# -------------------------------------------------------------------- prompt


@pytest.mark.parametrize("key", ["prompt", "input", "text"])
def test_prompt_aliases(key):
    """The facade sends "prompt"; the audio-task spellings stay usable so a
    direct caller can reuse one body shape across kinds."""
    assert VideoTaskRequest(**{key: "hello"}).prompt == "hello"


def test_prompt_is_required():
    with pytest.raises(ValidationError):
        VideoTaskRequest(image_path="/nfs/a.png")


# --------------------------------------------------------------- media paths


def test_no_media_is_text_to_video():
    req = _req()
    assert req.reference_image_paths() == []
    assert req.reference_video_paths() == []
    assert req.reference_audio_paths() == []


def test_image_path_is_comma_joined():
    """The facade joins list-valued inputs with "," (_LIST_INPUT_FIELDS)."""
    assert _req(image_path="/nfs/a.png,/nfs/b.png").reference_image_paths() == ["/nfs/a.png", "/nfs/b.png"]


@pytest.mark.parametrize(
    "raw",
    ["", "   ", ",", "/nfs/a.png,", " /nfs/a.png , "],
)
def test_blank_segments_are_dropped(raw):
    paths = _req(image_path=raw).reference_image_paths()
    assert paths in ([], ["/nfs/a.png"])


def test_last_frame_is_ordered_after_image():
    """MiniMax-H3 FL2VA pairs the Nth image with the Nth frame_index, so the
    facade's frame_indices=[0, -1] only lines up when the first-frame image comes
    first. An l2va request sends its single image in image_path with
    frame_indices=[-1]; the order rule is what keeps the two cases distinct."""
    req = _req(image_path="/nfs/first.png", last_frame_path="/nfs/last.png")
    assert req.reference_image_paths() == ["/nfs/first.png", "/nfs/last.png"]


def test_last_frame_alone_is_the_only_image():
    assert _req(last_frame_path="/nfs/last.png").reference_image_paths() == ["/nfs/last.png"]


def test_video_and_audio_paths_are_multi_valued():
    """H3 Ref2VA takes up to 3 reference videos and 3 standalone audio refs; the
    facade comma-joins them into one field each."""
    req = _req(video_path="/nfs/a.mp4,/nfs/b.mp4", audio_path="/nfs/a.wav,/nfs/b.wav")
    assert req.reference_video_paths() == ["/nfs/a.mp4", "/nfs/b.mp4"]
    assert req.reference_audio_paths() == ["/nfs/a.wav", "/nfs/b.wav"]


# ------------------------------------------------------- unsupported inputs

# Spelled out rather than read off VideoTaskRequest._UNSUPPORTED_REFERENCE_KEYS:
# reflecting the tuple under test would make these assertions tautological.
_UNSUPPORTED = [
    "image_reference",
    "video_reference",
    "audio_reference",
    "input_reference",
    "input_references",
]


@pytest.mark.parametrize("key", _UNSUPPORTED)
def test_byte_and_url_reference_keys_are_reported(key):
    """These belong to the multipart endpoints. Whether typed on
    VideoGenerationRequest or absent from it entirely, forwarding them ends in
    the same silent no-op — the route rejects instead."""
    assert _req(**{key: {"image_url": "https://example.com/a.png"}}).unsupported_reference_keys() == [key]


def test_absent_reference_keys_are_not_reported():
    assert _req(image_path="/nfs/a.png").unsupported_reference_keys() == []


@pytest.mark.parametrize("key", _UNSUPPORTED)
def test_reference_keys_nested_in_extra_params_are_reported(key):
    """extra_params is merged into extra_args, where every consumer reads by
    explicit key — so a reference nested there is read by nobody and the task
    would silently COMPLETE as text-to-video."""
    req = _req(extra_params={key: {"image_url": "https://example.com/a.png"}})
    assert req.unsupported_reference_keys() == [f"extra_params.{key}"]


def test_unrelated_extra_params_keys_stay_opaque():
    """Guardrail: this check must not grow into general key policing —
    extra_params is the sanctioned passthrough for undeclared engine params."""
    assert _req(extra_params={"task": "ref2va", "frame_indices": [0, -1]}).unsupported_reference_keys() == []


def test_unsupported_reference_keys_are_not_forwarded():
    video_request = _req(image_reference={"image_url": "https://example.com/a.png"}).to_video_request()
    assert video_request.image_reference is None


def test_multipart_only_keys_never_reach_the_video_request():
    """input_reference is not a VideoGenerationRequest field at all, so there is
    no attribute to come back None: pydantic's default extra="ignore" would drop
    it wordlessly. That is exactly why the route must reject it before
    to_video_request() ever runs."""
    video_request = _req(input_reference="/nfs/a.png").to_video_request()
    assert not hasattr(video_request, "input_reference")


# ---------------------------------------------------------- to_video_request


def test_generation_params_ride_through_extras():
    """Geometry/steps/seed are NOT re-declared on VideoTaskRequest: extra="allow"
    collects them and VideoGenerationRequest is the single schema that types
    them."""
    video_request = _req(
        width=864,
        height=480,
        aspect_ratio="16:9",
        num_inference_steps=20,
        flow_shift=12.0,
        seed=7,
        quality="high",
    ).to_video_request()

    assert (video_request.width, video_request.height) == (864, 480)
    assert video_request.aspect_ratio == "16:9"
    assert video_request.num_inference_steps == 20
    assert video_request.flow_shift == 12.0
    assert video_request.seed == 7
    assert video_request.quality == "high"


@pytest.mark.parametrize("steps", [1, 4, 8, 20, 25, 30, 50, 200])
def test_request_can_override_inference_steps_with_any_supported_positive_integer(steps):
    """Base and distilled checkpoints choose different defaults, but the HTTP
    contract deliberately remains request-overridable.  The product UI may
    advertise a few recommended values; it must not turn those suggestions into
    an engine-side enum (25 is the regression case)."""
    assert _req(num_inference_steps=steps).to_video_request().num_inference_steps == steps


def test_extra_params_survives_as_a_nested_object():
    """H3 selects its task through extra_params.task + extra_params.frame_indices
    (there is no top-level `task` field); the facade backfills both."""
    video_request = _req(extra_params={"task": "fl2va", "frame_indices": [0, -1], "duration": 8.0}).to_video_request()
    assert video_request.extra_params == {"task": "fl2va", "frame_indices": [0, -1], "duration": 8.0}


def test_none_extras_are_dropped_from_fields_set():
    """_run_and_extract gates most knobs on model_fields_set, so forwarding an
    explicit None would mark the field "provided" and overwrite the engine's own
    default with nothing."""
    video_request = _req(guidance_scale=None, generate_sound=None).to_video_request()
    assert "guidance_scale" not in video_request.model_fields_set
    assert "generate_sound" not in video_request.model_fields_set


def test_untouched_knobs_stay_unset():
    video_request = _req().to_video_request()
    assert video_request.model_fields_set == {"prompt"}


def test_route_owned_keys_are_not_forwarded():
    """A caller that sends both spellings of the prompt leaves the unmatched one
    in model_extra; it must not reach the generation request."""
    req = VideoTaskRequest(prompt="wins", input="stray", task_id="t-1", save_result_path="/nfs/out.mp4")
    video_request = req.to_video_request()
    assert video_request.prompt == "wins"
    assert (video_request.model_extra or {}) == {}


def test_model_is_forwarded_only_when_set():
    assert _req().to_video_request().model is None
    assert _req(model="minimax-h3-fl2va").to_video_request().model == "minimax-h3-fl2va"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_inference_steps": 0},  # ge=1
        {"num_inference_steps": 500},  # le=200
        {"width": 0},  # ge=1
        {"num_outputs_per_prompt": 11},  # le=10
        {"quality": "ultra"},  # not a DIFFUSION_QUALITY_LEVELS value
        {"size": "480P"},  # SizeStr is WIDTHxHEIGHT, not a tier word
        {"seconds": "0"},  # SecondStr is a positive integer string
    ],
)
def test_out_of_range_params_raise_at_build_time(kwargs):
    """The route turns this into a 400. Validating at submit is the whole point:
    the caller gets an answer instead of a task that dies on the GPU, and a
    rejected request never takes a queue slot."""
    req = _req(**kwargs)
    with pytest.raises(ValidationError):
        req.to_video_request()


# --------------------------------------------------- hand-written reference_order


def test_a_hand_written_reference_order_is_reported():
    """``reference_order`` is derived from ``references``, never supplied.

    ``VideoTaskRequest`` forwards extras, so a caller can put it in the body and
    it rides through untouched. Nothing then keeps it consistent with the media
    that actually arrived in the buckets, and the first reader is serving —
    inside the background job, after ``reserve()`` has taken a queue slot. The
    route asks this question before that happens.
    """
    assert _req(reference_order=[{"type": "image", "index": 0}]).supplies_route_derived_order() is True
    assert _req(image_path="/nfs/a.png").supplies_route_derived_order() is False
    # Only a value counts: an explicit null expresses no order, same as absence.
    assert _req(reference_order=None).supplies_route_derived_order() is False


def test_a_hand_written_reference_order_never_reaches_the_generation_request():
    """Route-owned, so even if the 400 were missed it would not be forwarded.

    Belt and braces on purpose: a dropped order silently builds the video from
    the *canonical* bucket order, which is a different video, and that is the
    failure the route rejects rather than absorbs.
    """
    video_request = _req(image_path="/nfs/a.png", reference_order=[{"type": "video", "index": 7}]).to_video_request()
    assert video_request.reference_order is None
    assert (video_request.model_extra or {}) == {}


def test_the_derived_order_is_still_supplied_by_the_route():
    """The guard is about the *caller* writing it; the route still derives it."""
    req = VideoTaskRequest(
        prompt="p",
        references=[{"type": "video", "path": "/nfs/a.mp4"}, {"type": "image", "path": "/nfs/b.png"}],
    )
    assert req.supplies_route_derived_order() is False
    forwarded = req.to_video_request().reference_order
    assert [(entry.type, entry.index) for entry in forwarded] == [("video", 0), ("image", 0)]


@pytest.mark.parametrize(
    "entry",
    [
        {"index": 0},  # no modality at all
        {"type": "image"},  # no index at all
        {"type": "picture", "index": 0},  # not a modality the pipeline buckets
        {"type": "image", "index": "a"},  # serving uses it as a list subscript
        {"type": "image", "index": -1},  # negative subscripts wrap, silently
        {"type": "image", "index": 0, "path": "/nfs/a.png"},  # media rides in buckets
    ],
)
def test_malformed_order_entries_are_refused_by_the_schema(entry):
    """Typed entries, so a bad one is a ValidationError the route turns into 400.

    As a loose ``list[dict[str, Any]]`` every one of these passed submit and
    blew up later on ``entry["type"]`` / ``entry["index"]``, by which point the
    caller had a PENDING task instead of an answer.
    """
    from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest

    with pytest.raises(ValidationError):
        VideoGenerationRequest(prompt="p", reference_order=[entry])


def test_well_formed_order_entries_are_accepted():
    from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest

    request = VideoGenerationRequest(
        prompt="p", reference_order=[{"type": "audio", "index": 2}, {"type": "video", "index": 0}]
    )
    assert [(entry.type, entry.index) for entry in request.reference_order] == [("audio", 2), ("video", 0)]
