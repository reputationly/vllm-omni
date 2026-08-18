# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether an ordered ``references`` list can be admitted at the HTTP boundary.

``POST /v1/tasks/video/`` *manufactures* ``reference_order`` from every
non-empty ``references`` list, and a ``legacy`` pipeline rejects any explicit
order outright. Since ``legacy`` is the DEFAULT contract, the plainest ordered
request on a stock deployment used to return 200 PENDING, take a queue slot, and
fail inside the job with an engine message the caller never asked for.

The route already refused a *caller-written* ``reference_order`` with a 400, so
it knew the field was dangerous on this surface — it just validated the only
half it had not created itself.

This is the third time a contract-dependent decision was made outside the
process that resolves the contract (the other two: the front-end config
projection, and the cross-process config view). The capability probe below is
the same shape as ``reference_images_bind_output_canvas``, deliberately: the
answer has to be derivable from an ``od_config`` alone, because that is all the
serving layer has.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Config:
    """The attributes the capability probe reads off a diffusion config."""

    def __init__(self, model_class_name, contract=None):
        self.model_class_name = model_class_name
        self.minimax_h3_inference_contract = contract
        self.minimax_h3_admission_policy = None


class _EngineClient:
    def __init__(self, od_config):
        self._od_config = od_config

    def get_diffusion_od_config(self):
        return self._od_config


def _handler(od_config):
    """A serving object with only the engine client wired, which is all this reads."""
    from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo

    handler = object.__new__(OmniOpenAIServingVideo)
    handler._engine_client = _EngineClient(od_config)
    return handler


# --------------------------------------------------------------- the capability


def test_models_that_do_not_read_the_field_are_unaffected():
    """The order only means something where a pipeline reads it."""
    from vllm_omni.diffusion.model_metadata import honours_explicit_reference_order

    for model in ("WanImageToVideoPipeline", "WanPipeline", None, "SomeUnknownPipeline"):
        assert honours_explicit_reference_order(model, _Config(model)) is True


def test_legacy_h3_cannot_honour_an_order():
    from vllm_omni.diffusion.model_metadata import honours_explicit_reference_order

    for model in ("MiniMaxH3Pipeline", "MiniMaxH3ModularPipeline"):
        assert honours_explicit_reference_order(model, _Config(model, "legacy")) is False
        # No contract configured at all IS legacy — the default deployment, and
        # the one the bug actually shipped on.
        assert honours_explicit_reference_order(model, _Config(model, None)) is False


def test_the_official_contract_carries_the_order_end_to_end():
    from vllm_omni.diffusion.model_metadata import honours_explicit_reference_order

    for model in ("MiniMaxH3Pipeline", "MiniMaxH3ModularPipeline"):
        assert honours_explicit_reference_order(model, _Config(model, "official_diffusers_v1")) is True


def test_a_bad_contract_does_not_turn_into_a_400_about_reference_order():
    """An unusable contract is the pipeline's error, with the pipeline's message.

    Answering "cannot honour" here would replace a startup-config diagnostic with
    a 400 that blames the caller's references.
    """
    from vllm_omni.diffusion.model_metadata import honours_explicit_reference_order

    assert honours_explicit_reference_order("MiniMaxH3Pipeline", _Config("MiniMaxH3Pipeline", "nonsense")) is True


# ------------------------------------------------------ what the route can see


def test_the_serving_layer_answers_from_the_config_alone():
    """No GPU, no engine start: the route asks this while the request is alive."""
    assert _handler(_Config("MiniMaxH3Pipeline", "legacy")).honours_explicit_reference_order is False
    assert _handler(_Config("MiniMaxH3Pipeline", "official_diffusers_v1")).honours_explicit_reference_order is True


def test_an_unreachable_config_never_invents_a_rejection():
    """A probe that cannot see the engine must not manufacture a 400."""
    assert _handler(None).honours_explicit_reference_order is True


# --------------------------------------------------- the condition the route checks


def test_a_plain_ordered_request_is_exactly_the_case_that_used_to_slip_through():
    """Both halves of the route's guard, on the request that shipped broken.

    The order is non-empty (the route derives it from ``references``) and the
    default instance cannot honour it — which is precisely the pair that
    previously produced PENDING-then-FAILED and now produces a 400 before
    ``reserve()``.
    """
    from vllm_omni.diffusion.model_metadata import honours_explicit_reference_order
    from vllm_omni.entrypoints.openai.protocol.video_tasks import VideoTaskRequest

    request = VideoTaskRequest(
        prompt="a cat on a skateboard",
        references=[
            {"type": "image", "path": "/nfs/a.png"},
            {"type": "video", "path": "/nfs/b.mp4"},
        ],
    )

    assert request.reference_order() == [("image", 0), ("video", 0)]
    assert honours_explicit_reference_order("MiniMaxH3Pipeline", _Config("MiniMaxH3Pipeline", None)) is False


def test_the_bucketed_fields_stay_admissible_on_every_contract():
    """The escape hatch the 400 points at has to actually exist.

    A request built from ``image_path`` / ``video_path`` expresses no order, so
    there is nothing for a legacy instance to refuse — it canonicalizes by
    modality and always did.
    """
    from vllm_omni.entrypoints.openai.protocol.video_tasks import VideoTaskRequest

    request = VideoTaskRequest(prompt="p", image_path="/nfs/a.png", video_path="/nfs/b.mp4")

    assert request.reference_order() == []
    assert request.to_video_request().reference_order is None
