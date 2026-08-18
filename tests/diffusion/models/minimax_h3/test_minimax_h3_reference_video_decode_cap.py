# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How much of a reference video survives the HTTP layer's decode.

Every H3 request is snapped up to the next ``17 * n + 5``, and the pipeline
conditions on the reference against that *aligned* count. The route decoded to
the count the caller asked for instead, so a 120-frame request produced a
120-frame reference for a 124-frame clip — and the official encoder then snaps
its VAE input DOWN to the previous multiple, 107, discarding a whole 17-frame
chunk of conditioning that the source video actually had.

Narrow to reach and quiet when reached: it needs ``/v1/videos``, a video
reference that decodes to frames rather than to a path (inline bytes, or a URL
that is not a downloadable source), and an explicit ``num_frames``/``seconds``.
Nothing errors. The clip is simply conditioned on less than it was given.

The fix is *which layer is asked*, not the arithmetic. The neighbouring
``reference_video_decode_spec`` classmethod cannot answer this: it is handed
``(num_frames, extra_args)`` — a request, with no config — so it can express a
rule about the model but never about the instance, and the contract is an
instance property. A capability probe reading ``od_config`` can, which makes
this the fourth of its kind and the same shape as the other three.

Legacy is deliberately left alone. It aligns too, but fills the target by
repeating frame slots instead of truncating, so the same shortfall costs a
slot-mapping shift rather than a dropped chunk — and changing what legacy
decodes changes what production generates.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _config(contract=None, model="MiniMaxH3Pipeline"):
    return SimpleNamespace(
        model_class_name=model,
        minimax_h3_inference_contract=contract,
        minimax_h3_admission_policy=None,
        diffusion_runtime_environ=None,
    )


class _EngineClient:
    def __init__(self, od_config):
        self._od_config = od_config

    def get_diffusion_od_config(self):
        return self._od_config


def _handler(od_config):
    from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo

    handler = object.__new__(OmniOpenAIServingVideo)
    handler._engine_client = _EngineClient(od_config)
    return handler


# ------------------------------------------------------------------ the capability


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (120, 124),  # the case that silently lost a chunk
        (124, 124),  # already on the lattice: unchanged
        (346, 362),
        (362, 362),
    ],
)
def test_official_decodes_to_the_count_the_pipeline_will_target(requested, expected):
    from vllm_omni.diffusion.model_metadata import reference_video_decode_frame_cap

    cap = reference_video_decode_frame_cap("MiniMaxH3Pipeline", _config("official_diffusers_v1"), num_frames=requested)
    assert cap == expected


def test_the_cap_is_exactly_what_the_pipeline_aligns_to():
    """Not a second implementation of the formula — the same function.

    The defect class this belongs to is "ported the formula, not the call
    order", so a hand-rolled ``+17`` here would be the identical mistake one
    layer up.
    """
    from vllm_omni.diffusion.model_metadata import reference_video_decode_frame_cap
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_align_frame_count

    for requested in (108, 119, 120, 200, 345, 361):
        assert reference_video_decode_frame_cap(
            "MiniMaxH3Pipeline", _config("official_diffusers_v1"), num_frames=requested
        ) == minimax_h3_align_frame_count(requested)


@pytest.mark.parametrize("contract", [None, "legacy"])
def test_legacy_decodes_exactly_what_it_decoded_before(contract):
    from vllm_omni.diffusion.model_metadata import reference_video_decode_frame_cap

    for requested in (120, 124, 209, 384):
        assert (
            reference_video_decode_frame_cap("MiniMaxH3Pipeline", _config(contract), num_frames=requested) == requested
        )


def test_other_models_are_untouched():
    from vllm_omni.diffusion.model_metadata import reference_video_decode_frame_cap

    for model in ("WanImageToVideoPipeline", "Cosmos3OmniDiffusersPipeline", None):
        assert reference_video_decode_frame_cap(model, _config(model=model), num_frames=120) == 120


def test_no_requested_count_stays_no_cap():
    """``None`` means "keep the whole reference" and must not become a number."""
    from vllm_omni.diffusion.model_metadata import reference_video_decode_frame_cap

    assert reference_video_decode_frame_cap("MiniMaxH3Pipeline", _config("official_diffusers_v1"), num_frames=None) is (
        None
    )


def test_an_unusable_contract_keeps_the_historical_cap():
    from vllm_omni.diffusion.model_metadata import reference_video_decode_frame_cap

    assert reference_video_decode_frame_cap("MiniMaxH3Pipeline", _config("nonsense"), num_frames=120) == 120


# --------------------------------------------------------- what the serving layer answers


def test_the_serving_layer_answers_from_the_config_alone():
    assert _handler(_config("official_diffusers_v1")).reference_video_decode_frame_cap(120) == 124
    assert _handler(_config("legacy")).reference_video_decode_frame_cap(120) == 120


def test_an_unreachable_config_changes_nothing():
    """A probe that cannot see the engine keeps the route's own fallback."""
    assert _handler(None).reference_video_decode_frame_cap(120) == 120


# ------------------------------------------------------------- what the route builds


def _request(num_frames):
    from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest

    return VideoGenerationRequest(prompt="a cat on a skateboard", num_frames=num_frames)


def test_the_route_asks_the_handler_and_uses_the_answer():
    from vllm_omni.entrypoints.openai import api_server

    spec = api_server._reference_video_decode_spec(_request(120), None, _handler(_config("official_diffusers_v1")))
    assert (spec.max_frames, spec.keep) == (124, "first")


def test_the_route_without_a_handler_is_the_old_behaviour():
    """Every caller that never learned about this keeps working unchanged."""
    from vllm_omni.entrypoints.openai import api_server

    spec = api_server._reference_video_decode_spec(_request(120), None)
    assert spec.max_frames == 120


def test_a_model_that_answers_for_itself_still_wins(monkeypatch):
    """Cosmos-3's classmethod encodes a rule about the model, not the instance.

    Its answer is taken before the handler is consulted, so wiring the handler
    in cannot have moved a model that was already answering.
    """
    from vllm_omni.diffusion.models.interface import ReferenceVideoDecodeSpec
    from vllm_omni.entrypoints.openai import api_server

    class _Model:
        @classmethod
        def reference_video_decode_spec(cls, *, num_frames, extra_args):
            del num_frames, extra_args
            return ReferenceVideoDecodeSpec(max_frames=17, keep="last")

    monkeypatch.setattr(api_server, "_diffusion_model_classes", lambda _stage_configs: [_Model])
    spec = api_server._reference_video_decode_spec(_request(120), None, _handler(_config("official_diffusers_v1")))

    assert (spec.max_frames, spec.keep) == (17, "last")
