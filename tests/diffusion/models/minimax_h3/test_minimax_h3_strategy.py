# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup resolution of the MiniMax-H3 inference contract and admission policy."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_default_is_legacy_and_matches_production_today():
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    strategy = resolve_strategy(inference_contract=None, admission_policy=None, environ={})

    assert strategy.name == "legacy"
    assert strategy.is_official is False
    assert strategy.admission_policy == "production_safe_v1"
    # Every legacy value is the shipped behaviour; a change here is a regression,
    # not a refactor, so they are pinned individually rather than as a blob.
    assert strategy.rng_mode == "legacy"
    assert strategy.visual_condition_noise_shape_mode == "legacy_oversized_slice"
    assert strategy.fl2va_keyframe_resize_mode == "legacy_stretch"
    assert strategy.reference_image_geometry_mode == "legacy_canvas_prestretch"
    assert strategy.reference_order_mode == "legacy_bucket_canonicalization"
    assert strategy.reference_video_decode_mode == "legacy_h264_intermediate"
    assert strategy.reference_video_target_truncation is False
    assert strategy.reference_audio_resample_mode == "legacy_double_resample"
    assert strategy.reference_audio_target_truncation is False
    assert strategy.model_validation_semantics == "legacy"
    assert strategy.default_num_frames("t2va") == 209
    assert strategy.default_num_frames("fl2va") == 209
    assert strategy.default_num_frames("ref2va") == 124


def test_official_contract_resolves_every_field_to_the_oracle():
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    strategy = resolve_strategy(inference_contract="official_diffusers_v1", admission_policy=None, environ={})

    assert strategy.is_official
    assert strategy.rng_mode == "official_diffusers_v1"
    assert strategy.visual_condition_noise_shape_mode == "condition_shape"
    assert strategy.fl2va_keyframe_resize_mode == "official_cover_crop"
    assert strategy.reference_image_geometry_mode == "official_short_edge"
    assert strategy.reference_order_mode == "ordered_references"
    assert strategy.reference_video_decode_mode == "official_lossless_frames"
    assert strategy.reference_video_target_truncation is True
    assert strategy.reference_audio_resample_mode == "official_single_resample"
    assert strategy.reference_audio_target_truncation is True
    assert strategy.model_validation_semantics == "official"
    # The oracle's workflow default is 124 for every task, not just ref2va.
    assert {task: strategy.default_num_frames(task) for task in ("t2va", "fl2va", "ref2va")} == {
        "t2va": 124,
        "fl2va": 124,
        "ref2va": 124,
    }


def test_unknown_contract_fails_loudly_rather_than_falling_back():
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    with pytest.raises(ValueError, match="inference_contract"):
        resolve_strategy(inference_contract="official", admission_policy=None, environ={})
    with pytest.raises(ValueError, match="admission_policy"):
        resolve_strategy(inference_contract="legacy", admission_policy="anything", environ={})


@pytest.mark.parametrize(
    "env_key",
    [
        # The short edge is deliberately absent: it scales the reference without
        # reshaping it, and the product has decided detail is negotiable while
        # proportions are not. These two still change the shape.
        "VLLM_OMNI_H3_REF_IMAGE_NO_UPSCALE",
        "VLLM_OMNI_H3_REF_IMAGE_MAX_PIXELS",
    ],
)
def test_official_contract_refuses_shape_changing_reference_switches(env_key):
    """A switch that reshapes the reference would make the run neither contract.

    Failing at startup is the point: the alternative is a result that nothing in
    the output would reveal as non-official.
    """
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    with pytest.raises(ValueError, match=env_key):
        resolve_strategy(
            inference_contract="official_diffusers_v1",
            admission_policy=None,
            environ={env_key: "768"},
        )


def test_legacy_keeps_the_reference_image_switches_usable():
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    strategy = resolve_strategy(
        inference_contract="legacy",
        admission_policy=None,
        environ={"VLLM_OMNI_H3_REF_IMAGE_SHORT_EDGE": "768"},
    )
    assert strategy.name == "legacy"


def test_describe_reports_the_resolved_contract():
    """Startup log and result metadata read from this, so it has to be complete."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    described = resolve_strategy(
        inference_contract="official_diffusers_v1", admission_policy="parity_fixture_v1", environ={}
    ).describe()

    assert described["inference_contract"] == "official_diffusers_v1"
    assert described["admission_policy"] == "parity_fixture_v1"
    # No field may be missing, or an operator cannot tell from the log which
    # contract a stored artifact came from.
    from vllm_omni.diffusion.models.minimax_h3.strategy import MiniMaxH3InferenceStrategy

    expected = {field for field in MiniMaxH3InferenceStrategy.__dataclass_fields__ if field != "name"}
    assert expected | {"inference_contract"} == set(described)


def test_env_selects_the_contract_when_config_is_silent():
    """Startup-level env fallback, for deployments without the config field yet."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    strategy = resolve_strategy(
        inference_contract=None,
        admission_policy=None,
        environ={"VLLM_OMNI_H3_INFERENCE_CONTRACT": "official_diffusers_v1"},
    )
    assert strategy.is_official


def test_explicit_config_beats_the_env_fallback():
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    strategy = resolve_strategy(
        inference_contract="legacy",
        admission_policy=None,
        environ={"VLLM_OMNI_H3_INFERENCE_CONTRACT": "official_diffusers_v1"},
    )
    assert strategy.name == "legacy"


def test_model_level_validation_follows_the_contract():
    """Duration and reference ratio are model semantics; both move with it."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    legacy = resolve_strategy(inference_contract="legacy", admission_policy=None, environ={})
    official = resolve_strategy(inference_contract="official_diffusers_v1", admission_policy=None, environ={})

    # The oracle generates 5..15 s; vLLM's entry widened that to 2..16.
    assert legacy.output_duration_seconds == (2.0, 16.0)
    assert official.output_duration_seconds == (5.0, 15.0)
    # The oracle accepts 1:4..4:1; vLLM's entry narrowed it to 0.4..2.5.
    assert legacy.reference_image_aspect_ratio_range == (0.4, 2.5)
    assert official.reference_image_aspect_ratio_range == (0.25, 4.0)


def test_official_reference_ratio_range_admits_what_the_oracle_admits():
    """A 3:1 reference is legal for the oracle and illegal for legacy."""
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape
    from vllm_omni.errors import OmniClientError

    tall = Image.new("RGB", (3072, 1024))
    with pytest.raises(OmniClientError, match=r"\[0.4, 2.5\]"):
        _reference_image_shape(tall, aspect_ratio_range=(0.4, 2.5))
    assert _reference_image_shape(tall, aspect_ratio_range=(0.25, 4.0)) == (6144, 2048)


def test_parity_policy_allows_only_the_short_edge_and_records_it():
    """An oracle that cannot run is not a stricter test than a smaller one.

    The released short edge of 2048 makes the official conditioner's ref2va
    presentation too large to encode on the hardware the oracle has, so a parity
    fixture may lower it — and only it. The resolved value is reported, so a
    dump taken at another geometry can never be read as the released one.
    """
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    strategy = resolve_strategy(
        inference_contract="official_diffusers_v1",
        admission_policy="parity_fixture_v1",
        environ={"VLLM_OMNI_H3_REF_IMAGE_SHORT_EDGE": "1024"},
    )
    assert strategy.is_official
    assert strategy.reference_image_short_edge == 1024
    assert strategy.describe()["reference_image_short_edge"] == 1024

    # Everything else about the geometry stays fixed, even here.
    for env_key in ("VLLM_OMNI_H3_REF_IMAGE_NO_UPSCALE", "VLLM_OMNI_H3_REF_IMAGE_MAX_PIXELS"):
        with pytest.raises(ValueError, match="relaxes only"):
            resolve_strategy(
                inference_contract="official_diffusers_v1",
                admission_policy="parity_fixture_v1",
                environ={env_key: "1"},
            )


def test_shape_changing_switches_stay_parity_only_in_production():
    """Scale is negotiable in production; shape is not.

    Superseded an earlier rule that refused the short edge in production too.
    That rule was written before the stretched-reference defect was confirmed
    visually and before it was noticed that preserving the ratio at a smaller
    short edge costs *fewer* tokens than today's stretched 2048 — so refusing
    the knob bought nothing and blocked the cheaper correct option.
    """
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    with pytest.raises(ValueError, match="parity_fixture_v1"):
        resolve_strategy(
            inference_contract="official_diffusers_v1",
            admission_policy="production_safe_v1",
            environ={"VLLM_OMNI_H3_REF_IMAGE_MAX_PIXELS": "1032192"},
        )


def test_parity_short_edge_must_be_a_usable_geometry():
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    for bad in ("1000", "0", "-32", "many"):
        with pytest.raises(ValueError):
            resolve_strategy(
                inference_contract="official_diffusers_v1",
                admission_policy="parity_fixture_v1",
                environ={"VLLM_OMNI_H3_REF_IMAGE_SHORT_EDGE": bad},
            )


def test_default_official_short_edge_is_the_released_one():
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    for policy in ("production_safe_v1", "parity_fixture_v1"):
        strategy = resolve_strategy(inference_contract="official_diffusers_v1", admission_policy=policy, environ={})
        assert strategy.reference_image_short_edge == 2048


def test_reference_geometry_is_selectable_independently_of_the_contract():
    """Two variables have to be separable to attribute an observed difference.

    `official` changes the RNG stream and the reference geometry together, so a
    quality difference between it and `legacy` cannot be assigned to either. A
    legacy-RNG run with the fixed geometry is the control that separates them —
    and, separately, it is the smallest change that fixes the confirmed
    squashed-reference defect without altering every seed's output.
    """
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    control = resolve_strategy(
        inference_contract="legacy",
        admission_policy=None,
        environ={"VLLM_OMNI_H3_REF_IMAGE_GEOMETRY": "official_short_edge"},
    )
    assert control.name == "legacy"
    assert control.rng_mode == "legacy"  # unchanged
    assert control.reference_image_geometry_mode == "official_short_edge"  # fixed
    assert control.describe()["reference_image_geometry_mode"] == "official_short_edge"

    # And the reverse, for an official run that keeps the old geometry.
    reverse = resolve_strategy(
        inference_contract="official_diffusers_v1",
        admission_policy=None,
        environ={"VLLM_OMNI_H3_REF_IMAGE_GEOMETRY": "legacy_canvas_prestretch"},
    )
    assert reverse.rng_mode == "official_diffusers_v1"
    assert reverse.reference_image_geometry_mode == "legacy_canvas_prestretch"


def test_unknown_geometry_mode_is_rejected():
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    with pytest.raises(ValueError, match="REF_IMAGE_GEOMETRY"):
        resolve_strategy(
            inference_contract="legacy",
            admission_policy=None,
            environ={"VLLM_OMNI_H3_REF_IMAGE_GEOMETRY": "official"},
        )


def test_short_edge_is_a_production_knob_when_the_aspect_ratio_is_preserved():
    """Detail is negotiable; the reference's proportions are not.

    A smaller short edge trades sharpness for tokens and keeps the reference's
    shape. The canvas pre-stretch trades the shape itself, which is the defect
    the product refuses to ship — so the short edge is allowed in production
    once the geometry is preserved, while the switches that would distort stay
    parity-only. Preserving the ratio at a smaller short edge is also *cheaper*
    than today's stretched 2048, so correctness here does not cost tokens.
    """
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    strategy = resolve_strategy(
        inference_contract="official_diffusers_v1",
        admission_policy="production_safe_v1",
        environ={"VLLM_OMNI_H3_REF_IMAGE_SHORT_EDGE": "1024"},
    )
    assert strategy.reference_image_short_edge == 1024
    assert strategy.reference_image_geometry_mode == "official_short_edge"

    # Refused when combined with a switch that would distort the geometry.
    with pytest.raises(ValueError):
        resolve_strategy(
            inference_contract="official_diffusers_v1",
            admission_policy="production_safe_v1",
            environ={
                "VLLM_OMNI_H3_REF_IMAGE_SHORT_EDGE": "1024",
                "VLLM_OMNI_H3_REF_IMAGE_NO_UPSCALE": "1",
            },
        )


def test_dropping_the_prestretch_widens_the_admission_envelope():
    """Legacy's [0.4, 2.5] only held because the pre-stretch hid the raw ratio.

    Found in the isolation matrix: a legacy engine with the geometry override
    rejected the very 1664x656 (2.54) reference that full-legacy accepts, because
    full-legacy validates the pre-stretched 1344x768 canvas (1.75) instead of the
    image the user sent. Preserving the aspect ratio and keeping the old envelope
    would turn a silent squash into a hard 400 for real inputs, so the envelope
    follows the geometry. It gates admission only, so widening it cannot change
    a result that was already admitted.
    """
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    stretched = resolve_strategy(inference_contract="legacy", admission_policy=None, environ={})
    assert stretched.reference_image_aspect_ratio_range == (0.4, 2.5)
    assert 1664 / 656 > stretched.reference_image_aspect_ratio_range[1]

    preserved = resolve_strategy(
        inference_contract="legacy",
        admission_policy=None,
        environ={"VLLM_OMNI_H3_REF_IMAGE_GEOMETRY": "official_short_edge"},
    )
    assert preserved.reference_image_aspect_ratio_range == (0.25, 4.0)
    low, high = preserved.reference_image_aspect_ratio_range
    assert low < 1664 / 656 < high

    # Explicitly asking for the legacy geometry keeps the legacy envelope.
    explicit = resolve_strategy(
        inference_contract="legacy",
        admission_policy=None,
        environ={"VLLM_OMNI_H3_REF_IMAGE_GEOMETRY": "legacy_canvas_prestretch"},
    )
    assert explicit.reference_image_aspect_ratio_range == (0.4, 2.5)


def test_the_legacy_short_edge_knob_is_not_shadowed_by_the_strategy():
    """A knob that silently stops working is worse than one that was removed.

    The pipeline resolves the short edge as `short_edge or env`, so once the
    strategy started supplying its own 2048 — a truthy value — the env could
    never be reached under legacy. Deploy files kept setting it and it kept
    reading as live while doing nothing. It is parsed here instead, for every
    contract, so the resolved strategy carries what the operator asked for.
    """
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    default = resolve_strategy(inference_contract="legacy", admission_policy=None, environ={})
    assert default.reference_image_short_edge == 2048

    overridden = resolve_strategy(
        inference_contract="legacy",
        admission_policy=None,
        environ={"VLLM_OMNI_H3_REF_IMAGE_SHORT_EDGE": "1024"},
    )
    assert overridden.reference_image_short_edge == 1024

    # Validated identically regardless of contract.
    for contract in ("legacy", "official_diffusers_v1"):
        with pytest.raises(ValueError, match="multiple of 32"):
            resolve_strategy(
                inference_contract=contract,
                admission_policy=None,
                environ={
                    "VLLM_OMNI_H3_REF_IMAGE_SHORT_EDGE": "1000",
                    "VLLM_OMNI_H3_REF_IMAGE_GEOMETRY": "official_short_edge",
                },
            )


def test_the_two_noise_axes_can_be_selected_independently():
    """`official` moves the RNG and the condition-noise shape together.

    Five arms on 2026-08-17 showed hair length tracking the contract axis exactly
    and the geometry axis not at all, which narrows the identity drift to these
    two — and no further, because they always moved together. Each has to be
    selectable on its own or the attribution stops here.
    """
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    official_rng_legacy_shape = resolve_strategy(
        inference_contract="legacy",
        admission_policy=None,
        environ={"VLLM_OMNI_H3_RNG_MODE": "official_diffusers_v1"},
    )
    assert official_rng_legacy_shape.rng_mode == "official_diffusers_v1"
    assert official_rng_legacy_shape.visual_condition_noise_shape_mode == "legacy_oversized_slice"

    legacy_rng_official_shape = resolve_strategy(
        inference_contract="official_diffusers_v1",
        admission_policy=None,
        environ={"VLLM_OMNI_H3_RNG_MODE": "legacy"},
    )
    assert legacy_rng_official_shape.rng_mode == "legacy"
    assert legacy_rng_official_shape.visual_condition_noise_shape_mode == "condition_shape"

    shape_only = resolve_strategy(
        inference_contract="legacy",
        admission_policy=None,
        environ={"VLLM_OMNI_H3_CONDITION_NOISE_SHAPE": "condition_shape"},
    )
    assert shape_only.visual_condition_noise_shape_mode == "condition_shape"
    assert shape_only.rng_mode == "legacy"

    # The resolved value is what `describe()` reports, so a run cannot claim a
    # contract whose noise axes were overridden without saying so.
    assert official_rng_legacy_shape.describe()["rng_mode"] == "official_diffusers_v1"

    for key, value in (
        ("VLLM_OMNI_H3_RNG_MODE", "nonsense"),
        ("VLLM_OMNI_H3_CONDITION_NOISE_SHAPE", "nonsense"),
    ):
        with pytest.raises(ValueError, match="must be one of"):
            resolve_strategy(inference_contract="legacy", admission_policy=None, environ={key: value})
