# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The contract has to travel from deploy YAML to the H3 pipeline.

The failure this guards against is specific and silent: a deployment writes
``minimax_h3_inference_contract: official_diffusers_v1``, the field is dropped
somewhere in the plumbing, and the instance serves legacy while every log,
metric and artifact says nothing is wrong. Two things therefore have to hold —
the field arrives, and an unusable value fails loudly instead of falling back.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_stage_deploy_config_carries_the_contract_fields():
    """Both fields exist on the deploy surface a YAML populates."""
    from vllm_omni.config.stage_config import StageDeployConfig

    fields = StageDeployConfig.__dataclass_fields__
    assert "minimax_h3_inference_contract" in fields
    assert "minimax_h3_admission_policy" in fields
    # Absent means legacy; the deploy surface must not invent a default.
    assert fields["minimax_h3_inference_contract"].default is None
    assert fields["minimax_h3_admission_policy"].default is None


def test_diffusion_config_carries_the_contract_fields():
    """The pipeline reads them off the diffusion config, so they have to land there."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    fields = OmniDiffusionConfig.__dataclass_fields__
    assert "minimax_h3_inference_contract" in fields
    assert "minimax_h3_admission_policy" in fields


def _diffusion_stage_engine_values(**deploy_fields):
    """Run a deploy stage through the real override plumbing, not a stand-in.

    An earlier version of this test copied ``asdict(StageDeployConfig)`` and
    asserted the keys survived, which is true of any field whatsoever and so
    proved nothing. The step that actually rejects a field is
    ``_validate_stage_engine_override_ownership``, reached only from
    ``_stage_engine_values``; go through it.
    """
    from vllm_omni.config.omni_config import _stage_engine_values
    from vllm_omni.config.stage_config import StageDeployConfig, StageExecutionType, StagePipelineConfig

    return _stage_engine_values(
        StageDeployConfig(stage_id=0, **deploy_fields),
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
        ),
    )


def test_stage_fields_reach_the_diffusion_overrides():
    """A YAML that sets the contract must start, and the value must arrive."""
    values = _diffusion_stage_engine_values(
        minimax_h3_inference_contract="official_diffusers_v1",
        minimax_h3_admission_policy="parity_fixture_v1",
    )
    diffusion = dict(values.diffusion.to_kwargs())
    assert diffusion["minimax_h3_inference_contract"] == "official_diffusers_v1"
    assert diffusion["minimax_h3_admission_policy"] == "parity_fixture_v1"


def test_the_contract_fields_have_a_structured_owner():
    """They must be declared on the diffusion projection, not only on the deploy surface.

    Ownership is what the validator checks. A field present on
    ``StageDeployConfig`` but absent from ``_DiffusionConfigProjection`` makes
    every deployment that sets it fail at startup.
    """
    from vllm_omni.config.omni_config import _DIFFUSION_OWNED_STAGE_ENGINE_FIELDS

    assert "minimax_h3_inference_contract" in _DIFFUSION_OWNED_STAGE_ENGINE_FIELDS
    assert "minimax_h3_admission_policy" in _DIFFUSION_OWNED_STAGE_ENGINE_FIELDS


def test_an_unowned_stage_field_still_fails_loudly():
    """The guard the previous test relies on has to be live, not vacuous."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="no structured config owner"):
        _diffusion_stage_engine_values(engine_extras={"minimax_h3_nonexistent_knob": "x"})


def test_absent_contract_fields_do_not_reach_the_overrides():
    """Unset means legacy: the keys must not appear at all, not appear as None."""
    diffusion = dict(_diffusion_stage_engine_values().diffusion.to_kwargs())
    assert "minimax_h3_inference_contract" not in diffusion
    assert "minimax_h3_admission_policy" not in diffusion


def test_config_value_beats_the_env_and_reaches_the_strategy():
    """What a deployment writes wins over an operator's stray env var."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    class _Config:
        minimax_h3_inference_contract = "official_diffusers_v1"
        minimax_h3_admission_policy = "parity_fixture_v1"

    config = _Config()
    strategy = resolve_strategy(
        inference_contract=config.minimax_h3_inference_contract,
        admission_policy=config.minimax_h3_admission_policy,
        environ={"VLLM_OMNI_H3_INFERENCE_CONTRACT": "legacy"},
    )
    assert strategy.is_official
    assert strategy.admission_policy == "parity_fixture_v1"


def test_an_unusable_contract_fails_instead_of_falling_back():
    """The whole point: no silent legacy fallback for a typo."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    for bad in ("official", "OFFICIAL_DIFFUSERS_V1", "v1", ""):
        if bad == "":
            # Empty is indistinguishable from unset and is legacy by design.
            assert resolve_strategy(inference_contract=bad, admission_policy=None, environ={}).name == "legacy"
            continue
        with pytest.raises(ValueError, match="inference_contract"):
            resolve_strategy(inference_contract=bad, admission_policy=None, environ={})


def test_the_resolved_contract_is_observable_for_an_operator():
    """An artifact must be attributable to a contract after the fact."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    described = resolve_strategy(
        inference_contract="official_diffusers_v1", admission_policy=None, environ={}
    ).describe()
    # The pipeline logs exactly this dict at startup.
    assert described["inference_contract"] == "official_diffusers_v1"
    assert described["rng_mode"] == "official_diffusers_v1"
    assert described["reference_image_geometry_mode"] == "official_short_edge"
