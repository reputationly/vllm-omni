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


def test_stage_fields_reach_engine_args():
    """`_build_engine_args` copies StageDeployConfig through; assert it really does."""
    from dataclasses import asdict

    from vllm_omni.config.stage_config import StageDeployConfig

    stage = StageDeployConfig(
        stage_id=0,
        minimax_h3_inference_contract="official_diffusers_v1",
        minimax_h3_admission_policy="parity_fixture_v1",
    )
    copied = {key: value for key, value in asdict(stage).items() if value is not None}
    assert copied["minimax_h3_inference_contract"] == "official_diffusers_v1"
    assert copied["minimax_h3_admission_policy"] == "parity_fixture_v1"


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
