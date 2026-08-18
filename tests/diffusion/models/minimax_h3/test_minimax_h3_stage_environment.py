# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A stage-scoped contract selection has to reach whoever answers for it.

``stages[].runtime.env`` is applied while the stage starts and restored the
moment it is running, so the variables live in the stage process and nowhere
else. Every other process — the HTTP serving layer above all — resolves the
contract from an environment where the operator's selection is simply absent,
and answers ``legacy`` for a worker configured as ``official``: reference images
pre-stretched onto the canvas, ordered references refused with a 400, on every
request, with both sides logging the contract they each believe in.

This is the fourth occurrence of one accident and the first that is not a config
*field*: the geometry, RNG and condition-noise knobs have no field at all, they
exist only as environment variables, so the mapping itself is what has to cross.
It crosses as ``diffusion_runtime_environ`` — carried from the stage's deploy
config into the diffusion engine args, and laid back over ``os.environ`` by
``contract_environ`` on both sides, in the precedence
``stage_runtime_env`` itself applies.

Worth stating because it is the reason this file exists at all: as of the
upstream stage-runtime refactor (#3855) nothing calls ``stage_runtime_setup``,
so ``runtime.env`` is currently applied to no process whatsoever. Carrying it as
config is therefore not merely a repair of the serving side — it is the only
road the selection has, and it puts both sides on the same one.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_CONTRACT = "VLLM_OMNI_H3_INFERENCE_CONTRACT"
_GEOMETRY = "VLLM_OMNI_H3_REF_IMAGE_GEOMETRY"


def _config(**extra):
    """What a probe reads off a diffusion config, or off the cross-process view."""
    return SimpleNamespace(
        model_class_name="MiniMaxH3Pipeline",
        minimax_h3_inference_contract=extra.pop("contract", None),
        minimax_h3_admission_policy=None,
        diffusion_runtime_environ=extra.pop("stage_env", None),
        **extra,
    )


# ------------------------------------------------------- the environment itself


def test_a_stage_variable_is_read_even_though_this_process_never_had_it():
    from vllm_omni.diffusion.models.minimax_h3.strategy import contract_environ

    assert _CONTRACT not in os.environ, "the point of the test is that this process does not have it"
    environ = contract_environ(_config(stage_env={_CONTRACT: "official_diffusers_v1"}))
    assert environ[_CONTRACT] == "official_diffusers_v1"


def test_the_process_environment_still_shows_through():
    """The overlay adds; it does not replace. Other variables must survive."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import contract_environ

    environ = contract_environ(_config(stage_env={_CONTRACT: "official_diffusers_v1"}))
    assert set(os.environ) <= set(environ)


def test_the_stage_wins_where_both_carry_the_key(monkeypatch):
    """``stage_runtime_env`` overwrites the process value, so the overlay must too."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import contract_environ

    monkeypatch.setenv(_CONTRACT, "legacy")
    environ = contract_environ(_config(stage_env={_CONTRACT: "official_diffusers_v1"}))
    assert environ[_CONTRACT] == "official_diffusers_v1"


def test_no_stage_environment_is_the_process_environment_unchanged():
    """The overwhelmingly common case must not pay for, or be changed by, this."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import contract_environ

    assert contract_environ(_config()) is os.environ
    assert contract_environ(None) is os.environ
    assert contract_environ(_config(stage_env={})) is os.environ


# ------------------------------------------------- what the serving layer answers


def test_the_capability_probes_see_a_stage_scoped_contract():
    """The two answers that were wrong in the field, on a config with no fields set."""
    from vllm_omni.diffusion.model_metadata import (
        honours_explicit_reference_order,
        reference_images_bind_output_canvas,
    )

    stage = _config(stage_env={_CONTRACT: "official_diffusers_v1"})
    assert reference_images_bind_output_canvas("MiniMaxH3Pipeline", stage) is False
    assert honours_explicit_reference_order("MiniMaxH3Pipeline", stage) is True

    blind = _config()
    assert reference_images_bind_output_canvas("MiniMaxH3Pipeline", blind) is True
    assert honours_explicit_reference_order("MiniMaxH3Pipeline", blind) is False


def test_a_stage_scoped_geometry_selection_reaches_the_probe():
    """The knob with no config field: without the mapping it cannot cross at all."""
    from vllm_omni.diffusion.model_metadata import reference_images_bind_output_canvas

    stage = _config(stage_env={_GEOMETRY: "official_short_edge"})
    assert reference_images_bind_output_canvas("MiniMaxH3Pipeline", stage) is False


def test_an_unusable_stage_value_does_not_become_a_serving_side_rejection():
    """It fails where it is applied, in the pipeline, with the pipeline's message."""
    from vllm_omni.diffusion.model_metadata import (
        honours_explicit_reference_order,
        reference_images_bind_output_canvas,
    )

    stage = _config(stage_env={_CONTRACT: "nonsense"})
    assert reference_images_bind_output_canvas("MiniMaxH3Pipeline", stage) is True
    assert honours_explicit_reference_order("MiniMaxH3Pipeline", stage) is True


# ----------------------------------------------------- how it gets onto the config


def _shipped_h3_stage(stage_env=None):
    """Merge the shipped 4-card H3 deploy file, optionally with a stage env.

    Resolved off the installed package rather than the repo tree, because this
    file has to run in the engine image too, where ``vllm_omni`` is where the
    YAML lives and the checkout is only ``tests/``.
    """
    import inspect
    from pathlib import Path

    import vllm_omni
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
    from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

    deploy_dir = Path(inspect.getfile(vllm_omni)).parent / "deploy"
    deploy = load_deploy_config(deploy_dir / "minimax_h3_dit.yaml")
    if stage_env is not None:
        deploy.stages[0].env = stage_env
    (stage,) = merge_pipeline_deploy(OMNI_PIPELINES["minimax_h3_dit"], deploy)
    return stage


def test_a_diffusion_stage_carries_its_runtime_env_into_the_engine_args():
    """``env`` is reserved out of the generic copy, so this needs its own write.

    On the shipped 4-card deploy file, because the thing being pinned is that an
    ordinary H3 deployment gets the mapping — not that a synthetic stage does.
    """
    stage = _shipped_h3_stage({_CONTRACT: "official_diffusers_v1", _GEOMETRY: "official_short_edge"})
    assert stage.yaml_engine_args["diffusion_runtime_environ"] == {
        _CONTRACT: "official_diffusers_v1",
        _GEOMETRY: "official_short_edge",
    }


def test_a_deployment_that_declares_no_stage_env_carries_no_mapping():
    """The shipped file as it actually ships: nothing added, nothing to explain."""
    assert "diffusion_runtime_environ" not in _shipped_h3_stage().yaml_engine_args


def test_the_cross_process_view_carries_it_too():
    """Out of process the view is all the serving layer sees."""
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    assert "diffusion_runtime_environ" in AsyncOmniEngine._DIFFUSION_CONTRACT_VIEW_FIELDS


def test_the_stage_env_parser_is_the_one_the_stage_itself_uses():
    """Two parsers of one config surface is how the two sides come to disagree."""
    from vllm_omni.engine.stage_init_utils import stage_runtime_env_mapping

    assert stage_runtime_env_mapping(0, {"env": {_CONTRACT: "official_diffusers_v1"}}) == {
        _CONTRACT: "official_diffusers_v1"
    }
    assert stage_runtime_env_mapping(0, None) == {}
    assert stage_runtime_env_mapping(0, {}) == {}
    # Values are stringified, because an environment holds strings and a YAML
    # scalar does not: `env: {FOO: 8}` must not reach os.environ as an int.
    assert stage_runtime_env_mapping(0, {"env": {"FOO": 8}}) == {"FOO": "8"}
