# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The deployed H3 sigma schedules must survive the solver's stricter validation.

Upstream #8378 routes denoising through ``H3SampleSolver`` instead of calling
``minimax_h3_euler_eta0_step`` directly. ``res_multistep_coeffs`` validates far
more than the old euler path did: at least two sigmas, each finite and inside
``[0, 1]``, and **strictly decreasing**. The sync manual §5.0 A4 flagged "whether
the production schedule is strictly decreasing" as an open risk, so pin it here.

The deploy configs pin ``num_inference_steps`` but the flow shifts live in the
VDN artifact's ``metadata.json`` (12.0 video / 3.0 audio for the singularity VDN
turbo adapters), so the shifts are spelled out below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

# (deploy config, steps, video shift, audio shift) as deployed on 4xA100-40G.
PRODUCTION_SIGMA_CASES = [
    ("minimax_h3_singularity_vdn8_fl2va_a100_40g.yaml", 8, 12.0, 3.0),
    ("minimax_h3_singularity_vdn8_ref2va_a100_40g.yaml", 8, 12.0, 3.0),
]


def _find_key(node: Any, key: str) -> Any:
    """Depth-first search for the first occurrence of ``key`` in parsed YAML."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


@pytest.mark.parametrize(("config_name", "num_steps", "video_shift", "audio_shift"), PRODUCTION_SIGMA_CASES)
def test_production_h3_sigma_schedules_are_accepted_by_the_solver(
    config_name: str, num_steps: int, video_shift: float, audio_shift: float
) -> None:
    from vllm_omni.diffusion.models.minimax_h3.sampling import create_h3_sample_solver
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    repo_root = Path(__file__).resolve().parents[4]
    config_path = repo_root / "deploy-configs" / config_name
    assert config_path.is_file(), f"deploy config not found: {config_path}"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pinned_steps = _find_key(config, "num_inference_steps")
    assert pinned_steps == num_steps, (
        f"{config_name} pins num_inference_steps={pinned_steps!r}; the schedule asserted here "
        f"was derived for {num_steps} steps. Update PRODUCTION_SIGMA_CASES with the deploy config."
    )

    for label, shift in (("video", video_shift), ("audio", audio_shift)):
        sigmas = minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=shift, base_schedule=None)

        # Mirrors res_multistep_coeffs' preconditions, so a failure here points at
        # the schedule rather than at the solver.
        assert len(sigmas) >= 2, f"{label}: need at least two sigmas, got {len(sigmas)}"
        assert all(0.0 <= s <= 1.0 for s in sigmas), f"{label}: sigmas must lie in [0, 1], got {sigmas}"
        assert all(a > b for a, b in zip(sigmas, sigmas[1:])), f"{label}: sigmas must strictly decrease, got {sigmas}"

        # The regression this file exists for: constructing the solver must not raise.
        solver = create_h3_sample_solver("euler", sigmas)
        assert solver is not None


def test_production_sigma_schedules_differ_between_video_and_audio() -> None:
    """A sanity check that the two shifts really produce distinct schedules."""
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    video = minimax_h3_time_shift_sigmas(num_steps=8, shift_scale=12.0, base_schedule=None)
    audio = minimax_h3_time_shift_sigmas(num_steps=8, shift_scale=3.0, base_schedule=None)
    assert video != audio
