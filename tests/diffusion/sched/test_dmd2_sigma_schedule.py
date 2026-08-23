# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas
from vllm_omni.diffusion.sched import DMD2SigmaSchedule

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_step_count_is_intervals_not_boundaries():
    schedule = DMD2SigmaSchedule.from_positions([1.0, 0.75, 0.5, 0.25, 0.0])
    assert schedule.num_inference_steps == 4


def test_turbo_four_nfe_matches_documented_video_and_audio_sigmas():
    schedule = DMD2SigmaSchedule.from_positions([1.0, 0.75, 0.5, 0.25, 0.0])
    assert schedule.shifted_sigmas(12.0) == pytest.approx([1.0, 0.97297297, 0.92307692, 0.8, 0.0], abs=1e-7)
    assert schedule.shifted_sigmas(3.0) == pytest.approx([1.0, 0.9, 0.75, 0.5, 0.0], abs=1e-7)


def test_turbo_four_checkpoint_accepts_eight_requested_nfe():
    schedule = DMD2SigmaSchedule.from_positions([1.0, 0.75, 0.5, 0.25, 0.0])
    positions = schedule.positions_for_num_inference_steps(8)
    assert len(positions) == 9
    assert positions == tuple((8 - index) / 8 for index in range(9))
    assert len(minimax_h3_time_shift_sigmas(num_steps=8, shift_scale=12.0, base_schedule=positions)) - 1 == 8


@pytest.mark.parametrize("num_steps", [1, 4, 8, 20, 30, 50])
def test_dynamic_schedule_runs_exactly_the_requested_nfe(num_steps):
    sigmas = minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=12.0)
    assert len(sigmas) == num_steps + 1
    assert sigmas[0] == 1.0
    assert sigmas[-1] == 0.0


@pytest.mark.parametrize(
    "positions",
    [
        [],
        [1.0],
        [0.9, 0.5, 0.0],
        [1.0, 0.5, 0.1],
        [1.0, 0.4, 0.6, 0.0],
        [1.0, 0.5, 0.5, 0.0],
        [1.0, float("nan"), 0.0],
        [1.0, float("inf"), 0.0],
        [1.0, float("nan"), 0.4, 0.0],
        [1.0, float("inf"), 0.4, 0.0],
    ],
)
def test_malformed_positions_are_rejected(positions):
    with pytest.raises(ValueError):
        DMD2SigmaSchedule.from_positions(positions)


def test_one_base_schedule_drives_several_shift_scales():
    schedule = DMD2SigmaSchedule.from_positions([1.0, 0.7, 0.4, 0.15, 0.0])

    assert schedule.shifted_sigmas(12.0) == pytest.approx([1.0, 0.9655172, 0.8888889, 0.6792453, 0.0], abs=1e-6)
    assert schedule.shifted_sigmas(3.0) == pytest.approx([1.0, 0.875, 0.6666667, 0.3461539, 0.0], abs=1e-6)


def test_non_positive_shift_scale_is_rejected():
    schedule = DMD2SigmaSchedule.from_positions([1.0, 0.5, 0.0])

    with pytest.raises(ValueError):
        schedule.shifted_sigmas(0.0)


def test_absent_metadata_key_means_undistilled_but_empty_is_malformed():
    assert DMD2SigmaSchedule.from_metadata({}) is None
    assert DMD2SigmaSchedule.from_metadata({"base_schedule": [1.0, 0.5, 0.0]}).base_schedule == (1.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        DMD2SigmaSchedule.from_metadata({"base_schedule": []})
