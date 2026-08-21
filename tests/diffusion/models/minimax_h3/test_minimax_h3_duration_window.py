# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which ``num_frames`` a contract admits — checked against what actually runs.

Every H3 request is snapped up to the next ``17 * n + 5`` the video VAE can
encode. The bug this file pins was an ORDERING one, not a formula one: the
duration ceiling was applied to the count the caller asked for, and the
alignment was applied afterwards, so the number that was validated was never the
number that was generated. Official does it the other way round, and says why
(``modular_pipelines/minimax_h3/before_denoise.py``)::

    # The duration the request generates is the one of the *aligned* frame count, so that is what the
    # ceiling has to hold for: 346 frames would otherwise pass the check and then be rounded up to 362.

Both ends of the window move when the check moves, and the two ends were decided
differently on 2026-08-18:

* Low end — adopted as-is. Rejecting 108 frames was stricter than official for
  nothing: they align to 124, a perfectly legal 5.167 s clip.
* High end — deliberately NOT adopted. ``duration=15`` aligns to 362 frames
  (15.083 s), which official refuses; we keep serving it. It is the most common
  request there is, the ceiling is a product limit rather than a model one, and
  0.083 s of extra video is not something a caller can observe as a contract
  violation. Being a superset by exactly one alignment step is the point.

Legacy is not touched by any of this and its admissible set must stay
bit-identical — it is what production runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

FPS = 24


def _pipeline(contract: str):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3.strategy import legacy_strategy, official_diffusers_v1_strategy

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.strategy = legacy_strategy() if contract == "legacy" else official_diffusers_v1_strategy()
    return pipeline


def _sampling(**extra):
    return SimpleNamespace(
        fps=FPS,
        num_frames=extra.pop("num_frames", 1),
        height=None,
        width=None,
        extra_args={"aspect_ratio": "auto", **extra},
    )


def _frames(contract: str, **request) -> int:
    """``num_frames`` as resolved, i.e. after alignment — what actually runs."""
    _, _, num_frames, _, _ = _pipeline(contract)._resolve_shape("ref2va", _sampling(**request), None)
    return num_frames


# ------------------------------------------------------------------ the window


def test_legacy_admits_exactly_what_it_admitted_before():
    """2 s to 16 s at 24 fps, on the requested count. Production's envelope."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import legacy_strategy

    assert legacy_strategy().requested_frame_window(FPS) == (48, 384)


def test_official_widens_at_the_bottom_and_by_one_step_at_the_top():
    from vllm_omni.diffusion.models.minimax_h3.strategy import official_diffusers_v1_strategy

    # 108 is the smallest count that aligns to 124 (= the 5 s floor snapped up);
    # 362 is 360 (= the 15 s ceiling) snapped up.
    assert official_diffusers_v1_strategy().requested_frame_window(FPS) == (108, 362)


def test_the_window_is_stated_in_frames_so_alignment_can_be_expressed():
    """A seconds bound cannot say "the next 17n+5 above this", which is the whole bug.

    Both ends of the official window land on, or one step below, the alignment
    lattice — a property no pair of floats could carry.
    """
    from vllm_omni.diffusion.models.minimax_h3.strategy import official_diffusers_v1_strategy
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_align_frame_count

    low, high = official_diffusers_v1_strategy().requested_frame_window(FPS)
    assert minimax_h3_align_frame_count(low) == 124
    assert minimax_h3_align_frame_count(low - 1) == 107, "one below the window must fall to the previous step"
    assert minimax_h3_align_frame_count(high) == high


# --------------------------------------------------- what the pipeline resolves


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (108, 124),  # official accepts, we used to reject
        (119, 124),
        (120, 124),
        (345, 345),  # the last count official itself accepts
        (346, 362),  # official rejects from here up; we keep serving it
        (360, 362),
        (362, 362),
    ],
)
def test_official_resolves_the_aligned_count_across_the_whole_window(requested, expected):
    assert _frames("official_diffusers_v1", num_frames=requested) == expected


def test_the_most_common_request_still_works():
    """``duration=15`` — the gateway's longest preset — must not become a 400.

    It generates 362 frames, 15.083 s. Official refuses exactly this; keeping it
    is the decision this file records, not an oversight.
    """
    assert _frames("official_diffusers_v1", duration=15.0) == 362


def test_a_short_request_that_aligns_into_the_window_is_admitted():
    """4.5 s asks for 108 frames and gets a 5.167 s clip, which is legal."""
    assert _frames("official_diffusers_v1", duration=4.5) == 124


@pytest.mark.parametrize("requested", [107, 363])
def test_official_still_refuses_outside_the_window(requested):
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError, match="output duration must be in"):
        _frames("official_diffusers_v1", num_frames=requested)


# ------------------------------------------------------------- legacy unchanged


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (48, 56),
        (124, 124),
        (209, 209),
        (360, 362),
        (384, 396),
    ],
)
def test_legacy_resolves_what_it_always_did(requested, expected):
    assert _frames("legacy", num_frames=requested) == expected


@pytest.mark.parametrize("requested", [47, 385])
def test_legacy_rejects_what_it_always_did(requested):
    """Both ends unchanged: 47 frames and 385 frames were errors and stay errors.

    Note 384 resolves to 396 frames — legacy has always generated past its own
    ceiling, because it too aligned after checking. That is not fixed here: the
    check moved only under the official contract, since moving it under legacy
    would change what a live gateway is allowed to submit.
    """
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError, match="output duration must be in"):
        _frames("legacy", num_frames=requested)


def test_the_legacy_rejection_message_is_untouched():
    """Pinned verbatim elsewhere (``test_client_error_across_worker``): the facade
    forwards this string to callers, so its bounds and its rounding are contract."""
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError) as excinfo:
        _frames("legacy", duration=29.958)
    assert str(excinfo.value) == "MiniMax H3 output duration must be in [2, 16] seconds, got 29.958"


# ------------------------------------------------------- one window, two checks


def test_both_entry_paths_read_the_same_window():
    """``duration=`` and ``num_frames=`` are two doors into one rule.

    They used to be checked against separately-derived bounds — the float
    against ``output_duration_seconds``, the count against the same pair
    re-divided by fps — which is what let a request pass one and fail the other's
    intent. Same window now, so the same request is admitted through either door.
    """
    from vllm_omni.errors import OmniClientError

    assert _frames("official_diffusers_v1", duration=108 / FPS) == 124
    assert _frames("official_diffusers_v1", num_frames=108) == 124

    for request in ({"duration": 107 / FPS}, {"num_frames": 107}):
        with pytest.raises(OmniClientError, match=r"\[4.5, 15.083\] seconds"):
            _frames("official_diffusers_v1", **request)


def test_the_latent_shapes_follow_the_aligned_count_not_the_requested_one():
    """The downstream consequence of validating the wrong number.

    Latent T is derived from the aligned count, so a request admitted on its
    requested count would have planned a schedule for a clip length that is never
    produced.
    """
    _, _, num_frames, latent_t, audio_t = _pipeline("official_diffusers_v1")._resolve_shape(
        "ref2va", _sampling(duration=15.0), None
    )
    assert (num_frames, latent_t) == (362, 107)
    assert audio_t == round(362 / FPS * 40)


# ---------------------------------------------------- deployment-level ceiling


def test_deployment_can_lower_the_output_ceiling():
    """A memory-tight instance shrinks the admission window without touching
    model semantics: 8 s requests 192 frames, which sits on the alignment
    lattice, so the pipeline's enforced ceiling is exactly 192."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import (
        MINIMAX_H3_MAX_OUTPUT_SECONDS_ENV,
        resolve_strategy,
    )

    strategy = resolve_strategy(
        inference_contract="official_diffusers_v1",
        admission_policy=None,
        environ={MINIMAX_H3_MAX_OUTPUT_SECONDS_ENV: "8"},
    )
    assert strategy.output_duration_seconds == (5.0, 8.0)
    assert strategy.requested_frame_window(FPS) == (108, 192)


def test_deployment_ceiling_must_stay_above_the_contract_floor():
    from vllm_omni.diffusion.models.minimax_h3.strategy import (
        MINIMAX_H3_MAX_OUTPUT_SECONDS_ENV,
        resolve_strategy,
    )

    with pytest.raises(ValueError, match="below the contract minimum"):
        resolve_strategy(
            inference_contract="official_diffusers_v1",
            admission_policy=None,
            environ={MINIMAX_H3_MAX_OUTPUT_SECONDS_ENV: "3"},
        )


@pytest.mark.parametrize("raw", ["abc", "-1", "0", "inf", "nan"])
def test_deployment_ceiling_rejects_unusable_values(raw):
    from vllm_omni.diffusion.models.minimax_h3.strategy import (
        MINIMAX_H3_MAX_OUTPUT_SECONDS_ENV,
        resolve_strategy,
    )

    with pytest.raises(ValueError):
        resolve_strategy(
            inference_contract="official_diffusers_v1",
            admission_policy=None,
            environ={MINIMAX_H3_MAX_OUTPUT_SECONDS_ENV: raw},
        )


def test_deployment_ceiling_warns_when_it_widens_past_the_contract(caplog):
    import logging

    from vllm_omni.diffusion.models.minimax_h3.strategy import (
        MINIMAX_H3_MAX_OUTPUT_SECONDS_ENV,
        resolve_strategy,
    )

    with caplog.at_level(logging.WARNING):
        strategy = resolve_strategy(
            inference_contract="official_diffusers_v1",
            admission_policy=None,
            environ={MINIMAX_H3_MAX_OUTPUT_SECONDS_ENV: "30"},
        )

    assert strategy.output_duration_seconds[1] == 30.0
    assert "widens the admission window past the contract maximum" in caplog.text


def test_deployment_ceiling_is_quiet_when_it_narrows(caplog):
    import logging

    from vllm_omni.diffusion.models.minimax_h3.strategy import (
        MINIMAX_H3_MAX_OUTPUT_SECONDS_ENV,
        resolve_strategy,
    )

    with caplog.at_level(logging.WARNING):
        resolve_strategy(
            inference_contract="official_diffusers_v1",
            admission_policy=None,
            environ={MINIMAX_H3_MAX_OUTPUT_SECONDS_ENV: "5.5"},
        )

    assert "widens the admission window" not in caplog.text
