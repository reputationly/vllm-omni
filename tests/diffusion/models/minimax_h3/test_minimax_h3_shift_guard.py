# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Release-pinned shift-scale guard tests.

A partition whose ``_minimax_h3`` metadata explicitly carries
``sigma_shift_scales`` and whose ``distilled`` block pins the schedule
(``supports_num_inference_steps_override`` false) also pins the shifts its
sigma positions were distilled at. Request ``flow_shift``/``audio_flow_shift``
may repeat or omit the pinned values but not override them; every other
partition keeps the legacy behavior of honoring request values.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _pipeline(*, pinned: bool, video: float = 12.0, audio: float = 2.0) -> Any:
    """A pipeline instance carrying only what the helper under test touches."""
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    instance = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(instance)
    instance.default_video_shift = video
    instance.default_audio_shift = audio
    instance._shift_pinned_by_partition = {"fl2va": pinned, "ref2va": pinned}
    return instance


def test_pinned_partition_accepts_matching_values() -> None:
    pipeline = _pipeline(pinned=True)
    assert pipeline._resolve_shift_scales("t2va", {"flow_shift": 12.0, "audio_flow_shift": 2.0}) == (12.0, 2.0)


def test_pinned_partition_rejects_video_mismatch() -> None:
    pipeline = _pipeline(pinned=True)
    with pytest.raises(OmniClientError, match="flow_shift must be 12"):
        pipeline._resolve_shift_scales("t2va", {"flow_shift": 11.0})


def test_pinned_partition_rejects_audio_mismatch() -> None:
    pipeline = _pipeline(pinned=True)
    with pytest.raises(OmniClientError, match="audio_flow_shift must be 2"):
        pipeline._resolve_shift_scales("t2va", {"audio_flow_shift": 3.0})


def test_pinned_partition_uses_metadata_when_absent() -> None:
    pipeline = _pipeline(pinned=True)
    assert pipeline._resolve_shift_scales("t2va", {}) == (12.0, 2.0)


def test_pinned_partition_rejects_non_numeric_override() -> None:
    pipeline = _pipeline(pinned=True)
    with pytest.raises(OmniClientError, match="audio_flow_shift must be 2"):
        pipeline._resolve_shift_scales("t2va", {"audio_flow_shift": "high"})


def test_non_pinned_partition_honours_request_values() -> None:
    pipeline = _pipeline(pinned=False)
    assert pipeline._resolve_shift_scales("t2va", {"flow_shift": 8.0, "audio_flow_shift": 3.0}) == (8.0, 3.0)


def test_release_shift_pinned_from_metadata() -> None:
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _read_release_shift_pinned,
    )

    pinned = {
        "sigma_shift_scales": {"video": 12.0, "audio": 2.0},
        "distilled": {"supports_num_inference_steps_override": False},
    }
    assert _read_release_shift_pinned(pinned)
    # Override allowed -> request shifts stay free.
    assert not _read_release_shift_pinned(
        {
            "sigma_shift_scales": {"video": 12.0, "audio": 2.0},
            "distilled": {"supports_num_inference_steps_override": True},
        }
    )
    # No explicit scales -> legacy request resolution.
    assert not _read_release_shift_pinned({"distilled": {"supports_num_inference_steps_override": False}})
    assert not _read_release_shift_pinned({})
    # Scales without a distilled block -> no schedule to pin them against.
    assert not _read_release_shift_pinned({"sigma_shift_scales": {"video": 12.0}})
