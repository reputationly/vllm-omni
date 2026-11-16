# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json

import pytest

from vllm_omni.metrics.usage import (
    PIXELS_PER_IMAGE_TOKEN,
    SPEC,
    build_usage,
    record_usage,
    validate_usage,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# Every pair below was measured on the live MiniMax-H3-Image CN endpoint on
# 2026-11-16 (`usage.output_tokens`). 5472x3654 pins the floor semantics:
# 3654 // 32 == 114, so a round() implementation would produce 19665 and fail here.
VENDOR_OUTPUT_TOKENS = [
    ((2048, 2048), 4096),
    ((4096, 4096), 16384),
    ((2720, 1536), 4080),
    ((7104, 2368), 16428),
    ((5472, 3654), 19494),
    # A 1024x1024 request is still billed 4096 output tokens by the vendor: the
    # generated canvas is floored at 2K, exactly like our ``min_generation_area``.
    ((1024, 1024), 1024),
]


@pytest.mark.parametrize(("size", "expected"), VENDOR_OUTPUT_TOKENS)
def test_output_image_tokens_match_vendor(size: tuple[int, int], expected: int) -> None:
    assert SPEC.image_tokens(*size) == expected


def test_one_image_token_covers_32_pixels() -> None:
    assert PIXELS_PER_IMAGE_TOKEN == 32
    assert SPEC.image_tokens(32, 32) == 1
    assert SPEC.image_tokens(31, 31) == 0


def test_text_to_image_usage_matches_vendor_total() -> None:
    usage = build_usage(width=2048, height=2048, generated_images=1, text_tokens=15)
    payload = usage.to_dict()
    assert payload["generated_images"] == 1
    assert payload["output_tokens"] == 4096
    assert payload["input_tokens"] == 15
    assert payload["total_tokens"] == 4111  # vendor: 4111
    assert payload["usage_source"] == "engine"
    assert payload["usage_version"] == "h3-image-v1"


def test_text_tokens_scale_with_n() -> None:
    """The vendor bills the prompt once per produced image (n=2 gave text 15 -> 30)."""
    usage = build_usage(width=2048, height=2048, generated_images=2, text_tokens=15)
    payload = usage.to_dict()
    assert payload["output_tokens"] == 8192
    assert payload["input_tokens_details"]["text_tokens"] == 30
    assert payload["text_token_factor"] == 2


def test_reference_vision_tokens_include_markers() -> None:
    usage = build_usage(
        width=2048,
        height=2048,
        generated_images=1,
        text_tokens=30,
        reference_vision_counts=(1024, 1024, 1024),
    )
    payload = usage.to_dict()
    # 3 x (1024 patches + <vision_start>/<vision_end>)
    assert payload["input_tokens_details"]["image_tokens"] == 3078
    assert payload["input_tokens_details"]["text_tokens"] == 30
    assert usage.per_reference_tokens == (1026, 1026, 1026)


def test_reference_latent_tokens_are_internal_only() -> None:
    usage = build_usage(
        width=2048,
        height=2048,
        generated_images=1,
        text_tokens=10,
        reference_vision_counts=(1024,),
        reference_latent_tokens=2048,
    )
    payload = usage.to_dict()
    assert payload["reference_latent_tokens"] == 2048
    # Internal cost accounting must never leak into the billed totals.
    assert payload["input_tokens"] == 10 + 1026


def test_canvas_mismatch_is_reported_not_raised() -> None:
    usage = build_usage(
        width=2048,
        height=2048,
        generated_images=1,
        text_tokens=10,
        canvas=(4096, 4096),
    )
    problems = validate_usage(usage)
    assert any("canvas_tokens" in problem for problem in problems)


def test_encoder_token_budget_invariant() -> None:
    usage = build_usage(width=2048, height=2048, generated_images=1, text_tokens=100)
    assert validate_usage(usage, text_and_vision_tokens=100) == []
    assert validate_usage(usage, text_and_vision_tokens=50)


def test_usage_audit_roundtrip(tmp_path, monkeypatch) -> None:
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("VLLM_OMNI_USAGE_LOG", str(log))
    usage = build_usage(width=2048, height=2048, generated_images=1, text_tokens=15).to_dict()
    record_usage(request_id="req-1", usage=usage, status="success", problems=["x"])
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["request_id"] == "req-1"
    assert rows[0]["usage"]["total_tokens"] == 4111
    assert rows[0]["problems"] == ["x"]


def test_usage_audit_disabled_is_a_noop(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_USAGE_LOG", raising=False)
    record_usage(request_id="req-2", usage={"total_tokens": 1})
    assert list(tmp_path.iterdir()) == []
