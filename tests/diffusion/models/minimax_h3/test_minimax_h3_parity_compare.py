# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The parity comparator itself, which has to be trustworthy before its output is."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_TOOLS = Path(__file__).resolve().parents[4] / "tools" / "minimax_h3_parity"


@pytest.fixture(scope="module", autouse=True)
def _tools_on_path():
    sys.path.insert(0, str(_TOOLS))
    yield
    sys.path.remove(str(_TOOLS))


def test_discrete_stages_are_compared_without_tolerance():
    """A token difference must never be softened into a small number."""
    from compare import compare_stage

    result = compare_stage(
        "tokens",
        {"prompt_token_ids": [1, 2, 3], "token_tags": [1, 1, 1]},
        {"prompt_token_ids": [1, 2, 4], "token_tags": [1, 1, 1]},
    )
    assert result["exact_stage"] is True
    assert result["mismatched"] == ["prompt_token_ids"]
    assert result["fields"]["prompt_token_ids"]["first_difference"] == {
        "index": 2,
        "official": 3,
        "candidate": 4,
    }
    assert result["fields"]["token_tags"]["equal"] is True


def test_numeric_stages_report_metrics_and_no_verdict():
    """No threshold is applied: the envelope has to be measured first."""
    from compare import compare_stage

    result = compare_stage(
        "prompt_embeds",
        {"hidden": [1.0, 2.0, 3.0]},
        {"hidden": [1.0, 2.0, 3.001]},
    )
    metrics = result["fields"]["hidden"]
    assert metrics["kind"] == "tolerant"
    assert metrics["max_abs"] == pytest.approx(0.001, abs=1e-9)
    assert metrics["cosine"] == pytest.approx(1.0, abs=1e-6)
    # Crucially: no pass/fail key.
    assert "equal" not in metrics and "passed" not in metrics
    assert result["mismatched"] == []


def test_report_names_the_first_divergence_not_the_last():
    """A later stage's numbers are consequences of an earlier mismatch."""
    from compare import build_report

    report = build_report(
        official={
            "tokens": {"prompt_token_ids": [1, 2]},
            "packing": {"token_tags": [0, 1]},
            "latents_final": {"latents": [1.0]},
        },
        candidate={
            "tokens": {"prompt_token_ids": [1, 3]},
            "packing": {"token_tags": [0, 0]},
            "latents_final": {"latents": [9.0]},
        },
        manifest={},
    )
    assert report["first_exact_divergence"] == "tokens"
    assert "consequences" in report["verdict"]


def test_missing_fields_are_reported_rather_than_skipped():
    """A field only one side dumped is a finding, not an absence of one."""
    from compare import compare_stage

    result = compare_stage("packing", {"cu_seqlens": [0, 10]}, {})
    assert result["fields"]["cu_seqlens"]["kind"] == "missing"
    assert result["mismatched"] == ["cu_seqlens"]


def test_length_mismatch_is_an_error_not_a_metric():
    from compare import compare_tolerant

    assert "error" in compare_tolerant([1.0, 2.0], [1.0])


def test_agreement_is_reported_plainly():
    from compare import build_report

    report = build_report(
        official={"tokens": {"prompt_token_ids": [1, 2]}},
        candidate={"tokens": {"prompt_token_ids": [1, 2]}},
        manifest={"note": "same run"},
    )
    assert report["first_exact_divergence"] is None
    assert report["verdict"] == "exact stages agree"
    assert report["manifest"] == {"note": "same run"}


def test_candidate_only_pad_accounting_is_information_not_divergence():
    """vLLM's 64-alignment pad has no oracle counterpart, by design.

    The brief asks for the pad to be reported and shown isolated, not removed,
    so a `vllm_`-prefixed field the oracle lacks must not read as a mismatch —
    while an ordinary field the oracle lacks still must.
    """
    from compare import compare_stage

    result = compare_stage(
        "packing",
        {"sequence_length": 100},
        {"sequence_length": 100, "vllm_pad_rows": 28, "vllm_padded_sequence_length": 128},
    )
    assert result["mismatched"] == []
    assert result["fields"]["vllm_pad_rows"] == {"kind": "candidate_only", "value": 28}

    # The exemption is prefix-scoped: anything else one-sided is still a finding.
    strict = compare_stage("packing", {"sequence_length": 100}, {"sequence_length": 100, "token_tags": [1]})
    assert strict["mismatched"] == ["token_tags"]
