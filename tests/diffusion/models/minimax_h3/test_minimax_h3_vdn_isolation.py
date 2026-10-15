# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the MiniMax-H3 VDN checkpoint dispatch."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_vdn_names_are_isolated():
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3, vdn, vdnh3

    assert pipeline_minimax_h3.VDNCheckpoint is vdn.VDNCheckpoint
    assert pipeline_minimax_h3.UpstreamVDNCheckpoint is vdnh3.VDNCheckpoint
    assert pipeline_minimax_h3.UpstreamVDNCheckpoint is not pipeline_minimax_h3.VDNCheckpoint


@pytest.mark.parametrize("has_lora_path", [True, False], ids=["with-lora-path", "without-lora-path"])
def test_vdnh3_checkpoint_does_not_claim_vdn_window_attn(tmp_path, has_lora_path):
    from vllm_omni.diffusion.models.minimax_h3 import vdnh3

    lora_path = str(tmp_path) if has_lora_path else None
    if has_lora_path:
        (tmp_path / "model_spec.json").write_text("{}\n", encoding="utf-8")

    od_config = SimpleNamespace(
        lora_path=lora_path,
        diffusion_attention_config=SimpleNamespace(
            default=SimpleNamespace(backend="VDN_WINDOW_ATTN"),
            per_role={},
        ),
    )

    assert vdnh3.VDNCheckpoint.from_od_config(od_config, object()) is None
