from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _lora(path: Path, *, alpha: str | None) -> Path:
    metadata = None if alpha is None else {"alpha": alpha}
    save_file(
        {
            "layer.lora_A.default.weight": torch.ones(2, 3),
            "layer.lora_B.default.weight": torch.ones(4, 2),
        },
        path,
        metadata=metadata,
    )
    return path


def test_oracle_adds_the_checkpoint_alpha_when_the_cli_omits_it(tmp_path):
    from tools.minimax_h3_turbo.run_official_turbo_group_offload import resolve_lora_alpha_argv

    checkpoint = _lora(tmp_path / "turbo.safetensors", alpha="128")
    assert resolve_lora_alpha_argv(["--lora-path", str(checkpoint), "--inference-steps", "4"]) == [
        "--lora-path",
        str(checkpoint),
        "--inference-steps",
        "4",
        "--lora-alpha",
        "128",
    ]


def test_oracle_accepts_only_an_explicit_alpha_that_matches(tmp_path):
    from tools.minimax_h3_turbo.run_official_turbo_group_offload import resolve_lora_alpha_argv

    checkpoint = _lora(tmp_path / "turbo.safetensors", alpha="8")
    argv = ["--lora-path", str(checkpoint), "--lora-alpha=8"]
    assert resolve_lora_alpha_argv(argv) == argv
    with pytest.raises(ValueError, match="disagrees"):
        resolve_lora_alpha_argv(["--lora-path", str(checkpoint), "--lora-alpha", "128"])


def test_oracle_refuses_to_guess_when_checkpoint_has_no_alpha(tmp_path):
    from tools.minimax_h3_turbo.run_official_turbo_group_offload import resolve_lora_alpha_argv

    checkpoint = _lora(tmp_path / "turbo.safetensors", alpha=None)
    with pytest.raises(ValueError, match="refusing to guess"):
        resolve_lora_alpha_argv(["--lora-path", str(checkpoint)])


def test_oracle_without_a_lora_is_unchanged():
    from tools.minimax_h3_turbo.run_official_turbo_group_offload import resolve_lora_alpha_argv

    assert resolve_lora_alpha_argv(["--inference-steps", "20"]) == ["--inference-steps", "20"]
