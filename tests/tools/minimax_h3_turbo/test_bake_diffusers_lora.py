import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.minimax_h3_turbo.bake_diffusers_lora import bake

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_bake_diffusers_lora_preserves_layout_and_audits_every_target(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    weights = {
        "block.attn.to_q.weight": torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
        "block.ff.net.0.proj.weight": torch.arange(32, dtype=torch.bfloat16).reshape(8, 4),
        "untouched.weight": torch.ones(3, 3, dtype=torch.bfloat16),
    }
    save_file(weights, base / "diffusion_pytorch_model-00001-of-00001.safetensors")
    weight_map = {key: "diffusion_pytorch_model-00001-of-00001.safetensors" for key in weights}
    (base / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    (base / "config.json").write_text("{}", encoding="utf-8")

    lora = tmp_path / "adapter.safetensors"
    factors = {
        "block.attn.to_q.lora_A.default.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        "block.attn.to_q.lora_B.default.weight": torch.arange(12, dtype=torch.bfloat16).reshape(6, 2),
        "block.ff.net.0.proj.lora_A.default.weight": torch.ones(2, 4, dtype=torch.bfloat16),
        "block.ff.net.0.proj.lora_B.default.weight": torch.ones(8, 2, dtype=torch.bfloat16),
    }
    save_file(factors, lora, metadata={"alpha": "1"})

    output = tmp_path / "fused"
    provenance = bake(base=base, lora_path=lora, output=output)
    fused = load_file(output / "diffusion_pytorch_model-00001-of-00001.safetensors")

    for target in ("block.attn.to_q", "block.ff.net.0.proj"):
        expected = (
            weights[target + ".weight"].float()
            + 0.5
            * (factors[target + ".lora_B.default.weight"].float() @ factors[target + ".lora_A.default.weight"].float())
        ).to(torch.bfloat16)
        torch.testing.assert_close(fused[target + ".weight"], expected, atol=0, rtol=0)
    torch.testing.assert_close(fused["untouched.weight"], weights["untouched.weight"], atol=0, rtol=0)
    assert provenance["effective_lora_scale"] == 0.5
    assert provenance["fusion_verification"]["verified_target_tensors"] == 2
    assert provenance["fusion_verification"]["max_abs_error"] == 0.0
    assert json.loads((output / "fusion_provenance.json").read_text()) == provenance
