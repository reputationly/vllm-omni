import json
from pathlib import Path

import pytest

from tools.minimax_h3_turbo.assemble_distilled_partition import assemble, uniform_base_schedule

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _base(tmp_path: Path, partition: str = "fl2va") -> Path:
    base = tmp_path / "base"
    base.mkdir()
    for component in ("audio_vae", "processor", "text_encoder", "tokenizer", "video_vae"):
        (base / component).mkdir()
    (base / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3Pipeline",
                "_minimax_h3": {
                    "partition": partition,
                    "tasks": ["ref2va"] if partition == "ref2va" else ["t2va", "fl2va"],
                    "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
                },
            }
        )
    )
    return base


def test_uniform_schedule_has_one_more_boundary_than_nfe():
    assert uniform_base_schedule(4) == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert len(uniform_base_schedule(8)) == 9


def test_assemble_writes_partition_scoped_schedule_and_relative_links(tmp_path):
    base = _base(tmp_path)
    transformer = tmp_path / "fused" / "transformer"
    transformer.mkdir(parents=True)
    output = tmp_path / "turbo"

    assemble(
        base_partition=base,
        fused_transformer=transformer,
        output=output,
        num_inference_steps=4,
        video_shift=6.0,
        audio_shift=3.0,
        source_lora="turbo4.safetensors",
    )

    release = json.loads((output / "model_index.json").read_text())["_minimax_h3"]
    assert release["base_schedule"] == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert release["sigma_shift_scales"] == {"video": 6.0, "audio": 3.0}
    assert release["distilled"]["num_inference_steps"] == 4
    assert release["distilled"]["recommended_num_inference_steps"] == 4
    assert release["distilled"]["supports_num_inference_steps_override"] is True
    assert (output / "transformer").is_symlink()
    assert not (output / "transformer").readlink().is_absolute()


def test_assemble_refuses_to_modify_an_existing_output(tmp_path):
    base = _base(tmp_path)
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError):
        assemble(
            base_partition=base,
            fused_transformer=transformer,
            output=output,
            num_inference_steps=4,
            video_shift=12.0,
            audio_shift=3.0,
            source_lora=None,
        )
