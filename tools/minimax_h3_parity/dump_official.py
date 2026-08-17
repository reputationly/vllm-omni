# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the pinned official MiniMax-H3 harness and dump it stage by stage.

The oracle is the Diffusers ``MiniMaxH3ModularPipeline`` at the commit this task
pins, driven in its own environment. Two things about how it is run matter
enough to be recorded in the manifest rather than assumed:

* **Execution topology.** The official pipeline is single-process and has no
  tensor parallelism, and the BF16 transformer is 62 GB — more than one 40 GB
  card holds. It therefore runs with accelerate's ``device_map`` sharding
  layers across the available GPUs. That is *also* an execution topology, not
  "the unsharded truth", so the difference between it and vLLM-Omni's TP is a
  topology difference on both sides, and the report must not present it as
  vLLM's error alone.
* **Stage selection.** Dumping every step of a 50-step run writes tens of GB.
  Stages are opt-in, and the default set is the cheap upstream ones the doc
  requires to be fixed first.

This never runs inside the serving environment: the oracle needs the pinned
Diffusers, and vLLM-Omni is built against another one.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

PINNED_DIFFUSERS_COMMIT = "d6726f38a0c5ca6c06a8f227fb7bade3486ed98d"

# Cheap, weight-free stages first — the doc's order. `prompt_embeds` needs the
# conditioner, the rest need the transformer or the VAE.
DEFAULT_STAGES = ("geometry", "tokens", "packing", "rng")
WEIGHTED_STAGES = ("prompt_embeds", "dit_input_step0", "dit_velocity_step0", "scheduler_step0", "latents_final")


def _summarize(tensor: Any, *, full: bool = False) -> Any:
    """A tensor as JSON: values when small, shape plus digest when not."""
    import hashlib

    import torch

    if not isinstance(tensor, torch.Tensor):
        return tensor
    detached = tensor.detach().to("cpu").contiguous()
    if full or detached.numel() <= 4096:
        return {
            "dtype": str(detached.dtype).removeprefix("torch."),
            "shape": list(detached.shape),
            "data": detached.flatten().tolist(),
        }
    flat = detached.flatten().to(torch.float64)
    return {
        "dtype": str(detached.dtype).removeprefix("torch."),
        "shape": list(detached.shape),
        "sha256": hashlib.sha256(detached.numpy().tobytes()).hexdigest(),
        "head": flat[:16].tolist(),
        "tail": flat[-16:].tolist(),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }


def _manifest(args: argparse.Namespace, request: dict[str, Any]) -> dict[str, Any]:
    import torch

    devices = [torch.cuda.get_device_name(index) for index in range(torch.accelerator.device_count())]
    return {
        "role": "official",
        "oracle": "huggingface/diffusers MiniMaxH3ModularPipeline",
        "diffusers_commit": os.environ.get("H3_DIFFUSERS_COMMIT", PINNED_DIFFUSERS_COMMIT),
        "model_path": str(args.model),
        "workflow": args.workflow,
        "reference_image_short_edge": args.reference_image_short_edge,
        "request": request,
        "stages": list(args.stages),
        # The topology is part of the result, not a footnote: the oracle is
        # sharded across GPUs because it has no TP and does not fit on one.
        "execution": {
            "device_map": args.device_map,
            "num_gpus": len(devices),
            "gpus": devices,
            "tensor_parallel": False,
            "note": "accelerate layer sharding; not an unsharded single-GPU reference",
        },
        "versions": {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "platform": f"{platform.system()}-{platform.machine()}",
        },
    }


def _dump_weight_free(request: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Geometry, packing and RNG — everything the oracle computes without weights."""
    import torch
    from diffusers.modular_pipelines.minimax_h3 import before_denoise
    from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as mod
    from diffusers.utils.torch_utils import randn_tensor

    multiple, short_edge, max_pixels = 32, 768, 768 * 1344
    if request.get("width") and request.get("height"):
        height, width = int(request["height"]), int(request["width"])
    else:
        height, width = mod.resolve_canvas_size(16, 9, multiple, short_edge, max_pixels)
    num_frames = mod.align_num_frames(int(request["num_frames"]), 17, 5)
    latent_t = mod.video_latent_num_frames(num_frames, 17, 5)
    latent_h, latent_w = height // 16, width // 16
    audio_t = mod.audio_latent_num_frames(num_frames)

    geometry = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_latent_frames": latent_t,
        "latent_height": latent_h,
        "latent_width": latent_w,
        "num_audio_latents": audio_t,
    }

    text_len = int(request.get("num_text_tokens", 8))
    anchors = tuple(request.get("keyframe_anchors", ()))
    position_ids, token_tags, video_indices, audio_indices, text_indices, cond_video, cond_audio = (
        before_denoise.MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.ones(text_len, dtype=torch.long),
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            (1, 2, 2),
            2,
            2,
            0,
            anchors,
        )
    )
    packing = {
        "position_ids": _summarize(position_ids),
        "token_tags": _summarize(token_tags),
        "video_indices": _summarize(video_indices),
        "audio_indices": _summarize(audio_indices),
        "text_indices": _summarize(text_indices),
        "num_condition_video_rows": int(cond_video),
        "num_condition_audio_rows": int(cond_audio),
        "sequence_length": int(position_ids.shape[0]),
    }

    generator = torch.Generator(device="cpu").manual_seed(int(request.get("seed", 42)))
    draws = {}
    for index, shape in enumerate(request.get("condition_shapes", [])):
        drawn = randn_tensor((1, 24, *shape), generator=generator, device=torch.device("cpu"), dtype=torch.float32)
        draws[f"visual_condition_{index}"] = _summarize(drawn)
    draws["video"] = _summarize(
        randn_tensor(
            (1, 24, latent_t, latent_h, latent_w), generator=generator, device=torch.device("cpu"), dtype=torch.float32
        )
    )
    draws["audio"] = _summarize(
        randn_tensor((audio_t * 2, 32), generator=generator, device=torch.device("cpu"), dtype=torch.float32)
    )
    import hashlib

    draws["final_generator_state_sha256"] = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()

    return {"geometry": geometry, "packing": packing, "rng": draws}


def _dump_prompt_embeds(args: argparse.Namespace, request: dict[str, Any]) -> dict[str, Any]:
    """The conditioning hidden state, off the official code path.

    Only the conditioner is loaded — 63 GB in BF16, so it is sharded across the
    visible GPUs with accelerate. MiniMax-H3 reads ``hidden_states[50]``, which
    is *not* the last one: the final layer is post-norm and is not what the
    released weights were trained against, so taking `last_hidden_state` here
    would silently compare the wrong tensor.
    """
    import torch
    from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    model_root = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_root / "processor")

    # A ref2va presentation is the prompt PLUS a label and a vision block per
    # reference, so comparing a text-only encode against one would be comparing
    # two different requests. The references are normalized exactly as the
    # oracle's setup step does — short edge 2048, aspect preserved, 32-aligned,
    # upscaling included, no area cap — because the vision token count follows
    # directly from that geometry.
    vision_inputs: dict[str, Any] = {}
    image_token_counts: list[int] = []
    references: list[Any] = []
    if request.get("image_paths"):
        from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference
        from PIL import Image

        images = []
        for path in request["image_paths"]:
            image = Image.open(path).convert("RGB")
            width, height = image.size
            scale = args.reference_image_short_edge / min(width, height)
            target_h = max(32, round(height * scale / 32) * 32)
            target_w = max(32, round(width * scale / 32) * 32)
            if image.size != (target_w, target_h):
                image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            images.append(image)
            references.append(MiniMaxH3ImageReference(image=image))
        features = processor.image_processor(images=images, return_tensors="pt")
        vision_inputs = {"pixel_values": features["pixel_values"], "image_grid_thw": features["image_grid_thw"]}
        merge = processor.image_processor.merge_size**2
        image_token_counts = [int(grid.prod()) // merge for grid in features["image_grid_thw"]]

    if references:
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

        token_ids, token_tags = MiniMaxH3Ref2VATextEncoderStep._build_presentation(
            tokenizer, request["prompt"], references, image_token_counts, [], []
        )
    else:
        token_ids = tokenizer(request["prompt"], add_special_tokens=False)["input_ids"]
        token_tags = [1] * len(token_ids)

    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_root / "text_encoder",
        dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    # `get_image_features` is the documented boundary: it returns exactly what
    # gets scattered into `inputs_embeds`, i.e. post-merge, LLM-width. Hooking
    # the tower module instead captures the pre-merge [patches, vision_dim]
    # tensor, which is not the same quantity vLLM-Omni emits — comparing the two
    # measures the choice of hook, not the model.
    captured: dict[str, Any] = {}
    if vision_inputs:
        with torch.no_grad():
            features = encoder.model.get_image_features(
                vision_inputs["pixel_values"].to(encoder.device, encoder.dtype),
                vision_inputs["image_grid_thw"].to(encoder.device),
            )
        # `pooler_output` is the per-image split of the merged embeddings, i.e.
        # the tensors that get scattered into `inputs_embeds`.
        pooled = getattr(features, "pooler_output", features)
        if isinstance(pooled, (list, tuple)):
            pooled = torch.cat([part for part in pooled], dim=0)
        captured["image_embeds"] = pooled if isinstance(pooled, torch.Tensor) else None

    embeds = get_qwen3vl_prompt_embeds(
        encoder,
        processor,
        token_ids,
        vision_inputs,
        text_encoder_layer=50,
        device=encoder.device,
        dtype=torch.bfloat16,
    )
    import hashlib

    import numpy as np

    np.save(Path(args.out) / "prompt_embeds.npy", embeds.detach().float().cpu().numpy())
    if captured.get("image_embeds") is not None:
        tower_out = captured["image_embeds"].detach().float().cpu().contiguous()
        flat = tower_out.flatten().to(torch.float64)
        np.save(Path(args.out) / "image_embeds.npy", tower_out.numpy())
        (Path(args.out) / "vision_output.json").write_text(
            json.dumps(
                {
                    "image_embeds": {
                        "dtype": str(captured["image_embeds"].dtype).removeprefix("torch."),
                        "shape": list(tower_out.shape),
                        "sha256": hashlib.sha256(tower_out.numpy().tobytes()).hexdigest(),
                        "head": flat[:8].tolist(),
                        "mean": float(flat.mean()),
                        "std": float(flat.std()),
                    }
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
    vision_record = {}
    for name, value in vision_inputs.items():
        detached = value.detach().float().cpu().contiguous()
        flat = detached.flatten().to(torch.float64)
        vision_record[name] = {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(detached.shape),
            "sha256": hashlib.sha256(detached.numpy().tobytes()).hexdigest(),
            "head": flat[:8].tolist(),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
        }
        if name.startswith("pixel_"):
            np.save(Path(args.out) / f"{name}.npy", detached.numpy())
    (Path(args.out) / "vision_inputs.json").write_text(json.dumps(vision_record, indent=1) + "\n", encoding="utf-8")
    return {
        "prompt": request["prompt"],
        "token_ids": [int(value) for value in token_ids],
        "token_tags": [int(value) for value in token_tags],
        "text_encoder_layer": 50,
        "hidden": _summarize(embeds.float()),
        "hidden_shape": list(embeds.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="BF16 MiniMax-H3 repository root")
    parser.add_argument("--workflow", default="ref2va", choices=("t2va", "fl2va", "ref2va"))
    parser.add_argument("--request", type=Path, required=True, help="request JSON")
    parser.add_argument("--out", type=Path, required=True, help="dump directory")
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    parser.add_argument("--device-map", default="auto", help="accelerate device_map for the sharded oracle")
    parser.add_argument(
        "--reference-image-short-edge",
        type=int,
        default=2048,
        help=(
            "Short edge a reference image is encoded at. 2048 is the released value; a smaller one makes the "
            "ref2va presentation small enough for the unsharded oracle to encode, and is recorded in the "
            "manifest so the dump is never mistaken for the released geometry."
        ),
    )
    parser.add_argument(
        "--weight-free-only",
        action="store_true",
        help="dump only the stages that need no checkpoint (runs anywhere, including CPU)",
    )
    args = parser.parse_args()
    args.stages = tuple(stage.strip() for stage in args.stages.split(",") if stage.strip())

    src = os.environ.get("H3_DIFFUSERS_SRC")
    if src:
        sys.path.insert(0, src)

    request = json.loads(args.request.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    dumps = _dump_weight_free(request)
    requested_weighted = [stage for stage in args.stages if stage in WEIGHTED_STAGES]
    if requested_weighted and args.weight_free_only:
        raise SystemExit(f"--weight-free-only excludes {', '.join(requested_weighted)}")
    if "prompt_embeds" in requested_weighted:
        dumps["prompt_embeds"] = _dump_prompt_embeds(args, request)
        requested_weighted.remove("prompt_embeds")
    if requested_weighted:
        raise SystemExit(
            "stages "
            + ", ".join(requested_weighted)
            + " need the transformer or the VAE loaded; that path is not implemented in this revision. "
            "Report them as pending rather than as passing."
        )

    for name, payload in dumps.items():
        if name in args.stages:
            (args.out / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    try:
        manifest = _manifest(args, request)
    except Exception:  # pragma: no cover - torch without CUDA
        manifest = {"role": "official", "request": request, "stages": list(args.stages)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {sorted(path.name for path in args.out.glob('*.json'))} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
