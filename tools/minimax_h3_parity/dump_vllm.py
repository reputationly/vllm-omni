# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump vLLM-Omni's MiniMax-H3 stages in the shape ``compare.py`` expects.

The upstream stages — geometry, packing, RNG — are pure functions on both sides,
so this runs without a checkpoint, without CUDA and without an engine. That is
the point: the stages the task brief requires to be fixed *first* are also the
ones that can be compared on any machine, so a divergence there is caught before
anyone waits on a GPU.

vLLM-Omni pads its packed sequence up to a multiple of 64 and the official
layout has none, so the packed tensors are emitted over the canonical prefix and
the pad is reported separately — as its own fields, not as a difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def _summarize(tensor: Any, *, full: bool = False) -> Any:
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


def _dump(request: dict[str, Any], contract: str) -> dict[str, dict[str, Any]]:
    import torch

    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _resolve_output_canvas
    from vllm_omni.diffusion.models.minimax_h3.request_noise import MiniMaxH3RequestNoisePlan
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy
    from vllm_omni.diffusion.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER

    strategy = resolve_strategy(inference_contract=contract, admission_policy=None, environ={})

    if request.get("width") and request.get("height"):
        height, width = int(request["height"]), int(request["width"])
    else:
        height, width = _resolve_output_canvas(16 / 9, 768)
    num_frames = MINIMAX_H3_SHAPE_PLANNER.align_frame_count(int(request["num_frames"]))
    latent_t = MINIMAX_H3_SHAPE_PLANNER.video_latent_t(num_frames)
    latent_h, latent_w = height // 16, width // 16
    audio_t = MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(num_frames / 24)

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
    anchor_to_index = {"first": 0, "last": -1}
    packed = minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=bool(anchors),
        keyframe_frame_indices=[anchor_to_index[a] for a in anchors] if anchors else None,
        frame_count=num_frames if anchors else None,
    )
    used = text_len + int((~packed["update_mask"]).sum()) + audio_t * 2 + latent_t * (latent_h // 2) * (latent_w // 2)
    packing = {
        # Canonical prefix only; the 64-alignment pad is vLLM's own and is
        # reported below rather than compared against an oracle that has none.
        "position_ids": _summarize(packed["img_position_ids"][:used]),
        "token_tags": _summarize(packed["token_tags"][:used]),
        "video_indices": _summarize(packed["img_pos"]),
        "audio_indices": _summarize(packed["audio_pos"]),
        "text_indices": _summarize(packed["text_pos"]),
        "num_condition_video_rows": int((~packed["update_mask"]).sum()),
        "num_condition_audio_rows": int((~packed.get("audio_update_mask", torch.ones(1, dtype=torch.bool))).sum())
        if "audio_update_mask" in packed
        else 0,
        "sequence_length": used,
        "vllm_padded_sequence_length": int(packed["seq_len"]),
        "vllm_pad_rows": int(packed["seq_len"]) - used,
    }

    plan = MiniMaxH3RequestNoisePlan(rng_mode=strategy.rng_mode, seed=int(request.get("seed", 42)))
    draws: dict[str, Any] = {}
    condition_shapes = [tuple(shape) for shape in request.get("condition_shapes", [])]
    if condition_shapes:
        for index, drawn in enumerate(plan.draw_visual_condition_noise(condition_shapes, target_latent_t=latent_t)):
            draws[f"visual_condition_{index}"] = _summarize(drawn)
    draws["video"] = _summarize(plan.draw_video_noise(latent_t=latent_t, latent_h=latent_h, latent_w=latent_w))
    draws["audio"] = _summarize(plan.draw_audio_noise(audio_t=audio_t))
    state = plan.generator_state()
    draws["final_generator_state_sha256"] = (
        hashlib.sha256(state.numpy().tobytes()).hexdigest() if state is not None else "legacy-per-draw-generators"
    )

    return {"geometry": geometry, "packing": packing, "rng": draws}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract", default="official_diffusers_v1", choices=("legacy", "official_diffusers_v1"))
    parser.add_argument("--stages", default="geometry,packing,rng")
    args = parser.parse_args()
    stages = tuple(stage.strip() for stage in args.stages.split(",") if stage.strip())

    request = json.loads(args.request.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, payload in _dump(request, args.contract).items():
        if name in stages:
            (args.out / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    import torch

    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "role": "vllm_omni",
                "inference_contract": args.contract,
                "request": request,
                "stages": list(stages),
                "execution": {"note": "upstream stages are pure functions; no engine, no CUDA"},
                "versions": {
                    "torch": torch.__version__,
                    "python": sys.version.split()[0],
                    "platform": f"{platform.system()}-{platform.machine()}",
                },
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {sorted(path.name for path in args.out.glob('*.json'))} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
