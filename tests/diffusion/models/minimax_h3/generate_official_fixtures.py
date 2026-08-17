# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze the MiniMax-H3 official Diffusers contract into a JSON fixture.

``test_minimax_h3_official_contract.py`` compares vLLM-Omni against the fixture
this writes, so the ordinary test run needs neither a Diffusers checkout nor
network access. Regenerate only when the pinned oracle moves:

    H3_DIFFUSERS_SRC=/path/to/diffusers/src \\
        python tests/diffusion/models/minimax_h3/generate_official_fixtures.py

The pinned oracle for this task is Diffusers
``d6726f38a0c5ca6c06a8f227fb7bade3486ed98d`` (``0.40.0.dev0``), whose
``MiniMaxH3ModularPipeline`` is the public entry point named by MiniMax-H3's
root ``model_index.json``. The commit is recorded in the fixture and asserted by
the test, so a fixture regenerated against a different oracle cannot pass
unnoticed.

Everything captured here is a pure function of its arguments: no weights, no
device, no network. That is deliberate — it is what makes the official contract
testable on a laptop, and it is the first executable step of the alignment task.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PINNED_DIFFUSERS_COMMIT = "d6726f38a0c5ca6c06a8f227fb7bade3486ed98d"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "minimax_h3_official_contract_v1.json"

# The released checkpoint's geometry, spelled out here rather than read off a
# checkpoint so the fixture stays weight-free.
CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
REFERENCE_IMAGE_SHORT_EDGE = 2048
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
PATCH_SIZE = (1, 2, 2)
AUDIO_CHANNELS = 2
TEXT_TAG, VIDEO_TAG, AUDIO_TAG = 1, 0, 2


def _resolve_oracle_src() -> Path:
    raw = os.environ.get("H3_DIFFUSERS_SRC")
    if not raw:
        raise SystemExit(
            f"H3_DIFFUSERS_SRC must point at the `src` directory of a Diffusers checkout at {PINNED_DIFFUSERS_COMMIT}."
        )
    src = Path(raw).expanduser().resolve()
    if not (src / "diffusers" / "modular_pipelines" / "minimax_h3").is_dir():
        raise SystemExit(f"{src} does not look like a Diffusers `src` with the minimax_h3 modular pipeline.")
    return src


def _oracle_commit(src: Path) -> str:
    """The checkout's HEAD, so a fixture always names the oracle it came from.

    ``H3_DIFFUSERS_COMMIT`` covers the case where the oracle was copied without
    its ``.git`` — the fixture has to be regenerated in the runtime it will be
    compared against (see the RNG note in ``_rng_cases``), and that runtime is a
    container holding only the source tree.
    """
    override = os.environ.get("H3_DIFFUSERS_COMMIT", "").strip()
    if override:
        return override
    try:
        return subprocess.run(
            ["git", "-C", str(src.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _tensor(value: Any) -> dict[str, Any]:
    """A torch tensor as JSON. float64 round-trips through Python's repr."""
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a tensor, got {type(value)}")
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": list(value.shape),
        "data": value.flatten().tolist(),
    }


def _tensor_digest(value: Any) -> dict[str, Any]:
    """A tensor as an exact but compact record: shape, dtype and a byte digest.

    Used where carrying every value would bloat the fixture (audio waveforms run
    to thousands of samples). A sha256 over the exact bytes is still an exact
    comparison, and the head/tail samples make a failure readable instead of
    just "digest differs".
    """
    import hashlib

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a tensor, got {type(value)}")
    contiguous = value.contiguous()
    flat = contiguous.flatten()
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(contiguous.numpy().tobytes()).hexdigest(),
        "head": flat[:8].tolist(),
        "tail": flat[-8:].tolist(),
    }


def _geometry_cases(mod) -> list[dict[str, Any]]:
    cases = []
    for aspect_w, aspect_h in [
        (16, 9),
        (9, 16),
        (1, 1),
        (21, 9),
        (4, 3),
        (3, 4),
        (2560, 1024),
        (1344, 768),
        # The extremes the released checkpoint documents.
        (4, 1),
        (1, 4),
    ]:
        height, width = mod.resolve_canvas_size(
            aspect_w, aspect_h, CANVAS_MULTIPLE, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS
        )
        cases.append({"aspect_width": aspect_w, "aspect_height": aspect_h, "height": height, "width": width})
    return cases


def _frame_cases(mod) -> list[dict[str, Any]]:
    cases = []
    # 120/124 are the 5 s boundary, 355..362 bracket the 15 s ceiling, and 209
    # is the current vLLM t2va/fl2va default.
    for requested in [1, 5, 21, 22, 120, 124, 125, 209, 345, 346, 355, 360, 361, 362]:
        aligned = mod.align_num_frames(requested, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK)
        cases.append(
            {
                "requested": requested,
                "aligned": aligned,
                "video_latent_frames": mod.video_latent_num_frames(aligned, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK),
                "audio_latents": mod.audio_latent_num_frames(aligned),
            }
        )
    return cases


def _scheduler_cases(scheduler_cls) -> list[dict[str, Any]]:
    import torch

    cases = []
    for shift in (12.0, 3.0):
        for steps in (2, 4, 6, 20, 50):
            scheduler = scheduler_cls(shift=shift)
            scheduler.set_timesteps(steps)
            case = {
                "shift": shift,
                "num_inference_steps": steps,
                "sigmas": _tensor(scheduler.sigmas),
                "timesteps": _tensor(scheduler.timesteps),
                "num_forward_passes": int(scheduler.timesteps.numel()),
            }
            # One deterministic Euler step, to pin the recurrence rather than
            # just the schedule. Values are arbitrary but fixed.
            sample = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(3, 4)
            velocity = torch.linspace(0.5, -0.5, 12, dtype=torch.float32).reshape(3, 4)
            timestep = scheduler.timesteps[0]
            case["step_input_sample"] = _tensor(sample)
            case["step_input_velocity"] = _tensor(velocity)
            case["step_timestep"] = float(timestep)
            case["step_prev_sample"] = _tensor(scheduler.step(velocity, timestep, sample, return_dict=False)[0])
            # `scale_noise` is what noises a conditioning anchor to t = 0.999.
            case["scale_noise_0999"] = _tensor(scheduler.scale_noise(sample, 0.999, velocity))
            cases.append(case)
    return cases


def _packed_case(step_cls, *, text_len, latent_t, latent_h, latent_w, audio_t, anchors):
    import torch

    text_token_tags = torch.full((text_len,), TEXT_TAG, dtype=torch.long)
    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_video_rows,
        num_condition_audio_rows,
    ) = step_cls.build_packed_sequence(
        text_token_tags,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        PATCH_SIZE,
        AUDIO_CHANNELS,
        AUDIO_TAG,
        VIDEO_TAG,
        anchors,
    )
    return {
        "text_len": text_len,
        "latent_t": latent_t,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "audio_t": audio_t,
        "keyframe_anchors": list(anchors),
        "position_ids": _tensor(position_ids),
        "token_tags": _tensor(token_tags),
        "video_indices": _tensor(video_indices),
        "audio_indices": _tensor(audio_indices),
        "text_indices": _tensor(text_indices),
        "num_condition_video_rows": int(num_condition_video_rows),
        "num_condition_audio_rows": int(num_condition_audio_rows),
    }


def _ref2va_case(step_cls, *, text_len, latent_t, latent_h, latent_w, audio_t, blocks):
    """One ref2va layout. ``blocks`` mirrors what the reference encoder emits."""
    import torch

    class _Ref:
        def __init__(self, kind, has_audio):
            self.kind = kind
            self.has_audio = has_audio

    references, condition_latents, audio_condition_latents = [], [], []
    for block in blocks:
        kind = block["kind"]
        has_audio = "audio_t" in block
        references.append(_Ref(kind, has_audio))
        if kind in ("image", "video"):
            frames = block.get("latent_t", 1)
            condition_latents.append(torch.zeros(1, 24, frames, block["latent_h"], block["latent_w"]))
        if has_audio:
            audio_condition_latents.append(torch.zeros(block["audio_t"] * AUDIO_CHANNELS, 32))

    text_token_tags = torch.full((text_len,), TEXT_TAG, dtype=torch.long)
    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_video_rows,
        num_condition_audio_rows,
    ) = step_cls.build_ref2va_packed_sequence(
        text_token_tags,
        references,
        condition_latents,
        audio_condition_latents,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        PATCH_SIZE,
        AUDIO_CHANNELS,
        AUDIO_TAG,
        VIDEO_TAG,
    )
    return {
        "text_len": text_len,
        "latent_t": latent_t,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "audio_t": audio_t,
        "blocks": blocks,
        "position_ids": _tensor(position_ids),
        "token_tags": _tensor(token_tags),
        "video_indices": _tensor(video_indices),
        "audio_indices": _tensor(audio_indices),
        "text_indices": _tensor(text_indices),
        "num_condition_video_rows": int(num_condition_video_rows),
        "num_condition_audio_rows": int(num_condition_audio_rows),
    }


def _rng_cases(randn_tensor) -> list[dict[str, Any]]:
    """The official draw order: conditions, then video, then audio, one generator.

    ``MiniMaxH3PrepareConditionLatentsStep`` draws one noise tensor per visual
    condition at that condition's own shape, then ``MiniMaxH3PrepareLatentsStep``
    draws the video latent and the channel-major audio rows — all off the single
    generator the request was given, in that order. Shapes are kept tiny so the
    fixture can carry every drawn value rather than a digest.
    """
    import hashlib

    import torch

    scenarios = [
        {"name": "t2va", "conditions": [], "latent_t": 2, "latent_h": 2, "latent_w": 2, "audio_t": 3},
        {"name": "fl2va_first", "conditions": [[1, 2, 2]], "latent_t": 2, "latent_h": 2, "latent_w": 2, "audio_t": 3},
        {
            "name": "fl2va_first_last",
            "conditions": [[1, 2, 2], [1, 2, 2]],
            "latent_t": 2,
            "latent_h": 2,
            "latent_w": 2,
            "audio_t": 3,
        },
        {
            "name": "ref2va_image_audio",
            "conditions": [[1, 4, 4]],
            "latent_t": 2,
            "latent_h": 2,
            "latent_w": 2,
            "audio_t": 3,
        },
        {
            "name": "ref2va_multi",
            "conditions": [[1, 4, 4], [1, 2, 2], [2, 4, 6]],
            "latent_t": 3,
            "latent_h": 2,
            "latent_w": 4,
            "audio_t": 2,
        },
    ]

    cases = []
    for scenario in scenarios:
        generator = torch.Generator(device="cpu").manual_seed(42)
        draws = []
        for latent_t, latent_h, latent_w in scenario["conditions"]:
            shape = (1, 24, latent_t, latent_h, latent_w)
            draws.append(
                {
                    "kind": "visual_condition",
                    "tensor": _tensor(
                        randn_tensor(shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)
                    ),
                }
            )
        video_shape = (1, 24, scenario["latent_t"], scenario["latent_h"], scenario["latent_w"])
        draws.append(
            {
                "kind": "video",
                "tensor": _tensor(
                    randn_tensor(video_shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)
                ),
            }
        )
        audio_shape = (scenario["audio_t"] * AUDIO_CHANNELS, 32)
        draws.append(
            {
                "kind": "audio",
                "tensor": _tensor(
                    randn_tensor(audio_shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)
                ),
            }
        )
        cases.append(
            {
                "name": scenario["name"],
                "seed": 42,
                "conditions": scenario["conditions"],
                "latent_t": scenario["latent_t"],
                "latent_h": scenario["latent_h"],
                "latent_w": scenario["latent_w"],
                "audio_t": scenario["audio_t"],
                "draws": draws,
                # How far the request advanced the stream. Two implementations
                # can match on every tensor and still leave the generator in
                # different places, which would diverge on the next draw.
                "final_generator_state_sha256": hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
            }
        )
    return cases


def _audio_normalization_cases(setup_cls) -> list[dict[str, Any]]:
    """The official soundtrack recipe, straight off the oracle's own staticmethod.

    Deterministic non-silent waveforms, so a dropped truncation or a doubled
    resample cannot pass by producing zeros either way.
    """
    import torch

    target_sample_rate = 32000
    cases = []
    for sample_rate in (16000, 32000, 44100, 48000):
        for channels in (1, 2):
            for max_duration in (0.05, 0.2):
                num_samples = int(sample_rate * 0.25)
                ramp = torch.linspace(-0.9, 0.9, num_samples, dtype=torch.float32)
                waveform = torch.stack([ramp * (index + 1) / channels for index in range(channels)])
                normalized = setup_cls._normalize_audio_condition(
                    waveform, sample_rate, target_sample_rate, max_duration=max_duration
                )
                cases.append(
                    {
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "max_duration": max_duration,
                        "target_sample_rate": target_sample_rate,
                        "input": _tensor_digest(waveform),
                        "output": _tensor_digest(normalized),
                    }
                )
    return cases


def _reference_video_cases(setup_cls) -> list[dict[str, Any]]:
    """The official fps resample and target truncation, as an exact index map.

    The canvas rule is passed in as arguments, so a tiny frame size can be made
    a fixed point of ``resolve_canvas_size`` — no LANCZOS pass runs, and each
    output frame still carries the index of the source frame it came from. That
    turns "which source frame lands in which slot" into something comparable
    exactly, instead of via resampled pixels.
    """
    import numpy as np

    # (4, 8) is a fixed point of resolve_canvas_size(8, 4, multiple=2, short=4).
    tiny_multiple, tiny_short_edge, tiny_max_pixels = 2, 4, 10_000
    height, width = 4, 8

    cases = []
    for source_fps in (24.0, 30.0, 25.0, 23.976, 60.0, 12.0):
        for source_frames, num_frames in ((60, 40), (60, 200), (37, 24)):
            frames = np.zeros((source_frames, height, width, 3), dtype=np.uint8)
            for index in range(source_frames):
                frames[index] = index  # < 255, so the index survives byte-exact
            normalized = setup_cls._normalize_video_condition(
                frames,
                source_fps,
                num_frames,
                tiny_multiple,
                tiny_short_edge,
                tiny_max_pixels,
                24.0,
            )
            cases.append(
                {
                    "source_fps": source_fps,
                    "source_frames": source_frames,
                    "num_frames": num_frames,
                    "target_fps": 24.0,
                    "canvas_multiple": tiny_multiple,
                    "canvas_short_edge": tiny_short_edge,
                    "canvas_max_pixels": tiny_max_pixels,
                    "output_frames": int(normalized.shape[0]),
                    # Which source frame each output slot came from.
                    "source_index_per_slot": [int(frame[0, 0, 0]) for frame in normalized],
                }
            )
    return cases


def _conditioner_sampling_cases(encoder_cls) -> list[dict[str, Any]]:
    """Which normalized frames the conditioner reads, and each block's label.

    The labels are rendered into the prompt, so they are token contract: a
    different timestamp is a different tokenization, not a cosmetic change.
    """
    import numpy as np

    cases = []
    for num_frames in (24, 25, 37, 48, 124, 362):
        # uint8 cannot carry an index past 255, so it is written across two
        # channels as base-256 digits; a 362-frame reference is a 15 s request.
        frames = np.zeros((num_frames, 2, 2, 3), dtype=np.uint8)
        for index in range(num_frames):
            frames[index, :, :, 0] = index // 256
            frames[index, :, :, 1] = index % 256
        sampled, block_timestamps = encoder_cls._sample_video_condition_frames(frames, 24.0, 2.0, 2)
        cases.append(
            {
                "num_frames": num_frames,
                "fps": 24.0,
                "sample_fps": 2.0,
                "temporal_patch": 2,
                "frame_indices": [int(frame[0, 0, 0]) * 256 + int(frame[0, 0, 1]) for frame in sampled],
                "block_timestamps": [float(value) for value in block_timestamps],
                "rendered_labels": [f"<{value:.1f} seconds>" for value in block_timestamps],
            }
        )
    return cases


def _reference_image_cases() -> list[dict[str, Any]]:
    """The official image-reference geometry: short edge 2048, upscale, no cap."""
    cases = []
    for width, height in [
        (1344, 768),
        (2560, 1024),
        (1024, 1024),
        (448, 256),
        (768, 1344),
        (1000, 1000),
        (3072, 1024),  # 3:1 — legal for the oracle, which allows 1:4..4:1
        (1024, 3072),
    ]:
        scale = REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
        target_height = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        target_width = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        cases.append(
            {
                "source_width": width,
                "source_height": height,
                "target_width": target_width,
                "target_height": target_height,
                "rows": (target_width // CANVAS_MULTIPLE) * (target_height // CANVAS_MULTIPLE),
            }
        )
    return cases


def _fl2va_keyframe_cases() -> list[dict[str, Any]]:
    """The official `fl2va` keyframe arithmetic, as integers.

    The first keyframe is stretched onto the canvas; the follower is cover-scaled
    with ``round`` and centred with ``(resized - target) // 2``. Only the integer
    plan is frozen here — the pixel comparison lives in the test, which replays
    these numbers through PIL.
    """
    cases = []
    for (src_w, src_h), (width, height) in [
        ((1344, 768), (1344, 768)),
        ((2560, 1024), (1344, 768)),
        ((768, 1344), (1344, 768)),
        ((1000, 1000), (1344, 768)),
        ((1333, 767), (1344, 768)),  # odd sizes -> 1 px centre offset
        ((999, 501), (1344, 768)),
    ]:
        scale = max(width / src_w, height / src_h)
        resized_w = max(width, round(src_w * scale))
        resized_h = max(height, round(src_h * scale))
        left = max(0, (resized_w - width) // 2)
        top = max(0, (resized_h - height) // 2)
        cases.append(
            {
                "source_width": src_w,
                "source_height": src_h,
                "canvas_width": width,
                "canvas_height": height,
                "follower_resized_width": resized_w,
                "follower_resized_height": resized_h,
                "follower_crop_left": left,
                "follower_crop_top": top,
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args()

    src = _resolve_oracle_src()
    sys.path.insert(0, str(src))

    from diffusers.modular_pipelines.minimax_h3 import before_denoise, before_encoder, encoders
    from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as mod
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    from diffusers.utils.torch_utils import randn_tensor

    layout = before_denoise.MiniMaxH3PrepareLayoutStep
    ref_layout = before_denoise.MiniMaxH3Ref2VAPrepareLayoutStep

    commit = _oracle_commit(src)
    if commit != PINNED_DIFFUSERS_COMMIT:
        print(
            f"WARNING: oracle checkout is at {commit}, not the pinned {PINNED_DIFFUSERS_COMMIT}.",
            file=sys.stderr,
        )

    fixture: dict[str, Any] = {
        "meta": {
            "oracle": "huggingface/diffusers MiniMaxH3ModularPipeline",
            "diffusers_commit": commit,
            "pinned_diffusers_commit": PINNED_DIFFUSERS_COMMIT,
            "generator": "tests/diffusion/models/minimax_h3/generate_official_fixtures.py",
            "torch_version": __import__("torch").__version__,
            "platform": f"{platform.system()}-{platform.machine()}",
            "geometry": {
                "canvas_multiple": CANVAS_MULTIPLE,
                "canvas_short_edge": CANVAS_SHORT_EDGE,
                "canvas_max_pixels": CANVAS_MAX_PIXELS,
                "reference_image_short_edge": REFERENCE_IMAGE_SHORT_EDGE,
                "frames_per_chunk": FRAMES_PER_CHUNK,
                "latents_per_chunk": LATENTS_PER_CHUNK,
                "patch_size": list(PATCH_SIZE),
                "audio_channels": AUDIO_CHANNELS,
                "tags": {"text": TEXT_TAG, "video": VIDEO_TAG, "audio": AUDIO_TAG},
            },
        },
        "canvas": _geometry_cases(mod),
        "frames": _frame_cases(mod),
        "scheduler": _scheduler_cases(MiniMaxH3Scheduler),
        "rng": _rng_cases(randn_tensor),
        "reference_audio": _audio_normalization_cases(before_encoder.MiniMaxH3Ref2VASetupStep),
        "reference_video": _reference_video_cases(before_encoder.MiniMaxH3Ref2VASetupStep),
        "conditioner_sampling": _conditioner_sampling_cases(encoders.MiniMaxH3Ref2VATextEncoderStep),
        "reference_image_geometry": _reference_image_cases(),
        "fl2va_keyframes": _fl2va_keyframe_cases(),
        "packed_t2va_fl2va": [
            _packed_case(layout, text_len=7, latent_t=2, latent_h=8, latent_w=8, audio_t=3, anchors=()),
            _packed_case(layout, text_len=5, latent_t=7, latent_h=6, latent_w=10, audio_t=4, anchors=("first",)),
            _packed_case(layout, text_len=5, latent_t=7, latent_h=6, latent_w=10, audio_t=4, anchors=("last",)),
            _packed_case(layout, text_len=3, latent_t=12, latent_h=8, latent_w=6, audio_t=5, anchors=("first", "last")),
            # >= 16 latent frames: the pairwise-vs-sequential summation of the
            # rotary span diverges in the last ulp from here on, and the two
            # call sites must keep their own order.
            _packed_case(layout, text_len=4, latent_t=17, latent_h=4, latent_w=4, audio_t=3, anchors=("last",)),
        ],
        "packed_ref2va": [
            _ref2va_case(
                ref_layout,
                text_len=6,
                latent_t=3,
                latent_h=8,
                latent_w=8,
                audio_t=4,
                blocks=[{"kind": "image", "latent_h": 6, "latent_w": 10}],
            ),
            _ref2va_case(
                ref_layout,
                text_len=4,
                latent_t=3,
                latent_h=8,
                latent_w=8,
                audio_t=4,
                blocks=[
                    {"kind": "image", "latent_h": 6, "latent_w": 10},
                    {"kind": "image", "latent_h": 4, "latent_w": 4},
                ],
            ),
            _ref2va_case(
                ref_layout,
                text_len=4,
                latent_t=3,
                latent_h=8,
                latent_w=8,
                audio_t=4,
                blocks=[{"kind": "image", "latent_h": 6, "latent_w": 10}, {"kind": "audio", "audio_t": 2}],
            ),
            _ref2va_case(
                ref_layout,
                text_len=5,
                latent_t=4,
                latent_h=8,
                latent_w=8,
                audio_t=5,
                blocks=[{"kind": "video", "latent_t": 2, "latent_h": 6, "latent_w": 8, "audio_t": 3}],
            ),
            # Heterogeneous interleave: the order is semantic twice over, and
            # this is the case a modality-bucketed request cannot express.
            _ref2va_case(
                ref_layout,
                text_len=5,
                latent_t=4,
                latent_h=8,
                latent_w=8,
                audio_t=5,
                blocks=[
                    {"kind": "video", "latent_t": 2, "latent_h": 6, "latent_w": 8, "audio_t": 3},
                    {"kind": "image", "latent_h": 4, "latent_w": 6},
                    {"kind": "audio", "audio_t": 2},
                ],
            ),
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.1f} KiB) from diffusers {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
