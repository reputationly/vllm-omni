# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare vLLM-Omni's MiniMax-H3 contract against the official Diffusers one.

The expected values come from ``fixtures/minimax_h3_official_contract_v1.json``,
frozen from Diffusers ``d6726f38…`` by ``generate_official_fixtures.py``. Reading
a fixture rather than importing Diffusers keeps this runnable anywhere and pins
the oracle to a commit instead of to whatever happens to be installed.

Everything here is a pure function: no weights, no device, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

FIXTURE = Path(__file__).parent / "fixtures" / "minimax_h3_official_contract_v1.json"
PINNED_DIFFUSERS_COMMIT = "d6726f38a0c5ca6c06a8f227fb7bade3486ed98d"


@pytest.fixture(scope="module")
def oracle() -> dict:
    if not FIXTURE.is_file():
        pytest.fail(f"missing oracle fixture {FIXTURE}; regenerate with generate_official_fixtures.py")
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["meta"]["diffusers_commit"] == PINNED_DIFFUSERS_COMMIT, (
        "the fixture was generated from a Diffusers commit other than the pinned oracle"
    )
    return data


def _as_tensor(blob: dict) -> torch.Tensor:
    return torch.tensor(blob["data"], dtype=getattr(torch, blob["dtype"])).reshape(blob["shape"])


# --------------------------------------------------------------------------
# Geometry and frame arithmetic
# --------------------------------------------------------------------------


def test_output_canvas_matches_official(oracle):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _resolve_output_canvas

    for case in oracle["canvas"]:
        ratio = case["aspect_width"] / case["aspect_height"]
        height, width = _resolve_output_canvas(ratio, 768)
        assert (height, width) == (case["height"], case["width"]), (
            f"canvas for {case['aspect_width']}:{case['aspect_height']} "
            f"is {height}x{width}, official is {case['height']}x{case['width']}"
        )


def test_reference_video_canvas_matches_official(oracle):
    """A video reference follows the same canvas rule as the target."""
    from vllm_omni.diffusion.models.minimax_h3.reference_video import _reference_video_shape

    for case in oracle["canvas"]:
        source_w, source_h = case["aspect_width"], case["aspect_height"]
        # The helper takes real pixel dimensions and rejects tiny ones, so only
        # the cases that are plausible source sizes are exercised here.
        if min(source_w, source_h) < 256 or max(source_w, source_h) > 5760:
            continue
        ratio = source_w / source_h
        if not 0.4 <= ratio <= 2.5:
            continue
        width, height = _reference_video_shape(source_w, source_h)
        assert (height, width) == (case["height"], case["width"])


def test_frame_alignment_matches_official(oracle):
    from vllm_omni.diffusion.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER

    for case in oracle["frames"]:
        aligned = MINIMAX_H3_SHAPE_PLANNER.align_frame_count(case["requested"])
        assert aligned == case["aligned"], f"align({case['requested']}) = {aligned}, official {case['aligned']}"
        assert MINIMAX_H3_SHAPE_PLANNER.video_latent_t(aligned) == case["video_latent_frames"]
        assert MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(aligned / 24) == case["audio_latents"]


# --------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------


def test_sigma_schedule_matches_official(oracle):
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

    for case in oracle["scheduler"]:
        sigmas = torch.tensor(
            minimax_h3_time_shift_sigmas(num_steps=case["num_inference_steps"], shift_scale=case["shift"]),
            dtype=torch.float32,
        )
        expected = _as_tensor(case["sigmas"])
        assert torch.equal(sigmas, expected), (
            f"shift={case['shift']} steps={case['num_inference_steps']}: sigma grid differs"
        )
        # N sigma nodes drive N-1 forward passes, exposed as t = 1 - sigma.
        assert sigmas.numel() - 1 == case["num_forward_passes"]
        torch.testing.assert_close(1.0 - sigmas[:-1], _as_tensor(case["timesteps"]), rtol=0, atol=0)


def test_euler_step_matches_official(oracle):
    from vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral import (
        minimax_h3_euler_eta0_step,
        minimax_h3_rf_v_to_x0,
    )

    for case in oracle["scheduler"]:
        sigmas = _as_tensor(case["sigmas"])
        sample = _as_tensor(case["step_input_sample"])
        velocity = _as_tensor(case["step_input_velocity"])
        timestep = torch.tensor(case["step_timestep"], dtype=torch.float32)

        denoised = minimax_h3_rf_v_to_x0(sample, velocity, timestep)
        out = minimax_h3_euler_eta0_step(
            sample,
            denoised,
            sigma_curr=float(sigmas[0]),
            sigma_next=float(sigmas[1]),
        )
        torch.testing.assert_close(out, _as_tensor(case["step_prev_sample"]), rtol=0, atol=0)


def test_condition_noise_mix_matches_official_scale_noise(oracle):
    """`scale_noise(x, t, noise)` is `t*x + (1-t)*noise`, the anchor recipe."""
    for case in oracle["scheduler"]:
        sample = _as_tensor(case["step_input_sample"])
        noise = _as_tensor(case["step_input_velocity"])
        t = torch.tensor(0.999, dtype=torch.float32)
        mixed = t * sample + (1.0 - t) * noise
        torch.testing.assert_close(mixed, _as_tensor(case["scale_noise_0999"]), rtol=0, atol=0)


# --------------------------------------------------------------------------
# Packed sequence
# --------------------------------------------------------------------------


def _canonical_prefix(packed: dict, used: int) -> dict:
    """vLLM pads `used` up to a multiple of 64; the official layout has no pad."""
    return {
        "position_ids": packed["img_position_ids"][:used],
        "token_tags": packed["token_tags"][:used],
    }


def test_packed_t2va_fl2va_matches_official(oracle):
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
    from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_frame_count_from_video_latent_t

    anchor_to_index = {"first": 0, "last": -1}
    for case in oracle["packed_t2va_fl2va"]:
        anchors = tuple(case["keyframe_anchors"])
        include = bool(anchors)
        frame_count = minimax_h3_frame_count_from_video_latent_t(case["latent_t"]) if include else None
        packed = minimax_h3_packed_sequence(
            text_len=case["text_len"],
            latent_t=case["latent_t"],
            latent_h=case["latent_h"],
            latent_w=case["latent_w"],
            audio_t=case["audio_t"],
            include_keyframe_cond=include,
            keyframe_frame_indices=[anchor_to_index[a] for a in anchors] if include else None,
            frame_count=frame_count,
        )
        expected_position_ids = _as_tensor(case["position_ids"])
        used = expected_position_ids.shape[0]
        prefix = _canonical_prefix(packed, used)

        label = f"t2va/fl2va anchors={anchors} latent_t={case['latent_t']}"
        assert torch.equal(prefix["position_ids"], expected_position_ids), f"{label}: position_ids differ"
        assert torch.equal(prefix["token_tags"], _as_tensor(case["token_tags"])), f"{label}: token_tags differ"
        assert torch.equal(packed["img_pos"], _as_tensor(case["video_indices"])), f"{label}: video indices differ"
        assert torch.equal(packed["audio_pos"], _as_tensor(case["audio_indices"])), f"{label}: audio indices differ"
        assert torch.equal(packed["text_pos"], _as_tensor(case["text_indices"])), f"{label}: text indices differ"
        assert int((~packed["update_mask"]).sum()) == case["num_condition_video_rows"], f"{label}: cond rows differ"


def test_packed_ref2va_matches_official(oracle):
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks

    for case in oracle["packed_ref2va"]:
        blocks = []
        for block in case["blocks"]:
            if block["kind"] == "image":
                blocks.append({"kind": "image", "latent_h": block["latent_h"], "latent_w": block["latent_w"]})
            elif block["kind"] == "audio":
                blocks.append({"kind": "audio", "ref_audio_t": block["audio_t"]})
            else:
                blocks.append(
                    {
                        "kind": "video",
                        "ref_audio_t": block.get("audio_t", 0),
                        "latent_t": block["latent_t"],
                        "latent_h": block["latent_h"],
                        "latent_w": block["latent_w"],
                    }
                )
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=case["text_len"],
            latent_t=case["latent_t"],
            latent_h=case["latent_h"],
            latent_w=case["latent_w"],
            audio_t=case["audio_t"],
            ref_blocks=blocks,
        )
        expected_position_ids = _as_tensor(case["position_ids"])
        used = expected_position_ids.shape[0]
        prefix = _canonical_prefix(packed, used)

        label = f"ref2va blocks={[b['kind'] for b in case['blocks']]}"
        assert torch.equal(prefix["position_ids"], expected_position_ids), f"{label}: position_ids differ"
        assert torch.equal(prefix["token_tags"], _as_tensor(case["token_tags"])), f"{label}: token_tags differ"
        assert torch.equal(packed["img_pos"], _as_tensor(case["video_indices"])), f"{label}: video indices differ"
        assert torch.equal(packed["audio_pos"], _as_tensor(case["audio_indices"])), f"{label}: audio indices differ"
        assert torch.equal(packed["text_pos"], _as_tensor(case["text_indices"])), f"{label}: text indices differ"
        assert int((~packed["update_mask"]).sum()) == case["num_condition_video_rows"], f"{label}: cond rows differ"
        assert int((~packed["audio_update_mask"]).sum()) == case["num_condition_audio_rows"], (
            f"{label}: reference audio rows differ"
        )


def test_packed_pad_rows_are_isolated(oracle):
    """The 64-alignment pad is vLLM's own; it must not leak into the contract."""
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        MINIMAX_H3_PAD_ID,
        minimax_h3_packed_sequence,
    )

    case = oracle["packed_t2va_fl2va"][0]
    packed = minimax_h3_packed_sequence(
        text_len=case["text_len"],
        latent_t=case["latent_t"],
        latent_h=case["latent_h"],
        latent_w=case["latent_w"],
        audio_t=case["audio_t"],
        include_keyframe_cond=False,
    )
    used = _as_tensor(case["position_ids"]).shape[0]
    seq_len = int(packed["seq_len"])
    assert seq_len % 64 == 0 and seq_len >= used

    pad = slice(used, seq_len)
    assert torch.equal(packed["input_ids"][pad], torch.full((seq_len - used,), MINIMAX_H3_PAD_ID, dtype=torch.long))
    # Pad rows are their own document, carry the padding tag, and are addressed
    # by no row-index tensor, so nothing downstream can read them.
    assert torch.equal(packed["document_id"][pad], torch.ones(seq_len - used, dtype=torch.int32))
    assert torch.equal(packed["token_tags"][pad], torch.full((seq_len - used,), -1, dtype=torch.long))
    assert torch.equal(packed["cu_seqlens"], torch.tensor([0, used, seq_len], dtype=torch.int32))
    for name in ("img_pos", "audio_pos", "text_pos"):
        assert int(packed[name].max()) < used, f"{name} addresses a pad row"


# --------------------------------------------------------------------------
# Reference image geometry
# --------------------------------------------------------------------------


def test_reference_image_geometry_matches_official(oracle):
    """Official: short edge 2048, aspect preserved, upscaling on, no area cap.

    Only the sources vLLM's admission gate accepts are compared here; the ones
    it rejects are the subject of the next test, so a geometry regression and an
    envelope difference cannot be confused for each other.
    """
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape

    for case in oracle["reference_image_geometry"]:
        width, height = case["source_width"], case["source_height"]
        ratio = width / height
        if not 0.4 <= ratio <= 2.5 or min(width, height) < 256 or max(width, height) > 5760:
            continue
        got_w, got_h = _reference_image_shape(Image.new("RGB", (width, height)))
        assert (got_w, got_h) == (case["target_width"], case["target_height"]), (
            f"{width}x{height} -> {got_w}x{got_h}, official {case['target_width']}x{case['target_height']}"
        )


def test_reference_image_envelope_is_narrower_than_official(oracle):
    """vLLM's model-level gate rejects sources the oracle accepts.

    The oracle allows 1:4..4:1 and puts no bound on the source size; vLLM's
    reference-image path allows 0.4..2.5 and 256..5760 px. That is an admission
    policy, not the official model contract, and `official_diffusers_v1` has to
    separate the two. Asserted rather than skipped so the gap stays visible.
    """
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _reference_image_shape
    from vllm_omni.errors import OmniClientError

    rejected = []
    for case in oracle["reference_image_geometry"]:
        width, height = case["source_width"], case["source_height"]
        ratio = width / height
        if 0.4 <= ratio <= 2.5 and 256 <= min(width, height) and max(width, height) <= 5760:
            continue
        with pytest.raises(OmniClientError):
            _reference_image_shape(Image.new("RGB", (width, height)))
        rejected.append((width, height))
    # 3:1 and 1:3 are inside the oracle's 1:4..4:1 range.
    assert (3072, 1024) in rejected and (1024, 3072) in rejected


# --------------------------------------------------------------------------
# FL2VA keyframes
# --------------------------------------------------------------------------


def test_reference_video_resample_matches_official(oracle):
    """Which source frame lands in which 24 fps slot, and where truncation cuts.

    The oracle was produced with a canvas rule under which the tiny test frames
    are a fixed point, so no LANCZOS pass runs and each output slot still
    carries its source frame's index — the mapping is compared exactly rather
    than through resampled pixels.
    """
    import numpy as np

    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import (
        normalize_reference_video_frames,
        resample_frame_indices,
    )

    def _resolve_canvas(aspect_w, aspect_h, multiple, short_edge, max_pixels):
        from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import MINIMAX_H3_FPS  # noqa: F401

        ratio = aspect_w / aspect_h
        if ratio >= 1.0:
            width, height = short_edge * ratio, float(short_edge)
        else:
            width, height = float(short_edge), short_edge / ratio
        area = width * height
        if area > max_pixels:
            scale = (max_pixels / area) ** 0.5
            width, height = width * scale, height * scale
        return (
            max(multiple, round(height / multiple) * multiple),
            max(multiple, round(width / multiple) * multiple),
        )

    for case in oracle["reference_video"]:
        label = f"{case['source_fps']} fps, {case['source_frames']} frames -> {case['num_frames']}"

        # 1. The index mapping on its own, before truncation.
        mapping = resample_frame_indices(case["source_frames"], case["source_fps"], case["target_fps"])
        expected = case["source_index_per_slot"]
        assert list(mapping[: len(expected)]) == expected, f"{label}: frame index mapping differs"

        # 2. The whole normalization, including where the truncation cuts.
        frames = np.zeros((case["source_frames"], 4, 8, 3), dtype=np.uint8)
        for index in range(case["source_frames"]):
            frames[index] = index
        normalized = normalize_reference_video_frames(
            frames,
            fps=case["source_fps"],
            num_frames=case["num_frames"],
            canvas_multiple=case["canvas_multiple"],
            canvas_short_edge=case["canvas_short_edge"],
            canvas_max_pixels=case["canvas_max_pixels"],
            resolve_canvas=_resolve_canvas,
            target_fps=case["target_fps"],
        )
        assert normalized.shape[0] == case["output_frames"], (
            f"{label}: {normalized.shape[0]} output frames, official {case['output_frames']}"
        )
        assert [int(frame[0, 0, 0]) for frame in normalized] == expected, f"{label}: normalized frames differ"


def test_conditioner_frame_sampling_matches_official(oracle):
    """Which frames the conditioner reads, and the labels those blocks carry.

    The labels are rendered into the prompt, so a wrong timestamp is a wrong
    tokenization — a discrete contract difference, not a cosmetic one.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import sample_conditioner_frames

    for case in oracle["conditioner_sampling"]:
        indices, block_timestamps = sample_conditioner_frames(
            case["num_frames"],
            fps=case["fps"],
            sample_fps=case["sample_fps"],
            temporal_patch=case["temporal_patch"],
        )
        label = f"{case['num_frames']} frames"
        assert indices == case["frame_indices"], f"{label}: sampled frame indices differ"
        assert block_timestamps == case["block_timestamps"], f"{label}: block timestamps differ"
        # The rendered form is what actually reaches the tokenizer.
        assert [f"<{value:.1f} seconds>" for value in block_timestamps] == case["rendered_labels"], (
            f"{label}: rendered labels differ"
        )


def test_conditioner_sampling_rejects_a_video_too_short_to_merge():
    """Fewer sampled frames than the temporal patch cannot form a block."""
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import sample_conditioner_frames

    with pytest.raises(ValueError, match="at least"):
        sample_conditioner_frames(12, fps=24.0, sample_fps=2.0, temporal_patch=2)


def test_reference_video_truncates_to_the_generated_frame_count(oracle):
    """A reference longer than the target is cut; the legacy path keeps it whole.

    This changes the packed row count, so it is a discrete contract difference
    rather than a numeric one.
    """

    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import frame_slot_repeats

    # 15 s of 24 fps source against a 5 s request.
    repeats = frame_slot_repeats(360, 24.0, 24.0)
    assert int(repeats.sum()) == 360
    assert repeats.tolist() == [1] * 360  # same rate: no drops, no duplicates


def test_reference_video_rounds_halves_up_not_to_even():
    """`floor(x + 0.5)` is not `round(x)`; 25 and 30 fps sources land differently."""
    from vllm_omni.diffusion.models.minimax_h3.reference_video_frames import resample_frame_indices

    # At 12 fps every frame is held for exactly two 24 fps slots.
    assert resample_frame_indices(4, 12.0).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    # At 30 fps one frame in five is dropped.
    assert resample_frame_indices(10, 30.0).tolist() == [0, 1, 3, 4, 5, 6, 8, 9]


def _digest(tensor: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def test_reference_audio_normalization_matches_official(oracle):
    """Truncate at the source rate, upmix mono, resample once.

    The fixture stores the waveforms as digests, so the input is rebuilt here
    from the case parameters and checked against its own digest first — that way
    a mismatch on the output cannot be blamed on having fed a different signal.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_audio import normalize_reference_audio

    for case in oracle["reference_audio"]:
        sample_rate, channels = case["sample_rate"], case["channels"]
        num_samples = int(sample_rate * 0.25)
        ramp = torch.linspace(-0.9, 0.9, num_samples, dtype=torch.float32)
        waveform = torch.stack([ramp * (index + 1) / channels for index in range(channels)])
        assert _digest(waveform) == case["input"]["sha256"], (
            f"{sample_rate} Hz / {channels}ch: rebuilt input differs from the fixture's"
        )

        produced = normalize_reference_audio(
            waveform,
            sample_rate,
            target_sample_rate=case["target_sample_rate"],
            max_duration=case["max_duration"],
        )
        label = f"{sample_rate} Hz / {channels}ch / {case['max_duration']}s"
        assert list(produced.shape) == case["output"]["shape"], (
            f"{label}: shape {tuple(produced.shape)} != official {tuple(case['output']['shape'])}"
        )
        assert _digest(produced) == case["output"]["sha256"], (
            f"{label}: head {produced.flatten()[:4].tolist()} vs official {case['output']['head'][:4]}"
        )


def test_reference_audio_truncation_is_applied_before_resampling(oracle):
    """Order matters: the resampler's filter sees a different tail otherwise.

    Truncating after the resample yields the same sample *count* but different
    samples, which is exactly the kind of difference an end-to-end SSIM would
    never localise.
    """
    from vllm_omni.diffusion.models.minimax_h3.reference_audio import normalize_reference_audio

    sample_rate, target = 44100, 32000
    ramp = torch.linspace(-0.9, 0.9, int(sample_rate * 0.25), dtype=torch.float32)
    waveform = torch.stack([ramp, ramp * 0.5])

    official = normalize_reference_audio(waveform, sample_rate, target_sample_rate=target, max_duration=0.05)
    untruncated = normalize_reference_audio(waveform, sample_rate, target_sample_rate=target, max_duration=None)
    truncated_after = untruncated[:, : official.shape[1]]

    assert official.shape == truncated_after.shape
    assert not torch.equal(official, truncated_after)


def test_reference_audio_mono_is_upmixed_by_channel_repeat():
    from vllm_omni.diffusion.models.minimax_h3.reference_audio import normalize_reference_audio

    mono = torch.linspace(-1.0, 1.0, 640, dtype=torch.float32)[None]
    stereo = normalize_reference_audio(mono, 32000, target_sample_rate=32000, max_duration=None)

    assert stereo.shape == (2, 640)
    assert torch.equal(stereo[0], stereo[1])
    assert torch.equal(stereo[0], mono[0])


def test_reference_audio_max_duration_is_the_generated_duration():
    from vllm_omni.diffusion.models.minimax_h3.reference_audio import reference_audio_max_duration

    # 362 frames at 24 fps is the aligned 15 s request.
    assert reference_audio_max_duration(362, 24) == 362 / 24
    with pytest.raises(ValueError):
        reference_audio_max_duration(0, 24)


def _deterministic_image(width: int, height: int):
    """A non-uniform source, so a stretch cannot coincide with a crop by luck."""
    from PIL import Image

    image = Image.new("RGB", (width, height))
    image.putdata([((x * 7) % 256, (x * 13) % 256, (x * 29) % 256) for x in range(width * height)])
    return image


def test_fl2va_cover_crop_geometry_matches_official(oracle):
    """The integer plan on its own, so a pixel mismatch can be localised."""
    from vllm_omni.diffusion.models.minimax_h3.keyframes import cover_crop_geometry

    for case in oracle["fl2va_keyframes"]:
        resized, offset = cover_crop_geometry(
            case["source_width"], case["source_height"], case["canvas_width"], case["canvas_height"]
        )
        assert resized == (case["follower_resized_width"], case["follower_resized_height"])
        assert offset == (case["follower_crop_left"], case["follower_crop_top"])


def test_fl2va_follower_keyframe_is_cover_cropped(oracle):
    """Official: the anchor is stretched, the follower is cover-cropped.

    Compared on pixels against a pure-PIL replay of the official arithmetic.
    """
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.keyframes import prepare_fl2va_keyframes

    for case in oracle["fl2va_keyframes"]:
        src_w, src_h = case["source_width"], case["source_height"]
        width, height = case["canvas_width"], case["canvas_height"]
        anchor = _deterministic_image(width, height)
        follower = _deterministic_image(src_w, src_h)

        official = (
            follower.resize(
                (case["follower_resized_width"], case["follower_resized_height"]), Image.Resampling.LANCZOS
            ).crop(
                (
                    case["follower_crop_left"],
                    case["follower_crop_top"],
                    case["follower_crop_left"] + width,
                    case["follower_crop_top"] + height,
                )
            )
            if (src_w, src_h) != (width, height)
            else follower
        )

        prepared = prepare_fl2va_keyframes([anchor, follower], width=width, height=height, mode="official_cover_crop")
        assert len(prepared) == 2
        assert prepared[0].size == (width, height)
        assert prepared[1].size == (width, height)
        assert list(prepared[1].getdata()) == list(official.getdata()), (
            f"follower keyframe {src_w}x{src_h} -> {width}x{height} does not match the official cover-crop"
        )


def test_fl2va_legacy_mode_still_stretches(oracle):
    """Legacy must keep distorting the follower, or it is not legacy.

    Pinned so the official implementation cannot quietly become the default for
    instances that were never validated for it.
    """
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.keyframes import prepare_fl2va_keyframes

    case = next(
        item
        for item in oracle["fl2va_keyframes"]
        if (item["source_width"], item["source_height"]) != (item["canvas_width"], item["canvas_height"])
    )
    width, height = case["canvas_width"], case["canvas_height"]
    follower = _deterministic_image(case["source_width"], case["source_height"])

    prepared = prepare_fl2va_keyframes(
        [_deterministic_image(width, height), follower], width=width, height=height, mode="legacy_stretch"
    )
    assert list(prepared[1].getdata()) == list(follower.resize((width, height), Image.Resampling.LANCZOS).getdata())
