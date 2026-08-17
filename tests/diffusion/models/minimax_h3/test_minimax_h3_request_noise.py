# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MiniMax-H3 request RNG contract, both modes.

``legacy`` must keep producing exactly what the pipeline produced before the
draws were extracted into ``request_noise.py`` — that is a refactor, and a
refactor that changes a seed's output is a regression. ``official_diffusers_v1``
must reproduce the pinned Diffusers draw order, tensor for tensor, including
where it leaves the generator.

No weights, no device, no network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

FIXTURE = Path(__file__).parent / "fixtures" / "minimax_h3_official_contract_v1.json"


@pytest.fixture(scope="module")
def rng_oracle() -> list[dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["rng"]


def _as_tensor(blob: dict) -> torch.Tensor:
    return torch.tensor(blob["data"], dtype=getattr(torch, blob["dtype"])).reshape(blob["shape"])


# --------------------------------------------------------------------------
# legacy: the refactor must not move a single value
# --------------------------------------------------------------------------


def test_legacy_initial_rows_match_the_pre_refactor_recipe():
    """Spelled out rather than imported, so it pins behaviour, not code."""
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_patchify_video_latent
    from vllm_omni.diffusion.models.minimax_h3.request_noise import (
        MINIMAX_H3_RNG_LEGACY,
        MiniMaxH3RequestNoisePlan,
    )

    seed, latent_t, latent_h, latent_w, audio_t = 1234, 3, 4, 6, 5

    # What `_initial_noise` did: a fresh generator for the video, and another
    # fresh generator on the SAME seed for the audio.
    video = torch.randn(
        1,
        24,
        latent_t,
        latent_h,
        latent_w,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    )
    expected_video_rows = minimax_h3_patchify_video_latent(video, patch_size=(1, 2, 2))
    expected_audio_rows = torch.randn(
        audio_t * 2,
        32,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    )

    plan = MiniMaxH3RequestNoisePlan(rng_mode=MINIMAX_H3_RNG_LEGACY, seed=seed)
    video_rows, audio_rows = plan.initial_rows(latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, audio_t=audio_t)

    assert torch.equal(video_rows, expected_video_rows)
    assert torch.equal(audio_rows, expected_audio_rows)


def test_legacy_visual_condition_draw_matches_the_shipped_condition_noise():
    """The legacy oversized draw-and-slice, checked through the real helper.

    Feeding clean rows of zeros makes the helper's output ``(1 - t) * noise``,
    so comparing outputs compares the draws without reaching inside it.
    """
    from vllm_omni.diffusion.models.minimax_h3.condition_noise import minimax_h3_imgvid_cond_noise_aug_rows
    from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_patchify_video_latent
    from vllm_omni.diffusion.models.minimax_h3.request_noise import (
        MINIMAX_H3_RNG_LEGACY,
        MiniMaxH3RequestNoisePlan,
    )

    seed, noise_aug, target_latent_t = 7, 0.999, 4
    shapes = [(1, 4, 6), (2, 2, 2)]
    rows = sum(t * (h // 2) * (w // 2) for t, h, w in shapes)
    clean = torch.zeros(rows, 96, dtype=torch.float32)

    produced = minimax_h3_imgvid_cond_noise_aug_rows(
        clean,
        condition_shapes=shapes,
        target_latent_t=target_latent_t,
        imgvid_cond_num_frames=len(shapes),
        seed=seed,
        noise_aug=noise_aug,
    )

    plan = MiniMaxH3RequestNoisePlan(rng_mode=MINIMAX_H3_RNG_LEGACY, seed=seed)
    drawn = plan.draw_visual_condition_noise(shapes, target_latent_t=target_latent_t)
    # The helper mixes with a float32 `timestep` tensor; folding the scalar in
    # float64 instead would move the result by an ulp and hide a real drift.
    timestep = torch.tensor(noise_aug, dtype=torch.float32)
    expected = torch.cat(
        [
            timestep * torch.zeros_like(minimax_h3_patchify_video_latent(noise, patch_size=(1, 2, 2)))
            + (1.0 - timestep) * minimax_h3_patchify_video_latent(noise, patch_size=(1, 2, 2))
            for noise in drawn
        ]
    )

    torch.testing.assert_close(produced, expected, rtol=0, atol=0)


def test_legacy_video_and_audio_share_the_stream_start():
    """The legacy quirk itself: audio restarts the stream the video used."""
    from vllm_omni.diffusion.models.minimax_h3.request_noise import (
        MINIMAX_H3_RNG_LEGACY,
        MiniMaxH3RequestNoisePlan,
    )

    plan = MiniMaxH3RequestNoisePlan(rng_mode=MINIMAX_H3_RNG_LEGACY, seed=99)
    video = plan.draw_video_noise(latent_t=2, latent_h=2, latent_w=2)
    audio = plan.draw_audio_noise(audio_t=2)
    # 4 * 32 = 128 audio values against the first 128 of the video draw.
    assert torch.equal(audio.flatten()[:96], video.flatten()[:96])
    assert plan.generator_state() is None


# --------------------------------------------------------------------------
# official_diffusers_v1: reproduce the pinned oracle
# --------------------------------------------------------------------------


def test_official_draw_sequence_matches_oracle(rng_oracle):
    from vllm_omni.diffusion.models.minimax_h3.request_noise import (
        MINIMAX_H3_RNG_OFFICIAL_V1,
        MiniMaxH3RequestNoisePlan,
    )

    for case in rng_oracle:
        plan = MiniMaxH3RequestNoisePlan(rng_mode=MINIMAX_H3_RNG_OFFICIAL_V1, seed=case["seed"])
        produced = []
        if case["conditions"]:
            produced += plan.draw_visual_condition_noise(case["conditions"], target_latent_t=case["latent_t"])
        produced.append(
            plan.draw_video_noise(latent_t=case["latent_t"], latent_h=case["latent_h"], latent_w=case["latent_w"])
        )
        produced.append(plan.draw_audio_noise(audio_t=case["audio_t"]))

        expected = [_as_tensor(draw["tensor"]) for draw in case["draws"]]
        assert len(produced) == len(expected), f"{case['name']}: drew {len(produced)} tensors, oracle {len(expected)}"
        for index, (got, want) in enumerate(zip(produced, expected)):
            kind = case["draws"][index]["kind"]
            assert got.shape == want.shape, f"{case['name']} draw {index} ({kind}): shape {got.shape} != {want.shape}"
            assert torch.equal(got, want), f"{case['name']} draw {index} ({kind}): values differ"

        state = plan.generator_state()
        assert state is not None
        digest = hashlib.sha256(state.numpy().tobytes()).hexdigest()
        assert digest == case["final_generator_state_sha256"], (
            f"{case['name']}: the request consumed a different amount of the stream than the oracle"
        )


def test_official_rejects_out_of_contract_draw_order():
    from vllm_omni.diffusion.models.minimax_h3.request_noise import (
        MINIMAX_H3_RNG_OFFICIAL_V1,
        MiniMaxH3RequestNoisePlan,
    )

    plan = MiniMaxH3RequestNoisePlan(rng_mode=MINIMAX_H3_RNG_OFFICIAL_V1, seed=42)
    plan.draw_video_noise(latent_t=2, latent_h=2, latent_w=2)
    with pytest.raises(RuntimeError, match="in that order"):
        plan.draw_visual_condition_noise([(1, 2, 2)], target_latent_t=2)


def test_official_changes_t2va_output_for_the_same_seed():
    """The contracts differ even with nothing to condition on.

    Worth asserting rather than assuming: it is the reason the official mode is
    a startup-level choice instead of a transparent fix.
    """
    from vllm_omni.diffusion.models.minimax_h3.request_noise import (
        MINIMAX_H3_RNG_LEGACY,
        MINIMAX_H3_RNG_OFFICIAL_V1,
        MiniMaxH3RequestNoisePlan,
    )

    shape = {"latent_t": 2, "latent_h": 2, "latent_w": 2}
    legacy = MiniMaxH3RequestNoisePlan(rng_mode=MINIMAX_H3_RNG_LEGACY, seed=42)
    official = MiniMaxH3RequestNoisePlan(rng_mode=MINIMAX_H3_RNG_OFFICIAL_V1, seed=42)

    # The video draw is the stream start in both, so it agrees...
    assert torch.equal(legacy.draw_video_noise(**shape), official.draw_video_noise(**shape))
    # ...and the audio draw is where they part company.
    assert not torch.equal(legacy.draw_audio_noise(audio_t=3), official.draw_audio_noise(audio_t=3))


def test_unknown_rng_mode_is_rejected():
    from vllm_omni.diffusion.models.minimax_h3.request_noise import MiniMaxH3RequestNoisePlan

    with pytest.raises(ValueError, match="rng_mode"):
        MiniMaxH3RequestNoisePlan(rng_mode="official", seed=0)
