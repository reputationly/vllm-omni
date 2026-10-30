# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pika-style renoise sampler and distilled-release contract tests.

The renoise solver implements the upstream Pika update: per non-final step it
predicts x0 and lands on the next sigma boundary by re-noising with a fresh
Gaussian draw, ``x_next = (1 - sigma_next) * x0 + sigma_next * eps``. All draws
come from one dedicated per-request generator the pipeline seeds with
``seed + 1`` and shares between the video and audio solvers, which the loops
step in that order.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_SIGMAS = [1.0, 0.5, 0.25, 0.0]


def _fl2va_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs = dict(
        task="fl2va",
        text_embeddings=torch.zeros(2, 4),
        text_tags=torch.ones(2, dtype=torch.long),
        seed=42,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        num_frames=5,
        num_steps=3,
        video_shift=12.0,
        audio_shift=3.0,
        base_schedule=None,
        visual_condition=torch.linspace(0, 1, 6 * 96, dtype=torch.float32).reshape(6, 96),
        visual_condition_shape=(1, 4, 6),
        audio_condition=None,
        ref_audio_t=None,
        keyframe_frame_indices=[0],
    )
    kwargs.update(overrides)
    return kwargs


def _pipeline() -> Any:
    """A pipeline instance carrying only what the helpers under test touch."""
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    instance = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(instance)
    instance.device = torch.device("cpu")
    return instance


def test_normalize_h3_sampler_accepts_renoise():
    from vllm_omni.diffusion.models.minimax_h3.sampling import normalize_h3_sampler

    assert normalize_h3_sampler(None) == "euler"
    assert normalize_h3_sampler("euler") == "euler"
    assert normalize_h3_sampler("res_multistep") == "res_multistep"
    assert normalize_h3_sampler("renoise") == "renoise"
    with pytest.raises(ValueError, match="Unsupported MiniMax H3 sampler"):
        normalize_h3_sampler("dpm")


def test_renoise_solver_requires_a_generator():
    from vllm_omni.diffusion.models.minimax_h3.sampling import create_h3_sample_solver

    with pytest.raises(ValueError, match="requires a dedicated torch.Generator"):
        create_h3_sample_solver("renoise", _SIGMAS)


def test_renoise_final_step_returns_x0_without_drawing_noise(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.sampling import create_h3_sample_solver

    generator = torch.Generator().manual_seed(43)
    solver = create_h3_sample_solver("renoise", _SIGMAS, generator=generator)
    state = torch.randn(6, 96, generator=torch.Generator().manual_seed(1))
    x0 = torch.randn(6, 96, generator=torch.Generator().manual_seed(2)) * 0.1

    draws = []
    real_randn = torch.randn

    def counting_randn(*args, **kwargs):
        draws.append((args, kwargs))
        return real_randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", counting_randn)
    solver.step(state, x0, 0)
    solver.step(state, x0, 1)
    assert len(draws) == 2

    generator_state = generator.get_state().clone()
    out = solver.step(state, x0, 2)

    assert len(draws) == 2
    assert torch.equal(generator.get_state(), generator_state)
    assert out.dtype == state.dtype
    assert torch.equal(out, x0)


def test_renoise_intermediate_step_matches_the_formula():
    from vllm_omni.diffusion.models.minimax_h3.sampling import create_h3_sample_solver

    seed = 7
    state = torch.randn(4, 96, generator=torch.Generator().manual_seed(1))
    x0 = torch.randn(4, 96, generator=torch.Generator().manual_seed(2)) * 0.1
    eps = torch.randn(state.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)

    generator = torch.Generator().manual_seed(seed)
    solver = create_h3_sample_solver("renoise", _SIGMAS, generator=generator)
    out = solver.step(state, x0, 0)

    sigma_next = _SIGMAS[1]
    expected = ((1.0 - sigma_next) * x0.float() + sigma_next * eps).to(state.dtype)
    assert torch.equal(out, expected)
    # A sigma below one must actually mix the draw in.
    assert not torch.equal(out, (1.0 - sigma_next) * x0)


def test_renoise_same_seed_reproduces_the_update():
    from vllm_omni.diffusion.models.minimax_h3.sampling import create_h3_sample_solver

    state = torch.randn(4, 96, generator=torch.Generator().manual_seed(1))
    x0 = torch.randn(4, 96, generator=torch.Generator().manual_seed(2)) * 0.1

    def run(seed: int) -> list[torch.Tensor]:
        solver = create_h3_sample_solver("renoise", _SIGMAS, generator=torch.Generator().manual_seed(seed))
        outs = []
        out = state
        for step in range(len(_SIGMAS) - 1):
            out = solver.step(out, x0, step)
            outs.append(out)
        return outs

    first, repeat, other = run(43), run(43), run(44)
    for left, right in zip(first, repeat, strict=True):
        assert torch.equal(left, right)
    # The noise draws diverge per seed on every non-final step; the terminal
    # step is the deterministic x0 passthrough and must not.
    assert not torch.equal(first[0], other[0])
    assert not torch.equal(first[1], other[1])
    assert torch.equal(first[2], other[2])


def test_renoise_video_steps_before_audio_on_one_generator():
    """Both solvers share one generator, and the video draw comes first."""
    from vllm_omni.diffusion.models.minimax_h3.sampling import create_h3_sample_solver

    sigmas_video = [1.0, 0.9, 0.4, 0.0]
    sigmas_audio = [1.0, 0.8, 0.3, 0.0]
    video_state = torch.randn(6, 96, generator=torch.Generator().manual_seed(1))
    audio_state = torch.randn(4, 32, generator=torch.Generator().manual_seed(2))
    video_x0 = torch.randn(6, 96, generator=torch.Generator().manual_seed(3)) * 0.1
    audio_x0 = torch.randn(4, 32, generator=torch.Generator().manual_seed(4)) * 0.1

    generator = torch.Generator().manual_seed(11)
    video_solver = create_h3_sample_solver("renoise", sigmas_video, generator=generator)
    audio_solver = create_h3_sample_solver("renoise", sigmas_audio, generator=generator)

    out_video = video_solver.step(video_state, video_x0, 0)
    out_audio = audio_solver.step(audio_state, audio_x0, 0)

    reference = torch.Generator().manual_seed(11)
    eps_video = torch.randn(video_state.shape, generator=reference, dtype=torch.float32)
    eps_audio = torch.randn(audio_state.shape, generator=reference, dtype=torch.float32)
    torch.testing.assert_close(out_video, (1.0 - 0.9) * video_x0 + 0.9 * eps_video)
    torch.testing.assert_close(out_audio, (1.0 - 0.8) * audio_x0 + 0.8 * eps_audio)


def test_release_metadata_reads_sampler_and_conditioning_numerics():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _read_conditioning_numerics,
        _read_release_sampler,
    )

    assert _read_release_sampler({}) is None
    assert _read_release_sampler({"sampler": "renoise"}) == "renoise"
    with pytest.raises(ValueError, match="Unsupported MiniMax H3 sampler"):
        _read_release_sampler({"sampler": "bogus"})

    assert _read_conditioning_numerics({}) is None
    assert _read_conditioning_numerics({"conditioning_numerics": "bf16_conditioning_v1"}) == "bf16_conditioning_v1"
    assert _read_conditioning_numerics({"conditioning_numerics": "fp32_conditioning_v1"}) == "fp32_conditioning_v1"
    with pytest.raises(ValueError, match="conditioning_numerics"):
        _read_conditioning_numerics({"conditioning_numerics": "fp16_conditioning_v1"})


def test_request_sampler_falls_back_to_the_partition_default():
    pipeline = _pipeline()
    assert pipeline._resolve_request_sampler("t2va", {}) == "euler"
    assert pipeline._resolve_request_sampler("t2va", {"sampler": "renoise"}) == "renoise"

    pipeline._sampler_by_partition = {"fl2va": "renoise", "ref2va": "euler"}
    assert pipeline._resolve_request_sampler("t2va", {}) == "renoise"
    assert pipeline._resolve_request_sampler("fl2va", {}) == "renoise"
    assert pipeline._resolve_request_sampler("ref2va", {}) == "euler"
    # A request-level value still wins over the release default.
    assert pipeline._resolve_request_sampler("t2va", {"sampler": "euler"}) == "euler"
    with pytest.raises(ValueError, match="Unsupported MiniMax H3 sampler"):
        pipeline._resolve_request_sampler("t2va", {"sampler": "bogus"})


def test_res_multistep_still_rejects_a_pinned_base_schedule():
    pipeline = _pipeline()
    base_schedule = [1.0, 0.75, 0.5, 0.25, 0.0]

    with pytest.raises(ValueError, match="res_multistep"):
        pipeline._build_denoise_inputs(
            **_fl2va_kwargs(sampler="res_multistep", base_schedule=base_schedule, num_steps=4)
        )
    for sampler in ("euler", "renoise"):
        inputs = pipeline._build_denoise_inputs(
            **_fl2va_kwargs(sampler=sampler, base_schedule=base_schedule, num_steps=4)
        )
        assert inputs["sampler"] == sampler


def test_keyframe_bf16_rounding_only_applies_under_the_bf16_policy(monkeypatch):
    import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as mod

    pipeline = _pipeline()
    captured: list[torch.Tensor] = []
    original = mod.minimax_h3_imgvid_cond_noise_aug_rows

    def spy(clean_rows, **kwargs):
        captured.append(clean_rows.clone())
        return original(clean_rows, **kwargs)

    monkeypatch.setattr(mod, "minimax_h3_imgvid_cond_noise_aug_rows", spy)
    kwargs = _fl2va_kwargs()

    for policy, expect_rounded in (
        (None, False),
        ("fp32_conditioning_v1", False),
        ("bf16_conditioning_v1", True),
    ):
        captured.clear()
        pipeline._conditioning_numerics_by_partition = {"fl2va": policy}
        pipeline._build_denoise_inputs(**kwargs)
        assert len(captured) == 1
        expected = kwargs["visual_condition"].to(torch.bfloat16).to(torch.float32)
        if expect_rounded:
            assert torch.equal(captured[0], expected)
            assert not torch.equal(captured[0], kwargs["visual_condition"])
        else:
            assert torch.equal(captured[0], kwargs["visual_condition"])


def _encoder_prompt(task: str = "t2va") -> dict[str, Any]:
    from vllm_omni.diffusion.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderConditioning

    conditioning = MiniMaxH3EncoderConditioning(
        hidden_states=torch.ones(1, 5120, dtype=torch.bfloat16),
        token_tags=torch.ones(1, dtype=torch.int64),
        task=task,
        height=768,
        width=1344,
        num_frames=124,
        latent_t=MINIMAX_H3_SHAPE_PLANNER.video_latent_t(124),
        audio_t=MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(124 / 24),
    )
    return {"prompt": "test", "additional_information": {"encoder_output": conditioning.to_omni_payload()}}


def _request_batch(extra_args: dict[str, Any]):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(
        quality="lossless",
        width=1344,
        height=768,
        fps=24,
        num_frames=124,
        num_inference_steps=6,
        extra_args={"task": "t2va", "aspect_ratio": "16:9", **extra_args},
    )
    return DiffusionRequestBatch(
        [OmniDiffusionRequest(prompt=_encoder_prompt(), sampling_params=sampling, request_id="renoise")]
    )


def _distilled_renoise_pipeline(diffuse_calls: list[dict[str, Any]]):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.sched import DMD2SigmaSchedule

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "combined"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace()
    pipeline._base_schedule_by_partition = {
        "fl2va": DMD2SigmaSchedule.from_positions([1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0]),
        "ref2va": None,
    }
    pipeline._sampler_by_partition = {"fl2va": "renoise", "ref2va": None}
    pipeline.load_text_encoder = False
    pipeline.load_vae_encoder = False
    pipeline._quality_policy = Mock()
    pipeline._quality_policy.resolve.return_value = SimpleNamespace(cache_dit=None)
    pipeline._cache_dit_runtime = SimpleNamespace(prepare=lambda spec: None)

    def diffuse(**kwargs):
        diffuse_calls.append(kwargs)
        return torch.zeros(1), torch.zeros(1)

    pipeline.diffuse = diffuse
    pipeline.decode = Mock(return_value=(torch.zeros(1, 3, 1, 1, 1), torch.zeros(1)))
    # The CPU platform has no allocator cache to release; the sampler contract
    # under test lives upstream of that boundary.
    pipeline._release_stage_cache = lambda: None
    return pipeline


def test_release_sampler_default_reaches_diffuse_and_request_overrides():
    diffuse_calls: list[dict[str, Any]] = []
    pipeline = _distilled_renoise_pipeline(diffuse_calls)

    pipeline.forward(_request_batch({}))
    assert diffuse_calls[-1]["sampler"] == "renoise"

    pipeline.forward(_request_batch({"sampler": "euler"}))
    assert diffuse_calls[-1]["sampler"] == "euler"

    pipeline.forward(_request_batch({"sampler": "renoise"}))
    assert diffuse_calls[-1]["sampler"] == "renoise"


def test_request_res_multistep_rejects_the_pinned_release_schedule():
    pipeline = _distilled_renoise_pipeline([])

    with pytest.raises(ValueError, match="res_multistep"):
        pipeline.forward(_request_batch({"sampler": "res_multistep"}))
