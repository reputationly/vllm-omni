# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.condition_noise import minimax_h3_imgvid_cond_noise_rows
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _MINIMAX_H3_DENOISE_INPUT_KEYS,
    MiniMaxH3Pipeline,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def pipeline():
    # Exercise real CPU tensor methods without loading model weights.
    instance = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(instance)
    instance.device = torch.device("cpu")
    return instance


@pytest.mark.parametrize("seed", [0, 42, 2101])
@pytest.mark.parametrize("shape", [(2, 4, 6, 3), (7, 8, 10, 12)])
def test_background_initial_noise_matches_direct_sampling(pipeline, seed, shape):
    latent_t, latent_h, latent_w, audio_t = shape
    params = dict(seed=seed, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, audio_t=audio_t)
    rng_state = torch.get_rng_state().clone()
    expected = pipeline._initial_noise(**params)
    with ThreadPoolExecutor(max_workers=1) as executor:
        actual = executor.submit(pipeline._initial_noise, **params).result()

    for direct, prepared in zip(expected, actual, strict=True):
        assert torch.equal(prepared.view(torch.int32), direct.view(torch.int32))
        assert prepared.device.type == "cpu"
        assert prepared.dtype == torch.float32
        assert prepared.is_contiguous()
    assert torch.equal(torch.get_rng_state(), rng_state)


def _ref2va_kwargs(seed):
    return dict(
        task="ref2va",
        text_embeddings=torch.zeros(2, 4),
        text_tags=torch.ones(2, dtype=torch.long),
        seed=seed,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        num_frames=22,
        num_steps=3,
        video_shift=12.0,
        audio_shift=3.0,
        base_schedule=None,
        visual_condition=torch.arange(6 * 96, dtype=torch.float32).reshape(6, 96) / 100,
        visual_condition_shape=(1, 4, 6),
        audio_condition=torch.arange(4 * 32, dtype=torch.float32).reshape(4, 32) / 100,
        ref_audio_t=2,
        ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 6}, {"kind": "audio", "ref_audio_t": 2}],
    )


@pytest.mark.parametrize("seed", [42, 2101])
def test_ref2va_precomputed_noise_preserves_denoise_inputs(pipeline, seed):
    kwargs = _ref2va_kwargs(seed)
    expected = pipeline._build_denoise_inputs(**kwargs)
    with ThreadPoolExecutor(max_workers=1) as executor:
        noise = executor.submit(
            pipeline._initial_noise, seed=seed, latent_t=2, latent_h=4, latent_w=6, audio_t=3
        ).result()
        visual_noise = executor.submit(
            minimax_h3_imgvid_cond_noise_rows,
            condition_shapes=[kwargs["visual_condition_shape"]],
            target_latent_t=kwargs["latent_t"],
            imgvid_cond_num_frames=1,
            seed=seed,
        ).result()
    actual = pipeline._build_denoise_inputs(
        **kwargs, precomputed_initial_noise=noise, precomputed_visual_condition_noise=visual_noise
    )

    for key in ("video_rows", "audio_rows", "cond_anchor", "audio_anchor"):
        assert torch.equal(actual[key].view(torch.int32), expected[key].view(torch.int32))
    assert actual["sigmas_video"] == expected["sigmas_video"]
    assert actual["sigmas_audio"] == expected["sigmas_audio"]
    assert torch.equal(actual["branch"].token_tags_dev, expected["branch"].token_tags_dev)


def test_shared_denoise_kwargs_exclude_first_seed_noise(pipeline):
    context = {key: None for key in _MINIMAX_H3_DENOISE_INPUT_KEYS}
    context.update(_ref2va_kwargs(42))
    context["precomputed_initial_noise"] = pipeline._initial_noise(
        seed=42, latent_t=2, latent_h=4, latent_w=6, audio_t=3
    )
    context["precomputed_visual_condition_noise"] = torch.zeros(4, 96)

    kwargs = pipeline._denoise_kwargs(context)
    assert "precomputed_initial_noise" not in kwargs
    assert "precomputed_visual_condition_noise" not in kwargs
    assert kwargs["seed"] == 42
    assert kwargs["latent_t"] == 2
    assert kwargs["latent_h"] == 4
    assert kwargs["latent_w"] == 6
    assert kwargs["audio_t"] == 3


def test_consumed_host_noise_is_released_before_denoising(pipeline, monkeypatch):
    kwargs = _ref2va_kwargs(42)
    noise = pipeline._initial_noise(seed=42, latent_t=2, latent_h=4, latent_w=6, audio_t=3)
    visual_noise = minimax_h3_imgvid_cond_noise_rows(
        condition_shapes=[kwargs["visual_condition_shape"]], target_latent_t=2, imgvid_cond_num_frames=1, seed=42
    )
    references = [weakref.ref(tensor) for tensor in (*noise, visual_noise)]
    context = {"precomputed_initial_noise": noise, "precomputed_visual_condition_noise": visual_noise}
    del noise, visual_noise
    # Real CPU-to-meta copies have distinct storage without loading model weights.
    pipeline.device = torch.device("meta")

    def stop_before_denoising(task):
        assert all(reference() is None for reference in references)
        raise RuntimeError("checked noise lifetime")

    monkeypatch.setattr(pipeline, "_transformer_for_task", stop_before_denoising)
    with pytest.raises(RuntimeError, match="checked noise lifetime"):
        pipeline.diffuse(
            **kwargs,
            request_context=context,
        )


@pytest.mark.parametrize("failure_stage", ["text", "media"])
def test_encoder_failure_leaves_no_noise_worker_and_an_untouched_context(pipeline, monkeypatch, failure_stage):
    """Deliberate deviation from upstream #8577 -- see the producer site and manual §3.2.

    Upstream prepares the Ref2VA initial/visual-condition noise on a worker thread inside
    ``_prepare_local_conditioning``, so this test used to assert that two futures were
    submitted and cancelled when the encoder failed (it also monkeypatched
    ``module.ThreadPoolExecutor``, which no longer exists here).

    The producer is off because the rows it drew came from ``self._initial_noise`` while
    ``diffuse``'s non-guided fallback draws from ``MiniMaxH3RequestNoisePlan``, and
    `diffuse` tests the precomputed branch first -- so restoring it would silently change
    the noise of every non-guided Ref2VA request (manual §5.3).

    What must still hold: the encoder failure propagates, no worker is left behind, and
    nothing is written into the request context. Restoring the producer turns this red.
    """

    def encode_prompt(prepared):
        if failure_stage == "text":
            raise RuntimeError("text encoder failed")
        return torch.zeros(2, 4), torch.ones(2, dtype=torch.long)

    def fail_media(media):
        raise RuntimeError("media encoder failed")

    pipeline._fasth3 = None
    pipeline.supported_tasks = ("ref2va",)
    pipeline.od_config = OmniDiffusionConfig()
    monkeypatch.setattr(pipeline, "encode_prompt", encode_prompt)
    monkeypatch.setattr(pipeline, "_encode_local_media", fail_media)
    sampling = OmniDiffusionSamplingParams(height=64, width=64, seed=42, extra_args={"task": "ref2va", "duration": 4.4})
    prompt = {"prompt": "reference", "multi_modal_data": {"image": Image.new("RGB", (256, 256))}}
    context: dict = {}

    with pytest.raises(RuntimeError, match=f"{failure_stage} encoder failed"):
        pipeline._prepare_local_conditioning(prompt, sampling, request_context=context)

    assert "precomputed_initial_noise" not in context
    assert "precomputed_visual_condition_noise" not in context
