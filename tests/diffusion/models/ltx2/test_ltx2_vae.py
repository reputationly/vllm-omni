# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for LTX video VAE tiling and distributed decode behavior."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ltx_base_vocoder_keeps_native_dtype(monkeypatch):
    import vllm_omni.diffusion.models.ltx2.ltx2_runtime as ltx_runtime

    class FakeBaseVocoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_dtype = None

        def forward(self, value):
            self.input_dtype = value.dtype
            return value

    monkeypatch.setattr(
        ltx_runtime.torch,
        "autocast",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("base vocoder must not use autocast")),
    )
    vocoder = FakeBaseVocoder()

    output = ltx_runtime._run_ltx_vocoder(vocoder, torch.ones(1, dtype=torch.bfloat16))

    assert vocoder.input_dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16


class TestLTXDiffusionDecoder:
    @pytest.mark.parametrize("mode", ["spatial_shard_height", "spatial_shard_width"])
    def test_distributed_diffusion_decoder_rejects_non_tile_parallel_modes(self, mode):
        from vllm_omni.diffusion.models.ltx2.ltx2_diffusion_decoder_distributed import (
            DistributedLTX2VideoDiffusionDecoderModel,
        )

        model = object.__new__(DistributedLTX2VideoDiffusionDecoderModel)
        torch.nn.Module.__init__(model)
        model.distributed_executor = SimpleNamespace(set_parallel_size=lambda *_args, **_kwargs: None)

        with pytest.raises(ValueError, match="only supports vae_parallel_mode='tile'"):
            model.set_parallel_size(2, mode=mode)

    def test_distributed_diffusion_decoder_accepts_tile_parallel_mode(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_diffusion_decoder_distributed import (
            DistributedLTX2VideoDiffusionDecoderModel,
        )

        calls = []
        model = object.__new__(DistributedLTX2VideoDiffusionDecoderModel)
        torch.nn.Module.__init__(model)
        model.distributed_executor = SimpleNamespace(
            set_parallel_size=lambda parallel_size, mode: calls.append((parallel_size, mode))
        )

        model.set_parallel_size(4, mode="tile")

        assert calls == [(4, "tile")]

    def test_short_clip_keeps_stage5_temporal_context_then_crops_output(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_diffusion_decoder import (
            LTX2VideoDiffusionDecoder3d,
            LTX2VideoDiffusionDecoderModel,
        )

        class IdentityAttention:
            def build_block_mask(self, _hidden_states):
                return None

        class IdentityBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = IdentityAttention()

            def forward(self, hidden_states, _block_mask):
                return hidden_states

        class IdentityUpsample(torch.nn.Module):
            def forward(self, hidden_states, *, drop_leading_frame):
                assert drop_leading_frame
                return hidden_states

        decoder = object.__new__(LTX2VideoDiffusionDecoder3d)
        torch.nn.Module.__init__(decoder)
        decoder.det_stages = torch.nn.ModuleList([torch.nn.ModuleList([IdentityBlock()])])
        decoder.upsamples = torch.nn.ModuleList([IdentityUpsample()])
        decoder.trailing_pad_latent_frames = 2
        decoder.temporal_compression_ratio = 8
        decoder.stage5_kernel = (11, 3, 3)

        # 17 frames contain one real frame followed by 16 propagated ghost
        # frames. The reference keeps 11 for stage-5 NATTEN instead of
        # cropping the tensor immediately back to one.
        context = decoder.forward_stage_4(torch.zeros(1, 17, 3, 3, 4))
        assert context.shape == (1, 11, 3, 3, 4)

        class FakeDecoder(torch.nn.Module):
            def forward(self, z, *, generator, num_inference_steps):
                assert generator is None
                assert num_inference_steps is None
                return torch.zeros(z.shape[0], 3, 11, z.shape[3] * 2, z.shape[4] * 2)

        model = object.__new__(LTX2VideoDiffusionDecoderModel)
        torch.nn.Module.__init__(model)
        model.decoder = FakeDecoder()
        model.use_tiling = False
        model.tile_sample_min_num_frames = 80
        model.tile_sample_min_height = 768
        model.tile_sample_min_width = 768
        model.temporal_compression_ratio = 8
        model.spatial_compression_ratio = 2

        output = model.decode(torch.zeros(1, 4, 1, 2, 3), generator=None, return_dict=False)[0]
        assert output.shape == (1, 3, 1, 4, 6)

    @pytest.mark.parametrize(
        ("model_version", "extras", "expected"),
        [
            ("2.5", {}, True),
            ("2.5", {"ltx2_use_conv_vae": True}, False),
            ("2.3", {}, False),
        ],
    )
    def test_ltx25_diffusion_decoder_is_default_with_conv_vae_opt_in(self, model_version, extras, expected):
        from vllm_omni.diffusion.models.ltx2.ltx2_components import _ltx2_use_diffusion_decoder

        assert _ltx2_use_diffusion_decoder(SimpleNamespace(extras=extras), model_version) is expected

    def test_conv_vae_opt_in_rejects_non_boolean_model_extra(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_components import _ltx2_use_diffusion_decoder

        with pytest.raises(TypeError, match="ltx2_use_conv_vae"):
            _ltx2_use_diffusion_decoder(SimpleNamespace(extras={"ltx2_use_conv_vae": "true"}), "2.5")

    def test_native_diffusion_decoder_conversion_splits_qkv_and_folds_gates(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_diffusion_decoder import (
            convert_ltx25_native_diffusion_decoder_state_dict,
        )

        qkv_weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        qkv_bias = torch.arange(6, dtype=torch.float32)
        projection_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        projection_bias = torch.ones(2)
        gate = torch.tensor([2.0, 3.0])
        mean = torch.tensor([0.1, 0.2])
        std = torch.tensor([0.3, 0.4])
        native = {
            "encoder.ignored": torch.ones(1),
            "per_channel_statistics.mean-of-means": mean,
            "vae.per_channel_statistics.std-of-means": std,
            "decoder.type_emb": torch.ones(2),
            "decoder.coarse_head.weight": torch.ones(1),
            "decoder.det_stages.0.0.attn.qkv.weight": qkv_weight,
            "decoder.det_stages.0.0.attn.qkv.bias": qkv_bias,
            "decoder.det_stages.0.0.attn.q_norm.weight": torch.tensor([5.0, 6.0]),
            "decoder.det_stages.0.0.attn.k_norm.weight": torch.tensor([7.0, 8.0]),
            "decoder.diff_blocks.0.attn.proj.weight": projection_weight,
            "decoder.diff_blocks.0.attn.proj.bias": projection_bias,
            "decoder.diff_blocks.0.gate_msa": gate,
            "vae.decoder.t_embedder.mlp.0.weight": torch.ones(2, 2),
        }

        converted = convert_ltx25_native_diffusion_decoder_state_dict(native)

        torch.testing.assert_close(converted["latents_mean"], mean)
        torch.testing.assert_close(converted["latents_std"], std)
        torch.testing.assert_close(converted["decoder.det_stages.0.0.attn.to_q.weight"], qkv_weight[:2])
        torch.testing.assert_close(converted["decoder.det_stages.0.0.attn.to_k.weight"], qkv_weight[2:4])
        torch.testing.assert_close(converted["decoder.det_stages.0.0.attn.to_v.weight"], qkv_weight[4:])
        torch.testing.assert_close(converted["decoder.det_stages.0.0.attn.to_q.bias"], qkv_bias[:2])
        torch.testing.assert_close(converted["decoder.det_stages.0.0.attn.to_k.bias"], qkv_bias[2:4])
        torch.testing.assert_close(converted["decoder.det_stages.0.0.attn.to_v.bias"], qkv_bias[4:])
        torch.testing.assert_close(converted["decoder.det_stages.0.0.attn.norm_q.weight"], torch.tensor([5.0, 6.0]))
        torch.testing.assert_close(converted["decoder.det_stages.0.0.attn.norm_k.weight"], torch.tensor([7.0, 8.0]))
        torch.testing.assert_close(
            converted["decoder.diff_blocks.0.attn.to_out.0.weight"], gate.unsqueeze(1) * projection_weight
        )
        torch.testing.assert_close(converted["decoder.diff_blocks.0.attn.to_out.0.bias"], gate * projection_bias)
        assert "decoder.t_embedder.timestep_embedder.linear_1.weight" in converted
        assert not any("type_emb" in key or "coarse" in key or "gate_msa" in key for key in converted)

    def test_native_diffusion_decoder_conversion_rejects_invalid_fused_qkv(self):
        from vllm_omni.diffusion.models.ltx2.ltx2_diffusion_decoder import (
            convert_ltx25_native_diffusion_decoder_state_dict,
        )

        with pytest.raises(ValueError, match="not divisible by 3"):
            convert_ltx25_native_diffusion_decoder_state_dict(
                {"decoder.det_stages.0.0.attn.qkv.weight": torch.ones(5, 2)}
            )

    def test_decode_uses_diffusion_decoder_without_conv_vae_conditioning(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        seen = {}

        class ConvVae:
            dtype = torch.float32
            config = SimpleNamespace(timestep_conditioning=True)

            def decode(self, *_args, **_kwargs):
                raise AssertionError("the convolutional VAE decoder must not run")

        class DiffusionDecoder:
            dtype = torch.float32

            def decode(self, latents, *, generator, return_dict):
                seen["latents"] = latents
                seen["generator"] = generator
                seen["return_dict"] = return_dict
                return (latents + 1,)

        class AudioVae:
            dtype = torch.float32

            def decode(self, audio_latents, *, return_dict):
                assert return_dict is False
                return (audio_latents + 2,)

        class VideoProcessor:
            def postprocess_video(self, video, *, output_type):
                seen["output_type"] = output_type
                return video

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.use_diffusion_decoder = True
        pipe.vae = ConvVae()
        pipe.diffusion_decoder = DiffusionDecoder()
        pipe.audio_vae = AudioVae()
        pipe.vocoder = torch.nn.Identity()
        pipe.video_processor = VideoProcessor()

        generator = torch.Generator().manual_seed(123)
        latents = torch.ones(1, 1)
        output = pipe._decode_output(
            latents=latents,
            audio_latents=torch.ones(1, 1),
            output_type="pt",
            connector_prompt_embeds=torch.ones(1, 1),
            generator=generator,
            device=torch.device("cpu"),
            decode_timestep=0.7,
            decode_noise_scale=0.5,
            prompt_batch_size=1,
        )

        torch.testing.assert_close(seen["latents"], latents)
        assert seen["generator"] is generator
        assert seen["return_dict"] is False
        assert seen["output_type"] == "pt"
        torch.testing.assert_close(output.output[0], latents + 1)
        torch.testing.assert_close(output.output[1], torch.full((1, 1), 3.0))

    def test_diffusion_decode_denormalizes_with_decoder_statistics(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.use_diffusion_decoder = True
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1
        pipe.vae = SimpleNamespace(
            latents_mean=torch.full((2,), 100.0),
            latents_std=torch.full((2,), 10.0),
            config=SimpleNamespace(scaling_factor=1.0),
        )
        pipe.diffusion_decoder = SimpleNamespace(
            latents_mean=torch.tensor([10.0, 20.0]),
            latents_std=torch.tensor([2.0, 4.0]),
            config=SimpleNamespace(scaling_factor=2.0),
        )
        pipe.audio_vae = SimpleNamespace(
            latents_mean=torch.tensor(0.0),
            latents_std=torch.tensor(1.0),
        )
        forward_ctx = SimpleNamespace(
            latent_num_frames=1,
            latent_height=1,
            latent_width=1,
            original_audio_num_frames=1,
            latent_mel_bins=2,
        )

        video, audio = pipe._unpack_and_denormalize_stage(
            forward_ctx,
            torch.tensor([[[1.0, 2.0]]]),
            torch.zeros(1, 1, 2),
        )

        torch.testing.assert_close(video.flatten(), torch.tensor([11.0, 24.0]))
        torch.testing.assert_close(audio, torch.zeros(1, 1, 1, 2))

    def test_non_output_rank_does_not_enter_diffusion_decoder(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.ltx2_runtime as ltx_runtime
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        class ConvVae:
            config = SimpleNamespace(timestep_conditioning=False)

            def is_distributed_enabled(self):
                raise AssertionError("DiffVAE decode must not inspect ConvVAE collectives")

        class DiffusionDecoder:
            dtype = torch.float32

            def decode(self, *_args, **_kwargs):
                raise AssertionError("only the output rank may run DiffVAE decode")

        monkeypatch.setattr(ltx_runtime.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(ltx_runtime.torch.distributed, "get_rank", lambda: 1)
        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.use_diffusion_decoder = True
        pipe.vae = ConvVae()
        pipe.diffusion_decoder = DiffusionDecoder()

        output = pipe._decode_output(
            latents=torch.ones(1, 1),
            audio_latents=torch.ones(1, 1),
            output_type="pt",
            connector_prompt_embeds=torch.ones(1, 1),
            generator=None,
            device=torch.device("cpu"),
            decode_timestep=0.0,
            decode_noise_scale=None,
            prompt_batch_size=1,
        )

        assert output.output[0].numel() == 0
        assert output.output[1].numel() == 0

    def test_non_output_rank_enters_distributed_diffusion_decoder(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.ltx2_runtime as ltx_runtime
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        class ConvVae:
            config = SimpleNamespace(timestep_conditioning=False)

            def is_distributed_enabled(self):
                raise AssertionError("DiffVAE decode must query the active decoder")

        class DiffusionDecoder:
            dtype = torch.float32

            def __init__(self):
                self.decode_calls = 0

            def is_distributed_enabled(self):
                return True

            def decode(self, *_args, **_kwargs):
                self.decode_calls += 1
                return (torch.empty(0, 3, 0, 0, 0),)

        monkeypatch.setattr(ltx_runtime.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(ltx_runtime.torch.distributed, "get_rank", lambda: 1)
        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.use_diffusion_decoder = True
        pipe.vae = ConvVae()
        pipe.diffusion_decoder = DiffusionDecoder()

        output = pipe._decode_output(
            latents=torch.ones(1, 1),
            audio_latents=torch.ones(1, 1),
            output_type="pt",
            connector_prompt_embeds=torch.ones(1, 1),
            generator=None,
            device=torch.device("cpu"),
            decode_timestep=0.0,
            decode_noise_scale=None,
            prompt_batch_size=1,
        )

        assert pipe.diffusion_decoder.decode_calls == 1
        assert output.output[0].numel() == 0
        assert output.output[1].numel() == 0

    def test_diffusion_decoder_patch_parallel_size_one_uses_native_tiling(self, monkeypatch):
        from vllm_omni.diffusion.models.ltx2.ltx2_diffusion_decoder import LTX2VideoDiffusionDecoderModel
        from vllm_omni.diffusion.models.ltx2.ltx2_diffusion_decoder_distributed import (
            DistributedLTX2VideoDiffusionDecoderModel,
        )

        expected = torch.ones(1, 3, 1, 2, 2)
        seen = {}

        def native_tiled_decode(self, z, generator=None, num_inference_steps=None):
            seen["z"] = z
            seen["generator"] = generator
            seen["num_inference_steps"] = num_inference_steps
            return expected

        monkeypatch.setattr(LTX2VideoDiffusionDecoderModel, "tiled_decode", native_tiled_decode)
        model = object.__new__(DistributedLTX2VideoDiffusionDecoderModel)
        torch.nn.Module.__init__(model)
        model.is_distributed_enabled = lambda: False
        generator = torch.Generator().manual_seed(17)
        z = torch.zeros(1, 1, 1, 1, 1)

        result = DistributedLTX2VideoDiffusionDecoderModel.tiled_decode(
            model,
            z,
            generator=generator,
            num_inference_steps=1,
        )

        assert result is expected
        assert seen["z"] is z
        assert seen["generator"] is generator
        assert seen["num_inference_steps"] == 1

    def test_distributed_diffusion_tiles_preserve_serial_noise_order(self, monkeypatch):
        from diffusers.utils.torch_utils import randn_tensor

        from vllm_omni.diffusion.models.ltx2.ltx2_diffusion_decoder_distributed import (
            DistributedLTX2VideoDiffusionDecoderModel,
        )

        model = DistributedLTX2VideoDiffusionDecoderModel(
            out_channels=1,
            latent_channels=1,
            patch_size=1,
            decoder_head_dim=8,
            decoder_stage_channels=(8, 8, 8, 8, 8),
            decoder_stage_depths=(1, 1, 1, 1, 1),
            decoder_stage_kernels=((1, 1, 1),) * 4,
            decoder_upsample_strides=((1, 1, 1),) * 4,
            decoder_upsample_channel_reductions=(1, 1, 1, 1),
            decoder_stage5_kernel=(1, 1, 1),
            spatial_compression_ratio=1,
            temporal_compression_ratio=1,
        )
        model.tile_sample_min_num_frames = 2
        model.tile_sample_stride_num_frames = 1
        model.tile_sample_min_height = 2
        model.tile_sample_stride_height = 1
        model.tile_sample_min_width = 2
        model.tile_sample_stride_width = 1
        model.distributed_executor = SimpleNamespace(group=None)
        model.decoder.forward_stages_1_to_3 = lambda z: z.permute(0, 2, 3, 4, 1)
        monkeypatch.setattr(torch.distributed, "broadcast", lambda *_args, **_kwargs: None)

        z = torch.zeros(1, 1, 2, 3, 3)
        distributed_generator = torch.Generator().manual_seed(123)
        reference_generator = torch.Generator().manual_seed(123)
        tasks, grid_spec = model._distributed_tile_split(z, distributed_generator, num_inference_steps=1)

        assert grid_spec.grid_shape == (1, 3, 3)
        assert grid_spec.split_dims == (2, 3, 4)
        for task in tasks:
            tile_shape = model._tiled_pixel_shape_from_features(
                task.tensor,
                drop_leading_frame=task.drop_leading_frame,
                crop_trailing_ghost=task.crop_trailing_ghost,
            )
            expected = randn_tensor(tile_shape, generator=reference_generator, device=z.device, dtype=z.dtype)
            actual = randn_tensor(tile_shape, generator=task.noise_generator, device=z.device, dtype=z.dtype)
            torch.testing.assert_close(actual, expected)

        assert torch.equal(distributed_generator.get_state(), reference_generator.get_state())


class TestLTXOutputRank:
    @pytest.mark.parametrize(
        ("distributed_vae_state", "expected_decode_calls"),
        [(False, 0), (True, 1), (RuntimeError("unavailable"), None)],
    )
    def test_non_output_rank_only_enters_collective_vae_decode(
        self,
        monkeypatch,
        distributed_vae_state,
        expected_decode_calls,
    ):
        import vllm_omni.diffusion.models.ltx2.ltx2_runtime as ltx_runtime
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

        class FakeVae:
            dtype = torch.float32
            config = SimpleNamespace(timestep_conditioning=False)

            def __init__(self):
                self.decode_calls = 0

            def is_distributed_enabled(self):
                if isinstance(distributed_vae_state, Exception):
                    raise distributed_vae_state
                return distributed_vae_state

            def decode(self, *_args, **_kwargs):
                self.decode_calls += 1
                return (torch.ones(1, 1),)

        monkeypatch.setattr(ltx_runtime.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(ltx_runtime.torch.distributed, "get_rank", lambda: 1)
        pipe = object.__new__(LTX2Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.vae = FakeVae()
        # _decode_output 会先走一遍 _offload_dit_before_vae(),它读 od_config。
        # 真实 pipeline 在 __init__ 里必然有这个字段,这里是 object.__new__ 出来的
        # 半成品,得补上;enable_cpu_offload=False 让卸载分支立即返回,不干扰本用例。
        pipe.od_config = SimpleNamespace(enable_cpu_offload=False)

        decode_kwargs = {
            "latents": torch.ones(1, 1),
            "audio_latents": torch.ones(1, 1),
            "output_type": "np",
            "connector_prompt_embeds": torch.ones(1, 1),
            "generator": None,
            "device": torch.device("cpu"),
            "decode_timestep": 0.0,
            "decode_noise_scale": None,
            "prompt_batch_size": 1,
        }
        if isinstance(distributed_vae_state, Exception):
            with pytest.raises(RuntimeError, match="unavailable"):
                pipe._decode_output(**decode_kwargs)
            assert pipe.vae.decode_calls == 0
            return

        output = pipe._decode_output(**decode_kwargs)

        assert pipe.vae.decode_calls == expected_decode_calls
        assert output.output[0].numel() == 0
        assert output.output[1].numel() == 0


class TestLTX23VaeDistributedDecode:
    """Test LTX-2.3 distributed VAE helpers without loading weights."""

    def test_ltx23_video_vae_is_distributed_tile_only_class(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
            DistributedAutoencoderKLLTX2Video,
        )
        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin

        assert issubclass(DistributedAutoencoderKLLTX2Video, DistributedVaeMixin)
        assert not hasattr(DistributedAutoencoderKLLTX2Video, "patch_split")

    def test_ltx23_vae_executor_gathers_known_tile_shapes_and_returns_empty_on_non_rank0(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import LTX2VaeExecutor
        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
            DistributedOperator,
            GridSpec,
            TileTask,
        )

        z = torch.zeros(1, 1, 1, 1, 1)
        tile_output_shapes = {
            0: (1, 1, 1, 2, 2),
            1: (1, 1, 1, 2, 1),
            2: (1, 1, 1, 1, 2),
            3: (1, 1, 1, 1, 1),
        }
        tasks = [
            TileTask(0, (0, 0), z, workload=4),
            TileTask(1, (0, 1), z, workload=2),
            TileTask(2, (1, 0), z, workload=2),
            TileTask(3, (1, 1), z, workload=1),
        ]
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(2, 2),
            tile_spec={
                "max_tile_output_shape": (1, 1, 1, 2, 2),
                "tile_output_shapes": tile_output_shapes,
            },
            output_dtype=torch.float32,
        )
        # 这个 dict 同时存 tuple 和 dict 两种值,不标注的话 mypy 会按首次赋值把它
        # 收窄成 dict[Any, tuple[...]],后面存 merged_shapes 就报类型不兼容。
        seen: dict[str, Any] = {}

        def exec_tile(task):
            return torch.full(tile_output_shapes[task.tile_id], float(task.tile_id + 1))

        def merge_tiles(coord_tensor_map, passed_grid_spec):
            seen["merged_shapes"] = {coord: tuple(tile.shape) for coord, tile in coord_tensor_map.items()}
            assert passed_grid_spec is grid_spec
            return torch.stack(
                [
                    coord_tensor_map[(0, 0)].flatten()[0],
                    coord_tensor_map[(0, 1)].flatten()[0],
                    coord_tensor_map[(1, 0)].flatten()[0],
                    coord_tensor_map[(1, 1)].flatten()[0],
                ]
            )

        operator = DistributedOperator(split=lambda _z: (tasks, grid_spec), exec=exec_tile, merge=merge_tiles)

        rank0_executor = object.__new__(LTX2VaeExecutor)
        rank0_executor.parallel_size = 2
        rank0_executor.world_size = 2
        rank0_executor.rank = 0

        def gather_rank0(local_tile_tensor):
            assigned = rank0_executor._balance_tasks(tasks, 2)
            rank1_results = [(task.tile_id, exec_tile(task)) for task in assigned[1]]
            rank1_tile_tensor = rank0_executor._pack_local_tiles_without_meta(
                rank1_results,
                list(local_tile_tensor.shape),
                z.device,
                torch.float32,
            )
            seen["rank0_gather_shape"] = tuple(local_tile_tensor.shape)
            return [local_tile_tensor, rank1_tile_tensor]

        def fail_final_sync(*_args, **_kwargs):
            raise AssertionError("broadcast_result=False should not sync the final result")

        rank0_executor.gather_tensors = gather_rank0
        rank0_executor._sync_final_result = fail_final_sync

        rank0_result = rank0_executor.execute(z, operator, broadcast_result=False)

        torch.testing.assert_close(rank0_result, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert seen["rank0_gather_shape"] == (2, 1, 1, 1, 2, 2)
        assert seen["merged_shapes"] == {
            (0, 0): (1, 1, 1, 2, 2),
            (0, 1): (1, 1, 1, 2, 1),
            (1, 0): (1, 1, 1, 1, 2),
            (1, 1): (1, 1, 1, 1, 1),
        }

        non_rank0_executor = object.__new__(LTX2VaeExecutor)
        non_rank0_executor.parallel_size = 2
        non_rank0_executor.world_size = 2
        non_rank0_executor.rank = 1

        def gather_rank1(local_tile_tensor):
            seen["rank1_gather_shape"] = tuple(local_tile_tensor.shape)
            return None

        def fail_non_rank0_merge(*_args, **_kwargs):
            raise AssertionError("non-rank0 should not merge gathered tiles")

        non_rank0_executor.gather_tensors = gather_rank1
        non_rank0_executor._sync_final_result = fail_final_sync

        empty_result = non_rank0_executor.execute(
            z,
            DistributedOperator(
                split=lambda _z: (tasks, grid_spec),
                exec=exec_tile,
                merge=fail_non_rank0_merge,
            ),
            broadcast_result=False,
        )

        assert tuple(empty_result.shape) == (0,)
        assert seen["rank1_gather_shape"] == (2, 1, 1, 1, 2, 2)


class TestLTX23VaeTiling:
    """Test LTX-2.3 video VAE tile helpers without loading weights."""

    def test_ltx23_video_vae_tile_split_uses_native_ltx23_tile_geometry(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
            DistributedAutoencoderKLLTX2Video,
        )

        vae = SimpleNamespace(
            spatial_compression_ratio=32,
            tile_sample_min_height=512,
            tile_sample_min_width=512,
            tile_sample_stride_height=448,
            tile_sample_stride_width=448,
            temporal_compression_ratio=8,
            dtype=torch.float32,
        )

        z = torch.zeros(1, 2, 5, 16, 24)
        tasks, grid_spec = DistributedAutoencoderKLLTX2Video.tile_split(vae, z)

        assert grid_spec.grid_shape == (2, 2)
        assert grid_spec.split_dims == (3, 4)
        assert grid_spec.tile_spec["sample_height"] == 512
        assert grid_spec.tile_spec["sample_width"] == 768
        assert grid_spec.tile_spec["blend_height"] == 64
        assert grid_spec.tile_spec["blend_width"] == 64
        assert grid_spec.tile_spec["max_tile_output_shape"] == (1, 3, 33, 512, 512)
        assert grid_spec.tile_spec["tile_output_shapes"] == {
            0: (1, 3, 33, 512, 512),
            1: (1, 3, 33, 512, 320),
            2: (1, 3, 33, 64, 512),
            3: (1, 3, 33, 64, 320),
        }
        assert [task.grid_coord for task in tasks] == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert [tuple(task.tensor.shape) for task in tasks] == [
            (1, 2, 5, 16, 16),
            (1, 2, 5, 16, 10),
            (1, 2, 5, 2, 16),
            (1, 2, 5, 2, 10),
        ]
        assert [task.workload for task in tasks] == [5 * 16 * 16, 5 * 16 * 10, 5 * 2 * 16, 5 * 2 * 10]

    def test_ltx23_video_vae_tile_merge_blends_and_crops_like_tiled_decode(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
            DistributedAutoencoderKLLTX2Video,
        )
        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import GridSpec

        class FakeVae:
            def __init__(self):
                self.blend_calls = []

            def clear_cache(self):
                pass

            def blend_v(self, _previous, current, blend_height):
                self.blend_calls.append(("v", blend_height))
                return current

            def blend_h(self, _previous, current, blend_width):
                self.blend_calls.append(("h", blend_width))
                return current

        fake_vae = FakeVae()
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(2, 2),
            tile_spec={
                "sample_height": 10,
                "sample_width": 10,
                "blend_height": 1,
                "blend_width": 2,
                "tile_sample_stride_height": 5,
                "tile_sample_stride_width": 5,
            },
        )
        tiles = {
            (0, 0): torch.full((1, 3, 2, 6, 6), 1.0),
            (0, 1): torch.full((1, 3, 2, 6, 6), 2.0),
            (1, 0): torch.full((1, 3, 2, 6, 6), 3.0),
            (1, 1): torch.full((1, 3, 2, 6, 6), 4.0),
        }

        merged = DistributedAutoencoderKLLTX2Video.tile_merge(fake_vae, tiles, grid_spec)

        assert merged.shape == (1, 3, 2, 10, 10)
        assert fake_vae.blend_calls == [("h", 2), ("v", 1), ("v", 1), ("h", 2)]
        torch.testing.assert_close(merged[:, :, :, :5, :5], torch.ones(1, 3, 2, 5, 5))
        torch.testing.assert_close(merged[:, :, :, :5, 5:], torch.full((1, 3, 2, 5, 5), 2.0))
        torch.testing.assert_close(merged[:, :, :, 5:, :5], torch.full((1, 3, 2, 5, 5), 3.0))
        torch.testing.assert_close(merged[:, :, :, 5:, 5:], torch.full((1, 3, 2, 5, 5), 4.0))

    def test_ltx23_video_vae_tiled_decode_dispatches_to_tile_operator(self):
        from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_ltx2

        z = torch.zeros(1, 2, 1, 16, 24)
        expected = torch.ones(1, 3, 1, 512, 768)
        seen = {}

        class FakeExecutor:
            def execute(self, tensor, operator, broadcast_result=True):
                seen["tensor"] = tensor
                seen["operator"] = operator
                seen["broadcast_result"] = broadcast_result
                return expected

        vae = SimpleNamespace(distributed_executor=FakeExecutor(), is_distributed_enabled=lambda: True)
        vae.tile_split = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tile_split.__get__(vae)
        vae.tile_exec = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tile_exec.__get__(vae)
        vae.tile_merge = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tile_merge.__get__(vae)

        output = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tiled_decode(
            vae,
            z,
            temb=torch.tensor(0.5),
            return_dict=False,
        )

        assert len(output) == 1
        assert output[0] is expected
        assert seen["tensor"] is z
        assert seen["broadcast_result"] is False
        assert seen["operator"].split.__name__ == "tile_split"
        assert seen["operator"].merge.__name__ == "tile_merge"


class TestLTXPreVaeDitOffload:
    """``VLLM_OMNI_LTX2_OFFLOAD_DIT_BEFORE_VAE`` 的契约。

    这个开关的失败方式**全是静默的**:env 名在某次统一命名里被改掉、调用点被挪到
    ``output_type == "latent"`` 的早退分支之后、``_dit_modules`` 换了取值方式 ——
    三种改动都不报错、也不会让任何别的测试变红,唯一的症状是"显存没降下来",
    而那要起一个 4 卡实例跑一发 20 秒 1080p 盯 nvidia-smi 才看得见
    (实测:开了之后峰值 37.0 → 27.3~30.2 GB,引擎日志 ``freed 19.05 GiB``)。

    所以这里锁的是**接线还在不在**,不是"省了多少显存"——后者只能靠实测。
    与 H3 的 ``test_pre_vae_offload_releases_all_resident_dits`` 同源。
    """

    @staticmethod
    def _pipeline(monkeypatch, *, enable_cpu_offload=True):
        import vllm_omni.diffusion.models.ltx2.ltx2_runtime as ltx_runtime
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2DistilledOneStagePipeline
        from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook

        pipe = object.__new__(LTX2DistilledOneStagePipeline)
        torch.nn.Module.__init__(pipe)
        pipe.od_config = SimpleNamespace(enable_cpu_offload=enable_cpu_offload, pin_cpu_memory=False)
        pipe._dit_modules = ["transformer"]
        pipe.transformer = torch.nn.Linear(2, 2, bias=False)

        moved: list[torch.nn.Module] = []
        # device_type 设成 cpu,参数就算"在设备上",能走到真正的搬运分支;
        # 其余三个平台调用在 CPU 上没有意义,换成空实现。
        monkeypatch.setattr(ltx_runtime.current_omni_platform, "device_type", "cpu")
        monkeypatch.setattr(ltx_runtime.current_omni_platform, "synchronize", lambda: None)
        monkeypatch.setattr(ltx_runtime.current_omni_platform, "empty_cache", lambda: None)
        monkeypatch.setattr(ltx_runtime.current_omni_platform, "get_free_memory", lambda: 1024**3)
        monkeypatch.setattr(
            SequentialOffloadHook,
            "_move_params",
            lambda module, *_args, **_kwargs: moved.append(module),
        )
        return pipe, moved

    def test_offloads_dit_when_switch_on(self, monkeypatch):
        monkeypatch.setenv("VLLM_OMNI_LTX2_OFFLOAD_DIT_BEFORE_VAE", "1")
        pipe, moved = self._pipeline(monkeypatch)

        pipe._offload_dit_before_vae()

        assert moved == [pipe.transformer]

    def test_inert_when_env_unset(self, monkeypatch):
        """默认关。这条锁的是"补丁装上了也不改变现有部署的行为"这句承诺本身。"""
        monkeypatch.delenv("VLLM_OMNI_LTX2_OFFLOAD_DIT_BEFORE_VAE", raising=False)
        pipe, moved = self._pipeline(monkeypatch)

        pipe._offload_dit_before_vae()

        assert moved == []

    def test_inert_without_cpu_offload(self, monkeypatch):
        """没开 model-level offload 的部署不该被打扰:DiT 本来就该常驻,搬走它只是白费。"""
        monkeypatch.setenv("VLLM_OMNI_LTX2_OFFLOAD_DIT_BEFORE_VAE", "1")
        pipe, moved = self._pipeline(monkeypatch, enable_cpu_offload=False)

        pipe._offload_dit_before_vae()

        assert moved == []

    def test_runs_before_vae_decode(self, monkeypatch):
        """锁**调用点**:必须在 VAE 解码之前跑。

        解码那一段才是峰值所在(tile all-gather 的缓冲每张卡一份、rank 0 还要合并整段
        视频再转 fp32),挪到解码之后就等于这个开关白开 —— 而且不会有任何报错。
        """
        import vllm_omni.diffusion.models.ltx2.ltx2_runtime as ltx_runtime
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2DistilledOneStagePipeline

        order: list[str] = []

        class FakeVae:
            dtype = torch.float32
            config = SimpleNamespace(timestep_conditioning=False)

            def is_distributed_enabled(self):
                return True

            def decode(self, *_args, **_kwargs):
                order.append("decode")
                return (torch.ones(1, 1),)

        # 非输出 rank:解码后即早退,不必再搭 audio_vae
        monkeypatch.setattr(ltx_runtime.torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(ltx_runtime.torch.distributed, "get_rank", lambda: 1)

        pipe = object.__new__(LTX2DistilledOneStagePipeline)
        torch.nn.Module.__init__(pipe)
        pipe.vae = FakeVae()
        object.__setattr__(pipe, "_offload_dit_before_vae", lambda: order.append("offload"))

        pipe._decode_output(
            latents=torch.ones(1, 1),
            audio_latents=torch.ones(1, 1),
            output_type="np",
            connector_prompt_embeds=torch.ones(1, 1),
            generator=None,
            device=torch.device("cpu"),
            decode_timestep=0.0,
            decode_noise_scale=None,
            prompt_batch_size=1,
        )

        assert order == ["offload", "decode"]


class TestLTXPostprocessChunking:
    """分块 postprocess 与整段 postprocess 必须**逐位相等**。

    分块是为了砍掉解码阶段最大的那一块:diffusers 的 ``postprocess_video`` 对整段视频
    一次性 ``.float()``,4K×121 帧那份 fp32 就是 12.1 GB,与 all-gather 缓冲、merge
    结果同时存在(2026-08-31 实测峰值 31.6 GB / 40 GB)。

    为什么必须用单测而不是比对成片:LTX 的端到端输出**跨实例不可复现** —— 同一份代码在
    两个容器里跑出的像素 md5 不同(inductor 的 autotune 按实测耗时挑 kernel),
    但同一实例内连发两次完全一致。所以"改前改后产物一样"这种验证方法在这里无效,
    只能把等价性钉在这一层:postprocess 是逐帧的反归一化 + clamp + 维度变换,
    帧间无耦合,沿帧轴切开必然等价。
    """

    @staticmethod
    def _pipeline():
        from diffusers.video_processor import VideoProcessor

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2DistilledOneStagePipeline

        pipe = object.__new__(LTX2DistilledOneStagePipeline)
        torch.nn.Module.__init__(pipe)
        pipe.video_processor = VideoProcessor(vae_scale_factor=32)
        return pipe

    @pytest.mark.parametrize(
        ("num_frames", "chunk", "output_type"),
        [
            (121, 32, "np"),  # 生产菜单的 5 秒档
            (121, 32, "pt"),
            (64, 16, "np"),  # 整除
            (65, 16, "pt"),  # 有余数块
            (7, 32, "np"),  # 不足一块,走原路径
        ],
    )
    def test_chunked_matches_whole(self, monkeypatch, num_frames, chunk, output_type):
        import vllm_omni.diffusion.models.ltx2.ltx2_runtime as ltx_runtime

        monkeypatch.setenv(ltx_runtime.LTX2_POSTPROCESS_FRAME_CHUNK_ENV, str(chunk))
        pipe = self._pipeline()
        torch.manual_seed(0)
        video = torch.randn(1, 3, num_frames, 16, 24)

        whole = pipe.video_processor.postprocess_video(video.clone(), output_type=output_type)
        chunked = pipe._postprocess_video_chunked(video.clone(), output_type)

        if output_type == "pt":
            assert torch.equal(whole, chunked)
        else:
            assert whole.shape == chunked.shape
            assert np.array_equal(whole, chunked)

    def test_chunk_env_rejects_garbage(self, monkeypatch):
        """粒度配错要当场报错,不能静默回落 —— 静默回落等于这条优化悄悄失效。"""
        import vllm_omni.diffusion.models.ltx2.ltx2_runtime as ltx_runtime

        for bad in ("0", "-4", "abc"):
            monkeypatch.setenv(ltx_runtime.LTX2_POSTPROCESS_FRAME_CHUNK_ENV, bad)
            with pytest.raises(ValueError):
                ltx_runtime._postprocess_frame_chunk()
