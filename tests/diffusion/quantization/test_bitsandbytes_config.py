# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BitsAndBytes quantization config."""

import importlib.util
import sys
import types

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Module, Parameter
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.factory import SUPPORTED_QUANTIZATION_METHODS

# Everything outside ``TestCudaBnBSmoke`` mocks bitsandbytes out and runs on
# CPU; that class carries its own ``hardware_test`` mark for the GPU runner.
pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

cuda_available = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="GPU platform not available.")

bitsandbytes_available = pytest.mark.skipif(
    importlib.util.find_spec("bitsandbytes") is None,
    reason="bitsandbytes package not installed",
)


def _ensure_bitsandbytes_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a stub bitsandbytes package when the optional dep is absent.

    ``mocker.patch("bitsandbytes...")`` imports the target module; without a
    stub, core-model CI without bitsandbytes raises ModuleNotFoundError.
    """
    if importlib.util.find_spec("bitsandbytes") is not None:
        return
    bnb = types.ModuleType("bitsandbytes")
    bnb_functional = types.ModuleType("bitsandbytes.functional")
    bnb.functional = bnb_functional
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", bnb_functional)


def test_bitsandbytes_config_creation():
    """Test that BitsAndBytes config can be created."""
    config = build_quant_config("bitsandbytes")
    assert config is not None
    assert config.get_name() == "bitsandbytes"


def test_bitsandbytes_compress_statistics_defaults_off():
    """Nested absmax must stay opt-in on both construction paths.

    It has been observed to turn the output of wide projections into 1e30 --
    see ``DiffusionBitsAndBytesConfig.__init__``. Flipping this default back
    would silently re-enable that on every checkpoint that does not pin the
    flag, so it is asserted rather than left to review.
    """
    from vllm_omni.quantization.bitsandbytes_config import DiffusionBitsAndBytesConfig

    assert build_quant_config("bitsandbytes").compress_statistics is False
    assert DiffusionBitsAndBytesConfig.from_config({}).compress_statistics is False
    # An explicit opt-in still wins.
    assert DiffusionBitsAndBytesConfig.from_config({"compress_statistics": True}).compress_statistics is True


def test_prequantized_checkpoint_gets_vllms_config():
    """An already-quantized checkpoint must NOT get the online diffusion config.

    DiffusionBitsAndBytesConfig quantizes ``layer.weight`` during loading, so
    giving it uint8-packed 4-bit weights produces garbage, not an error. The
    pre-quantized path belongs to vLLM's BitsAndBytesConfig, which is what
    HunyuanImage-3's DiT was written against: it reads
    ``quant_config.llm_int8_skip_modules`` to keep the router gate in BF16
    (hunyuan_image3_transformer.py). An upstream merge silently swapped the class
    under it; this pins the routing back.
    """
    from vllm.model_executor.layers.quantization import get_quantization_config

    config = build_quant_config(
        {
            "quant_method": "bitsandbytes",
            "_load_in_4bit": True,
            "_load_in_8bit": False,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "llm_int8_skip_modules": ["mlp.gate"],
        }
    )
    assert isinstance(config, get_quantization_config("bitsandbytes"))
    assert config.get_name() == "bitsandbytes"
    # The load-bearing one: losing it silently re-quantizes the router gate.
    assert config.llm_int8_skip_modules == ["mlp.gate"]


def test_online_request_still_gets_the_diffusion_config():
    """No serialized markers means a BF16 checkpoint quantized at load time."""
    from vllm_omni.quantization.bitsandbytes_config import DiffusionBitsAndBytesConfig

    assert isinstance(build_quant_config("bitsandbytes"), DiffusionBitsAndBytesConfig)
    assert isinstance(
        build_quant_config({"quant_method": "bitsandbytes", "quant_type": "fp4"}),
        DiffusionBitsAndBytesConfig,
    )


def test_checkpoint_aliases_reach_the_online_config():
    """modules_to_not_convert has a home on the online config; it must reach it.

    Not a serialized marker, so this stays on the diffusion path -- a deploy-time
    skip list for online quantization. Dropping it would silently quantize layers
    the caller asked to leave alone.
    """
    config = build_quant_config({"quant_method": "bitsandbytes", "modules_to_not_convert": ["proj_out"]})
    assert config.ignored_layers == ["proj_out"]


def test_explicit_params_win_over_checkpoint_aliases():
    """An alias must never overwrite a value the caller set explicitly."""
    config = build_quant_config(
        {
            "quant_method": "bitsandbytes",
            "modules_to_not_convert": ["from_checkpoint"],
            "ignored_layers": ["explicit"],
        }
    )
    assert config.ignored_layers == ["explicit"]


def test_bitsandbytes_config_with_custom_params():
    """Test BitsAndBytes config with custom parameters."""
    config = build_quant_config(
        "bitsandbytes",
        quant_type="fp4",
        compress_statistics=False,
        ignored_layers=["to_out"],
    )
    assert config is not None
    assert config.quant_type == "fp4"
    assert config.compress_statistics is False
    assert "to_out" in config.ignored_layers


def test_supported_methods():
    """Test that supported methods list includes bitsandbytes."""
    assert "bitsandbytes" in SUPPORTED_QUANTIZATION_METHODS


def test_quantization_integration():
    """Test end-to-end quantization flow through OmniDiffusionConfig."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test", quantization_config="bitsandbytes")
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "bitsandbytes"

    config2 = OmniDiffusionConfig(
        model="test",
        quantization_config={
            "method": "bitsandbytes",
            "quant_type": "nf4",
            "compress_statistics": True,
        },
    )
    assert config2.quantization_config is not None
    assert config2.quantization_config.get_name() == "bitsandbytes"
    assert config2.quantization_config.quant_type == "nf4"
    assert config2.quantization_config.compress_statistics is True


def test_quantization_dict_not_mutated():
    """Test that passing a dict to quantization_config doesn't mutate it."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    original_dict = {"method": "bitsandbytes", "quant_type": "nf4"}
    dict_copy = original_dict.copy()

    OmniDiffusionConfig(model="test", quantization_config=original_dict)

    assert original_dict == dict_copy


def test_get_quant_method(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    """Test get_quant_method routing for CUDA."""
    from vllm_omni.quantization.bitsandbytes_config import BnBOnlineLinearMethod

    config = build_quant_config("bitsandbytes")

    def _fake_init(self, quant_config):
        pass

    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(BnBOnlineLinearMethod, "__init__", _fake_init)

    prefix = "test_layer"

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, BnBOnlineLinearMethod)

    config.ignored_layers = [prefix]
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, UnquantizedLinearMethod)


class TestBnBOnlineLinearMethod:
    @pytest.fixture
    def mock_quant_config(self, mocker):
        config = mocker.Mock()
        config.quant_type = "nf4"
        config.compress_statistics = True
        return config

    @pytest.fixture
    def mock_deps(self, mocker, monkeypatch: pytest.MonkeyPatch):
        _ensure_bitsandbytes_importable(monkeypatch)
        mock_qweight = torch.ones((64, 32), dtype=torch.uint8)
        mock_quant_state = mocker.Mock()
        mock_quant = mocker.patch(
            "bitsandbytes.functional.quantize_4bit",
            return_value=(mock_qweight, mock_quant_state),
            create=True,
        )
        # x is (2, 16, 32) -> flattened (32, 32); matmul must return (32, 64)
        # so apply can reshape back to (2, 16, 64).
        mock_matmul = mocker.patch(
            "bitsandbytes.matmul_4bit",
            return_value=torch.randn(32, 64),
            create=True,
        )
        return {
            "quant": mock_quant,
            "matmul": mock_matmul,
            "mock_qweight": mock_qweight,
            "mock_quant_state": mock_quant_state,
        }

    def test_process_weights_after_loading(self, mock_deps, mock_quant_config, mocker):
        from vllm_omni.quantization.bitsandbytes_config import BnBOnlineLinearMethod

        method = BnBOnlineLinearMethod(mock_quant_config)
        layer = Module()
        layer.weight = Parameter(torch.randn(64, 32))
        mocker.patch.object(torch.Tensor, "cuda", return_value=layer.weight.data)

        method.process_weights_after_loading(layer)

        mock_deps["quant"].assert_called_once()
        assert torch.equal(layer.weight.data, mock_deps["mock_qweight"])
        assert layer.quant_state is mock_deps["mock_quant_state"]
        assert layer.bnb_shape == (64, 32)

    def test_apply(self, mock_deps, mock_quant_config):
        from vllm_omni.quantization.bitsandbytes_config import BnBOnlineLinearMethod

        method = BnBOnlineLinearMethod(mock_quant_config)
        layer = Module()
        layer.weight = Parameter(torch.ones((64, 32), dtype=torch.uint8), requires_grad=False)
        layer.quant_state = mock_deps["mock_quant_state"]

        x = torch.randn(2, 16, 32, dtype=torch.float16)
        bias = torch.randn(64)
        output = method.apply(layer, x, bias)

        mock_deps["matmul"].assert_called_once()
        assert output.shape == (2, 16, 64)
        assert output.dtype == torch.float16


@pytest.fixture
def quant_config():
    from vllm_omni.quantization.bitsandbytes_config import DiffusionBitsAndBytesConfig

    return DiffusionBitsAndBytesConfig(
        quant_type="nf4",
        compress_statistics=True,
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@cuda_available
@bitsandbytes_available
class TestCudaBnBSmoke:
    """Smoke tests using real bitsandbytes CUDA kernels."""

    @pytest.fixture
    def real_layer(self):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn(128, 64, dtype=torch.float16, device="cuda"),
            requires_grad=False,
        )
        layer.logical_widths = [128]
        layer.input_size_per_partition = 64
        layer.output_size_per_partition = 128
        layer.orig_dtype = torch.float16
        return layer

    def test_real_cuda_quantize_4bit_shape_contract(self, quant_config):
        import bitsandbytes.functional as bnb_F

        weight = torch.randn(128, 64, dtype=torch.float16, device="cuda")
        qweight, quant_state = bnb_F.quantize_4bit(
            weight,
            quant_type=quant_config.quant_type,
            compress_statistics=quant_config.compress_statistics,
        )

        assert qweight.dtype == torch.uint8
        assert quant_state is not None

    def test_real_cuda_online_process_weights_after_loading(self, quant_config, real_layer):
        from vllm_omni.quantization.bitsandbytes_config import BnBOnlineLinearMethod

        method = BnBOnlineLinearMethod(quant_config)
        method.process_weights_after_loading(real_layer)

        assert real_layer.weight.dtype == torch.uint8
        assert hasattr(real_layer, "quant_state")
        assert real_layer.bnb_shape == (128, 64)

    def test_real_cuda_bnb_apply_forward(self, quant_config):
        from vllm_omni.quantization.bitsandbytes_config import BnBOnlineLinearMethod

        method = BnBOnlineLinearMethod(quant_config)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn(128, 64, dtype=torch.float16, device="cuda"),
            requires_grad=False,
        )
        method.process_weights_after_loading(layer)

        x = torch.randn(2, 16, 64, dtype=torch.float16, device="cuda")
        output = method.apply(layer, x)

        assert output.shape == (2, 16, 128)
        assert output.dtype == torch.float16
