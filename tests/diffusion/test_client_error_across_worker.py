# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A 4xx raised inside a diffusion worker must stay a 4xx at the engine boundary.

Regression guard for the MiniMax H3 symptom: an out-of-range ``duration``
raised ``OmniClientError`` in the pipeline, but every hop back to the API layer
carried only ``str(exc)``, so the request came back as HTTP 500.
"""

import pytest

from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind, DiffusionOutput
from vllm_omni.diffusion.executor.multiproc_executor import (
    MultiprocDiffusionExecutor,
    _async_output_error,
)
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MSG = "MiniMax H3 output duration must be in [2, 16] seconds, got 29.958"


def _worker_status(**overrides):
    """The reply shape ``_worker_busy_loop`` produces for a failed RPC."""
    status = {
        "status": "error",
        "error": _MSG,
        "error_status_code": 400,
        "error_type": "BadRequestError",
        "wave_id": 1,
    }
    status.update(overrides)
    return status


def test_error_dict_with_4xx_rebuilds_client_error():
    with pytest.raises(OmniClientError) as excinfo:
        MultiprocDiffusionExecutor._raise_for_rpc_error_dict(_worker_status())
    assert excinfo.value.status_code == 400
    assert excinfo.value.error_type == "BadRequestError"
    assert str(excinfo.value) == _MSG


def test_error_dict_without_status_stays_runtime_error():
    response = _worker_status(error_status_code=None, error_type="OutOfMemoryError")
    with pytest.raises(RuntimeError) as excinfo:
        MultiprocDiffusionExecutor._raise_for_rpc_error_dict(response)
    assert not isinstance(excinfo.value, OmniClientError)


def test_rank_statuses_all_4xx_rebuild_client_error():
    """Validation is deterministic, so every rank reports the same 4xx."""
    envelope = {
        "type": "diffusion_rpc_result",
        "method": "execute_model",
        "rank_statuses": [
            {"rank": i, "ok": False, "error": _MSG, "error_type": "BadRequestError", "error_status_code": 400}
            for i in range(4)
        ],
    }
    with pytest.raises(OmniClientError) as excinfo:
        MultiprocDiffusionExecutor._unwrap_rpc_result_envelope(envelope)
    assert excinfo.value.status_code == 400


def test_rank_statuses_mixed_failure_stays_500():
    """One genuine engine failure anywhere outranks the client errors."""
    envelope = {
        "type": "diffusion_rpc_result",
        "method": "execute_model",
        "rank_statuses": [
            {"rank": 0, "ok": False, "error": _MSG, "error_type": "BadRequestError", "error_status_code": 400},
            {"rank": 1, "ok": False, "error": "CUDA OOM", "error_type": "OutOfMemoryError", "error_status_code": None},
        ],
    }
    with pytest.raises(RuntimeError) as excinfo:
        MultiprocDiffusionExecutor._unwrap_rpc_result_envelope(envelope)
    assert not isinstance(excinfo.value, OmniClientError)


@pytest.mark.parametrize("kind", [AsyncOutputKind.RPC_RESULT, AsyncOutputKind.OUTPUT_READY])
def test_async_envelope_rebuilds_client_error(kind):
    msg = AsyncDiffusionOutput(
        kind=kind,
        rpc_id="rpc-1",
        error=_MSG,
        error_status_code=400,
        error_type="BadRequestError",
    )
    exc = _async_output_error(msg)
    assert isinstance(exc, OmniClientError)
    assert exc.status_code == 400


def test_async_envelope_without_status_stays_runtime_error():
    msg = AsyncDiffusionOutput(kind=AsyncOutputKind.RPC_RESULT, rpc_id="rpc-1", error="worker died")
    exc = _async_output_error(msg)
    assert isinstance(exc, RuntimeError) and not isinstance(exc, OmniClientError)


def test_from_exception_keeps_status_through_diffusion_output():
    """The executor's ``except Exception`` funnel must not flatten the status."""
    out = DiffusionOutput.from_exception(OmniClientError(_MSG))
    assert out.error_status_code == 400
    assert out.error_type == "BadRequestError"

    generic = DiffusionOutput.from_exception(RuntimeError("boom"))
    assert generic.error_status_code is None


def test_engine_postprocess_raises_client_error_not_runtime():
    """End of the chain: ``postprocess_output`` must re-raise as a client error."""
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    engine = object.__new__(DiffusionEngine)
    output = DiffusionOutput.from_exception(OmniClientError(_MSG))
    with pytest.raises(OmniClientError) as excinfo:
        DiffusionEngine.postprocess_output(engine, request=object(), output=output)
    assert excinfo.value.status_code == 400
