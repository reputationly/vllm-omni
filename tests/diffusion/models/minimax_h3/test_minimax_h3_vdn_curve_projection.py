# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The 1025-point affine projection, and the bias residual it produces.

Curve pruning narrows AdaLN's input from the 2688-wide time embedding to an 8-wide
sampled table, which an adapter trained on the full model cannot address. The fix is not
to project the delta's row space -- that discards ~93% of it -- but to fit the delta on
the 1025 timesteps AdaLN actually sees, WITH an intercept. The intercept is the whole
trick: without it the fit is worse than useless, because the curve does not pass through
the origin.

These tests pin both halves: that the projection reconstructs the delta on the grid, and
that the intercept it hands to ``folded_bias`` is applied there rather than dropped. A
dropped intercept would still load, still render, and still be most of the AdaLN response
missing.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

GRID, FULL_WIDTH, CURVE_WIDTH, RANK, OUT = 1025, 96, 8, 4, 32


def _curves(seed=0):
    """A full curve and an 8-wide table that spans it affinely, as pruning produces.

    Built as ``table @ basis + offset`` so the pruned table genuinely IS a curve-pruned
    view of the full one -- fitting a random pair would measure nothing but least squares.
    """
    torch.manual_seed(seed)
    table = torch.randn(GRID, CURVE_WIDTH, dtype=torch.float64)
    basis = torch.randn(CURVE_WIDTH, FULL_WIDTH, dtype=torch.float64)
    offset = torch.randn(FULL_WIDTH, dtype=torch.float64)
    return table, table @ basis + offset


def test_projection_reconstructs_the_delta_on_the_grid():
    from tools.minimax_h3.vdn_curve_projection import CurveProjector

    table, full = _curves()
    projector = CurveProjector.build(table, full)
    assert projector.reconstruction_error < 1e-10

    torch.manual_seed(1)
    lora_a = torch.randn(RANK, FULL_WIDTH)
    lora_b = torch.randn(OUT, RANK)
    a8, diff_b, error = projector.project(lora_a, lora_b)

    assert a8.shape == (RANK, CURVE_WIDTH)
    assert diff_b.shape == (OUT,)
    assert error < 1e-10

    # ...and the reconstruction is the delta the unpruned model would have applied.
    target = (full @ lora_a.double().T) @ lora_b.double().T
    fitted = (table @ a8.T) @ lora_b.double().T + diff_b
    torch.testing.assert_close(fitted, target, rtol=1e-8, atol=1e-8)


def test_dropping_the_intercept_destroys_the_fit():
    """The guard on the one mistake that looks like a simplification.

    A purely linear fit is the natural thing to write and it is catastrophically wrong;
    this pins the difference so nobody 'cleans up' the constant column.
    """
    from tools.minimax_h3.vdn_curve_projection import CurveProjector

    table, full = _curves(seed=2)
    with_intercept = CurveProjector.build(table, full).reconstruction_error

    linear_only = torch.linalg.pinv(table) @ full
    without = float(torch.linalg.norm(full - table @ linear_only) / torch.linalg.norm(full))

    assert with_intercept < 1e-10
    assert without > 0.1
    assert without > with_intercept * 1e6


def test_curve_is_rejected_when_the_grids_disagree():
    from tools.minimax_h3.vdn_curve_projection import CurveProjector

    table, full = _curves()
    with pytest.raises(ValueError, match="grid mismatch"):
        CurveProjector.build(table[:-1], full)


def test_bias_residual_lands_on_folded_bias(tmp_path):
    """``diff_b`` must reach the pruned parameter, in fp32, summed over adapters.

    ``fuse`` returns the weight unchanged for any name it has no patch for, so an
    intercept written to the artifact but never resolved onto ``folded_bias`` is silent:
    the bake reports success, the model loads, and the modulation is simply absent.
    """
    from vllm_omni.diffusion.models.minimax_h3.vdn import _BIAS_LAYOUT, VDNCheckpoint, _Patch

    name = "blocks.0.adaln_proj.folded_bias"
    residuals = [torch.randn(OUT, dtype=torch.float32), torch.randn(OUT, dtype=torch.float32)]
    checkpoint = VDNCheckpoint(
        source=tmp_path,
        spec=None,
        metadata={},
        injections={},
        patches={name: _Patch(layout=_BIAS_LAYOUT, biases=list(residuals))},
        head_dim=128,
        adapters=("turbo",),
    )

    base = torch.randn(OUT, dtype=torch.float32)
    fused = checkpoint.fuse(name, None, base.clone())
    torch.testing.assert_close(fused, base + residuals[0] + residuals[1])


def test_bias_residual_refuses_a_shape_that_is_not_the_parameter(tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.vdn import (
        _BIAS_LAYOUT,
        VDNCheckpoint,
        VDNCheckpointError,
        _Patch,
    )

    name = "blocks.0.adaln_proj.folded_bias"
    checkpoint = VDNCheckpoint(
        source=tmp_path,
        spec=None,
        metadata={},
        injections={},
        patches={name: _Patch(layout=_BIAS_LAYOUT, biases=[torch.randn(OUT + 1)])},
        head_dim=128,
        adapters=("turbo",),
    )
    with pytest.raises(VDNCheckpointError, match="bias residual"):
        checkpoint.fuse(name, None, torch.randn(OUT))
