# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The 1025-point affine projection that lets an AdaLN adapter meet a curve-pruned H3.

Curve pruning replaces the 2688-wide time embedding that 51 AdaLN projections consume
with an 8-wide sampled basis: ``adaln_proj.linear`` becomes ``[out, 8]`` reading
``time_embedder.table[t]``, plus an fp32 ``folded_bias``. An adapter trained on the full
model carries ``dW = lora_B @ lora_A`` of shape ``[out, 2688]``, which cannot be added to
an ``[out, 8]`` parameter.

Projecting ``dW``'s row space onto the basis does NOT work -- measured on the released
turbo adapter, the relative residual is 0.9978, so 93% of the delta is discarded. That
number is what a *linear* projection over all of R^2688 buys, and it is the wrong
question: AdaLN never sees an arbitrary vector. Its input is ``silu(time_embedder(t))``
for a timestep, so over a run it takes exactly the 1025 values the pruned table samples.
Fitting there, and *affinely*, is enough::

    S @ A.T  ~=  C @ A8.T + c              S = [1025, 2688] full post-SiLU curve
                                           C = [1025, 8]    pruned table
    design   =  [C | 1]                    the intercept column is what makes it work
    P        =  pinv(design) @ S           [9, 2688]
    A8       =  (P[:8] @ A.T).T            [rank, 8]     replaces lora_A
    diff_b   =  B @ (P[8] @ A.T)           [out]         added to folded_bias

Dropping the intercept destroys most of the response: the curve does not pass through
the origin, and the constant it is offset by carries most of the modulation.

The method is OpenVDN-adjacent prior art, published under Apache-2.0 as
``adaln_curve_affine_lstsq_pinv1025_with_diff_b`` in
``T8mars/comfyui-minimax-h3-audio-T8``. This is an independent implementation against
our own tensor names, with the same self-check: the error is reported per module, in
fp64, against the delta the unpruned model would have applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# The released curve-pruning geometry. These describe how a checkpoint WAS BUILT, so
# they are validated against the artifact rather than offered as options.
GRID_SIZE = 1025
CURVE_WIDTH = 8
FREQ_WIDTH = 256


def sinusoidal_time_embedding(t: torch.Tensor) -> torch.Tensor:
    """H3's timestep features: ``[cos(t*f) | sin(t*f)]`` over 128 log-spaced frequencies.

    Reproduced rather than read off the model because the pruned checkpoint no longer
    carries the 2688-wide path at all -- the whole point of pruning it away.
    """
    t32 = t.to(dtype=torch.float64, device="cpu").reshape(-1)
    half = FREQ_WIDTH // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half)
    arguments = t32[:, None] * frequencies[None, :]
    return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)


def full_adaln_curve(time_reference: dict[str, torch.Tensor], grid_size: int = GRID_SIZE) -> torch.Tensor:
    """``silu(time_embedder(t))`` on the uniform grid: exactly what AdaLN consumes.

    The trailing SiLU belongs to AdaLnProj, not to the time embedder; the pruned table
    holds the coordinates of the post-SiLU vector, so it has to be applied here for the
    two sides to describe the same quantity.
    """
    grid = torch.arange(grid_size, dtype=torch.float64) / float(grid_size - 1)
    embedding = sinusoidal_time_embedding(grid)
    hidden = torch.nn.functional.silu(
        torch.nn.functional.linear(
            embedding,
            time_reference["proj_in.weight"].to(torch.float64),
            time_reference["proj_in.bias"].to(torch.float64),
        )
    )
    time_embedding = torch.nn.functional.linear(
        hidden,
        time_reference["proj_out.weight"].to(torch.float64),
        time_reference["proj_out.bias"].to(torch.float64),
    )
    return torch.nn.functional.silu(time_embedding).contiguous()


@dataclass(frozen=True)
class CurveProjector:
    """``P`` such that ``[C | 1] @ P`` reconstructs the full curve ``S``.

    Built once and reused for all 51 modules: the fit depends only on the two curves,
    not on the adapter.
    """

    projection: torch.Tensor  # [CURVE_WIDTH + 1, full_width], fp64
    curve: torch.Tensor  # [grid, CURVE_WIDTH], fp64
    full_curve: torch.Tensor  # [grid, full_width], fp64

    @classmethod
    def build(cls, curve: torch.Tensor, full_curve: torch.Tensor) -> CurveProjector:
        if curve.shape[0] != full_curve.shape[0]:
            raise ValueError(f"grid mismatch: table has {curve.shape[0]} rows, curve has {full_curve.shape[0]}")
        c64 = curve.to(torch.float64)
        s64 = full_curve.to(torch.float64)
        design = torch.cat([c64, torch.ones(c64.shape[0], 1, dtype=torch.float64)], dim=1)
        return cls(projection=torch.linalg.pinv(design) @ s64, curve=c64, full_curve=s64)

    @property
    def reconstruction_error(self) -> float:
        """How well the 8-wide table plus a constant reproduces the 2688-wide curve.

        This is the ceiling on every adapter projected through it, and it is a property
        of the two checkpoints alone -- so it is worth reading before trusting any
        adapter's own number.
        """
        design = torch.cat([self.curve, torch.ones(self.curve.shape[0], 1, dtype=torch.float64)], dim=1)
        residual = self.full_curve - design @ self.projection
        return float(torch.linalg.norm(residual) / torch.linalg.norm(self.full_curve))

    def project(self, lora_a: torch.Tensor, lora_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        """``(A8, diff_b, relative_error)`` for one AdaLN module.

        ``relative_error`` is measured on the delta that actually reaches the residual
        stream -- ``B @ (A @ s)``, not ``A @ s`` -- because that is what a wrong fit
        would corrupt, and B can amplify or cancel whatever A got wrong.
        """
        a64 = lora_a.to(torch.float64)
        b64 = lora_b.to(torch.float64)
        coefficients = self.projection @ a64.T  # [CURVE_WIDTH + 1, rank]
        a8 = coefficients[:CURVE_WIDTH].T.contiguous()
        intercept = coefficients[CURVE_WIDTH].contiguous()
        diff_b = b64 @ intercept

        target = (self.full_curve @ a64.T) @ b64.T  # [grid, out]
        fitted = (self.curve @ a8.T) @ b64.T + diff_b
        error = float(torch.linalg.norm(target - fitted) / (torch.linalg.norm(target) + 1e-300))
        return a8, diff_b, error


__all__ = [
    "CURVE_WIDTH",
    "FREQ_WIDTH",
    "GRID_SIZE",
    "CurveProjector",
    "full_adaln_curve",
    "sinusoidal_time_embedding",
]
