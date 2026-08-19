# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

BASE_SCHEDULE_KEY = "base_schedule"


@dataclass(frozen=True)
class DMD2SigmaSchedule:
    """Recommended rectified-flow positions carried by a distilled checkpoint.

    Entries are sigma boundaries. A schedule with N+1 boundaries performs N
    transformer evaluations.  The checkpoint's schedule supplies its default
    and records the grid it was distilled on; callers may still request a
    different NFE, as the official MiniMax-H3-Turbo inference entry point does.
    """

    base_schedule: tuple[float, ...]

    def __post_init__(self) -> None:
        values = self.base_schedule
        if len(values) < 2:
            raise ValueError("DMD2 base_schedule needs at least 2 entries")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("DMD2 base_schedule entries must be finite")
        if values[0] != 1.0 or values[-1] != 0.0:
            raise ValueError("DMD2 base_schedule must start at 1.0 and end at 0.0")
        if any(curr <= nxt for curr, nxt in zip(values, values[1:], strict=False)):
            raise ValueError("DMD2 base_schedule must be strictly decreasing")

    @classmethod
    def from_positions(cls, base_schedule: Sequence[float]) -> DMD2SigmaSchedule:
        return cls(base_schedule=tuple(float(value) for value in base_schedule))

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        key: str = BASE_SCHEDULE_KEY,
    ) -> DMD2SigmaSchedule | None:
        raw = metadata.get(key)
        if raw is None:
            return None
        return cls.from_positions(raw)

    @property
    def num_inference_steps(self) -> int:
        return len(self.base_schedule) - 1

    def shifted_sigmas(self, shift_scale: float) -> list[float]:
        if shift_scale <= 0:
            raise ValueError("DMD2 shift_scale must be > 0")
        shift = float(shift_scale)
        return [shift * value / (1 + (shift - 1) * value) for value in self.base_schedule]

    def positions_for_num_inference_steps(self, num_inference_steps: int) -> tuple[float, ...]:
        """Return this grid at its native NFE, or a uniform N+1 grid otherwise."""
        steps = int(num_inference_steps)
        if steps < 1:
            raise ValueError("num_inference_steps must be at least 1")
        if steps == self.num_inference_steps:
            return self.base_schedule
        return tuple(float(steps - index) / steps for index in range(steps + 1))


__all__ = ["BASE_SCHEDULE_KEY", "DMD2SigmaSchedule"]
