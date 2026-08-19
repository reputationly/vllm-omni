# SPDX-License-Identifier: Apache-2.0
"""The reference-image short edge: its bounds, and the one parser for it.

This module exists because the knob had grown *two* parsers with different
answers. The pipeline's read-the-env path warned and degraded; the strategy's
startup path raised. They disagreed in four places, and one of the four
disagreements was unsafe:

    input                     pipeline (old)      strategy (old)
    "abc"                     warn -> 2048        raise
    16                        clamp -> 32         raise
    1000  (in range, not %32) 1000                raise
    10240 (%32, over 5760)    warn -> 2048        **accepted**

The last row is the reason this is one function now rather than two agreeing
ones. The invariant is stated once, here, and it is not "reject bad input" —
it is **no input may resolve to a target more expensive than the default**. A
clamp-up looks like the strict choice and is the opposite: 2560x1024 at a short
edge of 10240 packs 256000 rows, ~25x the 2048 tier, which is precisely the OOM
this knob was added to prevent. So the lower bound clamps (smaller is always
safer) and the upper bound falls back to the default (clamping up would not be).

The multiple-of-32 requirement the strategy briefly enforced is not a
requirement at all: ``_align_multiple`` rounds the *resolved* geometry to 32,
so 1000 has always been a usable value and legacy deployments may be carrying
one.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV = "VLLM_OMNI_H3_REF_IMAGE_SHORT_EDGE"
MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV = "VLLM_OMNI_H3_REF_IMAGE_NO_UPSCALE"
MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV = "VLLM_OMNI_H3_REF_IMAGE_MAX_PIXELS"

# The patch alignment every resolved reference-image geometry is rounded to.
MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE = 32

# Bounds on the normalization **target**. Deliberately not reused from
# reference_video.py's MINIMAX_H3_{MIN,MAX}_REFERENCE_DIMENSION: those bound the
# **uploaded** media's edge length, this bounds what it is scaled *to*. Two
# concepts that happen to sit near the same numbers.
#
# Lower bound is the patch alignment: below it `_align_multiple` rounds to zero.
# Upper bound is 5760: a target short edge above the largest admissible upload
# has no meaning.
MINIMAX_H3_REFERENCE_IMAGE_MIN_TARGET = MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE
MINIMAX_H3_REFERENCE_IMAGE_MAX_TARGET = 5760

# What production has always run at. Every degradation path below resolves here.
MINIMAX_H3_REFERENCE_IMAGE_DEFAULT_TARGET = 2048


def parse_reference_image_short_edge(raw: str) -> int:
    """Resolve one operator-supplied short edge, degrading rather than raising.

    A deploy file is edited by hand and read at startup, so a typo that hard
    fails takes the instance down; a typo that silently doubles the token cost
    per reference image takes the *fleet* down. This resolves the first case to
    the documented default and refuses to resolve the second at all.

    Args:
        raw: The operator's value, already stripped. Empty is the caller's
            business — an unset knob never reaches here.

    Returns:
        A short edge in ``[MIN_TARGET, MAX_TARGET]``, never larger than
        :data:`MINIMAX_H3_REFERENCE_IMAGE_DEFAULT_TARGET` for invalid input.
    """
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; falling back to %d",
            MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV,
            raw,
            MINIMAX_H3_REFERENCE_IMAGE_DEFAULT_TARGET,
        )
        return MINIMAX_H3_REFERENCE_IMAGE_DEFAULT_TARGET
    if value < MINIMAX_H3_REFERENCE_IMAGE_MIN_TARGET:
        # Clamping up is safe here and only here: the result is the cheapest
        # target there is, so it cannot betray the "I set this to save memory"
        # intent that a below-bounds value expresses.
        logger.warning(
            "%s=%d is below the %d-pixel patch alignment; clamping to %d",
            MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV,
            value,
            MINIMAX_H3_REFERENCE_IMAGE_MIN_TARGET,
            MINIMAX_H3_REFERENCE_IMAGE_MIN_TARGET,
        )
        return MINIMAX_H3_REFERENCE_IMAGE_MIN_TARGET
    if value > MINIMAX_H3_REFERENCE_IMAGE_MAX_TARGET:
        logger.warning(
            "%s=%d exceeds the %d-pixel ceiling; falling back to %d rather than clamping up",
            MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV,
            value,
            MINIMAX_H3_REFERENCE_IMAGE_MAX_TARGET,
            MINIMAX_H3_REFERENCE_IMAGE_DEFAULT_TARGET,
        )
        return MINIMAX_H3_REFERENCE_IMAGE_DEFAULT_TARGET
    return value


def parse_reference_image_no_upscale(raw: str) -> bool:
    """Resolve the opt-in no-upscale switch from its transported string."""
    return str(raw).strip() in {"1", "true", "True"}


def parse_reference_image_max_pixels(raw: str) -> int:
    """Resolve the reference-image area ceiling; zero keeps legacy uncapped."""
    raw = str(raw).strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; area cap stays disabled",
            MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV,
            raw,
        )
        return 0
    if value < 0:
        logger.warning(
            "%s=%d is negative; area cap stays disabled",
            MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV,
            value,
        )
        return 0
    minimum_area = MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE**2
    if 0 < value < minimum_area:
        logger.warning(
            "%s=%d is below one %dx%d patch; clamping to %d",
            MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV,
            value,
            MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
            MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
            minimum_area,
        )
        return minimum_area
    return value


__all__ = [
    "MINIMAX_H3_REFERENCE_IMAGE_DEFAULT_TARGET",
    "MINIMAX_H3_REFERENCE_IMAGE_MAX_TARGET",
    "MINIMAX_H3_REFERENCE_IMAGE_MAX_PIXELS_ENV",
    "MINIMAX_H3_REFERENCE_IMAGE_MIN_TARGET",
    "MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE",
    "MINIMAX_H3_REFERENCE_IMAGE_NO_UPSCALE_ENV",
    "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE_ENV",
    "parse_reference_image_max_pixels",
    "parse_reference_image_no_upscale",
    "parse_reference_image_short_edge",
]
