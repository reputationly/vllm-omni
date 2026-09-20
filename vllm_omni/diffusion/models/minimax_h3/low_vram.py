# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 activation-memory knobs: chunked FFN and head-grouped attention.

At 1088p/15s the DiT holds 216k packed rows, and the weights are only ~4.9 GiB per rank
of a 35.8 GiB peak -- the rest is activations, dominated by two per-block transients:

* ``fc1``'s fused ``[gate, up]`` output, ``rows x 2*ffn/TP``, ~3.1 GiB
* the packed ``q/k/v``, ``rows x 3*hidden/TP``, ~1.8 GiB

Both are splittable along an axis whose elements do not interact, so splitting them
trades peak memory for extra kernel launches and nothing else.

**Chunked FFN is exact, but only because it stops before ``fc2``.** A packed row's SwiGLU
depends on that row alone, so slicing the row axis through ``fc1`` and the activation is
bit-identical. Carrying the slices through ``fc2`` is NOT: that layer is a
``RowParallelLinear`` which all-reduces internally, so ``n`` chunks issue ``n`` collectives
and NCCL picks its reduction algorithm by message size -- changing the cross-rank
summation order. Measured on 5s/1088p with TP4, chunking through ``fc2`` left the video
bit-identical but moved the audio by 3.7% of full scale. The implementation therefore
buffers the activation and calls ``fc2`` once, which halves the saving and keeps the
numerics.

**Head grouping is exact in math but not necessarily in arithmetic.** Attention heads are
independent, so computing them in groups is the same function -- but a different call
shape can select a different kernel or reduction order, so results may differ in the last
bits. It is therefore off by default and separate from the FFN knob, rather than one
"low VRAM" switch that silently changes numerics.

Both default to off, so an unconfigured deployment behaves exactly as before.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

FFN_CHUNKS_ENV = "VLLM_OMNI_H3_FFN_CHUNKS"
FFN_ROW_THRESHOLD_ENV = "VLLM_OMNI_H3_FFN_ROW_THRESHOLD"
ATTENTION_HEAD_CHUNKS_ENV = "VLLM_OMNI_H3_ATTENTION_HEAD_CHUNKS"

# Below this many packed rows the transient is small enough that the extra collectives
# cost more than the memory they save. Mirrors the reference node's 4096.
DEFAULT_FFN_ROW_THRESHOLD = 4096

_FFN_ANNOUNCED = False


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def ffn_row_chunks(rows: int) -> int:
    """How many row chunks the SwiGLU should run in; 1 means run it whole."""
    chunks = _positive_int(FFN_CHUNKS_ENV, 1)
    if chunks == 1:
        return 1
    if rows <= _positive_int(FFN_ROW_THRESHOLD_ENV, DEFAULT_FFN_ROW_THRESHOLD):
        return 1
    # More chunks than rows would make empty slices, which some linear kernels reject.
    chunks = min(chunks, rows)
    global _FFN_ANNOUNCED
    if not _FFN_ANNOUNCED:
        _FFN_ANNOUNCED = True
        logger.info("H3 low-VRAM: chunking SwiGLU into %d row groups for %d packed rows", chunks, rows)
    return chunks


def attention_head_chunks(num_heads: int) -> int:
    """How many head groups attention should run in; 1 means one call."""
    chunks = _positive_int(ATTENTION_HEAD_CHUNKS_ENV, 1)
    return min(chunks, max(1, num_heads)) if chunks > 1 else 1


__all__ = [
    "ATTENTION_HEAD_CHUNKS_ENV",
    "DEFAULT_FFN_ROW_THRESHOLD",
    "FFN_CHUNKS_ENV",
    "FFN_ROW_THRESHOLD_ENV",
    "attention_head_chunks",
    "ffn_row_chunks",
]
