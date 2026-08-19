# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 audio-video diffusion support.

The pipeline is bound on first use rather than on import, because importing
*any* module in this package runs this file, and several of them are read by
processes that will never run the model. The serving layer's capability probes
are the ones that matter: they import ``.strategy`` to answer a boolean, and an
eager ``from .pipeline_minimax_h3 import ...`` here turned that into ~18 seconds
and ~9200 modules — torch, diffusers, transformers, vLLM and the H3 transformer
— pulled into the HTTP process. ``model_metadata`` already carries a comment
saying this must not happen; the comment could not make it true, because the
cost is paid by the *package*, not by the module being asked for.

Lazy binding leaves the public names exactly where callers expect them:
``from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline`` still
works and still pays for the pipeline — that caller asked for it.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_minimax_h3 import MiniMaxH3Pipeline, get_minimax_h3_post_process_func

_PIPELINE_EXPORTS = ("MiniMaxH3Pipeline", "get_minimax_h3_post_process_func")

__all__ = [
    "MiniMaxH3Pipeline",
    "get_minimax_h3_post_process_func",
]


def __getattr__(name: str):
    if name in _PIPELINE_EXPORTS:
        from . import pipeline_minimax_h3

        value = getattr(pipeline_minimax_h3, name)
        # Cached in the module namespace, so the import runs once per name and
        # every later attribute access costs what a plain import would have.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
