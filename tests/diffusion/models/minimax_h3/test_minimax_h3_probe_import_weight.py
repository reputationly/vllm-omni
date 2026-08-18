# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What answering a capability question is allowed to cost the serving process.

``diffusion/model_metadata.py`` is the lightweight boundary: the HTTP layer asks
it booleans about models it may not be serving, and it imports
``models.minimax_h3.strategy`` lazily with a comment saying the serving layer
must not pull a model package in to answer one.

The comment could not make that true. Importing *any* module of a package runs
the package's ``__init__``, and this one eagerly imported the pipeline — so the
lazy import bought nothing: ~18 s and ~9200 modules, including torch,
diffusers, transformers, vLLM and the H3 transformer, landing in the HTTP
process on the first probe. In a metadata-only deployment that is not merely
slow, it is a process that can fail on dependencies for a model it never runs.

Asserted in a subprocess, because by the time this test runs the session has
long since imported torch for other reasons. A test in-process could only
observe the cache.

No weights, no GPU.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _in_a_fresh_interpreter(body: str) -> list[str]:
    """Run ``body`` with a clean import cache; return its stdout lines.

    Failures come back as the child's stderr rather than as a bare non-zero
    exit, because the interesting failure here *is* an import error.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"the child interpreter failed:\n{result.stderr}"
    return [line for line in result.stdout.splitlines() if line.startswith("::")]


def test_the_capability_probe_does_not_import_the_pipeline():
    """The probe's own import path, exercised exactly as the serving layer runs it."""
    reported = _in_a_fresh_interpreter(
        """
        import sys
        from vllm_omni.diffusion.model_metadata import reference_images_bind_output_canvas

        class _Config:
            model_class_name = "MiniMaxH3Pipeline"
            minimax_h3_inference_contract = "official_diffusers_v1"
            minimax_h3_admission_policy = None
            diffusion_runtime_environ = None

        # The answer still has to be right; a probe that returns the default
        # because the import quietly failed would satisfy a weight budget alone.
        assert reference_images_bind_output_canvas("MiniMaxH3Pipeline", _Config()) is False
        print("::pipeline", "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3" in sys.modules)
        print("::transformer", any("minimax_h3_transformer" in name for name in sys.modules))
        print("::torch", "torch" in sys.modules)
        """
    )

    assert "::pipeline False" in reported, "the probe pulled the H3 pipeline into the serving process"
    assert "::transformer False" in reported, "the probe pulled the H3 transformer in"
    # torch is not asserted absent: `model_metadata` sits under `vllm_omni`,
    # whose own import chain reaches it. The pipeline is what this boundary can
    # actually keep out, and it is the part that costs seconds.


def test_the_package_still_exports_the_pipeline_to_whoever_asks_for_it():
    """Laziness must not become a missing name — the registry imports by attribute."""
    reported = _in_a_fresh_interpreter(
        """
        import sys
        from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

        print("::name", MiniMaxH3Pipeline.__name__)
        print("::loaded", "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3" in sys.modules)
        """
    )

    assert "::name MiniMaxH3Pipeline" in reported
    assert "::loaded True" in reported, "asking for the pipeline must still load it"


def test_an_unknown_attribute_is_still_an_attribute_error():
    """`__getattr__` swallowing typos would turn them into import-time mysteries."""
    from vllm_omni.diffusion.models import minimax_h3

    with pytest.raises(AttributeError, match="no attribute 'MiniMaxH4Pipeline'"):
        minimax_h3.MiniMaxH4Pipeline
