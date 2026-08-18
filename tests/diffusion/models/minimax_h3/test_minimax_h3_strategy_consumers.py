# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every contract field has to have a reader somewhere.

This exists because of a defect class, not a defect. Five of the six findings in
the 2026-08-18 review were the same shape: a semantic was declared — on
``StageDeployConfig``, on ``MiniMaxH3InferenceStrategy``, in a docstring — and
nothing downstream read it. Each one presented as an instance that quietly
served the other contract while every log said it was serving this one, which is
the worst possible failure mode for a parity effort: the artifact is wrong and
the evidence says it is right.

A per-field test would have to be remembered for the *next* field. This one is
mechanical: walk the dataclass, and fail if a field name never appears outside
``strategy.py``. It cannot prove the reader is correct, only that one exists —
which is precisely the check that was missing.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# Fields read through an accessor rather than by name. Each entry names the
# accessor, so an entry cannot be added without pointing at the real reader.
_READ_VIA_ACCESSOR = {
    "default_num_frames_by_task": "MiniMaxH3InferenceStrategy.default_num_frames",
    "name": "MiniMaxH3InferenceStrategy.is_official",
    # These two decide ONE thing together — which `num_frames` are admissible —
    # and they are read through the accessor precisely so they cannot be read
    # apart. The bug that made the accessor necessary was the duration bound
    # being applied to the requested frame count and the alignment being applied
    # after it, i.e. two readers of one rule disagreeing about ordering.
    "output_duration_seconds": "MiniMaxH3InferenceStrategy.requested_frame_window",
    "duration_validation_mode": "MiniMaxH3InferenceStrategy.requested_frame_window",
}


def _strategy_fields() -> list[str]:
    from vllm_omni.diffusion.models.minimax_h3.strategy import MiniMaxH3InferenceStrategy

    return list(MiniMaxH3InferenceStrategy.__dataclass_fields__)


def _package_sources() -> list[Path]:
    """Every ``vllm_omni`` source except ``strategy.py`` itself.

    Scoped to the ``minimax_h3`` package at first, which was too narrow and
    reported a false positive on ``reference_image_geometry_mode``: its consumer
    is ``vllm_omni/diffusion/model_metadata.py``, where the *serving* layer asks
    whether reference images are bound to the generated canvas. That is exactly
    where a contract field is most likely to be read — the pre-stretch this
    field gates happens before the pipeline is ever entered — so a guard that
    cannot see outside the model package would push a live field toward
    deletion. Parsing the whole tree costs a couple of seconds and removes the
    judgement call about which directories count.
    """
    import vllm_omni

    root = Path(inspect.getfile(vllm_omni)).parent
    return [path for path in sorted(root.rglob("*.py")) if path.name != "strategy.py"]


def _attribute_reads(sources: list[Path]) -> set[str]:
    """Attribute names read anywhere in the package, from the AST.

    Parsed rather than grepped so a field mentioned only in a comment or a
    docstring does not count as a consumer — a documented knob nobody reads is
    exactly the thing being guarded against.
    """
    names: set[str] = set()
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                # ``prepare_reference_videos_official(truncate_to_target=...)``
                # reads the field on the right-hand side; that read is already
                # an Attribute. Keywords are collected so a field forwarded
                # under its own name still counts.
                names.add(node.arg)
    return names


def test_every_contract_field_has_a_reader_outside_the_strategy():
    """A declared semantic with no consumer is a silently wrong instance."""
    read = _attribute_reads(_package_sources())
    unread = [field for field in _strategy_fields() if field not in read and field not in _READ_VIA_ACCESSOR]
    assert not unread, (
        f"MiniMaxH3InferenceStrategy declares {unread} but nothing in vllm_omni reads them. "
        "Wire the field to the behaviour it names, or delete it — a contract field that only reaches "
        "describe() makes the log claim a semantic the instance does not apply."
    )


def test_the_accessor_allowlist_names_real_accessors():
    """The escape hatch must not become a place to park unread fields."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import MiniMaxH3InferenceStrategy

    for field, accessor in _READ_VIA_ACCESSOR.items():
        assert field in MiniMaxH3InferenceStrategy.__dataclass_fields__
        owner, _, attribute = accessor.partition(".")
        assert owner == "MiniMaxH3InferenceStrategy"
        assert hasattr(MiniMaxH3InferenceStrategy, attribute), accessor
    # And the accessors really are reachable from the package.
    assert "default_num_frames" in _attribute_reads(_package_sources())
    assert "is_official" in _attribute_reads(_package_sources())


def test_the_guard_would_catch_a_newly_added_dead_field():
    """The test above has to be able to fail; prove it on a synthetic field."""
    read = _attribute_reads(_package_sources())
    assert "minimax_h3_field_nobody_reads" not in read


def test_describe_reports_every_field_it_still_declares():
    """An operator reads the contract off this dict, so it must be complete."""
    from vllm_omni.diffusion.models.minimax_h3.strategy import official_diffusers_v1_strategy

    described = official_diffusers_v1_strategy().describe()
    missing = [
        field
        for field in _strategy_fields()
        # ``name`` is reported under its meaning rather than its attribute name.
        if field != "name" and field not in described
    ]
    assert not missing, missing
    assert described["inference_contract"] == "official_diffusers_v1"
