# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The contract has to survive every hop between the YAML and the pipeline.

Two of the six review findings in this effort were the same accident: a contract
field was declared where it is *configured* and read where it is *used*, and one
of the hops in between never learned to carry it. Nothing failed loudly — the
layer that could not see the field answered ``legacy`` while the worker ran
``official``, on every request, with every log claiming ``official``.

The hops are enumerable, so the agreement between them can be a test rather than
a habit:

* ``config/stage_config.py`` — ``StageDeployConfig``, what a deploy YAML writes.
* ``config/omni_config.py`` — the diffusion-stage engine-override projection.
  A deploy field with no owner here is *rejected* by
  ``_validate_stage_engine_override_ownership``, so omitting it does not make
  the YAML ineffective, it makes it unusable.
* ``diffusion/data.py`` — ``OmniDiffusionConfig``, what the pipeline resolves
  the strategy from.
* ``engine/async_omni_engine.py`` — the cross-process view the *serving* layer
  sees when the diffusion stage runs in its own process. Omitted here, the
  front end answers for a contract the worker is not running.

Parsed rather than imported: this must run on a machine with no torch, and a
declaration is a syntactic fact.

The environment-variable path is not in this list, but only because it is
checked elsewhere and by a different means — an earlier version of this docstring
claimed it needed no hop at all, and that was wrong. A *process-scoped* variable
does reach every process without one. A *stage-scoped* one
(``stages[].runtime.env``) does not: it is applied while the stage starts and
restored immediately after, so it is a hop like any other, and the serving layer
never had it. That is carried as ``diffusion_runtime_environ`` and pinned by
``test_minimax_h3_stage_environment.py``; it is deliberately not
``minimax_h3_``-prefixed, because it is the transport for the contract rather
than a field of it.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_HOPS = (
    "config/stage_config.py",
    "config/omni_config.py",
    "diffusion/data.py",
    "engine/async_omni_engine.py",
)

_PREFIX = "minimax_h3_"


def _package_root() -> Path:
    import vllm_omni

    return Path(inspect.getfile(vllm_omni)).parent


def _carried_fields(relative_path: str) -> set[str]:
    """Contract field names this hop declares, as a field or as a string entry.

    Both spellings count because the hops are not the same kind of object: three
    declare dataclass fields, the cross-process view lists names in a tuple.
    """
    tree = ast.parse((_package_root() / relative_path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.startswith(_PREFIX):
                names.add(node.target.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith(_PREFIX):
                names.add(node.value)
    return names


def test_every_hop_carries_the_same_contract_fields():
    """One missing hop is a silently wrong instance, not a config error."""
    carried = {hop: _carried_fields(hop) for hop in _HOPS}
    expected = carried["config/stage_config.py"]
    mismatched = {hop: sorted(expected ^ fields) for hop, fields in carried.items() if fields != expected}
    assert not mismatched, (
        f"MiniMax-H3 contract fields differ across the config hops: {mismatched}. "
        f"StageDeployConfig declares {sorted(expected)}. A field that reaches some hops and not others "
        "makes one layer answer for a contract another layer is running — which is what "
        "reference_images_bind_output_canvas and honours_explicit_reference_order both hit."
    )


def test_the_fields_are_the_ones_resolve_strategy_actually_takes():
    """The hops must carry what the resolver reads, not merely agree with each other.

    Four consistent hops carrying the wrong two names would pass the test above
    and still resolve to ``legacy`` everywhere.
    """
    from vllm_omni.diffusion.models.minimax_h3.strategy import resolve_strategy

    parameters = set(inspect.signature(resolve_strategy).parameters)
    assert {name[len(_PREFIX) :] for name in _carried_fields("config/stage_config.py")} <= parameters


def test_the_probes_read_the_fields_off_a_config_by_those_names():
    """The consumers getattr() by literal name, so a rename has to reach them too."""
    root = _package_root()
    probes = (root / "diffusion/model_metadata.py").read_text(encoding="utf-8")
    for field in _carried_fields("diffusion/data.py"):
        assert f'"{field}"' in probes, f"{field} is carried to the pipeline but no capability probe reads it"
