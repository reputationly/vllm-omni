# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where ``resolve_model_config_path`` looks first, on every platform.

Resolving a model's deploy YAML is a CPU-side, pre-device concern — it runs
before anything is placed on an accelerator, and it is also what config
validation and CPU-only test containers call. It used to go through
``OmniPlatform.get_default_stage_config_path()``, which the base class left as
a bare ``NotImplementedError``; under ``UnspecifiedOmniPlatform`` (no
accelerator visible) that turned every such call into a crash with no message.

The base answer is now the directory the resolver already falls through to, so
these pin two things: that the default exists, and that it is not a *third*
answer invented alongside the per-platform ones.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_the_base_platform_answers_without_an_accelerator():
    from vllm_omni.platforms.interface import OmniPlatform, UnspecifiedOmniPlatform

    assert OmniPlatform.get_default_stage_config_path() == "vllm_omni/deploy"
    assert UnspecifiedOmniPlatform.get_default_stage_config_path() == "vllm_omni/deploy"


def test_the_default_is_the_directory_the_resolver_already_falls_through_to():
    """Not a new answer: ``_DEPLOY_DIR`` is the resolver's own last resort.

    If these two ever diverge, the base default stops being a no-op and starts
    being a policy, which is exactly what this file exists to prevent.
    """
    from vllm_omni.config.stage_config import _DEPLOY_DIR
    from vllm_omni.entrypoints.utils import PROJECT_ROOT
    from vllm_omni.platforms.interface import OmniPlatform

    assert PROJECT_ROOT / OmniPlatform.get_default_stage_config_path() == _DEPLOY_DIR


def test_every_shipped_platform_still_states_its_own():
    """The default must not become a reason to stop declaring it.

    XPU ships a different directory; the rest agree with the base. Both facts
    are load-bearing, so neither is left to inheritance by accident.

    Read as source, not imported: ``vllm_omni.platforms.cuda`` pulls in
    ``vllm.platforms.cuda``, which is only importable where that accelerator
    exists — and the whole point of this file is the machine where none does.
    """
    import ast
    from pathlib import Path

    root = Path(__import__("vllm_omni").__file__).parent / "platforms"
    expected = {
        "cuda": "vllm_omni/deploy",
        "rocm": "vllm_omni/deploy",
        "npu": "vllm_omni/deploy",
        "musa": "vllm_omni/deploy",
        "xpu": "vllm_omni/platforms/xpu/stage_configs",
    }

    for name, directory in expected.items():
        tree = ast.parse((root / name / "platform.py").read_text(encoding="utf-8"))
        returned = [
            node.body[-1].value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_default_stage_config_path"
            and isinstance(node.body[-1], ast.Return)
            and isinstance(node.body[-1].value, ast.Constant)
        ]
        assert returned == [directory], f"{name} platform: expected {directory!r}, found {returned!r}"


def test_resolution_survives_an_unspecified_platform(tmp_path, monkeypatch):
    """The call site the missing default used to break, forced on any machine.

    MiniMax-H3 is the live example: it is not an ``OMNI_PIPELINES`` key and
    ships no ``MiniMaxH3ModularPipeline.yaml``, so the honest answer is
    ``None`` — which is what lets the caller fall back deliberately instead of
    dying inside a platform probe. The platform is pinned here rather than
    inherited from the host, so a CUDA box tests the same path a CPU-only
    container takes.
    """
    import json

    from vllm_omni.entrypoints import utils
    from vllm_omni.platforms.interface import UnspecifiedOmniPlatform

    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline", "_diffusers_version": "0.36.0.dev0"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(utils, "current_omni_platform", UnspecifiedOmniPlatform())

    assert utils.resolve_model_config_path(str(tmp_path)) is None
