# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative config for co-located engines with exclusive GPU residency.

Some models do not fit on their hardware with every stage awake, yet fit
comfortably with one stage awake at a time. Those deploy as SEVERAL engines in
one server process, all loaded, at most one holding GPU weights — see
:mod:`vllm_omni.entrypoints.openai.stage_residency` for the coordinator and
``docs/实验报告/vLLM-Omni-HunyuanImage3-A100-NF4-实验与优化复盘.md`` §7.2/§11.1
for the measurements that motivate it.

This module only parses and validates the declaration. Example::

    mode: exclusive
    sleep_level: 1
    engines:
      - label: ar
        role: ar
        deploy_config: hunyuan_image3_ar_a100_40g.yaml
      - label: dit
        role: diffusion
        deploy_config: hunyuan_image3_dit_a100_40g.yaml

Why a separate file rather than a section inside a deploy YAML: a deploy config
describes ONE engine's stages, and these are independent engines that merely
share GPUs. Keeping the declaration separate also means a fleet of identical
instances references one baked artifact instead of each carrying a copy.

Relative ``deploy_config`` paths resolve against the residency file's own
directory, so a directory of configs can be baked into an image and moved as a
unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "ResidencyDeployment",
    "ResidencyEngineSpec",
    "load_residency_config",
]

# Only "one awake at a time" is implemented. Named rather than assumed so a
# future additive mode (e.g. "shared") is an explicit change, not a silent
# behavior drift for existing config files.
_SUPPORTED_MODES = ("exclusive",)

# Level 1 offloads weights to host RAM and supports a DMA restore; level 2
# discards them and AsyncOmni.wake_up() raises NotImplementedError afterwards,
# so a level-2 engine could never serve a second request.
_SUPPORTED_SLEEP_LEVEL = 1

# The role that produces the request's final output. Exactly one engine must
# carry it: it becomes the server's primary engine_client, so every existing
# endpoint keeps working against the engine that actually renders the result.
_TERMINAL_ROLE = "diffusion"
_SUPPORTED_ROLES = ("ar", _TERMINAL_ROLE)


@dataclass(frozen=True)
class ResidencyEngineSpec:
    """One engine in a residency group."""

    label: str
    role: str
    deploy_config: str

    @property
    def is_terminal(self) -> bool:
        return self.role == _TERMINAL_ROLE


@dataclass(frozen=True)
class ResidencyDeployment:
    """Parsed residency declaration."""

    mode: str
    sleep_level: int
    engines: tuple[ResidencyEngineSpec, ...]

    @property
    def terminal(self) -> ResidencyEngineSpec:
        """The engine whose output is the response (the server's primary)."""
        return next(spec for spec in self.engines if spec.is_terminal)

    def by_label(self, label: str) -> ResidencyEngineSpec:
        for spec in self.engines:
            if spec.label == label:
                return spec
        raise KeyError(f"unknown residency engine {label!r}; known: {[s.label for s in self.engines]}")

    def labels_by_role(self, role: str) -> list[str]:
        return [spec.label for spec in self.engines if spec.role == role]


def _require_mapping(raw: Any, path: Path) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: residency config must be a YAML mapping, got {type(raw).__name__}")
    return raw


def load_residency_config(path: str | Path) -> ResidencyDeployment:
    """Load and validate a residency declaration.

    Every failure is raised at load time (server startup) rather than tolerated:
    a residency group that is silently wrong does not fail until a request tries
    to wake a stage that is not there, by which point the model has spent
    minutes loading.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"residency config not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw = _require_mapping(yaml.safe_load(handle) or {}, config_path)

    mode = str(raw.get("mode", "exclusive"))
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"{config_path}: unsupported residency mode {mode!r}; supported: {list(_SUPPORTED_MODES)}")

    sleep_level = int(raw.get("sleep_level", _SUPPORTED_SLEEP_LEVEL))
    if sleep_level != _SUPPORTED_SLEEP_LEVEL:
        raise ValueError(
            f"{config_path}: sleep_level must be {_SUPPORTED_SLEEP_LEVEL}. Level 2 discards weights "
            "and cannot be woken, so the engine could not serve a second request."
        )

    raw_engines = raw.get("engines")
    if not isinstance(raw_engines, list) or not raw_engines:
        raise ValueError(f"{config_path}: 'engines' must be a non-empty list")

    engines: list[ResidencyEngineSpec] = []
    seen_labels: set[str] = set()
    for index, raw_engine in enumerate(raw_engines):
        if not isinstance(raw_engine, dict):
            raise ValueError(f"{config_path}: engines[{index}] must be a mapping")
        label = str(raw_engine.get("label") or "").strip()
        if not label:
            raise ValueError(f"{config_path}: engines[{index}] is missing 'label'")
        if label in seen_labels:
            raise ValueError(f"{config_path}: duplicate engine label {label!r}")
        seen_labels.add(label)

        role = str(raw_engine.get("role") or "").strip()
        if role not in _SUPPORTED_ROLES:
            raise ValueError(
                f"{config_path}: engines[{index}] ({label!r}) has role {role!r}; supported: {list(_SUPPORTED_ROLES)}"
            )

        deploy_config = str(raw_engine.get("deploy_config") or "").strip()
        if not deploy_config:
            raise ValueError(f"{config_path}: engines[{index}] ({label!r}) is missing 'deploy_config'")
        resolved = Path(deploy_config).expanduser()
        if not resolved.is_absolute():
            resolved = (config_path.parent / resolved).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{config_path}: engines[{index}] ({label!r}) deploy_config not found: {resolved}")

        engines.append(ResidencyEngineSpec(label=label, role=role, deploy_config=str(resolved)))

    terminal_labels = [spec.label for spec in engines if spec.is_terminal]
    if len(terminal_labels) != 1:
        raise ValueError(
            f"{config_path}: exactly one engine must have role {_TERMINAL_ROLE!r} "
            f"(it becomes the server's primary engine); found {terminal_labels or 'none'}"
        )

    deployment = ResidencyDeployment(mode=mode, sleep_level=sleep_level, engines=tuple(engines))
    logger.info(
        "Loaded residency config %s: mode=%s level=%d engines=%s (primary=%s)",
        config_path,
        mode,
        sleep_level,
        [f"{s.label}:{s.role}" for s in engines],
        deployment.terminal.label,
    )
    return deployment
