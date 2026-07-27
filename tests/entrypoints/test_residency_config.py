# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the residency declaration parser (CPU-only).

Every failure here must surface at server startup: a residency group that is
silently wrong does not fail until a request tries to wake an engine that is not
there, by which point the model has spent minutes loading.
"""

import textwrap

import pytest

from vllm_omni.entrypoints.openai.residency_config import (
    ResidencyDeployment,
    load_residency_config,
)


def _write(tmp_path, body: str, *, deploy_names=("ar.yaml", "dit.yaml")) -> str:
    for name in deploy_names:
        (tmp_path / name).write_text("stages: []\n", encoding="utf-8")
    path = tmp_path / "residency.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


_VALID = """
    mode: exclusive
    sleep_level: 1
    engines:
      - label: ar
        role: ar
        deploy_config: ar.yaml
      - label: dit
        role: diffusion
        deploy_config: dit.yaml
"""


def test_valid_config(tmp_path):
    cfg = load_residency_config(_write(tmp_path, _VALID))
    assert isinstance(cfg, ResidencyDeployment)
    assert cfg.mode == "exclusive"
    assert cfg.sleep_level == 1
    assert [e.label for e in cfg.engines] == ["ar", "dit"]
    assert cfg.terminal.label == "dit"
    assert cfg.labels_by_role("ar") == ["ar"]
    assert cfg.by_label("dit").is_terminal


def test_relative_deploy_paths_resolve_against_the_residency_file(tmp_path):
    """A directory of configs must be bakeable into an image and movable."""
    from pathlib import Path

    cfg = load_residency_config(_write(tmp_path, _VALID))
    # Compare resolved paths: on macOS the temp dir is itself a symlink
    # (/var -> /private/var), so raw string prefixes do not match.
    expected_dir = tmp_path.resolve()
    for engine in cfg.engines:
        resolved = Path(engine.deploy_config)
        assert resolved.parent == expected_dir
        assert resolved.is_file()


def test_defaults_when_omitted(tmp_path):
    cfg = load_residency_config(
        _write(
            tmp_path,
            """
            engines:
              - {label: dit, role: diffusion, deploy_config: dit.yaml}
            """,
        )
    )
    assert cfg.mode == "exclusive"
    assert cfg.sleep_level == 1


def test_single_terminal_engine_is_allowed(tmp_path):
    """A degenerate group of one is valid; it just has nothing to alternate with."""
    cfg = load_residency_config(
        _write(
            tmp_path,
            """
            engines:
              - {label: dit, role: diffusion, deploy_config: dit.yaml}
            """,
        )
    )
    assert cfg.terminal.label == "dit"


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="residency config not found"):
        load_residency_config(str(tmp_path / "nope.yaml"))


def test_missing_deploy_config_is_caught_at_load(tmp_path):
    path = tmp_path / "residency.yaml"
    path.write_text(
        textwrap.dedent(
            """
            engines:
              - {label: dit, role: diffusion, deploy_config: absent.yaml}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="deploy_config not found"):
        load_residency_config(str(path))


def test_sleep_level_2_is_rejected(tmp_path):
    """Level 2 discards weights and wake_up() is unimplemented after it, so the
    engine could never serve a second request."""
    body = _VALID.replace("sleep_level: 1", "sleep_level: 2")
    with pytest.raises(ValueError, match="sleep_level must be 1"):
        load_residency_config(_write(tmp_path, body))


def test_unsupported_mode(tmp_path):
    body = _VALID.replace("mode: exclusive", "mode: shared")
    with pytest.raises(ValueError, match="unsupported residency mode"):
        load_residency_config(_write(tmp_path, body))


@pytest.mark.parametrize(
    "body,match",
    [
        ("engines: []\n", "must be a non-empty list"),
        ("engines: notalist\n", "must be a non-empty list"),
        ("{}\n", "must be a non-empty list"),
    ],
)
def test_engines_must_be_a_non_empty_list(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        load_residency_config(_write(tmp_path, body))


def test_duplicate_labels(tmp_path):
    body = """
        engines:
          - {label: dup, role: ar, deploy_config: ar.yaml}
          - {label: dup, role: diffusion, deploy_config: dit.yaml}
    """
    with pytest.raises(ValueError, match="duplicate engine label"):
        load_residency_config(_write(tmp_path, body))


def test_missing_label(tmp_path):
    body = """
        engines:
          - {role: diffusion, deploy_config: dit.yaml}
    """
    with pytest.raises(ValueError, match="missing 'label'"):
        load_residency_config(_write(tmp_path, body))


def test_unknown_role(tmp_path):
    body = """
        engines:
          - {label: vae, role: decoder, deploy_config: dit.yaml}
    """
    with pytest.raises(ValueError, match="supported: \\['ar', 'diffusion'\\]"):
        load_residency_config(_write(tmp_path, body))


def test_missing_deploy_config_key(tmp_path):
    body = """
        engines:
          - {label: dit, role: diffusion}
    """
    with pytest.raises(ValueError, match="missing 'deploy_config'"):
        load_residency_config(_write(tmp_path, body))


@pytest.mark.parametrize(
    "roles,match",
    [
        (["ar", "ar"], "exactly one engine must have role"),
        (["diffusion", "diffusion"], "exactly one engine must have role"),
    ],
)
def test_terminal_engine_count(tmp_path, roles, match):
    lines = "\n".join(f"      - {{label: e{i}, role: {role}, deploy_config: ar.yaml}}" for i, role in enumerate(roles))
    with pytest.raises(ValueError, match=match):
        load_residency_config(_write(tmp_path, f"engines:\n{lines}\n"))


def test_by_label_unknown(tmp_path):
    cfg = load_residency_config(_write(tmp_path, _VALID))
    with pytest.raises(KeyError):
        cfg.by_label("nope")


def test_shipped_hunyuan_image3_config_is_valid():
    """The config baked into the image must parse, or every instance fails boot."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    shipped = repo_root / "deploy-configs" / "hunyuan_image3_a100_40g_residency.yaml"
    if not shipped.is_file():  # pragma: no cover - defensive
        pytest.skip(f"{shipped} not present")
    cfg = load_residency_config(shipped)
    assert [e.label for e in cfg.engines] == ["ar", "dit"]
    assert cfg.terminal.label == "dit"
    assert cfg.sleep_level == 1
