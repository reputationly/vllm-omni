# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the weight page-cache drop wrapper in diffusers_loader."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.diffusion.model_loader import diffusers_loader

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _iter(items):
    yield from items


def test_wrapper_drops_after_full_exhaustion(tmp_path, monkeypatch):
    f1 = tmp_path / "w1.safetensors"
    f2 = tmp_path / "w2.safetensors"
    f1.write_bytes(b"a" * 100)
    f2.write_bytes(b"b" * 200)
    drop = MagicMock(return_value=2)
    monkeypatch.setattr(diffusers_loader, "drop_file_page_cache", drop)

    items = [("a", torch.zeros(1)), ("b", torch.zeros(1))]
    gen = diffusers_loader._drop_cache_after_exhaustion(_iter(items), [str(f1), str(f2)])

    drop.assert_not_called()
    out = list(gen)
    assert len(out) == 2
    drop.assert_called_once_with([str(f1), str(f2)])


def test_wrapper_no_drop_on_early_close(tmp_path, monkeypatch):
    f1 = tmp_path / "w1.safetensors"
    f1.write_bytes(b"a")
    drop = MagicMock()
    monkeypatch.setattr(diffusers_loader, "drop_file_page_cache", drop)

    items = [("a", torch.zeros(1)), ("b", torch.zeros(1)), ("c", torch.zeros(1))]
    gen = diffusers_loader._drop_cache_after_exhaustion(_iter(items), [str(f1)])
    next(gen)
    gen.close()
    drop.assert_not_called()


def test_wrapper_gate_uses_env_predicate(monkeypatch):
    """The wrap in _get_weights_iterator is gated by the env-read predicate."""
    monkeypatch.delenv("VLLM_OMNI_DROP_WEIGHT_FILE_CACHE", raising=False)
    assert diffusers_loader.weight_file_cache_drop_enabled() is False
    monkeypatch.setenv("VLLM_OMNI_DROP_WEIGHT_FILE_CACHE", "1")
    assert diffusers_loader.weight_file_cache_drop_enabled() is True
