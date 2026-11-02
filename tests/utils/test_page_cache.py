# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for ``vllm_omni.utils.page_cache``."""

import os
from unittest.mock import MagicMock

import pytest

from vllm_omni.utils import page_cache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_drop_calls_fadvise_once_per_file(tmp_path, monkeypatch):
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"x")
    f2.write_bytes(b"y")
    fadvise = MagicMock()
    monkeypatch.setattr(os, "posix_fadvise", fadvise)

    advised = page_cache.drop_file_page_cache([str(f1), str(f2)])

    assert advised == 2
    assert fadvise.call_count == 2
    for call in fadvise.call_args_list:
        assert call.args[3] == page_cache._POSIX_FADV_DONTNEED


def test_drop_skips_missing_dirs_and_dedupes(tmp_path, monkeypatch):
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"x")
    subdir = tmp_path / "d"
    subdir.mkdir()
    fadvise = MagicMock()
    monkeypatch.setattr(os, "posix_fadvise", fadvise)

    advised = page_cache.drop_file_page_cache([str(f1), str(tmp_path / "missing.bin"), str(subdir), str(f1)])

    assert advised == 1
    assert fadvise.call_count == 1


def test_drop_continues_after_oserror(tmp_path, monkeypatch):
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"x")
    f2.write_bytes(b"y")
    calls = []

    def flaky(fd, off, length, advice):
        calls.append(fd)
        if len(calls) == 1:
            raise OSError("boom")

    monkeypatch.setattr(os, "posix_fadvise", flaky)

    advised = page_cache.drop_file_page_cache([str(f1), str(f2)])

    assert advised == 1
    assert len(calls) == 2


def test_drop_returns_zero_without_posix_fadvise(tmp_path, monkeypatch):
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"x")
    monkeypatch.delattr(os, "posix_fadvise", raising=False)

    assert page_cache.drop_file_page_cache([str(f1)]) == 0


@pytest.mark.parametrize(
    "name",
    ["request_file_cache_drop_enabled", "weight_file_cache_drop_enabled"],
)
@pytest.mark.parametrize("value", ["1", "TRUE", "yes", "On"])
def test_predicates_truthy(monkeypatch, name, value):
    env = (
        "VLLM_OMNI_DROP_REQUEST_FILE_CACHE"
        if name == "request_file_cache_drop_enabled"
        else "VLLM_OMNI_DROP_WEIGHT_FILE_CACHE"
    )
    monkeypatch.setenv(env, value)
    assert getattr(page_cache, name)() is True


@pytest.mark.parametrize(
    "name",
    ["request_file_cache_drop_enabled", "weight_file_cache_drop_enabled"],
)
@pytest.mark.parametrize("value", ["0", "false", "off", ""])
def test_predicates_falsy(monkeypatch, name, value):
    env = (
        "VLLM_OMNI_DROP_REQUEST_FILE_CACHE"
        if name == "request_file_cache_drop_enabled"
        else "VLLM_OMNI_DROP_WEIGHT_FILE_CACHE"
    )
    monkeypatch.setenv(env, value)
    assert getattr(page_cache, name)() is False


def test_predicates_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_DROP_REQUEST_FILE_CACHE", raising=False)
    monkeypatch.delenv("VLLM_OMNI_DROP_WEIGHT_FILE_CACHE", raising=False)
    assert page_cache.request_file_cache_drop_enabled() is False
    assert page_cache.weight_file_cache_drop_enabled() is False
