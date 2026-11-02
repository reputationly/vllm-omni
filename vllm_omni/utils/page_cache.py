# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in file page-cache dropping via ``posix_fadvise(POSIX_FADV_DONTNEED)``.

Bounds host page-cache growth from per-request inputs/outputs that will not be
re-read, and from model weight shards whose clean pages linger after load.
Only clean pages of the named files are affected; process memory is untouched.
"""

import os
from collections.abc import Iterable

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENV_REQUEST_DROP = "VLLM_OMNI_DROP_REQUEST_FILE_CACHE"
_ENV_WEIGHT_DROP = "VLLM_OMNI_DROP_WEIGHT_FILE_CACHE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
# Linux value; the os constant only exists where posix_fadvise does.
_POSIX_FADV_DONTNEED: int = getattr(os, "POSIX_FADV_DONTNEED", 4)

_no_fadvise_warned = False


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def request_file_cache_drop_enabled() -> bool:
    """Whether per-request input/output file pages should be dropped."""
    return _env_truthy(_ENV_REQUEST_DROP)


def weight_file_cache_drop_enabled() -> bool:
    """Whether weight shard pages should be dropped after load."""
    return _env_truthy(_ENV_WEIGHT_DROP)


def drop_file_page_cache(paths: Iterable[str | os.PathLike]) -> int:
    """Advise the kernel to drop page-cache pages for each given file.

    Returns the number of files advised. Missing paths, non-regular files, and
    duplicates are skipped. Per-file ``OSError`` is logged at debug level and
    never propagates. On platforms without ``os.posix_fadvise`` (non-Linux)
    this is a no-op returning 0.
    """
    global _no_fadvise_warned
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is None:
        if not _no_fadvise_warned:
            logger.debug("os.posix_fadvise unavailable; file page-cache drop disabled")
            _no_fadvise_warned = True
        return 0

    advised = 0
    seen: set[str] = set()
    for p in paths:
        path = os.fspath(p)
        if path in seen:
            continue
        seen.add(path)
        try:
            if not os.path.isfile(path):
                continue
        except OSError:
            continue
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as exc:
            logger.debug("posix_fadvise skip %s: %s", path, exc)
            continue
        try:
            fadvise(fd, 0, 0, _POSIX_FADV_DONTNEED)
            advised += 1
        except OSError as exc:
            logger.debug("posix_fadvise failed for %s: %s", path, exc)
        finally:
            os.close(fd)
    return advised
