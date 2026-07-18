# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FIFO task manager + backpressure for the async audio task API.

The generic in-memory store/registry (``stores.py``) has no queue bound; the
GPUStack async contract requires FIFO admission with 503 backpressure. This
mirrors LightX2V ``server/task_manager.py``: admission raises ``RuntimeError``
when the queue is full (the route turns that into HTTP 503).

Task state lives in ``AUDIO_TASK_STORE``; the running ``asyncio.Task`` lives in
``AUDIO_TASKS``. This manager only owns admission control, queue accounting,
and cancellation — the background job itself is started by the route (which has
the speech handler and app state).
"""

import asyncio
import os
import time
from pathlib import Path

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio_tasks import (
    AudioTaskResponse,
    AudioTaskStatus,
)
from vllm_omni.entrypoints.openai.stores import AUDIO_TASK_STORE, AUDIO_TASKS

logger = init_logger(__name__)

_TERMINAL = (
    AudioTaskStatus.COMPLETED,
    AudioTaskStatus.FAILED,
    AudioTaskStatus.CANCELLED,
)
_ACTIVE = (AudioTaskStatus.PENDING, AudioTaskStatus.PROCESSING)


def resolve_save_path(
    save_result_path: str, task_id: str, output_root: str, default_ext: str = ".wav"
) -> str:
    """Resolve the output path (semantics per LightX2V ``file_service``).

    Absolute -> used verbatim; relative -> under ``output_root``; empty ->
    ``task_id``; missing suffix -> ``default_ext`` appended.
    """
    raw = save_result_path or task_id
    path = Path(raw)
    if not path.is_absolute():
        path = Path(output_root) / raw
    if not path.suffix:
        path = path.with_suffix(default_ext)
    return str(path)


class AudioTaskManager:
    def __init__(self, max_queue_size: int = 8) -> None:
        self._store = AUDIO_TASK_STORE
        self._tasks = AUDIO_TASKS
        self._lock = asyncio.Lock()
        self._max = max_queue_size

    async def reserve(self, task_id: str, save_result_path: str) -> AudioTaskResponse:
        """Admit a task: reject duplicates and enforce the queue bound, then
        create the PENDING record. Raises ``RuntimeError`` when full or on a
        duplicate id (the route maps these to HTTP 503)."""
        async with self._lock:
            if await self._store.get(task_id) is not None:
                raise RuntimeError(f"Task ID {task_id} already exists")
            items = await self._store.list_values()
            active = sum(1 for t in items if t.status in _ACTIVE)
            if active >= self._max:
                raise RuntimeError(f"Task queue is full (max {self._max} tasks)")
            resp = AudioTaskResponse(
                task_id=task_id,
                status=AudioTaskStatus.PENDING,
                save_result_path=save_result_path,
            )
            await self._store.upsert(task_id, resp)
            return resp

    async def get_status(self, task_id: str) -> AudioTaskResponse | None:
        return await self._store.get(task_id)

    async def list_tasks(self) -> list[AudioTaskResponse]:
        return await self._store.list_values()

    async def queue_status(self) -> dict:
        items = await self._store.list_values()
        pending = sum(1 for t in items if t.status == AudioTaskStatus.PENDING)
        processing = [t for t in items if t.status == AudioTaskStatus.PROCESSING]
        active = pending + len(processing)
        return {
            "is_processing": len(processing) > 0,
            "current_task": processing[0].task_id if processing else None,
            "pending_count": pending,
            "active_count": active,
            "queue_size": self._max,
            "queue_available": max(0, self._max - active),
        }

    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending/processing task. Returns False if unknown or already
        terminal."""
        job = await self._store.get(task_id)
        if job is None or job.status in _TERMINAL:
            return False
        task = await self._tasks.get(task_id)
        if task is not None:
            task.cancel()
        await self._store.update_fields(
            task_id, {"status": AudioTaskStatus.CANCELLED, "end_time": time.time()}
        )
        return True


AUDIO_TASK_MANAGER = AudioTaskManager(
    max_queue_size=int(os.environ.get("VLLM_OMNI_AUDIO_MAX_QUEUE", "8"))
)
