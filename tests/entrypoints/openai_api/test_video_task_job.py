# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``_run_video_task_job``（``POST /v1/tasks/video/`` 的后台任务）对
``generate_video_bytes`` 返回形状的兼容性。

上游 #4728 让 ``generate_video_bytes`` 多回一个 ``video_metadata``（5 元组），这条 fork
独有的任务路径还按 4 个解包：视频生成完之后抛 ``too many values to unpack (expected 4)``，
任务报失败、结果被丢掉。这里钉住 4 元组和 5 元组都能落盘并标成 COMPLETED。
"""

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.protocol.audio_tasks import AudioTaskResponse, AudioTaskStatus
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest
from vllm_omni.entrypoints.openai.stores import AUDIO_TASK_STORE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeHandler:
    def __init__(self, result):
        self._result = result

    async def generate_video_bytes(self, request, reference_id, **_kwargs):
        del request, reference_id
        return self._result


@pytest.mark.parametrize(
    "result",
    [
        (b"mp4-bytes", {"diffusion": 1.0}, 0.0, None),
        (b"mp4-bytes", {"diffusion": 1.0}, 0.0, None, {"fps": 24}),
    ],
    ids=["four-tuple", "five-tuple-with-metadata"],
)
def test_video_task_job_persists_output_for_both_result_shapes(tmp_path, result):
    task_id = f"video-task-{len(result)}"
    out = tmp_path / "out.mp4"

    async def run():
        await AUDIO_TASK_STORE.upsert(task_id, AudioTaskResponse(task_id=task_id, status=AudioTaskStatus.PENDING))
        await api_server._run_video_task_job(
            _FakeHandler(result),
            VideoGenerationRequest(prompt="a cat stretching"),
            task_id,
            str(out),
            image_paths=[],
            video_paths=[],
            audio_paths=[],
            allowed_root=str(tmp_path),
            app_state=SimpleNamespace(server=None, engine_client=None),
        )
        return await AUDIO_TASK_STORE.get(task_id)

    job = asyncio.run(run())
    assert job is not None
    assert job.status is AudioTaskStatus.COMPLETED, job.error
    assert job.error is None
    assert out.read_bytes() == b"mp4-bytes"
