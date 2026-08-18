# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Putting rank 0's prepared references on every rank, without pickling pixels.

The native tiled video VAE encodes with collectives, so when VAE patch
parallelism is on every rank has to enter each reference encode with the same
input. ``broadcast_object_list`` was the right tool while a prepared reference
was a path plus a few scalars; the official decode path carries the normalized
frames in memory instead, and pickling those turns a few hundred bytes into up
to ~1.1 GB — 15 s at 24 fps on a 1344x768 canvas — materialized twice per rank,
once as the serialized byte tensor on the device and once as the unpickled array
on the host.

These pin the split: metadata on the object list, pixels on a chunked ``uint8``
broadcast. The collectives are faked, so this runs on one CPU process — what is
being tested is the *payload*, which is where the regression was.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeCollectives:
    """A two-rank broadcast, replayed inside one process.

    Rank 0 runs first and records every payload; the receiver runs second and is
    handed them back in order. Recording rather than sharing state is what lets
    the assertions look at exactly what would have gone over the wire.
    """

    def __init__(self):
        self.objects: list = []
        self.tensors: list[torch.Tensor] = []
        self.replay = False
        self._object_cursor = 0
        self._tensor_cursor = 0

    def broadcast_object_list(self, box, src=0, group=None, device=None):
        if not self.replay:
            self.objects.append(box[0])
            return
        box[0] = self.objects[self._object_cursor]
        self._object_cursor += 1

    def broadcast(self, tensor, src=0, group=None):
        if not self.replay:
            self.tensors.append(tensor.clone())
            return
        tensor.copy_(self.tensors[self._tensor_cursor])
        self._tensor_cursor += 1

    def rewind(self):
        self.replay = True
        self._object_cursor = 0
        self._tensor_cursor = 0


class _Pipeline:
    """Only what ``_broadcast_prepared_videos`` reads off the pipeline."""

    device = torch.device("cpu")

    def __init__(self):
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

        self._broadcast = MiniMaxH3Pipeline._broadcast_prepared_videos

    def run(self, prepared_videos, *, rank, chunk_frames=32):
        return self._broadcast(self, prepared_videos, group=None, rank=rank, chunk_frames=chunk_frames)


def _prepared(num_frames: int, height: int = 6, width: int = 8, *, with_audio: bool = True) -> dict:
    rng = np.random.default_rng(num_frames)
    return {
        "original_path": "/tmp/ref.mp4",
        "prepared_path": None,
        "frames": rng.integers(0, 256, size=(num_frames, height, width, 3), dtype=np.uint8),
        "audio": torch.zeros(2, 32000) if with_audio else None,
        "audio_sample_rate": 32000 if with_audio else None,
        "input_has_audio": with_audio,
        "width": width,
        "height": height,
        "start_time_seconds": 0.0,
        "duration_seconds": num_frames / 24.0,
    }


@pytest.fixture
def collectives(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3

    fake = _FakeCollectives()
    monkeypatch.setattr(pipeline_minimax_h3.dist, "broadcast_object_list", fake.broadcast_object_list)
    monkeypatch.setattr(pipeline_minimax_h3.dist, "broadcast", fake.broadcast)
    return fake


def test_the_frames_arrive_intact_on_a_receiving_rank(collectives):
    sources = [_prepared(40), _prepared(23, height=4, width=4)]

    sent_back = _Pipeline().run(sources, rank=0)
    collectives.rewind()
    received = _Pipeline().run(None, rank=1)

    # Rank 0 keeps its own dicts, audio included — the reconstruction is for
    # the other ranks, and handing it back here would drop the soundtrack that
    # only rank 0 goes on to encode.
    assert all(kept is source for kept, source in zip(sent_back, sources, strict=True))
    assert sent_back[0]["audio"] is not None

    assert len(received) == len(sources)
    for source, item in zip(sources, received, strict=True):
        assert np.array_equal(item["frames"], source["frames"])
        assert item["frames"].dtype == np.uint8
        assert item["prepared_path"] == source["prepared_path"]
        assert item["duration_seconds"] == source["duration_seconds"]
        # Only rank 0 encodes reference soundtracks, so paying to ship them
        # would be paying for something no other rank reads.
        assert item["audio"] is None


def test_no_pixels_ride_the_object_list(collectives):
    """The regression itself: this payload used to be the whole reference.

    Stated as a type invariant rather than a serialized byte count: what makes
    the payload cheap is that every value is a scalar the collective can
    pickle in constant time, and that is also the property a future field
    would have to violate to bring the pixels back.
    """
    sources = [_prepared(120, height=48, width=64)]
    _Pipeline().run(sources, rank=0)

    (payload,) = collectives.objects
    for item in payload:
        assert "frames" not in item
        assert "audio" not in item
        assert item["frames_shape"] == (120, 48, 64, 3)
        for key, value in item.items():
            if key == "frames_shape":
                continue
            assert isinstance(value, str | int | float | bool | type(None)), (
                f"{key!r} is a {type(value).__name__}; prepared-reference metadata should be metadata"
            )


def test_device_residency_is_bounded_by_the_chunk_not_by_the_reference(collectives):
    """A longer reference must not mean a larger allocation, only more of them."""
    chunk = 8
    short, long = _prepared(chunk), _prepared(10 * chunk)

    _Pipeline().run([short], rank=0, chunk_frames=chunk)
    short_peak = max(tensor.numel() for tensor in collectives.tensors)

    collectives.tensors.clear()
    _Pipeline().run([long], rank=0, chunk_frames=chunk)
    long_peak = max(tensor.numel() for tensor in collectives.tensors)

    assert short_peak == long_peak
    assert long_peak == chunk * 6 * 8 * 3
    assert len(collectives.tensors) == 10


def test_a_reference_shorter_than_one_chunk_moves_in_one_go(collectives):
    source = _prepared(5)

    _Pipeline().run([source], rank=0, chunk_frames=32)

    assert len(collectives.tensors) == 1
    assert collectives.tensors[0].shape[0] == 5


def test_a_chunk_boundary_that_does_not_divide_still_reconstructs(collectives):
    source = _prepared(37)

    _Pipeline().run([source], rank=0, chunk_frames=16)
    collectives.rewind()
    (received,) = _Pipeline().run(None, rank=1, chunk_frames=16)

    assert [tensor.shape[0] for tensor in collectives.tensors] == [16, 16, 5]
    assert np.array_equal(received["frames"], source["frames"])


def test_no_references_broadcasts_nothing_but_the_absence(collectives):
    assert _Pipeline().run(None, rank=0) is None
    collectives.rewind()
    assert _Pipeline().run(None, rank=1) is None
    assert collectives.tensors == []
