# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one ordered surface that used to drop its order: multipart uploads.

``/v1/videos`` takes ``input_references`` as an ordered list of files, and
``_persist_uploaded_media_references`` splits that list into image / video /
audio buckets. Splitting a list into buckets is exactly where an order stops
existing: under the official contract the pipeline then rebuilds a canonical
images-then-videos-then-audios order, so a caller who uploads *video, image*
gets the labels of *image, video* — a different prompt, silently, with a 200.

Two halves are pinned here. The helper has to *report* the order it destroyed,
and the route has to attach it only where the instance can honour one — a
legacy pipeline refuses an explicit order rather than ignoring it, so attaching
unconditionally would turn every mixed multipart request on the default
deployment into a 400.
"""

from __future__ import annotations

import asyncio
import io
import os
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Upload:
    """The three attributes `_persist_uploaded_media_references` actually reads."""

    def __init__(self, filename: str, content_type: str, payload: bytes):
        self.filename = filename
        self.content_type = content_type
        self.size = len(payload)
        self._stream = io.BytesIO(payload)

    async def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def _png(color=(11, 22, 33)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _image(name="a.png", color=(11, 22, 33)) -> _Upload:
    return _Upload(name, "image/png", _png(color))


def _video(name="a.mp4") -> _Upload:
    return _Upload(name, "video/mp4", b"not really an mp4, never decoded here")


def _audio(name="a.wav") -> _Upload:
    return _Upload(name, "audio/wav", b"not really a wav, never decoded here")


def _persist(uploads):
    from vllm_omni.entrypoints.openai.api_server import _persist_uploaded_media_references

    images, videos, audios, order = asyncio.run(_persist_uploaded_media_references(uploads))
    for path in list(videos) + list(audios):
        os.unlink(path)
    return images, videos, audios, order


# ------------------------------------------------ the helper reports the order


def test_the_upload_order_survives_the_split_into_buckets():
    images, videos, audios, order = _persist([_video("v0.mp4"), _image("i0.png"), _audio("a0.wav")])

    assert (len(images), len(videos), len(audios)) == (1, 1, 1)
    assert order == [("video", 0), ("image", 0), ("audio", 0)]


def test_the_reported_indices_are_positions_in_their_own_bucket():
    """The index is bucket-relative, because that is what the media travels as."""
    _, _, _, order = _persist(
        [
            _image("i0.png", (1, 2, 3)),
            _video("v0.mp4"),
            _image("i1.png", (4, 5, 6)),
            _video("v1.mp4"),
            _image("i2.png", (7, 8, 9)),
        ]
    )
    assert order == [("image", 0), ("video", 0), ("image", 1), ("video", 1), ("image", 2)]


def test_every_upload_is_accounted_for_exactly_once():
    uploads = [_audio("a0.wav"), _image("i0.png"), _audio("a1.wav"), _video("v0.mp4")]
    images, videos, audios, order = _persist(uploads)

    assert len(order) == len(uploads)
    counts = {"image": len(images), "video": len(videos), "audio": len(audios)}
    for kind, count in counts.items():
        indices = sorted(index for entry_kind, index in order if entry_kind == kind)
        assert indices == list(range(count))


def test_a_single_modality_upload_still_reports_its_order():
    """Not special-cased away: the canonical case has to keep working too."""
    _, _, _, order = _persist([_image("i0.png"), _image("i1.png")])
    assert order == [("image", 0), ("image", 1)]


def test_nothing_uploaded_reports_nothing():
    assert _persist([]) == ([], [], [], [])


# ------------------------------------------------ what the destroyed order cost


def test_the_bucket_rebuild_and_the_upload_order_disagree_on_the_labels():
    """Why this is a defect and not a spelling: the labels come out different."""
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import (
        MiniMaxH3OrderedReference,
        canonical_order_from_buckets,
        condition_labels,
    )

    _, _, _, order = _persist([_video("v0.mp4"), _image("i0.png")])
    assert order == [("video", 0), ("image", 0)]

    as_uploaded = [MiniMaxH3OrderedReference(kind, index) for kind, index in order]
    rebuilt = canonical_order_from_buckets(num_images=1, video_has_audio=[False], num_audios=0)

    assert condition_labels(as_uploaded) != condition_labels(rebuilt)
    assert condition_labels(as_uploaded) == [("video", 1), ("image", 1)]
    assert condition_labels(rebuilt) == [("image", 1), ("video", 1)]


def test_a_canonical_upload_order_is_the_bucket_rebuild_exactly():
    """So attaching it on a canonical request changes no number anywhere."""
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import (
        MiniMaxH3OrderedReference,
        canonical_order_from_buckets,
    )

    _, _, _, order = _persist([_image("i0.png"), _image("i1.png"), _video("v0.mp4"), _audio("a0.wav")])
    as_uploaded = [MiniMaxH3OrderedReference(kind, index) for kind, index in order]
    rebuilt = canonical_order_from_buckets(num_images=2, video_has_audio=[False], num_audios=1)

    assert [(r.kind, r.bucket_index) for r in as_uploaded] == [(r.kind, r.bucket_index) for r in rebuilt]


# ------------------------------------------------------- the route's attachment


def _handler(honours):
    """Only the capability probe matters; the rest of the handler never runs."""
    return SimpleNamespace(honours_explicit_reference_order=honours)


def test_an_official_instance_gets_the_upload_order_attached():
    from vllm_omni.entrypoints.openai.api_server import _multipart_reference_order

    attached = _multipart_reference_order(_handler(True), [("video", 0), ("image", 0)])
    assert [(entry.type, entry.index) for entry in attached] == [("video", 0), ("image", 0)]


def test_a_legacy_instance_gets_none_because_it_would_refuse_one():
    """Legacy raises on an explicit order, so this is the difference between a
    silently reordered request and a request that cannot be made at all."""
    from vllm_omni.entrypoints.openai.api_server import _multipart_reference_order

    assert _multipart_reference_order(_handler(False), [("video", 0), ("image", 0)]) is None


def test_an_empty_order_attaches_nothing_rather_than_an_empty_list():
    """An empty ``reference_order`` is a *requested* order of zero references,
    not the absence of one, and the validator would then reject the media."""
    from vllm_omni.entrypoints.openai.api_server import _multipart_reference_order

    assert _multipart_reference_order(_handler(True), []) is None


def test_a_handler_that_cannot_answer_is_treated_as_able():
    """Every model that reaches this branch supports mixed references, which no
    non-H3 pipeline does; a missing probe is a test double, not a deployment."""
    from vllm_omni.entrypoints.openai.api_server import _multipart_reference_order

    attached = _multipart_reference_order(SimpleNamespace(), [("image", 0)])
    assert [(entry.type, entry.index) for entry in attached] == [("image", 0)]


def test_the_attached_entries_are_what_the_request_schema_accepts():
    """Typed at the boundary, so a bad modality is a 400 and not a KeyError."""
    from vllm_omni.entrypoints.openai.api_server import _multipart_reference_order
    from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest

    order = _multipart_reference_order(_handler(True), [("audio", 0), ("image", 1)])
    request = VideoGenerationRequest(prompt="p", reference_order=order)
    assert [(entry.type, entry.index) for entry in request.reference_order] == [("audio", 0), ("image", 1)]


# ------------------------------------------ the one field that arrives past the split


def _completed(order, *, num_audios):
    from vllm_omni.entrypoints.openai.api_server import _order_with_separate_audio

    return _order_with_separate_audio(order, num_audios=num_audios)


def _entries(kinds_and_indices):
    from vllm_omni.entrypoints.openai.protocol.videos import ReferenceOrderEntry

    return [ReferenceOrderEntry(type=kind, index=index) for kind, index in kinds_and_indices]


def test_audio_supplied_beside_the_uploads_takes_its_place_in_the_order():
    """``audio_reference`` is the one media field ``input_references`` may join.

    It is deliberately outside the mutual-exclusion check, and its URLs are
    decoded into the audio bucket *after* the uploads — so an order derived from
    the uploads alone names fewer audio references than actually arrived. That is
    not a partial order the pipeline completes: the validator requires the order
    to name every reference, so a legal combination failed deep in the worker.
    """
    completed = _completed(_entries([("video", 0), ("image", 0)]), num_audios=2)
    assert [(entry.type, entry.index) for entry in completed] == [
        ("video", 0),
        ("image", 0),
        ("audio", 0),
        ("audio", 1),
    ]


def test_the_completed_order_is_one_the_validator_accepts():
    """The actual failure this prevents, at the layer that used to raise it."""
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import ordered_references_from_request

    uploads_only = _entries([("video", 0), ("image", 0)])
    buckets = {"num_images": 1, "video_has_audio": [False], "num_audios": 2}

    with pytest.raises(ValueError, match="2 audio reference"):
        ordered_references_from_request([(e.type, e.index) for e in uploads_only], **buckets)

    completed = _completed(uploads_only, num_audios=2)
    resolved = ordered_references_from_request([(e.type, e.index) for e in completed], **buckets)
    assert [(r.kind, r.bucket_index) for r in resolved] == [("video", 0), ("image", 0), ("audio", 0), ("audio", 1)]


def test_uploaded_audio_keeps_the_position_the_request_actually_stated():
    """Only the unnamed remainder is appended; a stated position is not moved."""
    completed = _completed(_entries([("audio", 0), ("image", 0)]), num_audios=3)
    assert [(entry.type, entry.index) for entry in completed] == [
        ("audio", 0),
        ("image", 0),
        ("audio", 1),
        ("audio", 2),
    ]


def test_an_order_that_already_names_every_audio_is_left_exactly_as_it_was():
    order = _entries([("audio", 1), ("image", 0), ("audio", 0)])
    assert _completed(order, num_audios=2) is order
    assert _completed(order, num_audios=0) is order


def test_no_order_stays_no_order():
    """A deployment that refuses an explicit order must not be handed one, and a
    request that never had an order has nothing to complete."""
    assert _completed(None, num_audios=2) is None
    assert _completed(None, num_audios=0) is None
