# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference order: the bucketed canonicalization, and the orders it cannot express."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_bucket_canonicalization_reproduces_the_shipped_order():
    """Images, then videos (soundtrack label first), then standalone audio.

    Spelled out against the shipped loops rather than against the helper, so it
    pins behaviour: this is the order every existing ref2va request has used.
    """
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import (
        canonical_order_from_buckets,
        condition_labels,
    )

    ordered = canonical_order_from_buckets(num_images=2, video_has_audio=[True, False], num_audios=1)
    assert [(reference.kind, reference.bucket_index) for reference in ordered] == [
        ("image", 0),
        ("image", 1),
        ("video", 0),
        ("video", 1),
        ("audio", 0),
    ]
    assert condition_labels(ordered) == [
        ("image", 1),
        ("image", 2),
        ("audio", 1),  # the first video's soundtrack, before its own label
        ("video", 1),
        ("video", 2),  # the second video carries no sound
        ("audio", 2),  # the standalone clip
    ]


def test_labels_number_per_modality_not_per_position():
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import (
        MiniMaxH3OrderedReference,
        condition_labels,
    )

    # An interleave the bucketed entry cannot produce: video, image, audio.
    ordered = [
        MiniMaxH3OrderedReference("video", 0, has_audio=True),
        MiniMaxH3OrderedReference("image", 0),
        MiniMaxH3OrderedReference("audio", 0, has_audio=True),
    ]
    assert condition_labels(ordered) == [("audio", 1), ("video", 1), ("image", 1), ("audio", 2)]


def test_order_changes_the_labels_which_is_why_order_is_semantic():
    """The same three references in another order are a different prompt."""
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import (
        MiniMaxH3OrderedReference,
        condition_labels,
    )

    image = MiniMaxH3OrderedReference("image", 0)
    video = MiniMaxH3OrderedReference("video", 0, has_audio=True)
    audio = MiniMaxH3OrderedReference("audio", 0, has_audio=True)

    assert condition_labels([image, video, audio]) != condition_labels([video, image, audio])


def test_visual_and_audio_orders_are_the_encoder_feed_order():
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import (
        MiniMaxH3OrderedReference,
        audio_bearing_order,
        visual_order,
    )

    ordered = [
        MiniMaxH3OrderedReference("video", 0, has_audio=True),
        MiniMaxH3OrderedReference("image", 0),
        MiniMaxH3OrderedReference("audio", 0, has_audio=True),
    ]
    # Only image and video contribute video rows; both video and audio contribute audio rows.
    assert visual_order(ordered) == [0, 1]
    assert audio_bearing_order(ordered) == [0, 2]


def test_an_image_may_not_claim_a_soundtrack():
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import MiniMaxH3OrderedReference

    with pytest.raises(ValueError, match="soundtrack"):
        MiniMaxH3OrderedReference("image", 0, has_audio=True)
    with pytest.raises(ValueError, match="kind"):
        MiniMaxH3OrderedReference("caption", 0)


def test_describe_records_which_order_was_used():
    """A stored artifact has to say whether the order was requested or derived."""
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import (
        MINIMAX_H3_ORDER_BUCKETS,
        canonical_order_from_buckets,
        describe_order,
    )

    ordered = canonical_order_from_buckets(num_images=1, video_has_audio=[True], num_audios=0)
    described = describe_order(ordered, MINIMAX_H3_ORDER_BUCKETS)

    assert described["reference_order_mode"] == MINIMAX_H3_ORDER_BUCKETS
    assert described["labels"] == ["<Image 1>", "<Audio 1>", "<Video 1>"]
    with pytest.raises(ValueError, match="order mode"):
        describe_order(ordered, "official")


def test_requested_order_is_validated_against_the_media_that_arrived():
    """Order and media travel separately, so they can disagree; say so clearly."""
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import ordered_references_from_request

    ordered = ordered_references_from_request(
        [("video", 0), ("image", 0), ("audio", 0)],
        num_images=1,
        video_has_audio=[True],
        num_audios=1,
    )
    assert [(r.kind, r.bucket_index, r.has_audio) for r in ordered] == [
        ("video", 0, True),
        ("image", 0, False),
        ("audio", 0, True),
    ]

    # Points past what arrived.
    with pytest.raises(ValueError, match="carried 1 image"):
        ordered_references_from_request([("image", 3)], num_images=1, video_has_audio=[], num_audios=0)
    # Names the same reference twice.
    with pytest.raises(ValueError, match="repeats"):
        ordered_references_from_request([("image", 0), ("image", 0)], num_images=1, video_has_audio=[], num_audios=0)
    # Leaves an arrived reference unnamed, which would silently drop it.
    with pytest.raises(ValueError, match="names"):
        ordered_references_from_request([("image", 0)], num_images=2, video_has_audio=[], num_audios=0)
    with pytest.raises(ValueError, match="kind"):
        ordered_references_from_request([("caption", 0)], num_images=0, video_has_audio=[], num_audios=0)


def test_request_schema_carries_and_orders_references():
    from vllm_omni.entrypoints.openai.protocol.video_tasks import VideoTaskRequest

    request = VideoTaskRequest(
        prompt="p",
        references=[
            {"type": "video", "path": "/nfs/a.mp4"},
            {"type": "image", "path": "/nfs/b.png"},
            {"type": "audio", "path": "/nfs/c.wav"},
        ],
    )
    # Buckets keep within-modality order; the order rides alongside.
    assert request.reference_image_paths() == ["/nfs/b.png"]
    assert request.reference_video_paths() == ["/nfs/a.mp4"]
    assert request.reference_audio_paths() == ["/nfs/c.wav"]
    assert request.reference_order() == [("video", 0), ("image", 0), ("audio", 0)]
    # It survives into the generation request, which is what serving reads.
    assert request.to_video_request().reference_order == [
        {"type": "video", "index": 0},
        {"type": "image", "index": 0},
        {"type": "audio", "index": 0},
    ]


def test_bucketed_requests_are_untouched():
    """No `references` means exactly the old behaviour, order included."""
    from vllm_omni.entrypoints.openai.protocol.video_tasks import VideoTaskRequest

    request = VideoTaskRequest(prompt="p", image_path="/a.png,/b.png", video_path="/v.mp4")
    assert request.reference_image_paths() == ["/a.png", "/b.png"]
    assert request.reference_order() == []
    assert request.conflicting_reference_inputs() == []
    # Declared but unset: absent order means the canonicalization applies.
    assert request.to_video_request().reference_order is None


def test_mixing_ordered_and_bucketed_inputs_is_reported_not_resolved():
    """A precedence rule is what a caller misremembers; refuse instead."""
    from vllm_omni.entrypoints.openai.protocol.video_tasks import VideoTaskRequest

    request = VideoTaskRequest(
        prompt="p",
        references=[{"type": "image", "path": "/b.png"}],
        image_path="/a.png",
    )
    assert request.conflicting_reference_inputs() == ["image_path"]
