# SPDX-License-Identifier: Apache-2.0
"""The order a `ref2va` request reads its references in.

For MiniMax-H3 the order is semantic three times over: it numbers the
``"<Picture i>"`` / ``"<Audio j>"`` / ``"<Video k>"`` labels of the prompt, it
fixes the order the request generator is consumed in, and it advances the shared
audio/video rotary clock. A different order is a different request, not a
different spelling of one.

The official interface takes an ordered heterogeneous list. vLLM-Omni's public
entry takes three modality buckets — ``multi_modal_data.image / video / audio``
— which cannot express an interleave such as *video, image, audio*, so the
pipeline has always rebuilt one fixed order from them. That rebuild is real
behaviour and stays available, but it is spelled out here as a named
canonicalization instead of being implied by the order of three loops, so a run
can say which order it used and a strict-official request can supply its own.

Both orders put a video's soundtrack label immediately before the video's own,
mirroring how its rows are packed; that is the oracle's rule, not a choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

MINIMAX_H3_ORDER_REQUEST = "ordered_references"
MINIMAX_H3_ORDER_BUCKETS = "legacy_bucket_canonicalization"
MINIMAX_H3_ORDER_MODES = frozenset({MINIMAX_H3_ORDER_REQUEST, MINIMAX_H3_ORDER_BUCKETS})

_KINDS = ("image", "video", "audio")
_PROMPT_KIND_NAMES = {"image": "Picture", "video": "Video", "audio": "Audio"}


@dataclass(frozen=True)
class MiniMaxH3OrderedReference:
    """One reference, and where its media sits in the request's buckets.

    Attributes:
        kind: ``image``, ``video`` or ``audio``.
        bucket_index: Position within that modality's bucket, 0-based. Keeping
            the bucket index rather than the media itself lets the order be
            resolved and tested before anything is decoded.
        has_audio: Whether this reference carries a soundtrack. Only a video
            can; an audio reference is one by definition.
    """

    kind: str
    bucket_index: int
    has_audio: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"reference kind must be one of {_KINDS}, got {self.kind!r}")
        if self.bucket_index < 0:
            raise ValueError(f"bucket_index must be non-negative, got {self.bucket_index}")
        if self.has_audio and self.kind == "image":
            raise ValueError("an image reference carries no soundtrack")


def canonical_order_from_buckets(
    *,
    num_images: int,
    video_has_audio: Sequence[bool],
    num_audios: int,
) -> list[MiniMaxH3OrderedReference]:
    """The order the bucketed entry has always produced: images, videos, audios.

    Args:
        num_images: Image references in the request.
        video_has_audio: One flag per video reference, in bucket order.
        num_audios: Standalone audio references.

    Returns:
        The references in packed order.
    """
    if num_images < 0 or num_audios < 0:
        raise ValueError("reference counts must be non-negative")
    ordered = [MiniMaxH3OrderedReference("image", index) for index in range(num_images)]
    ordered += [
        MiniMaxH3OrderedReference("video", index, has_audio=bool(flag)) for index, flag in enumerate(video_has_audio)
    ]
    ordered += [MiniMaxH3OrderedReference("audio", index, has_audio=True) for index in range(num_audios)]
    return ordered


def ordered_references_from_request(
    requested: Sequence[tuple[str, int]],
    *,
    num_images: int,
    video_has_audio: Sequence[bool],
    num_audios: int,
) -> list[MiniMaxH3OrderedReference]:
    """The caller's order, checked against the media that actually arrived.

    The order travels beside the modality buckets rather than inside them, so
    the two can disagree — a truncated upload, a dropped field. Validating here
    turns that into a clear error instead of an index error much deeper in, or
    worse, a silently mislabelled reference.

    Args:
        requested: ``(kind, index-within-its-bucket)`` in request order.
        num_images, num_audios: How many arrived in each bucket.
        video_has_audio: One soundtrack flag per video that arrived.

    Returns:
        The references in the requested order.
    """
    available = {"image": num_images, "video": len(video_has_audio), "audio": num_audios}
    seen: dict[str, set[int]] = {kind: set() for kind in _KINDS}
    ordered: list[MiniMaxH3OrderedReference] = []
    for position, entry in enumerate(requested):
        kind, index = str(entry[0]), int(entry[1])
        if kind not in _KINDS:
            raise ValueError(f"reference {position} has kind {kind!r}, expected one of {_KINDS}")
        if not 0 <= index < available[kind]:
            raise ValueError(
                f"reference {position} points at {kind} #{index}, but the request carried "
                f"{available[kind]} {kind} reference(s)"
            )
        if index in seen[kind]:
            raise ValueError(f"reference {position} repeats {kind} #{index}")
        seen[kind].add(index)
        ordered.append(
            MiniMaxH3OrderedReference(
                kind,
                index,
                has_audio=bool(video_has_audio[index]) if kind == "video" else kind == "audio",
            )
        )
    for kind, count in available.items():
        if len(seen[kind]) != count:
            raise ValueError(f"the request carried {count} {kind} reference(s) but the order names {len(seen[kind])}")
    return ordered


def condition_labels(references: Sequence[MiniMaxH3OrderedReference]) -> list[tuple[str, int]]:
    """The prompt's reference labels, in emission order with per-modality ordinals.

    A video that carries sound is labelled ``"<Audio j>"`` *before* ``"<Video
    k>"``, which mirrors the order its rows are packed in.

    Returns:
        ``(kind, ordinal)`` pairs, ordinals 1-based per modality.
    """
    counts = {"image": 0, "video": 0, "audio": 0}
    labels: list[tuple[str, int]] = []
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            labels.append(("audio", counts["audio"]))
        if reference.kind == "image":
            counts["image"] += 1
            labels.append(("image", counts["image"]))
        elif reference.kind == "video":
            counts["video"] += 1
            labels.append(("video", counts["video"]))
    return labels


def render_condition_label(kind: str, ordinal: int) -> str:
    """The exact token label the Qwen presentation writes into the prompt."""
    if kind not in _PROMPT_KIND_NAMES:
        raise ValueError(f"condition kind must be one of {_KINDS}, got {kind!r}")
    ordinal = int(ordinal)
    if ordinal <= 0:
        raise ValueError(f"condition ordinal must be positive, got {ordinal}")
    return f"<{_PROMPT_KIND_NAMES[kind]} {ordinal}>"


def audio_bearing_order(references: Sequence[MiniMaxH3OrderedReference]) -> list[int]:
    """Positions of the references that contribute audio rows, in packed order.

    The audio VAE is fed one clip per audio-bearing reference and the packed
    layout is built from each one's row count, so this is the order those
    encodings have to arrive in.
    """
    return [index for index, reference in enumerate(references) if reference.has_audio]


def visual_order(references: Sequence[MiniMaxH3OrderedReference]) -> list[int]:
    """Positions of the references that contribute video rows, in packed order."""
    return [index for index, reference in enumerate(references) if reference.kind in ("image", "video")]


def visual_bucket_slots(references: Sequence[MiniMaxH3OrderedReference], *, num_images: int) -> list[int]:
    """Where each visual reference sits in the *encoder's* output, in packed order.

    The VAEs are fed bucket by bucket — every image, then every video — because
    that is the order the request's buckets arrive in and re-ordering the encode
    itself would break the distributed video encoder's requirement that all
    ranks enter each reference in the same order. The reordering therefore
    happens once, afterwards, on the encoded rows; this is the permutation that
    does it.

    Args:
        num_images: Image references in the request, i.e. where the video
            bucket starts inside the concatenated visual rows.

    Returns:
        One index into the bucket-ordered visual list per visual reference, in
        packed order.
    """
    if num_images < 0:
        raise ValueError(f"num_images must be non-negative, got {num_images}")
    slots: list[int] = []
    for reference in references:
        if reference.kind == "image":
            slots.append(reference.bucket_index)
        elif reference.kind == "video":
            slots.append(num_images + reference.bucket_index)
    return slots


def audio_bucket_slots(
    references: Sequence[MiniMaxH3OrderedReference],
    *,
    video_has_audio: Sequence[bool],
) -> list[int]:
    """The same permutation for audio rows, in packed order.

    The audio bucket is likewise encoded in two runs: the soundtracks pulled
    out of reference videos (only the videos that carry one, in video-bucket
    order), then the standalone audio uploads. A video that carries no
    soundtrack occupies no slot, so the mapping is a running count rather than
    the video's own bucket index.
    """
    embedded_slot_of_video: list[int] = []
    embedded = 0
    for flag in video_has_audio:
        embedded_slot_of_video.append(embedded)
        if flag:
            embedded += 1

    slots: list[int] = []
    for position, reference in enumerate(references):
        if not reference.has_audio:
            continue
        if reference.kind == "video":
            if not video_has_audio[reference.bucket_index]:
                raise ValueError(
                    f"reference {position} is video #{reference.bucket_index}, which carries no soundtrack"
                )
            slots.append(embedded_slot_of_video[reference.bucket_index])
        elif reference.kind == "audio":
            slots.append(embedded + reference.bucket_index)
        else:  # pragma: no cover - MiniMaxH3OrderedReference rejects this
            raise ValueError(f"reference {position} is an {reference.kind} and carries no soundtrack")
    return slots


def describe_order(references: Sequence[MiniMaxH3OrderedReference], mode: str) -> dict:
    """A record of the order actually used, for result metadata.

    Emitted so a stored artifact says which order produced it. Under the
    bucketed entry the order is derived, and calling that "the official order"
    would be a claim the request never supported.
    """
    if mode not in MINIMAX_H3_ORDER_MODES:
        raise ValueError(f"order mode must be one of {sorted(MINIMAX_H3_ORDER_MODES)}, got {mode!r}")
    return {
        "reference_order_mode": mode,
        "reference_order": [
            {"kind": reference.kind, "bucket_index": reference.bucket_index, "has_audio": reference.has_audio}
            for reference in references
        ],
        "labels": [render_condition_label(kind, ordinal) for kind, ordinal in condition_labels(references)],
    }


__all__ = [
    "MINIMAX_H3_ORDER_BUCKETS",
    "MINIMAX_H3_ORDER_MODES",
    "MINIMAX_H3_ORDER_REQUEST",
    "MiniMaxH3OrderedReference",
    "audio_bearing_order",
    "audio_bucket_slots",
    "canonical_order_from_buckets",
    "ordered_references_from_request",
    "condition_labels",
    "describe_order",
    "render_condition_label",
    "visual_bucket_slots",
    "visual_order",
]
