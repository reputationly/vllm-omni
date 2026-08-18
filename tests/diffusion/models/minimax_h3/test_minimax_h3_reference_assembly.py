# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The reference order has to move the rows, not only the prompt labels.

The failure this guards against produces a perfectly valid video of the wrong
thing: a request asks for *video, image, audio*, the prompt is numbered in that
order, and the packed layout still carries images first — so ``<Video 1>``
labels the image's rows. Nothing raises, nothing logs, and the only symptom is
that the model was conditioned on the wrong reference.

Everything here is checked on the assembled artifacts (rows, shapes, audio
lengths, ``ref_blocks``) rather than on the labels, because agreeing labels
were exactly what the bug already had.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _rows(marker: float, count: int, width: int = 96) -> torch.Tensor:
    """Rows whose value identifies which reference they came from."""
    return torch.full((count, width), float(marker))


def _visual_rows(shape: tuple[int, int, int]) -> int:
    latent_t, latent_h, latent_w = shape
    return latent_t * (latent_h // 2) * (latent_w // 2)


def test_a_bucketed_request_is_assembled_exactly_as_before():
    """The identity case: same rows, same shapes, same blocks as the old loops."""
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import canonical_order_from_buckets
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _order_reference_conditions

    image_shape = (1, 8, 8)
    video_shape = (2, 4, 6)
    visual = torch.cat([_rows(1.0, _visual_rows(image_shape)), _rows(2.0, _visual_rows(video_shape))])
    audio = torch.cat([_rows(10.0, 80 * 2, width=32), _rows(20.0, 96 * 2, width=32)])

    references = canonical_order_from_buckets(num_images=1, video_has_audio=[True], num_audios=1)
    ordered_visual, shapes, ordered_audio, lengths, blocks = _order_reference_conditions(
        references,
        visual_condition=visual,
        visual_shapes=[image_shape, video_shape],
        num_images=1,
        audio_condition=audio,
        audio_lengths=[80, 96],
        video_has_audio=[True],
    )

    assert torch.equal(ordered_visual, visual)
    assert torch.equal(ordered_audio, audio)
    assert shapes == [image_shape, video_shape]
    assert lengths == [80, 96]
    assert blocks == [
        {"kind": "image", "latent_h": 8, "latent_w": 8},
        {"kind": "video_audio", "ref_audio_t": 80, "latent_t": 2, "latent_h": 4, "latent_w": 6},
        {"kind": "audio", "ref_audio_t": 96},
    ]


def test_an_interleaved_request_moves_the_rows_with_the_labels():
    """video, image, audio — the artifact that used to disagree with the prompt."""
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import (
        condition_labels,
        ordered_references_from_request,
    )
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _order_reference_conditions

    image_shape = (1, 8, 8)
    video_shape = (2, 4, 6)
    image_rows = _rows(1.0, _visual_rows(image_shape))
    video_rows = _rows(2.0, _visual_rows(video_shape))
    embedded_audio = _rows(10.0, 80 * 2, width=32)
    standalone_audio = _rows(20.0, 96 * 2, width=32)

    references = ordered_references_from_request(
        [("video", 0), ("image", 0), ("audio", 0)],
        num_images=1,
        video_has_audio=[True],
        num_audios=1,
    )
    ordered_visual, shapes, ordered_audio, lengths, blocks = _order_reference_conditions(
        references,
        # As encoded: images first, then videos; embedded audio, then standalone.
        visual_condition=torch.cat([image_rows, video_rows]),
        visual_shapes=[image_shape, video_shape],
        num_images=1,
        audio_condition=torch.cat([embedded_audio, standalone_audio]),
        audio_lengths=[80, 96],
        video_has_audio=[True],
    )

    assert torch.equal(ordered_visual, torch.cat([video_rows, image_rows]))
    assert torch.equal(ordered_audio, torch.cat([embedded_audio, standalone_audio]))
    assert shapes == [video_shape, image_shape]
    assert lengths == [80, 96]
    assert blocks == [
        {"kind": "video_audio", "ref_audio_t": 80, "latent_t": 2, "latent_h": 4, "latent_w": 6},
        {"kind": "image", "latent_h": 8, "latent_w": 8},
        {"kind": "audio", "ref_audio_t": 96},
    ]
    # And the labels the prompt gets describe that same sequence.
    assert condition_labels(references) == [("audio", 1), ("video", 1), ("image", 1), ("audio", 2)]


def test_the_packed_layout_agrees_with_the_assembled_rows():
    """The real invariant: as many condition positions as conditioning rows.

    Assembling the blocks and the rows separately is what allowed them to
    drift, so the assertion is made against the layout builder itself rather
    than against a second copy of the arithmetic.
    """
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import ordered_references_from_request
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        MINIMAX_H3_AUDIO_REF_COND_ID,
        minimax_h3_packed_sequence_ref2va_blocks,
    )
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _order_reference_conditions

    image_shape = (1, 8, 8)
    video_shape = (2, 4, 6)
    references = ordered_references_from_request(
        [("audio", 0), ("video", 0), ("image", 0)],
        num_images=1,
        video_has_audio=[True],
        num_audios=1,
    )
    ordered_visual, _, ordered_audio, _, blocks = _order_reference_conditions(
        references,
        visual_condition=torch.cat([_rows(1.0, _visual_rows(image_shape)), _rows(2.0, _visual_rows(video_shape))]),
        visual_shapes=[image_shape, video_shape],
        num_images=1,
        audio_condition=torch.cat([_rows(10.0, 80 * 2, width=32), _rows(20.0, 96 * 2, width=32)]),
        audio_lengths=[80, 96],
        video_has_audio=[True],
    )
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=16,
        latent_t=2,
        latent_h=16,
        latent_w=16,
        audio_t=48,
        ref_blocks=blocks,
    )
    # ``diffuse`` writes the generated rows where ``update_mask`` is true and
    # the conditioning rows into the complement, so the complement's size is
    # exactly how many conditioning rows the layout is expecting.
    assert int((~packed["update_mask"]).sum()) == int(ordered_visual.shape[0])
    assert int((~packed["audio_update_mask"]).sum()) == int(ordered_audio.shape[0])
    # And the standalone audio really is first, ahead of the video block.
    assert int(packed["input_ids"][16]) == MINIMAX_H3_AUDIO_REF_COND_ID


def test_a_row_count_that_disagrees_with_the_shapes_fails_here():
    """A short encode must not shift every later reference by a few rows."""
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import canonical_order_from_buckets
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _order_reference_conditions

    references = canonical_order_from_buckets(num_images=1, video_has_audio=[], num_audios=0)
    with pytest.raises(ValueError, match="conditioning rows"):
        _order_reference_conditions(
            references,
            visual_condition=_rows(1.0, 15),  # (1, 8, 8) is 16 rows
            visual_shapes=[(1, 8, 8)],
            num_images=1,
            audio_condition=None,
            audio_lengths=[],
            video_has_audio=[],
        )


def test_a_legacy_instance_refuses_an_explicit_order_rather_than_dropping_it():
    """Declared-but-unread is the defect class; make the contract answer for it."""
    from vllm_omni.diffusion.models.minimax_h3.ordered_references import MINIMAX_H3_ORDER_REQUEST
    from vllm_omni.diffusion.models.minimax_h3.strategy import legacy_strategy, official_diffusers_v1_strategy

    assert legacy_strategy().reference_order_mode != MINIMAX_H3_ORDER_REQUEST
    assert official_diffusers_v1_strategy().reference_order_mode == MINIMAX_H3_ORDER_REQUEST
