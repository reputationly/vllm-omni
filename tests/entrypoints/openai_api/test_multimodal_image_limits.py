# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The one admission rule every image-accepting route shares.

``/v1/images/edits`` and the image task routes have always applied it. The chat
routes used to answer it themselves off ``getattr(engine, "od_config", None)``,
which is ``None`` on an ``AsyncOmni`` deployment (only the
``get_diffusion_od_config()`` method exists there), so their limit check was
dead and every model admitted any number of images.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai.utils import (
    max_multimodal_image_inputs,
    resolve_diffusion_od_config,
    too_many_input_images_message,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("od_config", "expected"),
    [
        # Not knowable from here: never invent a rejection.
        (None, None),
        (SimpleNamespace(), None),
        # Declared single-image-only.
        (SimpleNamespace(supports_multimodal_inputs=False), 1),
        # Multi-image with no declared ceiling.
        (SimpleNamespace(supports_multimodal_inputs=True, max_multimodal_image_inputs=None), None),
        # Declared ceilings.
        (SimpleNamespace(supports_multimodal_inputs=True, max_multimodal_image_inputs=9), 9),
        (SimpleNamespace(supports_multimodal_inputs=True, max_multimodal_image_inputs=3), 3),
        # bool is an Integral in Python; it must not read as a count of 1.
        (SimpleNamespace(supports_multimodal_inputs=True, max_multimodal_image_inputs=True), None),
        (SimpleNamespace(supports_multimodal_inputs=True, max_multimodal_image_inputs=0), None),
    ],
)
def test_limit_rule(od_config, expected):
    assert max_multimodal_image_inputs(od_config) == expected


def test_rejection_text_matches_the_edits_route_wording():
    # tests/entrypoints/openai_api/test_image_server.py asserts this exact string.
    assert too_many_input_images_message(2, 1) == (
        "Received multiple input images. Only a single image is supported by this model."
    )
    assert too_many_input_images_message(12, 3) == (
        "Received 12 input images. At most 3 images are supported by this model."
    )


class _AsyncOmniLike:
    """Exposes the method but not the attribute, like the real ``AsyncOmni``."""

    def get_diffusion_od_config(self):
        return SimpleNamespace(supports_multimodal_inputs=False)


def test_engine_exposing_only_the_method_still_resolves():
    engine = _AsyncOmniLike()

    # The bug: reading the attribute alone sees nothing on this engine.
    assert getattr(engine, "od_config", None) is None
    # The fix: the shared resolver asks for the method first.
    assert max_multimodal_image_inputs(resolve_diffusion_od_config(engine)) == 1
