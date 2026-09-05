# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import _ensure_image_placeholders

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_single_image_gets_one_bare_placeholder():
    assert _ensure_image_placeholders("describe it", 1) == "<image>\ndescribe it"


def test_multiple_images_get_upstream_numbered_prefix():
    # Upstream's own multi-image form, from ``it2i_generate``.
    assert _ensure_image_placeholders("blend them", 3) == (
        "Image-1:<image>\nImage-2:<image>\nImage-3:<image>\nblend them"
    )


def test_caller_authored_placeholders_are_left_alone():
    prompt = "put <image> next to <image>"
    assert _ensure_image_placeholders(prompt, 2) == prompt


def test_partial_placeholders_are_topped_up_without_renumbering():
    # One marker authored, two images: top up bare placeholders rather than
    # renumbering, or the caller's own marker would lose its position.
    assert _ensure_image_placeholders("use <image> here", 2) == "<image>\nuse <image> here"


def test_more_placeholders_than_images_is_left_alone():
    prompt = "<image> <image> <image>"
    assert _ensure_image_placeholders(prompt, 1) == prompt


@pytest.mark.parametrize("n_images", [1, 2, 3, 9])
def test_placeholder_count_always_covers_every_image(n_images):
    # The invariant the expansion loops depend on: one placeholder per image, so
    # every image's ViT features get an <IMG_CONTEXT> span to land in.
    assert _ensure_image_placeholders("edit", n_images).count("<image>") == n_images
