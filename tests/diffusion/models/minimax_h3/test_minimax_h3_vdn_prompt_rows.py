# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The VDN branch must be seeded from the WHOLE prompt, on every task.

The branch's two directional scans start from a state written by the prompt rows. On
t2va those rows are one contiguous run, and taking the leading run is the whole prompt.
fl2va and ref2va interleave reference-media rows through the text region, so the leading
run there is a fraction of it -- 6 of 36 rows on a 15 s ref2va request. Seeding from that
fraction still renders, still produces video, and quietly conditions the branch on a
sixth of the prompt, so nothing downstream reports it.

``_text_state`` writes the prompt as a single delta-rule chunk with no causal scan inside
it, which is why the scattered rows may be gathered: the state depends on which rows are
present, not on their adjacency.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TEXT = 1  # MINIMAX_H3_TEXT_TAG
VIDEO, AUDIO = 0, 2


def _rows(tags):
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import _prompt_rows

    return _prompt_rows(torch.tensor(tags, dtype=torch.int32))


def test_contiguous_prompt_stays_a_slice():
    """t2va must keep the exact form it ships with today, not become a gather."""
    result = _rows([TEXT] * 5 + [AUDIO] * 3 + [VIDEO] * 4)
    assert result == (0, 5)


def test_interleaved_prompt_returns_every_prompt_row():
    """The ref2va layout: reference-media rows split the text region.

    The leading run is 2 rows; the prompt is 5. Returning the run would drop 3 of them.
    """
    tags = [TEXT, TEXT, VIDEO, VIDEO, VIDEO, TEXT, TEXT, TEXT, AUDIO, AUDIO]
    result = _rows(tags)
    assert isinstance(result, torch.Tensor)
    assert result.tolist() == [0, 1, 5, 6, 7]
    assert len(result) == sum(tag == TEXT for tag in tags)


def test_prompt_absent_is_none():
    assert _rows([VIDEO] * 4 + [AUDIO] * 2) is None


def test_gathering_contiguous_rows_equals_slicing_them():
    """The gather path must be the slice path when the rows happen to be adjacent.

    This is what licenses using it for ref2va: the two forms differ only in WHICH rows
    reach the branch, never in how those rows are turned into a state.
    """
    torch.manual_seed(0)
    hidden = torch.randn(12, 8)
    rows = torch.arange(3, 9)
    torch.testing.assert_close(hidden.index_select(0, rows), hidden[3:9])
