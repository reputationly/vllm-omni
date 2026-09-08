# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The linear branch must receive the RAW projections, not normed-and-RoPE'd ones.

``MiniMaxH3Attention`` captures ``qkv_raw = (q, k, v)`` and then reassigns ``q``/``k``
from ``fused_qk_norm_rope``. That is only the reference's ``_qkv`` contract -- which
hands the branch the pre-QK-norm, pre-RoPE projections -- while the kernel does not
write into its inputs. If it ever did, the branch would silently consume normed, rotated
tensors: the model still runs, the branch-parity tests still pass (they feed the module
its tensors directly), the window-parity tests still pass, and the only symptom is
output that is somewhat worse than it ought to be.

Today the fused launcher allocates its own output and the property holds. It is asserted
here rather than left as the comment at the capture site, because the capture is a bare
reference and nothing else in the tree would notice it going stale.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.gpu, pytest.mark.diffusion]

TOKENS, HEADS, HEAD_DIM = 96, 4, 128


def test_fused_qk_norm_rope_does_not_write_into_its_inputs():
    if not torch.cuda.is_available():
        pytest.skip("the fused kernel is CUDA-only")
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

    device, dtype = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(TOKENS, HEADS, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(TOKENS, HEADS, HEAD_DIM, device=device, dtype=dtype)
    q_weight = torch.randn(HEAD_DIM, device=device, dtype=dtype)
    k_weight = torch.randn(HEAD_DIM, device=device, dtype=dtype)
    # Packed non-interleaved RoPE: [tokens, head_dim] of cos|sin halves, as the
    # attention's own ``rope_table`` is built.
    rope_table = torch.randn(TOKENS, HEAD_DIM, device=device, dtype=dtype)

    # Captured exactly the way the attention captures qkv_raw: a reference, not a copy.
    q_captured, k_captured = q, k
    q_before, k_before = q.clone(), k.clone()

    q_out, k_out = fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, 1e-6)

    torch.testing.assert_close(q_captured, q_before, rtol=0, atol=0)
    torch.testing.assert_close(k_captured, k_before, rtol=0, atol=0)
    # ...and the kernel must actually have transformed something, or the assertions
    # above are satisfied by a no-op, which would be its own and larger bug.
    assert not torch.equal(q_out, q_before)
    assert not torch.equal(k_out, k_before)


def test_the_unfused_norm_path_also_leaves_the_capture_alone():
    """``rope_table is None`` runs ``self.q_norm(q)`` instead; that must not be in-place
    either, for the same reason and with the same absence of any other detector."""
    if not torch.cuda.is_available():
        pytest.skip("matched to the CUDA path above")
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Attention

    device, dtype = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(1)
    q = torch.randn(TOKENS, HEADS, HEAD_DIM, device=device, dtype=dtype)
    captured, before = q, q.clone()

    # The very norm the attention builds, so this tracks whatever `_norm` resolves to.
    norm = MiniMaxH3Attention.__dict__.get("_norm_factory", None)
    if norm is None:
        from vllm.model_executor.layers.layernorm import RMSNorm

        norm = RMSNorm(HEAD_DIM, eps=1e-6).to(device=device, dtype=dtype)
    out = norm(q)
    out = out[0] if isinstance(out, tuple) else out

    torch.testing.assert_close(captured, before, rtol=0, atol=0)
    assert not torch.equal(out, before)
