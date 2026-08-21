# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SLA block-sparse attention backend.

The gating tests run anywhere; the kernel tests need a CUDA device because the
selection and attention kernels are Triton.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="SLA kernels are Triton/CUDA")
requires_sla = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("sparse_linear_attention") is None,
    reason="sparse_linear_attention is not installed",
)


def _impl(sparsity, *, head_size=64, causal=False, qkv_layout="BSND", start_step=0, skip_layers=()):
    from vllm_omni.diffusion.attention.backends.sla_attn import SLAAttentionImpl

    return SLAAttentionImpl(
        num_heads=2,
        head_size=head_size,
        softmax_scale=head_size**-0.5,
        causal=causal,
        qkv_layout=qkv_layout,
        prefix="blocks.7.attn",
        backend_kwargs={"sparsity": sparsity, "start_step": start_step, "skip_layers": list(skip_layers)},
    )


def _metadata(used_len):
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

    return AttentionMetadata(extra={"max_seqlen_q": used_len, "valid_kv_length": used_len})


def test_backend_declares_its_contract():
    from vllm_omni.diffusion.attention.backends.sla_attn import SLAAttentionBackend

    assert SLAAttentionBackend.get_name() == "SLA_ATTN"
    assert SLAAttentionBackend.supported_platforms == ("cuda",)
    assert SLAAttentionBackend.get_supported_head_sizes() == [64, 128]


def test_missing_kernel_package_refuses_startup_instead_of_falling_back(monkeypatch):
    # A sparsity-distilled checkpoint run on a dense path still returns valid
    # video — slower and off-distribution — so a silent fallback is invisible to
    # callers and to monitoring. Resolution must fail loudly instead. Guard the
    # behaviour here so nobody restores a fallback in the platform layer.
    import importlib.util

    from vllm_omni.diffusion.attention.backends.sla_attn import SLAAttentionBackend

    # validate_available() imports find_spec at call time, so patching the
    # module attribute reaches it whether or not the package is installed here.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "sparse_linear_attention" else True)
    with pytest.raises(ValueError, match="pip install"):
        SLAAttentionBackend.validate_available()


def test_block_sparse_spec_reaches_the_backend():
    from vllm_omni.diffusion.data import BLOCK_SPARSE_BACKENDS, AttentionSpec

    assert "SLA_ATTN" in BLOCK_SPARSE_BACKENDS
    spec = AttentionSpec(backend="SLA_ATTN", block_sparse={"sparsity": 0.85, "start_step": 1})
    assert spec.backend_kwargs() == {"sparsity": 0.85, "start_step": 1}


def test_causal_and_wrong_layout_are_rejected_up_front():
    # A wrong axis or an unexpressible mask must fail while the operator can
    # still pick another backend, not silently produce a different attention.
    with pytest.raises(ValueError, match="causal"):
        _impl(0.85, causal=True)
    with pytest.raises(ValueError, match="qkv_layout"):
        _impl(0.85, qkv_layout="SBHD")


@pytest.mark.parametrize(
    ("sparsity", "used_len", "start_step", "skip_layers", "reason"),
    [
        (0.0, 4096, 0, (), "sparsity disabled"),
        (0.85, 512, 0, (), "too few key blocks"),
        (0.85, 4096, 0, (7,), "layer exempted"),
    ],
)
def test_plan_declines_and_the_layer_stays_dense(sparsity, used_len, start_step, skip_layers, reason):
    impl = _impl(sparsity, start_step=start_step, skip_layers=skip_layers)
    query = torch.zeros((1, used_len, 2, 64), dtype=torch.bfloat16)
    assert impl._resolve_plan(query, _metadata(used_len)) is None, reason


def test_plan_confines_itself_to_the_used_prefix():
    impl = _impl(0.85)
    query = torch.zeros((1, 4096, 2, 64), dtype=torch.bfloat16)
    plan = impl._resolve_plan(query, _metadata(3072))
    # Padding rows have no mask input in the kernel, so they must be excluded
    # from the sequence the sparse path sees rather than attended over.
    assert plan is not None
    assert plan.used_len == 3072
    assert plan.key_blocks == 3072 // 64


@requires_cuda
@requires_sla
@pytest.mark.parametrize("used_len", [4096, 3072])
def test_zero_sparsity_matches_dense_attention(used_len):
    # topk_ratio == 1 keeps every key block, so the kernel must reproduce dense
    # attention. This is the wiring check: layout, scale and the transpose in
    # and out are all wrong-detectable here.
    impl = _impl(0.999999, head_size=64)
    impl.sla = type(impl.sla)(sparsity=0.0, start_step=0, skip_layers=frozenset())
    torch.manual_seed(0)
    shape = (1, used_len, 2, 64)
    query, key, value = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))

    from vllm_omni.diffusion.attention.backends.sla_attn import SLAPlan

    plan = SLAPlan(used_len=used_len, key_blocks=used_len // 64)
    got = impl._forward_sparse(query, key, value, plan)

    reference = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2).float(),
        key.transpose(1, 2).float(),
        value.transpose(1, 2).float(),
        scale=impl.softmax_scale,
    ).transpose(1, 2)
    torch.testing.assert_close(got.float(), reference, atol=2e-2, rtol=2e-2)


@requires_cuda
@requires_sla
def test_sparse_output_drops_blocks_but_stays_finite():
    impl = _impl(0.85, head_size=64)
    torch.manual_seed(0)
    shape = (1, 4096, 2, 64)
    query, key, value = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))

    plan = impl._resolve_plan(query, _metadata(4096))
    assert plan is not None
    sparse = impl._forward_sparse(query, key, value, plan)
    dense = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2).float(),
        key.transpose(1, 2).float(),
        value.transpose(1, 2).float(),
        scale=impl.softmax_scale,
    ).transpose(1, 2)

    assert torch.isfinite(sparse).all()
    # Dropping 85% of the key blocks must change the result; an output that
    # still matched dense would mean the block map never reached the kernel.
    assert not torch.allclose(sparse.float(), dense, atol=1e-2)


@requires_cuda
@requires_sla
def test_padding_rows_are_zeroed_not_attended():
    impl = _impl(0.85, head_size=64)
    torch.manual_seed(0)
    shape = (1, 4096, 2, 64)
    query, key, value = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))

    used = 3072
    out = impl._forward_sparse(query, key, value, impl._resolve_plan(query, _metadata(used)))
    assert out.shape == query.shape
    assert torch.count_nonzero(out[:, used:]) == 0
    assert torch.isfinite(out[:, :used]).all()
