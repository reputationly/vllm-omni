# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SLA_SAGE2_ATTN: SLA block selection run on SpargeAttn's SageAttention2 kernel.

The gating tests run anywhere; the kernel tests need a CUDA device and both
`sparse_linear_attention` (block selection) and `spas_sage_attn` (the sage2
kernel) installed.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="sage2 kernel is a CUDA extension")
requires_sla = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("sparse_linear_attention") is None,
    reason="sparse_linear_attention is not installed",
)
requires_sparge = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("spas_sage_attn") is None,
    reason="spas_sage_attn is not installed",
)


def _impl(sparsity, *, head_size=64, causal=False, qkv_layout="BSND", start_step=0, skip_layers=()):
    from vllm_omni.diffusion.attention.backends.sla_sage2_attn import SLASage2AttentionImpl

    return SLASage2AttentionImpl(
        num_heads=2,
        head_size=head_size,
        softmax_scale=head_size**-0.5,
        causal=causal,
        qkv_layout=qkv_layout,
        prefix="blocks.7.attn",
        backend_kwargs={"sparsity": sparsity, "start_step": start_step, "skip_layers": list(skip_layers)},
    )


def _metadata(used_len, *, prefix_len=None):
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout

    video_layout = None
    if prefix_len is not None:
        # latent_grid is unused by this backend's own logic; the placeholder
        # only needs to satisfy VideoTokenLayout's shape.
        video_layout = VideoTokenLayout(prefix_len=prefix_len, latent_grid=(1, 1, max(1, used_len - prefix_len)))
    return AttentionMetadata(
        extra={"max_seqlen_q": used_len, "valid_kv_length": used_len},
        video_layout=video_layout,
    )


def test_backend_declares_its_contract():
    from vllm_omni.diffusion.attention.backends.sla_sage2_attn import SLASage2AttentionBackend

    assert SLASage2AttentionBackend.get_name() == "SLA_SAGE2_ATTN"
    assert SLASage2AttentionBackend.supported_platforms == ("cuda",)
    assert SLASage2AttentionBackend.get_supported_head_sizes() == [64, 128]


def test_missing_sla_package_refuses_startup(monkeypatch):
    import importlib.util

    from vllm_omni.diffusion.attention.backends.sla_sage2_attn import SLASage2AttentionBackend

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "sparse_linear_attention" else True)
    with pytest.raises(ValueError, match="pip install"):
        SLASage2AttentionBackend.validate_available()


def test_missing_sparge_package_refuses_startup(monkeypatch):
    import importlib.util

    from vllm_omni.diffusion.attention.backends.sla_sage2_attn import SLASage2AttentionBackend

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "spas_sage_attn" else True)
    with pytest.raises(ValueError, match="SpargeAttn"):
        SLASage2AttentionBackend.validate_available()


def test_block_sparse_spec_reaches_the_backend():
    from vllm_omni.diffusion.data import BLOCK_SPARSE_BACKENDS, AttentionSpec

    assert "SLA_SAGE2_ATTN" in BLOCK_SPARSE_BACKENDS
    spec = AttentionSpec(backend="SLA_SAGE2_ATTN", block_sparse={"sparsity": 0.85, "start_step": 1})
    assert spec.backend_kwargs() == {"sparsity": 0.85, "start_step": 1}


def test_causal_and_wrong_layout_are_rejected_up_front():
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
    assert plan is not None
    assert plan.used_len == 3072
    assert plan.key_blocks == 3072 // impl.block_k


def test_plan_prefix_len_defaults_to_zero_without_video_layout():
    # No video_layout published: the whole used_len is one region, matching
    # this backend's original (pre-fix) behavior.
    impl = _impl(0.85)
    query = torch.zeros((1, 4096, 2, 64), dtype=torch.bfloat16)
    plan = impl._resolve_plan(query, _metadata(4096))
    assert plan is not None
    assert plan.prefix_len == 0


def test_plan_reads_prefix_len_from_video_layout():
    impl = _impl(0.85)
    query = torch.zeros((1, 4096, 2, 64), dtype=torch.bfloat16)
    plan = impl._resolve_plan(query, _metadata(4096, prefix_len=800))
    assert plan is not None
    assert plan.prefix_len == 800
    # key_blocks counts the K side, which stays the full used_len — only the
    # query side shrinks to the video segment.
    assert plan.key_blocks == 4096 // impl.block_k


def test_plan_declines_when_prefix_consumes_the_whole_sequence():
    impl = _impl(0.85)
    query = torch.zeros((1, 4096, 2, 64), dtype=torch.bfloat16)
    assert impl._resolve_plan(query, _metadata(4096, prefix_len=4096)) is None


@requires_cuda
@requires_sla
@requires_sparge
@pytest.mark.parametrize("used_len", [4096, 3072])
def test_zero_sparsity_matches_dense_attention(used_len):
    # topk_ratio == 1 keeps every key block, so the kernel must reproduce dense
    # attention. This is the wiring check: layout, scale and the transpose in
    # and out are all wrong-detectable here. Measured on an A100 (sm80,
    # BLKQ=128/BLKK=64): max abs error 0.0030 at used_len=4096, 0.0028 at
    # used_len=3072 (mean ~3e-4) — the INT8 quantization path is actually
    # *tighter* than SLA_ATTN's pure-float Triton kernel (2e-2 tolerance)
    # here, not looser as originally assumed. atol/rtol below has ~3x margin
    # over the measured max, not a guess.
    impl = _impl(0.999999, head_size=64)
    impl.sla = type(impl.sla)(sparsity=0.0, start_step=0, skip_layers=frozenset())
    torch.manual_seed(0)
    shape = (1, used_len, 2, 64)
    query, key, value = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))

    from vllm_omni.diffusion.attention.backends.sla_sage2_attn import SLASage2Plan

    plan = SLASage2Plan(used_len=used_len, key_blocks=used_len // impl.block_k)
    got = impl._forward_sparse(query, key, value, plan)

    reference = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2).float(),
        key.transpose(1, 2).float(),
        value.transpose(1, 2).float(),
        scale=impl.softmax_scale,
    ).transpose(1, 2)
    torch.testing.assert_close(got.float(), reference, atol=1e-2, rtol=1e-2)


@requires_cuda
@requires_sla
@requires_sparge
def test_prefix_rows_get_exact_dense_attention_when_video_layout_is_set():
    # The whole point of the prefix split: prefix-row outputs must match dense
    # SDPA regardless of sparsity, because they never enter block selection.
    # This is bf16-FlashAttention-vs-float32-SDPA precision, not INT8-kernel
    # precision (that's SLA_SAGE2_ATTN's video path, tested separately below).
    # Measured on an A100 (sm80): max abs error 0.00045, mean 0.00005 — the
    # atol below has ~4x margin over the measured max, not a guess.
    impl = _impl(0.85, head_size=64)
    torch.manual_seed(0)
    used_len = 4096
    prefix_len = 800
    shape = (1, used_len, 2, 64)
    query, key, value = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))

    plan = impl._resolve_plan(query, _metadata(used_len, prefix_len=prefix_len))
    assert plan is not None
    assert plan.prefix_len == prefix_len
    out = impl._forward_sparse(query, key, value, plan)

    reference = torch.nn.functional.scaled_dot_product_attention(
        query[:, :prefix_len].transpose(1, 2).float(),
        key[:, :used_len].transpose(1, 2).float(),
        value[:, :used_len].transpose(1, 2).float(),
        scale=impl.softmax_scale,
    ).transpose(1, 2)
    torch.testing.assert_close(out[:, :prefix_len].float(), reference, atol=2e-3, rtol=2e-2)


@requires_cuda
@requires_sla
@requires_sparge
def test_video_rows_still_drop_blocks_when_prefix_is_exempted():
    impl = _impl(0.85, head_size=64)
    torch.manual_seed(0)
    used_len = 4096
    prefix_len = 800
    shape = (1, used_len, 2, 64)
    query, key, value = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))

    plan = impl._resolve_plan(query, _metadata(used_len, prefix_len=prefix_len))
    assert plan is not None
    out = impl._forward_sparse(query, key, value, plan)
    dense_video = torch.nn.functional.scaled_dot_product_attention(
        query[:, prefix_len:used_len].transpose(1, 2).float(),
        key[:, :used_len].transpose(1, 2).float(),
        value[:, :used_len].transpose(1, 2).float(),
        scale=impl.softmax_scale,
    ).transpose(1, 2)

    video_out = out[:, prefix_len:used_len]
    assert torch.isfinite(video_out).all()
    # Dropping blocks must change the result; matching dense would mean the
    # block map never reached the kernel for the video segment.
    assert not torch.allclose(video_out.float(), dense_video, atol=1e-2)


@requires_cuda
@requires_sla
@requires_sparge
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
    assert not torch.allclose(sparse.float(), dense, atol=1e-2)


@requires_cuda
@requires_sla
@requires_sparge
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
