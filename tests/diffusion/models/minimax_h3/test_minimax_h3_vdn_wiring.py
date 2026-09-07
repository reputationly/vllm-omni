# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""How the VDN halves compose inside MiniMaxH3Attention.

The window and the branch are each verified against upstream elsewhere. What is only
testable here is the assembly: the gate has to scale the attention output BEFORE the
output projection, the branch has to land on the video rows only, and the two
row-parallel projections have to share one all-reduce -- at the 15 s shape a second
collective costs a third of a block, so "it still renders" is not evidence it is right.
"""

import os

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout

pytestmark = [pytest.mark.core_model, pytest.mark.gpu, pytest.mark.diffusion]

HIDDEN, HEADS, HEAD_DIM = 256, 4, 64
GRID = (17, 3, 4)  # (latent frames, patched h, patched w)
TOKENS_PER_FRAME = GRID[1] * GRID[2]
PREFIX_ROWS = 9
TEXT_SPAN = (0, 5)  # prompt first, then audio -- the t2va packing


@pytest.fixture(scope="module")
def distributed():
    """The single-rank tensor-parallel group vLLM's parallel linears need to exist.

    The config context stays open for the whole fixture, not just the init: the parallel
    linears read it while they are being constructed too.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29517")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        yield
    cleanup_dist_env_and_memory()


def _arch():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTArchConfig

    return MiniMaxH3DiTArchConfig.from_mapping(
        {
            "num_attention_heads": HEADS,
            "attention_head_dim": HEAD_DIM,
            "hidden_size": HIDDEN,
            "num_layers": 1,
            "num_refiner_layers": 1,
            "ffn_dim": HIDDEN * 2,
            "in_channels": 24,
            "audio_in_channels": 32,
            "patch_size": [1, 2, 2],
            "text_dim": HIDDEN,
            "freq_dim": 256,
            "time_embed_hidden_dim": HIDDEN,
            "time_embed_dim": HIDDEN,
            "rope_freq_dim": 16,
            "rope_theta": 10000.0,
            "norm_eps": 1e-5,
            "qk_norm_eps": 1e-5,
            "final_norm_eps": 1e-5,
        }
    )


def _attention(enable_vdn: bool):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Attention

    attn = MiniMaxH3Attention(_arch(), None, prefix="blocks.0.attn")
    if enable_vdn:
        attn.enable_vdn_branch()
    attn = attn.to("cuda").eval()
    # vLLM's parallel linears allocate with torch.empty and expect a weight loader to
    # fill them. Freshly mapped CUDA pages read back as zeros, so an unpopulated model
    # makes every "did this term contribute?" test pass vacuously.
    with torch.no_grad():
        for parameter in attn.parameters():
            parameter.normal_(0.0, 0.02)
    return attn


def _call(attn, x, used_len, *, text_span=TEXT_SPAN):
    cu = torch.tensor([0, used_len, x.shape[0]], dtype=torch.int32, device="cuda")
    return attn(
        x,
        rope_table=None,
        cu_seqlens=cu,
        max_seqlen=used_len,
        packed_total=x.shape[0],
        video_layout=VideoTokenLayout(prefix_len=PREFIX_ROWS, latent_grid=GRID),
        vdn_text_span=text_span,
    )


def test_enabling_the_branch_moves_the_all_reduce_off_out_proj(distributed):
    """Both projections must be partials; the layer reduces their sum exactly once."""
    attn = _attention(enable_vdn=True)
    assert attn.out_proj.reduce_results is False
    assert attn.vdn.to_out_linear.reduce_results is False


def test_branch_takes_its_head_shard_from_the_attention(monkeypatch, distributed):
    """The branch must use THIS attention's head count, not one it derives itself.

    It consumes the attention's raw q/k/v, so any independent derivation is a second
    view of the same split that can disagree. It really can: the branch's fallback asks
    the DIFFUSION parallel state, which requires the DP/CFG/SP/PP groups and can be
    uninitialised while vLLM's TP group -- the one ``qkv_proj`` actually sharded on --
    is live. Simulated here by forcing that fallback to report "unsharded" while the
    attention is built sharded; before the fix the gate came out 56-wide against a
    14-head attention output.
    """
    from vllm_omni.diffusion.models.minimax_h3 import vdn_branch
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Attention

    monkeypatch.setattr(vdn_branch, "model_parallel_is_initialized", lambda: False)
    attn = MiniMaxH3Attention(_arch(), None, prefix="blocks.0.attn")
    attn.enable_vdn_branch()

    assert attn.vdn.linear_attention.local_heads == attn.num_heads
    assert attn.vdn.softmax_gate.local_heads == attn.num_heads


def test_enabling_is_idempotent(distributed):
    """The loader may reach this twice (offload swaps rebuild modules); rebuilding would
    discard weights that were already loaded into the branch."""
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Attention

    attn = MiniMaxH3Attention(_arch(), None, prefix="blocks.0.attn")
    attn.enable_vdn_branch()
    first = attn.vdn
    attn.enable_vdn_branch()
    assert attn.vdn is first


def test_gate_scales_the_attention_output_not_the_projected_residual(distributed):
    """``out_proj(gate * attn)``, not ``gate * out_proj(attn)``.

    The two differ only when the gate varies across heads -- with a scalar gate they are
    identical, because out_proj is linear -- so the gate is forced to a distinct
    constant per head and the layer is recomposed by hand. Both spellings render a
    plausible video, which is why this is asserted rather than eyeballed.
    """
    torch.manual_seed(5)
    attn = _attention(enable_vdn=True)
    used_len = PREFIX_ROWS + GRID[0] * TOKENS_PER_FRAME
    x = torch.randn(used_len + 3, HIDDEN, device="cuda", dtype=torch.bfloat16)
    per_head_logits = torch.tensor([-2.0, -0.5, 0.5, 2.0], device="cuda")

    with torch.no_grad():
        # Isolate the softmax half: the branch's contribution is verified separately.
        attn.vdn.to_out_linear.weight.zero_()
        attn.vdn.softmax_gate.up.weight.zero_()
        attn.vdn.softmax_gate.up.bias.copy_(per_head_logits)
        actual = _call(attn, x, used_len)

        cu = torch.tensor([0, used_len, x.shape[0]], dtype=torch.int32, device="cuda")
        qkv, _ = attn.qkv_proj(x)
        size = attn.num_heads * attn.head_dim
        q, k, v = (
            tensor.view(x.shape[0], attn.num_heads, attn.head_dim) for tensor in qkv.split([size, size, size], dim=-1)
        )
        raw = attn._run_packed_attention(
            attn.q_norm(q),
            attn.k_norm(k),
            v,
            cu_seqlens=cu,
            max_seqlen=used_len,
            packed_total=x.shape[0],
            video_layout=VideoTokenLayout(prefix_len=PREFIX_ROWS, latent_grid=GRID),
        )
        gate = torch.sigmoid(per_head_logits).view(1, attn.num_heads, 1)
        expected, _ = attn.out_proj((raw * gate.to(raw.dtype)).reshape(x.shape[0], -1))

    torch.testing.assert_close(actual, expected)


def test_branch_touches_only_the_video_rows(distributed):
    """A branch that leaked into the prefix would corrupt the prompt and audio rows.

    Compared against the same layer with the branch's output projection zeroed: every
    difference must sit inside the target video span.
    """
    torch.manual_seed(6)
    attn = _attention(enable_vdn=True)
    used_len = PREFIX_ROWS + GRID[0] * TOKENS_PER_FRAME
    x = torch.randn(used_len + 3, HIDDEN, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        with_branch = _call(attn, x, used_len).clone()
        saved = attn.vdn.to_out_linear.weight.detach().clone()
        attn.vdn.to_out_linear.weight.zero_()
        without_branch = _call(attn, x, used_len).clone()
        attn.vdn.to_out_linear.weight.copy_(saved)

    difference = (with_branch.float() - without_branch.float()).abs().sum(dim=-1)
    video_start = PREFIX_ROWS
    video_end = video_start + GRID[0] * TOKENS_PER_FRAME
    assert difference[:video_start].max() == 0
    assert difference[video_end:].max() == 0
    # The two anchor frames are dense on the softmax side, so the branch is zero there.
    inner = difference[video_start + TOKENS_PER_FRAME : video_end - TOKENS_PER_FRAME]
    assert inner.max() > 0
    assert difference[video_start : video_start + TOKENS_PER_FRAME].max() == 0
    assert difference[video_end - TOKENS_PER_FRAME : video_end].max() == 0


def test_short_clip_runs_the_dense_attention_and_no_branch(distributed):
    """When the window covers the clip the softmax side IS the full attention, so the
    branch must contribute exactly nothing rather than something small."""
    torch.manual_seed(7)
    attn = _attention(enable_vdn=True)
    grid = (8, 3, 4)  # a 15-frame window spans this clip
    used_len = PREFIX_ROWS + grid[0] * TOKENS_PER_FRAME
    x = torch.randn(used_len, HIDDEN, device="cuda", dtype=torch.bfloat16)
    layout = VideoTokenLayout(prefix_len=PREFIX_ROWS, latent_grid=grid)
    assert attn.vdn.plan_geometry(layout, used_len) is None

    with torch.no_grad():
        baseline = attn(
            x,
            rope_table=None,
            cu_seqlens=torch.tensor([0, used_len], dtype=torch.int32, device="cuda"),
            max_seqlen=used_len,
            packed_total=used_len,
            video_layout=layout,
            vdn_text_span=TEXT_SPAN,
        ).clone()
        attn.vdn.to_out_linear.weight.zero_()
        zeroed = attn(
            x,
            rope_table=None,
            cu_seqlens=torch.tensor([0, used_len], dtype=torch.int32, device="cuda"),
            max_seqlen=used_len,
            packed_total=used_len,
            video_layout=layout,
            vdn_text_span=TEXT_SPAN,
        )
    torch.testing.assert_close(baseline, zeroed)


def test_dense_h3_is_untouched(distributed):
    """No VDN artifact means no branch, no gate, and out_proj keeps its own all-reduce."""
    attn = _attention(enable_vdn=False)
    assert attn.vdn is None
    assert attn.out_proj.reduce_results is True
    names = {name for name, _ in attn.named_parameters()}
    assert not [name for name in names if "vdn" in name or "linear_attention" in name]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"attention_backend": "FLASH_ATTN"}, "complement of"),
        ({"ulysses_degree": 2}, "every row of the target video"),
        ({"ring_degree": 2}, "every row of the target video"),
        # AllGather-KV is mutually exclusive with the other two, so a check that looked
        # only at those saw 1 and 1 and passed it through.
        ({"allgather_degree": 2}, "every row of the target video"),
    ],
)
def test_runtime_combinations_that_would_render_the_wrong_video_are_refused(kwargs, message):
    """Each of these renders a plausible video that is not the model.

    A dense attention beside the branch counts everything outside the window twice; a
    sequence-parallel shard gives the frame scan only part of the clip. Neither raises
    on its own, so the check is explicit.
    """
    from vllm_omni.diffusion.models.minimax_h3.vdn_hybrid import validate_hybrid_runtime

    call = {
        "attention_backend": "VDN_WINDOW_ATTN",
        "ulysses_degree": 1,
        "ring_degree": 1,
        "allgather_degree": 1,
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=message):
        validate_hybrid_runtime(**call)


def test_the_supported_runtime_passes():
    from vllm_omni.diffusion.models.minimax_h3.vdn_hybrid import validate_hybrid_runtime

    validate_hybrid_runtime(attention_backend="VDN_WINDOW_ATTN", ulysses_degree=1, ring_degree=1, allgather_degree=1)
