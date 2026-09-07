# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A tensor-parallel head shard of the VDN branch must equal that slice of a full run.

This is the one part of the port with no upstream to compare against: OpenVDN shards
the branch for Ulysses, where a rank owns a head range and is *handed* the beta, gate
and frame mean its sequence owner computed. Our production shard is tensor parallel, so
every rank recomputes those from the full hidden state it already holds and only the
parameters divide.

That is only correct because nothing in the branch mixes heads. The two
``[hidden -> head_dim]`` bottlenecks (alpha's and the output gate's ``down``) are
replicated and the RMSNorm weight is per channel; everything else -- alpha's ``up``,
``A_log``, ``dt_bias``, ``beta_proj``, the short conv, the gate's ``up`` -- is per head.
If any of that were wrong the model would still run and still produce video.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HIDDEN, HEADS, HEAD_DIM = 128, 4, 32
GRID_H, GRID_W = 3, 4
TOKENS_PER_FRAME = GRID_H * GRID_W
NUM_FRAMES = 17
TEXT_LEN = 7


@pytest.fixture(autouse=True)
def exact_fp32_matmul():
    """This file asserts a shard is EXACTLY a slice of the full run, to 1e-5.

    That only holds while the fp32 GEMMs are fp32. Reduced-precision matmul modes are
    global process state, so a test that ran earlier and enabled one would make this
    file fail for a reason that has nothing to do with sharding -- and the two runs
    being compared have different shapes, so the reduced-precision error does not even
    cancel between them.
    """
    saved = torch.get_float32_matmul_precision()
    saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(saved)
    torch.backends.cuda.matmul.allow_tf32 = saved_tf32


def _branch(monkeypatch, *, tp_size, rank):
    """A branch built as rank ``rank`` of a ``tp_size``-way tensor-parallel group."""
    from vllm_omni.diffusion.models.minimax_h3 import vdn_branch

    monkeypatch.setattr(vdn_branch, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(vdn_branch, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(vdn_branch, "get_tensor_model_parallel_rank", lambda: rank)
    # fp32 so the shard comparison measures the sharding and not bf16 rounding: the
    # whole assertion is that a shard is EXACTLY a slice of the full run.
    return vdn_branch.VDNLinearBranch(HIDDEN, HEADS, HEAD_DIM, params_dtype=torch.float32)


def _shard_state(full_state, *, tp_size, rank):
    """The slice of a whole-model state dict this rank owns.

    Sharded through ``VDN_BRANCH_SHARD_AXES``, the same table the weight loaders read,
    so this asserts what it is meant to: that the branch's ARITHMETIC is separable along
    the axes that table declares. A parameter missing from the table is replicated, and
    a wrong entry shows up here as a numerical mismatch rather than as a plausible
    render.
    """
    from vllm_omni.diffusion.models.minimax_h3.vdn_branch import VDN_BRANCH_SHARD_AXES

    local_heads = HEADS // tp_size
    out = {}
    for name, tensor in full_state.items():
        axis = VDN_BRANCH_SHARD_AXES.get(name)
        if axis is None:  # alpha.down, output_gate.down, norm.weight
            out[name] = tensor.clone()
            continue
        unit = 1 if axis == "head" else HEAD_DIM
        out[name] = tensor[rank * local_heads * unit : (rank + 1) * local_heads * unit].clone()
    return out


def test_every_sharded_parameter_is_named_in_the_table():
    """A new per-head parameter that nobody adds to the table would be replicated.

    Replicating a per-head tensor loads and runs: every rank would use head 0..13's
    weights for its own heads. This pins the table against the module's real parameter
    set, so adding a parameter forces a decision about how it divides.
    """
    from vllm_omni.diffusion.models.minimax_h3.vdn_branch import (
        VDN_BRANCH_SHARD_AXES,
        VDNLinearBranch,
    )

    branch = VDNLinearBranch(HIDDEN, HEADS, HEAD_DIM, params_dtype=torch.float32)
    names = {name for name, _ in branch.named_parameters()}
    assert set(VDN_BRANCH_SHARD_AXES) <= names, sorted(set(VDN_BRANCH_SHARD_AXES) - names)
    replicated = names - set(VDN_BRANCH_SHARD_AXES)
    assert replicated == {"alpha.down.weight", "output_gate.down.weight", "norm.weight"}


def _inputs():
    torch.manual_seed(11)
    rows = NUM_FRAMES * TOKENS_PER_FRAME
    make = lambda n: torch.randn(n, HEADS, HEAD_DIM)  # noqa: E731
    return (
        torch.randn(rows, HIDDEN),
        (make(rows), make(rows), make(rows)),
        torch.randn(TEXT_LEN, HIDDEN),
        (make(TEXT_LEN), make(TEXT_LEN), make(TEXT_LEN)),
    )


@pytest.mark.parametrize("tp_size", [2, 4])
def test_head_shards_concatenate_to_the_full_run(monkeypatch, tp_size):
    from vllm_omni.diffusion.models.minimax_h3.vdn_window import window_bounds

    video_x, qkv, text_x, text_qkv = _inputs()
    bounds = window_bounds(NUM_FRAMES)
    call = dict(
        num_frames=NUM_FRAMES,
        tokens_per_frame=TOKENS_PER_FRAME,
        frame_size=(GRID_H, GRID_W),
        bounds=bounds,
        text_x=text_x,
        text_qkv_raw=text_qkv,
        skip_ends=True,
    )

    with monkeypatch.context() as patch:
        full = _branch(patch, tp_size=1, rank=0).eval()
        with torch.no_grad():
            expected = full(video_x, qkv, **call)
    full_state = full.state_dict()

    local_heads = HEADS // tp_size
    shards = []
    for rank in range(tp_size):
        with monkeypatch.context() as patch:
            branch = _branch(patch, tp_size=tp_size, rank=rank).eval()
        branch.load_state_dict(_shard_state(full_state, tp_size=tp_size, rank=rank))
        head_slice = slice(rank * local_heads, (rank + 1) * local_heads)
        with torch.no_grad():
            shards.append(
                branch(
                    video_x,
                    tuple(t[:, head_slice] for t in qkv),
                    **{**call, "text_qkv_raw": tuple(t[:, head_slice] for t in text_qkv)},
                )
            )

    # Each shard emits [rows, local_heads * head_dim]; the ranks' outputs concatenate on
    # the channel axis, which is exactly the input layout to_out_linear consumes as a
    # row-parallel projection.
    torch.testing.assert_close(torch.cat(shards, dim=-1), expected, rtol=1e-5, atol=1e-5)


def test_uneven_head_split_is_refused(monkeypatch):
    """3 ranks over 4 heads would silently drop a head on some rank."""
    with pytest.raises(ValueError, match="divide across"):
        _branch(monkeypatch, tp_size=3, rank=0)
