# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Loading a VDN-H3 artifact: what gets fused, what gets injected, what gets refused.

The failures this guards against all produce a server that starts and generates video:
an adapter delta that never met its tensor, a branch tensor that never reached a module,
a delta placed in the wrong QKV layout, or the artifact applied to a task it was not
trained for. The fixture mirrors the released ``stage-dmd-step-250`` layout -- two
adapters over an exploded branch directory -- at a size that fits in a tmp_path.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HIDDEN, HEADS, HEAD_DIM = 32, 4, 8
INNER = HEADS * HEAD_DIM
BLOCKS, REFINER_BLOCKS, RANK = 2, 1, 4

_SPEC = {
    "base": {"class_name": "MiniMaxH3Transformer3DModel", "source": "ckpts/h3-base"},
    "format_version": 2,
    "transforms": [
        {
            "type": "hybrid_attention",
            "version": 2,
            "config": {
                "anchor_frames": "both",
                "enable_softmax_gate": True,
                "linear_attention": {
                    "a_fp32": True,
                    "bridge": "alpha",
                    "delta_rule": "vdn_solve",
                    "enable_text_state": True,
                    "linear_head_dim": HEAD_DIM,
                    "short_conv": {"targets": ["k", "v"]},
                },
                "softmax_attention": {"chunk": 5, "radius": 1},
            },
        }
    ],
}


def _branch_tensors():
    out = {}
    for block in range(BLOCKS):
        prefix = f"transformer_blocks.{block}.attn"
        out[f"{prefix}.linear_attention.alpha.A_log"] = torch.randn(HEADS)
        out[f"{prefix}.linear_attention.alpha.dt_bias"] = torch.randn(INNER)
        out[f"{prefix}.linear_attention.alpha.down.weight"] = torch.randn(HEAD_DIM, HIDDEN)
        out[f"{prefix}.linear_attention.alpha.up.weight"] = torch.randn(INNER, HEAD_DIM)
        out[f"{prefix}.linear_attention.beta_proj.weight"] = torch.randn(HEADS, HIDDEN)
        out[f"{prefix}.linear_attention.norm.weight"] = torch.randn(HEAD_DIM)
        out[f"{prefix}.linear_attention.output_gate.down.weight"] = torch.randn(HEAD_DIM, HIDDEN)
        out[f"{prefix}.linear_attention.output_gate.up.weight"] = torch.randn(INNER, HEAD_DIM)
        out[f"{prefix}.linear_attention.output_gate.up.bias"] = torch.randn(INNER)
        for projection in ("k", "v"):
            out[f"{prefix}.linear_attention.short_conv.{projection}_sp.weight"] = torch.randn(INNER, 1, 5, 5)
            out[f"{prefix}.linear_attention.short_conv.{projection}_tm.weight"] = torch.randn(INNER, 1, 5)
        out[f"{prefix}.softmax_gate.up.weight"] = torch.randn(HEADS, HIDDEN)
        out[f"{prefix}.softmax_gate.up.bias"] = torch.randn(HEADS)
        out[f"{prefix}.to_out_linear.weight"] = torch.randn(HIDDEN, INNER)
    return out


def _adapter_tensors(name, *, with_mlp=False):
    """A LoRA over the attention projections, in the released key spelling.

    ``attn.orig.*`` on the DiT blocks (VDN wraps the original attention) and plain
    ``attn.*`` on the token refiner, which is never converted.
    """
    out = {}

    def pair(module, in_features, out_features):
        out[f"{module}.lora_A.{name}.weight"] = torch.randn(RANK, in_features) * 0.02
        out[f"{module}.lora_B.{name}.weight"] = torch.randn(out_features, RANK) * 0.02

    for block in range(BLOCKS):
        for projection in ("to_q", "to_k", "to_v"):
            pair(f"transformer_blocks.{block}.attn.orig.{projection}", HIDDEN, INNER)
        pair(f"transformer_blocks.{block}.attn.orig.to_out.0", INNER, HIDDEN)
        if with_mlp:
            pair(f"transformer_blocks.{block}.ff.net.0.proj", HIDDEN, 2 * HIDDEN)
            pair(f"transformer_blocks.{block}.ff.net.2", HIDDEN, HIDDEN)
    for block in range(REFINER_BLOCKS):
        for projection in ("to_q", "to_k", "to_v"):
            pair(f"token_refiner.refiner_blocks.{block}.attn.{projection}", HIDDEN, INNER)
        pair(f"token_refiner.refiner_blocks.{block}.attn.to_out.0", INNER, HIDDEN)
    return out


@pytest.fixture
def artifact(tmp_path):
    torch.manual_seed(0)
    root = tmp_path / "stage-dmd-step-250"
    (root / "linear_branch").mkdir(parents=True)
    (root / "model_spec.json").write_text(json.dumps(_SPEC))
    (root / "metadata.json").write_text(
        json.dumps({"metadata": {"turbo_num_steps": 8, "video_shift": 12.0, "audio_shift": 3.0}})
    )
    save_file(_branch_tensors(), str(root / "linear_branch" / "model.safetensors"))
    for name, with_mlp in (("default", False), ("turbo", True)):
        directory = root / "adapters" / name
        directory.mkdir(parents=True)
        save_file(_adapter_tensors(name, with_mlp=with_mlp), str(directory / "adapter_model.safetensors"))
    return root


def _load(root):
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpoint

    return VDNCheckpoint.from_path(root, head_dim=HEAD_DIM, num_blocks=BLOCKS)


def _base_stream():
    """Every parameter the fixture's adapters edit, in the native (grouped QKV) spelling.

    Needed whenever a test wants the adapter-completeness check to PASS so that a
    different check is the one under test.
    """
    stream = []
    for block in range(BLOCKS):
        stream.append((f"blocks.{block}.attn.qkv_proj.weight", torch.zeros(3 * INNER, HIDDEN)))
        stream.append((f"blocks.{block}.attn.out_proj.weight", torch.zeros(HIDDEN, INNER)))
        stream.append((f"blocks.{block}.mlp.fc1.weight", torch.zeros(2 * HIDDEN, HIDDEN)))
        stream.append((f"blocks.{block}.mlp.fc2.weight", torch.zeros(HIDDEN, HIDDEN)))
    for block in range(REFINER_BLOCKS):
        prefix = f"token_refiner.blocks.{block}.attn"
        stream.append((f"{prefix}.qkv_proj.weight", torch.zeros(3 * INNER, HIDDEN)))
        stream.append((f"{prefix}.out_proj.weight", torch.zeros(HIDDEN, INNER)))
    return stream


def test_reads_the_transform_off_the_spec(artifact):
    checkpoint = _load(artifact)
    assert checkpoint.spec.chunk == 5
    assert checkpoint.spec.radius == 1
    assert checkpoint.spec.anchor_frames == "both"
    assert checkpoint.spec.short_conv == ("k", "v")
    assert checkpoint.spec.enable_text_state is True
    assert checkpoint.adapters == ("default", "turbo")
    assert checkpoint.turbo_num_steps == 8


def test_a_directory_that_is_not_a_vdn_artifact_is_not_claimed(tmp_path):
    """An unrelated --lora-path must stay on the dynamic LoRA route."""
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    assert _load(tmp_path) is None


def test_branch_tensors_are_injected_under_the_hybrid_level(artifact):
    checkpoint = _load(artifact)
    produced = dict(checkpoint.apply([]))
    assert "blocks.0.attn.vdn.linear_attention.alpha.A_log" in produced
    assert "blocks.1.attn.vdn.softmax_gate.up.bias" in produced
    assert "blocks.0.attn.vdn.to_out_linear.weight" in produced
    assert all(name.startswith("blocks.") for name in produced)


def test_both_adapters_stack_on_the_same_projection(artifact):
    """stage-dmd carries default AND turbo, and both edit the same projections.

    Compared against the same artifact with turbo removed: applying only one adapter is
    a different model, and it is a difference nothing downstream would notice.
    """
    import shutil

    base = torch.zeros(HIDDEN, INNER)
    key = "blocks.0.attn.out_proj.weight"
    both = dict(_load(artifact).apply([(key, base.clone())]))[key]

    shutil.rmtree(artifact / "adapters" / "turbo")
    only_default = _load(artifact)
    assert only_default.adapters == ("default",)
    single = dict(only_default.apply([(key, base.clone())]))[key]

    assert torch.count_nonzero(single) > 0
    assert not torch.allclose(both, single)


def test_grouped_qkv_base_gets_one_interleaved_delta(artifact):
    """A native partition base stores q/k/v as one grouped matrix.

    The three separately-trained projections have to be folded back into head-group
    order; a delta concatenated the obvious way loads fine and scrambles the heads.
    """
    checkpoint = _load(artifact)
    base = torch.zeros(3 * INNER, HIDDEN)
    fused = dict(checkpoint.apply([("blocks.0.attn.qkv_proj.weight", base.clone())]))
    delta = fused["blocks.0.attn.qkv_proj.weight"]
    assert delta.shape == base.shape
    # Grouped layout is [q_head0, k_head0, v_head0, q_head1, ...]: every head group of
    # 3*head_dim rows carries all three projections, so no group is empty.
    groups = delta.reshape(HEADS, 3 * HEAD_DIM, HIDDEN)
    assert (groups.abs().sum(dim=(1, 2)) > 0).all()


def test_diffusers_base_gets_per_projection_deltas(artifact):
    """The released H3 transformer/ stores to_q/to_k/to_v separately."""
    checkpoint = _load(artifact)
    stream = [
        (f"transformer_blocks.0.attn.{projection}.weight", torch.zeros(INNER, HIDDEN))
        for projection in ("to_q", "to_k", "to_v")
    ]
    fused = dict(checkpoint.apply(stream))
    for projection in ("to_q", "to_k", "to_v"):
        assert torch.count_nonzero(fused[f"transformer_blocks.0.attn.{projection}.weight"]) > 0


def test_an_edit_that_never_met_a_tensor_is_fatal(artifact):
    """The base stream skipping a projection must not pass silently."""
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpointError

    checkpoint = _load(artifact)
    list(checkpoint.apply([("blocks.0.attn.out_proj.weight", torch.zeros(HIDDEN, INNER))]))
    with pytest.raises(VDNCheckpointError, match="never met a checkpoint tensor"):
        checkpoint.validate_fully_applied()


def test_a_branch_tensor_that_never_reached_a_module_is_fatal(artifact):
    """enable_vdn_branch not having run leaves the branch unpopulated but running."""
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpointError

    checkpoint = _load(artifact)
    produced = dict(checkpoint.apply(_base_stream()))
    # The adapters all met their tensors, so this isolates the injection check.
    with pytest.raises(VDNCheckpointError, match="never reached the model"):
        checkpoint.validate_fully_applied(loaded=set())
    assert any(name.startswith("blocks.0.attn.vdn.") for name in produced)


def test_a_complete_load_closes_cleanly(artifact):
    """The positive case, so the two failure tests above are not the only coverage."""
    checkpoint = _load(artifact)
    produced = dict(checkpoint.apply(_base_stream()))
    checkpoint.validate_fully_applied(loaded=set(produced))


def test_applying_twice_is_refused(artifact):
    """A second stream would fuse nothing and then pass its own completeness check."""
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpointError

    checkpoint = _load(artifact)
    list(checkpoint.apply([]))
    with pytest.raises(VDNCheckpointError, match="already been fused"):
        list(checkpoint.apply([]))


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_untrained_tasks_are_refused_by_default(artifact, task):
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError):
        _load(artifact).check_task(task)


def test_the_evaluation_flag_opens_fl2va_but_never_ref2va(artifact, monkeypatch):
    """ref2va is a different DiT partition, not a distribution question.

    H3 serves it from ``transformer_ref``, whose tensor names match the base VDN adapts
    and whose weights do not, and the pipeline builds the branch on the primary
    transformer only. Letting the evaluation flag through would run the window against a
    dense model with no complement -- output, but with every frame outside the window
    silently dropped.
    """
    from vllm_omni.diffusion.models.minimax_h3.vdn import (
        VDN_ALLOW_UNTRAINED_TASKS_ENV,
    )
    from vllm_omni.errors import OmniClientError

    monkeypatch.setenv(VDN_ALLOW_UNTRAINED_TASKS_ENV, "1")
    checkpoint = _load(artifact)
    checkpoint.check_task("fl2va")  # same partition: evaluable
    with pytest.raises(OmniClientError, match="different DiT partition"):
        checkpoint.check_task("ref2va")


class _Sampling:
    """The fields VDN's request guard reads off the real sampling object."""

    def __init__(self, *, steps=8, extra=None, lora_request=None):
        self.num_inference_steps = steps
        self.extra_args = extra or {}
        self.lora_request = lora_request


def test_the_distilled_shifts_are_read_from_the_REQUEST(artifact):
    """The shifts that matter arrive per request and override the pipeline defaults.

    An earlier version compared the pipeline's default against the constant -- two
    constants, always equal -- so the guard never fired on any request at all. Passing
    matching defaults with a mismatched request must still raise.
    """
    from vllm_omni.errors import OmniClientError

    checkpoint = _load(artifact)
    checkpoint.check_request(_Sampling(), video_shift=12.0, audio_shift=3.0)
    for extra in ({"flow_shift": 6.0}, {"audio_flow_shift": 1.0}):
        with pytest.raises(OmniClientError, match="requires"):
            checkpoint.check_request(_Sampling(extra=extra), video_shift=12.0, audio_shift=3.0)


def test_a_request_off_the_distilled_step_count_is_refused(artifact):
    """stage-dmd declares turbo_num_steps=8; sampling it at 50 is off its schedule."""
    from vllm_omni.errors import OmniClientError

    checkpoint = _load(artifact)
    assert checkpoint.turbo_num_steps == 8
    with pytest.raises(OmniClientError, match="num_inference_steps=8"):
        checkpoint.check_request(_Sampling(steps=50), video_shift=12.0, audio_shift=3.0)


def test_a_per_request_lora_is_refused(artifact):
    """lora_is_fused skips the dynamic LoRA manager, so nothing would apply it."""
    from vllm_omni.errors import OmniClientError

    checkpoint = _load(artifact)
    with pytest.raises(OmniClientError, match="per-request lora"):
        checkpoint.check_request(_Sampling(lora_request=object()), video_shift=12.0, audio_shift=3.0)


def test_an_unsupported_delta_rule_is_refused_at_load(artifact):
    """The branch's own guard is unreachable from serving: branch_kwargs never forwarded
    delta_rule, so a spec asking for another rule silently ran vdn_solve."""
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpointError

    spec = json.loads((artifact / "model_spec.json").read_text())
    spec["transforms"][0]["config"]["linear_attention"]["delta_rule"] = "sana_scaled"
    (artifact / "model_spec.json").write_text(json.dumps(spec))
    with pytest.raises(VDNCheckpointError, match="delta_rule"):
        _load(artifact)


@pytest.mark.parametrize("field, value", [("chunk", 7), ("radius", 2), ("anchor_frames", "columns")])
def test_a_non_released_window_is_refused_at_load(artifact, field, value):
    """The window backend is built before this spec is read, so a non-default geometry
    would leave the softmax half on the released values while the branch used these."""
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpointError

    spec = json.loads((artifact / "model_spec.json").read_text())
    config = spec["transforms"][0]["config"]
    if field == "anchor_frames":
        config["anchor_frames"] = value
    else:
        config["softmax_attention"][field] = value
    (artifact / "model_spec.json").write_text(json.dumps(spec))
    with pytest.raises(VDNCheckpointError, match="non-released window"):
        _load(artifact)


def test_a_head_dim_mismatch_is_refused(artifact):
    """The branch shares the attention's QKV projections, so the two cannot differ."""
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpoint, VDNCheckpointError

    with pytest.raises(VDNCheckpointError, match="head_dim"):
        VDNCheckpoint.from_path(artifact, head_dim=HEAD_DIM * 2, num_blocks=BLOCKS)


def test_a_missing_block_is_refused(artifact):
    """A gap leaves that block's dense attention with no complement."""
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpoint, VDNCheckpointError

    with pytest.raises(VDNCheckpointError, match="the model has"):
        VDNCheckpoint.from_path(artifact, head_dim=HEAD_DIM, num_blocks=BLOCKS + 1)


def test_a_non_lora_tensor_in_an_adapter_is_refused(artifact):
    """FastH3-style full-rank .diff tensors would be silently skipped otherwise."""
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpointError

    path = artifact / "adapters" / "default" / "adapter_model.safetensors"
    tensors = _adapter_tensors("default")
    tensors["transformer_blocks.0.norm1.diff"] = torch.randn(HIDDEN)
    save_file(tensors, str(path))
    with pytest.raises(VDNCheckpointError, match="non-LoRA tensors"):
        _load(artifact)


def _base_with_stamp(tmp_path, adapters):
    """A baked partition: transformer/config.json carrying the bake stamp."""
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDN_BAKED_STAMP_KEY

    base = tmp_path / "baked-partition"
    (base / "transformer").mkdir(parents=True)
    (base / "transformer" / "config.json").write_text(
        json.dumps({"hidden_size": HIDDEN, VDN_BAKED_STAMP_KEY: {"adapters": list(adapters)}})
    )
    return base


def test_baked_base_plus_an_artifact_that_still_has_adapters_is_refused(artifact, tmp_path):
    """The adapters would be applied a second time on top of themselves.

    Nothing downstream notices: the shapes match, the fusion succeeds, the server starts
    and renders. Only the picture is wrong, and only against a reference nobody has.
    """
    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpointError, check_bake_agreement

    checkpoint = _load(artifact)
    assert checkpoint.adapters == ("default", "turbo")
    with pytest.raises(VDNCheckpointError, match="applies them twice"):
        check_bake_agreement(checkpoint, _base_with_stamp(tmp_path, ["default", "turbo"]))


def test_raw_base_plus_a_branch_only_artifact_is_refused(artifact, tmp_path):
    """The mirror mistake: the LoRAs the branch was trained with never get applied."""
    import shutil

    from vllm_omni.diffusion.models.minimax_h3.vdn import VDNCheckpointError, check_bake_agreement

    shutil.rmtree(artifact / "adapters")
    checkpoint = _load(artifact)
    assert checkpoint.adapters == ()
    raw_base = tmp_path / "raw-partition"
    (raw_base / "transformer").mkdir(parents=True)
    (raw_base / "transformer" / "config.json").write_text(json.dumps({"hidden_size": HIDDEN}))
    with pytest.raises(VDNCheckpointError, match="would not be applied at all"):
        check_bake_agreement(checkpoint, raw_base)


def test_the_two_correct_pairings_pass(artifact, tmp_path):
    import shutil

    from vllm_omni.diffusion.models.minimax_h3.vdn import check_bake_agreement

    raw_base = tmp_path / "raw"
    (raw_base / "transformer").mkdir(parents=True)
    (raw_base / "transformer" / "config.json").write_text(json.dumps({"hidden_size": HIDDEN}))
    check_bake_agreement(_load(artifact), raw_base)  # raw base + full artifact

    shutil.rmtree(artifact / "adapters")
    check_bake_agreement(_load(artifact), _base_with_stamp(tmp_path, ["default", "turbo"]))


def test_an_unstamped_base_is_treated_as_raw(artifact, tmp_path):
    """A base with no stamp is assumed unbaked -- the stamp is the contract, and its
    absence must not silently permit the branch-only artifact."""
    from vllm_omni.diffusion.models.minimax_h3.vdn import baked_adapters_of

    assert baked_adapters_of(tmp_path / "does-not-exist") is None
    plain = tmp_path / "plain"
    (plain / "transformer").mkdir(parents=True)
    (plain / "transformer" / "config.json").write_text("{}")
    assert baked_adapters_of(plain) is None
