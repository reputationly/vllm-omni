# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Loading an OpenVDN VDN-H3 checkpoint over a MiniMax-H3 base.

VDN-H3 ships as an increment, not a release: ~5 GB of hybrid-attention weights over the
72 GB base H3 transformer, laid out as an exploded checkpoint directory::

    stage-dmd-step-250/
      model_spec.json                  base reference, the hybrid transform, adapter specs
      metadata.json                    the training recipe (steps, shifts)
      linear_branch/model.safetensors  the branch, the softmax gate, to_out_linear
      adapters/default/                a rank-64 LoRA on the attention projections
      adapters/turbo/                  the 8-step DMD adapter (stage-dmd only)

Two things happen here. The branch tensors are INJECTED -- they name parameters the base
transformer does not have, so they join the weight stream rather than folding into it.
The adapters are FUSED, because every one of them edits a projection the base does
provide, and a request-switchable LoRA cannot express an architecture change that the
branch weights were trained jointly with: serving base H3 with the branch, or the branch
without its adapters, is a different model either way.

Both checkpoint spellings of the base are accepted. H3's released ``transformer/`` is
diffusers-named with separate ``to_q``/``to_k``/``to_v``; our own partition artifacts are
natively named with one grouped ``qkv_proj``. VDN's ``model_spec.json`` points at the
former, our deployments mostly carry the latter, and a delta has to land in the right
layout for whichever arrives -- so the fusion resolves each streamed name the same way
``load_weights`` does instead of assuming one.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from safetensors import safe_open
from vllm.logger import init_logger

from vllm_omni.diffusion.models.minimax_h3.vdn_branch import VDN_DELTA_RULE
from vllm_omni.diffusion.models.minimax_h3.vdn_window import (
    ANCHOR_FRAME_MODES,
    VDN_ANCHOR_FRAMES,
    VDN_CHUNK,
    VDN_RADIUS,
)
from vllm_omni.errors import OmniClientError

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel

logger = init_logger(__name__)

MODEL_SPEC_FILE = "model_spec.json"
METADATA_FILE = "metadata.json"
BRANCH_DIR = "linear_branch"
ADAPTERS_DIR = "adapters"
# Written by the curve projection: the affine intercept, already multiplied through
# lora_B, as one fp32 vector per AdaLN module.
_DIFF_B_SUFFIX = ".diff_b"

# The transform this loader understands, as ``model_spec.json`` names it.
VDN_TRANSFORM_TYPE = "hybrid_attention"
VDN_TRANSFORM_VERSION = 2
VDN_BASE_CLASS = "MiniMaxH3Transformer3DModel"

# Only the text-to-video-and-audio path was trained. fl2va and ref2va put reference
# media in the prefix that the branch never saw during training.
VDN_SUPPORTED_TASKS = frozenset({"t2va"})
# Opt-in, per process, to render an untrained task anyway. For evaluation only; see
# check_task.
VDN_ALLOW_UNTRAINED_TASKS_ENV = "VLLM_OMNI_VDN_ALLOW_UNTRAINED_TASKS"
# ref2va runs a DIFFERENT DiT partition (``transformer_ref/``) than the one VDN's
# adapters were trained against (``transformer/``): identical tensor NAMES, different
# weights. An earlier version refused it outright on the grounds that a delta computed
# for one matrix is meaningless on another.
#
# That was too strong, and it was asserted without measuring. Measured 2026-09-08 in
# float64 over attention, MLP and embedder tensors spread across the depth: cosine
# 0.99953, relative L2 3.1% (consistency check 1 - rel^2/2 matches the cosine exactly).
# Ref2VA is a LIGHT FINE-TUNE of FL2VA, not an independently trained model, so the
# trained branch is a plausible -- though unvalidated -- initialisation there.
#
# So ref2va is treated like fl2va: refused by default, openable for evaluation. What
# makes it a separate constant is that it additionally needs the branch built on the
# SECOND DiT instance, which the pipeline must arrange.
VDN_TRANSFERRED_TASKS = frozenset({"ref2va"})
VDN_PARTITION_COSINE = 0.99953  # transformer/ vs transformer_ref/, measured
# Stamp that ``bake_vdn_adapters.py`` writes into the baked partition's transformer
# config.json, naming the adapters it folded in. It is what lets the loader tell a baked
# base from a raw one; the offline quantizer copies config.json through, so it survives
# INT8 conversion.
VDN_BAKED_STAMP_KEY = "vdn_baked_adapters"
# The shifts every released VDN schedule was distilled at (``metadata.json``).
VDN_VIDEO_SHIFT = 12.0
VDN_AUDIO_SHIFT = 3.0

_LORA_A = re.compile(r"\.lora_A\.[^.]+\.weight$")
_LORA_B = re.compile(r"\.lora_B\.[^.]+\.weight$")


class VDNCheckpointError(ValueError):
    """The artifact is a VDN checkpoint, but it cannot be applied as one."""


@dataclass(frozen=True)
class VDNTransformSpec:
    """The hybrid architecture, read off the checkpoint rather than configured.

    These describe how the weights were TRAINED. Serving any of them differently gives
    a model that runs and renders and is not the one that was released, so they are read
    from ``model_spec.json`` and passed to ``enable_vdn_branch`` verbatim.
    """

    chunk: int
    radius: int
    anchor_frames: str
    short_conv: tuple[str, ...]
    enable_text_state: bool
    delta_rule: str
    linear_head_dim: int

    @classmethod
    def from_model_spec(cls, spec: Mapping[str, Any]) -> VDNTransformSpec:
        transforms = spec.get("transforms") or []
        if len(transforms) != 1:
            raise VDNCheckpointError(f"expected exactly one transform in {MODEL_SPEC_FILE}, got {len(transforms)}")
        transform = transforms[0]
        if transform.get("type") != VDN_TRANSFORM_TYPE or transform.get("version") != VDN_TRANSFORM_VERSION:
            raise VDNCheckpointError(
                f"unsupported transform {transform.get('type')!r} v{transform.get('version')}; "
                f"this loader implements {VDN_TRANSFORM_TYPE!r} v{VDN_TRANSFORM_VERSION}"
            )
        config = transform["config"]
        softmax, linear = config["softmax_attention"], config["linear_attention"]
        anchor_frames = config["anchor_frames"]
        if anchor_frames not in ANCHOR_FRAME_MODES:
            raise VDNCheckpointError(f"unknown anchor_frames {anchor_frames!r}")
        if not config.get("enable_softmax_gate", False):
            raise VDNCheckpointError(
                "this checkpoint was trained without the softmax mass gate; the port always "
                "applies one, so it would scale the softmax branch the checkpoint did not scale"
            )
        parsed = cls(
            chunk=int(softmax["chunk"]),
            radius=int(softmax["radius"]),
            anchor_frames=anchor_frames,
            short_conv=tuple(linear["short_conv"]["targets"]),
            enable_text_state=bool(linear["enable_text_state"]),
            delta_rule=str(linear["delta_rule"]),
            linear_head_dim=int(linear["linear_head_dim"]),
        )

        # Refuse anything the serving path cannot actually honour, HERE, before a module
        # is built. Two of these guards used to live downstream and were unreachable:
        #
        #   delta_rule  VDNLinearBranch raises on a rule it does not implement, but the
        #               value never reached it -- branch_kwargs did not forward it, so a
        #               spec asking for sana_scaled silently ran vdn_solve.
        #   the window  MiniMaxH3Attention builds its Attention (and with it the window
        #               backend, which carries chunk/radius/anchor_frames) in __init__;
        #               enable_vdn_branch runs later and configures only the branch. A
        #               non-default geometry would leave the softmax half on the released
        #               defaults while the branch used the checkpoint's -- and the two
        #               halves are designed to be an exact partition of the sequence.
        #
        # Supporting a different window means plumbing this spec into the backend's
        # construction. Until that exists, refusing is the honest behaviour.
        if parsed.delta_rule != VDN_DELTA_RULE:
            raise VDNCheckpointError(
                f"{MODEL_SPEC_FILE} asks for delta_rule={parsed.delta_rule!r}; this port implements "
                f"{VDN_DELTA_RULE!r} only, and the branch would silently run that instead."
            )
        released = (
            ("chunk", parsed.chunk, VDN_CHUNK),
            ("radius", parsed.radius, VDN_RADIUS),
            ("anchor_frames", parsed.anchor_frames, VDN_ANCHOR_FRAMES),
        )
        differing = [
            f"{name}={value!r} (released {expected!r})" for name, value, expected in released if value != expected
        ]
        if differing:
            raise VDNCheckpointError(
                f"{MODEL_SPEC_FILE} declares a non-released window: {', '.join(differing)}. The "
                "window backend is constructed before this spec is read, so the softmax half "
                "would keep the released geometry while the linear branch used the checkpoint's, "
                "and the two would no longer partition the sequence."
            )
        return parsed

    def branch_kwargs(self) -> dict[str, Any]:
        """What ``MiniMaxH3Attention.enable_vdn_branch`` needs."""
        return {
            "chunk": self.chunk,
            "radius": self.radius,
            "anchor_frames": self.anchor_frames,
            "short_conv": self.short_conv,
            "enable_text_state": self.enable_text_state,
        }


@dataclass
class _Patch:
    """The low-rank pairs an adapter contributes to one native parameter.

    Keyed by layout slot so a grouped QKV parameter can collect the three projections
    that were trained separately, and so several adapters can stack on one parameter --
    ``stage-dmd`` merges both ``default`` and ``turbo`` into the same projections.
    """

    layout: str
    pairs: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    # Curve-projected AdaLN adapters additionally carry an fp32 intercept, which lands on
    # the pruned parameter's ``folded_bias`` rather than on a weight (see
    # ``tools/minimax_h3/vdn_curve_projection.py``). Several adapters may stack here too.
    biases: list[torch.Tensor] = field(default_factory=list)


def _strip_hybrid_level(module: str) -> str:
    """``attn.orig.to_q`` -> ``attn.to_q``.

    VDN wraps each block's attention in a HybridAttention that keeps the original module
    as ``.orig``, so its adapters are named one level down. Our attention IS the
    original, so the level is dropped rather than mapped.
    """
    return module.replace(".attn.orig.", ".attn.", 1)


class VDNCheckpoint:
    """A VDN artifact, ready to be folded into a MiniMax-H3 weight stream."""

    def __init__(
        self,
        *,
        source: Path,
        spec: VDNTransformSpec,
        metadata: Mapping[str, Any],
        injections: dict[str, torch.Tensor],
        patches: dict[str, _Patch],
        head_dim: int,
        adapters: tuple[str, ...],
    ) -> None:
        self._source = source
        self.spec = spec
        self.metadata = dict(metadata)
        self._injections = injections
        self._patches = patches
        self._head_dim = head_dim
        self.adapters = adapters
        self._applied: set[str] = set()
        self._injected: set[str] = set()

    @property
    def source(self) -> Path:
        return self._source

    @property
    def turbo_num_steps(self) -> int | None:
        """The step count the DMD adapter was distilled at, if this is that stage."""
        value = self.metadata.get("metadata", {}).get("turbo_num_steps")
        return None if value is None else int(value)

    # ---- discovery -----------------------------------------------------------------

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        head_dim: int,
        num_blocks: int,
        only_adapters: Sequence[str] | None = None,
    ) -> VDNCheckpoint | None:
        """Read the artifact at ``path``, or return None if it is not a VDN checkpoint.

        None rather than an error: an unrelated ``--lora-path`` must stay on the dynamic
        LoRA route. Once the directory does look like VDN, every later problem raises.

        ``only_adapters`` restricts which adapters are read. SERVING must never pass it:
        the released inference path merges every adapter, and applying a subset is a
        different model. It exists for the bake tool, whose one legitimate use is a base
        whose shapes cannot carry a particular adapter -- an r8-pruned partition
        refactorises AdaLN to rank 8, and the ``turbo`` adapter's AdaLN delta does not
        fit it (measured: 99.8% of the delta lies outside the pruned basis, so there is
        no honest projection either). Naming the subset is then a deliberate statement
        about what the artifact is, and the stamp records it.
        """
        root = Path(path)
        spec_file = root / MODEL_SPEC_FILE
        branch_file = root / BRANCH_DIR / "model.safetensors"
        if not spec_file.is_file() or not branch_file.is_file():
            return None

        spec_json = json.loads(spec_file.read_text())
        base = spec_json.get("base") or {}
        if base.get("class_name") != VDN_BASE_CLASS:
            raise VDNCheckpointError(f"{root} is built on {base.get('class_name')!r}, not {VDN_BASE_CLASS!r}")
        spec = VDNTransformSpec.from_model_spec(spec_json)
        if spec.linear_head_dim != head_dim:
            raise VDNCheckpointError(
                f"{root} trained its linear branch at head_dim={spec.linear_head_dim}, but this "
                f"model's attention head_dim is {head_dim}; the branch shares the attention's "
                "QKV projections, so the two cannot differ"
            )

        metadata_file = root / METADATA_FILE
        metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}

        injections = _read_branch(branch_file, num_blocks=num_blocks)
        patches, adapters = _read_adapters(root / ADAPTERS_DIR, only=only_adapters)
        logger.info(
            "VDN checkpoint %s: %d branch tensors, %d adapters (%s) editing %d parameters; "
            "window chunk=%d radius=%d anchors=%s",
            root,
            len(injections),
            len(adapters),
            ", ".join(adapters) or "none",
            len(patches),
            spec.chunk,
            spec.radius,
            spec.anchor_frames,
        )
        return cls(
            source=root,
            spec=spec,
            metadata=metadata,
            injections=injections,
            patches=patches,
            head_dim=head_dim,
            adapters=adapters,
        )

    # ---- fusion --------------------------------------------------------------------

    def _delta(self, patch: _Patch, slot: str, device: torch.device) -> torch.Tensor | None:
        """Sum of ``B @ A`` over every adapter contributing to one slot.

        Computed in fp32 and summed there: the released adapters all carry alpha equal
        to rank, so the merge is unscaled, and stacking two rank-64 updates in bf16
        would round twice for no reason.
        """
        contributions = patch.pairs.get(slot)
        if not contributions:
            return None
        delta = None
        for lora_a, lora_b in contributions:
            product = lora_b.to(device, torch.float32) @ lora_a.to(device, torch.float32)
            delta = product if delta is None else delta.add_(product)
        return delta

    def fuse(self, native_name: str, slot: str | None, weight: torch.Tensor) -> torch.Tensor:
        """``weight`` with this artifact's adapters added.

        ``slot`` is the QKV projection this tensor is, when the stream carries the three
        separately; None when it is a whole parameter (including a grouped QKV one).
        """
        patch = self._patches.get(native_name)
        if patch is None:
            return weight
        device = weight.device if weight.is_cuda else torch.device("cpu")

        if patch.layout == _BIAS_LAYOUT:
            # A curve-projected AdaLN intercept. fp32 throughout and added there: the
            # pruned forward adds folded_bias in fp32 for the same reason -- it carries
            # most of the modulation, and rounding it to bf16 before the sum loses more
            # than the projection itself does.
            # Checked BEFORE the accumulation, or a mismatched residual raises a bare
            # RuntimeError out of add_ and this message never runs.
            for residual in patch.biases:
                if residual.shape != weight.shape:
                    raise VDNCheckpointError(
                        f"VDN bias residual for {native_name} has shape {tuple(residual.shape)}, "
                        f"parameter is {tuple(weight.shape)}"
                    )
            delta = torch.zeros_like(weight, dtype=torch.float32, device=device)
            for residual in patch.biases:
                delta.add_(residual.to(device, torch.float32))
            self._applied.add(f"{native_name}:{_PLAIN_SLOT}")
            return delta.add_(weight.to(device, torch.float32)).to(weight.dtype)

        if patch.layout == _QKV_LAYOUT:
            if slot is not None:
                delta = self._delta(patch, slot, device)
                self._applied.add(f"{native_name}:{slot}")
            else:
                from vllm_omni.diffusion.models.minimax_h3.fasth3 import _place_in_grouped_qkv

                per_slot = {name: self._delta(patch, name, device) for name in _QKV_SLOTS}
                missing = sorted(name for name, value in per_slot.items() if value is None)
                if missing:
                    raise VDNCheckpointError(
                        f"{native_name} is one grouped QKV parameter but the adapters only cover "
                        f"{sorted(set(_QKV_SLOTS) - set(missing))}; a partial fuse would edit q "
                        "and leave k/v at the base"
                    )
                delta = _place_in_grouped_qkv(per_slot, head_dim=self._head_dim)
                self._applied.update(f"{native_name}:{name}" for name in _QKV_SLOTS)
        else:
            delta = self._delta(patch, _PLAIN_SLOT, device)
            if delta is not None and patch.layout == _SWAP_HALVES_LAYOUT:
                from vllm_omni.diffusion.models.minimax_h3.fasth3 import _swap_halves

                # The diffusers export packs the fused feed-forward projection
                # value-first while H3's native fc1 is gate-first.
                delta = _swap_halves(delta)
            self._applied.add(f"{native_name}:{_PLAIN_SLOT}")

        if delta is None:
            return weight
        if delta.shape != weight.shape:
            raise VDNCheckpointError(
                f"VDN delta for {native_name} has shape {tuple(delta.shape)}, parameter is {tuple(weight.shape)}"
            )
        return delta.add_(weight.to(device, non_blocking=True)).to(weight.dtype)

    def fuse_stream(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Fuse the adapters into the streamed base. Adapters only -- no injections.

        This is what BAKING wants: a base checkpoint with the adapters folded in, which
        can then be quantized offline. The branch tensors are not part of that; they stay
        a separate artifact because they name parameters the base does not have and
        because quantizing them is a different question (they carry the fp32-sensitive
        paths). Serving reaches this through ``apply``, which appends them.
        """
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
            _diffusers_qkv_target,
            _diffusers_to_partition_name,
        )

        if self._applied or self._injected:
            raise VDNCheckpointError(f"{self._source} has already been fused into this checkpoint")

        for source_name, weight in weights:
            if source_name in self._injections:
                raise VDNCheckpointError(
                    f"the base checkpoint already provides {source_name}, which this artifact "
                    "injects; the base would be silently discarded"
                )
            # Resolve exactly the way load_weights will, so a delta lands in whichever
            # layout this base is stored in rather than in the one we guessed.
            qkv_target = _diffusers_qkv_target(source_name)
            if qkv_target is not None:
                native, slot = qkv_target
            else:
                native, slot = _diffusers_to_partition_name(source_name), None
            yield source_name, self.fuse(native, slot, weight)

    def apply(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Fuse the adapters into the streamed base, then append the branch tensors."""
        yield from self.fuse_stream(weights)
        for name, tensor in self._injections.items():
            self._injected.add(name)
            yield name, tensor

    def validate_fully_applied(self, loaded: Iterable[str] | None = None, *, injections_expected: bool = True) -> None:
        """Close the load: every edit met a parameter and every injection arrived.

        A delta that never met its tensor is the failure that matters -- the server
        would come up, generate video, and simply not be the released model. The branch
        tensors need the stronger check: they land on modules the base transformer does
        not have, so if ``enable_vdn_branch`` never ran, ``load_weights`` only logs a
        skip and the layer would serve an unpopulated branch.
        """
        expected = {
            f"{name}:{slot}"
            for name, patch in self._patches.items()
            for slot in (_QKV_SLOTS if patch.layout == _QKV_LAYOUT else (_PLAIN_SLOT,))
        }
        missing = sorted(expected - self._applied)
        if missing:
            raise VDNCheckpointError(f"{len(missing)} VDN adapter edits never met a checkpoint tensor: {missing[:5]}")
        arrived = self._injected if loaded is None else set(loaded)
        absent = sorted(set(self._injections) - arrived) if injections_expected else []
        if absent:
            raise VDNCheckpointError(
                f"{len(absent)} VDN branch tensors never reached the model: {absent[:5]}. "
                "enable_vdn_branch must run before load_weights."
            )
        self._patches.clear()
        self._injections.clear()

    # ---- serving contract ----------------------------------------------------------

    def check_task(self, task: str) -> None:
        if task in VDN_SUPPORTED_TASKS:
            return
        if os.environ.get(VDN_ALLOW_UNTRAINED_TASKS_ENV) == "1":
            if task in VDN_TRANSFERRED_TASKS:
                logger.warning_once(
                    "%s=1: serving task=%r on a VDN checkpoint whose adapters were trained "
                    "against a different DiT partition. transformer_ref is a light fine-tune "
                    "of transformer (cosine %.5f, relative L2 3.1%%), so the transfer is "
                    "plausible -- but it is UNVALIDATED, and nothing downstream can tell you "
                    "it went wrong.",
                    VDN_ALLOW_UNTRAINED_TASKS_ENV,
                    task,
                    VDN_PARTITION_COSINE,
                )
                return
            # The escape hatch exists so the fl2va/ref2va question can be ANSWERED --
            # structurally they run (reference media sits in the prefix, which the window
            # keeps dense), so the only way to know is to render and look. It is opt-in
            # per process and warns on every request because the output is off the
            # distribution the branch was trained on, and nothing downstream can tell.
            logger.warning_once(
                "%s=1: serving task=%r on a VDN checkpoint that only trained %s. The linear "
                "branch was trained with the prompt as its only condition and has never seen "
                "reference media in the prefix; treat the output as an experiment, not a "
                "qualified configuration.",
                VDN_ALLOW_UNTRAINED_TASKS_ENV,
                task,
                # A string, not the sorted list: warning_once caches on its arguments,
                # so an unhashable one raises TypeError from inside the logger and the
                # request dies before it ever reaches generation.
                ", ".join(sorted(VDN_SUPPORTED_TASKS)),
            )
            return
        raise OmniClientError(
            f"VDN-H3 released {sorted(VDN_SUPPORTED_TASKS)} only, got task={task!r}. Its "
            "linear branch was trained with the prompt as its only condition; fl2va and "
            "ref2va put reference media in the prefix it never saw. Set "
            f"{VDN_ALLOW_UNTRAINED_TASKS_ENV}=1 to render one anyway for evaluation."
        )

    def check_request(self, sampling: Any, *, video_shift: float, audio_shift: float) -> None:
        """Refuse a request that would sample this artifact off its schedule.

        ``sampling`` is not optional. An earlier version took only the pipeline's default
        shifts and compared them against the constants -- two constants, always equal, so
        the guard never fired. The values that matter arrive per request in
        ``sampling.extra_args`` and override those defaults downstream.
        """
        if sampling is None:
            raise VDNCheckpointError("VDN request validation needs the sampling parameters")

        if getattr(sampling, "lora_request", None) is not None:
            # The adapters are in the weights and lora_is_fused skips the dynamic LoRA
            # manager, so nothing would apply a per-request one. Serving anyway ignores it.
            raise OmniClientError(
                f"this server fused {self._source} into the checkpoint at startup, so "
                "per-request lora is unavailable; drop the lora field"
            )

        steps = self.turbo_num_steps
        requested_steps = int(getattr(sampling, "num_inference_steps", 0) or 0)
        if steps is not None and requested_steps != steps:
            raise OmniClientError(
                f"this VDN artifact carries a {steps}-step DMD adapter and requires "
                f"num_inference_steps={steps}, got {requested_steps}. Sampling it elsewhere on "
                "the schedule is what the adapter was distilled to invert."
            )

        # The per-modality shifts turn the schedule's positions into the noise levels the
        # adapter saw. A request that moves them samples where it was never trained.
        extra = getattr(sampling, "extra_args", None) or {}
        for key, expected in (("flow_shift", video_shift), ("audio_flow_shift", audio_shift)):
            try:
                requested = float(extra.get(key, expected))
            except (TypeError, ValueError) as exc:
                raise OmniClientError(f"VDN-H3 requires {key}={expected:g}") from exc
            if not math.isclose(requested, expected):
                raise OmniClientError(f"VDN-H3 requires {key}={expected:g}, got {requested:g}")


_BIAS_LAYOUT = "bias"
_PLAIN_SLOT = "plain"
_QKV_SLOTS = ("q", "k", "v")
_QKV_LAYOUT = "qkv"
_SWAP_HALVES_LAYOUT = "swap_halves"


def _read_branch(path: Path, *, num_blocks: int) -> dict[str, torch.Tensor]:
    """The branch/gate/projection tensors, under the names this model gives them.

    ``transformer_blocks.N.attn.<...>`` -> ``blocks.N.attn.vdn.<...>``. The extra
    ``vdn`` level is where ``enable_vdn_branch`` hangs the hybrid extras, which keeps a
    dense H3 free of parameters no checkpoint fills.
    """
    out: dict[str, torch.Tensor] = {}
    seen_blocks: set[int] = set()
    with safe_open(path, framework="pt", device="cpu") as branch:
        for key in branch.keys():
            match = re.match(r"^transformer_blocks\.(\d+)\.attn\.(.+)$", key)
            if match is None:
                raise VDNCheckpointError(f"unexpected tensor in {path}: {key}")
            index, suffix = int(match.group(1)), match.group(2)
            seen_blocks.add(index)
            out[f"blocks.{index}.attn.vdn.{suffix}"] = branch.get_tensor(key)
    if seen_blocks != set(range(num_blocks)):
        raise VDNCheckpointError(
            f"{path} covers blocks {sorted(seen_blocks)[:3]}..., but the model has {num_blocks}; "
            "the branch is per block and a gap would leave dense attention with no complement"
        )
    return out


def _read_adapters(root: Path, *, only: Sequence[str] | None = None) -> tuple[dict[str, _Patch], tuple[str, ...]]:
    """Every adapter under ``adapters/``, resolved onto native parameter names.

    All of them by default: a stage-dmd artifact carries ``default`` and ``turbo`` and
    the released inference path merges both, so applying one is a different model.
    ``only`` narrows that to a named subset for the bake tool -- see
    ``VDNCheckpoint.from_path`` for the one case that justifies it. A name that is not
    present raises rather than silently producing a smaller artifact than asked for.
    """
    from vllm_omni.diffusion.models.minimax_h3.fasth3 import (
        _PLAIN,
        _SWAP_HALVES,
        _resolve_native_target,
    )

    patches: dict[str, _Patch] = {}
    names: list[str] = []
    if not root.is_dir():
        return patches, ()

    directories = sorted(p for p in root.iterdir() if p.is_dir())
    if only is not None:
        wanted = tuple(only)
        present = {p.name for p in directories}
        unknown = [name for name in wanted if name not in present]
        if unknown:
            raise VDNCheckpointError(
                f"{root} has no adapter(s) {unknown}; it carries {sorted(present)}. Baking a "
                "subset that does not exist would produce an artifact silently missing an edit."
            )
        directories = [p for p in directories if p.name in set(wanted)]

    for directory in directories:
        tensor_file = directory / "adapter_model.safetensors"
        if not tensor_file.is_file():
            raise VDNCheckpointError(f"adapter {directory.name} has no adapter_model.safetensors")
        names.append(directory.name)
        with safe_open(tensor_file, framework="pt", device="cpu") as adapter:
            keys = list(adapter.keys())
            # An adapter that carries anything but low-rank pairs is editing the model
            # in a way this loader does not reproduce -- FastH3, for instance, ships
            # full-rank ``.diff`` tensors. Silently skipping them would serve a
            # partially-applied adapter.
            unconsumed = sorted(
                k for k in keys if not (_LORA_A.search(k) or _LORA_B.search(k) or k.endswith(_DIFF_B_SUFFIX))
            )
            if unconsumed:
                raise VDNCheckpointError(
                    f"{tensor_file} carries {len(unconsumed)} non-LoRA tensors this loader does "
                    f"not apply: {unconsumed[:5]}"
                )
            for key in keys:
                if not _LORA_A.search(key):
                    continue
                # The adapter-name infix is part of the key (``.lora_A.turbo.weight``),
                # so the partner is found by substitution rather than by a fixed suffix.
                b_key = re.sub(r"\.lora_A\.([^.]+)\.weight$", r".lora_B.\1.weight", key)
                if b_key not in keys:
                    raise VDNCheckpointError(f"{tensor_file}: {key} has no lora_B partner")
                module = _strip_hybrid_level(re.sub(r"\.lora_A\.[^.]+\.weight$", "", key))
                resolved = _resolve_native_target(module)
                if resolved is None:
                    raise VDNCheckpointError(f"{tensor_file}: adapter target {module!r} has no parameter in this model")
                native_suffix, layout, _ = resolved
                native = f"{native_suffix}.weight"
                # FastH3 spells the QKV slots as the slot letters and the fused MLP as
                # swap_halves; reuse that vocabulary rather than inventing a parallel
                # one for the same checkpoint layout.
                slot = layout if layout in _QKV_SLOTS else _PLAIN_SLOT
                param_layout = (
                    _QKV_LAYOUT if layout in _QKV_SLOTS else (_SWAP_HALVES_LAYOUT if layout == _SWAP_HALVES else _PLAIN)
                )
                patch = patches.setdefault(native, _Patch(layout=param_layout))
                if patch.layout != param_layout:
                    raise VDNCheckpointError(f"{native} is claimed with two layouts: {patch.layout} and {param_layout}")
                patch.pairs.setdefault(slot, []).append((adapter.get_tensor(key), adapter.get_tensor(b_key)))

            for key in keys:
                if not key.endswith(_DIFF_B_SUFFIX):
                    continue
                module = _strip_hybrid_level(key[: -len(_DIFF_B_SUFFIX)])
                resolved = _resolve_native_target(module)
                if resolved is None:
                    raise VDNCheckpointError(f"{tensor_file}: adapter target {module!r} has no parameter in this model")
                native_suffix, layout, _ = resolved
                if layout in _QKV_SLOTS or layout == _SWAP_HALVES:
                    raise VDNCheckpointError(
                        f"{tensor_file}: {key} is a bias residual on {module!r}, which this model packs "
                        f"as {layout!r}; only the curve-pruned AdaLN projections carry one"
                    )
                # The residual belongs to the pruned parameter's folded_bias, not to the
                # weight it was fitted alongside. A base without one is an UNPRUNED
                # checkpoint, and this adapter has already been narrowed to a rank-8
                # curve it cannot undo -- so that is refused at fuse time, by name.
                native = f"{native_suffix.rsplit('.linear', 1)[0]}.folded_bias"
                patch = patches.setdefault(native, _Patch(layout=_BIAS_LAYOUT))
                if patch.layout != _BIAS_LAYOUT:
                    raise VDNCheckpointError(f"{native} is claimed with two layouts: {patch.layout} and {_BIAS_LAYOUT}")
                patch.biases.append(adapter.get_tensor(key))
    return patches, tuple(names)


def baked_adapters_of(model_path: str | Path | None) -> tuple[str, ...] | None:
    """Adapters already baked into the base at ``model_path``, or None if it carries no stamp."""
    if not model_path:
        return None
    config = Path(model_path) / "transformer" / "config.json"
    if not config.is_file():
        return None
    try:
        stamp = json.loads(config.read_text()).get(VDN_BAKED_STAMP_KEY)
    except (OSError, ValueError):
        return None
    if not stamp:
        return None
    return tuple(stamp.get("adapters") or ())


def check_bake_agreement(checkpoint: VDNCheckpoint, model_path: str | Path | None) -> None:
    """The artifact and the base must agree on who carries the adapters.

    Both ways of disagreeing are silent. Serving a BAKED base with an artifact that still
    has ``adapters/`` applies every LoRA a second time on top of itself; serving a RAW
    base with a branch-only artifact runs the branch with no adapters at all. Neither
    raises on its own, both load, and both render video that is not the released model.
    """
    baked = baked_adapters_of(model_path)
    if baked and checkpoint.adapters:
        raise VDNCheckpointError(
            f"{model_path} already has {list(baked)} baked into its weights, but "
            f"{checkpoint.source} still carries adapters/ ({list(checkpoint.adapters)}). "
            "Serving both applies them twice. Point --lora-path at a copy with adapters/ "
            "removed (linear_branch/ and model_spec.json are all that is needed)."
        )
    if not baked and not checkpoint.adapters:
        raise VDNCheckpointError(
            f"{checkpoint.source} carries no adapters/ and {model_path} is not a baked "
            "partition, so the LoRAs the branch was trained with would not be applied at "
            "all. Use the full VDN checkpoint here, or point at a base produced by "
            "tools/minimax_h3/bake_vdn_adapters.py."
        )


def resolve_vdn_checkpoint(
    od_config: Any,
    transformer: MiniMaxH3DiTModel,
    *,
    model_path: str | Path | None = None,
) -> VDNCheckpoint | None:
    """Claim ``--lora-path`` when it points at a VDN-H3 checkpoint directory.

    ``model_path`` is the partition THIS transformer's weights come from, and it is what
    the bake stamp is read off. It defaults to the served model, which is right for the
    primary DiT and wrong for the second one: on a ``combined`` partition the ref2va DiT
    loads from ``<root>/Ref2VA`` while ``od_config.model`` is the root, and
    ``bake_vdn_adapters.py`` stamps one ``transformer/`` per run. Reading the root's
    stamp for the ref DiT lets exactly the two disagreements ``check_bake_agreement``
    exists to catch through -- a double fuse, or a branch with no adapters at all.
    """
    lora_path = getattr(od_config, "lora_path", None)
    if isinstance(lora_path, list | tuple):
        if len(lora_path) != 1:
            return None
        lora_path = lora_path[0]
    if not lora_path:
        return None
    arch = transformer.arch
    checkpoint = VDNCheckpoint.from_path(
        lora_path,
        head_dim=arch.attention_head_dim,
        num_blocks=arch.num_layers,
    )
    if checkpoint is not None:
        check_bake_agreement(checkpoint, model_path if model_path is not None else getattr(od_config, "model", None))
    return checkpoint


__all__ = [
    "VDN_ALLOW_UNTRAINED_TASKS_ENV",
    "VDN_BAKED_STAMP_KEY",
    "baked_adapters_of",
    "check_bake_agreement",
    "VDN_PARTITION_COSINE",
    "VDN_TRANSFERRED_TASKS",
    "VDN_AUDIO_SHIFT",
    "VDN_SUPPORTED_TASKS",
    "VDN_VIDEO_SHIFT",
    "VDNCheckpoint",
    "VDNCheckpointError",
    "VDNTransformSpec",
    "resolve_vdn_checkpoint",
]
