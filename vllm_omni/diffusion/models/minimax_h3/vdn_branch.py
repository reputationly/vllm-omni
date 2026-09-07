# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""VDN-H3's linear-attention branch: everything the softmax window cannot see.

The window in ``vdn_window.py`` restricts each video token to 15 latent frames. This
module carries the rest -- a bidirectional delta-rule recurrence over frames, read out
per token -- so the pair is a partition of the sequence rather than a truncation of it.
Ported from ``OpenVDN/vdn-minimax-h3`` (``src/models/linear_attention/``) against the
released ``stage-dmd-step-250`` weights.

Per DiT block, one frame at a time::

    features   q, k, v      raw (pre-QK-norm, pre-RoPE) projections, [short conv ->]
                            SiLU, L2-norm on q/k, no RoPE
    statistics A = k^T diag(beta) k,  B = v^T diag(beta) k      [H, d, d] per frame
    scan       S_t = (S_{t-1} diag(alpha_t) + B_t) (I + A_t)^-1, forwards and backwards
    gather     the state just outside the window on each side, decayed in to frame t
    readout    q_t S_t^T, RMS-normed and gated

Three properties are load-bearing and none of them are visible in a shape check:

* **A must be fp32.** ``A = sum_s beta_s k_s k_s^T`` is symmetric in exact arithmetic,
  but computed as ``(k*beta)^T @ k`` in bf16 the (i,j) and (j,i) entries round
  differently. On real activations -- where patches within a frame are strongly
  correlated, so A's off-diagonals are large -- that asymmetry pushes the smallest
  eigenvalue of ``I + A`` below 1, and the Cholesky reads the lower triangle only and
  factorises an indefinite matrix. Random inputs do not reproduce it.
* **alpha must be fp32.** The scan multiplies alpha across every frame, so per-element
  error compounds; in bf16 the worst channels' retention is off by tens of percent
  after ~100 frames.
* **The frame mean must be taken in fp32.** ``video_x`` is bf16, so a bf16 mean throws
  away what the fp32 island downstream cannot recover.

The branch adds no projections: it reads the attention's own raw q/k/v, so under a
merged LoRA it sees the adapted projections for free.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from vllm_omni.diffusion.distributed.parallel_state import model_parallel_is_initialized

# Only the released rule is implemented. The reference also carries ``sana_scaled`` and
# ``vdn_scaled``, which no published checkpoint uses; a rule that is never exercised is
# a silent way to render the wrong model, since re-pointing it changes nothing
# observable except the output.
VDN_DELTA_RULE = "vdn_solve"

# Each directional scan starts from half the prompt state: both directions carry the
# same prompt and the gather adds them, so a half each keeps the sum at roughly one
# copy. Baked into the trained weights -- changing it retrains, it does not reconfigure.
TEXT_STATE_SCALE = 0.5

SHORT_CONV_KERNEL = 5

# How each parameter divides across a tensor-parallel head shard, relative to the module
# that owns it. "head" slices dim 0 in units of heads, "channel" in units of
# head_dim-wide channels, and anything absent is replicated -- the two
# ``[hidden -> head_dim]`` bottlenecks and the per-channel RMSNorm weight.
#
# One table, read by the weight loaders that shard a released checkpoint AND by the test
# that asserts a shard equals a slice of the full run. Two spellings of this would be two
# chances to disagree, and disagreeing produces a model that loads and renders.
VDN_BRANCH_SHARD_AXES = {
    "alpha.A_log": "head",
    "alpha.dt_bias": "channel",
    "alpha.up.weight": "channel",
    "beta_proj.weight": "head",
    "output_gate.up.weight": "channel",
    "output_gate.up.bias": "channel",
    "short_conv.k_sp.weight": "channel",
    "short_conv.k_tm.weight": "channel",
    "short_conv.v_sp.weight": "channel",
    "short_conv.v_tm.weight": "channel",
}

# The softmax mass gate lives on the hybrid wrapper rather than the branch, but shards
# the same way: one value per head.
VDN_GATE_SHARD_AXES = {"softmax_gate.up.weight": "head", "softmax_gate.up.bias": "head"}


def _slice_loader(offset: int, length: int):
    """A weight loader that keeps this rank's contiguous slice of dim 0.

    vLLM's loader machinery calls ``param.weight_loader(param, loaded_weight)`` when the
    attribute exists, so a released full-width tensor is narrowed on its way in rather
    than after. Without it ``default_weight_loader`` would copy a 56-head tensor into a
    14-head parameter and raise -- which is the good case; the bad one is a parameter
    that happens to have a compatible shape.
    """

    def weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight[offset : offset + length])

    return weight_loader


def attach_head_shard_loaders(
    module: nn.Module, axes: dict[str, str], *, head_start: int, local_heads: int, head_dim: int
) -> None:
    """Give every sharded parameter of ``module`` its head-slice loader."""
    named = dict(module.named_parameters())
    for name, axis in axes.items():
        param = named.get(name)
        if param is None:
            continue  # e.g. short_conv on a checkpoint that trained without one
        unit = 1 if axis == "head" else head_dim
        param.weight_loader = _slice_loader(head_start * unit, local_heads * unit)


def _head_shard(total_heads: int) -> tuple[int, int]:
    """This rank's contiguous head range, matching how QKVParallelLinear shards.

    The branch is per-head everywhere except two [hidden -> 128] bottlenecks, so a head
    shard is the whole story. It has to be the SAME shard the attention uses, because
    the branch consumes that attention's raw q/k/v.

    This is the FALLBACK, for a standalone build (parity tests, CPU-side tools). Inside
    a model it is overridden: ``VDNHybridAttention`` passes the head count straight off
    the attention's own QKV projection, because deriving it separately means two views
    of the shard that can disagree -- ``model_parallel_is_initialized`` here needs the
    diffusion groups, which can be absent while vLLM's TP group is live, and then this
    returns "unsharded" while the projection really is sharded.

    Outside an initialised group this is a single-process build, i.e. TP of degree one.
    Asking the TP group there would assert rather than answer.
    """
    if not model_parallel_is_initialized():
        return 0, total_heads
    tp_size = get_tensor_model_parallel_world_size()
    if total_heads % tp_size:
        raise ValueError(
            f"VDN branch needs the {total_heads} attention heads to divide across "
            f"tensor_parallel_size={tp_size}; it shards on the head axis to stay aligned "
            "with the QKV projection it reads."
        )
    local = total_heads // tp_size
    return get_tensor_model_parallel_rank() * local, local


class VDNFrameAlpha(nn.Module):
    """Per-frame forget gate ``alpha = exp(-exp(A_log) * softplus(delta + dt_bias))``.

    KDA's double-exponential gate in fla's layout: ``A_log`` is per head, the
    per-channel freedom is ``dt_bias``, and the down/up path is rank ``head_dim``. The
    whole thing runs in fp32 with autocast disabled -- see the module docstring.
    """

    def __init__(self, hidden_size: int, local_heads: int, head_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.local_heads, self.head_dim = local_heads, head_dim
        # `down` reads the full hidden state and is replicated; `up` is per (head,
        # channel) and carries the shard.
        self.down = nn.Linear(hidden_size, head_dim, bias=False, dtype=dtype)
        self.up = nn.Linear(head_dim, local_heads * head_dim, bias=False, dtype=dtype)
        self.A_log = nn.Parameter(torch.empty(local_heads, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.empty(local_heads * head_dim, dtype=dtype))

    def forward(self, frame_mean: torch.Tensor) -> torch.Tensor:
        """frame_mean: [F, hidden] fp32 -> alpha [F, H_local, head_dim] fp32."""
        # autocast off, and the WEIGHTS promoted too: casting only the input would be
        # an fp32 activation against an 8-mantissa-bit weight, which is not fp32 math.
        with torch.autocast(device_type=frame_mean.device.type, enabled=False):
            delta = F.linear(frame_mean.float(), self.down.weight.float())
            delta = F.linear(delta, self.up.weight.float())
            delta = delta + self.dt_bias.float()
            scale = torch.exp(self.A_log.float())[:, None]  # [H, 1], broadcast over d_k
            delta = delta.view(-1, self.local_heads, self.head_dim)
            return torch.exp(-scale * F.softplus(delta))


class VDNOutputGate(nn.Module):
    """The sigmoid gate both branches put on their output, at two granularities.

    ``head_dim=None`` is the softmax branch's gate: one value per (token, head). The
    windowed softmax renormalises to 1 no matter how little mass it saw, so this scales
    that branch back toward the share it actually captured -- a property of a
    distribution, hence per head. ``head_dim`` set is the linear branch's: one value per
    (token, head, channel), a routing decision on a new pathway, low rank.
    """

    def __init__(
        self,
        hidden_size: int,
        local_heads: int,
        head_dim: int | None = None,
        bottleneck: int | None = None,
        bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.local_heads, self.head_dim = local_heads, head_dim
        out_features = local_heads * (head_dim or 1)
        self.down = None if bottleneck is None else nn.Linear(hidden_size, bottleneck, bias=False, dtype=dtype)
        self.up = nn.Linear(bottleneck or hidden_size, out_features, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [tokens, hidden] -> [tokens, H_local, head_dim or 1] in (0, 1)."""
        gate = torch.sigmoid(self.up(x if self.down is None else self.down(x)))
        return gate.view(-1, self.local_heads, self.head_dim or 1)


class VDNSeparableShortConv(nn.Module):
    """Depthwise 5x5 spatial then depthwise 5-tap temporal, on the named projections.

    The released checkpoints convolve K and V and leave Q as the raw NoPE features. The
    effective 3-D kernel is the rank-1 outer product of the two halves: 30 parameters
    per channel instead of 125, and both halves ride tuned kernels where a dense 5^3
    depthwise Conv3d has no fast path anywhere.

    The layout costs nothing: tokens read as ``[T, H, W, C]`` ARE the channels_last form
    of ``[T, C, H, W]``, so the spatial conv takes a view and its output permutes back
    for free. Temporal padding is zero and symmetric -- non-causal, and it deliberately
    crosses VAE chunk boundaries.
    """

    def __init__(
        self,
        local_channels: int,
        projections: tuple[str, ...] = ("k", "v"),
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.projections = tuple(projections)
        for name in self.projections:
            setattr(
                self,
                f"{name}_sp",
                nn.Conv2d(
                    local_channels,
                    local_channels,
                    SHORT_CONV_KERNEL,
                    padding=SHORT_CONV_KERNEL // 2,
                    groups=local_channels,
                    bias=False,
                    dtype=dtype,
                ),
            )
            setattr(
                self,
                f"{name}_tm",
                nn.Conv1d(
                    local_channels,
                    local_channels,
                    SHORT_CONV_KERNEL,
                    padding=SHORT_CONV_KERNEL // 2,
                    groups=local_channels,
                    bias=False,
                    dtype=dtype,
                ),
            )

    def forward(
        self,
        projection: str,
        tokens: torch.Tensor,
        num_frames: int,
        frame_size: tuple[int, int],
    ) -> torch.Tensor:
        """tokens: [F*S, H, d] -> same shape, convolved. Unlisted projections pass through."""
        if projection not in self.projections:
            return tokens
        heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        grid_h, grid_w = frame_size
        channels = heads * head_dim
        volume = tokens.reshape(num_frames, grid_h, grid_w, channels).permute(0, 3, 1, 2)
        volume = F.conv2d(
            volume,
            getattr(self, f"{projection}_sp").weight,
            padding=SHORT_CONV_KERNEL // 2,
            groups=channels,
        )
        rows = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
        # The temporal half as shift-multiply-add rather than conv1d: a depthwise
        # temporal conv runs well below bandwidth here, and this spelling fuses with the
        # SiLU/L2-norm tail. The fp32 weight is cast explicitly -- elementwise multiply
        # is not an autocast op, so fp32 x bf16 would promote the whole pass.
        weight = getattr(self, f"{projection}_tm").weight.squeeze(1).to(rows.dtype)
        padded = F.pad(rows, (0, 0, 0, 0, SHORT_CONV_KERNEL // 2, SHORT_CONV_KERNEL // 2))
        out = padded[0:num_frames] * weight[:, 0].view(1, 1, -1)
        for tap in range(1, SHORT_CONV_KERNEL):
            out = out + padded[tap : tap + num_frames] * weight[:, tap].view(1, 1, -1)
        return out.reshape(-1, heads, head_dim)


class VDNBranchRMSNorm(nn.Module):
    """RMSNorm over head_dim, second moment accumulated through ``vector_norm``.

    Parameterisation matches ``nn.RMSNorm(dim, eps)``, but the spelling is not
    interchangeable with it at this shape ([F*S, H, d], F*S ~ 100k, bf16):

        x.pow(2).mean(dtype=fp32)   ~1e-3 relative error   (squares round to bf16 FIRST)
        x.float().pow(2).mean()     exact                  (a full fp32 copy of x, ~2.8 GiB)
        vector_norm(dtype=fp32)^2   ~1e-7 relative error   (no extra materialisation)

    The weight is cast DOWN to the activation dtype rather than the input being cast
    up: a bf16 x fp32 elementwise multiply would silently promote the whole tensor.
    """

    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / x.shape[-1]
        return x * torch.rsqrt(mean_square + self.eps).to(x.dtype) * self.weight.to(x.dtype)


def _activate(tokens: torch.Tensor, l2norm: bool) -> torch.Tensor:
    """SiLU, then L2-norm for q/k, preserving the input dtype (fla's L2Norm semantics)."""
    activated = F.silu(tokens)
    if not l2norm:
        return activated
    return F.normalize(activated, dim=-1, eps=1e-6).to(activated.dtype)


def frame_statistics(
    key_f: torch.Tensor, value_f: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame delta-rule statistics. key_f/value_f: [F, H, S, d]; beta: [F, H, S].

    A is accumulated in fp32 and explicitly symmetrised (see the module docstring); B is
    a plain readout that is never inverted and its error enters the state linearly, so
    it stays on the tensor cores in bf16 and is widened afterwards.

    The ``.contiguous()`` calls are load-bearing rather than defensive: the callers hand
    in ``.permute(0, 2, 1, 3)`` views, so the contraction axis S carries stride H*d and
    a batched cuBLAS GEMM given those strides runs at a fraction of its throughput.
    """
    with torch.autocast(device_type=key_f.device.type, enabled=False):
        key16 = key_f.contiguous()
        key32 = key16.float()
        scaled32 = (key32 * beta.unsqueeze(-1).float()).contiguous()
        value_beta = (value_f * beta.unsqueeze(-1).to(value_f.dtype)).contiguous()

        a_stat = torch.matmul(scaled32.transpose(-1, -2), key32)
        a_stat = 0.5 * (a_stat + a_stat.transpose(-1, -2))
        b_stat = torch.matmul(value_beta.transpose(-1, -2), key16).float()
        return a_stat, b_stat


def delta_rule_factors(
    alpha: torch.Tensor, a_stat: torch.Tensor, b_stat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``vdn_solve``: transition = diag(alpha) (I+A)^-1, injection = B (I+A)^-1.

    ``I + A`` is SPD, so this is a Cholesky. The inverse is formed once for all frames
    because the serial scan applies it repeatedly anyway, and it is built as one
    triangular solve plus a product rather than ``cholesky_solve`` (two solves): a
    batched triangular solve at 128x128 runs an order of magnitude below a batched GEMM
    at the same shape.
    """
    a32 = a_stat.float()
    eye = torch.eye(a32.shape[-1], device=a32.device, dtype=torch.float32).expand_as(a32)
    chol = torch.linalg.cholesky(a32 + eye)
    linv = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
    inverse = linv.transpose(-1, -2) @ linv  # symmetric by construction
    return alpha.unsqueeze(-1) * inverse, b_stat.float() @ inverse


def run_scans(
    transitions: torch.Tensor, injections: torch.Tensor, text_state: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward and reverse state banks, written into preallocated tensors.

    ``prefix[t]`` holds frames 0..t and ``suffix[t]`` holds frames t..F-1. Both scans
    start from the same seed, so every frame's two directional states carry the prompt
    while each frame is still injected exactly once per direction.

    ``baddbmm(inj, state, trans, out=bank[t])`` is the same three operands as
    ``state @ transitions[t] + injections[t]`` followed by a store, with cuBLAS doing
    the add in its fp32 epilogue: three launches per frame become one, on a recurrence
    that is launch-bound rather than compute-bound.
    """
    with torch.autocast(device_type=transitions.device.type, enabled=False):
        num_frames = transitions.shape[0]
        start = torch.zeros_like(injections[0]) if text_state is None else text_state.to(injections.dtype)
        prefix = torch.empty((num_frames, *start.shape), dtype=injections.dtype, device=start.device)
        suffix = torch.empty_like(prefix)

        state = start
        for frame in range(num_frames):
            torch.baddbmm(injections[frame], state, transitions[frame], out=prefix[frame])
            state = prefix[frame]

        state = start
        for frame in range(num_frames - 1, -1, -1):
            torch.baddbmm(injections[frame], state, transitions[frame], out=suffix[frame])
            state = suffix[frame]
        return prefix, suffix


def gather_outside_window(
    prefix_states: torch.Tensor,
    suffix_states: torch.Tensor,
    alpha: torch.Tensor,
    bounds: list[tuple[int, int]],
    text_state: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Everything OUTSIDE each frame's window, in that frame's frame of reference.

    For query frame t with window [lo, hi] the complement is exactly
    ``prefix_states[lo-1] + suffix_states[hi+1]`` -- one frame outside on each side,
    nothing counted twice -- each decayed in to t by the product of alpha over the
    frames in between. Decaying rather than applying the full transition is the point:
    the window's own frames are covered by the softmax side, so writing them again here
    would double count them.

    A frame whose window already touches a clip end has no neighbour on that side. The
    index is clamped to stay in range and the contribution then replaced by the state
    the scans STARTED from, so the boundary rows read the prompt instead of reading
    nothing -- the same arithmetic the interior rows get.
    """
    device = prefix_states.device
    num_frames = prefix_states.shape[0]
    last_before = torch.tensor([lo for lo, _ in bounds], device=device) - 1
    first_after = torch.tensor([hi for _, hi in bounds], device=device) + 1
    has_before = (last_before >= 0).view(-1, 1, 1, 1)
    has_after = (first_after < num_frames).view(-1, 1, 1, 1)

    state_before = prefix_states[last_before.clamp(min=0)]
    state_after = suffix_states[first_after.clamp(max=num_frames - 1)]
    if text_state is not None:
        seed = text_state.to(state_before.dtype)
        state_before = torch.where(has_before, state_before, seed)
        state_after = torch.where(has_after, state_after, seed)

    # prod alpha over a span as a difference of log-prefix sums: any (a, b) pair is one
    # subtraction rather than a product loop. The leading zero row makes the prefix
    # exclusive, so the empty product is 1.
    log_alpha = torch.log(alpha.clamp_min(1e-12))
    log_prefix = torch.cat([torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)])
    frames = torch.arange(num_frames, device=device)
    # The bridge index is NOT the gather index at the ends: a boundary row gathers a
    # clamped state it then discards, but must decay the seed over the frames it really
    # skipped -- from virtual -1 that is [0..t], from virtual F it is [t..F-1]. Clamping
    # both the same way decays the seed over one frame too few.
    from_before = torch.exp(log_prefix[frames + 1] - log_prefix[(last_before + 1).clamp(min=0)])
    from_after = torch.exp(log_prefix[first_after.clamp(max=num_frames)] - log_prefix[frames])
    # alpha is per KEY channel, so it broadcasts over d_v, not d_k.
    state_before = state_before * from_before.unsqueeze(2)
    state_after = state_after * from_after.unsqueeze(2)

    if text_state is not None:
        out = state_before + state_after  # both sides always contribute
    else:
        out = state_before * has_before + state_after * has_after
    return out.to(out_dtype)


class VDNLinearBranch(nn.Module):
    """The linear branch of one DiT block, sharded on the attention's head axis.

    The reference shards this for Ulysses, where a rank owns a head range and is handed
    the beta/gate/frame-mean its sequence owner computed. Our production shard is tensor
    parallel instead, which is strictly simpler: every rank holds the whole sequence, so
    the frame scan is local and only the parameters divide.
    """

    def __init__(
        self,
        hidden_size: int,
        total_heads: int,
        head_dim: int,
        *,
        short_conv: tuple[str, ...] = ("k", "v"),
        enable_text_state: bool = True,
        delta_rule: str = VDN_DELTA_RULE,
        # Every parameter in the block around this one is built at an explicit dtype;
        # torch's fp32 default would meet bf16 activations at the first matmul. The
        # released branch tensors are bf16, and the reference likewise casts the whole
        # branch down before inference -- the fp32 islands below promote what needs it.
        params_dtype: torch.dtype = torch.bfloat16,
        # The authoritative shard, when the caller has one. A model passes the head
        # count off the very QKV projection this branch reads, so the two cannot
        # disagree; only standalone builds fall back to deriving it.
        head_shard: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if delta_rule != VDN_DELTA_RULE:
            raise ValueError(
                f"VDN branch implements {VDN_DELTA_RULE!r} only, got {delta_rule!r}. The "
                "reference's other rules have no released weights, and swapping the rule "
                "changes nothing observable except the render."
            )
        self.head_start, self.local_heads = head_shard or _head_shard(total_heads)
        self.head_dim = head_dim
        self.enable_text_state = enable_text_state
        local_channels = self.local_heads * head_dim

        self.short_conv = VDNSeparableShortConv(local_channels, short_conv, dtype=params_dtype) if short_conv else None
        self.alpha = VDNFrameAlpha(hidden_size, self.local_heads, head_dim, params_dtype)
        self.beta_proj = nn.Linear(hidden_size, self.local_heads, bias=False, dtype=params_dtype)
        self.output_gate = VDNOutputGate(
            hidden_size, self.local_heads, head_dim, bottleneck=head_dim, dtype=params_dtype
        )
        # RMSNorm over head_dim: one weight vector shared by every head, so it is
        # replicated rather than sharded.
        self.norm = VDNBranchRMSNorm(head_dim, dtype=params_dtype)
        attach_head_shard_loaders(
            self,
            VDN_BRANCH_SHARD_AXES,
            head_start=self.head_start,
            local_heads=self.local_heads,
            head_dim=head_dim,
        )

    def _features(
        self,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        num_frames: int | None,
        frame_size: tuple[int, int] | None,
        *,
        use_conv: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[short conv ->] SiLU [-> L2-norm on q/k], no RoPE.

        ``use_conv=False`` is the text chunk: the conv is a (t, h, w) stencil over the
        video volume and the prompt has no such grid. Text keeps the rest of the chain
        so the delta rule it feeds is the one the video frames feed.
        """
        conv = self.short_conv if use_conv else None
        if conv is not None and frame_size is None:
            raise ValueError(
                "the VDN short conv needs the patched spatial grid; pass frame_size=(H, W). "
                "The token count alone cannot be factored back into one (1008 is 24x42, "
                "but also 16x63)."
            )
        out = []
        for projection, tokens in zip(("q", "k", "v"), qkv_raw, strict=True):
            if conv is not None:
                tokens = conv(projection, tokens, num_frames, frame_size)
            out.append(_activate(tokens, l2norm=projection != "v"))
        return tuple(out)

    def _text_state(
        self,
        text_x: torch.Tensor,
        text_qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """The whole prompt written into a zero state as ONE delta-rule chunk, halved.

        No causal scan inside the text: a chunk update over all L rows at once is what
        the video path does per frame, and the text encoder and token refiner have
        already written word order into every text hidden state. alpha plays no part --
        the old state is zero, so the transition multiplies nothing.
        """
        length = text_qkv_raw[1].shape[0]
        _, key, value = self._features(text_qkv_raw, None, None, use_conv=False)
        key = key.view(1, length, self.local_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(1, length, self.local_heads, self.head_dim).permute(0, 2, 1, 3)
        beta = torch.sigmoid(self.beta_proj(text_x))
        beta = beta.view(1, length, self.local_heads).permute(0, 2, 1)

        a_stat, b_stat = frame_statistics(key, value, beta)
        with torch.autocast(device_type=a_stat.device.type, enabled=False):
            ones = torch.ones(1, self.local_heads, self.head_dim, device=a_stat.device, dtype=a_stat.dtype)
            _, injection = delta_rule_factors(ones, a_stat, b_stat)
        return TEXT_STATE_SCALE * injection[0]

    def forward(
        self,
        video_x: torch.Tensor,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        num_frames: int,
        tokens_per_frame: int,
        frame_size: tuple[int, int],
        bounds: list[tuple[int, int]],
        text_x: torch.Tensor | None = None,
        text_qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        skip_ends: bool = True,
    ) -> torch.Tensor:
        """video_x: [F*S, hidden] -> linear readout [F*S, H_local*d], pre-``to_out_linear``.

        ``skip_ends`` is the partner of ``anchor_frames="both"``: the softmax side makes
        frames 0 and F-1 exact in both directions, so this branch drops them from its
        INPUT entirely -- not "scanned then ignored" -- and their readout rows are
        exactly zero, which is what keeps the two branches an exact partition.
        """
        if not skip_ends:
            return self._readout(
                video_x,
                qkv_raw,
                num_frames,
                tokens_per_frame,
                frame_size,
                bounds,
                text_x,
                text_qkv_raw,
            )
        if num_frames <= 2:  # the anchors ARE the clip
            return video_x.new_zeros(num_frames * tokens_per_frame, self.local_heads * self.head_dim)

        inner = slice(tokens_per_frame, (num_frames - 1) * tokens_per_frame)
        readout = self._readout(
            video_x[inner],
            tuple(tensor[inner] for tensor in qkv_raw),
            num_frames - 2,
            tokens_per_frame,
            frame_size,
            # With the two frames gone the bounds rebase by one: the complement of
            # [lo, hi] inside frames 1..F-2 is the same arithmetic on (lo-1, hi-1).
            [(lo - 1, hi - 1) for lo, hi in bounds[1 : num_frames - 1]],
            text_x,
            text_qkv_raw,
        )
        # Zero only the two anchor frames rather than the whole readout: at the 15 s
        # shape this tensor is ~1.4 GiB.
        out = readout.new_empty(num_frames * tokens_per_frame, readout.shape[-1])
        out[:tokens_per_frame].zero_()
        out[(num_frames - 1) * tokens_per_frame :].zero_()
        out[inner] = readout
        return out

    def _readout(
        self,
        video_x,
        qkv_raw,
        num_frames,
        tokens_per_frame,
        frame_size,
        bounds,
        text_x,
        text_qkv_raw,
    ) -> torch.Tensor:
        """The algorithm over exactly the frames it owns; no notion of anchors."""
        heads, head_dim = self.local_heads, self.head_dim
        shape_per_frame = (num_frames, tokens_per_frame, heads, head_dim)

        query, key, value = self._features(qkv_raw, num_frames, frame_size)
        query_by_frame = query.view(shape_per_frame).permute(0, 2, 1, 3)  # [F, H, S, d]
        key_by_frame = key.view(shape_per_frame).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape_per_frame).permute(0, 2, 1, 3)

        beta = torch.sigmoid(self.beta_proj(video_x))
        beta = beta.view(num_frames, tokens_per_frame, heads).permute(0, 2, 1)  # [F, H, S]
        a_stat, b_stat = frame_statistics(key_by_frame, value_by_frame, beta)

        # fp32 on the mean itself, not just inside alpha: video_x is bf16, and a bf16
        # mean has already thrown away what the fp32 island cannot recover.
        alpha = self.alpha(video_x.view(num_frames, tokens_per_frame, -1).mean(dim=1, dtype=torch.float32))

        text_state = None
        if self.enable_text_state and text_x is not None:
            text_state = self._text_state(text_x, text_qkv_raw)

        with torch.autocast(device_type=video_x.device.type, enabled=False):
            transitions, injections = delta_rule_factors(alpha, a_stat, b_stat)
        prefix_states, suffix_states = run_scans(transitions, injections, text_state)
        del transitions, injections

        gate = self.output_gate(video_x)
        linear_state = gather_outside_window(prefix_states, suffix_states, alpha, bounds, text_state, gate.dtype)
        del prefix_states, suffix_states  # ~0.7 GiB, dead before the readout runs

        readout = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))  # [F, H, S, d_v]
        readout = readout.permute(0, 2, 1, 3).reshape(num_frames * tokens_per_frame, heads, head_dim)
        return (self.norm(readout) * gate).reshape(-1, heads * head_dim)


__all__ = [
    "TEXT_STATE_SCALE",
    "VDN_DELTA_RULE",
    "VDNBranchRMSNorm",
    "VDNFrameAlpha",
    "VDNLinearBranch",
    "VDNOutputGate",
    "VDNSeparableShortConv",
    "delta_rule_factors",
    "frame_statistics",
    "gather_outside_window",
    "run_scans",
]
