# SageAttention

SageAttention backends provide lossy low-precision attention for diffusion
models. Validate output quality against `TORCH_SDPA` at the same seed before
using either backend in production.

## `SAGE_ATTN`

`SAGE_ATTN` uses SageAttention 2.2 with INT8-quantized attention and FP16
accumulation.

### Installation

Install SageAttention into the same environment as vLLM-Omni:

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
pip install . --no-build-isolation
```

Verify the installation:

```bash
python -c "import sageattention; print(sageattention.__file__)"
```

Select it globally:

```bash
vllm-omni serve <model> --diffusion-attention-backend SAGE_ATTN
```

## `SAGE_ATTN_3`

`SAGE_ATTN_3` uses the SageAttention3 Blackwell implementation.

### SageAttention3 installation

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention/sageattention3_blackwell
python setup.py install
```

Verify the installation:

```bash
python -c "import sageattn3; print(sageattn3.__file__)"
```

```bash
vllm-omni serve <model> --diffusion-attention-backend SAGE_ATTN_3
```

`SAGE_ATTN_3` requires CUDA, an importable `sageattn3`, and a Blackwell-class
GPU. Its kernel assumes the query-head count equals the key/value-head count.
GQA and MQA diffusion calls therefore fall back to PyTorch SDPA for
correctness.

## `SLA_SAGE2_ATTN`

`SLA_SAGE2_ATTN` is a block-sparse backend, not a dense one — it belongs in the
same family as [`SLA_ATTN`](../attention_backends.md#sla_attn-backend-and-sparsity-distilled-checkpoints)
and shares its `block_sparse` configuration keys (`sparsity`, `start_step`,
`skip_layers`) and its no-fallback startup contract. The difference is only in
how the selected key blocks are computed: `SLA_ATTN` runs a dense-float Triton
kernel over them; `SLA_SAGE2_ATTN` runs
[SpargeAttn](https://github.com/thu-ml/SpargeAttn)'s INT8-quantized
SageAttention2 block-sparse kernel — the `operator: "sage2"` path LightX2V's
own published reference configuration for `lightx2v/Minimax-h3-Turbo-SLA` uses.
Block selection itself still comes from `sparse_linear_attention.utils.get_block_map`
(the same package `SLA_ATTN` uses), so the two backends select identical key
blocks and only the compute kernel differs.

```bash
vllm-omni serve /path/to/MiniMax-H3-FL2VA-Turbo4-768p-SLA-BF16 \
  --diffusion-attention-config '{"default": {"backend": "SLA_SAGE2_ATTN",
      "block_sparse": {"sparsity": 0.85, "start_step": 0}}}'
```

### Installation

```bash
git clone https://github.com/thu-ml/SpargeAttn.git
cd SpargeAttn
TORCH_CUDA_ARCH_LIST=8.0 pip install . --no-build-isolation
```

Set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability (`8.0` for A100,
`9.0` for Hopper, `10.0` for Blackwell). `--no-build-isolation` is required
because the package's `setup.py` imports `torch` at module level, which breaks
pip's default isolated build environment. This also needs
`sparse_linear_attention` installed (see
[`SLA_ATTN`'s installation section](../attention_backends.md#sla_attn-backend-and-sparsity-distilled-checkpoints))
for block selection.

### What's validated and what isn't

Only the sm80/86/87 (Ampere) kernel branch has been run on real hardware (an
A100); the sm90+ branch is carried over from SpargeAttn's own architecture
dispatch unverified.

**On MiniMax-H3, this backend corrupted audio — confirmed on the real kernel,
not extrapolated, then fixed by exempting the packed sequence's prefix from
block selection.** On non-sm90 GPUs the kernel requires `BLKQ=128, BLKK=64`,
the same block size a prior `SLA_ATTN` experiment found corrupts audio when
forced onto the plain Triton kernel (a 128-row query block can straddle the
packed sequence's text/keyframe/audio/video boundary; see
`docs/实验报告/MiniMax-H3-SLA稀疏注意力-接入实测与暂缓结论-2026-08-22.md` §5).
Running `SLA_SAGE2_ATTN` for real on MiniMax-H3 (`boars10s`, sparsity=0.85,
2026-08-26) reproduced the same failure on the real SpargeAttn kernel: LUFS
-41.3 → -7.4, true peak -28.1 → +3.3 dBFS (clipping), LRA 2.7 → 26.3, against
the same-content `SLA_ATTN` baseline.

**Fix**: when the model publishes `AttentionMetadata.video_layout`, block
selection now only runs over the pure video segment
(`video_layout.prefix_len:used_len`); the prefix (text, visual conditions,
audio) always runs dense, so no query block can straddle the boundary. This
mirrors `RAINFUSION_ATTN`'s existing prefix-dense design, not something new.
It is a deliberate departure from `SLA_ATTN`'s whole-sequence selection to
match how the SLA adapter was distilled (see that backend's own docstring) —
worth doing here because the alternative was a confirmed, 100%-reproducible
correctness bug, not a hypothetical one. **This trades a theoretical
off-distribution risk on the prefix for fixing the audio corruption; validate
video quality, not just audio, before relying on this for MiniMax-H3 in
production** — the fix has been code-reviewed and unit-tested, but the
audio-clean / video-quality-unaffected claim above still needs a fresh
real-hardware generation to confirm now that the fix is in place. When no
`video_layout` is published, this backend still selects over the whole
sequence (unverified quality on such roles, same as before the fix).

Timing and memory before the fix: 155.1s wall-clock and 34.0 GiB peak GPU
memory on the same case, matching `SLA_ATTN`'s Triton path closely
(186.6s/34.0 GiB) — the corruption was a correctness bug in this specific
block size on packed multimodal sequences, not a performance or stability
problem, and the prefix-dense fix should only add the small cost of one dense
attention call over the (typically small) prefix.

For common configuration and platform routing, see the
[attention backend overview](../attention_backends.md).
