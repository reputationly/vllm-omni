# VDN-MiniMax-H3

> Hybrid window-softmax / linear attention over MiniMax H3, for long clips

## Summary

- Vendor: OpenVDN (derivative of MiniMaxAI/MiniMax-H3)
- Model: [`OpenVDN/vdn-minimax-h3`](https://huggingface.co/OpenVDN/vdn-minimax-h3)
- Task: T2VA only
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Hardware validated: 4x A100-PCIE-40G (TP4)
- Maintainer: Community

VDN-H3 replaces H3's dense video attention with two branches that partition the
sequence. A chunk-aligned window keeps each latent frame's attention exact over 15
frames — three whole VAE chunks, since H3's video VAE codes `17n + 5` pixel frames as
`5n + 2` latent frames — and a bidirectional linear-attention branch summarises
everything the window drops. Text, audio and the two anchor frames stay dense in both
directions.

It ships as a ~5 GB increment over the 72 GB base H3 transformer, not as a release.

## What it is and is not for

**The win scales with clip length, and only with clip length.** The window is a fixed 15
latent frames while dense attention grows with F². Per DiT block on one A100-PCIE-40G:

| Clip | Latent frames | Window covers | Dense attention | Windowed |
| --- | --- | --- | --- | --- |
| 5.2 s | 37 | 40.5% | 220 ms | 115 ms (1.91x) |
| 6.6 s | 47 | 31.9% | 355 ms | 153 ms (2.33x) |
| 15.1 s | 107 | 14.0% | 1825 ms | 393 ms (**4.65x**) |

Reproduce with `python benchmarks/diffusion/vdn_h3/bench_window_backend.py --frames 37,47,107`.

At 5-6 s the linear branch costs more than the window saves. Use the dense profiles
there. This recipe is for 15 s.

**It is orthogonal to few-step distillation.** VDN reduces the cost of each step;
LightX2V's Turbo LoRA reduces the number of steps. The released `stage-dmd-step-250`
already carries its own 8-step DMD adapter, so do **not** stack the Turbo LoRA on top —
that is a different distillation line with a different rank, alpha and target set.

## Measured end to end

15 s / 768p / t2va, VDN at 8 steps against the production quality bar (dense at 20
steps), on 4x A100-PCIE-40G with TP4. Same prompt, seed, shape and box per row; the
three `ex*` rows are OpenVDN's own released example prompts (4, 7 and 8 shots, heavy
camera and subject motion), the `lighthouse` row is a deliberately static scene.

| prompt | VDN 8-step | dense 20-step | speedup | sharpness VDN/dense | frame-to-frame VDN/dense |
| --- | ---: | ---: | ---: | --- | --- |
| ex0 ronin (4 shots) | 349 s | 895 s | **2.56x** | 102.3 / 51.2 | 28.84 / 23.86 |
| ex1 elf (7 shots) | 344 s | 929 s | **2.70x** | 148.7 / 76.2 | 29.90 / 26.18 |
| ex2 Rhine square (8 shots) | 347 s | 910 s | **2.62x** | 468.2 / 224.0 | 20.58 / 16.29 |
| lighthouse (static) | 357 s | 885 s | 2.48x | 72.6 / 77.5 | 2.24 / 3.55 |

Sharpness is Laplacian variance, frame-to-frame is mean absolute inter-frame difference;
both from `benchmarks/diffusion/vdn_h3` metrics over every 6th frame.

Two things to read off this rather than from a single sample:

- On content that actually moves, VDN is **~2x sharper** and carries **14-26% MORE**
  inter-frame motion than the 20-step dense bar.
- The static `lighthouse` row disagrees on both counts, and it is the outlier: its
  inter-frame deltas are 2-3.5 against 20-30 for the released prompts, an order of
  magnitude less motion. An early version of this recipe generalised from that one
  prompt and concluded VDN damped motion. It does not; the prompt did.

Per-block under TP4 at the 15 s shape, for where the time goes
(`torchrun --nproc_per_node=4 benchmarks/diffusion/vdn_h3/bench_tp4_block.py --frames 107`):
attention 463 -> 100 ms, GEMMs 100 ms unchanged, two all-reduces 267 ms unchanged, branch
+14 ms. The single-card 4.65x lands at roughly 1.7x per block because the collectives do
not shrink; after VDN, communication is 55% of a block and is where the next optimisation
belongs. The end-to-end 2.6x above is against 20 steps, so it also carries the 8-step
adapter -- the two effects are not separated by these numbers.

## fl2va, and the INT8 route

fl2va is served from the same DiT partition VDN adapts, so it runs -- but at 15 s / 768p
BF16 it OOMs by ~1.5 GiB (the shape already peaks at 34-36 GiB before VDN adds anything).
Baking the adapters into the partition and quantizing to INT8 W8A16 frees the headroom:

```bash
python tools/minimax_h3/bake_vdn_adapters.py \
  --src /nfs-data/models/MiniMax-H3/FL2VA \
  --vdn /nfs-data/models/VDN-MiniMax-H3/stage-dmd-step-250 \
  --dst /nfs-data/models/MiniMax-H3-FL2VA-VDN8-BF16 --link-siblings
python vllm_omni/quantization/tools/quantize_minimax_h3_int8.py \
  --src /nfs-data/models/MiniMax-H3-FL2VA-VDN8-BF16 \
  --dst /nfs-data/models/MiniMax-H3-FL2VA-VDN8-INT8-W8A16 \
  --activation-scheme weight_only
```

62 GB -> 43.8 GB, and `--lora-path` then has to point at a VDN directory with
`adapters/` REMOVED, or they are applied a second time on top of the baked ones.

Measured at 15 s fl2va against the production Turbo8 BF16 tier: 375 s vs 399 s, peak
39.1 vs 39.2 GiB, sharpness **32.8 vs 23.8 (+38%)**.

**Switch for the picture, not the clock.** 1.06x and equal memory is not a reason to
move; the sharpness is, and it was confirmed by eye against the production tier before
this recipe recommended it. The speed advantage is small here because that tier is
already an 8-step distill and W8A16 runs a Triton GEMM rather than BF16 cuBLAS -- the
window's saving is real but it is spent, not banked.

**Upstream has since added keyframe conditioning.** As of `4250fbc` (2026-09-09) the
released harness renders i2va, l2va and fl2va, and `encode_keyframes.py` documents a
detail worth knowing on our side too: MiniMax-H3 expects the prompt to OPEN with a
mode-specific instruction line naming how each reference picture aligns with the target
video (from the official `VIDEO_PROMPT_WRITING_GUIDE`). Our engine does not add it -- the
facade does, and the three lines are already there verbatim -- so a request that reaches
the engine directly, bypassing the facade, is conditioned off-spec even though the
keyframe still lands. Whether upstream also TRAINED those modes is a separate question
this note does not answer; the paragraph below still holds until it is.

Two caveats that do not go away by liking the output. fl2va is a task VDN never trained
on: the keyframe IS honoured (PSNR to the reference at frame 0 is 36.43 dB against the
dense control's 36.52, i.e. the same) but the branch never saw reference media in the
prefix. And the sharpness gap conflates the hybrid attention with VDN's DMD adapter
being a different distillation from LightX2V's Turbo8 -- it is not attributable to the
attention alone. Serving it needs `VLLM_OMNI_VDN_ALLOW_UNTRAINED_TASKS=1`, which warns
on every request for exactly this reason.

## The curve-pruned tier: 19.5 GB, and how the turbo adapter gets in

The tier above (43.8 GB, peaking 39.1 of 40 GiB) leaves 0.4 GiB of headroom, which is
not enough to ship. Curve pruning is what closes that, and for a while it looked
impossible to combine with VDN.

AdaLN is ~40% of H3's weights: 50 blocks x 96768 x 2688 x 2 B = 24.2 GiB. Pruning
replaces the 2688-wide time embedding those 51 projections consume with an 8-wide sampled
basis (`adaln_proj.linear` becomes `[out, 8]` behind `adaln_basis`, plus an fp32
`folded_bias`), which is where 62 -> 37.5 GB comes from. It is not a compromise: AdaLN's
input is never an arbitrary vector, it is `silu(time_embedder(t))`, so over a whole
render it only ever takes the 1025 values the pruned table samples. Reconstructing the
base AdaLN output both ways agrees to **1.05e-04**.

The problem is that VDN's `turbo` adapter edits exactly those 51 tensors with an
`[out, 2688]` delta. `turbo` was distilled by DMD against the frozen Stage-B VDN as its
real score, so it and the branch are a matched pair -- substituting production's Turbo8
splits that pair, and the result measurably regresses.

An earlier version of this document reported that the delta could not be projected onto
the pruned basis: relative residual **0.9978**, i.e. 93% discarded. That measurement was
of a *linear* projection over all of R^2688, and it was the wrong question. Fitting on
the 1025 points AdaLN actually visits, **affinely**, is enough:

```text
S @ A.T  ~=  C @ A8.T + c        S = [1025, 2688] full post-SiLU curve
                                 C = [1025, 8]    the pruned table
design   =  [C | 1]              the intercept column is the whole trick
A8       =  (P[:8] @ A.T).T      replaces lora_A
diff_b   =  B @ (P[8] @ A.T)     added to folded_bias, fp32
```

Drop the intercept and curve reconstruction goes from 1.4e-05 to **9.5e-01** -- the curve
does not pass through the origin, and the constant it is offset by carries most of the
modulation. With it, the 51 projected modules land at 2.6e-06 .. 2.2e-05.

The method is Apache-2.0 prior art (`adaln_curve_affine_lstsq_pinv1025_with_diff_b` in
`T8mars/comfyui-minimax-h3-audio-T8`); `tools/minimax_h3/vdn_curve_projection.py` is an
independent implementation against our tensor names, agreeing with its published
per-module errors to 3%.

```bash
# 1. project turbo onto the pruned partition's curve
python tools/minimax_h3/project_vdn_turbo_to_pruned_curve.py \
  --vdn    /nfs-data/models/VDN-MiniMax-H3/stage-dmd-step-250 \
  --pruned /nfs-data/models/MiniMax-H3-Ref2VA-Pruned-r8-BF16-partition \
  --full   /nfs-data/models/MiniMax-H3/Ref2VA \
  --dst    /nfs-data/models/VDN-MiniMax-H3/stage-dmd-step-250-curve-ref2va
# 2. bake both adapters in
python tools/minimax_h3/bake_vdn_adapters.py \
  --src /nfs-data/models/MiniMax-H3-Ref2VA-Pruned-r8-BF16-partition \
  --vdn /nfs-data/models/VDN-MiniMax-H3/stage-dmd-step-250-curve-ref2va \
  --dst /nfs-data/models/MiniMax-H3-Ref2VA-Pruned-VDNfull-BF16 --link-siblings
# 3. quantize
python vllm_omni/quantization/tools/quantize_minimax_h3_int8.py \
  --src /nfs-data/models/MiniMax-H3-Ref2VA-Pruned-VDNfull-BF16 \
  --dst /nfs-data/models/MiniMax-H3-Ref2VA-Pruned-VDNfull-INT8 \
  --activation-scheme weight_only
```

`--pruned` and `--full` must be the SAME partition's pruned and unpruned forms: the fit
is per-partition (each has its own `time_embedder`), and a projection made against
another partition's curve loads without complaint while sampling `turbo` on the wrong
modulation. The same applies to `--lora-path` at serve time -- `curve-fl2va-baked` and
`curve-ref2va-baked` are not interchangeable.

**FL2VA needs one extra step.** Its pruned base is diffusers-named, and the quantizer
matches tensors by *partition* names (`blocks.N.attn.qkv_proj.weight`), so it reports
"matched no tensors". Insert `tools/minimax_h3_turbo/convert_pruned_to_partition.py`
between steps 2 and 3. (This is why the pruned FL2VA tier had never been quantized:
production was serving pruned **BF16**.)

Measured at 15 s / 768p, four official multi-shot prompts plus re-seeds:

| | ref2va | fl2va |
| --- | --- | --- |
| artifact | 19.5 GB | 19.5 GB |
| time | 370-478 s | 361-394 s |
| pairs kept | 37.4% | 22.8% |
| OOM | none | none |

Pruned and unpruned agree at the same seed (sharpness 59.4/59.2 and 127.3/131.0), so the
19.5 GB is bought without a picture cost. fl2va keeps fewer pairs because it has fewer
global rows -- one reference image against ref2va's image plus reference audio, and
global rows are dense in both directions.

## ref2va runs on a different DiT, and the transfer is unvalidated

H3 serves ref2va from `transformer_ref/`, not the `transformer/` VDN's adapters were
trained against: identical tensor NAMES, different weights. An earlier version of this
document said a delta for one "does not transfer to the other" and that the loader
refuses the task with no flag to open it. That was asserted without measuring, and both
halves are now wrong.

Measured in float64 across attention, MLP and embedder tensors spread through the depth:
cosine **0.99953**, relative L2 **3.1%** (the consistency check `1 - rel^2/2` reproduces
the cosine). `transformer_ref/` is a light fine-tune of `transformer/`, not an
independently trained model, so the trained branch is a plausible initialisation there.

So ref2va is treated like fl2va: refused by default, opened for evaluation by
`VLLM_OMNI_VDN_ALLOW_UNTRAINED_TASKS=1`, which warns on every request. What makes it a
separate case is that it needs its own `VDNCheckpoint` built on the SECOND DiT instance
-- the artifact's fusion guard is single-use and its branch tensors are consumed once, so
one object cannot feed both streams.

Plausible is not validated. Nothing downstream can tell you the transfer went wrong, and
what has been measured on it is mixed: on the official examples ref2va under VDN is
sharper than production on some prompts and not on others. Treat a ref2va rollout as
needing its own visual acceptance, not as following from the t2va result.

Two operational notes for this path, both learned the hard way:

- the ref2va partition declares `sigma_shift_scales {video: 6.0, audio: 3.0}`, not
  t2va/fl2va's 12. Production silently ignores the request's `flow_shift` and always uses
  its own; VDN's `check_request` does not. Copying a t2va script's parameters therefore
  compares VDN at shift 12 against production at shift 6.
- a VDN artifact carrying its own `turbo` adapter keeps `turbo`'s distillation contract
  (shift 12), which is a property of the ARTIFACT, not of the partition.

## Weights

```bash
# The 72 GB base, if not already present. Use the ROOT transformer/ (unfused),
# not FL2VA/transformer/ (fused) -- VDN's model_spec.json names the former.
# Then the ~5.1 GB increment:
bash scripts/download_vdn_minimax_h3.sh
```

Layout under `$DEST/VDN-MiniMax-H3/stage-dmd-step-250/`:

```text
model_spec.json                  the hybrid transform and adapter specs
metadata.json                    the training recipe (turbo_num_steps, shifts)
linear_branch/model.safetensors  the branch, softmax gate and to_out_linear
adapters/default/                a rank-64 LoRA on the attention projections
adapters/turbo/                  the 8-step DMD adapter
```

`stage-b-step-2000` is the 50-step teacher: the same branch with `adapters/default`
only, and no `adapters/turbo`.

## Serving

```bash
vllm serve /nfs-data/models/MiniMax-H3 \
  --deploy-config deploy-configs/minimax_h3_vdn8_t2va_a100_40g.yaml \
  --lora-path /nfs-data/models/VDN-MiniMax-H3/stage-dmd-step-250 \
  --trust-remote-code
```

The two production tiers serve a baked artifact instead, with the branch-only directory
on `--lora-path`:

```bash
vllm serve /nfs-data/models/MiniMax-H3-Ref2VA-Pruned-VDNfull-INT8-vLLM \
  --deploy-config deploy-configs/minimax_h3_vdn8_ref2va_pruned_a100_40g.yaml \
  --lora-path /nfs-data/models/VDN-MiniMax-H3/stage-dmd-step-250-curve-ref2va-baked \
  --trust-remote-code
```

`0 adapters (none)` in the startup line is then CORRECT -- they are in the weights. Two
means `--lora-path` still points at a directory with `adapters/`, and every LoRA is being
applied a second time on top of itself.

The increment goes through `--lora-path` but is **not** a request-switchable LoRA. Its
linear branch is an architecture change that its two adapters were trained jointly with,
so it is fused and injected at load time; serving base H3 with the branch, or the branch
without its adapters, is a different model either way.

## Constraints

Four settings produce a plausible video when wrong. The runtime refuses all four, but
they are worth stating:

| Constraint | Why | Refused by |
| --- | --- | --- |
| `diffusion_attention_backend: VDN_WINDOW_ATTN` | The branch is the window's complement; beside a dense attention it counts everything outside the window twice | `validate_hybrid_runtime` at startup |
| `ulysses_degree: 1`, `ring_degree: 1` | The branch's scan runs over frames and needs the whole target video on each rank; shard with TP, which divides heads | `validate_hybrid_runtime` at startup |
| `task: t2va` | Only T2VA was trained; fl2va/ref2va put reference media in a prefix the branch never saw, and ref2va additionally runs a different DiT (cosine 0.99953). Both are refused unless `VLLM_OMNI_VDN_ALLOW_UNTRAINED_TASKS=1` | `check_task` per request |
| `flow_shift: 12.0` (audio 3.0) | Part of the distillation, not a request preference. It belongs to the ARTIFACT, read from its `metadata.json`, and the pipeline OVERRIDES the partition's declared value with it at startup -- ref2va's partition says 6.0, and production's non-VDN engine silently ignores the request's value while VDN does not. Do not copy a t2va script's parameters to compare the two | `check_request` per request |

Clips shorter than the window fall back to dense attention automatically and the branch
contributes exactly zero — a window that covers every frame *is* the original attention.

## Validation

```bash
# Geometry, backend, TP sharding, wiring, checkpoint loading and curve projection
pytest tests/diffusion/models/minimax_h3 -k vdn

# Parity against OpenVDN's own implementation (requires their checkout): the linear
# branch AND the window softmax, which for a while was checked only against a mask this
# repository also wrote -- if our reading of the geometry had been wrong, both sides of
# that test would have been wrong together and it would have stayed green.
VDN_REFERENCE_ROOT=/path/to/vdn-minimax-h3 pytest \
  tests/diffusion/models/minimax_h3/test_minimax_h3_vdn_branch_parity.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_vdn_window_parity.py
```

The branch has no reference output that a rendered frame would reveal — a wrong forget
gate or a wrongly rebased window bound just looks slightly worse — so the oracle is the
released code itself, given the same random weights and inputs. fp32 agreement is 1.4e-5
relative; bf16 agreement is ~0.3% mean relative, the same order as the reference's own
eager-versus-fused gap.

## License

VDN-H3 is a derivative of MiniMax-H3 under the MiniMax H3 Community License Agreement,
whose applicable territory **excludes the European Union, the United Kingdom, the
Republic of Korea and the United States of America**. Read it before use or
distribution.
