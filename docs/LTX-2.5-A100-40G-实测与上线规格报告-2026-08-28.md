# LTX-2.5 在 A100-40G 上的实测与上线规格报告

> 日期：2026-08-28　硬件：dev-gpustack-a100-0041~0045，每台 4×A100-PCIE-**40GB**（无 NVLink），128 核 / 251 GB 内存
> 镜像：`reputationly/vllm-omni:arm64-a100-latest`（diffusers 0.40.0 + transformers 5.14.1 + torch/vLLM omni）
> 代码基线：本地分支 `integration/upstream-sync-20260823`（落后 upstream/main 69 个提交）

## 0. 一句话结论

**LTX-2.5 蒸馏一阶段/两阶段在 4×A100-40G 上已全部跑通并出片**，最优配置为 `int8 量化 + CPU offload + 静态形状编译`。
**尺寸选型已定稿：只用官方桶 960×544 / 1920×1088**（方案 A，见 §5.0），不裁切、不自造尺寸。
**暂缓上线的唯一原因**：1080p 档的时长天花板卡在 ~11 秒，根因是 VAE 解码后 all-gather 的 fp32 整段视频张量放不下设备，
需等上游 PR **#6477**（把视频从设备上挪走、输出传输可配置）与 **#6615**（D2H 前的视频媒体契约）合入后重测。

---

## 1. 权重

| 项 | 值 |
| --- | --- |
| 主权重 | `/nfs-models/wuhanjisuan894/models/Lightricks/LTX-2.5-Diffusers`（计算节点视角 `/nfs-data/models/...`），manifest 口径 120.09 GB |
| 下载脚本 | `scripts/download_ltx25.sh`（`scripts/` 被 .gitignore 忽略，NFS 上另存一份 `/nfs-models/wuhanjisuan894/download_ltx25.sh`） |
| 上游仓 | HF 上 `Lightricks/LTX-2.5-Diffusers` 与 `Lightricks/LTX-2.5` 均为 gated；ModelScope 镜像匿名可拉 |

### 1.1 三个会静默选错的地方

1. **`transformer/` 是蒸馏档，`transformer_full/` 才是 Full/SFT**。两者分片数与字节数完全相同（4 片 / 37.98 GB），
   只能靠 `LTX25_FULL_COMPONENT_PROFILE.transformer_subfolder = "transformer_full"` 区分。
2. **`transformer/` 里存了同一份权重的两套分片**（4 片 + 8 片），`index.json` 只引用 4 片那套；
   8 片是未被引用的重复件（37.99 GB）。上游 PR #6234 修的就是误加载它。`connectors/` 同理（单文件 6.34 GB 与 2 分片重复）。
3. **sidecar 的相对路径不能改**。`resolve_ltx_artifact()`（`vllm_omni/diffusion/models/ltx2/ltx2_components.py`）
   先查 `<model>/<相对路径>`，查不到就 `hf_hub_download(Lightricks/LTX-2.5)` —— gated 仓，离线机器必炸。所以两个 sidecar 必须按原路径落进主目录：
   - `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors`（8.900 GB）
   - `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`（0.996 GB）

   主仓根目录另有一个同名 LoRA（**9.699 GB，不是同一个文件**）且路径不对，不能拿它顶。

### 1.2 没下的部分

`prompt_enhancer/`（10.21 GB）—— vllm-omni 代码零引用，且模型卡明确写着 `processor` 与 `prompt_enhancer`
"在 `model_index.json` 里列着但本仓没发，加载为 `None`"。2.5 的提示词增强要另配 `google/gemma-4-E2B-it`。

---

## 2. 唯一可跑的配置，以及每一项为什么不能少

```bash
--quantization int8 --enable-cpu-offload --no-diffusion-compile-dynamic
--ulysses-degree N --vae-patch-parallel-size N
```

### 2.1 必须量化

40 GB 卡上的死结是 **Gemma4-12B 文本编码器 24 GB，每 rank 一份，既不能切片也不能量化**：

- LTX 没有实现 `text_encoder_tp_size`（只有 MiniMax-H3 的 `models/minimax_h3/encoder.py` 有）
- LTX 没有接 per-component 量化（只有 flux2 与 H3 接了 `resolve_component_quant_config`）；
  且 LTX 的编码器是 HF 的 `AutoModelForImageTextToText`(Gemma4)，根本不接受 `quant_config` 参数，
  要量化只能改走 transformers 侧 bitsandbytes，或照 H3 写一个 vLLM 原生 Gemma4 编码器（几百行）

所以 DiT 必须量化到单卡装得下（bf16 38 GB → fp8/int8 19 GB）。本版支持的方法：
`{fp8, mxfp8, mxfp4, mxfp4_dualscale, int8}`。**fp8 与 int8 在 A100(SM80) 上都能用，不需要 SM89。**

### 2.2 必须用 CPU offload，且不能用 HSDP / layerwise

| 方案 | 结果 |
| --- | --- |
| `--use-hsdp` | **加载完即 38.5 GB，去噪前 OOM**。HSDP 本身有效（`ltx2_transformer.py` 有 `_hsdp_shard_conditions`，DiT 确实分到 9.5 GB/卡），但 `diffusion/model_loader/diffusers_loader.py` 的 `_load_model_with_hsdp` 在分片后**无条件** `module.to(target_device)` 搬 encoders/VAE/resident，把 CPU offload 的托管放置盖掉，24 GB 编码器被钉死在卡上 |
| `--enable-layerwise-offload` | **被主机 OOM killer 杀**（exit -9）。DiT 常驻主机内存每 rank 一份，4×38 = 152 GB。`offloader/sequential_backend.py` 的 docstring 里记着 H3 的同款教训 |
| `--enable-cpu-offload` | **可用**。日志：`Model-level offloading enabled: transformer <-> text_encoder, connectors (mutual exclusion); resident on GPU: vocoder`。编码阶段卡上是编码器，去噪阶段换成量化 DiT |

### 2.3 静态形状编译是最大的一笔收益

`diffusion_compile_granularity` 默认已经是 `regional`（`config/omni_config.py:719`），日志可见
`Regional compilation applied to 48 module(s)`。真正的开关是 `--no-diffusion-compile-dynamic`：

| 规格（int8, 单卡） | dynamic=True | dynamic=False | 提速 |
| --- | --- | --- | --- |
| 960×544 / 121 帧 | 84.2 s | **44.3 s** | **1.90×** |
| 1280×704 / 121 帧 | 93.7 s | **59.3 s** | 1.58× |
| 960×544 / 249 帧 | 99.3 s | **62.4 s** | 1.59× |

**显存完全不变**（差异 < 100 MB），只改速度。

> ⚠️ **代价**：静态编译按形状编。每个新的 (宽,高,帧数) 组合都会触发一次重新编译
> （日志 `lazy regional torch.compile with dynamic=False`）。**这条尚未实测重编译耗时**，
> 但它决定了对外必须是"固定菜单 + 启动预热"，不能开放任意分辨率。

`full` 粒度与 CPU offload 不兼容，用不了。**LTX-2.5 蒸馏一阶段不支持 cache-dit**（`ltx2_recipes.py:257` `supports_cache_dit=False`）。

### 2.4 帧数与尺寸的硬约束

- **latent token 必须能被 SP 整除**：`token = (W/32) × (H/32) × ((F-1)/8 + 1)`。
  除不尽会在去噪时直接报错，且 LTX 在 `ltx2_runtime.py` 明确拒绝 `advanced_uaa` 兜底。
  例：49 帧 → 30×17×**7** = 3570，不能被 4 整除 ✗；121 帧 → 30×17×**16** = 8160 ✓
- **帧数必须是 8k+1**：121 帧 = 5.04 s，249 帧 = 10.375 s，361 帧 = 15.04 s（@24fps）
- **一阶段宽高须被 32 整除，两阶段最终尺寸须被 64 整除**。
  所以**给不了标准 1280×720**（720÷32=22.5）也给不了 1920×1080（1080÷64=16.875）。
  官方桶 960×544 / 1920×1088 都是 30:17（1.7647），偏离 16:9 仅 0.74%，直接交付即可，不必裁切 —— 详见 §5.0
- **`--vae-use-tiling` 对 LTX 是空操作**（diffusers 的 tiling 有尺寸阈值，960×544 不触发）。
  真正管用的是 `--vae-patch-parallel-size N`（LTX 的 VAE 是 `DistributedAutoencoderKLLTX2Video`，支持 patch 并行）

---

## 3. 完整实测数据

全部为蒸馏档、8 步、seed 42、24 fps。峰值显存由 `nvidia-smi` 每 10 s 采样得到，**可能漏掉瞬时尖峰**。

### 3.1 量化方式对比（单卡，dynamic 编译，短提示词）

| 规格 | fp8 | int8 | int8 优势 |
| --- | --- | --- | --- |
| 960×544 / 121 帧 | 86.4 s / 29.9 GB | **84.2 s / 28.0 GB** | 2.5% |
| 1280×704 / 121 帧 | 98.8 s / 33.5 GB | **93.7 s / 33.5 GB** | 5.2% |
| 960×544 / 249 帧 | 107.1 s / 35.2 GB | **99.3 s / 35.2 GB** | 7.3% |

A100 有 INT8 张量核（624 TOPS，bf16 的 2 倍）而无 FP8 张量核，fp8 只能反量化回 bf16 算，是净亏时间的。
int8 规格越大提速越明显，但远不到 2 倍 —— attention、VAE、编码器轮换都不是 int8 GEMM。

### 3.2 卡数扩展性（int8 + 静态编译）—— **存在交叉点**

| 规格 | token | 1 卡 | 2 卡 | 4 卡 | 结论 |
| --- | ---: | --- | --- | --- | --- |
| 960×544 / 5 秒 | 8,160 | **44.3 s** | — | 56.5 s | 单卡快 28% |
| 1280×704 / 10 秒 ¹ | 28,160 | 107.8 s / 37.3 GB | 95.0 s / 31.1 GB | **80.1 s / 31.1 GB** | 4 卡快 35% |

¹ 长提示词（198 词）

**机理**：Ulysses 的 all-to-all 通信量随 token **线性**增长，attention 计算随 token **平方**增长。
小规格通信占主导，多卡净亏；大规格计算占主导，多卡开始赚。这批机器是 PCIe 版 A100 **无 NVLink**，交叉点因此偏高。

> 早期用 fp8 + dynamic 得到的"多卡一律更慢"结论（1卡 86.4 / 2卡 104.1 / 4卡 108.6）**只在小规格成立**，不可外推。

### 3.3 规格上限（int8 + 静态编译 + 长提示词）

#### 官方桶 960×544（一阶段）

| 帧数 | 时长 | 1 卡 | 2 卡 | 4 卡 | 最优 |
| ---: | --- | --- | --- | --- | --- |
| 121 | 5.04 s | **44.3 s / 27.9 GB** ¹ | — | 56.5 s / 31.1 GB ¹ | 1 卡 |
| 249 | 10.4 s | **62.4 s / 35.2 GB** ¹ | 69.8 s / 31.1 GB | — | 1 卡 |
| 361 | 15.04 s | 80.5 s / **39.7 GB** ⚠️ | — | **75.5 s / 33.5 GB** | **4 卡** |

¹ 短提示词，其余为 198 词长提示词

15 秒档单卡只剩 **0.3 GB** 余量，不能上生产；4 卡既安全又更快。

#### 官方桶 1920×1088（两阶段，4 卡）

| 帧数 | 时长 | 耗时 | 峰值显存/卡 | 结果 |
| ---: | --- | --- | --- | --- |
| 121 | 5.04 s | 100.0 s | 31.3 GB | ✅ |
| 249 | 10.4 s | 143.3 s | 37.2 GB | ✅ |
| 281 | 11.7 s | 150.6 s | **39.0 GB** | ⚠️ 仅剩 1 GB |
| 361 | 15.04 s | — | — | ❌ **OOM**（见 §3.4） |

#### 非官方桶（已验证可跑，未采用，见 §5.0）

| 规格 | 比例 | 部署 | 耗时 | 峰值显存/卡 |
| --- | --- | --- | --- | --- |
| 1280×704 / 249 帧 | 1.818 | 4 卡 | 80.1 s | 31.1 GB |
| 1280×704 / 361 帧 | 1.818 | 4 卡 | 105.0 s | 33.5 GB |
| 1280×736 / 121 帧 | 1.739 | 1 卡 | 59.3 s | 34.1 GB |
| 1280×736 / 361 帧 | 1.739 | 4 卡 | 106.0 s | 33.5 GB |
| 1536×864 / 121 帧 | **精确 16:9** | 1 卡 | 72.4 s | 37.0 GB |
| 2048×1152 / 121 帧（两阶段） | **精确 16:9** | 4 卡 | 116.2 s | 32.1 GB |

### 3.4 1080p 时长天花板的根因（**上线阻塞点**）

15 秒那次 OOM 发生在 `distributed_vae_executor.py:61 gather_tensors` —— **去噪已经跑完**，
死在把四张卡各自解码的分片 all-gather 成整段视频时：

```text
最终视频张量 = W × H × F × 3通道 × 4字节(fp32)
1920×1088×361 → 9.05 GB   （报错要 8.46 GiB，吻合）
1920×1088×281 → 7.04 GB   （勉强过）
1920×1088×249 → 6.24 GB   （安全）
1280×704×361  → 3.90 GB   （很安全 → 非官方桶的 1280 宽档能出 15 秒）
960×544×361   → 2.26 GB   （官方桶 15 秒档，安全）
```

即 **1080p 的时长上限由输出张量大小决定，与算力无关**，约 11 秒封顶。
官方桶 960×544 不受此限（15 秒仅 2.26 GB）。

**等待中的上游修复**：

- **#6477** `[Core][Diffusion] Reduce video on the device and make output transport configurable`
- **#6615** `[Core][Diffusion] Add a typed pre-D2H video media contract`

两者合入后应重测 1080p 15 秒/20 秒档。同批还应一并同步 #6189（LTX-2.5 Diffusion VAE decoder，已合）。

---

## 4. 画质评估：PSNR/SSIM 在这个模型上不可用

同 seed、同提示词，仅改动一个变量后的逐帧相似度：

| 对比 | PSNR-Y | SSIM-Y |
| --- | --- | --- |
| int8 vs fp8（960×544/121） | 17.9 dB | 0.56 |
| int8 vs fp8（1280×704/121） | 18.8 dB | 0.66 |
| **int8 静态编译 vs int8 动态编译**（只改编译模式） | 17.5 dB | 0.54 |
| **fp8 1 卡 vs fp8 4 卡**（只改并行度） | 22.1 dB | 0.76 |
| fp8 4 卡 vs 1 卡（249 帧） | 25.6 dB | 0.87 |

**仅仅改变编译模式就产生与"换量化方式"同量级的差异**，说明 8 步蒸馏模型的去噪轨迹对任何数值扰动都极敏感，
任何改动都会走到另一条轨迹上 —— 输出"不一样"不代表"更差"。**这类指标只能证明差异存在，不能用于判优。**

而且**本硬件上拿不到 bf16 参照基准**：bf16 DiT 38 GB 单卡装不下，走 HSDP 又会把编码器钉在卡上。
结论：画质只能靠人工评审或无参考感知指标。实际评审结果为 int8 可接受。

---

## 5.0 尺寸选型：只用官方桶（已定稿）

### 这一族模型没有 16:9 的官方档

| 来源 | 尺寸 | 比例 |
| --- | --- | --- |
| 模型仓 README（2.5 全部示例） | 960×544 → 两阶段 1920×1088 | **1.7647**（30:17） |
| vllm-omni recipe（LTX-2.5，`ltx2_recipes.py:131,189`） | 同上 | 1.7647 |
| vllm-omni recipe（LTX-2 两阶段） | 1536×1024 | 1.5（3:2） |
| LightX2V `configs/ltx2/`（15 处） | 768×512 → 1536×1024 | 1.5（3:2） |
| LightX2V `configs/ltx2/`（3 处） | 1280×768 | 1.667（5:3） |
| `LTXPipelineRecipe` 类默认 | 768×512 | 1.5 |

全是"按 32 对齐凑像素预算"的桶，**没有一个是 16:9**。同类产品也一样：MiniMax-H3 的 "768p" 是
**短边 768** 口径（`MINIMAX_H3_OUTPUT_SHORT_EDGE = 768`，按比例推宽度、`_align_multiple(…, 32)` 四舍五入、
再按 `768×1344` 像素上限缩放），它的 16:9 实际输出是 **1344×768**（1.75:1），同样不标准，且**不裁回**。

### 三个方案与决策

- **方案 A（采用）**：只用官方桶 960×544 / 1920×1088。偏离 16:9 仅 **0.74%**，在 16:9 播放器上上下各留约 3 px，
  肉眼不可见；且完全落在训练分布内。
- 方案 B：改用精确 16:9 的合法尺寸。W=512k / H=288k（两阶段 W=1024k / H=576k）都能整除，
  实测 1536×864（单卡 72.4 s）与 2048×1152 两阶段（4 卡 116.2 s）**都能跑通**，但**不在任何官方桶内**，
  离开训练分布的画质风险无法用指标衡量（见 §4），且多花 16~22% 时间。
- 方案 C：生成 1280×736 再裁到精确 1280×720（实测可行，`crop=1280:720:0:8`，无重采样、音轨直接 copy，
  耗时与 1280×704 持平）。但这是在解决一个不存在的问题 —— **官方桶 960×544 的 0.74% 偏差比 1280×736 的 2.17% 还小**。

**结论：采用方案 A。** 不裁切、不自造尺寸，只暴露 960×544 / 1920×1088 及其竖屏形式。

## 5. 对外菜单（方案 A，PR 合入后重新验证再定稿）

| 档位 | 尺寸 | 时长 | 部署 | 实测耗时 | 峰值显存/卡 | 整机(4卡)吞吐 |
| --- | --- | --- | --- | --- | --- | --- |
| 标准 | 960×544（+竖屏 544×960） | 5.04 s | 1 卡 | 44.3 s | 27.9 GB | **5.4 条/分** |
| 标准 | 960×544 | 10.4 s | 1 卡 | 62.4 s | 35.2 GB | 3.8 条/分 |
| 标准 | 960×544 | 15.04 s | **4 卡** | 75.5 s | 33.5 GB | 0.79 条/分 |
| 高清 | 1920×1088（两阶段） | 5.04 s | 4 卡 | 100.0 s | 31.3 GB | 0.60 条/分 |
| 高清 | 1920×1088（两阶段） | 10.4 s | 4 卡 | 143.3 s | 37.2 GB | 0.42 条/分 |

- 15 秒档**必须用 4 卡**：单卡 80.5 s / **39.7 GB**（仅剩 0.3 GB），且比 4 卡的 75.5 s 还慢
- 10 秒档**必须用 1 卡**：2 卡是 69.8 s，反而比单卡的 62.4 s 慢（16,320 token 仍在交叉点左侧）
- 1080p **给不到 15 秒**（§3.4），最多 11.7 秒且余量仅 1 GB，对外只承诺到 10 秒
- 竖屏与横屏 token 数相同，耗时显存一致，但**各自占一个编译形状**

部署要点：

- **一机四卡 = 4 个独立单卡实例**（标准 5 秒/10 秒档），不是一个四卡实例；15 秒档与 1080p 档整机编一组
- 1080p 与 15 秒档单独分机器，别跟单卡池混编 —— 一个这类请求会占住整机四卡
- 扩散任务串行（`max_num_seqs=1`），网关必须有队列 + 预估等待
- **每个形状在实例启动时做一次预热请求**，否则第一个真实用户吃到编译时间
- **不开放任意分辨率**，只接受菜单内的枚举值（静态编译按形状编）
- **提示词必须长**：模型卡原话是"在长段单段落视听描述上训练，短提示词会明显劣化"。前端要引导写整段（镜头/运动/光线/声音），或在门面层挂中译英 + 改写

---

## 6. 上线前仍欠的验证

1. **等 #6477 / #6615 合入后重测 1080p 15 秒/20 秒**（本报告的阻塞点）
2. **换形状的重编译耗时**未实测 —— 决定菜单能开多少个形状
3. **稳态单请求耗时**未实测 —— 本报告所有数字都含首次编译开销，真实值更低。需起 `vllm serve` 连发多请求测
4. **`/v1/videos/sync` 在线接口与网关对接**未验证
5. **1280×704/10 秒的单卡形态**只测了 2 条提示词，余量 2.7 GB，需 20~30 条不同长度题材的提示词压测确认无 OOM 尾部
6. **峰值显存测量精度**：现为 10 s 采样，应改用 `torch.cuda.max_memory_allocated` 或开 `--enable-diffusion-pipeline-profiler`

---

## 7. 复现方式

实验脚本在 `scripts/`（本地，未入 git）与调试机 `gpu41~45:/root/ltx25/`：

| 文件 | 用途 |
| --- | --- |
| `scripts/download_ltx25.sh` | 权重下载（ModelScope 主源、断点续传、逐文件字节校验） |
| `scripts/ltx25_go.sh` | 单次实验启动器，含 token 整除自检 |
| `scripts/ltx25_t2v.py` | 官方离线例子 + 本地补丁（暴露 `--diffusion-compile-granularity` / `--no-diffusion-compile-dynamic`） |
| `scripts/ltx25_prompt_long.txt` | 198 词的业务级长提示词 |

```bash
# 单卡标准档
QUANT=int8 bash /root/ltx25/ltx25_go.sh 名称 "0" 1 960 544 121 \
  LTX2DistilledOneStagePipeline --no-diffusion-compile-dynamic

# 4 卡 1080p 两阶段
QUANT=int8 PROMPT="$(cat /root/ltx25/ltx25_prompt_long.txt)" \
  bash /root/ltx25/ltx25_go.sh 名称 "0,1,2,3" 4 1920 1088 249 \
  LTX2DistilledTwoStagePipeline --no-diffusion-compile-dynamic
```

全部成片（18 个）在 `/nfs-output/ltx25_20260828/`。

---

## 8. 与官方口径的差距

官方宣称"几秒钟生成"，本硬件达不到，差距在硬件不在配置：

- **A100(SM80) 无 FP8 张量核**，官方数字是 H100/B200 上用真 fp8 算力跑的（H100 fp8 ≈ 2000 TFLOPS vs A100 bf16 ≈ 312）
- **PCIe 版无 NVLink**，多卡扩展性受限（见 §3.2 交叉点）
- `recipes/LTX/LTX-2.5.md` 写"1920×1088 两阶段峰值约 114 GB、80 GB 卡都不够"—— 那是 bf16 的账；
  **int8 + CPU offload 之后每卡只要 31~39 GB，4×40 GB 完全跑得动**，该文档的硬件门槛描述对量化路径不适用

商业 API 侧的规格上限（与开源权重不完全等同）：Fast 档最高 4K、6~20 秒；Pro 档 1080p、6/8/10 秒、24/25/50 fps；
且不能自由组合 —— 超过 10 秒掉到 720p/1080p @24/25fps，48/50fps 最长 10 秒。
