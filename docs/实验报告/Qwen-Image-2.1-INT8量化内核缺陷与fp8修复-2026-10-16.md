# Qwen-Image-2.1 INT8 量化内核缺陷与 fp8 修复实测（2026-10-16）

> 测试对象：`qwen-image-pro`（Qwen-Image-2.1-Turbo）在 4×A100-40G 上的速度、显存与画质。
> 起因：线上"一步要几十秒"的反常现象。
> 本文自包含：根因、证据、修复、全量对照数据、坑清单、复现命令。

---

## 1. 一句话结论

**`--diffusion-quantization-config={"method":"torchao_int8_weight_only"}` 让整个模型以 1.3% 的算力在跑。**

它的 int8 矩阵乘落在 PyTorch 的参考实现 `weight_int8pack_mm_kernel` 上（**不用 Tensor Core**），
该内核占 **99.9%** 的 GPU 时间。换成 fp8 后：

| 场景 | int8（原） | **fp8 + `*img_mlp*`（现）** | 提速 |
|---|---|---|---|
| t2i 1024² | 217 s | **3.2 s** | **69×** |
| t2i 2048² | 866 s | ~10.5 s | **82×** |
| i2i 10 图 | 1350 s | **64 s** | **21×** |

显存基本不变（19.8 GB → 23.9 GB），10 图 i2i 仍可跑。

**同一集群的 H3 用的是另一套离线 W8A16 方案，只比 BF16 慢 4%，无需改动**（§9）。

---

## 2. 现象与定位过程

### 2.1 现象

平台侧"一步要几十秒"。实测（4 卡 SP=4、8 步档）：

```text
512²       6.91 s/步
768²      15.39 s/步
1024²     27.33 s/步
1440×1920 71.60 s/步
```

**每步耗时严格正比于 token 数**（拟合 `T=a·L+b·L²` 得 `b≈-7e-9`，二次项占比 **1.1%**）——
**据此排除"O(N²) heavy attention 主导"**，瓶颈在线性层。

### 2.2 与硬件算力的差距

```text
每步实际消耗:              1081 GFLOP / token
模型结构所需(2×7e9 参数):    14 GFLOP / token
                        ⇒ 77 倍
```

硬件实测（同机 A100-PCIE）：

```text
BF16 8192² GEMM : 161.3 TFLOPS   ← 打满（纸面 ~156）
SM 时钟         : 全程 1410 MHz，无降频
SM 利用率       : 96–100%        ← GPU 确实在忙
功耗            : 119 W / 250 W  ← 但只有一半
```

**硬件健康、GPU 忙，但有效算力只有约 2 TFLOPS（峰值的 1.3%）。**

### 2.3 算子级剖析（决定性证据）

```text
墙钟 216.0 s   CUDA 时间合计 215.5 s   GPU 占比 99.8%

  99.9%  n=1280  215.25s  at::native::weight_int8pack_mm_kernel(float const*, signed char ...)
```

**一个内核吃掉 99.9% 的时间。** 调用次数结构性吻合：

```text
n = 1280 = 32 层 × 5 个线性层 × 8 步
```

其余全部算子合计 < 0.3 s（attention 仅 0.03 s）。

**根因：`weight_int8pack_mm_kernel` 是 PyTorch 的 int8 weight-only 参考实现，
在 CUDA 上做"逐元素反量化 + FMA"，完全不使用 Tensor Core。**

Fixed/Per-token 拆分更清楚：

| | 固定开销/步 | 每 token 开销 |
|---|---|---|
| BF16 | 0.166 s | 4.88e-5 s |
| int8 | 0.110 s | **6.64e-3 s** |
| | ↓ int8 权重小一半，固定开销确实更低 ✓ | **↑ 每 token 慢 136 倍** ← 内核劣化处 |

---

## 3. 为什么一直没被发现

查阅 `q21_probe_r1_fp8fail/probe.log`（2026-09-20，镜像 `arm64-a100-q21`）：

```text
✗ t2i_1024_fp8   → RuntimeError: Orchestrator initialization failed:
                    cutlass_scaled_mm_sm80_epilogue,
                    .../quantization/w8a8/cutlass/scaled_mm_c2x.cu:89
✗ i2i_1024_fp8   → 失败
✗ t2i_2048_fp8   → 同样崩溃
✓ t2i_1024_bf16  → 成功，40 步 18.33 s，峰值 39.38 GiB
```

**⇒ 当时 fp8 在 A100 上直接崩（vLLM 自带 cutlass w8a8 内核在 sm80 有 bug），
只有 BF16 可用，但 BF16 峰值 39.4 GB —— 40 GB 卡几乎顶满，10 图 i2i 必然 OOM。**

**⇒ 于是选了 `torchao_int8_weight_only`（显存 19.8 GB）。**

**但同一份日志里，BF16 那档引擎自报 `denoise_step_latency_ms = 458.1`（0.46 秒/步），
而 int8 是 27.25 秒/步 —— 慢 59 倍。选型时只看了显存，没看速度。**

**vLLM 0.31 rebase 后 fp8 路径已改变**（见 §5），fp8 现在可用。

---

## 4. 修复与验证

### 4.1 配置改动（GPUStack 后端参数，仅一条）

```diff
- --diffusion-quantization-config={"method":"torchao_int8_weight_only"}
+ --diffusion-quantization-config={"method":"fp8","ignored_layers":["*img_mlp*"]}
```

**环境变量无需改动**（`VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200` 保留，勿删）。

### 4.2 线上验证结果（2026-10-16 17:44，三副本全量重建）

启动日志关键行：

```text
Building quantization config: fp8
Your GPU does not have native support for FP8 computation but FP8 quantization is being used.
Weight-only FP8 compression will be used leveraging the Marlin kernel.

Marlin requires thread-tile padding for some weight shapes in this model.
Activations and/or outputs of the padded layers are padded/sliced on every forward;
performance may be degraded.
```

| 检查项 | 实测 | 判定 |
|---|---|---|
| 三副本配置 | 0001/0002/0003 命令均为新配置 | ✅ |
| **空闲显存** | **23,935 MiB** | ✅ 与基准值分毫不差 → `*img_mlp*` 生效 |
| t2i 512² | 1.36 s | ✅ |
| t2i 1024² | 3.16 s | ✅ |
| i2i 4 图 | 33.5 s / 峰值 28,529 MiB | ✅ |
| i2i 10 图 | **64.2 s / 峰值 37,931 MiB** | ✅ 未 OOM（余量 ~2.5 GB）|

---

## 5. 关于 A100 上的 fp8：它就是 W8A16

A100 是 **sm80，没有 fp8 Tensor Core**（fp8 从 sm89/Ada、sm90/Hopper 起才有）：

```text
torch._scaled_mm(fp8) → "only supported on CUDA devices with compute capability >= 9.0 or 8.9"
vllm platform.supports_fp8() = False
vllm QuACK fp8 GEMM          = tcgen05 指令，Blackwell 专用
```

**⇒ A100 上 `{"method":"fp8"}` 必然是"权重 fp8 + 激活 bf16"**：

- **权重 FP8（1 字节）** → 显存与 int8 同级
- **Marlin 内核**（vLLM 优化的混合精度 GEMM，走 Tensor Core）→ 速度与 BF16 同级
- **激活全程 BF16** → 精度损失仅来自权重

实测完全印证：fp8 速度 2.85 s ≈ BF16 的 2.51 s；fp8 显存 19.3 GB ≈ int8 的 19.8 GB。

> 文档 `docs/user_guide/quantization/fp8.md:29` 写的是 *"FP8 on Ampere **may** use a
> weight-only path where available"* —— 实测在 A100 上这不是 "may"，而是 **"only"**。

### 5.1 在线量化 vs 离线量化

当前用的是**在线量化**：

```python
class DiffusionFp8Config(OnlineQuantizationConfig):
    is_checkpoint_fp8_serialized = False    # 磁盘 checkpoint 是纯 BF16
    activation_scheme = "dynamic"
    QuantSpec(weight="fp8_per_tensor_static")
```

验证：`Qwen-Image-2.1-Turbo/model_index.json` 无 `quantization_config`；
`transformer/*.safetensors` 的 86 个张量全是 `BF16`。
**权重在加载时现场转 fp8，无需标定。**

- ✅ 优点：零标定成本，任意 BF16 checkpoint 可用
- ⚠️ 限制：代码显式拒绝 `weight_block_size`，权重只能 per-tensor 缩放

NFS 上**没有** Qwen-Image 的预量化离线版本，需要自己跑 ModelOpt 才能获得 per-block 缩放。

---

## 6. 三种精度全量对照（vs BF16 基线，同 prompt 同 seed，skimage 窗口化 SSIM）

### 6.1 指标

| 场景 | 变体 | PSNR | SSIM | 差异>16 占比 |
|---|---|---|---|---|
| 多字（国庆海报） | fp8 | 15.44 | 0.8412 | 17.0% |
| | **fp8+`*img_mlp*`** | **18.00** | **0.8748** | 14.1% |
| 小字（书页） | fp8 | 25.40 | 0.9032 | 9.4% |
| | **fp8+`*img_mlp*`** | **26.39** | **0.9206** | 7.8% |
| 风景编辑 | fp8 | 42.99 | 0.9934 | 0.0% |
| | **fp8+`*img_mlp*`** | **43.69** | **0.9937** | 0.0% |
| 人脸编辑 | fp8 | 45.64 | 0.9944 | 0.0% |
| | **fp8+`*img_mlp*`** | **46.41** | **0.9955** | 0.0% |
| 4 图多参考 | fp8 | 23.73 | 0.9088 | 9.3% |
| | **fp8+`*img_mlp*`** | **35.91** | **0.9795** | 1.5% |
| 10 图多参考 | 仅 fp8 可跑 | — | — | BF16 基线 OOM |

**汇总（vs BF16）：**

| 变体 | SSIM 均值 | SSIM 最低 | PSNR 均值 | PSNR 最低 |
|---|---|---|---|---|
| fp8 | 0.9282 | 0.8412 | 30.64 dB | 15.44 dB |
| **fp8 + `*img_mlp*`** | **0.9528** | **0.8748** | **34.08 dB** | **18.00 dB** |

### 6.2 肉眼结论（对照图见附件）

- **文字类场景：三个版本的文字内容、字形、笔画完全一致**。多字场景 15.4 dB 的差异
  **全部来自装饰元素**（烟花位置、白鸽姿态），**不是文字渲染出错**。
- **图片编辑类场景（人脸/风景）肉眼无差**（SSIM 0.994 / 0.993）。
- **`*img_mlp*` 的收益集中在多参考融合（0.909→0.980）与多字场景**。

### 6.3 BF16 的显存上限（i2i 参考图数量，1024²）

| 张数 | 结果 | 峰值/卡 |
|---|---|---|
| 1 | ✅ | 33,643 MiB |
| 2 | ✅ | 35,003 MiB |
| **4** | ✅ | 38,183 MiB |
| 6 / 8 / 10 | ❌ **OOM** | ~40.2 GB（撞顶）|

**⇒ BF16 最多 4 张参考图；fp8 可到 10 张（峰值 37.9 GB）。**

---

## 7. ⚠️ 坑清单（每条都踩过）

### 7.1 `ignored_layers` 是 **fnmatch 通配符**，写错静默失效

vLLM 的在线量化：

```python
ignore: list[str] = ...
"""Layers to skip quantization for. Online quantization also supports fnmatch-style patterns."""
...use_fnmatch=True
```

**没有通配符 = 精确匹配。** `["img_mlp"]` 匹配不到 `transformer_blocks.22.img_mlp`：

| 配置 | 空闲显存 | 说明 |
|---|---|---|
| `fp8` | 19,317 MiB | 基准 |
| `fp8` + `["*img_mlp*"]` | **23,935 MiB** | ✅ 层被跳过（保持 bf16，故显存升高）|
| `fp8` + `["img_mlp"]` | 19,317 MiB | ❌ **与基准分毫不差 ⇒ 一个层都没匹配** |

**不报错、不警告，只能靠显存变化发现。**
**⇒ `ignored_layers` 是否生效，唯一可靠的判据是显存。**

### 7.2 GPUStack 改模型配置**不会热生效**

改完 `PUT /v2/models/{id}` 后 `updated_at` 会变，但**运行中的副本不受影响**
（实测 `restart_count=0`，容器创建时间不变）。**必须重建实例。**

### 7.3 GPUStack 用浮动 tag `arm64-a100-latest`，事后无法追溯

模型规格 `image_name = None`，镜像由 `backend`+`backend_version` 解析成浮动 tag；
而实际跑的是**节点本地缓存的那一版**。GPUStack **不记录 digest**，
实例日志也在删除实例时一并删除（`Deleted serve log file`）。

**⇒ 任何一次部署当时用的是哪个构建，事后无法追溯。建议显式钉不可变 tag。**

### 7.4 vLLM-Omni 的模型名校验是**严格相等**

```python
if request.model is not None and request.model != model_name:
    raise HTTPException(400, "Model mismatch: ...")
```

服务标识是模型路径，所以传 `model: "qwen-image-pro"` 会 **400**。
**基线时期同样如此，非本次引入。** 调用方要么不传 `model`，要么传完整路径。

### 7.5 图片路径不记录请求参数

`/var/lib/gpustack/log/serve/*.log` 里图片请求只有访问日志：

```text
INFO:  10.0.0.139:35022 - "POST /v1/images/generations HTTP/1.1" 200 OK
```

**没有尺寸、没有步数、没有耗时。** GPUStack 侧无法回溯请求参数。

---

## 8. 复现材料

### 8.1 关键数据文件（`/nfs-output/`）

| 路径 | 内容 |
|---|---|
| `qcmp/BF16/`、`qcmp/fp8/`、`qcmp/fp8mlp/` | 各精度五个场景的输出 PNG |
| `qcmp/_montage/`、`qcmp/_montage2/` | 三方并排对照图 |
| `qmat/` | 10 张 t2i 生成的测试素材 |
| `prof_trace.json` | 完整 chrome trace（18,605 个 kernel）|
| `qmetrics.py`、`qmetrics2.py` | 指标计算脚本 |
| `scenarios.py`、`gen_materials.py`、`prof3.py` | 场景/素材/剖析脚本 |

### 8.2 本仓库脚本

```text
benchmarks/diffusion/quantization_quality.py   # 官方量化质量方法论（LPIPS）
vllm_omni/quantization/fp8_config.py           # DiffusionFp8Config（在线 fp8）
vllm_omni/quantization/tools/quantize_minimax_h3_int8.py   # H3 的离线 int8 工具
```

### 8.3 文档附件

`docs/实验报告/附件/qwen-image-int8-kernel/` 下四张三方对照图：

- `qwen-int8fp8-多字对照.jpg`（BF16 / fp8 / fp8+`*img_mlp*` 三方并排）
- `qwen-int8fp8-小字对照.jpg`
- `qwen-int8fp8-人脸对照.jpg`
- `qwen-int8fp8-四图对照.jpg`

> 为控制仓库体积，附件为 JPEG q92 压缩版（共 ~1 MB）；**未压缩的 PNG 原图**在
> `/nfs-output/qcmp/_montage/` 与 `_montage2/`。

---

## 9. 横向：H3 不受影响（对照结论）

`minimax-h3-fl2va` / `minimax-h3-ref2va` 用的是**完全不同的路径**：

```text
路径: MiniMax-H3-Singularity-VDN-INT8
transformer/config.json:
  {"quant_method": "int8",
   "is_checkpoint_int8_serialized": true,
   "activation_scheme": "weight_only",          ← W8A16（激活不量化）
   "ignored_layers": ["audio_patch_proj", "blocks.0.adaln_proj.linear", ...]}
```

- **离线**量化（预量化 checkpoint），非在线；**没有 `--diffusion-quantization-config`**
- `activation_scheme: weight_only` **正是** `docs/minimax-h3-ref2va-剪枝量化全量实测-2026-08-22.md`
  推荐的配置（该文档明确 **不要用 W8A8**：会削掉高动态瞬态事件与运动细节）
- 该文档实测：**W8A16 相对 BF16 只慢 4%**（651 s vs 624 s），显存 −27%，画质回到基准

| | Qwen-Image（已修） | **H3（健康）** |
|---|---|---|
| 量化方式 | `torchao_int8_weight_only`（在线） | 离线 W8A16 |
| 相对 BF16 | **慢 60–80 倍** ❌ | **只慢 4%** ✅ |
| 内核 | PyTorch 朴素 `weight_int8pack_mm` | vLLM cutlass int8 GEMM |

**⇒ 结论：`int8 权重 + bf16 激活`这个方案本身不慢；慢的是 torchao 的那个内核。
H3 无需改动。**

> 推论：理论上可以给 Qwen-Image 也做离线 int8 W8A16 checkpoint 以获得更好速度，
> 但显存与 fp8 相同（都是 8 bit 权重），而 fp8 已达成 3 s/1024²，**收益有限，暂不做**。

---

## 10. 遗留项

| # | 事项 | 说明 |
|---|---|---|
| 1 | **Marlin padding 提示** | 部分层 shape 不匹配做了 padding，每步多一次切片；当前速度已可接受，可后续优化 |
| 2 | **10 图 i2i 余量仅 ~2.5 GB** | 11 图以上或更大尺寸会 OOM，建议加上限告警 |
| 3 | **出图与 int8 时代不同** | 量化方式变化导致同 seed 出图不同（`*img_mlp*` 后 SSIM 0.9528），需业务侧确认 |
| 4 | **离线 fp8 未评估** | per-block 缩放可能让 `*img_mlp*` 都不需要，显存回到 19.3 GB；需 ModelOpt 标定流程 |
| 5 | **其他模型未扫** | `ltx2.5`(int8)、`kr2`(int8)、`id4`(bitsandbytes) 等在册但副本=0，未运行 |

---

## 11. 观察项（上线后 1–2 天）

| 观察 | 期望值 | 异常信号 |
|---|---|---|
| 空闲显存 | 23,935 MiB | 掉回 19,317 → 配置未生效 |
| t2i 1024² 耗时 | ~3 s | 涨到几百秒 → 量化回退 |
| 10 图 i2i | ~64 s | 500 错误 → OOM |
| GPU 峰值 | < 38 GB | 接近 40 GB → 需限流 |
