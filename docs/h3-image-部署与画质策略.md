# MiniMax-H3-Image 部署与画质策略

适用：`MiniMax-H3-Image`（16 MP 类图像模型）在 vLLM-Omni + GPUStack 上的部署、容量选档与画质治理。
数据全部来自本仓实测（4×A100-40G，TP4 + TE TP4，`torch 2.x / vLLM 0.33.0`），采集脚本见
`bench/prod-baseline-20261114/`，原始图片与对照在 NFS `/nfs-output/prod-baseline-20261114/`。

---

## 1. 与官方 release 的差异（只列影响画面或接口语义的）

| 维度 | 官方（README / examples） | 本仓实现 | 是否改画面 |
| --- | --- | --- | --- |
| 输出面积 | 1K / 2K / **4K（best results）**；自定义 655,360–20,000,000 px | 同左，**超限 400 拒绝** | 是（改了默认尺寸） |
| 默认尺寸 | API `auto` = 4K 面积 | **不传 size 时 4096×4096** | 是（默认从 1024² → 4K） |
| 宽高比 | `size` WIDTHxHEIGHT，或 `area`+`aspect_ratio`；比例 1:3–3:1 | 新增 `aspect_ratio`+`resolution`（11 档），仍支持 `size`；比例 >3:1 拒绝 | 是（新增入口） |
| 参考图数量 | ≤16 | ≤16，超出 400 | 否 |
| 参考图规格 | 每张 ≤64 MiB、边长 128–8192 px、比例 ≤4:1 | 同左，**全部 400 拒绝**（此前是静默 resize，且放宽到 8:1） | 是（不再接受超纲输入） |
| `min_generation_area` | 默认 4194304（2048²），范围 1024–16777216 | 完全一致（沿用官方默认） | — |
| prompt 扩写 | API 默认扩写（编辑恒定开启）：`H3-Image-Context-IR` / 本地 `MiniMax-H3-Image-PE` | **不做扩写**（由调用方或用 PE/skill 预处理） | 是（见 §6） |
| 并行 | 4 卡推荐 `--usp 4`（Ulysses SP） | 40G 卡只能用 TP4（SP4 需 81.3 GiB/卡） | 否（像素级微差） |
| 编译 | 上游默认行为 | **DiT 保持 eager**（编译会改 DiT 数值且无实测收益） | 否（更保守） |
| 参考-KV 复用 | 官方 H3-Image **无**此能力 | fork 自研接线（默认关、有损） | **是（默认关）** |
| 参考总像素预算 + 显存守护 | 官方无 | fork 自研（默认 `auto`，见 §3） | **是（>8 张或大图时降采样）** |

---

## 2. 11 档宽高比 × 3 档分辨率（New API）

请求侧：`size` 与 `aspect_ratio` **互斥**；`aspect_ratio` 可配 `resolution`（`1k`/`2k`/`4k`，默认 `4k`）。

```json
{"prompt": "...", "aspect_ratio": "16:9", "resolution": "4k"}
```

| 比例 | 4k（16 MP，官方原值） | 2k（≈4.2 MP） | 1k（≈1.05 MP） |
| --- | --- | --- | --- |
| 1:1 | 4096×4096 | 2048×2048 | 1024×1024 |
| 4:3 / 3:4 | 4736×3552 / 3552×4736 | 2368×1760 / 1760×2368 | 1184×896 / 896×1184 |
| 3:2 / 2:3 | 5024×3360 / 3360×5024 | 2496×1664 / 1664×2496 | 1248×832 / 832×1248 |
| 16:9 / 9:16 | 5472×3072 / 3072×5472 | 2720×1536 / 1536×2720 | 1376×768 / 768×1376 |
| 21:9 / 9:21 | 6272×2688 / 2688×6272 | 3136×1344 / 1344×3136 | 1568×672 / 672×1568 |
| 5:4 / 4:5 | 4576×3648 / 3648×4576 | 2304×1824 / 1824×2304 | 1152×928 / 928×1152 |

- 4k 是官方 16 MP 尺寸原值（README + `size_for()`），2k/1k 按同比例缩到 32 的整数倍。
- `1k` 档仍会被 `min_generation_area=2048²` 托到 2048² 画布再缩回（官方行为），耗时 ≈ 2048² 请求。
- 实测：`4096²` 21.1 s、`16:9@4k` 21.0 s、`21:9@2k` 5.05 s（单请求、4×A100-40G）。

---

## 3. 参考图预算与显存守护（fork 自研，默认 `auto`）

### 3.1 三个概念

| 概念 | 含义 | 位置 |
| --- | --- | --- |
| **短边带** | 每个消费者各自把参考图短边夹到区间：vision **1024**、VAE **1024–2048** | `_bound_references()` |
| **总像素预算** | 所有参考图总面积上限，超出则**同一比例**缩小（保相对大小与比例） | `reference_image_total_pixels` |
| **显存守护** | 跑之前用实测模型预测峰值，超了就先把预算降下来，避免 OOM | `_fit_reference_budget()` |

### 3.2 开关

```json
--additional-config '{"reference_image_total_pixels":"auto","reference_image_total_pixels_cap":33554432}'
```

- `auto`：预算 = `cap`（默认 33,554,432 ≈ 8×2048²），**再由显存守护按请求收紧**；
- 显式数字：固定预算；`0`：关闭（不建议，等于放弃兜底）。

### 3.3 实测：预算 → 耗时/显存（8×4K 参考，2048² 输出）

| 预算 | 每张等效短边 | 耗时 | 峰值/卡 |
| --- | --- | --- | --- |
| 8.4 Mpx | ~1024² | 34.4 s | 32.8 GiB |
| 16.8 Mpx | ~1448² | 67.9 s | 34.6 GiB |
| 25 Mpx | ~1768² | 112.2 s | 36.8 GiB |
| 33.5 Mpx（=不压） | 2048 短边 | 151.2 s | 37.6 GiB（**余量 1.8 GiB**） |

### 3.4 实测：显存模型（守护依据）

```text
峰值/卡 ≈ 28.3 GiB（4×TP4 常驻）+ 2.9 GiB（激活地板）
        + 0.19 GiB × 参考 Mpx
        + 0.8 GiB（参考数 > 8）
        + 0.2 GiB（画布 ≥ 16 MP）
allowance = device_total − 2 GiB
```

拟合点：8.4/16.8/25/33.5 Mpx × 8 参考 = 32.8/34.6/36.8/37.6 GiB；16 参考 33.5 Mpx = 38.5 GiB；
16 参考 33.5 Mpx + 8K 画布 = 38.6 GiB。**显存由参考像素主导，画布是次要项**（2K→8K 仅 +0.2 GiB/卡）。

### 3.5 守护实测（`auto` + cap 33.5 Mpx）

| 请求 | 降档 | 耗时 | 峰值/卡 |
| --- | --- | --- | --- |
| 8×1024² | 不降 | 31.3 s | 134334 MiB（与无守护逐像素一致） |
| 8×4K | 33.55M→33.35M | 141.5 s | 154198 MiB |
| 16×4K | 33.55M→**29.14M** | **130.4 s** | **152998 MiB** |

OOM 兜底策略：捕获 `torch.OutOfMemoryError` → `empty_cache()` → **单次**重试 `min(预算/2, 16.8 Mpx)`；再失败即 500。
注意：**硬 OOM 后进程碎片无法完全回收**（实测紧随其后的正常请求峰值会高 16 GiB），所以主防线是守护，不是重试。

---

## 4. 画质影响登记表（上线任何开关前必查）

### A. 会降画质

| 开关 | 层级/默认 | 损失 | 实测 |
| --- | --- | --- | --- |
| `reference_image_total_pixels` | 部署/`auto` | 参考降采样 | 8×4K 全压→SSIM 0.50；8×1024² 不触发→**SSIM 1.0000** |
| `--kv-offload-config`（参考-KV） | 部署/**关** | 参考 K/V 跨步复用（参考驱动区细节漂移） | 1–4 参考 SSIM 0.86–0.98；8 参考 0.66–0.91；耗时 −1~−19% |
| `FFN_CHUNKS` / head-group group（`low_vram.py`） | 部署/**关** | 分块改变 all-reduce 求和顺序 | 未在图像上量化，属破坏数值等价 |
| `output_format=jpeg/webp` | 请求/`png` | 有损编码 | — |
| int8 序列化 checkpoint | 由权重决定 | 文本嵌入量化 | 现用 BF16 不触发 |

### B. 改行为不改好坏

| 参数 | 层级/默认 | 影响 |
| --- | --- | --- |
| `min_generation_area` | 部署/4194304 | 实际生成画布（小尺寸请求被托到 2048² 再缩） |
| `size` / `aspect_ratio` / `resolution` | 请求 | 输出尺寸与比例 |
| 参考图顺序 | 请求 | `<Picture i>` 语义绑定 |
| `seed` + `generator_device=cpu` | 请求 | 同 seed 同噪声（官方 API 不支持指定 seed，本地支持） |

---

## 5. 部署配方

```bash
# 4×A100-40G（本集群）
vllm serve <path>/MiniMax-H3-Image --omni --port 40042 \
  --num-gpus 4 --tensor-parallel-size 4 --text-encoder-tp-size 4 \
  --allowed-local-media-path /nfs-output \
  --max-generated-image-size 20000000 \
  --additional-config '{"reference_image_total_pixels":"auto","reference_image_total_pixels_cap":33554432}'
```

- 80G 卡（H200/B200）优先官方 `--usp 4`：SP4 更快（H200 实测 0.88 s vs TP4 1.0 s），但需 81.3 GiB/卡；
- 单 H200：`--num-gpus 1 --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}'`；
- GPUStack：用**模型级 `image_name`** 覆盖镜像，避免影响同后端的其它模型；
- 环境：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（降碎片，配合薄余量配置）。

**副本独占**：33.5 Mpx 档余量只有 ~1.8 GiB/卡，若与其它副本共卡会立刻 OOM；保守档 16.8/25 Mpx。

---

## 6. 画质与 prompt

- 官方最佳在 **~16 MP**：追求画质就请求 4K 档（4096² / 4736×3552 / 5472×3072 / 6272×2688），不要用小尺寸再让它缩；
- **不做 prompt 扩写**是本仓与官方 API 最大的画质差距：官方 API 默认扩写（编辑恒定）；本地可在生成或编辑前使用 Context-IR CLI、`MiniMax-H3-Image-PE` 模型或 `h3-image-prompt-writing` skill；
- 官方 prompt 规范（`skills/h3-image-prompt-writing/references/edit-en.txt`）：一段英文，顺序为
  **源定义 → 目标概述与逐 Subject 关系 → 改动细节**；用 `<Picture N>`/`<Subject N>`，关系句必须显式
  （`Exactly <Subject N> is fully preserved.` / 部分保留 / 属性迁移 / 弱参考）；要显示的文字逐字引用并给位置；
  不要文字时写 `no text, letters or numbers in the image`；网格写 "2 rows and 3 columns" 而非 "2×3"。

### 本地 Context-IR 增强

`tools/minimax_h3/context_ir.py` 可在调用本地 H3-Image 前使用 OpenAI 兼容的对话接口完成提示词增强、规范校验和最多两次校验重修。它不自动接入服务端请求；使用者应将输出的 `revised_prompt` 和 `aspect_ratio` 用于后续图像请求。官方 PE 提示词受 MiniMax community license 限制，仓库不包含提示词原文；运行前从 Hugging Face 下载到本地：

```bash
huggingface-cli download MiniMaxAI/MiniMax-H3-Image-PE --include 'prompt/*'
export OPENAI_BASE_URL=https://maas.ovaijisuan.com/v1
export OPENAI_API_KEY=<your-api-key>
python tools/minimax_h3/context_ir.py \
  --model qwen3.8-flash-fp8 \
  --pe-dir ~/.cache/huggingface/hub/<downloaded-model-snapshot> \
  --image /path/to/reference.png \
  --prompt '把标题改成“秋季上新”，其它不变'
```

T2I 请求不传 `--image`；多个参考图可重复传入 `--image`。默认自动选择宽高比并最多重修两次。`--reasoning-effort` 可覆盖模型档位；未知模型默认使用 `low`。

| MaaS 模型 | Edit reasoning effort | T2I reasoning effort | 建议 |
| --- | --- | --- | --- |
| `qwen3.8-flash-fp8` | `none` | `low` | Edit 首选 |
| `deepseek-v4-1-flash` | `none` | `none` | Edit 备选；T2I 响应较快 |
| `glm-5.3-flash-w4a8` | `low` | `low` | 通用备选 |
| `minimax-m3.1-flash` | `low` | `low` | 通用备选，Edit 延迟较高 |

#### 本地策略评测

S3 为 edit 合规附录加校验重修（最多两次）；T2I appendix 为 T2I 合规附录加校验重修（最多两次）。下表为本地提示词策略实验结果：Edit 使用 16 个用例，T2I 使用 6 个用例；`合规`为规范校验通过数量，`文本召回`为源图文字召回率，`必需文字`为 T2I 请求指定文字的召回数量。不同模型及 reasoning effort 的结果不可直接视为服务端性能保证。

| Edit 策略 / 模型 | Effort | 合规 | 文本召回 | 平均 Subject 数 | p50 |
| --- | --- | --- | --- | --- | --- |
| S3 `qwen3.8-flash-fp8` | `none` | 16/16 | 1.00 | 7.1 | 10.4 s |
| S3 `deepseek-v4-1-flash` | `none` | 15/16 | 1.00 | — | 12.5 s |
| S3 `glm-5.3-flash-w4a8` | `low` | 16/16 | 0.99 | — | 19.3 s |
| S3 `minimax-m3.1-flash` | `low` | 16/16 | 0.98 | — | 80 s |
| S3 `minimax-m3.1-flash` | `minimal` | — | — | — | 67 s |
| S3 `qwen3.8-flash-fp8` | `low` | 16/16 | 0.98 | — | 26.8 s |

| T2I 策略 / 模型 | Effort | 合规 | 必需文字 | 平均输出字符数 | p50 |
| --- | --- | --- | --- | --- | --- |
| T2I appendix `qwen3.8-flash-fp8` | `low` | 6/6 | 6/6 | 1,646 | 17.0 s |
| T2I appendix `deepseek-v4-1-flash` | `none` | 6/6 | 6/6 | 1,377 | 8.5 s |
| T2I appendix `glm-5.3-flash-w4a8` | `low` | 6/6 | 6/6 | 1,125 | 10.8 s |
| T2I appendix `minimax-m3.1-flash` | `low` | 6/6 | 6/6 | 1,578 | 22.5 s |
| T2I appendix `qwen3.8-flash-fp8` | `none` | 6/6 | 6/6 | 1,194 | 9.4 s |

---

## 7. 运维检查清单

1. 启动日志确认：`reference-budget guard sees X GiB resident, Y GiB device, auto=True cap=33554432 px`；
2. 首次请求后看峰值：8 图 1024²≈33.6 GiB/卡、8×4K≈38.5 GiB/卡；
3. 出现 `CUDA OOM` 时检索 `retrying once at` 与 `exceeds the device allowance`；
4. OOM 后若峰值不回落（碎片化），重启副本；
5. 8K（7680×4320）会 400 —— 这是设计（超出官方 4–20 MP 区间）；
6. 参考图超 4:1 / 边长越界 / >64 MiB 会 400 —— 让调用方修输入，不要靠服务端猜。
