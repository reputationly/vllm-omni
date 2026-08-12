# vLLM-Omni · HunyuanImage-3.0 INT8 vs NF4 画质与指令遵从度实测

> 日期：2026-08-11 | 测试人：reputationly + Claude | 节点：`dev-gpustack-a100-0030`（4×A100 PCIE 40G，鲲鹏 ARM）
> 镜像：`reputationly/vllm-omni:arm64-a100-20260809-0612-3f4fe637`（py3.12 / torch 2.11.0+cu130 / **vllm 0.26.0 / bitsandbytes 0.50.0**）
> 产物：`/nfs-models/wuhanjisuan894/hy3_int8v2_ab_20260811/`
> 姊妹文档：`vLLM-Omni-HunyuanImage3-A100-NF4-实验与优化复盘.md`（NF4 上线复盘，速度/显存/拓扑以那份为准）

---

## 0. 结论先行

**决策：现网维持 NF4，INT8 归档不上线。**

三轮共 24 组 i2i 实测，同 seed / 同步数 / 同 yaml / 同输入 / 同驱动，唯一变量是权重目录：

| 维度 | 赢家 | 幅度 |
|---|---|---|
| 细节保持 | **INT8** | 高清组 10/10 全胜，平均 **1.26×**；源越锐差距越大 |
| 长文本准确性 | **NF4** | INT8 把 `FEED` 写成 `EED` 并造出 `FEX`；NF4 29 字符全对 |
| 身份保持 | **NF4** | INT8 在中文横幅组换了张脸；替换类 SSIM 一致偏低 |
| 短文本 / 中文 / 提示词遵从 | 平 | 通用 7 组两边都 7/7 |
| 显存 | **NF4** | 15.83 vs 23.73 GiB/rank；三图余量 **28% vs 8.6%** |
| 速度 | **NF4** | 快 15% |

INT8 的优势（微观纹理 +26%）在人像与文字场景**不体现**，而它的劣势（长文本拼错）属于"图看着没问题但字是错的"这类静默事故。加上 8.6% 的显存余量在生产上没有缓冲，**不划算**。

**但本轮更有长期价值的是两个被推翻的长期误解，见 §4。它们会改变以后所有关于「HY3 出图糊」的判断方向。**

---

## 1. 背景：为什么要重测

2026-07 的 LightX2V POC 曾判定「NF4/INT8 同糊 → 量化无罪」。该结论有两处缺陷：

1. **对照组不干净。** 当年唯一实跑过的 INT8 是 `jamesw767/HunyuanImage-3-Instruct-Distil-INT8`，它的 `llm_int8_skip_modules` 只保 vae/vision/embed/lm_head，把 **attention q/k/v/o_proj、MoE 路由 `mlp.gate`、`shared_mlp` 全量化成 INT8**。而 `mlp.gate` 在架构里是**显式声明为 fp32** 的（`modeling_hunyuan_image_3.py` 中 `nn.Linear(..., dtype=torch.float32)`）——量化路由器会直接改变 top-8 专家选择。
   `EricRollei/HunyuanImage-3.0-Instruct-Distil-INT8-v2` 把这三者全留 bf16，是干净得多的对照组，但**从未上过机**。
2. **路径已过时。** 当年走 HF transformers + eager MoE，现网已是 vLLM-Omni TP4+EP4 融合 MoE，不可比。

两份权重同源（均从 `tencent/HunyuanImage-3.0-Instruct-Distil` 量化），差别只在量化精度与 skip list。

---

## 2. 方法

### 2.1 变量控制

所有对照运行共用：`seed=42`、`num_inference_steps=8`、`guidance_scale=1.0`、同一份 deploy YAML（TP4+EP4、`enforce_eager`、单 DiT 引擎）、同一驱动脚本、同一批输入图与提示词。**唯一变量是 `--model` 指向的权重目录。**

### 2.2 热态计时

`Omni.generate()` 的生成器耗尽时会 `self.close()`（`omni.py::_run_generation_with_generator` 的 `finally`），**一个引擎只能 generate 一次**。因此所有用例在**一次 generate 调用**里批量提交，靠 `max_num_seqs=1` 保证顺序执行，按输出到达时间切分每组耗时与显存窗口。

**第 1 组含首步 JIT，一律不计入热态均值**——这正是姊妹报告 §7.3 对 11.5969 s 提出的同一个警告。

### 2.3 指标

- **LapVar**：Laplacian 二阶导方差 ×1e4，度量**微观纹理**
- **Grad%**：Sobel 梯度幅值均值，度量**边缘强度**
- **SSIM / PSNR**：两张输出之间的结构相似度，用来判断"差异是画得更细，还是画的内容不同"

⚠️ **LapVar 不是画质分。** 它同时被"同内容画得更细"和"画的内容不同"驱动。SSIM < 0.95 的组，LapVar 比值不可直接解读为画质差异（见 §3.2 的文字组反例）。

---

## 3. 实测数据

### 3.1 通用 7 组（环境/属性/局部/风格/背景/文字/多图融合）

| 指标 | NF4-v2 | INT8-v2 |
|---|---:|---:|
| 权重 | 15.83 GiB/rank | 23.73 GiB/rank |
| 峰值（7 组恒定） | **29363 MiB** | **37429 MiB** |
| 40 GiB 卡余量 | 11.3 GiB (28%) | 3.4 GiB (**8.6%**) |
| 热态均值 | **7.12 s** (0.89 s/step) | 8.19 s (1.02 s/step) |
| 三图融合 | 9.26 s | 10.34 s |
| 提示词遵从 | **7/7** | **7/7** |

**峰值差 8066 MiB ≈ 权重差 7.9 GiB**，说明激活开销两边完全相同。峰值在所有用例上恒定，与内容无关——对容量规划是好消息，不会有意外尖峰。

逐组 LapVar 比（INT8/NF4）：换季 1.98× / 裙子改色 1.20× / 加墨镜 1.20× / 水墨 1.17× / 换背景 1.41× / **文字 0.89×** / 三图融合 1.35×。

**但只有 SSIM ≥ 0.98 的三组（裙子 0.998、墨镜 0.997、背景 0.986）是干净对照**，其 LapVar 比为 1.20 / 1.20 / 1.41。文字组（SSIM 0.928）与三图组（SSIM 0.839）构图已明显不同，不可直接比较。

### 3.2 复杂文字编辑 7 组

设计要点：i2i 场景下文字的真难点不是"画出一行字"，而是**替换画面里已有的字**（要先读懂再重写）、**中文**、**长串/多行/数字**。前两组的输入是带 "GOOD DOG" 木牌的图，两个权重用同一张输入。

| 用例 | NF4 | INT8 | 判 |
|---|---|---|---|
| 英文替换 → `BAD CAT` | ✓ 干净，无残留 | ✓ 同 | 平 |
| 中文替换 → `旺财` | ✓ 字形正确 | ✓ 同 | 平 |
| 多行+数字（3 行） | ⚠️ 三行对，下方多乱码行 | ⚠️ 完全相同的毛病 | 平 |
| 中文横幅 `热烈欢迎` | ✓ 字对，**人脸保住** | ✓ 字对，**人脸换了** | **NF4** |
| 长英文（29 字符） | ✓ **全对** | ❌ `NOT EED THE FEX ANIMALS` | **NF4** |
| 霓虹 `OPEN 24H` | ✓ 拼写+辉光+反光 | ✓ 同 | 平 |
| 角标 `SAMPLE` | ✓ | ✓ | 平 |

**5 平 2 负。INT8 在文字场景没有任何优势。**

对输入图的保持度（SSIM）：英文替换 NF4 **0.9450** / INT8 0.9405；中文替换 NF4 **0.9474** / INT8 0.9441。差距小但方向一致，与中文横幅组的人脸漂移是同一现象。

> **反例警示**：通用组的 `text_render` 显示 INT8 LapVar 低 11%，一度被解读为"倒退"。放大看是**两张图画的是不同的牌子**——NF4 的木牌大而粗糙、文字分两行，INT8 的木牌小而光滑、文字排一行。低 11% 来自木牌占比与木纹，与量化精度无关。**这是 LapVar 被内容差异污染的典型案例。**

热态：NF4 **6.53 s** / INT8 7.64 s。

### 3.3 高清素材 10 组（同指令跨题材）

10 张 1024² t2i 素材覆盖：人像特写 / 雪豹毛发 / 英文招牌 / 中文茶馆 / 玻璃建筑 / 毛衣织物 / 东京夜景 / 白瓷低对比 / 镀铬引擎 / 蕨类森林。

统一指令：`Change the lighting to warm golden hour sunlight from the left. Keep the subject, composition and all fine details exactly as they are.`
——保持性指令，唯一变量是内容类型。

| 题材 | 源 LapVar | NF4 | 保持率 | INT8 | 保持率 | INT8/NF4 |
|---|---:|---:|---:|---:|---:|---:|
| 镀铬引擎 | **250.4** | 190.4 | **76.0%** | 254.2 | **101.5%** | 1.34× |
| 蕨类森林 | 67.0 | 60.4 | 90.1% | 70.9 | 105.9% | 1.18× |
| 人像特写 | 52.8 | 43.1 | 81.7% | 44.7 | 84.7% | **1.04×** |
| 东京夜景 | 44.5 | 73.2 | 164% | 98.7 | 222% | 1.35× |
| 雪豹毛发 | 35.6 | 47.4 | 133% | 53.6 | 151% | 1.13× |
| 英文招牌 | 21.7 | 28.9 | 133% | 35.8 | 165% | 1.24× |
| 玻璃建筑 | 17.3 | 45.2 | 262% | 63.4 | 367% | 1.40× |
| 中文茶馆 | 12.0 | 19.9 | 166% | 28.3 | 236% | 1.42× |
| 毛衣织物 | 6.8 | 11.8 | 173% | 15.2 | 224% | 1.29× |
| 白瓷低对比 | 0.9 | 1.2 | 132% | 1.5 | 160% | 1.21× |
| **平均** | 50.9 | 52.1 | **141%** | 66.6 | **182%** | **1.26×** |

SSIM(NF4, INT8) 平均 0.9846 —— 结构几乎一致，差的确实是微观纹理。热态：NF4 **7.28 s** / INT8 8.34 s。

**人像组只有 1.04×** —— 若业务以人像为主，INT8 的 7.9 GiB 买不到东西。

---

## 4. 两个被推翻的长期误解

### 4.1 「i2i 固有丢 80–90% 高频」——不成立

此前的判断来自狐狸换季一例：输入 LapVar 145.74（脸部）→ 输出 10~26，据此归因为"i2i 整图重生成（4096 latent token，每 token 扛 16×16 像素）+ 8 步蒸馏无力重建"，并认定是模型固有缺陷。

**§3.3 的 10 组推翻了它：用保持性指令时，8/10 的输出比输入更锐。**

真实规律是模型有一个**目标锐度**（LapVar ≈ 50）：

```
源 LapVar > 50  →  被磨软（镀铬 250→190、蕨类 67→60、人像 53→43）
源 LapVar < 50  →  被加细节（茶馆 12→20、织物 7→12、白瓷 0.9→1.2）
```

那 80–90% 的损失是**「换季」这条指令的产物**——它把背景改成了大散景，高频自然消失。**不是 i2i 的固有属性，是那次编辑内容的属性。**

这也给了旧报告"真实照片 OK、超锐 AI 图二次编辑显软"一个准确表述：不是分辨率问题，是超锐 AI 图的 LapVar 远高于模型目标锐度，必然被拉低。

**实践含义**：抱怨"HY3 出图糊"时，先看是不是编辑指令本身改掉了高频内容，再怀疑模型。

### 4.2 「高清 i2i」在 HY3 上不存在——1 Mpx 硬天花板

`get_cached_resolution_group(base_size=1024)` 只产出 **37 个分辨率桶，全部落在 0.786–1.049 Mpx**：

```
最大 1024×1024 = 1.049 Mpx（及 2048×512 等极端比例，同为 1.05 Mpx）
最小  768×1024 = 0.786 Mpx
```

实测输入→输出映射：

| 输入 | 输出桶 | 损失 |
|---|---|---|
| 928×1664 (1.54 Mpx) | 720×1280 (0.92 Mpx) | 降到 60% |
| 2048×2048 (4.19 Mpx) | 1024×1024 (1.05 Mpx) | **降到 1/4** |
| 3840×2160 (8.29 Mpx) | 1280×720 (0.92 Mpx) | **降到 1/9** |

**不管喂多大的图，输出永远约 1 Mpx。** 这解释了为何 `multi/man.png`（928×1664）传原尺寸进去、出来是 720×1280——不是参数传错，是管线强制归桶。

**实践含义**：任何"用 HY3 做高清图精修"的需求都不成立，需要外接超分。

---

## 5. 对旧报告「量化无罪」的修订

原文：*NF4/INT8 同糊 → 量化无罪。*

**拆成两句才准确：**

1. **「糊的主因不是量化」——成立。** 权重差异带来的是 ±20~40% 的微观纹理，而指令内容能带来数倍的高频变化（§4.1）。
2. **「NF4 与 INT8 画质相当」——不成立。** 当年这句是拿 jamesw767 得出的，而 jamesw767 恰好和 NF4 一样软（全图 LapVar 3.53 vs 3.35），**掩盖了 EricRollei v2 的差异**。干净对照下 INT8 细节高 26%，但长文本会拼错、身份保持更差。

---

## 6. 工程侧发现

### 6.1 上游 vLLM 的 bnb-int8 × MoE 是空桩，且是刻意的

`vllm/model_executor/layers/quantization/bitsandbytes.py`（v0.26.0）中
`BitsAndBytesMoEMethod._create_weights_8bit` 与 `._apply_8bit_dequant` 均为 `raise NotImplementedError`。
注意 **bnb 8-bit 对普通 Linear 是完整实现的**（`BitsAndBytesLinearMethod._apply_8bit_weight`，`MatmulLtState` + CB/SCB），缺的精确是 **bitsandbytes × int8 × MoE** 这一格。

留空是合理的：bnb int8 的卖点是 `MatmulLtState` 的离群值分解矩阵乘，而它是**逐 Linear 有状态**的，塞不进 16 个专家的 grouped GEMM。唯一可实现的形态是"反量化成 bf16 再喂 `fused_experts`"——比 compressed-tensors W8A8 慢、比 NF4 大，两头不讨好。

**HY3 撞上这一格是因为它没得选**：官方无量化版，社区量化仓（EricRollei / jamesw767）全是 bnb。
对照：MiniMax-H3 的 INT8 权重 `quant_method` 是 `"int8"`（vllm-omni 自己的 W8A8/W8A16），且 H3 是稠密 DiT 无专家——两个轴上都不在这一格，所以从没撞到。

### 6.2 compressed-tensors 加载 bug（**已修，与 INT8 无关**）

`vllm_omni/quantization/factory.py::_build_single()` 两处缺陷：

1. **名字归一化**：`_normalize_method_name()` 把 `compressed-tensors` 转成 `compressed_tensors`，然后拿去查**未归一化**的 vLLM 注册表 → 必然 miss。报错还特别误导：说"未知方法 `compressed_tensors`"，紧接着列出的支持列表里明明有 `compressed-tensors`。**任何 compressed-tensors 权重都会撞，与 HY3 无关。**
2. **构造方式**：注册表分支直接 `config_cls(**kwargs)`，而 `CompressedTensorsConfig.__init__` 要的是解析后的 `target_scheme_map`，不是 checkpoint 原始字段。函数 docstring 本就承诺 "via `from_config()`"，代码没做到。

修复保留在工作区。影响面已确认安全：`bitsandbytes` 在 `_OVERRIDES` 中，`_build_single` 提前 return，**现网 NF4 路径走不到这段代码**。

### 6.3 W8A8 转换：可行但不划算

bnb int8 与 compressed-tensors `int-quantized` W8A8（`strategy: channel`）是**同一种表示**——bnb 的 `SCB` 就是逐输出行 absmax，`CB = round(W×127/absmax)`。转换是恒等改写：`weight_scale = SCB/127`、去掉 `weight_format`、重写 `quantization_config`。**无需 BF16、无需校准**，83 GB 流式改写几十分钟（+4096 scale / −4096 marker，数字精确对上 32 层 × 64 专家 × 2 投影）。

实测（A100 上选到 TRITON 后端）：

| | bnb INT8 | W8A8 |
|---|---:|---:|
| 脸部 LapVar | 25.81 | 25.30 |
| 单图 it/s | 1.07–1.28 | 1.18 |
| 三图峰值 | 37429 MiB | **40429 MiB**（余 **1.3%**） |

**三条发现，两条与预期相反：**

1. **画质：激活量化几乎零代价。** 与 bnb INT8 的 PSNR 34.49 dB。此前引用的"Bernini v2v 雪花结案：编辑类任务禁 int8"风险**在本任务上没有应验**。
2. **速度：没提上去。** 仅比 bnb INT8 快 9%，仍慢于 NF4。说明 MoE GEMM 不是瓶颈，或 Triton kernel 未针对本 shape 调优（姊妹报告 §13 P1 待办：`E=16/N=3072/K=4096`，现吃默认配置）。
3. **显存：反而更费 3 GiB**（Triton workspace + 激活的 int8 副本），余量掉到 1.3%，不可用于生产。

---

## 7. 结论与建议

**维持 NF4。** INT8 的收益（微观纹理 +26%）局限于细节向题材，代价（+50% 显存、慢 15%、余量 28%→8.6%、长文本拼错）确定且不可接受。

**下一步的性价比排序**（均未做）：

1. **NF4 加步数 8 → 16/20** —— 不花一分显存、不动拓扑，直接打"目标锐度"这个主因。**若 16 步 NF4 超过 8 步 INT8 的 1.26×，整条 int8 路线可彻底结案。**
2. **MoE Triton tuning JSON**（姊妹报告 §13 P1）—— NF4/INT8 两条路都受益，不动数值语义
3. **VAE tiling/slicing** —— 现为 false，1024² 的 VAE decode 可能是峰值贡献者，未验证
4. **TeaCache** —— `hunyuan_image3_transformer.py` 有完整实现，但 DiT yaml 里 `cache_backend`/`cache_config` 为空 = 未启用。8 步冗余少、风险高，但是唯一能成倍降耗时的杠杆

**若未来仍需 INT8**：用 EricRollei v2，**不要**用 jamesw767（它把 fp32 路由器也量化了）。

---

## 8. 复现与产物

### 8.1 产物清单

```
/nfs-models/wuhanjisuan894/hy3_int8v2_ab_20260811/
├── hd_src/            10 张 1024² 素材
├── bhd_int8/ bhd_nf4/ 高清 10 组输出 + report.md + results.json
├── bhd_cmp/           ⭐ 10 张三联图（SOURCE | NF4 | INT8）
├── btext_int8/ btext_nf4/ btext_cmp/   文字专项 7 组
├── bench_int8/ bench_nf4/ cmp_pairs/   通用 7 组
├── fox_face_4way.png  脸部 1:1：输入/NF4/INT8/W8A8
├── text_cmp.png       文字组木牌差异（LapVar 污染案例）
└── fuse3img_*.png / fox_summer_*.png / peak_*.txt
```

权重：`…-NF4-v2`(48G) / `…-INT8-v2`(83G) / `…-W8A8`(83G) 于 `/nfs-models/wuhanjisuan894/models/`；
BF16 原版 `…-Instruct-Distil-BF16`(158G) 于 `/nfs-data/models/`（**本集群跑不动，仅作量化源与未量化基准**）。

### 8.2 脚本

`vllm-omni/scripts/`（**该目录被 `.gitignore:184` 忽略，不进版本库**）：

- `bench_hunyuan_image3_i2i.py` —— 单引擎顺序 i2i 压测，通用，与 INT8 无关。支持 `--case-set default|text` 与 `--src-dir` + `--uniform-prompt`（目录驱动，同指令跨题材）
- `convert_hunyuan_image3_bnb_int8_to_w8a8.py` —— bnb int8 → compressed-tensors W8A8 格式转换
- `download_hunyuan_image3_{int8,bf16}.sh` —— 权重下载（ModelScope/hf-mirror 多轮重试 + 完整性预检）

### 8.3 INT8 支持代码（**已从工作区回退**）

决定不上线后已 `git checkout` 还原，未提交。远端沙箱 `dev-gpustack-a100-0030:/root/hy3int8/vllm-omni/` 仍有完整实现。若需重做，改动面为三处：

1. **`vllm_omni/patch.py`** 新增 `_patch_bnb_moe_int8()`，填上游两个空桩。
   `_create_weights_8bit`：int8 **不打包**，参数保持逻辑形状 `w13 [E,2I,H]` / `w2 [E,H,I]`；
   **绝不能设 `use_bitsandbytes_4bit`** —— 该标志会把 `RoutedExperts.weight_loader`（`routed_experts.py:648`）切到扁平 packed 分支。不设即走标准 w1/w2/w3 路径，无需任何 bnb 专用加载代码。
   `_apply_8bit_dequant`：`CB × (SCB/127)` 直出 bf16。**不要**用 `int8_vectorwise_dequant`——它返回 FP32，会把每层临时缓冲从 0.8 GiB 翻到 1.6 GiB；折叠除法后只需一趟全张量运算。两者相对差 1 FP32 ULP，还原原始权重的误差**完全相同**。
2. **`hunyuan_image3_transformer.py`**：`_BNB_EXPERT_STATE_RE` 加 int8 分支——int8 的 state 是 `.weight` 的**兄弟键**（`…gate_and_up_proj.SCB` / `.weight_format`），不是 NF4 那种后缀（`…weight.absmax`），需用两个命名组保住 NF4 的 `base_name` 语义；`_reorder_hunyuan_gate_up_absmax` 泛化为 scales 共用（`gate_and_up_proj` 按 `[up, gate]` 存，vLLM w13 要 `[gate, up]`，absmax 与 SCB 同样需半区对调）；新增 `_bind_local_bnb_expert_states_int8()`，把 16 个 local expert 的 SCB `torch.stack` 成 `[E, rows]` 绑到 `bnb_scb`。
3. **deploy YAML**：单 DiT 引擎，`enable_sleep_mode: false`，其余与 NF4 版逐项相同（**故意如此**，保证输出差异只归因于权重）。

checkpoint 事实（已坐实）：只有 `mlp.experts.N.*` 被量化，`self_attn.{qkv,o}_proj`、`mlp.gate.wg`、`mlp.shared_mlp.*` 全是裸 bf16。`gate_and_up_proj.weight` int8 `(6144,4096)` + `SCB` fp32 `(6144,)`；`down_proj` `(4096,3072)` + `(4096,)`；`weight_format` 是标量 uint8 标记。单个专家的张量**可能跨分片**（layer1 expert30 的 gate_up 在 shard2、down 在 shard3），收集逻辑不可假设 per-expert 分片局部性。

---

## 9. 未做 / 存疑

- **没有 BF16 基准。** 157 GiB 权重 vs 四卡 160 GB 总显存，跑不动。所有画质判断都是量化版之间的相对比较。要拿绝对基准只能借 80G 卡或走官方 API。
- **步数实验未做**（§7 第 1 条），这是最该补的一块。
- **W8A8 未做 kernel tuning**，其"速度没提上去"的结论可能只对默认 Triton 配置成立。
- **单 seed。** 全部用 seed 42，未做多 seed 稳定性验证。文字类结论（尤其长文本失败）建议换 2~3 个 seed 复核后再当定论。
- **INT8 峰值 37429 MiB 是 1024² / ≤3 参考图条件下的。** 更大底图或第 4 张参考图未压测，8.6% 余量下大概率 OOM。
