# Ideogram 4 接入调研与路线选型报告

> 调研日期:2026-08-15
> 目标:在现网 GPUStack + new-api 栈上、40GB A100(ARM64 主机)把 Ideogram 4 跑起来
> 结论:**主路线走 vllm-omni 移植**,从上游 PR #4227 起步;官方 harness 只做 P0 数值基准,不做生产引擎
> 状态:纯调研,未改动任何代码

---

## 0. 结论摘要(TL;DR)

| 问题 | 结论 |
|---|---|
| 上游 vllm-omni 适配了吗 | **没有**。两个 PR(#4227 / #4788)都停在 2026-07-17,未合并 |
| `/Users/.../api/ideogram4` 是官方 harness 吗 | **是**。`github.com/ideogram-oss/ideogram4`,HEAD `990fe1c`(2026-06-30),3173 LOC / 13 个 py 文件 |
| 40G A100 能跑吗 | **能**。nf4 权重 16.12GB,1024²/2048² 都能跑;fp8(27.4GB)、int8 w8a8(29.2GB)只能勉强 1024²;bf16(53.59GB)不可能 |
| fp8 在 A100 上有加速吗 | **没有**。官方 fp8 是 dequant-to-bf16 + `F.linear`,不走 `_scaled_mm`,所以能在 A100 跑但**只省显存不提速** |
| 权重从哪拿 | **ModelScope**。HF 的 `ideogram-ai/*` 是 gated(401),ModelScope 镜像 `ApprovalMode=0` 直接可下 |
| 最省显存的生产档 | nf4 + `ostris/ideogram_4_unconditional_lora`(~11.0GB,单 DiT,`enable_cfg: false`) |
| 三条路线选哪条 | **vllm-omni**。harness 直接内嵌要重造整条 P1-P5 引擎链;LightX2V 与已定决议冲突,且其"核心路径不跑 transformers"铁律被 13 层激活抽取直接否掉 |
| **能不能拿去卖钱** | **不能,除非另拿授权**。权重是 Ideogram Non-Commercial 协议,§2 要求商用须由 Ideogram **另签协议且其全权决定是否给**;托管 API 对外服务同时踩中"商用"与"分发"两条。**详见 §5** |

---

## 1. 上游适配现状

搜索 vllm-project/vllm-omni 的 issue/PR(含已关闭)结果:

| PR | 内容 | 规模 | 状态 |
|---|---|---|---|
| **#4227** | Ideogram 4 主体适配(pipeline + transformer + registry) | 15 files,+2056 | `CHANGES_REQUESTED` —— review 意见是**缺 e2e 出图证据**,不是架构问题 |
| **#4788** | fp8 权重 loader | 5 files,+1347 | 停滞 |

两个 PR 均自 **2026-07-17** 起无更新,主干 `upstream/main`(`593b4045`,2026-08-09)无任何 Ideogram 相关代码。

**本仓状态**:本地 `main` 领先 `upstream/main` 46 commits / 122 files / +19010 −191。

> **可行起点**:从 #4227 分支起,把 #4788 的 fp8 loader 并进来。不要从 harness 白手起家。
> **未决**:#4227 分支的实际代码质量尚未评估 —— 这是整个选型里唯一还没落地的变量。

---

## 2. 模型架构

### 2.1 规格

| 字段 | 值 |
|---|---|
| `emb_dim` | 4608 |
| `num_layers` | 34 |
| `num_heads` | 18(head_dim = 256) |
| `intermediate` | 12288 |
| `adanln_dim` | 512 |
| `in_channels` | 128(= AE 32 通道 × patch 2²) |
| `rope_theta` | 5,000,000 |
| `mrope_section` | (24, 20, 20) |
| `norm_eps` | 1e-5 |
| `max_text_tokens` | 2048 |
| 采样 | Euler flow-matching,logit-normal schedule,**asymmetric CFG** |

单流 DiT:文本 token(Qwen3-VL 隐藏态)与图像 latent token 拼成一条序列,逐 block 由 timestep embedding 算出的 AdaLN 调制;attention 用 QK-RMSNorm + 3D MRoPE,文本与图像共享统一位置空间。

参数量按 config 手算 **9.28B**,与 README 的 "9.3B" 吻合。

### 2.2 三个非常规设计(移植的全部难点都在这里)

#### (a) 双 Transformer CFG —— 两份独立训练的 DiT

`pipeline_ideogram4.py:300-306`:

```python
conditional_transformer = _build_transformer(transformer_config, conditional_state_dict, device, dtype)
del conditional_state_dict
unconditional_transformer = _build_transformer(transformer_config, unconditional_state_dict, device, dtype)
```

同一份 config 建两次,但加载**两份不同权重**(`transformer/` 与 `unconditional_transformer/`)。所以常驻参数是 **18.6B**,不是 9.3B。这也是所有显存账的起点。

#### (b) 非对称 CFG —— 负分支不吃文本

负分支只喂图像 token(不带文本 padding),`neg_llm_features` 置零;合并公式(`pipeline_ideogram4.py:613`):

```python
v = gw_i * pos_v + (1.0 - gw_i) * neg_v
```

注意 `gw` 是**逐步变化**的(见 §2.3),不是常数。

#### (c) 13 层激活抽取 —— 硬依赖 transformers 内部件

`constants.py`(全文 11 行):

```python
SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3
IMAGE_POSITION_OFFSET = 65536
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
```

`_get_qwen3_vl_embeddings`(`pipeline_ideogram4.py:414-451`)不是"跑一个 text encoder 取最后一层",而是**手写展开 transformers 的 decoder loop**逐层 hook:

```python
language_model = self.text_encoder.language_model
inputs_embeds = language_model.embed_tokens(token_ids)
position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
causal_mask = create_causal_mask(config=language_model.config, inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask, past_key_values=None,
                                 position_ids=text_position_ids)
position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)
for layer_idx, decoder_layer in enumerate(language_model.layers):
    hidden_states = decoder_layer(...)
    if layer_idx in tap_set: captured[layer_idx] = hidden_states
```

13 层 × 4096 拼接 → `llm_features_dim = 53248`,且 `pipeline_ideogram4.py:480` 明确保留 **fp32**:

```python
return stacked.to(torch.float32)
```

**这一条是三路线选型的决定性因素**(见 §7)。

### 2.3 采样调度

`scheduler.py`(70 行):`LogitNormalSchedule` / `get_schedule_for_resolution` / `make_step_intervals`。
分辨率相关的 mu 漂移:`mu += 0.5 · log(pixels / 512²)`。

`sampler_configs.py` 预设:

| 预设 | 步数 | guidance schedule |
|---|---|---|
| `V4_QUALITY_48` | 48 | gw=3 前 3 步,gw=7 后 45 步;mu=0.0,std=1.5 |
| `V4_DEFAULT_20` | 20 | — |
| `V4_TURBO_12` | 12 | — |

### 2.4 VAE

`autoencoder.py` 文件头写着 `"""Flux2 KL autoencoder."""`,`AutoEncoderParams(ch=128, ch_mult=[1,2,4,4], z_channels=32)`。
→ **vllm-omni 已有 `flux2`,可直接复用,这块成本为零。**

### 2.5 patch 数学

`patch = patch_size(2) × ae_scale_factor(8) = 16`
- 1024² → **4096** image token
- 2048² → **16384** image token

---

## 3. 官方 harness 剖析

仓库:`github.com/ideogram-oss/ideogram4`,本地在 `/Users/reputationly/Desktop/code/api/ideogram4`

| 文件 | 行数 | 作用 |
|---|---|---|
| `pipeline_ideogram4.py` | 637 | 全部编排逻辑,最重要 |
| `modeling_ideogram4.py` | 379 | DiT 本体 |
| `quantized_loading.py` | 278 | bnb-4bit + 自定义 weight-only fp8 |
| `scheduler.py` | 70 | flow-matching schedule |
| `sampler_configs.py` | — | 三个预设 |
| `autoencoder.py` | — | Flux2 KL AE |
| `safety.py` / `magic_prompt.py` | — | **外网调用,生产必须处理** |
| `constants.py` | 11 | 见 §2.2 |

### 3.1 必须替换的稠密 attn_mask

`modeling_ideogram4.py:154-156`:

```python
attn_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)
out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
```

2048² 下这是 **16384 × 16384 × 18 heads** 的 bool 张量。**任何路线都必须换成 varlen**,所以"harness 零改动直接上生产"本来就不成立。

### 3.2 两颗外网炸弹

- `safety.py` → Hive moderation API
- `magic_prompt.py` → `https://api.ideogram.ai/v1/ideogram-v4/magic-prompt` + OpenRouter,system prompt 在 `magic_prompt_system_prompts/v1.txt`
- `_verify_prompts`(`pipeline_ideogram4.py:482-501`)命中即 `raise ValueError`

生产内网必须禁用或改本地实现。**好消息**:官方提供了在设备上跑的 `Ideogram4PromptEnhancerHead`(`diffusers/qwen3-vl-8b-instruct-lm-head`,`prompt_upsampling=True`),不必自建门面层。

### 3.3 一处明显浪费

`pipeline_ideogram4.py:551` —— LLM 在**整条 packed 序列**(含空图像槽位)上跑。移植时可优化。

---

## 4. 权重版本全景与 40G A100 显存账

### 4.1 官方 Model Zoo(`README.md:48-49`)

| 版本 | 设备 | Diffusers 支持 | 许可 |
|---|---|---|---|
| nf4 | CUDA | **Yes** | Ideogram 4 Non-Commercial |
| fp8 | All | **No** | Ideogram 4 Non-Commercial |

> ⚠️ **许可是 Ideogram 4 Non-Commercial** —— 这是本项目最硬的阻塞项,详见 **§5**。

### 4.2 显存账(HF/ModelScope API 实测权重体积)

| 路线 | 权重 | 1024² | 2048² |
|---|---|---|---|
| nf4 + `ostris/ideogram_4_turbotime_lora` | ~11.7 GB | ✅ | ✅ |
| nf4 + `ostris/ideogram_4_unconditional_lora` | ~11.0 GB | ✅ | ✅ |
| **官方 nf4**(te 5.48 / 2× dit 5.22 / vae 0.17) | **16.12 GB** | ✅ | ✅ |
| `fal/ideogram-v4-fast` / `-instant` + nf4 encoder | ~24.2 GB | ✅ | 紧 |
| `transformerlab/ideogram-4-int8-w8a8` | 29.2 GB | ✅ 紧 | ❌ |
| `ideogram-ai/ideogram-4-fp8` | ~27.4 GB | ✅ 紧 | ❌ |
| bf16 | 53.59 GB | ❌ | ❌ |

> 表中为**权重文件体积**,不含激活/中间张量。2048² 下 16384 token 的激活开销另计。

### 4.3 关键仓库清单(HF API 实测存在)

```
ideogram-ai/ideogram-4-nf4              # 官方 nf4(gated)
ideogram-ai/ideogram-4-nf4-diffusers    # 官方 nf4 diffusers 版(gated)
ideogram-ai/ideogram-4-fp8              # 官方 fp8(gated)
ostris/ideogram_4_turbotime_lora        # 少步蒸馏 LoRA
ostris/ideogram_4_unconditional_lora    # 去 uncond 分支 LoRA(单 DiT)
fal/ideogram-v4-fast                    # fal 蒸馏
fal/ideogram-v4-instant                 # fal 蒸馏
transformerlab/ideogram-4-int8-w8a8     # 第三方 int8 W8A8
```

---

## 5. 许可与合规(商用阻塞项)

> 本节是对许可原文的阅读记录,**不是法律意见**,最终以法务判断为准。
> 原文位置:`/Users/reputationly/Desktop/code/api/ideogram4/model_licenses/LICENSE-IDEOGRAM-4-NON-COMMERCIAL`
> 协议名:**Ideogram Non-Commercial Model Agreement**,Last Updated **2026-06-03**
>
> ⚠️ **本节内出现的 §x 除注明"本报告"外,一律指协议条款号,不是本报告章节号。**

### 5.1 两份许可要分开看

| 对象 | 许可 | 商用 |
|---|---|---|
| harness **代码**(`ideogram4` 仓,`LICENSE.md`) | **Apache 2.0** | ✅ 可以 |
| **模型权重** | **Ideogram Non-Commercial Model Agreement** | ❌ 不可以 |

代码可自由使用,**卡点全部在权重上**。

### 5.2 核心结论:对外卖钱必须先拿单独授权

协议 §2 原文:

> you are only authorized to exercise the rights under this Agreement for **Non-Commercial Purposes only**, and may not exercise any of the rights under this Agreement for other purposes **unless or until Company otherwise expressly grants you such rights in a separate agreement, which Company may grant or not grant in its sole discretion.**

即:商用权需要 Ideogram, Inc. **另行签署一份协议**,且**是否授予由其全权决定**。
README 与许可文件中**均未留商务联系方式**,目前已知入口只有 HF 仓库的 gate 页面。

### 5.3 为什么本项目场景必然落在"商用"一侧

**协议 §1(d) 定义的 Non-Commercial 四类,本项目一条都不占**:

| 类别 | 内容 | 是否适用 |
|---|---|---|
| (i) | 不直接或间接产生收入,且非为商业利益或金钱报酬 | ❌ 对外计费 |
| (ii) | 营利实体**仅**用于测试/评估/研发,且在**非生产环境**(明确排除 live systems、customer-facing applications) | ⚠️ 仅 P0–P3 适用 |
| (iii) | 个人研究/实验/爱好 | ❌ |
| (iv) | 慈善组织的慈善用途 | ❌ |

且 §1(d) 末尾明写:

> any use that involves training, fine tuning, or distilling AI models for commercial use or that involves **generating Output to include in, or to advertise or promote, revenue-generating products or services**, in each case, is not a Non-Commercial Purpose.

**协议 §1(a) 的 "Distribution" 明确包含托管 API**:

> including by providing or making the Model or its functionality available **as a hosted service via API, web access or any other electronic or remote means ("Hosted Service")**

本项目形态是 GPUStack 起实例 + new-api 网关对外发 API —— 属于教科书式的 Hosted Service,**同时踩中"商用"与"分发"两条**。

### 5.4 三个常见误读(都救不了场)

1. **"输出归我们" ≠ "能卖"**
   §7 确实写了 *"We claim no rights in outputs you generate using the Model."*,但 §1(d) 已把"生成 Output 用于营收产品"排除在非商用之外。**产出物归属与使用权限是两件事**。
   §7 另有限制:不得用 Output 训练/微调/蒸馏与 Ideogram 竞争的模型或产品。

2. **换第三方版本绕不开**
   §1(c) 的 "Model Derivative" 包含微调版、**以及通过迁移权重/参数得到的任何模型**。因此 `ostris/*_lora`、`fal/ideogram-v4-fast|instant`、`transformerlab/ideogram-4-int8-w8a8`、各类 GGUF/量化版 **全部是 Model Derivative**;§3(i) 又要求再分发条款 "no less restrictive"。**整棵衍生树都是 non-commercial。**

3. **从 ModelScope 下载 ≠ 绕过许可**
   HF 的 gate 只是**接受协议的触发动作**,不是许可本身。镜像开放是镜像方的合规问题,权重上的义务照样约束使用者。本报告 §8 推荐走 ModelScope 是为了解决**可达性**,不解决**授权**。

### 5.5 两条与现有工程计划直接冲突的条款

| 条款 | 内容 | 与计划的冲突 |
|---|---|---|
| **§4** | 不得 *"circumvent, remove, alter, deactivate, degrade or thwart"* 公司实施的 content filters / watermarking | §3.2 计划"内网禁用 `safety.py`"。稳妥做法是**替换为等效的本地审核**,而非直接删除这道门 |
| **§9** | Ideogram 可**随时通知终止**;终止后须**删除权重并停止使用**;§5–§10 条款在终止后继续有效 | 即使拿到商用授权,这也是必须登记的**业务连续性风险** |

其他需要法务过目的条款:

- **§3(iii)**:再分发须随附协议副本 + `Notice` 文件中保留指定的 attribution 文本;§3(iv) 修改过的文件须显著标注
- **§4**:并入 `https://ideogram.ai/legal/usage-policy`;禁止军事/监控/生物特征处理/高风险自动化决策等;**须按当地法律对 AI 生成内容作出标识**(境内另有生成合成内容标识要求)
- **§8**:使用方须对 Ideogram 作赔偿担保(indemnification)
- **§10**:适用**纽约州法**;Ideogram 可单方修改协议,继续使用即视为接受

### 5.6 分界线在 P4,不在 P0

| 阶段 | 能否进行 | 依据 |
|---|---|---|
| P0 harness 跑通、稳态矩阵、出图对拍 | ✅ | §1(d)(ii) 非生产环境的测试/评估/研发 |
| P0.5 移植进 vllm-omni、内部验证 | ✅ | 同上,只要不上线 |
| P1–P3 异步化 / 镜像 / GPUStack 内嵌(内部) | ✅ | 同上 |
| **P4 上生产、接 new-api 对外计费** | ❌ | §1(a) Hosted Service + §1(d) |

**技术调研与移植可照常推进,但 P4 之前必须拿到书面商用授权**,否则这套东西只能作内部工具。

### 5.7 建议动作

1. 将 `LICENSE-IDEOGRAM-4-NON-COMMERCIAL` 原文交法务,重点标 **§1(a)、§1(d)、§2、§4、§7、§9**。
2. 并行两条线:商务/法务向 Ideogram 询商用授权(已知入口仅 HF gate 页面);工程侧继续 P0/P0.5 —— 即便授权拿不到,varlen attention、双 DiT pipeline、13 层 tap 等工程件对接其他模型仍可复用。
3. **同时准备 B 方案**:若目标是"可商用的文生图能力",现在就应并行评估许可宽松(Apache/MIT 或明确允许商用)的替代模型。**该决策不应拖到 P4。**

---

## 6. 量化方案分析

### 6.1 官方 fp8:能在 A100 跑,但零加速

自定义 weight-only fp8:e4m3 + per-row fp32 scale,存在 `.weight_scale` buffer。
`quantized_loading.py:197-200` 的 `Fp8Linear.forward` 是决定性证据:

```python
w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
bias = self.bias.to(x.dtype) if self.bias is not None else None
return F.linear(x, w, bias)
```

**dequant 回 bf16 再做普通 matmul**,不碰 `_scaled_mm`,不需要 fp8 tensor core。
→ A100(Ampere,无 fp8 TC)完全能跑,**但一点也不会更快**。代码注释自己也写了:"The win is ~2x smaller Linear weights."

其他常量:`FP8_SCALE_SUFFIX = ".weight_scale"`,`FP8_TEXT_ENCODER_CONFIG_FLAG = "ideogram_fp8_weight_only"`。

### 6.2 第三方 int8 W8A8

`transformerlab/ideogram-4-int8-w8a8` 方案:

- 权重 per-channel + 激活 per-token 动态量化
- SmoothQuant α=0.5
- top-17 个 FFN `w2` 层保留 bf16(保护层)

**优点**:这套布局能映射到 vllm 原生的 `CompressedTensorsW8A8Int8`;且 A100 **有** int8 tensor core,理论上是唯一可能带来加速的量化路线。

**已识别的阻塞 bug** —— 该仓 `usage.py` 的加载顺序错了:

```
Ideogram4Pipeline.from_pretrained(...)   # 先把完整 fp8 栈 27.4GB 装进显存
load_int8(...)                            # 再叠加 20.4GB int8 权重
→ 峰值 ~48GB → 40G 卡在加载阶段就 OOM
```

修法:meta device 建骨架 + 逐层替换,不要先 materialize fp8。

**未决**:该仓 `safetensors_loader.py` / `recipe.json` 的张量布局尚未与 vllm 的 `CompressedTensorsW8A8Int8` 逐字段对齐核对。

### 6.3 与 A100 config 铁律的关系

按 `新引擎内嵌gpustack-工程化方法论.md` §2.4:

- `attn_type=torch_sdpa`(flash_attn3 是 Hopper 专属;sage_attn2 在 Qwen 上出 NaN)
- `rope_type=torch`
- 多卡不开 `cpu_offload`
- 蒸馏模型 `enable_cfg: false` ← **正对应 unconditional LoRA 那一档**
- **"int8 的价值是显存不是速度"** ← 与 §6.1 的实测结论完全一致

---

## 7. 三条路线选型分析

### 7.1 成本结构不对称的根因

方法论的 P0→P5 里,**P1 异步化 / P2 镜像 / P3 GPUStack 内嵌 / P3.5 new-api / P4 部署**是"每条引擎链一次性成本";**P0.5 引擎接模型**才是随模型变的成本。

关键事实:**vllm-omni 已经是 GPUStack 的内嵌 backend 并已在现网运行**。于是:

| | P0.5 接模型 | P1 异步化 | P2 arm64 镜像 | P3 GPUStack 6 处改动 | P3.5 new-api | 长期维护 |
|---|---|---|---|---|---|---|
| **A. harness 直接内嵌** | **0** | 全新 ~500 行 | 全新 Dockerfile + launcher + CI | 全套 | 新渠道 + 物化 | **多一条独立引擎链** |
| **B. 移植 vllm-omni** | 中 | 0 | 增量(重出镜像) | 0 | 加 model ID + profile | 与现有链合流 |
| **C. 移植 LightX2V** | **最高** | 0 | 增量 | 0 | 加 model ID | 与现有链合流 |

方法论 §1 的实测基线是"P3 第一个引擎 M1-M5 = 2 天,第二个引擎(IndexTTS)~10 行"。
路线 A 是在**再造一次第一个引擎**;B/C 吃的是"第二个引擎 ~10 行"的红利。

### 7.2 C. LightX2V —— 排除(两个独立理由)

1. **与 2026-08-09 决议冲突**:路线已定稿 vllm-omni,不切 LightX2V。
2. **它自己的规矩就否了这个模型**:`support_new_model` 铁律是"核心路径不跑 diffusers/transformers,权重离线转 x2v"。而 §2.2(c) 的 13 层激活抽取硬依赖 `create_causal_mask` / `rotary_emb` / 4D mrope position_ids 这些 transformers 内部件。要在 LightX2V 里做,等于用 LightX2V 算子重写 Qwen3-VL-8B 并保证 13 层中间激活逐层数值对齐(§13 要求"与上游 pipeline 数值对齐 7 步验证")。这是全工作量里最贵、收益为零的一块。

### 7.3 A. 官方 harness 包成独立引擎

**真优点**:P0.5 归零;双 DiT、13 层 tap、非对称 CFG、logit-normal schedule、逐步 gw 全是官方原版,**不承担数值走样风险**;nf4 16.12GB 现在就能在 40G 上跑。

**被低估的代价**:

- **重造整条链**:P1(5 端点 + `/ready` + FIFO + 状态字符串逐字核对含 `cancelled` 双 L + `.part{ext}` 原子写)、P2(ARM64 base + torch 约束冻结 + 训练依赖剔除 + CI arm 原生 runner + 双 tag)、P3(6 处源码改动 + 门面 4 处 + janitor 保护键 + model-catalog + UI 4 处枚举 + `Dockerfile.acr` COPY 清单)、P3.5(渠道 + `materializeXXXInputs` + `legacyInputKeys`)。方法论说的"2 天"是**踩完所有坑之后**的复现成本,不含 P2 的 ARM 依赖排雷。
- **harness 是 research 代码不是服务**:3173 行,单请求独占、无批处理、无并发、无显存复用;且 §3.1 的稠密 mask 本来就必须改。
- **两颗外网炸弹**(§3.2)。
- **ARM64 + bitsandbytes**:nf4 依赖 bnb,aarch64+CUDA wheel 供应本身就是方法论 §12 那类坑(A/B 都要面对,但 A 是在全新镜像上第一次踩)。

### 7.4 B. 移植 vllm-omni —— 推荐

P0.5 逐项拆解:

| 项 | 难度 | 说明 |
|---|---|---|
| VAE | **零** | Flux2 KL AE,vllm-omni 已有 `flux2`,直接复用 |
| text encoder 13 层 tap | **低** | vllm-omni 允许 text encoder 走 transformers,`_get_qwen3_vl_embeddings` 近乎原样搬。**这正是 C 路线做不到而 B 免费拿到的** |
| DiT 本体 | 中 | 34 层单流,结构最接近 `z_image`(pipeline 731 + transformer 1063 行),照抄骨架 |
| attention | 中 | 稠密 mask → varlen。这是**收益**不是成本;A100 按铁律固定 `torch_sdpa` + `rope_type=torch` |
| 双 DiT | 中 | `_DIFFUSION_MODELS` 是 arch→单个 (folder, module, class),但**不需要改 registry** —— 注册 pipeline,pipeline 内部持两个 transformer 实例即可 |
| 量化 | 中 | `vllm_omni/quantization/` 已有 `bitsandbytes_config` / `int8_config`;`Fp8Linear` 是纯 dequant + `F.linear`,移植零障碍 |
| 起点 | — | PR #4227 + #4788 |

**顺带拿到**:CFG 并行(双 DiT 天然可拆两卡)、`enable_cfg: false` 单 DiT 档(~11.0GB)、Cache-DiT、varlen;以及方法论第 185 行点名的"vLLM-Omni 是受理即起协程形态,必须显式区分排队与执行"—— 这坑在现有内嵌里**已踩过并修好**,新模型接进去是零成本。

**主风险:数值对齐**。非对称 CFG(uncond 只吃 image token)、逐步 gw schedule(`V4_QUALITY_48` = 3 步 gw3 + 45 步 gw7)、mu 随分辨率漂移 —— 错一个不会崩,只会"差一点",极难查。**这正是 harness 的正确用途:做对拍基准。**

---

## 8. 权重获取渠道

### 8.1 HF gated vs ModelScope 开放(同一文件同一时刻实测)

| 源 | 结果 |
|---|---|
| HuggingFace `ideogram-ai/ideogram-4-fp8` | `http=401`,`"Access to model ideogram-ai/ideogram-4-fp8 is restricted."` |
| ModelScope 镜像 | `http=200`,`size=1024`,`ApprovalMode = 0` |

→ **走 ModelScope**。

### 8.2 可用的 ModelScope API 端点(踩坑后确认)

**能用**:

```bash
# 搜索(参数必须字面是 search=,用 Name=/query=/keyword= 会静默返回无关结果)
https://www.modelscope.cn/openapi/v1/models?search=<q>&PageSize=100

# 列文件
https://www.modelscope.cn/api/v1/models/{org}/{name}/repo/files?Revision=master&Recursive=True
```

**404 / 不可用**:`/api/v1/models?Query=`、`/api/v1/dolphin/models`(GET 与 POST 均 404)、`/api/v1/search/models`(503)。

### 8.3 HF 侧枚举

```bash
curl -sS "https://huggingface.co/api/models?search=ideogram-4&limit=60"
```

---

## 9. 推荐执行路线

**主路线 B(移植 vllm-omni),从 PR #4227 起步;harness 不做生产引擎,做 P0 数值基准。**

两者不冲突 —— 方法论 §13 的 P0 本来就写着"harness 跑通稳态矩阵 + 判死结论",harness 的定位就是 P0 工具。

### P0(harness,不碰 GPUStack)
- [ ] ModelScope 拉 nf4
- [ ] 40G A100 单卡跑通 1024² / 2048²
- [ ] 固定 seed 存一批参考图 + 逐步 latent,作为移植对拍基准
- [ ] 实测 turbotime / unconditional 两个 LoRA 的显存与质量,定生产档位
- [ ] 产物防呆三检(体积 / 熵 / 黑屏)

### P0.5(vllm-omni)
- [ ] 拉 #4227 分支评估现状 ← **唯一未落地的决策变量**
- [ ] 合入 #4788 fp8 loader
- [ ] VAE 挂 flux2
- [ ] 13 层 tap 原样搬
- [ ] 稠密 mask → varlen attention
- [ ] 双 DiT 走 pipeline 内部持有
- [ ] 与 P0 基准逐步对拍

### P1-P4(基本为零)
- [ ] 重出 vllm-omni 镜像(补 bnb / Qwen3-VL 依赖)
- [ ] **先把模型注册进 `OMNI_PIPELINES`** —— 未注册的模型传 `--deploy-config` 会被整个静默丢掉,回落单卡默认档,20 分钟后才以 OOM 暴露
- [ ] model-catalog 加条目
- [ ] new-api 加 model ID + `defaultSteps`(步数读 `defaultSteps`,不要硬编码)
- [ ] GPUStack 侧 profile + 卡数

---

## 10. 未决事项

1. **#4227 分支质量**未评估。若太差,P0.5 成本会涨到接近"从零写 pipeline",是唯一会改变结论的变量。
2. `transformerlab/ideogram-4-int8-w8a8` 的 `safetensors_loader.py` / `recipe.json` 张量布局**未与 `CompressedTensorsW8A8Int8` 逐字段对齐**。
3. **商用授权(最高优先级)**:权重为 Ideogram Non-Commercial,对外计费须由 Ideogram 另行书面授权,且其有权不给。**这是唯一可能让整条技术路线作废的外部变量**,应与 P0 并行推进,不要等到 P4。详见 §5。
4. ARM64 上 bitsandbytes(CUDA)wheel 的供应链未验证。
5. ModelScope 下载 + 最小运行脚本尚未编写。

---

## 附录:调研中踩到的坑

| 坑 | 现象 | 规避 |
|---|---|---|
| `gh search issues --state all` | `invalid argument "all" for --state flag` | 去掉 `--state`,加 `--include-prs` |
| GitHub GraphQL | `Post "https://api.github.com/graphql": EOF` | 有界重试循环 |
| `git fetch upstream` | LibreSSL `SSL_ERROR_SYSCALL` | 同上,重试 |
| HF tree API 返回非 JSON | 实为 gated 仓的 401 HTML | 先判 HTTP 码再解析 |
| ModelScope 搜索参数 | `Name=` / `query=` / `keyword=` 静默返回无关结果 | 必须用 `search=` |
| 本机无 `diffusers` | `ModuleNotFoundError` | 用 WebSearch 确认官方 `Ideogram4Pipeline` 存在 |

**相关文档**:`/Users/reputationly/Desktop/code/api/gpustack/docs/新引擎内嵌gpustack-工程化方法论.md`(718 行,P0-P5 全流程 + A100 铁律 + 13 节 checklist + 附录索引)
