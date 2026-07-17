# vLLM-Omni 大模型(🔴 不装单卡)量化 / Offload 可行性调研

> 日期:2026-07-16 · 环境:鲲鹏 920 ARM64 + 4×A100 PCIE 40G(**无 NVLink**,sm_80)
> 背景:选型文档(`vLLM-Omni-语音模型全景与选型.md`)第 10 节把 **Fish Speech S2 Pro / Qwen3-Omni 30B MoE / MiniCPM-o 4.5** 标为 🔴(40G 单卡装不下)。本文回答:**能否通过多卡 / offload / 量化把它们在 4×40G 上跑起来,值不值。**
> 参考:LightX2V `docs/Wan2.2-I2V-实验测试报告.md`(offload/int8 实测经验)。
> ⚠️ 本文结论均为**调研 + 代码走查**,未真机实测;若要上,每个都需按 IndexTTS-2 / quantization skill 的 A/B 流程验证。

---

## 0. 一句话结论

**没有"下载即用、官方验证过"的捷径。** 三个 🔴 模型技术上都能多卡跑,但:

- **offload 不能照搬 Wan2.2 那套** —— 它们瓶颈是 AR(自回归),offload 对 AR 是吞吐杀手,而且 vllm-omni 的 offload 本来就只实现在扩散(DiT)路径。
- **量化才是正解**,但 **A100 sm_80 不支持 FP8**(网上一半量化权重直接出局),只能用 weight-only INT8 / W4A16(AWQ/GPTQ 4bit)。
- 官方量化权重**基本没有**;社区权重存在,但都是给各自原生栈打包的,**不保证 vllm-omni 能直接读**。

**当前决定:先测选型文档里那批 🟢 轻量模型(够用),这三个 🔴 模型暂缓,本文存档备查。**

---

## 1. 硬件红线:A100(sm_80)先天不支持 FP8

Wan2.2 报告结论:*"int8-torchao 在 A100/ARM 上不走 INT8 tensor core,矩阵乘仍按 bf16 算…int8 的价值是显存,不是速度。"*

**FP8 更进一步:sm_80 连 FP8 tensor core 都没有。** 需 compute capability ≥ 8.9(Ada / Hopper / Blackwell,即 4090 / H100 / 5090)。

后果:网上大量 FP8 量化权重(如 `drbaph/s2-pro-fp8`、`AEmotionStudio/fish-speech-s2-pro-fp8`)模型卡明写 **"requires Ada Lovelace or Blackwell"**,在 A100 上加载即报错。

→ **A100 上可用的量化只剩:weight-only INT8、W4A16(GPTQ/AWQ 4bit)。** 4bit 权重解量化回 fp16 算矩阵,sm_80 支持;**省显存、不提速**(和 int8 同理)。

---

## 2. 为什么 offload 不能照搬 Wan2.2

### 2.1 机制差异:扩散(算力受限) vs AR(访存受限)

| | Wan2.2(扩散 DiT) | Fish-S2 / Qwen3-Omni / MiniCPM-o(AR) |
|---|---|---|
| 计算模式 | N 步去噪**循环复用同一份权重**,单步算力重 | **每个 token 一次完整前向过所有层**,一秒几十上百 token |
| 瓶颈 | **算力受限**,搬权重能和计算 overlap | **访存受限**,搬权重卡在带宽 |
| MoE offload | 只把当前激活的 1 个专家放 GPU,可行 | 每 token 都要流一遍模型,30B(~60G)/token → 秒级出字 |
| 是否离线批任务 | 是,慢一倍能接受 | 否,交互/流式,延迟敏感 |

**一句话:offload 对扩散是"慢一点还能用",对 AR 是"直接废掉"。** 没 NVLink 的 PCIE 有效带宽 ~20GB/s,per-token 搬整个模型不可行。

### 2.2 代码印证:vllm-omni 的 offload 是扩散专用

- offload machinery 全在 **`vllm_omni/diffusion/offloader/`**(`LayerwiseOffloadHook` 等)。
- `enable_cpu_offload` / `enable_layerwise_offload` 配置项、`_layerwise_offload_blocks_attrs` 挂在 **DiT block** 上(如 `glm_tts_dit.py`)。
- **AR 解码路径没有走这套** —— 想用 offload 把 Qwen3-Omni 的 thinker 塞进 40G,框架层面也没这条路。

---

## 3. 多卡可行性(若不走单卡)

这几个都是**多 stage 模型**,vllm-omni 支持把不同 stage / TP 拆到多卡。关键约束:**PCIE 无 NVLink**。

- **Stage 并行**(不同阶段放不同卡,只传小 token 张量):跨卡通信量小,**PCIE 够用,几乎无损**。
- **TP 张量并行**(一个大模型切多卡,每层 all-reduce):**吃卡间带宽,无 NVLink 明显掉速**;MoE 还加 all-to-all,最惨。

| 模型 | 4×40G PCIE 能跑? | 怎么跑 | 代价 |
|---|---|---|---|
| **MiniCPM-o 4.5** | ✅ 能,较干净 | recipe 自带 **2/3/8 卡布局**;2 卡=thinker GPU0 + talker GPU1(纯 stage 拆);官方明说 "most 40GB+ pairs 只要 thinker 权重装得下" | 2 卡布局几乎无损 |
| **Qwen3-Omni 30B MoE** | ⚠️ 能,偏疼 | 官方 stage-based 三进程(thinker/talker/code2wav);但 30B bf16 ~60G,**thinker 单 stage 超 40G → 必须 TP=2** | thinker TP=2 走 PCIE all-reduce + MoE all-to-all,延迟最高档 |
| **Fish-S2 Pro** | ⚠️ 最勉强 | recipe **只给 1×A800-80G / 2×H100-80G**,峰值 ~48.9G;40G 无现成配方,TP=2 理论可行但 Fish 双 AR+DAC,TP 支持未验证 | 无官方 40G 路径,坑最深 |

---

## 4. 量化权重盘点

### 4.1 官方验证过的:三个基本都没有

| 模型 | 官方量化权重 | 说明 |
|---|---|---|
| Fish-S2 Pro | ❌ | `fishaudio/s2-pro` 官方**只发 bf16**,量化全是社区 |
| Qwen3-Omni | ❌ | 官方给非-Omni 的 Qwen3 发了 `-FP8`,**Omni 变体没做** |
| MiniCPM-o 4.5 | 🟡 int4/GGUF/BNB | openbmb 官方出了,但**给自家 llama.cpp/BNB 栈用**,非 vllm-omni 验证格式 |

**"下载即用、官方验证过"的量化权重 —— 对这三个不存在。**

### 4.2 A100 能用的社区权重(FP8 已排除)

| 模型 | 权重 | 格式 | 显存 | 坑 |
|---|---|---|---|---|
| Fish-S2 | `Imagilux/fishaudio-s2-pro` | INT8 weight-only | ~5G | 只量化 Slow AR,原为 AMD ROCm 调,CUDA 也能跑 |
| Fish-S2 | `baicai1145/s2-pro-w4a16` | GPTQ W4A16 | 更小 | 只量化 Slow AR 4B 主干,codec/Fast AR 仍高精度 |
| Qwen3-Omni | `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` | AWQ 4bit | thinker ~15-17G | 社区;4bit 单卡 40G 宽裕 |
| Qwen3-Omni | `cyankiwi/...-AWQ-8bit` | AWQ 8bit | ~30G | 质量更稳,也进 40G |
| MiniCPM-o | openbmb 官方 int4 | int4/BNB | — | 给自家栈用,vllm-omni 能否直接吃未验证 |

⚠️ **全是社区权重,且都为各自原生栈(ComfyUI / fishaudio 原生 / llama.cpp)打包**,不保证 vllm-omni 的 awq/gptq loader 能直接读(tensor 命名 + scale 布局要对得上)。

---

## 5. 框架量化支持:接线在、文档无

代码走查结论:

- ✅ **AR/thinker stage 已接 `quant_config`**:`qwen3_omni_moe_thinker.py`、`fish_speech_slow_ar.py` 里都有 → 架构上支持给 AR 主干加载量化权重,走上游 vllm 的 awq/gptq 路径。
- ✅ **可加载格式(AR 侧,委托上游 vllm)**:`awq`、`gptq`、`auto-round/inc`。A100 上 awq/gptq 4bit 可用,**fp8 不行**(需 sm_89+)。
- ❌ **本地"开机即量化"(on-the-fly)基本没戏**:awq/gptq 需**离线校准**产出 checkpoint,不是加载 bf16 时顺手量化;能 online 的 `int8`(`DiffusionInt8Config`)是**扩散专用**,不覆盖这些 AR 模型。
- ❌ **无任何针对 Fish-S2 / Qwen3-Omni / MiniCPM-o 的量化文档 / example** → 官方没验证过,上了就是第一个吃螃蟹的。

vllm-omni 量化统一入口:`vllm_omni.quantization.build_quant_config()`;本地 override 偏扩散/NPU/XPU(`gguf`/`int8`/`mxfp8`/`mxfp4`/`inc`/`modelopt`),AR 方法委托上游 vllm。

---

## 6. 结论 & 若要推进的 ROI 排序

**没有"下个量化权重就能跑"的成熟路径,每个都要一轮真机验证(格式兼容 + 质量 A/B + A100 确实不掉质量)。** 若一定要做:

1. **Qwen3-Omni AWQ-4bit(`cyankiwi`)最值得先 spike** —— Qwen3-Omni + AWQ 是上游 vllm 支持最成熟的组合,thinker `quant_config` 接线在,4bit 后 thinker ~16G 单卡随便装。风险:社区权重 + vllm-omni MoE-AWQ 加载要实测。
2. **Fish-S2 W4A16/INT8** —— 想要 44.1k 高质量再碰;权重是 fishaudio 原生格式,**大概率要转格式**才能喂 vllm-omni,坑更深。
3. **MiniCPM-o** —— 若做语音对话数字人,直接走 **2 卡 stage 拆的 bf16**(recipe 自带,PCIE 友好),比折腾量化更省心;官方 int4 与 vllm-omni 兼容性最不明朗。

### 重要提醒:大模型 ≠ 更好配音

- **纯配音音质与参数量不线性相关**:全家情感控制最强的是 IndexTTS-2(1.5B),48kHz 高保真是 VoxCPM2(2B)/ MOSS-Nano(0.1B)。音质主要由 codec/vocoder + 数据决定,不是 LLM 大小。
- Qwen3-Omni / MiniCPM-o 的价值在 **omni 能力(语音+视频+图理解、多轮语音对话)**,那是**交互式数字人**引擎,不是更好的**逐句配音**引擎。
- **判断标准**:目标是短剧逐句配音 → 这三个都别费劲,🟢 轻量那批够用;要做"和数字人实时语音聊天"产品 → 才上 MiniCPM-o(2 卡)。

---

## 7. 当前决定

**先测选型文档里那批 🟢 轻量模型(下载脚本 `scripts/download_speech_models.sh` 默认 12 个 + ACE-Step),感觉够用。** 本文三个 🔴 模型暂缓,等确有语音对话产品需求 / 想做 44.1k 高质量 A/B 时,再按第 6 节 ROI 顺序单独立项验证。

---

*配套文档:`vLLM-Omni-语音模型全景与选型.md`(全家族选型地图)、`IndexTTS2-vLLM-Omni-实验测试报告.md`(单模型深度实测模板)。*
