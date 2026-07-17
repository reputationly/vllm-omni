# vLLM-Omni 语音模型全景、能力对比与 ARM+4×A100 部署选型

> 日期:2026-07-15 · 环境目标:鲲鹏 920 ARM64 + 4×A100 PCIE 40G(无 NVLink)· 隔离网
> 数据来源:vllm-omni 仓 `docs/models/supported_models.md`、`recipes/`、`examples/{offline,online}_inference/text_to_speech/`、`vllm_omni/deploy/*.yaml`
> ⚠️ **可信度**:除 **IndexTTS-2 为本团队真机实测**(见 `IndexTTS2-vLLM-Omni-实验测试报告.md`),其余规格来自官方 recipe / HF 模型卡 / deploy 配置,**接入前均需按 IndexTTS-2 同样流程各测一轮**(arm64 sm_80 kernel 覆盖、显存、离线权重布局、崩溃边界)。部分参数标注"推断"表示文档未明载、按同类模型估计。

---

## 0. 为什么要这份文档

前期只看了在线 TTS README 的 13 个模型,**遗漏不少**:完整 `supported_models.md` 里能做**语音输出**的模型有 **25+**,还分成 TTS、音色设计、多说话人对话、音效/音乐、歌声、Omni 语音对话、ASR 七大类。本文把全部语音模型盘清、逐个挖特性、做横向大表,并给出 ARM+4×A100(40G) 的部署建议。

调用方式统一:`vllm serve <model> --omni --trust-remote-code --port <p> --deploy-config <yaml>`,OpenAI 兼容 `POST /v1/audio/speech`(音效/对话类走同端点不同参数)。IndexTTS-2 的实战(镜像构建、离线权重 overlay、tokenizer 坑、句级切分硬约束)是所有模型的通用模板。

---

## 1. 一句话结论 + 类别地图

**vllm-omni 是"一个引擎跑全部语音模型"** —— 同一镜像 + 不同 deploy yaml + 不同权重,按场景路由。你们规划的"内嵌 GPUStack 提供不同语音模型"完全可行。类别:

| 类别 | 干什么 | 代表模型 | 短剧配音用途 |
|---|---|---|---|
| **A 纯 TTS / 音色克隆** | 文本+参考音色→语音 | **IndexTTS-2**、Qwen3-TTS、CosyVoice3、MOSS-TTS、VoxCPM2、Fish-S2、GLM-TTS、Ming-omni-tts、Voxtral、Higgs-v2/v3、MiMo-Audio | **逐句配音主力** |
| **B 音色设计**(无需参考音频) | 文字描述→造一个新声线 | MOSS-VoiceGenerator、Qwen3-TTS-VoiceDesign、OmniVoice、Ming-flash-omni | 凭空造角色声线 |
| **C 多说话人对话** | 一次生成多角色对白 | **MOSS-TTSD** | 对手戏一次成型 |
| **D 音效 / 音乐** | 文字描述→音效/BGM | **MOSS-SoundEffect**、AudioX、Stable-Audio-Open、Ming-omni-tts(TTA) | 环境音、配乐、拟音 |
| **E 歌声** | 歌词+旋律→唱;或转换嗓音 | **SoulX-Singer**(SVS+SVC) | 角色唱段、主题曲 |
| **F Omni 语音对话** | 语音进语音出的对话模型 | Qwen3-Omni、MiniCPM-o 4.5、Covo-Audio-Chat、Qwen2.5-Omni、MiMo-Audio、Dynin-Omni、Aura-Omni | 交互式数字人 |
| **G ASR**(相邻,输入侧) | 语音→文字 | MiMo-V2.5-ASR | 字幕/对齐 |

---

## 2. A 类:纯 TTS / 音色克隆(配音主力)

按参数量从轻到重排(直接对应你们 40G 单卡能否装下)。

### IndexTTS-2 ✅ 已实测
- ~1.5B · 22.05kHz · 中英 · 2 阶段(GPT AR → S2Mel DiT + BigVGAN)
- **杀手锏:8 维情感向量(喜怒哀惧厌郁惊平)+ 强度**,还有情感音频/情感文本三模式——**A 类里情感控制最精细的**
- 实测:单请求 1.7-1.9s(RTF 0.37),8 并发 5.2× 实时,300 句零故障,显存 24.6G/40G
- ⚠️ 句级上限 216~324 字,超限杀引擎不自愈;seed 只锁 AR 不锁扩散
- 需参考音色(必需),无文字预设

### CosyVoice3 — 极轻量
- 0.5B · 24kHz · 阿里 FunAudioLLM · 2 阶段 · 流式
- 零样本克隆,成熟稳定;**参数最小之一,40G 卡可跑高密度多实例**
- 在线示例暂缺(仓库标注),先走离线

### Ming-omni-tts — 方言 + 长音频王
- 0.5B · **44.1kHz** · 2 阶段 · 流式
- **特色最丰富:风格 / IP 音色 / 方言(粤语等)/ 文生音效(TTA)/ 播客 / 语音+BGM / 语音+环境音**
- 音色用嵌入,**无需 ref_text**;要方言、要带配乐/环境音一次出、要长播客,选它

### Qwen3-TTS — 多语言 + 全能
- **1.7B / 0.6B 两档** · 24kHz · **600+ 语言(零样本多语)** · 2 阶段 · 流式 PCM + **WebSocket**
- 3 变体:CustomVoice(克隆)/ VoiceDesign(造音色)/ Base(需 ref_text)
- 有**预设音色 + `/v1/audio/voices` 上传管理**(IndexTTS 缺的);情感控制
- **出海多语言 + 实时 WebSocket + 音色库,综合能力最全,且轻量能装下** —— 强烈建议排第二个接

### MOSS-TTS-Nano — 最轻 + 高保真
- **0.1B(!)** · **48kHz** · 单阶段 AR + codec · 中/英/日 · 流式
- 参数极小但 48k 高保真;需 ref_audio(必需);有续写模式
- 追求"轻 + 高采样率"选它

### MOSS-TTS-Realtime — 低延迟
- 1.7B · 24kHz · **TTFB ~180ms** · 原生流式(async_chunk)
- 实时/边生成边播场景;比 8B 版轻,40G 可跑

### VoxCPM2 — 48k 高保真
- 2B · **48kHz** · 单阶段原生 AR · 30+ 语言 · 流式 · 续写
- **影视级音质候选**(48k vs IndexTTS 22k);RTX4090 24G 就能跑,40G 宽裕

### Higgs-Audio v2 / v3 — 情感韵律
- v2: 3B / v3: 4B · 24kHz · v3 **100+ 语言**
- 情感/韵律/风格控制;v3 官方建议 H100 80G(stage0 占 60%),**40G 上偏紧需实测**

### Voxtral TTS — 欧语强
- 4B · 24kHz · Mistral 系 · 预设音色 · 流式
- 英文/欧语质量好;上游 gated(要授权);40G 单卡可跑

### Fish Speech S2 Pro — 高质量但重
- 双 AR + DAC codec · **44.1kHz** · 需 ref_text
- 官方要 **A800 80G(~48.9G 峰值)或 2×H100** —— **40G 单卡装不下**,要多卡或降配,优先级靠后

### GLM-TTS — 智谱备选
- 24kHz · 中英 · 2 阶段(AR+DiT)· 流式 · 克隆需 ref_text
- 约 18-20G 显存,40G 宽裕;作主力备选

### MiMo-Audio-7B — Omni 兼 TTS
- 7B · 24kHz(推断)· 小米 · 2 阶段 · 11 种任务(TTS/克隆/理解/多轮)
- 8B 级,40G 单卡偏紧(见 §10);既是 TTS 也是 Omni,归到 F 类更合适

---

## 3. B 类:音色设计(无需参考音频,凭空造声线)

| 模型 | 规格 | 特点 |
|---|---|---|
| **MOSS-VoiceGenerator** | 1.7B · 24kHz | 文字指令(如"女性、低沉、英国口音")→ 合成符合描述的声音,轻量 |
| **Qwen3-TTS-VoiceDesign** | 1.7B · 24kHz | Qwen3-TTS 的音色设计变体,与克隆/Base 同架构 |
| **OmniVoice** | 24kHz | 音色设计 + 语言提示,轻量 |
| **Ming-flash-omni-TTS** | 44.1kHz · caption 控制 | 用字幕/caption 控风格+IP音色,无克隆;omni 模式要 4×H100,纯 TTS 模式 1×H100 |

用途:没有参考音频、要设计一个全新角色声线时用。IndexTTS/MOSS-TTS 只能克隆已有音色,这类补空白。

---

## 4. C 类:多说话人对话

**MOSS-TTSD-v1.0** · 8B · 24kHz · 用 `[S1]`/`[S2]` 标记多说话人 · 支持克隆 + 对话模式。
一次请求生成多角色对白,省掉自己逐句合成再拼接。对手戏、群聊场景。8B 级显存(见 §10)。

---

## 5. D 类:音效 / 音乐生成

| 模型 | 规格 | 特点 |
|---|---|---|
| **MOSS-SoundEffect** | 8B · 24kHz | 文字描述("雷声+雨打铁皮屋顶")→ 音效,**无需参考音频**;`duration_seconds` 控时长 |
| **AudioX** | 44.1kHz 立体声 | 音乐+音效,多任务(t2a/t2m/v2a/v2m…);**仅 L4 24G ~10G,最省显存**;离线,无 OpenAI 端点 |
| **Stable-Audio-Open** | 44.1kHz | 音乐+音效;RTX4090 24G ~12.6G;离线 |
| **Ming-omni-tts(TTA 模式)** | 44.1kHz | 顺带做文生音效,见 §2 |

用途:配音之外的音频资产——环境音、拟音、BGM。短剧后期能省外采。

---

## 6. E 类:歌声合成(全家唯一)

**SoulX-Singer** · 24kHz · 两个 pipeline:
- **SVS(歌声合成)**:歌词+旋律 → 唱
- **SVC(歌声转换)**:把一段唱转成目标音色

短剧的角色唱段、主题曲**只有它能做**,不可替代。⚠️ 权重要手动补 `phone_set.json`(HF 不发),批处理非流式。

---

## 7. F 类:Omni 语音对话(语音作为一种模态)

这些是"语音进、语音出"的对话大模型,不是纯 TTS,做交互式数字人/语音助手:

| 模型 | 规格 | 语音能力 | 显存 |
|---|---|---|---|
| **MiniCPM-o 4.5** | 24kHz 输出 | 完整 omni:文本+语音输出,视频/图/音输入,双向对话 | 2-8 卡可配 |
| **Covo-Audio-Chat** | 7B · 24kHz 出/16kHz 入 | fused_thinker_talker,单 AR 交错生成文本+音频 token | A100 80G(实际 ~18G) |
| **Qwen3-Omni** | 30B MoE(A3B 激活) | 语音理解为主,可经 talker 输出音频;WebSocket 实时 | **需多卡**(30B 权重) |
| **Qwen2.5-Omni** | 7B / 3B | omni 语音 | 7B 单卡 40G 可 |
| **MiMo-Audio-7B** | 7B · 24kHz | TTS+理解+多轮,11 任务 | 40G 偏紧 |
| **Dynin-Omni** | 3 阶段 | t2s 等 7 任务,含文本→语音 | A100 80G/~56G |
| **Aura-Omni** | 级联 | ASR→VL→Qwen3-TTS→Code2Wav 音视频对话系统 | 单卡分阶段 |

用途:如果产品要"和数字人语音聊天",走这类;纯配音不需要。

---

## 8. G 类:ASR(相邻,输入侧)

**MiMo-V2.5-ASR** · 7B · 16kHz 输入 · 纯语音转文字,无输出。用途:配音底稿转写、字幕对齐、时间戳。(注:Qwen3-Omni/MiniCPM-o 也自带 ASR 能力。)

---

## 9. 横向对比大表

> 显存列:🟢=40G 单卡宽裕 · 🟡=40G 单卡偏紧需实测 · 🔴=40G 装不下需多卡/H100。除 IndexTTS-2 外均为估计。

| 模型 | 类别 | 参数 | 采样率 | 语言 | 克隆 | 流式 | 情感/特色 | 40G单卡 | 在线API |
|---|---|---|---|---|---|---|---|---|---|
| **IndexTTS-2** ✅ | A | ~1.5B | 22.05k | 中英 | 需ref | ✗(兼容) | **8维情感向量**(最强) | 🟢 24.6G实测 | ✓ |
| Qwen3-TTS | A/B | 0.6/1.7B | 24k | **600+** | ✓ | PCM+**WS** | 音色库+设计+情感 | 🟢 | ✓ |
| CosyVoice3 | A | 0.5B | 24k | 中英 | ✓ | ✓ | 极轻量 | 🟢 | 离线 |
| Ming-omni-tts | A/D | 0.5B | **44.1k** | 中+方言 | ✓ | ✓ | **方言/播客/BGM/环境音** | 🟢 | ✓ |
| MOSS-TTS-Nano | A | **0.1B** | **48k** | 中英日 | 需ref | ✓ | 最轻+高保真 | 🟢 | ✓ |
| MOSS-TTS-Realtime | A | 1.7B | 24k | 多 | ✓ | ✓ **180ms** | 低延迟 | 🟢 | ✓ |
| VoxCPM2 | A | 2B | **48k** | 30+ | ✓ | ✓ | 48k高保真 | 🟢 | ✓ |
| GLM-TTS | A | ~? | 24k | 中英 | 需ref+text | ✓ | 备选 | 🟢 ~18G | ✓ |
| Higgs-Audio v2 | A | 3B | 24k | 多 | ✓ | ✗ | 情感韵律 | 🟡 | ✓ |
| Higgs-Audio v3 | A | 4B | 24k | **100+** | 可选ref | ✗ | 情感韵律 | 🟡 官方H100 | ✓ |
| Voxtral TTS | A/B | 4B | 24k | 欧语强 | ✓ | ✓ | 预设音色(gated) | 🟡 | ✓ |
| MOSS-TTS | A | 8B | 24k | 多 | ✓ | ✓ | 最高质量 | 🟡 ~26G | ✓ |
| MOSS-TTSD | C | 8B | 24k | 多 | ✓ | ✓ | **多说话人对话** | 🟡 | ✓ |
| MOSS-SoundEffect | D | 8B | 24k | — | — | ✓ | **文生音效** | 🟡 | ✓ |
| MOSS-VoiceGenerator | B | 1.7B | 24k | 多 | — | ✓ | **音色设计** | 🟢 | ✓ |
| Ming-flash-omni-TTS | B | ~? | 44.1k | 中 | caption | ✗ | caption控风格 | 🟡(TTS模式1×H100) | ✓ |
| OmniVoice | B | ~? | 24k | 多 | ✓ | ✗ | 音色设计+语言提示 | 🟢 | 离线 |
| Fish Speech S2 Pro | A | ~? | **44.1k** | 多 | 需ref+text | ✓ | 高质量 | 🔴 ~49G | ✓ |
| **SoulX-Singer** | E | ~? | 24k | — | prompt | ✗ | **唱歌 SVS+SVC** | 🟡 | 部分 |
| AudioX | D | ~? | 44.1k立体 | — | — | ✗ | 音乐+音效,最省显存 | 🟢 ~10G | 离线 |
| Stable-Audio-Open | D | ~? | 44.1k | — | — | ✗ | 音乐+音效 | 🟢 ~13G | 离线 |
| MiMo-Audio-7B | A/F | 7B | 24k | 多 | ✓ | ? | omni多任务 | 🟡 | ✓ |
| Covo-Audio-Chat | F | 7B | 24k | 多 | ✗ | ✗ | 语音对话 | 🟡 ~18G | ✓ |
| MiniCPM-o 4.5 | F | ~? | 24k | 中英 | ✗ | ✓ | omni双向 | 🔴 多卡 | ✓(chat) |
| Qwen3-Omni | F | 30B MoE | ? | 多 | ✗ | WS | omni理解+语音 | 🔴 多卡 | ✗(realtime) |
| Qwen2.5-Omni | F | 3/7B | ? | 多 | ✗ | ✓ | omni语音 | 🟡(7B) | ✓ |
| MiMo-V2.5-ASR | G | 7B | 16k入 | 多 | — | — | 纯ASR | 🟡 | 离线 |

---

## 10. ARM + 4×A100(40G) 部署建议

### 10.1 硬约束:显存分级(40G/卡,无 NVLink)

- **🟢 装得下且宽裕(≤2B 权重 + codec)**:IndexTTS-2、Qwen3-TTS、CosyVoice3、Ming-omni-tts、MOSS-Nano/Realtime、VoxCPM2、MOSS-VoiceGenerator、GLM-TTS、AudioX、Stable-Audio。**可单卡多实例或留并发余量。**
- **🟡 单卡偏紧(3-8B,权重+7-8G codec 逼近 40G)**:MOSS-TTS/TTSD/SoundEffect(8B ~26G)、Higgs v2/v3、Voxtral、MiMo-Audio、Covo、SoulX-Singer、Qwen2.5-Omni-7B。**能跑但并发余量小,需逐个实测峰值显存**(参考 IndexTTS 报告方法)。
- **🔴 单卡装不下,要多卡或换 H100**:Fish-S2(~49G)、Qwen3-Omni(30B)、MiniCPM-o 4.5。**A100/40G 上这几个要么多卡 ulysses/TP、要么放弃**(参考 Wan2.2 报告:bf16 多卡在 40G+256G 内存机器上有 CPU OOM 风险,大模型优先量化)。

### 10.2 四卡怎么排(生产拓扑建议)

短剧配音是**多模型并存、按场景路由**,不是单模型压满四卡。建议:

| 卡 | 常驻模型 | 理由 |
|---|---|---|
| GPU0 | **IndexTTS-2**(主力逐句配音) | 已验证,情感控制强,轻量 |
| GPU1 | **Qwen3-TTS**(多语言/音色库/实时) | 综合最全,补 IndexTTS 的多语言+预设音色 |
| GPU2 | **SoulX-Singer**(唱段)或 **Ming-omni-tts**(方言/带BGM) | 按内容需求择一,不可替代能力 |
| GPU3 | **机动 / 音效**(MOSS-SoundEffect / AudioX)或第 4 个 TTS 实例扛峰值 | 音效资产 + 弹性并发 |

- **提吞吐靠横向扩,不是调 batch**:IndexTTS 实测 max_num_seqs=4 已是单卡最优,8 反降。要更高并发就多卡多实例 + GPUStack 调度分发,不是单实例加 batch。
- **8B 类想上生产**:优先跑 40G 单卡(MOSS-TTS ~26G 可),不行再考虑量化(参考 Wan2.2:int8 省显存但 A100 无 int8 tensor core 不提速,价值是装得下)。

### 10.3 每个模型接入前必测(照 IndexTTS-2 模板)

1. **arm64 sm_80 kernel 覆盖**:构建期查 torch + vllm `.so` 含 sm_80(IndexTTS 已证 `vllm/vllm-openai` arm64 base 满足)
2. **离线权重布局**:附属模型(codec/vocoder)按各模型本地查找名放进模型目录,`HF_HUB_OFFLINE=1`
3. **deploy yaml 的 tokenizer 占位坑**:占位 tokenizer 改本地路径(IndexTTS 踩过)
4. **崩溃边界**:超长文本/异常输入是否杀引擎 → facade 前置校验 + `--restart`
5. **峰值显存 + 并发吞吐 + 长跑**:确认 40G 余量和无泄漏

### 10.4 facade 层通用(所有模型复用)

M4 facade 做的**句级切分、pyloudnorm 响度归一化、`loudness_lufs`/`gain_db` 音量参数、长度硬闸、任务队列**——引擎无关,一次做好全家复用。这是"多模型统一接入"的关键:引擎只管推理保持上游原样,策略全在 facade。

---

## 11. 给你们的落地优先级

1. **IndexTTS-2**(已验证)→ 先接 GPUStack Custom 后端投产
2. **Qwen3-TTS**(多语言/音色库/WebSocket,轻量)→ 第二个,补最多空白
3. **SoulX-Singer**(唱段)→ 有唱歌需求就接,不可替代
4. **Ming-omni-tts**(方言/带 BGM/播客)或 **VoxCPM2**(48k 高保真)→ 按内容需求
5. **MOSS-SoundEffect / AudioX**(音效资产)→ 后期增强
6. Omni 语音对话(MiniCPM-o 等)→ 只有做交互式数字人产品时才需要

---

*本文与 `IndexTTS2-vLLM-Omni-实验测试报告.md` 配套:后者是单模型深度实测,本文是全家族选型地图。所有非 IndexTTS-2 的规格接入前需按同样流程验证。*
