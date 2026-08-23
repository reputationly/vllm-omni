# MiniMax-Music3 接入调研与路线选型报告

> 调研日期：2026-08-16
> 调研范围：权重结构、模型事实、与 ACE-Step 1.5 的横向对比、第三方实测证据、
> GPUStack 接入路线、diffusers 工程化路线
> 权重状态：**已下载完成**，落 `/nfs-models/wuhanjisuan894/models/MiniMax-Music3`（54 GiB）
>
> 文档定位：这不是一份实验测试报告（我们**没有在自己的机器上跑过 Music3**），
> 而是一份**决策依据归档**。所有性能与音质结论都来自第三方，来源与日期逐条标注。
> 后面重启这个话题时，先看 §8「重启决策的触发条件」，再看 §7「未验证清单」。
>
> **2026-08-23 更正（决议已重启，见下方 §0 后的更正说明）**：本文档 §5.1 的
> cu130 结论、§6.1 的 `/v1/tasks/music/` 接入路径设想，**均基于「我们自己实现」
> 的假设，与 upstream vLLM-Omni 实际已完成的原生移植不符**，不要照抄这两节的
> 具体路径，只参考横向对比数据（能力子集、成本、音质证据）。详细更正与最新
> 执行方案见 `docs/vLLM-Omni-Upstream同步执行方案与开发约定-2026-08-23.md` §4/§8。

---

## 0. 决策记录

**2026-08-16 决议：暂不适配 MiniMax-Music3，音乐生成继续用 ACE-Step 1.5。**
**2026-08-23 决议重启：正式排期上线**（触发条件见 §8 第 1 条——产品侧确认要做精品档）。
下方决策依据仍是当时的真实记录，保留不改；决策本身已被上面新决议覆盖。

决策依据（按权重排序）：

1. **能力面是 ACE-Step 的真子集，且是结构性的**。MiniMax 没有释出完整 encoder，
   参考音频、音色克隆、cover、repaint **永远做不了**——不是等下个版本的问题。
   而我们门面里 `t2m / cover / repaint` 三个格子已经为 ACE-Step 开好了，
   接 Music3 只能填其中一格。
2. **成本与时延差一个数量级**。第三方实测 RTF 1.2–2.1，ACE-Step 0.17；
   同等产出的算力成本约 23×。
3. **授权带来的不是工程成本而是产品/法务成本**。ACE-Step 是 MIT；Music3 的社区许可
   要求商业产品 UI 上显著展示 "MiniMax-Music3"，并要求托管服务方实施、维护、
   测试并定期审查防侵权保障措施。
4. **我们集群吃不到它的最优性能**。它的 int8 加速核需要 cu130+，我们 A100 集群
   驱动 570.86.10 锁死在 cu128，升不上去（见 §5.1）。

**但结论不是"这个模型不行"**——第三方对比一致认为它的成品音质优于 ACE-Step。
正确的定位是**高端档**而非替代档，见 §3.5。什么条件下应该重开这个决策，见 §8。

---

## 1. 权重与下载（已完成）

### 1.1 下载脚本

`scripts/download_minimax_music3.sh`（431 行）

沿用 `download_minimax_h3_ref2va.sh` 的骨架（ModelScope 主源 → hf-mirror 回退、
实时测速、断点续传、失败追踪），外加 `download_minimax_h3_turbo_lora.sh` 的逐文件校验。

```bash
tmux new -s dl_mm3 -d 'bash /root/download_minimax_music3.sh'
tail -f /tmp/dl_minimax_music3.log
```

| 环境变量 | 作用 |
| --- | --- |
| `SET=full`（默认） | 两套权重 + 官方样例音频，57.35 GB |
| `SET=core` | 只要 diffusers ModularPipeline 的 7 个组件，28.52 GB |
| `SET=native` | 只要 `qwen_7B/` + 两个 `.pth`，28.80 GB |
| `SOURCE=hf` | 跳过魔搭 |
| `VERIFY_SHA=1` | 逐文件算 SHA-256（57 GB 要几分钟；默认只比字节数） |
| `STRICT=0` | 清单对不上只告警不退出 |

清单快照：2026-08-16，ModelScope `master` / HF `main` sha `fbdf52f`。
**两边逐文件字节数完全一致**，差异只有 `.gitattributes`（HF 329 B / MS 2473 B）
和 MS 独有的 `configuration.json`（74 B），这两个文件不进清单。

上游更新后重新生成清单：

```bash
curl -s 'https://modelscope.cn/api/v1/models/MiniMax/MiniMax-Music3/repo/files?Revision=master&Recursive=True' \
  | python3 -c 'import json,sys; [print(f"{f[\"Path\"]}|{f[\"Size\"]}|{f[\"Sha256\"]}") for f in json.load(sys.stdin)["Data"]["Files"] if f.get("Type")=="blob"]'
```

> ModelScope API 返回的 `Sha256` 已验证 = 文件内容的 sha256（对 `config.json`、
> `modular_model_index.json`、`scheduler_config.json` 三个小文件本地核过），
> 不是 git blob sha1，可以直接用于校验。

### 1.2 仓库结构：**一个仓装了两套互不重叠的权重**

这是这个仓最容易踩错的地方——`qwen_7B/` 不是补充件，它是 A 那套的**另一种打包**。

**A. diffusers ModularPipeline（`SET=core`，28.52 GB，25 个文件）**

`modular_model_index.json` 声明 `_class_name = MiniMaxMusic3ModularPipeline`，
blocks 类 `MiniMaxMusic3Blocks`，diffusers 版本 `0.40.0.dev0`，7 个组件：

| 子目录 | 体积 | 类 | 关键配置 |
| --- | --- | --- | --- |
| `transformer/` | 9.73 GB | `MiniMaxMusic3Transformer1DModel`（diffusers） | 36 层，32 头 × head_dim 64，in_channels 128，condition_dim 2048，ff_inner_dim 8192 — Flow Matching 2.4B，**fp32 存储** |
| `language_model/` | 17.17 GB | `Qwen3ForCausalLM`（transformers） | 全局 LLM 8B，出第 1 层 RVQ |
| `rvq_depth_decoder/` | 1.29 GB | `MiniMaxMusic3RVQDepthDecoder`（diffusers） | 4 层，hidden 4096，num_codebooks 8，audio_vocab_size 1024 — 局部 LLM 0.6B，出其余 7 层 |
| `vocoder/` | 0.22 GB | `MiniMaxMusic3Vocoder`（diffusers） | **sampling_rate 44100**，latent_channels 128，upsampling_ratios [8,8,4,2] — Flow-VAE Decoder 123M |
| `condition_encoder/` | 0.10 GB | `MiniMaxMusic3ConditionEncoder`（diffusers） | 输入 24 kHz / hop 960 → 输出 44.1 kHz / hop 512 |
| `tokenizer/` | 11 MB | `Qwen2Tokenizer`（transformers） | 带 `chat_template.jinja` |
| `scheduler/` | 483 B | `FlowMatchEulerDiscreteScheduler`（diffusers） | |

**vLLM-Omni 接入要对齐的是这一套。**

**B. 官方原生 / SGLang-Omni 那套（`SET=native`，28.80 GB，57 个文件）**

| 路径 | 体积 | 说明 |
| --- | --- | --- |
| `qwen_7B/qwen_7B/` | 18.46 GB | `AbabForCausalLM`，`model_type: mixtral`，vocab **200000**，36 层，hidden 4096，num_local_experts 1，内含 `decoder_num_layers: 4` 的 depth decoder、`audio_num_codebooks: 8` |
| `qwen_7B/qwen3-8B-tokenizer-music/` | 16 MB | 带音乐 token 的 tokenizer |
| `flowmatching_vae.pth` | 9.83 GB | 单文件最大 |
| `dav.pth` | 0.49 GB | |

⚠️ `qwen_7B/qwen_7B/config.json` 的 `auto_map` 指向 `configuration_abab.py` /
`modeling_abab.py`，**这两个 .py 不在仓里**。所以这套不能靠 `trust_remote_code`
直接 `from_pretrained`，必须由 sgl-omni 之类自带实现的引擎加载。

**默认下全套的理由**：`modular_model_index.json` 里没有任何一项引用那两个 `.pth`，
README 也没说 A 路径是否间接用到它们——这个判断**没有跑通验证过**，赌错要重下 10 GB，
省下的只有磁盘。另外要做「官方原生 vs 移植」的逐位对齐时，B 那套是参照物。

### 1.3 落地状态

```text
/nfs-models/wuhanjisuan894/models/MiniMax-Music3   54 GiB（du -sh）
├── assets/minimax_ttm.wav        36 MB   官方样例输出，做验收 A/B 的基线
├── condition_encoder/            96 MB
├── language_model/               16 GB
├── modular_model_index.json
├── qwen_7B/                      18 GB
├── rvq_depth_decoder/           1.3 GB
├── scheduler/  scripts/  tokenizer/
├── transformer/                 9.1 GB
├── vocoder/                      207 MB
├── dav.pth  flowmatching_vae.pth
└── config.json  README.md  LICENSE
```

85 个文件字节数全部对上。**未做 SHA-256 复核**（默认只比字节数），
需要时 `VERIFY_SHA=1 bash scripts/download_minimax_music3.sh` 重跑。

---

## 2. 模型事实

### 2.1 架构

| 组件 | 参数量 | 说明 |
| --- | --- | --- |
| Global LLM | 8B | 从 Qwen3-8B 初始化，负责长程结构，预测第 1 层 RVQ codebook |
| Local LLM | 0.6B | 逐帧预测其余 7 层声学 codebook |
| Flow Matching | 2.4B | 融合两个 LLM 的 hidden states 后合成连续表征 |
| Flow-VAE Decoder | 123M | 继承 MiniMax Speech 架构，出波形 |
| **合计** | **~11.1B** | |

Tokenizer 为 8 层 RVQ：语义 codebook 16,384 entries，7 个声学 codebook 各 1,024。
**推理时不走离散 tokenizer decoder**，而是用融合的连续 hidden states 合成——
官方把音质增益归因于此。

Planner 每音频秒吐 **25 帧**。一首 2:30 的歌 = **3750 次串行的 8B 前向**。
这是 §3.2 那个时延数字的根因，属于架构决定，优化不掉。

### 2.2 输入输出与硬约束

| 项 | 值 |
| --- | --- |
| 输入 | `prompt`（风格描述）+ `lyrics`（带 `[verse]`/`[chorus]` 等段落标记）+ `audio_duration` |
| 输出 | **44.1 kHz 立体声** |
| 时长上限 | 9000 帧 / 25 fps = **360 s** |
| prompt 上限 | 5000 token |
| 流式 | **不支持** |
| 设备 | **仅 CUDA** |
| 参考音频 / 音色克隆 | **不支持**（encoder 未释出，见 §4.3） |

> **文档错误更正**：README 写输出 32 kHz 16-bit stereo，但 `vocoder/config.json`
> 写 `sampling_rate: 44100`，Sogni 的实测也是 44.1 kHz。**以 44.1 kHz 为准**，
> README 那句是错的。（ACE-Step 是 48 kHz。）

### 2.3 官方推理路径

**diffusers（我们要用的）**——依赖**未合并的 PR** [huggingface/diffusers#14456](https://github.com/huggingface/diffusers/pull/14456)，必须钉 commit：

```bash
pip install git+https://github.com/huggingface/diffusers@dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d \
            transformers accelerate soundfile
```

```python
import soundfile as sf, torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("/nfs-models/wuhanjisuan894/models/MiniMax-Music3")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")
audio = pipe(prompt=prompt, lyrics=lyrics, audio_duration=60.0,
             generator=torch.Generator("cuda").manual_seed(7), output="audios")[0]
sf.write("song.wav", audio.T.float().cpu().numpy(), pipe.sampling_rate)
```

⚠️ PR 未合并意味着上游 API 还会变，我们的移植要跟着改。这是接入的一个持续成本。

**SGLang-Omni**：`sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3`，
暴露 `/v1/audio/speech`（歌词进 `input`，风格描述进 `instructions`）。
官方部署方案是 **2 卡**：GPU0 跑 Qwen3 + 8 层 RVQ 自回归，GPU1 跑 Flow Matching + DAV 解码。

**ComfyUI**：day-one 支持，需 ComfyUI ≥ 0.33.0。提供 int8 权重
（`minimax_music3_dit_int8_convrot.safetensors` /
`minimax_music3_text_encoder_pruned_int8_convrot.safetensors`）+ tiled decode 走低显存。

### 2.4 显存

| 来源 | 配置 | 显存 |
| --- | --- | --- |
| 官方 README | 满精度 | < 24 GB |
| 官方 README | `ComponentsManager.enable_auto_cpu_offload` | ~22 GB |
| 官方 README | `apply_group_offloading(pipe.language_model, offload_type="leaf_level")` | 8 GB 可跑 |
| 第三方 | RTX 6000 48G，无 batch 无优化 | 27 GB |
| Sogni | RTX 5090，int8 planner | ~11 GB |
| 日本实测 | RTX 4090，FP16 DiT + INT8 编码器 + tiled decode | ~14 GB |

显存不是瓶颈；**时延才是**。

### 2.5 授权

**`MiniMax-Music3 COMMUNITY LICENSE`**（Copyright © 2026 MiniMax），商用三条硬要求：

1. 必须在使用本软件的**商业产品或服务的用户界面上显著展示 "MiniMax-Music3"**；
2. 你及关联方由此产生的**年度总收入超过 2000 万美元**时，须事先取得 MiniMax
   书面授权（`api@minimax.io`，主题 "MiniMax-Music3 licensing - authorization request"）；
3. 只要**向第三方提供可生成输出的产品/服务/托管服务**，就必须在上线前及运营全程
   "实施、维护、测试并定期审查"合理且相称的技术与组织保障措施，以防止和缓解侵犯
   第三方知识产权的访问、使用与输出；不得停用、实质削弱或允许规避这些措施；
   并对下游接收方负责。

**ACE-Step 1.5 = MIT**，无附加条件。

> 我们卖的是 API 网关，第 1 条和第 3 条直接落到自己头上。第 3 条不是签字了事——
> 它要求一套可审计、持续运行的机制。这是产品和法务的事，工程吞不掉。

---

## 3. 与 ACE-Step 1.5 的横向对比

### 3.1 能力面：Music3 是真子集

| 能力 | ACE-Step 1.5 | Music3 |
| --- | --- | --- |
| 文生音乐 t2m | ✅ | ✅ |
| 参考音频条件输入 | ✅ | ❌ |
| Cover 翻唱生成 | ✅ | ❌ |
| Repaint 局部重绘 | ✅ | ❌ |
| 音轨分离 (stem) | ✅ | ❌ |
| 多轨叠加 (Add Layer) | ✅ | ❌ |
| Vocal2BGM 人声配伴奏 | ✅ | ❌ |
| 音频理解（抽 BPM/调式/caption） | ✅ | ❌ |
| LRC 歌词时间戳 | ✅ | ❌ |
| LoRA 个性化 | ✅（8 首歌 / 3090 上 1 小时） | ❌ |
| 元数据控制（BPM/key/拍号/时长） | ✅ | 部分（写在 caption 里，非硬控） |
| 时长 | 10 s – **600 s** | ≤ **360 s** |
| 时长精度 | **精确命中** | **上限而非目标**（见 §3.4） |
| 批量 | 最多 8 | 未提供 |
| 流式 | — | 不支持 |
| 采样率 | 48 kHz | 44.1 kHz |

**这一格的关键**：我们门面里 `_MUSIC_TASK_TYPES = {"t2m", "cover", "repaint"}` 三个
task_type 已经为 ACE-Step 开好，new-api 也实现了 `materializeMusicInputs`。
接 Music3 只能填 `t2m` 一格，另外两格它**结构性做不了**（§4.3）。

### 3.2 性能与成本

| 配置 | 2:30 的歌 | 峰值显存 | RTF |
| --- | --- | --- | --- |
| **ACE-Step xl-turbo · 我们的 A100 实测** | 240 s 曲 45 s / 600 s 曲 100 s | 26.5 GB | **0.17–0.19** |
| ACE-Step XL Turbo · RTX 5090（Sogni） | ~25 s（热态） | ~20 GB | 0.17 |
| Music3 int8 planner · RTX 5090（Sogni） | **~4–5 min**（实测 201–376 s） | ~11 GB | **1.6–2.1** |
| Music3 · RTX 4090（Sogni 推算） | ~6.5–8.5 min | ~14 GB | 2.6–3.4 |
| Music3 · RTX 4090（日本实测，FP16 DiT + INT8 编码器 + tiled decode，30 步 CFG 1.7） | 180 s 曲 219 s | — | **1.22** |

> 两份 4090 数据差得多（219 s vs 6.5–8.5 min），配置不同（int8 planner vs FP16 DiT +
> int8 text encoder + tiled decode）。保守取值：**Music3 RTF 落在 1.2–2.1 区间，
> ACE-Step 0.17，差 7–12 倍。**

我们自己的 ACE-Step A100 完整数据（`ACE-Step-1.5/docs/acestep-a100-实验测试报告.md`）：

| 时长 | 生成（热） | 峰值显存 | RTF |
| --- | --- | --- | --- |
| 60 s | 20 s | 23.8 G | 0.33 |
| 120 s | 35 s | 24.8 G | 0.29 |
| 240 s | 45 s | 24.9 G | 0.19 |
| 480 s | 85 s | 25.8 G | 0.18 |
| 600 s | 100 s | 26.5 G | 0.17 |

显存不随时长涨；单卡 1 副本，4×A100 节点 4 副本；每实例 ~6 请求/min。

**成本**（Sogni 按自家算力计价）：

| 长度 | Music3 | ACE-Step XL Turbo |
| --- | --- | --- |
| 60 s | ~$0.26 | ~$0.011 |
| 2:30 | ~$0.64 | ~$0.03 |
| 5:00 | ~$1.28 | — |

**约 23× 溢价**。除算力本身外，还因为 Music3 要占 24 GB+ 的机器档位，ACE-Step 20 GB 就够。

### 3.3 音质：Music3 赢，这点社区比较一致

- Sogni 结论：Music3 在**混音干净度、人声可信度、结构连贯性**上胜出，
  "more believable vocals in four languages"
- 社区反复提到的优点：乐器分离更清楚、混音/母带更真实、鼓和贝斯更好
- HF 讨论区：「输出比 ace-step 干净」「Good enough for open source」，
  质量大致对标 **Suno v3.5**
- 一句被反复引用的社区总结：**"AceStep is ahead in creativity, MiniMax has better sound quality."**

**Music3 强项曲风**：pop、cinematic、orchestral、piano、classical、**中文音乐**
**Music3 弱项曲风**：**metal、rock、EDM、experimental**

**ACE-Step 保住的优势**：创意度、非常规曲式、语言覆盖广度、prompt 控制精度、
编辑能力（cover / repaint / 分段重做）。

人声评价有分歧：有人觉得 Music3 人声更顺、更不合成感；也有人觉得**情绪偏平、节奏偏死**，
有评测原话是 "composition is solid but dynamic emotion is lacking"。

ACE-Step 一侧的自称：介于 Suno v4.5–v5；第三方报道其 SongEval 8.09 / AudioBox 7.42，
称在 SongEval 上超过 Suno v5。**两边的自称都不能直接对比**（评测方法完全不同），
也**没有任何一方发布过两者的同题 benchmark 分数**。

### 3.4 时长行为的差异（会影响产品与计费）

Sogni 六个 brief 全部要求 2:30：

| | brief 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| ACE-Step | 2:30 | 2:30 | 2:30 | 2:30 | 2:30 | 2:30 |
| Music3 | 2:31 | 2:52 | 2:06 | 2:11 | 3:00 | 2:14 |

**Music3 把 duration 当上限，自己决定什么时候结束并写结尾。**

两个衍生问题：

- **计费**：Sogni 的做法是按**请求时长**计费（因为模型可能提前结束）。
  我们 new-api 按 duration 计费的逻辑正好对得上，但"我要 3 分钟给了我 2:06"
  是要提前对用户说清的产品行为。
- **纯器乐坑**：如果段落标记只写 `[instrumental]`，会从 3 分钟上限直接掉到 ~20 秒。
  修法是用空段落标记 + 在 caption 里写明时长。

### 3.5 定位结论

Music3 **不是** ACE-Step 的替代，而是**高端档**：

> 贵 ~20 倍、慢 7–12 倍、少一半功能，换更好的成品音质。

Sogni 自己的落地方式就是把 Music3 作为 premium tier 与 ACE-Step 并存，
并明确 ACE-Step Turbo 仍然 "exceptionally competitive"。

选择判据（Sogni 版，我们认可）：

- **选 ACE-Step**：要速度、要精确时长、要控制、要编辑、要批量出稿
- **选 Music3**：歌本身就是交付物、成品保真度是第一优先级

---

## 4. 第三方证据清单

> 全部为第三方来源，我们**没有在自己的机器上验证过任何一条 Music3 的数据**。

### 4.1 Sogni Labs 正面对比（2026-08-14）— 目前唯一的同题对拍

**测试设置**：RTX 5090 + RTX 4090，comfy-worker 1.0.170，24 次渲染，
6 个匹配 brief + 12 曲风展示；**只取第一次生成**，唯一重跑规则是成品不足两分钟。

- ACE-Step XL Turbo recipe：8 步 / CFG 1 / shift 3 / euler-simple
- Music3 recipe：30 步 / CFG 1.7 / euler-simple，planner CFG 1.7，top-k 50，int8 planner

**6 个 brief**：

1. Anthemic pop-rock 女主唱，122 BPM E minor — 考爆发人声真实度、逐句歌词贴合、副歌抬升
2. Lo-fi hip-hop 器乐，78 BPM D♭ major — 考 Rhodes 音色、黑胶噪、律动
3. Cinematic orchestral trailer，100 BPM D minor — 考固定音型到全奏的动态、铜管弦乐真实度
4. Late-night jazz ballad 男声，62 BPM B♭ major — 考句法、刷鼓摇摆、房间感
5. Festival progressive house，126 BPM A minor — 考瞬态冲击、超低频、第二次 drop
6. Boom-bap rap verse，92 BPM G minor — 考卡点、密集歌词咬字、ad-lib 位置

**12 曲风展示（仅 Music3，RTX 5090，首次生成）**：
Mandopop/中文 68 BPM 2:13 · Reggaeton/西语 2:05 · City pop/日语 2:33 · Gospel 2:11 ·
Country duet 2:15 · A cappella 2:29 · Melodic metal 2:45 · Slow blues 2:21 ·
Synthwave 3:01 · Bossa nova × liquid D&B 2:55 · Solo piano nocturne 2:51 ·
"Paper Planes" 整曲 2:53（渲染耗时 5:26）

英/中/西/日四语均首次成功。

**重要工程细节**：Music3 一个 seed 同时控制 planner 和 sampler，所以可复现；
提高 top-k 增加多样性但不提高质量；caption 要写成 Global Metadata / Vocal Details /
Arrangement 三段式结构化描述，而不是标签堆砌。

### 4.2 HuggingFace 讨论区（`MiniMaxAI/MiniMax-Music3/discussions/4`）

- `rafiislam`：音质像 Suno V3.5
- `Alastar-Smith`：对开源模型来说相当好了，这个领域基本只有它和 ACE 两个选择
- `AekDevDev`：Good enough for open source，**输出比 ace-step 干净**
- `dummy9996`：反对"只有两个模型"的说法，认为 Stable Audio 3 比 ACE 好，
  ACE 在要 dubstep 时只会给廉价 ambient 和 pop
- `bghira`：**核心批评，见 §4.3**

讨论区**没有任何人报速度或显存数字**，也没有语言相关的观察。

### 4.3 结构性缺陷：encoder 未释出

`bghira` 在 HF 讨论区指出：与 ACE-Step 不同，**MiniMax 没有释出完整的 encoder**，
因此参考音频输入、音色克隆都不支持，用他的原话——
**"pretty much anything but t2a is just out of the question"**。

这解释了 §3.1 那张表里为什么 Music3 一整列都是 ❌：**不是没做，是权重没放**。
意味着 `cover` / `repaint` 两个 task_type 永远填不上，等版本更新也没用。

### 4.4 中文能力：证据打架，这是唯一没有定论的一格

| 来源 | 结论 |
| --- | --- |
| Sogni（2026-08-14） | 社区共识把**中文音乐**列为 Music3 最强项之一；12 曲展示里 Mandopop 首次生成即过 |
| MiniMax 官方博客 | 主推中文，demo 含沪语爵士、国语抒情、国语对唱，甚至能指定"台湾腔演唱" |
| MindStudio 评测 | 英文人声扎实，但**多语言输出偏弱** |

两份第三方评测直接对冲。**互联网上没有人能替我们回答这个问题。**

而我们自己的 ACE-Step 实验报告里 `t2m-vocal-zh` 那一行**至今是空的**
（`ACE-Step-1.5/docs/acestep-a100-实验测试报告.md` §5 P4 任务面整表未填）。
也就是说：我们线上跑着一个中文演唱质量从未验证过的音乐模型。

> **这才是真正该先做的事**：不是"要不要接 Music3"，而是"**ACE-Step 的中文唱得行不行**"。
> 它支持 8 首歌 1 小时训一个 LoRA，如果中文咬字有问题，先试试能不能自己修。

### 4.5 没找到的东西（负面结果也是结论）

- **没有**任何一方发布的同题 benchmark 分数（SongEval / AudioBox / FAD 等）
- **没有**知乎/B 站等中文社区的第一手实测对比
- **没有**任何 A100（sm_80）上的 Music3 性能数据——所有实测都在消费级 Ada/Blackwell 卡上
- **没有** Reddit（r/LocalLLaMA、r/StableDiffusion）的相关讨论进入搜索结果
- MiniMax 官方博客、ComfyUI 官方博客、模型卡，**三处全都不给任何速度数字**

---

## 5. 对我们集群特有的风险

### 5.1 ⚠️ int8 加速核需要 cu130+（**2026-08-23 更正：不适用于我们实际的接入路径，见下方加粗段**）

> **2026-08-23 更正**：本节的 int8/cu130 结论针对的是 ComfyUI 社区 int8 量化权重
> 和 Sogni 的测试路径。我们实际接入的是 **upstream vLLM-Omni 的原生移植版本**
> （`vllm_omni/model_executor/models/minimax_music3/`），其部署配置全程
> `dtype: bfloat16`（stage 0）/ `dtype: float32`（stage 1），**代码和部署配置里
> 完全没有 int8/fp8**，本节的 4× 罚单**不适用**。cu128 完全够用。详见
> `docs/vLLM-Omni-Upstream同步执行方案与开发约定-2026-08-23.md` §4。

Sogni 明确记录了这个坑：他们早期的 int8 数据差 **4 倍**，根因是**量化 kernel 在
CUDA-12 构建上退化回 eager PyTorch**，需要 **cu130+**；启动日志会打
`backend cuda: disabled` 作为特征。他们的做法是把 Music3 单独放到 CUDA-13 镜像轨的
24 GB+ worker 上，ACE-Step 留在 20 GB worker。

**我们的情况**（`ACE-Step-1.5/docs/acestep-a100-实验测试报告.md` 抬头）：

> 驱动 570.86.10 / CUDA 12.8 → **锁 cu128**（cu130 需驱动 ≥580，本机跑不了）

也就是说：**Sogni 那个"4–5 分钟"是我们吃不到的最优值**。我们要么吃这个 4× 罚单
（2:30 的歌 → 16–20 分钟），要么走 bf16 路径（更吃显存，速度另测）。

这是我们独有的、别人报告里不会有的额外折扣，**接入前必须实测确认落在哪一档**。
升驱动是整个 60 卡集群的事，不是为一个模型能做的决定。

### 5.2 门面时延参数是按 ACE-Step 秒级标定的

| 位置 | 常量 | 当前值 | Music3 的问题 |
| --- | --- | --- | --- |
| `gpustack/routes/videos.py:487` | `_DEFAULT_MUSIC_LATENCY` | 30 s | 差一到两个数量级 |
| `gpustack/config/config.py:199` | `lightx2v_music_max_queue_wait_seconds` | 90 s | 排队直接判超时 |
| `gpustack/server/video_progress.py:42` | music 阶段权重 | prepare 5 / encode 10 / **denoise 70** / decode 10 / save 5 | Music3 大头在 AR 解码不在 denoise，进度条会失真 |

前两个已有「按模型可配置队列等待上限」的机制（commit `e1efdc1a`）可挂；
第三个要么加一套 music-ar 权重表，要么在 planner 循环里报 `step/total_steps`。

### 5.3 输出格式

门面 `_output_ext` 对 music kind **硬编码 `.mp3`**（`videos.py:511`），
而 vllm-omni 现在音频一律出 wav。要么引擎侧转码，要么改门面按引擎给，
否则 `nfs_path` 的后缀与实际内容对不上。

### 5.4 diffusers 版本地狱

Music3 要钉未合并 PR 的 commit `dafe373`，H3 有自己的 diffusers 版本要求。
黑盒包 diffusers 等于把上游版本冲突搬进我们的 ARM 镜像。
**两个模型要两个 diffusers 版本时怎么办，目前没有答案**——这是 §6.2 那条路线
唯一的新增系统性风险，也是"diffusers 工程化"最该先解决的问题。

---

## 6. 若将来要接：技术路线（已调研完，可直接执行）

### 6.1 GPUStack 侧：音乐通道整条已经铺通了（**2026-08-23 更正：接入形态假设有误，见下方加粗段**）

> **2026-08-23 更正**：本节假设 Music3 会走 ACE-Step 那条 `t2m/cover/repaint`
> 异步 music 任务路径。实际 upstream 把 Music3 做成了 `tts_adapters/minimax_music3.py`，
> **走 `/v1/audio/speech` 的 TTS 风格接口**（`input`=歌词，`instructions`=风格描述），
> 不是本节设想的 `/v1/tasks/music/` 端点。gpustack 侧要怎么接这种"task_type 语义是
> music、但引擎侧是同步 speech 接口"的模型，需要重新查一遍 gpustack 的路由逻辑，
> 本节下方的具体路径（新增 `/v1/tasks/music/` 端点等）**不要照抄**。详见
> `docs/vLLM-Omni-Upstream同步执行方案与开发约定-2026-08-23.md` §4。

好消息是 ACE-Step 已经把路修完，Music3 不用新建通道：

| 层 | 现状 | 位置 |
| --- | --- | --- |
| task_type 白名单 | `t2m / cover / repaint` 已在 | `gpustack/routes/videos.py:244` |
| engine kind 路由 | → `music` → `POST /v1/tasks/music/` | `videos.py:493-503` |
| 输出扩展名 | `.mp3` | `videos.py:511` |
| 进度权重表 | music 档已有 | `server/video_progress.py:42` |
| 品类 | `CategoryEnum.MUSIC` | `schemas/models.py:52` |
| vLLM-Omni backend | 已内嵌（`BackendEnum.VLLM_OMNI`） | `worker/backends/vllm_omni.py` |
| vLLM-Omni 品类判定 | hints 表，默认 TEXT_TO_SPEECH | `scheduler/scheduler.py:724` |
| new-api task_type | `t2m/cover/repaint` 已在 | `relay/channel/task/gpustackplus/adaptor.go:82` |
| new-api 输入物化 | `materializeMusicInputs`（t2m 是空操作） | `adaptor.go:1370` |

而 vLLM-Omni 已有 `/v1/tasks/{audio,audiogen,image,video}/` 四个 submit 端点，
**唯独缺 `music`**（`vllm_omni/entrypoints/openai/api_server.py:4099` 是 audiogen 那份，约 40 行的模板）。

**推荐路径**：Music3 做成 vllm-omni 的 pipeline，引擎侧补一个 `/v1/tasks/music/` submit
（复用现有 `AUDIO_TASK_MANAGER` 和全局 status/cancel/queue 端点），沿用 `t2m` task_type。

三仓改动量：

| 仓 | 改动 |
| --- | --- |
| **vllm-omni** | ① pipeline 接入（主要工作量）② 抄 audiogen 那 40 行加 `/v1/tasks/music/` ③ 输出落 mp3 |
| **gpustack** | `scheduler.py:724` 的 `_VLLM_OMNI_CATEGORY_HINTS` 加一行 `("minimax-music3", CategoryEnum.MUSIC)`；加 model-catalog 条目 + 图标。**门面零改** |
| **new-api** | `adaptor.go:811` 的 t2m 推断现在只认 `acestep` 系名字，加规则；`MusicModelConfig` 加模型（只开 t2m tab） |

**不建议**：塞进 ACE-Step 引擎仓复用它的镜像/backend（架构完全不同，且与
"路线基于 vllm-omni"的 2026-08-09 定稿冲突）；或新起一个 `MINIMAX_MUSIC3` backend
（P2/P3 全套重来，没有收益）。

### 6.2 diffusers 工程化：结论是"扩 adapter，不新建引擎"

调研中确认：**vLLM-Omni 上游已经有一个黑盒 diffusers 适配层**。

`vllm_omni/diffusion/models/diffusers_adapter/pipeline_diffusers_adapter.py`，docstring 原话：

> Provides a black-box wrapper around any 🤗 Diffusers pipeline, enabling
> vLLM-Omni to directly serve Diffusers models with **near-zero per-model code**.

`load_format == "diffusers"` 即切进去（`diffusion/model_loader/diffusers_loader.py:684`），
整个去噪循环交给 diffusers 的 `__call__()`。`pipeline_utils.py` 留了 per-pipeline 钩子
registry（Wan 系已用于加载期注入 `boundary_ratio` / `flow_shift`）。

**它明确放弃的东西**（源码注释写死）：

```text
It does NOT support:
- CFG parallel
- Sequence parallel (requires model-specific attention surgery)
- TeaCache / Cache-DiT (requires hooking into transformer blocks)
- Step-wise execution (continuous batching)
```

外加 `supports_request_batch = False`。这四条恰好是 vLLM-Omni 相对裸 diffusers 的**全部**价值。

**选路判据**：

> **大头在 DiT、要多卡 / 量化 / Cache-DiT → 原生移植；
> 大头在 AR 解码、或小到单卡够 → diffusers adapter。**

按这条判：

- **Music3 是 adapter 路线的完美候选**。计算大头是 8B planner 的自回归解码
  （25 fps 串行），DiT 只有 2.4B。adapter 放弃的四样对它都不构成损失——
  sequence parallel 切不了 AR 解码，Cache-DiT 省的也不是它的大头。
- **H3 是反例**。62 GB DiT、要 INT8、要 TP/SP 多卡、要 Cache-DiT，四条全用得上，
  必须原生。它现在的状态是对的。

**前置改动**：Music3 是 `ModularPipeline`，而 adapter 现在硬编码
`DiffusionPipeline.from_pretrained()`，接不住。扩这个能力**不是 Music3 专用的**——
diffusers 的 modular 体系是新模型的主流出货方式，扩一次所有 modular 模型受益。

**H3 反过来证明了 adapter 值钱**：`docs/MiniMax-H3-官方Diffusers流程对齐开发与验收指南.md`
（806 行）是 H3 原生移植**做完之后**回头对齐官方契约的任务书，挖出 6 个 P0 级差异——
请求级 RNG 流、FL2VA 双关键帧几何、Ref2VA 图片语义、有损中间转码、音频重采样次数、
默认值包络。这些的共同特点是**不出片看不出来**。原生移植的隐性成本主要不在"写不出来"，
在"写出来跟官方不是一个东西"。adapter 天然零契约漂移——那是同一份代码在跑。

**正确姿势是两段式，不是二选一**：

1. 新模型先用 adapter 上线，拿正确性和 time-to-first-run；
2. 只有当时延/显存/QPS 不达标，**且 profiler 证明瓶颈在 DiT 上**，才做原生移植换并行和 cache，
   并拿 adapter 那条路当 oracle 做逐阶段 parity。

（H3 是反着做的——先移植后对齐，那 806 行任务书就是这个顺序的账单。）

**"diffusers 工程化"真正值得投的五件事**：

1. adapter 支持 `ModularPipeline`（通用能力，Music3 的前置）
2. **每模型的 diffusers 版本钉法与隔离**（§5.4，最痛且没答案的一条）
3. adapter 路径的显存档位 / offload 模板（Music3 官方三档正好可做 profile）
4. adapter 输出接上已有的 `/v1/tasks/*` + 进度上报（adapter 是黑盒 `__call__()`，
   拿不到步数，进度条现在会是死的）
5. 把 H3 那套阶段级 `.pt` fixture 对拍的 parity harness 抽成通用工具，
   而不是 H3 专用

**省不掉的部分**：权重下载、NFS、ARM 镜像、GPUStack backend 注册、门面 task_type、
`/v1/tasks/music/` 路由、mp3 落盘、new-api 规则、时延标定——一分不省。
diffusers 工程化能把"接一个新模型"从两周压到几天，但压不到零。

### 6.3 已知的部署前置

`--deploy-config` 对未注册模型静默失效：Music3 进 `OMNI_PIPELINES` 之前，
对着这个权重目录传 `--deploy-config` 会被整个丢掉并回落单卡默认档
（H3 踩过的同款坑），不要靠 deploy-config 来配显存。

---

## 7. 未验证清单（真要接时必须先做）

**权重已在 NFS 上，这些都是半天内能拿到的数。**

| # | 要验证什么 | 为什么重要 | 判据 |
| --- | --- | --- | --- |
| 1 | **中文咬字/唱腔盲测**，与 ACE-Step xl-turbo 同歌词同风格，每边 10 条双盲 | §4.4 第三方证据打架，是唯一没定论的一格 | Music3 需**明显优于** ACE-Step，否则连高端档的立足点都没有 |
| 2 | **cu128 上的实际耗时**（60 s / 240 s 曲） | §5.1，决定成本是 20× 还是 80× | 落在哪一档决定定价可行性 |
| 3 | **单卡能否跑完**（A100 40G） | 官方 SGLang 是 2 卡切分；若单卡不行，节点密度腰斩 | 单卡跑不完则成本再翻倍 |
| 4 | ACE-Step 的中文演唱基线 | 我们自己从没测过（§4.4） | 这条**独立于 Music3**，无论如何都该补 |
| 5 | 5 分钟长曲的结构连贯性 | Music3 宣称的强项 | 与 ACE-Step 600 s 曲对比 |
| 6 | 实际输出格式与采样率 | 确认 44.1 kHz（README 写 32 kHz 是错的） | 影响 mp3 转码链路 |
| 7 | 产物与 `assets/minimax_ttm.wav` 对拍 | 确认我们跑的路径与官方一致 | 后续判断移植有没有跑偏的唯一基线 |

**判死线（先定后测，避免测完再找理由）**：

- 第 1 项不显著胜出 → **直接判死**，能力面子集 + 授权负担 + 成本三条全占
- 第 2 项吃到 4× 罚单且第 1 项只是小胜 → 判死
- 第 3 项需要 2 卡 → 门槛再抬一档

---

## 8. 重启决策的触发条件

出现下列任一情况，值得重开这个议题：

1. **产品侧出现了"精品档"需求**——愿意为单曲付 ~20 倍成本、等 3–7 分钟、
   且不需要 cover/repaint 的用户档位（这是最可能的触发条件，因为技术结论已经清楚：
   Music3 是高端档而非替代档）
2. **ACE-Step 中文演唱被证明不达标**，且换 LM / 调 `vocal_language` / 训中文 LoRA 都修不好
3. **MiniMax 补发 encoder 权重**——那样 cover/repaint 才有可能，能力面对比会重写
4. **集群驱动升到 ≥580**（能上 cu130），§5.1 那个 4× 折扣消失
5. **diffusers PR #14456 合并进正式版**，版本钉死的持续成本消失
6. 出现**同题 benchmark**（SongEval / FAD 等）把两个模型放在一张表上

---

## 9. 复现命令

```bash
# ---- 权重（已完成，重跑会跳过已完整文件）----
tmux new -s dl_mm3 -d 'bash /root/download_minimax_music3.sh'
tail -f /tmp/dl_minimax_music3.log
VERIFY_SHA=1 bash scripts/download_minimax_music3.sh     # 需要 SHA-256 复核时

# ---- 官方 diffusers 路径（GPU 机，独立环境，别污染 vllm-omni 主环境）----
conda create -n mm3 python=3.12 -y && conda activate mm3
pip install git+https://github.com/huggingface/diffusers@dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d \
            transformers accelerate soundfile
# ModularPipeline.from_pretrained('/nfs-models/wuhanjisuan894/models/MiniMax-Music3')

# ---- ACE-Step 对照基线（已有 harness）----
REG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com
IMAGE="$REG/reputationly/acestep:arm64-a100-latest" \
  CKPT_DIR=/nfs-models/wuhanjisuan894/vllm-omni-speech/ACE-Step-1.5 \
  GPUS='"device=0"' CONFIG_PATH=acestep-v15-xl-turbo LM_MODEL=acestep-5Hz-lm-4B \
  DURATIONS="60 240" STEPS="8" BATCHES="2" \
  bash /root/stress_acestep.sh
```

测试纪律（照搬 `ACE-Step-1.5/docs/acestep-a100-实验测试报告.md`）：
热态稳态（连发丢首张取均值）；安静宿主（`docker rm -f` 清场再测）；
容器 `--memory=240g`；单容器复用扫压测；tmux 里跑长任务。

---

## 10. 一页速查

| 维度 | ACE-Step 1.5 | MiniMax-Music3 |
| --- | --- | --- |
| 授权 | **MIT** | 社区许可（UI 挂标 + $20M 门槛 + 托管方保障义务） |
| 能力面 | t2m + cover + repaint + stem + 多轨 + Vocal2BGM + 音频理解 + LRC + LoRA | **仅 t2m**（encoder 未释出） |
| 时长 | 10 s – 600 s，**精确命中** | ≤ 360 s，**上限而非目标** |
| 采样率 | 48 kHz | 44.1 kHz（README 写 32 kHz 是错的） |
| A100 实测 | 240 s 曲 45 s / 600 s 曲 100 s，26.5 G，RTF 0.17–0.19 | **无任何 A100 数据** |
| 消费卡实测 | 2:30 曲 ~25 s（5090） | 2:30 曲 4–5 min（5090，int8） |
| RTF | 0.17 | 1.2–2.1（且我们可能吃 4× 罚单） |
| 相对成本 | 1× | **~23×** |
| 音质 | 创意度、曲式多样、控制精度 | **混音干净度、人声可信度、结构连贯性** |
| 强项曲风 | 广（50+ 语言） | pop / cinematic / orchestral / piano / classical / 中文 |
| 弱项曲风 | 硬核曲风被质疑（社区个例） | **metal / rock / EDM / experimental** |
| 中文 | **我们从没测过** | 第三方证据打架 |
| GPUStack 现状 | 已内嵌全链路 | 零（但通道已铺好，见 §6.1） |
| 我们的部署 | 单卡 1 副本，4×A100 = 4 副本，~6 请求/min/实例 | 未知（官方 SGLang 是 2 卡） |

---

## 附录 A：参考索引

### 本仓 / 关联仓文档

| 内容 | 位置 |
| --- | --- |
| 下载脚本 | `scripts/download_minimax_music3.sh` |
| ACE-Step A100 实验测试报告（我们自己的实测） | `ACE-Step-1.5/docs/acestep-a100-实验测试报告.md` |
| ACE-Step GPUStack 异步门面 | `ACE-Step-1.5/acestep/api/http/tasks_facade_{routes,service}.py` |
| 新引擎内嵌 GPUStack 工程化方法论 | `gpustack/docs/新引擎内嵌gpustack-工程化方法论.md` |
| H3 官方 Diffusers 流程对齐任务书（parity 方法论） | `docs/MiniMax-H3-官方Diffusers流程对齐开发与验收指南.md` |
| Ideogram-4 接入调研（同类文档范式） | `docs/Ideogram-4-接入调研与路线选型报告.md` |

### 关键代码位置

| 内容 | 位置 |
| --- | --- |
| diffusers 黑盒适配层 | `vllm_omni/diffusion/models/diffusers_adapter/pipeline_diffusers_adapter.py` |
| adapter 选路 | `vllm_omni/diffusion/model_loader/diffusers_loader.py:684` |
| adapter per-pipeline 钩子 | `vllm_omni/diffusion/models/diffusers_adapter/pipeline_utils.py` |
| audiogen submit 端点（music 版的模板） | `vllm_omni/entrypoints/openai/api_server.py:4099` |
| 门面 music task_type / engine kind / 扩展名 | `gpustack/routes/videos.py:244,493,511` |
| 门面 music 时延与队列参数 | `gpustack/routes/videos.py:487` + `gpustack/config/config.py:199` |
| 进度阶段权重表 | `gpustack/server/video_progress.py:42` |
| vLLM-Omni 品类 hints | `gpustack/scheduler/scheduler.py:724` |
| ACE-Step selector（单卡 + 30 GiB 下限） | `gpustack/policies/candidate_selectors/acestep_resource_fit_selector.py` |
| new-api 音乐 task_type 与推断 | `new-api/relay/channel/task/gpustackplus/adaptor.go:82,811,1370` |

**外部来源**（括号内为查阅日期 2026-08-16 时的状态）

| 来源 | 用于本文哪一节 |
| --- | --- |
| [Sogni Labs — MiniMax Music 3 vs. ACE-Step 1.5 XL](https://blog.sogni.ai/blogs/minimax-music3-vs-ace-step-15-xl/)（2026-08-14，**唯一同题对拍**） | §3.2 §3.3 §3.4 §4.1 §5.1 |
| [HuggingFace 讨论区 #4](https://huggingface.co/MiniMaxAI/MiniMax-Music3/discussions/4) | §4.2 §4.3 |
| [MiniMax 官方博客](https://www.minimax.io/blog/minimax-music-3-0-next-generation-open-weights-production-ready-versatile-music-model)（零 benchmark） | §2.1 §4.4 |
| [ComfyUI 官方博客](https://blog.comfy.org/p/minimax-music-3-state-of-the-art)（零数字） | §2.2 |
| [ComfyUI 官方教程（int8 权重 + tiled decode）](https://docs.comfy.org/tutorials/audio/minimax/minimax-music-3) | §2.3 §2.4 |
| [MindStudio 评测（称多语言偏弱）](https://www.mindstudio.ai/blog/minimax-music-3-quality-review) | §4.4 |
| [StudioYebisu — 4090 实测 219 s](https://note.com/studio_yebisu/n/na7cada34077b) | §3.2 |
| [MiniMaxAI/MiniMax-Music3 模型卡](https://huggingface.co/MiniMaxAI/MiniMax-Music3) | §1 §2 |
| [ACE-Step 1.5 GitHub](https://github.com/ace-step/ACE-Step-1.5) | §3.1 §3.3 |
| [ACE-Step 1.5 项目页](https://ace-step.github.io/ace-step-v1.5.github.io/) | §3.3 |
