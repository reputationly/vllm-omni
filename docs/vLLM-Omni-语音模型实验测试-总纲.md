# vLLM-Omni 语音/音频模型 实验测试总纲

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0020(16-20 备用)
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(base `vllm/vllm-openai:v0.25.0` 官方 arm64,torch 2.11 cu130,含 sm_80)
> 权重根:`/nfs-models/wuhanjisuan894/vllm-omni-speech/`(挂容器同路径,`HF_HUB_OFFLINE=1` 离线)
> harness:`scripts/smoke/tts_bench.sh`(打 `/v1/audio/speech`:smoke/len/conc)
> 日期:2026-07-18 起(滚动填充)
>
> 方法论照搬 `LightX2V/docs/Wan2.2-*-实验测试报告.md` 与 `ACE-Step-1.5/docs/acestep-a100-实验测试报告.md`:
> **结论先行 → 权重 → P1 功能/配置 → P2 长度压测 → P3 采样/流式 → P4 任务面 → P5 并发吞吐(sizing)→ P6 崩溃边界 → 复现 → 速查**。

---

## 0. 目标

对 vllm-omni 支持、已下载的每个语音/音频模型,搞清 **4 件事**,每个模型出一份独立报告:
1. **功能** —— 支持哪些能力(预设/克隆/情感/多语/方言/多说话人/唱段/音效),实际效果。
2. **硬件匹配** —— A100-40G 单卡装不装得下,峰值显存,冷启动,采样率。
3. **调优** —— `max_num_seqs`、流式、采样参数、是否量化(A100 上量化只省显存不提速,一般不动)。
4. **压测** —— 长度崩溃边界、并发吞吐、RTF、每卡副本数(sizing)。

## 1. 测试纪律(血泪铁律,照搬)

1. **热态稳态**:每档连发 ≥3 次,**丢首(含加载/JIT 虚高)取均值**。冷态只用于记录加载时长。
2. **安静宿主**:测性能前 `docker rm -f` 清场,邻居容器读 NFS 权重会拖慢 2-4×。
3. 容器 `--memory=240g`(cgroup 护宿主);`--gpus '"device=N"'` 绑卡。
4. **单容器复用扫压测**:加载 ~几分钟/次(Qwen3-TTS ~5.5min),别每档重起;`tts_bench.sh len/conc` 打同一个已起容器。
5. **产物防呆**:wav 大小 > 几十 KB、`file` 是 WAVE、时长对齐;空/静音判失败。harness 用 Python 内置 `wave` 读时长(**零外部依赖,不需装 ffmpeg**);非 WAV 产物才回退 `ffprobe`。
6. tmux 里跑长压测;不覆盖 NFS 上正在执行的脚本。

## 2. 通用指标(每模型报告统一列)

| 指标 | 定义 |
|---|---|
| 冷启动加载 | 容器起到 `/ready` 200 的墙钟(含 torch.compile/CUDA graph/warmup) |
| 单请求 RTF | `生成墙钟(热) / 产物音频秒数`;<1 快于实时 |
| 峰值显存 | 生成期间 `nvidia-smi` 采样的该卡显存峰值(harness 自动采) |
| 并发吞吐 | 并发 N 下 条/s 或 条/min;`max_num_seqs` 最优点 |
| 崩溃边界 | 超长文本多少字杀引擎(容器 DOWN)→ facade 前置硬闸依据 |
| 采样率 | 产物 WAV 采样率(24k/44.1k/48k) |
| 音色一致性 | 同角色连出 20-50 句有无漂(短剧量产关键,人工听) |

## 3. 模型优先级与分组

| Tier | 模型 | 子目录 | 特点 | 额外依赖 |
|---|---|---|---|---|
| **1 配音主力** | **Qwen3-TTS** | `Qwen3-TTS-1.7B-CustomVoice` | 预设音色库+多语+情感(instructions) | — |
| 1 | **VoxCPM2** | `VoxCPM2` | 48k 高保真,可预设可克隆 | — |
| 1 | **CosyVoice3** | `Fun-CosyVoice3-0.5B-2512` | 0.5B 极轻量,克隆需 ref_audio+ref_text | — |
| **2 MOSS 家族** | MOSS-TTS-Nano | `MOSS-TTS-Nano` | 0.1B/48k,需 ref_audio | `MOSS_TTS_CODEC_PATH`=…/MOSS-Audio-Tokenizer-Nano |
| 2 | MOSS-TTS-Realtime | `MOSS-TTS-Realtime` | 低延迟流式 TTFB | `MOSS_TTS_CODEC_PATH`=…/MOSS-Audio-Tokenizer |
| 2 | MOSS-VoiceGenerator | `MOSS-VoiceGenerator` | 文字造声线(音色设计) | 同上 codec |
| 2 | MOSS-TTSD | `MOSS-TTSD-v1.0` | 8B 多说话人对话 | 同上 codec |
| **3 特色/其他** | Ming-omni-tts | `Ming-omni-tts-0.5B` | 方言/44.1k,`--enforce-eager` | — |
| 3 | GLM-TTS | `GLM-TTS` | 克隆需 ref_audio+ref_text | — |
| 3 | SoulX-Singer | `SoulX-Singer` | 唱段 SVS/SVC,`/v1/chat/completions` | Preprocess 权重 + 手动 phone_set.json |
| 3 | Stable-Audio-Open | `stable-audio-open-1.0` | 音乐/音效,`/v1/audio/generate` | gated(已下) |

**执行顺序**:Tier1(先测透、立报告模板)→ Tier2(MOSS,先补 codec)→ Tier3。**建议起点 = Qwen3-TTS**(serve 已验证,用它定报告模板),再 VoxCPM2(48k A/B)。

## 3.1 机群并行分配(0016-0020,20 卡)

全部单卡 → 一次性铺开并行测,不用串行等。分配(每台 4 卡):

| 机器 | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| **0016** 配音主力 | Qwen3-TTS | VoxCPM2 | CosyVoice3 | Ming-omni-tts(`--enforce-eager`) |
| **0017** MOSS 家族(全加 codec env) | MOSS-Nano(-Nano codec) | MOSS-Realtime | MOSS-VoiceGenerator | MOSS-TTSD |
| **0018** 其他 | GLM-TTS | Stable-Audio(`/v1/audio/generate`) | SoulX-Singer(需 phone_set) | (机动) |
| **0019** 压测/一致性 | Qwen3-TTS 副本(并发压测) | VoxCPM2 副本 | (机动) | (机动) |
| **0020** | 现 Qwen3-TTS 在跑(继续用) | (机动) | (机动) | (机动) |

**并行纪律**:
1. **每台内错峰启动 ≥2min/实例** —— 4 个模型同时冷启动会抢 NFS 冷读(T5/权重),方法论实测 3×并发冷读打进 18min 病态。起完一个等 `/ready` 200 + `MemAvailable≥15G` 再起下一个。
2. **前置**:0016-0019 需载镜像 —— `docker load < /nfs-models/_transfer/vllm-omni-arm64-a100.tar`(prepare-transfer 已生成)或直连 ACR `docker pull`;确认 NFS 挂载(权重)。**bench 无外部依赖**(时长走 Python `wave`,不用装 ffmpeg)。
3. **压测隔离**:功能/长度测在功能实例上跑;**并发压测放 0019/0020 的副本上**,别在功能实例上打并发污染显存峰值读数(安静宿主原则)。
4. 各机 tmux 挂 bench,结果回传汇总。

## 4. 每模型报告模板(P1–P6,照 ACE-Step)

每模型新建 `docs/实验报告/vLLM-Omni-<模型>-实验测试报告.md`,含:

- **0. 结论先行**:生产默认配置 / 每卡副本数 / 关键指标 / 判死踩坑(3-5 条)。
- **1. 环境与权重**:镜像 tag、子目录、附属依赖、serve 命令。
- **P1 功能/配置面**:预设音色 / 克隆 / 情感 / 多语 各发一版,记 gen/RTF/峰值显存/采样率/听感。
- **P2 长度压测**:`tts_bench.sh len` —— 字数扫,崩溃边界 + RTF/显存 vs 长度。
- **P3 采样/流式**:温度/topk 稳定性;流式 vs 非流式 TTFB;`max_num_seqs` 扫。
- **P4 任务面**:该模型特有能力(方言/多说话人/唱段/音效…)。
- **P5 并发/吞吐(sizing)**:`tts_bench.sh conc` —— 并发扫,QPS/尾延迟 → 每卡副本数建议。
- **P6 崩溃边界**:超长 / 空 / 非法输入 → 门面前置校验清单。
- **复现命令** + **一页速查**。

## 5. 复现(通用)

```bash
REG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly
IMG=$REG/vllm-omni:arm64-a100-latest
ROOT=/nfs-models/wuhanjisuan894/vllm-omni-speech

# 起服务(MOSS 系加 -e MOSS_TTS_CODEC_PATH=$ROOT/MOSS-Audio-Tokenizer[-Nano];Ming 加 --enforce-eager)
docker run -d --name omni-<m> --gpus '"device=0"' --memory=240g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $ROOT:$ROOT -p 8091:8091 \
  "$IMG" vllm serve "$ROOT/<子目录>" --omni --trust-remote-code --port 8091
until curl -sf localhost:8091/ready >/dev/null; do sleep 5; done; echo READY

# 压测(bench 无外部依赖,时长走 Python wave)
PORT=8091 VOICE=vivian CONTAINER=omni-<m> GPU_ID=0 bash scripts/smoke/tts_bench.sh smoke
PORT=8091 VOICE=vivian CONTAINER=omni-<m> GPU_ID=0 bash scripts/smoke/tts_bench.sh len
PORT=8091 VOICE=vivian CONTAINER=omni-<m> GPU_ID=0 CONC="1 2 4 8" bash scripts/smoke/tts_bench.sh conc
```

## 6. 进度追踪

| 模型 | 功能 | 长度压测 | 并发 | 崩溃边界 | 报告 | 状态 |
|---|---|---|---|---|---|---|
| Qwen3-TTS | 部分(冒烟+情感+长句已过) | ⬜ | ⬜ | 部分(960字未崩) | ⬜ | 进行中 |
| VoxCPM2 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 待测 |
| CosyVoice3 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 待测 |
| MOSS-Nano/Realtime/VoiceGen/TTSD | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 待补 codec |
| Ming-omni-tts | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 待测 |
| GLM-TTS | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 待测 |
| SoulX-Singer | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 待补 phone_set |
| Stable-Audio-Open | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 待测(/v1/audio/generate) |

---

*配套:`vLLM-Omni-全模型镜像构建与开测手册.md`(镜像/serve)、`vLLM-Omni-语音模型全景与选型.md`(选型)、`vLLM-Omni-大模型量化与Offload可行性调研.md`(不量化结论)。*
