# vLLM-Omni 全模型通用镜像:构建 / 分发 / 开测手册

> 日期:2026-07-17 · 环境:鲲鹏 920 ARM64 + 4×A100 PCIE 40G(sm_80,无 NVLink)· 测试机 16-20 号
> 目标:一个**通用 vllm-omni 镜像**跑全部语音/音频模型(`vllm serve <model> --omni`),不再一模型一镜像。
> 配套:`IndexTTS2-vLLM-Omni-实验测试报告.md`(单模型实测模板)、`vLLM-Omni-语音模型全景与选型.md`(选型)。

---

## 0. 核心结论:镜像本质是"官方 base + 装源码"

`docker/Dockerfile.cuda` 只有两件事:`FROM vllm/vllm-openai` + `uv pip install .`(整个 vllm-omni 源码)。
所以**一个镜像 = 一个 vllm-omni commit,天然覆盖该 commit 仓库里的全部模型**。不存在"某模型没打进去"的问题,只有"commit 够不够新认得这个模型架构"。

IndexTTS-2 报告里那个 `reputationly/vllm-omni:arm64-a100-20260714` 就是这么来的:
base `vllm/vllm-openai:v0.25.0`(官方 arm64)+ vllm-omni **main@62589203**(= 当前 HEAD)。
→ **它已经是全模型镜像**,和 `indextts2:arm64-a100-latest`(GPUStack 生产引擎包)是两回事。

**两条路二选一:**
- **复用**:直接 pull `vllm-omni:arm64-a100-20260714` 到 16-20,当前 commit 无新增模型就够用。
- **重打(本手册主线,便于以后自持)**:同一条命令出一个新日期 tag,流程如下。

---

## 1. 硬约束与预检(构建期确认,避免上机白跑)

| 项 | 要求 | 验证 |
|---|---|---|
| base 架构 | `vllm/vllm-openai:v0.25.0` 官方带 **arm64** manifest | `docker manifest inspect vllm/vllm-openai:v0.25.0 \| grep arm64` |
| GPU kernel | torch + vllm `.so` 含 **sm_80** | 见 §3 预检命令 |
| base tag | **必须覆盖默认的 v0.24.0** → 用 v0.25.0 | Dockerfile ARG 默认 stale,构建时 `--build-arg` 显式给 |
| aarch64 依赖 | 无 pynini(前端走 cn2an/g2p) | IndexTTS 已验证无碍 |

> FlashMLA 仅 sm_100 —— 与 TTS 无关,A100 上不用,忽略其告警。

---

## 2. 构建

照 index-tts / LightX2V 的模式,**vLLM-Omni 更省一段**:LightX2V 要自建 CUDA+torch base(PyPI aarch64 torch 是 CPU-only)+ `sync-base-to-dockerhub`;vLLM-Omni 直接用**官方 `vllm/vllm-openai:v0.25.0`** 当 base(自带 arm64 + sm_80 预编译 kernel,且本来就在 Docker Hub)→ **无需自建 base、无需 sync**,只 build app 层。

### 2.1 主路径:CI 出包(推荐)

`.github/workflows/build-arm64.yml`(已建,照 index-tts 的 `build-arm64.yml`):GitHub 原生 `ubuntu-24.04-arm` runner 从 Docker Hub 拉官方 base → build app 层 → 推 ACR(immutable `arm64-a100-<time>-<sha>` + floating `arm64-a100-latest`)→ 建 Release。

```bash
gh workflow run build-arm64.yml --ref main       # 或 GitHub Actions 页面点 Run
```

前置:在 `reputationly/vllm-omni` repo Settings → Secrets 配 `ACR_USERNAME` / `ACR_PASSWORD`(与 gpustack / index-tts fork 同名)。

### 2.2 后备:Apple Silicon Mac 本地构建

CI 不可用时(如仅本地调试),Mac 原生 arm64 直连 dockerhub 拉 base:

```bash
cd /Users/reputationly/Desktop/code/api/vllm-omni    # 仓库根,确认在目标 commit
git rev-parse HEAD                                    # 记下 commit,进 tag 便于追溯

REG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly
TAG=$REG/vllm-omni:arm64-a100-$(date +%Y%m%d)

docker build \
  --platform linux/arm64 \
  -f docker/Dockerfile.cuda \
  --build-arg BASE_IMAGE=vllm/vllm-openai:v0.25.0 \
  -t "$TAG" .

docker push "$TAG"
```

> 为什么在 Mac 不在节点:base 在 dockerhub,Mac 能直连;隔离网节点未必能拉 dockerhub。
> 若某台 A100 节点能直连 dockerhub,也可直接在节点上 build(省一次 push/pull)。

## 2b. 分发到隔离/内网节点(照 lx2v 的 tar 搬运法)

节点拉不到 ACR 时,走 NFS tar(与 `lx2v-node.sh prepare-transfer` 同思路):

```bash
# 在能连 ACR 的机器(如 238)拉 arm64 镜像并存 NFS tar(纯搬运,架构随意)
docker pull --platform linux/arm64 "$TAG"
docker save "$TAG" > /nfs-models/_transfer/vllm-omni-arm64-$(date +%Y%m%d).tar

# 在 16-20 每台
docker load < /nfs-models/_transfer/vllm-omni-arm64-<date>.tar
```

---

## 3. 上机前预检(在目标节点,1 分钟)

```bash
IMG=<你的镜像 tag>
# ① toolkit 能用(--gpus)
docker run --rm --gpus all "$IMG" nvidia-smi | head -15
# ② sm_80 kernel 覆盖 + 版本
docker run --rm --gpus all "$IMG" python -c "
import torch, vllm, vllm_omni
print('torch', torch.__version__, '| arch', torch.cuda.get_arch_list())
print('vllm', vllm.__version__)
"
# 期望:arch 列表含 'sm_80';vllm 0.25.x
```

toolkit 缺(`--gpus` 报错)→ 才需要 `lx2v-node.sh setup-base`(只配 docker/NFS/toolkit,**不入集群**);其余情况**不要跑 install**。

---

## 4. 通用启动模板 + 每模型清单

统一调用:`vllm serve <本地权重路径> --omni --trust-remote-code --port <p> [--deploy-config <yaml>]`,OpenAI 兼容 `POST /v1/audio/speech`。

```bash
IMG=<镜像 tag>
ROOT=/nfs-models/wuhanjisuan894/vllm-omni-speech
run() {  # run <子目录> <端口> [额外参数...]
  local sub=$1 port=$2; shift 2
  docker run -d --name omni-$sub --gpus '"device=0"' \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v $ROOT:$ROOT -p $port:$port \
    "$IMG" vllm serve "$ROOT/$sub" --omni --trust-remote-code --port $port "$@"
}
```

> `--gpus '"device=0"'` 单卡绑定;多实例分卡时换 device=1/2/3。拓扑:GPU0-1 一对、GPU2-3 一对,跨对(1↔2)走 NUMA 慢,多卡压同一对内。

### 4.1 逐模型 serve 命令(deploy-config 除注明外均自动加载)

> 大多数模型 deploy-config 按 model_type 自动加载,用上面的 `run <子目录> <端口>` 模板即可;
> 下表"额外参数"列是模板之外要补的 flag。端点默认 `POST /v1/audio/speech`,注明的除外。

| 子目录 | 模型 | ★ | 额外参数 | 端点 / 关键请求参数 | 输入要求 |
|---|---|---|---|---|---|
| `Qwen3-TTS-1.7B-CustomVoice` | Qwen3-TTS | ★1 | (无,自动加载 `qwen3_tts.yaml`) | `/v1/audio/speech`:`voice`=预设音色、`instructions`=风格 | **有预设音色**,无需 ref |
| `VoxCPM2` | VoxCPM2 | ★2 | (无) | `/v1/audio/speech`:`ref_audio` 可选 | 可预设可克隆;**48k** |
| `Fun-CosyVoice3-0.5B-2512` | CosyVoice3 | ★2 | (无) | `/v1/audio/speech`:`ref_audio`+`ref_text` | **克隆必需** ref_audio+ref_text |
| `Ming-omni-tts-0.5B` | Ming-omni-tts | ★3 | `--enforce-eager` | `/v1/audio/speech`:`instructions`(方言 JSON)、`ref_audio` | 音色嵌入,方言走 instructions;**44.1k** |
| `MOSS-TTS-Nano` | MOSS-Nano | ★3 | ⚠️需 codec(见 4.2) | `/v1/audio/speech`:`ref_audio` **必需** | ref_audio 必需;voice/ref_text 被忽略;**48k** |
| `MOSS-TTS-Realtime` | MOSS-Realtime | ★3 | ⚠️需 codec | `/v1/audio/speech`:`ref_audio` 必需,`stream=true` | 低延迟流式;24k |
| `GLM-TTS` | GLM-TTS | ★3 | (无) | `/v1/audio/speech`:`ref_audio`+`ref_text` **必需** | 克隆必需 ref_text;首请求慢(懒加载) |
| `MOSS-VoiceGenerator` | 音色设计 | ★4 | ⚠️需 codec | `/v1/audio/speech`:`input`=声线文字描述 | 无 ref;文字造声线 |
| `SoulX-Singer` | 歌声 SVS+SVC | ★4 | ⚠️见 4.3 | **`/v1/chat/completions`**:`input_audio`+`target_audio` | 缺依赖,见 4.3 |
| `stable-audio-open-1.0` | 音乐+音效 | ★5 | `--enforce-eager --gpu-memory-utilization 0.9`,**无 deploy-config** | **`/v1/audio/generate`**:`input`、`audio_length`(≤47s)、`guidance_scale` | 非 TTS;44.1k 立体声 |
| ~~`ACE-Step-v1-3.5B`~~ | 文生音乐 | ❌ | — | — | **本镜像不支持(见 4.4)** |

### 4.2 MOSS 三兄弟共享 codec(已补进下载脚本)

MOSS-Nano / Realtime / VoiceGenerator 起服务要一份 codec(上游默认从 HF 自动拉,`HF_HUB_OFFLINE=1` 下必须本地有)。已加进 `download_speech_models.sh` 默认集:
- `moss_codec` → `OpenMOSS-Team/MOSS-Audio-Tokenizer`(Realtime / VoiceGenerator 用)
- `moss_codec_nano` → `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`(Nano 用)

起服务时给容器传 `-e MOSS_TTS_CODEC_PATH=$ROOT/MOSS-Audio-Tokenizer`(Nano 用 `-Nano` 那个)。补下命令:
```bash
MODELS="moss_codec moss_codec_nano" bash download_speech_models.sh
```

### 4.3 SoulX-Singer 前处理权重(已补)+ 仍需手动 phone_set

1. **`SoulX-Singer-Preprocess` 权重** → 已加进下载脚本(`soulx_preprocess`),**下到 `$ROOT/SoulX-Singer-Preprocess`,与 `SoulX-Singer` 同级** —— 代码 `utils.py:172` 自动发现该同级目录,免配环境变量。
2. **`phone_set.json`**(HF 不发,仍需手动补进模型目录)。
3. 端点是 **`/v1/chat/completions`** 不是 audio/speech,且要显式 `--deploy-config vllm_omni/deploy/soulxsinger_svs.yaml`(SVC 用 `_svc.yaml`)+ `--enforce-eager`。

补下命令:`MODELS=soulx_preprocess bash download_speech_models.sh`

### 4.4 ACE-Step —— 独立处理,不在本镜像范围

当前 commit 的 vllm-omni **没有 ACE-Step 的 recipe / deploy-config / 模型实现**,本镜像 `vllm serve` 认不出。**由独立流程处理、最终嵌入 gpustack**,不在本手册/本镜像的开测范围。音乐/音效在本镜像内用已支持的 **Stable-Audio-Open** 兜底。

### 4.5 MOSS-TTSD(多说话人对话,对手戏一次成型)

- 权重:`MOSS-TTSD-v1.0/`(8B,`MossTTSDelayModel`,n_vq=16,24kHz)· deploy `moss_ttsd.yaml` 自动加载(2 stage 同卡:talker + codec decoder)。
- **bf16 原样跑,不量化**(~26G 单卡,40G 宽裕;A100 量化只省显存不提速,详见 `大模型量化与Offload可行性调研.md`)。
- **必需 codec**:`-e MOSS_TTS_CODEC_PATH=$ROOT/MOSS-Audio-Tokenizer`(复用已下的,非 Nano 版)。
- 端点 `/v1/audio/speech`;多说话人对话文本按 **MOSS-TTSD 官方 README 的说话人标记格式**(`[S1]`/`[S2]` 等)组织,`ref_audio` 做零样本克隆。deploy 默认 `max_num_seqs=4 / gpu_memory_utilization=0.85 / async_chunk`。

```bash
run() 用法(见 §4 模板)之外补 codec 环境变量:
docker run -d --name omni-moss-ttsd --gpus '"device=0"' \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e MOSS_TTS_CODEC_PATH=$ROOT/MOSS-Audio-Tokenizer \
  -v $ROOT:$ROOT -p 8098:8098 \
  "$IMG" vllm serve "$ROOT/MOSS-TTSD-v1.0" --omni --trust-remote-code --port 8098
```

> 短剧用法:对手戏/群聊一次生成,和"IndexTTS 逐句拼接"同脚本 A/B 听**衔接自然度 + 音色是否稳**。

---

## 5. 每模型必测 5 项(照 IndexTTS-2 模板)

1. **arm64 sm_80 kernel 覆盖** —— §3 预检已统一确认,单模型再看启动日志无 kernel 缺失告警。
2. **离线权重布局** —— 附属模型按各自本地查找名放进模型目录;`HF_HUB_OFFLINE=1` 起服务不联网。
3. **deploy yaml tokenizer 占位坑** —— 占位 tokenizer 改本地路径(IndexTTS 踩过)。
4. **崩溃边界** —— 超长文本/异常输入是否杀引擎 → 记录字数上限;facade 前置校验 + `--restart`。
5. **峰值显存 + 并发吞吐 + 长跑** —— 确认 40G 余量、`max_num_seqs` 最优点、无泄漏。

### 每模型记录一行

`模型 | 峰值显存 | 单请求RTF | 并发吞吐(max_num_seqs) | 崩溃边界(字数) | 长跑泄漏 | 采样率 | 在线端点`

---

## 6. 顺序建议

1. **§3 预检**(一台先做,确认 toolkit + sm_80 + 版本)
2. **Qwen3-TTS**(★1,补 IndexTTS 最多空白:多语言 + 音色库 + WS)先跑通全流程
3. 按 ★ 优先级逐个:VoxCPM2 / CosyVoice3 → Ming/MOSS 系 → SoulX-Singer(补 phone_set)→ 音效(Stable-Audio/ACE-Step)
4. 每个填 §5 记录行,汇总成各自的 mini 实测报告

---

## 7. Qwen3-TTS 开测三连(0020,复制即用)

先验证镜像+全链路的最省事路径(有预设音色,不用先备参考音频)。

### ① 上镜像(ACR 拉 或 NFS tar load)

```bash
REG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com
IMG=$REG/reputationly/vllm-omni:arm64-a100-latest
# 能连 ACR:
docker pull "$IMG"
# 隔离网走 tar(在 238/能连 ACR 的机器先 save 到 NFS,再在 0020):
#   docker load < /nfs-models/_transfer/vllm-omni-arm64-<date>.tar
```

### ② 起服务(GPU0,挂 NFS,离线)

```bash
ROOT=/nfs-models/wuhanjisuan894/vllm-omni-speech
docker run -d --name omni-qwen3tts --gpus '"device=0"' \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $ROOT:$ROOT -p 8091:8091 \
  "$IMG" vllm serve "$ROOT/Qwen3-TTS-1.7B-CustomVoice" --omni --trust-remote-code --port 8091

# 等就绪(轮询 health;首次加载 + torch.compile 可能 1-2 分钟)
until curl -sf http://localhost:8091/health >/dev/null; do echo "waiting..."; sleep 5; done; echo "READY"
# 看日志确认无 kernel/tokenizer 报错:
docker logs -f omni-qwen3tts   # Ctrl-C 退出
```

### ③ 冒烟测试(预设音色出一段 wav)

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"这是一次配音冒烟测试,声音清晰、语气自然。","voice":"vivian","language":"Chinese"}' \
  --output /tmp/qwen3tts_smoke.wav
ls -lh /tmp/qwen3tts_smoke.wav && file /tmp/qwen3tts_smoke.wav   # 有内容 + 是 WAV 即通
```

预设音色:`vivian / ryan / aiden / dylan / eric / serena / sohee / uncle_fu` 等(`GET /v1/audio/voices` 查全);情感/风格走 `instructions`,克隆走 `ref_audio`,音色库上传走 `POST /v1/audio/voices`。

**通过判据**:health 200 + 日志无 sm_80/tokenizer 报错 + wav 有声。这一步同时证明镜像本身 OK,后续模型照 §4 逐个换 `serve` 路径即可。

---

*第 4 节逐模型 serve 命令已落实(commit 62589203 全模型 recipe 齐备)。CLAP 评测用法由独立 agent 负责,不在本手册。*
