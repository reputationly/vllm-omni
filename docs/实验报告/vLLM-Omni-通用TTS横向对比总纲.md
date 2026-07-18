# vLLM-Omni · 通用 TTS 横向对比总纲

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0016/0017/0020
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 59add8a0,base `vllm/vllm-openai:v0.25.0`,torch 2.11 cu130,sm_80)
> 端点:统一 `POST /v1/audio/speech`;harness:`scripts/smoke/tts_bench.sh`(smoke/len/conc)
> 日期:2026-07-18
> 本文汇总以下单模型报告的横向对比,细节见各自报告:
> [Qwen3-TTS](vLLM-Omni-Qwen3-TTS-实验测试报告.md) · [IndexTTS-2](../IndexTTS2-vLLM-Omni-实验测试报告.md) · [VoxCPM2](vLLM-Omni-VoxCPM2-实验测试报告.md) · [MOSS-Nano](vLLM-Omni-MOSS-TTS-Nano-实验测试报告.md) · [CosyVoice3](vLLM-Omni-CosyVoice3-实验测试报告.md) · [GLM-TTS](vLLM-Omni-GLM-TTS-实验测试报告.md) · [Ming(失败)](vLLM-Omni-Ming-omni-tts-实验测试报告.md)

---

## 0. 一句话结论

- **6 个可用 TTS**:Qwen3-TTS(唯一带**预设音色**)+ **IndexTTS-2**(**情感控制最强**、已定案替换旧引擎)+ VoxCPM2 / MOSS-Nano / CosyVoice3 / GLM-TTS(**纯零样本克隆**)。**Ming-omni-tts 启动即崩(代码 bug),不可用**。
- **两条部署路线**:①**低显存多副本堆吞吐**(MOSS-Nano 1-4.5G、VoxCPM2 13.4G);②**高显存单副本质量向**(CosyVoice3 26.5G、GLM 29G、IndexTTS-2 24.6G、Qwen3-TTS 18-26G)。
- **全员通病**:长文本**共享 token 预算 → 输入越长音频越被截断**。多数是 **HTTP 200 静默截断**(音频封顶、话没说完);**⚠️ IndexTTS-2 最恶劣 —— 超 ~216 字触发 CUDA assert 直接杀死整个引擎、容器退出且不自愈**(需 `docker start` ~5min)。facade 必须句级切分 + 长度硬闸。
- **克隆是主流**,请求统一:`ref_audio`(`file://`/base64/URL)+ `ref_text` + serve 加 `--allowed-local-media-path`;IndexTTS-2 惯用 base64 data URL + 8 维情感向量。
- **采样率**:5 个 24kHz,**IndexTTS-2 唯一 22.05kHz**(下游混音/拼接要注意重采样)。

---

## 1. 主对比大表(全维度)

| 维度 | **Qwen3-TTS** | **IndexTTS-2** | **VoxCPM2** | **MOSS-TTS-Nano** | **CosyVoice3** | **GLM-TTS** | Ming-omni-tts |
|---|---|---|---|---|---|---|---|
| **可用性** | ✅ 生产就绪 | ✅ **生产就绪(已定案)** | ✅ 可用 | ✅ 可用 | ✅ 可用 | ✅ 可用 | ❌ **启动崩** |
| **音色方式** | **预设9 + 克隆 + 音色库** | 纯克隆 + **情感最强** | 纯克隆 | 纯克隆(+音色库) | 纯克隆(+SFT潜力) | 纯克隆(+音色库) | — |
| **端点** | `/v1/audio/speech` | 同 | 同 | 同 | 同 | 同 | — |
| **参数量** | 1.7B | GPT+S2Mel+BigVGAN | ~0.5B级 | Nano(极小) | 0.5B | 未知(显存大) | 0.5B |
| **架构** | 2阶段(talker AR+code2wav) | 2阶段(GPT AR+S2Mel扩散+BigVGAN) | AR 克隆 | AR 克隆(轻) | AR+flow(onnx) | AR 克隆 | Qwen2类(适配错) |
| **显存 idle** | ~18G | ~23.3G | **~13.4G** | **~1.0G** | ~21G | ~27.4G | — |
| **显存 peak** | ~26G(近满上下文) | ~24.6G(懒加载后恒定) | ~14.1G(几乎不涨) | **~4.5G** | ~26.5G | ~29G | — |
| **单卡副本数** | 1 | 1 | **2** | **8+** | 1 | 1 | — |
| **RTF(短文本)** | ~0.17 | ~0.37 | ~0.6 | **1.9~2.6(慢)** | 0.5~0.8 | 0.36~0.49 | — |
| **RTF(长文本)** | ~0.19 | 0.27(216字) | **0.12~0.21** | 0.24~0.27 | **1.18(200字,最慢)** | 0.40~0.44 | — |
| **冷启动** | ~5.5min | ~5.4min(BigVGAN compile) | 首条~34s | 首条快 | 首条~21s | 首条~54s | — |
| **完整合成上限** | ~200字 | ~216字 | **~1600字(~320s)** | **~1600字(~340s)** | **~200字(最短)** | ~200字 | — |
| **超长边界** | ~6400字 **HTTP 400 优雅拒绝** | ⚠️ **324字 CUDA assert 杀引擎(容器退出,不自愈)** | ~6400字 **HTTP 400** | **永不400,一律截断** | ≥400截断/≥800疑卡 | ≥400截断/≥800疑卡 | — |
| **并发扩展性** | conc16 **1.33条/s(有批收益)** | **max_num_seqs=4最优,conc8 5.2×实时,300句零失败** | 弱且抖(峰~1.5) | **恒0.04条/s(零收益,batch=1)** | 未测(慢) | 未测(慢) | — |
| **采样率** | 24kHz | ⚠️ **22.05kHz(唯一)** | 24kHz | 24kHz | 24kHz | 24kHz | — |
| **特殊依赖** | — | 附属overlay(w2v-bert/campplus/semantic_codec/bigvgan) | 镜像内 `voxcpm` | `HF_HOME`(codec离线) | 镜像内 `s3tokenizer` | — | — |
| **serve 必备 flag** | — | `--deploy-config`(tokenizer 本地路径) | `--allowed-local-media-path` | 同 + `HF_HOME` | 同 | 同 | — |

---

## 2. 分维度速览

### 2.1 显存 & 部署密度(单卡 A100 40G)

| 模型 | idle→peak | 单卡副本 | 4卡节点副本 | 路线 |
|---|---|---|---|---|
| MOSS-Nano | 1.0→4.5G | **8+** | **32+** | 多副本堆吞吐 |
| VoxCPM2 | 13.4→14.1G | **2** | 8 | 多副本 |
| Qwen3-TTS | 18→26G | 1 | 4 | 单副本(批处理) |
| CosyVoice3 | 21→26.5G | 1 | 4 | 单副本 质量向 |
| IndexTTS-2 | 23.3→24.6G | 1 | 4 | 单副本 情感向 |
| GLM-TTS | 27.4→29G | 1 | 4 | 单副本 质量向 |

### 2.2 速度(RTF,越低越快;<1 = 快于实时)

| 模型 | 短文本 | 长文本 | 评级 |
|---|---|---|---|
| VoxCPM2 | ~0.6 | **0.12~0.21** | ⭐⭐⭐ 最快 |
| Qwen3-TTS | ~0.17 | ~0.19 | ⭐⭐⭐ 快且稳 |
| IndexTTS-2 | ~0.37 | 0.27 | ⭐⭐ 快 |
| GLM-TTS | 0.36(warm) | 0.40~0.44 | ⭐⭐ 可用 |
| MOSS-Nano | 1.9~2.6 | 0.24~0.27 | ⭐ 短文本慢 |
| CosyVoice3 | 0.5~0.8 | **1.18(慢于实时)** | ⭐ 最慢 |

### 2.3 上下文预算(完整合成不截断的输入上限)

| 模型 | 完整上限 | 最长音频 | 超限行为 |
|---|---|---|---|
| VoxCPM2 | **~1600字** | ~320s | ≥6400字 HTTP 400(优雅) |
| MOSS-Nano | **~1600字** | ~340s | 永不400,一律静默截断 |
| Qwen3-TTS | ~200字 | ~48s | ~6400字 HTTP 400(优雅) |
| CosyVoice3 | ~200字 | ~63s | ≥400截断,≥800疑卡 |
| GLM-TTS | ~200字 | ~50s | ≥400截断,≥800疑卡 |
| IndexTTS-2 | ~216字 | ~30s | ⚠️ **≥324字 CUDA assert 杀引擎(最危险)** |

### 2.4 并发/吞吐(单实例)

| 模型 | 并发行为 | 单实例吞吐 | 提吞吐手段 |
|---|---|---|---|
| IndexTTS-2 | **max_num_seqs=4最优(batch=8无收益)**,300句零失败 | conc8 5.2×实时 ≈ 0.9s/句 | 多卡多实例横向扩(非调batch) |
| Qwen3-TTS | 批处理有收益(亚线性) | conc16 1.33条/s | 多卡多副本 + 批 |
| VoxCPM2 | 弱且抖动 | 峰 ~1.5条/s | 多副本 |
| MOSS-Nano | **零收益,恒batch=1** | ~2.6条/min | **纯多副本**(显存小易堆) |
| CosyVoice3/GLM | 未测(RTF~1慢) | — | 多卡单副本 |

---

## 3. 请求格式(统一克隆四件套 + Qwen3 特例)

**克隆模型(VoxCPM2 / MOSS-Nano / CosyVoice3 / GLM)统一**:
```bash
curl -s -X POST localhost:$PORT/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "input":"要合成的文本",
  "ref_audio":"file:///nfs-models/.../_ref_zh.wav",
  "ref_text":"参考音频的准确转写",
  "response_format":"wav"}' --output out.wav
```
- `ref_audio` **必须** URL / base64 data URL / `file://` URI(裸路径 → 400)。
- serve **必须**加 `--allowed-local-media-path $ROOT`(否则 `Cannot load local files`)。
- MOSS-Nano 额外:`-e HF_HOME=$ROOT/hf_cache`(codec 离线加载)。

**Qwen3-TTS(唯一预设音色)**:
```bash
-d '{"input":"文本","voice":"vivian","instructions":"愤怒、语速快","language":"Chinese"}'
```
预设音色 `aiden/dylan/eric/ono_anna/ryan/serena/sohee/uncle_fu/vivian`;`instructions` 控情感。

**IndexTTS-2(情感控制最强)**:惯用 base64 data URL + 8 维情感向量:
```bash
-d '{"input":"你到底想干什么！","ref_audio":"data:audio/wav;base64,<b64>",
     "extra_params":{"emo_vector":[0,0.8,0,0,0,0,0,0],"emo_alpha":0.9}}'
```
`emo_vector` 8 维顺序 **喜怒哀惧厌郁惊平**;另支持 `emo_audio`(参考情感音频)、`use_emo_text`+`emo_text`(文字描述情感)三种模式。⚠️ 情感参数包在 `extra_params` 里(与旧 server.py 的 `emotion_vector` 命名不同)。

---

## 4. 全员通病 & 门面(facade)刚需

| 坑 | 表现 | 门面动作 |
|---|---|---|
| **⚠️ IndexTTS-2 超长杀引擎** | ≥~216 字触发 CUDA device-side assert → **整个引擎死、容器退出、不自愈**(≠ 其它模型的优雅 400/静默截断) | **硬闸挡在引擎前**(建议 ≤180 字/句)+ 容器 `--restart unless-stopped` 兜底 |
| **长文本静默截断(其它模型)** | 输入吃光 token 预算,音频封顶但 **HTTP 200**(看着成功) | **句级切分**:单句 ≤ 各模型完整上限;逐句合成再 pyloudnorm 拼接 |
| **cps 判截断** | 字/音频秒 >6 = 截断(正常 ~4-5) | 产物侧校验 cps,超阈告警/重试 |
| **克隆 ref 格式** | 裸路径 400 | 门面统一转 `file://` 或 base64,serve 固定挂 media flag |
| **采样率不一** | IndexTTS-2 22.05kHz,其余 24kHz | 拼接/混音前统一重采样 |
| **冷启动慢** | Qwen3 ~5.5min、IndexTTS-2 ~5.4min、GLM 首条 ~54s | 就绪探针 `/ready`,预热后再上量 |
| **MOSS-Nano codec** | 缺 HF_HOME 启动崩 | 容器固定挂 `HF_HOME=$ROOT/hf_cache` |
| **克隆不字节级复现** | seed 只锁 AR 阶段,扩散阶段仍随机(IndexTTS-2 实测 md5 不一致) | 需严格复现的场景不适用;配音可重 roll |

---

## 5. 选型建议(按场景)

| 场景 | 首选 | 理由 |
|---|---|---|
| **配音/短剧/强情感克隆** | **IndexTTS-2** | 情感控制最强(8维向量/情感音频/情感文本),并发稳(300句零失败),已定案 |
| **要指定/多预设音色、情感可控** | **Qwen3-TTS** | 唯一预设音色 + instructions 情感 + 音色库 |
| **长文本/长音频(有声书、旁白)** | **VoxCPM2** | ~1600字/320s 完整,RTF 最快,显存中等 |
| **高并发、海量短文本(通知、播报)** | **MOSS-Nano** | 显存 1-4.5G,单卡 8+ 副本,靠数量堆吞吐 |
| **克隆音质向、可接受慢/单副本** | **CosyVoice3 / GLM** | 克隆质量向;CosyVoice3 功能最全(SFT/跨语言潜力) |
| **不推荐** | ~~Ming-omni-tts~~ | 启动崩(`lm_head` 适配 bug),待改代码 |

> ⚠️ 选 IndexTTS-2 必须先做 facade 句级切分 + 长度硬闸(≤180字/句)+ 容器 `--restart`,否则超长请求会杀死整个引擎(唯一有此风险的模型)。

---

## 6. 待补(下一轮)

- **克隆音质主观/客观评测**(用 CLAP 打分做说话人相似度 + MOS 听感),现仅验通路,未评质量。
- CosyVoice3 **SFT/跨语言/指令情感**专测(功能最全,潜力最大)。
- 各模型 **音色库 `/v1/audio/voices`** 上传复用、**流式 TTFB**、**采样稳定性**。
- CosyVoice3/GLM **≥800 字边界复测**(确认是超时/400/退化)、**并发扫**。
- **Ming-omni-tts** 修 `lm_head`/model class 适配后重测。
- **特殊玩法模型**(非通用 TTS):MOSS-TTSD 对话、MOSS-VoiceGenerator 音色设计、SoulX-Singer 歌声、AudioX 文/视频生音频、Stable-Audio 已单独出报告。
