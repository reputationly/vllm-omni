# vLLM-Omni · Qwen3-TTS 实验测试报告

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0020
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 4b2def0b,base `vllm/vllm-openai:v0.25.0`,torch 2.11 cu130,sm_80)
> 模型:`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`,权重 `/nfs-models/wuhanjisuan894/vllm-omni-speech/Qwen3-TTS-1.7B-CustomVoice`
> harness:`scripts/smoke/tts_bench.sh`(smoke/len/conc,打 `/v1/audio/speech`)
> 日期:2026-07-18
> 方法论照 `ACE-Step-1.5/docs/acestep-a100-实验测试报告.md`:结论先行 → 功能 → 长度 → 采样 → 任务 → 并发 → 崩溃 → 复现 → 速查。

---

## 0. 结论先行

1. **生产就绪,单卡宽裕**:2 阶段(talker AR + code2wav),常驻 **~18G/40G**,近满上下文峰值 ~26G,**单卡 1 副本**,4×A100 节点 4 副本。
2. **RTF ~0.17**(短文本),比实时快 ~6×;冷启动 ~5.5min(torch.compile + 双 stage CUDA graph + warmup)。
3. **⚠️ 长文本单请求会"静默截断"**:输入 text token 与输出 audio token **共享 `max_model_len=4096`**,输入越长音频越短、话说不全,**且返 HTTP 200**(看着成功,实则丢数据)。→ **长文本必须 facade 句级切分,逐句合成再拼**。完整合成的输入上限 ≈ **200 字/请求**。
4. **崩溃边界友好**:输入 token 超 4096(~6400 字)→ **HTTP 400 优雅拒绝,不杀引擎**(对比 IndexTTS-2 超长直接杀引擎)。
5. **每卡吞吐**:conc16 → 1.33 条/s ≈ **80 条/min/卡**(批处理有收益,吞吐随并发亚线性上升)。
6. **功能全**:9 个预设音色 + `instructions` 文字情感(默认输出自带表现力)+ 600+ 语言 + 克隆(Base 变体)+ 音色库上传(`/v1/audio/voices`)。采样率 **24kHz**。

---

## 1. 环境与权重

- serve(单卡,离线):
```bash
docker run -d --name omni-qwen3tts --gpus '"device=0"' --memory=240g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $ROOT:$ROOT -p 8091:8091 \
  "$IMG" vllm serve "$ROOT/Qwen3-TTS-1.7B-CustomVoice" --omni --trust-remote-code --port 8091
```
- deploy `vllm_omni/deploy/qwen3_tts.yaml` 自动加载(2 stage 同卡 device 0)。启动关键:`max_model_len=4096`(stage0 talker)/ `65536`(stage1 code2wav);`generation_config` 默认 `temperature=0.9, top_k=50, top_p=1.0`。
- 冷启动:`AsyncOmniEngine initialized in 331s`(~5.5min)。就绪探针 `/ready`(200 = warmup 完成)。

---

## 2. P1 — 功能 / 配置面

| 能力 | 支持 | 用法 | 状态 |
|---|---|---|---|
| 预设音色(9) | ✅ | `voice`=`aiden/dylan/eric/ono_anna/ryan/serena/sohee/uncle_fu/vivian` | ✅ 冒烟通过 |
| 情感/风格 | ✅ | `instructions`(如"愤怒、语速快");默认输出即自带表现力 | ✅ 冒烟通过(愤怒指令生效) |
| 多语言 | ✅ 600+ | `language`(Chinese/English/Auto…) | ⬜ 待多语专测 |
| 克隆(zero-shot) | ✅ | Base 变体:`ref_audio`+`ref_text` | ⬜ 待克隆专测(本变体为 CustomVoice) |
| 音色库上传/管理 | ✅ | `POST /v1/audio/voices` / `GET` / `DELETE` | ⬜ 待测 |
| 流式 | ✅ | `stream=true`,`stream_format="audio"` PCM / SSE;WebSocket `/v1/audio/speech/stream` | ⬜ 待测 TTFB |

采样率:**24kHz**(480044 B / 10.0s ≈ 48KB/s = 24k·16bit·mono)。

---

## 3. P2 — 长度压测(单请求,vivian,多样中文文本)

> 关键发现:**输入 text + 输出 audio 共享 max_model_len=4096**,长输入吃光音频预算 → 音频截断。
> `cps`(字/audio秒)正常 ~4-5;**>6 = 截断**(话没说完)。

| 字数 | http | 生成(热) | audio | cps 字/秒 | RTF | 峰值显存 | 判读 |
|---|---|---|---|---|---|---|---|
| 50 | 200 | 1.98s | 11.2s | 4.5 | 0.177 | 18165 | ✅ 完整 |
| 100 | 200 | 4.25s | 24.8s | 4.0 | 0.171 | 18165 | ✅ 完整 |
| 200 | 200 | 8.09s | 48.0s | 4.2 | 0.169 | 18165 | ✅ 完整(**完整上限附近**) |
| 3200 | 200 | 27.97s | 146.9s | 21.8 | 0.190 | 25773 | ⚠️ 截断 |
| 4000 | 200 | 19.39s | 101.7s | 39 | 0.191 | 25841 | ⚠️ 截断 |
| 4800 | 200 | 9.87s | 56.7s | 85 | 0.174 | 25995 | ⚠️ 严重截断 |
| 5200 | 200 | 7.56s | 34.1s | 153 | 0.222 | 26081 | ⚠️ 几乎没说 |
| 5600 | 200 | 3.51s | 11.7s | 479 | 0.301 | 26175 | ⚠️ 基本空 |
| 6400 | **400** | 0.05s | — | — | — | 18241 | ✅ **优雅拒绝**(超 max_model_len,不崩) |
| 12800 | 400 | 0.05s | — | — | — | — | ✅ 优雅拒绝 |

**结论**:
- **完整合成上限 ≈ 200 字/请求**(cps~4.5);200~6400 字 = **静默截断(HTTP 200 但音频不全)**;>~6400 字 = **HTTP 400 优雅拒绝**。
- **显存**:短文本平 18.2G;近满上下文(3200~5600 字)升到 ~26G(+8G KV),仍 < 40G。**长文本无显存瓶颈,瓶颈是 token 预算**。
- **运营刚需**:facade **句级切分 + 长度硬闸**(单句 ≤~200 字),逐句合成再 pyloudnorm 拼接。这是 Qwen3-TTS 上生产的前置条件。

---

## 4. P3 — 采样 / 流式 / max_num_seqs

| 项 | 现状 | 待测 |
|---|---|---|
| 默认采样 | `temperature=0.9, top_k=50, top_p=1.0`(generation_config) | 稳定性(同文本多次差异)、降温对一致性影响 |
| 流式 TTFB | 支持 `stream_format=audio`(PCM) / SSE / WS | ⬜ 首包延迟 |
| max_num_seqs | deploy 默认(见 qwen3_tts.yaml) | ⬜ 扫最优点 vs 吞吐 |

---

## 5. P4 — 任务面

| 任务 | 状态 |
|---|---|
| 预设音色 TTS | ✅ |
| 情感 TTS(instructions) | ✅(默认即有表现力) |
| 多语言 | ⬜ |
| 克隆(需切 Base 变体) | ⬜ |
| 音色库上传复用 | ⬜ |

---

## 6. P5 — 并发 / 吞吐(sizing)

> 100 字/请求(24.8s 音频),并发提交 N。

| 并发 | 总时长 | 吞吐(条/s) | 均摊(s/条) |
|---|---|---|---|
| 1 | 4.38s | 0.23 | 4.38 |
| 2 | 4.96s | 0.40 | 2.48 |
| 4 | 6.19s | 0.65 | 1.55 |
| 8 | 8.29s | 0.97 | 1.04 |
| 16 | 12.03s | **1.33** | 0.75 |

**结论**:吞吐随并发**亚线性上升**(批处理收益);conc16 = 1.33 条/s ≈ **80 条/min/卡**(100 字请求)。等效音频吞吐 conc16 ≈ 16×24.8s/12s ≈ **33s 音频/s(聚合 RTF ~0.03)**。**提吞吐靠多卡多副本 + GPUStack 分发,不是单实例加 batch**。

---

## 7. P6 — 崩溃边界(门面前置校验)

| 输入 | 实测 | facade 动作 |
|---|---|---|
| 输入 token > 4096(~6400 字) | **HTTP 400 秒拒,容器 up** | 前置长度硬闸(按 token 估),不必依赖引擎 |
| 200~6400 字 | **HTTP 200 但音频静默截断** ⚠️ | **句级切分**(单句 ≤~200 字)——这是比崩溃更隐蔽的坑 |
| 空 input | ⬜ | ⬜ 待测 |
| 非法 voice | ⬜ | ⬜ 待测 |

---

## 8. 复现命令

```bash
REG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly
IMG=$REG/vllm-omni:arm64-a100-latest
ROOT=/nfs-models/wuhanjisuan894/vllm-omni-speech
# serve(见 §1)后:
PORT=8091 VOICE=vivian CONTAINER=omni-qwen3tts GPU_ID=0 bash /nfs-models/_transfer/tts_bench.sh smoke
PORT=8091 VOICE=vivian CONTAINER=omni-qwen3tts GPU_ID=0 LENS="50 100 200 400 800 1600 3200 6400" bash /nfs-models/_transfer/tts_bench.sh len
PORT=8091 VOICE=vivian CONTAINER=omni-qwen3tts GPU_ID=0 CONC="1 2 4 8 16" bash /nfs-models/_transfer/tts_bench.sh conc
```

---

## 9. 一页速查

| 维度 | 结论 |
|---|---|
| 生产配置 | 单卡 1 副本;4×A100 节点 4 副本;24kHz |
| 显存 | idle 18G / 峰值 26G(近满上下文),均 <40G |
| RTF(短文本) | ~0.17,快 6× |
| 冷启动 | ~5.5min(torch.compile+双 stage) |
| 完整合成上限 | **~200 字/请求**(超了静默截断)→ **facade 句级切分刚需** |
| 崩溃边界 | ~6400 字 HTTP 400 优雅拒绝,不杀引擎 |
| 每卡吞吐 | ~80 条/min(100 字);聚合 RTF ~0.03 |
| 功能 | 9 预设音色 + instructions 情感 + 600+语言 + 克隆 + 音色库 |
| 踩坑 | ⚠️ 长文本 HTTP 200 静默截断(比崩溃更隐蔽);采样默认 temp0.9 有波动 |
| 待补 | 多语/克隆/音色库/流式 TTFB/max_num_seqs 扫/空&非法输入 |
