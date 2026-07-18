# vLLM-Omni · MOSS-TTS-Nano 实验测试报告(轻量零样本克隆 TTS)

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0017
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 59add8a0,base `vllm/vllm-openai:v0.25.0`,torch 2.11 cu130,sm_80)
> 模型:`MOSS-TTS-Nano`,权重 `/nfs-models/wuhanjisuan894/vllm-omni-speech/MOSS-TTS-Nano`;codec `MOSS-Audio-Tokenizer-Nano`(离线从 `hf_cache` 加载)
> harness:`scripts/smoke/tts_bench.sh`(smoke/len/conc,打 `/v1/audio/speech`)
> 日期:2026-07-18
> 方法论照 `ACE-Step-1.5/docs/acestep-a100-实验测试报告.md`:结论先行 → 功能 → 长度 → 采样 → 任务 → 并发 → 崩溃 → 复现 → 速查。

---

## 0. 结论先行

1. **纯克隆模型(无内置音色)**:必须 `ref_audio`(`file://`/URL/base64)+ `ref_text`;serve 需 `--allowed-local-media-path $ROOT`。
2. **显存极轻 —— 1.0~4.5G/40G**(随音频长度小幅上升),名副其实 "Nano" → **单卡可塞 8+ 副本**,密度冠军。
3. **codec 需离线预热**:codec 走 `AutoModel.from_pretrained("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano")`,离线容器必须挂 `-e HF_HOME=$ROOT/hf_cache`(cache 已预填),否则 `LocalEntryNotFoundError` 启动崩。
4. **速度偏慢**:短文本 **RTF 1.9~2.6(慢于实时)**,固定开销大;长文本摊薄到 **RTF 0.24~0.27**。绝对生成时间比 VoxCPM2 慢一档(1600 字 gen 90s vs VoxCPM2 43s)。
5. **完整合成上限 ≈ 1600 字/请求**(~340s 音频,cps~5);≥3200 字 = **静默截断(HTTP 200 但音频封顶 ~338s)**;**全程不 HTTP 400**(不像 VoxCPM2 超长会拒,Nano 一律接收后截断)。
6. **并发完全不 scale —— 恒 0.04 条/s**(等效 batch=1,请求全串行,每条 ~23s):conc 1→8 吞吐纹丝不动 → **提吞吐唯一手段是多副本**(好在显存极小,单卡堆副本毫无压力)。
7. 采样率 **24kHz**(WAV)。定位:**超轻量克隆 TTS**,适合高并发多副本部署(靠数量堆吞吐,不靠单实例)。

---

## 1. 环境与权重

- serve(单卡,离线,放行本地 ref,codec 走 cache):
```bash
docker run -d --name omni-moss-nano --gpus '"device=0"' --memory=240g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=$ROOT/hf_cache \
  -v $ROOT:$ROOT -p 8091:8091 \
  "$IMG" vllm serve "$ROOT/MOSS-TTS-Nano" --omni --trust-remote-code \
  --allowed-local-media-path "$ROOT" --port 8091
```
- `max_model_len=4096`(见 `/v1/models`)。
- **踩坑**:codec 用 HF-id 加载(不读 env `MOSS_TTS_CODEC_PATH`),离线必须 `HF_HOME` 指向预填了 `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` 的 cache。

---

## 2. P1 — 功能 / 配置面

| 能力 | 支持 | 用法 | 状态 |
|---|---|---|---|
| 预设音色 | ❌ | 无内置 speaker(400:`has no built-in speakers`) | — |
| 零样本克隆 | ✅ | `ref_audio`(file:///URL/base64)+ `ref_text` | ✅ 冒烟通过 |
| 音色库上传 | ✅(提示) | 400 报文提示 `POST /v1/audio/voices` | ⬜ 待测 |
| 情感/多语言/流式 | ⬜ | 待测 | ⬜ |

采样率:**24kHz**(960044 B / 10.0s ≈ 96KB/s?实为 24k·16bit·mono,含 WAV 头)。

---

## 3. P2 — 长度压测(单请求,ref=_ref_zh.wav,多样中文文本)

| 字数 | http | 生成(热) | audio | cps | RTF | 峰值显存 | 判读 |
|---|---|---|---|---|---|---|---|
| 50 | 200 | 28.67s | 13.68s | 3.7 | 2.096 | 1135 | ✅ 完整(慢) |
| 100 | 200 | 35.40s | 27.28s | 3.7 | 1.298 | 1171 | ✅ 完整 |
| 200 | 200 | 33.50s | 50.40s | 4.0 | 0.665 | 1337 | ✅ 完整 |
| 400 | 200 | 44.10s | 104.64s | 3.8 | 0.421 | 1691 | ✅ 完整 |
| 800 | 200 | 59.85s | 208.32s | 3.8 | 0.287 | 2413 | ✅ 完整 |
| 1600 | 200 | 90.50s | 340.72s | 4.7 | 0.266 | 3969 | ✅ 完整(**上限附近**) |
| 3200 | 200 | 80.95s | 338.80s | 9.4 | 0.239 | 3969 | ⚠️ 静默截断(音频封顶) |
| 6400 | 200 | 86.60s | 338.80s | 18.9 | 0.256 | 4453 | ⚠️ 静默截断(封顶不变) |

**结论**:
- **完整合成上限 ≈ 1600 字/请求(~340s 音频)**;≥3200 字 = **静默截断,音频恒封顶 ~338s**;**全程 HTTP 200,永不 400** —— 比 VoxCPM2(超长 400 拒绝)更"闷",facade 必须自己按 token 估长度硬闸。
- **显存随长度线性但极小**:1.1G(50字)→ 4.5G(6400字),满打满算 <5G。**长文本无任何显存压力**。
- **速度是短板**:短文本 RTF>1(慢于实时),长文本才摊到 ~0.25。

---

## 4. P3 — 采样 / 流式(待补)

| 项 | 现状 | 待测 |
|---|---|---|
| 采样稳定性 | 未测 | ⬜ |
| 流式 TTFB | 未知 | ⬜ |
| 音色库 `/v1/audio/voices` | 400 提示支持 | ⬜ |

---

## 5. P4 — 任务面

| 任务 | 状态 |
|---|---|
| 零样本克隆 TTS | ✅ |
| 音色库上传复用 | ⬜ |
| 情感/多语言 | ⬜ |

---

## 6. P5 — 并发 / 吞吐(sizing)

> 100 字/请求,并发提交 N。

| 并发 | 总时长 | 吞吐(条/s) | 均摊(s/条) |
|---|---|---|---|
| 1 | 27.99s | 0.04 | 27.99 |
| 2 | 48.35s | 0.04 | 24.18 |
| 4 | 95.05s | 0.04 | 23.76 |
| 8 | 185.52s | 0.04 | 23.19 |

**结论(关键)**:**吞吐恒 0.04 条/s,并发 1→8 完全不变,均摊稳定 ~23s/条 → 请求全串行,批处理零收益(等效 batch=1)**。这是本模型最硬的运营约束:
- **单实例吞吐 ≈ 2.6 条/min(100 字)**,无论并发多少。
- **提吞吐唯一手段 = 多副本**。好在显存仅 ~1-4.5G,**单卡可堆 8+ 副本**,4×A100 节点理论 32+ 副本 → 靠数量把聚合吞吐拉起来。
- 与 Qwen3-TTS(conc16 有批处理收益、1.33 条/s)形成鲜明对比:**Nano 走"多小副本"路线,不走"单大实例批处理"路线**。

---

## 7. P6 — 崩溃边界

| 输入 | 实测 | facade 动作 |
|---|---|---|
| ≥3200 字 | **HTTP 200 但音频静默截断(封顶 ~338s)**,永不 400 | 按 token 前置硬闸(单请求 ≤~1600 字)+ 句级切分 |
| 无 `ref_audio` | HTTP 400(`no built-in speakers, use ref_audio+ref_text`) | 门面强制带 ref |
| 裸路径 / 无 media flag | HTTP 400 | 统一 `file://` + serve `--allowed-local-media-path` |
| 缺 codec cache(HF_HOME) | **启动崩**(`LocalEntryNotFoundError`) | 挂 `HF_HOME=$ROOT/hf_cache` |
| 空 input / 非法 ref | ⬜ | ⬜ 待测 |

---

## 8. 复现命令

```bash
REG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly
IMG=$REG/vllm-omni:arm64-a100-latest
ROOT=/nfs-models/wuhanjisuan894/vllm-omni-speech
# serve(见 §1)后,克隆模式跑 harness:
export REF_AUDIO="file://$ROOT/_ref_zh.wav" REF_TEXT="你好,这是一段用于声音克隆的参考音频,语气平稳自然清晰。"
PORT=8091 CONTAINER=omni-moss-nano GPU_ID=0 bash /nfs-models/_transfer/tts_bench.sh smoke
PORT=8091 CONTAINER=omni-moss-nano GPU_ID=0 LENS="50 100 200 400 800 1600 3200 6400" bash /nfs-models/_transfer/tts_bench.sh len
PORT=8091 CONTAINER=omni-moss-nano GPU_ID=0 CONC="1 2 4 8 16" bash /nfs-models/_transfer/tts_bench.sh conc
```

---

## 9. 一页速查

| 维度 | 结论 |
|---|---|
| 类型 | **纯克隆**(无内置音色),`ref_audio`+`ref_text` 必填 |
| 生产配置 | 单卡 **8+ 副本**(显存 ~1-4.5G);4×A100 节点 32+ 副本;24kHz |
| 显存 | 1.1G(短)→ 4.5G(6400字),极轻 |
| RTF | 短文本 1.9-2.6(慢于实时);长文本 0.24-0.27 |
| 完整合成上限 | **~1600 字/请求(~340s 音频)**;超了静默截断,**永不 400** |
| 崩溃边界 | 无优雅 400,只截断;缺 `HF_HOME` codec cache 会启动崩 |
| 每实例吞吐 | **恒 ~2.6 条/min(batch=1,并发零收益)** → 靠多副本 |
| serve 必备 | `--allowed-local-media-path $ROOT` + `-e HF_HOME=$ROOT/hf_cache` |
| 踩坑 | ①codec HF-id 离线加载需 HF_HOME;②ref 必须 file://+media flag;③并发不 scale;④静默截断无 400 |
| 待补 | 采样稳定性/音色库/情感/多语言/流式 TTFB/空&非法输入 |
