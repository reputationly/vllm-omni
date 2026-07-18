# vLLM-Omni · CosyVoice3 实验测试报告(零样本克隆 TTS)

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0016
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 59add8a0;已内置 `s3tokenizer` 运行时依赖)
> 模型:`Fun-CosyVoice3-0.5B-2512`,权重 `/nfs-models/wuhanjisuan894/vllm-omni-speech/Fun-CosyVoice3-0.5B-2512`
> harness:`scripts/smoke/tts_bench.sh`(smoke/len,打 `/v1/audio/speech`)
> 日期:2026-07-18
> 方法论照 `ACE-Step-1.5/docs/acestep-a100-实验测试报告.md`。

---

## 0. 结论先行

1. **纯克隆模型(无内置音色)**:`ref_audio`(`file://`/URL/base64)+ `ref_text`;serve 需 `--allowed-local-media-path $ROOT`。依赖 `s3tokenizer`(旧镜像缺 → 新镜像已内置)。
2. **显存偏高**:短文本 ~21G,≥200 字升到 **~26.5G/40G** → **单卡 1 副本**(不像 VoxCPM2/Nano 能多副本)。
3. **速度是明显短板**:warm RTF **0.4~1.2**,**200 字时 RTF 1.18(慢于实时)**;绝对生成时间在本批克隆模型里最慢(len200 gen 75s)。
4. **完整合成上限最短 ≈ 200 字/请求**(cps~3-4);**400 字即截断**(cps 7)。比 VoxCPM2(~1600)、MOSS-Nano(~1600)差一大截。
5. **功能最全(潜力)**:官方支持 零样本克隆 / SFT 预设 / 跨语言 / 指令控制 / 细粒度情感,本轮只验了克隆通路。
6. 采样率 **24kHz**。定位:功能全但**慢 + 吃显存 + 短上下文**,适合**质量优先、短句、单副本**场景;不适合高吞吐长文本。

---

## 1. 环境与权重

```bash
docker run -d --name omni-cosyvoice3 --gpus '"device=1"' --memory=240g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=$ROOT/hf_cache \
  -v $ROOT:$ROOT -p 8092:8092 \
  "$IMG" vllm serve "$ROOT/Fun-CosyVoice3-0.5B-2512" --omni --trust-remote-code \
  --allowed-local-media-path "$ROOT" --port 8092
```
- 目录含多份 onnx(campplus / flow.decoder / speech_tokenizer_v3),本地加载正常。
- 依赖 `s3tokenizer`(镜像已装)。

---

## 2. P1 — 功能 / 配置面

| 能力 | 支持 | 状态 |
|---|---|---|
| 预设音色 | ❌(400:`no built-in speakers`) | — |
| 零样本克隆 | ✅ `ref_audio`+`ref_text` | ✅ 通过 |
| SFT / 跨语言 / 指令 / 情感 | ✅(官方) | ⬜ 待专测 |

采样率:**24kHz**。

---

## 3. P2 — 长度压测(单请求,ref 克隆,多样中文文本)

| 字数 | http | 生成(热) | audio | cps | RTF | 峰值显存 | 判读 |
|---|---|---|---|---|---|---|---|
| 50 | 200 | 7.11s | 13.48s | 3.7 | 0.527 | 21045 | ✅ 完整 |
| 100 | 200 | 19.67s | 24.00s | 4.2 | 0.820 | 21291 | ✅ 完整 |
| 200 | 200 | 75.04s | 63.44s | 3.2 | **1.183** | 26509 | ✅ 完整但**慢于实时** |
| 400 | 200 | 48.90s | 57.16s | 7.0 | 0.855 | 26511 | ⚠️ 截断 |
| ≥800 | — | — | — | — | — | 26511 | ⬜ 慢/未完整测(0% util 长挂,已 Ctrl-C) |

**结论**:
- **完整合成上限 ≈ 200 字/请求**,400 字即截断 —— 本批克隆模型里**上下文预算最小**。
- **显存 21→26.5G**(≥200 字 +5.5G),单卡 1 副本。
- **慢**:200 字 RTF 1.18,长请求 gen 时间陡增(len200 gen 75s),≥800 字实测长时间 0% util 无产出(疑长截断退化/卡),已中止。

---

## 4-5. 采样 / 任务 / 并发(待补)

- 并发未测(RTF~1 的慢模型,conc 低收益;单卡 1 副本,吞吐靠多卡)。
- SFT 预设 / 跨语言 / 指令情感通路未验(官方支持,潜力最大,值得专测)。

---

## 6. P6 — 崩溃边界

| 输入 | 实测 | facade 动作 |
|---|---|---|
| ≥400 字 | HTTP 200 静默截断(cps 7) | 句级切分,单句 ≤~200 字 |
| ≥800 字 | 长时间无产出(疑卡/退化) | ⬜ 需复测确认是否 400/超时;门面按 token 硬闸 ≤200 字 |
| 无 ref / 裸路径 / 无 media flag | HTTP 400 | file:// + `--allowed-local-media-path` |

---

## 7. 复现命令

```bash
ROOT=/nfs-models/wuhanjisuan894/vllm-omni-speech; H=/nfs-models/_transfer/tts_bench.sh
# serve(见 §1)后:
export REF_AUDIO="file://$ROOT/_ref_zh.wav" REF_TEXT="你好,这是一段用于声音克隆的参考音频,语气平稳自然清晰。"
PORT=8092 CONTAINER=omni-cosyvoice3 GPU_ID=1 bash $H smoke
PORT=8092 CONTAINER=omni-cosyvoice3 GPU_ID=1 LENS="50 100 200 400" bash $H len
```

---

## 8. 一页速查

| 维度 | 结论 |
|---|---|
| 类型 | **纯克隆**(无内置音色),功能全(潜力) |
| 生产配置 | 单卡 **1 副本**(显存 ~26.5G);24kHz |
| 显存 | 21G(短)→ 26.5G(≥200字) |
| RTF | 0.4~1.2,**200字 1.18(慢于实时)** —— 本批最慢 |
| 完整合成上限 | **~200 字/请求**(最短);400 字即截断 |
| 崩溃边界 | ≥400 截断;≥800 疑卡(待复测) |
| serve 必备 | `--allowed-local-media-path $ROOT`;镜像需 `s3tokenizer` |
| 踩坑 | ①旧镜像缺 s3tokenizer;②上下文预算小(200字);③慢+吃显存 |
| 待补 | SFT/跨语言/指令情感 专测、≥800 边界复测、并发、克隆音质 |
