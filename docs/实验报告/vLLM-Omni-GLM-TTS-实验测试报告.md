# vLLM-Omni · GLM-TTS 实验测试报告(零样本克隆 TTS)

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0017
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 59add8a0)
> 模型:`GLM-TTS`,权重 `/nfs-models/wuhanjisuan894/vllm-omni-speech/GLM-TTS`
> harness:`scripts/smoke/tts_bench.sh`(smoke/len,打 `/v1/audio/speech`)
> 日期:2026-07-18
> 方法论照 `ACE-Step-1.5/docs/acestep-a100-实验测试报告.md`。

---

## 0. 结论先行

1. **纯克隆模型(无内置音色)**:`ref_audio`(`file://`/URL/base64)+ `ref_text`;serve 需 `--allowed-local-media-path $ROOT`。(base voice `vivian` 直接 400:`GLM-TTS has no built-in speakers`)
2. **显存高**:~27~29G/40G(随长度上升)→ **单卡 1 副本**。
3. **速度中上(warm)**:冷启首条 ~54s,warm RTF **0.36~0.49**(比 CosyVoice3 快、比 VoxCPM2 稍慢),200 字仍 RTF 0.44(快于实时)。
4. **完整合成上限 ≈ 200 字/请求**(cps~4);**400 字即截断**(cps 6.5)。与 CosyVoice3 同档(短上下文)。
5. 采样率 **24kHz**。定位:克隆质量向,warm 速度可用,但**短上下文 + 高显存 + 单副本**,长文本需门面切分。

---

## 1. 环境与权重

```bash
docker run -d --name omni-glmtts --gpus '"device=1"' --memory=240g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=$ROOT/hf_cache \
  -v $ROOT:$ROOT -p 8092:8092 \
  "$IMG" vllm serve "$ROOT/GLM-TTS" --omni --trust-remote-code \
  --allowed-local-media-path "$ROOT" --port 8092
```

---

## 2. P1 — 功能 / 配置面

| 能力 | 支持 | 状态 |
|---|---|---|
| 预设音色 | ❌(400:`no built-in speakers`) | — |
| 零样本克隆 | ✅ `ref_audio`+`ref_text` | ✅ 通过 |
| 音色库上传 | ✅(400 提示 `POST /v1/audio/voices`) | ⬜ 待测 |
| 情感/多语言/流式 | ⬜ | ⬜ 待测 |

采样率:**24kHz**。

---

## 3. P2 — 长度压测(单请求,ref 克隆,多样中文文本)

| 字数 | http | 生成(热) | audio | cps | RTF | 峰值显存 | 判读 |
|---|---|---|---|---|---|---|---|
| smoke(~40) | 200 | 4.13~4.25s(warm)/54.4s(冷) | 11.44s | 3.5 | 0.36(warm) | 27443 | ✅ 完整 |
| 50 | 200 | 5.98s | 13.76s | 3.6 | 0.435 | 27527 | ✅ 完整 |
| 100 | 200 | 12.63s | 25.56s | 3.9 | 0.494 | 27851 | ✅ 完整 |
| 200 | 200 | 22.37s | 50.40s | 4.0 | 0.444 | 28517 | ✅ 完整(**上限附近**) |
| 400 | 200 | 24.48s | 61.08s | 6.5 | 0.401 | 29041 | ⚠️ 截断 |
| ≥800 | — | — | — | — | — | 29041 | ⬜ 慢/未完整测(0% util 长挂,已 Ctrl-C) |

**结论**:
- **完整合成上限 ≈ 200 字/请求**,400 字即截断 —— 短上下文,与 CosyVoice3 同档。
- **显存高且随长度升**:27.4→29G,单卡 1 副本。
- **warm 速度可用**(RTF ~0.4),但冷启首条慢(~54s)。

---

## 4-5. 采样 / 任务 / 并发(待补)

- 并发未测(单卡 1 副本,高显存,吞吐靠多卡)。
- 音色库 `/v1/audio/voices` 上传复用、情感、多语言未验。

---

## 6. P6 — 崩溃边界

| 输入 | 实测 | facade 动作 |
|---|---|---|
| ≥400 字 | HTTP 200 静默截断(cps 6.5) | 句级切分,单句 ≤~200 字 |
| ≥800 字 | 长时间无产出(疑卡/退化) | ⬜ 复测确认;门面 token 硬闸 ≤200 字 |
| base voice / 无 ref / 裸路径 | HTTP 400 | file:// + `--allowed-local-media-path` |

---

## 7. 复现命令

```bash
ROOT=/nfs-models/wuhanjisuan894/vllm-omni-speech; H=/nfs-models/_transfer/tts_bench.sh
# serve(见 §1)后:
export REF_AUDIO="file://$ROOT/_ref_zh.wav" REF_TEXT="你好,这是一段用于声音克隆的参考音频,语气平稳自然清晰。"
PORT=8092 CONTAINER=omni-glmtts GPU_ID=1 bash $H smoke
PORT=8092 CONTAINER=omni-glmtts GPU_ID=1 LENS="50 100 200 400" bash $H len
```

---

## 8. 一页速查

| 维度 | 结论 |
|---|---|
| 类型 | **纯克隆**(无内置音色) |
| 生产配置 | 单卡 **1 副本**(显存 ~29G);24kHz |
| 显存 | 27.4G → 29G(随长度) |
| RTF | warm 0.36~0.49(可用);冷启首条 ~54s |
| 完整合成上限 | **~200 字/请求**;400 字即截断 |
| 崩溃边界 | ≥400 截断;≥800 疑卡(待复测) |
| serve 必备 | `--allowed-local-media-path $ROOT` |
| 踩坑 | ①克隆-only(base voice 直接 400);②短上下文;③高显存 |
| 待补 | 音色库/情感/多语言/流式、≥800 边界复测、并发、克隆音质 |
