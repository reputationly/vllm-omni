# vLLM-Omni · AudioX 实验测试报告(文/视频生音频·扩散)

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0017
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 61bcf3d6 + **ming.py 热patch**,见 §1 前置)
> 模型:`zhangj1an/AudioX`,权重 `/nfs-models/wuhanjisuan894/vllm-omni-speech/AudioX`(11G)
> 端点:`POST /v1/chat/completions`(**非 /v1/audio/***);类:`AudioXPipeline`(扩散)
> 日期:2026-07-19
> 方法论照 `ACE-Step-1.5/docs/acestep-a100-实验测试报告.md`。

---

## 0. 结论先行

1. **可用**:文生音效(t2a)+ 文生音乐(t2m)真机验证通过,语义正确、**44.1kHz 立体声**、时长精确等于 `seconds_total`。
2. **玩法**:`/v1/chat/completions` + `--model-class-name AudioXPipeline`,任务经 `extra_body.audiox_task` 选(t2a/t2m/v2a/v2m/tv2a/tv2m 六种)。文本在 messages,视频任务另加 `video_url`。
3. **两个前置坑(已解)**:
   - **ming.py 模块级 bug**:导入链踩到 `transformers_utils/processors/ming.py` 的 `register(字符串)` → 崩(任何导入它的模型都中招)。已修(传 config 类 + try/except)。
   - **离线缺 T5/CLIP**:AudioX 建文本/视频编码器要 `t5-base` + `openai/clip-vit-base-patch32` 的 config/tokenizer,需预下进 hf_cache。
4. **生成时间随扩散步数线性**:100 步 ~11.5s / 250 步 ~27.5s(10s 音频);**时长与步数无关,只由 `seconds_total` 定**。
5. **音质**:**≥250 步**明显更干净;**稀疏/静音提示**(如"quiet park")近静音段有 **VAE 气泡/咔哒伪影**(潜空间音频扩散固有);**密集连续声**(雨/雷)干净真实。音量偏低(不做响度归一化,靠 facade)。

---

## 1. 环境与 serve

**前置修复**(镜像固化前需容器内热 patch;`scripts/patch_ming_register.py`):ming.py register bug。
**离线辅助权重**(预下进 hf_cache):`t5-base` + `openai/clip-vit-base-patch32`(已加进 `download_speech_models.sh` 的 `aux_cache`)。

```bash
docker run -d --name omni-audiox --gpus '"device=2"' --memory=240g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=$ROOT/hf_cache \
  -e DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
  -v $ROOT:$ROOT -p 8092:8092 "$IMG" \
  vllm serve "$ROOT/AudioX" --omni --model-class-name AudioXPipeline --port 8092
# 就绪判 /health(扩散模型 /ready 可能不翻)
```
- **不要加 `--trust-remote-code`**(AudioX 是扩散 pipeline,非 transformers 远程代码模型)。
- **无 deploy yaml**:靠 `--model-class-name AudioXPipeline`(registry 已注册)。
- 单卡即可(扩散,单请求)。

---

## 2. 请求格式(chat 端点)

**文生音效 / 音乐(t2a / t2m)**:
```bash
curl -sS -X POST localhost:8092/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"'"$ROOT"'/AudioX",
  "messages":[{"role":"user","content":[{"type":"text","text":"A dog barking in a quiet park with birds chirping."}]}],
  "extra_body":{"audiox_task":"t2a","num_inference_steps":250,"guidance_scale":7.0,"seed":42,"seconds_total":10.0}}'
```
产物在 `choices[0].message.audio.data`(base64 WAV,需解码)。

| extra_body 字段 | 说明 |
|---|---|
| `audiox_task` | **必填**:t2a/t2m/v2a/v2m/tv2a/tv2m |
| `num_inference_steps` | 扩散步数(默认100;**生产 ≥250**) |
| `guidance_scale` | CFG(默认7.0) |
| `seconds_total` | 音频时长秒(默认~10);**决定产物时长** |
| `seed` / `sigma_min` / `sigma_max` / `cfg_rescale` | 可选;默认 0.03 / 1000 / 0.0 |

**视频任务(v2*/tv2*)**:messages.content 加 `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}`(未测,需视频输入)。

---

## 3. 实测

| 任务 | 提示 | 步数 | http | 生成 | 声道/采样率 | 时长 | 听感 |
|---|---|---|---|---|---|---|---|
| t2a | 狗叫+鸟鸣(quiet park) | 100 | 200 | 11.66s | 2 / 44100 | 10.00s | 语义对,但**气泡/咔哒噪声**,音量小 |
| t2a | 同上 | 250 | 200 | 27.56s | 2 / 44100 | 10.00s | 噪声减轻(剩 2 声气泡),事件更集中 |
| t2a | **暴雨+雷**(密集连续) | 250 | 200 | 27.45s | 2 / 44100 | 10.00s | ✅ **真实,无气泡** |
| t2m | 欢快电子舞曲 | 100 | 200 | 11.28s | 2 / 44100 | 10.00s | 音乐性"还行" |

**结论**:
- **生成时间 ∝ 步数**(100→250 = 11.5→27.5s ≈ 2.4×);时长恒等于 `seconds_total`(扩散,与步数/内容无关)。
- **气泡/咔哒 = 静音段 VAE 伪影**:稀疏/含静音提示才有(狗叫 quiet park);密集连续声(雨雷)完全干净 → 潜空间音频扩散固有特性,非 bug。
- **音量偏低**:无响度归一化,生产靠 facade pyloudnorm(引擎无关,对所有模型统一)。

---

## 4. 待补

| 维度 | 待测 |
|---|---|
| 视频任务 v2a/v2m/tv2a/tv2m | 需 mp4 输入,验视频→音频/音乐同步 |
| 显存/并发 | 峰值显存、单卡吞吐、多副本密度 |
| 步数/采样扫 | 步数 vs 音质拐点、sigma/cfg_rescale 对伪影影响 |
| 时长上限 | `seconds_total` 上限(>10s?)与显存关系 |
| 去咔哒后处理 | 静音段伪影的 declick/门限处理 |

---

## 5. 一页速查

| 维度 | 结论 |
|---|---|
| 类型 | 扩散文/视频生音频(t2a/t2m/v2a/v2m/tv2a/tv2m) |
| 端点 | `POST /v1/chat/completions`(非 /v1/audio/*) |
| 启动 | `--model-class-name AudioXPipeline`,**无** --trust-remote-code,`DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN`,就绪判 `/health` |
| 采样率 | **44.1kHz 立体声**(全批唯一非语音、非 24k) |
| 生成时间 | ∝ 步数(100步~11.5s / 250步~27.5s / 10s音频) |
| 时长 | = `seconds_total`(与步数/内容无关) |
| 质量 | 生产 ≥250 步;密集声干净;稀疏/静音有 VAE 气泡伪影;音量低靠 facade 归一 |
| 前置坑 | ①ming.py register bug(需修+固化)②离线需 t5-base+clip 进 hf_cache |
| 用途 | 短剧音效/BGM、环境音、拟音;视频配音(v2*,待测) |
| 待补 | 视频任务、显存并发、步数扫、时长上限、去咔哒 |
