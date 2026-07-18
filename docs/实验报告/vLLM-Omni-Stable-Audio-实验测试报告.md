# vLLM-Omni · Stable-Audio-Open 实验测试报告(文生音乐/音效)

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0018
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 4b2def0b)
> 模型:`stabilityai/stable-audio-open-1.0`,权重 `/nfs-models/wuhanjisuan894/vllm-omni-speech/stable-audio-open-1.0`
> 端点:`POST /v1/audio/generate`(**非 TTS,不走 /v1/audio/speech**)
> 日期:2026-07-18

---

## 0. 结论先行

1. **单卡极宽裕**:显存 **~12.6G/40G**,几乎不随时长变 → **一卡可塞 2-3 副本**;4×A100 节点密度高。
2. **生成时间 ~4s 恒定,与音频时长无关**(扩散固定步数,非 AR)→ **越长 RTF 越好**:47s 音频仅 4.4s 出,**RTF 0.093(快 10×)**;10s 音频 RTF 0.43。
3. **44.1kHz 立体声**,http 200 稳定,产物随时长线性(10s→1.7M / 20s→3.4M / 47s→8.0M)。
4. 用途:文生音乐/音效(短剧 BGM、环境音、拟音),`input`=英文描述,`audio_length`≤~47s。
5. serve 需 `--enforce-eager --gpu-memory-utilization 0.9`;`/ready` 对非语音模型可能不翻 200,**用 `/health` 判就绪**。

---

## 1. 环境与 serve

```bash
docker run -d --name omni-stableaudio --gpus '"device=0"' --memory=240g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -v $ROOT:$ROOT -p 8091:8091 \
  "$IMG" vllm serve "$ROOT/stable-audio-open-1.0" --omni --trust-remote-code \
  --enforce-eager --gpu-memory-utilization 0.9 --port 8091
# 就绪:curl /health(非语音模型 /ready 可能不翻 200)
```

请求(文生音频):
```bash
curl -s -X POST localhost:8091/v1/audio/generate -H "Content-Type: application/json" \
  -d '{"input":"warm lofi hip hop beat with vinyl crackle and mellow piano","audio_length":20,"guidance_scale":7}' \
  --output out.wav
```

---

## 2. 生成矩阵(guidance_scale=7,单卡,已测 2026-07-18)

| audio_length | 生成 | http | 大小 | 格式 | 峰值显存 | RTF |
|---|---|---|---|---|---|---|
| 10s | 4.27s | 200 | 1.7M | 44.1k 16bit stereo | 12639 MiB | 0.43 |
| 20s | 3.93s | 200 | 3.4M | 44.1k stereo | 12645 MiB | 0.20 |
| 47s | 4.37s | 200 | 8.0M | 44.1k stereo | 12651 MiB | **0.093** |

---

## 3. 关键发现

- **扩散固定步数 → 生成时间恒定(~4s)**,与音频时长无关。与 AR(Qwen3-TTS RTF 恒定 ~0.17)、视频(时长受显存限)都不同:**Stable-Audio 时长几乎免费**,唯一变量是扩散步数。
- **显存平**(12639→12651,+12MiB/37s)→ 时长无显存瓶颈,~12.6G 常驻。
- **端点不同**:`/v1/audio/generate`(非 `/v1/audio/speech`),harness `tts_bench.sh` 不适用,用手动 curl。
- 与 ACE-Step(另一 agent 负责的音乐引擎)定位重叠;本镜像内 Stable-Audio 是"顺带支持"的音效/音乐兜底。

---

## 4. 待测

| 维度 | 待测 |
|---|---|
| guidance_scale / num_inference_steps 扫 | 对时间/质量的影响 |
| 崩溃边界 | audio_length > 47s?空 input?非法参数 |
| 并发/吞吐 | 单卡多请求 QPS(gen~4s → 理论 ~15 条/min/卡) |
| 质量/听感 | 音乐性、音效贴合度(人工;或 CLAP 客观) |

---

## 5. 一页速查

| 维度 | 结论 |
|---|---|
| 端点 | `POST /v1/audio/generate`(非 TTS) |
| 采样率 | 44.1kHz 立体声 |
| 显存 | ~12.6G/40G(不随时长)→ 单卡 2-3 副本 |
| 生成时间 | ~4s 恒定(扩散固定步数),与时长无关 |
| RTF | 0.43(10s)→ 0.093(47s),越长越快 |
| 时长上限 | ~47s(官方) |
| serve | `--enforce-eager --gpu-memory-utilization 0.9`;就绪判 `/health` |
| 用途 | 文生音乐/音效(短剧 BGM/环境音/拟音) |
| 踩坑 | 非语音模型 `/ready` 可能不翻 200(gate 在语音 warmup);长 docker run 行粘贴易吃换行,用单行 `;` |
