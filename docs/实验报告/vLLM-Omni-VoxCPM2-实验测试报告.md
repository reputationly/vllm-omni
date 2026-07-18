# vLLM-Omni · VoxCPM2 实验测试报告(零样本声音克隆 TTS)

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0016
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 59add8a0,base `vllm/vllm-openai:v0.25.0`,torch 2.11 cu130,sm_80;已内置 `voxcpm` 运行时依赖)
> 模型:`VoxCPM2`,权重 `/nfs-models/wuhanjisuan894/vllm-omni-speech/VoxCPM2`
> harness:`scripts/smoke/tts_bench.sh`(smoke/len/conc,打 `/v1/audio/speech`)
> 日期:2026-07-18
> 方法论照 `ACE-Step-1.5/docs/acestep-a100-实验测试报告.md`:结论先行 → 功能 → 长度 → 采样 → 任务 → 并发 → 崩溃 → 复现 → 速查。

---

## 0. 结论先行

1. **纯克隆模型(无内置音色)**:必须 `ref_audio`(URL / base64 / **`file://`**)+ `ref_text`;serve 需 `--allowed-local-media-path $ROOT` 放行本地 ref 文件。
2. **单卡宽裕**:常驻 **~13.4G/40G**,近满上下文峰值 ~14.1G(几乎不涨)→ **单卡可塞 2 副本**,4×A100 节点密度高。
3. **RTF 优秀(长文本)**:热态 **0.12–0.21**(快 5–8×),音频时长随字数线性(1600 字 → **320s** 音频),**长文本几乎免费**。短文本有固定开销(smoke 10s 音频 warm RTF ~0.6)。
4. **完整合成上限 ≈ 1600 字/请求**(cps~4-5);3200 字 = **静默截断(HTTP 200 但音频封顶 ~131s,cps 24)**;**6400 字 = HTTP 400 优雅拒绝,不杀引擎**。比 Qwen3-TTS(~200 字)宽松得多。
5. **并发弱且不稳**:AR 解码,批处理收益差,吞吐峰值仅 **~1.5 条/s/卡**,且 conc2/4 出现明显抖动(调度不稳)→ **提吞吐靠多副本 + GPUStack 分发,不是单实例加 batch**。
6. 采样率 **24kHz**(WAV)。用途:短剧/有声书**指定音色克隆**(先上传/给定参考音,再逐句合成)。

---

## 1. 环境与权重

- serve(单卡,离线,放行本地 ref):
```bash
docker run -d --name omni-voxcpm2 --gpus '"device=0"' --memory=240g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=$ROOT/hf_cache \
  -v $ROOT:$ROOT -p 8091:8091 \
  "$IMG" vllm serve "$ROOT/VoxCPM2" --omni --trust-remote-code \
  --allowed-local-media-path "$ROOT" --port 8091
# 就绪:curl /ready(200)
```
- **依赖**:镜像已内置 `voxcpm>=2.0`(旧镜像会 `ImportError: No module named 'voxcpm'` → 本轮新镜像已修复,`docker run --rm $IMG python3 -c "import voxcpm"` 通过)。
- **克隆请求**(关键 3 字段):
```bash
curl -s -X POST localhost:8091/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "input":"要合成的文本","ref_audio":"file:///.../\_ref_zh.wav",
  "ref_text":"参考音频的准确转写","response_format":"wav"}' --output out.wav
```
> `ref_audio` 不能是裸路径(报 `must be a URL / base64 / file://`);`file://` 需配 serve 端 `--allowed-local-media-path`(否则 `Cannot load local files without --allowed-local-media-path`)。

---

## 2. P1 — 功能 / 配置面

| 能力 | 支持 | 用法 | 状态 |
|---|---|---|---|
| 预设音色 | ❌ | 无内置 speaker | — |
| 零样本克隆 | ✅ | `ref_audio`(file:///URL/base64)+ `ref_text` | ✅ 冒烟通过(http 200 出 WAV) |
| 情感/风格 | ⬜ | 待测(是否支持 `instructions`) | ⬜ |
| 多语言 | ⬜ | 待测 | ⬜ |
| 流式 | ⬜ | 待测 | ⬜ |

采样率:**24kHz**(1213484 B / 12.64s ≈ 48KB/s = 24k·16bit·mono)。

---

## 3. P2 — 长度压测(单请求,ref=_ref_zh.wav,多样中文文本)

> `cps`(字/audio秒)正常 ~4-5;**>6 = 截断**(音频封顶,话没说完)。

| 字数 | http | 生成(热) | audio | cps | RTF | 峰值显存 | 判读 |
|---|---|---|---|---|---|---|---|
| 50 | 200 | 1.71s | 12.64s | 4.0 | 0.135 | 13677 | ✅ 完整 |
| 100 | 200 | 4.25s | 25.76s | 3.9 | 0.165 | 13135 | ✅ 完整 |
| 200 | 200 | 10.31s | 49.60s | 4.0 | 0.208 | 13677 | ✅ 完整 |
| 400 | 200 | 13.21s | 106.72s | 3.7 | 0.124 | 13677 | ✅ 完整 |
| 800 | 200 | 25.47s | 201.44s | 4.0 | 0.126 | 13677 | ✅ 完整 |
| 1600 | 200 | 42.89s | 320.16s | 5.0 | 0.134 | 13743 | ✅ 完整(**上限附近**) |
| 3200 | 200 | 19.56s | 131.04s | 24.4 | 0.149 | 14069 | ⚠️ 静默截断 |
| 6400 | **400** | 0.04s | — | — | — | 14069 | ✅ **优雅拒绝**(不崩) |

**结论**:
- **完整合成上限 ≈ 1600 字/请求**(~320s 音频,cps~5);~3200 字 = **静默截断(HTTP 200 但音频封顶)**;≥6400 字 = **HTTP 400 优雅拒绝**。
- **显存几乎不随长度变**(13.1→14.1G,+1G / 320s 音频)→ **长文本无显存瓶颈,瓶颈是 token 预算**。这与 Qwen3-TTS(近满上下文 +8G KV)不同,VoxCPM2 更省。
- **运营建议**:facade 长度硬闸(单请求 ≤~1600 字)+ 超长句级切分,逐句合成再拼接。

---

## 4. P3 — 采样 / 流式(待补)

| 项 | 现状 | 待测 |
|---|---|---|
| 采样稳定性 | 未测(同文本多次差异) | ⬜ |
| 流式 TTFB | 未知是否支持 | ⬜ |
| ref 质量敏感度 | ref_text 与音频转写是否需严格对齐 | ⬜(影响克隆音质) |

---

## 5. P4 — 任务面

| 任务 | 状态 |
|---|---|
| 零样本克隆 TTS | ✅ |
| 情感/多语言 | ⬜ |
| 多参考/长文本拼接 | ⬜ |

---

## 6. P5 — 并发 / 吞吐(sizing)

> 100 字/请求,并发提交 N。

| 并发 | 总时长 | 吞吐(条/s) | 均摊(s/条) |
|---|---|---|---|
| 1 | 5.65s | 0.18 | 5.65 |
| 2 | 46.73s | 0.04 | 23.36 |
| 4 | 57.95s | 0.07 | 14.49 |
| 8 | 6.43s | 1.24 | 0.80 |
| 16 | 10.95s | **1.46** | 0.68 |

**结论**:吞吐**低且抖动严重**(conc2/4 反比 conc1/8/16 慢一个量级,调度/批处理不稳定)。峰值 ~1.5 条/s/卡。**不能靠单实例加并发提吞吐**;生产靠**单卡 2 副本 + 多卡 + GPUStack 分发**。conc 抖动待复测确认(可能与 ref 重复编码/首请求 warmup 有关)。

---

## 7. P6 — 崩溃边界

| 输入 | 实测 | facade 动作 |
|---|---|---|
| ≥~6400 字(超 token 预算) | **HTTP 400 秒拒,容器 up** | 前置长度硬闸 |
| ~3200 字 | **HTTP 200 但音频静默截断**(cps 24) | 句级切分(单句 ≤~1600 字) |
| 裸路径 ref_audio | HTTP 400(`must be URL/base64/file://`) | 门面统一转 `file://` |
| 无 `--allowed-local-media-path` | HTTP 400(`Cannot load local files`) | serve 固定加该 flag |
| 空 input / 非法 ref | ⬜ | ⬜ 待测 |

---

## 8. 复现命令

```bash
REG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly
IMG=$REG/vllm-omni:arm64-a100-latest
ROOT=/nfs-models/wuhanjisuan894/vllm-omni-speech
# serve(见 §1)后,克隆模式跑 harness:
export REF_AUDIO="file://$ROOT/_ref_zh.wav" REF_TEXT="你好,这是一段用于声音克隆的参考音频,语气平稳自然清晰。"
PORT=8091 CONTAINER=omni-voxcpm2 GPU_ID=0 bash /nfs-models/_transfer/tts_bench.sh smoke
PORT=8091 CONTAINER=omni-voxcpm2 GPU_ID=0 LENS="50 100 200 400 800 1600 3200 6400" bash /nfs-models/_transfer/tts_bench.sh len
PORT=8091 CONTAINER=omni-voxcpm2 GPU_ID=0 CONC="1 2 4 8 16" bash /nfs-models/_transfer/tts_bench.sh conc
```

---

## 9. 一页速查

| 维度 | 结论 |
|---|---|
| 类型 | **纯克隆**(无内置音色),`ref_audio`+`ref_text` 必填 |
| 生产配置 | 单卡 **2 副本**(显存 ~13.4G);4×A100 节点 8 副本;24kHz |
| 显存 | idle ~13.4G / 峰值 ~14.1G(几乎不随长度),均 <40G |
| RTF(长文本) | 0.12–0.21,快 5–8×;短文本 ~0.6(固定开销) |
| 完整合成上限 | **~1600 字/请求(~320s 音频)** → 超了静默截断 |
| 崩溃边界 | ~6400 字 HTTP 400 优雅拒绝,不杀引擎 |
| 每卡吞吐 | ~1.5 条/s(峰值),并发抖动大 → 靠多副本 |
| serve 必备 flag | `--allowed-local-media-path $ROOT`(否则本地 ref 报 400) |
| 踩坑 | ①旧镜像缺 `voxcpm` 崩;②ref 必须 `file://`/URL/base64;③长文本 HTTP 200 静默截断;④并发抖动 |
| 待补 | 采样稳定性/情感/多语言/流式 TTFB/克隆音质(ref_text 对齐敏感度)/conc 复测 |
