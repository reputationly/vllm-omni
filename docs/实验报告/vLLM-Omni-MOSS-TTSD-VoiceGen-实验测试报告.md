# vLLM-Omni · MOSS-TTSD / MOSS-VoiceGenerator 实验测试报告(对话合成 / 音色设计)

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0017
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 59add8a0 + **codec 修复**,见 [MOSS 家族 codec 修复报告](vLLM-Omni-MOSS-full家族-codec-blocker.md))
> 模型:`MOSS-TTSD-v1.0`(8B,对话)· `MOSS-VoiceGenerator`(1.7B,音色设计)
> 端点:`POST /v1/audio/speech`;日期:2026-07-18
> 前置:本轮两模型能跑,依赖 vllm-omni full-MOSS codec 两处 bug 修复(proj 条件式 + 声道按路径)。

---

## 0. 结论先行

1. **两模型均已跑通**(修复 codec bug 后):MOSS-TTSD 对话合成 + MOSS-VoiceGenerator 音色设计,http 200,**mono 24kHz**,时长/语速正常,音色符合描述。
2. **玩法与普通 TTS 不同**:
   - **MOSS-TTSD**:输入是 **`[S1]/[S2]` 对话脚本**,`ref_audio`(说话人1)必填、`ref_audio_2`(说话人2)可选;多轮双人对话一次合成。
   - **MOSS-VoiceGenerator**:输入 `input`(要说的话)+ **`instructions`(音色文字描述)**,零样本"设计"出符合描述的新音色,**不需要 ref_audio**。
3. **显存**:MOSS-TTSD 8B talker + codec 在**单张 40G OOM**(官方 recipe 面向 H100 80G);VoiceGen 1.7B 也在单卡边缘 OOM。解法:**两 stage 分卡**(talker / codec 各一张 A100)。
4. **变体自动识别**:靠模型路径名(`_detect_moss_variant`),路径含 `TTSD`→对话、`VoiceGenerator`→音色设计,**请求不用传 task_type**。
5. **codec 共享**:全 full-MOSS 家族共用 `MOSS-Audio-Tokenizer`,故此修复同时解锁 TTSD/VoiceGen/SoundEffect/Realtime(SoundEffect/Realtime 未逐一验,预期一并可用)。

---

## 1. 环境与 serve(A100 40G 双卡)

**依赖修复**(镜像固化前需容器内热 patch;固化后免):`scripts/patch_moss_codec.py`。
**部署 yaml**(两 stage 分卡):`scripts/moss_ttsd_a100_40g.yaml` / `scripts/moss_voicegen_a100_40g.yaml`。

```bash
# MOSS-TTSD(8B,card0=talker / card1=codec)
docker run -d --name omni-moss-ttsd --gpus '"device=0,1"' --memory=240g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=$ROOT/hf_cache \
  -v $ROOT:$ROOT -p 8091:8091 "$IMG" \
  vllm serve "$ROOT/MOSS-TTSD-v1.0" --omni --trust-remote-code \
  --allowed-local-media-path "$ROOT" --deploy-config "$ROOT/moss_ttsd_a100_40g.yaml" --port 8091
# 就绪判 /ready
```
> 修复固化前,serve 前加 `python3 $ROOT/patch_moss_codec.py &&`。VoiceGen 同理,换模型名 + `moss_voicegen_a100_40g.yaml` + `--gpus '"device=2,3"'`。

---

## 2. MOSS-TTSD —— 对话合成

### 2.1 请求格式

```bash
curl -s -X POST localhost:8091/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "input":"[S1]你好,今天天气怎么样?[S2]天气很好,阳光灿烂,很适合出门散步。[S1]那我们一起去公园走走吧。[S2]好啊,正好活动一下筋骨。",
  "ref_audio":"file:///.../\_ref_zh.wav",
  "response_format":"wav"}' --output out.wav
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `input` | ✅ | 对话脚本,`[S1]`/`[S2]` 标说话人,文本紧跟标签 |
| `ref_audio` | ✅ | 说话人1参考音(`file://`/URL/base64) |
| `ref_audio_2` | ⬜ | 说话人2参考音;省略则两人同音色 |

### 2.2 实测

| 用例 | http | 生成 | 声道 | 采样率 | 时长 | 判读 |
|---|---|---|---|---|---|---|
| 4 轮对话(~60字) | 200 | 6.5s | **mono** | 24000 | 10.96s | ✅ 语速正常(修复前 stereo/5.6s,快一倍) |

- **修复前后对比**:未修 codec 声道 bug 时输出 stereo 24kHz / 5.6s(快一倍、错误立体声);修复后 mono / 10.96s(正常)。
- 单 ref 时两说话人同音色;双音色需 `ref_audio_2`(待测)。

---

## 3. MOSS-VoiceGenerator —— 音色设计

### 3.1 请求格式

```bash
curl -s -X POST localhost:8092/v1/audio/speech -H 'Content-Type: application/json' -d '{
  "input":"你好,欢迎收听今天的节目,希望你有个愉快的一天。",
  "instructions":"低沉磁性的中年男声,语速平缓沉稳。",
  "response_format":"wav"}' --output out.wav
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `input` | ✅ | 要合成的文本 |
| `instructions` | ✅ | 音色文字描述(如"低沉磁性男声");无 ref_audio |

### 3.2 实测

| 用例 | http | 生成 | 声道 | 采样率 | 时长 | 判读 |
|---|---|---|---|---|---|---|
| "低沉磁性中年男声"(~24字) | 200 | 6.2s | **mono** | 24000 | 7.60s | ✅ 音色符合描述(人耳确认) |

---

## 4. 待补

| 维度 | 待测 |
|---|---|
| TTSD 双音色 | 传 `ref_audio_2` 给 S1/S2 不同参考音,验说话人区分度 |
| 长度/崩溃边界 | 对话/描述超长时的截断/崩溃行为(harness 不适用,需手动) |
| 并发/吞吐 | 8B TTSD 双卡吞吐;VoiceGen 密度 |
| VoiceGen 可控性 | 不同 instructions(性别/年龄/情绪/语速)的音色差异 |
| SoundEffect / Realtime | 同 codec,预期可用,待逐一验证 |
| 显存优化 | 单卡低 util vs 双卡分离的性价比 |

---

## 5. 一页速查

| 维度 | MOSS-TTSD | MOSS-VoiceGenerator |
|---|---|---|
| 用途 | 多说话人对话(短剧/播客) | 零样本音色设计 |
| 参数 | 8B | 1.7B |
| 输入 | `[S1]/[S2]` 对话脚本 | `input` + `instructions` 描述 |
| 音色来源 | `ref_audio`(+`ref_audio_2`) | `instructions` 文字描述 |
| 采样率 | 24kHz mono | 24kHz mono |
| 部署 | 双卡(8B talker + codec 分卡) | 双卡(1.7B 也边缘 OOM) |
| 前置 | full-MOSS codec 修复(proj + 声道) | 同 |
| 状态 | ✅ 对话正常出音 | ✅ 音色符合描述 |
