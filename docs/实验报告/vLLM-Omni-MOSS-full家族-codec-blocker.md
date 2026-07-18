# vLLM-Omni · MOSS full-家族 codec bug —— 已根因 + 已修复

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80)· 测试机 0017
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 59add8a0)
> 日期:2026-07-18
> **状态:✅ 已定位为代码 bug 并修复,MOSS-TTSD 双卡真机验证出音(stereo 24kHz)。**

---

## 0. 结论先行

- **根因:代码 bug,非硬件/版本/下载问题。** `audio_tokenizer_v2.py` 的 encoder/decoder 块把 `input_proj`/`output_proj` **无条件**建成 `nn.Linear`,而真实 codec 在 in==out 的层用 `nn.Identity()`(无权重)→ 加载时凭空缺 6 个 proj 权重 → stage1 崩,**波及整个 full-MOSS 家族**(TTSD/VoiceGen/SoundEffect/Realtime 共用此 codec;仅 Nano 用单独 codec 不受影响)。
- **修复:2 行改条件式**(in!=out 才 Linear,否则 Identity),与本仓库 V1 codec / quantizer 写法一致。见 §4。
- **验证:** 打补丁后 MOSS-TTSD 起服务(8B talker + codec,**双卡分离** card0/card1 避 OOM),`[S1]/[S2]` 对话请求 **http 200 出 stereo 24kHz WAV**。
- **待固化:** 补丁已改进源码 `audio_tokenizer_v2.py`,需 commit + 官方重出镜像;之前是容器内热 patch。

---

## 1. 崩溃现象

以 MOSS-TTSD 为例,8B talker(stage0)加载正常,死在 stage1 codec:
```
File ".../vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_codec.py", line 680, in load_weights
RuntimeError: MOSS Audio Tokenizer weights were not fully loaded:
  loaded=1600/1606 missing=6 skipped=0 shape_mismatches=0;
  first_missing=['decoder.0.output_proj.weight','decoder.2.output_proj.weight','decoder.4.output_proj.weight',
                 'encoder.3.input_proj.weight','encoder.5.input_proj.weight']
```

---

## 2. 根因(已核到权重与代码)

**codec 加载流程**(`modeling_moss_tts_codec.py`):
- `_build_codec()`(line 727)**先试 V2 类** `MossAudioTokenizerV2Model`,失败才回落 V1。
- codec 路径来自 config `audio_tokenizer_name_or_path`,MOSS-TTSD config **未设**该字段 → 回落默认 `OpenMOSS-Team/MOSS-Audio-Tokenizer`。
- `load_weights()`(line 604)带一套 `_SUFFIX_REMAP` 映射 v1/v2 命名差异。

**权重实测**(`OpenMOSS-Team/MOSS-Audio-Tokenizer`,2 分片 safetensors,共 1600 键,16 个 proj 键):
| 位置 | checkpoint 实际有 | 模型(vendored V2)期望 |
|---|---|---|
| encoder.{3,5} | `output_proj` | `input_proj`(缺) |
| decoder.{0,2,4} | `input_proj` | `output_proj`(缺) |

**即:vendored V2 codec 类比当前 checkpoint 多声明了 6 个投影层**(encoder 侧要 input_proj、decoder 侧要 output_proj),而 checkpoint 里这些层是相反的命名/根本不存在。`missing=6 skipped=0` 说明 checkpoint 的 1600 键全部有归宿,但模型的 6 个参数无源可填。

- **不是 remap 可修**:不是简单 input↔output 换名(会破坏其它已匹配的层,如 encoder.1 两者都有),是**架构层数不一致**。
- **不是下载不全**:2 分片完整,1600 键干净加载。
- **不是硬件/依赖/离线问题**。

**判定**:vllm-omni 内置的 `MossAudioTokenizerV2` codec 实现,针对的是 `MOSS-Audio-Tokenizer` 的**某个特定 revision**;我们下载的当前版(2026-02 发布后有更新,3 月新增 ONNX/GGUF)结构已变,两者错位。

---

## 3. 影响面

MOSS-Audio-Tokenizer 是 **full-MOSS 家族共享 codec**(官方明确:服务 TTS/TTSD/VoiceGenerator/SoundEffect/Realtime)。故此 blocker 波及:

| 模型 | codec | 状态 |
|---|---|---|
| MOSS-TTS-Nano | MOSS-Audio-Tokenizer-**Nano** | ✅ 可用(codec 不同,不受影响) |
| MOSS-TTSD | MOSS-Audio-Tokenizer | ❌ stage1 codec 崩 |
| MOSS-VoiceGenerator | MOSS-Audio-Tokenizer | ❌ 预期同样崩(同 codec) |
| MOSS-SoundEffect | MOSS-Audio-Tokenizer | ❌ 预期同样崩(未下载) |
| MOSS-TTS-Realtime | MOSS-Audio-Tokenizer | ❌ orchestrator init 崩(疑同因;且不做流式) |

---

## 4. 修复(已实施)

**文件**:`vllm_omni/model_executor/models/moss_tts/audio_tokenizer_v2.py`(encoder/decoder 块 `__init__`,原 line 916/919)

```python
# 改前(无条件 Linear —— 在 in==out 的层凭空要 6 个不存在的 proj 权重):
self.input_proj = nn.Linear(input_dimension, d_model, bias=False)
self.output_proj = nn.Linear(d_model, output_dimension, bias=False)

# 改后(条件式,与 V1 codec line 467 / quantizer line 1121 一致):
self.input_proj = (
    nn.Linear(input_dimension, d_model, bias=False) if input_dimension != d_model else nn.Identity()
)
self.output_proj = (
    nn.Linear(d_model, output_dimension, bias=False) if d_model != output_dimension else nn.Identity()
)
```

`nn.Identity()` 对 forward 无影响(line 922-924 的 `input_proj(x)` 变恒等),且不再要求 checkpoint 里不存在的权重。

**验证脚本**(容器内热 patch,重出镜像前临时用):`scripts/patch_moss_codec.py`(幂等,serve 前跑)。

---

## 5. 处置

- **codec bug 已修**,full-MOSS 家族(TTSD 已验;VoiceGen/SoundEffect/Realtime 共用同 codec,预期一并解锁)从 blocker 转为可用。
- **显存**:8B talker + codec 在**单张 40G OOM**(官方 recipe 面向 H100 80G)。解法:**两 stage 分卡**(`scripts/moss_ttsd_a100_40g.yaml`:stage0→card0 / stage1→card1),`--gpus '"device=0,1"'`。
- **待办**:
  1. commit `audio_tokenizer_v2.py` 修复 + 官方重出镜像(固化,免运行时热 patch)。
  2. 镜像固化后验证 VoiceGen / SoundEffect / Realtime 是否一并可用。
  3. 出 MOSS-TTSD 正式实验报告(功能/长度/双音色/并发)。
- **复现(修复版)**:
  ```bash
  docker run -d --gpus '"device=0,1"' -e HF_HOME=$ROOT/hf_cache -v $ROOT:$ROOT -p 8091:8091 $IMG \
    bash -c "python3 $ROOT/patch_moss_codec.py && vllm serve $ROOT/MOSS-TTSD-v1.0 --omni --trust-remote-code \
    --allowed-local-media-path $ROOT --deploy-config $ROOT/moss_ttsd_a100_40g.yaml --port 8091"
  # [S1]/[S2] 对话请求 → http 200 stereo 24kHz WAV
  ```
