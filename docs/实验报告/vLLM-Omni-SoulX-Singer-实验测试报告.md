# vLLM-Omni · SoulX-Singer 实验测试报告(歌声合成 SVS / 转换 SVC·扩散)

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0017
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 61bcf3d6)
> 模型:`Soul-AILab/SoulX-Singer`,权重 `/nfs-models/wuhanjisuan894/vllm-omni-speech/SoulX-Singer`(5.3G)+ `SoulX-Singer-Preprocess`(6.5G)
> 端点:`POST /v1/chat/completions`(**多模态 chat**);类:`SoulXSingerPipeline`(FlowMatching 扩散)
> 日期:2026-07-19
> 方法论照 `ACE-Step-1.5/docs/acestep-a100-实验测试报告.md`。

---

## 0. 结论先行

1. **可用(SVS precomputed 模式验证通过)**:用预计算音符/歌词元数据合成歌声,http 200,**mono 24kHz,50.96s**(落在测试断言 50-52s 区间)。
2. **玩法**:`/v1/chat/completions` + deploy `soulxsinger_svs.yaml`(SVS)/ `soulxsinger_svc.yaml`(SVC),输入经 `extra_args` 给参考人声 + 目标旋律/元数据。
3. **三个离线前置坑(已解)**:
   - **model_type 识别**:模型目录无 `model_index.json`/model_type → 需写最小 `config.json`(`model_type=soulxsinger`,`architectures=[SoulXSingerPipeline]`)。
   - **phone_set.json**:HF 不带,从 GitHub 固定 commit 拉(1928 条音素,44KB)放进 `模型目录/phoneme/phone_set.json`。
   - **preprocess 权重**:`rmvpe.pt`(音高)在 `SoulX-Singer-Preprocess`(已下);**precomputed 模式跳过在线 FunASR/ROSVOT**(否则要额外下 ASR + 音符转写模型)。
4. **两种模式**:SVS(歌声合成,歌词+曲谱)/ SVC(歌声转换)。本轮验 SVS precomputed(最省离线);SVC 需 `model-svc.pt`(待验)。

---

## 1. 环境与 serve(单卡,SVS)

**离线前置**(一次性):
```bash
ROOT=/nfs-models/wuhanjisuan894/vllm-omni-speech
# ① phone_set.json(GitHub 固定 commit;Mac/镜像下后 SFTP)
#   raw.githubusercontent.com/Soul-AILab/SoulX-Singer/81aeb3ae.../soulxsinger/utils/phoneme/phone_set.json
#   → $ROOT/SoulX-Singer/phoneme/phone_set.json
# ② 最小 config.json 让 vllm-omni 认出模型类型
cat > "$ROOT/SoulX-Singer/config.json" <<'JSON'
{"model_type":"soulxsinger","architectures":["SoulXSingerPipeline"],"max_num_seqs":1}
JSON
# ③ 测试资产(仓库 tests/assets/soulxsinger/)→ $ROOT/soulxsinger_assets/
```

serve:
```bash
docker run -d --name omni-soulx --gpus '"device=0"' --memory=240g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=$ROOT/hf_cache \
  -e DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
  -v $ROOT:$ROOT -p 8093:8093 "$IMG" \
  vllm serve "$ROOT/SoulX-Singer" --omni --trust-remote-code \
  --deploy-config /app/vllm-omni/vllm_omni/deploy/soulxsinger_svs.yaml --port 8093
# 就绪判 /health;单卡 gpu_memory_utilization 0.5
```

---

## 2. 请求格式(precomputed SVS,chat 端点)

```bash
curl -sS -X POST localhost:8093/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"'"$ROOT"'/SoulX-Singer","modalities":["audio"],
  "messages":[{"role":"user","content":[{"type":"text","text":"soulx-singer"}]}],
  "num_inference_steps":32,"guidance_scale":3.0,"seed":42,
  "extra_args":{
    "prompt_metadata_path":"'"$A"'/zh_prompt.json",
    "target_metadata_path":"'"$A"'/music.json",
    "audio_path":"'"$A"'/zh_prompt.mp3",
    "preprocess_weights_dir":"'"$ROOT"'/SoulX-Singer-Preprocess",
    "language":"Mandarin","control":"score","vocal_sep":false,"auto_shift":true,"pitch_shift":0}}'
```
产物在 `choices[0].message.audio.data`(base64 WAV)。

| extra_args | 说明 |
|---|---|
| `prompt_metadata_path` / `target_metadata_path` / `audio_path` | **precomputed 模式**:预计算元数据(音符/歌词)+ 参考人声 wav → 跳过在线 preprocess |
| `preprocess_weights_dir` | rmvpe 等(音高) |
| `language` / `control` | Mandarin/Cantonese/English;`score`(曲谱)/`melody`(旋律) |
| `auto_shift` / `pitch_shift` / `vocal_sep` | 自动移调 / 手动半音 / 人声分离 |

**在线 preprocess 模式**(不用预计算元数据):改传 `prompt_audio` + `target_audio` + `preprocess_weights_dir` → 服务端跑 FunASR ASR + ROSVOT 音符转写(需额外下这些非 pip 权重,离线更重)。

---

## 3. 实测(SVS precomputed)

| 用例 | http | 生成 | 声道 | 采样率 | 时长 | 判读 |
|---|---|---|---|---|---|---|
| zh_prompt 音色唱 music 旋律 | 200 | 9.70s | mono | 24000 | 50.96s | ✅ 落在测试断言 50-52s 区间 |

- 生成时间 ~9.7s(32 步扩散),产 ~51s 歌声。
- 输出时长由目标曲谱/元数据决定(非固定)。

---

## 3.1 SVC + 集成 preprocess(POC 第二轮,2026-07-19)

镜像 61bcf3d6,单卡 A100 40G。两条未验证链路真机跑通:

| 用例 | 模式 | http | 声道/采样率 | 时长 | 判读 |
|---|---|---|---|---|---|
| zh_prompt 音色唱 music(**喂原始音频**) | SVS 集成 preprocess | 200 | mono/24000 | 51.40s | ✅ 服务器内联 ASR+音符+音高+人声分离,免预计算 JSON;偶发个别字不清(ASR 识错) |
| zh_prompt 音色转 music 歌声 | SVC 歌声转换 | 200 | mono/24000 | 51.48s | ✅ 换音色,无大问题 |

**离线依赖/前置(内嵌固化清单 —— 均为镜像/cache 缺项,POC 逐一补齐验通)**:
- **pip**:`BS-RoFormer`(人声分离+F0,SVC/集成 preprocess 都要)+ `g2pM`/`g2p-en`/`ToJyutping`(中/英/粤 G2P,集成 preprocess 识歌词后转音素要)
- **NLTK 数据**:`averaged_perceptron_tagger` + `cmudict`(g2p-en 初始化时下,离线需预置)
- **HF cache**:`openai/whisper-base`(SVC 的 WhisperEncoder 加载;`HF_HUB_OFFLINE=1` 下会崩 → 必须预填进 HF cache。HF 直连不通,用 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1` 下,~1.1G)
- **模型视图**:SVC 用**独立视图目录** `SoulX-Singer-svc`(`config.json` 声明 `architectures:[SoulXSingerSVCPipeline]` + `mv model-svc.pt` 进来),与 SVS 的 `SoulX-Singer` 分开;`--model` 各指各的
- **已有**:`SoulX-Singer-Preprocess`(rmvpe/paraformer/rosvot)+ ffmpeg(镜像已带)

**质量注记**:集成 preprocess 靠 ASR 自动识歌词,偶发个别字不清(识错传导 + 唱歌拉长字音);要字正腔圆用 precomputed(手工校对元数据)。SVS/SVC 两条 deploy 各用 `soulxsinger_{svs,svc}.yaml`。

---

## 4. 待补

| 维度 | 待测 |
|---|---|
| 控制 | melody vs score、pitch_shift、多语言(粤语/英文) |
| 显存/时长 | 峰值显存、更长曲目、并发 |
| 人声分离 | `vocal_sep=true`(带伴奏输入) |

---

## 5. 一页速查

| 维度 | 结论 |
|---|---|
| 类型 | 歌声合成(SVS)/ 转换(SVC),FlowMatching 扩散 |
| 端点 | `POST /v1/chat/completions`(多模态)+ `--deploy-config soulxsinger_svs.yaml` |
| 启动 | 单卡 gpu_mem 0.5;`--trust-remote-code`;`DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN`;就绪判 `/health` |
| 采样率 | 24kHz mono |
| 前置坑 | ①最小 config.json(SVS/SVC 各一份,architectures 区分)②phone_set.json(GitHub 拉)③集成 preprocess 缺 BS-RoFormer/g2pM/g2p-en/ToJyutping/NLTK 数据④SVC 缺 HF cache 的 openai/whisper-base |
| 实测 | precomputed SVS 9.7s→50.96s;集成 preprocess SVS 51.40s;SVC 51.48s;均 http 200,24k mono |
| 用途 | 歌声合成(给音色+曲谱/旋律)/ 歌声转换(换音色翻唱);集成 preprocess 免手写元数据,precomputed 更字正腔圆 |
| 待补 | 控制模式(score/pitch_shift)、多语言(粤/英)、显存并发、人声分离(vocal_sep) |
