# IndexTTS-2 on vLLM-Omni 实验测试报告

> 日期:2026-07-14 · 环境:dev-gpustack-a100-0020(10.0.0.94,新扩容 5 节点之一),A100 PCIE 40G 单卡,鲲鹏 920 ARM,隔离网
> 镜像:`crpi-…/reputationly/vllm-omni:arm64-a100-20260714`(base `vllm/vllm-openai:v0.25.0` 官方 arm64 + vllm-omni main@62589203)
> 权重:`/nfs-models/wuhanjisuan894/models/IndexTTS-2`(官方布局 + 附属模型 overlay,零主权重改动)
> 一句话结论:**单请求 3.8×、并发吞吐 7.5×,音质人耳验收通过,上游零 patch,替换旧 in-process 引擎定案。⚠️ 但超长文本(>~200 字/句)会触发 CUDA assert 杀死整个引擎且不可自愈——facade 必须做句级切分硬前置校验。**

---

## 1. 结论先行

| 维度 | 结论 |
|---|---|
| 单请求延迟(~4.7s 中文短句) | **1.7-1.9s(RTF≈0.37)** vs 旧 server.py 5.5-8s ≈ **3.8×** |
| 并发吞吐 | 8 并发 7.2s 完成 ≈ **5.2× 实时** ≈ 旧串行实现 **7.5×** |
| 音质 | 音色克隆/情感控制人耳验收通过;22.05kHz mono WAV,与旧实现同规格 |
| 显存 | 24.6G / 40G(双 stage 各 0.4 utilization 共卡) |
| 启动 | ~5.4 min(含 BigVGAN torch.compile 预热 3 min,一次性) |
| 上游改动 | **零 patch**。只带 1 个附属模型 overlay + 1 个 deploy yaml(tokenizer 路径修正 + vocoder 本地路径) |
| 架构决策 | index-tts 旧引擎退役;vllm-omni 照 lightx2v 模式内嵌 GPUStack Custom 后端,统一挂多语音模型 |
| ⚠️ 生产准入约束 | **单句安全上限 216~324 字之间**(模型 `max_text_tokens=600` 硬限);超长 → CUDA device-side assert → **整个引擎死亡且不可自愈**(容器退出)。facade 必须句级切分 + 长度硬校验挡在引擎前 |

## 1.1 缺口补测结果(2026-07-14 round 2)

| 用例 | 结果 | 备注 |
|---|---|---|
| 长文本 72 字 | ✅ 15.3s 音频 | |
| 长文本 216 字 | ✅ 29.9s 音频(8.0s 生成,RTF 0.27) | 句级上限比预想高 |
| 长文本 **324 字** | ❌ **500 + 引擎崩溃退出** | CUDA assert,文本 token 超 600 |
| 长文本 500 字 | ❌ 同上 | |
| 情感-音频模式(emo_audio) | ✅ 5.3s,RMS −20.6 dBFS | base64 data URL,与 ref_audio 同格式 |
| 情感-文本模式(use_emo_text+emo_text) | ✅ 6.2s,RMS −19.3 dBFS | 走 qwen0.6bemo4 推断情感 |
| 英文 | ✅ 6.7s | |
| 中英混排 | ✅ 10.6s | |
| seed 复现性(seed=42 ×2) | ⚠️ **md5 不一致**,时长/电平近似(5.6s / −35.7 vs −35.9) | seed 只锁 stage0 GPT AR,stage1 S2Mel 扩散仍有独立随机性 → 无法字节级复现 |
| 崩溃自愈 | ❌ 324 崩溃后 health=000,**容器退出,不能继续服务**,需 `docker start` 恢复(~5min) | 严重程度:整服务不可用,非单请求失败 |

### 长跑稳定性(300 句连跑,并发 4,max_num_seqs=4)

| 指标 | 结果 |
|---|---|
| 成功率 | **300/300(零失败),300/300 有效 RIFF 音频** |
| 墙钟 | 287s,等效 1.05 句/s(并发 4) |
| 单请求延迟(含排队) | p50 3.10s / p90 3.53s / p99 4.12s / max 4.71s |
| 显存 | 空载 23279 → 首句懒加载后 24619 MiB,**此后 300 句全程恒定(24619~24639 抖动)** |

> 那 1340MiB 增量是**首请求懒加载附属模型的一次性开销**(w2v-bert/campplus 等入显存),不是泄漏——中间采样 60/120/180/240/300 句显存纹丝不动。**结论:显存加载后稳定,无泄漏,长跑零故障,可生产。**

### max_num_seqs 4 vs 8 吞吐对比(16 并发打满)

| 配置 | 墙钟(排空16) | 吞吐 | 首波延迟 | 平均延迟 | 峰值显存 |
|---|---|---|---|---|---|
| **max_num_seqs=4** | **40.3s** | **0.40 句/s** | 4.74s | 25.7s | 24.7G |
| max_num_seqs=8 | 43.2s | 0.37 句/s | 9.20s | 33.3s | 24.8G |

> **调大 batch 无收益反略降**:A100 单卡在此模型已算力饱和(batch=4 就喂满 GPU),batch=8 只是让一波塞 8 个请求内部争抢算力(首波延迟 4.74→9.20s 翻倍),总时间没省。**结论:max_num_seqs=4 是单卡最优值,已回滚。提吞吐靠多卡多实例横向扩,不是调 batch。**

## 2. 为什么快(原因分析,已核到代码)

旧方案(index-tts `infer_v2` in-process)与 vllm-omni 是同一套权重、两套执行引擎:

| # | 因素 | 旧实现 | vllm-omni | 影响 |
|---|---|---|---|---|
| 1 | **AR 解码策略** | HF `generate` + **beam search num_beams=3**(infer_v2.py:532)| vLLM 采样(top_k 30 / top_p 0.8,无 beam) | AR 阶段算力直接 ÷3,这是最大头 |
| 2 | **AR 执行方式** | transformers python 逐 token 循环,eager attention | vLLM 运行时:paged KV cache + CUDA Graph + FlashAttention,采样含 GPU 上的 repetition_penalty | 每 token 开销大幅下降 |
| 3 | **S2Mel 扩散步数** | 25 步 Euler(infer_v2.py:645) | **12 步**(deploy yaml `diffusion_steps: 12`)+ DiT bf16 + DiT CUDA Graph | S2Mel 阶段 ≈÷2,画质(音质)人耳无感 |
| 4 | **BigVGAN 声码器** | eager(我们还关了 cuda_kernel 避开 JIT 坑) | CUDA Graph 捕获 + torch.compile(4 个 mel 长度桶) | vocoder 显著提速 |
| 5 | **并发模型** | 全局 python 锁,整条链路串行 | continuous batching(max_num_seqs=4)+ 两 stage 流水线(A 在 S2Mel 时 B 可进 GPT) | 并发吞吐质变 |
| 6 | **参考音色条件计算** | 每请求算 w2v-bert/campplus/codec | Speaker cache(512MB LRU)命中即跳过 | 同音色复用时省固定开销 |

> 注意行为差异:beam→采样意味着**同文本多次生成结果不同**(temperature 0.8)。对配音场景通常是好事(可重roll),需要复现性时在请求里固定 seed(vLLM 支持 per-request seed)。

## 3. 部署形态与关键产物

```
M4 facade(异步任务 + NFS 落盘 + pyloudnorm 归一化 + loudness_lufs/gain_db 参数)← 规划中
        │ HTTP /v1/audio/speech(OpenAI 标准)
GPUStack 网关/调度(Custom 后端注册)← 待办
        │
vllm-omni 容器(上游原样) ── IndexTTS-2 / 后续 MOSS-TTS、Qwen3-TTS、CosyVoice3…
```

| 产物 | 位置 | 说明 |
|---|---|---|
| 镜像 | ACR `reputationly/vllm-omni:arm64-a100-20260714` | Mac(Apple Silicon 原生 arm64)直接 `docker build` + push,`docker/Dockerfile.cuda` 原样,仅 `--build-arg BASE_IMAGE=vllm/vllm-openai:v0.25.0` |
| 附属模型 overlay | 已并入 NFS 模型目录 | `wav2vec2bert/`、`campplus.pth`、`semantic_codec.pth`(safetensors→pth 转换)、`bigvgan/`——全部源自旧 index-tts 的 `hf_cache/`,**无需重新下载** |
| deploy 配置 | `<模型目录>/indextts2-a100.yaml` | 上游 `vllm_omni/deploy/indextts2.yaml` + 2 处修正(见坑 #1、§4) |

**离线加载机制(核心认知)**:vllm-omni 加载附属模型的顺序是"模型目录本地文件 → HF hub 下载"。把文件按它的本地查找名放进模型目录(`wav2vec2bert/` 子目录、`campplus.pth`、`semantic_codec.pth`),再把 config 里的 vocoder 名字用 deploy yaml 的 `hf_overrides.vocoder.name` 指到本地 `bigvgan/` 目录,即可完全离线,不碰 HF cache。

**模型类型识别**:官方权重目录没有 config.json,vllm-omni 靠**路径名兜底匹配**(`config_factory.py`,路径含 "indextts2"/"IndexTTS-2" 即命中)。→ 挂载路径必须含该字样,如 `/models/IndexTTS-2`。

## 4. 踩坑记录

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | **tokenizer 占位符离线崩**(唯一真坑) | 上游 deploy yaml 写 `tokenizer: gpt2`,vLLM 0.25 即使 `skip_tokenizer_init: true` 也要先把它 resolve 成路径 → `HF_HUB_OFFLINE=1` 下 snapshot_download 抛 `LocalEntryNotFoundError`,Orchestrator 直接崩 | yaml 两处改 `tokenizer: /models/IndexTTS-2/qwen0.6bemo4-merge`(模型目录里现成的本地 tokenizer,本地路径不触发 HF 查询) |
| 2 | deploy yaml 没打进 pip 包 | pyproject package-data 只含 `stage_configs`,`vllm_omni/deploy/*.yaml` 不在 wheel 里 | 起服务显式 `--deploy-config`(镜像内 `/app/vllm-omni/vllm_omni/deploy/…` 有源码副本,或用模型目录里的自定义副本) |
| 3 | hf-mirror 当天挂(Xet 同款报错) | `hf download` 配好 `HF_HUB_DISABLE_XET=1` 仍 `Local entry not found` | 不依赖在线下载:附属模型全部复用旧 index-tts 攒好的 `hf_cache/`(见 §3),`semantic_codec` 用 safetensors→pth 十行脚本转换 |
| 4 | amd64 机器拉 arm64 镜像报 no matching manifest | manager(238)是 x86,直接 pull 报错 | `docker pull --platform linux/arm64`(纯搬运不运行,架构随意) |
| 5 | 非交互 docker pull "卡住" | nohup 日志长时间停在 `Pulling fs layer` | 非 TTY 模式没有进度条,大层下载中无输出;看网卡计数器确认在动(实测 ~5MB/s,与拉上海 ACR 历史一致) |
| 6 | `docker logs -f \| tail` 永远无输出 | `-f` 不结束,tail 等 EOF | `docker logs --tail 30 -f <名>`,别加管道 |
| 7 | Mac tar 在 Linux 解包刷警告 | `Ignoring unknown extended header LIBARCHIVE.xattr.com.apple.provenance` | 无害,macOS xattr 而已 |
| 8 | 系统 python 读不了参考音色 wav | `wave.Error: unknown format: 65534`(WAVE_FORMAT_EXTENSIBLE) | 用容器里的 soundfile 读;引擎自身不受影响 |
| 9 | 启动期无害告警 | 版本 mismatch RuntimeWarning(dev 版本号解析不了)、chat template warmup 失败、`talker_config`/`speech_tokenizer` 告警 | 全部与 TTS 路径无关,忽略 |

预检结论(构建期即确认,避免上机白跑):`vllm/vllm-openai:v0.25.0` 官方带 arm64;torch 2.11.0+cu130 与 vllm kernel `.so` 均含 **sm_80**(A100 可用;FlashMLA 仅 sm_100,与 TTS 无关);依赖无 pynini(vllm-omni 前端用 cn2an/g2p,aarch64 无碍)。

## 5. 实测数据(A100 单卡,~4.7s 中文短句,含 base64 参考音色上传)

| 用例 | 结果 |
|---|---|
| 首请求(含附属模型懒加载) | 3.4s |
| 单请求稳态 ×5 | 1.67 / 1.79 / 1.81 / 1.87 / 1.90 s |
| 4 并发(= max_num_seqs) | 总 3.96s,等效 ~1.0s/句 |
| 8 并发(排队两批) | 总 7.22s,等效 ~0.9s/句,**5.2× 实时吞吐** |
| 峰值显存 | 24.6G / 40G |
| 8 并发产物一致性 | 时长 4.4-5.1s、RMS −34.6~−37.1 dBFS,无静音/削波/杂音 |

情感控制:`extra_params.emo_vector`(8 维,顺序**喜怒哀惧厌郁惊平**)+ `emo_alpha`。⚠️ 参数名与旧 server.py 不同(旧:`emotion_vector`/`emotion_weight`,新:包在 `extra_params` 里)。

## 6. 音量结论与响度归一化选型

- normal −36.0 dBFS vs angry −21.3 dBFS(差 ~15dB):**情感向量的自然表达**,非 bug。两边实现都不做响度归一化(vllm-omni 注释明确与官方 infer_v2 int16 保存前数值等价)。
- 生产统一音量:**pyloudnorm 放 M4 facade**(引擎无关,对以后所有语音模型统一生效)。实测:5s 台词全流程(读+测量+线性增益+峰值保护+写)**2.1ms/条**;30min 顶格文件(22.05k mono, 79MB)**0.70s / 峰值内存 1.38GB**(Mac 实测,ARM 估 1.5-3s)——可用。
- 不用 ffmpeg 的理由:单遍 loudnorm 是动态归一化,短句有抽吸感;双遍要 spawn 两次子进程。ffmpeg 仅在超长/高规格音频(流式恒定内存)才有必要。

## 7. 跑法(复现)

```bash
# 起服务(节点)
docker run -d --name vllm-omni-tts --gpus '"device=0"' --network host \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /nfs-models/wuhanjisuan894/models/IndexTTS-2:/models/IndexTTS-2 \
  crpi-…/reputationly/vllm-omni:arm64-a100-20260714 \
  vllm serve /models/IndexTTS-2 --omni --trust-remote-code --port 8092 \
    --served-model-name IndexTTS-2 --deploy-config /models/IndexTTS-2/indextts2-a100.yaml

# 请求(注意 emo 参数在 extra_params 里)
B64=$(base64 -w0 /nfs-models/wuhanjisuan894/models/IndexTTS-2/voices/char1.wav)
printf '{"model":"IndexTTS-2","input":"你到底想干什么！","response_format":"wav","ref_audio":"data:audio/wav;base64,%s","extra_params":{"emo_vector":[0,0.8,0,0,0,0,0,0],"emo_alpha":0.9}}' "$B64" > /tmp/req.json
curl -s http://localhost:8092/v1/audio/speech -H 'Content-Type: application/json' -d @/tmp/req.json -o out.wav

# 新节点从零到可跑(238 上):免密 + 基础环境(docker/NFS/toolkit,不入 GPUStack)
ssh-copy-id root@<节点IP>
scp /root/lx2v-node.sh root@<节点IP>:/root/ && ssh root@<节点IP> 'bash /root/lx2v-node.sh setup-base'
```

## 8. 缺口 / 待办测试

**已测完(结果见 §1.1 / §5 及各稳定性小节)**:
- ✅ 长文本边界(72/216 通过,324/500 崩引擎)· 情感三模式(vector/audio/text)· 英文/中英混排 · seed 复现性 · 300 句长跑 · max_num_seqs 4→8 对比

**引擎侧仍未测(选做,不阻塞选型结论)**:
1. `indextts2_low_latency.yaml` 低延迟配置对比(本轮只测了默认 yaml + max_num_seqs 4/8)
2. 长参考音色(>15s)、低质量/带噪参考音色的鲁棒性
3. 极限并发下的排队公平性与超时行为

**集成侧待测(换引擎、facade 落地后才谈得上"生产就绪",非引擎选型范畴)**:
1. facade 句级切分 + 拼接的真实听感(句间停顿、韵律连贯)
2. pyloudnorm 归一化后的实际听感
3. GPUStack 调度下的多实例 / 模型热切换 / 健康检查联动
4. 端到端(facade → 网关 → 引擎 → NFS → OBS)全链路验收

**⚠️ 生产准入硬约束(必须先做,否则不可上线)**:
1. **facade 句级切分**:input 按标点(。！？;换行)切句,逐句调引擎再拼接。这层本就是句级 TTS 的正常用法,不是补丁。
2. **长度硬闸**:每句再按字符/token 数兜底(建议 ≤180 字/句留安全边际),超限截断或 400 拒绝——一个超长请求都不许到引擎。
3. **容器 `--restart unless-stopped`**:万一漏网,自动拉起兜底(代价 ~5min 重启空窗)。测试容器用 `-d` 起(非 --rm,崩溃后 `Exited(0)` 可 `docker start` 恢复),但没配 `--restart`,生产必须加。
4. (可选给上游提 issue)超长文本应在 CPU 侧 token 校验时优雅拒绝,而非到 GPU 触发 device-side assert 杀死引擎。根因:gpt `max_text_tokens=600` 嵌入表越界。栈顶 `gpu_ar_model_runner.py:1036 → CUDA error: device-side assert triggered`。

**工程待办**:
1. GPUStack 注册 vllm-omni Custom 后端(照 lightx2v 模式;镜像/启动命令/端口注入)
2. M4 facade:复用 index-tts 仓 api_server 模块(TaskManager/TaskWorker),进程内 infer 改 HTTP 调网关 + 句级切分 + pyloudnorm + `loudness_lufs`/`gain_db` 参数
3. `lx2v-node.sh` 的 `setup-base` 子命令 commit(当前在 gpustack 仓工作区未提交)
4. CI 出包(现为 Mac 手工构建;照 index-tts 的 build-arm64.yml 复制一条流水线 + ACR 种 base)
5. 其余 4 台新节点(10.0.0.45/173/192/246 = a100-0019~0016)已 setup-base,待入集群

---

*本文为 vllm-omni 替换 index-tts 自研 serving 的验证实录,所有数据真机实测。上一篇:`index-tts/docs/indextts2-arm64-集成与踩坑记录.md`(旧引擎接入全记录,权重下载/hf_cache 布局仍然有效)。*
