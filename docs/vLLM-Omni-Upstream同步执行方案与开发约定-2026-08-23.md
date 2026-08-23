# vLLM-Omni Upstream 同步执行方案与开发约定

> 调查日期：2026-08-23
> merge-base：`593b4045391b701fa51b90d38c6f176caaba7a74`
> 落后 upstream/main：199 commit
> 触发原因：需要 upstream 已合入的 IndexTTS-2.5、MiniMax Music3，以及一批 H3/HunyuanImage3 性能改进
> 目标：follow upstream，同时保住本地对 H3 的架构级优化与其他现网模型的定制，让本次及以后的同步冲突尽量少

---

## 0. 结论先行

- 全量重叠面（我们改过 且 upstream 也改过的文件）**只有 50 个**，我们本地改动总面是 219 个文件，upstream 是 1254 个文件——绝大多数改动双方互不相干，天然零冲突。
- 50 个重叠文件里，**真正需要人工重新设计的只有 2 类问题**（Int8 量化统一、参考视频解码），其余是可机械处理的并集或需要一次性验证的项。
- **IndexTTS-2.5 与 MiniMax Music3 的模型代码本身，我们本地零改动、零分叉**——upstream 是唯一来源，合并这两块本身没有冲突，工作量在于三仓（vllm-omni / gpustack / new-api）联动接入，不在于合并本身。
- merge-base 之后的 87 个本地 commit，全部由 `reputationly` 账户提交，来源清晰，没有混入其他作者或历史遗留提交。

---

## 1. 全景分类表

| 区域 | 文件/能力点 | 分类 | 处理方式 |
| --- | --- | --- | --- |
| **零冲突** | `attention/backends/{abstract,flash_attn}.py`、`layers/{norm,rope}.py`、`config/model.py`、buildkite yml | 我方 merge-base 后零改动 | 直接 accept upstream |
| **纯净拿来** | IndexTTS-2.5 全部文件、MiniMax Music3 全部文件 | upstream 唯一来源 | merge 自动生效，无需处理 |
| **纯并集** | `data.py`（7 个新字段）、`pipeline_registry.py`（注册表） | 双方各自新增不同 key，无语义重叠 | 保留双方新增项，体力活 |
| **独立函数** | `diffusers_loader.py`（量化 checkpoint 判断泛化） | 双方改的是同函数不同区域 | 独立 helper 原样保留，改调用点变量名 |
| **纯增量（Ref2VA ~85%）** | `reference_image_geometry.py`、`ordered_references.py`、`keyframes.py`、`request_noise.py`、契约架构 `strategy.py` | upstream 无等价物 | 迁移工作量，非决策工作量 |
| **我方独有修复** | RoPE `inv_freq` 真值初始化（`minimax_h3_transformer.py`） | upstream 完全没碰这块 | trivial，原样保留 |
| **我方独有能力** | AdaLN 剪枝 checkpoint 加载（`minimax_h3_transformer.py`，227 行） | upstream 没做「加载不同 checkpoint 变体」这件事 | moderate，需验证 SwiGLU 融合兼容性 |
| **A100 专属** | VAE 分块反归一化（`vae.py`） | 与 upstream 的 tiling-hang 修复区域零重叠 | trivial-moderate |
| **需人工甄别** | 参考视频流式解码（`reference_media_decode.py` 等） | 目标重叠但我方方案更完整，upstream 方案会砸在我方已验收的 legacy 路径上 | **不能直接 accept upstream**，见 §2.2 |
| **混合：骨架重复+我方独有修复** | DMD2 少步调度（`time_request.py`） | `base_schedule` 部分双方独立收敛到同一接口；均匀调度分支我方多修了一个 off-by-one，upstream 没修 | accept upstream 骨架 + 手工搬回我方 bug fix |
| **A100 专属，需统一设计** | Int8 量化（`encoder.py` + `minimax_h3_transformer.py` 的 `weight_scale` 重排 + AdaLN 加载共享的重排函数） | 三处共享同一挂载点，必须合并成一个任务 | **severe**，见 §2.1 |
| **未深挖，需补查** | `sequential_backend.py`（+273 行，单 commit `c7e56d68`）、HunyuanImage3（NF4 TP4+EP4，文本零冲突但需验证语义兼容） | 本轮扫描新发现，此前三轮 fork 未覆盖 | 见 §3 |
| **大量但低风险** | `api_server.py`（我方 +1766 行：异步任务端点、进度上报、HunyuanImage3 上线） | 逐 commit 核实均为纯新增功能端点，与 upstream 改动区域不重叠 | 低风险，随标准 merge 处理 |

---

## 2. 需要人工设计的工作（执行合并前必须先想清楚）

### 2.1 Int8 量化统一 —— 好消息：不用新写，直接复用现成框架

**背景**：`encoder.py` 手搓了一个 `_Int8WeightMixin`（离线 Int8 weight-only，checkpoint 自带 `weight_scale`），绕开了 vLLM 标准的 `LinearBase`/`quant_config` 框架。同一批改动（`c7e56d68`）还把 DiT 主干 `minimax_h3_transformer.py` 的 `weight_scale` 重排逻辑接了进来，且这处重排函数与 AdaLN 剪枝 checkpoint 加载共享同一个调用点——三者必须放进同一个设计任务。

**关键发现**：vllm_omni 自己在 `vllm_omni/quantization/int8_config.py` 已经有一套生产验证过的 `Int8WeightOnlyLinearMethod`，**当前正在 H3 的 DiT 侧使用**。编码器没接这套框架是历史遗留（8/9 号有过一次接 `LinearBase` 做 NF4 量化的尝试，撞上 CPU-offload 场景连续请求 500 的问题 #44 没解决，两周后写 Int8 mixin 时绕开了 `LinearBase`，但没注意到 DiT 侧同期已经独立解决了同类问题）。

**推荐方案**：

1. 编码器 4 个线性类改接 `LinearBase`，`quant_config` 穿透构造函数（参照 `minimax_h3_transformer.py` 里已跑通的写法）
2. 删除 `_Int8WeightMixin`，改为标准 `self.quant_method = quant_config.get_quant_method(self, prefix)`
3. `weight_loader`/`scale_loader` 的 TP 分片逻辑（`_FUSED_SOURCE_SHARDS`）原样保留——这是业务逻辑，跟量化方法无关
4. `create_weights` 改用 `Int8WeightOnlyLinearMethod` 已有的 `ModelWeightParameter`/`ChannelQuantScaleParameter` 构造逻辑
5. **不需要**跟着换 upstream 新版权重加载签名（`_load_weights` 手搓 index.json 解析可以留着不动）——量化接入和加载管线签名是解耦的两件事
6. 需要单独确认：`quant_config` 怎么从部署配置下发到编码器，跟 DiT 侧现在的触发机制是复用同一开关还是独立开关

**额外收益**：迁移后自动获得 `@torch._dynamo.disable`（防 torch.compile 下 RecursionError）和 CPU-offload 感知缓存——这是编码器当前实现里缺失、且已知会在类似场景炸的两个坑，顺带修掉。

**风险与工作量**：代码改动量中等偏小。**最大风险点是必须补一轮 CPU-offload 开启 + 连续多请求的回归测试**（P1-1 当年就是在这个具体场景栽的），不能只信任理论上的可迁移性。方案本身可行性没有疑问。

### 2.2 参考视频流式解码 —— 不能整体倒向任何一边

**背景**：`prepare_reference_videos_official()`（我方，`acc6bf8f`）用纯 Python 流式解码同时喂 VLM 条件器和 VAE，解决 15s 4K 参考视频原先要吃 ~36 GiB×2 主机内存的问题，峰值跟生成几何走而不是源分辨率。这条路径完全不经过 legacy 的 `_transcode_reference_video`。

upstream（`1f65814f #5978`）改的是**唯一一条路径**（`_transcode_reference_video`），把编码换成无损 RGB（`libx264rgb/rgb24 + crf 0`）+ 硬截帧数，目标是「音频按生成时长夹取」和「VLM 与 VAE 看到像素一致」。

**冲突点**：这两个目标我方也独立解决了，但**upstream 没解决主机内存随源分辨率线性膨胀的问题**——它依然是先转码到一个无损中间文件再读，大源文件的中间文件本身就大。

**处理原则**：

- 保留我方 `prepare_reference_videos_official` 全套，upstream 无等价物
- **不要**把 upstream 对 `_transcode_reference_video` 的无损化改动当成 legacy 路径的自动更新直接吃进来——这两个函数在我方契约体系里是正式验收过、承诺「零 md5 漂移」的 legacy 路径，upstream 这处改动如果要采纳，必须作为独立的、需要重新跑 md5 对拍验收的 legacy 行为变更来对待，跟这次同步解耦
- `_encode_audio_conditions` 的 `max_duration_seconds` 参数：我方 official 路径已用不同方式做了同样的事，upstream 这段可以跳过

**难度**：moderate（不是 severe）——因为我方 official 路径本来就不经过 legacy 那几个函数，物理上不太会产生行级冲突，主要工作是人工确认而非重新设计。

---

## 3. 需要验证、但不需要重新设计的工作

| 项 | 内容 | 动作 |
| --- | --- | --- |
| DMD2 少步调度骨架 | `time_request.py` 的 `base_schedule` 参数双方近乎逐字符相同 | accept upstream 骨架（放弃我方延迟 import，除非验证启动性能真受影响），**手工搬回我方均匀调度分支的 N+1 off-by-one 修复**——这处容易被流程化 merge 误判成纯冗余而丢掉，需在合并 checklist 里显式标注 |
| AdaLN 剪枝加载 × SwiGLU 融合 | 我方按源权重名做的半序判定（加载期）与 upstream `SiluAndMul()` 融合算子（前向期）的输入布局约定是否一致 | 跑一次实际前向验证，理论上安全但需要实测确认 |
| HunyuanImage3 NF4 TP4+EP4 | 文本层面与 upstream 的 paged KV cache 重构（`72ee535f`/`0a1846aa`/`6ae7ff78`）零行号重叠，但我方 `HunyuanImage3Model` 新增代码可能间接依赖了 upstream 删除/重构的 `ImageKVCacheManager`/`TimestepEmbedder` 旧接口 | merge 后跑一次 NF4 TP4+EP4 部署冒烟测试，确认接口没断 |
| `sequential_backend.py`（+273 行） | 单 commit `c7e56d68`（MiniMax H3 A100 production optimizations 打包提交的一部分），本轮扫描新发现，尚未做三方语义比对 | 排进下一次深挖，或合并前单独跑一次三方 diff 确认无冲突 |
| `bitsandbytes_config.py`（+38 行） | `15d2d7bf` BnB `compress_statistics` 默认改 False，与已放弃的 P1-1 NF4 尝试相关 | 小改动，随标准 merge 处理，无需单独设计 |
| VAE 分块反归一化 | upstream 保留了原始未分块的 `revert_tensor` 尾部代码 | 把尾巴替换成 `_revert_frames()`，注意 `get_dit_group()` → `get_world_group().device_group` 的并行 API 改名是否有其他调用点要同步改 |

---

## 4. IndexTTS-2.5 / MiniMax Music3 的实际接入工作量

模型代码本身零冲突，工作量在三仓联动：

### IndexTTS-2.5

vllm-omni 侧 merge 即可用（`INDEXTTS25_PIPELINE`、`indextts2_5.yaml` 部署配置、`IndexTTS25Adapter` 均已在 upstream 完整实现）。gpustack / new-api 侧需要的改动此前已单独讨论过（`speed` 异步链路透传、`lang`/`text_normalization` 折进 `extra_params`、体验区语言/语速控件、情感来源四选一），属于另一条独立工作线，跟这次 merge 解耦执行。

### MiniMax Music3

vllm-omni 侧模型代码（`model_executor/models/minimax_music3/`、`deploy/minimax_music3{,_2gpu}.yaml`、`tts_adapters/minimax_music3.py`）merge 后直接可用。**2026-08-23 调研已确认 upstream 实际的 API 形态，并把 gpustack/new-api 的具体改法定下来了**，与 8/16 报告的假设不同，详见下方。

**更正 1：API 形态是 TTS 风格，不是异步 music 任务**。upstream 把 Music3 做成 `tts_adapters/minimax_music3.py`，走 **`/v1/audio/speech`**——`input` 承载歌词，`instructions` 承载风格/编曲描述，没有 speaker、没有参考音频、没有 temperature（采样由 checkpoint 固定，guidance 1.5 + seeded top-k 50，不支持的采样参数直接拒绝而不是静默忽略）。8/16 报告 §6.1 设想的「新增 `/v1/tasks/music/` 端点」路径**不适用**——upstream 把它当成了一个 TTS 模型，不是 ACE-Step 那种异步 music 任务。

**更正 2：cu128 完全够用，没有性能罚单**。upstream 的原生移植版本部署配置显示 stage 0（8B AR planner）是 `dtype: bfloat16`，stage 1（flow matching + DAV 解码器）是 `dtype: float32`（注释：flow-matching solver 在 bf16 下精度会明显退化）——**全程没有 int8/fp8，代码和部署配置里搜不到任何量化痕迹**。8/16 报告里"int8 加速核需要 cu130+"的结论，针对的是 ComfyUI 社区放出的 int8 量化权重和 Sogni 测试用的那条路径，**跟我们实际要走的 vLLM-Omni 原生移植是两条不同的路，不适用**。基础镜像 `vllm/vllm-openai:v0.26.0` 我们本地和 upstream 版本号一致，这次同步不涉及基础镜像升级，两个新模型跑的是今天已经在生产上跑 H3/图片模型的同一套 cu128 镜像栈。**升级 cu130 这件事跟接入 Music3 完全脱钩，不再是这次工作的前置项。**

**是否顺带把 ACE-Step 也改成 `/v1/audio/speech`：不建议**。技术上异步机制两边通用不是障碍，但 ACE-Step 的能力形状跟 TTS 契约对不上——cover/repaint 需要参考音频（音频进音频出，不是文本进音频出），还有 stem 分离、多轨叠加、批量生成、精确时长控制，这些在 TTS 契约里没有干净的落点，硬套会做成到处开洞塞非标准字段的四不像。ACE-Step 现在的 `/v1/tasks/music/` 是为它的能力形状专门设计的，且是 `ACE-Step-1.5` 一个完全独立的代码仓，改造成本要到那边的服务层重写，收益（少一条路由分支）覆盖不了这个成本。维持现状。

#### gpustack 侧具体怎么改（已查清架构，多数是配置而非代码）

查了 `gpustack/routes/videos.py`、`gpustack/scheduler/scheduler.py`、model-catalog.yaml 现有条目，结论：

1. **路由：不用改代码，Music3 走 `task_type="tts"` 即可**。`_engine_kind()`（`videos.py:493`）纯按 `task_type` 派发，实际提交 URL 是 `f"v1/tasks/{kind}/"`（`videos.py:1379`）。全仓搜索确认 **vLLM-Omni 根本没有 `/v1/tasks/music/` 端点**（那是 ACE-Step 自己引擎的 API），所以 Music3 结构上不可能走 `_MUSIC_TASK_TYPES`。它要走 `_AUDIO_TASK_TYPES = {"tts"}` → `POST /v1/tasks/audio/`——这条路由已存在，实现是 `create_audio_task`（`api_server.py:4089`），内部用的正是 `Omnispeech` handler，跟 `/v1/audio/speech` 同一套 `tts_adapters` 注册表（现在已包含 `minimax_music3.py`）。**这条路由现成能用，零代码改动。**

2. **资源选择器：不用改代码**。`model.backend == BackendEnum.VLLM_OMNI` 统一走 `VLLMOmniResourceFitSelector`（`scheduler.py:496`），该选择器本就支持按 catalog 条目的 `gpu_selector.gpus_per_replica` 决定卡数（1 卡给小 TTS 模型，2 卡给 8B MOSS 系列）。Music3 两个 stage 都在 `devices: "0"`，显存 0.6+0.2 吃一张卡，`gpus_per_replica: 1` 即可，选择器逻辑无需改动。

3. **分类：加一行配置，不新增代码路径**。`_VLLM_OMNI_CATEGORY_HINTS`（`scheduler.py:724`）按模型名/来源子串匹配决定 `CategoryEnum`，默认落 `TEXT_TO_SPEECH`。**已确认 category 和请求路由完全解耦**（`videos.py` 全文搜不到任何按 `CategoryEnum` 做路由判断的代码，只有 `task_type` 参与路由）——所以可以放心把 Music3 标成 `CategoryEnum.MUSIC`（跟 ACE-Step 一样，UI 里正确归类到"音乐"而不是混进语音模型列表），同时路由仍然走 `tts` 桶，两件事互不影响。改法：给 `_VLLM_OMNI_CATEGORY_HINTS` 加一条 `("music3", CategoryEnum.MUSIC)`（用实际的模型名/repo_id 子串确认一下，避免误匹配）。gpustack-ui 侧 `MUSIC` 分类已经在为 ACE-Step 服务，不需要新增前端代码。

4. **model-catalog.yaml：新增条目**，参照 `MOSS-VoiceGenerator` 的模板（同为纯文本、无参考音频、走 vLLM-Omni 的 TTS 类模型，最接近 Music3 的形状）：`categories: [music]`、`backend: vLLMOmni`、`gpu_selector.gpus_per_replica: 1`、`backend_parameters: [--trust-remote-code, --deploy-config, /deploy-configs/minimax_music3.yaml]`（部署配置文件名需要跟镜像里实际打包的路径核对）。

5. **延迟/排队参数：需要单独配置，容易被忽略**。`_DEFAULT_AUDIO_LATENCY = 20`（秒，`videos.py:483`）是按 IndexTTS-2 那种几秒钟出一句话的 TTS 校准的，Music3 是分钟级生成，直接吃这个默认值会导致 `_check_admission` 的排队预估严重失真（可能提前限流，也可能放太多并发进去）。已确认有现成的按模型覆盖机制——`lightx2v_model_latency_seconds` / `lightx2v_model_queue_wait_seconds`（`config.py:207,221`，按模型名子串大小写不敏感匹配），给 Music3 加一条覆盖值即可，不用改代码，但**必须在上线前配置，容易漏掉**。

#### new-api 侧具体怎么改

走通 gpustack 的 `tts` 路由之后，new-api 侧原来设想的 `t2m/cover/repaint` 异步任务代码（`adaptor.go:811` 的 t2m 推断、`materializeMusicInputs`、`MusicModelConfig`）**全部不需要碰**——Music3 走的是 `IsOmniTTSModel` 那条既有分支：

- `IsOmniTTSModel`（`adaptor.go:1317`）需要加一个 Music3 的模型名子串匹配（目前只认 qwen3-tts/voxcpm/cosyvoice/glm-tts/moss）。不加的话会落进 `else` 分支走 `materializeTTSInputs`——那条路径要求 `voice` 参考音色必填，Music3 根本没有这个概念，会导致请求被错误拒绝
- `inferTaskType` 里 `acestep` → `"t2m"` 的兜底规则不受影响，Music3 模型名不会匹配这条，没有误判风险
- 歌词长度：Music3 典型 caption 长度可达数千字符（upstream 上限 5000 token / 20000 字符），远超普通 TTS 台词长度，需要用 `AudioModelConfig` 现成的按模型 `maxChars` 覆盖机制单独放宽，不能用 TTS 默认字数上限

#### 体验区放哪：2026-08-23 决定——放进「文生音乐」，按模型做特殊处理，不新开入口

放进现有的音乐 playground（`musicPlayground`/`useMusicGeneration.js`，ACE-Step 现在用的那个），跟 ACE-Step 同一个用户可感知的功能里选模型切换，不混进 `audioPlayground` 那 4 个语音玩法 tab，不新开独立入口。

这意味着**特殊处理不能按 task_type 分支，要按"选中的具体模型是不是 Music3"分支**——这跟 `_backfill_h3_engine_params`（`gpustack/routes/videos.py:855`）"gated on the resolved model's BACKEND/name，不是 task_type"是同一个模式，可以照抄这个思路。具体要处理的点：

- **提交时强制 `task_type="tts"`**：无论前端音乐 playground 内部怎么表达"这是一次文生音乐"，选中 Music3 时下发给 gpustack 的必须是 `task_type="tts"`，不能沿用 ACE-Step 的 `t2m`——这是路由能不能走通的硬要求（见上方"更正 1"）。这个覆盖建议放在 new-api 侧做（前端不用感知 task_type 差异，后端按模型名判断后改写），对齐现有 `_backfill_h3_engine_params` 那种"按 model 而非 task_type 特判"的既有模式。
- **UI 控件按模型隐藏/调整**：
    - 参考音频（cover/repaint 需要）：Music3 完全不支持，选中它时要隐藏，不能像 ACE-Step 一样露出上传入口
    - 时长语义不同：ACE-Step 是精确命中（10-600s），Music3 是上限而非目标（模型自己决定何时收尾，实测同样要 2:30 的歌出来是 2:06-3:00 不等，见 `docs/MiniMax-Music3-接入调研与路线选型报告.md` §3.4）——选中 Music3 时时长控件文案要改成"最长时长"一类的措辞，不能让用户以为能精确控时长
    - 风格描述的输入方式：Music3 的 caption 是结构化的（Global Metadata / Vocal Details / Arrangement 三段式），比 ACE-Step 的自由 prompt 更挑格式，UI 上可能需要引导文案或分段输入框，不能直接照搬 ACE-Step 的单一文本框体验
    - stem 分离/多轨叠加/Vocal2BGM/LRC 时间戳/批量生成：Music3 全部不支持，这些控件选中它时要隐藏

- **new-api 后端**：`IsOmniTTSModel` 加 Music3 模型名匹配（见上方"new-api 侧具体怎么改"），物化函数走 `materializeOmniTTSInputs`（无参考音频时返回 nil，天然适配 Music3 的纯文本输入），不能走 `materializeMusicInputs`（那条路径是为 ACE-Step 的 cover/repaint 设计的，会去找并不存在的参考音频字段）

这块前端改动量比 gpustack/new-api 那两层加起来还大，值得单独拆一个任务来做，不建议跟这次 upstream 同步的其他工作混在一起排期。

**已知的部署前置**：`--deploy-config` 对未注册模型静默失效，Music3 进 `OMNI_PIPELINES` 之前传 `--deploy-config` 会被整体丢弃回落单卡默认档（H3 踩过的同款坑）。

---

## 5. 执行阶段划分

不建议直接在 `main` 上一次性跑完整 merge。按阶段走，每阶段验证通过再进下一步：

1. **开集成分支**（复用或新开），先合 upstream 到分支上，不动 main
2. **机械处理零冲突/纯并集文件**——§1 表格里「零冲突」「纯并集」「独立函数」三类，占大头，风险最低，先扫掉
3. **Int8 量化统一**（§2.1）——设计已经清楚，实现 + CPU-offload 回归测试
4. **参考视频解码人工甄别**（§2.2）——按处理原则手工核对，不整体倒向任何一边
5. **§3 的验证项**——AdaLN/SwiGLU 前向验证、HunyuanImage3 NF4 TP4+EP4 冒烟、`sequential_backend.py` 补查、DMD2 off-by-one 手工搬回
6. **IndexTTS-2.5 / Music3 三仓联动**（§4）——与 3-5 步文件不重叠，可并行推进
7. **全量回归**：H3（Ref2VA + 剪枝档 + Turbo LoRA 全套现有测试）、HunyuanImage3、其他现网模型的标准冒烟，通过后集成分支合回 main

---

## 6. 面向未来的开发约定

这次比对暴露的规律很清楚：**冲突量不是由「改了多少行」决定的，是由「改在哪」决定的**。Ref2VA 走独立新文件 + 单点挂载，85% 零冲突；encoder.py 手写在共享类内部，撞上 upstream 同期改动，成了本次唯一的 severe 项。把这个规律固化成习惯：

1. **新能力优先新文件 + 单点挂载，而不是散改共享函数内部**。哪怕逻辑上跟某个共享函数强相关，也尽量把新逻辑封装成独立函数/类，共享函数里只留一行调用——upstream 改实现细节时，这一行调用大概率还在。

2. **涉及 vLLM 标准抽象（量化/并行/权重加载）时，先查有没有现成扩展点，再决定要不要手写**。encoder.py 的教训：如果当时先查一下，会发现 DiT 侧的 `Int8WeightOnlyLinearMethod` 直接能用。用同一套抽象、只是配置不同，天然比另起一套实现更容易合并。

3. **纯 bug fix（不含业务定制）考虑推回上游**。RoPE `inv_freq` 垃圾内存初始化、DMD2 均匀调度 off-by-one，都是跟 A100/业务无关的通用正确性问题，upstream 现在还带着这两个坑。留在本地仓意味着每次同步都要重新判断「这处该不该被当成冗余丢掉」；PR 被接受后这个冲突点从「需要人工核对」变成「彻底消失」。

4. **开启 `git config rerere.enabled true`**。已确认几处会在每次同步时反复出现的相同冲突模式（DMD2 那段、sigma schedule 的近乎重复实现），rerere 记录第一次的人工解决方式，下次自动应用。

5. **同步节奏改成小步快跑**，不要再攒到 199 个 commit。这次能做细致溯源，很大程度是因为逼到不得不做——落后越多，任何一处冲突都要靠翻 commit 历史「考古」才能搞清楚当初为什么这么改。建议 2-4 周同步一次，每次落后的 commit 数少一个数量级，冲突面小、上下文新鲜。

6. **维护 §7 的已知分叉点清单**，下次同步先查这份清单，不用重新做一遍取证。

---

## 7. 已知分叉点清单（本次调查固化结果，供下次同步复用）

| 文件/能力 | 我方改动性质 | 下次 upstream 再动这块时要注意什么 |
| --- | --- | --- |
| `minimax_h3/encoder.py` | Int8 weight-only 量化（若 §2.1 已执行，应已改为标准 `LinearBase`/`quant_method`） | 若已迁移完成，此项应从清单移除；若未迁移，upstream 任何触碰这 4 个线性类的改动都要重新走一遍语义比对 |
| `minimax_h3_transformer.py` 的 `weight_scale` 重排 | 与 encoder.py 同源的 Int8 支持，和 AdaLN 加载共享调用点 | 同上，跟 encoder.py 一起追踪 |
| `minimax_h3_transformer.py` 的 AdaLN 剪枝加载 | 支持 Turbo/剪枝档 checkpoint 的字段别名解析、权重改名、QKV 拆分 | upstream 若重构 `load_weights` 方法体或 `_reorder_grouped_qkv_to_qkv`，需重新核对这块 |
| `minimax_h3_transformer.py` 的 RoPE `inv_freq` | 真值初始化替代 `torch.empty` | upstream 若改 `MiniMaxH3Rope` 类本身，需确认真值初始化没被覆盖回未初始化状态 |
| `vae.py` 的分块反归一化 | 主机内存优化，与 upstream 的 tiling-hang 修复物理区域不同 | upstream 若再动 `decode()` 尾部（`revert_tensor` 之后），需重新核对挂载点 |
| `pipeline_minimax_h3.py` / `reference_video.py` 等 12 个新文件 | Ref2VA 双契约架构（legacy/official_diffusers_v1） | upstream 若开始做类似的参考图/参考视频能力，优先判断能不能收敛到同一套接口，而不是继续两条线并行 |
| `_transcode_reference_video`（legacy 路径） | 我方承诺零 md5 漂移的验收路径 | upstream 对这个函数的任何改动，都不能自动合并，必须走独立的、需要重新对拍验收的流程 |
| `time_request.py` 的均匀调度分支 | N+1 off-by-one 修复 | upstream 若修了这个 bug，此项从清单移除；若没修，每次同步都要防止被误判成冗余丢掉 |

---

## 8. cu128 → cu130 驱动升级现状排查（与本次同步解耦，另立项，2026-08-23 决定下周统一做）

排查动机：确认这次同步 IndexTTS-2.5 / MiniMax Music3 是否会被集群 CUDA 版本卡住。**结论：不会**（见 §4 Music3 更正 2），驱动升级不再是这次工作的前置项，但排查过程中拿到的现状值得记下来，供下周单独立项时直接用。

### 8.1 现网三个引擎的 CUDA 版本现状（本地仓库核实）

| 引擎 | 跑什么 | 生产实际用的镜像 | CUDA 版本 |
| --- | --- | --- | --- |
| **vLLM-Omni** | H3、图片模型、语音模型 | `vllm/vllm-openai:v0.26.0`（外部基础镜像，版本由 vLLM 官方项目控制） | 待查具体数字，但已确认这次同步不涉及基础镜像升级 |
| **LightX2V** | SeedVR2 超分 | `arronlee/lightx2v:arm64-cu128`（`Dockerfile_aarch64_app` 里 `BASE_IMAGE` 默认值） | **cu128** |
| **ACE-Step** | 音乐生成（t2m/cover/repaint） | `Dockerfile` 里 `ARG CUDA_VERSION=12.8.1` | **cu128** |

三个引擎目前全部锁在 cu128，跟宿主机驱动 570.86.10 的能力上限一致——驱动一动，三个引擎的镜像都要重新构建 + 验收，不是只改 vllm-omni 就完事。

**意外发现**：LightX2V 仓库里已经有一份 `Dockerfile_cu130`（base image `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`），但没有接进 arm64 生产构建脚本——之前有人做过一半探索，没收尾，下周可以直接从这个当起点。

### 8.2 下周排这件事的建议顺序

1. 实际连机器核对当前驱动版本（文档记的是 570.86.10，机群 30-50 台建议抽查 2-3 台而不只信一台；本次排查曾尝试连 gpu34 核实但因本机 macOS 缺 `timeout` 命令未完成，需重跑）
2. 查一下 vLLM 官方 `v0.26.0` 镜像具体钉的是哪个 CUDA 版本
3. 把 LightX2V 那份 `Dockerfile_cu130` 接进 arm64 构建流程，跑一次冒烟
4. 找一台机动池里空着的机器（`docs/200卡集群部署重规划-2026-08-22.md` §3.5 机动池 0021-0033 里有几台是空的）当金丝雀，先升这一台的驱动，三个引擎的镜像都在这台上过一遍验收，再决定要不要铺开
5. 升驱动是整个集群的事，不是为一个模型能做的决定——完成前，Music3/IndexTTS-2.5 的上线不应该等它

---

## 附录：参考文档索引

| 内容 | 位置 |
| --- | --- |
| H3 官方 Diffusers 流程对齐任务书 | `docs/MiniMax-H3-官方Diffusers流程对齐开发与验收指南.md` |
| MiniMax Music3 接入调研（2026-08-16，决议已于本次重启） | `docs/MiniMax-Music3-接入调研与路线选型报告.md` |
| IndexTTS2 vLLM-Omni 实验测试报告 | `docs/IndexTTS2-vLLM-Omni-实验测试报告.md` |
| H3 剪枝/量化相关交接文档 | `docs/minimax-h3-INT8剪枝叠加-交接-2026-08-20.md`、`docs/minimax-h3-ref2va-剪枝量化全量实测-2026-08-22.md` |
