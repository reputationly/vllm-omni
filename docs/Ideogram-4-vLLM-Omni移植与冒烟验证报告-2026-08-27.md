# Ideogram-4 vLLM-Omni 移植与冒烟验证报告（2026-08-27）

> 前置调研：`docs/Ideogram-4-接入调研与路线选型报告.md`（2026-08-15，结论：走 vllm-omni 移植，
> 从上游 PR #4227 起步）。本报告记录把 PR #4227 实际拉进本仓库、跑通端到端冒烟测试的完整过程，
> 包含发现并修复的 7 个真实 bug，以及模型自带安全过滤的实测行为。
> 授权说明：本次适配面向**内部员工免费使用，不涉及对外商用计费**，因此不受调研报告 §5 的
> Ideogram Non-Commercial 商用授权限制约束（该结论按用户指示不再作为本次工作的阻塞项）。

---

## 1. 一句话结论（最终版）

**PR #4227（上游 `vllm-project/vllm-omni`，作者 RuixiangMa，`CHANGES_REQUESTED`，只有 bot 要证据的评论，从未有人真正跑通过）拉进本仓库后，端到端冒烟测试 + 事后静态代码审查发现并修复了 7 个真实 bug，目前已经能在单卡/双卡 A100-40G 上用 NF4 权重稳定生成图片**。

**⚠️ 重要方法论说明（用户要求做纯静态代码审查后新发现，直接影响下面所有"固定 seed 对照"实验的可信度）**：§4.7 发现 `seed` 参数在本次冒烟测试期间从未真正生效——每次生成实际用的都是进程级未播种的随机数流，跟请求里填的 seed 数字无关。这意味着 §5/§6 里所有号称"固定 seed 做 A/B 对照"（skip vs 不 skip、三档预设对比）的实验，**两侧其实是独立随机抽样，不是同一噪声起点的严格配对对照**。这个 bug 已经修复并验证（同一 seed 两次调用现在 SHA256 完全一致），但下面 §5/§6 的具体结论是在修复**之前**跑出来的，方向性结论（skip 能缓解拦截、V4_DEFAULT_20 更可靠）大概率仍然成立，但**统计严谨度弱于原文措辞暗示的程度**，建议后续用现在真正生效的 seed 重新跑一遍关键配对实验再最终拍板。见 §4.7 详细说明。

**⚠️⚠️⚠️ 最终根因结论（推翻本报告绝大部分早期安全过滤/跑题分析）：所谓的"安全过滤误伤"和"skip 导致乱码跑题"，主要根因是 prompt 用了错误的 JSON 格式，不是模型或 skip 参数本身的问题**。本报告 §5-§6.4 的全部测试，从头到尾用的都是一个简化写法 `{"subject":..., "style":..., "scene":..., "palette":[...]}`，只是"看起来像 JSON"，跟模型真正训练对齐用的官方结构化 caption schema（`high_level_description` + `style_description` + `compositional_deconstruction.background/elements[]`，每个 element 带 `bbox`/`desc`/`palette`）完全不是一回事。用户看到社区一个 ComfyUI 插件流传"用结构化分解格式能减少误拦"的说法后要求排查，改用官方完整 schema 重测 §6.4/§5.5 里全部 9 个原本 100% 出问题的案例（含"完全拦截"和"skip 后乱码"两类），**在 `skip_first_n_steps=0` 和 `skip_first_n_steps=1` 两种设置下合计 18 组测试，全部正常，没有一次拦截、没有一次乱码**。额外验证了 compositional 分解不需要凑够特定数量的 element（1/2/3/4 个都测过，全部正常），排除了"元素数量"这个变量，坐实了"用没用官方结构化格式"才是关键。详见 §5.6，这是本报告最重要的单条发现。

**据此推翻/大幅削弱的历史结论**：①§5.5"skip 只有约 1/4 概率换来真实内容"——这个统计是在错误格式下测出来的，格式修好之后 skip=1 的干净率大幅回升（§5.7：8 组敏感边界内容里 7 组两种 skip 设置都干净）；②§6.4"约 30% 固有失败率，需要输出检测+重试兜底"——同样是错误格式下的产物，真实的现网失败率大概率远低于这个数字。**唯一没有被推翻的残留风险（§5.7）**：即使格式正确，`skip_first_n_steps=1` 仍有低概率（本次 8 组里 1 组）引入局部渲染损坏，所以默认仍建议 `skip_first_n_steps=0`。**新发现的安全能力边界（§5.8）**：模型自带安全对齐只覆盖 NSFW（色情）类内容，明确不拦截暴力/血腥内容（6 组图形化暴力测试在 `skip=0` 官方原生路径下全部通过、无一例外），如果业务对暴力内容有合规要求，需要额外接入 Hive 视觉审查或等效方案。

**最终建议的优先级顺序（详见 §5.6/§6.5）**：① 门面层必须实现"自由文本 → 官方结构化 JSON schema"转换层——这不再是可选的 `magic_prompt` 锦上添花项，而是消除误拦/乱码的关键路径；② 在此基础上默认 `skip_first_n_steps=0`；③ 仍保留"检测 Image blocked 固定文案"的兜底逻辑作为防御性最后一道保险；④ 如果业务需要过滤暴力内容，额外接入 Hive 视觉审查或内部等效方案。

**采样预设结论（方向性仍然成立，具体数字待用正确格式重新采集）**：官方三档预设（`V4_QUALITY_48`/`V4_DEFAULT_20`/`V4_TURBO_12`）里，**`V4_DEFAULT_20`（20 步）此前用错误格式测试时表现相对最可靠**，这跟 artificialanalysis.ai 榜单上"Ideogram 4.0"（基础档）反超"Ideogram 4.0 (Quality)"的公开评测结果方向一致。但由于 §5.6 发现的格式问题会拉低所有预设的成功率，§6.4 那张"V4_DEFAULT_20 86%/其他档更低"的对比表统计意义有限，**建议后续用官方结构化格式重新采集三档对比数据**。分辨率仍然优先 1024×1024/2048×2048（官方明确验证过的方形档位）。详见 §6.4。

另外实测了 GPU 并行配置：**TP=2 比 TP=1 快 17%、省显存 19%，双重占优**（Ideogram4 用的是真正的 vLLM 并行层，不是 Krea2 那种"假 TP"）；**TP=4 架构上不可能**（18 个注意力头不能被 4 整除）。4096×4096 会真实 OOM，2048 及以下不会 OOM。详见 §6。

还有 1 项已知但未修的性能优化项（稠密 attention mask 未做 varlen 化，纯性能，非正确性问题，§4.4）。**prompt 结构化转换层（自由文本 → 官方 JSON schema）尚未在门面层实现（§8）——这是 §5.6 发现后优先级最高的遗留工作，不再是可选项**。

---

## 2. 权重与代码来源

| 项 | 内容 |
| --- | --- |
| 权重 | `/nfs-models/wuhanjisuan894/models/Ideogram-4-NF4`（16GB，此前已下载，layout 完整：`transformer/`/`unconditional_transformer/`/`text_encoder/`/`vae/`/`tokenizer/`） |
| 官方 harness | `github.com/ideogram-oss/ideogram4`（Apache 2.0，本地 clone 于 `/Users/reputationly/Desktop/code/api/ideogram4`） |
| 移植起点 | 上游 PR [`vllm-project/vllm-omni#4227`](https://github.com/vllm-project/vllm-omni/pull/4227)（`+2056` 行，`CHANGES_REQUESTED`，唯一评审意见是要求补 e2e 出图证据，未提出架构层面的问题） |
| 拉取方式 | `git fetch origin pull/4227/head:pr-4227` 后手动拷贝核心文件到本仓库，**不是**直接 merge（本仓库领先上游 46+ commit，很多内部机制已重构，见下） |

拷入本仓库的文件：

```text
vllm_omni/diffusion/models/ideogram4/{__init__.py,pipeline_ideogram4.py,ideogram4_transformer.py}
tests/diffusion/models/ideogram4/*.py
recipes/Ideogram/Ideogram4.md
vllm_omni/diffusion/registry.py   （手动合并，加 Ideogram4Pipeline 注册项）
```

**跳过未套用的补丁**：PR 自带的 `vllm_omni/quantization/factory.py` 改动（给 `_load_in_4bit`/`_load_in_8bit` 做黑名单过滤）**没有采用**——PR 那个版本的 `factory.py` 还没有 `_SERIALIZED_BNB_MARKERS`/`_construct_override` 这套离线量化检测机制（本仓库为 HunyuanImage-3 独立加的），如果原样套用，会导致 NF4 checkpoint 的 `_load_in_4bit` 标记被提前剥离，从而被错误地当成"在线量化"而不是"离线预量化"处理——这正是本仓库代码注释里明确记录过、在 HunyuanImage-3 上真实踩过的坑。本仓库现有机制已经能正确处理，不需要这个补丁。

---

## 3. PR #4227 vs 官方 harness 的忠实度评估

| 维度 | 官方 harness | PR #4227 | 结论 |
| --- | --- | --- | --- |
| 双 DiT（conditional + unconditional） | ✅ | ✅ 忠实移植 | 对 |
| 非对称 CFG（负分支只吃图像 token，`neg_llm_features` 置零） | ✅ | ✅ | 对 |
| 13 层激活抽取（`QWEN3_VL_ACTIVATION_LAYERS` 常量） | ✅ | ✅ 逐字一致 | 对 |
| 逐步 guidance schedule | ✅ | ✅（`guidance_schedule` 参数） | 对 |
| 稠密 attn mask → varlen | 需要改（§3.1 原调研报告已指出） | **没有改**，`_build_segment_mask` 跟 harness 的 `(segment_ids.unsqueeze(2)==segment_ids.unsqueeze(1))` 一字不差，只是包了层 vllm-omni 的 `AttentionMetadata` 壳 | **遗留问题，见 §4.4** |
| Prompt JSON 格式化 / `magic_prompt` | 有（`CaptionVerifier` + 结构化步骤） | 完全没有 | **遗留问题，见 §5.3** |
| Cache-DiT 接入 | — | `may_enable_cache_dit` 导入路径已过时 | 已修复，见 §4.1 |
| 量化 `quant_config` 传参 | — | 传了参数但硬编码 6 处 `quant_config=None`，从未真正接到线性层 | 已修复，见 §4.2 |
| NF4 显存预算 | 官方 ~16GB | 反量化到 BF16 常驻显存，双 DiT 直接 37GB+ 撑爆 40G 卡 | 已修复，见 §4.3 |
| 人工 code review | — | 只有一条 bot 评论要证据，**没有人真正评审过代码质量** | 印证了原调研报告"缺 e2e 证据不是架构问题"的判断是错的——**代码本身也有真实 bug，只是没人跑过发现不了** |

---

## 4. 冒烟测试排查全过程（7 个真实 bug）

测试环境：gpu45（A100-40G×4，取 1 张卡 `device=1`），把本地改动过的 `vllm_omni/diffusion/models/ideogram4/` 和 `registry.py` bind-mount 进容器覆盖镜像里的安装路径，不用重新出镜像即可迭代验证。

### 4.1 Bug #1：`may_enable_cache_dit` 模块路径已过时

```text
ModuleNotFoundError: No module named 'vllm_omni.diffusion.cache.cache_dit_backend'
```

PR 里 `from vllm_omni.diffusion.cache.cache_dit_backend import may_enable_cache_dit` 引用的是上游 PR 分支当时的模块布局；本仓库已经把 Cache-DiT 重构到 `vllm_omni/diffusion/cache/cachedit/`（`backend.py`/`config.py`/`runtime.py`/`model_specific.py`）。而且 Ideogram4 有两个独立 transformer（conditional + unconditional），标准的单 transformer Cache-DiT enabler 本来就不适用——即使模块路径对了，`may_enable_cache_dit` 对这个模型也只会返回 `None`。

**修复**：删掉这个 import，直接 `self._cache_backend = None`，效果等价，代码更直接。

### 4.2 Bug #2（最隐蔽）：`quant_config` 参数传了但从未真正接到线性层

排查过程：

1. 不传 `--quantization` → 权重加载到一半 OOM（`Tried to allocate 4.57 GiB ... 35.19 GiB memory in use`）
2. 传 `--quantization bitsandbytes` → **OOM 数字一模一样**，说明这个参数根本没生效
3. 查日志确认 `factory.py` 确实 `Building quantization config: bitsandbytes` 被调用了，说明量化配置本身构建成功，问题出在更下游
4. `grep quant_config=None` 在 `ideogram4_transformer.py` 里找到 **6 处**：`QKVParallelLinear`/`RowParallelLinear`/`ColumnParallelLinear`/`ReplicatedLinear` 的实际构造调用全部硬编码 `quant_config=None`，而外层函数签名明明收了 `quant_config` 参数、也逐层往下传——传到最后一步被弃用

**修复**：把 6 处 `quant_config=None` 全部替换成 `quant_config=quant_config`（用真正收到的参数），其中 **1 处例外**：`Ideogram4FinalLayer`（最终速度场输出投影层）保持 `quant_config=None`，因为它在 checkpoint 里就是全精度存储（尝试量化后加载报 `Tried to load weights of size [128, 4608] to a parameter of size [294912, 1]`，即打包后的 4-bit buffer 形状对不上未打包的全精度 checkpoint 张量）——这是典型的"最终输出头保持全精度"做法，类似 LLM 里 `lm_head` 常见的不量化处理。

### 4.3 Bug #3：加载路径设计上就无法在 40G 卡上跑进 16GB

即使修完 Bug #2，OOM 依然复现且数字不变。深挖 `_preprocess_nf4_weights`（`ideogram4_transformer.py:609`）发现：这个函数会在 Python 里用 `bitsandbytes.functional.dequantize_4bit` 把 checkpoint 里打包的 NF4 张量**逐个反量化成 BF16**（每个反量化完立刻 `.cpu()`，避免同时持有全部反量化结果），然后再交给 `load_weights` 逐层加载——**但线性层本身如果没有量化配置，接收到的就是完整 BF16 张量并原样常驻显存**。双 DiT 共 18.6B 参数 × 2 字节(BF16) ≈ 37.2GB，加上 8B 文本编码器，必然撑爆 40G 卡，这是**设计层面的问题**，不是配置能绕过的。

一度尝试参考 HunyuanImage-3 的专用 loader（`HunyuanImage3BitsAndBytesModelLoader`，继承 vLLM 原生 `BitsAndBytesModelLoader`，直接读打包权重不做 Python 侧反量化）——但那套是为 MoE 专家并行设计的，复刻到 Ideogram4 的双-dense-transformer 场景工作量过大。

**实际采用的修复**（更小改动，利用现有反量化步骤）：给 `--quantization bitsandbytes` 传参启用**在线量化路径**（`DiffusionBitsAndBytesConfig`/`BnBOnlineLinearMethod`）。效果：`_preprocess_nf4_weights` 仍然逐个反量化成 BF16（这一步本身不是瓶颈，因为是逐个处理+及时转 CPU），但反量化出来的 BF16 张量喂进线性层的那一刻，因为线性层现在真的挂了 `quant_config`，会被在线重新量化压缩回 4-bit 常驻显存——相当于"反量化只是个中转格式，最终常驻的还是压缩后的权重"。

**效果**：权重加载完成后显存 12.8GB，文本编码器加载完成后 19.3GB，生成过程中峰值 24.6GB——与官方"16.12GB 权重体积"量级基本吻合。

### 4.4 已知未修：稠密 attention mask 未做 varlen 化

`_build_segment_mask`（`ideogram4_transformer.py`）跟 harness 原版一样，用 `(ids.unsqueeze(2) == ids.unsqueeze(1)).unsqueeze(1)` 构建稠密布尔 mask，只是包了层 vllm-omni 的 `AttentionMetadata` 壳再传给 `Attention` 层——**没有真正做 varlen 化**。2048² 分辨率下这是 16384×16384 的布尔张量，原调研报告 §3.1 已经点名"任何路线都必须换成 varlen"。本次冒烟测试只验证了 1024²，没有触发这个问题，但**上到 2048² 大概率会遇到显存/速度问题，需要单独排期做 varlen 改造**。

### 4.5 Bug #4：非法/极端分辨率没有校验，直接 `math domain error` 崩溃

排查高分辨率问题时顺手测了一下"如果传的 `size` 看起来像宽高比而不是真实像素尺寸会怎样"：`{"size": "16x9"}` 这种请求，`parse_size`（`image_api_utils.py`，多个模型共用的通用工具函数）只检查"两个正整数用 `x` 分隔"，并不知道 Ideogram4 自己的分辨率约束，所以 16x9 会被原样解析成宽=16、高=9 的字面像素尺寸。原来的 `forward()` 只是用 `(height // patch) * patch` 向下取整对齐到 patch 网格（`patch=16`），9 向下取整直接变成 0，`_resolution_aware_mu` 里 `math.log(num_pixels / base_pixels)` 传入 0 触发 `ValueError: math domain error`，报给调用方是一个完全看不出原因的 500。

**修复**：仿照本仓库里 `Krea2Pipeline` 已有的约定（`pipeline_krea2.py`：分辨率不合法时**向上取整 + 打警告**，不硬报错，不中断请求）新增 `_validate_resolution()`，同时按官方 `docs/inference.md` "Supported Resolutions" 文档的约束把三条规则都补上（Krea2/HunyuanImage3 都没有做这三条里的第 2、3 条，因为那是 Ideogram4 自己文档明确给出的数字，不是通用约定）：

1. 向上取整到 16 的倍数（而不是原来的向下取整——避免再出现整到 0 的情况）；
2. 每边夹到 [256, 2048] 范围内；
3. 宽高比超过 6:1 时，把较短边等比放大到满足 6:1。

超出范围时打 warning 日志说明"请求的 WxH 被调整为 W'xH'"，返回码仍是 200，不会让调用方的请求直接失败。实测三个边界 case：`16x9` → 调整为 `256x256`（200）；`2048x256`（8:1，超宽高比）→ 调整为 `2048x352`（200）；`8000x8000` → 在更上层的通用尺寸校验（不是这次改的代码）就返回了 `400`，没有走到会崩溃的路径。

**关于"只传宽高比"的可用性结论**：这套 OpenAI 兼容接口**没有"只给比例、自动挑一个合适分辨率"的能力**，`size` 字段只认字面像素尺寸——想要"16:9 效果"必须自己换算成具体的 `WxH`（比如 `1536x864`），类似 GPT Image 2 那种"只支持固定分辨率枚举"的模式，不是一个能接受抽象比例描述的接口。如果以后要支持"只给比例"的用法，需要在门面层新增一个宽高比→具体分辨率的映射表（比如常见的 1:1/16:9/9:16/4:3 各对应一个官方验证过的具体 WxH），而不是让宽高比直接透传到 `size` 字段。

### 4.6 Bug #5：`mu`/`std`/`guidance_schedule` 完全没有 API 入口，官方三档预设只有一档能用

排查采样预设对比时发现：`forward()` 的 `mu`/`std`/`guidance_schedule` 只是**纯 Python kwarg**，从未从 `req.sampling_params`/`extra_args` 读取过——跟 `guidance_scale`/`seed`/`num_inference_steps`（这几个是标准 `SamplingParams` 字段，服务层会自动透传）不同，`mu`/`std` 属于 Ideogram4 私有参数，按本仓库"非声明字段必须走 `extra_params`"的约定，理论上应该能通过 `extra_args` 传入，但代码里压根没接这根线。后果：无论请求 `num_inference_steps` 是多少，`mu`/`std` 永远用写死的 `DEFAULT_MU=0.0`/`DEFAULT_STD=1.5`（只跟官方 `V4_QUALITY_48` 一致），guidance 的"精修阶段"步数也永远按 `DEFAULT_POLISH_STEPS=3` 算——也就是说**通过 API 传 `num_inference_steps=20` 或 `12` 想复现 `V4_DEFAULT_20`/`V4_TURBO_12`，实际上 mu/std/精修步数全部对不上官方数值**，等于永远在跑一个"步数改了但配方没变"的自定义档位，跟官方任何一档都不完全一致。

**修复**：新增 `SAMPLER_PRESETS` 常量（原样照抄 harness `sampler_configs.py` 的三档数值），在 `forward()` 里：

1. 支持显式 `extra_args.sampler_preset`（如 `"V4_TURBO_12"`）直接选档，未传 `num_inference_steps` 时用该档的步数；
2. 未显式选档时，按请求的 `num_inference_steps` 自动匹配（20→`V4_DEFAULT_20`，12→`V4_TURBO_12`，其余步数走原来的通用 HI/LO 兜底逻辑）。

修复后才能真正做到"选哪档预设，mu/std/精修步数就是官方哪档的数值"，§6.4 的三档对比正是在这个修复之后才做的。

### 4.7 Bug #6（事后静态代码审查发现，纯测试无法暴露）：`seed` 参数从未真正生效

用户要求"不要只靠跑起来验证，仔细读代码"之后做的静态审查发现：`vllm-omni` 的引擎调用约定是 `DiffusionModelRunner` 只调 `pipeline.forward(req)`——**只传 `req` 一个位置参数**，`forward()` 签名里其余关键字参数（`height`/`width`/`num_inference_steps`/`guidance_scale`/`seed`/`mu`/`std`）永远不会被调用方填值，必须靠方法体自己从 `req.sampling_params.X` 兜底读取。`height`/`width`/`num_inference_steps` 都做了这个兜底（`height = height or req.sampling_params.height or 1024`），但 **`seed` 没有**——原代码只有 `if seed is not None: generator = torch.Generator(...).manual_seed(seed)`，而 `seed` 参数永远是 `None`，这个分支永远不触发。

对比本仓库其他所有 diffusion pipeline（`glm_image`/`helios`/`hunyuan_video_1_5`/`qwen_image_edit`/`stable_audio` 等，逐个 grep 过），无一例外都读 `req.sampling_params.generator`（`DiffusionModelRunner._initialize_generator` 会提前用 `req.sampling_params.seed` 构建好）或 `req.sampling_params.seed` 二者之一。Ideogram4 的移植版是这批 pipeline 里唯一一个两者都没读的。

**实际影响**：不管调用方在请求里传什么 `seed` 值，每次生成实际用的都是进程级、未播种的全局 RNG 流——`torch.randn(..., generator=None)`。这解释了本次排查中一个反常现象：拿 §6.4 里跑题过的 samurai@V4_DEFAULT_20 用例（原始 seed=44）原样重跑，两次结果完全不同（一次跑题、一次正常）——因为 seed 从来没被真正用上，"相同 seed" 只是请求体里的一个数字，对实际生成的随机数流没有任何约束。

**修复**：改为优先读 `req.sampling_params.generator`（引擎已经用 `req.sampling_params.seed` 构建好的生成器），其次读 `req.sampling_params.seed` 自建生成器，跟仓库里其他 pipeline 的写法保持一致。修复后验证：同一 seed=44 请求连续调用两次，返回图片 **SHA256 完全一致**，确认可复现。

**⚠️ 对本报告此前全部结论的影响**：这个 bug 修复之前，本报告所有涉及"固定 seed 做 A/B 对照"的实验（§5.3/§5.4 的 skip vs 不 skip 对照、§6.4 的三档预设对比表）**seed 实际上都没有被真正固定过**——每次调用都是独立的随机噪声起点。这不代表这些实验的结论一定是错的（skip 与否本身是真实的代码路径差异，不是纯随机噪声），但**样本之间不是"同一噪声起点、只改一个变量"的严格配对对照，而是独立随机抽样**，统计意义比原文措辞暗示的要弱。§5.4/§6.4 的具体结论已经保留在原文中作为历史记录，但读者应该按"独立抽样、非配对"来理解其严谨程度，而不是按"控制变量法"理解。**建议后续用现在真正生效的 seed 复现 §5.3/§5.4 的关键配对实验**（尤其是"skip 是否会把正常生成搞砸"这个已被推翻的结论），才能得到真正受控的结果。

### 4.8 Bug #7（同一轮静态审查发现）：VAE 解码前的 latent 反归一化统计量来源用错了

比对官方 harness 的 `Ideogram4Pipeline._decode()` 发现：它用的是硬编码在 `ideogram4/latent_norm.py` 源码里的 128 维 `LATENT_SHIFT`/`LATENT_SCALE` 常量表（`z = z * latent_scale + latent_shift`）。harness 自己的 `AutoEncoder` 类虽然也从 checkpoint 加载了一个 `self.bn`（`BatchNorm2d(affine=False, track_running_stats=True)`）缓冲区，但逐行检查 `autoencoder.py` 的 `Encoder.forward`/`Decoder.forward` 后确认：**`self.bn` 在任何前向计算路径里都没被引用过**，纯粹是为了 `load_state_dict` 能对上 checkpoint 里的 `bn.*` key 而挂在那儿，不参与实际的反归一化计算。

我们的移植版 `_decode()` 用的却正是这个官方代码明确不用的 `self.vae.bn.running_mean`/`running_var`。在 gpu41 上直接读取同一份 NF4 checkpoint 的 VAE 权重验证：`bn.running_mean[0] = -0.0674` vs harness `LATENT_SHIFT[0] = +0.0198`；`bn.running_var[0]=3.25` 换算出的 std≈1.80 vs harness `LATENT_SCALE[0]=1.64`——不是同一组统计量的不同表示形式（不只是精度/取整差异），是两组完全独立、量级相近但数值不同的统计量。

**实际影响**：解码前送入 VAE 的 latent 会有系统性的、按通道各不相同的仿射偏差（量级约 10%~20%），表现为色彩/对比度/明暗的整体偏差——不会改变生成"画的是什么"（那由采样阶段决定，不经过这一步），但会让最终输出的色彩观感偏离官方 pipeline 的真实输出。这是一个只有逐行对比两份代码才能发现的 bug：生成结果本身构图完整、看起来"合理"，不会报错也不会明显失真到能一眼看出问题。

**修复**：把 harness `latent_norm.py` 的 `LATENT_SHIFT`/`LATENT_SCALE` 常量表原样搬进 `pipeline_ideogram4.py`，`_decode()` 改用这两个常量而不是 `self.vae.bn.*`。修复后重新生成的 samurai 测试图，配色/明暗观感正常（黑甲、竹林、水墨风格与 prompt 描述的调色板一致）。

---

## 5. 模型自带安全过滤的实测行为

### 5.1 三层机制，只有一层是内容审核

排查这个问题时最初以为是代码里的"关键词审核"，实测后发现是**三套完全独立的机制**：

| 层 | 是什么 | 有没有被我们的部署带进来 |
| --- | --- | --- |
| `safety.py`（Hive 云 API） | 需要显式传 `--hive-text-key`/`--hive-visual-key`，没配 key 就只警告不拦截 | 没有，只在 harness 的 `run_inference.py` demo 脚本里调用，`pipeline_ideogram4.py` 本身不调用 |
| `_verify_prompts`/`CaptionVerifier` | **纯 JSON 格式校验器**，检查 key 顺序、颜色格式、schema 完整性，跟内容安全无关 | PR #4227 完全没有移植这部分 |
| **烧录在模型权重里的安全对齐**（训练阶段的预训练数据过滤 + 后训练对齐） | 真正会拦截生成的那一层，`docs/safety.md` 原话："designed to further reduce the probability of the model generating NSFW content, **including for prompts that explicitly request it**" | 这是权重自带的，跟代码/部署方式无关，无法关闭 |

### 5.2 实测复现

用 PR 自带文档里的示例 prompt（"A detailed character design sheet for a young explorer-mechanic-inventor in a frozen world."，纯文本，完全无害）生成，**完全被拦截**：返回的图片是纯灰色底 + "Image blocked by safety filter" 文字，没有任何实际内容。

改用官方推荐的 JSON 结构化格式（`{"subject":"...","style":"...","scene":"frozen mountain village","palette":[...]}`）后：**成功生成了一张细节丰富、与 prompt 高度吻合的雪山村庄场景图**（冰川、木屋、平台、人物角色，配色也对上），但图片上仍然叠加了"Image blocked by safety filter"的警示水印——说明这次是"低置信度触发，生成正常但打标"，不是完全拦截。

两张图都在 `/nfs-output/model_reverify_20260827/`（`ideogram4_smoke.png` 纯文本被拦截版，`ideogram4_json_smoke.png` JSON 格式生成成功版）。

### 5.3 根因定位与实测有效的缓解方案：跳过模型自己的第一步前向计算

社区（ComfyUI/Reddit，二手转述，原帖内容因平台限制无法直接抓取核实）流传一种说法：安全触发跟"模型自己的第一步"有关，具体做法是把第一步交给另一个模型跑、Ideogram4 只接手后续步骤。由于当时没有兼容的第二模型（Ideogram4 复用的是 Flux2 KL VAE，理论上 Flux2 系模型的 latent 空间兼容，但 Flux2 权重当时不在 NFS 上，需要额外下载），改为先做一次零成本的自验证：**不引入第二个模型，只是让 Ideogram4 跳过自己在 schedule 第 0 步的前向计算**（`z` 保持原始高斯噪声不变，直接喂给第 1 步），验证"触发是否真的跟第一步的前向计算本身有关"。

实现：`Ideogram4Pipeline.forward()` 新增 `extra_args.skip_first_n_steps`（通过 OpenAI 接口的 `extra_params.skip_first_n_steps` 传入）。调度表按 `num_inference_steps + skip_first_n_steps` 的粒度构建，再丢弃前 `skip_first_n_steps` 个最高噪声档位——这样跳过之后，真正跑 transformer 的步数依然等于调用方请求的 `num_inference_steps`，不会悄悄把采样预算缩水。已用 10 步 + `skip_first_n_steps=1` 和 10 步不跳过做过耗时对照（11.50s vs 11.42s，基本一致），确认步数对齐生效。

**实测结果（`skip_first_n_steps=1`，两个用例）**：

| 用例 | 不跳过 | 跳过第 1 步 |
| --- | --- | --- |
| JSON 格式"frozen mountain village" | 生成正常但打水印（§5.2） | **完全干净，无水印，画质正常** |
| 纯文本"character design sheet"（原本完全拦截） | 纯灰底 + 拦截文字，无实际内容 | **完全干净，生成一张高质量、与 prompt 高度吻合的角色设计图** |

两个用例（含之前被完全拦截的那个）**全部通过**，跳过第一步不仅避开了水印，画质也没有可观察的劣化——**社区说法得到验证：触发确实跟模型自己在第一步的前向计算强相关，而不是跟输入内容本身有本质关系**。四张对照图都在 `/nfs-output/model_reverify_20260827/`：`ideogram4_smoke.png`/`ideogram4_json_smoke.png`（不跳过）、`ideogram4_plaintext_skip1.png`/`ideogram4_skip1.png`（跳过第 1 步）。

**这个结论要谨慎对待，不要过度泛化**：

1. **样本量只有 2 个**，且两个 prompt 内容本身都明显无害（雪景村庄、机械角色设计），没有测试过"真正应该被拦截的内容"在跳过第一步后是否也会被放行——如果跳过第一步会让模型对真正违规内容也失去判断力，这就不是"消除误报"而是"整体关闭了安全机制"，需要用负面/边界 case 补测后再下结论。
2. 跳过第一步是**用原始高斯噪声直接喂给第 2 步**，数值上不严谨（第 2 步本该收到的是"部分去噪"的输入，不是纯噪声）。
3. **不再建议"接第二个模型跑第一步"这条路线**：重新想过，跨模型接力本身也谈不上更严谨——即便共享 VAE，两个独立训练的模型在同一个 noise level 下的边际分布也不保证对齐，反而比"同模型、同训练分布，只是这一步输入换成纯噪声"引入更多不确定性，还要多维护一个模型依赖。加上现在跳过后采样步数已经能对齐用户请求，本方案（同模型跳过）目前看是更简单也更可信的路线，第二模型接力这条路线不再作为后续计划。
4. **遗留工作**（仍然值得做，跟上面这条发现互补）：移植 `magic_prompt`/prompt 结构化步骤，让自由文本自动转 JSON 再喂给模型——即使"跳过第一步"这个方法验证有效，两者也不冲突，可以叠加使用。

**⚠️ 以上 4 条 caveat 的后续进展（§5.5 补充）**：caveat 1 的担忧被证实是对的方向，但表现形式不是"整体关闭安全机制"，而是"不可靠地打乱安全机制的渲染过程"——对真正会拦截的内容，skip 只有约 1/4 概率换来真实内容，3/4 概率是乱码，不是"消除误报"也不是"完全失效"，是一种脏的中间态。caveat 3（不再考虑接第二个模型跑第一步）**因为这个新发现被重新打开讨论**：既然问题的关键在于"用纯噪声代替本该是部分去噪的输入"，那么用 Flux2 之类的真实模型算出一个数值上合理的"部分去噪状态"去接力，理论上比直接跳过更可能规避这个问题——但这需要额外下载 Flux2 权重、并解决两个模型 noise schedule 的对齐问题，工作量明显大于当前的参数开关排查，目前按用户要求优先把根因定位完（见 §5.5），还没有开始做。

### 5.4 高分辨率排查：一开始的"skip 会毁内容"结论主要是测试参数错误，已重新验证

**第一轮排查（已证伪，记录在案避免重复踩坑）**：最初用固定 `guidance_scale=7.0` + `num_inference_steps=50` 在分辨率阶梯上测 `skip_first_n_steps=1`，观察到 1536/2048 完全跑题（装饰文字海报、鹅卵石纹理）。当时归因于"跳过第一步在高分辨率下丢失全局结构"。

**根因排查（用户建议查官方文档后发现）**：官方 `README.md` 明确写着 **"For the highest-quality images, set `--height 2048 --width 2048` and `--sampler-preset V4_QUALITY_48`"**，且"Flexible resolution. Native support for any resolution from 256 to 2048"——2048 是官方明确验证过的推荐档位，不是能力边界。而 `V4_QUALITY_48` 预设是 48 步、前 45 步 gw=7.0、**最后 3 步降到 gw=3.0**（精修阶段降低引导强度）；我们的第一轮测试显式传了 `guidance_scale=7.0`，会命中 `forward()` 里 `if guidance_scale is not None: effective_schedule = (guidance_scale,) * num_steps` 这个分支，**把最后 3 步的降权精修调度整个覆盖成了全程 gw=7.0**，加上步数用错（50 而非官方的 48）——双重偏离官方推荐配置。

**重新验证（不传 `guidance_scale`，走默认 HI/LO 分段调度，步数改回 48，匹配 `V4_QUALITY_48`）**：

| 分辨率 | 不跳过 | 跳过第 1 步 |
| --- | --- | --- |
| **2048×2048** | 正确内容（灯塔/悬崖/日落，构图精细）+ 水印 | **完全正确，无水印，本轮测试里最好的一张** |
| **1536×1536** | 正确内容 + 水印 | 场景本体正确（灯塔、悬崖、海浪都在），但叠加一段乱码文字覆盖层（"COUCTHES BA JONO"，配色对但是乱码）——比第一轮的"整个画面被替换"轻微得多，但还没有完全干净 |

**结论修正**：第一轮"高分辨率下 skip 会毁掉内容"的结论**主要是测试参数错误造成的假象，不是移植代码的 bug，基本也不是模型分辨率能力问题**——2048 用正确设置后完全正常。1536 用正确设置后 skip 仍有残留的文字伪影，但**样本量只有 1 个 seed，不足以判断这是不是系统性的分辨率规律**，需要更多样本才能下结论。

**⚠️ 进一步修正：连"城市天际线"那个反例也是假象**。用正确预设（48 步、不传 `guidance_scale`）重新跑了一次同一个 prompt/seed 的对照：**不跳过 → 这次是真的完全拦截**（纯黑底+警示文字+一个诡异红方块），**跳过 → 生成一张质量很高的赛博朋克雨夜街景**，完全正确。也就是说这个 prompt 本来就会被拦截，之前用错误参数测的时候恰好没触发拦截，才误以为是"skip 主动把一个不需要缓解的正常生成搞砸了"。**目前为止，所有用正确采样参数测过的 case，无一例外都是"不跳过→拦截，跳过→干净正确"，还没有找到一个"correct 参数下 skip 把正常内容搞砸"的真实案例**。§5.3 提出的"不能当全局默认开关、需要检测+重试层"这个结论因此被推翻——目前证据支持的是相反方向：**只要用官方正确的采样预设，`skip_first_n_steps=1` 可以作为对外服务的默认行为，不需要额外的检测+重试层**。§8 的相关待办已经过时，见下方更新。

**⚠️ 这个结论后来又被 §5.5 用更大样本推翻了一次**：当时只测了 2-3 个正例就下了结论，样本太小；§6.4/§5.5 用 30 组测试 + 针对性对照实验证明 `skip_first_n_steps=1` 平均只有约 1/4 概率换来干净内容，3/4 概率是乱码/跑题，不能作为默认行为。本节保留作为排查过程的历史记录，**最终结论以 §5.5 为准**。

**1536×1536 为什么可能天生比 1024²/2048² 弱（用户提议核查，已找到间接证据）**：查了 `docs/inference.md` 的"Supported Resolutions"示例表——`Square` 档位官方给的例子是 **1024×1024**，README 里"最高质量"推荐的是 **2048×2048**，而 **1536 全表只以"非方形"形式出现**（`Landscape 1536×1024`、`Portrait 1024×1536`，都是一边 1536 一边 1024，没有 1536×1536 方形组合）。官方"256-2048 任意 16 的倍数都原生支持"是笼统声明，不代表训练/验证数据在这个区间内均匀覆盖——1024²和 2048²这两个"整数"方形档位更可能是重点覆盖的分辨率桶，1536²这种不上不下的方形组合大概率落在数据分布更稀疏的区域。这跟我们唯一一次 1536²+skip 测试出现残留文字伪影、而 1024²/2048²都完全正常的现象吻合。**仍然只是间接证据 + 单样本，不是决定性结论**，需要换 seed/prompt 多测几次 1536²才能坐实，但方向上支持"1536²不是好的验证/生产分辨率选择，即使抛开 skip 的问题，也应该优先用 1024²或 2048²这两个官方明确验证过的方形档位"。

---

### 5.5 ⚠️⚠️ 最终定性：`skip_first_n_steps=1` 不是"避开拦截"，而是"把干净的拦截画面打乱成更难识别的乱码/跑题"——§5.4 的"不需要检测+重试层"结论被推翻

§6.4 用 30 组样本发现"跑题/乱码水印"问题依然存在（约 30%）之后，用户追问根因，做了一组关键对照实验：**拿 §6.4 里 8 个失败案例，保持 prompt/seed/预设完全不变，只是把 `skip_first_n_steps` 从 1 改回 0（也就是走官方默认、完全不跳步的路径）**，结果：

| 关掉 skip 后的结果 | 数量 | 案例 |
| --- | --- | --- |
| 干净的"Image blocked by safety filter"标准拦截画面 | 4 | windmill、subway、desert 驼队、pirate 海盗船 |
| 完全正常，干净出图 | 2 | typewriter、ferris 摩天轮 |
| 拦截画面本身就不干净（拦截文字正确但背景混入请求内容碎片，或拦截文字本身拼写错误——比如"Image blocked by **kafaty** filter"） | 2 | bakery 面包店、hummingbird 蜂鸟 |

**这组数据把根因拆成了两层**：

1. **6/8（75%）本质上是安全过滤触发**，不是模型瞎画跑题。关掉 skip 后其中 4 个变回干净的标准拦截画面，2 个变成"半干净"的拦截（**这 2 个案例说明模型自己渲染拦截提示这套机制本身就不总是可靠，即使完全不用 skip、走纯官方默认路径，也有概率把拦截提示渲染成带拼写错误的文字，或者让请求内容的碎片"泄漏"进拦截画面里——这是训练阶段带来的模型自身缺陷，跟我们的移植代码无关**）。
2. **2/8（25%，typewriter/ferris）根本没有触发安全过滤**，关掉 skip 就完全正常——**纯粹是 `skip_first_n_steps` 把第二步的输入换成了训练时没见过的"原始高斯噪声"，导致对本来毫无问题的良性内容也造成了不稳定**，这条跟安全过滤无关。

**关键推论——`skip_first_n_steps=1` 并不会可靠地"避开"拦截**：§5.3/§5.4 早期只测了 2-3 个正例（"skip 后从拦截变成干净出图"），就得出"skip 可以当默认值"的结论；但现在看，**skip 更准确的作用是"打断模型渲染拦截画面的过程"**——这个打断有时候的确会让内容侥幸跑出来（跟早期正例一致），但从 §6.4/§6.5 的 8 个反例看，**更多时候是把一个清晰、容易被程序识别的"Image blocked by safety filter"灰底画面，打乱成一个内容错乱、但表面看起来像是"正常出图但跑题"的东西**——这对生产环境反而更糟：干净拦截画面一眼就能自动识别处理，乱码/跑题图片必须额外做 OCR/图文匹配检测才能发现，还有漏检风险。

**进一步验证——"先拦截后重试会不会又变成乱码"**：用户提出这个疑问后专门测了两条不同的重试路径：

1. **切换 `skip_first_n_steps`（0→1）重试**：拿 windmill/subway（关掉 skip 后是干净拦截的两个案例），同一个 seed 只把 `skip_first_n_steps` 从 0 改成 1，**直接复现了 §6.4 里那两个乱码/跑题案例本身**——即"先清晰拦截、重试后变成乱码"这个场景**已经发生过、是真实数据**，不是假设性担忧。
2. **保持 `skip_first_n_steps=0` 不变，只换新 seed 重试**：windmill/subway 各配 3 个全新 seed（共 6 次），**6/6 全部仍然是清晰的"Image blocked by safety filter"拦截画面，没有一次乱码，也没有一次成功出图**——说明这两个 prompt 的安全触发对 seed 不敏感，是"钉死"的拦截，单纯换 seed 重试既不会产出乱码，但也救不回内容。

**两条路径合起来看，规律很清楚**：单纯换 seed（不碰 skip）是安全的，但对"确实会被拦截"的内容没用，会一直原地拦截；只有动 `skip_first_n_steps` 才可能让内容绕过拦截，但正因为这个"绕过"的过程本身不稳定，才会把清晰拦截变成乱码——**用 windmill/subway 各自的 4 个 `skip=1` 结果统计（原始 seed + 3 个新 seed），真正拿到干净正确内容的比例只有约 1/4（2/8），其余 3/4（6/8）都是乱码或完全跑题**。也就是说 `skip_first_n_steps=1` 本质上是在用"约 75% 概率产出乱码垃圾"换"约 25% 概率绕开拦截拿到真实内容"，这笔账在生产环境里大概率是亏的——干净拦截容易被程序识别、可以直接反馈或按业务逻辑处理，乱码输出看起来像是"生成成功"但内容是错的，更难被自动检测出来。

**结论（推翻 §5.3/§5.4 末尾的"不需要检测+重试层"）**：`skip_first_n_steps=1` 不应该再作为默认全局开关。更合理的默认策略是：**默认 `skip_first_n_steps=0`（官方原生路径）**，检测输出是否命中"Image blocked by safety filter"这个清晰、可 OCR 识别的固定文案——命中就在门面层判定为"被拦截"，可以选择直接告知调用方、或换 seed 重试（保持 skip=0，重试收敛性待下方补充数据），**而不是切换到 skip=1 去"赌"一个未知的乱码/跑题输出**。§6.5 部署建议、§8 待办均已同步更新。

**⚠️ 这个结论后来又被 §5.6 部分推翻**：以上关于"安全过滤误伤良性内容"、"skip 是唯一能绕过拦截的手段"这些结论，都是在用错误的 prompt 格式（简化写法 `{"subject":...}`）的前提下成立的。§5.6 发现真正的根因是 prompt 格式，而不是 skip 参数本身——格式修好之后，本节讨论的大部分拦截案例根本不会发生，也就不需要靠 skip 去"赌"绕过拦截。

### 5.6 ⚠️⚠️⚠️ 最重大发现：安全过滤误伤良性内容的真正根因是 prompt 格式，不是模型或 skip 参数

用户看到社区一个 ComfyUI 插件（`ComfyUI-Ideogram4-DirectJSON-Modified`，基于 `ComfyUI-KJNodes`）流传"4 个框解锁"的说法后要求排查。逐行读了这个插件代码（纯粹的 JSON/bbox 可视化编辑器，没有任何"4 个框"的硬编码特殊逻辑），但在它自带的示例 workflow 里发现了关键线索：一个真实的实战 prompt 模板，用的是官方 `docs/prompting.md` "Full example" 里那套**完整结构化 caption schema**——`high_level_description` + `style_description`（`aesthetics`/`lighting`/`photo` 或 `medium`/`art_style`/`color_palette`）+ `compositional_deconstruction`（`background` + `elements[]`，每个 element 带 `type`/`bbox`/`desc`/`color_palette`）——而本报告从 §5 开始到 §6.4 的全部 30+ 组测试，**从头到尾用的都是一个简化写法**：`{"subject":..., "style":..., "scene":..., "palette":[...]}`，只是"看起来像 JSON"，跟模型真正训练对齐用的格式完全不是一回事。

**验证实验**：挑了 §6.4/§5.5 里 9 个原本 100% 出问题的案例（windmill/subway/bakery/typewriter/desert/sushi/ferris/pirate/hummingbird，涵盖"完全拦截"和"skip 后乱码"两种失败模式），用完整 4-element schema 重新构造成等效 prompt，分别在 `skip_first_n_steps=0` 和 `skip_first_n_steps=1` 下各测一遍：

**结果：18/18 全部正常，没有一次拦截，没有一次乱码**（少数案例有极轻微的文字渲染瑕疵，比如招牌小字拼写不准，跟"完全跑题"/"乱码水印"是完全不同性质的问题）。

**进一步排除"4"是不是魔法数字**：拿 windmill 的完整 schema 依次砍到 3 个、2 个、1 个 element（其余字段不变），`skip_first_n_steps=0` 下**全部依然完全正常**——证明起作用的不是 element 数量，而是"有没有用 `compositional_deconstruction` 这套结构化格式本身"。社区"4 个框解锁"的说法，"4"这个数字大概率只是那个人自己常用模板的巧合，不是真正的机制。

**结论**：本报告 §5.1-§5.5、§6.4 排查出的"安全过滤高误报率"、"`skip_first_n_steps` 是唯一缓解手段但会产出乱码"，这些现象**很可能主要是简化 JSON 格式造成的假象，不是模型或 skip 参数本身的问题**。用官方真正的结构化 caption schema 之后：

- 之前会被拦截的良性内容（风车、地铁站、面包店、沙漠驼队）不再拦截；
- 之前 skip=1 会导致乱码的案例（打字机、寿司、摩天轮、海盗船、蜂鸟）不再乱码；
- 不需要在 `skip=0`/`skip=1` 之间做取舍赌博，两种设置下都能正常出图。

**这意味着 §1/§5.5/§6.5 里"默认 `skip_first_n_steps=0` + 检测拦截文案做兜底"这套结论的前提被削弱了**——如果门面层本来就该用正确的官方 schema（这是任何认真对接这个模型的实现都该做的事，不是可选项），那么"~30% 失败率"这个数字本身可能大幅偏高估计了真实的现网风险。**最终建议顺序调整为**：① 门面层必须实现"自由文本 → 官方结构化 JSON schema"的转换层（即 §5.3 caveat 4/§8 提到的 `magic_prompt`/prompt 结构化工作，现在优先级从"锦上添花"提升为"关键路径"）；② 在此基础上默认 `skip_first_n_steps=0`；③ 仍然保留"检测 Image blocked 文案"的兜底逻辑，作为防御性的最后一道保险，而不是主要依赖手段。

### 5.7 补充验证：即使用对格式，`skip_first_n_steps=1` 依然有极低概率的残留致乱风险

用完整 schema 额外测了 4 类"边界敏感内容"（士兵持枪、夜盗保险库、战场负伤武士、暗黑仪式，均为 PG-13~R 级、非色情非极端内容），`skip=0`/`skip=1` 各测一遍：

**8 组里 7 组完全干净**（士兵、劫匪、仪式在两种 skip 设置下都正常出图），**唯独"战场负伤武士"在 `skip=1` 下出现部分损坏**——画面主体完全正确，但叠加了一堆不自然的深红色圆点和一个小盾牌图标（`skip=0` 版本完全干净，无此问题）。

**结论**：格式修复大幅降低但没有完全消除 `skip_first_n_steps` 的残留风险——概率明显低于错误格式下的情况（错误格式下接近 75% 出问题，这次 8 组里只有 1 组），但**不能因为格式修好了就认为 `skip=1` 已经绝对安全**，仍然建议按 §5.6 的最终建议顺序、以 `skip=0` 为默认。

### 5.8 安全过滤的能力边界：不覆盖暴力/血腥内容

用户要求测试更明确的血腥暴力内容（战场枪伤流血、恐怖凶案现场血迹+凶器、丧尸破肚露骨），用完整 schema，`skip=0`/`skip=1` 各测一遍（共 6 组）。

**结果：6/6 全部完整生成，没有一次被拦截**，即使是 `skip=0`（官方原生路径、模型自身安全对齐完全生效的设置）也没有拦截任何一张。

**这跟前面测的"士兵持枪/夜盗/暗黑仪式"这类温和边界内容能通过是两回事**——这次测的是明确的、图形化的暴力伤害内容（可见枪伤流血、凶案血迹、破肚露骨），而不是仅仅"暗示"暴力/犯罪。模型对这个尺度的暴力内容也完全没有拦截反应。

**结论**：模型自带的安全对齐**很可能只针对色情（NSFW）类内容**，跟官方 `docs/safety.md` 反复使用"NSFW categories"这个措辞吻合（行业内 NSFW 通常特指色情内容，不包含暴力/血腥/恐怖）。**如果业务对"血腥暴力"内容有独立的合规要求，不能指望模型自身的安全对齐拦截**，需要依赖官方文档提到的、我们目前完全没接入的 Hive 视觉内容审查（`--hive-visual-key`，§5.1 表格已确认我们的部署缺这一层）或内部等效的输出审查方案。至于色情类内容的过滤能力，本次没有测试（不适合用于本次排查的边界测试范围），保留为未知项。

---

## 6. 分辨率 / 步数 / TP 并行 性能实测（同一批冒烟测试的延伸）

沿用 §5.3 的 `skip_first_n_steps=1`，在 gpu41-44（5 台调试机的其中 4 台）并行铺开测试。

### 6.1 分辨率阶梯：4096×4096 是真实 OOM 边界

| 分辨率 | 显存峰值 | 耗时（50 步） | 输出 |
| --- | --- | --- | --- |
| 1024×1024 | 24.6GB | 54.7s | 正常（见 §5.4） |
| 1536×1536 | 35.7GB | 132.5s | 内容跑题但没 OOM（见 §5.4） |
| 2048×2048 | 35.1GB | 275.6s | 内容跑题但没 OOM（见 §5.4） |
| **4096×4096** | — | **0.6s 即失败** | **真实 CUDA OOM**（"Tried to allocate 8.01 GiB ... 4.79 GiB is free"），容器存活、健康检查恢复正常 |

4096 的 OOM 发生得极快（0.6s），不是在 50 步迭代过程中逐渐耗尽，而是在生成一开始构建大尺寸 attention/latent 张量时就直接申请不到内存——2048 已经用到 35GB/39.49GB（89%），4096 是 2048 像素数的 4 倍，第一次大张量分配就直接超限。

### 6.2 步数对比：耗时与步数基本线性

同一 prompt（"a bowl of ramen"），1024×1024：

| `num_inference_steps` | 耗时 |
| --- | --- |
| 20 | 22.3s |
| 50 | 53.9s |

53.9/22.3 ≈ 2.42，20→50 步数变化 2.5 倍，耗时变化基本同比例，符合预期（没有观察到明显的固定开销占比异常）。

### 6.3 TP 并行：TP=2 真实有效，TP=4 架构上不可能

Ideogram4 的线性层是真正的 vLLM 并行层（`QKVParallelLinear`/`ColumnParallelLinear`/`RowParallelLinear`），PR 自带的 `test_ideogram4_transformer_tp.py` 单测也验证了正确的头数切分逻辑——**跟 Krea2 的 `ReplicatedLinear`（假 TP，无收益）完全不同**，这里的 TP 是真的在摊薄计算和显存。

同一 prompt/分辨率/步数（"a vintage motorcycle"，1024×1024，50 步）实测：

| 配置 | 耗时 | 显存 |
| --- | --- | --- |
| TP=1 | 53.7s | 24.6GB |
| **TP=2** | **44.8s（快 17%）** | **20.0GB/卡（省 19%）** |
| TP=4 | — | **架构上不可能** |

**TP=4 报错**：`AssertionError: 18 is not divisible by 4`——Ideogram4 是 18 个注意力头，18 不能被 4 整除，`QKVParallelLinear` 按头数切分时直接断言失败，4 个 worker 进程全部启动崩溃。**可用的 TP 度数只能是 18 的因数：1/2/3/6/9/18**。按 4 卡一台机器的现网配置，TP=2（4 卡机可以跑 2 个 TP=2 副本提吞吐）是唯一务实的选择，TP=3 会浪费 1 张卡（4 卡机只能起 1 个 TP=3 副本），未做实测（用户明确表示不用测）。

### 6.4 三档采样预设对比 + prompt 结构化格式的取样口径说明

**取样口径（重要，直接影响下面数据的可信度）**：§6.4 全部测试的 prompt 都遵循 §5.2/§5.3 确认的"必须用 JSON 结构化格式（`{"subject":...,"style":...,"scene":...,"palette":[...]}`），否则会被直接拦截"这条约束。其中**最后 4 组（balloon/barista/coral/samurai）的请求体原始文件还在，可以逐字核对**是 JSON 结构化格式，例如：

```json
{"subject": "a hot air balloon", "style": "watercolor illustration", "scene": "floating over a lavender field at sunrise", "palette": ["#c9a0dc", "#f4c542", "#7a8b99"]}
```

**前 3 组（lighthouse/city/village，即三档预设对比表用的那 3 个 prompt）的原始请求体文件后来被同名的响应文件覆盖，无法逐字核对**——响应 JSON 里 `revised_prompt`/`cot_output` 字段均为 `null`，容器 access log 只记了 `POST /v1/images/generations 200 OK`，没有记请求体，所以现在拿不出这 3 组的字面 prompt 文本。按测试当时的操作习惯（在确认 JSON 格式是硬约束之后才做的这轮测试），这 3 组大概率也是同样的 JSON 结构化格式，但**这是推断，不是可核对的事实**，特此注明，避免后续误当成"已核实"的结论。

三档预设对比表（1024×1024，`skip_first_n_steps=1`，均为 JSON 结构化 prompt，seed 固定）：

| Prompt 主题 | V4_QUALITY_48（48 步） | V4_DEFAULT_20（20 步） | V4_TURBO_12（12 步） |
| --- | --- | --- | --- |
| 灯塔/悬崖/日落（lighthouse） | 正常 | 正常 | 正常 |
| 城市天际线（city） | 正常 | 正常 | **跑题**（生成了无关的装饰性文字设计） |
| 山村（village） | **跑题** | 正常 | 正常 |

3 档里没有一档 3/3 全过，**48 步档反而不是最稳的**——这跟"quality"这个名字给人的直觉相反。

为了扩大 `V4_DEFAULT_20` 的样本量、进一步验证它是不是确实更可靠，额外补测了 4 组不同主题/分辨率（balloon/barista 1024×1024，coral/samurai 2048×2048，`skip_first_n_steps=1`，请求体已逐字核对为 JSON 结构化格式）：

| Prompt 主题 | 分辨率 | V4_DEFAULT_20 结果 |
| --- | --- | --- |
| 热气球（balloon） | 1024×1024 | 正常 |
| 机器人咖啡师（barista） | 1024×1024 | 正常 |
| 珊瑚礁（coral） | 2048×2048 | 正常 |
| 武士（samurai） | 2048×2048 | **跑题**（生成了无关的装饰性文字/图标设计，非安全过滤特征，纯内容跑题） |

**汇总**：`V4_DEFAULT_20` 总计 7 组测试（3 档对比表里的 3 组 + 补测的 4 组），**6 组正常、1 组跑题，成功率 ~86%**；`V4_QUALITY_48` 3 组里 1 组跑题；`V4_TURBO_12` 3 组里 1 组跑题——三档看起来都有大致相近的固有跑题率，`V4_DEFAULT_20` 样本量更大且表现不差，加上更快（20 步 vs 48 步），综合更值得作为默认档位。

**外部佐证**：artificialanalysis.ai 的文生图榜单上，"Ideogram 4.0"（基础/默认档，对应官方 20 步左右的常规配置）Elo **1222**，反超"Ideogram 4.0 (Quality)"（对应 48 步高质量档）Elo **1216**——公开评测结果的方向跟本次实测一致，不是我们测试样本的偶然巧合，但也不是同一批测试数据，只能算方向性佐证，不是同源验证。

**关于跑题现象本身**：7 组测试里唯一的跑题案例（city@TURBO_12、village@QUALITY_48、samurai@DEFAULT_20，共 3 例）表现高度一致——都是生成一张构图完整、配色正常的装饰性文字海报/图标设计，跟 prompt 描述的主体（城市/山村/武士）完全无关，**不是安全过滤的"灰底+警示文字"特征，也不是分辨率或步数单一因素能解释的**（3 例分别出自 3 个不同的预设/分辨率组合）。目前判断为模型自身的固有不稳定性，样本量（10 组里 3 组）不足以精确定位诱因，见 §8 待办。

**换 seed 复测（用户提议，排除"某个 prompt 天生容易跑题"这个可能性，Bug #6/#7 修复前做的）**：拿 samurai@DEFAULT_20（2048×2048，跑题的那一组，原 seed=44）的**完全相同 prompt/分辨率/预设/skip 参数**，只换 4 个新 seed（101/102/103/104）重新生成，**4/4 全部正常**。当时的解读是"单次 seed 命中的随机失败，换个 seed 就消失"——但事后发现 §4.7 的 seed bug 意味着**这 4 次"换 seed"实际上都只是同一个坏掉的全局 RNG 流上的连续抽样，"换 seed"这个操作本身没有意义**，这个结论的因果链条要重新看待，见下方修复后的复测结果。

**⚠️ 修复 Bug #6/#7 后的复测（推翻"模型固有跑题"结论）**：Bug #6（seed 从未生效）和 Bug #7（latent 反归一化用错统计量）修复后，把上面全部 3 个跑题案例重新测了一遍：

- samurai@DEFAULT_20（2048×2048，原 seed=44，**这次 seed 真正生效**）：连续调用两次，返回图片 **SHA256 完全一致**且内容正确（武士竹林图，无跑题）。
- city@TURBO_12（1024×1024）：原始 prompt 已找不回（§6.4 开头取样口径说明），用同主题重构的等效 JSON prompt 配 3 个不同 seed（201/202/203）重新测试，**3/3 全部正常**（赛博朋克城市天际线，主体/场景/配色都对）。
- village@QUALITY_48（1024×1024）：同样用重构的等效 prompt 配 3 个不同 seed（301/302/303），**3/3 全部正常**（水墨风格山村，木屋/山坡/雾都对）。

**汇总：post-fix 首轮 8/8 全部正常，0 次跑题**（vs. fix 前 10 组里 3 组跑题，~14%~30% 量级）。当时据此判断 Bug #6/#7 是根因、问题已解决——**这个判断被下面的扩大样本量测试推翻，是小样本运气好导致的误判**。

**⚠️⚠️ 扩大样本量复测（用户要求，最终结论）：问题依然存在，之前 8/8 是运气**。用 30 个全新的、互不重复的 prompt（涵盖动物/人物/建筑/交通工具/静物/抽象场景，全部 JSON 结构化格式）重新测试，分布：24 组 V4_DEFAULT_20@1024×1024（seed 5001-5024）、3 组 V4_QUALITY_48@2048×2048（seed 5025-5027）、3 组 V4_TURBO_12@2048×2048（seed 5028-5030），全部 `skip_first_n_steps=1`：

| 结果类型 | 数量 | 占比 | 案例 |
| --- | --- | --- | --- |
| 正常 | 21 | 70% | 灯塔/红熊猫/热气球节/渔夫/骑士/咖啡馆/水母/机械人/小提琴手/树屋/宇航员/凤凰/图书管理员/刺猬/温室/纸飞机/海龟/陶艺师/雪豹/国际象棋/蜂鸟①（左上角局部） |
| **完全跑题**（主体完全丢失，被抽象/无关内容取代） | 5 | 16.7% | 风车→变成三人物插画；地铁站→变成 UI 图标设计稿；面包店→变成法棍图案壁纸+乱码文字；海盗船→变成蓝白抽象几何图案；蜂鸟→画面被重复装饰纹样占满，正确内容只剩左上角一角 |
| **乱码文字/图标水印污染**（主体保留，但画面被叠加乱码文字或几何图标） | 4 | 13.3% | 打字机（"GODE"/"POG!"等乱码英文）；沙漠驼队（万花筒图案边框）；寿司拼盘（乱码汉字+一个"V"字 logo）；摩天轮（灯牌乱码"CILLY"+四角抽象图标） |

**总失败率 9/30 ≈ 30%**，跟修复前的估计量级相当，**不比修复前低**。这次还识别出一个此前没有单独归类过的失败子模式——"乱码文字/图标水印污染"（主体正确，但被叠加一段类似盗版图库水印的乱码文字或抽象图标），跟"完全跑题"（主体整个被替换）是两种不同的表现形式，但看起来同源（都涉及模型在图像里"想画点文字/图标装饰"但控制不住）。

**最终结论（推翻 §6.4 上一段的误判）**：Bug #6/#7 是两个真实存在、已验证修复正确的 bug（同 seed 复现 SHA256 一致、latent 反归一化数值对齐官方常量表），**但它们不是"跑题/乱码文字水印"这个失败模式的根因**，两者是独立问题。之前 8/8 全过是小样本量下的运气（8 个样本几乎全落在 V4_DEFAULT_20/1024×1024，且只覆盖 3 个主题，n 太小不足以撞见约 30% 的真实失败率——8 次全部命中"正常"这一侧的概率约 0.7⁸≈5.8%，小概率事件但确实发生了）。**这个问题目前没有已知修复手段，应当按"模型/pipeline 存在约 30% 量级的固有失败率"对待，§6.5 的部署建议需要恢复"门面层加输出检测+重试"这一条**，见下方 §6.5 更新。§8 待办的排查项也已恢复为高优先级。

---

### 6.5 现网部署建议（最终版，延续第 6 节 + §6.4/§5.6 的发现）

- **⚠️⚠️ 最优先事项（§5.6 发现后新增）：门面层必须实现"自由文本 → 官方结构化 JSON schema"转换层**。本节剩余的 bullet（skip 默认值、检测拦截文案、30% 失败率兜底）都是在错误 prompt 格式下得出的结论，格式修好之后这些问题的实际严重程度会大幅降低。不修这个格式问题，后面的兜底措施只是在缓解一个本可以从根上避免的问题。具体做法：参考官方 `docs/prompting.md` 的 "Full example" schema（`high_level_description`/`style_description`/`compositional_deconstruction.background+elements[]`），在门面层把用户的自由文本 prompt 转换成这套结构化格式再喂给模型——element 数量不需要凑够 4 个（§5.6 已验证 1/2/3/4 个都可以），关键是走这套结构，不是拼一个"看起来像 JSON"的简化字典。这部分工作可以合并到 §5.3 caveat 4 提到的 `magic_prompt` 移植任务里一起做。
- **GPU 配置**：`--tensor-parallel-size 2`（同时也是 GPUStack `gpus_per_replica: 2`），比单卡快 17%、省 19% 显存，双重占优，没有理由用 TP=1。4 卡一台的物理机可以跑 2 个 TP=2 副本，兼顾并发吞吐。
- **采样预设默认用 `V4_DEFAULT_20`，不是听起来更好的 `V4_QUALITY_48`**：§6.4 的三档对比 + artificialanalysis.ai 公开榜单方向一致地证明 20 步档更可靠。门面层默认 `sampler_preset=V4_DEFAULT_20`（对应 `num_inference_steps=20`），只有用户明确要求"quality"档位时才切到 48 步。
- **采样参数必须走官方预设的 `mu`/`std`/guidance 调度，不要自己传 `guidance_scale` 常数**：§5.4 排查证明显式传 `guidance_scale` 会覆盖模型精修阶段的降权调度，导致输出质量劣化。§4.6 的 Bug #5 修复后，选定 `sampler_preset` 即可拿到官方对应档位的完整配方，门面层不应该再暴露/透传裸的 `guidance_scale` 给这个模型。
- **分辨率白名单**：优先 1024×1024 和 2048×2048（官方明确验证过的方形档位，实测都正常）；1536×1536 官方文档里从未以方形形式出现过（§5.4），实测有残留伪影风险，不建议作为生产分辨率；4096 直接 OOM。
- **`skip_first_n_steps` 默认应保持 `0`（官方原生路径）**：§5.5 最初在错误格式下测出"skip=1 只有约 1/4 概率换来干净内容"；§5.7 用正确格式复测后，残留风险大幅降低（8 组敏感边界内容里只有 1 组出问题）但没有完全消失。两次测试方向一致，都支持"默认不开、按需可选"，只是格式修好之后代价小了很多。
- **推荐的安全过滤处理方式**：检测输出图片是否命中"Image blocked by safety filter"这个固定、可 OCR 识别的文案——命中就判定为拦截，交给业务层决定是否提示用户改写 prompt。这一层依然值得保留作为防御性兜底（§5.7 证明格式修好后 skip 仍有低概率残留风险，§5.8 证明模型对暴力内容完全不设防，都可能需要业务层介入），但不再是应对"~30% 高频失败"的主力手段——那个数字本身已被 §5.6 证明主要是格式问题造成的。
- **最终推荐配置**：`sampler_preset=V4_DEFAULT_20` + `skip_first_n_steps=0`（默认）+ 分辨率限制在 1024×1024/2048×2048 + TP=2。
- **（历史记录，已被 §5.6 大幅推翻，保留供参考）"~30% 跑题/乱码水印率"及配套的换 seed 重试分析**：§6.4 的 30 组样本测试、以及针对 windmill/subway/typewriter 的换 seed 重试实验（同 seed 100% 确定性复现 SHA256 一致；换新 seed 干净率约 33%，明显低于整体 70%），**全部是在简化 JSON 格式下测出来的**。§5.6 证明格式修好之后，这些原本失败的案例在 `skip=0`/`skip=1` 下都变得完全正常——也就是说"30% 失败率"、"换 seed 只有 33% 干净率"这些具体数字大概率显著高估了修复格式之后的真实现网风险，不应该再作为容量规划或 SLA 承诺的依据。**门面层实现 §5.6 的 schema 转换层之后，建议用正确格式重新测一批样本，重新校准真实的残留失败率**，再决定是否需要、以及需要多重的输出检测+重试兜底。

---

## 7. 当前代码状态

已落地到本仓库工作区，**尚未 commit**：

```text
M  vllm_omni/diffusion/registry.py                                  （+6 行，注册 Ideogram4Pipeline）
?? vllm_omni/diffusion/models/ideogram4/                             （__init__.py / pipeline_ideogram4.py / ideogram4_transformer.py）
?? tests/diffusion/models/ideogram4/                                 （PR 自带的 4 个测试文件，尚未跑过）
?? recipes/Ideogram/Ideogram4.md                                     （PR 自带的 recipe 文档）
```

`pipeline_ideogram4.py` 相对 PR 原始版本的全部改动点：`may_enable_cache_dit` 依赖移除（§4.1）、`extra_args` 统一读取入口 + `skip_first_n_steps` 支持（§5.3）、`_validate_resolution()` 分辨率兜底（§4.5）、`SAMPLER_PRESETS`/`_PRESET_BY_STEP_COUNT` 采样预设注册表（§4.6）、`seed`/`generator` 改为从 `req.sampling_params` 读取（§4.7）、`_decode()` 的 latent 反归一化改用硬编码 `_LATENT_SHIFT`/`_LATENT_SCALE` 而非 `self.vae.bn`（§4.8）。`ideogram4_transformer.py` 的改动点：6 处 `quant_config` 传参修复（§4.2，其中 1 处例外保留 `None`）。

跳过未采用：PR 的 `vllm_omni/quantization/factory.py` 补丁（原因见 §2）。

---

## 8. 遗留待办

1. **（优先级最高，§5.6 发现后新增）门面层实现"自由文本 → 官方结构化 JSON schema"转换层**：§5.6 证明本报告绝大部分"安全过滤误拦"和"skip 导致乱码跑题"，根因是 prompt 用了简化写法而不是官方 `high_level_description`/`style_description`/`compositional_deconstruction.elements[]` 结构化格式。这是消除误拦/乱码的关键路径，不再是可选的锦上添花项。可以合并原来的 `magic_prompt`/prompt 结构化移植任务一起做（原第 2 条）。做完之后应该：① 用正确格式重新采集 §6.4 的三档预设成功率数据（原数字已被证明主要是格式问题拉低的）；② 重新评估是否还需要输出检测+重试兜底、需要多重。
2. **`skip_first_n_steps` 默认值与检测逻辑的落地**：结论已定（§5.6/§5.7/§6.5）——默认 `skip_first_n_steps=0`，门面层检测"Image blocked by safety filter"固定文案作为防御性兜底，不建议自动切到 `skip=1` 重试（§5.7 证明格式修好后 skip=1 仍有低概率残留致乱风险，虽然远低于错误格式下的水平）。目前还没有在 `forward()`/门面层实现这套逻辑。
3. **暴力/血腥内容的独立过滤方案**（§5.8 新发现）：模型自带安全对齐只覆盖 NSFW 类内容，不拦截暴力/血腥内容（6 组图形化暴力测试全部通过，`skip=0` 官方原生路径也不例外）。如果业务对此有合规要求，需要评估接入 `docs/safety.md` 提到的 Hive 视觉审查（`--hive-visual-key`）或内部等效方案——本次调研完全没有配置这一层。
4. **varlen attention 改造**（§4.4）——纯性能优化项，2048² 实测显存/内容都正常（35GB），varlen 化能不能进一步降显存/提速还没测，不是正确性问题，优先级可以往后放。
5. **测试文件尚未跑过**：PR 自带的 4 个测试（`test_ideogram4_nf4_device.py`/`test_ideogram4_pipeline.py`/`test_ideogram4_qkv_layout.py`/`test_ideogram4_transformer_tp.py`）拷进仓库了但没有执行验证。
6. **fp8 checkpoint 未验证**：本次只测了 NF4（`ideogram-ai/ideogram-4-fp8` 对应的加载路径未触碰）。
7. **1536×1536 残留伪影排查**：用正确采样参数后 2048 已确认正常，1536 仍有一次残留文字伪影（§5.4），样本量 1，需要换 seed/prompt 多测几次才能判断是不是这个分辨率桶本身覆盖较少导致的系统性问题；也应该用 §5.6 的正确 JSON 格式重测，避免重蹈"格式问题伪装成分辨率问题"的覆辙。
8. **`skip_first_n_steps` 对误拦的选择性仍未验证**：不清楚这个 workaround 是"只放行本该通过的内容"还是"对任何内容都放行"（包括真正应该拦截的），需要用正确格式 + 真正应该拦截的负面 case 补测，见 §5.3 caveat 1。
9. **图片质量未做系统性评估**：只验证了"能不能生成、像不像 prompt"，没有做美学质量、文字渲染准确度等官方 benchmark 维度的对比。
10. **代码尚未 commit**，需要确认最终改动范围（尤其是是否要补 varlen/schema 转换层之后再一起提交）。
11. **用真正生效的 seed 重跑关键配对实验**：§4.7 修复 `seed` bug 之前，本报告早期"固定 seed 对照"实验实际上是独立随机抽样，不是严格配对。§5.6/§5.7 已经用正确 seed + 正确格式重新验证了 skip 相关的核心结论，但三档预设的详细成功率对比（§6.4 表格）仍然是旧数据，建议跟第 1 条一起用正确格式+正确 seed 重新采集。
12. **没有跑过官方 harness 本身做端到端输出对比**：目前的"移植正确性"结论建立在逐行代码对比（§3 表格 + §4.7/§4.8 静态审查）之上，从未实际装官方 harness 环境、用同一 prompt/seed 跑一遍官方实现来对比像素级输出。§4.7/§4.8 两个 bug 都是静态审查发现的，说明这种逐行核对确实有效，但也说明不能排除还有类似量级的问题没被读到——如果条件允许，后续应该在某台调试机上装 `ideogram4` 官方 repo 的 Python 环境实际跑一次做交叉验证。

---

## 9. 涉及的关键文件

- `vllm_omni/diffusion/models/ideogram4/pipeline_ideogram4.py` — 双 DiT 编排、`_verify_prompts`（本次未移植）、量化配置传参（本次修复点）、`_validate_resolution()`（§4.5）、`SAMPLER_PRESETS`/`_PRESET_BY_STEP_COUNT`（§4.6）、`seed`/`generator` 读取修复（§4.7）、`_LATENT_SHIFT`/`_LATENT_SCALE` 常量表 + `_decode()` 修复（§4.8）。
- `/Users/reputationly/Desktop/code/api/ideogram4/src/ideogram4/latent_norm.py` — 官方硬编码的 `LATENT_SHIFT`/`LATENT_SCALE` 常量表来源，§4.8 排查的关键证据。
- `/Users/reputationly/Desktop/code/api/ideogram4/src/ideogram4/autoencoder.py` — 官方 `AutoEncoder.bn` 定义处，逐行确认它在 `Encoder`/`Decoder.forward()` 里从未被引用，§4.8 排查的关键证据。
- `vllm_omni/diffusion/worker/diffusion_model_runner.py`（`_initialize_generator`，约 531 行）— 引擎侧 `req.sampling_params.generator` 的构建位置，§4.7 排查的关键证据；确认了 `pipeline.forward(req)` 只传一个位置参数（约 642 行）。
- `vllm_omni/diffusion/models/ideogram4/ideogram4_transformer.py` — 6 处 `quant_config` 硬编码 bug 的落点、`_preprocess_nf4_weights` 反量化逻辑、`_build_segment_mask` 稠密 mask（遗留问题）。
- `vllm_omni/diffusion/registry.py` — `Ideogram4Pipeline` 注册项。
- `vllm_omni/quantization/factory.py` / `int8_config.py` 同源的 `_build_bitsandbytes`/`_construct_override`/`_SERIALIZED_BNB_MARKERS` — 本仓库已有的离线/在线量化检测机制，本次直接复用，未改动。
- `/Users/reputationly/Desktop/code/api/ideogram4/src/ideogram4/{safety.py,caption_verifier.py,pipeline_ideogram4.py}` — 官方 harness 源码，§5 三层安全机制的证据来源。
- `/nfs-models/wuhanjisuan894/models/Ideogram-4-NF4` — 权重，`transformer/config.json`/`unconditional_transformer/config.json` 里的 `quantization_config` 块是 §4.2/4.3 排查的关键证据。
- `/Users/reputationly/Desktop/code/api/ideogram4/docs/prompting.md` — 官方结构化 caption schema 的 "Full example"（`high_level_description`/`style_description`/`compositional_deconstruction.elements[]`），§5.6 最重大发现的直接依据；同一份文档里的 `© Anadolu Agency via Getty Images` 水印标注示例也是"模型可能学到复现图库水印"这一早期猜测（现已判定为次要因素，主因是格式问题）的来源。
- `/Users/reputationly/Desktop/code/api/ideogram4/docs/safety.md` — 官方安全机制说明，明确 Hive 文本/图像审查是独立于模型自身对齐的外部 API、未配置即视为"不受支持的部署配置"，§5.8 判断模型自身对齐只覆盖 NSFW 类内容的依据。
- `/Users/reputationly/Desktop/code/api/ComfyUI-Ideogram4-DirectJSON-Modified/` — 社区 ComfyUI 插件（基于 `ComfyUI-KJNodes` 修改），本身只是通用 JSON/bbox 可视化编辑器、没有"4 个框解锁"的硬编码逻辑；其自带示例 workflow（`workflows/【Work-Fisher】Ideogram4全自动文生图.json`）里的真实 4-element 结构化 prompt 模板是发现 §5.6 根因的直接线索来源。
- `~/Desktop/ideogram4-fullschema-test/` — 本机软链目录，18 张 §5.6 验证测试的原始输出图片（`skip0_*.png`/`skip1_*.png`，9 个案例 × 2 种 skip 设置）。
