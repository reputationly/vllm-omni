# MiniMax-H3 官方 Diffusers 流程对齐：开发任务书与验收指南

> 面向后续执行开发的 agent。先完整读完本文，再改代码。
>
> 目标是让使用官方 BF16 权重的 vLLM-Omni H3 推理路径，在输入契约、随机数流、
> 媒体预处理、packed sequence、调度器和解码流程上尽量对齐官方公开 Diffusers
> harness。完成开发后，不自行宣布上线；提交本文要求的证据，由主 agent 复验。

---

## 1. 任务结论先行

这不是一次权重转换或性能优化任务，而是一次**推理契约对齐**任务。

必须实现并验证一条明确的官方兼容路径：

```text
MiniMax-H3 官方 BF16 权重
        +
官方 Diffusers 推理契约
        +
vLLM-Omni 分布式执行能力
```

目标分三层：

1. **离散契约完全一致**：token、shape、帧数、reference 顺序、position IDs、
   packed layout、sigma schedule 必须完全一致。
2. **随机输入完全一致**：相同 generator 初态下，条件噪声、视频噪声、音频噪声以及
   RNG 消费顺序必须与官方一致。
3. **浮点结果尽量接近**：文本编码器、DiT、VAE 在 TP、FlashAttention、分块解码等
   后端差异下允许 BF16/FP16 舍入误差，但必须量化误差并证明最终效果接近。

不要承诺多卡路径与官方单卡路径逐 bit 一致。可以承诺的是：流程契约一致、数学公式一致、
浮点误差受控、最终结果经过成套 A/B 验证。

这里的“官方兼容”特指**进入模型后的推理语义**。生产 API 对文件大小、容器、编码、总素材时长
和资源消耗可以保留更严格的准入规则，但必须与推理契约分开配置、分开记录，不能把生产准入
限制误写成 Diffusers 的模型契约。

---

## 2. 环境边界：Mac 禁止运行模型

### 2.1 Mac 只允许做什么

Mac 路径用于代码阅读和开发：

```text
/Users/reputationly/Desktop/code/api/vllm-omni
/Users/reputationly/Desktop/code/api/MiniMax-H3
/Users/reputationly/Desktop/code/api/diffusers
```

Mac 上只允许：

- 静态阅读和比较代码；
- 修改 vLLM-Omni 源码；
- 编写不加载权重的 CPU 单元测试；
- 导入 Diffusers 类、检查配置和函数签名；
- 运行格式检查、语法检查和纯函数测试。

Mac 上禁止：

- 下载 MiniMax-H3 权重；
- 加载任何 H3 checkpoint；
- 尝试用 CPU 或 MPS 运行 H3；
- 把模型权重、生成视频或大体积中间 tensor 写入仓库；
- 因为本机没有权重而擅自改成小模型、随机权重并据此宣布端到端等价。

### 2.2 GPU 服务器才允许做什么

以下操作只能在用户指定的 GPU 服务器进行：

- 加载官方 BF16 权重；
- 运行官方 Diffusers harness；
- 运行 vLLM-Omni 完整 H3；
- 保存真实 prompt embedding、DiT 输出、latent 和媒体结果；
- 做显存、延迟、SSIM、LPIPS、音频和人工 A/B 验证。

没有 GPU 服务器地址、权重路径或运行授权时，先完成不依赖这些信息的代码和测试，列出待运行
命令，但不要在 Mac 上补跑。

### 2.3 已授权的 A100 验证节点与免密入口

以下 20 台 A100 节点已经配置 SSH 公钥免密登录。三种写法等价，执行 agent 优先使用短别名：

| 节点 | 短别名 | 完整别名 | 原始入口 |
|---|---|---|---|
| 0021 | `gpu21` | `dev-gpustack-a100-0021` | `ssh -p 43043 root@111.172.214.16` |
| 0022 | `gpu22` | `dev-gpustack-a100-0022` | `ssh -p 43044 root@111.172.214.16` |
| 0023 | `gpu23` | `dev-gpustack-a100-0023` | `ssh -p 43045 root@111.172.214.16` |
| 0024 | `gpu24` | `dev-gpustack-a100-0024` | `ssh -p 43046 root@111.172.214.16` |
| 0025 | `gpu25` | `dev-gpustack-a100-0025` | `ssh -p 43047 root@111.172.214.16` |
| 0026 | `gpu26` | `dev-gpustack-a100-0026` | `ssh -p 43051 root@111.172.214.16` |
| 0027 | `gpu27` | `dev-gpustack-a100-0027` | `ssh -p 43052 root@111.172.214.16` |
| 0028 | `gpu28` | `dev-gpustack-a100-0028` | `ssh -p 43053 root@111.172.214.16` |
| 0029 | `gpu29` | `dev-gpustack-a100-0029` | `ssh -p 43054 root@111.172.214.16` |
| 0030 | `gpu30` | `dev-gpustack-a100-0030` | `ssh -p 43055 root@111.172.214.16` |
| 0031 | `gpu31` | `dev-gpustack-a100-0031` | `ssh -p 43056 root@111.172.214.16` |
| 0032 | `gpu32` | `dev-gpustack-a100-0032` | `ssh -p 43057 root@111.172.214.16` |
| 0033 | `gpu33` | `dev-gpustack-a100-0033` | `ssh -p 43058 root@111.172.214.16` |
| 0034 | `gpu34` | `dev-gpustack-a100-0034` | `ssh -p 43059 root@111.172.214.16` |
| 0035 | `gpu35` | `dev-gpustack-a100-0035` | `ssh -p 43060 root@111.172.214.16` |
| 0036 | `gpu36` | `dev-gpustack-a100-0036` | `ssh -p 43061 root@111.172.214.16` |
| 0037 | `gpu37` | `dev-gpustack-a100-0037` | `ssh -p 43062 root@111.172.214.16` |
| 0038 | `gpu38` | `dev-gpustack-a100-0038` | `ssh -p 43063 root@111.172.214.16` |
| 0039 | `gpu39` | `dev-gpustack-a100-0039` | `ssh -p 43064 root@111.172.214.16` |
| 0040 | `gpu40` | `dev-gpustack-a100-0040` | `ssh -p 43065 root@111.172.214.16` |

例如：

```bash
ssh gpu31
ssh dev-gpustack-a100-0040
ssh -p 43056 root@111.172.214.16
```

这些机器已由 GPUStack 纳管。用户确认它们当前没有运行具体业务，因此可以用于本任务的诊断、
对拍和 GPU 验证；这不代表可以改变节点的纳管状态。开始每次实验前仍应只读检查 GPU、进程、
容器和端口占用。如果发现任务开始后出现其他工作负载，换用空闲节点，不得停止、抢占或清理
不属于本任务的进程。

#### GPUStack 节点保护规则

- **不得停止、重启、升级、重装、重配或删除 `gpustackworker`**，不得修改它的 systemd、
  Docker、启动参数、注册信息、标签、token、数据目录和网络规则；
- 不得重启 Docker/containerd、NVIDIA driver、网络或整台机器，也不得执行 `docker system prune`
  等可能影响 GPUStack 的全局清理；
- 不做全局 `pip`/`conda`/`apt` 安装。使用任务专属目录、独立虚拟环境或独立容器；不得覆盖
  系统 Python、CUDA、PyTorch、Diffusers 或现有 vLLM-Omni 环境；
- 不修改现有模型、镜像、缓存和权重文件。需要复用时只读挂载；需要新版本时放到带 run-id 的
  独立路径；
- 测试进程、端口、容器、临时文件、环境变量和日志必须使用可识别的 run-id，禁止与 GPUStack
  worker 或其他任务共用名称；
- 实验只允许操作本任务自己创建的资源。不得 kill、删除、覆盖或 chmod/chown 不明资源；
- 问题解决或整体验证结束后，停止本任务进程和容器，释放端口与显存，删除本任务临时目录，
  撤销临时配置，并恢复到实验前状态；长期保留的 artifacts 只放在用户指定位置；
- 交付报告必须记录使用过的节点、时间段、创建和清理的资源，以及实验前后
  `gpustackworker` 状态。若无法确认环境已还原，不得宣称任务完成。

除非用户另行明确授权，`gpustackworker` 的任何异常都只允许做只读诊断并汇报，不能自行修复。

---

## 3. 固定官方 oracle

开发和验收必须固定版本。开始执行时记录实际 commit；除非用户明确要求升级，不要静默更新。

| 项目 | 本任务初始基线 | 作用 |
|---|---|---|
| MiniMax-H3 模型仓库 | `d21241f0a4b3acbb34c97dae47fa417b7065e438` | 权重配置、processor、model index |
| Hugging Face Diffusers | `d6726f38a0c5ca6c06a8f227fb7bade3486ed98d` (`0.40.0.dev0`) | 官方公开 harness |
| vLLM-Omni | `a160673c5165c5b31545a2e732b7d204a4c0245c` | 本次复核基线；H3 改动已提交 |

官方公开入口是 MiniMax-H3 根目录的 `model_index.json`：

```text
_class_name = MiniMaxH3ModularPipeline
_blocks_class_name = MiniMaxH3Blocks
```

因此，本任务中的“官方 harness”指固定 commit 的 Hugging Face Diffusers
`MiniMaxH3ModularPipeline`。MiniMax-H3 仓库中的 `FL2VA/`、`Ref2VA/` 还保留旧格式索引，
但没有提供一份可独立审计的旧版完整 pipeline 源码。不要把无法看到的 MiniMax 内部 harness
宣称为已经复现。

### 3.1 官方代码的阅读顺序

在 Diffusers 仓库依次阅读：

1. `src/diffusers/modular_pipelines/minimax_h3/modular_blocks_minimax_h3.py`
2. `src/diffusers/modular_pipelines/minimax_h3/before_encoder.py`
3. `src/diffusers/modular_pipelines/minimax_h3/encoders.py`
4. `src/diffusers/modular_pipelines/minimax_h3/before_denoise.py`
5. `src/diffusers/modular_pipelines/minimax_h3/denoise.py`
6. `src/diffusers/schedulers/scheduling_minimax_h3.py`
7. `src/diffusers/models/transformers/transformer_minimax_h3.py`
8. `src/diffusers/modular_pipelines/minimax_h3/decoders.py`
9. 两个 `autoencoder_kl_minimax_h3*.py`

在 vLLM-Omni 中逐一映射到：

```text
vllm_omni/diffusion/models/minimax_h3/
├── pipeline_minimax_h3.py
├── presentation.py
├── encoder.py
├── vae.py
├── condition_noise.py
├── packed_sequence.py
├── packed_tokens.py
├── time_request.py
├── denoise_loop.py
├── scheduling_minimax_h3_euler_ancestral.py
└── minimax_h3_transformer.py
```

### 3.2 oracle 的使用规则

- Diffusers 仓库只作为 oracle，尽量不要为了迁就 vLLM-Omni 修改它。
- 官方 runner 使用独立 Python 环境，不能污染现网 vLLM-Omni 环境。
- GPU 服务器安装与上表一致的 Diffusers commit。
- oracle 和 vLLM 必须指向同一份官方 BF16 checkpoint 内容。
- 每次结果都记录模型路径、revision、文件摘要、Diffusers commit、PyTorch/CUDA/cuDNN、GPU、
  attention backend 和完整请求参数。

---

## 4. 保护现有工作区和现网行为

开始任务时先执行：

```bash
git status --short
git diff -- vllm_omni/diffusion/models/minimax_h3
```

本次复核时工作区是干净的。此前 H3 修改已分别落在：

```text
99530748  H3 reference image memory/comments changes
226dd901  H3 reference image geometry switches and tests
```

执行时仍须重新记录 `git status`，因为用户可能在本文之后继续修改。任何新出现的改动都属于
用户；不得 reset、checkout、覆盖或删除。发生冲突时应在现有改动上最小化合并，并在交付说明
中逐项指出。

官方兼容行为在验收前不得静默替换现网默认值。实现必须具备明确的契约选择，例如启动级
`legacy` 与 `official_diffusers_v1` 两种模式。推理契约与生产准入策略是两个正交维度：

```text
inference_contract = legacy | official_diffusers_v1
admission_policy   = production_safe_v1 | parity_fixture_v1
```

`official_diffusers_v1` 决定 RNG、reference 顺序、默认帧数、几何、归一化、截断、packing 和
scheduler；`production_safe_v1` 决定上传文件大小、容器/codec 白名单和资源预算。生产策略可以
拒绝官方内存接口理论上能接收的素材，但拒绝原因必须属于 admission，不能改变已经接收素材的
模型语义。GPU oracle 对拍应使用双方都能接收的安全交集，或先离线解码成同一份内存 fixture。

具体接入现有配置系统时必须满足：

- 选择发生在启动或模型部署级，不允许同一实例按请求偷偷混用不同 RNG/预处理契约；
- 旧模式默认行为不变；
- 日志和结果 metadata 能明确显示当前契约；
- 后续验收通过后，产品可以把“原版 BF16”实例固定到官方兼容模式；
- Turbo、Pruned、W8A8 等实验实例不自动继承未经验证的官方兼容假设。

优先在 `OmniDiffusionConfig` 和 deploy YAML 中增加启动级字段，不继续堆环境变量。当前 HEAD
已经注册 deploy-only pipeline `minimax_h3_dit`，现有 H3 deploy YAML 也使用这个名称；显式
传入但无法应用的 deploy config 会报错，不是静默回退。仍必须补测试证明：

- `pipeline: minimax_h3_dit` 能把契约字段传到 H3 pipeline；
- pipeline 名称错误、字段无效或配置无法应用时启动失败；
- 启动日志和结果 metadata 输出最终解析后的契约与准入策略；
- 不存在“配置写了 official、实际运行 legacy”的静默路径。

### 4.1 reference 顺序需要请求契约支持

官方 `references` 是一条有序的异构列表。顺序会决定 `<Picture N>`、`<Video N>`、
`<Audio N>` 编号、condition-noise 消费顺序和共享 rotary 时钟。当前公开入口把素材拆成
`multi_modal_data.image/video/audio` 三个桶，pipeline 又按“图片 → 每个视频的音轨和视频 →
独立音频”重组，因而不能表达任意交错的官方请求。

严格官方路径必须新增或保留一条 canonical ordered references 表示，并从 API 解析一直透传到
presentation、媒体预处理、RNG 和 packing。legacy 桶输入继续保留；若 official 模式暂时兼容
桶输入，只能按明确的 legacy canonicalization 生成顺序并在 metadata 标记，不能声称支持任意
官方 reference 顺序。

不要把现有 `quality=lossless` 当作“官方 Diffusers 完全一致”。它目前只表示不启用
Cache-DiT 等有损近似，是 vLLM-Omni 自身的 native path。

---

## 5. 已确认的差异和开发优先级

按 P0 → P1 → P2 顺序处理。不要先调最终视频指标，必须先把上游输入契约对齐。

### 5.1 P0：请求级 RNG 必须对齐

官方契约使用**一个请求 generator 连续消费**：

```text
每个视觉 condition 的噪声（按 packed/reference 顺序）
    → 目标视频初始噪声
    → 目标音频初始噪声
```

当前 vLLM-Omni 的关键差异：

- 视频初始噪声新建 `manual_seed(seed)` generator；
- 音频初始噪声再次新建 `manual_seed(seed)` generator，因此不是简单的先后顺序不同，而是从
  与视频相同的随机流起点重新抽取；
- 每个视觉 condition 都重建 `manual_seed(seed)`，每个音频 condition 都重建
  `manual_seed(seed + 1)`；
- 视觉 condition 当前不是按自身 shape draw，而是先按
  `full_t = max(target_latent_t + n_cond, latent_t)` 生成更大的 tensor，再切前缀；
- 因而相同 seed 的随机流与官方不同。

实现要求：

- 为一次输出创建一个 CPU `torch.Generator`；
- 按官方 block 执行顺序把同一个 generator 传下去；
- 视觉 condition 一项一次、按**官方 condition 自身完整 shape** draw，顺序严格跟 ordered
  references 一致；不能只把 generator 改成共享而保留当前 `full_t` draw-and-slice；
- 再 draw 完整视频 latent；
- 再 draw channel-major 音频 rows；
- VAE posterior 的固定 `encode_seed=42` 是独立 RNG，不得消费请求 generator；
- Ref2VA clean audio anchor 不产生额外 condition-noise draw；
- 若未来支持用户传入预生成 `latents`/`audio_latents`，跳过对应 draw 后，后续 RNG 位置也要
  跟官方一致。

这是 seed/output 契约切换，不是对现网 seed 的透明 bugfix。即使 T2VA 没有 reference，官方
路径也会因视频与音频共用连续随机流而改变同 seed 输出；只能在新契约实例中启用，legacy
必须保持原结果。

测试要求：纯 CPU、无权重，至少覆盖 T2VA、单图 FL2VA、双图 FL2VA、图+音频 Ref2VA、
多 reference Ref2VA，并直接比较每次 draw 和最终 generator state。

### 5.2 P0：FL2VA 双关键帧几何必须对齐

官方行为：

- 第一张关键帧是 geometry anchor，LANCZOS 拉到目标画布；
- 第二张/尾关键帧先按 cover 比例 LANCZOS 缩放，再居中裁剪；
- 裁剪尺寸和中心偏移使用官方代码的 `round` 与整数公式，不能用一个“看起来一样”的通用 crop
  helper 替代。

当前 vLLM-Omni 对所有 FL2VA 图片直接 `resize((width, height), LANCZOS)`，尾帧宽高比不同时
会被拉伸。

测试必须覆盖：

- 首尾图同宽高比；
- 首尾图不同宽高比；
- 奇数尺寸、会产生 1 像素中心偏移的案例；
- 只传首帧、只传尾帧、首尾都传的官方矩阵。

输出像素应与从官方算法提取的纯 PIL oracle 完全相同。

### 5.3 P0：Ref2VA 图片语义必须钉死

在本文固定的官方 Diffusers commit 中，released checkpoint 的图片 reference 行为是：

- 固定短边 `2048`；
- 保持宽高比并对齐到 32；
- 小图也会上采样；
- 不使用目标视频画布；
- 不设置面积上限。

这是官方 BF16 基线的契约。当前 vLLM 三个资源保护开关默认全关时，行为已经等于这组默认值，
因此这里的工作重点是固定契约和防止冲突配置，不是改默认 resize 算法。

当前工作区对 reference 图片增加了短边、no-upscale 和面积上限开关。必须保留这些用户改动，
但在 `official_diffusers_v1` 模式下应强制或校验：

```text
short edge = 2048
no-upscale = false
max-pixels = disabled
```

若配置冲突，应启动失败或打印明确错误，不能静默生成一个不再属于官方契约的结果。

同时修正 `pipeline_minimax_h3.py` 中“2048 是本适配层自选值、无官方出处”的过时注释。
固定 oracle 的 `MiniMaxH3Ref2VASetupStep` 已明确声明
`ConfigSpec("reference_image_short_edge", 2048)`；代码注释必须引用这一事实，不能继续把它
描述成 Qwen 面积封顶策略。

### 5.4 P0：Ref2VA 视频不能经过有损中间转码

当前 vLLM-Omni 先用 ffmpeg 转成 `libx264 + yuv420p` 的临时 MP4，再给 VAE 和 Qwen
processor 使用；Qwen 路径还会从该 H.264 文件逐帧调用 ffmpeg 抽 PNG。这会引入颜色子采样、
量化和二次解码差异，也产生不必要的逐帧 subprocess。

官方公开路径的核心语义：

- 从源容器解码 RGB24 帧并读取真实 fps；
- 按 display rotation 转正；
- 按官方公式重采样到 24 fps；
- 截断到目标视频对齐后的 `num_frames`；
- 按 reference 自身比例解析 768 短边、`768*1344` 面积上限和 32 对齐画布；
- 使用 PIL LANCZOS 逐帧缩放；
- normalized frames 同时服务于 VAE condition 和 Qwen 视觉输入。

实现可以使用 PyAV、无损帧缓存或等价数据流，但不得引入 H.264/H.265/JPEG 等有损中间表示。
如果为内存原因需要流式处理，必须证明逐帧结果与官方 array 实现相同。

同时核对当前 vLLM “允许参考视频长于目标视频”的扩展。官方固定 commit 会把 normalized
reference 截到目标 `num_frames`。官方兼容模式必须服从固定 oracle；扩展行为只能留在 legacy
模式，并写明差异。

### 5.5 P0：Ref2VA 音频只能重采样一次

当前嵌入视频音轨路径会先用 ffmpeg 强制成 44.1 kHz，再由 audio VAE 转为 32 kHz。
官方路径在源采样率上截断，单声道扩为双声道，然后直接重采样到 32 kHz。

官方兼容模式必须：

- 保留解码后的原生采样率；
- 在原生采样率上按目标/参考时长截断；
- mono 通过通道复制变成 stereo；
- 只做一次到 32 kHz 的 resample；
- 音频 reference 的顺序与 presentation、packed audio rows 完全一致。

官方 `_normalize_audio_condition(max_duration=num_frames / fps)` 会把每条 reference 音频截到
目标视频对齐后的时长。当前 vLLM 主要按素材自身时长和总预算处理，未执行这条目标时长截断；
这是会改变 audio packed rows 数量的 P0 离散差异，必须实现并 exact 测试。

测试覆盖 16 kHz、32 kHz、44.1 kHz、48 kHz，mono/stereo，独立音频和视频内嵌音轨。

### 5.6 P0：默认值与模型级校验包络

固定 oracle 的 workflow 默认 `num_frames=124`；当前 vLLM 的 Ref2VA 默认 124，但 T2VA/FL2VA
默认 209。`official_diffusers_v1` 下三类任务默认都必须是 124，legacy 保持现值。

模型级输入语义也不同：

| 项目 | 固定 Diffusers oracle | 当前 vLLM 入口 |
|---|---|---|
| reference 宽高比 | `[1/4, 4]` | `[0.4, 2.5]` |
| reference 边长 | 仅要求正数 | `[256, 5760]` |
| 输出对齐后时长 | `[5, 15]` 秒 | `[2, 16]` 秒 |
| 文件/容器/codec/字节数 | 内存输入层不定义 | 有生产 gate |

official 模式的模型级几何、默认值和对齐后时长必须跟 oracle；文件大小、MP4/MOV、H.264/H.265、
AAC/MP3、fps 与资源预算等仍可由 `production_safe_v1` 拒绝。测试和错误码必须能区分“官方
模型契约不合法”和“生产准入策略拒绝”。如果产品保留更窄的 admitted domain，发布说明必须
写“官方推理语义兼容，但 API 输入域受限”，不能宣称整个 API surface 完全相同。

### 5.7 P1：presentation、text encoder 和 packed sequence

以下离散值必须完全一致：

- prompt token IDs；
- image/video placeholder 数量；
- `<Picture N>`、`<Video N>`、`<Audio N>` 的顺序和编号；
- Qwen image/video grid；
- Qwen video block timestamps；
- MiniMax token tags；
- video/audio/text row indices；
- position IDs、rotary clock；
- condition/update masks；
- `cu_seqlens` 和最大序列长度。

当前 `packed_sequence.py` 的核心网格、时间轴、音频 channel-major 布局和浮点求和顺序与固定
oracle 已经近乎逐行一致，应优先通过 direct oracle 测试保护，而不是先重写。vLLM 为运行时
把 `used` 补齐到 64 的倍数，并把 pad 放在第二个 document；官方没有这些 pad rows。比较规则：

- 对 canonical prefix `[0:used]` 的所有离散 tensor 做 exact comparison；
- 单独验证 pad rows 由 `cu_seqlens/document_id` 隔离，不进入 update/output masks；
- 证明是否存在 pad 不改变 canonical rows 的模型输出；
- 不直接拿包含 pad 的完整 tensor shape 与官方做错误的 zero-tolerance 比较。

文本 encoder 使用 Qwen3-VL 第 50 层未 post-norm hidden state。vLLM-Omni TP 实现允许 BF16
舍入差异，但必须先证明 token 与 vision 输入完全一致，再比较 embedding；不要把预处理错误归因
于 TP rounding。

参考视频进入 VAE 前的帧数也要做 pure-function 对拍。官方向下截到 `17n+5`；当前 vLLM 的远端
VAE processor 看起来会通过 chunk trim 做同类对齐，但尚未形成直接证据。覆盖临界帧数后再
判断是否存在缺口，当前不要把它列成已确认 P0。

### 5.8 P1：scheduler 和 denoise recurrence

两边应使用：

```text
video shift = 12
audio shift = 3
base sigmas = linspace(1, 0, num_inference_steps), CPU float32
t = 1 - sigma
N 个 sigma 节点 = N-1 次 DiT forward
x0 = xt + (1 - t) * velocity
Euler eta=0，以 float32 计算 sigma ratio 与 blend
```

纯函数测试必须对官方 `MiniMaxH3Scheduler` 做 exact comparison，包括完整 sigma、timesteps、
条件 row timestep plan 和每一步 scheduler 输出。不要只对几个手写常量。

当前 scheduler 和 condition-row timestep 公式经静态复核已基本对齐，应以 oracle 测试钉死；
不要在没有失败 fixture 的情况下重写。

### 5.9 P2：Transformer、Attention、TP/SP 和 VAE

当上面全部对齐后，再评估执行后端差异：

- checkpoint QKV/SwiGLU 排列；
- RoPE；
- RMSNorm；
- token refiner；
- TP row-parallel all-reduce；
- FlashAttention/SDPA/cuDNN attention；
- VAE FP32 权重、FP16 autocast、latent FP16 round-trip；
- VAE tiling/patch parallel；
- 输出反归一化和音视频 mux。

先比较单层或单步中间 tensor，最后才比较完整视频。不要为了追求 bitwise 相同关闭所有必要的
多卡部署能力；先给每种后端建立可解释的数值误差包络。

FL2VA keyframe VAE posterior 的固定 seed 42 当前可能恰好与官方使用同一条 CPU 默认 RNG 和
相同 shape，值可完全相同，但这依赖当前 checkpoint `vae_module.py` 的实现细节。增加一个无权重
fixture 或最小分布对象测试钉死该行为，避免后续把它误接到请求 generator 或因 device 迁移而
悄悄改变。

---

## 6. 推荐实现结构

两条契约实际只在少数策略点分叉。不要复制一套平行 pipeline，也不建议建立会迅速膨胀的
`LegacyContract`/`OfficialContract` 类层次。优先定义一个启动时解析完成的 frozen strategy
dataclass，通过现有 helper 透传：

```text
MiniMaxH3InferenceStrategy
├── name
├── rng_mode
├── visual_condition_noise_shape_mode
├── fl2va_keyframe_resize_mode
├── reference_order_mode
├── reference_video_decode_mode
├── reference_video_target_truncation
├── reference_audio_resample_mode
├── reference_audio_target_truncation
├── default_num_frames_by_task
└── model_validation_semantics
```

具体字段名可以调整，但必须在启动阶段把配置解析成不含歧义的值。legacy 最安全的实现形态是
尽量不改现有分支的执行语句，只在已确认的 6～8 个分歧点由 strategy 选择 official helper。
请求过程中不得再次从散落的环境变量猜测契约。

再把随机 draw 顺序集中到一个可单测的对象或函数中，例如：

```text
MiniMaxH3RequestNoisePlan
├── draw_visual_condition_noise(...)
├── draw_video_noise(...)
└── draw_audio_noise(...)
```

要求：

- 所有纯几何、采样、packing 和 scheduler helper 都能在 CPU、无权重条件下测试；
- GPU 模型 forward 不负责猜测 reference 预处理策略；
- legacy 与 official contract 的差异显式存在，不靠散落的环境变量组合推断；
- 现有 reference 图片资源保护开关继续可用，但不能伪装成官方模式；
- ordered references 从请求入口到 packing 全程保序，不能中途重新分桶再猜顺序；
- `admission_policy` 独立于 inference strategy，生产 gate 不侵入纯模型 helper；
- 不引入生产运行时对本地 Diffusers checkout 的强依赖。

---

## 7. 测试体系

### 7.1 Mac/普通 CI：不加载权重

新增或扩展以下测试：

```text
tests/diffusion/models/minimax_h3/
├── test_minimax_h3_official_contract.py
├── test_minimax_h3_reference_geometry.py
├── test_minimax_h3_packing.py
└── test_minimax_h3_contract.py
```

必须覆盖：

- RNG draw 顺序和 generator state；
- condition noise 的 exact draw shape，不只测试 seed；
- FL2VA 第一/尾帧像素结果；
- Ref2VA 图片 2048 策略；
- 视频 fps 重采样的 frame index；
- reference 截断、旋转、resize shape；
- 原生音频采样率到 32 kHz 的单次路径；
- reference 音频截到对齐后的目标时长；
- ordered image/video/audio interleave 从 API 到 presentation/packing 不丢序；
- official 三任务默认 124 帧，legacy 默认值不变；
- 模型校验与生产 admission 的错误类型可区分；
- 参考视频 VAE 前 `17n+5` 临界帧数对拍；
- token/presentation/packing/position IDs；
- canonical packed prefix 与官方 exact，64-alignment pad 独立验证；
- scheduler 的 sigmas、timesteps 和 step；
- keyframe posterior seed 42 的 CPU 行为；
- official 模式与冲突资源开关不能静默并存；
- deploy YAML 契约字段能到达 H3，错误配置显式失败；
- legacy 默认行为保持不变。

常规单元测试不应依赖开发机旁边恰好存在 Diffusers checkout。可采用两层方式：

1. 把官方纯函数输出固化成小型 fixture/expected tensor；
2. 另提供 opt-in parity 测试，通过显式环境变量定位固定 Diffusers checkout，重新生成并验证 fixture。

fixture 必须记录来源 commit 和生成脚本，不能手工抄一组不透明数字。

这应是本任务第一个可执行开发步骤。固定 Diffusers 中的 `resolve_canvas_size`、
`align_num_frames`、FL2VA 裁剪算术、两个 packed-sequence builder、scheduler，以及视频/音频
normalize helper 都可以在 Mac 上无权重运行。预计 packing/scheduler 大部分先绿，几何/RNG
先红；以真实失败结果更新差异清单，再碰产品代码。

### 7.2 GPU：阶段级 parity

官方 runner 与 vLLM runner 都应支持按阶段 dump，建议目录：

```text
<artifact-root>/<run-id>/
├── manifest.json
├── request.json
├── official/
│   ├── tokens.json
│   ├── geometry.json
│   ├── rng.safetensors
│   ├── prompt_embeds.safetensors
│   ├── packed.safetensors
│   ├── step_000.safetensors
│   ├── final_latents.safetensors
│   └── output.mp4
├── vllm_omni/
│   └── ...同名文件...
└── comparison.json
```

不得把这些大文件提交到 Git。脚本应支持选择性 dump，避免默认保存 50 步全部 tensor。

阶段比较顺序：

1. request/geometry；
2. token IDs/tags/position/packing；
3. RNG tensors；
4. prompt embeddings；
5. 第 0 步 DiT 输入；
6. 第 0 步 DiT velocity；
7. 第 0 步 scheduler 输出；
8. 选定中间步；
9. final video/audio latents；
10. VAE 输出和 MP4/WAV。

对 packed dump，官方 tensor 与 vLLM 的 canonical prefix 对拍；vLLM 额外的 64-alignment pad
单列在 manifest，验证它属于隔离 document。不要让预期存在的 pad rows 淹没真正差异。

前一阶段失败时先修前一阶段，不要继续用最终 SSIM 猜原因。

### 7.3 GPU 用例矩阵

所有用例使用官方 BF16 权重、`quality=lossless`、关闭 Cache-DiT/量化/LoRA/剪枝，显式设置
seed、步数、分辨率、帧数、shift。快速调试可用较少步数，最终验收至少有一组 50 sigma 节点。

| ID | 任务 | 条件 | 主要覆盖点 |
|---|---|---|---|
| T1 | T2VA | 纯文本 | RNG、text、joint video/audio denoise |
| F1 | FL2VA | 仅首帧 | geometry anchor、VAE condition |
| F2 | FL2VA | 首尾帧且宽高比不同 | cover-crop、condition 顺序 |
| R1 | Ref2VA | 单图 | 2048 图片策略、presentation |
| R2 | Ref2VA | 多图 | reference 顺序、RNG 多次 draw |
| R3 | Ref2VA | 图+独立音频 | audio resample、audio packing |
| R4 | Ref2VA | 带音轨视频 | fps、rotation、内嵌音轨 |
| R5 | Ref2VA | 视频+独立音频+图片，按异构顺序交错 | request 保序、rotary clock |

至少准备一个非 24 fps 视频、一个带 rotation metadata 的视频，以及 16/44.1/48 kHz 音频。

---

## 8. 验收判据

### 8.1 必须 exact 的项目

以下项目使用 `torch.equal`、逐项 JSON 比较或等价的 zero-tolerance 断言：

- 所有 token IDs、tags、indices、masks；
- output/reference shape 和对齐后的帧数；
- reference 顺序及 label 编号；
- position IDs、timestamps、canonical packed prefix；
- sigma 和 timestep tensors；
- CPU generator 输入相同时产生的 noise tensors；
- FL2VA 官方纯 PIL 预处理后的像素；
- 已经走官方同算法且不涉及设备 kernel 的纯 CPU tensor。

vLLM 为内核对齐增加的 pad rows 不要求与官方“形状相同”，但其边界、document ID、mask、
`cu_seqlens` 和对 canonical rows 的零影响必须 exact 验证。

### 8.2 允许容差的项目

以下项目同时报告 `max_abs`、`mean_abs`、`max_rel`、cosine，相应媒体再报告感知指标：

- Qwen3-VL prompt embeddings；
- DiT BF16 输出；
- TP/SP/attention backend 输出；
- VAE 编解码 tensor；
- final latents。

不要一开始拍脑袋设一个 SSIM 门槛。先建立三种基线误差：

1. 官方同环境重复运行的确定性；
2. 官方允许的不同 attention backend/执行拓扑之间的误差；
3. vLLM TP1/最接近官方后端与 vLLM 生产 TP/SP 后端之间的误差。

在 `comparison.json` 中给出误差包络，再由主 agent 确认最终阈值。执行 agent 不得仅以“视频看起来
差不多”作为通过理由。

### 8.3 人工验收

每个代表用例输出并排 A/B，隐藏来源和左右顺序，检查：

- 主体身份、纹理和参考一致性；
- 动作、镜头轨迹和首尾帧约束；
- 文本语义和物理一致性；
- 嘴型、节奏、音画同步；
- 语音内容、音色、声像、底噪；
- 画面闪烁、断裂和过锐。

人工验收由用户或主 agent 完成，执行 agent 只能准备材料，不能代替最终签字。

---

## 9. 交付物

执行 agent 交付时必须提供：

1. vLLM-Omni patch，且不包含模型权重或生成媒体；
2. 新增/更新的 CPU 单元测试；
3. 官方 Diffusers GPU runner；
4. vLLM-Omni 阶段 dump runner；
5. 自动比较脚本；
6. 固定版本的 manifest 示例；
7. GPU 执行命令和环境说明；
8. 全部 CPU 测试结果；
9. GPU artifacts 的绝对路径或可访问位置；
10. `comparison.json` 和简短差异报告；
11. 未通过项、已知限制和回滚方式；
12. `git status --short` 与 `git diff --stat`。
13. GPU 节点使用与还原记录：节点名、时间段、实验前后 worker/GPU 状态、本任务创建和清理的
    进程、容器、端口、目录及临时配置。

如果没有 GPU 权限，允许先交 1–7，但状态必须写成“等待 GPU 验证”，不能写“完成”或“已对齐”。

---

## 10. 禁止事项

- 不在 Mac 上下载或运行 H3 权重。
- 不修改官方 Diffusers oracle 来掩盖 vLLM 差异。
- 不启用 W8A8、FP8、NVFP4、Turbo、Pruned、LoRA、Cache-DiT 后做官方基线验收。
- 不把 vLLM `lossless` 的自比较 SSIM=1 当作对官方 harness 的证明。
- 不只比较最终 MP4；必须比较上游阶段。
- 不因为同 seed 输出不同就直接归咎于 BF16，先验证 RNG 和预处理。
- 不假设工作区永远干净；每次开工先检查并保护用户新产生的修改。
- 不改现网默认行为或部署配置，除非用户明确授权并且验收已通过。
- 不提交 checkpoints、视频、音频或大 tensor artifacts。
- 不把 production admission gate 冒充官方 Diffusers 模型校验。
- 不把分桶后推导出的固定顺序声称为官方任意 ordered references。

---

## 11. 执行顺序与停点

严格按以下顺序推进：

1. 记录三个仓库 commit、工作区状态和依赖版本。
2. 先写 direct CPU oracle parity 测试并在当前实现上运行；把 packing、scheduler、几何、RNG、
   视频帧数对齐和音频 normalization 的实际绿/红结果固化。
3. 依据测试结果更新 source mapping 和差异清单；已绿的 P1 只补回归保护，不重写。
4. 增加 ordered references 请求表示，以及分离的 `inference_contract` / `admission_policy` 配置；
   验证 deploy YAML 字段真正到达 H3。
5. 实现显式 `official_diffusers_v1` strategy，不改变 legacy 默认。
6. 依次修 RNG（含 draw shape）、FL2VA、Ref2VA 视频、音频；图片默认语义已对齐，只修契约、
   冲突校验和过时注释。
7. 对 presentation、packing 和 scheduler 只修 oracle 测试实际发现的缺口。
8. 跑完整 CPU 测试、lint、deploy-config 测试和 diff 检查。
9. 准备 GPU runners 和 manifest；到这里若无服务器权限就停止并汇报。
10. 选择空闲节点，记录实验前 GPU/process/container/port 与 `gpustackworker` 状态；不得改动
    worker，然后跑阶段级 parity，先 T1/F1/F2，再 R1–R5。
11. 形成误差包络、最终媒体和盲测材料。
12. 停止并删除本任务创建的运行资源，释放显存和端口，撤销临时配置；复查节点并证明已经
    恢复实验前状态。
13. 把本文第 9 节全部交付物发给主 agent，等待复验。

只有主 agent 完成代码审查、证据复跑和媒体验收后，才能讨论把“原版 BF16”产品实例切换到
官方兼容契约。优化版继续作为独立实验轨道存在。

---

## 12. 给执行 agent 的最终完成定义

满足以下全部条件才可声明“开发完成，等待主 agent 验收”：

- legacy 默认路径没有回归；
- `official_diffusers_v1` 是显式、可观察、可固定版本的契约；
- inference contract 与 production admission policy 已分离；
- strict official 路径能表达并保持异构 ordered references；
- P0 的 RNG 和媒体预处理差异全部有代码修复和 exact CPU 测试；
- official 模式三任务默认 124 帧，模型级校验与 oracle 一致；
- canonical 离散契约全部 exact，vLLM pad rows 被证明完全隔离；
- H3 deploy YAML 的契约字段已验证可达，错误配置不会静默回落；
- GPU 阶段结果、数值指标、版本 manifest 和媒体 A/B 齐全；
- 所有用过的 GPUStack 节点均已清理本任务资源并还原，`gpustackworker` 未被改动；
- 没有使用任何量化、蒸馏或 Cache 结果冒充官方基线；
- 所有未通过项均被诚实列出；
- 用户已有工作区改动完整保留。

如果缺少 GPU 结果，正确状态是：

```text
静态开发与 CPU 契约测试完成；官方 BF16 GPU parity 尚未验证，不具备上线或“官方一致”结论。
```
