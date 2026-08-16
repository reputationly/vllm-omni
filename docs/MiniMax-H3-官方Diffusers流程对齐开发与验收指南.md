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

---

## 3. 固定官方 oracle

开发和验收必须固定版本。开始执行时记录实际 commit；除非用户明确要求升级，不要静默更新。

| 项目 | 本任务初始基线 | 作用 |
|---|---|---|
| MiniMax-H3 模型仓库 | `d21241f0a4b3acbb34c97dae47fa417b7065e438` | 权重配置、processor、model index |
| Hugging Face Diffusers | `d6726f38a0c5ca6c06a8f227fb7bade3486ed98d` (`0.40.0.dev0`) | 官方公开 harness |
| vLLM-Omni | `99530748bd9eb9a49faf54fe882e1386048f39c1` | 开发起点；工作区另有未提交修改 |

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

本任务编写时，工作区已有用户修改：

```text
M  vllm_omni/diffusion/models/minimax_h3/denoise_loop.py
M  vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py
?? tests/diffusion/models/minimax_h3/test_minimax_h3_reference_geometry.py
```

这些改动属于用户，执行 agent 不得 reset、checkout、覆盖或删除。发生冲突时应在现有改动上
最小化合并，并在交付说明中逐项指出。

官方兼容行为在验收前不得静默替换现网默认值。实现必须具备明确的契约选择，例如启动级
`legacy` 与 `official_diffusers_v1` 两种模式；具体接入现有配置系统的方式由执行 agent 在阅读
配置代码后决定，但必须满足：

- 选择发生在启动或模型部署级，不允许同一实例按请求偷偷混用不同 RNG/预处理契约；
- 旧模式默认行为不变；
- 日志和结果 metadata 能明确显示当前契约；
- 后续验收通过后，产品可以把“原版 BF16”实例固定到官方兼容模式；
- Turbo、Pruned、W8A8 等实验实例不自动继承未经验证的官方兼容假设。

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
- 音频初始噪声再次新建 `manual_seed(seed)` generator；
- condition noise 又按 condition 重建 generator；
- 因而相同 seed 的随机流与官方不同。

实现要求：

- 为一次输出创建一个 CPU `torch.Generator`；
- 按官方 block 执行顺序把同一个 generator 传下去；
- 视觉 condition 一项一次 draw，顺序严格跟 references 一致；
- 再 draw 完整视频 latent；
- 再 draw channel-major 音频 rows；
- VAE posterior 的固定 `encode_seed=42` 是独立 RNG，不得消费请求 generator；
- Ref2VA clean audio anchor 不产生额外 condition-noise draw；
- 若未来支持用户传入预生成 `latents`/`audio_latents`，跳过对应 draw 后，后续 RNG 位置也要
  跟官方一致。

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

这是官方 BF16 基线的契约。`match` 是某些蒸馏模型建议的训练匹配策略，不属于本任务的官方
原版权重基线。

当前工作区对 reference 图片增加了短边、no-upscale 和面积上限开关。必须保留这些用户改动，
但在 `official_diffusers_v1` 模式下应强制或校验：

```text
short edge = 2048
no-upscale = false
max-pixels = disabled
```

若配置冲突，应启动失败或打印明确错误，不能静默生成一个不再属于官方契约的结果。

### 5.4 P0：Ref2VA 视频不能经过有损中间转码

当前 vLLM-Omni 先用 ffmpeg 转成 `libx264 + yuv420p` 的临时 MP4，再给 VAE 和 Qwen
processor 使用。这会引入颜色子采样、量化和二次解码差异。

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

测试覆盖 16 kHz、32 kHz、44.1 kHz、48 kHz，mono/stereo，独立音频和视频内嵌音轨。

### 5.6 P1：presentation、text encoder 和 packed sequence

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

文本 encoder 使用 Qwen3-VL 第 50 层未 post-norm hidden state。vLLM-Omni TP 实现允许 BF16
舍入差异，但必须先证明 token 与 vision 输入完全一致，再比较 embedding；不要把预处理错误归因
于 TP rounding。

### 5.7 P1：scheduler 和 denoise recurrence

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

### 5.8 P2：Transformer、Attention、TP/SP 和 VAE

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

---

## 6. 推荐实现结构

具体类名可以调整，但职责必须拆开，不能继续把所有逻辑塞进 pipeline `forward`：

```text
MiniMaxH3InferenceContract
├── LegacyContract
└── OfficialDiffusersV1Contract
    ├── resolve_output_geometry()
    ├── prepare_fl2va_keyframes()
    ├── prepare_ref2va_images()
    ├── decode_and_normalize_ref2va_video()
    ├── decode_and_normalize_ref2va_audio()
    └── make_request_generator()
```

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
- FL2VA 第一/尾帧像素结果；
- Ref2VA 图片 2048 策略；
- 视频 fps 重采样的 frame index；
- reference 截断、旋转、resize shape；
- 原生音频采样率到 32 kHz 的单次路径；
- token/presentation/packing/position IDs；
- scheduler 的 sigmas、timesteps 和 step；
- official 模式与冲突资源开关不能静默并存；
- legacy 默认行为保持不变。

常规单元测试不应依赖开发机旁边恰好存在 Diffusers checkout。可采用两层方式：

1. 把官方纯函数输出固化成小型 fixture/expected tensor；
2. 另提供 opt-in parity 测试，通过显式环境变量定位固定 Diffusers checkout，重新生成并验证 fixture。

fixture 必须记录来源 commit 和生成脚本，不能手工抄一组不透明数字。

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
| R5 | Ref2VA | 视频+独立音频+图片 | 混合 reference/rotary clock |

至少准备一个非 24 fps 视频、一个带 rotation metadata 的视频，以及 16/44.1/48 kHz 音频。

---

## 8. 验收判据

### 8.1 必须 exact 的项目

以下项目使用 `torch.equal`、逐项 JSON 比较或等价的 zero-tolerance 断言：

- 所有 token IDs、tags、indices、masks；
- output/reference shape 和对齐后的帧数；
- reference 顺序及 label 编号；
- position IDs、timestamps、packed layout；
- sigma 和 timestep tensors；
- CPU generator 输入相同时产生的 noise tensors；
- FL2VA 官方纯 PIL 预处理后的像素；
- 已经走官方同算法且不涉及设备 kernel 的纯 CPU tensor。

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

如果没有 GPU 权限，允许先交 1–7，但状态必须写成“等待 GPU 验证”，不能写“完成”或“已对齐”。

---

## 10. 禁止事项

- 不在 Mac 上下载或运行 H3 权重。
- 不修改官方 Diffusers oracle 来掩盖 vLLM 差异。
- 不启用 W8A8、FP8、NVFP4、Turbo、Pruned、LoRA、Cache-DiT 后做官方基线验收。
- 不把 `match` reference resize 用在官方原版基线。
- 不把 vLLM `lossless` 的自比较 SSIM=1 当作对官方 harness 的证明。
- 不只比较最终 MP4；必须比较上游阶段。
- 不因为同 seed 输出不同就直接归咎于 BF16，先验证 RNG 和预处理。
- 不覆盖用户已有未提交修改。
- 不改现网默认行为或部署配置，除非用户明确授权并且验收已通过。
- 不提交 checkpoints、视频、音频或大 tensor artifacts。

---

## 11. 执行顺序与停点

严格按以下顺序推进：

1. 记录三个仓库 commit、工作区状态和依赖版本。
2. 建立官方/vLLM source mapping 与差异清单。
3. 写 CPU oracle fixtures 和 parity 测试，先让它们在当前实现上暴露差异。
4. 实现显式 `official_diffusers_v1` 契约，不改变 legacy 默认。
5. 依次修 RNG、FL2VA、Ref2VA 图片、视频、音频。
6. 对齐 presentation、packing 和 scheduler。
7. 跑完整 CPU 测试、lint 和 diff 检查。
8. 准备 GPU runners 和 manifest；到这里若无服务器权限就停止并汇报。
9. 在 GPU 服务器跑阶段级 parity，先 T1/F1/F2，再 R1–R5。
10. 形成误差包络、最终媒体和盲测材料。
11. 把本文第 9 节全部交付物发给主 agent，等待复验。

只有主 agent 完成代码审查、证据复跑和媒体验收后，才能讨论把“原版 BF16”产品实例切换到
官方兼容契约。优化版继续作为独立实验轨道存在。

---

## 12. 给执行 agent 的最终完成定义

满足以下全部条件才可声明“开发完成，等待主 agent 验收”：

- legacy 默认路径没有回归；
- `official_diffusers_v1` 是显式、可观察、可固定版本的契约；
- P0 的 RNG 和媒体预处理差异全部有代码修复和 exact CPU 测试；
- 离散契约全部 exact；
- GPU 阶段结果、数值指标、版本 manifest 和媒体 A/B 齐全；
- 没有使用任何量化、蒸馏或 Cache 结果冒充官方基线；
- 所有未通过项均被诚实列出；
- 用户已有工作区改动完整保留。

如果缺少 GPU 结果，正确状态是：

```text
静态开发与 CPU 契约测试完成；官方 BF16 GPU parity 尚未验证，不具备上线或“官方一致”结论。
```
