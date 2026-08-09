# vLLM-Omni · ERNIE-Image-Turbo A100 海报生成实验与优化报告

> 文档性质：引擎适配核验、真机测试、提示词优化复盘与 GPUStack 部署建议  
> 测试日期：2026-08-01  
> 测试节点：`dev-gpustack-a100-0025`  
> 模型：`PaddlePaddle/ERNIE-Image-Turbo`  
> 引擎：vLLM-Omni `0.25.0`  
> 本地代码基线：`feature/hunyuan-image3-a100-nf4@5724f72f893b80142f82900b5d6f63be09990ded`  
> 一句话结论：**ERNIE-Image-Turbo 已在鲲鹏 ARM + 单张 A100-40G 上稳定完成中文海报生成。补充 A/B 证明：关闭 CPU offload 并把最大面积限制为 1024×1536 后，受控请求由 19.34 秒降至 6.20 秒，连续 10 轮 PE on/off 无显存泄漏。逐层 offload 虽能在单卡跑通 2160×3840，但耗时 52.9～75.7 秒且出现明显纵向重复构图，不建议作为原生 4K 生产档。详细排版提示建议关闭 PE；Turbo 生产参数锁定 8 步、guidance 1.0。**

---

## 0. 结论先行

| 维度 | 结论 |
|---|---|
| 可运行性 | ✅ vLLM-Omni 原生加载成功，无需模型代码 patch |
| 硬件形态 | ✅ 单张 A100-PCIE-40GB；生产建议关闭 CPU offload，并限制最大面积为 1,572,864 pixels |
| 4K | ⚠️ 逐层 offload 可跑通 2160×3840，但已超出官方推荐尺寸，构图重复且显存峰值约 39.3GB；建议低分辨率生成后独立超分 |
| 中文文字 | ✅ 本轮指定文字均能准确生成；8 张丰富画面海报中文字内容 8/8 正确 |
| 版式遵循 | 7/8 首轮严格通过；电影海报出现标题/副标题重复 |
| Turbo 推荐参数 | `num_inference_steps=8`、`guidance_scale=1.0` |
| 负向提示词 | ❌ `guidance_scale=1.0` 时不进入 CFG；同 seed 有/无负向提示词输出逐字节一致 |
| PE | ✅ 可在每个请求中用 `extra_params.apply_pe` 切换，不必重启服务 |
| PE 默认值 | vLLM-Omni 代码默认 `true`；海报生产建议业务层默认设为 `false` |
| 图生图/改稿 | ❌ 当前 ERNIE-Image-Turbo 及两套引擎实现都是 T2I，不支持图片输入 |
| GPUStack | ✅ 已有内置 `vLLMOmni` 后端、验证过的引擎镜像和 `/ready` 健康检查；必须显式配置 `categories:["image"]` |
| New API | ⚠️ 现有 GPUStackPlus 图片链路未透传 PE/步数/guidance；需小幅扩展适配器和体验区 |

当前建议不是继续盲目加负向词，而是：

1. 把所有硬约束写成正向、可见、可定位的描述；
2. 详细海报默认 `apply_pe=false`，短而模糊的创意提示再开启 PE；
3. 人物和飞行器尺寸用“相对主体 + 所在区域 + 角色层级”描述；
4. 需要基于原图改字、改色、局部重绘时切到 Qwen-Image-Edit，不要把 ERNIE T2I 当编辑模型使用。

---

## 1. 测试边界与环境

### 1.1 硬件与集群

| 项 | 配置 |
|---|---|
| CPU 架构 | ARM aarch64，鲲鹏平台 |
| GPU | 4× NVIDIA A100-PCIE-40GB |
| 本轮使用 | 单卡 GPU0 |
| 节点 | `dev-gpustack-a100-0025` |
| 访问方式 | `ssh -p 43047 root@111.172.214.16` |
| GPUStack worker | v2.2.0（`c6e6b91`） |
| 模型存储 | NFS 共享目录 |

### 1.2 模型权重

模型路径：

```text
/nfs-models/wuhanjisuan894/models/ERNIE-Image-Turbo
```

下载完成后的总大小约 30GB，关键组成如下：

| 组件 | 大小 |
|---|---:|
| Transformer shard 1 | 9.4GB |
| Transformer shard 2 | 5.7GB |
| Text encoder | 7.2GB |
| Prompt Enhancer | 7.2GB |
| VAE | 161MB |
| Tokenizer / PE tokenizer | 各约 17MB |

模型的 `model_index.json`、scheduler、Transformer、text encoder、VAE、PE 和两套 tokenizer 均通过关键文件校验。本轮下载来源是 ModelScope 的 `PaddlePaddle/ERNIE-Image-Turbo` snapshot。

### 1.3 镜像与服务

```text
image: crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/
       reputationly/vllm-omni:arm64-a100-latest
image id: be6068d0f779
image size: 33GB
container: ernie-omni-turbo-test
port: 8091
```

启动命令：

```bash
docker run -d \
  --name ernie-omni-turbo-test \
  --gpus device=0 \
  --ipc=host \
  -p 8091:8091 \
  -v /nfs-models:/nfs-models \
  -v /nfs-output:/nfs-output \
  --entrypoint vllm \
  crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-omni:arm64-a100-latest \
  serve /nfs-models/wuhanjisuan894/models/ERNIE-Image-Turbo \
  --omni --host 0.0.0.0 --port 8091 --enable-cpu-offload
```

实验结束复查时容器已经连续运行约 9 小时，状态仍为 `Up`。

---

## 2. 引擎支持核验

### 2.1 vLLM-Omni

ERNIE-Image 在 vLLM-Omni 中是原生 pipeline，而不是把 diffusers 脚本包在 HTTP 外层。关键事实：

- `support_image_input = False`：只支持文生图；
- PE 最大生成长度固定为 `512` token；
- PE 只在 rank 0 执行，再用 `broadcast_object_list` 广播；
- 请求参数名是 `extra_params.apply_pe`，缺省为 `true`；
- dummy/warmup 请求会强制关闭 PE；
- negative prompt 不经过 PE；
- PE 产生空结果或运行异常时会回退原始提示词；
- 已有 PE、RoPE、TP 单元测试；
- online/offline E2E 使用的是 `baidu/ERNIE-Image` Base，不是 Turbo 质量一致性测试。

本轮代码定位：

```text
vllm_omni/diffusion/models/ernie_image/pipeline_ernie_image.py
tests/diffusion/models/ernie_image/test_ernie_image_pe.py
tests/diffusion/models/ernie_image/test_ernie_image_rope.py
tests/diffusion/models/ernie_image/test_ernie_image_tp.py
tests/e2e/offline_inference/test_ernie_image_expansion.py
tests/e2e/online_serving/test_ernie_image_expansion.py
```

### 2.2 LightX2V 对照

LightX2V 也有 ERNIE-Image / Turbo 原生 T2I runner、Transformer、scheduler、VAE、Mistral text encoder 和 PE，Turbo 配置同样是 8 步、关闭 CFG、`guidance_scale=1.0`。但当前实现成熟度有几处差异：

| 项 | vLLM-Omni | LightX2V |
|---|---|---|
| 请求级 PE 开关 | `extra_params.apply_pe` | 配置项 `use_pe` |
| PE 最大新 token | 固定 512 | tokenizer `model_max_length`；官方文件实值 2048 |
| 多卡 PE 一致性 | rank0 生成并广播 | 未见 rank0-only 广播 |
| PE 异常处理 | warning + 回退原 prompt | 空串/异常回退保护较弱 |
| PE offload | 随引擎模型级 offload | 可每请求 GPU↔CPU 搬运 |
| ERNIE 测试 | PE/RoPE/TP + Base E2E | 未见专门测试与 parity 报告 |
| 图生图 | 不支持 | 不支持 |

因此本硬件形态先用 vLLM-Omni 建立质量和性能基线是更稳妥的选择。LightX2V 的价值主要在后续接入既有量化/offload 体系时做同 seed parity，而不是取代本轮首测引擎。

### 2.3 Turbo distilled、CFG 与负向词路径核验

当前 vLLM-Omni 的 CFG 条件是：

```python
guidance_scale > 1 and not self.is_distilled
```

`is_distilled` 构造默认值为 `False`，Turbo 权重的 `model_index.json` 没有该字段，当前 factory 也没有注入该参数。真机补测证明：`guidance_scale=2.0` 与 `1.0` 的同 seed 输出 hash 不同，说明当前 Turbo 运行时确实会在 guidance 大于 1 时进入 CFG。

但不应由此直接得出“必须把 Turbo 强制设为 `is_distilled=true`”：

- 官方资料确认 Turbo 是 DMD + RL 蒸馏版，推荐 8 步、guidance 1.0；
- 但官方 diffusers `ErnieImagePipeline` 没有 `is_distilled` 参数，它同样按 `guidance_scale > 1` 决定是否进入 CFG；
- 因此强制 `is_distilled=true` 会改变官方 pipeline 的参数语义，不能当成无争议的 parity 修复。

补测还发现了一个更具体的实现缺口：API 已把 `negative_prompt` 放进 prompt 字典，但 ERNIE pipeline `forward` 只取出了正向 `prompt`，没有从字典取出 `negative_prompt`。因此即使强制 guidance 2.0 进入 CFG，两个完全不同的负向提示词仍然产生逐字节一致的 PNG。

生产结论：Turbo 按官方参数锁定 `num_inference_steps=8`、`guidance_scale=1.0`，不需要为此改引擎或重出镜像。若要支持 Base 或实验性 CFG，再修复 `negative_prompt` 取值并补回归测试；`is_distilled` 则应经官方 parity 后决定保留、删除或改成显式策略。

---

## 3. 启动与资源占用

### 3.1 启动时间

| 阶段 | 观测值 |
|---|---:|
| 首次 Python/import 阶段 | 约 45 秒 |
| Transformer 两个 shard 加载 | 24.73 秒 |
| pipeline 报告的总模型加载 | 79.09 秒 |
| 容器启动到 health 200 | 约 109 秒 |

启动时启用了 `ModelLevelOffloadBackend`，Transformer 与 text encoder 互斥驻留；日志显示 torch.compile 捕获 36 个 `ErnieImageSharedAdaLNBlock`。

### 3.2 显存

| 状态 | GPU0 显存 |
|---|---:|
| pipeline 报告模型加载占用 | 13.73GiB |
| 加载完成后的进程占用 | 约 14.28GiB |
| 首轮 1024×1024 请求后 | 约 27.5GB |
| 1024×1536 海报批次峰值 | 约 29.9GB |

单卡 40GB 有明确余量。当前没有必要为了“能跑”启用多卡，后续只有在吞吐压测证明单实例不足时才考虑多实例横向扩容。

### 3.3 CPU offload / 无 offload 补充 A/B

2026-08-01 在空闲的 0025 节点，以相同镜像、prompt、seed、8 步和 1024×1536 做了重新验证。两个模式均为热 NFS 缓存，逐个启动、逐个清理，不与其他业务容器共卡。

| 模式 | PE | HTTP 总耗时 | 服务端耗时 | GPU 峰值 | 容器主机内存 | 结果 |
|---|---|---:|---:|---:|---:|---|
| 无 offload | off | 6.20s | 5.91s | 36873 MiB | 约 3～6 GiB | 成功 |
| CPU offload | off | 19.34s | 19.04s | 约 29851 MiB | 约 30.5 GiB | 成功 |
| 无 offload | on | 18.76s | 18.59s | 约 36911 MiB | 约 3～6 GiB | 成功 |
| CPU offload | on | 26.15s | 25.98s | 约 29851 MiB | 约 30.5 GiB | 成功 |

结论：

- PE off 时，无 offload 快约 3.1 倍；PE on 时快约 1.39 倍。PE on 的 prompt 扩写本身占据较多时间，所以整体收益比例较小。
- 无 offload 将 Transformer、文本编码器和 PE 权重保留在 GPU，显著减少 PCIe 阶段搬运，同时主机内存也更低。
- 1024×1536 第一次请求后显存由启动稳态约 31123 MiB 上升到 36873 MiB；这是 CUDA/编译缓存保留，不会立即回落。

#### 连续请求与 PE 交替稳定性

连续执行 10 轮 `PE off → on → off → on`：

- 10/10 HTTP 200；
- PE off 稳态约 6.0 秒；PE on 约 18.9～22.8 秒；
- 第 2～4 轮显存从 36873 小幅升到 36915 MiB；第 4～10 轮固定为 36915 MiB，不再阶梯增长；
- 容器主机内存稳定在约 3.20～3.21 GiB；
- 未出现 OOM、服务重启或输出失败。

两个 1024×1536 请求同时提交时，引擎明确为 `batch_size=1`，请求被串行处理：约 6.02 秒和 11.76 秒，总墙钟约 12 秒，没有形成双请求显存峰值。

#### 分辨率边界

| 尺寸 | 像素 | PE | 结果 | 观测显存 |
|---|---:|---|---|---:|
| 1024×1536 | 1,572,864 | off/on | 均成功 | 约 36915 MiB，稳定 |
| 1152×1728 | 1,990,656 | off/on | 均成功 | 一度达到 40321 MiB，仅剩约 122 MiB |
| 1280×1920 | 2,457,600 | off | 本次成功 | 1 秒采样未超过此前 40321 MiB，不能排除瞬时峰值 |

1152×1728 虽然跑通，但已没有生产安全余量；不同形状还会触发新的编译/allocator 缓存。生产必须通过：

```text
--max-generated-image-size 1572864
```

把最大面积锁定在 1024×1536。更大尺寸若确有需求，应使用逐层 offload 的独立实验模型池，不能与无 offload 快速池混用同一个模型配置。

### 3.4 逐层 offload 与 4K 极限验证

vLLM-Omni 的 ERNIE Transformer 已声明：

```python
_layerwise_offload_blocks_attrs = ["layers"]
```

使用 `--enable-layerwise-offload` 启动后，日志确认 36 个 DiT block 均安装逐层 offload hook。该机制在启动时固定启用：每次只预取当前/下一 block 的权重，文本编码器、PE、VAE 和非 block 模块继续驻留 GPU。它不是根据显存水位在 OOM 前自动切换，也不能卸载当前计算所需的注意力与 VAE 激活。

单张 A100-40G、PE off、8 步、guidance 1.0 的递增测试：

| 尺寸 | 像素 | HTTP 总耗时 | GPU 峰值 | 结果 |
|---|---:|---:|---:|---|
| 1024×1536 | 1,572,864 | 6.25s | 22841 MiB | 成功 |
| 1536×2048 | 3,145,728 | 13.55s | 29763 MiB | 成功 |
| 1920×2560 | 4,915,200 | 24.76s | 36965 MiB | 成功 |
| 2160×3840 | 8,294,400 | 52.87s | 38879 MiB | 成功 |

真实中文海报提示词再做 2160×3840 对照：

| PE | 服务端耗时 | 观测 GPU 峰值/驻留 | 质量观察 |
|---|---:|---:|---|
| off | 52.77s（HTTP 53.78s） | 峰值 39331 MiB；结束约 38877 MiB | 标题、星形在纵向重复，构图被拉长 |
| on | 75.72s | 运行中约 39361 MiB | 重复文字减少，但星形仍纵向重复；PE 额外增加约 23s |

两张输出均为真实 2160×3840 PNG、约 24MB，不是 API 尺寸虚报；但“能计算”不等于“具备原生 4K 质量”。百度官方模型卡推荐的尺寸为 1024×1024、848×1264、1264×848、768×1376、896×1200、1376×768、1200×896，均约 1MP，没有 4K 原生生成承诺。本轮 4K 的重复主体与文字正是明显的分布外表现。

测试产物：

```text
/nfs-output/ernie-image-turbo-test/layerwise-4k/
  pe-false.png
  pe-true.png
  pe-false-thumb.jpg
  pe-true-thumb.jpg
```

生产建议分成两个稳定配置，而不是在同一进程里临近 OOM 动态 offload：

1. **快速生成池**：不启用 offload，`--max-generated-image-size 1572864`；用于 1024×1536 及以下生成。
2. **4K 交付链路**：先按官方推荐尺寸或 1024×1536 生成，再交给独立超分服务放大到 4K。这样能保留模型熟悉的全局构图，超分只补像素细节。
3. **原生大图实验池**：仅在业务必须时单独部署 `--enable-layerwise-offload`，严格串行并保留显存安全余量；当前不建议把 2160×3840 暴露为稳定 SLA。

如果要实现“显存接近阈值时只卸载部分 block”，需要新增自适应常驻窗口、显存水位控制、编译图重捕获/缓存管理和请求调度，现有引擎没有这个能力。相较于两个固定模型池，这条路径复杂、难以预测延迟，也更容易把一次可控的 HTTP 400 变成进程级 OOM，暂不建议投入。

---

## 4. API、PE 与推荐请求格式

### 4.1 请求示例

```json
{
  "model": "/nfs-models/wuhanjisuan894/models/ERNIE-Image-Turbo",
  "prompt": "竖版科技主题商业海报……",
  "size": "768x1376",
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "seed": 31003,
  "extra_params": {
    "apply_pe": false
  }
}
```

注意：部分 recipe 写的是 `use_pe`，但当前 vLLM-Omni pipeline 实际读取的是 `apply_pe`。业务 API 建议暴露更容易理解的布尔字段 `use_prompt_enhancer`，在网关内部映射为：

```text
use_prompt_enhancer → extra_params.apply_pe
```

### 4.2 PE 开关结论

- PE 模型在服务启动时加载；
- 每个请求都能独立选择开或关；
- 切换 PE 不需要重启或重新加载权重；
- 不传 `apply_pe` 时，当前代码默认开启；
- 产品默认值建议改为关闭，避免详细文案和版式被二次改写；
- 对“一句话创意”“只有主题没有构图”的短提示词，可以开启 PE。

---

## 5. PE 同 seed 对照实验

### 5.1 条件

```text
size: 1024×1024
steps: 8
guidance_scale: 1.0
seed: 12345
```

提示词：

```text
现代极简科技发布会海报，深蓝色背景，中央准确显示大标题「智启未来」，
副标题「2026 人工智能创新大会」，底部显示「8月18日 · 武汉」，
清晰中文排版，高对比度，专业商业设计
```

### 5.2 结果

| 模式 | HTTP | 服务端耗时 | curl 总耗时 | 文字 |
|---|---:|---:|---:|---|
| PE off | 200 | 16.47s | 16.93s | 全部正确 |
| PE on | 200 | 17.35s | 17.85s | 全部正确 |

结论：

- 受控条件下 PE 增加约 0.89 秒总延迟；
- 两张图片 hash 不同，PE 明显改变了画面组织；
- 该短提示词上，PE off 更偏科技电路，PE on 更偏简洁抽象构图；
- 两者都没有出现错字，因此不能仅凭这组样本宣称 PE 提升了文字准确率；
- PE 的主要作用是重写/扩写意图，不是 OCR 后处理。

产物：

```text
/nfs-output/ernie-image-turbo-test/pe_off.png
/nfs-output/ernie-image-turbo-test/pe_on.png
```

---

## 6. 丰富画面海报批测

### 6.1 条件

```text
8 个主题
1024×1536
8 steps
guidance_scale=1.0
PE on
seed=24001～24008
```

### 6.2 结果

| # | 主题 | 指定文字 | 耗时 | 结果 |
|---:|---|---|---:|---|
| 1 | 未来城市 AI 大会 | 智启未来 / 2026 人工智能创新大会 / 8月18日·武汉 | 27.64s | 文字正确，版式通过 |
| 2 | 东方山水艺术展 | 山河入画 / 东方美学艺术特展 / 湖北省博物馆 | 27.27s | 文字正确，版式通过 |
| 3 | 夏日音乐节 | 夏日回声 / LIVE MUSIC FESTIVAL / 2026.08.22·武汉 | 27.98s | 文字正确，版式通过 |
| 4 | 桂花拿铁 | 秋日桂香 / 桂花拿铁·限时回归 / ¥28 | 28.59s | 文字正确，版式通过 |
| 5 | 川西旅行 | 向远方出发 / 探索世界的另一面 / 川西·7日深度旅行 | 30.53s | 文字正确，版式通过 |
| 6 | 运动鞋发布 | 突破极限 / BEYOND LIMITS / 全新战靴发布 | 26.66s | 文字正确，版式通过 |
| 7 | 世界环境日 | 让地球呼吸 / 每一次选择，都在改变未来 / 世界环境日 | 28.09s | 文字正确，版式通过 |
| 8 | 黑色电影展 | 光影之间 / 城市青年电影展 / 日期 | 27.25s | 文字正确，但标题和副标题重复 |

平均耗时：

```text
(27.64 + 27.27 + 27.98 + 28.59 + 30.53 + 26.66 + 28.09 + 27.25) / 8
= 28.10 秒/张
```

严格版式遵循率为 7/8；指定字符串渲染正确率为 8/8。第 8 张说明 PE 对详细海报不是纯收益：它可能为了“丰富设计”自行复制文字区。该样本随后做了 PE-off 单独重试，但因为提示词和对照记录不是完整同条件矩阵，不把它用于量化 PE 胜负。

产物目录：

```text
/nfs-output/ernie-image-turbo-test/visual-posters/
```

---

## 7. Turbo 负向提示词验证

### 7.1 实验条件

```text
size: 768×1376
steps: 8
guidance_scale: 1.0
seed: 31001
PE: off
```

使用同一个正向提示词生成两次：一次不传 `negative_prompt`，一次传入包含 logo、二维码、电话、水印、额外文字、畸形人体等内容的长负向提示词。

### 7.2 结果

| 请求 | 耗时 | 文件大小 | SHA-256 |
|---|---:|---:|---|
| 无负向词 | 7.21s | 3,172,779 bytes | `7f6e8faa45fa8a2b9ab0ec59737ad2e046db8f8a8dec9018945583fe42e03e29` |
| 有负向词 | 7.05s | 3,172,779 bytes | `7f6e8faa45fa8a2b9ab0ec59737ad2e046db8f8a8dec9018945583fe42e03e29` |

两张 PNG 逐字节一致。这不是“负向词作用较弱”，而是**在当前 Turbo 推荐参数下完全没有进入生成路径**。

原因是 pipeline 只有在 `guidance_scale > 1` 且模型未标记 distilled 时才做 classifier-free guidance。Turbo 推荐 `guidance_scale=1.0`，因此 negative embedding 根本不会参与去噪。

又做了一组 512×512、2 步的快速定位测试：

| guidance | 负向词 | 预热后耗时 | SHA-256 |
|---:|---|---:|---|
| 2.0 | red/crimson/scarlet | 3.480s | `5591e8d31269a2630db053658d6576d2a412bb1d1aee906a8666ac779120aaa0` |
| 2.0 | green/emerald/lime | 3.438s | `5591e8d31269a2630db053658d6576d2a412bb1d1aee906a8666ac779120aaa0` |
| 1.0 | red/crimson/scarlet | 3.172s | `94e94109422cca256cca6d911c7b5bdf635ee7f5c9a16394c753a4d353ada367` |

guidance 1.0/2.0 hash 不同，证明 2.0 确实开启了 CFG；但 guidance 2.0 下两组相反颜色的负向词 hash 仍一致，证明负向词没有从请求 prompt 字典进入 pipeline 局部变量。

不要为了让负向词“生效”而提高 guidance：这偏离 Turbo 官方推荐设置，还会增加 CFG 计算。应把限制改写成正向构图描述，并在 Turbo 体验区隐藏或标注负向词不生效。

---

## 8. “八一建军节”参考海报优化过程

### 8.1 目标拆解

参考图的核心视觉层级：

1. 深青蓝夜空与金蓝双主色；
2. 顶部小型金色五角星，明暗相间并向外放射光芒；
3. 主标题「八一建军节」；
4. 副标题「数字算力 护航强军」；
5. 三个标签「智能防御」「数据协同」「科技强国」；
6. 中部蓝黄相间五角星与月桂；
7. 下部发蓝光的芯片与城市电路；
8. 两名只见暗色背影的士兵，由芯片蓝光勾边；
9. 小型飞行器喷出金色气流；
10. 不需要原图左上 logo/机构名，也不需要底部电话、网址和二维码。

最终所有约束都改成“要画什么”，不再写“不要画什么”。

### 8.2 迭代记录

| 版本 | 主要改动 | 结果与问题 | 耗时 |
|---|---|---|---:|
| 初版 | 复现科技军旅构图，保留核心文案 | 出现与参考类似的主体，但存在不需要的边角信息倾向 | 约 7–8s |
| V2 | 强化士兵装备、五角星光芒、芯片蓝光、金色飞行器尾流 | 光效基本到位；出现 4 名正面/侧面士兵，飞行器偏战斗机；顶部星过大；左上出现空白占位框 | 7.93s |
| V3 | 明确“两名、背影、暗色、芯片蓝光照亮”；顶部星为中心星约 1/3 | 人数和背影视角基本正确；飞行器约 6 个；星形颜色正确；左上空白占位仍在 | 8.13s |
| V4 | 移除 negative prompt，也移除正向提示中的“不/不要/不得”；只陈述需要的五组文字和画面 | 左上空白 logo 占位消失；文字准确；无二维码、电话、网址和额外文字；士兵、无人机仍偏大 | 7.90s |
| V5 | 给士兵 18–20%、飞行器宽 8–10% 的数值比例 | 飞行器略缩小；士兵仍约 31–33%，说明绝对百分比遵循较弱 | 8.05s |
| V6 | 改用相对约束：士兵高度约芯片宽度一半、只在底部边缘；飞行器是远景图标 | 士兵降到约 27–29%，相对描述比百分比有效 | 8.00s |
| V7 | 过度压缩目标：士兵 10–12%、头部在画面 88% 高度、微型守卫；4 架小无人机 | 实际士兵约 22–24%；4 架无人机数量更稳定 | 8.03s |
| V8 | 继续压到 5–7%、头部在 93% 高度，强调最底部微型剪影 | 实际士兵约 19–21%，与参考图比例最接近；核心文字保持正确 | 8.00s |

V4 是关键转折：删掉负向提示和所有否定措辞后，左上角空白 logo 框反而消失。模型可能把“不要 logo”理解为“这里存在一个 logo 槽位，但内容为空”。

V5–V8 说明模型会主动放大人物和飞行器。若期望最终人物约占画高 20%，提示词需要把目标写得更小，例如 5–7%，并同时使用：

```text
相对尺寸 + 画面区域 + 视觉层级 + 遮挡关系
```

比单写“人物小一点”或“人物占 20%”更有效。

### 8.3 当前推荐提示词结构

```text
竖版 9:16 高级商业宣传海报，深青蓝夜空，电影级金蓝双色照明。

【严格文字】画面只出现以下五组可读文字：
1. 顶部中央主标题「八一建军节」
2. 主标题下方副标题「数字算力 护航强军」
3. 同一行三个小标签「智能防御」「数据协同」「科技强国」

【上部】一枚较小金色五角星，约为中部徽章星的三分之一，
高光金与暗金交替切面，向外放射细长金色光芒。

【中部】蓝色与金黄色相间的立体五角星，外圈月桂叶，
保持海报主视觉但为下方芯片留出空间。

【下部】大型未来芯片与电路城市，芯片向四周释放强烈蓝色体积光、
光环与粒子流。四架小型远景无人飞行器喷出细长金色气流。

【最底部前景】左右各一名微型士兵，只见穿战术装备和头盔的暗色背影，
站在画面最底边，人物高度按目标成图的 5%～7% 描述，
芯片蓝光从前方照亮轮廓，人物只是环境尺度参照，不抢主体。

整体中心对称、层级清晰、金蓝高对比、细节丰富、专业印刷海报质感。
```

这里不包含负向词，也不包含“无 logo、无二维码、无电话”等否定句。通过“只出现以下五组文字”与完整正向画面清单约束输出。

### 8.4 产物

```text
/nfs-output/ernie-image-turbo-test/august1-poster/
  august1_v2.png
  august1_v3.png
  august1_v4_positive_only.png
  august1_v5_scaled_subjects.png
  august1_v6_relative_scale.png
  august1_v7_miniature_subjects.png
  august1_v8_tiny_soldiers.png
  *_prompt.txt
```

PE-on V9 只完成了临时请求脚本准备，**尚未上传和执行**，因此本文不报告它的结果。

---

## 9. 海报提示词工程结论

### 9.1 文字约束

推荐：

```text
【严格文字】画面只出现以下 N 组可读文字：
1. 「……」
2. 「……」
```

每组文字同时写清位置、字号层级和关系。不要在画面描述中反复复述同一段文案，否则模型更容易生成重复文本区。

### 9.2 版式约束

有效描述顺序：

```text
画布比例 → 色彩与风格 → 上/中/下分区 → 各主体相对尺寸 → 光线关系 → 严格文字
```

海报提示词应更接近设计稿规格，而不是自然语言散文。

### 9.3 小主体约束

只写数值比例不够，应组合使用：

- “位于最底部 10% 区域”；
- “人物高度约为芯片宽度的某一比例”；
- “只作尺度参照，不是主体”；
- “被芯片和城市部分遮挡”；
- “头顶位置在画面高度 90% 以下”；
- “远景小图标/微型剪影”。

### 9.4 PE 策略

| 输入类型 | 建议 |
|---|---|
| 一句话主题、缺少构图 | PE on |
| 详细设计稿式 prompt | PE off |
| 严格中文文案、区域和数量 | PE off |
| 探索创意方向 | 同 seed 各跑一张 on/off 再选 |

---

## 10. GPUStack 与 New API 接入建议

当前 GPUStack 代码已包含内置 `vLLMOmni` 后端，版本 `1.0.0` 指向本轮已验证的 `reputationly/vllm-omni:arm64-a100-latest`，并使用 `/ready` 健康检查。本轮实测 `/health`、`/ready`、`/v1/models` 均返回 HTTP 200。

因此部署 ERNIE-Image-Turbo **不需要修改 vLLM-Omni 引擎，也不需要重出引擎镜像**。前提是生产 GPUStack server/worker 镜像已包含当前代码里的 `vLLMOmni` 内置后端；如果线上 GPUStack 版本更早，需升级的是 GPUStack 控制面，不是引擎镜像。

建议配置：

| 项 | 值 |
|---|---|
| Backend | vLLMOmni |
| Source | Local Path |
| Model path | `/nfs-models/wuhanjisuan894/models/ERNIE-Image-Turbo` |
| Image | 当前已验证的 `arm64-a100-latest` |
| GPU | 1×A100-40GB |
| CPU offload | **关闭**；1024×1536 上快约 3.1 倍，且连续 PE on/off 无泄漏 |
| Port | 由 GPUStack 注入 |
| Health timeout | 至少 150 秒，避免 109 秒启动过程被误杀 |

#### GPUStack页面逐字段填写

| 页面字段 | 最终值 |
|---|---|
| Name | `ernie-image-turbo` |
| Source | `Local Path` |
| Model Path | `/nfs-models/wuhanjisuan894/models/ERNIE-Image-Turbo` |
| Cluster | `a100-image-video` |
| Backend | `vLLMOmni` |
| Backend Version | 优先显式选择 `1.0.0`；若页面只有 `Auto`，会使用内置默认 `1.0.0` |
| Replicas | `3` |
| Scheduling Mode | `Manual` |
| GPU Selector（最终） | `dev-gpustack-a100-0018:cuda:1`、`0019:cuda:1`、`0020:cuda:1` |
| GPUs per Replica | `1` |
| Model Category | 必须选 `Image`，不要保留 `Auto` |
| Backend Parameter 1 | `--max-generated-image-size 1572864` |
| Backend Parameter 2 | `--allowed-local-media-path /nfs-output`（需要GPUStack异步图片/NFS输出时添加） |
| Environment Variables | 模型级不填；`GPUSTACK_EXTRA_MOUNTS=/nfs-output`、`GPUSTACK_MEDIA_ROOT=/nfs-output` 配在worker侧 |
| LoRA Adapters | 不填 |
| Allow Distributed Inference Across Workers | 关闭 |
| Auto-Restart On Error | 开启 |
| Enable Model Route | 开启 |
| Enable Generic Proxy | 关闭 |

现网复查确认 AudioX 已迁移到 `0010:cuda:3`、`0014:cuda:3`、`0023:cuda:3`，三个实例均为 `RUNNING`；`0018/0019/0020:cuda:1` 当前无实例占用。ERNIE 可以直接按3副本选择这三张GPU，不需要使用0026临时验收位。

管理 API 可按下面的结构创建（接口前缀是 `/v2`，不是推理用的 `/v1`）：

```json
{
  "name": "ernie-image-turbo",
  "source": "local_path",
  "local_path": "/nfs-models/wuhanjisuan894/models/ERNIE-Image-Turbo",
  "backend": "vLLMOmni",
  "backend_version": "1.0.0",
  "categories": ["image"],
  "replicas": 1,
  "gpu_selector": {"gpus_per_replica": 1},
  "backend_parameters": ["--max-generated-image-size 1572864"],
  "enable_model_route": true
}
```

`categories:["image"]` 必须显式传。当前 GPUStack 对未填类别的 `vLLMOmni` 模型会默认标成 `text_to_speech`，会影响 UI、模型路由和用量统计。不要手工传 `--host`、`--port`、`--served-model-name`，这些由 GPUStack 接管。无 offload 配置必须显式设置 `--max-generated-image-size 1572864`；否则默认的超大面积上限允许请求绕过已验证边界，可能造成 OOM。

GPUStack 的 OpenAI 直连路由在修改 `model` 后会重新序列化完整原始 JSON，因此嵌套的 `extra_params.apply_pe`、`num_inference_steps`、`guidance_scale`、`seed` 可以保留。同时 GPUStack 已有 `/v1/videos` 异步图片门面，会调用引擎的 `/v1/tasks/image/`、轮询状态并从 NFS 取回结果。引擎的异步图片端点已真机返回 `completed` 并成功写入 PNG。

使用异步门面时，worker 进程还需由管理员设置：

```text
GPUSTACK_EXTRA_MOUNTS=/nfs-output
GPUSTACK_MEDIA_ROOT=/nfs-output
```

前者把输出 NFS 挂载进推理容器，后者限定本地媒体白名单边界。

### 10.1 PE 的对外 API 设计

产品/API 层增加：

```json
{
  "use_prompt_enhancer": false
}
```

并映射到：

```json
{
  "extra_params": {
    "apply_pe": false
  }
}
```

建议对外只暴露易理解的 `use_prompt_enhancer` 布尔字段，由 New API 白名单映射成引擎原生字段；不要把任意 `extra_params` 整包放给外部用户。布尔字段要保留“显式 false”语义，不能用普通零值判断把 false 当成未提供。

### 10.2 New API 接入实施记录

2026-08-01 已在 New API 完成以下改动：

1. GPUStackPlus `ModelList` 增加全小写模型名 `ernie-image-turbo`；
2. 体验区将原开关文案统一为“提示词智能优化”，默认关闭；
3. 用户提示改为非技术表述：简短主题/创意建议开启，详细文案或严格版式建议关闭；
4. 体验区显式下发 `use_prompt_enhancer: true/false`，关闭值不会被丢失；
5. GPUStackPlus 仅对 ERNIE-Image-Turbo 将该字段映射到 `extra_params.apply_pe`；
6. ERNIE-Image-Turbo 服务端固定为 8 步、guidance 1.0，不依赖 pipeline 的 50 步/4.0 默认值；
7. 保留 HunyuanImage-3.0 原有提示词思考/改写能力；
8. 新增 Go 测试，覆盖未传、显式 `false`、`true`、非 ERNIE 模型，以及“用户别名经渠道路由到全小写 `ernie-image-turbo`”的参数保留场景。

验证结果：

```text
go test ./relay/channel/gpustackplus   PASS
bun run build (web/classic)            PASS
git diff --check                       PASS
```

对外请求建议：

```json
{
  "model": "ernie-image-turbo",
  "prompt": "一张高级感中文科技主题海报……",
  "size": "768x1376",
  "seed": 31001,
  "num_inference_steps": 8,
  "guidance_scale": 1.0,
  "use_prompt_enhancer": false
}
```

这部分需要重新发布 New API 服务，但不需要重出 vLLM-Omni 引擎镜像。发布后还需在真实 GPUStack 模型路由上各发一次开/关请求，验收最终输出 hash 不同且请求参数没有被中间层丢弃。

---

## 11. 当前不支持的能力

### 11.1 图生图/海报改稿

当前 ERNIE-Image-Turbo checkpoint 是文生图模型：

- vLLM-Omni 明确 `support_image_input=False`；
- LightX2V 当前 ERNIE runner 也是 T2I；
- 不能把参考海报原图上传后要求“删二维码、换标题、缩小人物”并保持其余像素不变。

如需改稿线，建议使用 Qwen-Image-Edit-2511；ERNIE-Image-Turbo 负责首次创作，Qwen Edit 负责基于图片的文案、色彩和局部元素修改。

### 11.2 尚未完成的验证

- GPUStack 真正创建托管实例后的 OpenAI 直路由 E2E；
- New API GPUStackPlus 增加 PE/步数/guidance 映射后的端到端验收；
- 100 张以上长跑和并发排队；
- vLLM-Omni 与官方 diffusers 同 prompt、同 seed parity；
- vLLM-Omni 与 LightX2V 同条件 parity；
- OCR 自动化 Word Accuracy / NED；
- CFG 路径 `negative_prompt` 取值修复与回归测试；
- `is_distilled` 与官方 diffusers 行为的 parity 决策；
- 八一详细提示词 PE on/off 的严格同条件 V9 对照。

---

## 12. 复现与检查命令

### 12.1 服务状态

```bash
ssh -p 43047 root@111.172.214.16
docker ps --filter name=ernie-omni-turbo-test
docker logs --tail 100 ernie-omni-turbo-test
curl -sS http://127.0.0.1:8091/health
```

### 12.2 查看生成产物

```bash
find /nfs-output/ernie-image-turbo-test -maxdepth 2 -type f -printf '%P|%s bytes\n' | sort
cat /nfs-output/ernie-image-turbo-test/visual-posters/summary.json
cat /nfs-output/ernie-image-turbo-test/august1-poster/summary.json
```

### 12.3 验证负向提示词是否产生差异

```bash
sha256sum \
  /nfs-output/ernie-image-turbo-test/august1-poster/no_negative.png \
  /nfs-output/ernie-image-turbo-test/august1-poster/with_negative.png
```

---

## 13. 下一阶段建议

按优先级排序：

1. 用 GPUStack 正式创建一个单卡托管实例，验证健康检查和 `extra_params` 透传；
2. 发布已完成的 New API ERNIE/“提示词智能优化”改动，完成真实路由 E2E；
3. 修复 CFG 路径的 `negative_prompt` 取值并补测试，但不把它作为 Turbo 上线前置；
4. 把 V8 详细提示词做严格同 seed PE on/off 对照；
5. 建立 10～20 条固定中文海报 prompt 集，覆盖中英混排、小字号、多文本区、数量与相对位置；
6. 接 OCR 自动评分，至少记录 Word Accuracy、NED、重复文字区和额外文字数；
7. 在同一套 prompt 上对比 ERNIE-Image-Turbo、Qwen-Image-2512、Z-Image；
8. 最后补 LightX2V parity，再决定是否把生产引擎迁到 LightX2V。

当前可以进入 GPUStack 托管验证阶段，但还不能把“单机测试通过”等同于“生产验收完成”。

---

## 14. 参考资料

- ERNIE-Image 官方模型页：<https://huggingface.co/baidu/ERNIE-Image>
- ERNIE-Image-Turbo 官方模型页（含推荐分辨率）：<https://huggingface.co/baidu/ERNIE-Image-Turbo>
- ERNIE-Image 百度官方发布：<https://ernie.baidu.com/blog/zh/posts/ernie-image/>
- ERNIE-Image-Turbo ModelScope 权重：`PaddlePaddle/ERNIE-Image-Turbo`
- vLLM-Omni ERNIE pipeline：`vllm_omni/diffusion/models/ernie_image/`
- vLLM-Omni ERNIE tests：`tests/diffusion/models/ernie_image/`、`tests/e2e/*/test_ernie_image_expansion.py`
- 本轮真机产物：`/nfs-output/ernie-image-turbo-test/`

---

*本文只记录已经实际执行或通过当前代码核验的结论。引擎异步图片任务已真机跑通；未运行的 V9、GPUStack 托管实例 E2E、New API 改造后 E2E 和跨引擎 parity 均明确列为待办。*
