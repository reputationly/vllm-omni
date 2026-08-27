# SenseNova-U1.5-8B-MoT 接入调研与路线选型报告

> 调研日期:2026-08-24
> 目标:评估 SenseNova-U1.5-8B-MoT(统一文生图/图生图/图片理解/文本对话模型)接入现网 GPUStack + new-api 栈、40GB A100 的可行路线
> 结论:**主路线走 vllm-omni**——已有的 `SenseNovaU1Pipeline` 架构上已覆盖 U1.5 用到的 pixel-head 结构,只差真机验证;LightX2V 路线(哪怕同步到最新上游)在服务端和 GPUStack 侧都有真实缺口要从零补
> 状态:纯调研,未改动任何代码,权重已下载完整(NFS `/nfs-models/wuhanjisuan894/models/SenseNova-U1.5-8B-MoT`,33GB,19 文件清单核对通过)

---

## 0. 结论摘要(TL;DR)

| 问题 | 结论 |
| --- | --- |
| 权重下载好了吗 | **是**。ModelScope `SenseNova/SenseNova-U1.5-8B-MoT`,19 个文件、33GB,字节数逐文件核对通过。下载脚本:`scripts/download_sensenova_u1_5_8b_mot.sh` |
| 模型架构是什么 | `NEOChatModel`(自定义架构,需 `--trust-remote-code`),Qwen3-8B LLM 骨干身兼文本编码器 + 去噪主干(无独立 VAE/文本编码器),外挂 flow-matching 图像生成头(`use_pixel_head=true`,卷积像素头 `ConvDecoder`,替代老版逐 token MLP 头) |
| 官方推理框架是什么 | **LightLLM(理解/文本流)+ LightX2V(图像生成)** 分离式架构,两引擎经共享内存交换状态,不是原生跑在 vLLM 或 SGLang 上 |
| vllm-omni 有没有适配过 | **有,而且很成熟**。`SenseNovaU1Pipeline` 15 个 commit 迭代(TP、LoRA、FP8、CFG-Parallel、TeaCache、CacheDiT),官方文档站(docs.vllm.ai)有专门 recipe,含 H200/2×H200-TP2/MI300X 三套真机实测数据。但目前只针对 U1(原版),从未跑过 U1.5 checkpoint |
| U1.5 用到的架构 vllm-omni 支不支持 | **代码层面支持,未经真机验证**。`ConvDecoder`/`PixelShuffle`/`use_pixel_head` 从 U1 支持的第一个 commit(`c9a8556c`)起就已经写好,不是这次才补的 |
| 40G A100 能不能跑(不谈量化) | **单卡不行,TP=2 大概率可以**。H200 单卡实测峰值显存 35.9GB(1536×2720/50步/think),40G 卡余量不到 4GB 太紧;H200 TP=2 实测 18.2GB/卡,装 A100 绰绰有余。TP=4 没测过但架构干净(32 attn head / 8 KV head 整除 4) |
| 量化能不能救场 | **vllm-omni 能,LightX2V 现成方案不能**。vllm-omni 的 `sensenova_u1_transformer.py` 全线用真实 vLLM TP-aware linear 类,`quant_config` 已经透传到每一层,跟 H3 用的 `DiffusionInt8Config` 是同一套基类,是"接线"工作量。LightX2V 现成的 `neopp_dense_fp8.json` 是 FP8,而 A100(Ampere,算力 8.0)**没有原生 FP8 Tensor Core**,这份配置在我们硬件上基本用不上 |
| LightX2V 是不是已经接入 GPUStack 了 | **是,但接的不是 neopp 能用的那种**。现有集成靠镜像内置的 `gpustack-lx2v-launcher`,硬编码只支持两个实测标定档位(`z-image/bf16`、`wan2.2-i2v/int8-4card`),`neopp` 在 gpustack 仓里零引用 |
| LightX2V 服务端支不支持 neopp 的核心能力 | **不支持**。neopp 靠 `load_kvcache`/`set_inference_params` 跨轮维持状态才叫"统一模型",这套东西在 `lightx2v/server/` 里零引用,只在裸 Python API 里跑通过,没接进 FastAPI 服务层 |
| 同步 LightX2V 到最新上游能解决吗 | **不能**。查过上游 `lightx2v/server/api/tasks/` 目录,只有 `common.py`/`image.py`/`sensenova_vision.py`/`video.py`,**没有 `neopp.py`**。上游确实新建了"单常驻多任务 server"架构(PR #1323),但那套是给 Bagel 和 SenseNova-Vision 建的,不是给 neopp,同步上游拿不到这块 |
| 最终建议 | **在 vllm-omni 接**。理由不是"之前定过路线",是代码证据本身:vllm-omni 已有单 pipeline 覆盖全部四种模态(t2i/i2i/i2t/t2t)、OpenAI 兼容 API、直接插现有 GPUStack+new-api 栈;LightX2V 路线要同时补 GPUStack 新 profile + LightX2V 服务端状态管理,两边都是从零写,工作量明显更大 |

---

## 1. 模型信息

### 1.1 权重

| 项 | 值 |
| --- | --- |
| 仓库 | ModelScope `SenseNova/SenseNova-U1.5-8B-MoT`(HF 同名,ModelScope 无需登录直接下) |
| 大小 | 19 个文件,35,071,157,076 字节(32.66 GiB) |
| 结构 | 标准 8 分片 safetensors(`model-00001~00008-of-00008`)+ tokenizer 全套,无独立 VAE/文本编码器子目录 |
| 官方发布节奏 | 2026-06-29 Infographic-V2 → 2026-07-16 Infographic-V3 → **2026-07-31 U1.5-Preview**(原生 4K、局部纹理、复杂排版)→ **2026-08-20 U1.5 正式版**(文字渲染、原生 4K 效率、材质光照全面提升) |
| NFS 落地路径 | `/nfs-models/wuhanjisuan894/models/SenseNova-U1.5-8B-MoT`,下载脚本 `scripts/download_sensenova_u1_5_8b_mot.sh`(19 文件全量 SHA-256 清单,ModelScope 主源 + hf-mirror 回退) |

### 1.2 架构

```json
{
  "architectures": ["NEOChatModel"],
  "auto_map": {
    "AutoConfig": "configuration_neo_chat.NEOChatConfig",
    "AutoModel": "modeling_neo_chat.NEOChatModel",
    "AutoModelForCausalLM": "modeling_neo_chat.NEOChatModel"
  },
  "model_type": "neo_chat",
  "llm_config": { "architectures": ["Qwen3ForCausalLM"], "pure_llm": false, "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8, ... },
  "use_pixel_head": true,
  "template": "neo1_0"
}
```

- **不是标准架构**,需要 `--trust-remote-code` 才能用仓内自带的 `modeling_neo_chat.py` 加载。
- **LLM 骨干是 Qwen3-8B**(`pure_llm=false`,说明不是纯文本模型),身兼文本编码器(靠 KV cache)和去噪主干(靠 MoT/Mixture-of-Tokenizers 分支),**无独立 VAE、无独立文本编码器**。
- **`use_pixel_head=true`**:图像生成头是卷积像素头(PixelShuffle + Conv2d,直接把 hidden states 解码成 RGB),不是老版逐 token MLP 头——这是 U1.5 相对 U1 的关键架构差异点。
- 官方开源仓:[github.com/OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)(2.9k star,272 fork)。

---

## 2. 官方推理框架

SenseNova-U1 系列的官方生产推理是**分离式双引擎架构**:

- **LightLLM**:负责理解、文本流式输出、控制流(多模态 prefill、自回归解码、批处理调度)。
- **LightX2V**:负责图像生成(迭代式像素空间去噪,并行策略和显存访问模式与理解路径完全不同)。

两者通过 pinned shared memory + 高性能 transfer kernel 交换生成状态,官方论文里给出的理由是:理解路径和生成路径的调度策略、并行策略、资源预算差异太大,塞进同一个单体运行时会不必要地耦合两边。**这不是 vLLM/SGLang 原生跑法**,SGLang 也没找到原生支持的证据。

---

## 3. vllm-omni 现状

### 3.1 已有支持的成熟度

`vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py`(1463 行)是**这次上游同步带进来的、已经很成熟的实现**,15 个 commit 迭代:

| commit | 内容 |
| --- | --- |
| `c9a8556c` | 新增 SenseNova-U1 支持(含 `ConvDecoder`/`use_pixel_head` 分支,从一开始就在) |
| `f4115bd7` | 修复 SupportsModuleOffload 导致的 import 问题 |
| `3753b21c` | LoRA 支持 |
| `b63b36de`→`963ba1ab` | gen-only FP8 量化(加了又因故 revert) |
| `272e6261` | CacheDiT 支持 |
| `ac1d1d74`/`dd3a4d0a` | CFG-Parallel(含 8B-MoT 专项) |
| `4028680b` | Fused RMSNorm + 3D RoPE 融合 kernel |
| `9a105bf9` | TeaCache 支持 |
| `b4b907e7` | 最新一次:修复 + 用规范化的 model config |

### 3.2 pipeline 分发机制

`vllm_omni/diffusion/registry.py:269-272` 已注册 `SenseNovaU1Pipeline`。**`OMNI_PIPELINES`(`vllm_omni/config/pipeline_registry.py`)里没有它,这是设计使然,不是遗漏**——该文件头部注释明确写着:"Single-stage diffusion models continue to use the `_create_default_diffusion_stage_cfg` fallback... for now we do not add them to registry"。SenseNova-U1 是无独立 VAE/文本编码器的单阶段统一模型,天然走这条 fallback 路径,`vllm serve <model> --omni` 一条命令直接起,**不需要 deploy-config**。

### 3.3 官方 recipe 实测数据

`recipes/SenseNova/SenseNova-U1.md` 有完整真机数据(均针对**原版 U1-8B-MoT**,未测过 U1.5):

| 硬件 | 显存 | 延迟 | 备注 |
| --- | --- | --- | --- |
| 1×H200(144GB) | 峰值 35.9GB 保留 / 35.1GB 已分配 | 32.1s(1536×2720,50步,think模式,CFG 4.0) | 模型加载 32.8GiB / 8.7s |
| 2×H200 TP=2 | **18.2GB/卡** 保留 / 17.9GB 已分配 | 28.3s(比 TP=1 快 ~12%) | "limited by serial CFG dual-forward and communication overhead"——CFG 两路目前串行,没吃到并行加速 |
| 1×AMD MI300X(192GB) | 35.67GB 保留 | 34.0s | ROCm 路径也验证过 |

启动/调用方式:

```bash
# 起服务(四种模态共用一个端点)
vllm serve SenseNova/SenseNova-U1-8B-MoT --omni --port 8091

# 文生图 / 图生图 / 图片理解 / 纯文本对话,都走 OpenAI 兼容 /v1/chat/completions
python examples/online_serving/sensenova_u1/openai_chat_client.py --prompt "..." --modality text2img
```

### 3.4 量化基础设施

`sensenova_u1_transformer.py` 全线用真实 vLLM 的 `ColumnParallelLinear`/`RowParallelLinear`/`QKVParallelLinear`,**每一层都已经接收并透传 `quant_config` 参数**(多处调用点),架构上和 H3 编码器用的 `DiffusionInt8Config`/`Int8WeightOnlyLinearMethod` 是**同一套基类**。但 `pipeline_sensenova_u1.py` 里**零处**真正赋值非空的 `quant_config`——是"插座焊好了没插电源",接 Int8 量化预计是复用现成方案的小工作量,不是重新设计。

---

## 4. LightX2V 现状

### 4.1 我们本地 fork 的支持

`/Users/reputationly/Desktop/code/api/LightX2V`(`reputationly/LightX2V`)内部把 SenseNova-U1 叫 **`neopp`**(取自 NEO-unify),已有独立的 runner/network/scheduler:

- `lightx2v/models/runners/neopp/neopp_runner.py`
- `lightx2v/models/networks/neopp/`
- `lightx2v/models/schedulers/neopp/`
- `examples/neopp/`(dense/moe 两种变体,含 8-step 蒸馏、fp8、parallel-cfg-seq 等配置样例)
- `configs/neopp/`(`neopp_dense.json`、`neopp_dense_fp8.json`、`neopp_dense_parallel_cfg_seq.json` 等)

这些**全部是真实上游 PR**(#965 → #1153,9 个 PR),不是我们自己写的,验证方式是 `git log -S`:`ConvDecoder` 从最早的 #965 就已经存在。

### 4.2 量化

`configs/neopp/neopp_dense_fp8.json`:`"dit_quantized": true, "dit_quant_scheme": "fp8-sgl"`——**真实可跑的配置文件**,不是架构占位,`neopp_runner.py` 里还有量化与 LoRA 互斥的显式校验逻辑,说明生产上真被用过。

**但这是 FP8,A100(Ampere,算力 8.0)没有原生 FP8 Tensor Core 支持**(需要 Hopper/算力 8.9+),这份配置大概率是照着 H100/H200 调的,直接搬到 A100 上要么跑不动要么退化成软件模拟拿不到真实收益——**这条路径在我们硬件上基本走不通**,想在 LightX2V 上量化,还是得从头补 Int8/NF4。

### 4.3 并行

`configs/neopp/neopp_dense_parallel_cfg_seq.json` 有真实可跑的 `seq_parallel` + `cfg_parallel` 组合,`model.py` 里是完整实现(含 shard 切分、all-gather、CFG 双分支并行处理)。**这块比 vllm-omni 更完整**——vllm-omni 目前只接了 CFG-Parallel(`CFGParallelMixin`),没有独立的 sequence-parallel 开关。

### 4.4 我们 fork 与真实上游的差距

我们本地这份 fork 的 `neopp` 相关代码最新只到 `86e30aac`(2026-07-07,支持 conv pixel head/`use_pixel_head`——这正是 U1.5 用的架构,上游在 U1.5-Preview 发布前 3 周就已经在铺路)。上游后面还有两条没同步进来:

| commit | 日期 | 内容 |
| --- | --- | --- |
| `369f3a4c` | 2026-08-12 | 支持 png/webp/jpeg for neopp,商汤工程师(`luoweichao@sensetime.com`)共同署名 |
| `0714ee81` | 2026-08-21(U1.5 正式发布次日) | neopp/ascend:支持昇腾平台 dense 模型;重构 moe 模块和配置 |

商汤工程师直接在 LightX2V 上游提交代码,说明这条线确实有官方背书,但**即便追上这两条,也拿不到第 5 节要讲的服务端能力**。

---

## 5. A100-40G 可行性(不谈量化)

单卡跑不动(H200 实测 35.9GB,40G 卡余量不到 4GB,太紧,大概率 OOM 或不稳定)。多卡的话:

- **TP=2**:H200 实测 18.2GB/卡,装 A100-40G 绰绰有余,显存余量比单卡满打满算宽裕得多。TP 支持是真实 vLLM TP-aware linear 类,不是占位。
- **TP=4**:没有现成 benchmark,但架构干净——U1.5 config 里 `num_attention_heads=32`、`num_key_value_heads=8`,都能被 4 整除,没有奇数头数切不匀的问题,按比例推算会落在 9-12GB/卡。
- **顺手的优化空间**:recipe 原文明确说 TP=2 "limited by serial CFG dual-forward"——CFG 两路目前串行跑,没吃到并行加速。若是 4 卡机器,**TP=2 × CFG-Parallel=2 组合起来正好用满 4 卡**,既有 TP 的显存收益,又能把 CFG 真正并行掉,理论上比单纯拉到 TP=4 更快。

**未验证的坑**:上面全部数据都是原版 U1-8B-MoT 在 H200 上测的。U1.5 用的 `ConvDecoder`(卷积像素头)是个普通 `nn.Module`,没做 TP 切分,参数量相对 42 层 Qwen3 骨干应该很小、大概率不影响,但没有真机数据确认。真正结论需要拿 A100 实测这个具体 checkpoint 才能定。

---

## 6. GPUStack 集成现状核查

用户提出"LightX2V 已经接入 GPUStack 了",核查后发现**接入是真的,但覆盖不到 neopp**:

- GPUStack 不直接调用 LightX2V 的 CLI/API,而是 shell 出去调一个**镜像内置的二进制** `gpustack-lx2v-launcher`(`gpustack/worker/backends/lightx2v.py:40, 199-217`),只传 `--model/--host/--port`,launcher 自己读镜像里烤死的 profile YAML,再拉起 `python -m lightx2v.server`(或 `torchrun` 多卡版)。
- 现有集成**硬编码只支持两个实测标定档位**(`docs/lightx2v-backend-design.md` §3):`z-image/bf16`(图片)、`wan2.2-i2v/int8-4card`(视频),外加几个 wan 系不同卡数的兜底档位。**没有"传路径进去它自己认"的通用逻辑**,新模型要接就得新建 profile、重新烤镜像。`neopp`/SenseNova 在 gpustack 仓里**零引用**。
- 对外 API 是 `/v1/videos`、`/v1/images/generations`,包了一层 LightX2V 自己的**非 OpenAI 异步任务 API**(提交→轮询→取结果),是一次性单发模式。

**LightX2V 服务端本身**:`lightx2v/server/main.py` 支持显式传 `--model_cls neopp` 启动,`neopp_runner` 也声明支持 `t2i`/`i2i`,理论上基础单发文生图能通。**但 neopp 之所以叫"统一模型",核心是靠 `load_kvcache`/`set_inference_params` 跨轮维持状态**(`examples/neopp/neopp_dense_1k.py:24-87`)——这套东西在 `lightx2v/server/` 里**零引用**,只在裸 Python API 里跑通过,没接进 FastAPI 服务层,跟 GPUStack 现在这套"提交任务→轮询"的一次性抽象完全不是一个形状。

真要走这条路,要**同时**做两件事:①GPUStack 侧建新 profile、扩展 launcher;②LightX2V 服务端要**新写代码**把跨轮状态暴露出来。做完也只拿到 t2i/i2i,聊天和图片理解(LightLLM 管的那半边)GPUStack 完全没碰过。

---

## 7. 同步 LightX2V 到最新上游能否解决

**不能**。GPUStack 侧的 profile 缺口是我们自己仓库的代码,跟 LightX2V 上游无关,无论如何都要自己补。

LightX2V 服务端那部分,查过真实上游 `lightx2v/server/api/tasks/` 目录:只有 `common.py`/`image.py`/`sensenova_vision.py`/`video.py`,**没有 `neopp.py`**。上游确实在 2026-08-03 的 PR #1323("Lightx2v supports Bagel and SenseNova-Vision with server API")新建了一套"single resident multi-task server"架构(`omni_vision_task/subtask` 层级、专属 task 文件),**但这套是给 Bagel 和 SenseNova-Vision 建的**——`SenseNova-Vision` 是另一个模型(偏理解向),跟 `SenseNova-U1`(neopp)是两码事,PR 改动的 40 多个文件一处 neopp 相关都没碰。

上游证明了"给统一模型建带状态的多任务 server"这套架构可行、有先例,但这个工作量**还没人替 neopp 做**。同步上游能拿到 ConvDecoder 架构更新、png/webp 格式支持、昇腾适配,拿不到 neopp 的多轮 KV-cache 服务端支持。

---

## 8. 路线选型结论

**在 vllm-omni 接,不在 LightX2V 接。** 这个结论不依赖"之前定过路线"这条决议,纯看代码证据:

1. **vllm-omni 已有单 pipeline 覆盖全部四种模态**(t2i/i2i/i2t/t2t),OpenAI 兼容 API,`ConvDecoder`/`use_pixel_head` 架构从第一个 commit 起就写好了,U1.5 需要的东西大概率已经在,剩下的工作是"验证 + 小修",不是"从零开发"。直接插现有 GPUStack + new-api 栈,零新增服务端架构。
2. **LightX2V 路线要同时补两处从零的工作**:GPUStack 侧新 profile + 重新烤镜像;LightX2V 服务端要照着 Bagel/SenseNova-Vision 那套"单常驻多任务 server"架构模式,自己给 neopp 重写一遍状态管理——即便追上游最新代码也补不上这块,因为上游压根没写。而且做完也只拿到生成这一半,不含 chat/理解。
3. **量化维度也是 vllm-omni 占优**:vllm-omni 的量化基础设施(`quant_config` 全线透传)跟 H3 已验证过的 `DiffusionInt8Config` 同源,是接线工作量;LightX2V 现成的量化方案是 FP8,A100 没有原生硬件支持,用不上,一样得从头补 Int8/NF4。
4. **并行维度 LightX2V 更完整**(`seq_parallel`+`cfg_parallel` vs vllm-omni 只有 CFG-Parallel)——这点算 LightX2V 的优势,但不足以抵消上面三条的差距。

**LightX2V 该发挥的位置,是当"官方对照基线"**:U1.5 用到的 `ConvDecoder`/`use_pixel_head` 路径在 vllm-omni 里从没跑过真机、没跟官方数值对齐过,真要验收接入正确性,可以照 H3 用 `h3_diffusers_oracle` 做逐位对齐的思路,拿 LightX2V(或官方 transformers 实现)的输出当基准,而不是把它当成要长期维护的第二条 serving 路径。

---

## 9. 待验证事项(接入前必须做)

1. **真机 A100-40G 冒烟**:拿刚下载好的 U1.5 checkpoint 通过 vllm-omni 现有 `SenseNovaU1Pipeline` 实际跑一次(先 TP=2 起步),确认加载不报错、`use_pixel_head`/`ConvDecoder` 分支真的被触发且输出合理。
2. **U1.5 新增 config 字段覆盖度核查**:U1.5 的 config.json 比 vllm-omni `SenseNovaU1Config` 定义时机(U1 时代)多出几个字段(`P_mean`/`P_std`/`base_shift`/`max_shift`/`base_image_seq_len`/`max_image_seq_len`/`noise_scale_max_value`/`concat_time_token_num`/`extra_num_layers_post`/`time_shift_type`),需要逐个确认现有 pipeline 是否读取/正确处理,还是被静默忽略。
3. **显存实测**:TP=2/TP=4 在真实 A100-40G 上的峰值显存,与 H200 外推数据(TP=2 → 18.2GB)做对照。
4. **如需量化**:参照 H3 encoder.py 的 `DiffusionInt8Config` 接入模式,给 `sensenova_u1_transformer.py` 的线性层接上真实 `quant_config`,而不是复用 LightX2V 的 FP8 方案。
5. **精度对照基线**:用 LightX2V(neopp,追到最新上游)或官方 `examples/t2i/inference.py` 跑一份参考输出,和 vllm-omni 移植版做 SSIM/PSNR 或逐位对齐,验证移植正确性。
