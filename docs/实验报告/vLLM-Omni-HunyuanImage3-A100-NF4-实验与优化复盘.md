# vLLM-Omni · HunyuanImage-3.0 4×A100-40G NF4 实验与优化复盘

> 文档性质：深度技术调查、失败复盘、实现说明与后续调优手册  
> 最终结论：**HunyuanImage-3.0-Instruct-Distil NF4 已在单机 4×A100 PCIe 40GB 上完成三参考图 I2I 实测，不再是“判死”状态**  
> 测试日期：2026-07-26  
> 主测试机：0023，`dev-gpustack-a100-0023`，`10.0.0.178`  
> 代码分支：`feature/hunyuan-image3-a100-nf4`  
> 调试基线提交：`41aba8b588e8dc514ee7876a15d1d6a4474dd367`  
> 注意：本文记录的是实验结束时的工作树状态，相关 HunyuanImage3 NF4 改动当时尚未提交。

---

## 0. 结论先行

旧 LightX2V/Hugging Face 路径得出的“4×A100-40G 判死”结论，只对下面这一组实现约束成立：

```text
80B BF16 权重约 168GB，超过四卡总显存 160GB
  ↓
必须量化
  ↓
BitsAndBytes NF4 能装下，但权重是 uint8 packed blob
  ↓
FlashInfer CUTLASS fused MoE 不能直接按 BF16 权重形状读取
  ↓
回退 Hugging Face eager MoE
  ↓
旧版 one-hot dispatch 内存爆炸 + 64 专家 Python loop 很慢
```

它不是 A100 硬件的绝对不可行证明。最终打通的路径是：

```text
HunyuanImage-3.0-Instruct-Distil-NF4
  ↓
HunyuanImage3 专用 prequant BitsAndBytes loader
  ↓
TP4 + EP4：每个 rank 持有 16 个完整 packed 专家
  ↓
vLLM fused MoE + paged KV cache
  ↓
AR 与 DiT 两个引擎 level-1 sleep/wake，任一时刻仅一个驻留 GPU
```

最终实测：

| 项目 | 结果 |
|---|---:|
| AR 权重加载后 | 14.93 GiB/卡 |
| AR 三参考图完整 think+recaption | 653 tokens / 90.32 秒 |
| AR 速度 | 7.23 token/s |
| AR 峰值 | GPU0–3：30745 / 30769 / 30747 / 30773 MiB |
| DiT 权重加载 | 15.8324 GiB/卡 |
| DiT 加载完成后 | 16.52 GiB/卡 |
| 8-step Distil/MeanFlow 三图 I2I | 11.5969 秒 |
| 8-step 峰值 | 29339 MiB/卡 |
| 双引擎 sleep/wake 最坏峰值 | 35039 / 34619 / 34619 / 34619 MiB |
| 最坏 GPU 余量 | 约 5.7 GiB |
| 两引擎都 sleep 时主机可用内存 | 44.26 GiB |
| AR wake / sleep | 2.6658 / 0.8948 秒 |
| DiT wake / sleep | 1.7871 / 2.3726 秒 |

8-step 输出能正确保持人物、服装、摄影棚背景和肩部蓝色金刚鹦鹉，图像清晰，说明这不是“只加载成功”的假阳性。

---

## 1. 调查边界与硬件事实

### 1.1 硬件

| 项 | 配置 |
|---|---|
| CPU 架构 | ARM aarch64 |
| GPU | 4×NVIDIA A100-PCIE-40GB |
| Compute Capability | sm_80 |
| GPU 总物理显存 | 4×40960 MiB |
| GPU 互联 | PCIe，无 NVLink |
| 主机内存 | 约 251 GiB，可用预算按 256GB 节点考虑 |
| NVIDIA Driver | 570.86.10 |
| Linux Kernel | 5.15.0-78-generic |
| 节点 | 单机；跨节点明确排除 |

A100 sm_80 有 BF16、FP16、TF32、INT8 Tensor Core，但没有 Hopper FP8 Tensor Core，也没有 Blackwell NVFP4/FP4 Tensor Core。这里的 NF4 是 weight-only 存储格式，由软件反量化后执行 BF16/WNA16 计算，不应与 Blackwell 原生 NVFP4 混淆。

### 1.2 软件与镜像

测试镜像：

```text
crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/
  reputationly/vllm-omni:arm64-a100-latest

image id: 0e0aacfa336c
image size: 33GB
```

本地 vLLM-Omni：

```text
repo: /Users/reputationly/Desktop/code/api/vllm-omni
branch: feature/hunyuan-image3-a100-nf4
base HEAD: 41aba8b588e8dc514ee7876a15d1d6a4474dd367
```

模型：

```text
/nfs-models/wuhanjisuan894/models/
  HunyuanImage-3.0-Instruct-Distil-NF4-v2
```

关键配置：

```text
cfg_distilled = true
use_meanflow = true
hidden_size = 4096
num_hidden_layers = 32
num_attention_heads = 32
num_key_value_heads = 8
num_experts = 64
num_experts_per_tok = 8
moe_intermediate_size = 3072
max_position_embeddings = 22800
```

### 1.3 为什么不能跨节点

本调查没有尝试把四个 TP rank 分布到多台机器：

1. 目标环境没有高速跨节点互联。
2. Hugging Face `device_map` 本身不是跨节点张量并行实现。
3. AR 与 DiT 每层都需要频繁 collective，普通网络会把运行变成通信瓶颈。
4. 已经在一台四卡机器上证明容量足够，因此没有跨节点的必要。

---

## 2. 旧 Catch-22 的代码级根因

参考材料：

- `LightX2V/docs/HunyuanImage3-实验测试报告.md`
- `HunyuanImage-3.0/docs/hunyuanimage3-gpustack适配方案.md`
- `HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py`

### 2.1 BF16 容量硬下限

80B 参数按 BF16 粗算：

```text
80×10^9 × 2 bytes ≈ 160GB（十进制）
```

加 embedding、norm、buffer、CUDA context 和 allocator 后，实际报告约 168GB。四张 40GB 卡虽然标称 160GB，但不可能把全部显存都用作权重，因此 BF16 不成立。

### 2.2 旧 one-hot routing 为什么会在长多图上下文爆炸

旧 `topkgating()` 构造：

```python
expert_mask = F.one_hot(...)
dispatch_mask = F.one_hot(token_priority, expert_capacity)
combine_weights = einsum(router_probs, dispatch_mask)
```

记：

```text
T = token 数
E = 64 个专家
K = top-k = 8
C ≈ T×K/E = 每专家平均容量
```

平衡路由下：

```text
dispatch_mask 约为 [T,E,C] bool  = T×E×C bytes
combine_weights 约为 [T,E,C] fp32 = 4×T×E×C bytes

合计 ≈ 5×T×E×(T×K/E)
     ≈ 5×K×T²
     ≈ 40×T² bytes
```

在四参考/多参考图上下文约 `T=18,700` 时：

```text
40×18,700² ≈ 13.99×10^9 bytes ≈ 13.0 GiB
```

这还没计入：

- 路由不均匀导致 `expert_capacity > T×K/E`；
- expert input/output；
- einsum 临时张量；
- attention workspace；
- allocator reserved memory 和碎片。

因此旧报告观察到约 20.9GB 单次 activation spike 是可信的。

“每 1000 tokens 增加约 1GB”应理解为 18k 上下文附近二次曲线的局部斜率，不是全区间线性规律。按上式，在 18.7k 附近多 1000 token，routing 两个主张量的增量约 1.39GiB。

### 2.3 上游后来已经修改 eager MoE，但只解决了一半

HunyuanImage-3.0 当前主干的 `HunyuanMoE.forward()` 已经不再使用旧的 `[T,E,C]` dispatch，而是：

```python
topk_weights, topk_idx = self.gate(..., topk_impl="easy")
hidden_states_repeated = hidden_states_flat.repeat_interleave(K, dim=0)
expert_outputs = torch.zeros_like(hidden_states_repeated)
for i in range(self.num_experts):
    ...
```

主要复制 buffer 变为 `[T×K,H]`，随 token 数线性增长：

```text
18,700×8×4,096×2 bytes ≈ 1.14 GiB/个
hidden_states_repeated + expert_outputs ≈ 2.28 GiB
```

这确实移除了旧版最严重的二次 one-hot 峰值，但仍然有：

- 64 专家 Python loop；
- 每个专家布尔筛选和 scatter；
- bitsandbytes Linear4bit 的大量小 kernel；
- 每 diffusion step 重复上述流程。

因此“更新 Hunyuan 上游”有价值，但不足以达到生产速度。

### 2.4 为什么 FlashInfer FP4/NVFP4 不是 A100 解法

Blackwell 机器能同时得到“小权重 + 原生 fused FP4 MoE”，所以绕开 Catch-22。A100 sm_80 没有 NVFP4 Tensor Core，FlashInfer 对相关 FP4/FP8 kernel 的架构要求不会因为软件 patch 而消失。

可以在 A100 上做 INT4 weight-only/BF16 compute，但那是 GPTQ/AWQ W4A16 或 NF4 解码路径，不是 Blackwell NVFP4。

---

## 3. 为什么选择 vLLM-Omni，而不是继续修 LightX2V harness

HunyuanImage-3.0 不是传统“一个文本编码器 + 一个 DiT”的普通扩散模型：

1. 同一个 80B MoE backbone 先作为自回归多模态模型理解参考图、思考并 recaption。
2. 然后同一体系的 diffusion stage 使用生成的语义和图像条件完成 denoising。
3. Instruct-Distil 使用 `cfg_distilled + meanflow`，8 步且单次 forward，不需要传统 CFG 双 forward。

vLLM-Omni 已具备：

- 多阶段 AR→Diffusion orchestrator；
- HunyuanImage3 stage bridge；
- vLLM paged attention/KV；
- TP 与 expert parallel；
- diffusion fused MoE；
- level-1 sleep/wake；
- image-editing/multi-image 请求语义。

LightX2V/HF harness 适合保留为正确性基线，但继续修它意味着自己重新实现上述运行时能力。

生产代码应收敛在 vLLM-Omni。HunyuanImage-3.0 仓库只作为：

- 官方 prompt/tokenizer/预处理语义参考；
- 上游更新 diff 来源；
- 结果正确性回归基线；
- kernel/量化异常定位工具。

---

## 4. 实验时间线与失败复盘

### 4.1 上游检查

首先检查 HunyuanImage-3.0 上游近期改动。关键发现是当前 eager MoE 已改为 DeepSeek 风格 `easy_topk + repeat_interleave + expert loop`，所以旧报告里的 one-hot 爆炸不能原样当作当前代码事实。

保留结论：

- 旧报告的测量对当时 remote-code 有效；
- 当前 eager activation 已从近似二次变成线性；
- eager 64 专家循环的性能问题仍存在；
- FlashInfer 与 BnB packed weight 的形状/格式冲突仍不是通用解决方案。

### 4.2 2+2 拓扑：容量边界过窄， operationally unsafe

尝试把 AR/DiT 或 TP 拆成两卡时，DiT TP2+EP2：

```text
权重加载约 28.31 GiB/卡
warmup 峰值约 38.239 GiB/卡
40GB 卡只剩约 2.7GB 名义空间
```

0022/0024 随后出现 SSH banner 无响应，需要重启。检查上一启动周期没有找到明确 Linux OOM、Xid 或 soft lockup；日志有 NVRM NVLink shutdown 类警告，Mooncake 也会探测 HCA/NVLink。

因此这里不能严谨地声称“唯一根因就是 CUDA OOM”，但可以得到部署结论：

> 2+2 在 40GB 卡上没有足够运行安全余量，不应作为生产拓扑。

后续固定 TP4+EP4。

### 4.3 Stock vLLM prequant BnB + TP4 被硬拒绝

AR 阶段第一次直接使用 stock BitsAndBytes loader，报错：

```text
ValueError: Prequant BitsAndBytes models with tensor parallelism is not supported.
```

这个限制对通用 packed tensor 是合理的：uint8 blob 没有普通二维权重可安全切片，错误切片会破坏 nibble 和 QuantState 对齐。

但 Hunyuan TP4+EP4 有一个可利用的更强条件：

```text
64 experts / EP4 = 16 complete experts per rank
```

诊断确认每个 rank 的 `expert_map` 确实只拥有 16 个完整 global experts，不需要切割单个专家 packed tensor。于是设计专用 loader，而不是移除 stock loader 的全局保护。

### 4.4 初始统一 Omni 尝试：BitsAndBytesConfig schema 不兼容

容器 `hy3-omni-nf4-0023` 报错：

```text
TypeError: Cannot instantiate BitsAndBytesConfig with kwargs {
  '_load_in_4bit': True,
  '_load_in_8bit': False,
  ...
}
```

checkpoint 的 `quantization_config` 带 Transformers 序列化出来的内部字段 `_load_in_4bit/_load_in_8bit`，而 vLLM 当前 `BitsAndBytesConfig` 构造函数不接受它们。

经验：

- 不要把 checkpoint JSON 原封不动 `**kwargs` 给 runtime quant config；
- loader 应只解析公开字段；
- checkpoint metadata、运行时 quant method 和实际 loader format 是三层概念，不能混为一谈。

### 4.5 remote-code 文件缺失

容器 `hy3-omni-nf4-clean-0023` 报错：

```text
OSError: /model does not appear to have a file named
configuration_hunyuan_image_3.py
```

原因是 HunyuanImage3 checkpoint 配置声明 remote-code auto map，但实验挂载的 `/model` 不包含对应 Python 文件，或调用方没有把 `trust_remote_code=True` 传到所有配置/tokenizer入口。

修复面包括：

- `end2end.py` 显式传 `trust_remote_code=True`；
- tokenizer 加载显式信任 remote code；
- `GenerationConfig.from_pretrained(..., trust_remote_code=True)`；
- 确保部署 checkpoint 目录包含 config 中 auto-map 指向的文件。

生产检查不能只看 `config.json` 和 safetensors 是否存在。

### 4.6 worker 父进程只显示 EOFError

`hy3-omni-nf4-clean-v2-0023` 以及若干 patch 迭代的父进程最终只显示：

```text
multiprocessing.connection.py: reader.recv()
EOFError
RuntimeError: Orchestrator initialization failed
```

这不是根因，只表示 diffusion worker 在把初始化结果写回父进程前退出。

遇到这种错误应：

1. 不要只看 orchestrator 最后 100 行。
2. 向前查每个 worker rank 的第一条 traceback。
3. 检查是否被 SIGKILL/OOM killer 杀死。
4. 同时记录 `nvidia-smi`、`free -h`、`dmesg/journalctl`。
5. 如果父日志丢失 rank 异常，先拆成 AR-only/DiT-only 单阶段复现。

本次正是通过拆分阶段，分别打通 AR 和 DiT 后，再回到双引擎方案。

### 4.7 load_format 被 vLLM 自动检测覆盖

即使 YAML 写：

```yaml
load_format: hunyuan_image3_bitsandbytes
```

vLLM 0.25 在检测到模型 quantization method 为 BitsAndBytes 时，仍会在 `create_engine_config()` / `create_load_config()` 内把它改成通用 `"bitsandbytes"`。

结果是专用 loader 注册成功，但永远不会被选中。

修复位于 `vllm_omni/patch.py`：

- 记录调用方明确请求的 `hunyuan_image3_bitsandbytes`；
- 创建 load config 时临时绕过 generic BnB auto-overwrite；
- 模型的 quantization method 仍保持 BitsAndBytes；
- 限制只影响显式选择该 loader 的 Hunyuan 请求。

这里不应修改全局默认 BnB 行为。

### 4.8 gate/up packed 数据与 scale 必须同步换序

Hunyuan checkpoint 的逻辑排列是：

```text
gate_and_up_proj = [up, gate]
```

vLLM fused MoE `w13` 期望：

```text
w13 = [gate, up]
```

只交换 packed weight 的两半、不交换 `absmax` block scales，会得到“形状正确、能运行、数值错误”的危险结果。专用 loader 对 scale 做同样的 half swap。

已经对 gate/down 进行重建数值测试：

```text
max_diff = 0
```

这是量化 loader 必须保留的回归测试。

### 4.9 router gate 不应错误套用 NF4

checkpoint 的 `llm_int8_skip_modules` 包含相对路径 `"mlp.gate"`。vLLM 的通用 skip matcher 对绝对 prefix/路径分量的匹配方式不会把它自动匹配到 `layers.N.mlp.gate`。

如果不处理，BF16 router gate 会被错误配置为 BnB quant layer。

在 `HunYuanSparseMoeBlock` 中显式识别该 checkpoint contract，将 router gate 的 `quant_config` 设为 `None`；专家 MLP 仍走 NF4。

### 4.10 DiT 流式权重加载必须先收集 QuantState

Diffusion pipeline 直接迭代 safetensors，量化 metadata 不是普通 parameter。如果直接交给 model parameter loader：

- 它会被当成未知权重；
- 或只加载 packed bytes，却没有为 `w13_weight/w2_weight` 绑定完整 `bnb_quant_state`。

解决方法：

1. 在 pipeline 权重 iterator 外包一层过滤器。
2. 识别 `experts.N.{gate_and_up_proj,down_proj}.weight.*` metadata。
3. 依据 EP `expert_map` 只保留本 rank 的 16 个专家。
4. 其余普通权重继续流式交给原 loader。
5. 全层收集完成后重建并绑定 fused local QuantState。

收集位置必须覆盖整个 iterator 生命周期；如果放在递归或多线程加载的错误层级，容易只得到部分 layer 的 state。

---

## 5. 专用 NF4 TP+EP loader 设计

文件：

```text
vllm_omni/model_executor/model_loader/
  hunyuan_image3_bitsandbytes_loader.py
```

### 5.1 安全边界

loader 只允许：

- architecture 为 `HunyuanImage3ForCausalMM` 或
  `HunyuanImage3ForConditionalGeneration`；
- checkpoint `quant_method == bitsandbytes`；
- prequant 4-bit；
- TP>1 时必须启用 EP；
- `RoutedExperts.expert_map` 存在；
- local expert id 连续；
- 每个本地专家的 gate/up 和 down QuantState 完整。

它明确拒绝：

- 通用 BnB 模型；
- prequant INT8；
- TP>1 但没有 EP；
- 缺少 QuantState；
- 非连续或无法解释的 expert map。

这比直接删除 stock vLLM 的 TP 保护安全。

### 5.2 每 rank 加载过程

伪代码：

```python
verify_hunyuan_and_nf4()
initialize_generic_bnb_loader_state()

for checkpoint_tensor in safetensors:
    if tensor belongs to non-local expert:
        skip
    elif tensor is local expert packed weight:
        send to model.load_weights()
    elif tensor is local expert quant metadata:
        collect QuantState parts
    else:
        use normal TP mapping

for every RoutedExperts module:
    sort local experts by local_id
    reconstruct nested/double quant scales
    swap gate/up absmax halves
    concatenate 16 local expert QuantStates
    bind state to w13_weight and w2_weight
```

### 5.3 为什么 TP4+EP4 可以，普通 TP4 不可以

普通 tensor parallel 可能把单个矩阵按 output/input dimension 切成四份。对 NF4 packed byte，切点必须同时满足：

- nibble 对齐；
- blocksize 对齐；
- absmax 对齐；
- nested quant state 对齐；
- gate/up logical layout 对齐。

Hunyuan EP 避开了这个问题：每个专家是完整单元，只在专家集合维度分配。

### 5.4 当前 vLLM BnB fused MoE 的真实执行方式

当前路径不是“一个 Triton kernel 原位读取 BnB packed weight”：

1. `BitsAndBytesMoEMethod` 使用 bitsandbytes 反量化本 rank 的 local `w13/w2`。
2. 反量化结果交给 vLLM `fused_experts`。
3. 避免 Python 64 专家循环和巨型 one-hot dispatch。

它会产生 BF16 临时专家权重，但每卡只有 16 个专家，实测显存仍足够。真正 NF4 dequant-in-fused-MoE 可以作为后续优化，不是可行性的前置条件。

---

## 6. 正确拓扑：TP4+EP4

### 6.1 AR 推荐配置

实测 AR-only YAML：

```yaml
pipeline: hunyuan_image3_ar
async_chunk: false

stages:
  - stage_id: 0
    max_num_seqs: 1
    gpu_memory_utilization: 0.75
    trust_remote_code: true
    enforce_eager: true
    enable_prefix_caching: false
    max_num_batched_tokens: 32768
    devices: "0,1,2,3"
    tensor_parallel_size: 4
    enable_expert_parallel: true
    load_format: hunyuan_image3_bitsandbytes
    safetensors_load_strategy: lazy
    hf_overrides:
      rope_parameters:
        mrope_section: [0, 32, 32]
        rope_type: default
    default_sampling_params:
      temperature: 0.0
      top_p: 0.95
      top_k: 1024
      max_tokens: 1024
      detokenize: true
```

注意：

- `max_num_seqs=1` 是容量优先的基线，不代表最终吞吐最优。
- `gpu_memory_utilization=0.75` 仍得到 7.56GiB/rank 的 KV pool。
- `enforce_eager=true` 用于先建立正确性基线，后续可单独验证 graph/compile。

### 6.2 DiT 配置原则

DiT 同样使用：

```text
devices = 0,1,2,3
tensor_parallel_size = 4
enable_expert_parallel = true
load_format / quantization = HunyuanImage3 NF4 对应路径
```

模型是 Distil/MeanFlow 时：

```text
num_inference_steps = 8
cfg_distilled = true
use_meanflow = true
```

不要按传统 CFG 设计成每 step 两次 backbone forward。

### 6.3 PCIe collective

四卡 PCIe、无 NVLink 时，日志显示 vLLM 对 >2 PCIe GPU 禁用不适配的 custom allreduce，走 PYNCCL/NCCL。

这会比 NVLink 慢，但不是功能 blocker。实测 AR 7.23 tok/s、DiT 后续 step 约 0.5–0.7 秒，证明通信代价可接受。

不要强制打开为 NVLink 优化的路径，也不要因为日志里出现 “custom allreduce disabled” 就判断初始化失败。

---

## 7. 端到端实测

### 7.1 顺序销毁/重建基线

容器：

```text
hy3-seq-i2i-tp4-ep4-0023
exit code: 0
```

流程：

```text
启动 AR
→ 三参考图 think+recaption
→ 完全 shutdown AR，GPU 回到约 4MiB
→ 启动 DiT
→ 1-step I2I
→ shutdown
```

AR：

```text
653 tokens
90.3208 s
7.23 token/s
```

输出正确识别男人、女人、金刚鹦鹉，并生成：

```text
</recaption><answer><boi><img_size_1024><img_ratio_34>
```

GPU 峰值：

```text
30745 / 30769 / 30747 / 30773 MiB
```

KV pool：

```text
7.56 GiB/rank
247,808 token capacity
max_model_len=22,800 时理论并发 10.87×
```

DiT：

```text
load: 15.8324 GiB/rank
post-load: 16.52 GiB/rank
1-step request: 12.3166 s
denoise: ~4.03 s
peak: 29335 MiB/rank
```

顺序冷启动总耗时 294.67 秒，说明容量和正确性成立，但每请求重新加载两次不适合生产。

### 7.2 双长驻引擎 level-1 sleep/wake

容器：

```text
hy3-dual-sleep-tp4-ep4-0023
exit code: 0
```

初始化：

```text
AR init: 139.8275 s
AR initial sleep: 13.1526 s
AR sleep 后 GPU: ~1067 MiB/卡

DiT init: 73.8398 s
DiT initial sleep: 33.1969 s
```

两引擎都 sleep 时：

```text
host available: 44.26 GiB
free 显示 shared 一度约 172GiB
节点仍能响应
```

热切换：

```text
AR wake: 2.6658 s
三图 I2T probe: 31.73 s，内容正确
AR sleep: 0.8948 s

DiT wake: 1.7871 s
1-step I2I: 8.2400 s
DiT sleep: 2.3726 s
```

全程精确峰值：

```text
35039 / 34619 / 34619 / 34619 MiB
```

结论：

- 256GB host RAM 足以保存两个 level-1 sleeping engine；
- 40GB GPU 足以唤醒任一 TP4+EP4 stage；
- 必须通过互斥调度保证两个 stage 不同时 wake；
- level-1 是本方案的一部分，不能随意改成 level-2；当前 AsyncOmni 唤醒语义对 level-2 有限制。

### 7.3 正确 8-step Distil/MeanFlow

容器：

```text
hy3-dit8-tp4-ep4-0023
exit code: 0
```

结果：

```text
8-step denoise progress: ~7.5 s
第 1 step: ~4.04 s（包含 JIT/首次路径开销）
后续 step: ~0.5–0.7 s
完整热请求: 11.5969 s
峰值: 29339 MiB/卡
```

第一个 step 明显包含首次编译/初始化成本。连续第二次 8-step 请求大概率更快，但这仍是待做的重复测试，不能把推测写成实测数据。

输出文件：

```text
/private/tmp/hy3-distil-8step-i2i.png
```

人工检查：

- 人物主体清晰；
- 红色上衣、黑色裤子保留；
- 摄影棚帘幕背景正确；
- 蓝色金刚鹦鹉在肩部；
- 1-step 模糊符合预期，8-step 质量显著提升。

---

## 8. KV cache 重新核算

配置：

```text
L=32 layers
Hkv=8
head_dim=4096/32=128
K+V=2
BF16=2 bytes
```

全模型每 token：

```text
32×8×128×2×2 = 131072 bytes = 128KiB/token
```

TP4 后每 rank：

```text
32KiB/token
17,700 tokens ≈ 553MiB/rank
22,800 tokens ≈ 713MiB/rank
```

vLLM 实测 7.56GiB KV pool 可提供 247,808 tokens：

```text
7.56GiB / 247,808 ≈ 32KiB/token/rank
```

两者吻合。

因此旧资料中的“17.7k context 约 8GB KV”不能直接视为单请求纯 KV 数学量。更可能混入 HF 静态预留、图像 embedding、attention workspace、复制和 allocator reserved。若要严格归因，应在 HF baseline 中抓 CUDA allocator snapshot。

FP8 KV 即使省一半，TP4 下单请求只减少约 0.27GiB/rank，而且 A100 没有 FP8 Tensor Core。当前 paged BF16 KV 已足够，不应优先投入 FP8 KV。

---

## 9. 六条候选优化路线的最终评级

| 排名 | 路线 | sm_80+ARM | 结论 | 风险/投入 |
|---:|---|---|---|---|
| 1 | vLLM-Omni TP4+EP4 + NF4 + sleep/wake | 是，已验证 | 生产主线 | 中 |
| 2 | BnB NF4 + vLLM fused MoE | 是，已验证 | 已包含在主线 | 中 |
| 3 | MoE token chunking | 是 | 可救 HF 峰值，不能救 eager 慢速 | 中 |
| 4 | GPTQ/AWQ INT4 + Triton WNA16 | Maybe | Ampere kernel 可行，模型量化工具链缺口大 | 高 |
| 5 | CPU expert/block swap + chunking | Maybe | 逐层太慢；stage sleep/wake 更好 | 高 |
| 6 | 低精度 KV | Maybe | 节省很小，不是主要矛盾 | 中 |

### 9.1 MoE chunking

只切 `HunyuanMoE.forward()` 的 token 维度是语义安全的前提：

- routing 逐 token；
- `moe_drop_tokens=false`；
- MoE 内没有 token 间依赖。

按 512/1024 token tile 可把旧 `[T,E,C]` 峰值从 `O(T²)` 限制为 `O(tile²)`，或把当前 `[T×K,H]` buffer 按 `tile/T` 比例下降。

不能把整个 transformer 直接切成相互隔离的 token 块，因为 self-attention 是全局的。全 backbone chunking 需要 paged/block attention 或正确 KV 复用。

### 9.2 GPTQ/AWQ

A100 可以执行 INT4 weight-only/W4A16，vLLM 也有 MoE WNA16 Triton 路径。容量与 NF4 接近，理论上可行。

难点：

- HunyuanImage3 unified remote-code 没有现成量化 adapter；
- AR 和 diffusion 共用 backbone；
- calibration 必须覆盖文本、多图、recaption 和 timestep；
- checkpoint 到 vLLM `w13/w2` 需要专用 mapping；
- 不能只用普通 LLM perplexity 验证 diffusion 精度。

在 NF4 路线已成功后，GPTQ/AWQ 是性能研究项，不是救火项。

### 9.3 真正 NF4 dequant-in-Triton

若后续要继续优化，需要实现：

- NF4 codebook；
- packed nibble 读取；
- blocksize/absmax；
- nested/double quant；
- Hunyuan `[up,gate]` 到 vLLM `[gate,up]`；
- EP local expert mapping；
- 与 routed top-k、grouped GEMM 融合。

工作量和数值风险都高。必须先有当前 loader 的逐值回归作为 oracle。

### 9.4 CPU block swap

逐层/逐专家从 CPU 经 PCIe 搬运，会在 32 层×8 diffusion steps 中反复付费。当前 stage-level sleep/wake 只在 AR/DiT 边界搬一次，实测 1–3 秒级，是更好的交换粒度。

---

## 10. 代码改动地图

### 10.1 新增专用 loader

```text
vllm_omni/model_executor/model_loader/
  hunyuan_image3_bitsandbytes_loader.py
```

职责：

- 注册 `hunyuan_image3_bitsandbytes`；
- 限制模型/量化类型；
- 读取 EP local expert map；
- 修复 gate/up scale 顺序；
- 融合 local expert QuantState；
- 绑定 BnB state。

### 10.2 `vllm_omni/patch.py`

职责：

- 提前 import/注册专用 loader，保证 worker subprocess 可见；
- 保留显式 `load_format=hunyuan_image3_bitsandbytes`；
- 避免 vLLM BnB auto-detection 把它覆盖为 generic loader。

### 10.3 `hunyuan_image3_transformer.py`

职责：

- 识别 BnB expert state tensor；
- 只收集本 rank 16 个专家；
- 重建 double-quant QuantState；
- 修复 gate/up `absmax` 顺序；
- 为 `w13/w2` 绑定 state；
- honor `"mlp.gate"` skip contract。

### 10.4 `pipeline_hunyuan_image3.py`

职责：

- `GenerationConfig` 加 `trust_remote_code=True`；
- 在普通权重 iterator 前过滤/收集 BnB metadata；
- 整个权重加载完成后绑定 expert QuantState。

### 10.5 tokenizer 与 example

文件：

```text
vllm_omni/diffusion/models/hunyuan_image3/
  hunyuan_image3_tokenizer.py

examples/offline_inference/hunyuan_image3/
  end2end.py
```

职责：

- 完整传递 `trust_remote_code=True`；
- 避免 checkpoint auto-map 在某一个入口被遗漏。

---

## 11. 生产形态建议

### 11.1 双引擎状态机

```text
BOOT
  ├─ init AR → sleep(level=1)
  └─ init DiT → sleep(level=1)

REQUEST
  ├─ acquire global stage mutex
  ├─ wake AR
  ├─ run multi-image think/recaption
  ├─ sleep AR
  ├─ wake DiT
  ├─ run 8-step MeanFlow
  ├─ sleep DiT
  ├─ release mutex
  └─ return image
```

硬约束：

- 任意时刻最多一个 engine awake；
- wake 失败不能继续唤醒另一个 stage；
- sleep 后应验证显存回落；
- 请求取消也必须在 `finally` 中恢复 stage 状态。

### 11.2 热态延迟预算

按完整 653-token AR 实测：

```text
AR wake          2.67s
AR generation   90.32s
AR sleep         0.90s
DiT wake         1.79s
8-step DiT      11.60s
DiT sleep        2.37s
----------------------
总计约          109.65s
```

AR 是绝对主耗时。后续性能优化应优先：

1. 减少不必要的 AR 输出 token；
2. 扫 `max_num_batched_tokens` 和 fused MoE config；
3. 对相同多图前缀评估 prefix caching；
4. 做请求级阶段 batching；
5. 最后才考虑重写 NF4 kernel。

如果业务已有高质量 recaption，可提供 DiT-only 快速路径，但这会改变模型原生理解/规划语义，不能默认替代完整 I2I。

### 11.3 节点规划

| 节点 | 用途 |
|---|---|
| 0023 | 主验证/候选生产节点 |
| 0022 | 回归与备用 |
| 0024 | 回归、kernel tuning 或 GPTQ/AWQ 研究 |
| 0021 | 保留旧 HF baseline |
| 0025 | 不占用，留给其他业务 |

2+2 实验导致 0022/0024 一度无响应并已重启。最终状态确认：

```text
0022/0023/0024 均健康
每张 GPU 约 1MiB used
主机 available 约 238–240GiB
无运行中的 Hunyuan/vLLM 实验容器
```

---

## 12. 复现与回归清单

### 12.1 启动前

- [ ] `uname -m` 为 `aarch64`
- [ ] 四张 GPU 都是 A100-PCIE-40GB
- [ ] 每卡空闲显存接近 40960MiB
- [ ] 主机 available memory 至少 220GiB
- [ ] checkpoint 包含 remote-code 文件
- [ ] config 的 `cfg_distilled/use_meanflow` 符合目标变体
- [ ] TP=4，EP 已启用
- [ ] loader format 没被改回 generic `bitsandbytes`
- [ ] 0025 未被占用

### 12.2 loader 单元回归

- [ ] 每 rank 恰好 16 个 local experts
- [ ] local expert id 连续
- [ ] 每层 16 份 gate/up state 完整
- [ ] 每层 16 份 down state 完整
- [ ] double quant 已解开或正确重建
- [ ] gate/up packed halves 与 absmax halves 同步交换
- [ ] 随机 local expert dequant `max_diff=0`
- [ ] router gate 保持 BF16
- [ ] 非 Hunyuan 模型被专用 loader 拒绝
- [ ] TP>1、EP=false 被拒绝

### 12.3 AR 回归

- [ ] 三参考图都能被识别
- [ ] 生成 `</recaption><answer><boi>`
- [ ] image size/ratio token 正确
- [ ] 速度不低于本次 7.23 token/s 太多
- [ ] 峰值不超过 32GiB/卡
- [ ] KV pool/token bytes 与 TP4 数学吻合

### 12.4 DiT 回归

- [ ] 8 steps，不误用传统 CFG 双 forward
- [ ] 峰值不超过 31GiB/卡
- [ ] 参考图身份/主体/关系保持
- [ ] 输出不是全黑、纯噪声或 1-step 模糊
- [ ] 连续两次请求验证首次 JIT 与 steady-state 差异

### 12.5 sleep/wake 回归

- [ ] AR sleep 后 GPU 约 1GiB/卡量级
- [ ] DiT sleep 后显存回落
- [ ] 两引擎都 sleep 时 host available > 35GiB
- [ ] AR wake < 5s
- [ ] DiT wake < 5s
- [ ] 全流程峰值 < 36GiB/卡
- [ ] 异常/取消后不会留下两个 awake engine

### 12.6 soak test

至少执行 20–50 次完整三图 I2I，记录：

```text
request latency
AR tokens/tokens-per-second
每卡 peak/idle memory
host available/shared memory
sleep/wake latency
NCCL/PYNCCL warnings
worker restart
输出图 hash/人工质量抽查
```

当前“功能可行”已经证明；生产稳定性仍以 soak test 为准。

---

## 13. 后续优化优先级

### P0：先把成功路径固化

1. 提交专用 loader 和相关 Hunyuan patch。
2. 增加 loader 数值单测。
3. 把 AR/DiT 配置收敛成正式 deploy YAML。
4. 实现全局 stage mutex 和异常恢复。
5. 做连续请求 soak。

### P1：A100 fused MoE tuning

日志提示当前 shape 使用默认 MoE config。目标至少覆盖：

```text
local E = 16
N = 3072
K/input hidden = 4096
top-k = 8
device = NVIDIA A100-PCIE-40GB
```

应对不同 token bucket benchmark 后写 tuning JSON。不要从 H100/Blackwell 复制配置。

### P1：加载与 NFS

`safetensors_load_strategy: lazy` 已配置，但专用 loader 继承的 iterator 是否完整 honor lazy/NFS 访问模式需要单独 profile。四个 rank 同时预读完整 NFS shard 会造成启动抖动和 page cache 压力。

建议记录：

- 每 rank 读字节数；
- NFS throughput；
- cold cache 与 hot cache 启动时间；
- host page cache；
- 是否重复扫描非本地专家。

### P2：AR token/调度优化

AR 90 秒占端到端绝大部分：

- 验证是否所有任务都需要 `think+recaption`；
- 对简单编辑比较 `recaption` 与 `think_recaption`；
- 限制最大输出 token，但必须检查截断；
- 评估相同参考图前缀复用；
- 以质量回归为前提做小批阶段调度。

### P3：GPTQ/AWQ 或自定义 NF4 kernel

只有在以下条件满足后再启动：

- 当前 NF4 方案稳定；
- profile 证明 MoE 反量化/临时 BF16 是主要瓶颈；
- AR token 数已优化；
- PCIe collective 不是第一瓶颈。

否则重新量化 80B unified backbone 的投入产出比很低。

---

## 14. 常见误判速查

| 现象 | 不应直接判断为 | 正确排查 |
|---|---|---|
| 父进程 `EOFError` | orchestrator 自身 bug | 查 worker 第一条 traceback、OOM、SIGKILL |
| custom allreduce disabled | 四卡不能跑 | PCIe >2 GPU 正常回退 PYNCCL/NCCL |
| NF4 checkpoint 约 46GB | 每卡只占 11.5GB | 还要算 dense、runtime、反量化临时量 |
| 17.7k 上下文显存高 | 全是 KV cache | 按 32KiB/token/rank 核算并抓 allocator |
| 1-step 图像模糊 | 权重加载错误 | Distil 应以正确 8-step 质量判断 |
| loader 能启动 | QuantState 一定正确 | 必须做逐值 dequant 对比 |
| `load_format` 写进 YAML | 一定选到该 loader | 检查 vLLM auto-detection 是否覆盖 |
| A100 不支持 NVFP4 | 所有 INT4 都不能用 | W4A16/NF4 软件解码仍可在 sm_80 执行 |
| 两卡能加载到 38GB | 可以生产 | 无安全余量，warmup/碎片即可击穿 |

---

## 15. 最终决策

### 生产主线

```text
vLLM-Omni
+ HunyuanImage3 专用 NF4 TP+EP loader
+ TP4/EP4
+ AR/DiT level-1 sleep/wake
+ 8-step Distil/MeanFlow
```

### 不采用

- 4 卡 BF16；
- A100 上 FlashInfer NVFP4；
- 2+2 常驻；
- LightX2V/HF eager 作为生产主引擎；
- 每请求销毁并重建两个模型；
- 跨节点 TP；
- 未经数值验证的 packed tensor 切片。

### 保留

- HunyuanImage-3.0 HF harness：正确性/reference；
- LightX2V 旧报告：历史基线和失败证据；
- GPTQ/AWQ：后续性能研究；
- token chunking：HF fallback 的峰值控制手段。

最终结论：

> 4×A100 PCIe 40GB、ARM aarch64、256GB RAM、无 NVLink 的单节点，可以把 HunyuanImage-3.0-Instruct-Distil NF4 跑到可部署水平。真正突破点不是 FP4，也不是简单减 KV，而是利用 TP4+EP4 保持每个 NF4 专家完整、使用 vLLM fused MoE，并在 AR/DiT 阶段边界做整引擎 sleep/wake。
