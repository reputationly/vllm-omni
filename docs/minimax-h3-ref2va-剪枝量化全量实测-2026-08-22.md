# MiniMax-H3 Ref2VA 剪枝 / 量化 全量实测报告（2026-08-22）

> 测试对象：MiniMax-H3 Ref2VA 在 4×A100-40G 上的显存、耗时与画质。
> 覆盖 54 次引擎运行，全部用官方素材 + 官方提示词 + 固定种子。
> 本文自包含：结论、方法、原始数字、复现命令都在里面。

---

## 1. 一句话结论

**推荐配置：剪枝 r8 + DiT Int8 权重量化（W8A16）+ 编码器 Int8**
显存峰值从满血的 34529 MiB 降到 **25087 MiB（−27%）**，耗时 651s 对 624s（慢 4%），
两个官方 case 的肉眼比较均**回到基准水平**。

> **不要用 W8A8（激活也量化）。**它更省（23759 MiB，−31%）也更快，但会**削掉高动态瞬态
> 事件与运动细节**——实测 case1 中段动作丢失、case2 窗外的闪电退化成一次亮光。
> 同一份 Int8 权重，仅把 `activation_scheme` 从 `dynamic` 换成 `weight_only`，问题即消失。
> 详见 §8。

serve 根目录：`/nfs-models/wuhanjisuan894/models/MiniMax-H3-Ref2VA-Pruned-r8-INT8-W8A16-encINT8-vLLM`

**关于输入上限**：§5.3 的包络是在 W8A8 上测的。W8A16 权重体积相同、峰值高约 1.3 GiB，
9 图 + 3 视频 15 秒那档在 W8A8 下就只剩 0.51 GiB 余量，**换成 W8A16 后大概率不再可跑**，
该档需要按 W8A16 重测后才能对外开放（见 §10）。

---

## 2. ⚠️ 勘误：上一轮的剪枝数据全部作废

`MiniMax-H3-Pruned-r8/` 里两份剪枝权重的命名与用途**相反**，上游 README 写得很清楚：

> `workflow="t2va"` and `workflow="fl2va"` load `transformer/`; `workflow="ref2va"` loads `transformer_ref/`

上一轮的 ref2va 剪枝件是从 `transformer/`（即 **FL2VA**）派生的。逐位证据：

| 比对 | 结果 |
|---|---|
| 上一轮产物 vs 官方 **FL2VA** 原件（attn/MLP/patch 投影 4 项） | **4/4 逐位相同** |
| 上一轮产物 vs 官方 **Ref2VA** 原件 | **0/4 相同** |
| 上一轮产物 vs 上游 `transformer`（FL2VA） | 仅 **312** 个张量不同，且 312 个**全是 LoRA 目标**（= 融合审计的 `verified_target_tensors`） |
| 上一轮产物 vs 上游 `transformer_ref`（Ref2VA） | **637/637 全不同** |

即那批产物是「**FL2VA 剪枝权重 + ref2v Turbo LoRA**」。

**作废范围**：`docs/minimax-h3-剪枝前后对比-2026-08-20.md` 与
`docs/minimax-h3-INT8剪枝叠加-交接-2026-08-20.md` 中所有"剪枝 r8"相关的**画质与保真结论**。
**显存与吞吐结论仍成立**——两分支架构完全一致（层数、维度、shape 全同）。

错误产物已保留但在目录内放置 `DEPRECATED-WRONG-SOURCE.md`；对应视频在看片目录里标注为
`_权重误用FL2VA_作废`。

### 为什么权重错了还能出正常视频

FL2VA 与 Ref2VA 是同一基座的两个微调分支，主干几乎相同（float64 计算）：

| 张量 | 余弦相似度 | 相对 L2 差异 |
|---|---:|---:|
| `proj_in.weight` | 0.99977 | 2.2% |
| `context_embedder.weight` | 0.99971 | 2.4% |
| `audio_proj_in.weight` | 0.99975 | 2.3% |
| `blocks.0.attn.to_q.weight` | 0.99970 | 2.5% |
| `blocks.25.ff.net.0.proj.weight` | 0.99951 | 3.1% |
| `blocks.49.attn.to_out.0.weight` | 0.99991 | 1.3% |
| `time_embedder.table` | 0.99985 | 1.9% |

"会画画"的权重差异只有 1~5%，所以画面不会塌；真正分工不同的是参考条件通路，
表现为**参考主体不像、指令没跟住**，而不是画面崩坏。

**根因是评测缺项**：所有指标（峰值显存、耗时、NFE 校验）都不测"参考图有没有被保住"，
所以肉眼扫一眼"有画面、动得像回事"就放过了。

> 注：`adaln_basis` / `norm_out.linear` 等 AdaLN 通路张量在两份剪枝件之间余弦低至 0.03 甚至负值，
> 但**不可直接比较**——每份剪枝 checkpoint 的低秩基是各自 SVD 出来的，基不同则坐标矩阵必然不同。

---

## 3. 权重产物

全部从**正确的 `transformer_ref`** 派生，逐位自检通过。

| 目录（`/nfs-models/wuhanjisuan894/models/` 下） | 蒸馏 | DiT | 编码器 | DiT 磁盘 |
|---|---|---|---|---:|
| `MiniMax-H3-Ref2VA-Pruned-r8-BF16-vLLM` | 无 | 剪枝 BF16 | BF16 | 37.5 GiB |
| `MiniMax-H3-Ref2VA-Pruned-r8-INT8-vLLM` | 无 | 剪枝 INT8 | BF16 | 19.5 GiB |
| `MiniMax-H3-Ref2VA-Pruned-r8-Turbo4-BF16-v2-vLLM` | 4 步 | 剪枝 BF16 | BF16 | 37.5 GiB |
| `MiniMax-H3-Ref2VA-Pruned-r8-Turbo4-INT8-v2-vLLM` | 4 步 | 剪枝 INT8 | BF16 | 19.5 GiB |
| `MiniMax-H3-Ref2VA-Pruned-r8-Turbo4-INT8-v2-encINT8-vLLM` | 4 步 | 剪枝 INT8 **W8A8** | INT8 | 19.5 GiB |
| **`MiniMax-H3-Ref2VA-Pruned-r8-Turbo4-INT8-W8A16-encINT8-vLLM`** | **4 步** | **剪枝 INT8 W8A16** | **INT8** | **19.5 GiB** |
| **`MiniMax-H3-Ref2VA-Pruned-r8-INT8-W8A16-encINT8-vLLM`** | **无** | **剪枝 INT8 W8A16** | **INT8** | **19.5 GiB** |
| `MiniMax-H3-Ref2VA-Pruned-r8-BF16-encINT8-vLLM` | 无 | 剪枝 BF16 | INT8 | 37.5 GiB |
| `MiniMax-H3-Ref2VA-Pruned-r8-Turbo8-768p-INT8-W8A16-encINT8-vLLM` | 8 步 | 剪枝 INT8 W8A16 | INT8 | 19.5 GiB |
| `MiniMax-H3-Ref2VA-INT8-vLLM` | 无 | 满血 INT8 | BF16 | 43.8 GiB |
| `MiniMax-H3-Ref2VA-INT8-encINT8-vLLM` | 无 | 满血 INT8 | INT8 | 43.8 GiB |
| `MiniMax-H3-Ref2VA-encINT8-vLLM` | 无 | 满血 BF16 | INT8 | 61.7 GiB |
| `Qwen3-VL-32B-H3Encoder-INT8` | — | — | INT8 编码器本体 | 25.3 GiB |

**2026-09-05 新增最后一行（Turbo8）**：上游 09-03 才发布 ref2v 8-step v1.0 768p，本表原来
没有 8 步档。链路与 Turbo4 完全相同（`transformer_ref` → 烘 LoRA → 转 partition →
INT8 → W8A16 → 装配 + encINT8），剪枝起点已用 `norm_out.linear.weight` 逐位确认是
Ref2VA 分支而非 FL2VA（§2 那个坑）。融合验真 312 targets / max_abs_error 0.0。

⚠️ 两项未验：**shift 取 6/3 是推的**（Turbo4 用 12/3，但那份权重训练在 544p，
新权重训练在 768p，而 768p 家族惯例是 6/3——ref2va 这条链路没有 768p 先例）；
**W8A16 输入包络仍未重测**（§10 第一行，对 Turbo4 就悬着，对 Turbo8 同样）。

**推荐的是 W8A16 那一行**（§1、§8）。它与 W8A8 版**共用同一份 Int8 张量文件**——
`-W8A16-partition/transformer` 里的 safetensors 是指向 `-INT8-partition` 的符号链接，
只有 `config.json` 的 `activation_scheme` 不同。所以两者磁盘占用不叠加。
Turbo4 档的 W8A16 版当时尚未组装；**已于 2026-08-22 20:42 补做**，serve 根目录
`MiniMax-H3-Ref2VA-Pruned-r8-Turbo4-INT8-W8A16-encINT8-vLLM`（已加入下表）。
现网 GPUStack 上注册名为 `minimax-h3-fast-ref2va`，模型路径就是它。此句保留原文以说明当时状态。

> 注意目录名里没有 `-v2`，但它的 INT8 张量是软链到 `-Turbo4-INT8-v2-partition` 的——
> `-v2` 是血统标记不是路径的一部分，别照着拼一个带 `-v2` 的 W8A16 目录名，盘上没有。

### 3.1 DiT 量化

工具 `vllm_omni/quantization/tools/quantize_minimax_h3_int8.py`，per-output-channel 对称 + 动态激活（W8A8）。
**37.5 GiB → 19.5 GiB，100 秒。**
`ignored_layers` 校验：51 个 `adaln_proj.linear`（50 层 + final）保留 BF16、token_refiner 8 个投影保留 BF16、
200 个 `weight_scale`（50 层 × 4 投影）齐全。

### 3.2 格式转换与 bit-parity

剪枝件是 diffusers 格式，INT8 工具只认 partition 格式，中间加了
`tools/minimax_h3_turbo/convert_pruned_to_partition.py`（本轮新写）：
qkv 按 head 交织融合、fc1 上下半块对调、其余仅改名，**637 → 533 张量，104~201 秒**。

用 `tools/minimax_h3_parity/verify_checkpoint_conversion.py` 逐位验证：

```
"identical": 429, "qkv_orders": {"grouped_per_head": 52}, "mlp_orders": {"up_gate": 52},
"mismatched": [], "unmapped_diffusers": [], "unmapped_partition": [],
"verdict": "checkpoints agree"
```

### 3.3 Turbo4 LoRA 融合审计

`minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`（rank 128 / alpha 8 / 等效 scale 0.0625）：

```
verified_target_tensors: 312, verified_values: 8945664, changed_values: 573861,
max_abs_error: 0.0
```

### 3.4 编码器量化（本轮新增）

先验明血统：H3 的 `text_encoder` 是官方 `Qwen/Qwen3-VL-32B-Instruct` 的**逐字节拷贝**——
14 个分片字节数全部相同，shard14 的 SHA256 与 HF 上一致（`e45b6c99…b13b`）。

工具 `vllm_omni/quantization/tools/quantize_qwen3vl_encoder_int8.py`（本轮新写）：

- 只量化 50 层 × 7 个投影 = **350 个张量**，per-output-channel 对称，**权重-only（W8A16）**
- 视觉塔（1.1 GiB）、`embed_tokens`（1.4 GiB）、所有 norm/bias 保留 BF16
- 丢弃**永不加载**的 layers 50–63 与 `lm_head` = 156 个张量（编码器只建 `min(64, 50)` 层）
- **62.1 GiB → 25.3 GiB，84 秒**

引擎侧 `encoder.py` 新增 `_Int8WeightMixin`：构建后按 checkpoint 的 `quantization_config` 决定是否切 int8，
**不改任何构造函数签名**，BF16 路径逐字不变（单测断言 `dense_weight() is weight`，零拷贝零转换）。

---

## 4. 测试方法

| 项 | 值 |
|---|---|
| 素材 | 官方测试集 `examples/ref2va_testset`（34 张 jpg）+ 阶梯素材 `/nfs-output/h3_scale/` |
| 提示词 | 官方 `prompts_ref2va_test.json` 六段式，**逐字不改** |
| 分辨率 | 1344×768（官方 demo 出片参数，实测 `r2va_direct_768p.mp4` 就是 1344×768 / 24fps） |
| 时长 | case1 = 官方 5.1667s（=124 帧，官方契约默认）；case7 = 官方 10s（=240 帧） |
| 种子 | 42（全部） |
| 并行 | TP4 + `--text-encoder-tp-size 4` + VAE patch-parallel 4，`--usp 1 --ring 1` |
| offload | `--enable-cpu-offload`（除专门标注的对照） |
| 契约 | `VLLM_OMNI_H3_INFERENCE_CONTRACT=official_diffusers_v1` |
| 度量 | nvidia-smi 每 2 秒采样 + torch 分配器累计最大值（本轮新增日志） |
| 机器 | gpu41–gpu50，4×A100-40G / 251 GB 内存 / 128 核，一机一臂 |

**NFE 硬校验**：每次运行统计引擎日志里 `rows video=` 行数，÷4（rank 数）= 实际步数，全部与请求一致。

---

## 5. 实测结果

### 5.1 官方 case1（3 图 / 5.17 秒）完整矩阵

| # | 配置 | NFE | 峰值 MiB | 耗时 | 宿主内存 |
|---|---|---:|---:|---:|---:|
| 0 | 满血 BF16 | 50 | 34549 | 1455s | 149.4 G |
| 1 | 满血 BF16 | 20 | 34529 | 624s | 149.9 G |
| 1b | 满血 BF16 + **编码器 INT8** | 20 | 34549 | 634s | 139.8 G |
| 2 | 满血 + **DiT INT8** | 20 | 29555 | 627s | 144.8 G |
| 3 | **剪枝** BF16 | 20 | 29555 | 622s | 126.4 G |
| 4 | **剪枝 + DiT INT8** | 20 | 29555 | 616s | 115.6 G |
| 5 | 满血 Turbo4 | 4 | 34507 | 217s | 149.7 G |
| 6 | 剪枝 Turbo4 | 4 | 29555 | 196s | 126.9 G |
| 7 | 剪枝 Turbo4 + DiT INT8 | 4 | 29555 | 196s | 114.5 G |
| **8** | **7 + 编码器 INT8** | 4 | **23759** | **174s** | **98.9 G** |
| 8′ | 8 但**关 offload** | 4 | 30011 | 168s | 34.2 G |

### 5.2 官方 case7（9 图 / 10 秒）

| 配置 | NFE | 峰值 MiB | 耗时 | 宿主内存 |
|---|---:|---:|---:|---:|
| 满血 BF16 | 50 | 40187 | **5194s** | 155.0 G |
| 满血 BF16 | 20 | 40187 | 2050s | 154.8 G |
| 满血 Turbo4 | 4 | 40107 | 543s | 155.4 G |
| 剪枝 + DiT INT8 | 4 | 37075 | 505s | 123.2 G |
| **全量化（+编码器）** | 4 | **31261** | **503s** | 109.7 G |
| 全量化 **关 offload** | 4 | 38673 | 482s | 40.0 G |

### 5.3 输入规模包络（全量化 Turbo4，开 offload）

| 输入规模 | 峰值 MiB | 余量* | 耗时 |
|---|---:|---:|---:|
| 官方 3 图 5.17s | 23759 | 16.7 G | 174s |
| 官方 9 图 10s | 31261 | 9.4 G | 503s |
| 9 图 + 3 视频 8s | 37523 | 3.4 G | 914s |
| 6 图 + 3 视频 15s | 38085 | 2.8 G | 982s |
| 9 图 + 3 视频 12s | **39785** | **1.1 G** | 1088~1108s |
| 9 图 + 3 视频 15s | **40405** | **0.5 G** | 1213~1218s |

\* 按 40960 MiB 名义容量算；实际可用只有 39.49 GiB，真实余量更小，见 §5.6。

12s 与 15s 各独立跑了 **3 次**，峰值分别是 39785/39785/39785 与 40405/40405/40405——**完全一致**，
说明该负载显存占用是确定性的。

### 5.3b 成分归因矩阵（NFE20，官方 case1 / case2，同种子）

用于 §8 的画质归因，也给出各成分的显存代价。

| 配置 | case1 峰值 | case1 耗时 | case2 峰值 | case2 耗时 | 画质 |
|---|---:|---:|---:|---:|---|
| 满血 BF16（基准） | 34529 | 624s | 34747 | 680s | ✅ |
| 只剪枝（DiT BF16） | 29555 | 622s | 29837 | 650s | ✅ |
| 只 DiT W8A8（不剪枝） | 29555 | 627s | 29837 | 634s | ❌ case2 丢闪电 |
| 剪枝 + DiT W8A8 | 29555 | 616s | — | — | ❌ case1 丢动作 |
| 剪枝 + 编码器 Int8（DiT 不量化） | 28363 | 619s | 28563 | 661s | ✅ |
| 无剪枝 + DiT W8A8 + 编码器 Int8 | 28613 | 593s | 28733 | 642s | ❌ |
| 剪枝 + DiT W8A8 + 编码器 Int8 | 23759 | — | 24043 | 606s | ❌ |
| **剪枝 + DiT W8A16 + 编码器 Int8** | **25087** | 651s | **24227** | 674s | ✅ **无损** |

W8A16 那一行是关掉 torch.compile 跑的（见 §5.7）；开编译为 24807 MiB / 670s。

### 5.4 满血 BF16 的包络（NFE20）

| 输入 | 峰值 MiB | 结果 |
|---|---:|---|
| 官方 3 图 5.17s | 34529 | ✅ 余 4.9 G |
| 1 图 + 3 视频 + 1 音频 8s | 36507 | ✅ 余 3.0 G（耗时 1356s） |
| 官方 9 图 10s | 40187 | ⚠️ 余 0.3 G，实质不可用 |
| 6 图 + 3 视频 15s | 40023 | ❌ **OOM**（需 3.79 G，仅剩 3.38 G） |
| 9 图 + 3 视频 8s / 12s / 15s | 40183 / 40083 / 40211 | ❌ OOM |

**满血档能带视频，但只能带轻的**：安全线约在"参考文件 ≤5 个 + 时长 ≤8 秒"。

### 5.5 CPU offload 开关

| 场景 | 峰值 MiB | 耗时 | 宿主内存 |
|---|---:|---:|---:|
| case1 全量化，开 | 23759 | 174s | 98.9 G |
| case1 全量化，关 | 30011（**+6252**） | 168s（快 3.4%） | **34.2 G（−64.7 G）** |
| case7 全量化，开 | 31261 | 503s | 109.7 G |
| case7 全量化，关 | 38673（**+7412**） | 482s（快 4.2%） | **40.0 G（−69.7 G）** |

**关 offload 的显存代价随输入规模上升**（+6.3 G → +7.4 G），换回的只有 3~4% 速度。
按此外推，9 图 + 3 视频 15s 关掉后约需 46.6 GiB，**必然 OOM**。

### 5.6 精确峰值（torch 分配器累计最大值）

`_log_step_memory` 本轮补上了 `max_memory_allocated/reserved`（分配器自维护，不漏尖峰）：

| 档位 | 驱动侧 free | torch max reserved | torch max allocated |
|---|---:|---:|---:|
| 9 图 + 3 视频 15s | **0.51 / 39.49 GiB** | 38.77 GiB | 36.40 GiB |
| 9 图 + 3 视频 12s | **0.64 / 39.49 GiB** | 38.16 GiB | 35.11 GiB |

两点修正：

1. **2 秒采样没有低估**——nvidia-smi 的 40405 MiB 与驱动侧 `39.49 − 0.51 = 38.98 GiB` 吻合，
   该负载显存曲线是平台型而非尖峰型。
2. **但真实余量比按 40960 算的更小**：卡的可用容量是 **39.49 GiB**，
   15s 档实际只剩 **0.51 GiB**。

---

### 5.7 W8A16 与 torch.compile

W8A16 此前**只要开着编译就必然启动失败**，栈是 AOT autograd 里的 `RecursionError`：
`regionally_compile` 把 DiT 每个 block 包成编译函数，block 的线性层进入
`Int8WeightOnlyLinearMethod.apply`，`apply` 里对 `nn.Parameter` 取值触发
`__torch_function__` 分发，又重入同一段编译图。

已修（提交见 §9）：权重与 scale 改为 detach 后的普通张量、按当前 storage 惰性缓存
（CPU offload 会搬运权重，加载期固化的视图会让 Triton 报 "cannot be accessed from
Triton (cpu tensor?)"），并给 `apply` 加 `@torch._dynamo.disable`。

代价：DiT 每个线性层成为图断点，**开编译反而比 eager 慢 19 秒**（670s vs 651s）。
所以这个修复不是提速，是让 W8A16 在默认配置下不再崩溃。**当前建议 W8A16 配
`--diffusion-compile-granularity none` 运行**（该取值同批新增）。

## 6. 七条可复用的规律

**① 显存峰值只由最高的那个阶段决定，其余阶段省多少都不进账。**
这是下面所有现象的总根源。

**② 峰值会在阶段之间"搬家"。**
满血 DiT 时峰值在 **denoise**；DiT 一旦压轻（剪枝或量化，**任一即可**），峰值就落回 **encode**。
实测：case1 满血 34529 出现在 261s（denoise）；剪枝/量化各臂 29555 出现在 21s（encode，全程平台）。

**③ 剪枝与 DiT 量化在 GPU 峰值上互相替代——但只在峰值仍落于 encode 平台时成立。**
case1（编码器仍为 BF16 时）：只量化 29555、只剪枝 29555、都上 29555（**一 MiB 不差**）；
case2 同样：只剪枝 29837、只量化 29837。
根因：剪枝砍 AdaLN（Int8 不碰），Int8 压 attn/MLP（剪枝不碰），权重上真互补——
但那时**峰值由 encode 平台决定**，DiT 权重省多少都不进账。

**前提一旦失效，替代关系就不成立。**编码器量化把 encode 平台压到约 24k 之后，
DiT 权重重新露出水面，剪枝省的部分变回可见：

| case | 剪枝 + 编码器 Int8（DiT 不量化） | 无剪枝 + DiT W8A8 + 编码器 Int8 | 差值 |
|---|---:|---:|---:|
| case1 | 28363 | 28613 | 250 |
| case2 | 28563 | 28733 | 170 |

而"剪枝 + DiT W8A8 + 编码器 Int8"是 23759 / 24043——比上面两者各低约 4600 MiB。
也就是说**三者都上时才是真叠加**，此时谁都不能省。

宿主内存与磁盘上的互补始终成立：宿主 144.8 → 126.4 → 115.6 G，DiT 磁盘 43.8 → 37.5 → 19.5 GiB。

**④ 编码器量化必须搭配 DiT 压缩，否则显存零收益。**

| DiT 状态 | 峰值在哪 | 编码器量化收益 |
|---|---|---:|
| 满血 BF16 | denoise | **0**（34529 → 34549） |
| 剪枝 + DiT INT8（case1） | encode | **−5796 MiB** |
| 剪枝 + DiT INT8（case7） | encode | **−5814 MiB** |

该收益是**常数**（约 5.8 GiB，= 编码器每卡 12.35 → 6.5 GiB），不随输入规模变化。
依赖链：`DiT 压轻 → 峰值落到 encode → 编码器量化才兑现`。

**⑤ 步数只买时间，完全不影响显存。**
满血 NFE20 = 34529、NFE50 = 34549（差 20 MiB）；case7 NFE20/50 均为 40187。
全量化各臂 NFE4/20/50 全部落在同一个数。
**推论：开放用户自选步数在稳定性上是安全的**，20 步能跑的输入，50 步也能跑。

**⑥ 蒸馏是唯一大幅省时间的手段，且几乎不省显存。**
case1：624s（20 步）→ 217s（Turbo4），快 **2.9 倍**，峰值只差 22 MiB。
case7：2050s → 543s，快 **3.8 倍**。

**⑦ 权重量化与激活量化必须分开评估——省显存的是前者，伤画质的是后者。**
DiT 的 Int8 有两档，权重完全相同，只差 `activation_scheme`：

| | 峰值(case1) | 耗时 | 画质 |
|---|---:|---:|---|
| W8A8（`dynamic`，权重+激活） | 23759 | 616s | ❌ 掉动作/闪电 |
| W8A16（`weight_only`，仅权重） | 25087 | 651s | ✅ 无损 |

**差 1328 MiB 和 5% 时间，换回全部画质。**这也意味着凡是"治激活离群值"的手段
（SmoothQuant、旋转量化 convrot）在这里才是对症的；而"治权重"的手段（GPTQ 校准等）
解决不了本文观察到的退化。详见 §8。

### 耗时模型（两点拟合，中间点未验证）

| 输入规模 | 每步耗时 | 固定开销 |
|---|---:|---:|
| 官方 3 图 5.17s | ≈ 27.7 s/步 | ≈ 70s |
| 官方 9 图 10s | ≈ 105 s/步 | ≈ 0 |

**每步耗时随输入规模差近 4 倍**，所以步数上限应按"预估耗时 ≤ 超时阈值"卡，不能拍固定步数。
按 `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=7200` 算：3 图 5.17s 可到约 255 步，9 图 10s 只能到约 68 步。

---

## 7. 部署建议

### 两个引擎，三档对外

| 引擎 | 权重 | 卡 | 对外档位 | 输入上限 |
|---|---|---|---|---|
| 引擎 1 | **剪枝 + DiT W8A16 + 编码器 Int8** + Turbo4 | 4 | **快**（固定 4 步） | 待按 W8A16 重测（见 §10） |
| 引擎 2 | **剪枝 + DiT W8A16 + 编码器 Int8** 非蒸馏 | 4 | **标准**（默认 20 步）+ **高质**（用户自调 30/50） | 同上 |

要点：

- **两个引擎用同一套压缩配置**，只差有没有融 Turbo4 LoRA。初版建议高质档走满血 BF16
  以规避量化风险，现在不必了：W8A16 在两个 case 上都无损，而满血的输入包络小得多
  （带视频参考基本不可用，见 §5.4）。
- **标准与高质权重相同、只差步数**，共用一个引擎（`supports_num_inference_steps_override: True`），
  8 张卡而不是 12 张。
- **快档不开放调步数**——Turbo4 把 4 步蒸馏进了权重，调高步数是偏离其训练工作点。
- **两个引擎都要带 `--diffusion-compile-granularity none`**（见 §5.7）。
- 网关三个模型条目，`defaultSteps` 分别 4 / 20 / 20，前者路由引擎 1。
  （注意：步数要按模型配 `defaultSteps`，不能写死在引擎族上。）
- **不要为省那 1.3 GiB 改回 W8A8**——它省的显存和时间都不足以补偿 §8 的画质退化。

### 必须设的护栏

1. **offload 一律开启。**它换来的 6~7 GiB 就是大输入能否跑通的全部本钱，代价只有 3~4% 速度。
2. **9 图 + 3 视频 15s 默认关闭。**在 W8A8 下真实余量就只有 0.51 GiB，属于"每次恰好没爆"；
   换成 W8A16 后峰值再高约 1.3 GiB，**该档大概率直接 OOM**，重测前不得开放。
3. **每引擎串行单请求。**极限档没有第二个请求的空间。
4. **提交时按档位包络硬拒 + 给出预估耗时**，不要让用户等几十分钟才拿到 OOM。
   超出包络时建议**自动降级到快档并告知**，而非直接拒绝。

---

## 8. 画质：W8A8 会削掉瞬态事件，W8A16 不会

**这一节的结论与本报告初版相反。**初版写的是"肉眼未见明显差异"，那是只看了一个 case、
且只比较了单帧清晰度的结果。逐个 case 看动态之后，发现了真实退化。

### 8.1 观察到的两处退化

| case | 症状 | 出现在 |
|---|---|---|
| case1（3 图 5.17s） | 中段一个动作**整个丢失** | 剪枝 + DiT W8A8 |
| case2（3 图 5.17s） | 窗外的**闪电退化成一次亮光** | 只 DiT W8A8（不剪枝也复现） |

两者都是**时序性**的：单帧截图看不出来，逐像素指标也测不出来。

### 8.2 归因：唯一元凶是激活量化

按成分逐个隔离，全部同素材、同官方提示词、同种子、NFE20：

| 成分 | case1 动作 | case2 闪电 | 判定 |
|---|---|---|---|
| 剪枝 r8（DiT 不量化） | ✅ 正常 | ✅ 闪电在 | 干净 |
| 编码器 Int8（权重-only） | ✅ 正常 | ✅ 闪电在 | 干净 |
| **DiT W8A8**（权重+激活） | ❌ 动作丢失 | ❌ 闪电丢失 | **元凶** |
| **DiT W8A16**（仅权重） | ✅ **动作在** | ✅ **闪电在** | 无损 |

最后两行是**同一份 Int8 权重**——W8A16 的 checkpoint 用符号链接复用 W8A8 的张量文件，
两者只有 `config.json` 里 `activation_scheme` 一个字段不同（`dynamic` vs `weight_only`）。
所以这是单变量对照：**问题出在激活量化，不在权重量化**。

推测机制：闪电是一次剧烈的激活幅值跃升，per-token 动态量化把整个 token 的 scale 交给
这个离群值，同 token 内其余通道分到的 8 bit 分辨率随之塌陷。剪枝会加剧它——case1 中
单独 W8A8 尚未露馅，叠加剪枝后露了，因为低秩调制误差本身也在制造离群值。

### 8.3 PSNR 为什么没抓到

同种子下 Int8 与 BF16 的 PSNR 约 23 dB（Y 21.3 dB）。这个数**既不能证明有问题、也不能
证明没问题**：4 步蒸馏采样下权重的任何扰动都会把整条采样轨迹带走，逐像素指标必然低。
初版把它解释成"轨迹发散、不代表画质下降"，方向对，但据此推出"没有可见差异"是过度解读。

### 8.4 证据强度

- 覆盖 **2 个官方 case**（case1 动作、case2 闪电），每个 case 都做了成分级隔离
- 判断方式是**肉眼看动态**，非指标
- 官方测试集共 8 个 case，其余 6 个（case4~8）**未跑 A/B**

**评测集缺一类指标**：既没有"参考主体保持度"，也没有"瞬态事件保留度"。前者是 §2 那个
权重取错分支能藏住 8 小时的根因，后者是本节这两处退化能通过全部现有检查的根因。

---

## 9. 复现

### 工具（本轮新增/修改）

| 文件 | 作用 |
|---|---|
| `vllm_omni/quantization/tools/quantize_qwen3vl_encoder_int8.py` | 编码器 INT8（新增） |
| `vllm_omni/diffusion/models/minimax_h3/encoder.py` | 编码器 Int8 加载路径（新增 `_Int8WeightMixin`） |
| `vllm_omni/diffusion/models/minimax_h3/denoise_loop.py` | 补 `max_memory_*` 精确峰值日志 |
| `tools/minimax_h3_turbo/convert_pruned_to_partition.py` | diffusers → partition（新增） |
| `tools/minimax_h3_turbo/run_official_ref2va_case.sh` | 官方 case 驱动，支持 `CPU_OFFLOAD=0` |
| `tools/minimax_h3_turbo/run_ref2va_scale_rung.sh` | 输入阶梯驱动 |
| `tools/minimax_h3_parity/verify_checkpoint_conversion.py` | 补 `norm_out.folded_bias` 改名规则 |
| `tools/minimax_h3_turbo/assemble_pruned_partition.py` | 支持两种 index 布局 |

### 端到端命令

```bash
# 1. diffusers 剪枝件 → partition
python3 tools/minimax_h3_turbo/convert_pruned_to_partition.py \
  --src  $M/MiniMax-H3-Pruned-r8/transformer_ref \
  --output $M/MiniMax-H3-Ref2VA-Pruned-r8-BF16-partition/transformer

# 2. bit-parity（必须 "checkpoints agree" 才继续）
python3 tools/minimax_h3_parity/verify_checkpoint_conversion.py \
  --diffusers $M/MiniMax-H3-Pruned-r8/transformer_ref \
  --partition $M/MiniMax-H3-Ref2VA-Pruned-r8-BF16-partition/transformer --heads 56

# 3. DiT INT8
python3 vllm_omni/quantization/tools/quantize_minimax_h3_int8.py \
  --src $M/MiniMax-H3-Ref2VA-Pruned-r8-BF16-partition \
  --dst $M/MiniMax-H3-Ref2VA-Pruned-r8-INT8-partition

# 4. 编码器 INT8
python3 vllm_omni/quantization/tools/quantize_qwen3vl_encoder_int8.py \
  --src $M/MiniMax-H3/Ref2VA/text_encoder \
  --dst $M/Qwen3-VL-32B-H3Encoder-INT8

# 5. 由 W8A8 派生 W8A16：复用同一份张量，只改 activation_scheme
#    （权重用符号链接，磁盘不翻倍）
mkdir -p $M/...-INT8-W8A16-partition/transformer && cd $_
ln -s $M/...-INT8-partition/transformer/*.safetensors .
ln -s $M/...-INT8-partition/transformer/model.safetensors.index.json .
python3 -c 'import json; c=json.load(open("SRC/config.json")); \
  c["quantization_config"]["activation_scheme"]="weight_only"; \
  json.dump(c, open("config.json","w"), indent=2)'

# 6. 组装 serve 根（Turbo 档需带 --num-inference-steps 与 --fusion-provenance）
python3 tools/minimax_h3_turbo/assemble_pruned_partition.py ...

# 7. serve（不要传 --quantization，已序列化；W8A16 需关闭编译，见 5.7）
vllm serve $M/MiniMax-H3-Ref2VA-Pruned-r8-INT8-W8A16-encINT8-vLLM --omni \
  --num-gpus 4 -tp 4 --enable-cpu-offload --text-encoder-tp-size 4 \
  --diffusion-compile-granularity none ...
```

### 产物位置

- 视频（按对照顺序命名的软链）：`/nfs-output/h3_official_eval/看片/`
- 官方 case 原始结果：`/nfs-output/h3_official_eval/results/<tag>/`
- 输入阶梯原始结果：`/nfs-output/h3_pruned_eval/ref2va_scale_20260820/<tag>/`
- 运行日志：`/nfs-output/h3_official_eval/logs/`、`/nfs-output/h3_int8_pruned/logs/`

---

## 10. 未完成 / 已知风险

| 项 | 状态 |
|---|---|
| **W8A16 下的输入包络** | **必须重测**。§5.3 的包络是 W8A8 的；W8A16 峰值高约 1.3 GiB，9图+3视频 15s（原余量 0.51 GiB）大概率不再可跑 |
| 官方 case4~8 的 A/B | **未跑**；画质结论目前建立在 case1（动作）与 case2（闪电）两个样本上 |
| "瞬态事件保留度"客观指标 | **无**。§8 那两处退化通过了现有全部检查，只有肉眼看动态才发现 |
| "参考主体保持度"客观指标 | **无**，这是 §2 权重取错分支能藏住 8 小时的根因 |
| W8A16 + 编译的性能 | 已能跑，但**开编译反比 eager 慢 19 秒**（§5.7）。要提速需把 Triton 桥接包成 custom_op 而非整体 `disable` |
| 耗时模型中间点 | ✅ **已验**：3 图 @30 步预测 901s、实测 937s，误差 3.8%（§6 耗时模型） |
| 参考图分批编码 | **未做**；全量化后激活占约 73%，是显存的下一个大头 |
| 旋转量化（convrot）/ SmoothQuant | **未评估**，但 §8 定位到激活量化之后，**这类手段才是对症的**；上游剪枝仓库 README 有 `enable_convrot()` 参考实现 |
| TP2+SP2 等并行策略 | **未测**；峰值在 encode 时无收益，只有当 denoise 重新成为峰值才有靶子 |

---

## 11. 附：全部 54 次运行原始记录

见 `/nfs-output/h3_official_eval/results/*/summary.json` 与
`/nfs-output/h3_pruned_eval/ref2va_scale_20260820/*/summary.json`。
每条含 `peak_gpu_memory_mib`、`peak_host_memory_gib`、`wall_seconds`、
`engine_log.new_row_lines`（÷4 = 实际 NFE）、输出 mp4 的 sha256 与字节数。
