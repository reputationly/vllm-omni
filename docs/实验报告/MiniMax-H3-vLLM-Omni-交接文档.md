# MiniMax-H3 on vLLM-Omni / A100-40G × 4 —— 交接文档

> 交接对象：接手的另一个 agent
> 截止时间：2026-08-09（本轮 output-opt/Ref2VA/FL2VA 结果见 §10.13）
> 目标：33B DiT + Qwen3-VL-32B 文本编码器、视频+音频联合生成，跑在 4× A100-PCIE-40GB 上
> 当前状态：**W8A8 TP4 生产候选**（480p/768p×15s 均通过）。正式仓 c7 集成已完成；
> 本轮 6 文件 output-opt 仍是未提交/未固化镜像的候选补丁。Ref2VA 中文叠声和 FL2VA
> 无对白遵从未通过，不得写成音频质量全部验收。

---

## 0. 三十秒速览

| 项 | 结论 | 置信度 |
|---|---|---|
| 拓扑 | **`-tp 4`**，本轮不再研究 TP2+SP2/USP4 | 高 |
| 量化 | **serialized W8A8 为默认候选**；NF4 已淘汰，W8A16 非默认 | 高 |
| 480p 15 秒档 | pre-barrier 严格 A/B：216.56 s、28155 MiB/卡；最终屏障版待镜像冒烟 | 中高 |
| 768p 10 秒档 | 历史 NF4/BF16 数据；当前容量基线用 15 秒 W8A8 结果 | — |
| 768p 15 秒档 | pre-barrier 严格 A/B：846.76 s、36617 MiB/卡；最终屏障版待镜像冒烟 | 中高 |
| 代码 | 正式仓 `c7e56d68` 已集成；当前 output-opt 为 6 文件未提交改动 | — |

---

## 1. 硬件与环境

### 1.1 机器

四台**完全同构**，全部在 `111.172.214.16`，靠 SSH 端口区分：

| 代号 | SSH 端口 | 备注 |
|---|---|---|
| 0024 | 43046 | |
| 0025 | 43047 | |
| 0026 | 43051 | |
| 0030 | 43055 | 另跑 arcreel/openchatcut，均无 GPU 占用，可忽略 |

- 规格：251 GB RAM / 128 核 / 4× A100-PCIE-40GB
- swap 只有 4 GB —— **宿主内存打爆就是 OOM killer 直接杀 worker，没有缓冲**
- 容器：`voh3`，宿主 `/root/h3` ↔ 容器 `/work`
- **docker bridge 网络**：未发布 `-p 8091:8091` 时，宿主 `127.0.0.1:8091`
  不通；可在容器内访问 `127.0.0.1:8091`，也可由宿主直连 `docker inspect`
  得到的容器 bridge IP。本轮两种方式都用过。
- 四台都跑 `gpustack-worker`（391–631 MiB，0.13–0.24% CPU，**不占显存**）。**这个容器不能停**，生产环境要用。对实验无影响。
- 管理节点 `root@111.172.214.42`，免密无端口，**无 GPU、无 ffmpeg**。按用户要求，**模型权重下载必须挂在管理节点执行，写到 NFS**。

### 1.2 存储

- NFS：`100.125.40.2:/share-LLM`
- 计算节点看到的 `/nfs-data/` == 管理节点的 `/nfs-models/wuhanjisuan894/`
- BF16 基础分区：`/nfs-data/models/MiniMax-H3/FL2VA`
- 当前 W8A8 生产候选权重：`/nfs-data/models/MiniMax-H3-FL2VA-INT8`
- 样片输出：`/nfs-data/h3_samples/`

### 1.3 A100-40G 的真实容量

- `nvidia-smi` 报 40960 MiB
- **PyTorch 实际可用 39.49 GiB** —— 所有显存算术都要用这个数，不是 40

---

## 2. 当前 W8A8 TP4 生产候选启动命令

> 历史文档在这里放过 `/work/vllm-omni-h3` + 在线 NF4 + `--enforce-eager`
> 的旧实验命令。该路径已淘汰，不得再当作 canonical 复制。

当前生产候选固定为 serialized W8A8 + TP4 + regional dynamic compile + pre-VAE
offload，单实例串行。output-opt 在本轮是用 c7 容器 + 挂载源码验证；生产发布
必须先把它固化到新的通用不可变镜像，不应让 GPUStack 依赖 `PYTHONPATH` overlay。
下面的环境变量和 `vllm serve` 命令在最终镜像的 `voh3` 容器内执行。

```bash
export VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0        # 0 = 关闭
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=14400
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400      # 注意：没有 _S 后缀
export VLLM_OMNI_H3_OFFLOAD_DIT_BEFORE_VAE=1   # 默认为 0，必须显式打开
export VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK=8

vllm serve /nfs-data/models/MiniMax-H3-FL2VA-INT8 --omni \
  --host 0.0.0.0 --port 8091 --trust-remote-code \
  --num-gpus 4 -tp 4 --usp 1 --ring 1 --max-num-seqs 1 \
  --enable-cpu-offload \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN \
  --diffusion-compile-granularity regional --diffusion-compile-dynamic \
  --safetensors-load-strategy lazy --disable-multithread-weight-load \
  --enable-diffusion-pipeline-profiler \
  --init-timeout 2400 --stage-init-timeout 2400
```

- `pre-VAE` 开关的代码默认仍为关；只写 `--enable-cpu-offload` 不会自动获得该优化。
- GPUStack 可以在实例外排队，但不等于 H3 有请求合批能力；当前服务必须
  `--max-num-seqs 1`，不要向同一实例并发两个 768p 请求。
- 冷启动实测约 5–6 分钟，启动等待循环仍必须监视服务进程是否存活。
- W8A8 首请求前常驻约 24.4 GiB/卡；完成至少一次请求、pre-VAE offload 生效后
  常驻约 14.0 GiB/卡。请求峰值不能按这两个 idle 数外推，直接看 §10.13 表格。

### 2.1 请求格式

下面的 `127.0.0.1` 示例也在 `voh3` 容器内执行；若从宿主发请求，应改用已发布
的宿主端口或容器 bridge IP。

```bash
curl -sS -X POST http://127.0.0.1:8091/v1/videos/sync --max-time 7200 \
  -F "prompt=${PROMPT}" -F "aspect_ratio=16:9" \
  -F "width=1344" -F "height=768" -F "fps=24" \
  -F "num_frames=360" -F "num_inference_steps=20" -F "flow_shift=12" -F "seed=11" \
  -F 'extra_params={"task":"t2va","audio_flow_shift":3.0}' \
  -o out.mp4 -w "http=%{http_code} wall=%{time_total}s bytes=%{size_download}"
```

`/v1/videos/sync` 直接返回 MP4 裸字节。

`fps` 固定为 24。调用方提交的是**请求帧数/时长**，只需落在当前 `[4, 15] s`
契约内；pipeline 会把请求帧数**向上对齐**到 `17n+5` 的实际输出帧数。请求本身
不应被网关按 `n % 17 == 5` 拒绝。常用档位：

| 请求时长 | 请求帧数 | 对齐后的实际输出帧数/视频时长 |
|---:|---:|---:|
| 5 s | 120 | 124 / 5.17 s |
| 8 s | 192 | 192 / 8.00 s |
| 10 s | 240 | 243 / 10.125 s |
| 12 s | 288 | 294 / 12.25 s |
| **15 s** | **360** | **362 / 15.08 s** |

> 网关防呆应检查请求时长为 4–15 秒、fps=24，并让 H3 pipeline 完成帧对齐。
> 直接提交 362 帧会先按 `362/24=15.083 s` 校验，因此超过 15 秒而被拒绝；
> 旧实验中的 719 帧/约 30 秒不属于当前 c7 API 契约。

---

## 3. 历史实验仓代码改动清单（非当前正式仓待办）

> 本节保留基线 commit `900a7f08`、实验仓
> `/Users/reputationly/Desktop/code/api/vllm-omni-h3`（节点 `/work/vllm-omni-h3`）的历史快照，
> 不能当作当前 c7 正式仓待办清单；当前状态以 §10.12/§10.13 为准。

```
14 files changed, 1720 insertions(+), 225 deletions(-)
```

| 文件 | 行数 | 内容 |
|---|---|---|
| `diffusion/models/minimax_h3/encoder.py` | +471 | 编码器卸载/换入换出主战场 |
| `diffusion/offloader/sequential_backend.py` | +290 | P0-1 换出不 copy、常驻判定 |
| `diffusion/models/minimax_h3/minimax_h3_transformer.py` | +290 | 历史仓曾含临时探针；正式 c7 已清除 |
| `diffusion/offloader/distributed_layerwise_backend.py` | +231 | DLO mmap 分片 |
| `diffusion/diffusion_engine.py` | +122 | 假墙超时 |
| `diffusion/models/minimax_h3/pipeline_minimax_h3.py` | +121 | |
| `diffusion/model_loader/diffusers_loader.py` | +89 | |
| `diffusion/models/minimax_h3/vae.py` | +89 | **#45 分块反归一化** |
| `diffusion/models/minimax_h3/denoise_loop.py` | +62 | |
| `entrypoints/openai/video_api_utils.py` | +48 | 前端逐帧单缓冲（修 #28 泄漏） |
| `quantization/int8_config.py` | +25 | |
| `quantization/bitsandbytes_config.py` | +17 | **`ignored_layers_match`** |
| `tests/...` (2 个) | +90 | |
| `quantization/tools/quantize_minimax_h3_int8.py` | 新增未跟踪 | **离线 int8 量化工具，还没用过** |

### 3.1 三处需要重点交接的改动

**(a) `quantization/bitsandbytes_config.py` —— `ignored_layers_match`（+17 行）**

`is_layer_skipped()` 默认是**整路径精确相等**，只适合离线 checkpoint 那种把每一层全名列出来的 `modules_to_not_convert`。要按子树排除（`["text_model"]` 保住整个文本编码器）就必须用子串匹配。

- `__init__` 增加 `ignored_layers_match: str = "exact"`，取值校验 `("exact","substring")`
- `from_config` 用 `get_from_keys_or(config, ["ignored_layers_match"], "exact")` 读
- `get_quant_method` 传 `skip_with_substr=self.ignored_layers_match == "substring"`

默认保持 `exact`，不改变既有 checkpoint 行为。

> 历史实验仓当时的 `int8_config.py` 还有同款缺口；正式 c7 已补齐
> `ignored_layers_match` / `skip_with_substr` 并通过 W8A8 权重加载验证，不再是待办。

**(b) `diffusion/models/minimax_h3/vae.py` —— #45 分块反归一化（+89 行）**

VAE 后处理原本要在显存里拿整片 float32 缓冲（768p 每帧 11.81 MiB，362 帧就是 4.18 GiB，前后要三份），这是 768p 时长天花板的真正成因，不是去噪阶段。

改动：`MINIMAX_H3_REVERT_FRAME_CHUNK = 16`（env 覆盖 `VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK`），新增 `_revert_frame_chunk()` / `_as_video_5d()` / `_revert_frames()`，按帧轴分块并组装到 **CPU** float32 缓冲。

**数学上是位级等价的，已证明**：`revert_tensor` 就是 torchvision `Normalize(mean=(-2.1179, -2.0357, -1.8044), std=(4.3668, 4.4643, 4.4444))` 再 `.clamp(0,1)`，纯逐元素运算，切帧不影响。预检在 chunk=16/7/1/64 四种切法下 `maxdiff=0.0`、`torch.equal=True`。

*判断文件是否已打补丁*：`grep -c REVERT_FRAME_CHUNK` → 打过是 **6**，`git show HEAD:` 的原版是 **0**。

同文件里 `MiniMaxH3AudioVAE.decode` 有一段纯注释的 NOTE，记录为什么删掉 `_AudioVAEDeterminismContext`——它设了 `cudnn.enabled = False`，把总耗时从 53–64 s 拖到 336/840/597 s。**别手贱加回去。**

**(c) `minimax_h3_transformer.py` —— 历史临时探针（正式 c7 已清除）**

`_adaln_probe` / `_adaln_probe5` / `self._probe_prefix` / `VLLM_OMNI_ENCODER_PROBE`。这些是 #41 分离"NF4 数据坏 vs kernel 坏"时加的，结论已经拿到，代码该回滚。原始版本在 0024 上留了备份：`/work/transformer_before_p5.py`。

正式 c7 已核对不存在上述符号，不需要再次删除；本段只用于解释旧实验仓来源。

---

## 4. 测试矩阵与结论

所有测试统一：prompt 固定、`seed=7`、`steps=20`、`flow_shift=12`、`extra_params={"task":"t2va","audio_flow_shift":3.0}`。文本编码器全程 BF16（靠 `ignored_layers:["text_model"]` 退回），所以量化那一栏变的只有 DiT。

### 4.1 拓扑对照（**结论已定，置信度高**）

**设想**：`--usp 4`（序列并行）比 `-tp 4`（张量并行）快，因为 PCIe 上 all-reduce 贵。问题是快多少、代价是什么。

**768p/243，两边都是 NF4，唯一变量是拓扑：**

| | usp4（0024） | tp4（0026） |
|---|---|---|
| wall | 449.4 / 385.4 / 373.2 s | 539.3 / 505.9 s |
| 稳态 | **373–385 s** | 506–539 s |
| 峰值显存/卡 | 38.0–38.7 GB | **34.7–36.4 GB** |
| 空载宿主 AnonPages | 216.5 GB | **160.4 GB** |
| 请求中 MemAvailable 最低 | **10.9–12.4 GB** | **63.3 GB** |
| 确定性 | r1≠r2≠r3 | r1==r2 逐字节相同 |

**480p/719（历史旧 API 的 30 秒实验档，两边都是 NF4；当前 c7 不接受）：**

| | tp4 | usp4 |
|---|---|---|
| 空载 MemAvailable | 76.5 GB | 24.4 GB |
| r1 | ✅ 673.8 s，30.6 GB/卡 | ❌ 500 @ 14.3 s（NCCL，363 字节） |
| r2 | ✅ 622.0 s，30.6 GB/卡 | ❌ 500 @ 1800.1 s |

**结论：用 `-tp 4`。**usp4 快 1.30×，但它是唯一失败过的拓扑：

| 拓扑 | 空载 MemAvailable | 首集合通信失败 |
|---|---|---|
| tp4 | 76–98 GB | **0 / 13 次请求** |
| usp4 | 19–24 GB | **4 次** |

机制（从代码读出来的）：DiT 组的 NCCL 通信子是**惰性创建**的（`pipeline_minimax_h3.py:220`，`dist.broadcast(shape, src=0, group=group)`），创建时 NCCL 的 channel buffer 走裸 `cudaMalloc`，**绕开 PyTorch 的 caching allocator**。usp4 把 DiT 在 CPU 侧复制 4 份，空载就吃掉 216 GB，只剩 22–24 GB 给系统——在这个水位上分配是掷硬币（0024 在 22.2 GB、0026 在 23.7 GB 赌赢过，0030 在 24.4 GB 输了）。tp4 把 DiT 切成 4 份，空载 160 GB 留 76 GB，13 次没输过。

> **两条被证伪、别再走的路**：
> 1. "0026 那台机器坏了" —— 0030 跑同一个 control 产出**逐字节相同**的 363 字节错误体（md5 `2a591304`）。是配置问题，不是硬件。
> 2. "宿主内存累积泄漏导致 pinned 分配失败" —— 0026 在 MemAvailable 24.1 GB 挂掉、0024 在 23.8 GB 成功。不是绝对水位，是 usp4 长期把宿主压在那条线上。

> **顺带发现的第四堵假墙**：`wall=1800.082s` 不是我们的超时（`VIDEO_SYNC_TIMEOUT` 设了 14400），是 PyTorch `ProcessGroupNCCL` watchdog 的默认 30 分钟。首请求把通信子打坏后，后续请求要等满 watchdog 才报错。

### 4.2 历史 NF4 量化对照（已由 W8A8 结论覆盖）

> 本节只保留 NF4 淘汰过程的证据，不是当前验证计划；生产选型见 §10.8/§10.13。

**设想**：NF4 省显存但可能掉画质，BF16 是没有量化的参照物（不是生产候选）。要分清"画质差异"和"本来就在抖"，所以每档跑 3 次。

**480p/124，tp4，唯一变量是 `--diffusion-quantization-config`：**

| 槽位 | BF16 | NF4 现状配方 | NF4 修正配方（C 臂） |
|---|---|---|---|
| r1 | 3,501,012 | 3,925,384 | 3,845,981 |
| r2 | 3,500,997 | **4,480,112** | **4,362,525** |
| r3 | 3,500,997 | 3,925,398 | *（跑到本文档写作时未出）* |
| wall | 80.1 / 78.8 / 78.9 s | 89.4 / 75.3 / 71.4 s | 77.8 / 73.9 s |
| 峰值/卡 | 29.4 GB | 24.1 GB | 24.0 / 24.4 GB |

**768p/243，tp4：**

| | BF16 | NF4 现状 |
|---|---|---|
| wall | 512.1 s | 539.3 / 505.9 s |
| 峰值/卡 | 37.1 GB | 34.7–36.4 GB |
| 字节 | 13,345,767 / 13,345,751 | 10,177,188 / 10,177,188 |

**已经确定的：**

1. **耗时上量化基本不影响**（768p/243：NF4 506–539 s vs BF16 512 s）。省的是显存，不是时间。
2. **BF16 是逐字节确定的**（768p/124 三轮 md5 全 `d4fb5c51`；480p r2==r3）。NF4 在 480p 不确定，在 768p/243 的 tp4 下反而 r1==r2。
3. **NF4 的"正常轮"就比 BF16 大 12.1%**（3,925,384 vs 3,501,012，480p/124 同 seed 同参数）。这不是偶发，是系统性偏移——每一帧都在多出高频内容。
4. **胖片是可复现的，不是随机的**：NF4 现状和 C 臂的胖片都落在 **r2 这个槽位**。所以"NF4 有 1/3 概率出坏片"这个说法不准确，更像是"某些请求序位上稳定出坏片"。

**⚠️ 一个必须纠正的统计学错误**：早前写过"BF16 0/6 次出胖片"。**这个分母是假的**——BF16 逐字节确定，重复跑同 seed 不产生新信息，实际只有 2 个独立样本（480p 一个、768p/124 一个）。`(2/3)² = 44%`，什么都定不了。要给 NF4 定罪必须**换 seed 做配对设计**，不是重复跑同 seed。

### 4.3 历史 NF4 配方问题（实验已停止）

**设想**：+12% 的抬升可能不是 NF4 格式的固有代价，而是我们把不该量化的层也量化了。

`bitsandbytes_config.py:100-124` 的 `get_quant_method()` 逻辑是：**只要是 `LinearBase`，不在 `ignored_layers` 里就一律量化**。我们只传了 `["text_model"]`。把 H3 transformer 的线性层列全：

| `minimax_h3_transformer.py` 行 | 层 | 该不该量化 |
|---|---|---|
| 493 / 509 / 641 / 653 | `qkv_proj` / `out_proj` / `fc1` / `fc2` | ✅ 参数量的绝大头 |
| **714** | AdaLN 里的 `self.linear` | ❌ 调制层，DiT 里最不能量化的 |
| **1070 / 1079** | `video_patch_proj` / `audio_patch_proj` | ❌ 输入嵌入 |
| **911 / 920** | `video_out` / `audio_out` | ❌ 输出头 |
| **1088** | `condition_proj` | ❌ 条件注入 |
| **412 / 421** | `proj_in` / `proj_out` | ❌ 进出投影 |

diffusers / bnb 的 DiT 配方从来都排除后面这几类：它们参数占比小，但权重上的小误差会变成整帧的尺度/偏移误差。

两条旁证：**#14** 就是"修 `adaln_proj` 按 weight.dtype 转激活，量化后截断成 uint8"——adaln 被量化后已经炸过一次。而 `pre45.py:45` 把 `("blocks.0.adaln_proj", False)`（adaln 应当被量化）当成**正确断言**写进预检还跑绿了，等于把 bug 验收成了预期行为。

**C 臂（修正配方）当时的实测记录：**

```json
"ignored_layers": ["text_model","patch_proj","condition_proj",
                   "proj_in","proj_out","video_out","audio_out"],
"ignored_layers_match": "substring"
```

子串匹配的 17 条断言全过，关键是 `proj_out`（transformer 出口，退 BF16）和 `out_proj`（注意力输出投影，继续 NF4）是两个不同的串，不会互相误伤；`adaln.linear` 这轮**没有**排除（它是 `hidden → 6*hidden`，每块一份，排了要真花显存）。

**初步结果：配方生效了（产物变了），但相对 BF16 只从 +12.1% 压到 +9.9%。**不是预期的量级。嫌疑往 AdaLN（这轮没排）和格式本身走。768p/243 那一发用来量排层的显存代价，**写作时还没出**。

### 4.4 #45 VAE 分块（**已完成，结论：正确且必要**）

**设想**：768p 的时长天花板卡在 VAE 后处理的整片 float32 缓冲，不在去噪。

**结论**：成立。分块之后 768p 从 260 帧（10.8 s）推到 362 帧（15.08 s，官方标称上限）。位级等价已证明（§3.1b）。

**#46 已撤回**（标 completed）：曾怀疑 209 帧那个码率异常（2.3 MB/s vs 同分辨率其他档 1.0–1.3 MB/s）是分块造成的。不成立——分块位级等价，数学上不可能改变画面；而且那个异常在 #45 之前的样片里就有。

---

## 5. 已解决问题总账

按发现顺序，都是踩过的坑。序号对应 task id。

| # | 问题 | 根因 / 修法 |
|---|---|---|
| 8 | H3 BF16 输出是噪音 | 见 #9/#10 |
| 9 | DLO mmap 加载绕开 `weight_loader`，H3 qkv 错位 | 走回 weight_loader |
| 10 | DLO mmap 路径不加载 checkpoint 里的 persistent buffer（`rope.inv_freq`） | 补加载 |
| 4 | 给 pipeline 加 `_remap_ckpt_key` 打开 DLO mmap 分片 | |
| 13 | 顺序卸载不搬 bitsandbytes `quant_state` | 换出/换入时一起搬 |
| 14 | `adaln_proj` 按 `weight.dtype` 转激活，量化后被截断成 uint8 | 按激活 dtype 转 |
| 15 | `_ASYNC_OUTPUT_TIMEOUT` 写死 30 秒，卡死大分辨率出片 | 改成可配 |
| 16 | cpu-offload 打爆宿主 251 GB，OOM killer 杀 worker | 见 #17/#38 |
| 17 | P0-1 换出时 copy 一份，改成切指针回 CPU 主本 | |
| 38 | 编码器换出改走 `_move_params`，消掉 60 GB pinned 影子缓冲 | |
| 21 | 音频解码首请求与后续有微小数值差异 | 加确定性上下文；**但该上下文关了 cudnn，拖慢 10 倍，最终删除**（见 §3.1b） |
| 22 | MP4 封装从 2.5 s 劣化到 180 s | |
| 28/29/30 | API 前端每请求泄漏 +568 MB anon（抖动真因） | 后处理逐帧单缓冲，消除 huge 分配；其他 pipeline 同款 `.numpy()` 零拷贝跨线程泄漏一并修 |
| 45 | VAE 后处理整片 float32 缓冲 = 768p 时长天花板 | 按帧分块 + 落 CPU |
| 6 | NCCL cuMem 与 `expandable_segments` 抢虚拟地址空间 | （已关闭，但本轮没能复核 `PYTORCH_CUDA_ALLOC_CONF` 确实设上了） |

### 5.1 三堵服务端超时墙（启动参数已绕开，代码默认值未改）

| 变量 | 默认 | 位置 | 行为 |
|---|---|---|---|
| `VLLM_OMNI_INPUT_WAIT_TIMEOUT_S` | 600 | `core/sched/omni_scheduler_mixin.py:53` | ≤0 可关闭 |
| `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S` | 300 | `diffusion/diffusion_engine.py:61` | |
| `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` | 600 | `entrypoints/openai/api_server.py:3188` | → HTTP 504，**注意没有 `_S` 后缀** |

`curl --max-time` 不能代替这三个服务端变量。当前 TP4 生产候选必须显式设为
`VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0`、`VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=14400`、
`VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400`；“已拆”只表示启动配置绕开了它们，不是
源码默认已改。

第四堵见 §4.1 末（NCCL watchdog 1800 s，不是我们的）。

---

## 6. 历史未确定问题（已由 §10 的 W8A8 主线覆盖）

> 本节保留早期 NF4 调研和推理过程，不能作为当前待办执行。NF4 已淘汰；正式 c7
> 已包含 INT8 `ignored_layers_match`、量化工具与 H3 集成，serialized W8A8 已完成
> 480p/768p 实测。现行待办只看 §8 和 §10.13。

### 6.1 历史：NF4 的 +12% 追因

**当时状态**：修正配方只压回 2 个百分点，三个候选尚未排开：

1. **AdaLN 仍在量化**（C 臂没排）。这是理论上最可疑的一个。代价：`hidden → 6*hidden` × 每块一份，排除要真花显存，需要先量。
2. **NF4 格式本身**。社区独立结论支持这个——见 §6.2。
3. **我们的 bnb 调用方式**。`bitsandbytes_config.py:159` 的 `quantize_4bit()` **没传 `blocksize`**，走 bnb 默认（CUDA 上是 64，属标准值，暂无疑点）。`apply()` 里 `bnb.matmul_4bit(x_2d, layer.weight.t(), quant_state=...)` 与上游 `Linear4bit.forward` 一致。

**当时建议、现不再执行的实验**：三臂配对，同 seed 同 prompt 同 steps。A=BF16、
B=NF4 现状、C=NF4 修正。该路线已被 W8A8 主线取代。

**1 个 seed 能答什么、不能答什么**：+12% 是系统性偏移，三臂共用一个 seed 本身就是配对的，一个 seed 就能读出偏移量。**答不了的是胖片率**——那是频率问题，必须多次抽样，而且要换 seed 才能分辨"内容触发"还是"纯数值不稳"。

**另一个可疑但没查的点**：`process_weights_after_loading` 里，若 `layer.weight.device == meta`，会调 `initialize_single_dummy_weight()` 填**随机权重**然后量化。在 `--safetensors-load-strategy lazy` + `LazyWeightMixin` 的路径下有没有可能真的走到？如果走到了那是灾难性的（输出应该是噪音而不是"略微更噪"），所以大概率没走到，但**没有验证过**。

### 6.2 历史：官方 / 社区量化权重调研

MiniMax 8/3 开源当天社区就出了量化版。**我们的 NF4 是自己在线量化的**（`bnb_F.quantize_4bit()` 在加载时现场量化），不是官方权重。

- 官方 **Comfy-Org repack 提供 bf16 / INT8 / pruned INT8**，最接近官方背书
- [Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot) —— INT8 约 21 GB / 混合 INT4-INT8 约 15.5 GB / INT4 约 11.3 GB
- [lilcheaty/MiniMax-H3-NVFP4](https://huggingface.co/lilcheaty/MiniMax-H3-NVFP4) —— 作者直说 **4-bit 相比 int8_convrot 有可见画质代价，int8 在中段运动上伪影更少**。NVFP4 要 Blackwell，A100 上是模拟路径，**我们用不上**
- 目录索引：[wildminder/awesome-minimax-H3](https://github.com/wildminder/awesome-minimax-H3)
- NF4 在这个生态里基本没人用，只有一个 DiffSynth-Studio 专用的

**社区独立测出的"4-bit 掉画质、int8 明显更稳"，跟我们量到的 +12% 方向一致。**

后续已在正式 c7 中补齐 `quantization/int8_config.py` 的 `ignored_layers_match`，并使用
仓内 `quantization/tools/quantize_minimax_h3_int8.py` 按官方 scale 语义生成 serialized
W8A8 权重；正确加载与出片见 §10.3–§10.13。因此“先补匹配逻辑、再尝试 INT8”
已经完成，不是当前待办。

> **下载权重必须挂在管理节点 `root@111.172.214.42` 执行，写到 NFS。**

### 6.3 历史："只有 NF4 能做 768p 15 秒"的错误判断

BF16 与 NF4 的显存差只有 34.7–36.4 vs 37.1 GB/卡，即 1–2 GB。362 帧装不下 BF16 缺的也是这个量级。

**但省显存的抓手不是 `--dlo-resident-layers`** —— 在 h3 仓里 grep `dlo.resident|dlo_resident|resident_layers|resident-layers` **零命中**，这个 CLI 开关不存在。只有 `module_collector.py:147` 的 `pipeline._resident_modules`，是 pipeline 自己声明的属性。要动得改代码，入口在 `sequential_backend.py:368` 的 `_keep_resident_if_fits`（也就是 task **#20**）。

### 6.4 历史：NF4 宿主 AnonPages 反直觉

BF16 空载 140.5 GB < NF4 空载 160.4 GB。权重更小的反而占更多宿主内存。

我的解释：`--safetensors-load-strategy lazy` 让 BF16 走 mmap 是 **file-backed**（算 page cache 不算 AnonPages），NF4 必须把量化结果落**匿名内存**（`bitsandbytes_config.py:159` 现场生成新张量）。

**另一个没排除的可能**：bnb 的量化中间态没释放——`weight` 那份 BF16 原本要到 `replace_parameter` 之后才失去引用。**值得查，那里可能还藏着一块可回收的内存。**

（早前预测过 BF16 会比 NF4 多吃 46.5 GB 宿主内存，**这个预测是错的**，实测反过来。）

### 6.5 历史 open task 快照（不是当前待办清单）

> 这张表保留旧实验仓的问题号便于追溯。其中 INT8 识别、scale 语义、
> `ignored_layers_match`、正式仓集成等已在 §10.12 收口；当前待办以 §8 和 §10.13 为准。

| # | 内容 |
|---|---|
| 5 | 离线量化 checkpoint 被误判为 online |
| 7 | DLO 往返丢失 cutlass 列主序 layout |
| 19 | P1-1 文本编码器量化 —— 现在可以一行表达：`ignored_layers:["o_proj","down_proj"], ignored_layers_match:"substring"` |
| 20 | P1-3 常驻判定改用真实预算（含峰值激活） |
| 23 | P1-0 宿主内存超账治理：把 198 GB 常驻压到预算内 |
| 33 | 拆解 24467 MiB/卡 空载常驻显存构成（目前的估计：BF16 编码器 TP 分片 ~16 GB + VAE/CUDA ctx/NCCL ~8 GB，**未逐项验证**） |
| 36 | 三堵假墙的默认值还没改到代码里 |
| 37 | 输出不确定性 —— 已收窄成"按请求序位可复现；NF4 跨序位漂移，BF16 不漂" |
| 42 | **已做**（2026-08-09）：全局默认 `compress_statistics=false`。证伪依据补齐了——bnb 0.50.0 `functional.py:1311-1313` 在 Python 侧就把 nested absmax 还原成满尺寸 fp32 再进 kernel，不存在越界读；`sequential_backend.py:208-215` 也正确搬了 nested state，不是设备残留。真因转 #56 |
| 44 | NF4 文本编码器换入换出后失效（首请求后全 500，`v must be finite`）。定位在 `sequential_backend.py:168-215` 的 `_move_params`；bnb 的 `QuantState.to()` **就地修改并返回 `None`** |
| 47 | DiT 组 NCCL 通信子惰性创建失败。**换 tp4 之后不复现**，但根因还在。建议修法：启动时趁卡还空，用一次 dummy broadcast 把通信子预热出来（约十几行） |
| 48 | usp4 宿主内存墙 —— 与 #47 是同一件事，见 §4.1 |

---

## 7. 运维陷阱（**踩过的，会静默坑人**）

1. **`pkill -f "vllm serve"` 会自杀** —— 命令行里含这个串。用 `pgrep -f "vllm ser""ve"` 把字符串拆开。
2. **`pkill -f <script>.sh` 在 `docker exec bash -lc` 里也会自杀** —— 同理，拆串。
3. **僵尸进程骗过 `pgrep`** —— `Z <defunct>` 内存早还了但 `pgrep` 一直匹配得到，曾空等 5 分钟。判据必须是
   `ps -eo stat,comm | awk '$2 ~ /vLLM-Omni/ && $1 !~ /^Z/'`。
   0026 重启前挂过 82 个僵尸。**容器重建时该加 `--init`。**
4. **SSH 有限流** —— 连续调用间隔 ≥15 秒，否则被拒。
5. **嵌套引号里的 `awk`/`sed` 会被吃掉** —— `ssh 'docker exec c bash -lc "... $N ..."'` 里的 `$N` 会被层层展开搞坏。**用 scp 传脚本文件，不要内联。**
6. **后台启动**二选一：优先用 `docker exec -d ... sh -lc 'exec vllm ...'`；若不用
   detached exec，再使用 `setsid nohup ... < /dev/null >log 2>&1 &`。不要把两套
   后台化机制机械叠加。
7. **容器里没有 `bc`、没有 `column`**；`ffmpeg`/`ffprobe`/`python3` 四台都有。
8. **容器时钟是 UTC，宿主 mtime 是本地时间（UTC+8）** —— 对时间戳会差 8 小时。
9. `/proc/sys/vm/compact_memory` 在容器里是只读的，写会报错（无害，脚本里 `|| true` 掉）。
10. 未发布 `-p 8091:8091` 时宿主 `127.0.0.1:8091` 不通；在容器内发请求，或从
    宿主直连 `docker inspect` 得到的容器 bridge IP。

---

## 8. 建议的下一步（按优先级）

1. 将已完成 pre-barrier 480p/768p 严格 A/B、最终屏障版 5 秒 FL2VA 回归的
   output-opt 6 文件作为独立提交，构建新的通用
   不可变镜像；公共 IPC
   的 CPU tensor 分支已补“读取 SHM 前等待 D2H stream”屏障，最终镜像仍须核验
   镜像内 commit/package/import SHA 后再做 480p/768p 最小真机冒烟。
2. GPUStack 生产配置固定 `--max-num-seqs 1`、pre-VAE env、三个长超时变量，并让
   T2VA 调用方必传具名 `aspect_ratio`。
3. 用最终镜像对同一 480p/768p 工作负载连续多轮，记录 p50/p95、worker RSS、
   container memory、MemAvailable 和 swap；不再引用历史 120→160 s 慢速状态做容量基线。
4. Ref2VA 中文叠声与 FL2VA “无对白”仍有自发唇动/人声分开跟踪；在这两项通过前，
   不宣称 H3 音频质量整体验收。
5. 网关加请求防呆：校验请求时长 4–15 秒、fps=24、T2VA 具名
   `aspect_ratio`；不要拒绝由 pipeline 向上对齐的非 `17n+5` 请求帧数。
6. 把三堵假墙的默认值改进代码（**#36**）。

---

## 9. 用户的常驻要求（**务必遵守**）

- **不要下载视频到本地。**产物放 NFS，给绝对路径，用户自己去看。
- **不做衍生素材** —— 没明确要求就不要截图、不做宫格图、不做视频拼接（拼接还会丢音轨）。验收生成质量**只报指标 + 给原始文件绝对路径**。
- **模型权重下载命令挂管理节点执行，下载到 NFS。**
- 改代码可以，但**改动落在 vllm-omni，不要落在 LightX2V**。
- **四台机器都要用起来**，不要闲置；但也**不要加机器**——排队就行。
- **结合代码分析，不要瞎说。**
- **要有全面貌的规划**，不要头疼医头脚疼医脚。
- **没用的代码要回滚。**
- **只做生产档**，可以选低分辨率。
- Ref2VA 技术回归已完成；暂不继续扫 steps/seed/audio_flow_shift。中文参考音频叠声
  作为独立音频质量问题跟踪，不能用 image-only 回归替代。
- `gpustack-worker` 容器**不能停**，生产环境要用。

---

## 附：TTS 相关（同项目其他部分，顺带记录）

非声明字段的引擎参数（`emo_audio` / `emo_vector` / `emo_alpha`）**必须走 `extra_params`**，放顶层会被 Pydantic 静默丢弃。

---

## 10. 2026-08-08 续测：INT8 审计、W8A8/W8A16 与 BF16 对照

> 本节是对 §6.2 和 §8 中 INT8 建议的实际执行结果；以下结论优先于前文“INT8 尚未验证”的描述。

### 10.1 自量化 INT8 checkpoint 是否量错

结论：**没有发现 checkpoint 生成错误**。在 CPU 上对 50 个 DiT block 的 200 个量化权重逐张量重算，量化值和 scale 均逐元素完全一致：

- `q_exact=200/200`
- `scale_exact=200/200`
- 共核查 `19,267,584,000` 个元素
- 聚合相对 RMSE `0.011770`
- 饱和比例 `0.000163`，量化为零比例 `0.016821`
- `all_checks_pass=true`

审计报告：`/nfs-data/h3_logs/minimax_h3_int8_full_weight_audit_cpu.json`  
完整日志：`/nfs-data/h3_logs/minimax_h3_int8_full_weight_audit_cpu.log`

这证明“离线量化工具按预期公式生成了权重”，但不单独证明推理端的 scale 排列/分片一定正确。推理端随后另行发现并修正了 fused QKV scale 的重排问题。

### 10.2 INT8 噪声的真实根因

早期 W8A8 全噪声不是量化权重坏了，而是 serialized per-output-channel scale 被 Cutlass 路径按非 channelwise 方式解释；修正 `Int8LinearMethod` 的 `is_channelwise=True` 后恢复正常。

W8A16 首次出现棕色坍缩则是**节点代码版本不一致**：0030 缺少 `minimax_h3_transformer.py` 中 fused QKV 对应的 per-channel `weight_scale` 重排。同步下面三个文件后，W8A16 同 seed 视频恢复正常：

- `minimax_h3_transformer.py`
- `denoise_loop.py`
- `pipeline_minimax_h3.py`

这两个故障都属于推理端加载/scale 布局问题，不能反推为离线 checkpoint 量化有误。

### 10.3 W8A16 原型实现与正确性

在 `int8_config.py` 增加 `activation_scheme=weight_only`：权重沿用同一份 per-output-channel INT8 checkpoint，激活保持 BF16；当前通过已有 MoT Triton W8A16 kernel 的单逻辑 expert 桥接。`mot_gemm.py` 同时修复了 Triton 对 W8A16/W4A16 分支 constexpr 的编译问题。

真实 TP4 shape 的独立 CUDA 对照，以及实际模型 `M=6912` 的 qkv/attn-out/fc1/fc2 投影对照均通过；相对误差约 `0.18%–0.26%`。临时运行时 probe 已从正式代码删除。

本地 Mac 环境没有 PyTorch；0030 容器缺 `pytest-mock`/`pytest-asyncio`，所以完整 pytest 未能启动。清理后的文件已通过容器内 `py_compile` 和配置导入检查；此前独立 CUDA kernel 测试及完整端到端生成已通过。

### 10.4 同 prompt/seed 的正式 209 帧、20 steps 对照

三个请求均为 832×480、209 帧、seed 11、同一复杂真实世界 prompt。接口传 20 steps，当前 tqdm 显示 19 次实际 denoise iteration。

| 模式 | 请求总耗时 | denoise | 稳态约每次 denoise | 峰值显存（GPU0/其余） |
|---|---:|---:|---:|---:|
| BF16 | 130.55 s | 118 s | 6.22 s | 29663 / 29237 MiB |
| W8A8 | 161.88 s | 141 s | 7.47 s | 25287 / 24861 MiB |
| W8A16（当前原型） | 211.78 s | 189 s | 9.99 s | 25267 / 24841 MiB |

短请求中记录的 `BF16 24.12 s / W8A8 27.69 s / W8A16 52.05 s` 是**整个 49 帧、8 steps HTTP 请求**，包含文本编码、denoise、VAE/audio decode 和封装，不是单步耗时。若简单除以 8，只能得到端到端摊销值，不能当作纯 denoise latency。

正式视频：

- BF16：`/nfs-data/h3_samples/bf16_clean_final_realworld_complex_832x480_209f_seed11.mp4`
- W8A8：`/nfs-data/h3_samples/int8_w8a8_clean_final_realworld_complex_832x480_209f_seed11.mp4`
- W8A16：`/nfs-data/h3_samples/int8_w8a16_clean_final_realworld_complex_832x480_209f_seed11.mp4`

### 10.5 画面、运动和音频指标

以 BF16 为像素参考（注意生成轨迹轻微漂移会显著降低 SSIM/PSNR，不等价于主观画质）：

| 样本 | W8A8 vs BF16 | W8A16 vs BF16 |
|---|---:|---:|
| 49 帧 / 8 steps SSIM | 0.907837 | **0.925332** |
| 49 帧 / 8 steps PSNR | 29.4706 dB | **30.0291 dB** |
| 209 帧 / 20 steps SSIM | **0.844707** | 0.832094 |
| 209 帧 / 20 steps PSNR | **26.5236 dB** | 25.3485 dB |

W8A16 在短片更接近 BF16，长片则 W8A8 更接近，不能据单 seed 宣称 W8A16 画质稳定优于 W8A8。用户肉眼确认修正后的 W8A16 正常，语音没有此前 NF4 的明显错误。

相邻帧平均亮度绝对差：BF16 `1.347225`、W8A8 `1.357491`、W8A16 `1.355894`，三者运动量接近。音频 RMS：BF16 `0.188519`、W8A8 `0.182397`、W8A16 `0.186225`；W8A16 更接近 BF16。W8A8/W8A16 仅有约 `0.00000358` 的样本达到 `|x|>=0.999`，可忽略。

### 10.6 资源占用

| 模式 | 四卡平均 GPU util | 四卡平均功耗 | 容器内存峰值 |
|---|---|---|---:|
| BF16 | 94.08/92.91/93.01/90.75% | 176.77/178.56/181.88/183.09 W | 162.5 GiB |
| W8A8 | 89.15/88.26/87.84/88.18% | 142.50/146.09/140.53/144.16 W | 184.1 GiB |
| W8A16 | 90.97/91.57/90.40/91.71% | 158.32/156.70/165.84/160.27 W | 214.6 GiB |

W8A8 与 W8A16 的权重文件大小相同，GPU 显存也几乎相同；相对 BF16 每卡约省 `4.4 GiB`。容器内存包含 CPU offload 权重、匿名页和 page cache，不等于模型净权重；当前 W8A16 没有宿主内存优势。

原始资源采样前缀：

- `/nfs-data/h3_metrics/bf16_clean_final_0025`
- `/nfs-data/h3_metrics/int8_w8a8_clean_final_0024`
- `/nfs-data/h3_metrics/int8_w8a16_clean_final_0030`

### 10.7 W8A8 性能复测异常

同一 W8A8 输出曾在旧服务中用 `120.806837 s` 生成，后来旧实验仓 clean restart
为 `161.875681 s`；两份文件 SHA-256 完全相同（`a0dcf9f5...15464a9`），实际都是
19 次 denoise，因此不是 steps 或生成路径变化。独占复测仍为 `159.199657 s`，说明
该旧运行时的慢速状态可稳定复现。

- 旧进程 denoise：约 `5.41 s/it`，峰值显存 `27767/27341 MiB`
- 旧实验仓 clean restart/复测：约 `7.20 s/it`，峰值显存 `25287/24841 MiB`
- 复测 SM 时钟平均约 `1308 MHz`、最高 `1410 MHz`，GPU 平均功耗约 140W；宿主 `kcompactd` 活跃

更低显存、更低功耗而速度变慢，线索更偏向 CPU offload/内存布局或运行时驻留状态，
而不是 INT8 数值路径变化，但根因至今未隔离。这是历史旧实验仓/旧运行时的基线；
120.81 s 和约 160 s 都不应继续用于当前生产容量规划。当前容量数据必须来自最终
不可变镜像、同一工作负载的多次热态 p50/p95。

### 10.8 当前生产建议

1. **继续以 vLLM-Omni + TP4 为主线。**四卡都实际参与计算；DiffSynth 单卡的结论不适用于这条链路。
2. **现阶段默认选 W8A8。**它的端到端画质已由用户确认正常，省约 4.4 GiB/卡；当前虽不比 BF16 快，但明显快于 W8A16 原型。
3. **W8A16 暂不作为生产默认。**它当前的实际优势只有 INT8 权重/显存占用和接近 BF16 的数值特性；磁盘/显存不比 W8A8 更省，且冷启动编译和稳态生成最慢。
4. W8A16 慢不是 W8A16 理论上的必然结论：当前实现每个 Linear 都创建 routing indices，并绕过一个通用双 expert MoT kernel，尚无 A100 专用 autotune。只有完成 routing-free 专用 GEMM 和 A100 profile 后，才值得重新判断其生产价值。
5. 若机器显存足够且优先追求最少量化风险，BF16 是数值基准最清楚的选项；它在
   480p/短请求数据中最快，但本轮 768p 长序列 W8A8 总耗时约快 1.7%，不能笼统
   宣称任一精度档在所有负载下最快，也不能宣称“量化必然加速”。

### 10.9 768p、15 秒上限实测（请求 360 帧，实际输出 362 帧）

> 本节更新了文档开头“768p 稳定档只有 10 秒”的旧摘要。旧上限的根因是 VAE 整片 float32 后处理缓冲；§3.1(b)/§4.4 的 16 帧分块修复已经消除了该限制。本轮又分别用 BF16、W8A8、W8A16 完整跑到 MP4 返回，三种模式均 HTTP 200。

统一参数：1344×768、24fps、请求 360 帧/15 秒、pipeline 对齐后实际输出 362 帧
（封装约 15.107 秒）、`num_inference_steps=20`、`flow_shift=12`、seed 23、同一个
复杂真实世界 prompt。调度器有 20 个 sigma/timestep 节点，实际执行 19 次 denoise
状态更新，所以 tqdm 为 19/19。

| 模式 | HTTP 总耗时 | engine 总耗时 | tqdm 去噪 | 稳态约每步 | engine 内非去噪 | MP4 编码 | 四卡峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 881.26 s | 873.84 s | 839 s | 43.71 s | 34.84 s | 7.38 s | 37703 / 37281 / 37281 / 37281 MiB |
| W8A8 | **865.90 s** | **853.86 s** | **791 s** | **41.25 s** | 62.86 s | 11.77 s | 38517 / 38517 / 38517 / 38517 MiB |
| W8A16 | 1284.01 s | 1255.64 s | 1165 s | 61.15 s | 90.64 s | 28.16 s | **33167 / 32745 / 32745 / 32745 MiB** |

`engine 内非去噪` 包含文本编码、latent 准备、VAE 视频/音频解码及后处理；现有日志没有给这些子阶段分别打点。根据逐秒显存/利用率轨迹，文本准备/编码约为 BF16 15 秒、W8A8 19 秒、W8A16 11 秒，余下约 20/44/80 秒主要是 VAE/audio/postprocess；这是轨迹估算，不应冒充精确 profiler 数据。去噪、engine、MP4 和 HTTP 总时间均为日志/接口精确值。

资源统计（覆盖整个请求的逐秒 GPU 采样；宿主统计从请求中段开始并按请求时间窗截取）：

| 模式 | 四卡平均 GPU util | 四卡平均功耗 | 容器内存平均/峰值 | 宿主可用内存最低 |
|---|---|---|---:|---:|
| BF16 | 96.76/96.34/96.30/96.61% | 193.42/191.25/199.68/200.68 W | 163.54 / 191.4 GiB | 77.78 GiB |
| W8A8 | 92.41/92.54/92.39/92.44% | 187.71/192.19/190.19/191.93 W | 208.31 / 220.2 GiB | 69.99 GiB |
| W8A16 | 91.87/91.85/91.80/91.94% | 163.64/167.49/173.91/173.46 W | 214.36 / 226.9 GiB | 55.27 GiB |

关键结论与 480p 不同：

1. **W8A8 在 768p 长序列下去噪略快于 BF16**，总时间快约 1.7%，但动态激活量化/Cutlass workspace 使峰值显存反而升到 38.5 GiB/卡，只剩约 2.4 GiB 的 `nvidia-smi` 余量，不能把“INT8 权重更小”直接等同于“请求峰值更低”。
2. **W8A16 才真正给 768p 留出明显 GPU 余量**（约 7.8 GiB/卡），代价是总耗时比 BF16 高约 45.7%，且宿主内存压力最大。当前 correctness-first MoT bridge 仍不适合默认生产。
3. BF16 也能完整跑通，且只比 W8A8 慢约 15 秒；如果单请求、无并发并优先质量，BF16 是更稳妥的默认值。若要给其他进程/并发留 GPU 余量，现有实现只有 W8A16 明显做到，但性能代价很大。
4. W8A8 首次在 14/19 被 HTTP 504 中止，是新启动命令漏配 `VLLM_OMNI_VIDEO_SYNC_TIMEOUT`、命中默认 600 秒，不是 OOM。补为 7200 秒后同参数完整成功；canonical 配置应继续使用 14400 秒。

产物：

- BF16：`/nfs-data/h3_samples/bf16_15s_768p_realworld_complex_1344x768_362f_s20_seed23.mp4`
- W8A8：`/nfs-data/h3_samples/int8_w8a8_15s_768p_retry_realworld_complex_1344x768_362f_s20_seed23.mp4`
- W8A16：`/nfs-data/h3_samples/int8_w8a16_15s_768p_realworld_complex_1344x768_362f_s20_seed23.mp4`

三份文件均为 1344×768、24fps、362 个可解码视频帧、15.107 秒。相对 BF16 的像素指标为：W8A8 SSIM `0.576712` / PSNR `17.146974 dB`，W8A16 SSIM `0.575851` / PSNR `17.246993 dB`。长时高分辨率生成的轨迹漂移很大，这些数值只能说明两种 INT8 与 BF16 的逐像素偏离量接近，不能代替主观画质判断。音频 mean/max volume：BF16 `-26.4/-1.6 dB`、W8A8 `-23.5/-0.8 dB`、W8A16 `-23.6/-0.2 dB`。

### 10.10 生产优化候选复测（regional 对照仍有混杂变量）

本轮在 1344×768、请求 360 帧/实际输出 362 帧、20 sigma 节点、seed 23 的同一
工作负载上继续复测，并把肉眼对比入口整理到：

`/nfs-data/h3_samples/production_optimization_compare_20260808`

目录中的文件是指向原始 MP4 的符号链接，`README.md` 记录了统一参数、阶段耗时、资源占用与正确性结论。

| 模式 | HTTP 总耗时 | engine 总耗时 | 去噪 | 稳态每次去噪 | 四卡峰值显存 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| BF16 TP4 原始 | 881.26 s | 873.84 s | 839.0 s | 43.71 s | 37703/37281/37281/37281 MiB | 基线 |
| BF16 TP4 + pre-VAE offload | 865.46 s | — | 828.5 s | — | 见说明 | 与原片 SHA-256 完全一致 |
| W8A8 TP4 原始 | 865.90 s | 853.86 s | 791.0 s | 41.25 s | 38517/38517/38517/38517 MiB | 原始 INT8 基线 |
| W8A8 TP4 + regional compile + pre-VAE offload | **856.11 s** | **834.96 s** | **757.64 s** | **39.56 s** | 37073/36649/36649/36649 MiB | 观察到的最快候选，非独立 regional A/B |
| W8A16 TP4 原始 | 1284.01 s | 1255.64 s | 1165.0 s | 61.15 s | 33167/32745/32745/32745 MiB | 旧 correctness-first kernel |
| W8A16 TP4 + A100 tuning | 965.22 s | 955.30 s | 889.5 s | 46.53 s | 38947/38523/38523/38523 MiB | 快约 24.8%，内容逐像素/逐 PCM 相同 |

这组数据中 regional compile 和 pre-VAE offload 同时变化，且原始运行是否处于同编译
缓存/同宿主状态未完全隔离。因此只能说 **W8A8 + TP4 + regional dynamic compile +
pre-VAE DiT offload 是当时观察到的最快候选**；HTTP -1.1%、engine -2.2%、去噪
-4.2% 不应当作 regional 的独立因果收益。要确认 regional 本身的价值，仍需在同代码、
同输出路径、同内容和同编译缓存状态下对 eager/无 compile 与 regional 做多次热态 A/B。
regional compile 可改变浮点执行轨迹；在设为生产默认前仍需肉眼验收。

W8A16 的 A100 profile 则是数值不变的性能优化：优化前后解码视频 SSIM 为 `1.000000`，音频 PCM MD5 相同。它仍慢于 W8A8，不作为默认生产档。

同一个热 W8A8 TP4 实例先跑完 768p×15 秒，再跑 832×480、请求 360 帧/实际
输出 362 帧也成功；
后者 HTTP `225.16 s`、engine `221.97 s`、去噪 `200.22 s`。因此 GPUStack 已有队列的
前提下，无需为 480p/768p 拆成两个实例；但这只证明“同实例串行混跑”。
H3 `forward` 当前要求单 prompt，没有请求合批/同实例并发的生产证据，部署仍应显式
`--max-num-seqs 1`，由 GPUStack 在实例外排队。

`TP2+SP2` 当前不作为生产候选：短请求比 TP4 更慢且显存更高。事后核对 NFS 日志发现
实际有四次 768p×15 秒 HTTP 500，不是文档原写的三次：前三次分别在 video embedding
（申请 1.08 GiB）和 Ulysses all-to-all contiguous（申请 374 MiB）报明确的 PyTorch tensor
OOM；第四次走完 block stack 后，在最终 SP gather 首次创建/使用独立 NCCL communicator
时报 `ncclUnhandledCudaError`，当时四卡已达 40441/40441/40381/40381 MiB。因此“显存
余量极差”的结论站得住，但第四次属于显存顶满时的 NCCL 通信组懒初始化失败，与
#47 是同类机制而非同一 group/callsite。由于尚未做 SP group 预热复验，不应把“预热后也
必然失败”写成已证实事实。相关失败实验代码已从本地实验仓回滚。

pre-VAE offload 在 VAE 解码前把已不用的 DiT 参数移回 CPU，实测约 0.12–0.16 秒，可释放每卡约 12–26 GiB 的实际 allocated 参数内存。W8A16 首版因用“第一个参数是否在 GPU”作为判断而漏触发，已改为检查任一 accelerator-resident 参数。

### 10.11 正式仓与 H3 实验仓的关系及出包约束

正式出包仓必须是：

`/Users/reputationly/Desktop/code/api/vllm-omni`

实验仓：

`/Users/reputationly/Desktop/code/api/vllm-omni-h3`

两者不是可直接覆盖的副本：

- 正式仓 `main@5724f72f` 从共同祖先 `62589203` 之后有 22 个自有提交，包含异步音视频 API、MOSS 修复、HunyuanImage-3 A100/NF4、Docker/CI 等生产定制。
- 实验仓固定在 `900a7f08`（`v0.26.0-3`），从共同祖先之后包含 214 个上游提交，才有 MiniMax-H3 的正式模型实现；实验仓另有约 19 个已跟踪文件和两个未跟踪路径的本地 H3 改动。
- 两棵已提交源码树相差 1267 个文件，约 147744 行新增、42917 行删除；不能只复制 `minimax_h3/`，因为 H3 依赖同期的 loader、offloader、量化、CLI 和 diffusion engine 接口。
- 对 `5724f72f -> 900a7f08` 做无落盘 merge 预演，真正的文本冲突只有 `async_omni.py`、`openai/api_server.py`、`processors/ming.py` 三个；其余 11 个双方都改过的路径可自动合并。但正式仓当前还有两处未提交内存优化，升级前必须单独保护和重放。
- 当前 `upstream/main@81b48e83` 又比实验基线多 52 个提交，包含 H3 权重完整性检查、帧转换内存上限、fused RMSNorm/RoPE、modular pipeline 等。它更适合作为后续生产候选基线，但没有经过本轮四机回归，不能直接把实验数据视为对最新上游的验证结果。

推荐集成顺序：先从正式仓建集成分支，将正式仓升级到一个固定上游 H3 基线并解决 3–4 个冲突；随后按功能重放 H3 正确性、W8A8/W8A16、pre-VAE offload、A100 kernel profile、监控与测试，逐项去掉已被最新上游吸收的补丁；最后在正式仓构建镜像，做 480p/768p、BF16/W8A8、异步视频 API、既有音频/MOSS/Hunyuan 回归。不要从实验仓直接出包，也不要把实验仓 2000 余行 WIP 一次性覆盖进正式仓。

### 10.12 最新上游集成与真实 W8A8 TP4 回归

已按 §10.11 的路线完成集成，而不是从实验仓覆盖正式仓：

- 上游基线固定为 `upstream/main@81b48e83`，合并提交 `f14e4a4f`。
- H3/A100 生产定制独立提交为 `c7e56d68`，便于单独审计或回退。
- 最新上游已吸收 modular H3 pipeline、严格缺权重检查、fused RMSNorm/RoPE、视频逐帧有界转换、量化敏感层排除等能力；旧实验仓的 NF4 probe、TP2+SP2 embedding 试验、旧 DLO mmap 补丁均未迁移。
- 迁移内容主要是 serialized INT8/W8A8 正确 scale 语义、可选 W8A16、QKV scale 同步重排、A100 kernel profile、VAE 有界分块、pre-VAE DiT offload、CPU-home 复用、跨线程 NumPy buffer re-own 以及相应测试。
- 静态/pre-commit 全通过；组合回归 `165 passed, 6 skipped, 4 deselected`，offloader GPU 测试 `6 passed`，真实 CUDA INT8 kernel 冒烟 `4 passed`。唯一未运行的完整 H100 FP8 质量测试需要外网 checkpoint，与本次 A100/INT8 路径无关。

真实服务使用 `/nfs-data/models/MiniMax-H3-FL2VA-INT8`、TP4、text-encoder TP4、VAE patch parallel 4、regional dynamic compile、model CPU offload 和 pre-VAE DiT offload。最新上游原先只认目录名恰好为 `FL2VA`/`Ref2VA` 的本地分区；量化目录会被误拼成 `...INT8/FL2VA/model_index.json`。现已改为读取 `model_index.json` 的 `_minimax_h3.partition` 元数据，重命名后的官方分区布局可直接启动，并有回归测试。

启动与 832×480、209 帧、20 sigma 节点、seed 11、复杂真实世界 prompt 的结果：

| 项目 | 首次请求 | 同实例第二次热请求 |
|---|---:|---:|
| 服务冷启动 | 169.34 s | — |
| HTTP 总耗时 | 140.75 s | 109.16 s |
| pipeline forward | 136.47 s | 105.51 s |
| 文本编码 | 1.21 s | 3.93 s |
| diffuse | 129.21 s（含首次 compile） | 96.82 s |
| VAE/audio decode | 5.96 s | 4.73 s |
| MP4 封装 | 1.46 s | 1.75 s |
| 四卡峰值 | 25951 MiB/卡 | 25939 MiB/卡 |
| 主机 available 最低 | 106.52 GiB | 104.46 GiB |

CPU-home 优化在本轮 offline W8A8 + lazy load + CPU offload 路径中记录为每 rank
编码器 `home-hit=12.82GB`、`shadow fresh=0.00GB`；第二次请求仍为
`shadow fresh=0.00GB`，宿主内存没有出现旧实现每请求新增约 50 GiB pinned
landing buffer 的阶梯式增长。这个快路径依赖参数由 CPU 加载、swap-in 经
`_move_params` 保留 CPU home、推理期权重只读，且未开
`VLLM_OMNI_OFFLOAD_COPY_BACK=1`；其他加载策略可能回退到 shadow copy，不得泛化为
所有 BF16/量化配置的普遍保证。生产验收可暂时开启 `VLLM_OMNI_OFFLOAD_DEBUG=1`，
确认每 rank `home-hit>0` 且 `shadow fresh=0`。pre-VAE offload 用约 0.11–0.13 秒
释放每卡约 14.3 GiB，随后两次请求均完整返回 HTTP 200。

产物：

- 首次（含 compile）：`/nfs-data/h3_samples/latest_upstream_w8a8_tp4_realworld_complex_832x480_209f_seed11.mp4`
- 第二次热态：`/nfs-data/h3_samples/latest_upstream_w8a8_tp4_warm2_realworld_complex_832x480_209f_seed11.mp4`
- 服务日志：`/nfs-data/h3_logs/latest_upstream_int8_w8a8_tp4_smoke_0026_20260808.log`

两份产物均为 832×480、209 帧、24 fps、8.732 秒，H.264 + 32 kHz 双声道 AAC，视频/音频流均全量解码通过。两次解码视频逐帧一致（SSIM `1.000000`）；AAC 解码 PCM MD5 不同，说明当前音频运行间不是 bit-exact。该现象不影响本次视频正确性和资源稳定性结论，但若 GPUStack 生产要求“同 seed 音频逐样本确定性”，需要单独定位 audio latent/解码/编码链路。

最新上游请求契约还要求 T2VA 显式传 `aspect_ratio`，即使同时传了 `width`/`height`；旧脚本必须补 `-F 'aspect_ratio=16:9'`，否则会在推理前返回 `t2va requires an explicit aspect_ratio`。

### 10.13 通用 c7 镜像四机回归与 Ref2VA 音频遗留（2026-08-09）

本轮统一使用通用多模型镜像（不是 H3 专用镜像）：

`crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-omni:arm64-a100-20260809-c7e56d68`

镜像 ID / registry digest 均为
`sha256:f62c4d4df44ad55ce032317cb3b1903467c5bd0801cb865b75a1fee8aee4bafc`；镜像内
HEAD 为 `c7e56d6873b616546d44c451925c71fed323abf7`，包版本为
`0.26.1.dev79+gc7e56d687`，A100 `sm_80` 可用。四台只重建实验容器 `voh3`，均未操作
`gpustack-worker`。

该 digest 只对 clean c7 基线负责。本节的 output-opt 结果是在同一 c7 容器环境中，
通过宿主挂载 `/work/vllm-omni-output-opt` 与 `PYTHONPATH` 运行的源码验证，并通过
进程 cwd、模块 origin 和文件 SHA 核验。它尚未固化进 `f62c...` 镜像；生产发布前
必须构建新的通用不可变镜像，再重复 import SHA 和最小真机回归。

#### W8A8 TP4、15 秒正式请求

两档均使用 20 个 sigma 节点、实际 19 次 denoise、`flow_shift=12`、
`audio_flow_shift=3`、seed 11、regional dynamic compile、CPU offload 与 pre-VAE DiT
offload：

| 画布/请求 | HTTP | engine E2E | diffuse | decode 关键路径 | MP4 | 峰值显存/卡 | 最低 MemAvailable |
|---|---:|---:|---:|---:|---:|---:|---:|
| 832×480 首次 | 275.61 s | 269.56 s | 235.15 s | 12.58 s | 5.82 s | 28191 MiB | 102.14 GiB |
| 832×480 热态 | 234.83 s | 231.43 s | 195.38 s | 8.43 s | 3.24 s | 28181 MiB | 98.55 GiB |
| 1344×768 首次 | 971.51 s | 960.55 s | 810.08 s | 54.96 s | 10.44 s | 36655 MiB | 79.36 GiB |
| 1344×768 热态 | 824.18 s | 818.59 s | 763.08 s | 36.53 s | 5.52 s | 36647 MiB | 61.76 GiB |

480p 首次 regional compile 的第一、第二次迭代分别约 45.36 s、18 s，之后约
10.1 s/步；768p 首次迭代约 84.02 s，之后稳定约 39.9–40.0 s/步。pre-VAE offload
分别约 0.14 s / 0.13–0.15 s，释放约 16.50 / 24.74 GiB 每卡。两档四份输出均为实际
362 帧、约 15.08 s 视频和 15.11 s AAC，音视频全流解码通过。

API 契约要提交 `duration=15.0`（或 `num_frames=360`），pipeline 再按 `17n+5`
对齐为 362 帧；直接把 `num_frames=362` 作为 24 fps 请求会因 15.083 秒超过
15 秒上限而被拒绝。

产物：

- 480p：`/nfs-data/h3_runs/c7e56d68_0024_480p15s_20260809/c7e56d68_generic_w8a8_tp4_480p15s_seed11_0024.mp4`
- 480p 热态：`/nfs-data/h3_runs/c7e56d68_0024_480p15s_20260809/c7e56d68_generic_w8a8_tp4_480p15s_seed11_0024_warm2.mp4`
- 768p：`/nfs-data/h3_samples/latest_c7e56d68_w8a8_tp4_768p_15s_seed11_0025_realworld_complex_1344x768_362f_s20_seed11.mp4`
- 768p 热态：`/nfs-data/h3_samples/latest_c7e56d68_w8a8_tp4_768p_15s_seed11_0025_warm2_realworld_complex_1344x768_362f_s20_seed11.mp4`

768p 首次请求比 §10.10 的 856.11 s 慢：除首次 compile 外，本轮还观察到约 88.01 s
的 worker output / NumPy 重归属到 engine 完成区间。同一实例热态复测后该区间降到约
9.87 s，该现象与大块 CPU/SHM 内存首次分配与缺页的一次性成本一致，但未通过
独立缺页计数/多次 clean restart 证明完整因果。可以确定它不是每个热请求的固定开销。
热态总时间 824.18 s，比首次快 15.17%，比 §10.10 旧优化数据
856.11 s 快 3.73%；diffuse 为 763.08 s，与旧数据 757.64 s 只差 0.72%。首次与热态
解码视频的 framemd5 完全一致；音频 PCM 仍非 bit-exact，这是已知运行间非确定性。

热态请求自带的留存采样显示 `MemAvailable` 最低为 61.755 GiB。现场还观察到
direct reclaim/kswapd 扫描，但该段独立 host CSV 未保存，因此仅作现场观察。容器
内存由 184.1 GiB 起步、194.3 GiB 峰值回落到 172.0 GiB，没有请求间阶梯式泄漏，
但 4.176 GiB 的 768p float32 视频在 worker、pinned staging、SHM 和 engine 后处理之间的
多次复制仍值得优化。

#### 输出路径内存优化与 480p 严格 A/B

本轮新增四个不改对外 float32 输出契约的修正：

1. CPU tensor 进入 SHM 时跳过多余的 pinned CPU staging，但在读取其 storage 前
   等待 D2H stream，避免异步 producer 尚未完成时把旧/半写数据送进 SHM；
2. H3 后处理按 16 帧 clamp 并写入 NumPy-owned NTHWC 最终缓冲，避免整片
   clamp 临时量和后续 re-own 整片复制；
3. `num_outputs_per_prompt=1` 时不再执行 `torch.cat([video])`，768p 可少拷贝
   4.176 GiB/rank；
4. `unique_reply_rank` 之外的 rank 执行完 collective/forward 后立即丢弃不可见的
   CPU 结果，不再让上一请求的整片视频贯穿下一次 forward。

单测对 NaN/inf/-0.0 做了逐 bit oracle 对照，并检查 NumPy ownership、CPU SHM
FP32/BF16 和 non-reply 结果生命周期。最终 IPC 屏障版本在 c7 容器环境完成
`144 passed, 2 deselected`，另直接执行新版 FP32/BF16 IPC oracle 两项均通过；终审
又把 H3 postprocess 扩成 FP32/BF16 × non-contiguous 输入，两项均通过。后一组有一项
替代 144-suite 中的旧单 case，因此最终改动共覆盖 147 个不同的同步检查；两个
deselected 是运行镜像未安装 pytest-asyncio 的异步用例。
运行镜像也缺 pytest-mock，且 0030 overlay 的 `test_ipc_async.py` 仍是旧断言；打包前
需同步本地新版测试并在完整 dev-test 环境重跑三个文件。六个本地改动文件的
pre-commit 全通过。

832×480、请求 360 帧/实际输出 362 帧，使用 clean c7 和 output-opt 分别重启服务，每个服务各跑 warmup+
warm2。对应序号的 clean/output-opt 完整 MP4 SHA-256 均完全一致，解码视频
PSNR=`inf`、SSIM=`1.0`，因此优化没有改变视频、音频或封装字节。严格同内容
热态 A/B：

| 指标 | clean c7 | output-opt | 变化 |
|---|---:|---:|---:|
| HTTP | 225.305 s | 216.563 s | -8.742 s / -3.88% |
| engine E2E | 219.540 s | 212.800 s | -6.740 s |
| max-forward 结束到 engine E2E | 9.896 s | 4.283 s | -5.613 s |
| worker RSS 峰值合计 | 138.698 GiB | 131.896 GiB | -6.802 GiB |
| `voh3` 进程 VmRSS 求和峰值 | 143.353 GiB | 134.900 GiB | -8.453 GiB |
| 请求期间 MemAvailable 下降 | 6.562 GiB | 2.033 GiB | -4.529 GiB |
| 四卡峰值显存 | 28159/28179/28179/28179 MiB | 28155 MiB/卡 | 基本不变 |

旧 generic warm2 与新 clean/output-opt 的内容不同（PSNR 24.608 dB、SSIM 0.7853），
不能拿 234.829 s 作严格性能 A/B。受控 clean/output-opt 对应产物逐字节一致，
证明输出补丁没有改变该受控样本的数值路径；旧实例差异仍可能来自环境、import、
请求序位或编译状态等未记录变量，原因未定，不能据此泛化为 H3 固有跨启动非确定。

output-opt 480p 产物：
`/nfs-data/h3_runs/c7e56d68_0024_output_opt_480p15s_20260809/c7e56d68_output_opt_w8a8_tp4_480p15s_seed11_0024_warm2.mp4`。
clean 同内容对照：
`/nfs-data/h3_runs/c7e56d68_0024_clean_restart_control_480p15s_20260809/c7e56d68_clean_restart_control_w8a8_tp4_480p15s_seed11_0024_warm2.mp4`。

这组 480p 严格 A/B 与下方 768p 一样完成于公共 IPC 读取屏障加入之前；它严格证明
其余三个输出优化的 bitwise 正确性和收益，但 216.563 s 只能作为最终候选估计，不能
冒充最终 6 文件状态的 15 秒实测值。最终屏障版当前只完成 5 秒 FL2VA 回归。

#### 768p 严格 A/B

1344×768、请求 360 帧/实际输出 362 帧在 0025 上也完成 clean c7 与 output-opt
各自重启、各跑 warmup+warm2 的严格控制。对应序号的完整 MP4、解码视频帧和
解码音频 PCM 均逐字节一致；去噪耗时基本不变，收益来自目标输出路径：

| 指标 | output-opt warmup | clean warmup | output-opt warm2 | clean warm2 |
|---|---:|---:|---:|---:|
| HTTP | 854.071 s | 1026.426 s | 846.756 s | 879.137 s |
| diffuse | 771.229 s | 774.582 s | 762.014 s | 763.564 s |
| forward 结束到 engine E2E | 21.139 s | 169.526 s | 16.164 s | 51.630 s |
| MP4 编码 | 10.635 s | 31.294 s | 9.083 s | 30.907 s |
| sampler 最低 MemAvailable | 99.285 GiB | 78.880 GiB | 91.732 GiB | 71.108 GiB |

- warmup 总耗时减少 172.355 s / 16.79%；热态减少 32.382 s / 3.68%。
- clean warmup worker RSS 从 123.337 增到 150.062 GiB，保留 26.725 GiB；第二轮
  又增加 0.665 GiB。output-opt warm2 仅从 130.281 增到 130.450 GiB（+0.170 GiB）。
- output-opt 将 warmup/warm2 的最低 MemAvailable 分别提高 20.405/20.624 GiB；
  四卡峰值显存仍约 36617 MiB/卡，GPU 路径基本不变。
- warmup pair 完整 MP4 SHA-256 均为 `91e4a476...`，warm2 pair 均为
  `9446f698...`。两组全流解码均通过。

严格 A/B 使用的是补公共 IPC 读取屏障之前的 output-opt；该屏障是随后发现的
正确性修复。最终屏障版已在 FL2VA 5 秒回归中验证与 clean oracle 逐字节相同，且
forward→engine gap 为 3.932 s，未观察到屏障带来的尾部延迟回退。因此这里的
3.68% 热态和约 20.6 GiB MemAvailable 收益可作为生产候选估计，但最终不可变镜像
仍需按 §8 做一次最小 480p/768p 冒烟。

产物：

- output-opt warm2：`/nfs-data/h3_samples/outputopt_full_w8a8_tp4_768p_15s_seed11_0025_warm2_realworld_complex_1344x768_362f_s20_seed11.mp4`
- clean warm2：`/nfs-data/h3_samples/clean_c7_w8a8_tp4_768p_15s_seed11_0025_warm2_realworld_complex_1344x768_362f_s20_seed11.mp4`
- 审计汇总：`/Users/reputationly/Desktop/code/api/h3_audit/outputopt_vs_clean_c7_tp4_768p_15s_0025_SUMMARY.md`

#### FL2VA 首帧回归

W8A8 TP4 使用同一 832×480 咖啡店首帧、5 s、20 sigma 节点、seed 43030 完成
FL2VA 回归。首次请求因 login shell 重置工作目录和 `PYTHONPATH`，实际运行的是
镜像内 clean c7，不能误标为 output-opt。重启后已通过主进程 cwd、环境变量、模块
origin 和三个文件 SHA 二次确认真正命中 output-opt。

| 指标 | clean c7 首请求 | 未加 IPC 屏障的 output-opt | 最终 IPC 屏障版 |
|---|---:|---:|---:|
| HTTP | 137.155 s | 86.988 s | 131.866 s |
| engine E2E | 135.425 s | 85.386 s | 130.327 s |
| diffuse | 94.448 s | 71.991 s | 99.609 s |
| decode max rank | 33.030 s | 4.405 s | 4.677 s |
| forward 结束到 engine E2E | 3.229 s | 4.860 s | 3.932 s |
| MP4 编码 | 1.603 s | 1.438 s | 1.446 s |

三次首请求的 encode/diffuse 冷态不同，总 E2E 和 decode 差额都不能归因给输出补丁；
single-cat、owned buffer 和 non-reply drop 也发生在 `decode()` profiler 区间之后。
这组数据只用于功能回归与 IPC 正确性，性能因果以 480p/768p 同机受控 A/B 为准。

最终 IPC 屏障版的完整 MP4 与 clean c7 逐字节相同，解码视频和音频 framemd5 manifest
也分别相同；未加屏障的 output-opt 文件不同。结合“CPU tensor 读取前只增加一次
stream wait”这一隔离变量，这强力支持旧差异来自异步 producer 完成前读取的竞态。
新增等待没有观察到输出尾部惩罚：最终 `engine E2E - max forward=3.932 s`，相对
未加屏障版 4.860 s 没有回退。

三份都复现同一个语义问题：提示要求“无对白/无唇动/无人声”，人物仍有连续唇形变化，
音轨 0.7–1.5 s 仍有明显谐波有声段。这是 FL2VA 无对白遵从问题，不是 Ref2VA
音频参考叠声问题。最终 IPC 产物：
`/nfs-data/h3_runs/c7e56d68_0030_output_opt_final_ipc_fl2va_5s_20260809/fl2va_0030_output_opt_final_ipc_first_frame_5s_832x480_s20_seed43030.mp4`。

#### Ref2VA 技术回归

BF16 TP4 eager 的图像+音频参考成功返回：832×480、124 帧、5.17 s、HTTP
164.45 s、diffuse 141.10 s、峰值约 29.82 GiB/卡。必须注意 Ref2VA 的独立音频是
**音色/说话风格参考**，不是要逐字复制或与目标画面对齐的现成对白；目标台词应在
prompt 中显式写为 `<d>[Chinese] ...</d>`。此前把 5.47 s 参考音频描述成“完整复述”并
强制生成 8 s 的样本属于错误用法，不能用于判断 Ref2VA 语音质量。

修正样本显式指定台词“大家好，欢迎来到我们的咖啡馆。今天的天气很好。”，Whisper
可完整转写，语速约 3.85 字/s；但用户听感确认“大家好，欢迎来到我们的咖啡馆”一段
仍有重音/双人叠声。拆出的左声道、右声道、左右平均单声道三份都有叠声，因此已排除
AAC 封装、左右声道延迟或简单立体声叠加。左右声道测得零延迟、相关系数 0.9875、
side/mid RMS 约 -22.0 dB，问题已经存在于单声道内部的模型生成结果。按用户要求先记录，
暂不继续扫 steps、seed 或 `audio_flow_shift`；生产质量验收暂标为：**接口/画面/可解码性
通过，中文语音清晰度未通过**。

相关文件：

- 修正视频：`/nfs-data/h3_samples/latest_image_c7e56d68_20260809/ref2va_0026_image_audio_corrected_5s_832x480_s20_seed42026.mp4`
- 原始音频：`/nfs-data/h3_samples/latest_image_c7e56d68_20260809/ref2va_0026_source_jiayan_zh.wav`
- 生成音频：`/nfs-data/h3_samples/latest_image_c7e56d68_20260809/ref2va_0026_image_audio_corrected_5s_generated_audio_32k_stereo.wav`
- 左/右/平均单声道：同目录 `ref2va_0026_corrected_5s_{left,right,mono_average}.wav`
- 精确 prompt：同目录 `ref2va_0026_image_audio_corrected_5s_prompt.txt`

output-opt 源码上另做了不带音频参考的 Ref2VA image-only 回归：HTTP 200，
832×480、124 帧/5.17 s，curl 189.75 s、engine 188.19 s、forward 181.83 s，
forward 结束到 engine E2E 约 6.36 s；四卡峰值约 31.34 GiB，最低
MemAvailable 109.57 GiB。全流解码通过，人物身份/背景/构图稳定，音轨
RMS 约 -60.66 dBFS，实质接近静音。源图人物本身张嘴，视频后半段才闭嘴，
因此 prompt 的“全程闭嘴”没有严格遵从，但本用例不涉及参考音频或叠声。产物：
`/nfs-data/h3_samples/latest_image_c7e56d68_20260809/ref2va_0026_output_opt_image_only_5s_832x480_s20_seed42026.mp4`。

视频参考 Ref2VA 也完整返回 HTTP 200：832×480、192 帧、8.0 s，总耗时
778.35 s；reference video encode 49.94 s、embedded audio encode 2.08 s、diffuse
699.71 s、decode 5.37 s，峰值 39669 MiB/卡、最低 MemAvailable 95.17 GiB。它只能证明
链路能跑通；去噪阶段 driver free 最低约 0.76 GiB，4×A100 40GB 上没有可靠的并发或
抖动余量，并且该样本没有显式写出目标对白，音频质量仍不能据此判定通过。

视频参考产物：
`/nfs-data/h3_runs/c7e56d68_0030_ref2va_video_20260809/ref2va_0030_video_reference_832x480_193f_s20_seed43030.mp4`。
文件名中的 `193f` 是请求前的历史预估标签；`ffprobe` 实测封装为 192 帧，
不应把文件名当作媒体帧数契约。
