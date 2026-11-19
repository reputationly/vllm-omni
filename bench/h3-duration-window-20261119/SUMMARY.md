# MiniMax-H3 出片时长窗口能否改成 [1, 20] 秒 · 现网配置核对 + 空闲机实测（2026-11-19）

> **性质**：现网（`ssh devm` GPUStack 集群 `a100-image-video`）**只读核对**；实验全部在 dev 空闲机 `gpu42`/`gpu55` 上，用**现网同款镜像 + 同款 pika deploy-config**，只改时长闸门。  
> 原始数据：`raw/ref2va_seq.log`、`raw/fl2va_seq.log`（每次请求的 `RESULT_JSON` + 产物 `ffprobe`）。
> 测试结束后两台机器的测试容器已删除，4 卡均回到 0 MiB；GPUStack 未被写入（无新增/改动 model 与 instance）。

---

## 1. 现网这两个模型是怎么配的

两个模型是 GPUStack `models` 表里的两条记录（id 149 = fl2va，id 151 = ref2va），backend `vLLMOmni 1.0.0`，**没有 `run_command`/`image_name` 覆盖**，全部走 backend 默认镜像 + deploy-config：

| 项 | minimax-h3-fl2va | minimax-h3-ref2va |
| --- | --- | --- |
| 权重 | `/nfs-models/wuhanjisuan894/models/Pika-H3-Flex-FL2VA-BF16-vLLM` | `/nfs-models/wuhanjisuan894/models/Pika-H3-Flex-Ref2VA-BF16-vLLM` |
| deploy-config | `/deploy-configs/minimax_h3_pika_fl2va_bf16_a100_40g.yaml` | `/deploy-configs/minimax_h3_pika_ref2va_bf16_a100_40g.yaml` |
| replicas | 7（ready 7） | 7（ready 7） |
| 单副本 | 4 × A100-PCIE-40G，`tensor_parallel_size: 4` | 同左 |
| 放置 | `SPREAD`；selector 预留 10 台 worker × 4 卡 | 同左 |
| 实际落地 | a100-**0022–0028** | a100-**0032–0038** |
| 端点示例 | 10.0.0.84:40034 …（7 个） | 10.0.0.148:40024 …（7 个） |
| 镜像 | `sha256:6881675a3afe…`（52.1 GB，`VLLM_BUILD_COMMIT=4ba787ec…`，vLLM 0.33.0） | 同左 |

`backend_parameters`（两者相同）：
```
--deploy-config=/deploy-configs/minimax_h3_pika_{fl2va,ref2va}_bf16_a100_40g.yaml
--allowed-local-media-path=/nfs-output --init-timeout=2400 --stage-init-timeout=2400
```

`env`（两者相同）：
```
VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=7200   VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200
VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0        VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_OMNI_H3_OFFLOAD_DIT_BEFORE_VAE=1   VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK=8
NUMPY_MADVISE_HUGEPAGE=0
```

deploy-config 的实质差异只有关键帧/参考图几何，与时长无关：

- **ref2va**：`task_type: ref2va`、`VLLM_OMNI_H3_INFERENCE_CONTRACT: legacy`、参考图 `official_short_edge` + short_edge 1536（无上采样、总像素 2400 万）、`vae_use_tiling`、`enable_cpu_offload`、`regional` 动态编译。
- **fl2va**：`task_type: fl2va`、`VLLM_OMNI_H3_FL2VA_KEYFRAME_RESIZE: source_center_crop`、`VLLM_OMNI_H3_FL2VA_CANVAS_POLICY: buckets`，其余同上。

> 注：`bench/prod-baseline-20261114/SUMMARY.md` 里写的 “replicas 10” 已过期，现在是 7+7；镜像 digest 也从 `8838e733…` 换成了 `6881675a…`。

---

## 2. 时长限制到底卡在哪：四道闸门，实际生效 **[4, 15] 秒**

按请求进入的顺序：

| # | 位置 | 现值 | 能否用 env / deploy-config 改 |
| --- | --- | --- | --- |
| 1 | API schema（`seconds` 字段） | `^[1-9]\d*$`（整数秒字符串） | 否（`0`、`1.5` 直接 400） |
| 2 | `model_executor/models/minimax_h3/encoder_processing.py` | `MINIMAX_H3_MIN_OUTPUT_SECONDS = 4.0`（**硬编码**） | 否 |
| 3 | `model_executor/models/minimax_h3/long_video.py::max_output_seconds()` | 未开 `long_video` → **15.0**；`long_video=true` → 30.0（full）/300.0（continuation） | 否（只能靠请求里传 `long_video`） |
| 4 | `diffusion/models/minimax_h3/strategy.py`（legacy contract） | `(2.0, 16.0)` | **只有上界**能用 `VLLM_OMNI_H3_MAX_OUTPUT_SECONDS` 覆盖；下界无 env |
| 5 | `pipeline_minimax_h3.py::_resolve_shape` | 复用 #4 的窗口，换成帧窗口 | 跟随 #4 |

**实测（现网镜像原样未改）**：`seconds=3` → 400，`seconds=16` → 400，`seconds=25` → 400，报错都是
`MiniMax H3 output duration must be in [4, 15] seconds`。
⇒ **现网有效窗口是 [4, 15] 秒**，文档里的 `[2, 16]` 已被 #2/#3 这两处新代码收窄，不要再按文档算。

另一个容易踩的点：`17n+5` 帧栅格（24 fps）会让“请求秒数”和“出片秒数”不一致——

| 请求 | 实际帧数 | 实际出片 |
| --- | --- | --- |
| 1 s | 39 | **1.625 s** |
| 2 s | 56 | 2.333 s |
| 5 s | 124 | 5.167 s |
| 15 s | 362 | 15.083 s |
| 20 s | 481 | 20.042 s |

栅格上最小的合法长度是 **22 帧 = 0.917 s**，所以“请求 1 秒”物理上不可能出 1.000 s 的片子（除非改对齐策略向下取 22 帧）。

---

## 3. 空闲机器

2026-11-19 10:45 扫了全部 55 台 worker（`nvidia-smi` 实测 + 容器清点）：

| 分类 | 机器 | 说明 |
| --- | --- | --- |
| **完全空闲** | a100-**0029/0030/0031/0039/0040/0042/0055** | 只有 `gpustack-worker`，4 卡 0 MiB，RAM 空闲 ~239 GiB |
| 空闲（有闲置试验容器，GPU 0 MiB） | 0021、0043、0045、0046、0048、0049 | 有 `v41trial` 等容器但不占卡 |
| 不要用 | 0041、0047、0050 | `vllm-backport`/`v41trial` 占 35–38 GiB/卡 |
| 已被现网占用 | 0022–0028（fl2va）、0032–0038（ref2va） | 常驻 0.9–1.1 GiB/卡（cpu_offload 空载） |
| selector 预留但未分配 | 0021、0029、0030（fl2va selector）、0031、0039、0040（ref2va selector） | 不建议手动占：GPUStack 扩容会调度上去 |

本次实测选了 **gpu42（10.0.0.7）** 和 **gpu55（10.0.0.180）**——不在任何 model 的 `gpu_selector` 里，也不承载任何实例。

---

## 4. 实测：[1, 20] 能不能跑通

做法：现网同款镜像 `sha256:6881675a`，同款 pika deploy-config，只把三处闸门改成 `[1, 20]`：

```
strategy.py            output_duration_seconds=(2.0, 16.0)  -> (1.0, 20.0)
encoder_processing.py  MINIMAX_H3_MIN_OUTPUT_SECONDS = 4.0  -> 1.0
long_video.py          非 long_video 上界 return 15.0        -> 20.0
```

（启动日志里 `MiniMax-H3 contract: {... 'output_duration_seconds': [1.0, 20.0] ...}` 已确认生效。）

### 4.1 结果

| 请求 | ref2va（gpu42） | fl2va（gpu55） |
| --- | --- | --- |
| 1 s | ✅ completed **41.2 s**，峰值 **19.7 GiB/卡**，产物 39 帧 / **1.625 s** | ✅ **44.0 s**，17.9 GiB/卡，39 帧 / **1.625 s** |
| 5 s（冷，进程首跑） | ✅ 169.8 s，24.9 GiB/卡 | ✅ 155.6 s，24.8 GiB/卡 |
| 5 s（热） | ✅ **86.5 s**，20.1 GiB/卡，124 帧 / 5.167 s | ✅ **78.2 s**，20.8 GiB/卡，124 帧 / 5.167 s |
| **20 s** | ✅ **458.2 s**，峰值 **29.1 GiB/卡**，产物 481 帧 / **20.042 s** | ✅ **437.4 s**，峰值 **28.3 GiB/卡**，481 帧 / **20.042 s** |
| 21 s | ⛔ 拒绝（`… in [4, 20] seconds`） | — |
| 0 s / 1.5 s | ⛔ schema 拒绝（`^[1-9]\d*$`） | ⛔ 同左 |

5 s 热发 78–86 s 与 11-14 的生产基线（fl2va 76.98 s、ref2va 84.77 s）吻合，说明测试机与现网副本可比。

20 s 的阶段耗时：

| 模型 | prepare | encode | denoise | decode | save | wall |
| --- | --- | --- | --- | --- | --- | --- |
| ref2va | 4.1 | 2.1 | **407.1** | 34.9 | 2.1 | 458.2 |
| fl2va | 4.1 | — | **392.6** | 30.8 | 2.1 | 437.4 |

### 4.2 结论

1. **[1, 20] 技术上可行**：pika BF16 + 4×A100-40G 上 20 s（481 帧）**没有 OOM、没有超时**，峰值 29.1/40.96 GiB（约 71%，余量 ~11 GiB/卡）。
2. **代价是时间**：20 s 单发 7.3–7.6 分钟，约为 5 s（78–86 s）的 **5.3 倍**；副本 `max_num_seqs: 1`，即每个副本同一时刻只服务一发。
3. **1 s 的语义要先定**：栅格决定了 1 s 请求出 1.625 s（+62%）。若产品要“接近 1 秒”，得把对齐改成向下取最近合法帧（22 帧 = 0.917 s）或允许非栅格帧数——这是产品决策，不是配置项。
4. **不是 deploy-config 的锅**：所有时长闸门都在镜像代码里（4 处），改配置没用；且 #2 的报错文案里 `[4, ...]` 是**硬编码字符串**，改常量时必须一起改，否则报错会继续骗人（本次实测已复现：窗口已是 `[1, 20]`，21 s 的报错仍写 `[4, 20]`）。

### 4.3 建议改法（三处代码，一处文案）

| 文件 | 改动 |
| --- | --- |
| `model_executor/models/minimax_h3/encoder_processing.py` | `MINIMAX_H3_MIN_OUTPUT_SECONDS = 4.0 → 1.0`；`duration_range = f"in [4, {max_seconds:g}]"` → `f"in [{MINIMAX_H3_MIN_OUTPUT_SECONDS:g}, {max_seconds:g}]"` |
| `model_executor/models/minimax_h3/long_video.py` | 非 `long_video` 分支 `return 15.0 → 20.0`（更稳的做法：做成 env/部署级可配，默认 15，允许部署抬到 20） |
| `diffusion/models/minimax_h3/strategy.py` | legacy `output_duration_seconds=(2.0, 16.0) → (1.0, 20.0)`（或给下界加 env，与已有的 `VLLM_OMNI_H3_MAX_OUTPUT_SECONDS` 对称） |
| 文档 | `docs/MiniMax-H3-官方Diffusers流程对齐开发与验收指南.md` 的 `[2, 16]`、以及任何写 “2–16 秒” 的地方都应更新为实际值 |

上线方式：改完需要重建镜像（现网是 pinned digest `6881675a`，只改 GPUStack 的 env/参数不生效）。**不需要**重启现网其它模型；4 处闸门的改动对 <15 s 的请求行为零影响（窗口只是放宽），可以灰度一台 worker 先验证再全量。

若短期不想重建镜像，唯一不改代码的折中是请求里传 `extra_params.long_video: true`（上界变 30 s，但下界仍 4 s，且会走 VDN/continuation 分支——pika BF16 配置没有 VDN，不推荐）。

---

## 5. 落地：出包与镜像验收（2026-11-19）

改动已提交并出包（分支 `feat/h3-duration-window-1-20`）：

| 项 | 值 |
| --- | --- |
| 改动 commit | `7ebda7f7a`（基于现网所在 commit `2fae9a2e`，只带这一个改动） |
| 不可变 tag | `arm64-a100-20261119-0355-7ebda7f7` |
| 别名 tag | `arm64-a100-h3dur-1-20`（**没有**挪动 `arm64-a100-latest`） |
| digest | `sha256:14d0da31969a9467a3813cd7a38897d23dc540513186f8f70ce02715bea8087c` |
| 基座 | `vllm/vllm-openai:v0.33.0@sha256:d825baf9…`（必须传 pinned digest，传浮动 tag 会被 `tools/prepare_ci_telemetry.py` 拦下） |
| CI run | https://github.com/reputationly/vllm-omni/actions/runs/44454689265 （22 min，成功） |

**单元测试**（在生产镜像 `6881675a` 里、用改动后的源码跑）：`test_minimax_h3_duration_window` /
`test_minimax_h3_long_video_limits` / `test_minimax_h3_strategy` / `test_client_error_across_worker` → **98 passed**。

**包级验收**（gpu42，直接拉新镜像、**不打任何补丁**、跑现网 pika ref2va 部署档）：

| 项 | 结果 |
| --- | --- |
| 镜像 digest 自证 | `sha256:14d0da31…`（与 release 一致） |
| 镜像内常量 | min `1.0`、`MINIMAX_H3_MAX_STANDARD_SECONDS = 20.0`、legacy 窗口 `(1.0, 20.0)`、文案由常量渲染 ✅ |
| `seconds=21` | failed：`must be in [1, 20] seconds, got 21.0` ✅（文案已不再是 `[4, 20]`） |
| `seconds=1` | completed，39 帧 / **1.625 s**，wall 100 s（含冷启动编译） |
| `seconds=20` | completed，481 帧 / **20.0417 s**，wall 458 s，denoise 410 s，峰值 **29.07 GiB/卡** |

原始记录：`raw/acc_shipped_image.log`。

GPUStack 侧怎么换成这个镜像（推荐路径、两个源码级坑、回滚陷阱、停机窗口）见 [UPGRADE.md](UPGRADE.md)。

---

## 6. 现网滚动：latest → main 构建，NFS 分发，删实例重建（2026-11-19 15:00–15:50）

按"用 `arm64-a100-latest` + 脚本 save 到 NFS + 在线升级 + 删除重建实例"的方式落地。

### 6.1 为什么这版要在 main 上出，而不是原来那个分支包

`latest` 当时指向 `0b4425e6c`（11-17，比现网在跑的 `2fae9a2e` 新一个 commit：`feat(tasks): carry billable usage on the async image task`）。
把 `latest` 挪到基于 `2fae9a2e` 的包，等于把那两个文件的计费上报回退掉。所以改动先 fast-forward 合入 `main`（`0b4425e6c..df481023b`），再从 main 出包——`latest` 单调前进。

| 项 | 值 |
| --- | --- |
| main 上的 commit | `8be56f218`（代码）+ `df481023b`（文档） |
| 镜像 | `arm64-a100-20261119-0635-df481023`，digest `sha256:154787d4…`，alias `arm64-a100-latest` |
| CI | run [44473208445](https://github.com/reputationly/vllm-omni/actions/runs/44473208445)，20 min 成功 |
| 上一步的氪版（作废） | `arm64-a100-20261119-0355-7ebda7f7` / `7ebda7f7a`（基于 `2fae9a2e`，**未**用于现网，`latest` 也没经过它） |

### 6.2 NFS 分发（脚本在 `dist/`）

```bash
# ① 在有镜像的机器上落盘（/nfs-models/images/）
./image-dist/bake_image_to_nfs.sh                     # 默认 crpi-…/vllm-omni:arm64-a100-latest
# 产物: vllm-omni-arm64-a100-latest-154787d46865.tar（16 G）+ .sha256 + .manifest

# ② 各 selector 机器载入并**覆盖本地同名 tag**（关键：本地 latest 原本是 4cce9e73）
./image-dist/load_image_from_nfs.sh <tar> <期望 image_id>
```

实测：20 台 selector（fl2va 0021–0030、ref2va 0031–0032…0040）在 15:09–15:19 以并发 5 全部载入，`arm64-a100-latest` 由 `4cce9e73` 变成 `154787d4`，20/20 校验通过。

> **这一步不是可选的**：滚动前 20 台本地 `arm64-a100-latest` 全是 `4cce9e73`（旧包）。不覆盖本地 tag 就删实例，容器会直接用本地旧镜像起来，镜像看起来升了、其实没升。

### 6.3 删实例重建（先 ref2va，后 fl2va）

| 步骤 | 时间 | 结果 |
| --- | --- | --- |
| DELETE `/v2/model-instances/{id}` × 7（ref2va） | 15:25 | 全部 200 |
| ref2va 重建 | 15:26 → 15:35 | STARTING×7 → RUNNING 7/7（**~9 min**） |
| DELETE × 7（fl2va） | 15:37 | 全部 200 |
| fl2va 重建 | 15:38 → 15:46 | RUNNING 7/7（**~8 min**） |

重建后实例名/端口都变了（例：`minimax-h3-ref2va-1p30g` @ 10.0.0.148:40059，`minimax-h3-fl2va-resv8` @ 10.0.0.84:40060）；GPUStack 模型路由按名字匹配，客户端无需改。

### 6.4 现网验收（直接打新副本，不经网关）

| 检查 | ref2va | fl2va |
| --- | --- | --- |
| 容器镜像 | `sha256:154787d4…` ✅ | `sha256:154787d4…` ✅ |
| 启动日志契约 | `output_duration_seconds': [1.0, 20.0]` ✅ | 同左 ✅ |
| `seconds=21` | failed `must be in [1, 20] seconds, got 21.0` ✅ | 同左 ✅ |
| `seconds=1` | completed，39 帧 / **1.625 s**（111.6 s，含冷启动） | completed，39 帧 / **1.625 s**（123.8 s） |
| `seconds=20` | completed，481 帧 / **20.0417 s**，453 s，峰值 **29.07 GiB/卡** | 包级验收已测（437 s / 481 帧），未重复压现网 |

### 6.5 遗留（需要知道的两件事）

1. **`latest` 现在就是这版**。其它 vLLMOmni 模型（audiox、indextts、qwen3-tts…）仍在旧 digest 的容器里跑，但**它们下次重建就会换到这个镜像**。因为它是 main 的快照（含 11-17 的计费 commit），不是回退，属预期行为。
2. **旧的两版只能靠不可变 tag 找回**：`arm64-a100-20261117-0116-0b4425e6`（原 latest）、`arm64-a100-20261116-1525-2fae9a2e`（现网原先在跑的 `6881675a`）。浮 tag 不再指向它们。
