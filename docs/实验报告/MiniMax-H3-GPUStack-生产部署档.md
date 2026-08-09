# MiniMax-H3 GPUStack 生产部署档（4×A100-PCIE-40G）

> 2026-08-09 定稿（#51）。这是 H3 上线的**权威配置**，与它冲突的历史命令一律以本文为准。
> 背景、踩坑过程、选型依据见 `MiniMax-H3-A100-40G-全局方案.md` 与
> `vLLM-Omni-MiniMax-H3-踩坑总账与选型依据.md`，本文只写"怎么起、能跑多少、边界在哪"。

---

## 1. 一句话

`/nfs-data/models/MiniMax-H3-FL2VA-INT8` + **tp4 + CPU offload + VAE tile 并行**，
单机 4 卡 A100-40G 起一个副本，**同一时刻只服务一个请求**。
生产档 **20 步 / flow_shift 12**，480p 端到端约 **70–72 s**。

---

## 2. 前置条件（少一条就起不来）

| 项 | 值 | 少了会怎样 |
|---|---|---|
| 权重 | `/nfs-data/models/MiniMax-H3-FL2VA-INT8`（离线 W8A8 INT8） | — |
| 显存 | 4×A100 40G，**整卡独占** | 峰值 24.1 GiB/卡，和别的模型混部必炸 |
| 宿主内存 | ≥ 200 GB 可用 | offload 常驻约 137 GB，不够会被 OOM killer 杀 worker |
| NFS 挂载 | `/nfs-data` 挂进容器 | 加载不到权重 |
| 量化参数 | **一个都不要传** | checkpoint 本身就是离线 INT8，传 `--quantization` 会被误判成在线量化路径 |

**`--deploy-config` 已可用（#57，2026-08-09）。** H3 现在注册在
`vllm_omni/config/pipeline_registry.py::OMNI_PIPELINES`，key 是 **`minimax_h3_dit`**：

- 生产档 YAML：`deploy-configs/minimax_h3_a100_40g.yaml`（随镜像走，见 §3.2）
- 上游通用档：`vllm_omni/deploy/minimax_h3_dit.yaml`

这是个**纯部署键**：`hf_architectures=()`、不填 `diffusers_class_name`、`deploy_only=True`，
`try_infer_model_type` 的任何一条路径都探测不到它，只能由 YAML 里的 `pipeline:` 字段点名。
所以**不传 `--deploy-config` 时的行为和以前一模一样**——裸 `vllm serve <H3路径>` 仍然解析不到
pipeline，§3 那条命令继续有效。

`deploy_only` 这个字段是 Codex 检视（2026-08-09）后补的，堵的是最后一级兜底：那一步拿注册键
（剥掉 `-`/`_` 后）去和**整条模型路径**做子串匹配，命中不是无害的——
`_create_legacy_from_registry` 会顺手加载该 pipeline 的 `default_deploy_config_name`，
于是裸 CLI 启动被**静默套上 4 卡 tp4 的默认档**。打了 flag 后这一级直接跳过该条目，
任何目录名都点不进来（`tests/config/test_minimax_h3_pipeline.py::test_key_shaped_path_does_not_auto_select`
钉住；把 flag 改回 False 该用例立刻红）。
key 不叫 `minimax_h3` 是第二道保险：`minimaxh3` 正好是 `minimaxh3fl2vaint8` 的子串。
注：`hunyuan_image3_ar` / `hunyuan_image3_dit` 有同样的暴露面，尚未打 flag（改它们属行为变更，另开）。

（历史：#51 之前传 `--deploy-config` 会被整个丢掉、回落单卡默认配置，加载到一半 OOM；
现在**未注册**的模型配 `--deploy-config` 直接 ValueError 退出，不再静默回落。）

### 2.1 GPUStack 纳管时的安全约束

`--allowed-local-media-path` **绝不能白名单 `/`**。引擎原生的 OpenAI 端点透过模型代理可达，
白名单 `/` 等于把宿主机任意文件用 `file://` 开放出去。解析顺序 fail-closed：

1. `GPUSTACK_MEDIA_ROOT` 显式指定 → 用它
2. 没设、但只有一个 EXTRA_MOUNT → 用那个挂载点
3. 多个挂载点又没设 MEDIA_ROOT → **不注入**（只告警）

---

## 3. 起服务（照抄）

```bash
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=7200   # 假墙 2：默认 30 s，大分辨率必超时
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200       # 假墙 3：注意无 _S 后缀
export VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0        # 假墙 1：置 0 关闭
export VLLM_OMNI_H3_OFFLOAD_DIT_BEFORE_VAE=1   # 进 VAE 前把 DiT 换出，腾 ~12.9 GiB/卡
export VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK=8   # VAE 后处理分块，拆掉整片 float32 缓冲（#45）
export VLLM_OMNI_H3_LOG_STEP_MEMORY=1          # 可选：每步打显存，排查用，稳定后可关

exec vllm serve /nfs-data/models/MiniMax-H3-FL2VA-INT8 --omni --host 0.0.0.0 --port 8091 \
  --trust-remote-code --num-gpus 4 -tp 4 --usp 1 --ring 1 --enable-cpu-offload \
  --text-encoder-tp-size 4 --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN --safetensors-load-strategy lazy \
  --disable-multithread-weight-load --enable-diffusion-pipeline-profiler \
  --init-timeout 2400 --stage-init-timeout 2400 \
  --diffusion-compile-granularity regional --diffusion-compile-dynamic
```

### 3.1 关键 flag 为什么是这个值

- **`-tp 4 --usp 1`**：tp4 把 33B DiT 切开，宿主还剩 76–98 GB 余量，13/13 请求全活；
  usp4 是**整模型复制 4 份**，宿主只剩 22–24 GB，NCCL 的裸 `cudaMalloc` 变成抛硬币，
  13 个请求死了 4 个。**不要因为 usp 快就换回去**——历史上 usp4 的延迟数据不适用于本档。
- **`--enable-cpu-offload`**：40G 卡装不下常驻，必须换入换出。
- **`--vae-parallel-mode tile --vae-use-tiling`**：VAE 解码是第二个显存尖峰，不分块会顶到 OOM。
- **`--disable-multithread-weight-load`**：多线程加载会把宿主内存瞬时拉高一倍，
  加载期就被 OOM killer 杀。**真正起作用的是这一个**。
- **`--safetensors-load-strategy lazy`**：**在扩散路径上是空转**，留着只是和历史命令对齐。
  `diffusion_model_runner.py` 构造 `DiffusersPipelineLoader` 前是 `LoadConfig()` 不带参数，
  `diffusers_loader.py:292` 读到的永远是默认值（`None`）。要删也行，删了产物不变。
- **`--diffusion-compile-granularity regional`**：整图编译在 52 个 block 上编不动；
  regional 冷启动多付约 17 s，之后稳态。

### 3.2 用 `--deploy-config` 起（等价形态，#57）

`deploy-configs/minimax_h3_a100_40g.yaml` 是 §3 那条命令的逐条转写，随镜像走
（`docker/Dockerfile.cuda` 把整个 `deploy-configs/` COPY 进去）。用它可以把 §3 里
`--tp/--usp/--ring/--enable-cpu-offload/--text-encoder-tp-size/--vae-*/--diffusion-*`
那一大串换成一个文件：

```bash
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=7200   # 六个 env 一个都不能少，见下
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200
export VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0
export VLLM_OMNI_H3_OFFLOAD_DIT_BEFORE_VAE=1
export VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK=8
export VLLM_OMNI_H3_LOG_STEP_MEMORY=1

exec vllm serve /nfs-data/models/MiniMax-H3-FL2VA-INT8 --omni --host 0.0.0.0 --port 8091 \
  --deploy-config /path/to/deploy-configs/minimax_h3_a100_40g.yaml \
  --init-timeout 2400 --stage-init-timeout 2400
```

**两样东西挪不进 YAML，别试**（都已按代码核过，不是猜的）：

1. **六个 `VLLM_OMNI_*` 环境变量**。stage 级 `env:` 会被解析进 `StageDeployConfig.env`
   然后原地丢掉——`stage_runtime_env` / `stage_runtime_setup`（`engine/stage_init_utils.py`）
   没有活的调用方，生产路径 `stage_engine_startup.py` → `setup_stage_devices` 只传 `devices`。
2. **`--init-timeout` / `--stage-init-timeout`**。既不是 `StageDeployConfig` 字段，
   也不是 pipeline-wide 字段，写进 YAML 会落进 `engine_extras` 再被过滤掉。

另外 `--num-gpus 4` 不用传：`num_gpus` 由 `parallel_config.world_size` 推导
（tp4×sp1×cfg1 = 4，`stage_init_utils.py:1382`）。

> **YAML 里并行度必须写成嵌套 `parallel_config:` 块。**
> `tensor_parallel_size` / `ulysses_degree` / `ring_degree` / `text_encoder_tp_size` /
> `vae_patch_parallel_size` / `vae_parallel_mode` 是 `DiffusionParallelConfig` 的字段，
> 不是 `OmniDiffusionConfig` 的。YAML 的 stage 字段是**平铺**进 `engine_args` 的
> （只有 CLI runtime_overrides 会被 `_apply_diffusion_parallel_runtime_overrides` 折进
> `parallel_config`），而 `OmniDiffusionConfig.from_kwargs` 只保留自己有的字段名——
> 平铺写 `tensor_parallel_size: 4` 会被**静默丢掉**，起来就是单卡。

---

## 4. 实测（干净宿主，0024，2026-08-09）

生产档 480p：`832×480 / duration=5.166667（124 帧）/ 20 步 / flow_shift=12 / seed=1101 / t2va`

| 轮次 | 端到端 | `forward` | 其中 `decode` | DiT s/it | 产物字节 |
|---|---|---|---|---|---|
| run1 | 72.13 s | 61.24 s | 2.68 s | 2.73 | 2,960,135 |
| run2 | 70.43 s | 61.29 s | 2.73 s | 2.73 | 2,960,135 |

- 产物 ffprobe：h264 832×480 / 124 帧 / 5.166667 s ＋ aac 32 kHz 立体声 / 163 帧 / 5.207 s。
- **两轮字节完全一致**：480p 在 tp4 上确定性成立（768p 的不确定性是另一回事，见 #37）。
- 资源：峰值 **24.1 GiB/卡**（torch allocated 23.25 / reserved 23.38），
  请求间空闲落回 **12.9 GiB/卡**（DiT 已换出）；宿主 **137 GB used / 111 GB available**。

其他档位（历史扫描，#24/#25，同 seed 同 prompt）：

| 档 | 20 步 | 50 步 |
|---|---|---|
| 480p 832×480 / 124 帧 | 51.5–75.2 s | 117.1–165.1 s |
| 768p 1344×768 / duration 4 s | 152.5 s | 303.6 s |

> ⚠️ 读数注意：**端到端会被宿主内存抖动放大，横比要看 `DiT s/it` 那列**（全程 2.22–2.73，很稳）。
> 另外**别拿 50 步的数去对 20 步的基线**——步数是线性的，50 步就是比 20 步慢 2.5 倍，
> 这不是劣化。

---

## 5. 请求参数边界

```bash
curl -sS -X POST http://127.0.0.1:8091/v1/videos/sync \
  -F 'prompt=...' \
  -F 'width=832' -F 'height=480' -F 'aspect_ratio=16:9' -F 'fps=24' \
  -F 'num_inference_steps=20' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5.166667,"audio_flow_shift":3.0}' \
  -o out.mp4
```

| 参数 | 约束 | 违反时 |
|---|---|---|
| `duration` | **[4, 15] s**（fps 固定 24） | HTTP 400，0.02–0.06 s 内返回，不占生成槽位 |
| 帧数 | `n % 17 == 5` | 自动对齐，不报错 |
| `aspect_ratio`（t2va） | 具名值六选一（`21:9`/`16:9`/`4:3`/`1:1`/`3:4`/`9:16`），`adaptive`/`auto` 不接受。**契约按「恒传」定**，见下方注 | HTTP 400 |
| `num_frames` | 必须走**顶层表单字段**，塞进 `extra_params` 无效（`serving_video.py:177`） | 静默回落成默认 209 帧 |
| `num_inference_steps` | 生产档 20 | 8 步/shift 5 画面是坏的，已目视否决 |
| `enable_frame_interpolation` | **H3 不支持**，传了也不生效 | 静默空转，产物字节不变（#59） |
| 并发 | **1** | H3 一次只服务一个请求，网关要串行排队 |

> **注：`aspect_ratio` 该不该恒传，代码与实测对不上，按「恒传」执行。**
> 92ad602c 记的是「传了 `width`/`height` 就可省（实测 200）」，但代码里
> `_resolve_shape`（`pipeline_minimax_h3.py:869-881`）是先无条件调用
> `_resolve_minimax_h3_aspect_ratio`，再走 `if height is None or width is None` 分支；
> 该函数对 `task == "t2va"` 且 `value is None` 直接
> `raise OmniClientError("t2va requires an explicit aspect_ratio")`（`:416-418`）。
> 取值只有 `extra_args["target"]["aspect_ratio"]` 与 `extra_args["aspect_ratio"]` 两处
> （`:871`）。#57 之后 H3 已注册进 `OMNI_PIPELINES`，`default_sampling_params_list` 不再必然为空，
> 所以这里**复核过 deploy-config**：`minimax_h3_a100_40g.yaml` 的 `default_sampling_params`
> 只设了 `num_inference_steps: 20`，**没有 `aspect_ratio`**——两种起法（裸 CLI / deploy-config）
> 下都兜不出默认值，按代码读**省不掉**。
> 那次 200 未能复现出成因（可能 `extra_params` 里还留着该键，或跑的是 fl2va）。
> 两种读法下「恒传」都是安全的，故契约按恒传定；要销掉这条冲突需在 0024 上重跑一次纯 t2va、
> `extra_params` 只留 `task`/`duration` 的对照。

**参数非法的返回码（2026-08-09 已修，#58）**：以前 `OmniClientError` 在 worker 里抛出后，
状态码在跨进程回传时丢了（三跳都只带 `str(exc)`），最终落进 `api_server.py` 的通用
`except Exception`，一律 500。现在 400 能原样穿回来，0024 实测：

| 用例 | 返回 | body |
|---|---|---|
| `num_frames=719`（29.958 s） | **400** / 0.062 s | `MiniMax H3 output duration must be in [4, 15] seconds, got 29.958` |
| `duration=2` | **400** / 0.042 s | `... got 2.0` |
| `fps=30` | **400** / 0.016 s | `MiniMax H3 output fps is fixed at 24` |
| 生产档合法请求 | 200 / 89.1 s（冷启首请求，含 regional 编译；稳态 70–72 s） | — |

**仍在的行为，不算缺陷但要知道**：帧数对齐 `n % 17 == 5` 是**只向上取**的，且发生在
`[4,15] s` 检查之后，所以一个 15.0 s 的请求实际出 362 帧 = 15.083 s。
不改：改成"对齐后再校验"只有两条出路——超界就拒（把合法请求拒掉）或往下退一档
（14.4–15.0 s 的请求统统塌到 345 帧 = 14.375 s），两者都比现在多超 0.083 s 更难受。
所以 `[4,15]` 是**请求时长**的界，交付时长最多再长 16/24 s。

---

## 6. 未决

- ~~#57 给 H3 注册 PipelineConfig~~ **已完成（2026-08-09）**，见 §2 / §3.2。
  按 HunyuanImage3 的纯部署键拆法做的：`minimax_h3_dit`，`hf_architectures=()`、
  不填 `diffusers_class_name`，裸 CLI 行为不变。
  落地：`vllm_omni/model_executor/models/minimax_h3/pipeline.py`、
  `vllm_omni/deploy/minimax_h3_dit.yaml`、`deploy-configs/minimax_h3_a100_40g.yaml`、
  `tests/config/test_minimax_h3_pipeline.py`（6 条，含"裸路径仍解析不到 pipeline"回归）。
  **仍未做真机对照**：YAML 形态与 §3 命令形态的产物一致性没跑过，上线前先跑一次同 seed 对照。
- #23 宿主内存超账治理——137 GB 常驻在独占机器上够用。**部署原则是一机一模型**，
  所以这条不是扩容约束，只有在影响自身稳定性时才值得动。
- #37 768p 同 seed 产物不确定。
