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

**不要传 `--deploy-config`。** H3 没注册进 `vllm_omni/config/pipeline_registry.py::OMNI_PIPELINES`
（它是纯 diffusers 仓，只有 `model_index.json`），deploy YAML 会被整个丢掉、回落成单卡默认配置，
然后在加载到一半时 OOM。2026-08-09 起这种组合会直接 ValueError 退出，不再静默回落。
要让 H3 支持 deploy-config 得先给它注册 PipelineConfig（#57）。

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
- **`--safetensors-load-strategy lazy --disable-multithread-weight-load`**：
  多线程加载会把宿主内存瞬时拉高一倍，加载期就被 OOM killer 杀。
- **`--diffusion-compile-granularity regional`**：整图编译在 52 个 block 上编不动；
  regional 冷启动多付约 17 s，之后稳态。

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
| `aspect_ratio`（t2va） | **没传 `width`/`height` 时必填**具名值，`adaptive`/`auto` 不接受；传了宽高就可省 | HTTP 400 |
| `num_frames` | 必须走**顶层表单字段**，塞进 `extra_params` 无效（`serving_video.py:177`） | 静默回落成默认 209 帧 |
| `num_inference_steps` | 生产档 20 | 8 步/shift 5 画面是坏的，已目视否决 |
| `enable_frame_interpolation` | **H3 不支持**，传了也不生效 | 静默空转，产物字节不变（#59） |
| 并发 | **1** | H3 一次只服务一个请求，网关要串行排队 |

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

- #57 给 H3 注册 PipelineConfig，让 `--deploy-config` 真正可用。**照 HunyuanImage3 的
  AR/DiT 拆法做**：那两个 key（`hunyuan_image3_ar` / `hunyuan_image3_dit`）都是
  `hf_architectures=()` 的纯部署键，任何 HF config 都探测不到它们，只能由 yaml 里的
  `pipeline:` 字段点名（`config_factory.py:236` 里这条优先级最高，直接短路自动探测）。
  H3 同样不填 `diffusers_class_name`，裸 CLI 启动的行为就一点不变，风险归零。
- #23 宿主内存超账治理——137 GB 常驻在独占机器上够用。**部署原则是一机一模型**，
  所以这条不是扩容约束，只有在影响自身稳定性时才值得动。
- #37 768p 同 seed 产物不确定。
