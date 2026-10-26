# MiniMax-H3 Turbo BF16：测试记录与产物清单（2026-10-25）

> **状态：产物不可用，质量结论待定位。**
> 本文只记录"做过什么、数字是多少、东西在哪"，不作为选型依据。
> 画面在 15s 长片与运动/暗光场景出现严重重影；该现象在文档原装 CLI 路线上同样复现（见第 3 轮），
> 因此**不能归因于本次的 deploy-config**。

## 1. 测试环境（实测）

### 1.1 宿主机与镜像

| 项 | 值 |
| --- | --- |
| 节点 | gpu9 `10.0.0.227`（4 步）、gpu10 `10.0.0.126`（8 步）、gpu26 `10.0.0.32`（对照 B）、gpu23 `10.0.0.6`（现网） |
| 宿主机 | `dev-gpustack-a100-000n`，251 GiB RAM，A100 PCIE 集群共 55 台 |
| GPU | 4 × NVIDIA A100-PCIE-40GB（40960 MiB/卡），实例按 TP4 吃满 4 卡 |
| 驱动 | `580.65.06` |
| 镜像 | `crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-omni:arm64-a100-latest` |
| 镜像 digest | `sha256:9ca7aa5aa12c9dc3a38fc85c553cd9a5461cf3c7728e5d3520652b2946ee3071` |
| 镜像构建 | `2026-10-15T06:11:34Z`，体积 12.9 GB（节点上仅此一个 vllm-omni 镜像） |
| 运行时 | Python 3.12.3 / torch 2.13.0+cu130 / CUDA 13.0 / vLLM 0.31.0 |
| 存储挂载 | `/nfs-models`、`/nfs-output`、`/nfs-data`（`/nfs-data` → `/nfs-models/wuhanjisuan894`） |
| 容器参数 | `--gpus all --ipc=host --network=host --shm-size 64g` |

### 1.2 我的部署配置（4 步 / 8 步共用一份）

deploy-config：镜像内 `/deploy-configs/minimax_h3_fl2va_bf16_a100_40g.yaml`

| 字段 | 取值 |
| --- | --- |
| pipeline / task | `minimax_h3_dit` / `fl2va` |
| `max_num_seqs` / `devices` | `1` / `"0,1,2,3"` |
| 并行 | TP 4、text_encoder_tp 4、vae_patch_parallel 4、`vae_parallel_mode: tile`、usp/ring/cfg 均 1 |
| 注意力 | `FLASH_ATTN`（yaml + 环境变量双写） |
| 编译 | `regional` + `dynamic` |
| 显存策略 | `enable_cpu_offload: true`、`vae_use_tiling: true`、`enable_multithread_weight_load: false` |
| stage env | `VLLM_OMNI_H3_FL2VA_KEYFRAME_RESIZE=official_cover_crop` |
| 步数 | yaml **不写死**，由模型元数据给默认值；本次请求显式传 4 / 8 |

进程环境变量（launcher 注入）：

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=7200
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200
VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0
VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200
VLLM_OMNI_H3_OFFLOAD_DIT_BEFORE_VAE=1
VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK=8   # 现行代码已不读取，仅兼容旧脚本，无实际作用
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
```

CLI：`vllm serve <model> --omni --deploy-config <cfg> --allowed-local-media-path /nfs-output --init-timeout 2400 --stage-init-timeout 2400 --host 0.0.0.0 --port <p> --trust-remote-code`

模型：

| 档位 | 路径 | 源 LoRA |
| --- | --- | --- |
| 4 步 | `/nfs-models/wuhanjisuan894/models/MiniMax-H3-FL2VA-Turbo4-768p-v1.3-BF16-vLLM` | `minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors` |
| 8 步 | `/nfs-models/wuhanjisuan894/models/MiniMax-H3-FL2VA-Turbo8-768p-v1.1-BF16-vLLM` | `minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors` |

两档元数据里各带自己的契约：4 步 `sigma_shift video=6.0 / audio=3.0`、`base_schedule` 5 点；
8 步同 shift、`base_schedule` 9 点；LoRA rank 128 / alpha 8 → 缩放 0.0625。

请求侧统一：`seed=42`、无参考图（纯 t2va）、分辨率与时长按用例。

### 1.3 对比的现网对象

| 项 | 值 |
| --- | --- |
| 节点 / 端口 | **gpu23**，`10.0.0.6:40011`（PID 3845738） |
| 进程启动时间 | `2026-10-18 22:56:55`，即已连续运行约 7 天 |
| 模型路径 | `/nfs-models/wuhanjisuan894/models/MiniMax-H3-Singularity-VDN-INT8` |
| deploy-config | `/nfs-models/wuhanjisuan894/deploy-configs/minimax_h3_singularity_vdn8_fl2va_a100_40g.yaml` |
| LoRA | `/nfs-models/wuhanjisuan894/models/VDN-MiniMax-H3/stage-dmd-step-250-curve-fl2va-baked2` |
| 关键差异 | `VDN_WINDOW_ATTN` 稀疏窗口注意力、`VLLM_OMNI_H3_OFFLOAD_COMPONENTS=1`、默认 8 步、TP4 |
| 副本数 | GPUStack 注册表写 `replicas: 3`，但**当前只有 gpu23 一台在服务**（gpu24 / gpu25 已无 vllm 进程） |

> ⚠️ **配置漂移**：现网读的是 `/nfs-models/wuhanjisuan894/deploy-configs/` 那份，
> 仓库里的同名文件多了一行 `VLLM_OMNI_H3_EXPERIMENTAL_SHORT_EDGE: "1080"`。
> 现网那份**没有**这一行，所以实际出片尺寸就是请求尺寸（1344×768），未被放大到 1080 短边。
> 我最初按仓库副本写成了"现网开 1080"，已按实测改正——这也说明**不能拿仓库 yaml 当现网配置的真相**。

**身份如何确认**：`GET /v1/models` 返回的模型路径与上表一致；参数与 `prod_models.json` 中
`minimax-h3-fl2va` 的 `backend_parameters` / `env` 逐项对得上；并且与今日自建的独立副本
（`spec_v13`，gpu37）在 p1/p3 两条上**同 seed 输出 sha256 逐字节相同**。

### 1.4 权重核实

四份 LoRA 均 624 张量 / BF16 / alpha=8（缩放 0.0625），声明载荷 == 实际载荷；
组装产物 `-vLLM` 的 `transformer` 软链指向**烘好的**目录（不是基座），
元数据记录 312 对因子、`max_abs_error=0`。

## 2. 三轮测试

### 第 1 轮：基础 7 条三臂对比（20:00–20:53）

- 节点：gpu9（4 步）、gpu10（8 步）、现网走线上副本。
- 用例：`p1`–`p7`（p1 为 15s，p4 为 1024×1024）。
- 起服务：`start_fl2va_bake.sh`（`--deploy-config minimax_h3_fl2va_bf16_a100_40g.yaml`，FLASH_ATTN，TP4）。
- 跑批：`run_ab_matrix.sh`（每档 7 条顺序执行）。
- 结果：21 条成片；延迟（热态）：4 步 58–59 s、8 步 92–97 s、现网 105–107 s；
  15s 首请求 4 步 379 s / 8 步 496 s（含一次性预热）。

### 第 2 轮：10 场景 + 3 比例 + 冷热分离（21:10–22:05）

- 节点：gpu9 / gpu10 / 线上副本；新增 gpu23 上的独立显存采样器。
- 用例：`t8_s02/04/05/07/12`、`t9_s15/17/19/20/24`（10 个内容场景），
  外加竖屏 `s09`、宽幕 `s21`、方形 `s22` 各跑 cold+hot，以及 `p1` 15s 跑 cold+hot。
- 起服务：同一 launcher（服务常驻，不重启）；跑批 `run_scen.sh` / `run_scen_prod.sh`。
- 结果：共 19+19+11 = 49 条成片。稳态：4 步 56–60 s、8 步 92–95 s、现网 104–107 s；
  15s 热态：4 步 223 s、8 步 373 s、现网 296–298 s。
  预热收敛：4 步要 3 条、8 步要 2–3 条，现网首条即热。

### 第 3 轮：配置归因诊断（22:20–22:40）

目的是判断"重影是不是我配置错了"。三组对照：

| 组 | 做法 | 结果 |
| --- | --- | --- |
| A | 文档原装 `repo/tools/…/run_vllm_turbo_eval.sh`（挂 `repo/` 旧源码） | **起不来**：`ImportError: cannot import name 'uses_diffusers_adapter'` |
| B | `start_turbo_cli.sh`：同一镜像、**不传 deploy-config**、改用文档里的 CLI 参数 | 正常起服务；**画面同样严重重影** |
| C | 运行时注入 LoRA（`--lora-backend distill`） | 未执行：基座元数据无 `base_schedule`，该路线天然带不上蒸馏调度，按设计就不等价 |

A 组的死因说明：`repo/vllm_omni/.../pipeline_minimax_h3.py` 是 8-18 的 118 KB，
镜像内是 10-15 的 249 KB，新旧混挂必然崩。**历史干净的 turbo 产物在当前镜像上无法复现**，
节点上也只剩一个镜像 digest，无旧镜像可回退。

B 组与我的路线在 15s 上逐格对比（`diag_abc/abc15_compare.png`）：两者都是两个骑手叠影，
现网是干净单人。**因此 deploy-config 不是元凶。**

### 已排除的配置项（逐项核对过代码）

| 怀疑 | 核对结果 |
| --- | --- |
| 步数 | 4 步对应 5 个 sigma 边界、8 步对应 9 个，与 `model_index.json` 一致 |
| 蒸馏调度 `base_schedule` | `_read_base_schedule(release)` 直接读模型元数据，两档各带各的表 |
| `flow_shift`（6.0 vs 12.0） | `default_video_shift` 取 `sigma_shift_scales.video`，turbo 取到 6.0 |
| 权重是否真烘进去 | `-vLLM/transformer` 软链指向烘好的目录，元数据记录 312 对因子、`max_abs_error=0` |
| deploy-config | 第 3 轮 B 组排除 |

## 3. 关键数字（4 卡整机墙钟，热态）

| 长度 | 4 步 | 8 步 | 现网 |
| --- | ---: | ---: | ---: |
| 5s（10 场景稳态） | 56–60 s | 92–95 s | 104–107 s |
| 15s | 223 s | 373 s | 296–298 s |

显存（单卡峰值）：BF16 5s 19.7–19.8 GB、15s 25.9–27.1 GB、冷启动 24.4 GB；
现网 5s 18.5–19.5 GB、15s 22.0 GB。宿主内存：BF16 峰值 185 GiB（空闲常驻 139 GiB）、现网 139 GiB。

## 4. 产物位置

根目录：`/nfs-output/h3_turbo_eval/`

### 4.1 成片（72 次运行）

`results/<臂>_<用例>/`，每个目录含 `output.mp4`、`request.json`、`summary.json`、
`progress.json`、`samples.jsonl`（逐秒 GPU/内存采样）、`models.json`。

| 臂前缀 | 条数 | 说明 |
| --- | ---: | --- |
| `t4v13_*` | 26 | BF16 4 步（含 warm5s、p1 cold/hot、3 种比例 cold/hot） |
| `t8v11_*` | 26 | BF16 8 步（同上） |
| `prodv2_*` | 18 | 线上现网副本 |
| `ctlB4v13_*` | 2 | 第 3 轮 B 组（无 yaml）对照 |

### 4.2 软链对比树（日常查看用这个）

`compare_turbo_4v8/`

```
README.md        口径说明 + 结论索引 + 全场景表
index.json       每格三臂的 tag / 墙钟 / 显存
pairs/           51 条软链：<场景>__{t4v13,t8v11,prodv2}.mp4
sheets/          17 张三行对照拼图
```

### 4.3 指标与抽帧分析

| 目录 | 内容 |
| --- | --- |
| `ab_analysis_4v8/` | 基础 7 条 4 vs 8：7 张拼图 + `metrics.json` + `frames/` |
| `ab_3way_5s/` | 基础 6 条三臂：6 张拼图 + 指标 |
| `ab_3way_15s_live/` | 15s 三臂：1 张拼图 |
| `ab_scen_3way/` | 10 场景三臂：10 张拼图 + `metrics.json` |
| `ab_analysis_dryrun/`、`ab_preview*/`、`ab_scen_preview/` | 早期冒烟与预览，可删 |

指标日志：`ab_analysis_4v8.log`、`ab_3way_5s.log`、`ab_3way_15s_live.log`、`ab_scen_3way.log`。

### 4.4 诊断证据（第 3 轮）

| 文件 | 说明 |
| --- | --- |
| `diag_abc/abc_compare.png` | 5s：A-yaml / B-no-yaml / 现网 三格 |
| `diag_abc/abc15_compare.png` | **15s 三格，重影最直观的一张** |
| `diag_abc/strip_abcB.png` | 历史 turbo4 v1.1 的连续 3 帧（干净，用作反证） |
| `diag_abc/*_01..03.png` | abcB/abcD/abcE 的连续帧原图 |

### 4.5 资源采样

| 文件 | 说明 |
| --- | --- |
| `prod_gpu_samples.csv` | gpu23 现网节点逐 2 秒 GPU 显存/利用率，1201 行 |
| `results/*/samples.jsonl` | 每次运行的逐秒采样（GPU + 宿主内存） |

### 4.6 脚本

| 脚本 | 用途 |
| --- | --- |
| `start_fl2va_bake.sh` | 起 BF16 Turbo 服务（带 deploy-config） |
| `start_turbo_cli.sh` | 起服务（文档 CLI 参数，无 deploy-config）＝第 3 轮 B 组 |
| `run_scen.sh` / `run_scen_prod.sh` | 场景矩阵（含预热、cold/hot、15s） |
| `run_ab_matrix.sh` / `run_prod_live.sh` | 第 1 轮跑批 |
| `run_hot15.sh` | 15s 冷热对照 |
| `analyze_turbo_ab.py` | 抽帧指标 + PSNR/SSIM + 拼图 |
| `run_analysis_*.sh` | 三档分析入口 |

服务日志：`t4v13.serve.log`、`t8v11.serve.log`、`prodv2.serve.log`、`ctlA4v13.serve.log`、`ctlB4v13.serve.log`；
跑批日志：`scen_*.log`、`ab_*.log`。

## 5. 当前未决

1. **重影根因未定位**。已排除配置项（步数/调度/shift/烘焙/deploy-config），
   剩余怀疑是当前镜像（10-15）的稠密路径相对 8-18 版本的回退，但无旧镜像可验证。
2. **没有可用参照**：历史干净产物（abcB/abcD）来自旧代码，无法在当前镜像复现。
3. **音频**：turbo 两档峰值 −30 dB、现网 −9 dB；但历史 turbo 与 20 步基线同样偏轻
   （−21～−31 dB），故"轻"是既有特性，是否另有失真要听感确认。
4. 实例 `h3cmp-t4v13`（gpu9）、`h3cmp-t8v11`（gpu10）、`h3-turbo-eval-ctlB4v13`（gpu26）
   仍在运行，未删除。
