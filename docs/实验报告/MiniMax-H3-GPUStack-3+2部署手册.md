# MiniMax-H3 GPUStack 3+2 部署手册

> 日期：2026-08-19。本文是 BF16 Base/Turbo 五实例的新部署入口；历史
> `MiniMax-H3-GPUStack-生产部署档.md` 中的 INT8 部署和压测记录仍保留，但不要用它的
> 权重路径或固定 20 步 YAML 新建本文的 3+2 实例。

## 1. 发布结论

3+2 是五个独立 GPUStack 模型：FL2VA Base/Turbo4/Turbo8，Ref2VA Base/Turbo4。
Ref2VA 暂无独立的上游 Turbo8，因此不是六个实例。

功能和部署配置已达到出包入口，但上线时必须使用本次 commit 构建的不可变镜像。
现有 `arm64-a100-latest` 只用于开发环境验证，不能代替“新镜像、不挂载源码”的最终冒烟。

## 2. 五个模型

| GPUStack 名称 | 持久模型路径 | 分区 | 不传步数时 | Deploy config |
|---|---|---|---:|---|
| `minimax-h3-fl2va-base` | `/nfs-models/wuhanjisuan894/models/MiniMax-H3/FL2VA` | FL2VA | 20 | `/deploy-configs/minimax_h3_fl2va_bf16_a100_40g.yaml` |
| `minimax-h3-fl2va-turbo4` | `/nfs-models/wuhanjisuan894/models/MiniMax-H3-FL2VA-Turbo4-768p-BF16-vLLM` | FL2VA | 4 | 同上 |
| `minimax-h3-fl2va-turbo8` | `/nfs-models/wuhanjisuan894/models/MiniMax-H3-FL2VA-Turbo8-BF16-vLLM` | FL2VA | 8 | 同上 |
| `minimax-h3-ref2va-base` | `/nfs-models/wuhanjisuan894/models/MiniMax-H3/Ref2VA` | Ref2VA | 20 | `/deploy-configs/minimax_h3_ref2va_bf16_a100_40g.yaml` |
| `minimax-h3-ref2va-turbo4` | `/nfs-models/wuhanjisuan894/models/MiniMax-H3-Ref2VA-Turbo4-BF16-vLLM` | Ref2VA | 4 | 同上 |

两份 YAML 都不写死 `num_inference_steps`。Base/Turbo 默认值由所选模型目录的元数据
决定，而请求中的显式值仍可覆盖它。这也是三个 FL2VA 复用一份 YAML、两个
Ref2VA 复用一份 YAML 的原因。

## 3. GPUStack 页面逐个新建

在“模型”→“部署模型”中选本地路径，然后对表中每一行各建一个模型。

| 页面字段 | 填写值 |
|---|---|
| 名称 / 模型路径 | 按 §2 对应行填写 |
| 类别 | `video` |
| 推理后端 | `vLLMOmni` |
| 后端版本 | `1.0.0` |
| 镜像 | 本次构建的 `arm64-a100-YYYYMMDD-HHMM-<commit>` 不可变 tag |
| 副本数 | `1` |
| 放置策略 | `spread` |
| 每副本 GPU 数 | `4` |
| 跨 worker 分布式推理 | 关闭 |
| 量化 | 不填（五个均为 BF16） |

FL2VA 三个实例的后端参数：

```text
--deploy-config=/deploy-configs/minimax_h3_fl2va_bf16_a100_40g.yaml
--init-timeout=3600
--stage-init-timeout=3600
```

Ref2VA 两个实例的后端参数：

```text
--deploy-config=/deploy-configs/minimax_h3_ref2va_bf16_a100_40g.yaml
--init-timeout=3600
--stage-init-timeout=3600
```

`vLLMOmni` 后端会自动加入 `vllm serve <model> --omni`、`--trust-remote-code`以及
GPUStack 分配的 `--host/--port`，不需要重复填写。不要手工传 `-tp`、`--num-gpus`、
VAE 并行或 CPU offload 参数；它们已在 YAML 中统一固定。

## 4. 环境变量和节点资源

五个模型均在 GPUStack 的模型环境变量中填：

```text
VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=7200
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200
VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0
VLLM_OMNI_H3_OFFLOAD_DIT_BEFORE_VAE=1
VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK=8
VLLM_OMNI_H3_INFERENCE_CONTRACT=legacy
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

灰度期可再加 `VLLM_OMNI_H3_LOG_STEP_MEMORY=1`；稳定后改为 `0` 或删除，避免长期打印
每步显存。Ref2VA 的保比例、不无效放大和面积上限已放在 Ref2VA YAML 的 stage
env 中，新版引擎会将它透传给 worker，不需要在页面重复配。

GPUStack worker 进程的管理员环境变量（不是模型 env）：

```text
GPUSTACK_EXTRA_MOUNTS=/nfs-models,/nfs-output
GPUSTACK_MEDIA_ROOT=/nfs-output
```

每个副本需要同一台 worker 上的 4 张 A100-40G，建议宿主可用内存不低于
200 GB。五个选项若要同时常驻，需要 20 张 A100，且是五组同机四卡；不能把一个
TP4 副本拆到不同 worker。

## 5. 步数规则

| 模型 | 不传时 | 产品建议 | API 显式传入 |
|---|---:|---|---|
| Base | 20 | UI 可给 20/25/30/50 常用档 | 1–200 的任意整数；25 步合法 |
| Turbo4 | 4 | 优先 4，可给 8 | 1–200 在协议上合法，但超过蒸馏有效区间通常无收益 |
| Turbo8 | 8 | 优先 8，可给 4 | 同上 |

优先级始终是“请求显式值 > 模型默认值”。引擎对步数的协议范围是 1–200；
管理界面上的推荐选项不应实现成引擎枚举。

new-api 的 `VideoModelConfig.models` 中至少要为五个同名模型配：

```json
{
  "minimax-h3-fl2va-base":     {"engine": "minimax-h3", "pipeline": false, "defaultSteps": 20},
  "minimax-h3-fl2va-turbo4":   {"engine": "minimax-h3", "pipeline": false, "defaultSteps": 4},
  "minimax-h3-fl2va-turbo8":   {"engine": "minimax-h3", "pipeline": false, "defaultSteps": 8},
  "minimax-h3-ref2va-base":    {"engine": "minimax-h3", "pipeline": false, "defaultSteps": 20},
  "minimax-h3-ref2va-turbo4":  {"engine": "minimax-h3", "pipeline": false, "defaultSteps": 4}
}
```

new-api 只在请求没有 `num_inference_steps` 时补 `defaultSteps`；用户明确传 25 等值时不会
被模型默认覆盖。模型名必须与 GPUStack 和渠道模型名一致，否则读不到对应的
`defaultSteps`。

## 6. 出包和上线门槛

1. 把本轮 H3 源码、两份部署 YAML、Turbo overlay 工具、测试和文档落在同一 commit。
2. 运行 `.github/workflows/build-arm64.yml`，保留 CI 生成的不可变 tag 与 digest。
3. 在 A100 上从该镜像直接启动，不挂载本地源码或 deploy-config overlay。
4. 五个实例分别核对分区和默认 NFE：Base 20、Turbo4 4、Turbo8 8。
5. 先灰度一个 FL2VA 和一个 Ref2VA，健康检查 `/ready` 为 200，再建齐五个模型。

启动日志必须能确认 `partition=fl2va/ref2va`、TP4、text encoder TP4、VAE patch4 和
`max_num_seqs=1`。如果日志显示单卡、错分区或 Turbo 在无覆盖请求下跑 20 步，立即停止灰度。

回滚时把新模型副本数设为 0，或回到上一个不可变镜像 tag；不删除
`/nfs-models` 中的 Base/Turbo 权重。
