# MiniMax-H3 FL2VA 三档上线配置（2026-08-22）

> 面向执行者。从出镜像到 GPUStack 建实例到 new-api 配置，逐条可照抄。
> 三档只服务 **768p（1344×768）**。
> **2026-08-22 更新：最快档由 SLA 改为 v1.1 稠密**——SLA 在手绘动画上会出现运动方向
> 反转（写实内容不受影响），详见 `docs/实验报告/MiniMax-H3-SLA稀疏注意力-接入实测与暂缓结论-2026-08-22.md`。
> 数据来源见 `docs/实验报告/vLLM-Omni-MiniMax-H3-768p-Turbo-v1.1-与-SLA-稀疏注意力实测与选型.md`。

---

## 1. 三档定义

| | 官方满血 | 快一点 | 最快 |
|---|---|---|---|
| 权重目录 | `MiniMax-H3/FL2VA` | `MiniMax-H3-FL2VA-Turbo8-BF16-vLLM` | `MiniMax-H3-FL2VA-Turbo4-768p-v1.1-BF16-vLLM` |
| 步数（NFE） | 20 | 8 | 4 |
| 注意力 | 稠密 FlashAttention | 稠密 FlashAttention | 稠密 FlashAttention |
| shift（video/audio） | 12 / 3 | 12 / 3 | 6 / 3 |
| 15 秒实测 | 886.8s | 400.1s | **226.6s** |
| 10 秒实测 | 527.2s / 485.4s | 未测（见 §6） | **141.0 ~ 146.4s** |
| 15 秒峰值显存 | 35.8 GiB | 34.4 GiB | 35.9 GiB |
| 服务时长 | 5-15s | 5-15s | 5-15s |

shift 与 base_schedule 跟着权重的 `model_index.json` 走，已经烘在 partition 里，**不用也不能在启动参数里配**。

权重路径：计算节点上是 `/nfs-data/models/<目录>`，管理节点上是
`/nfs-models/wuhanjisuan894/models/<目录>`，GPUStack 填计算节点视角的路径。

---

## 2. 出镜像

三档都不需要新依赖，按现有流程出即可：

```bash
docker build -f docker/Dockerfile.cuda -t <registry>/reputationly/vllm-omni:arm64-a100-<tag> .
```

> 曾为最快档在 Dockerfile 里加过 `sparse_linear_attention`（SLA kernel），随最快档改用 v1.1
> 一并移除——不给生产镜像留用不上的依赖和构建期的 GitHub 网络依赖。`SLA_ATTN` backend 代码
> 仍在仓库里，但默认永远不会被选中；将来重启该实验时装包 + 加一行 attention-config 即可。

## 3. GPUStack 建实例

三档 **共用同一份 deploy-config**，不要新建：权重路径由 `vllm serve <路径>` 给，步数是请求级的，yaml 里那个只是兜底。

### 3.1 三档共同部分

**后端参数**：
```
--omni
--trust-remote-code
--deploy-config=/deploy-configs/minimax_h3_a100_40g.yaml
--init-timeout=2400
--stage-init-timeout=2400
```

**环境变量（六个，一个都不能少）**：
```
VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=7200
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200
VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0
VLLM_OMNI_H3_OFFLOAD_DIT_BEFORE_VAE=1
VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK=8
VLLM_OMNI_H3_LOG_STEP_MEMORY=1
```

这六个**挪不进 yaml**：stage 级 `env:` 会被解析进 `StageDeployConfig.env` 然后原地丢掉，生产路径不读它。`--init-timeout` / `--stage-init-timeout` 同理，写进 yaml 会落进 `engine_extras` 再被过滤掉。

每档一个实例，4 卡，一机一档。

### 3.2 最快档

无额外参数——与另外两档完全相同，只是权重路径与 `defaultSteps` 不同。

> 若将来重新启用 SLA：加
> `--diffusion-attention-config={"default": {"backend": "SLA_ATTN", "block_sparse": {"sparsity": 0.85, "start_step": 0}}}`，
> 并确认镜像里装了 `sparse_linear_attention`（缺包会拒绝启动，不会静默降级）。

## 4. new-api 配置

三档各配模型级 `defaultSteps`：**20 / 8 / 4**，与 `engine: minimax-h3` 同层。

这是实际生效的步数来源。不配的话请求不带步数，会跑 yaml 里的默认 20 步——加速档的加速全部丢失，而且不会有任何报错。

**门面层仍要确保 `aspect_ratio` 随请求下发**：历史上 `duration=6` 的 t2va 因为网关漏传这个
字段全挂过。三档都服务 5-15 秒，不再需要最快档的时长白名单。

全档位：门面层统一加 **−1 dBTP 限幅**。15 秒档实测五个档位音频全部削波，**满血基座也削**（+0.9 dBFS），这不是加速档引入的问题。

---

## 5. 上线后验证清单

**① 实例就绪**：`/v1/models` 返回 200。冷启动到就绪约 6-7 分钟。

**② 三档都应看到 FlashAttention 被选中**（`Resolved diffusion attention backend 'FLASH_ATTN'`）。

**③ 三档的 shift 与步数来自各自的 partition 元数据**，不用也不能在启动参数里配。

**④ 起完先打一发预热请求再放流量**：实例加载后的首发是稳态的 1.4~2 倍（10 秒档 199.4s vs 121.7s），含 Triton kernel 编译与 compile 预热。这一发的产物与稳态产物**不同**（同 seed 也不同），别拿它做基线。

**⑤ 对基准**（稳态、10 秒档、boars 类内容）：最快档约 **120s**，v1.1 稠密约 146s，满血约 535s。偏离一倍以上就查是不是退回稠密了。

---

## 6. 已知缺口与风险

| 缺口 | 影响 | 建议 |
|---|---|---|
| 快一点档（Turbo8）**10 秒未测** | 只有 15 秒的 400.1s，10 秒是外推 | 上线前补一发即可（引擎起着就能打） |
| 最快档 **10 秒的 SSIM 未出** | 画质只在 5 秒与 15 秒两端有数据，10 秒靠夹逼 | 满血 20 步的 10 秒基准跑完即可补 |
| **0.7@10 秒未测** | 0.85 在 10 秒已实测 −17~21%，但不排除 0.7 更优 | 想再榨一点就补，约 20 分钟 |
| **音频内容零对比** | 只校验了容器属性与电平 | 上线前做一次 A/B，尤其最快档 |
| **主观画质无人确认** | 全部结论基于 SSIM/闪烁等客观代理 | 三档各看两条片子再放量 |
| token_refiner 在长序列上会走稀疏 | 2101 行 = 33 块，刚过 32 块阈值 | 若发现文本跟随变差，把 `_MIN_KEY_BLOCKS` 提到 64 让它恒定稠密 |

**画质代价（已知，需你确认可接受）**：最快档相对 v1.1 稠密，15 秒档 SSIM 低 0.016、闪烁高 10%；5 秒手绘差距更大（SSIM 低 0.05）——但 5 秒不在最快档的服务范围内。真实拍摄类内容 SLA 反而更接近满血。

**运维注意**：纳管实例任何重启都会从镜像起新容器，**容器内的临时修改一律会丢**。SLA 相关的一切（backend 代码、kernel 包）必须在镜像里，不能靠挂载或 PYTHONPATH 叠加。
