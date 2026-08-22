# MiniMax-H3 FL2VA 三档上线配置（2026-08-22）

> 面向执行者。从出镜像到 GPUStack 建实例到 new-api 配置，逐条可照抄。
> 三档只服务 **768p（1344×768）**；最快档只服务 **10-15 秒**。
> 数据来源见 `docs/实验报告/vLLM-Omni-MiniMax-H3-768p-Turbo-v1.1-与-SLA-稀疏注意力实测与选型.md`。

---

## 1. 三档定义

| | 官方满血 | 快一点 | 最快 |
|---|---|---|---|
| 权重目录 | `MiniMax-H3/FL2VA` | `MiniMax-H3-FL2VA-Turbo8-BF16-vLLM` | `MiniMax-H3-FL2VA-Turbo4-768p-SLA-v0.1-BF16-vLLM` |
| 步数（NFE） | 20 | 8 | 4 |
| 注意力 | 稠密 FlashAttention | 稠密 FlashAttention | **SLA 块稀疏 0.85** |
| shift（video/audio） | 12 / 3 | 12 / 3 | 6 / 3 |
| 15 秒实测 | 886.8s | 400.1s | **174.6s** |
| 10 秒实测 | 待补（见 §6） | 未测（见 §6） | **110.8 ~ 121.7s** |
| 15 秒峰值显存 | 35.8 GiB | 34.4 GiB | 35.8 GiB |
| 服务时长 | 5-15s | 5-15s | **10-15s** |

shift 与 base_schedule 跟着权重的 `model_index.json` 走，已经烘在 partition 里，**不用也不能在启动参数里配**。

权重路径：计算节点上是 `/nfs-data/models/<目录>`，管理节点上是
`/nfs-models/wuhanjisuan894/models/<目录>`，GPUStack 填计算节点视角的路径。

---

## 2. 出镜像

镜像里唯一的新增依赖是 SLA kernel（`docker/Dockerfile.cuda` Step 2b-1，本次已加）：

```dockerfile
ARG SLA_COMMIT=main
RUN uv pip install --no-cache-dir \
      "sparse_linear_attention @ git+https://github.com/thu-ml/SLA.git@${SLA_COMMIT}" \
      || echo "WARN: failed to install sparse_linear_attention (SLA_ATTN backend won't serve)"
```

纯 Triton，不编 CUDA。构建：

```bash
docker build -f docker/Dockerfile.cuda \
  --build-arg SLA_COMMIT=main \
  -t <registry>/reputationly/vllm-omni:arm64-a100-<tag> .
```

**建议把 `SLA_COMMIT` 固定到具体 sha**，别用 `main`——上游改了 kernel 签名会让镜像行为漂移，而这类漂移只在出图质量上体现，不报错。

**出完镜像必须验一行**（构建机上即可）：

```bash
docker run --rm <新镜像> python3 -c "import sparse_linear_attention, triton; print('SLA ok', triton.__version__)"
```

装失败时构建**不会**中断（沿用仓里可选包的 best-effort 惯例），所以这一步不做就可能出一个"看着正常、起最快档必失败"的镜像。失败也不会静默降级——`SLA_ATTN` 在解析期直接拒绝启动（见 §5 的日志）。

---

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

### 3.2 最快档额外参数

在共同参数之外再加一行（**这是已实测的写法**）：

```
--diffusion-attention-config={"default": {"backend": "SLA_ATTN", "block_sparse": {"sparsity": 0.85, "start_step": 0}}}
```

GPUStack 的参数框若对 JSON 里的空格/引号处理不好，改用点号形式（文档支持，但我没在 GPUStack 上实测过，用了要按 §5 验日志）：

```
--diffusion-attention-config.default.backend=SLA_ATTN
--diffusion-attention-config.default.block_sparse.sparsity=0.85
--diffusion-attention-config.default.block_sparse.start_step=0
```

**注意 `--diffusion-attention-backend` 与它互斥**，两个一起给会直接报错。

最快档建议再加一个环境变量，把上界钉在 15 秒：

```
VLLM_OMNI_H3_MAX_OUTPUT_SECONDS=15
```

（引擎侧只能限上界，**下界 10 秒必须在门面层挡**，见 §4。）

---

## 4. new-api 配置

三档各配模型级 `defaultSteps`：**20 / 8 / 4**，与 `engine: minimax-h3` 同层。

这是实际生效的步数来源。不配的话请求不带步数，会跑 yaml 里的默认 20 步——加速档的加速全部丢失，而且不会有任何报错。

**最快档还要在门面层加两条**：

1. **时长白名单 10-15 秒**。引擎侧契约最小值是 5 秒，不会拒绝 5 秒请求；而 5 秒档 0.85 的稀疏率既不是最优速度也不是最优画质。
2. **确保 `aspect_ratio` 随请求下发**。历史上 `duration=6` 的 t2va 因为网关漏传这个字段全挂过，做时长白名单时把 10/15 两个值都过一遍。

全档位：门面层统一加 **−1 dBTP 限幅**。15 秒档实测五个档位音频全部削波，**满血基座也削**（+0.9 dBFS），这不是加速档引入的问题。

---

## 5. 上线后验证清单

**① 实例就绪**：`/v1/models` 返回 200。冷启动到就绪约 6-7 分钟。

**② 最快档必须看到这两行**（只认引擎自己打的日志，不认 tag/env）：

```
Resolved diffusion attention backend 'SLA_ATTN' for role='self' via attention_config.default
SLA_ATTN active: sparsity=0.85 (keeps 514 of 1716 key blocks), start_step=0, exempt_layers=0, rows=109795
```

第二行只在**第一次请求**时打。看不到它就是没走稀疏——最常见原因是镜像里没装 SLA 包（那样会在启动期直接失败，见下）或参数没生效。

看到下面这行是**正常**的，那是 token_refiner 的短序列自动退回稠密：
```
SLA_ATTN staying dense: 76 rows is 2 key blocks, under the 32-block threshold
```

**③ 缺包的表现是启动失败，不是降级**：
```
ValueError: SLA_ATTN requires the `sparse_linear_attention` package ... pip install git+https://github.com/thu-ml/SLA.git
```
这是有意设计——稀疏蒸馏的权重跑在稠密路径上仍会正常出片，只是慢一截且偏离训练分布，静默降级对调用方和监控都不可见。

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
