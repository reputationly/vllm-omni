# Qwen-Image-2.1-Turbo：模型差异、序列并行标定与 600 秒超时根因

2026-10-15/16 在 `newapi/dev-gpustack` A100-40G 集群上实测。本文回答四个问题：
**Turbo 比基线快多少、快在哪**；**加卡能否线性加速**；**10 图 i2i 会不会 OOM**；
以及一个顺带查出的**影响所有扩散请求的 600 秒超时墙**。

环境：4×A100-PCIE-40GB 节点，镜像 `arm64-a100-latest`
（digest `sha256:9ca7aa5aa12c9dc3a38fc85c553cd9a5461cf3c7728e5d3520652b2946ee3071`，
构建自 main `8c289f7b`），量化 `torchao_int8_weight_only`（= 权重 int8 / 激活 bf16，即 w8a16），
`--vae-use-tiling`，`--extras={"prefix_kv_cache_dtype":"fp8"}`。

---

## 1. 两个模型的体积与构成差异

| 组件 | 基线 Qwen-Image-2.1 | Turbo | 差异 |
| --- | --- | --- | --- |
| **总计** | 33,131,616,978 B（30.86 GiB） | 32,493,327,662 B（30.26 GiB） | **−638 MB（−1.9%）** |
| transformer | 14.23 GB（2 分片，BF16） | 14.23 GB（2 分片，BF16） | **字节数完全相同，md5 不同** |
| text_encoder | 17.53 GB（**4 分片**，BF16） | 17.53 GB（**单文件**，BF16） | **同一份权重重打包**（总差 1,096 B） |
| vae | 1.35 GB（**F32**，238 张量） | 675 MB（**BF16**，238 张量） | **fp32 → bf16，正好 1/2** |
| scheduler 配置 | `shift_terminal=0.02`、`use_dynamic_shifting=true` | `shift_terminal=null`、`use_dynamic_shifting=false` | **8 步档在这里** |

**三条要点**：

1. **transformer 同架构、同张量形状、同 dtype ⇒ 同字节数，但权值不同**（蒸馏改的是权重值）。
   这就是"同步数下两者一样快"的原因：每步算量完全相同。
2. **text_encoder 没有变**，只是从 4 分片重打包成 1 个文件。
3. **Turbo 的加速来自步数**（内置 8 步档 vs 基线默认 50 步），**不是每步更快**。
   README 与 recipe 都写明：Turbo 的调度档随 checkpoint 保存、自动加载，
   **传 `num_inference_steps` 会被覆盖**，且 **不要开 CFG**（蒸馏档跑 CFG=1）。

---

## 2. Turbo vs 基线：同步数对照（关键结论）

同 prompt、同 seed、1024²：

| 配置 | 步数 | 耗时 | 出图 |
| --- | --- | --- | --- |
| 基线 | **8** | 219.2 / 218.3 s | ❌ **糊成一团，未收敛** |
| **Turbo** | **8**（内置） | **218.9 / 217.9 s** | ✅ 干净 |

**同步数下耗时完全相同（差 0.1%）**——Turbo 不是"每步更快"。

换成各自推荐档：

| 分辨率 | 基线 | Turbo | 倍数 |
| --- | --- | --- | --- |
| 512² | 50 步，342.9 s | 8 步，55.8 s | **6.1×** |
| 1024² | 50 步，**跑不了**（撞 600 s 超时，见 §4） | 8 步，218.9 s | **从"不可用"到"可用"** |

**两个模型在同一配置内都逐位可复现**（同 seed 两次 md5 相同）。

---

## 3. 序列并行（`--ulysses-degree`）标定

### 3.1 加速比：准线性

| 分辨率 | degree | GPU | 耗时 | 加速比 | peak/GPU |
| --- | --- | --- | --- | --- | --- |
| 1024² | 1 | 1 | 858.1 / 857.8 s | 1.00× | 19,577 MiB |
| 1024² | 2 | 2 | 432.2 / 432.0 s | **1.99×** | 19,799 MiB |
| 1024² | 4 | 4 | 219.1 / 218.2 s | **3.92×** | 19,775 MiB |
| 2048² | 1 | 1 | 3432.2 s | 1.00× | 20,715 MiB |
| 2048² | 2 | 2 | 1720.8 s | **1.995×** | 19,837 MiB |
| 2048² | 4 | 4 | 869.7 s | **3.946×** | 19,813 MiB |

效率 ≥98%，且**不随分辨率衰减**。

### 3.2 耗时严格随像素数线性

| degree | 1024² | 2048² | 比值 |
| --- | --- | --- | --- |
| 1 | 858.1 | 3432.2 | **4.000×** |
| 2 | 432.2 | 1720.8 | **3.982×** |
| 4 | 219.1 | 869.7 | **3.970×** |

⇒ 可用于外推：4096²@4卡 ≈ 869.7 × 4 ≈ 3480 s（**但会撞 600 s 超时，必须先调 §4 的变量**）。

### 3.3 显存几乎不随分辨率与卡数变化（最实用的一条）

```text
1024²:  19,577 / 19,799 / 19,775 MiB
2048²:  20,715 / 19,837 / 19,813 MiB      ← 只多约 1 GB
```

**单卡 2048² 也只要 20.7 GB**（`--vae-use-tiling` 切了 VAE 解码，SP 切了激活）。
**⇒ 显存不是放大分辨率的瓶颈，时间是；而时间可以靠加卡线性解决。**

### 3.4 ⚠️ 换并行度会改变出图

同 prompt、同 seed，**只有 `--ulysses-degree` 不同**：

```text
1024²  degree=1  md5 78645f9c757214d02880901d7d4f1e32
       degree=2  md5 e87c84ba87a2922501205feda31edbe4
       degree=4  md5 615db0938f2992071d58edf215085da4
2048²  degree=1  md5 c92d242ac0a32e8eb7adc976a99cc907
       degree=2  md5 fe8427fb41c49da0f495003587a2a6bd
       degree=4  md5 81358003f5d825e73a2a2ed9b2037fc4
```

每个度数**自身**逐位稳定（各跑两遍，run1 == run2），**但跨度数不一致**。
⇒ **"加卡提速"在有同-seed 复现要求的场景下不是无损操作**；若只要求同配置内可复现，则无影响。

---

## 4. 600 秒超时墙（本次最有价值的发现）

### 4.1 根因

```python
# vllm_omni/diffusion/executor/multiproc_executor.py
_DLO_DP_WAVE_TIMEOUT_S = float(os.environ.get("VLLM_OMNI_DLO_DP_WAVE_TIMEOUT", 600.0))
```

**默认 600 秒**。超时后报
`RPC call to execute_model timed out` → `500`，**并且引擎会死**
（`EngineDeadError` / `Stage-0 has no live replica`，GPU 显存全部释放，容器仍在 Up）。

### 4.2 影响面：不只是 i2i

| 触发场景 | 观测耗时 | 结果 |
| --- | --- | --- |
| 基线 1024² @50 步 | 620.8 s | **500 + 引擎死** |
| 10 图 i2i @1024²（Turbo，4 卡 SP） | 621.8 s | **500** |
| 10 图 i2i（同配置 + 7200 s） | 1353.8 s | **200 ✅** |
| Turbo 1024²/2048² 各度数 | 最高 3432.2 s | **200 ✅** |

最后两行值得注意：**Turbo 的 SP 测试里 degree=1 跑 858 s、degree=2 跑 1720 s、
2048²/degree=1 跑 3432 s，全都 200 通过了**。说明这条超时**并非对所有路径生效**
（可能与 DLO / DP wave 的启用条件有关）。目前只确认：**i2i 与基线 t2i 会撞上它**。
**在没搞清适用条件之前，最稳妥的做法是所有 vLLM-Omni 扩散实例都显式设大这个值。**

### 4.3 集群内的现状（不一致）

| 模型 | `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` |
| --- | --- |
| `minimax-h3-fl2va` | **7200** ✅ |
| `minimax-h3-ref2va` | **7200** ✅ |
| `qwen-image-pro` | **未设（=600）** ← 本次已修，见 §4.4 |
| `qwen-image-base-*` / `fast-*` / `seedvr2` | 未设 |

### 4.4 已做的修改

已通过 GPUStack API 把 `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200` 写入 `qwen-image-pro`（id=60）的模型规格
（改前完整规格备份在 `/tmp/qwen_image_pro_spec_BACKUP.json`）。

**⚠️ 尚未生效**：GPUStack 只在**重建容器**时注入模型级环境变量，运行中的容器不会热更新。
已实测确认运行中的 `qwen-image-pro-75571-run-0` 里**没有**这个变量。
**要生效必须重启这三个实例**（届时每个约需 5–10 分钟）。

---

## 5. 10 图 i2i

`POST /v1/images/edits`，**多图走 `image` 字段（数组）**，不是 `reference_image`（那个是单张）。

| 运行 | 超时设置 | 结果 | peak/GPU | OOM |
| --- | --- | --- | --- | --- |
| 第 1 次 | 默认 600 s | ❌ 500，621.8 s | 35,357 MiB | 0 行 |
| 第 2 次 | 7200 s | ✅ 200，1353.8 s | 36,657 MiB | 0 行 |

**结论**：

* **10 张 1024² 图不会 CUDA OOM**（峰值 36.7 / 40 GB，`OOM 相关行数: 0`），
  失败的真因是 §4 的超时。
* **但余量只剩约 3.3 GB** —— 图更大或更多就会真的 OOM。
* 输入用的是 ffmpeg 生成的彩条图，所以**本次验证的是"10 图这条路能跑完"，不是画面语义质量**；
  要验质量需用真实图片。

---

## 6. 一次事故与恢复（记录在案）

测基线 1024² 时用了默认 50 步，620.8 s 后撞 §4 的超时墙，**把 `qwen-image-pro-75571`
的引擎打死了**（`/health` → 503，`EngineDeadError`，GPU 掉到 4 MiB）。

**GPUStack 不会自愈**：模型规格里虽有 `restart_on_error=True`，但容器本身是 "healthy"，
死的是里面的引擎，所以不触发。worker 日志无任何动作。

处理：`docker restart` 该容器，**约 2 分钟恢复 200**（比预估的 5–10 分钟快）。
`docker restart` 不动 GPUStack 的记录（容器 ID 不变），比删实例重建轻。

---

## 7. 复现物料

| 脚本 | 位置（本文写作时） | 作用 |
| --- | --- | --- |
| `bench_img.py` | `gpu25:/nfs-output/` | 图片端点压测（可指定尺寸/次数/步数，落 PNG + md5） |
| `sp_test.sh` / `sp2048.sh` | 同上 | SP 标定（起容器 → 等就绪 → 压测 → 采样 peak 显存 → 清理） |
| `i2i_test.py` | 同上 | 多图 i2i（`image` 数组） |
| `i2i_launch.sh` | 同上 | 起容器 + 跑 i2i + 采样显存 + OOM 检查 |

复现要点：

```bash
# 起一个 Turbo 实例（4 卡 SP）
docker run -d --name q21turbo --gpus all --network host --shm-size=32g --ipc=host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200 \
  -v /nfs-models:/nfs-models -v /nfs-data:/nfs-data -v /nfs-output:/nfs-output \
  <IMAGE>:arm64-a100-latest \
  vllm serve /nfs-models/wuhanjisuan894/models/Qwen-Image-2.1-Turbo --omni \
  --allowed-local-media-path /nfs-output \
  --ulysses-degree=4 --vae-use-tiling --stage-init-timeout=1800 \
  --diffusion-quantization-config='{"method":"torchao_int8_weight_only"}' \
  --extras='{"prefix_kv_cache_dtype":"fp8"}' --host 0.0.0.0 --port 8092
```

⚠️ **不要传 `num_inference_steps`**、**不要开 CFG**（Turbo 的 8 步档与 CFG=1 是内建的）。
⚠️ 就绪约 5–7 分钟；`--stage-init-timeout` 要留够。

---

## 8. 小结（可直接引用的结论）

1. **Turbo 与基线体积几乎一样**（差 1.9%）；差异在 **transformer 权值（蒸馏）**、
   **vae 精度（fp32→bf16）** 和 **scheduler 档**。
2. **Turbo 不快在每步**，快在 **8 步 vs 50 步**；同步数下两者耗时相同，
   且基线在 8 步时**不收敛**。
3. **SP 准线性**（≥98% 效率），**耗时随像素线性**，**显存几乎不变** ——
   放大分辨率应优先加卡而不是加显存。
4. **换并行度会改变出图**（同 seed 也不同）。
5. **600 秒超时墙**是本次最值钱的发现：默认值会静默杀死长请求并带走引擎；
   H3 早已设 7200，`qwen-image-pro` 漏了（已修，待生效）。
6. **10 图 i2i 不 OOM**，卡的是超时；余量仅 3.3 GB。
