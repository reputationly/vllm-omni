# MiniMax-H3 Turbo4 BF16 完整实验记录（2026-10-25 ~ 10-26）

> **结论：本次不上线。** BF16 对手权重在 4 步下幻影/重影明显，运行时路径再改善也压不住 15s 长片。
> 本文档记录全过程、踩到的问题、以及"烘焙 vs 动态 LoRA"两种接入方式的具体做法，供后续复用。

---

## 0. 一句话结论表

| 问题 | 结论 |
| --- | --- |
| 4 步能否上线 | **否**。5s 可用但偏软；15s 有双影。 |
| 幻影根因 | **主要是 BF16 烘焙丢了 LoRA 增量**（0.0625 缩放下只保住 ~42%），换成运行时注入后 5s 崩坏消失；15s 重影另有权重侧原因。 |
| v1.1 / v1.2 / v1.3 怎么选 | **v1.1（alpha=128 → scale 1.0）唯一可无损烘焙且 15s 干净**；v1.2/v1.3 必须走运行时，且仍不如 v1.1。 |
| 强度调参能救吗 | 不能。0.25 过曝、0.5 全崩，调强度只是换一种崩法。 |
| 上游是否有同样反馈 | 有，HF discussion #57；但他们是叠了 VEDA/SLA/额外 LoRA 才崩，我们裸跑也崩。 |

---

## 1. 环境

| 项 | 值 |
| --- | --- |
| 节点 | gpu9/gpu10/gpu14/gpu21/gpu22/gpu26/gpu27/gpu28 等，每实例 TP4 吃 4×A100-PCIE-40G |
| 驱动 | 580.65.06 |
| 镜像 | `…/vllm-omni:arm64-a100-latest`，digest `sha256:9ca7aa5a…3071`，构建 **2026-10-15**，12.9 GB |
| 运行时 | Python 3.12.3 / torch 2.13.0+cu130 / CUDA 13.0 / vLLM 0.31.0 |
| 存储 | `/nfs-models`(→`/nfs-data`) 、`/nfs-output` |

**注意**：仓库里 `repo/` 的旧源码（8-18）与镜像内代码（10-15）**不能混挂**，
`ImportError: cannot import name 'uses_diffusers_adapter'` 立即崩。旧文档里的干净 turbo 产物在当前镜像上无法复现。

---

## 2. 实验流程（时间线）

### 2.1 三臂基础对比（10-25 晚）
- 4 步 / 8 步 / 现网（`MiniMax-H3-Singularity-VDN-INT8` + `curve-fl2va-baked2`）。
- 热态 4 卡墙钟：5s → 4 步 58–60 s、8 步 92–95 s、现网 104–107 s；15s → 223 / 373 / 296 s。
- 预热：4 步需 3 条、8 步 2–3 条、现网首条即热；**规则：前 3 条丢弃**。

### 2.2 场景矩阵（10-25 夜）
- 10 个内容场景 + 3 种比例（竖屏/宽幕/方形）+ 冷热分离，共 72 次运行。
- 比例切换额外开销 ≤3 s（无重编译）。

### 2.3 E0：量化烘焙损失（纯 CPU）
复刻 `bake_turbo_lora.py` 的 qkv 交错与 SwiGLU 重排，逐层用 fp32 算 `ΔW_ideal` 与 `ΔW_baked`：

| 变体 | scale | 保留比（中位） | 余弦（中位） |
| --- | ---: | ---: | ---: |
| v1.1 | 1.0 | **1.012** | 0.887 |
| v1.2 | 0.0625 | **0.359** | 0.270 |
| v1.3 | 0.0625 | **0.423** | 0.319 |
| 8 步 v1.1 | 0.0625 | 0.400 | 0.301 |

相对增量幅度 `‖ΔW‖/‖W‖`：v1.1 1.84e-3、v1.3 1.25e-4、v1.2 9.25e-5。
**BF16 的 ulp ≈ 0.39%** —— v1.3 的增量只有 0.03 个 ulp，烘焙等于把增量加进粗网格再取整。

### 2.4 E1：同引擎只换权重（盲评）
三台同镜像同 yaml，只换 v1.1 / v1.2 / v1.3，4 用例 × 2 seed，盲评三联图（行序随机、只标 A/B/C）：

| 用例 | v1.1 | v1.2 | v1.3 |
| --- | --- | --- | --- |
| neon s42/s43 | 干净 | 干净 | **崩（竖纹/糊）** |
| sports s42 | 干净 | 干净 | **崩（橙斑）** |
| p1 15s s42 | 干净 | 10s 三重影 | 早晚期重影 |
| p4（阴性对照） | 干净 | 干净 | 发虚 |

### 2.5 强度扫描（A）
把 v1.3 按 scale 0.25 / 0.5 重烘（保真度 0.77 / 0.94）：
霓虹 0.25 找回结构但过曝、0.5 全崩成噪点；运动 0.25 比 0.0625 更碎。
**强度不是单调轴，调参救不回来。**

### 2.6 运行时 LoRA（关键一格）
基座 + 请求级 LoRA（零烘焙损失），scale 1.0 / 2.0：
- **1.0：5s 崩坏消失**，正常街景/球场，偏软；两个 seed 都成立。
- 2.0：过曝/碎裂（与上游"2.0 更稳"不一致）。
- **15s：s42 在 10 s 仍有明显双影**，s43 基本单人 → 长片重影是 seed 相关的，没被修好。

### 2.7 运行时 v1.1 vs v1.3 的耗时/资源

| 路径 | 5s | 15s（热） | 峰值 GPU/卡 | 峰值宿主 |
| --- | ---: | ---: | ---: | ---: |
| v1.1 烘焙（1.0） | 58–60 s | **223 s** | 19.7 / 25.9 GB | 185.7 GiB |
| v1.3 运行时（1.0） | 61 s | **258 s** | **30.9 GB** | 133.0 GiB |
| v1.1 运行时（1.0） | **不可用**（worker 挂死） | — | — | — |

---

## 3. 踩到的问题（按严重度）

### 3.1 BF16 烘焙吃掉 LoRA 增量（根因级）
`W' = round_bf16(W + scale·B@A)`，当 `‖ΔW‖/‖W‖ ≪ ulp` 时绝大多数元素回到原值。
实测 v1.3 只有 4.6% 的元素变化。`max_abs_error=0` **不能**证明增量保住——它只证明"烘焙结果 == 本次有损舍入"。

### 3.2 `--partition-out` 元数据缺陷（已修）
旧实现把 base 分区的**所有**文件软链过去，包括 `model_index.json`：
出包分区读到基座契约（`shift 12`），蒸馏要求的 `shift 6` 被静默丢弃；而且软链导致后续任何元数据修改都会**穿透写回基座**。
修复见 §4.4。

### 3.3 我自己造成的基座损坏（已修复并核验）
手工补 s025/s05 元数据时用 `json.dump(open(path,"w"))`，穿透软链把
`MiniMax-H3/FL2VA/model_index.json` 覆盖成 s05 的内容。已按同族参考还原
（`shift 12/3`、无 `base_schedule`、无 `distilled`），并把 5 个分区改成真实文件。

### 3.4 存量分区 shift 判错（已修 1 个）
引擎 `lora.py:44-45` 明确：**768p → shift 6，544p → shift 12**。
据此复核 5 个分区，只有 `MiniMax-H3-FL2VA-Turbo4-768p-BF16` 错（12 应为 6），已改。

### 3.5 动态 LoRA 禁模型级 offload
`--enable-cpu-offload` 与动态 LoRA 冲突：
`RuntimeError: MiniMax-H3 Turbo dynamic LoRA does not support model-level CPU offload`。
关掉后基座常驻 GPU（单卡 29.5 GB），只能再开 `VLLM_OMNI_H3_OFFLOAD_COMPONENTS=1` 省回 ~10 GB。

### 3.6 `--lora-backend distill` 不被 H3 支持
`Pipeline MiniMaxH3Pipeline does not support loading distilled LoRA weights`。
H3 的 Turbo 适配器是**请求级动态 LoRA**（`_load_diffusion_lora_adapter` → `load_minimax_h3_turbo_lora`）。

### 3.7 `VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0` 会把请求永久挂起
表现为 GPU 0% 占用、无任何报错、任务永远 PENDING。设成有限值（如 900）后会明确失败。

### 3.8 同一实例内换适配器会挂死 worker
v1.3 → v1.1 切换后 `RPC call to execute_model timed out`（630 s），随后 `OmniEngineDeadError`。
**必须一实例一适配器。**

### 3.9 v1.1 运行时本身挂死（未定位）
全新进程、v1.1 作为首个适配器，预热仍 630 s 超时；v1.3 同配置 6 条全过。
两个文件结构完全一致（624 张量、同 key_format），差别只有 alpha 128 vs 8 → 嫌疑在 scale=1.0 的适配器应用路径。

### 3.10 容器僵尸导致 `docker rm -f` 失败
`PID is zombie and can not be killed`。加 `--init` 启动可避免；临时处理是等僵尸被回收后重试。

### 3.11 早期把"冷态"当"热态"下过结论
第一次 15s 报"现网更快"，实际是冷请求。**规则：前 3 条丢弃后再计时。**

### 3.12 单位错误
曾用"卡·秒"（×4）表达，实际应统一为"4 卡整机墙钟秒"。

---

## 4. 两种接入方式怎么做

### 4.1 选型判据（先算，再动手）

```
BF16 ulp ≈ 2^-8 ≈ 0.39%（相对）
ΔW_rel = ‖scale·B@A‖ / ‖W‖       # 用 tools/minimax_h3/compare_lora_deltas.py 或 E0 脚本量
```

| 条件 | 选择 |
| --- | --- |
| `ΔW_rel ≳ 1e-3`（约 0.5 ulp 以上） | **烘焙**：无损、推理零开销 |
| `ΔW_rel ≲ 1e-4`（不足 0.1 ulp） | **动态 LoRA**：烘焙会丢 50–65% 增量 |
| 需要频繁改强度 | 动态 LoRA（烘焙每档都要重出 62 GB） |

### 4.2 离线烘焙

```bash
python3 tools/minimax_h3/bake_turbo_lora.py \
  --base   /nfs-models/wuhanjisuan894/models/MiniMax-H3/FL2VA/transformer \
  --lora   /nfs-models/wuhanjisuan894/models/MiniMax-H3-Turbo-LoRA/<lora>.safetensors \
  --output <partition>/transformer \
  --partition-out <partition> \
  --contract-from <该 LoRA 所属发布版的 model_index.json>   # 基座非蒸馏时必填
```

要点：

- **`--output` 直接落在 `<partition>/transformer`**，这样 transformer 是实体目录，不会被软链出去。
- `--contract-from` 必须指向**该 LoRA 所属版本**的 `model_index.json`，否则会拒绝出包（这是有意为之，避免静默用错 shift）。
- 组件默认**实体拷贝**（自包含，约 +73 GB/分区）；`--symlink-components` 可退回软链省空间。
- 输出必须核对：`shift video/audio`、`sigma points`、`effective scale`，并确认 `model_index.json` 不是软链。
- 校验统计：日志里的 `‖delta‖/‖W‖` 中位数，若远低于该 LoRA 应有的相对幅度，说明 scale 取错。

**完成后的自检**

```bash
python3 - <<'PY'
import json,os,sys
p=sys.argv[1]+"/model_index.json"
m=json.load(open(p))["_minimax_h3"]
print("symlink:", os.path.islink(p), "shift:", m["sigma_shift_scales"],
      "pts:", len(m.get("base_schedule") or []), "scale:", (m.get("distilled") or {}).get("effective_lora_scale"))
PY <partition>
```

### 4.3 动态 LoRA（运行时注入）

**起服务**（基座 + 不带模型级 offload）

```bash
docker run -d --name <name> --init --gpus all --ipc=host --network=host --shm-size 64g \
  -v /nfs-models:/nfs-models -v /nfs-output:/nfs-output \
  -e VLLM_OMNI_H3_OFFLOAD_COMPONENTS=1 \
  -e VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=900 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  <image> vllm serve /nfs-models/wuhanjisuan894/models/MiniMax-H3/FL2VA --omni \
  --allowed-local-media-path /nfs-output --init-timeout 2400 --stage-init-timeout 2400 \
  --host 0.0.0.0 --port <port> --trust-remote-code
```

**请求体**（关键四个字段）

```json
{
  "prompt": "...",
  "num_inference_steps": 4,
  "seed": 42,
  "extra_params": {"flow_shift": 6, "audio_flow_shift": 3},
  "lora": {"name": "turbo4-v1.3", "path": "/nfs-models/.../minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors", "scale": 1.0}
}
```

注意事项：

- **一个实例只挂一个适配器**；中途换适配器会把 worker 挂死（§3.8）。
- 不能带 `--enable-cpu-offload`（§3.5）；代价是显存从 ~19.7 GB 涨到 ~30.9 GB。
- `flow_shift` 由请求传入；742 行那套 TurboSpec 校验只在文件名能被 `parse_turbo_filename` 识别时生效，
  未被识别时适配器仍会被应用，但**不会打印 `Turbo adapter` 日志**——排查时不要以"没有该日志"判定未生效，
  用"同 seed 有/无 LoRA 两次输出的 sha256 是否不同"判定。
- 已知缺陷：**v1.1（alpha=128）走这条路径会挂死 worker**（§3.9），v1.3（alpha=8）正常。

### 4.4 已修复的工具缺陷（`tools/minimax_h3/bake_turbo_lora.py`）

| 修复项 | 行为 |
| --- | --- |
| `--contract-from` | 显式指定蒸馏契约来源（shift / sigma 网格） |
| 契约缺失 | 基座非蒸馏且未给契约时**拒绝出包**，不再静默继承 |
| `model_index.json` | 一律写真实文件；目标处已有软链先 `unlink`，杜绝穿透 |
| 组件拷贝 | 默认实体拷贝，`--symlink-components` 可选软链 |
| `effective_lora_scale` | 自动写入实际融合 scale |
| 回归测试 | `tests/tools/minimax_h3/test_bake_turbo_lora_partition.py`（6 例，含"已有软链不得穿透"） |

---

## 5. 决策

**本次不上线。** 理由：

1. 5s 虽可用，但**偏软、噪点高于 v1.1**；运动场景 seed 42 仍有前景糊化。
2. 15s 存在**双影幻影**（s42 第 10 秒明显），运行时也压不住。
3. 上线需接受：+16% 长片耗时、+11 GB 显存、一实例一适配器、且 v1.1 运行时路径不可用。

若将来要重新评估，用 `docs/MiniMax-H3-准入门槛用例集.md` 里的用例集跑一遍门槛，全过再谈。

---

## 6. 产物索引

| 路径 | 内容 |
| --- | --- |
| `/nfs-output/h3_turbo_eval/results/` | 全部运行（`t4v13_*`/`t8v11_*`/`prodv2_*`/`e1v*`/`ea0*`/`rt13_*`），含 `output.mp4`、`request.json`、`summary.json`、`samples.jsonl` |
| `/nfs-output/h3_turbo_eval/rt13_videos/` | v1.3 运行时成片软链目录（6 条）+ README |
| `/nfs-output/h3_turbo_eval/e0/` | E0 逐层数据、强度扫描模拟 |
| `/nfs-output/h3_turbo_eval/e1_panels/`、`ea_4way/`、`rt_4way/`、`rt_vs_baked/` | 盲评拼图与映射 |
| `/nfs-output/h3_turbo_eval/compare_turbo_4v8/` | 三臂软链对比树 |
| `docs/实验报告/附件/h3-turbo-4v8/` | 仓库内关键图副本 |
| 相关文档 | `MiniMax-H3-Turbo-BF16-4步vs8步与现网对比-2026-10-25.md`、`MiniMax-H3-Turbo-E0E1-烘焙损失与换权重实测-2026-10-25.md`、`MiniMax-H3-Turbo-BF16-测试记录与产物清单-2026-10-25.md` |
| 准入用例 | `docs/h3-gate-cases/`（JSON）+ `docs/MiniMax-H3-准入门槛用例集.md`（规范） |
