# MiniMax-H3 Turbo LoRA 接入手册

> 面向执行者（人或 agent）。目标是：**上游发布一份新的 Turbo 步数蒸馏 LoRA 之后，
> 不重新构建镜像、不新增 deploy-config，把它变成 GPUStack 上一个可用的加速档。**
>
> 最近一次执行：2026-08-15，ref2v 4-step v0.1，已烘焙、未压测（见 §5）。

---

## 1. 先记住三条，其余都是细节

1. **不需要重新出镜像。** deploy-config 确实随镜像固化（`docker/Dockerfile.cuda`
   把整个 `deploy-configs/` COPY 到 `/deploy-configs/`），但**新的 Turbo 档不需要
   新的 yaml** —— 理由见 §4。
2. **融合缩放必须逐份读，不能按步数推。** 已知 `fl2v_8step_v1.0` 与
   `ref2v_4step_v0.1` 都是 0.0625，而同为 4 步的 `fl2v_4step_v1.0_768p` 是 1.0，
   差 16 倍。用错的产物能正常加载、只是出垃圾。
3. **基座别拿错。** `MiniMax-H3/FL2VA/transformer` 与 `MiniMax-H3/Ref2VA/transformer`
   各 62 GB、逐文件 sha 全不同。fl2v 的 LoRA 只能融进 FL2VA，ref2v 的只能融进 Ref2VA。

---

## 2. 三步流程

### 2.1 下载 LoRA

清单在 `scripts/download_minimax_h3_turbo_lora.sh` 的 `MANIFEST` 里，格式：

```
文件名|sha256|字节数|HF仓库|魔搭仓库|档位(core|compare|all)
```

sha256 与字节数从魔搭 API 取（脚本用 curl 逐字节校验，不用 `snapshot_download`——
后者只保证「下完了」，一份被 CDN 截断的 1.4 GB LoRA 要到出图才发现）：

```bash
curl -s "https://modelscope.cn/api/v1/models/lightx2v/Minimax-h3-Turbo/repo/files?Revision=master&Root=" \
  | python3 -c "
import json,sys
for f in (json.load(sys.stdin).get('Data') or {}).get('Files') or []:
    if f.get('Name','').endswith('.safetensors'):
        print(f\"{f['Name']}|{f.get('Sha256','')}|{f.get('Size','')}\")"
```

只收原生格式，**不要 `*_comfyui_bf16.safetensors`**（1.96 GB 那些）——扁平打包、
无 config/index，烘焙工具读不了。它唯一的用处是交叉验证：8step 那份的 ComfyUI 变体
metadata 里写死了 `training_scale=0.0625`。

加完清单在管理节点执行：

```bash
tmux new -s dl_h3turbo -d 'bash /root/download_minimax_h3_turbo_lora.sh'
tail -f /tmp/dl_minimax_h3_turbo_lora.log
```

**下完必须看这两行**（脚本自动打）：

```
xxx.safetensors: 624 tensors, blocks=50/50, refiner=2/2
    rank=128  alpha=8  融合缩放=0.0625
```

`blocks` 不足 50 说明 LoRA 只改了一部分主干，脚本会直接报错退出（H3 是
packed-sequence 联合 DiT，音频 token 和视频 token 过同一批 blocks，覆盖不全会让
音视频行为不一致）。**融合缩放这个数要记下来**，它是 §2.2 的验收基准。

### 2.2 烘焙进基座

vLLM-Omni 的 H3 pipeline **没有运行时 LoRA 挂载点**，所以是离线合并成一份 drop-in
的 transformer 目录。工具：`tools/minimax_h3/bake_turbo_lora.py`。

先空跑（不读权重，几秒出）：

```bash
python tools/minimax_h3/bake_turbo_lora.py \
  --base <FL2VA 或 Ref2VA>/transformer \
  --lora /nfs-data/models/MiniMax-H3-Turbo-LoRA/<新 LoRA>.safetensors \
  --dry-run
```

`--self-test` 会拿参考实现校验三个张量变换（key 重命名 / QKV per-head 交错 /
SwiGLU 半序），同样不碰权重。

正式烘（纯 CPU，**GPU 帮不上**：208 个线性层、rank 128 的矩阵乘只有 ~700 GFLOP
量级，瓶颈在读 62 GB + 写 62 GB 的 NFS I/O；工具里 `safe_open(device="cpu")`
也是写死的）：

```bash
python tools/minimax_h3/bake_turbo_lora.py \
  --base   <基座>/transformer \
  --lora   <新 LoRA>.safetensors \
  --output /nfs-data/models/<产物名>/transformer \
  --partition-out /nfs-data/models/<产物名>
```

耗时参考：2026-08-15 在计算节点用 lightx2v 镜像跑，13 个分片约 **7 分钟**。

**验收看这行**：

```
patched 208 tensors
||delta|| / ||W|| :  min=0.0000  median=0.0002  max=0.0019
```

工具的判据是「中位数远高于 ~1.0 通常意味着缩放用错了」。中位数落在 1e-4 量级是正常的
（LoRA 本就是小扰动）。产物目录应包含 `transformer / text_encoder / video_vae /
audio_vae / processor / tokenizer / model_index.json`。

> 要不要再量化成 INT8：看那条链路线上跑什么。FL2VA 的 Turbo8 档**线上就是 BF16**
> （`deploy-configs/minimax_h3_turbo8_bf16_a100_40g.yaml`，与 INT8 原始版成对同时在线），
> 所以 BF16 烘完可直接起。代价是显存：BF16 权重 62 GB，FL2VA Turbo8 实测 480p/209 帧
> 29.8 GiB/卡、768p/345 帧 36.66 GiB/卡（余量仅 2.8 GiB）。排不下再考虑量化。

### 2.3 在 GPUStack 上起实例

**模型路径**：填 §2.2 的产物目录。

**后端参数**：

```
--omni
--trust-remote-code
--deploy-config=/deploy-configs/<复用的现成 yaml>
--init-timeout=2400
--stage-init-timeout=2400
```

yaml 按 task 选现成的，**不要新建**：

| task | 复用哪份 |
|---|---|
| fl2va / t2va | `/deploy-configs/minimax_h3_a100_40g.yaml`（或 turbo8 那份，两者仅默认步数不同） |
| ref2va | `/deploy-configs/minimax_h3_ref2va_bf16_a100_40g.yaml` |

**环境变量**（六个一个都不能少）：

```
VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=7200
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200
VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0
VLLM_OMNI_H3_OFFLOAD_DIT_BEFORE_VAE=1
VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK=8
VLLM_OMNI_H3_LOG_STEP_MEMORY=1
```

**new-api 侧**：给该模型配模型级 `defaultSteps`（4 或 8）。这是实际生效的步数来源，见 §4。

---

## 4. 为什么不需要新 yaml、不需要重新出镜像

三件事各自独立，凑在一起就得出了这个结论：

1. **权重路径不在 yaml 里。** yaml 只描述 pipeline / stage 拓扑 / 并行度，权重由
   `vllm serve <路径>` 给。换权重不用换 yaml。
2. **步数是请求级的，yaml 里那个只是兜底默认值。**
   - new-api：`relay/channel/task/gpustackplus/adaptor.go:481` 按模型读 `defaultSteps`
     并写进请求体的 `num_inference_steps`（测试 `TestH3StepsFollowModelConfig` 钉住）
   - vLLM-Omni：`entrypoints/openai/serving_video.py:201-202` —— 请求带了就覆盖默认值
3. **同 task 的 turbo 档与基座档，yaml 除步数外逐字段相同。**（2026-08-15 用脚本比对
   `minimax_h3_ref2va_bf16` 与当时试建的 turbo4 版，除 `num_inference_steps` 外完全一致，
   遂删除后者。）

**真正需要新 yaml 的只有一种情况：并行度或 stage 拓扑要变**（tp 数量、vae 分块策略、
attention backend 之类）。步数、权重、LoRA 版本都不算。

### 4.1 两样东西挪不进 yaml，别试

（已按代码核过，见 `docs/实验报告/MiniMax-H3-GPUStack-生产部署档.md` §3.2）

1. **六个 `VLLM_OMNI_*` 环境变量**：stage 级 `env:` 会被解析进 `StageDeployConfig.env`
   然后原地丢掉，生产路径不读它。必须配在 GPUStack 的环境变量里。
2. **`--init-timeout` / `--stage-init-timeout`**：既不是 `StageDeployConfig` 字段也不是
   pipeline-wide 字段，写进 yaml 会落进 `engine_extras` 再被过滤掉。

另外若真要新建 yaml：**并行度必须写成嵌套 `parallel_config:` 块**。平铺写
`tensor_parallel_size: 4` 会被静默丢掉，起来就是单卡。

---

## 5. 当前状态（2026-08-15）

| 档位 | LoRA | 缩放 | 基座 | 产物 | 状态 |
|---|---|---|---|---|---|
| fl2va 480p | `fl2v_turbo_8step_v1.0` | 0.0625 | FL2VA | `MiniMax-H3-FL2VA-Turbo8-BF16` | 线上，实测 1.76x |
| fl2va 768p | `fl2v_turbo_4step_v1.0_768p` | 1.0 | FL2VA | `MiniMax-H3-FL2VA-Turbo4-768p-BF16` | 已烘 |
| **ref2va** | **`ref2v_turbo_4step_v0.1`** | **0.0625** | **Ref2VA** | **`MiniMax-H3-Ref2VA-Turbo4-BF16`** | **已烘，未压测** |

ref2v 这份的烘焙自检：`patched 208 tensors`，`||delta||/||W||` 中位数 0.0002、
最大 0.0019（最大值在 `blocks.49.attn.qkv_proj.weight`）。

**未验证项，上线前必须自己测，别照搬 fl2v Turbo8 的结论**：

- **步数与画质**：这是 **v0.1**（fl2v 两份已是 v1.0）。4 步的可用窗口、过锐阈值都没验过。
  fl2v Turbo8 的经验是「超过标定步数即开始过锐，20 步明显劣化」，4 步版只会更窄。
- **显存**：ref2va 要吃 ≤9 图 / ≤3 视频 / ≤3 音频的混合参考，序列比 fl2va 长，
  而 fl2va Turbo8 在 768p/345 帧已经只剩 2.8 GiB 余量。长输入必须单独压测。
- **音频**：现有 e2e 精度用例只校验 aac/32000Hz/2ch 这些容器属性，音频内容一个字节
  都没比，测不出蒸馏带来的底噪或声像退化，要单独 A/B。
- 768p 另有既存问题 #37（同 seed 产物不确定），做 A/B 时别把它的抖动算到 LoRA 头上。

---

## 6. 下一次：ref2v v1.0 8-step 出来时怎么做

上游（`lightx2v/Minimax-h3-Turbo`）目前 ref2v 只有 v0.1 4-step。等 v1.0/ 8-step 发布：

1. **拿校验值**：跑 §2.1 那段 curl，取新文件的 sha256 与字节数。
2. **加清单**：在 `download_minimax_h3_turbo_lora.sh` 的 `MANIFEST` 里加一行，档位标
   `core`。同时**把 `SET` 档位说明里的数量与总量改对**（现在是「三个生产候选，4.2 GB」）。
3. **下载并记下融合缩放**：绝不能假设它跟 v0.1 一样是 0.0625 —— fl2v 那边 v0.1→v1.0
   就出现过 alpha 从 8 变 128。以脚本打印的为准。
4. **烘焙**：基座是 `MiniMax-H3/Ref2VA/transformer`（不是 FL2VA），产物建议命名
   `MiniMax-H3-Ref2VA-Turbo8-BF16`。验收看 `||delta||/||W||` 中位数是否在 1e-4 量级。
5. **起实例**：GPUStack 新建，模型路径指向新产物，`--deploy-config` 仍然复用
   `/deploy-configs/minimax_h3_ref2va_bf16_a100_40g.yaml`，六个环境变量照配。
   **不要重新构建镜像。**
6. **new-api 配 `defaultSteps: 8`**（模型级，与 `engine` 同层）。这一步不做的话，
   请求不带步数，实际会跑 yaml 里的默认 20 步，加速全丢。
7. **压测**：至少覆盖 §5 那四个未验证项，尤其 ref2va 的长参考输入显存。

如果哪天上游改了 LoRA 的结构布局（key 命名、QKV 排布、SwiGLU 半序），`--dry-run`
会在断言处直接失败而不是默默产出垃圾 —— 那时才需要动 `bake_turbo_lora.py`。

---

## 7. 相关文件

| 用途 | 路径 |
|---|---|
| 下载脚本（含清单与结构校验） | `scripts/download_minimax_h3_turbo_lora.sh` ⚠️ 见下 |
| 烘焙工具 | `tools/minimax_h3/bake_turbo_lora.py` |
| 生产部署档（env / flag / 为什么是这个值） | `docs/实验报告/MiniMax-H3-GPUStack-生产部署档.md` |
| Turbo8 压测数据 | `docs/实验报告/vLLM-Omni-MiniMax-H3-Turbo8-480p-768p-压测对比报告.md` |
| 镜像构建（deploy-configs 固化位置） | `docker/Dockerfile.cuda` |

> ⚠️ **`scripts/` 整个目录在 `.gitignore` 里（`.gitignore:184`），下载脚本不进版本库。**
> 也就是说本文档描述的清单行、基座提示、结构校验那些改动，**只存在于工作区和已经
> scp 到节点的副本**（管理节点 `/root/download_minimax_h3_turbo_lora.sh`）。换机器
> 或换人接手时脚本可能已经不在，届时按 §2.1 的清单格式和 curl 取校验值的方法重建即可
> —— 这也是本文档要把那两段写全的原因。
