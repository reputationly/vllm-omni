# MiniMax-H3 Turbo LoRA 接入手册

> 面向执行者（人或 agent）。目标是：**上游发布一份新的 Turbo 步数蒸馏 LoRA 之后，
> 不重新构建镜像、不新增 deploy-config，把它变成 GPUStack 上一个可用的加速档。**
>
> 最近一次执行：2026-09-04，上游同时给两条链路各升了一代 —— ref2v **8-step v1.0 768p**
> 已下载/烘焙/组装（§5.2），fl2v 4-step **v1.2** 768p 已下载待烘（§5.3）。

---

## 1. 先记住三条，其余都是细节

1. **不需要重新出镜像。** deploy-config 确实随镜像固化（`docker/Dockerfile.cuda`
   把整个 `deploy-configs/` COPY 到 `/deploy-configs/`），但**新的 Turbo 档不需要
   新的 yaml** —— 理由见 §4。
2. **融合缩放必须逐份读，任何规律都别信。** 曾经有过一个看着成立的规律（「8 步都是
   alpha 8、4 步 768p 都是 alpha 128」），2026-09-04 的 `fl2v_4step_v1.2_768p` 直接把它
   打掉了：4 步、768p、alpha 8 → **0.0625**，而它要替代的 `v1.1_768p` 是 alpha 128 → 1.0。
   同目录、同任务、同分辨率、同步数、只差一个小版本号，缩放差 16 倍。
   现状（全部逐份读过 header）：`fl2v_8step_v1.0` / `fl2v_8step_v1.0_768p` /
   `fl2v_4step_v1.2_768p` / `ref2v_4step_v0.1` / `ref2v_8step_v1.0_768p` 都是 0.0625，
   只有 `fl2v_4step_v1.0_768p` 与 `v1.1_768p` 是 1.0。用错的产物能正常加载、只是出垃圾。
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

### 2.3 验证融合并组装 partition

正式目录必须由组装工具生成，并传入**真实 LoRA 文件**；只写一个来源文件名不再允许：

```bash
python tools/minimax_h3_turbo/assemble_distilled_partition.py \
  --base-partition <MiniMax-H3/FL2VA 或 Ref2VA> \
  --fused-transformer <已烘焙目录>/transformer \
  --output <正式的-vLLM目录> \
  --num-inference-steps <4或8> \
  --video-shift <6或12> \
  --audio-shift 3 \
  --source-lora <LoRA文件名> \
  --lora-checkpoint <LoRA绝对路径>
```

组装前会重新读取 checkpoint 的 SHA256、rank 和 alpha，并对 208 个 native target
逐个取确定性行，覆盖全部 312 对 A/B 因子。只有采样值逐个满足
`BF16(base + (alpha/rank) × B@A)` 才创建输出目录。错误倍率、错误基座、错误 LoRA、
`source_lora` 名称错配或缺失 alpha 都会失败，不能再生成“能加载但系统性偏移”的正式模型。

通过后，`model_index.json` 的 `_minimax_h3.distilled` 会记录：

- `source_lora_sha256 / lora_rank / lora_alpha / effective_lora_scale`；
- 验证方法、目标数、因子对数、采样值数、变化值数和最大误差；
- base/fused 的 index SHA256 与采样张量 SHA256。

旧 partition 可用 `tools/minimax_h3_turbo/lora_provenance.py --write-model-index`
在相同验证通过后原子回填；不得手工补数字。

### 2.4 在 GPUStack 上起实例

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

## 5. 当前状态（2026-09-04）

| 档位 | LoRA | 缩放 | 基座 | 产物 | 状态 |
|---|---|---|---|---|---|
| fl2va 480p | `fl2v_turbo_8step_v1.0` | 0.0625 | FL2VA | `MiniMax-H3-FL2VA-Turbo8-BF16` | 线上，实测 1.76x |
| fl2va 768p | `fl2v_turbo_4step_v1.0_768p` | 1.0 | FL2VA | `MiniMax-H3-FL2VA-Turbo4-768p-BF16` | 已烘，作 v1.1 的 A/B 基线 |
| fl2va 768p | `fl2v_turbo_4step_v1.1_768p` | 1.0 | FL2VA | `MiniMax-H3-FL2VA-Turbo4-768p-v1.1-BF16-vLLM` | 现网在跑 |
| fl2va 768p | `fl2v_turbo_8step_v1.0_768p` | 0.0625 | FL2VA | `MiniMax-H3-FL2VA-Turbo8-768p-BF16-vLLM` | 已烘已组装 |
| **fl2va 768p** | **`fl2v_turbo_4step_v1.2_768p`** | **0.0625** | **FL2VA** | **`MiniMax-H3-FL2VA-Turbo4-768p-v1.2-BF16-vLLM`** | **2026-09-04 新增，已烘已组装，未压测** |
| ref2va | `ref2v_turbo_4step_v0.1` | 0.0625 | Ref2VA | `MiniMax-H3-Ref2VA-Turbo4-BF16` | 已烘，未压测 |
| **ref2va** | **`ref2v_turbo_8step_v1.0_768p`** | **0.0625** | **Ref2VA** | **`MiniMax-H3-Ref2VA-Turbo8-768p-BF16-vLLM`** | **2026-09-04 新增，已烘已组装，未压测** |

### 5.1 v1.1 768p（2026-08-20 执行记录）

上游 08-20 07:35 UTC 只传了这一份权重的两种格式，**GitHub README 与 HF/魔搭提交信息都没有
任何 release note**，别对外引用一个不存在的 changelog。权重层面能确认的是：结构与 v1.0_768p
完全同构（624 tensors / blocks 50 / refiner 2 / rank 128 / alpha 128 → 缩放仍是 1.0），
312 对 A/B 因子逐个比过 ΔW=B@A，与 v1.0 的余弦中位数 0.885（p25 0.858 / min 0.608）、
范数中位比 1.158，**没有一层是原样搬过来的**。方向一致、幅度普遍变大 ~16%，是同配方继续
训练出的新检查点，不是换结构或换 recipe。

执行在 0036（当时 GPU 全空、239 GB 可用内存），vllm-omni 镜像里跑，烘焙 62 GB→62 GB 约
5 分钟。验收数字：

```
patched 208 tensors
||delta|| / ||W|| :  min=0.0006  median=0.0015  max=0.0069
fusion_verification: max_abs_error=0.0  verified_target_tensors=208
                     verified_factor_pairs=312  changed_values=3311777   （v1.0 同位置是 2960753）
sigma_shift_scales: video=6.0 audio=3.0   base_schedule: [1, .75, .5, .25, 0]
```

两个执行上的坑，重跑时照做：

1. **烘焙不要传 `--partition-out`。** 它顺手生成的可服务目录 shift 写死 12/3，而 768p 这档
   应是 6/3 —— v1.0 那次就在 `MiniMax-H3-FL2VA-Turbo4-768p-BF16/model_index.json` 留下了
   这么一份没有 `distilled` 块、shift 还是 12/3 的误导性目录（至今还在盘上，别拿它起服务，
   正式目录是同名带 `-vLLM` 后缀那个）。正式 partition 只由 `assemble_distilled_partition.py` 出。
2. **组装工具必须在仓库目录结构下跑。** `lora_provenance.py` 里 `from tools.minimax_h3.bake_turbo_lora
   import build_plan` 是绝对包路径，把三个 .py 拍平拷到同一个目录会 `ModuleNotFoundError:
   No module named 'tools'`（烘焙工具本身没这个依赖，所以只有第 3 步会挂）。节点上要摆成
   `<root>/tools/minimax_h3/` + `<root>/tools/minimax_h3_turbo/`，`cd <root>` 再跑。

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

### 5.2 ref2v 8-step v1.0 768p（2026-09-04 执行记录）

上游 09-03 传的，**ref2va 链路的第一次真升级**——此前该链路只有 544p 训练的 4-step v0.1。

权重层面：624 tensors / blocks 50/50 / refiner 2/2 / rank 128 / alpha 8 → **缩放 0.0625**
（与 v0.1 同为 0.0625，但这是巧合不是规律，见 §1.2）。sha256 三方核过：HF tree 的 LFS oid
== HF resolve 的 `x-linked-etag` == 魔搭 API 的 `Sha256` == `9bac880b…`。

烘焙在 gpu46 上跑（先 `drop_caches`，见 §5.4），vllm-omni 镜像里 13 个分片约 5 分钟：

```
patched 208 tensors
||delta|| / ||W|| :  min=0.0000  median=0.0001  max=0.0013   （最大在 blocks.49.mlp.fc2.weight）
```

与 v0.1 那次（median 0.0002 / max 0.0019）同量级，正常。

组装出的正式目录 `MiniMax-H3-Ref2VA-Turbo8-768p-BF16-vLLM`：

```
partition=ref2va  tasks=[ref2va]  sigma_shift_scales: video=6.0 audio=3.0
base_schedule: [1, .875, .75, .625, .5, .375, .25, .125, 0]     （8 NFE → 9 个 boundary）
fusion_verification: max_abs_error=0.0  verified_target_tensors=208
                     verified_factor_pairs=312  changed_values=489792
```

**⚠️ shift 6/3 是推的，上游没给。** 依据与改法见 §6。

**归因上的一个坑**：它跟现网 v0.1 之间**分辨率（544p→768p）与步数（4→8）同时变了**，
不是一条曲线上的两点。「质量好了多少」要跟 v0.1 同 prompt/同 seed 比，「快了多少」是
4 NFE 对 8 NFE 的不等步比较——如果 8 步只是打平 v0.1 的 4 步，对我们是退步不是升级。
两个结论别合成一个。§5.1 那四个未验证项（步数窗口 / 长参考输入显存 / 音频内容 /
issue #37 抖动）对这份同样一个都没验。

### 5.3 fl2v 4-step v1.2 768p（2026-09-04 执行记录）

名义上是现网 `v1.1_768p` 的**同档位直接替代**：同任务、同训练分辨率、同步数，所以
A/B 是干净的同步数对比。**但权重层面它不是 v1.1 的续训**，见下面那段。

**⚠️ 缩放从 v1.1 的 1.0 变成了 0.0625。** 照抄 v1.1 的烘焙命令会把增量放大 16 倍。
rank 128 / alpha 8 已用 HTTP Range 只取 header 核过，其 ComfyUI 变体 metadata 的
`training_scale=0.0625` / `training_alpha=8.0` 交叉验证。

烘焙（gpu46，13 分片约 5 分钟）与组装的验收数字：

```
patched 208 tensors
||delta|| / ||W|| :  min=0.0000  median=0.0001  max=0.0011   （最大在 blocks.49.attn.qkv_proj.weight）
partition=fl2va  tasks=[t2va, fl2va]  shift video=6.0 audio=3.0
base_schedule: [1, .75, .5, .25, 0]     （4 NFE → 5 个 boundary）
fusion_verification: max_abs_error=0.0  verified_target_tensors=208
                     verified_factor_pairs=312  changed_values=435470
```

产物 `MiniMax-H3-FL2VA-Turbo4-768p-v1.2-BF16` + `-vLLM`。

#### v1.2 不是 v1.1 的续训，做 A/B 前必须先知道这件事

v1.1 的 `||delta||/||W||` 中位数是 0.0015，v1.2 只有 0.0001 —— **施加到基座上的增量小了
一个数量级**。用 `tools/minimax_h3_turbo/compare_lora_deltas.py`（逐层算 `scale · B@A`
再比范数与余弦，默认每 25 个模块采一层、共 13 层）交叉核过：

| 比较 | 范数比 R/L 中位数 | 余弦中位数 | 读法 |
|---|---|---|---|
| v1.1 → **v1.2** | **0.036** | **+0.006**（min −0.007 / max +0.397） | 幅度只有 1/28，方向近乎正交 |
| 8step_v1.0_768p → **v1.2** | **1.228** | +0.043（max **+0.797**，深层） | 同量级，深层方向明显相关 |
| （对照）v1.0_768p → v1.1_768p | 1.160 | **+0.872** | 这才是「同配方续训」长什么样 |

最后一行是这张表的标尺：它复现了 §5.1 当初对全部 312 对因子做的那次比较（当时记的是
范数比 1.158、余弦 0.885），说明 13 层采样够用、工具也没算错。

也就是说 **v1.2 是从 alpha-8 那条线分出来的，不是 v1.0→v1.1 那条 alpha-128 线的下一代**，
尽管它的文件名让人以为是。余弦与缩放无关（是尺度不变量），所以这个结论不依赖 0.0625
取得对不对。

两条实操推论：

1. **0.0625 是对的，别去"修"它。** 三个独立证据：权重自己的 metadata、ComfyUI 变体的
   `training_scale`、以及它与同为 alpha 8 的 `8step_v1.0_768p` 落在同一个幅度量级
   （R/L 1.23）——而后者那条线（480p Turbo8）线上已经跑着、实测 1.76x。
2. **出片"比 v1.1 更接近未蒸馏基座"是预期行为，不是烘焙 bug。** 增量小 28 倍，
   画面更保守、离基座更近是必然的。这恰恰是 A/B 要量的东西（v1.1 是否过度偏移），
   别一看到"变化不大"就回头怀疑缩放。

### 5.4 两条执行环境上的坑（本次踩到的）

1. **魔搭不一定有镜像，而 hf-mirror 救不了 Xet 仓。** v1.2 上传 2 小时后魔搭仍无同步，
   hf-mirror 对这个仓是 302 跳到 `us.aws.cdn.hf.co`（Xet CDN）而**它不代理**。实测管理
   节点 aria2c `-x16` 只有 **49 KiB/s**（ETA 8~10 小时），gpu46/47/50 单连接 11/38/33 KiB/s。
   走通的路子是**在能直连 HF 的机器上下、再推到 NFS**：本地 16 路 range 并发 ~1.0 MB/s，
   到管理节点的上传实测 5.6 MB/s，1.38 GB 推过去 4 分钟。sshfs 挂 NFS 直写没有意义——
   瓶颈在下载侧不在传输侧，只是多一个失败面。
2. **烘焙前先 `drop_caches`。** 这台机 251 GB 内存里 137 GB 是页缓存，读 62 GB + 写
   62 GB 会被缓存挤占拖成假死锁（见 `a100-host-cache-starves-multiproc-runs` 那次
   4 卡 1658s→209s 的教训）。`sync; echo 3 > /proc/sys/vm/drop_caches` 一行的事。

---

## 6. 上游没有文档时，shift 和推荐步数怎么定

§2.3 那条组装命令里的 `--video-shift` / `--audio-shift` 不是自由参数，它跟权重一起
标定。旧的五份权重能照抄，是因为上游 GitHub 的 model-specs 表格逐份列了训练 shift。
**2026-09-04 新增的两份不在那张表里**（GitHub 仓最后一次提交是 08-27，HF 那边只有一行
"Upload ... with huggingface_hub"，没有 release note），所以我们只能推。

本次采用的推法与它的依据边界：

- **推 6/3，依据是训练分辨率而不是步数。** 表里所有标 `768p` 的权重都是 video 6 /
  audio 3，所有 544p 的都是 12 / 3；跨越 4 步与 8 步两档都成立。两份新权重文件名都带
  `768p`，所以取 6/3。
- **这是推测，不是证据。** 出片明显发糊/发飘时，第一个该换的假设就是它。
- **改 shift 的代价很低，别怕试错。** partition 目录只有一个真文件（`model_index.json`）
  加六个符号链接，重组一份不到一秒、不碰 62 GB 权重。换 shift 重跑 §2.3 即可，
  烘焙产物完全复用。

同理，`--num-inference-steps` 只能取权重名里那个数（4 或 8）。上游把「步数」定义为 NFE，
N 次 transformer evaluation 需要 N+1 个 sigma boundary，工具已经按这个语义生成
`base_schedule`，不要手改。

如果哪天上游改了 LoRA 的结构布局（key 命名、QKV 排布、SwiGLU 半序），`--dry-run`
会在断言处直接失败而不是默默产出垃圾 —— 那时才需要动 `bake_turbo_lora.py`。

---

## 7. 相关文件

| 用途 | 路径 |
|---|---|
| 下载脚本（含清单与结构校验） | `scripts/download_minimax_h3_turbo_lora.sh` ⚠️ 见下 |
| 烘焙工具 | `tools/minimax_h3/bake_turbo_lora.py` |
| 融合验真与来源回填 | `tools/minimax_h3_turbo/lora_provenance.py` |
| Partition 强约束组装 | `tools/minimax_h3_turbo/assemble_distilled_partition.py` |
| 新旧 LoRA 增量对比（判断是否续训） | `tools/minimax_h3_turbo/compare_lora_deltas.py` |
| 生产部署档（env / flag / 为什么是这个值） | `docs/实验报告/MiniMax-H3-GPUStack-生产部署档.md` |
| Turbo8 压测数据 | `docs/实验报告/vLLM-Omni-MiniMax-H3-Turbo8-480p-768p-压测对比报告.md` |
| 镜像构建（deploy-configs 固化位置） | `docker/Dockerfile.cuda` |

> ⚠️ **`scripts/` 整个目录在 `.gitignore` 里（`.gitignore:184`），下载脚本不进版本库。**
> 也就是说本文档描述的清单行、基座提示、结构校验那些改动，**只存在于工作区和已经
> scp 到节点的副本**（管理节点 `/root/download_minimax_h3_turbo_lora.sh`）。换机器
> 或换人接手时脚本可能已经不在，届时按 §2.1 的清单格式和 curl 取校验值的方法重建即可
> —— 这也是本文档要把那两段写全的原因。
