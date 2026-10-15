# MiniMax-H3 音频首请求偏差：排查记录与已接受现状

**状态：未修复，已决定接受并登记。** 本文记录完整前因后果、做过的全部实验、被证伪的
假设、一次失败的修复、以及一个很贵的等价性陷阱。目的是两件事：日后想接着查时有起点，
以及下次遇到"隔离能复现、真机不灵"这类问题时知道做过什么、别再走一遍。

* 发现于 2026-10-14，起因是 `sync/upstream-20261012` 合并的上游同步真机验收。
* 相关文档：`docs/sync-20261012/复核-2026-10-14-补测与修复.md` §10（原始记录）、§14（跨实例复核）。
* 结论一句话：**同一进程内，checkpoint 自带的音频解码器首次调用会产出不同的波形。
  但它不是真机现象的完整机制；调用侧的预热消不掉它。**

---

## 1. 症状

同 seed、同 prompt、同输入，**服务启动后第一个请求的音轨与后面不同**；
第 2/3/4 个请求彼此逐位相同。**视频流每次都逐位相同——只有音频。**

| 镜像 | 请求 | 视频流 md5 | 音频流 md5 |
| --- | --- | --- | --- |
| baseline（`origin/main`, v0.30） | cold（第 1 个） | `b86dd2f0…` | `92df960c…` |
| baseline | hot（第 3 个） | `b86dd2f0…` | **`92df960c…`**（相同） |
| **HEAD（v0.31 合并后）** | cold | `12c2b725…` | `04a5ef68…` |
| **HEAD** | run2 / run3 | `12c2b725…` | **`551c55ed…`** |

**baseline 上 cold == hot，HEAD 上 cold ≠ run2 == run3 ⇒ 这个偏差不是既存的，
是本轮合并引入的。**（这是**同一实例内**的比较，所以不受 §14「跨实例不可复现」的影响
——§14 讲的是跨服务启动之间的视频差异，两件事。）

量级（三次独立复现，三组产物两两比对）：

| 组 | 差异样本占比 | 平均 \|Δ\| | 最大 \|Δ\| |
| --- | --- | --- | --- |
| `d_run1` vs `d_run2` | 99.871% | 0.00591 | 0.06619 |
| `e_run1` vs `e_run2` | 100.000% | 0.00735 | 0.25174 |
| `f_run1` vs `f_run2` | 100% | 同上量级 | 内容相关 |

满幅 1.0，平均差约 **−44 dBFS**。**听感上很轻，但样本占比 100%。**

---

## 2. 定位：打点 `decode_latent`

### 2.1 打点方式

`patch_audio_probe.py` 把 `MiniMaxH3AudioVAE.decode_latent` 包一层，用**环境变量驱动**
（一份树跑两臂）：

| 环境变量 | 作用 |
| --- | --- |
| `H3_AUDIO_PROBE_LOG=<path>` | 打开 JSONL 打点 |
| `H3_AUDIO_DECODE_DETERMINISM=1` | 把 decode 包进 `_AudioVAEDeterminismContext`（候选修法之一） |

每次调用记录：**latent 哈希、输出波形哈希、调用序号、8 个 torch 全局后端的取值、
解码参数的设备与精度**。

配套：

* `launch_h3_audio.sh <A|B>` — A 臂不打确定性包裹，B 臂打；其余参数（同一镜像、同一档、
  同一模型、同一 LoRA）完全一致。
* `send3.py <host> <port> <tag> <N>` — 等服务就绪，连发 N 个**同 seed**请求，逐个落盘。

### 2.2 关键数据（3 次请求 × 4 rank）

```text
call 1 : latent=8e28b23be324 → waveform=c39449be735c
call 2 : latent=8e28b23be324 → waveform=bd7fa48c7c7d
call 3 : latent=8e28b23be324 → waveform=bd7fa48c7c7d
```

* **输入 latent 逐位相同**（4 个 rank 也相同）⇒ 文本编码器 / DiT / 噪声 / solver
  **全部确定性**，问题被夹在解码这一环。
* **只有波形不同**，且只差在第 1 次。
* 8 个全局开关在**全部调用、全部 rank 上只有一种取值**：
  `matmul_tf32=False, cudnn_tf32=True, cudnn_enabled=True,
  cudnn_deterministic=False, benchmark=False, f32_precision=highest, sdp=全开`
  ⇒ **不是环境状态被谁改过。**

---

## 3. 三个被证伪的假设

| 假设 | 结论 | 证据 |
| --- | --- | --- |
| 环境精度状态（TF32 / cuDNN）在首调用时不同 | **证伪** | 8 个开关在全部调用 / 全部 rank 上只有一种取值 |
| 是 cuDNN / TF32 的非确定性 | **证伪** | 把 `decode_latent` 包进 `_AudioVAEDeterminismContext`（关 cudnn、关 TF32、开 deterministic）后**仍然** call1 ≠ call2（`audB` 臂） |
| `weight_norm` 首次前向改了有效权重 | **证伪** | 有效权重指纹在任何前向之前 / call1 之后 / call2 之后都是 `7f63392e874dc76f` |

---

## 4. 隔离复现（秒级）

只加载 checkpoint 自带的音频 VAE（**605 MB，无 DiT、无文本编码器、无服务**），
在同一进程里对**同一个 latent** 连解多次：

```python
#!/usr/bin/env python3
"""Isolate MiniMaxH3AudioVAE.decode_latent: is the FIRST invocation different?"""
import hashlib, sys, torch
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE

COMPONENT = "/nfs-data/models/MiniMax-H3/FL2VA/audio_vae"

def md5t(t):
    return hashlib.md5(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()[:16]

def main():
    dev = torch.device("cuda", 0)
    vae = MiniMaxH3AudioVAE(COMPONENT, device=dev, trust_remote_code=True)
    vae.eval(); vae.remote.to(dev)
    ch = int(vae.config_dict["latent_channels"])
    torch.manual_seed(7)
    latent = torch.randn(2, ch, 207, device=dev, dtype=torch.float32)
    outs = []
    for i in range(1, 5):
        with torch.inference_mode():
            w = vae.decode_latent(latent)
        h = md5t(w); outs.append(h)
        print("  call %d -> %s" % (i, h), flush=True)
        del w
    print("distinct outputs:", len(set(outs)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

实测输出：

```text
decode call 1 -> 846d2561e36abaf8
decode call 2 -> ca5410915809fdde
decode call 3 -> ca5410915809fdde
decode call 4 -> ca5410915809fdde
distinct outputs: 2
```

⇒ **根因是 `self.remote.decode()`（checkpoint 自带、`trust_remote_code`、fp32）
在一个进程里的首次调用会走一条不同路径**，与输入、权重、后端开关都无关。

---

## 5. 等价性陷阱（**本文最贵的一课**）

隔离里能复现的那个机制，**不等于真机上那个现象的完整机制**。四种构造的对照：

| 构造 | 现象 | warm-up 有效吗 |
| --- | --- | --- |
| 直接 `.to(cuda)`，`decode_only=False` | call1 与后续不同 | **有效**（后续全部等于稳定值） |
| `load_device=cpu` + `PinnedModuleStager`，`decode_only=False`，带 load/decode/offload 循环 | call1 与后续不同（复现服务症状） | **有效** |
| `load_device=cpu` + stager，**`decode_only=True`（管线实际取值）** | **三次全同，根本不复现** | 无所谓 |
| **真机 4 rank 服务，`decode_only=True`** | call1 与后续不同 | **无效** |

探针脚本形态（`probe_audiovae5.py` 的骨架；注意 `_first_decode_absorbed` 是当时那版
**已回滚**的修法引入的开关，跑在干净树上要去掉那两行）：

```python
def run(decode_only, use_warmup, latent, label):
    dev, cpu = torch.device("cuda", 0), torch.device("cpu")
    vae = MiniMaxH3AudioVAE(COMPONENT, device=dev, load_device=cpu,
                            decode_only=decode_only, trust_remote_code=True)
    vae.eval()
    if not use_warmup:
        vae._first_decode_absorbed = True  # emulate pre-fix behaviour
    out = []
    for i in range(1, 4):
        vae.load_to_device()
        with torch.inference_mode():
            w = vae.decode_latent(latent)
        vae.offload_to_cpu()
        out.append(md5t(w))
    print("  %-42s -> %s  stable=%s" % (label, out, len(set(out)) == 1), flush=True)
```

**教训**：`decode_only` 这个开关决定了隔离里能不能复现。**没核对它之前，
任何"隔离里验证通过"都不能当作真机有效的证据。** 我就是在没核对的情况下先跑了
一次真机验收，白跑。

---

## 6. 一次失败的修复（已回滚，不在分支里）

### 6.1 修法形态

`MiniMaxH3AudioVAE` 里惰性做一次极小解码：

```python
probe = torch.zeros(1, latent_channels, 1)          # len=1 足够，与 shape 无关
with torch.inference_mode():
    self.remote.decode(probe)
```

隔离实测：

```text
reference settled value: ca5410915809fdde          # 未预热时第 2 次的值
warm-up(remote.decode, len=1) took 0.208s
  next 3 real decodes: ['ca5410915809fdde', 'ca5410915809fdde', 'ca5410915809fdde']
  all equal: True  equals settled: True
```

要点：**0.208 s**，与生产时长 / 分辨率无关（换 shape 预热同样有效）；
预热后每次真实解码都等于"稳定值"，**不改动稳态输出**，只是让第一个请求不再例外。

### 6.2 单测（5 passed，机制层面）

首次多一次解码、后续不再预热、探针与请求同设备同 dtype、预热失败不致命、
`encode_only` 不受影响。**隔离环境里也逐位验证过。**

### 6.3 真机验收：**失败**

同一份代码、同一档、同一台机。为排除"修法到底跑没跑"的推断，把 warm-up 本身也打了点
（`instrument_warmup.py`）：

```text
warmup_in   probe_shape=[1, 32, 1] device=cuda:0      ← 4 个 rank 都执行了
warmup_out  out_shape=[1, 1, 800]  out_md5=c87cebfd…  ← 没有 warmup_failed
decode_in   call=1
decode_out  call=1  wf=441400195b00                   ← 仍然与 call2 不同
decode_in   call=2
decode_out  call=2  wf=8767dab2de0f
decode_out  call=3  wf=8767dab2de0f
```

**预热确实在第一次真实解码之前、同一实例、同一 CUDA 设备上成功执行了，但首次解码仍然不同。**
（latent 三次都是 `0cd5a811dddd`，参数设备三次都是 cuda，全一致。）

| 运行 | run1 音频流 | run2 | run3 |
| --- | --- | --- | --- |
| 未加修法 | `291753459d5f71ad` | `a4efb845430a540b` | `a4efb845430a540b` |
| 加了修法 | 首次仍独立 | — | — |

两次运行的**形态完全相同**（run1 独立、run2 == run3），修法没有改变形态。
**结论：该修法在目标环境无效，已回滚，单测文件一并删除。**

### 6.4 为什么不能走"打开引擎自带 warm-up"这条路

H3 的 `dummy_run_num_frames = 0` 是有原因的，打开 `_make_dummy_request` 会让**服务起不来**：

```text
dummy run to warm up the model (1/1: 512x512, model-default frames)
Dummy run failed: checkpoint partition 'fl2va' supports ['fl2va', 't2va'], got task='ref2va'
→ /ready = 000
```

链路：`_make_dummy_request` 对任何声明支持音频输入的模型都注入一段音频 →
`_resolve_task(None, multi_modal_data)` 见 `audio is not None` 且 `audio_mode != "lock_source"`
→ 返回 `"ref2va"` → fl2va 分区的 `supported_tasks` 只有 `{'fl2va','t2va'}` → 抛错；
而 `run_startup_warmup` 捕获后 `self.close()` 再抛出，**引擎直接关闭**。

可参考的先例：`pipeline_auk.py:331` 用的是**管线内局部预热**，不是引擎 dummy 请求。

---

## 7. 当前理解与结论

1. **根因已定位到 checkpoint 自带的 `dac_*` 音频解码器**（`trust_remote_code`、fp32）：
   **一个进程内的首次调用会走一条不同路径**，只影响波形数值。
2. **但隔离复现的那个机制 ≠ 服务现象的全部机制。** 服务里还有一层尚未复现的状态，
   可能与 CPU-offload 暂存（`PinnedModuleStager`）、4 rank、或组件加载顺序有关。
   **单靠"先解一次"消不掉它。**
3. **状态：未修复、已回滚，分支里没有任何相关改动。**

---

## 8. 若恢复排查

### 8.1 下一步

**直接在 checkpoint 自带的 `dac_*` 解码器实现里找那个「只在首次调用才走」的分支**
——`/nfs-data/models/MiniMax-H3/FL2VA/audio_vae/`（`trust_remote_code`，加载时读的
就是这份代码）。

方向：看有没有 `lazy` / `first_call` / 缓存初始化 / `self._something is None` 之类的
首次分支，或某个模块在首次前向时才 `.to()` / 才建立 buffer。

**不要再在调用侧加保护**——已经试过一次（§6），隔离通过、真机无效。

### 8.2 复现物料（**注意：NFS 上的那份 scratch 已在收尾清理中删除**）

原先在 `a100-0008:/nfs-output/qfix-20261014/`，该目录**已清理**。现有物料：

| 物料 | 位置 | 备注 |
| --- | --- | --- |
| 五个隔离探针 `probe_audiovae{,2,3,4,5}.py` | 见下 | **需重新落到 NFS 或容器里** |
| 打点器 `patch_audio_probe.py` | 同上 | 环境变量驱动，见 §2.1 |
| warm-up 打点 `instrument_warmup.py` | 同上 | 见 §6.3 |
| 启停脚本 `launch_h3_audio.sh`、发射器 `send3.py` | 同上 | A/B 两臂 + 同 seed 三连发 |
| 打点后的 `vae.py` | 已删 | md5 曾是 `017ca92ee4c5b9b0d5a152a1e527019f`；可用 `patch_audio_probe.py` 重生成 |
| 两臂原始打点 `audA.log` / `audB.log`、产物 `aud{A,B}_run{1,2,3}.mp4` | 已删 | 关键数值已摘录在本文与 §10 |
| 底座镜像 | ACR `arm64-a100-sync-20261012-test`（当时那版） | 现在是 `arm64-a100-latest` |

**重建步骤**（≈10 分钟）：

1. 起一个带 GPU 的容器，挂 `/nfs-data`、`/nfs-models`、`/nfs-output`；
2. 把探针脚本放进容器（或 NFS）；
3. 跑 `probe_audiovae.py`（§4 那段），确认当前代码上**仍能复现**（秒级）；
4. 跑 `probe_audiovae5.py`（§5 那张表），确认**四种构造的分布没变**——
   第 3 行仍然不复现、第 4 行（真机）仍然复现，这是接着查的前提。

真机侧要起 4 rank 服务：H3 fl2va 单档，~20 min 就绪，约 140 GB RSS。

### 8.3 一个更省力的中间态

如果暂时不想动 checkpoint，只想要一个**可复现的交付行为**，还有个便宜做法：
**把服务重启后的第一个请求当作预热请求丢弃**（生产侧约定），
这样对外可观测的音轨就是稳定的 `551c55ed…`。代价是第一次调用白跑一次。

---

## 9. 接受决定与规避约定（2026-10-15）

**决定：接受，登记后搁置。** 理由与影响面：

* 影响面：**同 seed 重跑，音轨取决于服务是否刚启动**；**视频不受影响**，
  而视频是这类模型的主要交付物。
* 幅度：平均约 −44 dBFS，样本占比 100%，听感很轻。
* **对验收方法的影响（重要）**：任何"冷态第一个请求"的对比都不可复现。
  §6.3 那些冷态数字（含整文件 md5）都沾这一条，引用时须注明。
* **对外约定**：如需同 seed 音轨可复现，以"
  **重启后先发一个丢弃请求**"为预热约定；或在验收表里写明"冷态首请求的音轨不计入逐位比对"。

---

## 10. 时间线

| 时间 | 事件 |
| --- | --- |
| 2026-10-14 上午 | §6.3 真机验收发现"同 seed 冷热不一致"（`回归 ③`，当时只记了现象） |
| 2026-10-14 中午 | 打点定位：latent 相同、波形不同、只差首次；三个假设证伪 |
| 2026-10-14 下午 | 隔离探针复现（秒级）；实现 warm-up 修法 + 5 单测 |
| 2026-10-14 下午 | **真机验收失败**，回滚；查清 `decode_only` 等价性陷阱；查明引擎 dummy 路径不可用 |
| 2026-10-14 夜 | §14 复核 §6.3 头条时确认：跨实例本来就不可复现（与本文的**同实例**结论是两件事） |
| 2026-10-15 | 决定接受并登记；收尾清理删除了 NFS 上的 scratch |
