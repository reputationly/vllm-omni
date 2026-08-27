# SpargeAttn 出包修复 与 SenseNova-U1.5/Krea2/MiniMax-Music3/IndexTTS-2.5 GPU 配置实测报告（2026-08-27）

> 范围：(1) `docker/Dockerfile.cuda` 里 SpargeAttn（`SLA_SAGE2_ATTN` backend）编译从失败到修好的完整排查过程；
> (2) 修好的镜像批量推送到 50 台 GPUStack 节点后，SenseNova-U1.5 / Krea2 / MiniMax-Music3 / IndexTTS-2.5
> 四个模型的功能回归 + GPU 资源配置实测（TP 卡数、OOM 边界、Int8 量化）。
> 测试机器：`gpu41-45`（5 台 A100-40G×4 调试机，SSH 别名）。镜像构建走 GitHub Actions 标准流程，不占用调试机。

---

## 1. 一句话结论

- **出包**：3 个独立构建 bug 全部修复并已发布，最终镜像 `arm64-a100-20260827-0122-fdcb3183`
  （digest `sha256:7eff6bfd2afd0c9320b4b6ddf390bc8ace64da1621b5a7f07fdf415be4f84259`），已批量推送到全部 50 台节点。
- **功能回归**：4 个模型在新镜像上全部正常，没有因为本轮构建修复引入回归。
- **GPU 配置**（本报告的核心增量）：
    - SenseNova-U1.5：**TP=2**，比 TP=1 快 13%、显存省一半，且 **TP=1 在官方宣传的 4K 直出场景下会直接 OOM**，TP=2 则 4K 安全跑通。
    - Krea2：**TP 对它完全没用**（`ReplicatedLinear` 架构决定，见 §5.2），该用 1 卡；**真正有效的优化是 `--quantization int8`**——零代码改动，速度快 30%、显存省 30%，还把分辨率安全上限从"1024 都快扛不住"提到"1536 稳"。
    - MiniMax-Music3：**1 卡版比 2 卡版快 ~25%**，且单请求下没有 OOM——2 卡版当初为规避 OOM 而选的假设（"140GB 大卡"）在 A100-40G 实测下并不成立。

---

## 2. 出包问题排查（Dockerfile.cuda / SpargeAttn）

### 2.1 背景

`docker/Dockerfile.cuda` 里新增了编译 [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn) 的构建步骤，
用于给 `SLA_SAGE2_ATTN` 这个 diffusion attention backend（MiniMax-H3 sage2 分块稀疏路径的音频损坏修复）提供内核。
标准出包流程走 `build-arm64.yml`（GitHub Actions `ubuntu-24.04-arm` 云端 runner，**无 GPU、无 NVIDIA 驱动**），
这个前提是后面三个 bug 的共同根因。

### 2.2 Bug #1：`-lcuda` 链接失败

```text
/usr/bin/ld: cannot find -lcuda: No such file or directory
```

**根因**：构建机没有物理 GPU/驱动，所以真正的 `libcuda.so`（CUDA Driver API 库，区别于 `libcudart`/CUDA Runtime）
在构建机上不存在；但 `nvcc`/CUDA toolkit 本身是有的，前面的 `.cu`→`.o` 编译步骤能正常跑完，只有最后链接步骤失败。

**修复**：把 `LIBRARY_PATH` 指向 CUDA toolkit 自带的驱动 stub 库（`/usr/local/cuda/lib64/stubs/libcuda.so`，
这个文件存在的唯一目的就是给无驱动构建环境的链接器用）。

```dockerfile
TORCH_CUDA_ARCH_LIST=8.0 LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}" \
   pip install . --no-build-isolation
```

### 2.3 Bug #2：CUDA 数学库头文件缺失（`cusparse.h` → `cublas_v2.h`）

修复 Bug #1 后，第一个扩展模块（`csrc/qattn`）能正常链接了，但**同一个包里的第二个扩展模块**
（`csrc/fused/fused.cu`）报错：

```text
fatal error: cusparse.h: No such file or directory
```

**根因**：基础镜像的 CUDA toolkit 只带了 `nvcc`/`libcudart`，**没带 `libcusparse-dev` 这类数学库的开发头文件包**。
修了 `cusparse` 之后重新构建，同一个文件又报了第二个缺失：

```text
fatal error: cublas_v2.h: No such file or directory
```

**修复**：与其一个头文件一个头文件地试错，直接按检测到的 CUDA 版本把 `cusparse`/`cublas`/`cusolver`/`curand`/`cufft`
的 `-dev` 包一次装齐：

```dockerfile
RUN CUDA_VER_DASH="$(nvcc --version | sed -n 's/^Cuda compilation tools, release \([0-9]*\.[0-9]*\).*/\1/p' | tr '.' '-')"; \
    apt-get install -y --no-install-recommends \
        "libcusparse-dev-${CUDA_VER_DASH}" "libcublas-dev-${CUDA_VER_DASH}" \
        "libcusolver-dev-${CUDA_VER_DASH}" "libcurand-dev-${CUDA_VER_DASH}" \
        "libcufft-dev-${CUDA_VER_DASH}"
```

（构建机实测 CUDA 版本是 13.0，`libcusparse-dev-13-0` 等包能从 NVIDIA 官方 apt 源正常装上。）

### 2.4 Bug #3（最隐蔽）：`python` 命令不存在，模板实例化文件全部生成失败但无人报错

修完 Bug #2 后，`spas_sage_attn` 包整体 `pip install` 终于成功了，**但运行时 import 直接崩**：

```text
ImportError: .../spas_sage_attn/_qattn...so: undefined symbol:
_Z29SpargeAttentionSM80DispatchedILj128ELj64E...
```

**根因排查**：clone 了 SpargeAttn 官方仓库源码看 `setup.py`，发现它的 sage2 kernel 模板实例化 `.cu` 文件
是**运行时自动生成**的（不是仓库自带），生成方式是：

```python
def run_instantiations(src_dir):
    for py_file in Path(src_dir).rglob('*.py'):
        os.system(f"python {py_file}")   # 注意：是 python，不是 python3；返回码也不检查
```

而 `docker/Dockerfile.cuda` 里 `python3 → python` 的软链接**原来放在文件最后一步**，SpargeAttn 编译发生在
Dockerfile 中段——那会儿容器里还没有 `python` 命令，`os.system("python ...")` 静默失败（`os.system` 的
非零返回码从来没被检查），于是**一个模板实例化文件都没生成**。扩展模块的 `.cu`→`.o`→链接照常"成功"
（因为只是没有函数体，符号声明还在），只有真正调用时才会因为符号缺失而崩溃——这也是 Bug #3 比前两个更晚才
暴露的原因：pip 安装、模块 import 都不会触发它，只有真正跑一次 kernel 调用才会报错。

**修复**：把 `RUN ln -sf /usr/bin/python3 /usr/bin/python` 从 Dockerfile 末尾挪到最前面（Step 1 之后），
保证后续所有第三方 `setup.py` 都能看到 `python` 命令。

### 2.5 三次修复对应的 commit / 镜像

| commit | 内容 | 修复后镜像 tag |
| --- | --- | --- |
| `3b50e652` | 装 `libcusparse-dev` | `arm64-a100-20260826-2342-3b50e652`（仍不可用，见 §2.3） |
| `8780e372` | 补齐 cublas/cusolver/curand/cufft dev 头文件 | `arm64-a100-20260827-0105-8780e372`（能装上，import 崩，见 §2.4） |
| `fdcb3183` | `python` 软链接挪到文件最前面 | `arm64-a100-20260827-0122-fdcb3183`（**完全正常**） |

最终验证（gpu41/gpu45 双机确认）：

```python
import spas_sage_attn
from spas_sage_attn import _qattn, _fused          # 两个 CUDA 扩展都能 import
from spas_sage_attn.core import spas_sage2_attn_meansim_cuda  # 上层实际调用的函数也正常
```

`grep -rl "SLA_SAGE2_ATTN" vllm_omni/` 确认该 backend 已在
`vllm_omni/diffusion/attention/backends/registry.py` / `sla_sage2_attn.py` / `platforms/cuda/platform.py`
里正确注册。`Warning: Sage2++ NOT enabled` 是预期的（Sage2++ 需要 CUDA≥12.8 且不支持 sm80，与本次修复无关）。

### 2.6 GitHub Actions 侧的次要坑（记录，不是本次修复的一部分）

调试过程中还遇到过两次构建卡在 `queued` 状态永不启动、`gh run cancel` 报"运行已完成"这种自相矛盾的错误——
这是 GitHub ARM runner 队列的临时基础设施问题，不是仓库/workflow 配置问题（`grep concurrency: build-arm64.yml`
确认没有并发限制块）。排查方式是 `gh release list` 看有没有真正出包成功，而不是信任 `gh run`/`gh api` 的状态字段。

---

## 3. 50 台舰队批量升级

镜像修复完成后，用现有的 `api/gpustack/docs/scripts/lx2v-fleet.sh` + `lx2v-node.sh` 标准流程批量升级（这是已有
运维工具，本次没有新增）：

```bash
bash /root/lx2v-node.sh prepare-transfer               # 238 管理节点：拉最新镜像存 NFS tar（指纹去重，自动识别新版本）
bash /root/lx2v-fleet.sh -j 10 upgrade-engine --engine vllm-omni --offline   # 50 台并发 docker load
```

**关键机制**（读了 `lx2v-node.sh` 源码确认，不是猜的）：`upgrade-engine` 只是把新镜像 `docker load` 到本地，
**不会重启任何正在运行的容器**——真正切换到新镜像是后续在 GPUStack UI 上逐个删除实例、让它用新镜像自动重建
（脚本自己会提示"先删一个、等 Running 再删下一个，保持服务不断"）。所以这一步本身对现网没有中断风险。

结果：`bash lx2v-fleet.sh -j 10 status` 确认 **50/50 全部成功**。抽查 gpu41/gpu45（内网 IP 分别是
`10.0.0.134`/`10.0.0.11`，确认在这 50 台名单内）：镜像 ID `7eff6bfd2afd`，与构建日志里的推送 digest
`sha256:7eff6bfd2afd0c9320b4b6ddf390bc8ace64da1621b5a7f07fdf415be4f84259` 完全一致。

---

## 4. 四模型功能回归（新镜像）

清空 gpu41-45 上的历史测试容器后，用新镜像逐个重新部署验证，全部通过：

| 模型 | 机器 | 请求 | 结果 |
| --- | --- | --- | --- |
| IndexTTS-2.5 | gpu41 | `/v1/audio/speech`（带 ref_audio 克隆） | 200，22.05kHz 单声道，3.99s |
| SenseNova-U1.5 | gpu42 | `/v1/chat/completions`（modalities=image，TP=2） | 200，2048×2048 PNG，峰值显存 18.7GB（与官方 H200 TP=2 recipe 的 18.2GB 高度吻合） |
| Krea2 | gpu43 | `/v1/images/generations`（TP=2） | 200，1024×1024 PNG |
| MiniMax-Music3 | gpu44 | `/v1/audio/speech`（2 卡 deploy-config） | 200，32kHz 立体声，27.78s |

生成文件都在 `/nfs-output/model_reverify_20260827/`（各自机器上的绝对路径），未做任何拼接/截图处理。

---

## 5. GPU 配置调研与实测

### 5.1 GPUStack 后端机制（读 `api/gpustack` 源码确认，非猜测）

两条容易踩坑的规则：

1. **GPU 卡数不会从 `--tensor-parallel-size`/`--num-gpus` 自动推断**。`VLLMOmniServer`
   （`gpustack/worker/backends/vllm_omni.py`）直接把 backend_parameters 透传给 `vllm serve` CLI，
   但调度器实际预留几张卡看的是 `model.gpu_selector.gpus_per_replica`（`vllm_omni_resource_fit_selector.py`）——
   这个字段**必须在建模型时手动填**，不填默认按 1 卡调度。2 卡模型如果忘记填，会被错误地塞进 1 张卡。
2. **Category 不会自动识别成图像/音乐**。`scheduler.py` 里 `_vllm_omni_category` 只对 `minimax-h3` 开了
   hint 归类到 VIDEO，其余一律落到默认的 `TEXT_TO_SPEECH`。**Krea2、SenseNova-U1.5、MiniMax-Music3 都要在
   建模型时手动指定 Category**，否则会被错误地标成"语音合成"。

### 5.2 Krea2：TP 无效的根因 + Int8 量化实测

**TP=1/2/4 全部同速同显存**：

| 配置 | 生成耗时（1024×1024，8 步） | 显存峰值 |
| --- | --- | --- |
| TP=1 | 6.01s | 38.3GB（39.49GB 可用容量的 97%，余量仅 2.6GB） |
| TP=2 | 5.75s | 38.3GB/卡（**没有摊薄**） |
| TP=4 | 5.8-6.0s（**零提升**） | 38.3GB/卡（**没有摊薄**） |

**根因**（读 `vllm_omni/diffusion/models/krea2/krea2_transformer.py` 源码确认）：Krea2 的所有线性层
（Q/K/V/O 投影、MLP、projector、embedder）统一走一个 `_linear()` helper，返回的是 vLLM 的
`ReplicatedLinear`，**不是**真正做张量并行切分的 `ColumnParallelLinear`/`RowParallelLinear`/`QKVParallelLinear`：

```python
def _linear(in_features, out_features, bias, quant_config, prefix) -> ReplicatedLinear:
    # ReplicatedLinear (not nn.Linear) so the diffusion LoRA manager can wrap these projections.
    return ReplicatedLinear(...)
```

`ReplicatedLinear` 的权重在每个 rank 上都是完整的一份拷贝，不做任何切分——这是**为了让扩散模型的 LoRA
管理器能正确包装/合并这些投影层做的架构取舍**（Krea2 支持运行时加载 LoRA，官方 recipe 里有
`"lora": {"name": "darkbrush", ...}` 的示例），代价是牺牲了张量并行的性能收益。所以
`--tensor-parallel-size N` 对 Krea2 唯一的实际效果就是"多起 N 份完整模型的冗余拷贝"，没有任何速度/显存收益。

**Krea2 不支持图生图**：查了 `pipeline_krea2.py` 的 `forward()` 完整签名，只有
`prompt/height/width/num_inference_steps/guidance_scale` 等纯文生图参数，**没有 `image=` 入参**。
这是代码层面就不支持，不是没测出来。

**分辨率安全边界（BF16）**：1024×1024 就已经用掉 97% 显存，**1536×1536 和 2048×2048 都是真实的 CUDA
设备级 OOM**（"32.50 MiB is free"，不是软性 `gpu_memory_utilization` 上限）。两次 OOM 后容器都存活、
健康检查仍是 200，属于优雅降级，不是硬崩溃。

**CPU offload 现在不可用**：`vllm_omni/diffusion/offloader/` 有通用 offload 框架（Hunyuan-Image3、
Wan2.2-S2V、LTX2 等模型都接了），但 `pipeline_krea2.py` 全文没有引用它——即使传参数也不会生效，
要支持得先把代码接到 offloader 模块上，属于开发工作量而非配置项。

**`--quantization int8`：零代码改动的最大收益项**。`pipeline_krea2.py` 构造 transformer 时已经把
`quant_config=od_config.quantization_config` 完整传下去了（`krea2_transformer.py` 里每一层线性层
构造都带 `quant_config` 参数），而 vLLM-Omni 的 Int8 量化方法（`vllm_omni/quantization/int8_config.py`）
按 `isinstance(layer, LinearBase)` 通用分派，`ReplicatedLinear` 正是 `LinearBase` 的子类——所以
**不用改一行代码**，直接给 `vllm serve` 加 `--quantization int8` 就能生效（在线量化，加载时
BF16→Int8 转换，日志会打印 `Building quantization config: int8` → `Selected
CutlassInt8ScaledMMLinearKernel for Int8OnlineLinearMethod`）：

| 分辨率 | BF16 | Int8 |
| --- | --- | --- |
| 1024×1024 | 6.0s，38.3GB（97%，濒临 OOM） | **4.2-4.4s（快 ~30%），26.6GB（省 ~30%）** |
| 1536×1536 | **OOM** | 9.8s，35.3GB（正常，仍有余量） |
| 2048×2048 | **OOM** | 21.6s，39.1GB（99%，能跑但零余量，不建议作为生产档位） |

三次 Int8 生成后容器均健康，输出 PNG 文件大小与 BF16 几乎一致（4197415 vs 4197414 字节），
没有做逐像素质量比对，只做了文件层面的合理性检查。

> **附注（模型身份澄清）**：本次用的权重是 HuggingFace `krea/krea-2-turbo`（开源权重，模型卡明确写
> "Open-weight release"，12B 参数）。[artificialanalysis.ai 文生图榜单](https://artificialanalysis.ai/image/leaderboard/text-to-image)
> 上的 "Krea 2 Large"（$60/1k 张图，闭源 API）、"Krea 2 Medium Turbo"、"Krea 2 Medium" 是 Krea 官方
> **闭源商用产品线**的分级，模型卡完全没提这几个名字，**不是同一批权重**，两边的 Elo 分数不能互相套用。

### 5.3 SenseNova-U1.5：TP=2 完胜 + 4K OOM 边界

| 配置 | 耗时（2048×2048 文生图） | 显存峰值 |
| --- | --- | --- |
| TP=1 | 51.1s | 36.4GB |
| **TP=2** | **44.7s（快 13%）** | **18.7GB/卡（与官方 H200 recipe 的 18.2GB 高度吻合）** |

TP=2 额外验证：

- **img2img**（`_forward_it2i`，多图输入，`max_pixels_per_image = min(2048², 4096²/图片数)`）：51.5s，20.4GB，正常。
- **4K 直出**（`height=width=4096`，代码里没有硬上限，只做 grid_factor 取整）：**TP=2 正常跑通**
  （215s，24.6GB，验证输出确实是 4096×4096 PNG）；**TP=1 直接 CUDA OOM**（500 错误，"GPU 0 ... 8.56 MiB
  is free"，容器存活、健康检查仍 200，恢复正常）。

**结论**：TP=2 不是"能用"而是速度、显存、4K 安全性三重占优；如果生产要开放 4K 直出功能，
**TP=1 绝对不能用**，会直接把请求打挂。

### 5.4 MiniMax-Music3：1 卡反而更快，2 卡的"安全假设"不成立

`minimax_music3.yaml`（1 卡）v.s. `minimax_music3_2gpu.yaml`（2 卡）：diff 后发现 stage-0 的
`default_sampling_params`（`temperature: 1.0`，无固定 seed）完全一致——**生成的歌曲长度本身是随机的**，
所以不能直接比较单次调用的墙钟耗时，改用 RTF（生成耗时 / 生成音频时长）做归一化对比，各跑 3 次：

| 配置 | RTF（3 次） | 均值 | 显存峰值 |
| --- | --- | --- | --- |
| **1 卡（推荐）** | 1.35 / 1.22 / 1.22 | **~1.26** | 34-38GB（单请求下 3 次均未 OOM） |
| 2 卡 | 1.59 / 1.55 | ~1.57 | GPU0=34GB（talker）+ GPU1=14GB（decoder） |

1 卡版稳定快 ~20-25%。查两份 yaml 发现结构性差异：**1 卡版 stage-0 开了 `enforce_eager: false`
（CUDA graph capture），2 卡版是 `enforce_eager: true`**，2 卡版 yaml 自己的注释写着
"leave graph capture off until that path is verified graph-safe"（保守关闭，未验证 graph 安全性）——
这大概率是速度差的直接原因。

**当初选 2 卡版是为了规避 OOM**（1 卡版 yaml 注释假设"one 140 GB card"，在 A100-40G 上被判断为高风险），
但本次 3 次单请求实测都没有 OOM（显存峰值 34-38GB，39.49GB 可用容量的 86-96%）。

**遗留风险，尚未验证**：1 卡版 yaml 里 `max_num_seqs: 16`（= 8 个并发带 CFG 的请求）是否在真实并发负载下
仍然安全，本次只测了单请求串行场景，**没有做并发压测**。

---

## 6. GPUStack 部署配置建议（最终版）

| 模型 | Backend | gpus_per_replica | Category（手动填） | Backend Parameters |
| --- | --- | --- | --- | --- |
| IndexTTS-2.5 | vLLMOmni | 1（默认） | text_to_speech（默认对） | `--deploy-config /usr/local/lib/python3.12/dist-packages/vllm_omni/deploy/indextts2_5.yaml` |
| MiniMax-Music3 | vLLMOmni | **1**（改：原计划用 2，实测 1 卡更快且不 OOM） | music/非语音类（必填） | `--deploy-config /usr/local/lib/python3.12/dist-packages/vllm_omni/deploy/minimax_music3.yaml` |
| SenseNova-U1.5 | vLLMOmni | **2**（必填，勿用 1——4K 场景会 OOM） | image/多模态（必填） | `--num-gpus 2 --tensor-parallel-size 2` |
| Krea2 | vLLMOmni | **1**（TP 无效，见 §5.2） | image（必填） | `--quantization int8`（**新增，零代码成本换 30% 速度 + 30% 显存**）；门面层需把最大可请求分辨率硬编到 1024×1024（Int8 下 1536 也安全，但为保守起见先按 1024 上线，见下方待办） |

`--host`/`--port`/`--trust-remote-code`/`--allowed-local-media-path` 均由 backend 自动注入，
不需要在 Backend Parameters 里手填。

---

## 7. 遗留待办

1. **MiniMax-Music3 1 卡版并发压测**：`max_num_seqs: 16`（8 并发）在真实并发负载下是否仍然安全，未测。
2. **Krea2 生产分辨率档位**：Int8 下 1536×1536 已验证安全（35.3GB，有余量），2048×2048 能跑但零余量
   （39.1GB/39.49GB）——建议先按 1024×1024 或 1536×1536 上线，2048 档需要更多并发/边界测试后再开放。
3. **Krea2 Int8 输出质量**：目前只做了文件大小层面的合理性检查，没有做与 BF16 的逐像素/人工质量比对。
4. **SenseNova-U1.5 / MiniMax-Music3 是否也能吃到 Int8 免费午餐**：Krea2 能零代码接 Int8 是因为它的
   `pipeline_krea2.py`/`krea2_transformer.py` 已经把 `quant_config` 完整线通了；SenseNova-U1.5 的
   `sensenova_u1_transformer.py` 同样用了 vLLM 的 Parallel Linear + `quant_config`（但目前
   `pipeline_sensenova_u1.py` 从未真正赋值非 None 的 `quant_config`，是"接了线但没插上"状态，
   与本报告写作时的调研结论一致），MiniMax-Music3 的量化情况尚未核实——**都还没有实测验证**。
5. **Krea2 多并发请求（同卡 2 个以上 1024×1024 请求）会不会互相挤爆剩余显存**，未测。

---

## 8. 涉及的关键文件

- `docker/Dockerfile.cuda` — 本次 3 个构建修复的落点（commit `3b50e652`/`8780e372`/`fdcb3183`）。
- `vllm_omni/diffusion/models/krea2/{pipeline_krea2.py,krea2_transformer.py}` — Krea2 的 `ReplicatedLinear`
  架构、无 img2img 入参、`quant_config` 已线通的证据均在这两个文件里。
- `vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py` — `_forward_it2i`（img2img）、
  `_parse_request` 里的 4K 输出分辨率无硬上限逻辑。
- `vllm_omni/deploy/{minimax_music3.yaml,minimax_music3_2gpu.yaml}` — 1 卡/2 卡版本的 `enforce_eager` 差异。
- `vllm_omni/quantization/int8_config.py` / `factory.py` — Int8 量化按 `LinearBase` 通用分派的实现。
- `api/gpustack/gpustack/schemas/inference_backend.py` / `worker/backends/vllm_omni.py` /
  `scheduler/scheduler.py` / `policies/candidate_selectors/vllm_omni_resource_fit_selector.py` —
  GPUStack 侧 `gpus_per_replica`/Category 手动配置机制的源头。
- `api/gpustack/docs/scripts/{lx2v-node.sh,lx2v-fleet.sh}` — 50 台舰队批量升级用的既有运维工具。
