# GPUStack A100 集群模型副本与内存均衡部署规划

> 日期：2026-08-01  
> 集群：30 台 × 4 卡 NVIDIA A100 40GB，单机约 251 GiB RAM  
> 最终约束：`dev-gpustack-a100-0030` 保留为空；0029用于 Hunyuan Image 3第三副本；SeedVR2、Wan2.2-VACE 不部署；由管理员在 GPUStack 页面手动配置。

> **最终方案更新（2026-08-01）：以第9节“业务优先级加权方案”为最终配置依据。最新业务决策只保留0030为空闲节点；0029用于 Hunyuan Image 3第三副本。第2、3、8节为容量推演和历史备选，不用于最终配置。**

## 1. 结论

在保留 29、30 节点后，可用资源为 **28 台 / 112 卡**。将待上线的 19 个模型全部配置为 4 个物理节点副本，需要：

- 10 个单卡模型：`10 × 4 × 1 = 40` 卡；
- 3 个双卡模型：`3 × 4 × 2 = 24` 卡；
- 6 个四卡模型：`6 × 4 × 4 = 96` 卡；
- 合计 **160 卡**。

因此，“所有模型都部署到 4 个节点”在当前容量下不可行，缺少 **48 卡 / 12 台四卡节点**。如果还要保留 2 台空节点，集群总规模至少应为 **42 台**。

当前集群可落地的均衡方案是：

- 单卡模型：每个 4 副本、分布在 4 台物理机；
- 双卡模型：每个 4 副本、分布在 4 台物理机；
- 四卡独占模型：每个 2 副本、分布在 2 台物理机；
- 正好使用 112 卡，29、30 继续保留。

该方案让每个模型至少跨 2 台物理机；单卡、双卡模型跨 4 台。任意一台物理机故障后，路由仍有健康副本。

### 1.1 现网只读快照（2026-08-01 12:40 CST）

从 GPUStack 内置 PostgreSQL 的 `models`、`model_instances`、`workers` 表只读核验：当前所有已创建实例均为 `RUNNING`，共使用约 104/120 张 GPU，但还不是本文目标布局。

主要差异：

- `hunyuan-image-3` 当前为 4 个四卡副本，占用 0021～0024；目标降到 0021、0022 两个副本。
- `wan2.2-i2v` 当前为 0003、0004、0027 三个四卡副本；目标保留 0003、0004。
- `wan2.2-flf2v` 当前为 0028、0029；目标迁到 0027、0028，释放 0029。
- `qwen-image` 当前 6 副本集中在 0011～0013，每台两个；目标改为 0011、0012、0013、0025，每台一个。
- `qwen-image-edit` 当前 3 副本在 0011～0013；目标迁到 0018、0019、0020、0026并扩为 4 副本。
- `ltx2-v2a` 当前与 Bernini 同机部署在 0009、0010、0014、0015，是高 RAM 叠加；目标迁到 0016、0017、0023、0024。
- `ernie-image-turbo` 当前尚未创建；目标放在 0018、0019、0020、0026 的 `cuda:1`，与高主机内存、但使用另一张 GPU 的 Qwen Edit 配对。
- `seedvr2`、`wan2.2-vace` 均为 0 副本，保持不部署。

因此第 3 节是**目标配置**，不是对当前现网的描述；迁移必须按第 5 节顺序逐步执行。

## 2. 目标副本数

| 模型 | 单副本 GPU | 目标副本 | 占用 GPU | 物理节点数 |
|---|---:|---:|---:|---:|
| ace-step | 1 | 4 | 4 | 4 |
| audiox | 1 | 4 | 4 | 4 |
| indextts-2 | 1 | 4 | 4 | 4 |
| ltx2-v2a | 1 | 4 | 4 | 4 |
| qwen-image | 1 | 4 | 4 | 4 |
| qwen-image-edit | 1 | 4 | 4 | 4 |
| qwen3-tts | 1 | 4 | 4 | 4 |
| soulx-singer | 1 | 4 | 4 | 4 |
| z-image | 1 | 4 | 4 | 4 |
| ernie-image-turbo | 1 | 4 | 4 | 4 |
| bernini | 2 | 4 | 8 | 4 |
| moss-ttsd | 2 | 4 | 8 | 4 |
| moss-voicegen | 2 | 4 | 8 | 4 |
| hunyuan-image-3 | 4 | 2 | 8 | 2 |
| infinitetalk-480p | 4 | 2 | 8 | 2 |
| infinitetalk-720p | 4 | 2 | 8 | 2 |
| wan2.2-flf2v | 4 | 2 | 8 | 2 |
| wan2.2-i2v | 4 | 2 | 8 | 2 |
| wan2.2-t2v | 4 | 2 | 8 | 2 |
| **合计** |  | **64** | **112** | 28 台工作节点 |

SeedVR2、Wan2.2-VACE 保持 0 副本。

## 3. 精确节点与 GPU 规划

GPU 编号均为节点内的 `cuda:0`～`cuda:3`。

| 节点 | cuda:0 | cuda:1 | cuda:2 | cuda:3 | 内存搭配说明 |
|---|---|---|---|---|---|
| 0001 | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v | 四卡独占副本 1 |
| 0002 | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v | 四卡独占副本 2 |
| 0003 | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | 四卡独占副本 1 |
| 0004 | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | 四卡独占副本 2 |
| 0005 | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p | 四卡独占副本 1 |
| 0006 | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p | 四卡独占副本 2 |
| 0007 | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | 四卡独占副本 1 |
| 0008 | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | 四卡独占副本 2 |
| 0009 | bernini | bernini | z-image | ace-step | 高 RAM + 两个低 RAM |
| 0010 | bernini | bernini | z-image | ace-step | 高 RAM + 两个低 RAM |
| 0011 | moss-voicegen | moss-voicegen | qwen-image | indextts-2 | 低 RAM + 高 RAM + 低 RAM |
| 0012 | moss-voicegen | moss-voicegen | qwen-image | indextts-2 | 低 RAM + 高 RAM + 低 RAM |
| 0013 | moss-voicegen | moss-voicegen | qwen-image | indextts-2 | 低 RAM + 高 RAM + 低 RAM |
| 0014 | bernini | bernini | z-image | ace-step | 高 RAM + 两个低 RAM |
| 0015 | bernini | bernini | z-image | ace-step | 高 RAM + 两个低 RAM |
| 0016 | moss-ttsd | moss-ttsd | ltx2-v2a | audiox | 低 RAM + 高 RAM + 低 RAM |
| 0017 | moss-ttsd | moss-ttsd | ltx2-v2a | audiox | 低 RAM + 高 RAM + 低 RAM |
| 0018 | qwen-image-edit | ernie-image-turbo | qwen3-tts | soulx-singer | 高 RAM Qwen + 低 RAM ERNIE/语音 |
| 0019 | qwen-image-edit | ernie-image-turbo | qwen3-tts | soulx-singer | 高 RAM Qwen + 低 RAM ERNIE/语音 |
| 0020 | qwen-image-edit | ernie-image-turbo | qwen3-tts | soulx-singer | 高 RAM Qwen + 低 RAM ERNIE/语音 |
| 0021 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | 四卡独占、高 RAM 副本 1 |
| 0022 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | 四卡独占、高 RAM 副本 2 |
| 0023 | moss-ttsd | moss-ttsd | ltx2-v2a | audiox | 低 RAM + 高 RAM + 低 RAM |
| 0024 | moss-ttsd | moss-ttsd | ltx2-v2a | audiox | 低 RAM + 高 RAM + 低 RAM |
| 0025 | moss-voicegen | moss-voicegen | qwen-image | indextts-2 | 低 RAM + 高 RAM + 低 RAM |
| 0026 | qwen-image-edit | ernie-image-turbo | qwen3-tts | soulx-singer | 高 RAM Qwen + 低 RAM ERNIE/语音 |
| 0027 | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | 四卡独占副本 1 |
| 0028 | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | 四卡独占副本 2 |
| 0029 | 保留 | 保留 | 保留 | 保留 | 必须移除当前 wan2.2-flf2v 副本 |
| 0030 | 保留 | 保留 | 保留 | 保留 | 保持空闲 |

GPUStack 中每个模型使用 `spread`，并将 `gpu_selector.gpu_ids` 严格限制为表中对应 GPU。每个副本所需 GPU 数分别设置为 1、2 或 4。不要仅给出一批 GPU 后依赖调度器自由组合双卡/四卡副本；显式选择可避免同一模型的两个副本落到同一物理机。

## 4. 高内存原因与实测证据

以下为无业务请求时的只读快照；容器内存来自 cgroup，PSS/private 来自容器进程的 `/proc/*/smaps_rollup` 汇总。

| 模型/部署 | 节点现状 | cgroup 内存 | 关键证据 | 判断 |
|---|---:|---:|---|---|
| Bernini + LTX2-v2a | 0009 | 139.4 + 75.2 GiB | Bernini 匿名内存约 134.7 GiB；LTX 配置开启 DiT/VAE/Gemma CPU offload | 当前把两个高 RAM 流式模型放在一起，是 0009 达到 225.6 GiB 的主因 |
| Qwen Image ×2 + Edit ×1 | 0011 | 77.9 + 66.4 + 61.4 GiB | 三者配置均为 `cpu_offload=true, offload_granularity=block`；主机 shared 约 180 GiB、swap 已用约 3.3 GiB | 同节点堆叠三个 Qwen 不合理，应拆成每节点一个 |
| Hunyuan Image 3 | 0021 | 197.9 GiB | residence 配置为 AR/DiT 双引擎、`sleep_level: 1`；PSS 约 185.7 GiB | 属于为 4×A100 40G 设计的主机权重驻留，不是异常泄漏 |
| MOSS TTSD + VoiceGen | 0016 | 7.7 + 5.9 GiB | 整机仅约 25 GiB RAM | 适合作为高 RAM 模型的搭配项 |
| Z-Image | 0011 | 1.47 GiB | PSS 约 1.32 GiB | 适合与 Bernini 配对 |

### 4.0 是否应把模型尽量放入显存

原则不是统一打开或关闭 offload，而是优先采用各模型报告中已经验证的最快可用形态：

| 分类 | 模型 | 已验证结论 |
|---|---|---|
| **应常驻显存** | Z-Image | bf16 单卡峰值 21.8 GiB、热态 7.64 秒；int8 慢 2.86 倍，多卡只快 1.21 倍。应保持 bf16 单卡无 offload。 |
| **应常驻显存** | ACE-Step | XL Turbo + 4B LM 峰值 26.5 GiB，600 秒音乐仍可单卡运行，主机 Shmem 低于 200 MiB。 |
| **应常驻显存** | Qwen3-TTS | 常驻约 18 GiB，近满上下文峰值约 26 GiB；单卡足够。 |
| **主 DiT 应常驻显存** | Wan2.2 T2V/I2V/FLF2V | 现网为 4 卡 int8-triton，`cpu_offload=false`、仅 T5 在 CPU；这是报告验证的生产档。bf16 多卡会因每 rank 复制 CPU 权重导致主机 OOM。 |
| **主 DiT 应常驻显存** | InfiniteTalk 480p/720p | 现网为 4 卡 int8-triton，`cpu_offload=false`；短任务主机内存低。长音频内存上涨来自解码帧累计，不是权重流式搬运。 |
| **分阶段显存常驻** | MOSS TTSD / VoiceGen | 单卡会 OOM；生产方案是 talker 与 codec 两个 stage 各占一张卡，不应改成 CPU offload。 |
| **只能部分常驻** | Qwen Image/Edit | 58 GiB 模型无法完整塞入单卡。生产最优是 DiT block offload，但 Qwen2.5-VL 文本编码器必须留 GPU：`qwen25vl_cpu_offload=false`。merged8 峰值约 26.7 GiB、热态 17.0 秒；完全无 offload 的 int8 生成阶段实测 OOM。 |
| **只能部分常驻** | LTX2-v2a | Gemma bf16 24 GiB 与 DiT block 约 16 GiB 同时常驻时实测冲到 40431 MiB OOM；必须保留 block/VAE/Gemma offload。 |
| **流式反而更快** | Bernini | `BERNINI_BLOCK_OFFLOAD=buf` 约 100.6 秒；非块流式约 285.2 秒且显存升至约 31 GiB/卡。双专家整体换入换出的代价比逐块预取更高。 |
| **互斥阶段驻留** | Hunyuan Image 3 | AR 与 DiT 不能同时留在 4×40G；当前 level-1 sleep 让每次只唤醒一个引擎，是已验证的必要方案。 |
| **应常驻显存并限制尺寸** | ERNIE-Image-Turbo | 补充 A/B 已验证无 offload：1024×1536 PE off 为 6.20 秒、峰值约 36.9 GiB，连续 10 轮 PE on/off 不增长；CPU offload 同条件为 19.34 秒。生产关闭 offload，并把最大面积锁为 1,572,864 pixels。 |

所以应优化的是“**能安全常驻的组件留在 GPU**”，而不是把整个模型一律搬到 GPU。当前最典型的成功优化是 Qwen：DiT 仍需 block offload，但将文本编码器留在 GPU 后，T2I 从 28.2 秒降到 17.0 秒，Edit 曾从 481 秒降到约 38 秒冷态/21.6 秒热态。

相反，Wan/InfiniteTalk 已经是 int8 主 DiT 显存常驻，不属于高主机内存 offload 模型；它们占满四卡是为了低延迟与高分辨率，不能再与其他模型共卡。

### 4.1 Bernini

现网环境明确设置：

```text
BERNINI_BLOCK_OFFLOAD=buf
```

这会让两份 rank 的 DiT block 权重常驻 CPU，并逐块搬入 GPU。高 RAM 是该策略的直接代价。已有生产验证报告表明，在相同 848×480×81 案例中：

- 块流式约 100.6 秒，GPU 峰值约 13.8/12.6 GiB；
- 非块流式约 285.2 秒，GPU 峰值约 31/31 GiB；
- 输出逐帧一致，非块流式慢 2.83 倍。

因此现网不建议关闭 `buf`。短期优化是与低 RAM 模型配对。若业务愿意将规格限制在较短/较低分辨率，可另测单卡块流式档，理论上可把双 rank 权重副本和 GPU 消耗减半；在完成 720p/长帧回归前不能替换现网双卡档。

### 4.2 LTX2-v2a

现网配置同时启用：

```json
{
  "cpu_offload": true,
  "offload_granularity": "block",
  "vae_cpu_offload": true,
  "gemma_cpu_offload": true
}
```

这是单卡承载 22B DiT、Gemma 12B、VAE 的主要原因，也是约 75 GiB 主机内存的来源。不能直接关闭。可优化方向是分别 A/B `gemma_cpu_offload`、VAE 驻留和模型/块粒度，但必须采集推理阶段 GPU 峰值、首包和总时长；当前先按高 RAM 模型隔离。

### 4.3 Qwen Image / Edit

两套生产配置都启用了 block CPU offload。0011～0013 每台同时放置两个 Qwen Image 和一个 Qwen Edit，造成约 200 GiB 节点内存和 swap 压力。本规划将每台限制为一个 Qwen 系模型。

后续优化优先级：

1. 先完成拆散部署，观察 24 小时 RAM、swap、失败率；
2. 单独测试更高 GPU 驻留比例或关闭 block offload，确认 A100 40G 是否仍能覆盖目标分辨率；
3. 对比生成时长、GPU 峰值、主机 PSS、质量后再决定，不直接修改现网配置。

### 4.4 Hunyuan Image 3

这是验证过的双引擎互斥驻留方案。`sleep_level: 1` 在 AR/DiT 间保留可快速恢复的主机权重；level 2 当前不支持可靠的二次唤醒。因此约 198 GiB RAM 是换取 A100 40G 可运行和请求级 `bot_task` 的设计成本。该模型必须整机独占，不与任何其他模型搭配。

### 4.5 ERNIE-Image-Turbo

补充 A/B 已证明应关闭 CPU offload：

- 1024×1536、PE off：6.20 秒，峰值 36873 MiB；CPU offload 为 19.34 秒、约 29851 MiB；
- 10 轮 PE off/on 交替后显存稳定在 36915 MiB，主机内存约 3.2 GiB，无持续增长；
- 双请求被 `batch_size=1` 串行处理，没有合批显存峰值；
- 1152×1728 一度达到 40321 MiB，仅剩约 122 MiB，不具备生产余量。
- 逐层 offload 可将 1024×1536 峰值降到 22841 MiB，并跑通 2160×3840（约 52.9 秒、峰值约 38879～39331 MiB）；但 4K 海报出现明显纵向重复主体/文字，属于可运行而非可生产质量。

因此 GPUStack 参数不要传 `--enable-cpu-offload`，改传：

```text
--max-generated-image-size 1572864
```

ERNIE 已变成“高显存、低主机内存”模型，和高主机内存的 Qwen Edit 放在同一节点是合理搭配；二者使用不同 GPU，不存在显存叠加。

不要在生产实例里实现“接近 OOM 才动态 offload”。现有 vLLM-Omni 的 model-level/layerwise offload 都是启动时固定策略，切换会涉及权重驻留、torch.compile 图和 allocator 缓存。若未来有大图实验需求，应建立单独的 `--enable-layerwise-offload` 池；正式 4K 交付建议使用 ERNIE 低分辨率生成 + 独立超分，而不是让 ERNIE 原生生成 2160×3840。

## 5. 配置与迁移顺序

为避免一次性重建造成全路由不可用，按以下顺序逐模型调整，每一步等实例 `running/ready` 后再做下一步：

1. 将 Hunyuan Image 3 从 4 副本降为 2，固定在 0021、0022，释放 0023、0024。
2. 将 Wan2.2-I2V 固定为 0003、0004，释放 0027。
3. 将 Wan2.2-FLF2V 固定为 0027、0028，确认健康后清除 0029 上的副本；0029、0030 设置维护/保留标签，避免自动调度。
4. Bernini 保持 0009、0010、0014、0015；将 LTX2-v2a 从这些节点移走，再放入 Z-Image 和 ACE-Step。
5. 将 Qwen Image 收敛为 4 副本：0011、0012、0013、0025，每节点只保留一个。
6. 将 Qwen Edit 移至 0018、0019、0020、0026；释放 0011～0013 上的 Edit。
7. 按第 3 节配置 MOSS、LTX2、AudioX、IndexTTS 和其余语音模型。
8. ERNIE 不开启 CPU offload，配置 `--max-generated-image-size 1572864`；先在 0026:cuda:1 单副本验收，再依次扩到 0020、0019、0018。检查 1024×1536 后显存稳定在约 36.9 GiB，并确认超面积请求返回 HTTP 400。
9. 最终核对：SeedVR2/VACE 为 0；0029/0030 无业务容器；每个模型的副本位于不同 worker。

## 6. 上线后的监控阈值

- 主机 `MemAvailable`：建议长期大于 40 GiB；低于 30 GiB 告警。
- swap：部署完成后不应持续增长；持续增长说明高 RAM 配对过密。
- 每个实例：记录 idle、单请求峰值、并发请求峰值的 cgroup memory 和 GPU memory。
- 路由：单节点维护/断网演练后，确认所有模型仍有 ready 副本。
- 0029/0030：除 `gpustack-worker` 外不应存在模型容器。

## 7. 扩容方案

如果业务坚持 19 个模型全部达到 4 个物理节点副本：

- 活跃算力至少 160 卡，即 40 台四卡节点；
- 再保留 2 台空节点，总计至少 42 台；
- 相比当前 30 台需新增 12 台。

扩容前，当前方案优先保证所有模型都有跨机容灾，而不是让少数四卡模型占用四个节点、导致大量单卡/双卡模型无法达到四节点副本。

## 8. 备选方案：尽量统一为 3 个物理节点

如果目标从“单卡/双卡模型尽量 4 节点”改为“所有模型尽量 3 节点”，资源计算如下：

- 10 个单卡模型：`10 × 3 × 1 = 30` 卡；
- 3 个双卡模型：`3 × 3 × 2 = 18` 卡；
- 6 个四卡模型全部 3 节点：`6 × 3 × 4 = 72` 卡；
- 合计 120 卡，等于30台机器全部使用，无法保留 0029、0030。

在保留两个空节点、只有112卡可用的前提下，最均衡方案是：17个模型部署到3个物理节点，两个四卡高成本模型保留2节点。默认建议把高主机内存的 `hunyuan-image-3` 和高成本的 `infinitetalk-720p` 保留2副本，其余全部3副本。若业务优先级不同，可以将任意一个四卡模型的第三副本与它们一对一交换。

目标副本：

| 类型 | 模型 | 副本/物理节点 |
|---|---|---:|
| 单卡 | ace-step、audiox、indextts-2、ltx2-v2a、qwen-image、qwen-image-edit、qwen3-tts、soulx-singer、z-image、ernie-image-turbo | 各3 |
| 双卡 | bernini、moss-ttsd、moss-voicegen | 各3 |
| 四卡 | wan2.2-t2v、wan2.2-i2v、wan2.2-flf2v、infinitetalk-480p | 各3 |
| 四卡 | hunyuan-image-3、infinitetalk-720p | 各2 |
| 不部署 | seedvr2、wan2.2-vace | 0 |

精确节点规划：

| 节点 | cuda:0 | cuda:1 | cuda:2 | cuda:3 |
|---|---|---|---|---|
| 0001、0002、0015 | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v |
| 0003、0004、0024 | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v |
| 0005、0006、0026 | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p |
| 0007、0008 | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p |
| 0021、0022 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 |
| 0025、0027、0028 | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v |
| 0009、0010、0014 | bernini | bernini | z-image | ace-step |
| 0011、0012、0013 | moss-voicegen | moss-voicegen | qwen-image | indextts-2 |
| 0016、0017、0023 | moss-ttsd | moss-ttsd | ltx2-v2a | audiox |
| 0018、0019、0020 | qwen-image-edit | ernie-image-turbo | qwen3-tts | soulx-singer |
| 0029、0030 | 保留 | 保留 | 保留 | 保留 |

该方案正好使用112张GPU。所有单卡/双卡模型以及4个主要四卡模型都跨3台物理机；单机故障后至少还剩2个健康副本。`hunyuan-image-3` 和 `infinitetalk-720p` 单机故障后各剩1个副本，因此这两个模型应优先设置故障告警和维修恢复时限。

## 9. 业务优先级加权方案：重点模型多副本、保留1台空节点

根据新的业务优先级，增加 `bernini`、`hunyuan-image-3`、`qwen-image`，将 `wan2.2-t2v` 和 `infinitetalk-480p` 均设为2节点：

| 模型 | 目标副本/物理节点 | GPU/副本 | 总GPU |
|---|---:|---:|---:|
| bernini | 4 | 2 | 8 |
| hunyuan-image-3 | 3 | 4 | 12 |
| qwen-image | 5 | 1 | 5 |
| wan2.2-t2v | 2 | 4 | 8 |
| infinitetalk-480p | 2 | 4 | 8 |
| 其余9个单卡模型 | 各3 | 1 | 27 |
| moss-ttsd、moss-voicegen | 各3 | 2 | 12 |
| wan2.2-i2v、infinitetalk-720p、wan2.2-flf2v | 各3 | 4 | 36 |
| **合计** |  |  | **116** |

最终使用 **116张GPU/29台机器**，保留0030一台整机空闲。

精确节点规划：

| 节点 | cuda:0 | cuda:1 | cuda:2 | cuda:3 |
|---|---|---|---|---|
| 0001、0002 | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v |
| 0003、0004、0024 | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v |
| 0005、0006 | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p |
| 0007、0008、0026 | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p |
| 0021、0022、0029 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 |
| 0025、0027、0028 | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v |
| 0009 | bernini | bernini | z-image | ace-step |
| 0010 | bernini | bernini | z-image | audiox |
| 0014 | bernini | bernini | ace-step | audiox |
| 0015 | bernini | bernini | z-image | ace-step |
| 0011、0012、0013 | moss-voicegen | moss-voicegen | qwen-image | indextts-2 |
| 0016、0017 | moss-ttsd | moss-ttsd | ltx2-v2a | qwen-image |
| 0023 | moss-ttsd | moss-ttsd | ltx2-v2a | audiox |
| 0018、0019、0020 | qwen-image-edit | ernie-image-turbo | qwen3-tts | soulx-singer |
| 0030 | 保留 | 保留 | 保留 | 保留 |

该布局让5个 `qwen-image` 副本分别位于0011、0012、0013、0016、0017，没有同机重复，也避免与约139GiB主机内存的 Bernini 同机；0016、0017上的 Qwen Image 与约75GiB的 LTX2、低内存 MOSS TTSD 配对，预计总主机内存仍低于251GiB，但迁移后必须观察24小时 PSS、MemAvailable和swap。Bernini 4副本也各自独立物理机。Hunyuan Image 3的第三副本放到0029，0030作为唯一整机保留节点。

`infinitetalk-480p` 保留0005、0006两个物理节点副本，单机故障后仍有一个副本可用。由于只剩0030一台空节点，任何模型扩容、迁移或故障补副本都应优先使用0030，并在操作完成后恢复其空闲状态。

### 9.1 最终模型副本表

| 模型 | GPU/副本 | 最终副本数 | 物理节点 | 总GPU |
|---|---:|---:|---|---:|
| ace-step | 1 | 3 | 0009、0014、0015 | 3 |
| audiox | 1 | 3 | 0010、0014、0023 | 3 |
| bernini | 2 | 4 | 0009、0010、0014、0015 | 8 |
| hunyuan-image-3 | 4 | 3 | 0021、0022、0029 | 12 |
| indextts-2 | 1 | 3 | 0011、0012、0013 | 3 |
| infinitetalk-480p | 4 | 2 | 0005、0006 | 8 |
| infinitetalk-720p | 4 | 3 | 0007、0008、0026 | 12 |
| ltx2-v2a | 1 | 3 | 0016、0017、0023 | 3 |
| moss-ttsd | 2 | 3 | 0016、0017、0023 | 6 |
| moss-voicegen | 2 | 3 | 0011、0012、0013 | 6 |
| qwen3-tts | 1 | 3 | 0018、0019、0020 | 3 |
| qwen-image | 1 | 5 | 0011、0012、0013、0016、0017 | 5 |
| qwen-image-edit | 1 | 3 | 0018、0019、0020 | 3 |
| soulx-singer | 1 | 3 | 0018、0019、0020 | 3 |
| wan2.2-flf2v | 4 | 3 | 0025、0027、0028 | 12 |
| wan2.2-i2v | 4 | 3 | 0003、0004、0024 | 12 |
| wan2.2-t2v | 4 | 2 | 0001、0002 | 8 |
| z-image | 1 | 3 | 0009、0010、0015 | 3 |
| ernie-image-turbo | 1 | 3 | 0018、0019、0020 | 3 |
| seedvr2 | — | 0 | 不部署 | 0 |
| wan2.2-vace | — | 0 | 不部署 | 0 |
| **总计** |  | **58个实例** | **29台业务节点** | **116** |

### 9.2 最终30台节点逐卡表

| 节点 | cuda:0 | cuda:1 | cuda:2 | cuda:3 | 说明 |
|---|---|---|---|---|---|
| 0001 | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v | 四卡副本 |
| 0002 | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v | wan2.2-t2v | 四卡副本 |
| 0003 | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | 四卡副本 |
| 0004 | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | 四卡副本 |
| 0005 | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p | 四卡副本 |
| 0006 | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p | infinitetalk-480p | 四卡副本 |
| 0007 | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | 四卡副本 |
| 0008 | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | 四卡副本 |
| 0009 | bernini | bernini | z-image | ace-step | 高RAM + 低RAM |
| 0010 | bernini | bernini | z-image | audiox | 高RAM + 低RAM |
| 0011 | moss-voicegen | moss-voicegen | qwen-image | indextts-2 | Qwen + 低RAM |
| 0012 | moss-voicegen | moss-voicegen | qwen-image | indextts-2 | Qwen + 低RAM |
| 0013 | moss-voicegen | moss-voicegen | qwen-image | indextts-2 | Qwen + 低RAM |
| 0014 | bernini | bernini | ace-step | audiox | 高RAM + 低RAM |
| 0015 | bernini | bernini | z-image | ace-step | 高RAM + 低RAM |
| 0016 | moss-ttsd | moss-ttsd | ltx2-v2a | qwen-image | 部署后重点监控RAM |
| 0017 | moss-ttsd | moss-ttsd | ltx2-v2a | qwen-image | 部署后重点监控RAM |
| 0018 | qwen-image-edit | ernie-image-turbo | qwen3-tts | soulx-singer | 高RAM + 低RAM |
| 0019 | qwen-image-edit | ernie-image-turbo | qwen3-tts | soulx-singer | 高RAM + 低RAM |
| 0020 | qwen-image-edit | ernie-image-turbo | qwen3-tts | soulx-singer | 高RAM + 低RAM |
| 0021 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | 四卡独占、高RAM |
| 0022 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | 四卡独占、高RAM |
| 0023 | moss-ttsd | moss-ttsd | ltx2-v2a | audiox | 高RAM + 低RAM |
| 0024 | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | wan2.2-i2v | 四卡副本 |
| 0025 | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | 四卡副本 |
| 0026 | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | infinitetalk-720p | 四卡副本 |
| 0027 | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | 四卡副本 |
| 0028 | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | wan2.2-flf2v | 四卡副本 |
| 0029 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | hunyuan-image-3 | 第三副本、高RAM |
| 0030 | 保留 | 保留 | 保留 | 保留 | 唯一空闲应急节点 |

### 9.3 GPUStack 页面逐模型配置清单

GPU索引从0开始。双卡/四卡模型的同一行括号表示一个副本必须同时选择这些GPU；所有模型使用 `spread`，并将 `gpu_selector.gpu_ids` 限定为下表节点和GPU。

| GPUStack模型名称 | 副本数 | 节点与GPU（从0开始） |
|---|---:|---|
| `ace-step` | 3 | `dev-gpustack-a100-0009:[3]`；`dev-gpustack-a100-0014:[2]`；`dev-gpustack-a100-0015:[3]` |
| `audiox` | 3 | `dev-gpustack-a100-0010:[3]`；`dev-gpustack-a100-0014:[3]`；`dev-gpustack-a100-0023:[3]` |
| `bernini` | 4 | `dev-gpustack-a100-0009:[0,1]`；`dev-gpustack-a100-0010:[0,1]`；`dev-gpustack-a100-0014:[0,1]`；`dev-gpustack-a100-0015:[0,1]` |
| `hunyuan-image-3` | 3 | `dev-gpustack-a100-0021:[0,1,2,3]`；`dev-gpustack-a100-0022:[0,1,2,3]`；`dev-gpustack-a100-0029:[0,1,2,3]` |
| `indextts-2` | 3 | `dev-gpustack-a100-0011:[3]`；`dev-gpustack-a100-0012:[3]`；`dev-gpustack-a100-0013:[3]` |
| `infinitetalk-480p` | 2 | `dev-gpustack-a100-0005:[0,1,2,3]`；`dev-gpustack-a100-0006:[0,1,2,3]` |
| `infinitetalk-720p` | 3 | `dev-gpustack-a100-0007:[0,1,2,3]`；`dev-gpustack-a100-0008:[0,1,2,3]`；`dev-gpustack-a100-0026:[0,1,2,3]` |
| `ltx2-v2a` | 3 | `dev-gpustack-a100-0016:[2]`；`dev-gpustack-a100-0017:[2]`；`dev-gpustack-a100-0023:[2]` |
| `moss-ttsd` | 3 | `dev-gpustack-a100-0016:[0,1]`；`dev-gpustack-a100-0017:[0,1]`；`dev-gpustack-a100-0023:[0,1]` |
| `moss-voicegen` | 3 | `dev-gpustack-a100-0011:[0,1]`；`dev-gpustack-a100-0012:[0,1]`；`dev-gpustack-a100-0013:[0,1]` |
| `qwen3-tts` | 3 | `dev-gpustack-a100-0018:[2]`；`dev-gpustack-a100-0019:[2]`；`dev-gpustack-a100-0020:[2]` |
| `qwen-image` | 5 | `dev-gpustack-a100-0011:[2]`；`dev-gpustack-a100-0012:[2]`；`dev-gpustack-a100-0013:[2]`；`dev-gpustack-a100-0016:[3]`；`dev-gpustack-a100-0017:[3]` |
| `qwen-image-edit` | 3 | `dev-gpustack-a100-0018:[0]`；`dev-gpustack-a100-0019:[0]`；`dev-gpustack-a100-0020:[0]` |
| `soulx-singer` | 3 | `dev-gpustack-a100-0018:[3]`；`dev-gpustack-a100-0019:[3]`；`dev-gpustack-a100-0020:[3]` |
| `wan2.2-flf2v` | 3 | `dev-gpustack-a100-0025:[0,1,2,3]`；`dev-gpustack-a100-0027:[0,1,2,3]`；`dev-gpustack-a100-0028:[0,1,2,3]` |
| `wan2.2-i2v` | 3 | `dev-gpustack-a100-0003:[0,1,2,3]`；`dev-gpustack-a100-0004:[0,1,2,3]`；`dev-gpustack-a100-0024:[0,1,2,3]` |
| `wan2.2-t2v` | 2 | `dev-gpustack-a100-0001:[0,1,2,3]`；`dev-gpustack-a100-0002:[0,1,2,3]` |
| `z-image` | 3 | `dev-gpustack-a100-0009:[2]`；`dev-gpustack-a100-0010:[2]`；`dev-gpustack-a100-0015:[2]` |
| `ernie-image-turbo` | 3 | `dev-gpustack-a100-0018:[1]`；`dev-gpustack-a100-0019:[1]`；`dev-gpustack-a100-0020:[1]` |
| `seedvr2` | 0 | 不部署 |
| `wan2.2-vace` | 0 | 不部署 |

GPUStack页面填写要点：单卡模型设置 `gpus_per_replica=1`，`bernini`、`moss-ttsd`、`moss-voicegen` 设置为2，其余表中四卡模型设置为4。副本数必须与表中一致；不要只选择节点集合让调度器自行组合跨节点GPU。

### 9.4 Hunyuan Image 3 现网内存复核（2026-08-01）

部署完成后，GPUStack 在 0021、0022、0029 上显示约 76%～83% 的容器内存占用。只读检查结果如下：

| 节点 | 容器内存 | MemAvailable | 共享内存 | 匿名内存 | Swap已用 | 重启/OOM |
|---|---:|---:|---:|---:|---:|---|
| 0021 | 208.3 GiB（82.96%） | 42 GiB | 172 GiB | 19.8 GiB | 353 MiB | 0 / false |
| 0022 | 204.4 GiB（81.39%） | 42 GiB | 172 GiB | 19.8 GiB | 43 MiB | 0 / false |
| 0029 | 191.6 GiB（76.30%） | 43 GiB | 172 GiB | 18.3 GiB | 1.8 GiB | 0 / false |

三个实例均从同一时间启动并保持运行，空闲时四张 GPU 各约 2871 MiB，采样期间 `vmstat` 的 `si/so` 均为0。容器 cgroup 的高占用主要来自约172 GiB的共享内存权重映射及相关文件页；Linux `free` 列很低不代表只剩2～3 GiB可用，应以 `MemAvailable` 为主要压力指标。

结论：当前约83%属于 Hunyuan Image 3 的 `sleep_level: 1` 双引擎主机权重驻留成本，与此前约198 GiB基线一致，尚无逐请求泄漏或 OOM 证据。三台机器继续整机独占，不再混部其他模型，也不建议为了降低页面百分比清理页缓存、关闭 swap 或切到已验证二次唤醒不可靠的 level 2。

现网告警建议：

- `MemAvailable < 30 GiB` 持续5分钟：告警；低于20 GiB：严重告警；
- `vmstat si/so` 持续非0或 SwapUsed 随请求持续上升：检查内存压力；
- 每轮请求结束后容器内存呈阶梯式增长且不回落/不封顶：按疑似泄漏处理；
- 容器 `RestartCount > 0`、`OOMKilled=true` 或 worker 失联：立即摘除该副本。
