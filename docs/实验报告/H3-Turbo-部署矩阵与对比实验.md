# MiniMax-H3 Turbo 部署矩阵与对比实验

> 状态：本轮核心部署、步数、Diffusers 等价性和媒体条件矩阵已完成；本文不代表现网默认值已经改变。

## 1. 产品目标与命名边界

计划面向用户提供 FL2VA、Ref2VA 两类任务，并让用户在 Base、低步数 Turbo 和较高步数 Turbo 之间权衡画质与耗时。

权重版本与请求步数是两个不同概念：

- `Turbo4` 表示 4 NFE 蒸馏权重，不表示接口只能接收 4 步。
- `Turbo8` 表示独立的 8 NFE 蒸馏权重。
- `Turbo4 @ 8 NFE` 可以运行，但不能标成 `Turbo8`；它仍使用 Turbo4 权重。
- Base 默认 20 NFE，同时允许用户显式选择 30、50 等其他步数。
- Turbo 接口也保留 `num_inference_steps`；高于推荐 NFE 不报错，但收益需要用实验判断，不能默认宣称画质必然提高。

当前上游仓库发布情况：

| 任务 | 上游权重 | 推荐 NFE | 当前可用性 |
|---|---|---:|---|
| FL2VA / T2VA | Turbo4 768p v1.0 | 4 | 已下载、已组装、已实跑 |
| FL2VA / T2VA | Turbo8 v1.0 | 8 / 4 | 已下载、已组装、已实跑 |
| Ref2VA | Turbo4 v0.1 | 4 | 已下载、已组装、已实跑 4/8 NFE |
| Ref2VA | 独立 Turbo8 | - | 上游当前未发布，不能用 Turbo4 冒充 |

因此“六个版本”在权重层面目前只有五个有明确来源。若产品必须先暴露第六个选择，应准确标注为 `Ref2VA Turbo4（8 NFE）`；若要命名为 `Ref2VA Turbo8`，必须取得或训练独立的 Turbo8 权重。

## 2. 官方 Turbo 步数语义

官方 Turbo 推理脚本只校验 `inference_steps >= 1`，没有把 Turbo4 限定为只能输入 4。脚本将用户请求的 NFE 转换为：

```text
scheduler_grid_points = inference_steps + 1
```

原因是 MiniMax-H3 scheduler 的 sigma 列表包含终点 0；`N + 1` 个边界形成准确的 `N` 个 transformer 区间。

历史 vLLM-Omni 路径曾用 `N` 个边界，因此界面显示 4/8/20 步时实际只执行 3/7/19 次 transformer。实验分支已修正为准确 NFE，并通过日志逐 rank 计数验证：TP4 下 4 NFE 产生 16 条 step 记录，8 NFE 产生 32 条。

## 3. 固化资产

可复用 vLLM 模型 overlay：

```text
/nfs-models/wuhanjisuan894/models/MiniMax-H3-FL2VA-Turbo4-768p-BF16-vLLM
/nfs-models/wuhanjisuan894/models/MiniMax-H3-FL2VA-Turbo8-BF16-vLLM
/nfs-models/wuhanjisuan894/models/MiniMax-H3-Ref2VA-Turbo4-BF16-vLLM
```

官方 LoRA：

```text
/nfs-models/wuhanjisuan894/models/MiniMax-H3-Turbo-LoRA/
```

官方 Diffusers FL2VA BF16 oracle：

```text
/nfs-models/wuhanjisuan894/models/MiniMax-H3-Diffusers-FL2VA-BF16
```

实验请求、启动器和结果：

```text
/nfs-output/h3_turbo_eval/
```

`/nfs-output` 只保存本轮结果；模型权重均位于持久化的 `/nfs-models`。

## 4. 已完成的 vLLM-Omni TP4 性能结果

条件：4×A100、同一服务进程内热请求、640×384、5 秒、seed 42。冷请求包含首次编译，不与热请求混用。

| 权重 | 请求 NFE | 场景 | 热请求墙钟时间 | 峰值显存/卡 | 实际 step 日志 |
|---|---:|---|---:|---:|---:|
| FL2VA Base | 20 | 双关键帧、无人提示词 | 58.13 s | 27,755 MiB | 80 |
| FL2VA Turbo4 768p | 4 | 双关键帧、无人提示词 | 24.20 s | 27,755 MiB | 16 |
| FL2VA Turbo4 768p | 8 | 双关键帧、无人提示词 | 30.06 s | 27,775 MiB | 32 |
| FL2VA Turbo4 768p | 4 | T2VA 双人物动作 | 28.06 s | 27,695 MiB | 16 |
| FL2VA Turbo4 768p | 8 | T2VA 双人物动作 | 30.54 s | 27,695 MiB | 32 |
| FL2VA Turbo8 | 4 | 双关键帧、无人提示词 | 24.06 s | 27,755 MiB | 16 |
| FL2VA Turbo8 | 8 | 双关键帧、无人提示词 | 32.08 s | 27,755 MiB | 32 |
| FL2VA Turbo8 | 8 | T2VA 双人物动作 | 34.09 s | 27,695 MiB | 32 |

初步可确认：

- Turbo4 的 4 与 8 NFE 都真实生效；8 NFE 不是被接口忽略。
- 同权重从 4 增至 8 NFE 时，总耗时没有翻倍，因为文本编码、VAE 编解码和调度存在固定开销。
- 同条件下 Base 20 NFE 的热耗时是 Turbo4 4 NFE 的 2.40 倍、Turbo8 8 NFE 的 1.81 倍；三者显存和主机内存接近，所以 Turbo 当前主要节省计算时间和能耗，不减少常驻权重体积。
- 当前单样本中，Turbo4 8 NFE 的双人物轮廓比 4 NFE略稳定，但动作语义仍未完整执行；不能据此宣称“多跑必然更好”。
- 无人物的双野猪关键帧样本中，Base 20、Turbo4 4/8 和 Turbo8 4/8 都没有生成中间走过的女人；之前的“凭空人物”不是由是否采用 TP4 单独决定。

## 5. Ref2VA Turbo4 与参考图几何 A/B

条件：同一 Ref2VA Turbo4 v0.1 权重、4×A100、864×480、5 秒、同一张 1664×656 人脸参考图、同一提示词。每个性能样本均顺序执行，TP4 下 4 NFE 为 16 条 step 日志，8 NFE 为 32 条。

| 参考图几何 | seed | 请求 NFE | 热请求墙钟时间 | 峰值显存/卡 | 实际 step 日志 |
|---|---:|---:|---:|---:|---:|
| legacy：先拉伸到 864×480 | 42 | 4 | 54.13 s | 29,213 MiB | 16 |
| legacy：先拉伸到 864×480 | 43 | 4 | 56.12 s | 29,193 MiB | 16 |
| Turbo `match`：保比例、面积不超过目标画布 | 42 | 4 | 34.84 s | 28,153 MiB | 16 |
| Turbo `match`：保比例、面积不超过目标画布 | 43 | 4 | 36.08 s | 28,153 MiB | 16 |
| Turbo `match`：保比例、面积不超过目标画布 | 42 | 8 | 48.20 s | 28,153 MiB | 32 |

两颗 seed 的抽帧结论一致：legacy 输出的人脸明显沿输出画布方向变窄、变长；`match` 输出保留了原参考图更宽的人脸比例。`match` 同时将 4 NFE 热请求从 54–56 秒降到 35–36 秒，显存峰值减少约 1.0 GiB/卡。本轮因此支持保留“参考图不预拉伸、Turbo 默认采用 match 面积策略”，不支持保留 legacy pre-stretch。

并发污染说明：更早一组后台请求与前台请求重叠，出现 90.59/142.96 秒和 28/32 条新增 step 日志。这两条仅保留为问题证据，已从性能表排除；上表全部是重跑后的单请求结果。

## 6. 官方 Diffusers Turbo oracle

官方脚本 commit 固定为 `a7e148b`。4×A100-40G 的 FSDP2 路径完成 LoRA 融合和两大模型分片后，在复制未分片 VAE 等组件时达到 39.47 GiB/卡并 OOM；失败位置是组件放置，不是权重或采样。

随后用 Diffusers 官方 group-offload hook 重跑同一官方 CLI、LoRA loader、pipeline forward 和输出编码：文本编码器 leaf offload、DiT block offload，VAE/audio VAE 留在单卡。该路径不改权重、采样和计算图，只用于画质 oracle，不用于性能比较。960×544、124 帧、seed 42、4 NFE 的无人双野猪请求已成功出片：

```text
/nfs-output/h3_turbo_eval/official/turbo4_768_boars_group/0000_fl2va_seed42.mp4
```

五张抽帧中没有女人或其他新增人物。vLLM-Omni 也已补跑完全相同的 960×544、124 帧、seed 42、4 NFE 请求：

| vLLM 契约 | 热请求墙钟 | 峰值显存/卡 | 对 Diffusers SSIM | 对 Diffusers PSNR |
|---|---:|---:|---:|---:|
| legacy | 34.91 s | 28,513 MiB | 0.8941 | 27.24 dB |
| `official_diffusers_v1` 热跑1 | 42.46 s | 28,513 MiB | 0.9705 | 32.04 dB |
| `official_diffusers_v1` 热跑2 | 36.63 s | 28,513 MiB | 0.9705 | 32.04 dB |

两次 official 契约请求的 MP4 SHA256 一致，证明同实例、同 seed 可确定复现。第二次热跑与 legacy 的稳态差距为 1.72 秒（4.9%）；第一次的 42.46 秒包含还未稳定的首次执行/缓存成本，不能当成稳态惩罚。

第二次 official 热跑的引擎阶段拆分为：提示词编码 4.22 秒、DiT 去噪 22.50 秒、VAE 解码 3.56–3.71 秒、模型 forward 约 31.0 秒、引擎阶段 32.78 秒；另有约 3.85 秒用于保存 MP4 和任务轮询。因此稳态耗时的最大头是 DiT 去噪，其次是文本编码与 VAE 解码。

契约对齐将输出显著推近官方 Diffusers，说明 TP4 并行本身不是主差异源。但 SSIM/PSNR 只衡量“像不像官方”，不能证明主观画质或提示词遵从更好。本轮抽帧中两者主体、构图和色调均正常，没有证据支持为了画质把完整 official 契约设为产品默认。

## 7. Ref2VA 异构媒体端到端验证

同一 Ref2VA Turbo4 实例上，使用同一张图、同一段视频、同一段立体声音频、同一提示词与 seed 42，只交换 `references` 的顺序：

| references 顺序 | 墙钟 | 峰值显存/卡 | 实际 step 日志 |
|---|---:|---:|---:|
| video → image → audio（首次形状） | 128.28 s | 30,525 MiB | 16 |
| audio → image → video（热跑） | 76.16 s | 30,553 MiB | 16 |

引擎日志确认两种顺序均被原样传入，并均完成参考视频的流式无损取帧、VAE 条件编码和单次音频重采样。两份结果都是 640×384、24 fps、124 帧，都含 32 kHz 立体声 AAC，视频时长 5.167 秒，音频时长 5.207 秒。

两份输出的视频 SSIM 为 0.7736，证明引用顺序对视觉条件结果确实生效；音频通道 PSNR 约 168.5 dB，即供给的音频内容基本一致。抽帧中两种顺序的主体与动作都合理，顺序改变主要影响亮度、细节和位置，不足以宣称某一顺序画质更高。

## 8. 尚存的产品决策边界

- 完整 official 契约已证明更接近 Diffusers，但没有证明比 legacy 主观画质更好，因此继续作为对照开关，不直接切现网默认。
- Base/Turbo 的耗时、资源与固定样例画面已完成对照；单样本不足以给出通用主观画质排名。
- 独立 Ref2VA Turbo8 仍因上游未发布权重而不存在，这是资产边界，不是 vLLM 适配缺陷。

## 9. 当前保留建议

下表区分“发布基线”、“新建部署默认”和“实验开关”；不再把确定性修复与仅供对照的契约改动统称为“暂不上线”。

| 改动 | 当前建议 | 依据 |
|---|---|---|
| 准确 NFE（N+1 sigma 边界） | 纳入发布基线，不设兼容开关 | 修复 4/8/20 实际变成 3/7/19 的确定性错误；GPU step 计数已通过 |
| Base/Turbo 分区元数据、权重 overlay、来源记录 | 作为部署资产保留 | 三套已发布 Turbo 权重均可独立加载，由模型路径选择，原 Base 权重不被覆盖 |
| legacy/实验契约实例级隔离与启动日志 | 保留 | 默认行为不变，并能复现每次实验到底启用了哪些轴 |
| Ref2VA 参考图保比例 + Turbo `match` | 新建 Ref2VA Base/Turbo 部署默认开启 | 两 seed 都改善人物比例，同时降低耗时和显存；新部署无需继承 legacy pre-stretch，已有实例仍可留在 legacy 回滚档 |
| 完整 official RNG / condition-shape | 不作为产品默认 | 多 seed 只证明输出会变，没有系统性质量提升，效果近似换 seed |
| 官方固定短边 2048 | 不作为 Turbo 默认 | 资源成本高；Turbo 上游默认 `match`，本轮 `match` 也更快更省显存 |
| 无损流式参考视频、单次音频重采样 | 保留候选，不直上线 | CPU 用例和 GPU 端到端都已通过；两种顺序都产生完整音视频 |
| 有序异构 references | 保留候选，不直上线 | API/CPU/GPU 都已验证，输出对顺序有可测差异；上线前仍需 GPUStack 门面透传 |
| official 默认 124 帧及完整时长/准入语义 | 仅保留为 oracle/实验契约 | 产品没有复现 Diffusers 默认行为的硬要求，切换会改变既有调用语义 |
