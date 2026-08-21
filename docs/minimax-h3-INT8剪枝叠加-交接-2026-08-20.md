# MiniMax-H3 INT8 + 剪枝 r8 叠加方案 交接文档（2026-08-20）

> 目标读者：下一个接手这个任务的 agent。本文自包含，不依赖对话上下文。
> 前置背景文档：`docs/minimax-h3-剪枝前后对比-2026-08-20.md`（剪枝 r8 的 FL2VA/TP4 实测已收尾）。

## 1. 一句话结论

INT8（本地离线 W8A8）与剪枝 r8 **可以叠加，且二者互补**：剪枝只砍 AdaLN 调制层（2688→8），INT8 只量化剪枝没动的 attention/MLP 大层。两者合起来预计把 transformer 权重从满血 BF16 的 ~62 GiB 压到 ~20 GiB（TP4 每卡约省 10 GiB）。

当前唯一阻塞点：**剪枝 checkpoint 是 diffusers 格式，本地 INT8 工具只认 partition 格式**。需要先写一个 diffusers→partition 转换器，其余管线（剪枝加载、离线 INT8、deploy config）都已存在且各自验证过。

用户给的 ModelScope `Gluttony10/MiniMax-H3-INT8-CONVROT` 是第三方“旋转 INT8”单文件格式，**不接入**（见 §4）。

## 2. 为什么叠加是正解（权重拆解）

MiniMax-H3 Ref2VA 的 transformer 每层由三块组成，三块被两个方案分别覆盖：

- AdaLN 调制投影 `[96768, 2688]`：满血每层 ~520 MiB，50 层共 ~24 GiB。**剪枝把它降到 `[96768, 8]`，共 ~0.08 GiB**。INT8 不会碰它（官方 INT8 的 `ignored_layers` 明确保留 AdaLN，认为敏感）。
- attention（q/k/v/out，`[7168, 5376]` / `[5376, 7168]`）+ MLP（fc1 `[28672, 5376]`、fc2 `[5376, 14336]`）：每层 ~735 MiB，50 层共 ~36 GiB。**剪枝完全不碰它，INT8 把它压到 ~18 GiB**。
- token_refiner、final_layer、patch/condition 投影、norm 等：~2 GiB，两方案都保留 BF16。

磁盘占用实测/预估（transformer 权重）：

| 版本 | 磁盘 | TP4 每卡权重（÷4） |
|---|---:|---:|
| 满血 BF16（官方） | 61.7 GiB | ~15.4 GiB |
| 官方 INT8（未剪枝，已存在） | 43.8 GiB | ~11.0 GiB |
| 剪枝 BF16（已存在） | 37.5 GiB | ~9.4 GiB |
| **剪枝 + INT8（目标）** | **~20 GiB（预估）** | **~5.0 GiB** |

注意官方 INT8 只把 61.7→43.8 GiB（省 29%），因为 AdaLN 的 ~24 GiB 仍以 BF16 保留；剪枝恰好把这 24 GiB 拿走，所以二者叠加远强于各自单独。

## 3. 显存预期（对 ref2va 阶梯的意义）

已完成的剪枝阶梯（TP4、Turbo4/4 NFE、1344×768、40 GiB A100，官方 vs 剪枝）：

| 档位 | 官方 | 剪枝 r8 |
|---|---|---|
| 1图+3视频+1音频 8s | 36427 MiB | 32757 MiB |
| 6图+3视频 15s | OOM 40023 | 39761 MiB |
| 9图+3视频 8s | OOM 40183 | 40359 MiB |
| 9图+3视频 12s | OOM | OOM |
| 9图+3视频 15s | OOM | OOM |

剪枝实测只省 ~3.7 GiB/卡（小于权重差 ~6 GiB/卡），说明峰值里激活/KV 占大头。INT8 再省权重 ~4.4 GiB/卡（vs 剪枝 BF16），**能否把 9图 12s/15s 从 OOM 变成可用，只有实跑阶梯才知道**；若峰值是激活主导，INT8 也可能只是把 9图 8s 的余量从几十 MiB 扩到 ~4 GiB，生产更稳，但不一定能再上档。

## 4. 第三方 CONVROT 方案评估（结论：不用）

`https://modelscope.cn/models/Gluttony10/MiniMax-H3-INT8-CONVROT`（USER_UPLOAD，community，MiniMaxAI/MiniMax-H3 的 finetune 关系）：

- 文件是**单文件** `MiniMax-H3-Ref2VA-int8_convrot.safetensors`（34 GB）与 `MiniMax-H3-FL2VA-int8_convrot.safetensors`（34 GB），外加 `qwen3-vl-32b-int8_convrot.safetensors`（27 GB 文本编码器）和 VAE/LoRA。
- tensor 类型含 `I8/U8/F32/F16/BF16`，是“旋转（convrot）+ INT8”的权重表示；旋转后的权重值已不再是原始权重，推理端必须实现同一套旋转协议和 kernel 才能用。
- 当前 vLLM-Omni 的 H3 loader 只有 `quant_method: int8`（`DiffusionInt8Config`，per-output-channel 对称 + dynamic 激活）这一条离线路径，**没有 convrot 方法、没有对应 kernel、也没有该单文件/旋转协议的加载逻辑**；文本编码器换成 Qwen3-VL-32B convrot 更是另一整块集成。

结论：**不接 CONVROT**。用仓库自己的 `quantize_minimax_h3_int8.py`（官方 Ref2VA-INT8 已用它产出并服务过），只差“剪枝 diffusers → partition”这一步。若将来要追质量，可再单独评估 rotation 方案，但不在本轮范围。

## 5. 已核实的格式与路径

剪枝 checkpoint（diffusers 格式，共 637 张量、37.47 GiB）：

- `transformer/` 是符号链接指向 `MiniMax-H3-Ref2VA-Pruned-r8-Turbo4-BF16-transformer/`。
- `diffusion_pytorch_model.safetensors.index.json`（weight_map） + 14 个 `diffusion_pytorch_model-*-of-00014.safetensors` + `adaln_affine.safetensors`。
- `adaln_affine.safetensors` 只有 2 个张量：`adaln_basis [8,2688] F32`、`adaln_mean [2688] F32`。
- 其余 F32 缓冲（`time_embedder.table [1025,8]`、`norm_out.folded_bias`、`transformer_blocks.N.adaln_proj.folded_bias [96768]`）都在 14 个主 shard 里，weight_map 能定位。
- `config.json` 关键字段：`adaln_rank=8`、`time_table_size=1025`、`time_embed_dim=2688`、`hidden_size=5376`、`num_layers=50`、`ffn_dim=14336`、`num_attention_heads=56`、`attention_head_dim=128`、`num_refiner_layers=2`、`text_dim=5120`、`in_channels=24`、`audio_in_channels=32`、`patch_size=[1,2,2]`。

关键模型目录：

- 满血 BF16 vLLM：`/nfs-models/wuhanjisuan894/models/MiniMax-H3-Ref2VA-Turbo4-BF16-vLLM`（partition，`transformer/model.safetensors.index.json`，13 shard）
- 剪枝 BF16 vLLM：`/nfs-models/wuhanjisuan894/models/MiniMax-H3-Ref2VA-Pruned-r8-Turbo4-BF16-vLLM`（`transformer` 指到 diffusers 目录）
- 官方 INT8（未剪枝，已服务）：`/nfs-models/wuhanjisuan894/models/MiniMax-H3-Ref2VA-INT8`（partition + `config.json` 内含 `quantization_config`）
- 剪枝 diffusers transformer：`/nfs-models/wuhanjisuan894/models/MiniMax-H3-Ref2VA-Pruned-r8-Turbo4-BF16-transformer`

## 6. 执行计划（给下一个 agent）

1. **写转换器** `tools/minimax_h3_turbo/convert_pruned_to_partition.py`：diffusers 剪枝 → partition 格式（规则见 §7）。产出 `MiniMax-H3-Ref2VA-Pruned-r8-Turbo4-BF16-partition`。
2. **bit-parity 验证**：复用 `tools/minimax_h3_parity/verify_checkpoint_conversion.py`，对同一份剪枝权重跑 `--diffusers <剪枝diffusers> --partition <转换后partition> --heads 56`，必须 `verdict: checkpoints agree` 才继续。
3. **离线 INT8**：在 vllm-omni 镜像内跑 `python3 vllm_omni/quantization/tools/quantize_minimax_h3_int8.py --src <步骤1的partition根> --dst <剪枝INT8根>`。产出后检查 `config.json` 的 `quantization_config.ignored_layers` 是否包含所有 `blocks.N.adaln_proj.linear` 且不含任何 `blocks.N.attn.*/mlp.*`。
4. **组装 -vLLM 根**：给剪枝 INT8 包一个 `-vLLM` 目录（对称链接 text_encoder/VAE/processor/tokenizer + `model_index.json`）。`model_index.json` 直接复用剪枝 BF16 vLLM 的 `_minimax_h3` 元数据（`pruned` + `sigma_shift_scales` + `base_schedule` + `distilled`），只把 `pruned.transformer_index` 从 `diffusion_pytorch_model.safetensors.index.json` 改成 `model.safetensors.index.json`。
5. **TP4 阶梯压测**：同一请求集合（`/nfs-output/h3_pruned_eval/ref2va_scale_20260820/reqs/*.json`）跑 1i3v 8s / 6i3v 15s / 9i3v 8s / 9i3v 12s / 9i3v 15s，对照 §3 的剪枝 BF16 数据，得出新 OOM 边界。serve 命令不传 `--quantization`，用 `--deploy-config /deploy-configs/minimax_h3_ref2va_w8a8_a100_40g.yaml`。
6. **回写结论**：把“剪枝+INT8 vs 剪枝 BF16 vs 满血官方”的新峰值/边界写进 `docs/minimax-h3-剪枝前后对比-2026-08-20.md`，给最终部署选型建议（重点回答：ref2va 是否值得上 INT8+剪枝、能多给多少输入余量）。

## 7. 转换器精确规则（实现时照抄）

名称重映射（与 `minimax_h3_transformer.py` 的 `_DIFFUSERS_NAME_RENAMES` 完全一致，外加 `norm_out.folded_bias`）：

```
audio_proj_in.*          -> audio_patch_proj.*
audio_proj_out.*         -> final_layer.audio_out.*
context_embedder.*       -> condition_proj.*
norm_out.folded_bias     -> final_layer.adaln_proj.folded_bias
norm_out.linear.*        -> final_layer.adaln_proj.linear.*
norm_out.norm.*          -> final_layer.norm.*
proj_in.*                -> video_patch_proj.*
proj_out.*               -> final_layer.video_out.*
time_embedder.linear_1.* -> time_embedder.proj_in.*   (剪枝无此张量，留作兼容)
time_embedder.linear_2.* -> time_embedder.proj_out.*  (剪枝无此张量，留作兼容)
transformer_blocks.N.*   -> blocks.N.*
token_refiner.refiner_blocks.N.* -> token_refiner.blocks.N.*
.attn.norm_q.            -> .attn.q_norm.
.attn.norm_k.            -> .attn.k_norm.
.attn.to_out.0.          -> .attn.out_proj.
.ff.net.0.proj.          -> .mlp.fc1.
.ff.net.2.               -> .mlp.fc2.
```

张量布局变换（三个，都不能漏）：

1. **qkv 融合**：`*.attn.to_q/to_k/to_v.weight`（各自 `[7168, 5376]`）→ `*.attn.qkv_proj.weight` `[21504, 5376]`，按 **grouped per-head** 顺序交织：第 g 个头（g=0..55）为 `[q_g(128), k_g(128), v_g(128)]`。这是 `_reorder_grouped_qkv_to_qkv`（`num_query_groups=56, heads_per_group=1, head_dim=128`）的逆变换。**主块和 token_refiner 块都用同一规则**（token_refiner 也走 `arch.num_attention_heads=56`）。
2. **fc1 半块交换**：diffusers `ff.net.0.proj` 存 `[up, gate]`，partition/vLLM 用 `[gate, up]`。即把 `[28672, 5376]` 沿 dim0 上下两半对调。
3. **F32 缓冲原样保留**：`time_embedder.table`、`adaln_basis`、`adaln_mean`、`norm_out.folded_bias`、`transformer_blocks.N.adaln_proj.folded_bias` 保持 F32，不做任何 cast。

其余张量（norm、bias、`norm_out.linear`、`proj_in/out`、`context_embedder`、token_refiner 非 qkv 部分等）只改名、不改值、不改 dtype。

输出：

- `model.safetensors.index.json`（`metadata.total_size` 写真实字节数，`weight_map` 指向新 shard 名）。
- 若干 `model-*-of-*.safetensors`（新 shard 命名/分片可自由决定，推荐按原始 14 shard 对应切，避免单文件过大）。
- `config.json`：可以直接沿用剪枝 diffusers 的 `config.json`（`MiniMaxH3DiTArchConfig.from_mapping` 已支持 `ffn_dim/in_channels/freq_dim/...` 这些别名），但建议落成官方 INT8 风格的 canonical 字段名，并**保留 `adaln_rank=8`、`time_table_size=1025`**；不要写 `quantization_config`（交给步骤 3 的工具加）。

## 8. 风险与开放问题

- **端到端未验证**：剪枝加载路径和 INT8 加载路径各自验证过，但“剪枝 + INT8”组合没跑过。首次 smoke 务必用 NFE 硬校验（`rows video=`）和一次小图短时请求，确认 `post_load_weights` 的 F32 buffer 断言（`time_embedder.table/adaln_basis/adaln_mean/folded_bias`）在 INT8 下仍通过。
- **`should_quantize` 只匹配 `blocks.*` 前缀**：`token_refiner.blocks.*` 因前缀是 `token_refiner.` 不会被子串误量化，行为与官方 INT8 的 ignored_layers 一致（token_refiner 保留 BF16）。转换器无需特殊处理。
- **激活不变**：INT8 与剪枝都不减少 attention/MLP 的激活/KV；9图 12s/15s 若激活主导，INT8 也可能仍 OOM，需实测确认。
- **`model_index.json` 的 `pruned.transformer_index`** 字段是 provenance 元数据，不是加载依据；改成 `model.safetensors.index.json` 后，确认加载器不因该字段取值报错。
- **执行环境**：转换器/量化需要 torch + safetensors。gpu 机 host 的 `python3` 没有 `safetensors`（gpu36 实测 `ModuleNotFoundError`），须在 vllm-omni 镜像内跑。

## 9. 机器与已知坑

- 本轮只用 gpu31–gpu40（gpu21–30 已被 gpustack 纳管，不动）。
- 启动引擎必须加 `-e GLOO_SOCKET_IFNAME=lo`，否则 NCCL 建连可能超时（之前实测解决过）。
- 15s 参考视频上限在 `vllm_omni/.../reference_video.py:32` 的 `MINIMAX_H3_MAX_REFERENCE_DURATION = 15.0`，本轮不改。
- serve 离线 INT8 时**不要传 `--quantization`**（会触发 online 量化路径，把已序列化 INT8 再量一遍）。
- 评测脚本与聚合：`/nfs-output/h3_turbo_eval/run_vllm_turbo_eval.sh`、`run_vllm_eval_case.py`；结果目录 `/nfs-output/h3_pruned_eval/ref2va_scale_20260820/`。

## 10. 下一步建议（给决策者）

先做 §6 步骤 1–3（本地/镜像内、纯权重产物，风险低），产出剪枝 INT8 checkpoint 并用 parity 脚本证明转换无损；再做一次小图 smoke + 9图 8s 单档复测。只有拿到实测峰值后，再决定是否值得把 5 档阶梯全部重跑一遍，以及最终 ref2va 选型是否上“官方满血 + 输入限制”还是“剪枝 + INT8 放宽输入”。
