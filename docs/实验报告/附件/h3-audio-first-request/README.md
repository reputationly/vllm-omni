# 附件：MiniMax-H3 音频首请求偏差的探针与脚本

配套正文：`docs/实验报告/MiniMax-H3-音频首请求偏差-排查记录.md`。
这些是 2026-10-14 那次排查的原始脚本，**按当时的样子留存**，不是可复用的工具。

## 文件

| 文件 | 作用 |
| --- | --- |
| `probe_audiovae.py` | **最小复现**：只加载音频 VAE（605 MB，无 DiT/文本编码器），同一 latent 连解 4 次。秒级。 |
| `probe_audiovae2.py` | 机制与修法候选。 |
| `probe_audiovae3.py` | 验证修法原语：一次极小 `remote.decode()` 能否吸收掉瞬态。 |
| `probe_audiovae4.py` | 模拟**服务路径**：CPU-offload stager，每请求一次 load/decode/offload 循环。 |
| `probe_audiovae5.py` | **发现等价性陷阱的那个**：比对四种构造，暴露 `decode_only` 决定能否复现。 |
| `patch_audio_probe.py` | 给 `decode_latent` 打点（latent 哈希 / 波形哈希 / 8 个 torch 全局开关 / 设备精度）。环境变量驱动，一份树跑两臂。 |
| `instrument_warmup.py` | 给当时那个（已回滚的）warm-up 打点，证明它在真机上确实执行了。 |
| `send3.py` | 等服务就绪后连发 N 个同 seed 请求并落盘。 |
| `launch_h3_audio.sh` | A/B 两臂启停（A 不打确定性包裹，B 打）。 |

## 直接跑之前必读

1. **路径**：脚本里的 `/nfs-output/qfix-20261014/...` 是当时的 scratch 目录，
   **已在收尾清理中删除**。重跑前把 `COMPONENT`、`WATCH` 之类的常量指到当前环境。
2. **`probe_audiovae5.py` 会引用 `vae._first_decode_absorbed`**——那是当时那版
   **已回滚**的修法引入的开关。跑在干净树上请删掉 `if not use_warmup:` 那两行。
3. **别拿隔离结果当真是机结论**：这正是 §5 那个陷阱。
   核对 `decode_only` 之前，隔离里"验证通过"不算数。
