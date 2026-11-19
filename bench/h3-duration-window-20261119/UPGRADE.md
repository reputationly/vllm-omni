# GPUStack 换镜像 / 升级指引（2026-11-19 核对）

> 依据：`/Users/reputationly/Desktop/code/api/gpustack/docs`（`user-guide/inference-backend-management.md`、
> `user-guide/model-deployment-management.md`、`upgrade/cluster.md`）+ devm 现网只读核对 + `gpustack` 源码。
> 目标场景：把 `minimax-h3-ref2va` / `minimax-h3-fl2va` 换成 `arm64-a100-20261119-0355-7ebda7f7`（时长窗口 [1,20]）。

---

## 1. 三份文档各自管什么（先别搞混）

| 文档 | 管的事 |
| --- | --- |
| `user-guide/inference-backend-management.md` | **模型镜像放在哪**：`backend → version_configs.<version>.image_name`；内置 backend 加自定义版本时版本名必须以 `-custom` 结尾；部署时可用 `image_name`/`run_command` 做单部署覆盖（"Flexible Testing Deployment"） |
| `user-guide/model-deployment-management.md` | **改完怎么生效**：编辑部署**不会**作用到已运行的实例，必须删除旧 model instance，GPUStack 才会按新配置重建（`## Edit Model Deployment` 的 note、`## Delete Model Instance` 的 note） |
| `upgrade/cluster.md` | **升 GPUStack 自身（server/worker）**，不是模型镜像。⚠️ 它警告的是：fork 的 worker 和 server 必须是同一 build（用 image labels 比对 version/revision），**不要**给 fork worker 换上游同版本号镜像，否则调度全 pending（2026-11-15 事故） |

## 2. 现网实况（devm，2026-11-19 只读核对）

```
inference_backends: id=14  backend_name=vLLMOmni  default_version=1.0.0
  version_configs."1.0.0".image_name = crpi-…/reputationly/vllm-omni:arm64-a100-latest   ← 浮动 tag
  version_configs."1.0.0".run_command/entrypoint/env = null                                ← 命令来自内置默认
models: 20 个 vLLMOmni 模型，models.image_name 全为 NULL                                  ← 全部指向那个浮动 tag
```

| 事实 | 值 |
| --- | --- |
| 现网容器实际在跑 | digest `6881675a…` = tag `arm64-a100-20261116-1525-2fae9a2e`（commit `2fae9a2e`） |
| `arm64-a100-latest` 现在指向 | digest `4cce9e73…` = tag `arm64-a100-20261117-0116-0b4425e6`（commit `0b4425e6`） |
| 本次新包 | `arm64-a100-20261119-0355-7ebda7f7`（别名 `arm64-a100-h3dur-1-20`），digest `sha256:14d0da31…`，commit `7ebda7f7a`，基座 `vllm/vllm-openai:v0.33.0@sha256:d825baf9…` |

> **两个必须先知道的结论**（第 1 条已于 2026-11-19 15:00 处理：`latest` 现在就是 main 构建 `154787d4`，滚动过程见 §8）
> 1. **`latest` 已经比现网跑的版本新**。现在随便删掉一个实例，它重建时拿到的是 `4cce9e73`，不是当前的 `6881675a`。所以"回滚"不能靠把浮 tag 当默认值，必须显式钉不可变 tag。
> 2. **部署档（`--deploy-config`）是烘进镜像的**，改一行参数也要重出包（`bench/prod-baseline-20261114/HANDOVER.md` §2 的老结论，仍成立）。

## 3. 两个源码级坑（`gpustack/routes/models.py`）

1. **模型级 `image_name` 会被后续编辑静默抹掉**（`models.py:748-756`）：对非 CUSTOM backend 的模型做 update 时，只要该模型当前有 `run_command` 或 `image_name`，patch 会无条件把这两个字段清成 `NULL`。
   → 用「模型级 image_name 覆盖」上线后，**任何一次后续编辑（哪怕只改副本数）都会让它回落到浮动 tag**。
2. **删除实例 = 重建换镜像**，`DELETE /v2/model-instances/{id}` 后就地按当前 model 配置重建（副本数不变）。想停服务用 `Stop`（= 缩到 0 副本）。

## 4. 备选路径 A：新增 backend 版本 + 两个模型切版本（只影响选它的模型）

> 本次**没有**走这条，实际走的是 §8 的「推进 latest + NFS 预载 + 删实例」。保留备查：只想给两个模型换镜像、不让其它模型受影响时用这条。

### 4.1 UI 操作

1. **Model Service → Inference Backends → vLLMOmni → Edit → Version Configs → Add Version**
   - Version：`1.0.0-h3dur-custom`（内置 backend 的自定义版本**必须**以 `-custom` 结尾）
   - Image Name：`crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-omni:arm64-a100-20261119-0355-7ebda7f7`
   - Framework：`cuda`
   - Environment / Entrypoint / Execution Command：**留空**（与 `1.0.0` 完全一致，命令走内置默认 `vllm serve {{model_path}} --omni …`）
   - Save
2. **Models → minimax-h3-fl2va → Edit → Backend Version = `1.0.0-h3dur-custom` → Save**，ref2va 同样操作。
3. **删掉两个模型名下的旧实例**（各 7 个，共 14 个）：Models → 点 `>` 展开实例 → Operations → `Delete`。删完 GPUStack 自动重建。
4. 验收（§5）。回滚见 §6。

### 4.2 API 等价命令（现网已验证可读，写操作请先在测试模型上演练）

```bash
K="gpustack_0e980400d98fbf7e_b7ce9d4d3bbaa0b6491857f418a99faa"   # org_key，management+inference
H="Authorization: Bearer $K"; U=http://127.0.0.1/v2

# 0) 先看现状
curl -s -H "$H" $U/inference-backends/backend_name/vLLMOmni | python3 -m json.tool
curl -s -H "$H" $U/models/151/instances | python3 -c 'import sys,json;[print(i["id"],i["name"],i["state"]) for i in json.load(sys.stdin)["items"]]'

# 1) 给 backend 追加新版本（PUT 整个 version_configs，把 1.0.0 原样带上）
curl -s -H "$H" $U/inference-backends/backend_name/vLLMOmni > /tmp/vllm-omni-backend.json
#    编辑 /tmp/vllm-omni-backend.json：在 version_configs 里加
#    "1.0.0-h3dur-custom": {"image_name":"…/vllm-omni:arm64-a100-20261119-0355-7ebda7f7",
#                           "run_command":null,"entrypoint":null,"custom_framework":"cuda","env":null}
curl -s -X PUT -H "$H" -H 'Content-Type: application/json' \
     $U/inference-backends/14 -d @/tmp/vllm-omni-backend.json

# 2) 两个模型切版本（ModelUpdate 只送这个字段即可，其余保持不动）
curl -s -X PUT -H "$H" -H 'Content-Type: application/json' \
     $U/models/149 -d '{"backend_version":"1.0.0-h3dur-custom"}'
curl -s -X PUT -H "$H" -H 'Content-Type: application/json' \
     $U/models/151 -d '{"backend_version":"1.0.0-h3dur-custom"}'

# 3) 删旧实例，触发重建（对 14 个 id 循环）
curl -s -H "$H" $U/models/151/instances | python3 -c 'import sys,json;[print(i["id"]) for i in json.load(sys.stdin)["items"]]' \
  | xargs -I{} curl -s -X DELETE -H "$H" $U/model-instances/{}
```

## 5. 验收（新镜像应满足）

新实例起来后（冷启动实测 6–9 分钟，含权重加载）：

```bash
# ① 实例自报的契约窗口必须是 [1.0, 20.0]
ssh gpuNN 'docker logs <实例名>-run-0 2>&1 | grep "MiniMax-H3 contract" | head -1'
#   → … 'output_duration_seconds': [1.0, 20.0] …

# ② 上界：21 s 必须 400，且文案写 [1, 20]
# ③ 下界：1 s 必须 200，产出 39 帧 = 1.625 s
# ④ 20 s 必须 200，产出 481 帧 = 20.042 s
# 请求体与脚本：bench/prod-baseline-20261114/requests/pb_run2.sh（args: URL JSON TAG LIMIT CONTAINER）
```

2026-11-19 对**本次新镜像**的验收结果（gpu42、pika ref2va 部署档、1344×768）：

| 项 | 结果 |
| --- | --- |
| 镜像 digest | `sha256:14d0da31…`（与 release 一致） |
| `seconds=21` | failed：`must be in [1, 20] seconds, got 21.0` ✅ |
| `seconds=1` | completed，39 帧 / 1.625 s，wall 100 s（含冷启动编译） |
| `seconds=20` | completed，481 帧 / 20.0417 s，wall 458 s，峰值 29.07 GiB/卡 |

原始记录：`bench/h3-duration-window-20261119/raw/acc_shipped_image.log`。

## 6. 回滚与停机窗口

- **回滚**：把两个模型的 Backend Version 切回去，再删实例。⚠️ 切回 `1.0.0` 会拿到浮动 `latest`（=`4cce9e73`，**不是**原来的 `6881675a`）。要回到"原来那版"，必须把 `1.0.0` 的 image 临时钉成 `…:arm64-a100-20261116-1525-2fae9a2e`，或另建一个指向它的版本再切。
  → **建议**：本次上线时顺手把 `1.0.0` 的浮动 tag 也钉成不可变 digest，止住这个漂移。
- **停机窗口**：7 个副本并行重建，从删实例到第一个副本 healthy 约 7–10 分钟；这段时间该模型无可用副本（`max_num_seqs: 1`，每副本一次一发）。两个模型可以先后做，避免同时全挂。
- **可用性更好但不叫灰度的做法**：先把 Replicas 临时加到 8，等第 8 个副本在旧镜像上起来后再逐台删旧的；代价是多占 4 张卡，且用户已明确不需要灰度，故未采用。

## 7. 其它路径（了解即可，本次不推荐）

- **只改模型级 `image_name`**（HANDOVER 曾推荐）：一行 `PUT /v2/models/149 {"image_name":"…"}` 即可，但会踩 §3 的坑 1，后续任何编辑都会静默失效。
- **全局换 `1.0.0` 的 image**：20 个 vLLMOmni 模型下次重建一起换代，TTS/音频类风险未知，不做。
- **`upgrade/cluster.md` 的流程**：那是升 GPUStack server/worker 自身版本，与本次模型镜像无关。

## 8. 本次实际执行路径（2026-11-19 15:00–15:50，已完成）

选择的是「推进浮动 tag」而不是「新增 backend 版本」，因此**前置条件**是：新镜像必须基于 main 出包（否则等于回退 main 上的 `0b4425e6c`）。

```bash
# 1) 改动合入主干并在 main 出包（dest_tag=arm64-a100-latest）
git -C vllm-omni push origin HEAD:main          # 0b4425e6c..df481023b，fast-forward
gh workflow run build-arm64.yml --repo reputationly/vllm-omni --ref main -f dest_tag=arm64-a100-latest
#   → arm64-a100-20261119-0635-df481023 / digest sha256:154787d4…  （20 min）

# 2) 落 NFS（在任一有镜像的机器上）
./image-dist/bake_image_to_nfs.sh
#   → /nfs-models/images/vllm-omni-arm64-a100-latest-154787d46865.tar (16 G) + .sha256 + .manifest

# 3) 预载到所有可能承载实例的 worker，并覆盖本地同名 tag
for h in gpu21 gpu22 … gpu40; do
  ssh $h bash /nfs-models/images/_h3dist/load_image_from_nfs.sh \
      /nfs-models/images/vllm-omni-arm64-a100-latest-154787d46865.tar 154787d46865
done   # 20/20 OK

# 4) 删实例触发重建（先 ref2va 后 fl2va）
curl -s -H "$H" $U/models/151/instances | python3 -c '…' | xargs -I{} curl -s -X DELETE -H "$H" $U/model-instances/{}
#   ref2va 7/7 RUNNING ~9 min；fl2va 7/7 RUNNING ~8 min
```

**最重要的一条**：`docker load`/本地 tag 覆盖不可省。滚动前 20 台 worker 的 `arm64-a100-latest` 全是旧包 `4cce9e73`，Docker 创建容器时直接用本地 tag——不覆盖就会"镜像看起来升级了、容器还是旧的"。

验收结果与完整时间线见 `SUMMARY.md` §6。

## 9. 本次之后的镜像坐标

| 用途 | 引用 |
| --- | --- |
| 现网在跑（= 浮动 `arm64-a100-latest`） | `arm64-a100-20261119-0635-df481023` / `sha256:154787d4…` |
| 上一版 latest（原 4cce9e73） | `arm64-a100-20261117-0116-0b4425e6` |
| 滚动前现网在跑的（6881675a） | `arm64-a100-20261116-1525-2fae9a2e` |
| 本次作废的中间包（基于 2fae9a2e，未上现网） | `arm64-a100-20261119-0355-7ebda7f7` |
