# H3-Image 参考图优化：10 组真实编辑用例对照（NFS）

根目录：`/nfs-output/prod-baseline-20261114/h3i-cases/`

固定条件：`POST /v1/images/edits`，`size=2048x2048`，`seed=42`，`generator_device=cpu`；
4×A100-40G，`--num-gpus 4 --tensor-parallel-size 4 --text-encoder-tp-size 4`；同一提示词与同一批参考图，
**只改开关**。

| 变体目录 | 开关 |
| --- | --- |
| `base/` | 两个开关都关（release 默认） |
| `refkv_interval2/` | `--kv-offload-config '{"mode":"on","kv_refresh_interval":2}'` |
| `budget_2.1Mpx/` | `--additional-config '{"reference_image_total_pixels":2097152}'`（只跑了 2/3/4 参考图的 4 个用例） |

## 一、先看这 10 张对照（每条 = 一行，左基线/中参考-KV/右预算）

```
/nfs-output/prod-baseline-20261114/h3i-cases/compare/01_official_podcast_1ref__compare.png
/nfs-output/prod-baseline-20261114/h3i-cases/compare/02_mug_restyle_1ref__compare.png
/nfs-output/prod-baseline-20261114/h3i-cases/compare/03_poster_text_1ref__compare.png
/nfs-output/prod-baseline-20261114/h3i-cases/compare/04_interior_restyle_1ref__compare.png
/nfs-output/prod-baseline-20261114/h3i-cases/compare/05_portrait_oil_1ref__compare.png
/nfs-output/prod-baseline-20261114/h3i-cases/compare/06_mountain_season_1ref__compare.png
/nfs-output/prod-baseline-20261114/h3i-cases/compare/07_style_transfer_2ref__compare.png
/nfs-output/prod-baseline-20261114/h3i-cases/compare/08_product_pair_2ref__compare.png
/nfs-output/prod-baseline-20261114/h3i-cases/compare/09_person_room_view_3ref__compare.png
/nfs-output/prod-baseline-20261114/h3i-cases/compare/10_contact_sheet_4ref__compare.png
```

总览（左=基线，右=参考-KV，10 个用例）：

```
/nfs-output/prod-baseline-20261114/h3i-cases/compare/overview_base_left_vs_refkv_right.png
```

原始 2048² PNG（每张单独）：

```
/nfs-output/prod-baseline-20261114/h3i-cases/base/<case>.png
/nfs-output/prod-baseline-20261114/h3i-cases/refkv_interval2/<case>.png
/nfs-output/prod-baseline-20261114/h3i-cases/budget_2.1Mpx/<case>.png
```

指标表：`/nfs-output/prod-baseline-20261114/h3i-cases/compare/metrics.csv`

## 二、指标（`metrics.csv` 同内容）

| 用例 | 参考数 | 基线 (s) | 参考-KV (s) | Δ | 参考-KV PSNR/SSIM | 预算 Δ | 预算 PSNR/SSIM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01_official_podcast_1ref | 1 | 9.18 | 9.09 | −1.0% | 29.4 dB / 0.957 | — | — |
| 02_mug_restyle_1ref | 1 | 8.18 | 7.99 | −2.3% | 22.2 dB / 0.951 | — | — |
| 03_poster_text_1ref | 1 | 8.18 | 7.80 | −4.6% | 26.9 dB / 0.977 | — | — |
| 04_interior_restyle_1ref | 1 | 8.31 | 7.86 | −5.4% | 19.1 dB / **0.765** | — | — |
| 05_portrait_oil_1ref | 1 | 8.14 | 7.62 | −6.4% | 33.8 dB / 0.971 | — | — |
| 06_mountain_season_1ref | 1 | 8.19 | 7.84 | −4.3% | 26.7 dB / 0.949 | — | — |
| 07_style_transfer_2ref | 2 | 11.38 | 11.17 | −1.8% | 27.0 dB / 0.949 | +3.3%（未触发缩放，像素完全相同） | ∞ / 1.000 |
| 08_product_pair_2ref | 2 | 11.33 | 10.62 | −6.3% | 27.2 dB / 0.937 | +0.6%（同上报未触发） | ∞ / 1.000 |
| 09_person_room_view_3ref | 3 | 14.41 | 14.07 | −2.4% | 24.7 dB / 0.888 | **−9.9%** | 19.1 dB / 0.703 |
| 10_contact_sheet_4ref | 4 | 17.58 | 15.45 | **−12.1%** | 16.5 dB / 0.861 | **−17.7%** | 14.3 dB / 0.806 |

读法：

- **参考-KV**：SSIM 0.86–0.98（除 04 的 0.765，那是唯一肉眼可辨的差异——榻榻米编织与家具细节变了，构图不变），
  耗时 −1%~−12%，参考越多省得越多（4 参考 −12%）。
- **总像素预算**：2 参考不触发（2.1 M ≈ 上限，像素完全相同，仅噪声 ±3%）；3 参考 −9.9%、4 参考 −17.7%，
  代价是 SSIM 0.70–0.81（参考被缩小 ~30%，细节确实变了）。

## 三、素材

- 官方：`/nfs-models/wuhanjisuan894/models/MiniMax-H3-Image/examples/{reference1.png,prompt.edit.txt}`（用例 01 原封不动用官方提示词）
- 本次生成（H3-Image 文生图 1024²，seed 7）：`/nfs-output/prod-baseline-20261114/h3i-cases/refs/{mug,poster,interior,fisherman,mountain,robot}.png`

## 四、复现

```bash
# 清单与跑批脚本
/nfs-output/prod-baseline-20261114/h3i-cases/cases.py
/nfs-output/prod-baseline-20261114/h3i-cases/run_cases.py
/nfs-output/prod-baseline-20261114/h3i-cases/make_montages.py

# 对某个服务跑全部 10 个用例（在能访问该服务、且挂了 /nfs-output 的机器上）
python3 /nfs-output/prod-baseline-20261114/h3i-cases/run_cases.py \
  --url http://<host>:<port> --variant <变体名>
# 只跑部分
python3 .../run_cases.py --url ... --variant budget_2.1Mpx \
  --only 07_style_transfer_2ref,08_product_pair_2ref,09_person_room_view_3ref,10_contact_sheet_4ref
```
