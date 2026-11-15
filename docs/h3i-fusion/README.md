# H3-Image 多参考自然融合（8 图）— 输入 / 提示词 / 生成条件 / 原图

根目录：`/nfs-output/prod-baseline-20261114/h3i-fusion/`

- 输入素材：`inputs/`（11 张，全部由 H3-Image 文生图生成，1024²，seed 23）
- 输出原图（**单张 2048×2048，未拼接**）：`outputs/base/<case>.png`、`outputs/auto_ladder/<case>.png`、`outputs/budget_2.1Mpx/<case>.png`
- 机读清单（提示词+参考图+条件）：`cases.json`
- 复现脚本：`cases_fusion.py`、`run_fusion.py`

## 生成条件（三个变体完全相同，只差一个预算开关）

| 项 | 值 |
| --- | --- |
| 接口 | `POST /v1/images/edits`（multipart：`prompt` + N × `image` + `size` + `seed` + `generator_device` + `response_format` + `output_format`） |
| 分辨率 | `size=2048x2048` |
| 种子 | `seed=42`，`generator_device=cpu`（保证可复现） |
| 步数 | 6 步（PDD 蒸馏，release 固定） |
| 硬件/并行 | 4×A100-40G，`--num-gpus 4 --tensor-parallel-size 4 --text-encoder-tp-size 4` |
| 镜像 | `arm64-a100-20261114-1645-76e41b0f` + 本文所述 fork 补丁（预算/参考-KV） |
| 变体 | `base` = 不开预算；`auto_ladder` = `reference_image_total_pixels="auto"`（≤8 张自动关闭）；`budget_2.1Mpx` = 固定 2.1 Mpx |

## 用例 A：A_cafe_morning_8ref（8 参考）

输入（顺序即 `<Picture i>`）：

| # | 文件 | 内容 |
| --- | --- | --- |
| 1 | `/nfs-output/prod-baseline-20261114/h3i-fusion/inputs/girl.png` | 街拍人像：二十多岁女性，站姿，阴天漫射光 |
| 2 | `/nfs-output/prod-baseline-20261114/h3i-fusion/inputs/jacket.png` | 橄榄绿工装外套（挂拍产品图） |
| 3 | `/nfs-output/prod-baseline-20261114/h3i-fusion/inputs/bicycle.png` | 复古黑色城市自行车 + 藤篮（侧视） |
| 4 | `/nfs-output/prod-baseline-20261114/h3i-fusion/inputs/cafe.png` | 深绿遮阳篷的小咖啡馆门面，鹅卵石路 |
| 5 | `/nfs-output/prod-baseline-20261114/h3i-fusion/inputs/dog.png` | 柯基，站于鹅卵石地面 |
| 6 | `/nfs-output/prod-baseline-20261114/h3i-fusion/inputs/mugcoffee.png` | 陶瓷拿铁杯（带拉花） |
| 7 | `/nfs-output/prod-baseline-20261114/h3i-fusion/inputs/street.png` | 狭长鹅卵石街道、砖立面、阴天清晨 |
| 8 | `/nfs-output/prod-baseline-20261114/h3i-fusion/inputs/shopsign.png` | 咖啡馆窗上的手绘招牌（深绿底奶油字） |

提示词（原文）：

> Picture 1 is a young woman and she is the subject. Picture 2 is the olive utility jacket she wears.
> Picture 3 is her vintage bicycle, parked at her hip. Picture 4 is the cafe facade she is standing in
> front of. Picture 5 is her corgi, standing at her feet. Picture 6 is the latte she holds. Picture 7 is
> the cobblestone pavement under them. Picture 8 is the painted sign board hanging on the cafe window.
> Produce ONE photorealistic street photograph of this single moment, eye level, 35 mm, overcast morning
> light: the woman in that jacket in front of that cafe, holding that latte, her corgi beside her, the
> bicycle leaning at her hip, cobblestones underfoot and the sign on the window behind her. Keep the
> jacket's colour and cut, the cup, the dog's markings and the bike's frame recognisable. Single frame,
> no collage, no grid, no multiple panels, no lettering overlay.

结果：**单张自然街拍**——人穿着参考外套、手拿参考杯子、柯基在脚边、自行车带藤篮靠身侧、身后是参考咖啡馆门面与招牌、脚下鹅卵石。

## 用例 B：B_room_portrait_8ref（8 参考）

| # | 文件 | 内容 |
| --- | --- | --- |
| 1 | `.../h3i-fusion/inputs/girl.png` | 人像主体 |
| 2 | `.../h3i-fusion/inputs/jacket.png` | 她穿的外套 |
| 3 | `/nfs-output/prod-baseline-20261114/h3i-cases/refs/interior.png` | 她要站的房间 |
| 4 | `.../h3i-fusion/inputs/lamp.png` | 落地灯（点亮） |
| 5 | `.../h3i-fusion/inputs/plant.png` | 陶盆龟背竹 |
| 6 | `.../h3i-fusion/inputs/rug.png` | 米色羊毛地毯 |
| 7 | `/nfs-output/prod-baseline-20261114/h3i-cases/refs/chair.png` | 橡木扶手椅 |
| 8 | `/nfs-output/prod-baseline-20261114/h3i-cases/refs/mountain.png` | 窗外雪山湖景 |

提示词（原文）：

> Picture 1 is a young woman and she is the subject. Picture 2 is the olive jacket she wears.
> Picture 3 is the room she stands in. Picture 4 is the mid-century floor lamp beside her, switched on.
> Picture 5 is the potted monstera in the corner. Picture 6 is the rug on the floor. Picture 7 is the oak
> armchair in the background. Picture 8 is the view outside the room's window.
> Produce ONE photorealistic interior portrait: the woman wearing that jacket, standing in that room on
> that rug, the lamp lit beside her, the monstera in the corner, the armchair behind her, and that view
> through the window. Late afternoon side light, 50 mm, consistent perspective. Single frame, no collage,
> no grid, no multiple panels, no text.

结果：**单张室内人像**——人站地毯上、外套一致、落地灯亮着、龟背竹在角、扶手椅在背景、窗外是雪山湖。

## 用例 C：C_street_ride_8ref（8 参考）

| # | 文件 | 内容 |
| --- | --- | --- |
| 1 | `.../h3i-fusion/inputs/girl.png` | 骑车人 |
| 2 | `.../h3i-fusion/inputs/jacket.png` | 她穿的外套 |
| 3 | `.../h3i-fusion/inputs/bicycle.png` | 她骑的自行车 |
| 4 | `.../h3i-fusion/inputs/street.png` | 她骑行的鹅卵石街道 |
| 5 | `.../h3i-fusion/inputs/cafe.png` | 她经过的咖啡馆 |
| 6 | `.../h3i-fusion/inputs/dog.png` | 陪跑的柯基 |
| 7 | `.../h3i-fusion/inputs/shopsign.png` | 经过的咖啡馆招牌 |
| 8 | `.../h3i-cases/refs/mountain.png` | 街道尽头的雪山 |

提示词（原文）：

> Picture 1 is a young woman, the rider. Picture 2 is the olive jacket she wears. Picture 3 is the
> bicycle she rides. Picture 4 is the cobblestone street she rides down. Picture 5 is the cafe she is
> passing. Picture 6 is her corgi running alongside on a lead. Picture 7 is the painted cafe sign she
> passes under. Picture 8 is the mountain visible at the end of the street.
> Produce ONE photorealistic motion photograph of her riding that bicycle down that street in that
> jacket, the corgi alongside, passing the cafe with its sign, the mountain at the far end of the street.
> Slight panning blur on the cobblestones only, subject sharp. Single frame, no collage, no grid, no
> multiple panels, no text.

## 结果与开销

| 变体 | A | B | C | 峰值显存（4 卡合计） | 与 base 的像素差 |
| --- | --- | --- | --- | --- | --- |
| `base`（不开预算） | 31.7 s | 31.5 s | 31.5 s | 134358 MiB（33.6 GiB/卡） | — |
| `auto_ladder`（≤8 张自动关闭） | 31.8 s | 31.7 s | 31.6 s | 134358 MiB | **mean\|Δ\|=0.00，PSNR=∞，SSIM=1.0000（逐像素相同）** |
| `budget_2.1Mpx`（强制 2.1 Mpx） | **12.3 s（−61%）** | **12.3 s（−61%）** | **12.1 s（−62%）** | 129158 MiB（31.5 GiB/卡） | 0.405 / 0.650 / 0.453（肉眼可见损失） |

阶梯开关在 9 张以上才生效，同一部署同一进程，按请求里的参考图数量自动判定：

| 用例 | 参考数 | base | auto_ladder | 说明 |
| --- | --- | --- | --- | --- |
| 8 图融合（A/B/C） | 8 | 31.5–31.7 s | 31.6–31.8 s | 走 0 预算分支，**逐像素相同** |
| 10ref_catalog | 10 | 39.0 s | **31.0 s（−21%）** | 走 8.4 Mpx 分支 |
| 14ref_catalog | 14 | 55.5 s | **23.3 s（−58%）** | 走 6 Mpx 分支 |
| 12/16 参考（4K 源图） | 12/16 | OOM | 35.5 / 37.2 s（8.4 Mpx） | 6 Mpx 分支同样可用 |

## 结论

1. 8 图融合在 release 默认配置下就能做出**自然的单张画面**（上面三个用例），耗时 31.5 s/张、显存 33.6 GiB/卡。
2. `reference_image_total_pixels="auto"` 对 ≤8 张**完全零影响**（逐像素相同），只有 9 张以上才自动降采样并拿到 −21%～−58% 的收益 —— 可以放心设为默认。
3. 想要 8 图也提速（−61%）只能显式设 2.1 Mpx，代价是参考细节损失（SSIM 0.40–0.65），适合吞吐优先的批量场景。
