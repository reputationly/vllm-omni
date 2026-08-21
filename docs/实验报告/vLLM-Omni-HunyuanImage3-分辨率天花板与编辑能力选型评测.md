# vLLM-Omni · HunyuanImage-3.0 分辨率天花板与编辑能力选型评测

> 日期：2026-08-20 | 测试人：reputationly + Claude
> 被测服务（全部现网 GPUStack 实例）：
> - HY3：`dev-gpustack-a100-0022` `10.0.0.103:40039`，`HunyuanImage-3.0-Instruct-Distil-NF4-v2`，4×A100-40G，vLLMOmni 1.0.0
> - Qwen-Image-Edit：`dev-gpustack-a100-0020` `10.0.0.94:40046`，`Qwen-Image-Edit-2511`，1×A100-40G，LightX2V 1.0.0
> - Qwen-Image（素材生成）：`dev-gpustack-a100-0013` `10.0.0.51:40021`，`Qwen-Image-2512`，LightX2V 1.0.0
>
> 产物：`/nfs-models/wuhanjisuan894/{hy3_reso_repro,hy3_steps_prompt,hy3_bottask,qie_vs_hy3,qie_strength_sweep,bench_hy3_vs_qie}_20260820/`
> 姊妹文档：`vLLM-Omni-HunyuanImage3-INT8-vs-NF4-画质与遵从度实测.md`（本文订正其 §4.2 与 §9 各一处）

---

## 0. 结论先行

起因是「雪狐换季」一例长期被认为「修完变糊」，先后被归因于量化、分辨率、蒸馏步数。**全部不成立。**

| 被怀疑的因素 | 结论 | 依据 |
|---|---|---|
| 量化（NF4/INT8/W8A8） | **无关** | 四个权重版本输出**逐像素同尺寸**，画质差异只在微观纹理 |
| 输出分辨率被降采样 | **无关** | 输入 1024²、输出 1024²，一个像素没掉 |
| 蒸馏步数不够 | **几乎无关** | 8→50 步细节只涨 26%，且 SSIM 反降 |
| `bot_task` 未启用 | **有害** | recaption/think 比不加更差 |
| 编辑指令写法 | **有效但有限** | 主体细节 7.3% → 18.1% 封顶 |
| **模型选型** | **决定性** | 同任务 Qwen-Image-Edit 达 92%，HY3 26.4% |

**两条硬事实：**

1. **HY3 出图恒 ≈1 Mpx，请求参数控制不了**，官方已确认这是训练上限（§2）。
2. **HY3 的 i2i 没有任何保真机制**——`mask` 与 `strength` 均被**静默吞掉**（§4）。

**选型建议不是「换掉 HY3」，而是按任务类型分流（§6.4）**：保主体改环境走 Qwen-Image-Edit（差 3.5×），全局改色/光照/加减物体/风格化 HY3 够用且快 1.5×，文生图 HY3 仍是强项。

---

## 1. 背景

`INT8-vs-NF4` 报告 §4.1 已推翻过一次「i2i 固有丢 80–90% 高频」，给出的解释是「换季指令自己把背景改成了大散景」。但该结论只用 LapVar 佐证，且当时认定 §4.2 的 1 Mpx 天花板是「管线强制归桶」，未追到根因，也未验证请求层能否干预。本轮补齐这两块，并把范围扩到跨模型选型。

---

## 2. 分辨率天花板：请求控制不了，且是训练上限

### 2.1 代码事实

`ResolutionGroup(base_size=1024)`（`hunyuan_image3_transformer.py:535`）只产出 **37 个桶，面积全部落在 0.786–1.049 Mpx**。`build_image_info()` 把任何请求尺寸过一遍 `get_target_size()`（`:1525` 调用，`:634` 定义）归到**比例**最近的桶，**面积不参与决策**。

`size=auto` 不是「让 AR 决定」，而是 `width, height = pil_images[0].size`（`api_server.py:2372`）——直接取输入图尺寸，再归桶。

### 2.2 现网实测（0022，NF4-v2，steps=8 / cfg=1.0 / seed=42）

| case | 输入 | 请求 `size` | 实际输出 | 耗时 |
|---|---|---|---|---|
| A | 1024×1024 | `auto` | 1024×1024 | 101s（冷启动） |
| B | 1024×1024 | **`2048x2048`** | **1024×1024** | 14s |
| C | 1024×1024 | `1280x720` | 1280×720 | 43s |
| D | **2048×2048** | `auto` | **1024×1024** | 58s |

**B 是关键：明确要 2048×2048，服务照常 `200` 返回 1024×1024，无任何警告——静默归桶，调用方拿不到「你要的尺寸没给你」的信号。**

`size` 只决定**比例**，不决定**像素量**。

### 2.3 官方确认：1K 是训练上限

[Issue #6](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/issues/6) 中报告者同样发现词表里有 `img_size_1024/2048/4096` 等 token（实际全套为 256/512/768/1024/1536/2048/3072/4096/6144/8192，`config.json` 的 `image_base_size` 为 `None` → 取默认 1024）。腾讯 Jarvis73 2025-09-28 回复：

> Currently, HunyuanImage3.0 is trained to support image generation up to 1K resolution, so it is not yet capable of producing higher-resolution (up to 2K) images.

技术报告与之对应：预训练末段图像**短边至少 1024**。

**改 `image_base_size` 这条路已被社区证伪**——同一 issue 里直接改 `ResolutionGroup(base_size=...)` 的结果是「尺寸对了，`the content is entirely nonsensical`」。**不要再花成本尝试。**

至今无解：[Issue #91](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/issues)「is it possible to generate 2k resolution?」2026-03-16 开，零回复；#30「4K直出」同样无下文。

### 2.4 一处与官方契约的偏差

`HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS`（`hunyuan_image3_transformer.py:523`）有 8 项，官方 `image_processor.py:147` 只有 4 项（`1024x768 / 1280x720 / 768x1024 / 720x1280`）。**多出的 `512x512 / 640x640 / 768x768 / 896x896` 是 vllm-omni 自加的**，代码注释却声称「matching the official model」。

后果：这四项与 1024² 同比例，被 append 进 `Resolution(1024,1024).extra_res`，`match()` 按**面积最近**挑选，于是请求小正方形会真的得到小图：

| 请求 | 实际输出 | Mpx |
|---|---|---|
| 512x512 | **512x512** | 0.26 |
| 768x768 | **768x768** | 0.59 |
| 1024x1024 | 1024x1024 | 1.05 |
| 2048x2048 / 4096x4096 | 1024x1024 | 1.05 |
| 1920x1080 / 3840x2160 | 1280x720 | 0.92 |

官方行为下请求 512x512 应得到 1024×1024。**传小尺寸有反效果**，ratio token 映射不受影响（这四项不占新索引，表长仍 37）。

---

## 3. 输入尺寸与显存：订正旧报告 §9

`INT8-vs-NF4` §9 原文：

> INT8 峰值 37429 MiB 是 1024² / ≤3 参考图条件下的。**更大底图**或第 4 张参考图未压测，8.6% 余量下大概率 OOM。

**「更大底图」这半句不成立。** 参考图在进 VAE 之前就被归一化：`_build_cond_joint_image`（`pipeline_hunyuan_image3.py:248-275`）先 index-based 归桶再 `_resize_and_crop_center`，VAE token 恒定：

| 输入 | VAE 实际输入 | VAE token |
|---|---|---|
| 512×512 | 1024×1024 | 4096 |
| 2048×2048 | 1024×1024 | 4096 |
| 8192×8192 | 1024×1024 | 4096 |
| 3840×2160 | 1280×720 | 3600 |

ViT 分支走 Siglip2 naflex，`config.vit_processor.max_num_patches = 1024` 封顶，同样与输入尺寸无关。**每张参考图恒定 ≤ 4096 + 1024 = 5120 token。**

§2.2 的 case D 实测坐实：2048×2048 输入 58s 正常跑完，输出 1024×1024，无 OOM。

**「第 4 张参考图」那半句成立，但瓶颈可能不是显存**：生成图 4096 token + 3 张参考图 15360 ≈ 19456，加 CoT 文本已逼近 `max_position_embeddings = 22800`；4 张则为 4096 + 20480 = 24576，**越过位置预算**。此项为按 config 推算，未实测。

> 注：非标准比例输入会被 `_resize_and_crop_center` 中心裁边（如 928×1664 → 720×1280）。喂图前自行裁到目标比例比交给管线更可控。

---

## 4. HY3 的 i2i 没有保真机制（三重）

1. **`mask` 被静默吞掉。** `/v1/images/edits` 收 `mask_image` 并写入 `multi_modal_data`，但 HY3 的 `pre_process_func` 只读 `multi_modal_data["image"]`（`pipeline_hunyuan_image3.py:313-317`），从不读 mask。**做不了 inpainting**，无法「只改背景、主体不动」。
2. **`strength` 被静默吞掉。** `ar2diffusion` 只转发 `seed / num_inference_steps / guidance_scale / negative_prompt` 四个（`stage_input_processors/hunyuan_image3.py:220`），`vllm_omni/diffusion/models/hunyuan_image3/` 全目录搜不到 `strength`。**没有保留度旋钮。**
3. **架构上无原像素通路。** 自回归整图重生成，不是 diffusion 的部分去噪，SD/Flux 的低 denoise 保细节手法不适用。

叠加信息带宽：1024² 全图仅 **4096 latent token**，每个扛 16×16 像素。主体 ROI 约占 10% 面积，即全部毛发纹理由 **约 400 个 token** 承载。

> 另一处静默失效：`t2i` 不传 `size` 时恒出 1024×1024 正方形。`prepare_encode` 读 `sampling.height or 1024`（`pipeline_hunyuan_image3.py:1922`），而 AR 预测的比例只写进 bridge 的 prompt dict，无参考图时 `pre_process_func` 的回填分支（`:318-329`）不执行，无人消费。**此项为代码推断，未实测。**

---

## 5. HY3 内部调参的边界

方法：输入 `/nfs-models/wuhanjisuan894/hy3_t2i_out.png`（1024²，LapVar 228.72），指令固定为「换季」族，seed=42 / cfg=1.0 / size=auto。

指标除沿用 LapVar、SSIM 外，新增 **ROI 口径**：取**源图局部锐度 top10% 的像素**作为 ROI（主体毛发必然落在其中，覆盖 9.7% 像素），在同一 ROI 上比较所有输出。源图 `LapVar_ROI = 1590.78`。此口径避免了「中心 50%」这类拍脑袋的框，也降低了 §4.1 提过的「LapVar 被内容差异污染」风险。

### 5.1 步数扫描（原指令）

| 步数 | Lap_full | Lap_中心 | Lap_背景 | SSIM | ROI 占源图 | 耗时 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24.08 | 60.15 | 12.11 | 0.6450 | 7.3% | 12s |
| 16 | 26.86 | 67.62 | 13.33 | 0.6108 | 8.3% | 17s |
| 30 | 28.77 | 71.99 | 14.43 | 0.5788 | 8.7% | 25s |
| 50 | 30.38 | 76.25 | 15.16 | 0.5563 | 9.2% | 36s |

**8→50 步锐度只涨 26%，耗时涨 3 倍，且 SSIM 一路下降**——那点提升里有一部分是「改得更多」换来的，不是细节恢复。跑的是 8 步蒸馏版，8 步就是其训练工作点。**此路封闭**，`INT8-vs-NF4` §9 的「步数实验未做」至此填平。

### 5.2 指令扫描（8 步）

- `P1` = 原指令：`Change the season to summer: green forest, warm golden light, no snow`
- `P2` = P1 + `Keep the fox exactly as it is: preserve every strand of fur, the whiskers, the eyes and all fine facial details with maximum sharpness. Do not soften or blur the animal.`
- `P3` = P1 + `Keep the entire background in sharp focus with crisp detailed foliage and individual leaves visible. No bokeh, no depth-of-field blur...`

| 指令 | Lap_中心 | Lap_背景 | SSIM | ROI 占源图 |
|---|---:|---:|---:|---:|
| P1 | 60.15 | 12.11 | 0.6450 | 7.3% |
| **P2** | **101.78 (+69%)** | 12.21 | **0.6673 ↑** | 13.2% |
| P3 | 100.50 | **24.14 (+99%)** | 0.5359 | 12.7% |
| P3 @30步 | 143.84 | 29.19 | 0.4603 | **18.1%** |

**P2 是白捡的**：主体锐度 +69%，背景不动，**SSIM 反而升高**，步数不变（12s）。锐度与保真度同时上升，不可能靠「改更多内容」刷出来——这是本轮最硬的单点结论。

**P3 独立验证了 §4.1 的散景判断**：源图背景 Lap 53.49 → P1 打到 12.11（**掉 77%**），P3 拉回 24.14。「换季」这条指令确实自己把背景改成了大散景。

### 5.3 `bot_task`：有害，且实现有问题

| bot_task | ROI 占源图 |
|---|---:|
| 不传 | 13.2% |
| recaption | 8.3% |
| think | 8.3% |

两者不仅都劣于不传，且**输出图 md5 逐字节相同**（`d1bd6ce17587ce75`），耗时也与不传一致（12s）——recaption 与 think 在实现里走了同一条路径，未区分开。方向上已否定，未深追。

### 5.4 边界

HY3 内部能榨出的全部是 **7.3% → 18.1%（2.5×）**，离源图始终差 5 倍以上。「绝对不如原图锐」解决不了：源图是超锐 AI 图（本身即 HY3 t2i 产物），模型有自己的目标锐度。

---

## 6. 与 Qwen-Image-Edit-2511 的横向评测

### 6.1 方法

素材由**第三方模型** `Qwen-Image-2512` 生成（1024²，seed 42），避免用任一被测方自己的产物做输入。8 个场景覆盖背景替换、环境替换、文字编辑、颜色属性、光照天气、物体移除、动漫→写实、写实→油画。

两侧使用**完全相同的指令**，各自现网默认配置：HY3 `steps=8 / cfg=1.0 / size=auto`；Qwen-IE `size=1024x1024`、**不传** `i2i_denoise_strength`（理由见 §6.3）。seed 均为 42。

### 6.2 结果

| 场景 | 编辑类型 | HY3 细节 | QwenIE 细节 | HY3 SSIM | QwenIE SSIM | 赢家 |
|---|---|---:|---:|---:|---:|---|
| animal | 换季·保主体 | 26.4% | **92.0%** | 0.5055 | 0.4169 | **QwenIE 3.5×** |
| portrait | 换背景·保人脸 | 54.0% | **82.1%** | 0.5947 | **0.7361** | **QwenIE** |
| anime | 动漫→写实 | 11.4% | **34.1%** | 0.5503 | 0.5192 | QwenIE |
| signboard | 改文字 | 90.7% | **99.6%** | 0.7556 | **0.8052** | QwenIE |
| fabric | 改颜色 | **104.0%** | 56.5% | 0.5915 | 0.5755 | **HY3 1.8×** |
| architecture | 改天气光照 | **89.3%** | 68.7% | 0.7188 | **0.7849** | HY3 |
| stilllife | 移除物体 | **87.6%** | 77.0% | 0.8888 | **0.9027** | HY3 |
| landscape | 写实→油画 | **112.9%** | 101.5% | 0.4904 | 0.3483 | HY3 |
| **均值** | | **72.1%** | **76.4%** | | | **1.1×，4:4** |

耗时：HY3 稳定 11–12s，Qwen-IE 17–18s，**HY3 快 1.5×**。

**任务完成度校验**（排除「高分来自没干活」）：

| 场景 | 判据 | 源图 | HY3 | QwenIE |
|---|---|---|---|---|
| animal | greenness↑ / 雪↓ | +0.88 / 19.9% | +10.86 / 0.1% | +22.00 / 1.0% |
| fabric | greenness↑ | +3.73 | +15.49 | +21.69 |
| architecture | R−B↑ | −20.76 | +63.36 | +67.63 |
| stilllife | 蓝像素↓ | 2.60% | 0.00% | 0.00% |
| anime | 饱和度↓ | 50.39 | 47.71 | **57.45 ↑** |

除 anime 外**两侧均确实执行了编辑**，4:4 是真实能力分布。

### 6.3 规律

**Qwen-IE 赢的四个全是「主体必须原样保留、周围大改」**（换季保狐狸、换背景保人脸、改招牌其余不动、转写实保姿势发型服装）；**HY3 赢的四个全是「全局属性变换」**（整体改色、改天气光照、移除物体、转油画）——后者不存在必须逐像素保住的主体。

与架构一致：HY3 整图重生成，全局变换时重画一遍反而自然，细节按自身目标锐度重建；一旦要「保这块改那块」，因无局部保留机制（§4）主体连带被重画，即崩（animal 26.4%）。Qwen-IE 是 diffusion editing，参考图条件强，天生适合局部改动。

**雪狐案例并非 HY3 整体不行，而是正好撞在其最不擅长的任务类型上。**

> **附：Qwen-IE 的 `i2i_denoise_strength` 有死区。** ≤0.9 时低频被原图 latent 锁住、改不动内容，却已因 VAE round-trip + 不完整去噪损失 30%+ 高频（0.85/0.90 最差，锐度仅 54–58%）；0.95 起阶跃式生效。**不传（默认）最好**（锐度 95.2%），0.4–0.9 是纯浪费。该阶跃而非渐变的形态，疑因 Qwen-Image-Edit 本是指令编辑模型（参考图走 condition 通道），LightX2V 外挂的 img2img 式 strength 与之不完全兼容——**此解释未经源码验证**。

### 6.4 选型建议

**按任务类型分流，而非整体替换：**

| 任务类型 | 推荐 | 理由 |
|---|---|---|
| 保主体、改环境/背景 | **Qwen-Image-Edit** | HY3 死穴，差 3.5× |
| 全局改色 / 改光照 / 加减物体 / 风格化 | **HY3** | 细节不输且快 1.5× |
| 文生图 | **HY3** | 强项，1024² 内质量优 |
| 需要 >1 Mpx 输出 | **均不可** | 外接超分（SeedVR2 已在集群跑通） |

**落地动作：把 §5.2 的 `P2` 保护性描述固化为 new-api 编辑提示词模板的后缀**——免费 +69% 主体锐度、保真度同时上升、零延迟增加。`P3` 的禁 bokeh 句按需开关，不宜做全局默认（会让画面偏离原图更多）。

---

## 7. 复现与产物

```
/nfs-models/wuhanjisuan894/
├── hy3_t2i_out.png                     雪狐输入原图 1024²（= INPUT_snowfox.png，md5 f443962d…）
├── hy3_reso_repro_20260820/            §2.2 四个 case + 2048² 测试输入
├── hy3_steps_prompt_20260820/          §5.1/5.2 步数×指令 7 张
├── hy3_bottask_20260820/               §5.3 bot_task 3 张
├── qie_vs_hy3_20260820/                §6 雪狐单图对比 5 张
├── qie_strength_sweep_20260820/        §6.3 附 strength 边界 4 张
└── bench_hy3_vs_qie_20260820/          §6.2 八场景
    ├── src/   8 张中立素材（Qwen-Image-2512）
    ├── hy3/   8 张
    ├── qie/   8 张
    └── run.log  8 条指令全文 + http 码 + 耗时
```

评估脚本为一次性产物，未入库；核心口径见 §5 开头（ROI = 源图局部锐度 top10% 像素，SSIM 为 11×11 均匀窗实现）。

---

## 8. 未做 / 存疑

- **单 seed、每场景单张。** §6 的 4:4 要作为正式选型依据，建议每场景补 2–3 个 seed 与不同题材。
- **三项无法自动判定，必须人眼看图**：`signboard` 文字是否拼对（SILVER LEAF TEA）、`portrait` 鱼市背景是否合理、`landscape` 是否像油画。这三项在选型中权重不低。
- **`anime` 转写实存疑**：Qwen-IE 饱和度不降反升（50.39 → 57.45），可能未完成风格转换，其 34.1% 的细节优势未必有效。
- **细节保持指标不适用于风格转换类任务**：油画本就该少高频，`landscape` 的 112.9% 未必是优点。
- **§4 中 t2i 恒 1024² 一项为代码推断，未实测。**
- **第 4 张参考图越过位置预算一项为按 config 推算，未实测。**
- **Qwen-IE 的 `mask` 参数未测**，「轻微局部编辑」这一需求的可行性尚无结论。
- **未测 HY3 的 CFG。** 现网 `guidance_scale=1.0`（无 CFG），提高会使 batch 翻倍、显存翻倍，40G 卡上有 OOM 风险，未在现网实例上尝试。
