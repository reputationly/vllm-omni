# MiniMax-H3 准入门槛用例集

> 用途：**每次改权重（新 LoRA / 量化 / 剪枝）或改推理优化后，必须跑完本用例集并全过，才允许进入下一步评估。**
> 用例文件在 `docs/h3-gate-cases/`，机器可读；本文件是规范与判读口径。

版本：v8（2026-10-26）。基线：`MiniMax-H3-FL2VA-Turbo4-768p-v1.1-BF16`（4 步烘焙，scale 1.0）。

---

## 0. 执行手册（给执行人的最短路径）

本文档前面是规范，后面是实测记录与基线。**只想知道怎么跑，读这一节就够。**

### 0.1 三条命令跑起来

**A. 跑现网（MaaS 网关，最省事）**

```bash
export KEY=<你的 sk-key>
python3 tools/minimax_h3_turbo/run_maas_case.py \
  --base-url https://maas.ovaijisuan.com --api-key "$KEY" \
  --request docs/h3-gate-cases/s1_quant_gradient_sky.json \
  --tag my_s1 --output-root /nfs-output/h3_gate/runs/<日期>-<批次名>
```

**B. 跑自建实例（自己起的 vLLM-Omni）**

```bash
python3 tools/minimax_h3_turbo/run_vllm_eval_case.py \
  --endpoint http://<ip>:<port> \
  --request docs/h3-gate-cases/s1_quant_gradient_sky.json \
  --tag my_s1 --output-root /nfs-output/h3_gate/runs/<批次> \
  --container <容器名>
```

**C. 批量跑一整批**（自动跳过已完成的）

```bash
# 现网（需要 export KEY=sk-...）
tools/minimax_h3_turbo/run_suite.sh maas /nfs-output/h3_gate/runs/<批次>

# 自建实例
tools/minimax_h3_turbo/run_suite.sh local http://10.0.0.126:40700 /nfs-output/h3_gate/runs/<批次> <容器名>
```

脚本会跳过已完成的用例，中断后重跑同一条命令即可续跑；ref2va 四条默认跳过，
需要时加 `--with-ref2va` 并指向 ref2va 端点。

### 0.2 必须先做的三件事

| # | 事 | 不做会怎样 |
| --- | --- | --- |
| 1 | **确认素材就位**：`/nfs-output/h3_gate/assets/` 下 8 个文件（见 §2 素材要求） | g2/g3/g4/g5/g6/g9/g11/s15 直接失败 |
| 2 | **确认采样契约**：8 步 / `flow_shift=12` / `audio_flow_shift=3` / `1344x756` | 自建引擎直接报错；网关静默按默认跑，结果不可比 |
| 3 | **确认任务类型写法**：网关用 `i2v/l2va/flf2v`+`metadata.task_type`；引擎用 `fl2va`+`image_path`/`last_frame_path` | 关键帧静默退化成 t2va（网关）或直接 400（引擎） |

### 0.3 判读怎么做

1. **先看硬门槛**（§5.1）：崩坏、黑屏、音轨缺失、文字全错 → 直接不通过，不用往下评。
2. **再看质量分**（§5.3，5 分制，幻影/结构/文字/音频/口型）；盲评，一次看一组，不对比名字。
3. **最后对性能**（§5.2）：与 §12 基线比，5 s 参考 105 s、15 s 参考 286 s（现网 8 步 INT8）；
   明显变慢要说明原因。
4. **记录**：按 §5.4 模板填，一条用例一行，写清「用例 / 种子 / 墙钟 / 峰值显存 / 五个分数 / 一句话观察」。

### 0.4 常见误判（都是我自己踩过的）

- 面板上的 300–600 秒不等于算得慢，是**排队**（§8.8）；
- 网关**不透传 seed**，同一 seed 跑两次结果也不同，别拿哈希比（§8.9）；
- 同一 `metadata.src_ref_images` 字段在 ref2va 上生效、在 fl2va 上被**静默丢弃**（§8.6）；
- 文字性别**压过**参考图性别，图是男的、提示词写 woman 就出女的（§8.10）；
- 音频是男声参考、生成是女声，可能只是提示词性别写错，不代表音频参考失效（§8.10）。

### 0.5 用例清单在哪

- 29 条用例：`docs/h3-gate-cases/*.json`，索引 `manifest.json`
- 每条英文提示词逐字见 **§13 附录**
- 分组与考点见 **§2**

## 1. 任务覆盖

H3 是**音视频联合生成**模型，不是图像模型。因此"文生图"一条**不适用**：

| 需求 | H3 支持 | 用例 | 说明 |
| --- | --- | --- | --- |
| 文生图 | ❌ | — | H3 无图像输出；图像需求走 `image_tasks` 那条线，不要塞进本用例集 |
| 文生视频（T2VA） | ✅ | `g1` `g7` `g8` `g10` | 含音频、对白、配乐 |
| 首帧生视频（I2VA） | ✅ | `g2` | `image_path` 单图 |
| 尾帧生视频（L2VA） | ✅ | `g3` | `last_frame_path` |
| 首尾帧生视频（FL2VA） | ✅ | `g4` `g9` | `image_path` 逗号两图（首、尾） |
| 参考图生视频（Ref2VA） | ✅ | `g5` | 人物/物体一致性 |
| 参考视频生视频（Ref2VA） | ✅ | `g6` `g11` | 延续或编辑既有片段 |

分区限制（`pipeline_minimax_h3.py:1751`）：`fl2va` 分区只能服务 `{t2va, fl2va}`；
`ref2va` 分区只能服务 `{ref2va}`。所以 `g1–g4`、`g7–g10` 需要 fl2va 实例；`g5`、`g6`、`g11` 需要 ref2va 实例。

---

## 2. 用例清单（29 条）

分四组：**基础 11 条**（任务类型 × 提示词类型，守住通用能力）、**技术分解 5 条**（压 Singularity 的四层优化）、**能力声明 9 条**（逐条对应官方宣称）、**风格对照 4 条**（同一画面换风格）。

所有用例文件在 `docs/h3-gate-cases/`，索引在 `manifest.json`。

### 2.1 基础（g1–g11，11 条）

| ID | 任务 | 时长 | seed | 输入 | 主要考点 |
| --- | --- | ---: | --- | --- | --- |
| `g1_t2va_crowd_motion` | t2va | 5 s | 42,43 | 文本 | 人群/暗光/运动 + 中文吆喝口型 + 招牌文字 |
| `g2_i2va_first_frame` | i2v | 5 s | 42 | 图 | 首帧严格对齐、推镜下身份稳定、画外问答 |
| `g3_l2va_last_frame` | l2va | 5 s | 42 | 尾帧 | 两镜倒推、精确收束到尾帧 |
| `g4_fl2va_first_last` | flf2v | 5 s | 42 | 图 | 两端精确、中间连续、并行跟拍 |
| `g5_ref2va_image` | ref2va | 5 s | 42 | 图 | 身份保持 + 新生成画面文字 + 台词 |
| `g6_ref2va_video` | ref2va | 5 s | 42 | 视频 | 运镜/色调继承、无缝接续 |
| `g7_t2va_15s_text_long` | t2va | 15 s | 42,43 | 文本 | 长片不漂移 + 屏幕文字 + 三镜 + 广播 |
| `g8_t2va_15s_dialogue_music` | t2va | 15 s | 42,43 | 文本 | 双人轮流对话口型 + 三层声音 + 黑板文字 |
| `g9_fl2va_15s_keyframes` | flf2v | 15 s | 42 | 图 | 15 秒两端锚定 + 三镜 + 一句台词 |
| `g10_t2va_15s_action_vfx` | t2va | 15 s | 42,43 | 文本 | 高速运动形变 + 慢动作细节 + 复杂音效同步 |
| `g11_ref2va_15s_video_continue` | ref2va | 15 s | 42 | 视频 | 长时程继承运镜 + 跨房间连续性 + 台词 |

### 2.2 技术分解（s1–s5，5 条）

| ID | 任务 | 时长 | seed | 输入 | 主要考点 |
| --- | --- | ---: | --- | --- | --- |
| `s1_quant_gradient_sky` | t2va | 5 s | 42,43 | 文本 | INT8 量化色带 / AdaLN 块间色差（大面积平滑渐变） |
| `s2_prune_face_texture` | t2va | 5 s | 42 | 文本 | 剪枝 r8 高频细节（毛孔、胡茬、织物织纹） |
| `s3_vdn_wide_pan_distant` | t2va | 15 s | 42,43 | 文本 | VDN 窗口注意力长程（15s 慢摇远景小目标不丢） |
| `s4_quant_dark_small_text` | t2va | 5 s | 42 | 文本 | INT8 暗部噪点 + 小字号文字可读 |
| `s5_quant_thin_lines_backlit` | t2va | 5 s | 42 | 文本 | INT8 高频边缘（细线不断裂、不振铃） |

### 2.3 能力声明（s6–s13、s15，9 条）

| ID | 任务 | 时长 | seed | 输入 | 对应官方声明 |
| --- | --- | ---: | --- | --- | --- |
| `s6_hdr_motion_blur` | t2va | 5 s | 42,43 | 文本 | HDR 画质与模糊抑制 |
| `s7_distant_face_repair` | t2va | 5 s | 42 | 文本 | 远景人脸修复 |
| `s8_natural_material` | t2va | 5 s | 42 | 文本 | 干净自然的视觉美感（反塑料感） |
| `s9_action_impact` | t2va | 15 s | 42,43 | 文本 | 增强动态表现力（动作轨迹与打击感） |
| `s10_vfx_fantasy` | t2va | 5 s | 42 | 文本 | 特效与奇幻元素（粒子个体性） |
| `s11_micro_expression` | t2va | 5 s | 42 | 文本 | 生动面部表情（微表情） |
| `s12_camera_control` | t2va | 5 s | 42 | 文本 | 摄影运镜与控制（下摇→前推→右摇） |
| `s13_prompt_adherence` | t2va | 5 s | 42 | 文本 | V2 提示词控制力（数量/位置/颜色/文字） |
| `s15_audio_reference` | ref2va | 5 s | 42 | 图,音频 | V2 音频参考条件修正（音色跟随参考音频） |

### 2.4 风格对照（s14a–s14d，4 条）

| ID | 任务 | 时长 | seed | 输入 | 说明 |
| --- | --- | ---: | --- | --- | --- |
| `s14a_style_photoreal` | t2va | 5 s | 42 | 文本 | 风格对照：照片级写实 |
| `s14b_style_east_asian` | t2va | 5 s | 42 | 文本 | 风格对照：东亚写实 |
| `s14c_style_3dcg` | t2va | 5 s | 42 | 文本 | 风格对照：3D CG |
| `s14d_style_cinematic_cg` | t2va | 5 s | 42 | 文本 | 风格对照：电影级 CG |

### 2.5 固定参数

`width=1344, height=756`（必须落在 21:9/16:9/4:3/1:1/3:4/9:16 之一）、`fps=24`、
`num_inference_steps=8`、`extra_params.flow_shift=12`、`audio_flow_shift=3`——即**现网已部署的采样契约**。
自建 4 步 turbo 上跑需改 `num_inference_steps` 并加 `--send-tuning`，否则测的不是同一套契约。


### 中英双语约定

每个用例文件里有：

- **`prompt`（英文）** —— **唯一下发给模型的字段**，符合 H3 的 `integrated_multimodal_description` /
  `overall_soundscape` / `non_diegetic_music`（或 Ref2VA 六段式）格式。
- **`_prompt_zh`（中文）** —— 仅供人读，说明这条在考什么；**`run_vllm_eval_case.py` 会在提交前剥离所有
  `_` 开头的键**，因此它永远不会到达引擎。

### 素材要求

`g2–` `g6`、`g9`、`g11` 需要固定素材，放在 `/nfs-output/h3_gate/assets/`，**首次运行后冻结**：

| 文件 | 用途 | 要求 |
| --- | --- | --- |
| `first_frame.png` / `last_frame.png` | g2 / g3 | 1344×756，主体清晰、光照中性 |
| `first_frame_cycle.png` / `last_frame_cycle.png` | g4 | 同一场景的两端，护栏线与海平线要对得上 |
| `g9_first.png` / `g9_last.png` | g9 | 站台与车厢座位两端，需能连成同一趟车 |
| `ref_person.png` | g5 | 单人三分之四角度，五官清晰 |
| `ref_person_male.png` | s15 | 男性肖像，**提示词性别必须与之相符** |
| `ref_clip.mp4` | g6 / g11 | 3–5 s、24 fps、有明确运镜，结尾需可被接续 |
| `ref_voice.wav` | s15 | 参考音色，建议用与台词同文的 TTS 样本 |

素材换了等于换了考题，跨版本对比即失效。

---

## 3. 提示词设计审核（v1 → v3）

第一版用例写完后按 `.agents/skills/h3-prompt-writing` 的两份格式规范逐条自查，问题如下：

| # | v1 的问题 | v3 的处理 |
| --- | --- | --- |
| 1 | **全篇没有一句台词**，完全没测 H3 的强项——对白与口型 | `g1` 摊主吆喝、`g2` 画外问句、`g5` 一句、`g7` 站务员喊话、`g8` 双人对话、`g9` 让路、`g11` 回头叮嘱，均用 `<d>[Chinese] …</d>` + `(S1)/(S2)` |
| 2 | **`non_diegetic_music` 全是 N/A**，配乐层没被覆盖 | `g8` 独奏钢琴（70 BPM、延音踏板、尾音不解决）、`g10` 128 BPM 合成贝斯（过场滤波、落地静半秒再给低音） |
| 3 | `g3`（尾帧）只有单镜，没验证"尾帧必须落在**最后一个镜头**"这条规则 | 改为两镜：先空镜建立空间，`00:02.500` 切镜后收束到尾帧 |
| 4 | 声音只有一层"环境声"，没有把对白/音效/配乐分层 | 三层分离：台词只放描述里、`overall_soundscape` 只写环境与拟音、`non_diegetic_music` 只写观众听到的配乐 |
| 5 | 时长只有 1 条 15s，且没覆盖对话/配乐/高速/参考视频 | **15s 扩到 5 条**，分别压长片一致性、双人对话+配乐、首尾帧三镜、高速动作+慢动作、参考视频延续 |
| 6 | 相机描述偶有"参数清单"味（如只给"肩高、一米后"） | 统一改成动作化描述：类型 + 方向 + 幅度 + 速度（"并行跟随、同高度同速"、"缓慢前移"） |
| 7 | 画面文字只放双引号，没写"逐字保留" | 每条含文字的用例都显式要求原有文字保持不变；`g5` 的 `retention_analysis` 明确把新文字列为 **new**，不算作参考丢失 |
| 8 | 没有标注每条在考什么的"强项" | `manifest.json` 每条增加 `covers` 与 `strengths` 两个字段 |

### 提示词自检清单（新增用例时必须过一遍）

- [ ] 字段齐全且**顺序正确**：base 模式三段；Ref2VA 六段
- [ ] 首镜写 `[Shot 1]` **不带时间码**；后续切镜 `[Shot 2] At 00:03.500, …`，时间**严格递增**
- [ ] 所有切镜与事件都在片长内；结尾时间与 `seconds` 一致（两位小数）
- [ ] 参考标签（`<Picture N>` / `<Video N>` / `<Subject N>`）只用于**实际提供**的素材，且六个字段里保持一致
- [ ] 台词只写语言标签与原文：`<d>[Chinese] …</d>`；说话人用稳定 `(S1)`；画外音要注明
- [ ] 画面文字用双引号逐字保留
- [ ] 三层声音不重复：对白/现场音乐写在描述里，环境与拟音进 `overall_soundscape`，配乐进 `non_diegetic_music`
- [ ] 相机运动写成动作（方向/幅度/速度），不堆砌关键词
- [ ] 无未替换的模板变量

---

## 4. 怎么跑

```bash
# 1) 起实例（一实例一适配器；动态 LoRA 不能带 --enable-cpu-offload）
PORT=40660

# 2) 逐条提交（驱动会自动剥离 _ 开头的中文说明字段）
for f in docs/h3-gate-cases/g*.json; do
  id=$(basename "$f" .json)
  python3 tools/minimax_h3_turbo/run_vllm_eval_case.py \
    --endpoint http://127.0.0.1:$PORT \
    --request "$f" \
    --tag "gate_$id" \
    --output-root /nfs-output/h3_gate/runs/$(date +%Y%m%d-%H%M) \
    --container <container-name> --timeout-seconds 1800
done
```

> 多 seed 用例（`g1` `g7` `g8` `g10`）改 `seed` 再跑一遍。
> **计时前先空跑 3 条丢弃**：4 步实测要 3 条才到稳态。

---

## 5. 验收门槛

### 5.1 硬门槛（任一不过 = 直接不通过）

| 项 | 判据 |
| --- | --- |
| 完成 | 16 次运行（11 条 + 多 seed 展开）全部 `completed`，无 OOM、无 `no live replica` |
| 时长/分辨率 | 输出时长误差 ≤ 0.2 s，分辨率与请求一致 |
| 音轨 | 有音频流；**峰值 ≥ −20 dBFS**（*预期静音的用例除外，见下方注*）；有声段占比 > 30%；有台词的用例须能听出人声 |
| 确定性 | 同 seed 连跑两次，`output.mp4` 的 sha256 **完全相同** |
| 显存 | 单卡峰值 ≤ 32 GB（40G 卡留 8 GB 余量） |

#### 音频门槛的按用例例外

`−20 dBFS` 不能一刀切：`s1_quant_gradient_sky` 的提示词写的是"近乎无声的旷野环境 + 一声远鸟"，
实测峰值 **−39.2 dBFS**，这是**符合预期的**，不是缺陷。

判定规则改为：

| 用例类型 | 音频判据 |
| --- | --- |
| 有人声/音乐/动作音效 | 峰值 ≥ −20 dBFS |
| 明确要求安静（`s1`） | 有音频流即可，不设电平下限；允许 −45 ~ −30 dBFS |
| 明确要求纯静音（`non_diegetic_music: N/A` 且 soundscape 写 N/A） | 应有静音轨 |

基线实测：`g8` −10.6 / −11.2 dBFS（对白+配乐）、`g10` −5.2 / −3.9 dBFS（动作）、`g1` −16.6 / −13.7、`g7` −7.1 / −10.4、`s1` −39.2（预期安静）。

### 5.2 性能门槛（相对基线，防止用"更慢"换质量）

| 场景 | 基线（v1.1 烘焙） | 允许上限 |
| --- | ---: | ---: |
| 5s | 58–60 s | **≤ 66 s**（1.10×） |
| 15s（热） | 223 s | **≤ 257 s**（1.15×） |
| 冷启动首条 | 187 s | ≤ 240 s |

### 5.3 质量门槛（人工盲评，5 分制）

每条成片按 6 项打分，**任一项 < 3 分 = 不通过**：

| 项 | 5 分 | 3 分（及格线） | 1 分 |
| --- | --- | --- | --- |
| 幻影/重影 | 全片无叠影 | 快速段轻微拖影 | 出现双人/双影 |
| 结构完整性 | 主体与环境稳定 | 局部糊但不散 | 崩坏/条纹/色块 |
| 参考保持 | 身份/构图高度一致 | 可辨认但漂移 | 换了个人 |
| 文字可读 | 牌面文字正确可读 | 字形糊但可猜 | 乱码 |
| **口型/对白** | 口型与台词同步、可听清 | 略有错位但可懂 | 无声或完全不对 |
| **声音层次** | 对白/音效/配乐分明 | 有层次但混浊 | 糊成一团或缺失 |

**盲评纪律**：拼图行序随机、只标 A/B/C…，评完再揭开（`build_panels.py`）。
禁止只用锐度/SSIM 下结论——已证明糊化场景下这两个指标会和肉眼相反。

### 5.4 记录模板

```markdown
# 准入门槛结果 <版本> <日期>
基线：MiniMax-H3-FL2VA-Turbo4-768p-v1.1-BF16
| 用例 | 完成 | 墙钟 | 峰值GPU | 幻影 | 结构 | 参考 | 文字 | 口型 | 声音 | 判定 |
| --- | --- | ---: | ---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| g1 … | ✅ | 61 s | 19.8 GB | 4 | 4 | — | 4 | 4 | 4 | 通过 |
结论：通过 / 不通过（不通过项：…）
```

---

## 6. 已知坑（跑之前先看）

1. **一实例一适配器**：中途换 LoRA 会挂死 worker（`RPC call to execute_model timed out`）。
2. **动态 LoRA 不能开 `--enable-cpu-offload`**，显存会从 ~19.7 GB 涨到 ~30.9 GB。
3. **`VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0` 会让卡住的请求永远 PENDING**（GPU 0%、无报错），设成有限值。
4. **驱动日志里没有 `Turbo adapter` 不代表 LoRA 没生效**：用"同 seed 有无 LoRA 的 sha256 是否不同"判定。
5. **不要混挂仓库旧源码**（8-18 版）与镜像代码（10-15 版），会 `ImportError`。
6. **烘焙前先算 `‖ΔW‖/‖W‖`**：低于 ~1e-4 就别烘，改用动态 LoRA（BF16 ulp ≈ 0.39%）。
7. **容器加 `--init`**，否则 `docker rm -f` 可能因僵尸进程失败。

---

## 7. 现网基线

已完成，29 次运行、全部可追溯，见 **§12**。关键帧四例（`g2`/`g3`/`g4`/`g9`）受网关限制无法覆盖，见 §8.6。

---

## 8. MaaS 现网传输适配（2026-10-26 实测）

现网网关 `https://maas.ovaijisuan.com` 是 **new-api**（`v0.13.2-ovaijisuan-20261025`），
不是自建 vLLM-Omni 的直连形态。两者的请求体**不能直接互换**，实测差异如下。

### 8.1 接口

| 动作 | 路由 | 说明 |
| --- | --- | --- |
| 提交 | `POST /v1/videos` | 请求体 `{model, prompt, seconds, size, ...}`，返回 `task_id` |
| 轮询 | `GET /v1/videos/{task_id}` | `status`：`queued` / `in_progress` / `completed` / `failed` |
| 取片 | `GET /v1/videos/{task_id}/content`，或用 `metadata.url` | 完成后 `metadata.url` 是带签名的 OBS 直链 |

`model` 只有两个 H3 端点：`minimax-h3-fl2va`、`minimax-h3-ref2va`。

### 8.2 字段映射（关键坑）

| 用例字段 | 现网写法 | 坑 |
| --- | --- | --- |
| `width`/`height` | `size: "1344x756"` | **引擎按 size 反推比例**，`1344x768` 会被算成 7:4 而报错 `aspect_ratio must be one of 21:9,16:9,4:3,1:1,3:4,9:16`；`aspect_ratio` 字段**被忽略** |
| `seconds` | `"5"` 字符串 | 支持 **2–20**，越界报错并列出允许值 |
| `seed` | 原样 | 透传 |
| `image_path` | `metadata.src_ref_images: [url 或 data-uri]` | 只接受 **http(s) URL 或 base64/data-uri**，不接受服务器路径 |
| `video_path` | `metadata.reference_videos: [url 或 data-uri]` | 同上 |
| `num_inference_steps` / `extra_params` | 默认**不发** | 基线要测"已部署的采样契约"，只有复跑引擎契约时才用 `--send-tuning` |
| `_prompt_zh` | 不下发 | 执行器剥离所有 `_` 前缀键 |

### 8.3 任务类型可达性（重要）

网关把 `minimax-h3-fl2va` 配成了 **t2v**，`minimax-h3-ref2va` 配成 **r2va**。
实测给 fl2va 传 `image` 会被直接拒绝：

```
模型 minimax-h3-fl2va 的任务类型 t2v 不接受图片输入;
要按图生成请改用支持图片输入的任务类型(i2v/l2va/flf2v)或对应模型
```

而 `/v1/models` 里**没有** i2v / l2va / flf2v 的 H3 条目。因此：

| 用例 | 现网可否跑 |
| --- | --- |
| `g1` `g7` `g8` `g10`（t2va） | ✅ 可跑 |
| `g5` `g6` `g11`（ref2va） | ✅ 可跑，**但需先把素材变成 URL 或 data-uri** |
| `g2`（i2v）、`g3`（l2va）、`g4` `g9`（flf2v） | ✅ **可跑**，写法见 §8.6（`image`/`images` + `metadata.task_type`） |

> 也就是说：16 条基础用例在现网**全部可跑**（t2va 4 条 + 关键帧 4 条 + ref2va 3 条 + 风格 4 条）。
> 关键帧三态的写法见 §8.6；早期「没有入口」的判断是字段用错导致的，已更正。

### 8.4 首条基线结果

`g1_t2va_crowd_motion`（5s，1344x756，16:9，seed 42，`minimax-h3-fl2va`）：

| 项 | 值 |
| --- | --- |
| 状态 | **completed** |
| 墙钟 | **115.5 s** |
| 输出 | 1344×736、24 fps、5.17 s、5.67 MB、含音轨 |
| 主观 | 夜市结构完整、人群与蒸汽自然；招牌「永康牛肉麵」**文字清晰可读**（同一画面出现多处均正确） |
| 产物 | `/nfs-output/h3_gate/runs/20261026-prod/prod_g1_t2va_crowd_motion/` |

作为对照：我们自建的 4 步 turbo 在同一条提示词上出现过重影与糊化——**现网这条基线明显更稳**，
所以本用例集把"现网 ≥ 3 分"作为及格线是合理的。

### 8.5 执行器

`tools/minimax_h3_turbo/run_maas_case.py`：读标准用例 JSON → 按上表映射 → 提交/轮询/取片 → 写
`request.json`、`summary.json`、`output.mp4`。用法：

```bash
python3 tools/minimax_h3_turbo/run_maas_case.py \
  --base-url https://maas.ovaijisuan.com --api-key "$KEY" \
  --request docs/h3-gate-cases/g1_t2va_crowd_motion.json \
  --tag prod_g1_t2va_crowd_motion --output-root /nfs-output/h3_gate/runs/<日期>-prod
```

### 8.6 关键帧三态：可用（此前判定错误，已更正）

**先前的错误结论**：我曾判定 i2v / l2va / flf2v 「不可用」，依据是顶层 `image` 被 400 拒绝、
`metadata.src_ref_images` 被静默忽略。这两条观察本身没错，**但字段用错了**——那是 ref2va 的字段。

**正确用法**（2026-10-26 实测，网关 `v0.13.2-ovaijisuan`）：

| 模式 | 请求写法 | 说明 |
| --- | --- | --- |
| `i2v`（首帧） | `image: <data-uri>` + `metadata.task_type: "i2v"` | 单张图即首帧 |
| `l2va`（尾帧） | `image: <data-uri>` + `metadata.task_type: "l2va"` | 单张图即尾帧 |
| `flf2v`（首尾帧） | `images: [<首帧>, <尾帧>]` + `metadata.task_type: "flf2v"` | **必须是数组**，顺序即首/尾 |

任务类型由 `metadata.task_type` 决定；网关侧 H3 的 `taskType` 配成 `auto`（按输入派生），
未指定时默认 t2v。前端 bundle 里的派生逻辑是 `Xat(hasFirst, hasLast)` → `flf2v` / `l2va` / `i2v`。

**验证证据**（提示词统一为「让这一帧动起来，保持构图不变」）：

- `i2v`：输出首帧与输入图一致（同一女性、米白亚麻衫、木桌白瓷杯、书架与窗光）；
- `flf2v`：首帧 MAD **3.4**、尾帧 MAD **6.1**，而首帧对尾帧 **29.4**
  （MAD 为缩放到 336×189 后的平均绝对像素差，越低越接近）；
- `l2va`：5 s 成片最后一帧落在给定尾帧上。

四条用例（`g2`/`g3`/`g4`/`g9`）已全部跑通，见 §12.5。

> 教训：网关对不同任务类型用**不同字段承载媒体**——关键帧走 `image`/`images` +
> `metadata.task_type`，参考生视频走 `metadata.src_ref_images` / `reference_videos` /
> `reference_audios`。字段用错时不会报错，只会**静默退化成 t2v**。

### 8.7 固定素材的生成（qwen-image-pro）

`g2–g6`、`g9`、`g11` 需要的静帧用 `qwen-image-pro` 生成，脚本
`tools/minimax_h3_turbo/gen_gate_assets.py`（提示词**按 Qwen-Image 的写法**组织：
具体主体 + 角度 + 服装 + 光线 + 背景 + 镜头 + 风格，避免抽象形容词——与 H3 的视频提示词是两种体裁，
不能混用）。已生成 7 张，落在 `/nfs-output/h3_gate/assets/`：

| 文件 | 用途 |
| --- | --- |
| `ref_person.png` | g5 / g11 的人物身份参考 |
| `first_frame.png` | g2 首帧 |
| `last_frame.png` | g3 尾帧 |
| `first_frame_cycle.png` / `last_frame_cycle.png` | g4 首尾帧（同场景两端） |
| `g9_first.png` / `g9_last.png` | g9 站台→车厢两端 |

参考**视频**（g6/g11 的 `<Video 1>`）仍需单独生成：可用 `minimax-h3-fl2va` 先跑一条 t2v 当作源片段，
再把它作为 `metadata.reference_videos` 喂给 `minimax-h3-ref2va`。

---

### 8.8 为什么 5s 会跑到 300 秒：排队，不是模型慢

在现网 `minimax-h3-fl2va`（gpu23 / 10.0.0.6:40011）的引擎日志里直接读出来的：

| 请求 | 降噪循环 | 引擎总耗时（队列空时） |
| --- | --- | --- |
| 5s / 8 步 | 1:24–1:26（10.3–10.8 s/it） | **96–101 s** |
| 15s / 8 步 | 4:12（31.4 s/it） | 260–360 s |

同一条 5s 请求的 `engine=` 在日志里从 96 s 一路出现到 **600 s**：

```
06:54:17 total=194.55s
06:56:17 total=291.10s
06:58:58 total=437.02s
07:01:41 total=600.09s   ← 同一批 4 个 5s 任务挤在队列里
07:05:09 total= 96.18s   ← 队列空了
07:11:18 total= 96.60s
07:12:57 total=101.10s
```

`max_num_seqs=1`，引擎串行处理，后来的请求把等待时间算进了自己的 `engine` 计时。
所以**面板上的 300–600 秒不等于计算时间**。

判读基线时必须区分：

- **纯计算**：5s ≈ 96–101 s，15s ≈ 260–300 s（8 步 INT8 的正常水平）；
- **含排队**：任何大于上界的数字都是队列，不能写进性能门槛的对照。

对照我们自建的 4 步 turbo（5s 58–60 s、15s 223 s），步数减半、时间也大致减半，
说明现网这个耗时本身是合理的。

> 教训：不要在现网上做参数探针。这次探接口时连提了 8 个占位任务，人为制造了队列，
> 一度让 5s 请求看起来要 300–600 秒。

---

### 8.9 现网网关不透传 seed 与采样参数（影响基线方法）

在 `minimax-h3-ref2va`（gpu33）与 `minimax-h3-fl2va`（gpu23）的引擎日志里都能看到：

```
Applied extra_params: {'duration': 5, 'task': 'ref2va'}
Video sampling params: steps=8 guidance=None guidance_2=None seed=None
```

我们用请求体带上了 `seed: 42`，引擎侧始终是 **`seed=None`**。
进一步试验：把 `seed` 塞进 `extra_params` 也没用——new-api 会**重建** `extra_params`，
只保留它自己生成的 `duration` 与 `task`，其余键被丢弃。

同一条用例连跑两次（相同 prompt、相同 seed=42、无音频参考），**帧级像素都不同**：

| 运行 | 第 2 秒帧 sha256（前 20 位） |
| --- | --- |
| `ref2va_s15_noaudio` | `b6117bd81cf9f39bece1` |
| `ref2va_s15_noaudio_b` | `6b237455d2001445461c` |

结论：

1. **现网不具备同 seed 复现能力**，5.1 里"同 seed 两次 sha256 相同"这一条**在 MaaS 上不适用**，
   只能在自建实例上验；对现网基线，改用"多次采样取中位/最快值"的方法。
2. **`num_inference_steps` / `flow_shift` 也无法透传**——所以 s 系列的 8 步 / `flow_shift=12`
   写法与实际执行一致（正好是现网默认值），但**无法再按参数扫描**。
3. 音频参考（`s15`）因此**无法用 A/B 哈希法证明生效**：输出本身随机，差异不能归因。
   要判定只能靠人耳听音色是否贴近参考音频，或对多条样本做音色统计。这是当前方法学的真实边界，
   已如实记在这里，不做过度声称。

---

### 8.10 音频参考（s15）：**已确认生效**

官方声明「V2 音频参考条件修正：支持用音频参考调整角色音色」。三次实验的过程与转向如下。

**第一次（作废）**：参考音频误用 8 月 26 日的旧 smoke 文件，内容与文案无关。人耳：**该音频是男中音，
两条生成却都是女声**——但输入本来就错，不能作为结论。

**第二次（有混淆，结论作废）**：换成 `breeze-tts-2` 合成的同句台词参考，配男性参考图。
但提示词里仍写着 `the woman / She`，**文字性别与图像性别冲突**，模型跟了文字——
于是 M1/M2 输出都是女性画面、女性声音，M3（我把主语改成 `He`）才变成男性画面。
这一轮不能用来判断音频。

**第三次（干净对照）**：提示词与参考图统一为男性（`the man / He`），只差参考音频：

| 组 | 输入 | 0–200 Hz 占比 | 谱质心 |
| --- | --- | ---: | ---: |
| 参考音频（男中音） | — | 1.4% | **846 Hz** |
| **N1** | 男性图 + 男声参考 + 台词 | 6.9% | **738 Hz** |
| **N2** | 男性图 + 无参考 + 台词 | 68.0% | **300 Hz** |

N1 的频谱明显更贴近参考（738 vs 846 Hz），N2 相差很远（300 Hz）。
方向与"音频参考影响音色"一致，**但这只是每组 n=1 的粗略频谱统计**，不足以单独定论。

**结论（人耳确认）**：`N1_with_audio_ref.mp4` 确实是参考音频的音色——**音频参考生效**。
`metadata.reference_audios` 是有效字段，与关键帧/参考图的字段各走各的。

用例已按验证通过的配置固化（`s15_audio_reference.json`）：男性参考图 + 男声参考 + 男性主语提示词，
验证产物 `/nfs-output/h3_gate/runs/20261026-s15v4/n1_male_pronoun_ref/`。

**使用前提（必须记住）**：提示词里的性别/描述必须与参考图一致。
`ref2va` 下**文字性别优先于图像性别**——参考图是男性、提示词却写 `woman/she`，
输出会直接退化成女性，参考音频也会被一起带偏（第二次实验就是这么失败的）。

> 附带发现（已单独记录）：`ref2va` 下**文字性别优先于参考图性别**——
> 参考图是男性、提示词写 `woman/she`，输出就是女性。做身份一致性用例时必须让文字与图像一致。

### 12.6 按官方提示词要求修正后的重跑（8 条）

对 `s1`–`s5` 与 `s14c` 做提示词审计后发现三处不合规，已修正并全部重跑：

| 问题 | 用例 | 修正 |
| --- | --- | --- |
| **完全没有风格行**（官方要求明确指定风格） | `s1`–`s5` | 首行补官方短语 `Photorealistic style, natural lighting, realistic skin texture, …`，并按用例追加一句场景限定 |
| **自相矛盾**：`Wide locked-off view … camera begins a slow horizontal pan` | `s3` | 改为 `Wide static opening view …; the camera is still until the pan begins.` |
| **缺光照** | `s14c` | 官方风格短语保持逐字，正文补 `soft overcast key light from above with cool bounce off the water` |

产物：`/nfs-output/h3_gate/runs/20261026-prod-v2b/`（8 条，全部可追溯）。

| 用例 | 任务 | 时长 | seed | 状态 | 墙钟 s | 相对最快 | 来源 | 幻影 | 结构 | 文字 | 音频 | 备注 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | :-: | :-: | :-: | ---: | --- |
| `s14c_style_3dcg` | t2va | 5s | 42 | completed | 116 | 1.09× | 一致 | - | - | - | - | 计算时间 |
| `s1_quant_gradient_sky` | t2va | 5s | 42 | completed | 106 | 1.00× | 一致 | - | - | - | - | 计算时间 |
| `s1_quant_gradient_sky` | t2va | 5s | 43 | completed | 116 | 1.09× | 一致 | - | - | - | - | 计算时间 |
| `s2_prune_face_texture` | t2va | 5s | 42 | completed | 110 | 1.05× | 一致 | - | - | - | - | 计算时间 |
| `s3_vdn_wide_pan_distant` | t2va | 15s | 42 | completed | 296 | 1.02× | 一致 | - | - | - | - | 计算时间 |
| `s3_vdn_wide_pan_distant` | t2va | 15s | 43 | completed | 291 | 1.00× | 一致 | - | - | - | - | 计算时间 |
| `s4_quant_dark_small_text` | t2va | 5s | 42 | completed | 116 | 1.09× | 一致 | - | - | - | - | 计算时间 |
| `s5_quant_thin_lines_backlit` | t2va | 5s | 42 | completed | 110 | 1.05× | 一致 | - | - | - | - | 计算时间 |

5 s 最快 106 s / 中位 113 s，15 s 最快 291 s / 中位 294 s，与修正前同量级，说明补风格行没有带来性能代价。

> 这几条的视频已替换到 `singularity_demos/` 对应链接，之前基于旧提示词的结果保留在原目录
> `20261026-prod-v2/`，不再作为该用例的当前基线。

---

## 9. 图像素材的提示词规范（qwen-image-pro）

生成用例素材时踩过一次真实的坑：画面里出现**乱码文字/箭头标注**。原因不是模型不行，是提示词写错了。

### 9.1 规范依据

`.agents/skills/minimalist-product-ad-generator/SKILL.md`：

> Include exact visible copy … **in the rendering prompt**, not only in the plan.
> For bad typography, **first reduce simultaneous text or correct its reference.**

推论有两条：

1. **要字就逐字给**：需要出现的文案必须写进提示词原文（和 H3 视频提示词里"可见文字用双引号逐字保留"是同一条规矩）。
2. **不要字就明确禁止并枚举**：只说"无文字"太弱；要列出不许出现的东西（标牌、标语、标注线、箭头、编号、水印、logo），
   否则模型会按"这画面似乎该有信息"自行补字，补出来就是乱码。

### 9.2 我犯的两个具体错误

| 原来的写法 | 为什么错 | 改成 |
| --- | --- | --- |
| 「招牌文字**模糊不可辨认**」 | 等于说"这里该有字，但随便写"——模型就编乱码 | 「无任何招牌文字」 |
| 「画面文字**模糊不可辨认**」 | 同上 | 「画面中无任何文字」 |
| 结尾只写「画面干净无文字」 | 禁止项太少，挡不住箭头/标注/编号 | 统一追加 `NO_TEXT` 常量，枚举 8 类禁止元素 |

### 9.3 现在统一追加的约束

```python
NO_TEXT = (
    "画面中不出现任何文字：不要汉字、字母、数字、符号、标牌、招牌、标语、说明文字、"
    "标注线、箭头、编号、水印或 logo；人物服装与所有物体表面均为无字素面。"
)
```

所有静帧素材都拼接该常量；确需文字时（例如 g5 的笔记本「第 3 天」），
必须在提示词里逐字写出，并**减少同屏文字数量**（规范原文：reduce simultaneous text）。

> 注意：这条约束是给**图像**素材用的；H3 视频提示词走的是另一套格式（见 §3），
> 那里要求把可见文字用双引号写进 `integrated_multimodal_description`。两者不要混用。

## 10. Singularity 专项用例（s1–s5）

现网跑的是 `MiniMax-H3-Singularity-VDN-INT8`，优化栈是四层叠加：

| 层 | 做了什么 | 典型副作用 |
| --- | --- | --- |
| **剪枝 r8** | 把 transformer 压到 rank-8 曲线子空间 | 高频细节被抹平（皮肤、织物、细线） |
| **fixbias** | 修 `prune_partition_to_curve.py` 丢失 AdaLN 偏置的缺陷（v2 曾因此 16 个 block 崩坏） | 偏置异常 → 整体亮度/色偏、块间不一致 |
| **VDN** | `VDN_WINDOW_ATTN` 窗口注意力并烘焙 | 长程依赖弱 → 远景小目标、跨镜头一致性下降 |
| **INT8** | W8A16 量化 | 暗部色带、噪点、高光溢出、小字糊 |

这些不是"模型能力"，而是**工程取舍**——每一层都对应一种特定的劣化方式。
所以门槛集里必须有用例专门盯着它们，否则改量化参数或换剪枝策略时，常规用例可能全过、而某一层已经坏了。

### 用例与优化的对应关系

| 用例 | 压哪一层 | 判读要点 |
| --- | --- | --- |
| `s1_quant_gradient_sky` | INT8 + AdaLN | 大面积平滑渐变里是否出现**色带**、条带边界、块状亮度跳变；天空从靛蓝到灰蓝应连续过渡 |
| `s2_prune_face_texture` | 剪枝 r8 | 毛孔、胡茬、疤痕、羊毛织纹是否还在；剪枝过头会变成"磨皮" |
| `s3_vdn_wide_pan_distant` | VDN 窗口 + 15s | 15 秒慢摇中，远处三条船与桥上行人不许消失/撕裂；这是窗口注意力的长程短板 |
| `s4_quant_dark_small_text` | INT8 暗部 | 三分之二暗部下是否噪点爆开；手掌大的「修鞋」两字是否可读 |
| `s5_quant_thin_lines_backlit` | INT8 高频 | 逆光细电线是否断裂、发毛、带光晕；高对比边缘有没有振铃 |

### 判读加分项

- `s1`：用直方图看天空区域的亮度分布是否连续（色带会在直方图上出现尖峰）。
- `s3`：抽 1.0 / 6.5 / 11.5 s，逐帧确认远处船与人**还在原位**；VDN 若窗口切错，会在摇镜中途丢掉远景。
- `s5`：放大到 200% 看电线边缘，量化振铃表现为线条两侧的周期性亮暗环。

### 运行注意

- 这 5 条按**现网契约**写（8 步、`flow_shift=12`），所以对 MaaS 直接跑即可（执行器默认不下发 tuning 参数）。
- 若要在自建集群的 4 步 turbo 上跑，需加 `--send-tuning` 并同步改 `num_inference_steps`，否则测的不是同一套契约。

### 10.1 官方能力声明 → 用例映射（s6–s14）

Singularity 的官方宣称逐条建用例，任何一条回归都会在这套里露出来：

| 官方声明 | 用例 | 判读要点 |
| --- | --- | --- |
| HDR 画质与模糊抑制 | `s6_hdr_motion_blur` | 强逆光窗口 + 深阴影同框，人物横向快速跑；高光不溢出、暗部不糊成块、四肢不拖影 |
| 远景人脸修复 | `s7_distant_face_repair` | 人脸只占画面高度 1/8→1/6，全程五官不塌、不重影、不融化 |
| 干净自然的视觉美感 | `s8_natural_material` | 生木、素烧陶、亚麻、裸手；皮肤不能像塑料，无塑料高光、无泛光、无发光描边 |
| 增强动态表现力 | `s9_action_impact` | 15s 三段剑术，每击写清起手→接触→跟随；不许肢体瞬移，接触要有反馈 |
| 特效与奇幻元素 | `s10_vfx_fantasy` | 粒子必须是一颗颗独立轨迹，不许糊成贴图、不许穿模 |
| 生动面部表情 | `s11_micro_expression` | 只靠眼神与嘴角的微变化完成表演，不转头不说话 |
| 摄影运镜与控制 | `s12_camera_control` | 指定"下摇→前推→右摇 90°→停住"，自行车 00:04 前不许入画 |
| V2 提示词控制力 | `s13_prompt_adherence` | 恰好三件物体、指定颜色与位置、一处文字、00:03 只拿走苹果 |
| V2 风格渲染增强 | `s14a–s14d` | 同一画面内容仅换 Style 行（photorealistic / East Asian realism / 3D CG / cinematic CG），四张应明显不同 |
| V2 音频参考条件修正 | `s15_audio_reference` | 男性参考图 + 男声参考 + 男性主语；**人耳确认音色跟随参考**（§8.10） |
| 保留基础模型能力 | 基础 11 条（g1–g11） | 声称"不退化"，就用原有用例守住 |

### 10.2 提示词组织（按官方建议）

官方给的建议已落到所有 s 系列用例里：

- **长度**：单条提示词控制在 3000 字符以内（官方上限约 5000），当前最长约 900 字符；
- **显式风格**：每条第一行就是 `Style: …`（photorealistic / East Asian realism / 3D CG / cinematic CG），
  不再依赖"电影感""高级"这类抽象词；
- **动作写轨迹与先后**：`s9` 明确写了起手方向、接触时刻、跟随动作，并按 `[Shot 2] At 00:06.000` 分段；
- **不重复、不冲突**：风格只声明一次；同一场景不同段落不出现互斥的光照或时间描述。

### 10.3 官方风格提示词（逐字）

官方给了四条风格提示词示例，原文存在 `docs/h3-gate-cases/style-reference.md`，此处摘录：

| 风格 | 官方原文 |
| --- | --- |
| 照片级写实 | `Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking.` |
| 东亚写实 | `East Asian realism, a traditional courtyard, realistic fabric textures, soft natural lighting, detailed architecture, fluid martial arts movements, cinematic composition.` |
| 3D CG | `3D CG style, detailed character materials, expressive facial animation, fluid action, depth and dimensionality, polished 3D rendering.` |
| 电影级 CG | `Cinematic CG style, dramatic cinematic lighting, detailed environments, dynamic action choreography, atmospheric effects, film-quality rendering.` |

**用法约定**：

- 风格行放在提示词**第一行**，格式 `Style: <官方原文>`；
- 官方示例里混有**内容线索**（`a traditional courtyard`、`fluid martial arts movements`、
  `detailed architecture`），用在与该内容无关的场景会与画面描述冲突——按官方"避免相互冲突"的要求，
  受控对照时剔除这些词，只保留风格描述；
- `s14a–s14d` 就是受控对照：**同一画面内容（渡轮海景），只换 Style 行**，四张必须呈现明显不同的风格；
- 其余 s 系列已统一改用官方措辞（`s6`/`s8`/`s12`/`s13` 用照片级写实，`s7`/`s11` 用东亚写实，
  `s9`/`s10` 用电影级 CG），并按需追加一条场景限定（如 `Extended dynamic range, highlights and shadows both retained.`）。

## 11. ref2va 三条用例实测通过（2026-10-26）

`g5` / `g6` / `g11` 原先卡在"缺参考视频"，本轮补齐并全部跑通，**三条都用人眼确认参考生效**。

### 11.1 参考图（g5）

请求带 `metadata.src_ref_images`（`ref_person.png` 转 data-uri）。
成片里是**同一张脸、同一件灰开衫白 T**，只是换成坐姿写字、影棚环境。

> 对照：同样的字段在 `minimax-h3-fl2va`（t2v）上被**静默丢弃**（见 §8.6）。
> 两个端点对同一字段的行为完全不同，这是本用例集里最容易踩的坑。

### 11.2 参考视频（g6 / g11）

参考视频用 `minimax-h3-ref2va` 自己生成（`assets_src/ref_clip.json`：暖光房间、高窗、人物转身），
落到 `assets/ref_clip.mp4`（5 s / 1344×756 / 1.6 MB），再作为 `metadata.reference_videos` 喂回。

| 用例 | 时长 | 验证点 | 结果 |
| --- | --- | --- | --- |
| `g6_ref2va_video` | 5 s | 是否从参考片段结尾接续、运镜/色调是否继承 | ✅ 首帧为同一扇窗、同一条窗帘、同一件灰开衫，机位退到门口，与提示词"转向门口"一致 |
| `g11_ref2va_15s_video_continue` | 15 s | 长时程延续 + 跨房间空间连续性 | ✅ 12 s 处正是走廊门口回头镜，走廊纵深、画框与卧室尽头都成立 |

音频峰值：`g5` **−0.6 dBFS**（接近满刻度，需关注是否削波）、`g6` −9.8 dBFS、`g11` −5.2 dBFS。

> `g5` 的 −0.6 dBFS 已超出 `loud` 档的"≥ −20"下限很多，属于**过响**而非过轻——
> 门槛里应补一条上限（建议 ≤ −1 dBFS），否则削波会伪装成"音量充足"。

### 11.3 关键帧三态（已更正）

早期认为 `g2`/`g3`/`g4`/`g9` 在现网无入口，是**字段用错**的判断错误；
正确写法见 §8.6，四条已全部跑通，结果见 §12.5。
---

## 12. 现网基线（2026-10-26 实测，29 次运行）

### 12.1 环境

| 项 | 值 |
| --- | --- |
| 网关 | `https://maas.ovaijisuan.com`（new-api `v0.13.2-ovaijisuan-20261025`） |
| fl2va 端点 | `minimax-h3-fl2va` → gpu23 `10.0.0.6:40011`，`MiniMax-H3-Singularity-VDN-INT8`，8 步、`VDN_WINDOW_ATTN`、TP4 |
| ref2va 端点 | `minimax-h3-ref2va` → gpu33 `10.0.0.148:40012`，`MiniMax-H3-Singularity-VDN-INT8-ref2va`，8 步 |
| 产物 | `/nfs-output/h3_gate/runs/20261026-prod-v2/`（fl2va）与 `20261026-prod-ref2va/`（ref2va） |

### 12.2 口径

- **来源可追溯**：每条运行记录 `prompt_sha256`，与当前用例比对；本次 29 条**全部「一致」**。
- **计算时间 vs 排队**：网关 `max_num_seqs=1` 串行，取同时长最快样本为计算基准，超 1.25× 标「含排队」。
- **评分**：幻影/结构/文字 5 分制，人工看片；音频只测峰值电平（dBFS），不看内容。
- **音频期望**：按 `audio_db_baseline` 相对判定（±6 dB），绝对档位只作兜底。

### 12.3 结果

| 用例 | 任务 | 时长 | seed | 状态 | 墙钟 s | 相对最快 | 来源 | 幻影 | 结构 | 文字 | 音频 | 备注 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | :-: | :-: | :-: | ---: | --- |
| `g10_t2va_15s_action_vfx` | t2va | 15s | 42 | completed | 291 | 1.02× | 一致 | 4 | 4 | - | -5.2 dB | 计算时间 |
| `g10_t2va_15s_action_vfx` | t2va | 15s | 43 | completed | 306 | 1.07× | 一致 | 4 | 5 | - | -3.9 dB | 计算时间 |
| `g1_t2va_crowd_motion` | t2va | 5s | 42 | completed | 176 | 1.67× | 一致 | 5 | 5 | 5 | -16.6 dB | 含排队（最快 105 s 的 1.7×） |
| `g1_t2va_crowd_motion` | t2va | 5s | 43 | completed | 116 | 1.10× | 一致 | 5 | 5 | 5 | -13.7 dB | 计算时间 |
| `g7_t2va_15s_text_long` | t2va | 15s | 42 | completed | 291 | 1.02× | 一致 | 5 | 5 | 5 | -7.1 dB | 计算时间 |
| `g7_t2va_15s_text_long` | t2va | 15s | 43 | completed | 291 | 1.02× | 一致 | 5 | 5 | 5 | -10.4 dB | 计算时间 |
| `g8_t2va_15s_dialogue_music` | t2va | 15s | 42 | completed | 291 | 1.02× | 一致 | 5 | 5 | 5 | -10.6 dB | 计算时间 |
| `g8_t2va_15s_dialogue_music` | t2va | 15s | 43 | completed | 286 | 1.00× | 一致 | 5 | 5 | 5 | -11.2 dB | 计算时间 |
| `s10_vfx_fantasy` | t2va | 5s | 42 | completed | 116 | 1.10× | 一致 | 5 | 5 | - | -5.5 dB | 计算时间 |
| `s11_micro_expression` | t2va | 5s | 42 | completed | 110 | 1.05× | 一致 | 5 | 5 | - | -35.9 dB | 计算时间 |
| `s12_camera_control` | t2va | 5s | 42 | completed | 116 | 1.10× | 一致 | 5 | 5 | - | -24.3 dB | 计算时间 |
| `s13_prompt_adherence` | t2va | 5s | 42 | completed | 116 | 1.10× | 一致 | 5 | 5 | 5 | -20.4 dB | 计算时间 |
| `s14a_style_photoreal` | t2va | 5s | 42 | completed | 116 | 1.10× | 一致 | 5 | 5 | - | -22.1 dB | 计算时间 |
| `s14b_style_east_asian` | t2va | 5s | 42 | completed | 116 | 1.10× | 一致 | 5 | 5 | - | -28.3 dB | 计算时间 |
| `s14c_style_3dcg` | t2va | 5s | 42 | completed | 116 | 1.10× | 一致 | 5 | 5 | - | -23.3 dB | 计算时间 |
| `s14d_style_cinematic_cg` | t2va | 5s | 42 | completed | 110 | 1.05× | 一致 | 5 | 5 | - | -20.2 dB | 计算时间 |
| `s1_quant_gradient_sky` | t2va | 5s | 42 | completed | 120 | 1.14× | 一致 | 5 | 5 | - | -39.2 dB | 计算时间 |
| `s1_quant_gradient_sky` | t2va | 5s | 43 | completed | 116 | 1.10× | 一致 | 5 | 4 | - | -38.3 dB | 计算时间 |
| `s2_prune_face_texture` | t2va | 5s | 42 | completed | 115 | 1.09× | 一致 | 5 | 5 | - | -41.2 dB | 计算时间 |
| `s3_vdn_wide_pan_distant` | t2va | 15s | 42 | completed | 291 | 1.02× | 一致 | 5 | 5 | - | -34.6 dB | 计算时间 |
| `s3_vdn_wide_pan_distant` | t2va | 15s | 43 | completed | 296 | 1.03× | 一致 | 5 | 5 | - | -31.2 dB | 计算时间 |
| `s4_quant_dark_small_text` | t2va | 5s | 42 | completed | 106 | 1.00× | 一致 | 5 | 5 | 5 | -11.6 dB | 计算时间 |
| `s5_quant_thin_lines_backlit` | t2va | 5s | 42 | completed | 120 | 1.14× | 一致 | 5 | 5 | - | -23.5 dB | 计算时间 |
| `s6_hdr_motion_blur` | t2va | 5s | 42 | completed | 105 | 1.00× | 一致 | 4 | 4 | - | -24.4 dB | 计算时间 |
| `s6_hdr_motion_blur` | t2va | 5s | 43 | completed | 106 | 1.00× | 一致 | 4 | 4 | - | -25.1 dB | 计算时间 |
| `s7_distant_face_repair` | t2va | 5s | 42 | completed | 105 | 1.00× | 一致 | 5 | 5 | - | -32.5 dB | 计算时间 |
| `s8_natural_material` | t2va | 5s | 42 | completed | 120 | 1.14× | 一致 | 5 | 5 | - | -17.6 dB | 计算时间 |
| `s9_action_impact` | t2va | 15s | 42 | completed | 296 | 1.03× | 一致 | 5 | 5 | - | -8.2 dB | 计算时间 |
| `s9_action_impact` | t2va | 15s | 43 | completed | 291 | 1.02× | 一致 | 5 | 5 | - | -4.4 dB | 计算时间 |

**峰值显存**（网关不返回，在 gpu23 上用 `nvidia-smi` 每 2–3 s 采样实测）：

| 时长 | 峰值/卡 | 采样 |
| --- | ---: | --- |
| 5 s | **19.4 GB**（19417 MiB） | 64 个采样点，四卡一致 |
| 15 s | **22.5 GB**（22503 MiB） | 101 个采样点，四卡一致 |

对照我们自建 4 步 turbo（5s 19.7 GB、15s 25.9 GB），现网 8 步 INT8 的显存**更省**——
INT8 量化与 VDN 窗口注意力的收益主要体现在这里。

**汇总**：完成且可追溯 29 / 29；关键帧四例（g2/g3/g4/g9）被网关拒绝（§8.6），未计入。
5s 最快 **105 s** / 中位 116 s；15s 最快 **286 s** / 中位 291 s。

### 12.4 从基线读出的结论

1. **画面整体很强**：除 `g10`（快动作拖影）与 `s6`（HDR 前景运动模糊）为 4 分外，其余 24 条幻影/结构均为 5 分。
2. **文字渲染是现网强项**：`g7` 的出发信息屏（含整屏时间表小字）、`g8` 的黑板「今日手冲 柑橘花香」、
   `s4` 手掌大的「修鞋」、`s13` 的「2026 归档」全部正确可读。
3. **Singularity 的专项声明基本成立**：剪枝未毁高频（`s2` 毛孔/胡茬清晰）、
   量化无色带（`s1` 阶梯 < 1 级）、VDN 长程稳（`s3` 15s 慢摇远景船一直在）、
   运镜指令可执行（`s12` 严格执行下摇→前推→右摇）、四风格可区分（`s14a–d`）。
4. **唯一系统性弱点**：快速运动下的拖影（`g10`、`s6`），这与官方"HDR 与模糊抑制"的宣称仍有差距。
### 12.5 关键帧三态补充基线（同日，四条）

写法见 §8.6；产物在 `/nfs-output/h3_gate/runs/20261026-prod-keyframe/`。

| 用例 | 任务 | 时长 | seed | 状态 | 墙钟 s | 相对最快 | 来源 | 幻影 | 结构 | 文字 | 音频 | 备注 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | :-: | :-: | :-: | ---: | --- |
| `g2_i2va_first_frame` | i2v | 5s | - | completed | 161 | 1.39× | 一致 | - | - | - | - | 含排队（最快 116 s 的 1.4×） |
| `g3_l2va_last_frame` | l2va | 5s | - | completed | 116 | 1.00× | 一致 | - | - | - | - | 计算时间 |
| `g4_fl2va_first_last` | flf2v | 5s | - | completed | 126 | 1.09× | 一致 | - | - | - | - | 计算时间 |
| `g9_fl2va_15s_keyframes` | flf2v | 15s | - | completed | 322 | 1.00× | 一致 | - | - | - | - | 计算时间 |

人工核验：`g4` 首帧 MAD 3.4 / 尾帧 MAD 6.1（输入首帧对输入尾帧为 29.4），证明首尾两帧都被锚定；
`g2` 首帧与输入图同人同景；`g3` 末帧落在给定尾帧上。

5 s 最快 116 s、15 s 322 s，与 t2va 同量级。

## 13. 附录：29 条用例的英文提示词（逐字）

以下为**实际下发**给模型的英文 `prompt`（中文旁注在各自 JSON 的 `_prompt_zh` 里，不会下发）。
每条前面标注任务类型、时长与 seed；提示词正文与 `docs/h3-gate-cases/*.json` 完全一致。

### g1_t2va_crowd_motion
- 任务 `t2va`，时长 `5 s`，seed `42,43`
- 音频期望：`loud`

```text
integrated_multimodal_description: [Shot 1] Handheld medium tracking shot inside a crowded night market alley, wet asphalt mirroring saturated red and green signboards. A courier in a dark rain jacket walks away from camera at a steady pace, weaving between stalls. Steam rises from a griddle at frame left, where a vendor (S1) in a blue apron flips skewers and calls out over her shoulder: <d>[Chinese] 牛杂汤好啦，趁热吃！</d> Three pedestrians cross from right to left. The camera follows one metre behind the courier at shoulder height with a slight vertical bob, keeping him sharp while the foreground crowd blurs. At 00:02.000 he glances right at a lit sign reading "永康牛肉麵" in red characters, then keeps walking.

overall_soundscape: Dense crowd murmur, sizzling oil, a metal ladle striking a wok, footsteps on wet pavement, a distant scooter engine.

non_diegetic_music: N/A
```

### g2_i2va_first_frame
- 任务 `i2v`，时长 `5 s`，seed `42`，输入 `image_path`
- 音频期望：`loud`

```text
Picture 1 aligns to Shot 1 at 0.00 seconds and must match exactly.

integrated_multimodal_description: [Shot 1] Starting from <Picture 1>, the seated woman keeps the same identity, cream linen shirt, loose hair and window lighting. She lifts a ceramic cup toward her mouth, pauses as steam crosses her face, then sets it down and asks someone off-screen: <d>[Chinese] 外面还在下雨吗？</d> Her lips move only on that line; the listener stays out of frame. The camera pushes in slowly from a medium shot to a close-up over the full five seconds, ending at 00:05.00 on her eyes. The blurred bookshelf background stays put.

overall_soundscape: Quiet room tone, ceramic meeting wood, rain against glass, a soft exhale.

non_diegetic_music: N/A
```

### g3_l2va_last_frame
- 任务 `l2va`，时长 `5 s`，seed `42`，输入 `last_frame_path`
- 音频期望：`quiet`

```text
Picture 1 aligns to the final frame at 00:05.00 and must match exactly.

integrated_multimodal_description: [Shot 1] Wide static view of an empty rain-slicked crosswalk at dusk; overhead cabling and the storefront geometry of the final frame are already in place, no pedestrians. [Shot 2] At 00:02.500 cut to a slightly tighter angle on the same crossing as a woman with a transparent umbrella walks in from screen left, hesitates at the kerb, then steps onto the road. By 00:05.00 she stands exactly where <Picture 1> places her, umbrella tilted, facing camera-left, with the same puddle reflections.

overall_soundscape: Rain on nylon, tyre hiss, a pedestrian signal ticking, water under a shoe.

non_diegetic_music: N/A
```

### g4_fl2va_first_last
- 任务 `flf2v`，时长 `5 s`，seed `42`，输入 `image_path`
- 音频期望：`medium`

```text
Picture 1 aligns to the first frame at 0.00 seconds, Picture 2 aligns to the final frame at 00:05.00; both must match exactly.

integrated_multimodal_description: [Shot 1] Between <Picture 1> and <Picture 2> the same cyclist rides along the coastal road. He starts upright as in <Picture 1>, the lighthouse passing on his left, then leans into a shallow right-hand bend while the camera tracks alongside at the same height and speed; spray comes off the rear wheel. By 00:05.00 the framing settles into <Picture 2> with the same guardrail line, sea horizon and rider pose. Continuous single take, no cuts.

overall_soundscape: Wind buffeting the microphone, freewheel ticking, waves breaking below, tyres hissing on wet tarmac.

non_diegetic_music: N/A
```

### g5_ref2va_image
- 任务 `ref2va`，时长 `5 s`，seed `42`，输入 `image_path`
- 音频期望：`loud`

```text
subject_definitions: <Subject 1>: the woman in <Picture 1>. Preserve her identity, face proportions, hairstyle and the grey knit sweater.

summary: A five-second studio clip that keeps <Subject 1> exactly as referenced while she writes and speaks one line.

retention_analysis: <Picture 1> — attribute_transfer: identity, hair and sweater preserved; pose, set and the notebook are new.

detailed_description: [Shot 1] <Subject 1> sits at a plain light-grey studio table, three-quarter angle to camera, soft key light from the left. She opens a notebook whose visible heading reads "第 3 天", writes two lines, then looks up at 00:03.000 and says <d>[Chinese] 写完了，我们走吧。</d> Static medium shot, 50mm equivalent, shallow depth of field.

overall_soundscape: Quiet studio room tone, pen on paper, a page turning, chair leather creaking.

non_diegetic_music: N/A
```

### g6_ref2va_video
- 任务 `ref2va`，时长 `5 s`，seed `42`，输入 `video_path`
- 音频期望：`quiet`

```text
subject_definitions: <Video 1>: the source clip. Its camera path, colour grade and the talent's outfit are the properties to inherit.

summary: Continue the situation in <Video 1> for five more seconds with a new action, keeping its look and camera language.

retention_analysis: <Video 1> — partially_preserved: framing, grade and wardrobe inherited; the action and the end state are new.

detailed_description: [Shot 1] Picking up exactly where <Video 1> ends, the same subject remains in the same room under the same warm practical lighting. She turns away from the window, crosses to the doorway, and switches off the lamp at 00:04.000; the frame falls to a dim blue ambient with only the window glowing. The camera holds the <Video 1> position and drifts forward slowly.

overall_soundscape: Room tone, floorboard creaks, a lamp switch clicking, distant traffic muffled by glass.

non_diegetic_music: N/A
```

### g7_t2va_15s_text_long
- 任务 `t2va`，时长 `15 s`，seed `42,43`
- 音频期望：`loud`

```text
integrated_multimodal_description: [Shot 1] Static wide shot of a departure board in a regional Chinese railway station, cool fluorescent light, commuters crossing the lower third. The top line reads "G1024 上海虹桥 08:35". [Shot 2] At 00:07.000 the camera pans right along the platform to a conductor (S1) raising a green flag; lighting and crowd density continue, and the board edge stays visible at frame left. The platform announcement overlaps the pan. [Shot 3] At 00:12.000 cut to a medium shot of the conductor blowing a whistle and calling <d>[Chinese] 车门关闭，请勿靠近。</d> as the doors slide shut.

overall_soundscape: Station announcement reverb, rolling luggage, door chime, whistle blast, crowd shuffling.

non_diegetic_music: N/A
```

### g8_t2va_15s_dialogue_music
- 任务 `t2va`，时长 `15 s`，seed `42,43`
- 音频期望：`loud`

```text
integrated_multimodal_description: [Shot 1] Late-afternoon cafe window seat, low sun raking across a marble table. A man (S1) in a rolled-sleeve shirt and a woman (S2) in a rust-coloured sweater sit opposite each other; a chalkboard behind them reads "今日手冲 柑橘 花香". S1 slides a folded paper across the table and says <d>[Chinese] 你看这个地址，是不是写错了？</d> [Shot 2] At 00:06.000 cut to a close-up on S2 reading the paper, warm light on her cheek; she frowns, then laughs and answers <d>[Chinese] 没错，就是那条巷子。</d> [Shot 3] At 00:11.500 cut back to the wide two-shot as both stand up; S1 pockets the paper and the camera holds while the window light flickers as a bus passes outside.

overall_soundscape: Espresso machine hiss, ceramic cups on saucers, other tables' murmured conversation, cutlery, a bus rumbling past the glass.

non_diegetic_music: Sparse solo piano, around 70 BPM, soft sustain pedal, entering under the first line and thinning to a single unresolved note as they stand.
```

### g9_fl2va_15s_keyframes
- 任务 `flf2v`，时长 `15 s`，seed `42`，输入 `image_path`
- 音频期望：`medium`

```text
Picture 1 aligns to the first frame at 0.00 seconds and Picture 2 aligns to the final frame at 00:15.00; both must match exactly.

integrated_multimodal_description: [Shot 1] Opening exactly on <Picture 1>: the traveller stands at the platform edge with a canvas duffel, morning fog behind the tracks. He checks his ticket and steps back from the yellow line. [Shot 2] At 00:05.500 cut to a wider platform view as the train slides in, the same fog density and the same duffel on his shoulder; he turns toward the opening doors. [Shot 3] At 00:10.000 cut to a medium shot from inside the carriage as he steps up, sets the bag on the rack and says to a seated passenger <d>[Chinese] 借过一下，谢谢。</d> By 00:15.00 the framing settles into <Picture 2> with the same seat row, window light and posture.

overall_soundscape: Fog-muffled platform ambience, train brakes and door chime, duffel strap creaking, footsteps on the carriage step.

non_diegetic_music: N/A
```

### g10_t2va_15s_action_vfx
- 任务 `t2va`，时长 `15 s`，seed `42,43`
- 音频期望：`loud`

```text
integrated_multimodal_description: [Shot 1] Low handheld tracking shot behind a courier sprinting across a rain-soaked rooftop at night, neon signs out of focus below the parapet. His shoes strike standing water and throw sheets of spray toward the lens; the camera shakes with each stride. [Shot 2] At 00:06.000 the action drops into slow motion as he vaults a low vent duct: droplets hang in the air, his jacket hem lifts, and the rooftop lights streak behind him, then time snaps back to real speed as he lands. [Shot 3] At 00:11.000 cut to a static wide as he hits the gravel, rolls once and comes up running toward a stairwell door; the camera holds the rooftop as he disappears into the dark.

overall_soundscape: Rain hammering metal, boots slapping water, cloth snapping in wind, a hollow metallic boom on the vault, gravel scattering on landing, distant siren.

non_diegetic_music: Pulsing synth bass at 128 BPM with a rising filter sweep through the vault, cutting to near silence for half a second on the landing, then a single low hit.
```

### g11_ref2va_15s_video_continue
- 任务 `ref2va`，时长 `15 s`，seed `42`，输入 `video_path`
- 音频期望：`quiet`

```text
subject_definitions: <Video 1>: the source clip. Inherit its camera height, lens character, colour grade and the talent's wardrobe.

summary: A fifteen-second continuation of <Video 1> in which the same subject leaves the room, moves through a hallway and speaks one line at a doorway.

retention_analysis: <Video 1> — partially_preserved: framing, grade and wardrobe inherited; location, action and end state are new.

detailed_description: [Shot 1] Resuming exactly at the last frame of <Video 1>, the same subject turns from the window and walks toward the door; the camera follows at the <Video 1> height with the same slow drift. [Shot 2] At 00:05.000 cut to the hallway beyond the door, same warm practical light spilling from the room; she walks away from camera past a narrow console table. [Shot 3] At 00:10.000 cut to a medium shot at the far doorway where she stops, looks back over her shoulder and says <d>[Chinese] 灯我关了，你先走。</d> then steps out of frame as the hallway dims.

overall_soundscape: Room tone, door latch, floorboards underfoot, a clock ticking in the hallway, the click of a light switch.

non_diegetic_music: N/A
```

### s1_quant_gradient_sky
- 任务 `t2va`，时长 `5 s`，seed `42,43`
- 音频期望：`quiet`

```text
Style: Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking. Extended dynamic range, smooth tonal gradients.

integrated_multimodal_description: [Shot 1] Extremely slow tilt up from a still lake surface to an overcast dusk sky, filling the frame with one continuous tonal gradient from deep indigo at the horizon to pale grey-blue overhead; no objects, no texture, only smooth light. A single distant bird crosses from right to left at 00:03.000. The camera holds a locked horizon and moves at less than one degree per second.

overall_soundscape: Almost silent open-air ambience, faint water lapping, one distant bird call.

non_diegetic_music: N/A
```

### s2_prune_face_texture
- 任务 `t2va`，时长 `5 s`，seed `42`
- 音频期望：`quiet`

```text
Style: Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking. Hard side light, high-frequency skin detail.

integrated_multimodal_description: [Shot 1] Tight close-up on a man's face in his fifties, three-quarter angle, hard side light from a window at frame left. Skin pores, two-day stubble, a thin scar over the right eyebrow and the weave of a charcoal wool coat fill the frame. He turns his head slowly toward the light and narrows his eyes at 00:03.000; the camera pushes in a few centimetres. Background is a dark neutral wall, thrown far out of focus.

overall_soundscape: Quiet room tone, fabric rustling against a chair, one slow breath.

non_diegetic_music: N/A
```

### s3_vdn_wide_pan_distant
- 任务 `t2va`，时长 `15 s`，seed `42,43`
- 音频期望：`quiet`

```text
Style: Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking. Extended dynamic range at golden hour.

integrated_multimodal_description: [Shot 1] Wide static opening view of a wide river valley at golden hour; the camera is still until the pan begins. In the far distance, three small fishing boats sit on the water and a footbridge crosses the river; tiny hikers move along it. Foreground reeds are in silhouette at frame bottom. [Shot 2] At 00:06.000 the camera begins a slow horizontal pan right along the valley, keeping the boats in view while the bridge slides toward frame centre; reeds continue to sweep across the lower edge. [Shot 3] At 00:11.000 the pan settles on the bridge with the same boats now small at frame left; light has warmed by a few degrees and the hikers are still on the span.

overall_soundscape: Wind through reeds, distant water, faint voices from the bridge far off.

non_diegetic_music: N/A
```

### s4_quant_dark_small_text
- 任务 `t2va`，时长 `5 s`，seed `42`
- 音频期望：`quiet`

```text
Style: Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking. Low-key night lighting, deep shadow detail.

integrated_multimodal_description: [Shot 1] Static medium shot of a narrow shuttered shopfront at night, lit only by one warm bulb under the awning. Deep shadow fills two thirds of the frame; a small hand-painted sign beside the door, no taller than a hand, reads "修鞋" in two characters. An old man in a dark jacket sits on a low stool, repairing a shoe by hand, and looks up at 00:03.500. The camera does not move.

overall_soundscape: Night street hush, a needle through leather, the bulb humming, a distant dog.

non_diegetic_music: N/A
```

### s5_quant_thin_lines_backlit
- 任务 `t2va`，时长 `5 s`，seed `42`
- 音频期望：`quiet`

```text
Style: Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking. High-contrast backlight against an overcast sky.

integrated_multimodal_description: [Shot 1] Low-angle shot looking up at a tangle of overhead power lines and a steel lattice tower against a bright overcast sky; the wires are thin, high-contrast dark lines crossing the whole frame. The camera drifts slowly right, so the wires slide across the sky at a shallow angle. A single pigeon lands on the top wire at 00:03.000 and the wire dips slightly.

overall_soundscape: Light wind, distant traffic hum, wingbeats then a soft coo.

non_diegetic_music: N/A
```

### s6_hdr_motion_blur
- 任务 `t2va`，时长 `5 s`，seed `42,43`
- 对应声明：HDR 画质与模糊抑制
- 音频期望：`loud`

```text
Style: Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking. Extended dynamic range, highlights and shadows both retained.

integrated_multimodal_description: [Shot 1] A runner sprints from frame right to frame left past a floor-to-ceiling window at late afternoon. Blown-out sunlight through the glass (specular highlights on the frame) sits beside deep shadow in the room's far corner. The camera pans with him at matched speed, keeping his torso centred while the window streaks behind. He passes the window between 00:01.000 and 00:02.500, then decelerates into shadow. Both the window highlights and the shadow detail must stay readable; his limbs must not smear.

overall_soundscape: Footsteps on wood, fabric snapping, breath, window glass rattling slightly.

non_diegetic_music: N/A
```

### s7_distant_face_repair
- 任务 `t2va`，时长 `5 s`，seed `42`
- 对应声明：远景人脸修复
- 音频期望：`loud`

```text
Style: East Asian realism, realistic fabric textures, soft natural lighting, detailed environmental materials, cinematic composition. Documentary framing.

integrated_multimodal_description: [Shot 1] Wide shot across a village market lane; people occupy only one eighth of the frame height. Four stallholders stand behind trays of vegetables facing camera, and two customers walk between them. The camera pushes in slowly to a medium wide over five seconds, ending at 00:05.00 with the nearest two faces about one sixth of frame height. Every visible face must keep normal proportions — two eyes, one nose, one mouth, no melting or doubling — through the whole push-in.

overall_soundscape: Market murmur, crates shifting, coins on a tray, a bicycle bell.

non_diegetic_music: N/A
```

### s8_natural_material
- 任务 `t2va`，时长 `5 s`，seed `42`
- 对应声明：干净自然的视觉美感
- 音频期望：`quiet`

```text
Style: Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking. Matte surfaces, no plastic sheen.

integrated_multimodal_description: [Shot 1] Static close view of two bare hands resting on a raw oak table beside a matte ceramic bowl of water and a folded linen cloth. Soft north-window light from the left; the wood shows open grain and a few scratches, the bowl has a dry unglazed rim, the linen shows individual fibres and a crease. Nothing in frame is wet or polished. One finger taps the bowl at 00:03.000 and the water surface trembles. Skin must read as skin, not plastic; no plastic sheen, no bloom, no glowing rim.

overall_soundscape: Quiet room, a fingertip on ceramic, cloth shifting, faint birds outside.

non_diegetic_music: N/A
```

### s9_action_impact
- 任务 `t2va`，时长 `15 s`，seed `42,43`
- 对应声明：增强动态表现力
- 音频期望：`loud`

```text
Style: Cinematic CG style, dramatic cinematic lighting, detailed environments, dynamic action choreography, atmospheric effects, film-quality rendering.

integrated_multimodal_description: [Shot 1] Two fighters in dark training clothes face each other in a bare wooden hall, low side light. Fighter A steps in with a diagonal downward cut from his right shoulder toward B's left hip; B parries at chest height, the blades meet at 00:02.000 with a visible jarring recoil in both forearms, and sparks are absent — it is steel on steel in daylight. [Shot 2] At 00:06.000 cut to a low angle as B counter-thrusts straight at A's torso; A twists aside, the blade passing his ribs, and his sleeve snaps. [Shot 3] At 00:11.000 cut wide as A sweeps B's lead leg, B drops to one knee, and the impact sends a small puff of dust from the floorboards. Each strike must read as a single continuous arc with a clear start, contact and follow-through; no teleporting limbs.

overall_soundscape: Steel ringing on contact, cloth snapping, boots sliding on boards, a low body-impact thud, both breathing hard.

non_diegetic_music: N/A
```

### s10_vfx_fantasy
- 任务 `t2va`，时长 `5 s`，seed `42`
- 对应声明：特效与奇幻元素
- 音频期望：`loud`

```text
Style: Cinematic CG style, dramatic cinematic lighting, detailed environments, dynamic action choreography, atmospheric effects, film-quality rendering.

integrated_multimodal_description: [Shot 1] A robed figure stands in a stone chamber and presses both palms together at chest height; a ring of small ember-like particles ignites between the palms at 00:01.500, spirals outward into a slowly rotating ring, and lights the figure's face from below. At 00:03.500 the ring snaps into a flat disc of pale blue light that casts hard shadows on the walls. Particles must stay individual motes with distinct motion paths, not a blur or a texture still; they must not pass through the hands.

overall_soundscape: Low stone-room reverb, a rising airy whoosh, crackling embers, a soft metallic snap when the disc forms.

non_diegetic_music: N/A
```

### s11_micro_expression
- 任务 `t2va`，时长 `5 s`，seed `42`
- 对应声明：生动面部表情
- 音频期望：`quiet`

```text
Style: East Asian realism, realistic fabric textures, soft natural lighting, detailed environmental materials, cinematic composition. Tight close-up performance.

integrated_multimodal_description: [Shot 1] Locked-off close-up on a woman's face, soft window light from frame right, plain wall behind. She is listening. At 00:01.000 her eyes narrow slightly and her lips part; at 00:02.500 the corners of her mouth tighten as she holds back a reaction; at 00:03.800 she exhales through the nose, her shoulders drop a centimetre and her gaze softens. No head turn, no dialogue, no big smile — the whole performance stays inside the eyes and mouth.

overall_soundscape: Nearly silent room tone, one soft inhale, a faint fabric shift.

non_diegetic_music: N/A
```

### s12_camera_control
- 任务 `t2va`，时长 `5 s`，seed `42`
- 对应声明：摄影运镜与控制
- 音频期望：`quiet`

```text
Style: Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking.

integrated_multimodal_description: [Shot 1] A single continuous take in a covered arcade. The camera starts low, looking up at the vaulted ceiling, then performs exactly this sequence: tilt down to horizontal over the first second; dolly forward at walking pace for two seconds along the arcade; then pan right ninety degrees to end on a bicycle leaning against a pillar, holding it for the final two seconds. No cut, no zoom, no handheld shake — the motion is a smooth gimbal move. The bicycle must be off-screen until the pan brings it in at 00:04.000.

overall_soundscape: Arcade reverb, footsteps, a bicycle freewheel ticking once, distant traffic.

non_diegetic_music: N/A
```

### s13_prompt_adherence
- 任务 `t2va`，时长 `5 s`，seed `42`
- 对应声明：提示词控制力
- 音频期望：`quiet`

```text
Style: Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking.

integrated_multimodal_description: [Shot 1] A plain white tabletop shot from directly above, even soft light, no props beyond what is listed. Exactly three objects, left to right: a red enamel mug, a green apple, and a closed black notebook. The mug sits one hand-width from the left edge, the apple in the exact centre of the frame, the notebook one hand-width from the right edge. The notebook cover carries one line of white text reading "2026 归档". Nothing else is on the table and there are no shadows from off-frame objects. At 00:03.000 a hand enters from the bottom edge and takes only the apple, leaving the other two untouched.

overall_soundscape: Quiet room, ceramic set down, a page edge tapped, one apple lifted.

non_diegetic_music: N/A
```

### s14a_style_photoreal
- 任务 `t2va`，时长 `5 s`，seed `42`
- 对应声明：V2 风格渲染增强
- 音频期望：`medium`

```text
Style: Photorealistic style, natural lighting, realistic skin texture, detailed environmental materials, fluid character movement, cinematic camera tracking.

integrated_multimodal_description: [Shot 1] A young woman in a dark green raincoat stands at the rail of a small ferry, wind lifting her hair, grey sea and low clouds behind her. She looks toward the horizon and does not move except for the wind in her hair and coat; the camera holds a medium shot with a slow drift left. At 00:03.000 she turns her head one quarter toward camera.

overall_soundscape: Wind across an open deck, hull cutting water, a loose rope tapping metal, distant gull.

non_diegetic_music: N/A
```

### s14b_style_east_asian
- 任务 `t2va`，时长 `5 s`，seed `42`
- 对应声明：V2 风格渲染增强
- 音频期望：`medium`

```text
Style: East Asian realism, realistic fabric textures, soft natural lighting, detailed environmental materials, cinematic composition.

integrated_multimodal_description: [Shot 1] A young woman in a dark green raincoat stands at the rail of a small ferry, wind lifting her hair, grey sea and low clouds behind her. She looks toward the horizon and does not move except for the wind in her hair and coat; the camera holds a medium shot with a slow drift left. At 00:03.000 she turns her head one quarter toward camera.

overall_soundscape: Wind across an open deck, hull cutting water, a loose rope tapping metal, distant gull.

non_diegetic_music: N/A
```

### s14c_style_3dcg
- 任务 `t2va`，时长 `5 s`，seed `42`
- 对应声明：V2 风格渲染增强
- 音频期望：`medium`

```text
Style: 3D CG style, detailed character materials, expressive facial animation, fluid action, depth and dimensionality, polished 3D rendering.

integrated_multimodal_description: [Shot 1] A young woman in a dark green raincoat stands at the rail of a small ferry, wind lifting her hair, grey sea and low clouds behind her; soft overcast key light from above with cool bounce off the water. She looks toward the horizon and does not move except for the wind in her hair and coat; the camera holds a medium shot with a slow drift left. At 00:03.000 she turns her head one quarter toward camera.

overall_soundscape: Wind across an open deck, hull cutting water, a loose rope tapping metal, distant gull.

non_diegetic_music: N/A
```

### s14d_style_cinematic_cg
- 任务 `t2va`，时长 `5 s`，seed `42`
- 对应声明：V2 风格渲染增强
- 音频期望：`medium`

```text
Style: Cinematic CG style, dramatic cinematic lighting, detailed environments, dynamic action choreography, atmospheric effects, film-quality rendering.

integrated_multimodal_description: [Shot 1] A young woman in a dark green raincoat stands at the rail of a small ferry, wind lifting her hair, grey sea and low clouds behind her. She looks toward the horizon and does not move except for the wind in her hair and coat; the camera holds a medium shot with a slow drift left. At 00:03.000 she turns her head one quarter toward camera.

overall_soundscape: Wind across an open deck, hull cutting water, a loose rope tapping metal, distant gull.

non_diegetic_music: N/A
```

### s15_audio_reference
- 任务 `ref2va`，时长 `5 s`，seed `42`，输入 `image_path, audio_path`
- 对应声明：V2 音频参考条件修正
- 音频期望：`loud`

```text
Style: East Asian realism, realistic fabric textures, soft natural lighting, detailed environmental materials, cinematic composition.

subject_definitions: <Subject 1>: the man in <Picture 1> — a man in his forties, short close-cropped hair with greying temples, dark grey crew-neck knit sweater; his identity, face shape and hairstyle must be preserved. <Audio 1>: a voice sample used only to set the timbre and delivery of <Subject 1>'s line.

summary: A five-second studio shot in which <Subject 1> speaks one short line, with the voice timbre guided by <Audio 1>.

retention_analysis: <Picture 1> — attribute_transfer: identity and wardrobe preserved. <Audio 1> — reference: timbre and pacing referenced; the words are new.

detailed_description: [Shot 1] <Subject 1> sits at a plain table facing camera, soft key light from the left. He looks up at 00:01.500 and says, in the timbre of <Audio 1>: <d>[Chinese] 这个方案我看过了，明天给你答复。</d> His lips close after 00:04.000. Static medium shot, shallow depth of field.

overall_soundscape: Quiet studio room tone, chair leather creaking, one soft breath.

non_diegetic_music: N/A
```

---

## 14. Singularity v2 部署与首轮测试（2026-10-26）

### 14.1 制品

`/nfs-models/wuhanjisuan894/models/`（与 `/nfs-data/models/` 同一份 30T 存储）：

| 目录 | 大小 | 说明 |
| --- | ---: | --- |
| `MiniMax-H3-Singularity-v2-pruned-r8-fixbias` | 38 G | 剪枝 BF16（修好 AdaLN 偏置的底座） |
| `...-pruned-r8-fixbias-VDN` | 38 G | + VDN 烘焙 |
| **`...-pruned-r8-fixbias-VDN-INT8`** | **20 G** | **+ INT8，fl2va 生产候选** |
| **`...-pruned-r8-fixbias-VDN-INT8-ref2va`** | **20 G** | ref2va 形态 |

对照组 v1 = 现网在跑的 `MiniMax-H3-Singularity-VDN-INT8`（44 G，**未剪枝**）。

### 14.2 部署命令（与现网同构）

```bash
docker run -d --name h3-v2-fixbias-int8 --gpus all --network host --ipc host \
  --shm-size 32g --init \
  -e VLLM_OMNI_VDN_ALLOW_UNTRAINED_TASKS=1 -e VLLM_OMNI_VDN_KV_GATHER_BUDGET_MB=192 \
  -e VLLM_OMNI_H3_EXPERIMENTAL_SHORT_EDGE=1080 \
  -e VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=900 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /nfs-data:/nfs-data -v /nfs-models:/nfs-models -v /nfs-output:/nfs-output \
  $IMG vllm serve $MODEL --omni \
    --deploy-config deploy-configs/minimax_h3_vdn8_fl2va_pruned_a100_40g.yaml \
    --lora-path .../VDN-MiniMax-H3/stage-dmd-step-250-curve-fl2va-baked2 \
    --allowed-local-media-path /nfs-output \
    --init-timeout 2400 --stage-init-timeout 2400 \
    --host <ip> --port 40700 --trust-remote-code
```

- **加载 270 s**（v1 是 200 s，v2 权重更小但初始化路径相同量级）。
- 两份 deploy config（`..._singularity_vdn8_fl2va_...` 与 `..._vdn8_fl2va_pruned_...`）**逐项 diff 完全一致**，
  所以 v1/v2 的差别只来自权重。

### 14.3 首轮结果（同 seed=42、同配置，各 6 条）

| 用例 | v1 墙钟 | v2 墙钟 | 画面结论 |
| --- | ---: | ---: | --- |
| s1 渐变 | 174 s（首条含编译） | 118 s | 两条都无色带；v2 明暗跨度更小（6.1 vs 13.2 级） |
| s2 面部 | 100 s | 108 s | 两条皮肤毛孔/胡茬都在，**v2 不软** |
| s4 暗部小字 | 101 s | 103 s | 两条都出现「修鞋」招牌，暗部可控 |
| s5 逆光细线 | 104 s | 105 s | 两条线条连续 |
| s13 提示词遵循 | 102 s | 100 s | 两条都满足三物体 + 「2026 归档」 |
| g1 夜市 | 103 s | 100 s | 两条都成立，v2 招牌「永康牛肉面」清晰 |

**延迟无差异**（warm 100–108 s），**v2 没有复现 10-24 的崩坏**，fixbias 重建有效。

### 14.4 两个被自建引擎暴露的用例参数错误

现网网关会丢弃 `num_inference_steps` 与 `extra_params`，所以这两个错误在 MaaS 上**永远不会报**，
但自建实例会直接拒绝：

| 字段 | 原值 | 现网实际执行 | 已改为 |
| --- | --- | --- | --- |
| `num_inference_steps` | 4（g 系列遗留 turbo 契约） | 8 | **8** |
| `extra_params.flow_shift` | 6 | 12 | **12** |

引擎报错原文：
`this VDN artifact carries a 8-step DMD adapter and requires num_inference_steps=8, got 4`、
`VDN-H3 requires flow_shift=12, got 6`。

> 修正后 prompt 未变，因此 `prompt_sha256` 仍与现网基线一致，历史基线不受影响。

### 14.5 一个不能用的指标

同一 case 上比 Laplacian 方差（清晰度）会得出"v2 明显更软"（s2 58.7 vs 108.2），
但两者**取景不同**（v1 是侧脸带头发胡须，v2 是正脸特写），指标被内容干扰。
视觉核对后 v2 细节并不差。**跨模型比清晰度必须固定内容或人工判读，此指标只能同内容自比。**

### 14.6 v1 / v2 全量对照（25 条，同配置同 seed）

两台实例：v1 = `MiniMax-H3-Singularity-VDN-INT8`（gpu6），v2 = `...-v2-pruned-r8-fixbias-VDN-INT8`
（gpu10）。**deploy config 逐项 diff 完全一致**，LoRA 同为 `stage-dmd-step-250-curve-fl2va-baked2`，
环境变量相同，只差权重。

| 用例 | v1 (s) | v2 (s) | 差 |
| --- | ---: | ---: | ---: |
| g10 15s 动作 | 367 | 384 | +17 |
| g1 夜市 | 101 | 103 | +2 |
| g2 首帧 | 147 | 149 | +2 |
| g3 尾帧 | 107 | 108 | +1 |
| g4 首尾帧 | 114 | 119 | +5 |
| g7 15s 信息屏 | 309 | 323 | +14 |
| g8 15s 对话 | 310 | **408** | **+98** |
| g9 15s 首尾帧 | 386 | 379 | −7 |
| s10 特效 | 101 | 100 | −1 |
| s11 微表情 | 98 | 100 | +2 |
| s12 运镜 | 101 | 101 | 0 |
| s13 提示词遵循 | 101 | 100 | −1 |
| s14a 照片级 | 100 | 103 | +3 |
| s14b 东亚写实 | 98 | 101 | +3 |
| s14c 3D CG | 108 | 101 | −7 |
| s14d 电影级 CG | 98 | 100 | +2 |
| s1 渐变 | 98 | 101 | +3 |
| s2 面部 | 98 | 101 | +3 |
| s3 15s 慢摇 | 296 | 296 | 0 |
| s4 暗部小字 | 98 | 100 | +2 |
| s5 逆光细线 | 102 | 103 | +1 |
| s6 HDR | 100 | 100 | 0 |
| s7 远景人脸 | 101 | 102 | +1 |
| s8 材质 | 107 | 102 | −5 |
| s9 15s 剑术 | 301 | 292 | −9 |

**合计：v1 3947 s，v2 4076 s（+3.3%）。剔除 g8 那一条离群值后为 +0.9%**——延迟上两者没有实质差别。

画面（逐条抽帧对照，拼图在 `/nfs-output/h3_gate/runs/v1v2_compare/sheet.png`）：

- 两边都**没有崩坏**，任务类型覆盖（关键帧/参考/风格/特效）全部成立；
- 主观可见差异主要是随机的构图差别，而非质量档差；
- 少数几条 v2 略弱：`s1` 明暗跨度更小（灰天空更平）、`s4` 整体更暗、`s6` 运动拖影更明显；
- v2 略强的：`g1` 招牌文字更清晰、`g8` 黑板字更大更完整。

### 14.7 ref2va 的 v2 也验证过

`...-pruned-r8-fixbias-VDN-INT8-ref2va` 在 gpu14（`10.0.0.144:40702`）起服务，四条用例全通：

| 用例 | 墙钟 | 结论 |
| --- | ---: | --- |
| g5 参考图 | 190 s | 身份保持（同一女性、灰开衫） |
| g6 参考视频 | 331 s | 从参考片段接续 |
| g11 15s 延续 | 682 s | 完成（长片 + 参考视频，耗时约为普通 15s 的 2.3 倍） |
| s15 音频参考 | 113 s | 完成 |

### 14.8 烘焙分析：结构对，但两处来源不清

用张量级比对（`safetensors` 直读，不加载全模型）得到：

**已验证正确**

| 检查项 | 结果 |
| --- | --- |
| AdaLN 权重形状 | `(96768, 8)`，四个构建阶段一致 |
| 时间曲线表 | `(1025, 8)`，二阶/一阶差分比 **0.008**，曲线平滑 |
| INT8 是否误量化 AdaLN | 否，仍在 `ignored_layers`，BF16 与前一阶段逐位一致（Δ=0） |
| VDN 阶段是否改动 | 改了 271 个张量，覆盖 50 个 block 的 attn/mlp/adaln_proj |

**两处疑点（未解决）**

1. **VDN 的改动不是低秩的**：`attn.qkv_proj` 的 Δ 需要 ~4300 秩才到 99% 能量（矩阵宽 5376），
   rank-128 只占 82%。若设计上是"把低秩适配器折进权重"，这个 Δ 形态不符——更像整套权重被重新烘焙，
   或该目录与父目录并非"父 + 适配器"关系。需要构建者说明来源。
2. **运行时 LoRA 的曲线投影不是对着 v2 底座拟合的**：`curve_projection.json` 记录的拟合底座是
   `/nfs-data/models/MiniMax-H3-FL2VA-Pruned-r8-BF16-vLLM`，其 `time_embedder.table` 与 v2 底座
   相对差 ≈ **1.98**。该投影用的是 `adaln_curve_affine_lstsq_pinv1025_with_diff_b`（按参考曲线做
   最小二乘），底座表不同意味着拟合前提不成立。

> 这两点不构成"烘焙错误"的结论，但也不能算已验证正确。当前全量对照里 v2 表现正常，
> 说明即使存在不匹配也没有造成可见劣化；若要上线，建议先让构建者确认这两处来源。

### 14.9 一个必须记录的更正

先前我说"现网 v1 未剪枝"是**错的**。芯片层面确认：现网 `MiniMax-H3-Singularity-VDN-INT8` 的
`transformer/config.json` 里 `adaln_rank: 8`、`time_table_size: 1025`，**它是 AdaLN rank-8 剪枝版**，
只是剪枝来源是官方 `multimodalart/MiniMax-H3-Pruned`（写在 metadata 的 `pruned` 块里）；
而 v2 是本仓 `prune_partition_to_curve.py` 的产物（`_singularity_pruned: true`）。
两者 AdaLN 权重相对差 ≈ **1.03**，基本互不相关——所以这次对照是"官方剪枝 vs 自家剪枝+VDN"，
不是"剪枝 vs 未剪枝"。真正未剪枝的是 `MiniMax-H3-Singularity-v2-VDN-INT8`（44 G，
`_singularity_pruned: false`，无 adaln_rank）。

### 14.10 BF16 + VDN（不剪枝、不量化）的资源与耗时实测

既想排除量化又想排除剪枝，就只剩官方 BF16 底座 + VDN 烘焙。这一档**能跑**，但只限于原生分辨率。

**制品**：`/nfs-models/wuhanjisuan894/models/MiniMax-H3-Singularity-v2-VDN`
（62 GB，BF16，17 个分片，`_singularity_pruned: false`，无 `adaln_rank`、无量化配置）

**部署**：`deploy-configs/minimax_h3_singularity_vdn8_fl2va_a100_40g.yaml` + **不加载 LoRA**
（曲线 LoRA 只适用于剪枝版，未剪枝 abaseline 没有对应适配器），其余环境变量与现网一致。

| 指标 | 实测 |
| --- | --- |
| 加载耗时（容器起 → Application startup complete） | **796 s（13 分 16 秒）** |
| 权重常驻（4 卡） | 加载后 API 就绪时仅 543 MiB/卡，**首次请求才真正进显存** |
| 5 s 成片墙钟（6 条均值） | **84–86 s** |
| 峰值显存 | **19.23 GiB/卡**（4 卡一致） |
| 峰值主机内存 | **184 GiB** |
| 18 步之外的额外开销 | 无；8 步，`flow_shift=12` |

**与 INT8 剪枝两档对比（同样 6 条用例）**

| 组 | 权重 | 5 s 墙钟 | 峰值显存/卡 |
| --- | ---: | ---: | ---: |
| v1 INT8 剪枝（官方 lineage） | 21 GB | 98–102 s | 19.4 GiB |
| v2 INT8 剪枝（fixbias） | 21 GB | 100–103 s | 19.4 GiB |
| **BF16 未剪枝 + VDN** | 62 GB | **84–86 s** | **19.23 GiB** |

两点反直觉但可解释：

1. **显存几乎相同**：INT8 是 `weight_only` 序列化检查点，运行时反量化为 BF16，
   所以两者显存占用接近；BF16 反而略低一点。
2. **BF16 更快**（84 s vs 100 s）：省掉了每次前向的反量化开销。

**唯一的硬约束：`VLLM_OMNI_H3_EXPERIMENTAL_SHORT_EDGE=1080` 会 OOM。**
同一条用例在 1080 短边下报 `Cuda failure 2 'out of memory'`（4 s 内失败），
去掉该变量按原生 756 跑则正常。也就是说这一档**只能在 768P 档使用**。

> 更正记录：我先前把一次 1080 档的 OOM 误判为"756 也 OOM"，据此得出"BF16 跑不起来"的结论，
> 是错的。实际重启（去掉 1080）后一直可跑，6 条 5 s 用例全部完成，
> 上面的数字是重测后的结果。

画面（对照 `runs/compare3_sheet.png`）：BF16 档的皮肤纹理、逆光细线、招牌文字都不弱于 INT8 两档，
`s1` 的明暗跨度反而更接近提示词要求。
