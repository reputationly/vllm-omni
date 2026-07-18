# vLLM-Omni · Ming-omni-tts 实验测试报告(❌ 本镜像启动失败)

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 0017
> 镜像:`reputationly/vllm-omni:arm64-a100-latest`(digest 59add8a0)
> 模型:`Ming-omni-tts-0.5B`,权重 `/nfs-models/wuhanjisuan894/vllm-omni-speech/Ming-omni-tts-0.5B`
> 日期:2026-07-18

---

## 0. 结论先行

**❌ 无法在本镜像启动 —— 模型权重加载报错(代码适配 bug,非硬件/依赖/配置)。** 标记为"本镜像不支持",需改 vllm-omni 模型实现后再测。

---

## 1. 崩溃根因

serve 启动在 **StageEngineCoreProc 加载权重阶段**直接失败,容器退出:

```
File ".../vllm/model_executor/models/qwen2.py", line 438, in load_weights
File ".../vllm/model_executor/models/utils.py", line 392, in _load_module
    raise ValueError(msg)
ValueError: There is no module or parameter named 'lm_head' in Qwen2Model.
The available parameters belonging to (Qwen2Model) are: {'layers.*', 'embed_tokens.weight', 'norm.weight', ...}
```

**分析**:
- Ming-omni-tts 的 checkpoint 里带 `lm_head.*` 权重,但 vllm-omni 给它套用的是 **`Qwen2Model`**(不含 `lm_head` 的裸 backbone,`lm_head` 属于 `Qwen2ForCausalLM`)。
- `AutoWeightsLoader` 试图把 `lm_head` 塞进 `Qwen2Model` → 找不到该参数 → `ValueError`,引擎核初始化失败 → `Orchestrator initialization failed`。
- 属于**模型注册/权重映射不匹配**:要么该模型的类应为 `*ForCausalLM`(含 lm_head),要么其 `load_weights` 需 skip/映射 `lm_head`。
- 与 sm_80 / arm64 / 离线 / 依赖**无关**,`--enforce-eager`、`HF_HOME`、`--allowed-local-media-path` 均已挂,不影响。

---

## 2. 复现

```bash
docker run -d --name omni-ming --gpus '"device=0"' --memory=240g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=$ROOT/hf_cache \
  -v $ROOT:$ROOT -p 8091:8091 \
  "$IMG" vllm serve "$ROOT/Ming-omni-tts-0.5B" --omni --trust-remote-code --enforce-eager --port 8091
# 容器数秒内退出;根因:
docker logs omni-ming 2>&1 | grep "no module or parameter named 'lm_head'"
```

---

## 3. 处置

| 项 | 结论 |
|---|---|
| 现状 | ❌ 启动即崩,无法服务 |
| 类别 | vllm-omni 模型适配代码 bug(`lm_head` vs `Qwen2Model`) |
| 是否本轮可救 | 否(需改模型实现,非部署侧) |
| 建议 | 提 issue / 查该模型正确的 model class 与 `hf_to_vllm_mapper`;修好后再走通用 TTS 轮 |
| 待办 | 确认 Ming-omni-tts 是否需专用 model class(含 lm_head)或 weight skip 规则 |
