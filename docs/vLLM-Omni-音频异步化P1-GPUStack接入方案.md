# vLLM-Omni 音频异步化(P1)· GPUStack 接入方案

> 日期:2026-07-18 · 目标:给 vllm-omni 加一套**音频异步任务 API**,使其能按 GPUStack「新引擎内嵌方法论」的**异步契约**接入(和 IndexTTS 同款),支撑短剧配音(TTS/多人/长句/唱段/音乐)投产。
> 参考:`gpustack/docs/新引擎内嵌gpustack-工程化方法论.md` §4/§6;`index-tts/docs/indextts2-异步内嵌-改动方案.md`(第二引擎范式)。
> 环境:鲲鹏 ARM + 4×A100 40G,隔离网,权重预下 NFS。

---

## 0. 为什么必须异步(不是同步)

- **生成可能 >1 分钟**:MOSS-TTSD 多人长对话(上下文可达 3600s)、SoulX 唱段、Stable-Audio/ACE 音乐(扩散多步)、长文 TTS —— 同步 HTTP 要么网关超时要么连接中断。
- **平台一致**:GPUStack 门面 + new-api 已围绕异步契约(`/v1/tasks/{kind}/` + 轮询 + NFS 落盘)建成,IndexTTS 已按此接入。vllm-omni 说同款契约 → 门面/new-api 侧几乎零改。
- **多副本正确性**(方法论 §6.1):异步(有状态)引擎必须走内置 backend + 异步契约,否则纯 Custom + 轮询会命中别的副本 404。

**结论**:走 **B1 异步**,vllm-omni 引擎侧做 P1 改造。

---

## 1. 关键前提:不动任何单模型代码,不删同步端点

异步层**包在模型无关的服务 handler 之上**,模型侧零感知:

```
现同步:  POST /v1/audio/speech      → Omnispeech handler → 当前加载模型 → 返字节
P1 异步:  POST /v1/tasks/audio/      → 后台 job → 调同一个 Omnispeech handler → 落盘 save_result_path(NFS) → 轮询取件
                                        ↑ 只是把现有 handler 包进后台任务
```

- 所有 TTS 模型走同一 `Omnispeech` handler(`api_server.py:create_speech` → `Omnispeech(raw_request)`),与具体模型无关 → **每个模型实现一行不改**。
- 同步 `/v1/audio/speech` **保留**(方法论 §4.4:留作手测),只**新增**异步路由,**零回归**。

### 分期(P1 只做 tts,唱段/音乐后置)

| task_type | 端点 | 底层 handler | 覆盖 | 期 |
|---|---|---|---|---|
| **tts** | `/v1/tasks/audio/` | `/v1/audio/speech` → Omnispeech | Qwen3-TTS/VoxCPM2/CosyVoice3/Ming/MOSS-*/GLM | **P1** |
| singing | **新开** `/v1/tasks/singing/` | `/v1/chat/completions`(多模态) | SoulX-Singer | P2 |
| audio-gen | **新开** `/v1/tasks/audio-gen/` | `/v1/audio/generate` → OmniAudioGenerate | Stable-Audio(ACE 归独立 agent) | P2 |

**唱段/音乐各开独立端点**(不并进 `/v1/tasks/audio/`):请求字段、输入约束、输出语义都不同(SoulX 要 input_audio+target_audio、走 chat;音乐无音色只有文字+时长)。复用同一套异步基建(§3 的 store/task/原子落盘),但路由/协议/job 各自独立。门面侧对应加 `singing`/`audio-gen` task_type + `_engine_kind` 分支(P2 再做)。

---

## 2. 要实现的异步契约(照方法论 §4.1,门面逐字对齐)

> 路由结构照 LightX2V 源码(`lightx2v/server/api/`):**common 路由跨 kind 共享**(status/result/queue/delete/list),**每 kind 只加一个 `POST /v1/tasks/<kind>/` 提交路由**。

```
# 每 kind 提交(P1 只加 audio;singing/audio-gen 见 §1 P2)
POST   /v1/tasks/audio/            → 200 {task_id, status, save_result_path}
                                    | 503 队列满(RuntimeError→503) | 400 空文本/超长/缺参
# common(跨 kind 共享,audio 直接复用)
GET    /v1/tasks/{task_id}/status  → 200 {task_id, status, start_time, end_time,
                                          error, error_type, save_result_path}
                                    | 404 不存在(触发门面死亡重派)
GET    /v1/tasks/{task_id}/result  → 取件(可选,DONE 后)
GET    /v1/tasks/queue/status      → {is_processing, current_task, pending_count, active_count,
                                      queue_size, queue_available}
GET    /v1/tasks/                  → list
DELETE /v1/tasks/{task_id}         → 取消
GET    /ready                      → 503 加载/warmup 中 | 200 就绪(GPUStack health_check_path)
```

> ⚠️ 字段名以 LightX2V 源码为准:status 响应用 **`start_time/end_time`**(不是 created_at/completed_at);`error_type` 缺省 `""`。落地时再对一遍门面 `routes/videos.py` 的解析,以门面实读为准。

**状态字符串精确集合**(错一个字母 = 门面状态映射失效):
`pending | processing | completed | failed | cancelled`(**cancelled 双 L**)

| 引擎状态 | 门面映射(`gpustack/routes/videos.py:178-184`) |
|---|---|
| pending | ASSIGNED |
| processing | RUNNING |
| completed | DONE |
| failed | FAILED |
| cancelled | CANCELED(单 L) |

> ⚠️ 注意:**不要**照抄视频路径的 `queued/in_progress`(`protocol/videos.py:VideoGenerationStatus`)——那套不符合门面音频契约。音频用上面这 5 个。

---

## 3. 复用清单(现成的通用异步基建,~60%)

| 组件 | 位置 | 复用方式 |
|---|---|---|
| `AsyncDictStore[T]`(泛型任务状态存储) | `entrypoints/openai/stores.py:35-71` | 新建实例 `AUDIO_TASK_STORE: AsyncDictStore[AudioTaskResponse]` |
| `TaskRegistry`(asyncio 后台任务登记 + 自动清理) | `stores.py:12-33` | 新建 `AUDIO_TASKS: TaskRegistry` |
| 原子落盘(`NamedTemporaryFile` → `os.replace`)+ TTL | `storage.py:LocalStorageManager._save_sync:78-118` | 抽出/复用其"临时文件→原子改名"逻辑写到 `save_result_path` |
| `/health`(引擎死/未初始化返 503) | `api_server.py:health:1645-1673` | `/ready` 复用其存活检查 + 加 warmup 就绪门 |
| 任务生命周期模式(create→存→后台 job→更状态→落盘→轮询→content) | `api_server.py:_run_video_generation_job:2796-2877` + `create_video:3044-3075` | 照抄结构,换 audio handler |

---

## 4. 新增清单(音频专属,~200-300 行)

### 4.1 协议模型 — 新文件 `vllm_omni/entrypoints/openai/protocol/audio_tasks.py`

```python
class AudioTaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"          # 双 L

class AudioTaskRequest(BaseModel):
    # 文本(二选一,对齐 OpenAI speech + IndexTTS)
    input: str | None = None
    text: str | None = None
    # 音色:预设 或 克隆参考(NFS 绝对路径,由门面注入)
    voice: str | None = None
    ref_audio: str | None = None      # = spk_audio_path
    ref_text: str | None = None       # GLM/CosyVoice 需要
    instructions: str | None = None   # 情感/风格
    language: str | None = None
    # 情感(IndexTTS 兼容,当前模型多用 instructions)
    emo_audio: str | None = None
    emo_vector: list[float] | None = None   # len==8 校验
    emo_text: str | None = None
    emo_alpha: float | None = None          # 0.0-1.0
    # 异步契约(照 LightX2V BaseTaskRequest 语义)
    save_result_path: str = ""         # 空→缺省 task_id;绝对→原样写;相对→拼输出根;无后缀→自动补 .wav
    task_id: str | None = None         # 缺省自动生成
    response_format: str = "wav"
    model: str | None = None

class AudioTaskResponse(BaseModel):    # status 字段照 LightX2V task_manager.get_task_status
    task_id: str
    status: AudioTaskStatus
    start_time: float | None = None    # 不是 created_at
    end_time: float | None = None      # 不是 completed_at
    error: str | None = None
    error_type: str = ""
    save_result_path: str | None = None
```

**save_result_path 解析(照 LightX2V `file_service.get_output_path`)**:绝对路径原样用、相对拼引擎输出根、空则用 task_id、无后缀自动补 `.wav`。生产里门面注入**容器内可见的 NFS 绝对路径**(挂载后 host==container),引擎 `os.replace` 直接写入,facade 不落盘。

文本校验:空文本 → 400;超长 → 400。**上限必须每模型不同**(IndexTTS 216-324 字杀引擎,Qwen3-TTS 960 字仍稳——统一上限就抹掉了多模型的意义),分三层:

| 层 | 职责 | 载体 |
|---|---|---|
| **new-api(权威/面向用户)** | 每模型字数上限,产品级配置,可运营调整 | `AudioModelConfig` / `common/media_model_config.go` 的 `ValidateAudioTextForModel`(方法论 §7.2,已存在) |
| **引擎(防崩兜底)** | 该实例所载模型的**安全上限**,防 IndexTTS 式杀引擎 | 每实例 env `VLLM_OMNI_AUDIO_MAX_TEXT_LEN`(GPUStack 部署时按模型设),缺省给该 model_type 的实测崩溃边界 |
| 门面 | 基础校验(空/缺参) | `routes/videos.py` |

引擎侧**不写死单一常量**:每个 GPUStack 实例 = 一个模型,`VLLM_OMNI_AUDIO_MAX_TEXT_LEN` 按该模型崩溃边界(镜像手册 §5 第 4 项实测)配;权威的用户可见上限在 new-api 按模型配。

### 4.2 FIFO 队列 + 503 背压 — 新文件 `vllm_omni/entrypoints/openai/audio_task_manager.py`

视频异步路径**无队列上限**(无界 `asyncio.create_task`),音频契约要 FIFO + 背压,净新增(小):

```python
class AudioTaskManager:
    def __init__(self, max_queue_size: int = 8):  # env VLLM_OMNI_AUDIO_MAX_QUEUE
        self._store = AUDIO_TASK_STORE
        self._tasks = AUDIO_TASKS
        self._lock = asyncio.Lock()
        self._max = max_queue_size

    async def submit(self, req) -> AudioTaskResponse:
        async with self._lock:
            active = 已 pending + processing 计数
            if active >= self._max:
                raise RuntimeError("audio task queue full")   # 照 LightX2V,路由 except RuntimeError → 503
            # 建 PENDING 记录 + 起后台 job(见 4.3)

    async def queue_status(self) -> dict:       # is_processing/pending_count/active_count/queue_size/queue_available
    async def cancel(self, task_id) -> ...      # DELETE:cancel asyncio.Task + 标 cancelled
```

- 锁:`asyncio.Lock` 保护计数与状态;`PENDING→PROCESSING` 原子转换防 DELETE 竞态。
- 取消:reuse 视频的 `asyncio.CancelledError` 分支(`_run_video_generation_job:2865-2873`)。

### 4.3 后台 job — `_run_audio_generation_job()`(加到 api_server.py,照 `_run_video_generation_job` 改)

```
1. store.update(status=PROCESSING, start_time=now)
2. speech_req = AudioTaskRequest → OpenAICreateSpeechRequest(转字段, 强制非流式)
   audio_bytes, media_type = await handler._generate_audio_bytes(speech_req, request_id=task_id)
   # ⚠️ 不是 create_speech():它返回 FastAPI Response/StreamingResponse,不是裸 bytes。
   #   _generate_audio_bytes(serving_speech.py:3740)是非流式字节路径,返回 (bytes, media_type),
   #   正是 create_speech 非流式分支内部调用的那个(serving_speech.py:4103)。
3. 原子写 audio_bytes → req.save_result_path(tmp → os.replace,复用 storage 原子逻辑)
   ⚠️ 与视频不同:视频写自定 STORAGE_MANAGER 的 storage_path/{id};
      音频写门面注入的 save_result_path(NFS 绝对路径)
4. store.update(status=COMPLETED, end_time=now)          # 契约字段是 end_time,非 completed_at
   异常 → store.update(status=FAILED, end_time=now, error, error_type)
   CancelledError → store.update(status=CANCELLED, end_time=now)
```

### 4.4 路由(照 LightX2V:common 共享 + per-kind 提交)

- **只新增一个 kind 提交路由**:`POST /v1/tasks/audio/`(P1)。
- **common 路由跨 kind 共享**(P2 的 singing/audio-gen 直接复用,不重复写):`GET /v1/tasks/{id}/status`、`GET /v1/tasks/{id}/result`、`GET /v1/tasks/queue/status`、`GET /v1/tasks/`、`DELETE /v1/tasks/{id}`。
- 均调 `AudioTaskManager`。队列满时 `create_task` 抛 `RuntimeError` → 路由 `except RuntimeError: raise HTTPException(503, ...)`(照 LightX2V `api/tasks/video.py:36-37`)。

### 4.5 `/ready` 就绪门

`/health` 只保证"引擎已初始化",不保证 warmup 完成 → 首请求会吃 torch.compile/JIT。`/ready` 应在 **模型加载 + warmup 完成后才 200**:

```python
# startup 里 warmup()(现 api_server.py:966/1098)完成后:
raw_request.app.state.server_ready = True

@router.get("/ready")
async def ready(raw_request):
    if not getattr(raw_request.app.state, "server_ready", False):
        return JSONResponse({"ready": False}, status_code=503)
    # 再顺带 check_health()(复用 /health 逻辑)
    return JSONResponse({"ready": True})
```

> 这样 GPUStack 轮到 200 再挂流量,避开冷启动首请求(Qwen3-TTS ~6min、VoxCPM2 更久,见镜像手册 §7)。**同时 registry 里 `health_check_path="/ready"` + startup 宽限调大**。

---

## 5. 引擎明确不做(方法论 §4.4)

- **不持久化任务**:重启丢任务是特性 → status 404 → 门面 sweeper 死亡重派(每次换 `-r{n}` 新 `save_result_path`)。
- **不做鉴权/多租户**:门面负责。
- **不做输入清理**:janitor 负责。
- **保留同步 `/v1/audio/speech`**:手测用。

---

## 6. 文件改动清单

| 文件(仓库根为 `vllm-omni/`) | 动作 | 内容 |
|---|---|---|
| `vllm_omni/entrypoints/openai/protocol/audio_tasks.py` | **新建** | `AudioTaskStatus/Request/Response` |
| `vllm_omni/entrypoints/openai/audio_task_manager.py` | **新建** | `AudioTaskManager`(FIFO+背压+取消);队列满抛 `RuntimeError`(照 LightX2V,路由转 503) |
| `vllm_omni/entrypoints/openai/stores.py` | 改 | 加 `AUDIO_TASK_STORE` / `AUDIO_TASKS` 实例 |
| `vllm_omni/entrypoints/openai/storage.py` | 改(小) | 抽出"原子写到任意绝对路径"辅助(供 job 写 save_result_path) |
| `vllm_omni/entrypoints/openai/serving_speech.py` | 改(极小/可无) | 复用现有 `_generate_audio_bytes`(:3740)取字节,一般无需改 |
| `vllm_omni/entrypoints/openai/api_server.py` | 改 | `_run_audio_generation_job` + `POST /v1/tasks/audio/` + common 路由 + `/ready` + startup 设 `server_ready` |
| (可选)`vllm_omni/config/server_settings.py` | 改(小) | `VLLM_OMNI_AUDIO_MAX_QUEUE` 等配置项 |

**不改**:任何 `vllm_omni/model_executor/models/*`、deploy yaml、同步 `/v1/audio/speech`、`/v1/videos` 视频路径。

---

## 7. 验证(交付判据)

1. **契约冒烟**(单机,不经 GPUStack):
   ```
   POST /v1/tasks/audio/ {text, voice, save_result_path:/nfs-.../t1.wav} → {task_id, status:pending}
   GET  /v1/tasks/{id}/status  → 轮到 completed,save_result_path 有 WAV
   GET  /v1/tasks/queue/status → 计数正确
   DELETE /v1/tasks/{id}       → 进行中任务标 cancelled
   POST 并发 > max_queue       → 503(create_task 抛 RuntimeError)
   GET  /ready                 → 加载中 503,warmup 后 200
   ```
2. **状态串**:逐字核对门面 `_ENGINE_STATE_MAP`(cancelled 双 L)。
3. **原子写**:`save_result_path` 只见完整文件,无半截(kill 中途看无残留或 tmp)。
4. **长任务**:跑一条 >1min 的(MOSS-TTSD 长对话/长文),确认轮询稳定、不超时。
5. **回归**:同步 `/v1/audio/speech` 行为不变。

---

## 8. 之后(GPUStack 侧,方法论 §6.3,本方案完成后做)

引擎异步就绪后,gpustack fork 侧照 LightX2V 模式:BackendEnum、registry(image=vllm-omni、`health_check_path=/ready`、custom_framework=cuda)、`worker/backends/vllm_omni.py`(serve `vllm serve <path> --omni` + 塞离线 env + MOSS codec env)、serve_manager 映射、整卡 count-based selector(已验证 12 个全单卡 `gpus_per_replica=1`)、evaluator 自包含跳检、门面 4 处(`tts` 已存在,大概率零改)。详见 `vLLM-Omni-全模型镜像构建与开测手册.md`。

---

## 9. 已定 / 待确认

**已定:**
- ✅ **MAX_TEXT_LEN 每模型不同**:三层(new-api 权威 per-model / 引擎每实例 env 防崩兜底 / 门面基础校验),引擎不写死单一常量。见 §4.1。
- ✅ **P2 唱段/音乐各开独立端点**(`/v1/tasks/singing/`、`/v1/tasks/audio-gen/`),不并进 `/v1/tasks/audio/`,复用同一异步基建。见 §1。

- ✅ **save_result_path**:已对 LightX2V 源码确认 —— facade 注入容器内可见的 **NFS 绝对路径**,引擎原样 `os.replace` 写入(绝对原样/相对拼根/空用 task_id/无后缀补 `.wav`),facade 不落盘。见 §4.1。
- ✅ **契约字段/状态串**:已对 LightX2V `task_manager.py` / `api/tasks/` 源码 —— 状态串 `pending/processing/completed/failed/cancelled`;status 响应 `start_time/end_time`(非 created_at/completed_at);queue 六字段;common+per-kind 路由;503 走 RuntimeError。见 §2/§4。

**落地时最后一核(以门面实读为准):**
- 门面 `gpustack/routes/videos.py` 对 status 响应的字段解析,若与 LightX2V 有出入以门面为准(理论上一致,因 LightX2V 已跑通)。
- `/v1/tasks/audio/` 请求字段名(`input`/`text`、`ref_audio`/`spk_audio_path`)对齐 new-api adaptor 音频物化的实际注入键。

---

## 10. 真机验证记录(2026-07-18,0020)

P1 已按本方案落地(commit `fcb6b9d0`)并在 **鲲鹏 ARM + A100-40G(0020)** 真机跑通全契约。

**环境**:镜像 `vllm-omni:arm64-a100-latest`(digest `4b2def0b499e`,含 P1 代码)· 模型 `Qwen3-TTS-1.7B-CustomVoice`(2 stage)· `docker run … vllm serve … --omni`。

**启动 / 就绪**:`AsyncOmniEngine initialized in 331s`(冷启动 ~5.5min:torch.compile 37.9s + CUDA graph + code_predictor warmup);启动日志确认 7 条新路由全注册(`/ready`、`POST /v1/tasks/audio/`、`GET /v1/tasks/`、`/v1/tasks/queue/status`、`/v1/tasks/{id}/status`、`/v1/tasks/{id}/result`、`DELETE /v1/tasks/{id}`);`GET /ready → 200 OK`(warmup 后,`server_ready` 生效)。

**契约冒烟**(全绿):

| 步骤 | 结果 |
|---|---|
| `POST /v1/tasks/audio/`(voice=vivian) | `{"task_id":"audio_task_83f19…","status":"pending","start_time":null,"end_time":null,"save_result_path":"/tmp/async_smoke.wav"}` |
| `GET /v1/tasks/{id}/status` 轮询 | `"status":"completed"`,`start_time=1784364100.24`,`end_time=1784364102.01`(生成 ~1.77s)—— **字段是 start_time/end_time,非 created_at/completed_at** |
| 落盘(save_result_path,原子写) | `/tmp/async_smoke.wav` **203K** |
| `GET /v1/tasks/queue/status` | `{is_processing:false, current_task:null, pending_count:0, active_count:0, queue_size:8, queue_available:8}` 六字段全对 |
| `GET /v1/tasks/{id}/result` | `RIFF WAVE audio, 16 bit, mono 24000 Hz` |

**Codex 审查三点在真机确认成立**:① job 走 `_generate_audio_bytes` 拿到真字节(203K WAV,非坏掉的 FastAPI Response);② `end_time` 正确填充;③ 包路径对(路由全注册、无导入错)。

**结论**:P1 引擎异步契约(submit→poll→落盘 NFS→取件 + queue + /ready)真机验证通过,可供 GPUStack 门面按 IndexTTS 同款异步链路接入(P3)。

---

*配套:`vLLM-Omni-全模型镜像构建与开测手册.md`(镜像/开测)、`vLLM-Omni-语音模型全景与选型.md`(选型)、`vLLM-Omni-大模型量化与Offload可行性调研.md`(不量化结论)。*
