# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI-compatible API server bootstrap for vLLM-Omni.

This module owns app construction, server startup, app-state initialization,
and route bodies that have not yet moved to endpoint-owned modules."""

import base64
import io
import asyncio
import copy
import dataclasses
import json
import multiprocessing
import multiprocessing.forkserver as forkserver
import os

# Image generation API imports
import random
import signal
import socket
import time
from argparse import Namespace
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

import uvloop
import vllm.envs as envs
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
from pydantic import ValidationError
from starlette.datastructures import State
from starlette.types import ASGIApp, Receive, Scope, Send
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.chat_utils import ChatTemplateConfig, load_chat_template
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.entrypoints.launchers.launcher import serve_http, terminate_if_errored
from vllm.entrypoints.launchers.utils.server_utils import get_uvicorn_log_config
from vllm.entrypoints.mcp.tool_server import DemoToolServer, MCPToolServer, ToolServer
from vllm.entrypoints.openai.api_server import build_app as build_openai_app
from vllm.entrypoints.openai.api_server import setup_server as setup_openai_server
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

# yapf conflicts with isort for this block
# yapf: disable
# yapf: enable
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.pooling.classify.serving import ServingClassification
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding as OpenAIServingEmbedding
from vllm.entrypoints.pooling.pooling.serving import ServingPooling
from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.engine.protocol import ErrorResponse

# vLLM moved `base` from openai.basic.api_router to serve.instrumentator.basic.
# Keep a fallback for older/newer upstream layouts during rebase windows.
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    process_lora_modules,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.orca_metrics import metrics_header
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.speech_to_text.realtime.serving import OpenAIServingRealtime
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.entrypoints.speech_to_text.translation.serving import (
    OpenAIServingTranslation,
)
from vllm.logger import init_logger
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tasks import POOLING_TASKS
from vllm.tool_parsers import ToolParserManager
from vllm.utils import random_uuid
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.config.endpoint_policy import (
    shutdown_unsupported_routes,
)
from vllm_omni.diffusion.progress import PHASE_SAVE
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.openai.audio_task_manager import (
    AUDIO_TASK_MANAGER,
    resolve_save_path,
    visible_task_status,
)
from vllm_omni.engine.stage_init_utils import set_death_signal
from vllm_omni.engine.stage_runtime import OmniClientConfig
from vllm_omni.entrypoints.async_omni import ABORT_TIMEOUT_S
from vllm_omni.entrypoints.duplex.warmup import _warmup_duplex_realtime
from vllm_omni.entrypoints.duplex_omni import DuplexOmni
from vllm_omni.entrypoints.openai import app_state as openai_app_state
from vllm_omni.entrypoints.openai.app_state import (
    ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL,
    OmniAudioGenerate,
    OmniBatchChat,
    Omnichat,
    Omnispeech,
    Omnivideo,
    _get_engine_and_model,
)
from vllm_omni.entrypoints.openai.batch_serving import OmniOpenAIServingChatBatch
from vllm_omni.entrypoints.openai.chat_template import _load_model_chat_template_json
from vllm_omni.entrypoints.openai.diffusion import (
    MAX_UINT32_SEED,
    _generate_with_async_omni,
    apply_stage_default_sampling_params,
)
from vllm_omni.entrypoints.openai.errors import (
    _create_speech_error_json_response,
    _error_response_to_json_response,
)
from vllm_omni.entrypoints.openai.image_api_utils import (
    MAX_CONDITION_RESOLUTION,
    MIN_CONDITION_RESOLUTION,
    SUPPORTED_LAYERED_RESOLUTIONS,
    encode_image_base64_with_compression,
    parse_size,
    validate_layered_layers,
)
from vllm_omni.entrypoints.openai.images.helpers import (
    _build_hunyuan_edit_extra_args,
    _check_max_generated_image_size,
    _choose_output_format,
    _extract_images_from_result,
    _generated_size_str,
    _get_max_edit_input_images,
    _load_input_images,
    _update_if_not_none,
)
from vllm_omni.entrypoints.openai.lora import _get_lora_from_json_str, _parse_lora_request
from vllm_omni.entrypoints.openai.models import serving as openai_models_serving
from vllm_omni.entrypoints.openai.protocol.audio import (
    BatchSpeechRequest,
    OpenAICreateAudioGenerateRequest,
    OpenAICreateSpeechRequest,
)
from vllm_omni.entrypoints.openai.protocol.audio_tasks import (
    AudioTaskRequest,
    AudioTaskResponse,
    AudioTaskStatus,
)
from vllm_omni.entrypoints.openai.protocol.audiogen_tasks import AudioGenTaskRequest
from vllm_omni.entrypoints.openai.protocol.image_tasks import ImageTaskRequest
from vllm_omni.entrypoints.openai.protocol.images import (
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ResponseFormat,
)
from vllm_omni.entrypoints.openai.protocol.video_tasks import VideoTaskRequest
from vllm_omni.entrypoints.openai.protocol.rollout import (
    CreateSessionRequest,
    RolloutStepRequest,
)
from vllm_omni.entrypoints.openai.protocol.videos import (
    VideoDeleteResponse,
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoListResponse,
    VideoResponse,
)
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection
from vllm_omni.entrypoints.openai.residency_runtime import ResidencyBundle, build_residency_bundle
from vllm_omni.entrypoints.openai.rollout_session import (
    RolloutSessionCapacityError,
    RolloutSessionClosedError,
    RolloutSessionNotFoundError,
)
from vllm_omni.entrypoints.openai.serving_audio_generate import OmniOpenAIServingAudioGenerate
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_rl_rollout import ServingRLRollout
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler
from vllm_omni.entrypoints.openai.serving_video import (
    LatentEditInput,
    OmniOpenAIServingVideo,
    ReferenceAudio,
    ReferenceImage,
    ReferenceVideo,
)
from vllm_omni.entrypoints.openai.serving_video_output_stream import OmniStreamingVideoOutputHandler
from vllm_omni.entrypoints.openai.serving_video_stream import create_streaming_video_handler
from vllm_omni.entrypoints.openai.stage_params import (
    clone_sampling_params,
    get_default_sampling_params_list,
)
from vllm_omni.entrypoints.openai.storage import STORAGE_MANAGER, FileStorageHandle, atomic_write_bytes
from vllm_omni.entrypoints.openai.stores import (
    AUDIO_TASK_STORE,
    AUDIO_TASKS,
    VIDEO_STORE,
    VIDEO_TASKS,
)
from vllm_omni.entrypoints.openai.utils import (
    get_stage_type,
    too_many_input_images_message,
)
from vllm_omni.entrypoints.openai.video.generation.helpers import (
    VIDEO_SYNC_TIMEOUT_S,
    _cleanup_video_references,
    _parse_video_form,
    _resolve_video_runtime_context,
    _run_video_generation_job,
    _status_code_for_video_failure,
    _unpack_video_generation_result,
    video_response_from_request,
)
from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI
from vllm_omni.entrypoints.serve.omni_control.protocol import OmniSleepRequest, OmniWakeupRequest
from vllm_omni.entrypoints.serve.profile.protocol import ProfileRequest
from vllm_omni.entrypoints.serve.profile.utils import _should_enable_profiler_endpoints
from vllm_omni.entrypoints.serve.utils.errors import (
    _create_engine_error_json_response,
    _register_omni_exception_handlers,
)
from vllm_omni.entrypoints.serve.utils.routes import (
    _remove_route_from_router,
    remove_route_from_app,
)
from vllm_omni.entrypoints.utils import PureDiffusionLauncherAdapter
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.utils.forced_aligner import build_forced_aligner_config
from vllm_omni.utils.tracking_parser import TrackingArgumentParser, TrackingNamespace

logger = init_logger(__name__)
router = APIRouter()

VIDEO_ABORT_TIMEOUT_S = ABORT_TIMEOUT_S

profiler_router = APIRouter()


# Server entry points


async def omni_run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server.

    Unified entry point that automatically handles both LLM and Diffusion models
    through AsyncOmni, which manages multi-stage pipelines.
    """
    # Suppress Pydantic serialization warnings globally for multimodal content
    # (e.g., when ChatMessage.content is a list instead of str)
    import warnings as warnings_module

    warnings_module.filterwarnings("ignore", message=".*Pydantic.*serialization.*", category=UserWarning)
    warnings_module.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*", category=UserWarning)

    # Add process-specific prefix to stdout and stderr.
    decorate_logs("APIServer", skip_if_decorated=True)

    listen_address, sock = setup_openai_server(args, reuse_port=False)

    # Unified use of omni_run_server_worker, AsyncOmni automatically handles LLM and Diffusion models
    await omni_run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


def run_omni_api_server_worker_proc(
    listen_address: str,
    sock: socket.socket,
    args: TrackingNamespace,
    client_config: dict[str, Any] | None = None,
    **uvicorn_kwargs: object,
) -> None:
    """Entrypoint used by vLLM's API server process manager."""
    set_death_signal(signal.SIGTERM)
    manager_config = client_config or {}
    client_index = int(manager_config.get("client_index", 0))
    all_client_configs = getattr(args, "_omni_stage_client_configs", None)
    if not all_client_configs or not 0 <= client_index < len(all_client_configs):
        raise RuntimeError(f"Missing Omni stage client configuration for API server {client_index}")

    omni_client_config: OmniClientConfig = copy.deepcopy(all_client_configs[client_index])
    omni_client_config["client_count"] = int(manager_config.get("client_count", 1))
    omni_client_config["client_index"] = client_index

    stage_addresses = omni_client_config["stage_addresses"]
    first_stage_id = min(stage_addresses)
    first_replica_id = min(stage_addresses[first_stage_id])
    first_addresses = stage_addresses[first_stage_id][first_replica_id]
    for key in ("input_address", "output_address", "actual_address_pipe", "tensor_queue"):
        if key in manager_config:
            first_addresses[key] = manager_config[key]

    set_process_title("APIServer", str(client_index))
    decorate_logs("APIServer", skip_if_decorated=True)
    uvloop.run(
        omni_run_server_worker(
            listen_address,
            sock,
            args,
            client_config=omni_client_config,
            **uvicorn_kwargs,
        )
    )


async def omni_run_server_worker(
    listen_address: str,
    sock: socket.socket,
    args: TrackingNamespace,
    client_config: OmniClientConfig | None = None,
    **uvicorn_kwargs: object,
) -> None:
    """Run a single API server worker."""
    api_server_count = _resolve_api_server_count(args, client_config)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)
    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        from vllm.reasoning import ReasoningParserManager

        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    # In vllm-omni's multi-process architecture, create_engine_config() runs
    # in the subprocess (StageEngineCoreProc), not in the API server process.
    # Propagate reasoning_parser from CLI args into structured_outputs_config
    # here so that OmniOpenAIServingChat receives the correct value.
    # (Upstream vLLM does this in EngineArgs.create_engine_config().)
    if hasattr(args, "reasoning_parser") and args.reasoning_parser:
        if not args.structured_outputs_config.reasoning_parser:
            args.structured_outputs_config.reasoning_parser = args.reasoning_parser

    # Load logging config for uvicorn if specified
    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_omni(
        args,
        client_config=client_config,
    ) as engine_client:
        supported_tasks: tuple[str, ...]
        if hasattr(engine_client, "get_supported_tasks"):
            supported_tasks = tuple(await engine_client.get_supported_tasks())
        else:
            supported_tasks = ("generate",)
        if not supported_tasks:
            # Only default to "generate" when get_supported_tasks is not implemented;
            # TTS-only models intentionally return an empty set.
            if not hasattr(engine_client, "get_supported_tasks"):
                supported_tasks = ("generate",)

        # OMNI: Pass supported_tasks to build_app (required by upstream vLLM)
        app = build_openai_app(args, supported_tasks)
        app.state.api_server_count = api_server_count

        # OMNI: Remove upstream routes that we override with omni-specific handlers
        remove_route_from_app(app, "/v1/chat/completions", {"POST"})
        remove_route_from_app(app, "/v1/chat/completions/batch", {"POST"})
        remove_route_from_app(app, "/v1/models", {"GET"})  # Remove upstream /v1/models to use omni's handler
        remove_route_from_app(app, "/health", {"GET"})
        app.include_router(router)

        # OMNI: Override upstream exception handlers with Omni-aware versions
        # that understand the multi-stage orchestrator lifecycle.
        _register_omni_exception_handlers(app)

        await omni_init_app_state(engine_client, app.state, args)

        # After initializing the app state, shut down any endpoints that are model specific
        if hasattr(engine_client, "endpoint_restrictions"):
            shutdown_unsupported_routes(app, engine_client.endpoint_restrictions)
        else:
            logger.warning("engine client has no endpoint restrictions attribute")

        # Start background processes
        await STORAGE_MANAGER.start()

        # Conditionally register profiler endpoints based on stage YAML configs
        stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
        if _should_enable_profiler_endpoints(stage_configs):
            logger.warning("Profiler endpoints are enabled. This should ONLY be used for local development!")
            remove_route_from_app(app, "/start_profile", frozenset({"POST"}))
            remove_route_from_app(app, "/stop_profile", frozenset({"POST"}))
            app.include_router(profiler_router)

        vllm_config = await openai_app_state._get_vllm_config(engine_client)

        # Check if pure diffusion mode (vllm_config will be None)
        is_pure_diffusion = vllm_config is None
        if is_pure_diffusion:
            logger.info(
                "Starting vLLM API server (pure diffusion mode) on %s",
                listen_address,
            )
            # The vLLM 0.27 launcher's shutdown path reads
            # engine_client.vllm_config.shutdown_timeout; AsyncOmni.vllm_config
            # is None for pure diffusion (no comprehension stage), which would
            # crash handle_shutdown and hang teardown. Wrap app.state.engine_client
            # with a shim that only overrides vllm_config and forwards everything
            # else (get_vllm_config still returns None for pure-diffusion detection).
            app.state.engine_client = PureDiffusionLauncherAdapter(
                engine_client,
                shutdown_timeout=getattr(args, "shutdown_timeout", 0),
            )
        else:
            logger.info(
                "Starting vLLM API server %d on %s",
                vllm_config.parallel_config._api_process_rank,
                listen_address,
            )

        class _TimestampMiddleware:
            """Pure-ASGI outermost wrapper that stamps HTTP request arrival time.

            Wraps the fully-built Starlette app as an outer ASGI layer so no
            Starlette internals (user_middleware, middleware_stack, etc.) are
            touched. Websocket and lifespan scopes pass through unchanged.
            """

            def __init__(self, inner: ASGIApp) -> None:
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
                if scope["type"] == "http":
                    scope.setdefault("state", {})
                    scope["state"]["request_timestamp"] = time.time()

                await self._inner(scope, receive, send)

        # Startup duplex warmup (duplex_session.warmup_frames in the deploy
        # yaml): real /v1/realtime connections wait on this event so the
        # first client never pays cold-start costs.
        duplex_warmup_frames = 0
        if getattr(app.state, "openai_serving_duplex", None) is not None:
            duplex_cfg = getattr(engine_client, "duplex_session_config", None)
            duplex_warmup_frames = int(getattr(duplex_cfg, "warmup_frames", 0) or 0)
        app.state.duplex_warmup_done = asyncio.Event() if duplex_warmup_frames > 0 else None
        warmup_task: asyncio.Task | None = None
        if duplex_warmup_frames > 0:
            # Scheduled before serve_http (which may not return until
            # shutdown); the coroutine retries its self-connect until the
            # server socket is accepting.
            warmup_task = asyncio.create_task(_warmup_duplex_realtime(app, args, duplex_warmup_frames))

        shutdown_task = await serve_http(
            _TimestampMiddleware(app),
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            ssl_ciphers=args.ssl_ciphers,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )

        try:
            await shutdown_task
        finally:
            if warmup_task is not None:
                warmup_task.cancel()
            state = getattr(app, "state", None)
            serving_video = getattr(state, "openai_serving_video", None) if state is not None else None
            if serving_video is not None:
                serving_video.shutdown()
            serving_speech = getattr(state, "openai_serving_speech", None) if state is not None else None
            if serving_speech is not None:
                serving_speech.shutdown()
            sock.close()


@asynccontextmanager
async def build_async_omni(
    args: TrackingNamespace,
    *,
    disable_frontend_multiprocessing: bool | None = None,
    client_config: OmniClientConfig | None = None,
) -> AsyncIterator[EngineClient]:
    """Build an AsyncOmni instance from command-line arguments.

    Creates an async context manager that yields an AsyncOmni instance
    configured from the provided arguments. Handles forkserver setup if
    needed and ensures proper cleanup on exit.

    Args:
        args: Parsed command-line arguments containing model and configuration
        disable_frontend_multiprocessing: Optional flag to disable frontend
            multiprocessing
        client_config: Optional client configuration dictionary

    Yields:
        EngineClient instance (AsyncOmni) ready for use
    """
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        # The executor is expected to be mp.
        # Pre-import heavy modules in the forkserver process
        logger.debug("Setup forkserver with pre-imports")
        multiprocessing.set_start_method("forkserver")
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
        logger.debug("Forkserver setup complete!")

    # Context manager to handle async_omni lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    async with build_async_omni_from_stage_config(
        args,
        disable_frontend_multiprocessing=disable_frontend_multiprocessing,
        client_config=client_config,
    ) as async_omni:
        yield async_omni


def _should_serve_duplex(model: str, kwargs: dict[str, Any]) -> bool:
    """Select the serving engine without changing the model's duplex capability."""
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.config.stage_config import _DEPLOY_DIR, resolve_deploy_yaml

    pipeline_config = StageConfigFactory.get_pipeline_config(
        model=model,
        trust_remote_code=bool(kwargs.get("trust_remote_code")),
        deploy_config_path=kwargs.get("deploy_config"),
    )
    if pipeline_config is None or not getattr(pipeline_config, "duplex_plugin", None):
        return False

    deploy_path = kwargs.get("deploy_config")
    if deploy_path is None:
        if pipeline_config.default_deploy_config_name is None:
            raise ValueError("A duplex-capable model requires a deploy config with session_mode: turn or duplex")
        deploy_path = _DEPLOY_DIR / pipeline_config.default_deploy_config_name
    else:
        deploy_path = Path(deploy_path)
        if not deploy_path.exists() and deploy_path.parent == Path("."):
            deploy_path = _DEPLOY_DIR / deploy_path

    # Resolve base_config too, so the API and stage workers use the same mode.
    session_mode = resolve_deploy_yaml(deploy_path).get(
        "session_mode", getattr(pipeline_config, "default_session_mode", None)
    )
    if session_mode not in ("turn", "duplex"):
        raise ValueError("A duplex-capable model requires session_mode: turn or duplex in its deploy config")
    return session_mode == "duplex"


@asynccontextmanager
async def build_async_omni_from_stage_config(
    args: TrackingNamespace,
    *,
    disable_frontend_multiprocessing: bool = False,
    client_config: OmniClientConfig | None = None,
) -> AsyncIterator[EngineClient]:
    """Create AsyncOmni from stage configuration.

    Creates an AsyncOmni instance either in-process or using multiprocess
    RPC. Loads stage configurations from the model or from a specified path.

    Args:
        args: Parsed command-line arguments containing model and stage configs
        disable_frontend_multiprocessing: Flag to disable frontend multiprocessing
            for compatibility with existing CLI options
        client_config: Optional client configuration dictionary

    Yields:
        EngineClient instance (AsyncOmni) ready for use

    Note:
        Stage configurations are loaded from ``args.deploy_config`` when provided,
        otherwise from the model's default configuration.
    """

    if disable_frontend_multiprocessing:
        logger.warning("Ignoring --disable-frontend-multiprocessing for AsyncOmni runtime.")

    async_omni: EngineClient | None = None

    # Pre-load the model config so HuggingFace registers `transformers_modules`
    # in this process — only when the user has explicitly opted in via
    # `--trust-remote-code`. Stage workers consume the same flag through their
    # deploy config, but the API server process also needs the dynamic modules
    # for ZMQ pickle deserialization of stage outputs that reference them
    # (e.g. trust_remote_code models like MiniCPM-o).
    if getattr(args, "trust_remote_code", False) and getattr(args, "model", None):
        try:
            import os

            from transformers import AutoConfig

            # Hide GPUs so the custom config code doesn't allocate CUDA memory.
            saved = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            try:
                AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            finally:
                if saved is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = saved
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        except Exception as e:
            logger.debug("Pre-loading transformers_modules failed: %s", e)

    residency_bundle: ResidencyBundle | None = None
    try:
        kwargs = args.get_explicit_kwargs_dict()
        # This controls only the API-process WebSocket and is not an AsyncOmni option.
        kwargs.pop("robot_openpi_idle_timeout", None)
        model = kwargs.pop("model", None) or args.model
        kwargs.setdefault("log_stats", not args.disable_log_stats)
        if client_config is not None:
            kwargs["client_config"] = client_config

        # Server-side only: consumed here to decide the engine topology, and NOT
        # an AsyncOmni kwarg — leaving it in would hand the engine an argument it
        # does not accept.
        residency_path = kwargs.pop("residency_config", None) or getattr(args, "residency_config", None)
        if residency_path:
            if kwargs.get("deploy_config"):
                raise ValueError(
                    "--residency-config and --deploy-config are mutually exclusive: the residency "
                    "file names one deploy config per co-located engine."
                )
            residency_bundle = await build_residency_bundle(model=model, base_kwargs=kwargs, path=residency_path)
            async_omni = residency_bundle.primary
            # omni_init_app_state() receives only the primary engine, so ride
            # along on it to reach app state without changing that signature.
            async_omni._omni_residency_bundle = residency_bundle
        elif _should_serve_duplex(model, kwargs):
            async_omni = DuplexOmni(model=model, **kwargs)
        else:
            async_omni = AsyncOmni(model=model, **kwargs)

        # # Don't keep the dummy data in memory
        # await async_llm.reset_mm_cache()

        yield async_omni
    finally:
        if residency_bundle is not None:
            residency_bundle.shutdown()
        elif async_omni:
            async_omni.shutdown()


async def _init_duplex_app_state(
    engine_client: DuplexOmni,
    state: State,
    args: Namespace,
    base_model_paths: list[BaseModelPath],
    vllm_config: Any,
    request_logger: RequestLogger | None,
) -> None:
    """Minimal app state for a duplex server: only the surfaces a duplex session backs."""
    state.vllm_config = vllm_config
    state.diffusion_engine = None
    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,  # type: ignore[arg-type]
        base_model_paths=base_model_paths,
        lora_modules=None,
    )
    state.serving_tokenization = None
    state.serving_tokens = None
    # Replaced by the chat init below when the model serves chat.
    state.online_renderer = None
    for attribute in (
        "openai_serving_chat_batch",
        "openai_serving_completion",
        "openai_serving_responses",
        "openai_serving_embedding",
        "openai_serving_pooling",
        "openai_serving_classification",
        "openai_serving_scores",
        "openai_serving_transcription",
        "openai_serving_translation",
        "openai_serving_speech",
        "openai_serving_audio_generate",
        "openai_serving_video",
        "openai_streaming_speech",
        "openai_streaming_video",
        "openai_streaming_video_output",
        "openai_serving_realtime",
        "openai_serving_realtime_robot",
        "anthropic_serving_messages",
    ):
        setattr(state, attribute, None)
    state.openai_serving_duplex = OmniDuplexSessionHandler(duplex_omni=engine_client)
    # One engine, both surfaces. ``DuplexOmni`` extends ``AsyncOmni``, so the
    # ordinary chat service runs on it unchanged: a chat request is a turn-based
    # request on the same pipeline, not a session, and costs no admission slot.
    state.openai_serving_chat = await _init_duplex_chat(engine_client, state, args, request_logger)
    state.enable_server_load_tracking = getattr(args, "enable_server_load_tracking", False)
    state.server_load_metrics = 0
    if state.openai_serving_chat is not None:
        logger.info(
            "Duplex mode: serving %s over /v1/realtime?duplex=1 and /v1/chat/completions",
            engine_client.model,
        )
    else:
        logger.info(
            "Duplex mode: serving %s over /v1/realtime?duplex=1 only "
            "(/v1/chat/completions unavailable: the model does not declare supports_chat_completions)",
            engine_client.model,
        )


async def _init_duplex_chat(
    engine_client: DuplexOmni,
    state: State,
    args: Namespace,
    request_logger: RequestLogger | None,
) -> OmniOpenAIServingChat | None:
    """The ordinary chat service, on a duplex engine, when the model allows it.

    Gated on ``DuplexCapabilities.supports_chat_completions`` so the decision
    stays the model's: a duplex model that should not answer chat requests says
    so in its plugin, and the route reports "not available" rather than
    answering badly. ``endpoint_restrictions`` remains the per-deployment
    opt-out on top of this.
    """
    if not engine_client.duplex_capabilities.supports_chat_completions:
        return None
    supported_tasks: set[str] = {"generate"}
    if hasattr(engine_client, "get_supported_tasks"):
        supported_tasks = set(await engine_client.get_supported_tasks())
    if "generate" not in supported_tasks:
        return None

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is None:
        try:
            tokenizer = await engine_client.get_tokenizer()
        except Exception as exc:
            logger.debug("Could not inspect tokenizer chat_template before duplex chat init: %s", exc)
            tokenizer = None
        if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
            resolved_chat_template = _load_model_chat_template_json(args.model)

    state.online_renderer = OnlineRenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
    )
    return OmniOpenAIServingChat(
        engine_client=engine_client,
        models=state.openai_serving_models,
        response_role=args.response_role,
        online_renderer=state.online_renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_force_include_usage=args.enable_force_include_usage,
        enable_log_outputs=args.enable_log_outputs,
        enable_log_deltas=args.enable_log_deltas,
    )


async def omni_init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
) -> None:
    """Initialize the FastAPI application state for omni API server.

    Sets up the application state with model information, request logger,
    and other server configuration needed for handling API requests.
    Automatically detects pure diffusion mode (single diffusion stage) and
    handles it appropriately.

    Args:
        engine_client: Engine client instance (AsyncOmni)
        state: FastAPI application state object to initialize
        args: Parsed command-line arguments
    """
    # Get vllm_config from engine_client (following 0.14.0 pattern)
    vllm_config = await openai_app_state._get_vllm_config(engine_client)

    # Detect if it's pure Diffusion mode (single stage and is Diffusion)
    is_pure_diffusion = False
    if hasattr(engine_client, "stage_configs") and engine_client.stage_configs:
        stage_configs = engine_client.stage_configs
        if len(stage_configs) == 1:
            stage_type = get_stage_type(stage_configs[0])
            if stage_type == "diffusion":
                is_pure_diffusion = True
                logger.info("Detected pure diffusion mode (single diffusion stage)")

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    base_model_paths = [BaseModelPath(name=name, model_path=args.model) for name in served_model_names]
    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.args = args
    state.sleeping_stages = set()
    # Present only under --residency-config; None keeps every single-engine
    # deployment on exactly the previous code path.
    state.residency_bundle = getattr(engine_client, "_omni_residency_bundle", None)

    # For omni models
    state.stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
    model_name = served_model_names[0] if served_model_names else args.model

    # Initialize the API surface for the selected engine, not just the model.
    if isinstance(engine_client, DuplexOmni):
        await _init_duplex_app_state(engine_client, state, args, base_model_paths, vllm_config, request_logger)
        return

    # Pure Diffusion mode: use simplified initialization logic
    if is_pure_diffusion:
        state.vllm_config = None
        state.diffusion_engine = engine_client
        state.openai_serving_models = openai_models_serving._DiffusionServingModels(base_model_paths)
        # OMNI: tokenization endpoints are not supported in pure diffusion mode.
        state.serving_tokenization = None

        # Use for_diffusion method to create chat handler
        state.openai_serving_chat = OmniOpenAIServingChat.for_diffusion(
            diffusion_engine=engine_client,  # type: ignore
            model_name=model_name,
        )
        state.openai_serving_chat_batch = OmniOpenAIServingChatBatch.for_diffusion(
            diffusion_engine=engine_client,  # type: ignore
            model_name=model_name,
        )

        # audio related
        state.openai_serving_speech = None
        state.openai_serving_audio_generate = OmniOpenAIServingAudioGenerate.for_diffusion(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            model_name=model_name,
        )

        # video related
        diffusion_stage_configs = engine_client.stage_configs if hasattr(engine_client, "stage_configs") else None
        state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
            diffusion_engine=engine_client,  # type: ignore
            model_name=model_name,
            stage_configs=diffusion_stage_configs,
        )
        state.openai_streaming_video_output = OmniStreamingVideoOutputHandler(
            engine_client=engine_client,
            model_name=model_name,
            stage_configs=diffusion_stage_configs,
        )

        state.openai_serving_speech = OmniOpenAIServingSpeech.for_diffusion(
            diffusion_engine=engine_client,
            model_name=model_name,
            stage_configs=diffusion_stage_configs,
            allowed_local_media_path=getattr(args, "allowed_local_media_path", ""),
            allowed_media_domains=getattr(args, "allowed_media_domains", None),
        )
        state.openai_serving_duplex = None
        state.openai_streaming_speech = None
        state.openai_streaming_video = None
        state.openai_serving_realtime_robot = ServingRealtimeRobotOpenPI.create_policy_server(
            engine_client=engine_client,
            model_name=model_name,
        )
        if state.openai_serving_realtime_robot is not None:
            state.rl_rollout_serving = ServingRLRollout(state.openai_serving_realtime_robot)
        else:
            state.rl_rollout_serving = None

        state.enable_server_load_tracking = getattr(args, "enable_server_load_tracking", False)
        state.server_load_metrics = 0
        # Flip the readiness gate consumed by GPUStack's /ready probe. The speech/LLM
        # path sets state.server_ready after warmup (~L1114); this pure-diffusion
        # branch returns early and never reached it, so /ready stayed 503 forever and
        # GPUStack never marked diffusion instances (AudioX / SoulX-Singer /
        # Stable-Audio) Ready. The model is loaded + warmed by this point, so it is
        # safe to advertise readiness here.
        state.server_ready = True
        logger.info("Pure diffusion API server initialized for model: %s", model_name)
        return

    # LLM or multi-stage mode: use standard initialization logic
    if vllm_config is None:
        # Try to get vllm_config from engine_client
        vllm_config = await openai_app_state._get_vllm_config(engine_client)
        if vllm_config is None:
            logger.warning("vllm_config is None, some features may not work correctly")

    state.vllm_config = vllm_config

    # Get supported tasks
    supported_tasks: set[str] = {"generate"}
    if hasattr(engine_client, "get_supported_tasks"):
        supported_tasks = set(await engine_client.get_supported_tasks())
    logger.info("Supported tasks: %s", supported_tasks)

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is None:
        try:
            tokenizer = await engine_client.get_tokenizer()
        except Exception as exc:
            logger.debug("Could not inspect tokenizer chat_template before serving init: %s", exc)
            tokenizer = None
        if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
            resolved_chat_template = _load_model_chat_template_json(args.model)

    chat_template_config = ChatTemplateConfig(
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
    )

    if args.tool_server == "demo":
        tool_server: ToolServer | None = DemoToolServer()
        assert isinstance(tool_server, DemoToolServer)
        await tool_server.init_and_validate()
    elif args.tool_server:
        tool_server = MCPToolServer()
        await tool_server.add_tool_server(args.tool_server)
    else:
        tool_server = None

    # Merge default_mm_loras into the static lora_modules
    default_mm_loras = (
        vllm_config.lora_config.default_mm_loras
        if vllm_config is not None and vllm_config.lora_config is not None
        else {}
    )
    lora_modules = process_lora_modules(args.lora_modules, default_mm_loras)

    # Ensure `input_processor` and `model_config` exist on the engine client
    # for OpenAIServingModels compatibility.
    #
    # vLLM 0.20 dropped the `io_processor` kwarg from OpenAIServingRender and
    # neither `vllm.entrypoints.openai.*` nor `vllm.entrypoints.serve.*` reads
    # `engine_client.io_processor` anymore, so we no longer need to back-fill
    # it here. AsyncOmni still sets `self.io_processor` in its own __init__
    # for any vllm-omni internal callers that rely on it.
    if (
        not hasattr(engine_client, "input_processor")
        or engine_client.input_processor is None
        or not hasattr(engine_client, "model_config")
        or engine_client.model_config is None
    ):
        if vllm_config is not None:
            try:
                from vllm.v1.engine.input_processor import InputProcessor

                tokenizer = await engine_client.get_tokenizer()
                if tokenizer is not None:
                    if not hasattr(engine_client, "input_processor") or engine_client.input_processor is None:
                        engine_client.input_processor = InputProcessor(
                            vllm_config=vllm_config,
                        )
                        logger.info("Initialized input_processor for AsyncOmni")

                    if not hasattr(engine_client, "model_config") or engine_client.model_config is None:
                        engine_client.model_config = vllm_config.model_config
                        logger.info("Initialized model_config for AsyncOmni")
                else:
                    logger.warning("Cannot initialize processors: tokenizer is None. OpenAIServingModels may fail.")
            except Exception as e:
                logger.warning(
                    "Failed to initialize processors for AsyncOmni: %s. OpenAIServingModels may fail.",
                    e,
                )
        else:
            logger.warning("Cannot initialize processors: vllm_config is None. OpenAIServingModels may fail.")

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=lora_modules,
    )
    await state.openai_serving_models.init_static_loras()

    # NOTE: kept aligned with upstream `init_app_state`:
    # Use OnlineRenderer (replaced OpenAIServingRender which was removed upstream).
    state.online_renderer = OnlineRenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        log_error_stack=args.log_error_stack,
    )

    state.openai_serving_responses = (
        OpenAIServingResponses(
            engine_client,
            state.openai_serving_models,
            state.online_renderer,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            tool_server=tool_server,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_log_outputs=args.enable_log_outputs,
        )
        if "generate" in supported_tasks
        else None
    )

    _chat_kwargs = dict(
        engine_client=engine_client,
        models=state.openai_serving_models,
        response_role=args.response_role,
        online_renderer=state.online_renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_force_include_usage=args.enable_force_include_usage,
        enable_log_outputs=args.enable_log_outputs,
        enable_log_deltas=args.enable_log_deltas,
    )

    state.openai_serving_chat = OmniOpenAIServingChat(**_chat_kwargs) if "generate" in supported_tasks else None
    state.openai_serving_chat_batch = (
        OmniOpenAIServingChatBatch(**_chat_kwargs) if "generate" in supported_tasks else None
    )

    # Warm up chat template processing to avoid first-request latency
    # Upstream f5ffc59b6a removed OpenAIServingChat.warmup() and moved the
    # warmup onto the renderer (OnlineRenderer.warmup()); mirror upstream.
    state.online_renderer.warmup()

    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            state.openai_serving_models,
            online_renderer=state.online_renderer,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "generate" in supported_tasks
        else None
    )
    state.openai_serving_pooling = (
        ServingPooling(
            engine_client,
            state.openai_serving_models,
            supported_tasks=tuple(supported_tasks),
            request_logger=request_logger,
            chat_template_config=chat_template_config,
        )
        if any(task in POOLING_TASKS for task in supported_tasks)
        else None
    )
    state.openai_serving_embedding = (
        OpenAIServingEmbedding(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template_config=chat_template_config,
        )
        if "embed" in supported_tasks
        else None
    )
    state.openai_serving_classification = (
        ServingClassification(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template_config=chat_template_config,
        )
        if "classify" in supported_tasks
        else None
    )
    state.openai_serving_scores = (
        ServingScores(
            engine_client,
            state.openai_serving_models,
            supported_tasks=tuple(supported_tasks),
            request_logger=request_logger,
            chat_template_config=chat_template_config,
            log_error_stack=args.log_error_stack,
        )
        if any(t in supported_tasks for t in ("embed", "score", "token_embed"))
        else None
    )
    state.serving_tokenization = ServingTokenization(
        state.openai_serving_models,
        state.online_renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
    )
    state.openai_serving_transcription = (
        OpenAIServingTranscription(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )
    state.openai_serving_translation = (
        OpenAIServingTranslation(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )
    state.anthropic_serving_messages = (
        AnthropicServingMessages(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            online_renderer=state.online_renderer,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            default_chat_template_kwargs=args.default_chat_template_kwargs,
        )
        if "generate" in supported_tasks
        else None
    )
    state.serving_tokens = (
        ServingTokens(
            engine_client,
            state.openai_serving_models,
            state.online_renderer,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_log_outputs=args.enable_log_outputs,
            force_no_detokenize=args.tokens_only,
        )
        if "generate" in supported_tasks
        else None
    )

    state.openai_serving_speech = OmniOpenAIServingSpeech(
        engine_client,
        state.openai_serving_models,
        request_logger=request_logger,
        model_name=model_name,
        forced_aligner_enabled=build_forced_aligner_config(
            getattr(args, "forced_aligner", None),
            getattr(args, "forced_aligner_config", None),
        )
        is not None,
    )

    # Warm up speech pipeline (CUDA Graph capture, torch.compile) so the first
    # real user request is fast instead of paying a 100s compilation tax.
    await state.openai_serving_speech.warmup()

    # Speech pipeline is warm: flip the readiness gate consumed by /ready so
    # GPUStack only routes traffic (and marks the instance Ready) after warmup,
    # never during the multi-minute cold start.
    state.server_ready = True

    state.openai_serving_audio_generate = OmniOpenAIServingAudioGenerate(
        engine_client, state.openai_serving_models, request_logger=request_logger, model_name=model_name
    )

    state.openai_streaming_speech = OmniStreamingSpeechHandler(
        speech_service=state.openai_serving_speech,
    )
    state.openai_streaming_video = (
        create_streaming_video_handler(
            chat_service=state.openai_serving_chat,
            engine_client=engine_client,
        )
        if state.openai_serving_chat is not None
        else None
    )
    state.openai_serving_duplex = None
    state.openai_serving_realtime = OpenAIServingRealtime(
        engine_client=engine_client,
        models=state.openai_serving_models,
        request_logger=request_logger,
    )

    state.openai_serving_video = OmniOpenAIServingVideo(
        engine_client,
        model_name=served_model_names[0] if served_model_names else None,
        stage_configs=state.stage_configs,
    )
    state.openai_serving_realtime_robot = None
    state.rl_rollout_serving = None

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


async def _run_with_terminal_engine_awake(app_state: Any, *, request_id: str, run: Any) -> Any:
    """Await ``run()`` with the terminal engine awake under residency.

    A plain passthrough when residency is off.

    STREAMING is why this cannot be a bare ``async with``: a streaming call
    returns an async iterator BEFORE any generation happens, and the engine work
    occurs while the client consumes it. Releasing the session when the call
    returns would sleep the engine mid-stream — worse than never wrapping. So the
    session is held and handed to a wrapper iterator that closes it on
    exhaustion, client disconnect (``GeneratorExit``) or error, which means an
    abandoned stream cannot strand the engine awake either.

    Every call site that may stream must go through this rather than the bare
    context manager.
    """
    bundle = getattr(app_state, "residency_bundle", None)
    if bundle is None:
        return await run()

    stack = AsyncExitStack()
    await stack.__aenter__()
    try:
        await stack.enter_async_context(_terminal_engine_awake(app_state, request_id=request_id))
        result = await run()
    except BaseException:
        await stack.aclose()
        raise

    if not isinstance(result, AsyncGenerator) and not (
        hasattr(result, "__aiter__") and not isinstance(result, ErrorResponse)
    ):
        # Non-streaming: generation already finished, release immediately.
        await stack.aclose()
        return result

    async def _streaming_with_residency() -> AsyncGenerator:
        try:
            async for chunk in result:
                yield chunk
        finally:
            await stack.aclose()

    return _streaming_with_residency()


async def _chat_completion_with_residency(handler: Any, request: Any, raw_request: Request) -> Any:
    """Chat completion with residency-aware (stream-safe) engine wake."""
    return await _run_with_terminal_engine_awake(
        raw_request.app.state,
        request_id=f"chat-{random_uuid()}",
        run=lambda: handler.create_chat_completion(request, raw_request),
    )


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, "")
    handler = Omnichat(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Chat Completions API",
            )
        return base_server.create_error_response(message="The model does not support Chat Completions API")
    try:
        generator = await _chat_completion_with_residency(handler, request, raw_request)
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as e:
        logger.exception("Chat completion failed: %s", e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(),
            status_code=generator.error.code if generator.error else 400,
        )

    elif isinstance(generator, ChatCompletionResponse):
        # Completely bypass Pydantic serialization warnings for multimodal content
        # by converting to dict first, then serializing with warnings suppressed
        import json as json_lib
        import warnings as warnings_module

        # Temporarily suppress ALL Pydantic UserWarnings during serialization
        with warnings_module.catch_warnings():
            warnings_module.filterwarnings("ignore", category=UserWarning)
            warnings_module.filterwarnings("ignore", message=".*Pydantic.*", category=UserWarning)
            try:
                # Use serialize_as_any=True to bypass type checking
                response_dict = generator.model_dump(mode="json", serialize_as_any=True, warnings="none")
                return JSONResponse(
                    content=response_dict,
                    headers=metrics_header(metrics_header_format),
                )
            except Exception:
                # Fallback: convert to JSON string and parse back to avoid any serialization issues
                try:
                    response_json = generator.model_dump_json(warnings="none", serialize_as_any=True)
                    response_dict = json_lib.loads(response_json)
                    return JSONResponse(
                        content=response_dict,
                        headers=metrics_header(metrics_header_format),
                    )
                except Exception:
                    # Last resort: regular dump with warnings suppressed
                    with warnings_module.catch_warnings():
                        warnings_module.filterwarnings("ignore", category=UserWarning)
                        return JSONResponse(
                            content=generator.model_dump(mode="json", warnings="none"),
                            headers=metrics_header(metrics_header_format),
                        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/chat/completions/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_batch_chat_completion(request: BatchChatCompletionRequest, raw_request: Request):
    handler = OmniBatchChat(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Chat Completions API",
            )
        return base_server.create_error_response(message="The model does not support Chat Completions API")
    try:
        result = await handler.create_batch_chat_completion(request, raw_request)
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as e:
        logger.exception("Batched chat completion failed: %s", e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e

    if isinstance(result, ErrorResponse):
        return JSONResponse(
            content=result.model_dump(),
            status_code=result.error.code if result.error else 400,
        )
    return JSONResponse(content=result.model_dump(mode="json"))


_remove_route_from_router(router, "/v1/audio/speech", {"POST"})


@router.post(
    "/v1/audio/speech",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"audio/*": {}, "text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_speech(request: OpenAICreateSpeechRequest, raw_request: Request):
    """Generate speech audio from text using the loaded TTS model.

    Args:
        request: Speech synthesis request in OpenAI-compatible format.
        raw_request: Raw FastAPI request for accessing app state.

    Returns:
        The generated audio response, or an OpenAI-style error payload when
        the request cannot be fulfilled.

    Raises:
        HTTPException: If the server does not support speech generation or the
        synthesis request fails unexpectedly.
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Speech API",
            )
        err = base_server.create_error_response(
            message="The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )
        return _error_response_to_json_response(err, status_code=HTTPStatus.NOT_FOUND)
    try:
        result = await handler.create_speech(request, raw_request)
        if isinstance(result, ErrorResponse):
            return _error_response_to_json_response(result)
        return result
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


@router.post(
    "/v1/audio/speech/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_speech_batch(request: BatchSpeechRequest, raw_request: Request):
    handler = Omnispeech(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Speech API",
            )
        err = base_server.create_error_response(
            message="The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )
        return _error_response_to_json_response(err, status_code=HTTPStatus.NOT_FOUND)
    try:
        result = await handler.create_speech_batch(request)
        if isinstance(result, ErrorResponse):
            return _error_response_to_json_response(result)
        # exclude_none so optional per-item fields are omitted rather than
        # serialized as null: errored items drop `usage`/`audio_data`/`media_type`,
        # successful items drop `error`. Matches the documented batch response shape.
        return JSONResponse(content=result.model_dump(exclude_none=True))
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


@router.post(
    "/v1/audio/generate",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"audio/*": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_audio_generate(request: OpenAICreateAudioGenerateRequest, raw_request: Request):
    handler = OmniAudioGenerate(raw_request)
    if handler is None:
        base_server = getattr(raw_request.app.state, "serving_tokenization", None)
        if base_server is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="The model does not support Audio Generate API",
            )
        err = base_server.create_error_response(
            message="The model does not support Audio Generate API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )
        return _error_response_to_json_response(err, status_code=HTTPStatus.NOT_FOUND)
    try:
        result = await handler.create_audio_generate(request, raw_request)
        if isinstance(result, ErrorResponse):
            return _error_response_to_json_response(result)
        return result
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


@router.get(
    "/v1/audio/voices",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def list_voices(raw_request: Request):
    """List available TTS voices exposed by the loaded speech model.

    Args:
        raw_request: Raw FastAPI request for accessing app state.

    Returns:
        A JSON payload containing the sorted set of supported speaker names, or
        an OpenAI-style error response when the current server configuration
        does not support the Speech API.
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        return _create_speech_error_json_response(
            raw_request,
            "The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    speakers = sorted(handler._get_available_voices())

    # Get uploaded speakers details
    uploaded_speakers = []
    if hasattr(handler, "uploaded_speakers"):
        for voice_name, info in handler.uploaded_speakers.items():
            voice_entry = {
                "name": info.get("name", voice_name),
                "consent": info.get("consent", ""),
                "created_at": info.get("created_at", 0),
                "file_size": info.get("file_size", 0),
                "mime_type": info.get("mime_type", ""),
                "embedding_source": info.get("embedding_source", "audio"),
                "embedding_dim": info.get("embedding_dim"),
            }
            if info.get("ref_text"):
                voice_entry["ref_text"] = info["ref_text"]
            if info.get("speaker_description"):
                voice_entry["speaker_description"] = info["speaker_description"]
            uploaded_speakers.append(voice_entry)

    return JSONResponse(content={"voices": speakers, "uploaded_voices": uploaded_speakers})


def _resolve_api_server_count(args: Namespace, client_config: OmniClientConfig | None) -> int:
    """Resolve frontend topology once, from the launcher's client configuration."""
    requested = getattr(args, "api_server_count", None)
    actual = 1 if client_config is None else client_config.get("client_count")
    if type(actual) is not int or actual < 1:
        raise ValueError("API client configuration requires a positive integer client_count")
    if requested is not None and (type(requested) is not int or requested < 1 or requested != actual):
        raise ValueError("API server count does not match the frontend launch configuration")
    return actual


def _reject_process_local_state_with_multiple_api_workers(raw_request: Request, operation: str) -> None:
    """Reject access to frontend state that is not shared across API workers."""
    api_server_count = getattr(raw_request.app.state, "api_server_count", None)
    if type(api_server_count) is not int or api_server_count < 1:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="API worker topology is not initialized.",
        )
    if api_server_count > 1:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value,
            detail=(
                f"{operation} is not supported with --api-server-count > 1 because "
                "the operation uses process-local frontend state"
            ),
        )


async def _require_single_api_video_store(raw_request: Request) -> None:
    # VIDEO_STORE and VIDEO_TASKS are process-local, including reads and
    # cancellation. Keep this independent of the temporary diffusion launch
    # restriction: enabling multi-API diffusion alone cannot make jobs shared.
    _reject_process_local_state_with_multiple_api_workers(raw_request, "Asynchronous video jobs")


@router.post(
    "/v1/audio/voices",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def upload_voice(
    raw_request: Request,
    audio_sample: UploadFile | None = File(None),
    speaker_embedding: str | None = Form(None),
    consent: str = Form(...),
    name: str = Form(...),
    ref_text: str | None = Form(None),
    speaker_description: str | None = Form(None),
):
    """Upload a new voice for voice cloning.

    Accepts either an audio file or a pre-computed speaker embedding vector.
    These are mutually exclusive: provide one or the other.

    When using ``audio_sample``, the server stores the audio and extracts the
    speaker embedding on first use (Base task models only).

    When using ``speaker_embedding``, pass a JSON-encoded list of floats
    (1024-dim for 0.6B, 2048-dim for 1.7B). The voice is stored as a
    safetensors file and is immediately ready for use.

    Args:
        audio_sample: Audio file (max 10MB). Mutually exclusive with speaker_embedding.
        speaker_embedding: JSON-encoded float list. Mutually exclusive with audio_sample.
        consent: Consent recording ID
        name: Name for the new voice
        ref_text: Optional transcript of the audio for ICL (in-context
            learning) mode. When provided, voice clone requests using this
            voice will produce higher quality results.
        speaker_description: Optional free-form description of the voice
            (e.g. "warm speaker", "energetic narrator").
        raw_request: Raw FastAPI request

    Returns:
        JSON response with voice information
    """
    _reject_process_local_state_with_multiple_api_workers(raw_request, "Runtime voice upload")
    handler = Omnispeech(raw_request)
    if handler is None:
        return _create_speech_error_json_response(
            raw_request,
            "The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    try:
        if speaker_embedding is not None and audio_sample is not None:
            return _create_speech_error_json_response(
                raw_request, "'audio_sample' and 'speaker_embedding' are mutually exclusive"
            )
        if speaker_embedding is not None:
            result = await handler.upload_voice_embedding(speaker_embedding, consent, name)
        elif audio_sample is not None:
            result = await handler.upload_voice(
                audio_sample,
                consent,
                name,
                ref_text=ref_text,
                speaker_description=speaker_description,
            )
        else:
            return _create_speech_error_json_response(
                raw_request, "Either 'audio_sample' or 'speaker_embedding' must be provided"
            )

        return JSONResponse(content={"success": True, "voice": result})

    except ValueError as e:
        return _create_speech_error_json_response(raw_request, str(e))
    except Exception as e:
        logger.exception(f"Failed to upload voice: {e}")
        return _create_speech_error_json_response(
            raw_request,
            f"Failed to upload voice: {str(e)}",
            err_type="InternalServerError",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@router.delete(
    "/v1/audio/voices/{name}",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def delete_voice(name: str, raw_request: Request):
    """Delete an uploaded voice.

    Deletes the voice sample and associated metadata. Also removes any
    cached voice clone prompts for this voice.

    Args:
        name: Name of the voice to delete
        raw_request: Raw FastAPI request

    Returns:
        JSON response indicating success or failure
    """
    _reject_process_local_state_with_multiple_api_workers(raw_request, "Runtime voice deletion")
    handler = Omnispeech(raw_request)
    if handler is None:
        return _create_speech_error_json_response(
            raw_request,
            "The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    try:
        # Delete the voice
        success = await handler.delete_voice(name)
        if not success:
            return _create_speech_error_json_response(
                raw_request,
                f"Voice '{name}' not found",
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND,
            )

        return JSONResponse(content={"success": True, "message": f"Voice '{name}' deleted successfully"})

    except ValueError as e:
        return _create_speech_error_json_response(raw_request, str(e))
    except Exception as e:
        logger.exception(f"Failed to delete voice '{name}': {e}")
        return _create_speech_error_json_response(
            raw_request,
            f"Failed to delete voice: {str(e)}",
            err_type="InternalServerError",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@router.websocket("/v1/audio/speech/stream")
async def streaming_speech(websocket: WebSocket):
    """WebSocket endpoint for streaming text input TTS.

    Accepts text incrementally and returns audio for the buffered text on
    input.done, which flushes without closing the connection. See
    serving_speech_stream.py for protocol.
    """
    handler = getattr(websocket.app.state, "openai_streaming_speech", None)
    if handler is None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": "Streaming speech is not available",
            }
        )
        await websocket.close()
        return
    await handler.handle_session(websocket)


@router.websocket("/v1/video/chat/stream")
async def streaming_video_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming video input chat."""
    handler = getattr(websocket.app.state, "openai_streaming_video", None)
    if handler is None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": "Streaming video chat is not available",
            }
        )
        await websocket.close()
        return
    await handler.handle_session(websocket)


@router.websocket("/v1/realtime/video")
async def streaming_video_output(websocket: WebSocket):
    """WebSocket endpoint for streaming generated video output chunks."""
    handler = getattr(websocket.app.state, "openai_streaming_video_output", None)
    if handler is None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": "Streaming video generation is not available",
            }
        )
        await websocket.close()
        return
    await handler.handle_session(websocket)


@router.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket):
    """WebSocket endpoint for OpenAI-style realtime interactions."""
    duplex_handler = getattr(websocket.app.state, "openai_serving_duplex", None)
    duplex_query = websocket.query_params.get("duplex")
    use_duplex_realtime = duplex_handler is not None and (
        duplex_query is None or (isinstance(duplex_query, str) and duplex_query.lower() in {"1", "true", "on"})
    )
    if use_duplex_realtime and duplex_handler is not None:
        if await _reject_multi_api_duplex(websocket):
            return
        await _wait_for_duplex_warmup(websocket)
        await duplex_handler.handle_realtime_session(websocket)
        return

    if isinstance(duplex_query, str) and duplex_query.lower() in {"1", "true", "on"}:
        await websocket.accept()
        await websocket.send_json({"type": "error", "code": "unsupported", "error": "VAD realtime is not enabled"})
        await websocket.close(code=1008)
        return

    serving = getattr(websocket.app.state, "openai_serving_realtime", None)
    if serving is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "error": "Realtime API is not available", "code": "unsupported"})
        await websocket.close()
        return
    connection = RealtimeConnection(websocket, serving)
    await connection.handle_connection()


async def _wait_for_duplex_warmup(websocket: WebSocket) -> None:
    """Hold real clients until the startup duplex warmup finishes.

    The warmup connection marks itself with ``vllm_omni_warmup=1`` and passes through.
    """
    warmup_done = getattr(websocket.app.state, "duplex_warmup_done", None)
    if warmup_done is not None and not warmup_done.is_set() and websocket.query_params.get("vllm_omni_warmup") != "1":
        try:
            await asyncio.wait_for(warmup_done.wait(), timeout=120)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("Duplex warmup still running after 120 s; admitting the client anyway.")


@router.websocket("/v1/duplex")
async def duplex_websocket(websocket: WebSocket):
    """Alias of ``/v1/realtime?duplex=1``: the same Realtime duplex session protocol."""
    if await _reject_multi_api_duplex(websocket):
        return
    await _wait_for_duplex_warmup(websocket)
    handler = getattr(websocket.app.state, "openai_serving_duplex", None)
    if handler is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "error": "Duplex API is not available", "code": "unsupported"})
        await websocket.close()
        return
    await handler.handle_realtime_session(websocket)


@router.websocket("/v1/realtime/robot/openpi")
async def realtime_robot_openpi(websocket: WebSocket):
    """WebSocket endpoint for robot policy inference via OpenPI messages."""
    from vllm_omni.entrypoints.openpi.connection import (
        RobotRealtimeConnection,
    )

    serving = getattr(websocket.app.state, "openai_serving_realtime_robot", None)
    if serving is None:
        logger.warning(
            "Rejecting robot OpenPI WebSocket: policy serving is disabled. "
            "The diffusion stage must provide model_config.policy_server_config."
        )
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "error": "Robot policy not available",
                "code": "unsupported",
            }
        )
        await websocket.close()
        return
    state_args = getattr(websocket.app.state, "args", None)
    configured_timeout = getattr(state_args, "robot_openpi_idle_timeout", 30.0)
    idle_timeout = None if configured_timeout == 0 else configured_timeout
    connection = RobotRealtimeConnection(websocket, serving, idle_timeout=idle_timeout)
    await connection.handle_connection()


async def _reject_multi_api_duplex(websocket: WebSocket) -> bool:
    # Session credentials and replay state are local to each frontend; a shared
    # listening socket cannot route reconnects back to the session owner.
    count = getattr(websocket.app.state, "api_server_count", None)
    if count is None or (type(count) is int and count == 1):
        return False
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "error",
            "error": "Duplex sessions require a single API worker with initialized topology.",
            "code": "multi_api_duplex_unsupported",
        }
    )
    await websocket.close(code=1008)
    return True


# RL Rollout serving (RFC #3747, P0)


def _rl_rollout_serving(request: Request) -> ServingRLRollout:
    serving = getattr(request.app.state, "rl_rollout_serving", None)
    if serving is None:
        raise HTTPException(status_code=501, detail="RL rollout serving not available for this model.")
    return serving


@router.post("/v1/realtime/sessions")
async def create_rollout_session(body: CreateSessionRequest, request: Request):
    serving = _rl_rollout_serving(request)
    try:
        return (await serving.create_session(body)).model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RolloutSessionCapacityError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/v1/realtime/sessions/{session_id}/step")
async def rollout_step(session_id: str, body: RolloutStepRequest, request: Request):
    serving = _rl_rollout_serving(request)
    response = await serving.step(session_id, body)
    if response.error is not None:
        status_code = None
        if response.error.code == "session_not_found":
            status_code = 404
        elif response.error.code == "session_closed":
            status_code = 410
        elif response.error.code in {"invalid_request", "step_already_committed", "step_out_of_order"}:
            status_code = 400
        if status_code is not None:
            return JSONResponse(content=response.model_dump(), status_code=status_code)
    return response.model_dump()


@router.post("/v1/realtime/sessions/{session_id}/reset")
async def reset_rollout_session(session_id: str, request: Request):
    serving = _rl_rollout_serving(request)
    try:
        return (await serving.reset_session(session_id)).model_dump()
    except RolloutSessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    except RolloutSessionClosedError:
        raise HTTPException(status_code=410, detail=f"Session {session_id!r} is closed.")


@router.post("/v1/realtime/sessions/{session_id}/close")
async def close_rollout_session(session_id: str, request: Request):
    serving = _rl_rollout_serving(request)
    try:
        await serving.close_session(session_id)
        return {"session_id": session_id, "closed": True}
    except RolloutSessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    except RolloutSessionClosedError:
        return {"session_id": session_id, "closed": True}


@router.get("/v1/realtime/sessions/{session_id}/status")
async def rollout_session_status(session_id: str, request: Request):
    serving = _rl_rollout_serving(request)
    try:
        return (await serving.get_status(session_id)).model_dump()
    except RolloutSessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    except RolloutSessionClosedError:
        raise HTTPException(status_code=410, detail=f"Session {session_id!r} is closed.")


# Health and Model endpoints for diffusion mode


@router.get("/health")
async def health(raw_request: Request) -> JSONResponse:
    """Health check endpoint that works for both LLM and diffusion modes.

    Returns 200 OK if the server is healthy, 503 if the engine is dead.
    Mirrors vLLM upstream's /health which catches EngineDeadError -> 503.
    """
    engine_client = getattr(raw_request.app.state, "engine_client", None) or getattr(
        raw_request.app.state, "diffusion_engine", None
    )
    if engine_client is None:
        return JSONResponse(
            content={"status": "unhealthy", "reason": "No engine initialized"},
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )

    try:
        await engine_client.check_health()
        return JSONResponse(content={"status": "healthy"})
    except EngineDeadError:
        return JSONResponse(
            content={"status": "unhealthy"},
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )


@router.get("/ready")
async def ready(raw_request: Request) -> JSONResponse:
    """Readiness probe for the GPUStack backend (``health_check_path``).

    Unlike ``/health`` (engine-alive), ``/ready`` returns 200 only after the
    model is loaded *and* warmed up (``state.server_ready`` is set right after
    ``warmup()``), so the scheduler doesn't mark the instance Ready or route
    traffic during the multi-minute cold start. Mirrors LightX2V's launcher
    ``/ready`` (503 while loading, 200 when ready).
    """
    if not getattr(raw_request.app.state, "server_ready", False):
        return JSONResponse(
            content={"ready": False},
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )
    engine_client = getattr(raw_request.app.state, "engine_client", None) or getattr(
        raw_request.app.state, "diffusion_engine", None
    )
    if engine_client is None:
        return JSONResponse(
            content={"ready": False, "reason": "No engine initialized"},
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )
    try:
        await engine_client.check_health()
        return JSONResponse(content={"ready": True})
    except EngineDeadError:
        return JSONResponse(
            content={"ready": False},
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
        )


# Remove existing models endpoint if present (from vllm imports)
# to ensure our handler takes precedence
_remove_route_from_router(router, "/v1/models")


@router.get("/v1/models")
async def show_available_models(raw_request: Request) -> JSONResponse:
    """Show available models for both LLM and diffusion modes.

    Delegates to state.openai_serving_models which is set to either
    OpenAIServingModels (LLM) or _DiffusionServingModels (pure diffusion).
    """
    handler = getattr(raw_request.app.state, "openai_serving_models", None)
    if handler is not None:
        models = await handler.show_available_models()
        return JSONResponse(content=models.model_dump())
    return JSONResponse(content={"object": "list", "data": []})


# Image generation API endpoints


def _build_image_response_metrics(
    *,
    response_metrics: Any,
    stage_durations: Any,
    peak_memory_mb: Any,
) -> dict[str, Any]:
    """Merge detailed stage metrics with the legacy image timing fields."""
    metrics = dict(response_metrics) if isinstance(response_metrics, dict) else {}
    metrics["stage_durations"] = stage_durations or None
    metrics["peak_memory_mb"] = float(peak_memory_mb) if peak_memory_mb else None
    return metrics


def _build_image_generation_response(
    *,
    images: list[Image.Image],
    request: ImageGenerationRequest,
    stage_durations: Any,
    peak_memory_mb: Any,
    response_metrics: Any = None,
) -> ImageGenerationResponse | StreamingResponse:
    """Encode generated images and apply the requested response format."""
    output_format = _choose_output_format(request.output_format or "png", None)
    image_data = [
        ImageData(
            b64_json=encode_image_base64_with_compression(
                image, format=output_format, output_compression=request.output_compression
            ),
            revised_prompt=None,
        )
        for image in images
    ]
    response_kwargs: dict[str, Any] = {
        "created": int(time.time()),
        "data": image_data,
        "output_format": output_format,
        "metrics": _build_image_response_metrics(
            response_metrics=response_metrics,
            stage_durations=stage_durations,
            peak_memory_mb=peak_memory_mb,
        ),
    }
    size = _generated_size_str(images, request.size)
    if size is not None:
        response_kwargs["size"] = size
    response = ImageGenerationResponse(**response_kwargs)
    if request.response_format == ResponseFormat.FILE:
        return response.stream_response()
    return response


@router.post(
    "/v1/images/generations",
    dependencies=[Depends(validate_json_request)],
    response_model=None,
    responses={
        HTTPStatus.OK.value: {"model": ImageGenerationResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def generate_images(
    request: ImageGenerationRequest, raw_request: Request
) -> ImageGenerationResponse | StreamingResponse:
    """Generate images from text prompts using diffusion models.

    OpenAI DALL-E compatible endpoint for text-to-image generation.
    Only supports multi-stage omni mode with diffusion stages.

    Args:
        request: Image generation request with prompt and parameters
        raw_request: Raw FastAPI request for accessing app state

    Returns:
        ImageGenerationResponse with generated images as base64 PNG

    Raises:
        HTTPException: For validation errors, missing engine, or generation failures
    """
    request_timestamp = float(getattr(raw_request.state, "request_timestamp", time.time()))
    # Get engine client (AsyncOmni) from app state
    engine_client, model_name, stage_configs = _get_engine_and_model(raw_request)

    if request.model is not None and request.model != model_name:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(f"Model mismatch: request specifies '{request.model}' but server is running '{model_name}'."),
        )

    try:
        width: int | None = None
        height: int | None = None
        # Unify request construction for any multi-stage pipeline to avoid
        # divergence between /v1/images and /v1/chat/completions.
        if len(stage_configs) > 1:
            chat_handler = getattr(raw_request.app.state, "openai_serving_chat", None)
            if chat_handler is None:
                logger.warning("openai_serving_chat is not initialized for multi-stage /v1/images/generations")
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    detail="openai_serving_chat is not initialized for multi-stage image generation.",
                )

            effective_seed = request.seed if request.seed is not None else random.randint(0, MAX_UINT32_SEED)
            extra_body: dict[str, Any] = {
                "seed": effective_seed,
                "num_outputs_per_prompt": request.n,
            }
            if request.size is not None:
                width, height = parse_size(request.size)
                app_state_args = getattr(raw_request.app.state, "args", None)
                _check_max_generated_image_size(app_state_args, width, height)
                extra_body["size"] = request.size
            if request.negative_prompt is not None:
                extra_body["negative_prompt"] = request.negative_prompt
            if request.num_inference_steps is not None:
                extra_body["num_inference_steps"] = request.num_inference_steps
            if request.guidance_scale is not None:
                extra_body["guidance_scale"] = request.guidance_scale
            if request.true_cfg_scale is not None:
                extra_body["true_cfg_scale"] = request.true_cfg_scale
            if request.flow_shift is not None:
                extra_body["flow_shift"] = request.flow_shift
            if request.extra_params is not None:
                extra_body["extra_params"] = request.extra_params
            if request.generator_device is not None:
                extra_body["generator_device"] = request.generator_device
            if request.lora is not None:
                # Keep /images validation semantics: invalid LoRA should fail with 400.
                _parse_lora_request(request.lora)
                extra_body["lora"] = request.lora
            if request.bot_task is not None:
                extra_body["bot_task"] = request.bot_task
            if request.use_system_prompt is not None:
                extra_body["use_system_prompt"] = request.use_system_prompt
            if request.system_prompt is not None:
                extra_body["system_prompt"] = request.system_prompt
            if request.return_stage_metrics is not None:
                extra_body["return_stage_metrics"] = request.return_stage_metrics

            gen_request_id = f"img_gen-{random_uuid()}"
            async with _terminal_engine_awake(raw_request.app.state, request_id=gen_request_id):
                generation_result = await chat_handler.generate_diffusion_images(
                    prompt=request.prompt,
                    extra_body=extra_body,
                    request_id=gen_request_id,
                    raw_request=raw_request,
                    arrival_time=request_timestamp,
                )
            if isinstance(generation_result, ErrorResponse):
                return JSONResponse(
                    status_code=generation_result.error.code if generation_result.error else 400,
                    content=generation_result.model_dump(),
                )
            flat_images, stage_durations, peak_memory_mb, _, response_metrics = generation_result
            return _build_image_generation_response(
                images=flat_images,
                request=request,
                stage_durations=stage_durations,
                peak_memory_mb=peak_memory_mb,
                response_metrics=response_metrics,
            )

        # Build params - pass through user values directly
        prompt = OmniTextPrompt(prompt=request.prompt, modalities=["image"])
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt
        # Seed from the DEPLOY YAML's default_sampling_params, exactly as
        # /v1/tasks/image/ does. Starting blank here drops the shipped defaults and
        # lands on the pipeline's own fallback -- measured on
        # HunyuanImage-3.0-Instruct-Distil, whose deploy config asks for
        # num_inference_steps: 8: a request that simply omitted the field ran 50
        # steps in 36.0 s, against 5.5 s with steps=8. Same 6x blowup the tasks
        # route documents, on the route a gateway is most likely to call.
        #
        # It has to be this source, not apply_stage_default_sampling_params below:
        # that one reads the CLI --default-sampling-params JSON, which is a
        # DIFFERENT and usually empty channel. Seeding from it alone leaves the
        # deploy YAML just as ignored (verified: patched-vs-stock A/B both still
        # ran 50 steps until this was switched to the engine-client defaults).
        stage_defaults = get_default_sampling_params_list(engine_client)
        if stage_defaults:
            gen_params = clone_sampling_params(stage_defaults[0])
        else:
            gen_params = OmniDiffusionSamplingParams()
        # Legacy CLI-level defaults still apply on top of the YAML.
        apply_stage_default_sampling_params(
            getattr(getattr(raw_request.app.state, "args", None), "default_sampling_params", None),
            gen_params,
            "0",
        )
        _update_if_not_none(gen_params, "num_outputs_per_prompt", request.n)
        extra_args = dict(request.extra_params or {})
        if request.use_system_prompt is not None:
            extra_args["use_system_prompt"] = request.use_system_prompt
        if request.system_prompt is not None:
            extra_args["system_prompt"] = request.system_prompt
        if request.bot_task is not None:
            extra_args["bot_task"] = request.bot_task
        if request.flow_shift is not None:
            extra_args["flow_shift"] = request.flow_shift
        if extra_args:
            gen_params.extra_args = extra_args
        # Parse per-request LoRA (compatible with chat's extra_body.lora shape).
        lora_request, lora_scale = _parse_lora_request(request.lora) if request.lora is not None else (None, None)
        _update_if_not_none(gen_params, "lora_request", lora_request)
        _update_if_not_none(gen_params, "lora_scale", lora_scale)

        # Parse and add size if provided
        if request.size:
            width, height = parse_size(request.size)
            size_str = f"{width}x{height}"
        else:
            size_str = "model default"

        # Keep AR stage target grid in sync with requested output size.
        # GLM-Image consumes target_h/target_w via mm_processor_kwargs.
        if width is not None and height is not None:
            prompt["mm_processor_kwargs"] = {
                "target_h": height,
                "target_w": width,
            }
            # Backward-compatible fallback for processors reading top-level fields.
            prompt["height"] = height
            prompt["width"] = width
        app_state_args = getattr(raw_request.app.state, "args", None)
        _check_max_generated_image_size(app_state_args, width, height)

        _update_if_not_none(gen_params, "width", width)
        _update_if_not_none(gen_params, "height", height)

        # 3.3 Add optional parameters ONLY if provided
        _update_if_not_none(gen_params, "num_inference_steps", request.num_inference_steps)
        _update_if_not_none(gen_params, "guidance_scale", request.guidance_scale)
        _update_if_not_none(gen_params, "true_cfg_scale", request.true_cfg_scale)
        # If seed is not provided, generate a random one to ensure
        # a proper generator is initialized in the backend.
        # This fixes issues where using the default global generator
        # might produce blurry images in some environments.
        _update_if_not_none(
            gen_params, "seed", request.seed if request.seed is not None else random.randint(0, MAX_UINT32_SEED)
        )
        _update_if_not_none(gen_params, "generator_device", request.generator_device)
        _update_if_not_none(gen_params, "layers", request.layers)

        request_id = f"img_gen-{random_uuid()}"
        raw_request.state.request_metadata = RequestResponseMetadata(request_id=request_id)

        logger.debug(f"Generating {request.n} image(s) {size_str}")

        # Generate images using AsyncOmni (multi-stage mode)
        async with _terminal_engine_awake(raw_request.app.state, request_id=request_id):
            result = await _generate_with_async_omni(
                engine_client=engine_client,
                gen_params=gen_params,
                stage_configs=stage_configs,
                prompt=prompt,
                request_id=request_id,
                arrival_time=request_timestamp,
            )

        if result is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No output generated from multi-stage pipeline.",
            )

        # Extract images from result
        images = _extract_images_from_result(result)

        logger.debug(f"Successfully generated {len(images)} image(s)")

        stage_durations = getattr(result, "stage_durations", None)
        peak_memory_mb = getattr(result, "peak_memory_mb", None)
        response_metrics = getattr(result, "metrics", None) if request.return_stage_metrics else None
        return _build_image_generation_response(
            images=images,
            request=request,
            stage_durations=stage_durations,
            peak_memory_mb=peak_memory_mb,
            response_metrics=response_metrics,
        )

    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except HTTPException:
        raise
    except OmniClientError as e:
        logger.info("Client error during image generation: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))
    except Exception as e:
        logger.exception(f"Image generation failed: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Image generation failed: {str(e)}"
        )


@router.post(
    "/v1/images/edits",
    response_model=None,
    responses={
        HTTPStatus.OK.value: {"model": ImageGenerationResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def edit_images(
    raw_request: Request,
    image: list[UploadFile] | None = File(None),
    image_array: list[UploadFile] | None = File(None, alias="image[]"),
    url: list[str] | None = Form(None),
    url_array: list[str] | None = Form(None, alias="url[]"),
    prompt: str = Form(...),
    model: str = Form(None),
    n: int = Form(1),
    size: str = Form("auto"),
    response_format: str = Form("b64_json"),
    output_format: str | None = Form("png"),
    background: str | None = Form("auto"),
    output_compression: Annotated[int, Form(ge=0, le=100)] = 100,
    stream: bool = Form(False),
    user: str | None = Form(None),  # unused now
    # vllm-omni extensions for image editing
    mask_image: str | UploadFile | None = None,
    reference_image: str | UploadFile | None = None,
    # vllm-omni extensions for diffusion control
    negative_prompt: str | None = Form(None),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    guidance_scale_2: float | None = Form(None),
    strength: float | None = Form(None),
    true_cfg_scale: float | None = Form(None),
    seed: int | None = Form(None),
    generator_device: str | None = Form(None),
    # vllm-omni extension for per-request LoRA.
    lora: str | None = Form(None),  # Json string
    # vllm-omni extension for layered models (e.g., Qwen-Image-Layered)
    layers: int | None = Form(None),
    resolution: int | None = Form(None),  # See SUPPORTED_LAYERED_RESOLUTIONS
    # vllm-omni extension: pixel budget each condition image is resized to before the
    # text encoder and the VAE. Independent of `size` (which sets the generated image).
    # Only Qwen-Image 2.1 honours it today; other pipelines ignore it.
    condition_resolution: int | None = Form(None),
    # /v1/images/edits is always IT2I; only the prompting knobs are exposed.
    bot_task: str | None = Form(None),
    sys_type: str | None = Form(None),
    system_prompt: str | None = Form(None),
    return_stage_metrics: bool | None = Form(None),
) -> ImageGenerationResponse:
    """
    OpenAI-compatible image edit endpoint.
    """

    # 1. get engine and model
    request_timestamp = float(getattr(raw_request.state, "request_timestamp", time.time()))
    engine_client, model_name, stage_configs = _get_engine_and_model(raw_request)
    if model is not None and model != model_name:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(f"Model mismatch: request specifies '{model}' but server is running '{model_name}'."),
        )
    # 2. get output format & compression
    output_format = _choose_output_format(output_format, background)
    if response_format != "b64_json":
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Only response_format 'b64_json' is supported now.",
        )
    if stream and len(stage_configs) <= 1:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="stream=true is only supported for multi-stage image editing pipelines.",
        )
    try:
        # 2. Build prompt & images params
        cot_output = None
        omni_prompt = OmniTextPrompt(prompt=prompt, modalities=["image"])
        if negative_prompt is not None:
            omni_prompt["negative_prompt"] = negative_prompt
        input_images_list = []
        images = image or image_array
        urls = url or url_array
        if images:
            input_images_list.extend(images)
        if urls:
            input_images_list.extend(urls)
        if not input_images_list:
            raise HTTPException(status_code=422, detail="Field 'image' or 'url' is required")
        # Reject oversized multi-image edit requests before fetching or decoding
        # any inputs. This keeps over-limit URL requests from burning network,
        # CPU, and memory on work that will be rejected anyway.
        max_input_images = _get_max_edit_input_images(raw_request.app.state, engine_client)
        if max_input_images is not None and len(input_images_list) > max_input_images:
            detail = too_many_input_images_message(len(input_images_list), max_input_images)
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=detail,
            )
        # Match the offline path: RGB normalize when the caller opts into
        # Hunyuan-aware behavior. RGBA/P uploads otherwise diverge from offline.
        normalize_edit_images_rgb = bot_task is not None or sys_type is not None
        pil_images = await _load_input_images(input_images_list, normalize_rgb=normalize_edit_images_rgb)
        omni_prompt["multi_modal_data"] = {}
        omni_prompt["multi_modal_data"]["image"] = pil_images

        if mask_image is not None:
            # Mask role is different (alpha channel matters); never normalize.
            loaded = await _load_input_images([mask_image], normalize_rgb=False)
            omni_prompt["multi_modal_data"]["mask_image"] = loaded[0]

        if reference_image is not None:
            loaded = await _load_input_images([reference_image], normalize_rgb=normalize_edit_images_rgb)
            omni_prompt["multi_modal_data"]["reference_image"] = loaded[0]

        # 3 Build sample params
        # 3.0a Seed from the DEPLOY YAML's default_sampling_params. The CLI-level
        # apply_stage_default_sampling_params() below is a DIFFERENT and usually
        # empty channel, so on its own it leaves the YAML unread and the pipeline's
        # built-in default wins -- measured on HunyuanImage-3.0-Instruct-Distil
        # (YAML asks for num_inference_steps: 8): an edit request that omitted the
        # field ran 50 steps in 29.7 s, while the same route with an explicit
        # steps=8 and THREE reference images took 10.3 s. i2i is the editing route,
        # so this is the one production image-edit traffic actually lands on.
        stage_defaults = get_default_sampling_params_list(engine_client)
        gen_params = clone_sampling_params(stage_defaults[0]) if stage_defaults else OmniDiffusionSamplingParams()
        # 3.0 Init with system default values
        app_state_args = getattr(raw_request.app.state, "args", None)
        default_sample_param = getattr(app_state_args, "default_sampling_params", None)
        # Currently only have one diffusion stage.
        diffusion_stage_ids = [i for i, cfg in enumerate(stage_configs) if get_stage_type(cfg) == "diffusion"]
        if not diffusion_stage_ids:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                detail="No diffusion stage found in multi-stage pipeline.",
            )
        diffusion_stage_id = diffusion_stage_ids[0]
        apply_stage_default_sampling_params(
            default_sample_param,
            gen_params,
            str(diffusion_stage_id),
        )
        _update_if_not_none(gen_params, "num_outputs_per_prompt", n)
        # 3.1 Parse per-request LoRA (compatible with chat's extra_body.lora shape).
        lora_dict = _get_lora_from_json_str(lora)
        lora_request, lora_scale = _parse_lora_request(lora_dict)
        _update_if_not_none(gen_params, "lora_request", lora_request)
        _update_if_not_none(gen_params, "lora_scale", lora_scale)
        # 3.2 Validate resolution if provided
        if resolution is not None and resolution not in SUPPORTED_LAYERED_RESOLUTIONS:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Invalid resolution {resolution}. Supported resolutions: {SUPPORTED_LAYERED_RESOLUTIONS}.",
            )
        # Reject an out-of-range condition resolution here: too small silently degrades
        # reference fidelity, too large only surfaces as an OOM minutes into the request.
        if condition_resolution is not None and not (
            MIN_CONDITION_RESOLUTION <= condition_resolution <= MAX_CONDITION_RESOLUTION
        ):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=(
                    f"Invalid condition_resolution {condition_resolution}. Expected "
                    f"{MIN_CONDITION_RESOLUTION}-{MAX_CONDITION_RESOLUTION}."
                ),
            )
        # 3.2.1 Validate layers if provided
        try:
            layers = validate_layered_layers(layers)
        except ValueError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=str(e),
            ) from e
        # 3.2.2 Check for conflicting size and resolution parameters
        if resolution is not None and size.lower() != "auto":
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Cannot specify both 'resolution' and 'size'. "
                "Use 'resolution' with size='auto', or use 'size' without 'resolution'.",
            )

        # 3.3 Parse and add size if provided
        width, height = None, None
        size_was_auto = size.lower() == "auto"
        if size_was_auto:
            if resolution is None:
                # No resolution specified, use input image size
                width, height = pil_images[0].size
            # else: let pipeline calculate dimensions based on resolution
        else:
            width, height = parse_size(size)

        _check_max_generated_image_size(app_state_args, width, height, resolution)

        size_str = f"{width}x{height}" if width is not None and height is not None else "auto"

        # Keep AR stage target grid in sync with requested output size.
        # GLM-Image consumes target_h/target_w via mm_processor_kwargs.
        if width is not None and height is not None:
            omni_prompt["mm_processor_kwargs"] = {
                "target_h": height,
                "target_w": width,
            }
            # Backward-compatible fallback for processors reading top-level fields.
            omni_prompt["height"] = height
            omni_prompt["width"] = width

        _update_if_not_none(gen_params, "width", width)
        _update_if_not_none(gen_params, "height", height)
        gen_params.width_not_provided = size_was_auto
        gen_params.height_not_provided = size_was_auto

        # 3.4 Add optional parameters ONLY if provided
        _update_if_not_none(gen_params, "num_inference_steps", num_inference_steps)
        _update_if_not_none(gen_params, "guidance_scale", guidance_scale)
        _update_if_not_none(gen_params, "guidance_scale_2", guidance_scale_2)
        _update_if_not_none(gen_params, "strength", strength)
        _update_if_not_none(gen_params, "true_cfg_scale", true_cfg_scale)
        # If seed is not provided, generate a random one to ensure
        # a proper generator is initialized in the backend.
        # This fixes issues where using the default global generator
        # might produce blurry images in some environments.
        _update_if_not_none(gen_params, "seed", seed if seed is not None else random.randint(0, MAX_UINT32_SEED))
        _update_if_not_none(gen_params, "generator_device", generator_device)
        _update_if_not_none(gen_params, "layers", layers)
        _update_if_not_none(gen_params, "resolution", resolution)
        _update_if_not_none(gen_params, "condition_resolution", condition_resolution)

        extra_args = dict(getattr(gen_params, "extra_args", {}) or {})
        edit_extra_args = _build_hunyuan_edit_extra_args(
            bot_task=bot_task,
            sys_type=sys_type,
            system_prompt=system_prompt,
        )
        extra_args.update(edit_extra_args)
        if extra_args:
            gen_params.extra_args = extra_args

        # 4. Generate images
        request_id = f"img_edit-{random_uuid()}"
        raw_request.state.request_metadata = RequestResponseMetadata(request_id=request_id)
        logger.debug(f"Generating {n} image(s) {size_str}")

        if len(stage_configs) > 1:
            # Multi-stage pipeline (e.g. GLM-Image AR+Diffusion): route through
            # the chat handler so the AR stage gets correct max_tokens and
            # target_h/w (same path as /v1/images/generations).
            chat_handler = getattr(raw_request.app.state, "openai_serving_chat", None)
            if chat_handler is None:
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    detail="openai_serving_chat is not initialized for multi-stage image editing.",
                )

            # Encode input images to base64 for generate_diffusion_images.
            import base64
            import io as _io

            ref_b64_list: list[str] = []
            for _img in pil_images:
                buf = _io.BytesIO()
                _img.save(buf, format="PNG")
                ref_b64_list.append(base64.b64encode(buf.getvalue()).decode())

            effective_seed = seed if seed is not None else random.randint(0, MAX_UINT32_SEED)
            extra_body: dict[str, Any] = {
                "seed": effective_seed,
                "num_outputs_per_prompt": n,
            }
            # size="auto" resolves width/height from input image; forwarding
            # those would override AR-driven `<img_ratio_*>` token selection.
            if not size_was_auto:
                if width is not None:
                    extra_body["width"] = width
                if height is not None:
                    extra_body["height"] = height
            if negative_prompt is not None:
                extra_body["negative_prompt"] = negative_prompt
            if num_inference_steps is not None:
                extra_body["num_inference_steps"] = num_inference_steps
            if guidance_scale is not None:
                extra_body["guidance_scale"] = guidance_scale
            if guidance_scale_2 is not None:
                extra_body["guidance_scale_2"] = guidance_scale_2
            if strength is not None:
                extra_body["strength"] = strength
            if true_cfg_scale is not None:
                extra_body["true_cfg_scale"] = true_cfg_scale
            if layers is not None:
                extra_body["layers"] = layers
            if resolution is not None:
                extra_body["resolution"] = resolution
            if lora is not None:
                # Validate LoRA, then pass through.
                lora_dict = _get_lora_from_json_str(lora)
                _parse_lora_request(lora_dict)
                extra_body["lora"] = lora_dict
            if bot_task is not None:
                extra_body["bot_task"] = bot_task
            if sys_type is not None:
                extra_body["sys_type"] = sys_type
            if system_prompt is not None:
                extra_body["system_prompt"] = system_prompt
            if return_stage_metrics is not None:
                extra_body["return_stage_metrics"] = return_stage_metrics

            prompt_text = omni_prompt.get("prompt", "")
            # Forward the mask explicitly. This branch used to build
            # omni_prompt["multi_modal_data"]["mask_image"] above and then drop
            # it, because only prompt_text is passed on -- so a masked edit
            # silently ran UNMASKED on any multi-stage pipeline.
            mask_pil = (omni_prompt.get("multi_modal_data") or {}).get("mask_image")
            mask_b64 = _encode_images_png_b64([mask_pil])[0] if mask_pil is not None else None
            # Not a bare `async with`: with stream=true this returns an iterator
            # before generating, so the session must outlive the call and be
            # released only when the stream is consumed.
            generation_result = await _run_with_terminal_engine_awake(
                raw_request.app.state,
                request_id=request_id,
                run=lambda: chat_handler.generate_diffusion_images(
                    prompt=prompt_text,
                    extra_body=extra_body,
                    reference_images=ref_b64_list,
                    mask_image=mask_b64,
                    request_id=request_id,
                    arrival_time=request_timestamp,
                    stream=stream,
                    model=model_name,
                    output_format=output_format,
                    output_compression=output_compression,
                    size=size_str,
                    raw_request=raw_request,
                ),
            )
            if stream and not isinstance(generation_result, ErrorResponse):
                return StreamingResponse(
                    content=generation_result,
                    media_type="text/event-stream",
                )
            if isinstance(generation_result, ErrorResponse):
                raise HTTPException(
                    status_code=generation_result.error.code if generation_result.error else 400,
                    detail=generation_result.message,
                )
            images, stage_durations, peak_memory_mb, cot_output, response_metrics = generation_result
        else:
            # Single-stage diffusion: use the direct path.
            async with _terminal_engine_awake(raw_request.app.state, request_id=request_id):
                result = await _generate_with_async_omni(
                    engine_client=engine_client,
                    gen_params=gen_params,
                    stage_configs=stage_configs,
                    prompt=omni_prompt,
                    request_id=request_id,
                )
            images = _extract_images_from_result(result)
            stage_durations = getattr(result, "stage_durations", None)
            peak_memory_mb = getattr(result, "peak_memory_mb", None)
            response_metrics = getattr(result, "metrics", None) if return_stage_metrics else None

        if images is None:
            images = []
        logger.debug(f"Successfully generated {len(images)} image(s)")

        # Encode images to base64
        image_data = [
            ImageData(
                b64_json=encode_image_base64_with_compression(
                    img, format=output_format, output_compression=output_compression
                ),
                revised_prompt=None,
            )
            for img in images
        ]

        return ImageGenerationResponse(
            created=int(time.time()),
            data=image_data,
            output_format=output_format,
            size=_generated_size_str(images, size_str),
            cot_output=cot_output,
            metrics=_build_image_response_metrics(
                response_metrics=response_metrics,
                stage_durations=stage_durations,
                peak_memory_mb=peak_memory_mb,
            ),
        )

    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except HTTPException:
        raise
    except OmniClientError as e:
        logger.info("Client error during image edit: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e))
    except Exception as e:
        logger.exception(f"Image edit failed: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Image edit failed: {str(e)}")


@router.post(
    "/v1/videos",
    dependencies=[Depends(_require_single_api_video_store)],
    responses={
        HTTPStatus.OK.value: {"model": VideoResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def create_video(
    raw_request: Request,
    ctx: tuple[
        VideoGenerationRequest,
        OmniOpenAIServingVideo,
        str,
        ReferenceImage | None,
        ReferenceVideo | None,
        ReferenceAudio | None,
        str | None,
        LatentEditInput | None,
    ] = Depends(_parse_video_form),
) -> VideoResponse:
    """Create an asynchronous video generation job.

    Accepts multipart form-data (see ``_parse_video_form`` for parameters),
    persists a queued job record, and starts generation in the background.
    """
    (
        request,
        handler,
        effective_model_name,
        reference_image,
        reference_video,
        reference_audio,
        control_path,
        latent_edit_input,
    ) = ctx
    ref = video_response_from_request(effective_model_name, request)
    await VIDEO_STORE.upsert(ref.id, ref)
    task = asyncio.create_task(
        _run_video_generation_job(
            handler,
            request,
            ref.id,
            reference_image,
            reference_video,
            reference_audio,
            control_path,
            app_state=raw_request.app.state,
            latent_edit_input=latent_edit_input,
        )
    )
    await VIDEO_TASKS.upsert(ref.id, task)
    return ref


@router.post(
    "/v1/videos/sync",
    responses={
        HTTPStatus.OK.value: {"content": {"video/mp4": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def create_video_sync(
    raw_request: Request,
    ctx: tuple[
        VideoGenerationRequest,
        OmniOpenAIServingVideo,
        str,
        ReferenceImage | None,
        ReferenceVideo | None,
        ReferenceAudio | None,
        str | None,
        LatentEditInput | None,
    ] = Depends(_parse_video_form),
) -> Response:
    """Synchronous video generation endpoint.

    Accepts the same form parameters as ``POST /v1/videos`` but blocks until
    generation completes and returns raw video bytes (``video/mp4``) directly.
    Designed for benchmark and testing scenarios.

    Metadata is returned via response headers ``X-Request-Id``,
    ``X-Model``, and ``X-Inference-Time-S``.
    """
    (
        request,
        handler,
        effective_model_name,
        reference_image,
        reference_video,
        reference_audio,
        control_path,
        latent_edit_input,
    ) = ctx
    request_id = f"video_sync-{random_uuid()}"
    raw_request.state.request_metadata = RequestResponseMetadata(request_id=request_id)
    started_at = time.perf_counter()
    try:
        video_bytes, stage_durations, peak_memory_mb, _action, _video_metadata = _unpack_video_generation_result(
            await asyncio.wait_for(
                handler.generate_video_bytes(
                    request,
                    request_id,
                    reference_image=reference_image,
                    reference_video=reference_video,
                    reference_audio=reference_audio,
                    latent_edit_input=latent_edit_input,
                ),
                timeout=VIDEO_SYNC_TIMEOUT_S,
            ),
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=HTTPStatus.GATEWAY_TIMEOUT.value,
            detail=f"Video generation timed out after {VIDEO_SYNC_TIMEOUT_S}s.",
        )
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except HTTPException:
        raise
    except OmniClientError as exc:
        logger.info("Client error during sync video generation: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        logger.exception("Sync video generation failed for request_id=%s", request_id)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"Video generation failed: {str(exc)}",
        ) from exc
    finally:
        _cleanup_video_references(reference_video, reference_audio, control_path, latent_edit_input)
    inference_time_s = time.perf_counter() - started_at

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "X-Request-Id": request_id,
            "X-Model": effective_model_name,
            "X-Inference-Time-S": f"{inference_time_s:.3f}",
            "X-Stage-Durations": json.dumps(stage_durations, separators=(",", ":")),
            "X-Peak-Memory-MB": f"{peak_memory_mb:.3f}",
        },
    )


@router.get("/v1/videos", response_model=VideoListResponse, dependencies=[Depends(_require_single_api_video_store)])
async def list_videos(
    after: str | None = None,
    limit: int | None = Query(None, ge=0, le=100),
    order: Annotated[Literal["asc", "desc"], Query()] = "desc",
):
    """List stored video generation jobs.

    Args:
        after: Optional cursor indicating the last seen video ID.
        limit: Optional maximum number of jobs to return.
        order: Sort order for the returned jobs by creation time.

    Returns:
        A ``VideoListResponse`` containing paginated job metadata and cursor
        information.
    """
    jobs = await VIDEO_STORE.list_values()
    jobs.sort(key=lambda j: j.created_at, reverse=order == "desc")

    if after is not None:
        idx = next((i for i, job in enumerate(jobs) if job.id == after), None)
        jobs = [] if idx is None else jobs[idx + 1 :]

    has_more = False
    if limit is not None:
        has_more = len(jobs) > limit
        jobs = jobs[:limit]

    first_id, last_id = None, None
    if len(jobs) > 0:
        first_id = jobs[0].id
        last_id = jobs[-1].id

    return VideoListResponse(data=jobs, has_more=has_more, first_id=first_id, last_id=last_id)


@router.get("/v1/videos/{video_id}", response_model=None, dependencies=[Depends(_require_single_api_video_store)])
async def retrieve_video(video_id: str) -> VideoResponse | JSONResponse:
    """Retrieve metadata for a previously created video job.

    Args:
        video_id: Identifier returned by ``POST /v1/videos``.

    Returns:
        The stored ``VideoResponse`` for the requested job.

    Raises:
        HTTPException: If the video job does not exist.
    """
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if job.status == VideoGenerationStatus.FAILED:
        status_code = _status_code_for_video_failure(job.error)
        content = job.model_dump(mode="json")
        if content.get("error") is not None:
            content["error"]["code"] = status_code
        return JSONResponse(
            content=content,
            status_code=status_code,
        )
    return job


@router.delete("/v1/videos/{video_id}", dependencies=[Depends(_require_single_api_video_store)])
async def delete_video(video_id: str, raw_request: Request) -> VideoDeleteResponse:
    """Delete a stored video job and any generated output.

    In-flight jobs get a bounded engine abort, then frontend cancel. The job
    is re-read afterwards so a completed save is not orphaned.

    Args:
        video_id: Identifier of the video job to delete.

    Returns:
        A ``VideoDeleteResponse`` confirming the job was removed.

    Raises:
        HTTPException: If the video job does not exist, cancellation is still
        in progress, or output is not yet ready for a completed job.
    """
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")

    if job.status in (VideoGenerationStatus.QUEUED, VideoGenerationStatus.IN_PROGRESS):
        handler = raw_request.app.state.openai_serving_video
        try:
            await asyncio.wait_for(handler.abort_request(video_id), timeout=VIDEO_ABORT_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out aborting video request %s after %.1fs; "
                "engine abort is best-effort until the current batch drains",
                video_id,
                VIDEO_ABORT_TIMEOUT_S,
            )
        except Exception:
            logger.exception("Failed to abort in-flight video request %s", video_id)
        task = await VIDEO_TASKS.get(video_id)
        if task is not None:
            task.cancel()
            try:
                # Cancel cleanup may spend a full abort budget; +2s covers scheduling slack.
                await asyncio.wait_for(task, timeout=VIDEO_ABORT_TIMEOUT_S + 2.0)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=409, detail="Cancellation in progress. Please try again later.")
            except asyncio.CancelledError:
                pass

        job = await VIDEO_STORE.get(video_id)
        if job is None:
            return VideoDeleteResponse(id=video_id, deleted=True)
        if job.status in (VideoGenerationStatus.QUEUED, VideoGenerationStatus.IN_PROGRESS):
            await VIDEO_STORE.pop(video_id)
            return VideoDeleteResponse(id=job.id, deleted=True)

    if job.status is VideoGenerationStatus.FAILED:
        if job.file_name is not None:
            try:
                await STORAGE_MANAGER.delete(video_id)
            except Exception:
                logger.warning("Failed to delete stored artifact for failed video job %s", video_id, exc_info=True)

        await VIDEO_STORE.pop(video_id)
        return VideoDeleteResponse(id=job.id, deleted=True)

    if job.file_name is None:
        raise HTTPException(status_code=409, detail="Video output not yet available. Please try again later.")

    await STORAGE_MANAGER.delete(video_id)
    await VIDEO_STORE.pop(video_id)
    return VideoDeleteResponse(id=job.id, deleted=True)


@router.get("/v1/videos/{video_id}/content", dependencies=[Depends(_require_single_api_video_store)])
async def download_video(video_id: str) -> Response:
    """Download the generated file for a completed video job.

    Args:
        video_id: Identifier of the video job whose output should be returned.

    Returns:
        A ``FileResponse`` streaming the generated video file from local
        storage.

    Raises:
        HTTPException: If the job does not exist, is still in progress, or the
        generated file is missing from disk.
    """
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")

    if job.status == VideoGenerationStatus.FAILED:
        raise HTTPException(status_code=422, detail="Video generation failed. Check job status for error details.")
    if not job.file_name:
        raise HTTPException(status_code=404, detail="Generation is still in-progress")

    file_handle = await STORAGE_MANAGER.open(video_id)
    if file_handle is None:
        raise HTTPException(status_code=404, detail="Generated video file not found on disk")

    file_name = job.file_name or f"{video_id}.{job.file_extension}"
    if isinstance(file_handle, FileStorageHandle):
        response = FileResponse(path=file_handle.path, media_type=job.media_type, filename=file_name)
    else:
        raise HTTPException(
            status_code=500, detail=f"Server generated an unsupported file storage handle for file id {video_id}"
        )

    return response


# ---------------------------------------------------------------------------
# Async audio task API (GPUStack integration).
#
# Contract mirrors LightX2V / IndexTTS so the GPUStack facade drives TTS via
# submit + poll (needed because generation can exceed a minute). Reuses the
# generic store/registry (AUDIO_TASK_STORE / AUDIO_TASKS) and the same speech
# handler as the synchronous /v1/audio/speech; models are unaffected.
# ---------------------------------------------------------------------------

# Per-model safety cap (defense against engine-killing over-long text, e.g.
# IndexTTS' hard sentence cliff). Set per model deployment; the authoritative
# user-facing limit lives upstream in new-api.
_AUDIO_MAX_TEXT_LEN = int(os.environ.get("VLLM_OMNI_AUDIO_MAX_TEXT_LEN", "5000"))


async def _run_audio_generation_job(
    handler: OmniOpenAIServingSpeech,
    request: AudioTaskRequest,
    task_id: str,
    save_result_path: str,
    app_state: Any | None = None,
) -> None:
    job = await AUDIO_TASK_STORE.get(task_id)
    if job is None:
        logger.warning("Audio task %s missing before generation started; skipping", task_id)
        return

    await AUDIO_TASK_STORE.update_fields(task_id, {"status": AudioTaskStatus.PROCESSING, "start_time": time.time()})
    try:
        speech_request = request.to_speech_request()
        # _generate_audio_bytes is the non-streaming byte path: it returns
        # (bytes, media_type). create_speech() would return a FastAPI
        # Response/StreamingResponse instead, so we must call the lower-level
        # helper to get raw bytes for the NFS write.
        audio_bytes, _media_type = await handler._generate_audio_bytes(speech_request, request_id=task_id)
        if not isinstance(audio_bytes, bytes | bytearray):
            raise RuntimeError("Speech handler did not return raw audio bytes")
        await atomic_write_bytes(bytes(audio_bytes), save_result_path)
        logger.info("Audio task %s wrote %d bytes to %s", task_id, len(audio_bytes), save_result_path)
        await AUDIO_TASK_STORE.update_fields(task_id, {"status": AudioTaskStatus.COMPLETED, "end_time": time.time()})
    except asyncio.CancelledError:
        await AUDIO_TASK_STORE.update_fields(task_id, {"status": AudioTaskStatus.CANCELLED, "end_time": time.time()})
        raise
    except (EngineGenerateError, EngineDeadError) as exc:
        logger.exception("Audio task %s failed (engine error)", task_id)
        await AUDIO_TASK_STORE.update_fields(
            task_id,
            {
                "status": AudioTaskStatus.FAILED,
                "end_time": time.time(),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        # Background tasks can't propagate to FastAPI handlers; signal shutdown
        # when the engine is dead (same as the video job).
        if app_state is not None and isinstance(exc, EngineDeadError):
            terminate_if_errored(server=app_state.server, engine=app_state.engine_client)
    except Exception as exc:
        logger.exception("Audio task %s failed", task_id)
        await AUDIO_TASK_STORE.update_fields(
            task_id,
            {
                "status": AudioTaskStatus.FAILED,
                "end_time": time.time(),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )


@router.post(
    "/v1/tasks/audio/",
    responses={
        HTTPStatus.OK.value: {"model": AudioTaskResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
    },
)
async def create_audio_task(request: AudioTaskRequest, raw_request: Request) -> AudioTaskResponse:
    """Submit an asynchronous TTS task (returns immediately; poll for status)."""
    handler = Omnispeech(raw_request)
    if handler is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="The model does not support Speech API")

    if not (request.input or "").strip():
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="Empty synthesis text")
    if len(request.input) > _AUDIO_MAX_TEXT_LEN:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                f"Text too long ({len(request.input)} > {_AUDIO_MAX_TEXT_LEN}); "
                "tune VLLM_OMNI_AUDIO_MAX_TEXT_LEN for this model."
            ),
        )

    task_id = request.task_id or f"audio_task_{random_uuid()}"
    save_result_path = resolve_save_path(request.save_result_path, task_id, STORAGE_MANAGER.storage_path)

    try:
        ref = await AUDIO_TASK_MANAGER.reserve(task_id, save_result_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE.value, detail=str(exc)) from exc

    task = asyncio.create_task(
        _run_audio_generation_job(handler, request, task_id, save_result_path, app_state=raw_request.app.state)
    )
    await AUDIO_TASKS.upsert(task_id, task)
    return ref


# ---------------------------------------------------------------------------
# Async diffusion-audio task API (AudioX / SoulX-Singer).
#
# Sibling of the TTS async API above: same submit+poll contract and the SAME
# task store / manager / registry (so the global status/result/cancel endpoints
# below serve both). The only difference is the byte source — these models are
# diffusion models served through the CHAT handler in diffusion mode
# (modalities=["audio"]), which returns a base64 WAV in
# choices[0].message.audio.data. A new submit endpoint (/v1/tasks/audiogen/)
# runs that path and writes the decoded bytes to save_result_path.
# ---------------------------------------------------------------------------


async def _run_audio_gen_job(
    handler: OmniOpenAIServingChat,
    request: AudioGenTaskRequest,
    task_id: str,
    save_result_path: str,
    app_state: Any | None = None,
) -> None:
    job = await AUDIO_TASK_STORE.get(task_id)
    if job is None:
        logger.warning("Audiogen task %s missing before generation started; skipping", task_id)
        return

    await AUDIO_TASK_STORE.update_fields(task_id, {"status": AudioTaskStatus.PROCESSING, "start_time": time.time()})
    try:
        # raw_request is unused in the diffusion branch of create_chat_completion,
        # so calling with raw_request=None is safe (the None-guard is only on the
        # non-diffusion branch).
        chat_req = request.to_chat_request(request_id=task_id)
        resp = await handler.create_chat_completion(chat_req, raw_request=None)
        if isinstance(resp, ErrorResponse):
            # Surface the handler's error message; the except block below records
            # it as FAILED with the error text.
            raise RuntimeError(getattr(getattr(resp, "error", None), "message", None) or str(resp))
        # Diffusion audio returns a base64 WAV in choices[0].message.audio.data
        # (an OpenAIChatCompletionAudio). Decode it to raw bytes for the NFS write.
        audio_b64 = resp.choices[0].message.audio.data
        if not audio_b64:
            raise RuntimeError("Diffusion chat completion returned no audio data")
        audio_bytes = base64.b64decode(audio_b64)
        await atomic_write_bytes(audio_bytes, save_result_path)
        logger.info("Audiogen task %s wrote %d bytes to %s", task_id, len(audio_bytes), save_result_path)
        await AUDIO_TASK_STORE.update_fields(task_id, {"status": AudioTaskStatus.COMPLETED, "end_time": time.time()})
    except asyncio.CancelledError:
        await AUDIO_TASK_STORE.update_fields(task_id, {"status": AudioTaskStatus.CANCELLED, "end_time": time.time()})
        raise
    except (EngineGenerateError, EngineDeadError) as exc:
        logger.exception("Audiogen task %s failed (engine error)", task_id)
        await AUDIO_TASK_STORE.update_fields(
            task_id,
            {
                "status": AudioTaskStatus.FAILED,
                "end_time": time.time(),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        if app_state is not None and isinstance(exc, EngineDeadError):
            terminate_if_errored(server=app_state.server, engine=app_state.engine_client)
    except Exception as exc:
        logger.exception("Audiogen task %s failed", task_id)
        await AUDIO_TASK_STORE.update_fields(
            task_id,
            {
                "status": AudioTaskStatus.FAILED,
                "end_time": time.time(),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )


@router.post(
    "/v1/tasks/audiogen/",
    responses={
        HTTPStatus.OK.value: {"model": AudioTaskResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
    },
)
async def create_audio_gen_task(request: AudioGenTaskRequest, raw_request: Request) -> AudioTaskResponse:
    """Submit an asynchronous diffusion-audio task (AudioX / SoulX-Singer).

    Returns immediately with a PENDING record; poll the shared global status /
    result / cancel endpoints (keyed by task_id) exactly like the TTS async API.
    """
    handler = Omnichat(raw_request)
    if handler is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND.value, detail="The model does not support Chat Completion API"
        )

    if not (request.input or "").strip():
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="Empty generation text")

    # The GPUStack facade strips `model` from the task body (it is a control key;
    # the engine instance serves a single model), so `model` may be absent. Fill
    # it from this server's served model name so to_chat_request builds a
    # ChatCompletionRequest whose model matches the diffusion server's base model
    # (mirrors the TTS AudioTaskRequest, whose model is likewise optional).
    if not (request.model or "").strip():
        try:
            request.model = raw_request.app.state.openai_serving_models.base_model_paths[0].name
        except (AttributeError, IndexError):
            pass

    task_id = request.task_id or f"audiogen_task_{random_uuid()}"
    save_result_path = resolve_save_path(request.save_result_path, task_id, STORAGE_MANAGER.storage_path)

    try:
        ref = await AUDIO_TASK_MANAGER.reserve(task_id, save_result_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE.value, detail=str(exc)) from exc

    task = asyncio.create_task(
        _run_audio_gen_job(handler, request, task_id, save_result_path, app_state=raw_request.app.state)
    )
    await AUDIO_TASKS.upsert(task_id, task)
    return ref


# ---------------------------------------------------------------------------
# Async image task API (text-to-image / image editing).
#
# Third sibling of the TTS and diffusion-audio async APIs above, sharing their
# task store / manager / registry so the global status, result, cancel and queue
# endpoints below serve image tasks unchanged — only submit is new.
#
# It exists because the GPUStack facade dispatches strictly by engine kind,
# POSTing to ``v1/tasks/{kind}/`` (gpustack routes/videos.py), so an image model
# is only reachable through the facade if the engine exposes /v1/tasks/image/.
# The synchronous /v1/images/edits endpoint stays the direct-caller surface.
# ---------------------------------------------------------------------------


def _resolve_task_image_path(path: str, allowed_root: str | None, *, media_kind: str = "image") -> str:
    """Validate one facade-injected input media path.

    SECURITY: unlike the sync endpoints (which take bytes/URLs), this endpoint
    reads server-side paths. The facade owns ``image_path`` and only ever injects
    paths under its own NFS root, but this engine's raw OpenAI endpoints are ALSO
    reachable through the GPUStack model proxy, so a model user can call here
    directly with a hand-written path. Without a whitelist that is an arbitrary
    file read of every shared mount and every other tenant's output.

    We reuse vLLM's existing ``--allowed-local-media-path`` rather than inventing
    a second whitelist, so operators have one knob. Fail closed: when it is unset
    we refuse path inputs entirely instead of allowing everything.

    ``media_kind`` only shapes the error text — /v1/tasks/video/ feeds video and
    audio paths through the same confinement, and "Input image not found" for an
    mp3 sends whoever is debugging to the wrong field.
    """
    candidate = (path or "").strip()
    if candidate.startswith("file://"):
        candidate = url2pathname(urlparse(candidate).path)
    if not candidate:
        raise ValueError(f"Empty input {media_kind} path")
    if not os.path.isabs(candidate):
        raise ValueError(f"Input {media_kind} path must be absolute: {candidate!r}")
    if not allowed_root:
        raise ValueError(
            f"Refusing to read a local input {media_kind} because --allowed-local-media-path is not set. "
            "Set it to the facade's media root (never '/') to enable path inputs."
        )
    real_root = os.path.realpath(allowed_root)
    real_path = os.path.realpath(candidate)
    # commonpath, not startswith: "/nfs-output-evil" must not pass for root
    # "/nfs-output". realpath first so a symlink cannot escape the root.
    try:
        inside = os.path.commonpath([real_root, real_path]) == real_root
    except ValueError:  # different drives / mixed absolute-relative
        inside = False
    if not inside:
        raise ValueError(
            f"Input {media_kind} path {candidate!r} is outside --allowed-local-media-path {allowed_root!r}"
        )
    if not os.path.isfile(real_path):
        raise ValueError(f"Input {media_kind} not found: {candidate!r}")
    return real_path


def _resolve_task_media_paths(paths: list[str], allowed_root: str | None, *, media_kind: str) -> list[str]:
    """Confine a list of facade-injected paths, returning the resolved real paths."""
    return [_resolve_task_image_path(path, allowed_root, media_kind=media_kind) for path in paths]


def _task_allowed_media_root(app_state: Any, engine_client: Any) -> str:
    """Resolve the ``--allowed-local-media-path`` whitelist for a task endpoint.

    Read the CLI args FIRST: a pure-diffusion server (single diffusion stage) has
    no vllm_config, so ``engine_client.model_config`` either does not exist or does
    not carry ``allowed_local_media_path``, and reading only from there makes every
    path input fail closed even when the operator did pass the flag.
    """
    allowed_root = getattr(getattr(app_state, "args", None), "allowed_local_media_path", "") or ""
    if not allowed_root:
        model_config = getattr(engine_client, "model_config", None)
        allowed_root = getattr(model_config, "allowed_local_media_path", "") or ""
    return allowed_root


def _confine_task_output_path(path: str, allowed_roots: list[str]) -> str:
    """Reject an image-task output path that escapes the server's write roots.

    SECURITY, symmetric with :func:`_resolve_task_image_path`. ``save_result_path``
    is caller-controlled, ``resolve_save_path`` keeps an absolute value verbatim,
    and ``atomic_write_bytes`` creates parent directories and ``os.replace``\\ s the
    target — so an unconfined path is an arbitrary file write for anyone who can
    reach this engine directly through the GPUStack model proxy, not just the
    facade that normally dictates it.

    Relative paths were already resolved under the storage root by the caller, so
    only absolute ones need checking.
    """
    real_path = os.path.realpath(path)
    for root in allowed_roots:
        if not root:
            continue
        real_root = os.path.realpath(root)
        try:
            # commonpath, not startswith: "/nfs-output-evil" must not pass for
            # root "/nfs-output".
            if os.path.commonpath([real_root, real_path]) == real_root:
                return path
        except ValueError:  # different drives / mixed absolute-relative
            continue
    raise ValueError(
        f"save_result_path {path!r} is outside the permitted output roots {allowed_roots}. "
        "Use a relative path (resolved under the storage root) or a path under "
        "--allowed-local-media-path."
    )


def _load_task_images(paths: list[str], allowed_root: str | None, *, normalize_rgb: bool) -> list[Image.Image]:
    """Load facade-materialized inputs from disk into PIL images."""
    images: list[Image.Image] = []
    for raw in paths:
        resolved = _resolve_task_image_path(raw, allowed_root)
        try:
            img = Image.open(resolved)
            img.load()
        except Exception as exc:
            raise ValueError(f"Failed to open input image {raw!r}: {exc}") from exc
        images.append(img.convert("RGB") if normalize_rgb else img)
    return images


def _encode_images_png_b64(images: list[Image.Image]) -> list[str]:
    encoded: list[str] = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode())
    return encoded


@asynccontextmanager
async def _terminal_engine_awake(app_state: Any, *, request_id: str) -> AsyncIterator[None]:
    """Hold the residency mutex with the terminal engine awake, if residency is on.

    A no-op for ordinary single-engine servers.

    Under ``--residency-config`` every engine is parked asleep at boot, so ANY
    route that reaches ``engine_client.generate()`` directly — /v1/images/generations,
    /v1/images/edits, chat — would otherwise be rejected by AsyncOmni's
    sleeping-stage guard. Wrapping those call sites keeps them serving instead of
    failing, at the cost of a wake/sleep per request.

    Note these routes do NOT run the AR phase: they only wake the terminal
    (diffusion) engine. A ``bot_task`` asking for chain-of-thought therefore has
    no effect here; use /v1/tasks/image/ for the two-phase path.
    """
    bundle = getattr(app_state, "residency_bundle", None)
    if bundle is None:
        yield
        return
    async with bundle.residency.session(request_id=request_id) as session:
        async with session.awake(bundle.deployment.terminal.label):
            yield


# Keys that belong in gen_params.extra_args (prompt shaping the DiT reads from
# there), never as sampling-param attributes.
_EXTRA_ARGS_ONLY_KEYS = frozenset({"bot_task", "sys_type", "system_prompt", "use_system_prompt"})

# Keys the route has already resolved onto gen_params by hand; re-applying them
# from extra_body would undo request-level precedence (e.g. the AR bridge's
# width/height beating the caller's, or a generated seed).
_ROUTE_RESOLVED_PARAM_KEYS = frozenset(
    {
        "width",
        "height",
        "seed",
        "num_outputs_per_prompt",
        "num_inference_steps",
        "guidance_scale",
        "true_cfg_scale",
        "strength",
        "negative_prompt",
        "lora",
    }
)


async def _generate_task_images(
    *,
    request: ImageTaskRequest,
    app_state: Any,
    engine_client: Any,
    model_name: str,
    stage_configs: list[Any],
    pil_images: list[Image.Image],
    mask_image: Image.Image | None,
    extra_body: dict[str, Any],
    width: int | None,
    height: int | None,
    seed: int,
    n: int,
    task_id: str,
    bridge: dict[str, Any] | None = None,
) -> list[Image.Image]:
    """Render the images for one task, on whichever engine shape is deployed."""
    if width is not None and height is not None:
        extra_body["width"] = width
        extra_body["height"] = height
    size_str = f"{width}x{height}" if width is not None and height is not None else "auto"

    if len(stage_configs) > 1:
        # Fused multi-stage pipeline (AR + diffusion in one engine, e.g. the
        # two-stage GLM-Image / HunyuanImage-3.0 topology): route through the
        # chat handler so the AR stage gets the right max_tokens and target
        # grid — the same path /v1/images/edits uses.
        chat_handler = getattr(app_state, "openai_serving_chat", None)
        if chat_handler is None:
            raise RuntimeError("openai_serving_chat is not initialized for multi-stage image generation.")
        result = await chat_handler.generate_diffusion_images(
            prompt=request.prompt,
            extra_body=extra_body,
            reference_images=_encode_images_png_b64(pil_images) or None,
            mask_image=_encode_images_png_b64([mask_image])[0] if mask_image is not None else None,
            # The engine request id IS the task id, with no decoration: that is
            # what /v1/tasks/{id}/status looks progress up by (AsyncOmni.
            # get_progress matches the external id exactly), so a prefix here
            # would silently make image tasks unpollable.
            request_id=task_id,
            stream=False,
            model=model_name,
            output_format="png",
            size=size_str,
            # None by design: the job outlives its HTTP request. The handler
            # guards every raw_request use, and the only unguarded one is on the
            # streaming branch, which stream=False never reaches.
            raw_request=None,
        )
        if isinstance(result, ErrorResponse):
            raise RuntimeError(getattr(getattr(result, "error", None), "message", None) or str(result))
        images, _, _, _ = result
        return list(images)

    # Single diffusion stage. This also covers the terminal engine of an
    # exclusive-residency deployment, where AR already ran as its own engine and
    # its results arrived through extra_body.
    # Seed from the ENGINE's per-stage defaults (the deploy YAML), not a blank
    # object. _generate_with_async_omni feeds this to
    # build_stage_sampling_params_list(..., replace_diffusion_params=True), which
    # REPLACES a diffusion stage's defaults with whatever we hand it rather than
    # overlaying — so any field left unset here is lost, not inherited. Starting
    # blank silently drops the shipped 8-step Distil default and lands on the
    # pipeline's 50-step fallback: a 6x latency blowup for a request that simply
    # omitted num_inference_steps.
    stage_defaults = get_default_sampling_params_list(engine_client)
    if stage_defaults:
        gen_params = clone_sampling_params(stage_defaults[0])
    else:
        gen_params = OmniDiffusionSamplingParams()
    # Legacy CLI-level defaults (--default-sampling-params JSON) still apply on
    # top; this is a different source from the deploy YAML above.
    apply_stage_default_sampling_params(
        getattr(getattr(app_state, "args", None), "default_sampling_params", None),
        gen_params,
        "0",
    )
    _update_if_not_none(gen_params, "num_outputs_per_prompt", n)
    _update_if_not_none(gen_params, "width", width)
    _update_if_not_none(gen_params, "height", height)
    _update_if_not_none(gen_params, "seed", seed)
    for field_name in ("num_inference_steps", "guidance_scale", "true_cfg_scale", "strength"):
        _update_if_not_none(gen_params, field_name, getattr(request, field_name, None))
    # A caller-supplied guidance_scale of 0 means "no CFG", but 0 is falsey:
    # OmniDiffusionRequest.__post_init__ rewrites a falsey guidance_scale to 1.0
    # and leaves guidance_scale_provided False, after which HunyuanImage3
    # substitutes 5.0 (pipeline_hunyuan_image3.py) and each step runs TWO backbone
    # forwards. Flag anything the caller sent explicitly — including 0 — as
    # provided, so the value survives instead of being read as "unset".
    if request.guidance_scale is not None and hasattr(gen_params, "guidance_scale_provided"):
        gen_params.guidance_scale_provided = True
    # Per-request LoRA needs parsing into lora_request/lora_scale, not a raw
    # attribute copy — same handling the synchronous edit endpoint applies.
    lora_body = extra_body.get("lora")
    if isinstance(lora_body, dict) and lora_body:
        lora_request, lora_scale = _parse_lora_request(lora_body)
        _update_if_not_none(gen_params, "lora_request", lora_request)
        _update_if_not_none(gen_params, "lora_scale", lora_scale)

    # Sweep the remaining diffusion knobs the caller sent that this route does not
    # name explicitly (generator_device, layers, resolution, num_frames,
    # guidance_scale_2, and anything added later). The multi-stage path gets these
    # for free because it hands extra_body to the shared builder; without this the
    # single-stage branch — which is the residency deployment's terminal engine —
    # would silently drop them.
    for key, value in extra_body.items():
        if value is None or key in _EXTRA_ARGS_ONLY_KEYS or key in _ROUTE_RESOLVED_PARAM_KEYS:
            continue
        if hasattr(gen_params, key):
            setattr(gen_params, key, value)

    extra_args = dict(getattr(gen_params, "extra_args", {}) or {})
    # Reuse the sync endpoint's mapping rather than copying fields by name: the
    # DiT pipeline reads extra_args["use_system_prompt"], but callers send
    # "sys_type". A plain copy loop drops sys_type entirely, and also misses the
    # bot_task -> sys_type fallback that decides the system prefix when the
    # caller named only a prompt mode.
    extra_args.update(
        _build_hunyuan_edit_extra_args(
            bot_task=request.bot_task,
            sys_type=request.sys_type,
            system_prompt=request.system_prompt,
        )
    )
    # Values already resolved upstream win — notably the AR bridge's
    # use_system_prompt, which is the prefix AR actually conditioned on.
    for key in ("use_system_prompt", "system_prompt", "bot_task"):
        if extra_body.get(key) is not None:
            extra_args[key] = extra_body[key]
    if extra_args:
        gen_params.extra_args = extra_args

    prompt: OmniTextPrompt = {"prompt": request.prompt, "modalities": ["image"]}
    if request.negative_prompt is not None:
        prompt["negative_prompt"] = request.negative_prompt
    multi_modal: dict[str, Any] = {}
    if pil_images:
        multi_modal["image"] = pil_images
    if mask_image is not None:
        multi_modal["mask_image"] = mask_image
    if multi_modal:
        prompt["multi_modal_data"] = multi_modal

    if bridge:
        # ar2diffusion() returns the diffusion stage's PROMPT DICT, not loose
        # kwargs, and the pipeline reads the chain-of-thought from
        # prompt["extra"]["ar_generated_text"] (pipeline_hunyuan_image3.py).
        # Routing it through gen_params.extra_args instead silently drops it: the
        # AR phase runs, costs its full latency, and the image comes out
        # byte-identical to one generated without AR at all.
        bridge_extra = bridge.get("extra")
        if isinstance(bridge_extra, dict) and bridge_extra:
            prompt["extra"] = {**(prompt.get("extra") or {}), **bridge_extra}
        # These live at the TOP level of the bridge's dict; forwarding them keeps
        # the DiT's system prefix identical to the one AR conditioned on.
        for key in ("use_system_prompt", "system_prompt"):
            if bridge.get(key) is not None:
                prompt[key] = bridge[key]

    gen_result = await _generate_with_async_omni(
        engine_client=engine_client,
        gen_params=gen_params,
        stage_configs=stage_configs,
        prompt=prompt,
        # Bare task id, same reason as the multi-stage branch above.
        request_id=task_id,
    )
    return _extract_images_from_result(gen_result)


async def _run_ar_phase(
    ar_engine: Any,
    request: ImageTaskRequest,
    *,
    pil_images: list[Image.Image],
    width: int | None,
    height: int | None,
    request_id: str,
) -> dict[str, Any] | None:
    """Run the autoregressive phase and bridge its output to diffusion inputs.

    Returns the ``ar2diffusion`` dict (target size from the AR-predicted
    ``<img_size_*><img_ratio_*>`` tail, the truncated chain-of-thought, and the
    forwarded system-prompt settings), or None when AR produced nothing.

    The bridge is the SAME function the fused two-stage pipeline registers as its
    ``custom_process_input_func`` (see the HunyuanImage3 pipeline topology), so
    the standalone AR->DiT handoff cannot drift from the fused one. The
    DiT-only pipeline deliberately declares no bridge, which is why the caller
    has to invoke it here.
    """
    from vllm import SamplingParams

    from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
        build_prompt_tokens,
        resolve_stop_token_ids,
    )
    from vllm_omni.model_executor.stage_input_processors.hunyuan_image3 import ar2diffusion

    get_tokenizer = getattr(ar_engine, "get_tokenizer", None)
    tokenizer = await get_tokenizer() if callable(get_tokenizer) else None
    if tokenizer is None:
        raise RuntimeError("AR engine exposes no tokenizer; cannot build the HunyuanImage3 prompt.")

    task = "it2i" if pil_images else "t2i"
    built = build_prompt_tokens(
        request.prompt,
        tokenizer,
        task=task,
        bot_task=request.bot_task,
        sys_type=request.sys_type,
        custom_system_prompt=request.system_prompt,
        num_images=max(1, len(pil_images)),
    )
    stop_token_ids = resolve_stop_token_ids(task=task, bot_task=request.bot_task, tokenizer=tokenizer)

    ar_prompt: dict[str, Any] = {"prompt_token_ids": list(built.token_ids)}
    if pil_images:
        ar_prompt["multi_modal_data"] = {"image": pil_images}

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop_token_ids=stop_token_ids,
        detokenize=True,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    ar_output = None
    async for output in ar_engine.generate(ar_prompt, sampling_params, request_id=f"{request_id}-ar"):
        ar_output = output
    if ar_output is None:
        return None

    # The bridge reads the ORIGINAL request prompt for text/size/system-prompt
    # fallbacks, so hand it the same shape the fused orchestrator would.
    original_prompt: dict[str, Any] = {"prompt": request.prompt}
    if height is not None:
        original_prompt["height"] = height
    if width is not None:
        original_prompt["width"] = width
    if request.sys_type is not None:
        original_prompt["use_system_prompt"] = request.sys_type
    if request.system_prompt is not None:
        original_prompt["system_prompt"] = request.system_prompt
    if pil_images:
        original_prompt["multi_modal_data"] = {"image": pil_images}

    return ar2diffusion([ar_output], prompt=original_prompt, requires_multimodal_data=bool(pil_images))


async def _run_image_job(
    request: ImageTaskRequest,
    task_id: str,
    save_result_path: str,
    *,
    app_state: Any,
    engine_client: Any,
    model_name: str,
    stage_configs: list[Any],
    width: int | None,
    height: int | None,
) -> None:
    """Background image job.

    Deliberately takes ``app_state`` and pre-resolved engine handles instead of
    the ``Request``: this coroutine outlives the HTTP response that created it,
    and a Starlette ``Request`` is only valid for its own ASGI cycle. Retaining
    one would keep the finished request's scope (and body) alive for the whole
    job — minutes, for a slow image model — and would leave a trap for any future
    code that awaits on it (``is_disconnected()``, re-reading the body) inside a
    connection that is already gone. The TTS/audiogen jobs avoid this the same
    way, passing ``raw_request=None`` down to the handler.
    """
    job = await AUDIO_TASK_STORE.get(task_id)
    if job is None:
        logger.warning("Image task %s missing before generation started; skipping", task_id)
        return

    await AUDIO_TASK_STORE.update_fields(task_id, {"status": AudioTaskStatus.PROCESSING, "start_time": time.time()})
    try:
        # Normalize inputs to RGB only when the caller opts into Hunyuan-aware
        # behavior, mirroring /v1/images/edits: RGBA/P inputs otherwise diverge
        # from the offline path and can shift AR recaption before DiT runs.
        normalize_rgb = request.bot_task is not None or request.sys_type is not None
        allowed_root = _task_allowed_media_root(app_state, engine_client)
        input_paths = request.input_image_paths()
        # Count first, decode second. The route already rejected an over-limit
        # request at submit; this is defence in depth for any other caller, and
        # keeping the order means a rejected request never pays for opening and
        # decoding files it will not use.
        max_input_images = _get_max_edit_input_images(app_state, engine_client)
        if max_input_images is not None and len(input_paths) > max_input_images:
            raise ValueError(
                f"Received {len(input_paths)} input images; at most {max_input_images} are supported by this model."
            )
        pil_images = await asyncio.to_thread(_load_task_images, input_paths, allowed_root, normalize_rgb=normalize_rgb)

        # The mask's alpha channel carries the edit region, so it must never be
        # RGB-normalized (same rule as /v1/images/edits).
        mask_path = request.mask_image_path()
        mask_image = None
        if mask_path:
            loaded_mask = await asyncio.to_thread(_load_task_images, [mask_path], allowed_root, normalize_rgb=False)
            mask_image = loaded_mask[0]

        # An auto-sized edit has no width/height at submit, so the size check
        # there was a no-op: the OUTPUT size is the first input image's size,
        # which is only knowable once the file is read (the pipeline fills
        # sampling_params.width/height from it). Re-run the guard now — before any
        # GPU work — or a caller can bypass --max-generated-image-size simply by
        # feeding a very large reference image. The synchronous edit endpoint
        # closes the same hole by deriving the size from pil_images[0] first.
        if width is None and height is None and pil_images:
            auto_width, auto_height = pil_images[0].size
            try:
                _check_max_generated_image_size(getattr(app_state, "args", None), auto_width, auto_height)
            except HTTPException as exc:
                raise ValueError(exc.detail) from exc

        # width/height otherwise came from submit, already size-checked; do not
        # recompute here, so the validated value is the one actually used.
        seed = request.seed if request.seed is not None else random.randint(0, MAX_UINT32_SEED)
        n = request.n or 1

        extra_body = request.diffusion_extra_body()
        extra_body["seed"] = seed
        extra_body["num_outputs_per_prompt"] = n

        # Exclusive-residency deployments hold AR and diffusion as separate
        # engines and wake at most one at a time. The AR phase runs only when the
        # requested bot_task actually asks for chain-of-thought / recaption, so a
        # single deployment serves both the fast path (AR never woken) and the
        # quality path — decided per request, not per deployment.
        bundle: ResidencyBundle | None = getattr(app_state, "residency_bundle", None)
        bridged: dict[str, Any] | None = None
        async with AsyncExitStack() as residency_stack:
            if bundle is not None:
                from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import requires_ar_generation

                # One session spans BOTH phases, so the mutex is held for the
                # whole request and no other request can wake an engine midway.
                session = await residency_stack.enter_async_context(bundle.residency.session(request_id=task_id))
                ar_label = bundle.label_of("ar")
                if ar_label is not None and requires_ar_generation(request.bot_task):
                    async with session.awake(ar_label) as ar_group:
                        bridged = await _run_ar_phase(
                            ar_group.engine,
                            request,
                            pil_images=pil_images,
                            width=width,
                            height=height,
                            request_id=task_id,
                        )
                    if bridged:
                        # AR's <img_ratio_*> is the canonical target shape: it
                        # beats the request-side size, which the serving layer
                        # fills from the first reference image's bucket and so
                        # collapses non-square targets to square on multi-image
                        # requests.
                        width = bridged.get("width", width)
                        height = bridged.get("height", height)
                        # The AR stage OVERRIDES the size validated at submit:
                        # reso_group[ratio_idx] comes from the model's own
                        # <img_size_*><img_ratio_*> tokens and can be larger than
                        # anything the caller asked for (1024 is a common
                        # prediction). Without re-checking, the quality path lets
                        # a request that passed the cap at submit generate above
                        # it. This is the last point where the size is known
                        # before the diffusion engine wakes.
                        try:
                            _check_max_generated_image_size(getattr(app_state, "args", None), width, height)
                        except HTTPException as exc:
                            raise ValueError(f"AR-predicted output size rejected: {exc.detail}") from exc
                        # The rest of the bridge output (the CoT under "extra",
                        # the system-prompt settings) is applied to the diffusion
                        # PROMPT, not to extra_body — see _generate_task_images.
                        for key in ("use_system_prompt", "system_prompt"):
                            if key in bridged:
                                extra_body[key] = bridged[key]
                # AR is asleep again by now (awake() sleeps on the way out), so
                # the diffusion engine can take the whole card.
                await residency_stack.enter_async_context(session.awake(bundle.deployment.terminal.label))

            images = await _generate_task_images(
                request=request,
                app_state=app_state,
                engine_client=engine_client,
                model_name=model_name,
                stage_configs=stage_configs,
                pil_images=pil_images,
                mask_image=mask_image,
                extra_body=extra_body,
                width=width,
                height=height,
                seed=seed,
                n=n,
                task_id=task_id,
                bridge=bridged,
            )

        if not images:
            raise RuntimeError("Image generation returned no images")
        # The facade dictates ONE output path per task (its _output_ext gives
        # ".png"), so extra images from n>1 have nowhere to go. Write the first
        # and say so rather than silently dropping them.
        if len(images) > 1:
            logger.warning(
                "Image task %s produced %d images but save_result_path holds one; writing the first.",
                task_id,
                len(images),
            )
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        await atomic_write_bytes(buf.getvalue(), save_result_path)
        logger.info("Image task %s wrote %s", task_id, save_result_path)
        await AUDIO_TASK_STORE.update_fields(task_id, {"status": AudioTaskStatus.COMPLETED, "end_time": time.time()})
    except asyncio.CancelledError:
        await AUDIO_TASK_STORE.update_fields(task_id, {"status": AudioTaskStatus.CANCELLED, "end_time": time.time()})
        raise
    except (EngineGenerateError, EngineDeadError) as exc:
        logger.exception("Image task %s failed (engine error)", task_id)
        await AUDIO_TASK_STORE.update_fields(
            task_id,
            {
                "status": AudioTaskStatus.FAILED,
                "end_time": time.time(),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        if isinstance(exc, EngineDeadError):
            terminate_if_errored(server=app_state.server, engine=app_state.engine_client)
    except Exception as exc:
        logger.exception("Image task %s failed", task_id)
        await AUDIO_TASK_STORE.update_fields(
            task_id,
            {
                "status": AudioTaskStatus.FAILED,
                "end_time": time.time(),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )


@router.post(
    "/v1/tasks/image/",
    responses={
        HTTPStatus.OK.value: {"model": AudioTaskResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
    },
)
async def create_image_task(request: ImageTaskRequest, raw_request: Request) -> AudioTaskResponse:
    """Submit an asynchronous image generation / editing task.

    Returns immediately with a PENDING record; poll the shared global status /
    result / cancel endpoints (keyed by task_id) exactly like the TTS and
    diffusion-audio async APIs.
    """
    if not (request.prompt or "").strip():
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="Empty prompt")

    # Resolve the engine handles HERE, while the request is still alive: a
    # misrouted model then fails at submit with a proper status instead of
    # minutes later inside the job, and the job never needs the Request itself.
    engine_client, model_name, stage_configs = _get_engine_and_model(raw_request)
    app_state = raw_request.app.state

    # The facade strips `model` (control key), so absent means "this server's
    # model". A DIFFERENT name is a misrouted request: /v1/images/edits rejects
    # it, and silently serving it here would return an image from a model the
    # caller did not ask for.
    if not (request.model or "").strip():
        request.model = model_name
    elif request.model != model_name:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Model mismatch: request specifies '{request.model}' but server is running '{model_name}'.",
        )

    # Resolve the output size at SUBMIT time, for two reasons: the caller gets a
    # real 400 instead of a task that fails minutes later, and the rejection
    # happens BEFORE reserve() takes a queue slot and the job burns GPU. The
    # synchronous image endpoints run the same check.
    # Layered-model geometry, validated exactly as /v1/images/edits does.
    if request.resolution is not None and request.resolution not in SUPPORTED_LAYERED_RESOLUTIONS:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Invalid resolution {request.resolution}. Supported resolutions: {SUPPORTED_LAYERED_RESOLUTIONS}.",
        )
    try:
        validate_layered_layers(request.layers)
    except ValueError as exc:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(exc)) from exc
    if request.resolution is not None and (request.target_shape or request.aspect_ratio):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "Cannot specify both 'resolution' and an explicit size. Use 'resolution' alone, "
                "or use 'target_shape'/'aspect_ratio' without 'resolution'."
            ),
        )

    app_state_args = getattr(app_state, "args", None)
    max_pixels = getattr(app_state_args, "max_generated_image_size", None)
    width, height = request.output_size(max_pixels=max_pixels)
    # Pass `resolution` through: the size check has a resolution-only branch
    # (output is resolution x resolution), so omitting it lets a resolution-only
    # request slip past --max-generated-image-size entirely.
    _check_max_generated_image_size(app_state_args, width, height, request.resolution)

    # Validate bot_task at SUBMIT time. An unrecognized value would otherwise
    # survive all the way into the job: requires_ar_generation() answers True for
    # anything it does not know, so a residency deployment WAKES the AR engine
    # first and only then dies inside build_prompt_tokens with "Unknown bot_task",
    # far from the cause and after paying a wake. Fail here with the valid set
    # instead. (Note "image" is an upstream-HF spelling, not one of ours — the
    # equivalent here is "vanilla".)
    if request.bot_task is not None:
        from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import available_bot_tasks

        valid_bot_tasks = available_bot_tasks()
        if request.bot_task not in valid_bot_tasks:
            named = [value for value in valid_bot_tasks if value is not None]
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=(
                    f"Unknown bot_task {request.bot_task!r}. Valid values: {named} "
                    "(or omit the field). Use 'vanilla' for the no-AR fast path."
                ),
            )

    # A mask without a base image is not an edit; catch it here rather than
    # letting the pipeline receive a mask it has nothing to apply to.
    input_paths = request.input_image_paths()
    if request.mask_image_path() and not input_paths:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="image_mask_path requires at least one input image.",
        )

    # Reject an over-limit image count HERE: before the job opens and decodes
    # every referenced file, and before reserve() takes a queue slot. The
    # synchronous edit endpoint rejects on count for the same reason. Doing it at
    # submit also turns a task that would have been recorded FAILED minutes later
    # into a plain 400 for the caller.
    max_input_images = _get_max_edit_input_images(app_state, engine_client)
    if max_input_images is not None and len(input_paths) > max_input_images:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=too_many_input_images_message(len(input_paths), max_input_images),
        )

    task_id = request.task_id or f"image_task_{random_uuid()}"
    save_result_path = resolve_save_path(request.save_result_path, task_id, STORAGE_MANAGER.storage_path, ".png")
    # Confine the write target. The facade dictates a path under its own NFS
    # output root (which the GPUStack backend also passes as
    # --allowed-local-media-path), so both roots are legitimate; anything else is
    # a direct caller trying to write outside them.
    try:
        save_result_path = _confine_task_output_path(
            save_result_path,
            [
                STORAGE_MANAGER.storage_path,
                getattr(app_state_args, "allowed_local_media_path", "") or "",
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(exc)) from exc

    try:
        ref = await AUDIO_TASK_MANAGER.reserve(task_id, save_result_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE.value, detail=str(exc)) from exc

    task = asyncio.create_task(
        _run_image_job(
            request,
            task_id,
            save_result_path,
            app_state=app_state,
            engine_client=engine_client,
            model_name=model_name,
            stage_configs=stage_configs,
            width=width,
            height=height,
        )
    )
    await AUDIO_TASKS.upsert(task_id, task)
    return ref


# ---------------------------------------------------------------------------
# Async video task API (text-to-video / keyframe / reference-driven video).
#
# Fourth sibling of the TTS, diffusion-audio and image async APIs above, sharing
# their task store / manager / registry so the global status, result, cancel and
# queue endpoints below serve video tasks unchanged — only submit is new.
#
# Same reason for existing as /v1/tasks/image/: the GPUStack facade dispatches
# strictly by engine kind, POSTing to ``v1/tasks/{kind}/`` (gpustack
# routes/videos.py), and every task_type that is not image/audio/music/audiogen
# falls through to kind "video". POST /v1/videos and /v1/videos/sync cannot serve
# it — they are multipart-only and take references as uploaded bytes or URLs,
# while the facade sends JSON whose media inputs are absolute NFS paths it has
# already materialized. Those two stay the direct-caller surface.
# ---------------------------------------------------------------------------


async def _run_video_task_job(
    handler: OmniOpenAIServingVideo,
    request: VideoGenerationRequest,
    task_id: str,
    save_result_path: str,
    *,
    image_paths: list[str],
    video_paths: list[str],
    audio_paths: list[str],
    allowed_root: str,
    app_state: Any,
) -> None:
    """Render one video task and write the MP4 to ``save_result_path``.

    Mirrors ``_run_image_job``: the route resolved every handle while its HTTP
    request was alive, so this coroutine outlives it safely.

    The reference dataclasses are built with EMPTY ``cleanup_paths`` and
    ``_cleanup_video_references`` is deliberately never called here. Those inputs
    are the facade's NFS files, not temp copies we made — and for audio the
    cleanup helper falls back to deleting ``reference_audio.path`` itself when
    ``cleanup_paths`` is empty, which would erase the caller's uploaded reference
    the first time a task ran. The multipart path needs that helper because it
    spools uploads to /tmp; this path materializes nothing.
    """
    job = await AUDIO_TASK_STORE.get(task_id)
    if job is None:
        logger.warning("Video task %s missing before generation started; skipping", task_id)
        return

    await AUDIO_TASK_STORE.update_fields(task_id, {"status": AudioTaskStatus.PROCESSING, "start_time": time.time()})
    try:
        reference_image: ReferenceImage | None = None
        if image_paths:
            # RGB-normalized to match the multipart path's H3 handling
            # (_persist_uploaded_media_references converts every reference image),
            # so a PNG with alpha behaves identically whichever surface sent it.
            images = await asyncio.to_thread(_load_task_images, image_paths, allowed_root, normalize_rgb=True)
            reference_image = ReferenceImage(data=images if len(images) > 1 else images[0])
        # Videos and audio stay as PATHS: the reference encoders need the original
        # container streams (a video's soundtrack included), which is also what the
        # multipart path hands over once it has spooled the upload to disk.
        reference_video = ReferenceVideo(data=list(video_paths)) if video_paths else None
        reference_audio = (
            ReferenceAudio(path=audio_paths if len(audio_paths) > 1 else audio_paths[0]) if audio_paths else None
        )

        video_bytes, _stage_durations, _peak_memory_mb, _action = await handler.generate_video_bytes(
            request,
            task_id,
            reference_image=reference_image,
            reference_video=reference_video,
            reference_audio=reference_audio,
        )
        if not video_bytes:
            # generate_video_bytes returns b"" for action-only models, which have
            # no MP4 to persist. Writing a 0-byte file would report COMPLETED and
            # hand the facade an unplayable artifact.
            raise RuntimeError("Video generation returned no video bytes for this model.")

        # The engine request is over, so its live progress is gone; the write to
        # shared NFS is the last visible phase and happens right here.
        await AUDIO_TASK_STORE.update_fields(task_id, {"phase": PHASE_SAVE, "phase_progress": 0.0})
        await atomic_write_bytes(video_bytes, save_result_path)
        logger.info("Video task %s wrote %s", task_id, save_result_path)
        await AUDIO_TASK_STORE.update_fields(
            task_id,
            {
                "status": AudioTaskStatus.COMPLETED,
                "end_time": time.time(),
                # Push the last phase to its end. The status endpoint only
                # overlays live progress while PROCESSING, so without this a
                # COMPLETED task keeps returning the save/0.0 written above and
                # the facade folds that into a global percentage below 100.
                "phase_progress": 100.0,
            },
        )
    except asyncio.CancelledError:
        await AUDIO_TASK_STORE.update_fields(task_id, {"status": AudioTaskStatus.CANCELLED, "end_time": time.time()})
        raise
    except (EngineGenerateError, EngineDeadError) as exc:
        logger.exception("Video task %s failed (engine error)", task_id)
        await AUDIO_TASK_STORE.update_fields(
            task_id,
            {
                "status": AudioTaskStatus.FAILED,
                "end_time": time.time(),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        if isinstance(exc, EngineDeadError):
            terminate_if_errored(server=app_state.server, engine=app_state.engine_client)
    except Exception as exc:
        logger.exception("Video task %s failed", task_id)
        await AUDIO_TASK_STORE.update_fields(
            task_id,
            {
                "status": AudioTaskStatus.FAILED,
                "end_time": time.time(),
                # Keep OmniClientError's own error_type ("BadRequestError") rather
                # than flattening every failure to a class name: the facade copies
                # error_type through verbatim, and it is the only thing left that
                # distinguishes "the caller sent an illegal parameter" from "the
                # engine broke" once the submit call has already returned 200.
                "error": str(exc),
                "error_type": getattr(exc, "error_type", None) or type(exc).__name__,
            },
        )


@router.post(
    "/v1/tasks/video/",
    responses={
        HTTPStatus.OK.value: {"model": AudioTaskResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
    },
)
async def create_video_task(request: VideoTaskRequest, raw_request: Request) -> AudioTaskResponse:
    """Submit an asynchronous video generation task.

    Returns immediately with a PENDING record; poll the shared global status /
    result / cancel endpoints (keyed by task_id) exactly like the TTS,
    diffusion-audio and image async APIs.

    Everything knowable without the GPU is checked HERE — prompt, model identity,
    param ranges, media paths — so the caller gets a real 400 instead of a task
    that fails minutes later, and so a rejected request never takes a queue slot.
    Model-specific validation (MiniMax-H3's duration window, frame_indices arity,
    checkpoint-partition support) belongs to the pipeline and surfaces on the task
    record as FAILED with the engine's own message.
    """
    if not (request.prompt or "").strip():
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail="Empty prompt")

    # Reject byte/URL references outright. They are the multipart endpoints'
    # input model; here they would be silently ignored, which is the exact
    # failure mode that makes a misrouted request look like a model bug.
    unsupported = request.unsupported_reference_keys()
    if unsupported:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                f"{', '.join(unsupported)} not supported by /v1/tasks/video/; it takes server paths "
                "(image_path / last_frame_path / video_path / audio_path). Use POST /v1/videos for "
                "uploaded bytes or URL references."
            ),
        )

    # `reference_order` is derived from `references` by this route, so a
    # hand-written one has nothing keeping it consistent with the media that
    # arrived. Left to ride through it would first be read inside the job, after
    # reserve(), which is the late failure this route exists to prevent.
    if request.supplies_route_derived_order():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "reference_order is derived from `references` and cannot be supplied directly by "
                "/v1/tasks/video/. Send `references` as an ordered list of {type, path} entries; the "
                "order of that list IS the reference order."
            ),
        )

    # Resolve the handler HERE, while the request is still alive: a misrouted
    # model then fails at submit with a proper status instead of minutes later
    # inside the job, and the job never needs the Request itself.
    handler = Omnivideo(raw_request)
    if handler is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Video generation handler not initialized.",
        )
    app_state = raw_request.app.state
    app_model_name, app_stage_configs = _resolve_video_runtime_context(raw_request)
    model_name = handler.model_name or app_model_name or request.model or "unknown"
    # The facade strips `model` (control key), so absent means "this server's
    # model". A DIFFERENT name is a misrouted request: serving it silently would
    # return a video from a model the caller did not ask for. Same check
    # _parse_video_form runs for the multipart endpoints.
    if not (request.model or "").strip():
        request.model = model_name
    elif request.model != model_name:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Model mismatch: request specifies '{request.model}' but server is running '{model_name}'.",
        )
    handler.set_stage_configs_if_missing(app_stage_configs)

    # Range/format checks on the generation params happen in this constructor —
    # VideoGenerationRequest is the one schema that types them, so a bad
    # num_inference_steps or a malformed size is a 400 here rather than a
    # ValidationError escaping as a 500.
    try:
        video_request = request.to_video_request()
    except ValidationError as exc:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(exc)) from exc

    engine_client = getattr(app_state, "engine_client", None)
    allowed_root = _task_allowed_media_root(app_state, engine_client)
    try:
        image_paths = _resolve_task_media_paths(request.reference_image_paths(), allowed_root, media_kind="image")
        video_paths = _resolve_task_media_paths(request.reference_video_paths(), allowed_root, media_kind="video")
        audio_paths = _resolve_task_media_paths(request.reference_audio_paths(), allowed_root, media_kind="audio")
    except ValueError as exc:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(exc)) from exc

    # Knowable here, so it must not become a FAILED task minutes later.
    # _run_and_extract raises the same 400, but by then reserve() has spent a
    # queue slot and the facade only sees error_type="HTTPException" instead of
    # the BadRequestError that tells it the CALLER was wrong. The check there
    # stays: it also guards the multipart /v1/videos surface.
    conflicting = request.conflicting_reference_inputs()
    if conflicting:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "`references` carries its own order and cannot be combined with "
                f"{', '.join(conflicting)}. Send one or the other: resolving a precedence rule "
                "silently produces a different video than the caller asked for."
            ),
        )

    if image_paths and video_paths and not bool(getattr(handler, "supports_mixed_reference_inputs", False)):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="This diffusion model does not support mixed image and video references.",
        )

    # This route MANUFACTURES `reference_order` from `references`, and an
    # instance whose contract cannot honour one rejects it in the pipeline —
    # i.e. after reserve(), on the task record, for the most ordinary request
    # there is. Note what the check above it guards: a *caller-written* order is
    # already a 400 here, so the route knew this field was dangerous on this
    # surface and validated only the half it did not create itself.
    if request.reference_order() and not bool(getattr(handler, "honours_explicit_reference_order", True)):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "This instance canonicalizes references by modality and cannot honour the order of a "
                "`references` list. Send the bucketed fields (image_path / last_frame_path / video_path / "
                "audio_path), or deploy the instance with an inference contract that carries reference order."
            ),
        )

    task_id = request.task_id or f"video_task_{random_uuid()}"
    save_result_path = resolve_save_path(request.save_result_path, task_id, STORAGE_MANAGER.storage_path, ".mp4")
    # Confine the write target. The facade dictates a path under its own NFS
    # output root (which the GPUStack backend also passes as
    # --allowed-local-media-path), so both roots are legitimate; anything else is
    # a direct caller trying to write outside them.
    try:
        save_result_path = _confine_task_output_path(
            save_result_path,
            [
                STORAGE_MANAGER.storage_path,
                allowed_root,
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(exc)) from exc

    try:
        ref = await AUDIO_TASK_MANAGER.reserve(task_id, save_result_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE.value, detail=str(exc)) from exc

    task = asyncio.create_task(
        _run_video_task_job(
            handler,
            video_request,
            task_id,
            save_result_path,
            image_paths=image_paths,
            video_paths=video_paths,
            audio_paths=audio_paths,
            allowed_root=allowed_root,
            app_state=app_state,
        )
    )
    await AUDIO_TASKS.upsert(task_id, task)
    return ref


@router.get("/v1/tasks/", response_model=list[AudioTaskResponse])
async def list_audio_tasks() -> list[AudioTaskResponse]:
    return await AUDIO_TASK_MANAGER.list_tasks()


_LIVE_PROGRESS_GAP_LOGGED = False


def _warn_once_without_live_progress(engine_client: Any) -> None:
    """Say once that this deployment cannot report queueing or live progress.

    Both stage clients implement the two hooks (inline off the engine,
    out-of-process off its pumped state), so this normally stays quiet; it fires
    for a diffusion client that has neither. Such a client answers None to every
    lookup — indistinguishable, from the caller's side, from "nothing to report
    for this request". Left silent it reads as a broken progress bar; said once
    at the first poll that wanted it, it reads as the known limitation it is.

    Deliberately quiet when the engine has no such probe at all (plain TTS or AR
    deployments): they have no diffusion stage to report on, so there is no gap.
    """
    global _LIVE_PROGRESS_GAP_LOGGED
    if _LIVE_PROGRESS_GAP_LOGGED:
        return
    supports = getattr(engine_client, "supports_live_progress", None)
    if supports is None or supports():
        return
    _LIVE_PROGRESS_GAP_LOGGED = True
    logger.warning(
        "Task status: this deployment's diffusion stage client reports neither "
        "execution state nor live phase. Accepted-but-queued tasks will report "
        "'processing' and carry no phase; downstream progress falls back to an "
        "elapsed-time estimate."
    )


# Declared before the parametrized /v1/tasks/{task_id}/status so the literal
# "queue/status" is not captured as task_id="queue".
@router.get("/v1/tasks/queue/status")
async def get_audio_queue_status(raw_request: Request) -> JSONResponse:
    # Hand the manager the same execution probe the per-task endpoint uses, so
    # a job reported as pending there is not simultaneously this endpoint's
    # current_task.
    engine_client = getattr(raw_request.app.state, "engine_client", None)
    return JSONResponse(
        content=await AUDIO_TASK_MANAGER.queue_status(getattr(engine_client, "is_request_executing", None))
    )


@router.get("/v1/tasks/{task_id}/status", response_model=AudioTaskResponse)
async def get_audio_task_status(task_id: str, raw_request: Request) -> AudioTaskResponse:
    job = await AUDIO_TASK_MANAGER.get_status(task_id)
    if job is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="Task not found")
    if job.status == AudioTaskStatus.PROCESSING:
        # Accepted != running: several jobs are admitted at once and the
        # scheduler serializes them, so a job still waiting its turn reports as
        # pending (see visible_task_status for why that matters downstream).
        engine_client = getattr(raw_request.app.state, "engine_client", None)
        _warn_once_without_live_progress(engine_client)
        is_executing = getattr(engine_client, "is_request_executing", None)
        visible = visible_task_status(job.status, is_executing(task_id) if is_executing else None)
        if visible != job.status:
            return job.model_copy(update={"status": visible})

        # Live phase, read straight off the engine rather than stored on the job:
        # the pipeline runs in a worker process and pushes progress to the
        # executor asynchronously, so the freshest value is always the one we
        # pull at poll time. Every async task submits its engine request under
        # the bare task id — video, image, TTS and audiogen alike — so no side
        # mapping is needed. Keep it that way when adding a task type: this
        # lookup matches the external id exactly, so any decoration on the
        # submit side silently costs that task type its progress.
        progress = getattr(engine_client, "get_progress", None)
        live = progress(task_id) if progress is not None else None
        if live:
            job = job.model_copy(
                update={
                    "phase": live.get("phase"),
                    "phase_progress": live.get("phase_progress"),
                }
            )
    return job


@router.get("/v1/tasks/{task_id}/result")
async def get_audio_task_result(task_id: str) -> Response:
    job = await AUDIO_TASK_MANAGER.get_status(task_id)
    if job is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="Task not found")
    if job.status == AudioTaskStatus.FAILED:
        raise HTTPException(status_code=422, detail=f"Audio task failed: {job.error}")
    if job.status != AudioTaskStatus.COMPLETED or not job.save_result_path:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="Result not ready")
    if not os.path.exists(job.save_result_path):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="Result file not found on disk")
    return FileResponse(path=job.save_result_path, filename=os.path.basename(job.save_result_path))


@router.delete("/v1/tasks/{task_id}")
async def delete_audio_task(task_id: str) -> JSONResponse:
    cancelled = await AUDIO_TASK_MANAGER.cancel(task_id)
    if not cancelled and await AUDIO_TASK_MANAGER.get_status(task_id) is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="Task not found")
    return JSONResponse(content={"task_id": task_id, "cancelled": cancelled})


@profiler_router.post("/start_profile")
async def start_profile(raw_request: Request, request: ProfileRequest | None = None):
    """Start profiling for the engine.

    Args:
        request: Optional request body with stages to profile.
            - stages: List of stage IDs to profile. If None, profiles all stages.

    Example:
        POST /start_profile
        {"stages": [0, 1]}  # Profile only stages 0 and 1
    """
    try:
        stages = request.stages if request else None
        logger.info("Starting profiler for stages: %s", stages if stages else "all")
        engine_client = raw_request.app.state.engine_client
        await engine_client.start_profile(stages=stages)
        logger.info("Profiler started.")
        return JSONResponse(content={"status": "SUCCESS"})
    except Exception as e:
        logger.exception("Failed to start profiler: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Failed to start profiler: {str(e)}"
        )


@profiler_router.post("/stop_profile")
async def stop_profile(raw_request: Request, request: ProfileRequest | None = None):
    """Stop profiling for the engine.

    Args:
        request: Optional request body with stages to stop profiling.
            - stages: List of stage IDs to stop profiling. If None, stops all stages.

    Example:
        POST /stop_profile
        {"stages": [0, 1]}  # Stop profiling only stages 0 and 1
    """
    try:
        stages = request.stages if request else None
        logger.info("Stopping profiler for stages: %s", stages if stages else "all")
        engine_client = raw_request.app.state.engine_client
        await engine_client.stop_profile(stages=stages)
        logger.info("Profiler stopped.")
        return JSONResponse(content={"status": "SUCCESS"})
    except Exception as e:
        logger.exception("Failed to stop profiler: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=f"Failed to stop profiler: {str(e)}"
        )


@router.post("/v1/omni/sleep")
async def omni_sleep(request: OmniSleepRequest, raw_request: Request):
    _reject_process_local_state_with_multiple_api_workers(raw_request, "Sleep")
    engine_client = raw_request.app.state.engine_client
    sleeping_set = raw_request.app.state.sleeping_stages
    if not hasattr(engine_client, "sleep"):
        raise HTTPException(status_code=501, detail="Engine does not support sleep")
    acks = await engine_client.sleep(stage_ids=request.stage_ids, level=request.level)
    for sid in request.stage_ids:
        sleeping_set.add(sid)
    return {
        "status": "SUCCESS",
        "acks": [dataclasses.asdict(a) if dataclasses.is_dataclass(a) and not isinstance(a, type) else a for a in acks],
    }


@router.post("/v1/omni/wakeup")
async def omni_wakeup(request: OmniWakeupRequest, raw_request: Request):
    _reject_process_local_state_with_multiple_api_workers(raw_request, "Wakeup")
    engine_client = raw_request.app.state.engine_client
    sleeping_set = raw_request.app.state.sleeping_stages
    if not any(sid in sleeping_set for sid in request.stage_ids):
        return {"status": "SKIPPED", "reason": "Target stages are not sleeping."}
    if not hasattr(engine_client, "wake_up"):
        raise HTTPException(status_code=501, detail="Engine does not support wake_up")
    acks = await engine_client.wake_up(stage_ids=request.stage_ids)
    # Waking an AR stage is not enough to make it serve again: AsyncOmni.sleep()
    # sets ``_hold_admission_until_resume`` for AR/mixed stages and wake_up()
    # deliberately keeps ``_paused`` in that case, because it was built for the
    # trainer order pause -> abort -> sleep -> train -> wake -> resume.
    #
    # This endpoint pair is plain serving, not training: there is no /v1/omni/pause
    # and no /v1/omni/resume, so a caller who slept and woke an AR stage would get
    # SUCCESS here and then have every generate() block forever on ``_pause_cond``
    # with no endpoint left to recover it. Resume on the way out.
    #
    # No-op for diffusion-only engines (they never set the flag), so this is not
    # gated on stage role — a deployment that gains an AR stage must not regress.
    resume = getattr(engine_client, "resume_generation", None)
    if callable(resume):
        await resume(stage_ids=request.stage_ids)
    for sid in request.stage_ids:
        if sid in sleeping_set:
            sleeping_set.remove(sid)
    return {
        "status": "SUCCESS",
        "acks": [dataclasses.asdict(a) if dataclasses.is_dataclass(a) and not isinstance(a, type) else a for a in acks],
    }


if __name__ == "__main__":
    parser = TrackingArgumentParser(description="vLLM-Omni OpenAI-Compatible REST API server")
    parser = make_arg_parser(parser)
    # Ensure that passing --omni won't crash the server.
    # NOTE: the value here does not matter since we are always running the Omni server
    # when __main__ is called, i.e., --omni is only used when called through the entrypoints.
    parser.add_argument("--omni", action="store_true", default=False)
    args = parser.parse_args()
    # sync args.model to model_tag, because if we pass the model positionally,
    # args.model will be the default from vLLM's ModelConfig (currently
    # Qwen/Qwen3-0.6B) and crash cryptically.
    if args.model_tag is not None:
        args.model = args.model_tag
    asyncio.run(omni_run_server(args))
