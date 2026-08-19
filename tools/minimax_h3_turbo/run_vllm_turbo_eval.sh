#!/usr/bin/env bash
# Start one isolated MiniMax-H3 Turbo TP4 evaluation engine on a GPU worker.
# Usage: run_vllm_turbo_eval.sh TAG MODEL PORT [docker -e NAME=VALUE ...]
set -euo pipefail

TAG=$1
MODEL=$2
PORT=$3
shift 3

ROOT=/nfs-output/h3_turbo_eval
REPO=$ROOT/repo
PKG=/usr/local/lib/python3.12/dist-packages/vllm_omni
IMAGE=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-omni:arm64-a100-latest
NAME=h3-turbo-eval-$TAG
LOG=$ROOT/$TAG.serve.log

mkdir -p "$ROOT"
for source in \
  "$REPO/vllm_omni/diffusion/models/minimax_h3" \
  "$REPO/vllm_omni/diffusion/sched/sigma_schedule.py" \
  "$REPO/vllm_omni/diffusion/model_metadata.py" \
  "$REPO/vllm_omni/diffusion/data.py" \
  "$REPO/vllm_omni/config/stage_config.py" \
  "$REPO/vllm_omni/config/omni_config.py" \
  "$REPO/vllm_omni/entrypoints/openai/serving_video.py" \
  "$REPO/vllm_omni/entrypoints/openai/api_server.py" \
  "$REPO/vllm_omni/entrypoints/openai/protocol/video_tasks.py" \
  "$REPO/vllm_omni/entrypoints/openai/protocol/videos.py"; do
  test -e "$source" || { echo "missing evaluation source: $source" >&2; exit 5; }
done

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME is already running" >&2
  exit 1
fi
if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "port $PORT is already serving an engine" >&2
  exit 4
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true

{
  echo "started_at=$(date -Is)"
  echo "host=$(hostname)"
  echo "tag=$TAG"
  echo "model=$MODEL"
  echo "port=$PORT"
  echo "extra_serve_args=${EXTRA_SERVE_ARGS:-}"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
} >"$LOG"

docker run -d --name "$NAME" --gpus all --ipc=host --network host --shm-size 64g \
  -v /nfs-models:/nfs-models -v /nfs-output:/nfs-output -v /nfs-data:/nfs-data \
  -v "$REPO/vllm_omni/diffusion/models/minimax_h3:$PKG/diffusion/models/minimax_h3:ro" \
  -v "$REPO/vllm_omni/diffusion/sched/sigma_schedule.py:$PKG/diffusion/sched/sigma_schedule.py:ro" \
  -v "$REPO/vllm_omni/diffusion/model_metadata.py:$PKG/diffusion/model_metadata.py:ro" \
  -v "$REPO/vllm_omni/diffusion/data.py:$PKG/diffusion/data.py:ro" \
  -v "$REPO/vllm_omni/config/stage_config.py:$PKG/config/stage_config.py:ro" \
  -v "$REPO/vllm_omni/config/omni_config.py:$PKG/config/omni_config.py:ro" \
  -v "$REPO/vllm_omni/entrypoints/openai/serving_video.py:$PKG/entrypoints/openai/serving_video.py:ro" \
  -v "$REPO/vllm_omni/entrypoints/openai/api_server.py:$PKG/entrypoints/openai/api_server.py:ro" \
  -v "$REPO/vllm_omni/entrypoints/openai/protocol/video_tasks.py:$PKG/entrypoints/openai/protocol/video_tasks.py:ro" \
  -v "$REPO/vllm_omni/entrypoints/openai/protocol/videos.py:$PKG/entrypoints/openai/protocol/videos.py:ro" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=7200 \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
  -e VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0 \
  -e VLLM_OMNI_H3_OFFLOAD_DIT_BEFORE_VAE=1 \
  -e VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK=8 \
  -e VLLM_OMNI_H3_LOG_STEP_MEMORY=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 \
  -e DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
  "$@" \
  "$IMAGE" vllm serve "$MODEL" --omni --num-gpus 4 -tp 4 --usp 1 --ring 1 \
  --enable-cpu-offload --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN --disable-multithread-weight-load \
  --enable-diffusion-pipeline-profiler --init-timeout 3600 --stage-init-timeout 3600 \
  --diffusion-compile-granularity regional --diffusion-compile-dynamic \
  --allowed-local-media-path /nfs-output --host 0.0.0.0 --port "$PORT" --trust-remote-code \
  ${EXTRA_SERVE_ARGS:-} >>"$LOG" 2>&1

for attempt in $(seq 1 320); do
  if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >"$ROOT/$TAG.models.json" 2>/dev/null; then
    {
      echo "ready_after_seconds=$((attempt * 10))"
      docker inspect -f 'container={{.Id}} image={{.Config.Image}}' "$NAME"
      cat "$ROOT/$TAG.models.json"
    } >>"$LOG"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "$NAME exited during startup" >>"$LOG"
    docker logs "$NAME" >>"$LOG" 2>&1 || true
    exit 2
  fi
  sleep 10
done

echo "$NAME was not ready after 3200 seconds" >>"$LOG"
exit 3
