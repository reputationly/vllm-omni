#!/usr/bin/env bash
# The same Ref2VA request the TP4 ladder runs, on a single card.
# Usage: run_ref2va_tp1_case.sh RUNG MODEL [PREFIX]
#
# "Is TP4 actually faster than one card" is not answerable from the TP4 logs
# alone: the ladder never runs a single-card engine, and a 4-card wall time
# includes collective overhead that only a measured TP1 run can be weighed
# against.  Same request, same seed, same contract, one card.
set -uo pipefail
RUNG=$1
MODEL=$2
PREFIX=${3:-int8pruned_tp1}

ROOT=/nfs-output/h3_turbo_eval
OUT=/nfs-output/h3_pruned_eval/ref2va_scale_20260820
REQ=$OUT/reqs/ref2va_$RUNG.json
PORT=42098
NAME="h3-turbo-eval-$PREFIX-${RUNG//_/-}"
CASE_TAG="${PREFIX}_$RUNG"
LOG=/nfs-output/h3_int8_pruned/logs/$CASE_TAG.run.log
PKG=/usr/local/lib/python3.12/dist-packages/vllm_omni
REPO=$ROOT/repo
IMAGE=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-omni:arm64-a100-latest

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "--- tp1 start $(date -Is) rung=$RUNG host=$(hostname) model=$MODEL ---"
test -f "$REQ" || { echo "missing request: $REQ"; exit 5; }
docker rm -f "$NAME" >/dev/null 2>&1

docker run -d --name "$NAME" --gpus '"device=0"' --ipc=host --network host --shm-size 64g \
  -v /nfs-models:/nfs-models -v /nfs-output:/nfs-output \
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
  -e VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=7200 -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
  -e VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0 \
  -e VLLM_OMNI_H3_OFFLOAD_DIT_BEFORE_VAE=1 -e VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK=8 \
  -e VLLM_OMNI_H3_LOG_STEP_MEMORY=1 -e VLLM_OMNI_H3_INFERENCE_CONTRACT=official_diffusers_v1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 \
  -e GLOO_SOCKET_IFNAME=lo -e DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
  "$IMAGE" vllm serve "$MODEL" --omni --num-gpus 1 -tp 1 --usp 1 --ring 1 \
  --enable-cpu-offload --text-encoder-tp-size 1 \
  --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN --disable-multithread-weight-load \
  --enable-diffusion-pipeline-profiler --init-timeout 3600 --stage-init-timeout 3600 \
  --diffusion-compile-granularity regional --diffusion-compile-dynamic \
  --allowed-local-media-path /nfs-output --host 0.0.0.0 --port "$PORT" --trust-remote-code

for attempt in $(seq 1 360); do
  if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ready_after_seconds=$((attempt * 10))"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "engine exited during startup"
    docker logs "$NAME" 2>&1 | tail -40
    exit 2
  fi
  sleep 10
done

python3 "$ROOT/run_vllm_eval_case.py" \
  --endpoint "http://127.0.0.1:$PORT" \
  --request "$REQ" \
  --output-root "$OUT" \
  --tag "$CASE_TAG" \
  --container "$NAME" \
  --timeout-seconds 7200
rc=$?
echo "case rc=$rc tag=$CASE_TAG"
docker logs "$NAME" 2>&1 | grep -iE "out of memory|CUDA error" | tail -5
docker rm -f "$NAME" >/dev/null 2>&1
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
echo "--- tp1 end $(date -Is) rc=$rc ---"
exit "$rc"
