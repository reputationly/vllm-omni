#!/usr/bin/env bash
# One official MiniMax-H3 Ref2VA case against one checkpoint, under official conditions.
# Usage: run_official_ref2va_case.sh REQUEST_JSON MODEL TAG [NFE]
# Env:   CPU_OFFLOAD=0  serve with the weights resident in VRAM instead of streamed
#                       from host memory between phases (default 1, as the ladder ran)
#
# The request body is the official test-set entry verbatim — official prompt,
# official reference images, official duration, 1344x768 like the published
# demos.  Only the checkpoint changes between runs, so a visible difference is
# attributable to the checkpoint rather than to the material or the wording.
#
# NFE is passed only for the distilled arms.  The full-fat baseline is served
# without a step override so it uses the model's own default: pinning it to 4
# would silently turn the baseline into a Turbo run and erase the very
# comparison this exists to make.
#
# The engine is launched here rather than through the shared
# ``run_vllm_turbo_eval.sh`` for two reasons: that script hard-codes
# ``--enable-cpu-offload``, which is one of the variables under test; and it is
# shared with another workstream, so depending on it makes these results
# hostage to edits made for a different experiment.  The serve flags below are
# copied from it verbatim so the arms stay comparable with the ladder.
#
# Unlike the ladder runner this keeps the engine alive when the request fails,
# so a rejected body can be fixed and resubmitted without a 5-minute reload.
set -uo pipefail
REQ=$1
MODEL=$2
TAG=$3
NFE=${4:-}
CPU_OFFLOAD=${CPU_OFFLOAD:-1}

ROOT=/nfs-output/h3_turbo_eval
OUT=/nfs-output/h3_official_eval/results
PORT=42099
NAME="h3-turbo-eval-official-$TAG"
LOG=/nfs-output/h3_official_eval/logs/$TAG.run.log
BODY=/nfs-output/h3_official_eval/reqs/.$TAG.body.json
PKG=/usr/local/lib/python3.12/dist-packages/vllm_omni
REPO=$ROOT/repo
IMAGE=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-omni:arm64-a100-latest

mkdir -p "$(dirname "$LOG")" "$OUT"
exec >>"$LOG" 2>&1
echo "--- start $(date -Is) tag=$TAG host=$(hostname) model=$MODEL nfe=${NFE:-model-default} cpu_offload=$CPU_OFFLOAD ---"
test -f "$REQ" || { echo "missing request: $REQ"; exit 5; }

python3 - "$REQ" "$BODY" "$NFE" <<'PY'
import json, sys
req, out, nfe = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.load(open(req))
if nfe:
    body["num_inference_steps"] = int(nfe)
json.dump(body, open(out, "w"), ensure_ascii=False, indent=2)
print("request:", {k: (v if k != "prompt" else f"<{len(v)} chars>") for k, v in body.items() if k != "references"},
      "refs:", len(body["references"]))
PY

offload_flag="--enable-cpu-offload"
[ "$CPU_OFFLOAD" = "0" ] && offload_flag=""

docker rm -f "$NAME" >/dev/null 2>&1
# A previous run that failed is deliberately left serving for retry (see the tail
# of this script), and every run uses the same host port. Without this check the
# new container loses the bind, the readiness probe answers from the *old*
# engine, and the case silently measures the previous checkpoint — which is a
# result that looks completely normal and is entirely wrong.
if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "port $PORT is already serving an engine; refusing to start a second one"
  curl -sS --max-time 3 "http://127.0.0.1:$PORT/v1/models"
  docker ps --format '{{.Names}}' | grep h3- || true
  exit 4
fi

docker run -d --name "$NAME" --gpus all --ipc=host --network host --shm-size 64g \
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
  "$IMAGE" vllm serve "$MODEL" --omni --num-gpus 4 -tp 4 --usp 1 --ring 1 \
  $offload_flag --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN --disable-multithread-weight-load \
  --enable-diffusion-pipeline-profiler --init-timeout 3600 --stage-init-timeout 3600 \
  --diffusion-compile-granularity regional --diffusion-compile-dynamic \
  --allowed-local-media-path /nfs-output --host 0.0.0.0 --port "$PORT" --trust-remote-code

ready=0
for attempt in $(seq 1 360); do
  if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "ready_after_seconds=$((attempt * 10))"
    ready=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "engine exited during startup"
    docker logs "$NAME" 2>&1 | tail -40
    exit 2
  fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then
  # Falling through here would post to an engine that never came up and report
  # it as a request failure, hiding a startup problem behind a runner error.
  echo "$NAME was not ready after 3600 seconds"
  docker logs "$NAME" 2>&1 | tail -40
  exit 3
fi

# Belt and braces: the probe above proves *something* answers on this port, not
# that it is serving this checkpoint. ``/v1/models`` reports the model path, so
# compare it — measuring the wrong checkpoint is the one failure mode that
# produces a perfectly plausible result.
served=$(curl -sS --max-time 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
if [ "$served" != "$MODEL" ]; then
  echo "port $PORT is serving $served, not the requested $MODEL"
  exit 4
fi
echo "serving_model_verified=$served"

python3 "$ROOT/run_vllm_eval_case.py" \
  --endpoint "http://127.0.0.1:$PORT" \
  --request "$BODY" \
  --output-root "$OUT" \
  --tag "$TAG" \
  --container "$NAME" \
  --timeout-seconds 14400
rc=$?
echo "case rc=$rc tag=$TAG"
if [ "$rc" -ne 0 ]; then
  echo "engine left running on port $PORT for retry"
  docker logs "$NAME" 2>&1 | grep -iE "out of memory|CUDA error" | tail -5
  echo "--- end $(date -Is) rc=$rc (engine kept) ---"
  exit "$rc"
fi

# The allocator's own maxima, which nvidia-smi sampling cannot see, exist only
# inside the container: harvest them before the teardown below deletes them.
echo "--- exact peak (torch allocator maxima) ---"
docker logs "$NAME" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep "H3 denoise step" | tail -4

docker rm -f "$NAME" >/dev/null 2>&1
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
echo "--- end $(date -Is) rc=0 ---"
