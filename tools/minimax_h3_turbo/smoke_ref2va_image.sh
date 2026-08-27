#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Smoke-test one Ref2VA tier against the image as shipped.
# Usage: smoke_ref2va_image.sh TAG MODEL PORT [NFE] [EXTRA_SERVE_ARG ...]
# Env:   REQ_FILE=<path>  request body to send (default: the 3-image official case1)
#
# Deliberately mounts no source: every other runner here bind-mounts patched
# .py files over the installed package, which is fine for experiments but means
# a green run says nothing about the image. GPUStack-managed instances restart
# from the image and lose any in-container patch, so what has to be proven
# before deployment is precisely the unmounted path. Only the data volumes are
# mounted.
set -uo pipefail
TAG=$1
MODEL=$2
PORT=$3
NFE=${4:-}
shift 4 2>/dev/null || shift $#

OUT=/nfs-output/h3_official_eval/results
REQ=${REQ_FILE:-/nfs-output/h3_official_eval/reqs/case1_3img.json}
NAME="h3-smoke-$TAG"
LOG=/nfs-output/h3_official_eval/logs/smoke-$TAG.log
BODY=/nfs-output/h3_official_eval/reqs/.smoke-$TAG.json
IMAGE=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/vllm-omni:arm64-a100-latest

mkdir -p "$(dirname "$LOG")" "$OUT"
exec >>"$LOG" 2>&1
echo "--- smoke start $(date -Is) tag=$TAG host=$(hostname) model=$MODEL nfe=${NFE:-model-default} ---"
echo "image digest: $(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null)"

test -f "$REQ" || { echo "missing request: $REQ"; exit 5; }
# Echo the input scale: the request path is a parameter now, and a run that
# silently used the default one is indistinguishable afterwards from a run that
# used the intended one — the numbers look perfectly reasonable either way.
python3 - "$REQ" "$BODY" "$NFE" <<'PY'
import json, sys
req, out, nfe = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.load(open(req))
if nfe:
    body["num_inference_steps"] = int(nfe)
json.dump(body, open(out, "w"), ensure_ascii=False, indent=2)
kinds = {}
for r in body["references"]:
    kinds[r["type"]] = kinds.get(r["type"], 0) + 1
print("request:", req)
print("input_scale:", kinds, "seconds:", body.get("seconds", "(model default)"),
      "size:", f'{body.get("width")}x{body.get("height")}', "steps:", body.get("num_inference_steps", "(model default)"))
PY

if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "port $PORT already busy; refusing to start"
  exit 4
fi
docker rm -f "$NAME" >/dev/null 2>&1

docker run -d --name "$NAME" --gpus all --ipc=host --network host --shm-size 64g \
  -v /nfs-models:/nfs-models -v /nfs-output:/nfs-output \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT_S=7200 -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
  -e VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=0 \
  -e VLLM_OMNI_H3_OFFLOAD_DIT_BEFORE_VAE=1 -e VLLM_OMNI_H3_VAE_REVERT_FRAME_CHUNK=8 \
  -e VLLM_OMNI_H3_LOG_STEP_MEMORY=1 -e VLLM_OMNI_H3_INFERENCE_CONTRACT=official_diffusers_v1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 \
  -e GLOO_SOCKET_IFNAME=lo -e DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
  "$IMAGE" vllm serve "$MODEL" --omni --num-gpus 4 -tp 4 --usp 1 --ring 1 \
  --enable-cpu-offload --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN --disable-multithread-weight-load \
  --init-timeout 3600 --stage-init-timeout 3600 \
  --allowed-local-media-path /nfs-output --host 0.0.0.0 --port "$PORT" --trust-remote-code "$@"

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
[ "$ready" -eq 1 ] || { echo "not ready after 3600s"; docker logs "$NAME" 2>&1 | tail -40; exit 3; }

served=$(curl -sS --max-time 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
[ "$served" = "$MODEL" ] || { echo "port serves $served, not $MODEL"; exit 4; }
echo "serving_model_verified=$served"

python3 /nfs-output/h3_turbo_eval/run_vllm_eval_case.py \
  --endpoint "http://127.0.0.1:$PORT" --request "$BODY" --output-root "$OUT" \
  --tag "smoke-$TAG" --container "$NAME" --timeout-seconds 7200
rc=$?
echo "case rc=$rc"
docker logs "$NAME" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep "H3 denoise step" | tail -2
docker rm -f "$NAME" >/dev/null 2>&1
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
echo "--- smoke end $(date -Is) rc=$rc ---"
exit "$rc"
