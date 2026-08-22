#!/usr/bin/env bash
# One rung of the Ref2VA input-scale ladder, start to finish, on this host.
# Usage: run_ref2va_scale_rung.sh RUNG MODEL [PREFIX]
#
# Each rung holds a whole 4xA100 box (TP4 + CPU offload), so rungs are spread
# one per host and run concurrently; this script is what a host is handed.
# It reuses the engine launcher and the case runner that produced the existing
# ladder numbers, so a new checkpoint's peaks stay comparable to them.
set -uo pipefail
RUNG=$1
MODEL=$2
PREFIX=${3:-int8pruned}

ROOT=/nfs-output/h3_turbo_eval
OUT=/nfs-output/h3_pruned_eval/ref2va_scale_20260820
REQ=$OUT/reqs/ref2va_$RUNG.json
PORT=42097
SERVE_TAG="$PREFIX-${RUNG//_/-}"
CASE_TAG="${PREFIX}_$RUNG"
LOG=/nfs-output/h3_int8_pruned/logs/$CASE_TAG.run.log

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "--- rung start $(date -Is) rung=$RUNG host=$(hostname) model=$MODEL ---"
test -f "$REQ" || { echo "missing request: $REQ"; exit 5; }

# GLOO_SOCKET_IFNAME=lo: NCCL rendezvous otherwise times out on these boxes.
# The contract is not a tuning knob here: the ladder requests carry an ordered
# heterogeneous `references` list, which the default `legacy` contract rejects
# with a 400 before any generation happens.  The existing ladder numbers were
# taken under official_diffusers_v1, so a comparison has to be too.
bash "$ROOT/run_vllm_turbo_eval.sh" "$SERVE_TAG" "$MODEL" "$PORT" \
  -e GLOO_SOCKET_IFNAME=lo \
  -e VLLM_OMNI_H3_INFERENCE_CONTRACT=official_diffusers_v1
serve_rc=$?
echo "serve rc=$serve_rc tag=$SERVE_TAG"
if [ "$serve_rc" -ne 0 ]; then
  tail -60 "$ROOT/$SERVE_TAG.serve.log"
  echo "--- rung end $(date -Is) rc=$serve_rc (serve) ---"
  exit "$serve_rc"
fi

python3 "$ROOT/run_vllm_eval_case.py" \
  --endpoint "http://127.0.0.1:$PORT" \
  --request "$REQ" \
  --output-root "$OUT" \
  --tag "$CASE_TAG" \
  --container "h3-turbo-eval-$SERVE_TAG" \
  --timeout-seconds 7200
case_rc=$?
echo "case rc=$case_rc tag=$CASE_TAG"

# An OOM leaves the peak in the engine log, not in the summary, and leaves
# workers holding the cards; tear down so the box is reusable either way.
docker logs "h3-turbo-eval-$SERVE_TAG" 2>&1 | grep -iE "out of memory|CUDA error" | tail -5
# The allocator's own maxima, which nvidia-smi sampling cannot see, exist only
# inside the container: harvest them before the teardown below deletes them.
echo "--- exact peak (torch allocator maxima) ---"
docker logs "h3-turbo-eval-$SERVE_TAG" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep "H3 denoise step" | tail -4
docker rm -f "h3-turbo-eval-$SERVE_TAG" >/dev/null 2>&1
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "--- rung end $(date -Is) rc=$case_rc ---"
exit "$case_rc"
