#!/usr/bin/env bash
# One official MiniMax-H3 Ref2VA case against one checkpoint, under official conditions.
# Usage: run_official_ref2va_case.sh REQUEST_JSON MODEL TAG [NFE]
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
# Unlike the ladder runner this keeps the engine alive when the request fails,
# so a rejected body can be fixed and resubmitted without a 5-minute reload.
set -uo pipefail
REQ=$1
MODEL=$2
TAG=$3
NFE=${4:-}

ROOT=/nfs-output/h3_turbo_eval
OUT=/nfs-output/h3_official_eval/results
PORT=42099
NAME="h3-official-$TAG"
LOG=/nfs-output/h3_official_eval/logs/$TAG.run.log
BODY=/nfs-output/h3_official_eval/reqs/.$TAG.body.json

mkdir -p "$(dirname "$LOG")" "$OUT"
exec >>"$LOG" 2>&1
echo "--- start $(date -Is) tag=$TAG host=$(hostname) model=$MODEL nfe=${NFE:-model-default} ---"
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

bash "$ROOT/run_vllm_turbo_eval.sh" "official-$TAG" "$MODEL" "$PORT" \
  -e GLOO_SOCKET_IFNAME=lo \
  -e VLLM_OMNI_H3_INFERENCE_CONTRACT=official_diffusers_v1
serve_rc=$?
echo "serve rc=$serve_rc"
if [ "$serve_rc" -ne 0 ]; then
  tail -60 "$ROOT/official-$TAG.serve.log"
  echo "--- end $(date -Is) rc=$serve_rc (serve) ---"
  exit "$serve_rc"
fi

python3 "$ROOT/run_vllm_eval_case.py" \
  --endpoint "http://127.0.0.1:$PORT" \
  --request "$BODY" \
  --output-root "$OUT" \
  --tag "$TAG" \
  --container "h3-turbo-eval-official-$TAG" \
  --timeout-seconds 14400
rc=$?
echo "case rc=$rc tag=$TAG"
if [ "$rc" -ne 0 ]; then
  echo "engine left running on port $PORT for retry"
  docker logs "h3-turbo-eval-official-$TAG" 2>&1 | grep -iE "out of memory|CUDA error" | tail -5
  echo "--- end $(date -Is) rc=$rc (engine kept) ---"
  exit "$rc"
fi

docker rm -f "h3-turbo-eval-official-$TAG" >/dev/null 2>&1
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
echo "--- end $(date -Is) rc=0 ---"
