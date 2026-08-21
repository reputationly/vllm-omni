#!/usr/bin/env bash
# Serve one Ref2VA checkpoint and POST a ladder request verbatim, keeping the
# engine up and recording the *body* of whatever the API answers.
# Usage: probe_ref2va_request.sh TAG MODEL REQUEST_JSON
#
# run_vllm_eval_case.py reports only "HTTPError 400", which cannot distinguish a
# rejected checkpoint from a rejected request; this prints the server's reason.
set -uo pipefail
TAG=$1
MODEL=$2
REQ=$3
PORT=42097
ROOT=/nfs-output/h3_turbo_eval
LOG=/nfs-output/h3_int8_pruned/logs/probe_$TAG.log

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "--- probe start $(date -Is) tag=$TAG host=$(hostname) model=$MODEL ---"
bash "$ROOT/run_vllm_turbo_eval.sh" "probe-$TAG" "$MODEL" "$PORT" -e GLOO_SOCKET_IFNAME=lo
serve_rc=$?
echo "serve rc=$serve_rc"
[ "$serve_rc" -eq 0 ] || { tail -40 "$ROOT/probe-$TAG.serve.log"; exit "$serve_rc"; }

echo "=== /v1/models ==="
curl -sS "http://127.0.0.1:$PORT/v1/models"
echo
echo "=== POST /v1/tasks/video/ ==="
curl -sS -o /tmp/probe_$TAG.body -w 'http_code=%{http_code}\n' \
  -H 'Content-Type: application/json' \
  --data-binary @"$REQ" \
  "http://127.0.0.1:$PORT/v1/tasks/video/"
cat /tmp/probe_$TAG.body
echo
echo "--- probe end $(date -Is) (engine left running on port $PORT) ---"
