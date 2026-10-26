#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Run the whole gate suite against either the MaaS gateway or a local engine.
#
#   ./run_suite.sh maas  <batch-dir>                 # uses $KEY
#   ./run_suite.sh local <endpoint> <batch-dir> [container]
#
# Already-finished cases are skipped, so an interrupted batch can be resumed by
# running the same command again. ref2va cases are skipped unless --with-ref2va
# is passed, because they need a ref2va endpoint (a fl2va instance cannot serve them).
set -euo pipefail

MODE="${1:?usage: run_suite.sh maas <batch-dir> | run_suite.sh local <endpoint> <batch-dir> [container]}"
shift

WITH_REF2VA=0
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--with-ref2va" ]]; then WITH_REF2VA=1; else ARGS+=("$arg"); fi
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUITE="$ROOT/docs/h3-gate-cases"

case "$MODE" in
  maas)
    BATCH="${ARGS[0]:?maas mode needs a batch directory}"
    : "${KEY:?maas mode needs KEY exported with your sk-key}"
    MAAS_BASE_URL="${MAAS_BASE_URL:-https://maas.ovaijisuan.com}"
    OUT="$BATCH" MODE_NAME=maas
    ;;
  local)
    ENDPOINT="${ARGS[0]:?local mode needs an endpoint}"
    BATCH="${ARGS[1]:?local mode needs a batch directory}"
    CONTAINER="${ARGS[2]:-}"
    OUT="$BATCH" MODE_NAME=local
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
failed=0
for file in "$SUITE"/*.json; do
  case_id="$(basename "$file" .json)"
  [[ "$case_id" == "manifest" ]] && continue
  if [[ "$WITH_REF2VA" == "0" && "$case_id" =~ ^(s15|g5|g6|g11) ]]; then
    echo "SKIP $case_id (ref2va; pass --with-ref2va to include)"
    continue
  fi
  tag="${MODE_NAME}_${case_id}"
  if [[ -f "$OUT/$tag/summary.json" ]] && grep -q '"status": "completed"' "$OUT/$tag/summary.json"; then
    echo "SKIP $tag (already completed)"
    continue
  fi

  if [[ "$MODE_NAME" == "maas" ]]; then
    python3 "$HERE/run_maas_case.py" --base-url "$MAAS_BASE_URL" --api-key "$KEY" \
      --request "$file" --tag "$tag" --output-root "$OUT" >/dev/null 2>&1 || true
  else
    local_args=(--endpoint "$ENDPOINT" --request "$file" --tag "$tag" --output-root "$OUT")
    [[ -n "$CONTAINER" ]] && local_args+=(--container "$CONTAINER")
    python3 "$HERE/run_vllm_eval_case.py" "${local_args[@]}" >/dev/null 2>&1 || true
  fi

  status="$(python3 - "$OUT/$tag/summary.json" <<'PY' 2>/dev/null || echo ERR
import json, sys
d = json.load(open(sys.argv[1]))
print(d["status"]["status"], round(d.get("wall_seconds") or 0))
PY
)"
  echo "$tag -> $status"
  [[ "$status" == failed* || "$status" == ERR ]] && failed=$((failed + 1))

  # Local runs write inside the container; pull the clip out next to the summary.
  if [[ "$MODE_NAME" == "local" && -n "$CONTAINER" ]]; then
    docker cp "$CONTAINER:/tmp/storage/$OUT/$tag/output.mp4" "$OUT/$tag/output.mp4" >/dev/null 2>&1 || true
  fi
done

echo "done: $OUT (failures: $failed)"
[[ "$failed" -eq 0 ]]
