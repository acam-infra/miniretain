#!/bin/sh
# E2E with real providers: paddle-cpu OCR (2 pre-rendered page PNGs are NOT used;
# we feed the PDF directly) + gemini-web2api via SSH tunnel on 18083.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
export PATH=$HOME/tools/go/bin:$PATH
PY=$HOME/venvs/ocr/bin/python
export PADDLE_PDX_CACHE_HOME=$HOME/paddle_cache
export MINIRETAIN_PYTHON=$PY

PORT=${PORT:-18080}
./data/server -addr "127.0.0.1:${PORT}" -root "$ROOT" -python "$PY" > data/server.log 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true' EXIT

i=0
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  i=$((i + 1)); [ "$i" -gt 50 ] && { echo "server not healthy"; cat data/server.log; exit 1; }
  sleep 0.1
done
echo "health: OK"

BODY=$(python3 - "$1" <<EOF
import json, sys
print(json.dumps({
  "source_path": sys.argv[1],
  "target_lang": "zh",
  "ocr": {"provider": "paddle-cpu"},
  "llm": {"provider": "openai-compat", "batch_size": 40, "concurrency": 1,
          "max_retries": 1, "timeout_ms": 120000, "model": "gemini-3.6-flash"},
}))
EOF
)
RESP=$(curl -fsS -X POST "http://127.0.0.1:${PORT}/v1/jobs" -H 'Content-Type: application/json' -d "$BODY")
echo "create: $RESP"
ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')

while :; do
  JOB=$(curl -fsS "http://127.0.0.1:${PORT}/v1/jobs/${ID}")
  STATUS=$(printf '%s' "$JOB" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')
  [ "$STATUS" = "succeeded" ] && break
  [ "$STATUS" = "failed" ] && { echo "$JOB" | python3 -m json.tool; exit 1; }
  sleep 2
done
echo "job ${ID}: ${STATUS}"
curl -fsS "http://127.0.0.1:${PORT}/v1/jobs/${ID}/result" | python3 -m json.tool
echo "--- payload head ---"
head -c 800 "data/jobs/${ID}/translation.payload.json"
