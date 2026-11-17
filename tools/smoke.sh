#!/bin/sh
# End-to-end smoke: create a task over HTTP and drive OCR -> LLM scheduling.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
mkdir -p data

export PATH=/data/tools/go/bin:/usr/local/go/bin:$PATH
go build -o data/server ./cmd/server

PORT=${PORT:-18080}
./data/server -addr "127.0.0.1:${PORT}" -root "$ROOT" > data/server.log 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true' EXIT

i=0
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -gt 50 ]; then
    echo "server did not become healthy"; cat data/server.log; exit 1
  fi
  sleep 0.1
done
echo "health: OK"

BODY='{"source":"RetainPDF keeps layout and inline formulas.\n\nIt uses a self-developed typesetting engine.\n\nCross-page and cross-column flow are restored before translation.\n\nInline formula tokens are protected during translation.","target_lang":"zh","llm":{"batch_size":2,"concurrency":3,"max_retries":1}}'
RESP=$(curl -fsS -X POST "http://127.0.0.1:${PORT}/v1/jobs" -H 'Content-Type: application/json' -d "$BODY")
echo "create: $RESP"
ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')

i=0
while :; do
  JOB=$(curl -fsS "http://127.0.0.1:${PORT}/v1/jobs/${ID}")
  STATUS=$(printf '%s' "$JOB" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')
  [ "$STATUS" = "succeeded" ] && break
  [ "$STATUS" = "failed" ] && { echo "$JOB"; exit 1; }
  i=$((i + 1))
  [ "$i" -gt 100 ] && { echo "timeout"; exit 1; }
  sleep 0.1
done
echo "job ${ID}: ${STATUS}"

echo "--- result ---"
curl -fsS "http://127.0.0.1:${PORT}/v1/jobs/${ID}/result" | python3 -m json.tool

echo "--- artifacts ---"
ls -1 "data/jobs/${ID}"
