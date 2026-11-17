#!/bin/sh
# Layout-preserving end-to-end smoke: OCR -> grouped LLM -> render.
# Requirements: paddleocr/paddlepaddle, pymupdf, a CJK font and an
# OpenAI-compatible endpoint in OPENAI_BASE_URL + OPENAI_API_KEY.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
PY=${MINIRETAIN_PYTHON:-python3}
FONT=${MINIRETAIN_CJK_FONT:-}
[ -n "$FONT" ] || { echo "set MINIRETAIN_CJK_FONT to a TTF/OTF path"; exit 1; }
[ -f "$1" ] || { echo "usage: $0 <pdf>"; exit 1; }
SRC=$(cd "$(dirname -- "$1")" && pwd)/$(basename -- "$1")
WD="$ROOT/data/layout-smoke"
mkdir -p "$WD"
export MINIRETAIN_CJK_FONT="$FONT"

write_spec() {
  "$PY" - "$1" "$2" "$SRC" <<EOF
import json, sys
stage, wd, src = sys.argv[1:4]
print(json.dumps({
  "job_id": "layout-smoke", "stage": stage, "workdir": wd,
  "source": src, "target_lang": "zh",
  "ocr": {"provider": "paddle-cpu"},
  "llm": {"provider": "openai-compat", "batch_size": 20, "concurrency": 2,
          "max_retries": 1, "timeout_ms": 120000, "model": "gemini-3.6-flash"},
  "render": {"font": "$FONT", "output": "translated.pdf"},
}, ensure_ascii=False))
EOF
}

for stage in ocr llm render; do
  write_spec "$stage" "$WD" > "$WD/$stage.spec.json"
  echo "=== $stage ==="
  "$PY" "$ROOT/python/worker.py" --spec "$WD/$stage.spec.json" \
    --schema "$ROOT/contracts/task.spec.schema.json"
done
echo "PDF: $WD/translated.pdf"
