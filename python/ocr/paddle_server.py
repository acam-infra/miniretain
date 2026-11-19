#!/usr/bin/env python3
"""Lightweight local HTTP server wrapping PaddleOCR (stdlib only, no extra web deps).

Exposes:
  GET  /health -> version, models, device, coord_space
  POST /ocr    -> raw PNG bytes (?dpi=150) or JSON {"image_base64": "...", "dpi": 150}
                  returns pixel-space text blocks with [x0, y0, x1, y1] bboxes.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _PaddleEngine:
    def __init__(self, lang: str = "en", cache_dir: str | None = None, cpu_threads: int = 2) -> None:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        if cache_dir:
            os.environ["PADDLE_PDX_CACHE_HOME"] = cache_dir

        import paddle  # type: ignore
        import paddleocr  # type: ignore

        self.paddle_version = getattr(paddle, "__version__", "unknown")
        self.paddleocr_version = getattr(paddleocr, "__version__", "unknown")
        self.device = paddle.device.get_device()
        self.lang = lang
        self.lock = threading.Lock()

        t0 = time.perf_counter()
        self._ocr = paddleocr.PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
            enable_mkldnn=False,
            cpu_threads=cpu_threads,
        )
        self.init_seconds = round(time.perf_counter() - t0, 3)

        models: list[str] = []
        model_dir = Path(cache_dir or os.environ.get("PADDLE_PDX_CACHE_HOME", "")) / "official_models"
        if model_dir.is_dir():
            models = sorted(p.name for p in model_dir.iterdir() if p.is_dir())
        self.models = models or ["PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"]

    def health(self) -> dict:
        return {
            "status": "ok",
            "service": "miniretain-paddle-ocr",
            "paddle_version": self.paddle_version,
            "paddleocr_version": self.paddleocr_version,
            "device": self.device,
            "lang": self.lang,
            "models": self.models,
            "coord_space": "pixel",
            "bbox_format": "xyxy",
            "init_seconds": self.init_seconds,
        }

    def predict_png(self, png_bytes: bytes, dpi: float = 150.0) -> dict:
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(png_bytes)
            tmp_path = tf.name
        try:
            with self.lock:
                result = self._ocr.predict(tmp_path)
            blocks: list[dict] = []
            for r in result:
                texts = r.get("rec_texts") or []
                boxes = r.get("rec_boxes") if r.get("rec_boxes") is not None else []
                scores = r.get("rec_scores") if r.get("rec_scores") is not None else [1.0] * len(texts)
                for text, box, score in zip(texts, boxes, scores):
                    t = str(text).strip()
                    if not t:
                        continue
                    x0, y0, x1, y1 = (float(v) for v in box)
                    blocks.append({
                        "text": t,
                        "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                        "score": round(float(score), 4),
                    })
            return {
                "dpi": dpi,
                "coord_space": "pixel",
                "bbox_format": "xyxy",
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "blocks": blocks,
            }
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def make_handler(engine: _PaddleEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # noqa: D401
            return

        def _send_json(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path in ("/health", "/"):
                self._send_json(200, engine.health())
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/ocr":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > 50 * 1024 * 1024:
                self._send_json(400, {"error": f"invalid Content-Length: {length}"})
                return
            body = self.rfile.read(length)
            qs = urllib.parse.parse_qs(parsed.query)
            dpi = float(qs.get("dpi", ["150"])[0])
            ctype = (self.headers.get("Content-Type") or "").lower()
            try:
                if "application/json" in ctype:
                    req = json.loads(body.decode("utf-8"))
                    dpi = float(req.get("dpi", dpi))
                    png_bytes = base64.b64decode(req["image_base64"])
                else:
                    png_bytes = body
                out = engine.predict_png(png_bytes, dpi=dpi)
                self._send_json(200, out)
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": str(exc)})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Miniretain local PaddleOCR HTTP service")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=18085)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--cache-dir", default=os.environ.get("PADDLE_PDX_CACHE_HOME", ""))
    ap.add_argument("--cpu-threads", type=int, default=2)
    args = ap.parse_args()

    engine = _PaddleEngine(lang=args.lang, cache_dir=args.cache_dir or None, cpu_threads=args.cpu_threads)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", **engine.health()}, ensure_ascii=False), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
