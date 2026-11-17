#!/usr/bin/env python3
"""Single Python worker entrypoint, selected by spec.stage.

The Go server never imports this module; it launches it as a process and
exchanges files only. Both sides validate the spec against the same schema.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_schema, normalize  # noqa: E402
from llm.scheduler import Scheduler, get_client  # noqa: E402
from ocr.adapter import get_provider  # noqa: E402
from render.layout import group_lines, render_pdf  # noqa: E402


def _dump(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_ocr(spec: dict, wd: Path) -> dict:
    provider = get_provider(spec["ocr"]["provider"])
    pages = provider.parse(spec["source"])
    _dump(
        wd / "document.v1.json",
        {"version": "document.v1", "pages": pages},
    )
    return {
        "stage": "ocr",
        "pages": len(pages),
        "blocks": sum(len(p["blocks"]) for p in pages),
        "document": "document.v1.json",
    }


def run_llm(spec: dict, wd: Path) -> dict:
    doc = json.loads((wd / "document.v1.json").read_text(encoding="utf-8"))
    lines = [
        {"block_id": block["block_id"], "page_index": page["page_index"],
         "text": block["text"], "bbox": block.get("bbox")}
        for page in doc["pages"]
        for block in page["blocks"]
    ]
    # Group OCR lines in the same column into paragraphs for coherent translation,
    # keeping the union bbox of each group for the render stage.
    if any(line["bbox"] for line in lines):
        items = group_lines(lines)
    else:
        items = [
            {"block_id": line["block_id"], "page_index": line["page_index"],
             "text": line["text"], "bbox": None, "line_bboxes": None}
            for line in lines
        ]

    cfg = spec["llm"]
    scheduler = Scheduler(
        client=get_client(cfg["provider"]),
        concurrency=cfg["concurrency"],
        batch_size=cfg["batch_size"],
        max_retries=cfg["max_retries"],
        timeout_ms=cfg["timeout_ms"],
    )
    translated, stats = scheduler.run(items, spec["target_lang"], cfg["model"])

    payload = [
        {
            "block_id": item["block_id"],
            "page_index": item["page_index"],
            "bbox": item.get("bbox"),
            "line_bboxes": item.get("line_bboxes"),
            "translation": value,
        }
        for item, value in zip(items, translated)
    ]
    # Same ordering as the real project: payload first, manifest last. The
    # manifest is the commit point that tells the next stage the data is whole.
    _dump(wd / "translation.payload.json", {"version": "payload.v1", "items": payload})
    _dump(
        wd / "translation-manifest.json",
        {
            "version": "translation-manifest.v1",
            "target_lang": spec["target_lang"],
            "items": len(payload),
            "payload": "translation.payload.json",
            "stats": stats,
        },
    )
    return {
        "stage": "llm",
        "items": len(items),
        "translated": sum(1 for t in translated if t),
        **stats,
    }


def run_render(spec: dict, wd: Path) -> dict:
    """Paint translated paragraphs back at their original OCR bboxes."""
    payload = json.loads((wd / "translation.payload.json").read_text(encoding="utf-8"))
    doc = json.loads((wd / "document.v1.json").read_text(encoding="utf-8"))
    cfg = spec.get("render", {})
    font = cfg.get("font") or os.environ.get("MINIRETAIN_CJK_FONT", "")
    if not font or not Path(font).is_file():
        raise SystemExit("render stage needs an existing CJK font (render.font or MINIRETAIN_CJK_FONT)")
    widths = {
        page["page_index"]: page.get("width", 1275)
        for page in doc["pages"]
    }
    output = wd / cfg.get("output", "translated.pdf")
    stats = render_pdf(spec["source"], payload["items"], str(output), font, widths)
    return {"stage": "render", **stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="path to the stage spec JSON")
    ap.add_argument("--schema", default=None, help="path to task.spec.schema.json")
    args = ap.parse_args()

    schema = load_schema(args.schema)
    spec = normalize(
        json.loads(Path(args.spec).read_text(encoding="utf-8")), schema
    )
    wd = Path(spec["workdir"])
    wd.mkdir(parents=True, exist_ok=True)

    stage = spec["stage"]
    if stage == "ocr":
        result = run_ocr(spec, wd)
    elif stage == "llm":
        result = run_llm(spec, wd)
    elif stage == "render":
        result = run_render(spec, wd)
    else:
        raise SystemExit(f"unknown stage {stage!r}")

    _dump(wd / f"{stage}.result.json", result)


if __name__ == "__main__":
    main()
