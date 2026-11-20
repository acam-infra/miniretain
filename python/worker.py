#!/usr/bin/env python3
"""Single Python worker entrypoint, selected by spec.stage.

The Go server never imports this module; it launches it as a process and
exchanges files only. Both sides validate the spec against the same schema.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import load_schema, normalize  # noqa: E402
from llm.formula_protect import protect_formulas, restore_formulas  # noqa: E402
from llm.scheduler import Scheduler, get_client  # noqa: E402
from ocr.adapter import get_provider, validate_document_pages  # noqa: E402
from render.layout import find_cjk_font, group_lines, render_pdf  # noqa: E402
from render.typst_renderer import render_typst  # noqa: E402


def _dump(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_ocr(spec: dict, wd: Path) -> dict:
    t0 = time.perf_counter()
    ocr_cfg = spec["ocr"].get("config")
    provider = get_provider(spec["ocr"]["provider"], config=ocr_cfg)
    pages = validate_document_pages(provider.parse(spec["source"]))
    elapsed_s = round(time.perf_counter() - t0, 3)
    unreliable_pages = [p["page_index"] for p in pages if not p.get("ocr_reliable", True)]
    limitations = []
    if unreliable_pages:
        limitations.append(
            f"unreliable_ocr on scanned pages {unreliable_pages}: source blocks preserved"
        )
    doc_payload = {
        "version": "document.v1",
        "ocr_provider": spec["ocr"]["provider"],
        "ocr_reliable": len(unreliable_pages) == 0,
        "limitations": limitations,
        "pages": pages,
    }
    _dump(wd / "document.v1.json", doc_payload)
    role_counts: dict[str, int] = {}
    for p in pages:
        for b in p["blocks"]:
            r = b.get("layout_role", "body")
            role_counts[r] = role_counts.get(r, 0) + 1
    out = {
        "stage": "ocr",
        "provider": spec["ocr"]["provider"],
        "pages": len(pages),
        "blocks": sum(len(p["blocks"]) for p in pages),
        "roles": role_counts,
        "ocr_reliable": len(unreliable_pages) == 0,
        "limitations": limitations,
        "document": "document.v1.json",
        "elapsed_s": elapsed_s,
    }
    if getattr(provider, "service_info", None):
        out["paddle_service"] = provider.service_info
    if getattr(provider, "paddle_pages", None):
        out["paddle_pages"] = provider.paddle_pages
    if getattr(provider, "routed_providers", None):
        out["routed_providers"] = provider.routed_providers
    return out


def run_llm(spec: dict, wd: Path) -> dict:
    doc = json.loads((wd / "document.v1.json").read_text(encoding="utf-8"))
    raw_blocks = [
        {
            "block_id": block["block_id"],
            "page_index": page["page_index"],
            "reading_order": block.get("reading_order", 0),
            "text": block["text"],
            "bbox": block.get("bbox"),
            "line_bboxes": block.get("line_bboxes"),
            "layout_role": block.get("layout_role", "body"),
            "policy": block.get("policy", {"translate": True, "preserve_reason": None, "redact_source": True}),
            "content": block.get("content", {}),
            "continuation": block.get("continuation", {}),
        }
        for page in doc["pages"]
        for block in page["blocks"]
    ]
    if any(b.get("bbox") for b in raw_blocks):
        items = group_lines(raw_blocks)
    else:
        items = [
            {
                **b,
                "bbox": None,
                "line_bboxes": None,
            }
            for b in raw_blocks
        ]

    translatable_indices: list[int] = []
    translatable_inputs: list[dict] = []
    formula_tokens_by_idx: dict[int, dict[str, str]] = {}

    for idx, item in enumerate(items):
        policy = item.get("policy") or {}
        if policy.get("translate", True) and (item.get("text") or "").strip():
            protected_text, tokens = protect_formulas(
                item["text"], layout_role=item.get("layout_role", "body")
            )
            formula_tokens_by_idx[idx] = tokens
            translatable_indices.append(idx)
            translatable_inputs.append({**item, "text": protected_text})

    cfg = spec["llm"]
    scheduler = Scheduler(
        client=get_client(cfg["provider"]),
        concurrency=cfg["concurrency"],
        batch_size=cfg["batch_size"],
        max_retries=cfg["max_retries"],
        timeout_ms=cfg["timeout_ms"],
    )
    records, stats = scheduler.run(translatable_inputs, spec["target_lang"], cfg["model"])

    rec_by_item_idx = {
        idx: rec for idx, rec in zip(translatable_indices, records)
    }

    payload: list[dict] = []
    per_page: dict[int, dict] = {}
    formula_warnings: list[str] = []
    critical_failures: list[dict] = []
    non_critical_fallbacks: list[dict] = []
    non_critical_roles = frozenset({"title", "header_footer", "author_affiliation", "reference"})

    for idx, item in enumerate(items):
        p_idx = int(item["page_index"])
        p_stat = per_page.setdefault(
            p_idx,
            {"page_index": p_idx, "total": 0, "translatable": 0, "succeeded": 0, "preserved": 0, "failed": 0},
        )
        p_stat["total"] += 1
        policy = dict(item.get("policy") or {"translate": True, "preserve_reason": None, "redact_source": True})

        if idx not in rec_by_item_idx:
            p_stat["preserved"] += 1
            payload.append({
                "block_id": item["block_id"],
                "page_index": p_idx,
                "reading_order": item.get("reading_order", 0),
                "layout_role": item.get("layout_role", "body"),
                "policy": policy,
                "bbox": item.get("bbox"),
                "line_bboxes": item.get("line_bboxes"),
                "font_size_pt": (item.get("content") or {}).get("font_size_pt"),
                "is_bold": (item.get("content") or {}).get("is_bold", False),
                "text": item.get("text"),
                "translation": item.get("text"),
                "translation_status": "preserved",
                "attempts": 0,
                "error": None,
                "continuation": item.get("continuation", {}),
            })
            continue

        p_stat["translatable"] += 1
        rec = rec_by_item_idx[idx]
        t_status = rec["status"]
        trans_text = rec["translation"]
        err_msg = rec["error"]
        attempts = int(rec.get("attempts", 1))
        role = item.get("layout_role", "body")

        if t_status == "ok" and trans_text:
            restored, missing_toks = restore_formulas(trans_text, formula_tokens_by_idx.get(idx, {}))
            trans_text = restored
            if missing_toks:
                formula_warnings.append(f"{item['block_id']}: missing formula tokens {missing_toks}")
            p_stat["succeeded"] += 1
        else:
            diag = {
                "stage": "llm",
                "page_index": p_idx,
                "page_number": p_idx + 1,
                "block_id": item["block_id"],
                "layout_role": role,
                "status": t_status,
                "reason": err_msg or t_status,
                "attempts": attempts,
            }
            if role in non_critical_roles:
                # Non-critical metadata/title blocks may preserve source cleanly and let the job finish.
                policy["translate"] = False
                policy["redact_source"] = False
                policy["preserve_reason"] = f"fallback_{t_status}"
                trans_text = item.get("text")
                t_status = "preserved_fallback"
                p_stat["preserved"] += 1
                non_critical_fallbacks.append(diag)
            else:
                policy["redact_source"] = False
                p_stat["failed"] += 1
                critical_failures.append(diag)

        payload.append({
            "block_id": item["block_id"],
            "page_index": p_idx,
            "reading_order": item.get("reading_order", 0),
            "layout_role": role,
            "policy": policy,
            "bbox": item.get("bbox"),
            "line_bboxes": item.get("line_bboxes"),
            "font_size_pt": (item.get("content") or {}).get("font_size_pt"),
            "is_bold": (item.get("content") or {}).get("is_bold", False),
            "text": item.get("text"),
            "translation": trans_text,
            "translation_status": t_status,
            "attempts": attempts,
            "error": err_msg,
            "continuation": item.get("continuation", {}),
        })

    total_translatable = len(translatable_indices)
    succeeded_count = stats["succeeded_items"]
    failed_count = stats["failed_items"] + stats["missing_items"] + stats["empty_items"]
    critical_failed_count = len(critical_failures)
    allow_partial = bool(cfg.get("allow_partial", False))
    coverage = round(succeeded_count / total_translatable, 4) if total_translatable else 1.0

    _dump(wd / "translation.payload.json", {"version": "payload.v1", "items": payload})
    _dump(
        wd / "translation-manifest.json",
        {
            "version": "translation-manifest.v1",
            "target_lang": spec["target_lang"],
            "items": len(payload),
            "translatable_items": total_translatable,
            "preserved_items": len(payload) - total_translatable,
            "succeeded_items": succeeded_count,
            "failed_items": failed_count,
            "critical_failed_items": critical_failed_count,
            "non_critical_fallback_items": len(non_critical_fallbacks),
            "partial_mode": allow_partial,
            "partial_missing_body": bool(critical_failures and allow_partial),
            "failed_stage": "llm" if critical_failures else None,
            "critical_failures": critical_failures,
            "non_critical_fallbacks": non_critical_fallbacks,
            "coverage_ratio": coverage,
            "formula_warnings": formula_warnings,
            "payload": "translation.payload.json",
            "stats": stats,
            "per_page": [per_page[k] for k in sorted(per_page)],
        },
    )

    result = {
        "stage": "llm",
        "items": len(payload),
        "translatable": total_translatable,
        "preserved": len(payload) - total_translatable,
        "translated": succeeded_count,
        "failed_total": failed_count,
        "critical_failed": critical_failed_count,
        "non_critical_fallbacks": non_critical_fallbacks,
        "critical_failures": critical_failures,
        "failed_stage": "llm" if critical_failures else None,
        "partial_mode": allow_partial,
        "partial_missing_body": bool(critical_failures and allow_partial),
        "coverage_ratio": coverage,
        "formula_warnings": len(formula_warnings),
        "per_page": [per_page[k] for k in sorted(per_page)],
        **stats,
    }
    _dump(wd / "llm.result.json", result)

    if critical_failures and not allow_partial:
        summary = "; ".join(
            f"p{d['page_number']}/{d['block_id']}({d['layout_role']}): {d['reason']} [attempts={d['attempts']}]"
            for d in critical_failures[:8]
        )
        raise RuntimeError(
            f"llm stage failed: {critical_failed_count}/{total_translatable} critical body blocks failed "
            f"(allow_partial=false): {summary}"
        )

    return result


def run_render(spec: dict, wd: Path) -> dict:
    """Render translated document into pure Chinese (cn) and bilingual comparison versions."""
    payload = json.loads((wd / "translation.payload.json").read_text(encoding="utf-8"))
    doc = json.loads((wd / "document.v1.json").read_text(encoding="utf-8"))
    cfg = spec.get("render", {})
    engine = cfg.get("engine", "typst")

    output_cn = wd / cfg.get("output_cn", "translated_cn.pdf")
    output_merged = wd / cfg.get("output_merged", "translated_merged.pdf")
    legacy_output = wd / cfg.get("output", "translated.pdf")

    widths = {page["page_index"]: page.get("width", 1275) for page in doc["pages"]}
    coord_spaces = {page["page_index"]: page.get("coord_space", "pixel") for page in doc["pages"]}
    scanned = {page["page_index"]: bool(page.get("is_scanned", False)) for page in doc["pages"]}

    if engine == "typst":
        stats = render_typst(
            items=payload["items"],
            workdir=wd,
            output_cn=output_cn,
            output_merged=output_merged,
            legacy_output=legacy_output,
            font=cfg.get("font", ""),
            source=spec.get("source", ""),
            layout_mode=cfg.get("layout_mode", "auto"),
            bilingual_mode=cfg.get("bilingual_mode", "auto"),
            page_widths=widths,
            page_coord_spaces=coord_spaces,
            page_scanned=scanned,
            strict=bool(cfg.get("strict", False)),
        )
        return {"stage": "render", **stats}
    elif engine == "pymupdf":
        font = find_cjk_font(cfg.get("font", ""))
        bilingual_mode = cfg.get("bilingual_mode", "auto")
        resolved_bi = "side_by_side" if bilingual_mode in ("auto", "table") else bilingual_mode
        stats = render_pdf(
            source=spec["source"],
            payload=payload["items"],
            out=str(output_cn),
            font=font,
            page_widths=widths,
            out_merged=str(output_merged),
            bilingual_mode=resolved_bi,
            strict=bool(cfg.get("strict", False)),
            page_coord_spaces=coord_spaces,
            page_scanned=scanned,
        )
        if str(legacy_output) != str(output_cn):
            import shutil
            shutil.copyfile(output_cn, legacy_output)
        return {
            "stage": "render",
            "engine": "pymupdf",
            "output_cn": str(output_cn),
            "output_merged": str(output_merged),
            "output": str(legacy_output),
            **stats,
        }
    else:
        raise SystemExit(f"unknown render engine: {engine!r}")


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
