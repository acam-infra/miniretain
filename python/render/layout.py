"""Layout-preserving renderer with dry-run fit measurement and source preservation.

Never erases original text before verifying the translated text fits!
Tries sizes down to `min_size` inclusive. Overflowed blocks preserve the source
text and report their block_id explicitly.
"""
from __future__ import annotations

from pathlib import Path
import statistics

try:
    import pymupdf
except ImportError:
    pymupdf = None

from render.side_by_side import build_side_by_side_pdf, merge_interleaved


def find_cjk_font(preferred: str = "") -> str:
    """Locate an existing CJK TTF/OTF/TTC font file."""
    import os

    candidates = [
        preferred,
        os.environ.get("MINIRETAIN_CJK_FONT", ""),
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return path
    raise RuntimeError("no usable CJK font found (set render.font or MINIRETAIN_CJK_FONT)")


def _columns(lines: list[dict]) -> list[list[dict]]:
    """Split one page's lines into columns by x-centre gaps."""
    if len(lines) < 2:
        return [lines]
    ordered = sorted(lines, key=lambda b: (b["bbox"][0] + b["bbox"][2]) / 2)
    centres = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in ordered]
    gaps = [(centres[i + 1] - centres[i], i) for i in range(len(centres) - 1)]
    big = [g for g in gaps if g[0] > 220]
    if not big:
        return [lines]
    cuts = sorted(i for _, i in big)
    groups: list[list[dict]] = []
    start = 0
    for cut in cuts:
        groups.append(ordered[start : cut + 1])
        start = cut + 1
    groups.append(ordered[start:])
    col_centres = [
        statistics.median((b["bbox"][0] + b["bbox"][2]) / 2 for b in g)
        for g in groups
    ]
    if len(col_centres) > 1:
        reassigned: list[list[dict]] = [[] for _ in groups]
        for b in lines:
            c = (b["bbox"][0] + b["bbox"][2]) / 2
            idx = min(range(len(col_centres)), key=lambda j: abs(c - col_centres[j]))
            reassigned[idx].append(b)
        groups = [g for g in reassigned if g]
    return groups


def group_lines(blocks: list[dict], gap_factor: float = 1.2) -> list[dict]:
    """Merge consecutive OCR lines of the same column into translation blocks.

    Blocks that already carry a structured `layout_role` / `policy` from
    `normalize_pages` (e.g. `pymupdf` or pre-normalized `paddle-cpu`) are kept
    intact without re-merging across roles.
    """
    if blocks and all("policy" in b and "layout_role" in b for b in blocks):
        return [
            {
                **b,
                "line_bboxes": b.get("line_bboxes") or ([b["bbox"]] if b.get("bbox") else None),
            }
            for b in blocks
            if b.get("text", "").strip()
        ]

    by_page: dict[int, list[dict]] = {}
    for b in blocks:
        if b.get("bbox") and b.get("text", "").strip():
            by_page.setdefault(b["page_index"], []).append(b)

    out: list[dict] = []
    for page_index, lines in by_page.items():
        heights = [b["bbox"][3] - b["bbox"][1] for b in lines]
        line_h = statistics.median(heights) if heights else 20.0
        for column in _columns(lines):
            column = sorted(column, key=lambda b: b["bbox"][1])
            current: list[dict] = []
            for line in column:
                if current:
                    prev = current[-1]
                    gap = line["bbox"][1] - prev["bbox"][3]
                    if gap > gap_factor * line_h:
                        out.append(_merge(current, page_index))
                        current = []
                current.append(line)
            if current:
                out.append(_merge(current, page_index))
    return out


def _merge(lines: list[dict], page_index: int) -> dict:
    boxes = [b["bbox"] for b in lines]
    return {
        "block_id": lines[0]["block_id"],
        "page_index": page_index,
        "text": " ".join(b["text"].strip() for b in lines),
        "bbox": [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ],
        "line_bboxes": boxes,
        "layout_role": lines[0].get("layout_role", "body"),
        "policy": lines[0].get("policy", {"translate": True, "preserve_reason": None, "redact_source": True}),
    }


def pdf_rect(bbox: list[float], scale: float):
    """Pixel/point bbox -> PDF-point rect."""
    if pymupdf is None:
        raise RuntimeError("pymupdf is required for pdf_rect")
    return pymupdf.Rect(
        bbox[0] / scale, bbox[1] / scale, bbox[2] / scale, bbox[3] / scale
    )


def fit_textbox_size(
    page_width: float,
    page_height: float,
    rect: "pymupdf.Rect",
    text: str,
    font_file: str,
    start_size: float,
    min_size: float = 5.0,
    lineheight: float = 1.08,
    scratch_page: "pymupdf.Page | None" = None,
) -> tuple[bool, float, int]:
    """Dry-run fit on a scratch page BEFORE redacting.

    Tries from `start_size` down to `min_size` INCLUSIVE.
    Returns (fits, chosen_size, shrunk_steps).
    """
    if pymupdf is None:
        raise RuntimeError("pymupdf is required")
    own_doc = None
    spage = scratch_page
    if spage is None:
        own_doc = pymupdf.open()
        spage = own_doc.new_page(width=page_width, height=page_height)
        spage.insert_font(fontname="mr_fit", fontfile=font_file)
    try:
        size = max(min_size, start_size)
        steps = 0
        while True:
            rest = spage.insert_textbox(
                rect,
                text,
                fontname="mr_fit",
                fontsize=size,
                lineheight=lineheight,
                color=(0, 0, 0),
                align=0,
            )
            if rest >= 0:
                return True, round(size, 2), steps
            if size <= min_size + 1e-6:
                return False, round(min_size, 2), steps
            size = max(min_size, size - 0.5)
            steps += 1
    finally:
        if own_doc is not None:
            own_doc.close()


def _initial_font_size(item: dict, rect: "pymupdf.Rect", min_size: float) -> float:
    hint = item.get("font_size_pt") or (item.get("content") or {}).get("font_size_pt")
    if hint and isinstance(hint, (int, float)) and hint > 0:
        return min(14.0, max(min_size, float(hint) * 0.92))
    return min(14.0, max(min_size, rect.height * 0.78))


def render_pdf(
    source: str,
    payload: list[dict],
    out: str,
    font: str,
    page_widths: dict[int, float],
    min_size: float = 5.0,
    out_merged: str | None = None,
    bilingual_mode: str = "side_by_side",
    strict: bool = False,
    page_coord_spaces: dict[int, str] | None = None,
    page_scanned: dict[int, bool] | None = None,
) -> dict:
    """Write a copy of `source` with translated text painted at original bboxes.

    Measures and verifies fit BEFORE redacting original text. Blocks with
    policy.translate=False or blocks that overflow even at min_size are NOT
    redacted (source preserved) and are reported explicitly.
    """
    if pymupdf is None:
        raise RuntimeError("pymupdf is required for render_pdf")
    font = find_cjk_font(font)
    doc = pymupdf.open(source)
    placed = 0
    preserved = 0
    unplaced = 0
    failed = 0
    shrunk = 0
    unplaced_block_ids: list[str] = []
    failed_block_ids: list[str] = []
    per_page: list[dict] = []

    scratch_doc = pymupdf.open()
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            spage = scratch_doc.new_page(width=page.rect.width, height=page.rect.height)
            spage.insert_font(fontname="mr_fit", fontfile=font)
            coord_space = (page_coord_spaces or {}).get(page_index, "pixel")
            if coord_space == "pdf_pt":
                scale = 1.0
            else:
                scale = page_widths.get(page_index, page.rect.width) / page.rect.width
            is_scanned = bool((page_scanned or {}).get(page_index, False))

            page_items = [it for it in payload if it["page_index"] == page_index]
            p_placed = 0
            p_preserved = 0
            p_unplaced = 0
            p_failed = 0
            to_render: list[tuple[dict, pymupdf.Rect, float]] = []

            for it in page_items:
                bid = it.get("block_id", "unknown")
                policy = it.get("policy") or {}
                if policy.get("translate") is False:
                    preserved += 1
                    p_preserved += 1
                    continue

                t_status = it.get("translation_status", "ok")
                trans = (it.get("translation") or "").strip()
                bbox = it.get("bbox")
                if t_status != "ok" or not trans or not bbox:
                    failed += 1
                    p_failed += 1
                    failed_block_ids.append(bid)
                    continue

                rect = pdf_rect(bbox, scale)
                start_sz = _initial_font_size(it, rect, min_size)
                fits, chosen_sz, steps = fit_textbox_size(
                    page.rect.width,
                    page.rect.height,
                    rect,
                    trans,
                    font,
                    start_size=start_sz,
                    min_size=min_size,
                    scratch_page=spage,
                )
                shrunk += steps
                if not fits:
                    # Do NOT redact source! Preserve original block and report ID.
                    unplaced += 1
                    p_unplaced += 1
                    unplaced_block_ids.append(bid)
                    continue

                to_render.append((it, rect, chosen_sz))

            # Only redact blocks that actually fit!
            # Use fill=None on digital pages so vector graphics/colored boxes are never covered by white rects.
            redact_fill = (1, 1, 1) if is_scanned else None
            for it, _rect, _sz in to_render:
                for lb in it.get("line_bboxes") or [it["bbox"]]:
                    page.add_redact_annot(pdf_rect(lb, scale), fill=redact_fill)
            if to_render:
                page.apply_redactions(images=0, graphics=0, text=0)
                page.insert_font(fontname="mr", fontfile=font)
                for it, rect, chosen_sz in to_render:
                    rest = page.insert_textbox(
                        rect,
                        it["translation"].strip(),
                        fontname="mr",
                        fontsize=chosen_sz,
                        lineheight=1.08,
                        color=(0, 0, 0),
                        align=0,
                    )
                    if rest < 0:
                        unplaced += 1
                        p_unplaced += 1
                        unplaced_block_ids.append(it.get("block_id", "unknown"))
                    else:
                        placed += 1
                        p_placed += 1

            per_page.append({
                "page_index": page_index,
                "total_blocks": len(page_items),
                "placed": p_placed,
                "preserved": p_preserved,
                "unplaced": p_unplaced,
                "failed": p_failed,
                "width_pt": round(float(page.rect.width), 2),
                "height_pt": round(float(page.rect.height), 2),
            })

        doc.save(out, deflate=True)
    finally:
        scratch_doc.close()
        doc.close()

    if strict and (unplaced_block_ids or failed_block_ids):
        raise RuntimeError(
            f"render strict mode failed: unplaced={unplaced_block_ids}, failed={failed_block_ids}"
        )

    bilingual_info = None
    if out_merged:
        if bilingual_mode == "interleaved":
            bilingual_info = merge_interleaved(source, out, out_merged)
        else:
            bilingual_info = build_side_by_side_pdf(source, out, out_merged)

    return {
        "placed": placed,
        "preserved": preserved,
        "unplaced": unplaced,
        "failed": failed,
        "unplaced_block_ids": unplaced_block_ids,
        "failed_block_ids": failed_block_ids,
        "shrunk_steps": shrunk,
        "per_page": per_page,
        "output": out,
        "output_merged": out_merged if bilingual_info else None,
        "bilingual": bilingual_info,
    }
