"""Layout-preserving renderer.

OCR line blocks carry pixel-space bboxes (at a known DPI). group_lines() merges
neighbouring lines in the same column into paragraphs so translation is coherent;
render_pdf() then paints the translated text back into the ORIGINAL bounding
boxes of a copy of the source PDF. This is the layer the stub pipeline lacked:
translation quality aside, the output keeps the original page geometry, columns
and figure placement because only text rectangles are touched.
"""
from __future__ import annotations

import statistics

try:
    import pymupdf
except ImportError:
    pymupdf = None


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
    # reassign by nearest column centre (a centred title may straddle a cut)
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

    Input blocks: {block_id, page_index, text, bbox:[x0,y0,x1,y1]}.
    Output adds line_bboxes (each original line rect) and a union bbox.
    """
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
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes),
        ],
        "line_bboxes": boxes,
    }


def pdf_rect(bbox: list[float], scale: float):
    """Pixel bbox (at OCR DPI) -> PDF-point rect."""
    if pymupdf is None:
        raise RuntimeError("pymupdf is required for pdf_rect")
    return pymupdf.Rect(
        bbox[0] / scale, bbox[1] / scale, bbox[2] / scale, bbox[3] / scale
    )


def merge_interleaved(source_pdf: str, translated_pdf: str, out_merged: str) -> bool:
    """Interleave original and translated pages: p0_orig, p0_trans, p1_orig, p1_trans..."""
    try:
        doc_orig = pymupdf.open(source_pdf)
        doc_trans = pymupdf.open(translated_pdf)
        doc_out = pymupdf.open()
        max_p = max(len(doc_orig), len(doc_trans))
        for i in range(max_p):
            if i < len(doc_orig):
                doc_out.insert_pdf(doc_orig, from_page=i, to_page=i)
            if i < len(doc_trans):
                doc_out.insert_pdf(doc_trans, from_page=i, to_page=i)
        doc_out.save(out_merged, deflate=True)
        doc_out.close()
        doc_orig.close()
        doc_trans.close()
        return True
    except Exception:
        return False


def render_pdf(
    source: str,
    payload: list[dict],
    out: str,
    font: str,
    page_widths: dict[int, float],
    min_size: float = 5.0,
    out_merged: str | None = None,
) -> dict:
    """Write a copy of `source` with translated text painted at original bboxes.

    `page_widths` maps page_index -> OCR pixel width, used to derive the pixel
    to PDF-point scale. Returns simple render stats.
    """
    doc = pymupdf.open(source)
    placed = 0
    shrunk = 0
    for page_index in {it["page_index"] for it in payload}:
        if page_index >= len(doc):
            continue
        page = doc[page_index]
        scale = page_widths.get(page_index, 150 / 72) / page.rect.width
        items = [it for it in payload if it["page_index"] == page_index]
        items = [it for it in items if it.get("translation") and it.get("bbox")]
        # Redact only the original TEXT rectangles (not images/graphics), so the
        # English text is really removed from the content stream, not just covered.
        for it in items:
            for lb in it.get("line_bboxes", [it["bbox"]]):
                page.add_redact_annot(pdf_rect(lb, scale), fill=(1, 1, 1))
        if items:
            page.apply_redactions(images=0, graphics=0, text=0)
        page.insert_font(fontname="mr", fontfile=font)
        for it in items:
            text = it["translation"]
            rect = pdf_rect(it["bbox"], scale)
            size = min(14.0, max(min_size, rect.height * 0.78))
            while size > min_size:
                rest = page.insert_textbox(
                    rect, text, fontname="mr", fontsize=size,
                    lineheight=1.08, color=(0, 0, 0), align=0,
                )
                if rest >= 0:
                    break
                size -= 0.5
                shrunk += 1
            placed += 1
    doc.save(out, deflate=True)
    doc.close()

    merged_created = False
    if out_merged:
        merged_created = merge_interleaved(source, out, out_merged)

    return {
        "placed": placed,
        "shrunk_steps": shrunk,
        "output": out,
        "output_merged": out_merged if merged_created else None,
    }
