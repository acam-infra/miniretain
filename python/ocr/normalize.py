"""RetainPDF-aligned document.v1 normalizer.

Enriches raw OCR / PDF text blocks with:
- geometry (bbox, line_bboxes, coord_space, dpi, column_index)
- content (text, font_size_pt, is_bold, is_math_font)
- layout_role (title, heading, caption, figure_label, formula, header_footer, reference, body)
- policy (translate, preserve_reason, redact_source)
- reading_order and cross-column/cross-page continuation
"""
from __future__ import annotations

import re
import statistics

_CAPTION_RE = re.compile(r"^\s*(?:Fig\.|Figure|Table|Algorithm)\s*\d+[\s:.\-]", re.IGNORECASE)
_HEADING_RE = re.compile(
    r"^\s*(?:(?:[IVX]+|[0-9]+(?:\.[0-9]+)*)\.\s+\S+|Abstract\b|Introduction\b|Conclusion\b|References\b|Acknowledgments?\b)",
    re.IGNORECASE,
)
_REF_HEADING_RE = re.compile(r"^\s*(?:References|REFERENCES)\s*$")
_EQ_NUM_RE = re.compile(r"\(\d+\)\s*$")
_BIG_O_ONLY_RE = re.compile(r"^\s*O\([^)]+\)\s*$")
_MATH_CHARS = set("=∑∫∏∈∉≤≥≠≈→←↔±×÷√∞∂∇αβγδεθλμσφψωΔΣΠΩ")
_MATH_FONT_TOKENS = ("CMMI", "CMSY", "CMEX", "MSAM", "MSBM", "SYMBOL", "MTMI", "MTSY")


def _rect_intersects(a: list[float], b: list[float], pad: float = 2.0) -> bool:
    return not (
        a[2] + pad < b[0]
        or a[0] - pad > b[2]
        or a[3] + pad < b[1]
        or a[1] - pad > b[3]
    )


def _rect_area(r: list[float]) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _overlap_ratio(inner: list[float], outer: list[float], pad: float = 2.0) -> float:
    x0 = max(inner[0], outer[0] - pad)
    y0 = max(inner[1], outer[1] - pad)
    x1 = min(inner[2], outer[2] + pad)
    y1 = min(inner[3], outer[3] + pad)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = _rect_area(inner)
    return inter / area if area > 0 else 0.0


def cluster_figure_regions(
    drawing_rects: list[list[float]],
    image_rects: list[list[float]],
    page_width: float,
    page_height: float,
    margin: float = 18.0,
) -> list[list[float]]:
    """Cluster nearby vector drawings and images into figure bounding boxes."""
    rects: list[list[float]] = []
    for r in drawing_rects:
        w, h = r[2] - r[0], r[3] - r[1]
        # Ignore full-page borders or tiny hairlines (< 2pt both ways)
        if w > page_width * 0.96 and h > page_height * 0.96:
            continue
        if w < 2.0 and h < 2.0:
            continue
        rects.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
    for r in image_rects:
        w, h = r[2] - r[0], r[3] - r[1]
        if w > page_width * 0.96 and h > page_height * 0.96:
            continue
        rects.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])

    if not rects:
        return []

    # Iteratively merge intersecting/nearby rects within `margin`
    merged = [list(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out: list[list[float]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = list(merged[i])
            used[i] = True
            grew = True
            while grew:
                grew = False
                for j in range(len(merged)):
                    if used[j]:
                        continue
                    if _rect_intersects(cur, merged[j], pad=margin):
                        cur = [
                            min(cur[0], merged[j][0]),
                            min(cur[1], merged[j][1]),
                            max(cur[2], merged[j][2]),
                            max(cur[3], merged[j][3]),
                        ]
                        used[j] = True
                        grew = True
                        changed = True
            out.append(cur)
        merged = out

    # Keep only non-trivial regions (at least 30x20 pt)
    return [r for r in merged if (r[2] - r[0]) >= 30.0 and (r[3] - r[1]) >= 20.0]


def _is_formula_text(text: str, is_math_font: bool = False) -> bool:
    s = text.strip()
    if not s:
        return False
    if _BIG_O_ONLY_RE.match(s):
        return True
    if is_math_font and len(s) < 120:
        return True
    math_count = sum(1 for ch in s if ch in _MATH_CHARS)
    alpha_words = re.findall(r"[A-Za-z]{3,}", s)
    if _EQ_NUM_RE.search(s) and math_count >= 1 and len(alpha_words) <= 6:
        return True
    if len(s) <= 80 and math_count >= 2 and math_count / max(1, len(s)) >= 0.08 and len(alpha_words) <= 5:
        return True
    return False


def _assign_columns(blocks: list[dict], page_width: float) -> list[dict]:
    """Assign column_index and sort blocks in reading order."""
    if not blocks:
        return []
    mid = page_width * 0.5
    # Heuristic: if substantial blocks sit clearly left of mid and right of mid, 2-column
    left_blocks = [b for b in blocks if b.get("bbox") and b["bbox"][2] <= mid + 12.0]
    right_blocks = [b for b in blocks if b.get("bbox") and b["bbox"][0] >= mid - 12.0]
    is_two_col = len(left_blocks) >= 2 and len(right_blocks) >= 2

    if not is_two_col:
        ordered = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]) if b.get("bbox") else (0.0, 0.0))
        for b in ordered:
            b["column_index"] = 0
        return ordered

    # Partition into vertical bands separated by full-width spanning blocks (e.g. title, full-width fig/caption)
    ordered_by_y = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]) if b.get("bbox") else (0.0, 0.0))
    result: list[dict] = []
    band: list[dict] = []

    def _flush_band(items: list[dict]) -> None:
        if not items:
            return
        left = [x for x in items if (x["bbox"][0] + x["bbox"][2]) / 2 <= mid]
        right = [x for x in items if (x["bbox"][0] + x["bbox"][2]) / 2 > mid]
        for x in left:
            x["column_index"] = 0
        for x in right:
            x["column_index"] = 1
        left.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        right.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        result.extend(left)
        result.extend(right)

    for b in ordered_by_y:
        bbox = b.get("bbox")
        if not bbox:
            b["column_index"] = 0
            band.append(b)
            continue
        width = bbox[2] - bbox[0]
        spans_center = bbox[0] < mid - 25.0 and bbox[2] > mid + 25.0 and width > page_width * 0.48
        if spans_center:
            _flush_band(band)
            band = []
            b["column_index"] = 0
            result.append(b)
        else:
            band.append(b)
    _flush_band(band)
    return result


def normalize_pages(
    pages: list[dict],
    *,
    coord_space: str = "pdf_pt",
    dpi: float = 72.0,
) -> list[dict]:
    """Normalize raw pages into RetainPDF-aligned document.v1 structure."""
    in_references = False
    norm_pages: list[dict] = []
    prev_translatable_body: dict | None = None

    for page in pages:
        page_idx = int(page["page_index"])
        page_w = float(page.get("width") or 612.0)
        page_h = float(page.get("height") or 792.0)
        figure_regions = page.get("figure_regions") or []
        is_scanned = bool(page.get("is_scanned", False))
        ocr_reliable = bool(page.get("ocr_reliable", True))
        raw_blocks = page.get("blocks") or []

        # Compute median body font size if available
        font_sizes = [float(b["font_size_pt"]) for b in raw_blocks if b.get("font_size_pt")]
        median_font = statistics.median(font_sizes) if font_sizes else 10.0

        ordered = _assign_columns(list(raw_blocks), page_w)
        norm_blocks: list[dict] = []

        for order_idx, b in enumerate(ordered):
            text = (b.get("text") or "").strip()
            bbox = b.get("bbox")
            line_bboxes = b.get("line_bboxes") or ([bbox] if bbox else None)
            font_size = float(b["font_size_pt"]) if b.get("font_size_pt") else None
            is_bold = bool(b.get("is_bold", False))
            is_math_font = bool(b.get("is_math_font", False))
            col_idx = int(b.get("column_index", 0))

            # Role & Policy classification
            layout_role = "body"
            translate = True
            preserve_reason: str | None = None
            redact_source = True

            if not text:
                layout_role = "empty"
                translate = False
                preserve_reason = "empty"
                redact_source = False
            elif not ocr_reliable:
                layout_role = "unreliable"
                translate = False
                preserve_reason = "unreliable_ocr"
                redact_source = False
            elif _CAPTION_RE.match(text):
                layout_role = "caption"
                translate = True
            elif bbox and any(_overlap_ratio(bbox, fr, pad=4.0) >= 0.5 for fr in figure_regions):
                # Inside a figure/diagram region and NOT a caption
                if len(text) <= 120 or (bbox[2] - bbox[0]) < page_w * 0.35:
                    layout_role = "figure_label"
                    translate = False
                    preserve_reason = "figure_label"
                    redact_source = False
            elif _is_formula_text(text, is_math_font=is_math_font):
                layout_role = "formula"
                translate = False
                preserve_reason = "formula"
                redact_source = False
            elif bbox and (bbox[1] < page_h * 0.045 or bbox[3] > page_h * 0.96) and len(text) < 80:
                layout_role = "header_footer"
                translate = False
                preserve_reason = "header_footer"
                redact_source = False
            elif _REF_HEADING_RE.match(text):
                layout_role = "heading"
                in_references = True
                translate = True
            elif page_idx == 0 and font_size and font_size >= median_font * 1.35 and len(text) < 200:
                layout_role = "title"
                translate = True
            elif _HEADING_RE.match(text) and len(text) < 120:
                layout_role = "heading"
                translate = True
            elif in_references:
                layout_role = "reference"
                translate = True

            # Respect explicit pre-set policy if provided
            if isinstance(b.get("policy"), dict) and "translate" in b["policy"]:
                translate = bool(b["policy"]["translate"])
                preserve_reason = b["policy"].get("preserve_reason", preserve_reason)
                redact_source = bool(b["policy"].get("redact_source", translate))

            nb = {
                "block_id": b["block_id"],
                "page_index": page_idx,
                "reading_order": order_idx,
                "kind": layout_role,
                "layout_role": layout_role,
                "text": text,
                "bbox": bbox,
                "line_bboxes": line_bboxes,
                "geometry": {
                    "bbox": bbox,
                    "line_bboxes": line_bboxes,
                    "coord_space": coord_space,
                    "dpi": dpi,
                    "column_index": col_idx,
                },
                "content": {
                    "text": text,
                    "font_size_pt": font_size,
                    "is_bold": is_bold,
                    "is_math_font": is_math_font,
                },
                "policy": {
                    "translate": translate,
                    "preserve_reason": preserve_reason,
                    "redact_source": redact_source,
                },
                "continuation": {
                    "continued_from": None,
                    "continues_to": None,
                },
            }

            # Cross-column / cross-page continuation detection for body blocks
            if layout_role == "body" and translate:
                if prev_translatable_body is not None:
                    prev_txt = prev_translatable_body["text"].rstrip()
                    if (
                        prev_txt
                        and prev_txt[-1] not in ".!?。！？:"
                        and text
                        and (text[0].islower() or text[0] in ",;)")
                    ):
                        prev_translatable_body["continuation"]["continues_to"] = nb["block_id"]
                        nb["continuation"]["continued_from"] = prev_translatable_body["block_id"]
                prev_translatable_body = nb
            elif layout_role in ("heading", "title", "reference"):
                prev_translatable_body = None

            norm_blocks.append(nb)

        norm_pages.append({
            "page_index": page_idx,
            "width": page_w,
            "height": page_h,
            "coord_space": coord_space,
            "dpi": dpi,
            "is_scanned": is_scanned,
            "ocr_reliable": ocr_reliable,
            "figure_regions": figure_regions,
            "blocks": norm_blocks,
        })

    return norm_pages
