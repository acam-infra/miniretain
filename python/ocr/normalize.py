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
_NUMERIC_TICK_RE = re.compile(
    r"^(?:"
    r"[+\-−]?\d+(?:[.,]\d+)?(?:\s*(?:[kKmMgGtT%]|e[+\-−]?\d+|×\s*10[+\-−⁰¹²³⁴⁵⁶⁷⁸⁹\d]*))?"
    r"|10[+\-−⁰¹²³⁴⁵⁶⁷⁸⁹\d\-−]+"
    r"|(?:[+\-−]?\d+(?:[.,]\d+)?[kKmM%]?(?:\s+|$)){2,8}"
    r")$"
)
_MATH_CHARS = set("=∑∫∏∈∉≤≥≠≈→←↔±×÷√∞∂∇αβγδεζηθκλμνξπρστυφχψωΔΣΠΩ·−∼⊂⊆∪∩∀∃∝‖⟨⟩")
_MATH_FONT_TOKENS = ("CMMI", "CMSY", "CMEX", "MSAM", "MSBM", "SYMBOL", "MTMI", "MTSY")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_AFFIL_KW_RE = re.compile(
    r"\b(?:University|Institute|Research|Google|Microsoft|Berkeley|Stanford|MIT|CMU|Department|Laboratory|Labs)\b"
)
_CODE_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")
_CODE_CALL_RE = re.compile(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\s*\(")
_CODE_ASSIGN_RE = re.compile(r"^\s*(?:\d+\s+)?[A-Za-z_]\w*\s*=\s*[A-Za-z_]\w*\s*\(")
_CODE_IMPORT_RE = re.compile(r"(?:^|\s)(?:from\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s+import\s+|import\s+[A-Za-z_]\w*\s+as\s+)")
_ALGO_FUNC_RE = re.compile(r"^(?:Function|Procedure|Algorithm)\s+[A-Za-z_]\w*\s*\(")
_ALGO_IO_RE = re.compile(r"^(?:[A-Za-z0-9∗†‡]{1,3}\s+)?(?:Input|Output|Require|Ensure)\s*:")
_APPENDIX_HEADING_RE = re.compile(r"^\s*(?:Appendix\b|Supplementary\b|[A-Z]\.\s+[A-Z]\S*)", re.IGNORECASE)


def _is_code_listing(text: str) -> bool:
    s = text.strip()
    if not s or len(s) > 800:
        return False
    # Strip inline/display LaTeX math spans so LaTeX subscripts like \theta_b are never mistaken for snake_case code.
    s_nomath = re.sub(r"\$\$[^$]+\$\$|\$[^$\n]+\$|\\\([^\n]+?\\\)|\\\[[\s\S]+?\\\]", " ", s)
    if "←" in s_nomath and re.search(r"\b\d+:", s_nomath):
        return True
    if re.match(r"^(?:Function|Procedure|Algorithm)\s+[A-Za-z_]\w*\s*\(", s_nomath) and len(s) <= 220:
        return True
    if _CODE_IMPORT_RE.search(s_nomath) and len(s) < 220:
        return True
    if len(s) <= 160 and (_CODE_SNAKE_RE.search(s_nomath) or _CODE_CALL_RE.search(s_nomath) or _CODE_ASSIGN_RE.search(s_nomath)):
        return True
    if re.search(r"\b[a-z_]\w*\s*:\s*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", s_nomath) or '"""' in s_nomath or "'''" in s_nomath:
        return True
    return False


def _is_author_affiliation(
    text: str,
    page_idx: int,
    seen_abstract: bool,
    *,
    bbox: list[float] | None = None,
    page_h: float = 792.0,
    abstract_y0: float | None = None,
    font_size: float | None = None,
    median_font: float = 10.0,
) -> bool:
    if page_idx != 0:
        return False
    # Never misclassify large title text as an author/affiliation line
    if font_size and font_size >= median_font * 1.25:
        return False
    # Restrict to the geometric author region on page 0 (above Abstract, or top 35% of page)
    if bbox is not None:
        if abstract_y0 is not None:
            if bbox[3] > abstract_y0 + 6.0:
                return False
        elif bbox[1] > page_h * 0.35:
            return False
    elif seen_abstract:
        return False
    s = text.strip()
    if not s or len(s) > 220:
        return False
    if _EMAIL_RE.search(s):
        return True
    if any(ch in s for ch in "∗†‡§"):
        return True
    if _AFFIL_KW_RE.search(s) and not s.endswith("."):
        return True
    # Multiple "First Last<digit>" author entries on one line (with or without space before marker)
    if len(re.findall(r"\b[A-Z][a-z]+\s+(?:[A-Z]\.\s*)?[A-Z][a-z]+[\d∗†‡§,\s]*\d", s)) >= 2:
        return True
    return False


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


def _is_numeric_axis_tick(text: str) -> bool:
    """Detect pure numeric / log-scale exponent axis ticks (e.g. '35', '1000', '103', '10−1', '10³').

    Preserving these verbatim prevents PyMuPDF/Typst from flattening superscript
    exponents (like 10^3 -> 103) and keeps axis tick geometry untouched.
    """
    s = text.strip()
    if not s or len(s) > 32:
        return False
    return bool(_NUMERIC_TICK_RE.match(s))


def _is_formula_text(text: str, is_math_font: bool = False) -> bool:
    s = text.strip()
    if not s:
        return False
    if re.match(r"^\(\d+(?:\.\d+)*\)$", s):
        return True
    if _BIG_O_ONLY_RE.match(s):
        return True
    if is_math_font and len(s) < 120:
        return True
    math_count = sum(1 for ch in s if ch in _MATH_CHARS)
    alpha_words = re.findall(r"[A-Za-z]{3,}", s)
    if len(s) <= 8 and not alpha_words and re.search(r"[A-Za-z]", s) and not re.match(r"^(?:yes|no|on|off|ok|id|or|in|to|of|by|at|as|is|if|an|we|it|do|so|up)$", s, re.I):
        return True
    if len(s) <= 30 and math_count >= 1 and len(alpha_words) <= 2:
        return True
    if _EQ_NUM_RE.search(s) and math_count >= 1 and len(alpha_words) <= 6:
        return True
    if len(s) <= 80 and math_count >= 2 and math_count / max(1, len(s)) >= 0.08 and len(alpha_words) <= 5:
        return True
    if len(s) <= 100 and not s.endswith((".", "!", "?")) and re.search(r"[A-Za-z][A-Za-z0-9_]*\([^()\n]{1,40}\)\s*[=≈≤≥≠]", s):
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
        spans_center = bbox[0] < mid - 25.0 and bbox[2] > mid + 25.0 and width > page_width * 0.42
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
    seen_abstract = False
    norm_pages: list[dict] = []
    prev_translatable_body: dict | None = None

    for page in pages:
        page_idx = int(page["page_index"])
        page_w = float(page.get("width") or 612.0)
        page_h = float(page.get("height") or 792.0)
        figure_regions = page.get("figure_regions") or []
        table_regions = page.get("table_regions") or []
        is_scanned = bool(page.get("is_scanned", False))
        ocr_reliable = bool(page.get("ocr_reliable", True))
        raw_blocks = page.get("blocks") or []

        # Compute median body font size if available
        font_sizes = [float(b["font_size_pt"]) for b in raw_blocks if b.get("font_size_pt")]
        median_font = statistics.median(font_sizes) if font_sizes else 10.0

        ordered = _assign_columns(list(raw_blocks), page_w)
        abstract_y0: float | None = None
        if page_idx == 0:
            abs_ys = [
                float(rb["bbox"][1])
                for rb in ordered
                if rb.get("bbox")
                and re.match(r"^\s*(?:Abstract|1\.?\s+Introduction)\b", (rb.get("text") or "").strip(), re.IGNORECASE)
                and len((rb.get("text") or "").strip()) < 120
            ]
            if abs_ys:
                abstract_y0 = min(abs_ys)
        norm_blocks: list[dict] = []

        for order_idx, b in enumerate(ordered):
            raw_text = (b.get("text") or "").strip()
            text = re.sub(r"\*{1,2}", "", raw_text).strip()
            bbox = b.get("bbox")
            line_bboxes = b.get("line_bboxes") or ([bbox] if bbox else None)
            font_size = float(b["font_size_pt"]) if b.get("font_size_pt") else None
            is_bold = bool(b.get("is_bold", False))
            is_italic = bool(b.get("is_italic", False))
            is_math_font = bool(b.get("is_math_font", False))
            is_vertical = bool(b.get("is_vertical", False))
            if not is_vertical and bbox and len(text) < 80:
                bw = max(1.0, bbox[2] - bbox[0])
                bh = max(1.0, bbox[3] - bbox[1])
                if bh > bw * 2.5:
                    is_vertical = True
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
            elif is_vertical and len(text) < 120:
                layout_role = "figure_label"
                translate = False
                preserve_reason = "vertical_figure_label"
                redact_source = False
            elif _CAPTION_RE.match(text):
                layout_role = "caption"
                translate = True
            elif bbox and any(_overlap_ratio(bbox, tr, pad=3.0) >= 0.5 for tr in table_regions):
                # Table header / cell inside a detected bordered or booktabs table region
                layout_role = "table"
                translate = False
                preserve_reason = "table"
                redact_source = False
            elif bbox and any(_overlap_ratio(bbox, fr, pad=4.0) >= 0.5 for fr in figure_regions):
                # Inside a figure/diagram region and NOT a caption
                if len(text) <= 120 or (bbox[2] - bbox[0]) < page_w * 0.35:
                    layout_role = "figure_label"
                    translate = False
                    preserve_reason = "figure_label"
                    redact_source = False
            elif (
                bbox
                and (len(text) <= 45 or (bbox[2] - bbox[0]) < 80.0)
                and (not font_size or font_size <= max(8.5, median_font * 0.92))
                and any(_overlap_ratio(bbox, fr, pad=14.0) >= 0.5 for fr in figure_regions)
            ):
                # Axis / tick / legend label sitting just outside a figure frame
                layout_role = "figure_label"
                translate = False
                preserve_reason = "figure_label"
                redact_source = False
            elif _is_formula_text(text, is_math_font=is_math_font):
                layout_role = "formula"
                translate = False
                preserve_reason = "formula"
                redact_source = False
            elif _is_numeric_axis_tick(text):
                layout_role = "figure_label"
                translate = False
                preserve_reason = "numeric_axis_tick"
                redact_source = False
            elif (
                bbox
                and len(text) <= 45
                and (
                    (font_size and font_size <= 5.5)
                    or (bbox[3] - bbox[1]) <= 6.0
                    or (font_size and font_size <= 6.5 and (bbox[2] - bbox[0]) <= 60.0)
                )
            ):
                # Microscopic chart/plot axis/legend labels below Typst's minimum readable font floor
                layout_role = "figure_label"
                translate = False
                preserve_reason = "tiny_figure_label"
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
            elif in_references and not re.match(r"^\[\d+\]", text) and len(text) < 120 and not text.rstrip().endswith((".", ",")) and (
                _APPENDIX_HEADING_RE.match(text)
                or (is_bold and font_size and font_size >= median_font * 1.15 and re.match(r"^(?:[A-Z](?:\.\s+|\s+[A-Z])|Appendix|Supplementary)", text))
            ):
                layout_role = "heading"
                in_references = False
                translate = True
            elif _is_code_listing(text):
                layout_role = "figure_label"
                translate = False
                preserve_reason = "code_listing"
                redact_source = False
            elif page_idx == 0 and font_size and font_size >= median_font * 1.35 and len(text) < 200:
                layout_role = "title"
                translate = True
            elif _is_author_affiliation(
                text,
                page_idx,
                seen_abstract,
                bbox=bbox,
                page_h=page_h,
                abstract_y0=abstract_y0,
                font_size=font_size,
                median_font=median_font,
            ):
                layout_role = "header_footer"
                translate = False
                preserve_reason = "author_affiliation"
                redact_source = False
            elif _HEADING_RE.match(text) and len(text) < 120:
                layout_role = "heading"
                if re.match(r"^\s*(?:Abstract|1\.?\s+Introduction)\b", text, re.IGNORECASE):
                    seen_abstract = True
                translate = True
            elif in_references:
                layout_role = "reference"
                # Pure [N] + author-list lines (no title/venue/URL) are preserved as fixed identifiers
                if (
                    re.match(r"^\[\d+\]\s+[A-Z]\.", text)
                    and not any(q in text for q in ('“', '”', '"', "http", "doi:"))
                    and text.rstrip().endswith(",")
                ):
                    translate = False
                    preserve_reason = "reference_authors"
                    redact_source = False
                else:
                    translate = True

            # Respect explicit pre-set policy if provided
            if isinstance(b.get("policy"), dict) and "translate" in b["policy"]:
                translate = bool(b["policy"]["translate"])
                preserve_reason = b["policy"].get("preserve_reason", preserve_reason)
                redact_source = bool(b["policy"].get("redact_source", translate))

            final_text = raw_text if translate else text

            nb = {
                "block_id": b["block_id"],
                "page_index": page_idx,
                "reading_order": order_idx,
                "kind": layout_role,
                "layout_role": layout_role,
                "text": final_text,
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
                    "text": final_text,
                    "font_size_pt": font_size,
                    "is_bold": is_bold,
                    "is_italic": is_italic,
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

        # Horizontal-band formula propagation: protect equation fragments sharing a y-band with formula blocks
        formula_boxes = [
            (nb["bbox"], nb["text"])
            for nb in norm_blocks
            if nb["layout_role"] == "formula" and nb.get("bbox")
        ]
        if formula_boxes:
            for nb in norm_blocks:
                if not nb["policy"]["translate"] or nb["layout_role"] != "body" or not nb.get("bbox"):
                    continue
                bx0, by0, bx1, by1 = nb["bbox"]
                txt = nb["text"].strip()
                if not txt or len(txt) > 80 or txt.endswith("."):
                    continue
                b_mid_y = (by0 + by1) * 0.5
                for fbox, ftxt in formula_boxes:
                    fx0, fy0, fx1, fy1 = fbox
                    y_ov = min(by1, fy1) - max(by0, fy0)
                    f_mid_y = (fy0 + fy1) * 0.5
                    if y_ov <= 0 and abs(b_mid_y - f_mid_y) > 10.0:
                        continue
                    h_gap = max(0.0, max(bx0 - fx1, fx0 - bx1))
                    if h_gap > 45.0:
                        continue
                    long_words = re.findall(r"[a-z]{4,}", txt)
                    if "=" in txt or _EQ_NUM_RE.search(ftxt) or len(txt) <= 35 or len(long_words) <= 3:
                        nb["kind"] = "formula"
                        nb["layout_role"] = "formula"
                        nb["policy"] = {
                            "translate": False,
                            "preserve_reason": "formula_band",
                            "redact_source": False,
                        }
                        break

        # Algorithm pseudocode Input/Output header preservation:
        # Preserve structured Input/Output/Require/Ensure parameter lines immediately above
        # preserved Function/Procedure/Algorithm headers in the same column so split math
        # subscripts (e.g. "0 Output:") and overlapping header boxes stay intact as vector PDF.
        algo_anchors = [
            nb["bbox"]
            for nb in norm_blocks
            if not nb["policy"]["translate"]
            and nb.get("bbox")
            and _ALGO_FUNC_RE.match((nb.get("text") or "").strip())
        ]
        for ax0, ay0, ax1, _ay1 in algo_anchors:
            above = [
                nb
                for nb in norm_blocks
                if nb["policy"]["translate"]
                and nb["layout_role"] == "body"
                and nb.get("bbox")
                and nb["bbox"][1] >= ay0 - 95.0
                and nb["bbox"][3] <= ay0 + 2.0
                and nb["bbox"][0] >= ax0 - 20.0
                and nb["bbox"][0] <= ax1 + 40.0
                and nb["bbox"][2] <= max(ax1 + 100.0, ax0 + page_w * 0.48)
            ]
            io_blocks = [
                nb for nb in above if _ALGO_IO_RE.match((nb.get("text") or "").strip())
            ]
            if io_blocks:
                top_io_y = min(nb["bbox"][1] for nb in io_blocks)
                for nb in above:
                    if nb["bbox"][1] >= top_io_y - 2.0:
                        nb["layout_role"] = "figure_label"
                        nb["policy"] = {
                            "translate": False,
                            "preserve_reason": "algorithm_io_header",
                            "redact_source": False,
                        }

        # Deduplicate/merge geometrically contained translatable child blocks (e.g. split footnotes)
        def _box_area(box: list[float]) -> float:
            return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

        for child in norm_blocks:
            if not child["policy"]["translate"] or not child.get("bbox"):
                continue
            cb = child["bbox"]
            c_area = _box_area(cb)
            if c_area <= 0:
                continue
            best_parent = None
            best_p_area = float("inf")
            for parent in norm_blocks:
                if parent is child or not parent["policy"]["translate"] or not parent.get("bbox"):
                    continue
                pb = parent["bbox"]
                p_area = _box_area(pb)
                if p_area <= c_area * 1.05:
                    continue
                if _overlap_ratio(cb, pb, pad=1.0) >= 0.85 and p_area < best_p_area:
                    best_parent = parent
                    best_p_area = p_area
            if best_parent is not None:
                c_txt = " ".join(child["text"].split())
                p_txt = " ".join(best_parent["text"].split())
                if c_txt and c_txt not in p_txt:
                    if child["reading_order"] >= best_parent["reading_order"]:
                        merged_txt = best_parent["text"].rstrip() + " " + child["text"].strip()
                    else:
                        merged_txt = child["text"].strip() + " " + best_parent["text"].lstrip()
                    best_parent["text"] = merged_txt
                    best_parent["content"]["text"] = merged_txt
                child_lbs = child.get("line_bboxes") or [cb]
                parent_lbs = list(best_parent.get("line_bboxes") or [best_parent["bbox"]])
                for lb in child_lbs:
                    if lb not in parent_lbs:
                        parent_lbs.append(lb)
                best_parent["line_bboxes"] = parent_lbs
                pb = best_parent["bbox"]
                best_parent["bbox"] = [
                    min(pb[0], cb[0]),
                    min(pb[1], cb[1]),
                    max(pb[2], cb[2]),
                    max(pb[3], cb[3]),
                ]
                child["policy"] = {
                    "translate": False,
                    "preserve_reason": "contained_in_parent",
                    "redact_source": False,
                }

        norm_pages.append({
            "page_index": page_idx,
            "width": page_w,
            "height": page_h,
            "coord_space": coord_space,
            "dpi": dpi,
            "is_scanned": is_scanned,
            "ocr_reliable": ocr_reliable,
            "figure_regions": figure_regions,
            "table_regions": table_regions,
            "blocks": norm_blocks,
        })

    return norm_pages
