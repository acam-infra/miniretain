"""PyMuPDF native PDF layout extractor and Auto OCR provider."""
from __future__ import annotations

import re
import time
import pymupdf

from ocr.base import OCRConfigError, OCRProvider
from ocr.normalize import cluster_figure_regions, normalize_pages

_MATH_FONT_TOKENS = ("CMMI", "CMSY", "CMEX", "MSAM", "MSBM", "SYMBOL", "MTMI", "MTSY")
_REF_START_RE = re.compile(r"^\[\d+\]\s+")
_STANDALONE_EQ_NUM_RE = re.compile(r"^\(\d+(?:\.\d+)*\)$")
_TABLE_CAPTION_RE = re.compile(r"^(?:Table|Tab\.|Fig\.|Figure|Algorithm|Listing)\s*\d+", re.IGNORECASE)


def _join_lines_smart(lines_text: list[str]) -> str:
    """Join lines within a block, repairing hyphenation and wrapped URLs."""
    if not lines_text:
        return ""
    out = lines_text[0]
    for nxt in lines_text[1:]:
        prev_tail = out.rstrip()
        nxt_head = nxt.lstrip()
        # Repair hyphenated word breaks like "disag-" + "gregation"
        if prev_tail.endswith("-") and nxt_head and nxt_head[0].islower():
            out = prev_tail[:-1] + nxt_head
            continue
        # Repair wrapped URLs: "https:" + "//..." or "https://arxiv.org/ab" + "s/2507..."
        last_tok = prev_tail.rsplit(None, 1)[-1] if prev_tail else ""
        first_tok = nxt_head.split(None, 1)[0] if nxt_head else ""
        if (
            last_tok.startswith(("http://", "https://", "http:", "https:", "doi:"))
            and first_tok
            and (
                last_tok.endswith((":", "/", "-", "_", ".", "?", "=", "&"))
                or nxt_head.startswith(("//", "/"))
                or ("/" in last_tok and not any(ch in first_tok for ch in "，。；"))
            )
        ):
            # Only join without space if nxt_head looks like URL path continuation
            if nxt_head.startswith("//") or (
                last_tok.startswith(("http://", "https://"))
                and re.match(r"^[A-Za-z0-9_./?=&%+\-#]+$", first_tok)
                and not re.match(r"^\[\d+\]$", first_tok)
            ):
                out = prev_tail + nxt_head
                continue
        out = prev_tail + " " + nxt_head
    return out.strip()


def _split_reference_lines(lines: list[dict]) -> list[list[dict]]:
    """Split a block's lines for leading right-aligned equation numbers or new [N] references."""
    if len(lines) < 2:
        return [lines]
    # Split a leading right-aligned standalone equation number (e.g. "(9)") that PyMuPDF
    # grouped with the following body paragraph so the equation number stays preserved in place.
    first_txt = "".join(s.get("text", "") for s in (lines[0].get("spans") or [])).strip()
    if (
        _STANDALONE_EQ_NUM_RE.match(first_txt)
        and lines[0].get("bbox")
        and lines[1].get("bbox")
        and lines[0]["bbox"][0] > lines[1]["bbox"][0] + 60.0
    ):
        return [[lines[0]]] + _split_reference_lines(lines[1:])
    min_x0 = min(l["bbox"][0] for l in lines if l.get("bbox"))
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for line in lines:
        lt = "".join(s.get("text", "") for s in (line.get("spans") or [])).strip()
        lx0 = line["bbox"][0] if line.get("bbox") else min_x0
        if cur and _REF_START_RE.match(lt) and lx0 <= min_x0 + 6.0:
            groups.append(cur)
            cur = [line]
        else:
            cur.append(line)
    if cur:
        groups.append(cur)
    return groups


def _build_block_from_lines(
    block_id: str, lines: list[dict], page_rect_width: float
) -> tuple[dict | None, int]:
    line_bboxes: list[list[float]] = []
    line_texts: list[str] = []
    span_sizes: list[float] = []
    bold_chars = 0
    math_chars = 0
    char_count = 0
    is_vertical = False

    for line in lines:
        spans = line.get("spans") or []
        lt = "".join(s.get("text", "") for s in spans).strip()
        if not lt:
            continue
        line_texts.append(lt)
        if line.get("bbox"):
            line_bboxes.append([float(v) for v in line["bbox"]])
        ldir = line.get("dir")
        if ldir and abs(float(ldir[1])) > 0.5:
            is_vertical = True
        for s in spans:
            st = s.get("text", "")
            n = len(st.strip())
            if n == 0:
                continue
            char_count += n
            span_sizes.append(float(s.get("size", 10.0)))
            fname = (s.get("font") or "").upper()
            flags = int(s.get("flags", 0))
            if (flags & 16) or "BOLD" in fname:
                bold_chars += n
            if any(tok in fname for tok in _MATH_FONT_TOKENS):
                math_chars += n

    text = _join_lines_smart(line_texts)
    if not text or not line_bboxes:
        return None, 0
    bbox = [
        min(lb[0] for lb in line_bboxes),
        min(lb[1] for lb in line_bboxes),
        max(lb[2] for lb in line_bboxes),
        max(lb[3] for lb in line_bboxes),
    ]
    avg_size = sum(span_sizes) / len(span_sizes) if span_sizes else 10.0
    return {
        "block_id": block_id,
        "text": text,
        "bbox": bbox,
        "line_bboxes": line_bboxes,
        "font_size_pt": round(avg_size, 2),
        "is_bold": bold_chars > char_count * 0.6 if char_count else False,
        "is_math_font": math_chars > char_count * 0.5 if char_count else False,
        "is_vertical": is_vertical,
    }, len(text)


def _extract_table_regions(
    page: pymupdf.Page,
    drawing_rects: list[list[float]],
    blocks: list[dict] | None = None,
) -> list[list[float]]:
    """Detect bordered and minimal-border (booktabs) table regions on a digital PDF page.

    Combines PyMuPDF's native `find_tables()` (multi-row, multi-column) with a
    conservative horizontal-rule booktabs detector (>=3 aligned horizontal rules
    with a compact header band). Splits rule groups across intervening prose/captions
    or inter-table gaps and rejects fallback boxes that span already-detected tables,
    preventing two same-width tables from swallowing body text between them.
    """
    pw = float(page.rect.width)
    ph = float(page.rect.height)
    regions: list[list[float]] = []

    try:
        tf = page.find_tables()
        for t in getattr(tf, "tables", None) or []:
            if getattr(t, "row_count", 0) < 2 or getattr(t, "col_count", 0) < 2:
                continue
            tb = [float(v) for v in t.bbox]
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            if tw >= 60.0 and 20.0 <= th < ph * 0.85:
                regions.append(tb)
    except Exception:
        pass

    def _has_intervening_prose(gx0: float, gx1: float, y_lo: float, y_hi: float) -> bool:
        if not blocks or y_hi - y_lo < 12.0:
            return False
        for b in blocks:
            bb = b.get("bbox")
            if not bb:
                continue
            # Block sits vertically between the two rules and horizontally in the same band
            if bb[1] >= y_lo - 2.0 and bb[3] <= y_hi + 2.0 and bb[2] > gx0 + 10.0 and bb[0] < gx1 - 10.0:
                txt = (b.get("text") or "").strip()
                if not txt:
                    continue
                if _TABLE_CAPTION_RE.match(txt):
                    return True
                if len(txt) >= 40 or (len(txt) >= 20 and txt.endswith((".", ":", "?", "!"))):
                    return True
        return False

    def _distinct_y_count(rules: list[list[float]]) -> int:
        ys: list[float] = []
        for r in rules:
            if not ys or abs(r[1] - ys[-1]) > 4.0:
                ys.append(r[1])
        return len(ys)

    # Booktabs / minimal-border table detector: group aligned horizontal rules per table.
    hlines = [
        r for r in drawing_rects
        if (r[2] - r[0]) >= 60.0 and (r[3] - r[1]) <= 3.0 and (r[2] - r[0]) <= pw * 0.96
    ]
    # Bucket rules by horizontal alignment (x0, x1) first, then split vertically into independent tables.
    x_buckets: list[list[list[float]]] = []
    for r in sorted(hlines, key=lambda x: (x[1], x[0])):
        placed = False
        for bkt in x_buckets:
            if abs(r[0] - bkt[0][0]) <= 10.0 and abs(r[2] - bkt[0][2]) <= 10.0:
                bkt.append(r)
                placed = True
                break
        if not placed:
            x_buckets.append([r])

    groups: list[list[list[float]]] = []
    for bkt in x_buckets:
        cur: list[list[float]] = [bkt[0]]
        for r in bkt[1:]:
            prev_y = cur[-1][1]
            gap = r[1] - prev_y
            gx0 = min(x[0] for x in cur)
            gx1 = max(x[2] for x in cur)
            # Split into a new table if gap is large, prose/caption sits between rules,
            # or current group already forms a complete >=3-rule booktabs table before a >28pt gap.
            if (
                gap > 180.0
                or _has_intervening_prose(gx0, gx1, prev_y, r[1])
                or (_distinct_y_count(cur) >= 3 and gap > 28.0)
            ):
                groups.append(cur)
                cur = [r]
            else:
                cur.append(r)
        groups.append(cur)

    for g in groups:
        # Collapse near-duplicate double rules (within 4pt vertically)
        ys: list[float] = []
        for r in g:
            if not ys or abs(r[1] - ys[-1]) > 4.0:
                ys.append(r[1])
        if len(ys) < 3:
            continue
        # Booktabs tables have a compact header band between toprule and midrule (<= 45pt)
        if (ys[1] - ys[0]) > 45.0:
            continue
        box = [
            min(r[0] for r in g),
            min(r[1] for r in g),
            max(r[2] for r in g),
            max(r[3] for r in g),
        ]
        th = box[3] - box[1]
        if not (24.0 <= th <= ph * 0.75):
            continue
        # Reject any fallback box that overlaps/merges already-detected independent find_tables() regions
        overlaps_existing = False
        for er in regions:
            ix = max(0.0, min(box[2], er[2]) - max(box[0], er[0]))
            iy = max(0.0, min(box[3], er[3]) - max(box[1], er[1]))
            er_area = max(1.0, (er[2] - er[0]) * (er[3] - er[1]))
            if (ix * iy) / er_area >= 0.5:
                overlaps_existing = True
                break
        if not overlaps_existing:
            regions.append(box)

    return regions


def _extract_page_blocks(
    page: pymupdf.Page, *, scanned_char_threshold: int = 20
) -> tuple[list[dict], list[list[float]], list[list[float]], bool]:
    """Extract paragraph/line blocks, figure regions, and table regions from a digital PDF page."""
    d = page.get_text("dict")
    raw_blocks = d.get("blocks", [])
    drawing_rects = [list(dr["rect"]) for dr in page.get_drawings() if dr.get("rect")]
    image_rects = [list(b["bbox"]) for b in raw_blocks if b.get("type") == 1 and b.get("bbox")]
    figure_regions = cluster_figure_regions(
        drawing_rects, image_rects, page.rect.width, page.rect.height
    )

    blocks: list[dict] = []
    total_chars = 0
    for idx, b in enumerate(raw_blocks):
        if b.get("type") != 0:
            continue
        lines = b.get("lines") or []
        sub_groups = _split_reference_lines(lines)
        for sub_i, sub_lines in enumerate(sub_groups):
            bid = f"p{page.number}b{idx:04d}" if len(sub_groups) == 1 else f"p{page.number}b{idx:04d}_{sub_i}"
            blk, nchars = _build_block_from_lines(bid, sub_lines, page.rect.width)
            if blk is not None:
                blocks.append(blk)
                total_chars += nchars

    is_scanned = total_chars < scanned_char_threshold and len(image_rects) > 0
    blocks = _merge_inline_math_split_blocks(blocks)
    blocks = _merge_split_references(blocks)
    table_regions = _extract_table_regions(page, drawing_rects, blocks)
    return blocks, figure_regions, table_regions, is_scanned


def _merge_inline_math_split_blocks(blocks: list[dict]) -> list[dict]:
    """Merge paragraph blocks split mid-line by PyMuPDF at lowered math subscripts
    (e.g. 'ˆσ2' ending block N and 'b). Here...' starting block N+1 on the same line)."""
    if len(blocks) < 2:
        return blocks
    merged: list[dict] = []
    for blk in blocks:
        if (
            merged
            and not merged[-1].get("is_vertical")
            and not blk.get("is_vertical")
            and merged[-1].get("line_bboxes")
            and blk.get("line_bboxes")
        ):
            prev = merged[-1]
            cl = prev["line_bboxes"][-1]
            nl = blk["line_bboxes"][0]
            min_h = max(1.0, min(cl[3] - cl[1], nl[3] - nl[1]))
            y_ov = min(cl[3], nl[3]) - max(cl[1], nl[1])
            same_line = y_ov >= 0.45 * min_h
            # Subscript continues right where the previous line ended (often overlapping x by ~3-5pt)
            x_cont = (cl[2] - 10.0 <= nl[0] <= cl[2] + 8.0) and (nl[0] > prev["bbox"][0] + 40.0)
            same_col = True
            if len(blk["line_bboxes"]) >= 2:
                nxt_left = min(lb[0] for lb in blk["line_bboxes"][1:])
                same_col = abs(nxt_left - prev["bbox"][0]) <= 22.0
            if same_line and x_cont and same_col:
                prev_txt = prev["text"].rstrip()
                nxt_txt = blk["text"].lstrip()
                # If previous text ends in math superscript/Greek and next starts with short subscript
                if re.search(r"[ˆθµμσηλαβγδε]\d*$", prev_txt) and re.match(r"^[a-zA-Z0-9]{1,2}(?:[^a-zA-Z0-9]|$)", nxt_txt):
                    comb_text = prev_txt + nxt_txt
                else:
                    comb_text = _join_lines_smart([prev_txt, nxt_txt])
                joined_line = [
                    min(cl[0], nl[0]),
                    min(cl[1], nl[1]),
                    max(cl[2], nl[2]),
                    max(cl[3], nl[3]),
                ]
                comb_lbs = list(prev["line_bboxes"][:-1]) + [joined_line] + list(blk["line_bboxes"][1:])
                comb_bbox = [
                    min(prev["bbox"][0], blk["bbox"][0]),
                    min(prev["bbox"][1], blk["bbox"][1]),
                    max(prev["bbox"][2], blk["bbox"][2]),
                    max(prev["bbox"][3], blk["bbox"][3]),
                ]
                merged[-1] = {
                    **prev,
                    "text": comb_text,
                    "bbox": comb_bbox,
                    "line_bboxes": comb_lbs,
                }
                continue
        merged.append(blk)
    return merged


def _merge_split_references(blocks: list[dict]) -> list[dict]:
    """Merge a [N]-starting reference header block with its immediate continuation block."""
    if len(blocks) < 2:
        return blocks
    merged: list[dict] = []
    i = 0
    while i < len(blocks):
        cur = blocks[i]
        if (
            i + 1 < len(blocks)
            and _REF_START_RE.match(cur["text"])
            and not _REF_START_RE.match(blocks[i + 1]["text"])
            and not cur.get("is_vertical")
            and not blocks[i + 1].get("is_vertical")
            and abs(cur["bbox"][2] - blocks[i + 1]["bbox"][2]) < 30.0
            and 0 <= blocks[i + 1]["bbox"][1] - cur["bbox"][3] <= 12.0
        ):
            nxt = blocks[i + 1]
            # Keep merging if there's another continuation? Usually 1 continuation block
            comb_lines = _join_lines_smart([cur["text"], nxt["text"]])
            comb_lbs = list(cur["line_bboxes"]) + list(nxt["line_bboxes"])
            comb_bbox = [
                min(cur["bbox"][0], nxt["bbox"][0]),
                min(cur["bbox"][1], nxt["bbox"][1]),
                max(cur["bbox"][2], nxt["bbox"][2]),
                max(cur["bbox"][3], nxt["bbox"][3]),
            ]
            merged.append({
                **cur,
                "text": comb_lines,
                "bbox": comb_bbox,
                "line_bboxes": comb_lbs,
            })
            i += 2
            continue
        merged.append(cur)
        i += 1
    return merged


class PyMuPDFOCRProvider(OCRProvider):
    """Native PDF layout extractor using PyMuPDF (fast, preserves fonts & drawings)."""

    name = "pymupdf"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        auto_cfg = self.config.get("auto") if isinstance(self.config.get("auto"), dict) else {}
        thresh = self.config.get("scanned_char_threshold", auto_cfg.get("scanned_char_threshold", 20))
        self.scanned_char_threshold = int(thresh)
        if self.scanned_char_threshold < 0:
            raise OCRConfigError(f"scanned_char_threshold must be >= 0, got {self.scanned_char_threshold}")
        self.parse_seconds: float = 0.0

    def parse(self, source: str) -> list[dict]:
        t0 = time.time()
        doc = pymupdf.open(source)
        raw_pages: list[dict] = []
        try:
            for page in doc:
                blocks, figure_regions, table_regions, is_scanned = _extract_page_blocks(
                    page, scanned_char_threshold=self.scanned_char_threshold
                )
                raw_pages.append({
                    "page_index": page.number,
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "figure_regions": figure_regions,
                    "table_regions": table_regions,
                    "is_scanned": is_scanned,
                    "ocr_reliable": not is_scanned,
                    "blocks": blocks,
                })
        finally:
            doc.close()
        self.parse_seconds = round(time.time() - t0, 2)
        return normalize_pages(raw_pages, coord_space="pdf_pt", dpi=72.0)


class AutoOCRProvider(OCRProvider):
    """Uses native PyMuPDF on digital pages and routes scanned pages to local Paddle OCR."""

    name = "auto"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        auto_cfg = self.config.get("auto") if isinstance(self.config.get("auto"), dict) else {}
        self.scanned_char_threshold = int(auto_cfg.get("scanned_char_threshold", 20))
        self.fallback_provider = str(auto_cfg.get("fallback_provider") or "").strip()
        if self.fallback_provider and self.fallback_provider not in {"paddle-http", "paddle-cpu", "none"}:
            raise OCRConfigError(
                f"auto.fallback_provider must be one of ['paddle-http', 'paddle-cpu', 'none'], got {self.fallback_provider!r}"
            )
        self.service_info: dict = {}
        self.paddle_pages: list[int] = []
        self.routed_providers: dict[str, list[int]] = {}
        self.parse_seconds: float = 0.0

    def parse(self, source: str) -> list[dict]:
        import os

        t0 = time.perf_counter()
        self.service_info = {}
        self.paddle_pages = []
        self.routed_providers = {}
        if not source.lower().endswith(".pdf"):
            from ocr.adapter import StubOCRProvider
            pages = StubOCRProvider().parse(source)
            self.routed_providers = {"stub": [p["page_index"] for p in pages]}
            self.parse_seconds = round(time.perf_counter() - t0, 3)
            return pages

        pymupdf_prov = PyMuPDFOCRProvider(config={"scanned_char_threshold": self.scanned_char_threshold})
        pages = pymupdf_prov.parse(source)
        scanned_indices = {p["page_index"] for p in pages if p.get("is_scanned")}
        digital_indices = sorted(p["page_index"] for p in pages if not p.get("is_scanned"))

        paddle_http_cfg = (
            self.config.get("paddle_http") if isinstance(self.config.get("paddle_http"), dict) else {}
        )
        paddle_url = str(paddle_http_cfg.get("base_url") or os.environ.get("PADDLE_OCR_URL", "")).strip()

        if self.fallback_provider == "none":
            self.routed_providers = {"pymupdf": [p["page_index"] for p in pages]}
        elif (self.fallback_provider in ("", "paddle-http")) and paddle_url:
            from ocr.paddle_http_adapter import PaddleHTTPOCRProvider

            paddle_http = PaddleHTTPOCRProvider(base_url=paddle_url, config=self.config)
            try:
                self.service_info = paddle_http.health(timeout_s=min(paddle_http.health_timeout_s, 5.0))
            except Exception as exc:  # noqa: BLE001
                self.service_info = {"status": "unreachable", "error": str(exc)}
                if scanned_indices:
                    raise

            if scanned_indices:
                doc = pymupdf.open(source)
                try:
                    scanned_raw: list[dict] = []
                    for idx in sorted(scanned_indices):
                        scanned_raw.append(paddle_http.parse_page(doc[idx]))
                    scanned_norm = {
                        p["page_index"]: p
                        for p in normalize_pages(scanned_raw, coord_space="pixel", dpi=paddle_http.dpi)
                    }
                    pages = [scanned_norm.get(p["page_index"], p) for p in pages]
                    self.paddle_pages = sorted(scanned_indices)
                finally:
                    doc.close()
            if digital_indices:
                self.routed_providers["pymupdf"] = digital_indices
            if self.paddle_pages:
                self.routed_providers["paddle-http"] = list(self.paddle_pages)
        elif scanned_indices and (
            self.fallback_provider == "paddle-cpu" or len(scanned_indices) == len(pages)
        ):
            try:
                from ocr.paddle_adapter import PaddleOCRProvider

                paddle = PaddleOCRProvider(config=self.config)
                paddle_pages = {p["page_index"]: p for p in paddle.parse(source)}
                pages = [
                    paddle_pages[p["page_index"]] if p["page_index"] in scanned_indices else p
                    for p in pages
                ]
                self.paddle_pages = sorted(scanned_indices)
                if digital_indices:
                    self.routed_providers["pymupdf"] = digital_indices
                self.routed_providers["paddle-cpu"] = list(self.paddle_pages)
            except Exception:
                if self.fallback_provider == "paddle-cpu":
                    raise
                self.routed_providers = {"pymupdf": [p["page_index"] for p in pages]}
        else:
            self.routed_providers = {"pymupdf": [p["page_index"] for p in pages]}

        self.parse_seconds = round(time.perf_counter() - t0, 3)
        return pages
