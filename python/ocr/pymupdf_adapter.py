"""PyMuPDF native PDF layout extractor and Auto OCR provider."""
from __future__ import annotations

import time
import pymupdf

from ocr.normalize import cluster_figure_regions, normalize_pages

_MATH_FONT_TOKENS = ("CMMI", "CMSY", "CMEX", "MSAM", "MSBM", "SYMBOL", "MTMI", "MTSY")


def _extract_page_blocks(page: pymupdf.Page) -> tuple[list[dict], list[list[float]], bool]:
    """Extract paragraph/line blocks and figure regions from a digital PDF page."""
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
        line_bboxes: list[list[float]] = []
        line_texts: list[str] = []
        span_sizes: list[float] = []
        bold_chars = 0
        math_chars = 0
        char_count = 0

        for line in lines:
            spans = line.get("spans") or []
            lt = "".join(s.get("text", "") for s in spans).strip()
            if not lt:
                continue
            line_texts.append(lt)
            if line.get("bbox"):
                line_bboxes.append([float(v) for v in line["bbox"]])
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

        text = " ".join(line_texts).strip()
        if not text:
            continue
        # Fix hyphenated line breaks like "disag- gregation"
        text = text.replace("- ", "")
        total_chars += len(text)
        bbox = [float(v) for v in b["bbox"]]
        avg_size = sum(span_sizes) / len(span_sizes) if span_sizes else 10.0
        blocks.append({
            "block_id": f"p{page.number}b{idx:04d}",
            "text": text,
            "bbox": bbox,
            "line_bboxes": line_bboxes or [bbox],
            "font_size_pt": round(avg_size, 2),
            "is_bold": bold_chars > char_count * 0.6 if char_count else False,
            "is_math_font": math_chars > char_count * 0.5 if char_count else False,
        })

    is_scanned = total_chars < 20 and len(image_rects) > 0
    return blocks, figure_regions, is_scanned


class PyMuPDFOCRProvider:
    """Native PDF layout extractor using PyMuPDF (fast, preserves fonts & drawings)."""

    name = "pymupdf"

    def parse(self, source: str) -> list[dict]:
        t0 = time.time()
        doc = pymupdf.open(source)
        raw_pages: list[dict] = []
        try:
            for page in doc:
                blocks, figure_regions, is_scanned = _extract_page_blocks(page)
                raw_pages.append({
                    "page_index": page.number,
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "figure_regions": figure_regions,
                    "is_scanned": is_scanned,
                    "ocr_reliable": not is_scanned,
                    "blocks": blocks,
                })
        finally:
            doc.close()
        self.parse_seconds = round(time.time() - t0, 2)
        return normalize_pages(raw_pages, coord_space="pdf_pt", dpi=72.0)


class AutoOCRProvider:
    """Uses native PyMuPDF on digital PDFs; falls back or marks unreliable on scanned PDFs."""

    name = "auto"

    def parse(self, source: str) -> list[dict]:
        if not source.lower().endswith(".pdf"):
            from ocr.adapter import StubOCRProvider
            return StubOCRProvider().parse(source)
        pymupdf_prov = PyMuPDFOCRProvider()
        pages = pymupdf_prov.parse(source)
        scanned_pages = [p["page_index"] for p in pages if p.get("is_scanned")]
        if scanned_pages and all(p.get("is_scanned") for p in pages):
            # Try paddle-cpu if installed
            try:
                from ocr.paddle_adapter import PaddleOCRProvider
                paddle = PaddleOCRProvider()
                return paddle.parse(source)
            except Exception:
                # Keep unreliable_ocr markers so source content is protected
                return pages
        return pages
