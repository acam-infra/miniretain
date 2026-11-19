"""PaddleOCR CPU provider: PDF -> document.v1 pages."""
from __future__ import annotations

import os
import time

from ocr.normalize import normalize_pages


class PaddleOCRProvider:
    name = "paddle-cpu"

    def __init__(self) -> None:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        from paddleocr import PaddleOCR  # heavy import, lazy

        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
            enable_mkldnn=False,
        )
        import fitz

        self._fitz = fitz

    def parse(self, source: str) -> list[dict]:
        t0 = time.time()
        doc = self._fitz.open(source)
        pages: list[dict] = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            png = f"{source}.p{page.number}.png"
            pix.save(png)
            result = self._ocr.predict(png)
            os.remove(png)
            blocks = []
            for r in result:
                for i, (text, box) in enumerate(zip(r["rec_texts"], r["rec_boxes"])):
                    x0, y0, x1, y1 = (float(v) for v in box)
                    blocks.append({
                        "block_id": f"p{page.number}b{i:04d}",
                        "kind": "text",
                        "text": text,
                        "bbox": [x0, y0, x1, y1],
                    })
            pages.append({
                "page_index": page.number,
                "width": pix.width,
                "height": pix.height,
                "blocks": blocks,
            })
        self.parse_seconds = round(time.time() - t0, 2)
        return normalize_pages(pages, coord_space="pixel", dpi=150.0)
