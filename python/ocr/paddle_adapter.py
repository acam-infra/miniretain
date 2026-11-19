"""PaddleOCR CPU provider: PDF -> document.v1 pages."""
from __future__ import annotations

import os
import time

from ocr.base import OCRProvider
from ocr.normalize import normalize_pages


class PaddleOCRProvider(OCRProvider):
    name = "paddle-cpu"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        sub = self.config.get("paddle_cpu") if isinstance(self.config.get("paddle_cpu"), dict) else {}
        self.dpi = float(sub.get("dpi") if "dpi" in sub else self.config.get("dpi", 150.0))
        self.lang = str(sub.get("lang") or self.config.get("lang") or "en")
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        from paddleocr import PaddleOCR  # heavy import, lazy

        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=self.lang,
            enable_mkldnn=False,
        )
        import fitz

        self._fitz = fitz

    def parse(self, source: str) -> list[dict]:
        t0 = time.time()
        doc = self._fitz.open(source)
        pages: list[dict] = []
        for page in doc:
            pix = page.get_pixmap(dpi=int(self.dpi))
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
                "is_scanned": True,
                "ocr_reliable": True,
                "blocks": blocks,
            })
        self.parse_seconds = round(time.time() - t0, 2)
        return normalize_pages(pages, coord_space="pixel", dpi=self.dpi)
