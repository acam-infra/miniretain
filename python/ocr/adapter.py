"""OCR adapter boundary.

A real implementation (PyMuPDF / PaddleOCR / MinerU) only has to implement
OCRProvider.parse(source) -> pages in the document.v1 shape below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ocr.normalize import normalize_pages


class OCRProvider(ABC):
    name = "base"

    @abstractmethod
    def parse(self, source: str) -> list[dict]:
        """Return pages in document.v1 format."""


class StubOCRProvider(OCRProvider):
    """Offline stand-in: treats the source text as an already-OCR'd page."""

    name = "stub"

    def parse(self, source: str) -> list[dict]:
        blocks = []
        for para in source.split("\n\n"):
            text = " ".join(line.strip() for line in para.splitlines() if line.strip())
            if text:
                blocks.append({"block_id": f"b{len(blocks):04d}", "kind": "text", "text": text})
        raw_pages = [{"page_index": 0, "width": 595, "height": 842, "blocks": blocks}]
        return normalize_pages(raw_pages, coord_space="pdf_pt", dpi=72.0)


from ocr.paddle_adapter import PaddleOCRProvider  # noqa: E402,F401
from ocr.pymupdf_adapter import AutoOCRProvider, PyMuPDFOCRProvider  # noqa: E402,F401

REGISTRY: dict[str, OCRProvider] = {
    StubOCRProvider.name: StubOCRProvider(),
    PyMuPDFOCRProvider.name: PyMuPDFOCRProvider(),
    AutoOCRProvider.name: AutoOCRProvider(),
}


class _LazyPaddle(PaddleOCRProvider):
    """Registered lazily: heavy models load on first parse(), not at import."""

    def __init__(self) -> None:
        self._ocr = None
        self._fitz = None

    def parse(self, source: str) -> list[dict]:
        if self._ocr is None:
            super().__init__()
        return super().parse(source)


REGISTRY[_LazyPaddle.name] = _LazyPaddle()


def get_provider(name: str) -> OCRProvider:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown OCR provider {name!r}; available: {sorted(REGISTRY)}") from exc
