"""OCR adapter boundary.

A real implementation (PaddleOCR / MinerU / GLM) only has to implement
OCRProvider.parse(source) -> pages in the document.v1 shape below. The rest of
the service does not change. The stub keeps this demo runnable offline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class OCRProvider(ABC):
    name = "base"

    @abstractmethod
    def parse(self, source: str) -> list[dict]:
        """Return pages: [{page_index, width, height, blocks:[{block_id, kind, text}, ...]}]."""


class StubOCRProvider(OCRProvider):
    """Offline stand-in: treats the source text as an already-OCR'd page."""

    name = "stub"

    def parse(self, source: str) -> list[dict]:
        blocks = []
        for para in source.split("\n\n"):
            text = " ".join(line.strip() for line in para.splitlines() if line.strip())
            if text:
                blocks.append({"block_id": f"b{len(blocks):04d}", "kind": "text", "text": text})
        return [{"page_index": 0, "width": 595, "height": 842, "blocks": blocks}]


from ocr.paddle_adapter import PaddleOCRProvider  # noqa: E402,F401

REGISTRY: dict[str, OCRProvider] = {p.name: p for p in [StubOCRProvider()]}


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
