"""Core OCR provider base class, error hierarchy, and document.v1 output contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

ALLOWED_COORD_SPACES = frozenset({"pdf_pt", "pixel"})


class OCRError(RuntimeError):
    """Base error for all OCR provider failures."""


class OCRConfigError(OCRError, ValueError):
    """Raised when an OCR provider name is unknown or its configuration is invalid."""


class OCRBackendUnavailableError(OCRError):
    """Raised when an external OCR backend is unreachable or returns an HTTP/service error."""


class OCRContractError(OCRError, ValueError):
    """Raised when an OCR provider returns pages that violate the document.v1 contract."""


class OCRProvider(ABC):
    """Abstract base class for all Miniretain OCR backends."""

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def parse(self, source: str) -> list[dict]:
        """Parse ``source`` (PDF path or inline text) into normalized ``document.v1`` pages."""


def validate_document_pages(pages: list[dict]) -> list[dict]:
    """Validate that ``pages`` conforms to the Miniretain ``document.v1`` contract."""
    if not isinstance(pages, list):
        raise OCRContractError(f"document.v1 pages must be a list, got {type(pages).__name__}")

    for idx, page in enumerate(pages):
        if not isinstance(page, dict):
            raise OCRContractError(f"pages[{idx}] must be a dict, got {type(page).__name__}")
        page_index = page.get("page_index")
        if not isinstance(page_index, int) or isinstance(page_index, bool) or page_index < 0:
            raise OCRContractError(f"pages[{idx}].page_index must be a non-negative int, got {page_index!r}")

        for dim in ("width", "height"):
            val = page.get(dim)
            if not isinstance(val, (int, float)) or isinstance(val, bool) or val <= 0:
                raise OCRContractError(f"pages[{idx}].{dim} must be > 0, got {val!r}")

        coord_space = page.get("coord_space")
        if coord_space not in ALLOWED_COORD_SPACES:
            raise OCRContractError(
                f"pages[{idx}].coord_space={coord_space!r} not in {sorted(ALLOWED_COORD_SPACES)}"
            )
        dpi = page.get("dpi")
        if not isinstance(dpi, (int, float)) or isinstance(dpi, bool) or dpi <= 0:
            raise OCRContractError(f"pages[{idx}].dpi must be > 0, got {dpi!r}")

        if not isinstance(page.get("is_scanned"), bool):
            raise OCRContractError(f"pages[{idx}].is_scanned must be bool")
        if not isinstance(page.get("ocr_reliable"), bool):
            raise OCRContractError(f"pages[{idx}].ocr_reliable must be bool")

        blocks = page.get("blocks")
        if not isinstance(blocks, list):
            raise OCRContractError(f"pages[{idx}].blocks must be a list")

        for b_idx, blk in enumerate(blocks):
            if not isinstance(blk, dict):
                raise OCRContractError(f"pages[{idx}].blocks[{b_idx}] must be a dict")
            if not isinstance(blk.get("block_id"), str) or not blk["block_id"]:
                raise OCRContractError(f"pages[{idx}].blocks[{b_idx}].block_id must be a non-empty string")
            if not isinstance(blk.get("text"), str):
                raise OCRContractError(f"pages[{idx}].blocks[{b_idx}].text must be a string")
            if not isinstance(blk.get("layout_role"), str) or not blk["layout_role"]:
                raise OCRContractError(f"pages[{idx}].blocks[{b_idx}].layout_role must be a non-empty string")

            bbox = blk.get("bbox")
            if bbox is not None:
                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox)
                ):
                    raise OCRContractError(
                        f"pages[{idx}].blocks[{b_idx}].bbox must be None or [x0, y0, x1, y1], got {bbox!r}"
                    )
                if bbox[0] > bbox[2] + 1e-3 or bbox[1] > bbox[3] + 1e-3:
                    raise OCRContractError(
                        f"pages[{idx}].blocks[{b_idx}].bbox has inverted coordinates: {bbox!r}"
                    )

            policy = blk.get("policy")
            if not isinstance(policy, dict) or not isinstance(policy.get("translate"), bool):
                raise OCRContractError(
                    f"pages[{idx}].blocks[{b_idx}].policy must be a dict with boolean 'translate'"
                )

    return pages
