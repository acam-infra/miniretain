"""Core OCR provider base class, error hierarchy, and document.v1 output contract."""

from __future__ import annotations

import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any

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


class DeadlineFileReader:
    """Binary stream wrapper enforcing a monotonic wall-clock deadline while exposing __len__."""

    def __init__(self, stream: IO[bytes], size: int, deadline: float) -> None:
        self._stream = stream
        self._size = int(size)
        self._deadline = float(deadline)

    def __len__(self) -> int:
        return self._size

    def read(self, amt: int = -1) -> bytes:
        if time.monotonic() >= self._deadline:
            raise TimeoutError("OCR upload wall-clock deadline exceeded")
        if amt is None or amt < 0:
            amt = 65536
        return self._stream.read(amt)


def read_stream_with_deadline(resp: Any, deadline: float, *, chunk_size: int = 65536) -> bytes:
    """Read an HTTP response stream in chunks, aborting immediately if wall-clock deadline passes."""
    chunks: list[bytes] = []
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("OCR download wall-clock deadline exceeded")
        chunk = resp.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def is_digital_pdf(source: str, scanned_char_threshold: int = 20) -> bool:
    """Return True if ``source`` is a local PDF with reliable native digital text across all pages."""
    if not source or source.startswith(("http://", "https://")):
        return False
    p = Path(source)
    if not p.is_file() or p.suffix.lower() != ".pdf":
        return False
    try:
        import pymupdf

        doc = pymupdf.open(str(p))
    except Exception:
        return False
    try:
        if len(doc) == 0:
            return False
        total_chars = 0
        for page in doc:
            txt = (page.get_text("text") or "").strip()
            total_chars += len(txt)
            if len(txt) < scanned_char_threshold:
                raw_dict = page.get_text("dict")
                has_image = any(b.get("type") == 1 for b in raw_dict.get("blocks", []))
                if has_image or len(doc) == 1:
                    return False
        return total_chars >= max(20, scanned_char_threshold)
    except Exception:
        return False
    finally:
        doc.close()


def prepare_cloud_pdf(
    source: str,
    *,
    max_mb: float = 20.0,
    scanned_char_threshold: int = 20,
) -> tuple[str, str | None, dict[str, Any]]:
    """Compress oversized local PDFs into a temp copy for cloud OCR while preserving the lossless original.

    Returns ``(upload_path_or_url, temp_path_to_cleanup, meta)``.
    If even after compression the PDF exceeds ``max_mb`` and has a complete native digital text layer,
    ``meta["too_large_digital"]`` is set to ``True`` so callers can downgrade directly to ``pymupdf``.
    """
    if not source or source.startswith(("http://", "https://")) or max_mb <= 0:
        return source, None, {"compressed": False, "too_large_digital": False}
    src_path = Path(source)
    if not src_path.is_file() or src_path.suffix.lower() != ".pdf":
        return source, None, {"compressed": False, "too_large_digital": False}

    orig_bytes = src_path.stat().st_size
    orig_mb = round(orig_bytes / (1024 * 1024), 2)
    limit_bytes = int(max_mb * 1024 * 1024)
    if orig_bytes <= limit_bytes:
        return source, None, {"compressed": False, "orig_mb": orig_mb, "too_large_digital": False}

    fd, tmp_name = tempfile.mkstemp(prefix="miniretain_cloud_", suffix=".pdf")
    os.close(fd)
    comp_bytes = orig_bytes
    compressed_ok = False
    try:
        import pymupdf

        doc = pymupdf.open(str(src_path))
        try:
            doc.rewrite_images(
                dpi_threshold=96,
                dpi_target=72,
                quality=40,
                lossy=True,
                lossless=True,
                bitonal=True,
            )
            doc.save(tmp_name, garbage=4, deflate=True)
        finally:
            doc.close()
        comp_bytes = Path(tmp_name).stat().st_size
        compressed_ok = 0 < comp_bytes < orig_bytes
    except Exception:
        compressed_ok = False

    comp_mb = round(comp_bytes / (1024 * 1024), 2)
    if compressed_ok and comp_bytes <= limit_bytes:
        return tmp_name, tmp_name, {
            "compressed": True,
            "orig_mb": orig_mb,
            "comp_mb": comp_mb,
            "too_large_digital": False,
        }

    # Still over threshold (or compression failed): check if native digital text allows fast local downgrade.
    if is_digital_pdf(str(src_path), scanned_char_threshold=scanned_char_threshold):
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        return source, None, {
            "compressed": False,
            "orig_mb": orig_mb,
            "comp_mb": comp_mb,
            "too_large_digital": True,
        }

    if compressed_ok:
        return tmp_name, tmp_name, {
            "compressed": True,
            "orig_mb": orig_mb,
            "comp_mb": comp_mb,
            "too_large_digital": False,
        }

    try:
        os.unlink(tmp_name)
    except OSError:
        pass
    return source, None, {
        "compressed": False,
        "orig_mb": orig_mb,
        "too_large_digital": False,
    }
