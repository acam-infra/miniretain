"""Side-by-side and interleaved bilingual PDF composition (RetainPDF-aligned)."""
from __future__ import annotations

from pathlib import Path
import pymupdf


def _normalize_page_rotations(doc: pymupdf.Document) -> None:
    """Bake /Rotate into page content so show_pdf_page aligns left/right pages."""
    for page in doc:
        if page.rotation:
            page.remove_rotation()


def build_side_by_side_pdf(
    source_pdf: str | Path,
    translated_pdf: str | Path,
    output_pdf: str | Path,
) -> dict:
    """Compose a left-original / right-translated side-by-side PDF."""
    src_path = Path(source_pdf)
    trans_path = Path(translated_pdf)
    out_path = Path(output_pdf)
    if not src_path.is_file():
        raise FileNotFoundError(f"source pdf not found: {src_path}")
    if not trans_path.is_file():
        raise FileNotFoundError(f"translated pdf not found: {trans_path}")

    with pymupdf.open(src_path) as source_doc, pymupdf.open(trans_path) as translated_doc:
        if source_doc.page_count < 1:
            raise RuntimeError(f"source pdf has no pages: {src_path}")
        if translated_doc.page_count < 1:
            raise RuntimeError(f"translated pdf has no pages: {trans_path}")
        _normalize_page_rotations(source_doc)
        _normalize_page_rotations(translated_doc)
        page_count = max(source_doc.page_count, translated_doc.page_count)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        page_sizes: list[tuple[float, float]] = []

        with pymupdf.open() as out_doc:
            for page_index in range(page_count):
                s_rect = source_doc[page_index].rect if page_index < source_doc.page_count else None
                t_rect = translated_doc[page_index].rect if page_index < translated_doc.page_count else None
                left_w = float(s_rect.width if s_rect is not None else t_rect.width)
                right_w = float(t_rect.width if t_rect is not None else s_rect.width)
                page_h = max(
                    float(s_rect.height if s_rect is not None else 0.0),
                    float(t_rect.height if t_rect is not None else 0.0),
                )
                out_page = out_doc.new_page(width=left_w + right_w, height=page_h)
                if s_rect is not None:
                    out_page.show_pdf_page(
                        pymupdf.Rect(0, 0, left_w, page_h),
                        source_doc,
                        page_index,
                        keep_proportion=True,
                    )
                if t_rect is not None:
                    out_page.show_pdf_page(
                        pymupdf.Rect(left_w, 0, left_w + right_w, page_h),
                        translated_doc,
                        page_index,
                        keep_proportion=True,
                    )
                page_sizes.append((round(left_w + right_w, 2), round(page_h, 2)))
            out_doc.save(out_path, garbage=4, deflate=True)

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"side-by-side output missing or empty: {out_path}")
    with pymupdf.open(out_path) as verify_doc:
        if verify_doc.page_count != page_count:
            raise RuntimeError(
                f"side-by-side page count mismatch: expected {page_count}, got {verify_doc.page_count}"
            )
    return {
        "bilingual_mode": "side_by_side",
        "pages": page_count,
        "page_sizes": page_sizes,
        "output": str(out_path),
    }


def merge_interleaved(
    source_pdf: str | Path,
    translated_pdf: str | Path,
    output_pdf: str | Path,
) -> dict:
    """Interleave original and translated pages: p0_orig, p0_trans, p1_orig, p1_trans..."""
    src_path = Path(source_pdf)
    trans_path = Path(translated_pdf)
    out_path = Path(output_pdf)
    with pymupdf.open(src_path) as doc_orig, pymupdf.open(trans_path) as doc_trans:
        if doc_orig.page_count < 1 or doc_trans.page_count < 1:
            raise RuntimeError("interleaved merge requires non-empty source and translated PDFs")
        max_p = max(len(doc_orig), len(doc_trans))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open() as doc_out:
            for i in range(max_p):
                if i < len(doc_orig):
                    doc_out.insert_pdf(doc_orig, from_page=i, to_page=i)
                if i < len(doc_trans):
                    doc_out.insert_pdf(doc_trans, from_page=i, to_page=i)
            doc_out.save(out_path, deflate=True)
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"interleaved output missing or empty: {out_path}")
    return {
        "bilingual_mode": "interleaved",
        "pages": max_p * 2,
        "output": str(out_path),
    }
