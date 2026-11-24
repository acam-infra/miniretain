"""Unit tests for inline bold (**...**) and italic (*...*) style recognition and Typst rendering."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.mineru_cloud_adapter import _clean_html_text as mineru_clean_html
from ocr.normalize import normalize_pages
from ocr.paddle_cloud_adapter import _clean_html_text as paddle_clean_html
from ocr.pymupdf_adapter import _build_block_from_lines
from render.typst_renderer import render_typst, typst_rich_text_expr


def test_pymupdf_extracts_inline_bold_and_italic_runs_and_coalesces_wrapped_lines():
    lines = [
        {
            "bbox": [54.0, 100.0, 290.0, 112.0],
            "spans": [
                {"font": "NimbusRomNo9L-Medi", "flags": 20, "size": 10.0, "origin": [54.0, 109.0], "bbox": [54.0, 100.0, 100.0, 112.0], "text": "Batching."},
                {"font": "NimbusRomNo9L-Regu", "flags": 4, "size": 10.0, "origin": [104.0, 109.0], "bbox": [104.0, 100.0, 250.0, 112.0], "text": " Current serving systems use "},
                {"font": "NimbusRomNo9L-ReguItal", "flags": 6, "size": 10.0, "origin": [250.0, 109.0], "bbox": [250.0, 100.0, 290.0, 112.0], "text": "contin-"},
            ],
        },
        {
            "bbox": [54.0, 114.0, 290.0, 126.0],
            "spans": [
                {"font": "NimbusRomNo9L-ReguItal", "flags": 6, "size": 10.0, "origin": [54.0, 123.0], "bbox": [54.0, 114.0, 120.0, 126.0], "text": "uous batching"},
                {"font": "NimbusRomNo9L-Regu", "flags": 4, "size": 10.0, "origin": [120.0, 123.0], "bbox": [120.0, 114.0, 290.0, 126.0], "text": " [64] to improve GPU goodput."},
            ],
        },
    ]
    blk, _ = _build_block_from_lines("p3b0010", lines, 612.0)
    assert blk is not None
    assert blk["is_bold"] is False
    assert blk["is_italic"] is False
    assert blk["text"] == "**Batching.** Current serving systems use *continuous batching* [64] to improve GPU goodput."


def test_cloud_ocr_html_preserves_bold_and_italic_tags():
    raw = "<b>Prefill-decoding interference.</b> Adding a job slows down <i>continuous batching</i>."
    assert paddle_clean_html(raw) == "**Prefill-decoding interference.** Adding a job slows down *continuous batching*."
    assert mineru_clean_html(raw) == "**Prefill-decoding interference.** Adding a job slows down *continuous batching*."


def test_normalize_pages_keeps_styles_on_translatable_blocks_and_strips_on_preserved():
    pages = [
        {
            "page_index": 0,
            "width": 612.0,
            "height": 792.0,
            "figure_regions": [],
            "table_regions": [],
            "blocks": [
                {
                    "block_id": "b_cap",
                    "text": "**Figure 1:** Goodput under *synthetic* workload.",
                    "bbox": [54.0, 200.0, 290.0, 220.0],
                    "font_size_pt": 9.0,
                },
                {
                    "block_id": "b_code",
                    "text": "5 **eval_loader** = ZeusDataLoader(eval_set, batch_size)",
                    "bbox": [54.0, 260.0, 290.0, 275.0],
                    "font_size_pt": 9.0,
                },
            ],
        }
    ]
    norm = normalize_pages(pages, coord_space="pdf_pt", dpi=72.0)
    by_id = {b["block_id"]: b for b in norm[0]["blocks"]}
    assert by_id["b_cap"]["layout_role"] == "caption"
    assert by_id["b_cap"]["policy"]["translate"] is True
    assert "**Figure 1:**" in by_id["b_cap"]["text"]
    assert "*synthetic*" in by_id["b_cap"]["text"]

    assert by_id["b_code"]["policy"]["translate"] is False
    assert "**" not in by_id["b_code"]["text"]


def test_typst_rich_text_expr_renders_strong_and_emph_and_compiles_bold_italic_spans():
    expr = typst_rich_text_expr(
        r"**批处理。** 当前系统采用 *连续批处理*（*continuous batching*）处理矩阵 $Q$ ，计算 5 * 3 = 15。",
        use_mitex=True,
    )
    assert '#strong[#"批处理。"]' in expr
    assert '#emph[#"连续批处理"]' in expr
    assert '#emph[#"continuous batching"]' in expr
    assert "#mi(`Q`)" in expr
    assert "5 * 3 = 15" in expr

    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        src_pdf = wd / "src.pdf"
        doc = pymupdf.open()
        pg = doc.new_page(width=612, height=792)
        pg.insert_text((54, 100), "Batching. Current systems use continuous batching.", fontsize=10.0)
        doc.save(src_pdf)
        doc.close()

        res = render_typst(
            items=[
                {
                    "block_id": "p0b0",
                    "page_index": 0,
                    "reading_order": 0,
                    "layout_role": "body",
                    "policy": {"translate": True, "preserve_reason": None, "redact_source": True},
                    "bbox": [54.0, 88.0, 300.0, 140.0],
                    "line_bboxes": [[54.0, 88.0, 300.0, 140.0]],
                    "font_size_pt": 10.0,
                    "text": "**Batching.** Current systems use *continuous batching*.",
                    "translation": "**批处理。** 当前系统采用 *连续批处理*（*continuous batching*）提升吞吐量。",
                    "translation_status": "ok",
                }
            ],
            workdir=wd,
            output_cn=wd / "cn.pdf",
            output_merged=wd / "merged.pdf",
            legacy_output=None,
            font="",
            source=str(src_pdf),
            layout_mode="preserve",
            bilingual_mode="side_by_side",
            page_widths={0: 612.0},
            page_coord_spaces={0: "pdf_pt"},
            page_scanned={0: False},
            strict=True,
        )
        assert res["placed"] == 1
        assert res["unplaced"] == 0

        out_doc = pymupdf.open(wd / "cn.pdf")
        spans = [
            s
            for b in out_doc[0].get_text("dict")["blocks"]
            for l in b.get("lines", [])
            for s in l.get("spans", [])
            if (s.get("text") or "").strip()
        ]
        out_doc.close()
        bold_spans = [s for s in spans if "Bold" in (s.get("font") or "") or (int(s.get("flags", 0)) & 16)]
        italic_spans = [s for s in spans if "Italic" in (s.get("font") or "") or (int(s.get("flags", 0)) & 2)]
        assert any("批处理" in s["text"] for s in bold_spans), f"Expected bold span for 批处理, got {spans}"
        assert any("continuous batching" in s["text"] for s in italic_spans), f"Expected italic span, got {spans}"
