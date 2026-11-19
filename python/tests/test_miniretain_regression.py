"""Regression tests for Miniretain P0/P1 fixes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
import pytest

PY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_ROOT))

from llm.formula_protect import protect_formulas, restore_formulas  # noqa: E402
from llm.scheduler import LLMClient, Scheduler  # noqa: E402
from ocr.pymupdf_adapter import PyMuPDFOCRProvider  # noqa: E402
from render.layout import find_cjk_font, render_pdf  # noqa: E402
from render.side_by_side import build_side_by_side_pdf  # noqa: E402
from render.typst_renderer import render_typst  # noqa: E402
from worker import run_llm, run_ocr  # noqa: E402


def _make_pdf(path: Path, width: float = 612.0, height: float = 792.0) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    # Draw a colored figure box (top region) with labels inside
    fig_rect = pymupdf.Rect(100, 60, 400, 200)
    page.draw_rect(fig_rect, color=(0.1, 0.2, 0.8), fill=(0.85, 0.92, 1.0), width=1.5)
    page.insert_text((130, 110), "Router", fontsize=10, color=(0, 0, 0))
    page.insert_text((220, 110), "FFN", fontsize=10, color=(0, 0, 0))
    page.insert_text((130, 150), "O(GB)", fontsize=10, color=(0, 0, 0))
    # Caption below figure
    page.insert_text((50, 230), "Fig. 1: System overview.", fontsize=10, color=(0, 0, 0))
    # Body paragraph
    page.insert_textbox(
        pymupdf.Rect(50, 260, 300, 330),
        "Disaggregated inference separates prefill and decode stages for better throughput.",
        fontsize=10,
    )
    # Tiny 42x13 box
    page.insert_textbox(pymupdf.Rect(50, 360, 92, 373), "Tiny", fontsize=8)
    doc.save(path)
    doc.close()
    return path


def test_p0_overflow_preserves_source_and_reports_id(tmp_path: Path):
    src = tmp_path / "tiny.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((52, 60), "Orig", fontsize=8)
    doc.save(src)
    doc.close()

    font = find_cjk_font()
    payload = [{
        "block_id": "b_tiny",
        "page_index": 0,
        "bbox": [50.0, 50.0, 92.0, 63.0],
        "line_bboxes": [[50.0, 50.0, 92.0, 63.0]],
        "text": "Orig",
        "translation": "这是一段非常非常长的中文译文，绝对不可能放进一个只有42乘以13磅的极小文本框里面去。",
        "translation_status": "ok",
        "policy": {"translate": True, "redact_source": True},
    }]
    out_cn = tmp_path / "out_cn.pdf"
    stats = render_pdf(
        source=str(src),
        payload=payload,
        out=str(out_cn),
        font=font,
        page_widths={0: 612.0},
        page_coord_spaces={0: "pdf_pt"},
        strict=False,
    )
    assert stats["placed"] == 0
    assert stats["unplaced"] == 1
    assert stats["unplaced_block_ids"] == ["b_tiny"]

    # Source text must STILL be preserved (not silently erased!)
    out_doc = pymupdf.open(out_cn)
    assert "Orig" in out_doc[0].get_text()
    out_doc.close()

    # Strict mode must fail explicitly
    with pytest.raises(RuntimeError, match="b_tiny"):
        render_pdf(
            source=str(src),
            payload=payload,
            out=str(tmp_path / "strict.pdf"),
            font=font,
            page_widths={0: 612.0},
            page_coord_spaces={0: "pdf_pt"},
            strict=True,
        )


def test_p0_min_size_is_actually_tried(tmp_path: Path):
    src = tmp_path / "minsize.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(50, 50, 110, 62), "Hello", fontsize=8)
    doc.save(src)
    doc.close()

    font = find_cjk_font()
    payload = [{
        "block_id": "b_min",
        "page_index": 0,
        "bbox": [50.0, 50.0, 110.0, 62.0],
        "line_bboxes": [[50.0, 50.0, 110.0, 62.0]],
        "text": "Hello",
        "translation": "测试最小字号适配能力",
        "translation_status": "ok",
        "policy": {"translate": True, "redact_source": True},
    }]
    out_cn = tmp_path / "min_cn.pdf"
    stats = render_pdf(
        source=str(src),
        payload=payload,
        out=str(out_cn),
        font=font,
        page_widths={0: 612.0},
        page_coord_spaces={0: "pdf_pt"},
        min_size=5.0,
    )
    assert stats["placed"] == 1
    assert stats["unplaced"] == 0


class _FailingClient(LLMClient):
    name = "failing-test"

    def translate(self, items, target_lang, model, timeout_ms):
        raise RuntimeError("simulated upstream 503")


class _SentinelClient(LLMClient):
    name = "sentinel-test"

    def translate(self, items, target_lang, model, timeout_ms):
        return ["<translation failed: boom>" for _ in items]


def test_p0_scheduler_never_counts_failure_as_translated():
    items = [{"block_id": "b0", "text": "Hello"}]
    records, stats = Scheduler(_FailingClient(), concurrency=1, batch_size=1, max_retries=0, timeout_ms=1000).run(items, "zh", "m")
    assert stats["failed_batches"] == 1
    assert stats["succeeded_items"] == 0
    assert stats["failed_items"] == 1
    assert stats["failed_block_ids"] == ["b0"]
    assert records[0]["status"] == "failed"

    records2, stats2 = Scheduler(_SentinelClient(), concurrency=1, batch_size=1, max_retries=0, timeout_ms=1000).run(items, "zh", "m")
    assert stats2["succeeded_items"] == 0
    assert stats2["failed_items"] == 1
    assert records2[0]["status"] == "failed"


def test_p0_worker_run_llm_fails_on_incomplete(tmp_path: Path):
    wd = tmp_path / "job"
    wd.mkdir()
    (wd / "document.v1.json").write_text(json.dumps({
        "version": "document.v1",
        "pages": [{
            "page_index": 0,
            "width": 612.0,
            "height": 792.0,
            "blocks": [{
                "block_id": "b0",
                "text": "Hello",
                "layout_role": "body",
                "policy": {"translate": True, "redact_source": True},
            }],
        }],
    }), encoding="utf-8")
    spec = {
        "job_id": "j1",
        "stage": "llm",
        "workdir": str(wd),
        "source": "Hello",
        "target_lang": "zh",
        "ocr": {"provider": "stub"},
        "llm": {
            "provider": "openai-compat",
            "concurrency": 1,
            "batch_size": 1,
            "max_retries": 0,
            "timeout_ms": 200,
            "model": "m",
            "allow_partial": False,
        },
    }
    import os
    old_url = os.environ.get("OPENAI_BASE_URL")
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:19/v1"
    try:
        with pytest.raises(RuntimeError, match="b0"):
            run_llm(spec, wd)
    finally:
        if old_url is None:
            os.environ.pop("OPENAI_BASE_URL", None)
        else:
            os.environ["OPENAI_BASE_URL"] = old_url


def test_p0_side_by_side_fails_loudly_on_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_side_by_side_pdf(tmp_path / "missing.pdf", tmp_path / "b.pdf", tmp_path / "out.pdf")


def test_p1_figure_protection_and_typst_preserve_side_by_side(tmp_path: Path):
    src = _make_pdf(tmp_path / "fig_paper.pdf")
    wd = tmp_path / "wd"
    wd.mkdir()
    spec = {
        "job_id": "jfig",
        "stage": "ocr",
        "workdir": str(wd),
        "source": str(src),
        "target_lang": "zh",
        "ocr": {"provider": "pymupdf"},
        "llm": {
            "provider": "stub",
            "concurrency": 1,
            "batch_size": 8,
            "max_retries": 0,
            "timeout_ms": 5000,
            "model": "stub-v1",
            "allow_partial": False,
        },
    }
    ocr_res = run_ocr(spec, wd)
    assert ocr_res["roles"].get("figure_label", 0) >= 2
    assert ocr_res["roles"].get("caption", 0) == 1

    llm_res = run_llm(spec, wd)
    assert llm_res["failed_total"] == 0
    assert llm_res["preserved"] >= 2

    payload = json.loads((wd / "translation.payload.json").read_text(encoding="utf-8"))
    out_cn = wd / "cn.pdf"
    out_sbs = wd / "sbs.pdf"
    render_res = render_typst(
        items=payload["items"],
        workdir=wd,
        output_cn=out_cn,
        output_merged=out_sbs,
        source=str(src),
        layout_mode="preserve",
        bilingual_mode="side_by_side",
        page_widths={0: 612.0},
        page_coord_spaces={0: "pdf_pt"},
        page_scanned={0: False},
    )
    assert render_res["layout_mode"] == "preserve"
    assert render_res["unplaced"] == 0
    assert out_cn.is_file() and out_sbs.is_file()

    with pymupdf.open(out_cn) as cn_doc, pymupdf.open(out_sbs) as sbs_doc:
        assert len(cn_doc) == 1
        assert (round(cn_doc[0].rect.width), round(cn_doc[0].rect.height)) == (612, 792)
        txt = cn_doc[0].get_text()
        assert "Router" in txt and "FFN" in txt and "O(GB)" in txt
        assert len(sbs_doc) == 1
        assert (round(sbs_doc[0].rect.width), round(sbs_doc[0].rect.height)) == (1224, 792)


def test_p1_scanned_pdf_marks_unreliable_and_protects(tmp_path: Path):
    src = tmp_path / "scanned.pdf"
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200), False)
    pix.set_rect(pix.irect, (255, 255, 255))
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(pymupdf.Rect(50, 50, 550, 750), pixmap=pix)
    doc.save(src)
    doc.close()

    pages = PyMuPDFOCRProvider().parse(str(src))
    assert len(pages) == 1
    assert pages[0]["is_scanned"] is True
    assert pages[0]["ocr_reliable"] is False


def test_p1_formula_protect_roundtrip():
    src = "Memory is $O(GB)$ and bandwidth is O(KB) per layer [1]."
    prot, toks = protect_formulas(src)
    assert "$O(GB)$" not in prot
    assert "O(KB)" not in prot
    restored, missing = restore_formulas(f"内存为 {prot}", toks)
    assert missing == []
    assert "$O(GB)$" in restored and "O(KB)" in restored


def test_p1_vertical_axis_label_vs_body_text(tmp_path: Path):
    src = tmp_path / "axis.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    # Chart frame + rotated vertical axis label
    page.draw_rect(pymupdf.Rect(90, 60, 300, 220), color=(0, 0, 0), width=1.0)
    page.insert_text((65, 160), "of TTFT (%)", fontsize=8, rotate=90)
    # Normal horizontal body text outside chart
    page.insert_textbox(
        pymupdf.Rect(50, 260, 300, 310),
        "Normal horizontal body text outside the chart.",
        fontsize=10,
    )
    doc.save(src)
    doc.close()

    pages = PyMuPDFOCRProvider().parse(str(src))
    blocks = pages[0]["blocks"]
    axis_blocks = [b for b in blocks if "TTFT" in b["text"]]
    body_blocks = [b for b in blocks if "Normal horizontal" in b["text"]]
    assert len(axis_blocks) == 1
    assert axis_blocks[0]["layout_role"] == "figure_label"
    assert axis_blocks[0]["policy"]["translate"] is False
    assert axis_blocks[0]["policy"]["preserve_reason"] == "vertical_figure_label"
    assert len(body_blocks) == 1
    assert body_blocks[0]["layout_role"] == "body"
    assert body_blocks[0]["policy"]["translate"] is True


def test_p1_reference_field_protection_and_clickable_link(tmp_path: Path):
    src = tmp_path / "ref.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    ref_rect = pymupdf.Rect(50, 100, 300, 150)
    page.insert_textbox(
        ref_rect,
        '[9] N. Shazeer and J. Dean, “Sparse Mixture-of-Experts,” ICLR, 2017. https://openreview.net/forum?id=B1ckMDqlg',
        fontsize=8,
    )
    page.insert_link({
        "kind": pymupdf.LINK_URI,
        "from": pymupdf.Rect(50, 125, 260, 140),
        "uri": "https://openreview.net/forum?id=B1ckMDqlg",
    })
    doc.save(src)
    doc.close()

    prot, toks = protect_formulas(
        '[9] N. Shazeer and J. Dean, “Sparse Mixture-of-Experts,” ICLR, 2017. https://openreview.net/forum?id=B1ckMDqlg',
        layout_role="reference",
    )
    assert "[9]" not in prot
    assert "N. Shazeer" not in prot
    assert "https://openreview.net/forum?id=B1ckMDqlg" not in prot
    restored, missing = restore_formulas(prot, toks)
    assert missing == []
    assert "https://openreview.net/forum?id=B1ckMDqlg" in restored

    out_cn = tmp_path / "ref_cn.pdf"
    out_sbs = tmp_path / "ref_sbs.pdf"
    items = [{
        "block_id": "p0b0000",
        "page_index": 0,
        "bbox": [50.0, 100.0, 300.0, 150.0],
        "line_bboxes": [[50.0, 100.0, 300.0, 150.0]],
        "layout_role": "reference",
        "font_size_pt": 8.0,
        "text": "orig",
        "translation": '[9] N. Shazeer and J. Dean，“稀疏混合专家模型，”ICLR，2017。https://openreview.net/forum?id=B1ckMDqlg',
        "translation_status": "ok",
        "policy": {"translate": True, "redact_source": True},
    }]
    res = render_typst(
        items=items,
        workdir=tmp_path,
        output_cn=out_cn,
        output_merged=out_sbs,
        source=str(src),
        layout_mode="preserve",
        bilingual_mode="side_by_side",
        page_widths={0: 612.0},
        page_coord_spaces={0: "pdf_pt"},
        strict=True,
    )
    assert res["placed"] == 1
    assert res["unplaced"] == 0
    with pymupdf.open(out_cn) as cdoc:
        uris = [lk.get("uri") for lk in cdoc[0].get_links() if lk.get("uri")]
        assert "https://openreview.net/forum?id=B1ckMDqlg" in uris


def test_p1_typst_preserve_rejects_out_of_bounds_and_neighbor_overlap(tmp_path: Path):
    src = tmp_path / "bounds.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((52, 782), "EdgeOrig", fontsize=8)
    doc.save(src)
    doc.close()

    # Box near bottom edge with huge translation -> cannot fit at >= 5pt without overflowing
    items = [{
        "block_id": "b_edge",
        "page_index": 0,
        "bbox": [50.0, 772.0, 120.0, 785.0],
        "line_bboxes": [[50.0, 772.0, 120.0, 785.0]],
        "layout_role": "body",
        "text": "EdgeOrig",
        "translation": "这是一段非常长且绝对会超出页面底部边界或盒子高度的译文内容，不能被裁切或强行缩到5pt以下。",
        "translation_status": "ok",
        "policy": {"translate": True, "redact_source": True},
    }]
    out_cn = tmp_path / "edge_cn.pdf"
    out_sbs = tmp_path / "edge_sbs.pdf"
    res = render_typst(
        items=items,
        workdir=tmp_path,
        output_cn=out_cn,
        output_merged=out_sbs,
        source=str(src),
        layout_mode="preserve",
        page_widths={0: 612.0},
        page_coord_spaces={0: "pdf_pt"},
        strict=False,
    )
    assert res["placed"] == 0
    assert res["unplaced"] == 1
    assert res["unplaced_block_ids"] == ["b_edge"]
    with pymupdf.open(out_cn) as cdoc:
        assert "EdgeOrig" in cdoc[0].get_text()

    with pytest.raises(RuntimeError, match="b_edge"):
        render_typst(
            items=items,
            workdir=tmp_path,
            output_cn=tmp_path / "edge_strict.pdf",
            output_merged=tmp_path / "edge_strict_sbs.pdf",
            source=str(src),
            layout_mode="preserve",
            page_widths={0: 612.0},
            page_coord_spaces={0: "pdf_pt"},
            strict=True,
        )
