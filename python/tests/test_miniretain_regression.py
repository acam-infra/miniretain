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
    with pymupdf.open(out_sbs) as sdoc:
        left_uris = [lk.get("uri") for lk in sdoc[0].get_links() if lk.get("uri") and lk["from"].x0 < 612.0]
        right_uris = [lk.get("uri") for lk in sdoc[0].get_links() if lk.get("uri") and lk["from"].x0 >= 612.0]
        assert "https://openreview.net/forum?id=B1ckMDqlg" in left_uris
        assert "https://openreview.net/forum?id=B1ckMDqlg" in right_uris


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


def test_p1_typst_unbreakable_bold_word_and_wrapped_url_fit(tmp_path: Path):
    src = tmp_path / "horiz.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((12, 20), "OrigWide", fontsize=8)
    page.insert_text((12, 80), "OrigURL", fontsize=8)
    doc.save(src)
    doc.close()

    items = [
        {
            "block_id": "b_unbreakable",
            "page_index": 0,
            "bbox": [10.0, 10.0, 55.0, 25.0],  # 45x15 pt
            "line_bboxes": [[10.0, 10.0, 55.0, 25.0]],
            "layout_role": "heading",
            "font_size_pt": 12.0,
            "text": "OrigWide",
            "translation": "WWWGGGGGGWWWW",
            "translation_status": "ok",
            "policy": {"translate": True, "redact_source": True},
        },
        {
            "block_id": "b_url",
            "page_index": 0,
            "bbox": [10.0, 70.0, 210.0, 110.0],  # 200x40 pt
            "line_bboxes": [[10.0, 70.0, 210.0, 110.0]],
            "layout_role": "reference",
            "font_size_pt": 8.0,
            "text": "OrigURL",
            "translation": "论文地址：https://www.sciencedirect.com/science/article/pii/S2542435126003624",
            "translation_status": "ok",
            "policy": {"translate": True, "redact_source": True},
        },
    ]
    out_cn = tmp_path / "h_cn.pdf"
    res = render_typst(
        items=items,
        workdir=tmp_path,
        output_cn=out_cn,
        output_merged=tmp_path / "h_sbs.pdf",
        source=str(src),
        layout_mode="preserve",
        page_widths={0: 612.0},
        page_coord_spaces={0: "pdf_pt"},
        strict=False,
    )
    assert res["unplaced_block_ids"] == ["b_unbreakable"]
    assert res["placed"] == 1
    with pymupdf.open(out_cn) as cdoc:
        txt = cdoc[0].get_text()
        assert "OrigWide" in txt
        assert "WWWGGGGGGWWWW" not in txt
        for blk in cdoc[0].get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    if "sciencedirect" in sp["text"]:
                        assert sp["bbox"][2] <= 211.0


def test_p0_two_column_cjk_opening_punct_no_cross_column_misattribution(tmp_path: Path):
    """Regression for LAB-2112 / job e77489d8524bc45e p0b0014 compiled_horizontal_overflow.

    Right-column block p0b0021 wraps a line starting with full-width CJK '（', which
    previously hung left of p0b0021.x0 and got misattributed to left-column p0b0014.
    """
    src = tmp_path / "twocol.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612.0, height=792.0)
    page.insert_textbox(pymupdf.Rect(48.96, 721.91, 300.02, 747.81), "This work was funded.", fontsize=8)
    page.insert_textbox(pymupdf.Rect(311.98, 702.46, 563.04, 748.29), "LLMs process tokens.", fontsize=9)
    doc.save(src)
    doc.close()

    items = [
        {
            "block_id": "p0b0014",
            "page_index": 0,
            "bbox": [48.964, 721.906, 300.022, 747.809],
            "line_bboxes": [[48.964, 721.906, 300.022, 747.809]],
            "layout_role": "body",
            "font_size_pt": 7.97,
            "text": "This work was funded by the European Union.",
            "translation": "本研究部分由欧盟“地平线欧洲”研究与创新计划根据 Marie Skłodowska-Curie 资助协议 No. 101282240 资助。",
            "translation_status": "ok",
            "policy": {"translate": True, "redact_source": True},
        },
        {
            "block_id": "p0b0021",
            "page_index": 0,
            "bbox": [311.978, 702.462, 563.036, 748.290],
            "line_bboxes": [[311.978, 702.462, 563.036, 748.290]],
            "layout_role": "body",
            "font_size_pt": 9.96,
            "text": "LLMs represent text as tokens and process them through Transformer layers.",
            "translation": "LLM 将文本表示为 token，并通过一系列 Transformer 层处理这些 token。每一层通常包括一个注意力块和一个前馈网络（FFN）块；注意力块将 token 彼此关联，",
            "translation_status": "ok",
            "policy": {"translate": True, "redact_source": True},
        },
    ]

    res = render_typst(
        items=items,
        workdir=tmp_path,
        output_cn=tmp_path / "twocol_cn.pdf",
        output_merged=tmp_path / "twocol_sbs.pdf",
        source=str(src),
        layout_mode="preserve",
        bilingual_mode="side_by_side",
        page_widths={0: 612.0},
        page_coord_spaces={0: "pdf_pt"},
        strict=True,
    )
    assert res["placed"] == 2
    assert res["unplaced"] == 0
    assert res["failed"] == 0
    assert "0.15." in res["typst_version"] or res["typst_version"].startswith("typst ")
    assert "timings_s" in res and res["timings_s"]["total_s"] >= 0


def test_paddle_http_adapter_and_auto_scanned_routing(tmp_path: Path, monkeypatch):
    """Verify paddle-http coordinate contract and auto routing on a scanned image-only PDF."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from ocr.adapter import get_provider

    class _MockPaddle(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            body = json.dumps({
                "status": "ok",
                "paddle_version": "3.3.1",
                "paddleocr_version": "3.7.0",
                "device": "cpu",
                "models": ["PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"],
                "coord_space": "pixel",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or "0")
            _ = self.rfile.read(n)
            body = json.dumps({
                "dpi": 150.0,
                "coord_space": "pixel",
                "bbox_format": "xyxy",
                "elapsed_s": 0.01,
                "blocks": [
                    {"text": "Fig. 1: Scanned architecture.", "bbox": [100.0, 480.0, 700.0, 520.0], "score": 0.99},
                    {"text": "Disaggregated inference runs on orbital nodes.", "bbox": [100.0, 560.0, 900.0, 620.0], "score": 0.98},
                ],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), _MockPaddle)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        # Create a 1-page rasterized (scanned) PDF with no embedded text layer
        vec = _make_pdf(tmp_path / "vec.pdf")
        scanned = tmp_path / "scanned.pdf"
        with pymupdf.open(vec) as vdoc:
            pix = vdoc[0].get_pixmap(dpi=150)
            sdoc = pymupdf.open()
            spage = sdoc.new_page(width=vdoc[0].rect.width, height=vdoc[0].rect.height)
            spage.insert_image(spage.rect, stream=pix.tobytes("png"))
            sdoc.save(scanned)
            sdoc.close()

        monkeypatch.setenv("PADDLE_OCR_URL", f"http://127.0.0.1:{port}")
        auto = get_provider("auto")
        pages = auto.parse(str(scanned))
        assert len(pages) == 1
        assert pages[0]["coord_space"] == "pixel"
        assert pages[0]["is_scanned"] is True
        assert pages[0]["ocr_reliable"] is True
        assert auto.paddle_pages == [0]
        assert auto.service_info["paddleocr_version"] == "3.7.0"
        assert len(pages[0]["blocks"]) == 2
    finally:
        srv.shutdown()
        srv.server_close()


def test_scheduler_balanced_batching_preserves_order_and_parallel_retry():
    """Verify character-balanced LPT batching preserves exact item ordering and retries missing items."""
    calls: list[list[str]] = []

    class _FlakyOrderClient(LLMClient):
        name = "flaky-order"

        def translate(self, items, target_lang, model, timeout_ms):
            ids = [it["block_id"] for it in items]
            calls.append(ids)
            out = []
            for it in items:
                if it["block_id"] == "b2" and len(items) > 1:
                    out.append({"block_id": "b2", "status": "missing", "translation": None, "error": "omitted"})
                else:
                    out.append(f"[{target_lang}] {it['text']}")
            return out

    items = [
        {"block_id": f"b{i}", "text": ("long paragraph " * 40) if i % 2 == 0 else f"short {i}"}
        for i in range(6)
    ]
    records, stats = Scheduler(
        _FlakyOrderClient(), concurrency=3, batch_size=3, max_retries=1, timeout_ms=5000
    ).run(items, "zh", "m")
    assert [r["block_id"] for r in records] == [f"b{i}" for i in range(6)]
    assert all(r["status"] == "ok" for r in records)
    assert stats["succeeded_items"] == 6
    assert stats["retries"] >= 1
    assert stats["elapsed_s"] >= 0


def test_ocr_provider_registration_and_config_passing():
    """Verify custom OCR backend registration, config forwarding, and error contracts."""
    from ocr.adapter import (
        OCRConfigError,
        OCRProvider,
        get_provider,
        list_providers,
        register_provider,
        unregister_provider,
        validate_document_pages,
    )
    from ocr.normalize import normalize_pages

    class _DemoCustomOCR(OCRProvider):
        name = "demo-custom"

        def parse(self, source: str) -> list[dict]:
            prefix = (self.config.get("options") or {}).get("prefix", "custom")
            raw = [{
                "page_index": 0,
                "width": 612.0,
                "height": 792.0,
                "blocks": [{
                    "block_id": "p0b0000",
                    "text": f"{prefix}: {source}",
                    "bbox": [50.0, 50.0, 300.0, 70.0],
                }],
            }]
            return normalize_pages(raw, coord_space="pdf_pt", dpi=72.0)

    try:
        register_provider("demo-custom", _DemoCustomOCR)
        assert "demo-custom" in list_providers()

        with pytest.raises(OCRConfigError):
            register_provider("demo-custom", _DemoCustomOCR)

        register_provider("demo-custom", _DemoCustomOCR, replace=True)
        prov = get_provider("demo-custom", config={"options": {"prefix": "v2"}})
        pages = validate_document_pages(prov.parse("hello"))
        assert pages[0]["blocks"][0]["text"] == "v2: hello"

        with pytest.raises(OCRConfigError):
            get_provider("no-such-backend")
    finally:
        unregister_provider("demo-custom")


def test_ocr_document_v1_contract_validation():
    """Verify validate_document_pages rejects invalid coord_space, dpi, bbox, and policy."""
    from ocr.adapter import OCRContractError, validate_document_pages

    valid_page = {
        "page_index": 0,
        "width": 612.0,
        "height": 792.0,
        "coord_space": "pdf_pt",
        "dpi": 72.0,
        "is_scanned": False,
        "ocr_reliable": True,
        "figure_regions": [],
        "blocks": [{
            "block_id": "p0b0000",
            "text": "ok",
            "layout_role": "body",
            "bbox": [10.0, 10.0, 100.0, 25.0],
            "policy": {"translate": True, "preserve_reason": None, "redact_source": True},
        }],
    }
    assert len(validate_document_pages([valid_page])) == 1

    bad_coord = {**valid_page, "coord_space": "inches"}
    with pytest.raises(OCRContractError, match="coord_space"):
        validate_document_pages([bad_coord])

    bad_bbox = {
        **valid_page,
        "blocks": [{**valid_page["blocks"][0], "bbox": [200.0, 10.0, 100.0, 25.0]}],
    }
    with pytest.raises(OCRContractError, match="inverted"):
        validate_document_pages([bad_bbox])

    bad_policy = {
        **valid_page,
        "blocks": [{**valid_page["blocks"][0], "policy": {"translate": "yes"}}],
    }
    with pytest.raises(OCRContractError, match="policy"):
        validate_document_pages([bad_policy])


def test_auto_ocr_mixed_digital_and_scanned_routing_via_config(tmp_path: Path, monkeypatch):
    """Verify auto provider routes digital page -> PyMuPDF and scanned page -> paddle-http via config."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from ocr.adapter import get_provider, validate_document_pages

    class _MockPaddle(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            body = json.dumps({"status": "ok", "paddleocr_version": "3.7.0", "coord_space": "pixel"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or "0")
            _ = self.rfile.read(n)
            body = json.dumps({
                "dpi": 150.0,
                "coord_space": "pixel",
                "bbox_format": "xyxy",
                "blocks": [
                    {"text": "Scanned page paragraph text.", "bbox": [100.0, 200.0, 800.0, 260.0], "score": 0.99},
                ],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), _MockPaddle)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        # Build a 2-page PDF: page 0 digital text, page 1 rasterized scanned image
        vec = _make_pdf(tmp_path / "vec_mixed.pdf")
        mixed = tmp_path / "mixed.pdf"
        with pymupdf.open(vec) as vdoc:
            pix = vdoc[0].get_pixmap(dpi=150)
            mdoc = pymupdf.open()
            mdoc.insert_pdf(vdoc, from_page=0, to_page=0)
            spage = mdoc.new_page(width=vdoc[0].rect.width, height=vdoc[0].rect.height)
            spage.insert_image(spage.rect, stream=pix.tobytes("png"))
            mdoc.save(mixed)
            mdoc.close()

        monkeypatch.delenv("PADDLE_OCR_URL", raising=False)
        cfg = {
            "paddle_http": {"base_url": f"http://127.0.0.1:{port}", "dpi": 150},
            "auto": {"scanned_char_threshold": 20, "fallback_provider": "paddle-http"},
        }
        auto = get_provider("auto", config=cfg)
        pages = validate_document_pages(auto.parse(str(mixed)))
        assert len(pages) == 2
        assert pages[0]["coord_space"] == "pdf_pt" and pages[0]["is_scanned"] is False
        assert pages[1]["coord_space"] == "pixel" and pages[1]["is_scanned"] is True
        assert auto.routed_providers == {"pymupdf": [0], "paddle-http": [1]}

        # fallback_provider="none" disables scanned routing
        auto_none = get_provider("auto", config={"auto": {"fallback_provider": "none"}})
        pages_none = validate_document_pages(auto_none.parse(str(mixed)))
        assert pages_none[1]["is_scanned"] is True
        assert pages_none[1]["ocr_reliable"] is False
        assert auto_none.routed_providers == {"pymupdf": [0, 1]}
    finally:
        srv.shutdown()
        srv.server_close()


def test_typst_bin_env_override(monkeypatch):
    """Verify find_typst respects TYPST_BIN and fails clearly on invalid path."""
    from render.typst_renderer import find_typst

    real_typst = find_typst()
    monkeypatch.setenv("TYPST_BIN", real_typst)
    assert find_typst() == real_typst

    monkeypatch.setenv("TYPST_BIN", "/nonexistent/typst-bin-xyz")
    with pytest.raises(RuntimeError, match="TYPST_BIN"):
        find_typst()


def test_register_provider_is_lazy_for_required_options_endpoint(tmp_path: Path):
    """Registering a provider that requires ocr.options.endpoint in __init__ must not fail at registration time."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from ocr.adapter import (
        OCRConfigError,
        OCRProvider,
        get_provider,
        list_providers,
        register_provider,
        unregister_provider,
        validate_document_pages,
    )
    from ocr.normalize import normalize_pages

    received: list[dict] = []

    class _MockCustomHTTP(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or "0")
            payload = self.rfile.read(n)
            received.append({"path": self.path, "bytes": len(payload)})
            body = json.dumps({
                "blocks": [
                    {"text": "Custom backend paragraph.", "bbox": [80.0, 120.0, 500.0, 160.0]},
                ],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class _StrictCustomHTTPOCRProvider(OCRProvider):
        name = "strict-custom-http"

        def __init__(self, config: dict | None = None) -> None:
            super().__init__(config=config)
            opts = self.config.get("options") or {}
            self.endpoint = str(opts.get("endpoint") or "").rstrip("/")
            self.dpi = float(opts.get("dpi", 150.0))
            if not self.endpoint.startswith(("http://", "https://")):
                raise OCRConfigError("strict-custom-http requires ocr.options.endpoint (http/https URL)")

        def parse(self, source: str) -> list[dict]:
            import urllib.request

            doc = pymupdf.open(source)
            raw_pages: list[dict] = []
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                for page in doc:
                    pix = page.get_pixmap(dpi=int(self.dpi))
                    req = urllib.request.Request(
                        f"{self.endpoint}/ocr?dpi={self.dpi:.1f}&page_index={page.number}",
                        data=pix.tobytes("png"),
                        headers={"Content-Type": "image/png"},
                        method="POST",
                    )
                    with opener.open(req, timeout=15) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    blocks = [
                        {
                            "block_id": f"p{page.number}b{i:04d}",
                            "text": item["text"],
                            "bbox": [float(v) for v in item["bbox"]],
                        }
                        for i, item in enumerate(data.get("blocks", []))
                    ]
                    raw_pages.append({
                        "page_index": page.number,
                        "width": float(pix.width),
                        "height": float(pix.height),
                        "is_scanned": True,
                        "ocr_reliable": True,
                        "blocks": blocks,
                    })
            finally:
                doc.close()
            return normalize_pages(raw_pages, coord_space="pixel", dpi=self.dpi)

    srv = HTTPServer(("127.0.0.1", 0), _MockCustomHTTP)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        # 1. Registration must NOT instantiate with None or raise OCRConfigError
        register_provider(_StrictCustomHTTPOCRProvider.name, _StrictCustomHTTPOCRProvider)
        assert "strict-custom-http" in list_providers()

        # 2. Built-in backends remain completely unaffected
        assert get_provider("stub").name == "stub"
        assert get_provider("pymupdf").name == "pymupdf"

        # 3. Instantiating without required options.endpoint fails clearly
        with pytest.raises(OCRConfigError, match="ocr.options.endpoint"):
            get_provider("strict-custom-http")

        # 4. Instantiating with YAML-style ocr.options.endpoint succeeds and calls local HTTP server
        pdf_path = _make_pdf(tmp_path / "custom_http_in.pdf")
        prov = get_provider(
            "strict-custom-http",
            config={"options": {"endpoint": f"http://127.0.0.1:{port}", "dpi": 120}},
        )
        pages = validate_document_pages(prov.parse(str(pdf_path)))
        assert len(pages) == 1
        assert pages[0]["coord_space"] == "pixel"
        assert pages[0]["dpi"] == 120.0
        assert pages[0]["blocks"][0]["text"] == "Custom backend paragraph."
        assert len(received) == 1
        assert "dpi=120.0" in received[0]["path"]
        assert received[0]["bytes"] > 100
    finally:
        unregister_provider(_StrictCustomHTTPOCRProvider.name)
        srv.shutdown()
        srv.server_close()


def test_builtin_registry_defaults_and_instance_reuse():
    """Preserve baseline 680d079 REGISTRY contents, default instance reuse, lazy paddle-cpu, and isolated configured instances."""
    from ocr.adapter import REGISTRY, get_provider

    expected = ["auto", "mineru", "paddle-cloud", "paddle-cpu", "paddle-http", "pymupdf", "stub"]
    assert sorted(REGISTRY) == expected

    for name in expected:
        assert get_provider(name) is REGISTRY[name]
        assert get_provider(name) is get_provider(name)

    # paddle-cpu remains lazy until first parse()
    assert getattr(REGISTRY["paddle-cpu"], "_initialized", False) is False

    # Passing config constructs a fresh independent instance without mutating the default singleton
    custom_pymupdf = get_provider("pymupdf", config={"scanned_char_threshold": 5})
    assert custom_pymupdf is not REGISTRY["pymupdf"]
    assert custom_pymupdf.scanned_char_threshold == 5
    assert REGISTRY["pymupdf"].scanned_char_threshold == 20


