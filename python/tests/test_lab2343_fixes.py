"""LAB-2343 systemic regression tests for Miniretain."""

from __future__ import annotations

import http.server
import io
import json
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

import pymupdf
import pytest

PY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PY_ROOT.parent
sys.path.insert(0, str(PY_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from llm.formula_protect import protect_formulas, restore_formulas  # noqa: E402
from llm.openai_client import _is_verbatim_untranslated_echo  # noqa: E402
from ocr.base import OCRBackendUnavailableError, prepare_cloud_pdf  # noqa: E402
from ocr.mineru_cloud_adapter import MinerUCloudOCRProvider  # noqa: E402
from ocr.normalize import normalize_pages  # noqa: E402
from render.typst_renderer import _looks_like_currency_span, render_typst, typst_rich_text_expr  # noqa: E402
import worker as worker_mod  # noqa: E402


def _make_digital_pdf(path: Path, pages_text: list[str]) -> None:
    doc = pymupdf.open()
    for txt in pages_text:
        p = doc.new_page(width=612, height=792)
        p.insert_text((60, 120), txt, fontsize=11)
    doc.save(path)
    doc.close()


def test_mineru_prefers_preproc_blocks_and_recovers_para_cross_page_split(tmp_path: Path):
    src = tmp_path / "cross_page.pdf"
    _make_digital_pdf(src, [
        "Title of Paper on Page Zero\nBody ending on page zero column two.",
        "Continuation paragraph top of page one column one should stay on page one.",
    ])

    # Simulated layout.json mimicking MinerU v4: para_blocks merges p1 top para into p0 bottom block
    # and marks p1 block lines_deleted=True, while preproc_blocks preserves clean page separation.
    layout_with_preproc = {
        "pdf_info": [
            {
                "page_size": [612, 792],
                "preproc_blocks": [
                    {
                        "type": "title",
                        "level": 1,
                        "bbox": [60, 60, 520, 95],
                        "lines": [{"bbox": [60, 60, 520, 95], "spans": [{"type": "text", "content": "Paper Title Here"}]}],
                    },
                    {
                        "type": "text",
                        "bbox": [320, 700, 560, 760],
                        "lines": [{"bbox": [320, 700, 560, 760], "spans": [{"type": "text", "content": "Bottom col2 page 0"}]}],
                    },
                ],
                "para_blocks": [
                    {
                        "type": "title",
                        "level": 1,
                        "bbox": [60, 60, 520, 95],
                        "lines": [{"bbox": [60, 60, 520, 95], "spans": [{"type": "text", "content": "Paper Title Here"}]}],
                    },
                    {
                        "type": "text",
                        "bbox": [55, 65, 560, 760],
                        "lines": [
                            {"bbox": [320, 700, 560, 760], "spans": [{"type": "text", "content": "Bottom col2 page 0"}]},
                            {"bbox": [55, 65, 290, 105], "spans": [{"type": "text", "content": "Stolen top-left page 1"}]},
                        ],
                    },
                ],
            },
            {
                "page_size": [612, 792],
                "preproc_blocks": [
                    {
                        "type": "text",
                        "bbox": [55, 65, 290, 105],
                        "lines": [{"bbox": [55, 65, 290, 105], "spans": [{"type": "text", "content": "Clean top-left page 1"}]}],
                    },
                ],
                "para_blocks": [
                    {
                        "type": "text",
                        "bbox": [55, 65, 290, 105],
                        "lines_deleted": True,
                        "lines": [{"bbox": [55, 65, 290, 105], "spans": [{"type": "text", "content": "Deleted wrapper"}]}],
                    },
                ],
            },
        ]
    }

    prov = MinerUCloudOCRProvider(token="dummy-tok")
    with mock.patch.object(prov, "_run_with_slot", return_value=layout_with_preproc):
        pages = prov.parse(str(src))
    p0_texts = [b["text"] for b in pages[0]["blocks"]]
    p1_texts = [b["text"] for b in pages[1]["blocks"]]
    assert not any("Stolen" in t for t in p0_texts)
    assert any("Clean top-left page 1" in t for t in p1_texts)

    # Also test para_blocks-only fallback when preproc_blocks is absent
    layout_para_only = {
        "pdf_info": [
            {k: v for k, v in p.items() if k != "preproc_blocks"}
            for p in layout_with_preproc["pdf_info"]
        ]
    }
    with mock.patch.object(prov, "_run_with_slot", return_value=layout_para_only):
        pages2 = prov.parse(str(src))
    p0_texts2 = [b["text"] for b in pages2[0]["blocks"]]
    p1_texts2 = [b["text"] for b in pages2[1]["blocks"]]
    assert not any("Stolen" in t for t in p0_texts2)
    assert any("Bottom col2 page 0" in t for t in p0_texts2)
    assert any("Stolen top-left page 1" in t for t in p1_texts2)
    assert not any("Deleted wrapper" in t for t in p1_texts2)


def test_normalize_title_overlap_dedup_and_revtex_refs():
    pages = [
        {
            "page_index": 0,
            "width": 612.0,
            "height": 792.0,
            "blocks": [
                {
                    "block_id": "b_title",
                    "text": "Orbital Compute Systems at Scale",
                    "bbox": [50.0, 50.0, 560.0, 90.0],
                    "font_size_pt": 18.0,
                    "is_bold": True,
                },
                {
                    "block_id": "b_ghost",
                    "text": "Continuation text accidentally floating on title zone",
                    "bbox": [55.0, 58.0, 300.0, 86.0],
                    "font_size_pt": 9.5,
                },
                {
                    "block_id": "b_main",
                    "text": "First complete body paragraph on page zero.",
                    "bbox": [50.0, 200.0, 290.0, 300.0],
                    "font_size_pt": 9.5,
                },
                {
                    "block_id": "b_dup",
                    "text": "First complete body paragraph on page zero.",
                    "bbox": [95.0, 215.0, 330.0, 312.0],
                    "font_size_pt": 9.5,
                },
                {
                    "block_id": "b_ref_no_hdr",
                    "text": "[1] NASA Earth Science Division, Technical Report (2024).",
                    "bbox": [50.0, 620.0, 290.0, 645.0],
                    "font_size_pt": 8.5,
                    "layout_role_hint": "reference",
                },
            ],
        }
    ]
    norm = normalize_pages(pages, coord_space="pdf_pt", dpi=72.0)
    by_id = {b["block_id"]: b for b in norm[0]["blocks"]}
    assert by_id["b_title"]["layout_role"] == "title"
    assert by_id["b_title"]["policy"]["translate"] is True
    assert by_id["b_ghost"]["policy"]["translate"] is False
    assert by_id["b_ghost"]["policy"]["preserve_reason"] == "title_overlap_conflict"
    assert by_id["b_main"]["policy"]["translate"] is True
    assert by_id["b_dup"]["policy"]["translate"] is False
    assert by_id["b_dup"]["policy"]["preserve_reason"] == "overlap_conflict_dedup"
    assert by_id["b_ref_no_hdr"]["layout_role"] == "reference"


def test_mineru_wall_clock_timeout_and_non_cyclic_fallback(tmp_path: Path):
    src = tmp_path / "digital.pdf"
    _make_digital_pdf(src, ["Digital page 1 content with sufficient length for pymupdf detection."])

    # Local slow server that hangs on poll so deadline MUST abort promptly
    class _SlowHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)
            body = json.dumps({"code": 0, "data": {"batch_id": "b1", "file_urls": [f"http://127.0.0.1:{self.server.server_port}/upload"]}}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_PUT(self):
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.end_headers()

        def do_GET(self):
            time.sleep(1.5)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base_url = f"http://127.0.0.1:{srv.server_port}"
        # Configure cyclic fallback mineru <-> paddle-cloud with short deadlines; must NOT ping-pong
        cfg = {
            "mineru": {
                "base_url": base_url,
                "token": "tok",
                "timeout_s": 0.4,
                "total_timeout_s": 0.4,
                "poll_interval_s": 0.1,
                "fallback_provider": "paddle-cloud",
            },
            "paddle_cloud": {
                "job_url": f"{base_url}/jobs",
                "token": "tok",
                "timeout_s": 0.4,
                "total_timeout_s": 0.4,
                "poll_interval_s": 0.1,
                "fallback_provider": "mineru",
            },
        }
        prov = MinerUCloudOCRProvider(config=cfg)
        t0 = time.perf_counter()
        pages = prov.parse(str(src))
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.5, f"cyclic fallback hung or took too long: {elapsed:.2f}s"
        assert len(pages) == 1
        assert "Digital page 1" in pages[0]["blocks"][0]["text"]
        assert prov.service_info.get("fallback_triggered") in {"paddle-cloud", "pymupdf"}

        # With fallback_provider='none', it must fail quickly with OCRBackendUnavailableError
        cfg_none = {"mineru": {**cfg["mineru"], "fallback_provider": "none"}}
        prov_none = MinerUCloudOCRProvider(config=cfg_none)
        with pytest.raises(OCRBackendUnavailableError):
            prov_none.parse(str(src))
    finally:
        srv.shutdown()
        srv.server_close()


def test_prepare_cloud_pdf_large_threshold_downgrades_digital_or_compresses(tmp_path: Path):
    src = tmp_path / "big_digital.pdf"
    _make_digital_pdf(src, ["Native digital text paragraph exceeding threshold characters cleanly."])
    # Tiny threshold (0.0001 MB ~ 100 bytes) triggers compression/downgrade path cleanly
    upload_path, cleanup, meta = prepare_cloud_pdf(str(src), max_mb=0.0001)
    assert meta["orig_mb"] >= 0.0
    if cleanup:
        Path(cleanup).unlink(missing_ok=True)
    # Since compressed copy is still > ~100 bytes and has native digital text, too_large_digital triggers
    assert meta["too_large_digital"] is True or meta["compressed"] is True


def test_currency_vs_math_protection_and_typst_rendering():
    samples = [
        "成本为 $200/kg 到 $350/kg，运营支出从 $/kW/y 降至 ~$810/kW/y。",
        "范围在 $810–7,500/kW/y 到 ~$570–3,000/kW/y 之间。",
        "发射单价从 $460/kg (no reuse) to <~$15/kg 显著下降。",
    ]
    for s in samples:
        prot, toks = protect_formulas(s)
        assert not toks, f"currency falsely protected as formula: {toks} from {s}"
        expr = typst_rich_text_expr(s, use_mitex=True)
        assert "#mi(" not in expr, f"currency falsely compiled as #mi(): {expr}"

    mixed = "价格 $200/kg 到 $350/kg，同时能量公式 $E = mc^2$ 保持完整。"
    prot_m, toks_m = protect_formulas(mixed)
    assert list(toks_m.values()) == ["$E = mc^2$"]
    restored_m, missing = restore_formulas(prot_m, toks_m)
    assert not missing and restored_m == mixed
    expr_m = typst_rich_text_expr(mixed, use_mitex=True)
    assert "#mi(`E = mc^2`)" in expr_m
    assert expr_m.count("#mi(") == 1
    assert _looks_like_currency_span("200/kg to ", after_char="3") is True


def test_typst_narrow_currency_horizontal_overflow_recovery(tmp_path: Path):
    src = tmp_path / "narrow.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((312, 120), "CapEx $/kW/y ~$810/kW/y $460/kg (no reuse) to <~$15/kg", fontsize=8)
    doc.save(src)
    doc.close()

    items = [{
        "block_id": "p5b0003",
        "page_index": 0,
        "bbox": [310.0, 108.0, 392.0, 140.0],  # Narrow 82pt box with currency & hyphenated tokens
        "line_bboxes": [[310.0, 108.0, 392.0, 140.0]],
        "layout_role": "body",
        "font_size_pt": 8.0,
        "text": "CapEx $/kW/y ~$810/kW/y $460/kg (no reuse) to <~$15/kg",
        "translation": "资本支出 $/kW/y 降至 ~$810/kW/y，单价 $460/kg（无复用）降至 <~$15/kg。",
        "translation_status": "ok",
        "policy": {"translate": True, "redact_source": True},
    }]
    out_cn = tmp_path / "out_cn.pdf"
    out_sbs = tmp_path / "out_sbs.pdf"
    res = render_typst(
        items=items,
        workdir=tmp_path,
        output_cn=out_cn,
        output_merged=out_sbs,
        source=str(src),
        page_widths={0: 612.0},
        page_coord_spaces={0: "pdf_pt"},
        strict=True,
    )
    assert res["unplaced"] == 0
    assert out_cn.exists()


def test_verbatim_echo_exemptions_and_cli_no_strict_allow_partial(tmp_path: Path):
    # Address / affiliation / DOI / reference / acronym blocks should NOT trigger verbatim echo
    addr = {"block_id": "b1", "layout_role": "body", "text": "Department of Physics and Astronomy, Harvard University, Cambridge, MA 02138, USA"}
    ref_body = {"block_id": "b2", "layout_role": "body", "text": "[12] A. B. Author et al., Phys. Rev. Lett. 131 (2023), doi:10.1103/PhysRevLett.131.010101"}
    acronym = {"block_id": "b3", "layout_role": "body", "text": "NASA-GSFC / JWST-NIRCam / HST-WFC3 / ALMA-Band6 / ESO-VLT"}
    assert _is_verbatim_untranslated_echo(addr, addr["text"], "zh") is False
    assert _is_verbatim_untranslated_echo(ref_body, ref_body["text"], "zh") is False
    assert _is_verbatim_untranslated_echo(acronym, acronym["text"], "zh") is False

    # Genuine prose body still triggers verbatim echo
    prose = {"block_id": "b4", "layout_role": "body", "text": "Our measurements demonstrate strong atmospheric escape across young temperate planets."}
    assert _is_verbatim_untranslated_echo(prose, prose["text"], "zh") is True

    # Verify CLI --no-strict forwards allow_partial=True to job payload
    import miniretain_cli
    captured: dict = {}

    def fake_api_json(opener, method, url, payload=None, timeout=30):
        if method == "GET" and url.endswith("/health"):
            return 200, {"ok": True}
        if method == "POST" and url.endswith("/v1/jobs"):
            captured.update(payload or {})
            return 202, {"id": "job-test", "status": "queued"}
        if method == "GET" and url.endswith("/v1/jobs/job-test"):
            return 200, {"id": "job-test", "status": "succeeded", "stage": "render"}
        if method == "GET" and url.endswith("/v1/jobs/job-test/result"):
            cn = tmp_path / "res_cn.pdf"
            bi = tmp_path / "res_bi.pdf"
            cn.write_bytes(b"%PDF-1.4\n")
            bi.write_bytes(b"%PDF-1.4\n")
            return 200, {"result": {"ocr": {}, "llm": {}, "render": {"output_cn": str(cn), "output_merged": str(bi)}}}
        return 200, {}

    pdf = tmp_path / "dummy.pdf"
    _make_digital_pdf(pdf, ["Hello digital paper content."])
    with mock.patch.object(miniretain_cli, "api_json", side_effect=fake_api_json), mock.patch.object(
        sys, "argv", ["miniretain-cli", str(pdf), "--no-strict", "--out-dir", str(tmp_path / "out")]
    ):
        rc = miniretain_cli.main()
    assert rc == 0
    assert captured["llm"]["allow_partial"] is True
    assert captured["render"]["strict"] is False
