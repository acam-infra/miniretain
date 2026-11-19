"""Regressions for LAB-2121: multi-paper common defects discovered on Agentero corpus."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.openai_client import _is_verbatim_untranslated_echo  # noqa: E402
from ocr.normalize import normalize_pages  # noqa: E402
from render.typst_renderer import render_typst  # noqa: E402


def test_author_affiliation_code_listing_and_math_fragments_preserved():
    pages = [
        {
            "page_index": 0,
            "width": 612.0,
            "height": 792.0,
            "figure_regions": [],
            "blocks": [
                {
                    "block_id": "p0b0",
                    "text": "Attention Is All You Need",
                    "bbox": [100.0, 60.0, 400.0, 82.0],
                    "font_size_pt": 18.0,
                    "is_bold": True,
                },
                {
                    "block_id": "p0b1",
                    "text": "Niki Parmar∗ Google Research nikip@google.com",
                    "bbox": [100.0, 100.0, 320.0, 114.0],
                    "font_size_pt": 10.0,
                },
                {
                    "block_id": "p0b2",
                    "text": "Yichuan Wang1 Chi Wang2 Yanping Huang2 Zhifeng Chen2",
                    "bbox": [100.0, 120.0, 450.0, 134.0],
                    "font_size_pt": 10.0,
                },
                {
                    "block_id": "p0b3",
                    "text": "5 eval_loader = ZeusDataLoader(eval_set, batch_size)",
                    "bbox": [100.0, 200.0, 400.0, 212.0],
                    "font_size_pt": 9.0,
                },
                {
                    "block_id": "p0b4",
                    "text": "= min",
                    "bbox": [100.0, 230.0, 130.0, 240.0],
                    "font_size_pt": 9.0,
                },
                {
                    "block_id": "p0b5",
                    "text": "i .",
                    "bbox": [100.0, 250.0, 110.0, 260.0],
                    "font_size_pt": 9.0,
                },
            ],
        }
    ]
    norm = normalize_pages(pages, coord_space="pdf_pt", dpi=72.0)
    by_id = {b["block_id"]: b for b in norm[0]["blocks"]}
    assert by_id["p0b1"]["policy"]["translate"] is False
    assert by_id["p0b1"]["policy"]["preserve_reason"] == "author_affiliation"
    assert by_id["p0b2"]["policy"]["translate"] is False
    assert by_id["p0b2"]["policy"]["preserve_reason"] == "author_affiliation"
    assert by_id["p0b3"]["policy"]["translate"] is False
    assert by_id["p0b3"]["policy"]["preserve_reason"] == "code_listing"
    assert by_id["p0b4"]["policy"]["translate"] is False
    assert by_id["p0b4"]["policy"]["preserve_reason"] == "formula"
    assert by_id["p0b5"]["policy"]["translate"] is False
    assert by_id["p0b5"]["policy"]["preserve_reason"] == "formula"


def test_appendix_heading_resets_references_and_chart_labels_preserved():
    pages = [
        {
            "page_index": 0,
            "width": 612.0,
            "height": 792.0,
            "figure_regions": [[323.0, 363.0, 552.0, 396.0]],
            "blocks": [
                {"block_id": "p0b0", "text": "References", "bbox": [55.0, 60.0, 140.0, 75.0], "font_size_pt": 12.0, "is_bold": True},
                {"block_id": "p0b1", "text": "[1] A. Author. Paper title. 2024.", "bbox": [55.0, 80.0, 280.0, 92.0], "font_size_pt": 9.5},
                {"block_id": "p0b2", "text": "A Energy Savings Potential on GPUs", "bbox": [55.0, 120.0, 300.0, 136.0], "font_size_pt": 13.0, "is_bold": True},
                {"block_id": "p0b3", "text": "NeuMF", "bbox": [411.0, 398.5, 423.3, 406.3], "font_size_pt": 3.57},
            ],
        }
    ]
    norm = normalize_pages(pages, coord_space="pdf_pt", dpi=72.0)
    by_id = {b["block_id"]: b for b in norm[0]["blocks"]}
    assert by_id["p0b1"]["layout_role"] == "reference"
    assert by_id["p0b2"]["layout_role"] == "heading"
    assert by_id["p0b3"]["layout_role"] == "figure_label"
    assert by_id["p0b3"]["policy"]["translate"] is False


def test_verbatim_echo_exempts_author_email_code_but_catches_prose():
    prose = {"block_id": "b1", "layout_role": "body", "text": "We evaluate our distributed inference engine across multiple GPU clusters."}
    email_auth = {"block_id": "b2", "layout_role": "body", "text": "Niki Parmar∗ Google Research nikip@google.com"}
    code = {"block_id": "b3", "layout_role": "body", "text": "12 train_loader.report_metric(validation_metric)"}
    assert _is_verbatim_untranslated_echo(prose, prose["text"], "zh") is True
    assert _is_verbatim_untranslated_echo(email_auth, email_auth["text"], "zh") is False
    assert _is_verbatim_untranslated_echo(code, code["text"], "zh") is False


def test_typst_preexisting_overlap_and_exact_span_attribution():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src_pdf = td_path / "src.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=612.0, height=792.0)
        page.insert_text((108.0, 571.0), "[37] First reference line.", fontsize=10.0)
        page.insert_text((129.0, 582.0), "Advances in Neural Information Processing Systems, 2015.", fontsize=10.0)
        doc.save(src_pdf)
        doc.close()

        items = [
            {
                "block_id": "p0b0",
                "page_index": 0,
                "layout_role": "reference",
                "policy": {"translate": True, "redact_source": True},
                "bbox": [108.0, 563.4, 504.0, 574.5],
                "line_bboxes": [[108.0, 563.4, 504.0, 574.5]],
                "font_size_pt": 10.0,
                "text": "[37] First reference line.",
                "translation": "[37] 第一条参考文献内容。",
                "translation_status": "ok",
            },
            {
                "block_id": "p0b1",
                "page_index": 0,
                "layout_role": "reference",
                "policy": {"translate": True, "redact_source": True},
                "bbox": [129.0, 574.2, 365.2, 584.5],
                "line_bboxes": [[129.0, 574.2, 365.2, 584.5]],
                "font_size_pt": 10.0,
                "text": "Advances in Neural Information Processing Systems, 2015.",
                "translation": "《神经信息处理系统进展》（NeurIPS），2015。",
                "translation_status": "ok",
            },
        ]
        res = render_typst(
            items=items,
            workdir=td_path,
            output_cn=td_path / "cn.pdf",
            output_merged=td_path / "merged.pdf",
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
        assert res["placed"] == 2
        assert res["unplaced"] == 0


def test_scheduler_splits_timed_out_large_batch():
    from llm.retry import RetryPolicy
    from llm.scheduler import LLMClient, Scheduler

    class SplitTimeoutClient(LLMClient):
        name = "split-timeout"
        calls = []

        def translate(self, items, target_lang, model, timeout_ms):
            self.calls.append(len(items))
            if len(items) > 4:
                raise TimeoutError("The read operation timed out")
            return [f"[zh] {it['text']}" for it in items]

    items = [
        {"block_id": f"b{i}", "page_index": 0, "layout_role": "body", "text": f"Sentence {i}."}
        for i in range(8)
    ]
    sched = Scheduler(
        client=SplitTimeoutClient(),
        concurrency=2,
        batch_size=8,
        max_retries=1,
        timeout_ms=1000,
        retry_policy=RetryPolicy(max_retries=1),
    )
    recs, stats = sched.run(items, "zh", "m")
    assert stats["succeeded_items"] == 8
    assert stats["failed_items"] == 0
    assert stats["failed_batches"] == 0
    assert all(n <= 4 for n in SplitTimeoutClient.calls if n != 8)

