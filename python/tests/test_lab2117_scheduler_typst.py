"""LAB-2117 regression tests for modular LLM scheduler, global concurrency limiter,
429/retry policy, streaming item retry, document context, and Typst font sizing.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.context import build_document_context  # noqa: E402
from llm.limiter import ConcurrencyLimiter, FileSlotConcurrencyLimiter  # noqa: E402
from llm.openai_client import _is_verbatim_untranslated_echo  # noqa: E402
from llm.planner import CharBudgetLPTPlanner  # noqa: E402
from llm.retry import RetryPolicy, classify_error  # noqa: E402
from llm.scheduler import LLMClient, Scheduler  # noqa: E402
from ocr.pymupdf_adapter import PyMuPDFOCRProvider  # noqa: E402
from render.typst_renderer import _candidate_sizes, _compute_start_size, render_typst  # noqa: E402


class _TrackingFakeClient(LLMClient):
    name = "tracking-fake"

    def __init__(
        self,
        *,
        delay_s: float = 0.04,
        fail_429_count: int = 0,
        fail_fatal: bool = False,
        drop_first_item_once: bool = False,
        slow_block_id: str | None = None,
        slow_delay_s: float = 0.35,
    ) -> None:
        self.delay_s = delay_s
        self.fail_429_left = fail_429_count
        self.fail_fatal = fail_fatal
        self.drop_first_item_once = drop_first_item_once
        self.slow_block_id = slow_block_id
        self.slow_delay_s = slow_delay_s
        self._lock = threading.Lock()
        self.inflight = 0
        self.peak_inflight = 0
        self.total_calls = 0
        self._dropped_once = False

    def translate(self, items, target_lang, model, timeout_ms):
        with self._lock:
            self.total_calls += 1
            self.inflight += 1
            if self.inflight > self.peak_inflight:
                self.peak_inflight = self.inflight
            do_429 = False
            if self.fail_429_left > 0:
                self.fail_429_left -= 1
                do_429 = True
            do_drop = False
            if self.drop_first_item_once and len(items) > 1 and not self._dropped_once:
                if not any(it["block_id"] == self.slow_block_id for it in items):
                    self._dropped_once = True
                    do_drop = True
        try:
            if self.fail_fatal:
                raise RuntimeError("LLM HTTP 401: Unauthorized")
            if do_429:
                time.sleep(0.01)
                raise RuntimeError("LLM HTTP 429 retry_after=0.05: rate limit")
            if self.slow_block_id and any(it["block_id"] == self.slow_block_id for it in items):
                time.sleep(self.slow_delay_s)
            else:
                time.sleep(self.delay_s)
            out = []
            for idx, it in enumerate(items):
                if do_drop and idx == 0:
                    out.append({"block_id": it["block_id"], "status": "missing", "translation": None, "error": "omitted"})
                else:
                    out.append({"block_id": it["block_id"], "status": "ok", "translation": f"[{target_lang}] {it['text']}", "error": None})
            return out
        finally:
            with self._lock:
                self.inflight -= 1


def test_single_and_multi_job_peak_inflight_with_file_slot_limiter():
    items = [{"block_id": f"b{i}", "text": f"Academic block {i} " * 20} for i in range(24)]
    gw = _TrackingFakeClient(delay_s=0.06)

    with tempfile.TemporaryDirectory() as td:
        results_all = []

        def _run_one():
            limiter = FileSlotConcurrencyLimiter(local_limit=6, global_limit=4, slot_dir=td, poll_interval_s=0.005)
            planner = CharBudgetLPTPlanner(batch_size=4, max_chars=900, lpt=True)
            sc = Scheduler(gw, concurrency=6, batch_size=4, max_retries=1, timeout_ms=10000, planner=planner, limiter=limiter)
            res, st = sc.run(items, "zh", "fake-model")
            results_all.append((res, st))

        threads = [threading.Thread(target=_run_one) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # Global cap=4 strictly respected even with 2 concurrent jobs x local concurrency=6
    assert gw.peak_inflight <= 4, f"expected global peak_inflight <= 4, got {gw.peak_inflight}"
    assert len(results_all) == 2
    for res, st in results_all:
        assert st["succeeded_items"] == 24
        assert st["failed_items"] == 0
        assert st["global_concurrency_limit"] == 4
        assert [r["block_id"] for r in res] == [it["block_id"] for it in items]


def test_429_retry_after_and_metrics():
    items = [{"block_id": f"b{i}", "text": f"Sentence {i}"} for i in range(6)]
    gw = _TrackingFakeClient(delay_s=0.01, fail_429_count=2)
    policy = RetryPolicy(max_retries=3, rate_limit_base_s=0.02, rate_limit_cap_s=0.2, jitter=True)
    sc = Scheduler(gw, concurrency=2, batch_size=3, max_retries=3, timeout_ms=10000, retry_policy=policy)
    res, st = sc.run(items, "zh", "fake")
    assert st["succeeded_items"] == 6
    assert st["rate_limit_429"] == 2
    assert st["retries"] == 2
    assert st["requests"] == 4  # 2 initial batches + 2 retries
    assert [r["block_id"] for r in res] == [it["block_id"] for it in items]


def test_streaming_item_retry_overlaps_slow_batch():
    # Batch 1 has a slow block (0.35s); Batch 2 finishes in 0.02s with 1 missing item (0.02s retry).
    # With streaming retry, retry completes while Batch 1 is still running -> retry_s ~ 0.0s!
    items = [
        {"block_id": "slow_0", "text": "A" * 200},
        {"block_id": "slow_1", "text": "B" * 200},
        {"block_id": "fast_0", "text": "C" * 50},
        {"block_id": "fast_1", "text": "D" * 50},
    ]
    gw = _TrackingFakeClient(
        delay_s=0.02,
        drop_first_item_once=True,
        slow_block_id="slow_0",
        slow_delay_s=0.35,
    )
    planner = CharBudgetLPTPlanner(batch_size=2, max_chars=5000, lpt=True)
    sc = Scheduler(gw, concurrency=3, batch_size=2, max_retries=2, timeout_ms=10000, planner=planner)
    res, st = sc.run(items, "zh", "fake")
    assert st["succeeded_items"] == 4
    assert st["retries"] >= 1
    assert st["retry_s"] < 0.15, f"expected overlapped streaming retry_s < 0.15s, got {st['retry_s']}"
    assert st["elapsed_s"] < 0.50, f"expected total elapsed < 0.50s, got {st['elapsed_s']}"
    assert [r["block_id"] for r in res] == ["slow_0", "slow_1", "fast_0", "fast_1"]


def test_fatal_401_does_not_waste_retries():
    cat, _ = classify_error(RuntimeError("LLM HTTP 401: Unauthorized"))
    assert cat == "fatal"
    items = [{"block_id": "b0", "text": "Hello"}]
    gw = _TrackingFakeClient(fail_fatal=True)
    sc = Scheduler(gw, concurrency=1, batch_size=2, max_retries=3, timeout_ms=5000)
    res, st = sc.run(items, "zh", "fake")
    assert st["requests"] == 1
    assert st["retries"] == 0
    assert st["failed_items"] == 1
    assert res[0]["status"] == "failed"


def test_document_context_extraction():
    items = [
        {"block_id": "p0", "layout_role": "title", "text": "Disaggregated LLM Inference in Space"},
        {"block_id": "p1", "layout_role": "body", "text": "We study Prefill and Decode with KV Cache over ISL links in LEO."},
        {"block_id": "p2", "layout_role": "body", "text": "MoE LLM serving transfers KV Cache across ISL links."},
    ]
    ctx = build_document_context(items)
    assert "Disaggregated LLM Inference in Space" in ctx
    assert "KV Cache" in ctx
    assert "ISL" in ctx
    assert "图 N" in ctx


def test_verbatim_untranslated_echo_guard():
    prose = {
        "block_id": "p0b5",
        "layout_role": "body",
        "text": "Orbital data centers could enable large-scale AI inference across multiple satellites.",
    }
    assert _is_verbatim_untranslated_echo(prose, prose["text"], "zh") is True
    assert _is_verbatim_untranslated_echo(prose, "轨道数据中心有望实现大规模 AI 推理。", "zh") is False
    # Short labels / references are not flagged
    ref = {"block_id": "p6b1", "layout_role": "reference", "text": prose["text"]}
    assert _is_verbatim_untranslated_echo(ref, ref["text"], "zh") is False
    short = {"block_id": "p3b2", "layout_role": "body", "text": "1000"}
    assert _is_verbatim_untranslated_echo(short, "1000", "zh") is False


def test_typst_line_aware_start_and_fine_sizes():
    sizes = _candidate_sizes(9.75, 5.0)
    assert sizes[:4] == [9.75, 9.5, 9.25, 9.0]
    assert sizes[-1] == 5.0
    # Multi-line box gets 0.98x hint; single-line box gets capped by height*0.85
    multi = _compute_start_size(10.0, pymupdf.Rect(0, 0, 200, 60), 5.0, role="body")
    single = _compute_start_size(10.96, pymupdf.Rect(0, 0, 480, 11.15), 5.0, role="body")
    assert multi == 9.8
    assert single < 9.6

    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        src = wd / "src.pdf"
        doc = pymupdf.open()
        pg = doc.new_page(width=612, height=792)
        pg.insert_text((72, 100), "Original English paragraph text.", fontsize=10)
        doc.save(src)
        doc.close()

        items = [
            {
                "block_id": "p0b1",
                "page_index": 0,
                "layout_role": "body",
                "policy": {"translate": True, "preserve_reason": None, "redact_source": True},
                "bbox": [72, 85, 320, 145],
                "font_size_pt": 10.0,
                "text": "Original English paragraph text.",
                "translation": "这是一段用于验证 Typst 精细字号选择与统计上报的中文测试段落。",
                "translation_status": "ok",
            }
        ]
        st = render_typst(
            items=items,
            workdir=wd,
            output_cn=wd / "cn.pdf",
            output_merged=wd / "sbs.pdf",
            source=str(src),
            layout_mode="preserve",
            bilingual_mode="side_by_side",
            page_widths={0: 612},
            page_coord_spaces={0: "pdf_pt"},
            page_scanned={0: False},
            strict=True,
        )
        assert st["placed"] == 1
        assert "font_size_stats" in st
        assert st["font_size_stats"]["min_pt"] >= 9.5
        assert st["font_size_stats"]["min_size_floor_pt"] == 5.0


def test_retries_accounting_exhausted_429_and_item_retry():
    """Verify retries counts every additional call (batch retry + single-item retry),
    covering exhausted 429s, eventual success, fatal 401 fast-fail, and item retry with 429."""
    policy = RetryPolicy(max_retries=2, rate_limit_base_s=0.01, rate_limit_cap_s=0.05, jitter=False)

    # 1. Always-429 client, max_retries=2 -> requests=3, rate_limit_429=3, retries=2, failed_items=1
    gw_exhaust = _TrackingFakeClient(delay_s=0.0, fail_429_count=99)
    sc1 = Scheduler(gw_exhaust, concurrency=1, batch_size=4, max_retries=2, timeout_ms=5000, retry_policy=policy)
    res1, st1 = sc1.run([{"block_id": "b0", "text": "Hello"}], "zh", "fake")
    assert st1["requests"] == 3
    assert st1["rate_limit_429"] == 3
    assert st1["retries"] == 2
    assert st1["failed_items"] == 1
    assert st1["requests"] == st1["batches"] + st1["retries"]
    assert res1[0]["status"] == "failed"

    # 2. 429 twice then succeed -> requests=3, rate_limit_429=2, retries=2, succeeded=2
    gw_recover = _TrackingFakeClient(delay_s=0.0, fail_429_count=2)
    sc2 = Scheduler(gw_recover, concurrency=1, batch_size=4, max_retries=2, timeout_ms=5000, retry_policy=policy)
    res2, st2 = sc2.run([{"block_id": "b0", "text": "A"}, {"block_id": "b1", "text": "B"}], "zh", "fake")
    assert st2["requests"] == 3
    assert st2["rate_limit_429"] == 2
    assert st2["retries"] == 2
    assert st2["succeeded_items"] == 2
    assert st2["requests"] == st2["batches"] + st2["retries"]

    # 3. Fatal 401 -> requests=1, retries=0, failed_items=1
    gw_fatal = _TrackingFakeClient(fail_fatal=True)
    sc3 = Scheduler(gw_fatal, concurrency=1, batch_size=4, max_retries=2, timeout_ms=5000, retry_policy=policy)
    _, st3 = sc3.run([{"block_id": "b0", "text": "A"}], "zh", "fake")
    assert st3["requests"] == 1
    assert st3["retries"] == 0
    assert st3["failed_items"] == 1
    assert st3["requests"] == st3["batches"] + st3["retries"]

    # 4. Batch returns 1 missing item; targeted single-item retry hits 429 once then succeeds
    class _DropThen429Client(LLMClient):
        name = "drop-then-429"

        def __init__(self):
            self.calls = 0

        def translate(self, items, target_lang, model, timeout_ms):
            self.calls += 1
            if self.calls == 1:
                return [
                    {"block_id": items[0]["block_id"], "status": "missing", "translation": None, "error": "omitted"},
                    {"block_id": items[1]["block_id"], "status": "ok", "translation": "[zh] B", "error": None},
                ]
            if self.calls == 2:
                raise RuntimeError("LLM HTTP 429: rate limit")
            return [{"block_id": it["block_id"], "status": "ok", "translation": f"[zh] {it['text']}", "error": None} for it in items]

    sc4 = Scheduler(_DropThen429Client(), concurrency=2, batch_size=4, max_retries=2, timeout_ms=5000, retry_policy=policy)
    res4, st4 = sc4.run([{"block_id": "b0", "text": "A"}, {"block_id": "b1", "text": "B"}], "zh", "fake")
    assert st4["batches"] == 1
    assert st4["requests"] == 3  # 1 initial batch + 1 single-item retry + 1 429 retry
    assert st4["rate_limit_429"] == 1
    assert st4["retries"] == 2
    assert st4["succeeded_items"] == 2
    assert st4["requests"] == st4["batches"] + st4["retries"]
    assert [r["block_id"] for r in res4] == ["b0", "b1"]


def test_numeric_and_log_scale_axis_ticks_preserved_with_superscript_geometry():
    """Log-scale exponent ticks (10^3, 10^-1) and plain numeric ticks (5, 35, 1000)
    must be excluded from LLM translation & Typst redraw so superscript geometry is untouched,
    while the figure caption is translated and placed."""
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        src = wd / "fig3_ticks.pdf"
        doc = pymupdf.open()
        pg = doc.new_page(width=612, height=792)

        # Draw a plot frame (right of y-axis ticks so ticks sit outside figure_regions)
        pg.draw_rect(pymupdf.Rect(140, 80, 320, 200), color=(0, 0, 0), width=1.0)
        # Log-scale y-axis tick 10^3 with '3' raised by 3pt above '10'
        pg.insert_text((95, 95), "10", fontsize=7.0)
        pg.insert_text((103.5, 92), "3", fontsize=4.8)
        # Log-scale y-axis tick 10^-1 with '-1' raised by 3pt above '10'
        pg.insert_text((95, 130), "10", fontsize=7.0)
        pg.insert_text((103.5, 127), "-1", fontsize=4.8)
        # Plain numeric ticks
        pg.insert_text((98, 165), "35", fontsize=7.0)
        pg.insert_text((90, 195), "1000", fontsize=7.0)
        # Translatable figure caption below plot
        pg.insert_text((80, 235), "Fig. 3: Inter-satellite distance vs capacity for AFD budget.", fontsize=9.0)
        doc.save(src)
        doc.close()

        pages = PyMuPDFOCRProvider().parse(str(src))
        blocks = pages[0]["blocks"]
        caption_blocks = [b for b in blocks if b["layout_role"] == "caption"]
        tick_blocks = [b for b in blocks if (b.get("policy") or {}).get("preserve_reason") == "numeric_axis_tick"]
        assert len(caption_blocks) == 1, f"expected 1 caption block, got {caption_blocks}"
        assert caption_blocks[0]["policy"]["translate"] is True
        assert len(tick_blocks) >= 4, f"expected >=4 preserved numeric/log ticks, got {blocks}"
        for tb in tick_blocks:
            assert tb["policy"]["translate"] is False
            assert tb["policy"]["redact_source"] is False

        # Build payload (only translatable blocks translated; preserved blocks keep source)
        payload = []
        for b in blocks:
            tr_flag = (b.get("policy") or {}).get("translate", True)
            payload.append({
                "block_id": b["block_id"],
                "page_index": 0,
                "layout_role": b["layout_role"],
                "policy": b["policy"],
                "bbox": b["bbox"],
                "line_bboxes": b.get("line_bboxes"),
                "font_size_pt": (b.get("content") or {}).get("font_size_pt"),
                "text": b["text"],
                "translation": "图 3：满足 AFD 预算的星间距离与容量关系。" if tr_flag else b["text"],
                "translation_status": "ok" if tr_flag else "preserved",
            })

        cn_pdf = wd / "cn.pdf"
        sbs_pdf = wd / "sbs.pdf"
        st = render_typst(
            items=payload,
            workdir=wd,
            output_cn=cn_pdf,
            output_merged=sbs_pdf,
            source=str(src),
            layout_mode="preserve",
            bilingual_mode="side_by_side",
            page_widths={0: 612},
            page_coord_spaces={0: "pdf_pt"},
            page_scanned={0: False},
            strict=True,
        )
        assert st["placed"] == 1
        assert st["preserved"] >= 4
        assert st["unplaced"] == 0

        # Verify superscript '3' in 10^3 retains elevated y-position AND 0 pixel diff in tick clip!
        with pymupdf.open(src) as d_src, pymupdf.open(cn_pdf) as d_cn:
            tick_clip = pymupdf.Rect(85, 80, 125, 205)
            pix_src = d_src[0].get_pixmap(clip=tick_clip, dpi=150).samples
            pix_cn = d_cn[0].get_pixmap(clip=tick_clip, dpi=150).samples
            assert pix_src == pix_cn, "numeric/log-scale y-axis tick region pixels were altered!"

            spans = [
                sp
                for blk in d_cn[0].get_text("dict").get("blocks", [])
                for ln in blk.get("lines", [])
                for sp in ln.get("spans", [])
            ]
            s10 = [sp for sp in spans if sp["text"].strip() == "10" and sp["bbox"][1] < 105]
            s3 = [sp for sp in spans if sp["text"].strip() == "3" and sp["bbox"][1] < 105]
            assert s10 and s3, f"missing 10 / 3 spans in CN PDF: {[(s['text'], s['bbox']) for s in spans]}"
            # Superscript '3' baseline/top must stay clearly elevated above '10'
            assert s10[0]["origin"][1] - s3[0]["origin"][1] >= 2.0, (
                f"superscript 3 lost vertical elevation: 10={s10[0]['origin']}, 3={s3[0]['origin']}"
            )
            assert "图 3" in d_cn[0].get_text("text")

