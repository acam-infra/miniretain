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
