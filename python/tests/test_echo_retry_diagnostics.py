"""Mock-gateway & unit regressions for body echo detection, bounded single-block retry,
non-critical preservation, and structured failure reporting (LAB-2121)."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import pytest

from llm.openai_client import _is_verbatim_untranslated_echo
from ocr.normalize import normalize_pages
import worker as worker_mod


P0B0011_TEXT = (
    "In this paper, we present Zeus to address this question. Zeus is a plug-in "
    "optimization framework that automatically configures the batch size and GPU "
    "power limit to minimize the overall energy consumption and training time for "
    "DNN training jobs (§3). Unlike some recent works that only consider GPU-side "
    "profiling, Zeus optimizes end-to-end training energy efficiency."
)


class _MockGateway:
    """Thread-safe local OpenAI-compatible HTTP mock gateway for end-to-end worker.run_llm tests."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                prompt = body["messages"][0]["content"]
                block_ids: list[str] = []
                block_texts: dict[str, str] = {}
                for line in prompt.splitlines():
                    if "\t" in line:
                        bid, txt = line.split("\t", 1)
                        if bid.startswith("p"):
                            block_ids.append(bid)
                            block_texts[bid] = txt
                with parent._lock:
                    call_num = len(parent.calls) + 1
                    parent.calls.append(list(block_ids))
                mapping = parent.responder(call_num, block_ids, block_texts)
                resp_payload = {
                    "choices": [{"message": {"content": json.dumps(mapping, ensure_ascii=False)}}]
                }
                raw = json.dumps(resp_payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format, *args):  # noqa: A002
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _write_doc(wd: Path, blocks: list[dict]) -> None:
    doc = {
        "version": "document.v1",
        "ocr_provider": "pymupdf",
        "ocr_reliable": True,
        "limitations": [],
        "pages": [
            {
                "page_index": 0,
                "width": 612.0,
                "height": 792.0,
                "coord_space": "pdf_pt",
                "dpi": 72.0,
                "is_scanned": False,
                "ocr_reliable": True,
                "figure_regions": [],
                "blocks": blocks,
            }
        ],
    }
    (wd / "document.v1.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def _make_spec(wd: Path, *, max_retries: int, allow_partial: bool = False) -> dict:
    return {
        "job_id": "test-job",
        "stage": "llm",
        "workdir": str(wd),
        "source": "dummy.pdf",
        "target_lang": "zh",
        "ocr": {"provider": "pymupdf"},
        "llm": {
            "provider": "openai-compat",
            "concurrency": 2,
            "batch_size": 8,
            "max_retries": max_retries,
            "timeout_ms": 10000,
            "model": "mock-model",
            "allow_partial": allow_partial,
        },
    }


def test_p1_prose_with_section_url_placeholder_still_detects_verbatim_echo():
    # Body prose with §3 (2208.06102 p0b0011) MUST be caught on verbatim echo
    assert _is_verbatim_untranslated_echo(
        {"block_id": "p0b0011", "layout_role": "body", "text": P0B0011_TEXT},
        P0B0011_TEXT,
        "zh",
    ) is True

    # Body prose with inline URL and ⟪F0⟫ placeholder MUST still be caught on verbatim echo
    prose_with_url_and_formula = (
        "We release our implementation at https://github.com/example/zeus and evaluate "
        "the objective ⟪F0⟫ across multiple GPU clusters in production workloads."
    )
    assert _is_verbatim_untranslated_echo(
        {"block_id": "p0b0012", "layout_role": "body", "text": prose_with_url_and_formula},
        prose_with_url_and_formula,
        "zh",
    ) is True

    # Pure email/URL/placeholder block (no substantial prose left after stripping) stays exempt
    pure_meta = "contact@example.edu https://arxiv.org/abs/2208.06102 ⟪F0⟫ §3"
    assert _is_verbatim_untranslated_echo(
        {"block_id": "p0b0099", "layout_role": "body", "text": pure_meta},
        pure_meta,
        "zh",
    ) is False

    # Preserved roles remain exempt
    assert _is_verbatim_untranslated_echo(
        {"block_id": "p0b0001", "layout_role": "header_footer", "text": P0B0011_TEXT},
        P0B0011_TEXT,
        "zh",
    ) is False


def test_p2_title_with_research_keyword_prioritized_over_author_affiliation():
    pages = [
        {
            "page_index": 0,
            "width": 612.0,
            "height": 792.0,
            "figure_regions": [],
            "blocks": [
                {
                    "block_id": "p0b0000",
                    "text": "A Study of Research Infrastructure at Scale",
                    "bbox": [72.0, 72.0, 540.0, 96.0],
                    "font_size_pt": 20.0,
                    "is_bold": True,
                },
                {
                    "block_id": "p0b0001",
                    "text": "Jie You∗ Jae-Won Chung∗ Mosharaf Chowdhury University of Michigan",
                    "bbox": [167.2, 113.9, 447.8, 142.0],
                    "font_size_pt": 10.96,
                },
                {
                    "block_id": "p0b0002",
                    "text": "Abstract",
                    "bbox": [151.8, 180.1, 196.3, 192.1],
                    "font_size_pt": 11.96,
                    "is_bold": True,
                },
                {
                    "block_id": "p0b0003",
                    "text": "Training deep neural networks consumes substantial energy in modern clusters.",
                    "bbox": [54.0, 200.0, 290.0, 240.0],
                    "font_size_pt": 10.0,
                },
            ],
        }
    ]
    norm = normalize_pages(pages, coord_space="pdf_pt", dpi=72.0)
    by_id = {b["block_id"]: b for b in norm[0]["blocks"]}
    assert by_id["p0b0000"]["layout_role"] == "title"
    assert by_id["p0b0000"]["policy"]["translate"] is True
    assert by_id["p0b0001"]["layout_role"] == "header_footer"
    assert by_id["p0b0001"]["policy"]["translate"] is False
    assert by_id["p0b0001"]["policy"]["preserve_reason"] == "author_affiliation"


def test_mock_gateway_echo_recovers_on_single_block_retry_preserving_siblings(tmp_path, monkeypatch):
    sibling_text = "Deep neural networks have received ubiquitous adoption across many application domains."
    _write_doc(
        tmp_path,
        [
            {"block_id": "p0b0010", "reading_order": 0, "text": sibling_text, "layout_role": "body", "policy": {"translate": True, "preserve_reason": None, "redact_source": True}},
            {"block_id": "p0b0011", "reading_order": 1, "text": P0B0011_TEXT, "layout_role": "body", "policy": {"translate": True, "preserve_reason": None, "redact_source": True}},
        ],
    )

    def responder(call_num, block_ids, block_texts):
        out = {}
        for bid in block_ids:
            if bid == "p0b0010":
                out[bid] = "深度神经网络在许多应用领域得到了广泛采用。"
            elif bid == "p0b0011":
                if call_num == 1:
                    out[bid] = block_texts[bid]  # verbatim echo on first attempt
                else:
                    out[bid] = "在本文中，我们提出 Zeus 来解决这一问题（§3）。"
        return out

    with _MockGateway(responder) as gw:
        monkeypatch.setenv("OPENAI_BASE_URL", gw.url)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        res = worker_mod.run_llm(_make_spec(tmp_path, max_retries=2), tmp_path)

    # Call 1 was the 2-block batch; Call 2 retried ONLY p0b0011 (sibling p0b0010 reused).
    assert len(gw.calls) == 2
    assert set(gw.calls[0]) == {"p0b0010", "p0b0011"}
    assert gw.calls[1] == ["p0b0011"]
    assert res["requests"] == 2
    assert res["retries"] == 1
    assert res["translated"] == 2
    assert res["critical_failed"] == 0
    assert res["failed_stage"] is None
    payload = json.loads((tmp_path / "translation.payload.json").read_text(encoding="utf-8"))
    by_id = {it["block_id"]: it for it in payload["items"]}
    assert by_id["p0b0010"]["attempts"] == 1
    assert by_id["p0b0011"]["attempts"] == 2
    assert "§3" in by_id["p0b0011"]["translation"]


def test_mock_gateway_repeated_echo_exhausts_budget_complete_and_partial_modes(tmp_path, monkeypatch):
    def always_echo(_call_num, block_ids, block_texts):
        return {bid: block_texts[bid] for bid in block_ids}

    # 1) Complete mode (allow_partial=False) with max_retries=2 -> 3 attempts then fail with diagnostics
    wd_complete = tmp_path / "complete"
    wd_complete.mkdir()
    _write_doc(
        wd_complete,
        [{"block_id": "p0b0011", "reading_order": 0, "text": P0B0011_TEXT, "layout_role": "body", "policy": {"translate": True, "preserve_reason": None, "redact_source": True}}],
    )
    with _MockGateway(always_echo) as gw:
        monkeypatch.setenv("OPENAI_BASE_URL", gw.url)
        with pytest.raises(RuntimeError) as exc_info:
            worker_mod.run_llm(_make_spec(wd_complete, max_retries=2, allow_partial=False), wd_complete)

    msg = str(exc_info.value)
    assert "llm stage failed" in msg
    assert "p1/p0b0011" in msg
    assert "untranslated verbatim echo" in msg
    assert "attempts=3" in msg
    assert len(gw.calls) == 3

    diag = json.loads((wd_complete / "llm.result.json").read_text(encoding="utf-8"))
    assert diag["failed_stage"] == "llm"
    assert diag["critical_failed"] == 1
    assert diag["requests"] == 3
    assert diag["retries"] == 2
    assert diag["critical_failures"][0]["block_id"] == "p0b0011"
    assert diag["critical_failures"][0]["page_number"] == 1
    assert diag["critical_failures"][0]["attempts"] == 3

    # 2) Explicit partial mode (allow_partial=True) -> completes and marks missing body block
    wd_partial = tmp_path / "partial"
    wd_partial.mkdir()
    _write_doc(
        wd_partial,
        [{"block_id": "p0b0011", "reading_order": 0, "text": P0B0011_TEXT, "layout_role": "body", "policy": {"translate": True, "preserve_reason": None, "redact_source": True}}],
    )
    with _MockGateway(always_echo) as gw2:
        monkeypatch.setenv("OPENAI_BASE_URL", gw2.url)
        res_partial = worker_mod.run_llm(_make_spec(wd_partial, max_retries=1, allow_partial=True), wd_partial)

    assert len(gw2.calls) == 2
    assert res_partial["partial_mode"] is True
    assert res_partial["partial_missing_body"] is True
    assert res_partial["critical_failed"] == 1
    assert res_partial["critical_failures"][0]["attempts"] == 2
    payload = json.loads((wd_partial / "translation.payload.json").read_text(encoding="utf-8"))
    assert payload["items"][0]["policy"]["redact_source"] is False


def test_mock_gateway_zero_retries_fails_immediately(tmp_path, monkeypatch):
    _write_doc(
        tmp_path,
        [{"block_id": "p0b0011", "reading_order": 0, "text": P0B0011_TEXT, "layout_role": "body", "policy": {"translate": True, "preserve_reason": None, "redact_source": True}}],
    )
    with _MockGateway(lambda _n, ids, txts: {b: txts[b] for b in ids}) as gw:
        monkeypatch.setenv("OPENAI_BASE_URL", gw.url)
        with pytest.raises(RuntimeError) as exc_info:
            worker_mod.run_llm(_make_spec(tmp_path, max_retries=0, allow_partial=False), tmp_path)

    assert len(gw.calls) == 1
    assert "attempts=1" in str(exc_info.value)
    diag = json.loads((tmp_path / "llm.result.json").read_text(encoding="utf-8"))
    assert diag["requests"] == 1
    assert diag["retries"] == 0
    assert diag["critical_failures"][0]["attempts"] == 1


def test_non_critical_title_fallback_preserves_source_and_allows_completion(tmp_path, monkeypatch):
    title_text = "Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training"
    _write_doc(
        tmp_path,
        [
            {"block_id": "p0b0000", "reading_order": 0, "text": title_text, "layout_role": "title", "policy": {"translate": True, "preserve_reason": None, "redact_source": True}},
            {"block_id": "p0b0001", "reading_order": 1, "text": "Jie You University of Michigan", "layout_role": "header_footer", "policy": {"translate": False, "preserve_reason": "author_affiliation", "redact_source": False}},
            {"block_id": "p0b0011", "reading_order": 2, "text": P0B0011_TEXT, "layout_role": "body", "policy": {"translate": True, "preserve_reason": None, "redact_source": True}},
        ],
    )

    def responder(_n, ids, txts):
        out = {}
        for b in ids:
            if b == "p0b0000":
                out[b] = txts[b]  # title echoes verbatim
            else:
                out[b] = "在本文中，我们提出 Zeus 框架来优化 DNN 训练能耗（§3）。"
        return out

    with _MockGateway(responder) as gw:
        monkeypatch.setenv("OPENAI_BASE_URL", gw.url)
        res = worker_mod.run_llm(_make_spec(tmp_path, max_retries=1, allow_partial=False), tmp_path)

    assert res["critical_failed"] == 0
    assert res["failed_stage"] is None
    assert len(res["non_critical_fallbacks"]) == 1
    assert res["non_critical_fallbacks"][0]["block_id"] == "p0b0000"
    assert res["non_critical_fallbacks"][0]["layout_role"] == "title"
    payload = json.loads((tmp_path / "translation.payload.json").read_text(encoding="utf-8"))
    by_id = {it["block_id"]: it for it in payload["items"]}
    assert by_id["p0b0000"]["translation_status"] == "preserved_fallback"
    assert by_id["p0b0000"]["policy"]["redact_source"] is False
    assert by_id["p0b0000"]["translation"] == title_text
    assert by_id["p0b0011"]["translation_status"] == "ok"


def test_mock_gateway_http_401_reports_actual_single_attempt(tmp_path, monkeypatch):
    """Non-retryable HTTP 401 with max_retries=2 must report attempts=1 (matching 1 actual request, 0 retries)."""
    _write_doc(
        tmp_path,
        [{"block_id": "p0b0011", "reading_order": 0, "text": P0B0011_TEXT, "layout_role": "body", "policy": {"translate": True, "preserve_reason": None, "redact_source": True}}],
    )
    calls = []

    class Handler401(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            calls.append(1)
            raw = b'{"error":{"message":"invalid api key"}}'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler401)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}/v1")
        with pytest.raises(RuntimeError) as exc_info:
            worker_mod.run_llm(_make_spec(tmp_path, max_retries=2, allow_partial=False), tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert len(calls) == 1
    assert "attempts=1" in str(exc_info.value)
    diag = json.loads((tmp_path / "llm.result.json").read_text(encoding="utf-8"))
    assert diag["requests"] == 1
    assert diag["retries"] == 0
    assert diag["critical_failures"][0]["attempts"] == 1
    assert "401" in diag["critical_failures"][0]["reason"]


def test_batch_split_preserves_parent_attempts_in_child_counts():
    """When a >4-item batch exhausts timeout retries and splits in half, child items keep parent attempt counts."""
    from llm.retry import RetryPolicy
    from llm.scheduler import LLMClient, Scheduler

    class _TimeoutLargeBatchClient(LLMClient):
        name = "timeout-large-batch"

        def translate(self, items, target_lang, model, timeout_ms):
            if len(items) > 4:
                raise TimeoutError("simulated batch read timeout")
            return [f"[{target_lang}] {it['text']}" for it in items]

    policy = RetryPolicy(max_retries=1, rate_limit_base_s=0.0, rate_limit_cap_s=0.0, jitter=False)
    sc = Scheduler(
        _TimeoutLargeBatchClient(),
        concurrency=2,
        batch_size=8,
        max_retries=1,
        timeout_ms=1000,
        retry_policy=policy,
        enable_doc_context=False,
    )
    items = [{"block_id": f"p0b{i:04d}", "text": f"Sentence {i} for split test."} for i in range(6)]
    records, stats = sc.run(items, "zh", "fake")
    assert stats["succeeded_items"] == 6
    # Parent batch tried 2 times (1 + max_retries=1), then split into 2 halves each succeeding on attempt 1 -> 3 attempts/item
    assert all(rec["attempts"] == 3 for rec in records)
    assert stats["requests"] == 4


def test_algorithm_io_header_and_tiny_chart_label_preserved():
    """Structured Input/Output headers above Function Observe(...) and <=5.5pt chart labels stay preserved."""
    pages = [
        {
            "page_index": 5,
            "width": 612.0,
            "height": 792.0,
            "figure_regions": [],
            "blocks": [
                {"block_id": "p5b0024", "text": "Input: Batch size b and observed cost C", "bbox": [327.8, 86.8, 488.5, 96.9], "font_size_pt": 9.0},
                {"block_id": "p5b0025", "text": "Previous cost observations Cb for b Belief prior parameters ˆµ0 and ˆσ2", "bbox": [357.5, 98.7, 497.4, 122.3], "font_size_pt": 9.0},
                {"block_id": "p5b0026", "text": "0 Output: Belief posterior parameters ˆµb and ˆσ2", "bbox": [327.8, 115.7, 513.8, 134.4], "font_size_pt": 9.0},
                {"block_id": "p5b0028", "text": "Function Observe(b, C, Cb, ˆµ0, ˆσ2", "bbox": [327.8, 145.5, 469.3, 158.1], "font_size_pt": 9.0},
                {"block_id": "p20b0179", "text": "Batch Size", "bbox": [222.6, 670.7, 240.5, 674.9], "font_size_pt": 3.75},
            ],
        }
    ]
    norm = normalize_pages(pages, coord_space="pdf_pt", dpi=72.0)
    by_id = {b["block_id"]: b for b in norm[0]["blocks"]}
    for bid in ("p5b0024", "p5b0025", "p5b0026"):
        assert by_id[bid]["policy"]["translate"] is False
        assert by_id[bid]["policy"]["preserve_reason"] == "algorithm_io_header"
        assert by_id[bid]["policy"]["redact_source"] is False
    assert by_id["p20b0179"]["policy"]["translate"] is False
    assert by_id["p20b0179"]["policy"]["preserve_reason"] == "tiny_figure_label"


