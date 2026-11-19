"""LLM task scheduling boundary.

LLMClient is the only thing that talks to a model provider.
Responsibilities are cleanly separated into testable collaborators:
- BatchPlanner (`llm.planner`): character-budget packing & LPT ordering.
- ConcurrencyLimiter (`llm.limiter`): bounded in-process + optional cross-process
  global concurrency budget (prevents Go max_workers * llm.concurrency amplification).
- RetryPolicy (`llm.retry`): error classification, jittered backoff, Retry-After,
  and shared 429 cooldown across sibling workers.
- Document context (`llm.context`): zero-cost terminology & style anchor.
- Scheduler (this module): orchestrates execution, immediate streaming item retry
  (no barrier wait), deterministic block_id backfill, and explicit status metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import threading
import time

from llm.context import build_document_context
from llm.limiter import ConcurrencyLimiter, build_limiter_from_env
from llm.planner import BatchPlanner, CharBudgetLPTPlanner
from llm.retry import RetryPolicy

_SENTINEL_PREFIXES = ("<translation failed:", "<missing translation")


class LLMClient(ABC):
    name = "base"

    @abstractmethod
    def translate(
        self, items: list[dict], target_lang: str, model: str, timeout_ms: int
    ) -> list[str | dict]:
        """Translate a batch, returning one string or status dict per item, in order."""


class StubLLMClient(LLMClient):
    """Deterministic offline client so the whole path is testable without keys."""

    name = "stub"

    def translate(self, items, target_lang, model, timeout_ms):
        return [f"[{target_lang}] {item['text']}" for item in items]


from llm.openai_client import OpenAICompatLLMClient  # noqa: E402,F401

REGISTRY: dict[str, LLMClient] = {c.name: c for c in [StubLLMClient()]}
REGISTRY[OpenAICompatLLMClient.name] = OpenAICompatLLMClient()


def get_client(name: str) -> LLMClient:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown LLM provider {name!r}; available: {sorted(REGISTRY)}") from exc


def _normalize_item_result(item: dict, raw: object) -> dict:
    bid = item["block_id"]
    if isinstance(raw, dict):
        status = raw.get("status", "ok")
        trans = raw.get("translation")
        err = raw.get("error")
        ret_id = raw.get("block_id", bid)
        if ret_id != bid:
            return {
                "block_id": bid,
                "status": "failed",
                "translation": None,
                "error": f"block_id mismatch: expected {bid}, got {ret_id}",
            }
        if status == "ok":
            if not isinstance(trans, str) or not trans.strip():
                return {"block_id": bid, "status": "empty", "translation": None, "error": "empty translation"}
            if trans.strip().startswith(_SENTINEL_PREFIXES):
                return {"block_id": bid, "status": "failed", "translation": None, "error": trans.strip()}
        return {"block_id": bid, "status": status, "translation": trans, "error": err}

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {"block_id": bid, "status": "empty", "translation": None, "error": "empty translation"}
        if s.startswith(_SENTINEL_PREFIXES):
            return {"block_id": bid, "status": "failed", "translation": None, "error": s}
        return {"block_id": bid, "status": "ok", "translation": s, "error": None}

    return {
        "block_id": bid,
        "status": "missing" if raw is None else "failed",
        "translation": None,
        "error": f"unexpected translation value type: {type(raw).__name__}",
    }


class Scheduler:
    def __init__(
        self,
        client: LLMClient,
        concurrency: int,
        batch_size: int,
        max_retries: int,
        timeout_ms: int,
        *,
        planner: BatchPlanner | None = None,
        limiter: ConcurrencyLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
        enable_doc_context: bool = True,
    ) -> None:
        self.client = client
        self.concurrency = max(1, concurrency)
        self.batch_size = max(1, batch_size)
        self.max_retries = max(0, max_retries)
        self.timeout_ms = timeout_ms
        self.planner = planner or CharBudgetLPTPlanner(batch_size=self.batch_size)
        self.limiter = limiter or build_limiter_from_env(self.concurrency)
        self.retry_policy = retry_policy or RetryPolicy(max_retries=self.max_retries)
        self.enable_doc_context = bool(enable_doc_context)
        self._metrics_lock = threading.Lock()
        self.stats = {
            "batches": 0,
            "requests": 0,
            "retries": 0,
            "rate_limit_429": 0,
            "timeouts": 0,
            "failed_batches": 0,
            "peak_inflight": 0,
            "limiter_wait_s": 0.0,
            "total_items": 0,
            "succeeded_items": 0,
            "failed_items": 0,
            "missing_items": 0,
            "empty_items": 0,
            "failed_block_ids": [],
            "missing_block_ids": [],
            "empty_block_ids": [],
            "elapsed_s": 0.0,
            "batch_latencies_s": [],
            "max_batch_s": 0.0,
            "retry_s": 0.0,
        }

    def _run_batch(self, batch: list[dict], target_lang: str, model: str, doc_context: str = ""):
        t0 = time.perf_counter()
        last: Exception | None = None
        payload_batch = (
            [{**batch[0], "_doc_context": doc_context}, *batch[1:]]
            if (doc_context and batch)
            else batch
        )
        for attempt in range(self.retry_policy.max_retries + 1):
            self.retry_policy.wait_if_cooling_down()
            try:
                with self.limiter.acquire():
                    with self._metrics_lock:
                        self.stats["requests"] += 1
                    out = self.client.translate(payload_batch, target_lang, model, self.timeout_ms)
                if len(out) != len(batch):
                    raise RuntimeError(
                        f"batch length mismatch: expected {len(batch)}, got {len(out)}"
                    )
                return out, attempt, round(time.perf_counter() - t0, 3)
            except Exception as exc:  # noqa: BLE001 - classify and retry via policy
                last = exc
                decision = self.retry_policy.decide(exc, attempt)
                with self._metrics_lock:
                    if decision.category == "rate_limit":
                        self.stats["rate_limit_429"] += 1
                    elif decision.category == "timeout":
                        self.stats["timeouts"] += 1
                if decision.cooldown_s > 0:
                    self.retry_policy.note_cooldown(decision.cooldown_s)
                if decision.retryable:
                    with self._metrics_lock:
                        self.stats["retries"] += 1
                    if decision.delay_s > 0:
                        time.sleep(decision.delay_s)
                    continue
                raise
        assert last is not None
        raise last

    def _build_batches(self, items: list[dict]) -> list[list[tuple[int, dict]]]:
        """Compatibility helper delegating to the configured BatchPlanner."""
        return [pb.indexed_items for pb in self.planner.plan(items)]

    def run(self, items: list[dict], target_lang: str, model: str):
        t_run_0 = time.perf_counter()
        planned_batches = self.planner.plan(items)
        self.stats["batches"] = len(planned_batches)
        self.stats["total_items"] = len(items)
        results: list[dict | None] = [None] * len(items)
        batch_latencies: list[float] = []
        doc_context = build_document_context(items) if self.enable_doc_context else ""

        if not planned_batches:
            self.stats["elapsed_s"] = round(time.perf_counter() - t_run_0, 3)
            return [], self.stats

        pool_workers = max(1, min(self.concurrency, self.limiter.limit))
        main_done_at: float | None = None
        main_remaining = len(planned_batches)
        retried_indices: set[int] = set()

        with ThreadPoolExecutor(max_workers=pool_workers) as pool:
            # Map future -> (kind, payload_meta)
            # kind == "batch": meta = indexed_items
            # kind == "item_retry": meta = orig_idx
            pending: dict[Future, tuple[str, object]] = {}
            for pb in planned_batches:
                fut = pool.submit(self._run_batch, pb.items, target_lang, model, doc_context)
                pending[fut] = ("batch", pb.indexed_items)

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    kind, meta = pending.pop(fut)
                    if kind == "batch":
                        indexed_batch = meta
                        assert isinstance(indexed_batch, list)
                        batch = [it for _, it in indexed_batch]
                        try:
                            raw_out, _retries, lat_s = fut.result()
                            batch_latencies.append(lat_s)
                            norm = [
                                _normalize_item_result(item, val)
                                for item, val in zip(batch, raw_out)
                            ]
                        except Exception as exc:  # noqa: BLE001 - record explicit failure per item
                            err_msg = str(exc)
                            if (
                                self.max_retries > 0
                                and len(indexed_batch) > 4
                                and self.retry_policy.decide(exc, self.max_retries).category in ("timeout", "parse_error")
                            ):
                                mid = len(indexed_batch) // 2
                                for half in (indexed_batch[:mid], indexed_batch[mid:]):
                                    half_items = [it for _, it in half]
                                    with self._metrics_lock:
                                        self.stats["retries"] += 1
                                    hf = pool.submit(
                                        self._run_batch, half_items, target_lang, model, doc_context
                                    )
                                    pending[hf] = ("batch", half)
                                    main_remaining += 1
                                main_remaining -= 1
                                continue
                            self.stats["failed_batches"] += 1
                            norm = [
                                {
                                    "block_id": item["block_id"],
                                    "status": "failed",
                                    "translation": None,
                                    "error": err_msg,
                                }
                                for item in batch
                            ]

                        for (orig_idx, item), rec in zip(indexed_batch, norm):
                            results[orig_idx] = rec
                            # Immediate streaming single-item retry without waiting for all batches!
                            if (
                                self.max_retries > 0
                                and rec["status"] in ("missing", "empty")
                                and orig_idx not in retried_indices
                            ):
                                retried_indices.add(orig_idx)
                                with self._metrics_lock:
                                    self.stats["retries"] += 1
                                rf = pool.submit(
                                    self._run_batch, [item], target_lang, model, doc_context
                                )
                                pending[rf] = ("item_retry", orig_idx)

                        main_remaining -= 1
                        if main_remaining == 0:
                            main_done_at = time.perf_counter()
                    else:
                        orig_idx = meta
                        assert isinstance(orig_idx, int)
                        item = items[orig_idx]
                        try:
                            raw_out, _retries, _lat = fut.result()
                            results[orig_idx] = _normalize_item_result(item, raw_out[0])
                        except Exception as exc:  # noqa: BLE001
                            results[orig_idx] = {
                                "block_id": item["block_id"],
                                "status": "failed",
                                "translation": None,
                                "error": str(exc),
                            }

        t_end = time.perf_counter()
        self.stats["retry_s"] = (
            round(max(0.0, t_end - main_done_at), 3)
            if (main_done_at is not None and retried_indices)
            else 0.0
        )
        self.stats["batch_latencies_s"] = sorted(batch_latencies)
        self.stats["max_batch_s"] = max(batch_latencies) if batch_latencies else 0.0
        self.stats["elapsed_s"] = round(t_end - t_run_0, 3)

        lim_snap = self.limiter.snapshot()
        self.stats["peak_inflight"] = lim_snap["peak_inflight"]
        self.stats["limiter_wait_s"] = lim_snap["wait_s"]
        if "global_limit" in lim_snap:
            self.stats["global_concurrency_limit"] = lim_snap["global_limit"]

        for rec in results:
            assert rec is not None
            st = rec["status"]
            if st == "ok":
                self.stats["succeeded_items"] += 1
            elif st == "missing":
                self.stats["missing_items"] += 1
                self.stats["missing_block_ids"].append(rec["block_id"])
            elif st == "empty":
                self.stats["empty_items"] += 1
                self.stats["empty_block_ids"].append(rec["block_id"])
            else:
                self.stats["failed_items"] += 1
                self.stats["failed_block_ids"].append(rec["block_id"])

        return results, self.stats
