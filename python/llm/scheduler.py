"""LLM task scheduling boundary.

LLMClient is the only thing that talks to a model provider. Scheduler owns
batching, bounded concurrency, retry/backoff, result ordering, and explicit
per-item status accounting (no silent placeholder-as-success).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
    ) -> None:
        self.client = client
        self.concurrency = max(1, concurrency)
        self.batch_size = max(1, batch_size)
        self.max_retries = max(0, max_retries)
        self.timeout_ms = timeout_ms
        self.stats = {
            "batches": 0,
            "retries": 0,
            "failed_batches": 0,
            "total_items": 0,
            "succeeded_items": 0,
            "failed_items": 0,
            "missing_items": 0,
            "empty_items": 0,
            "failed_block_ids": [],
            "missing_block_ids": [],
            "empty_block_ids": [],
        }

    def _run_batch(self, batch, target_lang, model):
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                out = self.client.translate(batch, target_lang, model, self.timeout_ms)
                if len(out) != len(batch):
                    raise RuntimeError(
                        f"batch length mismatch: expected {len(batch)}, got {len(out)}"
                    )
                return out, attempt
            except Exception as exc:  # noqa: BLE001 - retry any client error
                last = exc
                if attempt < self.max_retries:
                    time.sleep(min(0.05 * (2**attempt), 0.5))
        raise last

    def run(self, items: list[dict], target_lang: str, model: str):
        batches = [
            (start, items[start : start + self.batch_size])
            for start in range(0, len(items), self.batch_size)
        ]
        self.stats["batches"] = len(batches)
        self.stats["total_items"] = len(items)
        results: list[dict | None] = [None] * len(items)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(self._run_batch, batch, target_lang, model): (start, batch)
                for start, batch in batches
            }
            for fut in as_completed(futures):
                start, batch = futures[fut]
                try:
                    raw_out, retries = fut.result()
                    self.stats["retries"] += retries
                    norm = [_normalize_item_result(item, val) for item, val in zip(batch, raw_out)]
                except Exception as exc:  # noqa: BLE001 - record explicit failure per item
                    self.stats["failed_batches"] += 1
                    err_msg = str(exc)
                    norm = [
                        {
                            "block_id": item["block_id"],
                            "status": "failed",
                            "translation": None,
                            "error": err_msg,
                        }
                        for item in batch
                    ]
                for offset, rec in enumerate(norm):
                    results[start + offset] = rec

        # Targeted single-item retry for any missing/empty items when max_retries > 0
        if self.max_retries > 0:
            retry_indices = [
                i for i, rec in enumerate(results)
                if rec is not None and rec["status"] in ("missing", "empty")
            ]
            for idx in retry_indices:
                item = items[idx]
                try:
                    raw_out, retries = self._run_batch([item], target_lang, model)
                    self.stats["retries"] += 1 + retries
                    rec = _normalize_item_result(item, raw_out[0])
                    results[idx] = rec
                except Exception as exc:  # noqa: BLE001
                    results[idx] = {
                        "block_id": item["block_id"],
                        "status": "failed",
                        "translation": None,
                        "error": str(exc),
                    }

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
