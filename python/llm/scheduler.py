"""LLM task scheduling boundary.

LLMClient is the only thing that talks to a model provider. Scheduler owns
batching, bounded concurrency, retry/backoff and result ordering. Swapping the
stub for a real HTTP client needs no change to the scheduler or the Go side.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class LLMClient(ABC):
    name = "base"

    @abstractmethod
    def translate(
        self, items: list[dict], target_lang: str, model: str, timeout_ms: int
    ) -> list[str]:
        """Translate a batch, returning one string per item, in the same order."""


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
        self.stats = {"batches": 0, "retries": 0, "failed_batches": 0}

    def _run_batch(self, batch, target_lang, model):
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.translate(batch, target_lang, model, self.timeout_ms), attempt
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
        results: list[str | None] = [None] * len(items)

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(self._run_batch, batch, target_lang, model): (start, batch)
                for start, batch in batches
            }
            for fut in as_completed(futures):
                start, batch = futures[fut]
                try:
                    translated, retries = fut.result()
                    self.stats["retries"] += retries
                except Exception as exc:  # noqa: BLE001 - keep the job runnable
                    self.stats["failed_batches"] += 1
                    translated = [f"<translation failed: {exc}>"] * len(batch)
                for offset, value in enumerate(translated):
                    results[start + offset] = value

        return results, self.stats
