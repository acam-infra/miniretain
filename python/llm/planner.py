"""Batch planning contracts for LLM translation.

Separates batch grouping & scheduling order (character-budget + LPT) from
execution, concurrency governance, and result backfill.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlannedBatch:
    """One scheduled batch preserving original item indices for deterministic backfill."""

    batch_id: int
    indexed_items: list[tuple[int, dict]]
    char_count: int
    pages: tuple[int, ...] = field(default_factory=tuple)

    @property
    def items(self) -> list[dict]:
        return [it for _, it in self.indexed_items]


class BatchPlanner:
    """Base contract for planning translation batches."""

    def plan(self, items: list[dict]) -> list[PlannedBatch]:
        raise NotImplementedError


class CharBudgetLPTPlanner(BatchPlanner):
    """Pack items into character-budgeted batches (<= batch_size items each)
    and order longest-processing-time (LPT) first to minimize parallel tail latency.

    Keeping consecutive reading-order blocks together within each batch preserves
    local coherence while LPT ensures heavy batches launch immediately.
    """

    def __init__(
        self,
        batch_size: int,
        max_chars: int | None = None,
        min_items_before_char_split: int = 3,
        lpt: bool = True,
    ) -> None:
        self.batch_size = max(1, int(batch_size))
        self.max_chars = (
            max(1, int(max_chars))
            if max_chars is not None
            else max(3600, self.batch_size * 280)
        )
        self.min_items_before_char_split = max(1, int(min_items_before_char_split))
        self.lpt = bool(lpt)

    def plan(self, items: list[dict]) -> list[PlannedBatch]:
        if not items:
            return []
        raw_batches: list[list[tuple[int, dict]]] = []
        cur: list[tuple[int, dict]] = []
        cur_chars = 0
        for idx, item in enumerate(items):
            ilen = len(item.get("text") or "")
            if cur and (
                len(cur) >= self.batch_size
                or (
                    cur_chars + ilen > self.max_chars
                    and len(cur) >= self.min_items_before_char_split
                )
            ):
                raw_batches.append(cur)
                cur = []
                cur_chars = 0
            cur.append((idx, item))
            cur_chars += ilen
        if cur:
            raw_batches.append(cur)

        planned: list[PlannedBatch] = []
        for bid, b in enumerate(raw_batches):
            c_count = sum(len(it.get("text") or "") for _, it in b)
            pages = tuple(
                sorted({int(it["page_index"]) for _, it in b if "page_index" in it})
            )
            planned.append(
                PlannedBatch(
                    batch_id=bid,
                    indexed_items=b,
                    char_count=c_count,
                    pages=pages,
                )
            )
        if self.lpt:
            planned.sort(key=lambda b: b.char_count, reverse=True)
        return planned
