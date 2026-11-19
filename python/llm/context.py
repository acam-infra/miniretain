"""Zero-cost deterministic document context & terminology anchor extraction.

Builds a compact cross-batch style/terminology hint from the document's own
title and recurring technical acronyms so parallel batches translate terms
consistently without extra LLM round-trips.
"""
from __future__ import annotations

from collections import Counter
import re

_ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z0-9\-]{1,10}\b")
_MULTI_UPPER_RE = re.compile(r"\b[A-Z]{2,}[0-9a-z]*\b")
_STOP_ACRONYMS = {
    "THE", "AND", "FOR", "WITH", "FROM", "THAT", "THIS", "INTO", "OVER",
    "IEEE", "FIG", "TABLE", "SECTION", "ABSTRACT", "INTRODUCTION", "CONCLUSION",
    "REFERENCES", "APPENDIX", "II", "III", "IV", "VI", "VII", "VIII",
}

# Canonical academic terms whose English form or unified Chinese translation
# should remain consistent across parallel batches when present in source text.
_CANONICAL_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bKV\s+[Cc]ache\b"), "KV Cache (keep exact casing)"),
    (re.compile(r"\bPrefill\b", re.IGNORECASE), "Prefill (keep English term)"),
    (re.compile(r"\bDecode\b", re.IGNORECASE), "Decode (keep English term)"),
    (re.compile(r"\bdisaggregat(?:ed|ion)\b", re.IGNORECASE), "disaggregated -> 分离式"),
    (re.compile(r"\binter-satellite\s+(?:optical\s+)?links?\b", re.IGNORECASE), "inter-satellite optical link -> 星间光链路 (ISL)"),
    (re.compile(r"\borbital\s+data\s+centers?\b", re.IGNORECASE), "orbital data center -> 轨道数据中心"),
    (re.compile(r"\battention\b", re.IGNORECASE), "attention -> 注意力"),
    (re.compile(r"\bexpert\b", re.IGNORECASE), "expert -> 专家"),
    (re.compile(r"\bground\s+station\b", re.IGNORECASE), "ground station -> 地面站"),
]


def build_document_context(items: list[dict], max_acronyms: int = 12) -> str:
    """Return a compact (< 350 chars) cross-batch terminology & style anchor."""
    if not items:
        return ""
    title_parts: list[str] = []
    acronym_counts: Counter[str] = Counter()
    full_text_chunks: list[str] = []

    for it in items:
        txt = (it.get("text") or "").strip()
        if not txt:
            continue
        if it.get("layout_role") == "title" and len(title_parts) < 2:
            title_parts.append(txt)
        full_text_chunks.append(txt)
        for m in _MULTI_UPPER_RE.findall(txt):
            if m not in _STOP_ACRONYMS and len(m) <= 10:
                acronym_counts[m] += 1

    joined = "\n".join(full_text_chunks)
    recurring = [
        tok for tok, cnt in acronym_counts.most_common(max_acronyms * 2)
        if cnt >= 2
    ][:max_acronyms]

    term_notes: list[str] = []
    for pat, hint in _CANONICAL_HINTS:
        if pat.search(joined):
            term_notes.append(hint)

    lines: list[str] = []
    if title_parts:
        title_str = " ".join(title_parts)[:140]
        lines.append(f"Paper title: {title_str}")
    if recurring:
        lines.append("Keep recurring acronyms unchanged: " + ", ".join(recurring))
    if term_notes:
        lines.append("Consistent terms across all blocks: " + "; ".join(term_notes[:8]))
    lines.append(
        "Use '图 N' for 'Fig. N'/'Figure N' and '表 N' for 'Table N'; "
        "preserve all numbers, units, citations [N], URLs, and ⟪F...⟫ placeholders exactly."
    )
    return "\n".join(lines)
