"""Inline formula, complexity notation, URL/DOI, and reference field protection."""
from __future__ import annotations

import re
from typing import Any

_PATTERNS = [
    # Display math $$...$$ and \[...\]
    re.compile(r"(?<!\$)\$\$(?!\$)[^$]+?(?<!\$)\$\$(?!\$)"),
    re.compile(r"\\\[[\s\S]+?\\\]"),
    # Inline math \(...\)
    re.compile(r"\\\([^\n]+?\\\)"),
    # LaTeX commands with braces like \mathcal{O}(...), \frac{...}{...}
    re.compile(r"\\[a-zA-Z]+\{[^}]*\}(?:\{[^}]*\})?"),
    # Big-O / memory notations like O(GB), O(KB), O(MB), O(n log n), O(1)
    re.compile(r"\bO\([A-Za-z0-9_+\-*/^.,\s]+\)"),
    # URLs and DOIs
    re.compile(r"https?://[^\s)\]]+"),
    re.compile(r"\bdoi:\s*[^\s,;]+", re.IGNORECASE),
]

_CURRENCY_UNIT_RE = re.compile(
    r"^\d[\d,.]*(?:\s*[-–—~]\s*\d[\d,.]*)?\s*/\s*(?:kg|g|mg|lb|ton|t|kW|MW|GW|kWh|MWh|Wh|W|y|yr|year|mo|month|d|day|hr|h|s|m|km|L|gal|unit|item|token|tokens|req|GB|TB|MB)\b",
    re.IGNORECASE,
)
_BARE_PROSE_WORD_RE = re.compile(r"(?<![\\A-Za-z_])[a-z]{3,}(?![A-Za-z_])")
_MATH_STOPWORDS = frozenset({
    "the", "and", "for", "sin", "cos", "tan", "sec", "csc", "cot", "log", "exp",
    "max", "min", "sup", "inf", "lim", "mod", "det", "dim", "arg", "deg", "ker",
    "var", "cov", "iff", "let", "set",
})


def _looks_like_non_math_dollar(inner: str, after_char: str = "") -> bool:
    if not inner or inner != inner.strip():
        return True
    if re.match(r"^(?:[<>~≈]+\s*)?/\s*[A-Za-z]", inner):
        return True
    if "\\" not in inner and _CURRENCY_UNIT_RE.match(inner):
        return True
    if after_char.isdigit() and not re.search(r"\\[A-Za-z]+", inner):
        return True
    if re.match(r"^\d[\d,.\s]*(?:[-–—]|to|and|or)?$", inner, re.IGNORECASE):
        return True
    if re.match(r"^\d", inner) and not any(ch in inner for ch in r"\^_{}=<>"):
        return True
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", inner) and "\\" not in inner:
        return True
    if not re.search(r"[\\^{}=]", inner):
        prose_words = [w for w in _BARE_PROSE_WORD_RE.findall(inner) if w not in _MATH_STOPWORDS]
        if len(prose_words) >= 3:
            return True
    return False


def _protect_inline_dollar_math(text: str, sub_fn: Any) -> str:
    if "$" not in text:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n and text[i + 1] == "$":
            out.append("\\$")
            i += 2
            continue
        if ch == "$":
            j = i + 1
            end = -1
            while j < n:
                if text[j] == "\n":
                    break
                if text[j] == "\\" and j + 1 < n and text[j + 1] == "$":
                    j += 2
                    continue
                if text[j] == "$":
                    end = j
                    break
                j += 1
            if end != -1 and end > i + 1:
                inner = text[i + 1 : end]
                after_char = text[end + 1] if end + 1 < n else ""
                if not _looks_like_non_math_dollar(inner, after_char=after_char):
                    out.append(sub_fn(text[i : end + 1]))
                    i = end + 1
                    continue
        out.append(ch)
        i += 1
    return "".join(out)

_REF_ID_RE = re.compile(r"^\[\d+\]")
_REF_AUTHORS_BEFORE_QUOTE_RE = re.compile(
    r"^(?:⟪F\d+⟫\s*)?([A-Z]\.\s*[^“\"]+?)(?=[“\"])"
)


def protect_formulas(text: str, layout_role: str = "body") -> tuple[str, dict[str, str]]:
    """Replace protected inline tokens (and reference identifiers/authors) with ⟪F0⟫..."""
    if not text:
        return text, {}
    tokens: dict[str, str] = {}
    out = text
    idx = 0

    def _sub_token(val: str) -> str:
        nonlocal idx
        key = f"⟪F{idx}⟫"
        idx += 1
        tokens[key] = val
        return key

    if layout_role == "reference":
        # Protect leading [N] reference marker
        m_id = _REF_ID_RE.match(out)
        if m_id:
            out = _sub_token(m_id.group(0)) + out[m_id.end():]

    for pat_idx, pat in enumerate(_PATTERNS):
        out = pat.sub(lambda m: _sub_token(m.group(0)), out)
        if pat_idx == 2:
            out = _protect_inline_dollar_math(out, _sub_token)

    if layout_role == "reference":
        # Protect author list between [N] and opening quote “..."
        m_auth = _REF_AUTHORS_BEFORE_QUOTE_RE.match(out)
        if m_auth and len(m_auth.group(1).strip()) >= 3:
            span_start, span_end = m_auth.span(1)
            auth_txt = m_auth.group(1).strip()
            # Keep trailing space before quote
            out = out[:span_start] + _sub_token(auth_txt) + " " + out[span_end:]

    return out, tokens


def restore_formulas(text: str, tokens: dict[str, str]) -> tuple[str, list[str]]:
    """Restore ⟪F...⟫ placeholders. Returns (restored_text, missing_keys)."""
    if not tokens:
        return text, []
    out = text
    missing: list[str] = []
    for key, val in tokens.items():
        if key in out:
            out = out.replace(key, val)
        else:
            missing.append(key)
    return out, missing
