"""Inline formula, complexity notation, URL/DOI, and reference field protection."""
from __future__ import annotations

import re

_PATTERNS = [
    # Display math $$...$$ and inline math $...$
    re.compile(r"\$\$[^$]+\$\$"),
    re.compile(r"(?<!\\)\$[^$\n]+\$"),
    # LaTeX commands with braces like \mathcal{O}(...), \frac{...}{...}
    re.compile(r"\\[a-zA-Z]+\{[^}]*\}(?:\{[^}]*\})?"),
    # Big-O / memory notations like O(GB), O(KB), O(MB), O(n log n), O(1)
    re.compile(r"\bO\([A-Za-z0-9_+\-*/^.,\s]+\)"),
    # URLs and DOIs
    re.compile(r"https?://[^\s)\]]+"),
    re.compile(r"\bdoi:\s*[^\s,;]+", re.IGNORECASE),
]

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

    for pat in _PATTERNS:
        out = pat.sub(lambda m: _sub_token(m.group(0)), out)

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
