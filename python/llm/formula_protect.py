"""Inline formula, complexity notation, and citation placeholder protection."""
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
    # URLs
    re.compile(r"https?://[^\s)\]]+"),
]


def protect_formulas(text: str) -> tuple[str, dict[str, str]]:
    """Replace protected inline tokens with ⟪F0⟫, ⟪F1⟫, ... placeholders."""
    if not text:
        return text, {}
    tokens: dict[str, str] = {}
    out = text
    idx = 0
    for pat in _PATTERNS:
        def _repl(m: re.Match) -> str:
            nonlocal idx
            val = m.group(0)
            key = f"⟪F{idx}⟫"
            idx += 1
            tokens[key] = val
            return key
        out = pat.sub(_repl, out)
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
