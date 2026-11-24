"""Ported formula normalization, layout safety, CJK text normalization, and redaction guard
mechanisms from upstream wxyhgk/retain-pdf (retainpdf_math & retainpdf_pipeline).
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import pymupdf

MAX_INLINE_MATH_CHARS = 1200
MIN_SAFE_CONTENT_HEIGHT_PT = 8.0
MAX_FORMULA_INSET_HEIGHT_RATIO = 0.18
LONG_INLINE_MATH_MIN_BODY_CHARS = 24
LONG_INLINE_MATH_WIDTH_RATIO = 0.42
FORMULA_WIDTH_UNIT_PT_RATIO = 1.0
MIN_CLEANUP_SEGMENT_WIDTH_PT = 1.0
MIN_CLEANUP_SEGMENT_HEIGHT_PT = 1.5
MIN_CLEANUP_SEGMENT_AREA_PT2 = 2.0

_ROW_SEPARATOR_CONTEXT_RE = re.compile(r"\\(?:begin|substack|atop|matrix|cases|array)\b")
_DOUBLED_COMMAND_RE = re.compile(r"\\{2,}(?=[A-Za-z]{2,})")
_SCRIPT_KEEPING_COMMANDS = {"limits", "nolimits", "displaylimits"}
_LEFT_RIGHT_RE = re.compile(r"\\(left|right)(?![A-Za-z])")

_ANGSTROM_COMMAND_RE = re.compile(r"\\(AA|aa)(?![A-Za-z])(?:\{\})?")
_RING_ACCENT_RE = re.compile(r"\\r(?:\s*\{\s*([Aa])\s*\}|\s+([Aa])(?![A-Za-z]))")
_MATHRING_A_RE = re.compile(r"\\mathring(?:\s*\{\s*([Aa])\s*\}|\s+([Aa])(?![A-Za-z]))")
_LATEX_TEXT_GROUP_RE = re.compile(r"\\(?:text|textrm|textup|textnormal|textit|textbf|textsf|texttt)\s*\{")
_ANGSTROM_BY_LETTER = {"A": "Å", "a": "å"}

_SQRT_ARG_RE = re.compile(
    r"√(?:\{([^{}]+)\}|\(([^()]+)\)|((?:\\[A-Za-z]+|[^\s^_{}()+\-−=<>.,;:])(?:_(?:\{[^{}]+\}|[^\s{}])|\^(?:\{[^{}]+\}|[^\s{}]))*))"
)
_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_PURE_SUPERSCRIPT_RE = re.compile(
    r"^\$?\s*(?:\{\})?\^\s*(?:\{[0-9a-zA-Z,\-–*†‡§\\dagger\\ddagger\s]+\}|[0-9a-zA-Z*†‡§]|\\dagger|\\ddagger)\s*\$?$"
)
_SCRIPT_OR_TALL_MATH_RE = re.compile(
    r"[_^]|\\(?:frac|dfrac|tfrac|sqrt|sum|prod|int|iint|iiint|lim|underset|overset|substack)\b"
)
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
_LATEX_MARKER_RE = re.compile(r"\\[A-Za-z]{2,}|\^\{|_\{")


def strip_control_chars(text: str) -> str:
    return _CTRL_CHAR_RE.sub("", text) if text else text


def is_cjk_char(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def is_fullwidth_punctuation(char: str) -> bool:
    if not char or len(char) != 1:
        return False
    code = ord(char)
    if not (0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF or char in "“”‘’—…·"):
        return False
    return unicodedata.category(char).startswith("P")


def is_pure_superscript(expr: str) -> bool:
    """Return True for citation/footnote pure superscript formulas like $^{14}$ or $^\\dagger$."""
    return bool(_PURE_SUPERSCRIPT_RE.match(str(expr or "").strip()))


def collapse_doubled_command_backslashes(expr: str) -> str:
    """Collapse OCR/LLM doubled command backslashes (\\\\alpha -> \\alpha) outside row-separator environments."""
    text = str(expr or "")
    if _ROW_SEPARATOR_CONTEXT_RE.search(text):
        return text
    return _DOUBLED_COMMAND_RE.sub(r"\\", text)


def _script_argument_end(text: str, start: int) -> int:
    i = start
    while i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text):
        return i
    if text[i] == "{":
        depth = 0
        while i < len(text):
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return i
    if text[i] == "\\":
        j = i + 1
        while j < len(text) and text[j].isalpha():
            j += 1
        return max(j, i + 2)
    return i + 1


def separate_double_scripts(expr: str) -> str:
    """Insert {} between duplicate subscripts (F_1_* -> F_1{}_*) or superscripts (x^2^3 -> x^2{}^3)."""
    text = str(expr or "")
    if "_" not in text and "^" not in text:
        return text
    out: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(text):
        char = text[i]
        if char in "_^":
            if char in seen:
                out.append("{}")
                seen = set()
            end = _script_argument_end(text, i + 1)
            out.append(text[i:end])
            seen.add(char)
            i = end
            continue
        if char == " ":
            out.append(char)
            i += 1
            continue
        if char == "\\":
            j = i + 1
            while j < len(text) and text[j].isalpha():
                j += 1
            j = max(j, i + 2)
            if text[i + 1 : j] not in _SCRIPT_KEEPING_COMMANDS:
                seen = set()
            out.append(text[i:j])
            i = j
            continue
        if char == "{":
            end = _script_argument_end(text, i)
            out.append(text[i:end])
            seen = set()
            i = end
            continue
        out.append(char)
        seen = set()
        i += 1
    return "".join(out)


def balance_left_right(expr: str) -> str:
    """Balance unmatched \\left / \\right delimiters with empty \\left. / \\right. delimiters."""
    text = str(expr or "")
    if "\\left" not in text and "\\right" not in text:
        return text
    open_count = 0
    missing_left = 0
    for match in _LEFT_RIGHT_RE.finditer(text):
        if match.group(1) == "left":
            open_count += 1
        elif open_count:
            open_count -= 1
        else:
            missing_left += 1
    if not open_count and not missing_left:
        return text
    return "\\left. " * missing_left + text + " \\right." * open_count


def _ring_letter(match: re.Match[str]) -> str:
    return _ANGSTROM_BY_LETTER[match.group(1) or match.group(2)]


def _replace_in_latex_text_groups(expr: str, replacer) -> str:
    chunks: list[str] = []
    index = 0
    while True:
        match = _LATEX_TEXT_GROUP_RE.search(expr, index)
        if match is None:
            chunks.append(expr[index:])
            return "".join(chunks)
        body_start = match.end()
        cursor = body_start
        depth = 1
        while cursor < len(expr):
            char = expr[cursor]
            if char == "\\":
                cursor += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        if depth != 0:
            chunks.append(expr[index:])
            return "".join(chunks)
        chunks.append(expr[index:body_start])
        chunks.append(replacer(expr[body_start:cursor]))
        index = cursor


def normalize_angstrom_in_math(expr: str) -> str:
    if "\\" not in (expr or ""):
        return expr
    expr = _replace_in_latex_text_groups(expr, lambda body: _MATHRING_A_RE.sub(_ring_letter, body))
    expr = _ANGSTROM_COMMAND_RE.sub(lambda match: _ANGSTROM_BY_LETTER[match.group(1)[0]], expr)
    return _RING_ACCENT_RE.sub(_ring_letter, expr)


def normalize_angstrom_in_text(text: str) -> str:
    if "\\" not in (text or ""):
        return text
    return _ANGSTROM_COMMAND_RE.sub(lambda match: _ANGSTROM_BY_LETTER[match.group(1)[0]], text)


def _compact_ocr_math_whitespace(expr: str) -> str:
    r"""Compact extra spaces emitted by MinerU / PaddleOCR inside LaTeX subscripts, superscripts,
    commands (e.g. '\Phi_ {k} (q _ {k})', 'D _ {k} ^ {\max}', '\frac {a}{b}'), and wrap multi-letter
    set words like '\{sat, gnd\}' in \mathrm{...}."""
    s = expr
    s = re.sub(
        r"\\(mathrm|text|textrm|mathbf|operatorname)\s*\{\s*([A-Za-z](?:\s+[A-Za-z])+)\s*\}",
        lambda m: "\\" + m.group(1) + "{" + "".join(m.group(2).split()) + "}",
        s,
    )
    prev = None
    while s != prev:
        prev = s
        s = re.sub(r"([_^])\s+\{", r"\1{", s)
        s = re.sub(r"_\{\s*([^{}]+?)\s*\}", lambda m: "_{" + m.group(1).strip() + "}", s)
        s = re.sub(r"\^\{\s*([^{}]+?)\s*\}", lambda m: "^{" + m.group(1).strip() + "}", s)
        s = re.sub(r"(\\[A-Za-z]+|[A-Za-z0-9\}\)\]])\s+([_^])", r"\1\2", s)
        s = re.sub(r"\\(frac|dfrac|tfrac|sqrt|hat|bar|tilde|vec|mathcal|mathrm|mathbf|mathbb|operatorname)\s+\{", r"\\\1{", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"(?<=[A-Za-z0-9\}])\s+\(", "(", s)
    s = re.sub(r"\\\{\s+", r"\{", s)
    s = re.sub(r"\s+\\\}", r"\}", s)
    s = re.sub(r"\s+/\s+", "/", s)
    s = re.sub(
        r"\\\{\s*([a-z]{2,}(?:\s*,\s*[a-z]{2,})+)\s*\\\}",
        lambda m: "\\{" + ", ".join(f"\\mathrm{{{w.strip()}}}" for w in m.group(1).split(",")) + "\\}",
        s,
    )
    return s

def prepare_formula_for_render(expr: str, *, display: bool = False) -> str:
    """Ported upstream retainpdf_math formula preparation before passing to MiTeX."""
    if not expr:
        return expr
    s = strip_control_chars(expr)
    s = _SQRT_ARG_RE.sub(lambda m: rf"\sqrt{{{m.group(1) or m.group(2) or m.group(3)}}}", s)
    s = s.replace("″", "''").replace("′", "'")
    s = collapse_doubled_command_backslashes(s)
    s = _compact_ocr_math_whitespace(s)
    s = separate_double_scripts(s)
    s = balance_left_right(s)
    s = re.sub(r"\\cal\s+([A-Za-z])", r"\\mathcal{\1}", s)
    s = re.sub(r"\\mathscr\b", r"\\mathcal", s)
    s = re.sub(r"\\rrangle\b", r"\\rangle", s)
    s = re.sub(r"\\llangle\b", r"\\langle", s)
    s = re.sub(r"\\langlen\b", r"\\langle n", s)
    s = re.sub(r"\\angle(?=[A-Za-z])", r"\\angle ", s)
    s = re.sub(r"\\textsuperscript\s*\{\s*([^{}]+?)\s*\}", r"^{\1}", s)
    s = re.sub(r"\\(?: textcircled|circled)\s*\{\s*\\times\s*\}", r"\\otimes", s)
    s = re.sub(r"\\(?:textcircled|circled)\s*\{\s*\\parallel\s*\}", r"\\circ", s)
    s = re.sub(r"\\(?:textcircled|circled)\s*\{\s*([^{}]+?)\s*\}", r"\1", s)
    s = re.sub(r"\\(?:scriptstyle|scriptscriptstyle|textstyle|displaystyle)\b\s*", "", s)
    s = normalize_angstrom_in_math(s)
    return s


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def close_trailing_inline_math(text: str) -> str:
    """Close a truncated trailing inline math $... at the end of a block if it contains LaTeX."""
    source = str(text or "")
    singles = [
        index
        for index, char in enumerate(source)
        if char == "$"
        and not _is_escaped(source, index)
        and source[index - 1 : index] != "$"
        and source[index + 1 : index + 2] != "$"
    ]
    if len(singles) % 2 == 0:
        return source
    tail = source[singles[-1] + 1 :]
    if (
        not tail.strip()
        or len(tail) > MAX_INLINE_MATH_CHARS
        or "\n" in tail
        or any(is_cjk_char(char) for char in tail)
        or not _LATEX_MARKER_RE.search(tail)
    ):
        return source
    return source.rstrip() + "$"


@dataclass(frozen=True)
class FormulaSafetyInsets:
    top_pt: float = 0.0
    bottom_pt: float = 0.0

    @property
    def total_pt(self) -> float:
        return self.top_pt + self.bottom_pt

    @property
    def active(self) -> bool:
        return self.total_pt > 0.01


def extract_formula_bodies(text: str) -> list[str]:
    """Return list of math expression bodies inside $...$, $$...$$, \\(...\\), \\[...\\]."""
    if not text:
        return []
    bodies: list[str] = []
    for m in re.finditer(
        r"\$\$([^$]+)\$\$|\\\[([\s\S]+?)\\\]|\\\(([^\n]+?)\\\)|(?<!\\)\$([^$\n]+?)\$",
        text,
    ):
        body = (m.group(1) or m.group(2) or m.group(3) or m.group(4) or "").strip()
        if body:
            bodies.append(body)
    return bodies


def formula_needs_extra_descent(formula_text: str) -> bool:
    return bool(_SCRIPT_OR_TALL_MATH_RE.search(str(formula_text or "")))


def block_has_deep_formulas(text: str) -> bool:
    return any(formula_needs_extra_descent(f) for f in extract_formula_bodies(text))


def formula_safety_insets_pt(
    text: str,
    *,
    font_size_pt: float,
    box_height_pt: float,
) -> FormulaSafetyInsets:
    if font_size_pt <= 0 or box_height_pt <= MIN_SAFE_CONTENT_HEIGHT_PT:
        return FormulaSafetyInsets()
    formulas = extract_formula_bodies(text)
    if not formulas:
        return FormulaSafetyInsets()
    has_deep = any(formula_needs_extra_descent(f) for f in formulas)
    top_ratio = 0.07 if has_deep else 0.045
    bottom_ratio = 0.18 if has_deep else 0.11
    top = min(max(font_size_pt * top_ratio, 0.25), 1.15)
    bottom = min(max(font_size_pt * bottom_ratio, 0.55), 2.6)
    available = max(0.0, box_height_pt - MIN_SAFE_CONTENT_HEIGHT_PT)
    cap = min(box_height_pt * MAX_FORMULA_INSET_HEIGHT_RATIO, available)
    total = top + bottom
    if cap <= 0 or total <= 0:
        return FormulaSafetyInsets()
    if total > cap:
        scale = cap / total
        top *= scale
        bottom *= scale
    return FormulaSafetyInsets(top_pt=round(top, 2), bottom_pt=round(bottom, 2))


def _formula_visible_units(formula_text: str) -> float:
    compact = re.sub(r"\\[A-Za-z]+|[{}\s]", "", formula_text or "")
    if not compact:
        return 0.0
    command_bonus = len(_LATEX_COMMAND_RE.findall(formula_text or "")) * 0.45
    script_bonus = len(re.findall(r"[_^]", formula_text or "")) * 0.15
    op_bonus = len(re.findall(r"[=+\-−/<>≤≥∈∉⊆⊂×·∼]", formula_text or "")) * 0.35
    return max(0.0, len(compact) * 0.52 + command_bonus + script_bonus + op_bonus)


def has_long_inline_math_layout_risk(
    text: str,
    *,
    font_size_pt: float,
    box_width_pt: float,
) -> bool:
    """Ported from upstream formula_safety.py: return True when a block contains a wide inline
    formula that could stretch CJK justification gaps if justify: true is left enabled."""
    if font_size_pt <= 0 or box_width_pt <= MIN_SAFE_CONTENT_HEIGHT_PT:
        return False
    for formula in extract_formula_bodies(text):
        f = formula.strip()
        if len(f) < LONG_INLINE_MATH_MIN_BODY_CHARS:
            continue
        vis = _formula_visible_units(f)
        est_w = vis * font_size_pt * FORMULA_WIDTH_UNIT_PT_RATIO
        if est_w >= box_width_pt * LONG_INLINE_MATH_WIDTH_RATIO:
            return True
    return False


def _is_usable_fragment(
    rect: pymupdf.Rect,
    *,
    min_width_pt: float = MIN_CLEANUP_SEGMENT_WIDTH_PT,
    min_height_pt: float = MIN_CLEANUP_SEGMENT_HEIGHT_PT,
    min_area_pt2: float = MIN_CLEANUP_SEGMENT_AREA_PT2,
) -> bool:
    return (
        not rect.is_empty
        and rect.width >= min_width_pt
        and rect.height >= min_height_pt
        and rect.width * rect.height >= min_area_pt2
    )


def subtract_guard_from_rect(
    rect: pymupdf.Rect,
    guard: pymupdf.Rect,
    *,
    min_width_pt: float = MIN_CLEANUP_SEGMENT_WIDTH_PT,
    min_height_pt: float = MIN_CLEANUP_SEGMENT_HEIGHT_PT,
    min_area_pt2: float = MIN_CLEANUP_SEGMENT_AREA_PT2,
) -> list[pymupdf.Rect]:
    overlap = pymupdf.Rect(rect) & pymupdf.Rect(guard)
    if overlap.is_empty or overlap.width <= 0 or overlap.height <= 0:
        return [pymupdf.Rect(rect)]
    candidates = [
        pymupdf.Rect(rect.x0, rect.y0, rect.x1, overlap.y0),
        pymupdf.Rect(rect.x0, overlap.y1, rect.x1, rect.y1),
        pymupdf.Rect(rect.x0, overlap.y0, overlap.x0, overlap.y1),
        pymupdf.Rect(overlap.x1, overlap.y0, rect.x1, overlap.y1),
    ]
    return [
        frag
        for frag in candidates
        if _is_usable_fragment(
            frag,
            min_width_pt=min_width_pt,
            min_height_pt=min_height_pt,
            min_area_pt2=min_area_pt2,
        )
    ]


def split_rect_away_from_formula_guards(
    rect: pymupdf.Rect,
    guards: list[pymupdf.Rect],
    *,
    min_width_pt: float = MIN_CLEANUP_SEGMENT_WIDTH_PT,
    min_height_pt: float = MIN_CLEANUP_SEGMENT_HEIGHT_PT,
    min_area_pt2: float = MIN_CLEANUP_SEGMENT_AREA_PT2,
) -> list[pymupdf.Rect]:
    """Ported from upstream formula_guard.py / segments.py: split a text redaction rectangle
    around preserved display formula bounding boxes so display equations are never clipped."""
    if rect.is_empty:
        return []
    fragments = [pymupdf.Rect(rect)]
    for guard in guards:
        if guard.is_empty:
            continue
        next_fragments: list[pymupdf.Rect] = []
        for fragment in fragments:
            next_fragments.extend(
                subtract_guard_from_rect(
                    fragment,
                    guard,
                    min_width_pt=min_width_pt,
                    min_height_pt=min_height_pt,
                    min_area_pt2=min_area_pt2,
                )
            )
        fragments = next_fragments
        if not fragments:
            break
    return fragments
