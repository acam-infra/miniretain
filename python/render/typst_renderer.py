"""Typst renderer for Miniretain: supports layout-preserving per-page overlay
(RetainPDF-aligned) with Typst-native measure() + compiled-span fit verification
and A4 streaming fallback.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pymupdf

from render.layout import find_cjk_font, pdf_rect
from render.side_by_side import build_side_by_side_pdf, merge_interleaved

_URL_RE = re.compile(r"https?://[^\s)\]>]+")
_ASCII_RUN_RE = re.compile(r"[A-Za-z0-9_.,:;!?@#$%^&*()+=\-/<>\\\[\]{}|~`'\"]+")
_MITEX_IMPORT_LINE = '#import "@preview/mitex:0.2.7": mi, mitex'


def find_typst_package_cache_dir() -> str | None:
    """Return Typst package cache directory containing preview/mitex/0.2.7 if available."""
    candidates: list[Path] = []
    env_cache = os.environ.get("TYPST_PACKAGE_CACHE_PATH", "").strip()
    if env_cache:
        candidates.append(Path(env_cache))
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        candidates.append(Path(xdg_cache) / "typst" / "packages")
    candidates.extend([
        Path.home() / ".cache" / "typst" / "packages",
        Path("/root/.cache/typst/packages"),
        Path("/home/ccds/.cache/typst/packages"),
    ])
    seen: set[str] = set()
    for base in candidates:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        if (base / "preview" / "mitex" / "0.2.7" / "typst.toml").is_file():
            return key
    return None


def _typst_pkg_args(pkg_cache_dir: str | None) -> list[str]:
    return ["--package-cache-path", pkg_cache_dir] if pkg_cache_dir else []


def find_typst() -> str:
    """Find a usable typst binary (respecting TYPST_BIN override first)."""
    env_bin = os.environ.get("TYPST_BIN", "").strip()
    if env_bin:
        resolved = shutil.which(env_bin)
        if resolved:
            return resolved
        if os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
            return env_bin
        raise RuntimeError(f"TYPST_BIN={env_bin!r} not found or not executable.")
    candidate = shutil.which("typst")
    if candidate:
        return candidate
    for p in [
        "/usr/local/bin/typst",
        "/usr/bin/typst",
        "/home/ccds/.local/bin/typst",
        "/home/jiangminuo/.local/bin/typst",
        os.path.expanduser("~/.local/bin/typst"),
        os.path.expanduser("~/bin/typst"),
    ]:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise RuntimeError("typst binary not found. Please install typst or specify path.")


def escape_typst(text: str) -> str:
    """Escape text for safe literal embedding inside Typst markup."""
    if not text:
        return ""
    s = text.replace("\\", "\\\\")
    for ch in ["#", "$", "@", "[", "]", "*", "_", "`", "<", ">"]:
        s = s.replace(ch, "\\" + ch)
    s = s.replace("//", "\\/\\/").replace("/*", "\\/\\*")
    return s


def typst_str_literal(text: str) -> str:
    """Escape text as a Typst quoted string literal "..." (immune to all markup/comments)."""
    if not text:
        return '""'
    s = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _breakable_url_display(url: str) -> str:
    """Insert zero-width spaces after URL separators so Typst can wrap long URLs."""
    chunks = re.split(r"([/?&=])", url)
    out: list[str] = []
    i = 0
    while i < len(chunks):
        part = chunks[i]
        if i + 1 < len(chunks) and chunks[i + 1] in "/?&= ":
            part += chunks[i + 1]
            i += 2
        else:
            i += 1
        if part:
            out.append(part)
    if len(out) <= 1:
        return f"#{typst_str_literal(url)}"
    return "#sym.zws ".join(f"#{typst_str_literal(p)}" for p in out)


def _typst_raw_literal(expr: str) -> str:
    """Wrap raw LaTeX in Typst backtick raw string without backslash/quote escaping."""
    max_bt = max((len(m.group(0)) for m in re.finditer(r"`+", expr)), default=0)
    fence = "`" if max_bt == 0 else "`" * max(3, max_bt + 1)
    if max_bt > 0 and (expr.startswith("`") or expr.endswith("`")):
        return f"{fence} {expr} {fence}"
    return f"{fence}{expr}{fence}"


def _looks_like_currency_span(inner: str) -> bool:
    """Return True if $...$ content looks like currency (e.g. '$50 ... $100') rather than math."""
    if not inner or inner != inner.strip():
        return True
    if re.match(r"^\d[\d,.\s]*(?:[-–—]|to|and|or)?$", inner, re.IGNORECASE):
        return True
    if re.match(r"^\d", inner) and not any(ch in inner for ch in r"\^_{}=+*/<>()[]"):
        return True
    return False


def _split_math_segments(text: str) -> list[tuple[str, str]]:
    """Split non-URL text into ('text'|'inline_math'|'display_math', content) segments."""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    i = 0
    n = len(text)

    def flush_buf() -> None:
        if buf:
            out.append(("text", "".join(buf)))
            buf.clear()

    while i < n:
        ch = text[i]
        # Escaped dollar \$ is literal text
        if ch == "\\" and i + 1 < n and text[i + 1] == "$":
            buf.append("$")
            i += 2
            continue
        # Display math $$...$$
        if text.startswith("$$", i):
            end = text.find("$$", i + 2)
            if end != -1:
                inner = text[i + 2 : end].strip()
                if inner:
                    flush_buf()
                    out.append(("display_math", inner))
                    i = end + 2
                    continue
        # Display math \[...\]
        if text.startswith(r"\[", i):
            end = text.find(r"\]", i + 2)
            if end != -1:
                inner = text[i + 2 : end].strip()
                if inner:
                    flush_buf()
                    out.append(("display_math", inner))
                    i = end + 2
                    continue
        # Inline math \(...\)
        if text.startswith(r"\(", i):
            end = text.find(r"\)", i + 2)
            if end != -1 and "\n" not in text[i + 2 : end]:
                inner = text[i + 2 : end].strip()
                if inner:
                    flush_buf()
                    out.append(("inline_math", inner))
                    i = end + 2
                    continue
        # Inline math $...$
        if ch == "$" and not text.startswith("$$", i):
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
            if end != -1:
                inner = text[i + 1 : end]
                if not _looks_like_currency_span(inner):
                    flush_buf()
                    out.append(("inline_math", inner))
                    i = end + 1
                    continue
        buf.append(ch)
        i += 1
    flush_buf()
    return out


def _render_non_url_expr(segment: str, *, use_mitex: bool) -> str:
    if not use_mitex or ("$" not in segment and r"\(" not in segment and r"\[" not in segment):
        return f"#{typst_str_literal(segment)}"
    parts: list[str] = []
    for kind, content in _split_math_segments(segment):
        if kind == "inline_math":
            parts.append(f"#mi({_typst_raw_literal(content)})")
        elif kind == "display_math":
            parts.append(f"#mitex({_typst_raw_literal(content)})")
        else:
            parts.append(f"#{typst_str_literal(content)}")
    return "".join(parts) if parts else '""'


def typst_rich_text_expr(text: str, *, use_mitex: bool = True) -> str:
    """Render text with clickable #link(url)[breakable_url] and #mi/#mitex for math."""
    parts: list[str] = []
    pos = 0
    for m in _URL_RE.finditer(text):
        if m.start() > pos:
            parts.append(_render_non_url_expr(text[pos:m.start()], use_mitex=use_mitex))
        url = m.group(0).rstrip(".,;:")
        trail = m.group(0)[len(url):]
        ul = typst_str_literal(url)
        parts.append(f"#link({ul})[{_breakable_url_display(url)}]")
        if trail:
            parts.append(_render_non_url_expr(trail, use_mitex=use_mitex))
        pos = m.end()
    if pos < len(text):
        parts.append(_render_non_url_expr(text[pos:], use_mitex=use_mitex))
    return "".join(parts) if parts else '""'


def _unbreakable_segments(text: str) -> list[str]:
    """Extract unbreakable horizontal segments (ASCII runs, split at URL break points, excluding math)."""
    segs: list[str] = []
    pos = 0
    for m in _URL_RE.finditer(text):
        before = text[pos : m.start()]
        for kind, seg_text in _split_math_segments(before):
            if kind == "text":
                for am in _ASCII_RUN_RE.finditer(seg_text):
                    segs.append(am.group(0))
        url = m.group(0).rstrip(".,;:")
        for part in re.findall(r"[^/?&=]+[/?&=]?", url):
            segs.append(part)
        pos = m.end()
    tail = text[pos:]
    for kind, seg_text in _split_math_segments(tail):
        if kind == "text":
            for am in _ASCII_RUN_RE.finditer(seg_text):
                segs.append(am.group(0))
    # Also keep longest single CJK char as minimum width check
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        segs.append("汉")
    # Deduplicate preserving longest segments first
    uniq = sorted(set(s for s in segs if s), key=len, reverse=True)
    return uniq[:12]


def _typst_version(typst_bin: str) -> str:
    try:
        proc = subprocess.run([typst_bin, "--version"], capture_output=True, text=True, timeout=10)
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _candidate_sizes(start_sz: float, min_size: float) -> list[float]:
    """Generate descending candidate font sizes (0.25pt step above 7.0pt, 0.5pt below, inclusive of min_size)."""
    sizes: list[float] = []
    sz = max(min_size, round(start_sz * 4) / 4)
    while sz > min_size + 1e-6:
        r = round(sz, 2)
        if not sizes or abs(sizes[-1] - r) > 1e-6:
            sizes.append(r)
        step = 0.25 if sz > 7.0 + 1e-6 else 0.5
        sz = max(min_size, round((sz - step) * 4) / 4)
    floor = round(min_size, 2)
    if not sizes or abs(sizes[-1] - floor) > 1e-6:
        sizes.append(floor)
    return sizes


def _compute_start_size(
    hint: float | None,
    rect: pymupdf.Rect,
    min_size: float,
    role: str = "body",
) -> float:
    """Line-aware initial font size.

    Multi-line blocks (paragraphs, captions, references) start closer to the source
    font size (0.98x) so CJK body text retains full readability; single-line boxes
    are capped by box height (0.85x) because CJK em-boxes are taller than Latin glyphs.
    """
    max_cap = 16.0 if role == "title" else 14.0
    if hint and isinstance(hint, (int, float)) and hint > 0:
        h_val = float(hint)
        if rect.height <= h_val * 1.45:
            raw = min(h_val * 0.92, rect.height * 0.85)
        else:
            raw = h_val * 0.98
        return min(max_cap, max(min_size, raw))
    return min(max_cap, max(min_size, rect.height * 0.78))


def measure_blocks_with_typst(
    typst_bin: str,
    font_dir: str,
    font_families: str,
    workdir: Path,
    candidates: list[dict],
    probe_name: str = "fit_probe.typ",
    pkg_cache_dir: str | None = None,
) -> dict[tuple[int, float], tuple[float, float]]:
    """Measure exact Typst (max_token_width_pt, block_height_pt) for all candidates.

    Unbreakable token widths scale linearly with font size, so each candidate's
    segments are measured once at its reference size and scaled for other sizes.
    """
    if not candidates:
        return {}
    needs_mitex = any(
        "#mi(" in c.get("rich_expr", "") or "#mitex(" in c.get("rich_expr", "")
        for c in candidates
    )
    lines: list[str] = []
    if needs_mitex:
        lines.append(_MITEX_IMPORT_LINE)
    lines.extend([
        f'#set text(font: {font_families}, lang: "zh", fill: black, overhang: false)',
        "",
    ])
    ref_sizes: dict[int, float] = {}
    for c in candidates:
        idx = c["idx"]
        sizes = c["sizes"]
        if not sizes:
            continue
        raw_w = max(4.0, c["width"])
        w = max(4.0, raw_w - 4.5) if raw_w > 40.0 else raw_w
        weight = c["weight"]
        rich = c["rich_expr"]
        segs = c["segments"]
        ref_sz = sizes[0]
        ref_sizes[idx] = ref_sz
        if segs:
            seg_exprs = ", ".join(
                f'measure([#set text(size: {ref_sz:.2f}pt, weight: "{weight}"); #{typst_str_literal(s)}]).width / 1pt'
                for s in segs
            )
            lines.append(
                f'#context {{ let tw = calc.max({seg_exprs}); '
                f'[#metadata((kind: "tw", idx: {idx}, sz: {ref_sz:.2f}, tw: tw)) <fit>] }}'
            )
        for sz in sizes:
            leading = "0.35em" if sz < 8.5 else "0.45em"
            lines.append(
                f'#context {{ let m = measure(block(width: {w:.2f}pt)'
                f'[#set text(size: {sz:.2f}pt, weight: "{weight}"); '
                f'#set par(leading: {leading}, justify: true); {rich}]); '
                f'[#metadata((kind: "h", idx: {idx}, sz: {sz:.2f}, h: m.height / 1pt)) <fit>] }}'
            )
    probe_typ = workdir / probe_name
    probe_typ.write_text("\n".join(lines), encoding="utf-8")
    proc = subprocess.run(
        [typst_bin, "query", "--font-path", font_dir, *(_typst_pkg_args(pkg_cache_dir) if needs_mitex else []), str(probe_typ), "<fit>"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"typst measure query failed: {proc.stderr.strip()}")
    rows = json.loads(proc.stdout)
    ref_tw: dict[int, tuple[float, float]] = {}
    heights: dict[tuple[int, float], float] = {}
    for row in rows:
        val = row.get("value") or {}
        kind = val.get("kind")
        idx = int(val["idx"])
        sz = round(float(val["sz"]), 2)
        if kind == "tw":
            ref_tw[idx] = (sz, float(val["tw"]))
        elif kind == "h":
            heights[(idx, sz)] = float(val["h"])
    out: dict[tuple[int, float], tuple[float, float]] = {}
    for (idx, sz), h in heights.items():
        if idx in ref_tw and ref_tw[idx][0] > 0:
            r_sz, r_tw = ref_tw[idx]
            tw = r_tw * (sz / r_sz)
        else:
            tw = 0.0
        out[(idx, sz)] = (tw, h)
    return out


def _overlaps_neighbor(
    placed_box: tuple[float, float, float, float],
    self_idx: int,
    neighbor_boxes: list[tuple[int, tuple[float, float, float, float]]],
    tol_pt: float = 1.5,
    self_orig_box: tuple[float, float, float, float] | None = None,
) -> bool:
    for n_idx, nb in neighbor_boxes:
        if n_idx == self_idx:
            continue
        x0 = max(placed_box[0], nb[0])
        y0 = max(placed_box[1], nb[1])
        x1 = min(placed_box[2], nb[2])
        y1 = min(placed_box[3], nb[3])
        if (x1 - x0) > tol_pt and (y1 - y0) > tol_pt:
            if self_orig_box is not None:
                ox0 = max(self_orig_box[0], nb[0])
                oy0 = max(self_orig_box[1], nb[1])
                ox1 = min(self_orig_box[2], nb[2])
                oy1 = min(self_orig_box[3], nb[3])
                # Ignore pre-existing source bbox touch/overlap if rendered box didn't grow past it
                if (ox1 - ox0) > 0 and (oy1 - oy0) > 0 and (y1 - y0) <= (oy1 - oy0) + tol_pt:
                    continue
            return True
    return False


def _emit_overlay_typ(
    page_meta: list[dict],
    to_place_by_page: dict[int, list[tuple[dict, pymupdf.Rect, float, int]]],
    font_families: str,
    use_mitex: bool = True,
) -> str:
    body_lines: list[str] = []
    needs_mitex = False
    for pm in page_meta:
        page_index = pm["page_index"]
        pw, ph = pm["pw"], pm["ph"]
        if page_index > 0:
            body_lines.append("#pagebreak()")
        body_lines.append(
            f"#set page(width: {pw:.2f}pt, height: {ph:.2f}pt, margin: 0pt, fill: none)"
        )
        for it, rect, chosen_sz, _local_i in to_place_by_page.get(page_index, []):
            is_bold = bool(it.get("is_bold") or (it.get("content") or {}).get("is_bold"))
            role = it.get("layout_role", "body")
            weight = "bold" if (is_bold or role in {"title", "heading"}) else "regular"
            raw_w = max(4.0, float(rect.width))
            w = max(4.0, raw_w - 4.5) if raw_w > 40.0 else raw_w
            h = max(2.0, float(rect.height))
            leading = "0.35em" if chosen_sz < 8.5 else "0.45em"
            rich = typst_rich_text_expr(it["translation"].strip(), use_mitex=use_mitex)
            if "#mi(" in rich or "#mitex(" in rich:
                needs_mitex = True
            body_lines.append(
                f'#place(top + left, dx: {rect.x0:.2f}pt, dy: {rect.y0:.2f}pt, '
                f'block(width: {w:.2f}pt, height: {h:.2f}pt, clip: false)'
                f'[#set text(size: {chosen_sz:.2f}pt, weight: "{weight}"); '
                f'#set par(leading: {leading}, justify: true); {rich}])'
            )
    header: list[str] = []
    if needs_mitex:
        header.append(_MITEX_IMPORT_LINE)
    header.extend([
        '#set page(margin: 0pt, fill: none)',
        f'#set text(font: {font_families}, lang: "zh", fill: black, overhang: false)',
        '#set par(justify: true, leading: 0.45em)',
        '',
    ])
    return "\n".join(header + body_lines)


def _find_compiled_overflow_blocks(
    overlay_doc: pymupdf.Document,
    page_meta: list[dict],
    to_place_by_page: dict[int, list[tuple[dict, pymupdf.Rect, float, int]]],
    tol_pt: float = 1.0,
) -> dict[str, str]:
    """Verify actual rendered glyph spans in compiled overlay.pdf stay inside their block rects."""
    bad: dict[str, str] = {}
    for pm in page_meta:
        page_index = pm["page_index"]
        pw, ph = pm["pw"], pm["ph"]
        placed_list = to_place_by_page.get(page_index, [])
        if not placed_list or page_index >= len(overlay_doc):
            continue
        opage = overlay_doc[page_index]
        # Build spatial index of placed blocks on this page
        for blk in opage.get_text("dict").get("blocks", []):
            for line in blk.get("lines", []):
                line_compact = "".join("".join((s.get("text") or "") for s in line.get("spans", [])).split())
                line_probe = line_compact[:12]
                for span in line.get("spans", []):
                    sx0, sy0, sx1, sy1 = span["bbox"]
                    span_w = max(0.5, sx1 - sx0)
                    span_h = max(0.5, sy1 - sy0)
                    span_area = span_w * span_h
                    # Primary match: pick placed block with highest text + 2D overlap.
                    # Using full line_probe avoids misattributing short accent/symbol spans (e.g. "ˆ")
                    # to an overlapping adjacent block that happens to share the same single glyph.
                    sp_compact = "".join((span.get("text") or "").split())
                    sp_probe = line_probe if len(line_probe) >= 3 else sp_compact[:12]
                    best = None
                    best_score: tuple[int, float, float, float, float] | None = None
                    for it, rect, _sz, local_i in placed_list:
                        tr_compact = "".join((it.get("translation") or "").split())
                        t_match = 1 if (sp_probe and sp_probe in tr_compact) else 0
                        ix = min(sx1, rect.x1) - max(sx0, rect.x0)
                        iy_exact = min(sy1, rect.y1) - max(sy0, rect.y0)
                        if ix > 0.0 and iy_exact > 0.0:
                            x_cov = ix / span_w
                            y_cov = iy_exact / span_h
                            area_cov = (ix * iy_exact) / span_area
                            dist = abs(sx0 - rect.x0) + abs(sy0 - rect.y0)
                            score = (t_match, round(area_cov, 4), round(y_cov, 4), round(x_cov, 4), -dist)
                            if best_score is None or score > best_score:
                                best_score = score
                                best = (it, rect, local_i)
                    if best is None:
                        for it, rect, _sz, local_i in placed_list:
                            tr_compact = "".join((it.get("translation") or "").split())
                            t_match = 1 if (sp_probe and sp_probe in tr_compact) else 0
                            ix = min(sx1, rect.x1) - max(sx0, rect.x0)
                            iy = min(sy1, rect.y1 + 4.0) - max(sy0, rect.y0 - 4.0)
                            if ix > 0.0 and iy > 0.0:
                                x_cov = ix / span_w
                                area_cov = (ix * iy) / span_area
                                dist = abs(sx0 - rect.x0) + abs(sy0 - rect.y0)
                                score = (t_match, round(x_cov, 4), round(area_cov, 4), 0.0, -dist)
                                if best_score is None or score > best_score:
                                    best_score = score
                                    best = (it, rect, local_i)
                    if best is None:
                        # Fallback for spans that escaped all rects: match nearest in same vertical/column band
                        best_dist = 1e9
                        left_pad = max(8.0, span_h)
                        for it, rect, _sz, local_i in placed_list:
                            if (
                                rect.x0 - left_pad <= sx0 <= rect.x1 + 8.0
                                and rect.y0 - 4.0 <= sy0 <= rect.y1 + 4.0
                            ):
                                d = abs(sx0 - rect.x0) + abs(sy0 - rect.y0)
                                if d < best_dist:
                                    best_dist = d
                                    best = (it, rect, local_i)
                    if best is None:
                        continue
                    it, rect, local_i = best
                    bid = it.get("block_id", "unknown")
                    if sx1 > rect.x1 + tol_pt:
                        bad[bid] = "compiled_horizontal_overflow"
                    elif sy1 > rect.y1 + tol_pt:
                        bad[bid] = "compiled_vertical_overflow"
                    elif sx1 > pw + 0.5 or sy1 > ph + 0.5 or sx0 < -0.5 or sy0 < -0.5:
                        bad[bid] = "compiled_out_of_bounds"
                    elif _overlaps_neighbor((sx0, sy0, sx1, sy1), local_i, pm["neighbor_boxes"], self_orig_box=(rect.x0, rect.y0, rect.x1, rect.y1)):
                        bad[bid] = "compiled_neighbor_overlap"
    return bad


def _render_typst_preserve(
    source_pdf: str,
    items: list[dict],
    workdir: Path,
    output_cn: Path,
    output_merged: Path,
    legacy_output: Path | None,
    font: str,
    page_widths: dict[int, float],
    page_coord_spaces: dict[int, str],
    page_scanned: dict[int, bool],
    bilingual_mode: str,
    min_size: float = 5.0,
    strict: bool = False,
) -> dict:
    """RetainPDF-style per-page Typst overlay onto redacted source PDF."""
    t_total_0 = time.perf_counter()
    pkg_cache_dir = find_typst_package_cache_dir()
    use_mitex = bool(pkg_cache_dir)
    typst_bin = find_typst()
    cjk_font_file = find_cjk_font(font)
    font_dir = str(Path(cjk_font_file).parent)
    font_families = '("Noto Serif CJK SC", "Noto Sans CJK SC", "Source Han Serif SC", "Droid Sans Fallback", "Libertinus Serif")'

    doc = pymupdf.open(source_pdf)
    preserved = 0
    unplaced = 0
    failed = 0
    shrunk = 0
    unplaced_block_ids: list[str] = []
    unplaced_reasons: dict[str, str] = {}
    failed_block_ids: list[str] = []

    probe_candidates: list[dict] = []
    page_meta: list[dict] = []
    saved_links_by_page: dict[int, list[dict]] = {}

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            pw = float(page.rect.width)
            ph = float(page.rect.height)
            coord_space = page_coord_spaces.get(page_index, "pdf_pt")
            scale = 1.0 if coord_space == "pdf_pt" else (page_widths.get(page_index, pw) / pw)
            saved_links_by_page[page_index] = [
                {"from": pymupdf.Rect(lk["from"]), "uri": lk["uri"]}
                for lk in page.get_links()
                if lk.get("kind") == pymupdf.LINK_URI and lk.get("uri")
            ]

            page_items = [it for it in items if it["page_index"] == page_index]
            neighbor_boxes: list[tuple[int, tuple[float, float, float, float]]] = []
            for local_i, it in enumerate(page_items):
                if it.get("bbox"):
                    r = pdf_rect(it["bbox"], scale)
                    neighbor_boxes.append((local_i, (r.x0, r.y0, r.x1, r.y1)))

            p_preserved = 0
            p_failed = 0
            p_candidates: list[dict] = []

            for local_i, it in enumerate(page_items):
                bid = it.get("block_id", "unknown")
                policy = it.get("policy") or {}
                if policy.get("translate") is False:
                    preserved += 1
                    p_preserved += 1
                    continue

                t_status = it.get("translation_status", "ok")
                trans = (it.get("translation") or "").strip()
                bbox = it.get("bbox")
                if t_status != "ok" or not trans or not bbox:
                    failed += 1
                    p_failed += 1
                    failed_block_ids.append(bid)
                    continue

                rect = pdf_rect(bbox, scale)
                hint = it.get("font_size_pt") or (it.get("content") or {}).get("font_size_pt")
                is_bold = bool(it.get("is_bold") or (it.get("content") or {}).get("is_bold"))
                role = it.get("layout_role", "body")
                start_sz = _compute_start_size(hint, rect, min_size, role=role)
                weight = "bold" if (is_bold or role in {"title", "heading"}) else "regular"
                cand = {
                    "idx": len(probe_candidates),
                    "local_i": local_i,
                    "item": it,
                    "rect": rect,
                    "width": max(4.0, float(rect.width)),
                    "height": max(2.0, float(rect.height)),
                    "weight": weight,
                    "rich_expr": typst_rich_text_expr(trans, use_mitex=use_mitex),
                    "segments": _unbreakable_segments(trans),
                    "sizes": _candidate_sizes(start_sz, min_size),
                }
                probe_candidates.append(cand)
                p_candidates.append(cand)

            page_meta.append({
                "page_index": page_index,
                "pw": pw,
                "ph": ph,
                "scale": scale,
                "is_scanned": bool(page_scanned.get(page_index, False)),
                "total_blocks": len(page_items),
                "neighbor_boxes": neighbor_boxes,
                "p_preserved": p_preserved,
                "p_failed": p_failed,
                "candidates": p_candidates,
            })

        # Step 2: Two-tier Typst fit measurement.
        # Pass 1 probes only the top size (and one fallback size) per block; Pass 2
        # probes remaining sizes ONLY for blocks that did not fit in Pass 1.
        t_probe_0 = time.perf_counter()
        pass1_candidates: list[dict] = []
        for c in probe_candidates:
            szs = c["sizes"]
            p1_szs = [szs[0]] if len(szs) <= 1 else [szs[0], szs[min(2, len(szs) - 1)]]
            pass1_candidates.append({**c, "sizes": p1_szs})
        measured = measure_blocks_with_typst(
            typst_bin, font_dir, font_families, workdir, pass1_candidates, probe_name="fit_probe.typ", pkg_cache_dir=pkg_cache_dir
        )

        # Check which candidates need Pass 2 (remaining sizes)
        pass2_candidates: list[dict] = []
        for pm in page_meta:
            pw, ph = pm["pw"], pm["ph"]
            for cand in pm["candidates"]:
                rect = cand["rect"]
                szs = cand["sizes"]
                if len(szs) <= 1:
                    continue
                p1_top = szs[0]
                tw0, mh0 = measured.get((cand["idx"], round(p1_top, 2)), (1e9, 1e9))
                fits_top = (
                    tw0 <= rect.width + 0.5
                    and mh0 <= rect.height + 0.5
                    and rect.x0 >= -0.5
                    and rect.y0 >= -0.5
                    and rect.x0 + rect.width <= pw + 0.5
                    and rect.y0 + mh0 <= ph + 0.5
                    and not _overlaps_neighbor(
                        (rect.x0, rect.y0, rect.x0 + rect.width, rect.y0 + mh0),
                        cand["local_i"],
                        pm["neighbor_boxes"],
                        self_orig_box=(rect.x0, rect.y0, rect.x1, rect.y1),
                    )
                )
                if not fits_top:
                    rem_szs = [s for s in szs if (cand["idx"], round(s, 2)) not in measured]
                    if rem_szs:
                        pass2_candidates.append({**cand, "sizes": rem_szs})
        if pass2_candidates:
            measured2 = measure_blocks_with_typst(
                typst_bin, font_dir, font_families, workdir, pass2_candidates, probe_name="fit_probe_p2.typ", pkg_cache_dir=pkg_cache_dir
            )
            measured.update(measured2)
        fit_probe_s = round(time.perf_counter() - t_probe_0, 3)

        # Step 3: Select largest size >= min_size passing token width, height, bounds, non-overlap
        to_place_by_page: dict[int, list[tuple[dict, pymupdf.Rect, float, int]]] = {}
        for pm in page_meta:
            page_index = pm["page_index"]
            pw, ph = pm["pw"], pm["ph"]
            to_render: list[tuple[dict, pymupdf.Rect, float, int]] = []

            for cand in pm["candidates"]:
                it = cand["item"]
                bid = it.get("block_id", "unknown")
                rect = cand["rect"]
                chosen_sz = None
                fail_reason = "overflow"

                for step_i, sz in enumerate(cand["sizes"]):
                    tw, mh = measured.get((cand["idx"], round(sz, 2)), (1e9, 1e9))
                    if tw > rect.width + 0.5:
                        fail_reason = "horizontal_overflow"
                        continue
                    if mh > rect.height + 0.5:
                        fail_reason = "vertical_overflow"
                        continue
                    placed_box = (rect.x0, rect.y0, rect.x0 + rect.width, rect.y0 + mh)
                    if (
                        placed_box[0] < -0.5
                        or placed_box[1] < -0.5
                        or placed_box[2] > pw + 0.5
                        or placed_box[3] > ph + 0.5
                    ):
                        fail_reason = "out_of_bounds"
                        continue
                    if _overlaps_neighbor(
                        placed_box,
                        cand["local_i"],
                        pm["neighbor_boxes"],
                        self_orig_box=(rect.x0, rect.y0, rect.x1, rect.y1),
                    ):
                        fail_reason = "neighbor_overlap"
                        continue
                    chosen_sz = sz
                    shrunk += step_i
                    break

                if chosen_sz is None:
                    unplaced += 1
                    unplaced_block_ids.append(bid)
                    unplaced_reasons[bid] = fail_reason
                    continue

                to_render.append((it, rect, chosen_sz, cand["local_i"]))

            to_place_by_page[page_index] = to_render

        # Step 4: Compile overlay.pdf BEFORE redacting source PDF and verify actual rendered glyph spans!
        t_overlay_0 = time.perf_counter()
        overlay_typ = workdir / "overlay.typ"
        overlay_pdf = workdir / "overlay.pdf"
        overlay_typ.write_text(
            _emit_overlay_typ(page_meta, to_place_by_page, font_families, use_mitex=use_mitex),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [typst_bin, "compile", "--font-path", font_dir, *_typst_pkg_args(pkg_cache_dir), str(overlay_typ), str(overlay_pdf)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"typst overlay compile failed: {proc.stderr.strip()}")

        with pymupdf.open(overlay_pdf) as check_doc:
            compiled_bad = _find_compiled_overflow_blocks(check_doc, page_meta, to_place_by_page)

        # Retry any compiled-overflow blocks step-by-step down to min_size
        for _retry_step in range(24):
            if not compiled_bad:
                break
            changed = False
            for page_index, plist in list(to_place_by_page.items()):
                updated = []
                for it, rect, csz, li in plist:
                    bid = it.get("block_id", "unknown")
                    if bid in compiled_bad and csz > min_size + 1e-6:
                        step = 0.25 if csz > 7.0 + 1e-6 else 0.5
                        new_sz = max(min_size, round(csz - step, 2))
                        shrunk += 1
                        changed = True
                        updated.append((it, rect, new_sz, li))
                    else:
                        updated.append((it, rect, csz, li))
                to_place_by_page[page_index] = updated
            if not changed:
                break
            overlay_typ.write_text(
                _emit_overlay_typ(page_meta, to_place_by_page, font_families, use_mitex=use_mitex),
                encoding="utf-8",
            )
            proc_retry = subprocess.run(
                [typst_bin, "compile", "--font-path", font_dir, *_typst_pkg_args(pkg_cache_dir), str(overlay_typ), str(overlay_pdf)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc_retry.returncode != 0:
                raise RuntimeError(f"typst overlay retry compile failed: {proc_retry.stderr.strip()}")
            with pymupdf.open(overlay_pdf) as check_doc2:
                compiled_bad = _find_compiled_overflow_blocks(check_doc2, page_meta, to_place_by_page)

        if compiled_bad:
            for page_index, plist in list(to_place_by_page.items()):
                kept = []
                for tup in plist:
                    bid = tup[0].get("block_id", "unknown")
                    if bid in compiled_bad:
                        unplaced += 1
                        unplaced_block_ids.append(bid)
                        unplaced_reasons[bid] = compiled_bad[bid]
                    else:
                        kept.append(tup)
                to_place_by_page[page_index] = kept
            # Recompile clean overlay.pdf without overflowing blocks
            overlay_typ.write_text(
                _emit_overlay_typ(page_meta, to_place_by_page, font_families, use_mitex=use_mitex),
                encoding="utf-8",
            )
            proc2 = subprocess.run(
                [typst_bin, "compile", "--font-path", font_dir, *_typst_pkg_args(pkg_cache_dir), str(overlay_typ), str(overlay_pdf)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc2.returncode != 0:
                raise RuntimeError(f"typst overlay recompile failed: {proc2.stderr.strip()}")

        if strict and (unplaced_block_ids or failed_block_ids):
            raise RuntimeError(
                f"render strict mode failed: unplaced={unplaced_block_ids} "
                f"(reasons={unplaced_reasons}), failed={failed_block_ids}"
            )

        overlay_compile_s = round(time.perf_counter() - t_overlay_0, 3)

        # Step 5: Now redact ONLY verified fitted blocks on source PDF, composite overlay, and count `placed`
        t_comp_0 = time.perf_counter()
        placed = 0
        per_page: list[dict] = []
        with pymupdf.open(overlay_pdf) as overlay_doc:
            if len(overlay_doc) != len(doc):
                raise RuntimeError(
                    f"overlay page count mismatch: source={len(doc)}, overlay={len(overlay_doc)}"
                )
            for pm in page_meta:
                page_index = pm["page_index"]
                page = doc[page_index]
                scale = pm["scale"]
                to_render = to_place_by_page.get(page_index, [])
                redact_fill = (1, 1, 1) if pm["is_scanned"] else None
                for it, _rect, _sz, _li in to_render:
                    for lb in it.get("line_bboxes") or [it["bbox"]]:
                        page.add_redact_annot(pdf_rect(lb, scale), fill=redact_fill)
                if to_render:
                    page.apply_redactions(images=0, graphics=0, text=0)
                    page.show_pdf_page(page.rect, overlay_doc, page_index, overlay=True)
                    placed += len(to_render)

                # Restore clickable URI links on rendered page (handling wrapped URLs)
                existing_uris = {
                    (round(lk["from"].x0, 1), round(lk["from"].y0, 1), lk.get("uri"))
                    for lk in page.get_links()
                    if lk.get("uri")
                }
                words = page.get_text("words")
                for orig_lk in saved_links_by_page.get(page_index, []):
                    uri = orig_lk["uri"]
                    found_rects = page.search_for(uri)
                    if not found_rects:
                        domain = uri.split("://", 1)[-1].split("/", 1)[0]
                        found_rects = [
                            pymupdf.Rect(w[0], w[1], w[2], w[3])
                            for w in words
                            if domain in w[4].replace("\u200b", "")
                            or (
                                len(w[4].replace("\u200b", "")) >= 6
                                and w[4].replace("\u200b", "") in uri
                                and ("/" in w[4] or "=" in w[4] or "?" in w[4])
                            )
                        ]
                    target_rects = found_rects if found_rects else [orig_lk["from"]]
                    for tr in target_rects:
                        sig = (round(tr.x0, 1), round(tr.y0, 1), uri)
                        if sig not in existing_uris:
                            page.insert_link({"kind": pymupdf.LINK_URI, "from": tr, "uri": uri})
                            existing_uris.add(sig)

                p_unplaced = sum(
                    1 for c in pm["candidates"] if c["item"].get("block_id") in unplaced_reasons
                )
                per_page.append({
                    "page_index": page_index,
                    "total_blocks": pm["total_blocks"],
                    "placed": len(to_render),
                    "preserved": pm["p_preserved"],
                    "unplaced": p_unplaced,
                    "failed": pm["p_failed"],
                    "width_pt": round(pm["pw"], 2),
                    "height_pt": round(pm["ph"], 2),
                })

        output_cn.parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_cn, garbage=2, deflate=True)
        redact_composite_s = round(time.perf_counter() - t_comp_0, 3)
    finally:
        doc.close()

    if not output_cn.is_file() or output_cn.stat().st_size == 0:
        raise RuntimeError(f"output_cn missing or empty: {output_cn}")

    if legacy_output and str(legacy_output) != str(output_cn):
        shutil.copyfile(output_cn, legacy_output)

    t_bi_0 = time.perf_counter()
    if bilingual_mode == "interleaved":
        bilingual_info = merge_interleaved(source_pdf, output_cn, output_merged)
    else:
        bilingual_info = build_side_by_side_pdf(source_pdf, output_cn, output_merged)
    bilingual_s = round(time.perf_counter() - t_bi_0, 3)
    total_s = round(time.perf_counter() - t_total_0, 3)

    chosen_sizes = [
        round(csz, 2)
        for plist in to_place_by_page.values()
        for _it, _rect, csz, _li in plist
    ]
    start_by_id = {
        c["item"].get("block_id", "unknown"): c["sizes"][0]
        for c in probe_candidates
        if c["sizes"]
    }
    shrunk_blocks = sum(
        1
        for plist in to_place_by_page.values()
        for it, _rect, csz, _li in plist
        if csz < start_by_id.get(it.get("block_id", "unknown"), csz) - 1e-6
    )
    size_hist: dict[str, int] = {}
    for s in sorted(chosen_sizes):
        k = f"{s:.2f}".rstrip("0").rstrip(".")
        size_hist[k] = size_hist.get(k, 0) + 1
    if chosen_sizes:
        s_sorted = sorted(chosen_sizes)
        font_size_stats = {
            "min_size_floor_pt": round(min_size, 2),
            "min_pt": s_sorted[0],
            "p10_pt": s_sorted[int(len(s_sorted) * 0.1)],
            "median_pt": round(s_sorted[len(s_sorted) // 2], 2),
            "mean_pt": round(sum(s_sorted) / len(s_sorted), 2),
            "max_pt": s_sorted[-1],
            "shrunk_blocks": shrunk_blocks,
            "shrunk_steps": shrunk,
            "histogram": size_hist,
        }
    else:
        font_size_stats = {
            "min_size_floor_pt": round(min_size, 2),
            "shrunk_blocks": 0,
            "shrunk_steps": shrunk,
            "histogram": {},
        }

    return {
        "engine": "typst",
        "layout_mode": "preserve",
        "typst_version": _typst_version(typst_bin),
        "font_file": cjk_font_file,
        "placed": placed,
        "preserved": preserved,
        "unplaced": unplaced,
        "failed": failed,
        "unplaced_block_ids": unplaced_block_ids,
        "unplaced_reasons": unplaced_reasons,
        "failed_block_ids": failed_block_ids,
        "shrunk_steps": shrunk,
        "font_size_stats": font_size_stats,
        "per_page": per_page,
        "output_cn": str(output_cn),
        "output_merged": str(output_merged),
        "output": str(legacy_output or output_cn),
        "bilingual": bilingual_info,
        "timings_s": {
            "fit_probe_s": fit_probe_s,
            "overlay_compile_s": overlay_compile_s,
            "redact_composite_s": redact_composite_s,
            "bilingual_s": bilingual_s,
            "total_s": total_s,
        },
    }


def _render_typst_flow(
    items: list[dict],
    workdir: Path,
    output_cn: Path,
    output_merged: Path,
    legacy_output: Path | None,
    font: str,
) -> dict:
    """A4 streaming Typst fallback (used when source has no PDF/bbox geometry)."""
    typst_bin = find_typst()
    font_list = '("Noto Serif CJK SC", "Noto Sans CJK SC", "Libertinus Serif", "Droid Sans Fallback", "Source Han Serif SC", "SimSun")'
    if font:
        font_list = f'("{font}", "Noto Serif CJK SC", "Libertinus Serif", "Droid Sans Fallback")'

    cn_typ = workdir / "document_cn.typ"
    cn_lines = [
        '#set page(',
        '  paper: "a4",',
        '  margin: (x: 2cm, y: 2.2cm),',
        '  header: align(right)[#text(size: 8pt, fill: luma(120))[RetainPDF 中文译本 (Chinese Edition)]],',
        '  footer: context align(center)[#counter(page).display("1 / 1", both: true)]',
        ')',
        f'#set text(font: {font_list}, size: 10.5pt, lang: "zh")',
        '#set par(justify: true, leading: 0.85em, first-line-indent: 2em)',
        '',
    ]

    placed = 0
    failed = 0
    failed_block_ids: list[str] = []
    for i, it in enumerate(items):
        if it.get("translation_status", "ok") != "ok":
            failed += 1
            failed_block_ids.append(it.get("block_id", f"idx_{i}"))
            continue
        trans = (it.get("translation") or "").strip()
        if not trans:
            continue
        escaped = escape_typst(trans)
        if i == 0 and len(trans) < 80 and not trans.endswith(("。", "！", "？", ".")):
            cn_lines.append(f'= {escaped}\n')
        else:
            cn_lines.append(f'#par[{escaped}]\n')
        placed += 1

    cn_typ.write_text("\n".join(cn_lines), encoding="utf-8")
    proc_cn = subprocess.run([typst_bin, "compile", str(cn_typ), str(output_cn)], capture_output=True, text=True)
    if proc_cn.returncode != 0:
        raise RuntimeError(f"typst compile cn failed: {proc_cn.stderr.strip()}")

    bi_typ = workdir / "document_bilingual.typ"
    bi_lines = [
        '#set page(',
        '  paper: "a4",',
        '  margin: (x: 1.5cm, y: 1.8cm),',
        '  header: align(right)[#text(size: 8pt, fill: luma(120))[RetainPDF 双语对照合并版 (Bilingual Merged)]],',
        '  footer: context align(center)[#counter(page).display("1 / 1", both: true)]',
        ')',
        f'#set text(font: {font_list}, size: 9.5pt)',
        '#table(',
        '  columns: (1fr, 1fr),',
        '  stroke: (x, y) => if y == 0 { (bottom: 1.5pt + black) } else { (bottom: 0.5pt + luma(210)) },',
        '  fill: (x, y) => if y == 0 { luma(245) } else if calc.even(y) { luma(252) } else { none },',
        '  inset: 8pt,',
        '  table.header([*Original (原文)*], [*Translation (译文)*]),',
    ]
    for it in items:
        orig = (it.get("text") or "").strip()
        trans = (it.get("translation") or "").strip()
        if not orig and not trans:
            continue
        bi_lines.append(f'  [{escape_typst(orig)}], [{escape_typst(trans)}],')
    bi_lines.append(')')
    bi_typ.write_text("\n".join(bi_lines), encoding="utf-8")
    proc_bi = subprocess.run([typst_bin, "compile", str(bi_typ), str(output_merged)], capture_output=True, text=True)
    if proc_bi.returncode != 0:
        raise RuntimeError(f"typst compile bilingual failed: {proc_bi.stderr.strip()}")

    if legacy_output and str(legacy_output) != str(output_cn):
        shutil.copyfile(output_cn, legacy_output)

    return {
        "engine": "typst",
        "layout_mode": "flow",
        "typst_version": _typst_version(typst_bin),
        "items": len(items),
        "placed": placed,
        "failed": failed,
        "failed_block_ids": failed_block_ids,
        "output_cn": str(output_cn),
        "output_merged": str(output_merged),
        "output": str(legacy_output or output_cn),
    }


def render_typst(
    items: list[dict],
    workdir: Path,
    output_cn: Path,
    output_merged: Path,
    legacy_output: Path | None = None,
    font: str = "",
    source: str = "",
    layout_mode: str = "auto",
    bilingual_mode: str = "auto",
    page_widths: dict[int, float] | None = None,
    page_coord_spaces: dict[int, str] | None = None,
    page_scanned: dict[int, bool] | None = None,
    strict: bool = False,
) -> dict:
    """Compile items into Chinese and bilingual PDFs using Typst."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    has_pdf = bool(source) and source.lower().endswith(".pdf") and Path(source).is_file()
    has_bbox = any(it.get("bbox") for it in items)

    resolved_layout = layout_mode
    if resolved_layout == "auto":
        resolved_layout = "preserve" if (has_pdf and has_bbox) else "flow"

    if resolved_layout == "preserve":
        if not has_pdf:
            raise RuntimeError("layout_mode='preserve' requires a PDF source file")
        resolved_bi = "side_by_side" if bilingual_mode in ("auto", "table") else bilingual_mode
        return _render_typst_preserve(
            source_pdf=source,
            items=items,
            workdir=workdir,
            output_cn=output_cn,
            output_merged=output_merged,
            legacy_output=legacy_output,
            font=font,
            page_widths=page_widths or {},
            page_coord_spaces=page_coord_spaces or {},
            page_scanned=page_scanned or {},
            bilingual_mode=resolved_bi,
            strict=strict,
        )

    return _render_typst_flow(
        items=items,
        workdir=workdir,
        output_cn=output_cn,
        output_merged=output_merged,
        legacy_output=legacy_output,
        font=font,
    )
