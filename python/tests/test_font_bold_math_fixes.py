"""Regression tests for font selection, Heiti bold contrast, bold detection, and inline math fixes."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.openai_client import _extract_json_object
from ocr.pymupdf_adapter import (
    _build_block_from_lines,
    _is_math_split_boundary,
    _is_span_bold,
    _render_line_with_evidenced_math,
)
from render.typst_renderer import (
    _build_font_families,
    _sanitize_latex_for_mitex,
    render_typst,
    typst_rich_text_expr,
)


def test_build_font_families_respects_config_and_uses_heiti_for_bold():
    body_ff, bold_ff = _build_font_families("", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
    assert '"Noto Serif CJK SC"' in body_ff
    assert body_ff.index('"Noto Serif CJK SC"') < body_ff.index('"Noto Sans CJK SC"')
    # Default bold switches CJK to Heiti (Noto Sans CJK SC) for crisp contrast
    assert bold_ff.index('"Noto Sans CJK SC"') < bold_ff.index('"Noto Serif CJK SC"')

    # Explicit family name is respected consistently in both body and bold
    body_custom, bold_custom = _build_font_families("Source Han Sans SC", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
    assert body_custom.index('"Source Han Sans SC"') < body_custom.index('"Noto Serif CJK SC"')
    assert bold_custom.index('"Source Han Sans SC"') < bold_custom.index('"Noto Sans CJK SC"')

    body_serif, bold_serif = _build_font_families("Noto Serif CJK SC", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
    assert bold_serif.index('"Noto Serif CJK SC"') < bold_serif.index('"Noto Sans CJK SC"')


def test_extended_bold_font_detection_and_ultralight_guard():
    for fname in ("SFBX1000", "LMBX10", "LinLibertineTB", "PTMB8R", "NimbusRomNo9L-Medi", "CMBX12"):
        assert _is_span_bold({"font": fname, "flags": 4}), f"Expected {fname} to be detected as bold"
    # UltraLight / UltraThin must NEVER be flagged as bold
    for fname in ("HelveticaNeue-UltraLight", "SFProDisplay-Ultralight", "AvenirNext-UltraLight", "CMMI10"):
        assert not _is_span_bold({"font": fname, "flags": 4}), f"Expected {fname} NOT to be bold"

    # Paragraph containing math operator (∈) still extracts inline bold lead-in
    lines = [
        {
            "bbox": [54.0, 100.0, 290.0, 112.0],
            "spans": [
                {"font": "SFBX1000", "flags": 4, "size": 10.0, "origin": [54.0, 109.0], "bbox": [54.0, 100.0, 108.0, 112.0], "text": "Theorem 1."},
                {"font": "SFRM1000", "flags": 4, "size": 10.0, "origin": [110.0, 109.0], "bbox": [110.0, 100.0, 290.0, 112.0], "text": " For every x ∈ S, the cost is bounded."},
            ],
        }
    ]
    blk, _ = _build_block_from_lines("p0b1", lines, 612.0)
    assert blk is not None
    assert "**Theorem 1.**" in blk["text"]


def test_evidenced_latin_sqrt_unicode_math_and_anti_fragmentation():
    # Latin variable + proven multi-char subscript (d_k, d_model, √d_k)
    spans = [
        {"font": "CMR10", "flags": 4, "size": 10.0, "origin": [54.0, 109.0], "text": "Dimension "},
        {"font": "CMMI10", "flags": 6, "size": 10.0, "origin": [100.0, 109.0], "text": "d"},
        {"font": "CMMI7", "flags": 6, "size": 7.0, "origin": [106.0, 111.5], "text": "model"},
        {"font": "CMR10", "flags": 4, "size": 10.0, "origin": [130.0, 109.0], "text": " scaled by "},
        {"font": "CMSY10", "flags": 4, "size": 10.0, "origin": [175.0, 109.0], "text": "√"},
        {"font": "CMMI10", "flags": 6, "size": 10.0, "origin": [183.0, 109.0], "text": "d"},
        {"font": "CMMI7", "flags": 6, "size": 7.0, "origin": [189.0, 111.5], "text": "k"},
        {"font": "CMR10", "flags": 4, "size": 10.0, "origin": [195.0, 109.0], "text": "."},
    ]
    out = _render_line_with_evidenced_math(spans)
    assert r"$d_{\mathrm{model}}$" in out
    assert r"$\sqrt{d_k}$" in out

    # √𝐸 works properly
    spans_sqrt_u = [
        {"font": "CMSY10", "flags": 4, "size": 10.0, "origin": [54.0, 109.0], "text": "√"},
        {"font": "STIXMath", "flags": 4, "size": 10.0, "origin": [62.0, 109.0], "text": "𝐸"},
    ]
    assert _render_line_with_evidenced_math(spans_sqrt_u) == r"$\sqrt{𝐸}$"

    # Bare Unicode Math Alphanumeric without sub/sup does NOT fragment S𝑘(𝑡) into S$𝑘$($𝑡$)
    spans_sk = [
        {"font": "NimbusRom", "flags": 4, "size": 10.0, "origin": [54.0, 109.0], "text": "where S"},
        {"font": "STIXMath", "flags": 4, "size": 10.0, "origin": [84.0, 109.0], "text": "𝑘"},
        {"font": "NimbusRom", "flags": 4, "size": 10.0, "origin": [90.0, 109.0], "text": "("},
        {"font": "STIXMath", "flags": 4, "size": 10.0, "origin": [94.0, 109.0], "text": "𝑡"},
        {"font": "NimbusRom", "flags": 4, "size": 10.0, "origin": [98.0, 109.0], "text": ") holds."},
    ]
    assert "$𝑘$" not in _render_line_with_evidenced_math(spans_sk)

    # Touching math spans never produce adjacent $$ (prevents ⟪F0⟫ display-math mis-match)
    spans_adj = [
        {"font": "STIXMath", "flags": 4, "size": 10.0, "origin": [54.0, 109.0], "text": "𝜔"},
        {"font": "STIXMath", "flags": 4, "size": 7.0, "origin": [60.0, 111.5], "text": "𝐷"},
        {"font": "STIXMath", "flags": 4, "size": 10.0, "origin": [68.0, 109.0], "text": "𝜔"},
        {"font": "STIXMath", "flags": 4, "size": 7.0, "origin": [74.0, 111.5], "text": "𝐸"},
    ]
    out_adj = _render_line_with_evidenced_math(spans_adj)
    assert "$$" not in out_adj

    # Footnote markers (†, digit on italic a) must NOT become math
    spans_dag = [
        {"font": "CMR10", "flags": 4, "size": 10.0, "origin": [54.0, 109.0], "text": "a"},
        {"font": "CMR7", "flags": 4, "size": 7.0, "origin": [60.0, 105.0], "text": "†"},
    ]
    assert "$" not in _render_line_with_evidenced_math(spans_dag)

    # Closing parenthesis must NOT be swallowed as a subscript (fixes i_))
    spans_paren = [
        {"font": "CMMI10", "flags": 6, "size": 10.0, "origin": [54.0, 109.0], "text": "i"},
        {"font": "CMR7", "flags": 4, "size": 7.0, "origin": [58.0, 111.5], "text": ")"},
    ]
    assert "i_)" not in _render_line_with_evidenced_math(spans_paren)

    # ASCII control characters are stripped
    spans_ctrl = [
        {"font": "CMR10", "flags": 4, "size": 10.0, "origin": [54.0, 109.0], "text": "value \x132 + 1"},
    ]
    assert "\x13" not in _render_line_with_evidenced_math(spans_ctrl)


def test_math_split_boundary_accepts_unicode_math_alnum_subscript():
    prev_lines = [
        {
            "bbox": [327.9, 463.6, 484.7, 476.8],
            "spans": [
                {"font": "NimbusRom", "flags": 4, "size": 9.8, "origin": [328.0, 472.0], "bbox": [328.0, 464.0, 468.0, 475.0], "text": "let "},
                {"font": "STIXMath", "flags": 4, "size": 9.8, "origin": [470.0, 472.0], "bbox": [470.0, 464.0, 478.0, 475.0], "text": "Ψ"},
                {"font": "STIXMath", "flags": 4, "size": 6.8, "origin": [479.0, 468.0], "bbox": [479.0, 463.6, 484.7, 470.0], "text": "★"},
            ],
        }
    ]
    nxt_lines = [
        {
            "bbox": [479.0, 466.8, 558.2, 478.0],
            "spans": [
                {"font": "STIXMath", "flags": 4, "size": 6.8, "origin": [479.2, 474.5], "bbox": [479.2, 470.0, 484.5, 476.5], "text": "𝑘"},
                {"font": "NimbusRom", "flags": 4, "size": 9.8, "origin": [486.0, 472.0], "bbox": [486.0, 464.0, 558.2, 475.0], "text": " be the score"},
            ],
        }
    ]
    assert _is_math_split_boundary(prev_lines, nxt_lines)


def test_mitex_minimal_sqrt_and_unicode_passthrough():
    # √d_{model}, √𝐸, √(a+b) all convert cleanly without crashing MiTeX
    assert _sanitize_latex_for_mitex(r"√d_{model}") == r"\sqrt{d_{model}}"
    assert _sanitize_latex_for_mitex("√𝐸") == r"\sqrt{𝐸}"
    assert _sanitize_latex_for_mitex("√(a+b)") == r"\sqrt{a+b}"
    # Unicode math symbols pass through untouched to MiTeX raw backticks (no ‖ -> || degradation)
    assert _sanitize_latex_for_mitex("‖x‖_2 + 𝜔_𝐷 ≤ 0") == "‖x‖_2 + 𝜔_𝐷 ≤ 0"
    assert "\x13" not in _sanitize_latex_for_mitex("a\x13b")


def test_extract_json_object_handles_bfnrtu_latex_commands_and_real_newlines():
    raw = r'{"p0b1": "设 $\theta_k$, $\frac{1}{2}$, $\beta$, $\rho$, $\nabla$, $\underline{x}$, $\upsilon$, $a \\\alpha$\n换行。"}'
    obj = _extract_json_object(raw)
    val = obj["p0b1"]
    assert r"\theta_k" in val
    assert r"\frac{1}{2}" in val
    assert r"\beta" in val
    assert r"\rho" in val
    assert r"\nabla" in val
    assert r"\underline{x}" in val
    assert r"\upsilon" in val
    assert "\n换行。" in val


def test_typst_bold_uses_heiti_font_in_compiled_pdf():
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        src_pdf = wd / "src.pdf"
        doc = pymupdf.open()
        pg = doc.new_page(width=612, height=792)
        pg.insert_text((54, 100), "Batching. Current systems use continuous batching.", fontsize=10.0)
        doc.save(src_pdf)
        doc.close()

        res = render_typst(
            items=[
                {
                    "block_id": "p0b0",
                    "page_index": 0,
                    "reading_order": 0,
                    "layout_role": "body",
                    "policy": {"translate": True, "preserve_reason": None, "redact_source": True},
                    "bbox": [54.0, 88.0, 300.0, 140.0],
                    "line_bboxes": [[54.0, 88.0, 300.0, 140.0]],
                    "font_size_pt": 10.0,
                    "text": "**Batching.** Current systems use $d_k$ scaling.",
                    "translation": "**批处理。** 当前系统采用 $√d_{model}$ 与 $𝜔_𝐷 ≥ 0$ 约束。",
                    "translation_status": "ok",
                }
            ],
            workdir=wd,
            output_cn=wd / "cn.pdf",
            output_merged=wd / "merged.pdf",
            legacy_output=None,
            font="",
            source=str(src_pdf),
            layout_mode="preserve",
            bilingual_mode="side_by_side",
            page_widths={0: 612.0},
            page_coord_spaces={0: "pdf_pt"},
            page_scanned={0: False},
            strict=True,
        )
        assert res["placed"] == 1
        assert res["unplaced"] == 0
        otyp = (wd / "overlay.typ").read_text(encoding="utf-8")
        assert '#show strong: set text(font: ("Libertinus Serif", "Noto Sans CJK SC"' in otyp
        assert r"#mi(`\sqrt{d_{model}}`)" in otyp
        assert "#mi(`𝜔_𝐷 ≥ 0`)" in otyp


def test_author_affiliation_catches_ieee_and_plain_comma_author_lists():
    from ocr.normalize import _is_author_affiliation
    assert _is_author_affiliation("Zheng Lin, Student Member, IEEE, Yuxin Zhang, and Xianhao Chen", page_idx=0, seen_abstract=False, bbox=[72.0, 140.0, 540.0, 155.0], page_h=792.0, abstract_y0=240.0, font_size=10.0, median_font=10.0)
    assert _is_author_affiliation("A. Reddy, B. Smith, C. Kumar, and D. Lee", page_idx=0, seen_abstract=False, bbox=[72.0, 140.0, 540.0, 155.0], page_h=792.0, abstract_y0=240.0, font_size=10.0, median_font=10.0)
