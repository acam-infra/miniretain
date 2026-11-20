"""Regression tests for bordered/booktabs table preservation, algorithm & display formula
preservation, and shared Typst inline/display math spacing (LAB-2121)."""
from __future__ import annotations

from pathlib import Path
import tempfile
import pymupdf

from ocr.normalize import normalize_pages
from ocr.pymupdf_adapter import PyMuPDFOCRProvider
from render.typst_renderer import (
    _INLINE_MATH_GAP,
    find_typst_package_cache_dir,
    render_typst,
    typst_rich_text_expr,
)


def test_booktabs_minimal_border_table_preserves_cells_but_translates_caption_and_body():
    """A 3-rule booktabs table (no vertical borders) must preserve header/cells while
    keeping Table caption and adjacent same-column body prose translatable."""
    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "booktabs.pdf"
        doc = pymupdf.open()
        pg = doc.new_page(width=612, height=792)

        # Booktabs 3 horizontal rules (toprule, midrule, bottomrule) in left column
        for y in (80.0, 98.0, 150.0):
            pg.draw_line((54.0, y), (290.0, y), color=(0, 0, 0), width=0.8)
        pg.insert_text((60, 92), "Model       GPU       VRAM", fontsize=8.5)
        pg.insert_text((60, 114), "DeepSpeech2 A40 PCIe  48GB", fontsize=8.5)
        pg.insert_text((60, 132), "ResNet-50   V100 PCIe 32GB", fontsize=8.5)

        # Table caption below bottomrule
        pg.insert_text((54, 168), "Table 1: Hardware specifications used in our evaluation.", fontsize=9.0)
        # Adjacent body paragraph in the SAME column below caption
        pg.insert_text(
            (54, 198),
            "In this section we describe our experimental methodology and evaluation baselines.",
            fontsize=10.0,
        )
        doc.save(pdf_path)
        doc.close()

        pages = PyMuPDFOCRProvider().parse(str(pdf_path))
        blocks = pages[0]["blocks"]
        assert len(pages[0].get("table_regions") or []) >= 1

        table_blocks = [b for b in blocks if b["layout_role"] == "table"]
        caption_blocks = [b for b in blocks if b["layout_role"] == "caption"]
        body_blocks = [b for b in blocks if b["layout_role"] == "body"]

        assert len(table_blocks) >= 1
        assert all(b["policy"]["translate"] is False for b in table_blocks)
        assert all(b["policy"]["redact_source"] is False for b in table_blocks)
        assert len(caption_blocks) == 1
        assert caption_blocks[0]["policy"]["translate"] is True
        assert len(body_blocks) >= 1
        assert any("methodology" in b["text"] and b["policy"]["translate"] is True for b in body_blocks)


def test_display_equation_with_greek_eta_and_unicode_minus_preserved():
    """Display equation fragments like 2208.06102 p6 Eq. (9) using η, ·, − must be preserved."""
    pages = [
        {
            "page_index": 5,
            "width": 612.0,
            "height": 792.0,
            "figure_regions": [],
            "table_regions": [],
            "blocks": [
                {"block_id": "p5b0018", "text": "where the regret of choosing bt is defined as", "bbox": [53.6, 598.1, 229.6, 609.7], "font_size_pt": 10.0},
                {"block_id": "p5b0019", "text": "Regret(bt;η)", "bbox": [62.9, 616.2, 119.7, 627.1], "font_size_pt": 10.0},
                {"block_id": "p5b0020", "text": "= Epochs(bt)·EpochCost(bt;η)−min", "bbox": [65.1, 631.0, 226.1, 642.1], "font_size_pt": 10.0},
                {"block_id": "p5b0021", "text": "b,p Cost(b, p;η).", "bbox": [213.3, 631.1, 285.3, 648.3], "font_size_pt": 10.0},
            ],
        }
    ]
    norm = normalize_pages(pages, coord_space="pdf_pt", dpi=72.0)
    by_id = {b["block_id"]: b for b in norm[0]["blocks"]}
    assert by_id["p5b0018"]["policy"]["translate"] is True
    for bid in ("p5b0019", "p5b0020", "p5b0021"):
        assert by_id[bid]["layout_role"] == "formula"
        assert by_id[bid]["policy"]["translate"] is False
        assert by_id[bid]["policy"]["redact_source"] is False


def test_typst_shared_inline_and_display_math_spacing_rules():
    # 1. CJK <-> #mi() gets weak 0.18em gap; CJK punctuation stays tight; English space preserved
    expr = typst_rich_text_expr(
        r"打包成矩阵 $Q$ ，缩放到 $\frac{1}{\sqrt{d_k}}$ 倍，与 matrix $W_i^Q$ model 参与计算。",
        use_mitex=True,
    )
    assert f'#"打包成矩阵"{_INLINE_MATH_GAP}#mi(`Q`)#"，缩放到"{_INLINE_MATH_GAP}#mi(`\\frac{{1}}{{\\sqrt{{d_k}}}}`){_INLINE_MATH_GAP}#"倍，与 matrix "#mi(`W_i^Q`)#" model 参与计算。"' == expr

    # 2. Explicit newline before inline math is preserved without stray gap
    nl_expr = typst_rich_text_expr("第一行\n$d_k$ 维度", use_mitex=True)
    assert r'#"第一行' in nl_expr
    assert '#mi(`d_k`)' in nl_expr
    assert f'#mi(`d_k`){_INLINE_MATH_GAP}#"维度"' in nl_expr

    # 3. Display math uses compact predictable block spacing
    disp = typst_rich_text_expr(r"如下：$$\sum_{i=1}^{d_k} q_i k_i$$", use_mitex=True)
    assert "#block(above: 0.45em, below: 0.45em, breakable: false)[#mitex(`\\sum_{i=1}^{d_k} q_i k_i`)]" in disp


def test_typst_strict_compile_two_column_inline_math_spacing():
    """Verify measure() and overlay.typ compile cleanly in strict mode with inline math spacing."""
    if not find_typst_package_cache_dir():
        return
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        src_pdf = wd / "src.pdf"
        doc = pymupdf.open()
        pg = doc.new_page(width=612, height=792)
        pg.insert_text((54, 100), "Placeholder source text for strict two-column inline math check.", fontsize=10.0)
        doc.save(src_pdf)
        doc.close()

        items = [
            {
                "block_id": "p0b0000",
                "page_index": 0,
                "reading_order": 0,
                "layout_role": "body",
                "policy": {"translate": True, "preserve_reason": None, "redact_source": True},
                "bbox": [54.0, 88.0, 294.0, 148.0],
                "line_bboxes": [[54.0, 88.0, 294.0, 148.0]],
                "font_size_pt": 9.5,
                "text": "In practice we pack queries into matrix Q and scale by 1/sqrt(d_k).",
                "translation": r"在实践中，我们将查询打包成矩阵 $Q$ ，按 $\frac{1}{\sqrt{d_k}}$ 缩放，与 matrix $W_i^Q$ 参与计算，点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$ 。",
                "translation_status": "ok",
            }
        ]
        res = render_typst(
            items=items,
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
        overlay_typ = (wd / "overlay.typ").read_text(encoding="utf-8")
        assert _INLINE_MATH_GAP in overlay_typ
        assert '#mi(`Q`)#"，按"' in overlay_typ


def test_two_same_width_booktabs_tables_do_not_swallow_intervening_body():
    """Exact reproduction: two same-width 3-rule booktabs tables at x=60..280, y=(100,116,151)
    and y=(225,241,276) with body prose at y=198 must keep body prose translatable."""
    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "two_tables.pdf"
        doc = pymupdf.open()
        pg = doc.new_page(width=612, height=792)

        # Table 1: y=(100, 116, 151)
        for y in (100.0, 116.0, 151.0):
            pg.draw_line((60.0, y), (280.0, y), color=(0, 0, 0), width=0.8)
        pg.insert_text((66, 112), "Metric       Value", fontsize=8.5)
        pg.insert_text((66, 130), "Throughput   120", fontsize=8.5)
        pg.insert_text((66, 146), "Latency      4.2", fontsize=8.5)

        # Intervening body prose at y=198
        pg.insert_text(
            (60, 198),
            "This body paragraph sits between two same-width tables and must be translated.",
            fontsize=10.0,
        )

        # Table 2: y=(225, 241, 276)
        for y in (225.0, 241.0, 276.0):
            pg.draw_line((60.0, y), (280.0, y), color=(0, 0, 0), width=0.8)
        pg.insert_text((66, 237), "Hardware     Count", fontsize=8.5)
        pg.insert_text((66, 255), "V100         4", fontsize=8.5)
        pg.insert_text((66, 271), "A40          8", fontsize=8.5)

        doc.save(pdf_path)
        doc.close()

        pages = PyMuPDFOCRProvider().parse(str(pdf_path))
        t_regions = pages[0].get("table_regions") or []
        assert len(t_regions) == 2
        # Neither region may span from y=100 to y=276
        assert all((r[3] - r[1]) < 80.0 for r in t_regions)

        blocks = pages[0]["blocks"]
        mid_body = [b for b in blocks if "between two same-width tables" in b["text"]]
        assert len(mid_body) == 1
        assert mid_body[0]["layout_role"] == "body"
        assert mid_body[0]["policy"]["translate"] is True

        tbl_blocks = [b for b in blocks if b["layout_role"] == "table"]
        assert len(tbl_blocks) >= 2
        assert all(b["policy"]["translate"] is False for b in tbl_blocks)


def test_protect_and_restore_formulas_handles_paren_and_bracket_math_spans():
    """protect_formulas must protect whole \\(...\\) and \\[...\\] spans before internal LaTeX commands."""
    from llm.formula_protect import protect_formulas, restore_formulas

    src = (
        r"Given \(x_i = y_i + 1\) and display \[\frac{1}{\sqrt{d_k}} + \mathcal{O}(n)\], "
        r"we also keep $z_k$ and $$a + b$$ intact."
    )
    protected, tokens = protect_formulas(src, layout_role="body")
    assert len(tokens) == 4
    vals = set(tokens.values())
    assert r"\(x_i = y_i + 1\)" in vals
    assert r"\[\frac{1}{\sqrt{d_k}} + \mathcal{O}(n)\]" in vals
    assert "$z_k$" in vals
    assert "$$a + b$$" in vals
    # Raw LaTeX spans must not leak to LLM prompt text
    assert r"\(x_i = y_i + 1\)" not in protected
    assert r"\frac" not in protected

    restored, missing = restore_formulas(protected, tokens)
    assert missing == []
    assert restored == src


def test_inline_math_subscript_split_blocks_merged_protected_and_rendered():
    """Reproduce 2208.06102 p6 Gaussian paragraph: mid-line split at 'ˆσ2' / 'b).' and '1/ˆσ2' / 'b can'
    must merge into one intact body block, reconstruct LaTeX accents/sub/sup, produce non-empty
    protect_formulas tokens, and compile cleanly via Typst MiTeX in strict mode."""
    from llm.formula_protect import protect_formulas, restore_formulas
    from ocr.pymupdf_adapter import _merge_inline_math_split_blocks

    raw_blocks = [
        {
            "block_id": "p5b0048",
            "text": "Specifically, the cost distribution has unknown mean θb. That is, θb ∼N (ˆµb, ˆσ2",
            "bbox": [317.9, 423.8, 528.6, 470.8],
            "line_bboxes": [[317.9, 423.8, 558.0, 446.5], [317.9, 457.9, 528.6, 470.8]],
            "font_size_pt": 9.96,
            "is_bold": False,
            "is_math_font": False,
            "is_vertical": False,
        },
        {
            "block_id": "p5b0049",
            "text": "b). Here it is important to note that 1/ˆσ2",
            "bbox": [317.9, 459.4, 558.0, 482.2],
            "line_bboxes": [[524.9, 459.4, 558.0, 471.9], [317.9, 469.9, 449.4, 482.2]],
            "font_size_pt": 9.96,
            "is_bold": False,
            "is_math_font": False,
            "is_vertical": False,
        },
        {
            "block_id": "p5b0050",
            "text": "b can be thought of as confidence, choosing ˆθb ∼N (ˆµb, ˆσ2",
            "bbox": [317.9, 471.5, 558.4, 542.5],
            "line_bboxes": [[445.7, 471.5, 558.4, 484.6], [317.9, 528.9, 554.4, 542.5]],
            "font_size_pt": 9.96,
            "is_bold": False,
            "is_math_font": False,
            "is_vertical": False,
        },
        {
            "block_id": "p5b0051",
            "text": "b) (Algorithm 1).",
            "bbox": [550.7, 531.1, 558.8, 543.6],
            "line_bboxes": [[550.7, 531.1, 558.8, 543.6]],
            "font_size_pt": 9.96,
            "is_bold": False,
            "is_math_font": False,
            "is_vertical": False,
        },
    ]

    merged = _merge_inline_math_split_blocks(raw_blocks)
    assert len(merged) == 1
    assert merged[0]["block_id"] == "p5b0048"

    norm = normalize_pages(
        [
            {
                "page_index": 5,
                "width": 612.0,
                "height": 792.0,
                "figure_regions": [],
                "table_regions": [],
                "blocks": merged,
            }
        ],
        coord_space="pdf_pt",
        dpi=72.0,
    )
    blk = norm[0]["blocks"][0]
    assert blk["layout_role"] == "body"
    assert blk["policy"]["translate"] is True
    assert r"$\theta_b \sim \mathcal{N}(\hat{\mu}_b, \hat{\sigma}_b^2)$" in blk["text"]
    assert r"$1/\hat{\sigma}_b^2$" in blk["text"]
    assert r"$\hat{\theta}_b \sim \mathcal{N}(\hat{\mu}_b, \hat{\sigma}_b^2)$" in blk["text"]

    protected, tokens = protect_formulas(blk["text"], layout_role="body")
    assert len(tokens) == 4
    vals = set(tokens.values())
    assert r"$\theta_b \sim \mathcal{N}(\hat{\mu}_b, \hat{\sigma}_b^2)$" in vals
    assert r"$1/\hat{\sigma}_b^2$" in vals
    assert r"$\hat{\theta}_b \sim \mathcal{N}(\hat{\mu}_b, \hat{\sigma}_b^2)$" in vals

    # Simulate translated prose preserving placeholders, restore, and compile via Typst MiTeX
    zh_with_toks = "具体而言，开销分布的未知均值为 ⟪F0⟫，即 ⟪F1⟫。其中 ⟪F2⟫ 表示置信度，并采样 ⟪F3⟫（算法 1）。"
    restored, missing = restore_formulas(zh_with_toks, tokens)
    assert missing == []

    if find_typst_package_cache_dir():
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            src_pdf = wd / "src.pdf"
            doc = pymupdf.open()
            pg = doc.new_page(width=612, height=792)
            pg.insert_text((318, 440), "Gaussian placeholder", fontsize=10.0)
            doc.save(src_pdf)
            doc.close()
            res = render_typst(
                items=[
                    {
                        "block_id": "p5b0048",
                        "page_index": 0,
                        "reading_order": 0,
                        "layout_role": "body",
                        "policy": {"translate": True, "preserve_reason": None, "redact_source": True},
                        "bbox": [317.6, 423.8, 559.7, 543.6],
                        "line_bboxes": [[317.6, 423.8, 559.7, 543.6]],
                        "font_size_pt": 9.96,
                        "text": blk["text"],
                        "translation": restored,
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
            overlay = (wd / "overlay.typ").read_text(encoding="utf-8")
            assert r"#mi(`\theta_b \sim \mathcal{N}(\hat{\mu}_b, \hat{\sigma}_b^2)`)" in overlay
            assert r"#mi(`1/\hat{\sigma}_b^2`)" in overlay


