"""Typst renderer for Miniretain: supports layout-preserving per-page overlay
(RetainPDF-aligned) and A4 streaming fallback.

Generates:
1. Pure Chinese translated document (`output_cn`, e.g. translated_cn.pdf)
2. Bilingual comparison document (`output_merged`, side-by-side in preserve mode,
   or A4 comparison table in flow mode)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pymupdf

from render.layout import find_cjk_font, fit_textbox_size, pdf_rect
from render.side_by_side import build_side_by_side_pdf, merge_interleaved


def find_typst() -> str:
    """Find a usable typst binary."""
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
    # Prevent // and /* from being parsed as Typst comments
    s = s.replace("//", "\\/\\/").replace("/*", "\\/\\*")
    return s


def typst_str_literal(text: str) -> str:
    """Escape text as a Typst quoted string literal "..." (immune to all markup/comments)."""
    if not text:
        return '""'
    s = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _typst_version(typst_bin: str) -> str:
    try:
        proc = subprocess.run([typst_bin, "--version"], capture_output=True, text=True, timeout=10)
        return proc.stdout.strip()
    except Exception:
        return "unknown"


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
    typst_bin = find_typst()
    cjk_font_file = find_cjk_font(font)
    font_dir = str(Path(cjk_font_file).parent)
    font_families = '("Noto Serif CJK SC", "Noto Sans CJK SC", "Source Han Serif SC", "Droid Sans Fallback", "Libertinus Serif")'

    doc = pymupdf.open(source_pdf)
    placed = 0
    preserved = 0
    unplaced = 0
    failed = 0
    shrunk = 0
    unplaced_block_ids: list[str] = []
    failed_block_ids: list[str] = []
    per_page: list[dict] = []

    overlay_pages_typ: list[str] = [
        '#set page(margin: 0pt, fill: none)',
        f'#set text(font: {font_families}, lang: "zh", fill: black)',
        '#set par(justify: true, leading: 0.45em)',
        '',
    ]

    scratch_doc = pymupdf.open()
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            pw = float(page.rect.width)
            ph = float(page.rect.height)
            spage = scratch_doc.new_page(width=pw, height=ph)
            spage.insert_font(fontname="mr_fit", fontfile=cjk_font_file)
            coord_space = page_coord_spaces.get(page_index, "pdf_pt")
            scale = 1.0 if coord_space == "pdf_pt" else (page_widths.get(page_index, pw) / pw)
            is_scanned = bool(page_scanned.get(page_index, False))

            page_items = [it for it in items if it["page_index"] == page_index]
            p_placed = 0
            p_preserved = 0
            p_unplaced = 0
            p_failed = 0
            to_render: list[tuple[dict, pymupdf.Rect, float]] = []

            for it in page_items:
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
                start_sz = (
                    min(14.0, max(min_size, float(hint) * 0.92))
                    if hint and isinstance(hint, (int, float)) and hint > 0
                    else min(14.0, max(min_size, rect.height * 0.78))
                )
                fits, chosen_sz, steps = fit_textbox_size(
                    pw, ph, rect, trans, cjk_font_file, start_size=start_sz, min_size=min_size, scratch_page=spage
                )
                shrunk += steps
                if not fits:
                    # Keep source block intact and report ID
                    unplaced += 1
                    p_unplaced += 1
                    unplaced_block_ids.append(bid)
                    continue
                to_render.append((it, rect, chosen_sz))

            # Redact ONLY fitted translatable blocks (fill=None on digital pages)
            redact_fill = (1, 1, 1) if is_scanned else None
            for it, _rect, _sz in to_render:
                for lb in it.get("line_bboxes") or [it["bbox"]]:
                    page.add_redact_annot(pdf_rect(lb, scale), fill=redact_fill)
            if to_render:
                page.apply_redactions(images=0, graphics=0, text=0)

            # Emit Typst overlay page with exact original dimensions
            if page_index > 0:
                overlay_pages_typ.append("#pagebreak()")
            overlay_pages_typ.append(f"#set page(width: {pw:.2f}pt, height: {ph:.2f}pt, margin: 0pt, fill: none)")
            for it, rect, chosen_sz in to_render:
                trans_lit = typst_str_literal(it["translation"].strip())
                is_bold = bool(it.get("is_bold") or (it.get("content") or {}).get("is_bold"))
                role = it.get("layout_role", "body")
                weight = "bold" if (is_bold or role in {"title", "heading"}) else "regular"
                w = max(4.0, rect.width)
                h = max(4.0, rect.height)
                # Slightly tighter leading when font shrunk to prevent vertical overflow in Typst
                leading = "0.38em" if chosen_sz < 8.5 else "0.45em"
                overlay_pages_typ.append(
                    f'#place(top + left, dx: {rect.x0:.2f}pt, dy: {rect.y0:.2f}pt, '
                    f'block(width: {w:.2f}pt, height: {h:.2f}pt, clip: false)'
                    f'[#set text(size: {chosen_sz:.2f}pt, weight: "{weight}"); '
                    f'#set par(leading: {leading}, justify: true); #{trans_lit}])'
                )
                placed += 1
                p_placed += 1

            per_page.append({
                "page_index": page_index,
                "total_blocks": len(page_items),
                "placed": p_placed,
                "preserved": p_preserved,
                "unplaced": p_unplaced,
                "failed": p_failed,
                "width_pt": round(pw, 2),
                "height_pt": round(ph, 2),
            })

        if strict and (unplaced_block_ids or failed_block_ids):
            raise RuntimeError(
                f"render strict mode failed: unplaced={unplaced_block_ids}, failed={failed_block_ids}"
            )

        overlay_typ = workdir / "overlay.typ"
        overlay_pdf = workdir / "overlay.pdf"
        overlay_typ.write_text("\n".join(overlay_pages_typ), encoding="utf-8")
        proc = subprocess.run(
            [typst_bin, "compile", "--font-path", font_dir, str(overlay_typ), str(overlay_pdf)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"typst overlay compile failed: {proc.stderr.strip()}")

        with pymupdf.open(overlay_pdf) as overlay_doc:
            if len(overlay_doc) != len(doc):
                raise RuntimeError(
                    f"overlay page count mismatch: source={len(doc)}, overlay={len(overlay_doc)}"
                )
            for page_index in range(len(doc)):
                doc[page_index].show_pdf_page(doc[page_index].rect, overlay_doc, page_index, overlay=True)

        output_cn.parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_cn, garbage=4, deflate=True)
    finally:
        scratch_doc.close()
        doc.close()

    if not output_cn.is_file() or output_cn.stat().st_size == 0:
        raise RuntimeError(f"output_cn missing or empty: {output_cn}")

    if legacy_output and str(legacy_output) != str(output_cn):
        shutil.copyfile(output_cn, legacy_output)

    if bilingual_mode == "interleaved":
        bilingual_info = merge_interleaved(source_pdf, output_cn, output_merged)
    else:
        bilingual_info = build_side_by_side_pdf(source_pdf, output_cn, output_merged)

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
        "failed_block_ids": failed_block_ids,
        "shrunk_steps": shrunk,
        "per_page": per_page,
        "output_cn": str(output_cn),
        "output_merged": str(output_merged),
        "output": str(legacy_output or output_cn),
        "bilingual": bilingual_info,
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
