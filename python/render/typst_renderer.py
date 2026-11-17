"""Typst renderer for Miniretain: compiles translated items into PDF.

Supports generating:
1. Pure Chinese translated document (`output_cn`, e.g. translated_cn.pdf)
2. Bilingual merged comparison document (`output_merged`, e.g. translated_merged.pdf)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_typst() -> str:
    """Find a usable typst binary."""
    candidate = shutil.which("typst")
    if candidate:
        return candidate
    for p in [
        "/home/jiangminuo/.local/bin/typst",
        "/usr/local/bin/typst",
        "/usr/bin/typst",
        os.path.expanduser("~/.local/bin/typst"),
        os.path.expanduser("~/bin/typst"),
    ]:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise RuntimeError("typst binary not found. Please install typst or specify path.")


def escape_typst(text: str) -> str:
    """Escape Typst markup characters."""
    if not text:
        return ""
    # Backslash first
    s = text.replace("\\", "\\\\")
    for ch in ["#", "$", "@", "[", "]", "*", "_", "`", "<", ">"]:
        s = s.replace(ch, "\\" + ch)
    return s


def render_typst(
    items: list[dict],
    workdir: Path,
    output_cn: Path,
    output_merged: Path,
    legacy_output: Path | None = None,
    font: str = "",
) -> dict:
    """Compile items into Chinese and bilingual merged PDFs using Typst."""
    typst_bin = find_typst()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    font_list = '("Libertinus Serif", "Droid Sans Fallback", "Source Han Serif SC", "SimSun")'
    if font:
        font_list = f'("{font}", "Libertinus Serif", "Droid Sans Fallback")'

    # 1. Generate document_cn.typ (Pure Chinese)
    cn_typ = workdir / "document_cn.typ"
    cn_lines = [
        '#set page(',
        '  paper: "a4",',
        '  margin: (x: 2cm, y: 2.2cm),',
        '  header: align(right)[#text(size: 8pt, fill: luma(120))[RetainPDF 中文译本 (Chinese Edition)]],',
        '  footer: align(center)[#counter(page).display("1 / 1", both: true)]',
        ')',
        f'#set text(font: {font_list}, size: 10.5pt, lang: "zh")',
        '#set par(justify: true, leading: 0.85em, first-line-indent: 2em)',
        '',
    ]

    for i, it in enumerate(items):
        trans = (it.get("translation") or "").strip()
        if not trans:
            continue
        escaped = escape_typst(trans)
        if i == 0 and len(trans) < 80 and not trans.endswith(("。", "！", "？", ".")):
            cn_lines.append(f'= {escaped}\n')
        else:
            cn_lines.append(f'#par[{escaped}]\n')

    cn_typ.write_text("\n".join(cn_lines), encoding="utf-8")
    cmd_cn = [typst_bin, "compile", str(cn_typ), str(output_cn)]
    proc_cn = subprocess.run(cmd_cn, capture_output=True, text=True)
    if proc_cn.returncode != 0:
        raise RuntimeError(f"typst compile cn failed: {proc_cn.stderr.strip()}")

    # 2. Generate document_bilingual.typ (Bilingual Merged Comparison)
    bi_typ = workdir / "document_bilingual.typ"
    bi_lines = [
        '#set page(',
        '  paper: "a4",',
        '  margin: (x: 1.5cm, y: 1.8cm),',
        '  header: align(right)[#text(size: 8pt, fill: luma(120))[RetainPDF 双语对照合并版 (Bilingual Merged)]],',
        '  footer: align(center)[#counter(page).display("1 / 1", both: true)]',
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
        esc_orig = escape_typst(orig)
        esc_trans = escape_typst(trans)
        bi_lines.append(f'  [{esc_orig}], [{esc_trans}],')

    bi_lines.append(')')
    bi_typ.write_text("\n".join(bi_lines), encoding="utf-8")
    cmd_bi = [typst_bin, "compile", str(bi_typ), str(output_merged)]
    proc_bi = subprocess.run(cmd_bi, capture_output=True, text=True)
    if proc_bi.returncode != 0:
        raise RuntimeError(f"typst compile bilingual failed: {proc_bi.stderr.strip()}")

    if legacy_output and str(legacy_output) != str(output_cn):
        shutil.copyfile(output_cn, legacy_output)

    return {
        "engine": "typst",
        "items": len(items),
        "output_cn": str(output_cn),
        "output_merged": str(output_merged),
        "output": str(legacy_output or output_cn),
    }
