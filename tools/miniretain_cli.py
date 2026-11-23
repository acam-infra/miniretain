#!/usr/bin/env python3
"""miniretain-cli: Config-driven layout-preserving PDF translation, API/IP pool & batch parallel reading CLI.

Supports:
- Single or batch inputs: pass 1 or N PDF paths / URLs / arXiv IDs, or --batch-file / --batch-dir
- Automatic YAML config loading (--config / MINIRETAIN_CONFIG / ./miniretain.yaml / ~/.config/miniretain/cli.yaml / ~/miniretain/miniretain.yaml)
- Multi-endpoint load balancing (--api-pool http://100.79.113.126:18080,http://100.72.71.98:18080)
- AI translation via cc (http://100.99.0.3:8083/v1) with multi-API-key & IP/proxy pool rotation
- OCR providers: auto | pymupdf | mineru (MinerU Cloud v4) | paddle-cloud (PaddleOCR-VL-1.6 Cloud) | paddle-http | paddle-cpu

Examples:
  miniretain-cli paper.pdf
  miniretain-cli 2611.20893 --ocr mineru --out-dir ./out
  miniretain-cli paper.pdf --ocr paddle-cloud --fallback-ocr mineru
  miniretain-cli 2611.20893 2208.06102 --batch-workers 2 --api-pool http://100.79.113.126:18080,http://100.72.71.98:18080
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ARXIV_ID_RE = re.compile(r"^(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$", re.IGNORECASE)
ALLOWED_OCR_PROVIDERS = ("auto", "pymupdf", "mineru", "paddle-cloud", "paddle-http", "paddle-cpu", "stub")


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line_no_comment = re.sub(r"\s+#.*$", "", raw_line)
        if not line_no_comment.strip() or line_no_comment.lstrip().startswith("#"):
            continue
        indent = len(line_no_comment) - len(line_no_comment.lstrip(" "))
        stripped = line_no_comment.strip()
        if ":" not in stripped:
            continue
        key, val_raw = stripped.split(":", 1)
        key = key.strip()
        val_str = val_raw.strip()
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if not val_str or val_str == "{}":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            if (val_str.startswith('"') and val_str.endswith('"')) or (val_str.startswith("'") and val_str.endswith("'")):
                parent[key] = val_str[1:-1]
            elif val_str.lower() in ("true", "false"):
                parent[key] = val_str.lower() == "true"
            else:
                try:
                    parent[key] = int(val_str) if "." not in val_str else float(val_str)
                except ValueError:
                    parent[key] = val_str
    return root


def discover_config_path(explicit_path: str | None) -> Path | None:
    candidates: list[Path] = []
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {explicit_path}")
        return p
    if os.environ.get("MINIRETAIN_CONFIG"):
        candidates.append(Path(os.environ["MINIRETAIN_CONFIG"]).expanduser())
    candidates.extend([
        Path.cwd() / "miniretain.yaml",
        Path.home() / ".config" / "miniretain" / "cli.yaml",
        Path.home() / "miniretain" / "miniretain.yaml",
        Path("/home/agentuser/miniretain/miniretain.yaml"),
        Path("/root/miniretain/miniretain.yaml"),
    ])
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def resolve_api_url_from_config(cfg: dict[str, Any]) -> str:
    client_cfg = cfg.get("client") if isinstance(cfg.get("client"), dict) else {}
    if client_cfg.get("api_url"):
        return str(client_cfg["api_url"]).strip().rstrip("/")
    srv_cfg = cfg.get("server") if isinstance(cfg.get("server"), dict) else {}
    addr = str(srv_cfg.get("addr") or "").strip()
    if addr:
        if addr.startswith(("http://", "https://")):
            return addr.rstrip("/")
        if addr.startswith(":"):
            return f"http://127.0.0.1{addr}"
        if addr.startswith("0.0.0.0:"):
            return f"http://127.0.0.1:{addr.split(':', 1)[1]}"
        return f"http://{addr}"
    return "http://127.0.0.1:18080"


def resolve_input_pdf(source: str, work_dir: Path) -> tuple[Path, str]:
    p = Path(source).expanduser()
    if p.exists() and p.is_file():
        return p.resolve(), p.stem

    m = ARXIV_ID_RE.match(source.strip())
    if m:
        arxiv_id = m.group(1)
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        slug = f"arxiv_{arxiv_id.replace('.', '_')}"
    elif source.startswith("http://") or source.startswith("https://"):
        url = source.strip()
        if "arxiv.org/abs/" in url:
            url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/")
            if not url.endswith(".pdf"):
                url += ".pdf"
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", url.rstrip("/").split("/")[-1].replace(".pdf", "")) or "paper"
    else:
        raise FileNotFoundError(f"Input not found as local file, URL, or arXiv ID: {source}")

    dest = (work_dir / f"{slug}.pdf").resolve()
    if not dest.exists() or dest.stat().st_size < 1024:
        print(f"[miniretain-cli] Downloading {url} -> {dest} ...", file=sys.stderr)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; MiniretainCLI/1.2)"})
        with opener.open(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
    return dest, slug


def api_json(opener, method: str, url: str, payload: dict | None = None, timeout: int = 30) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": body}


def build_reading_guide(job_dir: Path, slug: str, job_id: str, result_meta: dict) -> str:
    doc_path = job_dir / "document.v1.json"
    trans_path = job_dir / "translation.payload.json"
    if not doc_path.exists() or not trans_path.exists():
        return ""

    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    trans = json.loads(trans_path.read_text(encoding="utf-8"))
    trans_map: dict[str, str] = {}
    items = trans.get("items") or trans.get("blocks") or []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and it.get("block_id"):
                trans_map[it["block_id"]] = (
                    it.get("translation")
                    or it.get("translated_text")
                    or it.get("text")
                    or ""
                )

    ocr_m = result_meta.get("ocr", {})
    llm_m = result_meta.get("llm", {})
    ren_m = result_meta.get("render", {})

    lines = [
        f"# {slug} — Miniretain 双语精读速览报告",
        "",
        "## 1. 翻译与排版对账摘要",
        f"- **Job ID**: `{job_id}`",
        f"- **总页数**: `{len(doc.get('pages', []))}` 页",
        f"- **OCR 后端与统计**: `{ocr_m.get('provider', 'auto')}`，识别 `{ocr_m.get('blocks', 0)}` 块，路由 `{json.dumps(ocr_m.get('routed_providers', {}), ensure_ascii=False)}`（耗时 `{ocr_m.get('stage_wall_ms', 0)} ms`）",
        f"- **LLM 翻译统计**: 已译 `{llm_m.get('translated', 0)} / {llm_m.get('translatable', 0)}` 块，原位保源 `{llm_m.get('preserved', 0)}` 块，批次 `{llm_m.get('batches', 0)}`，重试 `{llm_m.get('retries', 0)}`（耗时 `{llm_m.get('stage_wall_ms', 0)} ms`）",
        f"- **Typst 渲染统计**: 严格落版 `{ren_m.get('placed', 0)}` 块，原位保源 `{ren_m.get('preserved', 0)}` 块，未落版 `{ren_m.get('unplaced', 0)}` 块，失败 `{ren_m.get('failed', 0)}` 块（引擎 `{ren_m.get('typst_version', 'typst 0.15.1')}`，耗时 `{ren_m.get('stage_wall_ms', 0)} ms`）",
        "",
        "## 2. 逐页核心译文与章节导览（中英对照）",
        "",
    ]

    for page in doc.get("pages", []):
        pno = page.get("page_number", page.get("number", 0) + 1)
        lines.append(f"### 第 {pno} 页")
        for blk in page.get("blocks", []):
            bid = blk.get("block_id") or blk.get("id") or ""
            role = blk.get("layout_role") or blk.get("role") or "body"
            src_txt = (blk.get("text") or "").strip().replace("\n", " ")
            zh_txt = (trans_map.get(bid) or "").strip().replace("\n", " ")
            if role in ("title", "heading", "abstract", "caption") and (zh_txt or src_txt):
                tag = role.upper()
                if zh_txt and zh_txt != src_txt:
                    lines.append(f"- **[{tag} · `{bid}`]** {zh_txt}  \n  *原文*: `{src_txt}`")
                else:
                    lines.append(f"- **[{tag} · `{bid}` (保源)]** `{src_txt}`")
            elif role == "body" and zh_txt:
                lines.append(f"- **正文 `{bid}`**: {zh_txt}")
            elif role in ("formula", "table") and src_txt:
                lines.append(f"- *[{role.upper()} 保源 · `{bid}`]* `{src_txt[:100]}`")
        lines.append("")

    return "\n".join(lines)


def process_single_paper(
    source: str,
    *,
    api_base: str,
    out_dir: Path,
    ocr_req: dict[str, Any],
    llm_req: dict[str, Any],
    render_base: dict[str, Any],
    target_lang: str,
    summary: bool,
    cfg_path: Path | None,
) -> dict[str, Any]:
    opener = _no_proxy_opener()
    code, health = api_json(opener, "GET", f"{api_base}/health")
    if code != 200 or not health.get("ok"):
        raise RuntimeError(f"Server health check failed ({code}) at {api_base}: {health}")

    pdf_path, slug = resolve_input_pdf(source, out_dir)
    req_body = {
        "source_path": str(pdf_path),
        "target_lang": target_lang,
        "ocr": ocr_req,
        "llm": llm_req,
        "render": {
            **render_base,
            "output_cn": f"{slug}_CN.pdf",
            "output_merged": f"{slug}_Side-by-Side.pdf",
        },
    }

    t0 = time.time()
    code, job = api_json(opener, "POST", f"{api_base}/v1/jobs", req_body)
    if code not in (200, 202) or "id" not in job:
        raise RuntimeError(f"Failed to submit job ({code}) for {source}: {job}")

    job_id = job["id"]
    print(
        f"[miniretain-cli] Submitted job_id={job_id} (ocr={ocr_req.get('provider')}, model={llm_req.get('model')}, api={api_base}) for {pdf_path.name}",
        file=sys.stderr,
    )

    while True:
        code, status_obj = api_json(opener, "GET", f"{api_base}/v1/jobs/{job_id}")
        st = status_obj.get("status")
        stage = status_obj.get("stage", "")
        elapsed = time.time() - t0
        print(f"[miniretain-cli] [{elapsed:5.1f}s] job={job_id} ({slug}) status={st} stage={stage}", file=sys.stderr)
        if st == "succeeded":
            break
        if st == "failed":
            _, res_err = api_json(opener, "GET", f"{api_base}/v1/jobs/{job_id}/result")
            return {
                "source": source,
                "slug": slug,
                "job_id": job_id,
                "status": "failed",
                "api_base": api_base,
                "job": status_obj,
                "diagnostics": res_err,
            }
        time.sleep(2.0)

    code, raw_res = api_json(opener, "GET", f"{api_base}/v1/jobs/{job_id}/result")
    result_meta = raw_res.get("result", raw_res)
    wall_time_s = round(time.time() - t0, 3)

    render_info = result_meta.get("render", {})
    cn_src = Path(render_info.get("output_cn", ""))
    merged_src = Path(render_info.get("output_merged", ""))
    job_dir = cn_src.parent if cn_src.exists() else Path(f"data/jobs/{job_id}")

    cn_dst = out_dir / f"{slug}_CN.pdf"
    merged_dst = out_dir / f"{slug}_Side-by-Side.pdf"
    if cn_src.exists():
        shutil.copy2(cn_src, cn_dst)
    if merged_src.exists():
        shutil.copy2(merged_src, merged_dst)

    guide_dst = out_dir / f"{slug}_Reading_Guide.md"
    if summary and job_dir.exists():
        guide_md = build_reading_guide(job_dir, slug, job_id, result_meta)
        if guide_md:
            guide_dst.write_text(guide_md, encoding="utf-8")

    return {
        "source": source,
        "slug": slug,
        "job_id": job_id,
        "status": "succeeded",
        "config_file": str(cfg_path) if cfg_path else None,
        "api_base": api_base,
        "ocr_provider": ocr_req.get("provider"),
        "llm_model": llm_req.get("model"),
        "wall_time_s": wall_time_s,
        "output_cn_pdf": str(cn_dst) if cn_dst.exists() else str(cn_src),
        "output_bilingual_pdf": str(merged_dst) if merged_dst.exists() else str(merged_src),
        "reading_guide_md": str(guide_dst) if guide_dst.exists() else None,
        "ocr": result_meta.get("ocr"),
        "llm": result_meta.get("llm"),
        "render": render_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Miniretain config-driven PDF translator, API/IP pool & batch reading CLI")
    parser.add_argument("sources", nargs="*", help="One or more PDF paths, PDF URLs, or arXiv IDs (e.g. 2611.20893)")
    parser.add_argument("-c", "--config", default=None, help="Path to YAML config file (auto-discovered if omitted)")
    parser.add_argument("--batch-file", default=None, help="File containing one PDF path / URL / arXiv ID per line")
    parser.add_argument("--batch-dir", default=None, help="Directory of PDF files to translate in batch")
    parser.add_argument("--batch-workers", type=int, default=2, help="Number of concurrent papers in batch mode (default: 2)")
    parser.add_argument("--api", default=None, help="Miniretain server base URL (overrides config server.addr)")
    parser.add_argument("--api-pool", default=None, help="Comma-separated pool of Miniretain server URLs for batch load balancing")
    parser.add_argument("--ocr", choices=ALLOWED_OCR_PROVIDERS, default=None, help="OCR provider: auto | pymupdf | mineru | paddle-cloud | paddle-http | paddle-cpu")
    parser.add_argument("--fallback-ocr", choices=("paddle-http", "paddle-cloud", "mineru", "paddle-cpu", "none"), default=None, help="Fallback OCR provider on scanned pages or quota failover")
    parser.add_argument("--out-dir", default="./miniretain_out", help="Output directory for translated PDFs and guide")
    parser.add_argument("--target-lang", default=None, help="Target language (default from YAML or zh)")
    parser.add_argument("--model", default=None, help="LLM model name on cc gateway (default from YAML or gemini-3.6-flash)")
    parser.add_argument("--batch-size", type=int, default=None, help="Items per LLM batch")
    parser.add_argument("--concurrency", type=int, default=None, help="Per-job LLM concurrency")
    parser.add_argument("--no-strict", action="store_true", help="Disable strict layout overflow check")
    parser.add_argument("--summary", action="store_true", default=True, help="Generate structured Markdown reading guide")
    args = parser.parse_args()

    all_sources: list[str] = list(args.sources or [])
    if args.batch_file:
        bf = Path(args.batch_file).expanduser()
        for line in bf.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                all_sources.append(s)
    if args.batch_dir:
        bd = Path(args.batch_dir).expanduser()
        for p in sorted(bd.glob("*.pdf")):
            all_sources.append(str(p.resolve()))
    if not all_sources:
        parser.error("Provide at least one source (PDF path, URL, or arXiv ID) or --batch-file / --batch-dir")

    cfg_path = discover_config_path(args.config)
    cfg: dict[str, Any] = {}
    if cfg_path:
        cfg = _parse_simple_yaml(cfg_path.read_text(encoding="utf-8"))
        print(f"[miniretain-cli] Loaded config: {cfg_path}", file=sys.stderr)

    ocr_cfg = cfg.get("ocr") if isinstance(cfg.get("ocr"), dict) else {}
    llm_cfg = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
    ren_cfg = cfg.get("render") if isinstance(cfg.get("render"), dict) else {}

    primary_api = (args.api or os.environ.get("MINIRETAIN_API") or resolve_api_url_from_config(cfg)).rstrip("/")
    api_endpoints = [
        u.strip().rstrip("/")
        for u in (args.api_pool or os.environ.get("MINIRETAIN_API_POOL") or primary_api).split(",")
        if u.strip()
    ]
    ocr_provider = args.ocr or str(ocr_cfg.get("provider") or "auto")
    target_lang = args.target_lang or str(cfg.get("target_lang") or "zh")
    llm_model = args.model or str(llm_cfg.get("model") or "gemini-3.6-flash")
    batch_size = int(args.batch_size if args.batch_size is not None else (llm_cfg.get("batch_size") or 18))
    concurrency = int(args.concurrency if args.concurrency is not None else (llm_cfg.get("concurrency") or 8))
    max_retries = int(llm_cfg.get("max_retries") if llm_cfg.get("max_retries") is not None else 2)
    timeout_ms = int(llm_cfg.get("timeout_ms") or 120000)
    strict_mode = False if args.no_strict else bool(ren_cfg.get("strict", True))

    ocr_req: dict[str, Any] = {"provider": ocr_provider}
    if args.fallback_ocr:
        ocr_req["config"] = {
            "auto": {"fallback_provider": args.fallback_ocr},
            "paddle_cloud": {"fallback_provider": args.fallback_ocr},
            "mineru": {"fallback_provider": args.fallback_ocr},
        }

    llm_req: dict[str, Any] = {
        "provider": "openai-compat",
        "model": llm_model,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "max_retries": max_retries,
        "timeout_ms": timeout_ms,
        "allow_partial": False,
    }
    render_base: dict[str, Any] = {
        "engine": str(ren_cfg.get("engine") or "typst"),
        "layout_mode": str(ren_cfg.get("layout_mode") or "preserve"),
        "bilingual_mode": str(ren_cfg.get("bilingual_mode") or "side_by_side"),
        "strict": strict_mode,
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(all_sources) == 1:
        res = process_single_paper(
            all_sources[0],
            api_base=api_endpoints[0],
            out_dir=out_dir,
            ocr_req=ocr_req,
            llm_req=llm_req,
            render_base=render_base,
            target_lang=target_lang,
            summary=args.summary,
            cfg_path=cfg_path,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res.get("status") == "succeeded" else 3

    # Multi-paper parallel batch mode across api_endpoints pool
    ep_cycle = itertools.cycle(api_endpoints)
    results: list[dict[str, Any]] = []
    t_batch0 = time.time()
    workers = max(1, min(args.batch_workers, len(all_sources)))
    print(f"[miniretain-cli] Starting batch of {len(all_sources)} papers (workers={workers}, endpoints={api_endpoints})", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {
            ex.submit(
                process_single_paper,
                src,
                api_base=next(ep_cycle),
                out_dir=out_dir,
                ocr_req=ocr_req,
                llm_req=llm_req,
                render_base=render_base,
                target_lang=target_lang,
                summary=args.summary,
                cfg_path=cfg_path,
            ): src
            for src in all_sources
        }
        for fut in as_completed(fut_map):
            src = fut_map[fut]
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                results.append({"source": src, "status": "failed", "error": str(exc)})

    succeeded_cnt = sum(1 for r in results if r.get("status") == "succeeded")
    batch_summary = {
        "mode": "batch",
        "total_papers": len(all_sources),
        "succeeded": succeeded_cnt,
        "failed": len(all_sources) - succeeded_cnt,
        "wall_time_s": round(time.time() - t_batch0, 3),
        "results": results,
    }
    (out_dir / "batch_summary.json").write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(batch_summary, ensure_ascii=False, indent=2))
    return 0 if succeeded_cnt == len(all_sources) else 3


if __name__ == "__main__":
    raise SystemExit(main())
