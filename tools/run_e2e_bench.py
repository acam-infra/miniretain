#!/usr/bin/env python3
"""Reproducible end-to-end runner & benchmark for Miniretain Go server.

Starts the built Go server, submits jobs via POST /v1/jobs, polls GET /v1/jobs/{id},
fetches GET /v1/jobs/{id}/result, and records full per-stage & total wall timings
plus PDF geometry, figure pixel-preservation, and link checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http_json(method: str, url: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _no_proxy_opener().open(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, {"raw": raw}


def _verify_arxiv_pdfs(source_pdf: Path, cn_pdf: Path, sbs_pdf: Path) -> dict:
    with pymupdf.open(source_pdf) as sdoc, pymupdf.open(cn_pdf) as cdoc, pymupdf.open(sbs_pdf) as bdoc:
        cn_pages = [(round(p.rect.width, 2), round(p.rect.height, 2)) for p in cdoc]
        sbs_pages = [(round(p.rect.width, 2), round(p.rect.height, 2)) for p in bdoc]

        # Page 2 (index 1) Fig. 1 region pixel diff
        fig1_clip = pymupdf.Rect(50, 75, 560, 290)
        sp2 = sdoc[1].get_pixmap(dpi=100, clip=fig1_clip)
        cp2 = cdoc[1].get_pixmap(dpi=100, clip=fig1_clip)
        fig1_diff = sum(1 for a, b in zip(sp2.samples, cp2.samples) if a != b)

        # Page 4 (index 3) vertical axis labels region pixel diff
        p4_clips = [
            pymupdf.Rect(42, 70, 70, 210),
            pymupdf.Rect(306, 70, 332, 210),
        ]
        p4_axis_diff = 0
        for clip in p4_clips:
            sa = sdoc[3].get_pixmap(dpi=100, clip=clip)
            ca = cdoc[3].get_pixmap(dpi=100, clip=clip)
            p4_axis_diff += sum(1 for a, b in zip(sa.samples, ca.samples) if a != b)

        # Page 7 (index 6) references [1]-[13] and URI links
        p7_text = cdoc[6].get_text()
        refs_present = [i for i in range(1, 14) if f"[{i}]" in p7_text]
        src_uris = sorted({lk["uri"] for p in sdoc for lk in p.get_links() if lk.get("uri")})
        cn_uris = sorted({lk["uri"] for p in cdoc for lk in p.get_links() if lk.get("uri")})
        sbs_left_uris = sorted({
            lk["uri"] for p in bdoc for lk in p.get_links()
            if lk.get("uri") and lk["from"].x0 < p.rect.width / 2
        })
        sbs_right_uris = sorted({
            lk["uri"] for p in bdoc for lk in p.get_links()
            if lk.get("uri") and lk["from"].x0 >= p.rect.width / 2
        })

        return {
            "cn_page_count": len(cdoc),
            "cn_page_sizes": sorted(set(cn_pages)),
            "sbs_page_count": len(bdoc),
            "sbs_page_sizes": sorted(set(sbs_pages)),
            "page2_fig1_diff_bytes": fig1_diff,
            "page4_axis_diff_bytes": p4_axis_diff,
            "page7_refs_count": len(refs_present),
            "page7_refs_missing": [i for i in range(1, 14) if i not in refs_present],
            "src_unique_uris": len(src_uris),
            "cn_unique_uris": len(cn_uris),
            "sbs_left_unique_uris": len(sbs_left_uris),
            "sbs_right_unique_uris": len(sbs_right_uris),
            "uris_match_source": (cn_uris == src_uris and sbs_left_uris == src_uris and sbs_right_uris == src_uris),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="miniretain repo root")
    ap.add_argument("--python", required=True, help="python executable for worker")
    ap.add_argument("--pdf", required=True, help="source PDF path")
    ap.add_argument("--port", type=int, default=18098)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--ocr-provider", default="auto")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--batch-size", type=int, default=14)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--verify-arxiv", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pdf = Path(args.pdf).resolve()
    bin_path = root / "data" / "miniretain-server"
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(["go", "build", "-o", str(bin_path), "./cmd/server"], cwd=root, check=True)

    srv_cmd = [str(bin_path), "-addr", f"127.0.0.1:{args.port}", "-root", str(root), "-python", args.python]
    log_path = root / "data" / f"server_{args.port}.log"
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(srv_cmd, cwd=root, stdout=log_fh, stderr=subprocess.STDOUT)

    try:
        base_url = f"http://127.0.0.1:{args.port}"
        for _ in range(50):
            try:
                code, _ = _http_json("GET", f"{base_url}/health", timeout=2.0)
                if code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("miniretain server failed to become healthy")

        sha256 = hashlib.sha256(pdf.read_bytes()).hexdigest()
        req_body = {
            "source_path": str(pdf),
            "target_lang": "zh",
            "ocr": {"provider": args.ocr_provider},
            "llm": {
                "provider": "openai-compat",
                "batch_size": args.batch_size,
                "concurrency": args.concurrency,
                "max_retries": 2,
                "timeout_ms": 120000,
                "model": args.model,
                "allow_partial": False,
            },
            "render": {
                "engine": "typst",
                "layout_mode": "preserve",
                "bilingual_mode": "side_by_side",
                "output_cn": f"{pdf.stem}_CN.pdf",
                "output_merged": f"{pdf.stem}_Side-by-Side.pdf",
                "strict": True,
            },
        }

        runs_out = []
        for r_idx in range(1, args.runs + 1):
            t0 = time.perf_counter()
            post_code, post_resp = _http_json("POST", f"{base_url}/v1/jobs", req_body, timeout=30.0)
            if post_code != 202:
                raise RuntimeError(f"POST /v1/jobs failed: HTTP {post_code} {post_resp}")
            jid = post_resp["id"]

            get_code, job_resp = 0, {}
            for _ in range(300):
                get_code, job_resp = _http_json("GET", f"{base_url}/v1/jobs/{jid}", timeout=10.0)
                st = job_resp.get("status")
                if st in ("succeeded", "failed"):
                    break
                time.sleep(0.25)
            res_code, res_resp = _http_json("GET", f"{base_url}/v1/jobs/{jid}/result", timeout=10.0)
            total_s = round(time.perf_counter() - t0, 3)

            run_rec = {
                "run": r_idx,
                "job_id": jid,
                "status": job_resp.get("status"),
                "post_http": post_code,
                "get_job_http": get_code,
                "get_result_http": res_code,
                "total_wall_s": total_s,
                "job": job_resp,
                "result": res_resp,
            }
            if job_resp.get("status") == "succeeded" and args.verify_arxiv:
                r_stage = (res_resp.get("result") or {}).get("render") or {}
                run_rec["verify"] = _verify_arxiv_pdfs(
                    pdf, Path(r_stage["output_cn"]), Path(r_stage["output_merged"])
                )
            runs_out.append(run_rec)
            ocr_ms = ((job_resp.get("result") or {}).get("ocr") or {}).get("stage_wall_ms")
            llm_ms = ((job_resp.get("result") or {}).get("llm") or {}).get("stage_wall_ms")
            ren_ms = ((job_resp.get("result") or {}).get("render") or {}).get("stage_wall_ms")
            print(
                f"[run {r_idx}/{args.runs}] job={jid} status={job_resp.get('status')} "
                f"total={total_s:.2f}s (ocr={ocr_ms}ms llm={llm_ms}ms render={ren_ms}ms)",
                flush=True,
            )

        ok_runs = [r for r in runs_out if r["status"] == "succeeded"]
        summary = {
            "source_pdf": str(pdf),
            "source_sha256": sha256,
            "start_cmd": " ".join(srv_cmd),
            "request_redacted": req_body,
            "runs_total": len(runs_out),
            "runs_succeeded": len(ok_runs),
            "median_total_s": round(statistics.median([r["total_wall_s"] for r in ok_runs]), 3) if ok_runs else None,
            "runs": runs_out,
        }
        Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        log_fh.close()


if __name__ == "__main__":
    main()
