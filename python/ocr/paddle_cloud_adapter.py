"""PaddleOCR Cloud API (AI Studio PaddleOCR-VL-1.6) adapter with Multi-Key & IP/Proxy Pool rotation."""
from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pymupdf

from common.pool import (
    EndpointKeyProxyPool,
    PoolSlot,
    build_opener_for_proxy,
    load_lines_from_file_or_str,
)
from ocr.base import OCRBackendUnavailableError, OCRConfigError, OCRProvider
from ocr.normalize import cluster_figure_regions, normalize_pages
from ocr.pymupdf_adapter import _extract_table_regions

DEFAULT_PADDLE_JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_PADDLE_MODEL = "PaddleOCR-VL-1.6"
_TAG_RE = re.compile(r"<[^>]+>")


def _clean_html_text(raw: str) -> str:
    if not raw:
        return ""
    txt = html.unescape(_TAG_RE.sub(" ", raw))
    return " ".join(txt.split()).strip()


def _resolve_tokens(cfg: dict[str, Any], sub: dict[str, Any]) -> list[str]:
    token_file = str(
        sub.get("token_file")
        or os.environ.get("PADDLE_CLOUD_TOKEN_FILE", "")
        or (Path.home() / ".miniretain" / "paddle_cloud.key")
    ).strip()
    tokens = load_lines_from_file_or_str(
        explicit_list=sub.get("tokens") if isinstance(sub.get("tokens"), list) else None,
        explicit_value=str(sub.get("token") or cfg.get("paddle_token") or ""),
        file_path=token_file,
        env_var=str(sub.get("token_env") or "PADDLE_CLOUD_TOKEN"),
    )
    if not tokens:
        for alt_env in ("PADDLEOCR_TOKEN", "RETAIN_PADDLE_API_TOKEN"):
            if os.environ.get(alt_env, "").strip():
                tokens.extend(load_lines_from_file_or_str(env_var=alt_env))
    return tokens


class PaddleCloudOCRProvider(OCRProvider):
    """Calls PaddleOCR Cloud Job API (PaddleOCR-VL-1.6) with multi-token & proxy pool rotation."""

    name = "paddle-cloud"

    def __init__(
        self,
        job_url: str | None = None,
        model: str | None = None,
        token: str | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=config)
        sub = self.config.get("paddle_cloud") if isinstance(self.config.get("paddle_cloud"), dict) else {}
        opts = self.config.get("options") if isinstance(self.config.get("options"), dict) else {}
        merged_sub = {**opts, **sub}

        self.job_url = str(
            job_url
            or merged_sub.get("job_url")
            or os.environ.get("PADDLE_CLOUD_JOB_URL", DEFAULT_PADDLE_JOB_URL)
        ).strip().rstrip("/")
        self.model = str(
            model
            or merged_sub.get("model")
            or os.environ.get("PADDLE_CLOUD_MODEL", DEFAULT_PADDLE_MODEL)
        ).strip()
        self.tokens = [token] if token else _resolve_tokens(self.config, merged_sub)
        self.proxies = load_lines_from_file_or_str(
            explicit_list=merged_sub.get("proxies") if isinstance(merged_sub.get("proxies"), list) else None,
            explicit_value=str(merged_sub.get("proxy_pool") or ""),
            file_path=str(merged_sub.get("proxy_file") or os.environ.get("PADDLE_CLOUD_PROXY_FILE", "")),
            env_var="PADDLE_CLOUD_PROXY_POOL",
        ) or ["direct"]
        self.fallback_provider = str(
            merged_sub.get("fallback_provider") or os.environ.get("PADDLE_CLOUD_FALLBACK", "")
        ).strip()
        self.timeout_s = float(
            timeout_s
            if timeout_s is not None
            else (merged_sub.get("timeout_s") or os.environ.get("PADDLE_CLOUD_TIMEOUT_S", "300"))
        )
        self.poll_interval_s = float(
            poll_interval_s
            if poll_interval_s is not None
            else (merged_sub.get("poll_interval_s") or os.environ.get("PADDLE_CLOUD_POLL_INTERVAL_S", "3.0"))
        )
        if not self.job_url.startswith(("http://", "https://")):
            raise OCRConfigError(f"paddle-cloud job_url must be http(s), got {self.job_url!r}")
        if self.timeout_s <= 0:
            raise OCRConfigError(f"paddle-cloud timeout_s must be > 0, got {self.timeout_s!r}")

        self.pool = EndpointKeyProxyPool.from_config(
            default_base_url=self.job_url,
            default_model=self.model,
            api_keys=self.tokens or [""],
            proxies=self.proxies,
        )
        self.parse_seconds: float = 0.0
        self.last_job_id: str = ""
        self.service_info: dict[str, Any] = {}

    def _run_with_slot(self, slot: PoolSlot, source: str) -> tuple[list[dict], dict]:
        if not slot.api_key:
            raise OCRConfigError(
                "paddle-cloud token is not configured (set ocr.paddle_cloud.token_file, tokens list, PADDLE_CLOUD_TOKEN, or ~/.miniretain/paddle_cloud.key)"
            )
        opener = build_opener_for_proxy(slot.proxy)
        optional_payload = {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }
        headers = {
            "Authorization": f"bearer {slot.api_key}",
            "Accept": "application/json",
        }
        if source.startswith(("http://", "https://")):
            body = json.dumps({
                "fileUrl": source,
                "model": slot.model,
                "optionalPayload": optional_payload,
            }).encode("utf-8")
            req = urllib.request.Request(
                slot.base_url,
                data=body,
                headers={**headers, "Content-Type": "application/json"},
                method="POST",
            )
        else:
            file_path = Path(source)
            if not file_path.is_file():
                raise OCRConfigError(f"paddle-cloud source file not found: {source}")
            boundary = f"----MiniretainPaddleCloud{int(time.time() * 1000)}"
            parts: list[bytes] = []
            for k, v in (("model", slot.model), ("optionalPayload", json.dumps(optional_payload))):
                parts.append(
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8")
                )
            file_bytes = file_path.read_bytes()
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_path.name}\"\r\nContent-Type: application/pdf\r\n\r\n".encode("utf-8")
                + file_bytes
                + f"\r\n--{boundary}--\r\n".encode("utf-8")
            )
            req = urllib.request.Request(
                slot.base_url,
                data=b"".join(parts),
                headers={**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
        try:
            with opener.open(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode("utf-8", errors="replace")
            raise OCRBackendUnavailableError(f"paddle-cloud submit HTTP {exc.code} (slot={slot.slot_id}): {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OCRBackendUnavailableError(f"paddle-cloud submit failed (slot={slot.slot_id}): {exc}") from exc

        err_code = int(payload.get("errorCode", 0) or 0)
        if err_code != 0:
            raise OCRBackendUnavailableError(
                f"paddle-cloud submit error (slot={slot.slot_id}): code={err_code} msg={payload.get('errorMsg')}"
            )
        job_id = str((payload.get("data") or {}).get("jobId") or "").strip()
        if not job_id:
            raise OCRBackendUnavailableError(f"paddle-cloud returned empty jobId: {payload}")
        self.last_job_id = job_id

        deadline = time.monotonic() + self.timeout_s
        jsonl_url = ""
        while time.monotonic() < deadline:
            poll_req = urllib.request.Request(f"{slot.base_url}/{job_id}", headers=headers, method="GET")
            try:
                with opener.open(poll_req, timeout=60) as resp:
                    poll_payload = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise OCRBackendUnavailableError(f"paddle-cloud poll failed for {job_id}: {exc}") from exc

            data = poll_payload.get("data") or {}
            state = str(data.get("state") or "").strip()
            if state == "done":
                jsonl_url = str((data.get("resultUrl") or {}).get("jsonUrl") or "").strip()
                if not jsonl_url:
                    raise OCRBackendUnavailableError(f"paddle-cloud job {job_id} done without jsonUrl")
                break
            if state == "failed":
                raise OCRBackendUnavailableError(
                    f"paddle-cloud job {job_id} failed: {data.get('errorMsg') or 'unknown error'}"
                )
            time.sleep(max(0.5, self.poll_interval_s))

        if not jsonl_url:
            raise OCRBackendUnavailableError(f"paddle-cloud job {job_id} timed out after {self.timeout_s}s")

        dl_req = urllib.request.Request(jsonl_url, method="GET")
        try:
            with opener.open(dl_req, timeout=180) as resp:
                raw_text = resp.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            raise OCRBackendUnavailableError(f"paddle-cloud download jsonl failed: {exc}") from exc

        layout_results: list[dict] = []
        data_info: dict = {}
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            res = obj.get("result") or {}
            items = res.get("layoutParsingResults") or []
            if isinstance(items, list):
                layout_results.extend(items)
            if not data_info and isinstance(res.get("dataInfo"), dict):
                data_info = res["dataInfo"]
        return layout_results, data_info

    def parse(self, source: str) -> list[dict]:
        t0 = time.perf_counter()
        max_attempts = min(max(1, len(self.pool.slots)), 4)
        tried_slots: set[str] = set()
        last_exc: Exception | None = None
        layout_results: list[dict] | None = None
        data_info: dict = {}

        for _ in range(max_attempts):
            slot = self.pool.acquire(exclude_slot_ids=tried_slots)
            tried_slots.add(slot.slot_id)
            st0 = time.perf_counter()
            try:
                layout_results, data_info = self._run_with_slot(slot, source)
                self.pool.release_success(slot, time.perf_counter() - st0)
                break
            except OCRConfigError:
                self.pool.release_failure(slot, is_rate_limit=False, cooldown_s=0.0, error_msg="config_error")
                raise
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                is_rl = "429" in msg or "10010" in msg or "quota" in msg.lower() or "rate" in msg.lower()
                self.pool.release_failure(slot, is_rate_limit=is_rl, cooldown_s=20.0 if is_rl else 8.0, error_msg=msg)
                last_exc = exc

        if layout_results is None:
            if self.fallback_provider and self.fallback_provider != self.name:
                from ocr.adapter import get_provider
                fb = get_provider(self.fallback_provider, self.config)
                pages = fb.parse(source)
                self.service_info = {
                    "provider": self.name,
                    "fallback_triggered": self.fallback_provider,
                    "pool": self.pool.snapshot(),
                }
                self.parse_seconds = round(time.perf_counter() - t0, 3)
                return pages
            if last_exc is not None:
                raise last_exc
            raise OCRBackendUnavailableError("paddle-cloud pool exhausted")

        self.service_info = {
            "provider": self.name,
            "model": self.model,
            "job_id": self.last_job_id,
            "pool": self.pool.snapshot(),
        }

        doc = pymupdf.open(source)
        raw_pages: list[dict] = []
        pages_meta = data_info.get("pages") if isinstance(data_info.get("pages"), list) else []
        try:
            for page_idx, page in enumerate(doc):
                pdf_w = float(page.rect.width)
                pdf_h = float(page.rect.height)
                page_payload = layout_results[page_idx] if page_idx < len(layout_results) else {}
                pruned = page_payload.get("prunedResult") or {}
                meta_p = pages_meta[page_idx] if page_idx < len(pages_meta) and isinstance(pages_meta[page_idx], dict) else {}
                raw_w = float(pruned.get("width") or meta_p.get("width") or pdf_w)
                raw_h = float(pruned.get("height") or meta_p.get("height") or pdf_h)
                sx = (pdf_w / raw_w) if raw_w > 0 else 1.0
                sy = (pdf_h / raw_h) if raw_h > 0 else 1.0

                drawing_rects = [list(dr["rect"]) for dr in page.get_drawings() if dr.get("rect")]
                raw_dict = page.get_text("dict")
                image_rects = [
                    list(ib["bbox"])
                    for ib in raw_dict.get("blocks", [])
                    if ib.get("type") == 1 and ib.get("bbox")
                ]
                figure_regions = cluster_figure_regions(drawing_rects, image_rects, pdf_w, pdf_h)
                table_regions = _extract_table_regions(page, figure_regions)

                blocks: list[dict] = []
                parsing_res_list = pruned.get("parsing_res_list") or []
                for blk in parsing_res_list:
                    label = str(blk.get("block_label") or "text").strip().lower()
                    raw_bbox = blk.get("block_bbox") or [0, 0, 0, 0]
                    if len(raw_bbox) < 4:
                        continue
                    bbox = [
                        round(float(raw_bbox[0]) * sx, 2),
                        round(float(raw_bbox[1]) * sy, 2),
                        round(float(raw_bbox[2]) * sx, 2),
                        round(float(raw_bbox[3]) * sy, 2),
                    ]
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    raw_content = str(blk.get("block_content") or "").strip()

                    if label in {"image", "chart", "header_image", "footer_image", "seal"}:
                        figure_regions.append(bbox)
                        continue
                    if label == "table":
                        table_regions.append(bbox)
                        text_clean = _clean_html_text(raw_content) or "[TABLE]"
                        blocks.append({
                            "block_id": f"p{page_idx}b{len(blocks):04d}",
                            "kind": "text",
                            "text": text_clean,
                            "bbox": bbox,
                            "line_bboxes": [bbox],
                            "font_size_pt": 9.0,
                            "is_bold": False,
                            "is_math_font": False,
                        })
                        continue

                    text_clean = _clean_html_text(raw_content) if "<" in raw_content else " ".join(raw_content.split()).strip()
                    if not text_clean:
                        continue

                    bh = max(8.0, bbox[3] - bbox[1])
                    est_lines = max(1, round(bh / 12.0))
                    is_title = label == "doc_title"
                    is_heading = label == "paragraph_title"
                    is_formula = label in {"display_formula", "formula", "formula_number"}
                    font_sz = 14.0 if is_title else (11.0 if is_heading else min(10.5, max(7.5, bh / (est_lines * 1.18))))

                    blocks.append({
                        "block_id": f"p{page_idx}b{len(blocks):04d}",
                        "kind": "text",
                        "text": text_clean,
                        "bbox": bbox,
                        "line_bboxes": [bbox],
                        "font_size_pt": round(font_sz, 2),
                        "is_bold": is_title or is_heading,
                        "is_math_font": is_formula,
                    })

                raw_pages.append({
                    "page_index": page_idx,
                    "width": pdf_w,
                    "height": pdf_h,
                    "figure_regions": figure_regions,
                    "table_regions": table_regions,
                    "is_scanned": False,
                    "ocr_reliable": True,
                    "blocks": blocks,
                })
        finally:
            doc.close()

        self.parse_seconds = round(time.perf_counter() - t0, 3)
        return normalize_pages(raw_pages, coord_space="pdf_pt", dpi=72.0)
