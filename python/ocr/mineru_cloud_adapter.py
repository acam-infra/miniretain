"""MinerU Cloud API v4 (https://mineru.net/api/v4) adapter with Multi-Key & IP/Proxy Pool rotation."""
from __future__ import annotations

import html
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
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

DEFAULT_MINERU_BASE_URL = "https://mineru.net"
DEFAULT_MINERU_MODEL_VERSION = "vlm"
_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9\-]*(?:\s+[^<>]*)?>")


_BOLD_TAG_RE = re.compile(r"<(?:b|strong)\b[^>]*>(.*?)</(?:b|strong)>", re.IGNORECASE | re.DOTALL)
_ITALIC_TAG_RE = re.compile(r"<(?:i|em)\b[^>]*>(.*?)</(?:i|em)>", re.IGNORECASE | re.DOTALL)


def _clean_html_text(raw: str) -> str:
    if not raw:
        return ""
    s = _BOLD_TAG_RE.sub(lambda m: f"**{m.group(1).strip()}**" if m.group(1).strip() else "", raw)
    s = _ITALIC_TAG_RE.sub(lambda m: f"*{m.group(1).strip()}*" if m.group(1).strip() else "", s)
    txt = html.unescape(_TAG_RE.sub(" ", s))
    out = " ".join(txt.split()).strip()
    out = re.sub(r"\*\*\s+\*\*", " ", out)
    out = re.sub(r"(?<!\*)\*(?!\*)\s+(?<!\*)\*(?!\*)", " ", out)
    return out


def _resolve_mineru_tokens(cfg: dict[str, Any], sub: dict[str, Any]) -> list[str]:
    token_file = str(
        sub.get("token_file")
        or os.environ.get("MINERU_TOKEN_FILE", "")
        or (Path.home() / ".miniretain" / "mineru.key")
    ).strip()
    tokens = load_lines_from_file_or_str(
        explicit_list=sub.get("tokens") if isinstance(sub.get("tokens"), list) else None,
        explicit_value=str(sub.get("token") or cfg.get("mineru_token") or ""),
        file_path=token_file,
        env_var=str(sub.get("token_env") or "MINERU_TOKEN"),
    )
    if not tokens and os.environ.get("MINERU_API_TOKEN", "").strip():
        tokens.extend(load_lines_from_file_or_str(env_var="MINERU_API_TOKEN"))
    return tokens


class MinerUCloudOCRProvider(OCRProvider):
    """Calls MinerU Cloud API v4 (vlm / pipeline) with multi-token & proxy pool rotation."""

    name = "mineru"

    def __init__(
        self,
        base_url: str | None = None,
        model_version: str | None = None,
        language: str | None = None,
        token: str | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=config)
        sub = self.config.get("mineru") if isinstance(self.config.get("mineru"), dict) else {}
        opts = self.config.get("options") if isinstance(self.config.get("options"), dict) else {}
        merged_sub = {**opts, **sub}

        self.base_url = str(
            base_url
            or merged_sub.get("base_url")
            or os.environ.get("MINERU_BASE_URL", DEFAULT_MINERU_BASE_URL)
        ).strip().rstrip("/")
        self.model_version = str(
            model_version
            or merged_sub.get("model_version")
            or os.environ.get("MINERU_MODEL_VERSION", DEFAULT_MINERU_MODEL_VERSION)
        ).strip()
        self.language = str(
            language
            or merged_sub.get("language")
            or os.environ.get("MINERU_LANGUAGE", "en")
        ).strip()
        self.tokens = [token] if token else _resolve_mineru_tokens(self.config, merged_sub)
        self.proxies = load_lines_from_file_or_str(
            explicit_list=merged_sub.get("proxies") if isinstance(merged_sub.get("proxies"), list) else None,
            explicit_value=str(merged_sub.get("proxy_pool") or ""),
            file_path=str(merged_sub.get("proxy_file") or os.environ.get("MINERU_PROXY_FILE", "")),
            env_var="MINERU_PROXY_POOL",
        ) or ["direct"]
        self.fallback_provider = str(
            merged_sub.get("fallback_provider") or os.environ.get("MINERU_FALLBACK", "")
        ).strip()
        self.timeout_s = float(
            timeout_s
            if timeout_s is not None
            else (merged_sub.get("timeout_s") or os.environ.get("MINERU_TIMEOUT_S", "600"))
        )
        self.poll_interval_s = float(
            poll_interval_s
            if poll_interval_s is not None
            else (merged_sub.get("poll_interval_s") or os.environ.get("MINERU_POLL_INTERVAL_S", "3.0"))
        )
        if not self.base_url.startswith(("http://", "https://")):
            raise OCRConfigError(f"mineru base_url must start with http(s), got {self.base_url!r}")
        if self.timeout_s <= 0:
            raise OCRConfigError(f"mineru timeout_s must be > 0, got {self.timeout_s!r}")

        self.pool = EndpointKeyProxyPool.from_config(
            default_base_url=self.base_url,
            default_model=self.model_version,
            api_keys=self.tokens or [""],
            proxies=self.proxies,
        )
        self.parse_seconds: float = 0.0
        self.last_batch_id: str = ""
        self.service_info: dict[str, Any] = {}

    def _api_request(
        self,
        opener: urllib.request.OpenerDirector,
        slot: PoolSlot,
        method: str,
        url: str,
        payload: dict | None = None,
        timeout: int = 60,
    ) -> dict:
        if not slot.api_key:
            raise OCRConfigError(
                "mineru token is not configured (set ocr.mineru.token_file, tokens list, MINERU_TOKEN, or ~/.miniretain/mineru.key)"
            )
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Authorization": f"Bearer {slot.api_key}",
        }
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with opener.open(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode("utf-8", errors="replace")
            raise OCRBackendUnavailableError(f"mineru HTTP {exc.code} (slot={slot.slot_id}, {url}): {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OCRBackendUnavailableError(f"mineru request failed (slot={slot.slot_id}, {url}): {exc}") from exc

        if int(out.get("code", -1)) != 0:
            raise OCRBackendUnavailableError(
                f"mineru API error (slot={slot.slot_id}): code={out.get('code')} msg={out.get('msg')}"
            )
        return out

    def _run_with_slot(self, slot: PoolSlot, source: str) -> dict:
        opener = build_opener_for_proxy(slot.proxy)
        if source.startswith(("http://", "https://")):
            res = self._api_request(
                opener,
                slot,
                "POST",
                f"{slot.base_url}/api/v4/extract/task",
                {
                    "url": source,
                    "model_version": slot.model,
                    "is_ocr": True,
                    "enable_formula": True,
                    "enable_table": True,
                    "language": self.language,
                },
            )
            task_id = str((res.get("data") or {}).get("task_id") or "").strip()
            if not task_id:
                raise OCRBackendUnavailableError(f"mineru returned empty task_id: {res}")
            self.last_batch_id = task_id
            deadline = time.monotonic() + self.timeout_s
            zip_url = ""
            while time.monotonic() < deadline:
                st_res = self._api_request(opener, slot, "GET", f"{slot.base_url}/api/v4/extract/task/{task_id}")
                d = st_res.get("data") or {}
                state = str(d.get("state") or "").strip()
                if state == "done":
                    zip_url = str(d.get("full_zip_url") or "").strip()
                    if not zip_url:
                        raise OCRBackendUnavailableError(f"mineru task {task_id} done without full_zip_url")
                    break
                if state == "failed":
                    raise OCRBackendUnavailableError(f"mineru task {task_id} failed: {d.get('err_msg')}")
                time.sleep(max(0.5, self.poll_interval_s))
            if not zip_url:
                raise OCRBackendUnavailableError(f"mineru task {task_id} timed out after {self.timeout_s}s")
        else:
            file_path = Path(source)
            if not file_path.is_file():
                raise OCRConfigError(f"mineru source file not found: {source}")

            batch_res = self._api_request(
                opener,
                slot,
                "POST",
                f"{slot.base_url}/api/v4/file-urls/batch",
                {
                    "enable_formula": True,
                    "enable_table": True,
                    "language": self.language,
                    "model_version": slot.model,
                    "files": [{"name": file_path.name, "is_ocr": True, "data_id": file_path.stem[:32]}],
                },
            )
            bdata = batch_res.get("data") or {}
            batch_id = str(bdata.get("batch_id") or "").strip()
            file_urls = bdata.get("file_urls") or []
            if not batch_id or not file_urls:
                raise OCRBackendUnavailableError(f"mineru batch upload URL missing: {batch_res}")
            self.last_batch_id = batch_id

            import requests
            sess = requests.Session()
            if not slot.proxy or slot.proxy.lower() in ("direct", "none", "noproxy"):
                sess.trust_env = False
            else:
                sess.trust_env = False
                sess.proxies = {"http": slot.proxy, "https": slot.proxy}
            try:
                put_resp = sess.put(file_urls[0], data=file_path.read_bytes(), timeout=180)
                if put_resp.status_code not in (200, 201, 204):
                    raise RuntimeError(f"OSS PUT HTTP {put_resp.status_code}: {put_resp.text[:200]}")
            except Exception as exc:  # noqa: BLE001
                raise OCRBackendUnavailableError(f"mineru OSS PUT failed (slot={slot.slot_id}): {exc}") from exc

            deadline = time.monotonic() + self.timeout_s
            zip_url = ""
            while time.monotonic() < deadline:
                st_res = self._api_request(
                    opener, slot, "GET", f"{slot.base_url}/api/v4/extract-results/batch/{batch_id}"
                )
                ext_list = (st_res.get("data") or {}).get("extract_result") or []
                if ext_list:
                    item = ext_list[0]
                    state = str(item.get("state") or "").strip()
                    if state == "done":
                        zip_url = str(item.get("full_zip_url") or "").strip()
                        if not zip_url:
                            raise OCRBackendUnavailableError(f"mineru batch {batch_id} done without full_zip_url")
                        break
                    if state == "failed":
                        raise OCRBackendUnavailableError(f"mineru batch {batch_id} failed: {item.get('err_msg')}")
                time.sleep(max(0.5, self.poll_interval_s))
            if not zip_url:
                raise OCRBackendUnavailableError(f"mineru batch {batch_id} timed out after {self.timeout_s}s")

        dl_req = urllib.request.Request(zip_url, method="GET")
        try:
            with opener.open(dl_req, timeout=180) as resp:
                zip_bytes = resp.read()
        except Exception as exc:  # noqa: BLE001
            raise OCRBackendUnavailableError(f"mineru zip download failed: {exc}") from exc

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = zf.namelist()
            if "layout.json" in names:
                return json.loads(zf.read("layout.json").decode("utf-8"))
            for name in names:
                if name.endswith("layout.json"):
                    return json.loads(zf.read(name).decode("utf-8"))
        raise OCRBackendUnavailableError("mineru result zip did not contain layout.json")

    @staticmethod
    def _split_cross_column_lines(lines: list[dict], sx: float, sy: float, pdf_w: float) -> list[list[dict]]:
        """Split a MinerU para_block's lines when they jump across columns (e.g. bottom of col 1 -> top of col 2)."""
        if len(lines) < 2:
            return [lines] if lines else []
        groups: list[list[dict]] = []
        cur: list[dict] = []
        prev_x0: float | None = None
        prev_y0: float | None = None
        for ln in lines:
            lb = ln.get("bbox")
            if isinstance(lb, list) and len(lb) >= 4:
                x0 = float(lb[0]) * sx
                y0 = float(lb[1]) * sy
                if cur and prev_x0 is not None and prev_y0 is not None:
                    if abs(x0 - prev_x0) > pdf_w * 0.25 or y0 < prev_y0 - 24.0:
                        groups.append(cur)
                        cur = [ln]
                        prev_x0, prev_y0 = x0, y0
                        continue
                prev_x0, prev_y0 = x0, y0
            cur.append(ln)
        if cur:
            groups.append(cur)
        return groups

    @staticmethod
    def _extract_lines_and_text(blk: dict, sx: float, sy: float) -> tuple[str, list[list[float]]]:
        chunks: list[str] = []
        line_bboxes: list[list[float]] = []
        for line in blk.get("lines") or []:
            lb = line.get("bbox")
            if isinstance(lb, list) and len(lb) >= 4:
                line_bboxes.append([
                    round(float(lb[0]) * sx, 2),
                    round(float(lb[1]) * sy, 2),
                    round(float(lb[2]) * sx, 2),
                    round(float(lb[3]) * sy, 2),
                ])
            line_parts: list[str] = []
            raw_spans = list(line.get("spans") or [])
            flat_spans = []
            for sp in raw_spans:
                if isinstance(sp, dict) and str(sp.get("type") or "").strip().lower() == "hyperlink" and isinstance(sp.get("children"), list):
                    flat_spans.extend(c for c in sp["children"] if isinstance(c, dict))
                elif isinstance(sp, dict):
                    flat_spans.append(sp)
            for span in flat_spans:
                stype = str(span.get("type") or "text").strip().lower()
                content = str(span.get("content") or "").strip()
                if not content and span.get("html"):
                    content = _clean_html_text(str(span.get("html") or ""))
                if not content:
                    continue
                if stype == "inline_equation":
                    clean_eq = content.strip("$").strip()
                    if clean_eq:
                        line_parts.append(f"${clean_eq}$")
                elif stype in {"interline_equation", "equation"}:
                    clean_eq = content.strip("$").strip()
                    if clean_eq:
                        line_parts.append(f"$${clean_eq}$$")
                else:
                    line_parts.append(content)
            if line_parts:
                chunks.append(" ".join(line_parts))
        return " ".join(chunks).strip(), line_bboxes

    def parse(self, source: str) -> list[dict]:
        t0 = time.perf_counter()
        max_attempts = min(max(1, len(self.pool.slots)), 4)
        tried_slots: set[str] = set()
        last_exc: Exception | None = None
        layout: dict | None = None

        for _ in range(max_attempts):
            slot = self.pool.acquire(exclude_slot_ids=tried_slots)
            tried_slots.add(slot.slot_id)
            st0 = time.perf_counter()
            try:
                layout = self._run_with_slot(slot, source)
                self.pool.release_success(slot, time.perf_counter() - st0)
                break
            except OCRConfigError:
                self.pool.release_failure(slot, is_rate_limit=False, cooldown_s=0.0, error_msg="config_error")
                raise
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                is_rl = "429" in msg or "quota" in msg.lower() or "limit" in msg.lower()
                self.pool.release_failure(slot, is_rate_limit=is_rl, cooldown_s=20.0 if is_rl else 8.0, error_msg=msg)
                last_exc = exc

        if layout is None:
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
            raise OCRBackendUnavailableError("mineru pool exhausted")

        self.service_info = {
            "provider": self.name,
            "model_version": self.model_version,
            "batch_id": self.last_batch_id,
            "pool": self.pool.snapshot(),
        }

        pdf_info = layout.get("pdf_info") if isinstance(layout.get("pdf_info"), list) else []
        doc = pymupdf.open(source)
        raw_pages: list[dict] = []
        try:
            for page_idx, page in enumerate(doc):
                pdf_w = float(page.rect.width)
                pdf_h = float(page.rect.height)
                p_info = pdf_info[page_idx] if page_idx < len(pdf_info) and isinstance(pdf_info[page_idx], dict) else {}
                psize = p_info.get("page_size") if isinstance(p_info.get("page_size"), list) else []
                raw_w = float(psize[0]) if len(psize) >= 1 and float(psize[0]) > 0 else pdf_w
                raw_h = float(psize[1]) if len(psize) >= 2 and float(psize[1]) > 0 else pdf_h
                sx = pdf_w / raw_w
                sy = pdf_h / raw_h

                drawing_rects = [list(dr["rect"]) for dr in page.get_drawings() if dr.get("rect")]
                raw_dict = page.get_text("dict")
                image_rects = [
                    list(ib["bbox"])
                    for ib in raw_dict.get("blocks", [])
                    if ib.get("type") == 1 and ib.get("bbox")
                ]
                figure_regions = cluster_figure_regions(drawing_rects, image_rects, pdf_w, pdf_h)
                table_regions = _extract_table_regions(page, drawing_rects)
                blocks: list[dict] = []

                def add_leaf(leaf: dict, parent_type: str = "") -> None:
                    btype = str(leaf.get("type") or parent_type or "text").strip().lower()
                    rb = leaf.get("bbox")
                    if not isinstance(rb, list) or len(rb) < 4:
                        return
                    bbox = [
                        round(float(rb[0]) * sx, 2),
                        round(float(rb[1]) * sy, 2),
                        round(float(rb[2]) * sx, 2),
                        round(float(rb[3]) * sy, 2),
                    ]
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        return

                    if btype in {"image", "image_body"}:
                        figure_regions.append(bbox)
                        return
                    if btype in {"table_body"}:
                        table_regions.append(bbox)
                        txt, lbs = self._extract_lines_and_text(leaf, sx, sy)
                        blocks.append({
                            "block_id": f"p{page_idx}b{len(blocks):04d}",
                            "kind": "text",
                            "text": txt or "[TABLE]",
                            "bbox": bbox,
                            "line_bboxes": lbs or [bbox],
                            "font_size_pt": 9.0,
                            "is_bold": False,
                            "is_math_font": False,
                        })
                        return

                    raw_lines = leaf.get("lines") or []
                    line_groups = self._split_cross_column_lines(raw_lines, sx, sy, pdf_w) if len(raw_lines) >= 2 else [raw_lines]
                    if len(line_groups) > 1:
                        for lg in line_groups:
                            sub_lbs = [
                                [round(float(l["bbox"][0]) * sx, 2), round(float(l["bbox"][1]) * sy, 2), round(float(l["bbox"][2]) * sx, 2), round(float(l["bbox"][3]) * sy, 2)]
                                for l in lg if isinstance(l.get("bbox"), list) and len(l["bbox"]) >= 4
                            ]
                            if not sub_lbs:
                                continue
                            sub_bbox = [
                                min(b[0] for b in sub_lbs),
                                min(b[1] for b in sub_lbs),
                                max(b[2] for b in sub_lbs),
                                max(b[3] for b in sub_lbs),
                            ]
                            add_leaf({**leaf, "lines": lg, "bbox": [sub_bbox[0] / sx, sub_bbox[1] / sy, sub_bbox[2] / sx, sub_bbox[3] / sy]}, parent_type=parent_type)
                        return
                    txt, lbs = self._extract_lines_and_text(leaf, sx, sy)
                    if not txt:
                        return
                    bh = max(8.0, bbox[3] - bbox[1])
                    n_lines = max(1, len(lbs) or round(bh / 12.0))
                    if btype in {"header", "footer", "page_number", "page_header", "page_footer", "aside_text"}:
                        blocks.append({
                            "block_id": f"p{page_idx}b{len(blocks):04d}",
                            "kind": "header_footer",
                            "text": txt,
                            "bbox": bbox,
                            "line_bboxes": lbs or [bbox],
                            "font_size_pt": 8.5,
                            "is_bold": False,
                            "is_math_font": False,
                            "policy": {"translate": False, "preserve_reason": "header_footer", "redact_source": False},
                        })
                        return
                    is_title = btype == "title" and int(leaf.get("level", 1) or 1) == 1 and page_idx == 0 and bbox[1] < pdf_h * 0.25
                    is_heading = btype == "title" and not is_title
                    is_formula = btype in {"interline_equation", "equation", "formula"}
                    font_sz = 14.0 if is_title else (11.0 if is_heading else min(10.5, max(7.5, bh / (n_lines * 1.18))))
                    # Probe native PDF span font size inside bbox when digital text layer is available
                    try:
                        clip_d = page.get_text("dict", clip=pymupdf.Rect(bbox))
                        sp_szs = [
                            float(sp["size"])
                            for b_d in clip_d.get("blocks", [])
                            if b_d.get("type") == 0
                            for l_d in b_d.get("lines", [])
                            for sp in l_d.get("spans", [])
                            if (sp.get("text") or "").strip() and sp.get("size") and float(sp["size"]) >= 6.0
                        ]
                        if sp_szs:
                            sp_szs.sort()
                            font_sz = sp_szs[len(sp_szs) // 2]
                    except Exception:
                        pass
                    blk_entry = {
                        "block_id": f"p{page_idx}b{len(blocks):04d}",
                        "kind": "formula" if is_formula else "text",
                        "text": txt,
                        "bbox": bbox,
                        "line_bboxes": lbs or [bbox],
                        "font_size_pt": round(font_sz, 2),
                        "is_bold": is_title or is_heading,
                        "is_math_font": is_formula,
                    }
                    if is_formula:
                        blk_entry["policy"] = {"translate": False, "preserve_reason": "formula", "redact_source": False}
                    blocks.append(blk_entry)

                for blk in p_info.get("para_blocks") or []:
                    children = blk.get("blocks")
                    if isinstance(children, list) and children:
                        for child in children:
                            add_leaf(child, parent_type=str(blk.get("type") or ""))
                    else:
                        add_leaf(blk)

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
