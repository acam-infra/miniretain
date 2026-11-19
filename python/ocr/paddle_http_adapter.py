"""HTTP client OCR provider for a local PaddleOCR service (pixel-space xyxy contract)."""
from __future__ import annotations

import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pymupdf

from ocr.base import OCRBackendUnavailableError, OCRConfigError, OCRProvider
from ocr.normalize import cluster_figure_regions, normalize_pages


def _is_tailnet_or_local(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        return False


def _opener_for(url: str) -> urllib.request.OpenerDirector:
    if _is_tailnet_or_local(url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _detect_scanned_figure_regions(
    pix: pymupdf.Pixmap,
    text_boxes: list[list[float]],
) -> list[list[float]]:
    """Detect non-text graphic/figure regions on a rasterized scanned page.

    Downsamples to a coarse grid, masks out OCR text boxes, and clusters dense
    non-white pixel regions (diagrams, charts, plots) so embedded labels inside
    figures can be protected from destructive redaction.
    """
    w, h = pix.width, pix.height
    if w < 32 or h < 32:
        return []
    n_comp = pix.n
    samples = pix.samples
    # Sample on a coarse 8px grid for speed (pure Python, <15ms per page)
    step = 8
    gw = max(1, w // step)
    gh = max(1, h // step)
    # Build text mask in grid coordinates
    text_mask = [[False] * gw for _ in range(gh)]
    for b in text_boxes:
        gx0 = max(0, int(b[0] // step) - 1)
        gy0 = max(0, int(b[1] // step) - 1)
        gx1 = min(gw - 1, int(b[2] // step) + 1)
        gy1 = min(gh - 1, int(b[3] // step) + 1)
        for gy in range(gy0, gy1 + 1):
            row = text_mask[gy]
            for gx in range(gx0, gx1 + 1):
                row[gx] = True

    ink_rects: list[list[float]] = []
    stride = w * n_comp
    for gy in range(gh):
        py = min(h - 1, gy * step + step // 2)
        row_off = py * stride
        row_mask = text_mask[gy]
        for gx in range(gw):
            if row_mask[gx]:
                continue
            px = min(w - 1, gx * step + step // 2)
            idx = row_off + px * n_comp
            r = samples[idx]
            g = samples[idx + 1] if n_comp >= 3 else r
            b_ch = samples[idx + 2] if n_comp >= 3 else r
            # Non-white / colored / graphic pixel
            if r < 238 or g < 238 or b_ch < 238:
                ink_rects.append([gx * step, gy * step, (gx + 1) * step, (gy + 1) * step])

    if len(ink_rects) < 12:
        return []
    # Merge nearby non-text ink cells into candidate graphic regions
    merged = cluster_figure_regions(ink_rects, [], float(w), float(h), margin=step * 2.5)
    # Keep substantial figure/chart areas (>= 100x60 px at 150dpi)
    return [r for r in merged if (r[2] - r[0]) >= 100.0 and (r[3] - r[1]) >= 60.0]


class PaddleHTTPOCRProvider(OCRProvider):
    """Calls a local/tailnet PaddleOCR HTTP server (see python/ocr/paddle_server.py)."""

    name = "paddle-http"

    def __init__(
        self,
        base_url: str | None = None,
        dpi: float | None = None,
        timeout_s: float | None = None,
        health_timeout_s: float | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(config=config)
        sub = self.config.get("paddle_http") if isinstance(self.config.get("paddle_http"), dict) else {}
        cfg_url = sub.get("base_url") or self.config.get("base_url")
        cfg_dpi = sub.get("dpi") if "dpi" in sub else self.config.get("dpi")
        cfg_timeout = sub.get("timeout_s") if "timeout_s" in sub else self.config.get("timeout_s")
        cfg_health_timeout = (
            sub.get("health_timeout_s")
            if "health_timeout_s" in sub
            else self.config.get("health_timeout_s")
        )

        raw_url = base_url or cfg_url or os.environ.get("PADDLE_OCR_URL", "http://127.0.0.1:18085")
        self.base_url = str(raw_url).strip().rstrip("/")
        self.dpi = float(
            dpi if dpi is not None else (cfg_dpi if cfg_dpi is not None else os.environ.get("PADDLE_OCR_DPI", "150"))
        )
        self.timeout_s = float(
            timeout_s
            if timeout_s is not None
            else (cfg_timeout if cfg_timeout is not None else os.environ.get("PADDLE_OCR_TIMEOUT_S", "180"))
        )
        self.health_timeout_s = float(
            health_timeout_s
            if health_timeout_s is not None
            else (
                cfg_health_timeout
                if cfg_health_timeout is not None
                else os.environ.get("PADDLE_OCR_HEALTH_TIMEOUT_S", "5")
            )
        )
        if not self.base_url.startswith(("http://", "https://")):
            raise OCRConfigError(f"paddle-http base_url must start with http:// or https://, got {self.base_url!r}")
        if self.dpi <= 0:
            raise OCRConfigError(f"paddle-http dpi must be > 0, got {self.dpi!r}")
        if self.timeout_s <= 0:
            raise OCRConfigError(f"paddle-http timeout_s must be > 0, got {self.timeout_s!r}")
        self.service_info: dict = {}
        self.parse_seconds: float = 0.0

    def health(self, timeout_s: float | None = None) -> dict:
        use_timeout = float(timeout_s if timeout_s is not None else self.health_timeout_s)
        url = f"{self.base_url}/health"
        req = urllib.request.Request(url, method="GET")
        try:
            with _opener_for(url).open(req, timeout=use_timeout) as resp:
                info = json.loads(resp.read().decode("utf-8"))
                self.service_info = info
                return info
        except Exception as exc:  # noqa: BLE001
            raise OCRBackendUnavailableError(f"paddle-http health failed ({url}): {exc}") from exc

    def parse_page(self, page: pymupdf.Page, dpi: float | None = None) -> dict:
        use_dpi = float(dpi or self.dpi)
        pix = page.get_pixmap(dpi=int(use_dpi))
        png_bytes = pix.tobytes("png")
        url = f"{self.base_url}/ocr?dpi={use_dpi:.1f}&page_index={page.number}"
        req = urllib.request.Request(
            url,
            data=png_bytes,
            headers={"Content-Type": "image/png"},
            method="POST",
        )
        try:
            with _opener_for(url).open(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", errors="replace")
            raise OCRBackendUnavailableError(
                f"paddle-http HTTP {exc.code} on page {page.number}: {detail}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise OCRBackendUnavailableError(
                f"paddle-http request failed on page {page.number}: {exc}"
            ) from exc

        raw_blocks = data.get("blocks") or []
        blocks: list[dict] = []
        text_boxes: list[list[float]] = []
        for i, b in enumerate(raw_blocks):
            bbox = [float(v) for v in b["bbox"]]
            text_boxes.append(bbox)
            blocks.append({
                "block_id": f"p{page.number}b{i:04d}",
                "kind": "text",
                "text": b["text"],
                "bbox": bbox,
                "line_bboxes": [bbox],
            })

        # Detect figure regions from vector drawings/images (scaled to pixel space)
        # and, for scanned raster pages, from non-text graphic ink regions.
        scale = use_dpi / 72.0
        drawing_rects = [
            [dr["rect"][0] * scale, dr["rect"][1] * scale, dr["rect"][2] * scale, dr["rect"][3] * scale]
            for dr in page.get_drawings()
            if dr.get("rect")
        ]
        raw_dict = page.get_text("dict")
        image_rects = [
            [ib["bbox"][0] * scale, ib["bbox"][1] * scale, ib["bbox"][2] * scale, ib["bbox"][3] * scale]
            for ib in raw_dict.get("blocks", [])
            if ib.get("type") == 1 and ib.get("bbox")
        ]
        figure_regions = cluster_figure_regions(
            drawing_rects, image_rects, float(pix.width), float(pix.height), margin=18.0 * scale
        )
        if not figure_regions:
            figure_regions = _detect_scanned_figure_regions(pix, text_boxes)

        return {
            "page_index": page.number,
            "width": float(pix.width),
            "height": float(pix.height),
            "figure_regions": figure_regions,
            "is_scanned": True,
            "ocr_reliable": True,
            "blocks": blocks,
        }

    def parse(self, source: str) -> list[dict]:
        t0 = time.perf_counter()
        self.health()
        doc = pymupdf.open(source)
        raw_pages: list[dict] = []
        try:
            for page in doc:
                raw_pages.append(self.parse_page(page, dpi=self.dpi))
        finally:
            doc.close()
        self.parse_seconds = round(time.perf_counter() - t0, 3)
        return normalize_pages(raw_pages, coord_space="pixel", dpi=self.dpi)
