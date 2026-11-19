# Miniretain OCR 后端扩展与契约说明

Miniretain 将 OCR / PDF 解析层收敛在 `python/ocr/`，通过统一的 `OCRProvider` 接口、`register_provider` 注册表与 `document.v1` 输出契约解耦下游翻译和 Typst/PyMuPDF 渲染。

---

## 1. 核心模块与注册机制

| 文件 | 职责 |
|---|---|
| `python/ocr/base.py` | 抽象基类 `OCRProvider`、异常体系（`OCRError` / `OCRConfigError` / `OCRBackendUnavailableError` / `OCRContractError`）以及输出校验器 `validate_document_pages()` |
| `python/ocr/adapter.py` | 提供商注册表 `REGISTRY`、注册函数 `register_provider()`、`unregister_provider()`、`list_providers()` 与工厂解析函数 `get_provider(name, config=None)` |
| `python/ocr/normalize.py` | `normalize_pages()` 与 `cluster_figure_regions()`：统一分栏排序、角色分类、策略生成与跨栏/跨页续接 |
| `python/ocr/pymupdf_adapter.py` | 内置 `pymupdf` 与 `auto` 后端 |
| `python/ocr/paddle_http_adapter.py` | 内置 `paddle-http` 后端（对接 `python/ocr/paddle_server.py`） |
| `python/ocr/paddle_adapter.py` | 内置 `paddle-cpu` 懒加载本地进程内后端 |

### 1.1 注册与配置传递

```python
from ocr.adapter import get_provider, list_providers, register_provider

# 查询所有已注册后端
print(list_providers())
# ['auto', 'paddle-cpu', 'paddle-http', 'pymupdf', 'stub']

# 无额外配置时返回注册表默认实例（兼容旧调用方）
provider = get_provider("pymupdf")

# 带配置字典时由注册工厂构造独立实例，不污染全局单例
provider = get_provider(
    "auto",
    config={
        "paddle_http": {"base_url": "http://127.0.0.1:18085", "dpi": 150, "timeout_s": 180},
        "auto": {"scanned_char_threshold": 20, "fallback_provider": "paddle-http"},
        "options": {},
    },
)
```

在单 YAML 启动模式下，Go 服务端会将 YAML 的 `ocr.paddle_http`、`ocr.paddle_cpu`、`ocr.auto` 与 `ocr.options` 序列化进 `spec.ocr.config`（遵循 `contracts/task.spec.schema.json`），`python/worker.py:run_ocr()` 自动将其传给 `get_provider(spec["ocr"]["provider"], config=spec["ocr"].get("config"))`。

> **延迟实例化约定**：`register_provider(name, MyProviderClass)` 仅保存工厂引用，**不会在模块导入/注册阶段调用 `MyProviderClass(None)`**。因此自定义后端可以安全地在 `__init__(self, config=None)` 中校验必填参数（如 `ocr.options.endpoint`），只有该后端被实际选中并调用 `get_provider(name, config=...)` 时才会实例化，不会影响 worker 导入或其他内置后端。

---

## 2. `document.v1` 输出契约

`OCRProvider.parse(source: str) -> list[dict]` 必须返回经 `normalize_pages(...)` 规范化后的页面列表。`worker.py` 在落盘 `document.v1.json` 之前会调用 `validate_document_pages(pages)` 强制校验。

### 2.1 页面级字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `page_index` | `int` | **0 起始**页号（`0, 1, 2, ...`） |
| `width` / `height` | `float` (> 0) | 当前坐标空间下的页面宽高 |
| `coord_space` | `"pdf_pt"` \| `"pixel"` | 坐标空间：`pdf_pt` 为 PDF 点坐标（左上角原点，72 DPI）；`pixel` 为光栅像素坐标（左上角原点，按 `dpi` 缩放） |
| `dpi` | `float` (> 0) | `pdf_pt` 固定为 `72.0`；`pixel` 为渲染光栅图的实际 DPI（默认 `150.0`）。下游渲染器统一按 `72.0 / dpi` 或 `pdf_width / page_width` 换算回 PDF 点坐标 |
| `is_scanned` | `bool` | 是否判定为扫描页 |
| `ocr_reliable` | `bool` | OCR 结果是否可靠；若为 `false`，规范化层会把该页块标为 `layout_role="unreliable"` 且 `policy.translate=false` 以保护源内容 |
| `figure_regions` | `list[[x0,y0,x1,y1]]` | 图表/矢量图/插图聚类包围框，用于在规范化阶段保护图内标签 |
| `blocks` | `list[dict]` | 按阅读顺序排列的文本块列表 |

### 2.2 块级字段与角色/策略

| 字段 | 类型 | 说明 |
|---|---|---|
| `block_id` | `str` | 页内唯一块 ID（如 `p0b0001`） |
| `page_index` | `int` | 所属页号（0 起始） |
| `reading_order` | `int` | 页内阅读顺序（双栏页自动按横跨带 + 左栏→右栏排序） |
| `layout_role` | `str` | 版面角色：`title`、`heading`、`body`、`caption`、`figure_label`、`formula`、`header_footer`、`reference`、`empty`、`unreliable` |
| `text` | `str` | 清洗后的块文本 |
| `bbox` | `[x0,y0,x1,y1]` \| `None` | 左上到右下的非反转包围框（`x0 <= x1, y0 <= y1`） |
| `line_bboxes` | `list[[x0,y0,x1,y1]]` \| `None` | 块内逐行包围框 |
| `geometry` | `dict` | `{bbox, line_bboxes, coord_space, dpi, column_index}` |
| `content` | `dict` | `{text, font_size_pt, is_bold, is_math_font}` |
| `policy` | `dict` | `{translate: bool, preserve_reason: str\|None, redact_source: bool}` |
| `continuation` | `dict` | `{continued_from: str\|None, continues_to: str\|None}` |

**角色与保护策略**：
- `figure_label`（落入 `figure_regions` 且非图注，或竖排图轴标签）、`formula`（数学字体/公式行/`O(...)`）、`header_footer`、纯 `[N]` 作者行以及 `unreliable` 块默认 `translate=false, redact_source=false`，避免擦除图表或破坏公式。
- `caption`（`Fig.` / `Figure` / `Table` / `Algorithm`）、`title`、`heading`、`body` 与常规 `reference` 默认 `translate=true, redact_source=true`。

---

## 3. 扫描页判定与 `auto` 路由规则

1. **扫描页判定**（`pymupdf_adapter.py:_extract_page_blocks`）：
   - 当某页原生提取字符数 `< scanned_char_threshold`（默认 `20`，可在 YAML `ocr.auto.scanned_char_threshold` 配置）**且**页面存在至少 1 个图像块（`len(image_rects) > 0`）时，标记 `is_scanned = True`。
2. **`auto` 路由逻辑**（`AutoOCRProvider.parse`）：
   - **非 `.pdf` 文本输入** → 走 `stub`（`pdf_pt`, 72 DPI）。
   - **数字 PDF 页（`is_scanned=False`）** → 直接保留 `pymupdf` 矢量提取结果（`pdf_pt`, 72 DPI），保留精确字号、加粗、数学字体与链接。
   - **扫描 PDF 页（`is_scanned=True`）**：
     - `fallback_provider == "none"`：不路由外部 OCR，保留 `ocr_reliable=False` 并记录限制。
     - `fallback_provider` 为 `""` 或 `"paddle-http"` 且配置了 `paddle_http.base_url` / `PADDLE_OCR_URL`：探测 `/health`，仅将扫描页发往 `paddle-http`（`pixel`, `dpi`），与数字页按页号合并；在 `ocr.result.json` 中记录 `paddle_service`、`paddle_pages` 与 `routed_providers`。
     - `fallback_provider == "paddle-cpu"`（或未配 HTTP 且全篇均为扫描页）：回落到本地进程内 `paddle-cpu`。

---

## 4. 错误契约

| 异常类 | 基类 | 触发场景 |
|---|---|---|
| `OCRConfigError` | `OCRError, ValueError` | 未知 provider 名称、重复注册未设 `replace=True`、非法 `base_url`/`dpi`/`timeout_s`/`fallback_provider` |
| `OCRBackendUnavailableError` | `OCRError` (`RuntimeError`) | 外部 OCR HTTP 服务 `/health` 或 `/ocr` 连接失败、超时、非 2xx 响应 |
| `OCRContractError` | `OCRError, ValueError` | Provider 返回结果违反 `document.v1` 结构要求（非法 `coord_space`、反转 `bbox`、缺少 `policy.translate` 等） |

---

## 5. 接入下一个 OCR 后端（开发示例）

以下示例展示如何接入一个自定义 HTTP OCR 后端（例如 MinerU / Surya / 自研服务）：

### 步骤 1：实现 `OCRProvider` 子类

新建 `python/ocr/custom_http_adapter.py`：

```python
from __future__ import annotations

import json
import urllib.request
import pymupdf

from ocr.base import OCRBackendUnavailableError, OCRConfigError, OCRProvider
from ocr.normalize import normalize_pages


class CustomHTTPOCRProvider(OCRProvider):
    name = "custom-http"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        opts = self.config.get("options") or {}
        self.endpoint = str(opts.get("endpoint") or "").rstrip("/")
        self.dpi = float(opts.get("dpi", 150.0))
        if not self.endpoint.startswith(("http://", "https://")):
            raise OCRConfigError("custom-http requires ocr.options.endpoint (http/https URL)")

    def parse(self, source: str) -> list[dict]:
        doc = pymupdf.open(source)
        raw_pages: list[dict] = []
        try:
            for page in doc:
                pix = page.get_pixmap(dpi=int(self.dpi))
                req = urllib.request.Request(
                    f"{self.endpoint}/ocr?dpi={self.dpi:.1f}&page_index={page.number}",
                    data=pix.tobytes("png"),
                    headers={"Content-Type": "image/png"},
                    method="POST",
                )
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                try:
                    with opener.open(req, timeout=60) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                except Exception as exc:
                    raise OCRBackendUnavailableError(f"custom-http failed on page {page.number}: {exc}") from exc

                blocks = [
                    {
                        "block_id": f"p{page.number}b{i:04d}",
                        "text": item["text"],
                        "bbox": [float(v) for v in item["bbox"]],
                    }
                    for i, item in enumerate(data.get("blocks", []))
                ]
                raw_pages.append({
                    "page_index": page.number,
                    "width": float(pix.width),
                    "height": float(pix.height),
                    "is_scanned": True,
                    "ocr_reliable": True,
                    "blocks": blocks,
                })
        finally:
            doc.close()
        return normalize_pages(raw_pages, coord_space="pixel", dpi=self.dpi)
```

### 步骤 2：注册并暴露到跨语言契约

1. 在 `python/ocr/adapter.py` 中导入并注册：
   ```python
   from ocr.custom_http_adapter import CustomHTTPOCRProvider
   register_provider(CustomHTTPOCRProvider.name, CustomHTTPOCRProvider)
   ```
2. 在 `contracts/task.spec.schema.json` 的 `properties.ocr.properties.provider.enum` 中加入 `"custom-http"`（Go 配置校验与 HTTP API 会自动从该单一真源识别新后端，无需改 Go 硬编码列表）。
3. 在单 YAML 配置中直接选用：
   ```yaml
   ocr:
     provider: "custom-http"
     options:
       endpoint: "http://127.0.0.1:19000"
       dpi: 150
   ```
4. 编写回归测试：调用 `get_provider("custom-http", config=...)` 并用 `validate_document_pages(pages)` 验证输出契约。
