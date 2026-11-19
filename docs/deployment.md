# Miniretain 部署与运行指南

本指南提供从干净代码检出到单 YAML 启动、健康检查、提交 PDF 翻译任务并获取中文版与左右双语版 PDF 产物的完整可复制命令。

---

## 1. 环境依赖安装

### 1.1 Go 与 Python 运行环境

- **Go**: `1.23+`（用于编译服务端二进制）
- **Python**: `3.10+`（worker 进程，需要 `pymupdf`；跑回归需要 `pytest`）

```bash
# 创建独立 Python 虚拟环境并安装核心依赖
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install pymupdf pytest
.venv/bin/python -c "import pymupdf; print('PyMuPDF:', pymupdf.__version__)"
```

### 1.2 安装 Typst `0.15.1` 与 CJK 字体

Miniretain 保版式渲染针对官方稳定版 **Typst `0.15.1`** 验证（已修复 CJK 光学左悬挂与跨栏归属问题）。

```bash
# Linux x86_64 安装 Typst 0.15.1
curl -fL -o /tmp/typst.tar.xz \
  https://github.com/typst/typst/releases/download/v0.15.1/typst-x86_64-unknown-linux-musl.tar.xz
echo "29273eaa04f6d00edd0c2bec578f565fc9c65be856bfbffc894567c68ed0b237  /tmp/typst.tar.xz" | sha256sum -c -
tar -xJf /tmp/typst.tar.xz -C /tmp
sudo install -m 0755 /tmp/typst-x86_64-unknown-linux-musl/typst /usr/local/bin/typst
typst --version
# 预期输出：typst 0.15.1 (9dfd3a08)
```

安装开源中文字体（Ubuntu / Debian 示例）：

```bash
sudo apt-get update && sudo apt-get install -y fonts-noto-cjk fontconfig
fc-cache -fv
fc-list : family | grep -i "Noto Serif CJK\|Noto Sans CJK" | sort -u
```

---

## 2. 启动下游依赖服务

### 2.1 本地 PaddleOCR HTTP 服务（用于扫描页或 `ocr.provider: paddle-http`）

将 PaddleOCR 模型常驻在独立 HTTP 进程中，避免每个任务重复冷加载模型。可在当前节点或内网/Tailscale 节点运行：

```bash
# 在已安装 paddlepaddle + paddleocr 的虚拟环境中启动
python3 python/ocr/paddle_server.py \
  --host 0.0.0.0 \
  --port 18085 \
  --lang en
```

验证服务健康：

```bash
curl --noproxy '*' -sS http://<paddle-host>:18085/health | jq .
# 预期返回包含 "status": "ok", "coord_space": "pixel", "bbox_format": "xyxy"
```

> 若仅处理数字 PDF，可将 YAML 中 `ocr.provider` 设为 `pymupdf`（或 `auto` 且 `auto.fallback_provider: none`），无需启动 PaddleOCR 服务。

### 2.2 OpenAI-Compatible 翻译网关（如 `geminiweb2api` Go）

准备兼容 `/v1/chat/completions` 的网关（例如 `geminiweb2api` Go，默认监听 `:8083`）并将密钥写入环境变量或只读密钥文件：

```bash
# 方式 A：写入密钥文件（推荐，配合 YAML llm.api_key_file）
mkdir -p ~/.config/miniretain
printf '%s\n' '<YOUR_GATEWAY_KEY>' > ~/.config/miniretain/gw2a.key
chmod 600 ~/.config/miniretain/gw2a.key

# 方式 B：通过环境变量传递（配合 YAML llm.api_key_env）
export OPENAI_API_KEY='<YOUR_GATEWAY_KEY>'

# 连通性验证
curl --noproxy '*' -sS http://<gw2a-host>:8083/v1/models \
  -H "Authorization: Bearer $(cat ~/.config/miniretain/gw2a.key)" | jq .
```

---

## 3. 复制单 YAML 配置并构建启动

### 3.1 复制并编辑配置

```bash
cp configs/miniretain.example.yaml ./miniretain.yaml
```

按实际环境修改 `./miniretain.yaml` 中关键字段（例如 `runtime.python` 指向 `.venv/bin/python`，`ocr.paddle_http.base_url` 与 `llm.base_url` 指向实际地址，凭据使用 `llm.api_key_env` 或 `llm.api_key_file`，**不要在 YAML 中写明文密钥**）。

### 3.2 构建与启动

```bash
go build -o ./bin/miniretain ./cmd/server
./bin/miniretain --config ./miniretain.yaml
```

也支持通过环境变量 `MINIRETAIN_CONFIG=./miniretain.yaml ./bin/miniretain` 指定配置文件。

---

## 4. API 验证与获取产物

### 4.1 健康检查

```bash
curl --noproxy '*' -sS http://127.0.0.1:18080/health | jq .
```

正常返回示例（仅暴露脱敏摘要，不包含任何密钥）：

```json
{
  "ok": true,
  "time": "2026-11-19T12:00:00Z",
  "config_mode": "yaml",
  "max_workers": 2,
  "ocr_provider": "auto",
  "llm_provider": "openai-compat",
  "llm_model": "gemini-3.6-flash",
  "render_enabled": true,
  "render_engine": "typst"
}
```

### 4.2 提交 PDF 翻译任务

当 YAML 已配置 `ocr`、`llm` 和 `render.enabled: true` 时，最简请求只需提供 `source_path`：

```bash
JOB_ID=$(curl --noproxy '*' -sS -X POST http://127.0.0.1:18080/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"source_path": "/abs/path/to/paper.pdf"}' | jq -r '.id')
echo "JOB_ID=$JOB_ID"
```

如需在单次请求中覆盖部分任务参数（例如临时切换 OCR 后端或关闭严格模式）：

```bash
curl --noproxy '*' -sS -X POST http://127.0.0.1:18080/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "/abs/path/to/paper.pdf",
    "target_lang": "zh",
    "ocr": {"provider": "pymupdf"},
    "llm": {"model": "gemini-3.6-flash", "concurrency": 4},
    "render": {"strict": true, "bilingual_mode": "side_by_side"}
  }' | jq .
```

> 若 YAML 开启了 `render.enabled: true`，但某次请求只想跑 OCR+LLM、跳过渲染，可在请求体显式传 `"render": null`。

### 4.3 查询状态、结果与定位产物

```bash
# 轮询任务状态（queued -> running -> succeeded | failed）
curl --noproxy '*' -sS "http://127.0.0.1:18080/v1/jobs/${JOB_ID}" | jq '{id, status, stage, error}'

# 任务 succeeded 后获取完整阶段统计与产物路径
curl --noproxy '*' -sS "http://127.0.0.1:18080/v1/jobs/${JOB_ID}/result" | jq .
```

任务目录位于 `<paths.data_dir>/<job_id>/`（默认 `./data/jobs/<job_id>/`），关键产物：

| 文件 | 说明 |
|---|---|
| `translated_cn.pdf` | 纯中文保版式 PDF（路径与 `result.render.output_cn` 一致） |
| `translated_merged.pdf` | 左原右译双语对照 PDF（路径与 `result.render.output_merged` 一致） |
| `document.v1.json` | 规范化后的 OCR 文档（含页尺寸、坐标空间、角色与策略） |
| `translation-manifest.json` | 翻译对账清单（逐页统计、覆盖率、公式保护告警） |
| `ocr.result.json` / `llm.result.json` / `render.result.json` | 各阶段执行结果与耗时 |

---

## 5. YAML 配置字段表

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `server.addr` | string | `:18080` | HTTP 监听地址 |
| `server.read_header_timeout_ms` | int | `5000` | HTTP 读取请求头超时（ms，须 `>= 1`） |
| `paths.root` | string | YAML 所在目录 | 项目根目录（须包含 `contracts/` 与 `python/worker.py`） |
| `paths.data_dir` | string | `<paths.root>/data/jobs` | 任务工作目录与产物落盘根目录 |
| `paths.schema` | string | `<paths.root>/contracts/task.spec.schema.json` | 跨语言契约 schema 路径（也读 `MINIRETAIN_SCHEMA`） |
| `runtime.python` | string | `python3` | Python 可执行文件名或路径 |
| `runtime.typst` | string | `""`（自动查找） | Typst 可执行文件名或路径（通过 `TYPST_BIN` 传给 worker） |
| `runtime.max_workers` | int | `2` | Go 侧全局并发 Python worker 进程上限（须 `>= 1`） |
| `target_lang` | string | `zh` | 默认目标语言 |
| `ocr.provider` | string | schema 默认 `stub` | OCR 后端：`auto` \| `pymupdf` \| `paddle-http` \| `paddle-cpu` \| `stub` |
| `ocr.paddle_http.base_url` | string | `""`（回落 `PADDLE_OCR_URL`） | PaddleOCR HTTP 地址（`http://` 或 `https://`） |
| `ocr.paddle_http.dpi` | float | `150` | 扫描页光栅化 DPI（须 `> 0`） |
| `ocr.paddle_http.timeout_s` | float | `180` | 单页 `/ocr` 请求超时秒数（须 `> 0`） |
| `ocr.paddle_http.health_timeout_s` | float | `5` | `/health` 探测超时秒数（须 `> 0`） |
| `ocr.paddle_cpu.dpi` | float | `150` | 本地进程内 PaddleOCR DPI（须 `> 0`） |
| `ocr.paddle_cpu.lang` | string | `en` | 本地 PaddleOCR 语言 |
| `ocr.auto.scanned_char_threshold` | int | `20` | 扫描页判定字符数阈值（须 `>= 0`） |
| `ocr.auto.fallback_provider` | string | `paddle-http` | `auto` 遇到扫描页时的路由目标：`paddle-http` \| `paddle-cpu` \| `none` |
| `ocr.options` | map | `{}` | 透传给自定义/未来 OCR 后端的扩展参数（位于 `spec.ocr.config.options`） |
| `llm.provider` | string | schema 默认 `stub` | LLM 后端：`openai-compat` \| `stub` |
| `llm.base_url` | string | `""`（回落 `OPENAI_BASE_URL`） | OpenAI-compatible 网关 URL（含 `/v1`） |
| `llm.api_key_env` | string | `""` | 存放 API Key 的环境变量名（如 `OPENAI_API_KEY`） |
| `llm.api_key_file` | string | `""` | 存放 API Key 的密钥文件路径（启动时检查可读性） |
| `llm.model` | string | schema 默认 `stub-v1` | 默认模型名（如 `gemini-3.6-flash`） |
| `llm.batch_size` | int | schema 默认 `8` | 每批翻译块数（须 `>= 1`） |
| `llm.concurrency` | int | schema 默认 `2` | LLM 调度器并发请求数（须 `>= 1`） |
| `llm.max_retries` | int | schema 默认 `2` | 单批失败重试次数（须 `>= 0`） |
| `llm.timeout_ms` | int | schema 默认 `30000` | 单次 LLM 请求超时（ms，须 `>= 1`） |
| `llm.allow_partial` | bool | `false` | 是否允许部分块翻译失败仍继续 |
| `render.enabled` | bool | `false` | 为 `true` 时，请求省略 `render` 也默认执行渲染阶段 |
| `render.engine` | string | schema 默认 `typst` | 渲染引擎：`typst` \| `pymupdf` |
| `render.font` | string | `""`（自动发现） | 字体族名（Typst）或字体文件路径（PyMuPDF） |
| `render.layout_mode` | string | schema 默认 `auto` | `auto` \| `preserve` \| `flow` |
| `render.bilingual_mode` | string | schema 默认 `auto` | `auto` \| `side_by_side` \| `interleaved` \| `table` |
| `render.output` | string | `translated.pdf` | 兼容旧版输出文件名 |
| `render.output_cn` | string | `translated_cn.pdf` | 纯中文输出文件名 |
| `render.output_merged` | string | `translated_merged.pdf` | 双语合并输出文件名 |
| `render.strict` | bool | `false` | 严格模式：存在任意未落版/溢出块即令渲染阶段失败 |

---

## 6. 路径解析、优先级与旧方式兼容

### 6.1 路径解析规则

- YAML 中的相对路径（`paths.root`、`paths.data_dir`、`paths.schema`、`llm.api_key_file`，以及包含 `/` 或 `.` 前缀的 `runtime.python`、`runtime.typst`、`render.font`）一律**相对于该 YAML 文件所在目录**解析，与启动时当前工作目录无关。
- 不含路径分隔符的可执行文件名（如 `python3`、`typst`）与字体族名（如 `Noto Serif CJK SC`）保持原样，按 `PATH` 或系统字体库查找。
- CLI 显式传入的相对路径（如 `-root ./miniretain`）相对于当前工作目录（`cwd`）解析。

### 6.2 配置优先级（高 → 低）

1. **单次 HTTP 请求体（`POST /v1/jobs`）**：允许覆盖任务级字段 `target_lang`、`ocr.*`、`llm.*`（不含凭据和网关地址）、`render.*`（含 `"render": null` 显式跳过渲染）。
2. **显式 CLI 参数**：仅当命令行实际传入 `-addr`、`-root`、`-python`、`-max-workers` 时才覆盖 YAML 对应项。
3. **YAML 配置文件**：`--config <file.yaml>` 或环境变量 `MINIRETAIN_CONFIG`。
4. **进程环境变量**：`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_API_KEY_FILE`、`PADDLE_OCR_URL`、`PADDLE_OCR_DPI`、`PADDLE_OCR_TIMEOUT_S`、`TYPST_BIN`、`MINIRETAIN_SCHEMA`。
5. **契约与内置默认值**：`contracts/task.spec.schema.json` 与服务端内置默认值。

### 6.3 兼容已有 CLI / 环境变量启动方式

不传 `--config` 时，旧命令完全保持原行为（`render.enabled` 默认 `false`，任务默认值由 `contracts/task.spec.schema.json` 提供）：

```bash
OPENAI_BASE_URL=http://127.0.0.1:8083/v1 \
OPENAI_API_KEY_FILE=~/.config/miniretain/gw2a.key \
PADDLE_OCR_URL=http://127.0.0.1:18085 \
  ./bin/miniretain -addr :18080 -root . -python .venv/bin/python -max-workers 2
```

---

## 7. 常见故障定位

| 现象 / 报错 | 原因 | 排查与处理 |
|---|---|---|
| 启动报 `field xxx not found in type config.XXX` | YAML 中存在拼写错误、未知字段，或误写了明文 `api_key` | 对照第 5 节字段表修正；密钥请改用 `llm.api_key_env` 或 `llm.api_key_file` |
| 启动报 `cannot unmarshal !!str into int` | 字段类型错误（如 `max_workers: "2"`） | 去掉数字/布尔值两侧引号 |
| 启动报 `ocr.provider "xxx" is invalid` | 指定了未在 schema 注册的后端名 | 使用 `auto` / `pymupdf` / `paddle-http` / `paddle-cpu` / `stub`，或按 `docs/ocr-backends.md` 同步注册 |
| 启动报 `llm.api_key_file ... not accessible` | 密钥文件路径不存在或相对路径基准写错 | 注意相对路径以 YAML 文件所在目录为基准，或改用绝对路径 |
| `curl` 访问 Tailscale `100.x.x.x` 返回 `SOCKS5` 错误 | 系统代理拦截了内网/Tailscale IP（`curl` 不支持 `NO_PROXY=100.*` 通配） | `curl` 加 `--noproxy '*'`；Python 端 `paddle_http_adapter.py` 与 `openai_client.py` 已对 `127.0.0.1`/私有网段/`100.64.0.0/10` 自动绕过代理 |
| `paddle-http health failed` | 本地 Paddle 服务未启动、端口不通或防火墙阻断 | 在服务端运行 `curl --noproxy '*' http://<host>:18085/health`，核对 `ocr.paddle_http.base_url` |
| `TYPST_BIN=... not found` 或字体渲染异常 | `runtime.typst` 指向无效路径，或未安装 Noto CJK 字体 | 运行 `typst --version` 确认 `0.15.1`，运行 `fc-list : family \| grep Noto` 确认中文字体 |
| 严格模式渲染失败 `unplaced=[...]` | 译文过长超出原块框且无法进一步缩字号 | 查看 `render.result.json` 的 `unplaced_reasons`；调试时可临时将请求体 `render.strict` 设为 `false` |
