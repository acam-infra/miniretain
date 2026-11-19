# miniretain

一个面向论文与技术文档的保版式 PDF 翻译服务（「Go 服务端 + Python worker」架构），沿 `wxyhgk/retain-pdf` 抽象并强化了**跨语言单一真源契约**、**单 YAML 声明式启动**、**可扩展 OCR 后端接口**以及基于 **Typst `0.15.1`** 的原页 overlay 渲染与左右双语对照输出。

- 详细安装、部署、配置表与排障见：[docs/deployment.md](docs/deployment.md)
- OCR 后端注册、输出契约与扩展开发手册见：[docs/ocr-backends.md](docs/ocr-backends.md)
- 流水线架构、并发调度、翻译质量与 Typst 字号优化实测见：[docs/architecture-and-optimization.md](docs/architecture-and-optimization.md)
- 单 YAML 启动示例配置见：[configs/miniretain.example.yaml](configs/miniretain.example.yaml)

---

## 核心特性

1. **单 YAML 声明式启动（`--config <file.yaml>`）**
   - 一份 YAML 集中声明监听地址、项目/任务目录、Python/Typst 可执行文件、worker 并发数、OCR 后端与连接参数、OpenAI-compatible 翻译网关与模型默认值、渲染默认值。
   - 凭据仅支持环境变量名（`llm.api_key_env`）或密钥文件（`llm.api_key_file`），禁止明文落盘；未知字段、类型错误和无效后端在启动阶段立即报错退出。
   - 完整保留旧版 CLI 参数与环境变量启动方式；优先级为：`HTTP 请求体 > 显式 CLI 参数 > YAML 配置 > 环境变量 > schema 默认值`。
2. **可扩展 OCR 后端与 `document.v1` 契约**
   - 内置 `auto`、`pymupdf`、`paddle-http`、`paddle-cpu`、`stub`，可由 YAML 或单次请求按需选用。
   - `auto` 对数字 PDF 页走 PyMuPDF（`pdf_pt` @ 72 DPI，保留矢量图、字号、数学字体与超链接），对扫描页自动路由至 `paddle-http`（`pixel` @ 150 DPI）。
   - 提供 `register_provider()` / `get_provider(name, config)` / `validate_document_pages()` 与结构化错误类型，便于接入新 OCR 后端。
3. **保版式 Typst `0.15.1` 渲染与双产物输出**
   - 支持原页坐标 overlay（`layout_mode: preserve`）与流式排版（`flow`），单次任务同时输出纯中文版（`translated_cn.pdf`）与左原右译双语对照版（`translated_merged.pdf`）。
   - 支持图内标签与公式保护、双栏阅读顺序与跨页续接、严格落版校验（`render.strict: true`）以及字符均衡 LPT 并发翻译调度。

---

## 快速启动（单 YAML）

```bash
# 1. 复制配置模板并按需修改 python/typst/paddle/llm 地址与密钥引用
cp configs/miniretain.example.yaml ./miniretain.yaml

# 2. 编译并启动服务
go build -o ./bin/miniretain ./cmd/server
./bin/miniretain --config ./miniretain.yaml

# 3. 健康检查
curl --noproxy '*' -sS http://127.0.0.1:18080/health | jq .

# 4. 提交 PDF 任务并轮询结果
JOB_ID=$(curl --noproxy '*' -sS -X POST http://127.0.0.1:18080/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"source_path": "/abs/path/to/paper.pdf"}' | jq -r '.id')
curl --noproxy '*' -sS "http://127.0.0.1:18080/v1/jobs/${JOB_ID}" | jq .
curl --noproxy '*' -sS "http://127.0.0.1:18080/v1/jobs/${JOB_ID}/result" | jq .
```

产物默认落在 `./data/jobs/<job_id>/` 下的 `translated_cn.pdf` 与 `translated_merged.pdf`。完整步骤（含 Typst `0.15.1`、字体、Paddle HTTP、geminiweb2api Go）见 [docs/deployment.md](docs/deployment.md)。

---

## 目录结构

- `cmd/server/main.go` — 服务入口，支持 `--config <file.yaml>` 及兼容旧 CLI 参数。
- `internal/config/` — 单 YAML 严格解析、路径解析、启动校验与 worker 环境装配。
- `internal/api/handler.go` — 6 个 HTTP 接口与请求级参数合并。
- `internal/worker/` — 跨语言契约校验（`spec.go`）与子进程信号量执行器（`executor.go`）。
- `internal/job/job.go` — 内存任务状态存储。
- `contracts/task.spec.schema.json` — Go / Python 共享的唯一契约真源。
- `configs/miniretain.example.yaml` — 可直接复制的单 YAML 配置模板。
- `python/worker.py` — 单 worker 入口，按 `spec.stage` 分派 `ocr` / `llm` / `render`。
- `python/ocr/` — `OCRProvider` 基类与契约校验（`base.py`）、注册表（`adapter.py`）、规范化器（`normalize.py`）、PyMuPDF/Auto（`pymupdf_adapter.py`）、Paddle HTTP 客户端/服务端（`paddle_http_adapter.py` / `paddle_server.py`）与本地 Paddle CPU（`paddle_adapter.py`）。
- `python/llm/` — 公式占位保护（`formula_protect.py`）、文档术语上下文锚点（`context.py`）、字符预算与 LPT 批次规划（`planner.py`）、进程内/跨进程全局并发预算（`limiter.py`）、错误分类与抖动重试（`retry.py`）、流式并发调度器（`scheduler.py`）与 OpenAI-compatible 客户端（`openai_client.py`）。
- `python/render/` — Typst `0.15.1` 原页 overlay 渲染器（`typst_renderer.py`）、左右双语合成（`side_by_side.py`）与 PyMuPDF 渲染器（`layout.py`）。
- `docs/deployment.md` / `docs/ocr-backends.md` — 部署运维指南与 OCR 后端接入文档。

---

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查（返回运行状态及脱敏配置摘要） |
| `POST` | `/v1/jobs` | 创建任务，立即返回 `202` 与 `id`，后台异步执行 |
| `GET` | `/v1/jobs` | 列出所有任务 |
| `GET` | `/v1/jobs/{id}` | 查询单个任务状态、当前阶段与归一化参数 |
| `GET` | `/v1/jobs/{id}/result` | 获取已完成任务的各阶段统计与产物路径（未完成返回 `409`） |
| `POST` | `/v1/jobs/{id}/retry` | 重新入队执行 |

---

## 测试与回归

```bash
# Go 单元与配置校验回归
go test -count=1 ./...

# Python OCR / LLM / Typst 0.15.1 回归
pytest -q python/tests/test_miniretain_regression.py

# 离线冒烟脚本（无需外部密钥）
sh tools/smoke.sh
sh tools/smoke_typst.sh
```

---

## 已知边界

- 任务元数据存储在内存中，进程重启后内存索引清空（磁盘产物保留在 `paths.data_dir/<job_id>/`）；生产环境可替换 `internal/job` 为 SQLite。
- 未内置客户端侧全局限流与鉴权体系，面向公网暴露前应加反向代理与限流策略。
