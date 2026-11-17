# miniretain

一个最小可跑的「Go 服务端 + Python worker」PDF 翻译服务骨架，从 `wxyhgk/retain-pdf`
抽象而来，只保留服务端核心：对外任务接入、OCR 接入边界、LLM 任务调度。
目标是把跨语言契约做到**单一真源**，并显式管理并发边界。

> 边界：验证的是服务链路与契约，不是翻译质量。OCR/LLM 真实 provider 已接入
> （Paddle CPU + OpenAI-compatible 网关），但排版重组/公式保护/段落续接未实现。

## 目录

- `cmd/server/main.go` — 服务入口，`-max-workers` 控制进程级并发。
- `internal/api/handler.go` — 6 个 HTTP 接口（对外任务接入）。
- `internal/worker/executor.go` — 用外部进程驱动 Python worker，信号量限流。
- `internal/worker/spec.go` — 从 schema 加载并生成 spec（不硬编码默认值）。
- `internal/job/job.go` — 内存任务存储（生产换 SQLite，API 不变）。
- `python/worker.py` — 单 worker 入口，按 `spec.stage` 分派 ocr / llm。
- `python/ocr/adapter.py` — `OCRProvider` 抽象；`paddle_adapter.py` 是 CPU 实现。
- `python/llm/scheduler.py` — `LLMClient` 抽象 + `Scheduler`（分批/并发/重试/保序）。
- `python/llm/openai_client.py` — OpenAI-compatible HTTP client（stdlib urllib）。
- `contracts/task.spec.schema.json` — Go/Python 共用单一真源的契约。
- `tools/smoke.sh` / `tools/smoke_real.sh` — stub / 真实 provider 端到端冒烟。

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/v1/jobs` | 建任务，立即返回 202 + id，后台执行 |
| GET | `/v1/jobs` | 列出任务 |
| GET | `/v1/jobs/{id}` | 查单个任务状态 |
| GET | `/v1/jobs/{id}/result` | 取结果（仅 succeeded） |
| POST | `/v1/jobs/{id}/retry` | 重新入队 |

建任务请求体示例：

```json
{
  "source_path": "/abs/path/to/doc.pdf",
  "target_lang": "zh",
  "ocr": {"provider": "paddle-cpu"},
  "llm": {"provider": "openai-compat", "batch_size": 40, "concurrency": 4,
          "max_retries": 1, "timeout_ms": 120000, "model": "gemini-3.6-flash"}
}
```

## 并发模型（concurrency）

并发分两层，彼此独立：

**1. Go 进程级：`-max-workers`（默认 2）**
`cmd/server/main.go:19` 的 `-max-workers` 传入 `worker.New`，在
`internal/worker/executor.go:26,36` 用一个带缓冲 channel `sem` 作信号量，
限制**全服务同时运行的 Python worker 进程数**。每个 stage 进入前获取信号量，
退出时释放（`executor.go:81-86`）。进程用 `exec.CommandContext` 启动
（`executor.go:90`），请求/上下文取消会连带杀掉子进程，避免孤儿进程。
这一层控制的是整机的进程/CPU 压力；单个 job 的 ocr→llm 仍是串行 stage。

**2. Python 请求级：`llm.concurrency`（schema 默认 2）**
`python/llm/scheduler.py:82` 用 `ThreadPoolExecutor(max_workers=concurrency)`
并发发送分批 LLM 请求，`as_completed`（`:87`）收集结果，并按
`results[start+offset] = value`（`:95-96`）**按原始下标回填**，所以完成顺序
不影响输出顺序。每批失败按指数退避重试（`:63-72`，`min(0.05*2^attempt, 0.5)`）。
`concurrency` 来自单一真源 `contracts/task.spec.schema.json:53-58`，
`python/worker.py:54` 读取，Go/Python 两侧都不硬编码兜底值。

**实测口径（诚实说明）**：真实端到端跑（`tools/smoke_real.sh:30`）里
`llm.concurrency` 被显式设为 **1**，即 LLM 请求是**串行**的——所以之前
「1412 块约 35 分钟」的数字**并未使用并发**。代码支持把 `llm.concurrency`
调到 4~8 来并行请求，也支持 `-max-workers > 1` 并行多个 job，但那次验证
两者都没打开。另外**未实现客户端限流**：提高并发会直接压到单一网关上，
生产环境应先加 limiter 再放大并发。

## 跨语言契约

`contracts/task.spec.schema.json` 是 Go 与 Python 之间唯一的字段/默认值来源。
Go 侧 `internal/worker/spec.go` 与 Python 侧 `python/contract.py` 都加载同一文件
做校验与默认值填充。新增字段只改 schema，避免 retain-pdf 那种「两边各写一份
兜底值、某天静默分叉且无报错」的问题。

## 跑法

```sh
go test ./...
sh tools/smoke.sh          # stub provider，无需密钥
# 真实 provider：先建隧道 ssh -N -L 18083:127.0.0.1:8083 ccds@100.99.0.3
OPENAI_API_KEY_FILE=... OPENAI_BASE_URL=http://127.0.0.1:18083/v1 \
  MINIRETAIN_PYTHON=... sh tools/smoke_real.sh <pdf>
```

密钥只从环境变量/文件读取，不写入源码或产物。

## 已知边界

- 任务存储在内存，进程重启即丢失；生产换 SQLite。
- OCR/LLM 为逐块平铺翻译，无版面重组/公式保护/段落续接。
- 无限流、无持久化 checkpoint、无原子 rename、无权限体系。
