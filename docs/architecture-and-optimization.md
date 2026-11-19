# Miniretain Pipeline Architecture, Concurrency Scheduling, Translation Quality & Typst Font Sizing

## 1. End-to-End Pipeline Flow, Data Contracts & Failure Semantics

Miniretain executes each job (`POST /v1/jobs`) as isolated Python worker stages orchestrated by the Go `Executor` (`internal/worker/executor.go`). Go and Python share a single schema (`contracts/task.spec.schema.json`) and exchange files only under `<data_dir>/<job_id>/`.

```
Source PDF
  │
  ▼ [Stage 1: OCR] python/worker.py:run_ocr → python/ocr/adapter.py + normalize.py
  │  Input : source PDF + ocr.spec.json
  │  Output: document.v1.json (pages, blocks, bbox, line_bboxes, layout_role, policy, reading_order)
  │  Failure: invalid/unreachable provider or schema mismatch fails stage immediately;
  │           unreliable scanned pages set ocr_reliable=false & policy.translate=false (source preserved).
  │  Measured (arXiv 2611.20893, 7p digital via auto→pymupdf): ~1.1–1.5s
  │
  ▼ [Stage 2: LLM] python/worker.py:run_llm → formula_protect.py → llm/{context,planner,limiter,retry,scheduler,openai_client}.py
  │  Input : document.v1.json + llm.spec.json (+ optional MINIRETAIN_LLM_GLOBAL_CONCURRENCY / SLOT_DIR)
  │  Step 2a (Filter & Protect):
  │    - Blocks with policy.translate=false (figure_label, numeric_axis_tick, formula, header_footer,
  │      unreliable_ocr, reference_authors) are preserved verbatim (redact_source=false), keeping
  │      log-scale exponent ticks (e.g., 10^3, 10^-1) and numeric axis ticks pixel-untouched.
  │    - Translatable blocks pass through protect_formulas(): display/inline math, LaTeX commands,
  │      O(...) complexity, URLs, DOIs, and reference [N]/authors become ⟪F0⟫... placeholders.
  │  Step 2b (Document Context):
  │    - build_document_context() extracts paper title + recurring acronyms + canonical term rules
  │      in <2ms (zero extra LLM calls) and attaches it as a cross-batch terminology anchor.
  │  Step 2c (Batch Planning — CharBudgetLPTPlanner):
  │    - Packs consecutive reading-order blocks under (<= batch_size, <= max_chars) and orders
  │      Longest-Processing-Time (LPT) first to launch heavy batches immediately.
  │  Step 2d (Bounded Execution, Streaming Retry & Backfill — Scheduler + Limiter + RetryPolicy):
  │    - ConcurrencyLimiter / FileSlotConcurrencyLimiter enforces both per-job concurrency and
  │      optional cross-process global gateway cap (runtime.max_llm_concurrency).
  │    - RetryPolicy classifies 429 (honors Retry-After + shared sibling cooldown + equal jitter),
  │      5xx/timeout (jittered backoff), and 4xx fatal (no wasted retries).
  │    - Missing/empty items are immediately re-enqueued into the active worker pool as soon as
  │      their batch completes (no full-wave barrier wait), and backfilled by orig_idx & block_id.
  │  Output: translation.payload.json, translation-manifest.json, llm.result.json
  │  Failure: if allow_partial=false (default) and any translatable item is failed/missing/empty,
  │           run_llm raises RuntimeError and aborts before render.
  │  Measured (arXiv 2611.20893, 112 translatable / 72 preserved, gemini-3.6-flash): ~15–35s (80–90% of wall time)
  │
  ▼ [Stage 3: Render] python/worker.py:run_render → python/render/typst_renderer.py + side_by_side.py
     Input : source PDF + document.v1.json + translation.payload.json + render.spec.json
     Step 3a (Line-Aware Start & Two-Tier Typst Measure):
       - Multi-line blocks start at 0.98x source font_size_pt; single-line boxes cap at
         min(0.92x hint, 0.85x rect.height) to account for CJK em-box height.
       - Candidate sizes descend in 0.25pt steps above 7.0pt and 0.5pt steps down to min_size (5.0pt).
       - Pass 1 probes 2 sizes per block; Pass 2 probes remaining sizes only for non-fitting blocks.
     Step 3b (Compiled Overlay Verification & Strict Mode):
       - Compiles overlay.typ → overlay.pdf BEFORE redacting source PDF.
       - Inspects rendered glyph spans for horizontal/vertical overflow, page bounds, and neighbor
         overlap; retries overflowing blocks up to 6 steps down to min_size.
       - In strict=true mode, any unplaced or failed block aborts with explicit block_ids/reasons.
     Step 3c (Redact, Composite & Side-by-Side):
       - Redacts ONLY verified placed blocks (fill=None on digital pages so vector figures stay intact),
         composites overlay.pdf, restores clickable URI links, and builds Side-by-Side PDF (2x width).
     Output: *_CN.pdf (612x792 pt), *_Side-by-Side.pdf (1224x792 pt), render.result.json (incl. font_size_stats)
     Measured (arXiv 2611.20893, 7p, Typst 0.15.1): ~2.2–2.7s
```

## 2. Module Boundaries & Interface Contracts

To support future model providers, custom schedulers, and render policies without breaking `LLMClient`, single-YAML config, or `POST /v1/jobs`:

| Module | Contract | Responsibility |
|---|---|---|
| `python/llm/planner.py` | `BatchPlanner.plan(items) -> list[PlannedBatch]` | Character-budget grouping + LPT ordering; preserves `orig_idx` for backfill. |
| `python/llm/limiter.py` | `ConcurrencyLimiter.acquire()` + `snapshot()` | Per-process semaphore (`ConcurrencyLimiter`) and cross-process POSIX `flock` slot pool (`FileSlotConcurrencyLimiter`, kernel-auto-released on crash). |
| `python/llm/retry.py` | `RetryPolicy.decide(exc, attempt) -> RetryDecision` | Error taxonomy (`rate_limit`, `transient_http`, `timeout`, `parse_error`, `fatal`), `Retry-After` parsing, equal jitter, and shared 429 cooldown. |
| `python/llm/context.py` | `build_document_context(items) -> str` | Zero-RTT document title, recurring acronym, and terminology consistency anchor. |
| `python/llm/scheduler.py` | `Scheduler.run(items, target_lang, model)` | Orchestrates planner + limiter + retry + streaming item retry + deterministic `block_id` backfill + observable metrics (`requests`, `peak_inflight`, `rate_limit_429`, `timeouts`, `limiter_wait_s`). |
| `internal/config/config.go` & `internal/worker/executor.go` | `runtime.max_llm_concurrency` (optional YAML int) | Propagates `MINIRETAIN_LLM_GLOBAL_CONCURRENCY` & `MINIRETAIN_LLM_SLOT_DIR` to `llm` stage workers; keeps legacy CLI and existing YAML 100% compatible. |

## 3. Bottleneck Analysis & Experimental Results (LAB-2117)

### 3.1 Multi-Job Gateway Amplification (Go Server + Local Fake Gateway, Cap = 8)

Two concurrent `POST /v1/jobs` submissions of arXiv `2611.20893` (112 translatable blocks each) against the same Go server (`runtime.max_workers=2`, `llm.concurrency=8`, `llm.batch_size=18`, gateway rejects >8 in-flight with HTTP 429 + `Retry-After: 0.15`):

| Configuration | Gateway Peak In-Flight ↓ | Total Requests ↓ | HTTP 429 ↓ | Total Retries ↓ | Translated (Job 1 / Job 2) |
|---|---:|---:|---:|---:|---|
| Baseline (`max_llm_concurrency` unset) | 13–16 | 24 | 8 | 8 | 112/112, 112/112 |
| **Optimized (`max_llm_concurrency: 8`)** | **8** (-38% to -50%) | **16** (-33.3%) | **0** (-100%) | **0** (-100%) | 112/112, 112/112 |

### 3.2 Live Full-Paper Benchmark on arXiv `2611.20893` (7 Pages, Strict Mode)

Same input (`2611.20893.pdf`), same model (`gemini-3.6-flash`), same `geminiweb2api` Go gateway (`per_ip_concurrent=32`, `per_ip_rpm=300`), same `batch_size=18`, `concurrency=8`, `max_workers=2`, `strict=true`, Typst `0.15.1 (9dfd3a08)`:

| Version & Run | Cold/Warm | Total Wall (s) ↓ | OCR (s) | LLM (s) ↓ | Max Batch (s) | Retry Wait (s) ↓ | Requests / Peak / 429 / Retries | Render (s) | Placed / Preserved / Failed |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| `536ecd3` Run 1 | cold | 38.866 | 1.524 | 34.254 | 23.761 | 10.478 | 11* / 8* / 0 / 3 | 2.153 | 112 / 72 / 0 |
| `536ecd3` Run 2 | warm | 53.034 | 1.126 | 48.496 | 40.102 | 8.390 | 11* / 8* / 0 / 3 | 2.553 | 112 / 72 / 0 |
| `536ecd3` Run 3 | warm | 21.105 | 1.121 | 16.560 | 16.559 | 0.000 | 8* / 8* / 0 / 0 | 2.510 | 112 / 72 / 0 |
| **`536ecd3` Median** | — | **38.866** | 1.126 | **34.254** | 23.761 | 8.390 | — | 2.510 | 112 / 72 / 0 |
| `LAB-2117` Run 1 (`afcefde188018d85`) | cold | 21.373 | 1.580 | 16.498 | 16.492 | 0.000 | 8 / 8 / 0 / 0 | 2.197 | 112 / 72 / 0 |
| `LAB-2117` Run 2 (`cc947e23e582caa6`) | warm | 20.614 | 1.122 | 16.130 | 16.123 | 0.000 | 8 / 8 / 0 / 0 | 2.341 | 112 / 72 / 0 |
| `LAB-2117` Run 3 (`d9d938c0be3c6184`) | warm | 22.373 | 1.137 | 17.682 | 17.677 | 0.000 | 8 / 8 / 0 / 0 | 2.610 | 112 / 72 / 0 |
| **`LAB-2117` Median** | — | **21.373** (**-45.0%**) | 1.137 | **16.498** (**-51.8%**) | **16.492** | **0.000** | **8 / 8 / 0 / 0** | **2.341** | **112 / 72 / 0** |

*\*`536ecd3` did not record `requests` or `peak_inflight` directly; inferred as `batches + retries` with `concurrency=8`.*

### 3.3 Typst Font-Size Distribution Comparison (`2611.20893`, Strict Mode, 112 Placed Blocks)

| Strategy | Min Floor (pt) | Actual Min (pt) ↑ | Median (pt) ↑ | Mean (pt) ↑ | Max (pt) | Shrunk Blocks | Unplaced / Failed | Render Time (s) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `536ecd3` (`0.92×`, `0.5pt` step, cap 14pt) | 5.0 | 5.0 | 9.0 | 8.31 | 14.0 | 9 steps | 0 / 0 | 2.51–2.63 |
| **`LAB-2117` (line-aware `0.98×`, `0.25pt` step, title cap 16pt)** | 5.0 | **5.5** (+0.5) | **9.5** (+0.5) | **8.67** (+0.36) | **16.0** | 6 blocks (15 steps) | **0 / 0** | 2.20–2.61 |

### 3.4 Honest Limitations & Items Below Ideal Target

- **Minor acronym casing drift**: Document context eliminated untranslated `Fig. N` (`0` remaining vs `Fig. 4` in baseline) and fixed `attention`/`expert` translation, but `KV Cache` vs `KV cache` still showed occasional casing variation (`12` vs `9` in `afcefde188018d85`). Deterministic post-pass normalization would be needed if 100% strict casing is required.
- **Single-job gateway headroom**: Because the live `geminiweb2api` Go instance is configured with `per_ip_concurrent=32`, single-job runs at `concurrency=8` do not trigger 429s; the cross-process `max_llm_concurrency` limiter's benefit manifests under multi-job load or tighter upstream quotas.
- **Pure-numeric & log-scale axis ticks (fixed in Round 2)**: `normalize.py` now classifies pure numeric and power-of-10 exponent axis ticks (`5..35`, `10³..10⁻¹`, `10..1000`) as `figure_label` (`preserve_reason="numeric_axis_tick"`, `translate=false`, `redact_source=false`), excluding them from LLM translation and Typst redraw so superscript elevation (~3 pt) stays pixel-identical to the source PDF.
- **Retry accounting invariant (fixed in Round 2)**: `scheduler.py` increments `retries` at dispatch time for every additional call (both batch retries and targeted single-item retries), guaranteeing `requests == batches + retries` across success, exhausted retries, and fatal-4xx fast-fail.

