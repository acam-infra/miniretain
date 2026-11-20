// Package api is the external task intake surface.
package api

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"miniretain/internal/job"
	"miniretain/internal/worker"
)

// JobDefaults holds server-level task defaults (typically loaded from --config YAML).
// Per-request JSON fields in POST /v1/jobs override these defaults.
type JobDefaults struct {
	TargetLang    string
	OCR           map[string]any
	LLM           map[string]any
	Render        map[string]any
	RenderEnabled bool
}

// Options configures optional server-level defaults and redacted health metadata.
type Options struct {
	Defaults   JobDefaults
	HealthInfo map[string]any
}

type Handler struct {
	Store      *job.Store
	Exec       *worker.Executor
	Schema     *worker.Schema
	Defaults   JobDefaults
	HealthInfo map[string]any
}

func New(store *job.Store, exec *worker.Executor, schema *worker.Schema) http.Handler {
	return NewWithOptions(store, exec, schema, Options{})
}

func NewWithOptions(store *job.Store, exec *worker.Executor, schema *worker.Schema, opts Options) http.Handler {
	h := &Handler{
		Store:      store,
		Exec:       exec,
		Schema:     schema,
		Defaults:   opts.Defaults,
		HealthInfo: opts.HealthInfo,
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", h.health)
	mux.HandleFunc("POST /v1/jobs", h.create)
	mux.HandleFunc("GET /v1/jobs", h.list)
	mux.HandleFunc("GET /v1/jobs/{id}", h.get)
	mux.HandleFunc("GET /v1/jobs/{id}/result", h.result)
	mux.HandleFunc("POST /v1/jobs/{id}/retry", h.retry)
	return mux
}

func (h *Handler) health(w http.ResponseWriter, _ *http.Request) {
	resp := map[string]any{"ok": true, "time": time.Now().UTC()}
	for k, v := range h.HealthInfo {
		if k == "ok" || k == "time" {
			continue
		}
		resp[k] = v
	}
	writeJSON(w, 200, resp)
}

type createReq struct {
	Source     string          `json:"source"`
	SourcePath string          `json:"source_path"`
	TargetLang string          `json:"target_lang"`
	OCR        map[string]any  `json:"ocr"`
	LLM        map[string]any  `json:"llm"`
	Render     json.RawMessage `json:"render"`
}

func (h *Handler) create(w http.ResponseWriter, r *http.Request) {
	var req createReq
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	if err := dec.Decode(&req); err != nil {
		writeErr(w, 400, "invalid JSON: "+err.Error())
		return
	}
	if req.Source == "" && req.SourcePath == "" {
		writeErr(w, 400, "source or source_path must not be empty")
		return
	}
	if req.SourcePath != "" {
		abs, aerr := filepath.Abs(req.SourcePath)
		if aerr != nil {
			writeErr(w, 400, "source_path invalid: "+req.SourcePath)
			return
		}
		if _, serr := os.Stat(abs); serr != nil {
			writeErr(w, 400, "source_path not accessible: "+req.SourcePath)
			return
		}
		req.Source = abs
	}
	id, err := newID()
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	lang := req.TargetLang
	if lang == "" {
		lang = h.Defaults.TargetLang
	}
	if lang == "" {
		lang = "zh"
	}
	ocr := mergeMaps(h.Defaults.OCR, req.OCR)
	llm := mergeMaps(h.Defaults.LLM, req.LLM)
	taskParams := map[string]any{
		"job_id":      id,
		"stage":       "ocr",
		"workdir":     "pending",
		"source":      req.Source,
		"target_lang": lang,
		"ocr":         ocr,
		"llm":         llm,
	}
	renderRaw := bytes.TrimSpace(req.Render)
	switch {
	case len(renderRaw) == 0:
		if h.Defaults.RenderEnabled {
			taskParams["render"] = mergeMaps(h.Defaults.Render, nil)
		}
	case bytes.Equal(renderRaw, []byte("null")):
		// Explicit null disables default render for this job.
	default:
		var reqRender map[string]any
		if err := json.Unmarshal(renderRaw, &reqRender); err != nil {
			writeErr(w, 400, "invalid render object: "+err.Error())
			return
		}
		taskParams["render"] = mergeMaps(h.Defaults.Render, reqRender)
	}
	params, err := worker.Normalize(taskParams, h.Schema)
	if err != nil {
		writeErr(w, 400, "bad task parameters: "+err.Error())
		return
	}
	now := time.Now().UTC()
	j := &job.Job{ID: id, Status: job.Queued, Stage: "ocr", Params: params, CreatedAt: now, UpdatedAt: now}
	if err := h.Store.Create(j); err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	go h.Exec.Run(context.Background(), j)
	writeJSON(w, 202, map[string]any{"id": id, "status": j.Status, "poll": "/v1/jobs/" + id})
}

func (h *Handler) list(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{"jobs": h.Store.List()})
}

func (h *Handler) get(w http.ResponseWriter, r *http.Request) {
	j, ok := h.Store.Get(r.PathValue("id"))
	if !ok {
		writeErr(w, 404, "job not found")
		return
	}
	writeJSON(w, 200, j)
}

func (h *Handler) result(w http.ResponseWriter, r *http.Request) {
	j, ok := h.Store.Get(r.PathValue("id"))
	if !ok {
		writeErr(w, 404, "job not found")
		return
	}
	if j.Status != job.Succeeded {
		resp := map[string]any{"id": j.ID, "status": j.Status, "stage": j.Stage, "error": j.Error}
		if j.Result != nil {
			resp["result"] = j.Result
		}
		writeJSON(w, 409, resp)
		return
	}
	writeJSON(w, 200, map[string]any{"id": j.ID, "result": j.Result})
}

func (h *Handler) retry(w http.ResponseWriter, r *http.Request) {
	j, ok := h.Store.Get(r.PathValue("id"))
	if !ok {
		writeErr(w, 404, "job not found")
		return
	}
	h.Store.Update(j.ID, func(jj *job.Job) {
		jj.Status = job.Queued
		jj.Stage = "ocr"
		jj.Error = ""
		jj.Result = nil
	})
	go h.Exec.Run(context.Background(), j)
	writeJSON(w, 202, map[string]any{"id": j.ID, "status": job.Queued})
}

func newID() (string, error) {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("generate id: %w", err)
	}
	return hex.EncodeToString(b), nil
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"error": msg})
}

func mergeMaps(defaults, overrides map[string]any) map[string]any {
	out := make(map[string]any, len(defaults)+len(overrides))
	for k, v := range defaults {
		if sub, ok := v.(map[string]any); ok {
			out[k] = mergeMaps(sub, nil)
		} else {
			out[k] = v
		}
	}
	for k, v := range overrides {
		if subOver, ok := v.(map[string]any); ok {
			if subDef, ok2 := out[k].(map[string]any); ok2 {
				out[k] = mergeMaps(subDef, subOver)
				continue
			}
			out[k] = mergeMaps(nil, subOver)
			continue
		}
		out[k] = v
	}
	return out
}
