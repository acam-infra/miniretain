// Package api is the external task intake surface.
package api

import (
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

type Handler struct {
	Store  *job.Store
	Exec   *worker.Executor
	Schema *worker.Schema
}

func New(store *job.Store, exec *worker.Executor, schema *worker.Schema) http.Handler {
	h := &Handler{Store: store, Exec: exec, Schema: schema}
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
	writeJSON(w, 200, map[string]any{"ok": true, "time": time.Now().UTC()})
}

type createReq struct {
	Source     string         `json:"source"`
	SourcePath string         `json:"source_path"`
	TargetLang string         `json:"target_lang"`
	OCR        map[string]any `json:"ocr"`
	LLM        map[string]any `json:"llm"`
	Render     map[string]any `json:"render"`
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
		lang = "zh"
	}
	ocr := req.OCR
	if ocr == nil {
		ocr = map[string]any{}
	}
	llm := req.LLM
	if llm == nil {
		llm = map[string]any{}
	}
	taskParams := map[string]any{
		"job_id":      id,
		"stage":       "ocr",
		"workdir":     "pending",
		"source":      req.Source,
		"target_lang": lang,
		"ocr":         ocr,
		"llm":         llm,
	}
	if req.Render != nil {
		taskParams["render"] = req.Render
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
		writeJSON(w, 409, map[string]any{"id": j.ID, "status": j.Status, "error": j.Error})
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
