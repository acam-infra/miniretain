package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"miniretain/internal/job"
	"miniretain/internal/worker"
)

func TestCreateJobAppliesSchemaDefaultsAndValidates(t *testing.T) {
	schema, err := worker.LoadSchema("../../contracts/task.spec.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	store := job.NewStore()
	exec := worker.New("python3", "../../python/worker.py", "../../contracts/task.spec.schema.json", t.TempDir(), 1, store)
	h := New(store, exec, schema)

	// Invalid render.layout_mode must be rejected with 400
	badBody, _ := json.Marshal(map[string]any{
		"source": "hello",
		"render": map[string]any{"layout_mode": "invalid-mode"},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/jobs", bytes.NewReader(badBody))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for invalid layout_mode, got %d: %s", rec.Code, rec.Body.String())
	}

	// Valid health check
	hReq := httptest.NewRequest(http.MethodGet, "/health", nil)
	hRec := httptest.NewRecorder()
	h.ServeHTTP(hRec, hReq)
	if hRec.Code != http.StatusOK {
		t.Fatalf("expected 200 on /health, got %d", hRec.Code)
	}
}

func TestJobDefaultsAndPerRequestOverrides(t *testing.T) {
	schema, err := worker.LoadSchema("../../contracts/task.spec.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	store := job.NewStore()
	exec := worker.New("python3", "../../python/worker.py", "../../contracts/task.spec.schema.json", t.TempDir(), 1, store)
	h := NewWithOptions(store, exec, schema, Options{
		Defaults: JobDefaults{
			TargetLang: "zh",
			OCR: map[string]any{
				"provider": "pymupdf",
				"config": map[string]any{
					"auto": map[string]any{"scanned_char_threshold": 25},
				},
			},
			LLM: map[string]any{
				"provider":    "stub",
				"model":       "yaml-default-model",
				"batch_size":  16,
				"concurrency": 4,
			},
			Render: map[string]any{
				"engine": "typst",
				"strict": true,
			},
			RenderEnabled: true,
		},
		HealthInfo: map[string]any{
			"config_mode":  "yaml",
			"ocr_provider": "pymupdf",
		},
	})

	// 1. /health includes safe metadata
	hReq := httptest.NewRequest(http.MethodGet, "/health", nil)
	hRec := httptest.NewRecorder()
	h.ServeHTTP(hRec, hReq)
	if hRec.Code != http.StatusOK {
		t.Fatalf("expected 200 on /health, got %d", hRec.Code)
	}
	var healthBody map[string]any
	if err := json.Unmarshal(hRec.Body.Bytes(), &healthBody); err != nil {
		t.Fatal(err)
	}
	if healthBody["ok"] != true || healthBody["config_mode"] != "yaml" || healthBody["ocr_provider"] != "pymupdf" {
		t.Fatalf("unexpected /health payload: %+v", healthBody)
	}

	// 2. Request omitting ocr/llm/render inherits YAML defaults (including render.enabled=true)
	body1, _ := json.Marshal(map[string]any{
		"source": "hello world",
		"ocr":    map[string]any{"provider": "stub"}, // override only provider
		"llm":    map[string]any{"model": "req-override-model"},
	})
	req1 := httptest.NewRequest(http.MethodPost, "/v1/jobs", bytes.NewReader(body1))
	rec1 := httptest.NewRecorder()
	h.ServeHTTP(rec1, req1)
	if rec1.Code != http.StatusAccepted {
		t.Fatalf("expected 202, got %d: %s", rec1.Code, rec1.Body.String())
	}
	var resp1 map[string]any
	if err := json.Unmarshal(rec1.Body.Bytes(), &resp1); err != nil {
		t.Fatal(err)
	}
	j1, ok := store.Get(resp1["id"].(string))
	if !ok {
		t.Fatal("job not found in store")
	}
	ocrMap := j1.Params["ocr"].(map[string]any)
	llmMap := j1.Params["llm"].(map[string]any)
	renderMap, hasRender := j1.Params["render"].(map[string]any)
	if ocrMap["provider"] != "stub" {
		t.Fatalf("expected request override ocr.provider=stub, got %v", ocrMap["provider"])
	}
	if llmMap["model"] != "req-override-model" || llmMap["batch_size"] != 16 {
		t.Fatalf("expected llm override + yaml default batch_size=16, got %+v", llmMap)
	}
	if !hasRender || renderMap["engine"] != "typst" || renderMap["strict"] != true {
		t.Fatalf("expected default render stage enabled with typst/strict, got %+v (has=%v)", renderMap, hasRender)
	}

	// 3. Explicit "render": null disables default render for that job
	body2 := []byte(`{"source":"hello","ocr":{"provider":"stub"},"render":null}`)
	req2 := httptest.NewRequest(http.MethodPost, "/v1/jobs", bytes.NewReader(body2))
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec2, req2)
	if rec2.Code != http.StatusAccepted {
		t.Fatalf("expected 202, got %d: %s", rec2.Code, rec2.Body.String())
	}
	var resp2 map[string]any
	_ = json.Unmarshal(rec2.Body.Bytes(), &resp2)
	j2, _ := store.Get(resp2["id"].(string))
	if _, hasRender2 := j2.Params["render"]; hasRender2 {
		t.Fatalf("expected render omitted when request passes render:null, got %+v", j2.Params["render"])
	}
}

func TestFailedJobResultReturnsStageAndDiagnostics(t *testing.T) {
	store := job.NewStore()
	j := &job.Job{
		ID:     "fail-diag-1",
		Status: job.Failed,
		Stage:  "llm",
		Error:  "llm stage failed: p1/p0b0011: untranslated verbatim echo [attempts=3]",
		Result: map[string]any{
			"llm": map[string]any{
				"failed_stage":    "llm",
				"critical_failed": 1,
			},
		},
	}
	if err := store.Create(j); err != nil {
		t.Fatal(err)
	}
	h := New(store, nil, nil)
	req := httptest.NewRequest(http.MethodGet, "/v1/jobs/fail-diag-1/result", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Fatalf("expected 409, got %d", rec.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["stage"] != "llm" {
		t.Fatalf("expected stage=llm in 409 body, got %+v", body)
	}
	resMap, ok := body["result"].(map[string]any)
	if !ok || resMap["llm"] == nil {
		t.Fatalf("expected llm diagnostics in 409 result body, got %+v", body)
	}
}
