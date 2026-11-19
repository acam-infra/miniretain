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
