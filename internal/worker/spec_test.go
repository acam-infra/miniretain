package worker

import "testing"

// This test is the cross-language guard: it proves Go takes batch_size (and
// every other default) from the schema, not from a second hardcoded copy.
func TestBuildSpecUsesSchemaDefaults(t *testing.T) {
	s, err := LoadSchema("../../contracts/task.spec.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	raw := map[string]any{
		"job_id":      "j1",
		"stage":       "llm",
		"workdir":     "/tmp/j1",
		"source":      "hello",
		"target_lang": "zh",
		"ocr":         map[string]any{},
		"llm":         map[string]any{}, // no batch_size on purpose
	}
	spec, err := BuildSpec(raw, s)
	if err != nil {
		t.Fatal(err)
	}
	if spec.LLM.BatchSize != 8 {
		t.Fatalf("batch_size = %d, want 8 from schema", spec.LLM.BatchSize)
	}
	if spec.LLM.Concurrency != 2 || spec.LLM.MaxRetries != 2 || spec.LLM.TimeoutMS != 30000 {
		t.Fatalf("defaults not applied: %+v", spec.LLM)
	}
	if spec.OCR.Provider != "stub" || spec.LLM.Provider != "stub" {
		t.Fatalf("provider defaults not applied: %+v", spec)
	}
	if spec.LLM.AllowPartial {
		t.Fatalf("allow_partial default should be false")
	}
}

func TestBuildSpecRenderDefaultsAndBooleans(t *testing.T) {
	s, err := LoadSchema("../../contracts/task.spec.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	raw := map[string]any{
		"job_id":      "j3",
		"stage":       "render",
		"workdir":     "/tmp/j3",
		"source":      "hello.pdf",
		"target_lang": "zh",
		"ocr":         map[string]any{"provider": "pymupdf"},
		"llm":         map[string]any{"allow_partial": true},
		"render":      map[string]any{"strict": true},
	}
	spec, err := BuildSpec(raw, s)
	if err != nil {
		t.Fatal(err)
	}
	if spec.OCR.Provider != "pymupdf" {
		t.Fatalf("ocr.provider = %q, want pymupdf", spec.OCR.Provider)
	}
	if !spec.LLM.AllowPartial {
		t.Fatalf("llm.allow_partial should be true")
	}
	if spec.Render == nil || spec.Render.LayoutMode != "auto" || spec.Render.BilingualMode != "auto" || !spec.Render.Strict {
		t.Fatalf("render defaults/booleans mismatch: %+v", spec.Render)
	}
}

func TestBuildSpecRejectsMissingRequired(t *testing.T) {
	s, err := LoadSchema("../../contracts/task.spec.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	raw := map[string]any{"job_id": "j2"} // most fields missing
	if _, err := BuildSpec(raw, s); err == nil {
		t.Fatal("expected error for missing required fields")
	}
}
