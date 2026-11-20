package worker

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"miniretain/internal/job"
)

func TestRunStageClearsStaleResultOnRetry(t *testing.T) {
	schemaPath, err := filepath.Abs(filepath.Join("..", "..", "contracts", "task.spec.schema.json"))
	if err != nil {
		t.Fatal(err)
	}

	makeJob := func(id string) *job.Job {
		return &job.Job{
			ID:     id,
			Status: job.Queued,
			Stage:  "ocr",
			Params: map[string]any{
				"source":      "hello",
				"target_lang": "zh",
				"ocr":         map[string]any{"provider": "stub"},
				"llm":         map[string]any{"provider": "stub"},
			},
		}
	}

	t.Run("crash before writing new result does not surface stale diagnostics", func(t *testing.T) {
		tmp := t.TempDir()
		dataRoot := filepath.Join(tmp, "jobs")
		jobID := "retry-stale-1"
		wd := filepath.Join(dataRoot, jobID)
		if err := os.MkdirAll(wd, 0o755); err != nil {
			t.Fatal(err)
		}
		stalePath := filepath.Join(wd, "ocr.result.json")
		if err := os.WriteFile(stalePath, []byte(`{"stale":true,"failed_stage":"ocr"}`), 0o644); err != nil {
			t.Fatal(err)
		}

		// Fake worker exits non-zero immediately without writing ocr.result.json.
		fakeWorker := filepath.Join(tmp, "fail_no_file.sh")
		if err := os.WriteFile(fakeWorker, []byte("#!/bin/sh\necho 'crash before result' >&2\nexit 1\n"), 0o755); err != nil {
			t.Fatal(err)
		}

		store := job.NewStore()
		j := makeJob(jobID)
		if err := store.Create(j); err != nil {
			t.Fatal(err)
		}
		exec := New("/bin/sh", fakeWorker, schemaPath, dataRoot, 1, store)
		exec.Run(context.Background(), j)

		got, _ := store.Get(jobID)
		if got.Status != job.Failed {
			t.Fatalf("expected failed status, got %s", got.Status)
		}
		if _, err := os.Stat(stalePath); !os.IsNotExist(err) {
			t.Fatalf("expected stale ocr.result.json to be removed, stat err=%v", err)
		}
		if got.Result != nil && got.Result["ocr"] != nil {
			t.Fatalf("expected no stale ocr diagnostics in job.Result, got %+v", got.Result["ocr"])
		}
	})

	t.Run("failure after writing fresh result surfaces only fresh diagnostics", func(t *testing.T) {
		tmp := t.TempDir()
		dataRoot := filepath.Join(tmp, "jobs")
		jobID := "retry-fresh-2"
		wd := filepath.Join(dataRoot, jobID)
		if err := os.MkdirAll(wd, 0o755); err != nil {
			t.Fatal(err)
		}
		stalePath := filepath.Join(wd, "ocr.result.json")
		if err := os.WriteFile(stalePath, []byte(`{"stale":true,"attempt_tag":"old"}`), 0o644); err != nil {
			t.Fatal(err)
		}

		// Fake worker writes fresh diagnostics to cwd/ocr.result.json then exits 1.
		fakeWorker := filepath.Join(tmp, "fail_with_fresh.sh")
		script := "#!/bin/sh\nprintf '{\"fresh\":true,\"attempt_tag\":\"new\"}' > ocr.result.json\necho 'fresh diag fail' >&2\nexit 1\n"
		if err := os.WriteFile(fakeWorker, []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}

		store := job.NewStore()
		j := makeJob(jobID)
		if err := store.Create(j); err != nil {
			t.Fatal(err)
		}
		exec := New("/bin/sh", fakeWorker, schemaPath, dataRoot, 1, store)
		exec.Run(context.Background(), j)

		got, _ := store.Get(jobID)
		if got.Status != job.Failed {
			t.Fatalf("expected failed status, got %s", got.Status)
		}
		if got.Result == nil || got.Result["ocr"] == nil {
			t.Fatalf("expected fresh ocr diagnostics in job.Result, got %+v", got.Result)
		}
		b, _ := json.Marshal(got.Result["ocr"])
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		if m["fresh"] != true || m["attempt_tag"] != "new" {
			t.Fatalf("expected fresh diagnostics, got %+v", m)
		}
		if _, hasStale := m["stale"]; hasStale {
			t.Fatalf("unexpected stale key in fresh diagnostics: %+v", m)
		}
	})
}
