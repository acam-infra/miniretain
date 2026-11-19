package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"miniretain/internal/job"
)

// Executor drives the Python worker as an external process. This is the same
// shape as retain-pdf's pipeline (one process per stage, files as the only
// handoff), kept deliberately small.
type Executor struct {
	Python     string // python3
	WorkerPath string // absolute path to python/worker.py
	SchemaPath string // absolute path to contracts/task.spec.schema.json
	DataRoot   string // absolute path to data/jobs
	MaxWorkers int
	Store      *job.Store

	sem chan struct{}
}

func New(python, workerPath, schemaPath, dataRoot string, maxWorkers int, store *job.Store) *Executor {
	if maxWorkers < 1 {
		maxWorkers = 1
	}
	return &Executor{
		Python: python, WorkerPath: workerPath, SchemaPath: schemaPath,
		DataRoot: dataRoot, MaxWorkers: maxWorkers, Store: store,
		sem: make(chan struct{}, maxWorkers),
	}
}

// Run executes ocr -> llm, plus an optional render stage when the caller asks
// for layout-preserving output. Each stage has its own spec/result files so a
// failed stage can be retried independently.
func (e *Executor) Run(ctx context.Context, j *job.Job) {
	stages := []string{"ocr", "llm"}
	if _, ok := j.Params["render"]; ok {
		stages = append(stages, "render")
	}
	setStatus(e.Store, j.ID, job.Running, stages[0], "")
	for _, stage := range stages {
		if err := e.runStage(ctx, j, stage); err != nil {
			setStatus(e.Store, j.ID, job.Failed, stage, err.Error())
			return
		}
	}
	setStatus(e.Store, j.ID, job.Succeeded, stages[len(stages)-1], "")
}

func (e *Executor) runStage(ctx context.Context, j *job.Job, stage string) error {
	wd := filepath.Join(e.DataRoot, j.ID)
	if err := os.MkdirAll(wd, 0o755); err != nil {
		return fmt.Errorf("mkdir workdir: %w", err)
	}

	raw := cloneMap(j.Params)
	raw["stage"] = stage
	raw["job_id"] = j.ID
	raw["workdir"] = wd

	schema, err := LoadSchema(e.SchemaPath)
	if err != nil {
		return err
	}
	spec, err := BuildSpec(raw, schema)
	if err != nil {
		return fmt.Errorf("build spec for %s: %w", stage, err)
	}
	specBytes, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		return err
	}
	specPath := filepath.Join(wd, stage+".spec.json")
	if err := os.WriteFile(specPath, specBytes, 0o644); err != nil {
		return err
	}

	select {
	case e.sem <- struct{}{}:
		defer func() { <-e.sem }()
	case <-ctx.Done():
		return ctx.Err()
	}

	// The worker reads BOTH the spec and the schema; no defaults are hardcoded
	// in Python either.
	started := time.Now()
	cmd := exec.CommandContext(ctx, e.Python, e.WorkerPath,
		"--spec", specPath, "--schema", e.SchemaPath)
	cmd.Dir = wd
	out, err := cmd.CombinedOutput()
	durMS := time.Since(started).Milliseconds()
	if err != nil {
		tail := string(out)
		if len(tail) > 800 {
			tail = tail[len(tail)-800:]
		}
		return fmt.Errorf("python %s stage failed: %w: %s", stage, err, strings.TrimSpace(tail))
	}

	resultPath := filepath.Join(wd, stage+".result.json")
	b, err := os.ReadFile(resultPath)
	if err != nil {
		return fmt.Errorf("read %s result: %w", stage, err)
	}
	var result map[string]any
	if err := json.Unmarshal(b, &result); err != nil {
		return fmt.Errorf("decode %s result: %w", stage, err)
	}
	result["stage_wall_ms"] = durMS
	e.Store.Update(j.ID, func(jj *job.Job) {
		if jj.Result == nil {
			jj.Result = map[string]any{}
		}
		jj.Result[stage] = result
	})
	return nil
}

func setStatus(s *job.Store, id string, st job.Status, stage, errMsg string) {
	s.Update(id, func(j *job.Job) {
		j.Status = st
		j.Stage = stage
		if errMsg != "" {
			j.Error = errMsg
		}
	})
}
