package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"miniretain/internal/job"
)

const (
	defaultOCRStageTimeout    = 900 * time.Second
	defaultLocalOCRTimeout    = 180 * time.Second
	defaultLLMStageTimeout    = 1800 * time.Second
	defaultRenderStageTimeout = 600 * time.Second
)

// Executor drives the Python worker as an external process. This is the same
// shape as retain-pdf's pipeline (one process per stage, files as the only
// handoff), kept deliberately small.
type Executor struct {
	Python     string // python3
	WorkerPath string // absolute path to python/worker.py
	SchemaPath string // absolute path to contracts/task.spec.schema.json
	DataRoot          string // absolute path to data/jobs
	MaxWorkers        int
	MaxLLMConcurrency int // optional global in-flight LLM request budget across all concurrent jobs (0 = per-job only)
	StageTimeouts     map[string]time.Duration // optional per-stage timeout overrides ("ocr", "llm", "render")
	Store             *job.Store
	ExtraEnv          []string // optional KEY=VAL overrides passed to Python worker processes

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
			if stage == "ocr" && errors.Is(err, context.DeadlineExceeded) {
				if fbJob := e.buildOCRFallbackJob(j); fbJob != nil {
					if fbErr := e.runStage(ctx, fbJob, "ocr"); fbErr == nil {
						continue
					} else {
						err = fmt.Errorf("%w; fallback ocr failed: %v", err, fbErr)
					}
				}
			}
			setStatus(e.Store, j.ID, job.Failed, stage, err.Error())
			return
		}
	}
	setStatus(e.Store, j.ID, job.Succeeded, stages[len(stages)-1], "")
}

func toFloat64(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	default:
		return 0, false
	}
}

func (e *Executor) computeStageTimeout(j *job.Job, stage string) time.Duration {
	if e.StageTimeouts != nil {
		if d := e.StageTimeouts[stage]; d > 0 {
			return d
		}
	}
	switch stage {
	case "ocr":
		ocrMap, _ := j.Params["ocr"].(map[string]any)
		prov, _ := ocrMap["provider"].(string)
		if prov == "pymupdf" || prov == "stub" {
			return defaultLocalOCRTimeout
		}
		cfgMap, _ := ocrMap["config"].(map[string]any)
		var totalSec float64
		for _, subKey := range []string{"mineru", "paddle_cloud"} {
			if sub, ok := cfgMap[subKey].(map[string]any); ok {
				if ts, ok := toFloat64(sub["total_timeout_s"]); ok && ts > 0 {
					totalSec += ts
				} else if ts, ok := toFloat64(sub["timeout_s"]); ok && ts > 0 {
					totalSec += ts
				}
			}
		}
		if totalSec > 0 {
			dur := time.Duration(totalSec+120) * time.Second
			if dur > 1500*time.Second {
				dur = 1500 * time.Second
			}
			if dur < 120*time.Second {
				dur = 120 * time.Second
			}
			return dur
		}
		return defaultOCRStageTimeout
	case "llm":
		return defaultLLMStageTimeout
	case "render":
		return defaultRenderStageTimeout
	default:
		return defaultOCRStageTimeout
	}
}

func (e *Executor) buildOCRFallbackJob(j *job.Job) *job.Job {
	if j == nil {
		return nil
	}
	ocrOrig, _ := j.Params["ocr"].(map[string]any)
	origProv, _ := ocrOrig["provider"].(string)
	if origProv == "" || origProv == "pymupdf" || origProv == "stub" {
		return nil
	}
	cfgOrig, _ := ocrOrig["config"].(map[string]any)
	subKey := origProv
	if origProv == "paddle-cloud" {
		subKey = "paddle_cloud"
	} else if origProv == "paddle-http" {
		subKey = "paddle_http"
	}
	fb := ""
	if sub, ok := cfgOrig[subKey].(map[string]any); ok {
		fb, _ = sub["fallback_provider"].(string)
	}
	if fb == "" {
		if autoSub, ok := cfgOrig["auto"].(map[string]any); ok {
			fb, _ = autoSub["fallback_provider"].(string)
		}
	}
	fbLow := strings.ToLower(strings.TrimSpace(fb))
	if fbLow == "none" {
		return nil
	}
	targetProv := "pymupdf"
	if fbLow == "paddle-http" {
		targetProv = "paddle-http"
	}
	paramsCopy := cloneMap(j.Params)
	ocrNew := map[string]any{
		"provider": targetProv,
		"config": map[string]any{
			"fallback_from_timeout": origProv,
		},
	}
	if cfgOrig != nil {
		if t, ok := cfgOrig["scanned_char_threshold"]; ok {
			ocrNew["config"].(map[string]any)["scanned_char_threshold"] = t
		}
	}
	paramsCopy["ocr"] = ocrNew
	return &job.Job{
		ID:     j.ID,
		Params: paramsCopy,
	}
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

	// Remove any stale result file from a previous /retry run so a crash before
	// writing a fresh result never surfaces stale diagnostics from an earlier run.
	resultPath := filepath.Join(wd, stage+".result.json")
	if rmErr := os.Remove(resultPath); rmErr != nil && !os.IsNotExist(rmErr) {
		return fmt.Errorf("remove stale %s result: %w", stage, rmErr)
	}

	// The worker reads BOTH the spec and the schema; no defaults are hardcoded
	// in Python either.
	stageTimeout := e.computeStageTimeout(j, stage)
	stageCtx, cancel := context.WithTimeout(ctx, stageTimeout)
	defer cancel()

	started := time.Now()
	cmd := exec.CommandContext(stageCtx, e.Python, e.WorkerPath,
		"--spec", specPath, "--schema", e.SchemaPath)
	cmd.Dir = wd
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	cmd.WaitDelay = 5 * time.Second
	extraEnv := append([]string(nil), e.ExtraEnv...)
	if stage == "llm" && e.MaxLLMConcurrency > 0 {
		slotDir := filepath.Join(e.DataRoot, ".llm_slots")
		if err := os.MkdirAll(slotDir, 0o755); err != nil {
			return fmt.Errorf("mkdir llm slot dir: %w", err)
		}
		extraEnv = append(extraEnv,
			fmt.Sprintf("MINIRETAIN_LLM_GLOBAL_CONCURRENCY=%d", e.MaxLLMConcurrency),
			"MINIRETAIN_LLM_SLOT_DIR="+slotDir,
		)
	}
	if len(extraEnv) > 0 {
		cmd.Env = mergeEnv(os.Environ(), extraEnv)
	}
	out, err := cmd.CombinedOutput()
	durMS := time.Since(started).Milliseconds()
	if b, rerr := os.ReadFile(resultPath); rerr == nil {
		var result map[string]any
		if uerr := json.Unmarshal(b, &result); uerr == nil {
			result["stage_wall_ms"] = durMS
			e.Store.Update(j.ID, func(jj *job.Job) {
				if jj.Result == nil {
					jj.Result = map[string]any{}
				}
				jj.Result[stage] = result
			})
		} else if err == nil {
			return fmt.Errorf("decode %s result: %w", stage, uerr)
		}
	} else if err == nil {
		return fmt.Errorf("read %s result: %w", stage, rerr)
	}
	if err != nil {
		tail := string(out)
		if len(tail) > 800 {
			tail = tail[len(tail)-800:]
		}
		if stageErr := stageCtx.Err(); stageErr != nil {
			return fmt.Errorf("python %s stage timed out after %s: %w: %s", stage, stageTimeout, stageErr, strings.TrimSpace(tail))
		}
		return fmt.Errorf("python %s stage failed: %w: %s", stage, err, strings.TrimSpace(tail))
	}
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

func mergeEnv(base, overrides []string) []string {
	idx := make(map[string]int, len(base))
	out := make([]string, 0, len(base)+len(overrides))
	for _, kv := range base {
		k, _, _ := strings.Cut(kv, "=")
		idx[k] = len(out)
		out = append(out, kv)
	}
	for _, kv := range overrides {
		k, _, _ := strings.Cut(kv, "=")
		if pos, ok := idx[k]; ok {
			out[pos] = kv
		} else {
			idx[k] = len(out)
			out = append(out, kv)
		}
	}
	return out
}
