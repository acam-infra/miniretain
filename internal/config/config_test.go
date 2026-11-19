package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"miniretain/internal/worker"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	abs, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	return abs
}

func loadTestSchema(t *testing.T) *worker.Schema {
	t.Helper()
	s, err := worker.LoadSchema(filepath.Join(repoRoot(t), "contracts", "task.spec.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestLoadExampleYAMLValidates(t *testing.T) {
	root := repoRoot(t)
	tmp := t.TempDir()
	// Copy example YAML into repo-relative temp file pointing to repoRoot
	raw, err := os.ReadFile(filepath.Join(root, "configs", "miniretain.example.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	patched := strings.Replace(string(raw), `root: "."`, `root: "`+root+`"`, 1)
	cfgPath := filepath.Join(tmp, "miniretain.yaml")
	if err := os.WriteFile(cfgPath, []byte(patched), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(cfgPath)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	schema := loadTestSchema(t)
	if err := cfg.Validate(schema); err != nil {
		t.Fatalf("Validate failed: %v", err)
	}

	if cfg.Server.Addr != ":18080" {
		t.Fatalf("addr = %q, want :18080", cfg.Server.Addr)
	}
	if cfg.Paths.DataDir != filepath.Join(tmp, "data", "jobs") {
		t.Fatalf("data_dir = %q, want relative to YAML dir %q", cfg.Paths.DataDir, filepath.Join(tmp, "data", "jobs"))
	}
	if cfg.OCR.Provider != "auto" || cfg.LLM.Provider != "openai-compat" || cfg.LLM.Model != "gemini-3.6-flash" {
		t.Fatalf("unexpected providers/model: ocr=%+v llm=%+v", cfg.OCR, cfg.LLM)
	}
	if !cfg.RenderEnabledByDefault() || cfg.Render.Engine != "typst" {
		t.Fatalf("unexpected render config: %+v", cfg.Render)
	}
}

func TestLoadRejectsUnknownField(t *testing.T) {
	tmp := t.TempDir()
	cfgPath := filepath.Join(tmp, "bad.yaml")
	content := `
server:
  addr: ":18080"
  nonexistent_field: 123
`
	if err := os.WriteFile(cfgPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Load(cfgPath)
	if err == nil || !strings.Contains(err.Error(), "nonexistent_field") {
		t.Fatalf("expected unknown field error mentioning nonexistent_field, got: %v", err)
	}
}

func TestLoadRejectsWrongType(t *testing.T) {
	tmp := t.TempDir()
	cfgPath := filepath.Join(tmp, "badtype.yaml")
	content := `
runtime:
  max_workers: "not-a-number"
`
	if err := os.WriteFile(cfgPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Load(cfgPath)
	if err == nil {
		t.Fatal("expected type mismatch error for runtime.max_workers")
	}
}

func TestValidateRejectsInvalidBackendsAndMissingKeyFile(t *testing.T) {
	root := repoRoot(t)
	schema := loadTestSchema(t)

	cases := []struct {
		name    string
		yaml    string
		wantSub string
	}{
		{
			name: "invalid ocr provider",
			yaml: "paths:\n  root: " + root + "\nocr:\n  provider: magic-ocr\n",
			wantSub: "ocr.provider",
		},
		{
			name: "invalid llm provider",
			yaml: "paths:\n  root: " + root + "\nllm:\n  provider: unknown-llm\n",
			wantSub: "llm.provider",
		},
		{
			name: "invalid render engine",
			yaml: "paths:\n  root: " + root + "\nrender:\n  engine: latex\n",
			wantSub: "render.engine",
		},
		{
			name: "zero max_workers",
			yaml: "paths:\n  root: " + root + "\nruntime:\n  max_workers: 0\n",
			wantSub: "runtime.max_workers",
		},
		{
			name: "missing api_key_file",
			yaml: "paths:\n  root: " + root + "\nllm:\n  api_key_file: ./does-not-exist.key\n",
			wantSub: "llm.api_key_file",
		},
		{
			name: "paddle-http without base_url",
			yaml: "paths:\n  root: " + root + "\nocr:\n  provider: paddle-http\n",
			wantSub: "paddle_http.base_url",
		},
		{
			name: "invalid auto fallback_provider",
			yaml: "paths:\n  root: " + root + "\nocr:\n  provider: auto\n  auto:\n    fallback_provider: invalid\n",
			wantSub: "fallback_provider",
		},
	}

	t.Setenv("PADDLE_OCR_URL", "")
	t.Setenv("OPENAI_BASE_URL", "")

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := filepath.Join(t.TempDir(), "cfg.yaml")
			if err := os.WriteFile(p, []byte(tc.yaml), 0o644); err != nil {
				t.Fatal(err)
			}
			cfg, err := Load(p)
			if err != nil {
				t.Fatalf("unexpected load error: %v", err)
			}
			err = cfg.Validate(schema)
			if err == nil || !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("Validate() err = %v, want substring %q", err, tc.wantSub)
			}
		})
	}
}

func TestCLIOverridesAndWorkerEnvSecrets(t *testing.T) {
	root := repoRoot(t)
	tmp := t.TempDir()
	keyFile := filepath.Join(tmp, "secret.key")
	if err := os.WriteFile(keyFile, []byte("super-secret-value\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("TEST_MINIRETAIN_KEY", "env-secret-123")

	yamlText := `
server:
  addr: ":18080"
paths:
  root: ` + root + `
runtime:
  python: python3
  typst: /usr/local/bin/typst
  max_workers: 2
ocr:
  provider: paddle-http
  paddle_http:
    base_url: http://127.0.0.1:18085
    dpi: 200
llm:
  provider: openai-compat
  base_url: http://127.0.0.1:8083/v1
  api_key_env: TEST_MINIRETAIN_KEY
  api_key_file: ./secret.key
  model: gemini-3.6-flash
`
	cfgPath := filepath.Join(tmp, "cfg.yaml")
	if err := os.WriteFile(cfgPath, []byte(yamlText), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(cfgPath)
	if err != nil {
		t.Fatal(err)
	}
	newAddr := ":19999"
	newWorkers := 4
	if err := cfg.ApplyCLIOverrides(CLIOverrides{Addr: &newAddr, MaxWorkers: &newWorkers}); err != nil {
		t.Fatal(err)
	}
	if cfg.Server.Addr != ":19999" || cfg.EffectiveMaxWorkers() != 4 {
		t.Fatalf("CLI override failed: addr=%q workers=%d", cfg.Server.Addr, cfg.EffectiveMaxWorkers())
	}

	schema := loadTestSchema(t)
	if err := cfg.Validate(schema); err != nil {
		t.Fatalf("Validate failed: %v", err)
	}

	env := cfg.WorkerEnv()
	joined := strings.Join(env, "\n")
	for _, want := range []string{
		"TYPST_BIN=/usr/local/bin/typst",
		"OPENAI_BASE_URL=http://127.0.0.1:8083/v1",
		"OPENAI_API_KEY=env-secret-123",
		"OPENAI_API_KEY_FILE=" + keyFile,
		"PADDLE_OCR_URL=http://127.0.0.1:18085",
		"PADDLE_OCR_DPI=200",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("WorkerEnv missing %q; got:\n%s", want, joined)
		}
	}

	// Health summary and LLM defaults must NEVER contain secrets.
	health := cfg.HealthSummary()
	for k, v := range health {
		if strings.Contains(strings.ToLower(k), "key") || strings.Contains(strings.ToLower(k), "secret") {
			t.Fatalf("HealthSummary leaked secret key %q=%v", k, v)
		}
	}
	llmDef := cfg.LLMDefaultsMap()
	if _, ok := llmDef["api_key"]; ok {
		t.Fatal("LLMDefaultsMap must not expose api_key")
	}
}
