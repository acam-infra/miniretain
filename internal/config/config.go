// Package config implements single-YAML declarative startup configuration for Miniretain.
//
// Precedence (highest to lowest):
//  1. Per-request HTTP JSON body (POST /v1/jobs) for allowed task fields
//  2. Explicit CLI flags (-addr, -root, -python, -max-workers)
//  3. YAML config file (--config <file.yaml>)
//  4. Legacy environment variables (OPENAI_BASE_URL, OPENAI_API_KEY, PADDLE_OCR_URL, TYPST_BIN, ...)
//  5. Built-in / contracts/task.spec.schema.json defaults
package config

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"miniretain/internal/worker"
)

const (
	DefaultAddr                = ":18080"
	DefaultReadHeaderTimeoutMS = 5000
	DefaultPython              = "python3"
	DefaultMaxWorkers          = 2
)

// Config is the root declarative YAML configuration.
type Config struct {
	Server     ServerConfig  `yaml:"server"`
	Paths      PathsConfig   `yaml:"paths"`
	Runtime    RuntimeConfig `yaml:"runtime"`
	TargetLang string        `yaml:"target_lang"`
	OCR        OCRConfig     `yaml:"ocr"`
	LLM        LLMConfig     `yaml:"llm"`
	Render     RenderConfig  `yaml:"render"`

	// ConfigFile holds the resolved absolute path of the loaded YAML file, or "" in legacy CLI mode.
	ConfigFile string `yaml:"-"`
}

type ServerConfig struct {
	Addr                string `yaml:"addr"`
	ReadHeaderTimeoutMS *int   `yaml:"read_header_timeout_ms"`
}

type PathsConfig struct {
	Root    string `yaml:"root"`
	DataDir string `yaml:"data_dir"`
	Schema  string `yaml:"schema"`
}

type RuntimeConfig struct {
	Python            string `yaml:"python"`
	Typst             string `yaml:"typst"`
	MaxWorkers        *int   `yaml:"max_workers"`
	MaxLLMConcurrency *int   `yaml:"max_llm_concurrency"`
	OCRTimeoutS       *int   `yaml:"ocr_timeout_s"`
	LLMTimeoutS       *int   `yaml:"llm_timeout_s"`
	RenderTimeoutS    *int   `yaml:"render_timeout_s"`
}

type OCRConfig struct {
	Provider    string            `yaml:"provider"`
	PaddleHTTP  PaddleHTTPConfig  `yaml:"paddle_http"`
	PaddleCPU   PaddleCPUConfig   `yaml:"paddle_cpu"`
	PaddleCloud PaddleCloudConfig `yaml:"paddle_cloud"`
	MinerU      MinerUConfig      `yaml:"mineru"`
	Auto        AutoOCRConfig     `yaml:"auto"`
	Options     map[string]any    `yaml:"options"`
}

type PaddleCloudConfig struct {
	JobURL           string   `yaml:"job_url"`
	Model            string   `yaml:"model"`
	TokenEnv         string   `yaml:"token_env"`
	TokenFile        string   `yaml:"token_file"`
	Tokens           []string `yaml:"tokens"`
	ProxyFile        string   `yaml:"proxy_file"`
	Proxies          []string `yaml:"proxies"`
	FallbackProvider string   `yaml:"fallback_provider"`
	TimeoutS         *float64 `yaml:"timeout_s"`
	TotalTimeoutS    *float64 `yaml:"total_timeout_s"`
	LargePDFMB       *float64 `yaml:"large_pdf_mb"`
	PollIntervalS    *float64 `yaml:"poll_interval_s"`
}

type MinerUConfig struct {
	BaseURL          string   `yaml:"base_url"`
	ModelVersion     string   `yaml:"model_version"`
	Language         string   `yaml:"language"`
	TokenEnv         string   `yaml:"token_env"`
	TokenFile        string   `yaml:"token_file"`
	Tokens           []string `yaml:"tokens"`
	ProxyFile        string   `yaml:"proxy_file"`
	Proxies          []string `yaml:"proxies"`
	FallbackProvider string   `yaml:"fallback_provider"`
	TimeoutS         *float64 `yaml:"timeout_s"`
	TotalTimeoutS    *float64 `yaml:"total_timeout_s"`
	LargePDFMB       *float64 `yaml:"large_pdf_mb"`
	PollIntervalS    *float64 `yaml:"poll_interval_s"`
}

type PaddleHTTPConfig struct {
	BaseURL        string   `yaml:"base_url"`
	DPI            *float64 `yaml:"dpi"`
	TimeoutS       *float64 `yaml:"timeout_s"`
	HealthTimeoutS *float64 `yaml:"health_timeout_s"`
}

type PaddleCPUConfig struct {
	DPI  *float64 `yaml:"dpi"`
	Lang string   `yaml:"lang"`
}

type AutoOCRConfig struct {
	ScannedCharThreshold *int   `yaml:"scanned_char_threshold"`
	FallbackProvider     string `yaml:"fallback_provider"`
}

type LLMConfig struct {
	Provider       string   `yaml:"provider"`
	BaseURL        string   `yaml:"base_url"`
	APIKeyEnv      string   `yaml:"api_key_env"`
	APIKeyFile     string   `yaml:"api_key_file"`
	FallbackModels []string `yaml:"fallback_models"`
	ProxyFile      string   `yaml:"proxy_file"`
	Proxies        []string `yaml:"proxies"`
	PoolFile       string   `yaml:"pool_file"`
	Model        string `yaml:"model"`
	BatchSize    *int   `yaml:"batch_size"`
	Concurrency  *int   `yaml:"concurrency"`
	MaxRetries   *int   `yaml:"max_retries"`
	TimeoutMS    *int   `yaml:"timeout_ms"`
	AllowPartial *bool  `yaml:"allow_partial"`
}

type RenderConfig struct {
	Enabled       *bool  `yaml:"enabled"`
	Engine        string `yaml:"engine"`
	Font          string `yaml:"font"`
	LayoutMode    string `yaml:"layout_mode"`
	BilingualMode string `yaml:"bilingual_mode"`
	Output        string `yaml:"output"`
	OutputCN      string `yaml:"output_cn"`
	OutputMerged  string `yaml:"output_merged"`
	Strict        *bool  `yaml:"strict"`
}

// CLIOverrides captures only flags explicitly passed on the command line.
type CLIOverrides struct {
	Addr       *string
	Root       *string
	Python     *string
	MaxWorkers *int
}

// Default returns the baseline configuration used when no --config YAML is provided.
func Default() *Config {
	mw := DefaultMaxWorkers
	rht := DefaultReadHeaderTimeoutMS
	return &Config{
		Server: ServerConfig{
			Addr:                DefaultAddr,
			ReadHeaderTimeoutMS: &rht,
		},
		Paths: PathsConfig{
			Root: ".",
		},
		Runtime: RuntimeConfig{
			Python:     DefaultPython,
			MaxWorkers: &mw,
		},
	}
}

// Load reads and strictly decodes a YAML config file (rejecting unknown fields and type mismatches).
// Relative paths in the YAML file are resolved relative to the directory containing the YAML file.
func Load(path string) (*Config, error) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolve config path %q: %w", path, err)
	}
	raw, err := os.ReadFile(absPath)
	if err != nil {
		return nil, fmt.Errorf("read config %q: %w", absPath, err)
	}

	var cfg Config
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true)
	if err := dec.Decode(&cfg); err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("parse config %q: %w", absPath, err)
	}
	// Reject unexpected second YAML document.
	var extra any
	if err := dec.Decode(&extra); err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("parse config %q: trailing document: %w", absPath, err)
	}

	cfg.ConfigFile = absPath
	cfg.applyDefaultsAndResolvePaths(filepath.Dir(absPath))
	return &cfg, nil
}

// ResolveLegacy finalizes a Config created without --config using cwd for relative paths.
func (c *Config) ResolveLegacy() error {
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	c.applyDefaultsAndResolvePaths(cwd)
	return nil
}

// ApplyCLIOverrides applies explicitly provided CLI flags over YAML/default values.
// Relative paths passed on the CLI are resolved relative to the current working directory.
func (c *Config) ApplyCLIOverrides(o CLIOverrides) error {
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	if o.Addr != nil {
		c.Server.Addr = *o.Addr
	}
	if o.Root != nil {
		rootAbs := resolveRel(cwd, *o.Root)
		// If data_dir / schema were derived from the old root, re-derive them when root changes.
		oldDefaultData := filepath.Join(c.Paths.Root, "data", "jobs")
		oldDefaultSchema := filepath.Join(c.Paths.Root, "contracts", "task.spec.schema.json")
		c.Paths.Root = rootAbs
		if c.Paths.DataDir == "" || c.Paths.DataDir == oldDefaultData {
			c.Paths.DataDir = filepath.Join(rootAbs, "data", "jobs")
		}
		if c.Paths.Schema == "" || c.Paths.Schema == oldDefaultSchema {
			c.Paths.Schema = filepath.Join(rootAbs, "contracts", "task.spec.schema.json")
		}
	}
	if o.Python != nil {
		c.Runtime.Python = resolveExecOrPath(cwd, *o.Python)
	}
	if o.MaxWorkers != nil {
		mw := *o.MaxWorkers
		c.Runtime.MaxWorkers = &mw
	}
	return nil
}

func (c *Config) applyDefaultsAndResolvePaths(baseDir string) {
	if strings.TrimSpace(c.Server.Addr) == "" {
		c.Server.Addr = DefaultAddr
	}
	if c.Server.ReadHeaderTimeoutMS == nil {
		v := DefaultReadHeaderTimeoutMS
		c.Server.ReadHeaderTimeoutMS = &v
	}
	if strings.TrimSpace(c.Paths.Root) == "" {
		c.Paths.Root = baseDir
	} else {
		c.Paths.Root = resolveRel(baseDir, c.Paths.Root)
	}
	if strings.TrimSpace(c.Paths.DataDir) == "" {
		c.Paths.DataDir = filepath.Join(c.Paths.Root, "data", "jobs")
	} else {
		c.Paths.DataDir = resolveRel(baseDir, c.Paths.DataDir)
	}
	if strings.TrimSpace(c.Paths.Schema) == "" {
		if envSchema := strings.TrimSpace(os.Getenv("MINIRETAIN_SCHEMA")); envSchema != "" {
			c.Paths.Schema = envSchema
		} else {
			c.Paths.Schema = filepath.Join(c.Paths.Root, "contracts", "task.spec.schema.json")
		}
	} else {
		c.Paths.Schema = resolveRel(baseDir, c.Paths.Schema)
	}
	if strings.TrimSpace(c.Runtime.Python) == "" {
		c.Runtime.Python = DefaultPython
	} else {
		c.Runtime.Python = resolveExecOrPath(baseDir, c.Runtime.Python)
	}
	if strings.TrimSpace(c.Runtime.Typst) == "" {
		c.Runtime.Typst = strings.TrimSpace(os.Getenv("TYPST_BIN"))
	} else {
		c.Runtime.Typst = resolveExecOrPath(baseDir, c.Runtime.Typst)
	}
	if c.Runtime.MaxWorkers == nil {
		mw := DefaultMaxWorkers
		c.Runtime.MaxWorkers = &mw
	}
	if strings.TrimSpace(c.LLM.APIKeyFile) != "" {
		c.LLM.APIKeyFile = resolveRel(baseDir, c.LLM.APIKeyFile)
	}
	if strings.TrimSpace(c.LLM.ProxyFile) != "" {
		c.LLM.ProxyFile = resolveRel(baseDir, c.LLM.ProxyFile)
	}
	if strings.TrimSpace(c.LLM.PoolFile) != "" {
		c.LLM.PoolFile = resolveRel(baseDir, c.LLM.PoolFile)
	}
	if strings.TrimSpace(c.OCR.PaddleCloud.TokenFile) != "" {
		c.OCR.PaddleCloud.TokenFile = resolveRel(baseDir, c.OCR.PaddleCloud.TokenFile)
	}
	if strings.TrimSpace(c.OCR.PaddleCloud.ProxyFile) != "" {
		c.OCR.PaddleCloud.ProxyFile = resolveRel(baseDir, c.OCR.PaddleCloud.ProxyFile)
	}
	if strings.TrimSpace(c.OCR.MinerU.TokenFile) != "" {
		c.OCR.MinerU.TokenFile = resolveRel(baseDir, c.OCR.MinerU.TokenFile)
	}
	if strings.TrimSpace(c.OCR.MinerU.ProxyFile) != "" {
		c.OCR.MinerU.ProxyFile = resolveRel(baseDir, c.OCR.MinerU.ProxyFile)
	}
	if strings.TrimSpace(c.Render.Font) != "" && looksLikePath(c.Render.Font) {
		c.Render.Font = resolveRel(baseDir, c.Render.Font)
	}
}

func resolveRel(baseDir, p string) string {
	p = strings.TrimSpace(p)
	if p == "~" || strings.HasPrefix(p, "~/") || strings.HasPrefix(p, `~\`) {
		if home, err := os.UserHomeDir(); err == nil && home != "" {
			if p == "~" {
				return filepath.Clean(home)
			}
			return filepath.Clean(filepath.Join(home, p[2:]))
		}
	}
	if filepath.IsAbs(p) {
		return filepath.Clean(p)
	}
	return filepath.Clean(filepath.Join(baseDir, p))
}

func looksLikePath(s string) bool {
	return strings.Contains(s, "/") || strings.Contains(s, `\`) || strings.HasPrefix(s, ".")
}

func resolveExecOrPath(baseDir, s string) string {
	s = strings.TrimSpace(s)
	if looksLikePath(s) {
		return resolveRel(baseDir, s)
	}
	return s
}

// EffectiveMaxWorkers returns the configured max worker count.
func (c *Config) EffectiveMaxWorkers() int {
	if c.Runtime.MaxWorkers == nil {
		return DefaultMaxWorkers
	}
	return *c.Runtime.MaxWorkers
}

// EffectiveMaxLLMConcurrency returns the optional global in-flight LLM request budget (0 = unset).
func (c *Config) EffectiveMaxLLMConcurrency() int {
	if c.Runtime.MaxLLMConcurrency == nil {
		return 0
	}
	return *c.Runtime.MaxLLMConcurrency
}

// EffectiveReadHeaderTimeoutMS returns the configured HTTP read header timeout in ms.
func (c *Config) EffectiveReadHeaderTimeoutMS() int {
	if c.Server.ReadHeaderTimeoutMS == nil {
		return DefaultReadHeaderTimeoutMS
	}
	return *c.Server.ReadHeaderTimeoutMS
}

// StageTimeouts returns optional per-stage hard execution timeouts configured in runtime YAML.
func (c *Config) StageTimeouts() map[string]time.Duration {
	out := map[string]time.Duration{}
	if c.Runtime.OCRTimeoutS != nil && *c.Runtime.OCRTimeoutS > 0 {
		out["ocr"] = time.Duration(*c.Runtime.OCRTimeoutS) * time.Second
	}
	if c.Runtime.LLMTimeoutS != nil && *c.Runtime.LLMTimeoutS > 0 {
		out["llm"] = time.Duration(*c.Runtime.LLMTimeoutS) * time.Second
	}
	if c.Runtime.RenderTimeoutS != nil && *c.Runtime.RenderTimeoutS > 0 {
		out["render"] = time.Duration(*c.Runtime.RenderTimeoutS) * time.Second
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// Validate checks all configuration fields, paths, enums, ranges, URLs, and credential files.
func (c *Config) Validate(schema *worker.Schema) error {
	if strings.TrimSpace(c.Server.Addr) == "" {
		return fmt.Errorf("config validation: server.addr must not be empty")
	}
	if c.EffectiveReadHeaderTimeoutMS() < 1 {
		return fmt.Errorf("config validation: server.read_header_timeout_ms must be >= 1, got %d", c.EffectiveReadHeaderTimeoutMS())
	}
	if c.EffectiveMaxWorkers() < 1 {
		return fmt.Errorf("config validation: runtime.max_workers must be >= 1, got %d", c.EffectiveMaxWorkers())
	}
	if c.Runtime.MaxLLMConcurrency != nil && *c.Runtime.MaxLLMConcurrency < 1 {
		return fmt.Errorf("config validation: runtime.max_llm_concurrency must be >= 1, got %d", *c.Runtime.MaxLLMConcurrency)
	}
	if c.Runtime.OCRTimeoutS != nil && *c.Runtime.OCRTimeoutS < 1 {
		return fmt.Errorf("config validation: runtime.ocr_timeout_s must be >= 1, got %d", *c.Runtime.OCRTimeoutS)
	}
	if c.Runtime.LLMTimeoutS != nil && *c.Runtime.LLMTimeoutS < 1 {
		return fmt.Errorf("config validation: runtime.llm_timeout_s must be >= 1, got %d", *c.Runtime.LLMTimeoutS)
	}
	if c.Runtime.RenderTimeoutS != nil && *c.Runtime.RenderTimeoutS < 1 {
		return fmt.Errorf("config validation: runtime.render_timeout_s must be >= 1, got %d", *c.Runtime.RenderTimeoutS)
	}

	workerPy := filepath.Join(c.Paths.Root, "python", "worker.py")
	if _, err := os.Stat(workerPy); err != nil {
		return fmt.Errorf("config validation: paths.root=%q missing python/worker.py: %w", c.Paths.Root, err)
	}
	if _, err := os.Stat(c.Paths.Schema); err != nil {
		return fmt.Errorf("config validation: paths.schema=%q not accessible: %w", c.Paths.Schema, err)
	}

	// Validate OCR settings against schema enums and value bounds.
	ocrProviders := schema.Properties["ocr"].Properties["provider"].Enum
	if c.OCR.Provider != "" && !slices.Contains(ocrProviders, c.OCR.Provider) {
		return fmt.Errorf("config validation: ocr.provider %q is invalid; allowed: %v", c.OCR.Provider, ocrProviders)
	}
	if c.OCR.PaddleHTTP.BaseURL != "" {
		if err := validateHTTPURL("ocr.paddle_http.base_url", c.OCR.PaddleHTTP.BaseURL); err != nil {
			return err
		}
	}
	if c.OCR.Provider == "paddle-http" {
		effectiveURL := strings.TrimSpace(c.OCR.PaddleHTTP.BaseURL)
		if effectiveURL == "" {
			effectiveURL = strings.TrimSpace(os.Getenv("PADDLE_OCR_URL"))
		}
		if effectiveURL == "" {
			return fmt.Errorf("config validation: ocr.provider is \"paddle-http\" but neither ocr.paddle_http.base_url nor PADDLE_OCR_URL is set")
		}
		if err := validateHTTPURL("ocr.paddle_http.base_url", effectiveURL); err != nil {
			return err
		}
	}
	if c.OCR.PaddleHTTP.DPI != nil && *c.OCR.PaddleHTTP.DPI <= 0 {
		return fmt.Errorf("config validation: ocr.paddle_http.dpi must be > 0, got %v", *c.OCR.PaddleHTTP.DPI)
	}
	if c.OCR.PaddleHTTP.TimeoutS != nil && *c.OCR.PaddleHTTP.TimeoutS <= 0 {
		return fmt.Errorf("config validation: ocr.paddle_http.timeout_s must be > 0, got %v", *c.OCR.PaddleHTTP.TimeoutS)
	}
	if c.OCR.PaddleHTTP.HealthTimeoutS != nil && *c.OCR.PaddleHTTP.HealthTimeoutS <= 0 {
		return fmt.Errorf("config validation: ocr.paddle_http.health_timeout_s must be > 0, got %v", *c.OCR.PaddleHTTP.HealthTimeoutS)
	}
	if c.OCR.PaddleCPU.DPI != nil && *c.OCR.PaddleCPU.DPI <= 0 {
		return fmt.Errorf("config validation: ocr.paddle_cpu.dpi must be > 0, got %v", *c.OCR.PaddleCPU.DPI)
	}
	if c.OCR.Auto.ScannedCharThreshold != nil && *c.OCR.Auto.ScannedCharThreshold < 0 {
		return fmt.Errorf("config validation: ocr.auto.scanned_char_threshold must be >= 0, got %d", *c.OCR.Auto.ScannedCharThreshold)
	}
	if c.OCR.Auto.FallbackProvider != "" {
		allowedFallback := []string{"paddle-http", "paddle-cpu", "paddle-cloud", "mineru", "none"}
		if !slices.Contains(allowedFallback, c.OCR.Auto.FallbackProvider) {
			return fmt.Errorf("config validation: ocr.auto.fallback_provider %q is invalid; allowed: %v", c.OCR.Auto.FallbackProvider, allowedFallback)
		}
	}

	// Validate LLM settings.
	llmProviders := schema.Properties["llm"].Properties["provider"].Enum
	if c.LLM.Provider != "" && !slices.Contains(llmProviders, c.LLM.Provider) {
		return fmt.Errorf("config validation: llm.provider %q is invalid; allowed: %v", c.LLM.Provider, llmProviders)
	}
	if c.LLM.BaseURL != "" {
		if err := validateHTTPURL("llm.base_url", c.LLM.BaseURL); err != nil {
			return err
		}
	}
	if c.LLM.Provider == "openai-compat" && c.ConfigFile != "" {
		effectiveBase := strings.TrimSpace(c.LLM.BaseURL)
		if effectiveBase == "" {
			effectiveBase = strings.TrimSpace(os.Getenv("OPENAI_BASE_URL"))
		}
		if effectiveBase == "" {
			return fmt.Errorf("config validation: llm.provider is \"openai-compat\" but neither llm.base_url nor OPENAI_BASE_URL is set")
		}
		if err := validateHTTPURL("llm.base_url", effectiveBase); err != nil {
			return err
		}
	}
	if c.LLM.APIKeyFile != "" {
		info, err := os.Stat(c.LLM.APIKeyFile)
		if err != nil {
			return fmt.Errorf("config validation: llm.api_key_file %q not accessible: %w", c.LLM.APIKeyFile, err)
		}
		if info.IsDir() {
			return fmt.Errorf("config validation: llm.api_key_file %q is a directory, expected a file", c.LLM.APIKeyFile)
		}
	}
	if c.LLM.BatchSize != nil && *c.LLM.BatchSize < 1 {
		return fmt.Errorf("config validation: llm.batch_size must be >= 1, got %d", *c.LLM.BatchSize)
	}
	if c.LLM.Concurrency != nil && *c.LLM.Concurrency < 1 {
		return fmt.Errorf("config validation: llm.concurrency must be >= 1, got %d", *c.LLM.Concurrency)
	}
	if c.LLM.MaxRetries != nil && *c.LLM.MaxRetries < 0 {
		return fmt.Errorf("config validation: llm.max_retries must be >= 0, got %d", *c.LLM.MaxRetries)
	}
	if c.LLM.TimeoutMS != nil && *c.LLM.TimeoutMS < 1 {
		return fmt.Errorf("config validation: llm.timeout_ms must be >= 1, got %d", *c.LLM.TimeoutMS)
	}

	// Validate Render settings.
	renderProps := schema.Properties["render"].Properties
	if c.Render.Engine != "" && !slices.Contains(renderProps["engine"].Enum, c.Render.Engine) {
		return fmt.Errorf("config validation: render.engine %q is invalid; allowed: %v", c.Render.Engine, renderProps["engine"].Enum)
	}
	if c.Render.LayoutMode != "" && !slices.Contains(renderProps["layout_mode"].Enum, c.Render.LayoutMode) {
		return fmt.Errorf("config validation: render.layout_mode %q is invalid; allowed: %v", c.Render.LayoutMode, renderProps["layout_mode"].Enum)
	}
	if c.Render.BilingualMode != "" && !slices.Contains(renderProps["bilingual_mode"].Enum, c.Render.BilingualMode) {
		return fmt.Errorf("config validation: render.bilingual_mode %q is invalid; allowed: %v", c.Render.BilingualMode, renderProps["bilingual_mode"].Enum)
	}

	return nil
}

func validateHTTPURL(field, raw string) error {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return fmt.Errorf("config validation: %s=%q must be a valid http:// or https:// URL", field, raw)
	}
	return nil
}

// WorkerEnv builds the environment variable overrides passed to Python worker subprocesses.
// Secrets are resolved from env or file references and never serialized into spec files.
func (c *Config) WorkerEnv() []string {
	var env []string
	if strings.TrimSpace(c.Runtime.Typst) != "" {
		env = append(env, "TYPST_BIN="+c.Runtime.Typst)
	}
	if strings.TrimSpace(c.LLM.BaseURL) != "" {
		env = append(env, "OPENAI_BASE_URL="+strings.TrimSpace(c.LLM.BaseURL))
	}
	if envName := strings.TrimSpace(c.LLM.APIKeyEnv); envName != "" {
		if val := strings.TrimSpace(os.Getenv(envName)); val != "" {
			env = append(env, "OPENAI_API_KEY="+val)
		}
	}
	if strings.TrimSpace(c.LLM.APIKeyFile) != "" {
		env = append(env, "OPENAI_API_KEY_FILE="+c.LLM.APIKeyFile)
	}
	if strings.TrimSpace(c.OCR.PaddleHTTP.BaseURL) != "" {
		env = append(env, "PADDLE_OCR_URL="+strings.TrimSpace(c.OCR.PaddleHTTP.BaseURL))
	}
	if c.OCR.PaddleHTTP.DPI != nil {
		env = append(env, "PADDLE_OCR_DPI="+strconv.FormatFloat(*c.OCR.PaddleHTTP.DPI, 'f', -1, 64))
	}
	if c.OCR.PaddleHTTP.TimeoutS != nil {
		env = append(env, "PADDLE_OCR_TIMEOUT_S="+strconv.FormatFloat(*c.OCR.PaddleHTTP.TimeoutS, 'f', -1, 64))
	}
	if c.OCR.PaddleHTTP.HealthTimeoutS != nil {
		env = append(env, "PADDLE_OCR_HEALTH_TIMEOUT_S="+strconv.FormatFloat(*c.OCR.PaddleHTTP.HealthTimeoutS, 'f', -1, 64))
	}
	if len(c.LLM.FallbackModels) > 0 {
		env = append(env, "OPENAI_FALLBACK_MODELS="+strings.Join(c.LLM.FallbackModels, ","))
	}
	if strings.TrimSpace(c.LLM.ProxyFile) != "" {
		env = append(env, "OPENAI_PROXY_FILE="+strings.TrimSpace(c.LLM.ProxyFile))
	}
	if len(c.LLM.Proxies) > 0 {
		env = append(env, "OPENAI_PROXY_POOL="+strings.Join(c.LLM.Proxies, ","))
	}
	if strings.TrimSpace(c.LLM.PoolFile) != "" {
		env = append(env, "OPENAI_POOL_FILE="+strings.TrimSpace(c.LLM.PoolFile))
	}
	if strings.TrimSpace(c.OCR.PaddleCloud.JobURL) != "" {
		env = append(env, "PADDLE_CLOUD_JOB_URL="+strings.TrimSpace(c.OCR.PaddleCloud.JobURL))
	}
	if strings.TrimSpace(c.OCR.PaddleCloud.Model) != "" {
		env = append(env, "PADDLE_CLOUD_MODEL="+strings.TrimSpace(c.OCR.PaddleCloud.Model))
	}
	if strings.TrimSpace(c.OCR.PaddleCloud.TokenFile) != "" {
		env = append(env, "PADDLE_CLOUD_TOKEN_FILE="+strings.TrimSpace(c.OCR.PaddleCloud.TokenFile))
	}
	if strings.TrimSpace(c.OCR.PaddleCloud.ProxyFile) != "" {
		env = append(env, "PADDLE_CLOUD_PROXY_FILE="+strings.TrimSpace(c.OCR.PaddleCloud.ProxyFile))
	}
	if len(c.OCR.PaddleCloud.Proxies) > 0 {
		env = append(env, "PADDLE_CLOUD_PROXY_POOL="+strings.Join(c.OCR.PaddleCloud.Proxies, ","))
	}
	if strings.TrimSpace(c.OCR.PaddleCloud.FallbackProvider) != "" {
		env = append(env, "PADDLE_CLOUD_FALLBACK="+strings.TrimSpace(c.OCR.PaddleCloud.FallbackProvider))
	}
	if strings.TrimSpace(c.OCR.MinerU.BaseURL) != "" {
		env = append(env, "MINERU_BASE_URL="+strings.TrimSpace(c.OCR.MinerU.BaseURL))
	}
	if strings.TrimSpace(c.OCR.MinerU.ModelVersion) != "" {
		env = append(env, "MINERU_MODEL_VERSION="+strings.TrimSpace(c.OCR.MinerU.ModelVersion))
	}
	if strings.TrimSpace(c.OCR.MinerU.Language) != "" {
		env = append(env, "MINERU_LANGUAGE="+strings.TrimSpace(c.OCR.MinerU.Language))
	}
	if strings.TrimSpace(c.OCR.MinerU.TokenFile) != "" {
		env = append(env, "MINERU_TOKEN_FILE="+strings.TrimSpace(c.OCR.MinerU.TokenFile))
	}
	if strings.TrimSpace(c.OCR.MinerU.ProxyFile) != "" {
		env = append(env, "MINERU_PROXY_FILE="+strings.TrimSpace(c.OCR.MinerU.ProxyFile))
	}
	if len(c.OCR.MinerU.Proxies) > 0 {
		env = append(env, "MINERU_PROXY_POOL="+strings.Join(c.OCR.MinerU.Proxies, ","))
	}
	if strings.TrimSpace(c.OCR.MinerU.FallbackProvider) != "" {
		env = append(env, "MINERU_FALLBACK="+strings.TrimSpace(c.OCR.MinerU.FallbackProvider))
	}
	return env
}

// OCRConfigMap constructs the non-secret provider configuration map forwarded via spec.ocr.config.
func (c *Config) OCRConfigMap() map[string]any {
	out := map[string]any{}
	paddleHTTP := map[string]any{}
	if strings.TrimSpace(c.OCR.PaddleHTTP.BaseURL) != "" {
		paddleHTTP["base_url"] = strings.TrimSpace(c.OCR.PaddleHTTP.BaseURL)
	}
	if c.OCR.PaddleHTTP.DPI != nil {
		paddleHTTP["dpi"] = *c.OCR.PaddleHTTP.DPI
	}
	if c.OCR.PaddleHTTP.TimeoutS != nil {
		paddleHTTP["timeout_s"] = *c.OCR.PaddleHTTP.TimeoutS
	}
	if c.OCR.PaddleHTTP.HealthTimeoutS != nil {
		paddleHTTP["health_timeout_s"] = *c.OCR.PaddleHTTP.HealthTimeoutS
	}
	if len(paddleHTTP) > 0 {
		out["paddle_http"] = paddleHTTP
	}

	paddleCPU := map[string]any{}
	if c.OCR.PaddleCPU.DPI != nil {
		paddleCPU["dpi"] = *c.OCR.PaddleCPU.DPI
	}
	if strings.TrimSpace(c.OCR.PaddleCPU.Lang) != "" {
		paddleCPU["lang"] = strings.TrimSpace(c.OCR.PaddleCPU.Lang)
	}
	if len(paddleCPU) > 0 {
		out["paddle_cpu"] = paddleCPU
	}

	autoCfg := map[string]any{}
	if c.OCR.Auto.ScannedCharThreshold != nil {
		autoCfg["scanned_char_threshold"] = *c.OCR.Auto.ScannedCharThreshold
	}
	if strings.TrimSpace(c.OCR.Auto.FallbackProvider) != "" {
		autoCfg["fallback_provider"] = strings.TrimSpace(c.OCR.Auto.FallbackProvider)
	}
	if len(autoCfg) > 0 {
		out["auto"] = autoCfg
	}
	paddleCloud := map[string]any{}
	if strings.TrimSpace(c.OCR.PaddleCloud.JobURL) != "" {
		paddleCloud["job_url"] = strings.TrimSpace(c.OCR.PaddleCloud.JobURL)
	}
	if strings.TrimSpace(c.OCR.PaddleCloud.Model) != "" {
		paddleCloud["model"] = strings.TrimSpace(c.OCR.PaddleCloud.Model)
	}
	if strings.TrimSpace(c.OCR.PaddleCloud.TokenFile) != "" {
		paddleCloud["token_file"] = strings.TrimSpace(c.OCR.PaddleCloud.TokenFile)
	}
	if len(c.OCR.PaddleCloud.Tokens) > 0 {
		paddleCloud["tokens"] = c.OCR.PaddleCloud.Tokens
	}
	if strings.TrimSpace(c.OCR.PaddleCloud.ProxyFile) != "" {
		paddleCloud["proxy_file"] = strings.TrimSpace(c.OCR.PaddleCloud.ProxyFile)
	}
	if len(c.OCR.PaddleCloud.Proxies) > 0 {
		paddleCloud["proxies"] = c.OCR.PaddleCloud.Proxies
	}
	if strings.TrimSpace(c.OCR.PaddleCloud.FallbackProvider) != "" {
		paddleCloud["fallback_provider"] = strings.TrimSpace(c.OCR.PaddleCloud.FallbackProvider)
	}
	if c.OCR.PaddleCloud.TimeoutS != nil {
		paddleCloud["timeout_s"] = *c.OCR.PaddleCloud.TimeoutS
	}
	if c.OCR.PaddleCloud.TotalTimeoutS != nil {
		paddleCloud["total_timeout_s"] = *c.OCR.PaddleCloud.TotalTimeoutS
	}
	if c.OCR.PaddleCloud.LargePDFMB != nil {
		paddleCloud["large_pdf_mb"] = *c.OCR.PaddleCloud.LargePDFMB
	}
	if c.OCR.PaddleCloud.PollIntervalS != nil {
		paddleCloud["poll_interval_s"] = *c.OCR.PaddleCloud.PollIntervalS
	}
	if len(paddleCloud) > 0 {
		out["paddle_cloud"] = paddleCloud
	}
	mineruCfg := map[string]any{}
	if strings.TrimSpace(c.OCR.MinerU.BaseURL) != "" {
		mineruCfg["base_url"] = strings.TrimSpace(c.OCR.MinerU.BaseURL)
	}
	if strings.TrimSpace(c.OCR.MinerU.ModelVersion) != "" {
		mineruCfg["model_version"] = strings.TrimSpace(c.OCR.MinerU.ModelVersion)
	}
	if strings.TrimSpace(c.OCR.MinerU.Language) != "" {
		mineruCfg["language"] = strings.TrimSpace(c.OCR.MinerU.Language)
	}
	if strings.TrimSpace(c.OCR.MinerU.TokenFile) != "" {
		mineruCfg["token_file"] = strings.TrimSpace(c.OCR.MinerU.TokenFile)
	}
	if len(c.OCR.MinerU.Tokens) > 0 {
		mineruCfg["tokens"] = c.OCR.MinerU.Tokens
	}
	if strings.TrimSpace(c.OCR.MinerU.ProxyFile) != "" {
		mineruCfg["proxy_file"] = strings.TrimSpace(c.OCR.MinerU.ProxyFile)
	}
	if len(c.OCR.MinerU.Proxies) > 0 {
		mineruCfg["proxies"] = c.OCR.MinerU.Proxies
	}
	if strings.TrimSpace(c.OCR.MinerU.FallbackProvider) != "" {
		mineruCfg["fallback_provider"] = strings.TrimSpace(c.OCR.MinerU.FallbackProvider)
	}
	if c.OCR.MinerU.TimeoutS != nil {
		mineruCfg["timeout_s"] = *c.OCR.MinerU.TimeoutS
	}
	if c.OCR.MinerU.TotalTimeoutS != nil {
		mineruCfg["total_timeout_s"] = *c.OCR.MinerU.TotalTimeoutS
	}
	if c.OCR.MinerU.LargePDFMB != nil {
		mineruCfg["large_pdf_mb"] = *c.OCR.MinerU.LargePDFMB
	}
	if c.OCR.MinerU.PollIntervalS != nil {
		mineruCfg["poll_interval_s"] = *c.OCR.MinerU.PollIntervalS
	}
	if len(mineruCfg) > 0 {
		out["mineru"] = mineruCfg
	}

	if len(c.OCR.Options) > 0 {
		out["options"] = c.OCR.Options
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// OCRDefaultsMap returns default task.ocr fields derived from YAML.
func (c *Config) OCRDefaultsMap() map[string]any {
	out := map[string]any{}
	if c.OCR.Provider != "" {
		out["provider"] = c.OCR.Provider
	}
	if cfg := c.OCRConfigMap(); len(cfg) > 0 {
		out["config"] = cfg
	}
	return out
}

// LLMDefaultsMap returns default task.llm fields derived from YAML (without credentials).
func (c *Config) LLMDefaultsMap() map[string]any {
	out := map[string]any{}
	if c.LLM.Provider != "" {
		out["provider"] = c.LLM.Provider
	}
	if c.LLM.Model != "" {
		out["model"] = c.LLM.Model
	}
	if c.LLM.BatchSize != nil {
		out["batch_size"] = *c.LLM.BatchSize
	}
	if c.LLM.Concurrency != nil {
		out["concurrency"] = *c.LLM.Concurrency
	}
	if c.LLM.MaxRetries != nil {
		out["max_retries"] = *c.LLM.MaxRetries
	}
	if c.LLM.TimeoutMS != nil {
		out["timeout_ms"] = *c.LLM.TimeoutMS
	}
	if c.LLM.AllowPartial != nil {
		out["allow_partial"] = *c.LLM.AllowPartial
	}
	return out
}

// RenderDefaultsMap returns default task.render fields derived from YAML.
func (c *Config) RenderDefaultsMap() map[string]any {
	out := map[string]any{}
	if c.Render.Engine != "" {
		out["engine"] = c.Render.Engine
	}
	if c.Render.Font != "" {
		out["font"] = c.Render.Font
	}
	if c.Render.LayoutMode != "" {
		out["layout_mode"] = c.Render.LayoutMode
	}
	if c.Render.BilingualMode != "" {
		out["bilingual_mode"] = c.Render.BilingualMode
	}
	if c.Render.Output != "" {
		out["output"] = c.Render.Output
	}
	if c.Render.OutputCN != "" {
		out["output_cn"] = c.Render.OutputCN
	}
	if c.Render.OutputMerged != "" {
		out["output_merged"] = c.Render.OutputMerged
	}
	if c.Render.Strict != nil {
		out["strict"] = *c.Render.Strict
	}
	return out
}

// RenderEnabledByDefault returns whether POST /v1/jobs enables render when omitted by caller.
func (c *Config) RenderEnabledByDefault() bool {
	return c.Render.Enabled != nil && *c.Render.Enabled
}

// HealthSummary returns a redacted, non-secret summary suitable for GET /health.
func (c *Config) HealthSummary() map[string]any {
	mode := "cli"
	if c.ConfigFile != "" {
		mode = "yaml"
	}
	out := map[string]any{
		"config_mode":    mode,
		"max_workers":    c.EffectiveMaxWorkers(),
		"ocr_provider":   c.OCR.Provider,
		"llm_provider":   c.LLM.Provider,
		"llm_model":      c.LLM.Model,
		"render_enabled": c.RenderEnabledByDefault(),
		"render_engine":  c.Render.Engine,
	}
	if g := c.EffectiveMaxLLMConcurrency(); g > 0 {
		out["max_llm_concurrency"] = g
	}
	return out
}
