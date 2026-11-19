package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"miniretain/internal/api"
	"miniretain/internal/config"
	"miniretain/internal/job"
	"miniretain/internal/worker"
)

func main() {
	configPath := flag.String("config", "", "path to single-YAML declarative config file (or MINIRETAIN_CONFIG)")
	addr := flag.String("addr", config.DefaultAddr, "listen address (overrides YAML server.addr when explicitly set)")
	root := flag.String("root", ".", "project root containing contracts/ and python/ (overrides YAML paths.root when explicitly set)")
	python := flag.String("python", config.DefaultPython, "python executable (overrides YAML runtime.python when explicitly set)")
	maxWorkers := flag.Int("max-workers", config.DefaultMaxWorkers, "max concurrent python worker processes (overrides YAML runtime.max_workers when explicitly set)")
	flag.Parse()

	explicit := map[string]bool{}
	flag.Visit(func(f *flag.Flag) {
		explicit[f.Name] = true
	})

	cfgFile := *configPath
	if cfgFile == "" {
		cfgFile = os.Getenv("MINIRETAIN_CONFIG")
	}

	var (
		cfg *config.Config
		err error
	)
	if cfgFile != "" {
		cfg, err = config.Load(cfgFile)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		cfg = config.Default()
		if err := cfg.ResolveLegacy(); err != nil {
			log.Fatal(err)
		}
	}

	var overrides config.CLIOverrides
	if explicit["addr"] {
		overrides.Addr = addr
	}
	if explicit["root"] {
		overrides.Root = root
	}
	if explicit["python"] {
		overrides.Python = python
	}
	if explicit["max-workers"] {
		overrides.MaxWorkers = maxWorkers
	}
	if err := cfg.ApplyCLIOverrides(overrides); err != nil {
		log.Fatal(err)
	}

	schema, err := worker.LoadSchema(cfg.Paths.Schema)
	if err != nil {
		log.Fatal(err)
	}
	if err := cfg.Validate(schema); err != nil {
		log.Fatal(err)
	}

	store := job.NewStore()
	exec := worker.New(
		cfg.Runtime.Python,
		filepath.Join(cfg.Paths.Root, "python", "worker.py"),
		cfg.Paths.Schema,
		cfg.Paths.DataDir,
		cfg.EffectiveMaxWorkers(),
		store,
	)
	exec.ExtraEnv = cfg.WorkerEnv()
	exec.MaxLLMConcurrency = cfg.EffectiveMaxLLMConcurrency()

	handler := api.NewWithOptions(store, exec, schema, api.Options{
		Defaults: api.JobDefaults{
			TargetLang:    cfg.TargetLang,
			OCR:           cfg.OCRDefaultsMap(),
			LLM:           cfg.LLMDefaultsMap(),
			Render:        cfg.RenderDefaultsMap(),
			RenderEnabled: cfg.RenderEnabledByDefault(),
		},
		HealthInfo: cfg.HealthSummary(),
	})

	srv := &http.Server{
		Addr:              cfg.Server.Addr,
		Handler:           handler,
		ReadHeaderTimeout: time.Duration(cfg.EffectiveReadHeaderTimeoutMS()) * time.Millisecond,
	}

	modeDesc := "cli"
	if cfg.ConfigFile != "" {
		modeDesc = fmt.Sprintf("yaml(%s)", cfg.ConfigFile)
	}
	log.Printf("miniretain listening on %s (mode=%s, root=%s, data_dir=%s, python=%s, max_workers=%d, ocr=%s, llm=%s)",
		cfg.Server.Addr, modeDesc, cfg.Paths.Root, cfg.Paths.DataDir,
		cfg.Runtime.Python, cfg.EffectiveMaxWorkers(),
		emptyAsDefault(cfg.OCR.Provider, "schema-default"),
		emptyAsDefault(cfg.LLM.Provider, "schema-default"),
	)
	log.Fatal(srv.ListenAndServe())
}

func emptyAsDefault(v, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}
