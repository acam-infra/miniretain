package main

import (
	"flag"
	"log"
	"net/http"
	"path/filepath"
	"time"

	"miniretain/internal/api"
	"miniretain/internal/job"
	"miniretain/internal/worker"
)

func main() {
	addr := flag.String("addr", ":18080", "listen address")
	root := flag.String("root", ".", "project root (contains contracts/ and python/)")
	python := flag.String("python", "python3", "python executable")
	maxWorkers := flag.Int("max-workers", 2, "max concurrent python worker processes")
	flag.Parse()

	rootAbs, err := filepath.Abs(*root)
	if err != nil {
		log.Fatal(err)
	}
	schemaPath := filepath.Join(rootAbs, "contracts", "task.spec.schema.json")
	schema, err := worker.LoadSchema(schemaPath)
	if err != nil {
		log.Fatal(err)
	}

	store := job.NewStore()
	exec := worker.New(*python,
		filepath.Join(rootAbs, "python", "worker.py"),
		schemaPath,
		filepath.Join(rootAbs, "data", "jobs"),
		*maxWorkers,
		store)

	srv := &http.Server{
		Addr:              *addr,
		Handler:           api.New(store, exec, schema),
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Printf("miniretain listening on %s (schema=%s, python=%s, max_workers=%d)",
		*addr, schemaPath, *python, *maxWorkers)
	log.Fatal(srv.ListenAndServe())
}
