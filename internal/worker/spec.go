// Package worker owns the Go <-> Python contract.
//
// The important design point: defaults and required fields are NOT declared here
// as Go constants. They are read from contracts/task.spec.schema.json, the same
// file the Python worker reads. This is the fix for the failure mode we saw in
// retain-pdf, where Rust and Python each hardcoded a batch_size fallback and a
// missing key silently produced two different translation behaviours.
package worker

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Spec mirrors the schema's typed fields. It is a typed VIEW over the contract,
// not the authority for defaults/required fields.
type Spec struct {
	JobID      string `json:"job_id"`
	Stage      string `json:"stage"`
	Workdir    string `json:"workdir"`
	Source     string `json:"source"`
	TargetLang string `json:"target_lang"`
	OCR        struct {
		Provider string `json:"provider"`
	} `json:"ocr"`
	LLM struct {
		Provider     string `json:"provider"`
		Concurrency  int    `json:"concurrency"`
		BatchSize    int    `json:"batch_size"`
		MaxRetries   int    `json:"max_retries"`
		TimeoutMS    int    `json:"timeout_ms"`
		Model        string `json:"model"`
		AllowPartial bool   `json:"allow_partial"`
	} `json:"llm"`
	Render *struct {
		Engine        string `json:"engine"`
		Font          string `json:"font"`
		Output        string `json:"output"`
		OutputCN      string `json:"output_cn"`
		OutputMerged  string `json:"output_merged"`
		LayoutMode    string `json:"layout_mode"`
		BilingualMode string `json:"bilingual_mode"`
		Strict        bool   `json:"strict"`
	} `json:"render,omitempty"`
}

type Prop struct {
	Type       string         `json:"type"`
	Enum       []string       `json:"enum"`
	Default    any            `json:"default"`
	Required   []string       `json:"required"`
	Properties map[string]Prop `json:"properties"`
}

type Schema struct {
	Required   []string         `json:"required"`
	Properties map[string]Prop  `json:"properties"`
}

// LoadSchema reads the single source of truth. Path may be overridden with
// MINIRETAIN_SCHEMA; default is relative to the process working directory
// (the project root when started by tools/smoke.sh).
func LoadSchema(path string) (*Schema, error) {
	if path == "" {
		path = os.Getenv("MINIRETAIN_SCHEMA")
	}
	if path == "" {
		path = "contracts/task.spec.schema.json"
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read schema %q: %w", path, err)
	}
	var s Schema
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, fmt.Errorf("parse schema %q: %w", path, err)
	}
	return &s, nil
}

// BuildSpec applies schema defaults to raw, validates required/enum, and
// unmarshals into the typed Spec. Missing required keys are a hard error —
// never a silent fallback on either side of the boundary.
func BuildSpec(raw map[string]any, s *Schema) (Spec, error) {
	var out Spec
	node, err := Normalize(raw, s)
	if err != nil {
		return out, err
	}
	b, err := json.Marshal(node)
	if err != nil {
		return out, err
	}
	if err := json.Unmarshal(b, &out); err != nil {
		return out, fmt.Errorf("decode spec: %w", err)
	}
	return out, nil
}

// Normalize returns a validated copy of raw with schema defaults filled in.
// Both the API layer and the executor call this, so a request that would
// produce a different behaviour on one side of the boundary is rejected early.
func Normalize(raw map[string]any, s *Schema) (map[string]any, error) {
	node := cloneMap(raw)
	root := Prop{Required: s.Required, Properties: s.Properties}
	applyDefaults(node, root)
	if err := validate(node, root, "$"); err != nil {
		return nil, err
	}
	return node, nil
}

func cloneMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if child, ok := v.(map[string]any); ok {
			out[k] = cloneMap(child)
		} else {
			out[k] = v
		}
	}
	return out
}

func applyDefaults(node map[string]any, p Prop) {
	for name, child := range p.Properties {
		if _, ok := node[name]; !ok {
			if child.Default != nil {
				node[name] = child.Default
			}
		}
		if child.Properties != nil {
			if nested, ok := node[name].(map[string]any); ok {
				applyDefaults(nested, child)
			}
		}
	}
}

func validate(node map[string]any, p Prop, path string) error {
	for _, name := range p.Required {
		if _, ok := node[name]; !ok {
			return fmt.Errorf("%s: missing required field %q", path, name)
		}
	}
	for name, child := range p.Properties {
		v, ok := node[name]
		if !ok {
			continue
		}
		if child.Type == "object" || child.Properties != nil {
			nested, ok := v.(map[string]any)
			if !ok {
				return fmt.Errorf("%s.%s: expected object", path, name)
			}
			if err := validate(nested, child, path+"."+name); err != nil {
				return err
			}
		}
		switch child.Type {
		case "string":
			if _, ok := v.(string); !ok {
				return fmt.Errorf("%s.%s: expected string, got %T", path, name, v)
			}
		case "integer":
			n, ok := v.(float64)
			if !ok || n != math.Trunc(n) {
				return fmt.Errorf("%s.%s: expected integer, got %v", path, name, v)
			}
		case "boolean":
			if _, ok := v.(bool); !ok {
				return fmt.Errorf("%s.%s: expected boolean, got %T", path, name, v)
			}
		}
		if len(child.Enum) > 0 {
			s, ok := v.(string)
			if !ok {
				return fmt.Errorf("%s.%s: expected string", path, name)
			}
			found := false
			for _, e := range child.Enum {
				if s == e {
					found = true
					break
				}
			}
			if !found {
				return fmt.Errorf("%s.%s: %q not in %v", path, name, s, child.Enum)
			}
		}
	}
	return nil
}
