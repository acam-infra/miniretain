// Package job is the in-memory task registry for the minimal service.
// A production build would swap this for SQLite; the API surface stays the same.
package job

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

type Status string

const (
	Queued    Status = "queued"
	Running   Status = "running"
	Succeeded Status = "succeeded"
	Failed    Status = "failed"
)

type Job struct {
	ID        string         `json:"id"`
	Status    Status         `json:"status"`
	Stage     string         `json:"stage"`
	Params    map[string]any `json:"-"` // normalized base spec, not exposed raw
	Result    map[string]any `json:"result,omitempty"`
	Error     string         `json:"error,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
}

type Store struct {
	mu    sync.RWMutex
	jobs  map[string]*Job
	order []string
}

func NewStore() *Store {
	return &Store{jobs: map[string]*Job{}}
}

func (s *Store) Create(j *Job) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.jobs[j.ID]; exists {
		return fmt.Errorf("job %q already exists", j.ID)
	}
	s.jobs[j.ID] = j
	s.order = append(s.order, j.ID)
	return nil
}

func (s *Store) Get(id string) (*Job, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	j, ok := s.jobs[id]
	if !ok {
		return nil, false
	}
	c := *j
	c.Result = copyMap(j.Result)
	return &c, true
}

func (s *Store) List() []*Job {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*Job, 0, len(s.order))
	for _, id := range s.order {
		c := *s.jobs[id]
		c.Result = copyMap(s.jobs[id].Result)
		out = append(out, &c)
	}
	sort.Slice(out, func(i, k int) bool { return out[i].CreatedAt.After(out[k].CreatedAt) })
	return out
}

func copyMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// Update applies a mutation while holding the write lock. Handlers must not
// keep the returned pointer after this call.
func (s *Store) Update(id string, fn func(*Job)) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[id]
	if !ok {
		return false
	}
	fn(j)
	j.UpdatedAt = time.Now()
	return true
}
