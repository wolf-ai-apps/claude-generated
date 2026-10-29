// Package prom is a tiny, dependency-free Prometheus text-format exporter.
// It exists so the academy tools build with the standard library only;
// production code should use github.com/prometheus/client_golang.
package prom

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
)

type metric interface {
	write(w io.Writer)
	name() string
}

// Registry holds metrics and serves them at /metrics.
type Registry struct {
	mu      sync.Mutex
	metrics []metric
}

func NewRegistry() *Registry { return &Registry{} }

func (r *Registry) add(m metric) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.metrics = append(r.metrics, m)
}

// ServeHTTP writes all metrics in Prometheus text exposition format.
func (r *Registry) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	r.mu.Lock()
	ms := append([]metric(nil), r.metrics...)
	r.mu.Unlock()
	sort.Slice(ms, func(i, j int) bool { return ms[i].name() < ms[j].name() })
	for _, m := range ms {
		m.write(w)
	}
}

// Labels are rendered in sorted key order.
type Labels map[string]string

func (l Labels) String() string {
	if len(l) == 0 {
		return ""
	}
	keys := make([]string, 0, len(l))
	for k := range l {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = fmt.Sprintf("%s=%q", k, l[k])
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func withLabel(l Labels, k, v string) Labels {
	out := Labels{k: v}
	for kk, vv := range l {
		out[kk] = vv
	}
	return out
}

// Gauge is a value that goes up and down.
type Gauge struct {
	mu     sync.Mutex
	n, h   string
	labels Labels
	v      float64
}

func (r *Registry) NewGauge(name, help string, labels Labels) *Gauge {
	g := &Gauge{n: name, h: help, labels: labels}
	r.add(g)
	return g
}
func (g *Gauge) Set(v float64) { g.mu.Lock(); g.v = v; g.mu.Unlock() }
func (g *Gauge) Add(v float64) { g.mu.Lock(); g.v += v; g.mu.Unlock() }
func (g *Gauge) Value() float64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.v
}
func (g *Gauge) name() string { return g.n }
func (g *Gauge) write(w io.Writer) {
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n%s%s %g\n", g.n, g.h, g.n, g.n, g.labels, g.Value())
}

// Counter only goes up. Name it with a _total suffix.
type Counter struct{ Gauge }

func (r *Registry) NewCounter(name, help string, labels Labels) *Counter {
	c := &Counter{Gauge{n: name, h: help, labels: labels}}
	r.add(c)
	return c
}
func (c *Counter) Inc() { c.Add(1) }
func (c *Counter) write(w io.Writer) {
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s counter\n%s%s %g\n", c.n, c.h, c.n, c.n, c.labels, c.Value())
}

// Histogram counts observations into cumulative buckets.
type Histogram struct {
	mu     sync.Mutex
	n, h   string
	labels Labels
	bounds []float64
	counts []uint64
	sum    float64
	count  uint64
}

func (r *Registry) NewHistogram(name, help string, labels Labels, bounds []float64) *Histogram {
	h := &Histogram{n: name, h: help, labels: labels, bounds: bounds, counts: make([]uint64, len(bounds))}
	r.add(h)
	return h
}

func (h *Histogram) Observe(v float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for i, b := range h.bounds {
		if v <= b {
			h.counts[i]++
		}
	}
	h.sum += v
	h.count++
}
func (h *Histogram) name() string { return h.n }
func (h *Histogram) write(w io.Writer) {
	h.mu.Lock()
	defer h.mu.Unlock()
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s histogram\n", h.n, h.h, h.n)
	for i, b := range h.bounds {
		fmt.Fprintf(w, "%s_bucket%s %d\n", h.n, withLabel(h.labels, "le", fmtBound(b)), h.counts[i])
	}
	fmt.Fprintf(w, "%s_bucket%s %d\n", h.n, withLabel(h.labels, "le", "+Inf"), h.count)
	fmt.Fprintf(w, "%s_sum%s %g\n%s_count%s %d\n", h.n, h.labels, h.sum, h.n, h.labels, h.count)
}

func fmtBound(b float64) string {
	if math.IsInf(b, 1) {
		return "+Inf"
	}
	return fmt.Sprintf("%g", b)
}

// LatencyBuckets are seconds buckets suitable for TTFT and end-to-end latency.
var LatencyBuckets = []float64{0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10, 20, 40, 80}

// ITLBuckets are seconds buckets suitable for inter-token latency.
var ITLBuckets = []float64{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.25, 0.5, 1}
