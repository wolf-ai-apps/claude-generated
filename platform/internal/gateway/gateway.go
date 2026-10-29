package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/wolf-ai-apps/claude-generated/platform/internal/openai"
	"github.com/wolf-ai-apps/claude-generated/platform/internal/prom"
)

type Config struct {
	Backends       []*Backend
	Router         Router
	Tenants        map[string]float64 // API key -> tokens per minute; empty = no auth
	MaxBodyBytes   int64
	HealthInterval time.Duration
}

type backendMetrics struct {
	requests *prom.Counter
	errors   *prom.Counter
	inflight *prom.Gauge
}

type Gateway struct {
	cfg     Config
	buckets map[string]*TokenBucket
	client  *http.Client
	reg     *prom.Registry
	limited *prom.Counter
	ttfb    *prom.Histogram

	mu       sync.RWMutex
	backends []*Backend
	bm       map[string]backendMetrics // by backend name; Prometheus series are never deleted
}

func New(cfg Config) *Gateway {
	if cfg.MaxBodyBytes == 0 {
		cfg.MaxBodyBytes = 10 << 20
	}
	if cfg.Router == nil {
		cfg.Router = &RoundRobin{}
	}
	g := &Gateway{
		cfg: cfg, buckets: map[string]*TokenBucket{}, reg: prom.NewRegistry(), bm: map[string]backendMetrics{},
		// No overall timeout: generations can stream for minutes. Cancellation comes from the client's context.
		client: &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 256, ResponseHeaderTimeout: 2 * time.Minute}},
	}
	for k, tpm := range cfg.Tenants {
		g.buckets[k] = NewTokenBucket(tpm)
	}
	g.SetBackends(cfg.Backends)
	g.limited = g.reg.NewCounter("gateway_rate_limited_total", "Requests rejected by token budget.", nil)
	g.ttfb = g.reg.NewHistogram("gateway_time_to_first_byte_seconds", "Time from request to first upstream byte.", nil, prom.LatencyBuckets)
	return g
}

// Backends returns the current backend set.
func (g *Gateway) Backends() []*Backend {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.backends
}

// SetBackends replaces the backend set. Backends with a name already known keep
// their state (in-flight count, health), so routing stays stable across refreshes.
func (g *Gateway) SetBackends(bs []*Backend) {
	g.mu.Lock()
	defer g.mu.Unlock()
	old := map[string]*Backend{}
	for _, b := range g.backends {
		old[b.Name] = b
	}
	next := make([]*Backend, 0, len(bs))
	for _, b := range bs {
		if o, ok := old[b.Name]; ok {
			b = o
		}
		next = append(next, b)
		if _, ok := g.bm[b.Name]; !ok {
			l := prom.Labels{"backend": b.Name}
			g.bm[b.Name] = backendMetrics{
				requests: g.reg.NewCounter("gateway_backend_requests_total", "Requests sent to a backend.", l),
				errors:   g.reg.NewCounter("gateway_backend_errors_total", "Failed requests to a backend.", l),
				inflight: g.reg.NewGauge("gateway_backend_inflight", "In-flight requests per backend.", l),
			}
		}
	}
	g.backends = next
}

func (g *Gateway) metricsFor(b *Backend) backendMetrics {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.bm[b.Name]
}

// DiscoverLoop resolves a DNS name (e.g. a Kubernetes headless Service, which
// returns one A record per ready pod) and updates the backend set until ctx is done.
func (g *Gateway) DiscoverLoop(ctx context.Context, host, port, scheme string, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		ips, err := net.DefaultResolver.LookupHost(ctx, host)
		if err != nil {
			log.Printf("discovery: %v", err)
		} else {
			sort.Strings(ips)
			bs := make([]*Backend, 0, len(ips))
			for _, ip := range ips {
				if b, err := NewBackend(scheme + "://" + net.JoinHostPort(ip, port)); err == nil {
					bs = append(bs, b)
				}
			}
			if len(bs) > 0 {
				if len(bs) != len(g.Backends()) {
					log.Printf("discovery: %s -> %d backends", host, len(bs))
				}
				g.SetBackends(bs)
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func (g *Gateway) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", g.chat)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.Handle("GET /metrics", g.reg)
	return mux
}

// HealthLoop probes each backend's /health until ctx is done.
func (g *Gateway) HealthLoop(ctx context.Context) {
	iv := g.cfg.HealthInterval
	if iv == 0 {
		iv = 5 * time.Second
	}
	t := time.NewTicker(iv)
	defer t.Stop()
	for {
		for _, b := range g.Backends() {
			c, cancel := context.WithTimeout(ctx, 2*time.Second)
			req, _ := http.NewRequestWithContext(c, http.MethodGet, b.URL.JoinPath("/health").String(), nil)
			resp, err := g.client.Do(req)
			ok := err == nil && resp.StatusCode == http.StatusOK
			if resp != nil {
				resp.Body.Close()
			}
			cancel()
			if b.Healthy() != ok {
				log.Printf("backend %s healthy=%v", b.Name, ok)
			}
			b.healthy.Store(ok)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func jsonError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": msg, "code": code}})
}

func (g *Gateway) chat(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, g.cfg.MaxBodyBytes))
	if err != nil {
		jsonError(w, http.StatusRequestEntityTooLarge, "body too large")
		return
	}
	var req openai.ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		jsonError(w, http.StatusBadRequest, "invalid JSON")
		return
	}
	prompt := req.PromptText()

	if len(g.buckets) > 0 {
		key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		b, ok := g.buckets[key]
		if !ok {
			jsonError(w, http.StatusUnauthorized, "invalid API key")
			return
		}
		maxTok := req.MaxTokens
		if maxTok <= 0 {
			maxTok = 128
		}
		cost := float64(openai.EstimateTokens(prompt) + maxTok)
		if ok, wait := b.Take(cost); !ok {
			g.limited.Inc()
			if wait > 0 {
				w.Header().Set("Retry-After", fmt.Sprint(int(math.Ceil(wait.Seconds()))))
			}
			jsonError(w, http.StatusTooManyRequests, fmt.Sprintf("token budget exceeded (request costs %.0f tokens)", cost))
			return
		}
	}

	order := g.cfg.Router.Order(prompt, g.Backends())
	if len(order) == 0 {
		jsonError(w, http.StatusServiceUnavailable, "no healthy backends")
		return
	}
	// Retry on a different replica only if the failure happened before any byte
	// reached the client. Once streaming starts, a retry would duplicate output.
	attempts := min(2, len(order))
	var lastErr error
	for _, b := range order[:attempts] {
		wrote, err := g.forward(w, r, b, body, start)
		if err == nil || wrote {
			return
		}
		lastErr = err
		if errors.Is(r.Context().Err(), context.Canceled) {
			return
		}
	}
	jsonError(w, http.StatusBadGateway, "upstream error: "+lastErr.Error())
}

// forward proxies one attempt. It reports whether any response bytes were written.
func (g *Gateway) forward(w http.ResponseWriter, r *http.Request, b *Backend, body []byte, start time.Time) (bool, error) {
	m := g.metricsFor(b)
	m.requests.Inc()
	b.inFlight.Add(1)
	m.inflight.Add(1)
	defer func() { b.inFlight.Add(-1); m.inflight.Add(-1) }()

	// The upstream request carries the client's context: if the client
	// disconnects, the upstream connection closes and the engine aborts the
	// generation, freeing its batch slot and KV cache blocks.
	up, err := http.NewRequestWithContext(r.Context(), http.MethodPost, b.URL.JoinPath("/v1/chat/completions").String(), bytes.NewReader(body))
	if err != nil {
		return false, err
	}
	up.Header.Set("Content-Type", "application/json")
	resp, err := g.client.Do(up)
	if err != nil {
		m.errors.Inc()
		return false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 500 {
		m.errors.Inc()
		io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		return false, fmt.Errorf("%s returned %d", b.Name, resp.StatusCode)
	}
	for _, h := range []string{"Content-Type", "Cache-Control", "X-Backend", "X-Cached-Tokens"} {
		if v := resp.Header.Get(h); v != "" {
			w.Header().Set(h, v)
		}
	}
	if w.Header().Get("X-Backend") == "" {
		w.Header().Set("X-Backend", b.Name)
	}
	w.Header().Set("X-Accel-Buffering", "no") // tell nginx-style proxies not to buffer the stream
	w.WriteHeader(resp.StatusCode)
	fl, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	first := true
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if first {
				g.ttfb.Observe(time.Since(start).Seconds())
				first = false
			}
			if _, werr := w.Write(buf[:n]); werr != nil {
				return true, werr
			}
			if fl != nil {
				fl.Flush() // flush every chunk: buffering here would destroy TTFT and ITL
			}
		}
		if rerr == io.EOF {
			return true, nil
		}
		if rerr != nil {
			return true, rerr
		}
	}
}
