// Package gateway is an OpenAI-compatible LLM gateway: it authenticates
// tenants, enforces token budgets, routes requests to inference replicas,
// streams responses through, and propagates client cancellation upstream.
package gateway

import (
	"fmt"
	"hash/fnv"
	"net/url"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Backend is one inference replica (a vLLM pod, a SageMaker endpoint, a mockllm process).
type Backend struct {
	Name     string
	URL      *url.URL
	inFlight atomic.Int64
	healthy  atomic.Bool
}

func NewBackend(raw string) (*Backend, error) {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("invalid backend URL %q", raw)
	}
	b := &Backend{Name: u.Host, URL: u}
	b.healthy.Store(true)
	return b, nil
}

func (b *Backend) InFlight() int64 { return b.inFlight.Load() }
func (b *Backend) Healthy() bool   { return b.healthy.Load() }

// Router picks a backend for a request. key is the routing key (prompt prefix).
// It returns backends in preference order so the caller can retry on the next one.
type Router interface {
	Name() string
	Order(key string, bs []*Backend) []*Backend
}

func healthy(bs []*Backend) []*Backend {
	out := make([]*Backend, 0, len(bs))
	for _, b := range bs {
		if b.Healthy() {
			out = append(out, b)
		}
	}
	return out
}

// RoundRobin ignores load and content. It is the baseline that LLM traffic breaks:
// one 32k-token prompt and one 10-token prompt count as "one request" each.
type RoundRobin struct{ n atomic.Uint64 }

func (r *RoundRobin) Name() string { return "round-robin" }
func (r *RoundRobin) Order(_ string, bs []*Backend) []*Backend {
	h := healthy(bs)
	if len(h) == 0 {
		return nil
	}
	start := int(r.n.Add(1)-1) % len(h)
	return append(h[start:], h[:start]...)
}

// LeastLoaded sends to the replica with the fewest in-flight requests at this gateway.
// Production routers use engine-reported load (queue depth, KV cache usage) instead.
type LeastLoaded struct{}

func (LeastLoaded) Name() string { return "least-loaded" }
func (LeastLoaded) Order(_ string, bs []*Backend) []*Backend {
	h := healthy(bs)
	sort.SliceStable(h, func(i, j int) bool { return h[i].InFlight() < h[j].InFlight() })
	return h
}

// PrefixAware sends requests that share a prompt prefix to the same replica,
// so that replica's prefix cache already holds the KV blocks (lower TTFT, less GPU work).
// It uses rendezvous (highest-random-weight) hashing, so adding or removing a
// replica only remaps the keys that replica owned, plus a bounded-load check:
// if the preferred replica is far busier than average, spill to the next one.
type PrefixAware struct {
	PrefixChars int     // how much of the prompt forms the key
	LoadFactor  float64 // spill when inFlight > LoadFactor * mean + 1
}

func (p PrefixAware) Name() string { return "prefix-aware" }

func (p PrefixAware) Key(prompt string) string {
	n := p.PrefixChars
	if n <= 0 {
		n = 256
	}
	if len(prompt) > n {
		return prompt[:n]
	}
	return prompt
}

func score(key, backend string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(key))
	h.Write([]byte{0})
	h.Write([]byte(backend))
	return h.Sum64()
}

func (p PrefixAware) Order(key string, bs []*Backend) []*Backend {
	h := healthy(bs)
	if len(h) == 0 {
		return nil
	}
	k := p.Key(key)
	sort.SliceStable(h, func(i, j int) bool { return score(k, h[i].Name) > score(k, h[j].Name) })
	lf := p.LoadFactor
	if lf <= 0 {
		lf = 1.5
	}
	var total int64
	for _, b := range h {
		total += b.InFlight()
	}
	limit := lf*float64(total)/float64(len(h)) + 1
	// Stable partition: replicas under the load limit first, preserving hash order.
	var under, over []*Backend
	for _, b := range h {
		if float64(b.InFlight()) <= limit {
			under = append(under, b)
		} else {
			over = append(over, b)
		}
	}
	return append(under, over...)
}

// TokenBucket enforces a tokens-per-minute budget per tenant.
// LLM rate limits are in tokens, not requests: a request's cost is
// prompt tokens + max_tokens, charged up front.
type TokenBucket struct {
	mu       sync.Mutex
	capacity float64
	tokens   float64
	perSec   float64
	last     time.Time
	now      func() time.Time
}

func NewTokenBucket(tokensPerMinute float64) *TokenBucket {
	return &TokenBucket{capacity: tokensPerMinute, tokens: tokensPerMinute, perSec: tokensPerMinute / 60, now: time.Now}
}

// Take tries to spend n tokens. On failure it returns how long until n tokens are available.
func (b *TokenBucket) Take(n float64) (bool, time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := b.now()
	if !b.last.IsZero() {
		b.tokens = min(b.capacity, b.tokens+now.Sub(b.last).Seconds()*b.perSec)
	}
	b.last = now
	if n > b.capacity {
		return false, 0 // can never succeed
	}
	if b.tokens >= n {
		b.tokens -= n
		return true, 0
	}
	wait := (n - b.tokens) / b.perSec
	return false, time.Duration(wait * float64(time.Second))
}
