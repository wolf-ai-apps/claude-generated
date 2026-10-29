// Package mockllm simulates an LLM inference server well enough to teach
// serving, routing, autoscaling and benchmarking on a laptop with no GPU.
//
// It models the behaviours that matter to the platform layer:
//   - a bounded number of running sequences (vLLM --max-num-seqs)
//   - a KV cache token budget that requests reserve while running
//   - FCFS queueing when either limit is hit (vllm:num_requests_waiting)
//   - prefill time proportional to uncached prompt tokens (TTFT)
//   - decode time per token that grows with batch size (ITL)
//   - block-hash prefix caching, so routing affinity changes TTFT
//   - cancellation: a disconnected client frees its slot immediately
package mockllm

import (
	"container/list"
	"context"
	"errors"
	"hash/fnv"
	"sync"
	"time"
)

type Config struct {
	Name             string
	Model            string
	MaxRunning       int           // max concurrent sequences
	KVCapacityTokens int           // total KV cache tokens across all running sequences
	PrefillPerToken  time.Duration // prefill cost per uncached prompt token
	BaseITL          time.Duration // decode step time with one running sequence
	ITLPerRunning    time.Duration // extra decode step time per additional running sequence
	BlockChars       int           // prefix cache block size in characters (~4 chars/token)
	CacheBlocks      int           // prefix cache capacity in blocks (LRU)
}

func DefaultConfig() Config {
	return Config{
		Name: "mock-0", Model: "mock-llama-3.1-8b",
		MaxRunning: 32, KVCapacityTokens: 64 * 1024,
		PrefillPerToken: 200 * time.Microsecond,
		BaseITL:         15 * time.Millisecond, ITLPerRunning: 500 * time.Microsecond,
		BlockChars: 64, CacheBlocks: 4096,
	}
}

var ErrTooLarge = errors.New("request needs more KV cache than the server has")

type waiter struct {
	need     int
	ch       chan struct{}
	admitted bool
}

// Engine is the simulated scheduler and KV cache.
type Engine struct {
	cfg     Config
	Metrics *Metrics

	mu      sync.Mutex
	running int
	kvUsed  int
	queue   []*waiter

	cacheMu  sync.Mutex
	cacheLRU *list.List
	cacheIdx map[uint64]*list.Element
}

func NewEngine(cfg Config, m *Metrics) *Engine {
	return &Engine{cfg: cfg, Metrics: m, cacheLRU: list.New(), cacheIdx: map[uint64]*list.Element{}}
}

func (e *Engine) Config() Config { return e.cfg }

// Running returns the current number of running sequences.
func (e *Engine) Running() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.running
}

func (e *Engine) fits(need int) bool {
	return e.running < e.cfg.MaxRunning && e.kvUsed+need <= e.cfg.KVCapacityTokens
}

func (e *Engine) publish() {
	e.Metrics.Running.Set(float64(e.running))
	e.Metrics.Waiting.Set(float64(len(e.queue)))
	e.Metrics.KVUsage.Set(float64(e.kvUsed) / float64(e.cfg.KVCapacityTokens))
}

// Admit blocks until the request is scheduled or ctx is cancelled.
// need is the KV tokens to reserve (prompt + max output).
func (e *Engine) Admit(ctx context.Context, need int) error {
	if need > e.cfg.KVCapacityTokens {
		return ErrTooLarge
	}
	e.mu.Lock()
	if len(e.queue) == 0 && e.fits(need) {
		e.running++
		e.kvUsed += need
		e.publish()
		e.mu.Unlock()
		return nil
	}
	w := &waiter{need: need, ch: make(chan struct{})}
	e.queue = append(e.queue, w)
	e.publish()
	e.mu.Unlock()

	select {
	case <-w.ch:
		return nil
	case <-ctx.Done():
		e.mu.Lock()
		if w.admitted {
			e.mu.Unlock()
			e.Release(need)
			return ctx.Err()
		}
		for i, q := range e.queue {
			if q == w {
				e.queue = append(e.queue[:i], e.queue[i+1:]...)
				break
			}
		}
		e.publish()
		e.mu.Unlock()
		return ctx.Err()
	}
}

// Release frees a running slot and its KV reservation, then admits waiters in FCFS order.
func (e *Engine) Release(need int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.running--
	e.kvUsed -= need
	for len(e.queue) > 0 && e.fits(e.queue[0].need) {
		w := e.queue[0]
		e.queue = e.queue[1:]
		e.running++
		e.kvUsed += w.need
		w.admitted = true
		close(w.ch)
	}
	e.publish()
}

// CachedPrefixTokens looks up the prompt in the block-hash prefix cache and
// inserts its blocks. Like vLLM, each block's hash chains the previous
// block's hash, so a block only matches if everything before it matches too.
func (e *Engine) CachedPrefixTokens(prompt string) int {
	bs := e.cfg.BlockChars
	if bs <= 0 || e.cfg.CacheBlocks <= 0 {
		return 0
	}
	e.cacheMu.Lock()
	defer e.cacheMu.Unlock()
	var prev uint64
	matched, stillMatching := 0, true
	for i := 0; i+bs <= len(prompt); i += bs {
		h := fnv.New64a()
		var pb [8]byte
		for k := 0; k < 8; k++ {
			pb[k] = byte(prev >> (8 * k))
		}
		h.Write(pb[:])
		h.Write([]byte(prompt[i : i+bs]))
		prev = h.Sum64()
		if el, ok := e.cacheIdx[prev]; ok {
			e.cacheLRU.MoveToFront(el)
			if stillMatching {
				matched++
			}
			continue
		}
		stillMatching = false
		e.cacheIdx[prev] = e.cacheLRU.PushFront(prev)
		if e.cacheLRU.Len() > e.cfg.CacheBlocks {
			old := e.cacheLRU.Back()
			e.cacheLRU.Remove(old)
			delete(e.cacheIdx, old.Value.(uint64))
		}
	}
	return matched * bs / 4
}

// PrefillTime is the simulated prefill duration for the uncached part of the prompt.
func (e *Engine) PrefillTime(promptTokens, cachedTokens int) time.Duration {
	n := promptTokens - cachedTokens
	if n < 1 {
		n = 1 // the last token always needs computing to produce logits
	}
	return time.Duration(n) * e.cfg.PrefillPerToken
}

// StepTime is the simulated decode step time at the current batch size.
func (e *Engine) StepTime() time.Duration {
	r := e.Running()
	if r < 1 {
		r = 1
	}
	return e.cfg.BaseITL + time.Duration(r-1)*e.cfg.ITLPerRunning
}

// Sleep waits for d or until ctx is done.
func Sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
