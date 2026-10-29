// Package loadgen is a streaming-aware load generator for OpenAI-compatible
// inference servers. It measures what users feel (TTFT, ITL, end-to-end)
// and what operators pay for (throughput, goodput at an SLO).
//
// Open-loop mode schedules arrivals on a Poisson clock that does not wait for
// responses, which avoids coordinated omission: a slow server cannot slow the
// arrival rate down and hide its own latency.
package loadgen

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/wolf-ai-apps/claude-generated/platform/internal/openai"
)

type Config struct {
	URL          string  // base URL, e.g. http://localhost:8000
	Model        string  // model name sent in the request
	APIKey       string  // optional bearer token
	Rate         float64 // open loop: mean requests/second (Poisson). 0 = closed loop
	Concurrency  int     // closed loop: number of concurrent virtual users
	Requests     int     // total requests to send
	PromptMin    int     // prompt tokens, uniform in [PromptMin, PromptMax]
	PromptMax    int
	OutputMin    int // output tokens, uniform in [OutputMin, OutputMax]
	OutputMax    int
	PrefixTokens int // shared system-prompt tokens per prefix group (0 = none)
	PrefixGroups int // number of distinct shared prefixes (e.g. tenants or conversations)
	SLOTTFT      time.Duration
	SLOITL       time.Duration // applied to each request's mean inter-token latency (TPOT)
	Timeout      time.Duration
	Seed         int64
}

// Result is one request's measurements.
type Result struct {
	Start        time.Duration // offset from test start
	TTFT         time.Duration
	E2E          time.Duration
	ITLs         []time.Duration
	OutputTokens int
	PromptTokens int
	Backend      string
	Err          string
}

// TPOT is mean time per output token after the first.
func (r Result) TPOT() time.Duration {
	if r.OutputTokens < 2 {
		return 0
	}
	return (r.E2E - r.TTFT) / time.Duration(r.OutputTokens-1)
}

type Percentiles struct {
	P50, P90, P95, P99, Mean, Max float64 // milliseconds
}

type Summary struct {
	Requests      int
	Errors        int
	Duration      time.Duration
	RequestRate   float64 // completed req/s
	OutputTokRate float64 // output tok/s
	TTFT          Percentiles
	TPOT          Percentiles
	ITL           Percentiles
	E2E           Percentiles
	SLOMet        int
	Goodput       float64 // req/s meeting SLO
	Backends      map[string]int
}

// Percentile uses the nearest-rank method on a sorted copy.
func Percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := int(math.Ceil(p/100*float64(len(sorted)))) - 1
	if rank < 0 {
		rank = 0
	}
	if rank >= len(sorted) {
		rank = len(sorted) - 1
	}
	return sorted[rank]
}

func percentiles(ds []time.Duration) Percentiles {
	if len(ds) == 0 {
		return Percentiles{}
	}
	v := make([]float64, len(ds))
	sum := 0.0
	for i, d := range ds {
		v[i] = float64(d) / float64(time.Millisecond)
		sum += v[i]
	}
	sort.Float64s(v)
	return Percentiles{P50: Percentile(v, 50), P90: Percentile(v, 90), P95: Percentile(v, 95),
		P99: Percentile(v, 99), Mean: sum / float64(len(v)), Max: v[len(v)-1]}
}

// Summarize aggregates results.
func Summarize(cfg Config, rs []Result, wall time.Duration) Summary {
	s := Summary{Requests: len(rs), Duration: wall, Backends: map[string]int{}}
	var ttft, tpot, itl, e2e []time.Duration
	outTok := 0
	for _, r := range rs {
		if r.Err != "" {
			s.Errors++
			continue
		}
		s.Backends[r.Backend]++
		ttft = append(ttft, r.TTFT)
		e2e = append(e2e, r.E2E)
		if t := r.TPOT(); t > 0 {
			tpot = append(tpot, t)
		}
		itl = append(itl, r.ITLs...)
		outTok += r.OutputTokens
		if (cfg.SLOTTFT == 0 || r.TTFT <= cfg.SLOTTFT) && (cfg.SLOITL == 0 || r.TPOT() <= cfg.SLOITL) {
			s.SLOMet++
		}
	}
	sec := wall.Seconds()
	if sec > 0 {
		s.RequestRate = float64(len(rs)-s.Errors) / sec
		s.OutputTokRate = float64(outTok) / sec
		s.Goodput = float64(s.SLOMet) / sec
	}
	s.TTFT, s.TPOT, s.ITL, s.E2E = percentiles(ttft), percentiles(tpot), percentiles(itl), percentiles(e2e)
	return s
}

// Arrivals returns n Poisson arrival offsets with the given mean rate.
func Arrivals(rng *rand.Rand, n int, rate float64) []time.Duration {
	out := make([]time.Duration, n)
	t := 0.0
	for i := range out {
		t += rng.ExpFloat64() / rate
		out[i] = time.Duration(t * float64(time.Second))
	}
	return out
}

// text returns a string of roughly n tokens (4 chars per token) of random letters,
// so distinct requests never share an accidental prefix.
func text(rng *rand.Rand, n int) string {
	var b strings.Builder
	b.Grow(n * 4)
	for i := 0; i < n; i++ {
		for j := 0; j < 3; j++ {
			b.WriteByte(byte('a' + rng.Intn(26)))
		}
		b.WriteByte(' ')
	}
	return b.String()
}

func uniform(rng *rand.Rand, lo, hi int) int {
	if hi <= lo {
		return lo
	}
	return lo + rng.Intn(hi-lo+1)
}

type job struct {
	at   time.Duration
	body []byte
	pt   int
}

func buildJobs(cfg Config) []job {
	rng := rand.New(rand.NewSource(cfg.Seed))
	prefixes := make([]string, max(cfg.PrefixGroups, 1))
	for i := range prefixes {
		if cfg.PrefixTokens > 0 {
			prefixes[i] = text(rng, cfg.PrefixTokens)
		}
	}
	var arrivals []time.Duration
	if cfg.Rate > 0 {
		arrivals = Arrivals(rng, cfg.Requests, cfg.Rate)
	}
	jobs := make([]job, cfg.Requests)
	for i := range jobs {
		pt := uniform(rng, cfg.PromptMin, cfg.PromptMax)
		ot := uniform(rng, cfg.OutputMin, cfg.OutputMax)
		msgs := []openai.Message{}
		if cfg.PrefixTokens > 0 {
			msgs = append(msgs, openai.Message{Role: "system", Content: prefixes[rng.Intn(len(prefixes))]})
		}
		msgs = append(msgs, openai.Message{Role: "user", Content: text(rng, pt)})
		req := openai.ChatRequest{Model: cfg.Model, Messages: msgs, MaxTokens: ot, Stream: true,
			StreamOptions: &openai.StreamOptions{IncludeUsage: true}, IgnoreEOS: true}
		b, _ := json.Marshal(req)
		jobs[i] = job{body: b, pt: pt + cfg.PrefixTokens}
		if arrivals != nil {
			jobs[i].at = arrivals[i]
		}
	}
	return jobs
}

// Do sends one streaming request and records timings.
func Do(ctx context.Context, client *http.Client, cfg Config, body []byte) Result {
	var r Result
	start := time.Now()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(cfg.URL, "/")+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		r.Err = err.Error()
		return r
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	defer resp.Body.Close()
	r.Backend = resp.Header.Get("X-Backend")
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		r.Err = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
		return r
	}
	sse := openai.NewSSEReader(resp.Body)
	last := time.Time{}
	done := false
	for {
		data, err := sse.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			r.Err = err.Error()
			return r
		}
		if string(data) == "[DONE]" {
			done = true
			break
		}
		var c openai.ChatChunk
		if err := json.Unmarshal(data, &c); err != nil {
			r.Err = "bad chunk: " + err.Error()
			return r
		}
		if c.Usage != nil {
			r.PromptTokens = c.Usage.PromptTokens
		}
		if len(c.Choices) == 0 || c.Choices[0].Delta.Content == "" {
			continue
		}
		now := time.Now()
		if last.IsZero() {
			r.TTFT = now.Sub(start)
		} else {
			r.ITLs = append(r.ITLs, now.Sub(last))
		}
		last = now
		r.OutputTokens++
	}
	r.E2E = time.Since(start)
	switch {
	case r.OutputTokens == 0:
		r.Err = "no tokens received"
	case !done:
		// The connection closed before the server said it was finished: the user saw a
		// truncated answer. Count it as an error, never as a fast success.
		r.Err = fmt.Sprintf("stream truncated after %d tokens (no [DONE])", r.OutputTokens)
	}
	return r
}

// Run executes the load test and returns per-request results and wall time.
func Run(ctx context.Context, cfg Config) ([]Result, time.Duration) {
	if cfg.Timeout == 0 {
		cfg.Timeout = 5 * time.Minute
	}
	client := &http.Client{Timeout: cfg.Timeout, Transport: &http.Transport{MaxIdleConnsPerHost: 1024}}
	jobs := buildJobs(cfg)
	results := make([]Result, len(jobs))
	var wg sync.WaitGroup
	t0 := time.Now()

	if cfg.Rate > 0 { // open loop
		for i, j := range jobs {
			if d := time.Until(t0.Add(j.at)); d > 0 {
				select {
				case <-time.After(d):
				case <-ctx.Done():
					wg.Wait()
					return results[:i], time.Since(t0)
				}
			}
			wg.Add(1)
			go func(i int, j job) {
				defer wg.Done()
				r := Do(ctx, client, cfg, j.body)
				r.Start = time.Since(t0) - r.E2E
				r.PromptTokens = max(r.PromptTokens, j.pt)
				results[i] = r
			}(i, j)
		}
	} else { // closed loop
		next := make(chan int)
		for w := 0; w < max(cfg.Concurrency, 1); w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range next {
					s := time.Since(t0)
					r := Do(ctx, client, cfg, jobs[i].body)
					r.Start = s
					results[i] = r
				}
			}()
		}
		for i := range jobs {
			select {
			case next <- i:
			case <-ctx.Done():
			}
		}
		close(next)
	}
	wg.Wait()
	return results, time.Since(t0)
}

// Format renders a summary table.
func Format(cfg Config, s Summary) string {
	var b strings.Builder
	mode := fmt.Sprintf("open loop, Poisson %.2f req/s", cfg.Rate)
	if cfg.Rate == 0 {
		mode = fmt.Sprintf("closed loop, %d concurrent users", cfg.Concurrency)
	}
	fmt.Fprintf(&b, "Mode: %s\n", mode)
	fmt.Fprintf(&b, "Requests: %d  errors: %d  wall: %.1fs\n", s.Requests, s.Errors, s.Duration.Seconds())
	fmt.Fprintf(&b, "Throughput: %.2f req/s  %.0f output tok/s\n", s.RequestRate, s.OutputTokRate)
	fmt.Fprintf(&b, "%-6s %9s %9s %9s %9s %9s\n", "ms", "p50", "p90", "p95", "p99", "max")
	for _, row := range []struct {
		n string
		p Percentiles
	}{{"TTFT", s.TTFT}, {"TPOT", s.TPOT}, {"ITL", s.ITL}, {"E2E", s.E2E}} {
		fmt.Fprintf(&b, "%-6s %9.1f %9.1f %9.1f %9.1f %9.1f\n", row.n, row.p.P50, row.p.P90, row.p.P95, row.p.P99, row.p.Max)
	}
	if cfg.SLOTTFT > 0 || cfg.SLOITL > 0 {
		pct := 0.0
		if ok := s.Requests - s.Errors; ok > 0 {
			pct = 100 * float64(s.SLOMet) / float64(s.Requests)
		}
		fmt.Fprintf(&b, "SLO (TTFT<=%v, TPOT<=%v): %d/%d met (%.1f%%)  goodput %.2f req/s\n",
			cfg.SLOTTFT, cfg.SLOITL, s.SLOMet, s.Requests, pct, s.Goodput)
	}
	if len(s.Backends) > 1 || (len(s.Backends) == 1 && !hasKey(s.Backends, "")) {
		keys := make([]string, 0, len(s.Backends))
		for k := range s.Backends {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		fmt.Fprintf(&b, "Backends:")
		for _, k := range keys {
			fmt.Fprintf(&b, " %s=%d", k, s.Backends[k])
		}
		fmt.Fprintln(&b)
	}
	return b.String()
}

func hasKey(m map[string]int, k string) bool { _, ok := m[k]; return ok }
