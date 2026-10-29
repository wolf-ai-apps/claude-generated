package loadgen

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/wolf-ai-apps/claude-generated/platform/internal/mockllm"
)

func TestPercentileNearestRank(t *testing.T) {
	v := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	if got := Percentile(v, 50); got != 5 {
		t.Errorf("p50 = %v", got)
	}
	if got := Percentile(v, 95); got != 10 {
		t.Errorf("p95 = %v", got)
	}
	if got := Percentile(v, 0); got != 1 {
		t.Errorf("p0 = %v", got)
	}
}

func TestArrivalsMeanRate(t *testing.T) {
	a := Arrivals(rand.New(rand.NewSource(1)), 20000, 10)
	got := float64(len(a)) / a[len(a)-1].Seconds()
	if math.Abs(got-10) > 0.3 {
		t.Fatalf("observed rate %.2f want ~10", got)
	}
}

func TestRunAgainstMock(t *testing.T) {
	c := mockllm.DefaultConfig()
	c.Name = "a"
	c.PrefillPerToken = 10 * time.Microsecond
	c.BaseITL = 2 * time.Millisecond
	srv := httptest.NewServer(mockllm.NewServer(c).Handler())
	defer srv.Close()

	for _, cfg := range []Config{
		{URL: srv.URL, Model: "m", Rate: 200, Requests: 20, PromptMin: 50, PromptMax: 100, OutputMin: 5, OutputMax: 10, Seed: 1},
		{URL: srv.URL, Model: "m", Concurrency: 4, Requests: 20, PromptMin: 50, PromptMax: 100, OutputMin: 5, OutputMax: 10, Seed: 1},
	} {
		rs, wall := Run(context.Background(), cfg)
		s := Summarize(cfg, rs, wall)
		if s.Errors != 0 {
			t.Fatalf("errors: %+v", rs[0])
		}
		if s.TTFT.P50 <= 0 || s.ITL.P50 < 1.5 {
			t.Fatalf("implausible timings: %+v", s)
		}
		if s.Backends["a"] != 20 {
			t.Fatalf("backend counts %v", s.Backends)
		}
	}
}

func TestSharedPrefixHitsCache(t *testing.T) {
	c := mockllm.DefaultConfig()
	c.PrefillPerToken = 50 * time.Microsecond
	c.BaseITL = time.Millisecond
	ms := mockllm.NewServer(c)
	srv := httptest.NewServer(ms.Handler())
	defer srv.Close()
	cfg := Config{URL: srv.URL, Model: "m", Concurrency: 1, Requests: 10, PromptMin: 20, PromptMax: 20,
		OutputMin: 2, OutputMax: 2, PrefixTokens: 1000, PrefixGroups: 1, Seed: 2}
	Run(context.Background(), cfg)
	hits := ms.E.Metrics.PrefixHits.Value()
	if hits < 8000 { // 9 warm requests x ~1000 cached tokens
		t.Fatalf("prefix cache hits = %v tokens, expected > 8000", hits)
	}
}

func TestTruncatedStreamIsAnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 3; i++ {
			fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x \"}}]}\n\n")
		}
		// connection closes without data: [DONE]
	}))
	defer srv.Close()
	r := Do(context.Background(), http.DefaultClient, Config{URL: srv.URL}, []byte(`{}`))
	if !strings.Contains(r.Err, "truncated") {
		t.Fatalf("want truncation error, got %+v", r)
	}
}
