package mockllm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/wolf-ai-apps/claude-generated/platform/internal/openai"
)

func fastConfig() Config {
	c := DefaultConfig()
	c.PrefillPerToken = time.Microsecond
	c.BaseITL = time.Millisecond
	c.ITLPerRunning = 0
	return c
}

func TestAdmitQueuesWhenFullAndCancelRemovesWaiter(t *testing.T) {
	c := fastConfig()
	c.MaxRunning = 1
	e := NewEngine(c, NewMetrics("m"))
	if err := e.Admit(context.Background(), 10); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error)
	go func() { done <- e.Admit(ctx, 10) }()
	time.Sleep(10 * time.Millisecond)
	if got := e.Metrics.Waiting.Value(); got != 1 {
		t.Fatalf("waiting = %v want 1", got)
	}
	cancel()
	if err := <-done; err == nil {
		t.Fatal("expected cancellation error")
	}
	if got := e.Metrics.Waiting.Value(); got != 0 {
		t.Fatalf("waiting after cancel = %v want 0", got)
	}
	e.Release(10)
	if e.Running() != 0 {
		t.Fatalf("running = %d want 0", e.Running())
	}
}

func TestReleaseAdmitsFIFO(t *testing.T) {
	c := fastConfig()
	c.MaxRunning = 1
	e := NewEngine(c, NewMetrics("m"))
	e.Admit(context.Background(), 1)
	order := make(chan int, 2)
	for i := 1; i <= 2; i++ {
		go func(i int) {
			e.Admit(context.Background(), 1)
			order <- i
			e.Release(1)
		}(i)
		time.Sleep(5 * time.Millisecond) // enqueue 1 before 2
	}
	e.Release(1)
	if a, b := <-order, <-order; a != 1 || b != 2 {
		t.Fatalf("order = %d,%d want 1,2", a, b)
	}
}

func TestKVBudgetLimitsAdmission(t *testing.T) {
	c := fastConfig()
	c.KVCapacityTokens = 100
	e := NewEngine(c, NewMetrics("m"))
	if err := e.Admit(context.Background(), 101); err != ErrTooLarge {
		t.Fatalf("want ErrTooLarge, got %v", err)
	}
	e.Admit(context.Background(), 80)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := e.Admit(ctx, 30); err == nil {
		t.Fatal("second request should wait for KV space")
	}
}

func TestPrefixCache(t *testing.T) {
	e := NewEngine(fastConfig(), NewMetrics("m"))
	sys := strings.Repeat("You are a helpful support agent. ", 20) // 660 chars
	if got := e.CachedPrefixTokens(sys + "question one"); got != 0 {
		t.Fatalf("cold cache hit %d", got)
	}
	got := e.CachedPrefixTokens(sys + "a different question")
	want := (len(sys) / 64) * 64 / 4
	if got != want {
		t.Fatalf("warm cache hit = %d tokens want %d", got, want)
	}
	if got := e.CachedPrefixTokens("X" + sys); got != 0 {
		t.Fatalf("changing the first character must break the hash chain, got %d", got)
	}
}

func TestStreamingEndpoint(t *testing.T) {
	srv := httptest.NewServer(NewServer(fastConfig()).Handler())
	defer srv.Close()
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":true,"stream_options":{"include_usage":true}}`
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	r := openai.NewSSEReader(resp.Body)
	tokens, sawUsage, sawDone := 0, false, false
	for {
		d, err := r.Next()
		if err == io.EOF {
			break
		}
		if string(d) == "[DONE]" {
			sawDone = true
			continue
		}
		var c openai.ChatChunk
		if err := json.Unmarshal(d, &c); err != nil {
			t.Fatal(err)
		}
		if c.Usage != nil {
			sawUsage = c.Usage.CompletionTokens == 5
		}
		if len(c.Choices) > 0 && c.Choices[0].Delta.Content != "" {
			tokens++
		}
	}
	if tokens != 5 || !sawUsage || !sawDone {
		t.Fatalf("tokens=%d usage=%v done=%v", tokens, sawUsage, sawDone)
	}
}

func TestMetricsEndpoint(t *testing.T) {
	srv := httptest.NewServer(NewServer(fastConfig()).Handler())
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	for _, name := range []string{"vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:time_to_first_token_seconds_bucket"} {
		if !strings.Contains(string(b), name) {
			t.Errorf("metrics missing %s", name)
		}
	}
}
