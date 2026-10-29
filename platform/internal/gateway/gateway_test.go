package gateway

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/wolf-ai-apps/claude-generated/platform/internal/mockllm"
)

func backends(t *testing.T, n int) []*Backend {
	var bs []*Backend
	for i := 0; i < n; i++ {
		b, err := NewBackend(fmt.Sprintf("http://replica-%d:8000", i))
		if err != nil {
			t.Fatal(err)
		}
		bs = append(bs, b)
	}
	return bs
}

func TestRoundRobinSpreadsEvenly(t *testing.T) {
	bs := backends(t, 3)
	rr := &RoundRobin{}
	counts := map[string]int{}
	for i := 0; i < 300; i++ {
		counts[rr.Order("", bs)[0].Name]++
	}
	for _, b := range bs {
		if counts[b.Name] != 100 {
			t.Fatalf("counts %v", counts)
		}
	}
}

func TestRoundRobinSkipsUnhealthy(t *testing.T) {
	bs := backends(t, 2)
	bs[0].healthy.Store(false)
	for i := 0; i < 4; i++ {
		if got := (&RoundRobin{}).Order("", bs)[0]; got != bs[1] {
			t.Fatal("picked unhealthy backend")
		}
	}
}

func TestLeastLoaded(t *testing.T) {
	bs := backends(t, 3)
	bs[0].inFlight.Store(5)
	bs[1].inFlight.Store(1)
	bs[2].inFlight.Store(3)
	if got := (LeastLoaded{}).Order("", bs)[0]; got != bs[1] {
		t.Fatalf("picked %s", got.Name)
	}
}

func TestPrefixAwareIsStickyAndSpreads(t *testing.T) {
	bs := backends(t, 4)
	p := PrefixAware{PrefixChars: 64}
	sys := strings.Repeat("You are tenant A's support bot. ", 3) // longer than PrefixChars
	first := p.Order(sys+"question 0", bs)[0]
	for i := 0; i < 20; i++ {
		if got := p.Order(fmt.Sprintf("%squestion %d", sys, i), bs)[0]; got != first {
			t.Fatal("same prefix routed to a different replica")
		}
	}
	seen := map[string]bool{}
	for i := 0; i < 200; i++ {
		seen[p.Order(fmt.Sprintf("tenant %d system prompt", i), bs)[0].Name] = true
	}
	if len(seen) != 4 {
		t.Fatalf("distinct prefixes should spread over all replicas, got %d", len(seen))
	}
}

func TestPrefixAwareSpillsWhenOverloaded(t *testing.T) {
	bs := backends(t, 2)
	p := PrefixAware{LoadFactor: 1.25}
	pref := p.Order("hot tenant", bs)[0]
	pref.inFlight.Store(20)
	if got := p.Order("hot tenant", bs)[0]; got == pref {
		t.Fatal("expected spill to the less loaded replica")
	}
}

func TestPrefixAwareMinimalRemapOnRemoval(t *testing.T) {
	bs := backends(t, 4)
	p := PrefixAware{}
	before := map[string]string{}
	for i := 0; i < 400; i++ {
		k := fmt.Sprintf("key-%d", i)
		before[k] = p.Order(k, bs)[0].Name
	}
	removed := bs[3].Name
	moved := 0
	for k, owner := range before {
		now := p.Order(k, bs[:3])[0].Name
		if owner != removed && now != owner {
			moved++
		}
	}
	if moved != 0 {
		t.Fatalf("%d keys not owned by the removed replica were remapped", moved)
	}
}

func TestTokenBucket(t *testing.T) {
	now := time.Unix(0, 0)
	b := NewTokenBucket(600) // 10 tokens/s
	b.now = func() time.Time { return now }
	if ok, _ := b.Take(600); !ok {
		t.Fatal("full bucket should allow 600")
	}
	ok, wait := b.Take(100)
	if ok || wait != 10*time.Second {
		t.Fatalf("ok=%v wait=%v, want false 10s", ok, wait)
	}
	now = now.Add(10 * time.Second)
	if ok, _ := b.Take(100); !ok {
		t.Fatal("should refill after 10s")
	}
	if ok, _ := b.Take(601); ok {
		t.Fatal("request larger than capacity must fail")
	}
}

func startMock(t *testing.T, name string, itl time.Duration) (*mockllm.Server, *httptest.Server) {
	c := mockllm.DefaultConfig()
	c.Name = name
	c.PrefillPerToken = time.Microsecond
	c.BaseITL = itl
	ms := mockllm.NewServer(c)
	s := httptest.NewServer(ms.Handler())
	t.Cleanup(s.Close)
	return ms, s
}

const chatBody = `{"model":"m","messages":[{"role":"user","content":"hello"}],"max_tokens":%d,"stream":true}`

func TestProxyStreamsAndAuth(t *testing.T) {
	_, up := startMock(t, "a", time.Millisecond)
	b, _ := NewBackend(up.URL)
	g := New(Config{Backends: []*Backend{b}, Tenants: map[string]float64{"k1": 10000}})
	gw := httptest.NewServer(g.Handler())
	defer gw.Close()

	req, _ := http.NewRequest("POST", gw.URL+"/v1/chat/completions", strings.NewReader(fmt.Sprintf(chatBody, 3)))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("no key: status %d", resp.StatusCode)
	}

	req, _ = http.NewRequest("POST", gw.URL+"/v1/chat/completions", strings.NewReader(fmt.Sprintf(chatBody, 3)))
	req.Header.Set("Authorization", "Bearer k1")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	out, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(string(out), "[DONE]") || resp.Header.Get("X-Backend") != "a" {
		t.Fatalf("status %d backend %q body %s", resp.StatusCode, resp.Header.Get("X-Backend"), out)
	}

	req, _ = http.NewRequest("POST", gw.URL+"/v1/chat/completions", strings.NewReader(fmt.Sprintf(chatBody, 20000)))
	req.Header.Set("Authorization", "Bearer k1")
	resp, _ = http.DefaultClient.Do(req)
	resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("over budget: status %d", resp.StatusCode)
	}
}

func TestClientCancelFreesUpstreamSlot(t *testing.T) {
	ms, up := startMock(t, "a", 20*time.Millisecond)
	b, _ := NewBackend(up.URL)
	gw := httptest.NewServer(New(Config{Backends: []*Backend{b}}).Handler())
	defer gw.Close()

	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "POST", gw.URL+"/v1/chat/completions", strings.NewReader(fmt.Sprintf(chatBody, 1000)))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 10)
	resp.Body.Read(buf) // stream has started
	if ms.E.Running() != 1 {
		t.Fatalf("running = %d want 1", ms.E.Running())
	}
	cancel()
	resp.Body.Close()
	deadline := time.Now().Add(time.Second)
	for ms.E.Running() != 0 {
		if time.Now().After(deadline) {
			t.Fatal("upstream generation was not cancelled")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestRetryOnDeadBackend(t *testing.T) {
	_, up := startMock(t, "alive", time.Millisecond)
	dead, _ := NewBackend("http://127.0.0.1:1") // nothing listens here
	alive, _ := NewBackend(up.URL)
	g := New(Config{Backends: []*Backend{dead, alive}, Router: &RoundRobin{}})
	gw := httptest.NewServer(g.Handler())
	defer gw.Close()
	for i := 0; i < 4; i++ {
		resp, err := http.Post(gw.URL+"/v1/chat/completions", "application/json", strings.NewReader(fmt.Sprintf(chatBody, 2)))
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("attempt %d: status %d", i, resp.StatusCode)
		}
	}
}

func TestSetBackendsKeepsState(t *testing.T) {
	bs := backends(t, 2)
	g := New(Config{Backends: bs})
	bs[0].inFlight.Store(3)
	again := backends(t, 3) // same first two names plus one new
	g.SetBackends(again)
	cur := g.Backends()
	if len(cur) != 3 || cur[0] != bs[0] || cur[0].InFlight() != 3 {
		t.Fatalf("existing backend state lost: %+v", cur)
	}
}

func TestDiscoverLoopResolvesLocalhost(t *testing.T) {
	g := New(Config{})
	ctx, cancel := context.WithCancel(context.Background())
	go g.DiscoverLoop(ctx, "localhost", "8000", "http", time.Hour)
	deadline := time.Now().Add(2 * time.Second)
	for len(g.Backends()) == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	cancel()
	if len(g.Backends()) == 0 {
		t.Fatal("no backends discovered for localhost")
	}
}
