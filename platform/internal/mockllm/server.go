package mockllm

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/wolf-ai-apps/claude-generated/platform/internal/openai"
	"github.com/wolf-ai-apps/claude-generated/platform/internal/prom"
)

// Metrics mirrors the vLLM metric names that dashboards and autoscalers use,
// so everything you build against the mock works against real vLLM.
type Metrics struct {
	Reg         *prom.Registry
	Running     *prom.Gauge
	Waiting     *prom.Gauge
	KVUsage     *prom.Gauge
	TTFT        *prom.Histogram
	ITL         *prom.Histogram
	E2E         *prom.Histogram
	PromptToks  *prom.Counter
	GenToks     *prom.Counter
	PrefixQuery *prom.Counter
	PrefixHits  *prom.Counter
	Finished    *prom.Counter
	Aborted     *prom.Counter
}

func NewMetrics(model string) *Metrics {
	r := prom.NewRegistry()
	l := prom.Labels{"model_name": model}
	return &Metrics{
		Reg:         r,
		Running:     r.NewGauge("vllm:num_requests_running", "Number of requests in model execution batches.", l),
		Waiting:     r.NewGauge("vllm:num_requests_waiting", "Number of requests waiting to be processed.", l),
		KVUsage:     r.NewGauge("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.", l),
		TTFT:        r.NewHistogram("vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.", l, prom.LatencyBuckets),
		ITL:         r.NewHistogram("vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.", l, prom.ITLBuckets),
		E2E:         r.NewHistogram("vllm:e2e_request_latency_seconds", "Histogram of end to end request latency in seconds.", l, prom.LatencyBuckets),
		PromptToks:  r.NewCounter("vllm:prompt_tokens_total", "Number of prefill tokens processed.", l),
		GenToks:     r.NewCounter("vllm:generation_tokens_total", "Number of generation tokens processed.", l),
		PrefixQuery: r.NewCounter("vllm:prefix_cache_queries_total", "Prefix cache queries, in terms of number of queried tokens.", l),
		PrefixHits:  r.NewCounter("vllm:prefix_cache_hits_total", "Prefix cache hits, in terms of number of cached tokens.", l),
		Finished:    r.NewCounter("vllm:request_success_total", "Count of successfully processed requests.", l),
		Aborted:     r.NewCounter("mockllm_requests_aborted_total", "Requests cancelled by the client before finishing.", l),
	}
}

var words = strings.Fields("the quick brown fox jumps over the lazy dog while a model streams tokens one at a time from a key value cache")

// Server is the HTTP front end of the mock engine.
type Server struct {
	E   *Engine
	seq atomic.Int64
}

func NewServer(cfg Config) *Server {
	return &Server{E: NewEngine(cfg, NewMetrics(cfg.Model))}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.chat)
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.Handle("GET /metrics", s.E.Metrics.Reg)
	return mux
}

func (s *Server) models(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"object":"list","data":[{"id":%q,"object":"model","owned_by":"academy"}]}`, s.E.cfg.Model)
}

func httpError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": msg, "code": code}})
}

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req openai.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		httpError(w, http.StatusBadRequest, "messages must not be empty")
		return
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = 128
	}
	prompt := req.PromptText()
	promptTokens := openai.EstimateTokens(prompt)
	need := promptTokens + req.MaxTokens
	w.Header().Set("X-Backend", s.E.cfg.Name)

	ctx := r.Context()
	m := s.E.Metrics
	if err := s.E.Admit(ctx, need); err != nil {
		if errors.Is(err, ErrTooLarge) {
			httpError(w, http.StatusBadRequest, fmt.Sprintf("prompt+max_tokens (%d) exceeds KV capacity (%d)", need, s.E.cfg.KVCapacityTokens))
			return
		}
		m.Aborted.Inc()
		return
	}
	defer s.E.Release(need)

	cached := s.E.CachedPrefixTokens(prompt)
	if cached > promptTokens {
		cached = promptTokens
	}
	m.PromptToks.Add(float64(promptTokens))
	m.PrefixQuery.Add(float64(promptTokens))
	m.PrefixHits.Add(float64(cached))
	w.Header().Set("X-Cached-Tokens", fmt.Sprint(cached))

	if err := Sleep(ctx, s.E.PrefillTime(promptTokens, cached)); err != nil {
		m.Aborted.Inc()
		return
	}

	id := fmt.Sprintf("chatcmpl-%s-%d", s.E.cfg.Name, s.seq.Add(1))
	created := time.Now().Unix()
	usage := openai.Usage{PromptTokens: promptTokens, CompletionTokens: req.MaxTokens, TotalTokens: promptTokens + req.MaxTokens}

	if !req.Stream {
		var out strings.Builder
		for i := 0; i < req.MaxTokens; i++ {
			if i == 0 {
				m.TTFT.Observe(time.Since(start).Seconds())
			} else {
				st := s.E.StepTime()
				if err := Sleep(ctx, st); err != nil {
					m.Aborted.Inc()
					return
				}
				m.ITL.Observe(st.Seconds())
			}
			out.WriteString(words[i%len(words)] + " ")
			m.GenToks.Inc()
		}
		m.E2E.Observe(time.Since(start).Seconds())
		m.Finished.Inc()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(openai.ChatResponse{
			ID: id, Object: "chat.completion", Created: created, Model: s.E.cfg.Model,
			Choices: []openai.Choice{{Message: openai.Message{Role: "assistant", Content: out.String()}, FinishReason: "length"}},
			Usage:   usage,
		})
		return
	}

	fl, ok := w.(http.Flusher)
	if !ok {
		httpError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	send := func(c openai.ChatChunk) error {
		b, _ := json.Marshal(c)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return err
		}
		fl.Flush()
		return nil
	}
	last := time.Now()
	for i := 0; i < req.MaxTokens; i++ {
		if i == 0 {
			m.TTFT.Observe(time.Since(start).Seconds())
		} else {
			if err := Sleep(ctx, s.E.StepTime()); err != nil {
				m.Aborted.Inc()
				return
			}
			now := time.Now()
			m.ITL.Observe(now.Sub(last).Seconds())
			last = now
		}
		d := openai.Delta{Content: words[i%len(words)] + " "}
		if i == 0 {
			d.Role = "assistant"
		}
		var fr *string
		if i == req.MaxTokens-1 {
			s := "length"
			fr = &s
		}
		if err := send(openai.ChatChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: s.E.cfg.Model,
			Choices: []openai.ChunkChoice{{Delta: d, FinishReason: fr}}}); err != nil {
			m.Aborted.Inc()
			return
		}
		m.GenToks.Inc()
	}
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		send(openai.ChatChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: s.E.cfg.Model,
			Choices: []openai.ChunkChoice{}, Usage: &usage})
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	fl.Flush()
	m.E2E.Observe(time.Since(start).Seconds())
	m.Finished.Inc()
}
