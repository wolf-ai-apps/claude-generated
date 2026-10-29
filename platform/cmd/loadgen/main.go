// loadgen benchmarks an OpenAI-compatible server (vLLM, SGLang, the gateway, or mockllm).
//
//	go run ./cmd/loadgen -url http://localhost:8000 -rate 5 -n 200 -slo-ttft 500ms -slo-tpot 50ms
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"time"

	"github.com/wolf-ai-apps/claude-generated/platform/internal/loadgen"
)

func main() {
	var c loadgen.Config
	flag.StringVar(&c.URL, "url", "http://localhost:8000", "server base URL")
	flag.StringVar(&c.Model, "model", "mock-llama-3.1-8b", "model name")
	flag.StringVar(&c.APIKey, "api-key", os.Getenv("OPENAI_API_KEY"), "bearer token")
	flag.Float64Var(&c.Rate, "rate", 0, "open-loop Poisson request rate (req/s); 0 = closed loop")
	flag.IntVar(&c.Concurrency, "concurrency", 8, "closed-loop concurrent users")
	flag.IntVar(&c.Requests, "n", 100, "number of requests")
	flag.IntVar(&c.PromptMin, "prompt-min", 256, "min prompt tokens")
	flag.IntVar(&c.PromptMax, "prompt-max", 1024, "max prompt tokens")
	flag.IntVar(&c.OutputMin, "output-min", 64, "min output tokens")
	flag.IntVar(&c.OutputMax, "output-max", 256, "max output tokens")
	flag.IntVar(&c.PrefixTokens, "prefix-tokens", 0, "shared system prompt tokens")
	flag.IntVar(&c.PrefixGroups, "prefix-groups", 1, "number of distinct shared prefixes")
	flag.DurationVar(&c.SLOTTFT, "slo-ttft", 0, "TTFT SLO, e.g. 500ms")
	flag.DurationVar(&c.SLOITL, "slo-tpot", 0, "per-request mean inter-token latency SLO, e.g. 50ms")
	flag.DurationVar(&c.Timeout, "timeout", 5*time.Minute, "per-request timeout")
	flag.Int64Var(&c.Seed, "seed", 42, "random seed (same seed = same workload)")
	out := flag.String("out", "", "write per-request results and summary as JSON to this file")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	rs, wall := loadgen.Run(ctx, c)
	s := loadgen.Summarize(c, rs, wall)
	fmt.Print(loadgen.Format(c, s))
	for _, r := range rs {
		if r.Err != "" {
			fmt.Fprintf(os.Stderr, "first error: %s\n", r.Err)
			break
		}
	}
	if *out != "" {
		b, _ := json.MarshalIndent(map[string]any{"config": c, "summary": s, "results": rs}, "", "  ")
		if err := os.WriteFile(*out, b, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}
