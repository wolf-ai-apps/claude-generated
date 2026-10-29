// mockllm is a GPU-free, OpenAI-compatible fake inference server with
// realistic queueing, prefill/decode timing, prefix caching and vLLM-style metrics.
//
//	go run ./cmd/mockllm -addr :8000 -name replica-a
package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/wolf-ai-apps/claude-generated/platform/internal/mockllm"
)

func main() {
	c := mockllm.DefaultConfig()
	addr := flag.String("addr", ":8000", "listen address")
	flag.StringVar(&c.Name, "name", c.Name, "replica name, returned in the X-Backend header")
	flag.StringVar(&c.Model, "model", c.Model, "model name to report")
	flag.IntVar(&c.MaxRunning, "max-num-seqs", c.MaxRunning, "max concurrently running sequences")
	flag.IntVar(&c.KVCapacityTokens, "kv-tokens", c.KVCapacityTokens, "KV cache capacity in tokens")
	flag.DurationVar(&c.PrefillPerToken, "prefill-per-token", c.PrefillPerToken, "prefill time per uncached prompt token")
	flag.DurationVar(&c.BaseITL, "itl", c.BaseITL, "decode step time at batch size 1")
	flag.DurationVar(&c.ITLPerRunning, "itl-per-seq", c.ITLPerRunning, "extra decode step time per extra running sequence")
	flag.IntVar(&c.CacheBlocks, "cache-blocks", c.CacheBlocks, "prefix cache capacity in 64-char blocks (0 disables)")
	flag.Parse()

	s := mockllm.NewServer(c)
	log.Printf("mockllm %s serving %s on %s (max-num-seqs=%d kv-tokens=%d)", c.Name, c.Model, *addr, c.MaxRunning, c.KVCapacityTokens)
	log.Fatal(http.ListenAndServe(*addr, s.Handler()))
}
