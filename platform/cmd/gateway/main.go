// gateway is an OpenAI-compatible LLM gateway with pluggable routing and per-tenant token budgets.
//
//	go run ./cmd/gateway -addr :9000 -policy prefix \
//	  -backends http://localhost:8001,http://localhost:8002 -tenants team-a=200000,team-b=50000
package main

import (
	"context"
	"flag"
	"log"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/wolf-ai-apps/claude-generated/platform/internal/gateway"
)

func main() {
	addr := flag.String("addr", ":9000", "listen address")
	backends := flag.String("backends", "http://localhost:8000", "comma-separated backend base URLs")
	policy := flag.String("policy", "round-robin", "routing policy: round-robin, least-loaded, prefix")
	prefixChars := flag.Int("prefix-chars", 256, "prefix policy: prompt characters used as the routing key")
	tenants := flag.String("tenants", "", "comma-separated apikey=tokens_per_minute; empty disables auth")
	health := flag.Duration("health-interval", 5*time.Second, "backend health check interval")
	dns := flag.String("discover", "", "host:port to resolve for backends instead of -backends, e.g. vllm-headless.inference.svc.cluster.local:8000")
	flag.Parse()

	var bs []*gateway.Backend
	if *dns == "" {
		for _, raw := range strings.Split(*backends, ",") {
			b, err := gateway.NewBackend(strings.TrimSpace(raw))
			if err != nil {
				log.Fatal(err)
			}
			bs = append(bs, b)
		}
	}
	var router gateway.Router
	switch *policy {
	case "round-robin":
		router = &gateway.RoundRobin{}
	case "least-loaded":
		router = gateway.LeastLoaded{}
	case "prefix":
		router = gateway.PrefixAware{PrefixChars: *prefixChars, LoadFactor: 1.5}
	default:
		log.Fatalf("unknown policy %q", *policy)
	}
	ts := map[string]float64{}
	if *tenants != "" {
		for _, kv := range strings.Split(*tenants, ",") {
			k, v, ok := strings.Cut(kv, "=")
			tpm, err := strconv.ParseFloat(v, 64)
			if !ok || err != nil {
				log.Fatalf("bad tenant %q, want key=tokens_per_minute", kv)
			}
			ts[k] = tpm
		}
	}
	g := gateway.New(gateway.Config{Backends: bs, Router: router, Tenants: ts, HealthInterval: *health})
	if *dns != "" {
		host, port, err := net.SplitHostPort(*dns)
		if err != nil {
			log.Fatalf("bad -discover %q: %v", *dns, err)
		}
		go g.DiscoverLoop(context.Background(), host, port, "http", 10*time.Second)
	}
	go g.HealthLoop(context.Background())
	log.Printf("gateway on %s policy=%s backends=%d tenants=%d", *addr, router.Name(), len(bs), len(ts))
	srv := &http.Server{Addr: *addr, Handler: g.Handler(), ReadHeaderTimeout: 10 * time.Second}
	log.Fatal(srv.ListenAndServe())
}
