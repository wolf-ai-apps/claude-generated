// infercalc prints back-of-envelope inference estimates.
//
//	go run ./cmd/infercalc -model llama-3.1-8b -gpu a10g -context 4096 -prompt 1024 -price 1.0
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/wolf-ai-apps/claude-generated/platform/internal/calc"
)

func main() {
	model := flag.String("model", "llama-3.1-8b", "model preset: "+strings.Join(calc.Keys(calc.Models), ", "))
	gpu := flag.String("gpu", "a10g", "GPU preset: "+strings.Join(calc.Keys(calc.GPUs), ", "))
	gpus := flag.Int("gpus", 1, "number of GPUs (tensor parallel degree)")
	wdt := flag.String("dtype", "bf16", "weight dtype: bf16, fp16, fp8, int8, int4")
	kvdt := flag.String("kv-dtype", "bf16", "KV cache dtype: bf16, fp8")
	ctx := flag.Int("context", 4096, "average tokens (prompt + output) held per sequence")
	prompt := flag.Int("prompt", 1024, "average prompt tokens (for prefill estimate)")
	memUtil := flag.Float64("mem-util", 0.9, "fraction of GPU memory the engine may use")
	overhead := flag.Float64("overhead-gb", 1.0, "per-GPU overhead for activations, CUDA graphs, etc.")
	price := flag.Float64("price", 0, "$/hour for all GPUs together (0 = skip cost)")
	batch := flag.Int("batch", 0, "concurrent sequences to evaluate (0 = max that fits)")
	mbu := flag.Float64("mbu", 0.75, "achievable memory bandwidth utilization")
	mfu := flag.Float64("mfu", 0.5, "achievable FLOPs utilization")
	flag.Parse()

	m, ok := calc.Models[*model]
	if !ok {
		fail("unknown model %q", *model)
	}
	g, ok := calc.GPUs[*gpu]
	if !ok {
		fail("unknown gpu %q", *gpu)
	}
	in := calc.Input{
		Model: m, GPU: g, NumGPUs: *gpus, WeightDType: *wdt, KVDType: *kvdt,
		ContextTokens: *ctx, PromptTokens: *prompt, MemUtil: *memUtil, OverheadGB: *overhead,
		PricePerHour: *price, BatchSize: *batch, MBU: *mbu, MFU: *mfu,
	}
	r, err := calc.Estimate(in)
	if err != nil {
		fail("%v", err)
	}
	fmt.Print(calc.Report(in, r))
}

func fail(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "infercalc: "+format+"\n", a...)
	os.Exit(2)
}
