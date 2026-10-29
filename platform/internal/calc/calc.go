// Package calc implements the back-of-envelope math for LLM inference:
// weight memory, KV cache size, concurrency, decode ceilings and cost.
//
// Every number here is an estimate. The point is to predict within ~2x
// before you measure, then explain the gap after you measure.
package calc

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// Model holds the architecture fields that matter for inference sizing.
// They map directly to a Hugging Face config.json.
type Model struct {
	Name             string
	Params           float64 // total parameters
	ActiveParams     float64 // parameters touched per token (== Params for dense models)
	Layers           int     // num_hidden_layers
	Hidden           int     // hidden_size
	Heads            int     // num_attention_heads
	KVHeads          int     // num_key_value_heads (GQA)
	HeadDim          int     // hidden_size / num_attention_heads unless overridden
	MaxContextTokens int     // max_position_embeddings
}

// GPU holds the hardware fields that matter for inference sizing.
type GPU struct {
	Name         string
	MemoryGB     float64 // HBM / GDDR capacity, decimal GB
	BandwidthGBs float64 // peak memory bandwidth, GB/s
	BF16TFLOPs   float64 // dense BF16/FP16 tensor-core peak, TFLOP/s
}

// Bytes per element for common dtypes.
var DTypeBytes = map[string]float64{
	"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5, "fp4": 0.5,
}

// Models is a small preset catalog. Values are from the public config.json files.
var Models = map[string]Model{
	"qwen2.5-0.5b": {Name: "Qwen2.5-0.5B-Instruct", Params: 0.494e9, ActiveParams: 0.494e9,
		Layers: 24, Hidden: 896, Heads: 14, KVHeads: 2, HeadDim: 64, MaxContextTokens: 32768},
	"llama-3.1-8b": {Name: "Llama-3.1-8B-Instruct", Params: 8.03e9, ActiveParams: 8.03e9,
		Layers: 32, Hidden: 4096, Heads: 32, KVHeads: 8, HeadDim: 128, MaxContextTokens: 131072},
	"llama-3.1-70b": {Name: "Llama-3.1-70B-Instruct", Params: 70.6e9, ActiveParams: 70.6e9,
		Layers: 80, Hidden: 8192, Heads: 64, KVHeads: 8, HeadDim: 128, MaxContextTokens: 131072},
	"mixtral-8x7b": {Name: "Mixtral-8x7B-Instruct", Params: 46.7e9, ActiveParams: 12.9e9,
		Layers: 32, Hidden: 4096, Heads: 32, KVHeads: 8, HeadDim: 128, MaxContextTokens: 32768},
}

// GPUs is a preset catalog of approximate datasheet peaks. Always check the
// vendor datasheet for the exact SKU you rent; cloud variants can differ.
var GPUs = map[string]GPU{
	"t4":        {Name: "NVIDIA T4 16GB", MemoryGB: 16, BandwidthGBs: 320, BF16TFLOPs: 65},
	"a10g":      {Name: "NVIDIA A10G 24GB", MemoryGB: 24, BandwidthGBs: 600, BF16TFLOPs: 70},
	"l4":        {Name: "NVIDIA L4 24GB", MemoryGB: 24, BandwidthGBs: 300, BF16TFLOPs: 121},
	"l40s":      {Name: "NVIDIA L40S 48GB", MemoryGB: 48, BandwidthGBs: 864, BF16TFLOPs: 362},
	"a100-40gb": {Name: "NVIDIA A100 40GB", MemoryGB: 40, BandwidthGBs: 1555, BF16TFLOPs: 312},
	"a100-80gb": {Name: "NVIDIA A100 80GB", MemoryGB: 80, BandwidthGBs: 2039, BF16TFLOPs: 312},
	"h100":      {Name: "NVIDIA H100 SXM 80GB", MemoryGB: 80, BandwidthGBs: 3350, BF16TFLOPs: 989},
	"h200":      {Name: "NVIDIA H200 141GB", MemoryGB: 141, BandwidthGBs: 4800, BF16TFLOPs: 989},
}

// Input describes one sizing question.
type Input struct {
	Model         Model
	GPU           GPU
	NumGPUs       int     // tensor-parallel degree
	WeightDType   string  // dtype of weights
	KVDType       string  // dtype of KV cache
	ContextTokens int     // average (prompt + output) tokens held per sequence
	PromptTokens  int     // average prompt length, for prefill time
	MemUtil       float64 // fraction of GPU memory the engine may use (vLLM --gpu-memory-utilization)
	OverheadGB    float64 // activations, CUDA graphs, allocator slack, per GPU
	PricePerHour  float64 // $/hour for all NumGPUs together; 0 = skip cost
	BatchSize     int     // concurrent sequences to evaluate throughput at; 0 = max that fits
	MFU           float64 // achievable fraction of peak FLOPs for prefill (0.3-0.6 typical)
	MBU           float64 // achievable fraction of peak bandwidth for decode (0.6-0.85 typical)
}

// Result holds the derived estimates.
type Result struct {
	WeightGB            float64
	KVBytesPerToken     float64
	KVGBPerSequence     float64
	UsableGB            float64
	KVBudgetGB          float64
	Fits                bool
	MaxConcurrentSeqs   int
	BatchSize           int
	DecodeTokPerSecSeq  float64 // per-sequence decode ceiling at BatchSize
	DecodeTokPerSecAll  float64 // aggregate decode throughput at BatchSize
	DecodeStepMs        float64
	PrefillMs           float64 // TTFT lower bound for one prompt with an idle GPU
	CostPerMillionToken float64 // output tokens, at BatchSize; 0 if no price
}

// KVBytesPerToken = 2 (K and V) x layers x kv_heads x head_dim x bytes.
func KVBytesPerToken(m Model, kvBytes float64) float64 {
	return 2 * float64(m.Layers) * float64(m.KVHeads) * float64(m.HeadDim) * kvBytes
}

// Estimate runs the full sizing.
func Estimate(in Input) (Result, error) {
	wb, ok := DTypeBytes[in.WeightDType]
	if !ok {
		return Result{}, fmt.Errorf("unknown weight dtype %q", in.WeightDType)
	}
	kb, ok := DTypeBytes[in.KVDType]
	if !ok {
		return Result{}, fmt.Errorf("unknown kv dtype %q", in.KVDType)
	}
	if in.NumGPUs < 1 {
		in.NumGPUs = 1
	}
	if in.MemUtil == 0 {
		in.MemUtil = 0.9
	}
	if in.MFU == 0 {
		in.MFU = 0.5
	}
	if in.MBU == 0 {
		in.MBU = 0.75
	}
	if in.ContextTokens <= 0 {
		return Result{}, fmt.Errorf("context tokens must be > 0")
	}

	var r Result
	n := float64(in.NumGPUs)
	r.WeightGB = in.Model.Params * wb / 1e9
	r.KVBytesPerToken = KVBytesPerToken(in.Model, kb)
	r.KVGBPerSequence = r.KVBytesPerToken * float64(in.ContextTokens) / 1e9
	r.UsableGB = in.GPU.MemoryGB * in.MemUtil * n
	r.KVBudgetGB = r.UsableGB - r.WeightGB - in.OverheadGB*n
	r.Fits = r.KVBudgetGB > r.KVGBPerSequence
	if r.Fits {
		r.MaxConcurrentSeqs = int(math.Floor(r.KVBudgetGB / r.KVGBPerSequence))
	}

	r.BatchSize = in.BatchSize
	if r.BatchSize <= 0 || (r.Fits && r.BatchSize > r.MaxConcurrentSeqs) {
		r.BatchSize = r.MaxConcurrentSeqs
	}
	if r.BatchSize < 1 {
		r.BatchSize = 1
	}

	// Decode: every step reads the active weights once plus each sequence's KV cache.
	// Tensor parallelism splits those bytes across GPUs (ignoring all-reduce cost).
	activeWeightBytes := in.Model.ActiveParams * wb
	kvBytesPerStep := float64(r.BatchSize) * r.KVBytesPerToken * float64(in.ContextTokens)
	bw := in.GPU.BandwidthGBs * 1e9 * in.MBU * n
	stepSec := (activeWeightBytes + kvBytesPerStep) / bw
	// Compute floor: 2 FLOPs per active parameter per token.
	computeSec := 2 * in.Model.ActiveParams * float64(r.BatchSize) / (in.GPU.BF16TFLOPs * 1e12 * in.MFU * n)
	stepSec = math.Max(stepSec, computeSec)
	r.DecodeStepMs = stepSec * 1e3
	r.DecodeTokPerSecSeq = 1 / stepSec
	r.DecodeTokPerSecAll = float64(r.BatchSize) / stepSec

	// Prefill: compute bound, 2 x active params x prompt tokens.
	if in.PromptTokens > 0 {
		flops := 2 * in.Model.ActiveParams * float64(in.PromptTokens)
		r.PrefillMs = flops / (in.GPU.BF16TFLOPs * 1e12 * in.MFU * n) * 1e3
	}

	if in.PricePerHour > 0 {
		tokensPerHour := r.DecodeTokPerSecAll * 3600
		r.CostPerMillionToken = in.PricePerHour / tokensPerHour * 1e6
	}
	return r, nil
}

// Report renders a human-readable summary that shows its work.
func Report(in Input, r Result) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Model: %s   GPU: %d x %s\n", in.Model.Name, max(in.NumGPUs, 1), in.GPU.Name)
	fmt.Fprintf(&b, "Weights (%s):        %.1f GB  = %.2fB params x %.1f bytes\n",
		in.WeightDType, r.WeightGB, in.Model.Params/1e9, DTypeBytes[in.WeightDType])
	fmt.Fprintf(&b, "KV cache per token:  %.0f KiB = 2 x %d layers x %d kv_heads x %d head_dim x %.1f bytes (%s)\n",
		r.KVBytesPerToken/1024, in.Model.Layers, in.Model.KVHeads, in.Model.HeadDim, DTypeBytes[in.KVDType], in.KVDType)
	fmt.Fprintf(&b, "KV per sequence:     %.2f GB  at %d tokens of context\n", r.KVGBPerSequence, in.ContextTokens)
	fmt.Fprintf(&b, "Usable memory:       %.1f GB  (%.0f%% of %.0f GB x %d)\n",
		r.UsableGB, in.MemUtil*100, in.GPU.MemoryGB, max(in.NumGPUs, 1))
	fmt.Fprintf(&b, "KV budget:           %.1f GB  = usable - weights - overhead\n", r.KVBudgetGB)
	if !r.Fits {
		fmt.Fprintf(&b, "DOES NOT FIT: not enough memory for even one sequence. Quantize, add GPUs, or shorten context.\n")
		return b.String()
	}
	fmt.Fprintf(&b, "Max concurrent seqs: %d\n", r.MaxConcurrentSeqs)
	fmt.Fprintf(&b, "At batch %d:\n", r.BatchSize)
	fmt.Fprintf(&b, "  decode step        %.1f ms\n", r.DecodeStepMs)
	fmt.Fprintf(&b, "  per-user speed     %.1f tok/s\n", r.DecodeTokPerSecSeq)
	fmt.Fprintf(&b, "  aggregate          %.0f tok/s\n", r.DecodeTokPerSecAll)
	if r.PrefillMs > 0 {
		fmt.Fprintf(&b, "Prefill (%d tok):   %.0f ms  (TTFT floor on an idle GPU)\n", in.PromptTokens, r.PrefillMs)
	}
	if r.CostPerMillionToken > 0 {
		fmt.Fprintf(&b, "Cost:                $%.3f per 1M output tokens at $%.2f/h\n", r.CostPerMillionToken, in.PricePerHour)
	}
	return b.String()
}

// Keys returns sorted preset names for help text.
func Keys[T any](m map[string]T) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}
