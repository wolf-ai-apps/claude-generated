package calc

import (
	"math"
	"testing"
)

func TestKVBytesPerTokenLlama8B(t *testing.T) {
	// 2 x 32 x 8 x 128 x 2 bytes = 131072 bytes = 128 KiB
	got := KVBytesPerToken(Models["llama-3.1-8b"], 2)
	if got != 131072 {
		t.Fatalf("kv bytes per token = %v, want 131072", got)
	}
}

func TestKVBytesPerTokenLlama70B(t *testing.T) {
	// 2 x 80 x 8 x 128 x 2 = 327680 bytes = 320 KiB
	if got := KVBytesPerToken(Models["llama-3.1-70b"], 2); got != 327680 {
		t.Fatalf("got %v", got)
	}
}

func TestEstimate8BOnA10G(t *testing.T) {
	r, err := Estimate(Input{
		Model: Models["llama-3.1-8b"], GPU: GPUs["a10g"], NumGPUs: 1,
		WeightDType: "bf16", KVDType: "bf16", ContextTokens: 4096, PromptTokens: 1024,
		MemUtil: 0.9, OverheadGB: 1, BatchSize: 1, MBU: 1, MFU: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(r.WeightGB-16.06) > 0.01 {
		t.Errorf("weights = %.2f GB, want ~16.06", r.WeightGB)
	}
	if !r.Fits {
		t.Fatal("8B bf16 should fit on a 24GB GPU")
	}
	// 21.6 usable - 16.06 weights - 1 overhead = 4.54 GB; 4096 tokens x 128KiB = 0.537 GB/seq -> 8 seqs
	if r.MaxConcurrentSeqs != 8 {
		t.Errorf("max seqs = %d, want 8", r.MaxConcurrentSeqs)
	}
	// Single-stream decode ceiling is about bandwidth / bytes: 600e9 / (16.06e9 + 0.54e9) ~ 36 tok/s
	if r.DecodeTokPerSecSeq < 34 || r.DecodeTokPerSecSeq > 38 {
		t.Errorf("decode = %.1f tok/s, want ~36", r.DecodeTokPerSecSeq)
	}
}

func TestEstimate70BDoesNotFitOneGPU(t *testing.T) {
	r, err := Estimate(Input{
		Model: Models["llama-3.1-70b"], GPU: GPUs["h100"], NumGPUs: 1,
		WeightDType: "bf16", KVDType: "bf16", ContextTokens: 4096,
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.Fits {
		t.Fatal("70B bf16 (~141 GB) must not fit on one 80GB GPU")
	}
	r4, _ := Estimate(Input{
		Model: Models["llama-3.1-70b"], GPU: GPUs["h100"], NumGPUs: 4,
		WeightDType: "bf16", KVDType: "bf16", ContextTokens: 4096,
	})
	if !r4.Fits || r4.MaxConcurrentSeqs < 50 {
		t.Fatalf("70B on 4xH100 should fit with plenty of KV room, got %+v", r4)
	}
}

func TestBatchingImprovesAggregateThroughput(t *testing.T) {
	base := Input{Model: Models["llama-3.1-8b"], GPU: GPUs["h100"], NumGPUs: 1,
		WeightDType: "bf16", KVDType: "bf16", ContextTokens: 2048}
	one := base
	one.BatchSize = 1
	many := base
	many.BatchSize = 32
	r1, _ := Estimate(one)
	r32, _ := Estimate(many)
	if r32.DecodeTokPerSecAll < 10*r1.DecodeTokPerSecAll {
		t.Fatalf("batch 32 aggregate %.0f should be >10x batch 1 %.0f", r32.DecodeTokPerSecAll, r1.DecodeTokPerSecAll)
	}
	if r32.DecodeTokPerSecSeq >= r1.DecodeTokPerSecSeq {
		t.Fatal("per-user speed should drop as batch grows")
	}
}

func TestCost(t *testing.T) {
	r, _ := Estimate(Input{Model: Models["llama-3.1-8b"], GPU: GPUs["l4"], WeightDType: "fp8",
		KVDType: "fp8", ContextTokens: 2048, PricePerHour: 1.0, BatchSize: 16})
	want := 1.0 / (r.DecodeTokPerSecAll * 3600) * 1e6
	if math.Abs(r.CostPerMillionToken-want) > 1e-9 {
		t.Fatalf("cost %.4f want %.4f", r.CostPerMillionToken, want)
	}
}

func TestUnknownDType(t *testing.T) {
	if _, err := Estimate(Input{Model: Models["llama-3.1-8b"], GPU: GPUs["l4"], WeightDType: "fp7", KVDType: "bf16", ContextTokens: 1}); err == nil {
		t.Fatal("expected error")
	}
}
