# Level 0: Why inference matters

> Training builds the model once. Inference runs it billions of times, so that's where speed, cost and reliability are decided.

**You will:** understand what happens between "send" and the last token, and learn the four numbers that inference engineering optimizes.
**Time:** 1–2 hours · **Hardware:** laptop (Tier 0)

---

## 1. Concepts

### 1.1 Training vs inference

| | Training | Inference |
|---|---|---|
| What happens | Adjust billions of weights using gradients | Use fixed weights to produce outputs |
| How often | Once (weeks, then occasional fine-tunes) | Every user request, forever |
| Optimized for | Total throughput over days | Latency per user **and** cost per token |
| Failure looks like | A job restarts from a checkpoint | Users wait, time out, or leave |
| Who owns it | Research / ML teams | ML engineers **and** platform/SRE teams |

A model trained once might serve 100 million requests a day. Over a model's lifetime, inference compute usually exceeds training compute by a wide margin. So small inference efficiency gains (10% more tokens per GPU) turn into large amounts of money.

### 1.2 What happens when you press "send"

A request to a chat model, step by step:

```
 client ──HTTP──► API gateway ──► router ──► inference engine (on a GPU node)
                  auth, quotas    picks a     1. tokenize: text -> token IDs
                                  replica     2. queue: wait for a batch slot
                                              3. PREFILL: run the whole prompt through the model
                                                 in one pass -> first output token
                                              4. DECODE: loop, one token per forward pass,
                                                 each streamed back immediately
                                              5. stop at EOS / max_tokens / stop string
 client ◄──SSE stream of tokens────────────── 6. detokenize each token to text
```

Two phases, with very different behavior. The rest of the course builds on this split:

- **Prefill** processes all prompt tokens in parallel. It's one big computation and it's *compute-bound*. It sets the **time to first token**.
- **Decode** produces one token per step. Each step has to read all of the model's weights from GPU memory, so it's *memory-bandwidth-bound*. It sets the **speed of the stream**.

### 1.3 The four numbers

| Metric | Definition | What the user feels | Typical target (chat) |
|---|---|---|---|
| **TTFT**, time to first token | request sent → first token received | "Is it thinking?" | p95 < 0.5–1 s |
| **ITL**, inter-token latency (also **TPOT**, time per output token) | gap between consecutive tokens | Reading speed of the stream | p95 < 50–100 ms (≥ 10–20 tok/s) |
| **Throughput** | tokens/s (or requests/s) across all users | (operator) how much one GPU serves | as high as the SLO allows |
| **Cost** | $ per million tokens | (business) unit economics | depends on model/hardware |

The core tension: **batching more users per GPU raises throughput and lowers cost, but each user's tokens arrive more slowly.** Inference engineering means getting the most throughput while still meeting the latency targets. That throughput is called **goodput** (level 9).

### 1.4 Who does what

- **ML/engine engineers** make each forward pass faster: kernels, batching, memory management, quantization (levels 2–8).
- **Platform engineers** make the fleet fast, cheap and reliable: containers, Kubernetes, routing, autoscaling, observability, cost (levels 9–14).
- The best engineers in either role understand both, because every platform decision (routing, autoscaling signal, instance type) depends on how the engine behaves.

---

## 2. Worked example: reading a token timeline

A user asks a question with a 1,000-token prompt (system prompt + chat history) and gets a 200-token answer:

```
t=0 ms      request leaves the client
t=40 ms     arrives at engine (network + gateway)
t=40-120    waits in queue (other requests occupy the batch)
t=120-320   prefill of 1,000 tokens (200 ms)        -> first token emitted
t=330       TTFT as seen by the client = 330 ms
t=330-4330  199 more tokens at 20 ms each (4.0 s)    -> ITL = 20 ms (50 tok/s)
t=4330      done. End-to-end latency = 4.3 s
```

What each slice tells you:
- 40 ms network/gateway: platform overhead. It should be tiny and stable.
- 80 ms queueing: **the system is near capacity.** This is the first place latency grows under load.
- 200 ms prefill: proportional to prompt length. Prefix caching (level 7) can remove most of it for a repeated system prompt.
- 20 ms per decode step: set by memory bandwidth and batch size (level 6).

## 3. Lab: feel TTFT and ITL

No GPU needed. Start the academy's mock inference server. It simulates queueing, prefill and decode timing, and uses the same API as vLLM:

```bash
cd platform
go run ./cmd/mockllm -addr :8000
```

In another terminal:

```bash
python3 levels/00-why-inference/labs/stream_timing.py --url http://localhost:8000 --max-tokens 32
```

Expected output (numbers vary):

```
    42.0 ms  TTFT    42.0 ms  'the '
    57.3 ms  ITL     15.3 ms  'quick '
    ...
TTFT 42.0 ms | tokens 32 | ITL p50 15.3 ms | decode speed 65.3 tok/s | end-to-end 517 ms
```

Experiments:
1. Make the prompt longer (`--prompt "$(python3 -c 'print("word " * 2000)')"`). TTFT grows, ITL doesn't. Why?
2. Restart the mock with `-itl 40ms`. What changes for the user, and what stays the same?
3. If you have [Ollama](https://ollama.com) installed: `ollama run qwen2.5:0.5b` once, then point the script at `--url http://localhost:11434 --model qwen2.5:0.5b`. Now you're timing a real model on your laptop.

## 4. Check yourself

<details><summary>1. Why does a longer prompt increase TTFT but not ITL?</summary>
The whole prompt is processed during prefill, before the first token. Decode steps each process one new token. They do attend to the longer history, but in most regimes the per-step cost is dominated by reading the weights, not by prompt length.</details>

<details><summary>2. A dashboard shows TTFT p95 rising from 300 ms to 3 s while ITL stays flat. What's the most likely cause?</summary>
Queueing: requests wait longer for a slot because the system is near capacity, or a few very long prompts are hogging prefill. Flat ITL means the batch that *is* running isn't slower; requests are just waiting to get in.</details>

<details><summary>3. Why can't you just report "average latency"?</summary>
LLM latency depends on output length (a 500-token answer always takes longer than a 5-token one), and users feel TTFT and ITL separately. Averages also hide the tail. Report TTFT and ITL percentiles, and end-to-end latency normalized by output length.</details>

## 5. Going deeper
- Anyscale / Databricks / BentoML engineering blogs on LLM inference metrics
- Level 9 of this course: defining SLOs and measuring them correctly

**Next → [Level 1: Computer basics a GPU builds on](../01-computer-basics/README.md)**
