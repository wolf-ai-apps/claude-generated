"""Lab 7a: static vs continuous batching vs chunked prefill, simulated.

    python3 batching_sim.py

A step-level simulator of an inference engine. Each engine step costs

    step_ms = BASE + PREFILL_PER_TOKEN * prefill_tokens + DECODE_PER_SEQ * decoding_seqs

BASE models reading all weights once per step (decode is memory-bound), so
adding sequences to a step is cheap: the core reason batching works.
"""

import random
from dataclasses import dataclass, field

BASE_MS = 8.0               # read the weights once (e.g. 16 GB at ~2 TB/s)
PREFILL_PER_TOKEN_MS = 0.02
DECODE_PER_SEQ_MS = 0.15


def step_ms(prefill_tokens: int, decoding: int) -> float:
    return BASE_MS + PREFILL_PER_TOKEN_MS * prefill_tokens + DECODE_PER_SEQ_MS * decoding


@dataclass
class Req:
    rid: int
    arrival: float
    prompt: int
    output: int
    prefilled: int = 0
    token_times: list = field(default_factory=list)

    @property
    def done(self):
        return len(self.token_times) >= self.output


def workload(n=200, rate_per_s=20.0, seed=0):
    rng = random.Random(seed)
    t, out = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate_per_s) * 1000
        # a mix of short chats and occasional long documents
        prompt = rng.choice([64, 128, 256, 512]) if rng.random() < 0.9 else rng.choice([4096, 8192])
        out.append(Req(i, t, prompt, rng.randint(32, 256)))
    return out


def run_static(reqs, batch_size=16):
    """Collect a batch, run it until the LONGEST request finishes, then start the next."""
    now, i, q = 0.0, 0, sorted(reqs, key=lambda r: r.arrival)
    while i < len(q):
        now = max(now, q[i].arrival)
        batch = [r for r in q[i:i + batch_size] if r.arrival <= now] or [q[i]]
        i += len(batch)
        now += step_ms(sum(r.prompt for r in batch), 0)          # prefill everyone together
        for r in batch:
            r.token_times.append(now)
        while not all(r.done for r in batch):                     # finished seqs still hold their slot
            now += step_ms(0, len(batch))
            for r in batch:
                if not r.done:
                    r.token_times.append(now)
    return reqs


def run_continuous(reqs, max_seqs=64, chunk_tokens=None):
    """Iteration-level scheduling: admit and retire requests at every step.

    chunk_tokens=None: a newly admitted prompt is prefilled entirely in one step.
    chunk_tokens=N:    each step processes at most N tokens (decodes count 1 each);
                       long prompts are prefilled in chunks across several steps.
    """
    pending = sorted(reqs, key=lambda r: r.arrival)
    running: list[Req] = []
    now = 0.0
    while pending or running:
        if not running and pending and pending[0].arrival > now:
            now = pending[0].arrival
        while pending and pending[0].arrival <= now and len(running) < max_seqs:
            running.append(pending.pop(0))

        decoding = [r for r in running if r.prefilled >= r.prompt]
        prefilling = [r for r in running if r.prefilled < r.prompt]
        budget = float("inf") if chunk_tokens is None else max(chunk_tokens - len(decoding), 0)
        work = []
        for r in prefilling:
            if budget <= 0:
                break
            n = int(min(r.prompt - r.prefilled, budget))
            work.append((r, n))
            budget -= n

        now += step_ms(sum(n for _, n in work), len(decoding))
        for r in decoding:
            r.token_times.append(now)
        for r, n in work:
            r.prefilled += n
            if r.prefilled >= r.prompt:
                r.token_times.append(now)                        # prefill emits the first token
        running = [r for r in running if not r.done]
    return reqs


def report(name, reqs):
    ttft = sorted(r.token_times[0] - r.arrival for r in reqs)
    itl = sorted(b - a for r in reqs for a, b in zip(r.token_times, r.token_times[1:]))
    total_tokens = sum(r.output for r in reqs)
    span = max(r.token_times[-1] for r in reqs) - min(r.arrival for r in reqs)
    p = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]
    print(f"{name:<34}{total_tokens / span * 1000:>8.0f} tok/s"
          f"{p(ttft, .5):>9.0f}{p(ttft, .99):>9.0f} ms TTFT p50/p99"
          f"{p(itl, .5):>7.1f}{p(itl, .99):>7.1f} ms ITL p50/p99")
    return {"tok_s": total_tokens / span * 1000, "ttft_p50": p(ttft, .5), "itl_p99": p(itl, .99)}


def main():
    print(f"200 requests, Poisson 20 req/s, 10% long prompts (4-8k tokens)\n")
    report("static batching (16)", run_static(workload()))
    report("continuous batching", run_continuous(workload()))
    report("continuous + chunked prefill 512", run_continuous(workload(), chunk_tokens=512))
    print("\nStatic: requests wait for whole batches and for the slowest member -> huge TTFT."
          "\nContinuous: the GPU always has work and requests join immediately."
          "\nChunked prefill: an 8k-token prompt no longer freezes everyone's next token -> lower ITL p99.")


if __name__ == "__main__":
    main()
