"""Lab 5b: turning logits into the next token.

    python3 sampling.py

The model outputs one score (logit) per vocabulary entry. A sampler turns
those scores into a choice. Every OpenAI-style parameter maps to a step here.
"""

import numpy as np


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def apply_repetition_penalty(logits, previous, penalty):
    """CTRL-style penalty (as in HF/vLLM): shrink positive logits, grow negative ones."""
    out = logits.copy()
    for t in set(previous):
        out[t] = out[t] / penalty if out[t] > 0 else out[t] * penalty
    return out


def filter_probs(probs, top_k=0, top_p=1.0, min_p=0.0):
    p = probs.copy()
    if top_k > 0:
        kth = np.sort(p)[-top_k]
        p[p < kth] = 0
    if top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        # keep the smallest set whose mass reaches top_p (always keep the top token)
        cutoff = np.searchsorted(cum, top_p * cum[-1]) + 1
        p[order[cutoff:]] = 0
    if min_p > 0:
        p[p < min_p * p.max()] = 0
    return p / p.sum()


def sample(logits, rng, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
           previous=(), repetition_penalty=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    if repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, previous, repetition_penalty)
    if temperature == 0:
        return int(np.argmax(logits))                 # greedy
    probs = filter_probs(softmax(logits / temperature), top_k, top_p, min_p)
    return int(rng.choice(len(probs), p=probs))


VOCAB = ["Paris", "Lyon", "France", "the", "a", "beautiful", "banana", "42", ".", "!"]
LOGITS = np.array([6.0, 3.5, 3.0, 2.0, 1.5, 1.2, -1.0, -2.0, 0.5, 0.0])


def histogram(rng, n=2000, **kw):
    counts = np.zeros(len(VOCAB), int)
    for _ in range(n):
        counts[sample(LOGITS, rng, **kw)] += 1
    return {VOCAB[i]: f"{c / n:.0%}" for i, c in enumerate(counts) if c}


def main():
    rng = np.random.default_rng(0)
    print('Prompt: "The capital of France is" -> next-token distribution\n')
    print("raw probabilities:", {v: f"{p:.1%}" for v, p in zip(VOCAB, softmax(LOGITS)) if p > 0.005})
    for name, kw in [("greedy (temperature=0)", dict(temperature=0)),
                     ("temperature=0.5", dict(temperature=0.5)),
                     ("temperature=1.0", dict(temperature=1.0)),
                     ("temperature=1.5", dict(temperature=1.5)),
                     ("temperature=1.5, top_k=3", dict(temperature=1.5, top_k=3)),
                     ("temperature=1.5, top_p=0.9", dict(temperature=1.5, top_p=0.9)),
                     ("temperature=1.5, min_p=0.1", dict(temperature=1.5, min_p=0.1))]:
        print(f"{name:<28} {histogram(rng, **kw)}")
    print("\nHigher temperature flattens the distribution; top-k/top-p/min-p cut off the long tail"
          " so 'banana' never appears.")


if __name__ == "__main__":
    main()
