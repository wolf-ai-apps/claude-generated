"""Lab 7d: speculative decoding, and proof that it is lossless.

    python3 speculative.py

A cheap draft model proposes k tokens; the target model scores all k in ONE
forward pass (decode is memory-bound, so checking k tokens costs about the
same as generating one). Each draft token x is accepted with probability
min(1, p(x)/q(x)); on rejection we sample from the residual max(0, p - q).
The output distribution is exactly the target's (Leviathan et al. 2023).
"""

import numpy as np

V = 8  # vocabulary size


def target_dist(ctx):
    """Stand-in for the big model: a deterministic distribution per context."""
    rng = np.random.default_rng(hash(tuple(ctx[-2:])) % 2**32)
    logits = rng.standard_normal(V) * 2
    p = np.exp(logits - logits.max())
    return p / p.sum()


def draft_dist(ctx, quality=0.8):
    """Stand-in for the small model: the target blurred towards uniform."""
    return quality * target_dist(ctx) + (1 - quality) / V


def speculative_step(ctx, k, rng, quality=0.8):
    """One target forward pass. Returns the accepted tokens (1 to k+1 of them)."""
    drafted, qs, c = [], [], list(ctx)
    for _ in range(k):                         # k cheap draft steps
        q = draft_dist(c, quality)
        x = int(rng.choice(V, p=q))
        drafted.append(x)
        qs.append(q)
        c.append(x)
    out, c = [], list(ctx)
    for x, q in zip(drafted, qs):              # verified together in one target pass
        p = target_dist(c)
        if rng.random() < min(1.0, p[x] / q[x]):
            out.append(x)
            c.append(x)
            continue
        residual = np.maximum(p - q, 0)
        out.append(int(rng.choice(V, p=residual / residual.sum())))
        return out
    out.append(int(rng.choice(V, p=target_dist(c))))   # all accepted: free bonus token
    return out


def acceptance_rate(ctx, quality):
    """alpha = sum_x min(p(x), q(x)): the chance a draft token is accepted."""
    p, q = target_dist(ctx), draft_dist(ctx, quality)
    return float(np.minimum(p, q).sum())


def expected_tokens(alpha, k):
    """Expected tokens per target pass: (1 - alpha^(k+1)) / (1 - alpha)."""
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def main():
    rng = np.random.default_rng(0)
    ctx = [1, 2]
    n = 20000
    first = np.zeros(V)
    for _ in range(n):
        first[speculative_step(ctx, k=4, rng=rng)[0]] += 1
    print("1) Lossless: distribution of the first emitted token")
    print("   target p :", np.round(target_dist(ctx), 3))
    print("   observed :", np.round(first / n, 3))

    print("\n2) Speed-up depends on acceptance rate and draft cost (draft step = 5% of a target step)")
    print(f"   {'draft quality':<14}{'alpha':>7}{'k':>4}{'tokens/pass':>13}{'speed-up':>10}")
    for quality in (0.5, 0.8, 0.95):
        alpha = np.mean([acceptance_rate([a, b], quality) for a in range(V) for b in range(V)])
        for k in (2, 4, 8):
            e = expected_tokens(alpha, k)
            speedup = e / (1 + 0.05 * k)
            print(f"   {quality:<14}{alpha:>7.2f}{k:>4}{e:>13.2f}{speedup:>9.2f}x")
    print("\nAt large batch sizes decode becomes compute-bound, verifying k tokens is no longer"
          "\n'free', and speculation can make throughput worse. Measure before enabling it.")


if __name__ == "__main__":
    main()
