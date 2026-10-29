"""Lab 5c: structured output by masking tokens.

"JSON mode" and tool-calling guarantee valid output the same way: at every
step, compute which tokens keep the output valid and set every other token's
logit to -inf before sampling. Engines compile a JSON schema or grammar into
an automaton to do this fast (vLLM uses xgrammar/guidance/outlines backends).

Here the "grammar" is a fixed set of allowed strings stored in a prefix trie,
which is enough to show the mechanism.

    python3 constrained.py
"""

import numpy as np

from sampling import softmax


class Trie:
    def __init__(self, words):
        self.root = {}
        for w in words:
            node = self.root
            for ch in w:
                node = node.setdefault(ch, {})
            node["$"] = {}  # end marker

    def allowed_next(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node:
                return set()
            node = node[ch]
        return set(node)


def constrained_decode(logit_fn, vocab, allowed_outputs, rng, temperature=1.0, max_steps=50):
    """vocab: list of single-character tokens plus '$' (end of output)."""
    trie = Trie(allowed_outputs)
    out = ""
    for _ in range(max_steps):
        logits = logit_fn(out).astype(float)
        ok = trie.allowed_next(out)
        mask = np.array([tok in ok for tok in vocab])
        logits[~mask] = -np.inf                       # the whole trick
        tok = vocab[int(rng.choice(len(vocab), p=softmax(logits / temperature)))]
        if tok == "$":
            return out
        out += tok
    return out


def main():
    vocab = list("abcdefghijklmnopqrstuvwxyz _\"{}:,") + ["$"]
    rng = np.random.default_rng(0)
    noisy_model = lambda prefix: rng.standard_normal(len(vocab)) * 3   # a model that knows nothing
    labels = ['{"sentiment": "positive"}', '{"sentiment": "negative"}', '{"sentiment": "neutral"}']

    print("Unconstrained samples from a random model:")
    for _ in range(3):
        print("  ", "".join(vocab[i] for i in rng.integers(0, len(vocab) - 1, 25)))
    print("\nConstrained to the schema {sentiment: positive|negative|neutral}:")
    for _ in range(5):
        s = constrained_decode(noisy_model, vocab, labels, rng)
        assert s in labels
        print("  ", s)
    print("\nEven a model that outputs noise can only produce valid JSON. A real model's logits"
          "\nmake the choice meaningful; the mask makes it valid.")


if __name__ == "__main__":
    main()
