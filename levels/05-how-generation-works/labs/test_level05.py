import unittest

import numpy as np

from constrained import constrained_decode
from sampling import filter_probs, sample, softmax
from tiny_llama import Config, KVCache, forward, generate_no_cache, generate_with_cache, init_weights


class TestKVCache(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(layers=2, d=64, heads=4, kv_heads=2, ff=128)
        self.w = init_weights(self.cfg, seed=3)

    def test_cached_logits_match_full_recompute(self):
        toks = list(range(10, 30))
        full = forward(self.cfg, self.w, toks)
        cache = KVCache(self.cfg)
        forward(self.cfg, self.w, toks[:12], cache)         # prefill 12
        step = [forward(self.cfg, self.w, [t], cache)[0] for t in toks[12:]]  # decode 8
        np.testing.assert_allclose(np.array(step), full[12:], atol=1e-10)

    def test_generation_identical(self):
        prompt = [1, 2, 3, 4, 5]
        a = generate_no_cache(self.cfg, self.w, prompt, 10)
        b, cache = generate_with_cache(self.cfg, self.w, prompt, 10)
        self.assertEqual(a, b)
        self.assertEqual(cache.length, len(prompt) + 9)  # last token is never fed back

    def test_cache_size_formula(self):
        cache = KVCache(self.cfg)
        forward(self.cfg, self.w, list(range(7)), cache)
        per_token = 2 * self.cfg.layers * self.cfg.kv_heads * self.cfg.hd * 2
        self.assertEqual(cache.nbytes(), 7 * per_token)


class TestSampling(unittest.TestCase):
    def test_greedy(self):
        self.assertEqual(sample([1, 5, 2], np.random.default_rng(), temperature=0), 1)

    def test_top_k(self):
        p = filter_probs(softmax(np.array([5.0, 4, 3, 2, 1])), top_k=2)
        self.assertEqual(np.count_nonzero(p), 2)
        self.assertAlmostEqual(p.sum(), 1.0)

    def test_top_p_keeps_top_token(self):
        p = filter_probs(softmax(np.array([10.0, 0, 0])), top_p=0.1)
        self.assertEqual(np.count_nonzero(p), 1)

    def test_min_p(self):
        p = filter_probs(np.array([0.5, 0.3, 0.15, 0.05]), min_p=0.2)
        self.assertEqual(np.count_nonzero(p), 3)

    def test_repetition_penalty_discourages_repeat(self):
        rng = np.random.default_rng(0)
        self.assertEqual(sample([2.0, 1.9], rng, temperature=0, previous=[0], repetition_penalty=1.5), 1)


class TestConstrained(unittest.TestCase):
    def test_always_valid(self):
        vocab = list("abcxyz") + ["$"]
        rng = np.random.default_rng(1)
        allowed = ["abc", "ax", "zzz"]
        for _ in range(50):
            out = constrained_decode(lambda _: rng.standard_normal(len(vocab)), vocab, allowed, rng)
            self.assertIn(out, allowed)


if __name__ == "__main__":
    unittest.main()
