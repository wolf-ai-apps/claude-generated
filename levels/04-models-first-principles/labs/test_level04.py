import unittest

import numpy as np

from attention import attention, causal_mask
from bpe import BPE, CORPUS
from count_params import LLAMA_31_70B, LLAMA_31_8B, count


class TestBPE(unittest.TestCase):
    def test_roundtrip_and_compression(self):
        tok = BPE()
        tok.train(CORPUS, vocab_size=350)
        for text in ["The KV cache.", "नमस्ते", "", "a"]:
            self.assertEqual(tok.decode(tok.encode(text)), text)
        s = "The KV cache stores the keys"
        self.assertLess(len(tok.encode(s)), len(s.encode()) / 2)


class TestParams(unittest.TestCase):
    def test_llama_8b(self):
        self.assertAlmostEqual(count(LLAMA_31_8B)["total"] / 1e9, 8.03, places=2)

    def test_llama_70b(self):
        self.assertAlmostEqual(count(LLAMA_31_70B)["total"] / 1e9, 70.55, delta=0.1)


class TestAttention(unittest.TestCase):
    def test_gqa_equals_mha_with_repeated_kv(self):
        rng = np.random.default_rng(1)
        q = rng.standard_normal((4, 5, 8))
        k = rng.standard_normal((2, 5, 8))
        v = rng.standard_normal((2, 5, 8))
        gqa = attention(q, k, v)
        mha = attention(q, np.repeat(k, 2, 0), np.repeat(v, 2, 0))
        np.testing.assert_allclose(gqa, mha)

    def test_causal_first_token_only_sees_itself(self):
        rng = np.random.default_rng(2)
        q = rng.standard_normal((1, 3, 4))
        k = rng.standard_normal((1, 3, 4))
        v = rng.standard_normal((1, 3, 4))
        np.testing.assert_allclose(attention(q, k, v)[0, 0], v[0, 0])

    def test_mask_with_cache_offset(self):
        # one new query at position 3 attending to 4 cached keys: nothing masked
        self.assertFalse(causal_mask(1, 4).any())


if __name__ == "__main__":
    unittest.main()
