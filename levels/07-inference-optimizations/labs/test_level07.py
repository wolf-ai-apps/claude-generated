import unittest

import numpy as np

from batching_sim import run_continuous, run_static, workload
from paged_kv import BlockAllocator, OutOfBlocks, Sequence, contiguous_waste, paged_waste
from quantize import (dequant_int4_group, dequant_int8, quant_int4_group, quant_int8_per_channel,
                      rel_err)
from speculative import V, expected_tokens, speculative_step, target_dist


def ttft_p50(reqs):
    t = sorted(r.token_times[0] - r.arrival for r in reqs)
    return t[len(t) // 2]


def itl_p99(reqs):
    t = sorted(b - a for r in reqs for a, b in zip(r.token_times, r.token_times[1:]))
    return t[int(0.99 * len(t))]


class TestBatching(unittest.TestCase):
    def test_every_request_completes_with_all_tokens(self):
        for reqs in (run_static(workload(50)), run_continuous(workload(50)),
                     run_continuous(workload(50), chunk_tokens=256)):
            for r in reqs:
                self.assertEqual(len(r.token_times), r.output)

    def test_continuous_beats_static_ttft(self):
        self.assertLess(ttft_p50(run_continuous(workload())), ttft_p50(run_static(workload())) / 3)

    def test_chunked_prefill_cuts_itl_tail(self):
        self.assertLess(itl_p99(run_continuous(workload(), chunk_tokens=512)),
                        itl_p99(run_continuous(workload())))


class TestPagedKV(unittest.TestCase):
    def test_prefix_blocks_are_shared(self):
        a = BlockAllocator(32, block_size=4)
        s1, s2 = Sequence(a), Sequence(a)
        s1.prefill(list(range(10)))
        s2.prefill(list(range(8)) + [99])
        self.assertEqual(s2.cached_tokens, 8)
        self.assertEqual(s1.table[:2], s2.table[:2])
        self.assertEqual(s2.tokens(), list(range(8)) + [99])

    def test_hash_chain_requires_whole_prefix(self):
        a = BlockAllocator(32, block_size=4)
        Sequence(a).prefill([1, 2, 3, 4, 5, 6, 7, 8])
        s = Sequence(a)
        s.prefill([0, 0, 0, 0, 5, 6, 7, 8])  # same 2nd block, different 1st block
        self.assertEqual(s.cached_tokens, 0)

    def test_copy_on_write(self):
        a = BlockAllocator(16, block_size=4)
        p = Sequence(a)
        p.prefill([1, 2, 3, 4, 5], use_prefix_cache=False)
        c = p.fork()
        p.append(6)
        c.append(7)
        self.assertEqual(p.tokens(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(c.tokens(), [1, 2, 3, 4, 5, 7])
        self.assertEqual(p.table[0], c.table[0])        # full block still shared
        self.assertNotEqual(p.table[1], c.table[1])     # partial block was copied

    def test_free_and_out_of_blocks(self):
        a = BlockAllocator(2, block_size=2)
        s = Sequence(a)
        s.prefill([1, 2, 3, 4], use_prefix_cache=False)
        with self.assertRaises(OutOfBlocks):
            s.append(5)
        s.free()
        self.assertEqual(a.num_free(), 2)

    def test_paged_wastes_less(self):
        lengths = [100, 900, 1500, 30]
        self.assertLess(paged_waste(lengths, 16), 0.02)
        self.assertGreater(contiguous_waste(lengths, 2048), 0.5)


class TestQuantize(unittest.TestCase):
    def setUp(self):
        self.w = (np.random.default_rng(0).standard_normal((64, 256)) * 0.02).astype(np.float32)

    def test_int8(self):
        q, s = quant_int8_per_channel(self.w)
        self.assertEqual(q.dtype, np.int8)
        self.assertLess(rel_err(self.w, dequant_int8(q, s)), 0.01)

    def test_int4_packing_and_error(self):
        p, s, z = quant_int4_group(self.w, group=128)
        self.assertEqual(p.nbytes, self.w.size // 2)
        e = rel_err(self.w, dequant_int4_group(p, s, z, group=128))
        self.assertLess(e, 0.12)
        self.assertGreater(e, 0.01)


class TestSpeculative(unittest.TestCase):
    def test_lossless(self):
        rng = np.random.default_rng(1)
        ctx = [3, 4]
        n = 30000
        counts = np.zeros(V)
        for _ in range(n):
            counts[speculative_step(ctx, k=3, rng=rng, quality=0.6)[0]] += 1
        np.testing.assert_allclose(counts / n, target_dist(ctx), atol=0.012)

    def test_expected_tokens(self):
        self.assertAlmostEqual(expected_tokens(0.0, 4), 1.0)
        self.assertAlmostEqual(expected_tokens(0.5, 1), 1.5)


if __name__ == "__main__":
    unittest.main()
