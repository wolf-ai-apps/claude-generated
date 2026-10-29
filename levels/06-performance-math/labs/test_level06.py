import unittest

from roofline import attainable_tflops, intensity, ridge


class TestRoofline(unittest.TestCase):
    def test_decode_intensity_equals_batch_in_bf16(self):
        self.assertEqual(intensity(1), 1)
        self.assertEqual(intensity(64), 64)
        self.assertEqual(intensity(64, bytes_per_weight=1), 128)

    def test_h100_ridge(self):
        self.assertAlmostEqual(ridge("h100"), 295, delta=1)

    def test_batch1_decode_is_memory_bound(self):
        # 3.35 TB/s x 1 FLOP/byte = 3.35 TFLOP/s, a fraction of 989
        self.assertAlmostEqual(attainable_tflops("h100", intensity(1)), 3.35)
        self.assertEqual(attainable_tflops("h100", intensity(4096)), 989)


if __name__ == "__main__":
    unittest.main()
