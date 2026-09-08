import json
from pathlib import Path
import unittest

import numpy as np
import torch

from scripts import kv_cache_precision_pareto as runner


class KVCachePrecisionParetoTests(unittest.TestCase):
    def test_requested_modes_are_real_byte_dtypes_not_fake_4bit(self):
        self.assertEqual(runner.mode_dtype("BF16"), torch.bfloat16)
        self.assertEqual(runner.mode_dtype("FP8_E4M3"), torch.float8_e4m3fn)
        self.assertEqual(runner.mode_dtype("FP8_E5M2"), torch.float8_e5m2)
        self.assertNotIn("INT4", runner.MODES)

    def test_fp8_payload_is_one_byte_and_not_bf16(self):
        x = torch.tensor([[[-1.0, 0.0, 1.0, 2.0]]], dtype=torch.bfloat16)
        q = runner.quantize_cache(x, "FP8_E4M3")
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(q.element_size(), 1)
        self.assertEqual(runner.quantize_cache(x, "BF16").dtype, torch.bfloat16)

    def test_fixed_input_shape_contract(self):
        cfg = json.loads(Path("results/kv_cache_precision_pareto/experiment_config.json").read_text())
        self.assertEqual(cfg["prompt_lengths"], [2048, 8192])
        self.assertEqual(cfg["output_length"], 256)
        self.assertEqual(cfg["performance_concurrency"], [1, 4, 8, 16, 32])
        self.assertEqual(cfg["weights_dtype"], cfg["compute_dtype"], "weights/compute must remain BF16")

    def test_bootstrap_is_paired_shape(self):
        values = np.arange(8, dtype=np.float64)
        draws = np.random.default_rng(424242).integers(0, 8, size=(2000, 8))
        mean, low, high = __import__("scripts.analyze_kv_cache_precision_pareto", fromlist=["ci"]).ci(values, draws).values()
        self.assertEqual(draws.shape, (2000, 8))
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_protocol_marks_gates_provisional_and_forbids_restore_path(self):
        protocol = Path("results/kv_cache_precision_pareto/protocol.md").read_text()
        self.assertIn("provisional", protocol)
        self.assertIn("reconstructs the entire cache to BF16", protocol)
        self.assertIn("directly", protocol)
        self.assertIn("NO-GO", protocol)


if __name__ == "__main__":
    unittest.main()
