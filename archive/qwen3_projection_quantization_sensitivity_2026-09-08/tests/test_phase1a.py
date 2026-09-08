"""Synthetic arithmetic/contract tests only; fixtures are NOT experiment data."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from scripts.phase1a import (CONDITIONS, CONTEXTS, FIELDS, ROOT, analyze, fake_quantize,
                            metric_from_logits, paired_bootstrap, read_and_validate_rows,
                            sample_blocks, score, token_hash, validate_config, validate_controls)


class Phase1aTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "scripts/phase1a_config.json").read_text())

    def test_config_is_fixed(self):
        validate_config(self.cfg)
        for key, bad in (("sample_blocks", 63), ("group_size", 64), ("use_cache", True), ("seed", 0)):
            cfg = dict(self.cfg, **{key: bad})
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_rtn_matches_independent_numpy_reference(self):
        rng = np.random.default_rng(7)
        weight = torch.tensor(rng.normal(size=(3, 256)), dtype=torch.bfloat16)
        weight[0, :128] = 0
        weight[1, 128:] *= 25
        original = weight.clone()
        for bits in (4, 8):
            actual = fake_quantize(weight, bits)
            expected = np.zeros(weight.shape, dtype=np.float32)
            # Deliberately use row/group loops instead of the runner's reshape.
            for row in range(3):
                for start in (0, 128):
                    group = weight[row, start:start + 128].float().numpy()
                    scale = np.max(np.abs(group)) / np.float32(2 ** (bits - 1) - 1)
                    if scale != 0:
                        expected[row, start:start + 128] = np.clip(np.rint(group / scale), -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1) * scale
            self.assertTrue(torch.equal(actual, torch.tensor(expected).to(torch.bfloat16)))
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(actual[0, :128], torch.zeros(128, dtype=torch.bfloat16)))
            self.assertTrue(torch.equal(weight, original))

    def test_round_ties_even_and_invalid_groups(self):
        weight = torch.zeros((1, 128), dtype=torch.bfloat16)
        weight[0, :7] = torch.tensor([7, -7, .5, 1.5, 2.5, -.5, -1.5])
        self.assertEqual(fake_quantize(weight, 4)[0, :7].tolist(), [7, -7, 0, 2, 2, 0, -2])
        with self.assertRaises(ValueError):
            fake_quantize(torch.zeros((2, 129), dtype=torch.bfloat16), 4)
        with self.assertRaises(ValueError):
            fake_quantize(weight, 3)
        weight[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            fake_quantize(weight, 4)

    def test_sampling_boundaries_tail_and_insufficient_data(self):
        tokens = list(range(70 * 2049 + 17))
        indices, blocks, n_full = sample_blocks(tokens, self.cfg)
        self.assertEqual(n_full, 70)
        self.assertEqual(blocks.shape, (64, 2049))
        self.assertEqual(len(set(indices)), 64)
        self.assertTrue(np.array_equal(indices, np.random.default_rng(42).choice(70, 64, replace=False)))
        for i, index in enumerate(indices):
            self.assertTrue(np.array_equal(blocks[i], tokens[index * 2049:(index + 1) * 2049]))
            self.assertEqual(blocks[i, 1:513].size, 512)
            self.assertEqual(blocks[i, 1:2049].size, 2048)
        with self.assertRaisesRegex(ValueError, "64 required"):
            sample_blocks(tokens[:64 * 2049 - 1], self.cfg)

    def test_metric_direction_nll_finite_and_alignment(self):
        ref = torch.tensor([[[.8, .2], [.3, .7]]], dtype=torch.float32)
        test = torch.tensor([[[.4, .6], [.6, .4]]], dtype=torch.float32)
        labels = torch.tensor([[0, 1]])
        nll, kl = metric_from_logits(test.log(), ref.log(), labels)
        expected_nll = -math.log(.4) * 2
        expected_kl = .8 * math.log(2) + .2 * math.log(1 / 3) + .3 * math.log(.5) + .7 * math.log(1.75)
        self.assertAlmostEqual(nll.item(), expected_nll, places=6)
        self.assertAlmostEqual(kl.item(), expected_kl, places=6)
        self.assertEqual(nll.dtype, torch.float32)
        self.assertEqual(kl.dtype, torch.float32)
        reverse = metric_from_logits(ref.log(), test.log(), labels)[1]
        self.assertGreater(abs(kl.item() - reverse.item()), .001)
        self.assertEqual(metric_from_logits(ref.log(), ref.log(), labels)[1].item(), 0)
        with self.assertRaises(ValueError):
            metric_from_logits(test.log(), ref.log(), labels[:, :1])
        with self.assertRaises(ValueError):
            metric_from_logits(test.log() * float("nan"), ref.log(), labels)

    def test_chunked_metric_matches_dense_including_partial_chunk(self):
        generator = torch.Generator().manual_seed(9)
        head = torch.nn.Linear(5, 11, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.randn(head.weight.shape, generator=generator))
        hidden = torch.randn(1, 17, 5, generator=generator)
        reference = torch.randn(1, 17, 5, generator=generator)
        labels = torch.randint(0, 11, (1, 17), generator=generator)
        dense = metric_from_logits(head(hidden), head(reference), labels)
        for chunk in (1, 4, 17, 64):
            actual = score(head, hidden, reference, labels, chunk)
            for a, b in zip(actual, dense):
                self.assertAlmostEqual(a, b.item() / 17, places=5)
        nll = F.cross_entropy(head(hidden).reshape(-1, 11), labels.reshape(-1))
        self.assertTrue(torch.allclose(dense[0] / 17, nll, atol=1e-6, rtol=1e-6))

    def test_paired_bootstrap_preserves_pairs(self):
        draws = np.random.default_rng(42).integers(0, 64, (2000, 64))
        x = np.arange(64, dtype=np.float64) ** 2
        self.assertEqual(paired_bootstrap(x, x, draws), (0, 0, 0))
        self.assertEqual(paired_bootstrap(x + 1, x, draws), (1, 1, 1))
        expected = (x - x[::-1])[draws].mean(axis=1)
        result = paired_bootstrap(x, x[::-1], draws)
        self.assertTrue(np.allclose(result[1:], np.percentile(expected, [2.5, 97.5])))

    def test_control_gate_rejects_missing_or_blocked_evidence(self):
        for manifest in ({}, {"status": "blocked"}, {"status": "measured", "config": self.cfg,
                                                       "completed_conditions": []}):
            with self.assertRaises(ValueError):
                validate_controls(manifest)

    def test_launcher_records_inventory_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mock_smi = root / "nvidia-smi"
            mock_smi.write_text("#!/bin/bash\necho 'synthetic inventory failure' >&2\nexit 3\n")
            mock_smi.chmod(0o700)
            output = root / "output"
            result = subprocess.run(["bash", str(ROOT / "scripts/run_phase1a.sh"), str(output)],
                                    env={**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"]},
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("synthetic inventory failure", result.stdout)
            self.assertIn("0/14", (output / "phase1a_blocked.md").read_text())
            self.assertFalse((output / "phase1a_run.json").exists())

    def test_result_contract_detects_tampering_and_ppl_reduction(self):
        # Synthetic fixture exists only in TemporaryDirectory, never results/.
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            indices, blocks, n_full = sample_blocks(list(range(70 * 2049)), self.cfg)
            np.savez_compressed(out / "phase1a_tokens.npz", block_indices=indices, token_ids=blocks)
            manifest = {"data": {"sample_indices": indices.tolist(), "full_blocks": n_full}, "rerun_command": "unit test fixture, NOT an experiment"}
            rows = []
            for c in CONDITIONS:
                for ctx in CONTEXTS:
                    for s in range(64):
                        delta = 0 if c == "BF16" else .25
                        nll = 1 + s % 2 + delta
                        rows.append(dict(condition=c, projection="none" if c == "BF16" else c[:2], bits=16 if c == "BF16" else int(c[2:]),
                                         context=ctx, sample_id=s, block_index=indices[s], n_tokens=ctx,
                                         input_sha256=token_hash(blocks[s, :ctx + 1]), labels_sha256=token_hash(blocks[s, 1:ctx + 1]),
                                         nll=nll, delta_nll=delta, kl=delta))

            def write_rows():
                with open(out / "phase1a.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)

            write_rows()
            self.assertEqual(len(read_and_validate_rows(out, manifest)), 896)
            # CSV alone is not completion: absence of controls must block report.
            with self.assertRaises(ValueError):
                analyze(out, manifest)
            self.assertFalse((out / "phase1a_report.md").exists())
            # Isolate statistical arithmetic from the control gate for this fixture.
            with patch("scripts.phase1a.validate_controls"):
                analyze(out, manifest)
            analysis = json.loads((out / "phase1a_analysis.json").read_text())
            self.assertAlmostEqual(analysis["summary"][0]["ppl"], math.exp(1.5))
            self.assertEqual(len(analysis["paired_comparisons"]), 24)
            rows[1]["labels_sha256"] = "invalid"
            write_rows()
            with self.assertRaisesRegex(ValueError, "Input/labels"):
                read_and_validate_rows(out, manifest)
            rows[1] = rows[0]
            write_rows()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_and_validate_rows(out, manifest)


if __name__ == "__main__":
    unittest.main()
