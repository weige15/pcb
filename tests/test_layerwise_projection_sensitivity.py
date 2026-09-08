"""Synthetic contract/control tests only; fixtures are never experiment results."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import layerwise_projection_sensitivity as lw


class LayerwiseTests(unittest.TestCase):
    def test_grid(self):
        self.assertEqual(len(lw.CONDITIONS), 109)
        self.assertEqual(len(set(lw.CONDITIONS)), 109)
        self.assertEqual({lw.identity(c) for c in lw.CONDITIONS[1:]},
                         {(i, p, 4) for i in range(36) for p in ("WQ", "WK", "WV")})
        for bad in ("L36_WQ4", "L00_WQ8", "WQ4", "L00_WO4"):
            with self.assertRaises(ValueError):
                lw.identity(bad)

    def test_full_bitwise_state_checks_and_exception_restore(self):
        # Tiny CPU test model; only target inventory lookup is patched.
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        attn = torch.nn.Module()
        model.model.layers[0].self_attn = attn
        attn.q_proj = torch.nn.Linear(128, 2, bias=False, dtype=torch.bfloat16)
        attn.k_proj = torch.nn.Linear(128, 2, bias=False, dtype=torch.bfloat16)
        model.register_buffer("buffer", torch.tensor([0.0]))
        original = {n: t.detach().clone() for n, t in lw.state_tensors(model).items()}
        hashes = lw.model_hashes(model)
        name = "model.layers.0.self_attn.q_proj"
        with torch.no_grad(), patch.object(lw, "targets_for", return_value={name: attn.q_proj}):
            ctrl = {}
            with lw.single_intervention(model, "L00_WQ4", original, hashes, ctrl):
                self.assertEqual(lw.changed_state(model, original), [name + ".weight"])
            self.assertTrue(ctrl["restored"])
            self.assertTrue(ctrl["unchanged_during_eval"])
            self.assertEqual(lw.model_hashes(model), hashes)
            ctrl = {}
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                with lw.single_intervention(model, "L00_WQ4", original, hashes, ctrl):
                    raise RuntimeError("simulated forward error")
            self.assertTrue(ctrl["restored"])
            self.assertNotIn("unchanged_during_eval", ctrl)
            self.assertEqual(lw.model_hashes(model), hashes)
            # Signed zero would pass numeric equality but must fail bitwise checks.
            model.buffer[0] = -0.0
            self.assertEqual(lw.changed_state(model, original), ["buffer"])
            with self.assertRaisesRegex(ValueError, "original BF16"):
                with lw.single_intervention(model, "L00_WQ4", original, hashes, {}):
                    pass
            model.buffer.copy_(original["buffer"])
            with self.assertRaisesRegex(ValueError, "restoration"):
                with lw.single_intervention(model, "L00_WQ4", original, hashes, {}):
                    attn.k_proj.weight[0, 0] += 1
            self.assertTrue(torch.equal(attn.q_proj.weight, original[name + ".weight"]))

    def fixture(self):
        indices = np.random.default_rng(42).choice(145, 64, replace=False)
        blocks = np.arange(64 * 2049, dtype=np.int64).reshape(64, 2049)
        name = "model.layers.0.self_attn.q_proj"
        manifest = {"original_state_sha256": {name + ".weight": "original", **{str(i): "hash" for i in range(399)}}}
        checks = [dict(sample_id=s, start=start, positions=64, logits_bitwise_equal=True,
                       cross_entropy_sum=1.0, metric_nll_sum=1.0) for s in (0, 63) for start in (0, 1984)]
        base = dict(condition="BF16", status="complete", rows=[], control=dict(state_tensors_checked=400,
                    unchanged_during_eval=True, historical_baseline_exact=True, api_alignment=checks))
        for s in range(64):
            base["rows"].append(dict(condition="BF16", layer=-1, projection="none", bits=16, context=2048,
                n_tokens=2048, sample_id=s, block_index=int(indices[s]), input_sha256=lw.token_hash(blocks[s]),
                labels_sha256=lw.token_hash(blocks[s, 1:]), nll=2.0, delta_nll=0.0, kl=0.0))
        result = deepcopy(base)
        result["condition"] = "L00_WQ4"
        for r in result["rows"]:
            r.update(condition="L00_WQ4", layer=0, projection="WQ", bits=4, nll=2.125, delta_nll=.125, kl=.125)
        result["control"] = dict(pre_matches_original=True, state_tensors_checked=400, name=name,
            shape=[4096, 2560], parameters=4096 * 2560, original_sha256="original", quantized_sha256="quantized",
            relative_l2_error=.1, changed_tensors=[name + ".weight"], non_target_unchanged=True, applied_exact=True,
            unchanged_during_eval=True, restored=True, restored_outputs=[dict(sample_id=s, hidden_bitwise_equal=True,
                                                                            nll=2.0, kl=0.0) for s in (0, 63)])
        return manifest, indices, blocks, base, result

    def test_contract_rejects_uncovered_controls_and_wrong_rows(self):
        manifest, indices, blocks, base, result = self.fixture()
        lw.validate_condition(base, manifest, indices, blocks)
        lw.validate_condition(result, manifest, indices, blocks, base)
        for key in ("pre_matches_original", "non_target_unchanged", "applied_exact", "unchanged_during_eval", "restored"):
            bad = deepcopy(result)
            bad["control"][key] = False
            with self.assertRaises(ValueError, msg=key):
                lw.validate_condition(bad, manifest, indices, blocks, base)
        for key, value in (("state_tensors_checked", 399), ("changed_tensors", ["wrong.weight"]),
                           ("shape", [1024, 2560]), ("original_sha256", "wrong"), ("restored_outputs", [])):
            bad = deepcopy(result)
            bad["control"][key] = value
            with self.assertRaises(ValueError, msg=key):
                lw.validate_condition(bad, manifest, indices, blocks, base)
        for key, value in (("sample_id", 1), ("context", 512), ("bits", 8), ("layer", 1),
                           ("labels_sha256", "wrong"), ("delta_nll", 0), ("kl", float("nan"))):
            bad = deepcopy(result)
            bad["rows"][0][key] = value
            with self.assertRaises(ValueError, msg=key):
                lw.validate_condition(bad, manifest, indices, blocks, base)
        bad = deepcopy(base)
        bad["control"]["api_alignment"] = bad["control"]["api_alignment"][:2]
        with self.assertRaises(ValueError):
            lw.validate_condition(bad, manifest, indices, blocks)

    def test_resume_inventory_rejects_corruption_and_incomplete_gate(self):
        manifest, indices, blocks, base, result = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "conditions").mkdir()
            np.savez_compressed(out / "sampled_token_blocks.npz", block_indices=indices, token_ids=blocks)
            lw.save_json(out / "conditions/BF16.json", base)
            lw.save_json(out / "conditions/L00_WQ4.json", result)
            self.assertEqual(list(lw.completed_results(out, manifest)), ["BF16", "L00_WQ4"])
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                lw.completed_results(out, manifest, complete=True)
            manifest["condition_sha256"] = {"L00_WQ4": lw.file_hash(out / "conditions/L00_WQ4.json")}
            result["rows"][0]["nll"] += 1
            lw.save_json(out / "conditions/L00_WQ4.json", result)
            with self.assertRaisesRegex(ValueError, "Changed committed"):
                lw.completed_results(out, manifest)
            (out / "conditions/L00_WQ4.json").unlink()
            with self.assertRaisesRegex(ValueError, "Missing previously"):
                lw.completed_results(out, manifest)


if __name__ == "__main__":
    unittest.main()
