"""Synthetic contract/control tests only; fixtures are never experiment results."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import layerwise_projection_sensitivity as lw
from scripts import analyze_layerwise_projection_sensitivity as analysis


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
            cache = out / "baseline_hidden.pt"
            cache.write_bytes(b"synthetic cache identity fixture")
            base["hidden_cache"] = dict(path=cache.name, sha256=lw.file_hash(cache))
            lw.save_json(out / "conditions/BF16.json", base)
            lw.save_json(out / "conditions/L00_WQ4.json", result)
            self.assertEqual(list(lw.completed_results(out, manifest)), ["BF16", "L00_WQ4"])
            # Common inventory gate runs even when no future forward is needed.
            cache.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "cache changed"):
                lw.completed_results(out, manifest)
            cache.unlink()
            with self.assertRaisesRegex(ValueError, "Missing BF16 hidden cache"):
                lw.completed_results(out, manifest)
            cache.write_bytes(b"synthetic cache identity fixture")
            missing_metadata = deepcopy(base)
            del missing_metadata["hidden_cache"]
            lw.save_json(out / "conditions/BF16.json", missing_metadata)
            with self.assertRaisesRegex(ValueError, "cache metadata"):
                lw.completed_results(out, manifest)
            lw.save_json(out / "conditions/BF16.json", base)
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


class LayerwiseAnalysisTests(unittest.TestCase):
    def test_concentration_limits_and_zero_denominator(self):
        r = analysis.concentration(np.ones(36))
        self.assertAlmostEqual(r["top4_share"], 4 / 36)
        self.assertEqual(r["effective_layers"], 36)
        self.assertEqual((r["layers_for_50pct"], r["layers_for_80pct"]), (18, 29))
        values = np.zeros(36)
        values[17] = 7
        r = analysis.concentration(values)
        self.assertEqual(r["ranked_layers"][:4], [17, 0, 1, 2])
        self.assertEqual(r["top4_share"], 1)
        self.assertEqual(r["effective_layers"], 1)
        self.assertEqual(r["layers_for_80pct"], 1)
        r = analysis.concentration(np.zeros(36))
        self.assertIsNone(r["top4_share"])
        self.assertIsNone(r["effective_layers"])
        self.assertIsNone(analysis.concentration_ci(np.zeros((2000, 36)))["top4_share_ci"])
        with self.assertRaises(ValueError):
            analysis.concentration(-np.ones(36))

    def test_full_statistics_pairing_ppl_and_positive_score_sum(self):
        # Deliberately synthetic, no manifests/files: pure statistical arithmetic.
        results = {}
        for c in lw.CONDITIONS:
            layer, p, _ = lw.identity(c)
            effect = {"none": 0, "WQ": .25, "WK": .125, "WV": -.125}[p]
            results[c] = {"rows": [dict(nll=1 + s % 2 + effect, delta_nll=effect, kl=abs(effect)) for s in range(64)]}
        draws = np.random.default_rng(42).integers(0, 64, size=(2000, 64))
        result = analysis.summarize(results, draws)
        self.assertEqual(len(result["summary"]), 109)
        self.assertEqual(len(result["paired_comparisons"]), 216)
        self.assertEqual(len(result["concentration"]), 8)
        self.assertAlmostEqual(result["summary"][0]["ppl"], math.exp(1.5))
        self.assertNotAlmostEqual(result["summary"][0]["ppl"], (math.exp(1) + math.exp(2)) / 2)
        for row in result["paired_comparisons"]:
            self.assertEqual(row["ci_low"], row["mean"])
            self.assertEqual(row["ci_high"], row["mean"])
        qk = result["paired_comparisons"][0]
        self.assertEqual(qk["mean"], .125)
        all_nll = next(r for r in result["concentration"] if r["projection"] == "ALL" and r["metric"] == "delta_nll")
        self.assertEqual(all_nll["layer_scores"], [.375] * 36)  # Sum positive parts, not positive part of sum.
        wv_nll = next(r for r in result["concentration"] if r["projection"] == "WV" and r["metric"] == "delta_nll")
        self.assertEqual(wv_nll["negative_mean_layers"], list(range(36)))
        self.assertEqual(wv_nll["defined_resamples"], 0)
        self.assertIsNone(wv_nll["top4_share"])
        with tempfile.TemporaryDirectory() as tmp:
            files = analysis.figures(Path(tmp), result)
            self.assertEqual(len(files), 6)
            self.assertTrue(all((Path(tmp) / p).stat().st_size > 1000 for p in files))

    def test_reranking_not_fixed_top_layers(self):
        # Winner changes in every draw: dynamic top4 mass stays one.
        scores = np.zeros((36, 36))
        np.fill_diagonal(scores, 1)
        r = analysis.concentration_ci(scores)
        self.assertEqual(r["top4_share_ci"], [1, 1])
        self.assertEqual(r["effective_layers_ci"], [1, 1])
        self.assertEqual(r["defined_resamples"], 36)

    def test_projection_order_retains_exact_ties(self):
        self.assertEqual(analysis.projection_order(dict(WQ=0, WK=0, WV=0)), "WQ = WK = WV")
        self.assertEqual(analysis.projection_order(dict(WQ=1, WK=1, WV=0)), "WQ = WK > WV")
        self.assertEqual(analysis.projection_order(dict(WQ=0, WK=1, WV=1)), "WK = WV > WQ")
        self.assertEqual(analysis.projection_order(dict(WQ=1, WK=1 + 1e-12, WV=0)), "WK > WQ > WV")

    def test_interrupted_attempt_keeps_prior_valid_commits_for_analysis(self):
        fixture = LayerwiseTests()
        _, indices, blocks, base, result = fixture.fixture()
        originals = {f"model.layers.{i}.self_attn.{suffix}.weight": "original" for i in range(36) for suffix in lw.PROJECTIONS.values()}
        originals.update({str(i): "unused" for i in range(400 - len(originals))})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "conditions").mkdir()
            (out / "attempts").mkdir()
            np.savez_compressed(out / "sampled_token_blocks.npz", block_indices=indices, token_ids=blocks)
            cache = out / "baseline_hidden.pt"
            cache.write_bytes(b"synthetic cache identity fixture; not a model cache")
            base["hidden_cache"] = dict(path=cache.name, sha256=lw.file_hash(cache))
            old = dict(status="blocked", original_state_matches_history=True, state_tensors=400, tied_lm_head=True,
                       runtime_attention_backend="sdpa", skipped_completed=[], versions={"torch": "test"})
            new = dict(old, status="complete", final_state_matches_checkpoint=True,
                       skipped_completed=lw.CONDITIONS[:2], newly_completed=lw.CONDITIONS[2:])
            lw.save_json(out / "attempts/old.json", old)
            lw.save_json(out / "attempts/new.json", new)
            manifest = dict(status="measured", original_state_sha256=originals, condition_sha256={},
                            completed_conditions=lw.CONDITIONS, attempts=["attempts/old.json", "attempts/new.json"])
            for c in lw.CONDITIONS:
                record = deepcopy(base if c == "BF16" else result)
                record["condition"] = c
                record["attempt"] = "attempts/old.json" if c in lw.CONDITIONS[:2] else "attempts/new.json"
                if c != "BF16":
                    layer, p, bits = lw.identity(c)
                    name = f"model.layers.{layer}.self_attn.{lw.PROJECTIONS[p]}"
                    width = 4096 if p == "WQ" else 1024
                    record["control"].update(name=name, shape=[width, 2560], parameters=width * 2560, changed_tensors=[name + ".weight"])
                    for row in record["rows"]:
                        row.update(condition=c, layer=layer, projection=p, bits=bits)
                path = out / "conditions" / (c + ".json")
                lw.save_json(path, record)
                manifest["condition_sha256"][c] = lw.file_hash(path)
            lw.save_json(out / "run_manifest.json", manifest)
            with patch.object(lw, "verify_inputs", return_value={"versions": {"torch": "test"}}):
                _, accepted = analysis.validated_results(out)
                self.assertEqual(len(accepted), 109)
                self.assertEqual(json.loads((out / "attempts/old.json").read_text())["status"], "blocked")
                before = dict(manifest["condition_sha256"])
                with patch.object(lw, "run_missing", side_effect=AssertionError("must not run inference")), \
                     patch.object(sys, "argv", ["layerwise_projection_sensitivity.py", "--output", str(out), "--resume"]):
                    lw.main()
                self.assertEqual(json.loads((out / "run_manifest.json").read_text())["condition_sha256"], before)
                # A blocked attempt's *unrestored* commit must still be rejected.
                path = out / "conditions/L00_WQ4.json"
                bad = json.loads(path.read_text())
                bad["control"]["restored"] = False
                lw.save_json(path, bad)
                manifest["condition_sha256"]["L00_WQ4"] = lw.file_hash(path)
                lw.save_json(out / "run_manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "restored"):
                    analysis.validated_results(out)

    def test_launcher_failure_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smi = root / "nvidia-smi"
            smi.write_text("#!/bin/bash\necho 'synthetic no GPU inventory' >&2\nexit 3\n")
            smi.chmod(0o700)
            output = root / "output"
            result = subprocess.run(["bash", str(lw.ROOT / "scripts/run_layerwise_projection_sensitivity.sh"), str(output)],
                                    env={**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"]}, text=True, capture_output=True)
            self.assertEqual(result.returncode, 3)
            self.assertIn("0/109", (output / "blocked.md").read_text())
            self.assertFalse((output / "run_manifest.json").exists())
        for script in ("layerwise_projection_sensitivity.py", "analyze_layerwise_projection_sensitivity.py", "verify_layerwise_projection_sensitivity.py"):
            result = subprocess.run([sys.executable, str(lw.ROOT / "scripts" / script), "--help"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()
