#!/usr/bin/env python3
"""Independent CPU artifact audit: source text/tokens, checkpoint/NumPy RTN, paired stats, concentration."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import layerwise_projection_sensitivity as lw
from scripts.analyze_layerwise_projection_sensitivity import validated_results


def check(ok, message):
    if not ok:
        raise ValueError(message)


def close(actual, expected, message):
    check(np.allclose(actual, expected, atol=1e-12, rtol=1e-12), message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=lw.OUTPUT)
    args = parser.parse_args()
    out = args.output.resolve()
    manifest, results = validated_results(out)
    history = json.loads((out / "historical_run.json").read_text())
    am = json.loads((out / "analysis_manifest.json").read_text())
    check(am["condition_sha256"] == manifest["condition_sha256"], "Analysis measured different conditions")
    for name, digest in am["source_sha256"].items():
        check(lw.file_hash(ROOT / name) == digest and lw.file_hash(out / "source" / name) == digest, "Analysis source changed")
    for name, digest in am["artifacts_sha256"].items():
        check(lw.file_hash(out / name) == digest, f"Analysis artifact changed: {name}")
    evidence = dict(status="RUNNING", historical_files_preserved=len(manifest["history_sha256"]),
                    real_conditions=109, interventions=108, full_state_tensors_per_control=400,
                    restored_predictive_checks=216, producing_attempts=len({r["attempt"] for r in results.values()}))

    data = history["data"]
    check(lw.file_hash(data["parquet"]) == data["parquet_sha256"], "Dataset parquet changed")
    texts = pq.read_table(data["parquet"], columns=["text"])["text"].to_pylist()
    check(len(texts) == data["rows"] and all(isinstance(t, str) for t in texts), "Text rows/nulls changed")
    text = "\n\n".join(texts)
    check(hashlib.sha256(text.encode()).hexdigest() == data["text_utf8_sha256"], "Text order/join changed")
    for name, digest in data["tokenizer_assets_sha256"].items():
        check(lw.file_hash(Path(data["tokenizer_source"]) / name) == digest, "Tokenizer asset changed")
    tokenizer = AutoTokenizer.from_pretrained(data["tokenizer_source"], local_files_only=True, trust_remote_code=False)
    tokens = np.asarray(tokenizer(text, add_special_tokens=False, truncation=False, padding=False)["input_ids"], dtype="<i8")
    check(hashlib.sha256(tokens.tobytes()).hexdigest() == data["token_stream_sha256_le_int64"], "Token stream differs")
    check(len(tokens) == data["total_tokens"] and len(tokens) // 2049 == data["full_blocks"] and
          len(tokens) % 2049 == data["dropped_tail_tokens"], "Block/tail lengths differ")
    indices, blocks = lw.load_samples(out)
    check(np.array_equal(indices, np.random.default_rng(42).choice(len(tokens) // 2049, 64, replace=False)), "Sampling changed")
    for s, i in enumerate(indices):
        check(np.array_equal(blocks[s], tokens[i * 2049:(i + 1) * 2049]), "Saved block differs from original text")
    evidence["retokenization"] = dict(rows=len(texts), tokens=len(tokens), blocks=64, predictions_per_block=2048,
                                       tail_discarded=len(tokens) % 2049, exact=True)

    # The independent all-layer historical RTN4 has the same per-module weights,
    # although its measured outcomes are intentionally NOT reused as interventions.
    historical_quant = {m["name"]: m["quantized_sha256"] for r in history["interventions"]
                        if r["stage"] == "formal" and r["condition"].endswith("4") for m in r["modules"]}
    records = {r["control"]["name"] + ".weight": r["control"] for c, r in results.items() if c != "BF16"}
    model_path = Path(history["model_source"])
    check(lw.file_hash(model_path / "config.json") == history["checkpoint_config_sha256"], "Model config changed")
    torch.set_num_threads(4)
    checked_parameters, checked_rtn = 0, 0
    for shard, digest in history["checkpoint_sha256"].items():
        check(lw.file_hash(model_path / shard) == digest, f"Checkpoint changed: {shard}")
        with safe_open(str(model_path / shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name not in manifest["original_state_sha256"]:
                    continue
                weight = f.get_tensor(name)
                check(weight.dtype == torch.bfloat16 and lw.tensor_hash(weight) == manifest["original_state_sha256"][name],
                      f"Original BF16 parameter differs: {name}")
                checked_parameters += 1
                if name not in records:
                    continue
                record = records[name]
                values = weight.float().numpy()
                groups = values.reshape(values.shape[0], -1, 128)
                # Match pinned CUDA FP32 division by scalar (reciprocal multiply),
                # independently via NumPy; no tolerant replacement for RTN hashes.
                scales = np.max(np.abs(groups), axis=-1, keepdims=True) * np.float32(1.0 / 7.0)
                integers = np.rint(groups / np.where(scales == 0, np.float32(1), scales))
                dequant = torch.from_numpy((np.clip(integers, -7, 7) * scales).reshape(values.shape)).to(torch.bfloat16)
                digest = lw.tensor_hash(dequant)
                check(digest == record["quantized_sha256"] == historical_quant[record["name"]], f"Independent RTN4 differs: {name}")
                delta = dequant.float().numpy().astype(np.float64) - values
                rel = float(np.sqrt((delta ** 2).sum() / (values.astype(np.float64) ** 2).sum()))
                check(math.isclose(rel, record["relative_l2_error"], rel_tol=2e-4, abs_tol=1e-7), f"Weight error differs: {name}")
                checked_rtn += 1
    check(checked_parameters == 398 and checked_rtn == 108, "Incomplete checkpoint/RTN coverage")
    evidence["checkpoint_rtn"] = dict(original_parameters=398, runtime_buffers_matched_to_history=2,
                                      independent_numpy_rtn4_hashes_exact=108, historical_quantizer_hashes_exact=108)

    with open(out / "block_metrics.csv", newline="") as f:
        reader = csv.DictReader(f)
        check(reader.fieldnames == lw.FIELDS, "CSV fields differ")
        raw = list(reader)
    check(len(raw) == 6976, "Expected 6976 CSV measurements")
    seen = set()
    for row in raw:
        for key in ("layer", "bits", "context", "sample_id", "block_index", "n_tokens"):
            row[key] = int(row[key])
        for key in ("nll", "delta_nll", "kl"):
            row[key] = float(row[key])
        key = (row["condition"], row["sample_id"])
        check(key not in seen, "Duplicate CSV row")
        seen.add(key)
        check(row == results[key[0]]["rows"][key[1]], "CSV differs from committed measurement")
    evidence["raw_csv"] = dict(rows=len(raw), scored_positions=sum(r["n_tokens"] for r in raw),
                                distinct_scored_positions=64 * 2048, exact_condition_rows=True)
    analysis = json.loads((out / "sensitivity_analysis.json").read_text())
    draws = np.random.default_rng(42).integers(0, 64, size=(2000, 64))
    check(np.array_equal(draws, np.load(out / "paired_bootstrap_indices.npy", allow_pickle=False)), "Bootstrap draws differ")
    check(len(analysis["summary"]) == 109 and {r["condition"] for r in analysis["summary"]} == set(lw.CONDITIONS), "Incomplete summary")
    for row in analysis["summary"]:
        source = results[row["condition"]]["rows"]
        values = np.array([r["nll"] for r in source])
        close([values.mean(), math.exp(values.mean())], [row["nll"], row["ppl"]], "NLL/PPL mean mismatch")
        for metric in ("delta_nll", "kl"):
            values = np.array([r[metric] for r in source])
            lo, hi = np.percentile(np.mean(values[draws], axis=1), [2.5, 97.5])
            close([values.mean(), lo, hi], [row[metric], row[metric + "_ci_low"], row[metric + "_ci_high"]], "Baseline paired CI mismatch")
    expected_pairs = {(i, a + " - " + b, m) for i in range(36) for a, b in (("WQ", "WK"), ("WQ", "WV"), ("WK", "WV")) for m in ("delta_nll", "kl")}
    check(len(analysis["paired_comparisons"]) == 216 and {(r["layer"], r["contrast"], r["metric"]) for r in analysis["paired_comparisons"]} == expected_pairs,
          "Incomplete paired comparisons")
    for row in analysis["paired_comparisons"]:
        a, b = row["contrast"].split(" - ")
        x = np.array([r[row["metric"]] for r in results[f"L{row['layer']:02d}_{a}4"]["rows"]])
        y = np.array([r[row["metric"]] for r in results[f"L{row['layer']:02d}_{b}4"]["rows"]])
        distribution = x[draws].mean(axis=1) - y[draws].mean(axis=1)
        lo, hi = np.percentile(distribution, [2.5, 97.5])
        close([x.mean() - y.mean(), lo, hi], [row["mean"], row["ci_low"], row["ci_high"]], "Independent paired bootstrap differs")
    check(len(analysis["concentration"]) == 8 and {(r["projection"], r["metric"]) for r in analysis["concentration"]} ==
          {(p, m) for p in ("WQ", "WK", "WV", "ALL") for m in ("delta_nll", "kl")}, "Incomplete concentration profiles")
    for row in analysis["concentration"]:
        projections = list(lw.PROJECTIONS) if row["projection"] == "ALL" else [row["projection"]]
        values = np.array([[[r[row["metric"]] for r in results[f"L{i:02d}_{p}4"]["rows"]] for i in range(36)] for p in projections])
        scores = np.clip(values.mean(axis=2), 0, None).sum(axis=0)
        ranking = sorted(range(36), key=lambda i: (-scores[i], i))
        close(scores, row["layer_scores"], "Concentration layer scores differ")
        check(ranking == row["ranked_layers"], "Concentration ranking differs")
        close(scores.sum(), row["total_score"], "Concentration total differs")
        if scores.sum() == 0:
            check(row["top4_share"] is None and row["cumulative_share"] is None, "Zero denominator fabricated")
        else:
            cumulative = np.cumsum([scores[i] for i in ranking]) / scores.sum()
            close(cumulative, row["cumulative_share"], "Cumulative curve differs")
            close([cumulative[0], cumulative[3], cumulative[7], scores.sum() ** 2 / np.dot(scores, scores)],
                  [row["top1_share"], row["top4_share"], row["top8_share"], row["effective_layers"]], "Concentration summary differs")
            check(sum(v < .5 for v in cumulative) + 1 == row["layers_for_50pct"] and
                  sum(v < .8 for v in cumulative) + 1 == row["layers_for_80pct"], "Coverage layer counts differ")
        # Explicit draw loop, recomputing both means and rankings, independent of
        # the vectorized production bootstrap/concentration implementation.
        top4, effective = [], []
        for draw in draws:
            sampled = np.clip(values[:, :, draw].mean(axis=2), 0, None).sum(axis=0)
            total = sampled.sum()
            if total > 0:
                top4.append(sum(sorted(sampled, reverse=True)[:4]) / total)
                effective.append(total ** 2 / np.dot(sampled, sampled))
        check(len(top4) == row["defined_resamples"], "Defined bootstrap count differs")
        if top4:
            close(np.percentile(top4, [2.5, 97.5]), row["top4_share_ci"], "Reranked concentration CI differs")
            close(np.percentile(effective, [2.5, 97.5]), row["effective_layers_ci"], "Effective layers CI differs")
        else:
            check(row["top4_share_ci"] is row["effective_layers_ci"] is None, "Undefined CI fabricated")
    for name, expected in (("layer_summary.csv", analysis["summary"]), ("paired_comparisons.csv", analysis["paired_comparisons"])):
        with open(out / name, newline="") as f:
            rows = list(csv.DictReader(f))
        check(rows == [{k: str(v) for k, v in r.items()} for r in expected], f"{name} differs from analyzed values")
    check(json.loads((out / "concentration.json").read_text()) == analysis["concentration"], "Concentration export differs")
    from PIL import Image
    for name in am["figures"]:
        path = out / name
        if path.suffix == ".png":
            with Image.open(path) as im:
                check(im.width > 1000 and im.height > 800, "Undersized figure")
                im.verify()
        else:
            check(path.read_bytes().startswith(b"%PDF-"), "Invalid PDF header")
    evidence["statistics"] = dict(summary_cells=109, baseline_metric_cis=218, paired_metric_contrasts=216,
                                   concentration_profiles=8, draws=[2000, 64], independent_recalculation=True)
    evidence["figure_files"] = am["figures"]
    evidence["verifier_sha256"] = lw.file_hash(Path(__file__))
    evidence["status"] = "PASS: independent artifact audit; manual prompt-to-artifact and visual review still required"
    lw.save_json(out / "artifact_verification.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
