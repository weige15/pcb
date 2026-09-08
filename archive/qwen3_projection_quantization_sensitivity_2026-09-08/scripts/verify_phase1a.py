#!/usr/bin/env python3
"""Artifact audit, including independent NumPy RTN/statistics and re-tokenization.

No GPU/model forward and no synthetic measurements. This supplements (does not
replace) real smoke/control logs and the human prompt-to-artifact audit.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
import torch
from transformers import AutoTokenizer

from phase1a import ROOT, file_hash, read_and_validate_rows, validate_controls


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--compare-smoke", type=Path, help="Optional independently executed pre-formal smoke output directory")
    args = parser.parse_args()
    out = args.output.resolve()
    run = json.loads((out / "phase1a_run.json").read_text())
    check(run["status"] == "measured", "Only a finished measured run can be audited")
    validate_controls(run)
    keyed = read_and_validate_rows(out, run)
    evidence = {"rows": len(keyed), "conditions": len(run["completed_conditions"]),
                "real_controls_gate": "PASS", "source_hashes": {}, "artifact_hashes": {}}
    for group, base, hashes in (("source_hashes", ROOT, run["source_sha256"]),
                                ("artifact_hashes", out, run["artifacts_sha256"])):
        for name, expected in hashes.items():
            check(file_hash(base / name) == expected, f"Changed {group}: {name}")
            evidence[group][name] = expected
    check(json.loads((out / "phase1a_config.json").read_text()) == run["config"], "Saved config mismatch")
    # Verify actual source text order and token IDs, not only their stored hashes.
    data = run["data"]
    parquet = Path(data["parquet"])
    check(file_hash(parquet) == data["parquet_sha256"], "Parquet changed")
    texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    text = "\n\n".join(texts)
    check(hashlib.sha256(text.encode("utf-8")).hexdigest() == data["text_utf8_sha256"], "Text join/hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(out / "tokenizer", local_files_only=True, trust_remote_code=False)
    tokens = np.asarray(tokenizer(text, add_special_tokens=False, padding=False, truncation=False)["input_ids"], dtype="<i8")
    check(hashlib.sha256(tokens.tobytes()).hexdigest() == data["token_stream_sha256_le_int64"], "Token stream mismatch")
    check(len(tokens) == data["total_tokens"] and len(tokens) // 2049 == data["full_blocks"] and
          len(tokens) % 2049 == data["dropped_tail_tokens"], "Block/tail counts mismatch")
    with np.load(out / "phase1a_tokens.npz", allow_pickle=False) as saved:
        indices = np.random.default_rng(42).choice(len(tokens) // 2049, 64, replace=False)
        check(np.array_equal(indices, saved["block_indices"]), "Sampling differs")
        for i, block_index in enumerate(indices):
            check(np.array_equal(tokens[block_index * 2049:(block_index + 1) * 2049], saved["token_ids"][i]), "Saved token IDs differ from text")
    evidence["retokenization"] = {"rows": len(texts), "tokens": len(tokens), "full_blocks": len(tokens) // 2049,
                                  "tail_tokens": len(tokens) % 2049, "all_64_blocks_match": True}
    # Verify original parameter hashes against the checkpoint and reconstruct every
    # distinct intervention via NumPy, independently of fake_quantize's PyTorch code.
    model = Path(run["model_source"])
    for name, expected in run["checkpoint_sha256"].items():
        check(file_hash(model / name) == expected, f"Checkpoint shard changed: {name}")
    check(file_hash(model / "config.json") == run["checkpoint_config_sha256"], "Checkpoint config changed")
    for name, expected in data["tokenizer_assets_sha256"].items():
        check(file_hash(Path(data["tokenizer_source"]) / name) == expected, f"Tokenizer source changed: {name}")
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    records = {}
    for intervention in run["interventions"]:
        for module in intervention["modules"]:
            records.setdefault(module["name"] + ".weight", []).append((int(intervention["condition"][2:]), module))
    checked_parameters, checked_quantizations, checked_module_records = 0, 0, 0
    torch.set_num_threads(4)
    for shard in sorted(set(index.values())):
        with safe_open(str(model / shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name not in run["original_state_sha256"]:
                    continue
                w = f.get_tensor(name)
                check(w.dtype == torch.bfloat16, f"Non-BF16 checkpoint parameter: {name}")
                check(digest(w) == run["original_state_sha256"][name], f"Runtime baseline differs from original: {name}")
                checked_parameters += 1
                by_bits = {}
                for bits, record in records.get(name, []):
                    if bits not in by_bits:
                        values = w.float().numpy()
                        groups = values.reshape(values.shape[0], values.shape[1] // 128, 128)
                        bound = np.float32(2 ** (bits - 1) - 1)
                        # PyTorch CUDA lowers division by a host scalar to FP32
                        # reciprocal multiplication. Direct NumPy division can
                        # differ by one scale ULP and flip half-step RTN cases.
                        # Match that recorded arithmetic, still require exact
                        # BF16 hashes for every module (never an allclose bypass).
                        scales = np.max(np.abs(groups), axis=2, keepdims=True) * np.float32(1.0 / float(bound))
                        integers = np.rint(groups / np.where(scales == 0, np.float32(1), scales))
                        quant = torch.from_numpy((np.clip(integers, -bound, bound) * scales).reshape(values.shape)).to(torch.bfloat16)
                        # Float64 norm is an audit reference; allow FP32 reduction roundoff.
                        diff = quant.float().numpy().astype(np.float64) - values
                        rel = np.sqrt(np.sum(diff ** 2) / np.sum(values.astype(np.float64) ** 2))
                        by_bits[bits] = (digest(quant), float(rel))
                        checked_quantizations += 1
                    quant_hash, rel = by_bits[bits]
                    check(quant_hash == record["quantized_sha256"], f"Independent RTN differs: {name} bit={bits}")
                    check(math.isclose(rel, record["relative_l2_error"], rel_tol=2e-4, abs_tol=1e-7), f"Relative L2 error differs: {name}")
                    checked_module_records += 1
    check(checked_parameters == 398, f"Expected 398 unique checkpoint parameters, got {checked_parameters}")
    check(checked_quantizations == 216 and checked_module_records == 252, "Incomplete RTN evidence audit")
    evidence["checkpoint_and_rtn"] = {"original_parameters_hashed": checked_parameters,
                                      "independent_quantizations_bitwise_equal": checked_quantizations,
                                      "module_records_checked": checked_module_records}
    # Independently resample the two arms using identical draws, rather than the
    # production difference-vector implementation; no GPU forward or new samples.
    analysis = json.loads((out / "phase1a_analysis.json").read_text())
    draws = np.random.default_rng(42).integers(0, 64, size=(2000, 64))
    check(np.array_equal(draws, np.load(out / "phase1a_bootstrap_indices.npy", allow_pickle=False)), "Bootstrap draw mismatch")
    with open(out / "phase1a.csv", newline="") as f:
        raw = list(csv.DictReader(f))
    for row in analysis["summary"]:
        subset = [r for r in raw if r["condition"] == row["condition"] and int(r["context"]) == row["context"]]
        check(len(subset) == 64, "Summary missing observations")
        mean = np.array([float(r["nll"]) for r in subset], dtype=np.float32).mean(dtype=np.float32)
        check(float(mean) == row["nll"] and math.exp(float(mean)) == row["ppl"], "Wrong aggregate NLL/PPL")
        for metric in ("delta_nll", "kl"):
            check(float(np.array([float(r[metric]) for r in subset], dtype=np.float32).mean()) == row[metric], f"Summary {metric} mismatch")
    for row in analysis["paired_comparisons"]:
        a, b = row["contrast"].split(" - ")
        x = np.array([keyed[a + str(row["bits"]), row["context"], s][row["metric"]] for s in range(64)])
        y = np.array([keyed[b + str(row["bits"]), row["context"], s][row["metric"]] for s in range(64)])
        distribution = x[draws].mean(axis=1) - y[draws].mean(axis=1)
        lo, hi = np.percentile(distribution, [2.5, 97.5])
        check(np.allclose([np.mean(x) - np.mean(y), lo, hi], [row["mean"], row["ci_low"], row["ci_high"]], atol=1e-12, rtol=1e-12), "Independent paired bootstrap mismatch")
    check(len(analysis["summary"]) == 14 and len(analysis["paired_comparisons"]) == 24, "Incomplete statistics")
    evidence["independent_statistics"] = {"summary_cells": 14, "paired_contrasts": 24, "draws": [2000, 64], "passed": True}
    # Recorded pre-formal smoke is a reproducibility control, never a new sample.
    if args.compare_smoke is not None:
        smoke = json.loads((args.compare_smoke / "phase1a_run.json").read_text())
        check(smoke["config"] == run["config"], "Pre-formal smoke config differs")
        check(smoke["smoke"]["records"] == run["smoke"]["records"], "Deterministic smoke did not reproduce")
        for r in smoke["smoke"]["records"]:
            check(keyed["BF16", r["context"], 0]["nll"] == r["baseline_nll"] and
                  keyed["WQ4", r["context"], 0]["nll"] == r["wq4_nll"] and
                  keyed["WQ4", r["context"], 0]["kl"] == r["wq4_kl"], "Formal rows differ from smoke")
        evidence["separate_smoke_reproduced_exactly"] = True
    evidence["status"] = "PASS: artifact audit; see phase1a_audit.md for requirement-to-evidence mapping"
    (out / "phase1a_verification.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in evidence.items() if not k.endswith("hashes")}, indent=2))


if __name__ == "__main__":
    main()
