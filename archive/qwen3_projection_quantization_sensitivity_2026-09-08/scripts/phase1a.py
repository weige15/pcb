#!/usr/bin/env python3
"""Phase 1a only: grouped RTN projection sensitivity, no packed low-bit kernels."""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["BF16", "WQ8", "WQ4", "WK8", "WK4", "WV8", "WV4"]
CONTEXTS = [512, 2048]
PROJECTIONS = {"WQ": "q_proj", "WK": "k_proj", "WV": "v_proj"}
FIELDS = ["condition", "projection", "bits", "context", "sample_id", "block_index",
          "n_tokens", "input_sha256", "labels_sha256", "nll", "delta_nll", "kl"]


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def tensor_hash(tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def model_hashes(model):
    # Include non-persistent buffers as well as every unique parameter. Tied head
    # shares embed_tokens storage; its identity is checked separately at load time.
    return {name: tensor_hash(t) for name, t in
            list(model.named_parameters()) + list(model.named_buffers())}


def validate_config(cfg):
    fixed = {"model_id": "Qwen/Qwen3-4B", "dataset_id": "Salesforce/wikitext",
             "dataset_config": "wikitext-2-raw-v1", "split": "test",
             "dtype": "bfloat16", "attention_backend": "sdpa", "sdpa_kernel": "FLASH_ATTENTION",
             "batch_size": 1, "use_cache": False, "group_size": 128,
             "conditions": CONDITIONS, "contexts": CONTEXTS, "block_tokens": 2049,
             "sample_blocks": 64, "seed": 42, "metric_chunk_positions": 64,
             "bootstrap_replicates": 2000, "bootstrap_seed": 42}
    for key, expected in fixed.items():
        require(cfg[key] == expected, f"Phase 1a fixed setting {key}: expected {expected!r}")
    for key in ("model_revision", "tokenizer_revision", "dataset_revision"):
        require(len(cfg[key]) == 40 and all(c in "0123456789abcdef" for c in cfg[key]),
                f"Pin {key} to a full commit hash")


def sample_blocks(tokens, cfg):
    n_full = len(tokens) // cfg["block_tokens"]
    require(n_full >= cfg["sample_blocks"],
            f"DATA BLOCKED: only {n_full} full blocks; 64 required (no sample reduction)")
    indices = np.random.default_rng(cfg["seed"]).choice(n_full, cfg["sample_blocks"], replace=False)
    stream = np.asarray(tokens[:n_full * cfg["block_tokens"]], dtype=np.int64)
    return indices, stream.reshape(n_full, cfg["block_tokens"])[indices].copy(), n_full


def prepare_data(cfg, out, manifest):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq

    # Read the exact official cached test parquet directly, preserving its row order.
    # No datasets-script execution, network access, or implicit revision resolution.
    model_path = Path(snapshot_download(cfg["model_id"], revision=cfg["model_revision"], local_files_only=True,
                                        allow_patterns=["config.json", "generation_config.json", "model*.safetensors", "model.safetensors.index.json"]))
    tok_path = Path(snapshot_download(cfg["model_id"], revision=cfg["tokenizer_revision"], local_files_only=True,
                                      allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]))
    data_path = Path(snapshot_download(cfg["dataset_id"], repo_type="dataset",
                                       revision=cfg["dataset_revision"], local_files_only=True,
                                       allow_patterns=[cfg["dataset_config"] + "/test-00000-of-00001.parquet"]))
    parquet = data_path / cfg["dataset_config"] / "test-00000-of-00001.parquet"
    table = pq.read_table(parquet, columns=["text"])
    require(table.num_rows > 0 and table.column("text").null_count == 0, "Missing/null test text")
    text = "\n\n".join(table.column("text").to_pylist())
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True, trust_remote_code=False)
    ids = tokenizer(text, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
    indices, blocks, n_full = sample_blocks(ids, cfg)
    np.savez_compressed(out / "phase1a_tokens.npz", block_indices=indices, token_ids=blocks)
    tokenizer.save_pretrained(out / "tokenizer")
    manifest["data"] = {
        "parquet": str(parquet), "parquet_sha256": file_hash(parquet), "rows": table.num_rows,
        "join": "two newlines between every row, including empty rows; original test order",
        "text_utf8_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_stream_sha256_le_int64": token_hash(ids), "total_tokens": len(ids),
        "full_blocks": n_full, "dropped_tail_tokens": len(ids) % cfg["block_tokens"],
        "sample_indices": indices.tolist(), "sample_token_ids": "phase1a_tokens.npz",
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_source": str(tok_path),
        "tokenizer_call": {"add_special_tokens": False, "truncation": False, "padding": False},
        "tokenizer_settings": "tokenizer/tokenizer_config.json",
        "tokenizer_assets_sha256": {p.name: file_hash(p) for p in sorted(tok_path.iterdir())
                                    if p.name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")},
    }
    manifest["model_source"] = str(model_path)
    manifest["checkpoint_sha256"] = {p.name: file_hash(p) for p in sorted(model_path.glob("*.safetensors"))}
    manifest["checkpoint_config_sha256"] = file_hash(model_path / "config.json")
    return model_path, indices, blocks


def fake_quantize(weight, bits, group_size=128):
    require(bits in (4, 8), "Only 4/8-bit interventions")
    require(weight.ndim == 2 and weight.shape[1] % group_size == 0, "Groups must not cross output rows")
    require(weight.dtype == torch.bfloat16 and bool(torch.isfinite(weight).all()), "Finite BF16 weight required")
    groups = weight.float().reshape(weight.shape[0], -1, group_size)
    qmax = 2 ** (bits - 1) - 1
    scale = groups.abs().amax(dim=-1, keepdim=True) / qmax
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # torch.round uses nearest, ties to even; zero groups remain exactly zero.
    dequant = (groups / safe_scale).round().clamp(-qmax, qmax) * scale
    return dequant.reshape_as(weight).to(torch.bfloat16)


def targets_for(model, projection):
    suffix = PROJECTIONS[projection]
    expected = {f"model.layers.{i}.self_attn.{suffix}" for i in range(36)}
    modules = {name: m for name, m in model.named_modules() if name.endswith("." + suffix)}
    require(set(modules) == expected, f"Unexpected {projection} module set")
    rows = 4096 if projection == "WQ" else 1024
    require(all(tuple(m.weight.shape) == (rows, 2560) for m in modules.values()), "Unexpected projection shape")
    return modules


@contextmanager
def intervention(model, projection, bits, original_hashes, manifest, out, stage):
    modules = targets_for(model, projection)
    original = {name: m.weight.detach().cpu().clone() for name, m in modules.items()}
    require(model_hashes(model) == original_hashes, "Condition did not start from original checkpoint state")
    record = {"stage": stage, "condition": projection + str(bits), "modules": [], "restored": False}
    manifest.setdefault("interventions", []).append(record)
    try:
        with torch.no_grad():
            for name, module in modules.items():
                w = module.weight
                quant = fake_quantize(w, bits)
                rel = ((quant.float() - w.float()).norm() / w.float().norm()).item()
                record["modules"].append({"name": name, "shape": list(w.shape), "parameters": w.numel(),
                                          "relative_l2_error": rel, "original_sha256": tensor_hash(w),
                                          "quantized_sha256": tensor_hash(quant)})
                w.copy_(quant)
        applied = model_hashes(model)
        changed = {k for k in applied if applied[k] != original_hashes[k]}
        expected = {name + ".weight" for name in modules}
        require(changed == expected, f"Unexpected changed tensors: {changed ^ expected}")
        record.update(changed_tensors=sorted(changed), non_target_unchanged=True,
                      parameters=sum(m["parameters"] for m in record["modules"]))
        save_json(out / "phase1a_run.json", manifest)
        yield
        require(model_hashes(model) == applied, "Weights/buffers changed during eval")
        record["unchanged_during_eval"] = True
    finally:
        with torch.no_grad():
            for name, module in modules.items():
                module.weight.copy_(original[name])
        require(model_hashes(model) == original_hashes, "Failed to restore original state")
        record["restored"] = True
        save_json(out / "phase1a_run.json", manifest)


def hidden_for(model, ids):
    require(not model.training and ids.shape[0] == 1, "eval mode, batch size 1 required")
    result = model.model(input_ids=ids, use_cache=False, return_dict=True)
    require(result.past_key_values is None, "KV cache must be disabled")
    hidden = result.last_hidden_state[:, :-1, :]
    require(hidden.dtype == torch.bfloat16 and bool(torch.isfinite(hidden).all()), "Invalid hidden states")
    require(hidden.shape[1] == ids.shape[1] - 1, "Shifted scoring alignment failure")
    return hidden


def metric_from_logits(logits, reference_logits, labels):
    require(logits.shape == reference_logits.shape and logits.shape[:-1] == labels.shape,
            "Logits/labels alignment mismatch")
    require(bool(torch.isfinite(logits).all() & torch.isfinite(reference_logits).all()), "Nonfinite logits")
    log_test = F.log_softmax(logits.float(), dim=-1)
    log_ref = F.log_softmax(reference_logits.float(), dim=-1)
    nll_sum = -log_test.gather(-1, labels.unsqueeze(-1)).sum(dtype=torch.float32)
    kl_sum = (log_ref.exp() * (log_ref - log_test)).sum(dtype=torch.float32)
    require(bool(torch.isfinite(nll_sum) & torch.isfinite(kl_sum)), "Nonfinite metrics")
    return nll_sum, kl_sum


def score(head, hidden, reference_hidden, labels, chunk_size):
    require(hidden.shape == reference_hidden.shape and hidden.shape[:2] == labels.shape,
            "Hidden/labels alignment mismatch")
    nll = torch.zeros((), device=hidden.device, dtype=torch.float32)
    kl = torch.zeros_like(nll)
    for start in range(0, labels.shape[1], chunk_size):
        stop = min(start + chunk_size, labels.shape[1])
        # Keep only two chunks of logits, never all blocks' full distributions.
        logits = head(hidden[:, start:stop])
        ref_logits = head(reference_hidden[:, start:stop].to(hidden.device))
        ns, ks = metric_from_logits(logits, ref_logits, labels[:, start:stop])
        nll += ns
        kl += ks
    result = (nll / labels.numel()).item(), (kl / labels.numel()).item()
    require(all(math.isfinite(x) for x in result) and result[1] >= -1e-6, "Invalid NLL/KL")
    # Preserve tiny negative roundoff instead of biasing the measurement by clipping.
    return result


def smoke(model, blocks, cfg, original_hashes, manifest, out):
    ref = {}
    records = []
    for context in CONTEXTS:
        ids = torch.tensor(blocks[0, :context + 1], device="cuda").unsqueeze(0)
        hidden = hidden_for(model, ids)
        metrics = score(model.lm_head, hidden, hidden, ids[:, 1:], cfg["metric_chunk_positions"])
        # Compare first and last scoring chunks against the actual CausalLM API.
        api_checks = []
        for start in (0, context - cfg["metric_chunk_positions"]):
            positions = torch.arange(start, start + cfg["metric_chunk_positions"], device="cuda")
            actual = model(input_ids=ids, use_cache=False, logits_to_keep=positions, return_dict=True)
            manual = model.lm_head(hidden[:, start:start + cfg["metric_chunk_positions"]])
            require(actual.past_key_values is None and torch.equal(actual.logits, manual), "Chunked head differs from CausalLM API")
            labels = ids[:, 1:][:, start:start + cfg["metric_chunk_positions"]]
            direct = F.cross_entropy(actual.logits.float().reshape(-1, actual.logits.shape[-1]), labels.reshape(-1), reduction="sum")
            ns, ks = metric_from_logits(manual, actual.logits, labels)
            require(torch.allclose(ns, direct, atol=1e-4, rtol=1e-6) and ks.item() == 0, "NLL/shift control failed")
            api_checks.append({"start": start, "positions": len(positions), "logits_bitwise_equal": True,
                               "cross_entropy_sum": direct.item(), "metric_nll_sum": ns.item()})
        ref[context] = (hidden.cpu(), metrics)
        records.append({"context": context, "sample_id": 0, "baseline_nll": metrics[0],
                        "baseline_self_kl": metrics[1], "causal_lm_api_checks": api_checks})
    with intervention(model, "WQ", 4, original_hashes, manifest, out, "smoke"):
        for record in records:
            context = record["context"]
            ids = torch.tensor(blocks[0, :context + 1], device="cuda").unsqueeze(0)
            hidden = hidden_for(model, ids)
            nll, kl = score(model.lm_head, hidden, ref[context][0], ids[:, 1:], cfg["metric_chunk_positions"])
            record.update(wq4_nll=nll, wq4_kl=kl, scored_positions=context, finite=True)
    for record in records:
        context = record["context"]
        ids = torch.tensor(blocks[0, :context + 1], device="cuda").unsqueeze(0)
        hidden = hidden_for(model, ids)
        require(torch.equal(hidden.cpu(), ref[context][0]), "Restored baseline hidden states differ")
        restored = score(model.lm_head, hidden, ref[context][0], ids[:, 1:], cfg["metric_chunk_positions"])
        require(restored == ref[context][1] and restored[1] == 0, "Restored baseline metrics differ")
        record.update(restored_hidden_bitwise_equal=True, restored_nll=restored[0], restored_kl=restored[1])
    manifest["smoke"] = {"passed": True, "records": records, "finished_utc": now(),
                         "note": "one selected block, both contexts; controls only, not independent samples"}
    save_json(out / "phase1a_run.json", manifest)
    print("SMOKE PASS: BF16/WQ4, intervention hashes, CausalLM alignment, restored baseline", flush=True)


def formal(model, indices, blocks, cfg, original_hashes, manifest, out):
    reference = {}
    baseline_nll = {}
    with open(out / "phase1a.csv", "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        def evaluate(condition):
            for context in CONTEXTS:
                start = time.monotonic()
                for sample_id, block_index in enumerate(indices):
                    tokens = blocks[sample_id, :context + 1]
                    ids = torch.tensor(tokens, device="cuda").unsqueeze(0)
                    hidden = hidden_for(model, ids)
                    key = (context, sample_id)
                    if condition == "BF16":
                        reference[key] = hidden.cpu()
                    nll, kl = score(model.lm_head, hidden, reference[key], ids[:, 1:], cfg["metric_chunk_positions"])
                    if condition == "BF16":
                        baseline_nll[key] = nll
                    writer.writerow({"condition": condition, "projection": "none" if condition == "BF16" else condition[:2],
                                     "bits": 16 if condition == "BF16" else int(condition[2:]),
                                     "context": context, "sample_id": sample_id, "block_index": int(block_index),
                                     "n_tokens": context, "input_sha256": token_hash(tokens), "labels_sha256": token_hash(tokens[1:]),
                                     "nll": nll, "delta_nll": float(np.float32(nll) - np.float32(baseline_nll[key])), "kl": kl})
                    f.flush()
                    if (sample_id + 1) % 8 == 0:
                        print(f"{now()} {condition} ctx={context} blocks={sample_id + 1}/64 elapsed={time.monotonic() - start:.1f}s", flush=True)
                manifest["completed_conditions"].append({"condition": condition, "context": context, "blocks": 64,
                                                          "scored_tokens": 64 * context, "seconds": time.monotonic() - start})
                save_json(out / "phase1a_run.json", manifest)

        evaluate("BF16")
        require(model_hashes(model) == original_hashes, "Baseline eval changed model state")
        for condition in CONDITIONS[1:]:
            with intervention(model, condition[:2], int(condition[2:]), original_hashes, manifest, out, "formal"):
                evaluate(condition)
    manifest["final_state_matches_checkpoint"] = model_hashes(model) == original_hashes
    require(manifest["final_state_matches_checkpoint"], "Final model state mismatch")


def paired_bootstrap(a, b, draws):
    # A single shared block-index resample is applied to both arms, never two
    # independent resamples. Float64 here is statistical analysis of FP32 metrics.
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    means = difference[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975], method="linear")
    return float(difference.mean()), float(lo), float(hi)


def read_and_validate_rows(out, manifest):
    with np.load(out / "phase1a_tokens.npz", allow_pickle=False) as data:
        indices, blocks = data["block_indices"], data["token_ids"]
    require(blocks.shape == (64, 2049) and indices.shape == (64,) and len(set(indices)) == 64, "Invalid saved samples")
    require(indices.tolist() == manifest["data"]["sample_indices"], "Sample index mismatch")
    expected_indices = np.random.default_rng(42).choice(manifest["data"]["full_blocks"], 64, replace=False)
    require(np.array_equal(indices, expected_indices), "Sampling seed/order mismatch")
    with open(out / "phase1a.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 896, f"Expected 896 rows, got {len(rows)}")
    keyed = {}
    for row in rows:
        for k in ("bits", "context", "sample_id", "block_index", "n_tokens"):
            row[k] = int(row[k])
        for k in ("nll", "delta_nll", "kl"):
            row[k] = float(row[k])
        c, ctx, s = row["condition"], row["context"], row["sample_id"]
        require(c in CONDITIONS and ctx in CONTEXTS and 0 <= s < 64, "Unknown condition/sample")
        key = (c, ctx, s)
        require(key not in keyed, f"Duplicate row {key}")
        keyed[key] = row
        require(row["projection"] == ("none" if c == "BF16" else c[:2]) and row["bits"] == (16 if c == "BF16" else int(c[2:])), "Condition identity mismatch")
        require(row["block_index"] == indices[s] and row["n_tokens"] == ctx, "Scoring identity mismatch")
        require(row["input_sha256"] == token_hash(blocks[s, :ctx + 1]) and row["labels_sha256"] == token_hash(blocks[s, 1:ctx + 1]), "Input/labels mismatch")
        require(all(math.isfinite(row[k]) for k in ("nll", "delta_nll", "kl")) and row["kl"] >= -1e-6, "Invalid metrics")
    for (c, ctx, s), row in keyed.items():
        base = keyed[("BF16", ctx, s)]
        require(row["delta_nll"] == float(np.float32(row["nll"]) - np.float32(base["nll"])), "Delta NLL mismatch")
        if c == "BF16":
            require(row["delta_nll"] == 0 and row["kl"] == 0, "Baseline comparison not zero")
    return keyed


def validate_controls(manifest):
    require(manifest.get("status") in ("running", "measured"), "Blocked/incomplete run cannot produce a results report")
    validate_config(manifest["config"])
    completed = manifest.get("completed_conditions", [])
    require(len(completed) == 14 and {(r["condition"], r["context"]) for r in completed} ==
            {(c, ctx) for c in CONDITIONS for ctx in CONTEXTS}, "Missing/duplicate completed conditions")
    require(all(r["blocks"] == 64 and r["scored_tokens"] == 64 * r["context"] for r in completed), "Incomplete scoring coverage")
    smoke_result = manifest.get("smoke", {})
    require(smoke_result.get("passed") is True, "Smoke controls missing/failed")
    records = smoke_result.get("records", [])
    require(len(records) == 2 and {r["context"] for r in records} == set(CONTEXTS), "Incomplete smoke contexts")
    for r in records:
        require(r["sample_id"] == 0 and r["scored_positions"] == r["context"] and r["finite"] is True,
                "Smoke scoring coverage failed")
        require(r["restored_hidden_bitwise_equal"] is True and r["restored_nll"] == r["baseline_nll"] and
                r["baseline_self_kl"] == r["restored_kl"] == 0, "Restoration controls failed")
        require(all(math.isfinite(r[k]) for k in ("baseline_nll", "wq4_nll", "wq4_kl", "restored_nll")), "Invalid smoke metrics")
        checks = r["causal_lm_api_checks"]
        require(len(checks) == 2 and {v["start"] for v in checks} == {0, r["context"] - 64} and
                all(v["positions"] == 64 and v["logits_bitwise_equal"] is True and
                    math.isclose(v["cross_entropy_sum"], v["metric_nll_sum"], rel_tol=1e-6, abs_tol=1e-4) for v in checks),
                "CausalLM alignment checks missing/failed")
    records = manifest.get("interventions", [])
    require(len(records) == 7 and {(r["stage"], r["condition"]) for r in records} ==
            {("smoke", "WQ4")} | {("formal", c) for c in CONDITIONS[1:]}, "Missing intervention controls")
    for r in records:
        projection = r["condition"][:2]
        expected = {f"model.layers.{i}.self_attn.{PROJECTIONS[projection]}" for i in range(36)}
        width = 4096 if projection == "WQ" else 1024
        require(all(r.get(k) is True for k in ("non_target_unchanged", "unchanged_during_eval", "restored")), "Intervention control failed")
        require(len(r["modules"]) == 36 and {m["name"] for m in r["modules"]} == expected and
                set(r["changed_tensors"]) == {n + ".weight" for n in expected}, "Incorrect intervention module inventory")
        require(r["parameters"] == 36 * width * 2560, "Parameter count mismatch")
        for m in r["modules"]:
            require(m["shape"] == [width, 2560] and m["parameters"] == width * 2560 and
                    math.isfinite(m["relative_l2_error"]) and m["relative_l2_error"] > 0 and
                    m["original_sha256"] == manifest["original_state_sha256"][m["name"] + ".weight"] and
                    m["quantized_sha256"] != m["original_sha256"], "Invalid modified-module evidence")
    require(manifest.get("final_state_matches_checkpoint") is True, "Final checkpoint restoration not verified")


def analyze(out, manifest):
    validate_controls(manifest)
    rows = read_and_validate_rows(out, manifest)
    draws = np.random.default_rng(42).integers(0, 64, size=(2000, 64))
    np.save(out / "phase1a_bootstrap_indices.npy", draws)
    summary = []
    paired = []
    for ctx in CONTEXTS:
        for c in CONDITIONS:
            subset = [rows[c, ctx, s] for s in range(64)]
            values = {k: np.array([r[k] for r in subset], dtype=np.float32) for k in ("nll", "delta_nll", "kl")}
            nll = float(values["nll"].mean(dtype=np.float32))
            summary.append({"context": ctx, "condition": c, "tokens": ctx * 64, "nll": nll,
                            "ppl": math.exp(nll), "delta_nll": float(values["delta_nll"].mean(dtype=np.float32)),
                            "kl": float(values["kl"].mean(dtype=np.float32))})
        for bits in (8, 4):
            for a, b in (("WQ", "WK"), ("WQ", "WV"), ("WK", "WV")):
                for metric in ("delta_nll", "kl"):
                    x = [rows[a + str(bits), ctx, s][metric] for s in range(64)]
                    y = [rows[b + str(bits), ctx, s][metric] for s in range(64)]
                    mean, lo, hi = paired_bootstrap(x, y, draws)
                    paired.append({"context": ctx, "bits": bits, "contrast": a + " - " + b,
                                   "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi})
    save_json(out / "phase1a_analysis.json", {"summary": summary, "paired_comparisons": paired,
                                            "bootstrap": {"seed": 42, "replicates": 2000, "unit": "selected block", "ci": "percentile 95%, linear quantiles"}})
    lines = ["# Phase 1a — Qwen3-4B projection-weight sensitivity", "",
             "## 結果與方法", "", "14/14 條件、896 筆逐 block 實測；每條件 64 個 blocks。單位：natural-log nats/token。",
             "BF16 原始 Qwen/Qwen3-4B；WikiText-2 raw test；同一批 seed 42 非重疊 2049-token blocks，分別取前 513/2049 tokens。",
             "Transformers eval、batch=1、SDPA（固定 PyTorch FLASH_ATTENTION kernel）、use_cache=False；純文字 teacher forcing。",
             "所有層一次只改 Q/K/V 中一種 weight；每 output row 以 input 維分組 128、FP32 scale、symmetric RTN、dequant 回 BF16。",
             "NLL/KL 的 log-softmax 與 reduction 為 FP32；保存 baseline 最終 BF16 hidden states 於 CPU，透過未修改的 lm_head 按 64 位置重建兩組 logits 計分，不囤完整 logits。",
             "PPL = exp(所有計分 token 的平均 NLL)；各 block 計分長度一致，故等同 block NLL 均值，不是 block PPL 均值。", "",
             "| Context（預測數） | 條件 | NLL | PPL | ΔNLL vs BF16 | KL(BF16 || test) |",
             "|---:|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['context']} | {row['condition']} | {row['nll']:.8f} | {row['ppl']:.6f} | {row['delta_nll']:+.8f} | {row['kl']:.8f} |")
    lines += ["", "## Q/K/V paired 比較（探索性 95% CI）", "",
              "相同 block 索引配對，2000 次 bootstrap、seed 42；所有比較共用保存的 draws，percentile CI。",
              "差值方向為左項減右項；ΔNLL 差等於 NLL 差。未做多重比較校正，不能作確認性顯著性宣稱。", "",
              "| Context | bit | 差值 | metric | mean | 95% CI |", "|---:|---:|---|---|---:|---|"]
    for row in paired:
        lines.append(f"| {row['context']} | {row['bits']} | {row['contrast']} | {row['metric']} | {row['mean']:+.8f} | [{row['ci_low']:+.8f}, {row['ci_high']:+.8f}] |")
    lines += ["", "## 描述性結論", ""]
    for ctx in CONTEXTS:
        for bits in (8, 4):
            group = [r for r in summary if r["context"] == ctx and r["condition"] in [p + str(bits) for p in PROJECTIONS]]
            for metric in ("delta_nll", "kl"):
                ranked = sorted(group, key=lambda r: r[metric], reverse=True)
                lines.append(f"- context={ctx}, {bits}-bit，{metric} 由大到小：" + " > ".join(f"{r['condition']} ({r[metric]:.8f})" for r in ranked) + "。這只是本次均值排序，請合併 paired CI 判讀。")
    crossing = sum(r["ci_low"] <= 0 <= r["ci_high"] for r in paired)
    lines += [f"- {len(paired)} 個探索性 paired CI 中 {crossing} 個跨零；跨零不是已證明等價。接受沒有明顯差異，不挑選或加跑條件追求排序。", "",
              "## 控制與可重跑證據", "",
              "- 正式跑前以一個已選 block 的兩個 contexts 跑 BF16/WQ4；控制結果保存在 `phase1a_run.json:smoke`。不把控制或 deterministic 重跑當獨立樣本。",
              "- 第一/最後 64 個 scoring positions 的 chunked lm_head logits 與 CausalLM API bitwise 一致；shifted NLL 與直接 cross entropy 核對。",
              "- 每次介入核對所有 unique parameters 與 buffers 的 SHA256，只改 36 個目標 weights；評估前後不變，還原後全模型 state hash 回 baseline。",
              "- 小樣本還原後 hidden states bitwise 一致、NLL 相同、KL=0；正式末態 hash 回原始 checkpoint。所有逐 block logits/metrics 檢查有限。",
              "- 詳細 module 名稱、shape、參數量、相對 L2 誤差在 `phase1a_run.json:interventions`；Q 每條件 377,487,360 個參數，K/V 各 94,371,840。",
              "- 固定設定與實際命令：`phase1a_run.json`；原始逐 block：`phase1a.csv`；抽樣索引/token IDs：`phase1a_tokens.npz`；tokenizer 設定：`tokenizer/`。",
              "- 統計可獨立重算：`python scripts/phase1a.py --analyze-only --output results`；bootstrap draws：`phase1a_bootstrap_indices.npy`；未四捨五入摘要/CI：`phase1a_analysis.json`。",
              "", "```bash", manifest["rerun_command"], "```", "",
              "## 限制", "",
              "- 結論只限此 Qwen3-4B checkpoint、WikiText test 抽樣、RTN 規則與兩個 contexts。不是跨模型/資料/seed 的推論。",
              "- Qwen3-4B 是 GQA（Q 32 heads、KV 8 heads）；Q 介入參數量為 K/V 的四倍，是原生端到端敏感度，不是等參數量/等成本比較。",
              "- fake quantization 後仍 BF16 GEMM；沒有 packed low-bit GEMM，不宣稱加速或實際記憶體壓縮。",
              "- activation、其他權重與運算設定未量化；沒有啟用 KV cache，不從本實驗推論 KV lifetime、prefill/decode 差別或 serving 效益。",
              "- 相鄰原文 blocks 可能相關；block bootstrap 只描述此有限樣本下的探索性變異，不保證獨立性或人口層級涵蓋率；多重比較未校正。",
              "- BF16 前向與 FP32 reduction 有數值誤差；小 KL 與均值排序應審慎解讀。統計運算使用保存的 FP32 measurements，不把 deterministic 重跑算獨立樣本。",
              "- 未開始 activation/KV cache 或下一階段；不以任一排序作下一階段已成立的證據。", ""]
    (out / "phase1a_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/phase1a_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--smoke-only", action="store_true", help="Controls only; never represents a completed experiment")
    parser.add_argument("--analyze-only", action="store_true", help="Recompute statistics from saved measurements; no model forward")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        manifest = json.loads((out / "phase1a_run.json").read_text())
        analyze(out, manifest)
        print("ANALYSIS PASS: validated 896 rows and recomputed paired CIs")
        return
    # Never silently overwrite prior measurements, including an interrupted run.
    require(not (out / "phase1a_run.json").exists() and not (out / "phase1a.csv").exists(),
            "Output already contains a run. Choose a new --output; do not overwrite evidence.")
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    command = shlex.join([sys.executable, *sys.argv])
    save_json(out / "phase1a_config.json", cfg)
    rerun = (f"CUDA_VISIBLE_DEVICES={shlex.quote(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))} "
             "bash scripts/run_phase1a.sh results/rerun_$(date -u +%Y%m%dT%H%M%SZ) "
             f"--config {shlex.quote(str(out / 'phase1a_config.json'))}" + (" --smoke-only" if args.smoke_only else ""))
    manifest = {"status": "starting", "started_utc": now(), "config": cfg, "command": command,
                "cwd": str(Path.cwd()), "completed_conditions": [], "interventions": [],
                "rerun_command": rerun,
                "source_sha256": {os.path.relpath(p, ROOT): file_hash(p) for p in
                                  [ROOT / "scripts/phase1a.py", ROOT / "scripts/run_phase1a.sh", args.config.resolve(), ROOT / "phase1a.md"]},
                "environment": {k: os.environ.get(k) for k in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "LD_LIBRARY_PATH")},
                "host": platform.node(), "platform": platform.platform(), "python": sys.version,
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "huggingface-hub", "tokenizers", "numpy", "safetensors", "pyarrow", "accelerate", "datasets")}}
    save_json(out / "phase1a_run.json", manifest)
    try:
        manifest["gpu_inventory_before"] = subprocess.check_output(["nvidia-smi"], text=True)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one selected usable CUDA GPU required")
        require(torch.cuda.is_bf16_supported(), "BF16 CUDA required")
        manifest["cuda"] = {"build": torch.version.cuda, "device": str(torch.cuda.get_device_properties(0)),
                            "free_total_bytes": list(torch.cuda.mem_get_info())}
        torch.manual_seed(42)
        np.random.seed(42)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        manifest["numerics"] = {"deterministic_algorithms": True, "tf32": False, "cpu_threads": 4,
                                "metric_dtype": "float32", "statistical_bootstrap_dtype": "float64",
                                "quantization": "symmetric round-nearest ties-even, FP32 maxabs/qmax per output-row group128, zero groups stay zero, BF16 dequant",
                                "logits_storage": "two 64-position chunks; baseline final hidden states retained losslessly as BF16 on CPU"}
        model_path, indices, blocks = prepare_data(cfg, out, manifest)
        save_json(out / "phase1a_run.json", manifest)
        from transformers import AutoModelForCausalLM
        from torch.nn.attention import SDPBackend, sdpa_kernel
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0",
                                                    attn_implementation=cfg["attention_backend"], local_files_only=True, trust_remote_code=False)
        model.eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        require(all(p.dtype == torch.bfloat16 for p in model.parameters()), "All original parameters must be BF16")
        require(model.config.num_attention_heads == 32 and model.config.num_key_value_heads == 8 and len(model.model.layers) == 36, "Wrong GQA/model structure")
        require(model.lm_head.weight is model.model.embed_tokens.weight, "Unexpected lm_head tying")
        manifest["runtime_model_config"] = model.config.to_dict()
        manifest["runtime_attention_backend"] = model.config._attn_implementation
        manifest["tied_lm_head"] = True
        for p in PROJECTIONS:
            targets_for(model, p)
        original_hashes = model_hashes(model)
        manifest["original_state_sha256"] = original_hashes
        manifest["status"] = "smoke"
        save_json(out / "phase1a_run.json", manifest)
        # Disable all other SDPA kernels: no implicit backend fallback per length.
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            smoke(model, blocks, cfg, original_hashes, manifest, out)
            if args.smoke_only:
                manifest["status"] = "smoke_only_not_complete"
            else:
                manifest["status"] = "running"
                formal(model, indices, blocks, cfg, original_hashes, manifest, out)
                analyze(out, manifest)
                manifest["status"] = "measured"
        manifest["finished_utc"] = now()
        manifest["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        artifacts = [out / name for name in ("phase1a_config.json", "phase1a_tokens.npz", "phase1a.csv",
                                             "phase1a_bootstrap_indices.npy", "phase1a_analysis.json", "phase1a_report.md")]
        artifacts += sorted((out / "tokenizer").iterdir())
        manifest["artifacts_sha256"] = {str(p.relative_to(out)): file_hash(p) for p in artifacts if p.is_file()}
        save_json(out / "phase1a_run.json", manifest)
        print(f"FINISHED status={manifest['status']} conditions={len(manifest['completed_conditions'])}/14 output={out}", flush=True)
    except Exception:
        error = traceback.format_exc()
        manifest.update(status="blocked", error=error, failed_utc=now())
        save_json(out / "phase1a_run.json", manifest)
        path = out / "phase1a_blocked.md"
        with open(path, "a") as f:
            f.write(f"\n\n## {now()} — 本輪阻塞，未完成\n\nCommand: `{command}`\n\n"
                    f"Completed conditions: {len(manifest['completed_conditions'])}/14. See phase1a_run.json for exact completed cells.\n\n"
                    f"```text\n{error}\n```\n\n未完成項：其餘條件、未通過的控制、完整 NLL/KL/CI 與完成稽核。保留已有數據；不自動重試、換模型、減少樣本或擴大範圍。"
                    "若缺 GPU/checkpoint/data 或 OOM，請使用 `/goal pause`，提供可用資源後通知繼續。\n")
        raise


if __name__ == "__main__":
    main()
