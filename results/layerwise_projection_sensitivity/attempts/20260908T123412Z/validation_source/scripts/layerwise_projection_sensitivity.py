#!/usr/bin/env python3
"""Fixed single-layer WQ/WK/WV RTN4 experiment; atomic, verified condition resume."""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.projection_quantization_sensitivity import (
    PROJECTIONS, fake_quantize, file_hash, hidden_for, metric_from_logits,
    model_hashes, now, require, save_json, score, targets_for, tensor_hash, token_hash,
)

CONFIG = ROOT / "scripts/layerwise_projection_sensitivity_config.json"
OUTPUT = ROOT / "results/layerwise_projection_sensitivity"
CONDITIONS = ["BF16"] + [f"L{i:02d}_{p}4" for i in range(36) for p in PROJECTIONS]
FIELDS = ["condition", "layer", "projection", "bits", "context", "sample_id", "block_index",
          "n_tokens", "input_sha256", "labels_sha256", "nll", "delta_nll", "kl"]
SOURCES = ["scripts/layerwise_projection_sensitivity.py", "scripts/projection_quantization_sensitivity.py",
           "scripts/layerwise_projection_sensitivity_config.json", "scripts/run_layerwise_projection_sensitivity.sh"]


def identity(condition):
    require(condition in CONDITIONS, f"Unknown condition {condition}")
    return (-1, "none", 16) if condition == "BF16" else (int(condition[1:3]), condition[4:6], 4)


def load_samples(out):
    with np.load(out / "sampled_token_blocks.npz", allow_pickle=False) as f:
        indices, blocks = f["block_indices"], f["token_ids"]
    require(indices.shape == (64,) and blocks.shape == (64, 2049), "64 full blocks required")
    require(indices.dtype == blocks.dtype == np.int64 and len(set(indices)) == 64, "Invalid samples")
    return indices, blocks


def state_tensors(model):
    return dict(list(model.named_parameters()) + list(model.named_buffers()))


def changed_state(model, original, replacement=None):
    """Exact full-state comparison on device, not tolerance-based sampling."""
    current = state_tensors(model)
    require(current.keys() == original.keys(), "State inventory changed")
    changed = []
    for name, value in current.items():
        expected = replacement[1] if replacement and name == replacement[0] else original[name]
        require(value.dtype == expected.dtype and value.shape == expected.shape, f"State layout changed: {name}")
        if not torch.equal(value.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)):
            changed.append(name)
    return changed


@contextmanager
def single_intervention(model, condition, original, original_hashes, control):
    layer, projection, bits = identity(condition)
    require(layer >= 0, "BF16 is not an intervention")
    name = f"model.layers.{layer}.self_attn.{PROJECTIONS[projection]}"
    module = targets_for(model, projection)[name]
    key = name + ".weight"
    require(changed_state(model, original) == [], "Intervention did not start from original BF16 state")
    control.update(pre_matches_original=True, state_tensors_checked=len(original),
                   comparison="full uint8 bitwise equality against immutable original GPU copies",
                   name=name, shape=list(module.weight.shape), parameters=module.weight.numel(),
                   original_sha256=original_hashes[key], restored=False)
    quant = fake_quantize(module.weight, bits, 128)
    control.update(quantized_sha256=tensor_hash(quant),
                   relative_l2_error=((quant.float() - module.weight.float()).norm() / module.weight.float().norm()).item())
    try:
        module.weight.copy_(quant)
        changed = changed_state(model, original)
        require(changed == [key], f"Expected exactly {key}; changed: {changed}")
        require(changed_state(model, original, (key, quant)) == [], "Applied quantization differs")
        control.update(changed_tensors=changed, non_target_unchanged=True, applied_exact=True)
        yield
        require(changed_state(model, original, (key, quant)) == [], "State changed during evaluation")
        control["unchanged_during_eval"] = True
    finally:
        module.weight.copy_(original[key])
        require(changed_state(model, original) == [], "Failed full-state restoration")
        control["restored"] = True


def validate_condition(result, manifest, indices, blocks, baseline=None):
    c = result["condition"]
    layer, projection, bits = identity(c)
    require(result["status"] == "complete", "Uncommitted condition")
    rows = result["rows"]
    require(len(rows) == 64 and [r["sample_id"] for r in rows] == list(range(64)), "Missing/duplicate blocks")
    for s, row in enumerate(rows):
        require((row["condition"], row["layer"], row["projection"], row["bits"], row["context"], row["n_tokens"]) ==
                (c, layer, projection, bits, 2048, 2048), "Condition/scoring identity mismatch")
        require(row["block_index"] == int(indices[s]) and row["input_sha256"] == token_hash(blocks[s]) and
                row["labels_sha256"] == token_hash(blocks[s, 1:]), "Input/labels mismatch")
        require(all(math.isfinite(row[k]) for k in ("nll", "delta_nll", "kl")) and row["nll"] >= 0 and row["kl"] >= -1e-6,
                "Invalid metrics")
        base = row if baseline is None else baseline["rows"][s]
        require(row["delta_nll"] == float(np.float32(row["nll"]) - np.float32(base["nll"])), "Delta NLL mismatch")
        if c == "BF16":
            require(row["delta_nll"] == row["kl"] == 0, "Nonzero baseline self comparison")
    ctrl = result["control"]
    require(ctrl["state_tensors_checked"] == len(manifest["original_state_sha256"]) == 400, "Incomplete state coverage")
    if c == "BF16":
        require(ctrl["unchanged_during_eval"] is True and ctrl["historical_baseline_exact"] is True, "BF16 control failed")
        checks = ctrl["api_alignment"]
        require(len(checks) == 4 and {(r["sample_id"], r["start"]) for r in checks} ==
                {(s, start) for s in (0, 63) for start in (0, 1984)}, "Incomplete API alignment checks")
        for r in checks:
            require(r["positions"] == 64 and r["logits_bitwise_equal"] is True and
                    math.isclose(r["cross_entropy_sum"], r["metric_nll_sum"], rel_tol=1e-6, abs_tol=1e-4), "API alignment failed")
    else:
        require(baseline is not None, "Interventions require validated BF16")
        for key in ("pre_matches_original", "non_target_unchanged", "applied_exact", "unchanged_during_eval", "restored"):
            require(ctrl.get(key) is True, f"Intervention control failed: {key}")
        name = f"model.layers.{layer}.self_attn.{PROJECTIONS[projection]}"
        width = 4096 if projection == "WQ" else 1024
        require(ctrl["name"] == name and ctrl["changed_tensors"] == [name + ".weight"] and
                ctrl["shape"] == [width, 2560] and ctrl["parameters"] == width * 2560, "Not exactly one expected weight")
        require(ctrl["original_sha256"] == manifest["original_state_sha256"][name + ".weight"] and
                ctrl["quantized_sha256"] != ctrl["original_sha256"] and
                math.isfinite(ctrl["relative_l2_error"]) and ctrl["relative_l2_error"] > 0, "Invalid RTN evidence")
        restored = ctrl["restored_outputs"]
        require(len(restored) == 2 and [r["sample_id"] for r in restored] == [0, 63], "Missing restore outputs")
        for r in restored:
            require(r["hidden_bitwise_equal"] is True and r["nll"] == baseline["rows"][r["sample_id"]]["nll"] and r["kl"] == 0,
                    "Restored predictive output differs")


def validate_hidden_cache(out, baseline):
    cache = baseline.get("hidden_cache", {})
    require(isinstance(cache.get("path"), str) and isinstance(cache.get("sha256"), str), "Missing BF16 hidden cache metadata")
    path = out / cache["path"]
    require(not Path(cache["path"]).is_absolute() and path.resolve().is_relative_to(out.resolve()), "Cache path outside output")
    require(path.is_file(), "Missing BF16 hidden cache")
    require(file_hash(path) == cache["sha256"], "BF16 hidden cache changed")


def completed_results(out, manifest, complete=False):
    indices, blocks = load_samples(out)
    results = {}
    allowed = {c + ".json" for c in CONDITIONS}
    require(all(p.name in allowed for p in (out / "conditions").glob("*.json")), "Unknown result condition")
    for c in CONDITIONS:
        path = out / "conditions" / (c + ".json")
        recorded = manifest.get("condition_sha256", {}).get(c)
        if not path.exists():
            require(recorded is None, f"Missing previously committed result {c}")
            continue
        if recorded:
            require(file_hash(path) == recorded, f"Changed committed result {c}")
        result = json.loads(path.read_text())
        require(result["condition"] == c, "Filename/condition mismatch")
        validate_condition(result, manifest, indices, blocks, results.get("BF16"))
        if c == "BF16":
            validate_hidden_cache(out, result)
        results[c] = result
    require(not complete or set(results) == set(CONDITIONS), f"Incomplete: {len(results)}/109 conditions")
    return results


def verify_inputs(out, manifest, *, require_current_source=False):
    require(manifest["config"] == json.loads(CONFIG.read_text()), "Fixed configuration changed")
    for name, expected in manifest["source_sha256"].items():
        require(file_hash(out / "source" / name) == expected, f"Source snapshot changed: {name}")
        # CPU-only validation can be repaired without rewriting producer hashes.
        # Any NEW measurement still requires the exact original producer source;
        # use its saved source/scripts runner to resume an older partial run.
        if require_current_source:
            require(file_hash(ROOT / name) == expected, f"Runner source changed: {name}; use the preserved producer snapshot for missing-condition resume")
    for name, expected in manifest["input_sha256"].items():
        require(file_hash(out / name) == expected, f"Input changed: {name}")
    for name, expected in manifest["history_sha256"].items():
        require(file_hash(Path(manifest["history_dir"]) / name) == expected, f"Historical artifact changed: {name}")
    indices, _ = load_samples(out)
    history = json.loads((out / "historical_run.json").read_text())
    require(indices.tolist() == history["data"]["sample_indices"] and
            np.array_equal(indices, np.random.default_rng(42).choice(history["data"]["full_blocks"], 64, replace=False)), "Sampling changed")
    return history


def initialize(out, history_dir):
    history = json.loads((history_dir / "phase1a_run.json").read_text())
    cfg = json.loads(CONFIG.read_text())
    require(history["status"] == "measured" and history["final_state_matches_checkpoint"] is True, "History not measured/restored")
    for key, value in cfg.items():
        if key in history["config"]:
            require(value == history["config"][key], f"Historical setting changed: {key}")
    for name, expected in history["artifacts_sha256"].items():
        require(file_hash(history_dir / name) == expected, f"Corrupt historical artifact: {name}")
    history_hashes = {str(p.relative_to(history_dir)): file_hash(p) for p in sorted(history_dir.rglob("*"))
                      if p.is_file() and not p.is_relative_to(out)}
    for src, dst in ((history_dir / "phase1a_run.json", "historical_run.json"),
                     (history_dir / "phase1a_tokens.npz", "sampled_token_blocks.npz"),
                     (history_dir / "phase1a.csv", "historical_block_metrics.csv"), (CONFIG, "experiment_config.json")):
        require(not (out / dst).exists(), f"Refusing to overwrite {dst}")
        shutil.copyfile(src, out / dst)
    for name in SOURCES:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    manifest = {"status": "initialized", "started_utc": now(), "config": cfg, "history_dir": str(history_dir),
                "history_sha256": history_hashes, "source_sha256": {n: file_hash(ROOT / n) for n in SOURCES},
                "input_sha256": {n: file_hash(out / n) for n in ("historical_run.json", "sampled_token_blocks.npz",
                                    "historical_block_metrics.csv", "experiment_config.json", "protocol.md")},
                "original_state_sha256": history["original_state_sha256"], "condition_sha256": {}, "attempts": [],
                "rerun_command": "CUDA_VISIBLE_DEVICES=<idle-GPU-UUID> bash scripts/run_layerwise_projection_sensitivity.sh <NEW_OUTPUT>",
                "resume_command": f"CUDA_VISIBLE_DEVICES=<idle-GPU-UUID> bash scripts/run_layerwise_projection_sensitivity.sh {shlex.quote(str(out))} --resume"}
    save_json(out / "run_manifest.json", manifest)
    return manifest


def api_checks(model, blocks, references):
    checks = []
    for s in (0, 63):
        ids = torch.tensor(blocks[s], device="cuda").unsqueeze(0)
        for start in (0, 1984):
            positions = torch.arange(start, start + 64, device="cuda")
            actual = model(input_ids=ids, use_cache=False, logits_to_keep=positions, return_dict=True)
            manual = model.lm_head(references[s:s + 1, start:start + 64].to("cuda"))
            require(actual.past_key_values is None and torch.equal(actual.logits, manual), "CausalLM logits differ")
            labels = ids[:, start + 1:start + 65]
            direct = F.cross_entropy(actual.logits.float().reshape(-1, actual.logits.shape[-1]), labels.reshape(-1), reduction="sum")
            ns, ks = metric_from_logits(manual, actual.logits, labels)
            require(torch.allclose(ns, direct, atol=1e-4, rtol=1e-6) and ks.item() == 0, "Shifted CE mismatch")
            checks.append(dict(sample_id=s, start=start, positions=64, logits_bitwise_equal=True,
                               cross_entropy_sum=direct.item(), metric_nll_sum=ns.item()))
    return checks


def evaluate(model, condition, indices, blocks, references, baseline, partial_path):
    layer, projection, bits = identity(condition)
    rows = []
    start = time.monotonic()
    for s in range(64):
        ids = torch.tensor(blocks[s], device="cuda").unsqueeze(0)
        hidden = hidden_for(model, ids)
        if condition == "BF16":
            references[s].copy_(hidden[0].cpu())
        nll, kl = score(model.lm_head, hidden, references[s:s + 1], ids[:, 1:], 64)
        base_nll = nll if condition == "BF16" else baseline["rows"][s]["nll"]
        rows.append(dict(condition=condition, layer=layer, projection=projection, bits=bits, context=2048,
                         sample_id=s, block_index=int(indices[s]), n_tokens=2048, input_sha256=token_hash(blocks[s]),
                         labels_sha256=token_hash(blocks[s, 1:]), nll=nll,
                         delta_nll=float(np.float32(nll) - np.float32(base_nll)), kl=kl))
        if (s + 1) % 8 == 0:
            save_json(partial_path, {"status": "partial_unvalidated", "condition": condition, "rows": rows})
            print(f"{now()} {condition} blocks={s + 1}/64 seconds={time.monotonic() - start:.1f}", flush=True)
    return rows


def run_missing(out, manifest, results, attempt, attempt_path, max_conditions):
    history = verify_inputs(out, manifest, require_current_source=True)
    indices, blocks = load_samples(out)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), "Select exactly one usable BF16 CUDA GPU")
    attempt.update(host=platform.node(), python=sys.version,
                   versions={p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "safetensors", "huggingface-hub", "tokenizers", "accelerate")},
                   cuda={"build": torch.version.cuda, "device": str(torch.cuda.get_device_properties(0)), "free_total_bytes": list(torch.cuda.mem_get_info())})
    for p, version in attempt["versions"].items():
        require(version == history["versions"][p], f"Historical runtime version changed: {p}")
    require(torch.cuda.mem_get_info()[0] >= 19 * 1024 ** 3, "Need >=19 GiB free GPU memory for model + exact-state controls")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model_path = Path(history["model_source"])
    for name, expected in history["checkpoint_sha256"].items():
        require(file_hash(model_path / name) == expected, f"Checkpoint changed: {name}")
    require(file_hash(model_path / "config.json") == history["checkpoint_config_sha256"], "Checkpoint config changed")
    from transformers import AutoModelForCausalLM
    from torch.nn.attention import SDPBackend, sdpa_kernel
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0",
                                               attn_implementation="sdpa", local_files_only=True, trust_remote_code=False)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    require(all(p.dtype == torch.bfloat16 for p in model.parameters()), "Non-BF16 parameters")
    require(len(model.model.layers) == 36 and model.config.num_attention_heads == 32 and model.config.num_key_value_heads == 8,
            "Unexpected model/GQA structure")
    require(model.lm_head.weight is model.model.embed_tokens.weight, "Unexpected untied head")
    for p in PROJECTIONS:
        targets_for(model, p)
    require(model_hashes(model) == manifest["original_state_sha256"], "Loaded model differs from archived original checkpoint")
    original = {n: t.detach().clone() for n, t in state_tensors(model).items()}
    attempt.update(original_state_matches_history=True, state_tensors=400, tied_lm_head=True,
                   runtime_model_config=model.config.to_dict(), runtime_attention_backend=model.config._attn_implementation,
                   numerics=history["numerics"], status="running")
    save_json(attempt_path, attempt)
    baseline = results.get("BF16")
    if baseline:
        cache = baseline["hidden_cache"]
        require(file_hash(out / cache["path"]) == cache["sha256"], "BF16 hidden cache changed")
        references = torch.load(out / cache["path"], map_location="cpu", weights_only=True)
        require(references.shape == (64, 2048, 2560) and references.dtype == torch.bfloat16 and bool(torch.isfinite(references).all()), "Invalid baseline cache")
    else:
        references = torch.empty((64, 2048, 2560), dtype=torch.bfloat16)
    pending = [c for c in CONDITIONS[1:] if c not in results]
    if max_conditions is not None:
        pending = pending[:max_conditions]
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for c in (["BF16"] if baseline is None else []) + pending:
            start = time.monotonic()
            ctrl = {}
            partial = attempt_path.parent / (c + ".partial.json")
            try:
                if c == "BF16":
                    rows = evaluate(model, c, indices, blocks, references, None, partial)
                    with open(out / "historical_block_metrics.csv", newline="") as f:
                        historical = {int(r["sample_id"]): float(r["nll"]) for r in csv.DictReader(f)
                                      if r["condition"] == "BF16" and int(r["context"]) == 2048}
                    require(len(historical) == 64 and all(r["nll"] == historical[r["sample_id"]] for r in rows), "BF16 does not reproduce historical 2048 baseline")
                    ctrl.update(api_alignment=api_checks(model, blocks, references), historical_baseline_exact=True,
                                state_tensors_checked=400, unchanged_during_eval=changed_state(model, original) == [])
                    require(ctrl["unchanged_during_eval"], "BF16 evaluation mutated model")
                    cache_path = attempt_path.parent / "baseline_hidden.pt"
                    torch.save(references, cache_path)
                    hidden_cache = {"path": str(cache_path.relative_to(out)), "sha256": file_hash(cache_path)}
                else:
                    with single_intervention(model, c, original, manifest["original_state_sha256"], ctrl):
                        rows = evaluate(model, c, indices, blocks, references, baseline, partial)
                    restored = []
                    for s in (0, 63):
                        ids = torch.tensor(blocks[s], device="cuda").unsqueeze(0)
                        hidden = hidden_for(model, ids)
                        same = torch.equal(hidden.cpu(), references[s:s + 1])
                        nll, kl = score(model.lm_head, hidden, references[s:s + 1], ids[:, 1:], 64)
                        require(same and nll == baseline["rows"][s]["nll"] and kl == 0, "Restored output differs")
                        restored.append(dict(sample_id=s, hidden_bitwise_equal=same, nll=nll, kl=kl))
                    require(changed_state(model, original) == [], "Restoration control forward mutated state")
                    ctrl["restored_outputs"] = restored
                result = dict(status="complete", condition=c, rows=rows, control=ctrl,
                              attempt=str(attempt_path.relative_to(out)), finished_utc=now(), seconds=time.monotonic() - start)
                if c == "BF16":
                    result["hidden_cache"] = hidden_cache
                validate_condition(result, manifest, indices, blocks, baseline)
                destination = out / "conditions" / (c + ".json")
                require(not destination.exists(), f"Refusing to overwrite completed {c}")
                save_json(destination, result)
                results[c] = result
                if c == "BF16":
                    baseline = result
                manifest["condition_sha256"][c] = file_hash(destination)
                manifest["completed_conditions"] = [v for v in CONDITIONS if v in results]
                save_json(out / "run_manifest.json", manifest)
                print(f"COMMITTED {c}: {len(results)}/109; all controls PASS", flush=True)
            except BaseException:
                save_json(attempt_path.parent / (c + ".failed_controls.json"), {"condition": c, "control": ctrl, "error": traceback.format_exc()})
                raise
        require(changed_state(model, original) == [], "Final state differs")
    attempt.update(final_state_matches_checkpoint=True, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--history", type=Path, default=ROOT / "results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-conditions", type=int, help="Bound this invocation's complete interventions, not total conditions/blocks")
    args = parser.parse_args()
    require(args.max_conditions is None or 1 <= args.max_conditions <= 108, "max-conditions must be 1..108")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / ".run.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = out / "run_manifest.json"
    require(args.resume == manifest_path.exists(), "Existing run requires --resume; new run must omit --resume")
    if not (out / "protocol.md").exists():
        shutil.copyfile(OUTPUT / "protocol.md", out / "protocol.md")
    (out / "conditions").mkdir(exist_ok=True)
    attempt_dir = out / "attempts" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    attempt_dir.mkdir(parents=True)
    attempt_path = attempt_dir / "attempt.json"
    command = shlex.join([sys.executable, *sys.argv])
    attempt = {"status": "starting", "started_utc": now(), "command": command, "cwd": str(Path.cwd()),
               "environment": {k: os.environ.get(k) for k in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    manifest = None
    try:
        attempt["validation_source_sha256"] = {n: file_hash(ROOT / n) for n in SOURCES}
        for name in SOURCES:
            target = attempt_dir / "validation_source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        manifest = json.loads(manifest_path.read_text()) if args.resume else initialize(out, args.history.resolve())
        verify_inputs(out, manifest)
        results = completed_results(out, manifest)
        attempt["skipped_completed"] = list(results)
        manifest["attempts"].append(str(attempt_path.relative_to(out)))
        manifest["condition_sha256"] = {c: file_hash(out / "conditions" / (c + ".json")) for c in results}
        save_json(manifest_path, manifest)
        if len(results) < 109:
            attempt["gpu_inventory_before"] = subprocess.check_output(["nvidia-smi"], text=True)
            save_json(attempt_path, attempt)
            manifest["status"] = "running"
            save_json(manifest_path, manifest)
            run_missing(out, manifest, results, attempt, attempt_path, args.max_conditions)
        else:
            attempt["no_forward_needed"] = True
        completed_results(out, manifest, complete=len(results) == 109)
        manifest.update(status="measured" if len(results) == 109 else "partial", completed_conditions=list(results), updated_utc=now())
        attempt.update(status="complete" if len(results) == 109 else "bounded_partial", finished_utc=now(),
                       newly_completed=[c for c in results if c not in attempt["skipped_completed"]])
        save_json(attempt_path, attempt)
        save_json(manifest_path, manifest)
        print(f"FINISHED {len(results)}/109 status={manifest['status']} output={out}", flush=True)
    except BaseException:
        error = traceback.format_exc()
        attempt.update(status="blocked", error=error, failed_utc=now())
        save_json(attempt_path, attempt)
        done = [] if manifest is None else manifest.get("completed_conditions", [])
        if manifest is not None:
            manifest.update(status="blocked", last_error=error)
            save_json(manifest_path, manifest)
        with open(out / "blocked.md", "a") as f:
            f.write(f"\n## {now()} 未完成\n\nCommand: `{command}`\n\nCompleted {len(done)}/109: {done}\n\n"
                    f"Missing: {[c for c in CONDITIONS if c not in done]}\n\n```text\n{error}\n```\n"
                    "保留歷史/完成條件/partial與控制證據。不要縮樣本或換模型。缺資源請提供原checkpoint/data或>=19GiB空閒BF16 GPU，使用者可 `/goal pause` 後通知繼續。\n")
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
