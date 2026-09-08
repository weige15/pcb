#!/usr/bin/env python3
"""Independent CPU-only verifier for the static KV-cache Pareto artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/kv_cache_precision_pareto"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def arr_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def ci(values: np.ndarray, draws: np.ndarray) -> dict:
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(means, 0.025, method="linear")),
            "ci_high": float(np.quantile(means, 0.975, method="linear"))}


def p95(values: list[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95, method="linear"))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); out = args.output.resolve()
    manifest = json.loads((out / "run_manifest.json").read_text())
    cfg = manifest["config"]
    require(manifest.get("status") == "measured", "Run is not measured")
    require(manifest.get("probe_status") == "passed", "Direct-kernel smoke did not pass")
    for name, expected in manifest["source_sha256"].items():
        require(sha(ROOT / name) == expected, f"Source drift: {name}")
    for name, expected in manifest["input_sha256"].items():
        require(sha(out / name) == expected, f"Input drift: {name}")
    for name, expected in manifest["history_sha256"].items():
        require(sha(ROOT / name) == expected, f"Historical artifact drift: {name}")
    prompt_manifest = json.loads((out / "prompt_manifest.json").read_text())
    require(prompt_manifest["config_sha256"] == sha(out / "experiment_config.json"), "Prompt/config mismatch")
    with np.load(out / "prompts.npz", allow_pickle=False) as f:
        arrays = {name: f[name] for name in f.files}
    for name, expected in prompt_manifest["array_sha256"].items(): require(arr_sha(arrays[name]) == expected, f"Prompt hash drift: {name}")
    for length in cfg["prompt_lengths"]:
        require(arrays[f"performance_prompt_{length}"].shape == (cfg["performance_prompts_per_length"], length), f"Prompt shape {length}")
        require(arrays[f"performance_target_{length}"].shape == (cfg["performance_prompts_per_length"], cfg["output_length"]), f"Target shape {length}")
        require(arrays[f"quality_prompt_{length}"].shape == (cfg["quality_prompts_per_length"], length), f"Quality shape {length}")
    require(arrays["retrieval_prompt_8192"].shape == (cfg["retrieval_prompts"], 8192), "Retrieval prompt shape")

    modes = [x["mode"] for x in manifest["probe"] if x.get("status") == "complete"]
    require("BF16" in modes and "FP8_E4M3" in modes, "Required BF16/FP8 smoke modes absent")
    with open(out / "raw_metrics.csv", newline="") as f: raw = list(csv.DictReader(f))
    runs = [json.loads(x) for x in (out / "raw_runs.jsonl").read_text().splitlines() if x.strip()]
    profiles = [json.loads(x) for x in (out / "raw_cache_profiles.jsonl").read_text().splitlines() if x.strip()]
    retrieval = json.loads((out / "retrieval_raw.json").read_text())
    require(raw and runs and profiles, "Raw artifact is empty")
    expected_q = len(modes) * len(cfg["prompt_lengths"]) * cfg["quality_prompts_per_length"] * cfg["output_length"]
    qraw = [r for r in raw if r["phase"] == "quality"]
    require(len(qraw) == expected_q, f"Expected {expected_q} quality rows, got {len(qraw)}")
    require(len(retrieval) == len(modes) * cfg["retrieval_prompts"], "Retrieval row count")
    retrieval_metric_rows = [r for r in raw if r.get("phase") == "retrieval"]
    require(len(retrieval_metric_rows) == len(retrieval), "Raw retrieval metrics missing")
    perf_runs = [r for r in runs if r.get("phase") == "performance"]
    expected_perf = len(modes) * len(cfg["prompt_lengths"]) * len(cfg["performance_concurrency"]) * (cfg["warmups"] + cfg["repeats"])
    require(len(perf_runs) == expected_perf, f"Expected {expected_perf} performance run records, got {len(perf_runs)}")
    for mode in modes:
        for length in cfg["prompt_lengths"]:
            for n in cfg["performance_concurrency"]:
                rows = [r for r in perf_runs if r.get("mode") == mode and r.get("prompt_length") == length and r.get("concurrency") == n]
                require(len(rows) == cfg["warmups"] + cfg["repeats"], f"Performance cell incomplete {mode}/{length}/{n}")
                measured = [r for r in rows if r.get("repeat") in range(cfg["repeats"])]
                require(len(measured) == cfg["repeats"], f"Measured repeats incomplete {mode}/{length}/{n}")
                for r in measured:
                    require(r.get("status") in ("complete", "oom_or_error"), "Unrecorded performance failure")
                    metric_rows = [m for m in raw if m.get("phase") == "performance" and m.get("mode") == mode and
                                   int(m["prompt_length"]) == length and int(m["concurrency"]) == n and
                                   int(m["repeat"]) == r["repeat"]]
                    if r["status"] == "complete":
                        require(len(r["timestamps_ms"]) == cfg["output_length"], "Per-token timestamps missing")
                        require(len(r["output_token_ids"]) == n, "Per-request outputs missing")
                        require(len(metric_rows) == n and {int(m["request_id"]) for m in metric_rows} == set(range(n)),
                                f"Per-request performance metrics missing {mode}/{length}/{n}/{r['repeat']}")
                    else:
                        require(len(metric_rows) == 1 and metric_rows[0].get("status") == "oom_or_error",
                                f"Performance error metric missing {mode}/{length}/{n}/{r['repeat']}")
    # Direct cache and byte accounting are checked from every completed system run.
    complete_profiles = [p for p in profiles if p.get("status") == "complete" and p.get("mode") in modes]
    expected_profile_count = sum(x.get("status") == "complete" for x in manifest["probe"])
    expected_profile_count += len(modes) * len(cfg["prompt_lengths"]) * cfg["quality_prompts_per_length"]
    expected_profile_count += len(modes) * cfg["retrieval_prompts"]
    expected_profile_count += sum(r.get("status") == "complete" for r in perf_runs)
    require(len(complete_profiles) == expected_profile_count,
            f"Expected {expected_profile_count} completed cache profiles, got {len(complete_profiles)}")
    require(complete_profiles, "No completed cache profile")
    for p in complete_profiles:
        mode = p["mode"]; elem = 2 if mode == "BF16" else 1
        require(p.get("num_layers") == 36, f"Layer dimension missing from cache profile: {p}")
        expected = 2 * p["num_layers"] * p["batch"] * math.ceil(p["max_length"] / cfg["page_size"]) * cfg["page_size"] * 8 * 128 * elem
        require(p["configured_payload_bytes"] == expected and p["cache_payload_bytes"] == expected, f"Cache bytes mismatch: {p}")
        require(p["direct_kernel_read"] is True and p["whole_cache_bf16_restore"] is False, "Non-direct cache admitted")
        require(p["cache_tensor_passed_to_kernel_dtype"] in ("torch.bfloat16", "torch.float8_e4m3fn", "torch.float8_e5m2"), "Unknown cache dtype")
        if mode != "BF16": require(p["cache_tensor_passed_to_kernel_dtype"] != "torch.bfloat16", "Compressed cache restored to BF16")
    four = manifest.get("capability_probe", {}).get("four_bit", {})
    require(four.get("attempted") is True, "4-bit capability probe was not attempted")
    require(four.get("supported") is False, "4-bit was not honestly classified")
    require(four.get("direct_paged_fp4_symbols") == [], "A direct paged FP4 API was found but not tested as a mode")

    # Recompute quality summaries independently at prompt level.
    qindex = {}
    for r in qraw:
        key = (r["mode"], int(r["prompt_length"]), int(r["request_id"]))
        qindex.setdefault(key, []).append(r)
    draws = np.random.default_rng(cfg["quality_bootstrap_seed"]).integers(0, cfg["quality_prompts_per_length"], size=(cfg["quality_bootstrap_replicates"], cfg["quality_prompts_per_length"]))
    expected_quality = []
    for mode in modes:
        for length in cfg["prompt_lengths"]:
            deltas = np.asarray([np.mean([float(r["delta_nll"]) for r in qindex[(mode, length, p)]]) for p in range(cfg["quality_prompts_per_length"])])
            kls = np.asarray([np.mean([float(r["kl"]) for r in qindex[(mode, length, p)]]) for p in range(cfg["quality_prompts_per_length"])])
            expected_quality.append({"mode": mode, "prompt_length": length, "delta_nll": ci(deltas, draws), "kl": ci(kls, draws),
                                    "prompt_means_delta_nll": deltas.tolist(), "prompt_means_kl": kls.tolist(),
                                    "n_prompts": len(deltas), "tokens_per_prompt": cfg["output_length"]})
    quality_summary = json.loads((out / "quality_summary.json").read_text())
    require(quality_summary["quality"] == expected_quality, "Independent quality summary differs")
    retrieval_summary = []
    base = np.mean([r["em"] for r in retrieval if r["mode"] == "BF16"])
    for mode in modes:
        vals = [r["em"] for r in retrieval if r["mode"] == mode]
        decline = (base - np.mean(vals)) * 100.0
        retrieval_summary.append({"mode": mode, "em": float(np.mean(vals)), "em_count": int(sum(vals)), "n": len(vals),
                                  "decline_percentage_points_vs_bf16": float(decline),
                                  "pass": decline <= cfg["quality_gates"]["retrieval_em_decline_percentage_points_max"]})
    require(quality_summary["retrieval"] == retrieval_summary, "Independent retrieval summary differs")

    # Recompute the performance aggregates, provisional gates, and compressed
    # Pareto set without importing the analysis script.
    perf = []
    for mode in modes:
        for length in cfg["prompt_lengths"]:
            for concurrency in cfg["performance_concurrency"]:
                cells = [r for r in perf_runs if r.get("mode") == mode and r.get("prompt_length") == length and
                         r.get("concurrency") == concurrency and r.get("repeat") in range(cfg["repeats"])]
                complete = [r for r in cells if r.get("status") == "complete"]
                ok = len(complete) == cfg["repeats"]
                cell = {"mode": mode, "prompt_length": length, "concurrency": concurrency,
                        "repeats": cfg["repeats"], "complete_repeats": len(complete), "stable": ok}
                if ok:
                    ttft = [float(r["ttft_ms"]) for r in complete]
                    tpot = [float(r["tpot_ms"]) for r in complete]
                    wall = [float(r["wall_ms"]) for r in complete]
                    slo_t = cfg["provisional_slo"]["p95_ttft_ms"][str(length)]
                    slo_p = cfg["provisional_slo"]["p95_tpot_ms"]
                    goodputs = []
                    for r in complete:
                        served = sum(1 for _ in range(concurrency) if r["ttft_ms"] <= slo_t and r["tpot_ms"] <= slo_p)
                        goodputs.append(served / (r["wall_ms"] / 1000.0))
                    cell.update(p95_ttft_ms=p95(ttft), p95_tpot_ms=p95(tpot), mean_wall_ms=float(np.mean(wall)),
                                mean_slo_goodput_rps=float(np.mean(goodputs)), p95_slo_goodput_rps=p95(goodputs),
                                max_gpu_allocated_bytes=max(int(r.get("gpu_allocated_after_cache_bytes", 0)) for r in complete))
                else:
                    cell.update(p95_ttft_ms=None, p95_tpot_ms=None, mean_wall_ms=None,
                                mean_slo_goodput_rps=0.0, p95_slo_goodput_rps=0.0,
                                max_gpu_allocated_bytes=None,
                                errors=[r.get("error", "") for r in cells if r.get("status") != "complete"])
                perf.append(cell)
    perf_by = {(r["mode"], r["prompt_length"], r["concurrency"]): r for r in perf}
    max_stable = {}
    for mode in modes:
        for length in cfg["prompt_lengths"]:
            stable = [n for n in cfg["performance_concurrency"] if perf_by[(mode, length, n)]["stable"]]
            max_stable[(mode, length)] = max(stable) if stable else 0
    for row in perf:
        baseline = perf_by[("BF16", row["prompt_length"], row["concurrency"])]
        if row["stable"] and row["mode"] != "BF16" and baseline["stable"]:
            row["tpot_improvement_fraction_vs_bf16"] = 1.0 - row["p95_tpot_ms"] / baseline["p95_tpot_ms"]
            row["goodput_improvement_fraction_vs_bf16"] = (row["mean_slo_goodput_rps"] / baseline["mean_slo_goodput_rps"] - 1.0) if baseline["mean_slo_goodput_rps"] > 0 else None
        else:
            row["tpot_improvement_fraction_vs_bf16"] = None
            row["goodput_improvement_fraction_vs_bf16"] = None
    gates = []
    for mode in modes:
        qrows = [r for r in expected_quality if r["mode"] == mode]
        qpass = all(r["delta_nll"]["ci_high"] <= cfg["quality_gates"]["delta_nll_ci_high_max"] and
                    r["kl"]["ci_high"] <= cfg["quality_gates"]["kl_ci_high_max"] for r in qrows)
        rpass = next(r["pass"] for r in retrieval_summary if r["mode"] == mode)
        matched = [r for r in perf if r["mode"] == mode and r["stable"] and
                   ((r["tpot_improvement_fraction_vs_bf16"] is not None and r["tpot_improvement_fraction_vs_bf16"] >= cfg["quality_gates"]["performance_improvement_min_fraction"]) or
                    (r["goodput_improvement_fraction_vs_bf16"] is not None and r["goodput_improvement_fraction_vs_bf16"] >= cfg["quality_gates"]["performance_improvement_min_fraction"]))]
        conc_route = []
        for length in cfg["prompt_lengths"]:
            n = max_stable[(mode, length)]; bn = max_stable[("BF16", length)]
            if n >= math.ceil(bn * (1.0 + cfg["quality_gates"]["max_concurrency_improvement_min_fraction"])) and n and bn:
                current = perf_by[(mode, length, n)]; base_row = perf_by[("BF16", length, bn)]
                if current["p95_ttft_ms"] <= base_row["p95_ttft_ms"] * (1.0 + cfg["quality_gates"]["ttft_regression_max_fraction_for_concurrency_route"]):
                    conc_route.append({"prompt_length": length, "mode_max_concurrency": n, "bf16_max_concurrency": bn})
        gates.append({"mode": mode,
                      "system_admissible": mode == "BF16" or any(p.get("mode") == mode and p.get("direct_kernel_read") and not p.get("whole_cache_bf16_restore") for p in complete_profiles),
                      "quality_pass": qpass and rpass, "nll_kl_pass": qpass, "retrieval_pass": rpass,
                      "performance_pass": bool(matched or conc_route), "matched_performance_points": matched,
                      "concurrency_route_points": conc_route, "gate_pass": qpass and rpass and bool(matched or conc_route)})

    candidates = []
    for row in perf:
        if row["mode"] == "BF16" or not row["stable"]:
            continue
        qrow = next(x for x in expected_quality if x["mode"] == row["mode"] and x["prompt_length"] == row["prompt_length"])
        gate = next(x for x in gates if x["mode"] == row["mode"])
        cache_bytes = next((p["configured_payload_bytes"] for p in profiles if p.get("phase") == "performance" and
                            p.get("mode") == row["mode"] and p.get("prompt_length") == row["prompt_length"] and
                            p.get("concurrency") == row["concurrency"] and p.get("status") == "complete"), None)
        candidates.append({"id": f"{row['mode']}/L{row['prompt_length']}/N{row['concurrency']}", "mode": row["mode"],
                           "prompt_length": row["prompt_length"], "concurrency": row["concurrency"],
                           "quality_pass": gate["quality_pass"], "gate_pass": gate["gate_pass"],
                           "delta_nll_ci_high": qrow["delta_nll"]["ci_high"], "kl_ci_high": qrow["kl"]["ci_high"],
                           "p95_ttft_ms": row["p95_ttft_ms"], "p95_tpot_ms": row["p95_tpot_ms"],
                           "goodput_rps": row["mean_slo_goodput_rps"], "cache_bytes": cache_bytes})
    for c in candidates:
        peers = [x for x in candidates if x["prompt_length"] == c["prompt_length"] and x["id"] != c["id"]]
        def no_worse(x):
            return (x["delta_nll_ci_high"] <= c["delta_nll_ci_high"] and x["kl_ci_high"] <= c["kl_ci_high"] and
                    x["p95_ttft_ms"] <= c["p95_ttft_ms"] and x["p95_tpot_ms"] <= c["p95_tpot_ms"] and
                    x["goodput_rps"] >= c["goodput_rps"] and
                    (c["cache_bytes"] is None or x["cache_bytes"] is None or x["cache_bytes"] <= c["cache_bytes"]))
        c["dominated"] = any(no_worse(x) and x != c for x in peers)
    pareto_points = [c for c in candidates if not c["dominated"]]
    passing_pareto = [c for c in pareto_points if c["gate_pass"]]
    max_stable_rows = [{"mode": m, "prompt_length": l, "max_stable_concurrency": max_stable[(m, l)]}
                       for m in modes for l in cfg["prompt_lengths"]]
    perf_summary = json.loads((out / "performance_summary.json").read_text())
    require(perf_summary["performance"] == perf and perf_summary["max_stable_concurrency"] == max_stable_rows and
            perf_summary["gates"] == gates, "Independent performance/gate summary differs")
    pareto = json.loads((out / "pareto_points.json").read_text())
    require(pareto["status"] == "analyzed" and pareto["provisional"] is True, "Pareto analysis missing/provisional flag missing")
    require(pareto["performance"] == perf and pareto["max_stable_concurrency"] == max_stable_rows and
            pareto["gates"] == gates and pareto["pareto_points"] == pareto_points and
            pareto["passing_pareto_points"] == passing_pareto, "Independent Pareto result differs")
    report_text = (out / "pareto_report.md").read_text()
    require((out / "completion_audit.md").is_file(), "Completion audit missing")
    require(("**NO-GO (provisional):**" in report_text) == (len(passing_pareto) == 0), "Report decision inconsistent with recomputed result")
    report = {"status": "verified", "checks": {"source_hashes": True, "history_hashes": True, "inputs": True,
              "prompt_shapes_hashes": True, "raw_quality_rows": len(qraw), "raw_retrieval_rows": len(retrieval_metric_rows), "raw_performance_runs": len(perf_runs),
              "cache_profiles": len(complete_profiles), "direct_kernel_provenance": True, "four_bit_unsupported_recorded": True,
              "quality_recomputed": True, "retrieval_recomputed": True, "performance_recomputed": True,
              "gates_recomputed": True, "pareto_recomputed": True},
              "modes": modes, "expected_performance_records": expected_perf,
              "oom_or_error_records": sum(r.get("status") == "oom_or_error" for r in perf_runs),
              "verified_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    (out / "verifier_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    audit = (out / "completion_audit.md").read_text()
    audit = audit.replace("Independent verifier and test result | independent_verifier.py output verifier_report.json | PENDING verifier",
                          "Independent verifier and test result | independent_verifier.py output verifier_report.json; all independent checks pass | PASS")
    audit = audit.replace("Independent verifier re-read and recomputed artifacts | verifier_report.json; all independent checks pass | PENDING verifier",
                          "Independent verifier re-read and recomputed artifacts | verifier_report.json; all independent checks pass | PASS")
    audit = audit.replace("The final verifier must update the PENDING row only after independently re-reading and recomputing the artifacts.",
                          "Independent verifier completed: source/history/input hashes, matrix completeness, cache bytes, provenance, quality summaries, and retrieval summary were independently recomputed.")
    (out / "completion_audit.md").write_text(audit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
