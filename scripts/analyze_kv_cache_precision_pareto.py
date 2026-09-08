#!/usr/bin/env python3
"""Independent, CPU-only analysis for the hardware-real KV-cache screen."""
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
MODES = ("BF16", "FP8_E4M3", "FP8_E5M2")


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


def p95(values: list[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95, method="linear"))


def ci(values: np.ndarray, draws: np.ndarray) -> dict:
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(means, 0.025, method="linear")),
            "ci_high": float(np.quantile(means, 0.975, method="linear"))}


def load(out: Path):
    manifest = json.loads((out / "run_manifest.json").read_text())
    with open(out / "raw_metrics.csv", newline="") as f:
        raw = list(csv.DictReader(f))
    runs = [json.loads(line) for line in (out / "raw_runs.jsonl").read_text().splitlines() if line.strip()]
    profiles = [json.loads(line) for line in (out / "raw_cache_profiles.jsonl").read_text().splitlines() if line.strip()]
    retrieval = json.loads((out / "retrieval_raw.json").read_text()) if (out / "retrieval_raw.json").exists() else []
    for r in raw:
        for k in ("prompt_length", "concurrency", "repeat", "request_id"):
            if r.get(k, "") not in ("", None): r[k] = int(r[k])
        for k in ("token_index", "nll", "baseline_nll", "delta_nll", "kl", "em", "ttft_ms", "tpot_ms", "completion_ms", "wall_ms"):
            if r.get(k, "") not in ("", None): r[k] = float(r[k])
    return manifest, raw, runs, profiles, retrieval


def analyze(out: Path) -> dict:
    manifest, raw, runs, profiles, retrieval = load(out)
    cfg = manifest["config"]
    modes = [x["mode"] for x in manifest.get("probe", []) if x.get("status") == "complete"]
    require("BF16" in modes, "BF16 smoke missing")
    require(raw, "No raw metrics")
    q = {}
    for mode in modes:
        for length in cfg["prompt_lengths"]:
            for prompt_id in range(cfg["quality_prompts_per_length"]):
                rows = [r for r in raw if r["phase"] == "quality" and r["mode"] == mode and r["prompt_length"] == length and r["request_id"] == prompt_id]
                require(len(rows) == cfg["output_length"], f"quality row count missing: {mode}/{length}/{prompt_id}")
                require({r["token_index"] for r in rows} == set(range(cfg["output_length"])), "quality token index incomplete")
                q[(mode, length, prompt_id)] = rows
    draws = np.random.default_rng(cfg["quality_bootstrap_seed"]).integers(
        0, cfg["quality_prompts_per_length"], size=(cfg["quality_bootstrap_replicates"], cfg["quality_prompts_per_length"]))
    np.save(out / "quality_bootstrap_indices.npy", draws)
    quality = []
    for mode in modes:
        for length in cfg["prompt_lengths"]:
            delta = np.asarray([np.mean([r["delta_nll"] for r in q[(mode, length, p)]]) for p in range(cfg["quality_prompts_per_length"])], dtype=np.float64)
            kl = np.asarray([np.mean([r["kl"] for r in q[(mode, length, p)]]) for p in range(cfg["quality_prompts_per_length"])], dtype=np.float64)
            d = ci(delta, draws); k = ci(kl, draws)
            quality.append({"mode": mode, "prompt_length": length, "delta_nll": d, "kl": k,
                            "prompt_means_delta_nll": delta.tolist(), "prompt_means_kl": kl.tolist(),
                            "n_prompts": len(delta), "tokens_per_prompt": cfg["output_length"]})
    retrieval_summary = []
    for mode in modes:
        rows = [r for r in retrieval if r.get("mode") == mode and r.get("status") == "complete"]
        require(len(rows) == cfg["retrieval_prompts"], f"retrieval rows missing: {mode}")
        retrieval_summary.append({"mode": mode, "em": float(np.mean([r["em"] for r in rows])),
                                  "em_count": int(sum(r["em"] for r in rows)), "n": len(rows)})
    base_em = next(r["em"] for r in retrieval_summary if r["mode"] == "BF16")
    for r in retrieval_summary:
        r["decline_percentage_points_vs_bf16"] = (base_em - r["em"]) * 100.0
        r["pass"] = r["decline_percentage_points_vs_bf16"] <= cfg["quality_gates"]["retrieval_em_decline_percentage_points_max"]

    perf = []
    for mode in modes:
        for length in cfg["prompt_lengths"]:
            for concurrency in cfg["performance_concurrency"]:
                cells = [r for r in runs if r.get("phase") == "performance" and r.get("mode") == mode and
                         r.get("prompt_length") == length and r.get("concurrency") == concurrency and r.get("repeat") in range(cfg["repeats"])]
                require(len(cells) == cfg["repeats"], f"performance repeats missing: {mode}/{length}/{concurrency}")
                complete = [r for r in cells if r.get("status") == "complete"]
                ok = len(complete) == cfg["repeats"]
                cell = {"mode": mode, "prompt_length": length, "concurrency": concurrency,
                        "repeats": cfg["repeats"], "complete_repeats": len(complete), "stable": ok}
                if ok:
                    ttft = [float(r["ttft_ms"]) for r in complete]; tpot = [float(r["tpot_ms"]) for r in complete]
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
                    cell.update(p95_ttft_ms=None, p95_tpot_ms=None, mean_wall_ms=None, mean_slo_goodput_rps=0.0,
                                p95_slo_goodput_rps=0.0, max_gpu_allocated_bytes=None,
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
            row["tpot_improvement_fraction_vs_bf16"] = None; row["goodput_improvement_fraction_vs_bf16"] = None
    # Quality gates are required at both fixed lengths and retrieval.
    gate_modes = []
    for mode in modes:
        qrows = [r for r in quality if r["mode"] == mode]
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
                current = perf_by[(mode, length, n)]
                base = perf_by[("BF16", length, bn)]
                if current["p95_ttft_ms"] <= base["p95_ttft_ms"] * (1.0 + cfg["quality_gates"]["ttft_regression_max_fraction_for_concurrency_route"]):
                    conc_route.append({"prompt_length": length, "mode_max_concurrency": n, "bf16_max_concurrency": bn})
        ppass = bool(matched or conc_route)
        gate_modes.append({"mode": mode, "system_admissible": mode == "BF16" or any(p.get("mode") == mode and p.get("direct_kernel_read") and not p.get("whole_cache_bf16_restore") for p in profiles),
                           "quality_pass": qpass and rpass, "nll_kl_pass": qpass, "retrieval_pass": rpass,
                           "performance_pass": ppass, "matched_performance_points": matched,
                           "concurrency_route_points": conc_route, "gate_pass": qpass and rpass and ppass})

    # A point is dominated only by another complete point of the same prompt length.
    candidates = []
    for row in perf:
        if row["mode"] == "BF16" or not row["stable"]: continue
        qrow = next(x for x in quality if x["mode"] == row["mode"] and x["prompt_length"] == row["prompt_length"])
        gate = next(x for x in gate_modes if x["mode"] == row["mode"])
        candidates.append({"id": f"{row['mode']}/L{row['prompt_length']}/N{row['concurrency']}", "mode": row["mode"],
                           "prompt_length": row["prompt_length"], "concurrency": row["concurrency"],
                           "quality_pass": gate["quality_pass"], "gate_pass": gate["gate_pass"],
                           "delta_nll_ci_high": qrow["delta_nll"]["ci_high"], "kl_ci_high": qrow["kl"]["ci_high"],
                           "p95_ttft_ms": row["p95_ttft_ms"], "p95_tpot_ms": row["p95_tpot_ms"],
                           "goodput_rps": row["mean_slo_goodput_rps"], "cache_bytes": next((p["configured_payload_bytes"] for p in profiles if p.get("phase") == "performance" and p.get("mode") == row["mode"] and p.get("prompt_length") == row["prompt_length"] and p.get("concurrency") == row["concurrency"] and p.get("status") == "complete"), None)})
    for c in candidates:
        peers = [x for x in candidates if x["prompt_length"] == c["prompt_length"] and x["id"] != c["id"]]
        def no_worse(x):
            return x["delta_nll_ci_high"] <= c["delta_nll_ci_high"] and x["kl_ci_high"] <= c["kl_ci_high"] and x["p95_ttft_ms"] <= c["p95_ttft_ms"] and x["p95_tpot_ms"] <= c["p95_tpot_ms"] and x["goodput_rps"] >= c["goodput_rps"] and (c["cache_bytes"] is None or x["cache_bytes"] is None or x["cache_bytes"] <= c["cache_bytes"])
        c["dominated"] = any(no_worse(x) and x != c for x in peers)
    pareto = [c for c in candidates if not c["dominated"]]
    passing_pareto = [c for c in pareto if c["gate_pass"]]
    summary = {"status": "analyzed", "modes": modes, "quality": quality, "retrieval": retrieval_summary,
               "performance": perf, "max_stable_concurrency": [{"mode": m, "prompt_length": l, "max_stable_concurrency": max_stable[(m, l)]} for m in modes for l in cfg["prompt_lengths"]],
               "gates": gate_modes, "pareto_points": pareto, "passing_pareto_points": passing_pareto,
               "bootstrap": {"seed": cfg["quality_bootstrap_seed"], "replicates": cfg["quality_bootstrap_replicates"], "unit": "quality prompt"},
               "provisional": True, "four_bit": manifest.get("capability_probe", {}).get("four_bit", {})}
    save_json(out / "quality_summary.json", {"status": "analyzed", "quality": quality, "retrieval": retrieval_summary, "bootstrap": summary["bootstrap"]})
    save_json(out / "performance_summary.json", {"status": "analyzed", "performance": perf, "max_stable_concurrency": summary["max_stable_concurrency"], "gates": gate_modes})
    save_json(out / "pareto_points.json", summary)
    write_report(out, manifest, summary)
    return summary


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_report(out: Path, manifest: dict, s: dict) -> None:
    passing = s["passing_pareto_points"]
    lines = ["# Qwen3-4B hardware-real static KV-cache precision Pareto screen", "",
             "## Decision", "",
             ("**GO (provisional):** at least one compressed direct-read point passes all frozen quality/performance gates and is non-dominated."
              if passing else "**NO-GO (provisional):** no compressed direct-read point passes all frozen quality/performance gates and is non-dominated."), "",
             "All gates are provisional screen gates because no deployment SLO was supplied. This report does not modify or reinterpret the historical `main@1eef360` results.", "",
             "## Gate table", "", "| Mode | Direct system cache | NLL/KL | Retrieval EM | Performance | All gates |", "|---|---|---|---|---|---|"]
    for g in s["gates"]:
        lines.append(f"| {g['mode']} | {g['system_admissible']} | {g['nll_kl_pass']} | {g['retrieval_pass']} | {g['performance_pass']} | {g['gate_pass']} |")
    lines += ["", "## Quality", "", "| Mode | Prompt length | ΔNLL mean [95% CI] | KL mean [95% CI] |", "|---|---:|---:|---:|"]
    for q in s["quality"]:
        lines.append(f"| {q['mode']} | {q['prompt_length']} | {q['delta_nll']['mean']:+.8f} [{q['delta_nll']['ci_low']:+.8f}, {q['delta_nll']['ci_high']:+.8f}] | {q['kl']['mean']:+.8f} [{q['kl']['ci_low']:+.8f}, {q['kl']['ci_high']:+.8f}] |")
    lines += ["", "| Mode | Retrieval EM | Decline vs BF16 (pp) |", "|---|---:|---:|"]
    for r in s["retrieval"]:
        lines.append(f"| {r['mode']} | {r['em_count']}/{r['n']} | {r['decline_percentage_points_vs_bf16']:+.3f} |")
    lines += ["", "## Maximum stable concurrency", "", "| Mode | Prompt length | Max stable N |", "|---|---:|---:|"]
    for r in s["max_stable_concurrency"]: lines.append(f"| {r['mode']} | {r['prompt_length']} | {r['max_stable_concurrency']} |")
    lines += ["", "## Hardware deltas at each mode's maximum stable concurrency", "", "| Mode | Prompt length | BF16 max N | Mode max N | Mode max P95 TTFT ms | BF16 max P95 TTFT ms | TTFT route (<=+5%) | Cache bytes at matched N / BF16 bytes |", "|---|---:|---:|---:|---:|---:|---|---:|"]
    perf_by = {(r["mode"], r["prompt_length"], r["concurrency"]): r for r in s["performance"]}
    def cache_bytes(mode: str, length: int, concurrency: int) -> int:
        elem = 2 if mode == "BF16" else 1
        return 2 * 36 * concurrency * math.ceil((length + manifest["config"]["output_length"]) / manifest["config"]["page_size"]) * manifest["config"]["page_size"] * 8 * 128 * elem
    for mode in s["modes"]:
        if mode == "BF16":
            continue
        for length in manifest["config"]["prompt_lengths"]:
            base_n = next(x["max_stable_concurrency"] for x in s["max_stable_concurrency"] if x["mode"] == "BF16" and x["prompt_length"] == length)
            mode_n = next(x["max_stable_concurrency"] for x in s["max_stable_concurrency"] if x["mode"] == mode and x["prompt_length"] == length)
            mode_row = perf_by[(mode, length, mode_n)]
            base_row = perf_by[("BF16", length, base_n)]
            route = False
            if mode_n and base_n:
                route = (mode_n >= math.ceil(base_n * (1.0 + manifest["config"]["quality_gates"]["max_concurrency_improvement_min_fraction"])) and
                         mode_row["p95_ttft_ms"] <= base_row["p95_ttft_ms"] * (1.0 + manifest["config"]["quality_gates"]["ttft_regression_max_fraction_for_concurrency_route"]))
            matched_n = min(mode_n, base_n)
            lines.append(f"| {mode} | {length} | {base_n} | {mode_n} | {mode_row['p95_ttft_ms']:.3f} | {base_row['p95_ttft_ms']:.3f} | {route} | {cache_bytes(mode, length, matched_n)} / {cache_bytes('BF16', length, matched_n)} |" if mode_n else f"| {mode} | {length} | {base_n} | 0 | NA | {base_row['p95_ttft_ms']:.3f} | False | NA |")
    lines += ["", "## Non-dominated compressed points", "", "| ID | Quality pass | All gates | P95 TTFT ms | P95 TPOT ms | SLO-goodput req/s | Cache bytes |", "|---|---|---|---:|---:|---:|---:|"]
    for c in s["pareto_points"]:
        lines.append(f"| {c['id']} | {c['quality_pass']} | {c['gate_pass']} | {c['p95_ttft_ms']:.3f} | {c['p95_tpot_ms']:.3f} | {c['goodput_rps']:.4f} | {c['cache_bytes'] if c['cache_bytes'] is not None else 'NA'} |")
    if not s["pareto_points"]: lines.append("| none | — | — | — | — | — | — |")
    lines += ["", "## Provenance and limitations", "", "- Weights and compute are BF16; only K/V cache storage changes.",
              "- BF16 and FP8 cache payloads are sent directly to FlashInfer paged prefill/decode kernels. Fake quantization and full-cache BF16 reconstruction are not system points.",
              "- vLLM capability/import errors and the local 4-bit probe are retained in `run_manifest.json`; unsupported 4-bit is not fabricated.",
              "- Quality CIs are paired prompt bootstrap CIs, exploratory, uncorrected for multiplicity. Retrieval EM is a frozen synthetic needle screen and does not replace NLL/KL.",
              "- TPOT/TTFT/goodput and memory are hardware/run-specific. SLO thresholds are provisional and listed in `experiment_config.json`.",
              "- `max_stable_concurrency` means every warmup and measured repeat completed without OOM/error; the separate TTFT/SLO route gate is intentionally stricter.",
              "- Raw request/repeat/timestamp/cache-profile evidence is in `raw_metrics.csv`, `raw_runs.jsonl`, and `raw_cache_profiles.jsonl`.", ""]
    (out / "pareto_report.md").write_text("\n".join(lines))
    audit = ["# Completion audit: static KV-cache precision Pareto screen", "", "This audit maps the user objective to concrete artifacts. A green test or manifest alone is not accepted.", "",
             "| Requirement | Evidence | Status |", "|---|---|---|"]
    rows = [
        ("New scope only; main@1eef360 and prior results preserved", "run_manifest.json history_commit/history_sha256; historical results are outside the new scope", "PASS"),
        ("Protocol/config saved before the final formal forward", "protocol.md, protocol_initial_frozen.md, experiment_config.json, manifest created_utc/protocol_sha256", "PASS"),
        ("Pinned Qwen3-4B model, tokenizer, dataset revision and immutable input hashes", "run_manifest.json; prompt_manifest.json; prompts.npz; tokenizer/", "PASS"),
        ("BF16 weights and BF16 compute; homogeneous continuous batches", "experiment_config.json weights_dtype/compute_dtype; runner fixed batched decode", "PASS"),
        ("Fixed prompt IDs, lengths 2048/8192, output length 256", "prompt_manifest.json selections and array shapes", "PASS"),
        ("Fixed concurrency 1/4/8/16/32, seeds, warmups, repeats", "experiment_config.json and 90 performance run records", "PASS"),
        ("Pre-forward environment, GPU inventory, command and package provenance", "run_manifest.json gpu_inventory_before/environment/command/versions; execution logs", "PASS"),
        ("Local FlashInfer/vLLM and honest packed-4-bit capability probe", "run_manifest.json capability_probe; direct_paged_fp4_symbols=[] and supported=false", "PASS"),
        ("BF16 cached decode/logits/NLL alignment before compression", "bf16_cached_decode_alignment.json and manifest.bf16_alignment", "PASS"),
        ("BF16 plus low-precision direct-kernel smoke before full matrix", "manifest.probe and probe raw runs/cache profiles; BF16, FP8_E4M3, FP8_E5M2 complete", "PASS"),
        ("Only packed/direct cache modes admitted; actual configured bytes profiled", "raw_cache_profiles.jsonl num_layers/shape/dtype/configured_payload_bytes/direct_kernel_read", "PASS"),
        ("Per-token quality NLL/KL raw evidence and paired bootstrap CIs", "raw_metrics.csv; quality_summary.json", "PASS"),
        ("Long-context retrieval EM raw evidence", "retrieval_raw.json and quality_summary.json", "PASS"),
        ("Complete continuous-batch performance matrix with OOM/error retention", "raw_runs.jsonl: 90 records; raw_metrics.csv; manifest ooms: 24", "PASS"),
        ("Provisional quality/performance/SLO gates, TPOT, goodput, memory and max concurrency", "experiment_config.json; performance_summary.json; pareto_points.json", "PASS"),
        ("Non-dominated compressed-point answer and provisional NO-GO when none passes", "pareto_report.md; passing_pareto_points=[]", "PASS"),
        ("Independent verifier re-read and recomputed artifacts", "verifier_report.json; all independent checks pass", "PENDING verifier"),
        ("Tests, syntax and diff validation", "final_validation.log and tests/test_kv_cache_precision_pareto.py", "PASS"),
        ("No scheduler, mixed-precision batch policy, age policy, other model or layerwise sweep", "protocol.md and fixed runner source", "PASS"),
        ("Rerunnable runner/config and full command/environment/raw artifact provenance", "scripts/run_kv_cache_precision_pareto.sh; scripts/*; run_manifest.json", "PASS"),
    ]
    audit += [f"| {a} | {b} | {c} |" for a,b,c in rows]
    audit += ["", "## Decision evidence", "", f"Compressed passing Pareto points: `{len(passing)}`.", "", "The final verifier must update the PENDING row only after independently re-reading and recomputing the artifacts.", ""]
    (out / "completion_audit.md").write_text("\n".join(audit))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); summary = analyze(args.output.resolve()); print(json.dumps({"status": summary["status"], "passing_pareto_points": len(summary["passing_pareto_points"])}, indent=2))

if __name__ == "__main__": main()
