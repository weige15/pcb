#!/usr/bin/env python3
"""Rebuild layerwise tables/paired bootstrap/concentration figures from real completed conditions."""
import argparse
from collections import Counter
import csv
import io
import importlib.metadata
import json
import math
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import layerwise_projection_sensitivity as lw
from scripts.projection_quantization_sensitivity import paired_bootstrap

DRAW_SHAPE = (2000, 64)


def validate_producing_attempt(attempt, condition):
    # Individual commits already passed full-state and predictive restoration.
    # A later failed/interrupted condition must not invalidate those commits or
    # force their remeasurement. Preserve the failed attempt's original status.
    lw.require(attempt["status"] in ("complete", "bounded_partial", "blocked", "running") and
               attempt["original_state_matches_history"] is True and attempt["state_tensors"] == 400 and
               attempt["tied_lm_head"] is True and attempt["runtime_attention_backend"] == "sdpa",
               "Producing attempt identity controls incomplete")
    lw.require(condition not in attempt["skipped_completed"], "Condition was not produced in this attempt")
    if attempt["status"] in ("complete", "bounded_partial"):
        lw.require(attempt["final_state_matches_checkpoint"] is True and condition in attempt["newly_completed"],
                   "Successful producing attempt has inconsistent final controls")


def projection_order(values):
    ordered = sorted(values, key=values.get, reverse=True)
    text = ordered[0]
    for previous, current in zip(ordered, ordered[1:]):
        text += (" = " if values[previous] == values[current] else " > ") + current
    return text


def validated_results(out):
    manifest = json.loads((out / "run_manifest.json").read_text())
    history = lw.verify_inputs(out, manifest)
    lw.require(manifest["status"] == "measured" and set(manifest["completed_conditions"]) == set(lw.CONDITIONS) and
               set(manifest["condition_sha256"]) == set(lw.CONDITIONS), "Only fully measured runs can produce a report")
    results = lw.completed_results(out, manifest, complete=True)
    cache = results["BF16"]["hidden_cache"]
    lw.require(lw.file_hash(out / cache["path"]) == cache["sha256"], "Baseline hidden cache changed")
    for result in results.values():
        path = result["attempt"]
        lw.require(path in manifest["attempts"], "Condition missing producing attempt")
        attempt = json.loads((out / path).read_text())
        validate_producing_attempt(attempt, result["condition"])
        for p, version in attempt["versions"].items():
            lw.require(history["versions"][p] == version, "Runtime drift")
    return manifest, results


def concentration(scores):
    """Descriptive distribution of nonnegative scores, not additive quantization damage."""
    scores = np.asarray(scores, dtype=np.float64)
    lw.require(scores.shape == (36,) and np.isfinite(scores).all() and (scores >= 0).all(), "Invalid concentration scores")
    ranking = np.argsort(-scores, kind="stable")
    total = scores.sum()
    record = dict(layer_scores=scores.tolist(), ranked_layers=ranking.tolist(), total_score=float(total))
    if total == 0:
        return dict(record, cumulative_share=None, top1_share=None, top4_share=None, top8_share=None,
                    layers_for_50pct=None, layers_for_80pct=None, effective_layers=None)
    cumulative = np.cumsum(scores[ranking]) / total
    return dict(record, cumulative_share=cumulative.tolist(), top1_share=float(cumulative[0]),
                top4_share=float(cumulative[3]), top8_share=float(cumulative[7]),
                layers_for_50pct=int(np.searchsorted(cumulative, .5) + 1),
                layers_for_80pct=int(np.searchsorted(cumulative, .8) + 1),
                effective_layers=float(total ** 2 / (scores ** 2).sum()))


def concentration_ci(resampled_scores):
    total = resampled_scores.sum(axis=1)
    valid = total > 0
    if not valid.any():
        return dict(top4_share_ci=None, effective_layers_ci=None, defined_resamples=0)
    scores = resampled_scores[valid]
    top4 = np.sort(scores, axis=1)[:, -4:].sum(axis=1) / total[valid]
    effective = total[valid] ** 2 / (scores ** 2).sum(axis=1)
    return dict(top4_share_ci=np.quantile(top4, [.025, .975]).tolist(),
                effective_layers_ci=np.quantile(effective, [.025, .975]).tolist(), defined_resamples=int(valid.sum()))


def summarize(results, draws):
    summary, paired, concentrations = [], [], []
    for c in lw.CONDITIONS:
        result = results[c]
        values = {k: np.array([r[k] for r in result["rows"]], dtype=np.float64) for k in ("nll", "delta_nll", "kl")}
        layer, projection, bits = lw.identity(c)
        row = dict(condition=c, layer=layer, projection=projection, bits=bits, context=2048,
                   blocks=64, scored_tokens=131072, nll=float(values["nll"].mean()), ppl=math.exp(float(values["nll"].mean())))
        for metric in ("delta_nll", "kl"):
            mean, lo, hi = paired_bootstrap(values[metric], np.zeros(64), draws)
            row.update({metric: mean, metric + "_ci_low": lo, metric + "_ci_high": hi})
        summary.append(row)
    for layer in range(36):
        for a, b in (("WQ", "WK"), ("WQ", "WV"), ("WK", "WV")):
            for metric in ("delta_nll", "kl"):
                x = [r[metric] for r in results[f"L{layer:02d}_{a}4"]["rows"]]
                y = [r[metric] for r in results[f"L{layer:02d}_{b}4"]["rows"]]
                mean, lo, hi = paired_bootstrap(x, y, draws)
                paired.append(dict(layer=layer, contrast=a + " - " + b, metric=metric, mean=mean, ci_low=lo, ci_high=hi))
    for metric in ("delta_nll", "kl"):
        all_scores, all_boot = [], []
        for projection in lw.PROJECTIONS:
            values = np.array([[r[metric] for r in results[f"L{i:02d}_{projection}4"]["rows"]] for i in range(36)])
            # Clip only the derived layer score, never raw NLL/KL measurements.
            scores = np.maximum(values.mean(axis=1), 0)
            boot = np.maximum(values[:, draws].mean(axis=2).T, 0)
            all_scores.append(scores)
            all_boot.append(boot)
            concentrations.append(dict(projection=projection, metric=metric,
                negative_mean_layers=np.flatnonzero(values.mean(axis=1) < 0).tolist(),
                **concentration(scores), **concentration_ci(boot)))
        concentrations.append(dict(projection="ALL", metric=metric, negative_mean_layers=[],
            **concentration(np.sum(all_scores, axis=0)), **concentration_ci(np.sum(all_boot, axis=0))))
    return dict(summary=summary, paired_comparisons=paired, concentration=concentrations,
                bootstrap=dict(seed=42, replicates=2000, unit="same selected block indices across all conditions",
                               ci="exploratory percentile 95%; no multiplicity correction; FP64 statistics on FP32 block measurements",
                               concentration="rerank layers in every shared-block bootstrap draw; undefined draws excluded and counted"),
                concentration_definition="max(mean block metric, 0) per layer/projection; ALL sums three independently intervened scores, not joint damage")


def csv_text(rows, fields=None):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def figures(out, analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2,
                         "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"WQ": "#0072B2", "WK": "#E69F00", "WV": "#009E73", "ALL": "#444444"}
    markers = {"WQ": "o", "WK": "s", "WV": "^", "ALL": "D"}
    labels = {"delta_nll": r"$\Delta$NLL vs BF16 (nats/token)", "kl": "KL(BF16 || test) (nats/token)"}
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    generated = []

    def export(fig, name):
        for ext in ("png", "pdf"):
            path = folder / (name + "." + ext)
            metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {"Software": "layerwise_projection_sensitivity"}
            fig.savefig(path, dpi=300, bbox_inches="tight", metadata=metadata)
            generated.append(str(path.relative_to(out)))
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), layout="constrained")
    for ax, metric in zip(axes, ("delta_nll", "kl")):
        for p in lw.PROJECTIONS:
            rows = [r for r in analysis["summary"] if r["projection"] == p]
            x = [r["layer"] for r in rows]
            ax.plot(x, [r[metric] for r in rows], label=p, color=colors[p], marker=markers[p], markersize=3, linewidth=1.3)
            ax.fill_between(x, [r[metric + "_ci_low"] for r in rows], [r[metric + "_ci_high"] for r in rows], color=colors[p], alpha=.13)
        ax.axhline(0, color="black", linestyle=":", linewidth=.8)
        ax.set(xlabel="Intervened layer (0-based)", ylabel=labels[metric], xticks=[0, 5, 10, 15, 20, 25, 30, 35])
        ax.legend(ncol=3)
    fig.suptitle("Qwen3-4B: one projection weight at RTN4; 64 blocks × 2048 predictions\nBands: exploratory paired-block 95% CIs (not multiplicity-adjusted)", fontsize=11)
    export(fig, "layerwise_metrics")

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, layout="constrained")
    for row_idx, contrast in enumerate(("WQ - WK", "WQ - WV", "WK - WV")):
        for col_idx, metric in enumerate(("delta_nll", "kl")):
            ax = axes[row_idx, col_idx]
            rows = [r for r in analysis["paired_comparisons"] if r["contrast"] == contrast and r["metric"] == metric]
            ax.plot(range(36), [r["mean"] for r in rows], color="#0072B2", marker="o", markersize=3, linewidth=1.2)
            ax.fill_between(range(36), [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], color="#0072B2", alpha=.18)
            ax.axhline(0, color="black", linestyle="--", linewidth=.8)
            ax.set(title=f"{contrast}: {'ΔNLL' if metric == 'delta_nll' else 'KL'}", ylabel="Paired difference (nats/token)", xticks=[0, 5, 10, 15, 20, 25, 30, 35])
            if row_idx == 2:
                ax.set_xlabel("Intervened layer (0-based)")
    fig.suptitle("Same-layer Q/K/V paired comparisons; exploratory 95% CIs", fontsize=12)
    export(fig, "paired_layerwise_differences")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, metric in zip(axes, ("delta_nll", "kl")):
        for r in analysis["concentration"]:
            if r["metric"] == metric and r["cumulative_share"] is not None:
                p = r["projection"]
                ax.plot(range(1, 37), r["cumulative_share"], label=p, color=colors[p], marker=markers[p],
                        markevery=[0, 3, 7, 15, 23, 35], markersize=4, linewidth=1.4)
        ax.plot([0, 36], [0, 1], color="#888888", linestyle="--", linewidth=.9, label="Uniform")
        ax.axvline(4, color="#999999", linestyle=":", linewidth=.8)
        ax.axhline(.5, color="#999999", linestyle=":", linewidth=.8)
        ax.set(xlabel="Number of highest-score layers (reranked per curve)", ylabel="Cumulative score share",
               title="Positive mean ΔNLL" if metric == "delta_nll" else "Nonnegative mean KL", xlim=(0, 36), ylim=(0, 1.02))
        ax.legend(ncol=2, fontsize=8, loc="lower right")
    fig.suptitle("Concentration of single-weight effects (ALL is a score sum, not joint quantization)", fontsize=11)
    export(fig, "layer_concentration")
    return generated


def report(out, analysis, manifest):
    summary, pairs = analysis["summary"], analysis["paired_comparisons"]
    by_condition = {r["condition"]: r for r in summary}
    base = by_condition["BF16"]
    attempts = [json.loads((out / p).read_text()) for p in manifest["attempts"]]
    gpu = shlex.quote(attempts[0]["environment"]["CUDA_VISIBLE_DEVICES"])
    prefix = f"CUDA_VISIBLE_DEVICES={gpu} bash scripts/run_layerwise_projection_sensitivity.sh"
    resume = f"{prefix} {shlex.quote(str(out))} --resume"
    rerun = f"{prefix} {shlex.quote(str(out))}/rerun_$(date -u +%Y%m%dT%H%M%SZ)"
    lines = ["# Qwen3-4B 逐層 projection-weight 敏感度", "", "## 結果與範圍", "",
        "**109/109 真實條件 = BF16 + 108 次單層/單projection RTN4介入；6976逐block列，每條件64 blocks ×2048預測位置。** 層號均0-based。",
        f"BF16 NLL **{base['nll']:.9f}**、PPL **{base['ppl']:.6f}**；本輪64個block的BF16 NLL與歷史2048條件逐筆完全相同。",
        "沿用原checkpoint/tokenizer、相同64 blocks、row內group128對稱RTN、FP32 scale與metric reduction，dequant後BF16前向。KL方向為 `P_BF16 || P_test`。",
        "每次只改一個weight；沒有量化activations或KV。GQA的Q參數量為K/V四倍，不是等參數或等成本比較。沒有packed low-bit GEMM/加速/記憶體壓縮宣稱。", "",
        "## 是否集中在少數層？", "",
        "下表是**描述性集中度**：先算每層64 blocks的平均，再取非負分數 `max(mean metric,0)`；原始負ΔNLL仍完整保留，沒有刪掉或改写測量。",
        "ALL把同一層Q/K/V三個獨立介入的分數相加，只用於layer排行，**不是三者聯合介入的測量，不能假設損害可加**。",
        "預先固定top4（36層約11%）占比≥50%為『高度集中』的描述性參考，不是假設檢定/普遍門檻。effective layers=(Σs)²/Σs²；均勻36層時為36。",
        "top4 CI每次以相同block索引重抽全部層並重新排名，不固定事後挑出的top層。", "",
        "| 分數 | projection | top1占比 | top4占比 [95% CI] | top8占比 | 50%/80%所需層數 | effective layers [95% CI] | top4層 |",
        "|---|---|---:|---|---:|---|---|---|"]
    for r in analysis["concentration"]:
        if r["total_score"] == 0:
            lines.append(f"| {r['metric']} | {r['projection']} | undefined | undefined | undefined | undefined | undefined | 無正分數 |")
            continue
        lo, hi = r["top4_share_ci"]
        elo, ehi = r["effective_layers_ci"]
        lines.append(f"| {r['metric']} | {r['projection']} | {r['top1_share']:.1%} | {r['top4_share']:.1%} [{lo:.1%}, {hi:.1%}] | {r['top8_share']:.1%} | {r['layers_for_50pct']} / {r['layers_for_80pct']} | {r['effective_layers']:.2f} [{elo:.2f}, {ehi:.2f}] | {', '.join(map(str, r['ranked_layers'][:4]))} |")
    lines += ["", "### 有限結論", ""]
    for metric in ("delta_nll", "kl"):
        r = next(r for r in analysis["concentration"] if r["projection"] == "ALL" and r["metric"] == metric)
        if r["total_score"] > 0:
            verdict = "符合預先固定的高度集中描述性參考" if r["top4_share"] >= .5 else "不符合預先固定的高度集中描述性參考"
            lines.append(f"- ALL {metric}：top4層 {r['ranked_layers'][:4]} 占 {r['top4_share']:.1%}，{verdict}；達80%需 {r['layers_for_80pct']} 層。不能只憑均值top層宣稱跨資料可重現。")
        for p in lw.PROJECTIONS:
            rows = sorted([r for r in summary if r["projection"] == p], key=lambda r: r[metric], reverse=True)
            lines.append(f"- {p} {metric} 最大4層：" + "; ".join(f"L{r['layer']:02d}={r[metric]:+.7f}" for r in rows[:4]) + "。")
        orderings = Counter(projection_order({p: by_condition[f"L{i:02d}_{p}4"][metric] for p in lw.PROJECTIONS}) for i in range(36))
        lines.append(f"- {metric} 的36層Q/K/V均值排序頻數：" + "；".join(f"{order}: {count}層" for order, count in sorted(orderings.items())) + "。這是描述性排序，不是每層差異已被確認。")
        selected_pairs = [r for r in pairs if r["metric"] == metric]
        crossing = sum(r["ci_low"] <= 0 <= r["ci_high"] for r in selected_pairs)
        lines.append(f"- {metric} 的108個同層Q/K/V配對CI有 {crossing} 個跨零；跨零不是證明等價，不追求一致排序或追加條件。")
    negative = [r for r in summary[1:] if r["delta_nll"] < 0]
    lines += [f"- {len(negative)}/108 介入平均ΔNLL為負。這是固定語料上的局部改善/數值效應，不能解釋成普遍有益；KL和ΔNLL須分開判讀。", "",
        "![逐層NLL與KL](figures/layerwise_metrics.png)", "![配對比較](figures/paired_layerwise_differences.png)",
        "![集中度](figures/layer_concentration.png)", "", "## 完整逐層摘要", "",
        "單位nats/token；表中NLL是所有131072預測位置的平均（由等長64 block NLL以FP64彙總），PPL=exp(mean NLL)，不平均block PPL。",
        "全部108介入的vs-BF16探索性CI在 `layer_summary.csv`，全部216個Q/K/V contrast/metric在 `paired_comparisons.csv`。", "",
        "| layer | projection | NLL | PPL | ΔNLL [95% CI] | KL [95% CI] |", "|---:|---|---:|---:|---|---|"]
    for r in summary:
        lines.append(f"| {r['layer']} | {r['projection']} | {r['nll']:.8f} | {r['ppl']:.6f} | {r['delta_nll']:+.8f} [{r['delta_nll_ci_low']:+.8f}, {r['delta_nll_ci_high']:+.8f}] | {r['kl']:.8f} [{r['kl_ci_low']:.8f}, {r['kl_ci_high']:.8f}] |")
    lines += ["", "## 控制、續跑與可重現證據", "",
        "- 首末block ×首末64位置chunk logits與CausalLM API bitwise一致；shifted NLL核對直接cross entropy。",
        "- 每次全400個parameters/buffers的uint8 bitwise比較：原始入口、剛介入、eval後、還原後。改變差集恰一個指定weight，108次皆通過。原始400個SHA256在每次載入都等於歷史checkpoint。",
        "- 每個還原另驗首末block hidden bitwise相同、NLL相同、KL=0；各條件JSON記錄module/shape/參數數/relative L2/原始與量化hash及全部控制。",
        "- `conditions/*.json` 是原子提交的原始條件；`attempts/` 保留執行環境、時間、命令、每8 blocks的partial和BF16 lossless cache；partial不是額外樣本或完成條件。",
        "- `run_manifest.json` 保存設定、來源快照hash、唯讀歷史hash、完成條件hash和實際attempt路徑。`source/scripts/` 保留產生本輪數據的runner，不修改歷史來源/數據。",
        "- resume不重跑已提交條件。只有109條件的每條控制與來源全部有效才可產出本報告。若某attempt在後續條件失敗，保留其blocked狀態，不因此重跑之前已原子提交且通過還原的條件。",
        "- 所有圖由保存的真實數值生成，同名PDF提供向量版；bootstrap draws是seed42、2000×64相同block索引。", "",
        "```bash", "source ~/.venv/bin/activate", "nvidia-smi", "# 選擇空閒GPU；新目錄重跑全部條件：",
        rerun, "# 上面沿用本次GPU UUID；執行前必須確認仍空閒，否則指定另一張足夠空閒GPU。",
        "# 中斷後只補未完成條件（先解決資源/錯誤，不刪舊結果）：",
        resume, "# 不執行model forward，重算表格與圖：",
        f"python scripts/analyze_layerwise_projection_sensitivity.py --output {shlex.quote(str(out))}",
        f"python scripts/verify_layerwise_projection_sensitivity.py --output {shlex.quote(str(out))}", "```", "",
        "## 限制", "",
        "- 僅限本Qwen3-4B checkpoint、固定WikiText-2 raw test的64 blocks、2048預測位置與group128 RTN4；沒有新的資料seed、獨立replicate或跨硬體精確重現保證。",
        "- 相鄰文本blocks可能相關；2000 bootstrap只描述此有限block樣本的探索性不確定性。108介入的baseline比較及216個projection contrasts未做多重比較校正。",
        "- BF16數值擾動可能影響微小KL與ΔNLL；負ΔNLL和不同metric排序不能單獨作機制或普適因果推論。",
        "- 集中度是獨立介入分數的描述，不估計同時量化多層的可加性、scheduler策略或serving收益。top層身份為本樣本事後排序，不作確認性結論。",
        "- 沒有擴展到scheduler、KV/activation量化、prefill/decode、其他bit或模型；接受無一致排序。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=lw.OUTPUT)
    args = parser.parse_args()
    out = args.output.resolve()
    manifest, results = validated_results(out)
    draws = np.random.default_rng(42).integers(0, 64, size=DRAW_SHAPE)
    analysis = summarize(results, draws)
    np.save(out / "paired_bootstrap_indices.npy", draws)
    lw.save_json(out / "sensitivity_analysis.json", analysis)
    (out / "block_metrics.csv").write_text(csv_text([r for c in lw.CONDITIONS for r in results[c]["rows"]], lw.FIELDS))
    (out / "layer_summary.csv").write_text(csv_text(analysis["summary"]))
    (out / "paired_comparisons.csv").write_text(csv_text(analysis["paired_comparisons"]))
    lw.save_json(out / "concentration.json", analysis["concentration"])
    plots = figures(out, analysis)
    (out / "sensitivity_report.md").write_text(report(out, analysis, manifest))
    source = Path(__file__).resolve()
    shutil.copyfile(source, out / "source/scripts" / source.name)
    artifacts = ["block_metrics.csv", "layer_summary.csv", "paired_comparisons.csv", "concentration.json",
                 "paired_bootstrap_indices.npy", "sensitivity_analysis.json", "sensitivity_report.md", *plots]
    lw.save_json(out / "analysis_manifest.json", dict(
        source_sha256={str(source.relative_to(ROOT)): lw.file_hash(source)},
        condition_sha256=manifest["condition_sha256"], artifacts_sha256={n: lw.file_hash(out / n) for n in artifacts},
        conditions=109, interventions=108, rows=6976, paired_contrasts=216, figures=plots,
        command=shlex.join([sys.executable, *sys.argv]), python=sys.version,
        versions={p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "pillow", "torch")},
        plotting_backend="Agg", figure_reproducibility="Numerical data/plot source pinned; byte identity tested only in this environment"))
    print("ANALYSIS PASS: 109 conditions, 6976 rows, 216 paired contrasts, 8 concentration profiles, 6 PNG/PDF figures")


if __name__ == "__main__":
    main()
