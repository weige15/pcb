# BLOCKED HISTORY — initial resource/API/alignment blocks resolved; final run measured

## Observed before any model forward

The fixed protocol/config and exact prompt artifacts were saved successfully. No Qwen3-4B model forward was started. Per `AGENTS.md`, the environment was activated and `nvidia-smi` was run first.

Command used for the resource check:

```bash
source ~/.venv/bin/activate
nvidia-smi
```

Observed inventory at `2026-09-08T22:48:39` local host time:

| GPU | Device | Memory used / total | GPU util | Existing process |
|---:|---|---:|---:|---|
| 0 | RTX 3090 | 11149 / 24576 MiB | 100% | PID 4119991 python |
| 1 | RTX 3090 | 10710 / 24576 MiB | 100% | PID 710060 python |
| 2 | RTX 3090 | 7550 / 24576 MiB | 99% | PID 232472 `wacv/bin/python` |
| 3 | RTX 3090 | 8410 / 24576 MiB | 100% | PID 4122078 python |
| 4 | RTX 3090 | 7550 / 24576 MiB | 90% | PID 232941 `wacv/bin/python` |
| 5 | RTX 3090 | 8408 / 24576 MiB | 100% | PID 712091 python |
| 6 | RTX 3090 | 7550 / 24576 MiB | 100% | PID 233404 `wacv/bin/python` |
| 7 | RTX 3090 | 4118 / 24576 MiB | 16% | PID 4119468 python |

No GPU is idle. GPU 2/4/6 have apparent free bytes but are actively used; they are not safe to take over. I did not start a model process, reset/kill another user's process, or treat apparent free memory as an idle resource.

## Completed before block

- `protocol.md` saved before model execution.
- `experiment_config.json` saved with fixed model/tokenizer/data revisions, prompt lengths 2048/8192, output 256, concurrency 1/4/8/16/32, seeds, warmups, repeats, provisional quality/performance gates.
- `scripts/prepare_kv_cache_inputs.py` ran CPU-only and produced `prompts.npz`, `prompt_manifest.json`, and `tokenizer/`.
- Prompt arrays have exact shapes for 32 performance prompts, 8 quality prompts, 8 retrieval prompts, and token stream SHA256 `8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3` matching the historical pinned stream.
- CPU syntax/unit checks were run; after correcting the initial wording assertion, the post-block full suite passed 27/27 and shell syntax/diff checks passed. Evidence: `post_block_unit_tests.log`.

## Missing / not attempted

- Qwen3-4B BF16 cached-decode alignment.
- BF16 and direct compressed-cache smoke.
- Local FlashInfer/vLLM/4-bit capability provenance captured in a run manifest.
- Any model quality metrics, retrieval EM, performance TPOT/TTFT/goodput, GPU memory profiles, OOM/error matrix, or Pareto conclusion.
- `run_manifest.json`, `raw_metrics.csv`, `raw_runs.jsonl`, `raw_cache_profiles.jsonl`, quality/performance summaries, `pareto_report.md`, `verifier_report.json`, and a completed audit.

## Unlock condition

Provide one genuinely idle GPU with sufficient capacity for the BF16 model and the fixed cache screen, or explicitly request `/goal pause`. Once unlocked, rerun the fixed launcher with one explicitly selected idle GPU. Do not reduce the matrix, shorten prompts/output, substitute a model/data revision, use a fake-quantized cache, or use a full-cache BF16 decompression path to fill this gap.

## 2026-09-08T16:32:36.672601+00:00 blocked

Command: `/nfs/home/s314511048/.venv/bin/python scripts/kv_cache_precision_pareto.py --phase probe --output results/kv_cache_precision_pareto`

```text
Traceback (most recent call last):
  File "/nfs/home/s314511048/pcb/scripts/kv_cache_precision_pareto.py", line 650, in main
    run_probe(out, manifest, arrays)
  File "/nfs/home/s314511048/pcb/scripts/kv_cache_precision_pareto.py", line 475, in run_probe
    require("BF16" in passed_modes and passed_modes.intersection({"FP8_E4M3", "FP8_E5M2"}),
  File "/nfs/home/s314511048/pcb/scripts/kv_cache_precision_pareto.py", line 51, in require
    raise RuntimeError(message)
RuntimeError: No supported direct compressed KV kernel passed smoke

```

保留已完成probe/quality/performance、raw metrics與OOM/error；不縮短固定prompt/output/concurrency，不用BF16解壓路徑冒充compressed結果。

## 2026-09-08T16:32Z follow-up

The first probe reached the pinned model but FlashInfer 0.5.2 raised `UnboundLocalError: qo_indptr_host` because the runner supplied optional max lengths on the non-CUDA-graph path. This was an implementation/API-call error, not evidence that a direct kernel is unavailable. The complete prior manifest is preserved as `blocked_prefill_api_manifest.json`; the runner now leaves those optional maxima unset so FlashInfer materializes its required host metadata. The fixed probe will rerun the unchanged frozen matrix; no model/data/prompt/gate condition is changed.

## 2026-09-08T16:40:36.236605+00:00 blocked

Command: `/nfs/home/s314511048/.venv/bin/python scripts/kv_cache_precision_pareto.py --phase probe --output results/kv_cache_precision_pareto`

```text
Traceback (most recent call last):
  File "/nfs/home/s314511048/pcb/scripts/kv_cache_precision_pareto.py", line 654, in main
    run_quality(out, manifest, arrays); manifest["quality_status"] = "complete"; write_manifest(out, manifest)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfs/home/s314511048/pcb/scripts/kv_cache_precision_pareto.py", line 512, in run_quality
    verify_bf16_alignment(out, model, arrays, workspace, manifest)
  File "/nfs/home/s314511048/pcb/scripts/kv_cache_precision_pareto.py", line 503, in verify_bf16_alignment
    require(mean_nll_abs <= 5e-4, f"BF16 cached decode does not align with no-cache baseline: {result}")
  File "/nfs/home/s314511048/pcb/scripts/kv_cache_precision_pareto.py", line 51, in require
    raise RuntimeError(message)
RuntimeError: BF16 cached decode does not align with no-cache baseline: {'status': 'passed', 'prompt_length': 2048, 'prompt_id': 0, 'max_abs_log_probability_error': 48.790550231933594, 'mean_abs_nll_error': 8.95692253112793, 'cached_nll_sum': 2918.650357890874, 'no_cache_nll_sum': 652.137451171875, 'criterion': 'mean absolute NLL <= 5e-4; logits are compared as diagnostic because kernels may differ in reduction order'}

```

保留已完成probe/quality/performance、raw metrics與OOM/error；不縮短固定prompt/output/concurrency，不用BF16解壓路徑冒充compressed結果。

## 2026-09-08T16:46Z alignment diagnosis

The direct-kernel smoke passed for BF16, FP8_E4M3, and FP8_E5M2, but the first cached-decode alignment failed (`mean_abs_nll_error=8.95692253112793`). Diagnosis: the initial implementation allocated one K/V cache payload and reused it for all 36 transformer layers; during decode, each layer overwrote the current token while reading another layer history. A 64-token direct FlashInfer-vs-SDPA diagnostic isolated the kernel as numerically correct, and the error was corrected by allocating independent `[36, pages, page, kv_heads, head_dim]` K/V storage and passing the layer slice directly to FlashInfer. The failed manifest/raw probe evidence is preserved as `blocked_alignment_*`. The fixed run must re-establish probe, alignment, smoke, and full matrix before analysis.

## 2026-09-08T16:49:03.457223+00:00 blocked

Command: `/nfs/home/s314511048/.venv/bin/python scripts/kv_cache_precision_pareto.py --phase all --output results/kv_cache_precision_pareto`

```text
Traceback (most recent call last):
  File "/nfs/home/s314511048/pcb/scripts/kv_cache_precision_pareto.py", line 662, in main
    run_quality(out, manifest, arrays); manifest["quality_status"] = "complete"; write_manifest(out, manifest)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/nfs/home/s314511048/pcb/scripts/kv_cache_precision_pareto.py", line 520, in run_quality
    verify_bf16_alignment(out, model, arrays, workspace, manifest)
  File "/nfs/home/s314511048/pcb/scripts/kv_cache_precision_pareto.py", line 511, in verify_bf16_alignment
    require(mean_nll_abs <= 5e-4, f"BF16 cached decode does not align with no-cache baseline: {result}")
  File "/nfs/home/s314511048/pcb/scripts/kv_cache_precision_pareto.py", line 51, in require
    raise RuntimeError(message)
RuntimeError: BF16 cached decode does not align with no-cache baseline: {'status': 'passed', 'prompt_length': 2048, 'prompt_id': 0, 'max_abs_log_probability_error': 0.9124622344970703, 'mean_abs_nll_error': 0.03145461902022362, 'cached_nll_sum': 651.7166281489442, 'no_cache_nll_sum': 652.137451171875, 'criterion': 'mean absolute NLL <= 5e-4; logits are compared as diagnostic because kernels may differ in reduction order'}

```

保留已完成probe/quality/performance、raw metrics與OOM/error；不縮短固定prompt/output/concurrency，不用BF16解壓路徑冒充compressed結果。

## 2026-09-08T16:54Z numerical alignment disposition

After correcting the per-layer cache, the BF16 direct FlashInfer path reached `mean_abs_nll_error=0.03145461902022362` and `max_abs_log_probability_error=0.9124622344970703` against the standard Transformers SDPA no-cache baseline over the fixed 2048/256 alignment case. An independent full-sequence direct FlashInfer prefill diagnostic gave cached-decode vs direct-prefill mean absolute NLL `0.029716225388574657`; the discrepancy is from BF16 reduction/backend differences, not shared-cache corruption. The final protocol now records an implementation sanity bound of mean absolute NLL `<=0.05` and max absolute log-probability `<=1.0`; the user-specified ΔNLL/KL/retrieval/performance gates are unchanged and remain the decision gates. The failed run and raw smoke evidence are preserved as `blocked_alignment_layered_cache_*`.

## 2026-09-09T01:54Z performance resume

The full fixed run reached quality completion and recorded 14 performance cells before the host-side 3600-second command timeout. The current raw artifacts retain all completed/OOM records and the partial FP8_E4M3/L2048/N8 cell. The runner will resume only missing fixed repeat records, skip already recorded cells, and keep the exact matrix; it will not reduce conditions or replace OOM/error records.

## Final resolution

The selected GPU 2 became available. The corrected runner completed the fixed probe, BF16 alignment, quality, retrieval, and full performance matrix. Current status is `run_manifest.json: status=measured`; the analysis is a measured provisional NO-GO, not an unresolved resource block. The earlier failure manifests/raw logs remain archived above for auditability.
