# Completion audit: static KV-cache precision Pareto screen

This audit maps the user objective to concrete artifacts. A green test or manifest alone is not accepted.

| Requirement | Evidence | Status |
|---|---|---|
| New scope only; main@1eef360 and prior results preserved | run_manifest.json history_commit/history_sha256; historical results are outside the new scope | PASS |
| Protocol/config saved before the final formal forward | protocol.md, protocol_initial_frozen.md, experiment_config.json, manifest created_utc/protocol_sha256 | PASS |
| Pinned Qwen3-4B model, tokenizer, dataset revision and immutable input hashes | run_manifest.json; prompt_manifest.json; prompts.npz; tokenizer/ | PASS |
| BF16 weights and BF16 compute; homogeneous continuous batches | experiment_config.json weights_dtype/compute_dtype; runner fixed batched decode | PASS |
| Fixed prompt IDs, lengths 2048/8192, output length 256 | prompt_manifest.json selections and array shapes | PASS |
| Fixed concurrency 1/4/8/16/32, seeds, warmups, repeats | experiment_config.json and 90 performance run records | PASS |
| Pre-forward environment, GPU inventory, command and package provenance | run_manifest.json gpu_inventory_before/environment/command/versions; execution logs | PASS |
| Local FlashInfer/vLLM and honest packed-4-bit capability probe | run_manifest.json capability_probe; direct_paged_fp4_symbols=[] and supported=false | PASS |
| BF16 cached decode/logits/NLL alignment before compression | bf16_cached_decode_alignment.json and manifest.bf16_alignment | PASS |
| BF16 plus low-precision direct-kernel smoke before full matrix | manifest.probe and probe raw runs/cache profiles; BF16, FP8_E4M3, FP8_E5M2 complete | PASS |
| Only packed/direct cache modes admitted; actual configured bytes profiled | raw_cache_profiles.jsonl num_layers/shape/dtype/configured_payload_bytes/direct_kernel_read | PASS |
| Per-token quality NLL/KL raw evidence and paired bootstrap CIs | raw_metrics.csv; quality_summary.json | PASS |
| Long-context retrieval EM raw evidence | retrieval_raw.json and quality_summary.json | PASS |
| Complete continuous-batch performance matrix with OOM/error retention | raw_runs.jsonl: 90 records; raw_metrics.csv; manifest ooms: 24 | PASS |
| Provisional quality/performance/SLO gates, TPOT, goodput, memory and max concurrency | experiment_config.json; performance_summary.json; pareto_points.json | PASS |
| Non-dominated compressed-point answer and provisional NO-GO when none passes | pareto_report.md; passing_pareto_points=[] | PASS |
| Independent verifier re-read and recomputed artifacts | verifier_report.json; all independent checks pass | PASS |
| Tests, syntax and diff validation | final_validation.log, completion_audit_evidence.log, and tests/test_kv_cache_precision_pareto.py | PASS |
| No scheduler, mixed-precision batch policy, age policy, other model or layerwise sweep | protocol.md and fixed runner source | PASS |
| Rerunnable runner/config and full command/environment/raw artifact provenance | scripts/run_kv_cache_precision_pareto.sh; scripts/*; run_manifest.json | PASS |

## Decision evidence

Compressed passing Pareto points: `0`.

Independent verifier completed: source/history/input hashes, matrix completeness, cache bytes, provenance, quality summaries, and retrieval summary were independently recomputed.
