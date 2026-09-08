# Qwen3-4B hardware-real static KV-cache precision Pareto screen

## Decision

**NO-GO (provisional):** no compressed direct-read point passes all frozen quality/performance gates and is non-dominated.

All gates are provisional screen gates because no deployment SLO was supplied. This report does not modify or reinterpret the historical `main@1eef360` results.

## Gate table

| Mode | Direct system cache | NLL/KL | Retrieval EM | Performance | All gates |
|---|---|---|---|---|---|
| BF16 | True | True | True | False | False |
| FP8_E4M3 | True | False | True | False | False |
| FP8_E5M2 | True | False | True | False | False |

## Quality

| Mode | Prompt length | ΔNLL mean [95% CI] | KL mean [95% CI] |
|---|---:|---:|---:|
| BF16 | 2048 | +0.00000000 [+0.00000000, +0.00000000] | +0.00000000 [+0.00000000, +0.00000000] |
| BF16 | 8192 | +0.00000000 [+0.00000000, +0.00000000] | +0.00000000 [+0.00000000, +0.00000000] |
| FP8_E4M3 | 2048 | -0.00418213 [-0.01490048, +0.00581819] | +0.01373672 [+0.00866426, +0.01960580] |
| FP8_E4M3 | 8192 | +0.00299052 [-0.00615346, +0.01277117] | +0.01061922 [+0.00906815, +0.01240682] |
| FP8_E5M2 | 2048 | +0.00985928 [-0.01078194, +0.03101197] | +0.03583197 [+0.02968785, +0.04341535] |
| FP8_E5M2 | 8192 | +0.00465681 [-0.01702116, +0.02574918] | +0.03233271 [+0.02637735, +0.03818504] |

| Mode | Retrieval EM | Decline vs BF16 (pp) |
|---|---:|---:|
| BF16 | 0/8 | +0.000 |
| FP8_E4M3 | 0/8 | +0.000 |
| FP8_E5M2 | 0/8 | +0.000 |

## Maximum stable concurrency

| Mode | Prompt length | Max stable N |
|---|---:|---:|
| BF16 | 2048 | 16 |
| BF16 | 8192 | 4 |
| FP8_E4M3 | 2048 | 32 |
| FP8_E4M3 | 8192 | 8 |
| FP8_E5M2 | 2048 | 32 |
| FP8_E5M2 | 8192 | 8 |

## Hardware deltas at each mode's maximum stable concurrency

| Mode | Prompt length | BF16 max N | Mode max N | Mode max P95 TTFT ms | BF16 max P95 TTFT ms | TTFT route (<=+5%) | Cache bytes at matched N / BF16 bytes |
|---|---:|---:|---:|---:|---:|---|---:|
| FP8_E4M3 | 2048 | 16 | 32 | 14903.991 | 9506.923 | False | 2717908992 / 5435817984 |
| FP8_E4M3 | 8192 | 4 | 8 | 32991.309 | 26286.627 | False | 2491416576 / 4982833152 |
| FP8_E5M2 | 2048 | 16 | 32 | 14835.125 | 9506.923 | False | 2717908992 / 5435817984 |
| FP8_E5M2 | 8192 | 4 | 8 | 33207.507 | 26286.627 | False | 2491416576 / 4982833152 |

## Non-dominated compressed points

| ID | Quality pass | All gates | P95 TTFT ms | P95 TPOT ms | SLO-goodput req/s | Cache bytes |
|---|---|---|---:|---:|---:|---:|
| FP8_E4M3/L2048/N1 | False | False | 5493.837 | 74.643 | 0.0409 | 169869312 |
| FP8_E4M3/L2048/N4 | False | False | 5934.332 | 76.128 | 0.1590 | 679477248 |
| FP8_E4M3/L2048/N8 | False | False | 7564.683 | 80.422 | 0.2913 | 1358954496 |
| FP8_E4M3/L2048/N16 | False | False | 9759.660 | 83.197 | 0.5181 | 2717908992 |
| FP8_E4M3/L8192/N1 | False | False | 23384.478 | 78.416 | 0.0233 | 622854144 |
| FP8_E4M3/L8192/N4 | False | False | 26995.720 | 78.004 | 0.0856 | 2491416576 |
| FP8_E5M2/L2048/N8 | False | False | 7318.518 | 81.295 | 0.2897 | 1358954496 |
| FP8_E5M2/L2048/N16 | False | False | 9696.873 | 81.806 | 0.5253 | 2717908992 |
| FP8_E5M2/L8192/N1 | False | False | 23223.969 | 77.129 | 0.0236 | 622854144 |
| FP8_E5M2/L8192/N4 | False | False | 27053.563 | 76.756 | 0.0862 | 2491416576 |

## Provenance and limitations

- Weights and compute are BF16; only K/V cache storage changes.
- BF16 and FP8 cache payloads are sent directly to FlashInfer paged prefill/decode kernels. Fake quantization and full-cache BF16 reconstruction are not system points.
- vLLM capability/import errors and the local 4-bit probe are retained in `run_manifest.json`; unsupported 4-bit is not fabricated.
- Quality CIs are paired prompt bootstrap CIs, exploratory, uncorrected for multiplicity. Retrieval EM is a frozen synthetic needle screen and does not replace NLL/KL.
- TPOT/TTFT/goodput and memory are hardware/run-specific. SLO thresholds are provisional and listed in `experiment_config.json`.
- `max_stable_concurrency` means every warmup and measured repeat completed without OOM/error; the separate TTFT/SLO route gate is intentionally stricter.
- Raw request/repeat/timestamp/cache-profile evidence is in `raw_metrics.csv`, `raw_runs.jsonl`, and `raw_cache_profiles.jsonl`.
