# Qwen3-4B hardware-real static KV-cache precision Pareto screen protocol

**Scope frozen before any model forward.** This is a new, independent result scope. It does not modify, merge, or reinterpret `main@1eef360`, `results/phase1a*`, or `results/layerwise_projection_sensitivity/`. The only model/data inputs are the pinned local Qwen3-4B checkpoint and the pinned WikiText test parquet used by the historical experiment.

## Question and provisional gates

With model weights and compute fixed at BF16 and homogeneous continuous batches, does a compressed KV-cache format that is read directly by the GPU attention kernel produce a non-dominated point satisfying quality limits while improving P95 TPOT, SLO-goodput, GPU memory, or maximum stable concurrency?

No deployment SLO was supplied. The following are **provisional, screen-only gates**, frozen here and recorded as such:

- quality: paired bootstrap 95% CI upper bound for mean ΔNLL (compressed − BF16) `<= 0.005` nats/token;
- quality: paired bootstrap 95% CI upper bound for mean `KL(BF16 || compressed) <= 0.002` nats/token;
- quality: long-context retrieval EM decline `<= 1.0` percentage point from BF16;
- performance: at least one of P95 TPOT improvement `>= 10%`, SLO-goodput improvement `>= 10%`, or maximum stable concurrency improvement `>= 20%`;
- if claiming the concurrency route, compressed P95 TTFT at its maximum stable concurrency must not be more than `5%` worse than BF16 at BF16's maximum stable concurrency. Comparisons are reported both at matched concurrency and at each mode's maximum stable concurrency.

For SLO-goodput only, because no deployment SLO exists, the frozen provisional operational SLO is P95 TTFT `<= 10,000 ms` at prompt length 2048 and `<= 30,000 ms` at prompt length 8192, and P95 TPOT `<= 250 ms`. SLO-goodput is the number of requests whose individual TTFT and TPOT satisfy both limits divided by wall-clock batch duration. These thresholds are not production claims.

A quality or performance point is eligible for the system Pareto table only when its cache payload is compressed/packed, its configured payload byte count is measured, and the attention kernel directly consumes that representation. Any mode that first reconstructs the entire cache to BF16, or any fake quantization path, is diagnostic-only and is excluded from system results.

## Fixed model, data, and numerical conditions

- Model: `Qwen/Qwen3-4B`, local snapshot revision `1cfa9a7208912126459214e8b04321603b3df60c`; tokenizer same revision; all model weights and compute BF16; no weight quantization, tensor parallelism, or model replacement.
- Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`, `test`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`; official cached parquet only; rows joined by two newlines in original order; tokenizer called with `add_special_tokens=False`, no truncation or padding.
- Quality inputs are 8 fixed prompt IDs per length. Each is a non-overlapping contiguous token span of prompt length 2048 or 8192 followed by 256 fixed teacher-forced target tokens. The 32 performance prompt IDs per length are separate non-overlapping spans and include the first 8 quality IDs. `prompts.npz` and `prompt_manifest.json` are immutable input artifacts and contain every token ID and source/hash.
- Context lengths: exactly 2048 and 8192 prompt tokens. Generated/teacher-forced output length: exactly 256 tokens. No chat template, system prompt, or hidden thinking tokens.
- Quality: BF16 and every supported compressed mode use identical prompt and target IDs. For each target token, compare BF16 logits and compressed-cache logits in FP32 log-softmax/reduction; report per-prompt NLL, ΔNLL, KL direction `P_BF16 || P_mode`, and token counts. Quality runs use deterministic greedy output only for the separate retrieval EM check.
- Retrieval: 8 fixed 8192-token prompts made from the pinned token stream plus a frozen needle/answer template, tokenized and saved before model forward. Each has a unique answer string and exactly 256 generated tokens. EM is normalized exact match of the first generated line against the frozen answer; all raw decoded text and token IDs are retained. Retrieval is a quality diagnostic, not a replacement for NLL/KL.
- Seeds: input selection seed `424242`; generation seed `424242`; torch and NumPy seed `424242`; deterministic algorithms on where supported; TF32 disabled; 4 CPU threads. Sampling is greedy (`temperature=0`, `top_p=1`, `top_k=0`), EOS is not used to shorten the fixed 256-token measurement.
- Performance: homogeneous continuous batches with all requests of one prompt length, same 256-token decode horizon, no prefix-cache reuse, no request mixing, no dynamic scheduler policy. Concurrency values are exactly `1, 4, 8, 16, 32`. Warmups are 1 and measured repeats are 2 per mode × length × concurrency. A measured batch is launched together, prefills together, then executes one batched decode step per token; wall-clock, GPU-event timing, TTFT, per-token timestamps, P95 TPOT, and per-request completion are saved. OOM and kernel errors are recorded, never replaced by smaller fixed conditions.
- Static cache: cache storage is allocated to the full `prompt_length + 256` capacity per request before prefill. Pages are fixed at 16 tokens and allocated for every layer/request. The same cache layout, batch, prompt IDs, decode horizon, and attention implementation are used across modes.

## Modes and admissibility

- `BF16`: BF16 K/V payload consumed directly by the FlashInfer paged prefill/decode attention kernels; baseline and system reference.
- `FP8_E4M3`: one-byte `torch.float8_e4m3fn` K/V payload, static per-layer scale `1.0`, consumed directly by FlashInfer paged prefill/decode attention kernels. No whole-cache BF16 materialization is permitted. The exact configured cache tensor bytes and scales are profiled and saved.
- `FP8_E5M2`: one-byte `torch.float8_e5m2` K/V payload, same direct-kernel and byte-accounting rules; included only if the local FlashInfer kernel accepts and executes it.
- 4-bit: probe the local direct-kernel APIs for an actual packed 4-bit KV format. No 4-bit mode is invented if unsupported. A failed capability probe is recorded with the complete exception and is not represented as a speed/memory result.
- A mode is `system_admissible=true` only after a direct-kernel smoke test proves that the cache tensor dtype/layout is passed to and consumed by the attention kernel, a nontrivial cache payload is resident in the configured cache tensors, no full-cache BF16 restoration occurs, and actual cache payload bytes are recorded from tensor allocation/profiling. A mode that fails this test remains `unsupported` or `diagnostic_only`.

## Baseline and smoke ordering

1. Save this protocol, `experiment_config.json`, source/config hashes, GPU inventory, and command before model forward.
2. Prepare and hash all prompt/tokenizer/data artifacts without model forward.
3. Probe local FlashInfer/vLLM capabilities. vLLM is not silently substituted if its installed extension is incompatible; the exact import error is saved. Direct FlashInfer is the intended backend.
4. Load the pinned model with BF16 weights. Run BF16 cached decode first and compare cached logits to a no-cache BF16 reference on fixed prompt/target positions. This alignment gate must pass before any compressed run.
5. Run one BF16 and one supported compressed-mode smoke at prompt length 2048, concurrency 1, output 256. Check finite logits, direct-kernel provenance, exact cache bytes, and restoration/cleanup.
6. Only after those gates pass, run the complete fixed quality and performance matrix. Missing resources, missing pinned assets, OOM, or lack of a direct compressed kernel cause `blocked.md` with exact command/error/completed/missing lists and stop; no reduced experiment or decompression path is accepted.

## Stored evidence and analysis

- `scripts/kv_cache_precision_pareto.py`, `results/kv_cache_precision_pareto/experiment_config.json`, `scripts/run_kv_cache_precision_pareto.sh`, `scripts/prepare_kv_cache_inputs.py`: reproducible runner/config/launcher.
- `run_manifest.json`: revisions, hashes, commands, environments, GPU/driver inventory, backend/kernel and cache provenance, fixed matrix, status, completion, OOM/errors, and artifact hashes.
- `prompts.npz`, `prompt_manifest.json`, `tokenizer/`: exact prompt IDs, targets, retrieval prompts/answers, tokenizer assets, and hashes.
- `raw_metrics.csv`: one row per mode × phase × context × concurrency × repeat × request/token where applicable; no aggregate-only replacement. `raw_cache_profiles.jsonl` records configured/allocated/resident bytes and direct-read provenance per run.
- `quality_summary.json`, `performance_summary.json`, `pareto_report.md`: paired prompt bootstrap CI, retrieval EM, matched performance comparisons, gates, non-dominated points, and a provisional NO-GO if no low-precision point passes all gates.
- `independent_verifier.py`: rechecks protocol/config/source/input/artifact hashes, prompt identities, quality bootstrap/EM calculations, cache-byte accounting, direct-kernel provenance fields, matrix/row completeness, and no-go/pareto consistency without model forward.
- `completion_audit.md`: prompt-to-artifact audit with every explicit requirement and evidence path; it must not claim completion from a green manifest alone.

This protocol does not implement dynamic scheduling, mixed-precision batches, age-based policy, other models, or a layerwise sweep.
