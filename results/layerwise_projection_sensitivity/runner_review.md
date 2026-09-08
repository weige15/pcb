## Review

### Finding — P1: completed-run resume bypasses BF16 cache validation

**Location:** `scripts/layerwise_projection_sensitivity.py:145–163,291–297,388–398`

The cache’s existence/hash is checked only inside `run_missing()`. When all 109 condition JSONs exist, `main()` skips that function and reports `measured`/`complete` without checking the cache.

**Source-proven failure case:** a completed bundle with a deleted or corrupted `baseline_hidden.pt` still passes the no-forward resume path. Neither `validate_condition()` nor `completed_results()` even requires BF16 cache metadata. This contradicts `results/layerwise_projection_sensitivity/protocol.md:22`, which requires cache hashes to be verified on resume.

The existing synthetic BF16 fixture omits `hidden_cache` and is deliberately accepted (`tests/test_layerwise_projection_sensitivity.py:71–72,90,122`), so current tests miss this gap.

**Smallest fix:** validate BF16 cache metadata, existence and SHA256 in the common CPU-only artifact-validation path, including no-op resume and final completion. Add temporary-fixture tests for missing/corrupt cache and missing metadata; assert no inference occurs.

### Correct

- **Intervention and restoration:** full parameter/buffer bitwise comparisons surround exactly one group128 RTN4 weight replacement; `finally` restores the target, and any remaining state mutation aborts rather than commits (`scripts/layerwise_projection_sensitivity.py:53–97,321–340`).
- **Historical identity and scoring:** checkpoint/config hashes and loaded model-state hashes are checked; saved 64×2049 blocks produce 2048 shifted predictions each (`scripts/layerwise_projection_sensitivity.py:45–50,165–178,232–245,268–286`). The reused quantizer and scorer match the specified RTN and KL direction.
- **Missing-only execution:** committed files are validated, pending conditions exclude existing results, and destination overwrite is refused (`scripts/layerwise_projection_sensitivity.py:145–162,299–304,339–340`). The launcher uses one explicit GPU and no scheduler.
- **Real evidence:** the first attempt records `bounded_partial`, BF16 plus L00_WQ4, and final restoration. The resumed attempt explicitly skips those two conditions. The inspected log reached **18/109**, not completion. Available intervention control records report 400 checked tensors and successful restoration; sampled L00_WQ4/L04_WV4 records include exact single-weight differences and restored endpoint outputs (`results/layerwise_projection_sensitivity/conditions/L00_WQ4.json:966–1004`; corresponding L04_WV4 lines).
- **No observed bogus completion of the active run:** manifest/attempt status remains `running`. The finding concerns the future completion-validation path, not evidence that current measurements are wrong.

**Merge verdict: OK with notes — fix the P1 validation gap before release/final acceptance.**

**Validation limits:** read-only static/artifact inspection only. No commands, inference, file changes or job operations were performed. Tests and independent binary/hash verification remain unperformed; the live inventory changed during inspection.
