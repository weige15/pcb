## Review

### Correct
- **Grouped RTN matches the specification:** row-local groups of 128, FP32 max-absolute scales, symmetric bounds, nearest-even rounding, zero-group handling, and BF16 dequantization (`scripts/phase1a.py:134–144`).
- **Interventions are isolated and reversible:** exact 36-module target sets and shapes, checkpoint-state checks before intervention, full parameter/buffer hashes for non-target controls, and restoration in `finally` (`scripts/phase1a.py:147–190`). Shapes and tied-head assumptions match the locally cached pinned model configuration.
- **Metrics are scientifically aligned:** BF16 reference hidden states, unchanged chunked `lm_head`, FP32 log-softmax/reductions, correctly directed `KL(P_BF16 || P_test)`, and shifted next-token labels (`scripts/phase1a.py:193–231,278–303`). Smoke controls compare chunked logits with the installed CausalLM implementation and verify restoration (`234–275`).
- **Sampling and statistics match:** fixed non-overlapping blocks, seed-42 sampling without replacement, shared samples across conditions, paired bootstrap draws, and PPL from mean NLL rather than mean block PPL (`scripts/phase1a.py:86–130,319–386`).
- Model/tokenizer/data commit pins, source/checkpoint hashes, saved token IDs, tokenizer settings, runtime versions, and hardware metadata provide substantial provenance coverage (`scripts/phase1a_config.json:2–24`; `scripts/phase1a.py:115–130,451–476`).

### Findings

1. **P1 — Analysis-only can falsely attest that failed controls passed.**  
   **Location:** `scripts/phase1a.py:328–362,388–419,440–444`.  
   `--analyze-only` validates the CSV and sample identities but never verifies the manifest’s smoke results, intervention restoration/evaluation checks, completed-condition inventory, or final checkpoint-state check. The report nevertheless unconditionally claims these controls passed.

   **Reachable failure:** the final CSV row is flushed before the final intervention’s post-evaluation/restoration checks (`300–314`, `182–190`). A failure there leaves all 896 rows and a blocked manifest. Analysis-only then accepts those rows and emits a success-style report, contradicting the specification’s requirement that completion includes necessary controls.

   **Smallest fix:** gate report generation on validated smoke/control evidence, all 14 condition records, all six formal interventions’ success/restoration flags, and `final_state_matches_checkpoint`. Reject blocked/incomplete manifests in analysis-only; do not require `status == "measured"` inside the current in-process analysis call unless call ordering is adjusted.

2. **P1 — Missing-GPU launcher failures bypass the required blocked artifact.**  
   **Location:** `scripts/run_phase1a.sh:3–6,12–16`.  
   With `set -e`, a missing or failing `nvidia-smi` terminates the launcher before output/log initialization and before Python’s exception handler. Thus the documented launch path does not save the command, error, completion count, or missing work to `phase1a_blocked.md` for this explicitly specified resource failure.

   **Smallest fix:** initialize output/logging before the GPU inventory and explicitly handle inventory failure by writing the blocked artifact and exiting nonzero. Keep GPU inventory before any Python execution.

3. **P2 — The advertised rerun command drops the actual configuration and mode.**  
   **Location:** `scripts/phase1a.py:433–436,450–455`; `scripts/run_phase1a.sh:16`.  
   The CLI supports `--config` and `--smoke-only`, but `rerun_command` always invokes the default configuration and a formal run. An accepted alternate pinned configuration is therefore not reproduced; a smoke-only run’s advertised rerun instead starts formal measurements. The separately recorded actual command retains arguments but points to the already-used output, which the overwrite guard rejects.

   **Smallest fix:** construct the fresh-output rerun command from the actual invocation, preserving configuration and mode. Prefer a saved configuration snapshot so subsequent edits to the original config cannot change the rerun.

### Validation and residual risks
- Read-only source inspection; no files changed, shell commands executed, or GPU/model experiments run.
- No Phase 1a tests were present in the inspected paths at review time.
- Runtime GPU/backend compatibility, smoke controls, memory limits, and the 896 real measurements remain unverified by this review.
- Parent should add CPU regression tests for rejecting analysis after failed final controls, launcher inventory failure recording, and rerun argument preservation.

**Merge verdict: BLOCK** pending the two P1 acceptance/error-path fixes. The inspected quantization and metric mathematics otherwise match the specification.

---

## Parent resolution record — 2026-09-08 (not a new reviewer verdict)

All three findings were fixed before the formal run:

- P1 report gate: `validate_controls()` now checks all14 cells, smoke metrics/alignment, seven intervention records, original/quantized module evidence, restoration and final checkpoint state. On the actual completed manifest, ten deliberate failure variants (including failed final restoration despite896 CSV rows) were rejected without changing real outputs. Evidence: `phase1a_final_checks_20260908T103535Z.log`.
- P1 early launcher failure: logging/ERR handling now starts before venv/GPU inventory; synthetic inventory failure returns exit3, saves blocked/log output and does not start Python. The unit test passed.
- P2 rerun preservation: a config snapshot is saved and the fresh-output command uses it, preserving GPU selection and `--smoke-only` when applicable. Both actual smoke/formal manifest commands were inspected; source/config hashes match between runs.

Subsequent real validation: 10 unit tests PASS,14 conditions/896 rows complete, standalone smoke exactly reproduced,398 original checkpoint parameter hashes match,216 independent NumPy RTN intervention hashes match,24 paired CI calculations match. See `phase1a_audit.md` for the full requirement-to-evidence audit. The original read-only review above is preserved; its BLOCK verdict describes the pre-fix source, not a claimed post-fix reviewer approval.