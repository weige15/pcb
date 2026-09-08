## Review

### Correct
- **Scoring:** `scripts/projection_quantization_sensitivity.py:196–234` correctly aligns hidden position *t* with label *t+1*, computes token-mean NLL, and evaluates **KL(BF16 || intervention)** using FP32 log-softmax/reductions. Tiny negative KL roundoff remains unmodified.
- **Pairing:** `paired_bootstrap` at `:322–329` resamples within-block differences. The analyzer uses identical draws across conditions, projections, and layers; condition validation enforces matching ordered blocks (`scripts/layerwise_projection_sensitivity.py:100–115`).
- **Concentration:** `scripts/analyze_layerwise_projection_sensitivity.py:47–112` implements the protocol’s positive-part scores, cumulative shares, effective-layer count, and shared-block bootstrap **with reranking**. ALL sums independently clipped projection scores, not measured joint damage. Zero-score denominators are undefined.
- **Interpretation/plots:** Layer and contrast ordering match their plotted values (`:145–185`). Negative ΔNLL remains visible. Report/protocol distinguish descriptive rankings, unadjusted exploratory intervals, non-equivalence of crossing zero, and limited generalization.

### Findings — severity ranked

1. **P1 — Failure/resume can finish measurement but permanently fail the analysis gate.**  
   **Location:** `scripts/analyze_layerwise_projection_sensitivity.py:36–41`.  
   The gate requires every producing attempt to finish successfully and contain `final_state_matches_checkpoint` and `newly_completed`. However, the runner commits individually validated conditions before finishing an attempt (`scripts/layerwise_projection_sensitivity.py:333–347`), records a later failure as `blocked` (`:403–409`), and skips those committed conditions on resume (`:384`). Attempt-wide completion fields are only populated on success (`:352`, `:398–399`).

   **Concrete recovery sequence:** attempt A commits BF16 and an intervention, then fails on a subsequent intervention; attempt B resumes and completes all missing conditions. The manifest becomes measured, but committed conditions still reference blocked attempt A, so both analyzer and verifier reject them. This contradicts the preserve-completed-work resume contract in `results/layerwise_projection_sensitivity/protocol.md:21–23`.

   **Smallest fix:** support explicit recovery validation of individually committed, fully controlled conditions from interrupted attempts; preserve original provenance and require their recorded per-condition restoration controls rather than treating later attempt failure as invalidating earlier measurements. Add a commit → later failure → resume → analysis-gate regression test. Do not merely relabel failed attempts successful.

2. **P2 — Exact ties are reported as strict projection rankings.**  
   **Location:** `scripts/analyze_layerwise_projection_sensitivity.py:224–225`.  
   The report joins a stable sort with `" > "`. If all three means are equal—for example, zero ΔNLL—it prints `WQ > WK > WV`, solely from dictionary order. The “descriptive” disclaimer does not make that strict inequality true.

   **Smallest fix:** represent exact ties with `=` or a separate tied category; retain near-ties as numerical orderings without introducing an arbitrary tolerance. Test all-equal and two-way-tied means.

3. **P2 — Analysis-runtime provenance is incomplete.**  
   **Location:** `scripts/analyze_layerwise_projection_sensitivity.py:265–284`.  
   Analysis saves source/artifact hashes and deterministic draws, but not its actual Python, NumPy, or Matplotlib versions. Checking saved inference-attempt versions (`:42–43`) does not establish the environment executing analysis. Consequently, a figure-generation environment cannot be reconstructed from the recorded analysis manifest.

   **Smallest fix:** record actual analysis-runtime versions, command, and plotting backend in `analysis_manifest.json`; distinguish numerical reproducibility from byte-identical figure reproduction.

### Unverified limits
- Read-only source review only: no commands, synthetic checks, tests, inference, or artifact writes were performed.
- Existing tests cover scorer direction, pairing, and runner controls, but the inspected test files do not exercise the new analyzer/concentration/report or interrupted-producing-attempt recovery.
- Verifier statistics are independently recalculated, but runtime controls largely remain recorded attestations. Figure checks verify hashes/file structure and dimensions, not rendered labels or plotted data (`scripts/verify_layerwise_projection_sensitivity.py:187–199`); manual visual review remains necessary.
- Final measurements, numerical audit results, figures, and report are not yet available for validation. Their absence during the running experiment is **not a finding**.
- Git diff/staging state was not inspected.

**Merge verdict: OK with notes.** Fix the P1 recovery defect before release/completion acceptance.
