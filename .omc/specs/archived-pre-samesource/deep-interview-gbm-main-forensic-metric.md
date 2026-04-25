# Deep Interview Spec: GBM-main pipeline + 15-cell forensic metric

## Metadata
- Interview ID: gbm-main-forensic-metric
- Rounds: 7
- Final Ambiguity: 30% (early exit — user signaled design understanding complete; remaining gaps are calibration parameters, not design decisions)
- Type: brownfield (autoresearch-splice)
- Generated: 2026-04-17
- Threshold: 0.2 (proceeded below with documented gaps)
- Status: BELOW_THRESHOLD_EARLY_EXIT

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.85 | 0.35 | 0.298 |
| Constraint Clarity | 0.80 | 0.25 | 0.200 |
| Success Criteria | 0.30 | 0.25 | 0.075 |
| Context Clarity | 0.85 | 0.15 | 0.128 |
| **Total Clarity** | | | **0.70** |
| **Ambiguity** | | | **0.30** |

## Goal

Invert the current detection pipeline: replace DSP-first-then-classifier with **multi-class GBM-as-primary, DSP-as-feature-source**. Simultaneously introduce a single balanced metric that evenly optimizes performance across all (dataset × splice-regime × tier) cells while penalizing false positives on clean files via a smooth multiplier.

## Architecture (operational)

```python
# Primary detection loop
for chunk in sliding_windows(audio, W=60, S=50):   # existing sliding (detector.py)
    for t, probs in gbm.score_dense(chunk, stride_s=0.2):
        p_splice = probs['hard_cut'] + probs['crossfade']
        if p_splice < THRESHOLD:
            continue
        label = 'hard_cut' if probs['hard_cut'] > probs['crossfade'] else 'crossfade'
        emit(chunk_offset + t, label=label, probability=p_splice,
             evidence=shap_values(feats))

# File-level merge: dedupe within 1s across overlapping chunks (existing logic)
```

Multi-class softmax — no post-hoc DSP gate. DSP scores feed GBM as features; they surface in SHAP decompositions for forensic reports, not as filters.

## Feature vector (~66 dims per candidate position)

### Local DSP scores at t (4 dims)
- `phase_z_at(t)` — phase discontinuity z-score
- `t2_z_at(t)` — crossfade CQT Hotelling T² z-score
- `cpe_z_at(t)` — complex prediction error z-score
- `pairwise_proximity(t)` — distance to nearest pairwise block-structure split

### Pre/post statistical comparison (~25 dims)
Pre window = [t − 2s, t], post window = [t, t + 2s].
- MFCC-13 mean delta vector (13 dims)
- Spectral summary deltas: centroid, rolloff, flux, contrast, flatness, bandwidth (6 dims)
- Noise-floor color features (bottom-10%-energy frames pre vs post): KL divergence, centroid delta, rolloff delta, flatness delta, pre/post bottom-10% energy (6 dims)

### Pitch / voicing continuity (8 dims)
- F0 pre/post mean, delta (3 dims)
- F0 jitter delta (1 dim)
- Voicing probability pre, post, delta (3 dims)
- F0 continuity score across t (1 dim)

### Energy envelope / ZCR (6 dims)
- RMS dB pre, post, delta
- ZCR pre, post, delta

### Boundary region features (3 dims)
Around the ±200ms zone of t itself:
- Spectral flux peak
- Energy ratio pre-vs-post inside boundary
- Phase coherence

### Mel-patch PCA tail (20 dims)
PCA(20) of the 2s mel patch centered at t. Safety net for patterns the named features miss. Lower dim than current 50 to reduce dilution.

**Total: ~66 dims, tabular, GBM-native.**

## Metric: `combined`

**Formula:**
```
combined = GM(15 cells, floored at 0.01) × (1 − total_clean_fp / max_allowed)
```

Where the 15 cells are:

```
# 12 spliced-F1 cells  (dataset × regime × tier)
singing_t1_random,    singing_t1_quiet_matched,
singing_t2_random,    singing_t2_quiet_matched,
korean_t1_random,     korean_t1_quiet_matched,
korean_t2_random,     korean_t2_quiet_matched,
english_t1_random,    english_t1_quiet_matched,
english_t2_random,    english_t2_quiet_matched,

# 3 clean-score cells  (one per dataset)
singing_clean_score,
korean_clean_score,
english_clean_score,
```

- **Geometric mean** with per-cell floor of 0.01 (avoids zero-collapse when a cell has no signal yet).
- **Multiplicative FP penalty** — soft, not a hard gate. `max_allowed` is a tunable denominator.
- Weighted-by-cells implicitly: every cell contributes 1/15 to log-mean.

## Constraints

- **ML is allowed.** K-DFC guidance: explainability trumps classical-only. Feature importance + SHAP decompositions must be exportable.
- **Sliding-window orchestrator stays.** W=60s, S=50s (detector.py). GBM runs inside each chunk.
- **DSP functions must be refactorable into callable-at-time-t form.** Currently phase/T²/CPE compute full-file curves; we need `phase_z_at(t)`, etc. that consume the already-computed chunk curve and sample at t.
- **Computational budget: 240s per full eval.**
- **FP-free guarantee is highest-priority outcome.** The smooth FP multiplier is the optimization signal; K-DFC forensic acceptance requires the actual reported clean_fp to be 0 in practice.
- **Singing generator (`corpora/scripts/...`) needs regeneration to include boundary_energy split** — currently only speech datasets have the regime field. Without singing regen, `singing_t*_quiet_matched` cells don't exist.

## Non-goals

- Neural networks (still excluded; GBM remains the model class).
- Real-time inference (batch eval is fine).
- Multi-splice-per-file detection (single splice per file still, per current ground_truth format — stays a future extension).
- Automatic hyperparameter tuning — THRESHOLD / stride / max_allowed are set explicitly, not grid-searched inside the eval loop.

## Acceptance Criteria

- [ ] `dataset_registry.py` extends `aggregate_combined` (or a new `forensic_combined`) with the 15-cell × FP-multiplier formula and a unit test covering the shape.
- [ ] `corpora/data/` includes a regenerated singing dataset with `boundary_energy` populated (random + quiet_matched per tier).
- [ ] `features.py` (new) computes the ~66-dim feature vector given `(audio, sr, t, chunk_precomputed_context)`. Exposes per-feature names for SHAP labeling. Feature extraction is deterministic (seed-pinned).
- [ ] `train_classifier.py` is updated (or replaced) to train a 3-class GBM (`not_splice` / `hard_cut` / `crossfade`) on the new feature vector. Multi-class F1 CV ≥ the current binary F1 0.85 on the training distribution.
- [ ] `detector.py`'s `detect_splices` is refactored to call `gbm.score_dense(chunk, stride)` instead of running DSP-first; dense stride defaults to 200ms, configurable via `ANALYSIS_STRIDE_S`.
- [ ] DSP functions expose `phase_z_at(chunk, t)`, `t2_z_at(chunk, t)`, `cpe_z_at(chunk, t)` as single-position accessors that reuse per-chunk precomputed arrays (no full-detector rerun per candidate).
- [ ] `evaluate.py` integration patch in `.omc/evaluate_integration.md` is updated to reflect: per-dataset + per-regime + per-tier breakdown; new 15-cell aggregator; FP multiplier semantics.
- [ ] End-to-end eval completes in < 240 s (per program.md). `verify_agent.py` passes HIGH on the new `combined`.
- [ ] Forensic report sidecar (JSON) exports per-detection SHAP values + top-3 contributing features, so "why was this flagged as hard-cut" is answerable.

## Success gating (two-phase)

**Phase 1 — Implementation-done (this spec's scope):** all acceptance criteria above pass. Architecture wired, pipeline runs end-to-end < 240 s, verify HIGH. Whatever value `combined` has at that point is the new baseline. **Ralph/autopilot finishes here.**

**Phase 2 — Research target (separate effort):** autoresearch loop iterates hyperparameters, feature choices, training data curation to push `combined` from Phase-1 baseline toward **≥ 0.65**. This is a multi-iteration loop with its own verify_agent guardrails. Not in scope for this spec's implementation.

## Parameters to calibrate (not design decisions)

These four values remain numeric choices to set during execution — decide empirically once the pipeline runs.

| Parameter | Role | Initial proposal | Calibration path |
|-----------|------|------------------|------------------|
| `ANALYSIS_STRIDE_S` | GBM dense-scan step | 0.2 s | Sweep {0.1, 0.2, 0.5}; smaller = higher recall, more compute |
| `THRESHOLD` | P(any_splice) above which we emit | 0.5 | PR curve on a held-out split; pick point where precision ≥ 0.90 |
| `max_allowed` (FP multiplier denominator) | Scaling factor for clean-file FP penalty | 15 (matches current DSP bound) | Sensitivity analysis: rerun with {5, 15, 30}; pick value that makes the penalty meaningful without zeroing combined on edge cases |
| `target_combined` | **Research target** (not implementation gate) | **0.65** | Autoresearch loop chases this AFTER implementation completes. Implementation-done = spec acceptance criteria pass, regardless of current combined value. |

## Assumptions Exposed & Resolved

| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| DSP must be the primary detector (classical-only) | K-DFC interview: "ML is OK if explainable" | ML elevated to primary; DSP demoted to feature source |
| Phase-discontinuity gate improves precision | Spikes exist but don't pass single-feature significance; GBM can combine sub-threshold evidence | Gate dropped — `phase_z` becomes GBM feature, not filter |
| One F1 per dataset is enough | Random vs quiet_matched are structurally different attacks with different detector responses | 15-cell metric: per-dataset × regime × tier |
| FP hard-gate at any count > 0 | Some FPs may be acceptable residual noise on long files | Smooth multiplier `(1 - fp/max_allowed)`; FP-free remains the EMPIRICAL target |
| Classifier filter on DSP candidates | Quiet_matched candidates never propose — filter is too late | GBM scans densely (every 200 ms), not just at DSP candidates |
| Feature set = 25,600 mel values → PCA(50) | Feature importance is diluted (top-10 = 29.6%); explainability suffers | ~66 named hand-crafted features + 20-dim PCA tail for residuals |
| Binary splice / no-splice | Need hard-cut vs crossfade labeling anyway; external DSP rule is fragile | Multi-class GBM (not_splice / hard_cut / crossfade); label falls out of softmax |
| Geometric mean is enough | Harmonic was first choice but collapses when speech_quiet F1 ≈ 0 | GM with 0.01 floor preserves gradient while keeping inverse-proportional pressure |

## Technical Context (brownfield)

- **Current classifier:** `.omc/classifier/fp_classifier.joblib`. Pipeline: `StandardScaler → PCA(50) → GradientBoostingClassifier(300 trees, depth 3)`. CV F1 ~0.85 binary. Feature importance spread across 50 PCs — top-1 = 5.6%.
- **Current sliding detector:** `detector.py` `detect_splices()` already does 60s/50s sliding. DSP runs first via `_detect_in_chunk()` (phase + crossfade + CPE + pairwise), emits candidates, then `ml_eval.py` filters via OOF GBM. New pipeline reverses this.
- **Existing feature extraction:** `generate_patches.py::extract_mel_patch(audio, sr, t)` — produces 128×200 mel patch centered at t. Reusable for the PCA tail.
- **Metric aggregator scaffolding:** `dataset_registry.py::aggregate_combined()` already supports geometric mean + 0.01 floor. Need extension for the 15-cell breakdown + FP multiplier.
- **Datasets & regimes:** Korean + English already have `boundary_energy` field (random / quiet_matched) per `ground_truth.json`. Singing (`corpora/data/spliced/ground_truth.json`) does NOT — needs regeneration.
- **Sliding-window cache:** `detect_splices` memoizes by audio fingerprint. Must preserve the cache when rewriting to GBM-first to keep budget under 240s.

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| Chunk | supporting | audio, sr, offset_s, W=60, S=50 | Sliding over file via `detect_splices` |
| CandidatePosition | supporting | t (seconds within chunk), chunk_offset_s | Scanned every `ANALYSIS_STRIDE_S` |
| FeatureVector | core | ~66 named dims (DSP scores + pre/post stats + pitch + energy + mel-PCA tail) | Extracted at each CandidatePosition |
| GBMClassifier | core | multi-class {not_splice, hard_cut, crossfade}, softmax output | Trained on FeatureVector |
| Detection | core | absolute_t, label, probability, shap_values | Emitted when `p_splice > THRESHOLD` |
| DSPScoreAtT | supporting | phase_z, t2_z, cpe_z, pairwise_proximity | Feeds FeatureVector, never gates |
| Dataset | supporting | id, path, eval_weight, train_weight | dataset_registry.py |
| BoundaryEnergy | supporting | regime ∈ {random, quiet_matched}, pre_rms_db, post_rms_db | Field in ground_truth.json |
| MetricCell | core | dataset × regime × tier → F1, or dataset → clean_score | 15 cells total |
| FPMultiplier | supporting | 1 - total_clean_fp / max_allowed | Scales combined |
| CombinedMetric | core | GM(15 cells, floor=0.01) × FPMultiplier | Primary optimization target |
| ForensicReport | supporting | per-detection SHAP + top-contrib features + label rationale | Exported sidecar JSON |

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Stability |
|-------|-------------|-----|---------|--------|-----------|
| 1 | 7 | 7 | 0 | 0 | N/A |
| 2 | 8 | 1 | 0 | 7 | 87% |
| 3 | 9 | 1 | 0 | 8 | 89% |
| 4 | 10 | 1 | 0 | 9 | 90% |
| 5 | 11 | 1 | 0 | 10 | 91% |
| 6 | 11 | 0 | 1 | 10 | 100% |
| 7 | 12 | 1 | 0 | 11 | 92% |

Ontology stabilized early — every later round just added one refinement (BoundaryEnergy regime split → 15-cell metric → FP multiplier → multi-class GBM → ForensicReport). No conceptual reframes.

## Interview Transcript

<details>
<summary>Full Q&A (7 rounds)</summary>

### Round 1 — Goal
**Q:** What is the "newly stated goal" the single metric must measure?
**A:** Weighted cross-domain harmonic / geometric mean across (dataset × regime), FP gate as hard constraint.
**Ambiguity:** 64% (goal clarity jumped from 0 to 0.4 — metric shape known, aggregation cells + FP gate still open)

### Round 2 — Goal (refinement)
**Q:** Over how many cells does the harmonic mean aggregate? (3 / 5 / 9 / flexible)
**A:** 3-cell with singing regime split — effectively 6 cells.
**Ambiguity:** 59%

### Round 3 — Goal (FP gate)
**Q:** When clean FPs appear, what happens to the metric?
**A:** Smooth multiplicative penalty `(1 - clean_fp / max_allowed)`, AND cells jump to 18 (dataset × regime × tier labeling).
**Ambiguity:** 58%

### Round 4 — Goal (cell structure)
**Q:** Which 18? (clean doesn't have regime/tier — Cartesian breaks)
**A:** 15 real (12 spliced F1 + 3 clean_score), FP multiplier is the 18th "dim."
**Ambiguity:** 48%

### Round 5 — Goal (HM vs GM, floor)
**Q:** Harmonic collapses to 0 at current speech_quiet F1 ≈ 0. HM, GM, or weighted?
**A:** Geometric mean + floor at 0.01. Matches `aggregate_combined()` already in dataset_registry.
**Ambiguity:** 45%

### Round 6 — Constraints (main detector role)
**Q:** What does "GBM as main" mean operationally?
**A:** GBM scans every 200 ms, DSP gates each hit — selected initially; revised in round 7.
**Ambiguity:** 35%

### Round 7 — Constraints (feature set + gate removal)
**Q:** What features does GBM see? (mel only / mel + DSP / named only / everything)
**Meta:** User asked for SOTA research instead of selecting. Recommendation: ~66-dim hybrid (DSP scores + pre/post stats + pitch/voicing + mel-PCA tail).
**Follow-up:** "If phase_z is already a feature, does the post-GBM gate add anything?"
**Resolution:** Drop the gate (double-counting). Replace with multi-class GBM (softmax gives label directly). DSP scores remain features + surface in SHAP for forensic reports.
**Ambiguity:** 30%

### Exit
User: "Now I think I understand how it works. Good." → crystallizing with remaining gaps as calibration parameters.

</details>

## Changelog

- 2026-04-17: Initial crystallization after 7-round interview. User signaled understanding complete; 4 parameter values (`THRESHOLD`, `stride`, `max_allowed`, `target_combined`) explicitly deferred to calibration phase.
