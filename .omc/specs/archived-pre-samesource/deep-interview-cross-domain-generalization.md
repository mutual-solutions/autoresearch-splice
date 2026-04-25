# Deep Interview Spec: Cross-Domain Generalization (Tier 4 redesign)

## Metadata
- Rounds: 7
- Final Ambiguity: 20%
- Type: brownfield
- Generated: 2026-04-17
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.85 | 0.35 | 0.298 |
| Constraint Clarity | 0.75 | 0.25 | 0.188 |
| Success Criteria | 0.75 | 0.25 | 0.188 |
| Context Clarity | 0.85 | 0.15 | 0.128 |
| **Total clarity** | | | **0.801** |
| **Ambiguity** | | | **20%** |

## Goal
Redesign the splice detector + classifier pipeline to generalize across singing, Korean speech, and English speech domains. Current state: detector is singing-specific (combined_full=0.769 on singing, 0.163 on Korean speech, unknown on English). Target state: `combined_speech / combined_full >= 0.6` across multiple domains. Accept significant singing regression if it buys cross-domain balance. Expand training data by synthesizing splices from clean public English speech.

## Constraints
- NO neural networks / GPU (existing program.md constraint).
- Full eval time budget stays 120s (existing program.md constraint).
- Singing regression down to combined_full ≈ 0.50 is acceptable if ratio > 0.6 is hit.
- Pause autoresearch loop during this work — it optimizes singing combined and would fight against cross-domain changes.
- No license-constrained data. Use MIT/CC/public-domain speech corpora (LibriSpeech, VCTK, Mozilla Common Voice).
- Existing `.omc/classifier/generate_patches.py` + `generate_splices.py` patterns are the reference.

## Non-Goals
- Real-time / streaming detection.
- Detecting neural-network-generated audio (deepfakes).
- Replacing the classical-DSP constraint with ML-only detection.
- Optimizing for any single domain past the ratio target.

## Acceptance Criteria

### Dataset synthesis (English speech)
- [ ] 100+ clean English utterances downloaded from a public corpus (MIT/CC license).
- [ ] 40+ synthesized splices with controlled parameters (xfade ms ∈ {10, 50, 100, 200}, same/different speakers) mirroring the existing tier1/tier2 structure.
- [ ] Ground truth JSON in the same schema as `data/spliced/ground_truth.json` (fields: path, spliced, tier, crossfade_ms, splice_time_sec).
- [ ] Located at `data/english-speech/` (symlinked to an external drive if size matters).
- [ ] Reproducible: a `synthesize_english_splices.py` script regenerates the dataset from a seed.

### Tier 4 detector redesign
- [ ] Per-sample-rate / per-domain threshold tables replace hardcoded singing thresholds. Constants like phase GPD α, crossfade T² z-floor, pairwise gate, CPE AGC window become keyed on a domain tag (auto-inferred from sample rate + loudness statistics OR passed explicitly).
- [ ] Input-side audio normalization added at evaluate.py load time (peak or RMS normalization; exact choice justified).
- [ ] Silence threshold becomes relative to the post-normalization floor (not fixed -45 dBFS).
- [ ] Detector code remains explainable (no neural nets).

### Classifier retraining
- [ ] Classifier training patches regenerated from all three domains (singing + Korean speech + English speech).
- [ ] Optional: gain-augmented patches (±15 dB random gain per patch) for loudness invariance.
- [ ] OOF GroupKFold purity preserved — eval-file patches still held out.

### Eval infrastructure
- [ ] `evaluate.py` reports three metrics: combined_full (singing), combined_speech_ko (Korean), combined_speech_en (English).
- [ ] A computed ratio metric: `min(combined_speech_*) / combined_full`.
- [ ] Primary success metric: `ratio >= 0.6`.
- [ ] Full eval (3 domains + classifier) completes under 120s.

### Autoresearch compatibility
- [ ] Autoresearch loop prompt updated to optimize `ratio` (or a combined_cross metric), not just combined_full.
- [ ] Existing autoresearch baseline (combined_full=0.769) documented as pre-redesign reference.
- [ ] Loop can be paused and restarted without losing the Tier 4 changes (committed base).

### Regression guard
- [ ] Existing verified keeps (detector-v7 through v15) remain tagged in git history.
- [ ] A rollback procedure documented: if the redesign fails, revert to v15 tag.

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|---|---|---|
| Normalization alone will fix generalization | User accepted that volume norm alone is insufficient; multi-factor redesign needed | Tier 4 covers normalization + thresholds + classifier retrain |
| Singing combined must not regress | Contrarian mode surfaced that cross-domain balance is the real goal | Large singing regression (down to ~0.5) accepted |
| "New dataset" = one thing | User clarified both Korean speech AND ~/Downloads amateur real-world were meant | Plus English synthesized from public sources |
| Scope is minimal viable | Simplifier mode asked for smallest slice | User chose full Tier 4 — accepted higher complexity |
| Existing eval metric suffices | Ratio metric introduces a new cross-domain measure | Add `ratio = min(combined_speech_*) / combined_full`, target ≥ 0.6 |

## Technical Context (brownfield findings)
- `detector.py` has 4 detection modes: phase (hard cuts), crossfade T² (CQT PSD change-point), pairwise (block structure), CPE (complex prediction error). All thresholds singing-tuned.
- `ml_eval.py` trains classifier per-eval with OOF GroupKFold; uses pre-generated patches + eval-time patches.
- `evaluate.py` has a secondary DSP-only Korean speech eval (commit 52f7054). This gives us the ratio metric plumbing already.
- Pairwise gate recently made sample-rate-adaptive (commit fb02ee4) — raised combined_speech from 0.000 to 0.163.
- Singing eval: 90 files (50 clean + 20 tier1 + 20 tier2), sr=44.1kHz.
- Korean speech eval: 60 files (20 clean + 40 spliced), sr=16kHz, xfade_ms ∈ {0, 10, 50, 100, 200}.
- Autoresearch loop is actively running; would fight against singing regressions.

## Ontology
| Entity | Type | Fields | Relationships |
|---|---|---|---|
| Domain | core | name, sample_rate, corpus_source | has many Files |
| File | core | path, duration, sample_rate, spliced, gt_times, domain | belongs to Domain |
| Detector | core | method (phase/crossfade/pairwise/cpe), thresholds, sample_rate_adaptive | emits Detections |
| Classifier | core | training_patches, OOF_threshold, hyperparams | filters Detections |
| Threshold | core | name, value, domain_key | configures Detector |
| Metric | core | name (combined_full/combined_speech_*/ratio), value | measures pipeline |

## Phasing (implementation order)
1. **Phase A — English dataset synthesis** (foundation). Build `data/english-speech/` + synthesis script. Add `combined_speech_en` to evaluate.py. Publish baseline numbers.
2. **Phase B — Input normalization + adaptive silence**. Add peak-norm. Recalibrate silence threshold. Measure ratio change on all three domains.
3. **Phase C — Per-domain thresholds**. Introduce a domain key; move hardcoded singing constants into a lookup table keyed on sr or loudness class.
4. **Phase D — Classifier retrain on 3 domains**. Add English patches to training pool. Optional gain augmentation. Re-measure.
5. **Phase E — Autoresearch integration**. Update loop prompt to optimize `ratio`. Document baselines. Re-enable autoresearch.

## Exit criteria (must all pass)
- `ratio = min(combined_speech_ko, combined_speech_en) / combined_full >= 0.6` on fresh eval.
- Combined_full not below 0.50 on singing.
- Full eval (3 domains + classifier) ≤ 120s.
- No neural-network imports in detector.py.
- Commits land on a new branch (e.g., `cross-domain/apr17`) and are not disturbed by autoresearch resets.
