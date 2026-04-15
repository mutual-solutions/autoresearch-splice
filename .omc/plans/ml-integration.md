# Explainable ML Integration — Audio Splice Detection

**Created:** 2026-04-15
**Complexity:** MEDIUM-HIGH
**Phases:** 3 (immediate / next / later)

---

## RALPLAN-DR

### Principles

1. **Explainability First** — Every detection must carry a human-readable attribution (Korean AI Basic Act compliance). No black-box predictions.
2. **No Regression on DSP** — The ML classifier is a post-filter, not a replacement. DSP detectors remain the source of candidate splice points.
3. **Evaluation Integrity** — File-level cross-validation prevents data leakage. No patches from the same audio file appear in both train and test folds.
4. **Minimal Dependency Surface** — Tree-based models only (scikit-learn, SHAP). No torch, no GPU, no heavy runtimes.
5. **Two-Track Independence** — Autoresearch modifies `detector.py` on its own loop. The classifier trains separately on patches from a frozen detector snapshot.

### Decision Drivers

1. **Korean AI Basic Act (AI 기본법) compliance** — SHAP attribution per detection is non-negotiable for legal admissibility of forensic output.
2. **False positive rate on clean files** — The classifier's primary value is filtering FPs without dropping TPs; clean_score is the bottleneck metric.
3. **Reproducibility under autoresearch churn** — The detector changes frequently; the classifier pipeline must be rebuildable from any detector snapshot.

### Viable Options

#### Option A: SHAP-on-GradientBoosting (Recommended)

Replace the current GradientBoosting+MLP ensemble with a single GradientBoosting pipeline. Use `shap.TreeExplainer` for per-detection attribution.

- **Pros:** Native TreeExplainer is exact and fast (no sampling). Single model simplifies serialization. GradientBoosting alone already competitive (train_classifier.py line 138 already checks if GB-only beats ensemble). SHAP waterfall/force plots directly usable in reports.
- **Cons:** Loses MLP ensemble diversity. Slightly lower ceiling on complex patch patterns.
- **Risk:** Low. GB is already the dominant model in the existing ensemble.

#### Option B: XGBoost with SHAP

Switch to XGBoost for potentially better performance and tighter SHAP integration (`xgboost.XGBClassifier` is a first-class SHAP citizen).

- **Pros:** Faster training. Better handling of missing values. XGBoost TreeExplainer is well-tested. Can tune `scale_pos_weight` for class imbalance.
- **Cons:** Adds `xgboost` dependency (not currently in pyproject.toml). Marginal gain over sklearn GB for this data size. Extra dependency to maintain.
- **Risk:** Low-medium. Dependency addition needs justification given the small dataset.

#### Option C (Invalidated): RandomForest + SHAP

RandomForest with TreeExplainer. Invalidated because: RF feature importance is mean-decrease-impurity, which is biased toward high-cardinality features in PCA space. RF also doesn't support `predict_proba` calibration as well as boosted models, which matters for threshold-based FP filtering. The existing train_classifier.py already chose GB over RF, suggesting RF was considered and dropped.

### ADR

- **Decision:** Option A — SHAP-on-GradientBoosting (single GB pipeline, drop MLP, add SHAP).
- **Drivers:** Legal compliance requires exact SHAP values; GB TreeExplainer provides this without approximation. Existing code already favors GB-only when it outperforms the ensemble.
- **Alternatives considered:** XGBoost (viable but adds dependency for marginal gain), RandomForest (invalidated due to PCA bias and calibration weakness).
- **Why chosen:** Zero new dependencies beyond `shap`. Exact TreeExplainer. Already proven in current pipeline. Simplifies model from ensemble to single pipeline.
- **Consequences:** Lose MLP ensemble diversity; if future data shows GB plateau, revisit XGBoost. SHAP adds ~2s per detection to inference (acceptable under relaxed 5min target).
- **Follow-ups:** Monitor clean_score after retrain. If GB-only drops >2% F1 vs ensemble, reconsider XGBoost.

---

## Context

The autoresearch-splice project uses classical DSP detectors to find candidate splice points in audio files. An existing GradientBoosting+MLP ensemble (`fp_classifier.joblib`) filters false positives via `fp_filter.py`. However:

- The current classifier uses a naive 80/20 train/test split with no file-level grouping (patches from the same file can leak across train/test).
- There is no explainability — the model outputs a probability but no attribution.
- The MLP component violates the "NO neural networks" constraint in `program.md`.
- `program.md` enforces a 60s hard eval cap that needs relaxing to accommodate ML inference + SHAP computation.
- Zeroth-Korean speech data (9.6GB FLAC) is planned but not yet extracted at `data/zeroth-korean/`.

### Key Files

| File | Role | Modify? |
|------|------|---------|
| `.omc/classifier/train_classifier.py` | Training script (GB+MLP ensemble) | YES — rewrite |
| `.omc/classifier/generate_patches.py` | Patch extraction from detector output | YES — add file-level group metadata |
| `.omc/coordination/fp_filter.py` | Runtime FP filter in detector pipeline | YES — add SHAP output |
| `program.md` | Autoresearch constraints | YES — relax eval time, allow tree ML |
| `prepare.py` | Evaluation oracle | NO (immutable) |
| `detector.py` | DSP detector | NO (autoresearch-managed) |
| `pyproject.toml` | Dependencies | YES — add `shap` |

---

## Phase 1: Cross-Validation + SHAP + program.md (Immediate)

### Guardrails

**Must Have:**
- 5-fold file-level cross-validation (no same-file patches in train and test)
- SHAP TreeExplainer attribution for every detection
- MLP removed from training pipeline
- program.md updated to allow tree-based ML and relax eval time

**Must NOT Have:**
- Neural networks (MLP, torch, tensorflow)
- GPU dependencies
- Modifications to prepare.py or detector.py
- Breaking changes to fp_filter.py's `filter_detections()` interface

### Task 1.1: Add file-level grouping to patch generation

**File:** `.omc/classifier/generate_patches.py`

**Work:**
- The existing `manifest.json` already records `"file"` per patch. Extend `generate_patches.py` to also save a `file_ids.npy` array parallel to `patches.npy` and `labels.npy`, where each element is an integer file index.
- This array is consumed by the training script for `GroupKFold` splitting.

**Acceptance Criteria:**
- `patches/file_ids.npy` exists after running `generate_patches.py`
- Each unique file in `manifest.json` maps to a unique integer in `file_ids.npy`
- Array length matches `patches.npy` and `labels.npy`

### Task 1.2: Rewrite training script with 5-fold file-level CV + SHAP

**File:** `.omc/classifier/train_classifier.py`

**Work:**
- Remove MLPClassifier and VotingClassifier imports and usage.
- Replace `train_test_split` with `GroupKFold(n_splits=5)` using `file_ids` as groups.
- Train a single `GradientBoostingClassifier` pipeline (StandardScaler -> PCA -> GB).
- Report per-fold metrics (accuracy, precision, recall, F1) and mean/std.
- After CV, retrain on full dataset and save as `fp_classifier.joblib`.
- Add SHAP integration: compute `shap.TreeExplainer` on the PCA-transformed test fold, save a sample SHAP summary plot to `.omc/classifier/shap_summary.png`.
- Save SHAP expected value and feature names metadata alongside the model.

**Acceptance Criteria:**
- Running `train_classifier.py` prints 5-fold CV metrics with mean +/- std
- No fold contains patches from a file that appears in another fold's split
- `fp_classifier.joblib` is a single GradientBoosting pipeline (no MLP)
- `shap_summary.png` generated
- No imports of `MLPClassifier` or `VotingClassifier` remain

### Task 1.3: Add SHAP attribution to fp_filter.py

**File:** `.omc/coordination/fp_filter.py`

**Work:**
- After loading the model, instantiate `shap.TreeExplainer` on the GB step of the pipeline.
- For each detection, compute SHAP values on the PCA-reduced feature vector.
- Return enriched output: `filter_detections()` returns `list[float]` (unchanged interface), but also exposes a `filter_detections_explained()` that returns `list[dict]` with keys: `time`, `confidence`, `shap_top_features` (top 5 PCA components by absolute SHAP value), `shap_values`.
- The detector's existing call to `filter_detections` remains unchanged (backward compatible).

**Acceptance Criteria:**
- `filter_detections(audio, sr, detections)` still returns `list[float]` — no breaking change
- `filter_detections_explained(audio, sr, detections)` returns list of dicts with SHAP attribution
- Each dict contains at minimum: `time`, `confidence`, `shap_top_features`
- SHAP values sum to approximately (prediction - expected_value) for each detection

### Task 1.4: Update program.md constraints

**File:** `program.md`

**Work:**
- Add tree-based ML (GradientBoosting, XGBoost, RandomForest) to the "Allowed techniques" list.
- Add SHAP explainability as a requirement for any ML model used.
- Change eval time constraint from "under 60 seconds" to "under 5 minutes target" with note that the 60s cap applies to DSP-only eval, ML post-processing is additive.
- Explicitly keep the "NO neural networks" constraint.

**Acceptance Criteria:**
- "Allowed techniques" section includes tree-based ML with SHAP requirement
- Eval time language reads "< 5 min target" not "under 60 seconds"
- "NO neural networks" constraint preserved verbatim
- No other constraints weakened

### Task 1.5: Add `shap` dependency

**File:** `pyproject.toml`

**Work:**
- Add `shap>=0.43.0` to dependencies.
- Run `uv lock` to update lockfile.

**Acceptance Criteria:**
- `uv run python -c "import shap; print(shap.__version__)"` succeeds
- `uv.lock` updated

---

## Phase 2: Zeroth-Korean Data Integration + Retrain (Next)

### Guardrails

**Must Have:**
- Korean speech audio integrated into training data
- Classifier retrained with Korean speech patches
- Clean_score maintained or improved on Korean audio

**Must NOT Have:**
- Modifications to the DSP detector for Korean-specific tuning (that is autoresearch's job)
- Download scripts that assume network access at runtime

### Task 2.1: Zeroth-Korean data preparation

**Work:**
- Verify Zeroth-Korean FLAC extraction at `data/zeroth-korean/` is complete (9.6GB).
- Write a script `.omc/classifier/prepare_korean_patches.py` that:
  - Loads Korean speech FLAC files.
  - Runs them through `detect_splices()` — these are clean recordings, so all detections are FPs.
  - Generates negative (label=0) patches for each detection.
  - Also generates synthetic splices by concatenating random segments from different speakers, creating positive (label=1) patches with known splice points.
  - Outputs `korean_patches.npy`, `korean_labels.npy`, `korean_file_ids.npy`.

**Acceptance Criteria:**
- Script runs on extracted Zeroth-Korean data without errors
- Produces patches with both positive and negative labels
- Synthetic splices use cross-speaker concatenation (different speaker IDs)
- File IDs are unique and do not collide with existing patch file IDs

### Task 2.2: Merge datasets and retrain

**Work:**
- Update `train_classifier.py` to accept `--korean` flag that loads and merges Korean patches with existing patches.
- Re-run 5-fold file-level CV on merged dataset.
- Compare metrics (F1, precision, recall, clean_score) with and without Korean data.
- Save updated model if metrics improve.

**Acceptance Criteria:**
- `train_classifier.py --korean` merges both patch sets
- CV metrics reported for merged dataset
- Comparison table: {metric: baseline_value, korean_value, delta}
- Model only replaced if combined score improves

---

## Phase 3: Full Autoresearch Integration with Relaxed Eval (Later)

### Guardrails

**Must Have:**
- Autoresearch loop uses the ML-filtered detector seamlessly
- SHAP report generated per evaluation run
- Evaluation completes within 5-minute target

**Must NOT Have:**
- Changes to the autoresearch experiment loop logic (run_autoresearch.sh)
- Classifier retraining during autoresearch runs

### Task 3.1: SHAP report format in detection output

**Work:**
- When `detector.py` calls `fp_filter.py`, optionally generate a SHAP report file.
- Define report format: JSON file per evaluation run with structure:
  ```json
  {
    "run_id": "<commit_hash>",
    "timestamp": "<ISO8601>",
    "detections": [
      {
        "file": "example.wav",
        "time_s": 12.5,
        "confidence": 0.87,
        "verdict": "keep",
        "shap_attribution": {
          "top_features": [
            {"component": "PCA_3", "shap_value": 0.15, "direction": "positive"},
            {"component": "PCA_7", "shap_value": -0.08, "direction": "negative"}
          ],
          "expected_value": 0.42
        }
      }
    ]
  }
  ```
- Save to `.omc/classifier/reports/<run_id>.json`.

**Acceptance Criteria:**
- SHAP report JSON generated for each evaluation run
- Each detection includes top-5 SHAP feature attributions
- Report is valid JSON and parseable
- Report generation adds < 30s to total eval time

### Task 3.2: End-to-end integration test

**Work:**
- Run full evaluation pipeline: `uv run prepare.py` with ML filter + SHAP active.
- Verify total time < 5 minutes.
- Verify SHAP report generated.
- Verify no regression in combined score vs. DSP-only baseline.
- Document baseline vs ML-integrated metrics.

**Acceptance Criteria:**
- `uv run prepare.py` completes in < 5 minutes with ML filter active
- SHAP report exists in `.omc/classifier/reports/`
- Combined score >= baseline combined score (from `.omc/coordination/baseline_metrics.json`)
- No clean file false positives introduced by ML filter

---

## Success Criteria (Overall)

1. **Phase 1:** 5-fold file-level CV implemented, SHAP attribution per detection, MLP removed, program.md updated. All verifiable by running `train_classifier.py` and inspecting output.
2. **Phase 2:** Korean speech data integrated, classifier retrained, metrics compared. Verifiable by running `train_classifier.py --korean`.
3. **Phase 3:** SHAP reports generated per eval run, total eval < 5 min, no regression. Verifiable by running `prepare.py` end-to-end.

---

## Open Questions

- [ ] Is the Zeroth-Korean extraction complete? `data/zeroth-korean/` appears empty — need to confirm status before Phase 2 can begin.
- [ ] What is the current combined score baseline? Need to check `baseline_metrics.json` to set a regression threshold.
- [ ] Should SHAP reports be human-readable (HTML with plots) or machine-readable (JSON only)? Affects Phase 3 report format.
- [ ] For Korean synthetic splices, how many cross-speaker concatenations are sufficient? Need to balance dataset size vs training time.
