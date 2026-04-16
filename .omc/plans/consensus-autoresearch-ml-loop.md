# Consensus Plan: Autoresearch ML-in-Loop Integration

## ADR
- **Decision**: Integrate GradientBoosting classifier into autoresearch eval loop
- **Drivers**: DSP produces only 1 FP → classifier has no training data. Loosening DSP thresholds + in-loop classifier training allows joint optimization.
- **Alternatives considered**: (A) Background classifier training (rejected: stale classifiers, no joint optimization), (B) Subprocess orchestration in shell (rejected: fragile parsing, slower), (C) Classifier inside detector.py (rejected: autoresearch agent could break it)
- **Why chosen**: Direct import in prepare.py via helper module. OOF predictions eliminate circularity. Quality gate prevents garbage classifier from derailing search.
- **Consequences**: ~60s per iteration (was ~30s). Classifier CV F1 < 0.60 falls back to DSP-only.
- **Follow-ups**: Tune quality gate threshold after initial runs. Consider incremental training if dataset grows.

## RALPLAN-DR

### Principles
1. Backward compatibility: `prepare.py` without `--with-classifier` must produce identical output
2. Separation of concerns: DSP research (detector.py) stays decoupled from ML pipeline
3. Bounded freedom: DSP FP ≤ 15 ceiling prevents threshold race-to-bottom
4. Honest metric: OOF predictions, no train-on-test contamination
5. Fail-safe bootstrap: System works on first iteration with no prior classifier

### Decision Drivers
1. `combined:` must remain the single optimization target — same regex, same parsing
2. Autoresearch agent only edits detector.py — everything else is protected
3. ~60s iteration budget is acceptable

### Viable Options
- **Option A (chosen)**: prepare.py imports helper `ml_eval.py`, runs ML pipeline when `--with-classifier` is set
- **Option B (rejected)**: Shell orchestration via subprocesses — fragile string parsing between steps, 3-5s overhead, error propagation across subprocesses is brittle

## Implementation Steps

### Step 0: pyproject.toml — Add dependencies
Add to `[project] dependencies`:
```
scikit-learn>=1.3.0
joblib>=1.3.0
```
Then `uv lock && uv sync`.

Note: `shap` is NOT added as hard dependency — it's only used by `fp_filter.py`'s explained mode (triggered by env var), not by the eval loop.

### Step 1: prepare.py — Minimal changes
1. Update docstring: "IMMUTABLE" → "Evaluation oracle. Protected from autoresearch agent modification."
2. Refactor `evaluate()`: return per-file results alongside aggregates. Add `per_file` key to return dict:
   ```python
   per_file_results.append({
       "name": name,
       "path": wav_path,
       "gt_times": case["gt_times"],
       "det_times": det_times,
       "spliced": case["spliced"],
       "tier": case["tier"],
   })
   ```
   Do NOT store audio/sr arrays — too much memory. `ml_eval.py` reloads audio from path.
3. Add `--with-classifier` argparse flag
4. When flag is set (at bottom of `__main__`):
   ```python
   if args.with_classifier:
       from ml_eval import evaluate_with_classifier
       ml_result = evaluate_with_classifier(result, args.data_dir)
       # ml_eval prints its own metrics
       # Print combined: as the LAST line
       if ml_result.get("bound_exceeded"):
           print(f"combined: 0.000000")  # NOT exit code 2
       elif ml_result.get("classifier_quality") == "LOW":
           print(f"combined: {result['combined']:.6f}")
       else:
           print(f"combined: {ml_result['combined_full']:.6f}")
   ```
5. When flag is NOT set: print `combined:` exactly as before (no change)

**Key: `combined:` is always present, always the LAST metric line printed.** No exit code 2 — the autoresearch agent and shell would break. Instead, FP bound exceeded → combined: 0.000000 (guaranteed discard since it's lower than any previous best).

### Step 2: ml_eval.py (NEW) — ML orchestration helper
Located at project root alongside prepare.py.

```python
"""ML evaluation helper for prepare.py --with-classifier."""

import os
import sys
import numpy as np
import soundfile as sf

# Add classifier paths
_proj = os.path.dirname(os.path.abspath(__file__))
_classifier_dir = os.path.join(_proj, ".omc", "classifier")
_coord_dir = os.path.join(_proj, ".omc", "coordination")
for p in [_classifier_dir, _coord_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from prepare import match_detections, TOLERANCE_S
from generate_patches import extract_mel_patch, is_tp
from train_classifier import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
import joblib


DSP_FP_BOUND = 15
CLASSIFIER_QUALITY_GATE = 0.60
N_FOLDS = 5
```

Function: `evaluate_with_classifier(dsp_results, data_dir) -> dict`:

1. Extract per-file data from `dsp_results["per_file"]`
2. Count dsp_clean_fp from `dsp_results["clean_fp"]`
3. Print `combined_dsp: {dsp_results["combined"]:.6f}`
4. Print `dsp_clean_fp: {dsp_results["clean_fp"]}`
5. If dsp_clean_fp > DSP_FP_BOUND:
   - Print `dsp_fp_bound: EXCEEDED`
   - Return `{"bound_exceeded": True, "dsp_clean_fp": N}`
6. Generate patches in-memory:
   - For each file in per_file: reload audio, extract mel patches at each detection time
   - Label: TP if within 1s of GT, FP otherwise
   - Add random negatives from clean files (3 per file) and spliced files (2 per file)
   - Build file_ids array for GroupKFold grouping
7. If total patches < 20 or unique files < 5:
   - Print `classifier_quality: INSUFFICIENT_DATA`
   - Return `{"classifier_quality": "LOW", "cv_f1": 0.0, "combined_full": None}`
8. 5-fold GroupKFold CV:
   - For each fold: train pipeline, predict on test fold
   - Collect OOF predictions: `{(file_name, det_time): proba}` for each test-fold detection
   - Compute per-fold F1
9. Compute classifier CV F1 from ALL OOF predictions
10. Print `classifier_cv_f1: {cv_f1:.6f}`
11. If cv_f1 < CLASSIFIER_QUALITY_GATE:
    - Print `classifier_quality: LOW`
    - Return `{"classifier_quality": "LOW", "cv_f1": cv_f1, "combined_full": None}`
12. Use OOF predictions to filter detections per file:
    - For each file: remove detections where OOF proba < 0.5
    - Re-run `match_detections(gt_times, filtered_det_times, TOLERANCE_S)` per file
    - Aggregate TP/FP/FN across all files
13. Compute filtered metrics:
    - precision, recall, F1 from filtered aggregates
    - clean_fp from filtered clean file detections
    - clean_score = 1.0 - (filtered_clean_fp / max(clean_files, 1))
    - combined_full = filtered_F1 × filtered_clean_score
14. Print `classifier_quality: OK`
15. Print `combined_full: {combined_full:.6f}`
16. Train final model on ALL data, save to `.omc/classifier/fp_classifier.joblib` (for web demo)
17. Return result dict

### Step 3: CLAUDE.md — Update protection language
Change line 25:
```
- `prepare.py` and `program.md` are immutable -- never modify them.
```
To:
```
- `prepare.py` and `program.md` are protected -- only the human modifies them. The autoresearch agent must never modify them.
```

### Step 4: program.md — Update autoresearch instructions
- Optimization target: `combined` (prints combined_full when classifier is good, combined_dsp otherwise)
- Replace "NO neural networks" with: "NO neural networks. Explainable tree-based ML (GradientBoosting) is used automatically by the eval oracle — do not add ML code to detector.py."
- Add DSP FP bound: "DSP clean_fp must stay ≤ 15. If exceeded, combined drops to 0 (guaranteed discard)."
- Eval command: `uv run prepare.py --with-classifier`
- Time budget: "Each full evaluation run must complete in under 120 seconds"
- Add: "Adjusting DSP threshold values (GPD_ALPHA, CPE_CONFIRM_SIGMA, T2_ALPHA, etc.) is a valid hypothesis type."
- Keep/discard: same rule (if combined improved → keep)

### Step 5: run_autoresearch.sh — Update loop
- Claude prompt: change eval command to `uv run prepare.py --with-classifier`
- Parsing: `combined:` regex unchanged (`grep -oE 'combined[: ]+[0-9]+\.[0-9]+' | tail -1` already picks last match)
- Also capture `combined_dsp:` for logging to results.tsv
- Remove background classifier training blocks:
  - keep-pending handler: remove lines 127-132 (the `(cd "$PROJECT_DIR" && ...` background block)
  - legacy keep handler: remove lines 179-184 (same pattern)
  - KEEP version management code (tags, snapshots, versions.json) intact
- Timeout: change 90s kill threshold to 150s
- Update results.tsv header: add combined_dsp column

### Step 6: verify_agent.py — Update verification
- Fix regex: `re.search(r"combined:\s*([\d.]+)", output)` → 
  `matches = re.findall(r"^combined:\s*([\d.]+)", output, re.MULTILINE)` then `float(matches[-1])`
- check_metric_rerun: run `prepare.py --with-classifier` instead of `prepare.py`
- Add check: parse `dsp_clean_fp:` line and verify ≤ 15
- Keep prepare.py in PROTECTED_FILES (protection is against agent, human changes are committed)

### Step 7: test_regression.py — Update regression tests
- Keep all existing DSP-only checks unchanged
- Add: `dsp_clean_fp ≤ 15` check
- Add optional `--with-classifier` test block:
  ```python
  try:
      from ml_eval import evaluate_with_classifier
      # ... run and check combined >= 0.40
  except ImportError:
      print("  ML eval: SKIP (sklearn not available)")
  ```
- If classifier_quality == "LOW": skip combined_full threshold (expected early on)

### Step 8: baseline_metrics.json — Manual update
After first successful `prepare.py --with-classifier` run:
- Add `combined_dsp`, `dsp_clean_fp`, `classifier_cv_f1` fields
- This is a manual step before starting the autoresearch loop

## Acceptance Criteria
1. `uv run prepare.py --with-classifier` runs end-to-end, prints combined_dsp and combined
2. `uv run prepare.py` without flag produces identical output to current version
3. DSP FP bound (≤15) enforced: combined drops to 0 when exceeded
4. OOF predictions used for combined_full (not final model)
5. Quality gate: CV F1 < 0.60 → combined = combined_dsp
6. program.md updated with new metric, eval command, FP bound
7. run_autoresearch.sh uses --with-classifier, parses combined
8. Background classifier training removed from run_autoresearch.sh
9. test_regression.py includes dsp_clean_fp check
10. verify_agent.py regex fixed and checks combined + dsp_fp_bound
11. Full iteration ≤ 120s

## Edge Cases
- **GroupKFold < 5 groups**: If fewer than 5 unique files have detections, reduce n_splits to min(5, n_groups). If n_groups < 2, return classifier_quality: LOW.
- **Zero FP patches**: If DSP produces 0 FP, classifier trains on GT-missed TPs + random negatives only. Quality gate will fire (CV F1 < 0.60), falling back to DSP-only.
- **sklearn not installed**: `--with-classifier` import fails → print error and exit 1 (not silent fallback).

## Files Changed
1. `pyproject.toml` — add sklearn, joblib
2. `prepare.py` — add --with-classifier flag, return per_file results, update docstring
3. `ml_eval.py` (NEW) — ML orchestration helper
4. `CLAUDE.md` — immutable → protected
5. `program.md` — new metric, eval command, FP bound, time budget
6. `run_autoresearch.sh` — eval command, remove background training, timeout
7. `.omc/coordination/verify_agent.py` — fix regex, add dsp_fp_bound check
8. `test_regression.py` — add dsp_clean_fp check, optional ML test
9. `.omc/coordination/baseline_metrics.json` — add new fields (manual)
