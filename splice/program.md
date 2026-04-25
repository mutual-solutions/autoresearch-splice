# autoresearch-splice

Autonomous research for **audio splice detection using classical signal processing**.

## Research goal

Maximize **combined score** on a labeled evaluation set.
The evaluation oracle prints multiple metrics. The LAST line is always:
- `combined: 0.XXX` — THIS is the metric to optimize. Use it for keep/discard decisions.

`combined` equals `boundary_f1`: boundary-level F1 with a **250 ms collar** tolerance,
evaluated on `data/eval/korean_iter1/eval/`. The detector returns a list of
`(time_s, label)` pairs; a predicted boundary matches a ground-truth boundary if it
falls within ±250 ms and the label is correct.

**3-class detection**: labels are `cross_voice`, `same_voice_edit`, and `unknown`.
- Return `unknown` by default — emit a specific label only when class-specific evidence
  supports it.
- Per-class F1 is reported in `splice_class_breakdown` for diagnostic purposes.
  `unknown`-labeled predictions are excluded from per-class F1 but **do count** in the
  aggregate `boundary_f1`.

There are no splice-free files in this corpus. `combined` is purely `boundary_f1`;
no FP-penalty multiplier is applied.

## Constraints — NON-NEGOTIABLE

- **NO neural networks**. No torch, no tensorflow, no sklearn MLPs, no gradient descent.
  Explainable tree-based ML (HistGradientBoosting, multinomial) is the only allowed
  classifier. Do NOT add neural-network code to detector.py.
- **NO GPU**. CPU-only. scipy, librosa, numpy, sklearn only.
- **Every detection must be explainable**: each splice point returned by `detector.py`
  must be attributable to a specific statistical test or spectral anomaly.
- **Each full evaluation run must complete in under 240 seconds** on a laptop CPU.
- **250 ms collar tolerance**: a predicted boundary matches ground truth if it falls
  within ±250 ms of the true boundary time.
- **Voice-pair holdout** (fixed, do not change):
  - `test` split: speakers DaeBuHo, Kanna
  - `eval` split: speakers Sunwoo, Joon
  - `train` split: remaining 7 speakers
- **Augmentation chain** (already baked into corpus on disk; the loop does NOT
  re-augment): deterministic, file-hash seeded — synthetic exp-decay RIR
  (T60 ~ U(0.2, 0.6)) + pink noise (SNR ~ N(22, 4) clipped to [10, 35]) +
  Opus 32 kbps codec roundtrip.
- **Adjusting detector constants** (thresholds, stride, window geometry) is a valid
  hypothesis type.

Allowed techniques (non-exhaustive):
- Spectral flux, spectral centroid, spectral rolloff, MFCC discontinuities
- Phase discontinuity analysis (STFT phase unwrapping)
- Noise floor / RMS energy estimation
- Welch's t-test, KS-test, Mann-Whitney U, likelihood ratio tests
- Autocorrelation, zero-crossing rate
- Room impulse response fingerprinting (cepstral distance)
- Bispectrum / higher-order statistics
- GMM on hand-crafted features (fit per segment, compare likelihoods)
- Peak-picking, threshold detection, change-point detection (CUSUM, PELT)

## Files

- **`splice/evaluate.py`** — evaluation oracle (protected). Runs `splice/detector.py`
  on `data/eval/korean_iter1/eval/`, computes boundary-F1 with 250 ms collar vs ground
  truth, prints metrics. **Do NOT modify** (only the human edits this).
- **`splice/detector.py`** — splice detector. Returns `list[(time_s, label)]` where
  `label ∈ {"cross_voice", "same_voice_edit", "unknown"}`. Default label is `"unknown"`;
  emit a specific label only with class-specific evidence. Edit `GBM_THRESHOLD`,
  `GBM_MIN_SEP_S`, or `ANALYSIS_STRIDE_S` for instant-effect experiments.
- **`splice/features.py`** — feature extractor feeding the GBM. Extending `FEATURE_NAMES`
  requires retraining via `splice/classifier/train_classifier.py`.
- **`splice/classifier/train_classifier.py`** — 3-class HistGradientBoosting (multinomial)
  trainer. Training data sourced from `data/eval/korean_iter1/train/`. `make_pipeline()`
  is the GBM hyperparameter knob (`n_estimators`, `max_depth`, `learning_rate`,
  `subsample`). Retrain with `uv run python splice/classifier/train_classifier.py`.
- **`splice/program.md`** — instructions for the agent (this file). Only the human edits this.

## Setup

1. Branch: `autoresearch/korean-iter1` (checked out from `apr15-final-2026-04-25`).
2. Read `README.md`, `splice/evaluate.py`, and `splice/detector.py` in full.
3. Verify `data/eval/korean_iter1/{train,eval,test}/` each contain per-conversation
   `.opus` + `.json` files and a per-split `ground_truth.json`.
4. Initialize `results.tsv` with just the header row.
5. Confirm `autoresearch/baseline_metrics.json` exists with the korean-iter1 schema
   (Step 12 of the pivot plan writes this from the first real eval run).
6. Confirm and begin.

## Experiment loop

LOOP FOREVER:

1. Check git state (current branch and commit).
2. Read `splice/detector.py` and `results.tsv` to understand where you are.
3. Form a **hypothesis**: a specific idea expected to improve combined score.
   Valid knobs: splice/detector.py constants (GBM_THRESHOLD, GBM_MIN_SEP_S,
   ANALYSIS_STRIDE_S), splice/features.py feature additions (requires retrain), or
   splice/classifier/train_classifier.py hyperparameters (requires retrain).
   Write it as a one-line comment at the top of your planned change.
4. Edit the relevant file(s) with the smallest viable change. If the change is
   feature- or hyperparameter-level, also run
   `uv run python splice/classifier/train_classifier.py` to refresh the bundle.
5. `git commit -m "hypothesis: <one line>"`
6. Run evaluation: `uv run python splice/evaluate.py --shap > run.log 2>&1`
   - Must finish in <240s. If it hangs past 270s, kill it (treat as crash).
7. Read results: `grep "^boundary_f1:\|^combined:\|^precision:\|^recall:" run.log`
8. Log to `results.tsv` (untracked):
   `commit  combined  precision  recall  cross_voice_f1  same_voice_edit_f1  unknown_label_count  status  description`
9. If `combined` (= boundary_f1) **improved** (strictly higher):
   a. **Verify**: Run `uv run python autoresearch/supervisor_agent.py --verify --agent-name autoresearch --reported-combined <score>`
   b. If verify **PASSES** (exit 0): keep the commit, advance branch. Log status=`keep`.
   c. If verify **FAILS** (exit 1): `git reset --hard HEAD~1`. Log status=`verify-fail`. Treat as discard.
   d. Per-class breakdown (`cross_voice_f1`, `same_voice_edit_f1`) is informational — it
      does not gate keep/discard decisions.
10. If equal or worse: `git reset --hard HEAD~1`. Log status=`discard`.
11. **Stop signal**: If `.omc/autoresearch-stop` exists, stop the loop immediately.
12. Print `RESULT:<status>` as the last line (keep, discard, or verify-fail).

## results.tsv format

Tab-separated. Do NOT use commas in descriptions.

```
commit	combined	precision	recall	cross_voice_f1	same_voice_edit_f1	unknown_label_count	status	description
a1b2c3d	0.000000	0.00	0.00	0.00	0.00	0	keep	baseline
b2c3d4e	0.712000	0.80	0.65	0.70	0.68	4	keep	spectral flux peak-pick 3-sigma
c3d4e5f	0.690000	0.75	0.62	0.64	0.61	7	discard	lower threshold hurt precision
```

## Simplicity criterion

All else equal, simpler is better. A marginal F1 gain with 30 extra lines is not worth it.
Deleting code and matching or beating prior F1 is always a win.

## Crash handling

If `splice/evaluate.py` crashes (import error, shape mismatch, etc.):
- If it's a trivial fix (typo, wrong axis), fix and re-run.
- If the idea is fundamentally broken, log as `crash` and reset.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human anything.
Run experiments indefinitely until manually interrupted.
If you run out of ideas: read DSP papers in code comments, recombine near-misses,
try more radical feature combinations. The loop never stops.
