# autoresearch-splice

Autonomous research for **audio splice detection using classical signal processing**.

## Research goal

Maximize **combined score** on a labeled test set.
The evaluation oracle prints multiple metrics. The LAST line is always:
- `combined: 0.XXX` — THIS is the metric to optimize. Use it for keep/discard decisions.

When `--with-classifier` is used, `combined` equals the classifier-filtered F1 × clean_score
(`combined_full`). The classifier trains on pre-generated patches (singing + Korean) plus
eval-set patches with OOF cross-validation. There is no quality gate — `combined_full` is
always used. If the classifier hurts the score, the experiment is naturally discarded.

**DSP FP bound**: DSP clean_fp must stay ≤ 15. If exceeded, `combined` drops to 0 (guaranteed discard).
The autoresearch agent can freely adjust DSP threshold values to increase recall, as long as
DSP FP stays within this bound. The classifier handles FP suppression after DSP.

## Constraints — NON-NEGOTIABLE

- **NO neural networks**. No torch, no tensorflow, no sklearn MLPs, no gradient descent.
  Explainable tree-based ML (GradientBoosting) is used automatically by the eval oracle —
  do NOT add ML code to detector.py.
- **NO GPU**. CPU-only. scipy, librosa, numpy only.
- **Every detection must be explainable**: each splice point returned by `detector.py`
  must be attributable to a specific statistical test or spectral anomaly.
- **Each full evaluation run must complete in under 240 seconds** on a laptop CPU. (Raised from 120s to accommodate cross-domain eval: singing + korean-splice DSP + classifier always regenerates korean-splice patches from current detector.)
- **DSP clean_fp ≤ 15**: If DSP alone produces more than 15 FP on clean files,
  combined drops to 0 and the iteration is discarded.
- **Adjusting DSP threshold values** (GPD_ALPHA, CPE_CONFIRM_SIGMA, T2_ALPHA, etc.)
  is a valid hypothesis type.

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

- **`evaluate.py`** — evaluation oracle (protected). Runs `detector.py` on test data,
  computes F1 vs ground truth, prints metrics. **Do NOT modify** (only the human edits this).
- **`detector.py`** — splice detector. Runs a multi-class GBM over features from
  `features.py` at every `ANALYSIS_STRIDE_S` candidate. Edit `GBM_THRESHOLD`,
  `GBM_MIN_SEP_S`, or `ANALYSIS_STRIDE_S` for instant-effect experiments. Must stay
  importable as `detect_splices(audio, sr) -> list[float]`.
- **`features.py`** — 75-dim feature extractor feeding the GBM. Extending
  FEATURE_NAMES requires retraining via `.omc/classifier/train_classifier.py`.
- **`.omc/classifier/train_classifier.py`** — multi-class GBM trainer. `make_pipeline()`
  is the GBM hyperparameter knob (`n_estimators`, `max_depth`, `learning_rate`,
  `subsample`). Retrain with `uv run python .omc/classifier/train_classifier.py`.
- **`program.md`** — instructions for the agent (this file). Only the human edits this.

## Setup

1. Agree on a run tag (e.g. `apr15`). Branch `autoresearch/<tag>` must not exist.
2. `git checkout -b autoresearch/<tag>` from main.
3. Read `README.md`, `evaluate.py`, and `detector.py` in full.
4. Verify `data/spliced/` contains WAV files and `ground_truth.json`.
5. Initialize `results.tsv` with just the header row.
6. Confirm and begin.

## Experiment loop

LOOP FOREVER:

1. Check git state (current branch and commit).
2. Read `detector.py` and `results.tsv` to understand where you are.
3. Form a **hypothesis**: a specific idea expected to improve combined score.
   Valid knobs: detector.py constants (GBM_THRESHOLD, GBM_MIN_SEP_S,
   ANALYSIS_STRIDE_S), features.py feature additions (requires retrain), or
   .omc/classifier/train_classifier.py hyperparameters (requires retrain).
   Write it as a one-line comment at the top of your planned change.
4. Edit the relevant file(s) with the smallest viable change. If the change is
   feature- or hyperparameter-level, also run
   `uv run python .omc/classifier/train_classifier.py` to refresh the bundle.
5. `git commit -m "hypothesis: <one line>"`
6. Run evaluation: `uv run evaluate.py --shap > run.log 2>&1`
   - Must finish in <240s. If it hangs past 270s, kill it (treat as crash).
7. Read results: `grep "^splice_f1:\|^clean_score:\|^combined:" run.log`
8. Log to `results.tsv` (untracked):
   `commit  combined  splice_f1  clean_score  precision  recall  fp_rate  clean_fp  status  description`
9. If `combined` **improved** (strictly higher):
   a. **Verify**: Run `uv run python .omc/coordination/verify_agent.py --agent-name autoresearch --reported-combined <score>`
   b. If verify **PASSES** (exit 0): keep the commit, advance branch. Log status=`keep`.
   c. If verify **FAILS** (exit 1): `git reset --hard HEAD~1`. Log status=`verify-fail`. Treat as discard.
10. If equal or worse: `git reset --hard HEAD~1`. Log status=`discard`.
11. **Stop signal**: If `.omc/autoresearch-stop` exists, stop the loop immediately.
12. Print `RESULT:<status>` as the last line (keep, discard, or verify-fail).

## results.tsv format

Tab-separated. Do NOT use commas in descriptions.

```
commit	combined	splice_f1	clean_score	precision	recall	fp_rate	clean_fp	status	description
a1b2c3d	0.000000	0.00	0.00	0.00	keep	baseline
b2c3d4e	0.712000	0.80	0.64	0.05	keep	spectral flux peak-pick 3-sigma
c3d4e5f	0.690000	0.75	0.64	0.08	discard	lower threshold hurt precision
```

## Simplicity criterion

All else equal, simpler is better. A marginal F1 gain with 30 extra lines is not worth it.
Deleting code and matching or beating prior F1 is always a win.

## Crash handling

If `evaluate.py` crashes (import error, shape mismatch, etc.):
- If it's a trivial fix (typo, wrong axis), fix and re-run.
- If the idea is fundamentally broken, log as `crash` and reset.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human anything.
Run experiments indefinitely until manually interrupted.
If you run out of ideas: read DSP papers in code comments, recombine near-misses,
try more radical feature combinations. The loop never stops.
