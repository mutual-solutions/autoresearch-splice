# autoresearch-splice

Autonomous research for **audio splice detection using classical signal processing**.

## Research goal

Maximize **combined score** = F1 × clean_score on a labeled test set.
Three metrics reported by the evaluation oracle:
- `splice_f1: 0.XXX` — F1 score on splice detection (higher is better)
- `clean_score: 0.XXX` — 1.0 if zero FP on clean files, penalized per clean FP (higher is better)
- `combined: 0.XXX` — F1 × clean_score. THIS is the metric to optimize. Both must be high.

Use `combined` for keep/discard decisions. An improvement in F1 that adds clean FPs is NOT an improvement.

## Constraints — NON-NEGOTIABLE

- **NO neural networks**. No torch, no tensorflow, no sklearn MLPs, no gradient descent.
- **NO GPU**. CPU-only. scipy, librosa, numpy only.
- **Every detection must be explainable**: each splice point returned by `detector.py`
  must be attributable to a specific statistical test or spectral anomaly.
- **Each full evaluation run must complete in under 60 seconds** on a laptop CPU.

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

- **`prepare.py`** — immutable evaluation oracle. Runs `detector.py` on test data,
  computes F1 vs ground truth, prints metrics. **Do NOT modify.**
- **`detector.py`** — the single file you edit. Must be importable as a module
  (`detect_splices(audio, sr)`) AND runnable as a CLI (`python detector.py file.wav`).
- **`program.md`** — instructions for the agent (this file). Only the human edits this.

## Setup

1. Agree on a run tag (e.g. `apr15`). Branch `autoresearch/<tag>` must not exist.
2. `git checkout -b autoresearch/<tag>` from main.
3. Read `README.md`, `prepare.py`, and `detector.py` in full.
4. Verify `data/spliced/` contains WAV files and `ground_truth.json`.
5. Initialize `results.tsv` with just the header row.
6. Confirm and begin.

## Experiment loop

LOOP FOREVER:

1. Check git state (current branch and commit).
2. Read `detector.py` and `results.tsv` to understand where you are.
3. Form a **hypothesis**: a specific classical DSP idea expected to improve F1.
   Write it as a one-line comment at the top of your planned change.
4. Edit `detector.py` with the smallest viable change that tests the hypothesis.
5. `git commit -m "hypothesis: <one line>"`
6. Run evaluation: `uv run prepare.py > run.log 2>&1`
   - Must finish in <60s. If it hangs past 90s, kill it (treat as crash).
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

If `prepare.py` crashes (import error, shape mismatch, etc.):
- If it's a trivial fix (typo, wrong axis), fix and re-run.
- If the idea is fundamentally broken, log as `crash` and reset.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human anything.
Run experiments indefinitely until manually interrupted.
If you run out of ideas: read DSP papers in code comments, recombine near-misses,
try more radical feature combinations. The loop never stops.
