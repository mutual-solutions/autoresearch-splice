# Deep Interview Spec: Test-Set Eval Gate (Overfitting Detector)

## Metadata
- Interview ID: 2026-04-27-test-set-eval-gate
- Rounds: 4
- Final Ambiguity Score: 8.75%
- Type: brownfield
- Generated: 2026-04-27 (KST)
- Threshold: 20%
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.95 | 0.35 | 0.333 |
| Constraint Clarity | 0.90 | 0.25 | 0.225 |
| Success Criteria | 0.85 | 0.25 | 0.213 |
| Context Clarity | 0.95 | 0.15 | 0.143 |
| **Total Clarity** | | | **0.9125** |
| **Ambiguity** | | | **0.0875** |

## Goal

Detect overfitting of the autoresearch loop to the eval-60 sample by adding a
periodic test-set eval that runs against the held-out `data/eval/korean_iter1/test/`
split and surfaces a comparison metric to the operator.

The autoresearch loop currently uses `data/eval/korean_iter1/eval/` (1030 conversations,
deterministic `random.sample(60, seed=0)`) as both the optimization target AND the
keep/discard signal. Every iteration's `combined` is computed on the same 60-file
sample. The agent has been climbing this metric (0.0749 → 0.1512 over 60+ iterations)
with no orthogonal signal that the gain generalizes. The test split (4278 convs,
voice-pair holdout `{DaeBuHo, Kanna}` distinct from eval's `{Sunwoo, Joon}`) has
never been touched by the loop. This iteration adds the operator-side instrumentation
needed to detect when eval-perf and test-perf decouple (= overfitting signature).

## Constraints

- **Loop must remain operational throughout the change.** No long stop window.
  Implementation lands in commits that are safe to apply to a running tree.
- **Test-eval invocation must NOT touch protected files.** `splice/evaluate.py`
  already supports `--data-dir` (line 405), so the existing oracle is reused
  with a different argument. No `MIGRATE-PROTECTED` commit needed.
- **Test-eval auto-fires every 10 keeps**, in the background, AFTER an iteration
  completes. Does NOT block iteration execution. Fires from
  `_maybe_periodic_maintain` in `run_autoresearch.sh:622` or a parallel hook.
- **Sample size: 60 conversations from test split** (matches eval cadence).
  Use a different RANDOM_SEED than the eval sample (e.g., seed=1 instead of
  the eval's seed=0) to ensure the chosen 60 are not accidentally the
  easiest-to-classify by structure.
- **Time budget per fire: 240 s** (matches eval budget). At ~50-min wallclock per
  10-keep window, 4-min test-eval overhead = 8%. Acceptable.
- **Output schema** (`autoresearch/test_baseline.json`) — full RESULTS_TSV
  decomposition + comparison fields:
  ```
  {
    "timestamp": "ISO-8601",
    "git_sha": "<short>",
    "test_combined": float,
    "test_f0_5": float,
    "test_f1": float,
    "test_precision": float,
    "test_recall": float,
    "test_clean_fp_per_min": float,
    "test_clean_fp_penalty": float,
    "test_n_files": 60,
    "test_cross_voice_f1": float,
    "test_same_voice_edit_f1": float,
    "test_unknown_label_count": int,
    "eval_combined_at_time": float,    // copied from baseline_metrics.json
    "gap_pct": float,                  // (test - eval) / eval
    "test_random_seed": 1
  }
  ```
- **History file**: `autoresearch/test_history.jsonl` — append the same dict
  on every fire. Enables time-series plotting and trajectory analysis.
- **WARN policy**: emit `wrapper.test_eval.gap_warn` event via the unified logger
  when `gap_pct < -0.20` (i.e., test_combined is more than 20% below eval).
  No automated halt. Operator decides on intervention.
- **No agent-prompt change.** The agent does NOT see the test result.
  Surfacing it would reintroduce a leakage path (agent optimizes against
  test patterns, eventually overfits both sets).

## Non-Goals

- **No CLEAN_FP_POSITIONS surfacing in this iteration.** That's a separate
  follow-up (Round-1 Goal option 2 / "Push the eval metric higher"). This
  iteration is purely diagnostic instrumentation.
- **No OOF metrics delta emit by `train_classifier.py` in this iteration.**
  Different deliverable.
- **No discard-revert auditor in this iteration.** Different deliverable
  (wrapper integrity, not eval honesty).
- **No automated loop halt on gap detection.** Advisory only. Halt-on-gap
  was explicitly rejected in Round 3 to avoid false-positive halts on
  noisy single-iter spikes.
- **No agent-prompt injection of test results.** Explicitly rejected in
  Round 3 to avoid the leakage path.
- **No splice/evaluate.py modification.** Reuse the existing `--data-dir`
  flag. Protected file stays untouched.
- **No test-set rerun on every iteration.** Every 10 keeps only.

## Acceptance Criteria

- [ ] **AC-1**: `./run_autoresearch.sh test_eval` subcommand exists as a manual
  trigger. Runs `splice/evaluate.py --data-dir data/eval/korean_iter1/test/`
  with `RANDOM_SEED=1` (passed as env var or CLI arg). Writes
  `autoresearch/test_baseline.json` atomically (`tempfile + os.replace`).
  Appends to `autoresearch/test_history.jsonl`.
- [ ] **AC-2**: Manual smoke test passes: `./run_autoresearch.sh test_eval`
  on the current loop's HEAD produces a valid `test_baseline.json` matching
  the schema above, and `test_combined` is a float in `[0, 1]`.
- [ ] **AC-3**: Auto-fire hook in `run_autoresearch.sh` fires after every
  10 keeps. Does NOT block iteration execution (runs in background or
  is bounded by 240s timeout). On fire, writes the artifact and emits
  `wrapper.test_eval.fired` event to the unified logger.
- [ ] **AC-4**: WARN event `wrapper.test_eval.gap_warn` is emitted when
  the computed `gap_pct < -0.20`. Verified by a synthetic test (mock a
  test_combined value that triggers the threshold).
- [ ] **AC-5**: First real auto-fire during the running loop produces a
  valid artifact. Verified by `tail -1 autoresearch/test_history.jsonl`
  showing a recent timestamp + valid schema.
- [ ] **AC-6**: Loop continues running uninterrupted throughout the
  implementation and first auto-fire. `tmux ls` shows session alive,
  `.omc/supervisor-crash-counter.txt` stays at 0, no abnormal
  `loop.ended` event in `.omc/logs/autoresearch.jsonl`.

## Assumptions Exposed & Resolved

| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| Modifying `splice/evaluate.py` is required | Could a non-protected wrapper-side script reuse it? | `evaluate.py:405` already supports `--data-dir`. No protected-file change needed. |
| Test-eval should run on every iteration | Cost vs. signal? | Every 10 keeps is sufficient for trajectory detection without dominating loop overhead. |
| Operator should decide on halt | Or auto-halt on gap? | Auto-halt rejected (Round 3) — too noisy on single-iter spikes; advisory WARN preferred. |
| Agent should see test result | Or stay blind to it? | Agent stays blind (Round 3) — surfacing reintroduces leakage path. |
| Sample 60 is enough | Or use 200, or full 4278? | Sample 60 chosen (Round 4) — matches eval cadence, comparable signal, low overhead. |
| Same RANDOM_SEED as eval | Risk of correlated easy-fits | Different seed (=1) chosen so test sample is structurally independent. |

## Technical Context

### Brownfield findings (verified during interview)

- **`splice/evaluate.py`** (protected): already accepts `--data-dir <path>`,
  defaults to `data/eval/korean_iter1/eval/`. Reuse with
  `--data-dir data/eval/korean_iter1/test/`. Random seed is a module
  constant `RANDOM_SEED = 0` (line 65) — needs CLI/env override path
  for AC-1 (seed=1 on test). One-line edit (NEW arg, doesn't change
  existing behavior — could be done as `MIGRATE-PROTECTED` or via
  env var override that doesn't touch the file).
- **`run_autoresearch.sh`**: editable. Has `_maybe_periodic_maintain()`
  at line 622 that fires every 10 results.tsv rows — same pattern can
  be reused for test-eval.
- **`autoresearch/baseline_metrics.json`**: current best value lives here;
  the script that writes it (in `_do_keep_path`) is the model for the
  atomic-write pattern AC-1 needs.
- **`autoresearch/logger.py`**: unified structured logger (US-515). Use
  `from autoresearch.logger import get_logger; log = get_logger("test_eval")`
  for new emissions. Add `wrapper.test_eval.fired` and
  `wrapper.test_eval.gap_warn` to the event taxonomy in `logger.py`'s
  docstring.
- **Test split inventory** (verified):
  `data/eval/korean_iter1/test/` contains 4278 `.opus` + 4279 `.json`
  (extra is `ground_truth.json`). Voice holdout = `{DaeBuHo, Kanna}`
  per `autoresearch/baseline_metrics.json:voices_holdout.test`.
- **CLAUDE.md test budget**: "Held-out test evaluation must complete in
  under 1200 seconds (20 min)." Sample 60 keeps us at ~240s, well within.

### Affected files (best estimate)

| File | Change type | Protected? |
|------|-------------|------------|
| `run_autoresearch.sh` | NEW: `test_eval` subcommand + auto-fire hook | No |
| `autoresearch/test_baseline.json` | NEW artifact | N/A |
| `autoresearch/test_history.jsonl` | NEW artifact | N/A |
| `autoresearch/logger.py` | Add 2 events to event taxonomy docstring | No |
| `splice/evaluate.py` | OPTIONAL: env-var seed override (~5 LOC); could be avoided by passing seed via wrapper script | YES (would need `MIGRATE-PROTECTED:` prefix if touched) |
| `scripts/test_eval.py` (NEW) | OPTIONAL: thin wrapper around evaluate.py that injects seed override + handles atomic baseline write + history append + gap WARN. Avoids touching protected `evaluate.py`. | No |

**Recommended path**: implement `scripts/test_eval.py` so `splice/evaluate.py`
stays untouched. The new script imports `splice.evaluate.evaluate()`,
overrides `RANDOM_SEED` via monkeypatch or env var, runs against the test
dir, writes artifacts.

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| Eval Sample | core | data_dir, n_files, random_seed | scored by `combined` |
| Test Sample | core (NEW) | data_dir, n_files, random_seed=1 | scored by `test_combined`; held out from optimization |
| Baseline Metrics | core | combined, f0_5, P, R, clean_fp_per_min | written per keep |
| Test Baseline | core (NEW) | test_combined, eval_combined_at_time, gap_pct | written per auto-fire |
| Test History | supporting (NEW) | append-only JSONL of Test Baselines | trajectory of overfit signal |
| Gap Warning | supporting (NEW) | event when gap_pct < -0.20 | advisory; no halt |
| Auto-Fire Hook | supporting (NEW) | trigger every N=10 keeps | runs in background |

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 3 (Eval, Loop, Operator) | 3 | - | - | N/A |
| 2 | 5 (+ Test Sample, Test Baseline) | 2 | - | 3 | 60% |
| 3 | 7 (+ Auto-Fire Hook, Gap Warning) | 2 | - | 5 | 71% |
| 4 | 7 (+ Test History, ~Operator) | 1 | -1 | 6 | 86% |

Final ontology stable at 7 entities; the model is converged.

## Interview Transcript

<details>
<summary>Full Q&A (4 rounds)</summary>

### Round 1 — Goal
**Q:** What outcome are you trying to produce with the next development iteration?
**A:** "Make the loop honest" — anti-overfitting + observability path.
**Ambiguity:** ~70% (bucket clear, scope unclear)

### Round 2 — Goal sub-scope
**Q:** Which subset of the loop-honesty moves is in scope for THIS iteration?
**A:** "Test-set eval gate only" — single surgical change.
**Ambiguity:** ~50%

### Round 3 — Constraints (policy)
**Q:** How should the test-set eval run, and what should happen when the eval/test gap is large?
**A:** "Auto every N keeps + advisory" — WARN at gap > 20%, no halt, no agent-prompt injection.
**Ambiguity:** ~24%

### Round 4 — Success Criteria
**Q:** What does 'done' look like — sample size, artifact schema, completion signal?
**A:** "Standard 60 + full schema" — sample 60 (different seed), full RESULTS_TSV decomposition + history file, smoke-test + first auto-fire.
**Ambiguity:** ~9% (PASSED)

</details>
