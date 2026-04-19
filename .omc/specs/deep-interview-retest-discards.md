# Deep Interview Spec: verify_agent --retest — systematic replay of discards after eval-pipeline fixes

## Metadata
- Interview ID: retest-discards-2026-04-18
- Rounds: 3
- Final Ambiguity: 19%
- Type: brownfield
- Status: PASSED (threshold met)

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|---|---|---|---|
| Goal | 0.80 | 0.35 | 0.28 |
| Constraint | 0.75 | 0.25 | 0.19 |
| Success | 0.85 | 0.25 | 0.21 |
| Context | 0.85 | 0.15 | 0.13 |
| **Total** | | | **0.81** |
| **Ambiguity** | | | **19%** |

## Goal
When a bug in the eval/wrapper pipeline silently corrupted N discards
(e.g., the HistGBM `feature_importances_` AttributeError that just
caused 4+ `combined=0.010` false-catastrophes), the operator must be
able to systematically replay every discard since the bug-introduction
SHA and recover any hypothesis whose *real* combined now beats the
current baseline. Replay runs are owned by `verify_agent.py` (the
project's single verification boundary); full recoveries land as
cherry-picks + baseline updates using the same path as normal keeps.

## Constraints
- Scope: **all discards since the bug-introduction SHA** — operator
  provides the SHA as CLI arg. No catastrophic-only gating — the bug
  might have caused smaller silent drops too.
- Owner: **`verify_agent.py --retest <from-sha>`** subcommand. Extends
  the existing verification boundary; no new separate script.
- Compute budget: ~5 min × N retests (retrain + eval per commit).
  Acceptable — runs only when operator triggers it after a fix.
  Provide `--limit N` to cap retests per invocation.
- Oracle integrity: retest uses the currently-decrypted eval corpus
  (inherits OMC_EVAL_DATA_ROOT from caller). Must NOT re-decrypt;
  must NOT touch eval.tar.gz.enc. Autoresearch loop must be stopped
  so eval corpus lifecycle is stable during retest.
- Safety: `--dry-run` mode prints the report without cherry-picking.
  `--limit N` caps retest count per invocation.
- Cherry-pick conflicts: skip with WARN log; don't force-merge.
- Classifier compatibility: if a discard's features.py sha differs
  from the current, retest invokes train_classifier.py (same
  US-505 staleness gate logic) before eval.

## Non-Goals
- NOT a continuously-running automation — operator-triggered only.
- NOT a retest-on-every-fix trigger; operator decides when to run.
- NOT modifying how future discards get logged — results.tsv stays
  append-only; recoveries append new `retest-keep:` commits, not
  rewrite history.
- NOT a replacement for verify_agent's strict-improvement path.

## Acceptance Criteria
- [ ] `uv run python .omc/coordination/verify_agent.py --retest <from-sha>`
      subcommand exists. Iterates `results.tsv` rows with status=discard
      AND commit-date-after-from-sha, in chronological order (oldest first).
- [ ] For each candidate discard: cherry-pick into a temporary `git worktree`
      (isolation so HEAD of the live repo isn't disturbed), invoke
      train_classifier.py if features.py sha drifted (US-505 reuse), run
      `evaluate.py --shap`, parse combined from RESULTS_TSV.
- [ ] If replayed combined > current baseline strictly: cherry-pick the
      commit permanently onto HEAD; run the existing verify_agent
      structural checks (via direct function call, not subprocess); if
      HIGH/MEDIUM confidence → tag `detector-v<N>`, update
      baseline_metrics.json (single-commit `baseline:` like the wrapper's
      keep path), append a `note: recovered <sha> by retest` to
      research_notes.md. Update `current_baseline` in-memory so later
      retests compare against the just-recovered bar.
- [ ] Order: chronological by commit date (oldest discard first), so a
      recovery increases the bar for subsequent retests — mimics the
      original loop's ordering semantics.
- [ ] `--dry-run` flag: prints report (old combined vs new combined per
      candidate, predicted recoveries) without any cherry-pick or baseline
      update.
- [ ] `--limit N` flag: process at most N candidates in this invocation.
- [ ] Cherry-pick conflict on a specific discard: log WARN, skip that
      one, continue. Do NOT abort the whole run.
- [ ] Running autoresearch loop detected (tmux session alive): refuse
      to start retest with a clear error message. Operator must stop
      first.
- [ ] Output: `.omc/retest_report.md` with per-candidate: original sha,
      original combined, retest combined, delta, outcome (recovered /
      still-lower / conflict / eval-crash).
- [ ] Dry-run on the current state returns in <5s (no eval runs — just
      analyzes results.tsv).
- [ ] Self-test: synthetic results.tsv with one known-bad-and-good
      recovered hypothesis triggers the cherry-pick path end-to-end
      (mockable via evaluate.py stub for the test only — real test on
      the user's machine with real discards).

## Technical Context
- verify_agent.py already hosts: check_metric_rerun (runs evaluate.py),
  check_git_diff_audit, check_preflight, check_clean_fp_bound,
  check_anomaly. Retest reuses check_metric_rerun semantics and adds a
  new orchestration loop on top.
- Isolation: use `git worktree add .omc/retest-worktree <sha>` so the
  main tree isn't disturbed. Cleanup via `git worktree remove` on exit
  (atexit).
- Staleness gate reuse: lift the features.py-sha comparison block from
  run_autoresearch.sh into a helper in verify_agent.py so both the loop
  and retest use the same auto-retrain trigger.
- Oracle corpus: inherits `OMC_EVAL_DATA_ROOT` from the caller's shell.
  retest subcommand errors out if OMC_EVAL_DATA_ROOT is unset (the
  wrapper normally sets it; operator must source a decryption manually
  if running outside the wrapper, OR retest runs with the wrapper
  stopped but the tmp dir still mounted — add a warning either way).
- Results.tsv: read-only input. Retest outputs go to .omc/retest_report.md
  + research_notes.md + baseline_metrics.json (when recovering).
- Commit semantics: the cherry-pick retains the original `hypothesis:`
  commit message (preserves provenance); the baseline update commits
  as `baseline: combined=<new> after retest-recovery <orig-sha>`.

## Interview Transcript
<details>
<summary>Full Q&A (3 rounds)</summary>

### Round 1 — Scope
**Q:** Which discarded hypotheses should be retested after a wrapper/eval bug fix?
**A:** All discards since bug was introduced.
**Ambiguity:** 33% → next target: Success Criteria.

### Round 2 — Recovery action
**Q:** When a retested discard now beats the current baseline, what happens?
**A:** Auto cherry-pick + baseline update.
**Ambiguity:** 25% → next target: Constraints (ownership).

### Round 3 — Owner
**Q:** Who/what owns the retest action?
**A:** verify_agent.py --retest <from-sha> mode.
**Ambiguity:** 19% → threshold met.
</details>
