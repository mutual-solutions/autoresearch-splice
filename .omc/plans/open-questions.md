# Open Questions

## Agent Coordination v2 - 2026-04-15
- [ ] chmod 555 reversibility: should we provide a restore script (`chmod 755`) for when data needs updating? — Matters because updating GT data requires temporarily lifting permissions
- [ ] find -type f returns 0 at top level but subdirs show 91 files — may be a macOS indexing quirk, preflight should use explicit per-subdir counting (already planned), but worth verifying on a clean clone
- [ ] Phase 2 trigger threshold: how many concurrent conflict incidents justify promoting lock infrastructure? — Avoids premature or delayed promotion

## Retest Discards (US-514) - 2026-04-18
- [ ] Use `cherry-pick -x` for provenance annotation on recovery commits? — Spec is silent; `-x` is strictly additive but changes commit message shape
- [ ] Retest report header: include summary counters (`N conflicts / M recovered`) or per-row detail only? — Spec silent; defaulting to both for operator scan-ability
- [ ] `atexit` registration vs pytest teardown interaction — confirm context-manager + `atexit.unregister` pattern during Step 8 implementation
- [ ] Re-run `evaluate.py` on main tree after worktree replay for clean strict-improvement audit, or reuse in-worktree output? — Recommendation: reuse, with opt-in `--retest-re-eval` flag for paranoia
- [ ] Should `.omc/retest-worktree` get an explicit `.gitignore` entry, or does the current `.omc/` umbrella cover it? — Verify during Step 5

## Retest Discards (US-514) rev-2 post-Architect - 2026-04-18
- [ ] Sentinel file contents: include `git rev-parse HEAD` at sentinel-write time so operators can tell "clean cherry-pick that failed during baseline commit" apart from "conflicted cherry-pick that failed earlier"? — Phase-1 could ship with just `{sha, started_at}` and extend later; minor
- [ ] `--retest-re-eval` default: off (reuse worktree eval) in phase 1, per "Why chosen" — confirm with operator once a first real-use batch runs
- [ ] Phase-2 trigger condition (3rd consumer of keep path) — who decides and how do we notice? Candidates: scheduled re-keep rehearsal command, cross-project keep auditor. Document in wiki if and when one materializes.
- [ ] Resume hint on corpus-purged abort: the `--from-sha <last-recovered-sha>` message needs to be derivable from on-disk state after a crash — confirm that `baseline_metrics.json["git_sha"]` is always the last recovery sha (never overwritten by the loop while retest is running, since the loop refuses to start while the sentinel exists)
- [ ] `_do_keep_path` / `_do_ensure_classifier_fresh` shell-function extraction: the loop body currently references `$reported`, `$hypothesis_commit`, `$hypothesis_subject` as loop-scoped vars; extraction must pass these as positional args. Verify during Step 1 that no closure-captured loop var is silently read inside the extracted body.

## Retest Discards (US-514) rev-3 post-Architect ITERATE - 2026-04-18
- [ ] `--retest-re-eval` flag (opt-in fresh re-run of `check_metric_rerun` on the main tree after cherry-pick, vs reusing worktree output) — default off in phase 1; worth surfacing to operators, or keep hidden until a real re-eval need shows up?
- [ ] Sentinel contents extension (`HEAD` sha at write time) — phase-1 ship as-is (`{sha, started_at}`) or include `HEAD` now? See ADR Follow-up #6.
- [ ] `atexit` + pytest session teardown: step-8 tests should use a context-manager variant and call `atexit.unregister` after assertion. Confirm fixture pattern during implementation.
- [ ] Phase-2 trigger condition wording: is "a third consumer of the keep path appears" concrete enough, or should the ADR name specific candidate consumers (rehearsal command, cross-project auditor) as gating examples?
