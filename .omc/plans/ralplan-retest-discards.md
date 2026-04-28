# RALPLAN — US-514: `verify_agent.py --retest <from-sha>` — systematic replay of discards

**Plan ID:** `ralplan-retest-discards`
**Source spec:** `.omc/specs/deep-interview-retest-discards.md` (ambiguity 19%, PASSED)
**Date:** 2026-04-18 (rev 3 — post-Architect ITERATE, phase-1 scope)
**Mode:** consensus (SHORT)
**Status:** revised; phase 1 only; Python helper lift deferred to phase 2 on explicit trigger

---

## Architect Asks — Resolutions

| # | Architect ask | Resolution in this revision |
|---|---|---|
| 1 | Drop the Python helper lift (`keep_path.py`, `classifier_staleness.py`); do not touch the 70-line hot keep path in `run_autoresearch.sh`. | **Accepted.** Prior rev-2 steps 1–2 deleted. Reuse is achieved via two new internal wrapper subcommands appended additively at the bottom of `run_autoresearch.sh` — `_keep_path <sha> <combined> [<retest-origin-sha>]` and `_ensure_classifier_fresh`. Net wrapper delta: +~30 additive lines at EOF, zero deletes from hot-path logic. See step 1. |
| 2 | Add SIGKILL/OOM safety via a `.omc/retest-in-progress` sentinel file. | **Accepted.** Written before the main-tree `git cherry-pick`; deleted only after the baseline commit + `baseline_metrics.json` refresh land. `run_autoresearch.sh`'s `start` / `_loop` / `_loop_restart` refuse to start when the sentinel exists, with an error pointing at `.omc/retest-report.md` and suggesting `rm -f .omc/retest-in-progress` after manual inspection. See steps 4, 6, 7 and the Sentinel callout. |
| 3 | Fix dual-baseline race: no in-memory `current_baseline`; disk `baseline_metrics.json` is the only source of truth. | **Accepted.** `_current_baseline()` re-reads `.omc/coordination/baseline_metrics.json["combined"]` fresh before every candidate comparison and before every `_try_recover` guard. Nothing cached across iterations. See step 5 and the Disk-sourced baseline callout. |
| 4 | Add corpus-staleness periodic check (`stat $OMC_EVAL_DATA_ROOT`); retest never re-decrypts; operator owns the eval corpus lifecycle. | **Accepted.** Checked once at startup (`_require_eval_env`) and again between every candidate (`_check_corpus_alive`). On miss: clean batch abort, cutoff sha recorded in the report, remediation line names `--from-sha <last-recovered-sha>`. Retest contains zero `eval_crypto.py` calls. See step 4 and the Corpus Staleness callout. |
| 5 | Timestamp-tie tiebreaker: `%ct` has 1s resolution; secondary sort = `results.tsv` append order. | **Accepted.** `_load_retest_candidates` captures 0-based `row_index` for every TSV row. Sort key is `(commit_time, row_index)` ascending; Python's stable sort makes it deterministic. See step 3. |
| 6 | Split into phases. Phase 1 = WARN follow-up + `--retest` subcommand with bash-verb reuse + sentinel + all 5 soundness gaps. Phase 2 = Python helper lift, deferred, gated on a 3rd consumer. | **Accepted.** This plan is phase 1 only. Phase 2 captured in ADR Follow-ups as a deferred item with explicit trigger condition. |

---

## Context

A pipeline bug (most recently: HistGBM `feature_importances_` AttributeError swallowed by `evaluate.py`'s per-domain exception handler, collapsing the aggregate `combined` to the GM floor of 0.01) can silently flag a batch of legitimate hypotheses as discards. Once the bug is fixed, there is no operator-facing way to replay those discards and cherry-pick any whose *real* combined now beats baseline. This plan adds that capability as a new subcommand on `verify_agent.py` (the project's single verification boundary).

Bundled follow-up from the Architect review of US-511: `check_git_diff_audit` silently falls back to working-tree + staged diffs when `OMC_HEAD_BEFORE` is unset. Retest calls this check without the wrapper's env wrapping, so the silent degradation is exactly the scenario the WARN needs to surface. Folded into phase 1.

**Autoresearch loop status constraint:** retest requires a stopped loop (shared eval corpus lifecycle, single writer to `baseline_metrics.json`). The subcommand refuses to start if the `autoresearch` tmux session is alive. Symmetrically, the loop refuses to start if the retest sentinel file is present.

---

## RALPLAN-DR Summary

### Principles (5)

1. **Additive, not replacement.** `--retest` is a new subcommand alongside the existing strict-improvement flow. The hot keep path in `run_autoresearch.sh` is untouched. Reuse happens by calling the wrapper's own existing bash via new internal verbs (`_keep_path`, `_ensure_classifier_fresh`) appended at the bottom of the script.
2. **Isolation before mutation.** Replay happens inside a `git worktree add .omc/retest-worktree <sha>`. The live working tree is not mutated until a recovery is confirmed HIGH/MEDIUM and we re-enter the main worktree to cherry-pick. Lifecycle is cleaned via an `atexit` handler plus a sentinel file so SIGKILL/OOM/crash paths never silently corrupt the main tree.
3. **Oracle integrity inherited, not recreated.** Retest refuses to run unless `OMC_EVAL_DATA_ROOT` is already set and the path still exists (APFS can purge `/tmp/*` mid-run). Retest never calls `scripts/eval_crypto.py decrypt` and never touches `data/eval.tar.gz.enc`. Operator owns the decrypt lifecycle.
4. **Chronological ordering with a disk-sourced bar.** Candidates are processed oldest-first by `(commit_time, results.tsv_row_index)` — the row-index tiebreaker makes ordering deterministic on 1s-resolution `%ct` collisions. After each confirmed recovery, the rolling bar is re-read from `.omc/coordination/baseline_metrics.json` on disk (single source of truth). No in-memory baseline scalar that could desync on crash-between-commit-and-update.
5. **Dry-run by construction.** `--dry-run` bypasses every mutating operation: no worktree, no cherry-pick, no baseline update, no note append, **no sentinel write**. It reads `results.tsv` + `baseline_metrics.json` only and returns in <5 s. Report write is the one I/O side-effect allowed.

### Decision Drivers (top 3)

1. **Operator trust in the recovery path.** A botched retest that cherry-picks a hypothesis over a broken worktree is worse than never having retest. Worktree isolation + atexit cleanup + sentinel file + chronological replay + disk-sourced rolling bar are all in service of this.
2. **Minimum refactor risk to the running loop.** The wrapper's keep path compiles a lot of hard-won safety (guarded stash, orphan-hypothesis reset, amend-safety branches, per-phase timing). Lifting it to Python mid-project buys speculative reuse against real regression risk. Phase 1 chooses additive bash verbs; phase 2 revisits only if a 3rd consumer appears.
3. **Bounded blast radius on conflicts and crashes.** Cherry-pick conflict → skip. Eval crash → skip. Retrain crash → skip. Sentinel present at loop start → loop refuses to start and points at the report. Every failure mode has a named outcome and a named remediation.

### Viable Options (≥2)

| Option | Description | Pros | Cons | Verdict |
|---|---|---|---|---|
| **A (chosen, revised): subcommand + worktree + internal wrapper verbs** | `verify_agent.py --retest <from-sha>`. Worktree isolation. Recovery invokes `./run_autoresearch.sh _keep_path <sha> <combined>` and `_ensure_classifier_fresh` — two new internal subcommands added additively at the bottom of the wrapper. Sentinel file guards SIGKILL. Disk-sourced rolling baseline. | Zero churn on the running keep path; single source of truth for keep semantics remains the wrapper's existing bash; sentinel closes the SIGKILL hole; minimal phase-1 surface. | `subprocess.run(['bash', 'run_autoresearch.sh', '_keep_path', ...])` is one extra hop vs direct Python call; acceptable given scale (≤20 recoveries/batch). | **Chosen.** |
| **B: subcommand + worktree + Python helper lift** (prior rev-1 design) | Lift keep path + staleness gate into `.omc/coordination/keep_path.py` + `classifier_staleness.py`; rewrite the wrapper's keep block to call them. | Ultimately cleaner if 3 consumers exist. | Touches 70 lines of hot-path bash mid-project with only 2 consumers in sight; semantic parity test is expensive to write and brittle; violates "minimum refactor risk" driver. | **Deferred to phase 2.** Re-evaluate if a 3rd consumer of the keep path appears. ADR Follow-up #1. |
| **C: retest re-implements keep semantics inline in Python** | Retest re-implements tag / baseline-json / note logic in Python directly. | Smaller diff on `run_autoresearch.sh`. | Guarantees semantic drift from loop keeps over time. Violates the single-source-of-truth principle. | **Invalidated.** |
| **D: in-place replay (no worktree)** | `git stash` + `git checkout <sha>` + eval + `git checkout -`. | No worktree plumbing. | Crash mid-replay leaves the live tree on an arbitrary SHA; the spec explicitly requires `git worktree add`. | **Invalidated.** |
| **E: separate `retest.py` script** | Top-level script, not a verify_agent subcommand. | Clean separation. | Violates spec owner constraint ("verify_agent.py --retest mode"). Second verification boundary. | **Invalidated.** |

---

## Acceptance Criteria (derived from spec; each testable)

1. `uv run python .omc/coordination/verify_agent.py --retest <from-sha>` exists, parses a short SHA, exits non-zero with a clear message if the SHA is unknown.
2. Subcommand refuses to start (exit 2) if `tmux has-session -t autoresearch` returns 0. Error message names the remedy (`./run_autoresearch.sh stop`).
3. Subcommand refuses to start (exit 2) if `OMC_EVAL_DATA_ROOT` is unset, or if the directory it names is missing (`stat`). Error message names the remedy, including the `--from-sha <current-position>` resume hint on purged-corpus abort.
4. Subcommand refuses to start (exit 2) if `.omc/retest-in-progress` sentinel already exists. Error message points at `.omc/retest-report.md` and suggests `rm -f .omc/retest-in-progress` after manual inspection.
5. Candidate set = all rows in `results.tsv` with `status=discard` AND commit-date strictly after `<from-sha>`'s commit date. Sort key: `(commit_time_ct, row_index)` ascending. Rows whose commit is missing from git (squashed/lost) are surfaced with outcome `missing`.
6. For each candidate: worktree created at `.omc/retest-worktree`, candidate SHA cherry-picked. Cherry-pick conflict → log WARN + paths, `cherry-pick --abort`, outcome `conflict`, continue.
7. Inside the worktree: classifier staleness check via `./run_autoresearch.sh _ensure_classifier_fresh`. Retrain crash → outcome `retrain-crash`, continue.
8. Inside the worktree: run `evaluate.py --shap`, parse `combined` from the last `RESULTS_TSV:` line. Eval crash or parse failure → outcome `eval-crash`, continue.
9. Between each candidate: `stat $OMC_EVAL_DATA_ROOT` sanity check. If missing, clean batch abort with a remediation message; report records the cutoff candidate. Retest never re-decrypts.
10. Recovery path (retest `combined > current_baseline_from_disk` strictly):
    a. Write `.omc/retest-in-progress` sentinel (`{"sha": <cand_sha>, "started_at": <iso8601>}`).
    b. Exit the worktree, main-tree `git cherry-pick -x <cand_sha>`.
    c. Run structural checks (`check_git_diff_audit`, `check_anomaly`, `check_preflight`, `check_clean_fp_bound`) as direct function calls, reusing the worktree's eval output.
    d. On LOW → `git reset --hard ORIG_HEAD`, remove sentinel, outcome `verify-fail`, continue.
    e. On HIGH/MEDIUM → invoke `./run_autoresearch.sh _keep_path <cand_sha> <combined> --retest-origin <cand_sha>` which performs the wrapper's existing tag + baseline_metrics.json refresh + baseline commit + versions.json append + note append inside the same extracted shell function the loop keep uses.
    f. After the wrapper returns 0 and the baseline commit subject is confirmed (`git log -1 --format=%s` starts with `baseline:`), remove `.omc/retest-in-progress`.
    g. Re-read `.omc/coordination/baseline_metrics.json["combined"]` from disk as the rolling bar for the next candidate.
11. `--dry-run`: skip every mutation including sentinel write. Report has `predicted` column = `original_combined - baseline_at_dryrun_start`, outcome `dry-run-skipped`, zero subprocess calls to `evaluate.py`. <5 s on current state.
12. `--limit N`: process at most `N` candidates in chronological order. Unprocessed candidates named in the report's tail.
13. `.omc/retest-report.md` columns: `sha`, `original_combined`, `retest_combined` (or `NA`), `delta`, `outcome ∈ {recovered, still-lower, conflict, eval-crash, retrain-crash, verify-fail, missing, corpus-purged, dry-run-skipped}`, `note`.
14. `atexit` handler: on any exit path (normal, exception, SIGINT), `git worktree remove --force .omc/retest-worktree` (idempotent). `atexit` alone is not sufficient for SIGKILL — the sentinel file covers that path by blocking the loop until operator inspection.
15. `run_autoresearch.sh start` / `_loop_restart` / `_loop` refuse to start if `.omc/retest-in-progress` exists. Clear error naming `.omc/retest-report.md` and the manual-cleanup remediation.
16. `check_git_diff_audit` returns `("WARN", "OMC_HEAD_BEFORE unset; commit-level audit skipped")` when env var is empty AND working-tree + staged diffs are clean. Existing FAIL paths (protected-file edit) unchanged. `main()`'s existing "any WARN → MEDIUM" rule preserved.
17. Self-test (`--retest-self-test`): synthetic two-row `results.tsv` + stubbed `evaluate.py` exercises cherry-pick path end-to-end in a temp repo. <10 s.

---

## Implementation Steps (ordered, phase 1 scope)

### Step 1 — Add internal wrapper verbs (`_keep_path`, `_ensure_classifier_fresh`) (~+30 bash)

- **Edit:** `run_autoresearch.sh`. Current case dispatcher lives at lines 942–1027 (`case "${1:-help}" in ... esac`). Add two new case branches just before `*)` at line 1023.
- **Mechanical extraction first, zero logic change:** wrap the existing inline keep block (the ~70 lines that perform tag + `baseline_metrics.json` refresh + baseline commit + `versions.json` append + note append inside the `_loop` keep branch) in a new shell function `_do_keep_path()` defined once near the top of the script. The loop's keep branch is changed from an inline block to a single call `_do_keep_path "$@"` — no other edits. Acceptance: byte-identical `git log -1 --format=%B` between a loop-keep and a retest-keep of the same fixture (`--retest-origin` absent).
- Same mechanical wrap for the classifier staleness gate (currently the features.py sha vs classifier meta sha block — see grep around line 694). Wrap in `_do_ensure_classifier_fresh()`.
- New case branches:
  - `_keep_path) shift; _do_keep_path "$@" ;;` — positional args `<hypothesis_sha> <reported_combined> [--retest-origin <orig_sha>]`. When `--retest-origin` is present, `_do_keep_path` routes through a thin branch that appends the origin to the `note:` payload and to the `baseline:` commit subject for provenance. When absent, commit messages are byte-identical to today's loop keep.
  - `_ensure_classifier_fresh) _do_ensure_classifier_fresh ;;`
- No Python helpers created. No `keep_path.py`, no `classifier_staleness.py`.
- Net wrapper delta: +~30 additive lines at the dispatcher plus two mechanical `function() { ... }` wraps (zero body edits, move-only). Hot keep path is not touched.

### Step 2 — `check_git_diff_audit` WARN on missing `OMC_HEAD_BEFORE` (~+8 / −1)

- **Edit:** `.omc/coordination/verify_agent.py::check_git_diff_audit`. Return `("WARN", "OMC_HEAD_BEFORE unset; commit-level audit skipped")` when env empty AND `unstaged + staged` clean. Protected-file FAIL paths unchanged.
- `main()`'s existing WARN→MEDIUM mapping is preserved; no logic change there.
- *Acceptance:* unit test asserts WARN on empty env + clean tree; PASS on env-set + clean committed diff; FAIL on protected-file edit regardless of env.

### Step 3 — Retest candidate enumeration with deterministic tiebreaker (~+60)

- **Edit:** `.omc/coordination/verify_agent.py`.
- `@dataclass RetestCandidate`: `sha: str`, `original_combined: float | None`, `commit_time: int | None`, `row_index: int`, `description: str`, plus mutable outcome fields.
- `def _load_retest_candidates(results_tsv: Path, from_sha: str) -> list[RetestCandidate]`: walks the TSV capturing `row_index` (0-based append order) for every row. Keeps rows where `status=="discard"` and `git show -s --format=%ct <sha>` is strictly greater than `%ct` of `<from_sha>`. Missing-SHA rows → `outcome="missing"`, still included for the report.
- Sort key: `(commit_time, row_index)` ascending. Python's stable sort makes the tuple tiebreaker sufficient.
- *Acceptance:* unit test with a 5-row fixture (two discards sharing `%ct`, ordered by append) yields the earlier-row-index first regardless of input shuffle.

### Step 4 — Preflight guards + sentinel + corpus-staleness check (~+90)

- **Edit:** `.omc/coordination/verify_agent.py`.
- `def _tmux_session_alive(name: str) -> bool`: `tmux has-session -t <name>` → `returncode == 0`. Missing tmux binary → False.
- `def _require_eval_env() -> None`: fail fast if `OMC_EVAL_DATA_ROOT` unset. Then `stat` the path; if missing, exit with remediation message including `--from-sha <current-position>` resume hint.
- `def _check_corpus_alive() -> None`: re-runs the path-exists check. Called once at startup and once between every candidate. On miss, record the next-candidate sha as the cutoff into the report and abort cleanly (exit 1). Never calls `eval_crypto.py`.
- `_RETEST_SENTINEL = Path(".omc/retest-in-progress")`.
- `def _refuse_if_sentinel() -> None`: called at subcommand entry. Exit 2 with message naming the report + the `rm -f .omc/retest-in-progress` remediation if the sentinel is present.
- *Acceptance:* CLI smoke tests for every guard (tmux alive, env unset, env dir purged, sentinel present). Each emits the right remediation string.

### Step 5 — Worktree lifecycle + per-candidate replay (~+140)

- **Edit:** `.omc/coordination/verify_agent.py`.
- `_RETEST_WORKTREE = Path(".omc/retest-worktree")`.
- `def _worktree_setup() -> None`: `git worktree add --detach .omc/retest-worktree HEAD`. Register `atexit.register(_worktree_cleanup)` exactly once via a module-level sentinel boolean.
- `def _worktree_cleanup() -> None`: best-effort `git -C .omc/retest-worktree cherry-pick --abort`, then `git worktree remove --force .omc/retest-worktree`. Tolerate absence. Does NOT delete `.omc/retest-in-progress` — sentinel deletion is only on the happy path inside `_try_recover` step f.
- `def _replay_one(cand: RetestCandidate) -> None`: inside the worktree, `git cherry-pick -x <sha>`. On conflict → `cherry-pick --abort`, `outcome="conflict"`, return. Invoke `./run_autoresearch.sh _ensure_classifier_fresh` (step 1). Retrain non-zero → `outcome="retrain-crash"`, return. Invoke `evaluate.py --shap`. Parse final `RESULTS_TSV:` for `combined=`. Before returning, `git reset --hard <worktree_base>` so the next candidate cherry-picks onto a clean tree.
- **Disk-sourced rolling bar:** `def _current_baseline() -> float`: reads `.omc/coordination/baseline_metrics.json` and returns `data["combined"]`. Called fresh before every candidate comparison and again inside `_try_recover` step 1. **No in-memory carry.** If the crash-between-commit-and-update window opens, the next `--retest` invocation picks up the current on-disk truth — there is no stale scalar to reconcile.
- *Acceptance:* integration test (fixture repo + stub `evaluate.py`) completes one replay in <2 s and leaves the worktree clean. `atexit` test raises mid-loop and asserts worktree is gone post-exit.

### Step 6 — Recovery path with sentinel bracketing (~+85)

- **Edit:** `.omc/coordination/verify_agent.py`.
- `def _try_recover(cand: RetestCandidate) -> bool`:
  1. `baseline = _current_baseline()` (re-read from disk — no in-memory `current_baseline`).
  2. If `cand.retest_combined is None or cand.retest_combined <= baseline`: outcome `still-lower`, return False.
  3. `_RETEST_SENTINEL.write_text(json.dumps({"sha": cand.sha, "started_at": _utc_iso()}))`.
  4. `git cherry-pick -x <cand.sha>` on the main tree.
  5. Call `check_git_diff_audit`, `check_anomaly`, `check_preflight`, `check_clean_fp_bound` as direct function calls. Reuse the worktree's captured eval output for `check_metric_rerun` unless `--retest-re-eval` was passed.
  6. On LOW confidence: `git reset --hard ORIG_HEAD`, delete sentinel, `outcome="verify-fail"`, return False.
  7. On HIGH/MEDIUM: `subprocess.run(["bash", "run_autoresearch.sh", "_keep_path", cand.sha, str(cand.retest_combined), "--retest-origin", cand.sha], check=True)`. This performs tag + `baseline_metrics.json` refresh + baseline commit + `versions.json` append + note append, all inside the wrapper's existing hot path.
  8. Confirm `git log -1 --format=%s HEAD` matches `^baseline:`. If so, delete `.omc/retest-in-progress`. Outcome `recovered`. Return True.
- *Invariant:* sentinel exists iff the main tree is between main-tree `cherry-pick` and `_keep_path`'s baseline commit. Any crash in that window leaves the sentinel on disk; operator inspection + manual rm is the recovery path; the loop refuses to start in the meantime.
- *Acceptance:* integration test with 2-row fixture (one still-lower, one recovered) produces exactly one `detector-v<N>` tag, one `baseline:` commit, one `note:` commit; sentinel absent at end.

### Step 7 — CLI wiring, report writer, loop-side sentinel guard (~+90)

- **Edit:** `.omc/coordination/verify_agent.py::main`.
- Short-circuit on `--retest` (same pattern as `--diagnose`): parse `--retest <sha>`, `--dry-run`, `--limit <N>`, `--retest-re-eval`, `--retest-self-test`.
- `def run_retest(from_sha, dry_run, limit, re_eval) -> int`: orchestrates guards → enumerate → (optional worktree setup) → per-candidate loop with inter-candidate `_check_corpus_alive` → report write → return 0 on no FAIL outcomes or on dry-run; 1 otherwise.
- `def _write_retest_report(path, candidates, from_sha, dry_run) -> None`: Markdown with header (invocation, from-sha, timestamp, dry-run flag) and a per-candidate table matching AC #13. Footer: `N recovered / M still-lower / K conflicts / J eval-crash / I retrain-crash / H verify-fail / G missing / F corpus-purged / E dry-run-skipped` summary line.
- **Edit:** `run_autoresearch.sh` — add sentinel guard at the top of `start)`, `_loop)`, and `_loop_restart)` branches (~+3 lines each, shared via a helper `_refuse_if_retest_sentinel`). If `[ -f "$PROJECT_DIR/.omc/retest-in-progress" ]`, echo remediation (pointer to `.omc/retest-report.md`) and exit 1.
- *Acceptance:* CLI smoke tests for each guard (tmux alive, env unset, env dir purged, sentinel present, unknown from-sha). `--dry-run` on live state <5 s. Loop-start refusal covered by a shell test that `touch`es the sentinel then runs `./run_autoresearch.sh start`.

### Step 8 — Tests, self-test harness, docs (~+130 / CLAUDE.md +20)

- **New:** `.omc/coordination/tests/test_retest.py` covering:
  - `test_candidate_enumeration_ordering_with_ct_tiebreaker` (two discards sharing `%ct`; append-order tiebreak)
  - `test_dry_run_no_mutation` (no new tags, no baseline write, no sentinel file, no worktree, zero `evaluate.py` subprocess calls)
  - `test_worktree_atexit_cleanup_on_exception`
  - `test_cherry_pick_conflict_skips_without_abort_leftover`
  - `test_recovery_reads_baseline_from_disk_each_iteration` (two recovered candidates; second sees disk-updated baseline, not a cached scalar)
  - `test_sentinel_blocks_loop_start` (pytest subprocess test: `touch .omc/retest-in-progress` then `./run_autoresearch.sh start` → exit 1 with expected stderr)
  - `test_sentinel_survives_sigkill_window` (kill between cherry-pick and `_keep_path` → assert sentinel remains)
  - `test_corpus_purged_mid_batch_reports_cutoff`
  - `test_keep_path_verb_byte_identical_commit_message` (loop-keep fixture vs retest-keep fixture without `--retest-origin` → same `git log -1 --format=%B`)
  - `test_check_git_diff_audit_warn_on_missing_env`
- **New:** `--retest-self-test` wires the fixture scenarios into a `uv run` one-liner.
- **Edit:** `CLAUDE.md` — append a "## Retest" section: operator workflow (stop loop → source decrypt env → `uv run python .omc/coordination/verify_agent.py --retest <sha> [--limit N] [--dry-run]` → read `.omc/retest-report.md`), plus a "## Retest sentinel" note documenting the manual-cleanup path if the sentinel is ever found on a live tree. Explicitly state: retest does NOT re-decrypt; operator owns `OMC_EVAL_DATA_ROOT` lifecycle.
- **Edit:** `.gitignore` — add `.omc/retest-in-progress` and `.omc/retest-worktree/`.

---

## Callouts

### Internal wrapper verbs (phase-1 reuse mechanism)
- `_keep_path <sha> <combined> [--retest-origin <orig_sha>]`: calls `_do_keep_path`, which is the wrapper's existing keep-path bash mechanically wrapped in a shell function (body unchanged). When `--retest-origin` is passed, the note payload and baseline commit subject carry the origin for provenance (`baseline: combined=<X> after retest-recovery <orig-sha>`; `note: keep <sha> (recovered from retest of <orig-sha>)`). When absent, commit messages are byte-identical to today's loop keep.
- `_ensure_classifier_fresh`: calls `_do_ensure_classifier_fresh` — the existing features.py-sha-vs-classifier-meta-sha check, mechanically wrapped.
- Subprocess call site: `subprocess.run(["bash", "run_autoresearch.sh", "_keep_path", sha, combined, "--retest-origin", sha], check=True, cwd=project_dir)`. `CalledProcessError` bubbles up; the recovery flow catches and converts to `outcome="verify-fail"` with stderr captured in the report.

### Sentinel file (SIGKILL/OOM safety)
- Path: `.omc/retest-in-progress`. Contents: `{"sha": "<cand_sha>", "started_at": "<iso8601>"}`.
- Written just before main-tree `git cherry-pick -x`. Deleted only after `_keep_path` returns 0 AND `git log -1 --format=%s` confirms the `baseline:` commit landed.
- `run_autoresearch.sh` refuses `start` / `_loop_restart` / `_loop` while the file is present. Error message:
  ```
  ERROR: .omc/retest-in-progress sentinel found — a retest crashed mid-recovery.
  Inspect: .omc/retest-report.md and `git log --oneline -10` on main.
  Recover: verify the HEAD commit (cherry-pick may be partial), then `rm -f .omc/retest-in-progress`.
  ```
- `atexit` cleanup is best-effort and does NOT delete the sentinel on abnormal exits — that would defeat the SIGKILL hole. Sentinel is deleted only on the one happy path inside `_try_recover`.

### Disk-sourced rolling baseline (race fix)
- `_current_baseline()` is called fresh inside the candidate loop right before every comparison, and again inside `_try_recover` step 1. Never cached across iterations.
- If a recovery commit lands but a crash prevents the function from re-reading, the next invocation of `--retest` picks up the current on-disk truth — there is no stale in-memory scalar to reconcile.
- The `--from-sha <current-position>` resume hint (emitted on corpus-purged abort) uses the last successfully-recovered sha, recoverable from the most recent `baseline:` commit's `git_sha` field in `baseline_metrics.json` plus the `results.tsv` row after it.

### Corpus-staleness periodic check
- APFS tmp reaper can purge `/tmp/*` mid-run. A 4-hour retest batch is very much in range.
- `_check_corpus_alive()`: `Path(os.environ["OMC_EVAL_DATA_ROOT"]).exists()`. Called once at startup and once between every candidate.
- On miss: record the next-candidate sha as the batch cutoff into the report, emit a single stderr line:
  ```
  ERROR: OMC_EVAL_DATA_ROOT ($path) no longer exists (APFS purge likely).
  Re-decrypt and resume with: --retest <sha> --from-sha <last-recovered-sha>
  ```
- Retest does NOT auto re-decrypt. Eval corpus lifecycle is owned by the operator. Plan explicitly states this in step 4 and AC #9.

### Timestamp-tie tiebreaker
- Sort key: `(git show -s --format=%ct <sha>, row_index_in_results_tsv)`. Both ascending. Python's stable sort makes this deterministic.
- `row_index` captures append order (which mirrors observation order because `log_to_results_tsv` appends synchronously after each verified keep/discard).
- Unit test in step 8 fixes two discards to the same `%ct` and asserts the earlier-row-index candidate is processed first regardless of list-input shuffle.

### Cherry-pick conflict handling
- Worktree cherry-pick conflict → capture `git status --porcelain`; `cherry-pick --abort`; outcome `conflict`. Tree restored; batch continues.
- Main-tree cherry-pick verify-fail → `git reset --hard ORIG_HEAD` (set by cherry-pick itself). Never `git reset --hard HEAD~1` — unsafe if unrelated commits landed.
- Main-tree cherry-pick uses `-x` for explicit `(cherry picked from commit <sha>)` provenance on recovery commits.

### Dry-run semantics
- No `git worktree add`, no `git cherry-pick`, no `uv run python evaluate.py`, no baseline write, no note, no tag, **no sentinel write**.
- Allowed side-effect: write `.omc/retest-report.md`. `predicted` column = `original_combined - baseline_at_dryrun_start`.
- Unit test pins zero subprocess calls to `evaluate.py` and zero writes under `.git/` and no creation of `.omc/retest-in-progress`.

### `OMC_HEAD_BEFORE` WARN follow-up
- Included in phase 1 because retest calls `check_git_diff_audit` without the wrapper's env wrapping.
- `main()`'s WARN→MEDIUM rule preserved. Retest's `_try_recover` treats MEDIUM as acceptable (spec AC: "HIGH/MEDIUM → recover") so the WARN surfaces degraded-audit mode without blocking legitimate recoveries.

---

## ADR

**Decision:** Implement US-514 as `verify_agent.py --retest <from-sha>` with git-worktree isolation, sentinel-file SIGKILL safety, disk-sourced rolling baseline, row-index tiebreaker on `%ct` collisions, periodic corpus-staleness checks, and a bundled `check_git_diff_audit` WARN-on-missing-env follow-up. Keep-path and staleness-gate reuse is achieved by adding two internal subcommand verbs to `run_autoresearch.sh` (`_keep_path`, `_ensure_classifier_fresh`) — zero churn on the hot keep-path logic. Python helper lift is deferred to phase 2 on an explicit trigger condition.

**Drivers:**
1. Operator trust in the recovery path (worktree isolation + atexit + sentinel file + chronological rolling baseline re-read from disk).
2. Minimum refactor risk to the running loop (additive bash verbs; no Python extraction of hot-path logic at this stage).
3. Bounded blast radius on single-candidate failures and SIGKILL (outcome taxonomy + sentinel-backed loop guard).

**Alternatives considered:**
- Option B (rev-1 Python helper lift) — **deferred to phase 2** under an explicit trigger (3rd consumer of the keep path appears).
- Option C (retest re-implements keep semantics in Python) — invalidated: guarantees semantic drift.
- Option D (in-place replay, no worktree) — invalidated by spec.
- Option E (separate `retest.py`) — invalidated by spec ownership constraint.

**Why chosen:** Revised Option A is the minimum phase-1 surface that satisfies every soundness gap the Architect flagged (SIGKILL, dual-baseline race, corpus staleness, `%ct`-tie ordering) while reducing refactor risk to zero on the wrapper's hot keep path. Reuse fidelity is preserved because `_keep_path` invokes the exact same bash the loop already runs (mechanical shell-function wrap, body unchanged, byte-identical commit messages under normal use).

**Consequences:**
- `verify_agent.py` grows from ~510 to ~900 lines; still single-file, still the single verification boundary.
- `run_autoresearch.sh` grows by ~30 additive lines at the dispatcher (two new case branches + the `_refuse_if_retest_sentinel` helper wired into `start`/`_loop`/`_loop_restart`) plus two mechanical shell-function wraps of existing hot-path blocks. No body edits to the keep-path or staleness-gate logic.
- `.omc/retest-worktree/` and `.omc/retest-in-progress` appear during retest runs; `.gitignore` explicit entries added in phase 1.
- `check_git_diff_audit` becomes noisier for non-wrapper callers — intentional; WARN degrades `main()`'s verdict to MEDIUM, which retest treats as acceptable.
- No new Python module directories are created in phase 1. Phase 1's Python surface remains inside `.omc/coordination/verify_agent.py` and its tests.

**Follow-ups:**
1. **Phase 2 — Python helper lift (deferred).** Trigger condition: a third consumer of the keep path materializes (current known consumers are the loop keep and retest recovery; a plausible third is a scheduled "re-keep latest N" rehearsal command or a cross-project keep auditor). At that point, lift the `_do_keep_path` shell-function body into `.omc/coordination/keep_path.py` and the staleness gate into `classifier_staleness.py`; have the wrapper and `verify_agent.py` both call them. Phase 2 must include a byte-identical commit-message parity test against commit messages produced by the phase-1 bash path.
2. Parallelize via multiple worktrees (`.omc/retest-worktree-<N>`) if operators run batches >20. Out of scope for phase 1.
3. `git gc` post-batch if repeated worktree add/remove bloats `.git/`. Benchmark after first real-use batch.
4. Extend self-test fixture to cover corpus-purge-mid-batch once we have a real-use run to validate against.
5. If classifier-meta schema evolves, update `_ensure_classifier_fresh`'s sha-compare path in the same PR. Phase 2's Python lift would re-home this logic.
6. Enrich sentinel contents with `git rev-parse HEAD` at write time for clearer post-crash diagnosis ("clean cherry-pick that failed during baseline commit" vs "conflicted cherry-pick that failed earlier"). Low priority; phase 1 ships with `{sha, started_at}`.

---

## Open Questions (for the Critic pass)

- [ ] `--retest-re-eval` flag (opt-in fresh re-run of `check_metric_rerun` on the main tree after cherry-pick, vs reusing worktree output) — default off. Worth surfacing to operators? Spec is silent.
- [ ] Sentinel contents extension (`HEAD` sha at write time) — phase-1 ship-as-is or include now? See ADR Follow-up #6.
- [ ] `atexit` + `pytest` session teardown: step-8 tests should use a context-manager variant and call `atexit.unregister` after assertion. Confirm fixture pattern during implementation.
- [ ] Phase-2 trigger condition wording: is "a third consumer of the keep path appears" concrete enough, or should the ADR name specific candidate consumers (rehearsal command, cross-project auditor) as gating examples?
