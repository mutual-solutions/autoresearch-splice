# RALPLAN: Loop Stability (24-hour non-stop operation)

**Status:** DRAFT — awaiting consensus review
**Date:** 2026-04-18
**Branch:** `autoresearch/apr15` @ `484c477`
**Trigger:** ENOSPC crash after ~15 hours; disk 100% full from unbounded feature_cache growth + stale /tmp eval dirs

---

## ADR

**Decision:** Prevent disk-full crashes via bounded growth (absolute GB cap on feature_cache, startup /tmp sweep, preflight df gate) and improve crash observability (ENOSPC trap). Do NOT attempt mid-flight auto-recovery or autonomous space reclamation.

**Drivers:**
1. Repo posture is "surface failures, don't swallow them" — auto-prevention over auto-recovery
2. 228 GB SSD with 26 GB consumed by 8 stale feature_cache shas is the root cause
3. Ungraceful deaths bypass atexit cleanup, leaking ~346 MB per exit in /tmp

**Alternatives considered:**
- (A) External cron/watchdog daemon — rejected: adds operational complexity, violates scope constraint (no external monitoring, no Python daemon)
- (B) Mid-flight space reclamation (detect low disk during iteration, free cache) — rejected: ambiguity about which cache is safe to drop during eval; violates auto-prevention philosophy
- (C) Bounded growth + preflight gates (CHOSEN) — all shell-native, additive, no control flow changes

**Why chosen:** Option C is the lightest-weight path that eliminates the root cause. All changes are additive to `run_autoresearch.sh`. No iteration-loop control flow changes. Rollback is trivial (revert the commit).

**Consequences:** Feature cache limited to ~2 shas of headroom. If features.py churn rate exceeds iteration rate, cold cache misses increase. Acceptable: cache rebuild is ~3 min per domain, vs 15-hour crash recovery.

**Follow-ups:** Monitor cache hit rate over next 48 hours. If cold misses spike, consider raising cap from 8 GB to 12 GB.

---

## Design Decisions (5 open questions resolved)

### Q1: feature_cache cap value → 8 GB

Justification: Each sha dir is ~3.3 GB. 8 GB allows the current sha (~3.3 GB) plus one prior sha (~3.3 GB) with 1.4 GB headroom for partial writes. Two shas covers the common case: current features.py + the one being replaced during a features hypothesis. Three shas would require 10+ GB and cuts into SSD headroom on a 228 GB drive. 8 GB is 3.5% of total disk — conservative.

### Q2: preflight disk threshold → 5 GB

Justification: Worst-case single iteration disk cost = feature_cache growth (3.3 GB, new sha) + eval decrypt (~350 MB) + retrain output (~200 MB) + log growth (~50 MB) = ~3.9 GB. 5 GB provides ~1.1 GB headroom above this. 2 GB is insufficient — a features hypothesis that triggers a new cache sha would immediately breach. 5 GB is 2.2% of total disk.

### Q3: ERR trap feasibility → use write-guarded pattern, not global ERR trap

Bash's ERR trap has three problems under tmux: (a) `set -e` already active means ERR trap fires on any non-zero, not just ENOSPC; (b) subshell isolation means traps set in the main shell don't propagate into `$(...)` captures; (c) tmux session exit on parent shell death races the trap handler. The reliable pattern is: guard the write-sensitive sites (`_log` calls, redirect-to-file, `echo >> $file`) with explicit ENOSPC detection. Specifically: add a `_disk_check` function called at iteration boundaries (start of each iteration + after eval + after retrain) that checks `df` and emits CRITICAL + touches STOP_FILE if below 1 GB. This catches the gradual decline that precedes ENOSPC without relying on ERR trap timing.

### Q4: orphan-handling default → proceed-as-current (option b)

Justification: Auto-discarding (`git reset --hard HEAD~1`) risks losing a hypothesis that was actually an improvement — the crash happened during eval, so the hypothesis was never scored. Proceeding treats HEAD as the current state and lets the next iteration build on it or replace it naturally. If the orphan was bad, the next eval will score it and discard. If good, it survives. This matches the existing `_detect_deep_orphans` warn-only posture. The current auto-reset at lines 528-532 is too aggressive — replace it with the proceed-as-current logic.

**Change from current behavior:** Lines 528-532 currently `_guarded_reset HEAD~1` on any orphan. The new logic will instead: (a) log the orphan as WARN, (b) check if the orphan's detector.py/features.py differ from HEAD~1, (c) if they differ (real hypothesis), skip reset and proceed — the iteration loop will naturally evaluate or replace it; (d) if identical (commit was metadata-only), reset as before since nothing is lost.

### Q5: testing strategy → layered: unit shell tests + integration smoke + manual 24h validation

- **Unit (shell):** Test `_cleanup_stale_tmp_eval` by creating fake `/tmp/.ar-eval-test-*` dirs with old timestamps, running the function, asserting they're gone. Test `_check_disk_space` by mocking `df` output via function override. Test `_prune_feature_cache_gb` by creating dirs with known sizes and asserting eviction order.
- **Integration (smoke):** Add a `_selftest` subcommand to `run_autoresearch.sh` that runs the three unit-level checks above in-process. Exits 0 on pass, 1 on fail. No external test framework needed — pure bash.
- **Manual (24h):** Operator starts loop, monitors via `status`/`dashboard` for 24 hours. Success = no `loop.crash` events in JSONL log, feature_cache stays under 8 GB, no stale /tmp dirs accumulate.
- **ENOSPC simulation is out of scope.** Real ENOSPC requires a loopback filesystem or `fallocate` tricks that are fragile on macOS APFS. The `_check_disk_space` gate is the defense — test the gate, trust the gate.

---

## File-touched table

| File | Function/Section | Change | Est. delta lines |
|------|-----------------|--------|-----------------|
| `run_autoresearch.sh` | new `_cleanup_stale_tmp_eval()` | Sweep `/tmp/.ar-eval-*` dirs older than 2h | +15 |
| `run_autoresearch.sh` | new `_check_disk_space()` | `df`-based gate, emit CRITICAL + STOP_FILE if < 5 GB | +12 |
| `run_autoresearch.sh` | new `_prune_feature_cache_gb()` | Replace US-504 prune block with LRU + absolute 8 GB cap | +30 |
| `run_autoresearch.sh` | `run_loop()` lines 520-532 | Replace auto-reset with proceed-as-current orphan logic | +10 / -5 |
| `run_autoresearch.sh` | `run_loop()` after line 534 | Call `_cleanup_stale_tmp_eval`, `_check_disk_space` | +4 |
| `run_autoresearch.sh` | iteration loop (line ~595) | Call `_check_disk_space` at iteration start | +2 |
| `run_autoresearch.sh` | after eval phase (line ~1045) | Call `_check_disk_space` after eval completes | +2 |
| `run_autoresearch.sh` | US-504 block (lines 564-579) | Replace with call to `_prune_feature_cache_gb` | +1 / -16 |
| `run_autoresearch.sh` | new `_selftest` case block | Smoke tests for the 3 new functions | +40 |
| `run_autoresearch.sh` | case block (bottom) | Add `selftest)` entry | +3 |

**Total estimated delta:** +120 / -21 = net +99 lines in `run_autoresearch.sh` (single file).

---

## Task flow (5 steps)

### Step 1: feature_cache LRU prune with absolute GB cap

**What:** Replace the US-504 "keep only current sha" prune block (lines 564-579) with `_prune_feature_cache_gb()` that enforces an 8 GB absolute cap using LRU eviction by directory mtime.

**Implementation detail:**
- New function `_prune_feature_cache_gb()` near `_eval_cleanup` (line ~36):
  - Reads `FEATURE_CACHE_CAP_MB=8192` (constant at top of file, easy to tune).
  - Lists all subdirs of `.omc/feature_cache/`, sorted by mtime ascending (oldest first).
  - Sums sizes via `du -sm`. While total > cap, delete oldest dir (log each deletion as `wrapper.feature_cache.prune.lru`).
  - Never deletes the dir matching current `$OMC_FEATURES_PY_SHA` (active cache).
- Call site 1: replace lines 564-579 with `_prune_feature_cache_gb`.
- Call site 2: after each keep-path completion (line ~1161), call `_prune_feature_cache_gb` to bound growth mid-session.
- Keep event name `feature_cache.prune` for audit continuity; add `.lru` suffix for new evictions.

**Acceptance criteria:**
- [ ] `.omc/feature_cache/` total size never exceeds 8 GB (verified by `du -sm` after 20+ iterations with features.py churn).
- [ ] Current sha's cache dir is never deleted.
- [ ] `feature_cache.prune.lru` events appear in JSONL log when eviction occurs.
- [ ] `_selftest` case exercises this function with mock dirs.

**Rollback:** Revert the function + restore the original 4-line prune block.

### Step 2: startup sweep of stale /tmp eval dirs

**What:** Add `_cleanup_stale_tmp_eval()` that removes `/tmp/.ar-eval-*` directories older than 2 hours. Called once at loop start, before eval decrypt.

**Implementation detail:**
- New function near `_eval_cleanup` (line ~36):
  - `find /tmp -maxdepth 1 -name '.ar-eval-*' -type d -mmin +120` (macOS `find` supports `-mmin`).
  - For each match: `rm -rf "$d"`, log `wrapper.tmp_eval.sweep path="$d"`.
  - Log count of swept dirs: `wrapper.tmp_eval.sweep_summary count=N freed_mb=M`.
- Call site: `run_loop()` after orphan detection (line ~534), before eval decrypt (line ~546).

**Acceptance criteria:**
- [ ] After simulated ungraceful exit (create 3 dirs `/tmp/.ar-eval-test-{1,2,3}` with `touch -t` 3 hours ago), next `./run_autoresearch.sh _loop` sweeps all 3.
- [ ] Dirs younger than 2 hours are NOT swept.
- [ ] `wrapper.tmp_eval.sweep_summary` event in JSONL log shows count + freed_mb.
- [ ] `_selftest` case exercises this.

**Rollback:** Remove the function and its call site.

### Step 3: preflight df gate

**What:** Add `_check_disk_space()` that refuses to proceed if available disk space < 5 GB. Called at loop start, at each iteration start, and after eval completes.

**Implementation detail:**
- New function:
  - `avail_kb=$(df -k "$PROJECT_DIR" | awk 'NR==2 {print $4}')` — portable across macOS/Linux.
  - Threshold: `DISK_MIN_FREE_KB=$((5 * 1024 * 1024))` (5 GB, constant at top of file).
  - If below threshold at loop start: `_log CRITICAL wrapper preflight.disk_critical avail_kb=$avail_kb threshold_kb=$DISK_MIN_FREE_KB`, print operator message to stderr, `exit 1`.
  - If below threshold mid-iteration: `_log CRITICAL wrapper disk.low avail_kb=$avail_kb`, `touch "$STOP_FILE"` (graceful stop after current phase completes — do NOT `exit 1` mid-iteration to avoid orphaning state).
- Call sites:
  1. `run_loop()` after `_cleanup_stale_tmp_eval` / before eval decrypt — hard exit.
  2. Iteration loop start (line ~595) — soft stop via STOP_FILE.
  3. After eval phase completes (line ~1045) — soft stop via STOP_FILE.

**Acceptance criteria:**
- [ ] With < 5 GB free (simulated via lowered threshold in selftest), `_loop` refuses to start with CRITICAL log + exit 1.
- [ ] Mid-iteration low-disk emits CRITICAL + sets STOP_FILE (loop exits after current iteration, no orphan).
- [ ] Normal operation (>5 GB) proceeds without interference.
- [ ] `_selftest` exercises the gate with a mock `df` override.

**Rollback:** Remove the function and its 3 call sites.

### Step 4: improved orphan handling (proceed-as-current)

**What:** Replace the auto-reset orphan logic (lines 528-532) with smarter proceed-as-current behavior.

**Implementation detail:**
- Replace lines 527-532 with:
  ```
  orphan_subject=$(git log -1 --format=%s 2>/dev/null)
  if echo "$orphan_subject" | grep -q "^hypothesis:"; then
      orphan_sha=$(git log -1 --format=%h)
      # Check if hypothesis changed detector.py or features.py vs parent
      if git diff --quiet HEAD~1 HEAD -- splice/detector.py splice/features.py 2>/dev/null; then
          # Metadata-only orphan — safe to reset
          _log WARN wrapper orphan.hypothesis sha="$orphan_sha" \
              subject="$orphan_subject" action="reset (metadata-only)"
          _guarded_reset HEAD~1
      else
          # Real hypothesis with code changes — proceed as current
          _log WARN wrapper orphan.hypothesis sha="$orphan_sha" \
              subject="$orphan_subject" action="proceed (code changes present)"
      fi
  fi
  ```
- No sentinel file mechanism needed — the decision is deterministic based on diff content.

**Acceptance criteria:**
- [ ] Orphan with detector.py changes: loop starts without reset, orphan commit preserved in history, next iteration builds on it.
- [ ] Orphan with only metadata changes: loop resets as before.
- [ ] Both cases emit WARN-level log with `action=` field distinguishing the path.
- [ ] No regression in `_detect_deep_orphans` behavior (it remains warn-only).

**Rollback:** Restore the original 5-line auto-reset block.

### Step 5: selftest subcommand + integration smoke

**What:** Add `./run_autoresearch.sh selftest` that validates Steps 1-3 in-process.

**Implementation detail:**
- New `selftest)` case in the bottom `case` block:
  - **Test A (cache prune):** Create 4 fake dirs under a temp `.omc/feature_cache_test/` with known sizes (2 GB simulated via sparse files or just size assertions). Call `_prune_feature_cache_gb` with a 6 GB cap. Assert oldest dir removed, newest 2 retained.
  - **Test B (tmp sweep):** Create `/tmp/.ar-eval-selftest-{old,new}`. `touch -t` the old one 3 hours back. Call `_cleanup_stale_tmp_eval`. Assert old removed, new retained.
  - **Test C (disk gate):** Override `df` with a function that returns fake output. Call `_check_disk_space` with threshold. Assert exit code / STOP_FILE behavior.
  - Print PASS/FAIL per test. Exit 0 if all pass, 1 otherwise.
  - Clean up all temp artifacts on exit.

**Acceptance criteria:**
- [ ] `./run_autoresearch.sh selftest` exits 0 on a healthy system.
- [ ] Each sub-test has PASS/FAIL output.
- [ ] No side effects on real `.omc/feature_cache/` or loop state.
- [ ] Selftest runs in < 5 seconds.

**Rollback:** Remove the case block (~40 lines).

---

## Commit sequence

**Single commit.** All 5 steps touch only `run_autoresearch.sh` and are tightly coupled (disk-safety is one coherent concern). Splitting would create intermediate states where some guards exist but not others — unhelpful for bisect since the guards are independent but the selftest validates all of them.

```
feat(wrapper): disk-safety guards for 24h loop stability (US-518)

- feature_cache LRU prune with 8 GB absolute cap (replaces US-504 current-sha-only prune)
- startup sweep of stale /tmp/.ar-eval-* dirs older than 2h
- preflight df gate (5 GB threshold): hard-exit at startup, soft-stop mid-iteration
- smarter orphan handling: proceed-as-current for code-change hypotheses
- selftest subcommand for smoke-testing all guards
```

**Gate-green invariant:** The commit is additive to `run_autoresearch.sh`. Existing iteration loop control flow is unchanged. `./run_autoresearch.sh selftest` must pass before the commit is accepted. `./run_autoresearch.sh status` must still work. No protected files touched.

---

## Loop-safety invariants preserved

1. **_guarded_reset is the only reset path** — no new `git reset --hard` calls added.
2. **STOP_FILE is the only graceful-stop mechanism** — disk gate uses it mid-iteration, does not `exit 1` during iteration.
3. **EXIT/HUP/INT/TERM trap unchanged** — `_eval_cleanup` still fires on all catchable signals.
4. **Retest sentinel guard unchanged** — `_refuse_if_retest_sentinel` still gates `start`, `_loop`, `_loop_restart`.
5. **Guard stash/pop unchanged** — `_GUARD_PATHS` and `_guard_stash_push`/`_guard_stash_pop` not modified.
6. **No new Python subcommands** — all logic is shell-native.
7. **No protected files touched** — `evaluate.py`, `program.md`, eval data, manifest, preflight.py untouched.

---

## Success criteria (mapped to items)

| # | Criterion | Validates |
|---|-----------|-----------|
| 1 | `.omc/feature_cache/` never exceeds 8 GB across 50+ iterations | Step 1 |
| 2 | After simulated ungraceful exit, next start sweeps stale /tmp dirs | Step 2 |
| 3 | Loop refuses to start with < 5 GB free (CRITICAL log + exit 1) | Step 3 |
| 4 | Mid-iteration low-disk sets STOP_FILE (graceful exit, no orphan) | Step 3 |
| 5 | Orphan with code changes preserved; metadata-only orphan reset | Step 4 |
| 6 | `./run_autoresearch.sh selftest` exits 0 | Step 5 |
| 7 | 24-hour continuous run with no `loop.crash` events | All |
| 8 | Any disk-related failure emits CRITICAL event before wrapper exits | Steps 3-4 |

---

## Guardrails

**Must have:**
- All changes in `run_autoresearch.sh` only
- Shell-native (no new Python files)
- Constants (`FEATURE_CACHE_CAP_MB`, `DISK_MIN_FREE_KB`) at top of file for easy tuning
- Backward-compatible: `start`, `stop`, `status`, `dashboard`, `rollback` unchanged
- Event names maintain audit continuity with existing JSONL schema

**Must NOT have:**
- Changes to iteration loop control flow (`while true` / `continue` / `break` paths)
- Changes to `_guarded_reset`, `_guard_stash_push/pop`, or `_do_keep_path`
- New Python files or subcommands
- External monitoring, cron, or daemon
- Changes to protected files
- Auto-recovery logic (mid-flight space reclamation, auto-restart on ENOSPC)
