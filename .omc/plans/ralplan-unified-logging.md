# RALPLAN — US-515: Unified logging (phase-split; Python-side now, bash-side deferred)

**Plan ID:** `ralplan-unified-logging`
**Source spec:** `/Users/yejunjang/Projects/mutual/autoresearch-splice/.omc/specs/deep-interview-unified-logging.md` (ambiguity 18%, PASSED)
**Date:** 2026-04-18
**Mode:** consensus (SHORT) — Rev-4, single-PR two-commit landing per Critic ask 7
**Status:** draft; autoresearch loop STOPPED during phase 1 migration

---

## Prior Revision Asks (condensed)

**Architect (Rev-1→2):** SURFACE-split accepted; RESULTS_TSV exempt; renderer uses `mkdir(parents=True, exist_ok=True)` + `open('a')`; `_write_retest_report` carve-out; pre-merge smoke test gate; `shap_report.py` 6 sites included.

**Critic Rev-3 (C1–C8):** C1 smoke stub Python-only (Step 6); C2 clean-iteration definition + 14-day cap (phase-2 gate); C3 evaluate.py allowlist (Step 3); C4 `sys.path.insert('.omc/coordination')` import; C5 `OMC_LOGGER_DISABLED`; C6 `fcntl.flock` rotation; C7 subsystem rule (Rev-4 mechanizes); C8 `shim_usage` gate (Rev-4 drops — see R5).

## Rev-4 Critic Asks (resolved in THIS revision)

| # | Critic request | Resolution |
|---|---|---|
| R1 | `.omc` leading-dot breaks `python -m omc.*`; use direct file invocation | Step 6 + acceptance (p) rewritten to `uv run python .omc/coordination/tests/test_smoke_iteration.py`; any `__init__.py` / `omc.coordination` package claim struck |
| R2 | `_block_*` is a features convention, NOT detector; use mechanical `_diag(level, component, reason, ...)` 2nd-arg rule | Step 2 rule simplified: `subsystem = f"detector.{component}"` and `f"features.{component}"` — no AST walk |
| R3 | `.omc/classifier/train_classifier.py` imports `_diag` + 4 call sites | Added to Files Touched and Step 2; subsystem=`classifier.train`; grep acceptance includes this file |
| R4 | Only `phase_stats.py` consumes `.omc/autoresearch.log`; 5 other scripts read data artifacts | Step 4 scoped to `phase_stats.py` + `dashboard.py`; non-log readers dropped from phase-1 reader migration |
| R5 | `shim_usage==0` phase-2 gate is tautological (wrapper still emits in phase 1) | Dropped from phase-2 gate; phase-2 criteria trimmed to 4 items |
| R6 | evaluate.py count contradiction | Files Touched now reads `+30/−0` (dual-emit, no print deletion) |
| R7 | Commit 1 `validate_logs.py --audit` failure state undefined | Chosen: single-PR two-commit branch (infra + migration); never merged solo |
| R8 | Allowlist line-range precision 251-258 excludes 254-255 | Corrected to `251-253, 256-258` |
| R9 | `OMC_LOGGER_DISABLED` × smoke test + `_debug_log_rotate_if_needed` disposition | Smoke fixture must NOT set `OMC_LOGGER_DISABLED=1`; both `_debug_log_write` and `_debug_log_rotate_if_needed` retired in Step 2 |

---

## Context

Log emission is ad-hoc along four orthogonal axes: schema, writer, destination, taxonomy. All collapse to one writer + one schema + one file + one event taxonomy — sequenced by SURFACE (Python phase 1; bash phase 2). Phase 1 ships Python-side logger + dual-format reader shim while the wrapper keeps legacy `echo … >> $LOG_FILE`. Phase 2 migrates the wrapper and drops the shim. Autoresearch loop stopped during phase 1 migration; restarts on mixed output that the shim parses.

---

## RALPLAN-DR Summary

### Principles (5)

1. **One writer per surface in phase 1; one writer overall in phase 2.** Every Python emission goes through `.omc/coordination/logger.py`. Bash remains on legacy in phase 1. Named carve-outs: `evaluate.py`'s `RESULTS_TSV:` line (authoritative wrapper contract, phase 3a) and `verify_agent.py`'s `_write_retest_report` (operator artifact, phase 3b).
2. **Oracle redaction is a property of the logger, not callers.** `_ORACLE_DENYLIST_RE` lives in `logger.emit()`. `retest.diagnose.*` auto-sets `oracle_sensitive=True`.
3. **Claude sees Markdown, never JSONL.** Canonical log is JSONL. Claude-visible renderer appends Markdown blocks to `.omc/research_notes.md`. US-503 prompt contract preserved.
4. **Readers break in lockstep with their writer.** Phase 1 dual-format shim keeps log-consuming readers producing non-empty output. Data-artifact consumers (results.tsv, research_notes.md, SHAP JSON, detector.py source) are unaffected.
5. **Audit is a CI gate, not a convention.** `validate_logs.py --audit` tiered — phase 1 Python-only, phase 2 adds bash rules. `--parse` round-trips every JSONL line.

### Decision Drivers (top 3)

1. **Phase-split reduces blast radius** — one-PR retrofit stacks Python + bash + reader + file-deletion + evaluate.py-exemption risk.
2. **Dual-format compat shim is cheap and correct** — ~30 lines in `log_reader.py`; scheduled removal in phase 2.
3. **Protected-file exemption must be bounded** — phase 1 covers `print()` only; `RESULTS_TSV:` line stays; one commit, human-authored, flagged in commit message.

### Viable Options

| Option | Verdict |
|---|---|
| **A (chosen): phase-split by SURFACE — Python phase 1 now, bash phase 2 deferred** | Self-contained phase 1; shim prevents reader breakage; exemption bounded. |
| B: original single-PR full retrofit | **Invalidated** — stacks too many failure modes. |
| C: AXIS-split (4 PRs) | **Invalidated** — dual-writer at every boundary. |
| D: structlog + keep bash forever | **Invalidated** — drops bash axis permanently. |

---

## Phase 1 — Files Touched

| File | Action | Δ lines | Notes |
|---|---|---|---|
| `.omc/coordination/logger.py` | **NEW** | +200 | Logger, `get_logger`, `emit`, rotation (fcntl.flock), redaction, taxonomy, renderer, `OMC_LOGGER_DISABLED`. |
| `scripts/log_reader.py` | **NEW** | +130 | `iter_events()` dual-format shim. |
| `scripts/validate_logs.py` | **NEW** | +90 | `--audit` (Python tier), `--parse`. |
| `.omc/coordination/tests/test_smoke_iteration.py` | **NEW** | +120 | Python-only smoke stub; direct file invocation. |
| `.omc/coordination/tests/test_logger.py` | **NEW** | +150 | Schema/redaction/renderer/rotation/disabled-env tests. |
| `.omc/coordination/tests/test_log_readers.py` | **NEW** | +80 | `test_dual_format_merge`. |
| `.omc/coordination/tests/fixtures/mixed_log_sample/` | **NEW** | fixtures | `autoresearch.jsonl` + `autoresearch.log` + expected reader outputs. |
| `detector.py` | edit | +5 / −25 | 21 `_diag` sites → `_log.emit(...)`; remove `_debug_log_write` (`_debug_log_rotate_if_needed` at line 637 also retired — R9). |
| `features.py` | edit | +3 / −5 | 5 `_diag` sites; delete `from detector import _diag`. |
| `ml_eval.py` | edit | +2 / −2 | 1 `_diag` site; delete import. |
| `.omc/classifier/shap_report.py` | edit | +3 / −6 | 6 `_diag` sites. |
| `.omc/classifier/train_classifier.py` | edit | +3 / −5 | **R3:** `_diag` import at L38 + 4 call sites (L134, L142, L225, L231) migrate; subsystem=`classifier.train`. |
| `evaluate.py` | **maintainer edit (exempt)** | **+30 / −0** | **R6:** dual-emit additions only; no print deletion. Allowlist in Step 3. |
| `scripts/dashboard.py` | edit | +15 / −20 | Adds `iter_events()` via shim to supplement results.tsv read; existing data-artifact reads unchanged (R4). |
| `scripts/phase_stats.py` | edit | +20 / −40 | **R4 primary reader:** `iter_events(subsystem="wrapper", event="iteration.phase")` replaces `.omc/autoresearch.log` grep. |
| `.omc/coordination/verify_agent.py` | edit | +10 / −15 | Logger for internal progress; `_write_retest_report` retained; imports `_ORACLE_DENYLIST_RE` from logger. |
| `.gitignore` | edit | +2 / −1 | Add `.omc/logs/`, `.omc/logs/child-stderr.log`. |
| `CLAUDE.md` | edit | +15 | Logger API, exemption record, audit gate, phase-2 gate criteria, carve-outs. |

**Removed from phase 1 (R4):** `scripts/diagnose_repeat_rate.py` (reads results.tsv + research_notes.md), `scripts/notebook_digest.py` (reads research_notes.md), `scripts/shap_shift.py` (reads reports/<sha>/*.json), `scripts/tunable_frontier.py` (reads detector.py source + results.tsv). These are data-artifact consumers, not log consumers; they do not need the compat shim. Their migration is out of phase-1 scope unless log-consumption features are added (not required by spec).

**NOT touched in phase 1:** `run_autoresearch.sh`, `.omc/autoresearch.log`, `.omc/research_notes.md`, `.omc/last_reflection.md`, `.omc/last_eval.log`, `.omc/tunable_frontier.txt`, `.omc/shap_rollup.json`, `.omc/retest-report.md`.

---

## Phase 1 — Ordered Implementation Steps

### Step 1 — Logger + dual-format reader + validators (infrastructure commit)

**`.omc/coordination/logger.py`** (~200 lines). Module-docstring pins:

- **Import convention (C4, refined per R1):** callers use `import sys; sys.path.insert(0, '.omc/coordination'); from logger import get_logger`. The `.omc` prefix (leading dot) prevents packaging as `omc.coordination` under `python -m`; all invocation must be direct file paths.
- **`OMC_LOGGER_DISABLED=1` env var (C5):** when set, `emit()` returns immediately (no-op). Smoke-test fixture (Step 6) must NOT set this; fixtures rely on real emission to probe redaction/parse (R9).

Key class sketch:

```python
SCHEMA_VERSION = 1
LOG_PATH = Path(".omc/logs/autoresearch.jsonl")
MAX_SIZE_BYTES = 50 * 1024 * 1024
MAX_ROLLS = 10
_ORACLE_DENYLIST_RE = re.compile(r"\b(combined(?:_[a-z]+)?|splice_f1|clean_score)\s*=\s*[0-9.]+")

class Logger:
    def emit(self, level, event, *, oracle_sensitive=False, claude_visible=False, **kv):
        if os.environ.get("OMC_LOGGER_DISABLED") == "1": return
        if event.startswith("retest.diagnose."): oracle_sensitive = True
        if oracle_sensitive:
            kv = {k: (_ORACLE_DENYLIST_RE.sub(r"\1=<REDACTED>", v) if isinstance(v, str) else v)
                  for k, v in kv.items()}
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                if LOG_PATH.stat().st_size > MAX_SIZE_BYTES: self._rotate_locked()
                rec = {"schema_version": SCHEMA_VERSION,
                       "ts": datetime.now(timezone.utc).isoformat(),
                       "level": level, "subsystem": self.subsystem,
                       "event": event, "claude_visible": claude_visible, **kv}
                f.write(json.dumps(rec) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def render_claude_visible_events(*, since_ts=None, target=Path(".omc/research_notes.md")):
    """File-creation: target.parent.mkdir(parents=True, exist_ok=True); open(target, 'a')."""
```

**Event taxonomy (module docstring):**
- `eval.run`, `eval.crash`, `eval.dataset.result`, `eval.dataset.error`
- `classifier.retrain.trigger|complete|failed`
- `retest.candidate.begin|end`, `retest.recovery.committed`, `retest.diagnose.traceback`, `retest.diagnose.per_domain_error` (force `oracle_sensitive`)
- `diag.detector.<component>.*`, `diag.features.features.*`, `diag.ml.eval.*`, `diag.shap.report.*`, `diag.classifier.train.*`
- `pipeline.failure`
- Reserved for phase 2: `wrapper.*`, `note.appended`, `tunable.frontier.snapshot`

**`scripts/log_reader.py`** (~130 lines) — dual-format shim:

```python
def iter_events(*, subsystem=None, event=None, level=None, since=None,
                jsonl_path=Path(".omc/logs/autoresearch.jsonl"),
                legacy_path=Path(".omc/autoresearch.log")):
    """Phase 1: merge-sort JSONL + legacy shell-prefix events.
    Phase 2 deletes legacy_path handling — SCHEDULED REMOVAL MARKER."""
```

**`scripts/validate_logs.py`** (~90 lines):
- `--audit` (phase 1 tier): grep `_diag(` in detector/features/ml_eval/shap_report/train_classifier (must be 0); bare `print(` in non-CLI Python per Step-3 allowlist. **Does NOT gate `echo … >> $LOG_FILE` in phase 1.** Audit is no-op against commit 1 by design; enforcement activates with commit 2 (see Migration Ordering for single-PR two-commit landing, R7).
- `--parse <path>`: `json.loads` each line; required keys `schema_version`, `ts`, `level`, `subsystem`, `event`; warn on unknown events.

**Acceptance:** `test_logger.py::{test_emit_schema, test_redaction_forced_on_retest_diagnose, test_renderer_creates_missing_file, test_rotation, test_logger_disabled_env}` all pass.

### Step 2 — Python caller migration (single atomic commit)

**Subsystem mapping rule (C7, rewritten per R2):** `detector._diag`'s signature is `_diag(level, component, reason, **ctx)`. The 2nd positional arg IS the component (observed values: `gbm`, `slider`, `pairwise`, `hotelling`, `refine`, `zscore`, `robust_zscore`). Migration is **mechanical**:

- `detector.py` call `_diag("INFO", "gbm", "dedupe", ...)` → `get_logger("detector.gbm").emit("INFO", "diag.gbm.dedupe", ...)`. No AST walk; no `_block_*` heuristic.
- `features.py` uses component=`features` throughout → `subsystem = "features.features"` (flat).
- `ml_eval.py`: subsystem=`ml.eval`.
- `.omc/classifier/shap_report.py`: subsystem=`shap.report`.
- `.omc/classifier/train_classifier.py` (R3): subsystem=`classifier.train`; imports `_diag` from detector.py at line 38 and calls at lines 134, 142, 225, 231. Must migrate in the same commit as detector.py or next retrain breaks.

**Migrations:**
1. `detector.py` — delete inline `_diag()` + `_debug_log_write` + `_debug_log_rotate_if_needed` (line 637) (R9); add:
   ```python
   import sys; sys.path.insert(0, '.omc/coordination')
   from logger import get_logger
   ```
   21 call sites rewritten using mechanical rule above.
2. `features.py` — 5 sites; delete `from detector import _diag`.
3. `ml_eval.py` — 1 site; delete import.
4. `.omc/classifier/shap_report.py` — 6 sites.
5. **`.omc/classifier/train_classifier.py` (R3) — 4 sites (L134, L142, L225, L231); delete `from detector import _diag` at L38.** Per CLAUDE.md, this file runs after every features.py edit, so migration cannot lag.

**Ordering:** detector first (others import `_diag` from it); single commit covers all five atomically.

**Acceptance:** `grep -c '_diag(' detector.py features.py ml_eval.py .omc/classifier/shap_report.py .omc/classifier/train_classifier.py` == 0 (R3). `validate_logs.py --audit` Python tier exits 0. Manual smoke: `uv run python .omc/classifier/train_classifier.py` succeeds post-commit.

### Step 3 — `evaluate.py` one-time maintainer migration (bounded)

**Human operator authors this commit.** Commit message body contains `PROTECTED-FILE EXEMPTION: evaluate.py modified under one-time US-515 phase-1 unified-logging migration exemption. Scope: ~30 dual-emit logger.emit() companions. RESULTS_TSV: line NOT touched. No print() deleted.`

#### evaluate.py print() allowlist (C3, R8)

Grep enumerated 65 `print()` sites. All STAY as CLI output; ~30 gain DUAL-EMIT companion `logger.emit()` calls; `RESULTS_TSV:` untouched. **No prints are deleted** (R6).

| Line(s) | Category | Disposition |
|---|---|---|
| 32, 128, 330, 591 | stderr error for missing input | STAY + DUAL-EMIT `eval.input.error` |
| 148-149, 156, 169, 231, 235, 342-343, 351, 358, 372, 391, 395 | per-file progress (SKIP/HIT/MISS/ERROR/separator/header) | STAY (pure CLI decoration, no dual-emit) |
| **251-253, 256-258**, 261 | `evaluate()` final metric summary (R8: lines 254-255 are non-print) | STAY + DUAL-EMIT `eval.metrics.splice` / `eval.metrics.clean` |
| 276, 280-281, 285, 290-292, 301 | FP distribution + crossfade + loc_accuracy | STAY + DUAL-EMIT `eval.fp.distribution`, `eval.crossfade.breakdown`, `eval.loc_accuracy` |
| 404-408, 451-457 | `evaluate_opus_32k()` metrics + aggregate | STAY + DUAL-EMIT `eval.opus32k.metrics` / `eval.opus32k.aggregate` |
| 488-495 | `evaluate_cross_dataset()` aggregate | STAY + DUAL-EMIT `eval.aggregate` |
| 558 | single-dataset combined line | STAY + DUAL-EMIT `eval.single.combined` |
| 584, 598 | Touch ID prompt + decrypt status | STAY (interactive UX, no dual-emit) |
| 612, 614, 626, 627 | cross-dataset headers + MISSING + `combined_{ds.id}: ERROR` + end marker | STAY; line 626 DUAL-EMIT `eval.dataset.error` JSONL (string is authoritative wrapper contract at `verify_agent.py:325`; retain verbatim) |
| 632, 634-636 | aggregate header + `combined_{id}` / `combined_mean` / `combined_min` | STAY + DUAL-EMIT `eval.dataset.result` + `eval.aggregate` |
| 654 | `RESULTS_TSV: ...` | **STAY verbatim; no dual-emit** (authoritative wrapper contract; phase 3a migrates) |
| 657, 659 | final `combined: X.Y` + `elapsed: Xs` | STAY + DUAL-EMIT `eval.run.complete` |

**Rule:** metric/error lines get dual-emit (string for human+wrapper, JSONL for future consumers); separators/headers/interactive-UX stay pure-string; `RESULTS_TSV:` is the singular carve-out.

**Net effect:** ~30 new `logger.emit()` companion calls; **0 prints deleted (R6)**; `RESULTS_TSV:` verbatim.

**Acceptance:** `grep -n 'print(' evaluate.py` output matches the allowlist line-by-line with the R8 split. Commit message contains PROTECTED-FILE EXEMPTION marker. `test_wrapper_parse_paths.py::test_results_tsv_parse_unchanged` passes.

### Step 4 — Log-reader migration (scoped per R4)

**In scope (log consumers):**
- **`phase_stats.py`** — only reader that parses `.omc/autoresearch.log`; switches to `iter_events(subsystem="wrapper", event="iteration.phase")` via the dual-format shim. Shim synthesizes `wrapper.iteration.phase` events from legacy `PHASE:` lines.
- **`dashboard.py`** — primary consumer of `results.tsv` + `baseline_metrics.json` + `shap_rollup.json` + `versions.json` + `git log`; NOT a log consumer, but benefits from optional `iter_events()` augmentation for recent-iteration overlay. Existing data-artifact reads remain unchanged.

**Out of scope (data-artifact consumers, NOT log consumers — per R4 verification):**
- `diagnose_repeat_rate.py` (reads results.tsv + research_notes.md)
- `notebook_digest.py` (reads research_notes.md)
- `shap_shift.py` (reads reports/<sha>/*.json)
- `tunable_frontier.py` (reads detector.py source + results.tsv)

These 4 scripts do not need the compat shim; they consume data artifacts that are not being migrated in phase 1. Spec does not require adding log-consumption features to them.

**Fixture:** `.omc/coordination/tests/fixtures/mixed_log_sample/` contains both JSONL + legacy log + `expected_phase_stats.txt` + `expected_dashboard.txt`. Post-migration output via shim must be byte-identical. Test: `test_log_readers.py::test_dual_format_merge`.

### Step 5 — Carve-outs + gitignore + CLAUDE.md

**Named phase-1 carve-outs:** `RESULTS_TSV:` line, `_write_retest_report`, bash wrapper emitters, `.omc/research_notes.md` (wrapper-owned in phase 1), `.omc/last_reflection.md`.

**`.gitignore`:** add `.omc/logs/`, `.omc/logs/child-stderr.log`.

**CLAUDE.md additions:**
- Logger API sketch + import convention (direct file path, NOT `python -m omc.*` — R1) + `OMC_LOGGER_DISABLED` docs
- evaluate.py phase-1 exemption record (commit SHA, date)
- Audit gate command + tiering note
- Phase-2 gate criteria (see gate section below, revised per R5)
- Carve-outs list
- Policy: `logger.emit()` for Python; `>> $LOG_FILE` permitted only in `run_autoresearch.sh` until phase 2

### Step 6 — Pre-merge smoke test gate (C1 — Python-only stub; R1 invocation fix)

**Mandatory before phase-1 PR merges.** Python-only stub — does NOT invoke the wrapper or run `evaluate.py` on real data.

**New file:** `.omc/coordination/tests/test_smoke_iteration.py` (~120 lines). First line: `import sys; sys.path.insert(0, '.omc/coordination')` so `from logger import ...` and `from log_reader import ...` resolve. Simulates one iteration's worth of `logger.emit` calls against a temp fixture:

1. **Emit fixture events** into `tmp/autoresearch.jsonl`: prompt-build events (wrapper-shim-synthesized from paired legacy-line fixture), `eval.run` + `eval.dataset.result x3` + `eval.aggregate`, `classifier.retrain.trigger|complete`, `retest.candidate.begin|end`, `retest.diagnose.traceback` with a combined=0.47 in the detail (redaction probe), keep/discard wrapper events via legacy-line fixture, `note.appended` placeholder. **Fixture must NOT set `OMC_LOGGER_DISABLED=1`** (R9) — redaction + parse gates require real emission.
2. **Parse gate:** `subprocess.run(["uv", "run", "python", "scripts/validate_logs.py", "--parse", tmp_log])` must exit 0.
3. **Reader gate:** run `dashboard.py` and `phase_stats.py` against the fixture (readers respect `OMC_LOG_OVERRIDE` env var for test-time log path); assert each stdout is non-empty. Other readers are out of phase-1 scope (R4).
4. **Redaction gate:** re-parse fixture; assert `retest.diagnose.traceback` record's `detail` field contains `<REDACTED>`, not `0.47`.

**Runnable gate command (PR description, R1 fix):**

```bash
uv run python .omc/coordination/tests/test_smoke_iteration.py
```

Direct file invocation is mandatory. `.omc` has a leading dot, which prevents Python from importing it as package `omc.coordination` under `-m`; any `__init__.py` / `python -m omc.coordination.tests...` scheme from Rev-3 is struck. The smoke test file's first line inserts `.omc/coordination` onto `sys.path` so sibling imports resolve. Wrapper is NOT invoked. No `OMC_EVAL_DATA_ROOT` or fixture audio required. Runs in <5 s. PR is not mergeable if any of the four gates fails.

---

## Migration Ordering (phase 1) — Single-PR Two-Commit Landing (R7)

**Landing strategy: ONE pull request, TWO commits on the same branch.** Commit 1 is never merged to `master` solo; both commits land together. This gives clear git history (infra vs caller migration) without exposing a broken intermediate state on the main branch.

1. **Commit 1 (first on branch): `infra(US-515): unified logger + dual-format reader + validators + smoke stub`.**
   New files: logger.py, log_reader.py, validate_logs.py, test fixtures, test_logger.py, test_log_readers.py, test_smoke_iteration.py. No caller migration. `validate_logs.py --audit` is a no-op (design: audit reports current state, does not fail CI at this commit). Audit enforcement kicks in via commit 2's test suite.

2. **Commit 2 (second on branch): `migrate(US-515): Python callers → unified logger (phase 1)`.**
   Atomic: detector/features/ml_eval/shap_report/train_classifier (R3) + evaluate.py (maintainer-exempt) + phase_stats.py + dashboard.py + verify_agent.py redaction consolidation + .gitignore + CLAUDE.md. Post-commit: `validate_logs.py --audit` exits 0; smoke stub passes; loop restart produces mixed JSONL + legacy logs; log-consuming readers parse both via shim.

**Branch gate:** PR cannot merge until HEAD (commit 2) passes smoke stub and all acceptance criteria. Commit 1 alone would fail CI post-merge (no callers yet migrated) — this is why it never merges solo.

**Rollback:** `git revert` commit 2 restores emitters; commit 1 infra stays inert and harmless (no callers reference it).

---

## Acceptance Criteria (phase 1)

- [ ] (a) `Logger("test").emit("INFO", "diag.test.smoke", k=1)` writes one JSONL record with required keys. `test_emit_schema`.
- [ ] (b) `iter_events()` on mixed fixture yields events from both sources in timestamp-merged order. `test_dual_format_merge`.
- [ ] (c) Event taxonomy registry matches emitted events. `validate_logs.py --parse` warns on unknowns.
- [ ] (d) Oracle redaction: `retest.diagnose.traceback` with `detail="combined=0.47 crashed"` produces `<REDACTED>`. `test_redaction_forced_on_retest_diagnose`.
- [ ] (e) Renderer: `rm -f tmp/notes.md && render_claude_visible_events(target=tmp/"notes.md")` creates file via `parent.mkdir` + `open('a')`. `test_renderer_creates_missing_file`.
- [ ] (f) **(R3)** `grep -c '_diag(' detector.py features.py ml_eval.py .omc/classifier/shap_report.py .omc/classifier/train_classifier.py` == 0; `validate_logs.py --audit` Python tier exits 0.
- [ ] (g) `evaluate.py` exempt commit has PROTECTED-FILE EXEMPTION marker. `grep -n 'print(' evaluate.py` matches the Step-3 allowlist line-by-line (R8 split honored). **Zero prints deleted (R6).**
- [ ] (h) `RESULTS_TSV:` line unchanged. `test_wrapper_parse_paths.py::test_results_tsv_parse_unchanged` passes.
- [ ] (i) **(R4)** `phase_stats.py` and `dashboard.py` produce non-empty output on mixed fixture.
- [ ] (j) Carve-outs documented in CLAUDE.md with phase-2/3 follow-up links.
- [ ] (k) `validate_logs.py --audit` Python tier exits 0 post-commit-2; does NOT gate bash.
- [ ] (l) `validate_logs.py --parse .omc/logs/autoresearch.jsonl` exits 0 on fresh iteration output.
- [ ] (m) Rotation: synthetic 50MB+1 test produces rolled file; 10-roll prune. `test_rotation`.
- [ ] (n) US-514 retest: `verify_agent.py --retest --dry-run` produces `.omc/retest-report.md` + `retest.candidate.*` JSONL.
- [ ] (o) Stopped-loop restart produces mixed JSONL + legacy output; phase_stats/dashboard output qualitatively equivalent to pre-migration.
- [ ] (p) **(R1)** Smoke stub gate: `uv run python .omc/coordination/tests/test_smoke_iteration.py` exits 0. **Direct file invocation; NOT `python -m`.**
- [ ] (q) `OMC_LOGGER_DISABLED=1` no-ops `emit()`. `test_logger_disabled_env`. **Smoke fixture does NOT set this env var (R9).**
- [ ] (r) **(R3)** `uv run python .omc/classifier/train_classifier.py` completes without `ImportError` post-commit-2.

---

## Phase 2 — Deferred Bash-Side Migration (separate follow-up PR)

### Phase-2 gate (revised per R5 — 4 criteria, `shim_usage` removed)

**Gate fires when either condition is met, whichever comes first:**

1. **Clean-iterations criterion:** ≥20 clean phase-1 iterations. A **clean iteration** is:
   - No `validate_logs.py --parse` failure on that iteration's JSONL log, AND
   - No reader empty-output regression (phase_stats.py + dashboard.py each produce non-empty output), AND
   - No new logger-surfaced bug filed against `logger.py` or `log_reader.py` in the interval since phase-1 merge.

2. **Calendar backstop (C2):** If 20 clean iterations are NOT reached within **14 days of phase-1 merge**, phase 2 auto-triggers against the current state regardless.

**R5 rationale — `shim_usage==0` criterion DROPPED:** The wrapper's ~65 bash `echo >> $LOG_FILE` sites emit legacy format every iteration throughout phase 1. `log_reader.py`'s shim parses them and would increment any `shim_usage` counter on every iteration. The gate "zero shim_usage in trailing 20 iterations" would therefore only be satisfiable AFTER phase 2 runs, which makes it a tautology (the thing it gates is a prerequisite for satisfying it). The 4 remaining criteria above are sufficient: parse health + reader non-regression + no new logger bug + calendar cap.

Gate status is tracked in `.omc/coordination/phase2_gate.md` (updated per iteration by wrapper post-hook in phase 1 follow-up).

### Scope (drafted; full plan as separate RALPLAN when gate fires)

- Add `_log` shell function to `run_autoresearch.sh` (Python subprocess; latency measured, socket daemon fallback if >5% overhead).
- Retrofit ~65 `>> $LOG_FILE` sites → `_log subsystem event k=v …`.
- Wrapper `_append_note` → `note.appended` JSONL; renderer wires to production `research_notes.md`.
- Delete legacy files: `.omc/autoresearch.log`, `.omc/last_eval.log`, `.omc/last_reflection.md`, `.omc/tunable_frontier.txt`, `.omc/autoresearch-debug.log`.
- Readers drop dual-format shim; `log_reader.py` simplifies.
- `validate_logs.py --audit` adds bash tier: `grep -n 'echo .*>>.*LOG_FILE' run_autoresearch.sh` must return 0.
- `tunable_frontier.py` → `tunable.frontier.snapshot` JSONL event.

**NOT touched in phase 2:** `evaluate.py` `RESULTS_TSV:` (phase 3a), `verify_agent.py` `_write_retest_report` (phase 3b).

---

## ADR

### Decision
Unify logging across all four axes via SURFACE-split phased retrofit. Phase 1 (this PR, single-PR two-commit landing) Python-side + dual-format shim; phase 2 (follow-up) bash migration + shim removal. `RESULTS_TSV:` and `_write_retest_report` are named phase-3 carve-outs.

### Drivers
1. Phase-split reduces blast radius — one-PR retrofit stacks too many failure modes.
2. Log-consuming readers break in lockstep with their writer — dual-format shim keeps phase_stats/dashboard non-empty; data-artifact consumers are unaffected (R4).
3. Bounded exemption on evaluate.py — phase-1 maintainer edit scoped to dual-emit companions; `RESULTS_TSV:` stays; zero prints deleted (R6).

### Alternatives considered
- **B (single-PR full retrofit):** architect-rejected as risk-stacked.
- **C (AXIS-split: 4 PRs):** architect-identified as strictly worse; dual-writer at every boundary.
- **D (structlog + keep bash forever):** drops bash axis.

### Why chosen
SURFACE-split sequences all four axes (phase 1 proves schema; phase 2 extends to wrapper). Shim is transient (~30 lines) with scheduled removal and a 2-criterion phase-2 trigger (clean iterations OR 14-day calendar cap; tautological `shim_usage` gate removed per R5). Carve-outs are deliberate. Smoke stub is runnable via direct file invocation (R1). Allowlist partitions all 65 evaluate.py prints with line-range precision (C3 + R8). `train_classifier.py` co-migrates with detector.py to prevent retrain breakage (R3). Subsystem mapping is mechanical from `_diag`'s existing 2nd arg (R2) — no AST walk.

### Consequences
- **Positive:** phase 1 reviewable in isolation; rollback bounded to Python-side; renderer file-creation unit-tested; `shap_report.py` + `train_classifier.py` migrations included; `OMC_LOGGER_DISABLED` gives tests a clean no-op knob (smoke fixture intentionally does NOT set it, R9); `fcntl.flock` insures rotation against subprocess concurrency; mechanical subsystem rule removes migration ambiguity (R2); both `_debug_log_write` and `_debug_log_rotate_if_needed` retired in commit 2 (R9).
- **Negative:** two commits on one branch instead of one commit (R7 mitigation); legacy emitters survive phase 1 (mitigated by shim + audit tiering); compat shim is transient (mitigated by phase-2 gate); phase-1 JSONL lacks wrapper.* events natively (phase_stats reads them via shim).

### Follow-ups
1. **Phase 2 — bash wrapper migration.** Gate: ≥20 clean phase-1 iterations (parse-OK + no reader regression + no new logger bug) OR 14-day calendar backstop, whichever comes first (R5). Full plan as separate RALPLAN.
2. **Phase 3a — `RESULTS_TSV:` migration.** Migrate string-line format → `eval.results.tsv` JSONL event; update wrapper's 5 parser sites. Second evaluate.py maintainer-exempt commit.
3. **Phase 3b — `_write_retest_report` migration.** Migrate to `render_claude_visible_events(target=.omc/retest-report.md, filter=retest.*)`.
4. **Phase 3c — diagnose parser migration.** `verify_agent.py:325` regex → `eval.dataset.error` JSONL consumption. Bundled with 3a.
5. **Phase 3d — data-artifact reader enrichment (deferred).** `diagnose_repeat_rate.py`, `notebook_digest.py`, `shap_shift.py`, `tunable_frontier.py` gain optional `iter_events()` augmentation IF spec later requires log-derived features in those tools. Out of phase-1 scope per R4.
6. **Shell `_log` latency fallback.** If phase-2 `uv run python` per-call overhead >5% for 10 consecutive iterations, swap to socket daemon or named pipe.
7. **Child-process stderr structuring.** Phase 2 deliverable: `.omc/logs/child-stderr.log` sidecar.
8. **SQLite index over JSONL.** Deferred; `iter_events()` is fine at ~50k records/month.

---

## Open Questions

None blocking phase 1. Rev-3 resolutions preserved; Rev-4 resolutions above.

Items intentionally deferred (not blocking): per-component subsystem granularity (resolved by Step 2 mechanical rule, R2), child-process stderr handling (phase 2), data-artifact reader enrichment (phase 3d).

---

## Plan Summary

**Plan saved to:** `.omc/plans/ralplan-unified-logging.md`
**Scope (phase 1):** ~11 edited + 7 new files, ~+540 / −135 lines. Single-PR two-commit landing on stopped loop. Complexity MEDIUM.

**Key Deliverables:** (1) `logger.py` + `log_reader.py` (dual-format shim) + `validate_logs.py`; (2) Python `_diag`/print migration in detector/features/ml_eval/shap_report/**train_classifier** (R3) + evaluate.py dual-emit (R6); (3) renderer with file-creation semantics (unit-tested); (4) 2 log-consuming readers (phase_stats + dashboard) on `iter_events()` via shim (R4); (5) runnable Python-only smoke stub via **direct file invocation** (R1); (6) 2-criterion phase-2 gate — clean iterations OR 14-day calendar cap (R5); (7) evaluate.py print() allowlist with **line-range-precise** partition (R8); (8) mechanical subsystem-mapping rule from `_diag`'s 2nd arg (R2); (9) named carve-outs + `_debug_log_write` + `_debug_log_rotate_if_needed` retired (R9).

**Does this plan capture the Rev-4 Critic ITERATE verdict?**
- "proceed" — Begin phase-1 Commit 1 (infra) on feature branch.
- "adjust [X]" — Return to planner.
- "restart" — Discard.
