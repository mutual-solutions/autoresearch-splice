# US-517: `supervisor_agent.py --maintain` Subcommand

**Status:** PLAN (awaiting human confirmation)
**Branch:** `autoresearch/apr15`
**HEAD at plan time:** `a7673f6`

---

## Revision log

- **rev-3 (current):** +static-grep invocation gate (arch #1), +no-traceback crash fallback (arch #2), +sentinel-timing documentation (arch #3). +4th crash category (`retrain_crash` + `unclassifiable`); +crash-counter state file `.omc/supervisor-crash-counter.txt`.
- **rev-2:** +rename `verify_agent.py` -> `supervisor_agent.py`, +argparse restructure (`--verify`/`--diagnose`/`--maintain`/`--retest` mutually exclusive group). Commit split changed from 2 to 2 (but commit 1 scope expanded: rename + argparse + update all callers). New acceptance criteria for zero stale references and explicit `--verify` flag parsing.
- **rev-1:** initial draft with 5 open-question resolutions.

---

## RALPLAN-DR

### Principles
1. **No autonomy creep** -- v1 triages and drafts specs only; no LLM, no code-gen beyond template fill.
2. **Loop safety first** -- the maintainer must never corrupt the loop's git state, kill tmux, or touch protected files.
3. **Fail-open for unknowns** -- unexpected errors in `--maintain` itself must not halt a productive loop.
4. **Minimal surface** -- reuse existing patterns (argparse mutually exclusive group, JSONL log reader, backlog markdown format).
5. **Operator override** -- a single sentinel file disables the feature without code changes.

### Decision Drivers
1. **Reliability** -- the loop runs unattended for hours; a buggy maintainer halting it spuriously is worse than no maintainer.
2. **Simplicity** -- stdlib only (no rapidfuzz dependency), clean argparse structure replacing the ad-hoc `sys.argv` parsing.
3. **Auditability** -- every action (backlog update, spec draft, halt signal) emits a JSONL event and is git-committed.

### Viable Options (commit strategy)

**Option A: Single commit (rename + argparse + maintain + wrapper + tests).** Atomic rollback, simple review. Risk: very large diff (rename touches 15+ files, argparse restructure touches the main script, new feature adds ~300 lines).

**Option B: Two commits (rename+argparse restructure, then maintain subcommand+wrapper hooks).** Commit 1 is purely mechanical (rename + argparse + caller updates). Commit 2 is the new feature. Allows testing that the rename is clean before the new feature lands. Better for review and `git bisect`.

**Decision: Option B (two commits).** Justification: the rename cascade touches ~15 files and the argparse restructure is a breaking CLI change. Verifying that existing tests pass after commit 1 (before adding new code) is essential. If commit 1 breaks something, commit 2's feature code is not entangled.

### ADR

- **Decision:** Rename `verify_agent.py` to `supervisor_agent.py`; restructure CLI from ad-hoc `sys.argv` parsing to `argparse.add_mutually_exclusive_group(required=True)` with explicit `--verify`/`--diagnose`/`--maintain`/`--retest` flags; add `--maintain` subcommand.
- **Drivers:** The file's role has grown beyond verification (diagnose, retest, now maintain). "supervisor" captures the supervisory role. The ad-hoc argv parsing was growing unsustainable with 4 subcommands; argparse enforces mutual exclusion and generates help text.
- **Alternatives considered:** (A) Keep the name `verify_agent.py` -- rejected because the file now has 4 distinct subcommands, only one of which is "verify." (B) Split into separate scripts (`verify.py`, `diagnose.py`, `maintain.py`, `retest.py`) -- rejected because they share `REPO_ROOT`, `PROTECTED_FILES`, log reader utilities, and the retest code imports verify helpers. (C) Keep manual `sys.argv` parsing for backward compat -- rejected because we own all callers (wrapper + CLAUDE.md examples) and can update atomically.
- **Why chosen:** Single rename + argparse restructure in commit 1 is mechanical and testable. All callers are in-tree (no external consumers). The mutually exclusive group prevents invalid flag combinations that the ad-hoc parser silently ignored.
- **Consequences:** Breaking CLI change -- every caller must add explicit `--verify` where the default was previously implicit. `splice/program.md` is PROTECTED and needs a one-time maintainer exemption. The `ralplan-folder-reorg.md` plan has 3 stale `from autoresearch.verify_agent import` references that become incorrect (those are in a historical plan doc, not executable code -- no action needed).
- **Follow-ups:** (a) Rename this plan file to `ralplan-supervisor-agent.md` after both commits land. (b) After v1 proves stable over 50+ iterations, consider auto-implementing low-risk items. (c) Consider adding `--status` query mode for operator.

---

## Resolved Open Questions (from rev-1, updated rev-3)

### Q1: Fuzzy-match algorithm
**Decision: `difflib.SequenceMatcher` ratio >= 0.5 on case-folded excerpt text.** Stdlib, handles insertions/deletions in natural language, no compiled dependency.

### Q2: Spec-drafting format
**Decision: Minimal template, paste excerpt + backlog metadata. No auto-expansion.** Template for `.omc/specs/deep-interview-<backlog-id>.md`.

### Q3: Crash-event sources (updated rev-3)
**Decision: Four crash categories, checked in order.** The classifier uses traceback presence, eval exit code, and faulting-frame heuristics:

1. **`retrain_crash`** -- last JSONL event matches `classifier.retrain.failed` OR `last_eval.log` traceback's top frame points to `splice/classifier/train_classifier.py` or sklearn internals. Policy: increment crash counter, halt after 3 consecutive.
2. **`pipeline_bug`** -- `last_eval.log` traceback's top frame is a runtime module (`autoresearch/*.py`, `splice/detector.py`, `splice/features.py`, `splice/ml_eval.py`, `splice/classifier/shap_report.py`) OR RESULTS_TSV is missing/malformed (parse_fail -- eval exited non-zero but no Python traceback). Policy: increment crash counter, halt after 3 consecutive.
3. **`hypothesis_content`** -- `last_eval.log` has no traceback AND eval exited 0 (catastrophic discard -- combined <= 0.05 but pipeline ran clean). Policy: do NOT increment crash counter; exit 0.
4. **`unclassifiable`** -- `last_eval.log` traceback present but top frame matches NONE of {runtime, classifier, retrain} AND eval exited != 0. Policy: do NOT increment (caution), emit a WARN event, exit 0.

Crash counter is stored in `.omc/supervisor-crash-counter.txt` (single-line integer). Incremented on `retrain_crash` or `pipeline_bug`; reset to 0 on any successful iteration (i.e., `keep`, `discard` with combined > 0.05, or `skip`). Gitignored.

### Q4: Iteration-count source
**Decision: `results.tsv` line count (minus 1 for header).** Periodic trigger fires when `iter_count % 10 == 0 && iter_count > 0`.

### Q5: What happens if `--maintain` itself crashes
**Decision: Warn-and-continue (exit code >= 2 treated as non-fatal).** The wrapper logs an ERROR event and continues.

---

## Context

- `verify_agent.py` currently uses manual `sys.argv` parsing for `--diagnose`, `--retest`, `--self-test` subcommands, short-circuiting before the `argparse` block that handles the original `--agent-name --reported-combined` verification flow.
- The wrapper (`run_autoresearch.sh`) calls verify_agent at 9 sites: 4 `--diagnose` sites (L945, L1015, L1031, L1056), 1 `--agent-name --reported-combined` verify site (L1078), and 4 comment/variable references (L480, L497, L992, L1070/L1074).
- The 4 `--diagnose` sites each correspond to a distinct crash category:
  - **L945** `retrain.auto.failed` -- retrain crash (has traceback)
  - **L1015** `eval.crash` -- evaluate.py non-zero exit (has traceback)
  - **L1031** `eval.parse_fail` -- RESULTS_TSV line missing/malformed (no traceback; eval exited 0 but output is bad)
  - **L1056** `discard.catastrophic` -- combined <= 0.05 (no crash, no traceback; eval succeeded with bad score)
- The enhancement backlog (`.omc/enhancement-backlog.md`) has 7 H2 entries with structured fields.
- Tests live in `autoresearch/tests/` using `unittest`/`pytest`.

---

## Work Objectives

1. Rename `verify_agent.py` to `supervisor_agent.py` with full reference cascade.
2. Restructure CLI to use `argparse.add_mutually_exclusive_group` with explicit `--verify`/`--diagnose`/`--maintain`/`--retest` flags.
3. Implement `--maintain` subcommand (triage, crash classify, spec draft).
4. Integrate into wrapper at crash and periodic trigger points.

---

## Guardrails

### MUST Have
- `git mv` for the rename (preserve blame).
- Mutually exclusive argparse group with `required=True` for `--verify`/`--diagnose`/`--maintain`/`--retest`.
- `--trigger` argument validated only when `--maintain` is active.
- Exit-code contract: 0=continue, 1=halt, >=2=unexpected (warn-and-continue).
- Fuzzy dedup against existing backlog entries before appending new ones.
- Spec drafting only for `status: pending` items with `risk: low` and `request_count >= 3`.
- Auto-defer after 3 fires with `status: spec_drafted` and no human action.
- JSONL event emission for every action.
- Sentinel file disable mechanism.
- All tests pass after each commit.
- Static-grep invocation gate for wrapper calls (see commit 1 acceptance criteria).

### MUST NOT
- Invoke any LLM.
- Generate code files (only `.omc/specs/deep-interview-<id>.md` templates and `.omc/enhancement-backlog.md` updates).
- Modify PROTECTED files (exception: `splice/program.md` gets a one-time maintainer exemption for the rename; `CLAUDE.md` is not in PROTECTED_FILES and is freely editable).
- Commit during `.omc/retest-in-progress` presence.
- Touch `.omc/plans/*` or overwrite existing `.omc/specs/deep-interview-*.md` files.
- Add backward-compat fallback for the old implicit-verify call shape (we own all callers).

---

## Files Touched

### Commit 1: Rename + argparse restructure

| File | Action | Details |
|------|--------|---------|
| `autoresearch/verify_agent.py` | `git mv` -> `autoresearch/supervisor_agent.py` | Rename |
| `autoresearch/supervisor_agent.py` | Edit | (a) Update module docstring: "supervisor_agent" not "verify_agent". (b) Replace manual `sys.argv` dispatch + trailing argparse block with `add_mutually_exclusive_group(required=True)` for `--verify`/`--diagnose`/`--maintain`/`--retest`. (c) `--maintain` initially raises `NotImplementedError("commit 2")` or prints stub + exits 0. (d) `--self-test` and `--retest-self-test` become positional or remain as non-exclusive debug flags outside the group. (e) Update internal comment at L243. |
| `run_autoresearch.sh` | Edit (9 sites) | Replace `verify_agent.py` with `supervisor_agent.py` at all 9 sites. At L1078 (the verify call), add `--verify` flag. At L945/L1015/L1031/L1056 (diagnose calls), already use `--diagnose` -- just update the path. Comments at L480/L497/L992/L1070/L1074 updated. |
| `autoresearch/tests/test_retest.py` | Edit | Update `VERIFY_AGENT_PATH` constant (L27) -> `SUPERVISOR_AGENT_PATH`. Update `_load_verify_agent` -> `_load_supervisor_agent`. Update docstring (L4, L9). Update all call sites (~8 `mod = _load_verify_agent()` -> `_load_supervisor_agent()`). |
| `autoresearch/tests/test_smoke_iteration.py` | Audit | Grep found 0 references -- no changes needed. |
| `autoresearch/tests/test_logger.py` | Audit | Grep found 0 references -- no changes needed. |
| `CLAUDE.md` | Edit (8 sites) | Update 4 invocation examples to use `supervisor_agent.py --verify` (L9), `supervisor_agent.py --retest --dry-run` (L88), `supervisor_agent.py --retest` (L94), plus path refs at L57, L201, L213, L214, L239. |
| `splice/program.md` | Edit (1 site) | **PROTECTED -- one-time maintainer exemption.** L90: `verify_agent.py --agent-name` -> `supervisor_agent.py --verify --agent-name`. |
| `.omc/evaluate_integration.md` | Edit (3 sites) | L180, L199, L249: update `verify_agent` -> `supervisor_agent`. |
| `scripts/phase_stats.py` | Edit (1 site) | L9 comment: `verify_agent` -> `supervisor_agent`. |

**NOT updated (historical/non-executable):**
- `.omc/research_notes.md` -- historical log entries, ~15 references. These are timestamped records; updating them would falsify history.
- `.omc/plans/ralplan-verify-agent-resume.md` -- historical plan, ~12 references. Same rationale.
- `.omc/plans/ralplan-folder-reorg.md` -- historical plan, 3 `from autoresearch.verify_agent import` references. Non-executable.
- `.omc/prd.json` -- 5 references in historical user story descriptions. Non-executable.

### Commit 2: `--maintain` subcommand + wrapper hooks + tests

| File | Action | Details |
|------|--------|---------|
| `autoresearch/supervisor_agent.py` | Edit (~300 lines) | Replace `--maintain` stub with full `run_maintain()` implementation. Add helper functions: `_parse_backlog`, `_serialize_backlog`, `_fuzzy_match`, `_normalize_text`, `_extract_enhancement_bullets`, `_draft_spec`, `_load_state`, `_save_state`, `_classify_crash`. |
| `autoresearch/tests/test_maintain.py` | New (~300 lines) | 12 test cases (see Task Flow below). |
| `autoresearch/tests/fixtures/maintain/` | New directory | Fixture files for test_maintain.py (updated for 4 crash categories). |
| `run_autoresearch.sh` | Edit (~30 lines) | Add `_maybe_periodic_maintain` function. Add crash-trigger `--maintain` calls after each of the 4 `--diagnose` sites. Add periodic-trigger call. Total: 2 new `supervisor_agent.py --maintain` invocation sites (1 crash helper called from 4 locations, 1 periodic). Post-commit-2 total sites in wrapper: 7 (5 renamed from commit 1 + 2 new). |
| `.omc/supervisor-crash-counter.txt` | New (runtime) | Single-line integer crash counter. Gitignored (see `.gitignore` edit below). |
| `.gitignore` | Edit (+1 line) | Append `.omc/supervisor-crash-counter.txt` so the runtime counter file does not accumulate in the working tree. Land atomically with commit 2 — don't defer. |
| `CLAUDE.md` | Edit | Add `--maintain` documentation section. |

---

## Task Flow

### Commit 1: Rename + argparse restructure + caller updates

#### 1.1 Rename the file

```bash
git mv autoresearch/verify_agent.py autoresearch/supervisor_agent.py
```

**Acceptance criteria:**
- `git status` shows rename, not delete+add.
- `git log --follow autoresearch/supervisor_agent.py` shows full history.

#### 1.2 Restructure argparse in `supervisor_agent.py`

Replace the current dispatch pattern (manual `sys.argv` checks at L1517-L1564, then argparse at L1566-L1568) with:

```python
def main():
    parser = argparse.ArgumentParser(
        description="Supervisor agent for the autoresearch loop: "
                    "verify hypotheses, diagnose crashes, maintain the pipeline."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true",
                      help="Verify agent work (requires --agent-name + --reported-combined)")
    mode.add_argument("--diagnose", action="store_true",
                      help="Diagnose last eval/retrain crash")
    mode.add_argument("--maintain", action="store_true",
                      help="Triage enhancements, classify crashes, draft specs")
    mode.add_argument("--retest", metavar="FROM_SHA",
                      help="Replay discarded hypotheses from FROM_SHA")

    # --verify subflags
    parser.add_argument("--agent-name", help="Name of the agent being verified (--verify)")
    parser.add_argument("--reported-combined", type=float,
                        help="Combined score reported by agent (--verify)")

    # --maintain subflags
    parser.add_argument("--trigger", choices=["crash", "periodic", "manual"],
                        default="manual",
                        help="Maintain trigger type (--maintain)")

    # --retest subflags
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview retest candidates without running (--retest)")
    parser.add_argument("--limit", type=int,
                        help="Max candidates to process (--retest)")
    parser.add_argument("--retest-re-eval", action="store_true",
                        help="Force re-evaluation (--retest)")
    parser.add_argument("--retest-origin", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.verify:
        if not args.agent_name or args.reported_combined is None:
            parser.error("--verify requires --agent-name and --reported-combined")
        # ... existing verify logic ...
    elif args.diagnose:
        sys.exit(run_diagnose())
    elif args.maintain:
        sys.exit(run_maintain(trigger=args.trigger))  # stub in commit 1
    elif args.retest:
        sys.exit(run_retest(args.retest, dry_run=args.dry_run,
                            limit=args.limit, re_eval=args.retest_re_eval))
```

Keep `--self-test` and `--retest-self-test` as non-mutually-exclusive debug flags (they are dev-only, not called by the wrapper):
```python
parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--retest-self-test", action="store_true", help=argparse.SUPPRESS)
```
These are checked before the mode dispatch: `if args.self_test: sys.exit(_diagnose_self_test())`.

Update the module docstring (L2-L5) to say `supervisor_agent.py` and list all 4 modes.

Update the internal comment at L243 to say `supervisor_agent.py --diagnose`.

**Acceptance criteria:**
- `PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --verify --agent-name test --reported-combined 0.5` -- argparse accepts it, enters verify flow.
- `PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --diagnose` -- enters diagnose flow.
- `PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --retest abc123 --dry-run` -- enters retest flow.
- `PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --maintain --trigger=manual` -- prints stub message, exits 0.
- `PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py` (no mode flag) -- argparse error: "one of the arguments --verify/--diagnose/--maintain/--retest is required".
- `PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --verify --diagnose` -- argparse error: mutually exclusive.
- `--self-test` and `--retest-self-test` still work.

#### 1.3 Update all callers

**`run_autoresearch.sh` (9 sites):**
- L945: `autoresearch/verify_agent.py --diagnose` -> `autoresearch/supervisor_agent.py --diagnose`
- L1015: same
- L1031: same
- L1056: same (+ L1058 error message string)
- L1078: `autoresearch/verify_agent.py \` (with `--agent-name` on next line) -> `autoresearch/supervisor_agent.py --verify \`
- L480, L497, L992, L1070, L1074: comment references updated

**`autoresearch/tests/test_retest.py`:**
- L4, L9: docstring references
- L27: `VERIFY_AGENT_PATH = REPO_ROOT / "autoresearch" / "verify_agent.py"` -> `SUPERVISOR_AGENT_PATH = REPO_ROOT / "autoresearch" / "supervisor_agent.py"`
- L31-L39: rename function `_load_verify_agent` -> `_load_supervisor_agent`, update module name string
- L115, L157, L221, L251, L339, L406, L428, L543: all `_load_verify_agent()` call sites

**`CLAUDE.md` (8 sites):**
- L9: verify invocation example + add `--verify` flag
- L57: "part of verify_agent.py" -> "part of supervisor_agent.py"
- L88: retest dry-run example
- L94: retest live example
- L201, L213, L214, L239: prose references

**`splice/program.md` (1 site, PROTECTED):**
- L90: `uv run python autoresearch/verify_agent.py --agent-name` -> `uv run python autoresearch/supervisor_agent.py --verify --agent-name`
- Maintainer exemption rationale: the file being renamed IS the tool that enforces PROTECTED_FILES. The PROTECTED_FILES list itself does not change (it references `autoresearch/manifest.json` and `autoresearch/preflight.py`, not `verify_agent.py`).

**`.omc/evaluate_integration.md` (3 sites):**
- L180, L199, L249: update references

**`scripts/phase_stats.py` (1 site):**
- L9: comment text

**Acceptance criteria:**
- `grep -rn "verify_agent" autoresearch/ scripts/ run_autoresearch.sh CLAUDE.md splice/program.md .omc/evaluate_integration.md` -> 0 matches.
- `uv run python -m pytest autoresearch/tests/test_retest.py -v` -- all tests pass.
- `uv run python -m pytest autoresearch/tests/ -v` -- all existing tests pass.
- `PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --self-test` -- passes.
- Static-grep invocation gate -- every non-comment `supervisor_agent.py` invocation in the wrapper must carry a mode flag:

```bash
# Every non-comment supervisor_agent.py invocation in run_autoresearch.sh
# must carry a mode flag. Failure = missing --verify after rename.
grep -n "supervisor_agent.py" run_autoresearch.sh \
  | grep -vE '\-\-(verify|diagnose|maintain|retest)' \
  | grep -v '^[0-9]*:#' \
  | tee /dev/stderr | grep -q . \
  && { echo "FAIL: bare invocation without mode flag"; exit 1; } \
  || echo "PASS: all invocations tagged"
```

#### 1.4 Commit

```
US-517: rename verify_agent -> supervisor_agent + argparse restructure

git mv verify_agent.py supervisor_agent.py. Restructure CLI from ad-hoc
sys.argv dispatch to argparse mutually exclusive group:
--verify / --diagnose / --maintain / --retest.

Breaking change: all callers must now pass --verify explicitly
(previously implicit when --agent-name was present).

Updated: run_autoresearch.sh (9 sites), test_retest.py,
CLAUDE.md (8 sites), splice/program.md (1, maintainer exemption),
evaluate_integration.md (3), scripts/phase_stats.py (1).
```

---

### Commit 2: `--maintain` subcommand + wrapper hooks + tests

#### 2.1 Implement `run_maintain(trigger: str) -> int`

Replace the stub from commit 1 with the full implementation.

**Common preamble (all triggers):**
- Check `.omc/maintainer-disabled` sentinel; if exists, emit JSONL `maintain.skipped` event, return 0.
- Check `.omc/retest-in-progress` sentinel; if exists, emit JSONL `maintain.skipped` event, return 0.
- Load `.omc/enhancement-backlog.md` via `_parse_backlog()`.
- Load `.omc/maintainer-state.json`. Create with defaults if missing.

> **Sentinel timing semantics:** The `.omc/maintainer-disabled` sentinel is checked ONCE at invocation start (wrapper-side before `uv run`, Python-side at function entry). If the operator creates the sentinel while `--maintain` is executing, the current invocation runs to completion; the disable takes effect at the next trigger. To stop an in-flight maintain call, the operator must cancel the enclosing `./run_autoresearch.sh` process (tmux-attach then Ctrl-C) -- there is no poll-based cancellation.

**Crash path (`trigger=crash`):**
- Read `.omc/last_eval.log`.
- Classify using `_classify_crash()` (see decision table below).
- For `pipeline_bug`/`retrain_crash`: increment crash counter in `.omc/supervisor-crash-counter.txt`. If >= 3, return 1 (halt).
- For `hypothesis_content`: reset crash counter to 0, return 0.
- For `unclassifiable`: do NOT increment crash counter, emit JSONL `maintain.crash.unclassifiable` WARN event, return 0.
- Emit JSONL `maintain.crash.classified` event with category.

**`_classify_crash()` decision table and implementation sketch:**

```python
def _classify_crash(log_text: str, eval_exit_code: int) -> str:
    """Classify a crash into one of 4 categories.

    Categories:
      retrain_crash     - retrain infrastructure failure
      pipeline_bug      - runtime module error or parse_fail
      hypothesis_content - catastrophic discard, pipeline ran clean
      unclassifiable    - traceback present but frame matches nothing known
    """
    traceback = _extract_last_traceback(log_text)

    if traceback is None:
        if eval_exit_code == 0:
            return 'hypothesis_content'  # catastrophic discard, no crash
        else:
            return 'pipeline_bug'  # parse_fail path (exited non-zero, no Python traceback)

    # Traceback is present -- classify by faulting frame
    top_frame = _parse_traceback(traceback)

    # Check retrain first (classifier/sklearn internals)
    RETRAIN_PATTERNS = [
        'splice/classifier/train_classifier.py',
        'sklearn/',
    ]
    if any(p in top_frame for p in RETRAIN_PATTERNS):
        return 'retrain_crash'

    # Check for last JSONL event matching classifier.retrain.failed
    if _last_jsonl_event_matches(log_text, 'classifier.retrain.failed'):
        return 'retrain_crash'

    # Check runtime modules
    RUNTIME_PATTERNS = [
        'autoresearch/',
        'splice/detector.py',
        'splice/features.py',
        'splice/ml_eval.py',
        'splice/classifier/shap_report.py',
    ]
    if any(p in top_frame for p in RUNTIME_PATTERNS):
        return 'pipeline_bug'

    # Traceback present but top frame matches nothing known
    return 'unclassifiable'
```

**Crash counter state:** `.omc/supervisor-crash-counter.txt` -- a single-line integer file. Incremented on `retrain_crash` or `pipeline_bug`; reset to 0 on any successful iteration (i.e., `keep`, `discard` with combined > 0.05, or `skip`). The counter reset on success is handled by the wrapper (see section 2.5), not by `--maintain` itself. Gitignored.

**Periodic path (`trigger=periodic`):**
- Scan `.omc/research_notes.md` for `(e) Wrapper enhancements` sections.
- Extract bullet points; fuzzy-match against existing backlog entries.
- Match found: bump `request_count`, update `last_seen`.
- No match: append new H2 entry with `status: pending`, `request_count: 1`.
- Scan for spec-draft-eligible entries (`status: pending`, `risk: low`, `request_count >= 3`): draft spec.
- Scan for auto-defer-eligible entries (`status: spec_drafted`, `spec_drafted_fires >= 3`): set `status: deferred`.
- Write updated backlog + state. Single git commit if any file changed.
- Return 0.

**Manual path (`trigger=manual`):**
- Run both crash classification (if log exists) and periodic triage. Return crash path's exit code.

**Acceptance criteria:**
- Crash path returns 1 only after 3+ consecutive infrastructure crashes.
- `hypothesis_content` resets the crash counter, does not increment.
- `unclassifiable` does not increment the crash counter, emits WARN.
- Periodic path appends, deduplicates, drafts specs, auto-defers.
- No file outside the allowed set is ever written.
- Git commit stages only `.omc/` paths; aborts (exit 2) if unexpected paths staged.

#### 2.2 Implement helper functions

- `_parse_backlog(path) -> list[dict]`
- `_serialize_backlog(entries, path)`
- `_fuzzy_match(needle, haystack, threshold=0.5) -> dict | None`
- `_normalize_text(s) -> str`
- `_extract_enhancement_bullets(notes_path, since_sha) -> list[str]`
- `_draft_spec(entry, specs_dir) -> Path | None`
- `_load_state(path) -> dict` / `_save_state(state, path)`
- `_classify_crash(log_text, eval_exit_code) -> str` (see 2.1 for full decision table)
- `_last_jsonl_event_matches(log_text, event_name) -> bool`

**Acceptance criteria:**
- `_parse_backlog` round-trips the bootstrapped backlog (parse then serialize produces identical output).
- `_fuzzy_match` returns correct entry for near-verbatim excerpts and None for unrelated text.
- `_draft_spec` creates the file and refuses to overwrite existing.
- `_classify_crash` returns the correct category for each of the 4 cases (see test cases).

#### 2.3 Loop-safety enforcement

At the top of `run_maintain()`:
- Assert no retest sentinel.
- Stage only specific `.omc/` files before commit.
- Verify `git diff --cached --name-only` contains only expected paths before committing.
- If unexpected paths staged, abort commit, emit ERROR JSONL, return 2.

#### 2.4 Tests (`autoresearch/tests/test_maintain.py`)

All tests use `unittest.TestCase` with `tempfile.TemporaryDirectory`. Fixtures in `autoresearch/tests/fixtures/maintain/`.

1. **`test_fresh_run_no_op`** -- Bootstrapped backlog, no new entries, trigger=periodic. Exit 0, no changes.
2. **`test_fuzzy_match_existing_entry`** -- Near-match bullet. `request_count` incremented.
3. **`test_new_request_appended`** -- Novel bullet. New H2 appended.
4. **`test_crash_pipeline_bug_halts`** -- 3rd consecutive pipeline_bug crash. Exit 1.
5. **`test_crash_retrain_crash_halts`** -- 3rd consecutive retrain_crash. Exit 1.
6. **`test_crash_hypothesis_content_continues`** -- No traceback, eval exit 0. Exit 0, counter reset.
7. **`test_crash_unclassifiable_continues`** -- Traceback present but unknown frame. Exit 0, counter NOT incremented, WARN emitted.
8. **`test_periodic_spec_draft`** -- Eligible entry gets spec drafted.
9. **`test_auto_defer_after_3_fires`** -- 3rd fire without human action. Status -> deferred.
10. **`test_disabled_sentinel`** -- Sentinel exists. Exit 0, no side effects.
11. **`test_retest_sentinel_skips`** -- Retest sentinel exists. Exit 0, no side effects.
12. **`test_backlog_roundtrip`** -- Parse + serialize = identical output.

**Note:** Rev-1/rev-2 had 10 test cases with only 2 crash categories (`pipeline_bug`, `hypothesis_content`). Rev-3 adds `retrain_crash` and `unclassifiable` categories, requiring 2 additional test cases (items 5 and 7 above). Fixtures must be updated to include sample tracebacks for all 4 categories: a sklearn traceback (retrain_crash), a `splice/detector.py` traceback (pipeline_bug), no traceback with exit 0 (hypothesis_content), and an unrecognized-frame traceback (unclassifiable).

**Acceptance criteria:**
- All 12 tests pass with `uv run python -m pytest autoresearch/tests/test_maintain.py -v`.
- Tests do not touch the real `.omc/` directory.

#### 2.5 Wrapper hooks (`run_autoresearch.sh`)

**Crash-trigger (after each of the 4 `--diagnose` sites):**
```bash
if [ ! -f "$PROJECT_DIR/.omc/maintainer-disabled" ]; then
    set +e
    uv run python autoresearch/supervisor_agent.py --maintain --trigger=crash \
        >>"$CHILD_STDERR_LOG" 2>&1
    _maintain_rc=$?
    set -e
    if [ $_maintain_rc -eq 1 ]; then
        touch "$STOP_FILE"
        _log WARN wrapper maintain.halt trigger=crash
    elif [ $_maintain_rc -ge 2 ]; then
        _log ERROR wrapper maintain.crash rc="$_maintain_rc"
    fi
fi
```

Extract as `_maybe_crash_maintain` function to avoid duplicating the block 4 times.

> **Sentinel timing semantics (wrapper side):** The `.omc/maintainer-disabled` sentinel check in the wrapper (`[ ! -f ... ]`) runs once per trigger site. If the operator creates the sentinel between two trigger sites in the same iteration, the earlier site may have already fired; the later site will see the sentinel and skip. There is no mid-execution poll. To stop an in-flight `--maintain` call, the operator must cancel the enclosing `./run_autoresearch.sh` process (tmux-attach then Ctrl-C).

**Crash counter reset on success:** After any successful iteration (`keep`, `discard` with combined > 0.05, or `skip`), the wrapper resets `.omc/supervisor-crash-counter.txt` to `0`:
```bash
echo 0 > "$PROJECT_DIR/.omc/supervisor-crash-counter.txt"
```

**Periodic-trigger (after `_iter_summary`, before `sleep`/`continue`):**
```bash
_maintain_iter_count=$(( $(wc -l < "$RESULTS" 2>/dev/null || echo 1) - 1 ))
if [ "$_maintain_iter_count" -gt 0 ] && [ $((_maintain_iter_count % 10)) -eq 0 ]; then
    if [ ! -f "$PROJECT_DIR/.omc/maintainer-disabled" ]; then
        set +e
        uv run python autoresearch/supervisor_agent.py --maintain --trigger=periodic \
            >>"$CHILD_STDERR_LOG" 2>&1
        _maintain_rc=$?
        set -e
        if [ $_maintain_rc -eq 1 ]; then
            touch "$STOP_FILE"
            _log WARN wrapper maintain.halt trigger=periodic
        elif [ $_maintain_rc -ge 2 ]; then
            _log ERROR wrapper maintain.crash rc="$_maintain_rc"
        fi
    fi
fi
```

Extract as `_maybe_periodic_maintain` function, called from both discard and keep paths.

**Acceptance criteria:**
- After commit 2, `grep -cn "supervisor_agent" run_autoresearch.sh` shows 7 sites (5 renamed from commit 1 + 2 new maintain invocations in the extracted functions).
- Crash maintain fires after `--diagnose` at all 4 crash paths.
- Periodic maintain fires at multiples of 10 iterations.
- Neither fires when `.omc/maintainer-disabled` exists.
- Static-grep invocation gate passes (all invocations carry a mode flag):

```bash
grep -n "supervisor_agent.py" run_autoresearch.sh \
  | grep -vE '\-\-(verify|diagnose|maintain|retest)' \
  | grep -v '^[0-9]*:#' \
  | tee /dev/stderr | grep -q . \
  && { echo "FAIL: bare invocation without mode flag"; exit 1; } \
  || echo "PASS: all invocations tagged"
```

#### 2.6 CLAUDE.md documentation

Add a section under "## Retest sentinel" documenting:
- The `--maintain` subcommand, triggers, and exit-code contract.
- The `.omc/maintainer-disabled` sentinel.
- The `.omc/supervisor-crash-counter.txt` crash counter file.
- The `.omc/maintainer-state.json` state file.
- The periodic cadence (every 10 iterations).
- The argparse restructure (explicit `--verify` required).
- The 4 crash categories and their policies.

#### 2.7 Commit

```
US-517: add supervisor_agent.py --maintain subcommand + wrapper hooks

Triage enhancement requests from research_notes against the backlog,
classify crash patterns (pipeline_bug/hypothesis_content/retrain_crash/
unclassifiable), draft ralplan specs for recurring low-risk items,
auto-defer stale specs.

Wrapper integration: crash-trigger after --diagnose (4 sites),
periodic-trigger every 10 iterations. Disable via .omc/maintainer-disabled.
Crash counter in .omc/supervisor-crash-counter.txt (reset on success).
```

---

## Success Criteria

1. `grep -rn "verify_agent" autoresearch/ scripts/ run_autoresearch.sh CLAUDE.md splice/program.md .omc/evaluate_integration.md` -> 0 matches after commit 1.
2. `PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --verify --agent-name test --reported-combined 0.5` -> successful argparse parse, enters verify flow.
3. `PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --maintain --trigger=manual` -> runs the maintain routine (commit 2).
4. `grep -cn "supervisor_agent" run_autoresearch.sh` -> 7 sites after commit 2.
5. `uv run python -m pytest autoresearch/tests/test_retest.py -v` -> all tests pass after commit 1.
6. `uv run python -m pytest autoresearch/tests/test_maintain.py -v` -> all 12 cases pass after commit 2.
7. `uv run python -m pytest autoresearch/tests/ -v` -> all tests pass after commit 2.
8. `touch .omc/maintainer-disabled && PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --maintain` -> exit 0, no side effects.
9. No PROTECTED files modified in commit 2. `splice/program.md` modified in commit 1 under maintainer exemption.
10. Both commits are clean (no unrelated changes, no debug prints).
11. Static-grep invocation gate passes after both commits (no bare `supervisor_agent.py` invocations without mode flags in wrapper).

---

## Rollback

**Commit 1 (rename + argparse):**
- `git revert <commit-1-sha>` restores old name and argv parsing. All callers revert too.
- Alternatively, `git mv autoresearch/supervisor_agent.py autoresearch/verify_agent.py` + manual caller restore.

**Commit 2 (maintain subcommand):**
- Immediate: `touch .omc/maintainer-disabled` (no restart needed).
- Permanent: `git revert <commit-2-sha>` removes wrapper hooks and Python code. Commit 1 (rename) stays.
- Full: revert both commits in reverse order.

---

## Estimated Complexity

**MEDIUM.** Commit 1: ~15 files touched but purely mechanical (rename + search-replace + argparse restructure). Commit 2: ~600 lines of new code across 3 files (supervisor_agent.py, test_maintain.py, run_autoresearch.sh). No new dependencies. No architectural changes beyond the rename. Rev-3 adds ~50 lines over rev-2 (crash classifier expansion + crash counter file + sentinel timing docs).

---

## Follow-ups (not in scope)

- [ ] Rename this plan file to `ralplan-supervisor-agent.md` after both commits land.
- [ ] After v1 stable over 50+ iterations: auto-implement low-risk items via `/ralplan + /autopilot`.
- [ ] `--maintain --status` query mode for operator.
- [x] ~~Add `.omc/supervisor-crash-counter.txt` to `.gitignore`~~ — folded into commit 2's file table; lands atomically with the counter's first use.

## Non-updated references (intentional)

- **`splice/evaluate.py:662`** (PROTECTED): contains a comment-level reference to `verify_agent`. NOT updated — the PROTECTED-file exemption cost is not worth a one-line comment rewrite, and the rename grep gate (`grep -n "supervisor_agent.py" run_autoresearch.sh`) specifically targets invocation sites in the wrapper, not comment prose elsewhere. Leave as-is; history via `git log --follow` remains traceable.
- **Historical `.omc/plans/*` and `.omc/specs/*`** mentioning `verify_agent`: frozen-in-time context from prior planning sessions. Not updated.
