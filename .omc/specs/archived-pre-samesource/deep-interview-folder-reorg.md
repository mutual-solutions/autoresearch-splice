# Deep Interview Spec: US-516 — Folder reorg (runtime / splice / state separation, full package-ification)

## Metadata
- Interview ID: folder-reorg-2026-04-19
- Rounds: 4
- Final Ambiguity: 15%
- Type: brownfield
- Status: PASSED (≤ 20%)

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|---|---|---|---|
| Goal | 0.85 | 0.35 | 0.30 |
| Constraint | 0.80 | 0.25 | 0.20 |
| Success | 0.85 | 0.25 | 0.21 |
| Context | 0.90 | 0.15 | 0.14 |
| **Total** | | | **0.85** |
| **Ambiguity** | | | **15%** |

## Goal

**Primary pain:** navigation. The repo is unreadable because source code, runtime state, plans/specs, classifier artifacts, operator tools, ad-hoc analysis, and interactive scratch all live side-by-side. `.omc/` has become a junk drawer with 15+ subpurposes; `scripts/` mixes operator UIs with library helpers; the root holds both the hot-path modules (`evaluate.py`, `detector.py`, `features.py`, `ml_eval.py`, `dataset_registry.py`) and the shell wrapper; ~17 `sys.path.insert` hacks paper over the missing package structure.

**Secondary wins (stated preferences, all agreed to land in-scope):**
- **Package-ification (`c` gate, user's words: "never thought about but I think it's a good idea")** — zero `sys.path.insert`, proper packages, clean imports.
- **Runtime / application harness boundary (`d` gate, user's words: "I want to make this my own ML research harness")** — isolate the reusable autoresearch runtime from the audio-splice-specific application so a future detection problem can fork, delete the splice dir, and start from a working harness skeleton.
- **Agent write-scope clarity (`b`, "not so important but preferable")** — easier to reason about what the autoresearch agent can/cannot edit once directories are semantically separated.

**Harness scope — explicit decision:** the reusable harness is **loop-only**. Detector, features, classifier, evaluate are *application code*, not harness. A new detection problem would copy: the autoresearch loop wrapper, the unified logger, verify_agent, baseline tracking, retest. It would write its own `evaluate.py`, its own detector/features/classifier. (Interview Round 2, option **a**.)

## Constraints

### Protected-file movements (Round 4, Q1 → answer (i))
- `evaluate.py` and `program.md` are currently PROTECTED from the autoresearch agent. **They move as part of US-516 under a one-time maintainer exemption** (same mechanism used in US-515 phase 1 for `evaluate.py`'s `print` → logger migration). `verify_agent.py`'s PROTECTED path regex must be updated in the same PR so the agent's diff audit finds the new locations.

### Git history preservation (Round 4, Q2 → answer (iii))
- **Don't care.** No `git mv` purity requirement. Commit messages carry the context; blame can be followed via `git log --follow` if someone needs it. This unlocks bundling renames with the content edits required for the import-clean gate.

### External references to moved paths (Round 4, Q3 → answer (iii), enumerated by scan)
Scan complete — all in-repo. No external `~/.claude/` settings/hooks reference repo paths. No `pyproject.toml` entry points. Concrete update list:

**Code:**
- `run_autoresearch.sh` — 8+ hardcoded paths: `.omc/coordination/log_cli.py`, `.omc/coordination/verify_agent.py` (5 sites: L944/1014/1030/1055/1077), `.omc/coordination/preflight.py` (comment), per-iteration artifacts.
- `evaluate.py:23,31` — `from detector import detect_splices`, `from logger import get_logger` + `sys.path.insert` block.
- `detector.py:31` — `sys.path.insert` to `.omc/coordination`.
- `features.py:50,60-61` — `from detector import (...)` + `sys.path.insert` + `from logger import`.
- `ml_eval.py:22,27` — `from detector import ...` + `from logger import`.
- `.omc/classifier/train_classifier.py:35,37-39,42-43` — 2× `sys.path.insert`, `from dataset_registry`, `from detector`, `from features`, `from logger`.
- `.omc/classifier/shap_report.py:28,29,34` — 2× `sys.path.insert` + `from logger`.
- `.omc/coordination/verify_agent.py:26,32-34` — PROTECTED_FILES regex, 2× `sys.path.insert`, `from log_reader`.
- `.omc/coordination/log_cli.py:37,39` — `sys.path.insert` + `from logger`.
- `.omc/coordination/logger.py:13` (docstring), `.omc/coordination/fp_filter.py:218` (path literal).
- `scripts/phase_stats.py:19-20`, `scripts/dashboard.py:31-32`, `scripts/tunable_frontier.py:30-31`, `scripts/notebook_digest.py:29-30`, `scripts/validate_logs.py:36+` — all 5 have `sys.path.insert` + `from logger` / `from log_reader`.
- `data_synth/regenerate_datasets.py:43` — inserts `scripts/`.
- `.omc/coordination/tests/test_logger.py:13`, `test_log_readers.py:15`, `test_retest.py`, `test_smoke_iteration.py:37` — all 4 have `sys.path.insert` and reference source paths.

**Docs:**
- `README.md` — refs `evaluate.py`.
- `program.md` — refs `evaluate.py`, `detector.py`, `features.py`, `.omc/classifier/train_classifier.py`, `.omc/coordination/verify_agent.py`.
- `CLAUDE.md` — refs `.omc/coordination/verify_agent.py`, `.omc/coordination/preflight.py`, `.omc/coordination/manifest.json`, `evaluate.py`.
- `.omc/evaluate_integration.md` — historical doc; refs `.omc/coordination/verify_agent.py`.

**Not touched in US-516:**
- `.omc/plans/*`, `.omc/specs/*` — historical planning artifacts; paths in them are frozen-in-time context, no need to retroactively rewrite.
- `.omc/prd.json` — same.
- Untracked debris at root (`detector_v*.py`, `*.stderr`, `*.stdout`, `*.log`, `*.png`) — leave alone (or sweep in a separate tidying commit).
- `~/.claude/` — scan confirms no refs.

### Timing (Round 4, Q4 → answer (ii), with loop-safety caveat)
- **Land immediately after phase-2 commits.** Not blocked on N clean iterations or retest backlog.
- **Loop-safety caveat:** ttys001 noted that phase-2 landed with the loop live, and any commit on `autoresearch/apr15` races with the wrapper's `_guarded_reset` and `_append_note` auto-commits. **US-516 execution stops the loop for its duration** (user picked option A in the pre-spec triage). Loop restart moves into US-516's acceptance criteria.

### Non-Goals
- NOT a rewrite. Pure structural + import-path changes; zero behavior changes, zero new features, zero new tests beyond what's needed to keep the existing gates green post-move.
- NOT refactoring `verify_agent.py` into separate `baseline.py`/`retest.py` modules. That's a separate PR if it ever happens.
- NOT pip-packaging. US-516 produces a clean *in-repo* package structure; it does not set up `pyproject.toml` entry points, does not publish to PyPI, does not split into multiple repos. The harness becomes *extractable* (gate d) but is not extracted in this PR.
- NOT moving phase-3 carve-outs. `RESULTS_TSV:` is still written by `evaluate.py`; `combined_{ds.id}: ERROR` is still printed; `_write_retest_report` is still a string writer. All three migrate later (phases 3a/3b/3c per US-515's follow-up list).
- NOT renaming `.omc/`. It shrinks to state-only but keeps its name (every doc and every external reference already calls it that).
- NOT touching `data/`, `data_synth/internals`, `reports/`, `.omc/feature_cache/`, `.omc/retest-worktree/` beyond updating paths that reference them.

## Acceptance Criteria

Four concrete gates (user's **a**+**b**+**c**+**d**, all in-scope):

- [ ] **(a) `ls` test.** Top-level `ls` produces **≤ 10 tracked entries**, each self-explanatory. `.omc/` contains *only* runtime state (logs, plans, specs, notes, sentinels, worktrees, feature cache) — **zero `*.py` files** under `.omc/`. `scripts/` contains *only* operator tools and analysis — **zero library helpers** imported by `detector.py` / `features.py` / the runtime.
- [ ] **(b) 30-second locate test.** For each of {logger, detector, features, evaluate, verify_agent, classifier trainer, SHAP reporter, wrapper loop script, baseline metrics, retest code}, locating the file without grep by walking top-level → subdir names takes < 30s. Directory names are nouns that match the thing inside them.
- [ ] **(c) Import-clean test.** `grep -rn "sys\.path\.insert" --include="*.py" .` returns **0 tracked matches** (`.venv/` and untracked debris excluded). All imports use proper package paths (`from autoresearch.logger import get_logger`, `from splice.detector import detect_splices`, etc.). The logger's import-convention docstring at `.omc/coordination/logger.py:13` is replaced with an `import` one-liner.
- [ ] **(d) Fork-friendly runtime test.** `cp -r` the repo to a scratch dir, delete the splice-application directory (whatever it ends up being named), and confirm the loop wrapper, logger, log_reader, log_cli, verify_agent, baseline tracker, and retest all still import cleanly and `uv run python .omc/coordination/tests/test_logger.py` passes. (The loop itself won't run end-to-end because there's no `evaluate.py`, but the runtime package is self-contained.)

**Regression gates (must continue to pass post-move):**
- [ ] `uv run python scripts/validate_logs.py --audit` (phase-1 Python tier + phase-2 bash tier) — 0 violations.
- [ ] `uv run python .omc/coordination/tests/test_smoke_iteration.py` — all 4 gates pass.
- [ ] `pytest .omc/coordination/tests/test_retest.py` — 10/10 pass.
- [ ] `pytest .omc/coordination/tests/test_logger.py .omc/coordination/tests/test_log_readers.py` — 14/14 + 9/9 pass.
- [ ] `uv run python .omc/classifier/train_classifier.py` (post-rename) completes without `ImportError`.
- [ ] End-to-end: after the PR lands, `./run_autoresearch.sh start` produces a live JSONL stream for ≥ 1 iteration with the usual events (`wrapper.loop.started`, `wrapper.iteration.start`, `eval.*`, `classifier.retrain.*`, `wrapper.iteration.phase`). Loop restart itself is part of US-516's closing acceptance.

**Operator gates:**
- [ ] `./run_autoresearch.sh stop` has been run and the sentinel `.omc/autoresearch-stop` is present BEFORE any US-516 commit lands.
- [ ] The reorg PR includes an updated `CLAUDE.md` migration note documenting the new structure + the fact that `evaluate.py` + `program.md` moved under a one-time maintainer exemption.
- [ ] `verify_agent.check_git_diff_audit` PROTECTED_FILES regex updated to match the new paths before loop restart (otherwise the first hypothesis iteration on the new layout would falsely clear the protected check or falsely trip it).

## Assumptions Exposed & Resolved

| Assumption | Challenge | Resolution |
|---|---|---|
| "Reorg" means "tidy the junk drawer" | The user also wants package-ification, harness extraction, and runtime/app separation — a reorg done without those collapses back into clutter in a few sprints | Four-gate acceptance (a+b+c+d) covers the full scope in one PR |
| Harness = whole framework (loop + classifier + features + detector abstractions) | That would take weeks and blur what "audio splice detection" means in the repo | Harness = loop-only. Detector/features/classifier/evaluate stay in the app side. New detection problems re-write those, not fork a plugin system. |
| evaluate.py is too protected to move | But US-515's phase-1 maintainer exemption already set the precedent that the human can edit it under a one-time exemption flag | Same mechanism. `verify_agent`'s PROTECTED regex updates in the same PR. |
| Git history must be preserved across the rename | Preserving `git log --follow` blame adds ordering constraints (rename-only commit separate from content edits) | User: don't care. Bundle renames + content edits in coherent commits; commit messages carry context. |
| US-516 can land on the live loop branch | Phase-2 just proved otherwise — the loop's `_guarded_reset` + `_append_note` auto-commits race with any manual commit on the same branch | Stop loop for US-516 duration (option A). Restart moves into acceptance. |

## Technical Context

### Current tree shape (what's getting reorganized)

**Root source files (to move into the new structure):**
- Hot path: `evaluate.py` (PROTECTED), `detector.py`, `features.py`, `ml_eval.py`, `dataset_registry.py`, `run_autoresearch.sh`.
- Docs: `README.md`, `program.md` (PROTECTED), `CLAUDE.md`, `AGENTS.md` (generated, gitignored).

**`.omc/` — current junk drawer (to split into source-goes-elsewhere + state-stays):**
- Source (must move out of `.omc/`): `coordination/{logger.py, log_cli.py, verify_agent.py, preflight.py, fp_filter.py, manifest.json, baseline_metrics.json}`, `coordination/tests/*`, `classifier/{train_classifier.py, shap_report.py}` + `classifier/fp_classifier.joblib` (artifact — decide: ship as runtime state or app-resident trained model).
- State (stays in `.omc/` — or a renamed state dir): `logs/`, `plans/`, `specs/`, `research_notes.md`, `last_eval.log`, `last_reflection.md`, `retest-worktree/`, `retest-in-progress`, `retest-report.md`, `autoresearch-stop`, `feature_cache/`, `shap_rollup.json`, `tunable_frontier.txt` (retired, see phase-2).

**`scripts/` — mixed-purpose bag (to split):**
- **Library helpers** (imported by the runtime or app): `log_reader.py`, `log_cli.py` (already under `.omc/coordination/`), `validate_logs.py` (CLI tool, but consumed by smoke gate). These move into the new `autoresearch/` or `splice/` packages.
- **Operator / analysis tools** (not imported by runtime): `dashboard.py`, `phase_stats.py`, `log_monitor.py`, `eval_crypto.py`, `test_crypto.py`, `tunable_frontier.py`, `diagnose_repeat_rate.py`, `notebook_digest.py`, `shap_shift.py`, `shap_rollup.py`. These stay under `scripts/` (or a renamed `tools/`).

**Data / artifacts** (already clean): `data/`, `data_synth/`, `reports/`, `results.tsv`.

### Proposed target structure (ralplan will refine naming and exact file placements)

```
autoresearch-splice/
├── autoresearch/                # RUNTIME PACKAGE — reusable loop harness
│   ├── __init__.py
│   ├── logger.py
│   ├── log_reader.py
│   ├── log_cli.py
│   ├── verify_agent.py
│   ├── preflight.py
│   ├── baseline_metrics.json    # config-ish; see open question
│   ├── manifest.json
│   └── tests/
├── splice/                      # APPLICATION PACKAGE — audio-splice-specific
│   ├── __init__.py
│   ├── detector.py
│   ├── features.py
│   ├── ml_eval.py
│   ├── evaluate.py              # moved under one-time maintainer exemption
│   ├── program.md               # same
│   ├── dataset_registry.py
│   └── classifier/
│       ├── __init__.py
│       ├── train_classifier.py
│       ├── shap_report.py
│       ├── fp_filter.py
│       └── fp_classifier.joblib # OR keep in .omc/state/classifier/
├── scripts/                     # OPERATOR TOOLS — thin CLI wrappers
│   ├── dashboard.py
│   ├── phase_stats.py
│   ├── log_monitor.py
│   ├── eval_crypto.py
│   ├── test_crypto.py
│   ├── tunable_frontier.py
│   ├── diagnose_repeat_rate.py
│   ├── notebook_digest.py
│   ├── shap_shift.py
│   ├── shap_rollup.py
│   └── validate_logs.py
├── tests/                       # optional consolidation — OR leave tests/ per-package
│   └── (per-package tests, or flat)
├── data/                        # unchanged
├── data_synth/                  # unchanged (path-refs to scripts/ updated)
├── reports/                     # unchanged (SHAP sidecars, gitignored)
├── .omc/                        # PURE STATE ONLY — no source code
│   ├── logs/
│   ├── plans/
│   ├── specs/
│   ├── research_notes.md
│   ├── last_eval.log
│   ├── last_reflection.md
│   ├── feature_cache/
│   ├── retest-worktree/
│   ├── retest-in-progress
│   ├── retest-report.md
│   ├── autoresearch-stop
│   └── shap_rollup.json
├── run_autoresearch.sh          # wrapper stays at root
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── uv.lock
└── results.tsv
```

**Open structural questions (for ralplan to resolve):**
- `fp_classifier.joblib` — runtime state (`.omc/state/classifier/`) or app-resident artifact (`splice/classifier/`)? The former reflects "joblib is regenerated by retrain"; the latter reflects "the trained model is part of the app's behavior and should not wander when state dirs get wiped."
- `baseline_metrics.json` + `manifest.json` — is there a case for keeping these in `.omc/state/` since they're wrapper-owned per-iteration state, not runtime source?
- Test consolidation: flat `tests/` dir, or `autoresearch/tests/` + `splice/tests/` per-package? The latter matches package ownership; the former is one-stop for `pytest` invocation.
- Package name: `autoresearch/` (matches repo name's prefix, clear intent), `runtime/` (clear role), `harness/` (clear role)? The spec's prose assumes `autoresearch/` but the planner can override.
- App dir name: `splice/` (matches repo name's suffix), `splice_detection/` (verbose but unambiguous), `app/` (cheap but vague)?

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|---|---|---|---|
| Runtime | package | logger, log_reader, log_cli, verify_agent, preflight, baseline_metrics, manifest | produces + reads LogEvent; supervises App |
| App (Splice) | package | detector, features, ml_eval, evaluate, dataset_registry, classifier | writes LogEvent; consumed by Runtime; PROTECTED: evaluate + program.md |
| OperatorTool | package (`scripts/`) | dashboard, phase_stats, log_monitor, eval_crypto, test_crypto, analysis-scripts | reads LogEvent + results.tsv + SHAP artifacts; not imported by Runtime or App |
| State | dir (`.omc/`) | logs/, plans/, specs/, research_notes.md, last_eval.log, last_reflection.md, retest artifacts, feature_cache/, sentinels | written by Runtime; read by OperatorTool; never contains source code |

## Interview Transcript
<details>
<summary>4 rounds</summary>

### Round 1 — Primary pain
**Q:** What's the *primary* pain US-516 is solving? (a) navigation / (b) agent write-scope isolation / (c) package-ification / (d) extractability / (e) all / (f) free-form.
**A:** (a) mostly, (b) not so important but preferable, (c) never thought about but I think it's a good idea, (d) yes, I want to make this my own ml research harness.
**Ambiguity:** 66% → next: Goal (what IS the harness?)

### Round 2 — Harness scope (ontology-stress)
**Q:** What would you copy to a new detection problem? (a) loop-only / (b) loop+classifier / (c) loop+classifier+features / (d) whole framework / (e) "draw the line at non-audio-specific" / (f) free-form.
**A:** (a). "I'd actually say I'd lean more towards (a) in the previous question than (d). I can't understand the code because the project structure is a mess."
**Clarification:** navigation is the primary pain, harness is the long-term direction. Scope = loop-only.
**Ambiguity:** 60% → next: Success

### Round 3 — Done signal
**Q:** How would you know the reorg is done? (a) ls test / (b) 30s locate / (c) import-clean / (d) runtime/app boundary / (e) a+b only / (f) free-form.
**A:** Should be all a–d.
**Ambiguity:** 31% → next: Constraints

### Round 4 — Constraint batch (4 questions)
**Q1 (protected-file moves):** (i) yes, one-time exemption / (ii) no, stay at root / (iii) shim both.
**A:** (i).
**Q2 (git history):** (i) critical / (ii) nice / (iii) don't care.
**A:** (iii).
**Q3 (external refs audit):** (i) all in-PR / (ii) repo only / (iii) unknown, find them first.
**A:** (iii) → resolved by the scan enumerated above.
**Q4 (timing):** (i) wait N iterations / (ii) immediately after phase-2 / (iii) wait for retest.
**A:** (ii) → with loop-safety caveat resolved by stopping the loop for US-516 duration.
**Ambiguity:** 15% → threshold met.
</details>

## Execution Notes (not part of the spec body; pipeline handoff)

**Next stage:** ralplan consensus plan on top of this spec. The planner/architect/critic should resolve the four open structural questions above and produce a migration sequence that:

1. Stops the autoresearch loop (user already did this pre-execution).
2. Lands the rename + import-path rewrite + doc updates in commits that individually keep the `validate_logs --audit` + smoke + retest gates green (ordering: infra package first with shim re-exports, then app migration, then shim removal + final cleanup — similar in spirit to US-515's two-commit landing, likely 3–5 commits here).
3. Updates `verify_agent.py`'s PROTECTED path regex inside the same landing so the first post-restart iteration's diff audit finds the correct locations.
4. Restarts the loop as the final acceptance step and observes ≥ 1 clean iteration on the new layout before declaring done.

**Scope controls for ralplan:**
- No new features, no new tests beyond what's needed to keep gates green post-move.
- No backwards-compat shims beyond the migration window. If a shim is added in commit N to keep tests green, it's deleted by commit N+1 (or the final commit).
- Maintainer exemption on `evaluate.py` + `program.md` is one-shot — the commit that moves them must include the verify_agent regex update.
- Loop-safety: no manual commits on the reorg branch after the loop is restarted.
