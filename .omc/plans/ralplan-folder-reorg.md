# RALPLAN: US-516 Folder Reorg — Runtime/App Separation + Package-ification

**Spec:** `.omc/specs/deep-interview-folder-reorg.md`
**Branch:** `autoresearch/apr15` (HEAD: 3baa6c8)
**Date:** 2026-04-18

## Revision log

| Rev | Date | Summary |
|-----|------|---------|
| 1 | 2026-04-18 | Initial plan |
| 2 | 2026-04-18 | Address 3 Architect MUST-FIX items: (1) REPO_ROOT parent-chain off-by-one audit with per-file delta table, (2) empirical import probe proving `uv run python <file>` fails for cross-package imports -- PYTHONPATH fix baked in, (3) `validate_logs.py:_MIGRATED_FILES` update timing clarified: commit 3 only, with explicit commit-boundary path validity table |
| 3 | 2026-04-18 | Address Critic ITERATE: (MUST-FIX) gate (a) reclassified -- literal <=10 is aspirational, enforceable sub-criteria with runnable checks replace it; (SHOULD-FIX) gate (a) runnable commands per sub-criterion, gate (c) switched to `git grep`, gate (d) reproduction recipe added, PYTHONPATH deletion regression test added to commits 2-3, all commit-boundary verification commands replaced with copy-paste shell blocks; (NIT) `fp_filter.py` placement override documented, loop-safety invariants made checkable |

---

## Principles

1. **Gate-green invariant.** Every commit boundary leaves all regression gates passing. No "temporarily broken" intermediate states.
2. **Zero sys.path.insert in tracked code.** The import-clean gate (c) is the hard deliverable. Every Python module uses proper package imports post-migration.
3. **Protected-file integrity.** `evaluate.py` and `program.md` move under a one-time maintainer exemption; `verify_agent.py`'s PROTECTED_FILES list updates atomically in the same commit.
4. **Loop-safety first.** The autoresearch loop is stopped before any commit lands. No manual commits after loop restart. Restart is the final acceptance gate.
5. **Minimal scope.** Pure structural + import-path changes. Zero behavior changes, zero new features, zero new tests beyond what gates require.

## Decision Drivers

1. **Gates must be green at every commit boundary.** The six regression gates (validate_logs --audit, test_smoke_iteration, test_retest, test_logger, test_log_readers, train_classifier import) plus py_compile on all tracked .py files constrain commit ordering: infrastructure (packages + __init__.py) must land before consumers move.
2. **Loop will restart on this tree.** Post-merge, `run_autoresearch.sh start` must resolve all new paths. Every Python invocation site in the shell wrapper (`uv run python ...`) and every PROTECTED_FILES entry must point to the new locations.
3. **`evaluate.py` is invoked by the wrapper as `uv run python evaluate.py`.** After the move to `splice/evaluate.py`, the wrapper's invocation line must change. Because `evaluate.py` imports `from detector import detect_splices` (a sibling in the same package), the invocation must use `uv run python splice/evaluate.py` with `PYTHONPATH` set to the repo root (see empirical proof in Appendix A). `uv run python -m splice.evaluate` also works but would require changes to every invocation site's argument passing.

## Viable Options

### Option A: Infra-first with atomic migration (CHOSEN)

Create `autoresearch/` and `splice/` packages with `__init__.py` in commit 1; move+rewrite all files in commit 2 (runtime) and commit 3 (app); final cleanup + doc updates in commit 4.

**Pros:** Each commit is self-contained. Gates stay green because infra exists before consumers reference it. No shims needed -- imports are rewritten in the same commit as the file move.
**Cons:** Commit 2-3 touch many files simultaneously. Large diffs but mechanically simple (find-and-replace imports).

### Option B: Shim re-exports + gradual migration

Create packages with shim modules that re-export from old locations (e.g., `autoresearch/logger.py` re-exports from `.omc/coordination/logger.py`). Migrate consumers in small commits. Remove shims last.

**Pros:** Smaller commits, easier to review. **Cons:** Shim lifecycle adds complexity. Doubles the import surface during migration. Risk of shims outliving the migration (spec explicitly forbids this). More commits means more gate-check cycles. **INVALIDATED:** The spec mandates no backwards-compat shims that outlive the migration, and the "don't care about git history" decision removes the only reason to do rename-only commits. Bundling rename+content is explicitly authorized.

### Option C: Feature-branch worktree then merge

Do the entire reorg in a separate worktree / branch, then merge back.

**Pros:** Clean separation of reorg work from the live branch. **Cons:** The loop runs on `autoresearch/apr15` -- a separate branch means cherry-picking hypotheses across the boundary. Merge conflicts are guaranteed given the number of files touched. The spec says "land immediately" not "land after branch merge". **INVALIDATED:** Unnecessary complexity for a stopped-loop migration. The branch IS the worktree.

## ADR

**Decision:** Option A -- infra-first atomic migration in 4 commits.

**Drivers:** Gate-green invariant requires packages to exist before consumers reference them. No-shim constraint rules out Option B. Same-branch requirement rules out Option C.

**Alternatives considered:** B (shim re-exports), C (worktree merge). Both invalidated above.

**Why chosen:** Simplest path that satisfies all four acceptance gates (ls test, 30s locate, import-clean, fork-friendly) while keeping regression gates green at every boundary. Matches US-515's two-commit landing pattern scaled to ~4 commits.

**Consequences:** Large diffs in commits 2-3 (20+ files each). Acceptable because changes are mechanical (path rewrites) not behavioral. `git log --follow` will track moves if needed. Requires `PYTHONPATH` export in `run_autoresearch.sh` (one line) to make direct file invocation work across packages.

**Follow-ups:** None required by this PR. Future phases (3a/3b/3c from US-515) and pyproject entry points are explicitly out of scope.

---

## Open Questions Resolution

### 1. Runtime package name: `autoresearch/`

**Rationale:** Matches the repo name prefix. `runtime/` is too generic (what runtime?). `harness/` is unclear to newcomers. The spec's prose already uses `autoresearch/` throughout. The fork-friendly gate (d) tests `from autoresearch.logger import get_logger` -- the package name IS the brand.

### 2. App package name: `splice/`

**Rationale:** Matches the repo name suffix. Short, unambiguous in this context. `splice_detection/` is verbose. `app/` is too generic and fails the 30s-locate test (what app?).

### 3. `fp_classifier.joblib` placement: `splice/classifier/fp_classifier.joblib`

**Rationale:** The trained model is part of the app's behavior, not ephemeral runtime state. It is regenerated by `train_classifier.py` (also in `splice/classifier/`), referenced by `detector.py` via a path relative to `__file__` (currently `os.path.join(os.path.dirname(__file__), ".omc", "classifier", "fp_classifier.joblib")`). Keeping it next to its producer (`train_classifier.py`) and consumer (`detector.py`'s `_GBM_MODEL_PATH`) is the simplest path. If `.omc/` state dirs get wiped, the model should NOT disappear -- it gates whether detection works at all.

### 4. `baseline_metrics.json` + `manifest.json`: stay in `autoresearch/`

**Rationale:** Both are runtime-owned. `baseline_metrics.json` is read/written by `verify_agent.py` (runtime). `manifest.json` is read by `preflight.py` (runtime). They are config/state for the loop harness, not the splice app. They move with the runtime package.

### 5. Test layout: per-package (`autoresearch/tests/`, `splice/tests/`)

**Rationale:** Tests live next to the code they test. `pytest autoresearch/tests/` and `pytest splice/tests/` are clear invocations. The existing test files already live under `.omc/coordination/tests/` -- they map naturally to `autoresearch/tests/`. No splice-specific tests exist today, so `splice/tests/` starts empty (with `__init__.py` for future use). `conftest.py` at repo root handles path setup for `uv run pytest`.

---

## MUST-FIX #1: `REPO_ROOT` Parent-Chain Audit

### The bug

`.omc/coordination/verify_agent.py:17-18`:
```python
SCRIPT_DIR = Path(__file__).parent          # .omc/coordination/
REPO_ROOT = SCRIPT_DIR.parent.parent        # .omc/ -> repo root (2 .parent)
```

After moving to `autoresearch/verify_agent.py`, `SCRIPT_DIR.parent` = repo root (1 level up). `SCRIPT_DIR.parent.parent` would go ONE LEVEL ABOVE the repo root. Every path derived from REPO_ROOT (BASELINE_PATH, git commands, retest worktree paths, PROTECTED_FILES resolution) breaks silently.

**Fix:** `REPO_ROOT = SCRIPT_DIR.parent` (ONE `.parent`, not two).

### Commit-level gate (added to commit 2)

After the runtime migration commit, run:
```bash
uv run python -c "from autoresearch.verify_agent import REPO_ROOT; assert REPO_ROOT.name == 'autoresearch-splice', f'REPO_ROOT wrong: {REPO_ROOT}'"
```
This invariant check MUST pass before proceeding to commit 3.

### Full parent-chain audit table

Every file that uses `Path(__file__).parent*`, `_HERE.parents[N]`, or `os.path.dirname(os.path.dirname(...))` to derive a repo-root or sibling-directory reference:

| File (current location) | Current chain | Current result | New location | New chain needed | Delta |
|---|---|---|---|---|---|
| `.omc/coordination/verify_agent.py` | `Path(__file__).parent.parent.parent` (actually `.parent` then `REPO_ROOT = SCRIPT_DIR.parent.parent`) | `.omc/coordination/` -> `.omc/` -> repo root (2 `.parent` on SCRIPT_DIR) | `autoresearch/verify_agent.py` | `REPO_ROOT = SCRIPT_DIR.parent` (1 `.parent`) | **CHANGE: remove one `.parent`** |
| `.omc/coordination/preflight.py` | `SCRIPT_DIR = Path(__file__).parent` then `MANIFEST_PATH = SCRIPT_DIR / "manifest.json"` | `.omc/coordination/manifest.json` (sibling ref, no repo-root derivation) | `autoresearch/preflight.py` | `MANIFEST_PATH = SCRIPT_DIR / "manifest.json"` (unchanged -- manifest.json moves to same directory) | **NO CHANGE** (sibling ref stays valid) |
| `.omc/coordination/fp_filter.py` | `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` then `/ "classifier"` | `dirname(.omc/coordination/) = .omc/` then `.omc/classifier/fp_classifier.joblib` | `autoresearch/fp_filter.py` | `dirname(autoresearch/) = repo_root/` then `repo_root/classifier/` -- **WRONG**, joblib moves to `splice/classifier/` | **CHANGE: rewrite to use REPO_ROOT / "splice" / "classifier" / "fp_classifier.joblib"** or absolute path via `__file__`-relative chain `Path(__file__).parent.parent / "splice" / "classifier" / "fp_classifier.joblib"` |
| `.omc/coordination/log_cli.py` | `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` (adds own dir to path) | Adds `.omc/coordination/` so `from logger import get_logger` works | `autoresearch/log_cli.py` | Remove sys.path.insert entirely; use `from autoresearch.logger import get_logger` | **CHANGE: delete sys.path.insert, rewrite import** |
| `.omc/coordination/logger.py` | No parent-chain derivation (only has docstring mentioning `sys.path.insert` as usage convention) | N/A | `autoresearch/logger.py` | N/A -- update docstring only | **NO CHANGE to logic** |
| `.omc/classifier/shap_report.py` | `_HERE = Path(__file__).resolve()` then `_ROOT = _HERE.parents[2]` | `.omc/classifier/shap_report.py` -> parents[0]=`.omc/classifier/`, parents[1]=`.omc/`, parents[2]=repo root | `splice/classifier/shap_report.py` | `splice/classifier/shap_report.py` -> parents[0]=`splice/classifier/`, parents[1]=`splice/`, parents[2]=repo root | **NO CHANGE** (still 2 levels deep, `parents[2]` still = repo root). But the `sys.path.insert` and `from logger import get_logger` must change to `from autoresearch.logger import get_logger`. The `_ROOT` variable itself remains correct. |
| `.omc/classifier/train_classifier.py` | `_HERE = Path(__file__).resolve()` then `_ROOT = _HERE.parents[2]` | Same as shap_report: parents[2] = repo root | `splice/classifier/train_classifier.py` | Same depth (2 levels). `parents[2]` still = repo root | **NO CHANGE** to `_ROOT`. But `_features_path = _HERE.parent.parent / "features.py"` (line 384) currently resolves to `.omc/features.py` -- wait, let me re-check. `_HERE.parent` = `.omc/classifier/` -> `.parent` = `.omc/`. `.omc/features.py` doesn't exist; the actual `features.py` is at root. **BUG IN CURRENT CODE?** No: `_HERE` is resolved. `_HERE.parent.parent` = resolved path of `.omc/` parent = repo root. So `_HERE.parent.parent / "features.py"` = `repo_root/features.py`. After move to `splice/classifier/`, `_HERE.parent.parent` = `splice/` parent = repo root. Still correct. But `features.py` moves to `splice/features.py`! So this path becomes `repo_root/features.py` which no longer exists. **CHANGE: `_features_path = _ROOT / "splice" / "features.py"`** or `_HERE.parent / "features.py"` (since features.py will be a sibling of `splice/classifier/`'s parent = `splice/`). |
| `.omc/coordination/fp_filter.py` (also uses) | `os.path.dirname(os.path.dirname(__file__))` / `"classifier"` | As above | As above | As above | As above |

### Summary of required changes by file

| File (new location) | Parent-chain fix needed | Import fix needed |
|---|---|---|
| `autoresearch/verify_agent.py` | YES: `REPO_ROOT = SCRIPT_DIR.parent` (remove one `.parent`) | YES: remove sys.path.insert |
| `autoresearch/preflight.py` | NO (sibling ref) | NO (no sys.path.insert, no cross-module imports beyond manifest.json) |
| `autoresearch/fp_filter.py` | YES: rewrite `_MODEL_DIR` to `os.path.join(REPO_ROOT, "splice", "classifier")` where `REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` ... wait, after move `dirname(dirname(autoresearch/fp_filter.py))` = repo root for resolved paths. Actually simpler: `_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splice", "classifier", "fp_classifier.joblib")` | NO sys.path.insert to remove |
| `autoresearch/log_cli.py` | NO parent-chain issue | YES: remove sys.path.insert, use `from autoresearch.logger import get_logger` |
| `autoresearch/logger.py` | NO | NO (docstring update only) |
| `splice/classifier/shap_report.py` | NO (`parents[2]` still correct) | YES: remove sys.path.insert, use `from autoresearch.logger import get_logger` |
| `splice/classifier/train_classifier.py` | YES: `_features_path` (line 384) must change from `_HERE.parent.parent / "features.py"` to `_ROOT / "splice" / "features.py"` (or `_HERE.parent.parent / "features.py"` which after move = `splice/features.py` -- actually `_HERE.parent.parent` = repo root, so NO, must use `_ROOT / "splice" / "features.py"`) | YES: remove sys.path.insert, rewrite `from dataset_registry import ...` to `from splice.dataset_registry import ...` etc. |

**Wait -- re-checking `train_classifier.py` line 384 more carefully.**

Current: `_HERE = Path(__file__).resolve()` where `__file__` = `.omc/classifier/train_classifier.py`. Resolved: `/Users/.../autoresearch-splice/.omc/classifier/train_classifier.py`. `_HERE.parent.parent` = `/Users/.../autoresearch-splice/.omc/` -> wait, `.parent` = `.omc/classifier/` no. `_HERE.parent` = `/Users/.../autoresearch-splice/.omc/classifier/`. `_HERE.parent.parent` = `/Users/.../autoresearch-splice/.omc/`. That's `.omc/`, NOT repo root. So `_HERE.parent.parent / "features.py"` = `.omc/features.py` which doesn't exist.

But `_ROOT = _HERE.parents[2]` = repo root. So there's already an inconsistency -- `_HERE.parent.parent` != `_ROOT`. The `_features_path` line is currently broken or never exercised. Let me check if it's actually used:

It's on line 384 inside a block that computes a git hash for features.py. If features.py doesn't exist at `.omc/features.py`, `git hash-object` fails and the try/except catches it. So it's a best-effort feature tracking that silently fails.

After move: `_HERE.parent.parent` = `splice/` (NOT repo root). `splice/features.py` DOES exist. So the line actually STARTS WORKING correctly by accident. But for correctness, it should use `_ROOT / "splice" / "features.py"`. Either way, the code is not broken by the move -- it gets better. Still, fix it to use the explicit correct path.

---

## MUST-FIX #2: Import Resolution — Empirical Proof

### Problem statement

The plan previously claimed `from autoresearch.X import Y` and `from splice.X import Y` resolve via `uv run python <file>` without `sys.path.insert`. The Architect was skeptical because `pyproject.toml` has no `[tool.setuptools.packages.find]` directive.

### Empirical test (conducted 2026-04-18)

**Setup:** Created temporary scaffolding in the repo root (not committed):
- `autoresearch/__init__.py` (empty)
- `autoresearch/logger.py` (copy of `.omc/coordination/logger.py`)
- `splice/__init__.py` (empty)
- `splice/evaluate_probe.py` (minimal: `from autoresearch.logger import get_logger; print("OK")`)

**Test 1: `uv run python splice/evaluate_probe.py`**
```
$ uv run python splice/evaluate_probe.py
Traceback (most recent call last):
  File ".../splice/evaluate_probe.py", line 1, in <module>
    from autoresearch.logger import get_logger
ModuleNotFoundError: No module named 'autoresearch'
EXIT=1
```

**Root cause:** `uv run python <file>` puts the FILE'S DIRECTORY on `sys.path[0]`, not the repo root:
```
sys.path[:5]: ['/Users/.../autoresearch-splice/splice', ...]
```
So `splice/` is on path, but the repo root is not. `from autoresearch.logger import ...` fails because Python looks for `splice/autoresearch/logger.py`.

**Test 2: `uv run python -m splice.evaluate_probe`**
```
$ uv run python -m splice.evaluate_probe
OK - direct import works
EXIT=0
```
**Why:** `-m` puts `''` (CWD = repo root) on `sys.path[0]`.

**Test 3: `uv run python -c "from autoresearch.logger import get_logger; print('ok')"`**
```
$ uv run python -c "from autoresearch.logger import get_logger; print('ok')"
ok
EXIT=0
```
**Why:** `-c` also puts `''` on `sys.path[0]`.

**Test 4: `PYTHONPATH=<repo_root> uv run python splice/evaluate_probe.py`**
```
$ PYTHONPATH=/Users/.../autoresearch-splice uv run python splice/evaluate_probe.py
OK - direct import works
EXIT=0
```

### Resolution: PYTHONPATH in `run_autoresearch.sh`

**Chosen strategy:** Add one line near the top of `run_autoresearch.sh`:
```bash
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
```

**Why this over the alternatives:**

| Strategy | Works? | Invasiveness | Gate (c) clean? |
|---|---|---|---|
| `PYTHONPATH` in wrapper | YES | 1 line in wrapper | YES (no .py files touched) |
| Switch all to `-m` invocation | YES | ~15 invocation sites change, argument passing must be verified | YES |
| `[build-system]` + editable install | YES | pyproject.toml change + `uv pip install -e .` step | YES |
| `sys.path.insert` in each file | YES | N files | NO (fails gate c) |

`PYTHONPATH` is the lowest-risk single-line change. It does NOT violate gate (c) because gate (c) greps for `sys.path.insert` in tracked `.py` files, and `PYTHONPATH` is a shell environment variable in a `.sh` file.

**Impact on `run_autoresearch.sh`:** Add `export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"` after the `PROJECT_DIR` assignment (around line 5-10). This ensures every `uv run python <file>` invocation sees the repo root on `sys.path`, making cross-package imports (`from autoresearch.X import Y`, `from splice.X import Y`) work regardless of which subdirectory the invoked file lives in.

**Impact on standalone invocations (outside wrapper):** Operators running `uv run python splice/evaluate.py --shap` directly must either:
1. Run from the repo root with `PYTHONPATH=$PWD uv run python splice/evaluate.py --shap`, or
2. Use `uv run python -m splice.evaluate --shap`.

This is documented in the CLAUDE.md update (commit 4).

---

## MUST-FIX #3: `validate_logs.py:_MIGRATED_FILES` Update Timing

### Problem statement

The rev-1 plan was confused about whether `validate_logs.py` updates in commit 2 or commit 3. The plan text contradicted itself.

### Resolution: crisp commit-boundary analysis

**`_MIGRATED_FILES` current value:**
```python
_MIGRATED_FILES = [
    REPO / "detector.py",
    REPO / "features.py",
    REPO / "ml_eval.py",
    REPO / ".omc" / "classifier" / "shap_report.py",
    REPO / ".omc" / "classifier" / "train_classifier.py",
]
```

**What moves when:**
- `detector.py`, `features.py`, `ml_eval.py` (root files) move in **commit 3** (app migration).
- `shap_report.py`, `train_classifier.py` (`.omc/classifier/`) also move in **commit 3** (app migration).
- NO `_MIGRATED_FILES` entries move in commit 2. Commit 2 only moves runtime files (`.omc/coordination/` -> `autoresearch/`).

**Commit-boundary path validity table:**

| Entry | After commit 1 | After commit 2 | After commit 3 |
|---|---|---|---|
| `REPO / "detector.py"` | EXISTS (root) | EXISTS (root, untouched) | GONE (moved to `splice/detector.py`) |
| `REPO / "features.py"` | EXISTS (root) | EXISTS (root, untouched) | GONE (moved to `splice/features.py`) |
| `REPO / "ml_eval.py"` | EXISTS (root) | EXISTS (root, untouched) | GONE (moved to `splice/ml_eval.py`) |
| `REPO / ".omc/classifier/shap_report.py"` | EXISTS | EXISTS (untouched) | GONE (moved to `splice/classifier/shap_report.py`) |
| `REPO / ".omc/classifier/train_classifier.py"` | EXISTS | EXISTS (untouched) | GONE (moved to `splice/classifier/train_classifier.py`) |

**Conclusion:** `validate_logs.py:_MIGRATED_FILES` updates in **commit 3 ONLY**. At the commit 2 boundary, all `_MIGRATED_FILES` paths still point to existing files, so `validate_logs --audit` passes without any change to `validate_logs.py`.

**`_MIGRATED_FILES` final value (commit 3):**
```python
_MIGRATED_FILES = [
    REPO / "splice" / "detector.py",
    REPO / "splice" / "features.py",
    REPO / "splice" / "ml_eval.py",
    REPO / "splice" / "classifier" / "shap_report.py",
    REPO / "splice" / "classifier" / "train_classifier.py",
]
```

---

## Final Directory Structure

```
autoresearch-splice/
├── autoresearch/                    # RUNTIME PACKAGE — reusable loop harness
│   ├── __init__.py                  # empty; marks package
│   ├── logger.py                    # from .omc/coordination/logger.py
│   ├── log_cli.py                   # from .omc/coordination/log_cli.py
│   ├── log_reader.py                # from scripts/log_reader.py
│   ├── verify_agent.py              # from .omc/coordination/verify_agent.py
│   ├── preflight.py                 # from .omc/coordination/preflight.py
│   ├── fp_filter.py                 # from .omc/coordination/fp_filter.py (spec places in splice/; overridden -- see NIT #7 note below)
│   ├── baseline_metrics.json        # from .omc/coordination/baseline_metrics.json
│   ├── manifest.json                # from .omc/coordination/manifest.json
│   └── tests/
│       ├── __init__.py
│       ├── fixtures/
│       │   └── mixed_log_sample/
│       │       └── autoresearch.jsonl
│       ├── test_logger.py           # from .omc/coordination/tests/test_logger.py
│       ├── test_log_readers.py      # from .omc/coordination/tests/test_log_readers.py
│       ├── test_retest.py           # from .omc/coordination/tests/test_retest.py
│       └── test_smoke_iteration.py  # from .omc/coordination/tests/test_smoke_iteration.py
├── splice/                          # APPLICATION PACKAGE — audio-splice-specific
│   ├── __init__.py                  # empty; marks package
│   ├── detector.py                  # from root detector.py
│   ├── features.py                  # from root features.py
│   ├── ml_eval.py                   # from root ml_eval.py
│   ├── evaluate.py                  # from root evaluate.py (PROTECTED, one-time exemption)
│   ├── dataset_registry.py          # from root dataset_registry.py
│   └── classifier/
│       ├── __init__.py
│       ├── train_classifier.py      # from .omc/classifier/train_classifier.py
│       ├── shap_report.py           # from .omc/classifier/shap_report.py
│       ├── fp_classifier.joblib     # from .omc/classifier/fp_classifier.joblib
│       ├── fp_classifier.meta.json  # from .omc/classifier/fp_classifier.meta.json
│       ├── cv_results.json          # from .omc/classifier/cv_results.json
│       └── versions.json            # from .omc/classifier/versions.json
├── scripts/                         # OPERATOR TOOLS — unchanged purpose
│   ├── __init__.py                  # NEW — enables `from scripts.splice_boundary import ...`
│   ├── validate_logs.py             # stays
│   ├── dashboard.py                 # stays
│   ├── phase_stats.py               # stays
│   ├── log_monitor.py               # untracked, stays
│   ├── eval_crypto.py               # stays
│   ├── test_crypto.py               # stays
│   ├── tunable_frontier.py          # stays
│   ├── diagnose_repeat_rate.py      # stays
│   ├── notebook_digest.py           # stays
│   ├── shap_rollup.py               # stays
│   ├── shap_shift.py                # stays
│   ├── splice_boundary.py           # stays (imported by data_synth only)
│   └── test_unlock.swift            # stays
├── data/                            # UNCHANGED
├── data_synth/                      # UNCHANGED (path ref to scripts/ updated)
├── reports/                         # UNCHANGED
├── .omc/                            # PURE STATE — zero .py files
│   ├── logs/                        # JSONL log + child-stderr
│   ├── plans/                       # planning artifacts (frozen)
│   ├── specs/                       # interview specs (frozen)
│   ├── state/                       # sessions
│   ├── research_notes.md
│   ├── last_eval.log
│   ├── last_reflection.md
│   ├── feature_cache/               # per-sha feature caches
│   ├── retest-worktree/
│   ├── retest-in-progress           # sentinel
│   ├── retest-report.md
│   ├── autoresearch-stop            # loop sentinel
│   ├── shap_rollup.json
│   ├── evaluate_integration.md      # historical doc
│   ├── autoresearch_progress.png
│   ├── progress.txt
│   ├── prd.json                     # frozen
│   ├── experiments/                  # energy_dip_test.py stays (scratch, not imported)
│   └── classifier/                  # SNAPSHOTS ONLY — detector_v*.py stay here
│       └── detector_v{1..23}.py     # untracked snapshots (not moved)
├── run_autoresearch.sh              # stays at root
├── README.md                        # stays at root
├── CLAUDE.md                        # stays at root
├── program.md                       # MOVED to splice/program.md (one-time exemption)
├── pyproject.toml                   # stays (no entry points added)
├── uv.lock                          # stays
├── results.tsv                      # stays
├── analysis.ipynb                   # stays
├── progress.png                     # stays
├── progress.txt                     # stays
└── .gitignore                       # stays
```

**NIT #7: `fp_filter.py` placement override.** The spec (line 158) places `fp_filter.py` in `splice/classifier/`. This plan overrides that to `autoresearch/fp_filter.py` as a **conservative default for an orphaned file** — grep confirms `fp_filter` is NOT currently imported by `verify_agent.py`, `detector.py`, or any live runtime/app path; it is dormant infrastructure from an earlier SHAP-export iteration. Keeping it beside `verify_agent.py` (its historical sibling in `.omc/coordination/`) preserves the post-move model-path fix (`_MODEL_DIR = splice/classifier/`) without tying it to the app package. If `fp_filter` is reactivated in a future PR, its placement should be revisited — `splice/classifier/` is the correct long-term home since it operates on audio detections, not loop-harness state. The fork-friendly gate (d) is unaffected because the file is currently unused.

**`scripts/log_reader.py` -> `autoresearch/log_reader.py`.** This is imported by runtime (`verify_agent.py`) and by operator tools (`dashboard.py`, `phase_stats.py`). It's a runtime library, not an operator tool. Scripts that import it use `from autoresearch.log_reader import iter_events`.

**Files deleted from old locations (net effect of moves):**
- `.omc/coordination/logger.py` -> `autoresearch/logger.py`
- `.omc/coordination/log_cli.py` -> `autoresearch/log_cli.py`
- `.omc/coordination/verify_agent.py` -> `autoresearch/verify_agent.py`
- `.omc/coordination/preflight.py` -> `autoresearch/preflight.py`
- `.omc/coordination/fp_filter.py` -> `autoresearch/fp_filter.py`
- `.omc/coordination/baseline_metrics.json` -> `autoresearch/baseline_metrics.json`
- `.omc/coordination/manifest.json` -> `autoresearch/manifest.json`
- `.omc/coordination/tests/*` -> `autoresearch/tests/*`
- `.omc/classifier/train_classifier.py` -> `splice/classifier/train_classifier.py`
- `.omc/classifier/shap_report.py` -> `splice/classifier/shap_report.py`
- `.omc/classifier/fp_classifier.joblib` -> `splice/classifier/fp_classifier.joblib`
- `.omc/classifier/fp_classifier.meta.json` -> `splice/classifier/fp_classifier.meta.json`
- `.omc/classifier/cv_results.json` -> `splice/classifier/cv_results.json`
- `.omc/classifier/versions.json` -> `splice/classifier/versions.json`
- `scripts/log_reader.py` -> `autoresearch/log_reader.py`
- Root `detector.py` -> `splice/detector.py`
- Root `features.py` -> `splice/features.py`
- Root `ml_eval.py` -> `splice/ml_eval.py`
- Root `evaluate.py` -> `splice/evaluate.py`
- Root `dataset_registry.py` -> `splice/dataset_registry.py`
- Root `program.md` -> `splice/program.md`

---

## Commit Sequence

### Commit 0: Pre-flight — stop loop + verify baseline

**Not a git commit.** Operator action: `./run_autoresearch.sh stop` and verify `.omc/autoresearch-stop` sentinel exists. Run all 6 regression gates to establish baseline green state.

---

### Commit 1: `us516: create package scaffolding`

**Files created (new):**
| File | Lines | Purpose |
|---|---|---|
| `autoresearch/__init__.py` | 1 | Package marker |
| `autoresearch/tests/__init__.py` | 1 | Test package marker |
| `splice/__init__.py` | 1 | Package marker |
| `splice/classifier/__init__.py` | 1 | Package marker |
| `splice/tests/__init__.py` | 1 | Future test placeholder |
| `scripts/__init__.py` | 1 | Enables `from scripts.splice_boundary import ...` |

**Files touched:** None existing.

**Note on conftest.py:** NOT created. `uv run pytest` with `PYTHONPATH` set (or `-m` invocation) handles package discovery. If tests fail without it during execution, the executor adds it as a gate-green fix with a comment explaining the exception.

**Commit-1 verification commands:**
```bash
# Package markers exist
test -f autoresearch/__init__.py && test -f splice/__init__.py && test -f splice/classifier/__init__.py && test -f scripts/__init__.py && echo "PASS: scaffolding" || echo "FAIL"

# Regression gates -- unchanged source, must still pass
uv run python scripts/validate_logs.py --audit
uv run python .omc/coordination/tests/test_smoke_iteration.py
uv run pytest .omc/coordination/tests/test_retest.py -v
uv run pytest .omc/coordination/tests/test_logger.py .omc/coordination/tests/test_log_readers.py -v

# py_compile new files
uv run python -m py_compile autoresearch/__init__.py
uv run python -m py_compile splice/__init__.py

# Loop sentinel still present
test -f .omc/autoresearch-stop && echo "PASS: sentinel present" || echo "FAIL: loop sentinel missing"
```

**Gate status after commit 1:**
- All 6 regression gates: GREEN (no source changes)
- py_compile: GREEN (new files are trivial)

---

### Commit 2: `us516: migrate runtime package (autoresearch/) + PYTHONPATH fix + update consumers`

**Files moved + rewritten:**
| File | From | Key changes |
|---|---|---|
| `autoresearch/logger.py` | `.omc/coordination/logger.py` | Remove docstring sys.path.insert instructions; update usage example |
| `autoresearch/log_reader.py` | `scripts/log_reader.py` | Update internal path refs |
| `autoresearch/log_cli.py` | `.omc/coordination/log_cli.py` | **Remove `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`**; change `from logger import get_logger` to `from autoresearch.logger import get_logger` |
| `autoresearch/verify_agent.py` | `.omc/coordination/verify_agent.py` | **FIX: `REPO_ROOT = SCRIPT_DIR.parent` (ONE `.parent`)**; `BASELINE_PATH = SCRIPT_DIR / "baseline_metrics.json"` (unchanged, sibling ref); update PROTECTED_FILES partially (see below) |
| `autoresearch/preflight.py` | `.omc/coordination/preflight.py` | No parent-chain change needed (sibling ref to `manifest.json` stays valid) |
| `autoresearch/fp_filter.py` | `.omc/coordination/fp_filter.py` | **FIX: `_MODEL_DIR` must change from `dirname(dirname(__file__)) / "classifier"` to a REPO_ROOT-based path**: `_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (= repo root after move) is WRONG because `dirname(autoresearch/) = repo_root` so `dirname(dirname(fp_filter.py))` = `dirname(autoresearch/)` = repo root. Then join `"splice", "classifier"` for the model path. Full fix: `_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splice", "classifier")` |
| `autoresearch/baseline_metrics.json` | `.omc/coordination/baseline_metrics.json` | Pure move |
| `autoresearch/manifest.json` | `.omc/coordination/manifest.json` | Pure move |
| `autoresearch/tests/test_logger.py` | `.omc/coordination/tests/test_logger.py` | Remove sys.path.insert; `from autoresearch.logger import ...` |
| `autoresearch/tests/test_log_readers.py` | `.omc/coordination/tests/test_log_readers.py` | Remove sys.path.insert; `from autoresearch.log_reader import ...` |
| `autoresearch/tests/test_retest.py` | `.omc/coordination/tests/test_retest.py` | Remove sys.path.insert; update all internal path refs |
| `autoresearch/tests/test_smoke_iteration.py` | `.omc/coordination/tests/test_smoke_iteration.py` | Remove sys.path.insert; `import autoresearch.logger as lg` |
| `autoresearch/tests/fixtures/` | `.omc/coordination/tests/fixtures/` | Pure move |

**Consumer updates (files NOT moved, but imports rewritten):**
| File | Change |
|---|---|
| `scripts/dashboard.py` | Remove sys.path.insert; `from autoresearch.log_reader import iter_events` |
| `scripts/phase_stats.py` | Remove sys.path.insert; `from autoresearch.log_reader import iter_events` |
| `scripts/tunable_frontier.py` | Remove sys.path.insert; `from autoresearch.logger import get_logger` |
| `scripts/notebook_digest.py` | Remove sys.path.insert; `from autoresearch.logger import get_logger` |
| `run_autoresearch.sh` | **Add `export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"`** after PROJECT_DIR assignment; update runtime path refs (log_cli.py, verify_agent.py, baseline_metrics.json) |

**`validate_logs.py` is NOT touched in commit 2.** All `_MIGRATED_FILES` entries point to root files and `.omc/classifier/` files, NONE of which move in commit 2. The audit gate passes with the existing paths.

**PROTECTED_FILES partial update in verify_agent.py (commit 2):**
```python
PROTECTED_FILES = [
    "evaluate.py",           # still at root (moves in commit 3)
    "program.md",            # still at root (moves in commit 3)
    "data/eval/*",
    "data/test/*",
    "autoresearch/manifest.json",    # moved in this commit
    "autoresearch/preflight.py",     # moved in this commit
]
```

**Commit-2 verification commands (REQUIRED before proceeding):**
```bash
# REPO_ROOT invariant -- must resolve to the repo, not its parent
uv run python -c "from autoresearch.verify_agent import REPO_ROOT; assert REPO_ROOT.name == 'autoresearch-splice', f'REPO_ROOT wrong: {REPO_ROOT}'"

# PYTHONPATH export must be present in wrapper
grep -q 'export PYTHONPATH' run_autoresearch.sh || echo "FAIL: PYTHONPATH export missing"

# Regression gates
uv run python scripts/validate_logs.py --audit
PYTHONPATH=$PWD uv run pytest autoresearch/tests/ -v --co -q | wc -l  # expect >=14 tests collected
PYTHONPATH=$PWD uv run pytest autoresearch/tests/ -v
PYTHONPATH=$PWD uv run python -c "from autoresearch.logger import get_logger; print('OK')"
PYTHONPATH=$PWD uv run python -c "from autoresearch.log_reader import iter_events; print('OK')"

# Gate (a-i) partial -- no tracked .py should be in .omc/ yet (nothing moved there)
git ls-files -- '.omc/**/*.py' | grep -v 'detector_v' | head -1 | grep -q . && echo "FAIL: tracked .py in .omc/" || echo "PASS"

# py_compile all tracked .py
git ls-files -- '*.py' | xargs -I{} uv run python -m py_compile {}

# Loop sentinel still present
test -f .omc/autoresearch-stop && echo "PASS: sentinel present" || echo "FAIL: loop sentinel missing"
```

**Gate status after commit 2:**
- validate_logs --audit: GREEN (`_MIGRATED_FILES` still point to root + `.omc/classifier/` paths, all of which still exist)
- test_smoke_iteration: GREEN (updated imports resolve via PYTHONPATH)
- test_retest: GREEN
- test_logger: GREEN
- test_log_readers: GREEN
- train_classifier: GREEN (not touched yet; still at `.omc/classifier/`, still uses sys.path.insert to reach root modules which still exist at root)
- py_compile: GREEN

---

### Commit 3: `us516: migrate app package (splice/) + PROTECTED exemption + update wrapper + validate_logs`

**Files moved + rewritten:**
| File | From | Key changes |
|---|---|---|
| `splice/detector.py` | root `detector.py` | Remove sys.path.insert; `from autoresearch.logger import get_logger`; update `_GBM_MODEL_PATH` to `os.path.join(os.path.dirname(__file__), "classifier", "fp_classifier.joblib")` |
| `splice/features.py` | root `features.py` | Remove sys.path.insert; `from autoresearch.logger import get_logger` |
| `splice/ml_eval.py` | root `ml_eval.py` | Remove sys.path.insert; `from splice.detector import ...`; `from autoresearch.logger import get_logger` |
| `splice/evaluate.py` | root `evaluate.py` | PROTECTED one-time exemption. `from splice.detector import detect_splices`; `from autoresearch.logger import get_logger`; remove sys.path.insert |
| `splice/dataset_registry.py` | root `dataset_registry.py` | Minimal path updates if any |
| `splice/program.md` | root `program.md` | Update all internal path refs |
| `splice/classifier/train_classifier.py` | `.omc/classifier/train_classifier.py` | Remove both sys.path.insert calls; `from splice.dataset_registry import DATASETS`; `from splice.detector import _build_chunk_context`; `from splice.features import ...`; `from autoresearch.logger import get_logger`; **FIX: `_features_path = _ROOT / "splice" / "features.py"`** (line 384, was `_HERE.parent.parent / "features.py"` which after move would resolve to `repo_root/features.py` -- but features.py moves to `splice/features.py`); `_ROOT = _HERE.parents[2]` stays correct (still 2 levels deep) |
| `splice/classifier/shap_report.py` | `.omc/classifier/shap_report.py` | Remove both sys.path.insert calls (the coordination one and the root one); `from autoresearch.logger import get_logger`; `_ROOT = _HERE.parents[2]` stays correct (still 2 levels deep) |
| `splice/classifier/fp_classifier.joblib` | `.omc/classifier/fp_classifier.joblib` | Binary move |
| `splice/classifier/fp_classifier.meta.json` | `.omc/classifier/fp_classifier.meta.json` | Move |
| `splice/classifier/cv_results.json` | `.omc/classifier/cv_results.json` | Move |
| `splice/classifier/versions.json` | `.omc/classifier/versions.json` | Move |

**Consumer updates:**
| File | Change |
|---|---|
| `autoresearch/verify_agent.py` | PROTECTED_FILES final update (see below) |
| `scripts/validate_logs.py` | **Update `_MIGRATED_FILES` to final paths** (see below) |
| `run_autoresearch.sh` | All remaining path refs updated (evaluate.py, train_classifier.py, detector.py snapshots, versions.json, etc.) |
| `data_synth/regenerate_datasets.py` | Remove sys.path.insert; `from scripts.splice_boundary import find_splice_point` |

**`validate_logs.py:_MIGRATED_FILES` final value (commit 3):**
```python
_MIGRATED_FILES = [
    REPO / "splice" / "detector.py",
    REPO / "splice" / "features.py",
    REPO / "splice" / "ml_eval.py",
    REPO / "splice" / "classifier" / "shap_report.py",
    REPO / "splice" / "classifier" / "train_classifier.py",
]
```

**PROTECTED_FILES final state (commit 3):**
```python
PROTECTED_FILES = [
    "splice/evaluate.py",
    "splice/program.md",
    "data/eval/*",
    "data/test/*",
    "autoresearch/manifest.json",
    "autoresearch/preflight.py",
]
```

**Wrapper path updates (`run_autoresearch.sh`):**
| Line(s) | Old | New |
|---|---|---|
| L5-10 (new) | — | `export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"` (if not already added in commit 2) |
| L23 (_log helper) | `.omc/coordination/log_cli.py` | `autoresearch/log_cli.py` |
| L284 (snapshot) | `cp "$PROJECT_DIR/detector.py" "$SNAP"` | `cp "$PROJECT_DIR/splice/detector.py" "$SNAP"` |
| L291 (baseline read) | `.omc/coordination/baseline_metrics.json` | `autoresearch/baseline_metrics.json` |
| L348-349 (baseline commit) | `.omc/coordination/baseline_metrics.json` | `autoresearch/baseline_metrics.json` |
| L356 (versions.json) | `.omc/classifier/versions.json` | `splice/classifier/versions.json` |
| L406,913,996 (features.py sha) | `"$PROJECT_DIR/features.py"` | `"$PROJECT_DIR/splice/features.py"` |
| L410,917 (meta.json) | `.omc/classifier/fp_classifier.meta.json` | `splice/classifier/fp_classifier.meta.json` |
| L423-427,929-933 (retrain) | `.omc/classifier/train_classifier.py` | `splice/classifier/train_classifier.py` |
| L437-438,965-966 (git add joblib) | `.omc/classifier/fp_classifier.joblib`, `.omc/classifier/fp_classifier.meta.json` | `splice/classifier/fp_classifier.joblib`, `splice/classifier/fp_classifier.meta.json` |
| L601,604 (baseline read) | `.omc/coordination/baseline_metrics.json` | `autoresearch/baseline_metrics.json` |
| L944,1014,1030,1055,1077 (verify_agent) | `.omc/coordination/verify_agent.py` | `autoresearch/verify_agent.py` |
| L1005 (evaluate) | `uv run python evaluate.py --shap` | `uv run python splice/evaluate.py --shap` |
| L1180 (commit log) | `detector.py evaluate.py ml_eval.py` | `splice/detector.py splice/evaluate.py splice/ml_eval.py` |
| L1195-1196 (versions.json) | `.omc/classifier/versions.json` | `splice/classifier/versions.json` |
| L1218-1223 (rollback) | `.omc/classifier/detector_v...`, `detector.py`, `.omc/classifier/fp_classifier.joblib` | `splice/classifier/fp_classifier.joblib`, `splice/detector.py` (snapshots stay in `.omc/classifier/`) |
| L682+ (SYSTEM_PROMPT) | `detector.py`, `evaluate.py`, `features.py`, `.omc/classifier/train_classifier.py`, `.omc/coordination/verify_agent.py`, `.omc/coordination/manifest.json`, `.omc/coordination/preflight.py` | All updated to new locations |
| L739-743 (agent instructions) | Old paths | New paths |

**Note:** The PYTHONPATH export is added in commit 2 (runtime migration) because that's where the first cross-package imports appear. The wrapper path refs that reference `.omc/coordination/*` also update in commit 2. The wrapper path refs that reference root `.py` files and `.omc/classifier/*` update in commit 3.

**Commit-3 verification commands:**
```bash
# Regression gates
uv run python scripts/validate_logs.py --audit
PYTHONPATH=$PWD uv run pytest autoresearch/tests/ -v

# Cross-package imports resolve
PYTHONPATH=$PWD uv run python -c "from splice.detector import detect_splices; from splice.features import extract_features; print('OK')"
PYTHONPATH=$PWD uv run python -c "from splice.evaluate import main; print('imports OK')" 2>/dev/null || echo "evaluate has no main -- check manually"
PYTHONPATH=$PWD uv run python -c "import splice.classifier.train_classifier; print('imports OK')"

# PYTHONPATH export still present (regression guard)
grep -q 'export PYTHONPATH' run_autoresearch.sh || echo "FAIL: PYTHONPATH export missing"

# Gate (c) partial -- no sys.path.insert in tracked .py
git grep -n 'sys\.path\.insert' -- '*.py' | head -1 | grep -q . && echo "FAIL: sys.path.insert remains" || echo "PASS"

# py_compile all tracked .py
git ls-files -- '*.py' | xargs -I{} uv run python -m py_compile {}

# Loop sentinel still present
test -f .omc/autoresearch-stop && echo "PASS: sentinel present" || echo "FAIL: loop sentinel missing"
```

**Gate status after commit 3:**
- validate_logs --audit: GREEN (`_MIGRATED_FILES` updated to new paths; files exist at new paths)
- test_smoke_iteration: GREEN (updated in commit 2)
- test_retest: GREEN (updated in commit 2)
- test_logger: GREEN (updated in commit 2)
- test_log_readers: GREEN (updated in commit 2)
- train_classifier: GREEN (moved + import-rewritten; PYTHONPATH ensures `from splice.features import ...` resolves)
- py_compile: GREEN

---

### Commit 4: `us516: doc updates + cleanup`

**Files updated:**
| File | Change |
|---|---|
| `CLAUDE.md` | All path refs updated; add US-516 migration note; update "Unified logging" import example; update Protected Files section; document `PYTHONPATH` requirement for standalone invocations |
| `README.md` | Update `evaluate.py` refs |
| `.omc/evaluate_integration.md` | Update `verify_agent.py` path refs |
| `autoresearch/logger.py` | Replace import-convention docstring with one-liner: `from autoresearch.logger import get_logger` |

**Files deleted:** Git handles moves as delete+add within a single commit. The "move" in commits 2-3 already deletes the old file and creates the new one. Commit 4 is purely doc updates. If `.omc/coordination/` is left with zero tracked files after commit 3, git drops the directory automatically.

**Commit-4 verification commands (full acceptance gate battery):**
```bash
# All 6 regression gates
uv run python scripts/validate_logs.py --audit
PYTHONPATH=$PWD uv run pytest autoresearch/tests/ -v

# Gate (a-i): zero tracked .py under .omc/
git ls-files -- '.omc/**/*.py' | grep -v -E 'detector_v|experiments/' | head -1 | grep -q . && echo "FAIL: tracked .py in .omc/ (excluding detector_v* snapshots + experiments/ scratch)" || echo "PASS: (a-i)"

# Gate (a-ii): no runtime/app imports from scripts/
git grep -l 'from scripts\.\|import scripts\.' -- 'autoresearch/**/*.py' 'splice/**/*.py' | head -1 | grep -q . && echo "FAIL: runtime/app imports from scripts/" || echo "PASS: (a-ii)"

# Gate (c): zero sys.path.insert in tracked .py
git grep -n 'sys\.path\.insert' -- '*.py' | head -1 | grep -q . && echo "FAIL: sys.path.insert in tracked .py" || echo "PASS: (c)"

# Gate (d): fork-friendly reproduction
TMPDIR=$(mktemp -d) && cp -r . "$TMPDIR/us516-fork-test" && \
cd "$TMPDIR/us516-fork-test" && rm -rf splice/ && \
PYTHONPATH="$TMPDIR/us516-fork-test" uv run python -c "from autoresearch.logger import get_logger; print('OK')" && \
PYTHONPATH="$TMPDIR/us516-fork-test" uv run pytest autoresearch/tests/test_logger.py -v && \
echo "PASS: (d)" || echo "FAIL: (d)"
cd - && rm -rf "$TMPDIR/us516-fork-test"

# PYTHONPATH regression guard
grep -q 'export PYTHONPATH' run_autoresearch.sh || echo "FAIL: PYTHONPATH export missing"

# Cross-package imports
PYTHONPATH=$PWD uv run python -c "from splice.detector import detect_splices; from splice.features import extract_features; print('OK')"
PYTHONPATH=$PWD uv run python -c "import splice.classifier.train_classifier; print('imports OK')"

# py_compile all tracked .py
git ls-files -- '*.py' | xargs -I{} uv run python -m py_compile {}

# Loop sentinel still present
test -f .omc/autoresearch-stop && echo "PASS: sentinel present" || echo "FAIL: loop sentinel missing"
```

**Gate status after commit 4:**
- All 6 regression gates: GREEN

**Acceptance gate (a) -- reclassified sub-criteria (Approach A):**

The spec's literal "<=10 tracked entries" yields 16 post-reorg (`analysis.ipynb` and `progress.png` are staged for deletion but `progress.txt`, `data_synth/`, `scripts/`, etc. remain). Achieving <=10 would require sweeping loose artifacts into a subdirectory or deleting them -- scope creep for US-516. The literal <=10 is **aspirational** and achievable as a future tidying PR if the user wants it.

The enforceable sub-criteria of gate (a) are:

**(a-i) Zero tracked `*.py` under `.omc/`** (excluding `.omc/classifier/detector_v*` which is untracked debris):
```bash
git ls-files -- '.omc/**/*.py' | grep -v 'detector_v' | head -1 | grep -q . && echo "FAIL: tracked .py in .omc/" || echo "PASS"
```

**(a-ii) `scripts/` contains only operator tools + analysis -- no library helpers imported by runtime/app packages.** This is a structural invariant, not a grep: `log_reader.py` (the only former library helper in `scripts/`) has moved to `autoresearch/`. No remaining `scripts/*.py` file is imported by any module in `autoresearch/` or `splice/`. Verification is manual review -- check that no `scripts/*.py` file appears as an import source in the two packages:
```bash
git grep -l 'from scripts\.\|import scripts\.' -- 'autoresearch/**/*.py' 'splice/**/*.py' | head -1 | grep -q . && echo "FAIL: runtime/app imports from scripts/" || echo "PASS"
```
Note: `data_synth/regenerate_datasets.py` legitimately imports `from scripts.splice_boundary import find_splice_point` -- this is a data-generation tool, not a runtime/app package, so it does not violate the gate.

**(a-iii) Every top-level entry's purpose is obvious from its name.** This is a subjective/vibes criterion. **Non-gating.** The executor should note any ambiguous names in the commit message but this sub-criterion does not block acceptance.

**Acceptance gate (b):** logger -> `autoresearch/logger.py`. detector -> `splice/detector.py`. 30s locate: pass.

**Acceptance gate (c):**
```bash
git grep -n 'sys\.path\.insert' -- '*.py' | head -1 | grep -q . && echo "FAIL: sys.path.insert in tracked .py" || echo "PASS"
```
Uses `git grep` (tracked files only, auto-excludes `.venv/` and untracked debris).

**Acceptance gate (d) -- fork-friendly reproduction recipe:**
```bash
TMPDIR=$(mktemp -d) && cp -r . "$TMPDIR/us516-fork-test" && \
cd "$TMPDIR/us516-fork-test" && rm -rf splice/ && \
PYTHONPATH="$TMPDIR/us516-fork-test" uv run python -c "from autoresearch.logger import get_logger; print('OK')" && \
PYTHONPATH="$TMPDIR/us516-fork-test" uv run pytest autoresearch/tests/test_logger.py -v && \
echo "PASS" || echo "FAIL"
cd - && rm -rf "$TMPDIR/us516-fork-test"
```

---

## Import Rewrite Strategy (Section e)

### Pattern: `sys.path.insert(0, '.omc/coordination'); from logger import get_logger`

**Becomes:** `from autoresearch.logger import get_logger`

Requires `PYTHONPATH` set to repo root (done by `run_autoresearch.sh`) or `-c`/`-m` invocation (which adds CWD to path).

### Why `uv run python <file>` does NOT add repo root to sys.path

**Empirically proven (see Appendix A).** `uv run python splice/evaluate.py` puts `splice/` (the file's parent directory) on `sys.path[0]`, NOT the repo root. This means `from autoresearch.logger import get_logger` fails with `ModuleNotFoundError` unless the repo root is on `sys.path` via another mechanism.

**Resolution:** `PYTHONPATH="${PROJECT_DIR}"` in `run_autoresearch.sh`. See MUST-FIX #2 above.

### Entry-point scripts

**`evaluate.py` (invoked as `uv run python splice/evaluate.py`):**
- Works because `PYTHONPATH` adds repo root. `from splice.detector import detect_splices` resolves.

**`train_classifier.py` (invoked as `uv run python splice/classifier/train_classifier.py`):**
- Same mechanism. `PYTHONPATH` ensures repo root is on path.

### Test files

**Current:** `sys.path.insert(0, str(REPO / ".omc" / "coordination"))` then `import logger as lg`.

**After:** `import autoresearch.logger as lg`. No `sys.path.insert` needed (pytest invoked via `uv run pytest` which adds CWD to path, or via `PYTHONPATH`).

### `data_synth/regenerate_datasets.py`

Currently: `sys.path.insert(0, str(_ROOT / "scripts"))` then `from splice_boundary import find_splice_point`.

**After:** `from scripts.splice_boundary import find_splice_point` (enabled by `scripts/__init__.py` created in commit 1).

### `.omc/experiments/energy_dip_test.py`

If tracked and contains `sys.path.insert`, rewrite imports to use packages. Low-risk scratch file.

---

## run_autoresearch.sh Path-Ref Updates (Section f)

The wrapper invokes Python files via `uv run python <path>`. After migration:

| Current invocation | New invocation |
|---|---|
| `uv run python "$PROJECT_DIR/.omc/coordination/log_cli.py" "$@"` | `uv run python "$PROJECT_DIR/autoresearch/log_cli.py" "$@"` |
| `uv run python .omc/coordination/verify_agent.py --diagnose ...` | `uv run python autoresearch/verify_agent.py --diagnose ...` |
| `uv run python .omc/classifier/train_classifier.py` | `uv run python splice/classifier/train_classifier.py` |
| `uv run python evaluate.py --shap` | `uv run python splice/evaluate.py --shap` |

**Critical addition:** `export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"` near top of wrapper (after `PROJECT_DIR` is set). Without this, cross-package imports fail for all `uv run python <file>` invocations (see Appendix A).

---

## Doc Updates (Section g)

### CLAUDE.md

| Section | Old text | New text |
|---|---|---|
| Mandatory Verification | `uv run python .omc/coordination/verify_agent.py` | `uv run python autoresearch/verify_agent.py` |
| Constraints | `evaluate.py`, `detector.py`, `features.py`, `.omc/classifier/train_classifier.py` | `splice/evaluate.py`, `splice/detector.py`, `splice/features.py`, `splice/classifier/train_classifier.py` |
| Protected Files | `.omc/coordination/manifest.json`, `.omc/coordination/preflight.py` | `autoresearch/manifest.json`, `autoresearch/preflight.py` |
| Preflight | `uv run python .omc/coordination/preflight.py` | `uv run python autoresearch/preflight.py` |
| Baseline | `.omc/coordination/baseline_metrics.json` | `autoresearch/baseline_metrics.json` |
| Retest | `uv run python .omc/coordination/verify_agent.py --retest` | `uv run python autoresearch/verify_agent.py --retest` |
| Unified logging import example | `sys.path.insert(0, ".omc/coordination"); from logger import get_logger` | `from autoresearch.logger import get_logger` |
| Unified logging invocation | `uv run python .omc/coordination/tests/test_smoke_iteration.py` | `uv run python autoresearch/tests/test_smoke_iteration.py` |
| Bash emission | `.omc/coordination/log_cli.py` | `autoresearch/log_cli.py` |
| Gates | `.omc/coordination/tests/test_smoke_iteration.py`, `scripts/validate_logs.py` | `autoresearch/tests/test_smoke_iteration.py`, `scripts/validate_logs.py` |
| New section | — | US-516 migration note + `PYTHONPATH` requirement for standalone invocations |

---

## Loop-Safety Invariant (Section h)

1. **Sentinel present.** `.omc/autoresearch-stop` MUST exist before any US-516 commit lands and throughout execution.
   ```bash
   test -f .omc/autoresearch-stop && echo "PASS" || echo "FAIL"
   ```
2. **No manual commits after restart.** After `./run_autoresearch.sh start` is invoked as the final acceptance step, no human commits on the branch. The loop owns the commit history from that point. *(Procedural invariant -- enforced by operator discipline, not a runnable check.)*
3. **Restart is the final gate.** Loop restart is not an intermediate step. It is invoked only after all 4 acceptance gates and all 6 regression gates are confirmed green. Observing >= 1 clean iteration with expected JSONL events completes US-516. *(This IS the commit-4 verification command battery above -- cross-reference.)*

---

## Rollback Plan (Section i)

### During migration (commits 1-4)

**If a gate fails after commit N:**
1. Diagnose which gate failed and why.
2. If the fix is small (typo in import path), fix forward with an additional commit.
3. If the fix is non-trivial, revert commits back to the last green state:
   - `git revert HEAD~N..HEAD` (creates revert commits, preserves history).
   - Alternatively, `git reset --hard <last-green-sha>` if the branch has no pushed state that others depend on.

**Recovery sequence:**
1. `git log --oneline -10` to identify the last green commit.
2. `git revert --no-commit HEAD~N..HEAD` to stage all reverts.
3. `git commit -m "revert: US-516 migration (gate failure in commit N)"`.
4. Re-run all 6 gates to confirm green.
5. Investigate the failure, fix, and re-attempt.

### After loop restart

If the first iteration fails on the new layout:
1. `./run_autoresearch.sh stop`
2. Read `.omc/logs/autoresearch.jsonl` for the failure event.
3. Fix the path issue (likely a missed reference in `run_autoresearch.sh`).
4. Commit the fix.
5. Restart.

---

## Scope Guardrails (Section j)

The plan MUST NOT include:
- No new features beyond what the reorg requires.
- No new tests beyond what keeps gates green.
- No refactoring of verify_agent.py into sub-modules.
- No pip-packaging / pyproject entry points / `[tool.setuptools]` changes.
- No touch to `.omc/plans/*`, `.omc/specs/*`, `.omc/prd.json`.
- No backwards-compat shims that outlive the migration.
- No changes to `data/`, `data_synth/internals`, `reports/`, `.omc/feature_cache/`, `.omc/retest-worktree/` beyond updating paths that reference them.
- No touch to untracked files.

**Exception (rev-2):** `PYTHONPATH` export in `run_autoresearch.sh` is not a new feature. It is infrastructure required to make cross-package imports work with direct file invocation, as empirically proven in Appendix A.

---

## File-Touched Summary

| Commit | Files created | Files moved | Files updated | Total files | Est. lines changed |
|---|---|---|---|---|---|
| 1 | 6 (5 __init__.py + scripts/__init__.py) | 0 | 0 | 6 | ~8 |
| 2 | 0 | 14 (runtime src + tests + fixtures + json) | 5 (scripts consumers + run_autoresearch.sh PYTHONPATH + runtime paths) | 19 | ~110 |
| 3 | 0 | 12 (app src + classifier + program.md) | 4 (verify_agent PROTECTED, validate_logs _MIGRATED_FILES, fp_filter _MODEL_PATH via commit 2 already, run_autoresearch.sh remaining paths) | 16 | ~160 |
| 4 | 0 | 0 | 4 (CLAUDE.md, README.md, evaluate_integration.md, logger.py docstring) | 4 | ~60 |
| **Total** | **6** | **26** | **13** | **45** | **~340** |

---

## Success Criteria

1. All 4 acceptance gates pass: (a) sub-criteria (a-i) and (a-ii) green, (a-iii) non-gating noted; (b) 30s locate; (c) zero `sys.path.insert` via `git grep`; (d) fork-friendly reproduction recipe passes.
2. All 6 regression gates pass.
3. `./run_autoresearch.sh start` produces >= 1 clean iteration with expected JSONL events.
4. Zero `sys.path.insert` in tracked .py files (gate c): `git grep -n 'sys\.path\.insert' -- '*.py'` returns empty.
5. Zero `.py` files under `.omc/` (tracked; untracked snapshots excluded): gate (a-i).
6. PROTECTED_FILES in verify_agent.py points to final locations.
7. CLAUDE.md contains US-516 migration note.
8. **NEW (rev-2):** `REPO_ROOT` invariant check passes: `uv run python -c "from autoresearch.verify_agent import REPO_ROOT; assert REPO_ROOT.name == 'autoresearch-splice'"`.
9. **NEW (rev-3):** `PYTHONPATH` export present in wrapper: `grep -q 'export PYTHONPATH' run_autoresearch.sh`.

---

## Appendix A: Empirical Import Probe (2026-04-18)

### Setup

Temporary scaffolding created in repo root (cleaned up, not committed):
```
autoresearch/__init__.py     (empty)
autoresearch/logger.py       (copy of .omc/coordination/logger.py)
splice/__init__.py           (empty)
splice/evaluate_probe.py     (from autoresearch.logger import get_logger; print("OK"))
```

### Results

**Test 1: `uv run python splice/evaluate_probe.py`**
```
$ uv run python splice/evaluate_probe.py
Traceback (most recent call last):
  File ".../splice/evaluate_probe.py", line 1, in <module>
    from autoresearch.logger import get_logger
ModuleNotFoundError: No module named 'autoresearch'
EXIT=1
```

**sys.path inspection:**
```
sys.path[:5]: ['splice', ...]
```
The file's parent directory (`splice/`) is on sys.path[0], NOT the repo root.

**Test 2: `uv run python -m splice.evaluate_probe`**
```
$ uv run python -m splice.evaluate_probe
OK - direct import works
EXIT=0
```
`-m` puts `''` (CWD) on sys.path[0].

**Test 3: `uv run python -c "from autoresearch.logger import get_logger; print('ok')"`**
```
ok
EXIT=0
```
`-c` puts `''` on sys.path[0].

**Test 4: `PYTHONPATH=$REPO uv run python splice/evaluate_probe.py`**
```
OK - direct import works
EXIT=0
```

### Conclusion

`uv run python <file>` does NOT add the repo root to sys.path. It adds the file's parent directory. Cross-package imports (`from autoresearch.X import Y` inside `splice/Y.py`) fail unless:
1. `PYTHONPATH` includes the repo root, OR
2. The script is invoked via `python -m <module>`, OR
3. The project is installed as an editable package.

Strategy (1) is chosen: `export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"` in `run_autoresearch.sh`.
