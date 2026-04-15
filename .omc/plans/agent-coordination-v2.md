# Agent Coordination System v2

**Status:** Revised per Architect + Critic feedback (2026-04-15)
**Scope:** 4 tasks (Phase 1) + 3 deferred tasks (Phase 2)
**Estimated complexity:** MEDIUM

---

## Context

Multiple Claude Code agents can run concurrently on the splice detector repo,
especially across git worktrees. Without coordination, agents may:
- Mutate shared test data (`data/spliced/`) causing non-reproducible results
- Run duplicate evaluations simultaneously
- Produce conflicting changes to shared files

This plan adds lightweight coordination: a data manifest for integrity checks,
a preflight script that validates the environment before any agent runs
`test_regression.py`, and OS-level read-only protection on the shared dataset.

## ADR: Phased Lightweight Coordination

**Decision:** Ship manifest + preflight + chmod as Phase 1 minimum viable unit.
Defer locking, cleanup, and AGENTS.md to Phase 2.

**Drivers:**
1. Data integrity is the highest-risk, highest-value protection
2. Preflight depends on manifest for expected counts and GT hash -- they ship together
3. Locking infrastructure is only needed when concurrent agents are confirmed in practice

**Alternatives considered:**
- **(A) Full 6-task ship** -- Original plan. Rejected: lock system has no proven need yet, adds maintenance burden, Critic flagged over-engineering.
- **(B) Manifest only** -- Too minimal. Preflight is the enforcement point; manifest without preflight is just documentation.
- **(C) Phase 1 (manifest + preflight + chmod) + Phase 2 (locks + cleanup + docs)** -- Chosen. Delivers the highest-value protection immediately while deferring unproven complexity.

**Why chosen:** Option C gives 80% of the safety value with 50% of the implementation cost. Phase 2 only triggers when concurrent agent runs are observed in practice.

**Consequences:** Agents have no mutual exclusion in Phase 1. If two agents run `test_regression.py` simultaneously, both will pass preflight and run. This is acceptable because evaluate() is read-only on the dataset.

**Follow-ups:** Monitor for concurrent agent conflicts. If observed, promote Phase 2.

---

## Guardrails

**Must Have:**
- All Phase 1 files live in `.omc/coordination/` (not a Python package, no `__init__.py`)
- Direct invocation only: `uv run python .omc/coordination/preflight.py`
- Lock paths (Phase 2) must use `git rev-parse --git-common-dir` for worktree-safe resolution
- `--skip-preflight` env var (`SKIP_PREFLIGHT=1`) bypass for emergencies

**Must NOT Have:**
- No `-m` module invocation, no package structure
- No modification of `prepare.py` or detector logic
- No changes to `data/spliced/` contents (read-only enforcement is the whole point)

---

## Phase 1 -- NOW (Minimum Viable Unit)

### Task 0: chmod 555 on shared data directory

**File:** None (OS operation only)

**Action:**
```bash
chmod -R 555 "$(realpath /Users/yejunjang/Projects/mutual/autoresearch-splice/data/spliced)"
# NOTE: data/spliced is a symlink to audio-splice-detector/data/spliced.
# This chmod applies to the SHARED source directory. Any scripts in
# audio-splice-detector that write to data/spliced/ will need `chmod 755`
# temporarily when regenerating data.
```

**Rationale:** Real OS-level read-only enforcement. Any agent or script attempting
to write into `data/spliced/` gets a permission error immediately, before any
coordination logic even runs.

**Acceptance criteria:**
- `stat -f "%Sp" "$(realpath data/spliced)"` shows `r-xr-xr-x`
- `touch data/spliced/test_write 2>&1` returns "Permission denied"
- `test_regression.py` still passes (evaluate() only reads)
- Recursive: all subdirs (clean/, tier1/, tier2/) and files are 555/444

---

### Task 1: Data integrity manifest

**File:** `.omc/coordination/manifest.json`

**Content:** Static JSON declaring the expected state of `data/spliced/`.

```json
{
  "version": 1,
  "dataset": "data/spliced",
  "expected_counts": {
    "tier1": 20,
    "tier2": 20,
    "clean": 50,
    "_note": "ground_truth.json is validated separately via ground_truth_sha256_prefix"
  },
  "ground_truth_sha256_prefix": "c76f4804df0f",
  "total_files": 91,
  "notes": "Counts exclude .DS_Store. GT hash is first 12 chars of sha256."
}
```

**Acceptance criteria:**
- File is valid JSON, parseable by `json.load()`
- `expected_counts` values match actual file counts in each subdir
- `ground_truth_sha256_prefix` matches `shasum -a 256 data/spliced/ground_truth.json | cut -c1-12`
- No code files reference this manifest yet (Task 2 wires it up)

---

### Task 2: Preflight validation script

**File:** `.omc/coordination/preflight.py`

**Functions:**

1. `load_manifest() -> dict`
   - Reads `manifest.json` from same directory (path relative to `__file__`)
   - Returns parsed dict

2. `check_data_integrity(manifest: dict) -> list[str]`
   - Counts files in each subdir of `data/spliced/`, excluding `.DS_Store`
   - Compares to `manifest["expected_counts"]`
   - Verifies `ground_truth.json` sha256 prefix matches
   - Returns list of error strings (empty = all clear)

3. `check_permissions(data_dir: str) -> list[str]`
   - Checks `data/spliced` is not writable by current user (`os.access(path, os.W_OK)`)
   - Returns warnings (not errors) if writable -- chmod may not have been applied yet

4. `run_preflight() -> bool`
   - Orchestrates checks, prints results
   - Returns True if all critical checks pass
   - Respects `SKIP_PREFLIGHT=1` env var: prints warning, returns True immediately
   - Exit code 0 on pass, 1 on failure when run as `__main__`

**Invocation:** `uv run python .omc/coordination/preflight.py`

**Agent ID generation (for logging):** `f"{branch_name}_{os.getpid()}"` where
`branch_name` comes from `git rev-parse --abbrev-ref HEAD`.

**Acceptance criteria:**
- Running `uv run python .omc/coordination/preflight.py` exits 0 on current repo state
- Temporarily renaming a tier1 file and rerunning causes exit 1 with clear error message
- `SKIP_PREFLIGHT=1 uv run python .omc/coordination/preflight.py` exits 0 with skip warning
- No `__init__.py` files anywhere in `.omc/`
- Script uses `pathlib` or `os.path` relative to `__file__` to find manifest.json
- Permissions check is a warning, not a blocker

---

### Task 6: Wire preflight into test_regression.py

**File:** `test_regression.py`

**Insertion point:** After line 89 (DATA_DIR existence check, before line 92 (`r1 = run_evaluate()`).

Current lines 87-92:
```python
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: test dataset not found at {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    # --- Two runs for determinism ---
    r1 = run_evaluate()
```

Insert between the `sys.exit(1)` block and the `# --- Two runs` comment:

```python
    # --- Preflight coordination check ---
    preflight_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   ".omc", "coordination", "preflight.py")
    if os.path.exists(preflight_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("preflight", preflight_path)
        preflight = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preflight)
        if not preflight.run_preflight():
            print("ERROR: preflight check failed", file=sys.stderr)
            sys.exit(1)
    else:
        print("  Preflight: SKIP (coordination module not found)")
```

**Behavior:**
- **Fail-fast vs graceful:** Graceful in test_regression.py ONLY. If preflight.py is
  missing, print skip message and continue. If preflight.py exists but checks fail,
  exit 1.
- This is the ONLY file that gets graceful degradation. All other consumers (Phase 2
  agents) must fail-fast if preflight fails.

**Acceptance criteria:**
- `uv run python test_regression.py` still passes with preflight.py present
- `uv run python test_regression.py` still passes with preflight.py absent (graceful skip)
- Corrupted data causes test_regression.py to exit 1 at preflight stage, before evaluate() runs
- No new imports at top of file (importlib.util is imported inline)
- Insertion does not disturb line numbers of existing metric checks

---

## Phase 2 -- LATER (Deferred until concurrent agents confirmed needed)

> **Trigger:** Promote to active when concurrent agent conflicts are observed in practice.

### Task 3: Lock manager (deferred)

**File:** `.omc/coordination/locks.py`

**Key design decision:** `DetectorLock` must resolve lock directory via
`git rev-parse --git-common-dir` to find shared `.omc/coordination/locks/` path.
Per-worktree lock dirs defeat the purpose of cross-worktree coordination.

```python
def _get_lock_dir() -> Path:
    common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        text=True
    ).strip()
    # common is relative to cwd or absolute
    common_path = Path(common).resolve()
    # .omc lives at repo root, not inside .git
    repo_root = common_path.parent if common_path.name == ".git" else common_path.parent
    return repo_root / ".omc" / "coordination" / "locks"
```

**Agent ID:** `f"{branch_name}_{os.getpid()}"` for uniqueness across worktrees.

**Scope:** `DetectorLock` context manager with acquire/release, stale lock detection
(configurable timeout, default 30 min), force-break capability.

### Task 4: Stale lock cleanup (deferred)

**File:** `.omc/coordination/cleanup.py`

**Scope:** CLI tool to list and remove stale locks. Stale = lock age > threshold
or PID no longer running.

### Task 5: AGENTS.md coordination docs (deferred)

**File:** `AGENTS.md`

**Scope:** Document preflight protocol, lock protocol, agent ID conventions,
and emergency bypass procedures.

---

## Success Criteria

**Phase 1 complete when:**
1. `data/spliced` is chmod 555 recursively
2. `manifest.json` exists with correct counts and GT hash
3. `preflight.py` runs standalone and validates data integrity
4. `test_regression.py` calls preflight before evaluate() with graceful fallback
5. All existing tests still pass: `uv run python test_regression.py` exits 0

**Phase 2 promotion criteria:**
- Evidence of two or more agents running test_regression.py concurrently
- Or explicit user decision to add locking infrastructure
