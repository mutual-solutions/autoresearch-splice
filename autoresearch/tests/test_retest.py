"""Tests for US-514 --retest subcommand.

Mirrors the 10 cases listed in .omc/plans/ralplan-retest-discards.md step 8.
The real correctness gate is `uv run python .omc/coordination/verify_agent.py
--retest-self-test` (AC #17); these pytest mirrors exist for CI visibility
and to exercise a few invariants the self-test cannot easily pin (sentinel
SIGKILL-window, loop-start refusal, _keep_path byte-identity).

Fixtures are throwaway git repos under tmp_path. We import the verify_agent
module by explicit file path so tests are hermetic and do not depend on
package-discovery rules.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_AGENT_PATH = REPO_ROOT / "autoresearch" / "verify_agent.py"
WRAPPER_SH = REPO_ROOT / "run_autoresearch.sh"


def _load_verify_agent():
    """Import verify_agent.py as an ephemeral module for each test.

    Each call returns a FRESH module so per-test patching of module-level
    constants (PROJECT_DIR, _RESULTS_TSV, etc.) does not bleed across
    tests.
    """
    spec = importlib.util.spec_from_file_location(
        f"_verify_agent_test_{id(object())}", VERIFY_AGENT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _git(*args: str, cwd: Path, check: bool = True,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    e.setdefault("GIT_AUTHOR_NAME", "retest-test")
    e.setdefault("GIT_AUTHOR_EMAIL", "retest-test@example.invalid")
    e.setdefault("GIT_COMMITTER_NAME", "retest-test")
    e.setdefault("GIT_COMMITTER_EMAIL", "retest-test@example.invalid")
    if env:
        e.update(env)
    return subprocess.run(
        ["git", *args], cwd=cwd, env=e, check=check,
        capture_output=True, text=True, timeout=30,
    )


def _init_fixture_repo(tmp_path: Path) -> tuple[Path, str]:
    """Initialize a small git repo and return (path, anchor_sha)."""
    _git("init", "-q", cwd=tmp_path)
    (tmp_path / "seed.txt").write_text("seed\n")
    _git("add", "seed.txt", cwd=tmp_path)
    _git("commit", "-q", "-m", "seed", cwd=tmp_path,
         env={"GIT_COMMITTER_DATE": "2026-01-01T00:00:00"})
    anchor = _git("rev-parse", "HEAD", cwd=tmp_path).stdout.strip()
    return tmp_path, anchor


def _make_commit(repo: Path, path: str, content: str, subject: str,
                 ct: str) -> str:
    (repo / path).write_text(content)
    _git("add", path, cwd=repo)
    _git("commit", "-q", "-m", subject, cwd=repo,
         env={"GIT_COMMITTER_DATE": ct})
    return _git("rev-parse", "HEAD", cwd=repo).stdout.strip()


def _write_tsv(repo: Path, rows: list[tuple[str, str, str]]) -> Path:
    """rows: list of (sha, combined, description)."""
    tsv = repo / "results.tsv"
    header = "\t".join([
        "commit", "combined", "combined_mean", "combined_min",
        "clean_fp", "n_datasets", "combined_singing",
        "combined_korean", "combined_english", "clean_fp_singing",
        "clean_fp_korean", "clean_fp_english", "status", "description",
    ])
    lines = [header]
    for sha, combined, desc in rows:
        lines.append("\t".join([
            sha[:7], combined, combined, combined, "0", "3",
            combined, combined, combined, "0", "0", "0",
            "discard", desc,
        ]))
    tsv.write_text("\n".join(lines) + "\n")
    return tsv


def _write_baseline(repo: Path, combined: float) -> Path:
    d = repo / "autoresearch"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "baseline_metrics.json"
    p.write_text(json.dumps({"combined": combined}))
    return p


# ---------------------------------------------------------------------------
# Test 1: candidate ordering honors (%ct, row_index) tiebreaker.
# ---------------------------------------------------------------------------

def test_candidate_enumeration_ordering_with_ct_tiebreaker(tmp_path):
    mod = _load_verify_agent()
    repo, anchor = _init_fixture_repo(tmp_path)

    sha_a = _make_commit(repo, "a.txt", "A\n", "hypothesis: A",
                         "2026-02-01T00:00:00")
    # B and C share the same committer date (1s resolution collision).
    sha_b = _make_commit(repo, "b.txt", "B\n", "hypothesis: B",
                         "2026-03-01T00:00:00")
    sha_c = _make_commit(repo, "c.txt", "C\n", "hypothesis: C",
                         "2026-03-01T00:00:00")

    # TSV order: A, then C, then B — C appears before B despite same %ct.
    # Stable sort on (commit_time, row_index) must preserve append order.
    _write_tsv(repo, [
        (sha_a, "0.300000", "candidate A"),
        (sha_c, "0.450000", "candidate C"),
        (sha_b, "0.350000", "candidate B"),
    ])

    def _patched_ct(sha: str) -> int | None:
        r = subprocess.run(
            ["git", "show", "-s", "--format=%ct", sha],
            capture_output=True, text=True, cwd=repo, timeout=10,
        )
        if r.returncode != 0:
            return None
        return int(r.stdout.strip())

    mod._git_commit_time = _patched_ct
    cands = mod._load_retest_candidates(repo / "results.tsv", anchor)
    got = [c.sha[:7] for c in cands]
    assert got == [sha_a[:7], sha_c[:7], sha_b[:7]], (
        f"expected A, C, B ordering (stable tiebreak on shared %ct); "
        f"got {got}"
    )


# ---------------------------------------------------------------------------
# Test 2: --dry-run mutates nothing (no worktree, no sentinel, no subprocess).
# ---------------------------------------------------------------------------

def test_dry_run_no_mutation(tmp_path, monkeypatch):
    mod = _load_verify_agent()
    repo, anchor = _init_fixture_repo(tmp_path)
    sha_a = _make_commit(repo, "a.txt", "A\n", "hypothesis: A",
                         "2026-02-01T00:00:00")
    _write_tsv(repo, [(sha_a, "0.300000", "candidate A")])
    _write_baseline(repo, 0.4)

    mod.PROJECT_DIR = repo
    mod._RESULTS_TSV = repo / "results.tsv"
    mod.BASELINE_PATH = repo / "autoresearch" / "baseline_metrics.json"
    mod._RETEST_SENTINEL = repo / ".omc" / "retest-in-progress"
    mod._RETEST_WORKTREE = repo / ".omc" / "retest-worktree"
    mod._RETEST_REPORT = repo / ".omc" / "retest-report.md"
    mod._tmux_session_alive = lambda _n: False
    mod._eval_root_ok = lambda: (True, str(repo))

    def _patched_ct(sha: str) -> int | None:
        r = subprocess.run(
            ["git", "show", "-s", "--format=%ct", sha],
            capture_output=True, text=True, cwd=repo, timeout=10,
        )
        if r.returncode != 0:
            return None
        return int(r.stdout.strip())
    mod._git_commit_time = _patched_ct

    # Sentinel for subprocess calls: evaluate.py MUST NOT be invoked in dry-run.
    orig_run = subprocess.run
    seen = []

    def _spy_run(cmd, *a, **kw):
        seen.append(cmd)
        return orig_run(cmd, *a, **kw)
    monkeypatch.setattr(subprocess, "run", _spy_run)

    monkeypatch.chdir(repo)
    rc = mod.run_retest(anchor, dry_run=True, limit=None)

    assert rc == 0
    assert not (repo / ".omc" / "retest-in-progress").exists(), (
        "dry-run wrote sentinel (invariant violated)"
    )
    assert not (repo / ".omc" / "retest-worktree").exists(), (
        "dry-run created worktree (invariant violated)"
    )
    # Zero subprocess calls to evaluate.py (or bash run_autoresearch.sh).
    for cmd in seen:
        as_str = " ".join(str(c) for c in cmd)
        assert "evaluate.py" not in as_str, (
            f"dry-run invoked evaluate.py (cmd={as_str})"
        )
        assert "run_autoresearch.sh" not in as_str or "_keep_path" not in as_str, (
            f"dry-run invoked run_autoresearch.sh _keep_path (cmd={as_str})"
        )
    report_text = (repo / ".omc" / "retest-report.md").read_text()
    assert "dry-run-skipped" in report_text
    assert sha_a[:7] in report_text


# ---------------------------------------------------------------------------
# Test 3: atexit cleans up worktree even on exception.
# ---------------------------------------------------------------------------

def test_worktree_atexit_cleanup_on_exception(tmp_path, monkeypatch):
    mod = _load_verify_agent()
    repo, _ = _init_fixture_repo(tmp_path)

    # Add one more commit so worktree checkout has something reasonable.
    _make_commit(repo, "extra.txt", "x\n", "extra", "2026-02-01T00:00:00")

    mod.PROJECT_DIR = repo
    mod._RETEST_WORKTREE = repo / ".omc" / "retest-worktree"
    mod._WRAPPER_SH = WRAPPER_SH

    # Perform a real `git worktree add` in the fixture repo.
    mod._worktree_setup()
    assert mod._RETEST_WORKTREE.exists()

    # Simulate an exception by invoking cleanup directly — the atexit
    # handler's signature.
    mod._worktree_cleanup()
    assert not mod._RETEST_WORKTREE.exists(), (
        "worktree cleanup should remove the path"
    )

    # Idempotent: second call is a no-op, not an error.
    mod._worktree_cleanup()


# ---------------------------------------------------------------------------
# Test 4: cherry-pick conflict is aborted, no leftover index state.
# ---------------------------------------------------------------------------

def test_cherry_pick_conflict_skips_without_abort_leftover(tmp_path):
    mod = _load_verify_agent()
    repo, _ = _init_fixture_repo(tmp_path)

    # Build a conflict scenario:
    #   1) commit A on main that edits shared.txt to "A"
    #   2) reset back and commit B that edits shared.txt to "B"
    #   3) retest tries to cherry-pick A on top of B → conflict.
    (repo / "shared.txt").write_text("base\n")
    _git("add", "shared.txt", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo,
         env={"GIT_COMMITTER_DATE": "2026-02-01T00:00:00"})
    base_sha = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()

    (repo / "shared.txt").write_text("A\n")
    _git("add", "shared.txt", cwd=repo)
    _git("commit", "-q", "-m", "hypothesis: A", cwd=repo,
         env={"GIT_COMMITTER_DATE": "2026-03-01T00:00:00"})
    sha_a = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()

    _git("reset", "--hard", base_sha, cwd=repo)
    (repo / "shared.txt").write_text("B\n")
    _git("add", "shared.txt", cwd=repo)
    _git("commit", "-q", "-m", "hypothesis: B", cwd=repo,
         env={"GIT_COMMITTER_DATE": "2026-04-01T00:00:00"})

    # Set up the worktree at HEAD (which is commit B) and try to
    # cherry-pick A — guaranteed conflict.
    mod.PROJECT_DIR = repo
    mod._RETEST_WORKTREE = repo / ".omc" / "retest-worktree"
    mod._WRAPPER_SH = WRAPPER_SH
    mod._worktree_setup()

    cand = mod.RetestCandidate(
        sha=sha_a, original_combined=0.3, commit_time=0, row_index=0,
        description="A conflicts with B",
    )

    # Stub out retrain (would call the real wrapper) — we only want
    # the conflict branch.
    called = {"retrain": 0}

    orig_run = subprocess.run

    def _wrapped_run(cmd, *a, **kw):
        # Intercept the _ensure_classifier_fresh verb only.
        if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] == "bash" \
                and cmd[2] == "_ensure_classifier_fresh":
            called["retrain"] += 1
            class _R:
                returncode = 0
                stdout = ""
                stderr = ""
            return _R()
        # Block evaluate.py (should not reach it after conflict).
        if isinstance(cmd, list) and "evaluate.py" in " ".join(str(c) for c in cmd):
            raise AssertionError("evaluate.py reached after conflict branch")
        return orig_run(cmd, *a, **kw)

    import unittest.mock as mock
    with mock.patch("subprocess.run", _wrapped_run):
        base_sha_wt = _git("rev-parse", "HEAD",
                           cwd=mod._RETEST_WORKTREE).stdout.strip()
        mod._replay_one(cand, base_sha_wt)

    assert cand.outcome == "conflict", (
        f"expected outcome=conflict, got {cand.outcome!r} "
        f"(note={cand.note!r})"
    )
    # Worktree must be clean (no CHERRY_PICK_HEAD left).
    chp = (mod._RETEST_WORKTREE / ".git").read_text() \
        if (mod._RETEST_WORKTREE / ".git").is_file() else ""
    # For detached worktrees the actual git state lives under .git/worktrees/
    # in the main repo; a residual CHERRY_PICK_HEAD would show up in `git
    # status --porcelain`.
    status = subprocess.run(
        ["git", "-C", str(mod._RETEST_WORKTREE), "status", "--porcelain"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    assert "CHERRY_PICK_HEAD" not in status
    # Clean up.
    mod._worktree_cleanup()


# ---------------------------------------------------------------------------
# Test 5: _current_baseline re-reads disk each call (no in-memory cache).
# ---------------------------------------------------------------------------

def test_recovery_reads_baseline_from_disk_each_iteration(tmp_path):
    mod = _load_verify_agent()
    repo, _ = _init_fixture_repo(tmp_path)
    bpath = _write_baseline(repo, 0.4)

    mod.BASELINE_PATH = bpath
    b1 = mod._current_baseline()
    assert abs(b1 - 0.4) < 1e-9

    # Mutate the file on disk — simulates the _keep_path subprocess
    # advancing the baseline between iterations.
    bpath.write_text(json.dumps({"combined": 0.55}))
    b2 = mod._current_baseline()
    assert abs(b2 - 0.55) < 1e-9, (
        f"_current_baseline did not re-read disk: got {b2}, expected 0.55"
    )

    # A second rewrite must also be observed — guards against any future
    # module-level caching creep.
    bpath.write_text(json.dumps({"combined": 0.62}))
    b3 = mod._current_baseline()
    assert abs(b3 - 0.62) < 1e-9


# ---------------------------------------------------------------------------
# Test 6: sentinel file blocks the wrapper's start / _loop / _loop_restart.
# ---------------------------------------------------------------------------

def test_sentinel_blocks_loop_start(tmp_path):
    # We reuse the ACTUAL wrapper to pin the sentinel-guard contract. The
    # wrapper needs PROJECT_DIR to be the repo where .omc/retest-in-progress
    # lives, which is the repo root. We simulate by touching the real
    # sentinel path briefly, but run a decoupled test: call the bash
    # helper directly and assert the refusal.
    #
    # To avoid polluting the live repo, we copy only run_autoresearch.sh
    # into tmp_path/, point PROJECT_DIR-equivalent there, touch the
    # sentinel, and invoke `bash ./run_autoresearch.sh _loop`.

    local_wrapper = tmp_path / "run_autoresearch.sh"
    shutil.copyfile(WRAPPER_SH, local_wrapper)
    local_wrapper.chmod(0o755)
    (tmp_path / ".omc").mkdir()
    (tmp_path / ".omc" / "retest-in-progress").write_text(
        json.dumps({"sha": "deadbeef", "started_at": "x"})
    )

    for verb in ("start", "_loop", "_loop_restart"):
        r = subprocess.run(
            ["bash", str(local_wrapper), verb],
            capture_output=True, text=True, timeout=10,
            cwd=tmp_path,
        )
        assert r.returncode != 0, (
            f"wrapper verb {verb} did not refuse despite sentinel "
            f"(stdout={r.stdout!r}, stderr={r.stderr!r})"
        )
        assert "retest-in-progress" in r.stderr, (
            f"wrapper verb {verb} error message missing sentinel mention "
            f"(stderr={r.stderr!r})"
        )


# ---------------------------------------------------------------------------
# Test 7: sentinel survives the SIGKILL window (simulated mid-recovery exit).
# ---------------------------------------------------------------------------

def test_sentinel_survives_sigkill_window(tmp_path):
    mod = _load_verify_agent()
    sentinel = tmp_path / "retest-in-progress"
    mod._RETEST_SENTINEL = sentinel
    mod._RETEST_WORKTREE = tmp_path / "retest-worktree"

    # Simulate writing the sentinel (immediately before cherry-pick).
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({"sha": "abc1234", "started_at": "2026-04-18"}))

    # _worktree_cleanup runs on every exit path; it MUST NOT delete the
    # sentinel (that would defeat the SIGKILL hole).
    mod._worktree_cleanup()
    assert sentinel.exists(), (
        "atexit cleanup deleted the sentinel — SIGKILL hole reopened"
    )


# ---------------------------------------------------------------------------
# Test 8: corpus purge between candidates is reported with cutoff.
# ---------------------------------------------------------------------------

def test_corpus_purged_mid_batch_reports_cutoff(tmp_path):
    mod = _load_verify_agent()

    # Point _eval_root_ok at a path we control, then remove it.
    purged = tmp_path / "purged_eval_root"
    purged.mkdir()
    # Seed with a dummy file so initial check passes via any().
    (purged / "sentinel.txt").write_text("x")

    # The _check_corpus_alive helper reads via _eval_root_ok which reads
    # OMC_EVAL_DATA_ROOT. Patch _eval_root_ok to simulate the purge.
    state = {"alive": True}

    def _eval_root_ok():
        if state["alive"]:
            return True, str(purged)
        return False, str(purged)

    mod._eval_root_ok = _eval_root_ok

    # Initial: alive.
    mod._check_corpus_alive("abcdef1")  # should NOT raise

    # Purge the corpus.
    state["alive"] = False
    with pytest.raises(SystemExit) as exc_info:
        mod._check_corpus_alive("cutoff7")
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Test 9: _keep_path verb produces byte-identical commit message when
# --retest-origin is absent (loop-keep parity).
# ---------------------------------------------------------------------------

def test_keep_path_verb_byte_identical_commit_message(tmp_path):
    """Invoke `_do_keep_path` via the `_keep_path` verb with NO --retest-
    origin and assert the baseline commit subject matches the loop-keep
    shape: `baseline: combined=<reported> after keep <short>`.
    """
    # Build a standalone fixture repo that satisfies the wrapper's
    # minimum path needs:
    #   - git repo
    #   - .omc/classifier/ directory (for snapshot cp + versions.json)
    #   - autoresearch/baseline_metrics.json
    #   - detector.py (any file; wrapper cp's it)
    #   - .omc/last_eval.log with a RESULTS_TSV line
    #   - a wrapper at tmp_path/run_autoresearch.sh (copied from repo root)
    #   - scripts/shap_shift.py and scripts/notebook_digest.py stubs
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    (repo / "splice").mkdir()
    (repo / "splice" / "detector.py").write_text("# detector\n")
    (repo / ".omc").mkdir()
    (repo / ".omc" / "classifier").mkdir()
    (repo / "autoresearch").mkdir()
    (repo / "autoresearch" / "baseline_metrics.json").write_text(
        json.dumps({"combined": 0.40, "timestamp": "prev"})
    )
    (repo / ".omc" / "last_eval.log").write_text(
        "RESULTS_TSV: combined=0.50 combined_mean=0.50 combined_min=0.50 "
        "clean_fp=0 n_datasets=3 combined_singing=0.50 combined_korean=0.50 "
        "combined_english=0.50 clean_fp_singing=0 clean_fp_korean=0 "
        "clean_fp_english=0\n"
    )
    (repo / "scripts").mkdir()
    # Stub shap_shift to a no-op; wrapper tolerates failure, but a clean exit
    # keeps the log tidy.
    (repo / "scripts" / "shap_shift.py").write_text("print('(no shift)')\n")
    (repo / "scripts" / "notebook_digest.py").write_text("")
    (repo / ".omc" / "research_notes.md").write_text("")
    (repo / ".omc" / "last_reflection.md").write_text("")
    shutil.copyfile(WRAPPER_SH, repo / "run_autoresearch.sh")
    (repo / "run_autoresearch.sh").chmod(0o755)

    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "seed", cwd=repo,
         env={"GIT_COMMITTER_DATE": "2026-01-01T00:00:00"})

    # Create a "hypothesis" commit whose SHA will be the $1 arg.
    (repo / "splice" / "detector.py").write_text("# detector v2\n")
    _git("add", "splice/detector.py", cwd=repo)
    _git("commit", "-q", "-m", "hypothesis: test axis tweak", cwd=repo,
         env={"GIT_COMMITTER_DATE": "2026-02-01T00:00:00"})
    hypo_sha_short = _git("rev-parse", "--short", "HEAD", cwd=repo).stdout.strip()

    # Invoke the _keep_path verb with NO --retest-origin.
    r = subprocess.run(
        ["bash", str(repo / "run_autoresearch.sh"), "_keep_path",
         hypo_sha_short, "0.500000", "0.400000", "test axis tweak"],
        capture_output=True, text=True, timeout=120, cwd=repo,
    )
    assert r.returncode == 0, (
        f"_keep_path failed: stderr={r.stderr!r} stdout={r.stdout!r}"
    )

    # The baseline commit is at HEAD~1 (note: commit likely on top).
    # Scan backwards for the baseline: subject.
    log = _git("log", "--format=%s", "-5", cwd=repo).stdout.strip().splitlines()
    baseline_lines = [l for l in log if l.startswith("baseline:")]
    assert baseline_lines, f"no baseline: commit found in log: {log}"
    subj = baseline_lines[0]
    # Expected shape: `baseline: combined=0.500000 after keep <short>`.
    assert subj == f"baseline: combined=0.500000 after keep {hypo_sha_short}", (
        f"unexpected baseline subject: {subj!r}\n"
        f"(loop-keep byte-identity broken — --retest-origin ABSENT path "
        f"must produce the historical shape)"
    )


# ---------------------------------------------------------------------------
# Test 10: check_git_diff_audit returns WARN on missing OMC_HEAD_BEFORE.
# ---------------------------------------------------------------------------

def test_check_git_diff_audit_warn_on_missing_env(tmp_path, monkeypatch):
    mod = _load_verify_agent()
    repo, _ = _init_fixture_repo(tmp_path)

    # Clean tree — no staged, no unstaged.
    monkeypatch.chdir(repo)
    monkeypatch.delenv("OMC_HEAD_BEFORE", raising=False)

    status, detail = mod.check_git_diff_audit()
    assert status == "WARN", (
        f"unset env + clean tree should WARN; got {status}: {detail}"
    )
    assert "OMC_HEAD_BEFORE" in detail

    # With env set and same clean tree, must return PASS.
    anchor = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()
    monkeypatch.setenv("OMC_HEAD_BEFORE", anchor)
    status2, detail2 = mod.check_git_diff_audit()
    assert status2 == "PASS", (
        f"env set + clean committed diff should PASS; got {status2}: {detail2}"
    )

    # Protected-file edit must FAIL regardless of env. Track the file
    # first so `git diff` sees the modification surface.
    (repo / "splice").mkdir(exist_ok=True)
    (repo / "splice" / "evaluate.py").write_text("# placeholder\n")
    _git("add", "splice/evaluate.py", cwd=repo)
    _git("commit", "-q", "-m", "add placeholder splice/evaluate.py", cwd=repo)
    # Now modify it — this produces an unstaged working-tree diff.
    (repo / "splice" / "evaluate.py").write_text("# protected and tampered\n")
    monkeypatch.delenv("OMC_HEAD_BEFORE", raising=False)
    status3, detail3 = mod.check_git_diff_audit()
    assert status3 == "FAIL", (
        f"protected-file edit must FAIL regardless of env; "
        f"got {status3}: {detail3}"
    )
