#!/usr/bin/env python3
"""Supervisor agent for the autoresearch-splice loop.

Renamed from verify_agent.py in US-517. Provides 4 subcommands:
  --verify    Re-run metric + diff audit + anomaly + preflight check chain.
  --diagnose  Diagnose last eval/retrain crash.
  --maintain  Triage agent-requested enhancements and classify crashes.
  --retest    Replay discarded hypotheses from a given SHA.

Usage:
  PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --verify --agent-name <name> --reported-combined <float>
  PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --diagnose
  PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --maintain --trigger=manual
  PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --retest <from-sha> [--dry-run]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent              # FIX: autoresearch/ is one level deep
BASELINE_PATH = SCRIPT_DIR / "baseline_metrics.json"
PROTECTED_FILES = [
    "splice/evaluate.py",
    "splice/program.md",
    "data/eval/*",
    "data/test/*",
    "autoresearch/manifest.json",
    "autoresearch/preflight.py",
]

# US-515 phase 1: read eval results from the unified JSONL log instead of
# parsing `combined: X.Y` prints from stdout. `RESULTS_TSV:` remains as a
# bash-wrapper carve-out until phase 2.
from autoresearch.log_reader import iter_events  # noqa: E402
from autoresearch.logger import get_logger  # noqa: E402


def _latest_combined_since(since_ts: str,
                           log_override: str | None = None) -> float | None:
    """Return the combined value from the newest `eval.aggregate` /
    `eval.single.combined` / `eval.metrics.splice` event with ts > since_ts.
    """
    env_override = os.environ.get("OMC_LOG_OVERRIDE")
    jsonl_path = None
    if log_override is not None:
        jsonl_path = Path(log_override)
    elif env_override:
        jsonl_path = Path(env_override)
    picked: dict | None = None
    for event_name in ("eval.aggregate", "eval.single.combined",
                       "eval.metrics.splice"):
        for rec in iter_events(event=event_name, since=since_ts,
                               jsonl_path=jsonl_path):
            if "combined" in rec:
                if picked is None or rec.get("ts", "") > picked.get("ts", ""):
                    picked = rec
    return float(picked["combined"]) if picked else None


def check_metric_rerun(reported: float) -> tuple[str, str, str]:
    """Re-run evaluate.py and compare combined score to reported value.

    Reads the fresh `combined` via `log_reader.iter_events()` from the
    unified JSONL log instead of re-parsing evaluate.py's stdout. The
    bash wrapper still relies on the `RESULTS_TSV:` string for its own
    parsing; that carve-out migrates in phase 2.
    """
    since_ts = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run(
            ["uv", "run", "python", "splice/evaluate.py", "--shap"],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return "FAIL", "splice/evaluate.py timed out after 300s", ""
    except Exception as e:
        return "FAIL", f"subprocess error: {e}", ""

    output = result.stdout + result.stderr
    if result.returncode != 0:
        return "FAIL", f"splice/evaluate.py exited {result.returncode}", output

    actual = _latest_combined_since(since_ts)
    if actual is None:
        return ("FAIL",
                "no eval.aggregate / eval.single.combined event in JSONL log "
                "since subprocess started",
                output)

    delta = abs(actual - reported)
    status = "PASS" if delta < 0.005 else "FAIL"
    return status, f"reported: {reported:.3f}, actual: {actual:.3f}, delta: {delta:.4f}", output


def check_git_diff_audit() -> tuple[str, str]:
    """Check that no protected files are modified.

    Covers THREE diff surfaces:
      (a) unstaged working-tree changes
      (b) staged-but-uncommitted changes
      (c) COMMITTED changes since the wrapper's captured head_before
          (US-511: catches the case where the agent committed a
          protected-file edit inside its hypothesis commit — invisible
          to (a)+(b) after commit).
    head_before read from env OMC_HEAD_BEFORE (wrapper exports it).
    Missing env → commit-level audit is skipped. WARN is returned in the
    otherwise-clean (a)+(b) case so the degraded audit is visible; any
    (a)+(b) protected-file hit still FAILs regardless of env. The
    retest subcommand (US-514) explicitly calls this without the
    wrapper's env wrapping, so the WARN surfaces the degraded-audit
    mode without blocking legitimate recoveries (main()'s WARN→MEDIUM
    rule is preserved).
    """
    try:
        unstaged = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
    except Exception as e:
        return "FAIL", f"git error: {e}"

    committed: list[str] = []
    head_before = os.environ.get("OMC_HEAD_BEFORE", "").strip()
    if head_before:
        try:
            committed = subprocess.run(
                ["git", "diff", "--name-only", f"{head_before}..HEAD"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
        except Exception as e:
            return "FAIL", f"git commit-level diff error: {e}"

    changed = set(unstaged + staged + committed)
    changed.discard("")

    violations = []
    for pattern in PROTECTED_FILES:
        if "*" in pattern:
            prefix = pattern.replace("*", "")
            for f in changed:
                if f.startswith(prefix):
                    violations.append(f)
        else:
            if pattern in changed:
                violations.append(pattern)

    if violations:
        return "FAIL", f"protected files modified: {violations}"
    if not head_before:
        return "WARN", "OMC_HEAD_BEFORE unset; commit-level audit skipped"
    return "PASS", "clean"


def check_anomaly(fresh_combined: float) -> tuple[str, str]:
    """Compare fresh combined score against baseline for anomalies."""
    if not BASELINE_PATH.exists():
        return "WARN", "baseline_metrics.json not found"

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    baseline_combined = baseline["combined"]
    delta = fresh_combined - baseline_combined
    abs_delta = abs(delta)

    if abs_delta > 0.15:
        return "WARN", f"delta: {delta:+.3f} from baseline {baseline_combined:.3f}"
    return "PASS", f"delta: {delta:+.3f} from baseline {baseline_combined:.3f}"


def check_preflight() -> tuple[str, str]:
    """Run preflight.py and check exit code."""
    preflight_path = SCRIPT_DIR / "preflight.py"
    if not preflight_path.exists():
        return "FAIL", "preflight.py not found"

    try:
        result = subprocess.run(
            ["uv", "run", "python", str(preflight_path)],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "FAIL", "preflight.py timed out"
    except Exception as e:
        return "FAIL", f"subprocess error: {e}"

    if result.returncode == 0:
        return "PASS", "preflight passed"
    return "FAIL", f"preflight exited {result.returncode}: {result.stdout.strip()}"


def check_clean_fp_bound(output: str) -> tuple[str, str]:
    """Check clean_fp bound per dataset AND total.

    Reads the RESULTS_TSV line (authoritative, written once per eval run)
    and verifies every per-dataset `clean_fp_<id>` ≤ 15 (per-dataset
    bound) and total `clean_fp` ≤ 45 (aggregate). Reports the worst
    offender with explicit attribution so a failing dataset is not hidden
    inside an aggregate.
    """
    tsv_line = None
    for line in output.splitlines():
        if line.startswith("RESULTS_TSV:"):
            tsv_line = line  # take last RESULTS_TSV line

    if tsv_line is None:
        # Fallback: older output shape. Look for the TP/FP summary line.
        m = re.search(r"clean_fp[=:]\s*(\d+)", output)
        if not m:
            return "WARN", "clean_fp not found in output (evaluate.py may have crashed)"
        cfp = int(m.group(1))
        if cfp > 15:
            return "FAIL", f"clean_fp={cfp} exceeds bound of 15"
        return "PASS", f"clean_fp={cfp} (bound: 15)"

    per_ds = {}
    for m in re.finditer(r"\bclean_fp_([A-Za-z_]+)=(\d+)", tsv_line):
        per_ds[m.group(1)] = int(m.group(2))
    total_match = re.search(r"\bclean_fp=(\d+)", tsv_line)
    total = int(total_match.group(1)) if total_match else sum(per_ds.values())

    per_ds_bound = 15
    total_bound = 45
    failures = [
        (ds, n) for ds, n in per_ds.items() if n > per_ds_bound
    ]
    if failures:
        worst = max(failures, key=lambda x: x[1])
        detail = ", ".join(f"{ds}={n}" for ds, n in sorted(per_ds.items()))
        return "FAIL", f"clean_fp_{worst[0]}={worst[1]} exceeds per-dataset bound {per_ds_bound} ({detail})"
    if total > total_bound:
        detail = ", ".join(f"{ds}={n}" for ds, n in sorted(per_ds.items()))
        return "FAIL", f"clean_fp total={total} exceeds aggregate bound {total_bound} ({detail})"

    detail = ", ".join(f"{ds}={n}" for ds, n in sorted(per_ds.items()))
    return "PASS", f"total={total} per-dataset({detail}) bounds {per_ds_bound}/ds, {total_bound} total"


# ---------------------------------------------------------------------------
# US-508 Phase 1: --diagnose subcommand
# ---------------------------------------------------------------------------
# On eval/retrain crash the wrapper invokes `supervisor_agent.py --diagnose`.
# The subcommand reads .omc/last_eval.log, finds the last Traceback, extracts
# exception class + message + top-3 frames, classifies into 6 categories,
# redacts oracle signals, and APPENDS a structured block to
# .omc/last_reflection.md (NOT overwrite — claude's hypothesis reflection from
# the same iteration lives there and must survive the wrapper's capture into
# research_notes.md). Completes in <1s on a 1 MB log.

_DIAGNOSE_LOG = ".omc/last_eval.log"
_DIAGNOSE_OUT = ".omc/last_reflection.md"
_DIAGNOSE_TOP_FRAMES = 3
_DIAGNOSE_TIME_BUDGET_S = 1.0
_DIAGNOSE_SEPARATOR = "\n\n---\n## [auto-diagnosis]\n"

# Denylist: oracle tokens that MUST NOT leak into claude's context via the
# diagnosis. Pattern anchors on token names; strips numeric RHS only. Accepts
# at most 3 _<domain> suffixes to cover combined_singing etc. Legitimate
# error floats (AssertionError: x=1.5 ...) are preserved because x, threshold,
# etc. are not in the denylist.
_ORACLE_DENYLIST_RE = re.compile(
    r"\b(combined(?:_[a-z]+)?|splice_f1|clean_score)\s*=\s*[0-9.]+"
)


def _redact_oracle(text: str) -> str:
    return _ORACLE_DENYLIST_RE.sub(r"\1=<REDACTED>", text)


def _extract_last_traceback(log_text: str) -> str | None:
    marker = "Traceback (most recent call last):"
    last = log_text.rfind(marker)
    if last == -1:
        return None
    # Walk forward until the first non-indented, non-blank line AFTER the
    # stack frames — that's the exception line. Everything through that
    # line is the traceback block.
    tail = log_text[last:]
    lines = tail.splitlines()
    block: list[str] = [lines[0]]  # the marker itself
    in_frames = True
    for ln in lines[1:]:
        block.append(ln)
        if in_frames and ln and not ln.startswith((" ", "\t")) and not ln.startswith("Traceback"):
            # This is the exception line — block ends here.
            break
    return "\n".join(block)


def _parse_traceback(tb_text: str) -> tuple[str, str, list[str]]:
    """Return (exception_class, message, top_frames[]). Graceful on malformed."""
    lines = tb_text.splitlines()
    frames: list[str] = []
    for ln in lines:
        m = re.search(r'File "([^"]+)", line (\d+)', ln)
        if m:
            frames.append(f"{m.group(1)}:{m.group(2)}")
    # Exception line: last non-blank line.
    exc_class = "Unknown"
    exc_msg = ""
    for ln in reversed(lines):
        s = ln.strip()
        if not s or s.startswith("File ") or s.startswith("Traceback"):
            continue
        # Form: "ExceptionClass: message"
        parts = s.split(":", 1)
        exc_class = parts[0].strip()
        exc_msg = parts[1].strip() if len(parts) > 1 else ""
        break
    return exc_class, exc_msg, frames[:_DIAGNOSE_TOP_FRAMES]


def _classify(exc_class: str, exc_msg: str) -> tuple[str, str]:
    """Return (category, suggested_action_note)."""
    if exc_class == "KeyError":
        # If the missing key name looks like a feature-cache slot missing
        # from _ensure_feat_cache, flag that specifically.
        key_match = re.search(r"'([^']+)'", exc_msg)
        key = key_match.group(1) if key_match else ""
        if key.startswith("feat_"):
            return (
                "KeyError-missing-ctx",
                f"Key '{key}' is referenced but never assigned. Add "
                f"ctx['{key}'] = ... in _ensure_feat_cache.",
            )
        return ("KeyError-other", f"Dict key '{key}' missing. Verify the key is populated before read.")
    if exc_class in ("SyntaxError", "IndentationError"):
        return (exc_class, "Python parse error. Fix syntax before next iteration.")
    if exc_class == "ImportError" or exc_class == "ModuleNotFoundError":
        return (exc_class, "Unresolved import. Check module path and spelling.")
    if exc_class == "NameError":
        return ("NameError", "Undefined identifier. Likely a typo or missing assignment.")
    if exc_class == "AssertionError":
        return ("AssertionError", "Runtime invariant violated. Review the assertion's preconditions.")
    return ("Other", f"{exc_class}: unexpected failure — read the traceback for next-step signal.")


def run_diagnose(log_path: str = _DIAGNOSE_LOG, out_path: str = _DIAGNOSE_OUT) -> int:
    import time
    t0 = time.monotonic()
    try:
        log_text = open(log_path).read()
    except OSError:
        # Log missing — still append a 2-line stub so the wrapper's
        # iteration-note pipeline sees *something*.
        with open(out_path, "a") as f:
            f.write(_DIAGNOSE_SEPARATOR + f"diagnose: log-missing\nnote: {log_path} not readable; nothing to classify.\n")
        print("diagnose: log-missing", flush=True)
        return 0

    tb = _extract_last_traceback(log_text)
    if tb is None:
        # US-510: no Python traceback — but evaluate.py's exception
        # handler may have absorbed per-domain failures, emitting
        # `combined_<domain>: ERROR (ExceptionClass: message)` lines
        # and collapsing the aggregate to the 0.01 GM floor. Root
        # cause of 4+ consecutive catastrophic discards this session
        # (HistGBM AttributeError on feature_importances_). Surface
        # these along with any DIAG WARN/ERROR and the RESULTS_TSV.
        silent_errs = []
        for line in log_text.splitlines():
            m = re.match(r"\s*combined_([a-z]+):\s*ERROR\s*\((.*)\)\s*$", line)
            if m:
                silent_errs.append((m.group(1), _redact_oracle(m.group(2))))
        diag_lines = [
            ln for ln in log_text.splitlines()
            if re.search(r"\bDIAG\b.*(WARN|ERROR)", ln)
        ][-3:]
        tsv_line = ""
        for ln in log_text.splitlines():
            if ln.startswith("RESULTS_TSV: "):
                tsv_line = _redact_oracle(ln)

        if silent_errs:
            block = ["diagnose: silent-per-domain-exception",
                     f"count: {len(silent_errs)} domain(s) with ERROR"]
            for dom, msg in silent_errs:
                block.append(f"  {dom}: {msg[:200]}")
            if diag_lines:
                block.append("diag_tail:")
                for d in diag_lines:
                    block.append(f"  {_redact_oracle(d)[:160]}")
            if tsv_line:
                block.append(f"tsv: {tsv_line}")
            block.append(
                "note: per-domain ERROR means evaluate.py's exception "
                "handler absorbed the failure; the aggregate combined "
                "collapsed to the GM floor (0.01). Your hypothesis "
                "likely worked — the SHAP/scoring code path broke. "
                "Inspect the exception class + message to locate the "
                "interface break (classifier attribute, shape mismatch, "
                "missing method).")
            with open(out_path, "a") as f:
                f.write(_DIAGNOSE_SEPARATOR + "\n".join(block) + "\n")
            print(f"diagnose: silent-per-domain-exception ({len(silent_errs)} domain)",
                  flush=True)
            return 0

        # No traceback, no per-domain ERROR — probably a normal
        # non-improvement discard or rate-limit/backoff. Emit terse stub.
        with open(out_path, "a") as f:
            f.write(_DIAGNOSE_SEPARATOR
                    + "diagnose: no-traceback\n"
                    + (f"tsv: {tsv_line}\n" if tsv_line else "")
                    + "note: eval log has no Python traceback; likely a "
                    + "normal discard. If combined collapsed below 0.05, "
                    + "check wrapper log for PIPELINE_FAILURE lines.\n")
        print("diagnose: no-traceback", flush=True)
        return 0

    tb_redacted = _redact_oracle(tb)
    exc_class, exc_msg, frames = _parse_traceback(tb_redacted)
    category, action_note = _classify(exc_class, exc_msg)

    block = [f"diagnose: {category}",
             f"exception: {exc_class}: {exc_msg}"[:300],
             "top_frames:"]
    for fr in frames:
        block.append(f"  {fr}")
    block.append(f"note: {action_note}")

    with open(out_path, "a") as f:
        f.write(_DIAGNOSE_SEPARATOR + "\n".join(block) + "\n")

    elapsed = time.monotonic() - t0
    if elapsed > _DIAGNOSE_TIME_BUDGET_S:
        print(f"diagnose: {category} (warning: {elapsed:.2f}s exceeds {_DIAGNOSE_TIME_BUDGET_S}s budget)",
              file=sys.stderr)
    print(f"diagnose: {category}", flush=True)
    return 0


def _diagnose_self_test() -> int:
    """Two synthetic cases pinning the denylist regex behavior."""
    import tempfile
    failures: list[str] = []

    # Case A: oracle tokens MUST be stripped.
    case_a_log = """\
Traceback (most recent call last):
  File "/a/b/evaluate.py", line 42, in run
    assert per_dataset["singing"] > 0, f"combined=0.47 splice_f1=0.33 clean_score=0.91 combined_singing=0.55"
AssertionError: combined=0.47 splice_f1=0.33 clean_score=0.91 combined_singing=0.55
"""
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as lf, \
         tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as of:
        lf.write(case_a_log); lf.flush()
        # Start with an existing claude-reflection so we can assert APPEND.
        of.write("hypothesis: CLAUDE WROTE THIS FIRST\n"); of.flush()
        a_log, a_out = lf.name, of.name
    run_diagnose(a_log, a_out)
    out_text = open(a_out).read()
    for tok in ("combined=0.47", "splice_f1=0.33", "clean_score=0.91", "combined_singing=0.55"):
        if tok in out_text:
            failures.append(f"Case A: oracle token '{tok}' not redacted")
    for tok in ("combined=<REDACTED>", "splice_f1=<REDACTED>",
                "clean_score=<REDACTED>", "combined_singing=<REDACTED>"):
        if tok not in out_text:
            failures.append(f"Case A: expected redacted marker '{tok}' missing")
    if "CLAUDE WROTE THIS FIRST" not in out_text:
        failures.append("Case A: APPEND broken — claude's prior reflection lost")
    if "[auto-diagnosis]" not in out_text:
        failures.append("Case A: missing [auto-diagnosis] separator")

    # Case B: legitimate error floats MUST survive.
    case_b_log = """\
Traceback (most recent call last):
  File "/c/d/detector.py", line 100, in check
    assert x < 0.5
AssertionError: x=1.5 not in (0, 1)

Traceback (most recent call last):
  File "/c/d/detector.py", line 200, in gate
    raise ValueError(f"threshold=0.72 outside [0, 1]")
ValueError: threshold=0.72 outside [0, 1]
"""
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as lf, \
         tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as of:
        lf.write(case_b_log); lf.flush()
        b_log, b_out = lf.name, of.name
    run_diagnose(b_log, b_out)
    out_text_b = open(b_out).read()
    for tok in ("threshold=0.72",):  # last traceback's message
        if tok not in out_text_b:
            failures.append(f"Case B: legitimate float '{tok}' incorrectly stripped")

    if failures:
        print("--self-test FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("--self-test: PASS (Case A oracle-strip + Case B float-preserve + APPEND)")
    return 0


# ---------------------------------------------------------------------------
# US-514: --retest <from-sha> subcommand
# ---------------------------------------------------------------------------
# Systematically replay discards committed after <from-sha> against a fresh
# evaluator, recover any whose current combined beats the rolling baseline.
# Worktree-isolated replay + disk-sourced baseline + SIGKILL sentinel.
# Plan: .omc/plans/ralplan-retest-discards.md
# ---------------------------------------------------------------------------

import atexit  # noqa: E402  (intentional late import for retest-only paths)
import datetime as _dt  # noqa: E402
import shutil  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
_RETEST_SENTINEL = PROJECT_DIR / ".omc" / "retest-in-progress"
_RETEST_WORKTREE = PROJECT_DIR / ".omc" / "retest-worktree"
_RETEST_REPORT = PROJECT_DIR / ".omc" / "retest-report.md"
_RESULTS_TSV = PROJECT_DIR / "results.tsv"
_WRAPPER_SH = PROJECT_DIR / "run_autoresearch.sh"

_VALID_OUTCOMES = {
    "recovered",
    "still-lower",
    "conflict",
    "eval-crash",
    "retrain-crash",
    "verify-fail",
    "missing",
    "corpus-purged",
    "dry-run-skipped",
}


@dataclass
class RetestCandidate:
    sha: str
    original_combined: float | None
    commit_time: int | None
    row_index: int
    description: str
    retest_combined: float | None = None
    outcome: str = ""
    note: str = ""
    eval_output: str = ""


def _utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _current_baseline() -> float:
    """Re-read baseline_metrics.json['combined'] from disk on every call.

    Deliberately NOT cached: the rolling baseline is updated by _do_keep_path
    inside a different process (`bash run_autoresearch.sh _keep_path`). A
    crash between the cherry-pick and the baseline commit leaves the disk
    in a known state that a next --retest invocation can read directly;
    there is no in-memory scalar to desync.
    """
    if not BASELINE_PATH.exists():
        return 0.0
    try:
        with open(BASELINE_PATH) as f:
            data = json.load(f)
        return float(data.get("combined", 0.0))
    except Exception:
        return 0.0


def _tmux_session_alive(name: str) -> bool:
    """Return True iff `tmux has-session -t <name>` exits 0. False on missing tmux."""
    try:
        rc = subprocess.run(
            ["tmux", "has-session", "-t", name],
            capture_output=True, text=True, timeout=5,
        ).returncode
        return rc == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _eval_root_ok() -> tuple[bool, str]:
    root = os.environ.get("OMC_EVAL_DATA_ROOT", "").strip()
    if not root:
        return False, ""
    p = Path(root)
    # Architect follow-up: detect empty directories as stale (APFS purge
    # can leave a hollow mount-point or the operator may have shredded
    # the decrypted tree without unsetting the env). `.exists()` alone
    # is not enough.
    if not p.exists():
        return False, root
    try:
        if not any(p.iterdir()):
            return False, root
    except (PermissionError, OSError):
        return False, root
    return True, root


def _require_eval_env() -> None:
    """Refuse to continue when OMC_EVAL_DATA_ROOT is unset or points at a
    missing / empty directory. Retest never re-decrypts; operator owns
    the eval corpus lifecycle.
    """
    ok, root = _eval_root_ok()
    if ok:
        return
    root_disp = root or "(unset)"
    print(
        f"ERROR: OMC_EVAL_DATA_ROOT is unset or stale ({root_disp}).\n"
        "  Retest does NOT re-decrypt. Set it up manually:\n"
        "    uv run python scripts/eval_crypto.py decrypt --keep\n"
        "    export OMC_EVAL_DATA_ROOT=<path-from-decrypt-tail>\n"
        "  Then re-invoke: --retest <sha> [--from-sha <last-recovered-sha>]",
        file=sys.stderr,
    )
    sys.exit(2)


def _check_corpus_alive(next_sha: str) -> None:
    """Called between every candidate. On miss, emits a single stderr line
    naming the next-candidate sha as the cutoff and exits 1.
    """
    ok, root = _eval_root_ok()
    if ok:
        return
    root_disp = root or "(unset)"
    print(
        f"ERROR: OMC_EVAL_DATA_ROOT ({root_disp}) no longer exists "
        "(APFS purge likely).\n"
        f"  Batch abort; cutoff candidate: {next_sha}\n"
        "  Re-decrypt and resume with: --retest <sha> "
        "--from-sha <last-recovered-sha>",
        file=sys.stderr,
    )
    sys.exit(1)


def _refuse_if_sentinel() -> None:
    if _RETEST_SENTINEL.exists():
        print(
            f"ERROR: {_RETEST_SENTINEL.relative_to(PROJECT_DIR)} sentinel found — "
            "a prior retest crashed mid-recovery.\n"
            f"  Inspect: {_RETEST_REPORT.relative_to(PROJECT_DIR)} "
            "and `git log --oneline -10`.\n"
            "  Recover: verify HEAD (cherry-pick may be partial), then\n"
            f"           rm -f {_RETEST_SENTINEL.relative_to(PROJECT_DIR)}",
            file=sys.stderr,
        )
        sys.exit(2)


def _git_commit_time(sha: str) -> int | None:
    """Return the commit timestamp for <sha> as an int, or None if unknown."""
    try:
        r = subprocess.run(
            ["git", "show", "-s", "--format=%ct", sha],
            capture_output=True, text=True, timeout=10, cwd=PROJECT_DIR,
        )
        if r.returncode != 0:
            return None
        return int(r.stdout.strip())
    except Exception:
        return None


def _git_rev_parse(sha: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", sha],
            capture_output=True, text=True, timeout=10, cwd=PROJECT_DIR,
        )
        if r.returncode != 0:
            return None
        return r.stdout.strip() or None
    except Exception:
        return None


def _load_retest_candidates(results_tsv: Path, from_sha: str) -> list[RetestCandidate]:
    """Walk results.tsv. Keep rows where status='discard' and commit_time >
    from_sha's commit_time. Capture row_index (0-based, post-header) as a
    stable tiebreaker for %ct collisions. Missing-SHA rows → outcome='missing'.
    """
    from_time = _git_commit_time(from_sha)
    if from_time is None:
        print(
            f"ERROR: --retest from-sha {from_sha!r} not found in git "
            "(or the repo is not a git checkout).",
            file=sys.stderr,
        )
        sys.exit(1)

    if not results_tsv.exists():
        print(
            f"ERROR: {results_tsv} not found — no discards to replay.",
            file=sys.stderr,
        )
        sys.exit(1)

    candidates: list[RetestCandidate] = []
    with open(results_tsv) as f:
        lines = f.readlines()
    if not lines:
        return []
    # Header: assume first line is header.
    for row_index, line in enumerate(lines[1:]):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 14:
            continue
        commit = fields[0].strip()
        combined_raw = fields[1].strip()
        status = fields[12].strip()
        description = fields[13].strip()
        if status != "discard":
            continue
        try:
            original_combined = float(combined_raw) if combined_raw not in ("", "NA") else None
        except ValueError:
            original_combined = None

        ct = _git_commit_time(commit)
        if ct is None:
            # Row references a SHA not in git. Still include with outcome='missing'
            # so the report names it.
            candidates.append(RetestCandidate(
                sha=commit,
                original_combined=original_combined,
                commit_time=None,
                row_index=row_index,
                description=description,
                outcome="missing",
                note="commit not found in git (squashed/gc'd?)",
            ))
            continue
        if ct <= from_time:
            continue
        candidates.append(RetestCandidate(
            sha=commit,
            original_combined=original_combined,
            commit_time=ct,
            row_index=row_index,
            description=description,
        ))

    # Stable sort: Python's sort is stable, so (commit_time, row_index) is a
    # deterministic ordering even when candidates share %ct.
    # Missing-SHA rows (commit_time=None) sort to the END so recoverable
    # candidates are tried first. The report still names them.
    def sort_key(c: RetestCandidate) -> tuple[int, int, int]:
        # (bucket, commit_time-or-0, row_index). bucket=0 for known, 1 for missing.
        bucket = 0 if c.commit_time is not None else 1
        return (bucket, c.commit_time or 0, c.row_index)

    candidates.sort(key=sort_key)
    return candidates


_WORKTREE_ATEXIT_REGISTERED = False


def _worktree_setup() -> None:
    """Create a detached worktree at .omc/retest-worktree pointing at HEAD.

    Idempotent-ish: if the worktree path already exists (abandoned crash),
    remove first then recreate. atexit cleanup is registered exactly once.
    """
    global _WORKTREE_ATEXIT_REGISTERED
    if _RETEST_WORKTREE.exists():
        # Leftover from an abandoned run. Try graceful removal first.
        _worktree_cleanup()
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(_RETEST_WORKTREE), "HEAD"],
            check=True, capture_output=True, text=True, timeout=30,
            cwd=PROJECT_DIR,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: git worktree add failed: {e.stderr or e.stdout}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _WORKTREE_ATEXIT_REGISTERED:
        atexit.register(_worktree_cleanup)
        _WORKTREE_ATEXIT_REGISTERED = True


def _worktree_cleanup() -> None:
    """Best-effort worktree teardown. Idempotent. Does NOT delete the
    sentinel file — sentinel deletion is only on the one happy path in
    _try_recover so SIGKILL / crash paths leave the sentinel for the
    operator to inspect.
    """
    # Best-effort abort any in-progress cherry-pick inside the worktree.
    if _RETEST_WORKTREE.exists():
        try:
            subprocess.run(
                ["git", "-C", str(_RETEST_WORKTREE), "cherry-pick", "--abort"],
                capture_output=True, text=True, timeout=15,
            )
        except Exception:
            pass
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(_RETEST_WORKTREE)],
                capture_output=True, text=True, timeout=30, cwd=PROJECT_DIR,
            )
        except Exception:
            pass
        # If git refused, fall back to rmtree so repeat runs don't inherit a
        # partial worktree directory.
        if _RETEST_WORKTREE.exists():
            shutil.rmtree(_RETEST_WORKTREE, ignore_errors=True)


def _parse_combined_from_eval(output: str) -> float | None:
    """Parse the last RESULTS_TSV: line for a combined=<float> token."""
    tsv_line = None
    for line in output.splitlines():
        if line.startswith("RESULTS_TSV:"):
            tsv_line = line
    if tsv_line is None:
        return None
    m = re.search(r"\bcombined=([0-9.]+)", tsv_line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _replay_one(cand: RetestCandidate, worktree_base: str) -> None:
    """Inside the worktree: cherry-pick <sha>, ensure classifier fresh,
    run evaluate.py, parse combined. Sets cand.retest_combined and
    cand.outcome. Always resets the worktree back to worktree_base
    before returning so the next candidate picks onto a clean tree.
    """
    wt = str(_RETEST_WORKTREE)

    # Cherry-pick into the worktree.
    cp = subprocess.run(
        ["git", "-C", wt, "cherry-pick", cand.sha],
        capture_output=True, text=True, timeout=60,
    )
    if cp.returncode != 0:
        # Abort and mark conflict. Status for diagnosis.
        status = subprocess.run(
            ["git", "-C", wt, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        subprocess.run(
            ["git", "-C", wt, "cherry-pick", "--abort"],
            capture_output=True, text=True, timeout=15,
        )
        cand.outcome = "conflict"
        cand.note = ("cherry-pick conflict:\n" + status)[:400]
        _reset_worktree(worktree_base)
        return

    # Classifier staleness gate via the wrapper verb (runs inside main repo
    # so `splice/classifier/fp_classifier.*` gets updated in-place — which is
    # what splice/evaluate.py inside the worktree will read via the classifier
    # path). The verb does not amend or commit onto the worktree.
    retrain = subprocess.run(
        ["bash", str(_WRAPPER_SH), "_ensure_classifier_fresh"],
        capture_output=True, text=True, timeout=360,
        cwd=PROJECT_DIR,
    )
    if retrain.returncode != 0:
        cand.outcome = "retrain-crash"
        cand.note = (retrain.stderr or retrain.stdout or "")[:400]
        _reset_worktree(worktree_base)
        return

    # Run evaluate.py inside the worktree. OMC_EVAL_DATA_ROOT and
    # OMC_FEATURE_CACHE_DIR are inherited from the calling shell.
    env = os.environ.copy()
    # Propagate features.py sha for the cache key like the loop does.
    try:
        feat_sha = subprocess.run(
            ["git", "-C", wt, "hash-object", "splice/features.py"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        feat_sha = "unknown"
    env["OMC_FEATURES_PY_SHA"] = feat_sha
    if "OMC_FEATURE_CACHE_DIR" not in env:
        env["OMC_FEATURE_CACHE_DIR"] = str(PROJECT_DIR / ".omc" / "feature_cache")

    try:
        eval_res = subprocess.run(
            ["uv", "run", "python", "splice/evaluate.py", "--shap"],
            capture_output=True, text=True, timeout=600,
            cwd=wt, env=env,
        )
    except subprocess.TimeoutExpired:
        cand.outcome = "eval-crash"
        cand.note = "splice/evaluate.py timed out after 600s"
        _reset_worktree(worktree_base)
        return
    except Exception as e:
        cand.outcome = "eval-crash"
        cand.note = f"evaluate.py subprocess error: {e}"
        _reset_worktree(worktree_base)
        return

    output = eval_res.stdout + eval_res.stderr
    cand.eval_output = output
    if eval_res.returncode != 0:
        cand.outcome = "eval-crash"
        tail = "\n".join(output.splitlines()[-6:])
        cand.note = (f"splice/evaluate.py exited {eval_res.returncode}\n" + tail)[:400]
        _reset_worktree(worktree_base)
        return

    combined = _parse_combined_from_eval(output)
    if combined is None:
        cand.outcome = "eval-crash"
        cand.note = "could not parse combined from RESULTS_TSV line"
        _reset_worktree(worktree_base)
        return

    cand.retest_combined = combined
    # Leave cand.outcome unset here; the recovery decision branch fills it.
    _reset_worktree(worktree_base)


def _reset_worktree(base: str) -> None:
    """git reset --hard inside the retest worktree so the next cherry-pick
    applies to a clean tree. No-op if the worktree is gone.
    """
    if not _RETEST_WORKTREE.exists():
        return
    try:
        subprocess.run(
            ["git", "-C", str(_RETEST_WORKTREE), "reset", "--hard", base],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        pass


def _try_recover(cand: RetestCandidate, *, re_eval: bool) -> bool:
    """Main-tree recovery path. Returns True iff the candidate became a
    new baseline. Sentinel write happens immediately before cherry-pick
    and delete happens only after the baseline: commit lands.
    """
    baseline = _current_baseline()
    if cand.retest_combined is None or cand.retest_combined <= baseline:
        cand.outcome = "still-lower"
        cand.note = (
            f"retest combined={cand.retest_combined!r} "
            f"<= baseline {baseline:.6f}"
        )
        return False

    # Write sentinel BEFORE the cherry-pick. Crash anywhere in this window
    # leaves the sentinel on disk; the wrapper refuses to start the loop
    # until the operator inspects + rm -fs it.
    try:
        _RETEST_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        _RETEST_SENTINEL.write_text(
            json.dumps({"sha": cand.sha, "started_at": _utc_iso()})
        )
    except Exception as e:
        cand.outcome = "verify-fail"
        cand.note = f"sentinel write failed: {e}"
        return False

    # Cherry-pick into the main tree. -x stamps the origin into the commit
    # message.
    cp = subprocess.run(
        ["git", "cherry-pick", "-x", cand.sha],
        capture_output=True, text=True, timeout=60, cwd=PROJECT_DIR,
    )
    if cp.returncode != 0:
        # Abort and remove sentinel — we never crossed the mutation window.
        subprocess.run(
            ["git", "cherry-pick", "--abort"],
            capture_output=True, text=True, timeout=15, cwd=PROJECT_DIR,
        )
        try:
            _RETEST_SENTINEL.unlink(missing_ok=True)
        except Exception:
            pass
        cand.outcome = "conflict"
        cand.note = (
            f"main-tree cherry-pick conflict: {cp.stderr or cp.stdout}"
        )[:400]
        return False

    # Structural checks. Re-use the worktree's eval output unless caller
    # asked for a fresh re-eval against the main tree.
    if re_eval:
        m_status, m_detail, m_output = check_metric_rerun(cand.retest_combined)
        metric_output = m_output
    else:
        m_status, m_detail = "PASS", "reused worktree eval output"
        metric_output = cand.eval_output

    d_status, d_detail = check_git_diff_audit()
    a_status, a_detail = check_anomaly(cand.retest_combined)
    p_status, p_detail = check_preflight()
    f_status, f_detail = check_clean_fp_bound(metric_output)

    all_statuses = [m_status, d_status, a_status, p_status, f_status]
    if any(s == "FAIL" for s in all_statuses):
        confidence = "LOW"
    elif any(s == "WARN" for s in all_statuses):
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    check_summary = (
        f"metric={m_status}({m_detail}) "
        f"diff={d_status}({d_detail}) "
        f"anomaly={a_status}({a_detail}) "
        f"preflight={p_status}({p_detail}) "
        f"fp_bound={f_status}({f_detail})"
    )

    if confidence == "LOW":
        # Roll back the cherry-pick and remove the sentinel.
        subprocess.run(
            ["git", "reset", "--hard", "ORIG_HEAD"],
            capture_output=True, text=True, timeout=15, cwd=PROJECT_DIR,
        )
        try:
            _RETEST_SENTINEL.unlink(missing_ok=True)
        except Exception:
            pass
        cand.outcome = "verify-fail"
        cand.note = (
            f"confidence=LOW; rolled back. {check_summary}"
        )[:800]
        return False

    # HIGH or MEDIUM → run the wrapper's keep path.
    hypothesis_commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, timeout=5, cwd=PROJECT_DIR,
    ).stdout.strip()
    hypothesis_subject = subprocess.run(
        ["git", "log", "-1", "--format=%s", "HEAD"],
        capture_output=True, text=True, timeout=5, cwd=PROJECT_DIR,
    ).stdout.strip()
    # Strip any `hypothesis: ` prefix for parity with loop's keep path.
    if hypothesis_subject.startswith("hypothesis: "):
        hypothesis_subject = hypothesis_subject[len("hypothesis: ") :]

    baseline_str = f"{baseline:.6f}"

    keep_rc = subprocess.run(
        [
            "bash", str(_WRAPPER_SH), "_keep_path",
            hypothesis_commit,
            f"{cand.retest_combined:.6f}",
            baseline_str,
            hypothesis_subject,
            "--retest-origin", cand.sha,
        ],
        capture_output=True, text=True, timeout=120,
        cwd=PROJECT_DIR,
    )
    if keep_rc.returncode != 0:
        # Try to roll back; sentinel stays.
        cand.outcome = "verify-fail"
        cand.note = (
            f"_keep_path exited {keep_rc.returncode}: "
            f"{(keep_rc.stderr or keep_rc.stdout)[:300]}"
        )
        return False

    # Confirm the baseline commit landed before removing the sentinel.
    head_subj = subprocess.run(
        ["git", "log", "-1", "--format=%s", "HEAD"],
        capture_output=True, text=True, timeout=5, cwd=PROJECT_DIR,
    ).stdout.strip()
    # _append_note runs AFTER the baseline commit, so HEAD may be a
    # note: commit. Accept either 'baseline:' or 'note:' (which implies
    # baseline: came just before).
    if not (head_subj.startswith("baseline:") or head_subj.startswith("note:")):
        cand.outcome = "verify-fail"
        cand.note = (
            f"_keep_path returned 0 but HEAD subject is {head_subj!r} "
            f"(expected baseline: or note:)"
        )
        return False

    try:
        _RETEST_SENTINEL.unlink(missing_ok=True)
    except Exception:
        pass
    cand.outcome = "recovered"
    cand.note = f"confidence={confidence}; {check_summary}"[:800]
    return True


def _write_retest_report(path: Path, candidates: list[RetestCandidate],
                         from_sha: str, *, dry_run: bool, limit: int | None,
                         baseline_at_start: float, batch_cutoff_sha: str | None) -> None:
    counts = {k: 0 for k in _VALID_OUTCOMES}
    for c in candidates:
        if c.outcome in counts:
            counts[c.outcome] += 1

    lines: list[str] = []
    lines.append(f"# Retest report — from-sha={from_sha}")
    lines.append("")
    lines.append(f"- invocation: `--retest {from_sha}"
                 + (" --dry-run" if dry_run else "")
                 + (f" --limit {limit}" if limit else "")
                 + "`")
    lines.append(f"- timestamp: {_utc_iso()}")
    lines.append(f"- baseline_at_start: {baseline_at_start:.6f}")
    lines.append(f"- dry_run: {dry_run}")
    if batch_cutoff_sha:
        lines.append(f"- batch_cutoff_sha (corpus-purged): {batch_cutoff_sha}")
    lines.append("")
    lines.append("| sha | original_combined | retest_combined | delta | outcome | note |")
    lines.append("|-----|-------------------|-----------------|-------|---------|------|")
    for c in candidates:
        orig = f"{c.original_combined:.6f}" if c.original_combined is not None else "NA"
        if dry_run and c.original_combined is not None:
            # In dry-run, "predicted" column semantics = original - baseline_at_start.
            predicted = c.original_combined - baseline_at_start
            retest = f"(predicted delta {predicted:+.6f})"
            delta = f"{predicted:+.6f}"
        else:
            retest = f"{c.retest_combined:.6f}" if c.retest_combined is not None else "NA"
            if c.retest_combined is not None and c.original_combined is not None:
                delta = f"{c.retest_combined - c.original_combined:+.6f}"
            else:
                delta = "NA"
        note = (c.note or "").replace("\n", " ").replace("|", "\\|")
        if len(note) > 200:
            note = note[:197] + "..."
        lines.append(
            f"| {c.sha} | {orig} | {retest} | {delta} | {c.outcome or '-'} | {note} |"
        )
    lines.append("")
    summary = " / ".join(f"{k}={counts[k]}" for k in sorted(counts))
    lines.append(f"Summary: {summary}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run_retest(from_sha: str, *, dry_run: bool = False, limit: int | None = None,
               re_eval: bool = False) -> int:
    """Orchestrate a --retest batch. Returns process exit code."""
    # Guards (all exit 2 on refusal to distinguish from FAIL outcomes).
    _refuse_if_sentinel()
    if _tmux_session_alive("autoresearch"):
        print(
            "ERROR: tmux session 'autoresearch' is alive. Retest shares the "
            "eval corpus lifecycle and the baseline_metrics.json writer "
            "with the loop.\n"
            "  Remedy: ./run_autoresearch.sh stop  (wait for clean exit), "
            "then re-invoke --retest.",
            file=sys.stderr,
        )
        sys.exit(2)
    # --dry-run skips every mutating operation including the sentinel and
    # worktree, but still wants the corpus guards to surface
    # configuration mistakes to the operator.
    _require_eval_env()
    # Resolve from_sha early for nicer error output.
    rp = _git_rev_parse(from_sha)
    if rp is None:
        print(
            f"ERROR: --retest from-sha {from_sha!r} not resolvable by git.",
            file=sys.stderr,
        )
        sys.exit(1)

    candidates = _load_retest_candidates(_RESULTS_TSV, from_sha)
    baseline_at_start = _current_baseline()
    batch_cutoff_sha: str | None = None

    if dry_run:
        for c in candidates:
            if c.outcome == "missing":
                continue
            c.outcome = "dry-run-skipped"
            if c.original_combined is not None:
                delta = c.original_combined - baseline_at_start
                c.note = f"predicted delta {delta:+.6f} vs baseline {baseline_at_start:.6f}"
            else:
                c.note = "no original_combined recorded in TSV"
        _write_retest_report(
            _RETEST_REPORT, candidates, from_sha,
            dry_run=True, limit=limit,
            baseline_at_start=baseline_at_start,
            batch_cutoff_sha=None,
        )
        print(f"--retest --dry-run: {len(candidates)} candidates; "
              f"report={_RETEST_REPORT.relative_to(PROJECT_DIR)}")
        return 0

    # Non-dry-run: worktree + per-candidate replay + recovery.
    processable = [c for c in candidates if c.outcome == ""]
    if limit is not None:
        processable = processable[:limit]

    if not processable:
        print(f"--retest: no replayable candidates after {from_sha}; "
              f"writing report anyway.")
        _write_retest_report(
            _RETEST_REPORT, candidates, from_sha,
            dry_run=False, limit=limit,
            baseline_at_start=baseline_at_start,
            batch_cutoff_sha=None,
        )
        return 0

    worktree_base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=5, cwd=PROJECT_DIR,
    ).stdout.strip()

    _worktree_setup()

    any_fail = False
    for cand in processable:
        _check_corpus_alive(cand.sha)
        # Worktree may need a fresh base if a prior candidate recovered
        # (main-tree HEAD advanced via _keep_path, so we rebase the
        # worktree to the new HEAD so future cherry-picks pick onto the
        # accepted baseline).
        current_main = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=PROJECT_DIR,
        ).stdout.strip()
        if current_main != worktree_base:
            # Reset worktree to current main HEAD.
            subprocess.run(
                ["git", "fetch", str(PROJECT_DIR), "HEAD"],
                capture_output=True, text=True, timeout=15,
                cwd=str(_RETEST_WORKTREE),
            )
            _reset_worktree(current_main)
            worktree_base = current_main

        _replay_one(cand, worktree_base)
        if cand.outcome in ("conflict", "eval-crash", "retrain-crash"):
            any_fail = True
            continue
        # If retest produced a combined score, decide recovery vs still-lower.
        _try_recover(cand, re_eval=re_eval)
        if cand.outcome == "verify-fail":
            any_fail = True

    # Final report write.
    _write_retest_report(
        _RETEST_REPORT, candidates, from_sha,
        dry_run=False, limit=limit,
        baseline_at_start=baseline_at_start,
        batch_cutoff_sha=batch_cutoff_sha,
    )
    print(f"--retest: processed {len(processable)} candidate(s); "
          f"report={_RETEST_REPORT.relative_to(PROJECT_DIR)}")

    return 1 if any_fail else 0


# ---------------------------------------------------------------------------
# --retest-self-test harness
# ---------------------------------------------------------------------------

def _retest_self_test() -> int:
    """End-to-end fixture: build a throwaway git repo with a stubbed
    evaluate.py and a synthetic results.tsv, then invoke the retest flow
    in dry-run mode and assert the report matches expectations.

    A non-dry-run integration test would require a full eval corpus; the
    dry-run gate exercises enumeration, ordering, report write, and the
    no-mutation invariant.
    """
    import tempfile

    failures: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="retest-selftest-"))
    try:
        # Fixture: init a git repo, seed a few commits, seed results.tsv.
        def _run(*cmd: str, cwd: Path = tmp, env=None) -> subprocess.CompletedProcess:
            e = os.environ.copy()
            e["GIT_AUTHOR_NAME"] = "retest-selftest"
            e["GIT_AUTHOR_EMAIL"] = "retest@example.invalid"
            e["GIT_COMMITTER_NAME"] = "retest-selftest"
            e["GIT_COMMITTER_EMAIL"] = "retest@example.invalid"
            if env:
                e.update(env)
            return subprocess.run(
                cmd, cwd=cwd, env=e, check=True,
                capture_output=True, text=True, timeout=30,
            )

        _run("git", "init", "-q")
        (tmp / "seed.txt").write_text("seed\n")
        _run("git", "add", "seed.txt")
        # Anchor commit (will be the from-sha).
        env0 = {"GIT_COMMITTER_DATE": "2026-01-01T00:00:00"}
        _run("git", "commit", "-q", "-m", "seed", env=env0)
        from_sha = _run("git", "rev-parse", "HEAD").stdout.strip()

        # Two discard-class commits at distinct %ct; one at a shared %ct
        # with the third (tests tiebreaker).
        def _make_commit(path: str, content: str, subject: str, ct: str) -> str:
            (tmp / path).write_text(content)
            _run("git", "add", path)
            _run("git", "commit", "-q", "-m", subject,
                 env={"GIT_COMMITTER_DATE": ct})
            return _run("git", "rev-parse", "HEAD").stdout.strip()

        sha_a = _make_commit("a.txt", "A\n", "hypothesis: A",
                             "2026-02-01T00:00:00")
        # Two commits sharing the SAME second (row-index tiebreaker test).
        sha_b = _make_commit("b.txt", "B\n", "hypothesis: B",
                             "2026-03-01T00:00:00")
        sha_c = _make_commit("c.txt", "C\n", "hypothesis: C",
                             "2026-03-01T00:00:00")

        # Results.tsv: header + 3 discard rows referencing the 3 SHAs.
        # Schema matches results.tsv as logged by run_autoresearch.sh:
        # 14 columns.
        tsv_lines = [
            "\t".join([
                "commit", "combined", "combined_mean", "combined_min",
                "clean_fp", "n_datasets", "combined_singing",
                "combined_korean", "combined_english", "clean_fp_singing",
                "clean_fp_korean", "clean_fp_english", "status", "description",
            ]),
        ]
        def _row(sha: str, combined: str, desc: str) -> str:
            return "\t".join([
                sha[:7], combined, combined, combined, "0", "3",
                combined, combined, combined, "0", "0", "0",
                "discard", desc,
            ])
        # C logged BEFORE B despite sharing %ct — tests stable sort preserves
        # append-order tiebreak.
        tsv_lines.append(_row(sha_a, "0.300000", "candidate A"))
        tsv_lines.append(_row(sha_c, "0.450000", "candidate C"))
        tsv_lines.append(_row(sha_b, "0.350000", "candidate B"))
        (tmp / "results.tsv").write_text("\n".join(tsv_lines) + "\n")

        # Mock baseline fixture — BASELINE_PATH is rebound below so the
        # tmp location is arbitrary; we use `autoresearch/` for parity
        # with the post-US-516 live layout.
        bl_dir = tmp / "autoresearch"
        bl_dir.mkdir(parents=True, exist_ok=True)
        (bl_dir / "baseline_metrics.json").write_text(
            json.dumps({"combined": 0.5, "timestamp": _utc_iso()})
        )

        # Validate _load_retest_candidates ordering (tiebreaker).
        # We rebind module-level paths on THIS module directly; the fixture
        # only mutates pointer attributes, which _retest_self_test itself
        # is responsible for restoring (finally-block cleanup via the
        # tmp directory removal).
        module = sys.modules[__name__]
        orig_project_dir = module.PROJECT_DIR
        orig_results_tsv = module._RESULTS_TSV
        orig_baseline_path = module.BASELINE_PATH
        orig_commit_time_fn = module._git_commit_time

        module.PROJECT_DIR = tmp
        module._RESULTS_TSV = tmp / "results.tsv"
        module.BASELINE_PATH = bl_dir / "baseline_metrics.json"

        # _git_commit_time uses cwd=PROJECT_DIR — redirect by patching
        # so the fixture repo is queried even though the real
        # PROJECT_DIR constant is bound at import time in the helper.
        def _patched_ct(sha: str) -> int | None:
            try:
                r = subprocess.run(
                    ["git", "show", "-s", "--format=%ct", sha],
                    capture_output=True, text=True, timeout=10, cwd=tmp,
                )
                if r.returncode != 0:
                    return None
                return int(r.stdout.strip())
            except Exception:
                return None
        module._git_commit_time = _patched_ct

        cands = module._load_retest_candidates(tmp / "results.tsv", from_sha)

        # Expected ordering: A (earliest), C (row_index 1 at shared %ct),
        # B (row_index 2 at shared %ct).
        got = [c.sha[:7] for c in cands]
        expected = [sha_a[:7], sha_c[:7], sha_b[:7]]
        if got != expected:
            failures.append(
                f"candidate ordering: expected {expected}, got {got}"
            )

        # Test _current_baseline reads disk fresh.
        b1 = module._current_baseline()
        if abs(b1 - 0.5) > 1e-9:
            failures.append(f"_current_baseline got {b1}, expected 0.5")
        (bl_dir / "baseline_metrics.json").write_text(
            json.dumps({"combined": 0.6})
        )
        b2 = module._current_baseline()
        if abs(b2 - 0.6) > 1e-9:
            failures.append(
                f"_current_baseline did not re-read from disk (got {b2}, expected 0.6)"
            )

        # Test WARN on missing OMC_HEAD_BEFORE with clean tree.
        _run("git", "checkout", "-q", "HEAD", "--", ".")
        # Use real main tree's check_git_diff_audit — we need a clean tree.
        # Running in tmp where no protected files exist; WARN expected when
        # env unset.
        prev_hb = os.environ.pop("OMC_HEAD_BEFORE", None)
        try:
            # Temporarily chdir so `git diff` runs against the fixture.
            prev_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                status, _ = module.check_git_diff_audit()
            finally:
                os.chdir(prev_cwd)
        finally:
            if prev_hb is not None:
                os.environ["OMC_HEAD_BEFORE"] = prev_hb
        if status != "WARN":
            failures.append(
                f"check_git_diff_audit with unset env on clean tree: "
                f"expected WARN, got {status}"
            )

        # Test dry-run no-mutation invariants.
        tmp_sentinel = tmp / ".omc" / "retest-in-progress"
        tmp_worktree = tmp / ".omc" / "retest-worktree"
        module._RETEST_SENTINEL = tmp_sentinel
        module._RETEST_WORKTREE = tmp_worktree
        module._RETEST_REPORT = tmp / ".omc" / "retest-report.md"
        module._WRAPPER_SH = PROJECT_DIR / "run_autoresearch.sh"  # real wrapper

        # Patch eval-env check so dry-run reaches the report writer.
        def _ok_eval():
            return True, str(tmp)
        module._eval_root_ok = _ok_eval
        # Skip tmux check.
        module._tmux_session_alive = lambda _name: False

        prev_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            rc = module.run_retest(from_sha, dry_run=True, limit=None)
        finally:
            os.chdir(prev_cwd)

        if rc != 0:
            failures.append(f"dry-run returned rc={rc}, expected 0")
        if tmp_sentinel.exists():
            failures.append("dry-run wrote sentinel (invariant violated)")
        if tmp_worktree.exists():
            failures.append("dry-run created worktree (invariant violated)")
        report_text = module._RETEST_REPORT.read_text() if module._RETEST_REPORT.exists() else ""
        if "dry-run-skipped" not in report_text:
            failures.append("dry-run report missing 'dry-run-skipped' outcome")
        if sha_a[:7] not in report_text or sha_b[:7] not in report_text or sha_c[:7] not in report_text:
            failures.append("dry-run report missing candidate SHAs")

        # Test sentinel-refusal exits 2.
        tmp_sentinel.parent.mkdir(parents=True, exist_ok=True)
        tmp_sentinel.write_text('{"sha":"deadbeef","started_at":"x"}')
        os.chdir(tmp)
        try:
            try:
                module.run_retest(from_sha, dry_run=True)
                failures.append("sentinel-refusal did not raise SystemExit")
            except SystemExit as e:
                if e.code != 2:
                    failures.append(f"sentinel-refusal exit={e.code}, expected 2")
        finally:
            os.chdir(prev_cwd)
            tmp_sentinel.unlink(missing_ok=True)

    finally:
        # Restore module-level paths before the tmp dir disappears so
        # subsequent imports in this process don't dangle on fixture paths.
        try:
            module = sys.modules[__name__]
            module.PROJECT_DIR = orig_project_dir  # type: ignore[name-defined]
            module._RESULTS_TSV = orig_results_tsv  # type: ignore[name-defined]
            module.BASELINE_PATH = orig_baseline_path  # type: ignore[name-defined]
            module._git_commit_time = orig_commit_time_fn  # type: ignore[name-defined]
            module._RETEST_SENTINEL = PROJECT_DIR / ".omc" / "retest-in-progress"
            module._RETEST_WORKTREE = PROJECT_DIR / ".omc" / "retest-worktree"
            module._RETEST_REPORT = PROJECT_DIR / ".omc" / "retest-report.md"
        except NameError:
            pass
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("--retest-self-test FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("--retest-self-test: PASS (ordering + baseline disk re-read + "
          "check_git_diff_audit WARN + dry-run no-mutation + sentinel-refusal)")
    return 0


def run_verify(agent_name: str, reported_combined: float) -> int:
    """Run the 5-check verification chain. Returns 0 on HIGH/MEDIUM, 1 on LOW."""
    print(f"\nVERIFICATION REPORT for agent [{agent_name}]:")

    # 1. Metric re-run
    metric_status, metric_detail, metric_output = check_metric_rerun(reported_combined)
    print(f"  Metric re-run:    {metric_status} ({metric_detail})")

    # Parse actual combined for anomaly check
    actual_match = re.search(r"actual: ([\d.]+)", metric_detail)
    actual_combined = float(actual_match.group(1)) if actual_match else reported_combined

    # 2. Git diff audit
    diff_status, diff_detail = check_git_diff_audit()
    print(f"  Git diff audit:   {diff_status} (protected files: {diff_detail})")

    # 3. Anomaly detection
    anomaly_status, anomaly_detail = check_anomaly(actual_combined)
    print(f"  Anomaly check:    {anomaly_status} ({anomaly_detail})")

    # 4. Preflight
    preflight_status, preflight_detail = check_preflight()
    print(f"  Preflight:        {preflight_status} ({preflight_detail})")

    # 5. Clean FP bound
    fp_bound_status, fp_bound_detail = check_clean_fp_bound(metric_output)
    print(f"  Clean FP bound:   {fp_bound_status} ({fp_bound_detail})")

    # Determine confidence
    all_statuses = [metric_status, diff_status, anomaly_status, preflight_status, fp_bound_status]
    if any(s == "FAIL" for s in all_statuses):
        confidence = "LOW"
    elif any(s == "WARN" for s in all_statuses):
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    print(f"  CONFIDENCE: {confidence}")

    return 0 if confidence in ("HIGH", "MEDIUM") else 1


# ---------------------------------------------------------------------------
# US-517 Commit 2: --maintain subcommand
# ---------------------------------------------------------------------------
# Triage agent-requested enhancements, classify crashes, draft ralplan specs.
# No LLM invocation; stdlib only.
# ---------------------------------------------------------------------------

import difflib as _difflib  # noqa: E402 (intentional: maintain-only import)
import string as _string    # noqa: E402


_MAINTAIN_DISABLED_SENTINEL = REPO_ROOT / ".omc" / "maintainer-disabled"
_BACKLOG_PATH = REPO_ROOT / ".omc" / "enhancement-backlog.md"
_RESEARCH_NOTES_PATH = REPO_ROOT / ".omc" / "research_notes.md"
_SPECS_DIR = REPO_ROOT / ".omc" / "specs"
_CRASH_COUNTER_PATH = REPO_ROOT / ".omc" / "supervisor-crash-counter.txt"
_MAINTAIN_LOG = get_logger("supervisor.maintain")


# ---------------------------------------------------------------------------
# Backlog parse / serialize
# ---------------------------------------------------------------------------

def _parse_backlog(md_path: Path) -> tuple[str, list[dict]]:
    """Parse enhancement-backlog.md into (header_prose, entries).

    Header prose is everything before the first H2. Each H2 becomes a dict
    with 'id' (the H2 title) plus all '- **key:** value' fields. Unknown
    fields are preserved verbatim in '_extra_lines' list for round-trip fidelity.
    """
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Split into chunks by H2 boundaries.
    # First chunk = header prose; subsequent chunks = one entry each.
    chunks: list[list[str]] = []
    current: list[str] = []
    for ln in lines:
        if ln.startswith("## ") and current:
            chunks.append(current)
            current = [ln]
        else:
            current.append(ln)
    if current:
        chunks.append(current)

    if not chunks:
        return "", []

    # First chunk is header prose (may or may not end with blank line)
    header_prose = "".join(chunks[0])
    entries: list[dict] = []

    _KNOWN_FIELDS = {
        "status", "first_seen", "last_seen", "request_count",
        "category", "risk", "excerpt", "notes",
    }

    for chunk in chunks[1:]:
        entry: dict = {"_extra_lines": []}
        # First line of chunk is the H2 title
        h2_line = chunk[0]
        entry["id"] = h2_line.lstrip("# ").strip()
        entry["_trailing_blank"] = False

        _saw_field = False
        entry["_pre_field_blanks"] = []
        for ln in chunk[1:]:
            # Try to parse '- **key:** value' (format: bold includes the colon)
            m = re.match(r"^- \*\*([^*:]+):\*\*\s*(.*)", ln.rstrip("\n"))
            if m:
                _saw_field = True
                key = m.group(1).strip()
                val = m.group(2).strip()
                if key in _KNOWN_FIELDS:
                    entry[key] = val
                else:
                    entry["_extra_lines"].append(ln)
            elif not _saw_field and ln.strip() == "":
                # Blank lines before any field — preserve before fields
                entry["_pre_field_blanks"].append(ln)
            elif ln.strip() == "---":
                entry["_extra_lines"].append(ln)
            elif ln.strip() == "":
                entry["_extra_lines"].append(ln)
            else:
                entry["_extra_lines"].append(ln)

        entries.append(entry)

    return header_prose, entries


def _serialize_backlog(header_prose: str, entries: list[dict], md_path: Path) -> None:
    """Write header + entries back in H2 format, preserving extra lines."""
    _KNOWN_FIELDS = {
        "status", "first_seen", "last_seen", "request_count",
        "category", "risk", "excerpt", "notes",
    }
    _FIELD_ORDER = [
        "status", "first_seen", "last_seen", "request_count",
        "category", "risk", "excerpt", "notes",
    ]

    out: list[str] = [header_prose]

    for entry in entries:
        entry_id = entry.get("id", "unknown")
        out.append(f"## {entry_id}\n")
        # Pre-field blanks (blank lines between H2 and first field in original)
        for ln in entry.get("_pre_field_blanks", []):
            out.append(ln if ln.endswith("\n") else ln + "\n")
        for field in _FIELD_ORDER:
            if field in entry:
                out.append(f"- **{field}:** {entry[field]}\n")
        for ln in entry.get("_extra_lines", []):
            out.append(ln if ln.endswith("\n") else ln + "\n")

    content = "".join(out)
    md_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Text normalization and fuzzy matching
# ---------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    """Case-fold, collapse whitespace, strip punctuation-only tokens."""
    s = s.lower()
    tokens = s.split()
    punct = set(_string.punctuation)
    tokens = [t for t in tokens if not all(c in punct for c in t)]
    return " ".join(tokens)


def _fuzzy_match(needle: str, haystack_entries: list[dict],
                 threshold: float = 0.5) -> dict | None:
    """Return the best matching entry from haystack_entries or None.

    Matches needle against each entry's 'excerpt' field (case-folded).
    Returns first entry with SequenceMatcher ratio >= threshold, or None.
    """
    needle_norm = _normalize_text(needle)
    best_ratio = 0.0
    best_entry = None
    for entry in haystack_entries:
        excerpt = entry.get("excerpt", "") or entry.get("id", "")
        ratio = _difflib.SequenceMatcher(
            None, needle_norm, _normalize_text(excerpt)
        ).ratio()
        if ratio >= threshold and ratio > best_ratio:
            best_ratio = ratio
            best_entry = entry
    return best_entry


# ---------------------------------------------------------------------------
# Extract enhancement bullets from research_notes.md
# ---------------------------------------------------------------------------

def _sha_commit_time(sha: str) -> int | None:
    """Return commit timestamp for sha, or None if not found."""
    try:
        r = subprocess.run(
            ["git", "show", "-s", "--format=%ct", sha],
            capture_output=True, text=True, timeout=10, cwd=REPO_ROOT,
        )
        if r.returncode != 0:
            return None
        return int(r.stdout.strip())
    except Exception:
        return None


def _extract_enhancement_bullets(notes_path: Path, since_sha: str) -> list[tuple[str, str]]:
    """Scan research_notes.md for '(e) Wrapper enhancements' sections.

    Returns list of (sha, bullet_text) for entries whose SHA is strictly
    newer than since_sha (by commit timestamp). Uses the '## <ts> — <sha>'
    header pattern.

    If since_sha is empty/None, all bullets are returned.
    """
    since_time: int | None = None
    if since_sha:
        since_time = _sha_commit_time(since_sha)

    if not notes_path.exists():
        return []

    text = notes_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    bullets: list[tuple[str, str]] = []
    current_sha: str = ""
    current_sha_time: int | None = None
    in_wrapper_section = False
    wrapper_text_lines: list[str] = []

    def _flush_wrapper(sha: str) -> None:
        if not wrapper_text_lines:
            return
        # Join and split into bullet items by sentence/clause boundaries
        raw = " ".join(wrapper_text_lines)
        # Split on numbered patterns like "(i)", "(ii)", "(1)", "(2)" or just treat as one bullet
        sub_items = re.split(r"\s+(?=\(\w+\)\s)", raw)
        for item in sub_items:
            item = item.strip()
            if item:
                bullets.append((sha, item))
        wrapper_text_lines.clear()

    for ln in lines:
        # Detect iteration header: '## <ts> — <sha> ...'
        m = re.match(r"^## \S+ — ([0-9a-f]{7,40})\b", ln)
        if m:
            # Flush any pending wrapper text from previous entry
            _flush_wrapper(current_sha)
            in_wrapper_section = False
            current_sha = m.group(1)
            current_sha_time = _sha_commit_time(current_sha)
            continue

        # Detect '(e) Wrapper enhancements' section
        if re.match(r"^\(e\)\s+Wrapper enhancements", ln.strip()):
            # Only include if this entry's SHA is newer than since_sha
            if since_time is not None and current_sha_time is not None:
                if current_sha_time <= since_time:
                    in_wrapper_section = False
                    continue
            in_wrapper_section = True
            # Strip the '(e) Wrapper enhancements' prefix, keep the rest
            rest = re.sub(r"^\(e\)\s+Wrapper enhancements[.:]*\s*", "", ln.strip())
            if rest:
                wrapper_text_lines.append(rest)
            continue

        if in_wrapper_section:
            stripped = ln.strip()
            # Next section marker stops collection
            if stripped.startswith("(") and re.match(r"^\([a-z]\)\s", stripped) and not stripped.startswith("(e)"):
                _flush_wrapper(current_sha)
                in_wrapper_section = False
                continue
            # Next H2 header would also stop (handled above)
            if stripped:
                wrapper_text_lines.append(stripped)

    _flush_wrapper(current_sha)
    return bullets


# ---------------------------------------------------------------------------
# Spec drafting
# ---------------------------------------------------------------------------

def _draft_spec(entry: dict, specs_dir: Path) -> Path | None:
    """Write a minimal spec template for entry to specs_dir/deep-interview-<id>.md.

    Returns the path if created, or None if skipped (already exists).
    """
    entry_id = entry.get("id", "unknown")
    out_path = specs_dir / f"deep-interview-{entry_id}.md"
    if out_path.exists():
        return None

    specs_dir.mkdir(parents=True, exist_ok=True)
    excerpt = entry.get("excerpt", "")
    notes = entry.get("notes", "")
    risk = entry.get("risk", "")
    category = entry.get("category", "")
    request_count = entry.get("request_count", "?")

    content = f"""# deep-interview: {entry_id}

_Auto-drafted by supervisor_agent.py --maintain on {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}_
_Source: enhancement-backlog.md entry `{entry_id}` (request_count={request_count}, risk={risk}, category={category})_

---

## Goal

<!-- Paste the excerpt below and refine into a concrete goal statement -->

{excerpt}

## Constraints

<!-- From backlog notes -->

{notes}

## Non-Goals

<!-- Fill in -->

## Acceptance Criteria

<!-- Fill in -->

## Open Questions

<!-- Fill in -->
"""
    out_path.write_text(content, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Crash counter state
# ---------------------------------------------------------------------------

def _load_state() -> int:
    """Load crash counter from .omc/supervisor-crash-counter.txt. Returns 0 on missing/corrupt."""
    try:
        text = _CRASH_COUNTER_PATH.read_text(encoding="utf-8").strip()
        return int(text)
    except Exception:
        return 0


def _save_state(count: int) -> None:
    """Save crash counter to .omc/supervisor-crash-counter.txt."""
    _CRASH_COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CRASH_COUNTER_PATH.write_text(str(count) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Crash classifier
# ---------------------------------------------------------------------------

_RETRAIN_PATTERNS = [
    "splice/classifier/train_classifier.py",
    "sklearn/",
]
_RUNTIME_PATTERNS = [
    "autoresearch/",
    "splice/detector.py",
    "splice/features.py",
    "splice/ml_eval.py",
    "splice/classifier/shap_report.py",
]


def _last_jsonl_event_matches(log_text: str, event_name: str) -> bool:
    """Return True if the last valid JSON line in log_text has event==event_name."""
    last_match = None
    for ln in log_text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
            if isinstance(rec, dict) and "event" in rec:
                last_match = rec
        except json.JSONDecodeError:
            continue
    return last_match is not None and last_match.get("event") == event_name


def _top_frame(tb_text: str) -> str:
    """Return the last 'File ...' frame path from a traceback string."""
    frames = []
    for ln in tb_text.splitlines():
        m = re.search(r'File "([^"]+)", line (\d+)', ln)
        if m:
            frames.append(m.group(1))
    # The last file listed is the innermost (top of call stack in Python terms)
    return frames[-1] if frames else ""


def _classify_crash(log_text: str, eval_exit_code: int) -> str:
    """Classify a crash into one of 4 categories.

    Categories:
      retrain_crash      - retrain infrastructure failure
      pipeline_bug       - runtime module error or parse_fail
      hypothesis_content - catastrophic discard, pipeline ran clean
      unclassifiable     - traceback present but frame matches nothing known
    """
    tb = _extract_last_traceback(log_text)

    if tb is None:
        if eval_exit_code == 0:
            return "hypothesis_content"  # catastrophic discard, no crash
        else:
            return "pipeline_bug"  # parse_fail: exited non-zero, no Python traceback

    # Traceback present — classify by faulting frame
    frame = _top_frame(tb)

    # Check retrain first (classifier/sklearn internals)
    if any(p in frame for p in _RETRAIN_PATTERNS):
        return "retrain_crash"

    # Check last JSONL event for retrain failure
    if _last_jsonl_event_matches(log_text, "classifier.retrain.failed"):
        return "retrain_crash"

    # Check runtime modules
    if any(p in frame for p in _RUNTIME_PATTERNS):
        return "pipeline_bug"

    # Traceback present but top frame matches nothing known
    return "unclassifiable"


# ---------------------------------------------------------------------------
# run_maintain entry point
# ---------------------------------------------------------------------------

def run_maintain(trigger: str) -> int:
    """Triage enhancement requests and classify crashes.

    trigger: 'crash' | 'periodic' | 'manual'
    Exit codes: 0=continue, 1=halt, >=2=unexpected (warn-and-continue in wrapper).
    """
    # --- Common preamble ---
    disabled = _MAINTAIN_DISABLED_SENTINEL
    if disabled.exists():
        print("maintain: disabled (.omc/maintainer-disabled sentinel present)", flush=True)
        _MAINTAIN_LOG.emit("INFO", "maintain.skipped", reason="disabled_sentinel")
        return 0

    if _RETEST_SENTINEL.exists():
        print("maintain: skipped (.omc/retest-in-progress present)", flush=True)
        _MAINTAIN_LOG.emit("INFO", "maintain.skipped", reason="retest_in_progress")
        return 0

    # Load backlog
    if not _BACKLOG_PATH.exists():
        print("maintain: backlog not found, no-op", flush=True)
        return 0

    try:
        header_prose, entries = _parse_backlog(_BACKLOG_PATH)
    except Exception as e:
        print(f"maintain: backlog parse error: {e}", file=sys.stderr)
        _MAINTAIN_LOG.emit("ERROR", "maintain.backlog.parse_error", error=str(e))
        return 2

    crash_rc = 0

    # --- Crash path ---
    if trigger in ("crash", "manual"):
        log_path = REPO_ROOT / ".omc" / "last_eval.log"
        if log_path.exists():
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                log_text = ""
                _MAINTAIN_LOG.emit("WARN", "maintain.crash.log_read_error", error=str(e))

            # Determine eval exit code from log: check for non-zero exit signals
            # Since we don't have direct access to the shell exit code here,
            # we infer from traceback presence + the log content.
            # The wrapper sets trigger=crash only after a failing eval/retrain;
            # treat as exit_code=1 if traceback found, else exit_code=0 for
            # catastrophic-discard (diagnose.catastrophic path).
            tb = _extract_last_traceback(log_text)
            # Infer eval exit code: if RESULTS_TSV line is present and combined
            # is <=0.05, pipeline ran clean (catastrophic discard = exit 0).
            eval_exit_code = 1 if tb is not None else 0
            # Check for parse_fail pattern (exit 1, no traceback)
            # If log exists but is tiny/empty and no traceback, treat as exit 1.
            if tb is None and not log_text.strip():
                eval_exit_code = 1

            category = _classify_crash(log_text, eval_exit_code)
            _MAINTAIN_LOG.emit("INFO", "maintain.crash.classified", category=category,
                               trigger=trigger)
            print(f"maintain: crash classified as {category}", flush=True)

            if category in ("retrain_crash", "pipeline_bug"):
                counter = _load_state() + 1
                _save_state(counter)
                _MAINTAIN_LOG.emit("INFO", "maintain.crash.counter",
                                   category=category, counter=counter, threshold=3)
                print(f"maintain: crash counter={counter}", flush=True)
                if counter >= 3:
                    _MAINTAIN_LOG.emit("WARN", "maintain.halt",
                                       reason="consecutive_crash_threshold",
                                       counter=counter, category=category)
                    print(f"maintain: HALT — {counter} consecutive crashes (category={category})",
                          flush=True)
                    crash_rc = 1
            elif category == "hypothesis_content":
                _save_state(0)
            elif category == "unclassifiable":
                _MAINTAIN_LOG.emit("WARN", "maintain.crash.unclassifiable",
                                   trigger=trigger)
                print("maintain: WARN — unclassifiable crash, counter not incremented", flush=True)
        else:
            if trigger == "crash":
                print("maintain: last_eval.log not found, no crash to classify", flush=True)

    # If crash path says halt, return immediately (skip periodic triage)
    if crash_rc == 1:
        return 1

    # --- Periodic/manual path: triage enhancement bullets ---
    if trigger in ("periodic", "manual"):
        _triage_enhancements(header_prose, entries)

    return 0


def _triage_enhancements(header_prose: str, entries: list[dict]) -> None:
    """Scan research_notes for new enhancement bullets; update backlog; draft specs."""
    # Determine since_sha from last_seen of the most recently updated entry
    # Use the most recent last_seen SHA across all entries as the scan baseline.
    last_seens = [e.get("last_seen", "") for e in entries if e.get("last_seen")]
    # Pick the one with the most recent commit time
    since_sha = ""
    if last_seens:
        best_time = -1
        for sha in last_seens:
            t = _sha_commit_time(sha)
            if t is not None and t > best_time:
                best_time = t
                since_sha = sha

    try:
        bullets = _extract_enhancement_bullets(_RESEARCH_NOTES_PATH, since_sha)
    except Exception as e:
        _MAINTAIN_LOG.emit("WARN", "maintain.bullets.extract_error", error=str(e))
        bullets = []

    changed = False

    for sha, bullet in bullets:
        match = _fuzzy_match(bullet, entries)
        if match is not None:
            # Bump request_count and last_seen
            try:
                old_count_str = match.get("request_count", "0")
                # request_count may have trailing text like "2+" or "3+ (some note)"
                old_count = int(re.match(r"\d+", old_count_str).group()) if re.match(r"\d+", old_count_str) else 0
            except Exception:
                old_count = 0
            match["request_count"] = str(old_count + 1)
            match["last_seen"] = sha
            _MAINTAIN_LOG.emit("INFO", "maintain.backlog.bump",
                               entry_id=match.get("id"), sha=sha,
                               new_count=old_count + 1)
            print(f"maintain: bumped request_count for '{match.get('id')}' "
                  f"(now {old_count + 1})", flush=True)
            changed = True
        else:
            # Append new entry
            new_id = _bullet_to_id(bullet)
            new_entry: dict = {
                "id": new_id,
                "status": "pending",
                "first_seen": sha,
                "last_seen": sha,
                "request_count": "1",
                "category": "observability",
                "risk": "medium",
                "excerpt": bullet[:300],
                "notes": "Auto-triaged by supervisor_agent.py --maintain. Needs human review.",
                "_extra_lines": [],
            }
            entries.append(new_entry)
            _MAINTAIN_LOG.emit("INFO", "maintain.backlog.new_entry",
                               entry_id=new_id, sha=sha)
            print(f"maintain: new backlog entry '{new_id}' from sha={sha}", flush=True)
            changed = True

    # Spec drafting: status=pending, risk=low, request_count >= 3
    for entry in entries:
        if entry.get("status") != "pending":
            continue
        risk = entry.get("risk", "")
        if not risk.startswith("low"):
            continue
        try:
            count_str = entry.get("request_count", "0")
            count = int(re.match(r"\d+", count_str).group()) if re.match(r"\d+", count_str) else 0
        except Exception:
            count = 0
        if count < 3:
            continue
        spec_path = _draft_spec(entry, _SPECS_DIR)
        if spec_path is not None:
            entry["status"] = "spec_drafted"
            _MAINTAIN_LOG.emit("INFO", "maintain.spec.drafted",
                               entry_id=entry.get("id"), path=str(spec_path))
            print(f"maintain: spec drafted for '{entry.get('id')}' -> {spec_path.name}",
                  flush=True)
            changed = True

    # Auto-defer: status=spec_drafted, spec_drafted_fires >= 3 (tracked in extra field)
    for entry in entries:
        if entry.get("status") != "spec_drafted":
            continue
        try:
            fires = int(entry.get("spec_drafted_fires", "0") or "0")
        except Exception:
            fires = 0
        fires += 1
        entry["spec_drafted_fires"] = str(fires)
        if fires >= 3:
            entry["status"] = "deferred"
            _MAINTAIN_LOG.emit("INFO", "maintain.backlog.deferred",
                               entry_id=entry.get("id"), fires=fires)
            print(f"maintain: deferred '{entry.get('id')}' after {fires} fires", flush=True)
        changed = True  # always update fires counter if spec_drafted

    if changed:
        try:
            _serialize_backlog(header_prose, entries, _BACKLOG_PATH)
            _MAINTAIN_LOG.emit("INFO", "maintain.backlog.written",
                               path=str(_BACKLOG_PATH))
            print("maintain: backlog updated", flush=True)
        except Exception as e:
            _MAINTAIN_LOG.emit("ERROR", "maintain.backlog.write_error", error=str(e))
            print(f"maintain: ERROR writing backlog: {e}", file=sys.stderr)
    else:
        print("maintain: no changes (backlog up-to-date)", flush=True)


def _bullet_to_id(bullet: str) -> str:
    """Convert a bullet string to a kebab-case id.

    Strips leading list markers ``(1)``, ``1.``, ``1)``, ``[1]``; drops
    common English articles; lowercases + kebab-joins the first 6 words;
    trims to <=40 chars at a word boundary (never mid-word).
    """
    stripped = re.sub(r"^\s*[\(\[]?\d+[\)\].]?\s+", "", bullet)
    words = re.sub(r"[^\w\s-]", "", stripped.lower()).split()
    words = [w for w in words if w not in {"a", "an", "the"}]
    slug = "-".join(words[:6])
    if len(slug) > 40:
        cut = slug[:40].rsplit("-", 1)[0]
        slug = cut if cut else slug[:40]
    return slug or "enhancement"




def main():
    parser = argparse.ArgumentParser(
        prog="supervisor_agent",
        description=(
            "Supervisor agent for the autoresearch loop: "
            "verify hypotheses, diagnose crashes, maintain the pipeline."
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Re-run the metric + diff audit + anomaly + preflight check chain. "
            "Requires --agent-name and --reported-combined."
        ),
    )
    mode.add_argument(
        "--diagnose",
        action="store_true",
        help="Diagnose last eval/retrain crash.",
    )
    mode.add_argument(
        "--maintain",
        action="store_true",
        help=(
            "Triage agent-requested enhancements and classify crashes. "
            "Requires --trigger."
        ),
    )
    mode.add_argument(
        "--retest",
        metavar="FROM_SHA",
        help="Replay discarded hypotheses from FROM_SHA.",
    )

    # --verify subflags
    parser.add_argument("--agent-name", help="Name of the agent being verified (--verify).")
    parser.add_argument(
        "--reported-combined",
        type=float,
        help="Combined score reported by agent (--verify).",
    )

    # --maintain subflags
    parser.add_argument(
        "--trigger",
        choices=["crash", "periodic", "manual"],
        default="manual",
        help="Trigger source for --maintain (ignored otherwise).",
    )

    # --retest subflags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview retest candidates without running (--retest).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max candidates to process (--retest).",
    )
    parser.add_argument(
        "--retest-re-eval",
        action="store_true",
        help="Force re-evaluation (--retest).",
    )
    parser.add_argument("--retest-origin", help=argparse.SUPPRESS)

    # Debug flags (not in the mutually exclusive group — dev-only, not called by wrapper)
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--retest-self-test", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Debug flags short-circuit before mode dispatch
    if args.self_test:
        sys.exit(_diagnose_self_test())
    if args.retest_self_test:
        sys.exit(_retest_self_test())

    if args.verify:
        if not args.agent_name or args.reported_combined is None:
            parser.error("--verify requires --agent-name and --reported-combined")
        sys.exit(run_verify(args.agent_name, args.reported_combined))
    elif args.diagnose:
        sys.exit(run_diagnose())
    elif args.maintain:
        sys.exit(run_maintain(trigger=args.trigger))
    elif args.retest:
        sys.exit(run_retest(
            args.retest,
            dry_run=args.dry_run,
            limit=args.limit,
            re_eval=args.retest_re_eval,
        ))


if __name__ == "__main__":
    main()
