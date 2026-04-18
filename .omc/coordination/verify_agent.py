#!/usr/bin/env python3
"""Mandatory Agent Verification System for autoresearch-splice.

Standalone CLI that verifies agent work via 4 independent checks.
Usage: uv run python .omc/coordination/verify_agent.py --agent-name <name> --reported-combined <float>
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASELINE_PATH = SCRIPT_DIR / "baseline_metrics.json"
PROTECTED_FILES = [
    "evaluate.py",
    "program.md",
    "data/eval/*",
    "data/test/*",
    ".omc/coordination/manifest.json",
    ".omc/coordination/preflight.py",
]


def check_metric_rerun(reported: float) -> tuple[str, str, str]:
    """Re-run evaluate.py and compare combined score to reported value."""
    try:
        result = subprocess.run(
            ["uv", "run", "python", "evaluate.py", "--shap"],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return "FAIL", "evaluate.py timed out after 300s", ""
    except Exception as e:
        return "FAIL", f"subprocess error: {e}", ""

    if result.returncode != 0:
        return "FAIL", f"evaluate.py exited {result.returncode}", ""

    output = result.stdout + result.stderr
    matches = re.findall(r"^combined:\s*([\d.]+)", output, re.MULTILINE)
    if not matches:
        return "FAIL", "could not parse combined score from output", output

    actual = float(matches[-1])
    delta = abs(actual - reported)
    status = "PASS" if delta < 0.005 else "FAIL"
    return status, f"reported: {reported:.3f}, actual: {actual:.3f}, delta: {delta:.4f}", output


def check_git_diff_audit() -> tuple[str, str]:
    """Check that no protected files are modified."""
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

    changed = set(unstaged + staged)
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
# On eval/retrain crash the wrapper invokes `verify_agent.py --diagnose`.
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
        with open(out_path, "a") as f:
            f.write(_DIAGNOSE_SEPARATOR + f"diagnose: no-traceback\nnote: eval log contains no Python traceback; crash cause unknown from log alone.\n")
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


def main():
    # --diagnose / --self-test short-circuit the original argparse so the
    # wrapper's existing `--agent-name X --reported-combined Y` call shape
    # is unaffected.
    if "--diagnose" in sys.argv:
        sys.exit(run_diagnose())
    if "--self-test" in sys.argv:
        sys.exit(_diagnose_self_test())

    parser = argparse.ArgumentParser(description="Verify agent work")
    parser.add_argument("--agent-name", required=True, help="Name of the agent being verified")
    parser.add_argument("--reported-combined", required=True, type=float, help="Combined score reported by agent")
    args = parser.parse_args()

    print(f"\nVERIFICATION REPORT for agent [{args.agent_name}]:")

    # 1. Metric re-run
    metric_status, metric_detail, metric_output = check_metric_rerun(args.reported_combined)
    print(f"  Metric re-run:    {metric_status} ({metric_detail})")

    # Parse actual combined for anomaly check
    actual_match = re.search(r"actual: ([\d.]+)", metric_detail)
    actual_combined = float(actual_match.group(1)) if actual_match else args.reported_combined

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

    sys.exit(0 if confidence in ("HIGH", "MEDIUM") else 1)


if __name__ == "__main__":
    main()
