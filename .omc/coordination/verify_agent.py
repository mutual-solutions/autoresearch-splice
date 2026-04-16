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
    "prepare.py",
    "data/spliced/*",
    ".omc/coordination/manifest.json",
    ".omc/coordination/preflight.py",
]


def check_metric_rerun(reported: float) -> tuple[str, str, str]:
    """Re-run prepare.py and compare combined score to reported value."""
    try:
        result = subprocess.run(
            ["uv", "run", "python", "prepare.py", "--with-classifier"],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "FAIL", "prepare.py timed out after 120s", ""
    except Exception as e:
        return "FAIL", f"subprocess error: {e}", ""

    if result.returncode != 0:
        return "FAIL", f"prepare.py exited {result.returncode}", ""

    output = result.stdout + result.stderr
    matches = re.findall(r"^combined:\s*([\d.]+)", output, re.MULTILINE)
    if not matches:
        return "FAIL", "could not parse combined score from output", output

    actual = float(matches[-1])
    delta = abs(actual - reported)
    status = "PASS" if delta < 0.001 else "FAIL"
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


def check_dsp_fp_bound(output: str) -> tuple[str, str]:
    """Check that DSP clean_fp is within bound."""
    match = re.search(r"dsp_clean_fp:\s*(\d+)", output)
    if not match:
        return "WARN", "dsp_clean_fp not found in output (may be running without --with-classifier)"
    dsp_fp = int(match.group(1))
    if dsp_fp > 15:
        return "FAIL", f"dsp_clean_fp={dsp_fp} exceeds bound of 15"
    return "PASS", f"dsp_clean_fp={dsp_fp} (bound: 15)"


def main():
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

    # 5. DSP FP bound
    fp_bound_status, fp_bound_detail = check_dsp_fp_bound(metric_output)
    print(f"  DSP FP bound:     {fp_bound_status} ({fp_bound_detail})")

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
