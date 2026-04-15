"""
Regression test for the audio splice detector.
Runs evaluate() twice for determinism, then checks metrics against baselines.

Usage:
    uv run python test_regression.py

Exit code 0 on pass, 1 on failure.
"""

import io
import contextlib
import os
import re
import shutil
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "spliced")

EXPECTED = {
    "combined_min": 0.45,
    "precision_min": 0.55,
    "recall_min": 0.40,
    "clean_fp_max": 5,
    "t1_recall_min": 0.50,
    "t2_recall_min": 0.25,
    "opus32k_combined_min": 0.35,
}

T1_COUNT = 20
T2_COUNT = 20

_TP_RE = re.compile(r"TP=(\d+)")


def run_evaluate():
    """Run prepare.evaluate(), capture output, return (result_dict, t1_tp, t2_tp)."""
    from prepare import evaluate

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = evaluate(DATA_DIR)
    lines = buf.getvalue().splitlines()

    t1_tp = t2_tp = 0
    for line in lines:
        stripped = line.strip()
        m = _TP_RE.search(stripped)
        if not m:
            continue
        tp = int(m.group(1))
        if stripped.startswith("T1 "):
            t1_tp += tp
        elif stripped.startswith("T2 "):
            t2_tp += tp

    result["t1_tp"] = t1_tp
    result["t2_tp"] = t2_tp
    return result


def run_evaluate_codec():
    """Run prepare.evaluate_codec(), capture output, return result dict."""
    from prepare import evaluate_codec

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = evaluate_codec(DATA_DIR)
    return result


def check(label, actual, threshold, op=">="):
    """Print one test line and return True if passed."""
    if op == ">=":
        passed = actual >= threshold
        print(f"  {label}: {actual:.3g} >= {threshold} {'PASS' if passed else 'FAIL (regression!)'}")
    else:  # "<="
        passed = actual <= threshold
        print(f"  {label}: {actual} <= {threshold} {'PASS' if passed else 'FAIL (regression!)'}")
    return passed


def main():
    print("Running regression tests...")

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: test dataset not found at {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    # --- Two runs for determinism ---
    r1 = run_evaluate()
    r2 = run_evaluate()

    det_pass = (
        r1["combined"] == r2["combined"]
        and r1["precision"] == r2["precision"]
        and r1["recall"] == r2["recall"]
        and r1["clean_fp"] == r2["clean_fp"]
        and r1["t1_tp"] == r2["t1_tp"]
        and r1["t2_tp"] == r2["t2_tp"]
    )
    print(f"  Determinism: {'PASS (2 runs identical)' if det_pass else 'FAIL (results differ between runs!)'}")
    failures = 0 if det_pass else 1

    # --- Metric checks (use first run) ---
    r = r1
    failures += 0 if check("Combined", r["combined"], EXPECTED["combined_min"]) else 1
    failures += 0 if check("Precision", r["precision"], EXPECTED["precision_min"]) else 1
    failures += 0 if check("Recall", r["recall"], EXPECTED["recall_min"]) else 1
    failures += 0 if check("Clean FP", r["clean_fp"], EXPECTED["clean_fp_max"], op="<=") else 1
    failures += 0 if check("T1 recall", r["t1_tp"] / T1_COUNT, EXPECTED["t1_recall_min"]) else 1
    failures += 0 if check("T2 recall", r["t2_tp"] / T2_COUNT, EXPECTED["t2_recall_min"]) else 1

    # --- Opus 32k codec robustness ---
    if shutil.which("ffmpeg") or os.path.exists("/opt/homebrew/bin/ffmpeg"):
        print("\nRunning Opus 32k codec check...")
        codec_result = run_evaluate_codec()
        failures += 0 if check("Opus32k combined", codec_result["opus32k_combined"],
                               EXPECTED["opus32k_combined_min"]) else 1
    else:
        print("\n  Opus32k: SKIP (ffmpeg not found)")

    print()
    if failures == 0:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(f"REGRESSION DETECTED - {failures} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
