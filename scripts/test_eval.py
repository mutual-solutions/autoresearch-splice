#!/usr/bin/env python3
"""Held-out test-set eval (overfitting detector for the autoresearch loop).

Runs `splice.evaluate.evaluate()` against `data/eval/korean_iter1/test/`
(4278 conversations, voice-pair holdout = {DaeBuHo, Kanna}, never touched
by the optimization loop) using a different RANDOM_SEED than the eval
sample. Compares the resulting `combined` to the current eval-set
baseline; surfaces a WARN event if the test_combined regresses more
than 20% relative to eval_combined (= overfitting signature).

This script does NOT mutate the agent-facing prompt; the agent stays
blind to test results to prevent leakage. Operator-side instrumentation
only.

Run manually:
    PYTHONPATH=$PWD uv run python scripts/test_eval.py
    PYTHONPATH=$PWD uv run python scripts/test_eval.py --seed 2 --data-dir <path>

Auto-fired by `run_autoresearch.sh` every 10 keeps via the
`test_eval` subcommand.

Artifacts:
    autoresearch/test_baseline.json    Latest test-eval snapshot (atomic write)
    autoresearch/test_history.jsonl    Append-only history of all fires
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# splice/detector.py contains a lazy `from features import ...` that only
# resolves when `splice/` is sys.path[0] (true when the wrapper runs
# `python splice/evaluate.py` as __main__, but NOT when this script
# imports `splice.evaluate` from scripts/). Prepend splice/ to sys.path
# BEFORE the late import so the lazy import resolves. Mirrors the same
# workaround in scripts/web_ui.py.
sys.path.insert(0, str(REPO_ROOT / "splice"))
DEFAULT_TEST_DIR = REPO_ROOT / "data" / "eval" / "korean_iter1" / "test"
BASELINE_PATH = REPO_ROOT / "autoresearch" / "baseline_metrics.json"
TEST_BASELINE_PATH = REPO_ROOT / "autoresearch" / "test_baseline.json"
TEST_HISTORY_PATH = REPO_ROOT / "autoresearch" / "test_history.jsonl"

GAP_WARN_THRESHOLD = -0.20  # WARN when (test - eval) / eval < this


def _git_sha_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True, timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Atomic JSON write via tempfile + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=str(path.parent), delete=False, suffix=".tmp",
    ) as tf:
        json.dump(data, tf, indent=2)
        tf.write("\n")
        tmppath = tf.name
    os.replace(tmppath, str(path))


def _append_history(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(data) + "\n")


def run_test_eval(test_dir: Path, seed: int) -> dict:
    """Invoke splice.evaluate.evaluate() with overridden seed against test_dir.

    Returns the metrics dict from evaluate().
    """
    # Late imports so the script can `--help` without loading splice/.
    import splice.evaluate as ev
    from autoresearch.logger import get_logger
    log = get_logger("test_eval")

    # Override the module-level seed for this invocation.
    original_seed = ev.RANDOM_SEED
    ev.RANDOM_SEED = seed
    try:
        log.emit("INFO", "wrapper.test_eval.start",
                 data_dir=str(test_dir), seed=seed,
                 original_eval_seed=original_seed)
        result = ev.evaluate(str(test_dir))
        return result
    finally:
        ev.RANDOM_SEED = original_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Held-out test-set eval (overfitting detector).",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_TEST_DIR),
                        help=f"Test eval dir. Default: {DEFAULT_TEST_DIR}")
    parser.add_argument("--seed", type=int, default=1,
                        help="RANDOM_SEED override (different from eval's "
                             "seed=0 to avoid correlated easy-fits). Default: 1")
    parser.add_argument("--gap-warn", type=float, default=GAP_WARN_THRESHOLD,
                        help=f"WARN threshold. Default: {GAP_WARN_THRESHOLD} "
                             "(WARN when (test-eval)/eval < this)")
    args = parser.parse_args(argv)

    test_dir = Path(args.data_dir)
    if not test_dir.is_dir():
        print(f"ERROR: test data dir not found: {test_dir}", file=sys.stderr)
        return 1

    # Load current eval baseline (for comparison).
    try:
        eval_baseline = json.loads(BASELINE_PATH.read_text())
        eval_combined = float(eval_baseline.get("combined", 0.0))
    except Exception as exc:
        print(f"WARN: could not read eval baseline: {exc}", file=sys.stderr)
        eval_combined = 0.0

    # Run the eval.
    result = run_test_eval(test_dir, args.seed)
    test_combined = float(result.get("combined", 0.0))

    gap_pct = ((test_combined - eval_combined) / eval_combined
               if eval_combined > 0 else 0.0)

    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
                     .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha_short(),
        "test_combined": test_combined,
        "test_f0_5": float(result.get("f0_5", 0.0)),
        "test_f1": float(result.get("f1", 0.0)),
        "test_precision": float(result.get("precision", 0.0)),
        "test_recall": float(result.get("recall", 0.0)),
        "test_clean_fp_per_min": float(result.get("clean_fp_per_min", 0.0)),
        "test_clean_fp_penalty": float(result.get("clean_fp_penalty", 1.0)),
        "test_n_files": int(result.get("n_files", 0)),
        "test_cross_voice_f1": float(result.get("cross_voice_f1", 0.0)),
        "test_same_voice_edit_f1": float(result.get("same_voice_edit_f1", 0.0)),
        "test_unknown_label_count": int(result.get("unknown_label_count", 0)),
        "eval_combined_at_time": eval_combined,
        "gap_pct": gap_pct,
        "test_random_seed": args.seed,
        "test_data_dir": str(test_dir),
    }

    _atomic_write_json(TEST_BASELINE_PATH, record)
    _append_history(TEST_HISTORY_PATH, record)

    # Structured emit.
    from autoresearch.logger import get_logger
    log = get_logger("test_eval")
    log.emit("INFO", "wrapper.test_eval.fired",
             test_combined=test_combined,
             eval_combined=eval_combined,
             gap_pct=gap_pct,
             git_sha=record["git_sha"],
             n_files=record["test_n_files"])

    if gap_pct < args.gap_warn:
        log.emit("WARN", "wrapper.test_eval.gap_warn",
                 test_combined=test_combined,
                 eval_combined=eval_combined,
                 gap_pct=gap_pct,
                 threshold=args.gap_warn,
                 msg=("test_combined regressed more than "
                      f"{abs(args.gap_warn)*100:.0f}% below eval — "
                      "possible overfitting signature"))
        print(f"GAP_WARN: test={test_combined:.6f} eval={eval_combined:.6f} "
              f"gap={gap_pct*100:+.1f}% (threshold {args.gap_warn*100:+.0f}%)")

    print(f"test_combined: {test_combined:.6f}")
    print(f"eval_combined: {eval_combined:.6f}")
    print(f"gap_pct:       {gap_pct*100:+.2f}%")
    print(f"baseline: {TEST_BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
