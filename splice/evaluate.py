"""
Audio splice detection evaluation oracle (korean-iter1 pivot).

Protected from autoresearch agent modification — only the human edits this.
This file was migrated by maintainer commit (MIGRATE-PROTECTED) for the
korean-iter1 boundary-F1 contract per
`.omc/plans/ralplan-korean-iter1-reset.md` § Step 9.

New contract (korean-iter1)
---------------------------
- Single dataset root: `data/eval/korean_iter1/eval/` containing
  `<conv_id>.opus` audio files and `ground_truth.json` with schema:
      { "<conv_id>": [{"time_s": float, "label": str}, ...], ... }
  Labels in {"cross_voice", "same_voice_edit"}.
- Detector contract: `detect_splices(audio, sr) -> list[tuple[float, str]]`
  where the second element is in {"cross_voice", "same_voice_edit",
  "unknown"}. Malformed/missing labels are coerced to "unknown" at the
  eval boundary (logged via `diag.eval.unknown_label_count`).
- Metric: greedy 1-to-1 nearest-within-collar boundary F1 with
  `COLLAR_S = 0.250 s`. Aggregate `combined = boundary_f1` is
  label-blind (counts unknown predictions normally). Per-class F1
  (`cross_voice_f1`, `same_voice_edit_f1`) FILTERS predictions to only
  those whose label exactly equals the class — `unknown` predictions
  contribute zero to either per-class F1. This preserves the
  voice-overfit detection signal.
- Sample-size cap: deterministic random.sample(60) per invocation
  (RANDOM_SEED=0) to fit the 240 s budget on the ~1100-file eval pool.

Wrapper grep contract (run_autoresearch.sh)
-------------------------------------------
- LAST line of stdout is `combined: <float>`.
- A `RESULTS_TSV: combined=<f> precision=<f> recall=<f> n_files=<i>
  cross_voice_f1=<f> same_voice_edit_f1=<f> unknown_label_count=<i>`
  line exists. The wrapper's `_tsv_field` helper extracts these via
  `<key>=<value>` regex (NOT positional columns), so the format is
  whitespace-separated `key=value` tokens.

Usage:
    PYTHONPATH=$PWD uv run python splice/evaluate.py
    PYTHONPATH=$PWD uv run python splice/evaluate.py --data-dir <path>
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import soundfile as sf

from splice.detector import detect_splices

# US-515 unified structured logger
from autoresearch.logger import get_logger
log = get_logger("eval")


# ---------------------------------------------------------------------------
# Korean-iter1 contract constants
# ---------------------------------------------------------------------------

COLLAR_S = 0.250
RANDOM_SEED = 0
N_EVAL_FILES = 60
VALID_LABELS = {"cross_voice", "same_voice_edit", "unknown"}

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eval", "korean_iter1", "eval",
)


# ---------------------------------------------------------------------------
# Pred-label normalization
# ---------------------------------------------------------------------------

def normalize_pred_tuple(p) -> tuple[float, str]:
    """Coerce a detector output element to (time_s: float, label: str).

    Defaults label to 'unknown' if missing, None, empty, or not in
    `VALID_LABELS`. Bare floats/ints are accepted for backward compat
    and labeled 'unknown'.

    Raises ValueError on shapes we cannot parse.
    """
    if isinstance(p, tuple) and len(p) == 2:
        time_s, label = p
        if not isinstance(label, str) or not label or label not in VALID_LABELS:
            label = "unknown"
        return (float(time_s), label)
    if isinstance(p, list) and len(p) == 2:
        # Some callers may emit lists rather than tuples.
        time_s, label = p
        if not isinstance(label, str) or not label or label not in VALID_LABELS:
            label = "unknown"
        return (float(time_s), label)
    if isinstance(p, (int, float)):
        return (float(p), "unknown")
    raise ValueError(f"Invalid prediction format: {p!r}")


def normalize_predictions(raw) -> tuple[list[tuple[float, str]], int]:
    """Apply normalize_pred_tuple to every element. Returns
    (normalized_list, n_coerced) where n_coerced counts entries whose
    label became 'unknown' due to coercion (NOT counting bare-float
    entries — those weren't carrying a label to begin with, but we still
    count them for visibility into detector contract drift)."""
    out: list[tuple[float, str]] = []
    n_coerced = 0
    for p in raw:
        normalized = normalize_pred_tuple(p)
        # Detect coercion: original was a 2-tuple/list with a non-unknown
        # label that got rewritten to 'unknown'. Bare floats are also
        # 'unknown' but did not carry a label originally — count them too
        # so the diagnostic surface contract drift either way.
        if normalized[1] == "unknown":
            if isinstance(p, (tuple, list)) and len(p) == 2:
                _, orig_label = p
                if orig_label != "unknown":
                    n_coerced += 1
            else:
                # bare float: contract drift from the new tuple shape
                n_coerced += 1
        out.append(normalized)
    return out, n_coerced


# ---------------------------------------------------------------------------
# Boundary F1 (greedy 1-to-1 within collar)
# ---------------------------------------------------------------------------

def boundary_f1(
    predictions: list[tuple[float, str]],
    ground_truth: list[tuple[float, str]],
    collar_s: float = COLLAR_S,
) -> tuple[float, float, float, int, int, int]:
    """Greedy 1-to-1 nearest matching within ±collar_s.

    Label-blind aggregate: predictions match GT regardless of label.
    Predictions are processed in time-sorted order; for each prediction we
    pick the nearest unmatched GT within collar.

    Returns (precision, recall, f1, tp, fp, fn).
    """
    preds_sorted = sorted(predictions, key=lambda p: p[0])
    matched_gt_idx: set[int] = set()
    tp = 0
    for pred_time, _label in preds_sorted:
        candidates = [
            (i, gt_time)
            for i, (gt_time, _) in enumerate(ground_truth)
            if i not in matched_gt_idx and abs(gt_time - pred_time) <= collar_s
        ]
        if not candidates:
            continue
        nearest_i, _ = min(candidates, key=lambda c: abs(c[1] - pred_time))
        matched_gt_idx.add(nearest_i)
        tp += 1
    fp = len(predictions) - tp
    fn = len(ground_truth) - len(matched_gt_idx)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def compute_class_f1(
    predictions: list[tuple[float, str]],
    ground_truth_class_only: list[tuple[float, str]],
    class_label: str,
    collar_s: float = COLLAR_S,
) -> float:
    """Per-class F1: filter predictions to ONLY label==class_label, then
    run greedy boundary matching against the GT subset for this class.

    Critical Item 3 semantics: 'unknown' predictions contribute ZERO to
    cross_voice_f1 and ZERO to same_voice_edit_f1. The aggregate
    boundary_f1 still counts them (label-blind).
    """
    filtered_preds = [p for p in predictions if p[1] == class_label]
    _p, _r, f1, _tp, _fp, _fn = boundary_f1(filtered_preds, ground_truth_class_only, collar_s)
    return f1


# ---------------------------------------------------------------------------
# Ground-truth loading (korean-iter1 schema)
# ---------------------------------------------------------------------------

def load_ground_truth(data_dir: str) -> dict[str, list[tuple[float, str]]]:
    """Load `data_dir/ground_truth.json` with schema:
        {"<conv_id>": [{"time_s": float, "label": str}, ...], ...}
    Returns {conv_id: [(time_s, label), ...]} with labels coerced into
    VALID_LABELS (unknown labels in GT logged as a warning — GT should
    only carry cross_voice/same_voice_edit but we don't crash).
    """
    gt_path = os.path.join(data_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        log.emit("ERROR", "eval.input.error",
                 kind="ground_truth_missing", path=gt_path)
        print(f"ERROR: ground_truth.json not found at {gt_path}", file=sys.stderr)
        sys.exit(1)
    with open(gt_path, "r") as f:
        raw = json.load(f)

    out: dict[str, list[tuple[float, str]]] = {}
    for conv_id, entries in raw.items():
        rows: list[tuple[float, str]] = []
        for e in entries:
            t = float(e["time_s"])
            lbl = e.get("label", "unknown")
            if lbl not in VALID_LABELS:
                lbl = "unknown"
            rows.append((t, lbl))
        rows.sort(key=lambda x: x[0])
        out[conv_id] = rows
    return out


# ---------------------------------------------------------------------------
# Eval loop (single dataset: data/eval/korean_iter1/eval/)
# ---------------------------------------------------------------------------

def evaluate(data_dir: str) -> dict:
    """Run boundary-F1 eval over a deterministic random.sample(60) of the
    korean_iter1 eval split. Returns metrics dict; prints the wrapper
    grep contract lines (RESULTS_TSV + final `combined:`)."""
    t_start = time.time()

    ground_truth = load_ground_truth(data_dir)
    if not ground_truth:
        log.emit("ERROR", "eval.input.error",
                 kind="empty_ground_truth", data_dir=str(data_dir))
        print("ERROR: ground_truth.json is empty", file=sys.stderr)
        sys.exit(1)

    all_conv_ids = sorted(ground_truth.keys())
    rng = random.Random(RANDOM_SEED)
    n_to_sample = min(N_EVAL_FILES, len(all_conv_ids))
    selected = rng.sample(all_conv_ids, n_to_sample)
    selected.sort()  # iterate deterministically (sample order is also seeded)

    log.emit("INFO", "eval.start",
             dataset="korean_iter1",
             data_dir=str(data_dir),
             n_files=len(selected),
             n_total_in_pool=len(all_conv_ids),
             collar_ms=int(COLLAR_S * 1000),
             random_seed=RANDOM_SEED)

    all_preds: list[tuple[float, str]] = []
    all_gt: list[tuple[float, str]] = []
    unknown_label_count = 0  # # of pred entries coerced to 'unknown'
    n_processed = 0
    n_skipped = 0
    n_errors = 0

    for idx, conv_id in enumerate(selected):
        audio_path = os.path.join(data_dir, f"{conv_id}.opus")
        if not os.path.exists(audio_path):
            # Try .wav as a fallback (some regen variants ship wav)
            alt = os.path.join(data_dir, f"{conv_id}.wav")
            if os.path.exists(alt):
                audio_path = alt
            else:
                log.emit("WARN", "eval.file.missing",
                         conv_id=conv_id, path=audio_path)
                n_skipped += 1
                continue

        try:
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        except Exception as exc:
            log.emit("ERROR", "eval.file.read_error",
                     conv_id=conv_id, path=audio_path,
                     exc_type=type(exc).__name__, message=str(exc))
            n_errors += 1
            continue

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        try:
            raw_preds = detect_splices(audio, sr)
        except Exception as exc:
            log.emit("ERROR", "eval.detector.error",
                     conv_id=conv_id,
                     exc_type=type(exc).__name__, message=str(exc))
            n_errors += 1
            # Treat as zero predictions; GT for this file still counts
            # as FN in the aggregate.
            raw_preds = []

        normalized, n_coerced = normalize_predictions(raw_preds)
        unknown_label_count += sum(1 for _, lbl in normalized if lbl == "unknown")

        # GT for this conv
        gt_for_conv = ground_truth[conv_id]

        all_preds.extend(normalized)
        all_gt.extend(gt_for_conv)
        n_processed += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(selected):
            log.emit("INFO", "eval.file.done",
                     file_idx=idx + 1,
                     conv_id=conv_id,
                     n_pred=len(normalized),
                     n_gt=len(gt_for_conv),
                     n_coerced=n_coerced)

    # ---- aggregate boundary-F1 (label-blind) ----
    precision, recall, f1, tp, fp, fn = boundary_f1(all_preds, all_gt, COLLAR_S)

    log.emit("INFO", "eval.boundary_f1",
             precision=precision, recall=recall, f1=f1,
             tp=tp, fp=fp, fn=fn,
             n_files=n_processed)

    # ---- per-class F1 (filtered preds; unknown excluded) ----
    gt_cv = [g for g in all_gt if g[1] == "cross_voice"]
    gt_sv = [g for g in all_gt if g[1] == "same_voice_edit"]
    cross_voice_f1 = compute_class_f1(all_preds, gt_cv, "cross_voice", COLLAR_S)
    same_voice_edit_f1 = compute_class_f1(all_preds, gt_sv, "same_voice_edit", COLLAR_S)

    log.emit("INFO", "eval.class_f1.cross_voice",
             f1=cross_voice_f1,
             n_preds=sum(1 for p in all_preds if p[1] == "cross_voice"),
             n_gt=len(gt_cv))
    log.emit("INFO", "eval.class_f1.same_voice_edit",
             f1=same_voice_edit_f1,
             n_preds=sum(1 for p in all_preds if p[1] == "same_voice_edit"),
             n_gt=len(gt_sv))

    log.emit("INFO", "diag.eval.unknown_label_count",
             count=unknown_label_count,
             sampled_files=n_processed)

    elapsed = time.time() - t_start
    log.emit("INFO", "eval.complete",
             combined=f1,
             n_files=n_processed,
             n_skipped=n_skipped,
             n_errors=n_errors,
             elapsed_s=elapsed)

    # ---- wrapper grep contract ----
    # RESULTS_TSV uses key=value tokens; run_autoresearch.sh uses
    # `_tsv_field` regex parsing on `<key>=<value>` (not positional
    # columns). The wrapper requires `combined=<float>` at minimum.
    results_tsv_parts = [
        f"combined={f1:.6f}",
        f"precision={precision:.6f}",
        f"recall={recall:.6f}",
        f"n_files={n_processed}",
        f"cross_voice_f1={cross_voice_f1:.6f}",
        f"same_voice_edit_f1={same_voice_edit_f1:.6f}",
        f"unknown_label_count={unknown_label_count}",
    ]
    print("RESULTS_TSV: " + " ".join(results_tsv_parts))

    # LAST line: `combined: <float>` (per-spec; preserved as final stdout
    # line so callers using `tail -1 | grep combined:` work).
    print(f"combined: {f1:.6f}")

    return {
        "combined": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_files": n_processed,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
        "cross_voice_f1": cross_voice_f1,
        "same_voice_edit_f1": same_voice_edit_f1,
        "unknown_label_count": unknown_label_count,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audio splice detection evaluation (korean-iter1 boundary-F1)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Path to the korean_iter1 eval directory containing "
             "ground_truth.json and <conv_id>.opus files. "
             f"Default: {DEFAULT_DATA_DIR}",
    )
    # Legacy flags accepted-and-ignored for wrapper compatibility. The
    # wrapper (`run_autoresearch.sh:1200`) invokes
    # `splice/evaluate.py --shap`; rejecting unknown flags would break the
    # eval pipeline. Each ignored flag is logged once via WARN so callers
    # see they are no-ops in the korean-iter1 contract.
    parser.add_argument("--shap", action="store_true",
                        help="(legacy, ignored in korean-iter1)")
    parser.add_argument("--codec", action="store_true",
                        help="(legacy, ignored in korean-iter1)")
    parser.add_argument("--multi", action="store_true",
                        help="(legacy, ignored in korean-iter1)")
    parser.add_argument("--test", action="store_true",
                        help="(legacy, ignored in korean-iter1)")
    args = parser.parse_args(argv)

    for legacy in ("shap", "codec", "multi", "test"):
        if getattr(args, legacy, False):
            log.emit("WARN", "eval.cli.legacy_flag_ignored",
                     flag=f"--{legacy}",
                     reason="korean_iter1_contract_does_not_support_this_flag")

    if not os.path.isdir(args.data_dir):
        log.emit("ERROR", "eval.input.error",
                 kind="data_dir_missing", path=args.data_dir)
        print(f"ERROR: data_dir does not exist: {args.data_dir}", file=sys.stderr)
        # Still emit the wrapper contract lines so the wrapper sees a
        # well-formed but zero result instead of crashing on missing
        # RESULTS_TSV.
        print("RESULTS_TSV: combined=0.000000 precision=0.000000 recall=0.000000 "
              "n_files=0 cross_voice_f1=0.000000 same_voice_edit_f1=0.000000 "
              "unknown_label_count=0")
        print("combined: 0.000000")
        return 1

    evaluate(args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
