"""Per-regime breakdown of DSP-only detector performance.

For each speech dataset, group spliced files by `boundary_energy` (random vs
quiet_matched) and report TP/FP/FN, precision, recall, F1, combined — so we
can see whether the K-DFC realistic attack pattern (quiet_matched) actually
is harder for our detector.
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import numpy as np

from detector import detect_splices
from evaluate import match_detections, TOLERANCE_S
from dataset_registry import DATASETS

# Silence DIAG noise for this ad-hoc script
os.environ["OMC_DIAG_LEVEL"] = "OFF"


def eval_one(data_dir):
    with open(os.path.join(data_dir, "ground_truth.json")) as f:
        gt = json.load(f)

    by_regime = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "files": 0})
    all_clean_fp = 0
    n_clean = 0

    for name, info in sorted(gt.items()):
        audio, sr = sf.read(os.path.join(data_dir, info["path"]), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        gt_times = [info["splice_time_sec"]] if info.get("spliced") else []
        det_times = detect_splices(audio, sr)
        tp, fp, fn, _ = match_detections(gt_times, det_times, TOLERANCE_S)

        if info.get("spliced"):
            regime = info.get("boundary_energy", "unknown")
            by_regime[regime]["tp"] += tp
            by_regime[regime]["fp"] += fp
            by_regime[regime]["fn"] += fn
            by_regime[regime]["files"] += 1
        else:
            all_clean_fp += fp
            n_clean += 1

    # Add a "clean" row for context
    return by_regime, all_clean_fp, n_clean


def summarize(label, by_regime, clean_fp, n_clean):
    print(f"\n=== {label} ===")
    print(f"{'regime':<18s} {'files':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} "
          f"{'prec':>6s} {'recall':>7s} {'F1':>6s}")
    for regime, counts in sorted(by_regime.items()):
        tp, fp, fn, files = counts["tp"], counts["fp"], counts["fn"], counts["files"]
        p = tp / (tp + fp) if (tp + fp) else 1.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"{regime:<18s} {files:>6d} {tp:>4d} {fp:>4d} {fn:>4d} "
              f"{p:>6.2f} {r:>7.2f} {f1:>6.2f}")
    # Aggregate across all regimes
    tot_tp = sum(c["tp"] for c in by_regime.values())
    tot_fp = sum(c["fp"] for c in by_regime.values())
    tot_fn = sum(c["fn"] for c in by_regime.values())
    p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 1.0
    r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    clean_score = max(0.0, 1.0 - clean_fp / max(n_clean, 1))
    combined = f1 * clean_score
    print(f"{'ALL':<18s} {sum(c['files'] for c in by_regime.values()):>6d} "
          f"{tot_tp:>4d} {tot_fp:>4d} {tot_fn:>4d} "
          f"{p:>6.2f} {r:>7.2f} {f1:>6.2f}")
    print(f"clean_fp: {clean_fp}/{n_clean} clean files   "
          f"clean_score: {clean_score:.2f}   combined: {combined:.3f}")


def main():
    speech_datasets = [d for d in DATASETS if d.id in ("korean", "english")]
    t0 = time.time()
    for ds in speech_datasets:
        if not ds.path.exists():
            print(f"SKIP {ds.id}: {ds.path}")
            continue
        by_regime, clean_fp, n_clean = eval_one(str(ds.path))
        summarize(ds.id, by_regime, clean_fp, n_clean)
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
