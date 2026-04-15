"""
Audio splice detection evaluation oracle.
IMMUTABLE — do not modify. The autoresearch loop reads this metric.

Usage:
    uv run python prepare.py
    uv run python prepare.py --data-dir /path/to/spliced
"""

import json
import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf

# Import the detector (the mutable file)
from detector import detect_splices

TOLERANCE_S = 1.0  # ±1s matching tolerance — "이 근처에 편집 있음"


def load_ground_truth(data_dir):
    """Load ground_truth.json and return list of test case dicts."""
    gt_path = os.path.join(data_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        print(f"ERROR: ground_truth.json not found at {gt_path}", file=sys.stderr)
        sys.exit(1)
    with open(gt_path, "r") as f:
        raw = json.load(f)

    cases = []
    for name, entry in raw.items():
        wav_path = os.path.join(data_dir, entry["path"])
        gt_times = []
        if entry.get("spliced", False):
            # Support single splice_time_sec or list of splice times
            st = entry.get("splice_time_sec")
            if isinstance(st, list):
                gt_times = [float(t) for t in st]
            elif st is not None:
                gt_times = [float(st)]
        cases.append({
            "name": name,
            "path": wav_path,
            "spliced": entry.get("spliced", False),
            "gt_times": sorted(gt_times),
            "tier": entry.get("tier", 0),
        })
    return cases


def match_detections(gt_times, det_times, tolerance):
    """
    Greedy closest-first matching.

    Returns (tp, fp, fn) counts.
    Each ground truth can match at most one detection and vice versa.
    """
    if not gt_times and not det_times:
        return 0, 0, 0
    if not gt_times:
        return 0, len(det_times), 0
    if not det_times:
        return 0, 0, len(gt_times)

    # Build all (distance, gt_idx, det_idx) pairs within tolerance
    pairs = []
    for gi, gt in enumerate(gt_times):
        for di, det in enumerate(det_times):
            dist = abs(gt - det)
            if dist <= tolerance:
                pairs.append((dist, gi, di))

    # Greedy: sort by distance, assign closest first
    pairs.sort(key=lambda x: x[0])
    matched_gt = set()
    matched_det = set()
    tp = 0
    for dist, gi, di in pairs:
        if gi not in matched_gt and di not in matched_det:
            tp += 1
            matched_gt.add(gi)
            matched_det.add(di)

    fp = len(det_times) - len(matched_det)
    fn = len(gt_times) - len(matched_gt)
    return tp, fp, fn


def evaluate(data_dir):
    """Run detector on all test files and compute aggregate metrics."""
    cases = load_ground_truth(data_dir)
    if not cases:
        print("ERROR: no test cases found", file=sys.stderr)
        sys.exit(1)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_files = len(cases)
    clean_files = 0
    clean_fp = 0
    errors = 0

    print(f"Evaluating {total_files} files from {data_dir}")
    print("-" * 60)

    for case in cases:
        name = case["name"]
        wav_path = case["path"]

        if not os.path.exists(wav_path):
            print(f"  SKIP {name}: file not found at {wav_path}")
            errors += 1
            continue

        # Load audio
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Run detector
        try:
            det_times = detect_splices(audio, sr)
        except Exception as e:
            print(f"  ERROR {name}: detector raised {e}")
            errors += 1
            # Count all ground truth as missed
            total_fn += len(case["gt_times"])
            continue

        # Ensure det_times is a sorted list of floats
        det_times = sorted(float(t) for t in det_times)

        # Match
        tp, fp, fn = match_detections(case["gt_times"], det_times, TOLERANCE_S)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Track clean file FPs
        if not case["spliced"]:
            clean_files += 1
            clean_fp += fp

        # Per-file summary
        status = "OK" if (tp == len(case["gt_times"]) and fp == 0) else "MISS"
        tier_str = f"T{case['tier']}"
        print(f"  {tier_str} {status:4s} {name}: "
              f"TP={tp} FP={fp} FN={fn} "
              f"gt={case['gt_times']} det={det_times}")

    print("-" * 60)

    # Aggregate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fp_rate = total_fp / total_files if total_files > 0 else 0.0

    # Clean score: 1.0 if no FP on clean files, 0.0 if all clean files have FP
    clean_score = 1.0 - (clean_fp / max(clean_files, 1))
    clean_score = max(0.0, clean_score)

    # Combined: F1 × clean_score. Both must be high to score well.
    combined = f1 * clean_score

    # Primary metric (what the autoresearch loop optimizes)
    print(f"splice_f1: {f1:.6f}")
    print(f"clean_score: {clean_score:.6f}")
    print(f"combined: {combined:.6f}")

    # Secondary metrics
    print(f"precision: {precision:.2f}")
    print(f"recall: {recall:.2f}")
    print(f"fp_rate: {fp_rate:.2f}")

    # Extra detail
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn} clean_fp={clean_fp} files={total_files} errors={errors}")

    return {
        "splice_f1": f1,
        "clean_score": clean_score,
        "combined": combined,
        "precision": precision,
        "recall": recall,
        "fp_rate": fp_rate,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "clean_fp": clean_fp,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio splice detection evaluation oracle"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "spliced"),
        help="Path to spliced test data directory (default: data/spliced/)",
    )
    args = parser.parse_args()

    t0 = time.time()
    evaluate(args.data_dir)
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.1f}s")
