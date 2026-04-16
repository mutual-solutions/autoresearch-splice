"""
Audio splice detection evaluation oracle.
Protected from autoresearch agent modification — only the human edits this.

Usage:
    uv run python evaluate.py
    uv run python evaluate.py --with-classifier   # DSP + ML pipeline
    uv run python evaluate.py --data-dir /path/to/spliced
    uv run python evaluate.py --codec             # also test Opus 32k roundtrip
"""

import json
import os
import subprocess
import sys
import tempfile
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
            "crossfade_ms": entry.get("crossfade_ms"),
        })
    return cases


def match_detections(gt_times, det_times, tolerance):
    """
    Greedy closest-first matching.

    Returns (tp, fp, fn, tp_distances) where tp_distances is a list of
    abs(gt - det) for each matched TP.
    Each ground truth can match at most one detection and vice versa.
    """
    if not gt_times and not det_times:
        return 0, 0, 0, []
    if not gt_times:
        return 0, len(det_times), 0, []
    if not det_times:
        return 0, 0, len(gt_times), []

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
    tp_distances = []
    for dist, gi, di in pairs:
        if gi not in matched_gt and di not in matched_det:
            tp += 1
            tp_distances.append(dist)
            matched_gt.add(gi)
            matched_det.add(di)

    fp = len(det_times) - len(matched_det)
    fn = len(gt_times) - len(matched_gt)
    return tp, fp, fn, tp_distances


FFMPEG = "/opt/homebrew/bin/ffmpeg"


def codec_roundtrip(wav_path, tmpdir):
    """Encode WAV -> Opus 32k -> decode back to WAV. Returns path or None on error."""
    ogg_path = os.path.join(tmpdir, "opus_test.ogg")
    out_path = os.path.join(tmpdir, "opus_back.wav")
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "32k", ogg_path],
            capture_output=True, check=True, timeout=30,
        )
        subprocess.run(
            [FFMPEG, "-y", "-i", ogg_path, "-ac", "1", "-ar", "44100", out_path],
            capture_output=True, check=True, timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        return None
    return out_path


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
    per_file_results = []
    fp_per_file = {}   # name -> fp count
    xfade_results = {}  # crossfade_ms -> {"tp": int, "total": int}
    all_loc_distances = []  # distances between TP detections and ground truth

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
        tp, fp, fn, tp_distances = match_detections(case["gt_times"], det_times, TOLERANCE_S)
        total_tp += tp
        all_loc_distances.extend(tp_distances)
        total_fp += fp
        total_fn += fn

        # Track per-file FP
        fp_per_file[name] = fp

        # Collect per-file results for ML pipeline
        per_file_results.append({
            "name": name,
            "path": wav_path,
            "gt_times": case["gt_times"],
            "det_times": det_times,
            "spliced": case["spliced"],
            "tier": case["tier"],
        })

        # Track T2 crossfade breakdown
        if case["tier"] == 2 and case["spliced"] and case.get("crossfade_ms") is not None:
            xf = case["crossfade_ms"]
            if xf not in xfade_results:
                xfade_results[xf] = {"tp": 0, "total": 0, "misses": []}
            xfade_results[xf]["tp"] += tp
            xfade_results[xf]["total"] += len(case["gt_times"])
            if fn > 0 or (tp == 0 and fp > 0):
                off = min(abs(g - d) for g in case["gt_times"] for d in det_times) if det_times else float("inf")
                xfade_results[xf]["misses"].append(
                    f"{name}: gt={case['gt_times']} det={det_times} (closest_off={off:.2f}s)"
                )

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

    # FP distribution
    fp_dist = {0: 0, 1: 0, 2: 0, "3+": 0}
    for fp_count in fp_per_file.values():
        if fp_count == 0:
            fp_dist[0] += 1
        elif fp_count == 1:
            fp_dist[1] += 1
        elif fp_count == 2:
            fp_dist[2] += 1
        else:
            fp_dist["3+"] += 1
    max_fp_file = max(fp_per_file.items(), key=lambda x: x[1]) if fp_per_file else ("", 0)
    n = len(fp_per_file)
    print("--- FP distribution ---")
    for bucket, label in [(0, "0"), (1, "1"), (2, "2"), ("3+", "3+")]:
        cnt = fp_dist[bucket]
        pct = int(round(cnt / n * 100)) if n > 0 else 0
        print(f"  Files with {label} FP: {cnt}/{n} ({pct}%)")
    print(f"  Max FP per file: {max_fp_file[1]} ({max_fp_file[0]})")

    # Crossfade breakdown (T2 only)
    if xfade_results:
        print("--- Crossfade breakdown ---")
        for xf in sorted(xfade_results):
            tp_xf = xfade_results[xf]["tp"]
            tot_xf = xfade_results[xf]["total"]
            pct = int(100 * tp_xf / tot_xf) if tot_xf > 0 else 0
            print(f"  {xf:>4}ms: {tp_xf}/{tot_xf} ({pct}%)")
            for miss in xfade_results[xf].get("misses", []):
                print(f"    MISS {miss}")

    # Localization accuracy (distance between detected and ground truth time for TPs)
    if all_loc_distances:
        loc_mean = float(np.mean(all_loc_distances))
        loc_median = float(np.median(all_loc_distances))
        loc_max = float(np.max(all_loc_distances))
    else:
        loc_mean = loc_median = loc_max = 0.0
    print(f"loc_accuracy: mean={loc_mean:.3f}s, median={loc_median:.3f}s, max={loc_max:.3f}s")

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
        "fp_distribution": fp_dist,
        "max_fp_file": max_fp_file,
        "t2_by_xfade": {xf: {"tp": v["tp"], "total": v["total"]} for xf, v in xfade_results.items()},
        "loc_distances": all_loc_distances,
        "loc_mean": loc_mean,
        "loc_median": loc_median,
        "per_file": per_file_results,
        "clean_files": clean_files,
    }


def evaluate_codec(data_dir):
    """Run detector on Opus 32k roundtripped files and compute codec-specific metrics."""
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
    codec_errors = 0

    print(f"\nOpus 32k codec evaluation on {total_files} files from {data_dir}")
    print("-" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        for case in cases:
            name = case["name"]
            wav_path = case["path"]

            if not os.path.exists(wav_path):
                print(f"  SKIP {name}: file not found at {wav_path}")
                errors += 1
                continue

            # Codec roundtrip
            decoded_path = codec_roundtrip(wav_path, tmpdir)
            if decoded_path is None:
                print(f"  CODEC_ERR {name}: ffmpeg failed, skipping")
                codec_errors += 1
                total_fn += len(case["gt_times"])
                continue

            # Load decoded audio
            audio, sr = sf.read(decoded_path, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            # Run detector
            try:
                det_times = detect_splices(audio, sr)
            except Exception as e:
                print(f"  ERROR {name}: detector raised {e}")
                errors += 1
                total_fn += len(case["gt_times"])
                continue

            det_times = sorted(float(t) for t in det_times)

            # Match
            tp, fp, fn, tp_distances = match_detections(case["gt_times"], det_times, TOLERANCE_S)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            if not case["spliced"]:
                clean_files += 1
                clean_fp += fp

            status = "OK" if (tp == len(case["gt_times"]) and fp == 0) else "MISS"
            tier_str = f"T{case['tier']}"
            print(f"  {tier_str} {status:4s} {name}: "
                  f"TP={tp} FP={fp} FN={fn} "
                  f"gt={case['gt_times']} det={det_times}")

    print("-" * 60)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    clean_score = 1.0 - (clean_fp / max(clean_files, 1))
    clean_score = max(0.0, clean_score)
    combined = f1 * clean_score

    print(f"opus32k_f1: {f1:.6f}")
    print(f"opus32k_combined: {combined:.6f}")
    print(f"opus32k_precision: {precision:.2f}")
    print(f"opus32k_recall: {recall:.2f}")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn} clean_fp={clean_fp} "
          f"files={total_files} errors={errors} codec_errors={codec_errors}")

    return {
        "opus32k_f1": f1,
        "opus32k_combined": combined,
        "opus32k_precision": precision,
        "opus32k_recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "clean_fp": clean_fp,
        "codec_errors": codec_errors,
    }


def evaluate_multi_codec(data_dirs):
    """Aggregate Opus 32k codec metrics across multiple data directories."""
    total_tp, total_fp, total_fn, total_clean_fp = 0, 0, 0, 0
    total_clean = 0
    total_codec_errors = 0

    for d in data_dirs:
        if not os.path.exists(d):
            continue
        result = evaluate_codec(d)
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        total_clean_fp += result["clean_fp"]
        total_codec_errors += result["codec_errors"]
        gt_path = os.path.join(d, "ground_truth.json")
        with open(gt_path) as f:
            gt = json.load(f)
        n_clean = sum(1 for e in gt.values() if not e.get("spliced", False))
        total_clean += n_clean

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    clean_score = 1.0 - (total_clean_fp / max(total_clean, 1))
    combined = f1 * clean_score

    print(f"\n{'='*60}")
    print(f"AGGREGATE OPUS 32k across {len(data_dirs)} datasets:")
    print(f"opus32k_f1: {f1:.6f}")
    print(f"opus32k_combined: {combined:.6f}")
    print(f"opus32k_precision: {precision:.2f}")
    print(f"opus32k_recall: {recall:.2f}")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn} clean_fp={total_clean_fp} "
          f"clean_total={total_clean} codec_errors={total_codec_errors}")


def evaluate_multi(data_dirs):
    """Evaluate across multiple data directories and aggregate."""
    total_tp, total_fp, total_fn, total_clean_fp = 0, 0, 0, 0
    total_files, total_clean = 0, 0

    for d in data_dirs:
        if not os.path.exists(d):
            continue
        result = evaluate(d)
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        total_clean_fp += result["clean_fp"]
        total_files += result["tp"] + result["fn"]  # spliced files
        # count clean files from ground truth
        gt_path = os.path.join(d, "ground_truth.json")
        with open(gt_path) as f:
            gt = json.load(f)
        n_clean = sum(1 for e in gt.values() if not e.get("spliced", False))
        total_clean += n_clean

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    clean_score = 1.0 - (total_clean_fp / max(total_clean, 1))

    combined = f1 * clean_score
    print(f"\n{'='*60}")
    print(f"AGGREGATE across {len(data_dirs)} datasets:")
    print(f"splice_f1: {f1:.6f}")
    print(f"clean_score: {clean_score:.6f}")
    print(f"combined: {combined:.6f}")
    print(f"precision: {precision:.2f}")
    print(f"recall: {recall:.2f}")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn} clean_fp={total_clean_fp} clean_total={total_clean}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio splice detection evaluation oracle"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "spliced"),
        help="Path to spliced test data directory (default: data/spliced/)",
    )
    parser.add_argument(
        "--multi", action="store_true",
        help="Evaluate on both normal and quiet datasets",
    )
    parser.add_argument(
        "--codec", action="store_true",
        help="Also evaluate after Opus 32k codec roundtrip (KakaoTalk standard)",
    )
    parser.add_argument(
        "--with-classifier", action="store_true",
        help="Run DSP + ML classifier pipeline (train + OOF filter + combined_full)",
    )
    args = parser.parse_args()

    t0 = time.time()
    if args.multi:
        base = os.path.dirname(os.path.realpath(args.data_dir))
        dirs = [args.data_dir]
        quiet = os.path.join(base, "spliced_quiet")
        if os.path.exists(quiet):
            dirs.append(quiet)
        for d in dirs:
            evaluate(d)
        evaluate_multi(dirs)
        if args.codec:
            for d in dirs:
                evaluate_codec(d)
            evaluate_multi_codec(dirs)
    else:
        result = evaluate(args.data_dir)
        if args.codec:
            evaluate_codec(args.data_dir)
        if args.with_classifier:
            from ml_eval import evaluate_with_classifier
            ml_result = evaluate_with_classifier(result, args.data_dir)
            if ml_result.get("bound_exceeded"):
                print(f"combined: 0.000000")
            elif ml_result.get("combined_full") is not None:
                print(f"combined: {ml_result['combined_full']:.6f}")
            else:
                print(f"combined: {result['combined']:.6f}")
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.1f}s")
