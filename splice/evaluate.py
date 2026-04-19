"""
Audio splice detection evaluation oracle.
Protected from autoresearch agent modification — only the human edits this.

Usage:
    uv run python evaluate.py
    uv run python evaluate.py --shap              # write per-detection SHAP reports
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
from splice.detector import detect_splices

# US-515 phase 1: unified structured logger companion emits.
# Dual-emit only: no print() is deleted; the RESULTS_TSV: contract line
# (search "RESULTS_TSV:" below) is NOT migrated. Covered by the one-time
# maintainer exemption recorded in CLAUDE.md (US-515).
from autoresearch.logger import get_logger
_ev_log = get_logger("eval")

TOLERANCE_S = 1.0  # ±1s matching tolerance — "이 근처에 편집 있음"


def load_ground_truth(data_dir):
    """Load ground_truth.json and return list of test case dicts."""
    gt_path = os.path.join(data_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        print(f"ERROR: ground_truth.json not found at {gt_path}", file=sys.stderr)
        _ev_log.emit("ERROR", "eval.input.error",
                     kind="ground_truth_missing", path=gt_path)
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
            # boundary_energy regime label: "random" / "quiet_matched" / None
            # (used for forensic per-cell breakdown; absent → "unknown").
            "boundary_energy": entry.get("boundary_energy"),
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
        _ev_log.emit("ERROR", "eval.input.error",
                     kind="no_test_cases", data_dir=str(data_dir),
                     fn="evaluate")
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
    # Per-tier × per-regime breakdown for the future 15-cell forensic metric.
    # Key: "t{tier}_{regime}" e.g. "t1_random" / "t2_quiet_matched". Values:
    # {tp, fp, fn, n_files}. Only spliced files contribute; clean files feed
    # clean_score separately.
    by_tier_regime: dict = {}

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

        # Per-tier × per-regime accumulation (spliced only).
        if case["spliced"]:
            regime = case.get("boundary_energy") or "unknown"
            cell = f"t{case['tier']}_{regime}"
            slot = by_tier_regime.setdefault(
                cell, {"tp": 0, "fp": 0, "fn": 0, "n_files": 0}
            )
            slot["tp"] += tp
            slot["fp"] += fp
            slot["fn"] += fn
            slot["n_files"] += 1

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

    _ev_log.emit("INFO", "eval.metrics.splice",
                 data_dir=str(data_dir),
                 splice_f1=f1, clean_score=clean_score, combined=combined)
    _ev_log.emit("INFO", "eval.metrics.clean",
                 data_dir=str(data_dir),
                 precision=precision, recall=recall, fp_rate=fp_rate,
                 tp=total_tp, fp=total_fp, fn=total_fn,
                 clean_fp=clean_fp, files=total_files, errors=errors)

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
    _ev_log.emit("INFO", "eval.fp.distribution",
                 buckets={str(k): v for k, v in fp_dist.items()},
                 n=n, max_fp_count=max_fp_file[1], max_fp_file=max_fp_file[0])

    # Crossfade breakdown (T2 only)
    if xfade_results:
        _xfade_summary = {}
        for xf in sorted(xfade_results):
            tp_xf = xfade_results[xf]["tp"]
            tot_xf = xfade_results[xf]["total"]
            pct = int(100 * tp_xf / tot_xf) if tot_xf > 0 else 0
            _xfade_summary[str(xf)] = {
                "tp": tp_xf, "total": tot_xf, "pct": pct,
                "misses": list(xfade_results[xf].get("misses", [])),
            }
        _ev_log.emit("INFO", "eval.crossfade.breakdown", per_xf=_xfade_summary)

    # Localization accuracy (distance between detected and ground truth time for TPs)
    if all_loc_distances:
        loc_mean = float(np.mean(all_loc_distances))
        loc_median = float(np.median(all_loc_distances))
        loc_max = float(np.max(all_loc_distances))
    else:
        loc_mean = loc_median = loc_max = 0.0
    _ev_log.emit("INFO", "eval.loc_accuracy",
                 mean_s=loc_mean, median_s=loc_median, max_s=loc_max,
                 n_samples=len(all_loc_distances))

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
        "by_tier_regime": by_tier_regime,
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
        _ev_log.emit("ERROR", "eval.input.error",
                     kind="no_test_cases", data_dir=str(data_dir),
                     fn="evaluate_codec")
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

    _ev_log.emit("INFO", "eval.opus32k.metrics",
                 f1=f1, combined=combined, precision=precision, recall=recall,
                 tp=total_tp, fp=total_fp, fn=total_fn, clean_fp=clean_fp,
                 files=total_files, errors=errors, codec_errors=codec_errors,
                 data_dir=str(data_dir))

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

    _ev_log.emit("INFO", "eval.opus32k.aggregate",
                 n_datasets=len(data_dirs),
                 f1=f1, combined=combined, precision=precision, recall=recall,
                 tp=total_tp, fp=total_fp, fn=total_fn,
                 clean_fp=total_clean_fp, clean_total=total_clean,
                 codec_errors=total_codec_errors)


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
    _ev_log.emit("INFO", "eval.aggregate",
                 mode="multi",
                 n_datasets=len(data_dirs),
                 splice_f1=f1, clean_score=clean_score, combined=combined,
                 precision=precision, recall=recall,
                 tp=total_tp, fp=total_fp, fn=total_fn,
                 clean_fp=total_clean_fp, clean_total=total_clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio splice detection evaluation oracle"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Single-dataset override for ad-hoc debugging. If omitted the "
             "evaluator iterates every entry in dataset_registry.DATASETS and "
             "reports the geometric-mean `combined` across them.",
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
        "--shap", action="store_true",
        help="Write per-detection SHAP explanation JSONs under reports/<git_sha>/",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Evaluate the held-out test split (data/test/<domain>/) instead "
             "of the eval split. Longer duration envelope, 20-min budget; "
             "NOT the metric autoresearch optimizes.",
    )
    args = parser.parse_args()

    t0 = time.time()
    if args.multi:
        # Legacy: --multi runs both normal and quiet variants of a single
        # tree. Retained for ad-hoc debugging; does not feed the cross-dataset
        # GM.
        base = os.path.dirname(os.path.realpath(
            args.data_dir or os.path.join(os.path.dirname(__file__), "data", "spliced")
        ))
        primary = args.data_dir or os.path.join(base, "spliced")
        dirs = [primary]
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
    elif args.data_dir:
        # Single-dataset ad-hoc mode. Prints `combined:` for THIS dataset
        # only; not aggregated into the GM.
        result = evaluate(args.data_dir)
        if args.codec:
            evaluate_codec(args.data_dir)
        if args.shap:
            from splice.ml_eval import export_shap_reports
            export_shap_reports(result, args.data_dir)
        _ev_log.emit("INFO", "eval.single.combined",
                     data_dir=str(args.data_dir), combined=result["combined"])
    else:
        # Default: iterate dataset_registry.DATASETS and report the
        # geometric-mean `combined` across all eval_weight>0 entries. This
        # is the metric autoresearch optimizes.
        from splice.dataset_registry import DATASETS, aggregate_combined
        if args.shap:
            from splice.ml_eval import export_shap_reports

        split_label = "test" if args.test else "eval"

        # If --test and the plaintext tree is missing but the encrypted
        # blob exists, invoke scripts/test_crypto.py decrypt to materialize
        # the dataset into a fresh /tmp dir gated by Touch ID. Register
        # cleanup so the plaintext is shredded on any exit path.
        test_root_override = None
        if args.test:
            _repo = os.path.dirname(os.path.abspath(__file__))
            _enc_blob = os.path.join(_repo, "data", "test.tar.gz.enc")
            _plain_any = any(
                os.path.isdir(str(ds.test_path)) for ds in DATASETS
            )
            if not _plain_any and os.path.exists(_enc_blob):
                import atexit
                import shutil as _shutil
                _script = os.path.join(_repo, "scripts", "test_crypto.py")
                print("Test dataset is encrypted. Triggering Touch ID prompt "
                      "to decrypt...", file=sys.stderr)
                _r = subprocess.run(
                    ["uv", "run", "python", _script, "decrypt", "--keep"],
                    capture_output=True, text=True,
                )
                if _r.returncode != 0:
                    print(f"test decrypt failed: {_r.stderr}", file=sys.stderr)
                    _ev_log.emit("ERROR", "eval.input.error",
                                 kind="test_decrypt_failed",
                                 stderr=_r.stderr.strip())
                    sys.exit(1)
                test_root_override = _r.stdout.strip()
                _parent_tmp = os.path.dirname(test_root_override)
                atexit.register(
                    lambda: _shutil.rmtree(_parent_tmp, ignore_errors=True)
                )
                print(f"decrypted test root: {test_root_override}",
                      file=sys.stderr)

        per_dataset: dict[str, float] = {}
        per_dataset_clean_fp: dict[str, int] = {}

        for ds in DATASETS:
            if ds.eval_weight <= 0:
                continue
            if args.test and test_root_override:
                ds_path = os.path.join(test_root_override, ds.id)
            else:
                ds_path = str(ds.test_path if args.test else ds.eval_path)
            if not os.path.isdir(ds_path):
                print(f"combined_{ds.id}: MISSING (no such directory: {ds_path})")
                continue
            print(f"\n=== Evaluating {split_label}: {ds.id} ({ds_path}) ===")
            try:
                ds_result = evaluate(ds_path)
                per_dataset[ds.id] = ds_result["combined"]
                per_dataset_clean_fp[ds.id] = ds_result["clean_fp"]
                if args.shap:
                    export_shap_reports(ds_result, ds_path)
                if args.codec:
                    evaluate_codec(ds_path)
            except Exception as _e:
                per_dataset[ds.id] = 0.0
                per_dataset_clean_fp[ds.id] = 0
                print(f"combined_{ds.id}: ERROR ({type(_e).__name__}: {_e})")
                _ev_log.emit("ERROR", "eval.dataset.error",
                             ds_id=ds.id, exc_type=type(_e).__name__,
                             message=str(_e))
            print(f"=== End {ds.id} ===\n")

        # Aggregate — geometric mean with floor to avoid zero-collapse on
        # new / untested datasets.
        agg = aggregate_combined(per_dataset, method="geometric", floor=0.01)
        total_clean_fp = sum(per_dataset_clean_fp.values())
        _ev_log.emit("INFO", "eval.dataset.result",
                     per_dataset=per_dataset,
                     per_dataset_clean_fp=per_dataset_clean_fp)
        _ev_log.emit("INFO", "eval.aggregate",
                     mode="cross-dataset",
                     combined=agg["combined"],
                     combined_mean=agg["combined_mean"],
                     combined_min=agg["combined_min"],
                     clean_fp=total_clean_fp,
                     n_datasets=len(per_dataset),
                     per_dataset=per_dataset,
                     per_dataset_clean_fp=per_dataset_clean_fp)
        # RESULTS_TSV: the bash wrapper (run_autoresearch.sh) still greps
        # this verbatim at 5 sites. Stays as a phase-2 carve-out until the
        # bash migration lands; the Python-side verify_agent / retest
        # path now reads `eval.aggregate` JSONL instead of re-parsing it.
        tsv_parts = [
            f"combined={agg['combined']:.6f}",
            f"combined_mean={agg['combined_mean']:.6f}",
            f"combined_min={agg['combined_min']:.6f}",
            f"clean_fp={total_clean_fp}",
            f"n_datasets={len(per_dataset)}",
        ]
        for ds_id in sorted(per_dataset):
            tsv_parts.append(f"combined_{ds_id}={per_dataset[ds_id]:.6f}")
            tsv_parts.append(f"clean_fp_{ds_id}={per_dataset_clean_fp.get(ds_id, 0)}")
        print("RESULTS_TSV: " + " ".join(tsv_parts))
    elapsed = time.time() - t0
    _ev_log.emit("INFO", "eval.run.complete", elapsed_s=elapsed)
