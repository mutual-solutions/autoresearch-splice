"""Duration-vs-accuracy sweep on the speech splice datasets.

Centers a window of width D around each file's GT splice time (or at the start
for clean files), runs the current detector, aggregates combined/recall/clean_fp
per D. Isolates the length effect from dataset composition by holding content
constant.

Output: sweep_duration.png
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import match_detections, TOLERANCE_S
from detector import detect_splices

DURATIONS_S = [15, 30, 60, 90, 120, 180, 240, 300]

DATASETS = {
    "korean": "corpora/data/korean-splice",
    "english": "corpora/data/english-splice",
}


def truncate_around_gt(audio, sr, gt_time, window_s):
    """Cut a window of width `window_s` centered on GT. Returns (audio, new_gt_time).
    If the file is shorter than window_s, return what we have.
    """
    n = len(audio)
    full_dur = n / sr
    if window_s >= full_dur:
        return audio, gt_time
    half = window_s / 2.0
    lo_t = max(0.0, gt_time - half)
    hi_t = min(full_dur, lo_t + window_s)
    lo_t = hi_t - window_s  # re-anchor if clipped on the right
    lo_s = int(round(lo_t * sr))
    hi_s = lo_s + int(round(window_s * sr))
    return audio[lo_s:hi_s], gt_time - lo_t


def truncate_from_start(audio, sr, window_s):
    n_samples = int(round(window_s * sr))
    return audio[:n_samples]


def run_dataset(data_dir, duration_s):
    gt_path = os.path.join(data_dir, "ground_truth.json")
    with open(gt_path) as f:
        gt_all = json.load(f)

    tp = fp = fn = 0
    clean_fp = 0
    n_spliced = n_clean = 0

    for name, info in sorted(gt_all.items()):
        path = os.path.join(data_dir, info["path"])
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if info.get("spliced"):
            n_spliced += 1
            gt_time = info["splice_time_sec"]
            a_cut, new_gt = truncate_around_gt(audio, sr, gt_time, duration_s)
            gt_times = [new_gt]
        else:
            n_clean += 1
            a_cut = truncate_from_start(audio, sr, duration_s)
            gt_times = []

        det_times = detect_splices(a_cut, sr)
        _tp, _fp, _fn, _ = match_detections(gt_times, det_times, TOLERANCE_S)
        tp += _tp
        fp += _fp
        fn += _fn
        if not info.get("spliced"):
            clean_fp += _fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    clean_score = max(0.0, 1.0 - clean_fp / max(n_clean, 1))
    combined = f1 * clean_score

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "clean_fp": clean_fp, "clean_files": n_clean,
        "precision": precision, "recall": recall, "f1": f1,
        "clean_score": clean_score, "combined": combined,
    }


def main():
    results = {lang: [] for lang in DATASETS}

    for lang, path in DATASETS.items():
        if not os.path.isdir(path):
            print(f"SKIP {lang}: {path} missing")
            continue
        print(f"\n=== Sweeping {lang} ===")
        for D in DURATIONS_S:
            t0 = time.time()
            r = run_dataset(path, D)
            elapsed = time.time() - t0
            r["D"] = D
            r["elapsed"] = elapsed
            results[lang].append(r)
            print(f"  D={D:3d}s  combined={r['combined']:.4f}  f1={r['f1']:.3f}  "
                  f"recall={r['recall']:.2f}  clean_fp={r['clean_fp']:2d}  "
                  f"elapsed={elapsed:.1f}s")

    # Save raw
    with open("sweep_duration.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"korean": "tab:blue", "english": "tab:orange"}
    for ax, metric, ylabel in [
        (axes[0, 0], "combined", "combined"),
        (axes[0, 1], "recall", "recall"),
        (axes[1, 0], "f1", "splice_f1"),
        (axes[1, 1], "clean_fp", "clean_fp count"),
    ]:
        for lang, rows in results.items():
            if not rows:
                continue
            xs = [r["D"] for r in rows]
            ys = [r[metric] for r in rows]
            ax.plot(xs, ys, "-o", color=colors.get(lang, "black"), label=lang)
        ax.set_xlabel("Audio duration (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs duration")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if metric in ("combined", "recall", "f1"):
            ax.set_ylim(-0.02, 1.0)

    fig.suptitle("Duration sweep — centered GT window", fontsize=13)
    fig.tight_layout()
    out = "sweep_duration.png"
    fig.savefig(out, dpi=120)
    print(f"\nWrote {out}")
    print(f"Wrote sweep_duration.json")


if __name__ == "__main__":
    main()
