"""
Energy dip detector experiment — LINEAR power domain.

Theory:
  During a linear crossfade of two uncorrelated signals A, B with equal power:
    E(t) = α(t)²·E_A + (1−α(t))²·E_B
  where α goes 1→0 over the crossfade window.

  At midpoint α=0.5:
    E_mid = 0.25·E_A + 0.25·E_B = 0.5·E_avg  →  −3 dB dip

  The quadratic E(t) = α²+(1−α)² profile has a minimum of 0.5 at α=0.5.
  Clean audio does NOT produce this pattern.

Approach:
  1. Decompose audio into K=6 octave subbands via bandpass filters.
  2. Compute per-band instantaneous power (|x|²) in short frames.
  3. Around each candidate splice point, fit the quadratic model
       P(frame) ~ E_A·α(frame)² + E_B·(1−α(frame))²
     and measure the residual vs. a flat (constant power) baseline.
  4. Score = residual_flat / residual_quad  (high → quadratic fits better → splice).

Usage:
  uv run python .omc/experiments/energy_dip_test.py
  uv run python .omc/experiments/energy_dip_test.py \\
      --data-dir /Users/yejunjang/Projects/mutual/audio-splice-detector/data/korean-splice
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import librosa
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

K = 6              # number of octave subbands
FRAME_HOP_S = 0.005   # 5 ms hop for power frames
FRAME_WIN_S = 0.020   # 20 ms window

# Search window around candidate splice: ±SEARCH_S seconds
SEARCH_S = 0.5

# Minimum detectable crossfade duration (seconds)
MIN_XF_S = 0.005   # 5 ms


# ─── Octave band decomposition ────────────────────────────────────────────────

def _octave_band_edges(sr: int, k: int = K) -> list[tuple[float, float]]:
    """Return k octave-spaced (low_hz, high_hz) pairs covering ~200Hz–sr/2."""
    low = 200.0
    bands = []
    for _ in range(k):
        high = min(low * 2.0, sr / 2.0 - 1.0)
        bands.append((low, high))
        low = high
        if low >= sr / 2.0 - 1.0:
            break
    return bands


def _bandpass_power(audio: np.ndarray, sr: int, lo: float, hi: float,
                    hop: int, win: int) -> np.ndarray:
    """Bandpass filter audio, then compute short-time power (linear, watts-like)."""
    from scipy.signal import butter, sosfilt
    nyq = sr / 2.0
    lo_n = max(lo / nyq, 0.001)
    hi_n = min(hi / nyq, 0.999)
    sos = butter(4, [lo_n, hi_n], btype="band", output="sos")
    filtered = sosfilt(sos, audio)

    # Short-time power via squared envelope
    n = len(filtered)
    n_frames = (n - win) // hop + 1
    power = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame = filtered[i * hop: i * hop + win]
        power[i] = np.mean(frame ** 2)
    return power


def compute_subband_power(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Returns array of shape (K, n_frames) with linear power per octave band.
    """
    bands = _octave_band_edges(sr)
    hop = max(1, int(FRAME_HOP_S * sr))
    win = max(hop, int(FRAME_WIN_S * sr))

    powers = []
    for lo, hi in bands:
        p = _bandpass_power(audio, sr, lo, hi, hop, win)
        powers.append(p)

    # Align lengths
    min_len = min(len(p) for p in powers)
    return np.stack([p[:min_len] for p in powers], axis=0)  # (K, n_frames)


# ─── Quadratic crossfade model ────────────────────────────────────────────────

def _alpha_ramp(n: int) -> np.ndarray:
    """Linear ramp α: 1 → 0 over n frames (crossfade profile)."""
    return np.linspace(1.0, 0.0, n)


def fit_quadratic_dip(power_band: np.ndarray) -> float:
    """
    Given a 1-D power profile (linear) over a window assumed to span a
    crossfade, fit the quadratic model:

      P(i) = E_A * α(i)² + E_B * (1−α(i))²

    Returns the score = residual_flat / residual_quad.
    High score → quadratic fits much better → crossfade signature detected.
    Score ≈ 1 → no improvement over flat → no dip.
    """
    n = len(power_band)
    if n < 4:
        return 1.0

    alpha = _alpha_ramp(n)
    beta = 1.0 - alpha

    # Build design matrix for least-squares: [α², β²]
    # P ≈ E_A * α² + E_B * β²
    A = np.stack([alpha ** 2, beta ** 2], axis=1)   # (n, 2)
    y = power_band

    # Non-negative least squares (power can't be negative)
    from scipy.optimize import nnls
    coeffs, _ = nnls(A, y)
    E_A, E_B = coeffs

    pred_quad = E_A * alpha ** 2 + E_B * beta ** 2
    pred_flat = np.full_like(power_band, np.mean(power_band))

    res_quad = np.mean((y - pred_quad) ** 2)
    res_flat = np.mean((y - pred_flat) ** 2)

    if res_quad < 1e-30:
        return 1.0

    return res_flat / res_quad


def energy_dip_score(subband_power: np.ndarray, splice_frame: int,
                     search_frames: int) -> float:
    """
    Score a candidate splice point by the quadratic dip in subband power.

    Averages the fit score across all K bands.
    Returns the median band score (robust to silent bands).
    """
    half = search_frames // 2
    start = max(0, splice_frame - half)
    end = min(subband_power.shape[1], splice_frame + half)
    if end - start < 4:
        return 1.0

    scores = []
    for band_idx in range(subband_power.shape[0]):
        window = subband_power[band_idx, start:end]
        # Skip silent bands
        if np.mean(window) < 1e-10:
            continue
        scores.append(fit_quadratic_dip(window))

    if not scores:
        return 1.0
    return float(np.median(scores))


# ─── Per-file detection ────────────────────────────────────────────────────────

class FileResult(NamedTuple):
    filename: str
    is_spliced: bool
    tier: int | None
    true_splice_s: float | None
    crossfade_ms: float | None
    best_score: float      # highest quadratic-dip score found
    best_time_s: float | None  # time of best score
    detected: bool         # whether a dip was found above threshold


def detect_file(filepath: Path, ground_truth: dict | None,
                threshold: float = 1.5) -> FileResult:
    """Run energy-dip detector on one file."""
    audio, sr = librosa.load(str(filepath), sr=None, mono=True)

    hop_samples = max(1, int(FRAME_HOP_S * sr))
    search_frames = max(4, int(SEARCH_S / FRAME_HOP_S))

    subband_power = compute_subband_power(audio, sr)  # (K, n_frames)
    n_frames = subband_power.shape[1]

    # Ground truth
    is_spliced = False
    tier = None
    true_splice_s = None
    crossfade_ms = None

    if ground_truth is not None:
        is_spliced = ground_truth.get("spliced", False)
        tier = ground_truth.get("tier")
        true_splice_s = ground_truth.get("splice_time_sec")
        crossfade_ms = ground_truth.get("crossfade_ms")

    # Candidate splice frames: scan every 10ms across the full file
    # (skip first/last 1 second)
    step = max(1, int(0.010 / FRAME_HOP_S))
    margin = max(1, int(1.0 / FRAME_HOP_S))
    candidate_frames = range(margin, n_frames - margin, step)

    best_score = 1.0
    best_frame = None
    for f in candidate_frames:
        s = energy_dip_score(subband_power, f, search_frames)
        if s > best_score:
            best_score = s
            best_frame = f

    best_time_s = None
    if best_frame is not None:
        best_time_s = (best_frame * hop_samples) / sr

    detected = best_score >= threshold

    return FileResult(
        filename=filepath.name,
        is_spliced=is_spliced,
        tier=tier,
        true_splice_s=true_splice_s,
        crossfade_ms=crossfade_ms,
        best_score=best_score,
        best_time_s=best_time_s,
        detected=detected,
    )


# ─── Dataset runner ───────────────────────────────────────────────────────────

def run_dataset(data_dir: Path, threshold: float = 1.5) -> list[FileResult]:
    """
    Run on a dataset directory.
    Expected layout:
      data_dir/
        ground_truth.json
        clean/       (or tier1/, tier2/ relative to ground_truth paths)
    """
    gt_path = data_dir / "ground_truth.json"
    gt: dict = {}
    if gt_path.exists():
        with open(gt_path) as f:
            gt = json.load(f)

    results = []
    processed = 0

    for filename, info in sorted(gt.items()):
        rel_path = info.get("path", filename)
        filepath = data_dir / rel_path
        if not filepath.exists():
            # Try flat search
            for subdir in ["clean", "tier1", "tier2"]:
                candidate = data_dir / subdir / filename
                if candidate.exists():
                    filepath = candidate
                    break

        if not filepath.exists():
            print(f"  SKIP (not found): {filename}", flush=True)
            continue

        print(f"  [{processed+1}/{len(gt)}] {filename} ...", end=" ", flush=True)
        try:
            r = detect_file(filepath, info, threshold=threshold)
            results.append(r)
            marker = "DIP" if r.detected else "---"
            print(f"{marker}  score={r.best_score:.3f}  "
                  f"(tier={r.tier}, xf_ms={r.crossfade_ms})", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
        processed += 1

    # Also scan any loose WAV files not in ground truth
    for wav in sorted(data_dir.rglob("*.wav")):
        rel = str(wav.relative_to(data_dir))
        # Check if already processed via gt
        if any(r.filename == wav.name for r in results):
            continue
        print(f"  [extra] {wav.name} ...", end=" ", flush=True)
        try:
            r = detect_file(wav, ground_truth=None, threshold=threshold)
            results.append(r)
            print(f"{'DIP' if r.detected else '---'}  score={r.best_score:.3f}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

    return results


# ─── Summary ──────────────────────────────────────────────────────────────────

def summarize(results: list[FileResult], label: str) -> None:
    t2 = [r for r in results if r.tier == 2]
    t1 = [r for r in results if r.tier == 1]
    clean = [r for r in results if not r.is_spliced]
    unknown = [r for r in results if r.tier is None and r.is_spliced]

    def _rate(subset: list[FileResult]) -> str:
        if not subset:
            return "N/A (0 files)"
        detected = sum(1 for r in subset if r.detected)
        return f"{detected}/{len(subset)} = {100*detected/len(subset):.1f}%"

    def _scores(subset: list[FileResult]) -> str:
        if not subset:
            return "N/A"
        sc = [r.best_score for r in subset]
        return (f"min={min(sc):.3f}  median={np.median(sc):.3f}  "
                f"max={max(sc):.3f}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")
    print(f"  T2 crossfade splices detected : {_rate(t2)}")
    print(f"  T1 hard-cut splices detected  : {_rate(t1)}")
    print(f"  Clean files (false positives) : {_rate(clean)}")
    if unknown:
        print(f"  Unknown spliced files         : {_rate(unknown)}")
    print()
    print(f"  Score distribution:")
    print(f"    T2      : {_scores(t2)}")
    print(f"    T1      : {_scores(t1)}")
    print(f"    Clean   : {_scores(clean)}")
    print()

    # Per-T2 detail
    if t2:
        print("  T2 file details:")
        for r in sorted(t2, key=lambda x: x.best_score, reverse=True):
            dip_delta = ""
            if r.true_splice_s is not None and r.best_time_s is not None:
                dip_delta = f"  Δt={r.best_time_s - r.true_splice_s:+.2f}s"
            xf = f"xf={r.crossfade_ms}ms" if r.crossfade_ms is not None else ""
            marker = "DIP" if r.detected else "---"
            print(f"    [{marker}] {r.filename:30s}  score={r.best_score:.3f}  {xf}{dip_delta}")

    print()
    # Honest assessment
    t2_dr = sum(1 for r in t2 if r.detected) / len(t2) if t2 else 0.0
    fp_rate = sum(1 for r in clean if r.detected) / len(clean) if clean else 0.0
    print(f"  Detection rate (T2): {100*t2_dr:.1f}%")
    print(f"  False positive rate (clean): {100*fp_rate:.1f}%")
    if t2_dr == 0.0:
        print("  ASSESSMENT: 3 dB dip is NOT detectable with this approach on this data.")
    elif t2_dr < 0.3:
        print("  ASSESSMENT: Weak signal — dip detectable in some cases but unreliable.")
    elif fp_rate > 0.2:
        print("  ASSESSMENT: High false positive rate undermines utility.")
    else:
        print("  ASSESSMENT: Dip is detectable with acceptable FP rate.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Energy dip detector experiment")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Users/yejunjang/Projects/mutual/autoresearch-splice/data/spliced"),
        help="Dataset directory containing ground_truth.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Score threshold for declaring a dip detected (default: 1.5)",
    )
    parser.add_argument(
        "--korean",
        action="store_true",
        help="Also run on Korean dataset after main dataset",
    )
    args = parser.parse_args()

    datasets = [(args.data_dir, str(args.data_dir))]
    if args.korean:
        korean_dir = Path(
            "/Users/yejunjang/Projects/mutual/audio-splice-detector/data/korean-splice"
        )
        datasets.append((korean_dir, "Korean dataset"))

    for data_dir, label in datasets:
        print(f"\nRunning on: {data_dir}")
        print("-" * 60)
        results = run_dataset(data_dir, threshold=args.threshold)
        summarize(results, label)


if __name__ == "__main__":
    main()
