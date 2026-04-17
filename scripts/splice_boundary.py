"""Boundary-aware splice point selection.

Three regimes:
  - 'random'         — uniform random in [lo, hi] (current, easy baseline)
  - 'quiet'          — splice time where audio_a local RMS is in bottom quiet_percentile
  - 'quiet_matched'  — both audio_a (pre) and audio_b (post) local RMS are in the
                       bottom quiet_percentile AND their dB values are within
                       match_tolerance_db of each other. Mirrors the K-DFC attack
                       pattern (quiet-to-quiet concatenation with level-matching).

Each generator records which regime was used in ground_truth.json as
`boundary_energy` plus the measured dB values at the splice point
(`boundary_rms_db_a`, `boundary_rms_db_b`) for later per-regime metric analysis.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def compute_rms_db(audio: np.ndarray, sr: int, win_s: float = 0.5) -> np.ndarray:
    """Sliding-window RMS in dB. `win_s` default 500ms matches human perception
    of short-term loudness and is wide enough to be stable across phonemes.
    """
    win = max(1, int(win_s * sr))
    sq = audio.astype(np.float64) ** 2
    rms = np.sqrt(uniform_filter1d(sq, size=win, mode="nearest"))
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def find_splice_point(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sr: int,
    rng: np.random.RandomState,
    mode: str = "random",
    splice_range: tuple[float, float] = (0.3, 0.7),
    quiet_percentile: float = 30.0,
    match_tolerance_db: float = 3.0,
    win_s: float = 0.5,
    margin_samples: int = 0,
) -> tuple[int, dict]:
    """Pick a sample index T where `audio_a[:T] + audio_b[T:]` will be spliced.

    Returns (sample_index, info_dict). info_dict carries the measured RMS dB
    values at the chosen splice point on both sides — use these to populate
    the ground_truth manifest for later per-regime analysis.

    Raises ValueError if the regime can't be satisfied. Callers should either
    retry with a different file pair or fall back to a weaker regime explicitly.
    """
    min_len = min(len(audio_a), len(audio_b))
    lo_s = int(splice_range[0] * min_len) + margin_samples
    hi_s = int(splice_range[1] * min_len) - margin_samples
    if hi_s <= lo_s:
        raise ValueError(
            f"splice_range yields empty window: lo={lo_s}, hi={hi_s}, min_len={min_len}"
        )

    if mode == "random":
        t = int(rng.uniform(lo_s, hi_s))
        # Still measure the dB for recordkeeping — cheap and useful.
        db_a = compute_rms_db(audio_a[:min_len], sr, win_s=win_s)
        db_b = compute_rms_db(audio_b[:min_len], sr, win_s=win_s)
        return t, {
            "mode": mode,
            "rms_db_a": float(db_a[t]),
            "rms_db_b": float(db_b[t]),
            "n_candidates": hi_s - lo_s,
        }

    # For quiet modes we need the sliding-RMS curves
    db_a = compute_rms_db(audio_a[:min_len], sr, win_s=win_s)
    db_b = compute_rms_db(audio_b[:min_len], sr, win_s=win_s)
    seg_a = db_a[lo_s:hi_s]
    seg_b = db_b[lo_s:hi_s]

    if mode == "quiet":
        thresh_a = float(np.percentile(seg_a, quiet_percentile))
        mask = seg_a < thresh_a
    elif mode == "quiet_matched":
        thresh_a = float(np.percentile(seg_a, quiet_percentile))
        thresh_b = float(np.percentile(seg_b, quiet_percentile))
        mask = (
            (seg_a < thresh_a)
            & (seg_b < thresh_b)
            & (np.abs(seg_a - seg_b) < match_tolerance_db)
        )
    else:
        raise ValueError(f"Unknown boundary_energy mode: {mode!r}")

    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        raise ValueError(
            f"No candidates for mode={mode!r} "
            f"(pool=[{lo_s},{hi_s}], quiet_percentile={quiet_percentile}, "
            f"match_tol={match_tolerance_db}dB)"
        )

    rel = int(rng.choice(candidates))
    t = lo_s + rel
    return t, {
        "mode": mode,
        "rms_db_a": float(db_a[t]),
        "rms_db_b": float(db_b[t]),
        "n_candidates": int(len(candidates)),
    }


if __name__ == "__main__":
    # Sanity self-test: synthetic audio with a clear quiet region
    sr = 16000
    dur = 30.0
    n = int(sr * dur)
    t_arr = np.arange(n) / sr
    # Loud sine except for a quiet window 10-20s
    audio_a = 0.3 * np.sin(2 * np.pi * 440 * t_arr)
    audio_b = 0.3 * np.sin(2 * np.pi * 330 * t_arr)
    # Make t in [10, 20]s much quieter on both
    mask = (t_arr > 10) & (t_arr < 20)
    audio_a[mask] *= 0.01
    audio_b[mask] *= 0.01

    rng = np.random.RandomState(42)

    # 'quiet_matched' should fall inside [10, 20]s
    t_s, info = find_splice_point(audio_a, audio_b, sr, rng, mode="quiet_matched")
    t_sec = t_s / sr
    assert 10.0 < t_sec < 20.0, f"quiet_matched landed at {t_sec:.2f}s, expected [10, 20]"
    assert info["rms_db_a"] < -30, info
    assert info["rms_db_b"] < -30, info

    # 'random' can land anywhere in [0.3, 0.7] × 30 = [9, 21]s
    t_s2, info2 = find_splice_point(audio_a, audio_b, sr, rng, mode="random")
    t_sec2 = t_s2 / sr
    assert 9.0 <= t_sec2 <= 21.0, (t_sec2, info2)

    print("splice_boundary: self-tests PASS")
    print(f"  quiet_matched: t={t_sec:.2f}s, rms_db_a={info['rms_db_a']:.1f}, "
          f"rms_db_b={info['rms_db_b']:.1f}, n_candidates={info['n_candidates']}")
    print(f"  random:        t={t_sec2:.2f}s, rms_db_a={info2['rms_db_a']:.1f}, "
          f"rms_db_b={info2['rms_db_b']:.1f}")
