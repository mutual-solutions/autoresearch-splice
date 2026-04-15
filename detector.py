"""
Audio splice detector — classical DSP only, no neural networks.

v11: Dual-mode detection.
- Mode 1 (hard cuts): Phase discontinuity — detects tier 1 splices.
- Mode 2 (crossfades): CQT subband PSD change-point detection via
  Hotelling's T² test — detects tier 2 crossfaded splices by finding
  statistically significant shifts in the frequency-band power distribution.
"""

from __future__ import annotations

import argparse
import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d
from scipy.stats import genpareto

WINDOW_S = 60.0
OVERLAP_S = 10.0


def detect_splices(audio: np.ndarray, sr: int) -> list[float]:
    """Run CPE (unified), crossfade, and noise floor detectors."""
    # CPE replaces phase detector (unified amplitude+phase)
    cpe_hits = _detect_cpe(audio, sr)
    xfade_hits = _detect_crossfade(audio, sr)

    all_hits = sorted(cpe_hits + xfade_hits)
    merged = []
    for t in all_hits:
        if not merged or t - merged[-1] > 1.0:
            merged.append(t)
    return merged


# ===== MODE 4: Complex Prediction Error (unified amplitude+phase) =====

def _detect_cpe(audio: np.ndarray, sr: int) -> list[float]:
    """
    Complex Prediction Error: unified amplitude-phase splice detector.

    D(f,t) = X(f,t) - X(f,t-1) · e^{j2πf·hop/sr}
    Score(t) = Σ_f |D(f,t)|²

    AGC normalizes amplitude so quiet regions contribute equally.
    Then GPD threshold on the CPE score curve.
    """
    hop_ms = 10
    hop_s = hop_ms / 1000.0
    n_fft = 1024
    hop_length = max(1, int(sr * hop_ms / 1000))

    # AGC: normalize amplitude so quiet sections are amplified
    agc_window = max(1, int(sr * 0.200))
    sq = audio ** 2
    local_power = uniform_filter1d(sq, size=agc_window, mode='constant')
    local_rms = np.sqrt(np.maximum(local_power, 1e-10))
    audio_agc = audio / local_rms

    # STFT on AGC'd audio
    _, _, Zxx = sp_signal.stft(audio_agc, fs=sr, nperseg=n_fft,
                                noverlap=n_fft - hop_length)

    # Complex prediction error
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    phase_advance = np.exp(1j * 2 * np.pi * freqs * (hop_length / sr))[:, np.newaxis]
    predicted = Zxx[:, :-1] * phase_advance
    error = Zxx[:, 1:] - predicted
    cpe_score = np.sum(np.abs(error) ** 2, axis=0)

    cpe_z = _zscore(cpe_score)

    # Silence suppression (on original audio, not AGC'd)
    silence = _silence_mask(audio, sr, hop_ms=hop_ms, threshold_db=-45)
    min_len = min(len(cpe_z), len(silence))
    cpe_z = cpe_z[:min_len]
    silence = silence[:min_len]

    # Quiet-boundary filter: suppress detections at silence edges
    fused = cpe_z * silence

    # GPD threshold
    non_silent = fused[silence > 0.5]
    n_tests = len(non_silent)
    threshold = _gpd_threshold(non_silent, n_tests=max(n_tests, 1), alpha=0.5)

    peaks = _peak_pick(fused, threshold=threshold, min_dist_s=5.0, hop_s=hop_s)

    # Cap at 1
    if len(peaks) > 1:
        peak_frames = [int(p / hop_s) for p in peaks]
        scores = [fused[min(f, min_len - 1)] for f in peak_frames]
        top_idx = [np.argmax(scores)]
        peaks = [peaks[top_idx[0]]]

    # Quiet boundary filter
    filtered = []
    check_samples = int(0.5 * sr)
    for p in peaks:
        center_sample = int(p * sr)
        left_start = max(0, center_sample - check_samples)
        right_end = min(len(audio), center_sample + check_samples)
        if center_sample - left_start < sr // 10 or right_end - center_sample < sr // 10:
            continue
        left_rms = np.sqrt(np.mean(audio[left_start:center_sample] ** 2))
        right_rms = np.sqrt(np.mean(audio[center_sample:right_end] ** 2))
        left_db = 20 * np.log10(max(left_rms, 1e-10))
        right_db = 20 * np.log10(max(right_rms, 1e-10))
        if left_db > -35 and right_db > -35:
            filtered.append(p)

    # Refine
    refined = []
    for p in filtered:
        r = _refine_splice_point(audio, sr, p, search_radius_s=0.3)
        refined.append(r)
    return refined


# ===== MODE 1: Phase discontinuity (hard cuts) =====

def _detect_phase(audio: np.ndarray, sr: int) -> list[float]:
    duration_s = len(audio) / sr
    if duration_s <= WINDOW_S + 5:
        return _analyze_segment_phase(audio, sr, offset_s=0.0)

    window_samples = int(WINDOW_S * sr)
    step_samples = int((WINDOW_S - OVERLAP_S) * sr)
    all_peaks = []
    pos = 0
    while pos < len(audio):
        end = min(pos + window_samples, len(audio))
        segment = audio[pos:end]
        if len(segment) < sr * 10:
            break
        peaks = _analyze_segment_phase(segment, sr, offset_s=pos / sr)
        all_peaks.extend(peaks)
        pos += step_samples

    all_peaks.sort()
    deduped = []
    for p in all_peaks:
        if not deduped or p - deduped[-1] > 1.0:
            deduped.append(p)
    return deduped


# ===== MODE 3: Noise floor jump (quiet sections) =====

def _detect_noise_floor_jump(audio: np.ndarray, sr: int) -> list[float]:
    """
    Detect splice by comparing noise floor level in quiet regions.
    Activates per-FRAME: only analyzes frames where local RMS < -20 dBFS.
    Loud frames are ignored (phase detector handles those).
    """
    frame_ms = 20
    frame_samples = max(1, int(sr * frame_ms / 1000))
    n_frames = len(audio) // frame_samples
    if n_frames < 50:
        return []

    frames = audio[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))

    # Quiet frames: absolute threshold -20 dBFS
    QUIET_THRESHOLD_DB = -20
    is_quiet = rms_db < QUIET_THRESHOLD_DB

    if np.sum(is_quiet) < 20:
        return []

    # Sliding two-window comparison on quiet-only frames
    hop_frames = max(1, int(0.5 / (frame_ms / 1000)))
    window_frames = max(10, int(5.0 / (frame_ms / 1000)))

    # Minimum fraction of quiet frames required in BOTH windows
    MIN_QUIET_FRAC = 0.15  # at least 15% of window must be quiet

    scores = []
    times = []

    for center in range(window_frames, n_frames - window_frames, hop_frames):
        left_mask = is_quiet[max(0, center - window_frames):center]
        right_mask = is_quiet[center:min(n_frames, center + window_frames)]

        left_quiet_frac = left_mask.sum() / max(len(left_mask), 1)
        right_quiet_frac = right_mask.sum() / max(len(right_mask), 1)

        # Both sides must have enough quiet frames
        if left_quiet_frac < MIN_QUIET_FRAC or right_quiet_frac < MIN_QUIET_FRAC:
            scores.append(0.0)
            times.append(center * frame_ms / 1000.0)
            continue

        left_noise = rms_db[max(0, center - window_frames):center][left_mask]
        right_noise = rms_db[center:min(n_frames, center + window_frames)][right_mask]

        if len(left_noise) < 3 or len(right_noise) < 3:
            scores.append(0.0)
            times.append(center * frame_ms / 1000.0)
            continue

        # Raw Welch's t-statistic (absolute threshold, no z-score)
        diff = abs(left_noise.mean() - right_noise.mean())
        se = np.sqrt(left_noise.var() / len(left_noise) +
                     right_noise.var() / len(right_noise))
        t_stat = diff / max(se, 0.01)

        scores.append(t_stat)
        times.append(center * frame_ms / 1000.0)

    if not scores:
        return []

    scores = np.array(scores)
    times = np.array(times)

    # Absolute threshold on t-statistic: t > 6 is very significant
    NOISE_T_THRESHOLD = 20.0

    min_dist_idx = max(1, int(5.0 / 0.5))
    peaks_idx, props = sp_signal.find_peaks(scores, height=NOISE_T_THRESHOLD,
                                             distance=min_dist_idx)
    if len(peaks_idx) == 0:
        return []

    # Top 1 only
    best = peaks_idx[np.argmax(props['peak_heights'])]
    return [times[best]]


def _step_detector_noise(curve: np.ndarray, half_win: int = 50) -> np.ndarray:
    """Step detector optimized for noise floor: absolute diff of left/right means."""
    n = len(curve)
    if n < 2 * half_win + 1:
        return np.zeros(n)
    cs = np.concatenate([[0], np.cumsum(curve)])
    idx = np.arange(n)
    l_start = np.maximum(idx - half_win, 0)
    r_end = np.minimum(idx + half_win, n)
    l_mean = (cs[idx] - cs[l_start]) / np.maximum(idx - l_start, 1)
    r_mean = (cs[r_end] - cs[idx]) / np.maximum(r_end - idx, 1)
    return np.abs(r_mean - l_mean)


# ===== MODE 2: CQT PSD change-point (crossfades) =====

def _detect_crossfade(audio: np.ndarray, sr: int) -> list[float]:
    """
    Detect crossfade splices via Hotelling's T² test on CQT subband PSD vectors.
    Compares the PSD distribution in a sliding left-window vs right-window.
    """
    K = 8  # number of constant-Q bands
    compare_window_s = 5.0  # seconds per comparison window
    hop_s = 0.5  # step between test points

    # --- CQT-like subband decomposition ---
    band_powers = _cqt_band_powers(audio, sr, K, frame_s=0.050, hop_s=0.020)
    # band_powers: (K, n_frames), each frame = 20ms

    frames_per_window = max(1, int(compare_window_s / 0.020))
    hop_frames = max(1, int(hop_s / 0.020))
    n_frames = band_powers.shape[1]

    if n_frames < 2 * frames_per_window + 1:
        return []

    # --- Sliding Hotelling T² test ---
    t2_scores = []
    test_times = []

    for center in range(frames_per_window, n_frames - frames_per_window, hop_frames):
        left = band_powers[:, center - frames_per_window: center].T   # (W, K)
        right = band_powers[:, center: center + frames_per_window].T  # (W, K)

        t2 = _hotelling_t2(left, right)
        t2_scores.append(t2)
        test_times.append(center * 0.020)

    if not t2_scores:
        return []

    t2_arr = np.array(t2_scores)
    times_arr = np.array(test_times)

    # --- Z-score and threshold ---
    t2_z = _zscore(t2_arr)

    # Silence suppression: suppress detections where audio is silent
    silence = _silence_mask(audio, sr, hop_ms=500, threshold_db=-45)
    # Resample silence mask to match t2 positions
    silence_at_test = np.interp(times_arr, np.arange(len(silence)) * 0.5, silence)
    t2_z = t2_z * (silence_at_test > 0.5).astype(float)

    p995 = np.percentile(t2_z, 99.5) if len(t2_z) > 10 else 5.0
    threshold = max(p995, 5.0)

    # Peak pick
    min_dist_idx = max(1, int(5.0 / hop_s))
    peaks_idx, props = sp_signal.find_peaks(t2_z, height=threshold,
                                             distance=min_dist_idx,
                                             prominence=threshold * 0.2)
    if len(peaks_idx) == 0:
        return []

    # Top 2 peaks
    heights = props['peak_heights']
    order = np.argsort(-heights)
    peaks_idx = peaks_idx[order[:1]]

    # Convert to times, filter quiet boundaries, refine
    candidates = [times_arr[idx] for idx in sorted(peaks_idx)]

    # Reject quiet-to-loud boundaries (same filter as phase detector)
    check_samples = int(0.5 * sr)
    energy_threshold_db = -35
    filtered = []
    for t in candidates:
        center_sample = int(t * sr)
        left_start = max(0, center_sample - check_samples)
        right_end = min(len(audio), center_sample + check_samples)
        if center_sample - left_start < sr // 10 or right_end - center_sample < sr // 10:
            continue
        left_rms = np.sqrt(np.mean(audio[left_start:center_sample] ** 2))
        right_rms = np.sqrt(np.mean(audio[center_sample:right_end] ** 2))
        left_db = 20 * np.log10(max(left_rms, 1e-10))
        right_db = 20 * np.log10(max(right_rms, 1e-10))
        if left_db > energy_threshold_db and right_db > energy_threshold_db:
            filtered.append(t)

    results = []
    for t in filtered:
        refined_t = _refine_splice_point(audio, sr, t, search_radius_s=1.0)
        results.append(refined_t)

    return results


def _cqt_band_powers(audio: np.ndarray, sr: int, K: int = 8,
                     frame_s: float = 0.050, hop_s: float = 0.020) -> np.ndarray:
    """
    Constant-Q-like subband power decomposition.
    K bands, log-spaced from ~80Hz to sr/2.
    Returns (K, n_frames) array of band power per frame.
    """
    hop_samples = max(1, int(sr * hop_s))
    frame_samples = max(1, int(sr * frame_s))
    n_frames = (len(audio) - frame_samples) // hop_samples + 1
    if n_frames < 1:
        return np.zeros((K, 1))

    # Define log-spaced band edges
    f_min = 80.0
    f_max = sr / 2.0
    band_edges = np.logspace(np.log10(f_min), np.log10(f_max), K + 1)

    # Compute STFT for frequency analysis
    n_fft = max(256, int(2 ** np.ceil(np.log2(frame_samples))))
    _, _, Zxx = sp_signal.stft(audio, fs=sr, nperseg=n_fft,
                                noverlap=n_fft - hop_samples,
                                boundary=None)
    power = np.abs(Zxx) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Sum power within each band
    band_powers = np.zeros((K, power.shape[1]))
    for k in range(K):
        mask = (freqs >= band_edges[k]) & (freqs < band_edges[k + 1])
        if mask.any():
            band_powers[k] = power[mask].sum(axis=0)

    # Convert to dB
    band_powers = 10 * np.log10(np.maximum(band_powers, 1e-10))

    return band_powers


def _hotelling_t2(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Two-sample Hotelling's T² test statistic.
    X: (n1, K) — samples from left window
    Y: (n2, K) — samples from right window
    Returns T² statistic (higher = more different).
    """
    n1, p = X.shape
    n2 = Y.shape[0]

    mean1 = X.mean(axis=0)
    mean2 = Y.mean(axis=0)
    diff = mean1 - mean2

    # Pooled covariance
    S1 = np.cov(X, rowvar=False, ddof=1) if n1 > 1 else np.eye(p) * 1e-6
    S2 = np.cov(Y, rowvar=False, ddof=1) if n2 > 1 else np.eye(p) * 1e-6
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

    # Regularize for numerical stability
    Sp += np.eye(p) * 1e-6

    # T² = n1*n2/(n1+n2) * diff' * Sp^{-1} * diff
    try:
        Sp_inv_diff = np.linalg.solve(Sp, diff)
        t2 = (n1 * n2 / (n1 + n2)) * np.dot(diff, Sp_inv_diff)
    except np.linalg.LinAlgError:
        t2 = 0.0

    return max(0.0, t2)


def _analyze_segment_phase(audio: np.ndarray, sr: int, offset_s: float = 0.0) -> list[float]:
    hop_ms = 10
    hop_s = hop_ms / 1000.0
    n_fft = 1024
    hop_length = max(1, int(sr * hop_ms / 1000))

    # --- STFT ---
    _, _, Zxx = sp_signal.stft(audio, fs=sr, nperseg=n_fft,
                                noverlap=n_fft - hop_length)

    # --- Primary feature: Phase discontinuity ---
    phase_disc = _phase_discontinuity(Zxx, sr, n_fft, hop_length)
    phase_z = _zscore(phase_disc)

    # --- Secondary: BCR for additional signal ---
    mag = np.abs(Zxx)
    bcr = _broadband_change_ratio(mag)
    bcr_z = _zscore(bcr)

    # --- Align ---
    min_len = min(len(phase_z), len(bcr_z))
    phase_z = phase_z[:min_len]
    bcr_z = bcr_z[:min_len]

    # --- Phase-only scoring ---
    fused = phase_z

    # --- Silence suppression ---
    silence = _silence_mask(audio, sr, hop_ms=hop_ms, threshold_db=-45)[:min_len]
    fused = fused * silence

    # --- GPD tail-based threshold with Bonferroni correction ---
    n_tests = int(np.sum(silence > 0.5))  # only non-silent frames count
    threshold = _gpd_threshold(fused[silence > 0.5], n_tests=n_tests, alpha=0.5)

    peaks = _peak_pick(fused, threshold=threshold, min_dist_s=5.0, hop_s=hop_s)

    # Cap at 1 — only the strongest phase detection per segment
    if len(peaks) > 1:
        peak_frames = [int(p / hop_s) for p in peaks]
        scores = [fused[min(f, min_len - 1)] for f in peak_frames]
        top_idx = np.argsort(scores)[-1:]
        peaks = [peaks[i] for i in sorted(top_idx)]

    # --- Reject detections at quiet-to-loud boundaries ---
    # A real splice joins two voiced segments. If either side is quiet,
    # it's likely a natural transition (breath, swallow, pause start/end).
    filtered_peaks = []
    check_radius_s = 0.5  # check 500ms on each side
    check_samples = int(check_radius_s * sr)
    energy_threshold_db = -35  # both sides must be above this

    for p in peaks:
        center = int(p * sr / 1000 * 1000)  # convert via hop
        center_sample = int(p * sr)
        left_start = max(0, center_sample - check_samples)
        left_end = center_sample
        right_start = center_sample
        right_end = min(len(audio), center_sample + check_samples)

        if left_end - left_start < sr // 10 or right_end - right_start < sr // 10:
            continue

        left_rms = np.sqrt(np.mean(audio[left_start:left_end] ** 2))
        right_rms = np.sqrt(np.mean(audio[right_start:right_end] ** 2))
        left_db = 20 * np.log10(max(left_rms, 1e-10))
        right_db = 20 * np.log10(max(right_rms, 1e-10))

        if left_db > energy_threshold_db and right_db > energy_threshold_db:
            filtered_peaks.append(p)

    # --- Refine with waveform ---
    refined = []
    for p in filtered_peaks:
        r = _refine_splice_point(audio, sr, p, search_radius_s=0.3)
        refined.append(r + offset_s)

    return refined


# ---------------------------------------------------------------------------
# Phase discontinuity (PRIMARY FEATURE)
# ---------------------------------------------------------------------------

def _phase_discontinuity(Zxx: np.ndarray, sr: int, n_fft: int,
                         hop_length: int) -> np.ndarray:
    """
    Measure abrupt changes in STFT phase evolution.

    1. Compute phase difference between consecutive frames
    2. Subtract expected phase advance (based on frequency and hop size)
    3. Square and sum across frequency bins = phase deviation energy
    4. First difference of this energy = spike at discontinuities
    """
    phase = np.angle(Zxx)

    # Phase difference
    dphi = np.diff(phase, axis=1)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi

    # Expected phase advance per bin
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    expected = 2 * np.pi * freqs * (hop_length / sr)
    deviation = dphi - expected[:, np.newaxis]
    deviation = (deviation + np.pi) % (2 * np.pi) - np.pi

    # Phase deviation energy per frame
    phase_energy = np.sum(deviation ** 2, axis=0)

    # First difference: detects abrupt changes
    d_phase = np.abs(np.diff(phase_energy))

    return d_phase


# ---------------------------------------------------------------------------
# BCR (secondary)
# ---------------------------------------------------------------------------

def _broadband_change_ratio(mag: np.ndarray) -> np.ndarray:
    diff = np.abs(mag[:, 1:] - mag[:, :-1])
    frame_median = np.median(diff, axis=0, keepdims=True)
    frame_median = np.maximum(frame_median, 1e-10)
    significant = diff > (frame_median * 4.0)
    return significant.sum(axis=0) / diff.shape[0]


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------

def _refine_splice_point(audio: np.ndarray, sr: int, coarse_time: float,
                         search_radius_s: float = 0.3) -> float:
    center = int(coarse_time * sr)
    radius = int(search_radius_s * sr)
    lo = max(0, center - radius)
    hi = min(len(audio), center + radius)
    if hi - lo < 100:
        return coarse_time

    segment = audio[lo:hi]
    sos = sp_signal.butter(2, 2000, btype='highpass', fs=sr, output='sos')
    hp = sp_signal.sosfilt(sos, segment)
    abs_diff = np.abs(np.diff(hp))
    peak_idx = np.argmax(abs_diff)
    return (lo + peak_idx) / sr


# ---------------------------------------------------------------------------
# GPD tail threshold
# ---------------------------------------------------------------------------

def _gpd_threshold(scores: np.ndarray, n_tests: int, alpha: float = 0.05) -> float:
    """
    Compute detection threshold using Generalized Pareto Distribution.

    1. Fit GPD to the upper tail (top 5%) of the score distribution
    2. Apply Bonferroni correction: per-test alpha = alpha / n_tests
    3. Return the score value where the tail probability = corrected alpha

    This gives an honest threshold that accounts for:
    - The actual heavy-tailed distribution (not Gaussian assumption)
    - The number of test points (longer files → higher threshold)
    """
    if len(scores) < 50:
        return float(np.max(scores) + 1) if len(scores) > 0 else 10.0

    # Use top 5% as the tail
    tail_quantile = 0.95
    u = float(np.percentile(scores, tail_quantile * 100))
    exceedances = scores[scores > u] - u

    if len(exceedances) < 10:
        # Not enough tail data, fall back to empirical quantile
        corrected_q = 1.0 - alpha / max(n_tests, 1)
        return float(np.percentile(scores, corrected_q * 100))

    # Fit GPD to exceedances
    try:
        shape, loc, scale = genpareto.fit(exceedances, floc=0)
    except Exception:
        corrected_q = 1.0 - alpha / max(n_tests, 1)
        return float(np.percentile(scores, corrected_q * 100))

    # Bonferroni-corrected per-test significance
    p_per_test = alpha / max(n_tests, 1)

    # Tail probability: P(X > u) ≈ (1 - tail_quantile)
    p_tail = 1.0 - tail_quantile

    # We want: P(X > threshold) = p_per_test
    # P(X > threshold) = p_tail * P(excess > threshold - u)
    # P(excess > threshold - u) = p_per_test / p_tail
    target_survival = p_per_test / p_tail

    if target_survival >= 1.0:
        # Threshold would be below u, use u as minimum
        return u

    # GPD inverse survival: x = scale/shape * ((survival)^{-shape} - 1) for shape != 0
    try:
        excess_threshold = genpareto.isf(target_survival, shape, loc=0, scale=scale)
        threshold = u + excess_threshold
    except Exception:
        threshold = u + exceedances.max()

    # Safety floor
    return max(float(threshold), u)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _silence_mask(audio: np.ndarray, sr: int, hop_ms: float = 10,
                  threshold_db: float = -45) -> np.ndarray:
    hop_samples = max(1, int(sr * hop_ms / 1000))
    window_samples = max(1, int(sr * 0.050))
    sq = audio ** 2
    smoothed = uniform_filter1d(sq, size=window_samples, mode='constant')
    rms = np.sqrt(smoothed[::hop_samples])
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
    return (rms_db > threshold_db).astype(np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x)
    sd = np.std(x)
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sd


def _peak_pick(curve: np.ndarray, threshold: float, min_dist_s: float,
               hop_s: float) -> list[float]:
    min_dist_frames = max(1, int(min_dist_s / hop_s))
    peaks_idx, props = sp_signal.find_peaks(curve, height=threshold,
                                             distance=min_dist_frames,
                                             prominence=threshold * 0.2)
    if len(peaks_idx) == 0:
        return []
    heights = props['peak_heights']
    order = np.argsort(-heights)
    peaks_idx = peaks_idx[order]
    return [float(idx) * hop_s for idx in sorted(peaks_idx)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_wav(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect splice points in a WAV file.")
    parser.add_argument("input_wav", help="Path to input WAV file")
    args = parser.parse_args()
    audio, sr = _load_wav(args.input_wav)
    splices = detect_splices(audio, sr)
    if splices:
        print(f"Detected {len(splices)} splice(s):")
        for t in splices:
            print(f"  {t:.3f}s")
    else:
        print("No splices detected.")


if __name__ == "__main__":
    main()
