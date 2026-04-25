"""Audio splice detector — multi-class GBM over a 75-dim feature vector.

`detect_splices(audio, sr)` slides a fixed-size analysis window, builds a
per-chunk feature context, and runs `gbm.predict_proba` at every
`ANALYSIS_STRIDE_S` candidate. Emissions above `GBM_THRESHOLD` are
deduped within `GBM_MIN_SEP_S` and returned as file-level splice times.

Feature components (see features.py) still include classical DSP signals
(phase discontinuity fused z-score, Hotelling T² z-score, complex
prediction error, pairwise block-structure proximity), computed from the
per-chunk context — but they feed the classifier, they do not gate the
output on their own. A multi-class bundle at
`splice/classifier/fp_classifier.joblib` is required; `detect_splices`
raises if the bundle is missing.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d
from scipy.stats import genpareto

from autoresearch.logger import get_logger

# Sliding analysis-window geometry.
# All detectors run inside a fixed-size chunk so GPD thresholds, n_tests
# Bonferroni terms, and pairwise N stay bounded regardless of total file
# length. Sweet spot from duration sweep (sweep_duration.png): 30–60s;
# W=60 gives crossfade T² and CPE enough context; S=30 (50% overlap) means
# splices near a chunk boundary are seen by two adjacent chunks.
ANALYSIS_WINDOW_S = 60.0
ANALYSIS_STEP_S = 30.0  # 50% overlap — W-S=30s, every splice within 30s of a boundary sees two chunks


_DETECT_CACHE: dict = {}
_DETECT_CACHE_MAX = 512

ANALYSIS_STRIDE_S = 0.12
# Training class ratio (~45/55 splice/not_splice) is far denser than the
# ~1:300 splice-per-candidate ratio in real audio, so proba skews high.
# An aggressive decision threshold compensates for the prior-probability
# mismatch without retraining.
GBM_THRESHOLD = 0.982
# Must exceed evaluate.py's 1.0s tolerance so one real splice cannot inflate
# into multiple detections when a high-probability plateau spans several
# adjacent candidates.
GBM_MIN_SEP_S = 1.5

# DSP-confirmation floor: drop hit_mask emits whose strongest DSP z-score
# signal (phase_z / T²_z / CPE_z) is below DSP_CONFIRMATION_MIN. GBM's
# softmax can emit on an overwhelming spec_*_delta signal alone (chord
# transition → big rolloff/centroid/bandwidth delta but smooth phase,
# modest T², low CPE). Real cross-source splices produce at least one
# DSP spike. Applied BEFORE dedupe so a strong-DSP neighbor can still
# win a cluster.
DSP_CONFIRMATION_CHANNELS = ("dsp_phase_z", "dsp_t2_z", "dsp_cpe_z")
DSP_CONFIRMATION_MIN = 2.0
# SUM-based companion floor stacked on top of the MAX gate. Real cross-source
# splices disrupt multiple physical signals simultaneously (mic/room mismatch
# fires phase AND T² AND CPE), so the cumulative DSP magnitude is high (sum
# 6-12). Single-channel firings — chord transitions firing only T² with smooth
# phase / low CPE — sum to ~3-5. Threshold 5.0 demands 3.0 of cumulative
# support beyond the MAX floor of 2.0, biting the borderline-FP band [4.5, 5.5]
# (chord transitions with one strong channel + partial support, speech phoneme
# shifts with T²≈2.5 + CPE≈1.0) while preserving multi-channel-confirmed real
# splices (sum 6-12).
DSP_SUM_MIN = 5.0

_GBM_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "classifier", "fp_classifier.joblib",
)

_GBM_BUNDLE: dict | None = None
_GBM_BUNDLE_UNAVAILABLE = False

# Per-file SHAP-sidecar inputs, keyed by _cache_key.
_DETECT_META: dict = {}


def _cache_key(audio: np.ndarray, sr: int):
    """Audio fingerprint + classifier mtime. The mtime clause makes retrains
    invalidate the cache automatically.
    """
    b = audio.tobytes()
    head = b[:4096]
    tail = b[-4096:] if len(b) > 4096 else b""
    try:
        model_mtime = int(os.path.getmtime(_GBM_MODEL_PATH) * 1000)
    except OSError:
        model_mtime = 0
    return (hash(head), hash(tail), int(sr), len(audio), model_mtime)


def _chunk_starts(n: int, sr: int) -> list[int]:
    """Sliding-window start samples. Always includes a tail chunk that
    reaches the end of the audio when the file doesn't divide evenly.
    """
    W = int(ANALYSIS_WINDOW_S * sr)
    S = int(ANALYSIS_STEP_S * sr)
    if n <= W:
        return [0]
    starts = list(range(0, n - W + 1, S))
    if starts[-1] + W < n:
        starts.append(n - W)
    return starts


def _load_gbm_bundle() -> dict | None:
    """Load the multi-class GBM bundle once per process.

    Accepts either the current on-disk format — a raw sklearn Pipeline
    alongside fp_classifier.meta.json — or the historical dict bundle.
    Binary / legacy classifiers (classes != multi-class) are rejected so
    the detector cleanly falls back to the DSP path.
    """
    global _GBM_BUNDLE, _GBM_BUNDLE_UNAVAILABLE
    if _GBM_BUNDLE is not None:
        return _GBM_BUNDLE
    if _GBM_BUNDLE_UNAVAILABLE:
        return None
    if not os.path.exists(_GBM_MODEL_PATH):
        get_logger("detector.gbm").emit("INFO", "diag.gbm.model_missing", path=_GBM_MODEL_PATH)
        _GBM_BUNDLE_UNAVAILABLE = True
        return None

    import json as _json
    try:
        import joblib
        model = joblib.load(_GBM_MODEL_PATH)
    except Exception as e:
        get_logger("detector.gbm").emit("WARN", "diag.gbm.load_failed",
              path=_GBM_MODEL_PATH, error=str(e))
        _GBM_BUNDLE_UNAVAILABLE = True
        return None

    if isinstance(model, dict):
        model = model.get("model")
        if model is None:
            get_logger("detector.gbm").emit("INFO", "diag.gbm.legacy_dict_missing_model",
                  path=_GBM_MODEL_PATH)
            _GBM_BUNDLE_UNAVAILABLE = True
            return None

    meta_path = os.path.splitext(_GBM_MODEL_PATH)[0] + ".meta.json"
    meta: dict = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = _json.load(f)
        except Exception as e:
            get_logger("detector.gbm").emit("WARN", "diag.gbm.meta_load_failed",
                  path=meta_path, error=str(e))

    try:
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        classes = [int(c) for c in clf.classes_]
    except Exception:
        classes = []
    if not classes or 0 not in classes or not any(c in classes for c in (1, 2)):
        get_logger("detector.gbm").emit("INFO", "diag.gbm.legacy_bundle_ignored",
              path=_GBM_MODEL_PATH, classes=str(classes),
              hint="retrain via .omc/classifier/train_classifier.py for 3-class")
        _GBM_BUNDLE_UNAVAILABLE = True
        return None

    _GBM_BUNDLE = {
        "model": model,
        "feature_names": meta.get("feature_names"),
        "label_names": meta.get("label_names",
                                ["not_splice", "hard_cut", "crossfade"]),
        "classes_": meta.get("classes_", classes),
    }
    return _GBM_BUNDLE


def detect_splices(audio: np.ndarray, sr: int) -> list[tuple[float, str]]:
    """Length-agnostic splice detector.

    Dense GBM scan at ANALYSIS_STRIDE_S. Requires a multi-class classifier
    bundle at .omc/classifier/fp_classifier.joblib; raises if missing.
    Results are memoized by audio fingerprint.

    Returns list of (time_s: float, label: str) tuples.
    label ∈ {"cross_voice", "same_voice_edit", "unknown"}.
    Default label is "unknown" — emit a specific label only when the detector
    has class-specific evidence.
    """
    key = _cache_key(audio, sr)
    if key in _DETECT_CACHE:
        return list(_DETECT_CACHE[key])

    bundle = _load_gbm_bundle()
    if bundle is None:
        raise RuntimeError(
            "Multi-class GBM bundle missing or invalid at "
            f"{_GBM_MODEL_PATH}. Retrain via "
            "`uv run python .omc/classifier/train_classifier.py`."
        )

    merged = _gbm_detect_splices(audio, sr, bundle, key=key)
    if len(_DETECT_CACHE) < _DETECT_CACHE_MAX:
        _DETECT_CACHE[key] = list(merged)
    return merged  # already list[tuple[float, str]]


def _iter_chunks(audio: np.ndarray, sr: int):
    """Yield (start_samples, offset_s, chunk) for each sliding window; skips
    chunks shorter than 5s with a DIAG INFO.
    """
    min_chunk = int(5 * sr)
    W = int(ANALYSIS_WINDOW_S * sr)
    for start in _chunk_starts(len(audio), sr):
        chunk = audio[start:start + W]
        if len(chunk) < min_chunk:
            get_logger("detector.slider").emit("INFO", "diag.slider.chunk_too_small",
                  start=f"{start/sr:.1f}", samples=len(chunk), min=min_chunk)
            continue
        yield start, start / sr, chunk


def _gbm_detect_splices(
    audio: np.ndarray, sr: int, bundle: dict, key: tuple,
) -> list[tuple[float, str]]:
    """Dense GBM scan at ANALYSIS_STRIDE_S; see `detect_splices`."""
    import time as _time
    from features import FEATURE_NAMES as _FN, extract_features

    model = bundle["model"]
    feature_names = bundle.get("feature_names") or _FN
    label_names = bundle.get("label_names") or ["not_splice", "hard_cut", "crossfade"]

    def _label(i: int) -> str:
        return label_names[i] if 0 <= i < len(label_names) else f"class_{i}"

    scan_total = 0
    all_emits: list[tuple[float, int, float, list[float]]] = []
    n_chunks = 0
    t_ctx_sum = 0.0
    t_feat_sum = 0.0
    t_pred_sum = 0.0
    gbm_emit_total = 0
    dsp_dropped_total = 0

    dsp_confirm_idx: list[int] = [
        feature_names.index(n) for n in DSP_CONFIRMATION_CHANNELS
        if n in feature_names
    ]

    file_t0 = _time.perf_counter()

    for _start, offset_s, chunk in _iter_chunks(audio, sr):
        n_chunks += 1
        t0 = _time.perf_counter()
        try:
            ctx = _build_chunk_context(chunk, sr)
        except Exception as e:
            get_logger("detector.gbm").emit("WARN", "diag.gbm.ctx_build_failed",
                  start=f"{offset_s:.2f}", error=str(e))
            continue
        t_ctx = _time.perf_counter() - t0
        t_ctx_sum += t_ctx

        chunk_dur_s = len(chunk) / sr
        t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S, dtype=np.float64)
        if len(t_grid) == 0:
            continue

        t0 = _time.perf_counter()
        rows: list[list[float]] = []
        for t_local in t_grid:
            feats = extract_features(chunk, sr, float(t_local), chunk_ctx=ctx)
            rows.append([feats[k] for k in feature_names])
        X = np.asarray(rows, dtype=np.float64)
        t_feat = _time.perf_counter() - t0
        t_feat_sum += t_feat

        t0 = _time.perf_counter()
        try:
            proba = model.predict_proba(X)
        except Exception as e:
            get_logger("detector.gbm").emit("WARN", "diag.gbm.predict_failed",
                  start=f"{offset_s:.2f}", error=str(e))
            continue
        t_pred = _time.perf_counter() - t0
        t_pred_sum += t_pred

        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        col_for = {int(c): i for i, c in enumerate(clf.classes_)}
        splice_cols = [(c, col_for[c]) for c in (1, 2) if c in col_for]
        if 0 not in col_for or not splice_cols:
            get_logger("detector.gbm").emit("WARN", "diag.gbm.unexpected_classes",
                  classes=sorted(col_for), start=f"{offset_s:.2f}")
            continue
        p_splice = 1.0 - proba[:, col_for[0]]
        scan_total += len(t_grid)

        hit_mask = p_splice > GBM_THRESHOLD
        gbm_emit_count = int(hit_mask.sum())
        gbm_emit_total += gbm_emit_count

        chunk_dsp_dropped = 0
        for i in np.flatnonzero(hit_mask):
            if dsp_confirm_idx:
                row = X[i]
                dsp_vals = [row[j] for j in dsp_confirm_idx]
                if max(dsp_vals) < DSP_CONFIRMATION_MIN or sum(dsp_vals) < DSP_SUM_MIN:
                    chunk_dsp_dropped += 1
                    continue
            t_local = float(t_grid[i])
            label_id, _col = max(splice_cols, key=lambda sc: proba[i, sc[1]])
            all_emits.append((
                float(offset_s + t_local),
                int(label_id),
                float(p_splice[i]),
                X[i].tolist(),
            ))
        dsp_dropped_total += chunk_dsp_dropped

        get_logger("detector.gbm").emit("INFO", "diag.gbm.chunk_scan_done",
              start=f"{offset_s:.2f}",
              scan_count=len(t_grid),
              gbm_emit=gbm_emit_count,
              dsp_dropped=chunk_dsp_dropped,
              t_ctx_ms=int(t_ctx * 1000),
              t_feat_ms=int(t_feat * 1000),
              t_pred_ms=int(t_pred * 1000))

    get_logger("detector.gbm").emit("INFO", "diag.gbm.file_summary",
          chunks=n_chunks,
          scan_total=scan_total,
          gbm_emit_total=gbm_emit_total,
          dsp_dropped_total=dsp_dropped_total,
          t_ctx_total_ms=int(t_ctx_sum * 1000),
          t_feat_total_ms=int(t_feat_sum * 1000),
          t_pred_total_ms=int(t_pred_sum * 1000),
          t_wall_ms=int((_time.perf_counter() - file_t0) * 1000))

    get_logger("detector.gbm").emit("INFO", "diag.gbm.scan_summary",
          scan_total=scan_total, emit_total=len(all_emits), chunks=n_chunks)

    # Greedy dedupe: pick highest-probability emissions first, suppress any
    # other emission within GBM_MIN_SEP_S.
    all_emits.sort(key=lambda e: -e[2])
    selected: list[tuple[float, int, float, list[float]]] = []
    for emit in all_emits:
        if all(abs(emit[0] - s[0]) >= GBM_MIN_SEP_S for s in selected):
            selected.append(emit)
    selected.sort(key=lambda e: e[0])

    _DETECT_META[key] = [
        {
            "time_sec": t,
            "label_id": label_id,
            "label": _label(label_id),
            "probability": p,
            "feature_vector": feats_vec,
            "feature_names": list(feature_names),
        }
        for (t, label_id, p, feats_vec) in selected
    ]

    return [(t, "unknown") for (t, _, _, _) in selected]


def get_detection_meta(audio: np.ndarray, sr: int) -> list[dict]:
    """Per-detection SHAP-sidecar payload from the most recent GBM scan of
    this audio. Empty list if the audio has not been scanned yet or
    produced no emissions above GBM_THRESHOLD.
    """
    return list(_DETECT_META.get(_cache_key(audio, sr), []))


# ===== MODE 5: Pairwise Segment Distance Matrix =====

def _detect_pairwise(audio: np.ndarray, sr: int,
                     segment_s: float = 5.0,
                     K: int = 8) -> tuple[float, float | None]:
    """
    Pairwise Segment Distance Matrix splice detector.

    Divides audio into N segments, computes Hotelling T² between ALL pairs,
    then detects block structure in the N×N distance matrix.
    A splice creates two blocks of similar segments with high cross-block distance.

    Returns (best_score, splice_time) or (0.0, None) if no block structure found.
    """
    duration_s = len(audio) / sr
    N = int(duration_s / segment_s)
    if N < 4:
        get_logger("detector.pairwise").emit("INFO", "diag.pairwise.audio_too_short",
              duration_s=f"{duration_s:.2f}", n_segments=N, required=4)
        return 0.0, None
    # O(N²) duration gate removed: the sliding orchestrator in detect_splices
    # bounds chunk size to ANALYSIS_WINDOW_S (~60s) → N ≈ 12, always tractable.

    # Compute CQT band powers for each segment. A segment that can't be
    # analyzed (too short) is skipped with an explicit DIAG and excluded
    # from the pairwise matrix by collapsing it to zero variance.
    segment_samples = int(segment_s * sr)
    segment_features = []
    for i in range(N):
        start = i * segment_samples
        end = start + segment_samples
        seg = audio[start:end]
        try:
            bp = _cqt_band_powers(seg, sr, K, frame_s=0.050, hop_s=0.020)
        except _FitError as e:
            get_logger("detector.pairwise").emit("WARN", "diag.pairwise.cqt_skipped", segment=i,
                  cause=e.reason, **e.context)
            bp = np.zeros((K, 2))  # 2 frames so _hotelling_t2 n_min==2 holds
        segment_features.append(bp.T)  # (n_frames, K)

    # Compute N×N distance matrix using Hotelling T². Individual pair
    # failures become 0.0 (no evidence of difference) with a DIAG line so
    # the silence isn't hidden.
    dist_matrix = np.zeros((N, N))
    n_failed = 0
    for i in range(N):
        for j in range(i + 1, N):
            try:
                t2 = _hotelling_t2(segment_features[i], segment_features[j])
            except _FitError as e:
                get_logger("detector.pairwise").emit("INFO", "diag.pairwise.t2_failed", i=i, j=j,
                      cause=e.reason, **e.context)
                t2 = 0.0
                n_failed += 1
            dist_matrix[i, j] = t2
            dist_matrix[j, i] = t2
    total_pairs = N * (N - 1) // 2
    if n_failed and n_failed / total_pairs > 0.1:
        get_logger("detector.pairwise").emit("WARN", "diag.pairwise.t2_mass_failure",
              n_failed=n_failed, n_total=total_pairs)

    # Find best split point via block structure score
    best_score = -1.0
    best_k = -1
    for k in range(2, N - 1):
        block_a = dist_matrix[:k, :k]
        block_b = dist_matrix[k:, k:]
        within_a = block_a[np.triu_indices(k, k=1)]
        within_b = block_b[np.triu_indices(N - k, k=1)]

        if len(within_a) == 0 or len(within_b) == 0:
            continue

        within_mean = (within_a.mean() + within_b.mean()) / 2.0
        if within_mean < 1e-6:
            continue

        between = dist_matrix[:k, k:]
        between_mean = between.mean()

        score = between_mean / within_mean
        if score > best_score:
            best_score = score
            best_k = k

    if best_k < 0 or best_score < 1.5:
        return 0.0, None

    # Convert split point to time
    splice_time = best_k * segment_s

    # Silence/energy check
    check_samples = int(2.0 * sr)
    center_sample = int(splice_time * sr)
    left_start = max(0, center_sample - check_samples)
    right_end = min(len(audio), center_sample + check_samples)
    if center_sample - left_start < sr // 10 or right_end - center_sample < sr // 10:
        return best_score, None
    left_rms = np.sqrt(np.mean(audio[left_start:center_sample] ** 2))
    right_rms = np.sqrt(np.mean(audio[center_sample:right_end] ** 2))
    left_db = 20 * np.log10(max(left_rms, 1e-10))
    right_db = 20 * np.log10(max(right_rms, 1e-10))
    if left_db < -35 or right_db < -35:
        return best_score, None

    # Refine the splice point
    refined = _refine_splice_point(audio, sr, splice_time, search_radius_s=segment_s / 2)
    return best_score, refined


# ===== MODE 1: Phase discontinuity (hard cuts) =====
# `_analyze_segment_phase` is called directly by `_detect_in_chunk`; the
# previous `_detect_phase` multi-segment wrapper is gone — file-level
# chunking lives in `detect_splices`.


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
        raise _FitError("audio_too_short_for_stft",
                        n_samples=len(audio), frame_samples=frame_samples)

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

    Raises `_FitError` when the fit is ill-defined (single-sample windows,
    singular pooled covariance). Callers decide how to handle failures —
    no silent substitution inside this function.
    """
    n1, p = X.shape
    n2 = Y.shape[0]

    if n1 < 2 or n2 < 2:
        raise _FitError("degenerate_sample", n1=n1, n2=n2)

    mean1 = X.mean(axis=0)
    mean2 = Y.mean(axis=0)
    diff = mean1 - mean2

    S1 = np.cov(X, rowvar=False, ddof=1)
    S2 = np.cov(Y, rowvar=False, ddof=1)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

    # Ridge regularization for numerical stability (visible via DIAG if heavy).
    ridge = 1e-6
    Sp = Sp + np.eye(p) * ridge
    # Near-singular pooled covariance is a warning, not a failure.
    cond = float(np.linalg.cond(Sp))
    if cond > 1e8:
        get_logger("detector.hotelling").emit("INFO", "diag.hotelling.near_singular_cov",
              cond=f"{cond:.1e}", ridge=ridge, p=p)

    try:
        Sp_inv_diff = np.linalg.solve(Sp, diff)
    except np.linalg.LinAlgError as err:
        raise _FitError("singular_cov", p=p, cond=f"{cond:.1e}") from err
    t2 = (n1 * n2 / (n1 + n2)) * float(np.dot(diff, Sp_inv_diff))
    return max(0.0, t2)


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
        get_logger("detector.refine").emit("INFO", "diag.refine.window_too_small",
              at_sec=f"{coarse_time:.3f}", window_samples=hi - lo)
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



class _FitError(Exception):
    """Raised when a statistical fit cannot be computed reliably.

    Carries `.reason` (short code) and `.context` (dict) so the caller can
    forward both into a DIAG line without re-parsing. Never swallowed
    silently — the only two sanctioned responses are (a) propagate, or
    (b) catch, emit DIAG, and invoke an explicitly named fallback.
    """
    def __init__(self, reason: str, **context):
        self.reason = reason
        self.context = context
        parts = [reason] + [f"{k}={v}" for k, v in context.items()]
        super().__init__(" ".join(parts))


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
    """Z-score normalization. Emits DIAG WARN for degenerate variance so the
    constant-signal case is visible rather than silently producing zeros.
    """
    mu = np.mean(x)
    sd = np.std(x)
    if sd < 1e-8:
        get_logger("detector.zscore").emit("WARN", "diag.zscore.degenerate_variance", n=len(x), std=f"{sd:.2e}")
        return np.zeros_like(x)
    return (x - mu) / sd


def _robust_zscore(x: np.ndarray) -> np.ndarray:
    """MAD-based robust z-score. Emits DIAG WARN for degenerate MAD so the
    constant-signal case is visible rather than silently producing zeros.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-8:
        get_logger("detector.robust_zscore").emit("WARN", "diag.robust_zscore.degenerate_mad", n=len(x), mad=f"{mad:.2e}")
        return np.zeros_like(x)
    return (x - med) / (mad * 1.4826)  # 1.4826 scales MAD to std for normal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ===========================================================================
# DSP single-position accessors (US-300)
#
# Each existing _detect_{phase, cpe, crossfade} function computes a score
# curve, applies silence masking, GPD-thresholds, peak-picks, and refines.
# For GBM-as-main, we want the CURVE part only — sampled at an arbitrary
# time t — so DSP scores can feed the GBM feature vector without re-running
# the whole detector pipeline. These helpers produce the pre-thresholding
# fused curves and cache them inside a chunk context dict.
#
# The existing detection pipeline is UNCHANGED; these are additive.
# ===========================================================================

def _compute_phase_fused_curve(chunk: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Return (fused_phase_z_curve, hop_s). Mirrors the curve-building portion
    of _analyze_segment_phase, up to the silence-masked fused curve — no peak
    picking / quiet-boundary filter / refine."""
    hop_ms = 10
    hop_s = hop_ms / 1000.0
    n_fft = 1024
    hop_length = max(1, int(sr * hop_ms / 1000))

    _, _, Zxx = sp_signal.stft(chunk, fs=sr, nperseg=n_fft,
                                noverlap=n_fft - hop_length)
    phase_disc = _phase_discontinuity(Zxx, sr, n_fft, hop_length)
    phase_z = _zscore(phase_disc)

    mag = np.abs(Zxx)
    bcr = _broadband_change_ratio(mag)
    bcr_z = _zscore(bcr)

    min_len = min(len(phase_z), len(bcr_z))
    phase_z = phase_z[:min_len]
    bcr_z = bcr_z[:min_len]
    fused = phase_z * (1.0 + np.clip(bcr_z, 0, None))

    silence = _silence_mask(chunk, sr, hop_ms=hop_ms, threshold_db=-45)[:min_len]
    fused = fused * silence
    return fused, hop_s


def _compute_cpe_fused_curve(chunk: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Return (fused_cpe_z_curve, hop_s). Mirrors _detect_cpe up to the
    silence-masked fused curve."""
    hop_ms = 10
    hop_s = hop_ms / 1000.0
    n_fft = 1024
    hop_length = max(1, int(sr * hop_ms / 1000))

    agc_window = max(1, int(sr * 0.300))
    local_power = uniform_filter1d(chunk ** 2, size=agc_window, mode='constant')
    local_rms = np.sqrt(np.maximum(local_power, 1e-10))
    audio_agc = chunk / local_rms

    _, _, Zxx = sp_signal.stft(audio_agc, fs=sr, nperseg=n_fft,
                                noverlap=n_fft - hop_length)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    phase_advance = np.exp(1j * 2 * np.pi * freqs * (hop_length / sr))[:, np.newaxis]
    predicted = Zxx[:, :-1] * phase_advance
    error = Zxx[:, 1:] - predicted
    cpe_score = np.sum(np.abs(error) ** 2, axis=0)
    cpe_z = _zscore(cpe_score)

    silence = _silence_mask(chunk, sr, hop_ms=hop_ms, threshold_db=-45)
    min_len = min(len(cpe_z), len(silence))
    fused = cpe_z[:min_len] * silence[:min_len]
    return fused, hop_s


def _compute_t2_z_curve(chunk: np.ndarray, sr: int,
                        compare_window_s: float = 5.0,
                        hop_s: float = 0.5,
                        K: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Return (t2_z_curve, test_times_sec). Mirrors _detect_crossfade up to the
    silence-masked t2_z curve. Returns (empty, empty) arrays if the chunk is
    too short to fit the compare window."""
    try:
        band_powers = _cqt_band_powers(chunk, sr, K, frame_s=0.050, hop_s=0.020)
    except _FitError:
        return np.array([]), np.array([])

    flux = np.sqrt(np.sum(np.diff(band_powers, axis=1) ** 2, axis=0))
    flux = np.concatenate([[flux[0]], flux])
    flux_smooth = uniform_filter1d(flux, size=5, mode='nearest')
    band_powers = np.vstack([band_powers, flux_smooth[np.newaxis, :]])

    frames_per_window = max(1, int(compare_window_s / 0.020))
    hop_frames = max(1, int(hop_s / 0.020))
    n_frames = band_powers.shape[1]
    if n_frames < 2 * frames_per_window + 1:
        return np.array([]), np.array([])

    t2_scores = []
    test_times = []
    for center in range(frames_per_window, n_frames - frames_per_window, hop_frames):
        left = band_powers[:, center - frames_per_window: center].T
        right = band_powers[:, center: center + frames_per_window].T
        try:
            t2 = _hotelling_t2(left, right)
        except _FitError:
            t2 = 0.0
        t2_scores.append(t2)
        test_times.append(center * 0.020)

    if not t2_scores:
        return np.array([]), np.array([])

    t2_arr = np.array(t2_scores)
    times_arr = np.array(test_times)
    t2_z = _robust_zscore(t2_arr)

    silence = _silence_mask(chunk, sr, hop_ms=100, threshold_db=-45)
    silence_at_test = np.interp(times_arr, np.arange(len(silence)) * 0.1, silence)
    t2_z = t2_z * (silence_at_test > 0.5).astype(float)
    return t2_z, times_arr


def _build_chunk_context(chunk: np.ndarray, sr: int) -> dict:
    """Precompute per-chunk feature inputs once per chunk: fused phase /
    CPE / T² z-score curves plus a single pairwise-block-structure summary.
    The returned dict feeds both the single-position DSP accessors
    (phase_z_at / t2_z_at / cpe_z_at / pairwise_proximity_at) and
    features.py via its own cache layer.
    """
    phase_curve, phase_hop_s = _compute_phase_fused_curve(chunk, sr)
    cpe_curve, cpe_hop_s = _compute_cpe_fused_curve(chunk, sr)
    t2_curve, t2_times = _compute_t2_z_curve(chunk, sr)

    try:
        pw_score, pw_time = _detect_pairwise(chunk, sr)
    except Exception:
        pw_score, pw_time = 0.0, None

    return {
        "sr": sr,
        "duration_s": len(chunk) / sr,
        "phase_curve": phase_curve,
        "phase_hop_s": phase_hop_s,
        "cpe_curve": cpe_curve,
        "cpe_hop_s": cpe_hop_s,
        "t2_curve": t2_curve,
        "t2_times": t2_times,
        "pw_score": float(pw_score),
        "pw_time": None if pw_time is None else float(pw_time),
    }


def _sample_uniform_curve(curve: np.ndarray, hop_s: float, t_sec: float) -> float:
    """Sample a uniform-hop curve at t_sec. Clamps to curve bounds."""
    if len(curve) == 0:
        return 0.0
    frame = int(round(t_sec / hop_s))
    frame = max(0, min(frame, len(curve) - 1))
    return float(curve[frame])


def phase_z_at(ctx: dict, t_sec: float) -> float:
    """DSP phase-discontinuity fused z-score at chunk-local time t_sec."""
    return _sample_uniform_curve(ctx["phase_curve"], ctx["phase_hop_s"], t_sec)


def cpe_z_at(ctx: dict, t_sec: float) -> float:
    """DSP complex-prediction-error fused z-score at chunk-local time t_sec."""
    return _sample_uniform_curve(ctx["cpe_curve"], ctx["cpe_hop_s"], t_sec)


def t2_z_at(ctx: dict, t_sec: float) -> float:
    """DSP crossfade Hotelling T² z-score at the t2 test point nearest t_sec.

    The t2 grid is sparse (0.5s hop) so we find the closest test point.
    Returns 0.0 if chunk was too short for T² tests.
    """
    times = ctx["t2_times"]
    curve = ctx["t2_curve"]
    if len(curve) == 0:
        return 0.0
    idx = int(np.argmin(np.abs(times - t_sec)))
    return float(curve[idx])


def pairwise_proximity_at(ctx: dict, t_sec: float) -> float:
    """How close t_sec is to the chunk's pairwise-block-split time. Returns
    0.0 when no pairwise split was found; otherwise pw_score scaled by a
    Gaussian centered on the split time (sigma=2.5s).
    """
    pw_time = ctx.get("pw_time")
    pw_score = ctx.get("pw_score", 0.0)
    if pw_time is None or pw_score == 0.0:
        return 0.0
    sigma_s = 2.5
    return float(pw_score * np.exp(-0.5 * ((t_sec - pw_time) / sigma_s) ** 2))


