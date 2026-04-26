"""Feature extraction module for GBM-based splice classification.

Extracts ~75 named features from a candidate splice position t_sec within
an audio chunk. All features are deterministic (no randomness in inference),
explainable (statistical / spectral basis), and computed with scipy/librosa/numpy
only -- no torch, tensorflow, or GPU.

Entry point:
    extract_features(audio, sr, t_sec, chunk_ctx=None) -> dict[str, float]

Also exports FEATURE_NAMES: list[str] in the exact order the dict is populated,
for SHAP labeling.

Feature blocks (77 dims):
    Block 1: DSP local @ t             (4)
    Block 2: MFCC-13 deltas            (13)
    Block 3: Spectral summary deltas    (6)
    Block 4: Noise-floor color          (6)
    Block 5: Pitch / voicing            (8)
    Block 6: Energy envelope / ZCR      (6)
    Block 7: Boundary region +/-200ms   (3)
    Block 8: ENF                        (5)
    Block 9: Codec artifact             (4)
    Block 10: Mel-PCA tail              (20)
    Block 11: Voiced-MFCC cosine dist   (1)
    Block 12: Voiced-chroma cosine dist (1)
    Block 13: Voiced-spec-contrast cos  (1)
    Block 14: Voiced/unvoiced MFCC asym (1)

Total: 4+13+6+6+8+6+3+5+4+20+1+1+1+1 = 79

Performance model:
    _ensure_feat_cache() runs ONCE per chunk, precomputing all expensive
    operations (pyin, ENF filter, mel-PCA, frame-level FFTs) and storing
    the results in chunk_ctx under 'feat_*' keys. Per-t calls then perform
    only cheap array slicing and nanmean operations.
    1000 calls on a cached ctx complete in well under 2s on a laptop CPU.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from scipy import signal as sp_signal

from splice.detector import (
    _build_chunk_context,
    _FitError,
    phase_z_at,
    t2_z_at,
    cpe_z_at,
    pairwise_proximity_at,
)

from autoresearch.logger import get_logger


# US-504: on-disk feature cache. Gated by OMC_FEATURE_CACHE_DIR +
# OMC_FEATURES_PY_SHA (set by run_autoresearch.sh only; stripped from
# the claude subprocess via `env -u`). Invalidation is by directory:
# each features.py sha lives in its own subdir, so any edit to
# features.py ignores every old cache automatically.

def _feature_cache_paths(audio: np.ndarray, sr: int) -> Optional[Path]:
    cache_dir = os.environ.get("OMC_FEATURE_CACHE_DIR")
    feat_sha = os.environ.get("OMC_FEATURES_PY_SHA")
    if not cache_dir or not feat_sha:
        return None
    h = hashlib.sha256()
    h.update(audio.tobytes())
    h.update(np.int64(sr).tobytes())
    return Path(cache_dir) / feat_sha / f"{h.hexdigest()[:16]}.pkl"


def _try_load_feat_cache(audio: np.ndarray, sr: int, ctx: dict) -> bool:
    path = _feature_cache_paths(audio, sr)
    if path is None or not path.is_file():
        return False
    try:
        with open(path, "rb") as f:
            cached = pickle.load(f)
    except Exception:
        return False
    if cached.get("feat_sr") != sr:
        return False
    cached_audio = cached.get("feat_audio")
    if not isinstance(cached_audio, np.ndarray) or cached_audio.shape != audio.shape:
        return False
    ctx.update(cached)
    return True


def _save_feat_cache(audio: np.ndarray, sr: int, ctx: dict) -> None:
    path = _feature_cache_paths(audio, sr)
    if path is None:
        return
    to_cache = {k: v for k, v in ctx.items() if k.startswith("feat_")}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Module-level PCA fit (lazy; re-used across calls in one process)
# ---------------------------------------------------------------------------

_PCA_FIT: Optional[object] = None

_MEL_PCA_PATH = ".omc/classifier/mel_pca_20.joblib"
_MEL_N_COMPONENTS = 20
_MEL_PATCH_LEN = 25600  # 128 mels * 200 frames

# ---------------------------------------------------------------------------
# FEATURE_NAMES -- built once at import time
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = (
    ["dsp_phase_z", "dsp_t2_z", "dsp_cpe_z", "dsp_pairwise_proximity"]
    + [f"mfcc_delta_{i:02d}" for i in range(1, 14)]
    + ["spec_centroid_delta", "spec_rolloff_delta", "spec_flux_delta",
       "spec_contrast_delta", "spec_flatness_delta", "spec_bandwidth_delta"]
    + ["nf_kl_divergence", "nf_centroid_delta", "nf_rolloff_delta",
       "nf_flatness_delta", "nf_bottom10_e_pre", "nf_bottom10_e_post"]
    + ["f0_mean_pre", "f0_mean_post", "f0_delta", "f0_jitter_delta",
       "voicing_prob_pre", "voicing_prob_post", "voicing_prob_delta",
       "f0_continuity_score"]
    + ["rms_db_pre", "rms_db_post", "rms_db_delta",
       "zcr_pre", "zcr_post", "zcr_delta"]
    + ["boundary_spec_flux_peak", "boundary_energy_ratio", "boundary_phase_coherence"]
    + ["enf_freq_mean_pre", "enf_freq_mean_post", "enf_freq_delta",
       "enf_phase_continuity", "enf_snr_db"]
    + ["codec_frame_align_offset", "codec_quant_residual_pre",
       "codec_quant_residual_post", "codec_double_compression_score"]
    + [f"mel_pca_{i:02d}" for i in range(1, 21)]
    + ["voiced_mfcc_cosine_dist"]
    + ["voiced_chroma_cosine_dist"]
    + ["voiced_spec_contrast_cosine_dist"]
    + ["voiced_unvoiced_mfcc_asymmetry"]
    + ["voiced_unvoiced_spec_contrast_asymmetry"]
    + ["stationarity_centroid_cv_1s"]
)

assert len(FEATURE_NAMES) == 81, f"Expected 81, got {len(FEATURE_NAMES)}"

# Shared hop/fft constants
_HOP = 512
_N_FFT = 2048
_N_MFCC = 13
_NF_FRAME_LEN = 2048  # 50ms @ 44100 ~ 2205; we use 2048 for FFT alignment
_NF_HOP_LEN = 1024    # ~25ms hop
_NF_N_BINS = 32


# ---------------------------------------------------------------------------
# Chunk-level precomputation (cached in ctx under 'feat_*' keys)
# ---------------------------------------------------------------------------

def _ensure_feat_cache(audio: np.ndarray, sr: int, ctx: dict) -> None:
    """Populate ctx with expensive precomputed arrays if not already done.

    All heavy computation (pyin, ENF filter, mel-PCA, frame FFTs for noise
    floor and codec, STFT for boundary phase) happens here exactly once.
    Per-t calls perform only cheap slicing on the cached arrays.

    US-504: on-disk cache, env-gated. Pass-through if cache env unset.
    """
    if "feat_audio" in ctx:
        return
    if _try_load_feat_cache(audio, sr, ctx):
        return

    audio_f32 = audio.astype(np.float32)
    audio_f64 = audio.astype(np.float64)
    n = len(audio)

    # ---- Frame-level spectral features ----
    mfcc = librosa.feature.mfcc(y=audio_f32, sr=sr, n_mfcc=_N_MFCC,
                                  hop_length=_HOP, n_fft=_N_FFT)
    onset = librosa.onset.onset_strength(y=audio_f32, sr=sr,
                                          hop_length=_HOP, n_fft=_N_FFT)
    centroid = librosa.feature.spectral_centroid(y=audio_f32, sr=sr,
                                                  hop_length=_HOP, n_fft=_N_FFT)
    rolloff = librosa.feature.spectral_rolloff(y=audio_f32, sr=sr,
                                                hop_length=_HOP, n_fft=_N_FFT)
    flatness = librosa.feature.spectral_flatness(y=audio_f32,
                                                  hop_length=_HOP, n_fft=_N_FFT)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio_f32, sr=sr,
                                                    hop_length=_HOP, n_fft=_N_FFT)
    contrast = librosa.feature.spectral_contrast(y=audio_f32, sr=sr,
                                                  hop_length=_HOP, n_fft=_N_FFT)
    zcr = librosa.feature.zero_crossing_rate(y=audio_f32, hop_length=_HOP)
    rms = librosa.feature.rms(y=audio_f32, hop_length=_HOP)
    chroma = librosa.feature.chroma_stft(y=audio_f32, sr=sr,
                                          hop_length=_HOP, n_fft=_N_FFT)

    # ---- yin pitch (full chunk) ----
    # Use librosa.yin (~30x faster than pyin, no Viterbi). Voicing is
    # derived heuristically from per-frame RMS + f0 validity since yin
    # does not emit a voicing probability.
    hop_f0 = 512
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0 = librosa.yin(
                audio_f32, fmin=50.0, fmax=500.0,
                frame_length=2048, hop_length=hop_f0,
            )
        # librosa.yin never returns NaN, but unstable frames can land at
        # the search-range edges (fmin/fmax). Treat edge hits as unvoiced.
        edge_mask = (f0 <= 50.5) | (f0 >= 499.5)
        f0 = np.where(edge_mask, np.nan, f0)
    except Exception as e:
        get_logger("features.features").emit("INFO", "diag.features.yin_degenerate",
              dur=f"{n/sr:.1f}s", error=str(e))
        nf = max(1, n // hop_f0)
        f0 = np.full(nf, np.nan)

    # Derived voicing probability: energy-weighted binary voiced/unvoiced.
    # A frame is voiced if (a) yin returned a non-NaN f0 AND (b) the
    # frame's RMS is above a chunk-relative silence floor.
    frame_len_vp = 2048
    rms_vp = []
    for i in range(len(f0)):
        start = i * hop_f0
        end = min(start + frame_len_vp, n)
        if end - start < 32:
            rms_vp.append(0.0)
        else:
            seg = audio_f64[start:end]
            rms_vp.append(float(np.sqrt(np.mean(seg ** 2))))
    rms_vp = np.asarray(rms_vp, dtype=np.float64)
    if rms_vp.size:
        # silence floor = max(1e-6, 0.1 * median RMS) — voiced requires both
        # a valid f0 and energy above this.
        rms_floor = max(1e-6, 0.1 * float(np.median(rms_vp)))
    else:
        rms_floor = 1e-6
    vp = np.where(~np.isnan(f0) & (rms_vp > rms_floor), 1.0, 0.0)

    # ---- Noise-floor: per-frame RMS and spectral histogram ----
    # Precompute frame-level FFT magnitudes for the noise-floor block.
    # Frame shape: _NF_FRAME_LEN samples, hop _NF_HOP_LEN.
    nf_frames = []
    nf_times = []
    idx = 0
    while idx + _NF_FRAME_LEN <= n:
        nf_frames.append(audio_f64[idx: idx + _NF_FRAME_LEN])
        nf_times.append((idx + _NF_FRAME_LEN / 2) / sr)
        idx += _NF_HOP_LEN
    if not nf_frames:
        nf_frames = [np.zeros(_NF_FRAME_LEN)]
        nf_times = [n / sr / 2]
    nf_fa = np.array(nf_frames)           # (n_frames, frame_len)
    nf_rms = np.sqrt(np.mean(nf_fa ** 2, axis=1))  # (n_frames,)
    # FFT magnitude per frame (rfft of frame_len)
    nf_fft_mag = np.abs(np.fft.rfft(nf_fa, axis=1))  # (n_frames, frame_len//2+1)
    nf_freqs = np.fft.rfftfreq(_NF_FRAME_LEN, d=1.0 / sr)
    nf_times_arr = np.array(nf_times)
    nf_bin_edges = np.logspace(np.log10(20.0), np.log10(sr / 2.0), _NF_N_BINS + 1)

    # Precompute per-frame histogram (n_frames, n_bins)
    nf_hist = np.zeros((len(nf_frames), _NF_N_BINS))
    for b in range(_NF_N_BINS):
        m = (nf_freqs >= nf_bin_edges[b]) & (nf_freqs < nf_bin_edges[b + 1])
        if m.any():
            nf_hist[:, b] = np.mean(nf_fft_mag[:, m], axis=1)
    row_sums = nf_hist.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    nf_hist /= row_sums  # normalize each frame's histogram

    # ---- ENF ----
    n_fft_enf = min(65536, 2 ** int(np.ceil(np.log2(max(n, 2)))))
    enf_snr_db = -999.0
    enf_hz = 50.0
    enf_ifreq = None
    enf_iphase = None
    enf_sos = None

    if n_fft_enf >= 512:
        chunk_fft = np.abs(np.fft.rfft(audio_f64, n=n_fft_enf)) ** 2
        freqs_enf = np.fft.rfftfreq(n_fft_enf, d=1.0 / sr)

        def _bp(ctr, bw=1.0):
            m = (freqs_enf >= ctr - bw/2) & (freqs_enf <= ctr + bw/2)
            return float(chunk_fft[m].sum()) if m.any() else 0.0

        pwr_50 = _bp(50) + _bp(100) + _bp(150)
        pwr_60 = _bp(60) + _bp(120) + _bp(180)
        enf_hz = 50.0 if pwr_50 >= pwr_60 else 60.0

        nb_mask = (freqs_enf >= enf_hz - 0.5) & (freqs_enf <= enf_hz + 0.5)
        nb_pwr = float(chunk_fft[nb_mask].sum()) if nb_mask.any() else 0.0
        total_pwr = float(chunk_fft.sum())
        enf_snr_db = float(10 * np.log10(max(nb_pwr, 1e-30) / max(total_pwr - nb_pwr, 1e-30)))

        if enf_snr_db >= -20.0:
            try:
                enf_sos = sp_signal.iirfilter(6, [enf_hz - 0.5, enf_hz + 0.5],
                                               btype='bandpass', fs=sr, output='sos')
                filtered = sp_signal.sosfiltfilt(enf_sos, audio_f64)
                analytic = sp_signal.hilbert(filtered)
                enf_iphase = np.unwrap(np.angle(analytic))
                enf_ifreq = np.diff(enf_iphase) / (2 * np.pi / sr)
            except Exception:
                get_logger("features.features").emit("INFO", "diag.features.enf_filter_failed")
        else:
            get_logger("features.features").emit("INFO", "diag.features.enf_absent",
                  enf_hz=enf_hz, snr_db=f"{enf_snr_db:.1f}")

    # ---- Boundary phase coherence: precompute full-chunk STFT phase ----
    _, _, Zxx_bnd = sp_signal.stft(audio_f64, fs=sr, nperseg=_N_FFT,
                                    noverlap=_N_FFT - _HOP)
    bnd_phase = np.angle(Zxx_bnd)               # (n_freq, T)
    bnd_unwrapped = np.unwrap(bnd_phase, axis=1) # (n_freq, T)
    # Mean phase advance per frequency bin (for residual normalization)
    if bnd_unwrapped.shape[1] > 1:
        bnd_dphi = np.diff(bnd_unwrapped, axis=1)  # (n_freq, T-1)
        bnd_mean_adv = np.mean(bnd_dphi, axis=1, keepdims=True)  # per-bin mean
        bnd_residual = bnd_dphi - bnd_mean_adv     # (n_freq, T-1)
        # Per-frame residual std (mean over freq bins): shape (T-1,)
        bnd_res_std_per_frame = np.std(bnd_residual, axis=0)  # (T-1,)
    else:
        bnd_res_std_per_frame = np.zeros(max(1, bnd_unwrapped.shape[1] - 1))

    # ---- Codec: HF energy ratio per frame (for pre/post window average) ----
    codec_hop = _HOP
    codec_nfft = _N_FFT
    codec_frames = []
    codec_times = []
    cidx = 0
    while cidx + codec_nfft <= n:
        codec_frames.append(audio_f64[cidx: cidx + codec_nfft])
        codec_times.append((cidx + codec_nfft / 2) / sr)
        cidx += codec_hop
    if not codec_frames:
        codec_frames = [np.zeros(codec_nfft)]
        codec_times = [n / sr / 2]
    codec_fa = np.array(codec_frames)
    codec_fft = np.abs(np.fft.rfft(codec_fa, axis=1)) ** 2
    codec_freqs = np.fft.rfftfreq(codec_nfft, d=1.0 / sr)
    hf_mask_codec = codec_freqs > sr / 4.0
    codec_total = codec_fft.sum(axis=1)
    codec_hf = codec_fft[:, hf_mask_codec].sum(axis=1) if hf_mask_codec.any() else np.zeros(len(codec_fa))
    codec_hf_ratio = codec_hf / np.maximum(codec_total, 1e-30)  # (n_frames,)
    codec_times_arr = np.array(codec_times)

    # Double-compression score (chunk-level, constant)
    env = np.mean(np.abs(Zxx_bnd) ** 2, axis=0)
    if len(env) > 2:
        env_fft = np.abs(np.fft.rfft(env)) ** 2
        noise_floor_env = float(np.median(env_fft[1:])) + 1e-30
        pf_mp3 = float(_HOP) / 576
        pf_aac = float(_HOP) / 1024

        def _pk(pf):
            idx = max(1, min(int(round(pf * len(env))), len(env_fft) - 1))
            return float(env_fft[idx]) / noise_floor_env

        dcs = float(max(_pk(pf_mp3), _pk(pf_aac)))
    else:
        dcs = 0.0

    # ---- Mel-PCA: compute once at chunk midpoint, store transform result ----
    t_mid_samp = n // 2
    win_lo = max(0, t_mid_samp - sr)
    win_hi = min(n, t_mid_samp + sr)
    seg_mel = audio_f32[win_lo:win_hi]
    if len(seg_mel) < _N_FFT:
        seg_mel = np.pad(seg_mel, (0, _N_FFT - len(seg_mel)))
    mel = librosa.feature.melspectrogram(y=seg_mel, sr=sr, n_mels=128,
                                          hop_length=_HOP, n_fft=_N_FFT)
    ref = float(mel.max()) if mel.max() > 0 else 1.0
    mel_db = librosa.power_to_db(mel, ref=ref)
    mel_flat = mel_db.flatten()
    if len(mel_flat) < _MEL_PATCH_LEN:
        mel_flat = np.pad(mel_flat, (0, _MEL_PATCH_LEN - len(mel_flat)))
    else:
        mel_flat = mel_flat[:_MEL_PATCH_LEN]

    # Run PCA transform now (once per chunk)
    global _PCA_FIT
    if _PCA_FIT is None:
        try:
            import joblib
            _PCA_FIT = joblib.load(_MEL_PCA_PATH)
        except Exception:
            get_logger("features.features").emit("INFO", "diag.features.mel_pca_self_fit",
                  fallback="joblib_load_failed_or_missing", path=_MEL_PCA_PATH)
            from sklearn.decomposition import PCA
            rs = np.random.RandomState(0)
            dummy = rs.randn(max(2, _MEL_N_COMPONENTS + 1), _MEL_PATCH_LEN).astype(np.float32)
            pca = PCA(n_components=_MEL_N_COMPONENTS, random_state=0)
            pca.fit(dummy)
            _PCA_FIT = pca

    try:
        mel_pca_vec = _PCA_FIT.transform(mel_flat.reshape(1, -1))[0].tolist()
    except Exception:
        mel_pca_vec = [0.0] * _MEL_N_COMPONENTS

    # ---- Store everything ----
    ctx["feat_audio"] = audio
    ctx["feat_sr"] = sr
    # spectral frames
    ctx["feat_mfcc"] = mfcc
    ctx["feat_onset"] = onset
    ctx["feat_centroid"] = centroid
    ctx["feat_rolloff"] = rolloff
    ctx["feat_flatness"] = flatness
    ctx["feat_bandwidth"] = bandwidth
    ctx["feat_contrast"] = contrast
    ctx["feat_zcr"] = zcr
    ctx["feat_rms"] = rms
    ctx["feat_chroma"] = chroma
    ctx["feat_frame_hop"] = _HOP
    # pitch
    ctx["feat_f0"] = f0
    ctx["feat_vp"] = vp
    ctx["feat_f0_hop"] = hop_f0
    # noise-floor
    ctx["feat_nf_hist"] = nf_hist           # (n_frames, n_bins)
    ctx["feat_nf_rms"] = nf_rms             # (n_frames,)
    ctx["feat_nf_times"] = nf_times_arr     # (n_frames,) in seconds
    ctx["feat_nf_bin_edges"] = nf_bin_edges
    # ENF
    ctx["feat_enf_hz"] = enf_hz
    ctx["feat_enf_ifreq"] = enf_ifreq
    ctx["feat_enf_iphase"] = enf_iphase
    ctx["feat_enf_snr_db"] = enf_snr_db
    # boundary phase
    ctx["feat_bnd_res_std"] = bnd_res_std_per_frame  # (T-1,) per stft frame
    ctx["feat_bnd_stft_hop"] = _HOP
    # codec
    ctx["feat_codec_hf"] = codec_hf_ratio   # (n_frames,)
    ctx["feat_codec_times"] = codec_times_arr
    ctx["feat_codec_dcs"] = dcs
    # mel-PCA
    ctx["feat_mel_pca_vec"] = mel_pca_vec   # list[float], length 20

    _save_feat_cache(audio, sr, ctx)


# ---------------------------------------------------------------------------
# Slice helper
# ---------------------------------------------------------------------------

def _slice_frames(arr: np.ndarray, hop: int, sr: int,
                  t_lo: float, t_hi: float) -> np.ndarray:
    """Slice the last axis of arr between [t_lo, t_hi] seconds.
    Returns at least 1 column (zeros) if the window is empty.
    """
    i_lo = max(0, int(t_lo * sr / hop))
    i_hi = max(i_lo + 1, int(np.ceil(t_hi * sr / hop)))
    i_hi = min(i_hi, arr.shape[-1])
    sliced = arr[..., i_lo:i_hi]
    if sliced.shape[-1] == 0:
        shape = list(arr.shape)
        shape[-1] = 1
        return np.zeros(shape, dtype=arr.dtype)
    return sliced


def _slice_time_arr(vals: np.ndarray, times: np.ndarray,
                    t_lo: float, t_hi: float) -> np.ndarray:
    """Select values where times falls in [t_lo, t_hi)."""
    mask = (times >= t_lo) & (times < t_hi)
    return vals[mask] if mask.any() else np.zeros(1)


def _safe_nanmean(x: np.ndarray, default: float = 0.0) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanmean(x)
    return float(default if (np.isnan(v) if not hasattr(v, '__len__') else False) else v)


# ---------------------------------------------------------------------------
# Block 1: DSP
# ---------------------------------------------------------------------------

def _block_dsp(ctx: dict, t_sec: float) -> dict[str, float]:
    return {
        "dsp_phase_z": phase_z_at(ctx, t_sec),
        "dsp_t2_z": t2_z_at(ctx, t_sec),
        "dsp_cpe_z": cpe_z_at(ctx, t_sec),
        "dsp_pairwise_proximity": pairwise_proximity_at(ctx, t_sec),
    }


# ---------------------------------------------------------------------------
# Block 2: MFCC-13 deltas
# ---------------------------------------------------------------------------

def _block_mfcc(ctx: dict, t_sec: float) -> dict[str, float]:
    mfcc = ctx["feat_mfcc"]
    hop = ctx["feat_frame_hop"]
    sr = ctx["feat_sr"]
    pre_w = _slice_frames(mfcc, hop, sr, t_sec - 2.0, t_sec)
    post_w = _slice_frames(mfcc, hop, sr, t_sec, t_sec + 2.0)
    delta = np.mean(post_w, axis=1) - np.mean(pre_w, axis=1)
    return {f"mfcc_delta_{i+1:02d}": float(delta[i]) for i in range(_N_MFCC)}


# ---------------------------------------------------------------------------
# Block 3: Spectral summary deltas
# ---------------------------------------------------------------------------

def _block_spectral(ctx: dict, t_sec: float) -> dict[str, float]:
    hop = ctx["feat_frame_hop"]
    sr = ctx["feat_sr"]

    def _delta(arr):
        pre = _slice_frames(arr, hop, sr, t_sec - 2.0, t_sec)
        post = _slice_frames(arr, hop, sr, t_sec, t_sec + 2.0)
        return float(np.mean(post)) - float(np.mean(pre))

    return {
        "spec_centroid_delta": _delta(ctx["feat_centroid"]),
        "spec_rolloff_delta": _delta(ctx["feat_rolloff"]),
        "spec_flux_delta": _delta(ctx["feat_onset"]),
        "spec_contrast_delta": _delta(ctx["feat_contrast"]),
        "spec_flatness_delta": _delta(ctx["feat_flatness"]),
        "spec_bandwidth_delta": _delta(ctx["feat_bandwidth"]),
    }


# ---------------------------------------------------------------------------
# Block 4: Noise-floor color
# ---------------------------------------------------------------------------

def _block_noise_floor(ctx: dict, t_sec: float) -> dict[str, float]:
    hist = ctx["feat_nf_hist"]        # (n_frames, n_bins)
    rms_arr = ctx["feat_nf_rms"]      # (n_frames,)
    times = ctx["feat_nf_times"]      # (n_frames,)
    bin_edges = ctx["feat_nf_bin_edges"]
    eps = 1e-6

    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    def _select(t_lo, t_hi):
        mask = (times >= t_lo) & (times < t_hi)
        if not mask.any():
            # fallback: nearest frame
            idx = int(np.argmin(np.abs(times - (t_lo + t_hi) / 2)))
            mask = np.zeros(len(times), dtype=bool)
            mask[idx] = True
        return hist[mask], rms_arr[mask]

    def _bottom10(h_arr, r_arr):
        k = max(1, int(np.ceil(len(r_arr) * 0.10)))
        idx = np.argsort(r_arr)[:k]
        return h_arr[idx], r_arr[idx]

    def _agg_hist(h_arr):
        h = np.mean(h_arr, axis=0)
        s = h.sum()
        return h / s if s >= eps else np.ones(len(bin_edges) - 1) / (len(bin_edges) - 1)

    def _centroid(h): return float(np.dot(h, bin_centers))
    def _rolloff(h, pct=0.85):
        idx = min(np.searchsorted(np.cumsum(h), pct), len(bin_centers) - 1)
        return float(bin_centers[idx])
    def _flatness(h):
        hh = np.maximum(h, eps)
        return float(np.exp(np.mean(np.log(hh))) / max(np.mean(hh), eps))

    pre_h, pre_r = _select(t_sec - 2.0, t_sec)
    post_h, post_r = _select(t_sec, t_sec + 2.0)
    pre_h10, pre_r10 = _bottom10(pre_h, pre_r)
    post_h10, post_r10 = _bottom10(post_h, post_r)

    ph = _agg_hist(pre_h10)
    qh = _agg_hist(post_h10)

    p = np.maximum(ph, eps); p /= p.sum()
    q = np.maximum(qh, eps); q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))

    pre_e_db = float(20 * np.log10(max(float(np.mean(pre_r10)), 1e-10)))
    post_e_db = float(20 * np.log10(max(float(np.mean(post_r10)), 1e-10)))

    return {
        "nf_kl_divergence": kl,
        "nf_centroid_delta": _centroid(qh) - _centroid(ph),
        "nf_rolloff_delta": _rolloff(qh) - _rolloff(ph),
        "nf_flatness_delta": _flatness(qh) - _flatness(ph),
        "nf_bottom10_e_pre": pre_e_db,
        "nf_bottom10_e_post": post_e_db,
    }


# ---------------------------------------------------------------------------
# Block 5: Pitch / voicing
# ---------------------------------------------------------------------------

def _block_pitch(ctx: dict, t_sec: float) -> dict[str, float]:
    f0 = ctx["feat_f0"]
    vp = ctx["feat_vp"]
    hop = ctx["feat_f0_hop"]
    sr = ctx["feat_sr"]

    pre_f0 = _slice_frames(f0, hop, sr, t_sec - 2.0, t_sec)
    post_f0 = _slice_frames(f0, hop, sr, t_sec, t_sec + 2.0)
    pre_vp = _slice_frames(vp, hop, sr, t_sec - 2.0, t_sec)
    post_vp = _slice_frames(vp, hop, sr, t_sec, t_sec + 2.0)

    def _jitter(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return 0.0
        mu = float(np.mean(valid))
        return float(np.std(valid) / mu) if mu > 1e-6 else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f0_mean_pre = float(np.nanmean(pre_f0)) if not np.all(np.isnan(pre_f0)) else 0.0
        f0_mean_post = float(np.nanmean(post_f0)) if not np.all(np.isnan(post_f0)) else 0.0

    vp_pre = float(np.mean(pre_vp))
    vp_post = float(np.mean(post_vp))

    # Edge F0: last/first 100ms around t
    edge_s = 0.1
    pre_edge = _slice_frames(f0, hop, sr, t_sec - edge_s, t_sec)
    post_edge = _slice_frames(f0, hop, sr, t_sec, t_sec + edge_s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pre_ef = float(np.nanmean(pre_edge)) if not np.all(np.isnan(pre_edge)) else float("nan")
        post_ef = float(np.nanmean(post_edge)) if not np.all(np.isnan(post_edge)) else float("nan")

    if np.isnan(pre_ef) or np.isnan(post_ef):
        f0_cont = 0.0
    else:
        f0_cont = float(np.clip(1.0 - abs(post_ef - pre_ef) / 50.0, 0.0, 1.0))

    return {
        "f0_mean_pre": f0_mean_pre,
        "f0_mean_post": f0_mean_post,
        "f0_delta": f0_mean_post - f0_mean_pre,
        "f0_jitter_delta": _jitter(post_f0) - _jitter(pre_f0),
        "voicing_prob_pre": vp_pre,
        "voicing_prob_post": vp_post,
        "voicing_prob_delta": vp_post - vp_pre,
        "f0_continuity_score": f0_cont,
    }


# ---------------------------------------------------------------------------
# Block 6: Energy / ZCR
# ---------------------------------------------------------------------------

def _block_energy(ctx: dict, t_sec: float) -> dict[str, float]:
    rms = ctx["feat_rms"]
    zcr = ctx["feat_zcr"]
    hop = ctx["feat_frame_hop"]
    sr = ctx["feat_sr"]

    pre_r = _slice_frames(rms, hop, sr, t_sec - 2.0, t_sec)
    post_r = _slice_frames(rms, hop, sr, t_sec, t_sec + 2.0)
    pre_z = _slice_frames(zcr, hop, sr, t_sec - 2.0, t_sec)
    post_z = _slice_frames(zcr, hop, sr, t_sec, t_sec + 2.0)

    rms_pre = float(20 * np.log10(max(float(np.mean(pre_r)), 1e-10)))
    rms_post = float(20 * np.log10(max(float(np.mean(post_r)), 1e-10)))

    return {
        "rms_db_pre": rms_pre,
        "rms_db_post": rms_post,
        "rms_db_delta": rms_post - rms_pre,
        "zcr_pre": float(np.mean(pre_z)),
        "zcr_post": float(np.mean(post_z)),
        "zcr_delta": float(np.mean(post_z)) - float(np.mean(pre_z)),
    }


# ---------------------------------------------------------------------------
# Block 7: Boundary region +/-200ms
# ---------------------------------------------------------------------------

def _block_boundary(audio: np.ndarray, sr: int, ctx: dict, t_sec: float) -> dict[str, float]:
    n = len(audio)
    t_samp = int(round(t_sec * sr))
    bnd_s = 0.2
    bnd_samples = int(bnd_s * sr)

    # Flux peak from precomputed onset curve
    onset = ctx["feat_onset"]
    hop = ctx["feat_frame_hop"]
    bnd_onset = _slice_frames(onset, hop, sr, t_sec - bnd_s, t_sec + bnd_s)
    flux_peak = float(np.max(bnd_onset))

    # Energy ratio
    lo = max(0, t_samp - bnd_samples)
    hi = min(n, t_samp + bnd_samples)
    pre_seg = audio[lo:t_samp].astype(np.float32)
    post_seg = audio[t_samp:hi].astype(np.float32)
    if len(pre_seg) == 0:
        pre_seg = np.zeros(1, dtype=np.float32)
    if len(post_seg) == 0:
        post_seg = np.zeros(1, dtype=np.float32)
    rms_pre = float(np.sqrt(np.mean(pre_seg ** 2)))
    rms_post = float(np.sqrt(np.mean(post_seg ** 2)))
    # Guard both sides against zero RMS (pure-silence pre/post windows) so
    # log10 can't return ±inf and break downstream StandardScaler validation.
    energy_ratio = float(
        20 * np.log10(max(rms_post, 1e-10) / max(rms_pre, 1e-10))
    )

    # Phase coherence from precomputed residual std per STFT frame
    bnd_res = ctx["feat_bnd_res_std"]   # (T-1,) in STFT hop units
    bnd_hop = ctx["feat_bnd_stft_hop"]
    bnd_slice = _slice_frames(bnd_res, bnd_hop, sr, t_sec - bnd_s, t_sec + bnd_s)
    mean_res_std = float(np.mean(bnd_slice))
    expected_std = float(np.pi / np.sqrt(3))
    norm_disc = min(mean_res_std / max(expected_std, 1e-10), 1.0)
    phase_coherence = float(1.0 - norm_disc)

    return {
        "boundary_spec_flux_peak": flux_peak,
        "boundary_energy_ratio": energy_ratio,
        "boundary_phase_coherence": phase_coherence,
    }


# ---------------------------------------------------------------------------
# Block 8: ENF
# ---------------------------------------------------------------------------

def _block_enf(ctx: dict, t_sec: float) -> dict[str, float]:
    _ZEROS = {k: 0.0 for k in ["enf_freq_mean_pre", "enf_freq_mean_post",
                                "enf_freq_delta", "enf_phase_continuity", "enf_snr_db"]}

    snr_db = ctx["feat_enf_snr_db"]
    enf_hz = ctx["feat_enf_hz"]
    ifreq = ctx["feat_enf_ifreq"]
    iphase = ctx["feat_enf_iphase"]
    sr = ctx["feat_sr"]

    if ifreq is None or iphase is None:
        z = dict(_ZEROS)
        z["enf_snr_db"] = float(snr_db)
        return z

    def _mfreq(t_lo, t_hi):
        lo = max(0, int(t_lo * sr))
        hi = min(len(ifreq), int(t_hi * sr))
        if lo >= hi:
            return float(enf_hz)
        return float(np.mean(ifreq[lo:hi]))

    enf_pre = _mfreq(t_sec - 2.0, t_sec)
    enf_post = _mfreq(t_sec, t_sec + 2.0)

    bnd_s = 0.1
    t_samp = int(round(t_sec * sr))
    ph_seg = iphase[max(0, t_samp - int(bnd_s * sr)): min(len(iphase), t_samp + int(bnd_s * sr))]
    if len(ph_seg) < 4:
        phase_cont = 0.0
    else:
        x = np.arange(len(ph_seg), dtype=np.float64)
        coeffs = np.polyfit(x, ph_seg, 1)
        res_std = float(np.std(ph_seg - np.polyval(coeffs, x)))
        expected_std = float(1.0 / (2 * np.pi * 0.5))
        phase_cont = float(1.0 - min(res_std / max(expected_std, 1e-10), 1.0))

    return {
        "enf_freq_mean_pre": enf_pre,
        "enf_freq_mean_post": enf_post,
        "enf_freq_delta": enf_post - enf_pre,
        "enf_phase_continuity": phase_cont,
        "enf_snr_db": float(snr_db),
    }


# ---------------------------------------------------------------------------
# Block 9: Codec artifact
# ---------------------------------------------------------------------------

def _block_codec(ctx: dict, t_sec: float) -> dict[str, float]:
    sr = ctx["feat_sr"]
    t_samp = int(round(t_sec * sr))
    mp3_frame = 576
    aac_frame = 1024

    mp3_dist = min(t_samp % mp3_frame, mp3_frame - t_samp % mp3_frame)
    aac_dist = min(t_samp % aac_frame, aac_frame - t_samp % aac_frame)
    align_offset = float(min(float(mp3_dist) / mp3_frame, float(aac_dist) / aac_frame))

    hf_ratio = ctx["feat_codec_hf"]    # (n_frames,)
    times = ctx["feat_codec_times"]    # (n_frames,)

    pre_vals = _slice_time_arr(hf_ratio, times, t_sec - 2.0, t_sec)
    post_vals = _slice_time_arr(hf_ratio, times, t_sec, t_sec + 2.0)

    return {
        "codec_frame_align_offset": align_offset,
        "codec_quant_residual_pre": float(np.mean(pre_vals)),
        "codec_quant_residual_post": float(np.mean(post_vals)),
        "codec_double_compression_score": ctx["feat_codec_dcs"],
    }


# ---------------------------------------------------------------------------
# Block 10: Mel-PCA tail (precomputed in _ensure_feat_cache)
# ---------------------------------------------------------------------------

def _block_mel_pca(ctx: dict) -> dict[str, float]:
    vec = ctx["feat_mel_pca_vec"]
    return {f"mel_pca_{i+1:02d}": float(vec[i]) for i in range(_MEL_N_COMPONENTS)}


# ---------------------------------------------------------------------------
# Block 11: Voiced-frame-only MFCC cosine distance
# ---------------------------------------------------------------------------

def _block_voiced_mfcc(ctx: dict, t_sec: float) -> dict[str, float]:
    # Vocal-tract / instrument timbre signature restricted to voiced frames.
    # Same singer across a chord boundary keeps voiced MFCC similar; a
    # cross-source splice shifts it. Sentinel 0.0 when either window has
    # no voiced frames so GBM sees a clean no-signal floor.
    mfcc = ctx["feat_mfcc"]
    vp = ctx["feat_vp"]
    hop = ctx["feat_frame_hop"]
    sr = ctx["feat_sr"]

    def _voiced_mean(t_lo: float, t_hi: float):
        m = _slice_frames(mfcc, hop, sr, t_lo, t_hi)        # (13, n_m)
        v = _slice_frames(vp, hop, sr, t_lo, t_hi)          # (n_v,)
        n = min(m.shape[1], v.shape[0])
        if n <= 0:
            return None
        m = m[:, :n]
        mask = v[:n].astype(bool)
        if not mask.any():
            return None
        return np.mean(m[:, mask], axis=1)

    pre_v = _voiced_mean(t_sec - 2.0, t_sec)
    post_v = _voiced_mean(t_sec, t_sec + 2.0)
    if pre_v is None or post_v is None:
        return {"voiced_mfcc_cosine_dist": 0.0}

    norm_pre = float(np.linalg.norm(pre_v))
    norm_post = float(np.linalg.norm(post_v))
    if norm_pre < 1e-10 or norm_post < 1e-10:
        return {"voiced_mfcc_cosine_dist": 0.0}

    cos_sim = float(np.dot(pre_v, post_v) / (norm_pre * norm_post))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return {"voiced_mfcc_cosine_dist": float(1.0 - cos_sim)}


# ---------------------------------------------------------------------------
# Block 12: Voiced-frame-only chroma cosine distance
# ---------------------------------------------------------------------------

def _block_voiced_chroma(ctx: dict, t_sec: float) -> dict[str, float]:
    # Pitch-class-profile continuity restricted to voiced frames. Within a
    # song's key most chord transitions share 3-5 of 12 pitch classes so
    # voiced-chroma cosine distance is small; cross-source splices shift
    # key entirely. On speech voiced=vowels whose per-vowel chroma swings
    # with prosody regardless of splice -> high-variance noise, GBM learns
    # low per-domain SHAP. Sentinel 0.0 when either window has no voiced
    # frames.
    chroma = ctx["feat_chroma"]
    vp = ctx["feat_vp"]
    hop = ctx["feat_frame_hop"]
    sr = ctx["feat_sr"]

    def _voiced_mean(t_lo: float, t_hi: float):
        c = _slice_frames(chroma, hop, sr, t_lo, t_hi)      # (12, n_c)
        v = _slice_frames(vp, hop, sr, t_lo, t_hi)          # (n_v,)
        n = min(c.shape[1], v.shape[0])
        if n <= 0:
            return None
        c = c[:, :n]
        mask = v[:n].astype(bool)
        if not mask.any():
            return None
        return np.mean(c[:, mask], axis=1)

    pre_v = _voiced_mean(t_sec - 2.0, t_sec)
    post_v = _voiced_mean(t_sec, t_sec + 2.0)
    if pre_v is None or post_v is None:
        return {"voiced_chroma_cosine_dist": 0.0}

    norm_pre = float(np.linalg.norm(pre_v))
    norm_post = float(np.linalg.norm(post_v))
    if norm_pre < 1e-10 or norm_post < 1e-10:
        return {"voiced_chroma_cosine_dist": 0.0}

    cos_sim = float(np.dot(pre_v, post_v) / (norm_pre * norm_post))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return {"voiced_chroma_cosine_dist": float(1.0 - cos_sim)}


# ---------------------------------------------------------------------------
# Block 13: Voiced-frame-only spectral-contrast cosine distance
# ---------------------------------------------------------------------------

def _block_voiced_spec_contrast(ctx: dict, t_sec: float) -> dict[str, float]:
    # 7-band peak-to-valley amplitude ratio (mastering / mix signature)
    # restricted to voiced frames. Within one song the mastering chain is
    # fixed, so chord transitions preserve per-band peak/valley profile;
    # cross-song splices cross mastering chains (different compression /
    # EQ / limiter settings). Self-gating on speech: voiced vowels have
    # per-phoneme formant peaks at different bands -> high-variance noise
    # -> GBM learns low per-domain SHAP on english/korean.
    contrast = ctx["feat_contrast"]
    vp = ctx["feat_vp"]
    hop = ctx["feat_frame_hop"]
    sr = ctx["feat_sr"]

    def _voiced_mean(t_lo: float, t_hi: float):
        c = _slice_frames(contrast, hop, sr, t_lo, t_hi)    # (7, n_c)
        v = _slice_frames(vp, hop, sr, t_lo, t_hi)          # (n_v,)
        n = min(c.shape[1], v.shape[0])
        if n <= 0:
            return None
        c = c[:, :n]
        mask = v[:n].astype(bool)
        if not mask.any():
            return None
        return np.mean(c[:, mask], axis=1)

    pre_v = _voiced_mean(t_sec - 2.0, t_sec)
    post_v = _voiced_mean(t_sec, t_sec + 2.0)
    if pre_v is None or post_v is None:
        return {"voiced_spec_contrast_cosine_dist": 0.0}

    norm_pre = float(np.linalg.norm(pre_v))
    norm_post = float(np.linalg.norm(post_v))
    if norm_pre < 1e-10 or norm_post < 1e-10:
        return {"voiced_spec_contrast_cosine_dist": 0.0}

    cos_sim = float(np.dot(pre_v, post_v) / (norm_pre * norm_post))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return {"voiced_spec_contrast_cosine_dist": float(1.0 - cos_sim)}


# ---------------------------------------------------------------------------
# Block 14: Voiced/unvoiced MFCC asymmetry (accompaniment vs voice)
# ---------------------------------------------------------------------------

def _block_voiced_unvoiced_mfcc_asymmetry(ctx: dict, t_sec: float) -> dict[str, float]:
    # Signed asymmetry between unvoiced and voiced pre/post MFCC cosine
    # distances. Positive = accompaniment changed more than voice (the
    # same-singer-cross-song signature); ~0 = either same-source (both low)
    # or different-singer-cross-song (both high and cancel). Paired
    # differencing cancels phoneme-correlated noise on speech where both
    # voiced and unvoiced MFCC vary together with context.
    mfcc = ctx["feat_mfcc"]
    vp = ctx["feat_vp"]
    hop = ctx["feat_frame_hop"]
    sr = ctx["feat_sr"]

    def _masked_mean(t_lo: float, t_hi: float, voiced: bool):
        m = _slice_frames(mfcc, hop, sr, t_lo, t_hi)        # (13, n_m)
        v = _slice_frames(vp, hop, sr, t_lo, t_hi)          # (n_v,)
        n = min(m.shape[1], v.shape[0])
        if n <= 0:
            return None
        m = m[:, :n]
        mask = v[:n].astype(bool)
        if not voiced:
            mask = ~mask
        if not mask.any():
            return None
        return np.mean(m[:, mask], axis=1)

    def _cos_dist(pre, post):
        if pre is None or post is None:
            return None
        np_pre = float(np.linalg.norm(pre))
        np_post = float(np.linalg.norm(post))
        if np_pre < 1e-10 or np_post < 1e-10:
            return None
        cs = float(np.dot(pre, post) / (np_pre * np_post))
        cs = max(-1.0, min(1.0, cs))
        return float(1.0 - cs)

    pre_v = _masked_mean(t_sec - 2.0, t_sec, voiced=True)
    post_v = _masked_mean(t_sec, t_sec + 2.0, voiced=True)
    pre_u = _masked_mean(t_sec - 2.0, t_sec, voiced=False)
    post_u = _masked_mean(t_sec, t_sec + 2.0, voiced=False)

    voiced_dist = _cos_dist(pre_v, post_v)
    unvoiced_dist = _cos_dist(pre_u, post_u)
    if voiced_dist is None or unvoiced_dist is None:
        return {"voiced_unvoiced_mfcc_asymmetry": 0.0}

    return {"voiced_unvoiced_mfcc_asymmetry": float(unvoiced_dist - voiced_dist)}


# ---------------------------------------------------------------------------
# Block 15: Voiced/unvoiced spectral-contrast asymmetry (mastering signature)
# ---------------------------------------------------------------------------

def _block_voiced_unvoiced_spec_contrast_asymmetry(ctx: dict, t_sec: float) -> dict[str, float]:
    # Signed asymmetry between unvoiced and voiced pre/post 7-dim
    # spectral_contrast cosine distances. Spec_contrast = peak-to-valley
    # amplitude ratio per frequency band — a mastering/compressor
    # fingerprint. Positive = unvoiced (drums / mastering tail) changed
    # more than voiced (vowel formant structure) = same-singer cross-song
    # splice signature. Negative = voiced shifts more than unvoiced =
    # intra-song chord cycle (same mastering, different vowel per phrase).
    contrast = ctx["feat_contrast"]
    vp = ctx["feat_vp"]
    hop = ctx["feat_frame_hop"]
    sr = ctx["feat_sr"]

    def _masked_mean(t_lo: float, t_hi: float, voiced: bool):
        c = _slice_frames(contrast, hop, sr, t_lo, t_hi)    # (7, n_c)
        v = _slice_frames(vp, hop, sr, t_lo, t_hi)          # (n_v,)
        n = min(c.shape[1], v.shape[0])
        if n <= 0:
            return None
        c = c[:, :n]
        mask = v[:n].astype(bool)
        if not voiced:
            mask = ~mask
        if not mask.any():
            return None
        return np.mean(c[:, mask], axis=1)

    def _cos_dist(pre, post):
        if pre is None or post is None:
            return None
        np_pre = float(np.linalg.norm(pre))
        np_post = float(np.linalg.norm(post))
        if np_pre < 1e-10 or np_post < 1e-10:
            return None
        cs = float(np.dot(pre, post) / (np_pre * np_post))
        cs = max(-1.0, min(1.0, cs))
        return float(1.0 - cs)

    pre_v = _masked_mean(t_sec - 2.0, t_sec, voiced=True)
    post_v = _masked_mean(t_sec, t_sec + 2.0, voiced=True)
    pre_u = _masked_mean(t_sec - 2.0, t_sec, voiced=False)
    post_u = _masked_mean(t_sec, t_sec + 2.0, voiced=False)

    voiced_dist = _cos_dist(pre_v, post_v)
    unvoiced_dist = _cos_dist(pre_u, post_u)
    if voiced_dist is None or unvoiced_dist is None:
        return {"voiced_unvoiced_spec_contrast_asymmetry": 0.0}

    return {"voiced_unvoiced_spec_contrast_asymmetry": float(unvoiced_dist - voiced_dist)}


# ---------------------------------------------------------------------------
# Block 16: Wide-window stationarity (clean-audio guard)
# ---------------------------------------------------------------------------

def _block_clean_audio_guard(ctx: dict, t_sec: float) -> dict[str, float]:
    # CV (std/mean) of spectral centroid over a single ±1s window centered
    # at t_sec. Existing features are pre/post deltas at t_sec or tight
    # ±200ms boundary stats; none measure wide-window stationarity. Low CV
    # = uniform stationary speech (clean-FP source); high CV = step-change
    # boundary (TP source). Gives the GBM a wide-context signal it can
    # use to suppress confident emits inside continuous clean speech.
    centroid = ctx["feat_centroid"]
    hop = ctx["feat_frame_hop"]
    sr = ctx["feat_sr"]
    win_s = 1.0
    cent_win = _slice_frames(centroid, hop, sr, t_sec - win_s, t_sec + win_s).flatten()
    if cent_win.size < 2:
        cv = 0.0
    else:
        mean = float(np.mean(cent_win))
        if mean > 1e-6:
            cv = float(np.std(cent_win) / mean)
        else:
            cv = 0.0
    return {"stationarity_centroid_cv_1s": cv}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_features(
    audio: np.ndarray,
    sr: int,
    t_sec: float,
    chunk_ctx: Optional[dict] = None,
) -> dict[str, float]:
    """Extract ~75 named features at candidate splice position t_sec.

    Parameters
    ----------
    audio : np.ndarray
        Full audio chunk (float32/float64, mono).
    sr : int
        Sample rate.
    t_sec : float
        Candidate splice time in seconds (chunk-local).
    chunk_ctx : dict | None
        Precomputed chunk context from _build_chunk_context(audio, sr).
        If None, it is built internally (expensive). Pass a cached context
        for dense scanning.

    Returns
    -------
    dict[str, float]
        Exactly 75 named features in FEATURE_NAMES order.
    """
    if chunk_ctx is None:
        chunk_ctx = _build_chunk_context(audio, sr)

    # Populate expensive precomputed arrays into ctx (once per chunk)
    _ensure_feat_cache(audio, sr, chunk_ctx)

    feats: dict[str, float] = {}
    feats.update(_block_dsp(chunk_ctx, t_sec))
    feats.update(_block_mfcc(chunk_ctx, t_sec))
    feats.update(_block_spectral(chunk_ctx, t_sec))
    feats.update(_block_noise_floor(chunk_ctx, t_sec))
    feats.update(_block_pitch(chunk_ctx, t_sec))
    feats.update(_block_energy(chunk_ctx, t_sec))
    feats.update(_block_boundary(audio, sr, chunk_ctx, t_sec))
    feats.update(_block_enf(chunk_ctx, t_sec))
    feats.update(_block_codec(chunk_ctx, t_sec))
    feats.update(_block_mel_pca(chunk_ctx))
    feats.update(_block_voiced_mfcc(chunk_ctx, t_sec))
    feats.update(_block_voiced_chroma(chunk_ctx, t_sec))
    feats.update(_block_voiced_spec_contrast(chunk_ctx, t_sec))
    feats.update(_block_voiced_unvoiced_mfcc_asymmetry(chunk_ctx, t_sec))
    feats.update(_block_voiced_unvoiced_spec_contrast_asymmetry(chunk_ctx, t_sec))
    feats.update(_block_clean_audio_guard(chunk_ctx, t_sec))

    assert len(feats) == len(FEATURE_NAMES), (
        f"Feature count mismatch: {len(feats)} != {len(FEATURE_NAMES)}"
    )

    # Sanitize: downstream StandardScaler rejects non-finite values. Any
    # inf/nan that escapes the per-block guards lands on a 0.0 floor with a
    # DIAG line so silent poisoning of the feature matrix stays visible.
    n_scrubbed = 0
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0
            n_scrubbed += 1
    if n_scrubbed:
        get_logger("features.features").emit("INFO", "diag.features.nonfinite_scrubbed",
              count=n_scrubbed, t_sec=f"{t_sec:.3f}")
    return feats


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    wav_path = "data/eval/singing/tier1/splice_t1_001.wav"
    print(f"Loading {wav_path} ...")
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    print(f"  sr={sr}  duration={len(audio)/sr:.1f}s")

    chunk = audio[: min(len(audio), 60 * sr)]
    print("Building chunk_ctx ...")
    ctx = _build_chunk_context(chunk, sr)

    t = 30.0
    print(f"Calling extract_features at t={t}s (first call, warms cache) ...")
    feats1 = extract_features(chunk, sr, t, chunk_ctx=ctx)

    # 1. Length check
    assert len(feats1) == 80, f"FAIL: got {len(feats1)} features"
    assert len(FEATURE_NAMES) == 80, f"FAIL: FEATURE_NAMES has {len(FEATURE_NAMES)}"
    print("PASS: len(dict) == len(FEATURE_NAMES) == 80")

    # 2. Bit-identical
    feats2 = extract_features(chunk, sr, t, chunk_ctx=ctx)
    for k in FEATURE_NAMES:
        v1, v2 = feats1[k], feats2[k]
        assert v1 == v2, f"FAIL: non-idempotent for {k}: {v1} != {v2}"
    print("PASS: bit-identical on repeated calls")

    # 3. 1000 calls timing
    N = 1000
    start = time.perf_counter()
    for _ in range(N):
        extract_features(chunk, sr, t, chunk_ctx=ctx)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 2000.0, f"FAIL: 1000 calls took {elapsed_ms:.0f}ms (budget 2000ms)"
    print(f"PASS: 1000 calls: {elapsed_ms:.0f} ms")

    # 4. Print first 5 and last 5
    print("\nFirst 5 features:")
    for k in FEATURE_NAMES[:5]:
        print(f"  {k}: {feats1[k]:.6f}")
    print("Last 5 features:")
    for k in FEATURE_NAMES[-5:]:
        print(f"  {k}: {feats1[k]:.6f}")
