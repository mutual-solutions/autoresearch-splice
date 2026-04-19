"""
False positive filter for splice detector.

Loads a trained sklearn classifier and filters detections by
computing mel spectrogram patches and running inference.

Includes SHAP explainability via shap.TreeExplainer.
"""

import hashlib
import json
import os
import numpy as np
import joblib
import librosa

# Patch extraction parameters (must match generate_patches.py)
PATCH_HALF_S = 2.0
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
TARGET_FRAMES = 200

# Default model path
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "splice", "classifier")
_MODEL_PATH = os.path.join(_MODEL_DIR, "fp_classifier.joblib")

# Cached model and explainer
_model = None
_explainer = None
_model_hash = None

# Accumulated SHAP results across calls (written at process exit)
_shap_results = []
_shap_atexit_registered = False


def _load_model():
    """Load the classifier model (cached)."""
    global _model, _model_hash
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"FP classifier model not found at {_MODEL_PATH}")
        _model = joblib.load(_MODEL_PATH)
        with open(_MODEL_PATH, "rb") as f:
            _model_hash = hashlib.sha256(f.read()).hexdigest()
    return _model


def _get_explainer():
    """Get or create a SHAP TreeExplainer for the GradientBoosting classifier (cached)."""
    global _explainer
    if _explainer is None:
        import shap
        model = _load_model()
        # The pipeline is: scaler -> pca -> clf
        # TreeExplainer operates on the final GradientBoosting estimator
        gb_clf = model.named_steps['clf']
        _explainer = shap.TreeExplainer(gb_clf)
    return _explainer


def _extract_mel_patch(audio, sr, center_s):
    """Extract a mel spectrogram patch centered at center_s."""
    center_sample = int(center_s * sr)
    half_samples = int(PATCH_HALF_S * sr)
    start = max(0, center_sample - half_samples)
    end = min(len(audio), center_sample + half_samples)

    segment = audio[start:end]
    if len(segment) < sr // 2:
        return None

    S = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize to [0, 1]
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

    # Pad or crop to TARGET_FRAMES
    if S_db.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
    elif S_db.shape[1] > TARGET_FRAMES:
        S_db = S_db[:, :TARGET_FRAMES]

    return S_db


def _transform_to_pca(model, X_flat):
    """Apply scaler + PCA from the pipeline to get PCA-space features."""
    X_scaled = model.named_steps['scaler'].transform(X_flat)
    X_pca = model.named_steps['pca'].transform(X_scaled)
    return X_pca


def filter_detections(audio, sr, detections, threshold=0.01):
    """Filter detections using the trained classifier.

    Args:
        audio: numpy array of audio samples
        sr: sample rate
        detections: list of detection times (seconds)
        threshold: minimum classifier confidence to keep a detection

    Returns:
        List of detection times where classifier confidence >= threshold.
    """
    if not detections:
        return detections

    model = _load_model()

    kept = []
    for det_time in detections:
        patch = _extract_mel_patch(audio, sr, float(det_time))
        if patch is None:
            # Can't extract patch -- keep detection to be safe
            kept.append(det_time)
            continue

        # Flatten and predict
        X = patch.reshape(1, -1)
        X_pca = _transform_to_pca(model, X)
        proba = model.named_steps['clf'].predict_proba(X_pca)[0, 1]

        if proba >= threshold:
            kept.append(det_time)

    # Side effect: accumulate SHAP results when env var is set
    if os.environ.get("SPLICE_SHAP_REPORT") == "1":
        explained = filter_detections_explained(audio, sr, detections, threshold)
        _shap_results.extend(explained)
        _register_shap_atexit()

    return kept


def filter_detections_explained(audio, sr, detections, threshold=0.01):
    """Filter detections with SHAP explainability.

    Args:
        audio: numpy array of audio samples
        sr: sample rate
        detections: list of detection times (seconds)
        threshold: minimum classifier confidence to keep a detection

    Returns:
        List of dicts:
        [{"time": float, "kept": bool, "confidence": float,
          "shap_top5": [{"feature": str, "value": float}]}]
    """
    if not detections:
        return []

    model = _load_model()
    explainer = _get_explainer()

    results = []
    for det_time in detections:
        patch = _extract_mel_patch(audio, sr, float(det_time))
        if patch is None:
            results.append({
                "time": float(det_time),
                "kept": True,
                "confidence": 1.0,
                "shap_top5": [],
            })
            continue

        X = patch.reshape(1, -1)
        X_pca = _transform_to_pca(model, X)
        proba = float(model.named_steps['clf'].predict_proba(X_pca)[0, 1])
        kept = proba >= threshold

        # Compute SHAP values for class 1 (splice)
        shap_values = explainer.shap_values(X_pca)
        if isinstance(shap_values, list):
            # Binary classification: [class0_shap, class1_shap]
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        # Get top 5 features by absolute SHAP value
        # NOTE: Feature names are PCA component indices, not human-readable
        # mel-spectrogram dimensions. This is a known limitation since PCA
        # components are linear combinations of the original 25600 features.
        top5_idx = np.argsort(np.abs(sv))[-5:][::-1]
        shap_top5 = [
            {"feature": f"pca_{i}", "value": float(sv[i])}
            for i in top5_idx
        ]

        results.append({
            "time": float(det_time),
            "kept": kept,
            "confidence": float(proba),
            "shap_top5": shap_top5,
        })

    return results


def _register_shap_atexit():
    """Register atexit handler to write accumulated SHAP results at process exit."""
    global _shap_atexit_registered
    if not _shap_atexit_registered:
        import atexit
        atexit.register(_flush_shap_report)
        _shap_atexit_registered = True


def _flush_shap_report():
    """Write accumulated SHAP results to .omc/classifier/shap_report.json."""
    if not _shap_results:
        return
    report_path = os.path.join(_MODEL_DIR, "shap_report.json")
    _load_model()  # ensure _model_hash is populated
    report = {
        "model_hash": _model_hash,
        "n_detections": len(_shap_results),
        "detections": _shap_results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"SHAP report written to {report_path} ({len(_shap_results)} detections)")
