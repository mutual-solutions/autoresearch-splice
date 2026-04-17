"""ML evaluation helper for evaluate.py --with-classifier.

The detector now runs the multi-class GBM in-loop and emits per-detection
metadata via detector.get_detection_meta(). This module just writes one
SHAP sidecar per detection and echoes combined_full = combined.
"""

from __future__ import annotations

import os
import sys

import soundfile as sf

_proj = os.path.dirname(os.path.abspath(__file__))
_classifier_dir = os.path.join(_proj, ".omc", "classifier")
if _classifier_dir not in sys.path:
    sys.path.insert(0, _classifier_dir)

from detector import _diag, _load_gbm_bundle, get_detection_meta
from shap_report import write_reports


def evaluate_with_classifier(dsp_results, data_dir):
    """Export SHAP sidecars for the GBM-filtered detections and echo
    combined_full. Raises if the multi-class bundle isn't loaded — the
    legacy binary OOF path was ripped in favor of the detector-internal
    GBM scan.
    """
    dsp_combined = dsp_results["combined"]
    print(f"combined_dsp: {dsp_combined:.6f}")
    print(f"dsp_clean_fp: {dsp_results['clean_fp']}")

    bundle = _load_gbm_bundle()
    if bundle is None:
        _diag("ERROR", "ml.eval", "bundle_missing",
              hint="retrain via .omc/classifier/train_classifier.py")
        raise RuntimeError("Multi-class GBM bundle missing or invalid; "
                           "retrain via .omc/classifier/train_classifier.py")

    model = bundle["model"]
    feature_names = bundle.get("feature_names")
    reports_root = os.path.join(_proj, "reports",
                                os.path.basename(os.path.normpath(data_dir)))
    os.makedirs(reports_root, exist_ok=True)

    total_reports = 0
    for pf in dsp_results["per_file"]:
        audio, sr = sf.read(pf["path"], dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        meta = get_detection_meta(audio, sr)
        if not meta:
            continue
        total_reports += write_reports(
            meta, reports_root, pf["name"], model, feature_names,
        )

    combined_full = dsp_combined
    print("classifier_quality: OK_GBM")
    print("classifier_mode: gbm_in_detector")
    print(f"reports_written: {total_reports}")
    print(f"combined_full: {combined_full:.6f}")

    return {
        "classifier_quality": "OK_GBM",
        "combined_full": combined_full,
        "reports_written": total_reports,
        "classifier_mode": "gbm_in_detector",
    }
