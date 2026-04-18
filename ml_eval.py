"""Per-detection SHAP sidecar exporter.

The detector runs the multi-class GBM in-loop and stores per-detection
metadata via detector.get_detection_meta(). This module writes one JSON
per detection under reports/<git_sha>/<data_dir>/ so each iteration of
the autoresearch loop leaves an immutable forensic trail.
"""

from __future__ import annotations

import os
import subprocess
import sys

import soundfile as sf

_proj = os.path.dirname(os.path.abspath(__file__))
_classifier_dir = os.path.join(_proj, ".omc", "classifier")
if _classifier_dir not in sys.path:
    sys.path.insert(0, _classifier_dir)

from detector import _load_gbm_bundle, get_detection_meta
from shap_report import write_reports

# US-515 phase 1: unified structured logger.
import os as _us515_os
import sys as _us515_sys
_us515_sys.path.insert(0, _us515_os.path.join(_us515_os.path.dirname(_us515_os.path.abspath(__file__)), ".omc", "coordination"))
from logger import get_logger  # noqa: E402


def _git_sha_short() -> str:
    """Best-effort short git sha; returns 'unknown' on failure so reports
    still land somewhere instead of raising.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_proj, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def export_shap_reports(dsp_results, data_dir) -> dict:
    """Write SHAP sidecars for every GBM detection in `dsp_results`.

    Output layout: reports/<git_sha>/<data_dir_basename>/<file>__t<t>s__<label>.json
    Keying by git sha preserves per-iteration history across autoresearch
    commits so feature-drift / surprise-keep forensic passes are possible
    long after the iteration lands.
    """
    bundle = _load_gbm_bundle()
    if bundle is None:
        get_logger("ml.eval").emit("ERROR", "diag.ml.eval.bundle_missing",
              hint="retrain via .omc/classifier/train_classifier.py")
        raise RuntimeError("Multi-class GBM bundle missing or invalid; "
                           "retrain via .omc/classifier/train_classifier.py")

    model = bundle["model"]
    feature_names = bundle.get("feature_names")
    sha = _git_sha_short()
    reports_root = os.path.join(
        _proj, "reports", sha,
        os.path.basename(os.path.normpath(data_dir)),
    )
    os.makedirs(reports_root, exist_ok=True)

    total_reports = 0
    for pf in dsp_results.get("per_file", []):
        audio, sr = sf.read(pf["path"], dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        meta = get_detection_meta(audio, sr)
        if not meta:
            continue
        total_reports += write_reports(
            meta, reports_root, pf["name"], model, feature_names,
        )

    print(f"shap_sha: {sha}")
    print(f"shap_out_dir: {reports_root}")
    print(f"reports_written: {total_reports}")

    return {
        "sha": sha,
        "reports_written": total_reports,
        "reports_dir": reports_root,
    }
