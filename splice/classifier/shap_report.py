"""Per-detection explanation sidecar for the multi-class GBM.

Always includes the 4 DSP scores in top_features so the forensic context
is preserved even when they are not in the top-k by |SHAP|. Falls back to
`feature_importances_` * |x| when the `shap` package is absent.

Robust-to-classifier-swap: not every classifier exposes
`feature_importances_` (HistGradientBoostingClassifier, ExtraTrees with
bootstrap=False and random splits, calibrated ensembles wrapping opaque
base estimators). Falling through to that attribute unconditionally
raises AttributeError which evaluate.py catches as a per-domain ERROR —
silently collapsing reported combined to the 0.01 GM floor and making
classifier-swap hypotheses look catastrophic when they were actually
producing valid detections. Guarded fallback uses uniform weights when
neither SHAP nor feature_importances_ is available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from autoresearch.logger import get_logger

_HERE = Path(__file__).resolve()


_DSP_PREFIX = "dsp_"


def _shap_values_per_sample(model, X: np.ndarray):
    """Return (shap_matrix, strategy_name). When shap is importable,
    returns per-predicted-class TreeExplainer values; otherwise a
    feature_importance × |scaled x| proxy.
    """
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    try:
        import shap  # type: ignore
        expl = shap.TreeExplainer(clf)
        if hasattr(model, "named_steps"):
            scaler = model.named_steps.get("scaler")
            X_in = scaler.transform(X) if scaler is not None else X
        else:
            X_in = X
        raw = expl.shap_values(X_in)
        if isinstance(raw, list):
            per_class = np.stack(raw, axis=0)  # (C, N, F)
            preds = clf.predict(X_in)
            classes = list(clf.classes_)
            col_idx = np.array([classes.index(int(p)) for p in preds])
            shap_vals = per_class[col_idx, np.arange(len(X_in)), :]
        else:
            shap_vals = raw
        return np.asarray(shap_vals, dtype=np.float64), "shap_tree"
    except ImportError:
        get_logger("shap.report").emit("INFO", "diag.shap.using_importance_fallback",
              cause="shap_not_installed")
    except Exception as e:
        get_logger("shap.report").emit("INFO", "diag.shap.using_importance_fallback",
              cause="tree_explainer_failed", error=str(e))

    try:
        imp = np.asarray(clf.feature_importances_, dtype=np.float64)
        strategy = "importance_weighted_magnitude"
    except AttributeError:
        # Classifier exposes neither a TreeExplainer-compatible booster
        # nor feature_importances_ (HistGradientBoostingClassifier,
        # bootstrap-free ExtraTreesClassifier, some CalibratedClassifierCV
        # wrappings). Use uniform weights so |X| magnitude alone drives
        # the ranking — informative enough for top-k picks and prevents
        # the silent collapse-to-0.01-floor failure.
        get_logger("shap.report").emit("WARN", "diag.shap.no_feature_importances",
              classifier_type=type(clf).__name__,
              fallback="uniform_weights")
        imp = np.ones(X.shape[1], dtype=np.float64)
        strategy = "uniform_weighted_magnitude"
    X_arr = np.asarray(X, dtype=np.float64)
    return np.abs(X_arr) * imp[np.newaxis, :], strategy


def _pick_top_features(
    feature_names: list[str],
    feature_vector: list[float],
    shap_row: np.ndarray,
    top_k: int = 3,
) -> list[dict]:
    """All DSP features first (always), then top_k non-DSP by |shap|."""
    def _row(i: int, name: str) -> dict:
        return {
            "name": name,
            "value": float(feature_vector[i]),
            "shap": float(shap_row[i]),
        }

    dsp_idx = [i for i, n in enumerate(feature_names) if n.startswith(_DSP_PREFIX)]
    non_dsp_idx = [i for i, n in enumerate(feature_names) if not n.startswith(_DSP_PREFIX)]
    non_dsp_idx.sort(key=lambda i: -abs(float(shap_row[i])))
    return [_row(i, feature_names[i]) for i in dsp_idx + non_dsp_idx[:top_k]]


def write_reports(
    detections: list[dict],
    reports_dir: str,
    source_file: str,
    model,
    feature_names: list[str],
) -> int:
    """Write one JSON per detection under reports_dir. Each detection is a
    dict with 'time_sec', 'label', 'probability', 'feature_vector'.
    """
    if not detections:
        return 0
    if not feature_names:
        get_logger("shap.report").emit("WARN", "diag.shap.missing_feature_names",
              source=source_file, hint="retrain to regenerate meta.json")
        return 0

    X = np.asarray([d["feature_vector"] for d in detections], dtype=np.float64)
    if X.shape[1] != len(feature_names):
        get_logger("shap.report").emit("WARN", "diag.shap.feature_length_mismatch",
              source=source_file, n_features=X.shape[1],
              n_names=len(feature_names))
        return 0

    shap_mat, strategy = _shap_values_per_sample(model, X)

    os.makedirs(reports_dir, exist_ok=True)
    stem = Path(source_file).stem if source_file else "file"
    for i, det in enumerate(detections):
        t = float(det["time_sec"])
        label = det["label"]
        report = {
            "file": source_file,
            "time_sec": t,
            "label": label,
            "label_id": int(det.get("label_id", -1)),
            "probability": float(det["probability"]),
            "top_features": _pick_top_features(
                feature_names, det["feature_vector"], shap_mat[i]
            ),
            "shap_strategy": strategy,
        }
        fname = f"{stem}__t{t:.3f}s__{label}.json".replace(os.sep, "_")
        with open(os.path.join(reports_dir, fname), "w") as f:
            json.dump(report, f, indent=2)

    get_logger("shap.report").emit("INFO", "diag.shap.reports_written",
          source=source_file, count=len(detections),
          strategy=strategy, out_dir=reports_dir)
    return len(detections)
