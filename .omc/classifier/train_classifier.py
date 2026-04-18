"""Multi-class (not_splice / hard_cut / crossfade) GBM trained on features.py.

Per spliced file: a 60s chunk centered on the GT splice; positives sampled
at GT + POS_OFFSETS_S (tier 1 → hard_cut, tier 2 → crossfade); negatives at
random positions ≥NEG_MIN_DIST_S away. Per clean file: negatives only, at
the midpoint chunk. GroupKFold is file_id-grouped so positives and their
negatives never leak across folds.

`make_pipeline()` is the GBM hyperparameter knob autoresearch can tune
(n_estimators / max_depth / learning_rate / subsample). After editing,
rerun this script to refresh the on-disk bundle consumed by detector.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_registry import DATASETS
from detector import _build_chunk_context, _diag
from features import FEATURE_NAMES, extract_features

LABEL_NOT_SPLICE = 0
LABEL_HARD_CUT = 1
LABEL_CROSSFADE = 2
LABEL_NAMES = ["not_splice", "hard_cut", "crossfade"]

CHUNK_S = 60.0  # must match detector.ANALYSIS_WINDOW_S
GT_RADIUS_S = 1.0  # positives within ±1s of GT
POS_OFFSETS_S = (0.0, 0.3, 0.6, -0.3, -0.6)
NEG_PER_SPLICED = 3
NEG_PER_CLEAN = 4
NEG_MIN_DIST_S = 2.0  # must be at least this far from any GT
RANDOM_STATE = 42
N_FOLDS = 5

MODEL_OUT = _HERE.parent / "fp_classifier.joblib"
CV_OUT = _HERE.parent / "cv_results.json"
DATASET_META_OUT = _HERE.parent / "training_manifest.json"


# ---------------------------------------------------------------------------
# Sklearn pipeline
# ---------------------------------------------------------------------------


def make_pipeline():
    """Multi-class HistGBM pipeline. Edit the hyperparameters here to
    adjust model capacity or regularization, then rerun the script to
    retrain.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=3,
            max_leaf_nodes=8,
            learning_rate=0.07,
            l2_regularization=1.0,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        )),
    ])


# ---------------------------------------------------------------------------
# Training-set construction
# ---------------------------------------------------------------------------


def _rng(name: str, salt: int = 0) -> np.random.RandomState:
    h = int(hashlib.md5(f"{name}:{salt}".encode()).hexdigest(), 16)
    return np.random.RandomState(h % (2**31))


def _anchor_chunk(audio: np.ndarray, sr: int, center_s: float | None) -> tuple[np.ndarray, float]:
    """Return (chunk, chunk_start_s) of length CHUNK_S centered on center_s
    (or at the file midpoint if center_s is None). Clamped to file bounds.
    """
    W = int(CHUNK_S * sr)
    n = len(audio)
    if center_s is None:
        start = max(0, (n - W) // 2)
    else:
        start = int(center_s * sr) - W // 2
        start = max(0, min(start, max(0, n - W)))
    chunk = audio[start: start + W]
    if len(chunk) < W:
        chunk = np.pad(chunk, (0, W - len(chunk)))
    return chunk, start / sr


def _gt_times_for(info: dict) -> list[float]:
    if not info.get("spliced", False):
        return []
    st = info.get("splice_time_sec")
    if isinstance(st, list):
        return [float(t) for t in st]
    if st is not None:
        return [float(st)]
    return []


def _tier_label(tier: int) -> int:
    if tier == 1:
        return LABEL_HARD_CUT
    if tier == 2:
        return LABEL_CROSSFADE
    return LABEL_NOT_SPLICE


def _iter_training_files():
    for ds in DATASETS:
        gt_path = ds.train_path / "ground_truth.json"
        if not gt_path.exists():
            _diag("WARN", "train", "gt_missing",
                  dataset=ds.id, path=str(gt_path))
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        for name, info in sorted(gt.items()):
            audio_path = ds.train_path / info["path"]
            if not audio_path.exists():
                _diag("INFO", "train", "audio_missing",
                      dataset=ds.id, name=name)
                continue
            yield ds.id, name, str(audio_path), info


def _load_mono(audio_path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def _sample_candidates(
    name: str,
    info: dict,
    chunk_start_s: float,
    chunk_dur_s: float,
) -> list[tuple[float, int]]:
    """Return [(t_local, label), ...] — times in chunk-local coordinates.

    For spliced files: positives at GT + POS_OFFSETS_S, negatives at random
    positions ≥ NEG_MIN_DIST_S from any GT. For clean files: only random
    negatives.
    """
    gt_file_times = _gt_times_for(info)
    tier = int(info.get("tier", 0))
    candidates: list[tuple[float, int]] = []

    # Positives
    for gt_t in gt_file_times:
        gt_local = gt_t - chunk_start_s
        if gt_local < 0 or gt_local > chunk_dur_s:
            continue
        for off in POS_OFFSETS_S:
            t_local = gt_local + off
            if 0.5 <= t_local <= chunk_dur_s - 0.5:
                candidates.append((t_local, _tier_label(tier)))

    # Negatives
    n_neg = NEG_PER_SPLICED if info.get("spliced", False) else NEG_PER_CLEAN
    rng = _rng(name)
    gt_local_list = [g - chunk_start_s for g in gt_file_times]
    tries = 0
    collected = 0
    while collected < n_neg and tries < n_neg * 20:
        tries += 1
        t_local = float(rng.uniform(0.8, chunk_dur_s - 0.8))
        if all(abs(t_local - g) >= NEG_MIN_DIST_S for g in gt_local_list):
            candidates.append((t_local, LABEL_NOT_SPLICE))
            collected += 1

    return candidates


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Construct (X, y, groups, manifest) where X is (N, 75).

    `groups` is a file-level id (shared across all candidates from one file)
    so GroupKFold splits whole files into train/test.
    """
    X_rows: list[list[float]] = []
    y: list[int] = []
    groups: list[int] = []
    manifest: list[dict] = []

    file_keys: dict[str, int] = {}
    next_fid = 0

    t0 = time.time()
    n_files = 0
    n_skip_short = 0

    for ds_id, name, audio_path, info in _iter_training_files():
        key = f"{ds_id}/{name}"
        if key not in file_keys:
            file_keys[key] = next_fid
            next_fid += 1
        fid = file_keys[key]

        try:
            audio, sr = _load_mono(audio_path)
        except Exception as e:
            _diag("WARN", "train", "load_failed",
                  dataset=ds_id, name=name, error=str(e))
            continue

        if len(audio) < int(CHUNK_S * sr * 0.5):
            n_skip_short += 1
            _diag("INFO", "train", "audio_too_short",
                  dataset=ds_id, name=name, dur=f"{len(audio)/sr:.1f}")
            continue

        gt_times = _gt_times_for(info)
        center = gt_times[0] if gt_times else None
        chunk, chunk_start_s = _anchor_chunk(audio, sr, center)
        chunk_dur_s = len(chunk) / sr

        candidates = _sample_candidates(name, info, chunk_start_s, chunk_dur_s)
        if not candidates:
            continue

        ctx = _build_chunk_context(chunk, sr)

        for t_local, label in candidates:
            feats = extract_features(chunk, sr, t_local, chunk_ctx=ctx)
            X_rows.append([feats[k] for k in FEATURE_NAMES])
            y.append(label)
            groups.append(fid)
            manifest.append({
                "dataset": ds_id,
                "file": name,
                "t_local": round(t_local, 3),
                "chunk_start_s": round(chunk_start_s, 3),
                "label": int(label),
                "tier": int(info.get("tier", 0)),
                "file_id": fid,
            })

        n_files += 1
        if n_files % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_files} files] rows={len(X_rows)} elapsed={elapsed:.1f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"Training-set built: files={n_files} skipped_short={n_skip_short} "
          f"rows={len(X_rows)} elapsed={elapsed:.1f}s")
    if not X_rows:
        raise RuntimeError(
            f"Training-set is empty. Processed {n_files} file(s), "
            f"skipped {n_skip_short} as too short. Check dataset paths in "
            f"dataset_registry.py and that CHUNK_S ({CHUNK_S}s) is not "
            f"larger than most files."
        )
    X = np.asarray(X_rows, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    g_arr = np.asarray(groups, dtype=np.int64)
    return X, y_arr, g_arr, manifest


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def _class_counts(y: np.ndarray) -> dict:
    return {str(int(c)): int(n) for c, n in zip(*np.unique(y, return_counts=True))}


def train() -> dict:
    print("=== Building training set ===", flush=True)
    X, y, groups, manifest = build_dataset()
    counts = _class_counts(y)
    print(f"X.shape={X.shape} class counts: {counts}")
    if len(X) == 0:
        raise RuntimeError("Training set is empty — check dataset paths.")

    with open(DATASET_META_OUT, "w") as f:
        json.dump({
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "feature_names": FEATURE_NAMES,
            "label_names": LABEL_NAMES,
            "class_counts": counts,
            "rows": manifest,
        }, f, indent=2)

    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        raise RuntimeError(f"Need at least 2 groups for CV, got {n_groups}.")
    n_splits = min(N_FOLDS, n_groups)

    oof_pred = np.full(len(y), -1, dtype=int)
    fold_results = []
    gkf = GroupKFold(n_splits=n_splits)

    print(f"\n=== GroupKFold n_splits={n_splits} ===", flush=True)
    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = make_pipeline()
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        oof_pred[te] = pred
        f1_w = float(f1_score(y[te], pred, average="weighted", zero_division=0))
        f1_m = float(f1_score(y[te], pred, average="macro", zero_division=0))
        fold_results.append({
            "fold": fold_idx + 1,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "f1_weighted": f1_w,
            "f1_macro": f1_m,
            "classes_test": sorted(set(int(v) for v in y[te])),
        })
        print(f"  Fold {fold_idx+1}: n_train={len(tr)} n_test={len(te)} "
              f"f1_w={f1_w:.4f} f1_m={f1_m:.4f}")

    valid = oof_pred >= 0
    y_valid = y[valid]
    pred_valid = oof_pred[valid]
    oof_labels = sorted(set(int(v) for v in y_valid))
    oof_f1_w = float(f1_score(y_valid, pred_valid, average="weighted", zero_division=0))
    oof_f1_m = float(f1_score(y_valid, pred_valid, average="macro", zero_division=0))
    per_class = f1_score(y_valid, pred_valid, labels=oof_labels,
                         average=None, zero_division=0).tolist()
    oof_f1_per = {int(c): float(v) for c, v in zip(oof_labels, per_class)}

    print("\n=== OOF metrics ===")
    print(f"  weighted F1: {oof_f1_w:.4f}")
    print(f"  macro F1:    {oof_f1_m:.4f}")
    print(f"  per-class F1: {oof_f1_per}")
    print(classification_report(
        y_valid, pred_valid, zero_division=0,
        target_names=[LABEL_NAMES[c] for c in oof_labels],
    ))

    cv_results = {
        "n_splits": n_splits,
        "n_groups": n_groups,
        "n_samples": int(len(y)),
        "class_counts": counts,
        "folds": fold_results,
        "oof_f1_weighted": oof_f1_w,
        "oof_f1_macro": oof_f1_m,
        "oof_f1_per_class": oof_f1_per,
    }
    with open(CV_OUT, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"Wrote {CV_OUT}")

    print("\n=== Final fit on all data ===", flush=True)
    final = make_pipeline()
    final.fit(X, y)
    joblib.dump(final, MODEL_OUT)
    meta_out = MODEL_OUT.with_suffix(".meta.json")

    # US-505: record the git-blob sha of features.py at train time so
    # the wrapper can detect drift and auto-retrain before evaluate.py.
    import subprocess as _sp
    _features_path = _HERE.parent.parent / "features.py"
    try:
        _features_sha = _sp.check_output(
            ["git", "hash-object", str(_features_path)],
            text=True, stderr=_sp.DEVNULL, timeout=2,
        ).strip()
    except Exception:
        _features_sha = None
    import datetime as _dt
    with open(meta_out, "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "label_names": LABEL_NAMES,
            "classes_": [int(c) for c in final.named_steps["clf"].classes_.tolist()],
            "trained_on": {
                "datasets": [ds.id for ds in DATASETS],
                "n_samples": int(len(y)),
                "class_counts": counts,
            },
            "features_py_sha": _features_sha,
            "training_timestamp": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, f, indent=2)
    print(f"Saved model → {MODEL_OUT}")
    print(f"Saved metadata → {meta_out}")

    return cv_results


if __name__ == "__main__":
    train()
