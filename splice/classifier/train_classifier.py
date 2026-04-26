"""3-class GBM trained on korean-iter1 per-conversation boundaries.

Training data: ``data/eval/korean_iter1/train/``
  - Per-conversation audio: ``{conv_id}.opus``
  - Per-conversation boundaries: ``{conv_id}.json``  (``{"boundaries": [...]}``
    where each boundary is ``{"time_s": float, "label": "cross_voice"|"same_voice_edit"}``)
  - Aggregate ground truth: ``ground_truth.json`` (``{conv_id: [{"time_s":, "label":}, ...]}``)

Classes: cross_voice=0, no_splice=1, same_voice_edit=2  (alphabetical).

Negatives: ~2x positives, sampled from non-boundary frames at least
NEG_MIN_DIST_S away from any boundary.  Deterministic per-conv seed.

``make_pipeline()`` is the GBM hyperparameter knob autoresearch can tune.
After editing, rerun this script to refresh the on-disk bundle consumed
by detector.py.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess as _sp
import time
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]

from splice.detector import _build_chunk_context
from splice.features import FEATURE_NAMES, extract_features
from autoresearch.logger import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 3-class labels — alphabetical order so classes_ is stable across sklearn fits
CLASS_NAMES: list[str] = ["cross_voice", "no_splice", "same_voice_edit"]

# Label encoder is shared: fit once on CLASS_NAMES so encoding is deterministic
_LE = LabelEncoder()
_LE.fit(CLASS_NAMES)

LABEL_CROSS_VOICE: int = int(_LE.transform(["cross_voice"])[0])    # 0
LABEL_NO_SPLICE: int = int(_LE.transform(["no_splice"])[0])        # 1
LABEL_SAME_VOICE_EDIT: int = int(_LE.transform(["same_voice_edit"])[0])  # 2

TRAIN_DIR = _ROOT / "data" / "eval" / "korean_iter1" / "train"

# Feature-window half-width centred on each boundary (seconds of audio)
WINDOW_HALF_S = 30.0  # 60s total window — matches existing CHUNK_S

NEG_MIN_DIST_S = 1.0   # negatives must be >= this far from any boundary
NEG_RATIO = 2.0        # negatives ≈ 2x positives
NEG_MAX_TRIES_MULT = 30  # tries = n_neg * this before giving up

N_FOLDS = 5
RANDOM_STATE = 42

MODEL_OUT = _HERE.parent / "fp_classifier.joblib"
META_OUT = MODEL_OUT.with_suffix(".meta.json")

# ---------------------------------------------------------------------------
# sklearn pipeline
# ---------------------------------------------------------------------------


def make_pipeline() -> Pipeline:
    """3-class HistGBM pipeline.  Edit hyperparameters here and rerun to retrain."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            max_leaf_nodes=32,
            learning_rate=0.07,
            l2_regularization=2.0,
            min_samples_leaf=40,
            class_weight={0: 1.0, 1: 1.0, 2: 2.0},
            loss="log_loss",          # multiclass-compatible default
            random_state=RANDOM_STATE,
        )),
    ])


# ---------------------------------------------------------------------------
# Deterministic RNG
# ---------------------------------------------------------------------------


def _rng(seed_str: str, salt: int = 0) -> np.random.RandomState:
    h = int(hashlib.md5(f"{seed_str}:{salt}".encode()).hexdigest(), 16)
    return np.random.RandomState(h % (2**31))


# ---------------------------------------------------------------------------
# Per-conversation data loading
# ---------------------------------------------------------------------------


def _load_conv_audio(conv_id: str) -> tuple[np.ndarray, int]:
    """Load opus conversation audio at 44100 Hz mono."""
    path = TRAIN_DIR / f"{conv_id}.opus"
    audio, sr = librosa.load(str(path), sr=44100, mono=True)
    return audio.astype(np.float32), int(sr)


def _load_conv_boundaries(conv_id: str) -> list[dict[str, Any]]:
    """Load per-conversation boundary list from {conv_id}.json.

    Supports both the full per-conv schema (``{"boundaries": [...]}``
    with optional top-level fields) and the aggregate ground_truth.json
    array format (``[{"time_s":, "label":}, ...]``).
    """
    path = TRAIN_DIR / f"{conv_id}.json"
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    return data.get("boundaries", [])


def _extract_window(
    audio: np.ndarray,
    sr: int,
    center_s: float,
) -> tuple[np.ndarray, float, float]:
    """Return (window_chunk, window_start_s, window_dur_s).

    Window is WINDOW_HALF_S on each side of center_s, clamped to file bounds.
    """
    W = int(WINDOW_HALF_S * 2 * sr)
    n = len(audio)
    start = int(center_s * sr) - W // 2
    start = max(0, min(start, max(0, n - W)))
    chunk = audio[start: start + W]
    if len(chunk) < W:
        chunk = np.pad(chunk, (0, W - len(chunk)))
    window_start_s = start / sr
    window_dur_s = len(chunk) / sr
    return chunk, window_start_s, window_dur_s


def _sample_negatives(
    conv_id: str,
    audio_dur_s: float,
    boundary_times: list[float],
    n_neg: int,
) -> list[float]:
    """Sample n_neg negative times from audio that are >= NEG_MIN_DIST_S from
    every boundary.  Returns times in file-absolute coordinates.

    Deterministic: seeded per conv_id.
    """
    rng = _rng(conv_id, salt=0)
    max_tries = max(n_neg * NEG_MAX_TRIES_MULT, 200)
    results: list[float] = []
    tries = 0
    while len(results) < n_neg and tries < max_tries:
        tries += 1
        t = float(rng.uniform(0.5, max(audio_dur_s - 0.5, 0.5)))
        if all(abs(t - b) >= NEG_MIN_DIST_S for b in boundary_times):
            results.append(t)
    return results


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _iter_conversations() -> list[str]:
    """Return sorted list of conversation IDs from aggregate ground_truth.json
    or by globbing for per-conv JSON files in TRAIN_DIR.
    """
    gt_agg = TRAIN_DIR / "ground_truth.json"
    if gt_agg.exists():
        with open(gt_agg) as fh:
            data = json.load(fh)
        return sorted(data.keys())
    # Fallback: glob for per-conv json files
    return sorted(p.stem for p in TRAIN_DIR.glob("*.json"))


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Construct (X, y, groups, manifest) for the korean-iter1 train split.

    y values are integer-encoded via _LE (alphabetical: cross_voice=0,
    no_splice=1, same_voice_edit=2).

    ``groups`` is a per-conversation integer id so GroupKFold keeps all
    samples from one conversation in the same fold.
    """
    log = get_logger("classifier.train")

    X_rows: list[list[float]] = []
    y_list: list[int] = []
    groups: list[int] = []
    manifest: list[dict] = []

    conv_ids = _iter_conversations()
    n_files = 0
    n_skip = 0
    per_class_count: dict[str, int] = {c: 0 for c in CLASS_NAMES}

    t0 = time.time()

    for fid, conv_id in enumerate(conv_ids):
        # Load audio
        try:
            audio, sr = _load_conv_audio(conv_id)
        except FileNotFoundError:
            log.emit("WARN", "diag.train.audio_missing", conv_id=conv_id)
            n_skip += 1
            continue
        except Exception as e:
            log.emit("WARN", "diag.train.load_failed", conv_id=conv_id, error=str(e))
            n_skip += 1
            continue

        # Load boundaries
        try:
            boundaries = _load_conv_boundaries(conv_id)
        except FileNotFoundError:
            log.emit("WARN", "diag.train.gt_missing", conv_id=conv_id)
            n_skip += 1
            continue
        except Exception as e:
            log.emit("WARN", "diag.train.gt_parse_failed", conv_id=conv_id, error=str(e))
            n_skip += 1
            continue

        audio_dur_s = len(audio) / sr
        boundary_times = [float(b["time_s"]) for b in boundaries]

        # --- Positives ---
        candidates: list[tuple[float, str]] = []
        for b in boundaries:
            label = b.get("label", "cross_voice")
            if label not in CLASS_NAMES or label == "no_splice":
                # no_splice should not appear in boundary lists; skip gracefully
                continue
            candidates.append((float(b["time_s"]), label))

        # --- Negatives ---
        n_pos = len(candidates)
        n_neg = max(1, round(n_pos * NEG_RATIO))
        neg_times = _sample_negatives(conv_id, audio_dur_s, boundary_times, n_neg)
        for t in neg_times:
            candidates.append((t, "no_splice"))

        if not candidates:
            n_skip += 1
            continue

        # --- Feature extraction ---
        # OPTIMIZATION: build the DSP chunk context ONCE per conversation on
        # the whole audio, then query features at each candidate's absolute
        # time. The original per-candidate context build was redundant (window
        # half-width was 30s, conversations average 48s — so every "window"
        # was essentially the whole audio anyway, rebuilt 30× per conv at
        # ~1.7s/build → ~30 hours for the full corpus). Per-conv build is
        # ~30× faster.
        try:
            conv_ctx = _build_chunk_context(audio, sr)
        except Exception as e:
            log.emit("WARN", "diag.train.ctx_failed",
                     conv_id=conv_id, error=str(e))
            n_skip += 1
            continue
        audio_dur = len(audio) / sr

        for t_abs, label_str in candidates:
            # Clamp absolute time to valid query range within the audio.
            t_query = float(np.clip(t_abs, 0.5, max(audio_dur - 0.5, 0.5)))
            try:
                feats = extract_features(audio, sr, t_query, chunk_ctx=conv_ctx)
            except Exception as e:
                log.emit("WARN", "diag.train.feature_failed",
                         conv_id=conv_id, t_abs=round(t_abs, 3), error=str(e))
                continue

            feat_vec = [feats[k] for k in FEATURE_NAMES]
            y_int = int(_LE.transform([label_str])[0])

            X_rows.append(feat_vec)
            y_list.append(y_int)
            groups.append(fid)
            per_class_count[label_str] = per_class_count.get(label_str, 0) + 1
            manifest.append({
                "conv_id": conv_id,
                "t_abs": round(t_abs, 3),
                "t_query": round(t_query, 3),
                "label": label_str,
                "y": y_int,
                "conv_fid": fid,
            })

        n_files += 1
        if n_files % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_files} convs] rows={len(X_rows)} elapsed={elapsed:.1f}s",
                  flush=True)

    elapsed = time.time() - t0
    n_pos_total = sum(v for k, v in per_class_count.items() if k != "no_splice")
    n_neg_total = per_class_count.get("no_splice", 0)
    print(
        f"Dataset built: convs={n_files} skipped={n_skip} rows={len(X_rows)} "
        f"positives={n_pos_total} negatives={n_neg_total} elapsed={elapsed:.1f}s",
        flush=True,
    )
    log.emit("INFO", "classifier.train.data_loaded",
             n_files=n_files, n_positive=n_pos_total, n_negative=n_neg_total,
             per_class_count=json.dumps(per_class_count))

    if not X_rows:
        raise RuntimeError(
            f"Training set is empty after processing {n_files} conversations "
            f"(skipped {n_skip}).  Check that {TRAIN_DIR} is populated."
        )

    X = np.asarray(X_rows, dtype=np.float64)
    y_arr = np.asarray(y_list, dtype=np.int64)
    g_arr = np.asarray(groups, dtype=np.int64)
    return X, y_arr, g_arr, manifest


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


def _hash_object(p: Path) -> str | None:
    try:
        return _sp.check_output(
            ["git", "hash-object", str(p)],
            text=True, stderr=_sp.DEVNULL, timeout=2,
        ).strip()
    except Exception:
        return None


def train() -> dict:
    log = get_logger("classifier.train")
    log.emit("INFO", "classifier.train.start")
    print("=== Building training set ===", flush=True)

    try:
        X, y, groups, manifest = build_dataset()
    except Exception as e:
        log.emit("ERROR", "classifier.retrain.failed", error=str(e))
        raise

    n_samples, n_features = X.shape
    print(f"X.shape={X.shape}  classes: {np.unique(y, return_counts=True)}")

    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        err = f"Need at least 2 conversation groups for CV, got {n_groups}."
        log.emit("ERROR", "classifier.retrain.failed", error=err)
        raise RuntimeError(err)

    n_splits = min(N_FOLDS, n_groups)

    # ---- OOF cross-validation ----
    oof_pred = np.full(len(y), -1, dtype=int)
    fold_results = []
    gkf = GroupKFold(n_splits=n_splits)

    print(f"\n=== GroupKFold n_splits={n_splits} ===", flush=True)
    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = make_pipeline()
        try:
            pipe.fit(X[tr], y[tr])
        except Exception as e:
            log.emit("ERROR", "classifier.retrain.failed", error=f"fold {fold_idx+1}: {e}")
            raise
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
        })
        log.emit("INFO", "classifier.train.oof_fold",
                 fold=fold_idx + 1, train_size=len(tr),
                 val_size=len(te), val_f1=round(f1_m, 4))
        print(f"  Fold {fold_idx+1}: n_train={len(tr)} n_test={len(te)} "
              f"f1_w={f1_w:.4f} f1_m={f1_m:.4f}", flush=True)

    valid = oof_pred >= 0
    y_valid = y[valid]
    pred_valid = oof_pred[valid]

    oof_labels_int = sorted(set(int(v) for v in y_valid))
    oof_labels_str = [CLASS_NAMES[i] for i in oof_labels_int]

    oof_f1_w = float(f1_score(y_valid, pred_valid, average="weighted", zero_division=0))
    oof_f1_m = float(f1_score(y_valid, pred_valid, average="macro", zero_division=0))
    per_class_f1_arr = f1_score(
        y_valid, pred_valid,
        labels=oof_labels_int, average=None, zero_division=0,
    ).tolist()
    oof_f1_per_class = {
        CLASS_NAMES[c]: float(v)
        for c, v in zip(oof_labels_int, per_class_f1_arr)
    }

    print("\n=== OOF metrics ===")
    print(f"  weighted F1: {oof_f1_w:.4f}")
    print(f"  macro F1:    {oof_f1_m:.4f}")
    print(f"  per-class:   {oof_f1_per_class}")
    print(classification_report(y_valid, pred_valid, zero_division=0,
                                target_names=oof_labels_str))

    # ---- Final fit on all data ----
    print("\n=== Final fit on all data ===", flush=True)
    final = make_pipeline()
    try:
        final.fit(X, y)
    except Exception as e:
        log.emit("ERROR", "classifier.retrain.failed", error=str(e))
        raise

    joblib.dump(final, MODEL_OUT)

    # ---- Write meta ----
    clf = final.named_steps["clf"]
    model_size_kb = int(MODEL_OUT.stat().st_size / 1024)

    # Build hyperparameter snapshot from make_pipeline's clf step params
    hyperparams = {k: v for k, v in clf.get_params().items()}

    classes_str = sorted(CLASS_NAMES)  # alphabetical
    assert classes_str == CLASS_NAMES, f"CLASS_NAMES must be alphabetical: {CLASS_NAMES}"

    # Count per-class in final dataset
    class_counts_final = {
        name: int(np.sum(y == int(_LE.transform([name])[0])))
        for name in CLASS_NAMES
    }

    meta = {
        "version": "korean_iter1_v1",
        "trained_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_train_files": n_groups,
        "n_train_samples": n_samples,
        "classes_": classes_str,
        "n_features": n_features,
        "feature_names": FEATURE_NAMES,
        "hyperparameters": hyperparams,
        "oof_metrics": {
            "f1_macro": round(oof_f1_m, 6),
            "f1_weighted": round(oof_f1_w, 6),
            "f1_per_class": {
                k: round(v, 6) for k, v in oof_f1_per_class.items()
            },
        },
        "class_counts": class_counts_final,
        "folds": fold_results,
        # git shas for drift detection (US-505)
        "features_py_sha": _hash_object(_HERE.parent.parent / "features.py"),
        "train_classifier_py_sha": _hash_object(_HERE),
        # voice_holdout is populated downstream by the operator if needed
        "voice_holdout": {"test": [], "eval": [], "train": []},
    }

    with open(META_OUT, "w") as fh:
        json.dump(meta, fh, indent=2)

    log.emit("INFO", "classifier.train.complete",
             f1_macro=round(oof_f1_m, 4),
             classes_count=len(CLASS_NAMES),
             model_size_kb=model_size_kb)

    print(f"Saved model  → {MODEL_OUT}")
    print(f"Saved meta   → {META_OUT}")
    print(f"n_classes=3  classes={classes_str}", flush=True)

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the 3-class (cross_voice / no_splice / same_voice_edit) "
                    "GBM on the korean-iter1 train split."
    )
    parser.add_argument(
        "--train-dir",
        default=str(TRAIN_DIR),
        help=f"Override training data directory (default: {TRAIN_DIR})",
    )
    args = parser.parse_args()

    if args.train_dir != str(TRAIN_DIR):
        import sys
        print(f"[WARN] --train-dir override: {args.train_dir}", file=sys.stderr)
        TRAIN_DIR = Path(args.train_dir)

    train()
