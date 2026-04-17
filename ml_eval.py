"""ML evaluation helper for evaluate.py --with-classifier.

Two modes: GBM-in-detector (just writes SHAP sidecars and echoes combined),
and legacy binary OOF (patches_combined/ + eval-set patches → GroupKFold
OOF filter over DSP detections). The mode is picked by whether
detector._load_gbm_bundle() returns a multi-class bundle.
"""

import json
import os
import sys

import numpy as np
import soundfile as sf

# Add classifier module paths
_proj = os.path.dirname(os.path.abspath(__file__))
_classifier_dir = os.path.join(_proj, ".omc", "classifier")
_coord_dir = os.path.join(_proj, ".omc", "coordination")
for p in [_classifier_dir, _coord_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate import match_detections, TOLERANCE_S
from generate_patches import extract_mel_patch, is_tp
from train_classifier import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
import joblib

# Shared diagnostic emitter (structured DIAG lines to stderr, filtered by OMC_DIAG_LEVEL)
from detector import _diag

# Import tunable params from ml_config (agent-editable)
import hashlib as _hashlib

from ml_config import (
    DSP_FP_BOUND, N_FOLDS, OOF_THRESHOLD, PCA_COMPONENTS,
    USE_PREGENERATED_PATCHES, PATCH_HALF_S, RANDOM_STATE,
)


def _stable_hash(s: str) -> int:
    """Deterministic hash across Python processes (hash() is randomized)."""
    return int(_hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


def _load_pregenerated_patches():
    """Load pre-generated patches from patches_combined/ as extra training data."""
    combined_dir = os.path.join(_classifier_dir, "patches_combined")
    patches_path = os.path.join(combined_dir, "patches.npy")
    labels_path = os.path.join(combined_dir, "labels.npy")
    manifest_path = os.path.join(combined_dir, "manifest.json")

    if not os.path.exists(patches_path):
        print("pregenerated_patches: NOT_FOUND")
        _diag("WARN", "ml.pregenerated", "patches_missing", path=patches_path)
        return None, None, None

    patches = np.load(patches_path)
    labels = np.load(labels_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build file IDs from manifest
    filenames = [entry["file"] for entry in manifest]
    unique_files = sorted(set(filenames))
    file_to_id = {name: idx for idx, name in enumerate(unique_files)}
    file_ids = np.array([file_to_id[name] for name in filenames])

    print(f"pregenerated_patches: {len(patches)} ({int(labels.sum())} pos, {int(len(labels) - labels.sum())} neg)")
    return patches, labels, file_ids


# Speech-splice datasets — source of truth in dataset_registry.DATASETS.
from dataset_registry import DATASETS


def _load_speech_splice_patches(splice_dir: str, tag: str):
    """Generate mel patches from a speech-splice dataset for classifier training.

    Runs the current detector on every file, extracts patches at detections
    + missed GTs + random clean negatives. No caching — patches must reflect
    the live detector state or they poison training.

    Returns (patches, labels, file_ids) or (None, None, None) if the dataset
    directory is missing or produces zero patches. Emits DIAG lines via the
    `ml.<tag>` component label so each language's events are traceable.
    """
    gt_path = os.path.join(splice_dir, "ground_truth.json")
    component = f"ml.{tag}"

    if not os.path.exists(gt_path):
        print(f"{tag}_splice_patches: NOT_FOUND (no ground_truth.json)")
        _diag("WARN", component, "gt_missing", path=gt_path)
        return None, None, None

    from detector import detect_splices

    with open(gt_path) as f:
        gt = json.load(f)

    all_patches = []
    all_labels = []
    all_file_ids = []
    file_to_id = {}
    next_id = 0
    n_missing_files = 0
    n_patch_none = 0

    for name, info in sorted(gt.items()):
        fpath = os.path.join(splice_dir, info["path"])
        if not os.path.exists(fpath):
            n_missing_files += 1
            _diag("INFO", component, "file_missing", name=name)
            continue

        if name not in file_to_id:
            file_to_id[name] = next_id
            next_id += 1
        fid = file_to_id[name]

        audio, sr = sf.read(fpath, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        duration = len(audio) / sr

        gt_time = info.get("splice_time_sec")
        gt_times = [gt_time] if gt_time else []
        det_times = detect_splices(audio, sr)

        for det_t in det_times:
            patch = extract_mel_patch(audio, sr, det_t)
            if patch is None:
                n_patch_none += 1
                _diag("INFO", component, "patch_extraction_failed",
                      file=name, at_sec=f"{det_t:.3f}")
                continue
            label = 1 if is_tp(det_t, gt_times) else 0
            all_patches.append(patch)
            all_labels.append(label)
            all_file_ids.append(fid)

        if info.get("spliced") and gt_time:
            already_covered = any(abs(gt_time - d) < TOLERANCE_S for d in det_times)
            if not already_covered:
                patch = extract_mel_patch(audio, sr, gt_time)
                if patch is not None:
                    all_patches.append(patch)
                    all_labels.append(1)
                    all_file_ids.append(fid)

        rng = np.random.RandomState(_stable_hash(name) + 100)
        if not info.get("spliced"):
            for _ in range(3):
                t = rng.uniform(PATCH_HALF_S, max(PATCH_HALF_S + 0.1, duration - PATCH_HALF_S))
                patch = extract_mel_patch(audio, sr, t)
                if patch is not None:
                    all_patches.append(patch)
                    all_labels.append(0)
                    all_file_ids.append(fid)

    if not all_patches:
        print(f"{tag}_splice_patches: NO_PATCHES")
        _diag("WARN", component, "no_patches_generated",
              missing_files=n_missing_files)
        return None, None, None

    if n_missing_files or n_patch_none:
        _diag("INFO", component, "extraction_summary",
              missing_files=n_missing_files, patch_none=n_patch_none,
              patches=len(all_patches))

    patches = np.array(all_patches)
    labels = np.array(all_labels, dtype=int)
    file_ids = np.array(all_file_ids)
    print(f"{tag}_splice_patches: {len(patches)} generated "
          f"({int(labels.sum())} pos, {int(len(labels) - labels.sum())} neg)")
    return patches, labels, file_ids


def _gbm_mode_reports(dsp_results, data_dir, bundle):
    """SHAP sidecar mode. The detector already filtered via predict_proba,
    so combined_full == dsp_combined; we just write one report per detection.
    """
    from detector import get_detection_meta
    from shap_report import write_reports

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

    combined_full = dsp_results["combined"]
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


def evaluate_with_classifier(dsp_results, data_dir):
    """Run classifier pipeline on DSP results, return ML-augmented metrics.

    Two modes:
      - GBM mode (US-305/307): when a multi-class bundle is loaded by
        `detector._load_gbm_bundle()`, the detections in `dsp_results`
        are already GBM-filtered. This function writes per-detection
        SHAP reports and reports combined_full = combined (no re-filter).
      - Legacy OOF mode: when the bundle is missing or the old binary
        Pipeline is present, fall back to the original OOF binary filter
        over pre-generated + eval-set patches.
    """
    dsp_clean_fp = dsp_results["clean_fp"]
    dsp_combined = dsp_results["combined"]
    per_file = dsp_results["per_file"]
    clean_files_count = dsp_results["clean_files"]

    print(f"combined_dsp: {dsp_combined:.6f}")
    print(f"dsp_clean_fp: {dsp_clean_fp}")

    # --- GBM short-circuit: detector already filtered via predict_proba ---
    from detector import _load_gbm_bundle, get_detection_meta
    bundle = _load_gbm_bundle()
    if bundle is not None:
        return _gbm_mode_reports(dsp_results, data_dir, bundle)

    # --- FP bound check ---
    if dsp_clean_fp > DSP_FP_BOUND:
        print("dsp_fp_bound: EXCEEDED")
        _diag("ERROR", "ml.eval", "dsp_fp_bound_exceeded",
              dsp_clean_fp=dsp_clean_fp, bound=DSP_FP_BOUND)
        return {"bound_exceeded": True, "dsp_clean_fp": dsp_clean_fp}

    # --- Generate eval-set patches in-memory ---
    eval_patches = []
    eval_labels = []
    eval_file_ids = []
    eval_sources = []
    file_to_id = {}
    next_file_id = 0

    for pf in per_file:
        name = pf["name"]
        if name not in file_to_id:
            file_to_id[name] = next_file_id
            next_file_id += 1
        fid = file_to_id[name]

        audio, sr = sf.read(pf["path"], dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        duration = len(audio) / sr

        gt_times = pf["gt_times"]
        det_times = pf["det_times"]

        # Patches from detector output
        for det_t in det_times:
            patch = extract_mel_patch(audio, sr, det_t)
            if patch is None:
                _diag("INFO", "ml.eval", "patch_extraction_failed",
                      file=name, at_sec=f"{det_t:.3f}", source="detection")
                continue
            label = 1 if is_tp(det_t, gt_times) else 0
            eval_patches.append(patch)
            eval_labels.append(label)
            eval_file_ids.append(fid)
            eval_sources.append(("detection", name, det_t))

        # Random negatives from clean files
        if not pf["spliced"]:
            rng = np.random.RandomState(_stable_hash(name))
            for _ in range(3):
                t = rng.uniform(PATCH_HALF_S, duration - PATCH_HALF_S)
                patch = extract_mel_patch(audio, sr, t)
                if patch is not None:
                    eval_patches.append(patch)
                    eval_labels.append(0)
                    eval_file_ids.append(fid)
                    eval_sources.append(("clean_random", name, t))

        # GT positions missed by detector
        if pf["spliced"]:
            for gt_t in gt_times:
                already_covered = any(abs(gt_t - d) < TOLERANCE_S for d in det_times)
                if not already_covered:
                    patch = extract_mel_patch(audio, sr, gt_t)
                    if patch is not None:
                        eval_patches.append(patch)
                        eval_labels.append(1)
                        eval_file_ids.append(fid)
                        eval_sources.append(("gt_missed", name, gt_t))

            # Random non-splice negatives from spliced files
            rng = np.random.RandomState(_stable_hash(name) + 1)
            for _ in range(2):
                t = rng.uniform(PATCH_HALF_S, duration - PATCH_HALF_S)
                if not any(abs(t - gt) < 3.0 for gt in gt_times):
                    patch = extract_mel_patch(audio, sr, t)
                    if patch is not None:
                        eval_patches.append(patch)
                        eval_labels.append(0)
                        eval_file_ids.append(fid)
                        eval_sources.append(("spliced_random_neg", name, t))

    n_eval = len(eval_patches)
    if n_eval == 0:
        print("classifier: NO_EVAL_PATCHES")
        _diag("WARN", "ml.eval", "no_eval_patches",
              fallback="dsp_only", combined=f"{dsp_combined:.4f}")
        print(f"combined: {dsp_combined:.6f}")
        return {"classifier_quality": "NO_DATA", "cv_f1": 0.0, "combined_full": dsp_combined}

    # --- Load pre-generated patches as extra training data ---
    pregen_X, pregen_y, pregen_fids = None, None, None
    if USE_PREGENERATED_PATCHES:
        all_pregen_patches = []
        all_pregen_labels = []
        all_pregen_fids = []
        fid_offset = next_file_id

        # 1. Singing + Korean pre-generated patches
        pregen_patches, pregen_labels, pregen_fids_raw = _load_pregenerated_patches()
        if pregen_patches is not None:
            all_pregen_patches.append(pregen_patches.reshape(len(pregen_patches), -1))
            all_pregen_labels.append(pregen_labels.astype(int))
            all_pregen_fids.append(pregen_fids_raw + fid_offset)
            fid_offset += int(pregen_fids_raw.max()) + 1

        # 2-n. Speech-splice datasets — each regenerated live from current detector.
        # Iterate dataset_registry.DATASETS so adding a new language is one entry.
        for ds in DATASETS:
            if ds.train_weight <= 0:
                continue
            sp_patches, sp_labels, sp_fids_raw = _load_speech_splice_patches(
                str(ds.path), ds.id
            )
            if sp_patches is not None:
                all_pregen_patches.append(sp_patches.reshape(len(sp_patches), -1))
                all_pregen_labels.append(sp_labels.astype(int))
                all_pregen_fids.append(sp_fids_raw + fid_offset)
                fid_offset += int(sp_fids_raw.max()) + 1

        if all_pregen_patches:
            pregen_X = np.vstack(all_pregen_patches)
            pregen_y = np.concatenate(all_pregen_labels)
            pregen_fids = np.concatenate(all_pregen_fids)

    # --- Build eval arrays ---
    X_eval = np.array(eval_patches).reshape(n_eval, -1)
    y_eval = np.array(eval_labels, dtype=int)
    fids_eval = np.array(eval_file_ids)

    n_unique_eval_groups = len(set(eval_file_ids))
    n_components = min(PCA_COMPONENTS, X_eval.shape[0] - 1, X_eval.shape[1])
    if pregen_X is not None:
        n_components = min(PCA_COMPONENTS, pregen_X.shape[0] + X_eval.shape[0] - 1, X_eval.shape[1])

    # --- OOF cross-validation on eval-set patches ---
    # Pre-generated patches are ALWAYS in the training fold (never tested on).
    # Only eval-set patches get OOF predictions.
    n_splits = min(N_FOLDS, max(n_unique_eval_groups, 2))
    if n_unique_eval_groups < 2:
        # Not enough groups for CV — fall back to DSP-only
        print(f"classifier: INSUFFICIENT_GROUPS ({n_unique_eval_groups})")
        _diag("WARN", "ml.eval", "insufficient_cv_groups",
              n_groups=n_unique_eval_groups, required=2, fallback="dsp_only")
        print(f"combined: {dsp_combined:.6f}")
        return {"classifier_quality": "LOW", "cv_f1": 0.0, "combined_full": dsp_combined}

    gkf = GroupKFold(n_splits=n_splits)
    oof_probas = np.full(n_eval, -1.0)
    oof_preds = np.full(n_eval, -1, dtype=int)

    for fold_idx, (eval_train_idx, eval_test_idx) in enumerate(gkf.split(X_eval, y_eval, groups=fids_eval)):
        # Training data = pre-generated (all) + eval training fold
        X_train = X_eval[eval_train_idx]
        y_train = y_eval[eval_train_idx]
        fids_train = fids_eval[eval_train_idx]

        if pregen_X is not None:
            X_train = np.vstack([pregen_X, X_train])
            y_train = np.concatenate([pregen_y, y_train])
            fids_train = np.concatenate([pregen_fids, fids_train])

        sw_train = compute_sample_weight("balanced", y_train)

        X_test = X_eval[eval_test_idx]

        fold_n_components = min(n_components, len(X_train) - 1)
        pipe = make_pipeline(fold_n_components)
        pipe.fit(X_train, y_train, clf__sample_weight=sw_train)

        oof_preds[eval_test_idx] = pipe.predict(X_test)
        oof_probas[eval_test_idx] = pipe.predict_proba(X_test)[:, 1]

    # CV F1 from OOF predictions (informational — no gate)
    valid_mask = oof_preds >= 0
    if valid_mask.sum() > 0:
        cv_f1 = float(f1_score(y_eval[valid_mask], oof_preds[valid_mask], zero_division=0))
    else:
        cv_f1 = 0.0
        _diag("WARN", "ml.eval", "no_valid_oof_preds",
              n_eval=n_eval, n_folds=n_splits)
    print(f"classifier_cv_f1: {cv_f1:.6f}")

    # --- Filter detections using OOF predictions (no quality gate) ---
    oof_lookup = {}
    for i, (source_type, name, t) in enumerate(eval_sources):
        if source_type == "detection" and oof_probas[i] >= 0:
            oof_lookup[(name, t)] = oof_probas[i]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    filtered_clean_fp = 0

    for pf in per_file:
        name = pf["name"]
        gt_times = pf["gt_times"]
        det_times = pf["det_times"]

        filtered_dets = []
        for det_t in det_times:
            proba = oof_lookup.get((name, det_t))
            if proba is None:
                # No OOF prediction — keep to be safe (visible via DIAG)
                _diag("INFO", "ml.filter", "no_oof_proba_keeping_det",
                      file=name, at_sec=f"{det_t:.3f}")
                filtered_dets.append(det_t)
            elif proba >= OOF_THRESHOLD:
                filtered_dets.append(det_t)

        tp, fp, fn, _ = match_detections(gt_times, filtered_dets, TOLERANCE_S)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if not pf["spliced"]:
            filtered_clean_fp += fp

    # Compute filtered metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    clean_score = 1.0 - (filtered_clean_fp / max(clean_files_count, 1))
    clean_score = max(0.0, clean_score)
    combined_full = f1 * clean_score

    print(f"classifier_quality: OK")
    print(f"ml_filtered_recall: {recall:.4f}")
    print(f"ml_filtered_precision: {precision:.4f}")
    print(f"ml_filtered_clean_fp: {filtered_clean_fp}")
    print(f"combined_full: {combined_full:.6f}")

    # --- Train final model on ALL data (for production) ---
    X_all = X_eval
    y_all = y_eval
    if pregen_X is not None:
        X_all = np.vstack([pregen_X, X_eval])
        y_all = np.concatenate([pregen_y, y_eval])
    sw_all = compute_sample_weight("balanced", y_all)
    final_n_components = min(n_components, len(X_all) - 1)
    final_pipe = make_pipeline(final_n_components)
    final_pipe.fit(X_all, y_all, clf__sample_weight=sw_all)
    model_path = os.path.join(_classifier_dir, "fp_classifier.joblib")
    joblib.dump(final_pipe, model_path)

    return {
        "classifier_quality": "OK",
        "cv_f1": cv_f1,
        "combined_full": combined_full,
        "filtered_precision": precision,
        "filtered_recall": recall,
        "filtered_f1": f1,
        "filtered_clean_fp": filtered_clean_fp,
    }
