"""
ML evaluation helper for prepare.py --with-classifier.

Runs GradientBoosting classifier pipeline in-loop:
1. Generate mel spectrogram patches from DSP detections
2. 5-fold GroupKFold CV with OOF predictions
3. Quality gate: CV F1 < 0.60 → fall back to DSP-only
4. Filter detections using OOF predictions (no train-on-test)
5. Recompute combined_full from filtered results
"""

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

from prepare import match_detections, TOLERANCE_S
from generate_patches import extract_mel_patch, is_tp, PATCH_HALF_S
from train_classifier import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
import joblib

DSP_FP_BOUND = 15
CLASSIFIER_QUALITY_GATE = 0.60
N_FOLDS = 5
PATCH_TARGET_FRAMES = 200  # must match generate_patches.py


def evaluate_with_classifier(dsp_results, data_dir):
    """Run classifier pipeline on DSP results, return ML-augmented metrics.

    Args:
        dsp_results: dict from prepare.evaluate() with per_file key
        data_dir: path to spliced test data directory

    Returns:
        dict with keys: bound_exceeded, classifier_quality, cv_f1, combined_full
    """
    dsp_clean_fp = dsp_results["clean_fp"]
    dsp_combined = dsp_results["combined"]
    per_file = dsp_results["per_file"]
    clean_files_count = dsp_results["clean_files"]

    print(f"combined_dsp: {dsp_combined:.6f}")
    print(f"dsp_clean_fp: {dsp_clean_fp}")

    # --- FP bound check ---
    if dsp_clean_fp > DSP_FP_BOUND:
        print(f"dsp_fp_bound: EXCEEDED")
        return {"bound_exceeded": True, "dsp_clean_fp": dsp_clean_fp}

    # --- Generate patches in-memory ---
    all_patches = []
    all_labels = []
    all_file_ids = []
    all_sources = []  # track which are detection-based vs random
    file_to_id = {}
    next_file_id = 0

    for pf in per_file:
        name = pf["name"]
        if name not in file_to_id:
            file_to_id[name] = next_file_id
            next_file_id += 1
        fid = file_to_id[name]

        # Load audio
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
                continue
            label = 1 if is_tp(det_t, gt_times) else 0
            all_patches.append(patch)
            all_labels.append(label)
            all_file_ids.append(fid)
            all_sources.append(("detection", name, det_t))

        # Random negatives from clean files
        if not pf["spliced"]:
            rng = np.random.RandomState(hash(name) % (2**31))
            for _ in range(3):
                t = rng.uniform(PATCH_HALF_S, duration - PATCH_HALF_S)
                patch = extract_mel_patch(audio, sr, t)
                if patch is not None:
                    all_patches.append(patch)
                    all_labels.append(0)
                    all_file_ids.append(fid)
                    all_sources.append(("clean_random", name, t))

        # For spliced files: add GT positions if detector missed them
        if pf["spliced"]:
            for gt_t in gt_times:
                already_covered = any(abs(gt_t - d) < TOLERANCE_S for d in det_times)
                if not already_covered:
                    patch = extract_mel_patch(audio, sr, gt_t)
                    if patch is not None:
                        all_patches.append(patch)
                        all_labels.append(1)
                        all_file_ids.append(fid)
                        all_sources.append(("gt_missed", name, gt_t))

            # Random non-splice negatives from spliced files
            rng = np.random.RandomState(hash(name) % (2**31) + 1)
            for _ in range(2):
                t = rng.uniform(PATCH_HALF_S, duration - PATCH_HALF_S)
                if not any(abs(t - gt) < 3.0 for gt in gt_times):
                    patch = extract_mel_patch(audio, sr, t)
                    if patch is not None:
                        all_patches.append(patch)
                        all_labels.append(0)
                        all_file_ids.append(fid)
                        all_sources.append(("spliced_random_neg", name, t))

    if len(all_patches) == 0:
        print("classifier_quality: INSUFFICIENT_DATA")
        return {"classifier_quality": "LOW", "cv_f1": 0.0, "combined_full": None}

    X = np.array(all_patches).reshape(len(all_patches), -1)
    y = np.array(all_labels, dtype=int)
    file_ids = np.array(all_file_ids)

    n_unique_groups = len(set(file_ids))
    if len(X) < 20 or n_unique_groups < 2:
        print("classifier_quality: INSUFFICIENT_DATA")
        return {"classifier_quality": "LOW", "cv_f1": 0.0, "combined_full": None}

    # --- 5-fold GroupKFold CV with OOF predictions ---
    n_splits = min(N_FOLDS, n_unique_groups)
    gkf = GroupKFold(n_splits=n_splits)
    n_components = min(100, len(X) - 1, X.shape[1])

    oof_probas = np.full(len(X), -1.0)  # OOF prediction for each sample
    oof_preds = np.full(len(X), -1, dtype=int)

    sample_weights = compute_sample_weight("balanced", y)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=file_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        sw_train = sample_weights[train_idx]

        pipe = make_pipeline(n_components)
        pipe.fit(X_train, y_train, clf__sample_weight=sw_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        oof_probas[test_idx] = y_proba
        oof_preds[test_idx] = y_pred

    # Compute CV F1 from OOF predictions
    valid_mask = oof_preds >= 0
    if valid_mask.sum() == 0:
        print("classifier_quality: INSUFFICIENT_DATA")
        return {"classifier_quality": "LOW", "cv_f1": 0.0, "combined_full": None}

    cv_f1 = float(f1_score(y[valid_mask], oof_preds[valid_mask], zero_division=0))
    print(f"classifier_cv_f1: {cv_f1:.6f}")

    if cv_f1 < CLASSIFIER_QUALITY_GATE:
        print("classifier_quality: LOW")
        return {"classifier_quality": "LOW", "cv_f1": cv_f1, "combined_full": None}

    # --- Filter detections using OOF predictions ---
    # Build lookup: (file_name, det_time) -> OOF proba
    oof_lookup = {}
    for i, (source_type, name, t) in enumerate(all_sources):
        if source_type == "detection" and oof_probas[i] >= 0:
            oof_lookup[(name, t)] = oof_probas[i]

    # Re-evaluate with filtered detections
    total_tp = 0
    total_fp = 0
    total_fn = 0
    filtered_clean_fp = 0

    for pf in per_file:
        name = pf["name"]
        gt_times = pf["gt_times"]
        det_times = pf["det_times"]

        # Filter: keep detections where OOF proba >= 0.5
        filtered_dets = []
        for det_t in det_times:
            proba = oof_lookup.get((name, det_t))
            if proba is None:
                # No OOF prediction (e.g., patch extraction failed) — keep to be safe
                filtered_dets.append(det_t)
            elif proba >= 0.5:
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

    print("classifier_quality: OK")
    print(f"combined_full: {combined_full:.6f}")

    # --- Train final model on ALL data (for production/web demo) ---
    final_pipe = make_pipeline(n_components)
    final_pipe.fit(X, y, clf__sample_weight=sample_weights)
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
