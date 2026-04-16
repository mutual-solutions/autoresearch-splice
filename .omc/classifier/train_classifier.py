"""
Train a GradientBoosting classifier to filter false positives from the
audio splice detector.

Uses 5-fold file-level cross-validation (GroupKFold) to prevent data
leakage between patches from the same audio file.

Input: flattened mel spectrogram patches (128 x 200 -> 25600-dim)
Output: probability of being a real splice

Uses PCA for dimensionality reduction before classification.
"""

import json
import os
import sys
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data():
    """Load patches, labels, and manifest. Return X, y, file_ids."""
    patch_dir = os.path.join(PROJECT_ROOT, ".omc", "classifier", "patches")
    patches = np.load(os.path.join(patch_dir, "patches.npy"))
    labels = np.load(os.path.join(patch_dir, "labels.npy"))

    with open(os.path.join(patch_dir, "manifest.json")) as f:
        manifest = json.load(f)

    # Extract file name from each manifest entry -> build file_ids
    filenames = [entry["file"] for entry in manifest]
    unique_files = sorted(set(filenames))
    file_to_id = {name: idx for idx, name in enumerate(unique_files)}
    file_ids = np.array([file_to_id[name] for name in filenames])

    # Save file_id_map.json for reproducibility
    map_path = os.path.join(patch_dir, "file_id_map.json")
    with open(map_path, "w") as f:
        json.dump(file_to_id, f, indent=2)
    print(f"Saved file_id_map.json ({len(unique_files)} unique files)")

    return patches, labels, file_ids


def make_pipeline(n_components):
    """Create a StandardScaler -> PCA -> GradientBoosting pipeline.

    Reads classifier hyperparameters from ml_config.py (agent-editable).
    Falls back to defaults if ml_config is not importable.
    """
    try:
        from ml_config import (
            N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, RANDOM_STATE,
        )
    except ImportError:
        N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, RANDOM_STATE = 200, 5, 0.1, 0.8, 42
    try:
        from ml_config import MAX_FEATURES
    except ImportError:
        MAX_FEATURES = None

    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ('clf', GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            subsample=SUBSAMPLE,
            max_features=MAX_FEATURES,
            random_state=RANDOM_STATE,
        ))
    ])


def train():
    patches, labels, file_ids = load_data()
    print(f"Loaded {len(patches)} patches, shape={patches.shape}")
    print(f"  TP (label=1): {int(labels.sum())}")
    print(f"  FP/neg (label=0): {int(len(labels) - labels.sum())}")

    # Flatten patches: (N, 128, 200) -> (N, 25600)
    X = patches.reshape(len(patches), -1)
    y = labels.astype(int)

    # Compute sample weights based on inverse class frequency
    sample_weights = compute_sample_weight("balanced", y)

    n_components = min(100, len(X) - 1, X.shape[1])

    # === 5-fold file-level cross-validation ===
    gkf = GroupKFold(n_splits=5)
    fold_metrics = []

    print(f"\n=== 5-Fold File-Level Cross-Validation ===")
    print(f"  PCA components: {n_components}")

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=file_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sw_train = sample_weights[train_idx]

        pipe = make_pipeline(n_components)

        # Fit with sample_weight passed to the classifier step
        pipe.fit(X_train, y_train, clf__sample_weight=sw_train)

        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        fold_metrics.append({
            "fold": fold_idx + 1,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_tp": int(y_train.sum()),
            "test_tp": int(y_test.sum()),
        })

        n_test_files = len(set(file_ids[test_idx]))
        print(f"\n  Fold {fold_idx + 1}: {n_test_files} test files, "
              f"{len(test_idx)} patches (TP={int(y_test.sum())})")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1:        {f1:.4f}")

    # Mean +/- std across folds
    accs = [m["accuracy"] for m in fold_metrics]
    precs = [m["precision"] for m in fold_metrics]
    recs = [m["recall"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]

    print(f"\n=== Mean +/- Std Across 5 Folds ===")
    print(f"  Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
    print(f"  Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
    print(f"  F1:        {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # Save CV results
    cv_results = {
        "n_splits": 5,
        "split_method": "GroupKFold (file-level)",
        "n_samples": len(X),
        "n_features_pca": n_components,
        "folds": fold_metrics,
        "mean": {
            "accuracy": float(np.mean(accs)),
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
            "f1": float(np.mean(f1s)),
        },
        "std": {
            "accuracy": float(np.std(accs)),
            "precision": float(np.std(precs)),
            "recall": float(np.std(recs)),
            "f1": float(np.std(f1s)),
        },
    }
    cv_path = os.path.join(PROJECT_ROOT, ".omc", "classifier", "cv_results.json")
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"\nCV results saved to {cv_path}")

    # === Train final model on ALL data ===
    print(f"\n=== Training Final Model on All Data ===")
    final_pipe = make_pipeline(n_components)
    final_pipe.fit(X, y, clf__sample_weight=sample_weights)

    model_path = os.path.join(PROJECT_ROOT, ".omc", "classifier", "fp_classifier.joblib")
    joblib.dump(final_pipe, model_path)
    print(f"Final model saved to {model_path}")

    return final_pipe


if __name__ == "__main__":
    train()
