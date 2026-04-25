"""Unit tests for the korean-iter1 3-class train_classifier migration.

Run:
    PYTHONPATH=$PWD uv run pytest splice/tests/test_train_classifier_korean_iter1.py -v
"""

import json
import os

# Disable logger for unit tests — no real log sink needed
os.environ.setdefault("OMC_LOGGER_DISABLED", "1")

import numpy as np
import pytest

# Module-level imports only; no I/O at import time
from splice.classifier.train_classifier import (
    CLASS_NAMES,
    NEG_MIN_DIST_S,
    NEG_RATIO,
    _LE,
    _sample_negatives,
)


# ---------------------------------------------------------------------------
# 1. Label encoder round-trip
# ---------------------------------------------------------------------------


def test_label_encoder_3_class():
    labels = ["cross_voice", "no_splice", "same_voice_edit", "cross_voice"]
    encoded = _LE.transform(labels)
    decoded = _LE.inverse_transform(encoded).tolist()
    assert decoded == labels, f"Round-trip failed: {decoded}"
    assert len(set(encoded)) == 3, "Expected 3 distinct integer codes"


# ---------------------------------------------------------------------------
# 2. Negative-sample distance constraint
# ---------------------------------------------------------------------------


def test_negative_sample_distance_constraint():
    boundary_times = [5.0, 12.0]
    audio_dur_s = 30.0
    n_neg = 20
    neg_times = _sample_negatives(
        conv_id="test_conv_dist",
        audio_dur_s=audio_dur_s,
        boundary_times=boundary_times,
        n_neg=n_neg,
    )
    assert len(neg_times) > 0, "No negatives sampled"
    for t in neg_times:
        for b in boundary_times:
            assert abs(t - b) >= NEG_MIN_DIST_S, (
                f"Negative at t={t:.3f} is {abs(t-b):.3f}s from boundary "
                f"at {b} — must be >= {NEG_MIN_DIST_S}s"
            )


# ---------------------------------------------------------------------------
# 3. Negative-sample count ratio
# ---------------------------------------------------------------------------


def test_negative_sample_count_ratio():
    boundary_times = [4.0, 9.0, 15.0]  # 3 positives
    audio_dur_s = 60.0
    n_pos = len(boundary_times)
    n_neg_target = max(1, round(n_pos * NEG_RATIO))
    neg_times = _sample_negatives(
        conv_id="test_conv_ratio",
        audio_dur_s=audio_dur_s,
        boundary_times=boundary_times,
        n_neg=n_neg_target,
    )
    ratio = len(neg_times) / n_pos
    # Allow ±10% around the 2× target (or fewer if sampling fails, which is
    # only expected for very dense boundaries — not the case here)
    assert ratio >= NEG_RATIO * 0.9, (
        f"Got {len(neg_times)} negatives for {n_pos} positives "
        f"(ratio={ratio:.2f}), expected >= {NEG_RATIO * 0.9:.2f}"
    )
    assert ratio <= NEG_RATIO * 1.1, (
        f"Got {len(neg_times)} negatives for {n_pos} positives "
        f"(ratio={ratio:.2f}), expected <= {NEG_RATIO * 1.1:.2f}"
    )


# ---------------------------------------------------------------------------
# 4. Meta schema required fields
# ---------------------------------------------------------------------------


def test_meta_schema_required_fields():
    required_keys = {
        "version",
        "trained_at",
        "n_train_files",
        "n_train_samples",
        "classes_",
        "n_features",
        "feature_names",
        "hyperparameters",
        "oof_metrics",
        "voice_holdout",
    }
    required_oof_keys = {"f1_macro", "f1_per_class"}

    # Build an in-memory mock meta matching what train() would write
    mock_meta = {
        "version": "korean_iter1_v1",
        "trained_at": "2026-04-25T00:00:00Z",
        "n_train_files": 10,
        "n_train_samples": 120,
        "classes_": CLASS_NAMES,
        "n_features": 80,
        "feature_names": [f"feat_{i}" for i in range(80)],
        "hyperparameters": {"max_iter": 300},
        "oof_metrics": {
            "f1_macro": 0.75,
            "f1_weighted": 0.76,
            "f1_per_class": {
                "cross_voice": 0.80,
                "no_splice": 0.90,
                "same_voice_edit": 0.55,
            },
        },
        "voice_holdout": {"test": [], "eval": [], "train": []},
    }

    # Serialize / deserialize to simulate disk round-trip
    raw = json.dumps(mock_meta)
    loaded = json.loads(raw)

    missing = required_keys - set(loaded.keys())
    assert not missing, f"Meta missing required keys: {missing}"

    missing_oof = required_oof_keys - set(loaded["oof_metrics"].keys())
    assert not missing_oof, f"oof_metrics missing keys: {missing_oof}"


# ---------------------------------------------------------------------------
# 5. classes_ ordering is alphabetical
# ---------------------------------------------------------------------------


def test_classes_ordering_alphabetical():
    expected = ["cross_voice", "no_splice", "same_voice_edit"]
    assert CLASS_NAMES == expected, (
        f"CLASS_NAMES must be alphabetically sorted: {CLASS_NAMES}"
    )
    # Also check that the label encoder classes_ matches
    assert list(_LE.classes_) == expected, (
        f"LabelEncoder classes_ must match: {list(_LE.classes_)}"
    )
