"""Unit tests for the korean-iter1 boundary-F1 contract in
`splice/evaluate.py` (Step 9 of `.omc/plans/ralplan-korean-iter1-reset.md`).

Run:
    PYTHONPATH=$PWD uv run pytest splice/tests/test_evaluate_korean_iter1.py -v
"""

import os

# Logger is silenced for unit tests — these tests only exercise pure
# scoring functions and never touch the real eval loop / disk.
os.environ.setdefault("OMC_LOGGER_DISABLED", "1")

from splice.evaluate import (  # noqa: E402
    COLLAR_S,
    boundary_f1,
    compute_class_f1,
    normalize_pred_tuple,
)


# ---------------------------------------------------------------------------
# boundary_f1 (label-blind)
# ---------------------------------------------------------------------------

def test_boundary_f1_perfect_match():
    preds = [(1.0, "cross_voice"), (2.0, "cross_voice"), (3.0, "cross_voice")]
    gt = [(1.0, "cross_voice"), (2.0, "cross_voice"), (3.0, "cross_voice")]
    p, r, f1, tp, fp, fn = boundary_f1(preds, gt, COLLAR_S)
    assert (tp, fp, fn) == (3, 0, 0)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_boundary_f1_no_match():
    # 2s gap between every pred and every GT — well beyond 0.25 collar.
    preds = [(10.0, "cross_voice"), (20.0, "cross_voice"), (30.0, "cross_voice")]
    gt = [(12.0, "cross_voice"), (22.0, "cross_voice"), (32.0, "cross_voice")]
    p, r, f1, tp, fp, fn = boundary_f1(preds, gt, COLLAR_S)
    assert tp == 0
    assert fp == 3
    assert fn == 3
    assert f1 == 0.0


def test_boundary_f1_partial_match():
    preds = [(1.0, "cross_voice"), (2.0, "cross_voice"), (50.0, "cross_voice")]
    gt = [(1.0, "cross_voice"), (2.0, "cross_voice"), (3.0, "cross_voice")]
    p, r, f1, tp, fp, fn = boundary_f1(preds, gt, COLLAR_S)
    assert tp == 2
    assert fp == 1
    assert fn == 1
    # P = 2/3, R = 2/3, F1 = 2/3
    assert abs(p - 2 / 3) < 1e-9
    assert abs(r - 2 / 3) < 1e-9
    assert abs(f1 - 2 / 3) < 1e-9


def test_boundary_f1_collar_boundary():
    # exactly collar away → matches (≤ collar)
    preds_in = [(1.0 + COLLAR_S, "cross_voice")]
    gt = [(1.0, "cross_voice")]
    _, _, f1_in, tp_in, fp_in, fn_in = boundary_f1(preds_in, gt, COLLAR_S)
    assert (tp_in, fp_in, fn_in) == (1, 0, 0)
    assert f1_in == 1.0

    # collar + 0.001 away → no match
    preds_out = [(1.0 + COLLAR_S + 0.001, "cross_voice")]
    _, _, f1_out, tp_out, fp_out, fn_out = boundary_f1(preds_out, gt, COLLAR_S)
    assert (tp_out, fp_out, fn_out) == (0, 1, 1)
    assert f1_out == 0.0


def test_boundary_f1_one_to_one():
    # 3 preds clustered within collar of a single GT — only one TP.
    preds = [
        (1.00, "cross_voice"),
        (1.05, "cross_voice"),
        (1.10, "cross_voice"),
    ]
    gt = [(1.0, "cross_voice")]
    _, _, _, tp, fp, fn = boundary_f1(preds, gt, COLLAR_S)
    assert tp == 1
    assert fp == 2
    assert fn == 0


# ---------------------------------------------------------------------------
# compute_class_f1 (Item 3 — REVISED)
# ---------------------------------------------------------------------------

def test_compute_class_f1_excludes_unknown():
    """An 'unknown' prediction must NOT count toward cross_voice_f1.

    This is the critical Item 3 semantics: a label-blind detector that
    emits 'unknown' for everything cannot inflate per-class F1.
    """
    preds = [(1.0, "unknown"), (2.0, "unknown")]
    gt_cv = [(1.0, "cross_voice")]
    f1 = compute_class_f1(preds, gt_cv, "cross_voice", COLLAR_S)
    assert f1 == 0.0, "unknown predictions must contribute zero to cross_voice_f1"


def test_compute_class_f1_only_correct_label():
    preds = [(1.0, "cross_voice"), (2.0, "same_voice_edit")]
    gt_cv = [(1.0, "cross_voice")]
    # The same_voice_edit pred is filtered out; cross_voice pred matches.
    f1 = compute_class_f1(preds, gt_cv, "cross_voice", COLLAR_S)
    assert f1 == 1.0


# ---------------------------------------------------------------------------
# normalize_pred_tuple
# ---------------------------------------------------------------------------

def test_normalize_pred_tuple_handles_bare_float():
    assert normalize_pred_tuple(1.5) == (1.5, "unknown")
    assert normalize_pred_tuple(0) == (0.0, "unknown")


def test_normalize_pred_tuple_invalid_label_becomes_unknown():
    assert normalize_pred_tuple((1.5, "garbage")) == (1.5, "unknown")
    assert normalize_pred_tuple((1.5, "")) == (1.5, "unknown")
    assert normalize_pred_tuple((1.5, None)) == (1.5, "unknown")
    # Valid labels survive
    assert normalize_pred_tuple((1.5, "cross_voice")) == (1.5, "cross_voice")
    assert normalize_pred_tuple((1.5, "same_voice_edit")) == (1.5, "same_voice_edit")
    assert normalize_pred_tuple((1.5, "unknown")) == (1.5, "unknown")


# ---------------------------------------------------------------------------
# Combined: aggregate counts unknown predictions, per-class does not
# ---------------------------------------------------------------------------

def test_combined_aggregate_counts_unknown_predictions():
    """3 preds at correct times — 2 'cross_voice', 1 'unknown'.
    3 GT all 'cross_voice'.

    Aggregate boundary_f1 == 1.0 (label-blind: all 3 match).
    cross_voice_f1 == 2/3 (the unknown pred is filtered out;
    only 2 cross_voice preds remain to match against 3 cross_voice GT).
    """
    preds = [
        (1.0, "cross_voice"),
        (2.0, "cross_voice"),
        (3.0, "unknown"),
    ]
    gt = [
        (1.0, "cross_voice"),
        (2.0, "cross_voice"),
        (3.0, "cross_voice"),
    ]
    _, _, agg_f1, agg_tp, agg_fp, agg_fn = boundary_f1(preds, gt, COLLAR_S)
    assert agg_tp == 3
    assert agg_fp == 0
    assert agg_fn == 0
    assert agg_f1 == 1.0

    gt_cv = [g for g in gt if g[1] == "cross_voice"]
    cv_f1 = compute_class_f1(preds, gt_cv, "cross_voice", COLLAR_S)
    # filtered preds = 2 cross_voice, gt = 3 cross_voice → P=2/2, R=2/3
    # F1 = 2*1*(2/3) / (1 + 2/3) = (4/3) / (5/3) = 4/5 = 0.8
    assert abs(cv_f1 - 0.8) < 1e-9, f"expected 0.8, got {cv_f1}"
