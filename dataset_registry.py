"""Single source of truth for datasets the detector is evaluated + trained on.

Adding a new dataset: append a single Dataset entry to DATASETS below.
No other file should hardcode dataset paths or ids.

- `eval_weight > 0`  → this dataset contributes to the aggregate `combined` metric.
- `train_weight > 0` → per-file patches go into the classifier training fold.

Singing uses pre-generated patches (see .omc/classifier/patches_combined/),
so its `train_weight` is 0 here — its training data comes from a separate path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
# All data now lives under data/ at the repo root. Split layout:
#   data/eval/<id>/    ← used by evaluate.py and baseline metric
#   data/train/<id>/   ← used by .omc/classifier/train_classifier.py
#   data/test/<id>/    ← held-out, run with test_eval.py (20-min budget)
#   data/sources/...   ← source audio pools (not directly consumed at eval time)
_DATA = _PROJECT_ROOT / "data"

# Eval-set isolation (see scripts/eval_crypto.py + run_autoresearch.sh).
# The wrapper decrypts data/eval.tar.gz.enc into a fresh /tmp dir and sets
# OMC_EVAL_DATA_ROOT before invoking evaluate.py so the plaintext corpus
# never lives on disk in the repo. Claude-subprocess iterations are spawned
# without this env var (via `env -u`), so they see the default
# data/eval/<id>/ paths — which do not exist on disk post-setup — and
# cannot locate the decrypted tree. Resolved once at import.
_EVAL_ROOT_OVERRIDE = os.environ.get("OMC_EVAL_DATA_ROOT") or None


def _eval_path(default: Path, ds_id: str) -> Path:
    if _EVAL_ROOT_OVERRIDE:
        return Path(_EVAL_ROOT_OVERRIDE) / ds_id
    return default


@dataclass(frozen=True)
class Dataset:
    id: str                 # short, unique; used in DIAG labels and metric keys
    eval_path: Path         # dir with tier1/ tier2/ clean/ ground_truth.json
    train_path: Path        # disjoint-source train split for the classifier
    test_path: Path         # held-out, disjoint-source test split
    eval_weight: float = 1.0
    train_weight: float = 1.0

    # Back-compat alias so pre-rename callers that still dereference `ds.path`
    # get the eval path (the original semantics before train/test splits
    # existed).
    @property
    def path(self) -> Path:
        return self.eval_path


DATASETS: list[Dataset] = [
    Dataset(
        id="singing",
        eval_path=_eval_path(_DATA / "eval" / "singing", "singing"),
        train_path=_DATA / "train" / "singing",
        test_path=_DATA / "test" / "singing",
    ),
    Dataset(
        id="korean",
        eval_path=_eval_path(_DATA / "eval" / "korean", "korean"),
        train_path=_DATA / "train" / "korean",
        test_path=_DATA / "test" / "korean",
    ),
    Dataset(
        id="english",
        eval_path=_eval_path(_DATA / "eval" / "english", "english"),
        train_path=_DATA / "train" / "english",
        test_path=_DATA / "test" / "english",
    ),
]


def aggregate_combined(
    per_dataset: dict[str, float],
    method: str = "geometric",
    floor: float = 0.01,
) -> dict:
    """Aggregate per-dataset `combined` scores into a single primary metric.

    Produces four outputs for reporting and optimization:
      - combined:      the selected aggregation (primary metric)
      - combined_mean: arithmetic mean across datasets
      - combined_min:  minimum across datasets (regression sentinel)
      - per_dataset:   echo of the input (so callers don't need to track both)

    method:
      - 'geometric' (default): inverse-proportional pressure — weak datasets pull
        the aggregate down more than strong datasets push it up. Each value is
        floored at `floor` before the mean to avoid zero-collapse when a new
        dataset hasn't produced any detections yet.
      - 'mean':  equal marginal weight per dataset.
      - 'min':   all pressure on the weakest dataset.
    """
    if not per_dataset:
        return {"combined": 0.0, "combined_mean": 0.0, "combined_min": 0.0, "per_dataset": {}}

    values = list(per_dataset.values())
    mean_val = float(np.mean(values))
    min_val = float(min(values))

    if method == "geometric":
        floored = np.array([max(v, floor) for v in values], dtype=np.float64)
        combined = float(np.exp(np.log(floored).mean()))
    elif method == "mean":
        combined = mean_val
    elif method == "min":
        combined = min_val
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}")

    return {
        "combined": combined,
        "combined_mean": mean_val,
        "combined_min": min_val,
        "per_dataset": dict(per_dataset),
    }


def forensic_combined(
    per_cell: dict[str, float],
    clean_fp: int,
    max_allowed: int = 15,
    floor: float = 0.01,
) -> dict:
    """15-cell geometric mean × smooth FP multiplier.

    Built for the Phase-1 forensic metric (spec: gbm-main-forensic-metric).
    Callers should pass a dict with 15 keys by convention:
      12 spliced F1 cells:  `<dataset>_t<tier>_<regime>`
                            e.g. singing_t1_random, korean_t2_quiet_matched
       3 clean-score cells: `<dataset>_clean_score`

    Formula:
      combined = GM(max(v, floor) for v in per_cell.values()) *
                 max(0, 1 - clean_fp / max(max_allowed, 1))

    The floor prevents zero-collapse when a cell has no signal yet; the
    FP multiplier is clamped to [0, 1].

    Returns:
      combined:          scalar primary metric ∈ [0, 1]
      combined_gm_cells: GM of floored cells, before FP multiplier
      fp_multiplier:     clamped (1 - clean_fp/max_allowed)
      per_cell:          echo of input
      n_cells_included:  number of cells in the GM
    """
    if not per_cell:
        return {
            "combined": 0.0,
            "combined_gm_cells": 0.0,
            "fp_multiplier": 1.0,
            "per_cell": {},
            "n_cells_included": 0,
        }

    floored = np.array(
        [max(float(v), floor) for v in per_cell.values()], dtype=np.float64
    )
    gm = float(np.exp(np.log(floored).mean()))

    raw_mult = 1.0 - float(clean_fp) / max(int(max_allowed), 1)
    fp_multiplier = max(0.0, min(1.0, raw_mult))

    combined = gm * fp_multiplier

    return {
        "combined": combined,
        "combined_gm_cells": gm,
        "fp_multiplier": fp_multiplier,
        "per_cell": dict(per_cell),
        "n_cells_included": len(per_cell),
    }


if __name__ == "__main__":
    # Sanity tests
    ids = [d.id for d in DATASETS]
    assert ids == ["singing", "korean", "english"], ids

    # Geometric mean with floor
    got = aggregate_combined(
        {"s": 0.738, "k": 0.25, "e": 0.05}, method="geometric", floor=0.01,
    )
    expected = float(np.exp((np.log(0.738) + np.log(0.25) + np.log(0.05)) / 3))
    assert abs(got["combined"] - expected) < 1e-4, (got["combined"], expected)
    assert abs(got["combined_mean"] - (0.738 + 0.25 + 0.05) / 3) < 1e-6
    assert got["combined_min"] == 0.05

    # Floor kicks in when a value is below it
    got_floor = aggregate_combined(
        {"a": 0.8, "b": 0.0}, method="geometric", floor=0.01,
    )
    expected_floor = float(np.exp((np.log(0.8) + np.log(0.01)) / 2))
    assert abs(got_floor["combined"] - expected_floor) < 1e-4

    # Mean and min
    got_mean = aggregate_combined({"a": 0.4, "b": 0.6}, method="mean")
    assert abs(got_mean["combined"] - 0.5) < 1e-9
    got_min = aggregate_combined({"a": 0.4, "b": 0.6}, method="min")
    assert got_min["combined"] == 0.4

    # --- forensic_combined tests ---
    # 15 synthetic cells all at 0.5, no FP
    uniform_cells = {f"cell_{i}": 0.5 for i in range(15)}
    got_f = forensic_combined(uniform_cells, clean_fp=0, max_allowed=15)
    assert abs(got_f["combined"] - 0.5) < 1e-9, got_f
    assert got_f["fp_multiplier"] == 1.0
    assert got_f["n_cells_included"] == 15

    # One cell at 0.01 (the floor), others at 0.5 — GM drops
    mixed_cells = dict(uniform_cells)
    mixed_cells["cell_0"] = 0.01
    got_mix = forensic_combined(mixed_cells, clean_fp=0, max_allowed=15)
    # GM = exp((14*log(0.5) + log(0.01))/15) ≈ 0.381
    expected_gm = float(np.exp((14 * np.log(0.5) + np.log(0.01)) / 15))
    assert abs(got_mix["combined_gm_cells"] - expected_gm) < 1e-6, got_mix
    # GM of 14×0.5 + 1×0.01 ≈ 0.385 — meaningfully below the uniform 0.5
    assert got_mix["combined"] < 0.40, got_mix
    assert got_mix["combined"] < 0.5, got_mix  # penalty signal confirmed

    # FP multiplier: 5/15 → 0.667
    got_fp = forensic_combined(uniform_cells, clean_fp=5, max_allowed=15)
    assert abs(got_fp["fp_multiplier"] - (1 - 5/15)) < 1e-9, got_fp
    assert abs(got_fp["combined"] - 0.5 * (1 - 5/15)) < 1e-9

    # Clamp: clean_fp > max_allowed → 0
    got_clamp = forensic_combined(uniform_cells, clean_fp=30, max_allowed=15)
    assert got_clamp["fp_multiplier"] == 0.0, got_clamp
    assert got_clamp["combined"] == 0.0

    print(f"DATASETS: {ids}")
    print(f"Self-tests: PASS (aggregate_combined + forensic_combined)")
    print(f"Example aggregation: {got}")
    print(f"Example forensic (15x0.5, fp=5, max=15): combined={got_fp['combined']:.4f} "
          f"gm={got_fp['combined_gm_cells']:.4f} fpm={got_fp['fp_multiplier']:.4f}")
