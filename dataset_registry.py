"""Single source of truth for datasets the detector is evaluated + trained on.

Adding a new dataset: append a single Dataset entry to DATASETS below.
No other file should hardcode dataset paths or ids.

- `eval_weight > 0`  → this dataset contributes to the aggregate `combined` metric.
- `train_weight > 0` → per-file patches go into the classifier training fold.

Singing uses pre-generated patches (see .omc/classifier/patches_combined/),
so its `train_weight` is 0 here — its training data comes from a separate path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
# Speech corpora live in the `corpora/` submodule (formerly the sibling
# audio-splice-detector repo, relocated 2026-04-17).
_CORPORA_DATA = _PROJECT_ROOT / "corpora" / "data"


@dataclass(frozen=True)
class Dataset:
    id: str                 # short, unique; used in DIAG labels and metric keys
    path: Path              # directory containing tier1/ tier2/ clean/ ground_truth.json
    eval_weight: float = 1.0
    train_weight: float = 1.0


DATASETS: list[Dataset] = [
    Dataset(
        id="singing",
        path=_PROJECT_ROOT / "data" / "spliced",
        eval_weight=1.0,
        train_weight=0.0,   # patches come from pre-generated patches_combined/
    ),
    Dataset(
        id="korean",
        path=_CORPORA_DATA / "korean-splice",
        eval_weight=1.0,
        train_weight=1.0,
    ),
    Dataset(
        id="english",
        path=_CORPORA_DATA / "english-splice",
        eval_weight=1.0,
        train_weight=1.0,
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

    print(f"DATASETS: {ids}")
    print(f"Self-tests: PASS")
    print(f"Example aggregation: {got}")
