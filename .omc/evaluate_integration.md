# evaluate.py integration with dataset_registry

`splice/evaluate.py` is protected — only you apply these edits.

## Goal

Replace the hardcoded Korean secondary eval block with a loop over
`dataset_registry.DATASETS`, compute a balanced `combined` via
`aggregate_combined`, and print per-dataset plus aggregate metrics.

Result: adding a dataset → one line in `dataset_registry.py`, no
further edits anywhere else.

## Files to edit

1. `splice/evaluate.py` (this repo) — one block replaced, a few lines added.
2. `autoresearch/baseline_metrics.json` — new primary baseline.

## Before

**NOTE (2026-04-17): The speech corpora moved.** The hardcoded path in
`evaluate.py:522` points to the old location (`audio-splice-detector/`);
the new location is the `corpora/` submodule inside this repo. Until
this patch is applied, the Korean secondary eval is SILENTLY SKIPPED
because `os.path.isdir(_speech_dir)` is False. Apply this patch to
restore the Korean eval (and add English) via `dataset_registry.DATASETS`.

Current block at `evaluate.py:518-542` (roughly):

```python
# Secondary eval: Korean speech (DSP-only, informational).
_speech_dir = "/Users/yejunjang/Projects/mutual/audio-splice-detector/data/korean-splice"
if os.path.isdir(_speech_dir):
    print("\n=== Secondary eval: Korean speech (DSP-only) ===")
    try:
        _speech = evaluate(_speech_dir)
        print(f"combined_speech: {_speech['combined']:.6f}  "
              f"(recall={_speech['recall']:.2f}, precision={_speech['precision']:.2f}, "
              f"clean_fp={_speech['clean_fp']})")
    except Exception as _e:
        print(f"combined_speech: ERROR ({type(_e).__name__}: {_e})")
    print("=== End secondary eval ===\n")

if args.with_classifier:
    from ml_eval import evaluate_with_classifier
    ml_result = evaluate_with_classifier(result, args.data_dir)
    if ml_result.get("bound_exceeded"):
        print(f"combined: 0.000000")
    elif ml_result.get("combined_full") is not None:
        print(f"combined: {ml_result['combined_full']:.6f}")
    else:
        print(f"combined: {result['combined']:.6f}")
```

## After

```python
from dataset_registry import DATASETS, aggregate_combined

# Iterate eval datasets (skip the primary — already run above as `result`).
primary_path = os.path.abspath(args.data_dir)
per_dataset_combined: dict[str, float] = {}

for ds in DATASETS:
    if ds.eval_weight <= 0:
        continue
    ds_path = str(ds.path)
    if os.path.abspath(ds_path) == primary_path:
        # Primary was run already; record its combined and skip re-running
        per_dataset_combined[ds.id] = result["combined"]
        continue
    if not os.path.isdir(ds_path):
        print(f"combined_{ds.id}: MISSING (no such directory: {ds_path})")
        continue

    print(f"\n=== Secondary eval: {ds.id} (DSP-only) ===")
    try:
        ds_result = evaluate(ds_path)
        per_dataset_combined[ds.id] = ds_result["combined"]
        print(f"combined_{ds.id}: {ds_result['combined']:.6f}  "
              f"(recall={ds_result['recall']:.2f}, "
              f"precision={ds_result['precision']:.2f}, "
              f"clean_fp={ds_result['clean_fp']})")
    except Exception as _e:
        per_dataset_combined[ds.id] = 0.0
        print(f"combined_{ds.id}: ERROR ({type(_e).__name__}: {_e})")
    print(f"=== End secondary eval: {ds.id} ===\n")

if args.with_classifier:
    from ml_eval import evaluate_with_classifier
    ml_result = evaluate_with_classifier(result, args.data_dir)
    if ml_result.get("bound_exceeded"):
        singing_full = 0.0
    elif ml_result.get("combined_full") is not None:
        singing_full = ml_result["combined_full"]
    else:
        singing_full = result["combined"]
    # Use classifier-filtered score for the primary dataset (id must match singing)
    per_dataset_combined["singing"] = singing_full

# Aggregate: geometric mean with a floor to avoid zero-collapse on new datasets.
agg = aggregate_combined(per_dataset_combined, method="geometric", floor=0.01)
for name, val in per_dataset_combined.items():
    print(f"combined_{name}: {val:.6f}")
print(f"combined_mean: {agg['combined_mean']:.6f}")
print(f"combined_min:  {agg['combined_min']:.6f}")
print(f"combined: {agg['combined']:.6f}")
```

Notes on the diff:

- **Preserves the last `combined:` line** (autoresearch parsers take the last one).
- **Primary singing gets classifier-filtered score**, since that's what the ML
  pass produces. Speech datasets stay DSP-only until you opt them in (future work).
- **Pre-existing `combined_dsp:` prints inside `evaluate_with_classifier`** remain
  unchanged — useful diagnostics.
- **`combined_speech` renamed to `combined_<id>`** — the fixed "speech" label
  goes away in favor of the registry id.

---

## Phase-1 metric update (forensic_combined, 15-cell + FP multiplier)

Once the Phase-1 implementation lands (spec:
`deep-interview-gbm-main-forensic-metric.md`), the aggregate line should
switch from `aggregate_combined` (3-cell GM over per-dataset combined) to
`forensic_combined` (15-cell GM × smooth FP multiplier). The rest of the
evaluate.py block is unchanged; just swap the final aggregator.

```python
from dataset_registry import DATASETS, forensic_combined

# per_dataset_results is produced by iterating DATASETS and calling
# evaluate(ds.path) — same as above — but now we capture per-regime / per-tier
# F1 from each dataset's ground_truth breakdown (boundary_energy + tier).
# eval_by_regime.py already does this breakdown; migrate its logic into
# evaluate.py inline so the per-cell numbers land in per_cell_results.

per_cell_results: dict[str, float] = {}

for ds in DATASETS:
    if ds.eval_weight <= 0:
        continue
    ds_result = evaluate(str(ds.path))      # existing
    by_regime = ds_result.get("by_regime", {})   # NEW: evaluate.py must emit this
    # by_regime shape: {"t1_random": {"tp":..,"fp":..,"fn":..}, "t1_quiet_matched": {...}, ...}
    for regime_key, counts in by_regime.items():
        f1 = _f1_from_counts(counts)
        per_cell_results[f"{ds.id}_{regime_key}"] = f1   # 4 cells per dataset
    per_cell_results[f"{ds.id}_clean_score"] = ds_result["clean_score"]   # 1 cell per dataset

total_clean_fp = sum(r["clean_fp"] for r in per_dataset_results.values())
agg = forensic_combined(per_cell_results, clean_fp=total_clean_fp, max_allowed=15)

# Print per-cell diagnostics
for name, val in per_cell_results.items():
    print(f"combined_{name}: {val:.6f}")
print(f"combined_gm_cells: {agg['combined_gm_cells']:.6f}")
print(f"fp_multiplier:     {agg['fp_multiplier']:.6f}")
print(f"total_clean_fp:    {total_clean_fp}")
print(f"n_cells_included:  {agg['n_cells_included']}")
print(f"combined: {agg['combined']:.6f}")
```

**Assumptions this block depends on:**

1. Every eligible dataset's `ground_truth.json` has `boundary_energy` populated
   on all spliced entries (korean + english already done; singing regen scheduled
   under PRD story US-302).
2. `evaluate()` emits a `by_regime` key in its result dict with counts broken
   down by `(tier, boundary_energy)`. If `evaluate()` is not extended, the
   per-regime breakdown can be computed inline using `eval_by_regime.py` logic.
3. `max_allowed=15` matches the existing DSP FP bound. Tighten when forensic
   acceptance requires it.

## baseline_metrics.json update (Phase-1)

Two baselines will coexist:

1. `combined` — old 3-cell aggregate (legacy baseline, kept for verify_agent
   continuity during the transition)
2. `combined_forensic` — new 15-cell × FP aggregate (the Phase-1 target)

Proposal — add a new JSON field rather than replacing:

```json
{
  "timestamp": "2026-04-17T<apply>",
  "git_sha": "REPLACE_AFTER_COMMIT",
  "combined": 0.704,
  "combined_forensic": <recorded at Phase-1 completion>,
  "combined_gm_cells": <recorded>,
  "fp_multiplier": <recorded>,
  "per_cell": { ... 15 cell values ... },
  "notes": "Phase-1 (gbm-main-forensic-metric) implementation-done baseline. Phase-2 target combined_forensic ≥ 0.65 via autoresearch."
}
```

This way `verify_agent.py`'s existing check continues unchanged while the
richer `combined_forensic` joins it as a diagnostic.

## baseline_metrics.json update

The new `combined` is a geometric mean, not a singing-only number. Today's
values:

```
singing (full):   0.742
korean (DSP):     0.270
english (DSP):    ~0.05
geometric (floor 0.01): ≈ 0.204
```

Replacement `autoresearch/baseline_metrics.json`:

```json
{
  "timestamp": "2026-04-17T14:00:00Z",
  "git_sha": "REPLACE_AFTER_COMMIT",
  "combined": 0.204,
  "combined_singing": 0.742,
  "combined_korean": 0.270,
  "combined_english": 0.049,
  "combined_dsp": 0.412,
  "splice_f1": 0.588,
  "clean_score": 0.700,
  "precision": 0.56,
  "recall": 0.62,
  "dsp_clean_fp": 15,
  "classifier_cv_f1": 0.851,
  "note": "Metric semantics changed from singing-only `combined_full` to geometric mean across all DATASETS in dataset_registry.py. Old baseline was 0.719 (singing-only)."
}
```

## Verify after applying

```
uv run python splice/evaluate.py --with-classifier
```

Expected:
- `combined_singing`, `combined_korean`, `combined_english` printed
- `combined_mean`, `combined_min`, `combined` printed
- `combined` value is the geometric mean (~0.20 today)
- Total elapsed stays under 240s

Then:
```
uv run python autoresearch/verify_agent.py --agent-name after-patch \
    --reported-combined <actual>
```
should return HIGH (delta < 0.15 from the updated baseline).

## Rollback

`git revert` the splice/evaluate.py + autoresearch/baseline_metrics.json commit.
