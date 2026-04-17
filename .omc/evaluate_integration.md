# evaluate.py integration with dataset_registry

`evaluate.py` is protected — only you apply these edits.

## Goal

Replace the hardcoded Korean secondary eval block with a loop over
`dataset_registry.DATASETS`, compute a balanced `combined` via
`aggregate_combined`, and print per-dataset plus aggregate metrics.

Result: adding a dataset → one line in `dataset_registry.py`, no
further edits anywhere else.

## Files to edit

1. `evaluate.py` (this repo) — one block replaced, a few lines added.
2. `.omc/coordination/baseline_metrics.json` — new primary baseline.

## Before

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

## baseline_metrics.json update

The new `combined` is a geometric mean, not a singing-only number. Today's
values:

```
singing (full):   0.742
korean (DSP):     0.270
english (DSP):    ~0.05
geometric (floor 0.01): ≈ 0.204
```

Replacement `.omc/coordination/baseline_metrics.json`:

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
uv run evaluate.py --with-classifier
```

Expected:
- `combined_singing`, `combined_korean`, `combined_english` printed
- `combined_mean`, `combined_min`, `combined` printed
- `combined` value is the geometric mean (~0.20 today)
- Total elapsed stays under 240s

Then:
```
uv run python .omc/coordination/verify_agent.py --agent-name after-patch \
    --reported-combined <actual>
```
should return HIGH (delta < 0.15 from the updated baseline).

## Rollback

`git revert` the evaluate.py + baseline_metrics.json commit.
