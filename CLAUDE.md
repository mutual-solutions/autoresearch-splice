# autoresearch-splice -- Agent Rules

## Mandatory Verification Protocol

Every subagent MUST be verified before its results are accepted.
After any agent reports a `combined` score, run:

```
uv run python .omc/coordination/verify_agent.py --agent-name <name> --reported-combined <score>
```

This performs 4 checks: metric re-run, git diff audit, anomaly detection, and preflight.

## Confidence Levels

- **HIGH** (all PASS): Accept the agent's work.
- **MEDIUM** (any WARN): Accept with caution; review anomaly details.
- **LOW** (any FAIL): REJECT. Revert the agent's commits (`git reset --hard` to last known good) and investigate. Do NOT proceed with LOW confidence results.

## Constraints (from program.md)

- NO neural networks, NO GPU. scipy/librosa/numpy only.
- Every detection must be explainable (statistical test or spectral anomaly).
- Full evaluation must complete in under 300 seconds (cross-dataset GM).
- Held-out test evaluation must complete in under 1200 seconds (20 min).
- `evaluate.py` and `program.md` are protected -- only the human modifies them. The autoresearch agent must never modify them.
- `detector.py` (GBM thresholds + sliding window geometry), `features.py` (feature set), and `.omc/classifier/train_classifier.py` (GBM hyperparameters) are edited for experiments. After features.py or hyperparameter edits, retrain with `uv run python .omc/classifier/train_classifier.py`.

## Data layout (post-unification)

```
data/
├── eval/{singing,korean,english}/    # primary metric — evaluate.py iterates these
├── train/{singing,korean,english}/   # train_classifier.py consumes these; disjoint sources
├── test/{singing,korean,english}/    # held-out, longer duration envelope, 20-min budget
└── sources/                          # raw audio pools (not consumed at eval time)
    ├── singing_wav/                  # 101 WAV files
    ├── zeroth-korean/                # 115 speakers
    └── LibriSpeech/dev-clean/        # 40 speakers
```

All splits share the same 20 tier1 + 20 tier2 + 20 clean layout and identical
encoding mix (WAV / FLAC / Opus / MP3-128). Duration envelope varies: eval /
train use 30-120s files, test uses 60-300s files. Source audio is split
deterministically across train / eval / test pools with **zero overlap** at
the file (singing) or speaker (korean, english) level.

## Protected Files

The verification system guards these from modification:
`evaluate.py`, `program.md`, `data/eval/**/*`, `data/test/**/*`, `.omc/coordination/manifest.json`, `.omc/coordination/preflight.py`

## Preflight

Before evaluation, run: `uv run python .omc/coordination/preflight.py`
This verifies dataset integrity (file counts, ground truth hash).
It runs automatically as part of verify_agent.py.

## Baseline

Baseline metrics are stored in `.omc/coordination/baseline_metrics.json`.
Anomaly detection flags deltas > 0.15 from baseline combined score.

## Retest (US-514)

After a pipeline-bug fix, discards from the affected window can be
systematically replayed against the fixed evaluator. This recovers
legitimate hypotheses that were discarded for the wrong reason. The loop
MUST be stopped before retest starts — retest shares the eval corpus
lifecycle and the `baseline_metrics.json` writer with the loop.

Operator workflow:

1. Stop the loop:
   ```
   ./run_autoresearch.sh stop
   ```
2. Decrypt the eval corpus into a `/tmp` directory and export
   `OMC_EVAL_DATA_ROOT`. Retest does **not** re-decrypt; the operator
   owns the decrypt lifecycle.
   ```
   uv run python scripts/eval_crypto.py decrypt --keep
   export OMC_EVAL_DATA_ROOT=<path-from-decrypt-tail>
   ```
3. Dry-run first to enumerate candidates and predicted deltas. <5 s;
   writes `.omc/retest-report.md`; no mutation.
   ```
   uv run python .omc/coordination/verify_agent.py --retest <from-sha> --dry-run
   ```
4. Live run when the preview looks right. Replays discards in
   chronological order against a disk-sourced rolling baseline and
   invokes `run_autoresearch.sh _keep_path` on any real improvement.
   ```
   uv run python .omc/coordination/verify_agent.py --retest <from-sha> [--limit N]
   ```

Outcomes (in the report): `recovered`, `still-lower`, `conflict`,
`eval-crash`, `retrain-crash`, `verify-fail`, `missing`,
`corpus-purged`, `dry-run-skipped`.

## Retest sentinel

Between the main-tree `git cherry-pick` and the baseline commit,
`--retest` writes `.omc/retest-in-progress` as a SIGKILL / OOM / crash
guard. The wrapper's `start`, `_loop`, and `_loop_restart` refuse to
run while the sentinel exists so a partial recovery cannot be
overwritten by a fresh hypothesis iteration.

If you find the sentinel on a stopped tree:

1. Read `.omc/retest-report.md` for the last attempted candidate.
2. `git log --oneline -10` — check whether the cherry-pick landed, the
   baseline commit landed, or neither.
3. If the cherry-pick is partial or wrong, `git reset --hard ORIG_HEAD`
   (the cherry-pick sets ORIG_HEAD) to roll it back.
4. Remove the sentinel so the loop can resume:
   ```
   rm -f .omc/retest-in-progress
   ```

Retest does NOT auto-delete the sentinel on abnormal exits — that
would defeat the SIGKILL hole. The sentinel is only removed on the
one happy path inside `_try_recover` after the `baseline:` commit is
confirmed to have landed.
