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
