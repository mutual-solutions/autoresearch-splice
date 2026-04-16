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
- Full evaluation must complete in under 120 seconds.
- `evaluate.py` and `program.md` are protected -- only the human modifies them. The autoresearch agent must never modify them.
- `detector.py` (DSP) and `ml_config.py` (ML params) are edited for experiments.

## Protected Files

The verification system guards these from modification:
`evaluate.py`, `data/spliced/*`, `.omc/coordination/manifest.json`, `.omc/coordination/preflight.py`

## Preflight

Before evaluation, run: `uv run python .omc/coordination/preflight.py`
This verifies dataset integrity (file counts, ground truth hash).
It runs automatically as part of verify_agent.py.

## Baseline

Baseline metrics are stored in `.omc/coordination/baseline_metrics.json`.
Anomaly detection flags deltas > 0.15 from baseline combined score.
