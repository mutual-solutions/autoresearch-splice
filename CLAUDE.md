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

## Unified logging (US-515 phases 1 + 2)

All emission — Python **and** bash wrapper — goes through
`.omc/coordination/logger.py`. Canonical log file:
`.omc/logs/autoresearch.jsonl` (gitignored; rotation-aware — 50 MB per
roll, keeps up to 10 rolls). Freeform subprocess stdout/stderr lives
alongside it at `.omc/logs/child-stderr.log` so the JSONL stream stays
structured.

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".omc", "coordination"))
from logger import get_logger
log = get_logger("detector.gbm")
log.emit("INFO", "diag.gbm.dedupe", before=120, after=80)
```

The `.omc/` directory has a leading dot, so `python -m omc.coordination.*`
is impossible — always invoke via direct file paths
(`uv run python .omc/coordination/tests/test_smoke_iteration.py`).

Level policy: `DEBUG` / `INFO` / `WARN` / `ERROR` / `CRITICAL`. Event names
are dotted and live in the taxonomy docstring at the top of `logger.py`
(`eval.*`, `classifier.*`, `retest.*`, `diag.*`, `pipeline.*`,
`wrapper.*`, `note.*`, `tunable.*`, `notebook.*`). `validate_logs.py --parse`
warns (not fails) on unknown events.

### Bash emission (phase 2)

`run_autoresearch.sh` emits structured events via its `_log` helper, which
shells out to `.omc/coordination/log_cli.py`:

```bash
_log LEVEL SUBSYSTEM EVENT [key=value ...]
# e.g.
_log INFO wrapper iteration.start consecutive_discards="$n"
```

Reserved kv flags: `claude_visible=true`, `oracle_sensitive=true`. Values
are always strings — use the Python logger directly when you need native
numeric types. Logging failures are suppressed; `_log` never blocks the
loop. Subprocess stdout/stderr (git, retrain, evaluate.py) redirects to
`$CHILD_STDERR_LOG` (= `.omc/logs/child-stderr.log`), NOT to the JSONL
log, so it cannot corrupt the structured stream.

### Oracle redaction

`logger.emit()` redacts oracle-denylist tokens (`combined=`, `splice_f1=`,
`clean_score=`, `combined_<domain>=`) from string kv values when either
`oracle_sensitive=True` is passed OR the event name starts with
`retest.diagnose.` (auto-enabled). Default is no redaction —
wrapper-owned events carry real metrics.

### Disabled mode

Set `OMC_LOGGER_DISABLED=1` to no-op every `emit()`. Used by tests that
need to import caller modules without a real log sink. The US-515 smoke
fixture deliberately does NOT set this — redaction/parse gates need
real emission.

### Gates

- `uv run python scripts/validate_logs.py --audit` — grep audit. Phase-1
  tier: 0 residual `_diag(` in migrated Python sources. Phase-2 tier: 0
  residual `echo ... >> $LOG_FILE` in `run_autoresearch.sh` (the wrapper
  must emit exclusively via `_log`).
- `uv run python scripts/validate_logs.py --parse .omc/logs/autoresearch.jsonl`
  — schema-parse every line; warn on events outside the taxonomy.
- `uv run python .omc/coordination/tests/test_smoke_iteration.py` —
  pre-merge smoke stub. Four gates: fixture emission, parse, reader
  non-empty, redaction.

### evaluate.py one-time maintainer exemption

Phase 1 migrated `evaluate.py` metric emissions (`splice_f1`, `clean_score`,
`combined`, opus32k, aggregate, per-dataset, FP/crossfade/loc breakdowns)
to the unified logger. The PROTECTED-FILE EXEMPTION is a one-time
maintainer edit — `verify_agent`'s diff audit only fires during
autoresearch hypothesis iterations, not maintainer commits, so this does
not collide with the agent loop. The `RESULTS_TSV:` string line remains
as a carve-out because `run_autoresearch.sh` greps it at 5 sites; that
migrates in phase 2 alongside the wrapper.

### Carve-outs (remaining after phase 2)

- `evaluate.py` `RESULTS_TSV:` line (`run_autoresearch.sh` greps at 4 sites;
  phase 3a). Mirrored to `.omc/logs/child-stderr.log` via `tee` but the
  authoritative parse target remains `.omc/last_eval.log`.
- `evaluate.py` `combined_{ds.id}: ERROR (...)` line (parsed by
  `verify_agent.run_diagnose`; phase 3c).
- `verify_agent._write_retest_report` — operator-facing Markdown (phase 3b).
- `.omc/last_reflection.md` — claude reflection scratch (IPC between the
  claude subprocess and `_append_note`, not a log). Truncated each
  iteration; remains out of the unified log.

### Phase-2 completion

Phase-2 migration landed with this PR. Scope:

- `_log` bash helper + `.omc/coordination/log_cli.py` CLI wrapper
- All 47+ `echo ... >> $LOG_FILE` sites in `run_autoresearch.sh` rewritten
  as `_log` calls
- Subprocess stdout/stderr redirected to `.omc/logs/child-stderr.log`
- Legacy-shim `_iter_legacy_events` deleted from `scripts/log_reader.py`
  (readers now pure JSONL)
- `scripts/tunable_frontier.py` emits `tunable.frontier.snapshot`; wrapper
  captures its stdout block directly for prompt injection
- `scripts/notebook_digest.py` emits `notebook.digest.compacted`
- Legacy files deleted: `.omc/autoresearch.log`, `.omc/autoresearch-debug.log`,
  `.omc/autoresearch-debug.log.1`, `.omc/tunable_frontier.txt`
- `scripts/validate_logs.py --audit` gained bash tier
- `.omc/coordination/tests/test_log_readers.py` rewritten for native JSONL
