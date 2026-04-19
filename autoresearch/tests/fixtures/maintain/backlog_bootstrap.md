# Enhancement Backlog

**Purpose:** Track wrapper/pipeline enhancement requests.

**Format:** One H2 per request.

## per-class-oof-f1

- **status:** pending
- **first_seen:** abc1234
- **last_seen:** abc1234
- **request_count:** 1
- **category:** observability
- **risk:** low (train_classifier.py emit expansion + prompt composer additional line)
- **excerpt:** Per-class OOF F1 in CURRENT STATE alongside the aggregate weighted F1 — currently I only see the aggregate.
- **notes:** Pure additive change; no risk.

## shap-delta-block

- **status:** pending
- **first_seen:** def5678
- **last_seen:** def5678
- **request_count:** 2
- **category:** observability
- **risk:** medium (needs SHAP rollup comparison logic + prompt composer change)
- **excerpt:** Surface a SHAP-DELTA block in CURRENT STATE for the most recent KEPT classifier vs prior KEPT classifier.
- **notes:** Implementation requires snapshot-at-keep mechanism.

## verify-agent-replay

- **status:** pending
- **first_seen:** abc1234
- **last_seen:** def5678
- **request_count:** 3
- **category:** new-subcommand
- **risk:** high (new subcommand, git state manipulation)
- **excerpt:** verify_agent.py replay commit against HEAD to re-run evaluate.py for a prior hypothesis.
- **notes:** High risk, needs ralplan consensus.

## per-domain-file-stats

- **status:** deferred
- **first_seen:** abc1234
- **last_seen:** abc1234
- **request_count:** 1
- **category:** new-script
- **risk:** low (new script in scripts/; no existing code paths changed)
- **excerpt:** A scripts/per_domain_file_stats.py emitting mean HPR/flatness/voicing per domain.
- **notes:** Deferred pending other items.

---

# Maintainer cron policy

Low-risk items may be auto-implemented.
