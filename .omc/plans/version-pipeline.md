# Plan: Detector-Classifier Version Pipeline

**Spec:** `/audio-splice-detector/.omc/specs/deep-interview-version-pipeline.md`
**Complexity:** LOW (~15 lines across 2 files, no new scripts)
**Date:** 2026-04-15

## RALPLAN-DR (Brief)

**Principles:**
1. Do not break the running autoresearch loop
2. Background work must not block iteration throughput
3. Version pairs (detector + classifier) must be atomic and rollback-able

**Decision Drivers:**
1. Autoresearch is actively running in tmux -- zero downtime required
2. Classifier retraining (25s CPU) overlaps with claude -p network wait -- no contention
3. Minimal line count keeps maintenance burden near zero

**Options:**
- **A: Inline bash in run_autoresearch.sh (chosen)** -- Add versioning logic directly in the keep-pending handler. Pros: no new files, ~10 lines, trivial to audit. Cons: shell-based JSON manipulation is brittle.
- **B: Separate version_manager.py script** -- Cleaner JSON handling via Python. Pros: proper JSON lib. Cons: adds a new file (violates constraint), more surface area, overkill for the data shape.
- Option B invalidated by explicit constraint "no new files except versions.json."

**ADR:**
- **Decision:** Inline bash + python one-liner for JSON updates
- **Why:** Matches existing pattern in run_autoresearch.sh (inline uv run python -c), minimizes diff, respects no-new-files constraint
- **Consequences:** JSON manipulation in shell is less robust than a dedicated script; mitigated by the simple flat structure of versions.json
- **Follow-ups:** If version count exceeds ~50, consider extracting to a script

---

## Context

- `run_autoresearch.sh` has a `keep-pending` case that runs `verify_agent.py` and logs results
- `train_classifier.py` at `.omc/classifier/train_classifier.py` trains on patches generated from `detector.py`
- No git tags of form `detector-v*` exist yet -- version numbering starts at 1
- `$reported` variable (combined score) is already extracted in the keep-pending handler

## Guardrails

**Must Have:**
- Existing autoresearch loop continues to work identically for discard/unknown/verify-fail paths
- Background retrain runs via `&` so next iteration starts immediately
- Rollback restores both detector.py and the classifier joblib

**Must NOT Have:**
- No changes to the claude prompt or experiment logic
- No changes to verify_agent.py
- No blocking operations in the keep-pending hot path

---

## Task Flow

### Step 1: Add versioning to keep-pending handler in run_autoresearch.sh

**Where:** Inside `keep-pending)` case, after the `VERIFIED KEEP` log line (line ~88)

**What:**
- Compute VERSION from existing git tags: `$(git tag -l 'detector-v*' | wc -l) + 1`
- `git tag "detector-v$VERSION"`
- `cp detector.py ".omc/classifier/detector_v${VERSION}.py"`
- Inline python one-liner to create/update `.omc/classifier/versions.json` with version entry (version, git_tag, git_sha, detector_snapshot, classifier, combined_dsp, combined_full=null, timestamp)
- Background: `DETECTOR_SNAPSHOT=... uv run python .omc/classifier/train_classifier.py && cp ... &`

**Acceptance Criteria:**
- [ ] After a verified keep, `git tag -l 'detector-v*'` shows the new tag
- [ ] `.omc/classifier/detector_v{N}.py` exists as a snapshot
- [ ] `versions.json` contains the new entry with correct combined_dsp score
- [ ] Background train starts and does not block the `sleep 2` before next iteration
- [ ] Discard / verify-fail paths are completely unchanged

### Step 2: Add DETECTOR_SNAPSHOT support to train_classifier.py

**Where:** `train_classifier.py`, near the top where `PROJECT_ROOT` is defined, and in `load_data()` or wherever detector.py is referenced for patch generation

**What:**
- Read `DETECTOR_SNAPSHOT = os.environ.get("DETECTOR_SNAPSHOT")`
- If set, the generate_patches step should use the snapshot file instead of live `detector.py`
- Note: train_classifier.py itself does not import detector.py -- it reads from `patches/` dir. The snapshot is used by `generate_patches.py`. Check whether train_classifier.py calls generate_patches.py or if they are separate. If separate, the env var pass-through may need to go in the background command chain instead.

**Acceptance Criteria:**
- [ ] `DETECTOR_SNAPSHOT=/path/to/snapshot uv run python .omc/classifier/train_classifier.py` uses the snapshot for patch generation
- [ ] Without the env var, behavior is identical to current (backward compatible)

### Step 3: Add rollback subcommand to run_autoresearch.sh

**Where:** Bottom of file, in the `case` statement alongside start/stop/status

**What:**
- New case `rollback)` that takes `$2` as version number
- `git checkout "detector-v$2" -- detector.py`
- `cp ".omc/classifier/classifier_v$2.joblib" ".omc/classifier/fp_classifier.joblib"`
- Print confirmation with version number
- Error if tag or classifier file doesn't exist

**Acceptance Criteria:**
- [ ] `./run_autoresearch.sh rollback 3` restores detector.py from tag and classifier from snapshot
- [ ] Error message if version doesn't exist
- [ ] Usage line updated to include rollback

### Step 4: Add version info to status subcommand

**Where:** `status)` case in run_autoresearch.sh

**What:**
- Read `latest` from `.omc/classifier/versions.json` (if exists)
- Print current detector version number

**Acceptance Criteria:**
- [ ] `./run_autoresearch.sh status` shows "Detector version: N" when versions.json exists
- [ ] Gracefully shows nothing extra if versions.json doesn't exist yet

---

## Success Criteria

1. Full autoresearch loop runs without any change in behavior for non-keep paths
2. Every verified keep produces: git tag + detector snapshot + versions.json entry + background classifier train
3. Rollback restores a complete detector+classifier pair
4. Status shows current version
5. Total diff is ~15 lines across run_autoresearch.sh and train_classifier.py

## Open Questions

- Does `generate_patches.py` import from `detector.py` directly, or does `train_classifier.py` call it? This determines where DETECTOR_SNAPSHOT env var needs to be consumed. Executor should check `.omc/classifier/generate_patches.py` imports before implementing Step 2.
