# RALPLAN — korean-iter1 corpus pivot + framework reset

**Source spec**: `.omc/specs/deep-interview-korean-iter1-reset.md` (LOCKED, ambiguity 7.5%)
**Mode**: deliberate consensus (destructive pivot)
**Branch target**: `autoresearch/korean-iter1` (off `master`); `autoresearch/apr15` archived
**Estimated total wall-clock**: ~4–6 hours engineer time + ~12 min compute (regen)

---

## v2 changelog

This revision applies the 7 fixes from the architect (`REVISE_MINOR`) + critic (`ITERATE`) consolidation. No other content was altered. Locked decisions (Principles, Drivers, Tensions A/B/D/E/F) are unchanged.

| # | Item | Section(s) changed | One-line summary |
|---|------|--------------------|------------------|
| 1 | Sentinel before destructive ops | §1 Tension E (sequence renumbered), §3 (new step 3.5; reversibility column updated) | Sentinel `.omc/korean-iter1-regen-in-progress` is now created BEFORE step 4's `git mv`/`mv` ops, not after. |
| 2 | Split step 11 into 11a + 11b | §3 (step 11 split; step 12 renumbered) | 11a = maintainer migration of `program.md`/`manifest.json`/`preflight.py` only (NO baseline write); 11b = real-baseline write atomically AFTER step 12 eval and BEFORE the gating `--verify`. |
| 3 | Default label `"unknown"` not `"cross_voice"` | §1 Tension C, §3 step 8, §4 unit tests | Unlabeled detector emissions default to `"unknown"`; per-class F1 in `splice_class_breakdown` excludes `unknown`; overall `combined = boundary_f1` still counts them (label-blind aggregate). |
| 4 | Ground-truth hash field rename (Option B) | §1 Tension F, §3 step 11a, §4 preflight test | Regenerator synthesizes a per-corpus `ground_truth.json` aggregating all `boundaries.json` files; `preflight.py` keeps existing logic; manifest field name unchanged. |
| 5 | Diff-audit window guarantee documented | §5 (new invariant), §6 ADR Consequences (Positive) | Maintainer commits land BEFORE `run_autoresearch.sh start`; the supervisor's `OMC_HEAD_BEFORE..HEAD` audit window is per-iteration and naturally excludes pre-loop maintainer history. |
| 6 | First-iteration `_guarded_reset` anchor (Option B — operator tag) | §3 (new step 11c), §4 anchor test | Operator runs `git tag iter1-anchor HEAD` after step 11b and before step 12; documented as the safe first-iteration discard target. The wrapper's existing `head_before` capture per-iteration already targets this commit on iter 1, but the tag is a defensive marker for manual recovery. |
| 7 | `MIGRATE-PROTECTED` exemption recognition (Option A) | §1 Principle 5, §3 step 8 + step 11a, §5 invariant | The supervisor's diff audit has NO commit-message-prefix exemption — verified by reading `autoresearch/supervisor_agent.py:102-162`. The actual exemption mechanism is the per-iteration `OMC_HEAD_BEFORE..HEAD` window. The `MIGRATE-PROTECTED:` prefix is retained as a human-readable convention only (greppable for operator review); enforcement is purely "land before loop start." |

---

## §1. RALPLAN-DR Summary

### Principles (5)

1. **Preserve apr15 history queryability** — every commit, every results.tsv row, every detector vN snapshot must remain reachable by SHA after the pivot. No `git branch -D`. No `rm` of `.omc/classifier/detector_v*.py`. Use `git tag` + archive moves, never deletions.
2. **No destructive op without sentinel/checkpoint** — the regenerate script touches a multi-GB dataset and the verifier diffs against baseline; partial state must be either complete or trivially rolled back. Stage destructive ops behind a sentinel file (`.omc/korean-iter1-regen-in-progress`) modeled on `.omc/retest-in-progress`. **(Item 1)** The sentinel is in place from step 3.5 through step 12; only removed on the happy path.
3. **Edit-then-augment for forensic realism** — same-voice edits cut the studio-clean source at silence boundaries, *then* the joined audio enters the noise/RIR/codec chain. This matches the real-world threat model ("edit a wav, re-encode for delivery") and avoids phase/decay-tail discontinuity at the join site.
4. **Reproducibility via on-disk artifact, not on-demand regeneration** — the augmented corpus is bytes-on-disk under `data/eval/korean_iter1/{train,eval,test}/` with `(file_hash, augment_seed)` in a manifest. The wrapper's eval loop must NOT regenerate; that blows the 240 s budget by ~3000%.
5. **Verifier 4-check audit must remain green from iteration 1** — every protected-file edit lands in a maintainer commit BEFORE `run_autoresearch.sh start`. The supervisor's diff-audit window (`OMC_HEAD_BEFORE..HEAD`, captured per-iteration) excludes pre-loop history by construction; this is the actual exemption mechanism. The `MIGRATE-PROTECTED:` commit-message prefix is a human-readable convention for operator review, NOT a code-recognized flag. **(Item 7)**

### Decision Drivers (top 3)

1. **Loop must restart cleanly within 1 day of operator effort.** All destructive + regeneration steps complete unattended in one sitting; `run_autoresearch.sh start` works on attempt 1.
2. **Verifier 4-check audit (`supervisor_agent.py --verify`) must remain green** across the schema migration. Anomaly detection compares delta against `baseline_metrics.json` — the new baseline must be written before the first hypothesis iteration so the delta is 0, not 0.589.
3. **No manual intervention required mid-regenerate.** The regenerator is `Ctrl-C`-safe (resumable via per-file output existence check) and emits unified-logger progress every 100 files so the operator can monitor via `scripts/log_monitor.py`.

### Tension Resolutions (locked choices)

#### Tension A — Augmented corpus on-disk vs on-demand
- **Chosen: on-disk, ~4 GB total under `data/eval/korean_iter1/`.**
- Options considered:
  - **(chosen) On-disk pre-augmented**: ~4 GB Opus, exact-byte reproducible, eval starts in <2 s. Cost: 4 GB disk, one-time 12-min regen.
  - **On-demand per-iteration**: saves disk, kills budget. ~12 min augmentation × 60 sampled files would dominate the 240 s budget by 3×. INVALIDATED by budget driver.
  - **On-demand per-file with cache**: hybrid — first iteration is slow, steady state cached. Cache invalidation tied to seed, but seed is hash(file_id) so static — degenerates into on-disk-after-warmup with worse first-iter UX. INVALIDATED as strictly inferior to chosen option.

#### Tension B — Edit-then-augment vs augment-then-edit
- **Chosen: edit-then-augment.**
- Options considered:
  - **(chosen) Edit-then-augment**: cut at silence in clean source, join, then noise+RIR+codec applied to joined audio. Forensic realism (matches "edit a wav and re-encode for delivery"). Smooth, undetectable join in waveform domain; the codec's frame boundaries land randomly relative to the cut. The detector must learn signal-domain cues (spectral micro-discontinuity, MDCT seam artifacts), not noise-tape phase skips.
  - **Augment-then-edit**: cut after augmentation. The cut produces a phase / decay-tail discontinuity at the join (noise tape skips, RIR tail truncates). Trivially detectable by frame-energy delta; the detector would over-fit to a non-realistic cue. INVALIDATED as not matching the threat model.

#### Tension C — Detector contract migration **(Item 3 — REVISED)**
- **Chosen: hard switch to `list[(time_s, label)]`. No flag-gated extension. Backward-compat at consumer boundary uses default label `"unknown"`, NOT `"cross_voice"`.**
- Options considered:
  - **(chosen) Hard switch with `"unknown"` default**: `detect_splices(audio, sr) -> list[tuple[float, str]]` where label ∈ `{"cross_voice", "same_voice_edit", "unknown"}`. The 167-row apr15 history is incompatible — irrelevant since the new branch is fresh. `splice/evaluate.py`'s scoring loop expects labeled tuples; unlabeled or string-malformed emissions get auto-labeled `"unknown"` at the eval boundary. **Critical accounting rule**: per-class F1 in `splice_class_breakdown` (`cross_voice_f1`, `same_voice_edit_f1`) is computed **excluding** `unknown` predictions from both numerator and denominator on the prediction side — so an agent that ships a label-blind detector cannot inflate `cross_voice_f1` by guessing the dominant class. Overall `combined = boundary_f1` still counts `unknown` predictions normally (label-blind aggregate). This preserves §2 Pre-mortem Scenario 3's voice-overfit detection signal: per-class F1 reflects only deliberate, label-aware predictions.
  - **Default `"cross_voice"`**: silently inflates `cross_voice_f1` and deflates `same_voice_edit_f1`. Defeats voice-overfit detection. INVALIDATED.
  - **Flag-gated `return_labels=False`**: preserves API for hypothetical apr15 merge. Adds a flag callers must thread through. The scoring loop still needs to handle two output shapes. INVALIDATED — the apr15 branch is archived as a tag, not a merge candidate; no caller will ever invoke the old shape post-pivot.
  - **Side-channel labels via second return value**: `detect_splices` returns `(times, labels)` tuple. Larger blast radius, more places to break. INVALIDATED as needlessly invasive.

#### Tension D — Voice-pair holdout selection
- **Chosen: test={DaeBuHo, Ondo}, eval={Sunwoo, Joon}, train={Kanna, ChloeCha, DangchanYeo, Donghyun, Eunha, Minho, Minwoo}.** This is the spec's suggestion; lock it for this iteration.
- Justification: split must satisfy (i) gender diversity in held-out, (ii) age/accent diversity in held-out, (iii) ≥1100 files per held-out voice for sample stability.
  - **Step 0 verification** (added to plan as gate): the regenerator script's first action enumerates voices and prints a `voice_diversity_check` table to the unified log: `voice_name, gender, file_count, mean_duration_s, mean_F0_Hz`. If test+eval voices fail to span both genders, the script exits with a clear error before any data is written. Operator confirms the table or overrides via `--voices-test=A,B --voices-eval=C,D`.
- Options rejected:
  - **Per-loop randomization**: defeats reproducibility. Verifier delta would oscillate by voice luck. INVALIDATED by driver 2.
  - **Adversarial split (highest-acoustic-distance pair as test)**: would maximize generalization signal but also maximize variance. Not worth the iteration cost in a fresh framework. INVALIDATED — keep simple, verify after first 3 iterations whether generalization gap warrants this in iter-2.
  - **5-fold CV**: 5× regen cost, 5× eval cost. INVALIDATED by 240 s budget.

#### Tension E — Order of destructive ops **(Item 1 — REVISED)**
- **Chosen sequence** (every step is reversible until step 6; sentinel covers steps 3.5 → 12):
  1. **Tag archive**: `git tag apr15-snapshot-2026-04-25 autoresearch/apr15`. Pure metadata, no destructive change. **Pushed to origin** (`git push origin apr15-snapshot-2026-04-25`) so the snapshot survives a local-disk loss.
  2. **Tag archive of HEAD work-in-progress** (current dirty tree): operator stashes uncommitted edits to `apr15-wip-stash`, commits any salvageable .omc/classifier/detector_v*.py snapshots they want preserved, and tags `apr15-final-2026-04-25`. The current dirty `M scripts/regenerate_datasets.py` and `M splice/dataset_registry.py` are committed first or stashed.
  3. **Branch off master**: `git checkout master && git pull && git checkout -b autoresearch/korean-iter1`. apr15 ref still exists.
  3.5. **Sentinel up (NEW)**: write `.omc/korean-iter1-regen-in-progress` BEFORE any `git mv`/`mv` op. The sentinel guards every destructive step from here through step 12's smoke-iteration completion. Wrapper's `start`, `_loop`, `_loop_restart` will refuse to run while it exists (modeled on the retest sentinel).
  4. **Move-not-delete archive**: `mkdir -p .omc/archive/apr15-snapshot/ && git mv autoresearch/baseline_metrics.json .omc/archive/apr15-snapshot/ && mv results.tsv .omc/archive/apr15-snapshot/ && mv .omc/feature_cache .omc/archive/apr15-snapshot/feature_cache 2>/dev/null && mv splice/classifier/fp_classifier.joblib* .omc/archive/apr15-snapshot/`. All recoverable from the archive dir, with the apr15 commits also unchanged.
  5. **Commit the move**: `git commit -m "reset: archive apr15 baseline + classifier bundle for korean-iter1 pivot"`. Now `git reset --hard HEAD~1` cleanly undoes step 4 only.
  6. **Regenerator runs (steps 5–12 of the implementation plan)**: sentinel removed only after step 12's smoke iteration succeeds.

  - SIGINT safety: between step 5 and step 12 the tree is in a *clean, runnable but no-baseline* state. Wrapper's `start` already refuses to run if `baseline_metrics.json` is missing — natural backstop. The sentinel adds a second backstop covering the full destructive + regen-in-flight + maintainer-migration window. **(Item 1)**
- Alternative orders rejected:
  - **Wipe before branch**: a SIGKILL between wipe and branch leaves apr15 in a broken state. INVALIDATED by sentinel/checkpoint principle.
  - **Branch before tag**: an accidental `git branch -D apr15` would lose the history with no tag pin. INVALIDATED — tag must precede every other op.
  - **Sentinel after `git mv`** (v1's order): a SIGKILL between `git mv` and sentinel-write leaves the tree mid-archive with no operator marker. INVALIDATED by Item 1 architect-fix-1.

#### Tension F — Ground-truth file format **(Item 4 — REVISED, Option B chosen)**
- **Chosen: per-conversation JSON, sibling to the audio file, PLUS a synthesized per-corpus `ground_truth.json` at corpus root for preflight compatibility.** `data/eval/korean_iter1/eval/conversation_00088.json` next to `data/eval/korean_iter1/eval/conversation_00088.opus`. Plus a single-file split-level manifest `_manifest.jsonl` with `(file_id, voice_holdout, augment_seed, n_boundaries, audio_sha256)` for fast iteration without O(N) JSON loads. Plus a per-split aggregated `ground_truth.json` (one per `{train,eval,test}/` subdir) that the regenerator emits as the LAST step of regen — content is `{file_id: [{time_s, label}], ...}` — so `autoresearch/preflight.py:82-89` can hash it via its existing logic with no code change.
- **Why Option B (synthesize `ground_truth.json`) over Option A (rename manifest field)**: option A would require editing `autoresearch/preflight.py` (a protected file) AND every other reader that looks at `ground_truth_sha256_prefix`. Option B keeps preflight's contract identical to the singing/korean/english datasets — one less protected-file edit, one less coupling point, and the `_manifest.jsonl` retains its purpose as the fast-load index for per-iteration code.
- Justification: matches existing `ground_truth.json` convention in `data/eval/{singing,korean,english}/`, easy to inspect during debugging, copy-pasteable into reports. The split-level manifest gives the wrapper a single-pass schema for per-file SHA verification (mit 4 of §2 scenario 1). The per-corpus `ground_truth.json` is the canonical preflight target — its SHA prefix is what `manifest.json["ground_truth_sha256_prefix"]` references.
- Options rejected:
  - **Single JSONL for all conversations in a split**: 1300-row file. Faster to load (~10 ms vs ~300 ms for 1300 separate JSON opens). Hard to inspect a single conversation's labels without `jq`. Schema versioning is awkward (any change rewrites the whole file). INVALIDATED — the `_manifest.jsonl` provides the fast-path without sacrificing per-file inspectability.
  - **Embed labels in audio metadata (Vorbis comment)**: cute, opaque to grep. INVALIDATED.
  - **Option A: rename manifest field to `manifest_sha256_prefix` + edit preflight.py**: requires a protected-file edit to `preflight.py` for no functional gain. Synthesizing `ground_truth.json` is strictly cheaper. INVALIDATED.

---

## §2. Pre-mortem (3 scenarios + mitigations)

### Scenario 1 (HIGHEST RISK) — Regenerate is non-deterministic across runs, breaking supervisor verify
**Failure mode**: a future loop run (post-corpus-corruption, post-disk-loss, post-OS-upgrade) re-runs the regenerator and produces audio bytes that differ from the originals. Verifier's `metric re-run` check fails because the `combined` score shifts on the same git SHA. Or: librosa version bump changes the augmentation pipeline output, silently invalidating every comparison in the loop's history.

**Why this is the worst scenario**: it's silent. The loop keeps running; metrics drift; weeks of hypotheses get ranked against a non-stationary baseline. Recovery requires a full corpus rebuild + history truncation.

**Mitigations** (defense in depth):
1. **Pin pipeline-affecting deps in `pyproject.toml`**: `librosa==<exact>`, `soundfile==<exact>`, `numpy==<exact>`, `scipy==<exact>`, `pyopus` (or whatever Opus binding) `==<exact>`. New plan step 4.
2. **`opus_version` environment fingerprint**: regenerator records `opusenc --version` output into the split-level `_manifest.jsonl`; preflight refuses to start if the recorded version mismatches the live binary.
3. **Deterministic regen smoke gate**: regenerator's `--verify-determinism` flag re-runs augmentation on 10 random files and byte-diffs the result against the on-disk version. Operator MUST run this once after regen and commit the green output to `.omc/korean-iter1-determinism-receipt.json`.
4. **Per-file SHA in manifest**: the `_manifest.jsonl` records `audio_sha256` of each augmented file; preflight verifies SHA on a 5-file sample every iteration startup (added to `autoresearch/preflight.py`).
5. **No silent regen**: the wrapper REFUSES to call the regenerator. Regen is operator-invoked only, behind an explicit `--regenerate` flag on a new `scripts/regenerate_korean_iter1.py`. The loop has no path to invoke it.

### Scenario 2 — Augmentation pipeline silently degrades audio (clipping, NaN through Opus)
**Failure mode**: pink noise + RIR convolution overshoots peak amplitude → clipping → Opus encoder's psychoacoustic model produces hash-different but perceptually similar files (deterministic but lossy). Or: a pathological RIR T60 sample (boundary case, T60 → 0.6 producing >1.0 amplitude) propagates a NaN into the Opus encoder, yielding silence that the detector sees as a single huge boundary at t=0. The loop spends 50 iterations training a GBM on garbage labels.

**Mitigations**:
1. **Per-augmentation invariant checks** baked into the pipeline:
   - `assert np.isfinite(audio).all()` after each stage.
   - `assert audio.max() < 0.999 and audio.min() > -0.999` (sub-1 peak) after RIR/noise; if violated, normalize-to-0.95 with a logged warning emit.
   - `assert duration_after / duration_before in [0.95, 1.05]` after Opus roundtrip (catches encoder drops).
2. **Spot-check audit step**: regenerator writes 5 random files (1 train, 2 eval, 2 test) into `.omc/regen-spot-check/` as raw WAV alongside their Opus version. Operator listens before the sentinel comes down. Plan step 7 is gated on operator approval.
3. **Per-file augmented stats in manifest**: `_manifest.jsonl` records `peak_db, rms_db, n_clipped_samples`. Anomaly: any file with `n_clipped_samples > 0` or `rms_db < -50` flagged in the regenerator's stdout summary.
4. **First-iter loop sanity assertion**: the smoke iteration at plan step 12 must show `precision > 0.05` and `recall > 0.05` — if both are <0.05, the corpus is suspect and the loop halts before iteration 2.

### Scenario 3 — Voice-pair holdout produces a too-narrow train split, GBM overfits to 7 voices
**Failure mode**: 7 train voices share a TTS model family, similar prosodic distribution, similar formant locations. The GBM learns "voice = boundary" cues that don't transfer to held-out voices. Eval F1 looks fine (eval voices are *also* TTS-family-similar), but the spectral/prosodic gap to the test holdout is wider than expected. The loop optimizes against eval, ships, and the test eval at iteration N reveals collapse.

**Mitigations**:
1. **Voice-diversity gate at regen step 0** (already in tension D resolution): print gender/age/F0 mean per voice. If train voices fail to span the eval/test voice F0 envelope by a factor of 0.8–1.25×, error out.
2. **Train/eval F1 gap monitor**: every iteration emits `diag.eval.train_eval_gap` event with `train_f1 - eval_f1`. Wrapper appends to results.tsv. If >5 iterations show gap > 0.20, supervisor maintain logs `WARN diag.eval.overfit_signal`.
3. **Mandatory test-eval at iter 5, 10, 25, 50**: not just at the end. New plan step 13. Diverging trend in test F1 vs eval F1 triggers a `note: test-eval-divergence` discard recommendation surfaced in the wrapper prompt.
4. **Doc'd escape hatch**: ADR follow-up — if test-eval gap exceeds 0.15 by iter 25, regenerate with 5 train voices instead of 7 (forces voice-agnostic features to dominate). One-line `--n-train-voices=5` re-run.
5. **Item 3 reinforcement**: per-class F1 in `splice_class_breakdown` excludes `"unknown"` predictions, so a label-blind classifier can NOT mask voice-overfit by guessing the dominant class. The `cross_voice_f1` / `same_voice_edit_f1` metrics reflect only deliberate label predictions — preserves the voice-overfit detection signal.

---

## §3. Sequenced Implementation Steps

Each step lists files, change, verification, reversibility, wall-clock.

**Sequencing note (Item 1)**: sentinel `.omc/korean-iter1-regen-in-progress` is created at step 3.5 (just before any destructive `mv`/`git mv` op) and removed at step 12 (after the smoke iteration succeeds). Steps 4 → 12 are guarded by it.

### Step 1 — Stash/commit dirty tree, tag apr15 snapshot

- **Files touched**: working tree only (no commits to `splice/`/`autoresearch/`).
- **Change**: operator decides per file whether to commit (`scripts/regenerate_datasets.py`, `splice/dataset_registry.py` — likely commit since they're work-in-progress dataset infra) or stash (`.omc/classifier/detector_v*.py` — these are research artifacts, commit them as a single `archive: detector vN snapshots from apr15` commit on the apr15 branch). Then:
  ```
  git tag -a apr15-snapshot-2026-04-25 autoresearch/apr15 -m "Pre-pivot snapshot"
  git tag -a apr15-final-2026-04-25 HEAD -m "Including dirty WIP commits"
  git push origin apr15-snapshot-2026-04-25 apr15-final-2026-04-25
  ```
- **Verification**: `git tag -l 'apr15-*' | wc -l` returns 2; `git ls-remote --tags origin | grep apr15` returns 2 lines.
- **Reversibility**: tags are deletable (`git tag -d ...`), no data lost. Stashes recoverable via `git stash list`.
- **Wall-clock**: ~10 min (operator review).

### Step 2 — Branch off master

- **Files touched**: none (branch creation only).
- **Change**:
  ```
  git checkout master && git pull origin master
  git checkout -b autoresearch/korean-iter1
  ```
- **Verification**: `git rev-parse --abbrev-ref HEAD` returns `autoresearch/korean-iter1`. `git log --oneline -1` shows master's tip.
- **Reversibility**: `git checkout autoresearch/apr15 && git branch -D autoresearch/korean-iter1`.
- **Wall-clock**: ~2 min.

### Step 3 — Scaffold regenerator script

- **Files touched (new)**:
  - `scripts/regenerate_korean_iter1.py` (new, ~600 lines target)
  - `scripts/_korean_iter1/` (helper module dir, optional — depends on whether splitting helps testability)
- **Change**: skeleton with subcommands: `--verify-source` (read tarball, list voices, print diversity table), `--regenerate` (full pipeline), `--verify-determinism` (re-run on 10 files + byte diff), `--smoke 5` (regen 5 conv slice for dev iteration), `--resume` (skip files already on disk).
- **Verification**: `PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --verify-source --tarball /Volumes/HIKSEMI/korean-iter-1-delivery.tar` prints the 11-voice diversity table and exits 0.
- **Reversibility**: `git rm scripts/regenerate_korean_iter1.py`. (No sentinel yet — see step 3.5.)
- **Wall-clock**: ~60 min skeleton + CLI.

### Step 3.5 — Write sentinel (NEW per Item 1)

- **Files touched (new)**: `.omc/korean-iter1-regen-in-progress` (sentinel, gitignored).
- **Change**: `touch .omc/korean-iter1-regen-in-progress`. Single line of operator intent: "korean-iter1 pivot in flight; do not start the loop."
- **Verification**: `ls .omc/korean-iter1-regen-in-progress` returns the path; `./run_autoresearch.sh start` (dry-test) refuses to run with a clear error message referencing the sentinel.
- **Reversibility**: `rm .omc/korean-iter1-regen-in-progress` — but ONLY if the operator is intentionally aborting the pivot.
- **Wall-clock**: 30 s.
- **Why before step 4**: every `git mv`/`mv` op in step 4 is destructive; a SIGKILL or operator interruption between step 4 and a downstream step would leave the tree in a half-archived state with nothing marking it as such. The sentinel is the single source of truth that "this tree is mid-pivot, do not run the loop."

### Step 4 — Archive apr15 baseline + classifier bundle

- **Files touched**: `autoresearch/baseline_metrics.json` (moved), `results.tsv` (moved), `.omc/feature_cache/` (moved), `splice/classifier/fp_classifier.joblib` + `.meta.json` (moved), `autoresearch/manifest.json` (preserved as apr15 reference).
- **Change**:
  ```
  mkdir -p .omc/archive/apr15-snapshot
  git mv autoresearch/baseline_metrics.json .omc/archive/apr15-snapshot/baseline_metrics_apr15.json
  mv results.tsv .omc/archive/apr15-snapshot/results_apr15.tsv  # untracked, preserve
  mv .omc/feature_cache .omc/archive/apr15-snapshot/feature_cache_apr15 2>/dev/null || true
  mv splice/classifier/fp_classifier.joblib .omc/archive/apr15-snapshot/ 2>/dev/null
  mv splice/classifier/fp_classifier.meta.json .omc/archive/apr15-snapshot/ 2>/dev/null
  cp autoresearch/manifest.json .omc/archive/apr15-snapshot/manifest_apr15.json
  git add .omc/archive/apr15-snapshot/
  git commit -m "reset: archive apr15 baseline + classifier bundle for korean-iter1 pivot"
  ```
- **Verification**: `ls autoresearch/baseline_metrics.json` returns "no such file"; `ls .omc/archive/apr15-snapshot/baseline_metrics_apr15.json` succeeds; `git log --oneline -1` shows the archive commit; `ls .omc/korean-iter1-regen-in-progress` still exists (sentinel preserved through this step).
- **Reversibility**: `git reset --hard HEAD~1` undoes the move (file restored from git index). The non-tracked moves (`results.tsv`, `feature_cache`) need `mv` back from the archive dir.
- **Wall-clock**: ~10 min.

### Step 5 — Pin pipeline-affecting deps + lock environment fingerprint

- **Files touched**: `pyproject.toml`, `uv.lock`, new file `.omc/korean-iter1-env-fingerprint.json`.
- **Change**: tighten librosa, soundfile, numpy, scipy, and Opus binding versions to exact pins. Generate fingerprint JSON with `{librosa, soundfile, numpy, scipy, opus_binary_version, opusenc_version_string, python_version}`.
- **Verification**: `PYTHONPATH=$PWD uv run python -c "import librosa, soundfile, numpy, scipy; print(librosa.__version__, soundfile.__version__, numpy.__version__, scipy.__version__)"` matches fingerprint.
- **Reversibility**: `git checkout -- pyproject.toml uv.lock`.
- **Wall-clock**: ~20 min.

### Step 6 — Implement core regenerator pipeline (edit-then-augment)

- **Files touched**: `scripts/regenerate_korean_iter1.py` (filled in).
- **Functions** (each with unit test in §4):
  - `assign_split(voice_name, voice_holdout_config) -> "train"|"eval"|"test"`: pure dispatch.
  - `pick_edit_positions(words, n_edits, seed, min_pad_ms=120) -> list[(start_ms, end_ms)]`: deterministic word-cut selection. Pseudocode:
    ```
    rng = numpy.random.default_rng(seed)
    eligible = [w for w in words if w.gap_before_ms >= 120 and w.gap_after_ms >= 120
                and not_overlapping_a_turn_boundary(w)]
    n = rng.integers(1, 4)  # 1..3
    if len(eligible) < n: return []  # no edits, file is GT-clean
    chosen = rng.choice(eligible, size=n, replace=False, sorted by start_ms)
    return [(w.start_ms, w.end_ms) for w in chosen]
    ```
  - `apply_word_cuts(audio, sr, cut_spans_ms) -> (edited_audio, join_timestamps_s)`: cut each span, concatenate, return the post-cut timestamp of each join (NOT the pre-cut position).
  - `augment(audio, sr, file_seed) -> augmented_audio`: pink noise (SNR ~ N(22, 4) clipped to [10, 35]), exp RIR (T60 ~ U(0.2, 0.6)), Opus 32 kbps roundtrip via `pyopus`/`opusenc` subprocess. Each randomization seeded via `hash(file_id) ^ stage_id`.
  - `compute_cross_voice_boundaries(transcript) -> list[float]`: turn `start_ms` for `idx > 0` after applying any time-shift from the cuts.
  - `write_outputs(out_dir, file_id, audio, sr, boundaries)`: writes `.opus` + `.json` + appends `_manifest.jsonl` row.
  - `synthesize_corpus_ground_truth(split_dir) -> dict` **(NEW per Item 4)**: aggregates every `*.json` boundary file under `split_dir/` into a single `{file_id: [{time_s, label}], ...}` map; writes to `split_dir/ground_truth.json` as the LAST step of regen for that split. This file is what `autoresearch/preflight.py:82-89` hashes via its existing logic — no preflight code change required.
- **Change**: implements all 7 functions with invariant assertions (§2 scenario 2 mitigation 1).
- **Verification**:
  - `PYTHONPATH=$PWD uv run pytest splice/tests/test_regenerate_korean_iter1.py -v` (see §4 unit tests) passes.
  - `PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --smoke 5 --tarball /Volumes/HIKSEMI/korean-iter-1-delivery.tar --out-dir /tmp/korean-iter1-smoke/` produces 5 `.opus` + 5 `.json` files in `/tmp/korean-iter1-smoke/eval|train|test/`, a `_manifest.jsonl` with 5 rows total, AND a `ground_truth.json` aggregate at each split root.
- **Reversibility**: `git reset --hard HEAD~N` (this work spans multiple commits — likely 4–6 small commits). Smoke output in `/tmp/` auto-cleans.
- **Wall-clock**: ~3 hours.

### Step 7 — Update `splice/dataset_registry.py` to include `korean_iter1`

- **Files touched**: `splice/dataset_registry.py`.
- **Change**: add a `Dataset(id="korean_iter1", eval_path=_eval_path(_DATA / "eval" / "korean_iter1" / "eval", "korean_iter1"), train_path=_DATA / "train" / "korean_iter1" / "train", test_path=_DATA / "test" / "korean_iter1" / "test", eval_weight=1.0, train_weight=1.0)` entry. Drop existing singing/korean/english to `eval_weight=0.0, train_weight=0.0` (already done in the dirty WIP — re-apply on the new branch).
- **Verification**: `PYTHONPATH=$PWD uv run python -c "from splice.dataset_registry import DATASETS; active = [d for d in DATASETS if d.eval_weight > 0]; assert len(active)==1 and active[0].id=='korean_iter1', active"`.
- **Reversibility**: `git checkout HEAD~1 -- splice/dataset_registry.py`.
- **Wall-clock**: ~10 min.

### Step 8 — Run regenerator on full corpus, operator audit

- **Files touched (output)**: `data/eval/korean_iter1/{train,eval,test}/*.opus`, `*.json`, `_manifest.jsonl`, **`ground_truth.json`** (per-split aggregate, Item 4). `.omc/regen-spot-check/` (5 wav+opus pairs for listening).
- **Change**: `PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --regenerate --tarball /Volumes/HIKSEMI/korean-iter-1-delivery.tar --out-dir data/eval/korean_iter1 --workers 8 --emit-spot-check 5 2>&1 | tee .omc/regen-fulllog.txt`.
- **Verification**:
  - File count: `find data/eval/korean_iter1/{train,eval,test} -name '*.opus' | wc -l` returns ~7360.
  - Manifest count matches: `wc -l data/eval/korean_iter1/{train,eval,test}/_manifest.jsonl` sums to ~7360.
  - **Per-split `ground_truth.json` exists** at each of `data/eval/korean_iter1/{train,eval,test}/ground_truth.json`; row count matches manifest count.
  - Per-split counts match expected ratio (~4500/1100/1300 for 7/2/2 voice split).
  - Operator listens to all 5 spot-check files; subjectively confirms speech audible, ambience reasonable, no clipping.
  - Determinism gate: `PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --verify-determinism --sample 10` exits 0; commit the receipt to `.omc/korean-iter1-determinism-receipt.json`.
- **Reversibility**: `rm -rf data/eval/korean_iter1/`. The audio is regeneratable in 12 min from the source tarball.
- **Wall-clock**: ~15 min compute + 15 min audit = 30 min wall.

### Step 9 — Migrate `splice/evaluate.py` to boundary-F1 with collar **(MAINTAINER COMMIT)**

- **Files touched (PROTECTED)**: `splice/evaluate.py`. **Single maintainer commit** with message: `MIGRATE-PROTECTED splice/evaluate.py: korean-iter1 boundary-F1 with 250 ms collar (pivot)`.
- **Item 7 note**: the `MIGRATE-PROTECTED` prefix is human-readable convention only; the supervisor's diff audit (`autoresearch/supervisor_agent.py:102-162`) does NOT recognize it. The exemption mechanism is purely temporal: this commit lands BEFORE `run_autoresearch.sh start`, so it sits OUTSIDE the per-iteration `OMC_HEAD_BEFORE..HEAD` audit window. Operator must verify all maintainer commits (steps 9, 10, 11a, 11b, 11c) are in place BEFORE step 12 invokes the loop.
- **Change**:
  - Replace the dataset iteration to load from `data/eval/korean_iter1/eval/` only (reads the per-conv `.json` files).
  - Replace the metric: `combined = boundary_f1` where boundary_f1 uses greedy-nearest-within-collar one-to-one matching:
    ```
    pseudocode (collar_aware_f1):
      sort_by_score(predictions, descending)
      matched_gt = set()
      tp = 0
      for pred in predictions:
        candidates = [g for g in gt if abs(g.time - pred.time) <= 0.250 and g not in matched_gt]
        if candidates:
          nearest = min(candidates, key=lambda g: abs(g.time - pred.time))
          matched_gt.add(nearest)
          tp += 1
      fp = len(predictions) - tp
      fn = len(gt) - len(matched_gt)
      precision = tp / (tp + fp) if (tp + fp) else 0
      recall = tp / (tp + fn) if (tp + fn) else 0
      f1 = 2*p*r / (p+r) if (p+r) else 0
    ```
  - **Per-class breakdown (REVISED per Item 3)**: split GT into `cross_voice` and `same_voice_edit` subsets. For each per-class F1 in `splice_class_breakdown`, **filter predictions to only those with that exact label** (i.e., exclude `"unknown"` and the other class). TP/FP/FN are then computed on the (filtered_predictions, class_GT) pair. This means an unlabeled detector emission (label = `"unknown"`) contributes ZERO to either `cross_voice_f1` or `same_voice_edit_f1`. Overall `combined = boundary_f1` is computed label-blind (counts all predictions regardless of label), so `unknown`-labeled detectors are still scored on aggregate boundary recall — they just can't claim per-class skill.
  - Pred-label normalization at eval boundary: any pred tuple whose 2nd element is missing, `None`, empty string, or not in `{"cross_voice", "same_voice_edit"}` is coerced to `"unknown"`. Logged as `diag.eval.unknown_label_count` per iteration.
  - LAST line still `combined: <float>` (preserves wrapper grep contract — see `splice/evaluate.py:558,660`).
  - Keep `RESULTS_TSV: ...` line; new fields are `combined precision recall n_files cross_voice_f1 same_voice_edit_f1 unknown_label_count` (US-515 phase-3a-equivalent maintainer change).
- **Verification**:
  - `PYTHONPATH=$PWD uv run python splice/evaluate.py 2>&1 | tail -3` prints `combined: <float between 0 and 1>`.
  - `PYTHONPATH=$PWD uv run python splice/evaluate.py 2>&1 | grep -E '^RESULTS_TSV:' | awk '{print NF}'` shows the expected new column count.
  - **Item 3 unit test**: `PYTHONPATH=$PWD uv run python -c "from splice.evaluate import compute_class_f1; preds=[(1.0,'unknown'),(2.0,'unknown')]; gt_cv=[(1.0,'cross_voice')]; assert compute_class_f1(preds, gt_cv, 'cross_voice', 0.25) == 0.0, 'unknown preds must NOT count toward cross_voice_f1'"`.
  - Wall-clock: `time PYTHONPATH=$PWD uv run python splice/evaluate.py` completes in <240 s on the 60-sample iter eval set (sample size logic added — random.seed(0) sampling of 60 from `data/eval/korean_iter1/eval/`).
- **Reversibility**: `git checkout HEAD~1 -- splice/evaluate.py` works (single-file commit).
- **Wall-clock**: ~90 min.

### Step 10 — Migrate `splice/detector.py` contract to `list[(time_s, label)]`

- **Files touched**: `splice/detector.py` (NOT protected — agent's, but this is a contract change so maintainer commit).
- **Change**: change return type of `detect_splices` to `list[tuple[float, str]]`. Existing call sites that emit unlabeled times pass `("unknown", )` as the default label **(Item 3 — was `"cross_voice"`)**. The agent is free to override during iteration with `"cross_voice"` or `"same_voice_edit"`; emitting `"unknown"` opts out of per-class F1 contribution. Document the contract in the function docstring and in `splice/program.md` (step 11a).
- **Verification**:
  - `PYTHONPATH=$PWD uv run python -c "from splice.detector import detect_splices; import numpy as np; out = detect_splices(np.zeros(44100), 44100); assert isinstance(out, list); assert all(isinstance(x, tuple) and len(x)==2 and isinstance(x[1], str) for x in out), out"`.
  - **Item 3**: `PYTHONPATH=$PWD uv run python -c "from splice.detector import detect_splices; import numpy as np; out = detect_splices(np.zeros(44100), 44100); assert all(x[1] in {'cross_voice','same_voice_edit','unknown'} for x in out), 'all labels must be in {cross_voice, same_voice_edit, unknown}'"`.
- **Reversibility**: `git checkout HEAD~1 -- splice/detector.py`.
- **Wall-clock**: ~30 min.

### Step 11 — Maintainer migration of protected files (split per Item 2)

This step is now SPLIT into 11a, 11b, 11c. The reason: v1's monolithic step 11 wrote a placeholder `baseline_metrics.json` BEFORE step 12's real eval. Between those steps, `--verify`'s anomaly check would compare against a placeholder, defeating the purpose. The split serializes: maintainer migrations first (no baseline yet), real eval, atomic baseline write from the real number, THEN gating verify.

#### Step 11a — Migrate `splice/classifier/train_classifier.py` to 3-class targets

- **Files touched**: `splice/classifier/train_classifier.py` (NOT protected — agent-editable, but this is a one-time framework migration so maintainer-flagged in commit message).
- **Change**: load training patches from `data/eval/korean_iter1/train/` (per-conv JSON gives boundary times+labels; extract feature windows ±N frames around each boundary). Targets become 3-class: `cross_voice`, `same_voice_edit`, `no_splice` (negative samples drawn from non-boundary frames). HistGradientBoosting in multinomial mode (sklearn `HistGradientBoostingClassifier` supports multiclass natively). Note: training data labels are always one of the 3 explicit classes; `"unknown"` is an EVAL-time label only and never appears in training targets.
- **Verification**:
  - `PYTHONPATH=$PWD uv run python splice/classifier/train_classifier.py 2>&1 | tail -5` shows `n_classes=3` in the meta output.
  - `splice/classifier/fp_classifier.joblib` written; `.meta.json` records `classes_=['cross_voice', 'no_splice', 'same_voice_edit']`.
  - Smoke load: `PYTHONPATH=$PWD uv run python -c "import joblib; m = joblib.load('splice/classifier/fp_classifier.joblib'); assert set(m.classes_) == {'cross_voice', 'no_splice', 'same_voice_edit'}"`.
- **Reversibility**: `git checkout HEAD~1 -- splice/classifier/train_classifier.py && rm -f splice/classifier/fp_classifier.joblib*`.
- **Wall-clock**: ~90 min.

#### Step 11b — Maintainer migration: `splice/program.md` + `autoresearch/manifest.json` + `autoresearch/preflight.py` (NO baseline write)

- **Files touched (PROTECTED)**: 3 files, three single-file maintainer commits (Item 7: prefix is human-readable convention; commits MUST land pre-loop-start to be outside the diff-audit window).
  - `MIGRATE-PROTECTED splice/program.md: korean-iter1 3-class boundary-F1 (pivot)` — rewrite the goal section, replace constraints (3-class, no clean_score, no DSP FP bound), update editable-file list. Document the `"unknown"` label semantics from Item 3. Flag explicitly to operator: "Re-read this file end-to-end before merging — protected, human-edited only."
  - `MIGRATE-PROTECTED autoresearch/manifest.json: korean-iter1 dataset + counts (pivot)` — `dataset` → `data/eval/korean_iter1/eval`, `expected_counts` → per-voice-pair file counts, `ground_truth_sha256_prefix` → first-12-chars-of(SHA of `data/eval/korean_iter1/eval/ground_truth.json`) (Item 4: this is the per-corpus aggregate emitted by step 6's `synthesize_corpus_ground_truth`).
  - `MIGRATE-PROTECTED autoresearch/preflight.py: korean-iter1 manifest SHA + env fingerprint (pivot)` — add `_manifest.jsonl` integrity check (5-file sample audio_sha256 verification per §2 scenario 1 mit 4); add env fingerprint check vs `.omc/korean-iter1-env-fingerprint.json`. The `ground_truth.json` hash check at lines 82-89 is UNTOUCHED — Item 4 chose Option B specifically to avoid editing this logic.
- **Important: `autoresearch/baseline_metrics.json` is NOT touched in 11b.** It does not exist on disk yet (it was archived in step 4). The supervisor's anomaly check has explicit handling for missing baseline (`autoresearch/supervisor_agent.py:167-168` returns WARN, not FAIL).
- **Verification**:
  - `git log --oneline -3 | grep MIGRATE-PROTECTED | wc -l` returns 3.
  - `PYTHONPATH=$PWD uv run python -c "import json; m=json.load(open('autoresearch/manifest.json')); assert 'korean_iter1' in m['dataset'] and 'ground_truth_sha256_prefix' in m"`.
  - `ls autoresearch/baseline_metrics.json` returns "no such file" (sanity — must not exist yet).
- **Reversibility**: per-file `git checkout HEAD~1 -- <path>` per commit.
- **Wall-clock**: ~60 min (program.md rewrite is the slow part; operator must read it).

#### Step 11c — Create iter1-anchor tag (Item 6 — Option B chosen)

- **Files touched**: none (tag-only operation).
- **Change**: `git tag -a iter1-anchor HEAD -m "Safe first-iteration discard target for korean-iter1 loop start"`.
- **Why Option B (operator tag) over Option A (wrapper-emitted marker commit)**: Option A would require editing `run_autoresearch.sh` to detect "first iteration ever on this branch" and create a marker commit. Adds wrapper complexity for a one-shot need. Option B is a single operator command, leaves no permanent wrapper code-path, and the tag is human-greppable (`git tag -l 'iter1-*'`) for any later forensic audit. The wrapper's existing per-iteration `head_before=$(git rev-parse HEAD)` capture (line 855) already targets the correct commit — `iter1-anchor` is a redundant safety marker, not a behavioral change.
- **Wrapper semantics confirmed**: `_guarded_reset "$head_before"` (lines 1142, 1215, 1237, 1265, 1297) resets to `head_before`, NOT `HEAD~1`. On iter 1, `head_before` is the SHA captured at the top of the iteration — which after step 11b's migration commits is the `MIGRATE-PROTECTED autoresearch/preflight.py` commit. A first-iteration discard correctly resets to that maintainer baseline. The `iter1-anchor` tag is a defensive duplicate so the operator can manually verify `git rev-parse iter1-anchor == git rev-parse HEAD` before invoking `run_autoresearch.sh start`.
- **Verification**:
  - `git tag -l iter1-anchor` returns the tag.
  - `git rev-parse iter1-anchor` equals `git rev-parse HEAD`.
  - **(§4 anchor test)** `git merge-base --is-ancestor iter1-anchor HEAD` exits 0 (anchor is reachable from HEAD).
- **Reversibility**: `git tag -d iter1-anchor`.
- **Wall-clock**: ~1 min.

### Step 12 — End-to-end smoke iteration + atomic baseline write (split per Item 2)

- **Files touched**: `autoresearch/baseline_metrics.json` (NEW — written from real eval, atomically).
- **Change**:
  - Run baseline eval: `PYTHONPATH=$PWD uv run python splice/evaluate.py | tee .omc/last_eval.log`. Capture `combined: X` value from the LAST line.
  - **Atomic baseline write (Item 2, 11b → 12 ordering)**: write `autoresearch/baseline_metrics.json` from the real `combined: X` plus the precision/recall/per-class fields parsed from the `RESULTS_TSV:` line. Schema:
    ```json
    {
      "timestamp": "2026-04-XX...",
      "git_sha": "<HEAD after step 11c>",
      "phase": "korean_iter1_pivot_initial",
      "combined": <iter-0 f1>,
      "precision": <p>,
      "recall": <r>,
      "n_files": 60,
      "collar_ms": 250,
      "splice_class_breakdown": {"cross_voice_f1": <f>, "same_voice_edit_f1": <f>},
      "voices_holdout": {"test": ["DaeBuHo","Ondo"], "eval": ["Sunwoo","Joon"], "train": ["Kanna","ChloeCha","DangchanYeo","Donghyun","Eunha","Minho","Minwoo"]},
      "regen_determinism_receipt_sha": "...",
      "unknown_label_count": <int>
    }
    ```
    Use `tempfile + os.replace` for atomic write. Commit as `MIGRATE-PROTECTED autoresearch/baseline_metrics.json: korean-iter1 iter-0 baseline (atomic from real eval)`.
  - Run verifier (now has a real baseline to compare against): `PYTHONPATH=$PWD uv run python autoresearch/supervisor_agent.py --verify --agent-name=baseline --reported-combined=<X>`. Anomaly delta should be 0.000; must return HIGH.
  - Remove sentinel: `rm .omc/korean-iter1-regen-in-progress` (Item 1: only removed on the happy path, after verify-HIGH).
  - Run wrapper smoke: `./run_autoresearch.sh start; sleep 600; ./run_autoresearch.sh stop`. Verify `results.tsv` has 1–3 new rows, no `iteration.start` event followed by uncategorized crash in `.omc/logs/autoresearch.jsonl`.
- **Verification**:
  - `wc -l results.tsv` returns ≥1.
  - `PYTHONPATH=$PWD uv run python scripts/validate_logs.py --parse .omc/logs/autoresearch.jsonl` zero schema errors.
  - `PYTHONPATH=$PWD uv run python autoresearch/tests/test_smoke_iteration.py` passes.
  - Sentinel is gone: `! ls .omc/korean-iter1-regen-in-progress` (returns "no such file").
- **Reversibility**: `./run_autoresearch.sh stop && touch .omc/autoresearch-stop`. Iteration commits live on the branch and can be `git reset` if undesired. `git reset --hard iter1-anchor` returns to the pre-loop maintainer baseline (Item 6 anchor's other purpose).
- **Wall-clock**: ~30 min (mostly waiting on iterations).

### Step 13 — Schedule recurring test-eval (mitigation for §2 scenario 3)

- **Files touched**: `run_autoresearch.sh` (small): add a `_run_test_eval_if_milestone` helper that, every Nth `--keep` (N=5), runs `splice/evaluate.py --split=test --n-files=200` and emits `diag.test_eval.snapshot` event with `combined`, `eval_combined`, and `gap`.
- **Change**: ~30 lines bash. New `--split` flag in `splice/evaluate.py` (already needs a flag for sampling — it can accept `--split=test|eval`).
- **Verification**: at iteration 5 the wrapper logs a `diag.test_eval.snapshot` event in the JSONL stream.
- **Reversibility**: `git checkout HEAD~1 -- run_autoresearch.sh splice/evaluate.py`.
- **Wall-clock**: ~30 min.

---

## §4. Expanded Test Plan

### Unit tests (new file: `splice/tests/test_regenerate_korean_iter1.py`)

| Function | Test cases |
|---|---|
| `assign_split` | 11 voices → exactly 7+2+2 partition; same input deterministic; unknown voice raises; respects override flag. |
| `pick_edit_positions` | 0 eligible words → returns `[]`; 1 eligible → returns 1; 100 eligible → returns 1–3 deterministic by seed; words too close to a turn boundary excluded; padding constraint enforced (fuzz 200 random word streams). |
| `apply_word_cuts` | output duration = input duration − sum(span lengths); join timestamps strictly ascending; no NaN; `apply_word_cuts(audio, sr, [])` returns audio unchanged. |
| `augment` | `np.isfinite(out).all()` and `out.max() < 0.999` always; same `file_seed` → byte-identical bytes (re-load and SHA); duration drift after Opus roundtrip <5%. |
| `compute_cross_voice_boundaries` | 1-turn transcript → no boundaries; 5-turn → 4 boundaries at correct timestamps; if a cut shifted time, boundaries after the cut shift accordingly. |
| `write_outputs` | round-trip: write then re-read JSON = original GT dict; manifest row schema matches (file_id, voice, split, n_boundaries, audio_sha256). |
| `synthesize_corpus_ground_truth` **(Item 4)** | aggregating 5 fixture per-conv JSONs produces a single `ground_truth.json` with 5 keys; SHA prefix is deterministic for the same input set; matches the SHA that `autoresearch/preflight.py:82-89` would compute. |

### Per-class F1 unit tests **(Item 3 — NEW)**

New file: `splice/tests/test_eval_per_class_f1.py`. Tests in `splice/evaluate.py`'s `compute_class_f1` (or equivalent):

| Test case | Assertion |
|---|---|
| `unknown` predictions vs `cross_voice` GT | `cross_voice_f1 == 0.0` (unknown excluded from prediction set) |
| `unknown` predictions vs `same_voice_edit` GT | `same_voice_edit_f1 == 0.0` (same logic) |
| `unknown` predictions vs all GT (label-blind aggregate) | `combined == boundary_f1` matches the predictions normally (unknown counted in aggregate) |
| Mixed: 1 `unknown` + 1 `cross_voice` pred vs 2 `cross_voice` GT | `cross_voice_f1` reflects only the 1 labeled prediction; aggregate F1 reflects both |
| Voice-overfit signal (Pre-mortem 3) | A detector that ALWAYS emits `unknown` shows `cross_voice_f1 == 0.0` AND `same_voice_edit_f1 == 0.0`, exposing voice-blind behavior even if aggregate F1 is high |
| Pred-label normalization | Pred tuple with `None`, empty string, or label outside `{"cross_voice", "same_voice_edit"}` coerced to `"unknown"`; `diag.eval.unknown_label_count` event emitted |

### Anchor test **(Item 6 — NEW)**

New test in `autoresearch/tests/test_korean_iter1_e2e.py`:

```python
def test_iter1_anchor_exists_and_reachable():
    rc = subprocess.run(["git", "tag", "-l", "iter1-anchor"], capture_output=True, text=True)
    assert rc.stdout.strip() == "iter1-anchor", "iter1-anchor tag missing — operator must run step 11c before loop start"
    rc = subprocess.run(["git", "merge-base", "--is-ancestor", "iter1-anchor", "HEAD"], capture_output=True)
    assert rc.returncode == 0, "iter1-anchor must be reachable from HEAD"
```

### Preflight test **(Item 4 — NEW)**

New test in `autoresearch/tests/test_korean_iter1_e2e.py`:

```python
def test_preflight_ground_truth_hash_matches_synthesized():
    """Per Item 4 Option B: regenerator's synthesize_corpus_ground_truth output
    must match autoresearch/preflight.py's hash check at line 82-89."""
    gt_path = Path("data/eval/korean_iter1/eval/ground_truth.json")
    assert gt_path.exists(), "regenerator's synthesize_corpus_ground_truth must emit per-corpus aggregate"
    sha = hashlib.sha256(gt_path.read_bytes()).hexdigest()[:12]
    manifest = json.load(open("autoresearch/manifest.json"))
    assert sha == manifest["ground_truth_sha256_prefix"], f"manifest sha {manifest['ground_truth_sha256_prefix']} ≠ actual {sha}"
```

### Integration test (new: `splice/tests/test_regenerate_smoke.py`)

- Runs `regenerate_korean_iter1.py --smoke 5` against a tiny fixture tarball (5 conv) checked into `splice/tests/fixtures/korean_iter1_micro.tar`. Asserts:
  - 5 `.opus` files written, 5 `.json` files written, 1 `_manifest.jsonl` with 5 rows, **1 `ground_truth.json` per split aggregate (Item 4)**.
  - Each `.opus` decodes via soundfile.read without error.
  - Boundaries in JSON include both classes if any same-voice edits were injected (else log a soft warn).
- Resets to a deterministic state for re-runs.

### End-to-end test (new: `autoresearch/tests/test_korean_iter1_e2e.py`)

- Runs `splice/evaluate.py` against the smoke output dir. Asserts:
  - Exit code 0.
  - LAST stdout line matches regex `^combined: \d+\.\d+$`.
  - `RESULTS_TSV:` line has exactly the expected column count (now includes `unknown_label_count`).
  - Wall-clock <30 s on the 5-conv slice.
- Includes the `test_iter1_anchor_exists_and_reachable` and `test_preflight_ground_truth_hash_matches_synthesized` cases above.

### Observability tests

- All regenerator stages emit `regen.<stage>.start` / `.complete` / `.error` events via `autoresearch.logger.get_logger("regen.korean_iter1")`. Per-100-file progress events.
- `scripts/validate_logs.py --parse` adds the `regen.*` prefix to its taxonomy whitelist.
- Test: a 5-conv smoke regen produces ≥10 `regen.*` events in the JSONL log.
- New event: `diag.eval.unknown_label_count` (Item 3) — emitted once per eval run; value is integer.

### Regression-blocking gate

- `PYTHONPATH=$PWD uv run pytest splice autoresearch -x` must pass at every commit. Existing tests that hardcode singing/korean/english under `weight > 0` need a one-line registry-mock fixture. The 49/49 baseline must hold.
- `PYTHONPATH=$PWD uv run python scripts/validate_logs.py --audit` must show zero residual `_diag(` (Python) and zero `echo ... >> $LOG_FILE` (bash).

---

## §5. Invariants

1. **(Item 5 — diff-audit window guarantee)** Maintainer commits (steps 9, 10, 11a, 11b, 11c) MUST land BEFORE `run_autoresearch.sh start` is invoked. The supervisor's diff audit operates on `OMC_HEAD_BEFORE..HEAD`, where `OMC_HEAD_BEFORE` is captured per-iteration at `run_autoresearch.sh:855` and exported at `run_autoresearch.sh:1283`. Pre-loop history is therefore outside the audit window by construction. **In-loop edits to protected files trigger the diff audit and halt the loop — this is intended behavior, not a bug.**
2. **(Item 7)** The `MIGRATE-PROTECTED:` commit-message prefix is human-readable convention only. The supervisor's `check_git_diff_audit` (`autoresearch/supervisor_agent.py:102-162`) does NOT parse commit messages — it only diffs the audit window. Operator-greppable for review (`git log --grep=MIGRATE-PROTECTED`) but not enforcement-level.
3. **(Item 1)** Sentinel `.omc/korean-iter1-regen-in-progress` is the single source of truth for "tree is mid-pivot." Created at step 3.5; removed at step 12 only on the happy path. Manual removal aborts the pivot without recovery — operator owns this decision.
4. **(Item 6)** `iter1-anchor` tag is the operator-verifiable safe state for the first iteration's discard target. The wrapper does NOT consult this tag; it's a manual recovery aid.
5. **(Item 2)** `autoresearch/baseline_metrics.json` is written EXACTLY ONCE during the pivot, in step 12, atomically (`tempfile + os.replace`), from the real eval output. No placeholder ever exists on disk.

---

## §6. ADR — Architecture Decision Record

### Decision
Pivot autoresearch-splice from a 3-domain (singing/korean/english) `splice_f1 × clean_score` framework to a single-corpus (korean-iter-1 TTS dialogue, 7360 conversations) **3-class boundary-F1 with 250 ms collar** framework, with deterministic edit-then-augment pre-processing yielding an on-disk reproducible corpus under `data/eval/korean_iter1/`. apr15 archived to `apr15-snapshot-2026-04-25` tag; new branch `autoresearch/korean-iter1` off master. Detector contract migrates to `list[(time_s, label)]` with `label ∈ {"cross_voice", "same_voice_edit", "unknown"}`; per-class F1 excludes `"unknown"` predictions; aggregate F1 counts them label-blind.

### Drivers
1. Loop must restart cleanly within 1 day of operator effort (clean baseline, no surprise crashes).
2. Verifier 4-check audit must remain green from iteration 1.
3. No manual intervention required mid-regenerate (resumable, deterministic, observable).

### Alternatives considered
| # | Alternative | Why rejected |
|---|---|---|
| A | Patch the apr15 framework with korean-iter1 as a 4th dataset | Existing 167-row history is already non-stationary across 8 git resets; mixing in a new dataset would compound the noise. The old `clean_score` and DSP FP gate are also irrelevant to the new corpus, requiring conditional code paths that bloat protected files. |
| B | On-demand augmentation per loop iteration (no on-disk corpus) | Blows the 240 s budget by ~3000% (12 min per file × 60 files). Disk savings (4 GB) trivial relative to the 39 GB external drive. |
| C | Real-world recording inclusion in the eval loop | Insufficient labeled data (operator has spot-check m4a only, no GT). Would require a parallel labeling pipeline that's out of scope. Spec § Non-Goals locks this out. |
| D | Augment-then-edit ordering | Produces non-realistic phase / decay-tail discontinuity at the join, would over-fit the detector to a non-realistic cue. |
| E | Multi-variant augmentation (N variants per file) | N× regen cost, N× disk, marginal generalization gain in a TTS-only corpus where the underlying acoustic distribution is already narrow. Defer to iter-2 if needed. |
| F | Default unlabeled detector emissions to `"cross_voice"` (v1's choice) | Silently inflates `cross_voice_f1`, defeats Pre-mortem 3 voice-overfit detection. v2 changes default to `"unknown"` per Item 3. |
| G | Edit `autoresearch/preflight.py` to look at `_manifest.jsonl` (Item 4 Option A) | Requires a protected-file edit and breaks parallel structure with singing/korean/english datasets. Synthesizing per-corpus `ground_truth.json` is strictly cheaper. |
| H | Wrapper-emitted iter1 marker commit (Item 6 Option A) | Requires a permanent code-path in `run_autoresearch.sh` to detect "first iter on this branch." Operator-tag is one command, leaves no permanent code, equally protective. |

### Why chosen
- Driver 1 (1-day restart): on-disk pre-augmented + sentinel-guarded destructive ops + per-step reversibility means the operator can stop after any step and resume. Worst case (full corpus rebuild) is 12 min compute + 15 min audit.
- Driver 2 (verifier green): per-file maintainer commits land BEFORE loop start and sit outside the per-iteration `OMC_HEAD_BEFORE..HEAD` audit window (Item 5 invariant). Baseline written atomically from real eval before the gating `--verify` invocation (Item 2). Determinism receipt commit means the verifier has a fixed point to compare against.
- Driver 3 (no mid-regen intervention): resumable regenerator (`--resume` skips files already on disk), per-stage invariant checks, structured logging via `autoresearch.logger`.

### Consequences
**Positive**:
- Single clean corpus = single clean optimization target. No more 3-domain weighted-mean fragility.
- Voice-pair holdout forces voice-agnostic features; first principled generalization signal in the loop's history.
- Same-voice content edits unlock a meaningful "splice-vs-no-splice" signal that voice-change-only iter-0 didn't have (33% of words editable at the chosen padding).
- New baseline schema is forward-compatible with iter-2 corpora (field `voices_holdout` generalizes to any speaker-id).
- Determinism receipt + env fingerprint give the verifier a reproducibility-audit tool nothing in apr15 had.
- **(Item 5)** The supervisor's per-iteration diff-audit window naturally excludes pre-loop maintainer history. Maintainer commits cost zero verifier-cycles AS LONG AS the operator lands them before `start`. This is a clean architectural property — no protected-file allowlist or commit-message-prefix parsing needed in the supervisor.
- **(Item 3)** `"unknown"` default label preserves voice-overfit detection: a label-blind detector cannot inflate per-class F1, so Pre-mortem 3's signal survives intentional underspecification.
- **(Item 4 Option B)** Per-corpus `ground_truth.json` synthesis keeps `autoresearch/preflight.py` untouched between corpora — same hash logic for singing/korean/english/korean_iter1.
- **(Item 1)** Sentinel-before-destructive sequencing ensures any SIGKILL between steps 4-12 leaves an operator-visible marker.

**Negative**:
- 4 GB on-disk corpus committed to operator's local disk (not gitignored — but the audio files don't enter git; only the `_manifest.jsonl` does, and even that may want gitignore depending on size; revisit).
- Lose all 167 rows of apr15 hypothesis history as actively-comparable data (still queryable as archive, but not loop-comparable).
- TTS-only corpus risk: detector might over-fit to ElevenLabs prosody artifacts that don't transfer to real-world recordings. Test-eval at milestones is the early warning; iter-2 corpus collection is the long-term answer.
- One-time 4–6 hours of engineer time to land the pivot; ~12 min compute + 15 min audit for regen.
- **(Item 7 honest cost)** The `MIGRATE-PROTECTED:` prefix gives no enforcement — purely operator discipline. If a maintainer commit accidentally lands AFTER `start`, the next iteration's `--verify` will FAIL with a protected-file violation, halting the loop. Mitigation: step 12's verification block explicitly checks all maintainer commits are in place before invoking `start`.
- **(Item 6 honest cost)** `iter1-anchor` is operator-managed; if forgotten, no test fails until the smoke iteration's anchor test runs. Step 11c is a hard prerequisite of step 12.

### Follow-ups (deliberately deferred)
1. **Real-world recording labeling pipeline + held-out eval set**. Operator's m4a samples remain spot-check only until a labeling tool exists.
2. **Sentence-level / multi-word edit injection**. Word-only is the iter-1 floor; if the detector saturates (>0.95 F1 by iter 30), graduate to phrase-level cuts.
3. **External noise/RIR corpora (MUSAN, BUT ReverbDB)** for augmentation diversity, when license review clears.
4. **Iter-2 corpus commission** once iter-1 plateaus on F1 or test-eval gap diverges.
5. **5-fold CV on voices** if the chosen 7+2+2 split shows train/eval gap >0.20.
6. **`--n-train-voices=5`** escape-hatch regen (mitigation in §2 scenario 3).
7. **Move `data/eval/korean_iter1/` audio bytes off the repo working tree** (e.g., into `~/.cache/autoresearch-splice/` with a symlink) if disk pressure becomes an issue.
8. **Phase 3a/b/c carve-outs** from US-515 (RESULTS_TSV migration, retest report logger, diagnose ERROR line) — out of scope for this pivot, parallel cleanup.
9. **Observability dashboard update** (`scripts/dashboard.py`) for the new schema fields (incl. `unknown_label_count`, `splice_class_breakdown`).
10. **Iter-1 → iter-2 corpus generation script** parameterized by voice-set + edit-density, to make iter-2 a config flip not a pivot.
11. **(Item 7 follow-up)** If maintainer migrations become common (US-518+), spec a one-line addition to `supervisor_agent.py` recognizing `MIGRATE-PROTECTED:` commit-message prefix as an explicit allowlist for the audit window. Out of scope for this pivot — current "land before start" discipline suffices for the one-shot reset.
12. **(Item 6 follow-up)** If iter1-anchor pattern proves useful across pivots, promote to wrapper (one-liner: `start` emits `iter1-anchor-<branch>` tag if not present). Currently operator-owned per Item 6 Option B.

---
