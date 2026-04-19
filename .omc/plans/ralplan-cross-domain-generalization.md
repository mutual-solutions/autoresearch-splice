# Cross-Domain Generalization — Consensus Plan

**Source spec**: `.omc/specs/deep-interview-cross-domain-generalization.md` (ambiguity 20%)
**Branch**: `cross-domain/apr17` (to be created from current HEAD; autoresearch branch `autoresearch/apr15` stays intact)
**Autoresearch state**: paused during this work; re-enabled in Phase E.

## Protected-file policy for this branch
`evaluate.py` is protected from the autoresearch agent per `CLAUDE.md` and `verify_agent.py`'s git-diff audit. This plan **explicitly lifts that protection on branch `cross-domain/apr17` only**, with human approval required for every evaluate.py edit. Rationale: the secondary speech-eval work (commit 52f7054) already set this precedent; the protection exists to prevent the autoresearch *loop* from gaming the oracle, not to freeze oracle development. Autoresearch is paused during this work, so the protection's purpose is temporarily suspended.

Operational rules:
- Every evaluate.py edit in this plan (A4, B1 if applied there) must be reviewed by the human before commit.
- Executor (autopilot/ralph) pauses after drafting evaluate.py changes and presents a diff for human sign-off.
- When Phase E re-enables autoresearch, evaluate.py returns to protected status on the autoresearch branch.
- `verify_agent.py` is NOT invoked during Phase A–E work on this branch.

## Requirements Summary
Redesign the splice detector + classifier pipeline to generalize across three domains — singing, Korean speech, synthesized English speech — with acceptance gate `min(combined_speech_ko, combined_speech_en) / combined_full >= 0.6`. Large singing regression (down to combined_full ≈ 0.50) acceptable. Classical DSP only, no neural networks. Eval budget stays at 120s. Add a public-English synthesized dataset, move singing-tuned thresholds into per-domain tables, retrain classifier on all three domains.

## RALPLAN-DR Summary (short mode)

### Principles
1. **Cross-domain generalization over per-domain optimization.** Prefer a ratio-balanced detector over a peak-performer on one domain.
2. **Reproducible dataset synthesis.** Every synthesized dataset must have a seeded generator script; no manual curation.
3. **Incremental phases with measurable baselines.** Each phase (A–E) produces a numbered, reproducible measurement before the next begins.
4. **Preserve autoresearch lineage.** Existing verified keeps (detector-v7 through v15) stay tagged in git history and are rollback-addressable.
5. **Classical DSP only.** Honor `program.md`: no torch / tf / sklearn MLPs / neural gradient descent.

### Decision Drivers (top 3)
1. **Ratio ≥ 0.6** (primary metric; all subordinate choices defer to this).
2. **120s eval budget** (hard; exceeding forces caching or lazy eval).
3. **No neural networks** (hard; architectural constraint).

### Viable Options Evaluated
**Option A (chosen): Full Tier 4 redesign on a new branch.**
- *Pros*: Highest ceiling on ratio. Clean separation from autoresearch. User explicitly chose this.
- *Cons*: Multi-day work. Likely large singing regression. All five phases must land to realize the benefit.

**Option B: Incremental with early exit.**
- *Pros*: Phase A alone (add English eval) reveals whether ratio metric even behaves; Phase B alone (input norm) may get us halfway. Shorter on-average cost.
- *Cons*: Risks settling for a local minimum; the spec explicitly asked for Tier 4. User rejected this in deep-interview Round 7 (Simplifier mode).

**Option C: Fork-and-merge.**
- *Pros*: Keeps `detector.py` on autoresearch branch untouched. Cross-domain work goes into a parallel detector variant; if it wins, cherry-pick back.
- *Cons*: Double maintenance. Hard to run autoresearch on the new variant without rewiring `evaluate.py`.

**Why A wins**: user-selected in a cleared-ambiguity interview; the ratio target (~3× speech improvement) requires multiple simultaneous fixes; Option C's merge cost exceeds any dev-safety gain because there's only one detector.py consumer (evaluate.py).

## Acceptance Criteria (testable)

| # | Criterion | Evidence |
|---|---|---|
| AC1 | `data/english-speech/` contains ≥100 clean utterances and ≥40 synthesized splices with `ground_truth.json` in the tier1/tier2 schema | `ls data/english-speech/{clean,tier1,tier2}/*.wav \| wc -l`, file count ≥140 |
| AC2 | `synthesize_english_splices.py` regenerates the dataset from a fixed seed with bit-identical audio output | Two consecutive runs produce `sha256` of all files identical |
| AC3 | `evaluate.py` prints three combined metrics: `combined_full`, `combined_speech_ko`, `combined_speech_en` | `grep -c '^combined' eval_output` returns ≥3 |
| AC4 | `evaluate.py` prints `cross_domain_ratio: N.NNN` where `ratio = min(speech_ko, speech_en) / combined_full` | regex match in output |
| AC5 | `cross_domain_ratio >= 0.60` on `--with-classifier` eval | direct metric read |
| AC6 | `combined_full >= 0.50` (singing regression bounded) | direct metric read |
| AC7 | Full `evaluate.py --with-classifier` eval completes in ≤120s on laptop CPU | `elapsed:` line |
| AC8 | **`detector.py` only** imports classical-DSP libs (scipy, numpy); no `import torch/tensorflow/sklearn` in detector.py. sklearn is permitted in ml_eval.py / train_classifier.py — tree-based ML is allowed per program.md | `grep -E 'import (torch\|tensorflow\|sklearn)' detector.py` returns empty |
| AC9 | Per-domain threshold configuration lives in a single lookup (e.g., `DOMAIN_THRESHOLDS` dict or `domain_config.py`) and drives all detector constants formerly hardcoded in detector.py | code inspection |
| AC10 | Classifier retrained on combined patches from all three domains; training run completes and writes a fresh `fp_classifier.joblib` | presence + mtime of file + log line |
| AC11 | Tags `detector-v7` through `detector-v15` still resolve to commits on the master history (not orphaned) | `git tag -l 'detector-v*' \| xargs -I{} git rev-parse {}` succeeds for each |
| AC12 | Autoresearch prompt in `run_autoresearch.sh` references `cross_domain_ratio` as the optimization target (not `combined`) | grep match |

## Implementation Steps

### Phase A — English-speech dataset synthesis (est. 2-3 hours)

A1. Create branch `cross-domain/apr17` from current `autoresearch/apr15` HEAD.

A2. Choose corpus: **VCTK** (110 English speakers, 48kHz, CC-BY-4.0, ~10GB total — small subset downloadable). Alternative: LibriSpeech dev-clean subset (~337MB).
   - Decision driver: VCTK is multi-speaker recorded in studio → clean splicing possible between different speakers (matches Korean-splice's multi-speaker splice pattern).
   - Fallback: LibriSpeech dev-clean if VCTK download fails.

A3. Write `data_synth/synthesize_english_splices.py` (new directory for dataset tooling):
   - Download 100 utterances from the chosen corpus (seed-fixed random selection).
   - Resample to **16 kHz mono float32** (matches Korean-splice, proves sr-adaptive code path).
   - Use the existing splice pipeline conceptually (the `generate_splices` tool exists in the parent `audio-splice-detector/` repo — copy the relevant generator into `data_synth/` so this repo is self-contained).
   - Produce 20 T1 (hard-cut) and 20 T2 (crossfade, xfade_ms ∈ {10, 50, 100, 200}) splices.
   - Write to `data/english-speech/{clean,tier1,tier2}/` and `data/english-speech/ground_truth.json`.

A4. Add secondary-eval support for English in `evaluate.py`:
   - Generalize the existing Korean-speech secondary block (commit 52f7054) into a loop over a domain config list.
   - Print `combined_speech_en: X` alongside `combined_speech_ko: X`.
   - Also print `cross_domain_ratio: min(speech_ko, speech_en) / combined_full`.

A5. Commit Phase A independently. Run full eval and capture baseline (likely `combined_speech_en ≈ 0.0` based on the Korean-speech experience — confirms the domain-tuning gap).

A6. **Corpus checksum manifest** (Principle 2 compliance): After successful download, compute `sha256sum` of every file in the selected-subset list and commit as `data_synth/corpus_checksums.sha256`. Subsequent `synthesize_english_splices.py` runs verify checksums before proceeding; if a file's checksum drifts (upstream encoding change), abort with a clear error instead of silently producing different patches.

A8. **`.gitignore` + repo-size policy**: `data/english-speech/**/*.wav` goes in `.gitignore` (same pattern as `data/spliced/` singing set, which lives via symlink). Only `data_synth/synthesize_english_splices.py`, `data_synth/corpus_checksums.sha256`, and `data/english-speech/ground_truth.json` are committed. The audio itself is reproducible from the corpus + script + seed, so the repo stays small.

A9. **Bounded corpus download**: `synthesize_english_splices.py` MUST use a targeted download strategy, not a full-corpus clone:
   - For VCTK: download the *info sheet* (speaker list, small), select 40 speakers by seed, then fetch only those speakers' files via HTTP range requests or direct per-file URLs (VCTK exposes per-speaker dirs).
   - For LibriSpeech fallback: `dev-clean` subset is 337MB; acceptable to download whole.
   - Maximum acceptable download: 1 GB. Abort with a clear error if the selection crosses this.
   - Cache downloaded files to `data_synth/corpus_cache/` (gitignored). Second run is a no-op on the download phase.

A7. **Decision gate at end of Phase A + B**: After Phase B completes, measure `cross_domain_ratio`. If ratio ≥ 0.45 after A+B alone, enter **reduced-scope mode** for Phase C:
   - Tune only 2 thresholds: `phase_gpd_alpha` and `pairwise_gate` (skip crossfade_t2_z_floor and cpe_gpd_alpha bisection).
   - Skip Phase D entirely unless post-C ratio is still < 0.55.
   This captures the antithesis's concern (Option B's evidence-based incrementality) without abandoning the Tier 4 structure. If ratio < 0.30 after A+B, proceed with full Tier 4 as written.

**Phase dependency note**: Phase C (per-domain thresholds) depends on Phase B (input normalization) because C3's threshold bisection is measured on post-normalization signal statistics. Reverting B after C would invalidate all tuned speech thresholds. Phases B and C are NOT independently revertible — treat them as a unit.

### Phase B — Input normalization (est. 1-2 hours)

B1. Add `normalize_audio(audio, sr, mode="peak")` helper in a new **`audio_io.py`** (prefer this over editing evaluate.py directly — minimizes protected-file surface):
   - `mode="peak"`: target peak = 0.95, never amplify beyond safe headroom.
   - `mode="rms"`: target RMS = -20 dBFS.
   - Return normalized array + original peak (for audit).
   - Import from evaluate.py and ml_eval.py.

B2. Apply at three consumption points:
   - `evaluate.py: evaluate()` — before `detect_splices()`. **Single-line import + one-line call** (protected-file edit requires human sign-off per Protected-file policy).
   - `ml_eval.py: evaluate_with_classifier()` — before `extract_mel_patch()`.
   - If the Downloads scanner (`scan_downloads.py` in splice-detector project) is re-run against the same pipeline, add there too. Note: `scan_downloads.py` lives in `/Users/yejunjang/Projects/mutual/mutual-website/splice-detector/scan_downloads.py`, not in this repo.

B3. Adjust `_silence_mask` threshold from fixed `-45 dBFS` to `-45 dBFS relative to normalized peak`. Since post-normalization peak is ~0 dBFS, this is effectively the same as before on normalized inputs, but now robust to any input scale.

B4. **Patch-generation normalization** (train/test parity): normalize audio at the patch-extraction layer (`.omc/classifier/generate_patches.py::extract_mel_patch`). This ensures training patches and eval-time patches see the same normalized signal. Without this, Phase B creates a silent train/test distribution mismatch: classifier trains on un-normalized pre-gen patches but predicts on normalized eval patches. Required even if D2 (gain augmentation) is skipped.

B5. **Invalidate stale caches**: delete `.omc/classifier/patches_korean_splice.npz` (loaded by `ml_eval.py:_load_korean_splice_patches`). It was generated from un-normalized audio pre-Phase-B; using it after normalization would inject un-normalized training patches into the now-normalized classifier. The cache will auto-regenerate on the next eval run with normalized audio.

B6. Commit Phase B. Re-measure all three `combined_*` metrics.

### Phase C — Per-domain thresholds (est. 3-4 hours)

C1. Add `domain_config.py`:
   - `infer_domain(audio, sr, metadata=None) → str` (one of "singing", "speech", "unknown").
   - **Inference priority (primary → fallback)**:
     1. If `metadata` dict contains `"domain"` key (read from ground_truth.json entry), use it verbatim.
     2. If the caller supplies an explicit domain tag, use it.
     3. Fallback: `sr >= 32000 → "singing"`, else `"speech"`.
   - The metadata-based path is the intended path for eval files; the sr-based fallback is only for ad-hoc inputs (e.g., scan_downloads.py). Eval-time `ground_truth.json` files MUST carry a `"domain"` field per entry (added retroactively in Phase A for English, and in a small migration for singing/korean datasets during Phase C).
   - `DOMAIN_THRESHOLDS: dict[str, dict]` with per-domain values for:
     - `phase_gpd_alpha` (0.02 singing, TBD speech)
     - `crossfade_t2_z_floor` (4.5 singing, TBD speech)
     - `pairwise_gate` (50 singing, 5 speech — already in place via fb02ee4)
     - `cpe_gpd_alpha` (0.1 singing, TBD speech)
     - `silence_threshold_db` (-45 default)

C2. Refactor `detector.py` to read thresholds from `domain_config`:
   - Pass domain tag into `_detect_phase`, `_detect_crossfade`, `_detect_cpe`, `_detect_pairwise`.
   - Replace hardcoded constants with lookups.
   - Keep the existing sr-adaptive pairwise-gate code path intact (should now become `DOMAIN_THRESHOLDS[domain]["pairwise_gate"]`).

C3. Tune speech thresholds via a short bisection grid (no full autoresearch needed — just explore one axis at a time):
   - Start with singing values.
   - For phase: try `alpha ∈ {0.05, 0.10, 0.15}` (more permissive) — pick the one that maximizes combined_speech_ko without pushing clean_fp above 5 on speech.
   - For crossfade: try `z_floor ∈ {3.0, 3.5, 4.0, 4.5}`.
   - Record picks in `domain_config.py` with comments.

C4. Commit Phase C. Re-measure.

### Phase D — Classifier retrain on 3 domains (est. 2-3 hours)

D1. Regenerate classifier patches including English:
   - Extend `ml_eval.py:_load_pregenerated_patches` to optionally load `patches_english_speech.npz` (generated by `.omc/classifier/generate_english_patches.py`, mirroring `generate_korean_patches.py`).
   - The OOF guard in ml_eval.py already keeps eval-file patches separate — extending to include English pregen patches is additive.

D2. Gain augmentation (conditional, see decision gate below):
   - In the patch-generation scripts, produce 3× copies per patch with random gain ±15 dB.
   - Complements Phase B's input normalization and B4's patch-level normalization.
   - **Decision gate**: if `cross_domain_ratio >= 0.55` after Phase C, D2 is optional. If ratio < 0.55, D2 is REQUIRED — the ratio gap cannot be closed without richer classifier invariance.

D3. **Speaker-level OOF purity** (not just file-level):
   - `patches_english_speech.npz` must NOT contain patches from the same VCTK speakers as the eval set.
   - Split VCTK speakers at Phase A: speaker IDs 1-70 → training patches, 71-110 → eval files.
   - The `synthesize_english_splices.py` script MUST enforce this split and record the assignment in `data/english-speech/ground_truth.json` (new field: `"speaker_split": "train" | "eval"`).
   - Without speaker-level split, the classifier could learn speaker-specific spectral patterns rather than splice patterns — a silent OOF leak that passes file-ID grouping but not speaker grouping.

D4. Commit Phase D. Re-measure.

### Phase E — Autoresearch re-integration (est. 1 hour)

E1. Update `run_autoresearch.sh` prompt:
   - Change `If combined > previous best:` to `If cross_domain_ratio > previous best:`.
   - Parse `cross_domain_ratio:` from `evaluate.py` output.
   - Update `verify_agent.py` to also check ratio delta.

E2. Reset `baseline_metrics.json` with post-redesign metrics (document pre-redesign baseline for rollback reference).

E3. Re-enable autoresearch on `cross-domain/apr17`. Let it run for at least 3 iterations to confirm it can make progress on the ratio metric.

E4. Final verification: run all AC1–AC12 checks, capture results in `.omc/plans/cross-domain-exit-report.md`.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| VCTK/LibriSpeech download is slow or blocked | Medium | Delays Phase A by hours | Fallback to LibriSpeech dev-clean (337MB, small). Cache to local disk; do not redownload. |
| Phase A reveals combined_speech_en is already ~0 → redesign confirmed needed but long | High | None — this is expected | Treat this as a checkpoint, not a failure. Confirms the problem is real on a second speech domain. |
| Speech threshold tuning (C3) pushes clean_fp past bound on speech | Medium | Can't hit ratio without breaking precision | Add per-domain clean_fp bound (e.g., 15 for singing, 5 for speech). Document. |
| Per-domain thresholds create a Simpson's paradox — each domain passes but average fails | Low | Hides regressions | Use `min(speech_*)` in the ratio, not average — this is already done (AC4). |
| Classifier gain augmentation degrades singing precision | Medium | combined_full drops past 0.50 floor | Gate D2 behind a Phase D1-only measurement; skip if not needed. |
| Autoresearch loop fights the redesign (reverts via git reset) | High during work | Loses progress | Pause autoresearch BEFORE starting Phase A; do not re-enable until Phase E. Use a separate branch so resets can't cross branches. |
| Eval time exceeds 120s when running 3 domains + classifier | Medium | Blocks AC7 | Architect-estimated total with all phases: 85-103s (~20s margin). **Concrete contingency**: if eval exceeds 100s after Phase D, switch to "lazy classifier" mode — train classifier on singing domain only, apply the trained model to speech domains as inference-only (no OOF training on speech). Preserves 120s budget at the cost of OOF rigor on speech (acceptable since current code has speech as DSP-only anyway). |
| `evaluate.py` protected-file violation during automated execution | High (without mitigation) | Verify-fail blocks all autopilot commits | Protected-file policy at top of plan explicitly lifts the protection on this branch with human sign-off. `verify_agent.py` is NOT invoked during Phase A–E. All evaluate.py edits are single-line additions (secondary speech-eval, single import line). Executor pauses after drafting and presents a diff to the human. |
| Stale Korean-splice patch cache contamination after Phase B | Medium | Silent classifier degradation | Phase B5 explicitly deletes `.omc/classifier/patches_korean_splice.npz`. Regeneration happens automatically on next eval. |
| Domain inference misclassifies when sr doesn't match convention | Low-Medium | Wrong thresholds applied silently | Primary inference path is metadata `"domain"` field in ground_truth.json; sr-based rule is explicit fallback only. Validated during Phase C2 refactor by running on known-domain inputs. |
| `generate_splices.py` from parent repo has incompatible dependencies | Low | Phase A blocked | Pin required libs in `data_synth/pyproject.toml` or fork the function into this repo. |

## Per-phase rollback

Every phase commits atomically so that any failure rolls back to the prior phase's HEAD. Commit messages follow `phase-X: <description>` so rollback is one-line.

| Phase | Rollback command | Recovery |
|---|---|---|
| A | `git reset --hard <phase-A-parent>` + `rm -rf data/english-speech data_synth` | Nothing downstream depends on Phase A, branch returns to pre-work state. |
| B | `git reset --hard <phase-A-head>` + `rm -f .omc/classifier/patches_korean_splice.npz` | Phase B deletions re-apply on next eval (cache auto-regens). Phase C cannot run without B (dependency note); rolling back B requires also rolling back C if C was started. |
| C | `git reset --hard <phase-B-head>` | Domain config becomes dead code; detector.py reverts to sr-adaptive pairwise + singing thresholds. |
| D | `git reset --hard <phase-C-head>` + `cp .omc/classifier/fp_classifier.joblib.bak .omc/classifier/fp_classifier.joblib` (if .bak exists) | Classifier returns to pre-D state. The `.bak` file is created automatically in train_classifier.py. |
| E | `git reset --hard <phase-D-head>` + `cp run_autoresearch.sh.bak run_autoresearch.sh` | Autoresearch config returns to pre-E; loop restartable. |
| Total abort | `git checkout autoresearch/apr15` | Abandon the entire branch; autoresearch untouched. |

## Verification Steps (end-to-end)

1. `git checkout cross-domain/apr17 && git log --oneline | head -20` shows phases A–E commits in order.
2. `ls data/english-speech/{clean,tier1,tier2}/ | wc -l` ≥ 140 (AC1).
3. `uv run python data_synth/synthesize_english_splices.py` produces reproducible files (AC2); run twice, diff sha256.
4. `uv run python evaluate.py --with-classifier > eval.log 2>&1` (AC3, AC4, AC7).
5. `grep -E '^combined|^cross_domain_ratio|^elapsed' eval.log` shows expected shape.
6. `python -c "r = 0.xx; assert r >= 0.60"` using parsed ratio (AC5).
7. `python -c "c = 0.yy; assert c >= 0.50"` using parsed singing combined (AC6).
8. `grep -E 'import (torch|tensorflow|sklearn\.neural_network|sklearn\.linear_model)' detector.py` returns empty (AC8).
9. `grep -c '^DOMAIN_THRESHOLDS\|domain_config' detector.py` > 0 (AC9).
10. Portable: `python -c "import os, datetime; p = '.omc/classifier/fp_classifier.joblib'; print(datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat())"` — verify mtime is within the Phase D commit window (AC10).
11. `for t in $(git tag -l 'detector-v*'); do git rev-parse $t >/dev/null && echo OK || echo FAIL; done` all OK (AC11).
12. `grep 'cross_domain_ratio' run_autoresearch.sh` non-empty (AC12).

## ADR (Architecture Decision Record)

**Decision**: Pursue Option A — full Tier 4 redesign on a new branch `cross-domain/apr17`, synthesizing English speech via VCTK (fallback LibriSpeech), introducing per-domain threshold tables, and retraining the classifier on three domains.

**Drivers**:
- Ratio ≥ 0.6 is the explicit success metric (deep-interview Round 3).
- User accepted large singing regression (down to 0.50) in Round 4 to buy generalization.
- User selected Tier 4 in Round 7 after Simplifier challenge.
- Existing evidence (tracer report) showed speech thresholds require structural change, not just norm.

**Alternatives considered**:
- Option B (Incremental with early exit): rejected as the primary plan but **partially absorbed as a decision gate at A7**. Option B's claim — that multi-factor fix may be overkill — is plausible but not measured. Structural analysis (pairwise gate alone moved ratio 0.00→0.21) suggests a single additional fix likely cannot close the gap to 0.60; however, the plan hedges by allowing reduced-scope Phase C and skipping Phase D if A+B already reach ratio ≥ 0.45. The ADR's earlier "mathematically unreachable" framing is softened to "structurally unlikely based on available evidence."
- Option C (Fork-and-merge): rejected because there is only one `detector.py` consumer; fork overhead > dev-safety gain.

**Why chosen**: Ratio ≥ 0.6 requires DSP-level, classifier-level, and data-level changes simultaneously. Only Tier 4 delivers all three in one effort. Autoresearch is paused, so singing regression is temporarily acceptable. User's interview signaled this choice with high confidence after contrarian mode.

**Consequences**:
- Positive: foundation for multi-domain detection; real-world iPhone amateur audio becomes tractable; future datasets plug into `DOMAIN_THRESHOLDS` without pipeline surgery.
- Negative: multi-day pause on autoresearch combined optimization (singing score may regress); codebase complexity grows (per-domain config); classifier training takes longer (three-domain patches).

**Follow-ups**:
- If ratio ≥ 0.6 is met but autoresearch can't improve on it, revisit whether autoresearch needs a different reward signal (e.g., Pareto frontier exploration over (singing, speech) tuples).
- Consider adding a third speech domain (e.g., Japanese) to stress-test generalization.
- Evaluate whether Phase D's gain augmentation belongs in the broader data-augmentation policy or only for speech.
- Once stable, merge `cross-domain/apr17` back into `autoresearch/apr15` as the new baseline.

## Changelog

### Consensus iteration 1 (architect)
Architect flagged 7 issues; all addressed:
1. B→C phase dependency note added (line 88).
2. "Mathematically unreachable" softened to "structurally unlikely" in ADR.
3. D2 gain-augmentation gated on measured ratio; B4 added for patch-level normalization parity.
4. D3 now speaker-level VCTK split (1-70 train, 71-110 eval).
5. Decision gate A7 added for evidence-based early-exit evaluation.
6. Corpus checksum manifest A6 added for Principle 2 compliance.
7. 100s eval-budget contingency (lazy classifier mode) added to risk table.

### Consensus iteration 2 (critic)
Critic issued REVISE with 1 critical, 3 major, 5 minor; addressed:
1. **CRITICAL evaluate.py protection**: new "Protected-file policy" section at top of plan explicitly lifts protection for this branch with human sign-off. Moved normalization helper to new `audio_io.py` to minimize protected-file surface.
2. Duplicate B4 numbering resolved (B4 = patch norm, B5 = cache invalidate, B6 = commit).
3. Domain inference now metadata-first with sr-based fallback (C1 revised).
4. Korean-splice patch cache invalidation step B5 added.
5. Verification step 10 made portable (Python `os.path.getmtime` instead of `stat -f`).
6. Risk table gained three new rows: protected-file violation, stale cache, domain misclassification.

### Consensus iteration 3 (critic re-review)
Critic re-review returned ACCEPT-WITH-RESERVATIONS — 3 minor items + 1 gap addressed:
1. AC8 wording clarified: sklearn permitted in ml_eval.py/train_classifier.py; only detector.py is classical-DSP-only.
2. `.gitignore` and repo-size policy added as A8: English WAVs are reproducible artifacts, not committed.
3. Bounded corpus-download added as A9: 1 GB cap with targeted speaker-selection download strategy.
4. Per-phase rollback table added before Verification Steps — explicit git commands per phase for atomic recovery.
