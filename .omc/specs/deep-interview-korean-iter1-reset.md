# Deep Interview Spec: korean-iter1 corpus pivot + framework reset

## Metadata
- Interview ID: korean-iter1-reset-2026-04-25
- Rounds: 6 (1 dataset-sanity sidebar + 5 design questions)
- Final Ambiguity: **7.5%**
- Type: brownfield
- Generated: 2026-04-25
- Threshold: 20%
- Status: **PASSED** (well under threshold)

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|------:|-------:|---------:|
| Goal Clarity      | 0.95 | 0.35 | 0.333 |
| Constraint Clarity| 0.95 | 0.25 | 0.238 |
| Success Criteria  | 0.85 | 0.25 | 0.213 |
| Context Clarity   | 0.95 | 0.15 | 0.143 |
| **Total Clarity** | | | **0.925** |
| **Ambiguity**     | | | **0.075** |

## Goal

Replace the singing/korean/english multi-domain eval framework with a single-corpus pipeline built around the korean-iter-1 synthetic dialogue dataset. The autoresearch loop optimises a **3-class boundary-F1 with 250 ms collar** on an augmented version of that corpus, where ground truth includes both naturally-occurring cross-voice splices (turn boundaries baked into the iter-1 audio) and synthetically-injected same-voice content edits (word-level cuts at silence-bounded positions in the existing audio). The agent freely modifies `splice/detector.py`, `splice/features.py`, and `splice/classifier/train_classifier.py` to maximise the new metric.

## Constraints

- **Corpus**: `/Volumes/HIKSEMI/korean-iter-1-delivery.tar` (20 GB, 7360 conversations, 99 h, 11 ElevenLabs TTS voices, 6 ambience classes, 50/50 wav/mp3, 44.1 kHz mono, mean 48.5 s).
- **Compute**: scipy/librosa/numpy + sklearn HistGradientBoosting only. NO neural nets, NO GPU.
- **Eval budget**: 240 s wall-clock for `splice/evaluate.py`.
- **Test budget**: 1200 s wall-clock for held-out test evaluation.
- **Augmentation chain (deterministic, file-hash seeded)**:
  - Pink-filtered white noise mixed at SNR ~ N(22, 4) dB (matches reference real-world recording).
  - Synthetic exponential-decay RIR with T60 ~ U(0.2, 0.6) s.
  - Opus 32 kbps codec round-trip.
  - All seeded by `hash(file_id)` so eval is reproducible across runs.
- **Splits (voice-pair holdout)**:
  - test voices: 2 of 11 (e.g. DaeBuHo, Ondo) — ~1300 files
  - eval voices: 2 of 11 (e.g. Sunwoo, Joon) — ~1100 files
  - train voices: remaining 7 — ~4500 files
  - Per-iteration eval = sample 60 from eval pool.
  - Held-out test = sample 200–400 from test pool (1200 s budget).
- **Same-voice edit injection**:
  - 1–3 word-level cuts per file, deterministic by file hash.
  - Constraint: silence ≥ 120 ms before AND after the deleted span (32.7% of words qualify).
  - Cut span removed; audio re-joined at the silence boundary; ground truth records the join timestamp.
- **Ground-truth labels (3-class)**:
  - `cross_voice` — natural turn boundaries from the iter-1 transcript (`start_ms` of turn N>0).
  - `same_voice_edit` — synthetic word-cut joins, exact timestamps.
  - `no_splice` — anywhere else.
- **Detector contract**: `detect_splices(audio, sr) -> list[(time_s, label)]` where label ∈ {cross_voice, same_voice_edit}. (Backward-compatible with `list[float]` if the agent ignores label and emits unlabeled boundaries.)
- **Verification**: `autoresearch/supervisor_agent.py --verify` (4-check audit unchanged in spirit, but `baseline_metrics.json` schema migrates).
- **Branch**: fresh `autoresearch/korean-iter1` from master. `autoresearch/apr15` archived as-is.
- **Storage**: corpus stored decrypted under `data/eval/korean_iter1/` (TTS-synthetic, no speaker-protection concern).

## Non-Goals

- Real-world recording labelling or eval-loop integration. The user's m4a samples are spot-check material only.
- Multi-variant augmentation (one fixed augmentation chain per file, not N variants).
- External noise / RIR corpora (MUSAN, BUT ReverbDB). Pure-procedural augmentation.
- Iter-2 corpus commission. Same-voice splices are synthesised from iter-1.
- Preserving the singing / english / korean (Zeroth) datasets as active eval domains. They stay in `dataset_registry.py` at `weight=0` for code-path continuity but contribute nothing to the score.
- Sentence-level or multi-word edit injection (word-only in this pivot; can be added later if word-only proves trivially solvable).
- Modifying `splice/evaluate.py` or `splice/program.md` outside the schema migration.

## Acceptance Criteria

- [ ] Branch `autoresearch/korean-iter1` exists, off master HEAD; `autoresearch/apr15` is left untouched.
- [ ] `data/eval/korean_iter1/{train,eval,test}/` populated by a regenerator script that:
      - extracts iter-1 audio + transcripts from the tarball
      - splits files by voice membership (2 test / 2 eval / 7 train voices)
      - injects 1–3 same-voice word-cuts per file (deterministic seed)
      - applies the augmentation chain (pink noise + synth RIR + Opus 32k roundtrip, deterministic seed)
      - writes ground-truth JSON per file with `boundaries: [{time_s, label}]`
- [ ] `splice/dataset_registry.py` has `korean_iter1` (weight=1, eval_weight=1, train_weight=1); singing/korean/english remain at weight=0.
- [ ] `splice/evaluate.py` migrated to:
      - iterate `data/eval/korean_iter1/eval/`
      - compute boundary-F1 with 250 ms collar (one prediction matches one GT, greedy nearest within collar)
      - print `combined: <boundary_f1>` as the last line (preserves wrapper grep contract)
- [ ] `autoresearch/baseline_metrics.json` has the new schema: `{combined: <f1>, precision, recall, n_files, collar_ms, splice_class_breakdown: {cross_voice_f1, same_voice_edit_f1}, ...}`.
- [ ] `splice/classifier/train_classifier.py` retrains the GBM on 3-class targets from `data/eval/korean_iter1/train/`. The `fp_classifier.joblib` artifact is rebuilt.
- [ ] Wipe completed: old `baseline_metrics.json`, `results.tsv`, `.omc/feature_cache/*`, `splice/classifier/fp_classifier.joblib*` are gone or moved to `.omc/archive/apr15-snapshot/`.
- [ ] `autoresearch/supervisor_agent.py --verify` passes on the first iter-1 run (preflight green, schema audit green, anomaly detection green).
- [ ] `run_autoresearch.sh start` runs the loop; first 3 iterations all complete without crash; results.tsv accumulates rows.
- [ ] `splice/program.md` updated with the new corpus / metric / class scheme (this file is human-edited only — flagged for the user to review/rewrite).

## Assumptions Exposed & Resolved

| Round | Assumption surfaced | Resolution |
|------:|---------------------|------------|
| 1 | "Iter-1 contains raw single-voice TTS chunks" (user's expectation) | Verified false: pre-rendered multi-voice dialogues with ambience baked in. User accepted as the eval corpus shape. |
| 2 | "`combined = splice_f1 × clean_score` is salvageable" | Rejected. Pivot to boundary-F1 with collar (SAD/diarization standard). No clean_score, no `clean_fp ≤ 15` guard. |
| 3 | "Augmenting 99 h of audio is expensive" | Benchmarked: ~12 min on M1 8-core for full pipeline. Cost is essentially free; doesn't constrain the design. |
| 4 | "Real-world recordings should be in the eval loop" | Rejected. Single-corpus boundary-F1 only; user's real m4a recordings are spot-check material. Real-world transfer is hoped-for, not measured. |
| 4 | "Augmentation requires external noise/RIR corpora for realism" | Rejected. Pure-procedural, file-hash-seeded augmentation. License-clean, verifier-friendly. |
| 5 | "Voice memorisation is acceptable in eval" | Rejected. Voice-pair holdout (2 test / 2 eval / 7 train) forces the classifier to learn voice-agnostic boundary features. |
| 6 (contrarian) | "Splice == voice change is the project's actual goal" | Rejected. User's word_alignment-based edit-out idea unlocked same-voice content edits at scale (85% of words editable at ≥80 ms padding). 3-class detector (cross_voice / same_voice_edit / no_splice). |
| 7 | "Continue iterating on `autoresearch/apr15`" | Rejected. Fresh branch off master, full wipe of baseline + history + classifier bundle, apr15 archived as-is. |

## Technical Context

- **Repo state at interview**: branch `autoresearch/apr15`, loop STOPPED (`.omc/autoresearch-stop` present), baseline `combined=0.589` (en 0.89 / ko 0.67 / sg 0.35), 167 iteration rows in `results.tsv`.
- **In-flight US-600 work**: was rebuilding english around AMI+ICSI for "same-source" splice generation (`progress.txt`, `.omc/specs/deep-interview-same-source-rebuild.md`). This pivot **supersedes US-600** — same-voice splices are now synthesised from iter-1 instead.
- **Editable files**: `splice/{detector,features,classifier/train_classifier}.py` (agent edits these). `splice/evaluate.py` and `splice/program.md` are human-edited only — both need a one-time maintainer migration as part of this reset.
- **Protected by verifier**: `splice/evaluate.py`, `splice/program.md`, `data/eval/**/*`, `data/test/**/*`, `autoresearch/manifest.json`, `autoresearch/preflight.py`. Verifier diff-audit will need a one-time exemption commit when these change.
- **Carve-outs that survive**: unified logging (US-515), supervisor maintain (US-517), retest sentinel — all schema-stable across the pivot.
- **Augmentation pipeline benchmark** (M1, this session):
  - Per file (full chain): ~0.7 s for ~40 s audio (~57× real-time)
  - Codec round-trip dominates (~65% of cost)
  - Whole corpus on 8 cores: ~12 min
- **Inter-word gap distribution** (sampled 200 conv, 14k words): median 120 ms, 85% of words have ≥80 ms padding both sides, 33% have ≥120 ms padding (the chosen edit threshold).
- **Reference real-world stats** (user's `음성 260417_161113_편집.m4a`, first 30 s):
  - silence floor −37.8 dB, voice RMS −15.4 dB, SNR ~22 dB
  - centroid 1113 Hz, rolloff99 8613 Hz, F0 ~134 Hz
  - target augmentation parameters chosen to match these.

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| Conversation | core domain | id, idx, format, duration_ms, ambience, speakers[], turn_count | has many Turns; belongs to Voice-Pair Split |
| Turn | core domain | idx, speaker, voice_id, voice_name, start_ms, end_ms, text, word_alignment[] | belongs to Conversation; has many Words |
| Word | core domain | word, start_ms, end_ms, gap_before_ms, gap_after_ms, editable | belongs to Turn |
| Voice | supporting | voice_name (1 of 11), gender, age, accent | appears in many Turns |
| Boundary (GT) | core domain | time_s, label ∈ {cross_voice, same_voice_edit} | belongs to Conversation |
| Augmentation | core domain | seed (hash), snr_db, t60_s, codec_bitrate | applied to Conversation |
| Split | supporting | name ∈ {train, eval, test}, voice_holdout[] | partitions Conversations |
| Detector Output | core domain | time_s, score, predicted_label | matched against Boundary within collar |
| Eval Result | core domain | combined (boundary_f1), precision, recall, per_class_f1 | written to results.tsv + baseline_metrics.json |

## Ontology Convergence

| Round | Entity Count | New | Stable | Stability |
|------:|-------------:|----:|-------:|----------:|
| 1 | 4 | 4 | – | N/A |
| 2 | 5 (+Boundary) | 1 | 4 | 80% |
| 3 | 6 (+Augmentation) | 1 | 5 | 83% |
| 4 | 7 (+Split) | 1 | 6 | 86% |
| 5 | 8 (+Detector Output) | 1 | 7 | 88% |
| 6 | 9 (+Eval Result) | 1 | 8 | 89% |

Domain model converged steadily; no entity was renamed or removed across rounds.

## Interview Transcript

<details>
<summary>Full Q&A (6 rounds + dataset-sanity sidebar)</summary>

### Sidebar — Dataset sanity check
**Trigger:** User asked "I thought the dataset was supposed to give my unedited audio. Is it not?"
**Investigation:** Probed waveform RMS at transcript timestamps for `conversation_00088.mp3`; rendered annotated + raw spectrograms; rendered LTAS overlay vs user's reference m4a; computed 16 statistics across 30-s slices.
**Finding:** Audio is pre-rendered multi-voice + ambience-mixed. Studio gap vs real-world: silence floor 10 dB lower, HF artifacts above 8 kHz, 60% higher centroid std at boundaries. Augmentation cost ~12 min on M1.

### Round 1 — Metric pivot (Success Criteria)
**Q:** What replaces `combined = splice_f1 × clean_score` given there are no clean files?
**A:** [Asked but redirected to dataset-sanity sidebar.]

### Round 2 — Eval target (Success Criteria, post-sidebar)
**Q:** What should the agent's optimisation target measure given the studio gap?
**A:** Augmented-corpus boundary-F1 with collar, no real-world holdout.
**Ambiguity:** 68% → 40%

### Round 3 — Augmentation chain (Constraint Clarity)
**Q:** What augmentation pipeline converts the studio-clean corpus into train/eval/test?
**A:** Synthetic everything (pink noise + exp RIR + Opus 32k), deterministic seed, no external corpora.
**Ambiguity:** 40% → 32%

### Round 4 — Splits + stratification (Success Criteria)
**Q:** How should 7360 files split, and on which axis?
**A:** Voice-pair holdout — 2 test voices, 2 eval voices, 7 train voices; eval=60 sampled per iter, test=200–400.
**Ambiguity:** 32% → 20%

### Round 5 — CONTRARIAN: is this really splice detection?
**Q:** Iter-1 has only voice-change boundaries. The detector will learn speaker change detection, not splice detection. Which problem are you actually solving?
**A:** [User redirected — "isn't the dialogue timestamped? You can edit-out specific words/sentences if there's silence before and after."]
**Investigation:** Probed inter-word gap distribution. 85.1% of words have ≥80 ms silence both sides; 32.7% at ≥120 ms. Plenty of edit-out positions.
**A (re-posed):** Cross-voice + same-voice word-cuts, mixed labels (3-class).
**Ambiguity:** 20% → 12%

### Round 6 — Reset scope (Context Clarity)
**Q:** How aggressive should the baseline-clear be?
**A:** Fresh branch off master, full wipe (baseline_metrics.json, results.tsv, feature_cache, classifier bundle), apr15 archived as-is, singing+english kept at weight=0 in registry.
**Ambiguity:** 12% → 7.5%

</details>

## Open operational items (non-blocking, defaults applied)

These were not interviewed individually but have working defaults baked into the spec; the user can override at execution time:

- **Collar size**: defaults to 250 ms (DIHARD/dscore convention). Tunable later if too forgiving / strict.
- **Encryption**: corpus stored decrypted under `data/eval/korean_iter1/` (TTS-synthetic). `eval_crypto.py` not invoked.
- **Same-voice edit count per file**: 1–3 (uniform random, file-hash seeded). Adjustable by single constant.
- **English/singing in registry**: kept at weight=0 (code paths preserved). Not deleted.
- **Loop safety preconditions**: `.omc/autoresearch-stop` removed by user before `start`; `.omc/retest-in-progress` absent (currently is); `.omc/supervisor-crash-counter.txt` reset to 0.
