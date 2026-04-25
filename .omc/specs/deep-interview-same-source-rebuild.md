# Deep Interview Spec: Same-Source Dataset Rebuild

## Metadata
- Interview ID: `samesource-rebuild-2026-04-21`
- Rounds: 5
- Final Ambiguity Score: **19%** (threshold: 20%)
- Type: brownfield
- Generated: 2026-04-21
- Status: PASSED
- Handoff: `/ralph` (NOT ralplan→autopilot, per user directive)

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal | 0.85 | 0.35 | 0.298 |
| Constraints | 0.75 | 0.25 | 0.188 |
| Success Criteria | 0.80 | 0.25 | 0.200 |
| Context | 0.85 | 0.15 | 0.128 |
| **Total Clarity** | | | **0.814** |
| **Ambiguity** | | | **0.186 (19%)** |

## Goal

Rebuild `autoresearch-splice` training/eval/test data so the detector learns **same-source same-speaker edit detection** (voice-memo cut scenario), not cross-source speaker-change detection (which is what the current corpus teaches). Use naturalistic recordings where the background noise floor carries a fingerprint, so removing a chunk breaks noise-floor continuity and is physically detectable.

**Scope reduced to English-only for this rebuild wave.** Korean and Singing domain CODE paths remain intact (plugin-style registry unchanged) so future corpora can be plugged in without architectural changes.

## Constraints

1. **Single domain this wave: `english`.** Sourced from AMI + ICSI Meeting Corpora (both CC BY 4.0). Per-speaker headset channels extracted from each meeting yield 10–60 min single-speaker continuous recordings with meeting-room ambient noise floor.

2. **Naturalistic noise floor required.** −30 to −50 dBFS ambient (HVAC, projector fan, room reverb). Studio-treated recordings are rejected at the corpus-selection level — their silence is too flat to carry noise-floor splice fingerprints.

3. **Source files must be unedited single-take continuous recordings.** No post-production cuts, no multi-take stitching. Any existing splice in training material becomes an unlabeled positive and poisons the classifier.

4. **Same-source splice generation.** For each generated splice: pick ONE source file, select two NON-OVERLAPPING windows with **randomized lengths L1 + L2 = target_duration**, L1 and L2 ≥ 2s, both ends quiet-matched. Concatenate (tier 1) or linear-crossfade (tier 2). Ground truth records `source_a == source_b` and a new explicit `same_source: true` flag.

5. **Drop the `random` regime entirely.** All splices are `quiet_matched` within the new paradigm. `boundary_energy` field is constant for spliced files.

6. **Keep both tiers:** tier 1 (hard cut) and tier 2 (crossfade). Same two-tier structure, within-source now.

7. **No neural networks, no GPU.** Detector remains HistGBM over engineered features (unchanged from current `splice/classifier/train_classifier.py`). Feature extraction in `splice/features.py` remains untouched at the architecture level; the dataset change is orthogonal to features/classifier logic.

8. **Extensibility required.** `splice/dataset_registry.py` keeps all three `Dataset` entries (singing, korean, english). Singing and Korean entries have their data dirs pointing at empty directories + `eval_weight=0, train_weight=0` until data is swapped in. Per-domain generator branches in `scripts/regenerate_datasets.py` remain — just English is regenerated this wave.

9. **Licensing:** CC BY 4.0 only for new corpus material. Allows commercial use + derivative redistribution. No CC-BY-NC-ND content (rules out TED-LIUM, Pansori). Attribution required per AMI/ICSI license terms in documentation.

10. **macOS tooling available:** ffmpeg, yt-dlp, librosa, soundfile, scipy, joblib, scikit-learn 1.7.2 (pinned).

## Non-Goals

- Rebuilding Korean or Singing data this wave. (Their generator code + registry slots remain, awaiting future data.)
- Replacing the forensic evaluation metric. Current splice_f1 × clean_score geometric-mean formula is inherited as-is.
- Removing the feature extraction or classifier architecture. Dataset change is orthogonal.
- Supporting cross-source detection going forward. The old cross-source corpus is archived (for reference/rollback) but not used.
- Touching the streamlit webapp (`~/Projects/mutual/mutual-website/splice-detector/`) — user will re-snapshot that separately once new baseline is established.
- Paid corpora (LDC Korean Broadcast News) or non-commercial (CHiME-6, TED-LIUM) sources in this wave.

## Acceptance Criteria

- [ ] **Corpus acquired:** AMI Meeting Corpus (100 meetings) and ICSI Meeting Corpus (75 meetings) downloaded to `data/sources/ami/` and `data/sources/icsi/`. Per-speaker headset channels extracted as single-speaker WAV files with meeting-room ambient preserved.
- [ ] **Generator refactored:** `scripts/regenerate_datasets.py` adds a same-source splice path. For English splices: pick one source file, sample two non-overlapping windows L1 + L2 = target_duration with L1,L2 ≥ 2s, feed to `find_splice_point(..., mode='quiet_matched', ...)`, concatenate (tier 1) or linear-crossfade (tier 2).
- [ ] **Ground truth schema extended:** each English splice row in `data/<split>/english/ground_truth.json` carries `same_source: true`, `source_a == source_b`, and `source_id` (the single speaker ID).
- [ ] **English data regenerated:** `data/train/english/`, `data/eval/english/`, `data/test/english/` all populated with 20+20+20 (tier1 + tier2 + clean) splice files per split, all with `same_source: true` for spliced files. Duration envelopes match existing schema (EVAL_TRAIN_DURATION_S / TEST_DURATION_S).
- [ ] **Old encrypted blobs archived:** `data/eval.tar.gz.enc` → `data/eval.tar.gz.enc.pre-samesource`, same for `data/test.tar.gz.enc`. Rollback path preserved.
- [ ] **Plaintext trees re-encrypted:** new `eval.tar.gz.enc` + `test.tar.gz.enc` produced via `scripts/eval_crypto.py encrypt` + `scripts/test_crypto.py encrypt` (Touch ID for test). Plaintext trees shredded per existing encrypt lifecycle.
- [ ] **Registry weights updated:** `splice/dataset_registry.py` retains `Dataset` entries for `korean` and `singing` with `eval_weight=0, train_weight=0` and paths pointing at empty directories (or `None` handled gracefully). `english` stays at its current weight.
- [ ] **Classifier retrained:** `PYTHONPATH=$PWD uv run python splice/classifier/train_classifier.py` runs clean on new English data. New joblib + meta.json committed. `features_py_sha` and `train_classifier_py_sha` recorded.
- [ ] **New baseline established:** single evaluation pass produces a new `autoresearch/baseline_metrics.json` with empirically-measured combined score. No target number precommitted; whatever the first pass gives becomes the baseline autoresearch climbs from.
- [ ] **State cleared:**
  - `.omc/logs/autoresearch.jsonl` and rotated logs: deleted.
  - `.omc/research_notes.md`, `.omc/last_reflection.md`: deleted (reset agent's memory of the old task).
  - `.omc/enhancement-backlog.md`: deleted.
  - `.omc/feature_cache/`: deleted entirely (old shas are no longer relevant).
  - `results.tsv`: archived to `results.tsv.pre-samesource`, new empty one created.
  - `.omc/specs/deep-interview-*` from prior agent-drafted specs: archived to `.omc/specs/archived-pre-samesource/`.
  - `autoresearch/baseline_metrics.json`: deleted — replaced by new empirical baseline.
  - `.omc/autoresearch-stop` sentinel: removed before restart.
- [ ] **PRD reset for new task:** existing `pass=true` stories remain (they're genuinely shipped). Stories referencing singing FPs / Korean FPs / cross-source behavior (US-518B, US-505b, etc.) stay as historical record. Add a new `US-600 same-source-baseline-established` story with `passes: true` once the baseline evaluation completes successfully.
- [ ] **Loop restarted:** `./run_autoresearch.sh start` launches successfully; `loop.started` event appears in the fresh `.omc/logs/autoresearch.jsonl`; first iteration proceeds without crash.
- [ ] **24h monitor armed:** restart count reset to 0 (new T0 defined); wakeup chain scheduled at 1h cadence with "dead — new crash" protocol intact.
- [ ] **Extensibility verified:** adding a dummy domain to `dataset_registry.py` at `eval_weight=0` doesn't crash the loop. Baseline re-eval with dummy domain added still runs. Demonstrates corpus can be plugged in later.

## Assumptions Exposed & Resolved

| Assumption | How it was challenged | Resolution |
|---|---|---|
| Current metric (splice_f1) measures cut-edit detection | User pointed out ALL current splices are cross-source (`source_a ≠ source_b` always) | Confirmed in ground_truth.json — the "detector" is a speaker-change classifier |
| Quiet-matched within same audio file is the right regime | User pushed back on naive half-length windows | Windows are randomly-sized (L1 + L2 = target, each ≥ 2s), not half-length |
| YouTube scraping is the answer | User: "most youtube videos will have cuts" | YouTube rejected as primary source — editing pollutes training signal |
| Studio-recorded data is fine | User: "studio is acoustically quiet — no noise floor" | Studio-treated recordings (LibriSpeech, Zeroth, KSponSpeech, LibriVox) all rejected |
| Concatenating utterances builds long-enough sources | User: "concat is also a splice" | Concatenation rejected — would inject unlabeled positives |
| 3 domains required | Research showed Korean corpus landscape is bad | Scope reduced to English-only THIS wave; architecture preserved for future re-addition |
| Need precommitted baseline target | Considered >0.5, >0.25, forensic AUC-PR alternatives | Decision: inherit metric formula, no target — let first eval establish the baseline |
| Rebuild means replacing architecture | User: "leave architecture intact, I want it extendable" | Only data changes; dataset_registry.py + per-domain generator code preserved |

## Technical Context (brownfield)

**Current code locations relevant to rebuild:**

- `scripts/regenerate_datasets.py` — splice generator. L384–447 contains the cross-source splice loop (`_pair()` picks two different sources). This is the primary edit site for adding the same-source path.
- `scripts/splice_boundary.py` — `find_splice_point(seg_a, seg_b, sr, ..., mode='quiet_matched')` already works for quiet-matched mode. Same API can accept two windows from the same source.
- `splice/dataset_registry.py` — `DATASETS: list[Dataset]` (US-200 shipped this plugin pattern). Each entry has `id`, `path`, `eval_weight`, `train_weight`.
- `splice/classifier/train_classifier.py` — iterates `DATASETS` with `train_weight > 0`. No changes needed; dormant domains just get skipped.
- `splice/evaluate.py` — iterates eval-weighted domains. Same story.
- `splice/features.py` (80 features) — unchanged.
- `splice/detector.py` (HistGBM + DSP) — unchanged.
- `splice/classifier/fp_classifier.{joblib,meta.json}` — regenerated after new training.
- `data/eval.tar.gz.enc`, `data/test.tar.gz.enc` — encrypted blobs, re-encrypted after plaintext regeneration.
- `data/sources/` — add `ami/` and `icsi/` subdirs.
- `run_autoresearch.sh` — loop wrapper, no changes needed. Auto-retrain gate (US-505b) picks up new classifier via sha.

**AMI / ICSI acquisition notes:**

- AMI: https://groups.inf.ed.ac.uk/ami/download/ — individual headset WAV channels (IS1000a.Mix-Headset.wav etc. per meeting + speaker) are directly downloadable.
- ICSI: same URL structure (https://groups.inf.ed.ac.uk/ami/icsi/download/).
- Both CC BY 4.0. Attribution required.
- Total expected disk: AMI ~60 GB, ICSI ~40 GB raw audio. Extracted per-speaker headset tracks: ~30 GB combined.
- Speaker diarization is pre-annotated in AMI/ICSI at word level; can be used to filter to single-speaker segments of ≥2 min continuous speech.

## Ontology

| Entity | Type | Fields | Relationships |
|---|---|---|---|
| Corpus | data source | id, path, license, per_file_length_range, acoustic_env | has many Source |
| Source | data source | source_id, language, speaker_id, file_path, duration_s, noise_floor_dbfs | belongs to Corpus, produces Splice |
| Splice | core domain | filename, tier, splice_time_sec, same_source, boundary_energy, source_a, source_b, crossfade_ms, regime | derived from Source(s) |
| Clean | core domain | filename, source_id, duration_s | derived from single Source |
| Detector | core domain | features (80), classifier (HistGBM), thresholds | predicts on Splice + Clean |
| User | actor | records custom data, initiates rebuild, runs autoresearch | operates Detector, curates Corpus |
| License | supporting | name (CC BY 4.0), commercial_use_ok, ml_training_ok, redistribution_ok | governs Corpus |
| AcousticEnv | supporting | noise_floor_dbfs, room_type, recording_device | characterizes Source |

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Stability |
|---|---|---|---|---|---|
| 1 | 6 | 6 | 0 | 0 | N/A |
| 2 | 6 | 0 | 0 | 6 | 100% |
| 3 | 8 | 2 (Corpus_candidate, License) | 0 | 6 | 75% |
| 4 | 7 (Corpus_candidate merged into Corpus) | 0 | 1 | 6 | 100% |
| 5 | 8 (+Clean, made explicit) | 1 | 0 | 7 | 88% |

## Interview Transcript

<details>
<summary>Full Q&A (5 rounds)</summary>

### Round 1 — Acoustic scope
**Q:** What acoustic environment range does the detector need to handle?
**A:** Phone-mic home/outdoor non-studio — all good EXCEPT source data must not have cuts/splices (most YouTube will have these).
**Key insight:** Added the "no pre-existing cuts in source" constraint that rules out edited content.
**Ambiguity after:** 42%

### Round 2 — Effort budget
**Q:** What's the effort budget for corpus sourcing given the constraints?
**A:** Research what exists, decide later.
**Key insight:** Trigger a research pass before committing to a corpus.
**Ambiguity after:** 38%

### Round 3 — Corpus research (non-question round)
Research agent surveyed: TED-LIUM, AMI, ICSI, CHiME-6, GigaSpeech, People's Speech, LibriVox, VoxPopuli, VoxForge, MuST-C, Korean Broadcast News LDC, KsponSpeech, ClovaCall, Zeroth, NIKL, Pansori TEDx, Sebasi.
**Key finding:** English has 2 STRONG FITs (AMI + ICSI, both CC BY 4.0). Korean has no clean fit.
**Ambiguity after:** 35%

### Round 4 — Domain scope (Contrarian Mode)
**Q:** Given the Korean corpus gap, what domain scope for the rebuild?
**A:** English-only (AMI + ICSI). Drop Korean and singing.
**Key insight:** Reduced scope to what has clean data. Architecture-preservation ask came later.
**Ambiguity after:** 31%

### Round 5 — Success metric
**Q:** How to evaluate? Match old baseline (0.589)? Any signal > random? Replace metric?
**A:** Inherit old metric formula, no target yet — let first eval establish new baseline empirically.
**Key insight:** Scientifically honest. No precommitment to a number we can't calibrate in advance.
**Ambiguity after:** 19% — threshold met.

### Post-threshold clarification
**User:** "I'm going to add in data in the future by the way. Leave the architecture intact because I want it to be extendable later on."
**Resolution:** Preserve dataset_registry entries for all 3 domains + per-domain generator code. Only English DATA gets rebuilt; multi-domain CODE pathways remain for later corpus plug-in.

</details>
