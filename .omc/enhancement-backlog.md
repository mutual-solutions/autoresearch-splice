# Enhancement Backlog

**Purpose:** Track wrapper/pipeline enhancement requests raised by the autoresearch agent in `research_notes.md` `(e) Wrapper enhancements` sections. Maintained by the periodic maintainer cron (`.claude/scheduled_tasks.json` entry, fires every 4 hours).

**Format:** One H2 per request. Fields:
- **id** — stable kebab-case identifier (used for fuzzy-match dedup)
- **status** — `pending` | `implemented` | `rejected` | `deferred` | `needs_human_review`
- **first_seen / last_seen** — hypothesis SHAs where first/most recently raised
- **request_count** — number of iterations raising this (rough recurrence)
- **category** — `observability` | `control-flow` | `new-script` | `protocol-change` | `new-subcommand`
- **risk** — `low` (autonomous impl OK) | `medium` (draft ralplan, human approves) | `high` (always needs human design pass)
- **excerpt** — verbatim quote from most recent raise
- **notes** — maintainer triage reasoning or impl references

## shap-delta-block

- **status:** pending
- **first_seen:** e228236
- **last_seen:** a4a01d9
- **request_count:** 2+
- **category:** observability
- **risk:** medium (needs SHAP rollup comparison logic + prompt composer change)
- **excerpt:** "Surface a SHAP-DELTA block in CURRENT STATE: for the most recent KEPT classifier vs prior KEPT classifier, show per-feature SHAP change. New features with near-zero SHAP are being ignored regardless of metric outcome — early pivot signal."
- **notes:** The rollup source is `.omc/shap_rollup.json` (regenerated per iteration by `scripts/shap_rollup.py`). Implementation requires: (1) track last-KEEP rollup snapshot alongside current, (2) compute per-feature delta, (3) compose a compact block, (4) inject into claude prompt alongside existing TOP PREDICTIVE FEATURES block. Medium because it touches the prompt composer + needs a snapshot-at-keep mechanism.

## singing-fp-dump

- **status:** pending
- **first_seen:** a4a01d9
- **last_seen:** a4a01d9
- **request_count:** 1 (recent + specific)
- **category:** observability
- **risk:** medium (requires one-time evaluate.py maintainer exemption)
- **excerpt:** "A SINGING-FP-DUMP block in CURRENT STATE: for each KEPT classifier, list the (file, t_sec, p_hard, p_cross) tuples for the 6 singing clean FPs. Direct visibility into the failure mode lets me design hypotheses that actually target the cluster instead of guessing."
- **notes:** `evaluate.py` is PROTECTED. Implementation requires: (1) maintainer-authored evaluate.py emit of `eval.fp.detail` JSONL per clean-FP in the singing domain, (2) verify_agent regex update (PROTECTED_FILES untouched; per-file emissions don't change the file's protected status), (3) wrapper composer reads latest `eval.fp.detail` events from JSONL and injects block. Cost/benefit favors implementation given singing is the stuck domain.

## per-class-oof-f1

- **status:** pending
- **first_seen:** a4a01d9
- **last_seen:** a4a01d9
- **request_count:** 1
- **category:** observability
- **risk:** low (train_classifier.py emit expansion + prompt composer additional line)
- **excerpt:** "Per-class OOF F1 in CURRENT STATE alongside the aggregate weighted F1 — currently I only see the aggregate."
- **notes:** `splice/classifier/train_classifier.py` already computes OOF weighted F1. Expand the `classifier.retrain.complete` JSONL event to include `oof_f1_hard_cut`, `oof_f1_crossfade`, `oof_f1_not_splice` fields (sklearn.metrics.f1_score with `average=None`). Wrapper composer adds a one-line per-class breakdown to CURRENT STATE. Pure additive change; no risk.

## verify-agent-replay-subcommand

- **status:** pending
- **first_seen:** e228236
- **last_seen:** a4a01d9
- **request_count:** 3+ (explicitly mentioned in 3 recent reflections)
- **category:** new-subcommand
- **risk:** high (new verify_agent subcommand, git cherry-pick surface, baseline-measurement handling; structurally similar to `--retest` but semantically different)
- **excerpt:** "`verify_agent.py --replay <commit> --against HEAD` re-runs evaluate.py for a prior hypothesis commit under the current code state; essential for confirming bug-killed hypotheses post-fix."
- **notes:** THE HIGHEST-LEVERAGE UNBLOCK. Three pre-US-510 hypotheses (isotonic `d25f455`, phase residual, FILE-LEVEL flatness) got combined=0.010 catastrophic discards from the `shap_report.py` AttributeError, NOT from hypothesis content. The no-repeat rule locks them out. `--replay` reruns the hypothesis's tunable changes on current code, unlocking re-evaluation without violating no-repeat. Design must mirror `--retest` for loop-safety (sentinel, rolling baseline) but without the "cherry-pick AND baseline commit" pairing — replay is a one-shot measurement, not a recovery. High risk because (a) new CLI surface, (b) git state manipulation in main tree, (c) baseline interaction. Needs ralplan consensus, not autonomous impl.

## failure-reason-tag-on-history

- **status:** pending
- **first_seen:** f6192f6
- **last_seen:** 0feddf9
- **request_count:** 3
- **category:** protocol-change
- **risk:** medium (heuristic classifier + wrapper HISTORY composer change)
- **excerpt:** "A `failure_reason: pipeline_bug | hypothesis_content` tag on HISTORY entries would unblock re-exploration of those axes."
- **notes:** Complements `--replay`. Heuristic: combined < 0.05 AND log has AttributeError/TypeError/ImportError crash → `pipeline_bug`. Otherwise → `hypothesis_content`. Applied retroactively to results.tsv rows at HISTORY-compose time. Auto-tagging of catastrophic 0.010 entries (another request) is the same mechanism.

## per-domain-file-stats-script

- **status:** spec_drafted
- **first_seen:** 39b9fb0
- **last_seen:** 0feddf9
- **request_count:** 3
- **category:** new-script
- **risk:** low (new script in scripts/; no existing code paths changed)
- **excerpt:** "A `scripts/per_domain_file_stats.py` emitting mean HPR/flatness/voicing per domain on EVAL (not training) so file-level gating thresholds come from the evaluation distribution."
- **notes:** Walk `data/eval/{singing,korean,english}/{tier1,tier2,clean}/`, compute per-file HPR (via librosa.decompose.hpss), flatness, voicing; report per-domain mean + quantile distribution. Enables the agent to design file-level routing gates using eval-corpus statistics instead of training-set heuristics. Pure read-only analysis tool; safe to auto-implement.

## shap-rollup-protocol-improvements

- **status:** deferred
- **first_seen:** e228236
- **last_seen:** e228236
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (cluster of "(e) Wrapper enhancements" sub-items across multiple reflections, all around SHAP observability — mostly captured by shap-delta-block above)
- **notes:** Deferred pending shap-delta-block impl; re-evaluate after that lands.

---

# Maintainer cron policy

**Autonomy boundary for autonomous implementation:**
- ✅ **Low-risk (auto-implement):** new script under `scripts/`, new JSONL emit site in existing emitter (non-PROTECTED), docstring fixes, `.gitignore` additions, README/CLAUDE.md doc updates.
- ⚠️ **Medium (draft ralplan, surface for human approval):** wrapper prompt composer additions, protocol changes (HISTORY format, etc.), evaluate.py edits (PROTECTED — always requires maintainer exemption), verify_agent subcommands.
- 🚫 **High (always human):** loop control flow, `_guarded_reset` paths, new sentinel mechanics, PROTECTED_FILES regex changes, classifier training pipeline.

**Loop-safety invariants (cron MUST respect):**
- NEVER `tmux kill-session -t autoresearch` (loses the iteration's in-flight work).
- NEVER delete `.omc/autoresearch-stop` unless finishing and explicitly restarting the loop.
- BEFORE any commit: send `./run_autoresearch.sh stop`, poll JSONL tail until `wrapper.loop.clean_shutdown` event appears, THEN commit, THEN restart if desired.
- NEVER commit during `retest-in-progress` sentinel presence.

**Dedup rule:**
Fuzzy match new research_notes enhancement-request bullets against backlog entries' `excerpt` field (case-insensitive token-set Jaccard ≥ 0.5). Match → bump `last_seen` + `request_count`. No match → new H2 entry with status=`pending`.

**Rejection criteria:**
A request moves to status=`rejected` (or `deferred`) when:
- It's been `needs_human_review` for >3 cron fires without action (defer automatically, note reason).
- Maintainer explicitly rejects via a commit touching this file with a `rejected` status change.
- It's a duplicate of an `implemented` item (dedup caught it after impl landed).
## 1-emit-a-chunklevelfeatureratio-metric-a
- **status:** pending
- **first_seen:** 0c5ba88
- **last_seen:** 0c5ba88
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) Emit a CHUNK-LEVEL-FEATURE-RATIO metric at train-time: for each FEATURE_NAMES entry, print the per-chunk coefficient of variation (std/mean within a chunk). Features with CV≈0 are constant-per-chunk (like file_hpr by design), letting me verify the feature was computed correctly and letting futur
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## 2-verify_agentpy-shapforfeature-name-pri
- **status:** pending
- **first_seen:** 0c5ba88
- **last_seen:** 0c5ba88
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) `verify_agent.py --shap-for-feature <name>` printing global SHAP rank + mean |SHAP| for a specified feature across the last N KEPT classifiers. Would let me detect when a new feature is truly being used by the model vs carried along at SHAP≈0.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## 3-a-datatrainclean_fp_positionsjson-auto
- **status:** pending
- **first_seen:** 0c5ba88
- **last_seen:** 7f5f0a0
- **request_count:** 3
- **category:** observability
- **risk:** medium
- **excerpt:** (3) A `data/train/clean_fp_positions.json` auto-emitted by evaluate.py listing (domain, file, t_sec, p_hard, p_cross) for every clean-file false positive — turns "singing clean_fp=6" from an opaque scalar into actionable per-position diagnostics that directly drive feature-design.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## 1-surface-evalside-perdomain-distributio
- **status:** pending
- **first_seen:** f10993b
- **last_seen:** f10993b
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) Surface eval-side per-domain distribution summary at top of CURRENT STATE (mean/min/max HPR / flatness / voicing per domain on the EVAL set) so scalar cutoffs are data-driven.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## 2-autoannotate-history-entries-with-fail
- **status:** pending
- **first_seen:** f10993b
- **last_seen:** f10993b
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) Auto-annotate HISTORY entries with `failure_reason` from eval.stderr pattern match (pipeline_bug / feature_mismatch / hypothesis_content).
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## 3-a-scriptsclean_fp_positionspy-last-cli
- **status:** pending
- **first_seen:** f10993b
- **last_seen:** f10993b
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) A `scripts/clean_fp_positions.py --last` CLI emitting the most recent KEPT classifier's singing FP positions (file, t_sec, p_hard, p_cross, p_splice) so feature-design and post-filter hypotheses can target actual failure cases instead of guessing.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## shap-delta-block-in-iteration-prompt-for
- **status:** pending
- **first_seen:** 7f5f0a0
- **last_seen:** 7f5f0a0
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) A SHAP-DELTA block in the iteration prompt: for the new KEPT classifier vs prior KEPT classifier, per-feature mean-|SHAP| change. Directly shows whether a newly-added feature is being USED or IGNORED, eliminating the current guessing in post-mortems.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsfeature_scale_comparisonpy
- **status:** pending
- **first_seen:** 7f5f0a0
- **last_seen:** 7f5f0a0
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `scripts/feature_scale_comparison.py <feature_name>` — plots the feature distribution on eval-splice-positive vs eval-clean-positive rows across domains, so a new feature's discrimination can be visualized BEFORE committing it.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## datatrainclean_fp_positionsjson-auto
- **status:** pending
- **first_seen:** bba6dbe
- **last_seen:** bba6dbe
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) `data/train/clean_fp_positions.json` auto-emitted by evaluate.py listing the 6 singing FPs' `(file, t_sec, p_hard, p_cross, p_splice)` tuples — turns an opaque scalar into actionable per-position diagnostics.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptstraining_aug_visualizepy
- **status:** pending
- **first_seen:** bba6dbe
- **last_seen:** bba6dbe
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) A `scripts/training_aug_visualize.py <feature_name>` plotting the feature distribution on original vs augmented training rows per domain — lets me verify the augmentation moved the not_splice distribution in the INTENDED direction (toward FP cluster) before committing.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## pre-iteration-retrain-cost-estimator
- **status:** pending
- **first_seen:** bba6dbe
- **last_seen:** bba6dbe
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) Pre-iteration retrain-cost estimator: `verify_agent --estimate-retrain-cost` that simulates the extra row count and projected retrain time from per-row timing — so I can judge whether a data-augmentation hypothesis will fit in budget before iterating. [auto] LOCAL KEEP at bba6dbe: top-3 features
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptstraining_aug_visualizepy---shift
- **status:** pending
- **first_seen:** f71c526
- **last_seen:** eabd6b2
- **request_count:** 3
- **category:** observability
- **risk:** medium
- **excerpt:** (1) `scripts/training_aug_visualize.py --shift +1 --feature spec_rolloff_delta` plotting augmented vs original feature distributions per domain on a representative sample, so I can verify aug lands in the intended feature-space region BEFORE committing.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean-fp-positions-block-in-current
- **status:** pending
- **first_seen:** f71c526
- **last_seen:** 423014c
- **request_count:** 2
- **category:** observability
- **risk:** medium
- **excerpt:** (2) A CLEAN-FP-POSITIONS block in CURRENT STATE listing the k clean-file false positives from the last KEPT classifier — turning "korean clean_fp=4" into actionable per-position diagnostics.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## supervisor_agentpy---class-breakdown
- **status:** pending
- **first_seen:** f71c526
- **last_seen:** f71c526
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `supervisor_agent.py --class-breakdown <commit>` emitting (domain, class_label, n_fp, n_tp) from a prior keep so I can target whether to widen not_splice (reduce FP) or widen splice classes (reduce FN) per domain.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## auto-retrain-trigger-log-one-line-per
- **status:** pending
- **first_seen:** d5cc597
- **last_seen:** d5cc597
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) An AUTO-RETRAIN TRIGGER LOG: one line per eval attempt showing "retrained: YES (features.py sha diff) | NO (using disk joblib sha X)". Turns the silent classifier staleness risk into a visible signal.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsclass_balance_probepy-emitting
- **status:** pending
- **first_seen:** d5cc597
- **last_seen:** d5cc597
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) `scripts/class_balance_probe.py` emitting predicted OOF F1 per class under current vs proposed class_weight settings for a given joblib — lets me preview rebalancing impact without a full eval cycle.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## per-domain-splice_recall-separated-from
- **status:** pending
- **first_seen:** d5cc597
- **last_seen:** d5cc597
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) Per-domain splice_recall separated from splice_f1 in CURRENT STATE. Recall loss vs precision loss have different remedies (aug vs threshold), and combined splice_f1 hides which.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## classifier_state-block-at-top-of-prompt
- **status:** pending
- **first_seen:** c0d438e
- **last_seen:** c0d438e
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) A CLASSIFIER_STATE block at the top of the prompt showing the on-disk joblib's class_counts / OOF_f1_weighted / OOF_f1_per_class / features_py_sha. Directly answers "what classifier will my code run against" and "what aug/sampling is baked in" — removes a major information gap.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## supervisor_agentpy---diff-classifier
- **status:** pending
- **first_seen:** c0d438e
- **last_seen:** 423014c
- **request_count:** 3
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `supervisor_agent.py --diff-classifier <new-sha> <old-sha>` reporting per-class OOF F1 delta between two joblibs — turns "retrain side-effect on splice recall" into a measurable quantity pre-commit.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## classifier_state-block-at-top-of-prompt
- **status:** pending
- **first_seen:** eabd6b2
- **last_seen:** eabd6b2
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLASSIFIER_STATE block at top of prompt showing on-disk joblib's class_counts, OOF_f1_weighted, OOF_f1_per_class, features_py_sha, training_manifest row count — removes the "which classifier will my code actually run against" ambiguity every training-data hypothesis currently carries.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptstraining_aug_visualizepy--
- **status:** pending
- **first_seen:** 423014c
- **last_seen:** 423014c
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) `scripts/training_aug_visualize.py --mechanism {pitch_shift,time_stretch,eq_tilt} --feature spec_rolloff_delta` plotting augmented vs original feature distributions per domain on a representative sample — lets me verify the mutation lands in the intended feature-space region BEFORE committing.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsseed_variance_probepy---n-seeds
- **status:** pending
- **first_seen:** d5484a4
- **last_seen:** d5484a4
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) `scripts/seed_variance_probe.py --n-seeds 5 --base-seed 42` — runs 5 quick HistGBM fits with different seeds on the current training data and reports per-seed OOF F1 mean/std + per-class F1 std, so I can preview whether a seed-ensemble hypothesis has enough seed-variance to be worth trying.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## classifier_state-block-in-current-state
- **status:** pending
- **first_seen:** d5484a4
- **last_seen:** d5484a4
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) CLASSIFIER_STATE block in CURRENT STATE showing on-disk joblib's class_counts, OOF F1, features_py_sha, AND model_class (HistGBM vs VotingClassifier vs GBM). Removes the classifier-identity ambiguity.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## per-domain-eval-runtime-printout-so-i
- **status:** pending
- **first_seen:** d5484a4
- **last_seen:** d5484a4
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) A per-domain EVAL runtime printout so I can plan 3x-inference hypotheses relative to the 300s budget.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-block-at-top-of
- **status:** pending
- **first_seen:** 273b8f5
- **last_seen:** 273b8f5
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) `CLEAN_FP_POSITIONS` block at top of CURRENT STATE with per-position DSP z-scores — would turn "singing clean_fp=2 / korean clean_fp=4" into actionable (file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z) tuples. Most detector-side hypotheses today are guessing which scalar signal will bite
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## post_filters_tried-block-one-line
- **status:** pending
- **first_seen:** 273b8f5
- **last_seen:** 273b8f5
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) `POST_FILTERS_TRIED` block — one-line summaries (mechanism, threshold, scope) of every post-filter hypothesis ever tried, so I can verify orthogonality pre-commit instead of chasing keyword matches in RECENT FAILED HYPOTHESES.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsdsp_signal_probepy---emit
- **status:** pending
- **first_seen:** 273b8f5
- **last_seen:** 273b8f5
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `scripts/dsp_signal_probe.py --emit-threshold 0.982 --dsp-channel {phase,t2,cpe}` that reports the distribution of the DSP z-scores on the subset of dense-scan positions where p_splice > 0.982, per domain — would let me preview the bite rate of a DSP-floor hypothesis BEFORE committing (how many
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## p_splice_cdf_tail-block-in-current
- **status:** pending
- **first_seen:** 7255ec6
- **last_seen:** 7255ec6
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) A `P_SPLICE_CDF_TAIL` block in CURRENT STATE showing, per domain, the count of p_splice values in each bin [0.980, 0.985, 0.990, 0.995, 0.999, 1.0] for both clean files (FP risk) and spliced files (TP recall) — lets me pick GBM_THRESHOLD deltas surgically from the actual CDF tail rather than fro
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## dsp_dropped_per_file-summary-block
- **status:** pending
- **first_seen:** 7255ec6
- **last_seen:** 7255ec6
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) A `DSP_DROPPED_PER_FILE` summary block showing per-domain how many emits the DSP floor bites per iteration — lets me see whether the filter is under-used (room to lower GBM_THRESHOLD) or saturated (tightening DSP_MIN is higher-leverage).
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-auto-emitted-by
- **status:** pending
- **first_seen:** 7255ec6
- **last_seen:** 7255ec6
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `CLEAN_FP_POSITIONS` JSON auto-emitted by evaluate.py with each position's full feature vector (at minimum: p_splice, p_hard, p_cross, dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_pairwise_proximity) — turns "korean clean_fp=4" from an opaque scalar into calibrated per-position diagnostics, the single
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** d256901
- **last_seen:** d256901
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) `CLEAN_FP_POSITIONS` JSON block in CURRENT STATE with per-position (domain, file, t_sec, label_id, p_hard, p_cross, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z) from the last kept classifier's eval — the single biggest blocker across every detector-side hypothesis (I keep guessing which channel w
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## dsp_drop_classification-diagnostic
- **status:** pending
- **first_seen:** d256901
- **last_seen:** d256901
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) A `DSP_DROP_CLASSIFICATION` diagnostic emitted per file: emits dropped by each gate component (MAX floor, class-specific phase_z, class-specific t2_z), so a keep/discard post-mortem can distinguish "filter under-fired" from "filter over-fired".
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsdsp_signal_probepy---by-class
- **status:** pending
- **first_seen:** d256901
- **last_seen:** d256901
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `scripts/dsp_signal_probe.py --by-class {hard_cut,crossfade} --channel {phase_z,t2_z,cpe_z} --threshold X` that simulates per-class channel-specific drops on a prior classifier's emit trace, so I can preview bite rate BEFORE committing.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** 865d92f
- **last_seen:** 865d92f
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) `CLEAN_FP_POSITIONS` JSON block in CURRENT STATE with per-position DSP z-scores from the last kept classifier's eval — the single biggest blocker across detector- side hypotheses. Most reasoning today is theory-driven instead of data-driven.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## dsp_gate_drop_rates-block-per-domain
- **status:** pending
- **first_seen:** 865d92f
- **last_seen:** 29c06cf
- **request_count:** 2
- **category:** observability
- **risk:** medium
- **excerpt:** (2) `DSP_GATE_DROP_RATES` block per domain: {file_count, gbm_emits, max_gate_dropped, sum_gate_dropped, survived} so I can preview SUM-threshold bite ranges from a prior eval's actual values.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsdsp_signal_probepy---gate-type
- **status:** pending
- **first_seen:** 865d92f
- **last_seen:** 865d92f
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `scripts/dsp_signal_probe.py --gate-type {max,sum,both} --threshold X` that simulates a candidate gate against a prior classifier's emit trace and reports per-domain TP-loss vs FP-drop counts — turns "what threshold should I pick" from a guess into a measurable pre-commit signal. [auto] (no SHAP
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** 29c06cf
- **last_seen:** 55b21e6
- **request_count:** 3
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE: per-position (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, sum, max) from the last kept classifier's eval — removes the biggest blocker across detector-side hypotheses. Every DSP-filter hypothesis today is calibrating
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsdsp_sum_sweeppy---thresholds
- **status:** pending
- **first_seen:** 29c06cf
- **last_seen:** 29c06cf
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `scripts/dsp_sum_sweep.py --thresholds 4.5,4.75,5.0,5.25,5.5` that replays the last keep's emit trace through each proposed SUM threshold and reports per-domain (TPs_lost, FPs_dropped), turning "4.5 vs 5.0" from a guess into a numeric decision. [auto] (no SHAP data for either 865d92f or 29c06cf)
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## lost_tp_positions-similar-json-block
- **status:** pending
- **first_seen:** eb8984e
- **last_seen:** eb8984e
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) LOST_TP_POSITIONS similar JSON block for TPs dropped between the last TWO keeps — lets a post-mortem pinpoint which class/domain the tightening hurt, rather than inferring from combined deltas.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsdsp_sum_class_probepy---hard-min
- **status:** pending
- **first_seen:** eb8984e
- **last_seen:** eb8984e
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `scripts/dsp_sum_class_probe.py --hard-min 5.0 --crossfade-min 4.5` that replays the last keep's emit trace through class-routed thresholds and reports per-domain (TPs_lost, FPs_dropped) — turns "class-routed SUM 4.5/5.0 vs HPR-gated SUM 4.5/5.0" from a guess into a numeric decision before commi
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## file_hpr_distribution-block-per-eval
- **status:** pending
- **first_seen:** 55b21e6
- **last_seen:** 55b21e6
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) FILE_HPR_DISTRIBUTION block: per-eval-domain min/median/max HPR, so HPR-gated hypotheses can pick thresholds from actual data.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsdsp_gate_sweeppy---axis
- **status:** pending
- **first_seen:** 55b21e6
- **last_seen:** 55b21e6
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `scripts/dsp_gate_sweep.py --axis hpr_gated_sum --thresholds "5.0/0.85/4.5,5.0/0.90/4.5"` that replays the last keep's emit trace through (sum_high, hpr_cutoff, sum_low) triples and reports per- domain (TPs_lost, FPs_dropped) — turns guess-thresholds into pre-commit numeric decisions.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-auto-emitted-by
- **status:** pending
- **first_seen:** 6b28235
- **last_seen:** 6b28235
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON auto-emitted by splice/evaluate.py: per-position (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_pairwise_proximity, file_hpr, sum, max) for every clean FP on the last kept classifier. Flip detector-side hypotheses from theory-driven to dat
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## lost_tp_positions-block-per-position
- **status:** pending
- **first_seen:** 6b28235
- **last_seen:** 6b28235
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) LOST_TP_POSITIONS block: per-position diagnostic for splice TPs that existed at last-keep but are missing at second-to-last keep. Would immediately confirm whether 29c06cf-lost singing TPs are hard_cut with low CPE (testable with this hypothesis) vs crossfade with some other profile.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsdsp_gate_sweeppy---axes
- **status:** pending
- **first_seen:** 6b28235
- **last_seen:** 6b28235
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `scripts/dsp_gate_sweep.py --axes "cpe:0.5,1.0,1.5,2.0 | sum:4.5,5.0,5.5 | max:1.5,2.0,2.5"` that replays a prior keep's emit trace through candidate gate combinations and reports per-domain (TPs_lost, FPs_dropped). Turns "pick a threshold" from a guess into a numeric decision.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-auto-emitted-by
- **status:** pending
- **first_seen:** 3e4e996
- **last_seen:** 3e4e996
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON auto-emitted by splice/evaluate.py including pairwise_proximity alongside phase/T²/CPE — turns DSP-gate hypotheses from theory-driven to data-driven. Single biggest blocker across 10+ detector hypotheses.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsdsp_gate_sweeppy---axes
- **status:** pending
- **first_seen:** 3e4e996
- **last_seen:** 3b365fe
- **request_count:** 2
- **category:** observability
- **risk:** medium
- **excerpt:** (2) `scripts/dsp_gate_sweep.py --axes "pw_confirm:1.0,1.5,2.0,2.5 | sum_confirmed:4.0,4.5,5.0"` replaying last keep's emit trace, reporting per-domain (TPs_recovered, FPs_re_admitted).
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## per-domain-breakdown-of
- **status:** pending
- **first_seen:** 3e4e996
- **last_seen:** 3b365fe
- **request_count:** 2
- **category:** observability
- **risk:** medium
- **excerpt:** (3) Per-domain breakdown of pairwise_proximity values on emits passing MAX>=2.0 but failing SUM>=5.0 — confirms biteable population before commit.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-auto
- **status:** pending
- **first_seen:** 3b365fe
- **last_seen:** 3b365fe
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block auto-emitted by splice/evaluate.py with per-position (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_pairwise_proximity, sum, max, effective_sum) from the last keep's eval — the single biggest blocker across 12+ detector-side hypothes
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsfeature_oof_previewpy---add
- **status:** pending
- **first_seen:** 634cdd2
- **last_seen:** 49bd0b1
- **request_count:** 3
- **category:** observability
- **risk:** medium
- **excerpt:** (1) `scripts/feature_oof_preview.py --add chroma_cosine_dist` that retrains once and reports per-domain OOF F1 delta vs current classifier, turning "is chroma worth adding" from a bet into a pre-commit numeric.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** 634cdd2
- **last_seen:** 634cdd2
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) `CLEAN_FP_POSITIONS` JSON block in CURRENT STATE — SUPER-persistent blocker; per position (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, top-5 most-influential features by absolute SHAP). Every feature-add hypothesis is guessing which scalar discriminates the failure
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsfeature_class_separabilitypy--
- **status:** pending
- **first_seen:** 634cdd2
- **last_seen:** 634cdd2
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `scripts/feature_class_separability.py --feature FEATURE_NAME` that computes Welch's t statistic between not_splice and (hard_cut+crossfade) training rows for a candidate feature, so I can rank candidate features by training-time separability before committing to a retrain cycle. --- ## [auto-di
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** 9fe41d1
- **last_seen:** 9fe41d1
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE: per-position (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_pairwise_proximity, chroma_pre, chroma_far, chroma_local_dist, chroma_far_dist, file_hpr) for every clean FP on the last keep. THE persistent blocker — eve
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptschroma_probepy---keep-sha-sha--
- **status:** pending
- **first_seen:** 9fe41d1
- **last_seen:** 9fe41d1
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) `scripts/chroma_probe.py --keep-sha <sha> --thresholds "0.05,0.10,0.15"` that replays the last keep's emit trace through a candidate chroma POST-FILTER and reports per-domain (TPs_lost, FPs_dropped). Turns "0.05 vs 0.10 vs 0.15" from a guess into a numeric pre-commit decision.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## chunk-edge-emit-count-per-file-how
- **status:** pending
- **first_seen:** 9fe41d1
- **last_seen:** 9fe41d1
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) Chunk-edge emit count per file: how many emits land in [chunk_dur - 8, chunk_dur] where persistence_far gates can't apply.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-auto
- **status:** pending
- **first_seen:** 68004ca
- **last_seen:** 68004ca
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block auto-emitted by `splice/evaluate.py` — persistent blocker across 12+ detector hypotheses. Per-FP (domain, file, t_sec, label_id, p_splice, TOP-5 feature values with SHAP > 100 on that domain) would flip every subsequent hypothesis from theory-driven to data-driven.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## detector_emit_trace-jsonl-alongside
- **status:** pending
- **first_seen:** 68004ca
- **last_seen:** 68004ca
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `DETECTOR_EMIT_TRACE` JSONL alongside results.tsv with (domain, file, t_sec, p_splice, pre_dsp_max, pre_dsp_sum, survives_gates) — would let post-mortems distinguish "filter dropped no emits" from "filter dropped emits but all were true splices". --- ## [auto-diagnosis] diagnose: no-traceback ts
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** 49bd0b1
- **last_seen:** 49bd0b1
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE with per-FP (domain, file, t_sec, label_id, p_splice, top-5 features by |SHAP|, voiced-frame counts in pre/post windows). Single biggest blocker across 13+ hypotheses — flips threshold/mask/feature design from theory-driven to data-driven.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## detector_bundle_sha-emit_count-line-in
- **status:** pending
- **first_seen:** 49bd0b1
- **last_seen:** 49bd0b1
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) DETECTOR_BUNDLE_SHA + EMIT_COUNT line in RESULTS_TSV / wrapper log — disambiguates the "exact-triple reproduction" mystery (is the gate not firing, or is the wrapper byte-reusing a stale joblib?). [auto] (no SHAP data for either 29c06cf or 49bd0b1)
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block
- **status:** pending
- **first_seen:** 32cac36
- **last_seen:** 67633f2
- **request_count:** 5
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block — persistent blocker for 14+ hypotheses; per-FP (domain, file, t_sec, label_id, p_splice, vp_pre_count, vp_post_count, dsp_phase_z, dsp_t2_z, dsp_cpe_z, top-5 |SHAP| features with values) would flip mask design from theory to data.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsfeature_oof_previewpy---add
- **status:** pending
- **first_seen:** 32cac36
- **last_seen:** ea1636c
- **request_count:** 13
- **category:** observability
- **risk:** medium
- **excerpt:** (2) `scripts/feature_oof_preview.py --add <feature_fn>` that trains once and reports per-domain OOF-F1 delta vs current — turns "is this feature worth a 3 min retrain" into a numeric.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## per-keep-shap-rollup-populated
- **status:** pending
- **first_seen:** 32cac36
- **last_seen:** 32cac36
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) Per-keep SHAP rollup populated immediately on keep commit (the "no keeps yet — rollup empty" marker means I have no per-domain SHAP for 49bd0b1, the freshest keep and the one whose regression I'm trying to understand). [auto] (no SHAP data for either 49bd0b1 or 32cac36)
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## per-keep-shap-rollup-populated
- **status:** pending
- **first_seen:** ff65865
- **last_seen:** ff65865
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) Per-keep SHAP rollup populated IMMEDIATELY on keep commit. The "no keeps yet — rollup empty" marker means I'm forming hypotheses with zero per-domain SHAP data on the classifier I'm trying to improve — the biggest information gap in the entire autoresearch loop.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## per-keep-shap-rollup-populated
- **status:** pending
- **first_seen:** 308aa5a
- **last_seen:** 308aa5a
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) Per-keep SHAP rollup populated IMMEDIATELY on keep commit — current "no keeps yet — rollup empty" marker means I'm forming hypotheses with zero per-domain SHAP data on the classifier I'm trying to improve, the biggest information gap in the loop.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## lost_tp_positions-between-consecutive
- **status:** pending
- **first_seen:** df6fc0a
- **last_seen:** df6fc0a
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) LOST_TP_POSITIONS between consecutive keeps — per-file list of real-splice positions detected at commit N-1 but lost at commit N. Would directly confirm/deny the "voiced_mfcc suppresses same-singer singing TPs" hypothesis driving THIS iteration.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** 1eda8e3
- **last_seen:** 0c3bf76
- **request_count:** 13
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — persistent blocker for 17+ consecutive iterations; per-FP (domain, file, t_sec, label_id, p_splice, vp_pre_frac, vp_post_frac, voiced_mfcc_dist, unvoiced_mfcc_dist, top-5 |SHAP| features with values). Every mask-based feature hypothesis has been c
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## discard-revert-sync-auditor-wrappers
- **status:** pending
- **first_seen:** 1eda8e3
- **last_seen:** aa4f141
- **request_count:** 6
- **category:** observability
- **risk:** medium
- **excerpt:** (3) DISCARD-REVERT SYNC auditor: the wrapper's discard path should diff features.py vs the baseline-sha features.py and ABORT / restore if they differ, so a discarded hypothesis's code can never silently persist into the next iteration's classifier. Currently e7ca9eb's voiced_spec_contrast block is
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## previous-iteration-outcome-header-in
- **status:** pending
- **first_seen:** 49fa2ca
- **last_seen:** 6384137
- **request_count:** 2
- **category:** observability
- **risk:** medium
- **excerpt:** (3) PREVIOUS-ITERATION-OUTCOME header in prompt ("last commit SHA: 99081f5 — DISCARD/KEEP/PENDING") resolves attribution ambiguity when keep/discard notes lag.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## discard-revert-sync-auditor-wrappers
- **status:** pending
- **first_seen:** ea1636c
- **last_seen:** ea1636c
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) DISCARD-REVERT SYNC auditor — the wrapper's discard path should diff features.py vs baseline-sha features.py and ABORT / restore-and-retry if they differ. This prevents exactly the e7ca9eb situation where a discarded hypothesis's code persists for 15+ iterations contaminating every subsequent at
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** ea1636c
- **last_seen:** ea1636c
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) CLEAN_FP_POSITIONS JSON block in CURRENT STATE (persistent blocker for 23+ iterations; per-FP feature vector + top-5 |SHAP| values) — flips the loop from theory-calibrated to data-driven feature design.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsprimary_tunable_previewpy--
- **status:** pending
- **first_seen:** 53d3ea0
- **last_seen:** 53d3ea0
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) `scripts/primary_tunable_preview.py --tunable DSP_SUM_MIN --value 5.5` — loads current classifier, runs detect on a cached 3-file eval subset, reports delta to current combined. Zero retrain cost. Turns "is this primary-tunable step worth a full 243s eval" into a 10-second numeric preview.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## current-detector-constants-block-in
- **status:** pending
- **first_seen:** 53d3ea0
- **last_seen:** 0cb551f
- **request_count:** 2
- **category:** observability
- **risk:** medium
- **excerpt:** (3) CURRENT DETECTOR CONSTANTS block in prompt — explicit list of (GBM_THRESHOLD, GBM_MIN_SEP_S, ANALYSIS_STRIDE_S, DSP_CONFIRMATION_MIN, DSP_SUM_MIN) values at HEAD, so I don't have to read detector.py every iteration to know current tunable state (and the per-tunable frontier's "current ?" rows re
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## discard-revert-sync-auditor-addresses
- **status:** pending
- **first_seen:** 0cb551f
- **last_seen:** 0cb551f
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) DISCARD-REVERT SYNC auditor — addresses the root cause of the 99081f5 hyperparam ghost AND the e7ca9eb feature ghost. Wrapper's discard path should diff each changed file vs baseline-sha file and ABORT + restore if drift remains. ~20 lines of shell guard after the discard decision. Would have pr
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## discard-revert-sync-auditor-wrapper
- **status:** pending
- **first_seen:** 67633f2
- **last_seen:** 67633f2
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) DISCARD-REVERT SYNC auditor — a wrapper guard that diffs features.py + train_classifier.py vs baseline-sha on discard and RESTORE if drift remains. Would have prevented both the e7ca9eb voiced_spec_contrast ghost (15+ iterations) and the 99081f5 capacity-bump ghost from contaminating attribution
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## current-detector-constants-current
- **status:** pending
- **first_seen:** 67633f2
- **last_seen:** 67633f2
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) CURRENT DETECTOR CONSTANTS + CURRENT CLASSIFIER HYPERPARAMS block in prompt — explicit snapshot of (GBM_THRESHOLD, GBM_MIN_SEP_S, ANALYSIS_STRIDE_S, DSP_CONFIRMATION_MIN, DSP_SUM_MIN) + (max_iter, max_depth, max_leaf_nodes, learning_rate, l2, min_samples_leaf) at HEAD, so per-tunable-frontier "c
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## discard-revert-sync-auditor-repeated
- **status:** pending
- **first_seen:** 960113c
- **last_seen:** 960113c
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) DISCARD-REVERT SYNC auditor (repeated for 5+ iterations): the wrapper's discard AND verify-fail paths must diff every changed file vs baseline-sha and RESTORE on drift. 99081f5 ghost persisted 10+ iterations after e7ca9eb's 15+-iteration precedent. ~20 lines of shell guard would prevent the next
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## current-classifier-state-block-in
- **status:** pending
- **first_seen:** 960113c
- **last_seen:** 960113c
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) CURRENT-CLASSIFIER-STATE block in CURRENT STATE: (max_depth, max_leaf_nodes, learning_rate, l2_regularization, min_samples_leaf, max_iter, joblib_sha, training_sha) so any divergence between baseline-keep-time config and HEAD config surfaces immediately.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block
- **status:** pending
- **first_seen:** 960113c
- **last_seen:** 960113c
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) CLEAN_FP_POSITIONS JSON block — the persistent 27+-iteration blocker. Every theory-calibrated hypothesis around singing clean_fp=3 could become data-driven with (domain, file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, voiced_mfcc_dist, voiced_unvoiced_mfcc_asymmetry, top-5 |SHAP|).
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## discard-revert-symmetry-semantics
- **status:** pending
- **first_seen:** b120d38
- **last_seen:** b120d38
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) **DISCARD-REVERT SYMMETRY SEMANTICS** — the discard path treats "combined matches baseline exactly" as no-improvement and reverts the change. When the change was a REVERT of a ghost commit (e.g., 960113c reverting 99081f5), this RESTORES the ghost instead of keeping the cleaner state. Proposed:
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## scriptsprimary_tunable_sweeppy--
- **status:** pending
- **first_seen:** b120d38
- **last_seen:** b120d38
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) `scripts/primary_tunable_sweep.py --tunable ANALYSIS_STRIDE_S --values 0.08,0.10,0.12,0.15,0.20` — loads current classifier, runs detect on a cached 3-file eval subset per value, reports per-value combined delta. Turns "is this stride worth a full 243s eval" into a 30-second sweep across 5 value
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## discard-revert-sync-auditor-persistent
- **status:** pending
- **first_seen:** 17d4aec
- **last_seen:** 17d4aec
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) **DISCARD-REVERT SYNC auditor** — persistent ask for 6+ iterations. Ghost state cycles (ea1636c feature ghost, 99081f5 hyperparam ghost, 960113c discarded revert re-instated the ghost) continue to contaminate attribution. ~20-line shell guard after the discard decision that diffs features.py + t
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** 17d4aec
- **last_seen:** e4a9c18
- **request_count:** 2
- **category:** observability
- **risk:** medium
- **excerpt:** (2) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — persistent blocker for 29+ iterations.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## current-classifier-hyperparams-snapshot
- **status:** pending
- **first_seen:** 17d4aec
- **last_seen:** 17d4aec
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) CURRENT CLASSIFIER HYPERPARAMS snapshot in prompt (learning_rate, max_depth, max_leaf_nodes, max_iter, l2, min_samples_leaf at HEAD) so I don't have to grep train_classifier.py every iteration to confirm ghost state.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## shap-rollup-pipeline-debug-no-keeps
- **status:** pending
- **first_seen:** db59c36
- **last_seen:** db59c36
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) SHAP rollup pipeline DEBUG — "no keeps yet" persists across 4+ recent keeps. Diagnose why the rollup writer isn't capturing them.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## current-classifier-hyperparams-oof
- **status:** pending
- **first_seen:** db59c36
- **last_seen:** db59c36
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) CURRENT CLASSIFIER HYPERPARAMS + OOF METRICS block in prompt: (max_depth, max_leaf_nodes, learning_rate, l2_regularization, min_samples_leaf, max_iter, OOF_weighted_F1, OOF_per_class_F1) at HEAD. Resolves the persistent "current ?" rows in the per-tunable frontier AND surfaces the classifier sta
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## discard-revert-symmetry-semantics-bug
- **status:** pending
- **first_seen:** 0c3bf76
- **last_seen:** 0c3bf76
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) DISCARD-REVERT SYMMETRY SEMANTICS bug fix (persistent ask) — wrapper's discard path should DETECT revert commits (commit message starts with "REVERT" AND references a prior-discarded SHA) and treat "matched baseline exactly" as SUCCESS-KEEP rather than no-improvement-discard. Would have kept 960
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## identical-combined-alert-if-n
- **status:** pending
- **first_seen:** 0c3bf76
- **last_seen:** 0c3bf76
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) IDENTICAL-COMBINED ALERT — if N consecutive iterations produce IDENTICAL combined within ±1e-5 AND IDENTICAL per-domain, emit a WARN event `wrapper.suspicious.identical_output_streak`. Last 4 iterations (17d4aec, db59c36, 0cb551f, 6384137) all hit combined=0.490700 exactly; this alert would flag
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## identical-combined-streak-diagnostic
- **status:** pending
- **first_seen:** df0ceec
- **last_seen:** df0ceec
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) **IDENTICAL-COMBINED-STREAK DIAGNOSTIC** — now persistent across 2 reflections. Emit `wrapper.suspicious.identical_output_streak` when N ≥ 3 consecutive iterations produce IDENTICAL combined ± 1e-5 AND IDENTICAL per-domain. Include predicted-emit-count comparison across the streak — if emit sets
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** df0ceec
- **last_seen:** df0ceec
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — persistent blocker for 32+ iterations. Without it every DSP / classifier / feature hypothesis around singing FPs is a theory-calibrated bet. Per-FP: (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, top-5 |SHAP|, leaf_sam
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## discard-revert-sync-auditor-99081f5
- **status:** pending
- **first_seen:** df0ceec
- **last_seen:** df0ceec
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) DISCARD-REVERT SYNC auditor — the 99081f5 ghost persists at HEAD (verified: max_depth=4, max_leaf_nodes=16 still in make_pipeline()) after 960113c's revert was itself discarded by the wrapper's symmetry-semantics bug. ~20 lines of shell guard parsing commit subjects for "REVERT" prefix + prior-S
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** 1a8b1fe
- **last_seen:** 1a8b1fe
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — persistent blocker across 33+ iterations, cited in every recent reflection. Per-FP: (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, voiced_chroma_cosine_dist, voiced_mfcc_cosine_dist, voiced_unvoiced_mfcc_asymmetry, top
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## identical-combined-streak-diagnostic
- **status:** pending
- **first_seen:** 1a8b1fe
- **last_seen:** 1a8b1fe
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) IDENTICAL-COMBINED-STREAK DIAGNOSTIC — emit `wrapper.suspicious.identical_output_streak` when N >= 3 consecutive iterations produce IDENTICAL combined ± 1e-5 AND IDENTICAL per-domain AND non-identical source diffs. Include sha256 of committed joblib / features.py / detector.py / train_classifier
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## discard-revert-symmetry-semantics-bug
- **status:** pending
- **first_seen:** 1a8b1fe
- **last_seen:** 88adb49
- **request_count:** 10
- **category:** observability
- **risk:** medium
- **excerpt:** (3) DISCARD-REVERT SYMMETRY SEMANTICS bug fix — wrapper treats "combined matches baseline exactly" as no-improvement and reverts, which RESTORED the 99081f5 capacity ghost via 960113c. Fix: if discarded commit subject starts "REVERT" and references a prior-discarded SHA, treat exact-match as KEEP no
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** 4773d1e
- **last_seen:** 0ae9231
- **request_count:** 7
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — persistent 34+-iteration blocker, cited in every recent reflection. Per-FP (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_max, dsp_sum, voiced_spec_contrast_cosine_dist, voiced_unvoiced_spec_contrast_ asymmetry, pe
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## us-505-coverage-fix-extend-sha-gate-to
- **status:** implemented
- **first_seen:** 4773d1e
- **last_seen:** d49284c
- **request_count:** 3
- **category:** observability
- **risk:** medium
- **excerpt:** (2) US-505 COVERAGE FIX — extend the sha-gate to trigger auto-retrain when train_classifier.py sha changes, not just features.py sha. Diagnostic evidence: 5 consecutive train_classifier.py-only iterations (17d4aec, db59c36, df0ceec, and two others) all produced IDENTICAL combined=0.490700 while 1a8b
- **notes:** SHIPPED. Both gate sites (L522 standalone + L1094 main loop) extended; meta.json records train_classifier_py_sha.
- **shipped_in:** 920dcdb (us505b, 2026-04-20)
## us-505-coverage-fix-extend-retrain-sha
- **status:** pending
- **first_seen:** ff88b22
- **last_seen:** 177d641
- **request_count:** 4
- **category:** observability
- **risk:** medium
- **excerpt:** (2) US-505 COVERAGE FIX — extend retrain sha gate to include train_classifier.py changes, not just features.py. Diagnostic evidence: 5+ consecutive train_classifier.py-only iterations all produced IDENTICAL combined=0.490700 while feature-count bumps produced distinct values. ~5-line shell extension
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** b7dc8bf
- **last_seen:** 1e57702
- **request_count:** 26
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — 39+-iteration blocker; per-FP (domain, file, t_sec, p_splice, dsp_*, chunk_duration_s, voiced_unvoiced_mfcc_asymmetry, voiced_unvoiced_spec_contrast_asymmetry, top-5 |SHAP|).
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## us-505-coverage-fix-extend-sha-gate-to
- **status:** pending
- **first_seen:** 78513fb
- **last_seen:** 88adb49
- **request_count:** 3
- **category:** observability
- **risk:** medium
- **excerpt:** (2) US-505 COVERAGE FIX — extend sha-gate to cover train_classifier.py changes. 5+ consecutive train_classifier.py-only iterations produced IDENTICAL 0.490700.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## discard-revert-symmetry-semantics-bug
- **status:** pending
- **first_seen:** dde4135
- **last_seen:** dde4135
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) DISCARD-REVERT SYMMETRY SEMANTICS bug — treat REVERT commits' exact-match as KEEP not DISCARD. ~15 lines of commit-message parse.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## shap-rollup-repair-rollup-empty-for
- **status:** pending
- **first_seen:** 6c6c254
- **last_seen:** 1e57702
- **request_count:** 13
- **category:** observability
- **risk:** medium
- **excerpt:** (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps is a long-standing bug; without per- feature attribution claude picks "theoretically orthogonal" not "what signal GBM actually uses."
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## us-505-coverage-assertion-920dcdb
- **status:** pending
- **first_seen:** 6c6c254
- **last_seen:** 6c6c254
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) US-505 COVERAGE ASSERTION — 920dcdb extended the sha-gate; a wrapper log line at retrain decision point ("train_classifier.py hash delta detected → auto- retraining" or "no hash delta → skip retrain") would close the feedback-loop verification gap so claude can confirm hyperparam iterations now
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## shap-rollup-repair-per-iteration-shap
- **status:** pending
- **first_seen:** 347c0ac
- **last_seen:** e234649
- **request_count:** 2
- **category:** observability
- **risk:** medium
- **excerpt:** (2) SHAP ROLLUP REPAIR — per-iteration SHAP top-K to .omc/classifier/shap_rollup.json on every keep, aggregate rolling-5 in wrapper. Without SHAP, feature selection is blind.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## us-505b-verification-trace-920dcdb
- **status:** pending
- **first_seen:** 347c0ac
- **last_seen:** 347c0ac
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) US-505b VERIFICATION TRACE — 920dcdb landed but no visible signal whether train_classifier.py-only iterations now force retrain; wrapper log line at retrain decision ("features.py sha Δ → retrain" vs "train_classifier.py sha Δ → retrain" vs "no Δ → skip") closes feedback-loop verification gap. A
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## us-505b-verification-trace-wrapper-log
- **status:** pending
- **first_seen:** c006d52
- **last_seen:** f0ccd94
- **request_count:** 6
- **category:** observability
- **risk:** medium
- **excerpt:** (3) US-505b VERIFICATION TRACE — wrapper log line at retrain decision point ("features.py sha Δ → retrain" vs "train_classifier.py sha Δ → retrain" vs "no Δ → skip") confirms 920dcdb works.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## shap-rollup-repair-rollup-empty-for
- **status:** pending
- **first_seen:** 27ddbf7
- **last_seen:** 055285f
- **request_count:** 26
- **category:** observability
- **risk:** medium
- **excerpt:** (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** 60196aa
- **last_seen:** 60196aa
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE with PEAK-WIDTH fields — per-FP (domain, file, t_sec, p_splice_at_t, p_splice_at_t_minus_stride, p_splice_at_t_plus_stride, chunk_duration_s, top-5 |SHAP|) so peak-width hypotheses become data-driven.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## us-505b-verification-trace-confirm
- **status:** pending
- **first_seen:** 60196aa
- **last_seen:** 60196aa
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) US-505b VERIFICATION TRACE — confirm hyperparam retrains fire post-920dcdb.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## retrain-actually-fired-trace-wrapper
- **status:** pending
- **first_seen:** 0cdd87e
- **last_seen:** 1e57702
- **request_count:** 10
- **category:** observability
- **risk:** medium
- **excerpt:** (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain decision: "features.py sha Δ XX→YY → retrain" vs "detector.py change only, classifier sha YY stable" with joblib-mtime sanity check. Would isolate the 0.4907 identical-streak root cause. Three unchanged highest-priority requests across 4
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## three-unchanged-highest-priority
- **status:** pending
- **first_seen:** 26a3687
- **last_seen:** 055285f
- **request_count:** 23
- **category:** observability
- **risk:** medium
- **excerpt:** Three unchanged highest-priority requests across 49+ iterations:
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-block-in
- **status:** pending
- **first_seen:** d4d35b1
- **last_seen:** 9274ace
- **request_count:** 2
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE with dense- p_splice fields — per-FP (domain, file, t_sec, p_splice, p_splice_ local_median_3s, p_splice_max_in_10s_window, dsp_phase_z, dsp_t2_z, dsp_cpe_z, chunk_duration_s, top-5 |SHAP|). Would turn every detector-side p_splice-distribution hypot
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## dense-p_splice-histogram-per-chunk-in
- **status:** pending
- **first_seen:** 3bcec76
- **last_seen:** 3bcec76
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) DENSE-P_SPLICE HISTOGRAM per chunk in diagnostic log — would isolate whether detector-side margin/rank filters are geometrically viable OR whether p_splice is saturated at 0.99+ making all margin-based filters meaningless. d4d35b1's catastrophic outcome amplifies this need.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-in-current
- **status:** pending
- **first_seen:** 4e67946
- **last_seen:** 4e67946
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP fields (including mfcc_far_post_dist and mfcc_near_post_dist to validate persistence-feature mechanism directly).
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## retrain-actually-fired-trace-with
- **status:** pending
- **first_seen:** 4e67946
- **last_seen:** 1b4fe7c
- **request_count:** 4
- **category:** observability
- **risk:** medium
- **excerpt:** (3) RETRAIN-ACTUALLY-FIRED TRACE with joblib-mtime sanity check.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## add-dsp_sum_min-dsp_confirmation_min-to
- **status:** pending
- **first_seen:** e2fc9de
- **last_seen:** 9274ace
- **request_count:** 3
- **category:** observability
- **risk:** medium
- **excerpt:** (3) ADD DSP_SUM_MIN / DSP_CONFIRMATION_MIN to the tunable frontier snapshot alongside GBM_THRESHOLD / GBM_MIN_SEP_S / ANALYSIS_STRIDE_S. Not tracking these gates means the agent has to git-log grep to know the axis state.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## clean_fp_positions-json-in-current
- **status:** pending
- **first_seen:** b5b1a0d
- **last_seen:** 055285f
- **request_count:** 14
- **category:** observability
- **risk:** medium
- **excerpt:** (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain, file, t_sec, p_splice, dsp_phase_z/t2_z/cpe_z, voicing, mfcc_post_retrospective_match, top-5 |SHAP|). Would settle every retrospective/past-history hypothesis data-driven.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## add-dsp_sum_min-dsp_confirmation_min
- **status:** pending
- **first_seen:** b5b1a0d
- **last_seen:** 8593da7
- **request_count:** 6
- **category:** observability
- **risk:** medium
- **excerpt:** (3) ADD DSP_SUM_MIN / DSP_CONFIRMATION_MIN / DSP_PHASE_MIN / DSP_SECOND_HIGHEST_MIN to the tunable frontier snapshot.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## per-domain-voicing_fraction
- **status:** pending
- **first_seen:** a23ab28
- **last_seen:** a23ab28
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) PER-DOMAIN voicing_fraction DISTRIBUTION in CURRENT STATE (median / p25 / p75 of voicing_prob_post over clean files per domain). Would directly justify gate thresholds for any music- gated feature.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## add-p_splice-neighborhood-snapshot-to
- **status:** pending
- **first_seen:** 1cded37
- **last_seen:** 1cded37
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) ADD a p_splice-NEIGHBORHOOD snapshot to CURRENT STATE: for each FP, the full p_splice vector in ±12s around the emit (or summary stats: count > thr, count > thr*0.95, max outside ±1.5s). Would enable data-driven tuning of any wide-window post-filter.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## per-domain-training-row-counts-in
- **status:** pending
- **first_seen:** e4a9c18
- **last_seen:** e4a9c18
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) PER-DOMAIN TRAINING ROW COUNTS in CURRENT STATE (read from training_manifest.json); would let sample_weight / class_weight hypotheses pick factors data-driven instead of by theory.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## augmentation-axis-frontier-snapshot-in
- **status:** pending
- **first_seen:** 7b49d40
- **last_seen:** 7b49d40
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) AUGMENTATION-AXIS FRONTIER SNAPSHOT in CURRENT STATE: SINGING_AUG_SHIFTS_SEMITONES current tuple, new SINGING_AUG_TIME_STRETCH_RATES tuple, total augmented row count per domain. Would prevent losing state on augmentation knobs the same way GBM_THRESHOLD / GBM_MIN_SEP_S / ANALYSIS_STRIDE_S have e
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## shap-rollup-repair-rollup-empty-for
- **status:** pending
- **first_seen:** ca2aa2f
- **last_seen:** ca2aa2f
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps; would let me verify whether 1eda8e3 MFCC asymmetry is load-bearing to decide whether a PERCUSSIVE cousin is redundant or additive.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## preprocessing-axis-frontier-snapshot-in
- **status:** pending
- **first_seen:** ca2aa2f
- **last_seen:** ca2aa2f
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) PREPROCESSING-AXIS FRONTIER SNAPSHOT in CURRENT STATE — track which source-separation paths have been tried (HPSS / voicing / full-mix) so pivots to new preprocessing axes are visible alongside GBM_* / augmentation frontier.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## per-domain-p_splice-quantiles
- **status:** pending
- **first_seen:** 425f6d6
- **last_seen:** 425f6d6
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) PER-DOMAIN p_splice QUANTILES (p50/p90/p95/p99 of p_splice at positive emits and clean FPs, per domain) in CURRENT STATE. Would make any threshold-tweak (global OR voicing-gated) directly data-driven.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## classifier-hyperparameter-frontier
- **status:** pending
- **first_seen:** f1e91ec
- **last_seen:** 79317c7
- **request_count:** 2
- **category:** observability
- **risk:** medium
- **excerpt:** (3) CLASSIFIER HYPERPARAMETER FRONTIER SNAPSHOT (n_estimators / max_depth / learning_rate / l2_regularization / min_samples_leaf / max_leaf_nodes — tried kept/failed per axis alongside GBM_*) so regularization-strength progression is visible like the primary-tunable frontier.
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
## classifier-hyperparameter-frontier
- **status:** pending
- **first_seen:** 055285f
- **last_seen:** 055285f
- **request_count:** 1
- **category:** observability
- **risk:** medium
- **excerpt:** (3) CLASSIFIER HYPERPARAMETER FRONTIER SNAPSHOT in CURRENT STATE (max_iter / max_depth / max_leaf_nodes / learning_rate / l2_regularization / min_samples_leaf / max_bins — tried kept/failed per axis alongside GBM_*). Would prevent losing state on regularization knobs the way GBM_THRESHOLD / GBM_MIN_
- **notes:** Auto-triaged by supervisor_agent.py --maintain. Needs human review.
