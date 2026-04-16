# Deep Interview Spec: ML Classifier Integration into Autoresearch Loop

## Metadata
- Rounds: 5
- Final Ambiguity Score: 17%
- Type: brownfield
- Generated: 2026-04-16
- Threshold: 20%
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.85 | 0.35 | 0.298 |
| Constraint Clarity | 0.85 | 0.25 | 0.213 |
| Success Criteria | 0.80 | 0.25 | 0.200 |
| Context Clarity | 0.80 | 0.15 | 0.120 |
| **Total Clarity** | | | **0.830** |
| **Ambiguity** | | | **17%** |

## Goal
Wire the pre-generated ML training data (1,384 patches from singing + Korean datasets) into the autoresearch evaluation loop, remove the classifier quality gate, and let the autoresearch agent co-optimize both DSP parameters (detector.py) and ML classifier parameters (ml_config.py) to maximize `combined_full`.

## Constraints
- No neural networks, no GPU — tree-based classifiers only (GradientBoosting, etc.)
- Full evaluation must complete in under 120 seconds (DSP + ML)
- `prepare.py` and `program.md` are protected — only the human modifies them
- Agent edits **detector.py** (DSP) and **ml_config.py** (ML params) — nothing else
- `ml_eval.py` is infrastructure — agent does not edit it
- Pre-generated patches (`.omc/classifier/patches_combined/`) are used as additional training data
- OOF (out-of-fold) evaluation on eval-set patches to prevent train-on-test leakage

## Non-Goals
- No new data collection or patch generation pipelines
- No neural network architectures
- No changes to the evaluation dataset (90 files: 50 clean + 20 tier1 + 20 tier2)
- Agent does not modify ml_eval.py, prepare.py, or program.md

## Acceptance Criteria
- [ ] `ml_config.py` exists with tunable classifier parameters (n_estimators, max_depth, learning_rate, PCA components, OOF threshold, etc.)
- [ ] `ml_eval.py` loads and uses pre-generated patches from `.omc/classifier/patches_combined/` as additional training data alongside on-the-fly eval patches
- [ ] `ml_eval.py` imports ML params from `ml_config.py` instead of hardcoded constants
- [ ] Classifier quality gate (CV F1 < 0.60 fallback) is removed — `combined_full` is always used when `--with-classifier` is passed
- [ ] `combined_full` = classifier-filtered F1 × clean_score is the single optimization target
- [ ] Full eval with `--with-classifier` completes within 120 seconds
- [ ] Autoresearch prompt in `run_autoresearch.sh` is updated to mention ml_config.py as editable and allow ML+DSP hypotheses
- [ ] `program.md` is updated to reflect the new ML experiment rules (human does this, but spec should note what changes are needed)
- [ ] No train-on-test leakage: pre-generated patches are training-only, eval patches use OOF
- [ ] `fp_filter` in detector.py is disabled during eval (no double-filtering) — OOF path only
- [ ] Duplicate patch constants (PATCH_HALF_S, N_MELS, etc.) unified into ml_config.py
- [ ] The autoresearch loop successfully runs at least 3 iterations with classifier output visible in the log

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| Agent only edits detector.py | ML params live in ml_eval.py — how does agent tune? | Create ml_config.py as second editable file |
| Quality gate (CV F1 > 0.60) is needed | Gate prevents bad classifiers from hurting score | Remove gate — combined_full score is self-correcting (bad classifier → lower score → discard) |
| "Any improvement" is a valid success criterion | Loop needs binary keep/discard | combined_full > previous best is the rule, same as DSP-only |
| Pre-generated patches are unused | ml_eval.py generates on-the-fly from 90 files only | Wire pre-generated patches as additional training data in infrastructure |
| Two classifier paths exist | fp_filter.py (runtime) vs ml_eval.py (OOF) could conflict | Use OOF path only during eval; disable fp_filter in detector.py during autoresearch |

## Technical Context

### Current State
- `ml_eval.py` generates patches on-the-fly from 90 eval files → ~200-300 patches → CV F1 = 0.095
- 1,384 pre-generated patches in `.omc/classifier/patches_combined/` (320 positive, 1064 negative) from singing + Korean data
- Classifier: GradientBoosting + PCA pipeline via `train_classifier.py`
- Quality gate at CV F1 < 0.60 causes fallback to DSP-only combined

### Required Changes
1. **Create `ml_config.py`** with tunable params (read by ml_eval.py, fp_filter.py, generate_patches.py)
   - Unify duplicate constants: PATCH_HALF_S, N_MELS, HOP_LENGTH, N_FFT, TARGET_FRAMES
   - Classifier params: n_estimators, max_depth, learning_rate, subsample, n_components
   - Eval params: OOF_THRESHOLD (default 0.5), N_FOLDS (default 5), DSP_FP_BOUND (15)
2. **Update `ml_eval.py`** to:
   - Load pre-generated patches from `.omc/classifier/patches_combined/` as additional training data
   - Import params from ml_config.py
   - Remove quality gate — always compute combined_full
   - Keep OOF for eval-set patches only (pre-generated patches are training-only)
3. **Disable `fp_filter` in detector.py during eval** — prevent double-filtering
   - OOF path (ml_eval.py) is the single classifier path during autoresearch
4. **Update `run_autoresearch.sh`** prompt to:
   - Allow ML+DSP hypotheses (not just "one specific DSP idea")
   - Mention ml_config.py as editable
   - Ensure `--with-classifier` is actually used
5. **Update `program.md`** (human task): document ml_config.py as editable, new experiment rules
6. **Update `CLAUDE.md`** (human task): add ml_config.py to editable files list

### Key Files
- `ml_eval.py` — classifier training/inference (infrastructure, modify now)
- `ml_config.py` — NEW: tunable ML params (agent-editable)
- `detector.py` — DSP detector (agent-editable, unchanged)
- `prepare.py` — orchestrates eval pipeline (protected)
- `run_autoresearch.sh` — loop wrapper (update prompt)
- `.omc/classifier/patches_combined/` — pre-generated training data
- `.omc/classifier/train_classifier.py` — pipeline definition (reference)

## Interview Transcript
<details>
<summary>Full Q&A (5 rounds)</summary>

### Round 1
**Q:** The ML classifier is designed to filter DSP false positives. What's your primary goal?
**A:** Full ML+DSP loop — let autoresearch experiment with both DSP parameters AND classifier tuning.
**Ambiguity:** 56%

### Round 2
**Q:** What would make the ML integration a success — combined_full beating 0.7237, or intermediate milestones?
**A:** Any improvement — even a working classifier that reduces FPs is progress worth keeping.
**Ambiguity:** 45%

### Round 3
**Q:** What should the agent be allowed to modify for ML experiments?
**A:** Infrastructure wire-up — we handle the data pipeline, agent only tunes parameters.
**Ambiguity:** 31%

### Round 4 (Contrarian)
**Q:** "Any improvement" is too vague for keep/discard. Should we lower the quality gate?
**A:** Remove gate entirely — let combined_full be the judge.
**Ambiguity:** 23%

### Round 5
**Q:** Agent only edits detector.py but classifier params live in ml_eval.py — how to resolve?
**A:** Separate ml_config.py — agent can edit both detector.py and ml_config.py.
**Ambiguity:** 17%

</details>
