# Deep Interview Spec: Autoresearch ML-in-Loop Integration

## Metadata
- Rounds: 5
- Final Ambiguity Score: 18%
- Type: brownfield
- Generated: 2026-04-16
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.92 | 35% | 0.32 |
| Constraint Clarity | 0.82 | 25% | 0.21 |
| Success Criteria | 0.65 | 25% | 0.16 |
| Context Clarity | 0.88 | 15% | 0.13 |
| **Total Clarity** | | | **0.82** |
| **Ambiguity** | | | **18%** |

## Goal
Restructure the autoresearch loop so that every iteration runs the full DSP→patch generation→classifier train→classifier inference pipeline, and optimizes `combined_full` (F1 × clean_score computed on classifier-filtered detections) instead of DSP-only `combined`. This allows DSP thresholds to be loosened for higher recall while the classifier handles FP suppression.

## How It Works

### Current Flow (DSP-only)
```
claude -p → edit detector.py → uv run prepare.py → combined → keep/discard
~30s per iteration
```

### New Flow (DSP + ML in-loop)
```
claude -p → edit detector.py → uv run prepare.py --with-classifier → 
  internally: detect_splices() → generate_patches → train_classifier → 
  filter_detections → compute combined_full → keep/discard
~60s per iteration
```

### Key Design Decisions

1. **Full retrain every iteration**: generate_patches + train_classifier + inference runs every iteration, not just on keep. ~30s additional per iteration is acceptable.

2. **Metric: classifier-filtered F1 × clean_score**: Same formula as current `combined`, but computed on detections AFTER classifier filtering. DSP that produces more FP but higher recall will score better if classifier successfully filters the FP.

3. **DSP FP upper bound: clean_fp ≤ 15**: Before classifier filtering, DSP alone may produce up to 15 FP on 50 clean files (30%). This prevents autoresearch from reducing all thresholds to zero. If DSP clean_fp > 15, the iteration is discarded regardless of combined_full.

4. **Threshold freedom**: Autoresearch agent can freely adjust DSP threshold values (GPD_ALPHA, CPE_CONFIRM_SIGMA, T2_ALPHA, etc.) as long as DSP clean_fp stays within bound. This allows the agent to find the optimal operating point for the DSP+ML pipeline.

5. **Version pipeline separation**: Autoresearch optimizes DSP+ML combined. Web demo uses DSP-only snapshots from versions.json `production` field. The two tracks are independent.

6. **prepare.py extension**: Add `--with-classifier` flag to prepare.py. When set:
   - Run detect_splices() as usual (DSP-only)
   - Report DSP-only metrics (combined_dsp, dsp_clean_fp)
   - If dsp_clean_fp > 15: print BOUND_EXCEEDED, exit with special code
   - Generate patches from current detections
   - Train classifier on patches (5-fold GroupKFold CV)
   - Filter detections through trained classifier
   - Report classifier-filtered metrics (combined_full)
   - Keep prepare.py as protected file (autoresearch agent cannot modify)

7. **First iteration bootstrap**: If no prior classifier exists, first iteration trains from scratch on whatever FP the current detector produces. Even with few FP (current: 1), the classifier will train — it just won't be very useful yet. As thresholds loosen across iterations, FP count grows, and classifier quality improves naturally.

## Changes Required

### prepare.py
- Add `--with-classifier` CLI flag
- When flag is set:
  - After DSP eval, check dsp_clean_fp <= 15 (bound)
  - Call generate_patches logic inline (or import)
  - Call train_classifier logic inline (or import)
  - Call filter_detections on all detections
  - Recompute F1 and clean_score on filtered results
  - Print both `combined_dsp` and `combined_full`
- Without flag: behaves exactly as before (backward compatible)
- Remains protected file (in CLAUDE.md / verify_agent.py)

### program.md
- Change optimization target from `combined` to `combined_full`
- Remove "NO neural networks" constraint — tree-based ML (GradientBoosting) is allowed
- Add DSP FP bound rule: "DSP clean_fp must stay ≤ 15. If exceeded, discard."
- Update eval command: `uv run prepare.py --with-classifier`
- Allow threshold adjustment as a valid hypothesis type

### run_autoresearch.sh
- Change eval command from `uv run prepare.py` to `uv run prepare.py --with-classifier`
- Remove background classifier training (now in-loop)
- Parse `combined_full` instead of `combined` for keep/discard
- Keep version pipeline (tag + snapshot) on keep
- Adjust time expectations (60s per iteration)

### test_regression.py
- Add DSP+ML regression thresholds alongside DSP-only thresholds
- DSP-only thresholds can be relaxed (lower precision OK if clean_fp ≤ 15)
- Add combined_full >= X threshold (to be determined after first runs)

### verify_agent.py
- Update to verify combined_full instead of combined
- Add dsp_clean_fp bound check (≤ 15)

### .omc/classifier/ (existing files)
- generate_patches.py: No structural changes, but will be called from prepare.py
- train_classifier.py: No structural changes, but will be called from prepare.py
- fp_filter.py: No structural changes, used for inference step

## Constraints
- prepare.py remains a protected file — autoresearch agent cannot modify it
- detector.py is the only file autoresearch agent edits
- Iteration time budget: ≤ 120s (relaxed from 60s to accommodate ML)
- Tree-based ML only (GradientBoosting). No neural networks.
- DSP clean_fp ≤ 15 hard bound
- 5-fold file-level GroupKFold CV (no data leakage)

## Non-Goals
- Web demo changes (stays on DSP-only production snapshot)
- Korean speech data integration (separate effort)
- New detection algorithms (autoresearch handles that)
- Classifier architecture changes (GradientBoosting stays)

## Acceptance Criteria
- [ ] `uv run prepare.py --with-classifier` runs end-to-end and prints both combined_dsp and combined_full
- [ ] `uv run prepare.py` (without flag) produces identical output to current version
- [ ] DSP FP bound (≤15) is enforced: exceeding it causes discard
- [ ] program.md updated with new metric, eval command, and FP bound rule
- [ ] run_autoresearch.sh calls `prepare.py --with-classifier` and parses combined_full
- [ ] Background classifier training removed from run_autoresearch.sh keep handler
- [ ] test_regression.py includes combined_full threshold test
- [ ] verify_agent.py checks combined_full and dsp_clean_fp bound
- [ ] First iteration bootstraps correctly with minimal FP data
- [ ] Version pipeline still works (tag + snapshot on keep)
- [ ] Full iteration completes in ≤ 120s

## Assumptions Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| Classifier should run in background | "30초면 길지 않아" | Full retrain in-loop, ~60s total OK |
| DSP must minimize FP | "FP 데이터가 없으면 classifier가 배울 게 없다" | DSP FP ≤15 bound, classifier handles filtering |
| Threshold 자유 조절 → lazy path | "bound 시키면 되잖아" | DSP FP bound prevents infinite loosening |
| prepare.py immutable | "prepare.py 확장하되 protected" | --with-classifier flag, agent cannot modify |
| Metric must change | "종합 성능 최적화" | combined_full = classifier-filtered F1 × clean_score |

## Interview Transcript
<details>
<summary>Full Q&A (5 rounds)</summary>

### Round 1
**Q:** 루프 안에 넣고 싶은 범위가 어디까지인가요? Retrain은 매 iteration? 매 keep? N번마다?
**A:** 매 iteration마다 full retrain

### Round 2
**Q:** 새 metric(combined_full)을 어떻게 정의할까요?
**A:** Classifier 후 F1 × clean_score 그대로

### Round 3
**Q:** DSP threshold 관리를 어떻게 할까요?
**A:** Version pipeline으로 분리 (autoresearch=loose+ML, web demo=DSP-only snapshot)

### Round 4 (Contrarian)
**Q:** Classifier filtering을 어디에서 적용할까요? + Threshold 조절 정책?
**A:** Threshold 자유조절을 허용하되 detector 단독 FP를 bound 시키자

### Round 5
**Q:** DSP FP bound 수치와 eval 위치?
**A:** FP≤15, prepare.py 확장

</details>
</summary>
</content>
