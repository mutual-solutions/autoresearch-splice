## 2026-04-18T14:55:19+09:00 — 4f40e1d (verify-fail, combined=0.463218)
subject: GBM sample_weight singing 2.0x untouched fit-time loss-rebalancing axis (clf__sample_weight never passed before — current train treats all 1200 rows equally regardless of domain), all primary tunables bracketed (GBM_THRESHOLD 0.982 by failed 0.981/0.9825/0.98/0.983 single, GBM_MIN_SEP_S 3.5 by failed 3.6/3.75/4.0, ANALYSIS_STRIDE_S 0.12 by failed 0.11/0.10, ANALYSIS_STEP_S 30 by failed 25/27.5/20, ANALYSIS_WINDOW_S 60 by failed 75, ANALYSIS_EDGE_S 0.5 by failed 1.0), all 7 GBM hyperparam axes failed (max_depth both directions, subsample 0.8, min_samples_leaf 5, max_features sqrt, learning_rate 0.05, n_estimators 300), all training-data-composition axes failed (NEG_MIN_DIST 3.5, POS_OFFSETS narrowed, NEG_PER_CLEAN 8, NEG_PER_SPLICED 6), spectral feature window down failed, joint THRESHOLD+MIN_SEP failed; SHAP top features identical across all 3 domains (spec_rolloff_delta/spec_centroid_delta/spec_bandwidth_delta dominate everywhere) but singing has 3-5x higher SHAP magnitude on same features, suggesting unweighted loss lets easy domains (english 0.756, korean 0.500) dominate fit while singing 0.272 with clean_fp=6 splice_f1~0.34 is treated as background noise; upweighting singing 2.0x makes it 800/(400+400+800)=50% of loss budget vs prior 400/1200=33%, principled rebalancing toward GM-bottleneck domain; modest 2.0x (not 3.0x) minimizes risk to other 2 domains, training-time-only change leaves all detector geometry/threshold/feature-extraction code untouched so any per-domain regression isolated to classifier loss surface
per-domain: combined_english=0.756098 combined_korean=0.476190 combined_singing=0.276056

# Iteration reflection

(a) Hypothesis: pass `sample_weight` to `pipe.fit()` in train_classifier.py
    upweighting singing-domain training samples by 2.0x (english/korean stay
    at 1.0x). Implemented by deriving weights from the manifest's `dataset`
    field and routing them to the GBM step via `clf__sample_weight=...`.

(b) WHY this direction. Every other axis is exhausted: all 6 primary
    tunables bracketed by failures on both sides, all 7 GBM hyperparam
    axes failed (max_depth both directions, subsample, min_samples_leaf,
    max_features, learning_rate, n_estimators), all 4 training-data-
    composition axes failed (NEG_MIN_DIST, POS_OFFSETS, NEG_PER_CLEAN,
    NEG_PER_SPLICED), feature-window down (2.0->1.0) failed, and joint
    THRESHOLD+MIN_SEP diagonal failed. The `sample_weight` argument to
    GBM.fit() has NEVER been touched — current training treats all 1200
    rows equally regardless of domain. Top SHAP features are identical
    across all three domains (spec_rolloff_delta / spec_centroid_delta /
    spec_bandwidth_delta dominate everywhere), but singing has 3-5x higher
    SHAP magnitude on the same features, suggesting the unweighted loss
    is letting the easy domains (english 0.756, korean 0.500) dominate the
    fit while singing (0.272, weakest, clean_fp=6, splice_f1~0.34) is
    treated as background noise. Upweighting singing 2.0x makes it 2/4
    of the loss budget vs english 1/4 + korean 1/4 — a principled
    rebalancing toward the GM-bottleneck domain. Modest 2.0x (not 3.0x)
    minimizes risk of tanking the other two domains.

(c) If this fails: try a NEW feature targeting splice-vs-chord-transition
    distinguishability — e.g., a "local novelty" feature that compares the
    candidate's spec_centroid_delta to deltas computed at ±5/±10s offsets
    in the same chunk. Real splices are unique step-changes; chord
    progressions are periodic, so the relative magnitude (current delta /
    median nearby delta) should separate them where the absolute delta
    cannot. This is genuinely untried feature engineering targeting the
    exact failure mode (singing FPs from song-natural chord transitions
    firing the same spec_*_delta features as real splices).

