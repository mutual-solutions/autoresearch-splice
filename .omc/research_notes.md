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

## 2026-04-18T14:58:14+09:00 — 2185e35 (verify-fail, combined=0.463218)
subject: add 3 local-novelty spectral features (centroid/rolloff/bandwidth) to attack singing's chord-transition-vs-splice confusion — |delta(t)|/median(|delta(t±4/±8s)|) clipped to [0,50], orthogonal to all 20+ prior absolute-delta/hyperparam/data-composition failures, directly targets SHAP rollup observation that top singing predictors spec_rolloff_delta/centroid_delta/bandwidth_delta are identical to english/korean tops but 3-5x higher magnitude (same absolute distribution, musical periodicity fires them at non-splices); ratio formulation produces novelty>>1 at true splices (neighboring offsets within stable musical section → small reference |delta|) and novelty≈1 at song-natural chord transitions (neighboring offsets in same chord progression → similar reference |delta|); FEATURE_NAMES grows 75→78 so classifier auto-retrains (US-505), minimal blast radius (one block edit), no detector.py or train_classifier.py changes; smoke test on data/train/singing/tier1/splice_t1_001.wav at t=30 confirms novelty magnitudes 22–45 at GT splice location distinctly above neutral=1
per-domain: combined_english=0.756098 combined_korean=0.476190 combined_singing=0.276056

(a) Hypothesis: add 3 **local-novelty** features to features.py `_block_spectral`
    — `spec_centroid_novelty`, `spec_rolloff_novelty`, `spec_bandwidth_novelty`.
    Each equals |delta(t)| / (median(|delta(t+k)|) + eps) for k ∈ {-8, -4, +4, +8}s
    within chunk bounds, clipped to [0, 50]. This is a RATIO (current local
    step-change vs surrounding step-changes in the same chunk), orthogonal
    to the existing absolute-delta features. FEATURE_NAMES grows 75→78 so
    the classifier auto-retrains on the new shape (US-505).

(b) WHY. Every tunable axis is exhausted: all 6 primary tunables bracketed
    by failures on both sides, all 7 GBM hyperparam axes failed, all 4
    training-data-composition axes failed, spectral window shrink 2→1s
    failed, joint THRESHOLD+MIN_SEP diagonal failed, sample_weight singing
    2.0x verify-failed. SHAP rollup: top singing predictors spec_rolloff_delta
    (1365), spec_centroid_delta (427), spec_bandwidth_delta (409) are
    identical to english/korean top features but 3–5x higher magnitude in
    singing — the classifier sees the same feature distribution but singing
    simply has more of it (at both real splices AND song-natural chord
    transitions). The absolute-delta feature cannot separate these two
    populations: a real splice's centroid_delta looks like a chord transition's
    centroid_delta — same magnitude, different cause. A RATIO feature changes
    the geometry: at a true splice, neighboring reference offsets are within
    a stable musical section so |delta| at ±4/±8s is small, producing
    novelty >> 1; at a chord transition during an ongoing progression,
    neighboring offsets (other chord changes in the cycle) have similar
    |delta|, producing novelty ≈ 1. Genuinely untried feature-engineering
    axis targeting the exact failure mode. Minimal blast radius: +3
    features in one block, no changes to detector.py or train_classifier.py,
    classifier auto-retrains on new shape.

(c) If this fails: try MFCC-delta novelty (same ratio formulation applied
    to the existing 13 mfcc_delta features) — expands the novelty idea to
    the voice-identity subspace which should help korean too. Alternatively,
    if novelty shows promise on korean/english but flat on singing, the
    issue is that the chunk's reference offsets themselves contain
    splice-like transitions; next move would be per-chunk z-score
    standardization of the delta features (chunk-local normalization).

## 2026-04-18T15:01:51+09:00 — d91fb98 (verify-fail, combined=0.463218)
subject: add mfcc_cosine_distance feature to _block_mfcc — one scale-invariant orthogonal-to-magnitude feature targeting singing's song-natural-chord-transition FP plateau (clean_fp=6, weakest-domain combined=0.272). Existing mfcc_delta_XX entangles magnitude and direction so spec_*_delta features dominate SHAP in all 3 domains with identical rank but 3-5x higher singing magnitude firing at chord transitions the same as real splices. Cosine distance 1 - (pre_mean . post_mean)/(||pre_mean|| ||post_mean||) is scale-free and measures ONLY timbral direction change in MFCC-13 space: a singing chord transition preserves singer/room/mic identity so mean-MFCC direction stays aligned (cos_dist ~ 0) while a true splice across songs shifts timbral direction (cos_dist > 0). Orthogonal to all prior failed axes: 6 primary tunables bracketed, 7 GBM hyperparam axes failed, 4 training-data-composition axes failed, spec feature-window down failed, sample_weight singing 2.0x verify-failed, local-novelty spec ratio features verify-failed (temporal-novelty ratio is fundamentally different from direction-change). FEATURE_NAMES 75 -> 76 so classifier auto-retrains (US-505). Minimum blast radius: single new feature in existing block, no detector.py / train_classifier.py / geometry / threshold edits. Smoke test at t=10 stable mid-content cos_dist=0.0015 confirms low value for unchanging source, bounded [0,2] with no NaN risk (epsilon + clip).
per-domain: combined_english=0.756098 combined_korean=0.476190 combined_singing=0.276056

## 2026-04-18 — hypothesis: MFCC pre/post cosine distance feature

(a) HYPOTHESIS. Add ONE new feature `mfcc_cosine_distance` to
    `_block_mfcc`: cosine distance between the mean MFCC-13 vectors
    of the pre and post 2.0s windows around t.
        d = 1 - (pre_mean . post_mean) / (||pre_mean|| * ||post_mean|| + eps)
    FEATURE_NAMES grows 75 -> 76. Classifier auto-retrains (US-505).
    Zero detector.py / train_classifier.py changes.

(b) WHY. Every prior keep/fail tells the same story: magnitude-based
    spec_*_delta features dominate SHAP in ALL three domains, but in
    singing they fire on song-natural chord transitions with the SAME
    magnitude distribution as true splices (driving clean_fp=6 plateau
    at weakest combined=0.272). Existing mfcc_delta_XX = post_mean - pre_mean
    entangles magnitude and direction: a loud->quiet transition of the
    SAME source produces nonzero delta, and so does a same-loudness
    transition between DIFFERENT sources. Cosine distance is scale-
    invariant -- it measures ONLY direction change in MFCC-13 space.
    For singing specifically, a chord transition preserves singer /
    room / mic identity so the mean-MFCC direction stays aligned
    (cos_dist ~ 0), while a real splice across songs shifts timbral
    centroid direction (cos_dist > 0). This is genuinely orthogonal
    to every prior failed axis:
      * all 6 primary tunables bracketed by failures;
      * all 7 GBM hyperparam axes failed (depth both dirs, subsample,
        min_samples_leaf, max_features, learning_rate, n_estimators);
      * all 4 training-data-composition axes failed;
      * spec feature-window down (2.0->1.0) failed;
      * sample_weight singing 2.0x verify-failed (0.463);
      * local-novelty spec features (centroid/rolloff/bandwidth
        ratio of |delta| / median(neighboring |delta|)) verify-failed
        (0.463) -- that approached the chord-transition problem via
        temporal-novelty ratios; this approaches it via direction
        change orthogonal to magnitude, a strictly different signal.
    Minimum viable change: one feature in one existing block, no
    detector/geometry changes, retrain auto-triggered by
    FEATURE_NAMES length change.

(c) IF THIS FAILS. Try the same cosine-distance idea on the mel-PCA
    embedding (mel_pca_01..20 are PC projections of mel-patches around
    t -- compare pre and post patches via cosine in 20-dim PCA space).
    That captures fuller spectral timbre including non-MFCC dimensions.
    Or, if MFCC-cosine shows promise on singing but flat/negative on
    english/korean, domain-gate the feature (set to 0 unless
    voicing_prob_pre > 0.5) so it only fires on voiced content.

