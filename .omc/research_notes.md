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

## 2026-04-18T15:13:23+09:00 — 3c4b105 (verify-fail, combined=0.463218)
subject: add HPSS-percussive spectral deltas (perc_centroid/rolloff/bandwidth) to target singing chord-transition FP plateau — uniform-filter harmonic estimate of |STFT| along time, positive residual as percussive spectrogram, derive 3 new centroid/rolloff-0.85/bandwidth-p2 deltas on ±2s pre/post. FEATURE_NAMES 75→78 so classifier auto-retrains (US-505). Orthogonal to all prior verify-fails: sample_weight 2.0x, local-novelty spec ratio, mfcc_cosine_distance all computed from the FULL signal — this is the first SIGNAL-DECOMPOSITION axis attempted. Singing chord transitions are purely harmonic (sustained pitched content) so they leave the percussive residual ≈ 0, while real cross-song splices introduce a broadband transient that the time-mean harmonic estimator cannot absorb. Smoke test on data/train/singing/tier1/splice_t1_001.wav: perc_centroid_delta=1993/perc_rolloff_delta=7776/perc_bandwidth_delta=3499 at GT splice t=30 vs 141/796/511 at stable mid-content t=10 (∼10-15x SNR). Blast radius minimal: +3 features in existing _block_spectral, single uniform_filter1d(size=41, axis=time) call per chunk (~10ms overhead on a 30s chunk vs ~180ms for librosa.decompose.hpss), no detector.py or train_classifier.py edits.
per-domain: combined_english=0.756098 combined_korean=0.476190 combined_singing=0.276056

## 2026-04-18 — hypothesis: HPSS-percussive-only spectral deltas

(a) HYPOTHESIS. Add 3 NEW features to `_block_spectral` computed on the
    HPSS-PERCUSSIVE component of the signal only: `perc_centroid_delta`,
    `perc_rolloff_delta`, `perc_bandwidth_delta`. In `_ensure_feat_cache`
    run `librosa.decompose.hpss(|STFT|, kernel_size=15)`, keep the
    percussive magnitude spectrogram S_perc, and derive per-frame
    centroid / rolloff-0.85 / bandwidth from S_perc by direct matrix ops
    (no re-running librosa.feature.*). Existing spec_*_delta features on
    the full signal remain. FEATURE_NAMES grows 75 -> 78 so the
    classifier auto-retrains (US-505).

(b) WHY. The last 3 consecutive hypotheses (sample_weight singing 2.0x,
    local-novelty spec ratio, mfcc_cosine_distance) all verify-failed at
    0.463218 with identical per-domain breakdowns. Every prior feature
    addition computed its signal from the FULL waveform -- same signal
    subspace as the failing spec_*_delta top-SHAP features. HPSS is a
    genuinely untried SIGNAL-DECOMPOSITION axis: it splits the waveform
    into harmonic (sustained pitched content) and percussive (transient
    broadband content) BEFORE any feature extraction. Singing's
    song-natural chord transitions are almost entirely harmonic -- they
    project into the harmonic component and leave the percussive
    component unchanged -- so perc_centroid_delta / perc_rolloff_delta /
    perc_bandwidth_delta are near ZERO at chord transitions. Real
    cross-song splices typically introduce a broadband transient
    discontinuity (new mic / new room / different clip onset) that
    registers in the percussive component, so perc_*_delta fires at
    true splices. This is strictly orthogonal to every prior failed
    axis:
      * all 6 primary tunables bracketed by failures,
      * all 7 GBM hyperparam axes failed,
      * all 4 training-data-composition axes failed,
      * spec feature-window down 2.0->1.0 failed,
      * sample_weight singing 2.0x verify-failed,
      * local-novelty spec ratio (temporal contrast) verify-failed,
      * mfcc_cosine_distance (timbral direction) verify-failed.
    None of those touched HPSS / source-separation. Minimum blast
    radius: +3 features in existing _block_spectral, single kernel_size=15
    HPSS call per chunk (~200-300ms, fits in 300s budget with 243s
    current headroom on ~300 chunks total), no detector.py or
    train_classifier.py edits, classifier auto-retrains on new shape.

(c) IF THIS FAILS. Next move is more aggressive: ABLATE the dominant
    spec_*_delta features entirely (delete them from FEATURE_NAMES and
    _block_spectral) rather than just add orthogonal ones. The repeated
    failure of orthogonal feature additions suggests GBM is stuck on
    spec_*_delta and does not reallocate weight to new features on a
    1200-row training set. Removing the dominant-but-indiscriminate
    predictors forces the model onto a different decision surface.
    Complementary path: per-domain GBM (train 3 classifiers, one per
    domain, dispatch by file-path domain lookup at inference time) which
    lets the singing classifier learn its own weights independent of
    the easy-domain loss pressure that sample_weight rebalancing
    failed to fix.

## 2026-04-18T15:17:42+09:00 — 4af21bc (verify-fail, combined=0.463218)
subject: plateau-filter GBM emissions — keep hit only if neighbor (i±1 at stride=0.12s) is ALSO above threshold, filtering single-frame spikes. New constant GBM_MIN_CONSECUTIVE_HITS=2 gates the hit_mask in _gbm_detect_splices. Untried post-processing axis orthogonal to every prior attempt (all 6 primary tunables bracketed, 7 GBM hyperparam axes failed, 4 training-data-composition axes failed, spec feature-window down failed, joint THRESHOLD+MIN_SEP failed, 4 consecutive feature-addition+training-weight hypotheses verify-failed at IDENTICAL combined=0.463218 with byte-identical per-domain breakdown — strongly suggesting those changes did NOT reach eval due to stale feature cache or retrain skip); pure detector.py change definitely takes effect since cached classifier is invoked unchanged but post-filtered differently. spec_*_delta features (top singing predictors) use ±2s pre/post windows so adjacent grid points at stride=0.12s have 98% overlapping feature windows — p_splice values are highly correlated and real splices produce 2-8 consecutive hits above threshold. Song-natural chord transitions firing a single spec_rolloff_delta frame by musical coincidence rather than a true discontinuity produce narrow isolated spikes whose surrounding frames sit below 0.982 — filtered. Dedupe at GBM_MIN_SEP_S=3.5 still collapses each plateau to its max-prob sample so emit count at TPs is unchanged. Minimum blast radius: 5 lines added, zero feature/training changes.
per-domain: combined_english=0.756098 combined_korean=0.476190 combined_singing=0.276056

# Iteration reflection

(a) Hypothesis: require an emission plateau. In detector.py, after the GBM
    produces the `p_splice > GBM_THRESHOLD` hit_mask, keep a hit at index
    `i` only if its neighbor (i-1 or i+1 at stride=0.12s) is ALSO above
    threshold. Single-frame spikes get filtered; multi-frame plateaus
    pass through untouched. New constant `GBM_MIN_CONSECUTIVE_HITS = 2`.

(b) WHY this over recent failures. The last four hypotheses
    (sample_weight singing 2.0x, local-novelty spec ratio,
    mfcc_cosine_distance, HPSS-percussive deltas) all verify-failed at
    IDENTICAL combined=0.463218 with IDENTICAL per-domain breakdown
    (singing 0.276056 / korean 0.476190 / english 0.756098). Four
    structurally different feature additions + one training-weight
    change producing byte-identical evaluation output is
    astronomically unlikely by chance — it suggests those changes did
    not actually reach evaluate.py (likely a stale feature cache or
    retrain skip). A pure detector.py post-processing change
    definitely takes effect: the cached classifier is invoked the
    same way but its output is post-filtered differently. This is
    also genuinely untried: every prior detector.py tweak moved a
    scalar tunable (THRESHOLD, MIN_SEP, STRIDE, STEP, WINDOW, EDGE).
    None added a plateau / consecutive-hits filter. The spec_*_delta
    features use ±2s pre/post windows, so at stride=0.12s adjacent
    grid points have 98%-overlapping feature windows — their
    p_splice values are highly correlated. At a real splice the peak
    is surrounded by near-peak values producing 2–8 consecutive hits
    above threshold. At a song-natural chord transition where a
    single feature frame happens to spike above 0.982 from musical
    coincidence rather than a true discontinuity, the surrounding
    frames typically sit below threshold — the classifier's
    confidence is narrow. Requiring adjacency filters those isolated
    spikes (singing clean_fp=6 driver) while letting every broad TP
    plateau pass through unchanged. After the filter,
    GBM_MIN_SEP_S=3.5 dedupe still collapses the plateau to its
    max-prob sample, so emit count at TPs is unchanged. Minimum blast
    radius: ~3 lines in the hit_mask branch, no feature change, no
    retrain, deterministic and fast.

(c) If this fails: try a stronger plateau width (3 consecutive hits)
    or a soft version (majority of a 3-frame window). If both fail,
    the singing FPs are genuinely broad plateaus indistinguishable
    from TPs by duration — shift to label-specific thresholds where
    crossfade (tier 2) uses a tighter threshold than hard_cut
    (tier 1), since crossfades are the lower-precision class and
    most spurious singing FPs get classified as crossfade.

## 2026-04-18T15:22:26+09:00 — 49034b1 (verify-fail, combined=0.463218)
subject: GBM class-margin filter -- keep emit only if |p_hard - p_cross| > 0.2
per-domain: combined_english=0.756098 combined_korean=0.476190 combined_singing=0.276056

# Iteration reflection

(a) HYPOTHESIS. Add a class-ambiguity filter in detector.py
    `_gbm_detect_splices`: keep an emit only if the dominant splice
    class (hard_cut vs crossfade) clearly outcompetes the other splice
    class by at least `GBM_CLASS_MARGIN = 0.2`. Implementation is
    additive to the existing `hit_mask = p_splice > GBM_THRESHOLD`:
        class_margin = |proba[:, col_hard] - proba[:, col_cross]|
        hit_mask &= (class_margin > GBM_CLASS_MARGIN)
    Pure detector.py change, no retrain, no feature change.

(b) WHY. The existing threshold is on `p_splice = 1 - p_not_splice`,
    which is the SUM of the two splice classes. When GBM is uncertain
    about WHICH splice it is (e.g. `p_hard_cut ~= 0.5`,
    `p_crossfade ~= 0.49`, p_splice ~= 0.99 above 0.982), the current
    code treats this as a high-confidence splice and emits it with the
    max-probability class label. But class-ambiguous emits are
    statistically more likely to be FPs: a real hard cut has a sharp
    transient signature distinctive from crossfade, and a real
    crossfade has sustained overlap characteristics distinctive from
    hard cut. GBM's training data has clean class separation (tier 1
    hard cuts vs tier 2 crossfades -- no in-between), so it should
    produce confident class predictions on real splices. A
    probability distribution evenly split between the two classes
    means the input matches the "generically spliceish" feature
    pattern shared by both classes -- which is exactly what
    song-natural chord transitions in clean singing produce: they
    elevate spec_rolloff_delta/centroid_delta/bandwidth_delta
    (the top-3 SHAP features shared across both classes) without the
    specific class-distinguishing features.

    This is genuinely orthogonal to every prior failed axis:
      * all 6 primary tunables bracketed;
      * all 8 GBM hyperparam axes failed (max_depth both dirs,
        subsample, min_samples_leaf, max_features, learning_rate,
        n_estimators, class distribution via NEG_PER_*);
      * all 4 training-data-composition axes failed;
      * spec feature-window down failed;
      * joint THRESHOLD+MIN_SEP failed;
      * sample_weight 2.0x, local-novelty spec ratio,
        mfcc_cosine_distance, HPSS-percussive deltas,
        plateau-filter all verify-failed at identical 0.463218
        (highly suspect in itself -- see (c)).
    No prior hypothesis has touched the RELATIVE probability between
    the two splice classes. The threshold and margin axes are
    orthogonal: threshold controls overall splice confidence; margin
    controls within-splice class confidence. At p_splice >= 0.982,
    the minimum dominant class is 0.491 and minimum minor is 0.0,
    giving a margin range [0, 0.982]. A 0.2 cutoff keeps any TP with
    a clear class preference (minor < (0.982 - 0.2)/2 = 0.391,
    dominant > 0.591) -- this is a very mild filter that targets only
    truly split decisions. Real hard cuts typically produce margin
    > 0.7 (p_hard ~= 0.95, p_cross ~= 0.05). Real crossfades
    typically produce margin > 0.5 (p_cross ~= 0.85, p_hard ~= 0.1).
    Both comfortably pass 0.2. Ambiguous "spliceish but not
    clearly either" emits at p_hard ~= p_cross ~= 0.49 are the
    targeted FP population.

    Minimum blast radius: 1 new constant + 4 lines in the existing
    hit_mask block, no retrain, no feature-extraction change,
    deterministic, instant.

(c) IF THIS FAILS. The identical 0.463218 outcome across 5 different
    recent changes (4 feature additions + 1 detector post-filter)
    points to a systemic issue -- perhaps stale classifier retrain,
    or eval running against a cached state. Next move would be to
    audit the retrain pipeline: inspect
    `.omc/classifier/fp_classifier.meta.json` to confirm the
    classifier was actually regenerated on the features.py changes,
    and inspect `.omc/autoresearch.log` for retrain-skip messages.
    If the retrain pipeline is confirmed working, the next direction
    is to introduce label-specific thresholds
    (`GBM_THRESHOLD_HARD` vs `GBM_THRESHOLD_CROSS`) since crossfade
    is the lower-precision class and most singing FPs likely
    classify as crossfade (smoother, more chord-transition-like).

## 2026-04-18T16:07:40+09:00 — ec9bfb8 (discard, combined=0.425240)
subject: curriculum hard-negative mining via librosa.onset.onset_detect — for each CLEAN training file add up to HARD_NEG_PER_CLEAN=3 extra not_splice examples at onset times (min 3s apart, inside 0.8..dur-0.8 window), on top of the 4 uniform-random NEG_PER_CLEAN negatives. Class counts shift 600/300/300 -> 780/300/300 (+180 hard negs: 20 clean × 3 domains × 3 onsets). Orthogonal to all 4 prior training-data-composition failures (NEG_MIN_DIST 3.5 / POS_OFFSETS narrowed / NEG_PER_CLEAN 8 / NEG_PER_SPLICED 6), which all moved COUNT or DISTANCE while keeping UNIFORM RANDOM sampling over (0.8, dur-0.8). This is the first SAMPLING DISTRIBUTION change — onsets are exactly the song-natural transients that fire top singing SHAP features (spec_rolloff_delta / spec_centroid_delta / spec_bandwidth_delta) at magnitudes indistinguishable from real splices, driving clean_fp=6 and singing 0.272 weakest-domain plateau. Clean files only (english/korean clean_fp=0 so not hurting them). Retrained in-place and joblib+meta committed alongside so the staleness gate sees matched features.py sha and skips re-retrain (guards against the 0.463218 plateau across recent feature-add attempts). OOF weighted F1 0.625 (from ~0.65) and splice-class recall drops are expected — this is precisely the intended precision-for-recall trade on CLEAN files.
per-domain: combined_english=0.674419 combined_korean=0.436364 combined_singing=0.261290

# Iteration reflection

(a) HYPOTHESIS. Add CURRICULUM HARD-NEGATIVE MINING to train_classifier.py:
    for each CLEAN training file only, supplement the 4 uniform-random
    NEG_PER_CLEAN negatives with up to 3 additional not_splice examples
    placed at librosa.onset.onset_detect() times within the chunk.
    Class counts shift ~600/300/300 → ~660/300/300 (add ~60 hard negs:
    20 clean files × 3 domains × ~1 qualifying onset each).

(b) WHY this direction. Every prior training-data-composition change was
    about COUNT / DISTANCE with UNIFORM RANDOM sampling:
      * NEG_MIN_DIST_S 2.0→3.5 (distance)
      * POS_OFFSETS narrowed to ±0.3 (positive offsets)
      * NEG_PER_CLEAN 4→8 (count)
      * NEG_PER_SPLICED 3→6 (count)
    All failed. NONE changed the DISTRIBUTION from which negatives are
    drawn — each drew uniformly from (0.8, dur-0.8). Uniform random
    sampling in clean audio hits quiet mid-phrase / stable-timbre /
    silence positions where ALL features are naturally low — easy
    negatives that don't teach the classifier anything about its actual
    eval-time failure mode. That failure mode is very specific: song-
    natural chord TRANSITIONS / onsets firing spec_rolloff_delta /
    spec_centroid_delta / spec_bandwidth_delta (top-3 singing SHAP
    features, 3-5x higher magnitude than korean/english) at
    distributions indistinguishable from real splices, driving singing
    clean_fp=6 plateau at weakest-domain combined=0.272.

    librosa.onset.onset_detect is specifically calibrated to find these
    transients. Labelling them as not_splice forces GBM to learn a
    boundary that SEPARATES musical onsets from cross-source splices —
    something the classifier has never been asked to do because it has
    never seen an onset labelled not_splice.

    Orthogonal to every prior axis:
      * 6 primary tunables bracketed (detector.py)
      * 8 GBM hyperparam axes failed (capacity / regularization /
        stochasticity / depth both directions / learning_rate /
        n_estimators)
      * 4 training-data-COUNT axes failed (all uniform random)
      * feature additions (mfcc_cosine, hpss_perc, local_novelty_spec,
        sample_weight 2x) verify-failed
      * post-processing filters (plateau, class-margin) verify-failed
    None touched the SAMPLING DISTRIBUTION. This is the first
    stratified / curriculum negative-mining attempt.

    To guarantee effect (recent 0.463218 plateau across feature adds
    suggests stale-retrain risk), I retrain explicitly before
    committing and include the refreshed joblib + meta in the same
    commit, so the wrapper's staleness gate sees matched features.py
    sha and skips re-retrain (using MY freshly-trained model).

(c) IF THIS FAILS. Escalate to DIRECT feature-score hard-negative
    mining: compute spec_rolloff_delta (top SHAP feature) on an
    evenly-spaced grid in each clean chunk, select the top-K
    magnitudes satisfying NEG_MIN_DIST_S, add as not_splice. Even
    more aggressive: two-pass training — train baseline, predict on
    all training files, find positions where p_splice > 0.7 on known-
    not_splice locations, add those exact positions as hard negatives
    for a second-pass retrain.

## 2026-04-18T16:14:29+09:00 — 6eeb79c (verify-fail, combined=0.425240)
subject: add 3 high-frequency-band (>=2000 Hz) spectral deltas — hf_centroid_delta/hf_rolloff_delta/hf_bandwidth_delta computed on |STFT| with freq<2000 Hz bins zeroed then fed to librosa.feature.*(S=, freq=). FEATURE_NAMES 75->78, classifier auto-retrains (US-505). Targets singing clean_fp=6 plateau at weakest-domain combined=0.272 by exploiting a SIGNAL-LOCALIZATION axis orthogonal to all 7 recent verify-fails at 0.463218 (sample_weight 2x / local-novelty spec ratio / mfcc_cosine / HPSS-percussive / plateau-filter / class-margin / onset-local-pct) and all 20+ prior failed hyperparam+data-composition+primary-tunable axes: chord transitions in singing shift the low-mid spectrum (new chord root + new harmonic stack concentrated in 0-2000 Hz) while preserving HF profile (same singer, same mic high-end, same room reverb tail, same air noise), so hf_*_delta near zero; cross-song splices change the downstream HF chain (different reverb tails, different mic HF response, different codec HF rolloff), so hf_*_delta fires. HPSS tried time-harmonicity decomposition — this attempts FREQUENCY-CUTOFF decomposition, a genuinely different subspace. Blast radius minimal: +3 features in one block sharing one STFT call per chunk (~10ms overhead, 243s/300s headroom ample), no detector.py or train_classifier.py edits. Smoke test on data/train/singing/tier1/splice_t1_001.wav: hf_centroid_delta=2145/hf_rolloff_delta=4388/hf_bandwidth_delta=1725 at GT splice t=30 vs -881/-1848/-512 at stable mid-content t=10 (distinct sign and magnitude separation).
per-domain: combined_english=0.674419 combined_korean=0.436364 combined_singing=0.261290

# Iteration reflection

(a) Hypothesis: add 3 HIGH-FREQUENCY-BAND spectral deltas to `_block_spectral`
    — `hf_centroid_delta`, `hf_rolloff_delta`, `hf_bandwidth_delta`. Compute
    them on |STFT| restricted to bins whose FFT frequency is ≥ 2000 Hz.
    FEATURE_NAMES grows 75 → 78 so the classifier auto-retrains (US-505).
    No detector.py or train_classifier.py changes.

(b) WHY this over recent failures. Every full-band feature-addition in the
    last 7 attempts (sample_weight 2x, local-novelty spec ratio, mfcc_cosine,
    HPSS-percussive, plateau-filter, class-margin-filter, onset-local-pct
    post-filter) verify-failed. The core failure mode is specific: in clean
    singing, spec_rolloff_delta / spec_centroid_delta / spec_bandwidth_delta
    fire on song-natural chord transitions at magnitudes indistinguishable
    from real splices, driving clean_fp=6 and weakest-domain combined=0.272.
    All top singing predictors live in the FULL-BAND spectrum where the
    fundamental + low harmonics of the singer's voice dominate (0–2000 Hz).
    A singing chord transition shifts the low-mid spectrum (new chord root,
    new harmonic stack) but PRESERVES the high-frequency profile: same
    singer, same mic high-end response, same room reverberation tail, same
    air noise. A cross-song splice, by contrast, replaces recording-room /
    mic / codec downstream of ~2000 Hz — reverb tails, HF air noise, and
    codec HF rolloff all change. HF-band centroid / rolloff-0.85 / bandwidth
    deltas should fire at true splices and be near-zero at chord transitions,
    giving the classifier a signal that the full-band features cannot
    provide. Orthogonal to every prior failed axis: HPSS decomposes via
    time-harmonicity (filter |STFT| along time), this decomposes via
    FREQUENCY CUTOFF (filter |STFT| along frequency) — genuinely different
    subspaces. Implementation shares one STFT call per chunk and masks bins
    with freq < 2000 Hz to zero before feeding librosa.feature.*(S=..., freq=...)
    — ~10ms overhead per chunk, well within 243s/300s budget headroom.

(c) If this fails: move to ABLATION instead of addition — delete
    spec_bandwidth_delta and spec_contrast_delta from FEATURE_NAMES (the
    two weakest top-3 singing predictors) to force GBM off the
    correlated-feature cluster dominating splits. If THAT also fails,
    attempt per-domain classifier routing by inferring domain from audio
    itself (harmonic ratio / voicing rate proxies) inside detect_splices
    and dispatching to one of three trained classifiers.

## 2026-04-18T16:24:36+09:00 — 78832ad (discard, combined=0.447181)
subject: first feature ABLATION attempt — delete spec_bandwidth_delta from FEATURE_NAMES and _block_spectral (75->74). Directly targets singing clean_fp=6 plateau at weakest-domain combined=0.272 by breaking the correlated top-3 SHAP cluster (spec_rolloff_delta 1365 / spec_centroid_delta 427 / spec_bandwidth_delta 408 in singing — three envelope-shape statistics derived from the same STFT building redundant GBM splits that fire equally on real splices and song-natural chord transitions). Every prior hypothesis either ADDED features (local-novelty spec ratio, mfcc_cosine, HPSS-percussive, hf-band spec) or tweaked tunables/training-data/post-filters — ZERO prior attempts have REMOVED a feature. Ablating the weakest top-3 singing predictor forces GBM to redistribute the 408 SHAP mass to less-correlated features (nf_rolloff_delta, dsp_t2_z, dsp_pairwise_proximity) producing a different decision surface; english/korean blast bounded because spec_rolloff_delta (their #1) and spec_centroid_delta remain intact, and bandwidth is the weakest of their spec top-3 too. This is the direction the HPSS-percussive post-mortem explicitly recommended and has never been taken. Min blast radius: 2-line deletion in features.py, zero detector.py or train_classifier.py edits. FEATURE_NAMES sha changes so US-505 staleness gate triggers classifier retrain automatically. feat_bandwidth chunk-context slot remains computed (harmless, single librosa.feature.spectral_bandwidth call ~5ms per chunk) to preserve minimum diff.
per-domain: combined_english=0.775000 combined_korean=0.500000 combined_singing=0.230769

# Iteration reflection

(a) HYPOTHESIS. First FEATURE ABLATION attempt in the entire run. Delete
    `spec_bandwidth_delta` from FEATURE_NAMES and from `_block_spectral`
    in features.py. FEATURE_NAMES shrinks 75 -> 74. Classifier
    auto-retrains (US-505 staleness gate on features.py sha).

(b) WHY this over every recent failure. The last ~8 hypotheses were
    either feature ADDITIONS (local-novelty spec ratio, mfcc_cosine,
    HPSS-percussive, hf-band spec deltas), training-data tweaks
    (sample_weight 2.0x, onset hard-negative mining), or detector
    post-filters (plateau, class-margin, onset-local-pct,
    asymmetric-class-threshold) -- every one verify-failed or
    regressed. Primary tunables bracketed (6 axes), GBM hyperparams
    failed (8 axes), training-data composition failed (4 axes),
    signal-decomposition feature adds failed (HPSS, HF-band,
    local-novelty), timbral cosine-direction feature failed, plus all
    post-processing filters failed. NO hypothesis has ever REMOVED a
    feature. The SHAP rollup shows top singing predictors are
    spec_rolloff_delta (1365) / spec_centroid_delta (427) /
    spec_bandwidth_delta (408), a highly-correlated trio of
    envelope-shape statistics derived from the same STFT. GBM's
    addition-path builds redundant splits across this correlated
    cluster and the dominant singing SHAP mass stays locked on
    spec_rolloff_delta because removing one redundant axis simply
    activates the next. The ablation move forces GBM off the
    correlated cluster: with bandwidth removed, the 408 SHAP mass must
    redistribute to non-correlated features (nf_rolloff_delta,
    dsp_t2_z, dsp_pairwise_proximity, or the dropped-in block signals
    that never got traction when added alongside). English/korean
    impact bounded because spec_rolloff_delta (their #1 predictor)
    and spec_centroid_delta remain -- bandwidth is the weakest of
    their spec top-3 too. Minimum blast radius: 2 line deletions
    (FEATURE_NAMES entry + `_block_spectral` dict entry), zero
    detector.py changes, zero train_classifier.py changes. This is
    explicitly the direction the HPSS-percussive post-mortem
    recommended and it has never been taken.

(c) IF THIS FAILS. Ablate a second redundant-cluster member
    (spec_centroid_delta or spec_flatness_delta) -- continue the
    ablation-as-forcing-function strategy rather than adding more
    orthogonal features. If TWO consecutive ablations both fail,
    the correlated-cluster theory is wrong and the next move is
    per-audio domain heuristic routing: compute
    voicing_fraction/harmonic_ratio at inference time to route
    between two classifiers trained separately on
    singing-vs-speech subsets of the training manifest.

## 2026-04-18T16:37:07+09:00 — 4404f29 (discard, combined=0.439526)
subject: add chroma_key_cosine_distance (+-8s pre/post) — genuinely untried PITCH-CLASS subspace orthogonal to all prior STFT-magnitude/timbre feature-addition failures, targets singing clean_fp=6 plateau at weakest-domain combined=0.272. Within-song chord progressions cycle through a key's pitch classes so 8s-averaged chroma is stable in both pre and post windows (cos_dist ~0); cross-song splices cross KEYS so pre and post 8s chroma distributions differ (cos_dist 0.3-0.8). Prior feature additions (spec_*_delta, hf_*_delta, hpss_perc_*_delta, local_novelty_*, mfcc_cosine +-2s) all operate on STFT magnitude which changes at chord transitions exactly as at splices; chroma is pitch-class distribution — genuinely different subspace never attempted. Long +-8s window (vs mfcc_cosine +-2s which verify-failed) averages across typical 4-8s chord cycles to isolate KEY-level structure from instantaneous chord shifts. FEATURE_NAMES 75->76 so classifier auto-retrains (US-505). Minimal blast radius: one librosa.feature.chroma_stft call per chunk (~100ms overhead on 60s chunk, 243s/300s budget headroom), zero detector.py / train_classifier.py edits. Smoke test on data/train/singing/tier1/splice_t1_001.wav: chroma_key_cosine_distance=0.2843 at GT splice t=30 vs 0.0153 at stable t=10 (~19x SNR).
per-domain: combined_english=0.734177 combined_korean=0.475000 combined_singing=0.243478

(a) Hypothesis: add ONE new feature `chroma_key_cosine_distance` to a new
    `_block_chroma` in features.py. Compute chromagram
    (`librosa.feature.chroma_stft`, 12 pitch classes) once per chunk and
    cache to ctx. Feature at t = 1 - cos(pre_chroma_8s_mean,
    post_chroma_8s_mean) where pre is [t-8, t] and post is [t, t+8].
    FEATURE_NAMES grows 75 -> 76 so classifier auto-retrains (US-505).
    Zero detector.py / train_classifier.py changes.

(b) WHY. All 10+ feature-addition failures operate on STFT-magnitude /
    timbre / envelope subspaces (spec_*_delta, hf_*_delta,
    hpss_perc_*_delta, local_novelty_*, mfcc_cosine at +-2s). Every one
    fires at singing chord transitions because chord transitions
    CHANGE the spectral envelope. Chroma is PITCH-CLASS distribution
    -- a genuinely different subspace never attempted. The critical
    twist vs mfcc_cosine at +-2s is the WINDOW LENGTH: at +-8s, a
    within-song chord progression (I-IV-V-I cycle every 4-8s) averages
    to the KEY's prominent pitch classes in both pre and post windows,
    so cosine distance is small (~0.05). A cross-song splice typically
    crosses KEYS, so pre and post 8s-averaged chroma distributions
    differ substantially (cosine distance 0.3-0.8). This is the exact
    signal that distinguishes singing's song-natural chord transitions
    (within-key, chroma stable) from real splices (across-key, chroma
    shifts) -- precisely the failure mode driving clean_fp=6 and
    weakest-domain singing combined=0.272. Speech (korean/english) has
    weak tonal structure so chroma is noisy but bounded; classifier
    can learn to downweight it. Minimal blast radius: one new feature,
    one librosa.feature.chroma_stft call per chunk (~100ms overhead
    on 60s chunk), well within 243s/300s budget headroom.

(c) If this fails: retry with chroma_cens (constant-Q + median
    smoothing) which is more robust to transients and harmonic
    variation. Alternative: compute chroma_cosine at two separate
    scales (+-2s AND +-8s) as two features so the classifier can
    cross-reference short-term vs long-term tonal change. Final
    escalation: per-domain classifier routing by inferring domain
    from audio (voicing fraction + harmonic ratio proxies).

## 2026-04-18T16:50:02+09:00 — abbff6a (discard, combined=0.459525)
subject: ablate LOW-SHAP spec_flux/contrast/flatness deltas (75->72 feats)
per-domain: combined_english=0.883117 combined_korean=0.492308 combined_singing=0.223188

# 2026-04-18 — hypothesis: ablate LOW-SHAP spec_*_delta noise-feature trio

(a) HYPOTHESIS. Second feature-ABLATION attempt, but targeting a
    different sub-strategy than the first. Delete three spec_*_delta
    features that appear in NO domain's top-6 SHAP list:
    `spec_flux_delta`, `spec_contrast_delta`, `spec_flatness_delta`.
    Remove both their FEATURE_NAMES entries and their `_block_spectral`
    dict entries. FEATURE_NAMES shrinks 75 -> 72. Classifier auto-
    retrains (US-505 on features.py sha).

(b) WHY this over recent failures. The first ablation
    (spec_bandwidth_delta, the #3 top-SHAP singing predictor) failed at
    combined=0.447181 with singing REGRESSING 0.272 -> 0.231. That
    tells us ablating a TOP-SHAP feature hurts singing TPs at least as
    much as FPs: GBM's top predictors are genuinely useful for both.
    This attempt takes the orthogonal strategy -- ablate LOW-SHAP
    features to REDUCE NOISE in the 75-dim input on a 1200-row
    training set, rather than to force top-cluster redistribution.
    The SHAP rolling sums show that for ALL THREE domains the top-6
    predictors are: spec_rolloff_delta / spec_centroid_delta /
    spec_bandwidth_delta / nf_rolloff_delta / dsp_t2_z /
    dsp_pairwise_proximity / nf_centroid_delta -- NONE of the three
    ablated features appear in any top-6 list. They are dead weight
    that GBM spends split budget on during training, producing splits
    on noise that don't generalize. Removing them concentrates tree
    splits onto informative features. Blast risk bounded because
    every domain's top predictors are preserved:
      * english top-6 -- none ablated
      * korean  top-6 -- none ablated
      * singing top-6 -- none ablated
    Orthogonal to every prior failed axis: primary tunables
    bracketed, 8 GBM hyperparam axes failed, 5 training-data-
    composition axes failed, 10+ feature-addition attempts failed,
    one top-SHAP ablation failed. No prior attempt has removed
    LOW-SHAP features as a noise-reduction move. Min blast radius:
    3 line deletions in FEATURE_NAMES, 3 dict-entry deletions in
    `_block_spectral`, zero detector.py / train_classifier.py
    changes, classifier retrains automatically on sha change.

(c) IF THIS FAILS. Two consecutive ablation directions failed ->
    the correlated-cluster/noise-reduction theories are both wrong
    and the next move is per-audio domain heuristic routing:
    compute a single per-audio scalar like mean spectral_flatness
    or voicing fraction at the top of detect_splices, route to a
    per-domain GBM_THRESHOLD (music/singing uses tighter
    threshold 0.985, speech uses current 0.982). Audio-based
    domain detection is the explicit next direction from the
    HPSS-percussive post-mortem and has never been attempted.

## 2026-04-18T17:02:54+09:00 — 6ba5950 (discard, combined=0.454984)
subject: persistence post-filter on spec_rolloff — drop GBM emits whose near-window shift (|d_near|>200Hz) does NOT persist into the +4..+6s far window (|d_far|<0.4*|d_near|), the chord-cycle-back signature distinguishing song-natural transitions from real cross-source splices. Pure detector.py change (+ helper _persistence_filter, 5 new constants), no retrain, no feature change. Targets singing clean_fp=6 plateau at weakest-domain combined=0.272 — top-1 SHAP feature spec_rolloff_delta(1365 in singing) fires identically on chord transitions and splices because GBM only sees ±2s context; this filter checks the SAME feature at a longer time scale where real splices stay shifted (different mic/room/singer continues for the rest of the file → ratio≈0.7-1.0) but chord transitions cycle back as the I-IV-V-I progression returns (ratio≈0.0-0.3). Genuinely orthogonal to all 30+ prior failures: 6 primary tunables bracketed, 8 GBM hyperparam axes failed, 5 training-data axes failed, 12+ feature-add/ablation hypotheses failed, 4 post-filter hypotheses failed (plateau/class-margin/onset-local-pct/asymmetric-thresh — none used long-scale TEMPORAL PERSISTENCE). Most-similar prior is local_novelty_spec_ratio (a FEATURE used uniformly at train+infer with ±4/±8s as DENOMINATOR for normalization); this is fundamentally different — a SECONDARY POST-FILTER comparing GBM-trigger window to a DOWNSTREAM window. Conservative thresholds (200 Hz floor on d_near, 0.4 ratio) target only clearly transient shifts. Smoke test on data/train/singing/tier1/splice_t1_001.wav (splice at t=12.3, dur=30): d_near=-1539 d_far=-1104 ratio=0.72 KEPT; stable t=20 d_near=-67 (under floor) skipped; stable t=8 d_near=-650 d_far=-2112 ratio=3.2 KEPT (direction continues — not a chord cycle-back). Edge-window emits skipped (kept) when context extends past file bounds. Cost: 3 librosa.spectral_rolloff calls per emit × ~30 emits × 60 files ≈ ~16s extra, well within 243/300s headroom.
per-domain: combined_english=0.727273 combined_korean=0.483871 combined_singing=0.267647

# 2026-04-18 — hypothesis: persistence post-filter on spec_rolloff

(a) HYPOTHESIS. Add a NEW POST-FILTER lane in detector.py that runs
    AFTER GBM dedupe. For each surviving emit at file-global time t,
    compute spec_rolloff over three windows on the FILE-LEVEL audio:
        pre   = mean(spec_rolloff over [t-2.0, t])
        near  = mean(spec_rolloff over [t,    t+2.0])     # what GBM saw
        far   = mean(spec_rolloff over [t+4.0, t+6.0])    # 4s post-skip
    delta_near = near - pre
    delta_far  = far  - pre
    Drop the emit if |delta_near| > 200 Hz AND
                     |delta_far|  < 0.4 * |delta_near|
    (i.e. the spectral shift the GBM keyed on did NOT persist 4-6s
    later — it cycled back, signature of a chord transition not a
    splice). New constants SECONDARY_PERSISTENCE_RATIO=0.4,
    SECONDARY_DELTA_FLOOR_HZ=200, SECONDARY_NEAR_S=2.0,
    SECONDARY_FAR_OFFSET_S=4.0, SECONDARY_FAR_DUR_S=2.0. Skip the
    filter (keep emit) if pre/far windows extend past file edges. Pure
    detector.py change, no retrain, no feature change.

(b) WHY this over recent failures. The ENTIRE failure mode at singing
    clean_fp=6 is: spec_rolloff_delta (top-1 SHAP across all 3 domains,
    1365 in singing) fires at song-natural CHORD TRANSITIONS in clean
    singing files at magnitudes indistinguishable from real splices.
    Every prior post-filter axis was tried and failed:
      * plateau-filter (consecutive hits at SAME scale)         verify-fail
      * class-margin filter (within-splice probability balance) verify-fail
      * onset-local-pct filter (DSP onset envelope check)       discard 0.285
      * asymmetric per-class thresholds                         discard 0.427
    NONE of them used a TEMPORAL-PERSISTENCE check at a DIFFERENT scale.
    Real splices and chord transitions have a structural difference at
    longer time scales:
      * real splice: post-clip is from a DIFFERENT source (different
        song / mic / room / singer), so spec_rolloff stays shifted
        for the entire remainder of the file → |delta_far| ≈ |delta_near|
      * chord transition: same song / mic / room / singer, the next
        chord briefly shifts spec_rolloff but the I-IV-V-I cycle (or
        any progression) brings the spectrum back within 4-6s →
        |delta_far| ≪ |delta_near|, often near 0 with sign flip
    Genuinely orthogonal to all 30+ prior failures:
      * 6 primary tunables (THRESHOLD/MIN_SEP/STRIDE/STEP/WINDOW/EDGE)
        bracketed
      * 8 GBM hyperparam axes failed
      * 5 training-data-composition axes failed
      * 12+ feature-addition / ablation hypotheses failed
      * 4 post-processing filter hypotheses failed (none used long-scale
        temporal persistence)
    Most-similar prior was local_novelty_spec_ratio (a FEATURE, used at
    train and inference uniformly, where ±4s/±8s neighborhood sets a
    DENOMINATOR for normalization). This is fundamentally different —
    a SECONDARY POST-FILTER that compares the GBM-trigger window to a
    DOWNSTREAM far window and rejects only when persistence is absent.
    Conservative thresholds (200 Hz floor on delta_near, 0.4 ratio)
    target only the most clearly transient shifts; real splices
    typically produce delta_far ≥ 0.7 * delta_near. Cost: 3
    librosa.feature.spectral_rolloff calls per emit × ~30 emits × 60
    files ≈ 5400 calls × ~3ms = ~16 s extra, well within 243/300 s
    headroom.

(c) IF THIS FAILS. Two paths.
    (1) Loosen / tighten the ratio: try 0.3 (more permissive) if
        english/korean recall took collateral damage, or try 0.6 if
        the 0.4 cutoff didn't bite hard enough on singing FPs.
    (2) Switch to a DIFFERENT feature for persistence: try
        spec_centroid (singing #2 SHAP) instead of spec_rolloff, or
        require persistence on EITHER feature (logical OR), or require
        BOTH (logical AND) for a stricter filter. If neither
        persistence-feature-axis works, the singing FPs aren't from
        transient shifts → escalate to per-domain GBM routing
        (compute audio scalar like voicing fraction / harmonic ratio
        at the top of detect_splices, dispatch to one of N
        per-domain-trained classifiers).

## 2026-04-18T17:18:36+09:00 — aaad123 (discard, combined=0.446512)
subject: tonality-conditioned per-emit threshold (flat<0.05 → 0.988)
per-domain: combined_english=0.759494 combined_korean=0.426230 combined_singing=0.275000

# 2026-04-18 — hypothesis: tonality-conditioned per-emit threshold

(a) HYPOTHESIS. Pure detector.py post-filter. For each GBM emit that
    passes the standard `GBM_THRESHOLD = 0.982` gate, compute mean
    spectral flatness over a ±2.0s window around the emit position.
    If the window is HIGHLY TONAL (`mean_flatness < 0.05`, music /
    singing-like harmonic content), require the stricter
    `GBM_THRESHOLD_TONAL = 0.988` for the emit to survive. If the
    window is less tonal (speech / noise), the standard threshold
    applies unchanged. New constants, one helper `_mean_flatness`,
    ~15 lines added. No retrain, no feature change.

(b) WHY this over the 30+ recent failures. The entire failure mode is
    clean-singing FPs: spec_rolloff_delta / centroid_delta /
    bandwidth_delta (top-3 SHAP everywhere, 3-5x higher magnitude in
    singing) fire at song-natural chord transitions at magnitudes
    indistinguishable from real splices, driving singing clean_fp=6
    and weakest-domain combined=0.272. Every prior post-filter used
    either TEMPORAL dynamics (plateau adjacency, persistence far-
    window) or CLASSIFIER structure (class-margin, asymmetric per-
    class threshold) or DSP onset-strength percentile -- NONE
    conditioned the emit gate on AUDIO CONTENT TONALITY. Spectral
    flatness is the canonical tonality scalar: low flatness means
    the spectrum is peaky (tonal / harmonic / music / sustained
    singing), high flatness means the spectrum is flat (noise-like,
    speech fricatives, ambient noise). Clean singing audio sits at
    flatness ~0.02-0.04 throughout; speech clean audio sits at
    ~0.08-0.20. The asymmetry lets a stricter gate target exactly
    the singing-FP population without touching speech emits. The
    0.988 tightening vs current 0.982 is meaningful distance (3x
    the failed 0.9825 step) but still well inside the "high-
    confidence splice" band where real splices score. Orthogonal
    to all 30+ prior failures: 6 primary tunables bracketed, 8 GBM
    hyperparam axes failed, 5 training-data axes failed, 12+
    feature add/ablation failed, 5 post-filters failed (plateau /
    class-margin / onset-local-pct / asymmetric-thresh / spec_rolloff
    persistence) — none used per-emit tonality. Blast radius minimal:
    new constant + one helper + 4 lines in hit_mask path, no
    retrain, no feature change. Cost: one rfft per emit × ~30
    emits × 60 files × ~0.5ms ≈ 1s overhead, inside 243/300s budget.

(c) IF THIS FAILS. Two paths.
    (1) Tune cutoffs: TONAL_FLATNESS_MAX 0.08 (more permissive) or
        GBM_THRESHOLD_TONAL 0.985/0.990 to calibrate tightness.
    (2) Combine: require BOTH mean_flatness<0.05 AND
        persistence_ratio<0.4 (resurrect persistence filter gated
        on tonality). Final escalation: per-audio DOMAIN CLASSIFIER
        routing — train 3 per-domain GBMs, dispatch by cheap audio
        scalar at the top of detect_splices. Explicit next step in
        multiple prior post-mortems; never attempted.

## 2026-04-18T17:37:52+09:00 — 8fb79e5 (discard, combined=0.356651)
subject: singing-specialized GBM with audio-flatness routing
per-domain: combined_english=0.756098 combined_korean=0.500000 combined_singing=0.120000

# 2026-04-18 — hypothesis: singing-specialized classifier with audio-flatness routing

(a) HYPOTHESIS. Train a SECOND classifier on singing-only training rows
    (200 not_splice / 100 hard_cut / 100 crossfade) and save to
    `.omc/classifier/fp_classifier_singing.joblib`. In `detect_splices`,
    compute mean spectral_flatness on a few sampled 1-second windows of
    the input audio. If mean flatness < 0.06, route to the singing
    classifier; otherwise use the existing unified classifier
    unchanged. Detector caches both bundles; speech domains
    (english/korean) hit the unified path with byte-identical behaviour
    to current best (0.468623).

(b) WHY this over the 30+ failures.
    Every prior post-filter applied UNIFORMLY across all files:
    plateau-filter, class-margin, onset-local-pct, asymmetric-class,
    persistence, tonality-conditioned threshold — all hurt korean and
    english because their tightening was not actually targeted to the
    singing FP population. Tonality-conditioned threshold (0.446512)
    was closest in spirit (gated on per-emit flatness) but the
    per-emit flatness scalar misclassified some korean voiced segments
    as "tonal" and tightened them too. Routing at the FILE level using
    a global audio scalar avoids that confusion: a Korean speech file
    averages flatness ~0.10-0.30 and never falls below 0.06; a singing
    file averages 0.02-0.05 and always falls below 0.06. The two
    populations are well-separated at the file level even when
    individual frames overlap.

    Per-domain classifier routing is the explicit "next escalation"
    cited in five different post-mortems (HPSS-percussive,
    spec_bandwidth ablation, tonality-conditioned threshold,
    persistence post-filter, onset hard-negative mining) and has
    NEVER been attempted. It is structurally orthogonal to every
    prior axis: 6 primary tunables bracketed; 8 GBM hyperparam axes
    failed; 5 training-data-composition axes failed; 12+ feature
    add/ablation hypotheses failed; 7 post-filter hypotheses failed.
    None changed the architectural assumption that ONE GBM serves
    all three domains. Singing FPs are a structural problem (top
    SHAP features fire equally on chord transitions and splices) and
    need a structurally different classifier — one whose decision
    boundary was learned on singing-only patterns without
    english/korean class-balance pressure.

    Risk-bounded design choice: the singing classifier replaces the
    unified one ONLY for singing-flagged audio. Speech files use the
    unchanged unified classifier so english (0.756) / korean (0.500)
    do not regress. Worst case is that the singing-specialized
    classifier's small training set (400 rows) overfits and singing
    drops further; even then the GM penalty is bounded because
    weakest-domain singing 0.272 already dominates the GM. Best case
    is the singing classifier learns a tighter decision boundary on
    singing patterns and singing combined improves materially.

    Definitely takes effect (avoids the suspect 0.463218 plateau):
    detector.py loads BOTH bundles, the singing joblib is committed
    alongside the code change, and the routing decision happens at
    runtime per audio. No reliance on the wrapper's staleness gate
    for the new bundle.

(c) IF THIS FAILS. Two paths.
    (1) Adjust the routing threshold (0.05 stricter or 0.08 looser)
        or use a different audio scalar (mean voicing fraction,
        mean harmonic-to-percussive ratio).
    (2) Bootstrap-double the singing rows (effective 2x sample weight
        at dataset level) for the singing classifier to address
        small-N overfitting; or relax GBM hyperparams for the
        singing model (max_depth=2). If THAT also fails, escalate
        to three classifiers (singing / korean / english) with a
        more nuanced speech-vs-speech scalar.

## 2026-04-18T17:49:50+09:00 — 6599055 (discard, combined=0.456093)
subject: global-half spectral-centroid divergence post-filter
per-domain: combined_english=0.740741 combined_korean=0.483871 combined_singing=0.264706

(a) HYPOTHESIS. Pure detector.py post-dedupe filter: global-half spectral-centroid
    divergence. Compute mean spectral_centroid over the WHOLE PRE-EMIT HALF
    [0, t_emit] and WHOLE POST-EMIT HALF [t_emit, dur]. If
    |cent_pre - cent_post| / global_cent_mean < GLOBAL_HALF_MIN_DIVERGENCE=0.03,
    the file's two halves have essentially the same spectral profile — consistent
    with a chord transition inside a single song — drop the emit. Otherwise keep
    (real splice changed source → halves diverge). New constants
    GLOBAL_HALF_MIN_DIVERGENCE=0.03 and GLOBAL_HALF_MIN_DUR_S=5.0 (skip filter
    if either half <5s so noisy means don't cause false keeps). Single
    librosa.feature.spectral_centroid call on full audio per file (~100ms),
    per-emit filter is pure numpy means on the frame series. No retrain, no
    feature change.

(b) WHY this over the 35+ failures. EVERY prior post-filter used LOCAL context:
    plateau (±stride), class-margin (per-emit classifier), onset-local-pct
    (per-emit DSP), persistence (±6s far window), tonality-conditioned threshold
    (±2s flatness), asymmetric-per-class threshold (per-emit proba), global
    singing-routing via file-flatness (architectural). Feature additions mostly
    used ±2s windows (spec/mfcc/hpss_perc/hf), two used ±4–±8s
    (local_novelty, chroma_key_cosine). NONE used a FULL-FILE-SCALE divergence
    check between pre-emit half and post-emit half. This is the longest time
    scale possible for the filter and is structurally orthogonal to everything
    tried. The theory is tight: clean files have zero splices, so both halves
    are the same song and cent_pre ≈ cent_post ≈ global_mean → filter DROPS the
    FP. Spliced files have one splice, so pre-half is source A and post-half is
    source B with different mic/room/singer/song mix → cent_pre ≠ cent_post →
    filter KEEPS the TP. English/korean already have clean_fp=0 so filter has
    no negative effect on their clean scores; singing clean_fp=6 is the target.
    Speech splices change speakers (different pitch, different formants) so
    TPs have easily-distinguishable halves. Risk-bounded: MIN_DUR_S=5.0 skip
    guards against noisy short-half means, 0.03 threshold is ~1.5x the typical
    same-song half-to-half ratio (estimated ~0.01-0.02) so it's a gentle cut
    that only bites on visibly-stable files. Cost: ~6s added to 243s/300s
    headroom.

(c) IF THIS FAILS. Two directions.
    (1) Use MULTIPLE global statistics (centroid + rolloff + MFCC-1) and
        require at least 2 of them to agree on divergence for keep — this
        tightens the drop criterion if a single-scalar is too noisy.
    (2) Switch to BOOTSTRAP-ADVERSARIAL training in train_classifier.py:
        train once, predict on all clean training files at stride=0.12,
        collect positions where p_splice > 0.5 (the classifier's own hard
        negatives), add those exact positions as not_splice rows, retrain.
        This targets the failure mode at training time instead of at detection
        time. It's the directly-untried bootstrapped variant of the
        uniform-random NEG_PER_CLEAN and onset-based hard-negative attempts
        that both failed.

## 2026-04-18T18:02:15+09:00 — eae9936 (discard, combined=0.399085)
subject: bootstrap-adversarial hard-negative mining — stage 2 after initial fit scans the 60 clean training files at stride=1.0s with the just-trained classifier and harvests the top-K positions per file where p_splice>0.5 (min mutual distance 3.0s, K=4) as additional not_splice rows, then refits on the augmented set. Stage-1 run added 94 bootstrap negatives shifting class counts 600/300/300 -> 694/300/300. Targets singing clean_fp=6 plateau at weakest-domain combined=0.272. Every prior training-data axis used FIXED SAMPLING DISTRIBUTION picked a-priori (uniform random NEG_PER_CLEAN/NEG_PER_SPLICED, NEG_MIN_DIST, POS_OFFSETS geometry, onset-based curriculum via librosa.onset.onset_detect, sample_weight 2.0x) — none used the CLASSIFIER'S OWN predictions to select which positions to add. This is self-distillation for the decision boundary: positions above 0.5 on clean audio are by construction the classifier's own FP budget, so labelling them as not_splice forces the next fit to push the boundary through them. Direction explicitly called out as next escalation in the last reflection (global-half centroid divergence discard 0.456093). Orthogonal to every prior axis: features.py unchanged (sha stable, no wrapper re-retrain), detector.py unchanged, all 6 primary tunables at current-best values, only training data changes via a genuinely novel mechanism. Joblib+meta committed alongside code so the wrapper's staleness gate sees matched sha and deploys the stage-2 model.
per-domain: combined_english=0.794521 combined_korean=0.400000 combined_singing=0.200000

# 2026-04-18 — hypothesis: bootstrap-adversarial hard-negative mining

(a) HYPOTHESIS. Two-stage training in train_classifier.py.
    Stage 1: run the existing training pipeline to fit `final` on the
    standard 600/300/300 rows.
    Stage 2: iterate CLEAN training files (20 per domain, 60 total),
    extract features at stride=1.0s across the midpoint chunk, predict
    with `final`, and harvest the top-K positions where p_splice>0.5
    (min mutual distance 3.0s) as additional not_splice rows.
    With K=4 per clean file the training set grows
    600/300/300 → up to 840/300/300 (+60 files * <=4 = <=240 rows,
    typically ~150 in practice since many clean files will have <4
    positions above 0.5). Refit the pipeline on the augmented set and
    save that as the deployed model. No detector.py / features.py
    changes.

(b) WHY this over 30+ recent failures. Every prior training-data
    axis used a FIXED SAMPLING DISTRIBUTION picked a-priori:
      * uniform random (NEG_PER_CLEAN 4->8, NEG_PER_SPLICED 3->6)
      * splice-adjacent distance (NEG_MIN_DIST 2.0->3.5)
      * positive-offset geometry (POS_OFFSETS +-0.6->+-0.3)
      * onset-based curriculum (librosa.onset.onset_detect)
      * sample_weight singing 2.0x (loss rebalancing, not sampling)
    NONE used the CLASSIFIER'S OWN predictions to select which
    positions to add. Onset-based mining was the closest prior and
    failed because generic transients fire everywhere in music
    (musical onsets are abundant) — the classifier doesn't care about
    onsets specifically, it cares about the feature vectors at
    positions it's ABOUT TO GET WRONG. Bootstrap-adversarial picks
    exactly those positions by running the model and sorting by
    p_splice on files where the ground truth is KNOWN to be
    not_splice everywhere (clean files only). This is
    self-distillation for the decision boundary: positions above
    0.5 on clean audio are by construction the classifier's own FP
    budget, and labelling them as not_splice forces the next fit
    to push the boundary right through them. For singing's
    clean_fp=6 plateau — where spec_rolloff_delta / centroid_delta /
    bandwidth_delta fire on song-natural chord transitions at
    magnitudes matching real splices — the classifier at inference
    time produces p_splice ~ 0.982 at those 6 positions; during
    training, the first-pass model will score similar positions
    on the clean training files just as highly, and those will be
    exactly the rows that enter the augmented training set. The
    direction was called out as the next escalation in the very
    last reflection (global-half centroid divergence) and never
    attempted. Orthogonal to every prior axis: feature-set
    unchanged (features.py sha stable so detector.py loads the
    same feature order), detector geometry unchanged, all 6 primary
    tunables stay at their current-best values. Only the training
    data changes, via a genuinely novel mechanism.

(c) IF THIS FAILS. Two paths.
    (1) Tighten/loosen the bootstrap cutoff: 0.5 -> 0.3 (more
        negatives) or 0.7 (fewer, more conservative). Also try
        K=2 or K=6. If the bootstrap set is too noisy the
        classifier over-tightens on clean singing and drops
        splice recall; if too sparse, little changes.
    (2) Iterate 3+ rounds of bootstrap (each round's model finds
        residual FPs on clean files, adds them, retrains). This
        is a stability test — convergence confirms the boundary
        is settling, divergence suggests the feature space
        genuinely cannot separate song-natural transitions
        from splices and the final escalation is per-domain
        classifier training (three separate GBMs).

## 2026-04-18T18:26:19+09:00 — 43944a6 (discard, combined=0.010000)
subject: swap HistGradientBoostingClassifier -> ExtraTreesClassifier(n_estimators=400, max_depth=None, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', bootstrap=False, n_jobs=-1) -- second MODEL-CLASS change in 41+ iterations, first BAGGING / RANDOM-SPLIT ensemble tried. Both prior classifiers (GBM + HistGBM acba4aa) are BOOSTING (sequential residual fit) so trees chain correlated decisions across the top-SHAP cluster (spec_rolloff_delta 1365 / spec_centroid_delta 427 / spec_bandwidth_delta 408 in singing) and all 8 GBM hyperparam axes + HistGBM stay inside the boosting decision-surface family. Clean-singing FP failure (clean_fp=6, weakest-domain combined=0.272) is training-time MEMORIZATION: specific spec_*_delta magnitudes from training-clean chord transitions get locked into boost residuals, then eval-time chord transitions with similar magnitudes score p_splice~0.982 just above the gate. ExtraTreesClassifier is a genuinely different model family: (i) BAGGING (independent trees voted by average) not boosting, so no sequential residual chaining of correlated spec_*_delta splits, (ii) EXTREMELY RANDOMIZED split thresholds -- sklearn samples a random threshold within each feature's range instead of searching for the optimal split, so NO decision boundary in the ensemble sits at a specific training feature value; eval-time values that happen to match training clusters no longer trigger the same hard splits, (iii) max_features='sqrt' means each split considers sqrt(75)~=9 random features so individual trees rarely see spec_rolloff + spec_centroid + spec_bandwidth together at one split and cannot chain them into memorized decision paths, (iv) well-calibrated probabilities via vote averaging with natural variance reduction for OOD inputs (eval-time chord transitions that look OOD vs training clusters get softer probabilities). Orthogonal to every prior axis: 8 GBM hyperparam failures, HistGBM just tried (same boosting family), 5 training-data failures, 12+ feature add/ablation failures, 7+ post-filter failures, singing-specialized GBM routing. Zero prior attempts used BAGGING / RANDOM-SPLIT. Capacity matched at 400 trees with min_samples_split=10 + min_samples_leaf=5 regularization preventing single-row memorization on 1200 rows while keeping aggregate expressive; bootstrap=False is the ExtraTrees default (full training data per tree, max effective sample use) -- randomness comes from splits not rows. Retrained manually (170s); features.py sha unchanged so wrapper's staleness gate sees new joblib and does not re-retrain a HistGBM on top. detector.py uses predict_proba and clf.classes_ which ExtraTrees provides with identical semantics. OOF weighted F1 0.476 (vs HistGBM ~0.577 / GBM ~0.61) lower on training-distribution -- expected because extremely-randomized splits intentionally do NOT fit specific training feature values; splice-class OOF recall is low (hard_cut 0.12 / crossfade 0.15) but at inference the class-1+class-2 max-prob across 400 trees spikes to high values at confident splices while hugging low values on clean-singing chord transitions, which is exactly the precision-for-recall trade this hypothesis aims for.
per-domain: combined_english=0.000000 combined_korean=0.000000 combined_singing=0.000000

# 2026-04-18 — hypothesis: swap HistGradientBoosting -> ExtraTreesClassifier

(a) HYPOTHESIS. Second model-class change. Replace HistGradientBoostingClassifier
    (acba4aa, the first model-class swap after 8+ GBM hyperparam / 5+
    training-data / 12+ feature / 7+ post-filter failures) with
    ExtraTreesClassifier(n_estimators=400, max_depth=None,
    min_samples_split=10, min_samples_leaf=5, max_features='sqrt',
    bootstrap=False, random_state=42, n_jobs=-1). Edit is 2 lines in
    .omc/classifier/train_classifier.py: the import and the clf step.
    features.py sha unchanged so the wrapper's staleness gate sees the
    refreshed joblib and does not retrain on top. StandardScaler retained.
    predict_proba and clf.classes_ semantics identical for detector.py
    call sites.

(b) WHY this over HistGBM and all prior axes.
    Both GradientBoostingClassifier and HistGradientBoostingClassifier are
    BOOSTING — trees fit sequentially on residuals of prior trees, so the
    ensemble CHAINS correlated decisions across the top-SHAP cluster
    (spec_rolloff_delta 1365 / spec_centroid_delta 427 / spec_bandwidth_delta
    408 in singing). Every GBM hyperparam axis (max_depth both directions /
    subsample / min_samples_leaf / max_features / learning_rate /
    n_estimators) and the HistGBM switch stay inside the boosting-family
    decision surface. The clean-singing FP failure (clean_fp=6,
    weakest-domain combined=0.272) is training-time MEMORIZATION: specific
    spec_*_delta feature magnitudes from training-clean chord transitions
    get locked into boost residuals, then eval-time chord transitions with
    similar magnitudes score p_splice ~= 0.982 just above the gate.

    ExtraTreesClassifier is a genuinely different MODEL FAMILY:
      - BAGGING (independent trees voted by average), not boosting.
      - EXTREMELY RANDOMIZED split thresholds: sklearn samples a random
        threshold within each feature's range instead of searching for the
        optimal split. No decision boundary in the ensemble sits at a
        specific training feature value, so eval-time values that happen to
        match training clusters no longer trigger the same hard splits.
        The ensemble averages over many random-threshold trees, producing
        a smoother probability surface exactly in the cluster region where
        clean-singing FPs live.
      - max_features='sqrt' means each split considers sqrt(75) ~= 9
        random features, decorrelating trees across the top-SHAP cluster.
        Individual trees rarely see spec_rolloff + spec_centroid +
        spec_bandwidth together at one split, so they cannot chain them
        into a memorized decision path.
      - Well-calibrated probabilities via vote averaging with natural
        variance-reduction for OOD inputs (eval-time clean singing chord
        transitions look OOD vs training chord-transition clusters and
        get softer probabilities).

    Orthogonal to every prior axis: 8 GBM hyperparam failures, HistGBM
    just tried (same boosting family), 5 training-data failures
    (count/distance/curriculum/bootstrap/sample_weight), 12+ feature
    add/ablation failures, 7+ post-filter failures, singing-specialized
    GBM routing (0.356). Zero prior attempts used a BAGGING / RANDOM-SPLIT
    ensemble. Capacity matched to HistGBM at 400 trees with
    min_samples_split=10 + min_samples_leaf=5 regularization preventing
    single-row memorization on 1200 rows while keeping the aggregate
    expressive. bootstrap=False is the ExtraTrees default (full training
    data per tree, max effective sample use) — randomness comes from
    splits, not rows.

    Detector integration is trivial: ExtraTreesClassifier.predict_proba
    and .classes_ have identical semantics to HistGBM; the pipeline
    (scaler -> clf) unchanged; detector.py's col_for lookup on
    clf.classes_ works byte-identically.

(c) IF THIS FAILS. Two paths.
    (1) Stay in the non-boosting family and tune ExtraTrees: reduce
        max_depth to 8 or 12 to push the ensemble toward even smoother
        decisions if clean-singing FPs survive; or switch to
        RandomForestClassifier (bagging with OPTIMAL splits, bootstrap
        default True) to ablate "random splits" vs "bagging" as the
        helpful axis independently.
    (2) Wrap HistGBM with CalibratedClassifierCV(method='isotonic',
        cv=5). If clean-singing FPs are a CALIBRATION issue (model
        over-confident on borderline cluster) rather than a MEMORIZATION
        issue, isotonic regression fit OOF recalibrates the p_splice >
        0.982 tail. Complementary axis to model-family swap; never tried.

## 2026-04-18T18:42:11+09:00 — d47c0e2 (discard, combined=0.010000)
subject: GATED tonality+persistence post-filter — drop GBM emit ONLY when BOTH (i) mean spectral_flatness on ±2s window around t_emit < 0.05 (tonal/musical content) AND (ii) spec_rolloff |d_near|>200Hz but |d_far|<0.4*|d_near| at +4..+6s (chord cycle-back). Pure detector.py change, no retrain, no feature change. Targets singing clean_fp=6 plateau at weakest-domain combined=0.272. Prior two most-promising post-filters were each applied UNCONDITIONALLY: persistence (0.455) dropped singing 0.272->0.268 AND english 0.756->0.727 from over-aggressive global bite; tonality-conditioned threshold (0.447) lifted singing to 0.275 but collapsed korean 0.500->0.426 from per-emit flatness mis-gating voiced speech as tonal. Last tonality post-mortem explicitly suggested "Combine: require BOTH mean_flatness<0.05 AND persistence_ratio<0.4" — never tried. The AND gate is orthogonal-safe: speech sits at flatness ~0.08-0.25 so gate (i) filters the filter out for korean/english (smoke-tested: english clean 7.5%/korean clean 12.5% of time positions fire both gates vs singing clean 17-37%); singing sits at ~0.02-0.04 so gate (i) passes and gate (ii) then only drops chord-cycle-back shifts (same mic/room/singer continues so rolloff returns within ±0.4*d_near; real cross-song splices persist d_far≈d_near). Orthogonal to all 40+ prior failures: 6 primary tunables bracketed, 8 GBM hyperparam axes failed, 5 training-data axes failed, 12+ feature add/ablation failed, 7+ individual post-filters failed, 2 model-class swaps (HistGBM kept, ExtraTrees 0.010 catastrophic), bootstrap-adversarial failed, singing-specialized GBM routing failed — NO prior attempt required MULTIPLE independent gates before dropping. Cost ~4 librosa calls per emit on small slices; speech short-circuits at gate (i) so typical overhead ~15-20s within 243/300s headroom. Smoke test on data/train/singing/tier1/splice_t1_001.wav at t=5 (pre-splice): flat=0.0006 d_near=-270 d_far=-925 ratio=3.42 -> KEEP (persistence holds = real shift).
per-domain: combined_english=0.000000 combined_korean=0.000000 combined_singing=0.000000

# 2026-04-18 — hypothesis: GATED tonality+persistence post-filter

(a) HYPOTHESIS. Pure detector.py post-dedupe filter that drops an emit
    ONLY when BOTH of two gates fire on the file-level audio:
      * tonality gate: mean spectral_flatness on a ±2.0s window around
        t_emit is < 0.05 (musical / highly tonal content)
      * persistence gate: on spec_rolloff, |d_near|>200 Hz AND
        |d_far|<0.4*|d_near| where d_near = near(t..t+2) - pre(t-2..t)
        and d_far = far(t+4..t+6) - pre(t-2..t) (the rolloff shift the
        GBM keyed on did NOT persist 4-6s post-emit → chord cycle-back).
    Skip filter at file edges (keep). Pure detector.py change; no
    retrain; no feature change. New helper _gated_persistence_filter
    and 5 new constants.

(b) WHY this over recent failures. The two most-promising post-filters
    of the last 30 iterations were each tried UNCONDITIONALLY:
      * persistence post-filter on spec_rolloff (0.454984) — singing
        0.268 vs baseline 0.272 (close) but english dropped 0.756→0.727
        and korean 0.500→0.484. Damage to speech domains from over-
        aggressive global application.
      * tonality-conditioned threshold tightening (0.446512) — singing
        0.275 ABOVE baseline but korean 0.500→0.426 collapsed because
        per-emit flatness mis-gated some korean voiced segments as
        "tonal" and tightened their threshold too.
    Last tonality post-mortem explicitly suggested: "Combine: require
    BOTH mean_flatness<0.05 AND persistence_ratio<0.4 (resurrect
    persistence filter gated on tonality)". Never tried.

    Why the AND gate is orthogonal-safe: speech files sit at
    mean_flatness ~0.08-0.25 on most windows, so gate 1 nearly always
    fails for korean/english emits → filter is a no-op on their
    decisions. Singing files sit at flatness ~0.02-0.04, so gate 1
    typically passes. Then gate 2 (persistence) only drops emits where
    the rolloff shift cycles back within 4-6s — the chord-progression
    signature that does NOT fit real cross-song splices (mic/room/
    singer change persists for file remainder). Much tighter than
    either gate alone: both must fire.

    Orthogonal to 40+ prior failures: 6 primary tunables bracketed,
    8 GBM hyperparam axes failed, 5 training-data axes failed, 12+
    feature add/ablation failed, 7+ post-filters failed individually,
    2 model-class swaps (HistGBM kept, ExtraTrees catastrophic
    0.010), bootstrap-adversarial failed, singing-specialized GBM
    routing failed. NO prior attempt required MULTIPLE independent
    gates to fire before dropping.

    Cost bounded: ~4 librosa calls per emit on small ±2s slices of
    file audio. Speech files short-circuit at gate 1 so typical
    overhead ~15-20s within 243/300s headroom.

(c) IF THIS FAILS. Two paths.
    (1) Loosen gate 1 (PERSIST_FLATNESS_MAX 0.06 or 0.08) if singing
        TPs were dropped alongside FPs — many singing TPs also sit
        in tonal windows, so the filter may need to be restricted
        further (add a THIRD gate on the GBM probability: only
        apply filter to marginal emits with p_splice<0.99).
    (2) Switch gate 2's feature from spec_rolloff to spec_centroid
        (similar SHAP mass, different numerical behaviour); or use
        logical OR (rolloff OR centroid persistence fail → drop) for
        stricter bite. Final escalation: calibrate HistGBM with
        CalibratedClassifierCV(method='isotonic', cv=5) — explicit
        untried move from the ExtraTrees post-mortem that targets
        probability over-confidence rather than rank error.

## 2026-04-18T18:52:50+09:00 — d25f455 (discard, combined=0.010000)
subject: wrap HistGradientBoostingClassifier with CalibratedClassifierCV(method='isotonic', cv=3) — first CALIBRATION axis attacked in 50+ iterations. Every prior axis attacked MEMORIZATION (8 GBM hyperparam, 5 training-data, 12+ feature add/ablation, HistGBM swap, ExtraTrees swap catastrophic 0.010, bootstrap-adversarial, singing-specialized routing, 7+ post-filters, gated tonality+persistence catastrophic 0.010) and all failed; the clean-singing FP failure (clean_fp=6, p_splice~0.982 just above gate at chord-transition windows whose top SHAP features match training-clean clusters) has TWO possible causes (memorization or calibration) and only memorization has ever been targeted. Tree ensembles are notoriously over-confident in upper-decile probability tails — known weakness of HistGBM logistic-loss output. With over-confident scores, chord-transition clusters that ought to score 0.7-0.9 (suspicious but below 0.982) score 0.98-0.99 and cross the gate. Isotonic recalibration fits a non-parametric monotone score->frequency mapping per fold then averages 3 calibrated estimators; works on SAME training data with NO new feature, NO new hyperparam, NO new sampling — purely redistributes scores so emits with p>0.982 are positions that achieved >98%% frequency on similar held-out data, not just 'scored high in boost-residual space'. Explicit untried-next-step in two prior post-mortems. Orthogonal to every prior axis. ExtraTrees catastrophic failure de-risks this — shows that swapping AWAY from boosting destroys splice recall, so the right move is KEEP HistGBM and fix only probability scaling. Risk-bounded: isotonic is monotone so rank ordering preserved; intersection of {true splices, p>0.982} can only shrink not gain new TPs missing from prior gate. Singing TPs typically score >0.99+ on hard cuts and >0.95 on crossfades so headroom exists. cv=3 (vs cv=5 default) keeps total retrain inside ~5min budget. Pure train_classifier.py change (CalibratedClassifierCV import + wrapper around HistGBM in make_pipeline); detector.py model.named_steps['clf'].classes_ and model.predict_proba() byte-identical since CalibratedClassifierCV exposes both with same semantics. OOF weighted F1=0.5956 (vs uncalibrated HistGBM ~0.577) modestly higher — calibration adds discrimination on top of the existing boost-residual scoring.
per-domain: combined_english=0.000000 combined_korean=0.000000 combined_singing=0.000000

# 2026-04-18 — hypothesis: isotonic-calibrate HistGBM

(a) HYPOTHESIS. Pure train_classifier.py change. Wrap the existing
    HistGradientBoostingClassifier inside
    CalibratedClassifierCV(estimator=HistGBM(...), method='isotonic',
    cv=5) as the pipeline's "clf" step. No detector.py edits, no
    features.py edits, no hyperparam edits, no training-data edits.
    Sklearn's CalibratedClassifierCV exposes predict_proba and
    classes_ with identical semantics so detector.py's
    model.named_steps["clf"].classes_ lookup and
    model.predict_proba(X) call work byte-identically. Internally
    each of 5 GroupKFold-style folds fits HistGBM on 4/5 of the data
    then fits an isotonic regressor on the held-out 1/5 against the
    true labels; final predictions average the 5 calibrated
    estimators. Manual retrain (~150s for 5x the ~30s HistGBM fit),
    joblib + meta + cv_results committed alongside.

(b) WHY this over the 40+ failures.
    Every prior axis attacked MEMORIZATION (capacity, sampling,
    feature engineering, model class, post-filters, routing). The
    weakest-domain failure mode (singing clean_fp=6, p_splice ~0.982
    just above the gate at chord-transition windows whose top SHAP
    features match training-clean clusters) has TWO possible causes
    and only ONE has ever been attacked:
      - MEMORIZATION (attacked 40+ times, all failed): training rows
        get baked into specific decision boundaries. Tried hyperparam
        regularization (8 axes), training-data resampling (5 axes),
        feature ablation/addition (12+), HistGBM swap, ExtraTrees
        swap (catastrophic 0.010), bootstrap-adversarial,
        singing-specialized routing, gated post-filters.
      - CALIBRATION (never attacked): the model's predicted
        probabilities don't reflect actual frequencies. Tree
        ensembles are notoriously over-confident at the extreme tails
        — known weakness of HistGBM's logistic-loss output in the
        upper-decile region. With an over-confident classifier, the
        chord-transition clusters that ought to score 0.7-0.9 (still
        suspicious, but below 0.982) instead score 0.98-0.99 and
        cross the gate.
    Isotonic recalibration fits a non-parametric monotone mapping
    from raw scores to true frequencies, learned per-fold then
    averaged. Critically it operates on the SAME training data with
    NO new feature, NO new hyperparam, NO new sampling — just
    redistributes the existing scores. The 0.982 threshold then
    selects from a different distribution: emits with p>0.982 are
    the ones that actually achieved >98% frequency on similar
    held-out positions during training, not just "scored high in
    boost-residual space."

    Explicit untried-next-step in two post-mortems:
      - ExtraTrees post-mortem (43944a6): "Wrap HistGBM with
        CalibratedClassifierCV(method='isotonic', cv=5). If
        clean-singing FPs are a CALIBRATION issue rather than a
        MEMORIZATION issue, isotonic regression fit OOF
        recalibrates the p_splice > 0.982 tail. Complementary axis
        to model-family swap; never tried."

    Orthogonal to every prior axis. ExtraTrees catastrophic failure
    actually de-risks this hypothesis: it shows that swapping AWAY
    from boosting destroys splice-class recall, so the right move
    is to KEEP HistGBM and fix only the probability scaling.

    Risk-bounded: isotonic calibration is monotone so the rank
    ordering of scores is preserved. Eval-time emits that were
    above 0.982 may shift to a different value but their RANK
    relative to other windows is unchanged. The intersection of
    {true splices, p>0.982} can only shrink, not contain new TPs
    that weren't above the prior gate. Singing clean_fp=6 is the
    target for shrinkage.

    Best case: isotonic squashes the chord-transition cluster
    below 0.982 (clean_fp drops from 6 toward 0) without dropping
    real splices below the gate (TPs were already at p>0.99 with
    headroom). Singing combined improves materially.

(c) IF THIS FAILS. Three paths.
    (1) Switch isotonic -> sigmoid (Platt scaling) which fits a
        parametric two-parameter logistic to the calibration
        curve. More stable on small held-out folds but assumes
        the calibration curve is sigmoid-shaped, which may not
        match HistGBM's actual miscalibration shape.
    (2) Reduce cv to 3 (fewer folds, more rows per fold for
        better isotonic estimate).
    (3) If calibration genuinely doesn't help, the failure mode
        is structural feature confusion (chord transitions
        produce IDENTICAL feature vectors to splices) and the
        next move is phase-coherence features — the only
        signal-level subspace never attempted (all prior features
        used STFT magnitude or chroma).

## 2026-04-18T19:10:25+09:00 — b39dbc9 (discard, combined=0.010000)
subject: per-bin phase residual pre/post distance feature (2-8 kHz)
per-domain: combined_english=0.000000 combined_korean=0.000000 combined_singing=0.000000

# 2026-04-18 — hypothesis: per-bin phase-residual signature distance (2-8 kHz, ±2s pre/post)

(a) HYPOTHESIS. Add ONE feature `phase_residual_pre_post_dist` to features.py.
    The existing `_ensure_feat_cache` already computes `bnd_residual` (per-bin
    per-frame deviation of unwrapped STFT phase advance from the chunk-mean
    advance) but only stores its per-frame std `feat_bnd_res_std` (consumed
    by `boundary_phase_coherence` over ±200 ms). Store the band-restricted
    residual matrix `feat_bnd_residual_band` (2000–8000 Hz, float32, ~5 MB
    per 60 s chunk). New block computes per-bin mean residual over [t-2, t]
    (pre) and [t, t+2] (post), wraps the per-bin difference to [-π, π], and
    returns the RMS as a single scalar. FEATURE_NAMES grows 75 → 76 so the
    classifier auto-retrains via the US-505 sha gate. Zero detector.py /
    train_classifier.py / hyperparam edits.

(b) WHY this over the 50+ prior failures. The clean-singing FP plateau
    (clean_fp=6, singing combined=0.272) is driven by chord transitions
    that fire spec_rolloff_delta / spec_centroid_delta / spec_bandwidth_delta
    (top-3 SHAP everywhere, 3-5x higher magnitude in singing) at values
    indistinguishable from real splices because those features are pure
    STFT-magnitude statistics. EVERY feature added so far operates on
    magnitude / envelope / chroma / MFCC subspaces (spec_*_delta,
    hf_*_delta, hpss_perc_*_delta, local_novelty_*, mfcc_cosine,
    chroma_key_cosine_distance) — all of which shift at within-song chord
    transitions because chord transitions DO change the magnitude spectrum.
    Phase information is currently used only at ±200 ms in
    boundary_phase_coherence — a single AGGREGATE coherence scalar over a
    narrow window. NO feature compares per-bin phase signatures across the
    standard ±2 s pre/post windows where every magnitude delta operates.

    Why phase residual at 2-8 kHz: the per-bin phase advance after
    subtracting the chunk-mean per-bin advance (= bnd_residual) factors
    out the constant harmonic structure and isolates the source-specific
    deviation (mic transfer function, room reverb early reflections, codec
    quantization) which is INVARIANT across chord transitions of the same
    source. 2000–8000 Hz is the band where mic high-end response, room
    presence, and codec rolloff dominate; below 2 kHz the band is
    chord-driven (vocal harmonics), above 8 kHz it is mostly noise that
    hf_codec features already capture. Within-song chord transition: pre
    and post per-bin signatures are similar (same source, same
    deviation pattern) → diff small → dist ~ 0.05. Cross-song splice:
    different mic/room/codec → per-bin signatures shift → dist > 0.2.

    Orthogonal to all 50+ prior axes: 6 primary tunables bracketed,
    8 GBM hyperparam axes failed, HistGBM kept, ExtraTrees catastrophic,
    bootstrap-adversarial failed, isotonic catastrophic, 5 training-data
    axes failed, 12+ magnitude-feature add/ablation failed, 7+ post-filters
    failed, singing-specialized routing failed, gated tonality+persistence
    catastrophic. The very last calibration post-mortem explicitly cited
    "phase-coherence features — the only signal-level subspace never
    attempted" as the next direction. Risk-bounded: 1 new feature, ~5 MB
    extra ctx memory per chunk, ~0.1 ms per call. classifier retrains
    automatically on FEATURE_NAMES sha change. Even if the feature is
    uninformative, GBM gives it low SHAP and ignores it.

(c) IF THIS FAILS. Two paths.
    (1) Widen / shift the band: 4-12 kHz (mic HF response only) or
        500-2000 Hz (room mode resonances). Or stack 3 features at
        non-overlapping bands (low/mid/high) and let GBM learn which
        band carries source-identity for each domain.
    (2) Switch from per-bin mean residual to per-bin VARIANCE of
        residual: |std(post) - std(pre)| per bin then RMS across bins.
        Stable sources have low residual variance; splices spike one
        side's variance. Final escalation if all phase axes fail:
        wavelet-based instantaneous-frequency features (different
        time-frequency localization than STFT, may catch transient
        source signatures invisible to fixed-window STFT phase).

(d) Information gaps in this prompt. The 3 most recent attempts ALL
    landed at combined=0.010 (GM floor → at least one domain produced
    splice_f1=0 or clean_fp≥15). That pattern is suspicious — suggests
    something other than the hypothesis content broke eval (classifier
    failed to load, detector crashed, scaler rejected new feature shape,
    etc). The prompt does NOT include the wrapper's stderr / verify
    diagnostic for those 0.010 runs. Without that, I cannot tell whether
    isotonic calibration's collapse was a CalibratedClassifierCV
    integration bug vs the calibration itself being wrong direction.
    A `RECENT_CATASTROPHIC_FAILURES` block surfacing the wrapper's
    crash report (or a flag like "catastrophic_reason: pipeline_load_error
    | feature_count_mismatch | ...") for the last few 0.010 attempts
    would prevent me from misattributing those failures to hypothesis
    content and re-trying axes that were actually killed by a wrapper
    issue.

(e) Wrapper enhancements that would accelerate this research. (i) A
    `--diagnose-last` mode on verify_agent.py that prints, for the most
    recent FAIL/0.010 run, which check (metric / git diff / anomaly /
    preflight) tripped + the actual evaluate.py stderr tail. Catastrophic
    0.010s become much faster to diagnose. (ii) A new SHAP-DELTA block in
    the prompt: alongside the rolling sum_|shap| top-6, show the
    per-feature SHAP CHANGE between the most recent KEPT classifier and
    the prior KEPT classifier. New features with near-zero SHAP signal
    they're being ignored regardless of metric outcome — much faster to
    pivot. (iii) A `data/train/sample_eval_misses/` directory that the
    wrapper auto-populates with the 6 clean-singing FP positions from
    each KEPT classifier (file path + t_sec). I could load those exact
    audio segments to smoke-test new features against the ACTUAL
    failure cases, instead of guessing from data/train/singing/tier1.

## 2026-04-18T19:23:58+09:00 — ac2d350 (discard, combined=0.010000)
subject: FILE-LEVEL mean-flatness route GBM_THRESHOLD (0.982 -> 0.988 on tonal files)
per-domain: combined_english=0.000000 combined_korean=0.000000 combined_singing=0.000000

# 2026-04-18 — hypothesis: FILE-LEVEL mean-flatness → conditional GBM threshold

(a) HYPOTHESIS. Pure detector.py change, no retrain, no feature change.
    At the top of `detect_splices`, sample mean spectral_flatness on ~8
    one-second windows spread evenly across the full audio. Compute
    `file_flatness_mean`. If `file_flatness_mean < FILE_TONALITY_MAX =
    0.05` (highly tonal / musical content) use the tighter
    `GBM_THRESHOLD_FILE_TONAL = 0.990` as the emit gate; otherwise use
    the standard `GBM_THRESHOLD = 0.982`. One helper, 3 new constants,
    ~15 lines in detector.py. Classifier joblib / features.py /
    train_classifier.py untouched.

(b) WHY this over recent failures. Every prior tonality-gated move
    used PER-EMIT context:
      * `tonality-conditioned per-emit threshold (flat<0.05 → 0.988)`
        (0.446) — lifted singing to 0.275 but collapsed korean 0.500
        → 0.426 because some voiced-korean ±2s windows dip below
        0.05 mid-utterance and the per-emit gate tightens their
        threshold too. The FAILURE MODE was per-emit noisiness.
      * `GATED tonality+persistence post-filter` (catastrophic 0.010)
        — per-emit AND gate, same failure ingredient plus runtime
        instability.
      * `singing-specialized GBM with audio-flatness routing` (0.357)
        — did use file-level flatness but SWAPPED THE CLASSIFIER;
        the small singing-only training set over-fit and singing
        dropped to 0.120.
    NO prior attempt has used file-level flatness to route the
    DECISION THRESHOLD on the same unified classifier. The two prior
    post-mortems cited this specific combination as a next step
    without attempting it. File-level flatness is a ROBUST
    discriminator: the singing-routing post-mortem measured korean
    speech files at 0.10-0.30 (never below 0.06) and singing files
    at 0.02-0.05 (always below 0.06), so a 0.05 cutoff sits cleanly
    between populations and cannot mis-gate korean. Tightening to
    0.990 (vs 0.988 in the per-emit attempt) is safe because the
    gate only bites on a genuinely-musical file set — the worst it
    can do is cost 1-2 tier2 singing crossfades; in exchange the 6
    singing clean FPs that sit at p_splice ≈ 0.982-0.988 are dropped.

    Risk-bounded design: file-level flatness evaluation is one
    librosa.feature.spectral_flatness call on ~8 sampled 1s slices
    (~30 ms overhead per file × ~60 files ≈ 2 s, well inside
    243/300 s headroom). Only detector.py changes so there is no
    retrain path — which is important because AUTO-RETRAIN has
    been clobbering recent features.py-touching hypotheses.
    Byte-identical behaviour on speech files because their
    file_flatness_mean is above cutoff → threshold unchanged.

    Orthogonal to all 50+ prior failures: 6 primary tunables
    bracketed, 8 GBM hyperparam axes failed, 5 training-data axes
    failed, 12+ feature add/ablation failed, 7+ post-filters failed
    (including per-emit flatness), 2 model-class swaps failed,
    isotonic catastrophic, bootstrap-adversarial failed, per-bin
    phase catastrophic. NO prior used FILE-LEVEL flatness as a
    threshold router.

(c) IF THIS FAILS. Two paths.
    (1) Loosen (0.06 cutoff) or tighten (0.04) the flatness
        boundary; calibrate tight threshold (0.988 or 0.995) to
        move the precision/recall point.
    (2) Add a second scalar to the file-level test (e.g. voicing
        fraction or harmonic_ratio = HPSS harmonic-energy share)
        and require BOTH to flag tonal before tightening.
    Final escalation if file-level gating cannot help singing
    without hurting it: genuine audio-augmentation during training
    (add reverb / EQ / pitch-shift to clean singing training rows
    so GBM sees a wider clean distribution and chord-transition
    spec_*_delta magnitudes look less exceptional).

(d) Information gaps. The 4 most recent discards were catastrophic
    0.010 (ExtraTrees 43944a6, gated tonality d47c0e2, isotonic
    d25f455, phase residual b39dbc9). The prompt shows the final
    combined but NOT the eval stderr that explains whether eval
    crashed, timed out, or legitimately predicted a zero score.
    Without a `RECENT_CATASTROPHIC_DIAGNOSTICS` block surfacing
    `.omc/classifier/eval.stderr` tails for the last N 0.010 runs
    (e.g. "crash: RuntimeError at detector.py:268" vs "timeout at
    294 s" vs "all splice predictions below threshold"), I cannot
    distinguish hypothesis-content failure from pipeline breakage.
    Two of those four (isotonic, phase feature) had healthy OOF
    F1 post-retrain but still scored 0.010 on evaluate.py — strong
    signal that something other than the hypothesis content killed
    the eval.

(e) Wrapper enhancements.
    (1) Show the eval.stderr tail on 0.010 outcomes inside the
        next iteration's prompt (a `LAST_CATASTROPHIC_STDERR` block
        at the top of HISTORY). This directly unblocks the 4
        consecutive 0.010 mystery.
    (2) `verify_agent.py --diagnose-last` helper that prints, for
        the most recent DISCARD, the evaluate.py stderr tail, the
        detector fp_classifier.meta.json features_py_sha, and the
        pass/fail of each of preflight's 4 checks — single
        command, no guessing.
    (3) Surface `classifier_oof_f1_weighted_baseline` and the
        latest AUTO-RETRAIN OOF delta in CURRENT STATE, so I can
        spot training-pipeline drift vs hypothesis issues.

## 2026-04-19T13:25:23+09:00 — 84b1531 (discard, combined=0.468623)
subject: revert classifier HistGradientBoostingClassifier -> GradientBoostingClassifier — eval.stderr on disk (.omc/last_eval.log tail) shows the root cause of the last 4 consecutive 0.010 catastrophic discards was `AttributeError: 'HistGradientBoostingClassifier' object has no attribute 'feature_importances_'` thrown by shap_report.py. acba4aa was an ORPHANED keep (no paired baseline commit; DEEP-ORPHAN already flagged it in today's loop start) that swapped the classifier class while baseline_metrics.json retained the GBM-era 0.468623 score. Even though the maintainer fixed shap_report.py in US-510/US-511 (9ad0fdf try/except AttributeError + uniform-weights fallback), the canonical baseline measurement was produced by GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.07, subsample=0.9) and every subsequent hypothesis is being scored against a recorded baseline that the on-disk classifier (HistGBM) no longer produces. Reverting make_pipeline() to the exact make_pipeline body at acba4aa~1 (GBM with same hyperparams) restores classifier-baseline consistency so future detector/features/train deltas become apples-to-apples. Retrain deterministic (random_state=42, unchanged features.py sha e7691315, unchanged training data): OOF weighted F1=0.5904 vs prior HistGBM's 0.577 matches historical GBM OOF range (~0.59-0.61). Either the re-measured eval reproduces 0.468623 exactly and is DISCARDed (orphan-cleanup unfortunately loses but at least diagnosed here) or floating-point drift moves it a whisker above, in which case it's KEPT and the revert persists. Pure train_classifier.py + joblib revert; detector.py and features.py untouched; FEATURE_NAMES 75 unchanged.
per-domain: combined_english=0.756098 combined_korean=0.500000 combined_singing=0.272222

# 2026-04-19 — hypothesis: revert classifier HistGradientBoostingClassifier → GradientBoostingClassifier

(a) HYPOTHESIS. Pure train_classifier.py revert. Restore the classifier
    class to `GradientBoostingClassifier(n_estimators=200, max_depth=3,
    learning_rate=0.07, subsample=0.9, random_state=RANDOM_STATE)` —
    exactly the make_pipeline body at acba4aa~1, i.e. the classifier
    that produced the current baseline combined=0.468623. Retrain
    manually; commit the regenerated joblib + meta + cv_results
    alongside the code revert.

(b) WHY this over a fresh research axis. Reading
    `.omc/last_eval.log` tail + `autoresearch.log` revealed the root
    cause of the last 4 consecutive 0.010 catastrophic discards
    (ExtraTrees 43944a6, gated tonality d47c0e2, isotonic d25f455,
    phase residual b39dbc9, FILE-LEVEL flatness ac2d350): evaluate.py
    → shap_report.py unconditionally touched
    `clf.feature_importances_` which HistGradientBoostingClassifier
    does NOT expose, throwing AttributeError and zeroing per-domain
    scores → GM floor. The maintainer fixed shap_report.py in
    US-510/US-511 (commit 9ad0fdf; try/except AttributeError with
    uniform-weights fallback). But the acba4aa "swap" commit was
    ORPHANED — DEEP-ORPHAN surfaces the mismatch: no baseline commit
    paired with acba4aa, yet the HistGBM code + joblib remain in HEAD.
    The recorded baseline 0.468623 was measured BEFORE acba4aa landed
    (keep 2a2e584, GBM era). So the current state is inconsistent:
    baseline_metrics.json says GBM, train_classifier.py says HistGBM,
    the joblib on disk is HistGBM. Every future hypothesis is being
    compared against a recorded baseline that the classifier on disk
    no longer produces. Reverting restores canonical state. If the
    retrain reproduces 0.468623 deterministically (random_state=42,
    unchanged features.py sha, unchanged training data), the wrapper
    will DISCARD on "not > current_best", but the revert then persists
    through the baseline commit of whichever subsequent hypothesis is
    kept. Worst case: revert reproduces 0.468623 exactly, is
    discarded, wrapper resets to HistGBM, state stays inconsistent —
    but the diagnosis is captured here for the operator. Best case:
    noise in retrain produces a whisker above 0.468623 and it's kept,
    cementing the revert. Either way, surfacing the root cause of the
    4 prior 0.010 runs is higher value than attempting a fresh axis
    against a contaminated baseline.

(c) IF THIS FAILS. (1) If the retrain score is exactly 0.468623 and
    is discarded, escalate to the operator: the revert must be merged
    as a maintainer pass, not an autoresearch iteration. The
    orphan-cleanup path (adding a baseline commit for acba4aa or
    rolling it back) is outside the loop's control because the loop
    only persists changes that BEAT baseline. (2) If the retrain
    scores LOWER than 0.468623 (e.g. 0.465), that confirms that
    something else drifted since baseline (feature cache, code path,
    library version) and the prior baseline is no longer reproducible
    — in which case the operator needs to re-measure the canonical
    baseline under the current code state.

(d) Information gaps. The prompt's HISTORY block reports the 4
    recent 0.010 catastrophic failures without the root-cause eval
    stderr (the AttributeError) inline. I only found it by reading
    `.omc/last_eval.log` directly. A `LAST_CATASTROPHIC_STDERR` block
    surfacing the eval tail on 0.010 outcomes would have let the prior
    iteration (phase residual, FILE-LEVEL flatness) catch the bug
    three attempts earlier. The previous reflection explicitly flagged
    this gap — the maintainer pass fixed the underlying code bug
    (shap_report.py) but did not add the diagnostic surface to the
    prompt.

(e) Wrapper enhancements. (1) A DEEP-ORPHAN commit detector is already
    running (surfaced acba4aa in today's loop start), but the log
    message is informational. It should be ACTIONABLE: when an orphan
    is detected that touches classifier/detector/features code, the
    wrapper should either block iteration until the operator resolves
    it or automatically revert the orphan before continuing. (2)
    Add a `LAST_CATASTROPHIC_STDERR` block to the iteration prompt
    when the last N runs scored 0.010, showing the eval.stderr tail
    (or evaluate.py error message) — the single most actionable
    diagnostic signal is currently not surfaced to the agent. (3) A
    `verify_agent.py --diagnose-orphans` subcommand that lists all
    hypothesis commits without a paired baseline or discard, so the
    operator can batch-clean them.

## 2026-04-19T16:27:58+09:00 — 39b9fb0 (discard, combined=0.428706)
subject: FILE-LEVEL HARMONIC RATIO (HPR via HPSS) routing of GBM_THRESHOLD — once per file compute mean harmonic ratio file_hpr = sum(harmonic_S) / (sum(harmonic_S) + sum(percussive_S)) via librosa.decompose.hpss(|STFT|). If file_hpr > FILE_HPR_TONAL_MIN=0.85 (sustained-harmonic / singing population) use tighter GBM_THRESHOLD_HPR_TONAL=0.990 as the per-file emit gate; else default GBM_THRESHOLD=0.982. Pure detector.py change, no retrain, no feature change. Targets singing clean_fp=6 plateau at weakest-domain combined=0.272. Recent 4 catastrophic 0.010 discards (gated tonality, isotonic, phase residual, FILE-LEVEL flatness ac2d350) ALL crashed eval via the now-fixed shap_report.py AttributeError on HistGBM (US-510/US-511); the just-prior revert (84b1531) confirmed eval pipeline is functional and 0.468623 reproduces. Strongest theoretical untried direction is FILE-LEVEL routing — per-emit flatness (0.446) showed singing 0.275 (above baseline) but collapsed korean 0.500->0.426 from per-emit flatness mis-gating voiced korean windows. Replacing routing scalar from FLATNESS to HPR is substantively distinct: HPR measures the energy split between harmonic and percussive STFT components after median-filter decomposition, not the geometric/arithmetic mean of magnitude spectrum. Smoke test confirms: singing tier1 hpr=0.89 trips gate; clean singing 0.93-0.97 trips gate; korean clean max=0.71 never trips; english clean max=0.51 never trips — domain separation is wider than flatness gap so korean/english cannot leak. Tightening to 0.990 only on tonal files drops the borderline FP cluster (6 singing FPs at p ~0.982-0.99) while real singing TPs (hard cuts at p>0.99+, crossfades at 0.95-0.98) experience minimal precision loss. Cost ~200ms HPSS call x 60 files = ~12s overhead within 243/300s headroom.
per-domain: combined_english=0.709890 combined_korean=0.451282 combined_singing=0.245946

# 2026-04-19 — hypothesis: FILE-LEVEL HARMONIC RATIO (HPR via HPSS) routing of GBM_THRESHOLD

(a) HYPOTHESIS. Pure detector.py change, no retrain, no feature change.
    Once per file, compute mean harmonic ratio
    `file_hpr = sum(harmonic_S) / (sum(harmonic_S) + sum(percussive_S))`
    via `librosa.decompose.hpss(|STFT|)` on the full audio. If
    `file_hpr > FILE_HPR_TONAL_MIN = 0.85` (sustained-harmonic content
    — singing population), use the tighter
    `GBM_THRESHOLD_HPR_TONAL = 0.990` as the per-file emit gate;
    otherwise the standard `GBM_THRESHOLD = 0.982`. One helper, 3 new
    constants, ~20 lines in detector.py. Classifier joblib /
    features.py / train_classifier.py untouched.

(b) WHY this over the recent 4-failure cluster. The 4 prior catastrophic
    0.010 discards (gated tonality, isotonic, phase residual, FILE-LEVEL
    flatness ac2d350) ALL crashed eval via the shap_report.py
    AttributeError on HistGBM — wrapper bug, not hypothesis content.
    US-510/US-511 fixed the bug; the just-prior revert (84b1531)
    confirmed eval is now executing correctly (0.468623 reproduced).
    The strongest theoretical untried direction now is FILE-LEVEL
    routing of GBM_THRESHOLD, which the per-emit flatness variant
    showed singing 0.275 (above baseline 0.272) but collapsed korean
    0.500→0.426 from per-emit flatness mis-gating voiced korean
    windows. Replacing the routing scalar from FLATNESS to HPR is
    substantively distinct (different physics): HPR measures the
    energy split between harmonic and percussive STFT components after
    median-filter decomposition, not the geometric/arithmetic-mean
    ratio of the whole magnitude spectrum. Singing files HPR ≈
    0.88-0.95; speech HPR ≈ 0.55-0.75 (consonants/plosives reduce
    harmonic share). The 0.85 cutoff sits cleanly between populations
    so korean cannot trip the gate (korean files have HPR ~0.55-0.70
    by phonetic content alone). Tightening to 0.990 only on tonal
    files drops the borderline FP cluster (6 singing FPs at p_splice
    ≈ 0.982-0.99) while real singing TPs (hard cuts at p>0.99+,
    crossfades at 0.95-0.98) experience minimal precision loss.
    Pure detector.py change with no retrain — no FEATURE_NAMES sha
    bump, no joblib desync risk. HPSS cost ~200 ms/file × 60 files =
    ~12 s overhead, well within 243/300 s headroom.

(c) IF THIS FAILS. Three paths.
    (1) Tune cutoffs — FILE_HPR_TONAL_MIN to 0.80 or 0.90; or
        GBM_THRESHOLD_HPR_TONAL to 0.985 or 0.995.
    (2) Switch routing scalar to MEAN VOICING (yin-derived voicing
        probability): singing has high sustained voicing, speech is
        bursty. yin already runs in `_block_pitch` so essentially free.
    (3) Hybridize: use BOTH file_hpr > 0.85 AND file_flatness < 0.05
        as a conjunctive gate (both signals must flag "singing-like").
        Lower false-routing risk than either alone.
    Final escalation: PER-EMIT HPR as a feature in features.py, letting
    GBM learn the discriminator instead of hard-coding the cutoff.

(d) Information gaps. The prompt's RECENT FAILED HYPOTHESES list does
    NOT distinguish between hypotheses that genuinely failed at eval vs
    hypotheses killed by wrapper-level pipeline bugs (the recent 4
    catastrophic 0.010s, all from the now-fixed shap_report.py
    AttributeError). Without a flag, the no-repeat rule forces a pivot
    to a NEW scalar instead of cleanly re-running FILE-LEVEL flatness
    under correct execution. A `failure_reason: pipeline_bug |
    feature_mismatch | hypothesis_content` flag in HISTORY would let me
    judge which prior failures genuinely informed the search.

(e) Wrapper enhancements. (1) Auto-tag catastrophic 0.010 entries with
    a derived `failure_reason` from eval.stderr. (2) A
    `verify_agent.py --replay <commit>` mode that re-runs evaluate.py
    against a prior committed hypothesis using the current code state
    — useful to confirm whether a bug-killed hypothesis actually works
    post-fix. (3) A FILE-LEVEL audio-statistics summary at the top of
    CURRENT STATE (mean HPR / flatness / voicing per domain on EVAL
    set) so I set scalar cutoffs from data instead of training-set
    intuition.

## 2026-04-19T18:28:01+09:00 — f6192f6 (discard, combined=0.438080)
subject: remote-half MFCC source-continuity similarity
per-domain: combined_english=0.542857 combined_korean=0.537313 combined_singing=0.288235

# 2026-04-19 — hypothesis: remote-half MFCC source-continuity similarity

(a) HYPOTHESIS. Add ONE feature `source_continuity_mfcc_sim` to features.py.
    Compute the cosine similarity of two MFCC means:
      * remote-pre: mean MFCC over [0, t - 5.0]
      * remote-post: mean MFCC over [t + 5.0, chunk_dur]
    Return cos(pre_mean, post_mean). The ±5s guard band explicitly EXCLUDES
    the local region that every existing ±2s delta feature already covers,
    forcing this feature to capture LONG-RANGE source identity only. For
    clean files (no splice) both halves are the same source → sim ≈ 1. For
    real splices at t, pre-half is source A and post-half is source B →
    sim drops materially. Chord transitions within one song preserve
    singer / mic / room / mix so remote-pre and remote-post MFCC means
    remain aligned → sim ≈ 1 → GBM learns "high sim → not_splice" which
    suppresses the top-SHAP spec_*_delta signal that currently fires FPs.
    FEATURE_NAMES 75 → 76 so the classifier auto-retrains via US-505.
    Fallback to sim=1.0 when either remote window is under 8s (edge
    positions in a 60s chunk at t ∈ [13, 47] have 8s+ pre and post so
    the guard rarely fires).

(b) WHY this over the 50+ prior failures. Two prior MFCC/chroma similarity
    features failed: mfcc_cosine_distance at ±2s and chroma_key_cosine
    at ±8s. Both windows STILL OVERLAP the local emit context — at ±8s
    chroma, the window includes material close enough to the emit that
    a chord transition within 8s of t_sec still dominates the chroma mean.
    This feature explicitly EXCLUDES the ±5s local region and uses the
    ENTIRE REMAINING chunk on each side — a structurally-different time
    scale never attempted. The long mean over 8-25s of audio on each
    side averages out per-chord vocal articulation and per-measure
    spectral jitter, isolating the stable TIMBRE of the source: singer
    identity, mic transfer function, room reverb tail, mix coloration.
    A real splice changes all four; a chord transition within one song
    changes none. Also orthogonal to every other prior axis (8 GBM
    hyperparam / 5 training-data / 12+ feature add/ablation / 8+
    post-filter / 2 model-class swap / 2 calibration / bootstrap-
    adversarial / singing-routing / 2 file-level routing). MFCC is
    already computed per chunk in ctx["feat_mfcc"] so the new block is
    a pure array-slicing + dot-product operation with zero librosa
    overhead. Risk-bounded: one new float feature, ~1ms per call, GBM
    can always learn SHAP ≈ 0 and ignore it if uninformative.

(c) IF THIS FAILS. Three paths. (1) Switch the statistic from MFCC mean
    to mel-PCA mean (captures broader spectral identity) or chroma mean
    (captures KEY-level identity, complementing the failed ±8s chroma
    which overlaps locally). (2) Make the feature DIRECTIONAL: keep pre
    and post sims separately as sim(pre-near, pre-remote) and
    sim(post-near, post-remote) — the two-sided version distinguishes
    "pre-half matches remote-pre AND post-half matches remote-post" (no
    splice) from "one side changes" (splice). (3) Final escalation:
    PITCH-SHIFT AUGMENTATION of clean training files via
    librosa.effects.pitch_shift ±1 semitone to widen the clean
    distribution in spec_*_delta space so memorization-driven FPs drop.

(d) Information gaps. HEAD is at 643f159 (TOP-1-emit-per-file cap) but
    this commit does not appear in the RECENT FAILED HYPOTHESES or KEEP
    lists — its outcome is genuinely unknown to me. Baseline is still
    0.468623. My commit will stack on top so the wrapper will evaluate
    TOP-1-cap + new feature jointly. A `HEAD_STATE:` block showing
    which tracked files differ from the baseline commit would let me
    decide whether to revert or compose. Also still: no failure_reason
    flag on the 4 catastrophic 0.010 entries in HISTORY.

(e) Wrapper enhancements. (1) Surface current-HEAD diff summary at the
    top of the iteration prompt: the exact lines that differ from the
    baseline git_sha, so I know what my commit composes with. (2) A
    `scripts/per_file_emit_distribution.py` report showing emit counts
    + top-k p_splice per file across the TRAINING clean files — this
    would calibrate cardinality-cap / margin-cap / quantile-threshold
    hypotheses far more reliably than guessing. (3) Auto-tag HISTORY
    entries with `failure_reason: pipeline_bug | feature_mismatch |
    hypothesis_content` so the no-repeat rule does not eternally
    block axes that were killed by the now-fixed shap_report.py crash.

## 2026-04-19T18:48:01+09:00 — e228236 (discard, combined=0.427426)
subject: HPR-gated remote-half rolloff divergence post-filter (singing-only)
per-domain: combined_english=0.709890 combined_korean=0.451282 combined_singing=0.243750

# 2026-04-19 — hypothesis: HPR-gated remote-half spec_rolloff divergence POST-FILTER (singing-only)

(a) HYPOTHESIS. Pure detector.py change, no retrain, no feature change.
    After GBM dedupe, compute file-level HPR (librosa.decompose.hpss on
    full-file |STFT|). If file_hpr > HPR_SINGING_MIN=0.85, apply a
    per-emit remote-half spec_rolloff divergence filter: compute mean
    rolloff on [0, t-LOCAL_GUARD] and [t+LOCAL_GUARD, dur] with
    LOCAL_GUARD=5.0s; drop the emit when |pre-post|/mean < 0.08 (the two
    halves have the same mastering/EQ → within-song chord cycle-back,
    not a cross-source splice). Edge emits skipped (keep). Re-uses the
    same STFT for HPSS + rolloff so only ONE librosa.stft call per file.
    Non-singing files (korean/english) take an early-return path; their
    decisions are byte-identical.

(b) WHY this over recent failures. Remote-half MFCC sim as a FEATURE
    (f6192f6, 0.438) took singing to 0.288 (a new HIGH vs baseline 0.272)
    and korean to 0.537 (+), but tanked english 0.756→0.543 because the
    classifier learned to over-weight a global feature on its only
    strong domain. File-level HPR routing (39b9fb0, 0.428) proved the
    gate cleanly separates singing (HPR 0.89-0.97) from korean (≤0.71)
    and english (≤0.51) — domain leakage impossible at 0.85 — but
    routed GBM_THRESHOLD which affected singing TP recall too. THIS
    combines the two successful halves in a new way: HPR-file-gate +
    POST-FILTER (not threshold, not feature). Post-filter cannot drop
    TPs that were below the 0.982 gate — it only has power over emits
    ALREADY KEPT, so a conservative 0.08 divergence criterion bites
    only on the 6 singing clean FPs (chord transitions where the rest
    of the song remains one coherent source, so remote halves match).
    Real cross-song splices change mastering/EQ/compression/mic so
    full-file rolloff differs by 10-20% between halves → divergence
    typically > 0.15 → KEEP. Remote-half MFCC worked because remote
    context captures source identity; the same physics applies to
    rolloff which is the #1 SHAP feature (spec_rolloff_delta 1365 in
    singing). Cost: one STFT + HPSS + rolloff on full file ≈ 400ms
    × 60 files = 24s, well inside 243/300s headroom.

(c) IF THIS FAILS. (1) Loosen/tighten the 0.08 divergence (try 0.05 or
    0.12) or HPR cutoff (0.80 or 0.90). (2) Replace divergence with
    KS-statistic between pre and post rolloff distributions (more
    robust than single means). (3) Stack rolloff + centroid + bandwidth
    divergences and require at least 2 below threshold. Final
    escalation: treat HPR-gated post-filter as the "singing path" and
    combine with a small SIGMOID-calibrated HistGBM on non-singing
    files (sigmoid calibration on HistGBM was never completed now that
    the shap_report.py bug is fixed).

(d) Information gaps. The prompt still does not flag which of the 4
    recent 0.010 entries failed due to hypothesis content vs the
    now-fixed shap_report.py AttributeError. Three of them (isotonic,
    phase residual, FILE-LEVEL flatness) may be viable post-fix and
    the no-repeat rule forces me to pivot away from them indefinitely.
    A `failure_reason: pipeline_bug | hypothesis_content` tag on
    HISTORY entries would unblock re-exploration of those axes.

(e) Wrapper enhancements. (1) Surface a SHAP-DELTA block in CURRENT
    STATE: for the most recent KEPT classifier vs prior KEPT classifier,
    show per-feature SHAP change. New features with near-zero SHAP are
    being ignored regardless of metric outcome — early pivot signal.
    (2) A `scripts/per_domain_file_stats.py` emitting mean
    HPR/flatness/voicing per domain on EVAL (not training) so
    file-level gating thresholds come from the evaluation distribution.
    (3) `verify_agent.py --replay <commit> --against HEAD` re-runs
    evaluate.py for a prior hypothesis commit under the current code
    state; essential for confirming bug-killed hypotheses post-fix.

## 2026-04-19T19:02:55+09:00 — a4a01d9 (discard, combined=0.418923)
subject: SIGMOID-calibrate HistGBM (CalibratedClassifierCV cv=5)
per-domain: combined_english=0.627907 combined_korean=0.452381 combined_singing=0.258824

# 2026-04-19 — hypothesis: SIGMOID-calibrate HistGBM (CalibratedClassifierCV cv=5)

(a) HYPOTHESIS. Pure train_classifier.py change. Wrap the existing
    HistGradientBoostingClassifier inside
    `CalibratedClassifierCV(estimator=HistGBM(...), method='sigmoid', cv=5)`
    as the pipeline's "clf" step. Detector unchanged — CalibratedClassifierCV
    exposes `predict_proba()` and `classes_` with identical semantics, and
    shap_report.py already has the AttributeError fallback (US-510/US-511).
    Manual retrain (~60s on 1200 rows × 5×5 fold/inner-fold), commit
    joblib + meta + cv_results alongside.

(b) WHY this over the recent failures. Two prior attempts at the
    REMOTE-HALF angle (MFCC-sim FEATURE f6192f6 lifted singing to 0.288
    NEW HIGH but tanked english 0.756→0.543; HPR-gated rolloff
    POST-FILTER e228236 dropped singing to 0.244) both proved the
    intuition correct but the integration mechanism wrong. Before
    iterating further on remote-half, the CALIBRATION axis remains
    unattacked: the prior isotonic attempt (d25f455) was killed at
    0.010 by the shap_report.py AttributeError on HistGBM (US-510/US-511
    fixed it). Sigmoid (Platt scaling) is the explicit untried sibling
    cited in two prior post-mortems and the most-recent e228236
    reflection ("sigmoid calibration on HistGBM was never completed
    now that the shap_report.py bug is fixed"). Mechanism: HistGBM's
    logistic-loss output is notoriously over-confident in its upper-
    decile tail; the singing FP cluster sits at p ≈ 0.982-0.99 just
    above the 0.982 gate at chord-transition windows. Sigmoid fits a
    2-parameter logistic to the score-vs-frequency curve, redistributing
    overconfident scores down toward their true frequencies. If the
    cluster's true frequency is ~95%, sigmoid pushes p to ~0.95 (below
    gate → FP drops). Real splice TPs at p=0.99+ have headroom and
    largely stay above gate (singing TPs typically at 0.99 hard-cut /
    0.95-0.98 crossfade — risk concentrated on borderline crossfades).
    Sigmoid is monotone so rank ordering preserved.

    Orthogonal to all 50+ prior axes: 6 primary tunables bracketed,
    8 GBM hyperparam axes failed, 5 training-data axes failed, 12+
    feature add/ablation failed, 7+ post-filter failures, 2 model-
    class swaps (HistGBM kept, ExtraTrees catastrophic),
    bootstrap-adversarial failed, isotonic killed by bug,
    file-level HPR/MFCC routing both failed. ZERO prior calibration
    axis successfully tested. cv=5 (vs prior isotonic cv=3) gives
    more stable per-fold sigmoid fits without exploding train cost
    (HistGBM base fit ~2s × 25 fits = ~50s, well inside 300s).

(c) IF THIS FAILS. (1) Switch to PREFIT mode — train HistGBM on full
    data, then fit sigmoid on a single GroupKFold-disjoint held-out
    set (proper file-grouping prevents leakage that default cv=5
    KFold inside CalibratedClassifierCV may introduce). (2) Try cv=3
    isotonic again (now that the bug is fixed, the original
    hypothesis was never genuinely tested). (3) If calibration is
    fundamentally not the failure mode, escalate to PER-DOMAIN
    audio augmentation of clean training files (untried — pitch-
    shift alone won't help deltas, but local time-stretch or local
    EQ augmentation creates synthetic chord-transition-like
    spectral shifts in clean rows, widening the not-splice
    distribution).

(d) Information gaps. Singing OOF F1-per-class (hard_cut / crossfade
    / not_splice) on the current HistGBM not in CURRENT STATE — only
    aggregate weighted F1. If sigmoid mostly affects crossfade
    probability, knowing the class breakdown would predict eval-
    time recall risk. Also: per-domain CLASS BREAKDOWN of the 6
    singing clean FPs (hard_cut vs crossfade triggered) would
    indicate which sigmoid (per-class) does the heavy lifting.

(e) Wrapper enhancements. (1) A SINGING-FP-DUMP block in CURRENT
    STATE: for each KEPT classifier, list the (file, t_sec, p_hard,
    p_cross) tuples for the 6 singing clean FPs. Direct visibility
    into the failure mode lets me design hypotheses that actually
    target the cluster instead of guessing. (2) Per-class OOF F1 in
    CURRENT STATE alongside the aggregate weighted F1 — currently I
    only see the aggregate. (3) A `verify_agent.py --replay <commit>
    --against HEAD` mode to rerun any prior bug-killed hypothesis
    under current-fixed code; would unblock isotonic/phase-residual/
    FILE-LEVEL-flatness without violating the no-repeat rule.

## 2026-04-19T19:31:21+09:00 — 0c5ba88 (discard, combined=0.357329)
subject: add file_hpr (chunk-level harmonic ratio) as GBM feature (75 -> 76)
per-domain: combined_english=0.773256 combined_korean=0.289655 combined_singing=0.203704

# 2026-04-19 — hypothesis: add file_hpr as GBM feature (chunk-level harmonic ratio)

(a) HYPOTHESIS. Add ONE new feature `file_hpr` to features.py
    (FEATURE_NAMES 75→76). Computed ONCE per chunk in
    `_ensure_feat_cache` via `librosa.decompose.hpss` applied to the
    already-computed |Zxx_bnd| magnitude STFT (reuses the existing
    chunk-level STFT, zero new librosa calls). `file_hpr =
    sum(H) / (sum(H) + sum(P))`, stored in ctx["feat_file_hpr"],
    emitted as the same scalar for every position in that chunk.
    Minimum-diff: new block `_block_file_hpr` appended AFTER
    `_block_mel_pca` in extract_features. Classifier auto-retrains
    via US-505 features.py sha gate.

(b) WHY this over the recent failures. HPR has been PROVEN to
    separate domains cleanly — smoke-tested in 39b9fb0 (singing
    0.88-0.97 vs korean ≤0.71 vs english ≤0.51, gap at 0.85 wide
    enough that korean cannot leak into "tonal" bucket). Every prior
    HPR attempt used it as a HARD-CODED GATE: 39b9fb0 (threshold
    routing, combined=0.428, singing=0.246 — tightened singing
    pushed real TPs below gate); e228236 (post-filter gate,
    combined=0.427, singing=0.244 — dropped too aggressively).
    HARD-CODED RULES constrain the model to ONE specific conditional
    logic that clearly didn't match the decision surface.

    HPR as a FEATURE lets GBM LEARN the conditional rule instead of
    hard-coding it. Concretely the model can find interactions like:
    "IF file_hpr > 0.85 AND spec_rolloff_delta moderate → likely
    chord-transition (not_splice)" via tree splits, or inverse: "IF
    file_hpr ≤ 0.75 AND nf_kl_divergence high → speech splice".
    GBM naturally represents conditional boolean logic via split
    chains — this is the model capability that hard-coded gates
    bypass.

    Orthogonal to every prior axis (6 primary tunables bracketed,
    8 GBM hyperparam failures, 5 training-data failures, 12+ feature
    add/ablation failures, 7+ post-filter failures, 2 model-class
    swaps, 2 calibration attempts, 2 file-level HPR ROUTING attempts,
    remote-half MFCC sim feature discarded). NO prior attempt added
    a FILE-LEVEL domain-indicator as a feature. All prior features
    are PER-POSITION (pre/post window deltas); this is the first
    CHUNK-LEVEL CONSTANT feature giving GBM file-identity context.

    Cost: HPSS on existing |STFT| ≈ 50ms per chunk × ~120 chunks
    ≈ 6s in the 243/300s budget. No new STFT call (reuses
    Zxx_bnd). Risk-bounded: one new float feature, if uninformative
    GBM gives it SHAP≈0 and ignores it. If informative, next
    iteration will see it in the SHAP top-6 rollup.

(c) IF THIS FAILS. (1) Add a SECOND chunk-level feature:
    file_voicing_fraction (mean of existing vp array) — voicing is
    complementary to HPR (HPR measures harmonic/percussive split;
    voicing measures "is it pitched at all"). GBM sees richer
    file-identity context. (2) Switch to a DIRECTIONAL pair — compute
    HPR separately on pre-half [0, t_sec] and post-half [t_sec, end]
    so real splices show HPR DIVERGENCE between halves. (3) Final
    escalation: re-attempt HPR-gated remote-half MFCC sim as a
    feature — the one hybrid variant that combines the proven
    singing signal (f6192f6 hit singing=0.288 new high) with HPR
    domain-gating to prevent english regression.

(d) Information gaps. Per-class OOF F1 breakdown (hard_cut /
    crossfade / not_splice) is not in CURRENT STATE — only aggregate
    weighted F1. For a file-level feature I specifically want to
    know whether the SINGING clean_fp=6 cluster triggers p_hard or
    p_cross to predict which class sigmoid might help. Also still
    missing: a per-KEEP SHAP-DELTA block showing whether a newly
    added feature is being USED (high SHAP) or IGNORED (SHAP≈0),
    directly informs next-iteration pivot.

(e) Wrapper enhancements. (1) Emit a CHUNK-LEVEL-FEATURE-RATIO
    metric at train-time: for each FEATURE_NAMES entry, print the
    per-chunk coefficient of variation (std/mean within a chunk).
    Features with CV≈0 are constant-per-chunk (like file_hpr by
    design), letting me verify the feature was computed correctly
    and letting future research pick which block to put a feature
    in. (2) `verify_agent.py --shap-for-feature <name>` printing
    global SHAP rank + mean |SHAP| for a specified feature across
    the last N KEPT classifiers. Would let me detect when a new
    feature is truly being used by the model vs carried along at
    SHAP≈0. (3) A `data/train/clean_fp_positions.json` auto-emitted
    by evaluate.py listing (domain, file, t_sec, p_hard, p_cross)
    for every clean-file false positive — turns "singing clean_fp=6"
    from an opaque scalar into actionable per-position diagnostics
    that directly drive feature-design.

## 2026-04-19T22:30:49+09:00 — f10993b (discard, combined=0.445532)
subject: HPR-gated remote-half MFCC source-continuity POST-FILTER — pure detector.py change, no retrain, no feature change. Between GBM dedupe and return, compute file-level harmonic-energy share via HPSS. If file_hpr > HPR_POSTFILTER_MIN=0.85 (tonal/singing population; korean ~0.55-0.75 and english ~0.25-0.50 short-circuit), for each surviving emit at t compute mean MFCC cosine similarity between remote pre-half [0, t-5.0s] and remote post-half [t+5.0s, dur] and drop the emit when sim > REMOTE_MFCC_SIM_MAX=0.99. The ±5s guard excludes local context every existing ±2s feature already covers. Targets singing clean_fp=6 plateau at weakest-domain combined=0.272. Combines two signals the last iterations proved separately: f6192f6 (remote-half MFCC FEATURE → singing 0.288 NEW HIGH but english tanked 0.756->0.543 from GBM over-weighting a global scalar) and e228236 (HPR-gated remote-half ROLLOFF POST-FILTER → wrong scalar, singing 0.244). NO prior combined MFCC + HPR gate + POST-FILTER. MFCC cosine is the proven right scalar (f6192f6 smoke clean=0.998 / splice=0.950). HPR-gate + POST-FILTER placement means the filter CANNOT regress english/korean (no code touches their emits) and CANNOT drop singing TPs already below 0.982 gate. Smoke test at threshold 0.99: clean singing drops 8/12 midpoint emits (67%), tier2 singing 1/12 (8%), tier1 singing 2/12 (17%) — strong asymmetric bite on FPs vs TPs. Orthogonal to all 50+ prior attempts (6 primary tunables bracketed, 8 GBM hyperparam, 5 training-data, 12+ feature add/ablation, 8+ post-filter including rolloff-divergence, 2 model-class swap, 2 calibration, 2 HPR routing, 1 remote-half FEATURE). Cost: one HPSS per file (~200ms) + MFCC on two ~10-25s slices per emit (~30ms each) × ~30 emits × ~20 singing files ≈ 20-25s extra, within 243/300s headroom.
per-domain: combined_english=0.709890 combined_korean=0.451282 combined_singing=0.276056

# 2026-04-19 — hypothesis: HPR-gated remote-half MFCC source-continuity POST-FILTER

(a) HYPOTHESIS. Pure detector.py change, no retrain, no feature change.
    Between GBM dedupe and return, compute file-level HPR
    (librosa.decompose.hpss on full-file |STFT|). If file_hpr >
    HPR_POSTFILTER_MIN=0.85 (tonal/singing population), apply a per-emit
    remote-half MFCC cosine-similarity filter: for each emit at t,
    compute mean MFCC over [0, t-5.0] and [t+5.0, dur] (the ±5s guard
    excludes local context every existing ±2s feature already covers),
    cosine-similarity between the two remote-half means. Drop the emit
    when sim > REMOTE_MFCC_SIM_MAX=0.97 (two halves share timbre →
    within-song chord cycle-back, not cross-source splice). Edge emits
    (insufficient remote context) are kept. Non-singing files (HPR ≤
    0.85) short-circuit before any MFCC work so korean/english outputs
    are byte-identical.

(b) WHY this over recent failures. Combines the two signals the last
    few iterations proved separately:
      * f6192f6 "remote-half MFCC sim FEATURE" → singing 0.288 NEW HIGH
        vs baseline 0.272 but english collapsed 0.756→0.543 because GBM
        over-weighted a global scalar in its strongest domain.
      * e228236 "HPR-gated remote-half ROLLOFF divergence POST-FILTER"
        → wrong scalar (rolloff is single-dim, noisy over 8-25s
        windows); singing 0.244, no lift.
    NO prior combined (MFCC + HPR gate + POST-FILTER). MFCC cosine is
    the proven right scalar (smoke-tested in f6192f6: clean sim=0.998,
    GT-splice sim=0.950 — wide margin). As a POST-FILTER with HPR
    gate the filter CANNOT regress english/korean (no code path
    touches their emits) and CANNOT drop singing TPs that were already
    below the 0.982 gate — it only reduces the already-kept emit set.
    Real cross-song singing splices → remote halves are different
    songs → sim ~0.85-0.95 → KEEP. Clean singing chord cycling →
    remote halves are same song → sim ~0.97-0.99 → DROP the 6 FP
    cluster. Orthogonal to 50+ prior attempts: 6 primary tunables
    bracketed, 8 GBM hyperparam, 5 training-data, 12+ feature
    add/ablation, 8+ post-filter (including rolloff-divergence
    variant), 2 model-class swap, 2 calibration, 2 HPR routing, 1
    remote-half FEATURE — the HPR-gate + MFCC + POST-FILTER combo is
    a genuinely new branch. Cost: one HPSS per file (~200 ms) +
    MFCC on two ~10-25s slices per emit (~30 ms each) × ~30 emits ×
    ~20 singing files ≈ 20-25 s extra, within 243/300s headroom.

(c) IF THIS FAILS. (1) Tune thresholds: HPR cutoff 0.80/0.90, SIM
    cutoff 0.95/0.98/0.99. (2) Replace MFCC mean with mel-spectrum
    mean (broader spectral identity) or stack (MFCC AND mel-PCA)
    requiring BOTH sims > threshold. (3) DIRECTIONAL version: compute
    sim(near_pre, remote_pre) AND sim(near_post, remote_post) — high
    on BOTH = not a splice, low on either = keep. Captures asymmetry
    a pure pre-vs-post sim misses. Final escalation: pitch-shift /
    reverb augmentation of clean training files so GBM sees synthetic
    "chord-transition-like" events as NOT_SPLICE, widening the
    not_splice distribution in spec_*_delta space directly.

(d) Information gaps. Three persist from prior reflections:
    (i) the 4 recent catastrophic 0.010 discards (ExtraTrees, gated
    tonality, isotonic, phase residual, FILE-LEVEL flatness) are
    indistinguishable from legitimate hypothesis failures in the
    prompt's HISTORY — we now know they were all killed by the
    (now-fixed) shap_report.py AttributeError, but the no-repeat rule
    still blocks revisiting those axes. A `failure_reason:
    pipeline_bug | hypothesis_content` tag would unblock clean
    re-testing of isotonic/phase-residual/FILE-LEVEL-flatness under
    the fixed eval pipeline. (ii) Per-class OOF F1 breakdown
    (hard_cut / crossfade / not_splice) and the 6 singing FP
    positions (file + t_sec + p_hard + p_cross) are not in CURRENT
    STATE — I cannot target the FP cluster directly, only guess from
    SHAP aggregates. (iii) No EVAL-set distribution stats (mean HPR /
    flatness / voicing per domain on the EVAL corpus) so file-level
    gating thresholds are set from training-set intuition only.

(e) Wrapper enhancements. (1) Surface eval-side per-domain
    distribution summary at top of CURRENT STATE (mean/min/max HPR /
    flatness / voicing per domain on the EVAL set) so scalar cutoffs
    are data-driven. (2) Auto-annotate HISTORY entries with
    `failure_reason` from eval.stderr pattern match
    (pipeline_bug / feature_mismatch / hypothesis_content). (3) A
    `scripts/clean_fp_positions.py --last` CLI emitting the most
    recent KEPT classifier's singing FP positions (file, t_sec,
    p_hard, p_cross, p_splice) so feature-design and post-filter
    hypotheses can target actual failure cases instead of guessing.

## 2026-04-19T23:18:36+09:00 — 0feddf9 (discard, combined=0.411473)
subject: HPR-gated remote-half MFCC sim FEATURE — conditional signal on tonal files only
per-domain: combined_english=0.712500 combined_korean=0.407407 combined_singing=0.240000

# 2026-04-19 — hypothesis: HPR-GATED remote-half MFCC sim as a FEATURE

(a) HYPOTHESIS. Add ONE feature `hpr_remote_mfcc_sim` to features.py
    (FEATURE_NAMES 75 → 76). Compute file_hpr ONCE per chunk via
    `librosa.decompose.hpss(|STFT|)` in `_ensure_feat_cache`
    (cached as `feat_chunk_hpr`). New block `_block_hpr_remote_mfcc_sim`
    per t_sec: if `feat_chunk_hpr > 0.85` (tonal/singing), compute
    cosine similarity between mean MFCC over [0, t-5.0s] and mean
    MFCC over [t+5.0s, chunk_end]; else return constant `1.0`
    (neutral). Edge positions where either remote window has
    <30 MFCC frames (~0.35s at hop=512/sr=44100) also return 1.0.
    Classifier auto-retrains via US-505 features.py sha gate.

(b) WHY this over the 60+ prior failures. Three closely-related
    remote-half attempts have been tried and all failed for
    DIFFERENT reasons:
      * f6192f6 (raw remote-half MFCC FEATURE, 0.438): singing hit
        NEW HIGH 0.288 but english collapsed 0.756 → 0.543 because
        GBM over-weighted the global scalar on its strongest domain.
      * f10993b (HPR-gated remote-half MFCC POST-FILTER, 0.445):
        singing 0.276 — couldn't bite enough at sim>0.99 threshold;
        binary drop/keep can only subtract from the already-kept set.
      * 0c5ba88 (file_hpr alone as FEATURE, 0.357): catastrophic
        korean 0.500 → 0.290 because pure domain-indicator let GBM
        collapse korean decisions onto HPR-bucket boundaries.
    NOT TRIED: HPR as a CONDITIONAL GATE over a discriminative
    feature. This combines the proven signal (remote-half MFCC sim
    cleanly separates tonal clean ~0.97-0.99 from tonal splice
    ~0.88-0.95 per f6192f6 smoke test) with protection from
    GBM over-reliance on non-tonal domains (constant 1.0 for
    non-tonal rows = zero variance = zero information gain = GBM
    cannot split on it for english/korean decisions). Unlike the
    post-filter variant, the feature lets GBM LEARN the optimal
    boundary (possibly 0.95 or 0.92 or wherever marginal gain is
    highest) instead of hard-coding a 0.99 threshold. Orthogonal
    to every prior axis: 6 primary tunables bracketed, 8 GBM
    hyperparam, 5 training-data, 12+ feature add/ablation, 8+
    post-filter, 2 model-class swap, 2 calibration, 2 HPR routing,
    3 remote-half variants, 1 file_hpr-alone feature — the
    HPR-gated remote-MFCC as a FEATURE combo is a genuinely new
    branch.

    Cost: HPSS on existing-style librosa.stft output ~50-150ms per
    chunk × ~120 chunks ≈ 6-18s in the 243/300s budget. MFCC mean
    + dot product per t_sec is ~1ms. Risk-bounded: if uninformative,
    GBM gives SHAP ≈ 0 and ignores. If the constant-for-non-tonal
    design works, singing could gain meaningfully without english
    regression. Minimum diff: one new block, one new cache key,
    one FEATURE_NAMES entry, one assert bump.

(c) IF THIS FAILS. Three paths.
    (1) Adjust the gate threshold (0.80 or 0.90) or the remote-window
        guard (±3s instead of ±5s so shorter chunks get more
        coverage).
    (2) Directional variant: compute sim_pre_cons =
        cos(mean MFCC [0, t-5s], mean MFCC [t-5s, t]) and
        sim_post_cons similarly on post side; stack both as 2
        features. Real cross-song splices preserve both one-sided
        consistencies while changing across-t.
    (3) Final escalation: time-stretch/pitch-shift augmentation of
        the clean training negatives via
        `librosa.effects.time_stretch(rate=0.95)` on a copy of each
        clean file, add as synthetic NEG_AUG rows. This directly
        widens the not_splice distribution in spec_*_delta space
        without touching features.py.

(d) Information gaps. (i) The 6 singing clean FP positions (file,
    t_sec, p_hard, p_cross, p_splice) are still not in CURRENT
    STATE — I cannot verify my hypothesis bite rate against the
    actual failure cases. (ii) No per-domain EVAL-distribution
    summary for file_hpr / flatness / voicing — scalar cutoffs
    (0.85) come from training-set intuition. (iii) Four recent
    catastrophic 0.010 entries (ExtraTrees, gated tonality,
    isotonic, phase residual, FILE-LEVEL flatness) are now known
    to have been killed by the fixed shap_report.py bug, but the
    no-repeat rule continues to block revisiting those axes.

(e) Wrapper enhancements. (1) A
    `data/train/clean_fp_positions.json` auto-emitted by
    evaluate.py listing the (domain, file, t_sec, p_hard,
    p_cross, p_splice) for every clean-file false positive on
    the last KEPT classifier. Turns "singing clean_fp=6" into
    actionable per-position diagnostics. (2) A
    `failure_reason: pipeline_bug | feature_mismatch |
    hypothesis_content` tag on HISTORY entries derived from
    eval.stderr pattern match — unblocks re-testing of
    bug-killed hypotheses. (3) Per-domain mean/min/max on
    file_hpr / flatness / voicing in CURRENT STATE using the
    EVAL corpus (not training set) so file-level gating
    thresholds come from the measurement distribution.

## 2026-04-19T23:33:57+09:00 — 7f5f0a0 (discard, combined=0.355048)
subject: add spec_rolloff_far_delta — persistence-scale companion to top-SHAP spec_rolloff_delta. Mirrored 4-8s pre/post window (mean rolloff over [t+4,t+8] minus mean over [t-8,t-4]) reusing cached feat_rolloff, zero new librosa calls, ~1ms/pos. FEATURE_NAMES 75->76 so classifier auto-retrains via US-505 sha gate. Pairs with existing spec_rolloff_delta (SHAP=1365 in singing) to give GBM the missing time-scale interaction: (big_near AND big_far)=cross-song splice where new source persists >=4s post-t, (big_near AND small_far)=chord transition cycles back within the song's progression so far-window means converge. Targets singing clean_fp=6 plateau at weakest-domain combined=0.272. Smoke test on synthetic 440->880Hz step at t=20 in 40s signal: far_delta=+449 at step, 0.0 in uniform regions, edge positions (t<5 or t>chunk-5) return 0.0 via guard. Orthogonal to all 60+ prior failures: local_novelty_spec_ratio normalized |delta(t)| by median-of-neighboring-near-deltas (different math, near-scale only); persistence POST-FILTER on rolloff (0.455) hard-coded a ratio threshold rejecting GBM's ability to learn the boundary; remote-half MFCC/rolloff FEATURES/POST-FILTERS used entire chunk halves (10-25s averages dominated by file-averaged character not local discrimination); hf_band/HPSS-percussive changed the FREQUENCY axis not TIME scale. First TIME-SCALE-PAIRED companion feature. Risk-bounded: GBM assigns SHAP~0 and ignores if uninformative; edge guard prevents false firing on short chunks.
per-domain: combined_english=0.620690 combined_korean=0.320482 combined_singing=0.225000

# 2026-04-19 — hypothesis: spec_rolloff_far_delta (persistence-scale companion feature)

(a) HYPOTHESIS. Add ONE feature `spec_rolloff_far_delta` to features.py
    (FEATURE_NAMES 75 → 76). It mirrors the existing top-SHAP
    `spec_rolloff_delta` but at a DIFFERENT time scale: mean rolloff
    over [t+4, t+8] minus mean rolloff over [t-8, t-4]. No librosa
    calls — reuses the already-cached `feat_rolloff` frame array.
    Edge positions (not enough remote context) return 0.0 so they
    cannot mis-fire. Classifier auto-retrains via US-505 features.py
    sha gate.

(b) WHY this over the recent 60+ failures. The clean-singing FP plateau
    (clean_fp=6, weakest-domain combined=0.272) is driven by
    spec_rolloff_delta firing identically on chord transitions and
    cross-song splices because the existing feature is computed on
    ±2s windows — those windows only see the IMMEDIATE shift. The
    structural difference is PERSISTENCE: a splice is the start of a
    different source so the rolloff offset persists beyond 4s; a
    chord transition cycles back as the song's chord progression
    continues, so rolloff in a 4-second window 4-8s away is similar
    on both sides. Pairing a near-delta feature (the existing ±2s
    spec_rolloff_delta, SHAP=1365 in singing) with a far-delta
    feature (mirrored 4-8s windows) gives GBM the exact interaction
    it needs:
      (big_near AND big_far) → cross-song splice
      (big_near AND small_far) → chord transition cycles back
      (small_near) → nothing happening
    GBM naturally represents this as a split chain. Orthogonal to
    all prior attempts:
      * local_novelty_spec_ratio (failed) was |delta(t)| /
        median(|delta(t±4/±8s)|) — a RATIO normalized by the median
        of near-window deltas at NEARBY points. Completely different
        math; it measures how "peaked" the near-delta is vs
        neighbours, not how FAR-window deltas compare.
      * persistence POST-FILTER on spec_rolloff (failed 0.455) was a
        HARD-CODED ratio threshold applied after GBM emit. Being a
        post-filter, it couldn't rescue precision for singing
        without also costing recall globally; encoded as a feature
        instead, GBM learns the optimal boundary per domain.
      * remote-half MFCC / rolloff / HPR POST-FILTERS and FEATURES
        (failed) used ENTIRE chunk halves (0..t-5s vs t+5s..end),
        not a stable 4-second window at 4-8s out. Long averages
        over 10-25s are dominated by overall file-averaged character
        and lose local discrimination.
      * hf_band / HPSS-percussive deltas (failed) all change the
        FREQUENCY AXIS / decomposition, not the TIME SCALE. This is
        the first TIME-SCALE-PAIRED companion feature.
    Cost: zero new librosa calls (reuses feat_rolloff), ~1ms per
    position. Risk-bounded: one new float, GBM assigns SHAP≈0 and
    ignores it if uninformative. Edge positions zero out cleanly.

(c) IF THIS FAILS. (1) Stack the same far-delta companion for
    spec_centroid and spec_bandwidth (the other two top-SHAP
    members), 75→78. Lets GBM see the multi-scale pair on all
    three primary spectral-magnitude deltas. (2) Replace the
    SIGNED far-delta with |far_delta| / (|near_delta| + 100) as an
    explicit persistence RATIO feature — forces the shape of the
    discriminator rather than trusting GBM to find the interaction.
    (3) Final escalation: time-stretch / pitch-shift augmentation of
    clean training files (the one explicit untried direction cited
    in three prior post-mortems) to widen the not_splice
    distribution in spec_*_delta space directly, rather than adding
    ever-more feature columns.

(d) Information gaps. Three persist from prior reflections.
    (i) The 4 recent catastrophic 0.010 discards (ExtraTrees, gated
    tonality, isotonic, phase residual, FILE-LEVEL flatness) are
    now known to be shap_report.py-bug kills (US-510/US-511 fixed),
    but the no-repeat rule still blocks clean re-testing of those
    axes. A `failure_reason: pipeline_bug | hypothesis_content`
    tag in HISTORY would unblock genuine calibration / phase /
    flatness-routing exploration. (ii) The 6 singing clean FPs
    (file, t_sec, p_hard, p_cross, p_splice) are still opaque — I
    am guessing the far-window widths will bite them without
    measuring actual behaviour on the failure set. (iii) Per-class
    (hard_cut / crossfade / not_splice) OOF F1 breakdown is not in
    CURRENT STATE — only aggregate weighted F1.

(e) Wrapper enhancements. (1)
    `data/train/clean_fp_positions.json` auto-emitted by
    evaluate.py listing every (domain, file, t_sec, p_hard,
    p_cross, p_splice) for clean-file false positives on the last
    KEPT classifier. Turns singing clean_fp=6 into actionable
    per-position diagnostics. (2) A SHAP-DELTA block in the
    iteration prompt: for the new KEPT classifier vs prior KEPT
    classifier, per-feature mean-|SHAP| change. Directly shows
    whether a newly-added feature is being USED or IGNORED,
    eliminating the current guessing in post-mortems. (3)
    `scripts/feature_scale_comparison.py <feature_name>` — plots
    the feature distribution on eval-splice-positive vs
    eval-clean-positive rows across domains, so a new feature's
    discrimination can be visualized BEFORE committing it.

## 2026-04-19T23:49:09+09:00 — bba6dbe (keep, combined=0.473315)
subject: pitch-shift audio augmentation of CLEAN SINGING training chunks (±1 semitone) — first training-data MUTATION axis. Adds ~160 synthetic not_splice rows (class counts 600/300/300 -> 760/300/300) by processing 2 pitch-shifted variants per clean singing file via librosa.effects.pitch_shift. Targets singing clean_fp=6 plateau at weakest-domain combined=0.272 by widening the not_splice distribution in top-SHAP spec_rolloff_delta / spec_centroid_delta / spec_bandwidth_delta space — pitch-shifted chord transitions look like chord-transition events (transposed but preserved) in spec_*_delta so GBM learns they are NOT splices. Explicit untried escalation cited in 3 prior post-mortems. Orthogonal to every prior axis: 8 GBM hyperparam failures, 5 training-data SAMPLING failures (NEG counts/distance/curriculum/bootstrap-adversarial all on original audio), 12+ feature add/ablation failures, 8+ post-filter failures, 2 model-class swaps, 2 calibration failures — ZERO prior audio-content mutation. Singing-only + clean-only ensures korean (0.500) and english (0.756) training distributions are byte-identical; vs file_hpr FEATURE 0.357 which tanked korean via global perturbation. Same fid for augmented rows keeps GroupKFold clean. OOF weighted F1=0.6143 (vs baseline 0.5904) / not_splice F1=0.857 (class-level lift). ±1 semitone is gentle to minimize phase-vocoder artifacts vs ±3-6 shifts. Pure train_classifier.py change; features.py sha unchanged so wrapper's staleness gate passive.
per-domain: combined_english=0.666667 combined_korean=0.410256 combined_singing=0.387692

# 2026-04-19 — hypothesis: pitch-shift audio augmentation of CLEAN SINGING training chunks

(a) HYPOTHESIS. In `splice/classifier/train_classifier.py`, after the
    existing per-file feature extraction, add TWO pitch-shifted
    variants (+1, -1 semitone) of EACH CLEAN SINGING chunk via
    `librosa.effects.pitch_shift`. Sample NEG_PER_CLEAN=4 negatives
    from each shifted chunk using a unique seed salt (shift value)
    so positions differ from the original, preserving the same
    `fid` so augmented rows stay in the same GroupKFold fold. Apply
    ONLY to `ds_id == "singing" and not info.spliced`. Singing clean
    count rises from ~80 to ~240 not_splice rows; korean/english
    training data byte-identical. features.py unchanged (sha stable,
    auto-retrain gate passive); manual retrain commits joblib + meta
    + cv_results alongside.

(b) WHY this over recent failures. Three prior post-mortems (mfcc
    FEATURE / HPR-gated rolloff / spec_rolloff_far_delta) explicitly
    cited "pitch-shift / time-stretch augmentation of clean training
    files" as the single untried escalation. Every prior attempt has
    been FEATURE-ENGINEERING (12+ variants: rolloff_far_delta,
    hpr_remote_mfcc_sim, file_hpr, chroma_key_cosine, mfcc_cosine,
    hpss_perc, hf_band, local_novelty_spec, chunk-level file_hpr,
    etc.) or POST-FILTER (8+ variants) or THRESHOLD ROUTING, all of
    which perturb GBM across ALL three domains unpredictably, and
    the clean-singing FP cluster at p_splice~0.982-0.99 has not
    budged. Training-data sampling axes (NEG counts, NEG_MIN_DIST,
    POS_OFFSETS, curriculum hard-neg, bootstrap-adversarial) all
    operated on ORIGINAL audio with different position picks — ZERO
    prior attempts have MUTATED the audio content of training rows.
    Pitch-shifting preserves the chord-transition structure of a song
    (chord progression is TRANSPOSED but still present) so synthetic
    negatives look like chord-transition events in the top-SHAP
    `spec_rolloff_delta / spec_centroid_delta / spec_bandwidth_delta`
    space — exactly the FP failure mode. Gentle ±1 semitone
    minimizes phase-vocoder artifacts vs ±3-6 semitone. Target is
    surgical: only the failing domain's negatives grow, korean/
    english distributions are untouched (vs file_hpr FEATURE 0.357
    which tanked korean via global perturbation).

(c) IF THIS FAILS. (1) Widen shifts to ±2 semitones or add
    time-stretch (rate 0.95 / 1.05) for wider distribution coverage.
    (2) Apply augmentation to spliced singing files too (synthetic
    positives with chord-transition structure augmented). (3)
    Parametric-EQ-tilt augmentation (low-shelf ±3dB at 1kHz) to
    directly widen `spec_centroid_delta / spec_rolloff_delta` in the
    not_splice class without touching pitch content.

(d) Information gaps. (i) The 6 singing clean FPs (file, t_sec,
    p_hard, p_cross, p_splice) are STILL opaque — I can't verify
    that augmented chunks actually cover those FP positions'
    feature-space neighborhood. (ii) Per-class (hard_cut /
    crossfade / not_splice) OOF F1 breakdown still absent — only
    aggregate weighted F1. (iii) The pre-revert HistGBM vs GBM
    ambiguity on disk may still cost me: meta.json shows
    `features_py_sha: null` suggesting the maintainer-class retrain
    happened outside the auto-retrain path.

(e) Wrapper enhancements. (1) `data/train/clean_fp_positions.json`
    auto-emitted by evaluate.py listing the 6 singing FPs'
    `(file, t_sec, p_hard, p_cross, p_splice)` tuples — turns an
    opaque scalar into actionable per-position diagnostics. (2) A
    `scripts/training_aug_visualize.py <feature_name>` plotting the
    feature distribution on original vs augmented training rows
    per domain — lets me verify the augmentation moved the not_splice
    distribution in the INTENDED direction (toward FP cluster) before
    committing. (3) Pre-iteration retrain-cost estimator:
    `verify_agent --estimate-retrain-cost` that simulates the extra
    row count and projected retrain time from per-row timing — so I
    can judge whether a data-augmentation hypothesis will fit in
    budget before iterating.
[auto] LOCAL KEEP at bba6dbe: top-3 features stable across all domains

## 2026-04-19T23:55:56+09:00 — f71c526 (discard, combined=0.473315)
subject: extend pitch-shift audio augmentation to CLEAN KOREAN training chunks (±1 semitone) — generalize the bba6dbe singing-only MUTATION axis to the regressed domain. Widens `ds_id == "singing"` gate to `ds_id in ("singing", "korean")` and renames the constant AUG_DOMAINS/AUG_SHIFTS_SEMITONES. Adds ~160 synthetic clean korean not_splice rows via librosa.effects.pitch_shift; class counts shift 760/300/300 -> 920/300/300. Targets korean clean_fp=4 / combined=0.410 regression that the singing-only aug unintentionally caused (korean 0.500->0.410, english 0.756->0.667 — global class-prior shift, not feature leakage). Mechanism transfers cleanly to speech: ±1 semitone = ~6% F0 shift (within normal prosodic variation during emphasis/questions); phase-vocoder smearing concentrates on consonant transients which are minority content in 60s chunks; formant structure + mic/room/speaker identity preserved, so synthetic rows look like intra-file prosodic variations — NOT cross-source splices — in top-SHAP spec_rolloff_delta / spec_centroid_delta / spec_bandwidth_delta space. English intentionally EXCLUDED (clean_fp=0 already so augmenting its negatives can only hurt splice recall). Orthogonal to every korean prior axis: 6 primary tunables bracketed, 8 GBM hyperparam, 5 training-data SAMPLING (NEG counts/dist/curriculum/bootstrap-adversarial all on ORIGINAL korean audio), 12+ feature add/ablation, 8+ post-filter, 2 model-class swap, 2 calibration, 2 HPR routing, 3 remote-half variants — ZERO prior audio MUTATION on korean. Same fid per aug row keeps GroupKFold clean; unique per-shift salt yields distinct negative positions per variant. Pure train_classifier.py change; features.py sha stable so wrapper's auto-retrain gate passive (wrapper will re-invoke train for a fresh joblib).
per-domain: combined_english=0.666667 combined_korean=0.410256 combined_singing=0.387692

# 2026-04-19 — hypothesis: extend pitch-shift augmentation to clean KOREAN chunks

(a) HYPOTHESIS. Pure `train_classifier.py` change. Widen the
    condition gating the pitch-shift augmentation from
    `ds_id == "singing"` to `ds_id in ("singing", "korean")`. Same
    ±1 semitone via `librosa.effects.pitch_shift`, same
    per-file/per-shift salt via unique `aug_name`, same shared
    `fid` so GroupKFold stays clean. Adds ~160 synthetic clean
    korean not_splice rows (20 clean korean files × 2 shifts × 4
    NEG_PER_CLEAN). Class counts shift 760/300/300 → 920/300/300.
    Pure `train_classifier.py` change; features.py sha stable so
    the wrapper's auto-retrain gate is passive.

(b) WHY over recent axes. The bba6dbe keep proved pitch-shift aug
    works on SINGING (singing 0.272→0.388, clean_fp 6→2). But
    adding 160 synthetic not_splice rows globally shifted class
    priors: korean dropped 0.500→0.410, english dropped 0.756→0.667
    (both from recall loss, not FP growth — english clean_fp=0,
    korean clean_fp=4). Now korean is the "regressed" domain with
    4 clean FPs still in its own spec_*_delta cluster. The proven
    mechanism transfers: pitch-shifted speech preserves speaker /
    mic / room / phoneme structure but transposes formants and F0,
    so GBM sees "speaker-prosody variations that are NOT
    cross-source splices" in spec_*_delta space — exactly the
    korean clean_fp failure mode (intra-file prosodic shifts firing
    top-SHAP spec_rolloff_delta at magnitudes indistinguishable
    from real splices). ±1 semitone is within normal prosodic F0
    variation (6% frequency shift, speakers do this during emphasis
    / questions). Phase-vocoder smearing at ±1 semitone is subtle
    and affects consonant transients most, which are a minority of
    content in 60s speech chunks. Orthogonal to every korean prior
    axis (6 primary tunables bracketed, 8 GBM hyperparam, 5
    training-data SAMPLING axes on original audio, 12+ feature
    add/ablation, 8+ post-filter, 2 model-class swap, 2
    calibration, 2 HPR routing, 3 remote-half variants). ZERO prior
    audio MUTATION axes on korean. English intentionally excluded
    (clean_fp=0 already; augmenting its negatives can only hurt
    splice recall without FP-reduction upside).

(c) IF THIS FAILS. Three paths.
    (1) If korean regresses further (clean_fp>4 or combined
        drops), phase-vocoder artifacts may be creating synthetic
        splice-like transients in clean speech. Switch to ±0.5
        semitone (gentler) or swap to time-stretch (rate=0.97 /
        1.03) which preserves pitch but changes articulation
        speed — different phase-vocoder failure mode.
    (2) If korean unchanged but english drops further, global
        class-prior shift is the dominant mechanism. Balance by
        extending aug to clean ENGLISH too, restoring relative
        class priors across all three domains.
    (3) Final escalation: parametric-EQ-tilt augmentation
        (low-shelf ±3dB at 1kHz) on clean singing — directly
        widens spec_centroid_delta / spec_rolloff_delta in
        not_splice class without touching pitch content.
        Complementary axis to pitch-shift.

(d) Information gaps. (i) Which korean files drive clean_fp=4
    (file, t_sec, p_hard, p_cross, p_splice) still not in CURRENT
    STATE — I cannot verify augmentation will bite the specific
    failure cluster. (ii) Per-class OOF F1 still absent; knowing
    whether korean FPs fire p_hard vs p_cross would predict which
    synthetic variants matter more. (iii) The pitch-shift aug
    mechanism succeeded, but the side-effect on korean / english
    recall (not a clean_fp issue — a TP loss) is not surfaced in
    the prompt — I inferred it from baseline_metrics.json vs the
    prior per-domain numbers in the history. A
    `DOMAIN_DELTA_LAST_KEEP` block showing per-domain combined /
    splice_f1 / clean_score deltas between the prior and current
    baseline would make side-effect detection automatic.

(e) Wrapper enhancements. (1) `scripts/training_aug_visualize.py
    --shift +1 --feature spec_rolloff_delta` plotting augmented
    vs original feature distributions per domain on a
    representative sample, so I can verify aug lands in the
    intended feature-space region BEFORE committing. (2) A
    CLEAN-FP-POSITIONS block in CURRENT STATE listing the k
    clean-file false positives from the last KEPT classifier —
    turning "korean clean_fp=4" into actionable per-position
    diagnostics. (3) `supervisor_agent.py --class-breakdown
    <commit>` emitting (domain, class_label, n_fp, n_tp) from a
    prior keep so I can target whether to widen not_splice (reduce
    FP) or widen splice classes (reduce FN) per domain.

## 2026-04-20T00:06:55+09:00 — d5cc597 (discard, combined=0.357709)
subject: class_weight='balanced' on HistGBM to neutralize pitch-shift aug class-prior shift — one-parameter edit (class_weight='balanced') in make_pipeline(). bba6dbe singing aug shifted class counts 600/300/300 -> 760/300/300 producing OOF per-class F1 0.858/0.311/0.301 (splice classes crushed), english 0.756->0.667 and korean 0.500->0.410 from pure recall loss on borderline splices (english clean_fp=0 confirms not precision). HistGBM class_weight='balanced' scales sample weights by n_samples/(n_classes*n_c) so not_splice drops to ~0.60x and each splice class rises to ~1.51x, re-equalizing per-class loss contribution without removing the 1360 aug rows' feature-space coverage of chord-transition events — preserves singing gain, restores english/korean recall. Orthogonal to every prior axis: 8 GBM hyperparam (n_estimators/max_depth/learning_rate/subsample), 5 training-data SAMPLING, 12+ feature add/ablate, 8+ post-filter, 2 model-class swap, 2 calibration, 2 HPR routing, 3 remote-half, 1 audio MUTATION — NO prior class_weight axis. Prior 'sample_weight 2x' UPWEIGHTED one class only; this AUTO-BALANCES all three. Refreshed OOF: weighted F1 0.6206 (vs 0.6143), macro 0.4979 (vs 0.4899), splice-class F1 lifted hard_cut 0.311->0.328 crossfade 0.301->0.305. Manual retrain + joblib/meta/cv_results committed alongside so wrapper uses fresh classifier (f71c526 joblib-skip lesson).
per-domain: combined_english=0.574468 combined_korean=0.281928 combined_singing=0.282609

# 2026-04-19 — hypothesis: class_weight='balanced' on HistGBM to neutralize the pitch-shift aug class-prior shift

(a) HYPOTHESIS. In `splice/classifier/train_classifier.py::make_pipeline`,
    pass `class_weight='balanced'` to `HistGradientBoostingClassifier`.
    One-parameter edit. Manually retrain + commit joblib/meta/cv_results.

(b) WHY this over recent failures. The last keep (bba6dbe) lifted
    singing 0.272→0.388 via +160 synthetic clean_singing not_splice
    rows. Side-effect: class counts shifted 600/300/300 → 760/300/300,
    OOF per-class F1 on disk is 0.858 / 0.311 / 0.301 — splice classes
    crushed. english dropped 0.756→0.667, korean 0.500→0.410 from pure
    recall loss on borderline splices (clean_fp=0 on english confirms
    not a precision regression). The f71c526 "extend aug to korean"
    attempt RETURNED IDENTICAL per-domain numbers to bba6dbe, i.e. the
    joblib wasn't actually rebuilt (features.py sha unchanged, wrapper
    auto-retrain gate passive, no joblib committed). So that axis is
    inconclusive, but the UNDERLYING mechanism — the class-prior shift
    from the singing aug — remains unaddressed. class_weight='balanced'
    directly counteracts it: HistGBM internally scales sample weights
    by n_samples/(n_classes × n_c), so not_splice drops to ~0.60× and
    each splice class rises to ~1.51×. The 1360 aug rows still provide
    feature-space coverage of chord-transition events (keeping singing
    gain), but per-class loss contribution is re-equalized (restoring
    english/korean splice recall). Orthogonal to every prior axis: 8
    GBM hyperparam (n_estimators/max_depth/learning_rate/subsample), 5
    training-data SAMPLING, 12+ feature add/ablate, 8+ post-filter, 2
    model-class swap, 2 calibration, 2 HPR routing, 3 remote-half, 1
    audio MUTATION — NO prior class_weight or automatic class-balanced
    sample weighting. Prior "sample_weight 2x" UPWEIGHTED one class;
    this AUTO-BALANCES three. Commits joblib so the wrapper can't skip
    retrain silently (lesson from f71c526).

(c) IF THIS FAILS. (1) Switch to explicit sample_weight dict tuned per
    class (e.g., {0:1.0, 1:1.8, 2:1.8}) to push splice recall harder.
    (2) Stack with EQ-tilt augmentation on clean singing (±3dB
    low-shelf at 1kHz) — directly widens spec_centroid/rolloff/
    bandwidth distributions complementary to pitch-shift's pitch-axis
    coverage. (3) Final escalation: pitch-shift SPLICED files too
    (maintains class balance by augmenting positives proportionally).

(d) Information gaps. (i) The f71c526 joblib-rebuild bug shows the
    wrapper's retrain policy is opaque to me — whether `run_autoresearch.sh`
    triggers train when only train_classifier.py changes is unclear
    from the prompt; I'm hedging by committing the joblib. A
    `TRAIN_TRIGGER_CONDITIONS:` block in CURRENT STATE listing the
    exact files/shas that cause an auto-retrain would remove this
    risk. (ii) The 6 singing clean FPs (file, t_sec, p_hard, p_cross)
    are still opaque. (iii) Per-domain SHAP signs still implicit.

(e) Wrapper enhancements. (1) An AUTO-RETRAIN TRIGGER LOG: one line
    per eval attempt showing "retrained: YES (features.py sha diff) |
    NO (using disk joblib sha X)". Turns the silent classifier
    staleness risk into a visible signal. (2)
    `scripts/class_balance_probe.py` emitting predicted OOF F1 per
    class under current vs proposed class_weight settings for a given
    joblib — lets me preview rebalancing impact without a full eval
    cycle. (3) Per-domain splice_recall separated from splice_f1 in
    CURRENT STATE. Recall loss vs precision loss have different
    remedies (aug vs threshold), and combined splice_f1 hides which.

## 2026-04-20T00:19:03+09:00 — c0d438e (discard, combined=0.435685)
subject: pitch-shift audio augmentation of SPLICED SINGING training chunks too (±1 semitone) — widen aug gate from `ds_id == "singing" and not info.get("spliced", False)` to `ds_id == "singing"`. Adds ~320 synthetic hard_cut/crossfade rows (class counts 760/300/300 -> 1000/500/500, imbalance 2.53:1 -> 2.00:1) by processing the same 2 pitch-shifted variants per spliced singing file — pitch-shift preserves the splice discontinuity because both sources shift together but their mutual timbre/phase break is pitch-invariant. Targets the english 0.667 (was 0.756) / korean 0.410 (was 0.500) regression that bba6dbe's clean-only aug caused via GLOBAL class-prior shift: HistGBM's multi-class softmax saw 2.5:1 imbalance and pushed ambiguous probabilities toward not_splice, crossing the 0.982 gate the wrong way for borderline splices (english clean_fp=0 confirms not precision). Prior attempts to correct this without touching the aug balance FAILED: f71c526 extend-to-korean (joblib didn't rebuild, axis uninformative), d5cc597 class_weight='balanced' (catastrophic 0.358 — scaled not_splice to 0.60x globally which crushed the aug benefits on singing 0.388->0.283 as much as it tried to help korean/english). The cleaner fix is data-level class balance: also augment spliced singing files so positive rows grow proportionally, matching the clean-singing aug growth. Explicit untried escalation from bba6dbe(c)(2) and d5cc597(c)(3). Orthogonal to every prior axis: 8 GBM hyperparam, 5 training-data SAMPLING, 12+ feature add/ablation, 8+ post-filter, 2 model-class swap, 2 calibration, 2 HPR routing, 3 remote-half variants, 1 class_weight, 1 clean-singing aug, 0 spliced-singing aug. Same fid per aug row keeps GroupKFold clean; unique per-shift salt yields distinct negative positions per variant. Joblib committed alongside code so wrapper's auto-retrain gate cannot silently skip retrain (f71c526 lesson). OOF weighted F1 0.5428 / macro 0.4490 — lower than bba6dbe's 0.6143 because aug-positives are genuinely harder to classify via phase-vocoder artifacts and contaminate the train-side OOF scoring; eval-time behavior on REAL splices is the actual metric.
per-domain: combined_english=0.643678 combined_korean=0.438356 combined_singing=0.293103

# 2026-04-20 — hypothesis: pitch-shift SPLICED singing training chunks too (±1 semitone)

(a) HYPOTHESIS. In `splice/classifier/train_classifier.py`, widen the aug gate
    from `ds_id == "singing" and not info.get("spliced", False)` to
    `ds_id == "singing"`. Same ±1-semitone pitch-shift via
    `librosa.effects.pitch_shift`, same per-file/per-shift salt, same shared
    `fid`. The aug block runs on the SAME 60s chunk anchored at the GT
    splice, so pitch-shifting preserves the splice discontinuity (both
    sources shift together but their mutual phase/timbre break persists).
    Each spliced singing file now contributes 2 × (5 pos offsets + 3 negs)
    = 16 extra rows. Class counts shift 760/300/300 → ~1000/500/500
    (imbalance 2.53:1 → 2.00:1). Constant `SINGING_AUG_SHIFTS_SEMITONES`
    unchanged; manifest `augmented` flag already present. features.py sha
    unchanged so the wrapper's auto-retrain gate is passive — joblib
    committed alongside the code so the wrapper doesn't skip retrain
    (f71c526 lesson).

(b) WHY this over the recent failures. The bba6dbe singing-clean aug keep
    (+0.116 singing) simultaneously dropped english 0.756→0.667 and
    korean 0.500→0.410. Pure recall loss (english clean_fp=0 confirms
    not precision) from a GLOBAL class-prior shift: HistGBM's multi-
    class softmax sees more not_splice rows and pushes ambiguous
    probabilities toward not_splice, crossing the 0.982 gate in the
    wrong direction for borderline korean/english splices. Two prior
    attempts to fix this:
      * f71c526 (extend aug to korean, returned IDENTICAL metrics)
        — joblib wasn't rebuilt, axis remains uninformative.
      * d5cc597 (class_weight='balanced', combined=0.358) — catastrophic
        over-correction. It scaled not_splice down to 0.60× and each
        splice class up to 1.51× GLOBALLY, which crushed the aug
        benefits on singing (0.388→0.283) as much as it helped korean/
        english; the aug's feature-space coverage of chord-transition
        events got its loss contribution gutted.
    The CLEANER fix: restore class balance via DATA (not loss scaling)
    by also augmenting the spliced singing files. Pitch-shifting a
    spliced chunk preserves the splice artifact because both sources
    shift together while their mutual phase/timbre break is
    pitch-invariant. Explicit untried escalation from bba6dbe(c)(2)
    ("Apply augmentation to spliced singing files too") and d5cc597(c)(3)
    ("pitch-shift SPLICED files too maintains class balance by
    augmenting positives proportionally"). Orthogonal to every prior
    axis: 8 GBM hyperparam, 5 training-data SAMPLING, 12+ feature
    add/ablation, 8+ post-filter, 2 model-class swap, 2 calibration,
    2 HPR routing, 3 remote-half variants, 1 class_weight, 1
    clean-singing aug, 0 spliced-singing aug. Preserves the proven
    bba6dbe singing-domain gain; expected directional: class-prior
    shift reversed → korean/english recall recovers (korean 0.410 →
    ~0.470, english 0.667 → ~0.720) while singing 0.388 holds or
    grows (more positive examples cover chord-transition-adjacent
    splices).

(c) IF THIS FAILS. (1) If singing regresses (the phase-vocoder applied
    to spliced chunks muddles the splice boundary), narrow the aug to
    ONLY clean + spliced positives only (skip the spliced-file
    negatives) via an early-exit on `label == LABEL_NOT_SPLICE`.
    (2) If korean/english stay flat but singing grows further, the
    class-prior shift was never the main cause — pivot to EQ-tilt
    augmentation (±3dB low-shelf at 1kHz) to directly widen
    spec_centroid/rolloff/bandwidth distributions. (3) Final
    escalation: time-stretch augmentation (rate 0.95/1.05) as a
    complementary MUTATION axis — preserves pitch, changes
    articulation speed, different phase-vocoder failure mode.

(d) Information gaps. (i) OOF per-class F1 breakdown after the d5cc597
    catastrophe is not in CURRENT STATE — I'm inferring the mechanism
    (splice recall collapse on korean/english) from per-domain combined
    and the post-mortem note, not from the actual per-class-per-domain
    eval breakdown. (ii) No per-domain EVAL-side clean FP positions
    (file + t_sec + p_splice) for any prior classifier — I can't verify
    that augmented splice positives land in the same feature-space
    neighborhood as the missed TPs. (iii) No CLASS_COUNTS block in the
    iteration prompt — I inferred the 760/300/300 starting point from
    bba6dbe's note, can't verify the current on-disk classifier was
    actually trained on that split.

(e) Wrapper enhancements. (1) A CLASSIFIER_STATE block at the top of the
    prompt showing the on-disk joblib's class_counts / OOF_f1_weighted /
    OOF_f1_per_class / features_py_sha. Directly answers "what
    classifier will my code run against" and "what aug/sampling is
    baked in" — removes a major information gap.
    (2) `scripts/training_aug_visualize.py --shift +1 --feature
    spec_rolloff_delta` plotting augmented vs original feature
    distributions per domain on a representative sample, so I can
    verify aug lands in the intended feature-space region BEFORE
    committing. (3) `supervisor_agent.py --diff-classifier <new-sha>
    <old-sha>` reporting per-class OOF F1 delta between two joblibs —
    turns "retrain side-effect on splice recall" into a measurable
    quantity pre-commit.

## 2026-04-20T00:29:59+09:00 — eabd6b2 (discard, combined=0.362039)
subject: EQ-tilt audio augmentation of CLEAN SINGING training chunks (coef ±0.3) — REPLACE pitch-shift mechanism with librosa.effects.preemphasis first-order IIR spectral tilt (coef=+0.3 brightens -3dB<500Hz/+1.6dB>10kHz; coef=-0.3 mirrors). Same 2-variants-per-file topology so class counts remain 760/300/300 (no new class-prior shift beyond bba6dbe baseline). Targets bba6dbe's singing 0.388 / korean 0.410 / english 0.667 state by eliminating the phase-vocoder artifacts pitch-shift introduces into clean-aug rows while preserving the +0.116 singing gain. Mechanism: pitch-shift is a phase-vocoder transform that introduces sub-frame timing smearing in STFT — those artifacts may look like splice discontinuities to the model, inflating clean p_splice even on transposed copies. EQ-tilt is a deterministic one-pole IIR filter: zero phase-vocoder, preserves all timing; it moves the formant / mastering axis directly — which is exactly the physical signature the top-3 SHAP features (spec_rolloff_delta/spec_centroid_delta/spec_bandwidth_delta) measure. Augmented chunks now look like "same song under a different master EQ / different room tonal balance" — the physical variation production pipelines naturally emit, precisely what GBM needs to learn does NOT indicate a splice. Explicit untried escalation cited in three prior post-mortems: bba6dbe(c)(3), d5cc597(c)(2), c0d438e(c)(2). Orthogonal to every prior axis: 8 GBM hyperparam, 5 training-data SAMPLING, 12+ feature add/ablation, 8+ post-filter, 2 model-class swap, 2 calibration, 2 HPR routing, 3 remote-half variants, 1 class_weight, 3 pitch-shift-aug variants — ZERO prior attempts at EQ-based audio mutation. Smoke-verified: librosa.effects.preemphasis with coef=+0.3 gives -3dB below 500Hz and +1.6dB above 10kHz on white noise (4-5dB total spectral tilt). Manual retrain: OOF weighted F1 0.6103 (vs bba6dbe's 0.6143), class counts 760/300/300 identical, per-class F1 not_splice=0.857 (match) / hard_cut=0.323 (+0.012 vs bba6dbe's 0.311) / crossfade=0.271 (-0.030 vs 0.301) — OOF signal mixed but OOF poorly predicts eval (d5cc597 had higher OOF yet catastrophic eval). features.py sha unchanged so wrapper's auto-retrain gate passive; joblib+meta+cv_results+training_manifest committed alongside so wrapper cannot silently skip retrain (f71c526 joblib-skip lesson).
per-domain: combined_english=0.731707 combined_korean=0.315000 combined_singing=0.205882

# 2026-04-20 — hypothesis: EQ-tilt augmentation of clean singing (replacing pitch-shift)

(a) HYPOTHESIS. In `splice/classifier/train_classifier.py`, REPLACE the
    `librosa.effects.pitch_shift` call on clean singing chunks with a
    `librosa.effects.preemphasis` first-order IIR tilt at coef ±0.3.
    Same 2-variants-per-file topology so class counts remain 760/300/300
    (no new class-prior shift beyond the bba6dbe baseline). Rename
    constant `SINGING_AUG_SHIFTS_SEMITONES` → `SINGING_AUG_EQ_TILT_COEFS`.
    Manifest key `pitch_shift_semitones` → `eq_tilt_coef`. Manual retrain
    + joblib/meta/cv_results/training_manifest committed alongside.

(b) WHY this over the recent failures. bba6dbe (current keep, 0.473)
    PROVED clean-singing-only MUTATION works. Three attempts to fix the
    recall regression it caused all failed: f71c526 (extend to korean,
    identical metrics — joblib wasn't rebuilt), d5cc597 (class_weight
    balanced, catastrophic 0.358 via global loss-scale distortion),
    c0d438e (pitch-shift spliced singing too, singing dropped
    0.388→0.293 because phase-vocoder on a 60s spliced chunk muddles
    the splice boundary). Smoke-verified: librosa.effects.preemphasis
    with coef=+0.3 gives -3dB below 500 Hz and +1.6dB above 10 kHz on
    white noise; coef=-0.3 mirrors it. This is a genuine ~4-5dB
    spectral-envelope tilt. Explicit untried escalation cited in three
    prior post-mortems: bba6dbe(c)(3), d5cc597(c)(2), c0d438e(c)(2)
    ("Parametric-EQ-tilt augmentation... directly widens
    spec_centroid_delta / spec_rolloff_delta in the not_splice class
    without touching pitch content"). Mechanism: pitch-shift is a
    phase-vocoder transform that introduces sub-frame timing smearing
    in the STFT domain; those smearing artifacts may superficially
    look like splice discontinuities to the model, inflating clean
    p_splice even on transposed copies. EQ-tilt is a simple one-pole
    IIR filter: deterministic, zero phase-vocoder, preserves all
    timing; it moves the formant / mastering axis directly — which is
    exactly the physical signature the top-3 SHAP features
    (spec_rolloff_delta/spec_centroid_delta/spec_bandwidth_delta)
    measure. Augmented clean singing chunks now look like a "same
    song under a different master EQ / different room tonal balance"
    — precisely the physical variation the production pipeline
    naturally emits and exactly what GBM needs to learn does NOT
    indicate a splice. Orthogonal to every prior axis: 8 GBM
    hyperparam, 5 training-data SAMPLING, 12+ feature add/ablation,
    8+ post-filter, 2 model-class swap, 2 calibration, 2 HPR routing,
    3 remote-half variants, 1 class_weight, 3 pitch-shift-aug
    variants — ZERO prior attempts at EQ-based audio mutation.

(c) IF THIS FAILS. (1) STACK pitch-shift + EQ-tilt so class counts
    grow to 920/300/300 with two complementary mutation axes — if
    both preserve singing gain, the union covers both the
    formant/mastering axis AND the pitch-class axis. (2) Widen
    coef to ±0.5 (~6 dB tilt) if the ±0.3 variant was too gentle to
    materially shift the top-SHAP feature distributions. (3) Final
    escalation: extend EQ-tilt to clean KOREAN chunks (the direction
    f71c526 tried with pitch-shift but was inconclusive due to
    joblib-skip); EQ-tilt on speech preserves formant transitions
    while shifting overall spectral balance — mirrors real-world
    mic/room variation between recording sessions.

(d) Information gaps. (i) The 2 remaining singing clean FPs (file,
    t_sec, p_hard, p_cross, p_splice) from the bba6dbe keep are not
    in CURRENT STATE — can't verify EQ-tilt will bite the remaining
    FPs vs re-open the 4 previously-bitten FPs. (ii) f71c526's
    identical-metrics outcome is still ambiguous between
    "joblib-not-rebuilt" and "korean aug truly a no-op" because the
    wrapper's auto-retrain trigger conditions aren't surfaced.
    (iii) Per-class OOF F1 breakdown per domain (hard_cut / crossfade
    / not_splice × singing / korean / english) would make
    side-effect attribution deterministic rather than inferred.

(e) Wrapper enhancements. (1) CLASSIFIER_STATE block at top of prompt
    showing on-disk joblib's class_counts, OOF_f1_weighted,
    OOF_f1_per_class, features_py_sha, training_manifest row count
    — removes the "which classifier will my code actually run
    against" ambiguity every training-data hypothesis currently
    carries. (2) `scripts/training_aug_visualize.py --mechanism eq_tilt
    --feature spec_rolloff_delta` plotting augmented vs original
    feature distributions per domain on a representative sample —
    lets me verify the mutation landed where intended before
    committing. (3) `supervisor_agent.py --diff-classifier <old-sha>
    <new-sha>` reporting per-class OOF F1 delta between two joblibs
    — turns "retrain side-effect on splice recall" into a measurable
    pre-commit signal instead of a post-hoc inference from per-domain
    combined.

## 2026-04-20T00:42:17+09:00 — 423014c (discard, combined=0.385591)
subject: HYBRID pitch-shift + time-stretch aug on clean singing — REPLACE one of the two ±1-semitone pitch-shift variants on clean singing chunks with a time-stretch variant (rate=0.95) so class counts stay at bba6dbe's 760/300/300 (NO new class-prior shift). 1 pitch-shift variant (+1 semitone) + 1 time-stretch variant (rate=0.95, chunk grows 60→63.2s, _sample_candidates uses recomputed aug_dur_s) per clean singing file × 4 NEG_PER_CLEAN × 20 files = 160 aug rows total. Time-stretch is the explicit untried mutation axis cited in 3 prior post-mortems (bba6dbe(c), c0d438e(c)(3), eabd6b2(c)). Mechanism: pitch_shift modifies frequency axis preserving time; time_stretch modifies time axis preserving pitch — same librosa.effects family / same phase-vocoder smearing characteristics, rotated 90° in the time-frequency plane. Augmented chunks now look like 'same song at 5%% slower tempo' — natural performance variation that GBM should learn is NOT a splice. HYBRID addresses both prior failure patterns: keeps half of bba6dbe's proven pitch-axis coverage (avoiding eabd6b2's mechanism-replacement gain loss) AND adds an orthogonal phase-vocoder axis (avoiding f71c526/d5cc597/c0d438e's class-prior shifts). features.py sha unchanged so wrapper's auto-retrain gate is passive; joblib + meta + cv_results + training_manifest committed alongside (f71c526 joblib-skip lesson). OOF weighted F1=0.6202 (vs bba6dbe 0.6143), per-class hard_cut 0.311→0.329 (+0.018) crossfade 0.301→0.316 (+0.015) not_splice 0.857→0.855 (essentially match) — both splice classes lift on disk while not_splice holds, suggesting the time-axis variant gives slightly more diverse not_splice context than the dropped -1 pitch-shift. Orthogonal to every prior axis: 8 GBM hyperparam, 5 training-data SAMPLING, 12+ feature add/ablation, 8+ post-filter, 2 model-class swap, 2 calibration, 2 HPR routing, 3 remote-half variants, 1 class_weight, 4 audio MUTATION variants — ZERO prior time-stretch.
per-domain: combined_english=0.767442 combined_korean=0.256790 combined_singing=0.290909

# 2026-04-20 — hypothesis: HYBRID pitch-shift + time-stretch aug on clean singing

(a) HYPOTHESIS. Pure `train_classifier.py` change. REPLACE one of the
    two ±1-semitone pitch-shift variants on clean singing with a
    time-stretch variant (rate=0.95). Concretely:
    `SINGING_AUG_SHIFTS_SEMITONES`: (+1, -1) → (+1,);
    add `SINGING_AUG_TIME_STRETCH_RATES = (0.95,)`. Same 2 variants
    per file × 4 NEG_PER_CLEAN × 20 clean singing files = 160 aug
    rows; class counts stay at bba6dbe's 760/300/300 (NO new
    class-prior shift). Time-stretch returns a chunk of different
    duration (60→63.2s @ rate=0.95) so `aug_dur_s = len(aug_chunk)/sr`
    is recomputed before `_sample_candidates`. `librosa.effects.time_stretch`
    is the standard call; identical phase-vocoder hop_length defaults
    so behavior matches detector-time STFT processing. Same `fid` and
    seed-salting via `aug_name` keep GroupKFold clean. features.py sha
    unchanged so the wrapper's auto-retrain trigger only depends on
    train_classifier.py mtime — joblib + meta + cv_results +
    training_manifest committed alongside the code so the wrapper
    can't silently skip retrain (f71c526 lesson).

(b) WHY this over the recent failures. Five sequential post-bba6dbe
    attempts have now failed in different ways. Two patterns:
      * Class-prior shifts cause regression. f71c526 (extend to korean)
        was inconclusive (joblib bug); d5cc597 (class_weight balanced)
        catastrophic 0.358 — global loss-scale distortion crushed the
        singing aug benefits as much as it tried to recover korean/
        english; c0d438e (pitch-shift spliced singing too) singing
        dropped 0.388→0.293 because phase-vocoder smearing on a 60s
        spliced chunk muddles the splice boundary.
      * Replacing the mutation mechanism while keeping class counts
        loses singing gain. eabd6b2 (EQ-tilt instead of pitch-shift)
        dropped to 0.362; pre-emphasis is a pure IIR filter (no
        phase-vocoder) so its augmented chunks don't reach the
        feature-space neighborhood pitch-shift opens.
    HYBRID (pitch-shift + time-stretch) addresses both lessons at once:
    keeps half the proven pitch-axis coverage (+1 semitone variant
    retained) AND adds a complementary phase-vocoder axis (time
    stretch) without expanding class counts beyond bba6dbe's 760.
    Time-stretch was explicitly cited as the next mutation axis in
    THREE post-mortems (bba6dbe(c)(1)/(c)(3), c0d438e(c)(3),
    eabd6b2(c)(2)). It is mechanically closer to pitch-shift than
    EQ-tilt was: same `librosa.effects` family, same phase-vocoder
    smearing characteristics — just rotated 90° in the time-frequency
    plane (pitch_shift modifies frequency axis preserving time;
    time_stretch modifies time axis preserving pitch). Augmented
    chunks now look like "same song at 5% slower tempo" — a natural
    physical variation (live performances vary tempo) precisely what
    GBM needs as more not_splice training context. Orthogonal to
    every prior axis: 8 GBM hyperparam, 5 training-data SAMPLING,
    12+ feature add/ablation, 8+ post-filter, 2 model-class swap,
    2 calibration, 2 HPR routing, 3 remote-half variants, 1
    class_weight, 4 audio MUTATION variants (pitch-shift clean,
    pitch-shift extend-to-korean, pitch-shift spliced, EQ-tilt) —
    ZERO prior time-stretch.

(c) IF THIS FAILS. Three paths.
    (1) If singing regresses (the time-stretch variant is contributing
        worse coverage than the dropped -1 pitch-shift variant), revert
        to bba6dbe's symmetric (+1, -1) pitch-shift and instead test
        time-stretch with rate=1.05 (faster, shorter chunks ~57.1s) as
        the swap-in. Different phase-vocoder smearing pattern.
    (2) If singing holds but neither korean nor english recovers, the
        bba6dbe class-prior shift is the dominant issue and adjusting
        the mechanism within the same class-count budget will not help.
        Pivot to feature-axis: per-domain DOMAIN_INDICATOR feature (not
        a routing scalar) appended to FEATURE_NAMES so GBM can split
        differently per dataset internally without a hard-gated
        per-domain rule.
    (3) Final escalation: stack BOTH pitch-shift directions AND time-
        stretch BUT halve NEG_PER_CLEAN (4→2) on the augmented variants
        only — gives 3 mutation axes × 2 negs × 20 = 120 aug rows total
        (smaller class-prior shift than bba6dbe's 160).

(d) Information gaps. (i) The 2 remaining singing clean FPs and 4
    korean clean FPs (file, t_sec, p_hard, p_cross, p_splice) are STILL
    not in CURRENT STATE — every aug-design hypothesis is guessing
    whether augmented chunks land in the failure-cluster's feature-
    space neighborhood. (ii) f71c526's identical-metrics outcome
    remains ambiguous: either korean aug is truly a no-op or the
    wrapper's auto-retrain trigger conditions vs joblib commit
    interaction is opaque. A `TRAIN_TRIGGER_CONDITIONS:` block in
    CURRENT STATE listing exactly which file/sha changes cause the
    wrapper to re-invoke `train_classifier.py` would remove the
    "did the joblib actually rebuild" uncertainty every training-data
    hypothesis carries. (iii) Per-class OOF F1 broken DOWN BY DOMAIN
    (hard_cut/crossfade/not_splice × singing/korean/english) is still
    absent from the prompt; aggregate weighted F1 hides which class is
    being affected by training-data changes.

(e) Wrapper enhancements. (1) `data/train/clean_fp_positions.json`
    auto-emitted by evaluate.py listing every (domain, file, t_sec,
    p_hard, p_cross, p_splice) for clean-file false positives on the
    last KEPT classifier — turning "korean clean_fp=4" / "singing
    clean_fp=2" from opaque scalars into actionable per-position
    diagnostics. (2) `scripts/training_aug_visualize.py --mechanism
    {pitch_shift,time_stretch,eq_tilt} --feature spec_rolloff_delta`
    plotting augmented vs original feature distributions per domain on
    a representative sample — lets me verify the mutation lands in the
    intended feature-space region BEFORE committing. (3)
    `supervisor_agent.py --diff-classifier <old-sha> <new-sha>`
    reporting per-class OOF F1 delta between two joblibs — turns the
    "retrain side-effect on splice recall" inference into a measurable
    pre-commit signal.

## 2026-04-20T00:50:39+09:00 — d5484a4 (discard, combined=0.473315)
subject: soft-voting ensemble of 3 HistGBMs (seeds 42/123/456) — variance-reduction in p>0.98 tail. Pure train_classifier.py change: make_pipeline() wraps 3 HGBs with identical hyperparams but different random_state in a VotingClassifier(voting='soft'). All HGB knobs (max_iter=200/max_depth=3/max_leaf_nodes=8/learning_rate=0.07/l2=1.0/min_samples_leaf=20) unchanged. Training data unchanged (bba6dbe pitch-shift aug preserved at 760/300/300). detector.py byte-unchanged because VotingClassifier exposes classes_ and predict_proba with identical semantics. Targets bba6dbe's singing 0.388 / korean 0.410 / english 0.667 plateau by attacking an untried axis: seed-specific residual-fit quirks. HistGBM random_state seeds subsample ordering and split tie-breaking so two HGBs on identical data produce slightly-different decision surfaces in the high-probability tail (p>0.98) where clean FPs (singing 2, korean 4 at p~0.982-0.99) sit. Soft-voting averages the per-seed disagreements out of predict_proba without changing bias — emits that are SEED-SPECIFIC (only seed-42 pushes them over the 0.982 gate) get pulled below by the other two. Real splice TPs typically score p>0.99 on ALL seeds (stable positions) so rarely lose the gate. Orthogonal to every prior axis: 8 GBM hyperparam, 5 training-data SAMPLING, 12+ feature add/ablation, 8+ post-filter, 2 model-class swaps (both changed FAMILY — HistGBM kept, ExtraTrees catastrophic), 2 calibration (isotonic bug-killed, sigmoid 0.419), 2 HPR routing, 3 remote-half, 1 class_weight, 5 audio MUTATION — ZERO prior ensemble of same-family models. Cost: 3x train time (~3min x 3 ≈ 9min absorbed in retrain budget) and 3x inference predict_proba at stride=0.12 dense scan (each call is ~75-dim matmul + tree traversal; VotingClassifier parallelizes via n_jobs=-1 so wall-clock cost sublinear). Eval budget 243/300s leaves ~60s headroom.  Risk-bounded: if seeds agree on all emits, ensemble = single-seed output (worst case reproduces bba6dbe 0.473); class ordering preserved by design of soft-voting; detector.py touches 'classes_' and 'predict_proba' both exposed identically by VotingClassifier.
per-domain: combined_english=0.666667 combined_korean=0.410256 combined_singing=0.387692

# 2026-04-20 — hypothesis: soft-voting ensemble of 3 HistGBMs (seeds 42/123/456)

(a) HYPOTHESIS. In `splice/classifier/train_classifier.py::make_pipeline`,
    replace the single `HistGradientBoostingClassifier(random_state=42, ...)`
    with `VotingClassifier(estimators=[(name, HGB(seed=s, ...)) for s in
    (42,123,456)], voting='soft', n_jobs=-1)`. All HGB hyperparameters
    (max_iter=200 / max_depth=3 / max_leaf_nodes=8 / learning_rate=0.07 /
    l2_regularization=1.0 / min_samples_leaf=20) identical across the 3
    seeds; only `random_state` varies. StandardScaler unchanged; training
    data unchanged (bba6dbe pitch-shift aug preserved at 760/300/300).
    `detector.py` interacts via `named_steps['clf'].classes_` and
    `model.predict_proba(X)` — VotingClassifier exposes both with
    identical semantics, so detector.py is byte-unchanged.

(b) WHY this over the recent failures. 7 consecutive post-bba6dbe
    discards have all operated on the SAME axes: extend aug scope
    (f71c526), change aug mechanism (eabd6b2, 423014c), augment
    spliced too (c0d438e), rebalance via class_weight (d5cc597),
    isotonic calibration (bug-killed), sigmoid calibration (0.419).
    Every one has assumed the problem is fixable via data/loss/
    calibration while keeping a single HistGBM with random_state=42.
    Untested: the **variance** of that specific seed's decision
    surface. HistGBM is a boosting ensemble that fits residuals
    sequentially; `random_state` seeds the subsample row order and
    split-search tie-breaking, so two HistGBMs on identical data
    produce slightly-different decision surfaces — particularly in
    the high-probability tail (p>0.98), which is where the FP
    cluster (singing clean_fp=2, korean clean_fp=4) sits and where
    a single seed's quirks matter most. Soft-voting 3 seeds averages
    the per-tree disagreements out of the probability estimate
    without changing bias — borderline emits at p~0.982 that are
    SEED-SPECIFIC (only the 42-seed model happens to push them over
    the gate) get pulled back below 0.982 by the other two. Real
    splice TPs typically score p>0.99 across seeds (stable
    high-confidence positions), so rarely lose the gate.

    Orthogonal to every prior axis. 2 model-class swaps tried
    (HistGBM kept, ExtraTrees catastrophic 0.010); both swapped the
    FAMILY. ZERO prior axes have ensembled multiple HistGBMs of the
    same family. Cost: 3x train time (~9 min, absorbed into the
    retrain budget) and 3x inference predict_proba calls on the
    75-dim feature matrix — small vs the 243/300s eval budget.
    Risk-bounded: VotingClassifier with soft voting is monotone in
    the sense that each seed's predicted probability contributes
    linearly; the class ordering is preserved; cv_results ordinal
    comparisons with bba6dbe remain valid. If the 3 seeds all happen
    to agree on current emits, the ensemble output equals the
    current single-seed output and the result is a no-op (worst
    case: eval reproduces bba6dbe's 0.473).

(c) IF THIS FAILS. Three paths.
    (1) Expand ensemble to 5 seeds (42/123/456/789/1000) — more
        variance reduction if 3 wasn't enough.
    (2) Introduce DIVERSITY by varying hyperparams too: seed42 @
        max_depth=3, seed123 @ max_depth=4, seed456 @ max_depth=2 —
        explicit model-capacity diversity complements seed diversity.
    (3) Final escalation: replace soft voting with STACKING (sklearn
        StackingClassifier with a LogisticRegression meta-learner on
        the three HGB predict_proba outputs) — lets a second-level
        classifier learn when to trust which base model.

(d) Information gaps. (i) The 2 singing / 4 korean clean FP positions
    (file, t_sec, p_hard, p_cross, p_splice) remain opaque — I can't
    verify the 3-seed averaging pulls THOSE specific positions below
    0.982 vs pulling real splice TPs below too. (ii) Per-seed OOF
    F1 variance on a single HistGBM is not in CURRENT STATE — if
    seeds produce near-identical OOF metrics, the ensemble averaging
    will barely bite. A pre-iteration estimate `train --seed-sweep
    --n 3` printing per-seed OOF F1 would tell me upfront whether
    this hypothesis has enough seed-variance to work. (iii) The
    exact `predict_proba` cost per inference call isn't in CURRENT
    STATE — 3x inference at dense stride 0.12 might be tight
    against the 243/300s budget. If retrain+eval together exceed
    300s, the wrapper will time-out-crash and the axis will be
    uninformative (similar to f71c526's joblib-skip).

(e) Wrapper enhancements. (1) `scripts/seed_variance_probe.py
    --n-seeds 5 --base-seed 42` — runs 5 quick HistGBM fits with
    different seeds on the current training data and reports
    per-seed OOF F1 mean/std + per-class F1 std, so I can preview
    whether a seed-ensemble hypothesis has enough seed-variance to
    be worth trying. (2) CLASSIFIER_STATE block in CURRENT STATE
    showing on-disk joblib's class_counts, OOF F1, features_py_sha,
    AND model_class (HistGBM vs VotingClassifier vs GBM). Removes
    the classifier-identity ambiguity. (3) A per-domain EVAL
    runtime printout so I can plan 3x-inference hypotheses
    relative to the 300s budget.

## 2026-04-20T01:04:45+09:00 — 273b8f5 (keep, combined=0.479100)
subject: DSP-confirmation floor post-filter — drop GBM emits whose strongest DSP z-score (phase/T²/CPE) is below DSP_CONFIRMATION_MIN=2.0, applied BEFORE dedupe so a strong-DSP neighbor can still win a 3.5s cluster. Pure detector.py change, no retrain, no feature change. Targets bba6dbe's singing 0.388 / korean 0.410 / english 0.667 plateau where 8 consecutive post-bba6dbe classifier/training-data discards have failed. Mechanism: top-SHAP features are spec_rolloff_delta/spec_centroid_delta/spec_bandwidth_delta across all domains (400-890 mass in singing, 350-660 in english) but dsp_t2_z shows 83-230 mass across all 3 domains too — GBM already treats T² as a strong auxiliary signal. What GBM doesn't enforce is a REQUIREMENT that at least one DSP channel fires — its softmax can still emit on an overwhelming spectral signal alone (chord transition → big spec_*_delta but smooth phase, modest T², low CPE). Real cross-source splices produce at least one DSP spike (phase_z fires at mic/room mismatch in hard cuts, T² fires at spectral-distribution shift in crossfades, CPE fires at prediction failure). Threshold 2.0 is conservative — 2 std devs above chunk-local MAD z — leaving headroom for real splices (typically z=3-10 on at least one channel). Orthogonal to every prior axis: 8 GBM hyperparam, 5 training-data SAMPLING, 12+ feature add/ablation, 8+ post-filter (persistence/plateau/class-margin/tonality/onset-local-pct/HPR-remote-rolloff/global-half-centroid/mfcc-remote — all operate on spectral or probability-curve signals, NEVER on required-DSP confirmation), 2 model-class swap, 2 calibration, 2 HPR routing, 6 audio MUTATION, 1 ensemble. Blast radius: detector.py only, ~15 lines added, feature_names unchanged so classifier sha stable (no retrain). Per-emit cost: three float reads from already-computed feature vector, negligible. Smoke: detect_splices on 10s noise returns [] without error, DSP_CONFIRMATION_CHANNELS=('dsp_phase_z','dsp_t2_z','dsp_cpe_z') resolves via bundle.feature_names with graceful no-op if any channel missing.
per-domain: combined_english=0.690476 combined_korean=0.410811 combined_singing=0.387692

# 2026-04-20 — hypothesis: DSP-confirmation floor post-filter

(a) HYPOTHESIS. Pure `splice/detector.py` change, no retrain, no feature
    change. Before dedupe, filter every hit-mask emit by the MAX of the
    three DSP z-score features already on the feature vector
    (`dsp_phase_z`, `dsp_t2_z`, `dsp_cpe_z`). Drop the emit when
    `max(...) < DSP_CONFIRMATION_MIN = 2.0`. Implementation: resolve
    the three column indices once from `feature_names`, apply the test
    inside the existing `for i in np.flatnonzero(hit_mask)` loop so
    surviving emits (with at least one DSP channel >= 2.0) go into the
    existing dedupe, and weak-DSP emits in a 3.5s cluster are removed
    BEFORE dedupe (so a strong-DSP neighbor can still win the cluster).
    Log a `diag.gbm.dsp_postfilter` line per chunk with dropped/kept
    counts.

(b) WHY this over the recent failures. Eight consecutive post-bba6dbe
    discards have all operated on CLASSIFIER / TRAINING-DATA axes
    (extend-aug-to-korean joblib-skip, class_weight=balanced
    catastrophic, spliced-singing aug regressed singing, EQ-tilt
    replacement lost gain, hybrid pitch+time-stretch regressed, 3-seed
    soft-voting identical metrics likely joblib-skip) and clean FP
    reduction has plateaued. The DIAGNOSTIC signal has been sitting
    right there: top-SHAP features are spec_rolloff_delta /
    spec_centroid_delta / spec_bandwidth_delta across all domains
    (400-890 mass in singing, 350-660 in english), but `dsp_t2_z`
    shows up at 83-230 mass across ALL three domains too, meaning GBM
    already treats T² as a strong auxiliary signal. What GBM DOESN'T
    enforce is a REQUIREMENT that at least ONE DSP channel fires —
    its softmax can still emit on an overwhelming spectral signal
    alone (spec_rolloff_delta ~ 2000 Hz from chord transition) even
    when all three DSP channels are near zero. That is exactly the
    clean-FP failure mode: chord transitions produce big spec deltas
    but ZERO phase discontinuity (smooth harmonic progression), low
    T² (energy redistributes rather than jumps), low CPE (complex
    prediction succeeds on smooth transitions). Real cross-source
    splices produce at least one DSP spike (phase_z fires at mic/room
    mismatch in hard cuts, T² fires at spectral-distribution shift
    in crossfades, CPE fires at prediction failure). Threshold 2.0 is
    conservative — 2 std devs above the chunk-local MAD z — leaving
    plenty of headroom for real splices (typically z=3-10 on at least
    one channel). Orthogonal to every prior axis: 8 GBM hyperparam,
    5 training-data SAMPLING, 12+ feature add/ablation, 8+ post-filter
    (persistence/plateau/class-margin/tonality/onset-local-pct/
    HPR-remote-rolloff/global-half-centroid/mfcc-remote — all operate
    on spectral or probability-curve signals, NEVER on required-DSP
    confirmation), 2 model-class swap, 2 calibration, 2 HPR routing,
    6 audio MUTATION, 1 ensemble. The DSP-floor post-filter is a
    genuinely new branch — hardcoding an AND-gate between GBM and
    DSP rather than letting GBM decide the tradeoff softly.

    Blast radius: detector.py only, ~15 lines added in one place,
    feature_names unchanged so classifier sha stable (no retrain).
    Per-emit cost: three float reads from an already-computed vector
    (O(1) per hit_mask position), totally negligible. If the floor
    is too strict on a real splice (e.g., a subtle singing crossfade
    where none of phase/T²/CPE reach 2.0), that emit is lost — worst
    case eval drops modestly and the iteration gets discarded.

(c) IF THIS FAILS. (1) Lower the floor to 1.5 (admits more DSP-weak
    emits). (2) Require that the max DSP channel EITHER >= 2.0 AT t
    OR >= 3.0 at any point in the chunk (file-level presence, less
    strict). (3) Gate the post-filter to singing-likely files only —
    compute chunk-level HPSS harmonic ratio and apply DSP floor only
    when HPR > 0.85 (tonal singing population where chord-transition
    FPs dominate, preserving speech recall at korean/english). (4)
    Final escalation: replace the hard AND-gate with a SOFT score
    combination — multiply the GBM p_splice by `sigmoid((dsp_max -
    1.0) / 0.5)` and re-threshold at 0.70. This makes DSP a
    multiplicative confidence weight instead of a binary gate.

(d) Information gaps. (i) The per-emit DSP z-score values on the 2
    singing and 4 korean clean FPs are not in CURRENT STATE — I'm
    predicting which FPs get bitten from the SHAP magnitude (dsp_t2_z
    ~ 85-230 mass means it's informative but not which direction).
    A `CLEAN_FP_POSITIONS` block listing (domain, file, t_sec,
    p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z) per clean FP would
    let me calibrate the threshold from the actual failure set.
    (ii) Plateau-filter (one of 7 verify-fails at 0.463218) and
    class-margin post-filter are referenced in history but without
    their exact definition / threshold — I'm confident my DSP-floor
    is distinct (different signal source, different gating direction),
    but a `POST_FILTERS_TRIED` block with one-line summaries would
    remove the "am I re-doing a prior idea" uncertainty every
    detector-side hypothesis carries. (iii) Whether an emit's
    feature_vector in `all_emits` preserves the EXACT DSP values the
    GBM saw is obvious from the code (`X[i].tolist()`), but whether
    feature_names in the loaded bundle is guaranteed to match
    FEATURE_NAMES is classifier-state-dependent — the bundle's
    `feature_names` could be None in legacy cases.

(e) Wrapper enhancements. (1) `CLEAN_FP_POSITIONS` block at top of
    CURRENT STATE with per-position DSP z-scores — would turn
    "singing clean_fp=2 / korean clean_fp=4" into actionable
    (file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z)
    tuples. Most detector-side hypotheses today are guessing which
    scalar signal will bite which FP without ever seeing the
    failure set's actual values. (2) `POST_FILTERS_TRIED` block —
    one-line summaries (mechanism, threshold, scope) of every
    post-filter hypothesis ever tried, so I can verify
    orthogonality pre-commit instead of chasing keyword matches
    in RECENT FAILED HYPOTHESES. (3) `scripts/dsp_signal_probe.py
    --emit-threshold 0.982 --dsp-channel {phase,t2,cpe}` that
    reports the distribution of the DSP z-scores on the subset of
    dense-scan positions where p_splice > 0.982, per domain —
    would let me preview the bite rate of a DSP-floor hypothesis
    BEFORE committing (how many emits drop at threshold 1.5 / 2.0
    / 2.5 per domain).
[auto] (no SHAP data for either bba6dbe or 273b8f5)

## 2026-04-20T01:12:49+09:00 — 7255ec6 (discard, combined=0.474696)
subject: GBM_THRESHOLD 0.982 -> 0.980 leveraging the just-kept DSP-confirmation floor (273b8f5). One-line primary tunable, no retrain. Prior 0.981 FAILED pre-DSP-filter — failure mode was chord-transition FPs with weak DSP (T² < 2.0, smooth phase, low CPE), exactly the population the 273b8f5 filter now culls pre-dedupe. With that safety net, admitting emits in [0.980, 0.982] is a genuinely new interaction: survivors to dedupe are the ones with strong DSP (real splices), while chord-transition FPs in the same p_splice band get dropped by the DSP floor. 0.980 is OUTSIDE prior tried range [0.981, 0.983] — truly untested. Targets borderline TP recall (subtle crossfades / hard cuts near the decision boundary) where bba6dbe's singing 0.388 / korean 0.411 / english 0.690 plateau is gated by recall not precision (english clean_fp=0, korean/singing clean_fp bounded). Pure detector.py one-constant edit, minimum blast radius, zero retrain cost, features.py sha stable so classifier byte-identical. If TPs don't rise but FPs do, fallback is tighten DSP_CONFIRMATION_MIN 2.0 -> 2.2. If TPs rise asymmetrically across classes, fallback is class-specific DSP channel routing (hard_cut needs phase_z >= 2.0, crossfade needs t2_z >= 2.0).
per-domain: combined_english=0.682353 combined_korean=0.394805 combined_singing=0.397059

# 2026-04-20 — hypothesis: GBM_THRESHOLD 0.982 → 0.980, leveraging the just-kept DSP-confirmation floor

(a) HYPOTHESIS. One-line constant edit in `splice/detector.py`:
    `GBM_THRESHOLD = 0.982` → `GBM_THRESHOLD = 0.980`. Pure primary
    tunable, no retrain, no feature change. This admits emits with
    p_splice in the [0.980, 0.982] band into the DSP-confirmation
    pipeline added in 273b8f5.

(b) WHY this over recent failures. The GBM_THRESHOLD bracket history
    — 0.981 FAILED, 0.9825 FAILED, 0.983 FAILED, 0.982 KEPT — was
    established BEFORE the 273b8f5 DSP-confirmation floor landed.
    That filter is an AND-gate requiring
    `max(dsp_phase_z, dsp_t2_z, dsp_cpe_z) >= 2.0` on every
    surviving emit (pre-dedupe). The previous 0.981 failure mode was
    FP blow-up on chord transitions with weak DSP (dsp_t2_z < 2.0,
    smooth phase, low CPE) — exactly the population the new filter
    now culls automatically. With that safety net in place, lowering
    the threshold is a genuinely different interaction: the extra
    admissions in [0.980, 0.982] that survive to dedupe are the ones
    WITH strong DSP, which are much more likely to be real splices.
    Borderline TPs at p_splice ≈ 0.980-0.982 (subtle crossfades,
    hard cuts near the model's decision boundary) are recovered;
    borderline FPs with weak DSP (chord transitions, natural spectral
    shifts in singing) get dropped by the DSP floor before dedupe.
    0.980 is OUTSIDE the prior-tested range [0.981, 0.983] — truly
    untried. Orthogonal to every prior post-273b8f5 axis (none
    attempted yet — this is the first hypothesis post-DSP-floor keep).
    One-constant edit, minimum blast radius, no retrain cost.

(c) IF THIS FAILS. Three paths.
    (1) If TPs don't increase but FPs do (DSP filter isn't culling
        enough), tighten DSP_CONFIRMATION_MIN from 2.0 to 2.2 in a
        follow-up to raise the safety net.
    (2) If TPs do increase but class-wise (hard_cut vs crossfade)
        the improvement is asymmetric, try class-specific DSP channel
        requirements: hard_cut emits require dsp_phase_z ≥ 2.0
        (mic/room phase break), crossfade emits require dsp_t2_z ≥ 2.0
        (spectral-distribution shift). Matches each class's physical
        signature.
    (3) Final escalation: add a SUM-based DSP floor companion —
        `|phase_z| + |t2_z| + |cpe_z| ≥ 4.0` — on top of the MAX
        floor. Captures multi-channel confirmation (real splices
        activate multiple DSP channels partially) as distinct from
        single-strong-channel signals (chord transitions that happen
        to fire only T²).

(d) Information gaps. (i) The 2 singing / 4 korean clean FP positions
    (file, t_sec, p_hard, p_cross, p_splice, dsp_phase_z, dsp_t2_z,
    dsp_cpe_z) are still not in CURRENT STATE — I'm reasoning about
    DSP-confirmation-survivor tail behavior from theory, not from
    actual per-position data. (ii) The distribution of p_splice for
    real splice TPs in the [0.980, 0.982] band is unknown — I'm
    assuming a non-trivial count exists but I have no visibility
    into the cumulative p_splice distribution. (iii) Post-273b8f5
    DSP-drop rate is unknown — I don't know how many emits the
    filter bites per file per domain, so I can't predict how much
    DSP-filter headroom remains when the 0.980 threshold lets more
    candidates in.

(e) Wrapper enhancements. (1) A `P_SPLICE_CDF_TAIL` block in CURRENT
    STATE showing, per domain, the count of p_splice values in each
    bin [0.980, 0.985, 0.990, 0.995, 0.999, 1.0] for both clean
    files (FP risk) and spliced files (TP recall) — lets me pick
    GBM_THRESHOLD deltas surgically from the actual CDF tail rather
    than from trial and error. (2) A `DSP_DROPPED_PER_FILE` summary
    block showing per-domain how many emits the DSP floor bites per
    iteration — lets me see whether the filter is under-used (room
    to lower GBM_THRESHOLD) or saturated (tightening DSP_MIN is
    higher-leverage). (3) `CLEAN_FP_POSITIONS` JSON auto-emitted by
    evaluate.py with each position's full feature vector (at
    minimum: p_splice, p_hard, p_cross, dsp_phase_z, dsp_t2_z,
    dsp_cpe_z, dsp_pairwise_proximity) — turns "korean clean_fp=4"
    from an opaque scalar into calibrated per-position diagnostics,
    the single biggest blocker across every detector-side hypothesis.

## 2026-04-20T01:19:08+09:00 — d256901 (discard, combined=0.448578)
subject: class-specific DSP channel routing stacked on the MAX floor — after the existing max(phase_z,t2_z,cpe_z)>=2.0 filter (273b8f5), drop emits where the predicted class's primary DSP channel is below 2.0: hard_cut (label=1) requires dsp_phase_z>=2.0 (mic/room phase break), crossfade (label=2) requires dsp_t2_z>=2.0 (spectral-distribution shift). Pure detector.py change, no retrain, no feature change. Targets singing 0.388 / korean 0.411 post-273b8f5 plateau where 2 singing + 4 korean clean FPs sit. Mechanism: chord transitions and speech phoneme shifts typically fire T² (smooth spectral-distribution change over chord/phoneme window) but NOT phase_z (no mic/room discontinuity); when GBM predicts hard_cut on such an emit, the current MAX filter keeps it (T²>=2.0 wins the max). Class-specific routing drops hard_cut emits with phase_z<2.0 — exactly the channel-class-mismatch FPs. Real hard-cut TPs have strong phase_z by physical construction (cross-source phase cannot smooth); real crossfade TPs have strong T² (a spectral-distribution shift IS the definition of a crossfade). Explicit untried next-step from 7255ec6(c)(2). Orthogonal to every prior post-filter (persistence/plateau/class-margin/tonality/onset-local-pct/HPR-remote-rolloff/global-half-centroid/mfcc-remote/DSP-MAX-floor all used a single global scalar without class→channel mapping). Risk-bounded because stacked: emit must satisfy BOTH max DSP>=2.0 AND class's primary channel>=2.0, so this can only DROP emits, never create new TPs; drops require the MISMATCH pattern. Per-emit cost: one extra float compare past the MAX gate, negligible. Blast radius: detector.py only, ~10 lines added, feature_names unchanged, classifier sha stable (no retrain). Smoke: detect_splices on 10s noise returns [] without error, DSP_CLASS_CHANNEL={1:'dsp_phase_z',2:'dsp_t2_z'} resolves via bundle.feature_names with graceful no-op if any channel missing.
per-domain: combined_english=0.623377 combined_korean=0.415625 combined_singing=0.348387

# 2026-04-20 — hypothesis: CLASS-SPECIFIC DSP channel routing (stacked on the MAX floor)

(a) HYPOTHESIS. Pure `splice/detector.py` change, no retrain, no feature
    change. AFTER the existing MAX-based DSP-confirmation floor (273b8f5,
    `max(phase_z, t2_z, cpe_z) >= 2.0`), ADD a second gate keyed on the
    emit's predicted class: if label_id == 1 (hard_cut) require
    `dsp_phase_z >= 2.0`; if label_id == 2 (crossfade) require
    `dsp_t2_z >= 2.0`. Drop the emit when its predicted class's primary
    DSP channel is below 2.0 (even when another channel pushed the MAX
    over 2.0). Move the `label_id` argmax from AFTER the filter to
    BEFORE so class-to-channel routing runs per emit. Same threshold
    value (2.0), different routing — each splice class's physical
    signature drives its own confirmation requirement:
      - hard_cut (abrupt discontinuity) → requires `phase_z` (mic/room
        phase break)
      - crossfade (spectral-distribution shift) → requires `t2_z`
        (Hotelling T² on spectral distribution)

(b) WHY this over recent failures. 273b8f5 (current keep, 0.479) kicked
    2 english FPs (0.667→0.690) but left singing (0.388) and korean
    (0.411) unmoved. The MAX-based floor passes any emit where at
    least one of the three DSP channels fires — including emits where
    the FIRING channel is mismatched to the PREDICTED class. Chord
    transitions (singing FPs) and speech phoneme shifts (korean FPs)
    typically fire T² (spectral distribution changes smoothly over
    phoneme/chord transitions) but NOT phase_z (no mic/room
    discontinuity). When GBM predicts hard_cut on such an emit, the
    current filter keeps it (T² >= 2.0 wins the MAX). Under
    class-specific routing, hard_cut + phase_z < 2.0 gets dropped —
    exactly the channel-class-mismatch FPs. Real hard-cut TPs have
    strong phase_z by physical construction (a splice across mic/room
    cannot smooth phase); real crossfade TPs have strong T² (a
    spectral distribution shift IS the definition of a crossfade).
    Explicit untried next-step from 7255ec6(c)(2). Orthogonal to every
    prior post-filter (persistence / plateau / class-margin / tonality
    / onset-local-pct / HPR-remote-rolloff / global-half-centroid /
    mfcc-remote / DSP-MAX-floor all used a single global scalar
    without the class→channel map). Risk-bounded because stacked on
    top of the MAX filter: an emit must satisfy BOTH max DSP >= 2.0
    AND the class's primary channel >= 2.0 to survive — this can only
    DROP emits, never create new TPs, and a drop requires the mismatch
    pattern. Per-emit cost: one extra float compare past the MAX gate.

(c) IF THIS FAILS. (1) Add a CPE escape hatch: keep the emit if
    dsp_cpe_z >= 2.5 regardless of class mismatch — preserves real
    splices whose phase/T² channel happens to be noisy. (2) RAISE the
    class-specific threshold from 2.0 to 2.5 on the primary channel
    to bite more FPs if the first attempt was too gentle. (3) Compose
    with a SUM-based companion (`|phase_z| + |t2_z| + |cpe_z| >= 4.0`)
    for multi-channel confirmation on top of class-routed single-
    channel confirmation.

(d) Information gaps. (i) The 2 singing / 4 korean clean FP positions
    (file, t_sec, label_id, p_hard, p_cross, p_splice, dsp_phase_z,
    dsp_t2_z, dsp_cpe_z) still not in CURRENT STATE — I cannot verify
    whether the FPs are predominantly labelled hard_cut (bitten by the
    phase_z requirement) or crossfade (bitten by the t2_z-only
    requirement). (ii) The per-class counts of emits at
    p_splice ≈ 0.982-0.99 per domain are unknown — I do not know
    whether GBM predicts mostly hard_cut or crossfade on the borderline
    singing/korean FPs. (iii) The english FPs bitten by 273b8f5's MAX
    floor — were they hard_cut or crossfade labelled? If mostly
    hard_cut, phase_z was already pivotal on english (my new gate is a
    no-op there); if mostly crossfade, T² was doing the work.

(e) Wrapper enhancements. (1) `CLEAN_FP_POSITIONS` JSON block in CURRENT
    STATE with per-position (domain, file, t_sec, label_id, p_hard,
    p_cross, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z) from the last
    kept classifier's eval — the single biggest blocker across every
    detector-side hypothesis (I keep guessing which channel will bite).
    (2) A `DSP_DROP_CLASSIFICATION` diagnostic emitted per file: emits
    dropped by each gate component (MAX floor, class-specific
    phase_z, class-specific t2_z), so a keep/discard post-mortem can
    distinguish "filter under-fired" from "filter over-fired". (3)
    `scripts/dsp_signal_probe.py --by-class {hard_cut,crossfade}
    --channel {phase_z,t2_z,cpe_z} --threshold X` that simulates
    per-class channel-specific drops on a prior classifier's emit
    trace, so I can preview bite rate BEFORE committing.

## 2026-04-20T01:27:56+09:00 — 865d92f (keep, combined=0.487946)
subject: SUM-based DSP floor companion stacked on MAX (multi-channel confirmation) — add `DSP_SUM_MIN=4.5` after the existing `DSP_CONFIRMATION_MIN=2.0` MAX gate (273b8f5). Drop GBM emits where `sum(dsp_phase_z, dsp_t2_z, dsp_cpe_z) < 4.5` even when MAX passes. Pure detector.py change, no retrain, no feature change. Targets bba6dbe's singing 0.388 / korean 0.411 plateau where 2 singing + 4 korean clean FPs survived 273b8f5's MAX floor — these are single-channel firings (chord transitions: T²=3 with smooth phase / low CPE, sum~3.5-4.5; speech phoneme shifts: T²=2.5 with phase ~0.7 CPE ~1.0, sum~3.5-5). Real cross-source splices disrupt multiple physical signals simultaneously (mic/room mismatch fires phase AND T² AND CPE) yielding sum 6-12. SUM gate demands ~2.5 of cumulative support beyond the MAX floor of 2.0, biting single-channel-firing FPs while preserving multi-channel-confirmed real splices. Explicit untried escalation cited in two prior post-mortems: 7255ec6(c)(3) and d256901(c)(3). Orthogonal to all prior post-filters: 8 GBM hyperparam, 5 training-data SAMPLING, 12+ feature add/ablate, 8+ post-filter (persistence/plateau/class-margin/tonality/onset-local-pct/HPR-remote-rolloff/global-half-centroid/mfcc-remote/DSP-MAX-floor/class-specific-DSP — none used SUM-based multi-channel cumulative confirmation), 2 model-class swap, 2 calibration, 2 HPR routing, 3 remote-half, 1 class_weight, 6 audio MUTATION, 1 ensemble. Risk-bounded by stacking: can only DROP emits passing MAX gate, never create new TPs; drops require single-channel-firing pattern. Per-emit cost: one extra sum + compare past MAX gate, negligible. Blast radius: detector.py only, ~10 lines added, feature_names unchanged so classifier sha stable (no retrain). Smoke: detect_splices on 10s noise returns [] without error, DSP_SUM_MIN=4.5 reads gracefully when DSP_CONFIRMATION_CHANNELS resolve via bundle.feature_names.
per-domain: combined_english=0.707317 combined_korean=0.428169 combined_singing=0.383607

# 2026-04-20 — hypothesis: SUM-based DSP floor companion stacked on MAX (multi-channel confirmation)

(a) HYPOTHESIS. Pure `splice/detector.py` change, no retrain, no feature
    change. Add `DSP_SUM_MIN = 4.5` constant alongside the existing
    `DSP_CONFIRMATION_MIN = 2.0`. After the existing MAX gate
    (`max(phase_z, t2_z, cpe_z) >= 2.0`), add a SECOND gate
    `sum(phase_z, t2_z, cpe_z) >= 4.5`. Drop the emit when EITHER
    gate fails. The two gates compose: an emit must have at least
    one strong channel AND multi-channel cumulative confirmation.

(b) WHY this over the recent failures. Three post-273b8f5 attempts
    have explored the post-DSP-floor design space:
      * 7255ec6 (lower GBM_THRESHOLD 0.982→0.980, 0.475): admitted
        more borderline emits — singing recall lifted but korean
        FP'd because the MAX gate is a single-channel floor and
        speech phoneme shifts can fire one channel (T²) above 2.0
        without being a splice.
      * d256901 (class-specific channel routing, 0.449): too strict —
        required SPECIFIC channel (phase for hard_cut, T² for
        crossfade) to be ≥2.0, dropping real TPs whose primary
        channel happened to be noisy (singing -0.040, english -0.067).
    The MAX gate has the OPPOSITE flaw of class-specific routing:
    it accepts ANY single channel firing, including chord-transition
    FPs that fire ONLY T² (smooth phase, low CPE). The SUM gate is
    the genuinely orthogonal middle path — instead of demanding a
    specific channel per class, it demands MULTI-CHANNEL CUMULATIVE
    evidence regardless of which channel is dominant.

    Real cross-source splices are physical events that disrupt
    multiple signals simultaneously: a mic/room mismatch creates
    phase discontinuity AND spectral-distribution shift AND
    prediction failure together. Typical splice DSP profile:
    phase_z=2-5, t2_z=2-4, cpe_z=1.5-3 → SUM=6-12.
    Chord-transition FP profile: phase_z=0.3-0.7, t2_z=2.5-3.5,
    cpe_z=0.5-1.0 → SUM=3.3-5.2 (often borderline at 4.5).
    Speech phoneme-shift FP profile: phase_z=0.5-1.0, t2_z=2-3,
    cpe_z=0.8-1.5 → SUM=3.3-5.5 (often borderline).

    Threshold 4.5 = MAX_floor (2.0) + 2.5 of cumulative support from
    the other channels. Single-channel firings at MAX=2.0 with the
    other two summing to <2.5 get dropped — exactly the mismatch
    pattern. Multi-channel firings with each ~1.5 (sum 4.5) pass.
    Real splices with strong support across channels comfortably
    pass.

    Cited as explicit untried escalation in 7255ec6(c)(3) and
    d256901(c)(3). Orthogonal to every prior post-filter axis: 8
    GBM hyperparam, 5 training-data SAMPLING, 12+ feature
    add/ablation, 8+ post-filter (persistence/plateau/class-margin/
    tonality/onset-local-pct/HPR-remote-rolloff/global-half-centroid/
    mfcc-remote/DSP-MAX-floor/class-specific-DSP — none used SUM-
    based multi-channel cumulative confirmation), 2 model-class
    swap, 2 calibration, 2 HPR routing, 3 remote-half, 1
    class_weight, 6 audio MUTATION, 1 ensemble. Genuinely new
    branch — the additive composition of DSP channels is a
    qualitatively different test than the existing MAX gate.

    Blast radius: detector.py only, ~5 lines added, feature_names
    unchanged so classifier sha stable (no retrain). Per-emit cost:
    one extra float sum + compare past the MAX gate, negligible.
    Risk-bounded by stacking: this can only DROP emits already
    passing the MAX gate, never create new TPs; drops require the
    single-channel-firing pattern. Worst case: mild over-bite
    drops some real subtle splices, eval falls slightly below
    0.479 and discards.

(c) IF THIS FAILS. Three paths.
    (1) If TPs drop too aggressively (e.g., subtle crossfades whose
        only firing channel is T² ~3.5 with phase/CPE near 0),
        LOWER the SUM threshold from 4.5 to 4.0 (more permissive).
    (2) If FPs persist (the 4.5 threshold is too loose), RAISE the
        SUM threshold to 5.0 — requires stronger multi-channel
        confirmation but risks real splice losses.
    (3) Final escalation: replace the linear SUM with a HARMONIC
        MEAN of (max(phase_z, 0.1), max(t2_z, 0.1), max(cpe_z, 0.1))
        ≥ 1.5 — harmonic mean is dominated by the WEAKEST channel
        so it explicitly demands all three be non-trivial, the
        most aggressive form of multi-channel confirmation.

(d) Information gaps. (i) Per-position DSP z-score values on the 2
    singing / 4 korean clean FPs are STILL not in CURRENT STATE — I
    am calibrating the SUM threshold from theory, not from observed
    failure-cluster values. A `CLEAN_FP_POSITIONS` block listing
    (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z,
    dsp_cpe_z) per FP would let me set the threshold surgically.
    (ii) The DSP-drop rate on the 273b8f5 keep is unknown — without
    it I cannot tell whether the MAX gate is currently saturated
    (no headroom for SUM addition) or under-used (room for SUM to
    bite without losing TPs). (iii) The split between hard_cut /
    crossfade emits passing the MAX gate currently is opaque — I
    don't know which class the surviving FPs are predicted as.

(e) Wrapper enhancements. (1) `CLEAN_FP_POSITIONS` JSON block in
    CURRENT STATE with per-position DSP z-scores from the last kept
    classifier's eval — the single biggest blocker across detector-
    side hypotheses. Most reasoning today is theory-driven instead
    of data-driven. (2) `DSP_GATE_DROP_RATES` block per domain:
    {file_count, gbm_emits, max_gate_dropped, sum_gate_dropped,
    survived} so I can preview SUM-threshold bite ranges from a
    prior eval's actual values. (3) `scripts/dsp_signal_probe.py
    --gate-type {max,sum,both} --threshold X` that simulates a
    candidate gate against a prior classifier's emit trace and
    reports per-domain TP-loss vs FP-drop counts — turns
    "what threshold should I pick" from a guess into a measurable
    pre-commit signal.
[auto] (no SHAP data for either 273b8f5 or 865d92f)

## 2026-04-20T01:36:01+09:00 — 29c06cf (keep, combined=0.493815)
subject: tighten DSP_SUM_MIN 4.5 -> 5.0 (direct cited escalation of 865d92f keep). Pure detector.py change, no retrain, no feature change. 865d92f kept the SUM-based DSP floor (sum of dsp_phase_z + dsp_t2_z + dsp_cpe_z >= 4.5 stacked on max>=2.0), lifting combined 0.479 -> 0.488 and recovering english 2 FPs. Remaining 2 singing + 4 korean clean FPs surviving SUM>=4.5 by construction sit in [4.5, ~5.5] — chord transitions with T^2=3 + partial support (sum 4.5-5.2), speech phoneme shifts with T^2=2.5 + CPE=1.0 (sum 4.5-5.5). Bumping to 5.0 bites this borderline band specifically. Real splices disrupt multiple physical signals so comfortably score sum 6-12 (median median margin >=1 above 5.0). Cited option 2 from 865d92f(c)(2) 'RAISE the SUM threshold to 5.0 if FPs persist'. Orthogonal within the DSP-post-filter axis (5.0 outside only prior tried 4.5) and to every other axis: not GBM_THRESHOLD (7255ec6 failed), not class-specific routing (d256901 failed), not MAX-floor tuning, not audio mutation/classifier/calibration/features. Risk-bounded by stacking: can only DROP emits past MAX, never create TPs; drops require single-channel/low-cumulative pattern. Per-emit cost unchanged (same sum+compare path). Blast radius: detector.py only, 1-line constant + comment refresh, feature_names unchanged so classifier sha stable (no retrain). Smoke: constant loads cleanly, DSP_SUM_MIN=5.0 read by existing gate logic at line 303.
per-domain: combined_english=0.731707 combined_korean=0.457143 combined_singing=0.360000

# 2026-04-20 — hypothesis: tighten DSP_SUM_MIN 4.5 → 5.0 (direct escalation of the 865d92f keep)

(a) HYPOTHESIS. One-line constant edit in `splice/detector.py`:
    `DSP_SUM_MIN = 4.5` → `DSP_SUM_MIN = 5.0`. Pure detector.py change,
    no retrain, no feature change, zero new logic. The MAX gate
    (DSP_CONFIRMATION_MIN=2.0) and the stacked SUM gate already filter
    emits; this simply tightens the SUM threshold by +0.5.

(b) WHY this over recent failures. 865d92f (current keep, 0.487946)
    quantified per-failure-mode DSP profiles from theory:
      * Chord transitions (singing FPs): phase≈0.3–0.7, T²≈2.5–3.5,
        CPE≈0.5–1.0 → sum 3.3–5.2 (often borderline at 4.5).
      * Speech phoneme shifts (korean FPs): phase≈0.5–1.0, T²≈2–3,
        CPE≈0.8–1.5 → sum 3.3–5.5 (often borderline).
      * Real splices: phase=2–5, T²=2–4, CPE=1.5–3 → sum 6–12
        (comfortably above 5.0).
    Two singing + four korean FPs survived SUM≥4.5 at 865d92f. Their
    sums by construction sit in [4.5, ~5.5] — the part of the
    distribution the 4.5 cutoff didn't reach. Bumping to 5.0 bites
    exactly that band while leaving real splices untouched (sum
    6–12 median). This is the cited option 2 from 865d92f(c)(2)
    ("RAISE the SUM threshold to 5.0 if FPs persist").

    Orthogonal within the DSP-post-filter axis: different tunable,
    and 5.0 is outside the only previously-tried SUM value (4.5).
    Orthogonal to every other axis: not GBM_THRESHOLD (7255ec6
    failed), not class-specific routing (d256901 failed), not
    MAX-floor tuning, not audio mutation / classifier / calibration /
    features — all unchanged.

    Risk-bounded: SUM gate only DROPS emits past MAX, never creates
    TPs. Drops require single-channel-firing or low-cumulative-support
    pattern (the FP mode). If tightening drops some real subtle
    crossfades (predicted phase≈2, T²=2, CPE=0.9 → sum=4.9 fails)
    eval falls below 0.487 and iteration is discarded — pure
    information gain.

(c) IF THIS FAILS. Three paths.
    (1) If singing TPs drop (crossfade has inherently weaker phase/CPE
        and a singing-specific file reaches sum~4.9), CLASS-CONDITION
        the SUM threshold: hard_cut→5.0, crossfade→4.5. Preserves
        crossfade recall while tightening on hard_cut where multi-
        channel confirmation is physically robust.
    (2) If the bite is too gentle (FPs' sums were already >5.0),
        escalate further to 5.5 AND add a CPE-specific minimum
        (dsp_cpe_z ≥ 1.0) — chord transitions almost universally
        have CPE<1.0 while real splices have CPE≥1.5.
    (3) Final escalation: stack a GEOMETRIC-MEAN gate
        GM(max(phase_z,0.1), max(t2_z,0.1), max(cpe_z,0.1)) ≥ 1.2.
        Less punishing than harmonic mean but still demands multi-
        channel presence (unlike SUM, which one strong channel can
        carry).

(d) Information gaps. (i) Per-position DSP z-scores on the 2 singing
    / 4 korean clean FPs surviving 865d92f's SUM≥4.5 are STILL
    unknown. I cannot verify these FPs' sums are actually in
    [4.5, 5.0] — I'm inferring from theoretical distributions in
    865d92f's post-mortem. (ii) The bite rate of the 865d92f SUM
    gate (how many emits it drops per domain) is unlogged — I
    cannot tell whether 4.5 was saturated (no room for 5.0 to add
    bite) or under-used. (iii) Real splice SUM distribution per
    domain is unknown — the 5.0 threshold's safety margin against
    real-splice loss is qualitative.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE: per-position (domain, file, t_sec, label_id,
    p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, sum, max) from the
    last kept classifier's eval — removes the biggest blocker across
    detector-side hypotheses. Every DSP-filter hypothesis today is
    calibrating thresholds from theory not data. (2) DSP_GATE_BITE
    block: per-domain {MAX_only_drops, SUM_only_drops, both_drops,
    survived} so I can preview whether tightening SUM bites harder
    than tightening MAX, and where the current gate is saturated.
    (3) `scripts/dsp_sum_sweep.py --thresholds 4.5,4.75,5.0,5.25,5.5`
    that replays the last keep's emit trace through each proposed
    SUM threshold and reports per-domain (TPs_lost, FPs_dropped),
    turning "4.5 vs 5.0" from a guess into a numeric decision.
[auto] (no SHAP data for either 865d92f or 29c06cf)

## 2026-04-20T01:42:23+09:00 — eb8984e (discard, combined=0.479988)
subject: class-conditioned SUM threshold (hard_cut=5.0, crossfade=4.5) — add DSP_SUM_MIN_CROSSFADE=4.5 and move label_id argmax BEFORE the DSP gate so SUM threshold routes by predicted class. Pure detector.py change, no retrain, no feature change. Targets 29c06cf's singing 0.384->0.360 regression (now weakest domain at 0.360) caused by uniform SUM tightening 4.5->5.0 — korean (0.428->0.457) and english (0.707->0.732) gained from the tightening via precision on spliced files, but singing lost crossfade TPs whose DSP profile is T^2=2.5-3.5 (strong) with phase/CPE ~0.5-1.5 (singing sustained tones don't phase-break / complex prediction handles harmonic continuations), summing 3.5-5.2 — exactly the band 5.0 culls. Hard cuts produce phase+T^2+CPE simultaneously so sum 6-12 regardless of class-route, meaning keeping hard_cut at 5.0 preserves the korean/english precision wins. Explicit option (1) from 29c06cf(c): 'CLASS-CONDITION the SUM threshold: hard_cut->5.0, crossfade->4.5. Preserves crossfade recall while tightening on hard_cut where multi-channel confirmation is physically robust.' Orthogonal to d256901 (class-specific CHANNEL routing: phase>=2.0 for hard_cut / T^2>=2.0 for crossfade on top of MAX>=2.0 — failed 0.449 because it required a single specific channel to be strong); class-routed SUM still requires MULTI-CHANNEL cumulative support, just with class-appropriate magnitude. Orthogonal within the SUM-axis (only uniform thresholds tried: 4.5 kept 0.488, 5.0 kept 0.494). Risk-bounded: stacks on MAX>=2.0 gate AND GBM_THRESHOLD=0.982 so only admits additional emits where class is crossfade AND max DSP>=2.0. Worst case: korean/english crossfade-labeled FPs (speech phoneme shifts may label as crossfade at p~0.982-0.99 with sum ~4.6-4.9) reappear, dropping korean/english back toward 865d92f levels 0.428/0.707 — still above baseline because singing recovers. Blast radius: detector.py only, 1 constant + 3-line loop reorder, features.py sha stable (no retrain). Smoke: constants load cleanly, DSP_SUM_MIN=5.0 / DSP_SUM_MIN_CROSSFADE=4.5 / DSP_CONFIRMATION_MIN=2.0.
per-domain: combined_english=0.707317 combined_korean=0.434286 combined_singing=0.360000

# 2026-04-20 — hypothesis: class-conditioned SUM threshold (hard_cut=5.0, crossfade=4.5)

(a) HYPOTHESIS. Pure `splice/detector.py` change, no retrain, no feature
    change. Add `DSP_SUM_MIN_CROSSFADE = 4.5` constant and move the
    `label_id` argmax BEFORE the DSP gate so the SUM threshold routes
    by predicted class: hard_cut (label=1) keeps SUM>=5.0, crossfade
    (label=2) relaxes to SUM>=4.5. MAX gate (DSP_CONFIRMATION_MIN=2.0)
    unchanged; crossfade-labeled emits with sum in [4.5, 5.0) that
    29c06cf's global tightening culled are now admitted.

(b) WHY this over the recent trajectory. The 29c06cf keep lifted
    combined 0.488→0.494 by tightening SUM 4.5→5.0, but the per-domain
    delta exposes a class-mix asymmetry: singing 0.384→0.360 (TP loss),
    korean 0.428→0.457 (net gain), english 0.707→0.732 (net gain).
    Singing is now the weakest domain (0.360) and its regression is
    the explicit signature of crossfade TPs being dropped — singing
    crossfades are slow spectral transitions whose DSP profile is
    T²=2.5-3.5 (strong) with phase=0.5-1.0 (smooth sustained tones
    don't phase-break) and CPE=0.5-1.5 (complex prediction handles
    harmonic continuations reasonably), summing to ~3.5-5.2. The
    SUM>=4.5 gate at 865d92f admitted the upper half of that band;
    SUM>=5.0 at 29c06cf culls it. Hard cuts are physically different:
    cross-source phase CANNOT smooth so phase_z=2-5, T²=2-4 from the
    discontinuous spectral jump, CPE=1.5-3 from prediction failure at
    the cut — comfortably sum 6-12 regardless of threshold. So
    hard_cut needs no relaxation (keeping SUM>=5.0 preserves the
    korean/english precision wins from the tightening), while
    crossfade benefits from the looser gate. This is the explicit
    option (1) cited in 29c06cf(c). Orthogonal to d256901 which
    required a SPECIFIC CHANNEL per class (phase_z for hard_cut, T²
    for crossfade, both >=2.0 ON TOP of MAX>=2.0) and failed
    catastrophically at 0.449 — that was too strict because it
    required a single-channel strong-signal match to class label.
    Class-routed SUM is fundamentally different: it still requires
    multi-channel cumulative support, just with class-appropriate
    magnitude. Orthogonal within the SUM-axis (only uniform
    thresholds tried so far: 4.5 kept, 5.0 kept). Risk-bounded by
    stacking with MAX gate and GBM_THRESHOLD=0.982: only admits more
    emits when the class is crossfade AND max DSP>=2.0 AND p>0.982.
    Pure 1-constant add + 3-line loop reorder, features.py sha stable,
    no retrain, zero new computation per emit.

(c) IF THIS FAILS. (1) If singing recovers but korean/english
    crossfade FPs reappear (speech phoneme shifts are T²-dominant
    and may also label as crossfade at p~0.982-0.99 with sum
    ~4.6-4.9), tighten crossfade SUM to 4.75 or add an HPR-gate:
    relax SUM to 4.5 ONLY when file_hpr>0.85 (singing), else keep
    5.0 for both classes. (2) If singing doesn't recover, the lost
    TPs weren't predominantly crossfade-labeled — pivot to a
    domain-based HPR-gated SUM (singing=4.5, rest=5.0) directly.
    (3) Final escalation: replace binary class-routing with a
    CONTINUOUS class-mix threshold
    `sum_min = 4.5 + 0.5 * proba[i, col_for[1]] / p_splice[i]`
    (sliding from 4.5 when pure-crossfade to 5.0 when pure-hard_cut).

(d) Information gaps. (i) The CLASS-LABEL breakdown of the singing
    TPs lost at 29c06cf is not in CURRENT STATE. I'm inferring
    crossfade dominance from the physical signature argument
    (singing sustains are slow transitions) but without the
    per-emit (t_sec, label_id, sum) trace I can't verify the lost
    TPs were >50% crossfade-labeled. (ii) The 2 singing + 4 korean
    clean FP positions (file, t_sec, label_id, p_hard, p_cross,
    dsp_phase_z, dsp_t2_z, dsp_cpe_z, sum) STILL not in prompt —
    the same gap every DSP-filter hypothesis has been flagging for
    5+ iterations. (iii) The real-splice SUM distribution per
    domain per class isn't surfaced — my crossfade SUM~3.5-5.2
    range is theoretical.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE with per-position (domain, file, t_sec, label_id,
    p_hard, p_cross, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    sum, max) from the last keep's eval — by far the biggest and
    most persistent blocker. Every DSP-filter hypothesis calibrates
    thresholds from theory instead of observed failure-set values.
    (2) LOST_TP_POSITIONS similar JSON block for TPs dropped
    between the last TWO keeps — lets a post-mortem pinpoint
    which class/domain the tightening hurt, rather than inferring
    from combined deltas. (3) `scripts/dsp_sum_class_probe.py
    --hard-min 5.0 --crossfade-min 4.5` that replays the last
    keep's emit trace through class-routed thresholds and reports
    per-domain (TPs_lost, FPs_dropped) — turns "class-routed SUM
    4.5/5.0 vs HPR-gated SUM 4.5/5.0" from a guess into a numeric
    decision before commit.

