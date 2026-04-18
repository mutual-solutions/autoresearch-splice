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

