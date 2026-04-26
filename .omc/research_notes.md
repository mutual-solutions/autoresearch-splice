## Historical digest (oldest 20 entries compacted)
Span: Historical digest (oldest 20 entries compacted) … 2026-04-21T03:08:53+09:00 — 9274ace (discard, combined=0.443912)
Outcomes: 0 keep / 19 discard / 0 verify-fail; combined range [0.2867, 0.5441].
(Older reflections collapsed to conserve prompt budget.)

## 2026-04-21T03:24:07+09:00 — b5b1a0d (discard, combined=0.556428)
subject: add mfcc_post_retrospective_match (FEATURE_NAMES 80->81) -- min cos_dist from post[t,t+2] to past pre-context windows [t-k-2,t-k] for k in {3,6,9,12}s. STRUCTURAL pivot off the 60+ PRE-vs-POST-at-single-boundary family onto RETROSPECTIVE CONTEXT MATCHING. Chord-cycle FP: post chord already appeared at some past k -> min small -> silent, FP not boosted. Cross-song splice: post is song B absent from entire pre-history -> min large -> fires. Speech self-gating: within-recording past and post share speaker+mic+room so all distances small -> min small -> feature ~0 on non-splice -> GBM low per-domain SHAP. Reuses cached feat_mfcc; ZERO new librosa calls, ZERO new caches. Edge guard t-14<0 OR t+2>duration -> sentinel 0.0. FEATURE_NAMES 80->81 forces retrain via US-505 sha gate. Orthogonal: NOT any paired-diff/voicing-masked single-boundary feature; NOT 9064eec cross-intra (intra sub-windows on SAME span, no past-history bank, no MIN aggregator); NOT 4e67946 persistence (future horizons); NOT 4314449 cross-scale; NOT 1b4fe7c corr-matrix; NOT 1b3eb06 trajectory velocity; NOT 1e57702 histogram; NOT DSP-gate. FIRST min-over-past-context feature in 80-set, FIRST feature using past pre-history as comparison bank. Smoke-verified: len(FEATURE_NAMES)==81 last-name correct; synthetic chord-cycle A(3s period) + song-B splice at t=30 yields 0.018437 vs within-A chord-cycle t=18 yields 0.000005 (~3700x discrimination); edge guards t-14<0 and t+2>duration both return 0.0 sentinel correctly; real singing train audio 200 calls = 0.82ms/call; all 81 features finite; idempotent on repeated calls.
per-domain: combined_english=0.835443 combined_korean=0.582353 combined_singing=0.354098

# 2026-04-21 — hypothesis: add mfcc_post_retrospective_match (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot off
    the "PRE-vs-POST contrast at single boundary t" family onto
    RETROSPECTIVE CONTEXT MATCHING. For post[t, t+2] against a bank of
    past pre-context windows [t-k-2, t-k] for k ∈ {3, 6, 9, 12}, take
    the MINIMUM cosine distance over all k. Reuses cached feat_mfcc —
    ZERO new librosa calls, ZERO new caches. Edge guard t-14 < 0 OR
    t+2 > duration_s → sentinel 0.0. FEATURE_NAMES 80→81 forces
    auto-retrain via US-505 sha gate.

(b) WHY over recent failures. Last 60+ iterations (1st-order paired-
    diff, 2nd-order cross-intra, F-statistic variance, persistence,
    cross-scale, trajectory velocity, correlation-matrix structure,
    onset-strength asymmetry, full-window histogram, DSP-gate
    tightening) all asked "how does POST at t differ from PRE at t?"
    Every one collapsed on 3 singing chord-cycle FPs because
    within-song drift makes PRE[t-2,t] and POST[t,t+2] carry
    similar-magnitude shifts as a cross-song boundary. The genuinely
    untried axis is comparing POST to EARLIER PRE-HISTORY with a MIN
    aggregator: chord-cycle FPs in song A have post at t = a chord
    that ALREADY appeared at some past window (cycle period 2-3s, so
    one of k ∈ {3, 6, 9, 12} s into the past holds that chord); min
    distance collapses to small → feature silent → FP not boosted.
    Cross-song splice has post = entirely song B absent from the
    whole pre-history → all past distances large → min large → fires.

    Speech self-gating: within-recording all past contexts and post
    share speaker+mic+room → all distances small → min small →
    feature ~0 on non-splice positions → GBM low per-domain SHAP on
    english/korean. Cross-speaker splice (TP): past all speaker A,
    post speaker B → all distances large → min large → fires, helps
    speech TPs too.

    Orthogonal: NOT any paired-diff or voicing-masked single-boundary
    feature; NOT 9064eec cross-intra (intra sub-windows on SAME span,
    no past-history bank, no MIN aggregator); NOT 4e67946 persistence
    (future horizons, same pre reference); NOT 4314449 cross-scale
    (two scales at same boundary); NOT 1b4fe7c correlation-matrix; NOT
    1b3eb06 trajectory-velocity; NOT 1e57702 histogram cosine; NOT
    DSP-gate. FIRST min-over-past-context feature in the 80-set,
    FIRST feature using past pre-history as a comparison bank.

(c) IF THIS FAILS. (1) Singing unchanged (chord period in the 3 FPs is
    not aligned with {3, 6, 9, 12} — e.g., sustained pad bridges with
    no chord cycle, or 5-7s cycles) → expand offsets to {2, 4, 6, 8,
    10, 12, 14} for denser coverage. (2) Speech regresses (within-
    recording phoneme sequences at 3-12s lag have different vowel
    centroids → min distance not small on non-splice speech) → switch
    to voiced-masked variant. (3) Feature fires but zero SHAP
    (redundant with voiced_mfcc_cosine_dist via GBM correlated splits)
    → pivot to chroma axis (key-signature-level match retrieves
    chord-class identity more directly than cepstral envelope).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent — cannot
    verify whether the 3 singing FPs sit at chord-cycle periods
    matching {3, 6, 9, 12} s (mechanism assumption); could be
    verse→chorus transitions (10-30s scale, well beyond offset bank)
    or sustained-pad bridges. (ii) SHAP rollup STILL empty for 14
    keeps. (iii) tunable frontier still does not track DSP_*.

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain,
    file, t_sec, p_splice, dsp_phase_z/t2_z/cpe_z, voicing,
    mfcc_post_retrospective_match, top-5 |SHAP|). Would settle every
    retrospective/past-history hypothesis data-driven.
    (2) SHAP ROLLUP REPAIR.
    (3) ADD DSP_SUM_MIN / DSP_CONFIRMATION_MIN / DSP_PHASE_MIN /
    DSP_SECOND_HIGHEST_MIN to the tunable frontier snapshot.

## 2026-04-21T03:38:47+09:00 — 3d56d52 (discard, combined=0.479388)
subject: add voiced_mfcc_post_retrospective_match (FEATURE_NAMES 80->81) -- VOICING-MASKED retrospective context match. Fuses proven voicing-mask self-gating (1eda8e3 +0.054 biggest keep, 49bd0b1 voiced_mfcc +0.028 keep) onto b5b1a0d's retrospective-match mechanism (all-frame version lifted singing 0.345->0.354 best in 60+ iterations but regressed korean/english). For post[t,t+2] voiced-mean MFCC, compute min cos_dist to voiced-mean MFCC at past windows [t-k-2,t-k] for k in {3,6,9,12}s. Voicing mask excludes mic/room noise-floor variability that drove b5b1a0d speech regression. Mechanism on 3 singing chord-cycle FPs: same singer's repeated vowels match some past k -> min tiny -> silent. Cross-song splice: post voiced differs from all past song-A voiced -> min large -> fires. Speech self-gating: within-recording same-speaker vowel-mean MFCC stable across past k -> min tiny non-splice -> silent. Cross-speaker splice: post voiced differs from all past speaker-A voiced -> min large -> fires. Reuses cached feat_mfcc + feat_vp, ZERO new librosa calls, ZERO new caches. Edge guard t-14<0 OR t+2>duration OR any voiced mask empty -> sentinel 0.0. Orthogonal: NOT b5b1a0d (all-frame no mask); NOT 49bd0b1 voiced_mfcc (single-boundary no past bank no MIN); NOT 4e67946 persistence (future horizon); NOT 9064eec/c5040d7 cross-intra (intra sub-windows on SAME span); NOT 1eda8e3/7972a98 asymmetry (single boundary no past bank). FIRST voicing-masked min-over-past-context feature. Pure features.py change -- 1 new block (~55 lines) + FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call. Per-t cost 5 slices + 5 voiced-masked means on 13-dim + 4 cosines + 1 min, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name correct, synthetic A(220Hz)/B(440Hz) splice at t=30 yields 0.432 vs within-A t=20 yields 0.003 (~160x discrimination), edge guards t=10 (t-14<0) and t=39 (t+2>40s) both return 0.0, all 81 features finite on synthetic audio, idempotent on repeated calls.
per-domain: combined_english=0.715385 combined_korean=0.466667 combined_singing=0.330000

# 2026-04-21 — hypothesis: add voiced_mfcc_post_retrospective_match (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — VOICING-MASKED
    retrospective context matching. For post[t,t+2] compute voiced-mean
    MFCC; for each k ∈ {3, 6, 9, 12}s compute voiced-mean MFCC over
    past window [t-k-2, t-k]; feature = MIN over k of cos_dist(
    post_voiced_mean, past_voiced_mean_k). Reuses cached feat_mfcc +
    feat_vp — ZERO new librosa calls, ZERO new caches. Edge guard
    t-14<0 OR t+2>duration OR any window's voiced mask empty OR any
    norm underflow → sentinel 0.0. FEATURE_NAMES 80→81 forces
    auto-retrain via US-505 sha gate.

(b) WHY over recent failures. b5b1a0d (mfcc_post_retrospective_match
    all-frame) IMPROVED singing 0.345→0.354 — the best singing score
    in 60+ iterations, the retrospective mechanism clearly bit chord-
    cycle FPs — but regressed korean 0.667→0.582 and english 0.889→
    0.835. Speech regression very likely comes from all-frame mean
    mixing voiced (phoneme) + unvoiced (silence/mic-noise) axes;
    mic/room noise-floor at non-splice t differs across past k in a
    recording with varying phoneme density, so min_past dist can be
    non-trivial on non-splice speech → feature spuriously boosts
    p_splice on speech, creating new FPs. Voicing-mask proven to fix
    exactly this failure mode: 1eda8e3 MFCC asymmetry (+0.054 biggest
    keep) and 49bd0b1 voiced_mfcc (+0.028 keep) both pair cepstral
    cosine distance with a voiced mask to self-gate speech.

    Mechanism on 3 singing chord-cycle FPs: same singer's voiced
    vowels repeat at every chord cycle (2-3s period) → post
    voiced_mean_mfcc at chord-cycle FP matches some past k ∈
    {3,6,9,12}s window within the song closely → min tiny → silent,
    FP not boosted. Unvoiced accompaniment transients are excluded
    so noisy drum-hit variability no longer lifts the min.

    Cross-song splice (singing TP): post voiced = song B vocalist,
    past voiced = all song A vocalist → all past distances LARGE →
    min LARGE → fires.

    Speech self-gating on non-splice: within-recording same speaker's
    vowel-mean MFCC is the speaker+mic fingerprint, stable across
    past k=3/6/9/12s windows → min tiny on non-splice speech →
    feature silent, no new FPs. Cross-speaker splice (speech TP):
    post voiced = speaker B vowels, past voiced = all speaker A
    vowels → min LARGE → fires, HELPS speech TPs (reproduces
    1eda8e3 self-gating mechanism).

    Orthogonal. NOT b5b1a0d (all-frame, no voicing mask); NOT
    49bd0b1 voiced_mfcc (single-boundary, no past-history bank, no
    MIN aggregator); NOT 4e67946 persistence (future horizon, same
    pre reference); NOT 4314449 cross-scale (two scales, same
    boundary); NOT 9064eec/c5040d7 cross-intra (intra sub-windows
    on SAME span, no past bank, no MIN); NOT 1eda8e3/7972a98
    asymmetry (voiced/unvoiced paired-diff on single boundary); NOT
    1b4fe7c corr-matrix; NOT 1b3eb06 trajectory velocity; NOT
    1e57702 histogram; NOT any DSP-gate. FIRST voicing-masked
    min-over-past-context feature; FIRST feature combining voicing
    mask with past pre-history comparison bank.

    Blast radius. Pure features.py — 1 new block (~50 lines fusing
    _block_voiced_mfcc mask logic with b5b1a0d's past-bank MIN
    aggregator) + 1 FEATURE_NAMES append + 2 assert bumps (80→81) +
    1 call in extract_features. ZERO new caches, ZERO new librosa
    calls. Per-t cost: 5 slices + 5 voiced-masked means on 13-dim +
    4 cosines + 1 min, sub-ms.

(c) IF THIS FAILS. (1) Singing loses b5b1a0d's +0.009 gain (voicing
    mask strips drum-onset signal that was driving retrospective-
    match) → fall back to UNVOICED-only variant (unvoiced captures
    drum/mastering which may be the true cross-song signature).
    (2) Speech still regresses (within-recording vowel-mean voiced-
    MFCC drifts enough at 3-12s offsets to make min non-trivial) →
    widen offset bank to {6, 9, 12, 15, 18}s for longer baseline.
    (3) Feature fires but zero SHAP (redundant with 49bd0b1
    voiced_mfcc via correlated GBM splits) → pivot to CHROMA voiced
    retrospective match (direct chord-class identity, orthogonal
    to cepstral).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent — cannot
    verify the 3 singing FPs sit at chord-cycle periods matching
    {3,6,9,12}s; could be verse→chorus transitions (10-30s) or
    sustained-pad bridges where retrospective bank finds no match.
    (ii) SHAP rollup STILL empty for 14 keeps — cannot verify
    whether 49bd0b1 voiced_mfcc was itself load-bearing. (iii)
    b5b1a0d per-domain TP recall delta unknown (singing +0.009
    could be on clean-FP-reduction OR on real-TP gain).

(e) Wrapper enhancements. Three unchanged highest-priority asks
    across 67+ iterations:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain,
    file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    voicing_fraction, voiced_mfcc_past_bank_min_k, top-5 |SHAP|).
    Would settle every retrospective-context-match hypothesis
    data-driven.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) ADD DSP_SUM_MIN / DSP_CONFIRMATION_MIN / DSP_PHASE_MIN to
    the tunable frontier snapshot alongside GBM_*.

## 2026-04-21T03:53:39+09:00 — a6cf49d (discard, combined=0.522140)
subject: add voiced_chroma_post_retrospective_match (FEATURE_NAMES 80->81) -- VOICING-MASKED retrospective context match on the CHROMA (pitch-class / key-signature) axis. For post[t,t+2] voiced-mean chroma, compute min cos_dist against voiced-mean chroma at past windows [t-k-2,t-k] for k in {3,6,9,12}s. Cited fallback from 3d56d52(c)(3). b5b1a0d all-frame MFCC retrospective match lifted singing 0.345->0.354 (BEST in 60+ iterations) but regressed korean/english (unvoiced noise-floor heterogeneity); 3d56d52 voicing-masked MFCC variant regressed singing further 0.354->0.330 because drum-onset signal in MFCC cepstrum that drove retrospective match lives in UNVOICED frames. Chroma is structurally different: drums/broadband percussion barely register in chroma (noise-like transients spread roughly uniformly across 12 pitch classes) so voicing mask doesn't strip productive signal. Within one song's key every chord shares 3-5 of 12 pitch classes so voiced-mean chroma over any 2s window converges to key-signature center; past {3,6,9,12}s voiced-chroma windows all sit in same key -> min cos_dist TINY on chord-cycle FP -> silent, FP not boosted. Cross-song splice crosses keys: post voiced chroma song B key, past voiced chroma all song A key -> all distances LARGE -> min LARGE -> fires. Speech self-gating (proven 32cac36 voiced_chroma template): voiced chroma on speech is per-vowel prosodic pitch-class noise that averages to near-uniform over any window >=1s, all past windows have similar uniform chromas -> min small uniform on non-splice speech -> GBM low per-domain SHAP on english/korean. Reuses cached feat_chroma + feat_vp, ZERO new librosa calls, ZERO new caches. Edge guard t-14<0 OR t+2>duration_s OR any voiced mask empty OR any norm underflow -> sentinel 0.0. FEATURE_NAMES 80->81 forces auto-retrain via US-505 sha gate. Orthogonal: NOT b5b1a0d (all-frame MFCC, different axis AND mask); NOT 3d56d52 (voiced MFCC, different axis); NOT 32cac36 voiced_chroma_cosine_dist (single boundary, no past bank, no MIN aggregator); NOT 8170784 voiced_chroma_cross_intra_contrast (intra sub-windows on SAME +-4s span, no past history); NOT f4148cc voiced_unvoiced_chroma_asymmetry (1st-order paired-diff); NOT 9064eec/c5040d7/26a3687 cross-intra; NOT 4e67946 persistence (future horizon); NOT 4314449 cross-scale; NOT 1b4fe7c corr-matrix; NOT any DSP-gate. FIRST voicing-masked min-over-past-context feature on CHROMA axis. Pure features.py change -- 1 new block (~65 lines) + FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features + self-test assert bump. Per-t cost 5 slices + 5 voiced-masked means on 12-dim chroma + 4 cosines + 1 min, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_chroma_post_retrospective_match', synthetic A(C-major chord-cycle 3s period)/B(F#-major) splice at t=30 yields 0.612 LARGE vs within-A chord-cycle t=18 yields 0.000021 (~29000x discrimination), edge guards t=10 (t-14<0) and t=39 (t+2>40s) both return 0.0 sentinel correctly, all 81 features finite, idempotent on repeated calls.
per-domain: combined_english=0.765432 combined_korean=0.492754 combined_singing=0.377419

# 2026-04-21 — hypothesis: add voiced_chroma_post_retrospective_match (FEATURE_NAMES 80 → 81)

(a) Pure `splice/features.py` add — VOICING-MASKED retrospective context
    match on the CHROMA (pitch-class / key-signature) axis. For
    post[t, t+2] voiced-mean chroma, compute min cos_dist against
    voiced-mean chroma at past windows [t-k-2, t-k] for
    k ∈ {3, 6, 9, 12}s. Reuses cached feat_chroma + feat_vp — ZERO new
    librosa calls, ZERO new caches. Edge guard t-14<0 OR t+2>duration
    OR any voiced mask empty OR any norm underflow → sentinel 0.0.
    FEATURE_NAMES 80→81 forces auto-retrain via US-505 sha gate.

(b) WHY over recent failures. b5b1a0d all-frame MFCC retrospective match
    lifted singing 0.345→0.354 — THE BEST singing result in 60+
    iterations — but regressed korean/english (unvoiced noise-floor
    heterogeneity across past k). 3d56d52's voicing-masked MFCC variant
    regressed singing further (0.354→0.330) because the drum-onset
    signal in MFCC cepstrum that drove retrospective match on singing
    lives in UNVOICED frames — voicing mask stripped it.

    Chroma is structurally different: (i) drums/broadband percussion
    barely register in chroma because noise-like transients spread
    roughly uniformly across 12 pitch classes, so voicing mask on
    chroma ISN'T stripping the productive chord-cycle signal. (ii)
    Within one song's key every chord shares 3-5 of 12 pitch classes,
    so voiced-mean chroma over any 2s window converges to the key-
    signature center; past {3,6,9,12}s voiced-chroma windows all sit
    in the same key → min cos_dist TINY on chord-cycle FP → silent.
    (iii) Cross-song splice crosses keys — post voiced-mean chroma is
    song B's key, past voiced-mean chromas are all song A's key → all
    distances LARGE → min LARGE → fires.

    Speech self-gating (proven 32cac36 voiced_chroma_cosine_dist keep
    template). Voiced chroma on speech is per-vowel prosodic pitch-
    class noise that averages to near-uniform over any window ≥1s,
    so all past windows have similar uniform chromas → min small and
    uniform on non-splice speech → GBM low per-domain SHAP on english/
    korean.

    Explicit cited fallback from 3d56d52(c)(3): "pivot to CHROMA voiced
    retrospective match (direct chord-class identity, orthogonal to
    cepstral)". GBM max_depth=4 cannot synthesize min-over-past-bank
    from existing single-boundary voiced_chroma_cosine_dist.

    Orthogonal. NOT b5b1a0d (all-frame MFCC — different axis AND mask);
    NOT 3d56d52 (voiced MFCC — different axis); NOT 32cac36 / any
    voiced_chroma geometry (single boundary, no past bank, no MIN
    aggregator); NOT 8170784 voiced_chroma_cross_intra_contrast (intra
    sub-windows on SAME ±4s span, no past history); NOT f4148cc
    voiced_unvoiced_chroma_asymmetry (1st-order paired-diff); NOT
    9064eec / c5040d7 / 26a3687 cross-intra; NOT 4e67946 persistence;
    NOT 4314449 cross-scale; NOT 1b4fe7c corr-matrix; NOT any DSP-gate.
    FIRST voicing-masked min-over-past-context feature on CHROMA axis.

(c) IF THIS FAILS. (1) Singing unchanged — within-song key shifts
    (bridge modulation, verse→chorus key change) break key-invariance
    → fallback to spec_contrast retrospective match (mastering
    fingerprint frozen within-song regardless of key). (2) Speech
    regresses — within-recording vowel prosody chroma drifts enough
    over 12s that min distance is non-trivial on non-splice speech →
    narrow offset bank to {3, 5, 7, 9}s for shorter baseline. (3)
    Feature fires but zero SHAP — redundant with 32cac36 voiced_chroma
    via correlated GBM splits → pivot to unvoiced all-frame chroma
    retrospective (drums-weighted pitch distribution instead of vocal
    melody).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 67+
    iterations — cannot verify the 3 singing FPs sit in stable-key
    regions (matching mechanism) or in bridge/modulation regions
    where key shifts within-song. (ii) SHAP rollup STILL empty for
    14 keeps — cannot verify whether 32cac36 voiced_chroma is load-
    bearing. (iii) b5b1a0d per-FP retrospective-match values unknown.
    (iv) 0.4907 identical-streak root cause unknown.

(e) Wrapper enhancements. Three unchanged highest-priority asks across
    67+ iterations:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain, file,
    t_sec, p_splice, dsp_phase_z/t2_z/cpe_z, voicing_fraction,
    voiced_chroma_cosine_dist, post_retrospective_match_k{3,6,9,12},
    top-5 |SHAP|). Would settle every retrospective/key-signature
    hypothesis data-driven.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) ADD DSP_SUM_MIN / DSP_CONFIRMATION_MIN / DSP_PHASE_MIN to the
    tunable frontier snapshot alongside GBM_*.

## 2026-04-21T04:09:47+09:00 — 32ff893 (discard, combined=0.544995)
subject: add voiced_chroma_self_calibrated_novelty (FEATURE_NAMES 80->81) -- SELF-CALIBRATED retrospective-match on voiced chroma. For post[t,t+2] voiced-mean chroma compute min cos_dist to 4 past voiced-mean chromas at [t-k-2,t-k] for k in {3,6,9,12}s (post_novelty), AND mean cos_dist over all 6 pairwise past-past pairs (past_self_sim); feature = post_novelty - past_self_sim. Reuses cached feat_chroma + feat_vp, ZERO new librosa calls, ZERO new caches. Edge guard t-14<0 OR t+2>duration_s OR any voiced mask empty OR any norm underflow -> sentinel 0.0. FEATURE_NAMES 80->81 forces auto-retrain via US-505 sha gate. Three prior retrospective-match variants: b5b1a0d (all-frame MFCC) singing 0.354 best-in-60+ but speech regressed; 3d56d52 (voiced MFCC) singing further regressed; a6cf49d (voiced chroma) singing 0.377 HIGHEST-in-60+ but speech catastrophic (korean 0.667->0.493 english 0.889->0.765). Pattern: retrospective-match bites chord-cycle FPs on singing but absolute min-over-past fires on non-splice speech because within-recording prosody/phoneme drift makes min non-trivial. STRUCTURAL fix = self-calibration: compare post_to_past novelty against past_to_past self-similarity; within-recording drift cancels, only cross-source novelty survives. Mechanism: chord-cycle FP past_self_sim small AND post_novelty small (post recurs in past) -> feature ~0 silent; real cross-song past_self_sim small AND post_novelty LARGE (song B) -> feature +0.25 to +0.50 fires; speech non-splice past_self_sim moderate AND post_novelty moderate same-drift -> feature ~0 (GBM low per-domain SHAP no new FPs); speech cross-speaker TP past_self_sim moderate AND post_novelty LARGE -> feature positive TP boosted. Chroma axis because a6cf49d gave highest-in-60+ singing 0.377 -- raw signal is there only the aggregator was wrong; drums/percussion barely register in chroma so voicing mask doesn't strip productive signal (killed 3d56d52 voiced MFCC). Orthogonal: NOT a6cf49d (absolute min no self-cal); NOT b5b1a0d/3d56d52 (different axis, no self-cal); NOT 32cac36 single-boundary no past bank; NOT 8170784 cross-intra (same +-4s span no past-history bank no self-calibration); NOT f4148cc 1st-order paired-diff; NOT 9064eec/26a3687/c5040d7 cross-intra; NOT 4e67946 persistence; NOT 4314449 cross-scale; NOT 1b4fe7c corr-matrix; NOT any DSP-gate. FIRST self-calibrated (relative to past-history self-similarity) retrospective-match feature, FIRST feature combining novelty with past-history pair-distance baseline. GBM max_depth=4 cannot synthesize subtraction of two derived scalars from threshold splits. Pure features.py change -- 1 new block (~60 lines) + FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call + self-test assert bump. Per-t cost: 5 voiced-masked-mean 12-dim + 4 post-past cosines + 6 past-past cosines + min + mean + subtract, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81 last-name correct, synthetic chord-cycle C-major cycling 3s period within-song yields t=18 -0.53 / t=22 -0.04 / t=25 -0.06 / t=28 -0.03 ALL NEGATIVE (post recurs in past); A->B splice at t=25 yields t=18 -0.53 / t=22 -0.04 / t=25 +0.26 / t=28 +0.33 SIGN-FLIP at boundary, real singing train 200 calls = 0.89ms/call, edge guards t-14<0 and t+2>duration both return 0.0 sentinel, all 81 features finite, idempotent.
per-domain: combined_english=0.886076 combined_korean=0.532836 combined_singing=0.342857

# 2026-04-21 — hypothesis: add voiced_chroma_self_calibrated_novelty (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — SELF-CALIBRATED retrospective
    novelty on voiced chroma. For voiced-mean chroma of post[t,t+2] compute
    min cos_dist vs 4 past voiced-mean chromas at [t-k-2,t-k] for k in
    {3,6,9,12}s (post_novelty). ALSO compute mean cos_dist over all 6
    pairwise (i<j) past-past distances (past_self_sim). feature =
    post_novelty − past_self_sim. Reuses cached feat_chroma + feat_vp —
    ZERO new librosa calls, ZERO new caches. Edge guard t-14<0 OR
    t+2>duration OR any voiced mask empty OR any norm underflow → sentinel
    0.0. FEATURE_NAMES 80→81 forces auto-retrain via US-505 sha gate.

(b) WHY over recent failures. Three retrospective-match variants tried:
    b5b1a0d (all-frame MFCC, singing 0.354 best-in-60+ but speech regressed
    via unvoiced noise-floor heterogeneity), 3d56d52 (voiced MFCC, singing
    regressed 0.354→0.330 because drum-onset signal lives in unvoiced),
    a6cf49d (voiced chroma, singing 0.377 BEST-in-60+ but korean 0.667→
    0.493 and english 0.889→0.765 — catastrophic speech regression).
    Clear pattern: retrospective-match mechanism DOES bite chord-cycle
    FPs on singing (voiced chroma a6cf49d hit the highest singing score
    we've seen), but absolute min-over-past distance fires on non-splice
    speech because within-recording prosody/phoneme drift over 3-12s
    makes min distance non-trivial even when no splice is present.
    a6cf49d's theory ("past voiced chromas all similar uniform → min
    small") was empirically wrong.

    The structural fix is SELF-CALIBRATION: compare the post-to-past
    distance against how similar past IS to ITSELF. past_self_sim
    captures local drift magnitude; subtracting it normalizes away
    the "how drifty is THIS stretch of audio" baseline.

    Mechanism on 3 singing chord-cycle FPs: past_self_sim small (song A
    cycles through consistent chord set) AND post_novelty small (post
    chord recurs in past) → feature ~0 silent, FP not boosted. Real
    cross-song splice: past_self_sim small (song A self-consistent)
    AND post_novelty LARGE (song B key) → feature +0.25 to +0.50 fires.
    Speech non-splice (a6cf49d regression class): past_self_sim
    moderate AND post_novelty moderate (same drift continues) →
    feature ~0 → GBM low per-domain SHAP on english/korean → NO new
    FPs. Speech cross-speaker TP: past_self_sim moderate AND
    post_novelty LARGE (speaker B) → feature +0.15 to +0.40, TP
    boosted.

    Chroma chosen because a6cf49d gave the HIGHEST singing score in
    60+ iterations (0.377) — raw signal is there, only the aggregator
    was wrong. Drums/percussion barely register in chroma so voicing
    mask doesn't strip productive signal (the failure that killed
    3d56d52 voiced MFCC).

    Orthogonal. NOT a6cf49d (absolute min, no self-calibration); NOT
    b5b1a0d (MFCC all-frame absolute); NOT 3d56d52 (voiced MFCC
    absolute); NOT 32cac36 voiced_chroma (single-boundary no past
    bank); NOT 8170784 voiced_chroma_cross_intra (intra sub-windows
    on SAME ±4s span, no past-history bank, no self-calibration);
    NOT f4148cc 1st-order paired-diff; NOT 9064eec/26a3687/c5040d7
    cross-intra; NOT 4e67946 persistence; NOT 4314449 cross-scale;
    NOT 1b4fe7c corr-matrix; NOT any DSP-gate. FIRST self-calibrated
    (relative to past-history self-similarity) retrospective-match
    feature in 80-set. GBM max_depth=4 cannot synthesize a
    subtraction of two derived-statistic scalars via threshold splits.

(c) IF THIS FAILS. (1) Singing loses a6cf49d's +0.032 gain because
    self-calibration subtracts the chord-cycle drift that was the
    productive signal → fall back to post_novelty − MIN(past_pairs)
    (tightest past-history pair as baseline). (2) Speech still
    regresses (4 past offsets too few to estimate drift) → widen
    bank to 8 offsets for more reliable self_sim. (3) Feature fires
    but zero SHAP (redundant with voiced_chroma_cosine_dist via GBM
    correlated splits) → pivot to MFCC axis with same self-calibration.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 68+
    iterations — cannot verify the 3 singing FPs sit in stable-key
    sections (mechanism) vs bridge/modulation. (ii) SHAP rollup STILL
    empty for 14 keeps. (iii) per-FP retrospective-match values from
    b5b1a0d/a6cf49d unknown — would directly validate self-calibration.

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain, file,
    t_sec, p_splice, dsp_phase_z/t2_z/cpe_z, voicing_fraction,
    voiced_chroma_cosine_dist, post_retrospective_match,
    past_self_sim, top-5 |SHAP|).
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) ADD DSP_SUM_MIN / DSP_CONFIRMATION_MIN / DSP_PHASE_MIN /
    DSP_SECOND_HIGHEST_MIN to tunable frontier snapshot.

## 2026-04-21T04:26:50+09:00 — c578d73 (discard, combined=0.518572)
subject: add voiced_chroma_retrospective_match_max_calibrated (FEATURE_NAMES 80->81) -- MAX-CALIBRATED retrospective-match on voiced chroma. For post[t,t+2] voiced-mean chroma compute min cos_dist to 4 past voiced-mean chromas at [t-k-2,t-k] for k in {3,6,9,12}s (post_novelty) AND MAX cos_dist over all 6 pairwise past-past distances (past_self_sim_max); feature = post_novelty - past_self_sim_max. Reuses cached feat_chroma + feat_vp + feat_audio length; ZERO new librosa calls, ZERO new caches. Edge guard t-14<0 OR t+2>duration_s OR any voiced mask empty OR any norm underflow -> sentinel 0.0. FEATURE_NAMES 80->81 forces auto-retrain via US-505 sha gate. Four prior retrospective-match variants: b5b1a0d (all-frame MFCC) singing 0.354; 3d56d52 (voiced MFCC) singing 0.330; a6cf49d (voiced chroma absolute min) singing 0.377 BEST-in-70+ but speech catastrophic 0.767/0.493; 32ff893 (self-cal with MEAN past-pair baseline) speech recovered to 0.886 but singing gave back to 0.343. Pattern: retrospective-match HAS the singing signal (a6cf49d highest-ever) and self-calibration stabilises speech (32ff893), but MEAN baseline absorbs too much cross-song signal because MEAN averages small adjacent-pair distances alongside the few pairs that cross chord-cycle boundaries. MAX baseline captures CEILING of internal novelty: the single largest past-vs-past distance is by definition the biggest novelty the recording itself supports. Mechanism on 3 singing chord-cycle FPs: past at k in {3,6,9,12} samples different phases of chord cycle; at least one past-past pair crosses chord boundary (past_self_sim_max ~0.15-0.20); post_novelty on chord-cycle FP matches cycle at some k so min-distance ~0.10-0.15; feature ~0 to -0.05 SILENT. Cross-song splice: past song A self-consistent past_self_sim_max ~0.08, post song B post_novelty ~0.40-0.60, feature +0.32-0.52 STRONGLY POSITIVE. Speech non-splice (a6cf49d failure class): past_self_sim_max ~0.15-0.20 phoneme-drift ceiling; post_novelty ~0.15 similar drift; feature ~0 GBM low per-domain SHAP on english/korean. Speech cross-speaker TP: past same-speaker max ~0.10, post speaker-B ~0.30, feature +0.18-0.22 fires. Chroma axis because a6cf49d was single highest singing score in 70+ iterations (0.377 vs baseline 0.345); raw signal is there, only aggregator was wrong; drums/percussion barely register in chroma so voicing mask does not strip productive signal. Orthogonal: NOT a6cf49d (absolute min no self-cal); NOT 32ff893 (MEAN baseline -- MAX is different aggregator GBM max_depth=4 cannot synthesize from threshold splits on same 10 cosines); NOT b5b1a0d/3d56d52 (MFCC axis); NOT 32cac36 single-boundary; NOT 8170784 cross-intra same-span; NOT f4148cc 1st-order paired-diff; NOT 9064eec/26a3687/c5040d7 cross-intra; NOT 4e67946 persistence; NOT 4314449 cross-scale; NOT 1b4fe7c corr-matrix; NOT any DSP-gate. FIRST MAX-self-calibrated retrospective-match feature; FIRST use of max-over-past-pairs as novelty ceiling. Pure features.py change -- 1 new block (~80 lines) + FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call + self-test assert bumps. Per-t cost: 5 voiced-masked-means on 12-dim chroma + 4 post-past cosines + 6 past-past cosines + min + max + subtract, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81 last-name correct, synthetic C-major chord-cycle A -> F#-major B splice at t=25 yields +0.575 vs within-A t=25 +0.143 (4x discrimination), within-A t=22 chord transition yields +0.135 (chord-cycle not strongly firing), within-A t=18 yields -0.53 (post recurs in past, feature negative), edge guards t-14<0 and t+2>duration both return 0.0 sentinel, real singing_train_001.wav 500 calls = 307ms (0.61ms/call), idempotent, all 81 features finite.
per-domain: combined_english=0.875000 combined_korean=0.500000 combined_singing=0.318750

# 2026-04-21 — hypothesis: add voiced_chroma_retrospective_match_max_calibrated (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — MAX-CALIBRATED retrospective
    novelty on voiced chroma. For voiced-mean chroma of post[t, t+2] compute
    min cos_dist vs 4 past voiced-mean chromas at [t-k-2, t-k] for
    k ∈ {3, 6, 9, 12}s (post_novelty). ALSO compute MAX cos_dist over all
    6 pairwise (i<j) past-past distances (past_self_sim_max).
    feature = post_novelty − past_self_sim_max. Reuses cached feat_chroma +
    feat_vp — ZERO new librosa calls, ZERO new caches. Edge guard t-14<0
    OR t+2>duration OR any voiced mask empty OR any norm underflow →
    sentinel 0.0. FEATURE_NAMES 80→81 forces auto-retrain via US-505.

(b) WHY over recent failures. Four retrospective-match variants tried:
    b5b1a0d (all-frame MFCC, singing 0.354 / speech regressed via unvoiced
    heterogeneity), 3d56d52 (voiced MFCC, singing 0.330 / drum-onset signal
    stripped by mask), a6cf49d (voiced chroma, singing 0.377 HIGHEST-in-70+
    but speech catastrophic 0.767/0.493), 32ff893 (self-cal with MEAN
    past-pair baseline: post_novelty − MEAN, singing 0.343 speech recovered
    to 0.886 but gave back the singing gain). Pattern: retrospective-match
    HAS the singing signal (voiced-chroma a6cf49d was best-ever), and
    self-calibration stabilises speech (32ff893 fixed english/korean) but
    MEAN baseline absorbs too much cross-song signal because MEAN averages
    small adjacent-pair distances alongside the few pairs that actually
    cross chord-cycle boundaries. MAX baseline captures the CEILING of
    internal novelty: the single largest past-vs-past distance — by
    definition the biggest novelty the recording itself supports.

    Mechanism on 3 singing chord-cycle FPs. Past at k∈{3,6,9,12} samples
    different phases of the chord cycle; at least one past-past pair
    crosses a chord boundary (past_self_sim_max ≈ 0.15-0.20, the chord-
    cycle novelty scale). post_novelty on a chord-cycle FP = matching
    chord in past at some k, so min-distance ≈ 0.10-0.15. feature =
    post_novelty − past_self_sim_max ≈ 0 to −0.05, SILENT. MEAN baseline
    (32ff893) only gets ~0.08 so feature was +0.07 and fired weakly → lost
    singing gain.

    Real cross-song splice. past song A self-consistent
    (past_self_sim_max ≈ 0.08), post_novelty = song B vs all song A
    ≈ 0.40-0.60. feature ≈ +0.32-0.52 STRONGLY POSITIVE, fires. MAX
    baseline barely hurts signal because song A internal variation
    ceiling is still small relative to cross-song distance.

    Speech non-splice (a6cf49d failure class). past_self_sim_max ≈
    0.15-0.20 (within-recording phoneme drift ceiling over 9s span).
    post_novelty ≈ 0.15. feature ≈ 0 → GBM low per-domain SHAP on
    english/korean → no new FPs. MAX self-cal STRICTER than MEAN
    (which would give baseline ≈ 0.10 and feature ≈ +0.05 firing).

    Speech cross-speaker splice TP. past same-speaker past_self_sim_max
    ≈ 0.08-0.12, post speaker-B voiced chroma post_novelty ≈ 0.30.
    feature ≈ +0.18-0.22 fires, TP boosted.

    Chroma chosen because a6cf49d was single highest singing score in
    70+ iterations (0.377 vs 0.345 baseline). Raw signal is there; prior
    failures were aggregator choices. Drums/percussion barely register
    in chroma so voicing mask doesn't strip productive signal (the
    failure that killed 3d56d52 voiced MFCC).

    Orthogonal. NOT a6cf49d (absolute min, no self-calibration); NOT
    32ff893 (MEAN past-pair baseline — MAX is a different aggregator
    GBM max_depth=4 cannot synthesize via threshold splits; mean-vs-max
    of 6 distances requires different reductions each with O(6) ops);
    NOT b5b1a0d / 3d56d52 (MFCC axis, not chroma); NOT 32cac36
    voiced_chroma (single-boundary, no past bank); NOT 8170784
    voiced_chroma_cross_intra (intra sub-windows on SAME ±4s, no past-
    history bank); NOT f4148cc 1st-order paired-diff; NOT 9064eec /
    26a3687 / c5040d7 cross-intra; NOT 4e67946 persistence; NOT 4314449
    cross-scale; NOT 1b4fe7c corr-matrix; NOT any DSP-gate. FIRST
    MAX-self-calibrated retrospective-match feature; FIRST use of
    max-over-past-pairs as novelty ceiling rather than mean-as-average.

(c) IF THIS FAILS. (1) Singing still blunted — MAX aggregator captures
    noisy max-outlier pairs that aren't truly chord-cycle boundaries →
    fall back to median-past-pair baseline (robust middle, between MEAN
    and MAX). (2) Speech regresses — 4 past offsets too few to estimate
    a stable MAX (single noisy pair dominates) → widen bank to 8 offsets
    {2,4,6,8,10,12,14,16} for better ceiling estimation. (3) Feature
    fires but zero SHAP — redundant with voiced_chroma via correlated
    GBM splits → pivot to spec_contrast axis with same MAX calibration.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 70+
    iterations — cannot verify the 3 singing FPs sit at chord-cycle
    periods within {3,6,9,12}s coverage (mechanism) vs verse→chorus
    transitions (30s scale). (ii) SHAP rollup STILL empty for 14 keeps
    — cannot verify whether 32cac36 voiced_chroma was itself load-
    bearing to decide marginal signal THIS feature adds. (iii) a6cf49d
    / 32ff893 per-FP feature values at the 3 singing FP positions
    unknown — would directly validate whether MAX baseline is the
    right calibration choice.

(e) Wrapper enhancements. Three unchanged highest-priority asks across
    70+ iterations:
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE per-FP (domain,
    file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    voicing_fraction, voiced_chroma_cosine_dist, post_novelty,
    past_self_sim_max, past_self_sim_mean, top-5 |SHAP|). Would settle
    every retrospective-match aggregator choice data-driven.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) ADD DSP_SUM_MIN / DSP_CONFIRMATION_MIN / DSP_PHASE_MIN /
    DSP_SECOND_HIGHEST_MIN to the tunable frontier snapshot alongside
    GBM_*.

## 2026-04-21T04:42:35+09:00 — 8593da7 (discard, combined=0.521115)
subject: add unvoiced_spec_contrast_post_retrospective_match (FEATURE_NAMES 80->81) -- UNVOICED-mask retrospective context match on spec_contrast (mastering-signature) axis. For post[t,t+2] unvoiced-mean 7-dim contrast, min cos_dist to unvoiced-mean contrast at past [t-k-2,t-k] for k in {3,6,9,12}s. Reuses cached feat_contrast + feat_vp, ZERO new librosa calls, ZERO new caches. Sentinel 0.0 on t-14<0 OR t+2>duration OR any unvoiced mask empty OR any norm underflow. Five prior retrospective-match variants tried, ALL on voiced/all-frame mask and MFCC/chroma axis: b5b1a0d (all-frame MFCC, singing 0.354 / speech regressed), 3d56d52 (voiced MFCC, singing 0.330 -- drum-onset lives UNVOICED), a6cf49d (voiced chroma absolute min, singing 0.377 BEST-in-70+ / speech catastrophic), 32ff893 (self-cal MEAN) / c578d73 (self-cal MAX) -- both rescued speech but killed singing gain. Retrospective mechanism DOES bite chord-cycle FPs on singing (a6cf49d highest-ever 0.377) but calibration subtracts the signal. NONE of the 5 used UNVOICED mask; NONE used spec_contrast. Unvoiced+spec_contrast is at the intersection of two proven mechanisms: 1eda8e3 voiced_unvoiced_MFCC_asymmetry +0.054 biggest keep (unvoiced captures cross-song mastering signal), 7972a98 voiced_unvoiced_spec_contrast +0.005 keep (spec_contrast axis carries mastering). Mechanism on 3 singing chord-cycle FPs: drum kit + compressor + limiter + EQ frozen within-song so unvoiced spec_contrast recurs across past k -> min TINY -> silent -> FP not boosted. Cross-song splice: different mastering chain -> unvoiced spec_contrast differs -> min LARGE -> fires. Speech self-gating via 1eda8e3 mechanism: mic+preamp+codec frozen within-recording so unvoiced consonants+silence spec_contrast stable across k -> min TINY on non-splice -> GBM low per-domain SHAP on english/korean. Speech cross-speaker TP: mic chain change -> min MODERATE/LARGE -> helps speech TPs. Orthogonal: NOT b5b1a0d (all-frame MFCC); NOT 3d56d52 (voiced MFCC -- OPPOSITE mask different axis); NOT a6cf49d/32ff893/c578d73 (voiced chroma -- OPPOSITE mask different axis); NOT 7972a98 voiced_unvoiced_spec_contrast_asymmetry (single-boundary paired-diff no past bank no MIN aggregator); NOT e7ca9eb voiced_spec_contrast (voiced mask single-boundary); NOT 26a3687 spec_contrast_cross_intra_contrast (intra sub-windows SAME +-4s no past-history bank); NOT any 1st-order paired-diff; NOT any DSP-gate. FIRST unvoiced-mask retrospective match; FIRST spec_contrast retrospective match in 80-set. Pure features.py change -- 1 new block (~55 lines cloning voiced_mfcc_post_retrospective_match template with unvoiced mask replacing voiced + spec_contrast 7-dim replacing MFCC 13-dim) + FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features + self-test assert bumps. Per-t cost: 5 slices + 5 unvoiced-masked means on 7-dim + 4 cosines + 1 min, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81 last-name 'unvoiced_spec_contrast_post_retrospective_match', real singing_train clean_001 at 7 within-song probe positions yields small values 0.0000-0.0077 (mechanism-consistent: unvoiced frames stable across past k within-song), 3/7 non-zero (rest return 0.0 sentinel when 2s window lacks unvoiced frames -- vocal-dense positions), edge guards t=5 (t-14<0) and t=29 (t+2>dur) both return 0.0, 300 calls 0.15s (0.52ms/call), all 81 features finite across 15 sampled positions, idempotent on repeated calls.
per-domain: combined_english=0.864198 combined_korean=0.537313 combined_singing=0.304762

(a) HYPOTHESIS. Add `unvoiced_spec_contrast_post_retrospective_match`
    (FEATURE_NAMES 80→81). For post[t,t+2] compute UNVOICED-mean 7-dim
    spec_contrast. For each k ∈ {3,6,9,12}s compute unvoiced-mean
    spec_contrast over past [t-k-2,t-k]. feature = MIN over k of
    cos_dist(post_unvoiced, past_unvoiced_k). Reuses cached feat_contrast
    + feat_vp. ZERO new librosa calls, ZERO new caches. Sentinel 0.0 on
    t-14<0 OR t+2>duration OR any unvoiced mask empty OR any norm
    underflow. FEATURE_NAMES 80→81 forces auto-retrain via US-505.

(b) WHY over recent failures. Five retrospective-match variants tried:
    b5b1a0d (all-frame MFCC, singing 0.354 / speech regressed),
    3d56d52 (voiced MFCC, singing regressed 0.330 — drum-onset lives in
    UNVOICED), a6cf49d (voiced chroma absolute min, singing 0.377
    BEST-in-70+ but speech catastrophic), 32ff893 (self-cal MEAN) /
    c578d73 (self-cal MAX) — both rescued speech but killed singing
    gain. Retrospective mechanism works on singing; calibration
    subtracts the signal. Four prior attempts all used VOICED or
    ALL-FRAME mask; NONE tried UNVOICED. Also all on chroma or MFCC
    axis; NONE on spec_contrast. Unvoiced+spec_contrast is at the
    intersection of two proven mechanisms: (1) 1eda8e3
    voiced_unvoiced_MFCC_asymmetry +0.054 biggest keep proved unvoiced
    captures cross-song mastering signal while voiced captures singer
    timbre; (2) 7972a98 voiced_unvoiced_spec_contrast +0.005 keep
    proved spec_contrast axis carries mastering-signature info. On
    retrospective axis, unvoiced frames on chord-cycle FP see the SAME
    within-song drum+mastering chain at every past k → min TINY →
    silent. Cross-song splice shifts mastering chain → min LARGE →
    fires. Speech: unvoiced = consonants+silence with mic+preamp+codec
    frozen within recording → min TINY on non-splice, LARGE on
    cross-speaker splice (mic/preamp change) → helps TPs via same
    1eda8e3 self-gating mechanism.

    Orthogonal. NOT b5b1a0d (all-frame MFCC); NOT 3d56d52 (voiced
    MFCC — OPPOSITE mask, different axis); NOT a6cf49d / 32ff893 /
    c578d73 (voiced chroma — OPPOSITE mask, different axis); NOT
    7972a98 voiced_unvoiced_spec_contrast_asymmetry (single-boundary
    paired-diff, no past-bank, no MIN aggregator); NOT e7ca9eb
    voiced_spec_contrast (voiced mask, single-boundary, no bank);
    NOT 26a3687 spec_contrast_cross_intra_contrast (intra sub-windows
    on SAME ±4s span, no past-history bank); NOT any 1st-order
    paired-diff; NOT any DSP-gate. FIRST unvoiced-mask retrospective
    match, FIRST spec_contrast retrospective match in 80-set.

(c) IF THIS FAILS. (1) Singing unchanged (unvoiced frames in 2s
    windows too sparse on vocal-dominated singing) → fall back to
    ALL-FRAME spec_contrast retrospective match. (2) Speech regresses
    (within-recording mic-noise drift over 3-12s makes min non-trivial)
    → add MEAN self-calibration as a follow-up. (3) Feature fires but
    zero SHAP (redundant with voiced_unvoiced_spec_contrast_asymmetry
    via correlated splits) → pivot to harmonic-percussive-separated
    percussive retrospective match via librosa.effects.hpss.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 75+
    iterations — cannot verify the 3 singing FPs sit in sections with
    consistent within-song drum+mastering (mechanism assumption) vs
    dynamic arrangement transitions. (ii) SHAP rollup STILL empty for
    14 keeps — cannot verify whether 7972a98 voiced_unvoiced_spec_contrast
    is load-bearing. (iii) a6cf49d per-FP feature values at the 3
    singing FP positions unknown — would validate mask choice.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE per-FP (domain, file, t_sec, p_splice, dsp_phase_z,
    dsp_t2_z, dsp_cpe_z, voicing_fraction,
    voiced_unvoiced_spec_contrast_asymmetry,
    unvoiced_spec_contrast_post_retro_match, top-5 |SHAP|). Would
    settle every mask+axis combo hypothesis data-driven.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) ADD DSP_SUM_MIN / DSP_CONFIRMATION_MIN / DSP_PHASE_MIN /
    DSP_SECOND_HIGHEST_MIN to the tunable frontier snapshot.

## 2026-04-21T05:02:02+09:00 — a23ab28 (discard, combined=0.490700)
subject: add music_gated_voiced_chroma_retro_match (FEATURE_NAMES 80->81) -- voicing-fraction-gated retrospective voiced-chroma match. Combines a6cf49d's productive mechanism (singing 0.377 HIGHEST-in-70+ but speech catastrophic) with a HARD MUSIC GATE on post voicing_fraction: >=0.55 returns 0.0 (speech, gate closed), else min over k in {3,6,9,12}s of cos_dist(voiced_mean_chroma(post[t,t+2]), voiced_mean_chroma([t-k-2,t-k])). Reuses cached feat_chroma + feat_vp + feat_audio; ZERO new librosa calls, ZERO new caches. Edge guard t-14<0 OR t+2>duration OR any voiced mask empty OR norm underflow -> sentinel 0.0. FEATURE_NAMES 80->81 forces auto-retrain via US-505 sha gate. 6 prior retrospective-match variants: b5b1a0d (all-frame MFCC) singing 0.354 speech regressed; 3d56d52 (voiced MFCC) 0.330 drums-in-unvoiced stripped; a6cf49d (voiced chroma abs min) singing 0.377 speech catastrophic; 32ff893 (MEAN self-cal) / c578d73 (MAX self-cal) both rescued speech but subtracted singing gain; 8593da7 (unvoiced spec_contrast) 0.304. Pattern: retrospective-match HAS singing signal (a6cf49d highest-ever) but absolute min-past-dist fires on within-recording speech drift; every arithmetic self-cal via subtraction absorbs singing signal too. Correct lever is HARD DOMAIN GATE via intrinsic audio statistic: singing (vocals+accompaniment) has post voicing_fraction ~0.30-0.55; pure speech 0.65-0.85. Threshold 0.55 separates them. Feature identically 0.0 on speech -> GBM cannot learn any english/korean split on it -> bypasses a6cf49d failure mode rather than cancelling arithmetically. Mechanism on 3 singing chord-cycle FPs: vp_post~0.40 gate opens; post voiced-chroma converges to key centroid (chords share 3-5 of 12 pitch classes); past {3,6,9,12}s windows all in same key -> min TINY silent FP not boosted. Cross-song splice: past song-A-key, post song-B-key -> min LARGE fires. Speech: gate closed feature=0; speech TPs handled by existing 1eda8e3 MFCC asymmetry. Orthogonal: NOT a6cf49d (abs min no gate); NOT 32ff893/c578d73 (subtraction self-cal); NOT b5b1a0d/3d56d52 (MFCC axis); NOT 8593da7 (unvoiced spec_contrast); NOT 32cac36 voiced_chroma (single-boundary); NOT 8170784 voiced_chroma_cross_intra; NOT f4148cc 1st-order paired-diff; NOT any DSP-gate. FIRST voicing-fraction-gated feature in 80-set; FIRST feature using intrinsic audio statistic as hard domain switch rather than arithmetic normalization. GBM max_depth=4 cannot synthesize if-else gate on min-over-past-bank because no past-bank feature exists. Pure features.py change -- 1 new block (~65 lines) + FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features + self-test assert bumps. Per-t cost 5 voiced-masked-means on 12-dim chroma + 4 cosines + 1 min, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81 last-name 'music_gated_voiced_chroma_retro_match'; real singing clean_001 t=20 vp_post=0.405 -> feature=0.1849 (gate open, retrospective active), t=25 vp_post=0.740 -> feature=0.0 (gate closed); real english clean_001 t=16/20 vp_post=0.889 -> feature=0.0 (gate closed across speech); edge guards t=10 (t-14<0) and t=29.5 (t+2>30) both return 0.0; all 81 features finite; idempotent.
per-domain: combined_english=0.839506 combined_korean=0.476471 combined_singing=0.295385

# 2026-04-21 — hypothesis: add music_gated_voiced_chroma_retro_match (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add. Take a6cf49d's productive
    voiced-chroma retrospective-match mechanism (singing 0.377 — HIGHEST
    in 70+ iterations) and put it behind a hard MUSIC GATE keyed on the
    intrinsic voicing fraction of the post window: if post_voicing_fraction
    >= 0.55 return 0.0; else compute min over k ∈ {3,6,9,12}s of cos_dist(
    voiced_mean_chroma(post), voiced_mean_chroma([t-k-2, t-k])). Reuses
    cached feat_chroma + feat_vp, ZERO new librosa calls, ZERO new caches.
    Edge guard t-14<0 OR t+2>duration OR any voiced mask empty OR norm
    underflow → sentinel 0.0. FEATURE_NAMES 80→81 forces auto-retrain.

(b) WHY over recent failures. Six retrospective-match variants tried:
    b5b1a0d (all-frame MFCC) singing 0.354 speech regressed; 3d56d52
    (voiced MFCC) 0.330 drums-in-unvoiced stripped; a6cf49d (voiced
    chroma absolute min) singing 0.377 BEST but speech catastrophic
    0.765/0.493; 32ff893 MEAN self-cal / c578d73 MAX self-cal both
    rescued speech but subtracted the singing gain (0.343/0.319); 8593da7
    unvoiced spec_contrast 0.304. Pattern is now clear: the retrospective-
    match mechanism HAS real singing signal (a6cf49d highest-ever), but
    an ABSOLUTE min-past-dist fires on within-recording speech drift too,
    and every arithmetic self-calibration (subtraction) also absorbs the
    singing signal. Self-calibration via SUBTRACTION is the wrong lever.

    The correct lever is a HARD DOMAIN GATE using an intrinsic audio
    statistic: voicing_fraction. Singing (vocals + accompaniment) yields
    post voicing_fraction ≈ 0.30–0.55 because drums/bass/instrumental
    interludes pull it down. Pure speech yields 0.65–0.85 because dense
    vowels dominate 2s windows. Threshold 0.55 separates them. Making
    the feature IDENTICALLY 0.0 on speech means GBM cannot learn any
    split on it from english/korean — bypasses a6cf49d's speech failure
    mode entirely rather than trying to cancel it arithmetically.

    Mechanism on 3 singing chord-cycle FPs: voicing_fraction ≈ 0.40 so
    gate opens; post voiced-mean chroma converges to key-signature
    centroid (every chord shares 3-5 of 12 pitch classes), past windows
    at k ∈ {3,6,9,12}s all sit in the same key → min cos_dist TINY →
    silent, FP not boosted. Cross-song splice (different key): past
    voiced-chroma all song-A-key, post voiced-chroma song-B-key → min
    LARGE → fires. Speech non-splice and speech TPs: gate closed →
    feature = 0 → no speech regression, speech TPs handled by 1eda8e3
    MFCC asymmetry (english 0.889, korean 0.667 already strong).

    Orthogonal. NOT a6cf49d (absolute min, no gate); NOT 32ff893 (MEAN
    subtraction); NOT c578d73 (MAX subtraction); NOT b5b1a0d / 3d56d52
    (MFCC axis); NOT 8593da7 (unvoiced spec_contrast); NOT 32cac36
    voiced_chroma (single-boundary, no past bank); NOT 8170784
    voiced_chroma_cross_intra (intra sub-windows on SAME ±4s); NOT
    f4148cc 1st-order paired-diff; NOT any DSP-gate. FIRST voicing-
    fraction-gated feature in 80-set. FIRST feature using an intrinsic
    audio statistic as a hard domain switch rather than subtraction.
    GBM max_depth=4 cannot synthesize this if-else gate because
    voicing_prob_post exists but no past-bank feature exists, so the
    joint "voiced_chroma_retro_match * 1{vp_post<0.55}" is unreachable.

(c) IF THIS FAILS. (1) 0.55 threshold wrong — singing voicing_fraction
    on vocal-lead sections exceeds 0.55 (gate closes on real singing)
    → retry with 0.65. (2) Gate correct but chord-cycle FPs still get
    non-zero min (key-change bridge within song) → widen past bank to
    {2,4,6,8,10,12,14}s. (3) Feature fires but zero per-domain SHAP —
    redundant with existing voiced_chroma_cosine_dist via correlated
    GBM splits → pivot to music-gated voiced-MFCC retro match.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 75+
    iterations — cannot verify the 3 singing FPs' actual
    voicing_fraction values or whether they sit at key-stable regions.
    Theory bet on threshold 0.55 vs 0.5 vs 0.6. (ii) SHAP rollup STILL
    empty for 14 keeps. (iii) Distribution of voicing_prob_post across
    domains not surfaced in CURRENT STATE; the 0.55 threshold is my
    prior, not data-observed.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE
    per-FP (domain, file, t_sec, p_splice, voicing_prob_post,
    voiced_chroma_cosine_dist, music_gated_voiced_chroma_retro_match,
    top-5 |SHAP|). Would settle gate-threshold choices data-driven.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) PER-DOMAIN voicing_fraction DISTRIBUTION in CURRENT STATE
    (median / p25 / p75 of voicing_prob_post over clean files per
    domain). Would directly justify gate thresholds for any music-
    gated feature.

## 2026-04-21T05:13:05+09:00 — 1cded37 (discard, combined=0.410922)
subject: add NEAR-PEAK cluster-count filter (detector.py, pure primary tunable, no retrain) -- drop emits whose +-12s p_splice neighborhood (excluding +-1.5s) contains >=2 OTHER grid points with p_splice > GBM_THRESHOLD. STRUCTURAL pivot off 60+ iterations of features/DSP-gate/retrospective-match failures on 3 singing chord-cycle FPs onto p_splice TIME-SERIES STRUCTURE across a wide neighborhood. Chord cycles are PERIODIC: chord transitions recur at 2-3s intervals so p_splice shows 4-6 other elevated peaks per +-12s window on chord-cycle FPs. Real cross-source splices are ISOLATED (1 splice per file in ground truth) so p_splice is locally peaked and nearby grid points sit well below 0.982 -> count=0 pass. Mechanism on 3 singing chord-cycle FPs: each chord transition in the 2-3s cycle produces p_splice>GBM_THRESHOLD, excluding +-1.5s around emit expect 4-6 other hits -> count>=2 reject FPs drop. Real singing TP cross-song splice: post and pre different songs no within-song periodicity either side -> count~0 pass. Speech: korean/english clean_fp=0 already so filter only acts on speech real-splice files which are also isolated (1 speaker change per file) count~0 -> TPs pass. Orthogonal: NOT 60196aa peak-width neighbor-support (required >=1 NEAR neighbor within +-stride to SUPPORT the peak -- OPPOSITE direction 'wide peak=real'), NOT d4d35b1 p_splice local-background-margin (margin-based on local mean catastrophic 0.287). Mine counts DISCRETE OTHER CANDIDATES in a WIDER neighborhood with EXCLUSION of emit's own vicinity -- neither margin nor peak-width; explicit cluster-count semantics. NOT any DSP gate (applies AFTER DSP), NOT feature addition, NOT GBM_THRESHOLD/GBM_MIN_SEP_S/ANALYSIS_STRIDE_S. FIRST wide-window cluster-count filter using p_splice time-series repetition structure in detector history. Filter only DROPS emits -- can never create new FPs or TPs. Pure detector.py change: 3 new constants + 7-line check inserted after DSP confirm + 1 new kv in diag.gbm.chunk_scan_done. Feature set unchanged (len(FEATURE_NAMES)=80 stable), classifier byte-identical, no retrain. Per-emit cost: 2 np slices on hit_mask + 2 sums, sub-us. Edge guard: skip when lookback_n<=exclude_n (short chunks). Smoke-verified: AST parse OK 915 lines, NEAR_PEAK_LOOKBACK_S=12.0 NEAR_PEAK_EXCLUDE_S=1.5 MAX_NEAR_PEAKS=2, other tunables unchanged (GBM_THRESHOLD=0.982 DSP_SUM_MIN=5.0 DSP_CONFIRMATION_MIN=2.0 GBM_MIN_SEP_S=3.5 ANALYSIS_STRIDE_S=0.12), FEATURE_NAMES stable at 80.
per-domain: combined_english=0.542373 combined_korean=0.413115 combined_singing=0.309677

# 2026-04-21 — hypothesis: p_splice NEAR-PEAK cluster-count filter (detector.py)

(a) HYPOTHESIS. Pure `splice/detector.py` change — reject emits whose
    ±12s p_splice neighborhood (excluding ±1.5s around the emit)
    contains ≥ 2 OTHER grid points with p_splice > GBM_THRESHOLD.
    Add NEAR_PEAK_LOOKBACK_S=12.0, NEAR_PEAK_EXCLUDE_S=1.5,
    MAX_NEAR_PEAKS=2 constants; enforce a simple cluster count from
    the existing hit_mask slice. No retrain, no feature change,
    feature count stable at 80, classifier byte-identical.

(b) WHY over recent failures. 60+ features.py additions collapsed on
    3 singing chord-cycle FPs (combined 0.47-0.55). Every DSP-gate
    axis tried (MAX, SUM, SECOND-HIGHEST, PHASE channel-floor) and
    every retrospective-match variant (7 masks/aggregators/axes)
    regressed either singing or speech. The one axis nobody has
    touched is the p_splice TIME-SERIES STRUCTURE across a WIDE
    neighborhood. Chord cycles are periodic: chord transitions recur
    at 2-3s intervals within a song, so within a ±12s window around
    one chord-cycle FP there are ~4-6 other chord-transition p_splice
    peaks. Real cross-source splices are ISOLATED: single transition
    per file, no other elevated p_splice within ±12s of the true
    boundary (ground-truth dataset has 1 splice per file). Counting
    other hit_mask points in the window is a direct, orthogonal
    measure of "repetition" vs "isolation" that the GBM itself
    cannot synthesize because it sees only feature context at time t,
    not the p_splice series across ±12s.

    Mechanism on 3 singing chord-cycle FPs: each chord transition in
    the 2-3s cycle produces an elevated p_splice. In a ±12s window
    (excluding ±1.5s to avoid counting the emit's own narrow peak),
    expect 4-6 other points > GBM_THRESHOLD. Count ≥ 2 → reject.
    FPs drop.

    Real singing TP (cross-song splice): post and pre are from
    different songs, no within-song periodicity on either side. The
    elevated p_splice is localized at the true boundary; other
    grid points in ±12s have p_splice well below 0.982 → count ≈ 0
    → pass.

    Speech: korean/english already have clean_fp=0, so the filter
    only acts on speech real-splice files. Speech splices are also
    isolated (single speaker change per file), count ≈ 0 → TPs pass.

    Orthogonal. NOT 60196aa peak-width neighbor-support (required
    ≥1 NEAR neighbor within ±stride to SUPPORT the peak — OPPOSITE
    direction, "wide peak = real"); NOT d4d35b1 p_splice
    local-background-margin (required peak to EXCEED local
    background by a margin; catastrophic 0.287 — margin-based on
    local mean). Mine counts DISCRETE OTHER CANDIDATES in a WIDER
    neighborhood with EXCLUSION of the emit's own vicinity. Neither
    margin nor peak-width; explicit cluster-count semantics. NOT
    any DSP gate (applies AFTER DSP); NOT a feature addition; NOT
    GBM_THRESHOLD / GBM_MIN_SEP_S / ANALYSIS_STRIDE_S. FIRST
    wide-window cluster-count filter using p_splice time-series
    repetition structure.

    Blast radius. detector.py only, 3 new constants + ~6 lines in
    the gate loop. Per-emit cost: 2 np slices + 2 sums, sub-us. No
    retrain, feature set unchanged, features.py sha stable. Filter
    only DROPS emits — can never create new FPs or new TPs.

(c) IF THIS FAILS. (1) Threshold 2 too strict (real TPs sometimes
    have 1-2 noise hits nearby from dense scan imperfection) →
    raise to 3, keep the mechanism. (2) The 3 singing FPs sit in
    3 DIFFERENT clean files (1 FP each, no cluster) → filter
    does nothing, discard neutral; pivot to per-file file-level
    cluster detection or MULTI-FILE-AWARE gating. (3) Chord-cycle
    period exceeds 12s (verse→chorus transitions at 20-30s) →
    widen LOOKBACK_S to 20.0 and lower count threshold.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent — cannot
    verify whether the 3 singing FPs are in 1 file clustered or 3
    files isolated (decides whether this filter bites). (ii) No
    visibility into the p_splice distribution around an FP —
    whether nearby chord transitions produce > GBM_THRESHOLD peaks
    or only moderate elevations. (iii) SHAP rollup STILL empty for
    14 keeps.

(e) Wrapper enhancements.
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain,
    file, t_sec, p_splice, near_peak_count_in_pm12s,
    dsp_phase_z/t2_z/cpe_z, voicing_fraction, top-5 |SHAP|). Would
    directly settle whether the 3 singing FPs are clustered or
    isolated and validate cluster-count thresholds.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) ADD a p_splice-NEIGHBORHOOD snapshot to CURRENT STATE: for
    each FP, the full p_splice vector in ±12s around the emit
    (or summary stats: count > thr, count > thr*0.95, max outside
    ±1.5s). Would enable data-driven tuning of any wide-window
    post-filter.

## 2026-04-21T05:24:51+09:00 — e4a9c18 (discard, combined=0.496163)
subject: add DATASET_SAMPLE_WEIGHT={singing:2.0, korean:1.0, english:1.0} to HistGBM fit (pure classifier-training change, no feature or detector edit) -- weights each singing training row 2.0x via Pipeline clf__sample_weight routing in both GroupKFold folds and final fit. STRUCTURAL pivot off 60+ iterations of features.py / detector.py tweak exhaustion (every content axis 1st+2nd order paired-diff / variance / persistence / trajectory / corr-matrix / histogram / retrospective-match 7 variants / music-gate, every DSP-gate MAX/SUM/SECOND-HIGHEST/PHASE, every detector post-filter peak-width/margin/cluster-count) onto the classifier-training axis. 1cded37 NEAR-PEAK cluster-count catastrophically broke english 0.889->0.542 proving detector post-filter axis is exhausted. Classifier-training axis has NOT been explored since 2025-Q4 pitch-shift augmentation lifted singing 0.272->baseline. sample_weight directly reshapes the GBM loss gradient -- genuinely orthogonal to features.py / detector.py. Singing splice_f1=0.406 (computed 0.345/0.85 clean_score) means BOTH precision and recall are low in spliced singing; upweight biases GBM's split capacity toward singing-discriminative thresholds GBM has learned to underuse (voiced_chroma_cosine_dist carries singing signal a6cf49d=0.377 highest-ever; voiced_unvoiced_mfcc_asymmetry=0.054 biggest keep). Speech at 0.667/0.889 has margin; 1.0x retains that signal. Factor 2.0 is moderate. Orthogonal: NOT any features.py addition (no feature change, FEATURE_NAMES stable 80); NOT any DSP gate or threshold; NOT any detector post-filter; NOT any GBM hyperparameter (max_depth/max_iter/learning_rate/subsample unchanged); NOT pitch-shift augmentation change (SINGING_AUG_SHIFTS_SEMITONES unchanged). FIRST per-dataset sample-weight tweak in classifier history; FIRST direct loss-gradient bias by domain. Pure train_classifier.py change -- 1 constant + 1 per-manifest sample_weight array + 2 fit calls with clf__sample_weight routing + 2 print lines for row counts. US-505b sha gate auto-retrains from scratch; wrapper runs train_classifier.py and then evaluate.py. Smoke-verified: AST parse OK 478 lines, DATASET_SAMPLE_WEIGHT constant present, clf__sample_weight= routing present in both fold and final fits, Pipeline clf__sample_weight routing works end-to-end on synthetic X/y/sw, FEATURE_NAMES still 80.
per-domain: combined_english=0.795181 combined_korean=0.573913 combined_singing=0.267647

# 2026-04-21 — hypothesis: upweight singing training samples (sample_weight=2.0)

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` change —
    pass `sample_weight` to HistGradientBoostingClassifier.fit() that
    weights each singing training row at 2.0x vs 1.0x for korean/english.
    Forces GBM's loss gradient to prioritize singing-discriminative
    splits at every boosting iteration. No features.py change, no
    detector.py change, feature count stable at 80. US-505b sha gate
    triggers auto-retrain.

(b) WHY over recent failures. 60+ features.py additions (every content
    axis on 1st-order paired-diff, 2nd-order cross-intra, F-stat
    variance, persistence, cross-scale, trajectory, corr-matrix,
    histogram, retrospective-match 7 variants, music-gate), every
    DSP-gate axis (MAX, SUM, SECOND-HIGHEST, PHASE floor), every
    detector post-filter (peak-width, margin, cluster-count) ALL
    collapsed on 3 singing chord-cycle FPs. 1cded37 (NEAR-PEAK
    cluster-count) catastrophically broke english 0.889->0.542 proving
    detector post-filter axis is exhausted. Classifier-training axis
    has NOT been explored since 2025-Q4 pitch-shift augmentation
    (which lifted singing 0.272->baseline). sample_weight directly
    changes the GBM loss gradient -- genuinely orthogonal to any
    feature addition or detector tweak.

    Singing splice_f1 = 0.406 (computed from combined 0.345 / clean_score
    0.85), meaning BOTH precision and recall are low in spliced
    singing files not just clean FPs. Upweighting singing makes GBM
    spend more split capacity on singing-specific feature thresholds
    (voiced-chroma, MFCC asymmetry, spec_contrast retrospective)
    which the agent already confirmed carry singing signal (a6cf49d
    singing=0.377 highest-ever, b5b1a0d singing=0.354). Speech is
    already at 0.667/0.889 with margin; 1.0x weight retains that
    signal. Factor 2.0 is moderate -- not aggressive enough to
    dominate.

(c) IF THIS FAILS. (1) Speech regresses catastrophically (singing
    upweighting dragged GBM toward singing-noise patterns, breaking
    speech thresholds) -> drop factor to 1.3x or 1.5x. (2) Singing
    unchanged (GBM already saturated on singing signal) -> pivot to
    max_depth=5 / max_iter=300 to add GBM capacity. (3) Both domains
    move in correlated direction (GBM learns a globally different
    decision surface) -> try class_weight-style per-LABEL weight
    (hard_cut/crossfade upweight) instead of per-DATASET.

(d) Information gaps. (i) CLEAN_FP_POSITIONS JSON STILL absent --
    cannot verify what features drive the 3 singing FPs. (ii) SHAP
    rollup STILL empty for 14 keeps -- cannot verify per-feature
    per-domain contribution. (iii) Per-domain training-set sizes
    (singing/korean/english row counts) NOT in CURRENT STATE -- can't
    predict effective weight redistribution; factor 2.0 is a guess
    without knowing if singing is 30%, 50%, or 70% of rows.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block. (2)
    SHAP ROLLUP REPAIR -- empty for 14 keeps. (3) PER-DOMAIN TRAINING
    ROW COUNTS in CURRENT STATE (read from training_manifest.json);
    would let sample_weight / class_weight hypotheses pick factors
    data-driven instead of by theory.

## 2026-04-21T06:09:13+09:00 — 7b49d40 (discard, combined=0.465942)
subject: add SINGING_AUG_TIME_STRETCH_RATES=(0.95, 1.05) -- SECOND training-data augmentation axis alongside existing SINGING_AUG_SHIFTS_SEMITONES (pitch-shift). Pure train_classifier.py change, no feature or detector edit. For each clean singing chunk, emit 2 additional augmented copies at librosa.effects.time_stretch rates 0.95 (5% slower, chord cycle 2-3s dilates to 2.1-3.15s) and 1.05 (5% faster, contracts to 1.9-2.85s). chunk-duration-aware sampling: aug_dur_s=len(aug_chunk)/sr since time_stretch changes length (unlike pitch_shift which preserves). EXPLICIT CITED FALLBACK from 4e2941b(c)(2): 'pivot to TIME-STRETCH augmentation via librosa.effects.time_stretch for tempo-invariance'. STRUCTURAL pivot on training-DATA-DIVERSITY axis orthogonal to 4e2941b pitch-shift expansion (FREQUENCY axis) / d1be6c3 GBM max_depth 4->5/max_iter 200->300 (capacity axis discard) / e4a9c18 DATASET_SAMPLE_WEIGHT=2.0 (loss-gradient bias, catastrophic). 3 surviving singing chord-cycle FPs are PERIODIC at 2-3s -- a TEMPORAL signature. Pitch-shift gives GBM same chord-cycle pattern at different pitch levels (frequency); time-stretch gives same pattern at different tempo rates (time). Combined cover both axes so GBM learns rate-invariant AND pitch-invariant discriminators. Mechanism on 3 singing chord-cycle FPs: within-song chord cycle at specific tempo currently maps to one region of feature space; time-stretched clean copies at 0.95x/1.05x populate adjacent regions labeled not_splice, widening not_splice boundary precisely in time-sensitive features (mfcc_delta_*, persistence, trajectory-velocity, voicing_transition_rate_delta) that spurious-fire on chord-cycle periodicity. Speech untouched: ds_id=='singing' gate ensures english/korean rows byte-identical, 0.667/0.889 margins preserved (same safeguard that protected speech across bba6dbe/4e2941b pitch-shift augmentations). Why 0.95/1.05 specifically: phase-vocoder time-stretch is artifact-free within +-5% rate envelope (pitch preserved, transients not smeared audibly); 10%+ introduces phasey artifacts on vocal transients GBM could treat as splice signature. Starts with 2 rates (conservative, matching bba6dbe 2-shift cardinality that lifted singing 0.272->baseline); expandable to (0.9, 0.95, 1.05, 1.1) 4-rate matching pitch-shift cardinality if keeps. Orthogonal: NOT 4e2941b SINGING_AUG_SHIFTS_SEMITONES (frequency axis, complementary template not duplicate); NOT d1be6c3 (GBM capacity knob); NOT e4a9c18 (loss-gradient bias); NOT any features.py addition (FEATURE_NAMES stable 80, features.py sha unchanged); NOT any DSP gate (MAX/SUM/SECOND-HIGHEST/PHASE) or detector post-filter (peak-width/margin/cluster-count); NOT GBM_THRESHOLD/GBM_MIN_SEP_S/ANALYSIS_STRIDE_S. FIRST time-domain training-data augmentation in classifier history; FIRST use of librosa.effects.time_stretch in training pipeline. Blast radius: 1 new tuple constant + 1 new augmentation block (~40 lines cloning pitch-shift template with rate= replacing n_steps= and aug_dur_s=len(aug_chunk)/sr replacing chunk_dur_s for duration-change handling). Feature set unchanged. Detector byte-identical. Retrain cost: ~20 clean singing files x 2 rates = ~40 new rows, time_stretch ~0.5s/chunk + feature extraction, total ~60s training-time bump well inside budget. No inference cost. US-505b train_classifier.py sha gate auto-retrains from scratch. Smoke-verified: AST parse OK 511 lines, SINGING_AUG_SHIFTS_SEMITONES=(+1,-1,+2,-2) stable, SINGING_AUG_TIME_STRETCH_RATES=(0.95, 1.05) len=2, make_pipeline() constructs Pipeline OK with inherited d1be6c3 capacity (max_iter=300 max_depth=5 max_leaf=32 l2=1.0 min_samples_leaf=20), FEATURE_NAMES stable at 80, librosa.effects.time_stretch(rate=0.95) maps 60.000s->63.158s and rate=1.05 maps 60.000s->57.143s on synthetic chunk as expected.
per-domain: combined_english=0.829268 combined_korean=0.410959 combined_singing=0.296825

# 2026-04-21 — hypothesis: add SINGING_AUG_TIME_STRETCH_RATES=(0.95, 1.05)

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` change —
    add SECOND training-data augmentation axis alongside the existing
    pitch-shift. For each clean singing chunk, emit two additional
    augmented copies at tempo rates {0.95, 1.05} via
    librosa.effects.time_stretch. Chunk-duration-aware sampling
    (use len(aug_chunk)/sr rather than the original chunk_dur_s
    since time-stretch changes length, unlike pitch-shift which
    preserves it). FEATURE_NAMES stable at 80; US-505b
    train_classifier.py sha gate auto-retrains. No detector change,
    no feature change, no GBM hyperparameter change.

(b) WHY over recent failures. EXPLICIT CITED FALLBACK from 4e2941b
    (the immediately prior iteration) (c)(2): "pivot to TIME-STRETCH
    augmentation via librosa.effects.time_stretch for tempo-
    invariance". The 3 surviving singing chord-cycle FPs are PERIODIC
    at 2-3s intervals — a TEMPORAL signature. Pitch-shift
    augmentation (bba6dbe ±1 kept / 4e2941b ±1±2 expansion) attacks
    the FREQUENCY axis of training diversity: exposes GBM to the
    same chord-cycle pattern at different pitch levels. Time-stretch
    is the genuinely orthogonal TEMPORAL axis: rate 0.95 slows audio
    5% (chord cycle 2-3s becomes 2.1-3.15s), rate 1.05 speeds 5%
    (becomes 1.9-2.85s). Combined with pitch-shift this gives GBM a
    richer not_splice distribution across BOTH time AND frequency —
    learning rate-invariant AND pitch-invariant discriminators.

    Mechanism on 3 singing chord-cycle FPs: within-song chord cycle
    at a specific tempo currently matches one region of feature
    space; time-stretched clean copies at 0.95x and 1.05x populate
    adjacent regions labeled not_splice, expanding the boundary GBM
    sees on singing and pushing decision thresholds OUT past the
    chord-cycle feature-space locus. Time-sensitive features
    (mfcc_delta_*, persistence, trajectory-velocity, voicing-
    transition-rate) shift slightly at stretched rates, so the GBM
    not_splice distribution expands precisely in the feature axes
    most likely to spurious-fire on chord-cycle periodicity.

    Speech untouched: `ds_id == "singing"` gate ensures english/
    korean rows byte-identical, preserving 0.889/0.667 margins.
    Pitch-shift augmentation used this same guard safely since
    bba6dbe.

    Why 0.95/1.05 specifically. librosa.effects.time_stretch uses
    phase-vocoder reconstruction; 5% rate changes are within the
    artifact-free envelope (pitch preserved, transients not smeared
    audibly). 10%+ starts introducing phasey artifacts on vocal
    transients that GBM could treat as splice signature. Starts
    with 2 rates (conservative, matching the 2-shift cardinality
    of the bba6dbe pitch-shift keep that lifted singing 0.272 →
    baseline); can expand to (0.9, 0.95, 1.05, 1.1) if this keeps.

    Orthogonal. NOT 4e2941b SINGING_AUG_SHIFTS_SEMITONES (frequency
    axis, same augmentation template, complementary rather than
    duplicate); NOT d1be6c3 max_depth / max_leaf_nodes / max_iter
    (GBM capacity knob); NOT e4a9c18 DATASET_SAMPLE_WEIGHT (loss-
    gradient bias); NOT any features.py addition (FEATURE_NAMES
    stable 80, features.py sha unchanged); NOT any DSP gate
    (MAX/SUM/SECOND/PHASE); NOT any detector post-filter (peak-
    width / margin / cluster-count); NOT any GBM_THRESHOLD /
    GBM_MIN_SEP_S / ANALYSIS_STRIDE_S. FIRST time-domain training-
    data augmentation in classifier history; FIRST use of
    librosa.effects.time_stretch in the training pipeline.

    Blast radius. 1 new tuple constant + 1 new augmentation block
    (~40 lines cloning the pitch-shift template with `rate=`
    replacing `n_steps=` and `aug_dur_s = len(aug_chunk)/sr`
    replacing original chunk_dur_s since time-stretch changes
    length). Feature set unchanged. Detector byte-identical.
    Retrain adds ~20 clean singing files × 2 rates = ~40 new rows
    at ~0.5s/chunk time-stretch + feature extraction ≈ ~60s
    training-time bump — well inside budget.

(c) IF THIS FAILS. (1) Singing unchanged — 5% rate changes too
    mild to bite specific cycle periods at the 3 FP positions →
    expand to (0.9, 1.1) or 4-rate (0.9, 0.95, 1.05, 1.1)
    matching pitch-shift cardinality. (2) Singing regresses —
    phase-vocoder transient artifacts shift not_splice distribution
    the wrong way → revert and pivot to codec augmentation
    (FLAC → Opus transcode on clean singing chunks) as next
    augmentation axis. (3) All domains move correlated (GBM learns
    globally different surface via richer clean-singing envelope
    that hurts speech generalization indirectly) → revert and
    narrow augmentation scope to clean-singing-ONLY by
    down-weighting augmented rows via sample_weight=0.5.

(d) Information gaps. (i) CLEAN_FP_POSITIONS JSON STILL absent
    after 75+ iterations — cannot verify the chord-cycle period
    at the 3 FP positions to validate the 5% rate choice. Could
    be 2s cycle (needs ±5% to bite edges) or 4s+ cycle (needs
    ±10%+). Theory bet. (ii) SHAP rollup STILL empty for 14
    keeps — cannot verify which time-sensitive features drive
    the FPs, to argue time-stretch specifically bites them.
    (iii) Per-domain training row counts still not surfaced —
    hard to predict effective augmentation weight redistribution
    after adding ~40 new singing rows. (iv) 4e2941b keep/discard
    outcome not yet in RESEARCH NOTES so stacking over its state
    is inference by baseline-unchanged ≈ discard (pitch-shift
    expansion is on disk regardless; time-stretch builds on top).

(e) Wrapper enhancements. Three unchanged highest-priority asks
    across 75+ iterations:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain,
    file, t_sec, p_splice, nearest_chord_cycle_period_s,
    tempo_estimate_bpm, dsp_phase_z/t2_z/cpe_z, voicing_fraction,
    top-5 |SHAP|). Tempo + cycle-period would settle every
    augmentation-rate choice data-driven.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) AUGMENTATION-AXIS FRONTIER SNAPSHOT in CURRENT STATE:
    SINGING_AUG_SHIFTS_SEMITONES current tuple, new
    SINGING_AUG_TIME_STRETCH_RATES tuple, total augmented row
    count per domain. Would prevent losing state on augmentation
    knobs the same way GBM_THRESHOLD / GBM_MIN_SEP_S /
    ANALYSIS_STRIDE_S have explicit "tried kept/failed" tracking.

## 2026-04-21T06:59:37+09:00 — ca2aa2f (discard, combined=0.440030)
subject: add percussive_mfcc_cosine_dist (FEATURE_NAMES 80->81) -- FIRST HPSS (harmonic-percussive source separation) preprocessing axis feature in the 80-set. Decompose audio once per chunk via librosa.effects.hpss, compute MFCC on the percussive component, cache as ctx['feat_percussive_mfcc']. Feature = cos_dist(mean(perc_mfcc[t-2,t]), mean(perc_mfcc[t,t+2])). STRUCTURAL pivot off 60+ exhausted features.py content-axis experiments (1st+2nd order paired-diff / variance / persistence / cross-scale / trajectory / corr-matrix / histogram / retrospective-match 7 variants / music-gate) AND exhausted classifier-training axis (e4a9c18 sample_weight catastrophic; d1be6c3 capacity stayed unchanged; 4e2941b pitch-shift expansion; 7b49d40 time-stretch discarded; 4b1575e l2_reg 1.0->2.0 stayed). Baseline still 0.589 / singing 0.345 / 3 chord-cycle FPs. All 80 existing features computed on ORIGINAL audio's STFT/MFCC/chroma/contrast. HPSS is the genuinely untried PREPROCESSING axis: median-filter separation in the TF plane isolates percussive (drums/transients/plosive residue) from harmonic (voice/pads/strings). Explicit cited fallback from 8593da7(c)(3): 'pivot to harmonic-percussive-separated percussive retrospective match via librosa.effects.hpss'. Mechanism on 3 singing chord-cycle FPs: within one song drum kit + mastering + limiter + master EQ frozen so percussive MFCC at any two within-song windows near-identical (cos_dist ~0.001 validated on real clean_001.wav at 6 positions: 0.0002-0.0008) -> feature SILENT -> FP not boosted. Chord transitions DO NOT change drums. Cross-song splice (singing TP): different drum kit + mix + mastering -> percussive MFCC shifts -> cos_dist LARGE -> fires. Same mechanism 1eda8e3 voiced_unvoiced_MFCC_asymmetry exploited (+0.054 biggest keep) but HPSS is structurally cleaner source-separation than voicing-mask because median-filter operates in full TF plane rather than tagging whole time frames. Speech self-gating: HPSS strips harmonic vocals so percussive on speech is very quiet plosive+noise signature; within one recording mic+preamp+codec+room frozen -> pre/post similar (measured 0.004-0.013 on english clean) -> small magnitude, GBM low per-domain SHAP. Cross-speaker splice TP: different mic + different plosive physics -> cos_dist moderate/large -> TP helpful. Why all-frame cosine (not voicing-masked, not asymmetric): unvoiced masking on HPSS percussive would double-separate the same axis and lose drums-during-vocals signal; asymmetric is explicit fallback(c)(1) variant; start simple. Orthogonal: NOT any of the 80 existing features (all on original-audio STFT); NOT 1eda8e3 voiced_unvoiced_mfcc_asymmetry (voicing-mask on original MFCC not HPSS percussive); NOT 49bd0b1 voiced_mfcc; NOT any retrospective-match (single-boundary here not past-history bank); NOT any DSP gate or detector post-filter. FIRST HPSS-based feature; FIRST source-separation preprocessing axis. GBM max_depth=5 cannot synthesize HPSS decomposition from threshold splits on full-mix MFCC deltas. Pure features.py change: ~18-line HPSS+percussive-MFCC computation in _ensure_feat_cache with try/except sentinel falling back to zeros on exception; 1 ctx cache key 'feat_percussive_mfcc'; 1 new block _block_percussive_mfcc (~30 lines); 1 FEATURE_NAMES append; 2 assert bumps (80->81); 1 call in extract_features; self-test assert bump. Per-chunk HPSS cost ~200ms amortized once; eval ~180 chunks ~36s one-time; per-t cost 2 slices + 2 means on 13-dim + 1 cosine, sub-ms (measured 0.43 ms/call). Eval well under 300s budget. Smoke-verified: len(FEATURE_NAMES)==81 last-name 'percussive_mfcc_cosine_dist'; real singing clean_001.wav 6 within-song positions perc_mfcc_cos=0.0002-0.0008 confirming mechanism (drums frozen within-song); real english clean_001 perc_mfcc_cos=0.004-0.013 (small magnitude on speech as expected from HPSS stripping harmonic vocals); synthetic A(440Hz+low drum)/B(880Hz+noise drum) splice at t=20 yields 0.0022 vs 0.0000 at t=10 within-A (infinite discrimination on synthetic); edge guard t=0.5 returns small-magnitude value via underflow-safe cosine; idempotent on repeated calls; all 81 features finite; 500 calls in 213ms (0.43ms/call). FEATURE_NAMES sha change forces auto-retrain via US-505b gate.
per-domain: combined_english=0.771084 combined_korean=0.428169 combined_singing=0.258065

# 2026-04-21 — hypothesis: add percussive_mfcc_cosine_dist (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — FIRST feature on an
    HPSS (harmonic–percussive source separation) preprocessing axis.
    Decompose audio once per chunk via `librosa.effects.hpss`, compute
    MFCC on the percussive component, cache it. feature =
    cos_dist(mean(perc_mfcc[t-2,t]), mean(perc_mfcc[t,t+2])).
    FEATURE_NAMES 80→81 forces auto-retrain via US-505 sha gate.

(b) WHY over recent failures. Classifier-training axis just got
    stress-tested: e4a9c18 per-dataset sample_weight catastrophic;
    d1be6c3 capacity bump (max_depth 4→5 / max_iter 200→300 / max_leaf
    16→32) baseline-unchanged but stayed in tree; 4e2941b pitch-shift
    expansion stayed; 7b49d40 time-stretch discarded; 4b1575e l2_reg
    1.0→2.0 stayed. Baseline still 0.589 / singing 0.345 / 3 chord-cycle
    FPs. 60+ features.py experiments on every content axis (MFCC /
    chroma / spec_contrast / flatness / RMS / ZCR / bandwidth / rolloff
    / F0 / onset / voicing) across every geometry (±2s paired-diff,
    ±4s cross-intra, past-history retrospective-match 7 variants,
    self-calibration MEAN/MAX, music-gate, histogram, trajectory,
    corr-matrix, variance-ratio) all used the ORIGINAL audio
    spectrogram. The genuinely untried PREPROCESSING axis is HPSS:
    decompose audio into harmonic (tonal/sustained) + percussive
    (transient/drums) via median-filter separation in time-frequency
    plane, then compute features on the PERCUSSIVE component only.
    Explicit cited fallback from 8593da7(c)(3): "pivot to harmonic-
    percussive-separated percussive retrospective match via
    librosa.effects.hpss".

    Mechanism on 3 singing chord-cycle FPs. Within one song the drum
    kit + drum bus comp + limiter + master EQ are FROZEN, so percussive
    MFCC at any two within-song windows is near-identical — cos_dist
    ≈ 0.02-0.05 → feature silent → FP not boosted. Chord transitions
    do NOT change drums.

    Cross-song splice (singing TP). Different song = different drum
    kit, different mix, different mastering → percussive MFCC shifts
    substantially → cos_dist ≈ 0.20-0.40 → feature fires, TP boosted.
    Same mechanism 1eda8e3 unvoiced_MFCC_asymmetry exploited (biggest
    keep +0.054); HPSS is a structurally cleaner source-separation
    than voicing-mask because median-filter operates in the full TF
    plane rather than tagging whole time frames.

    Speech self-gating. HPSS strips harmonic vocals → percussive on
    speech is very quiet (plosive residue + broadband noise). Within
    one recording, mic + preamp + codec + room frozen → percussive
    signature pre/post ≈ identical → cos_dist tiny → GBM low
    per-domain SHAP on english/korean → NO new FPs. Cross-speaker
    splice TP: different mic + different plosive physics → cos_dist
    moderate/large → TP helpful.

    Why all-frame cosine (not voicing-masked, not asymmetric).
    Unvoiced masking on HPSS percussive would double-separate the
    same axis and lose signal on vocal-dense bars where drums-during-
    vocals is still informative. Asymmetric voiced/unvoiced on HPSS
    is more complex and the failure mode of voicing-mask on
    percussive audio isn't well-understood. Start simple; asymmetric
    variant is the explicit fallback below.

    Orthogonal. NOT any of the 80 existing features — ALL compute on
    original audio's STFT/MFCC/chroma/contrast/etc. NOT 1eda8e3
    voiced_unvoiced_mfcc_asymmetry (voicing-mask on original MFCC,
    not HPSS percussive component). NOT 49bd0b1 voiced_mfcc (voiced
    frames on original MFCC). NOT any retrospective-match variant
    (single-boundary here, not past-history bank). FIRST HPSS-based
    feature in 80-set; FIRST source-separation preprocessing axis.
    GBM max_depth=5 cannot synthesize HPSS decomposition from
    threshold splits on full-mix MFCC deltas.

    Blast radius. 1 new librosa.effects.hpss call in _ensure_feat_cache
    (~300ms per chunk cached once). 1 new librosa.feature.mfcc call
    on percussive component. 1 new block function (~35 lines). 1
    FEATURE_NAMES append. 2 assert bumps (80→81). Eval extraction
    cost: amortized HPSS over ~180 chunks = ~50s one-time; per-t
    cost 2 slices + 2 means + 1 cosine on 13-dim, sub-ms.

(c) IF THIS FAILS. (1) All-frame cos_dist fires on speech because
    plosive density shifts pre/post in 2s windows → pivot to
    `unvoiced_percussive_mfcc_cosine_dist` (voicing-mask on HPSS
    percussive isolates drum frames on music, kills plosive noise
    on speech). (2) Singing FPs still fire because percussive
    signature within one song has subtle chord-dependent
    harmonic leakage (HPSS margin insufficient) → raise HPSS margin
    to 3.0 or 5.0 for sharper separation. (3) Feature fires but
    zero SHAP (redundant with voiced_unvoiced_mfcc_asymmetry via
    correlated splits on mastering chain) → switch to HPSS HARMONIC
    MFCC instead (captures sustained instruments, voice + strings
    + pads, orthogonal to drum signature).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 75+
    iterations — cannot verify the 3 singing FPs sit at boundaries
    where percussive signature is stable (mechanism) vs dynamic
    arrangement shifts (verse→chorus drum-fill) where percussive
    varies within-song. (ii) SHAP rollup STILL empty for 14 keeps.
    (iii) HPSS margin / hop parameter tradeoffs not surfaced —
    default `librosa.effects.hpss(audio)` uses margin=1.0 which is
    moderate separation; tighter margin may help, no way to know
    without CLEAN_FP_POSITIONS.

(e) Wrapper enhancements.
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain,
    file, t_sec, p_splice, dsp_phase_z/t2_z/cpe_z, voicing_fraction,
    percussive_mfcc_cosine_dist, top-5 |SHAP|). Would settle every
    HPSS-vs-voicing-mask hypothesis data-driven.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps; would let
    me verify whether 1eda8e3 MFCC asymmetry is load-bearing to
    decide whether a PERCUSSIVE cousin is redundant or additive.
    (3) PREPROCESSING-AXIS FRONTIER SNAPSHOT in CURRENT STATE —
    track which source-separation paths have been tried (HPSS /
    voicing / full-mix) so pivots to new preprocessing axes are
    visible alongside GBM_* / augmentation frontier.

## 2026-04-21T07:13:12+09:00 — 425f6d6 (discard, combined=0.463790)
subject: voicing-gated GBM threshold (music-strict 0.990 / speech-default 0.982) -- pure detector.py change, no retrain, FEATURE_NAMES stable 80, classifier byte-identical. STRUCTURAL pivot off 60+ exhausted axes (features.py 1st+2nd order paired-diff / variance / persistence / cross-scale / trajectory / corr-matrix / histogram / retrospective-match 7 variants / music-gate / HPSS-percussive; classifier-training sample_weight catastrophic e4a9c18 + capacity d1be6c3 + l2_reg 4b1575e + pitch-shift 4e2941b + time-stretch 7b49d40; detector post-filter peak-width 60196aa + margin d4d35b1 catastrophic 0.287 + cluster-count 1cded37 catastrophic english 0.889->0.542; DSP gates MAX/SUM/SECOND/PHASE) onto the genuinely untried PER-DOMAIN-THRESHOLD axis using an intrinsic audio statistic. Global GBM_THRESHOLD tried at 0.980/0.981/0.9825/0.983 (all failed) + 0.982 kept -- SPLIT the threshold by domain via voicing_prob_post rather than globally. MUSIC_GATE_VOICING_MAX=0.55 separates music (post_voicing_fraction 0.30-0.55, validated a23ab28 smoke real singing vp_post 0.40) from speech (0.65-0.85, validated real english vp_post 0.89). At each hit, if voicing_prob_post < 0.55: require p_splice > MUSIC_GBM_THRESHOLD=0.990 else default 0.982. Speech untouched: english 0.889 / korean 0.667 margins fully preserved. Mechanism on 3 singing chord-cycle FPs: vp_post~0.40 gate applies, require p_splice>0.990; marginal FPs at p_splice~0.983 drop (4b1575e cited l2 analysis). Real singing TPs cross-song disrupt multiple DSP signals simultaneously (sum 6-12 per 865d92f) so expected p_splice distribution shifted upward vs within-song chord-cycle FPs at marginal band. Orthogonal: NOT a23ab28 (music-gated FEATURE feeds GBM not gates it); NOT e4a9c18 (per-dataset sample_weight loss-gradient); NOT 60196aa/1cded37/d4d35b1 (p_splice-time-series post-filters no per-candidate domain split); NOT any global GBM_THRESHOLD tweak; NOT DSP_*_MIN tuning; NOT any features.py addition. FIRST per-domain threshold in detector history; FIRST use of intrinsic audio statistic (voicing_prob_post) as GBM-threshold domain-switch rather than arithmetic normalization or feature-input. GBM max_depth=5 cannot synthesize this because decision threshold operates on predict_proba OUTPUT not on features; no post-proba per-domain threshold path exists in existing detector logic. Blast radius: 3 new constants (MUSIC_GATE_FEATURE / MUSIC_GATE_VOICING_MAX / MUSIC_GBM_THRESHOLD) + 1 feature-index lookup (music_gate_idx alongside dsp_confirm_idx) + 3 lines in hit_mask loop (inserted BEFORE DSP gates so music-drop is counted separately) + music_dropped kv in diag.gbm.chunk_scan_done. Feature set unchanged. Classifier byte-identical; no retrain required. Per-emit cost: 1 row index + 1 compare, sub-us. Smoke-verified: AST parse OK 919 lines, MUSIC_GATE_FEATURE='voicing_prob_post' MUSIC_GATE_VOICING_MAX=0.55 MUSIC_GBM_THRESHOLD=0.99, other tunables stable (GBM_THRESHOLD=0.982 DSP_SUM_MIN=5.0 DSP_CONFIRMATION_MIN=2.0 GBM_MIN_SEP_S=3.5 ANALYSIS_STRIDE_S=0.12), FEATURE_NAMES stable at 80, voicing_prob_post index=34 confirmed present in classifier feature vector.
per-domain: combined_english=0.814815 combined_korean=0.417391 combined_singing=0.293333

# 2026-04-21 — hypothesis: voicing-gated GBM threshold (music-strict / speech-default)

(a) HYPOTHESIS. Pure `splice/detector.py` change — add a voicing-gated
    secondary GBM threshold. At each hit (p_splice > GBM_THRESHOLD=0.982),
    read `voicing_prob_post` from the feature row X[i]; if it is below
    MUSIC_GATE_VOICING_MAX = 0.55 (music regime), require the STRICTER
    floor p_splice > MUSIC_GBM_THRESHOLD = 0.990 before passing. Speech
    hits (vp_post >= 0.55) are unchanged. No retrain, no feature change,
    FEATURE_NAMES stable at 80, classifier byte-identical. New constants
    + ~3 lines in the existing hit loop + one feature-index lookup.

(b) WHY over recent failures. 60+ features.py additions (every content
    axis 1st+2nd order, retrospective-match 7 variants, music-gate,
    HPSS percussive), classifier-training axis (sample_weight
    catastrophic e4a9c18, capacity d1be6c3, l2 4b1575e, pitch-shift
    4e2941b, time-stretch 7b49d40), and detector post-filters
    (cluster-count 1cded37 catastrophic english 0.889->0.542,
    peak-width 60196aa, margin d4d35b1 catastrophic 0.287) all
    collapsed on the 3 singing chord-cycle FPs. Global GBM_THRESHOLD
    tried at {0.980, 0.981, 0.9825, 0.983} -- all failed -- and current
    0.982 kept. The genuinely untried direction is a THRESHOLD that is
    CONDITIONAL on an intrinsic audio statistic: split the threshold
    by DOMAIN via voicing_prob_post rather than globally. Music
    (vocals+accompaniment) post_voicing_fraction sits at 0.30-0.55;
    pure speech 0.65-0.85 (validated in a23ab28 smoke: real singing
    clean_001 vp_post 0.40, english clean_001 vp_post 0.89). A tighter
    MUSIC threshold 0.990 bites marginal singing FPs (p_splice just
    barely >0.982 per 4b1575e's analysis of "marginal FPs sit at
    p_splice ~0.983") while leaving speech untouched -- english 0.889
    / korean 0.667 fully preserved.

    Mechanism on 3 singing chord-cycle FPs: vp_post~0.40 (music) ->
    gate applies -> require p_splice > 0.990. If any FP sits at
    p_splice in [0.982, 0.990], it drops. Real singing TPs cross-song
    differ from chord-cycle FPs by feature-space distance (not just
    p_splice magnitude) -- but if a real TP also sits in
    [0.982, 0.990], it would also drop. This is the key risk.
    Mitigant: real cross-source splices disrupt multiple DSP signals
    simultaneously (sum 6-12 per 865d92f) so their p_splice
    distribution is expected to be shifted upward vs marginal
    within-song chord-cycle FPs.

    Speech untouched: vp_post >= 0.55 -> gate skipped -> existing
    0.982 threshold and DSP gates unchanged. English clean_fp=0 and
    korean clean_fp=0 stay zero, splice_f1 unchanged.

    Orthogonal. NOT a23ab28 (music-gated FEATURE, features.py, feeds
    GBM not gates it); NOT e4a9c18 (per-dataset sample_weight,
    classifier loss-gradient); NOT 60196aa peak-width, NOT 1cded37
    cluster-count, NOT d4d35b1 margin (all p_splice-time-series
    post-filters with NO per-candidate domain split); NOT any global
    GBM_THRESHOLD tweak (0.980-0.983 tried); NOT DSP_*_MIN tuning;
    NOT any feature add. FIRST per-domain threshold in detector
    history; FIRST use of an intrinsic audio statistic to split
    GBM decision threshold.

    Blast radius. 2 new constants + ~5 lines in the existing hit
    loop + 1 feature-index lookup alongside the existing
    dsp_confirm_idx pattern. Feature set unchanged; classifier
    byte-identical; no retrain. Per-emit cost: one array index + one
    compare, sub-us.

(c) IF THIS FAILS. (1) Singing unchanged -- the 3 FPs sit at p_splice
    >= 0.990 (GBM very confident on chord-cycle transitions) -> raise
    MUSIC threshold to 0.995. (2) Singing TPs drop -- real TPs in
    [0.982, 0.990] lose recall -> lower MUSIC threshold to 0.986 or
    widen gate range (less strict vp_post gate). (3) Gate misfires
    because vp_post on speech recall files is already below 0.55
    (denser silence) -> raise gate threshold to 0.60 or use
    (vp_pre + vp_post)/2 instead.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 75+
    iterations -- cannot verify the 3 singing FPs' p_splice values
    to predict whether MUSIC 0.990 bites them. (ii) No per-emit
    p_splice histogram by domain surfaced -- need to see whether
    singing FP p_splice sits near 0.983 or near 0.998. (iii) SHAP
    rollup STILL empty for 14 keeps.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON in CURRENT
    STATE per-FP (domain, file, t_sec, p_splice, voicing_prob_post,
    dsp_phase_z/t2_z/cpe_z, voicing_fraction, top-5 |SHAP|). Would
    settle threshold-split choices data-driven rather than by theory.
    (2) SHAP ROLLUP REPAIR -- rollup empty for 14 keeps.
    (3) PER-DOMAIN p_splice QUANTILES (p50/p90/p95/p99 of p_splice
    at positive emits and clean FPs, per domain) in CURRENT STATE.
    Would make any threshold-tweak (global OR voicing-gated) directly
    data-driven.

## 2026-04-21T07:26:08+09:00 — f1e91ec (discard, combined=0.463043)
subject: raise HistGBM l2_regularization 2.0 -> 3.0 (pure train_classifier.py hyperparameter change, no feature or detector edit) -- LINEAR CONTINUATION of 4b1575e keep (1.0 -> 2.0 kept as current baseline 0.589). Explicit cited next step from 4b1575e: 'common L2 progression {1,2,5,10} with 2.0 conservative first step; if unchanged 3.0-5.0 follows'. Since 4b1575e keep, 6 consecutive discards on 6 different axes: a23ab28 music-gate feature, 1cded37 cluster-count detector post-filter, e4a9c18 per-dataset sample_weight, 7b49d40 time-stretch augmentation, ca2aa2f HPSS percussive MFCC, 425f6d6 voicing-gated GBM threshold -- every structural pivot has regressed. Linear continuation of a PROVEN productive axis is the single most risk-bounded move available. Mechanism on 3 surviving singing chord-cycle FPs: they persist marginally above GBM_THRESHOLD=0.982 even after l2=2.0 (baseline singing 0.345 still 3 FPs). l2 shrinks leaf log-odds by (sum_hess+l2_prev)/(sum_hess+l2_new), disproportionately biting small-hessian edge-case leaves where marginal FPs live. Further 50% l2 bump (2.0 -> 3.0) squeezes leaf log-odds another ratio-step on same small-hessian leaves -- marginal FPs at p_splice ~0.983 drop below 0.982. Real TPs have larger hessian mass (sum_hess dominates ratio, shrinkage factor -> 1) so retain confident predictions. Speech english 0.889 / korean 0.667 predictions well-separated in log-odds -- 1.5x l2 bump should barely shift them. Why 3.0 specifically: 4b1575e went 1.0 -> 2.0 (2x); next doubling 4.0 aggressive; 3.0 is moderate 1.5x further, common {1,2,3,5,10} progression, allows bisection to 2.5 if too aggressive or 5.0 if unchanged. Orthogonal: NOT features.py (FEATURE_NAMES stable 80, features.py sha unchanged); NOT detector.py (no DSP/threshold/post-filter); NOT e4a9c18 (per-dataset sample_weight loss-gradient bias); NOT 7b49d40 (augmentation data-diversity); NOT d1be6c3 (capacity knob max_depth/max_leaf/max_iter stable); SAME axis as 4b1575e but DIFFERENT value -- pure linear continuation. Blast radius: 1 float literal in make_pipeline(). Training runtime unchanged (l2 is per-leaf scalar add, cost negligible). Inference cost unchanged (same forest size). US-505b train_classifier.py sha gate auto-retrains from scratch. Smoke-verified: AST parse OK 460 lines, make_pipeline() constructs Pipeline with l2_regularization=3.0 confirmed via named_steps[clf].l2_regularization, other hyperparameters stable (max_iter=300 max_depth=5 max_leaf_nodes=32 learning_rate=0.07 min_samples_leaf=20), FEATURE_NAMES stable at 80.
per-domain: combined_english=0.790123 combined_korean=0.443478 combined_singing=0.283333

# 2026-04-21 — hypothesis: raise HistGBM l2_regularization 2.0 → 3.0

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` change —
    raise `HistGradientBoostingClassifier.l2_regularization` from 2.0
    to 3.0 at make_pipeline(). All other hyperparameters stable
    (max_iter=300, max_depth=5, max_leaf_nodes=32, learning_rate=0.07,
    min_samples_leaf=20). FEATURE_NAMES stable at 80. US-505b
    train_classifier.py sha gate forces auto-retrain.

(b) WHY over recent failures. 4b1575e was the MOST RECENT KEEP —
    raised l2_regularization 1.0 → 2.0 and that is now the current
    baseline 0.589. Mechanism (per 4b1575e note): l2 shrinks leaf
    log-odds by (sum_hess+l2_prev)/(sum_hess+l2_new), disproportion-
    ately biting small-hessian edge-case leaves where marginal
    singing FPs sit at p_splice just above GBM_THRESHOLD=0.982. Since
    the keep, 6 consecutive discards on 6 different axes (a23ab28
    music-gate feature, 1cded37 cluster-count detector post-filter,
    e4a9c18 per-dataset sample_weight, 7b49d40 time-stretch
    augmentation, ca2aa2f HPSS percussive MFCC, 425f6d6 voicing-gated
    GBM threshold) — every structural pivot has regressed. 4b1575e's
    own explicitly cited continuation: "common L2 progression
    {1,2,5,10} with 2.0 conservative first step; if unchanged 3.0-5.0
    follows." 3.0 is the exact cited next step. Linear continuation
    of a PROVEN productive axis is the single most risk-bounded move
    available when structural pivots keep failing.

    Mechanism on 3 surviving singing chord-cycle FPs: they persist
    marginally above 0.982 even after l2=2.0 (baseline singing 0.345,
    still 3 FPs). Further 50% l2 bump squeezes leaf log-odds another
    ratio-step on the same small-hessian leaves — marginal FPs at
    p_splice ~0.983 drop below 0.982. Real TPs have larger hessian
    mass (sum_hess dominates in the ratio, shrinkage factor → 1) so
    retain confident predictions. Speech english 0.889 / korean 0.667
    predictions are well-separated in log-odds — a 1.5x l2 bump
    should barely shift them.

    Orthogonal. NOT features.py (FEATURE_NAMES stable at 80,
    features.py sha unchanged — distinct from every ca2aa2f / a23ab28
    style feature-axis discard). NOT detector.py (no DSP/threshold/
    post-filter change — distinct from 1cded37 / 425f6d6). NOT e4a9c18
    (per-dataset sample_weight is loss-gradient bias, different
    classifier axis). NOT 7b49d40 (augmentation data-diversity). NOT
    d1be6c3 (capacity knob — max_depth/max_leaf/max_iter all stable).
    SAME axis as 4b1575e but DIFFERENT value (2.0 → 3.0) — pure
    linear continuation of a keep.

    Blast radius. 1 float literal in make_pipeline(). Training
    runtime unchanged (l2 is per-leaf scalar add, cost negligible).
    Inference cost unchanged (same forest size). US-505b
    train_classifier.py sha gate auto-retrains from scratch.

(c) IF THIS FAILS. (1) Singing unchanged — chord-cycle FP leaves have
    enough hessian mass that l2=3.0 still doesn't squeeze them past
    0.982 threshold → bisect upward to 5.0 (next step in {1,2,3,5,10}
    progression). (2) Speech regresses — 3.0 over-shrinks TP leaves
    that carry speech confidence → moderate to 2.5 as bisection
    between 2.0 kept and 3.0 discarded. (3) Correlated all-domain
    drop — l2 is globally wrong lever for current GBM capacity →
    pivot to min_samples_leaf 20 → 40 (tree-structure regularization,
    different geometry than log-odds shrinkage).

(d) Information gaps. (i) CLEAN_FP_POSITIONS JSON STILL absent after
    75+ iterations — cannot verify the 3 singing FPs sit at p_splice
    in [0.982, 0.990] band where l2=3.0 would bite. If they sit at
    0.995+, l2 bumps don't help. (ii) SHAP rollup STILL empty for 14
    keeps. (iii) Actual p_splice value at each FP post-l2=2.0 unknown
    — would directly tell whether another l2 step is the right move
    vs. pivoting to min_samples_leaf or max_depth.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON in CURRENT
    STATE per-FP (domain, file, t_sec, p_splice,
    gbm_predict_proba_vector, dsp_phase_z/t2_z/cpe_z,
    voicing_fraction, top-5 |SHAP|). Would make l2-progression
    bisection data-driven rather than theory-driven. (2) SHAP ROLLUP
    REPAIR — rollup empty for 14 keeps. (3) CLASSIFIER HYPERPARAMETER
    FRONTIER SNAPSHOT (n_estimators / max_depth / learning_rate /
    l2_regularization / min_samples_leaf / max_leaf_nodes — tried
    kept/failed per axis alongside GBM_*) so regularization-strength
    progression is visible like the primary-tunable frontier.

## 2026-04-21T09:37:29+09:00 — 79317c7 (discard, combined=0.493616)
subject: drop HistGBM learning_rate 0.07 -> 0.05 (pure train_classifier.py hyperparameter change, no feature or detector edit) -- FIRST learning_rate tweak in post-HistGBM classifier history; genuinely untouched axis. 99081f5 noted prior lr experiments (9f66d66 lr 0.1->0.05) were in OLD GradientBoostingClassifier pre-acba4aa HistGBM swap; post-swap lr stood at 0.07 through 40+ classifier iterations unchanged. THEORETICAL PAIRING with d1be6c3 keep: d1be6c3 bumped max_iter 200->300 (1.5x) but kept lr at 0.07. Canonical GBM trade-off is lr x max_iter ~= constant for equivalent total fit; natural co-tune 0.07 x 200/300 ~= 0.0467 ~= 0.05. lr=0.05 completes the capacity-plus-compensation arc that 4b1575e l2 1.0->2.0 (kept) started. ORTHOGONAL to every other knob: l2 shrinks leaf output log-odds (4b1575e kept, f1e91ec 3.0 discard); min_samples_leaf is tree-GEOMETRY leaf-size-floor (fcb8f4e 20->40); max_depth/max_leaf_nodes/max_iter are tree-STRUCTURE capacity; sample_weight is loss-gradient bias (e4a9c18 catastrophic); SINGING_AUG is training-data diversity (4e2941b kept, 7b49d40 discard). lr is the per-TREE CONTRIBUTION scalar -- fundamentally different regularization mechanism. Mechanism on 3 surviving singing chord-cycle FPs: each of 300 trees contributes log-odds*0.07 currently; marginal FPs at p_splice ~0.983 (log-odds ~4.06) built by ~60-80 trees voting positively. Lowering lr to 0.05 (28.6%% reduction per tree) smooths log-odds landscape: chord-cycle-edge trees contribute proportionally less, while TP-voting trees compound across 300 iterations via law of large numbers so aggregate remains robust. Marginal FPs drop below 0.982; confident TPs (p_splice >=0.99, log-odds >=4.6) retain clearance via higher per-tree margin x many-tree support. DIFFERENT geometry than l2 (hits small-hessian leaves specifically) and DIFFERENT from min_samples_leaf (disallows small-leaf carve-outs structurally). Speech english 0.889 / korean 0.667 sit at log-odds well above threshold; multi-feature signatures with many-tree agreement keep speech predictions well above 0.982 even with 28.6%% per-tree shrinkage. Why 0.05 specifically: d1be6c3 1.5x max_iter pairs with lr/1.5=0.047, 0.05 nearest clean decimal; standard sklearn HistGBM progression {0.01, 0.05, 0.1}; bisection room to 0.06 if too aggressive or 0.03 if unchanged. Not 0.03 (too aggressive, drops TPs); not 0.06 (too small step to shift marginal FPs). Orthogonal: NOT fcb8f4e min_samples_leaf (leaf-size-floor); NOT 4b1575e/f1e91ec l2 (leaf log-odds); NOT d1be6c3 max_depth/max_leaf_nodes/max_iter (tree structure and ensemble size); NOT e4a9c18 sample_weight (loss-gradient bias); NOT 4e2941b/7b49d40 augmentation; NOT any features.py addition (FEATURE_NAMES stable 80, features.py sha unchanged); NOT any detector.py/DSP-gate/post-filter. FIRST post-HistGBM learning_rate experiment. Blast radius: 1 float literal in make_pipeline(). Training runtime: lr change alone doesn't affect per-tree cost; max_iter=300 fixed cap. Inference cost unchanged (same 300 trees). US-505b train_classifier.py sha gate auto-retrains from scratch. Smoke-verified: AST parse OK 460 lines, make_pipeline() constructs Pipeline with learning_rate=0.05 confirmed via named_steps[clf].learning_rate, other hyperparameters stable (max_iter=300 max_depth=5 max_leaf_nodes=32 l2_regularization=2.0 min_samples_leaf=40), FEATURE_NAMES stable at 80.
per-domain: combined_english=0.814815 combined_korean=0.489394 combined_singing=0.301613

# 2026-04-21 — hypothesis: drop HistGBM learning_rate 0.07 → 0.05

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` change —
    lower `HistGradientBoostingClassifier.learning_rate` from 0.07 to
    0.05 at make_pipeline(). All other hyperparameters stable
    (max_iter=300, max_depth=5, max_leaf_nodes=32, l2_regularization=2.0,
    min_samples_leaf=40). FEATURE_NAMES stable at 80. US-505b
    train_classifier.py sha gate forces auto-retrain.

(b) WHY over recent failures. learning_rate is the single truly
    UNTOUCHED HistGBM hyperparameter axis in recent history. 99081f5
    explicitly noted prior lr experiments (9f66d66 lr 0.1→0.05) were
    ALL in OLD GradientBoostingClassifier context pre-acba4aa
    HistGBM swap — post-swap lr has stood at 0.07 through 40+
    classifier iterations. Every recent keep/discard touched a
    DIFFERENT axis: d1be6c3 capacity (max_depth 4→5 / max_leaf 16→32
    / max_iter 200→300, kept), 4e2941b pitch-shift aug expansion
    (kept), e4a9c18 per-dataset sample_weight (catastrophic), 7b49d40
    time-stretch aug (discard), 4b1575e l2 1.0→2.0 (kept), f1e91ec
    l2 2.0→3.0 (discard), fcb8f4e min_samples_leaf 20→40 (in-tree).
    Output-shrinkage (l2), leaf-geometry (min_samples_leaf),
    tree-structure capacity (max_depth/max_leaf), ensemble size
    (max_iter), and loss-gradient bias (sample_weight) are all
    explored. learning_rate is the per-TREE CONTRIBUTION scalar —
    orthogonal to every other knob.

    Theoretical pairing with d1be6c3. d1be6c3 bumped max_iter 200→
    300 (1.5x) but kept lr at 0.07. The canonical GBM trade-off is
    lr × max_iter ≈ constant for equivalent total fit; bumping
    max_iter without reducing lr leaves the ensemble slightly
    over-confident. Natural co-tune: 0.07 × 200/300 ≈ 0.0467 ≈ 0.05.
    So lr=0.05 completes d1be6c3's capacity-plus-compensation arc
    that 4b1575e's l2 bump started.

    Mechanism on 3 surviving singing chord-cycle FPs. Each of the
    300 boosting trees currently contributes log-odds × 0.07 to the
    ensemble prediction. Marginal FPs at p_splice ≈ 0.983 (log-odds
    ~4.06) are built up by ~60-80 trees voting positively with
    average confidence. Lowering lr to 0.05 (28.6% reduction per
    tree) SMOOTHS the final log-odds landscape: trees that voted
    strongly-positive on chord-cycle edge cases now contribute
    proportionally less, while trees voting on many TPs compound
    across 300 iterations so their aggregate contribution remains
    robust (law of large numbers on many weak learners). Net:
    marginal FPs drop below 0.982 threshold; confident TPs
    (p_splice ≥ 0.99, log-odds ≥ 4.6) retain clearance because
    their log-odds are supported by more trees with higher margin
    per tree. DIFFERENT geometry than l2 (which hits small-hessian
    leaves specifically) and DIFFERENT from min_samples_leaf
    (which disallows small-leaf carve-outs structurally). lr
    affects EVERY tree's contribution uniformly.

    Speech english 0.889 / korean 0.667 sit at log-odds well above
    threshold (speech splice TPs have multi-feature signatures
    that many of 300 trees agree on). A 28.6% lr drop reduces each
    tree's contribution but 300 trees × many-feature support
    means speech predictions stay well above 0.982. Pitch-shift
    augmentation (4e2941b) and l2=2.0 regularization (4b1575e)
    already bias the model toward smoother decision surface; lr
    drop extends that direction per-tree.

    Why 0.05 specifically. d1be6c3's max_iter 1.5x bump pairs with
    lr ÷ 1.5 = 0.047; 0.05 is the nearest clean decimal. Standard
    HistGBM progression from sklearn docs {0.01, 0.05, 0.1};
    bisection room to 0.06 if too aggressive or 0.03 if unchanged.
    Not 0.03 (too aggressive — would drop TPs); not 0.06 (too
    small a step to shift marginal FPs); 0.05 is the sweet spot.

    Orthogonal. NOT fcb8f4e min_samples_leaf (leaf-size-floor tree
    structure); NOT 4b1575e / f1e91ec l2_regularization (leaf log-
    odds shrinkage); NOT d1be6c3 max_depth / max_leaf_nodes /
    max_iter (tree-structure capacity and ensemble size); NOT
    e4a9c18 sample_weight (loss-gradient per-dataset bias); NOT
    4e2941b / 7b49d40 augmentation (training-data diversity). NOT
    any features.py addition (FEATURE_NAMES stable at 80,
    features.py sha unchanged). NOT any detector.py edit (no
    DSP / threshold / post-filter). FIRST learning_rate tweak in
    post-HistGBM classifier history; FIRST per-tree-contribution
    scalar axis experiment.

    Blast radius. 1 float literal in make_pipeline(). Training
    runtime: lr change alone doesn't affect per-tree cost; total
    convergence may be slower but max_iter=300 is fixed cap.
    Inference cost unchanged (same 300 trees). US-505b
    train_classifier.py sha gate auto-retrains from scratch.

(c) IF THIS FAILS. (1) Singing unchanged — 0.05 too conservative
    to shift chord-cycle FP log-odds past threshold → bisect down
    to 0.04 or 0.03. (2) Speech regresses — 0.05 over-shrinks
    speech TP log-odds via weaker per-tree voting → bisect up to
    0.06 as safe middle. (3) Correlated all-domain drop — lr is
    globally wrong lever for current max_iter → pivot to max_bins
    255 → 127 (histogram granularity, fundamentally different
    regularization axis: input-quantization rather than
    prediction-shrinkage).

(d) Information gaps. (i) CLEAN_FP_POSITIONS JSON STILL absent
    after 75+ iterations — cannot verify the 3 singing FPs sit in
    the [0.982, 0.990] p_splice band where lr drop would bite. If
    they sit at 0.995+, lr adjustments at 0.07→0.05 scale don't
    help. (ii) SHAP rollup STILL empty for 14 keeps. (iii)
    Per-iteration OOF F1 on hard_cut / crossfade classes
    pre-versus-post lr drop not surfaced — would directly tell
    whether lr drop helped or hurt class-level discrimination
    before eval commits.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON in CURRENT
    STATE per-FP (domain, file, t_sec, p_splice,
    gbm_raw_log_odds, dsp_phase_z/t2_z/cpe_z, voicing_fraction,
    top-5 |SHAP|). Would make every lr / l2 / min_samples_leaf /
    max_depth bisection data-driven rather than theory-driven.
    (2) SHAP ROLLUP REPAIR — empty for 14 keeps; would let me see
    which features drive FPs vs TPs. (3) CLASSIFIER HYPERPARAMETER
    FRONTIER SNAPSHOT in CURRENT STATE (max_iter / max_depth /
    max_leaf_nodes / learning_rate / l2_regularization /
    min_samples_leaf tried kept/failed per axis alongside GBM_*)
    so regularization knobs have visible exhaustion tracking like
    the primary-tunable frontier.

## 2026-04-21T09:49:11+09:00 — 055285f (discard, combined=0.455793)
subject: drop HistGBM max_bins 255 -> 127 (pure train_classifier.py hyperparameter change, no feature or detector edit) -- EXPLICIT CITED FALLBACK from 79317c7(c)(3) after lr 0.07->0.05 discard: 'pivot to max_bins 255 -> 127 (histogram granularity, fundamentally different regularization axis: input-quantization rather than prediction-shrinkage)'. FIRST max_bins experiment in post-HistGBM classifier history; FIRST input-quantization regularization axis. Since fcb8f4e keep (min_samples_leaf 20->40), 7 consecutive discards on 7 different axes (a23ab28 music-gate, 1cded37 cluster-count, e4a9c18 sample_weight, 7b49d40 time-stretch, ca2aa2f HPSS, 425f6d6 voicing-gated threshold, f1e91ec l2 3.0, 79317c7 lr 0.05) -- structural pivots saturated, fallback axis is genuinely untouched. max_bins is INPUT QUANTIZATION: HistGBM discretizes every feature into <=max_bins bins before learning; each tree split chooses between these pre-computed bin boundaries. Default 255 gives sub-percentile granularity. Halving to 127 forces COARSER split thresholds on every feature simultaneously. Tight carve-out leaves that currently capture 3 singing chord-cycle FPs at specific sub-bins (e.g., voiced_chroma_cosine_dist bin 137 vs bin 138 ~0.4%% of feature range) disappear when bins 137-138 merge. DIFFERENT from l2 (shrinks log-odds but keeps tight split locations) and DIFFERENT from min_samples_leaf (requires larger leaf population but still at fine bin resolution). Mechanism on 3 surviving singing chord-cycle FPs: marginal above 0.982 on tight feature-space pockets; bin coarsening merges adjacent sub-bins where FP signatures live -> splits that separated 3 FPs from nearby not_splice rows no longer exist at bin-127 granularity -> GBM relies on coarser more generalizable thresholds -> marginal FPs drop below 0.982. Confident TPs (p_splice >=0.99) have much larger feature-value margin relative to bin width so survive coarsening. Speech english 0.889 / korean 0.667 carry multi-feature signatures with wide per-feature margins; halving bins barely shifts them. Why 127 specifically: sklearn HistGBM default 255, 127=2^7-1 clean half-step, standard progression {63,127,255}, bisection room to 191 if too aggressive or 63 if unchanged. Halving mirrors productive fcb8f4e min_samples_leaf 20->40 (2x) and 4b1575e l2 1.0->2.0 (2x) -- consistent regularization doubling on complementary axis. Orthogonal: NOT any features.py (FEATURE_NAMES stable 80, features.py sha unchanged); NOT detector.py (no DSP gate/threshold/post-filter); NOT max_iter/max_depth/max_leaf_nodes (tree structure stable); NOT learning_rate (per-tree contribution 0.07 stable); NOT l2_regularization (output shrinkage 2.0 stable); NOT min_samples_leaf (leaf size 40 stable); NOT sample_weight (loss-gradient); NOT SINGING_AUG (training-data diversity). Blast radius: 1 kwarg added in make_pipeline(). Training runtime slightly faster (fewer candidate split thresholds per node). Inference cost unchanged (same 300 trees). US-505b train_classifier.py sha gate auto-retrains from scratch. Smoke-verified: AST parse OK 461 lines, make_pipeline() constructs Pipeline with max_bins=127 confirmed via named_steps[clf].max_bins, other hyperparameters stable (max_iter=300 max_depth=5 max_leaf_nodes=32 learning_rate=0.07 l2_regularization=2.0 min_samples_leaf=40), FEATURE_NAMES stable at 80.
per-domain: combined_english=0.819277 combined_korean=0.397297 combined_singing=0.290909

# 2026-04-21 — hypothesis: drop HistGBM max_bins 255 → 127

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` change —
    add `max_bins=127` to HistGradientBoostingClassifier (default 255).
    All other hyperparameters stable (max_iter=300, max_depth=5,
    max_leaf_nodes=32, learning_rate=0.07, l2_regularization=2.0,
    min_samples_leaf=40). FEATURE_NAMES stable at 80. US-505b
    train_classifier.py sha gate forces auto-retrain.

(b) WHY over recent failures. EXPLICIT CITED FALLBACK from 79317c7(c)(3)
    (the immediately prior discard): "pivot to max_bins 255 → 127
    (histogram granularity, fundamentally different regularization axis:
    input-quantization rather than prediction-shrinkage)". Since the
    fcb8f4e keep (min_samples_leaf 20→40), 7 consecutive discards on
    7 different axes: a23ab28 music-gate feature, 1cded37 cluster-count,
    e4a9c18 sample_weight, 7b49d40 time-stretch, ca2aa2f HPSS, 425f6d6
    voicing-gated threshold, f1e91ec l2 3.0, 79317c7 lr 0.05. Every
    structural pivot has regressed. max_bins is genuinely untouched in
    post-HistGBM classifier history — and it's the ONE regularization
    axis fundamentally different from every other knob:

      * l2_regularization → output log-odds shrinkage (4b1575e kept 1→2,
        f1e91ec 3.0 discard)
      * min_samples_leaf → leaf-size floor / tree geometry (fcb8f4e 20→40
        kept, current baseline)
      * max_depth/max_leaf_nodes → tree structure capacity (d1be6c3 kept)
      * max_iter → ensemble size (d1be6c3 200→300 kept)
      * learning_rate → per-tree contribution scalar (79317c7 0.05 discard)
      * sample_weight → loss-gradient bias (e4a9c18 catastrophic)
      * augmentation → training-data diversity (4e2941b kept)

    max_bins is INPUT QUANTIZATION. HistGBM discretizes every feature
    into at most `max_bins` bins before learning; every tree split
    chooses between these pre-computed bin boundaries. Default 255
    gives sub-percentile granularity. Halving to 127 forces the GBM
    to see COARSER split thresholds on every feature simultaneously.
    Tight carve-out leaves that currently capture chord-cycle FPs at
    specific sub-bins (e.g., voiced_chroma_cosine_dist at bin 137 vs
    138 — ~0.4% of feature range) disappear when bins 137-138 merge.
    This is DIFFERENT from l2 (which shrinks leaf log-odds but keeps
    tight split locations) and DIFFERENT from min_samples_leaf (which
    requires larger leaf populations but still at fine bin resolutions).

    Mechanism on 3 surviving singing chord-cycle FPs: they persist
    marginally above GBM_THRESHOLD=0.982 on tight feature-space
    pockets. Bin coarsening merges adjacent sub-bins where FP-specific
    signatures live → splits that separated the 3 FPs from nearby
    not_splice training rows no longer exist at bin-127 granularity.
    The GBM has to rely on coarser, more generalizable thresholds.
    Marginal FPs drop below 0.982; confident TPs (p_splice ≥ 0.99)
    have much larger feature-value margin relative to bin width so
    survive coarsening. Speech english 0.889 / korean 0.667 carry
    multi-feature signatures with wide per-feature margins; halving
    bins barely shifts them.

    Why 127 specifically: sklearn HistGBM default 255. 127 = 2^7 - 1,
    clean half-step, standard progression {63, 127, 255}; allows
    bisection to 191 if too aggressive or 63 if unchanged. Halving
    mirrors the productive fcb8f4e min_samples_leaf 20→40 step (also
    2x) and 4b1575e l2 1.0→2.0 step (also 2x) — consistent
    regularization doubling on a complementary axis.

    Orthogonal. NOT any features.py addition (FEATURE_NAMES stable 80,
    features.py sha unchanged). NOT detector.py (no DSP gate / threshold
    / post-filter). NOT any prior classifier hyperparameter: max_bins
    is distinct from max_iter/max_depth/max_leaf_nodes (tree structure),
    learning_rate (per-tree contribution), l2_regularization (output
    shrinkage), min_samples_leaf (leaf size), sample_weight (loss
    gradient), augmentation (training data diversity). FIRST max_bins
    experiment in post-HistGBM classifier history; FIRST input-
    quantization regularization axis.

    Blast radius. 1 kwarg added in make_pipeline(). Training runtime
    slightly faster (fewer candidate split thresholds per node).
    Inference cost unchanged (same 300 trees). US-505b train_classifier.py
    sha gate auto-retrains from scratch.

(c) IF THIS FAILS. (1) Singing unchanged — 127 bins still fine enough
    to carve chord-cycle FP pockets → bisect down to 63. (2) Speech
    regresses — 127 over-coarsens speech TP discriminators (phoneme
    boundary features need fine resolution) → bisect up to 191 as
    safe middle between 127 and 255. (3) All domains move correlated
    (global coarsening wrong lever for current capacity) → pivot to
    early_stopping + validation_fraction (genuinely untouched
    classifier axis: built-in overfitting guard rather than direct
    regularization).

(d) Information gaps. (i) CLEAN_FP_POSITIONS JSON STILL absent after
    75+ iterations — cannot verify the 3 singing FPs sit at feature-
    space pockets where bin coarsening would specifically help vs
    generic high-p_splice regions where it wouldn't. (ii) SHAP rollup
    STILL empty for 14 keeps. (iii) Per-feature bin utilization (how
    many of 255 bins each feature's histogram currently populates)
    not surfaced — would directly tell whether halving 255→127 actually
    changes resolution on the features that drive FPs vs just
    features that are already coarse.

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain, file,
    t_sec, p_splice, gbm_predict_proba_vector, top-5 |SHAP|, top-5
    feature VALUES, feature-bin-index-at-FP for top-5 features).
    Would make every regularization choice data-driven instead of
    theory-driven.
    (2) SHAP ROLLUP REPAIR — empty for 14 keeps.
    (3) CLASSIFIER HYPERPARAMETER FRONTIER SNAPSHOT in CURRENT STATE
    (max_iter / max_depth / max_leaf_nodes / learning_rate /
    l2_regularization / min_samples_leaf / max_bins — tried kept/failed
    per axis alongside GBM_*). Would prevent losing state on
    regularization knobs the way GBM_THRESHOLD / GBM_MIN_SEP_S /
    ANALYSIS_STRIDE_S have explicit "tried kept/failed" tracking.

## 2026-04-21T10:01:57+09:00 — f6e05a7 (discard, combined=0.443084)
subject: enable HistGBM early_stopping=True validation_fraction=0.15 n_iter_no_change=15 tol=1e-4 (pure train_classifier.py hyperparameter change, no feature or detector edit) -- EXPLICIT CITED FALLBACK from 055285f(c)(3) after max_bins 127 discard: 'pivot to early_stopping + validation_fraction (genuinely untouched classifier axis: built-in overfitting guard rather than direct regularization)'. FIRST automatic-overfit-guard experiment in post-HistGBM classifier history. Since fcb8f4e min_samples_leaf 20->40 keep, 9 consecutive discards on 9 different axes (a23ab28 music-gate, 1cded37 cluster-count, e4a9c18 sample_weight, 7b49d40 time-stretch, ca2aa2f HPSS, 425f6d6 voicing-gated threshold, f1e91ec l2 3.0, 79317c7 lr 0.05, 055285f max_bins 127) -- every direct-regularization classifier-hyperparameter knob explored. Default HistGBM early_stopping='auto' disables when samples <10k (our ~2000 rows qualify) so currently GBM always trains all 300 iterations regardless of validation divergence. Forcing early_stopping=True with tol=1e-4 (default 1e-7 effectively never triggers) adds AUTOMATIC overfit guard picking iteration count data-driven rather than hard-capping at 300. Mechanism on 3 surviving singing chord-cycle FPs: GBM likely overfits to chord-cycle-specific feature-space pockets in later iterations (past ~150-200 boosting rounds), carving tight leaves separating 3 FPs from nearby not_splice training rows via bin-level splits. Validation loss on held-out 15%% split plateaus earlier than training loss (classic overfitting signature). Early stopping at n_iter_no_change=15 tol=1e-4 cuts training at knee ~150-220 iter BEFORE GBM carves overfit leaves. Marginal FPs at p_splice ~0.983 drop toward baseline GBM at shallower boosting depth. Confident TPs (p_splice >=0.99) decided by first ~50-100 trees where strongest discriminators compound -- val loss still improving there so those trees preserved. Fundamentally different from l2 (leaf log-odds shrinkage ALL leaves uniformly); min_samples_leaf (structural leaf-size floor globally); max_bins (input quantization globally); lr (per-tree contribution scalar uniformly). early_stopping TERMINATES training adaptively based on data-observed divergence rather than globally reshaping decision surface. Why 0.15/15/1e-4: validation_fraction=0.15 over default 0.1 because training set small (~2000 rows) -- 0.15 gives ~300-row val statistically meaningful but leaves ~1700 training; n_iter_no_change=15 slightly more permissive than default 10 to avoid premature stopping from noisy per-iter val loss; tol=1e-4 empirical regime (default 1e-7 effectively never triggers within 300 iter); three values jointly produce genuine operational early stopping not no-op. Orthogonal: NOT features.py (FEATURE_NAMES stable 80, features.py sha unchanged); NOT detector.py (no DSP gate/threshold/post-filter); NOT any prior hyperparameter tweak -- early_stopping/validation_fraction/n_iter_no_change/tol GENUINELY untouched knobs in classifier history. Blast radius: 4 new kwargs in HistGradientBoostingClassifier() constructor. Training runtime slightly faster on stopping (typically ~200 iter instead of 300). Inference cost lower (smaller forest if triggered) or unchanged (300 iter if not triggered). US-505b train_classifier.py sha gate auto-retrains from scratch. Smoke-verified: make_pipeline() constructs Pipeline with early_stopping=True validation_fraction=0.15 n_iter_no_change=15 tol=0.0001 confirmed via named_steps[clf] introspection, other hyperparameters stable (max_iter=300 max_depth=5 max_leaf_nodes=32 learning_rate=0.07 l2_regularization=2.0 min_samples_leaf=40), FEATURE_NAMES stable at 80.
per-domain: combined_english=0.732530 combined_korean=0.395833 combined_singing=0.300000

# 2026-04-21 — hypothesis: enable HistGBM early_stopping + validation_fraction

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` change — enable
    HistGradientBoostingClassifier built-in early stopping via
    `early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
    tol=1e-4`. All other hyperparameters stable (max_iter=300, max_depth=5,
    max_leaf_nodes=32, learning_rate=0.07, l2_regularization=2.0,
    min_samples_leaf=40). FEATURE_NAMES stable at 80. US-505b sha gate
    auto-retrains. EXPLICIT CITED FALLBACK from 055285f(c)(3) after
    max_bins 127 discard: "pivot to early_stopping + validation_fraction
    (genuinely untouched classifier axis: built-in overfitting guard
    rather than direct regularization)".

(b) WHY over recent failures. Since fcb8f4e min_samples_leaf 20→40 keep,
    9 consecutive discards on 9 different axes (a23ab28 music-gate,
    1cded37 cluster-count, e4a9c18 sample_weight, 7b49d40 time-stretch,
    ca2aa2f HPSS percussive MFCC, 425f6d6 voicing-gated threshold,
    f1e91ec l2 3.0, 79317c7 lr 0.05, 055285f max_bins 127). Every
    classifier hyperparameter knob explored (leaf-size-floor, leaf
    log-odds shrinkage, input quantization, per-tree contribution,
    tree structure capacity, ensemble size, loss-gradient bias,
    training-data diversity) — all either kept or discarded. The ONE
    genuinely untouched classifier-training axis is AUTO-REGULARIZATION
    VIA EARLY STOPPING. Default HistGBM behavior: `early_stopping='auto'`
    disables when samples <10k (our ~2000 rows qualify) so currently
    GBM always trains all 300 iterations regardless of validation
    divergence. Forcing early_stopping=True with tol=1e-4 (default
    1e-7 effectively never triggers) adds AUTOMATIC overfit guard
    that picks iteration count data-driven rather than hard-capping
    at 300.

    Mechanism on 3 surviving singing chord-cycle FPs: GBM likely
    overfits to chord-cycle-specific feature-space pockets in later
    iterations (past ~150-200 boosting rounds), carving tight leaves
    that separate 3 specific FPs from nearby not_splice training
    rows via bin-level splits. Validation loss on held-out 15% split
    plateaus earlier than training loss (classic overfitting
    signature). Early stopping at n_iter_no_change=15, tol=1e-4
    cuts training at the knee — ~150-220 iterations instead of 300 —
    BEFORE GBM carves these overfit leaves. Marginal FPs at p_splice
    ~0.983 drop toward baseline GBM predictions at shallower boosting
    depth. Confident TPs (p_splice ≥0.99) were decided by the first
    ~50-100 trees where the strongest discriminators compound;
    validation loss is still improving there so those trees are
    preserved.

    Fundamentally different from l2 (leaf log-odds shrinkage —
    shrinks ALL leaves uniformly); min_samples_leaf (structural
    leaf-size floor — disallows small leaves globally); max_bins
    (input quantization — coarsens splits globally); lr (per-tree
    contribution scalar — smooths every tree uniformly).
    early_stopping is the only axis that TERMINATES training
    adaptively based on data-observed divergence rather than globally
    reshaping the decision surface.

    Orthogonal. NOT any features.py addition (FEATURE_NAMES stable 80,
    features.py sha unchanged). NOT detector.py (no DSP gate/threshold/
    post-filter). NOT any prior hyperparameter tweak — early_stopping /
    validation_fraction / n_iter_no_change / tol are GENUINELY
    untouched knobs in classifier history. FIRST automatic-overfit-
    guard experiment in the post-HistGBM classifier history.

    Why 0.15 / 15 / 1e-4 specifically. validation_fraction=0.15 chosen
    over default 0.1 because our training set is small (~2000 rows) —
    0.15 gives ~300-row validation sample statistically meaningful
    but leaves ~1700 rows training. n_iter_no_change=15 slightly more
    permissive than default 10 to avoid premature stopping from noisy
    per-iter validation loss. tol=1e-4 is the empirical regime for
    GBM early stopping; default 1e-7 means effectively never triggers
    within 300 iterations. These three values jointly produce genuine
    operational early stopping, not a no-op. Room to bisect all three
    if the direction is right but magnitude wrong.

(c) IF THIS FAILS. (1) Singing unchanged — HistGBM's random
    train/val split leaks file groups between train+val so validation
    loss is over-optimistic and stopping doesn't trigger → lower tol
    to 1e-5 or raise n_iter_no_change to 25. (2) Speech regresses —
    0.15 val split removed critical speech training rows; early
    stopping at ~180 iterations drops confident multi-feature TPs →
    bisect validation_fraction down to 0.1 and keep n_iter_no_change=15.
    (3) All domains move correlated down — early stopping is globally
    wrong lever at this capacity → pivot to SCALER swap: replace
    StandardScaler with RobustScaler (quantile-based, resistant to
    outlier features, genuinely untouched preprocessing axis).

(d) Information gaps. (i) CLEAN_FP_POSITIONS JSON STILL absent after
    75+ iterations — cannot verify whether 3 singing FPs arise from
    overfit leaves carved in late boosting iterations vs stable-
    throughout-training signal. If FPs persist at p_splice=0.983
    even after 50 trees, early stopping won't help. (ii) SHAP rollup
    STILL empty for 14 keeps. (iii) Per-iteration OOF F1 or val loss
    curve across 300 boosting steps NOT surfaced — would directly
    tell whether validation loss plateaus around ~150-200 iter
    (early stopping bites) or keeps improving to 300 (no-op).

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain,
    file, t_sec, p_splice, gbm_predict_proba_vector,
    gbm_n_iter_effective, dsp_phase_z/t2_z/cpe_z, voicing_fraction,
    top-5 |SHAP|). Would directly settle whether FPs arise from
    late-iter overfitting vs early-iter mislearning and guide
    early_stopping tolerance.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) TRAINING LOSS CURVE SNAPSHOT in CURRENT STATE: per-iter
    train_loss + val_loss (if early_stopping on) OR train_loss +
    OOF F1 deltas across 50/100/200/300 iter checkpoints. Would
    make every subsequent classifier-hyperparameter / early-stopping
    tweak data-driven instead of theory-driven.

## 2026-04-21T10:15:22+09:00 — c7ff141 (discard, combined=0.485147)
subject: add HistGBM class_weight={0: 1.3, 1: 1.0, 2: 1.0} (pure train_classifier.py hyperparameter change, no feature or detector edit) -- FIRST class_weight experiment in classifier history; FIRST per-CLASS loss-gradient weighting. Since fcb8f4e keep (min_samples_leaf 20->40), 10 consecutive discards on 10 different axes (a23ab28 music-gate, 1cded37 cluster-count, e4a9c18 per-DATASET sample_weight catastrophic, 7b49d40 time-stretch, ca2aa2f HPSS, 425f6d6 voicing-gated threshold, f1e91ec l2 3.0, 79317c7 lr 0.05, 055285f max_bins 127, f6e05a7 early_stopping) -- every direct-regularization classifier-hyperparameter knob (l2/lr/max_bins/min_samples_leaf/early_stopping/max_depth/max_iter) AND every feature.py/detector.py axis explored. sklearn HistGradientBoostingClassifier class_weight parameter (added in sklearn 1.2) has NEVER been explored. DIFFERENT from e4a9c18 per-DATASET sample_weight (weighted rows by DOMAIN: singing 2x catastrophic) -- this weights rows by CLASS LABEL (not_splice 1.3x, hard_cut + crossfade 1.0x). class_weight is the ONE untouched loss-gradient axis that specifically targets the FP-vs-TP trade-off by class rather than by domain. Cited f6e05a7(c)(3) RobustScaler fallback is effectively no-op for HistGBM (bin-based, quantile-binning invariant to monotonic scaling transforms). Mechanism on 3 surviving singing chord-cycle FPs: upweight not_splice class 1.3x in loss -> GBM pushes harder to correctly classify negatives -> positive log-odds lowered globally. Marginal singing chord-cycle FPs at p_splice ~0.983 (log-odds ~4.06) lose ~20-30%% log-odds from class-loss weighting (1.3x weight compounds across 300 trees via class-imbalance adjusted loss gradient) -> drop below 0.982 threshold -> 3 singing FPs drop. Confident TPs (p_splice >0.99, log-odds >4.6) retain comfortable margin even after the class-loss penalty. Speech english 0.889 / korean 0.667 TPs sit at log-odds well above threshold with multi-feature tree support; 1.3x nudge on negatives minimally shifts them. Why 1.3 specifically: e4a9c18 used 2.0 per-dataset catastrophic, 1.3 is moderate conservative first step mirroring 4b1575e l2 1.0->2.0 (2x) and fcb8f4e min_samples_leaf 20->40 (2x) productive-keep magnitudes but smaller (class_weight acts more directly on loss gradient so smaller bump appropriate); bisection room to 1.2 if too aggressive or 1.5 if unchanged; class_weight='balanced' as automatic fallback. Orthogonal: NOT any features.py addition (FEATURE_NAMES stable 80, features.py sha unchanged); NOT detector.py (no DSP gate/threshold/post-filter); NOT e4a9c18 per-dataset sample_weight (different axis: by DOMAIN not by CLASS); NOT any l2/lr/max_bins/min_samples_leaf/max_depth/max_iter/early_stopping/augmentation. Blast radius: 1 new kwarg in make_pipeline(). Training runtime unchanged (class_weight is per-row loss scalar, negligible). Inference cost unchanged (same 300 trees). US-505b train_classifier.py sha gate auto-retrains from scratch. Smoke-verified: AST parse OK 462 lines, make_pipeline() constructs Pipeline with class_weight={0: 1.3, 1: 1.0, 2: 1.0} confirmed via named_steps[clf].class_weight, other hyperparameters stable (max_iter=300 max_depth=5 max_leaf_nodes=32 learning_rate=0.07 l2_regularization=2.0 min_samples_leaf=40), FEATURE_NAMES stable at 80, sklearn 1.7.2 accepts class_weight param end-to-end (fit + predict_proba working on synthetic 3-class X/y).
per-domain: combined_english=0.800000 combined_korean=0.473239 combined_singing=0.301613

# 2026-04-21 — hypothesis: HistGBM class_weight={0: 1.3, 1: 1.0, 2: 1.0}

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` change —
    add `class_weight={0: 1.3, 1: 1.0, 2: 1.0}` to HistGradientBoostingClassifier
    in make_pipeline(). Weights not_splice class 1.3x during loss computation,
    making GBM more conservative on positive predictions. All other
    hyperparameters stable. FEATURE_NAMES stable at 80. US-505b sha gate
    forces auto-retrain.

(b) WHY over recent failures. Since fcb8f4e min_samples_leaf 20→40 keep,
    10 consecutive discards on 10 different classifier/feature/detector
    axes. The sklearn HistGradientBoostingClassifier `class_weight`
    parameter (added in sklearn 1.2) has NEVER been explored — genuinely
    untouched axis. DIFFERENT from e4a9c18 per-DATASET sample_weight
    (which weighted rows by DOMAIN: singing 2x catastrophic) — this
    weights rows by CLASS label (not_splice 1.3x, hard_cut + crossfade
    1.0x). Cited f6e05a7(c)(3) RobustScaler fallback is effectively
    no-op for HistGBM (bin-based, scaling-invariant). Mechanism: loss
    gradient penalizes misclassifying negatives 1.3x more → GBM pushes
    harder to correctly classify negatives → positive log-odds lowered
    globally. Marginal singing chord-cycle FPs at p_splice ~0.983
    (log-odds ~4.06) lose ~20-30% log-odds from class-loss weighting
    → drop below 0.982 threshold. Confident TPs (p_splice >0.99,
    log-odds >4.6) retain margin. Factor 1.3 is conservative (e4a9c18
    used 2.0 catastrophic; 1.3 is moderate, room to bisect to 1.2 or
    1.5). Orthogonal to: every hyperparameter knob (l2, lr, max_bins,
    min_samples_leaf, early_stopping, max_depth, max_iter), every
    features.py experiment, every detector.py change, e4a9c18 per-dataset
    weighting. FIRST class_weight experiment; FIRST per-CLASS loss-gradient
    weighting.

(c) IF THIS FAILS. (1) Korean regresses (0.667→0.5): 1.3 too aggressive
    on the recall-limited domain → bisect to 1.2. (2) Singing unchanged
    (chord-cycle FPs at p_splice >0.990 where class-weight bump doesn't
    bite) → bisect up to 1.5 or try class_weight='balanced' which sklearn
    computes automatically from class frequencies. (3) All domains drop
    correlated (class_weight is globally wrong lever for current GBM
    calibration) → pivot to monotonic_cst on known-direction features
    (dsp_phase_z, dsp_t2_z, dsp_cpe_z monotonic-increasing on p_splice).

(d) Information gaps. (i) CLEAN_FP_POSITIONS JSON STILL absent after
    75+ iterations — cannot verify the 3 singing FPs sit at p_splice
    in [0.982, 0.990] band where class_weight bump would bite. If at
    0.995+, this doesn't help. (ii) SHAP rollup STILL empty for 14
    keeps. (iii) Exact class-count distribution from training_manifest.json
    not surfaced in CURRENT STATE — approximating not_splice:positive
    as ~60:40 from file-type arithmetic but don't know exact effective
    imbalance post-augmentation (pitch-shift adds singing clean negatives).

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON. (2) SHAP rollup
    repair. (3) CLASS-COUNT SNAPSHOT from training_manifest.json:
    not_splice/hard_cut/crossfade counts per domain post-augmentation,
    in CURRENT STATE. Would make class_weight and sample_weight
    hypotheses data-driven instead of theory-driven.

## 2026-04-21T10:29:28+09:00 — 4f94f3e (discard, combined=0.459090)
subject: add HistGBM interaction_cst='pairwise' (pure train_classifier.py hyperparameter change, no feature or detector edit) -- FIRST interaction_cst experiment in classifier history; FIRST feature-interaction-combinatorial regularization axis. Since fcb8f4e min_samples_leaf 20->40 keep, 11 consecutive discards on 11 different axes (a23ab28 music-gate, 1cded37 cluster-count, e4a9c18 per-DATASET sample_weight, 7b49d40 time-stretch, ca2aa2f HPSS, 425f6d6 voicing-gated threshold, f1e91ec l2 3.0, 79317c7 lr 0.05, 055285f max_bins 127, f6e05a7 early_stopping, c7ff141 class_weight) -- every MAGNITUDE-tuning classifier knob exhausted (l2/lr/max_bins/min_samples_leaf/max_iter/max_depth/max_leaf_nodes/class_weight/sample_weight/early_stopping). interaction_cst is STRUCTURAL-combinatorial rather than magnitude: constrains WHICH feature combinations trees may learn, not HOW MUCH they weight them. 'pairwise' restricts every tree to at most 2 features (GAM-like 2-way decomposition). Cited as natural multiclass replacement for c7ff141(c)(3) monotonic_cst fallback (sklearn raises ValueError for monotonic_cst on multiclass; interaction_cst supports multiclass -- smoke-verified fits 3-class synthetic data returns shape (200,3)). Mechanism on 3 surviving singing chord-cycle FPs: current max_depth=5/max_leaf_nodes=32 lets trees carve 5-feature conjunctions like 'voicing=0.4 AND chord_dist=0.1 AND mfcc_delta=0.2 AND tempo=110 AND onset=0.3 -> high p_splice' memorizing specific FP positions (unfindable as single 2-feature signal per 60+ failed features.py attempts). 'pairwise' makes such high-order conjunctions impossible -> FP-specific leaves disappear -> marginal FP p_splice ~0.983 drops below 0.982. Real TPs have SIMPLE 2-feature strong signals (DSP phase+t2+cpe pairs, voiced_unvoiced_mfcc_asymmetry + spec_contrast) -- 300 ensembled pairwise-trees compound these via law of large numbers, TPs retain strength. Speech english 0.889 / korean 0.667 multi-feature signatures decomposable to 2-feature ensembles retain clearance. Why 'pairwise' not 'no_interactions': no_interactions=decision stumps (1-feature) too aggressive dropping TP compounding; pairwise is conservative first step mirroring 4b1575e l2 1.0->2.0 and fcb8f4e min_samples_leaf 20->40 productive-first-step magnitudes. Bisection: 'no_interactions' if unchanged (stronger constraint); revert to None if regresses. Orthogonal: NOT any prior magnitude knob (l2/lr/max_bins/min_samples_leaf/max_depth/max_leaf_nodes/max_iter/class_weight/sample_weight/early_stopping -- all MAGNITUDE or STOPPING criteria); interaction_cst is COMBINATORIAL (restricts tree-topology space). NOT any features.py addition (FEATURE_NAMES stable 80, features.py sha unchanged); NOT detector.py (no DSP gate/threshold/post-filter); NOT augmentation axis (4e2941b/7b49d40). Blast radius: 1 new kwarg in make_pipeline(). Training runtime slightly faster (pairwise search cheaper per split than unrestricted). Inference cost unchanged (same 300 trees). US-505b train_classifier.py sha gate auto-retrains from scratch. Smoke-verified: AST parse OK 461 lines, make_pipeline() constructs Pipeline with interaction_cst='pairwise' confirmed via named_steps[clf].interaction_cst, other hyperparameters stable (max_iter=300 max_depth=5 max_leaf_nodes=32 learning_rate=0.07 l2_regularization=2.0 min_samples_leaf=40), FEATURE_NAMES stable at 80, sklearn accepts interaction_cst='pairwise' end-to-end (fit + predict_proba on synthetic 3-class X/y returns shape (300,3)).
per-domain: combined_english=0.814815 combined_korean=0.475000 combined_singing=0.250000

# 2026-04-21 — hypothesis: HistGBM interaction_cst='pairwise'

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` change —
    add `interaction_cst='pairwise'` kwarg to HistGradientBoostingClassifier
    in `make_pipeline()`. Restricts every tree to use at most 2 features
    per tree (GAM-like 2-way interaction decomposition). All other
    hyperparameters stable (max_iter=300, max_depth=5, max_leaf_nodes=32,
    learning_rate=0.07, l2_regularization=2.0, min_samples_leaf=40).
    FEATURE_NAMES stable at 80. US-505b sha gate auto-retrains.

(b) WHY over recent failures. Since fcb8f4e min_samples_leaf 20→40 keep,
    11 consecutive discards on 11 different axes: a23ab28 music-gate,
    1cded37 cluster-count, e4a9c18 sample_weight, 7b49d40 time-stretch,
    ca2aa2f HPSS, 425f6d6 voicing-gated threshold, f1e91ec l2 3.0,
    79317c7 lr 0.05, 055285f max_bins 127, f6e05a7 early_stopping,
    c7ff141 class_weight. Every MAGNITUDE-tuning classifier knob explored
    (l2/lr/max_bins/min_samples_leaf/max_iter/max_depth/max_leaf_nodes/
    class_weight/sample_weight/early_stopping). interaction_cst is
    STRUCTURAL rather than magnitude — it constrains WHICH feature
    combinations trees may learn, not HOW MUCH they weight them. Trees
    with max_depth=5 + max_leaf_nodes=32 can currently carve 5-feature
    conjunctions like "voicing=0.4 AND chord_dist=0.1 AND mfcc_delta=0.2
    AND tempo=110 AND onset=0.3 → high p_splice" — exactly the kind of
    memorization capacity that produces 3 specific singing chord-cycle
    FP carve-outs sitting marginally above 0.982 across 60+ iterations.
    Cited as explicit fallback c7ff141(c)(3): "pivot to monotonic_cst
    on known-direction features" — monotonic_cst is NOT multiclass-
    supported in sklearn (raises ValueError for classes>2);
    interaction_cst is its natural replacement, same spirit (structural
    constraint) different mechanism, multiclass-supported (smoke-verified).

    interaction_cst='pairwise' forces GAM-like 2-way decomposition.
    Mechanism on 3 singing chord-cycle FPs: the high-order feature
    conjunction memorizing FP positions (unfindable as single 2-feature
    signal — proven across 60+ failed features.py additions targeting
    single-mechanism FP separators) CANNOT be expressed. GBM must use
    2-feature trees that generalize across chord-cycle + non-chord-cycle
    boundaries similarly. FP-specific leaves disappear → marginal FP
    p_splice drops below 0.982.

    Real TPs have SIMPLE 2-feature strong signals: high DSP across
    phase+t2+cpe (2-feature trees on dsp pairs) + high MFCC delta +
    high chroma distance. 300 ensembled pairwise-trees compound
    these signals (law of large numbers) → TP predictions retain
    strength via many weak learners. Speech TPs similarly carry
    2-feature conjunctions (voiced_unvoiced_mfcc_asymmetry biggest
    keep is a single-feature-derived scalar already; combined with
    spec_contrast it's a 2-feature signature).

    Why 'pairwise' (not 'no_interactions'): 'no_interactions' allows
    only 1-feature trees (decision stumps) — too aggressive, drops
    TP signature compounding across feature pairs. 'pairwise' is the
    conservative first step on the structural axis (analogous to
    4b1575e l2 1.0→2.0 conservative first step, fcb8f4e min_samples_leaf
    20→40 first step). Bisection: 'no_interactions' if unchanged
    (stronger constraint); revert to None if regresses.

    Orthogonal. NOT any magnitude-tuning knob (l2/lr/max_bins/
    min_samples_leaf/max_iter/max_depth/max_leaf_nodes/class_weight/
    sample_weight/early_stopping) — all were MAGNITUDE or STOPPING
    criteria. interaction_cst is COMBINATORIAL: it restricts the
    space of learnable tree topologies. NOT any features.py change
    (FEATURE_NAMES stable 80, sha unchanged). NOT any detector.py
    change (no DSP gate / threshold / post-filter). NOT any
    augmentation axis. FIRST interaction_cst experiment in classifier
    history; FIRST feature-interaction-combinatorial axis.

    Blast radius. 1 new kwarg in make_pipeline(). Training runtime
    slightly faster ('pairwise' search is cheaper per split than
    unrestricted search). Inference cost unchanged (same 300 trees).
    US-505b train_classifier.py sha gate auto-retrains from scratch.
    Smoke-verified in-prompt: HistGBM with interaction_cst='pairwise'
    fits on 200-row 3-class synthetic data and predict_proba returns
    shape (200, 3) correctly.

(c) IF THIS FAILS. (1) Singing unchanged — 2-feature conjunctions
    also capture chord-cycle FPs because the 3 FPs are expressible
    as 2-feature patterns → pivot to 'no_interactions' (decision
    stumps, 1-feature constraint). (2) Speech regresses — TP signal
    requires 3+ feature ensembling GBM can't compound via 2-way
    only → bisect by passing explicit list-of-sets interaction_cst
    allowing DSP-triple (phase+t2+cpe) as one group plus everything
    else as another (partial structural constraint). (3) All
    domains drop correlated — interaction_cst wrong direction for
    current tree depth → pivot to warm-start isotonic calibration
    via CalibratedClassifierCV(method='isotonic', cv=5) wrapping
    the HistGBM (genuinely untouched post-hoc output-recalibration
    axis).

(d) Information gaps. (i) CLEAN_FP_POSITIONS JSON STILL absent
    after 75+ iterations — cannot verify if 3 chord-cycle FPs
    correspond to 5-feature carve-outs vs 2-feature boundaries.
    If they're 2-feature patterns, interaction_cst='pairwise'
    won't bite them. (ii) SHAP rollup STILL empty for 14 keeps —
    cannot verify which features' interactions drive the decisions.
    (iii) Per-iteration train/OOF loss curve + feature-per-tree
    distribution (how many features each of the 300 trees actually
    splits on) NOT surfaced — would directly tell whether GBM
    under current config is using 3+ feature trees (interaction_cst
    would bite) or mostly 1-2 feature trees (interaction_cst
    wouldn't bite).

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain,
    file, t_sec, p_splice, gbm_predict_proba_vector,
    dsp_phase_z/t2_z/cpe_z, voicing_fraction, top-5 |SHAP|).
    Would make every structural-constraint choice data-driven.
    (2) SHAP ROLLUP REPAIR — empty for 14 keeps.
    (3) TREE-STRUCTURE DIAGNOSTICS SNAPSHOT in CURRENT STATE:
    per-trained-GBM-run, distribution of (a) actual tree depths,
    (b) number of unique features used per tree, (c) which 5-10
    features dominate all splits. Would make interaction_cst /
    max_depth / max_leaf_nodes decisions data-driven by knowing
    whether GBM actually uses deep/wide trees or whether it's
    already naturally restricted.

## 2026-04-21T10:43:26+09:00 — 3061196 (discard, combined=0.491287)
subject: raise HistGBM min_samples_leaf 40 -> 80 (linear continuation of fcb8f4e keep 20->40)
per-domain: combined_english=0.800000 combined_korean=0.422535 combined_singing=0.350794

# 2026-04-21 — hypothesis: raise HistGBM min_samples_leaf 40 → 80

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` change — raise
    `HistGradientBoostingClassifier.min_samples_leaf` from 40 to 80 in
    `make_pipeline()`. All other hyperparameters stable (max_iter=300,
    max_depth=5, max_leaf_nodes=32, learning_rate=0.07,
    l2_regularization=2.0). FEATURE_NAMES stable at 80. US-505b sha gate
    auto-retrains.

(b) WHY over recent failures. LINEAR CONTINUATION of the proven productive
    fcb8f4e keep (min_samples_leaf 20→40, 2x doubling) — the same doubling
    magnitude that 4b1575e l2 1.0→2.0 also landed as a keep. Since fcb8f4e,
    12 consecutive discards on 12 different axes: every GENUINELY untouched
    classifier hyperparameter knob (sample_weight e4a9c18, l2 3.0 f1e91ec,
    lr 0.05 79317c7, max_bins 127 055285f, early_stopping f6e05a7,
    class_weight c7ff141, interaction_cst 4f94f3e), augmentation axes
    (time_stretch 7b49d40), features.py (HPSS ca2aa2f), detector
    post-filters (cluster-count 1cded37, voicing-gated threshold 425f6d6),
    and features (music-gate a23ab28). The structural-pivot well is dry;
    exploit the PROVEN axis at the next natural step.

    Why min_samples_leaf specifically (not another doubling of l2): l2
    2.0→3.0 was the nearest linear continuation already tried and DISCARDED
    (f1e91ec, 1.5x not 2x). min_samples_leaf at 40 has NOT had a follow-up
    doubling attempt. The 2x doubling that worked at fcb8f4e (20→40) is
    the natural continuation step; 40→60 would mirror the failed f1e91ec
    1.5x on l2, so 80 is the right magnitude.

    Mechanism on 3 surviving singing chord-cycle FPs. min_samples_leaf is
    a LEAF-GEOMETRY structural regularizer: each leaf must contain ≥N
    training rows before GBM is allowed to terminate a split there. At 40
    (2% of ~2000 rows), GBM can still carve leaves populated by ~40 rows
    from a specific chord-cycle feature-space pocket where 3 FPs + ~37
    nearby not_splice training rows land; GBM then assigns that leaf a
    small-positive log-odds sufficient to push the 3 FPs marginally above
    p_splice=0.982. At 80 (4%), such leaves MUST be MERGED with adjacent
    feature-space regions (dominated by clearly-negative not_splice rows)
    → the merged leaf's log-odds pulled down toward negative → the 3 FPs'
    p_splice drops below 0.982 → FPs drop. Confident TPs (p_splice ≥0.99)
    sit in leaves dominated by positive rows where merging with adjacent
    negative rows still leaves a positive majority; log-odds barely
    moves. Speech english 0.889 / korean 0.667 TPs have multi-feature
    signatures at well-populated feature-space regions (leaves already
    ≥80 rows); merging has NO effect.

    DIFFERENT from l2 (output log-odds shrinkage uniform across all
    leaves regardless of population — bites confident TPs and marginal
    FPs alike); DIFFERENT from max_bins (input quantization — merges
    adjacent bins at same density); DIFFERENT from lr (per-tree
    contribution scalar uniform). min_samples_leaf bites ONLY
    small-population leaves, which is precisely where chord-cycle
    FP-specific carve-outs live.

    Orthogonal. NOT 4b1575e/f1e91ec l2 (output shrinkage); NOT 79317c7
    lr (per-tree contribution); NOT 055285f max_bins (input
    quantization); NOT d1be6c3 max_iter/max_depth/max_leaf_nodes
    (tree structure capacity); NOT f6e05a7 early_stopping (adaptive
    termination); NOT c7ff141 class_weight (per-class loss gradient);
    NOT 4f94f3e interaction_cst (feature-combinatorial); NOT e4a9c18
    sample_weight (per-domain loss gradient); NOT any features.py /
    detector.py / augmentation axis. SAME axis as fcb8f4e keep but at
    the next natural value — linear continuation, not new axis.

    Blast radius. 1 integer literal in `make_pipeline()`. Training
    runtime marginally faster (fewer candidate splits survive population
    threshold). Inference cost unchanged. US-505b sha gate auto-retrains.

(c) IF THIS FAILS. (1) Singing unchanged — chord-cycle FP leaves already
    had ≥80 rows at 40→80 threshold, so merging doesn't happen → the
    structural axis for this mechanism is exhausted; pivot to ISOTONIC
    calibration via `CalibratedClassifierCV(method='isotonic', cv=5)`
    wrapping the pipeline (cited fallback from 4f94f3e(c)(3), genuinely
    untouched output-recalibration axis). (2) Speech regresses — 80
    starves GBM of leaves for phoneme-specific signatures on speech TPs
    → bisect to 60 (halfway). (3) All domains drop correlated —
    min_samples_leaf over-restricts tree expressiveness at current
    max_iter=300 → pair with capacity bump max_iter 300→500 at next
    iteration instead of narrowing min_samples_leaf further.

(d) Information gaps. (i) CLEAN_FP_POSITIONS JSON STILL absent after 75+
    iterations — cannot verify the 3 singing FPs sit in small-population
    leaves where min_samples_leaf doubling would bite vs well-populated
    leaves where it wouldn't. (ii) SHAP rollup STILL empty for 14 keeps.
    (iii) LEAF-POPULATION HISTOGRAM at current min_samples_leaf=40 NOT
    surfaced — would directly show whether 40 is near-saturating (most
    leaves already >80 rows, doubling is no-op) vs binding (many leaves
    at exactly 40-50 rows, doubling bites). Information asymmetry forces
    theory-driven choice.

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain, file,
    t_sec, p_splice, gbm_predict_proba_vector, gbm_leaf_population,
    top-5 |SHAP|, top-5 feature VALUES). Would make every regularization
    choice (especially min_samples_leaf / l2 / max_bins bisection)
    data-driven rather than theory-driven.
    (2) SHAP ROLLUP REPAIR — empty for 14 keeps.
    (3) CLASSIFIER HYPERPARAMETER FRONTIER SNAPSHOT in CURRENT STATE
    (max_iter / max_depth / max_leaf_nodes / learning_rate /
    l2_regularization / min_samples_leaf / max_bins / early_stopping /
    class_weight / interaction_cst — tried kept/failed per axis alongside
    GBM_* primary tunable frontier). Would prevent the exhaustion-
    tracking blind spot across 12 recent classifier discards on 12
    different axes.

## 2026-04-21T10:58:38+09:00 — 5de4e19 (discard, combined=0.455854)
subject: wrap HistGBM in BaggingClassifier(n_estimators=5, bootstrap=True)
per-domain: combined_english=0.804878 combined_korean=0.375000 combined_singing=0.313846

# 2026-04-21 — hypothesis: wrap HistGBM in BaggingClassifier(n_estimators=5, bootstrap=True)

(a) Replace the single HistGradientBoostingClassifier in make_pipeline()
    with BaggingClassifier(estimator=HistGBM(...), n_estimators=5,
    bootstrap=True, random_state=RANDOM_STATE, n_jobs=1). Base-GBM
    hyperparameters stable; FEATURE_NAMES stable 80; US-505b sha gate
    auto-retrains. Ensemble averages predict_proba across 5 HistGBMs
    each trained on a bootstrap sample of the ~2000 training rows.

(b) 13 consecutive discards since fcb8f4e keep cover every knob INSIDE a
    single HistGBM (l2/lr/max_bins/min_samples_leaf/max_iter/max_depth/
    max_leaf/interaction_cst/class_weight/sample_weight/early_stopping),
    plus features.py, augmentation, and detector post-filters. Bagging
    operates ABOVE single-classifier tuning — first ensemble-wrapping
    experiment in classifier history. The 3 singing chord-cycle FPs at
    p_splice ≈ 0.983 are MARGINAL: GBM trees that carve the specific
    chord-cycle feature-space pocket exist in some bootstraps but not
    others. Averaging across 5 sub-GBMs pulls marginal predictions
    toward ensemble mean (well below 0.982) → FPs drop. Confident TPs
    (p≥0.99 via many-feature agreement) are bootstrap-stable → average
    preserves them. Chosen over CalibratedClassifierCV(isotonic) because
    calibration is monotonic and cannot re-rank: if a marginal FP
    outranks a borderline TP under single GBM, isotonic preserves that
    order; bagging's bootstrap diversity CAN re-rank. Calibration
    remains in (c)(3) as fallback.

(c) If singing unchanged — FPs have low bootstrap variance (all 5
    sub-GBMs agree) → n_estimators 5→10 AND max_samples=0.5 to force
    more diversity. If speech regresses — bootstrap starves phoneme-
    specific leaves → pair bagging with GBM_THRESHOLD 0.982→0.975 on
    detector side (primary tunable, no retrain). If all domains drop
    correlated → revert and pivot to CalibratedClassifierCV(method=
    'isotonic', cv=5) wrapping the Pipeline (genuinely untouched
    post-hoc output recalibration).

(d) Same three gaps as last 75+ iterations: CLEAN_FP_POSITIONS still
    absent so cannot verify FPs sit in variance-heavy marginal pockets
    where bagging bites; SHAP rollup still empty for 14 keeps; per-seed
    predict_proba variance at FP positions not surfaced so bagging
    decision is theory-driven.

(e) NEW: ENSEMBLE-FRONTIER SNAPSHOT — n_estimators / max_samples /
    bootstrap_features tried kept/failed per axis in CURRENT STATE,
    alongside CLASSIFIER HYPERPARAMETER FRONTIER and primary-tunable
    frontier. Also unchanged asks: CLEAN_FP_POSITIONS JSON and SHAP
    rollup repair. Would make ensemble tuning data-driven; right now
    I am picking n_estimators=5 from canonical bagging magnitudes
    without any per-position variance evidence.

## 2026-04-21T11:14:02+09:00 — 4fe3877 (discard, combined=0.077430)
subject: wrap HistGBM in CalibratedClassifierCV(sigmoid, cv=5, ensemble=False)
per-domain: combined_english=0.255319 combined_korean=0.181818 combined_singing=0.000000

# 2026-04-21 — hypothesis: wrap HistGBM in CalibratedClassifierCV(sigmoid, cv=5, ensemble=False)

(a) Pure `splice/classifier/train_classifier.py` change — wrap the existing
    HistGradientBoostingClassifier in `CalibratedClassifierCV(estimator=HistGBM(...),
    method='sigmoid', cv=5, ensemble=False)`. Base-HistGBM hyperparameters
    stable (max_iter=300, max_depth=5, max_leaf_nodes=32, learning_rate=0.07,
    l2_regularization=2.0, min_samples_leaf=40). FEATURE_NAMES stable at 80.
    US-505b train_classifier.py sha gate auto-retrains. Post-hoc Platt
    (sigmoid) calibration learns a 2-parameter monotonic remapping of
    predict_proba output trained on OOF predictions via internal 5-fold CV.

(b) WHY over 5de4e19 BaggingClassifier discard and 14 prior classifier-knob
    failures. `ensemble=False` is the key lever: instead of 5de4e19's ensemble
    averaging across bootstrap-sampled GBMs (which pulled ALL probabilities
    toward the mean and hurt TPs), `ensemble=False` uses `cross_val_predict`
    to get OOF preds, fits ONE sigmoid calibrator on those, then applies
    that calibrator to ONE final HistGBM trained on all data. Inference =
    single GBM + single sigmoid remap → no ensemble variance reduction
    (which was 5de4e19's failure mode). Mechanism on 3 singing chord-cycle
    FPs: HistGBM's log-odds → sigmoid mapping is known to be over-confident
    on boosted classifiers with small calibration sets, especially in the
    high-probability tail. Marginal FPs at p_splice ~0.983 are exactly
    where sigmoid over-confidence bites hardest (logit ~4.06, close to
    model's edge-of-training-distribution). Platt calibration fits
    a * logit(p) + b, shrinking probabilities toward the empirical
    TP-rate at that predicted level. If the 3 chord-cycle FPs are
    mislabeled as confident positives but the empirical TP-rate at
    p_splice=0.983 is actually 0.85-0.90 (because similar chord-cycle
    patterns exist in OOF negatives), calibration maps 0.983 → ~0.88,
    dropping them below 0.982 threshold. Confident TPs at p_splice ≥0.99
    (logit ≥4.6) live in the tail where the sigmoid fit is flat, so
    they stay well above 0.982 after calibration. Genuinely orthogonal
    axis: NOT any prior HistGBM hyperparameter (magnitude/structural/
    loss-gradient knob). Platt sigmoid chosen over isotonic from 5de4e19
    cited fallback because: (i) 2-parameter fit is less noisy on ~2000
    rows than isotonic's step-function; (ii) isotonic tends to produce
    plateau regions that kill granular discrimination; (iii) sigmoid
    naturally reshapes the over-confidence mode of boosted models.
    Inner cv=5 uses default KFold, not GroupKFold — this LEAKS file-IDs
    between inner calibration train/test sets, which is a known risk
    but acceptable: the OUTER GroupKFold in train() still gives honest
    cross-validated OOF metrics, and the calibration quality only
    matters for inference-time output reshaping.

(c) IF THIS FAILS. (1) Singing unchanged — 3 chord-cycle FPs sit at
    p_splice ≥0.990 where sigmoid is flat → pivot to `method='isotonic',
    cv=5, ensemble=False` (isotonic allows non-monotonic tail-squashing
    that sigmoid cannot express). (2) Speech regresses — sigmoid
    fit shifts english/korean TPs down to 0.982 threshold → the calibrator
    was trained on leaked OOF and overcorrected → pivot to
    `cv=GroupKFold(5)` via an iterable split passed to `cv=` (ensures
    honest OOF preds for calibrator). (3) All domains drop correlated —
    post-hoc calibration is globally wrong lever for this problem →
    pivot to RandomForestClassifier swap (replace HistGBM with
    RandomForest(n_estimators=500, max_features='sqrt', max_depth=None);
    fundamentally different algorithm — bagging of deep unpruned
    random-feature-subset trees vs sequential boosting).

(d) Information gaps. (i) CLEAN_FP_POSITIONS JSON STILL absent after 75+
    iterations — cannot verify the 3 singing FPs sit in the [0.982, 0.990]
    miscalibration band where sigmoid Platt bites. If they're at p ≥0.998,
    sigmoid is flat and won't help. (ii) SHAP rollup STILL empty for 14
    keeps. (iii) Current calibration curve (reliability diagram of HistGBM:
    predicted p vs empirical TP-rate at p-bin) NOT surfaced in CURRENT
    STATE — would directly tell whether HistGBM needs calibration at all
    (if already well-calibrated, sigmoid fit is near-identity; if
    over-confident in [0.95, 1.0] tail, calibration bites). Forces
    theory-driven choice on a knob whose impact depends entirely on
    miscalibration magnitude.

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain, file,
    t_sec, p_splice, gbm_predict_proba_vector, dsp_phase_z/t2_z/cpe_z,
    voicing_fraction, top-5 |SHAP|).
    (2) SHAP ROLLUP REPAIR — empty for 14 keeps.
    (3) NEW: CALIBRATION SNAPSHOT in CURRENT STATE — reliability diagram
    of HistGBM on OOF predictions (predicted-p vs empirical-TP-rate at
    10 equal-frequency bins per class). Would make every calibration
    / threshold decision data-driven; right now it's impossible to tell
    whether sigmoid/isotonic are even the right lever.

## 2026-04-26T02:39:31+09:00 — babdaa6 (keep, combined=0.529698)
subject: GBM_MIN_SEP_S 1.25 -> 1.1 (continue bisection within [1.0, 1.25] bracket; trend monotonic) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from current best keep 4f92bdf(c)(2): 'GBM_THRESHOLD lowering remains strongest pivot if 1.25 saturates or regresses' -- 1.25 did NOT saturate (1.5 -> 0.501, 1.25 -> 0.521 is +0.020 gain on the proven productive axis, still pulling toward verify-failed-but-real 1.0 -> 0.537 data point). Bisection completion within [1.0, 1.25] bracket BEFORE pivoting to fresh axis is the explicit cited rule. Frontier marks 1.0 as 'failed' so exact retry blocked; 1.1 is fresh, sits between two known-real points (1.0 -> 0.537 real, 1.25 -> 0.521 real), still 4.4x the 0.25s collar -- comfortable margin against single boundary attracting two emits inside +-0.25s. Linear interpolation between (1.0, 0.537) and (1.25, 0.521) predicts ~0.531 at 1.1 -- modest but high-confidence step. Why 1.1 (not 1.125 exact midpoint, not 1.05): cleaner decimal; biased slightly toward productive end of bracket while staying fresh on frontier; if 1.1 lands in 0.527-0.535 band confirms monotonic continuation and next bisection is [1.0, 1.1] (e.g. 1.05); if regresses, bracket inverts to 1.2/1.15. Mechanism on Korean recall (P=0.855 R=0.293, recall-bottlenecked): greedy dedupe at 1.25s suppresses any adjacent emission within 1.25s of highest-prob hit. Korean voice-switch boundaries spaced ~1.0-1.25s apart still partially lose second TP under 1.25s rule. Cutting to 1.1s lets boundary pairs >=1.1s both surface; pairs spaced 1.1-1.25s gain both TPs while 4.4x-collar margin keeps single-boundary spurious-neighbor suppression intact. Why not pivot to GBM_THRESHOLD 0.982 -> 0.97 yet: cited rule defers pivot until current axis saturates or regresses, neither has happened. GBM_THRESHOLD axis only has the catastrophic 0.5 contract-bug trial in frontier so it remains pocketed for next iter if 1.1 saturates. Why not retry 1.0 exact: frontier guard against repeat hypotheses blocks exact retry; would need retest sentinel workflow which is operator-owned. Smoke-verified: GBM_MIN_SEP_S=1.1 confirmed via import; other tunables stable (GBM_THRESHOLD=0.982 ANALYSIS_STRIDE_S=0.12 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: GBM_MIN_SEP_S 1.25 → 1.1 (continue bisection within [1.0, 1.25])

(a) Pure `splice/detector.py` change — drop GBM_MIN_SEP_S from 1.25 to 1.1.
    No retrain, no feature change, classifier byte-identical. Other primary
    tunables stable (GBM_THRESHOLD=0.982, ANALYSIS_STRIDE_S=0.12,
    DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0). FEATURE_NAMES stable at 80.

(b) WHY over recent failures. Cited next-step from current best keep
    4f92bdf(c)(2): "GBM_THRESHOLD lowering remains strongest pivot if 1.25
    saturates or regresses." 1.25 did NOT saturate — 1.5 → 0.501,
    1.25 → 0.521 is a +0.020 gain on the proven productive axis, still
    pulling toward the verify-failed-but-real 1.0 → 0.537 data point.
    Bisection completion within the [1.0, 1.25] bracket BEFORE pivoting
    to a fresh axis is the explicit rule from the cited keep. Frontier
    marks 1.0 as 'failed' so exact retry is blocked by the wrapper guard;
    1.1 is fresh, sits between two known-real points (1.0 → 0.537 real,
    1.25 → 0.521 real), still 4.4x the 0.25s collar — comfortable margin
    against a single boundary attracting two emits inside ±0.25s. Linear
    interpolation between (1.0, 0.537) and (1.25, 0.521) predicts ~0.531
    at 1.1 — a modest but high-confidence step on a proven axis. Why
    1.1 (not 1.125 exact midpoint, not 1.05): cleaner decimal; biased
    slightly toward the productive end of the bracket while staying
    fresh on the frontier; if 1.1 lands in the 0.527-0.535 band it
    confirms monotonic continuation and the next bisection is [1.0, 1.1]
    (e.g. 1.05); if it regresses, the bracket inverts and the next step
    is 1.2 / 1.15. Mechanism on Korean recall (P=0.855 / R=0.293,
    recall-bottlenecked): greedy dedupe at 1.25s suppresses any adjacent
    emission within 1.25s of highest-prob hit. Korean voice-switch
    boundaries spaced ~1.0-1.25s apart still partially lose the second
    TP under the 1.25s rule. Cutting to 1.1s lets boundary pairs ≥1.1s
    both surface; pairs spaced 1.1-1.25s gain both TPs while 4.4x-collar
    margin keeps single-boundary spurious-neighbor suppression intact.

(c) IF THIS FAILS. (1) Combined regresses below 0.521 — bracket inverts;
    bisect upward to 1.2 or 1.15 next iter. (2) Combined saturates near
    0.521 — GBM_MIN_SEP_S axis exhausted within [1.0, 1.25]; pivot to
    GBM_THRESHOLD descent (cited strongest pivot — recall is the
    dominant bottleneck). First step 0.982 → 0.97 (fresh near current;
    only other GBM_THRESHOLD trial 0.5 was catastrophic 0.289 with
    contract bug). (3) Combined matches or exceeds 0.537 — direction
    confirmed across three real points; next iter try 1.05 (continue
    descent within now-validated band) OR retest the wrapper-blocked
    1.0 hypothesis via the retest sentinel workflow.

(d) Information gaps. Per-domain combined breakdown still NOT in
    baseline_metrics.json on this branch — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable — baseline_metrics.json has
    not captured it yet)". Cannot tell which of singing/korean/english
    is dragging the GM floor on korean-iter1, only see aggregate.
    cross_voice_f1=0 and same_voice_edit_f1=0 in baseline plus
    unknown_label_count=241 suggest the contract fix may still leak
    labels — cannot verify without per-domain visibility.

(e) Wrapper enhancements. (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON
    on the iter1 branch — currently captured for non-iter1 baselines
    but missing here, forcing every hypothesis to fly blind on per-domain
    decomposition. (2) GBM_MIN_SEP_S BISECTION HISTORY TABLE in CURRENT
    STATE — frontier shows the 'tried' set as a flat list {1.0, 1.25,
    1.5, 2.0} with kept/failed labels but doesn't show combined-per-value,
    so bisection direction has to be reconstructed from RESEARCH NOTES
    every iteration. A 4-row sorted table (value, combined,
    kept/failed/verify-fail-real) would make the next-step choice
    instant. (3) WRAPPER VERIFY-FAIL "REAL VS GUARDED" DISTINCTION in
    frontier — 1.0 → 0.537 real but verify-failed for non-metric
    reasons (per 6f6abbf preflight path bug) is currently treated as
    'failed' identically to a true regression, blocking the natural
    exact-retry. A 'verify_fail_reason' tag (corpus_purged |
    metric_regression | runtime_crash | preflight_path_bug) would let
    the frontier distinguish 'this value is bad' from 'this value was
    never actually evaluated'.
[auto] (no SHAP data for either 4f92bdf or babdaa6)

## 2026-04-26T03:16:07+09:00 — b6cbe35 (keep, combined=0.537055)
subject: GBM_MIN_SEP_S 1.1 -> 1.05 (continue bisection within now-validated [1.0, 1.1] bracket; both endpoints real, monotonic descent confirmed across 5 points) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from current best keep babdaa6(c)(3): 'next iter try 1.05 (continue descent within now-validated band)'. GBM_MIN_SEP_S axis now has 5 monotonic real data points: 2.0 -> 0.481, 1.5 -> 0.501, 1.25 -> 0.521, 1.1 -> 0.530, 1.0 -> 0.537 (real per 6f6abbf preflight bug; verify-failed only because legacy data/eval.tar.gz.enc had no korean_iter1 subdir). Each step toward 1.0 nets +0.008-0.020 monotonically. babdaa6 just landed +0.008 going 1.25 -> 1.1, productive but not saturating. Linear interpolation between (1.0, 0.537) and (1.1, 0.530): predicts ~0.533 at 1.05 -- modest high-confidence step. 1.05 is fresh on frontier (frontier has tried {1.0 failed-but-real, 1.1 kept, 1.25 kept, 1.5 kept, 2.0 kept}); 1.05 sits BETWEEN two known-real points; 4.2x the 0.25s collar -- still comfortable margin against single boundary attracting two emits inside +-0.25s. Why 1.05 not 1.025: cleaner decimal; 1.025 too fine without first confirming 1.05 lands in expected band; bisection step magnitude consistent with prior 1.25 -> 1.1 step (-0.15 vs -0.05). Why not pivot to GBM_THRESHOLD 0.982 -> 0.97 yet: cited rule defers pivot until current axis saturates or regresses, neither has happened. GBM_THRESHOLD axis only has the catastrophic 0.5 contract-bug trial in frontier so it remains pocketed for next iter if 1.05 saturates. Why not retry 1.0 exact: frontier guard against repeat hypotheses blocks exact retry; would need operator-owned retest sentinel workflow. Mechanism on Korean recall (P=0.855 R=0.293, recall-bottlenecked): greedy dedupe at 1.1s still suppresses any adjacent emission within 1.1s of highest-prob hit. Korean voice-switch boundaries spaced 1.05-1.10s apart still partially lose second TP under 1.1s rule. Cutting to 1.05s lets boundary pairs >=1.05s both surface; pairs spaced 1.05-1.10s gain both TPs while 4.2x-collar margin keeps single-boundary spurious-neighbor suppression intact. Bisection-completion-within-bracket-before-pivoting-to-fresh-axis is the explicit cited rule from prior keeps. Smoke-verified: GBM_MIN_SEP_S=1.05 confirmed via import; other tunables stable (GBM_THRESHOLD=0.982 ANALYSIS_STRIDE_S=0.12 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

(a) HYPOTHESIS. Drop GBM_MIN_SEP_S from 1.1 → 1.05 (pure splice/detector.py
    change, no retrain, no feature change). Cited next-step from current keep
    babdaa6(c)(3): "next iter try 1.05 (continue descent within now-validated
    band)". 1.05 is fresh on the frontier (frontier has tried 1.0/1.1/1.25/
    1.5/2.0). Other tunables stable: GBM_THRESHOLD=0.982, ANALYSIS_STRIDE_S=0.12.

(b) WHY. GBM_MIN_SEP_S axis now has 5 data points showing strict monotonic
    descent: 2.0→0.481, 1.5→0.501, 1.25→0.521, 1.1→0.530, 1.0→0.537 (real
    per 6f6abbf preflight bug). Each step toward 1.0 nets +0.008-0.020.
    babdaa6 just landed +0.008 going 1.25→1.1 — productive but not saturating.
    Linear interpolation between (1.0, 0.537) and (1.1, 0.530): predicts
    ~0.533 at 1.05 — high-confidence step. 1.05 sits BETWEEN two known-real
    points (1.0 real, 1.1 real); 4.2x the 0.25s collar — still comfortable
    margin against single-boundary spurious-neighbor inflation. Not pivoting
    to GBM_THRESHOLD 0.982→0.97 yet because cited rule defers pivot until
    current axis saturates or regresses; neither has happened. Not retrying
    1.0 exact — frontier guard blocks repeat hypothesis; would need
    operator-owned retest sentinel workflow. Bisection completion within the
    [1.0, 1.1] bracket (now both endpoints real) before pivoting to fresh
    axis is the explicit cited rule. Mechanism on Korean recall (P=0.855
    R=0.293, recall-bottlenecked): greedy dedupe at 1.1s still suppresses
    boundary pairs spaced 1.05-1.10s apart (Korean voice-switch boundaries
    span this band). Cutting to 1.05s lets pairs ≥1.05s both surface;
    pairs spaced 1.05-1.10s gain both TPs while 4.2x-collar margin keeps
    single-boundary spurious suppression intact.

(c) IF THIS FAILS. (1) Combined regresses below 0.530 — bracket inverts
    within [1.05, 1.1]; bisect upward to 1.075 next, OR pivot directly to
    GBM_THRESHOLD 0.982→0.97 (cited strongest pivot — recall is dominant
    bottleneck). (2) Combined saturates near 0.530 — GBM_MIN_SEP_S axis
    exhausted within productive band; pivot to GBM_THRESHOLD descent
    0.982→0.97 (genuinely fresh near current; only other GBM_THRESHOLD
    trial 0.5 was catastrophic 0.289 with contract bug, no clean trial
    near 0.97-0.95 band). (3) Combined matches or exceeds 0.537 — direction
    confirmed even more strongly across 6 real points; next iter try 1.025
    OR retest the wrapper-blocked 1.0 hypothesis via retest sentinel
    workflow (operator-owned). The operator could run --retest from this
    SHA to recover the 1.0 baseline if 1.05 lands at 0.534+.

(d) Information gaps. Per-domain combined breakdown still NOT in
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable — baseline_metrics.json has not
    captured it yet)". I've been tuning toward Korean recall by mechanism
    but cannot directly verify singing/english are not silently regressing.
    cross_voice_f1=0 / same_voice_edit_f1=0 / unknown_label_count=241 in
    initial baseline (pre-iter1) suggested label-encoding leak; contract
    fix in 4f92bdf+ should have resolved it but per-class F1 visibility
    still missing in the prompt. The frontier text shows GBM_MIN_SEP_S
    'tried' set as a flat list without paired combined values — bisection
    direction has to be reconstructed from RESEARCH NOTES every iter.

(e) Wrapper enhancements. (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON
    on iter1 branch — currently captured for non-iter1 baselines but
    missing here, forcing every hypothesis to fly blind on per-domain
    decomposition. (2) GBM_MIN_SEP_S BISECTION HISTORY TABLE in CURRENT
    STATE — frontier shows 'tried' as flat list {1.0/1.1/1.25/1.5/2.0}
    with kept/failed labels but no combined-per-value, so bisection
    direction has to be reconstructed every iteration. A 5-row sorted
    table (value, combined, kept/failed/verify-fail-real) would make the
    next-step choice instant. (3) WRAPPER VERIFY-FAIL "REAL VS GUARDED"
    DISTINCTION in frontier — 1.0→0.537 real but verify-failed for
    non-metric reasons (preflight path bug) is currently treated as
    'failed' identically to a true regression, blocking the natural
    exact-retry. A 'verify_fail_reason' tag (corpus_purged |
    metric_regression | runtime_crash | preflight_path_bug) would let the
    frontier distinguish "this value is bad" from "this value was never
    actually evaluated".
[auto] (no SHAP data for either babdaa6 or b6cbe35)

## 2026-04-26T04:26:23+09:00 — e357941 (discard, combined=0.537055)
subject: GBM_MIN_SEP_S 1.05 -> 1.025 (bisection within saturated [1.0, 1.05] bracket; 1.05 -> 0.537 matched verify-failed-but-real 1.0 -> 0.537, definitive saturation test) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from current best keep b6cbe35(c)(3): 'next iter try 1.025 OR retest the wrapper-blocked 1.0 hypothesis via retest sentinel workflow'. b6cbe35 just landed 1.05 at 0.537 -- EXACT match with verify-failed-but-real 1.0 -> 0.537 data point (real per 6f6abbf preflight bug; verify-failed only because legacy data/eval.tar.gz.enc had no korean_iter1 subdir). GBM_MIN_SEP_S axis now has 6 monotonic-then-plateau real data points: 2.0 -> 0.481, 1.5 -> 0.501, 1.25 -> 0.521, 1.1 -> 0.530, 1.05 -> 0.537, 1.0 -> 0.537. Productive descent from 2.0 to 1.05 plateaued at 0.537 within [1.0, 1.05] band -- strong saturation signal. 1.025 is bisection midpoint and definitively settles saturation: if 1.025 -> 0.537 or below, full saturation confirmed and next iter cleanly pivots to GBM_THRESHOLD 0.982 -> 0.975 (cited strongest pivot, recall is dominant bottleneck per P=0.855 R=0.293); if 1.025 -> 0.538+, descent continues marginally and next bisection is [1.0, 1.025]. Either outcome high-information. 1.025 is fresh on frontier (frontier set: {1.0 failed-but-real, 1.05 kept, 1.1 kept, 1.25 kept, 1.5 kept, 2.0 kept}); sits between two known-real points (1.0 -> 0.537, 1.05 -> 0.537); 4.1x the 0.25s collar -- still comfortable margin against single boundary attracting two emits inside +-0.25s; satisfies docstring 'Must exceed evaluate.py 1.0s tolerance' rule (1.025 > 1.0). Why 1.025 not 1.0 retry: frontier guard against repeat hypotheses blocks exact retry; 1.0 recovery requires operator-owned retest sentinel workflow. Why not pivot directly to GBM_THRESHOLD now: bisection-completion-within-bracket-before-pivoting-to-fresh-axis is the explicit cited rule from 5 consecutive keeps (b6cbe35, babdaa6, 720bd9c, 665d50e, original 1.0 try); one more data point gives definitive evidence to pivot with high confidence rather than abandoning a productive axis prematurely. Mechanism on Korean recall (P=0.855 R=0.293, recall-bottlenecked): greedy dedupe at 1.05s still suppresses any adjacent emission within 1.05s of highest-prob hit. Korean voice-switch boundaries occasionally span 1.025-1.05s band; cutting to 1.025s lets pairs >=1.025s both surface. If this band is empty (no GT boundary pairs spaced 1.025-1.05s apart in Korean corpus), result will saturate at 0.537 -- clean signal that the recall floor is boundary-pair-spacing-bounded and next lever is GBM_THRESHOLD descent. Smoke-verified: GBM_MIN_SEP_S=1.025 confirmed via import; other tunables stable (GBM_THRESHOLD=0.982 ANALYSIS_STRIDE_S=0.12 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: GBM_MIN_SEP_S 1.05 → 1.025 (bisection within saturated [1.0, 1.05] bracket)

(a) HYPOTHESIS. Pure `splice/detector.py` change — drop GBM_MIN_SEP_S
    from 1.05 to 1.025. No retrain, no feature change, classifier
    byte-identical. Other primary tunables stable: GBM_THRESHOLD=0.982,
    ANALYSIS_STRIDE_S=0.12, DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0.
    FEATURE_NAMES stable at 80. 1.025 is fresh on the frontier
    (frontier set: {1.0 failed-but-real, 1.05 kept, 1.1 kept, 1.25 kept,
    1.5 kept, 2.0 kept}).

(b) WHY over recent failures. EXPLICIT CITED NEXT-STEP from current best
    keep b6cbe35(c)(3): "Combined matches or exceeds 0.537 — direction
    confirmed even more strongly across 6 real points; next iter try
    1.025 OR retest the wrapper-blocked 1.0 hypothesis via retest
    sentinel workflow." Bisection-completion-within-bracket-before-
    pivoting-to-fresh-axis is the explicit cited rule from 5 consecutive
    keeps (b6cbe35, babdaa6, 720bd9c, 665d50e, original 1.0 try).
    GBM_MIN_SEP_S axis now has 6 real data points: 2.0→0.481, 1.5→0.501,
    1.25→0.521, 1.1→0.530, 1.05→0.537, 1.0→0.537 (real per 6f6abbf
    preflight bug fix). The exact match of 1.05→0.537 with the
    verify-failed-but-real 1.0→0.537 is STRONG saturation signal — the
    productive descent from 2.0 down to 1.05 has plateaued at 0.537
    within the [1.0, 1.05] band. 1.025 is the bisection midpoint and
    will definitively settle saturation: if 1.025 → 0.537 or below,
    saturation is fully confirmed and next iter cleanly pivots to
    GBM_THRESHOLD 0.982 → 0.975 (recall is the bottleneck per P=0.855
    R=0.293, threshold drop is the cited strongest pivot). If 1.025
    → 0.538+, the descent continues marginally and the next bisection
    is [1.0, 1.025]. Either outcome is informative. 1.025 sits between
    two known-real points (1.0 → 0.537, 1.05 → 0.537), at 4.1x the
    0.25s collar — still comfortable margin against single-boundary
    spurious-neighbor inflation, and still satisfies docstring's
    "Must exceed evaluate.py's 1.0s tolerance" rule (1.025 > 1.0).
    Why 1.025 not 1.0 retry: frontier guard against repeat hypotheses
    blocks exact retry; 1.0 recovery requires operator-owned retest
    sentinel workflow. Why not pivot directly to GBM_THRESHOLD now:
    the bisection rule says complete the active bracket first; one
    more data point (1.025) gives definitive evidence to pivot with
    high confidence rather than abandoning a productive axis prematurely.
    Mechanism on Korean recall: greedy dedupe at 1.05s still suppresses
    boundary pairs spaced 1.025-1.05s apart. Korean voice-switch
    boundaries occasionally span this band — cutting to 1.025s lets
    pairs ≥1.025s both surface. If this band is already empty (no GT
    boundary pairs spaced 1.025-1.05s apart in Korean corpus), result
    will saturate at 0.537 — clean signal that the recall floor is
    boundary-pair-spacing-bounded and the next lever is GBM_THRESHOLD.

(c) IF THIS FAILS. (1) Combined regresses below 0.537 — GBM_MIN_SEP_S
    saturation confirmed and 1.025 is just-below-noise; pivot
    immediately to GBM_THRESHOLD 0.982 → 0.975 (cited strongest pivot,
    recall bottleneck, fresh value between current 0.982 and historical
    0.97-kept). (2) Combined matches 0.537 exactly — full saturation
    confirmation; same pivot to GBM_THRESHOLD 0.982 → 0.975 next iter
    with rock-solid evidence that GBM_MIN_SEP_S band is exhausted in
    the [1.0, 1.05] productive range. (3) Combined exceeds 0.538 —
    descent continues marginally; next iter try 1.0125 or pivot to
    operator-owned retest sentinel for the wrapper-blocked 1.0 exact
    point.

(d) Information gaps. Per-domain combined breakdown STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable — baseline_metrics.json has
    not captured it yet)". Cannot verify singing/english are not
    silently regressing while I tune toward Korean recall. Frontier
    text shows GBM_MIN_SEP_S 'tried' as a flat list without paired
    combined values per attempt — bisection direction has to be
    reconstructed from RESEARCH NOTES every iteration. The 1.0 →
    0.537 verify-failed-but-real point is shown as 'failed' identical
    to a true regression, blocking the natural exact-retry without an
    operator workflow.

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch
    — currently captured for non-iter1 baselines but missing here,
    forcing every hypothesis to fly blind on per-domain decomposition.
    Even if just `combined_korean / combined_english / combined_singing`
    in three keys would unlock per-axis targeting.
    (2) GBM_MIN_SEP_S BISECTION HISTORY TABLE in CURRENT STATE — a
    6-row sorted table (value, combined, kept/failed/verify-fail-real)
    would make the saturation vs descent judgment instant rather than
    requiring me to reconstruct the trajectory from RESEARCH NOTES every
    iteration. Generalize to any frequently-bisected primary tunable.
    (3) WRAPPER VERIFY-FAIL "REAL VS GUARDED" DISTINCTION in frontier
    — a `verify_fail_reason` tag (corpus_purged | metric_regression |
    runtime_crash | preflight_path_bug) would let the frontier
    distinguish "this value is bad" from "this value was never actually
    evaluated", unlocking exact-retry of preflight-bugged points
    without manual operator retest sentinel workflow.

## 2026-04-26T05:37:11+09:00 — fa8f7b5 (keep, combined=0.544061)
subject: GBM_THRESHOLD 0.982 -> 0.978 (pivot to cited fresh axis after GBM_MIN_SEP_S saturated at 0.537 across 3 values 1.0/1.025/1.05) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from just-discarded e357941(c)(1+2): 'GBM_MIN_SEP_S saturation confirmed... pivot immediately to GBM_THRESHOLD 0.982 -> 0.975 (cited strongest pivot, recall bottleneck per P=0.855 R=0.293)'. Cited literal 0.975 BLOCKED by frontier (failed on legacy classifier); 0.978 is fresh first step in cited direction (-0.004 vs cited -0.007), strictly between current 0.982 and frontier-failed 0.975. GBM_MIN_SEP_S axis confirmed saturated: 7 real data points 2.0->0.481, 1.5->0.501, 1.25->0.521, 1.1->0.530, 1.05->0.537, 1.025->0.537 (just-discarded TIE), 1.0->0.537 (verify-failed-but-real per 6f6abbf preflight bug). Three values plateau at 0.537 -- bisection-completion-within-bracket-before-pivoting rule satisfied; axis exhausted in productive band. Frontier set on GBM_THRESHOLD: kept 0.97 (legacy pre-contract-fix classifier), failed 0.975 (legacy), failed 0.5 (contract-bug catastrophic 0.289 on iter1), current 0.982. 0.978 is genuinely fresh on this axis. Why 0.978 not 0.97 (legacy kept): legacy 0.97 was kept on pre-contract-fix classifier where p_splice formula was buggy (1 - P(cross_voice) gated wrong posterior); on current contract-fixed classifier (p_splice = P(cross_voice) + P(same_voice_edit) = 1 - P(no_splice) per resolved no_splice column), posterior distribution shifted, so legacy thresholds not directly transferable AND wrapper guard would block 0.97 as repeat. 0.978 is smallest meaningful fresh step probing whether corrected posterior responds to threshold descent without committing to larger drop a priori. Mechanism on Korean recall (P=0.855 R=0.293, dominant bottleneck): GBM_THRESHOLD gates which p_splice peaks emit at all (vs GBM_MIN_SEP_S which only dedupes already-emitted neighbors). Marginal TPs whose corrected p_splice lands in [0.978, 0.982) currently get suppressed and never reach dedupe; 0.978 opens that band -- if any Korean voice-switch TPs cluster in that band, they surface. Fresh axis chosen over ANALYSIS_STRIDE_S 0.12 -> 0.06 because: (i) cited explicit pivot is GBM_THRESHOLD; (ii) STRIDE descent is denser scan with same gate so won't add new emits unless probability surface has unhit local maxima between 0.12s strides -- unlikely given 0.25s collar already comfortably wider than 0.12s grid. Why not retest 1.0 GBM_MIN_SEP_S via sentinel: that's operator-owned workflow per CLAUDE.md retest sentinel doc, not in autoresearch loop scope. Why not bigger jump (0.97, 0.96): smallest fresh step preserves bisection optionality -- if 0.978 produces 0.541+, descent confirmed and next iter can step bigger to 0.974 or 0.97 via retest; if regresses, corrected posterior is sharper and we bisect upward to 0.98 immediately without large rollback. Smoke-verified: GBM_THRESHOLD=0.978 confirmed via import; other tunables stable (GBM_MIN_SEP_S=1.05 ANALYSIS_STRIDE_S=0.12 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: GBM_THRESHOLD 0.982 → 0.978 (pivot to cited fresh axis after GBM_MIN_SEP_S saturation)

(a) HYPOTHESIS. Pure `splice/detector.py` change — drop GBM_THRESHOLD from
    0.982 to 0.978. No retrain, no feature change, classifier byte-identical.
    Other primary tunables stable (GBM_MIN_SEP_S=1.05, ANALYSIS_STRIDE_S=0.12,
    DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0). FEATURE_NAMES stable at 80.
    0.978 is fresh on the frontier (frontier set: kept 0.97 [legacy], failed
    0.5 [contract-bug], failed 0.975 [legacy], current 0.982).

(b) WHY. EXPLICIT CITED PIVOT from e357941(c)(1+2): "GBM_MIN_SEP_S
    saturation confirmed... pivot immediately to GBM_THRESHOLD 0.982 → 0.975
    (cited strongest pivot, recall bottleneck)". GBM_MIN_SEP_S axis now has 7
    real data points and CONFIRMED saturation: 2.0→0.481, 1.5→0.501,
    1.25→0.521, 1.1→0.530, 1.05→0.537, 1.025→0.537 (just-discarded TIE),
    1.0→0.537 (verify-failed-but-real). Three values (1.0, 1.025, 1.05) all
    plateau at 0.537 — saturation is rock-solid. Bisection-completion-within-
    bracket-before-pivoting rule is now satisfied; the axis is exhausted in
    the productive band. Cited literal value 0.975 is BLOCKED by frontier
    (legacy classifier failed it); 0.978 is fresh, between current 0.982 and
    failed 0.975, modest first step in cited direction (-0.004 vs cited
    -0.007). Mechanism on Korean recall (P=0.855 R=0.293, dominant
    bottleneck): GBM_THRESHOLD gates which p_splice peaks emit at all (vs
    GBM_MIN_SEP_S which only dedupes already-emitted neighbors). Marginal TPs
    whose corrected p_splice = P(cross_voice) + P(same_voice_edit) lands in
    [0.978, 0.982) currently get suppressed and never reach dedupe; 0.978
    opens that band. Why 0.978 not 0.97 (kept on legacy): legacy 0.97 is from
    pre-contract-fix classifier where p_splice formula was buggy (1 -
    P(cross_voice) gated wrong posterior); on the corrected classifier the
    posterior distribution shifted, so legacy thresholds aren't directly
    transferable AND wrapper guard would block 0.97 as repeat. 0.978 is the
    smallest meaningful fresh step that probes whether the corrected
    posterior responds to threshold descent without committing to the larger
    drop a priori. If 0.978 lands at 0.541+ → continue descent (0.974, 0.97
    via retest sentinel, 0.96). If at 0.530-0.540 → axis is sensitive and
    saturating, pivot to ANALYSIS_STRIDE_S 0.12 → 0.06 (genuinely untouched,
    only kept 0.06 on legacy). If regresses → corrected posterior is sharper
    than legacy and the safe band is narrower, bisect upward to 0.98.

(c) IF THIS FAILS. (1) Combined regresses below 0.537 — corrected posterior
    is sharper than legacy, threshold band is narrower, bisect upward to
    0.98 next iter. (2) Combined matches 0.537 exactly — 0.978 produced no
    new emits (probability cluster gap), step too small; jump to 0.97 via
    operator-owned retest sentinel workflow OR pivot to ANALYSIS_STRIDE_S
    0.12 → 0.06 (genuinely untouched on iter1; legacy kept 0.06 on different
    classifier). (3) Combined exceeds 0.537 marginally — descent confirmed
    on corrected classifier; next iter try 0.974 (one micro-step below
    failed legacy 0.975 — fresh, tests whether the legacy non-monotonic
    surface still applies post-contract-fix).

(d) Information gaps. Per-domain combined breakdown STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable)". Cannot tell whether singing /
    english are silently regressing while I tune toward Korean recall. The
    GBM_THRESHOLD frontier text shows kept/failed values but doesn't tag
    them with classifier-context (pre/post-contract-fix), so legacy 0.97
    kept and legacy 0.975 failed cannot be distinguished from current-
    classifier values. The probability histogram of GBM outputs on the
    corrected classifier is also not surfaced — would tell whether 0.978 vs
    0.975 vs 0.97 are even producing different emits or hitting the same
    discrete probability buckets.

(e) Wrapper enhancements. (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON
    on iter1 branch — same ask as last 5 keeps, still missing. (2) FRONTIER
    CLASSIFIER-CONTEXT TAG: each frontier entry should carry a tag (e.g.,
    `classifier_sha=<8>`) so legacy 0.97-kept on a pre-contract-fix
    classifier can be distinguished from a same-value retry on the current
    classifier — would unlock legitimate retries when the underlying model
    has changed AND prevent wrapper guard from blocking informative
    experiments. (3) GBM PROBABILITY HISTOGRAM SNAPSHOT in CURRENT STATE:
    on the latest trained classifier, count of p_splice values in 10
    equal-width buckets across [0.9, 1.0]. Would make every threshold
    descent decision data-driven by showing whether 0.978 vs 0.975 vs 0.97
    are even gating different posterior tails.
[auto] (no SHAP data for either b6cbe35 or fa8f7b5)

## 2026-04-26T07:37:20+09:00 — 8093406 (keep, combined=0.551855)
subject: GBM_THRESHOLD 0.978 -> 0.974 (continue cited descent on contract-fixed classifier; one micro-step below failed legacy 0.975 to test whether legacy non-monotonicity survives contract fix) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from current best keep fa8f7b5(c)(3): 'descent confirmed on corrected classifier; next iter try 0.974 (one micro-step below failed legacy 0.975 -- fresh, tests whether the legacy non-monotonic surface still applies post-contract-fix)'. fa8f7b5 just landed +0.007 going 0.982 -> 0.978 on the contract-fixed classifier (0.537 -> 0.544), confirming threshold descent is productive on the corrected posterior. Linear interpolation predicts ~0.551 at 0.974 if same +0.007/-0.004 slope holds. 0.974 is fresh on frontier (set: failed 0.5 contract-bug catastrophic, failed legacy 0.975, kept legacy 0.97, kept legacy 0.975, kept current 0.978). Why 0.974 not 0.97 (legacy kept): legacy 0.97 was kept on pre-contract-fix classifier where p_splice formula was buggy (1 - P(cross_voice) gated wrong posterior); on current contract-fixed classifier (p_splice = P(cross_voice) + P(same_voice_edit) per resolved no_splice column), posterior distribution shifted, legacy thresholds not directly transferable AND wrapper guard would block 0.97 as repeat. 0.974 is smooth continuation (-0.004 step matching the proven productive 0.982 -> 0.978 step) sitting exactly one micro-step below the legacy-failed 0.975 -- directly tests whether legacy non-monotonic 0.975 dip survives the contract fix. If 0.974 lands at 0.548+, legacy non-monotonicity is gone and descent is monotonic on corrected classifier; next iter can step bigger to 0.97 via operator-owned retest sentinel. If 0.974 saturates or regresses, legacy non-monotonicity preserved (suggests intrinsic posterior cluster gap), pivot to ANALYSIS_STRIDE_S 0.12 -> 0.06 (untouched primary tunable on iter1, only kept value 0.06 from legacy). Mechanism on Korean recall (P=0.855 R=0.293, dominant bottleneck): GBM_THRESHOLD gates which p_splice peaks emit at all (vs GBM_MIN_SEP_S which only dedupes already-emitted neighbors). Marginal Korean voice-switch TPs whose corrected p_splice lands in [0.974, 0.978) currently get suppressed and never reach dedupe; 0.974 opens that band. If any Korean TPs cluster in that 4-thousandth-wide band, they surface; if not, corrected posterior is sharper than legacy and safe descent band is narrower than -0.004 per step. Why not pivot to ANALYSIS_STRIDE_S now: bisection-completion-within-bracket-before-pivoting rule from 7 prior keeps says continue productive axis until saturation/regression. GBM_THRESHOLD just produced +0.007 gain; not saturated. Why smallest fresh step (0.974 not 0.97 via retest): preserves bisection optionality -- if 0.974 produces 0.548+ descent confirmed and operator can step bigger via retest; if regresses, sharp posterior bisects to 0.976 immediately without large rollback. Why not bigger drop (0.97 directly): would need operator-owned retest sentinel workflow (frontier guard blocks repeat), out of autoresearch loop scope. Smoke-verified: GBM_THRESHOLD=0.974 confirmed via import; other tunables stable (GBM_MIN_SEP_S=1.05 ANALYSIS_STRIDE_S=0.12 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: GBM_THRESHOLD 0.978 → 0.974 (continue cited descent on contract-fixed classifier)

(a) HYPOTHESIS. Pure `splice/detector.py` change — drop GBM_THRESHOLD from
    0.978 to 0.974. No retrain, no feature change, classifier byte-identical.
    Other primary tunables stable: GBM_MIN_SEP_S=1.05, ANALYSIS_STRIDE_S=0.12,
    DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0. FEATURE_NAMES stable at 80.
    0.974 is fresh on the frontier (frontier set: failed 0.5 contract-bug,
    failed legacy 0.975, kept legacy 0.97, kept legacy 0.975, kept current
    0.978).

(b) WHY over recent failures. EXPLICIT CITED NEXT-STEP from current best keep
    fa8f7b5(c)(3): "Combined exceeds 0.537 marginally — descent confirmed on
    corrected classifier; next iter try 0.974 (one micro-step below failed
    legacy 0.975 — fresh, tests whether the legacy non-monotonic surface
    still applies post-contract-fix)." fa8f7b5 just landed +0.007 going
    0.982 → 0.978 on the contract-fixed classifier (0.537 → 0.544),
    confirming threshold descent is productive on the corrected posterior.
    Linear interpolation predicts ~0.551 at 0.974 if the same +0.007/-0.004
    slope holds. Why 0.974 not 0.97 (legacy kept): legacy 0.97 was kept on
    pre-contract-fix classifier where p_splice formula was buggy, posterior
    distribution since shifted under p_splice = P(cross_voice) +
    P(same_voice_edit), so legacy values aren't directly transferable; AND
    wrapper guard would block 0.97 as repeat. 0.974 is a smooth continuation
    (-0.004 step matching the proven productive 0.982 → 0.978 step) that
    sits exactly one micro-step below the legacy-failed 0.975 — directly
    tests whether the legacy non-monotonic 0.975 dip survives the contract
    fix. If 0.974 lands at 0.548+, the legacy non-monotonicity is gone and
    descent is monotonic on the corrected classifier; next iter can step
    bigger to 0.97 via operator-owned retest sentinel. If 0.974 saturates
    or regresses, the legacy non-monotonicity is preserved (suggests an
    intrinsic posterior cluster gap), and pivot to ANALYSIS_STRIDE_S
    0.12 → 0.06 (untouched primary tunable on iter1, only kept value 0.06
    is from legacy). Mechanism on Korean recall (P=0.855 R=0.293, dominant
    bottleneck): GBM_THRESHOLD gates which p_splice peaks emit at all.
    Marginal Korean voice-switch TPs whose corrected p_splice lands in
    [0.974, 0.978) currently get suppressed; 0.974 opens that band. If any
    Korean TPs cluster in that 4-thousandth-wide band, they surface; if
    not, the corrected posterior is sharper than the legacy and the safe
    descent band is narrower than -0.004 per step. Why not pivot to
    ANALYSIS_STRIDE_S now: bisection-completion-within-bracket-before-
    pivoting rule from 7 prior keeps says continue the productive axis
    until saturation/regression. GBM_THRESHOLD just produced a +0.007 gain;
    not saturated. Why smallest fresh step (0.974 not 0.97 via retest):
    preserves bisection optionality — if 0.974 produces 0.548+ descent
    confirmed and operator can step bigger via retest; if regresses, sharp
    posterior bisects to 0.976 immediately without large rollback. Why not
    bigger drop (0.97 directly): would need operator-owned retest sentinel
    workflow (frontier guard blocks repeat), out of autoresearch loop scope.

(c) IF THIS FAILS. (1) Combined regresses below 0.544 — corrected posterior
    has cluster gap in [0.974, 0.978), threshold descent saturated within
    that band; pivot to ANALYSIS_STRIDE_S 0.12 → 0.06 (genuinely untouched
    primary tunable on iter1). (2) Combined matches 0.544 exactly — descent
    is saturated at 0.544 across [0.974, 0.978]; same pivot to STRIDE
    descent next iter, OR retest the wrapper-blocked 0.97 hypothesis via
    operator-owned retest sentinel. (3) Combined exceeds 0.544 marginally
    — descent confirmed on corrected classifier across two values; next
    iter try 0.97 via retest sentinel (operator-owned) OR continue with
    0.972 via wrapper.

(d) Information gaps. Per-domain combined breakdown STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable — baseline_metrics.json has not
    captured it yet)". Cannot verify singing/english are not silently
    regressing while I tune toward Korean recall. The GBM_THRESHOLD
    frontier text shows kept/failed values but doesn't tag them with
    classifier-context (pre/post-contract-fix), so legacy 0.975 failed and
    legacy 0.97 kept cannot be distinguished from current-classifier
    values. The probability histogram of GBM outputs on the corrected
    classifier is also not surfaced — would directly tell whether 0.974 vs
    0.978 are even gating different posterior tails or hitting the same
    discrete probability buckets.

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch —
    same ask as last 6 keeps, still missing.
    (2) FRONTIER CLASSIFIER-CONTEXT TAG: each frontier entry should carry
    a tag (e.g., `classifier_sha=<8>`) so legacy 0.97-kept on a
    pre-contract-fix classifier can be distinguished from a same-value
    retry on the current classifier — would unlock legitimate retries
    when the underlying model has changed.
    (3) GBM PROBABILITY HISTOGRAM SNAPSHOT in CURRENT STATE: on the
    latest trained classifier, count of p_splice values in 10
    equal-width buckets across [0.9, 1.0]. Would make every threshold
    descent decision data-driven.
[auto] (no SHAP data for either fa8f7b5 or 8093406)

## 2026-04-26T08:11:00+09:00 — 83c31eb (keep, combined=0.554606)
subject: GBM_THRESHOLD 0.974 -> 0.972 (continue cited descent on contract-fixed classifier; wrapper-friendly micro-step since 0.97 is operator-owned retest territory) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from current best keep 8093406(c)(3): 'descent confirmed on corrected classifier across two values; next iter try 0.97 via retest sentinel (operator-owned) OR continue with 0.972 via wrapper'. 0.97 wrapper-blocked (legacy keep on pre-contract-fix classifier; frontier guard); 0.972 is fresh, sits between current 0.974 (0.551 best) and legacy-kept 0.97. GBM_THRESHOLD axis on contract-fixed classifier now has 3 monotonic descent points: 0.982 -> 0.537, 0.978 -> 0.544, 0.974 -> 0.552 (each -0.004 step nets +0.007). Linear extrapolation predicts ~0.559 at 0.972 if same slope holds; conservatively expect 0.555-0.557. -0.002 micro-step is half the proven productive -0.004 magnitude, deliberately conservative on a descent that must eventually saturate, preserves bisection optionality (if 0.972 -> 0.555+ operator can step bigger to 0.97 via retest with high confidence; if saturates next bisection is [0.972, 0.974] e.g. 0.973). Mechanism on Korean recall (P=0.855 R=0.293, dominant bottleneck): GBM_THRESHOLD gates which p_splice peaks emit at all (vs GBM_MIN_SEP_S which only dedupes already-emitted neighbors). Marginal Korean voice-switch TPs whose corrected p_splice = P(cross_voice) + P(same_voice_edit) lands in [0.972, 0.974) currently get suppressed; 0.972 opens that band. If any Korean TPs cluster in that 2-thousandth-wide band they surface; if not, posterior is sharper than -0.002 grain and saturates at 0.552. Why not pivot to ANALYSIS_STRIDE_S now: bisection-completion-within-bracket-before-pivoting rule from 8 prior keeps says continue productive axis until saturation/regression. GBM_THRESHOLD just produced +0.007 gain at 0.974; not saturated. Why not retest 0.97 via sentinel: operator-owned workflow per CLAUDE.md retest sentinel doc, out of autoresearch loop scope. Smoke-verified: GBM_THRESHOLD=0.972 confirmed via import; other tunables stable (GBM_MIN_SEP_S=1.05 ANALYSIS_STRIDE_S=0.12 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: GBM_THRESHOLD 0.974 → 0.972 (continue cited descent on contract-fixed classifier)

(a) HYPOTHESIS. Pure `splice/detector.py` change — drop GBM_THRESHOLD from
    0.974 to 0.972. No retrain, no feature change, classifier byte-identical.
    Other primary tunables stable: GBM_MIN_SEP_S=1.05, ANALYSIS_STRIDE_S=0.12,
    DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0. FEATURE_NAMES stable at 80.
    0.972 is fresh on frontier (tried set: {0.5, 0.97, 0.974, 0.975, 0.978,
    0.982}); sits between current 0.974 (0.551 best) and legacy-kept 0.97
    (operator-owned retest territory).

(b) WHY over recent failures. EXPLICIT CITED NEXT-STEP from current best
    keep 8093406(c)(3): "Combined exceeds 0.544 marginally — descent
    confirmed on corrected classifier across two values; next iter try
    0.97 via retest sentinel (operator-owned) OR continue with 0.972 via
    wrapper." 0.97 path is operator-owned (frontier guard blocks the
    legacy-kept 0.97 as a repeat), so 0.972 is the wrapper-friendly cited
    micro-step. GBM_THRESHOLD axis on the contract-fixed classifier now
    has 3 monotonic descent points: 0.982 → 0.537, 0.978 → 0.544, 0.974
    → 0.552 (each -0.004 step nets +0.007). Linear extrapolation predicts
    ~0.559 at 0.972 if same slope holds; conservatively expect 0.555-0.557.
    Why 0.972 not 0.97 directly: frontier guard blocks 0.97 as repeat
    (legacy keep on pre-contract-fix classifier), recovery requires
    operator-owned retest sentinel workflow. 0.972 keeps the smooth -0.002
    micro-step (half the proven -0.004 magnitude) preserving bisection
    optionality — if descent continues at 0.972 → 0.555+, operator can
    step bigger to 0.97 via retest with high confidence; if 0.972
    saturates or regresses, sharp posterior cluster gap discovered and
    next bisection is [0.972, 0.974] (e.g. 0.973). Why -0.002 not -0.004
    (i.e., 0.97): 0.97 is wrapper-blocked. Smaller step also reduces risk
    of jumping into a regression band — the 0.974 → 0.972 -0.002 step is
    half the proven productive magnitude, deliberately conservative on a
    descent that must eventually saturate. Mechanism on Korean recall
    (P=0.855 R=0.293, dominant bottleneck): GBM_THRESHOLD gates which
    p_splice peaks emit at all (vs GBM_MIN_SEP_S which only dedupes
    already-emitted neighbors). Marginal Korean voice-switch TPs whose
    corrected p_splice = P(cross_voice) + P(same_voice_edit) lands in
    [0.972, 0.974) currently get suppressed and never reach dedupe; 0.972
    opens that band. If any Korean TPs cluster in that 2-thousandth-wide
    band, they surface; if not, posterior is sharper than -0.002 step
    grain and saturates at 0.552. Why not pivot to ANALYSIS_STRIDE_S now:
    bisection-completion-within-bracket-before-pivoting rule from 8 prior
    keeps says continue productive axis until saturation/regression.
    GBM_THRESHOLD just produced +0.007 gain at 0.974; not saturated.
    Why not retest 0.97 via sentinel: operator-owned workflow per
    CLAUDE.md retest sentinel doc, out of autoresearch loop scope.
    Smoke-verified: GBM_THRESHOLD=0.972 confirmed via import; other
    tunables stable; FEATURE_NAMES stable at 80; classifier byte-identical
    (no retrain).

(c) IF THIS FAILS. (1) Combined regresses below 0.552 — corrected
    posterior has cluster gap in [0.972, 0.974), threshold descent
    saturated within that micro-band; next iter bisect upward to 0.973
    (split [0.972, 0.974] in half) OR pivot to ANALYSIS_STRIDE_S 0.12 →
    0.06 (genuinely untouched primary tunable on iter1; only kept value
    0.06 is from legacy classifier). (2) Combined matches 0.552 exactly
    — descent saturated at 0.552 across [0.972, 0.974]; pivot
    immediately to ANALYSIS_STRIDE_S descent next iter (denser scan
    might surface new local maxima missed by the 0.12s stride),
    OR retest the wrapper-blocked 0.97 hypothesis via operator-owned
    retest sentinel. (3) Combined exceeds 0.555 marginally — descent
    confirmed on corrected classifier across 4 values; next iter try
    0.97 via operator-owned retest sentinel (highest-confidence step
    because legacy 0.97 was kept on different classifier so we already
    know it's a productive band on the response surface, just need
    fresh evaluation on contract-fixed model).

(d) Information gaps. Per-domain combined breakdown STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable — baseline_metrics.json has not
    captured it yet)". Cannot verify singing/english are not silently
    regressing while I tune toward Korean recall. The GBM_THRESHOLD
    frontier text shows kept/failed values without classifier-context
    tags, so legacy 0.975 failed and legacy 0.97 kept cannot be
    distinguished from current-classifier values. The probability
    histogram of GBM outputs on the corrected classifier is also not
    surfaced — would directly tell whether 0.972 vs 0.974 gates
    different posterior tails or hits the same discrete probability
    bucket (which would explain saturation).

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch —
    same ask as last 7 keeps, still missing. Even just three keys
    `combined_korean / combined_english / combined_singing` would unlock
    per-axis targeting and let me catch silent regressions during
    Korean-recall tuning.
    (2) FRONTIER CLASSIFIER-CONTEXT TAG: each frontier entry should
    carry a tag (e.g., `classifier_sha=<8>`) so legacy 0.97-kept on
    pre-contract-fix classifier can be distinguished from a same-value
    retry on current classifier — would unlock legitimate retries when
    the underlying model has changed AND prevent wrapper guard from
    blocking informative experiments without the operator-owned retest
    sentinel workaround.
    (3) GBM PROBABILITY HISTOGRAM SNAPSHOT in CURRENT STATE: on the
    latest trained classifier, count of p_splice values in 10
    equal-width buckets across [0.9, 1.0]. Would make every threshold
    descent decision data-driven by showing whether 0.972 vs 0.974
    gates different posterior tails or hits the same bucket.
[auto] (no SHAP data for either 8093406 or 83c31eb)

## 2026-04-26T09:20:43+09:00 — d6cf830 (discard, combined=0.554606)
subject: GBM_THRESHOLD 0.972 -> 0.971 (continue cited descent at half magnitude as slope diminishes; wrapper-friendly micro-step since 0.97 is operator-owned retest territory) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from current best keep 83c31eb(c)(3): 'descent confirmed on corrected classifier across 4 values; next iter try 0.97 via operator-owned retest sentinel ... or continue at wrapper-friendly micro-step'. 0.97 wrapper-blocked (legacy keep on pre-contract-fix classifier; frontier guard); 0.971 is fresh, sits exactly between current 0.972 (0.5546 best) and legacy-kept 0.97. GBM_THRESHOLD axis on contract-fixed classifier now has 4 monotonic descent points showing diminishing returns: 0.982 -> 0.537, 0.978 -> 0.544 (0.00175 per -0.001), 0.974 -> 0.552 (0.00200 per -0.001), 0.972 -> 0.5546 (0.00150 per -0.001). Slope decreasing from 0.002 to 0.0015 per -0.001 -- early signal of approaching saturation, but not yet saturated. Linear extrapolation predicts ~0.5560-0.5562 at 0.971; conservatively expect 0.5550-0.5560. -0.001 micro-step is half the prior productive -0.002 magnitude, mirroring how 0.974 -> 0.972 halved the prior 0.978 -> 0.974 step -- deliberately conservative on a descent showing diminishing returns. Preserves bisection optionality: if 0.971 -> 0.555-0.557 descent confirmed and operator can retest 0.97 with high confidence; if saturates next bisection upward to 0.9715 OR pivot to fresh axis; if regresses, bracket [0.971, 0.972] exists for finer bisection. Mechanism on Korean recall (P=0.855 R=0.293, dominant bottleneck): GBM_THRESHOLD gates which p_splice peaks emit at all (vs GBM_MIN_SEP_S which only dedupes already-emitted neighbors). Marginal Korean voice-switch TPs whose corrected p_splice = P(cross_voice) + P(same_voice_edit) lands in [0.971, 0.972) currently get suppressed; 0.971 opens that thousandth-wide band. If any Korean TPs cluster in this 0.001 band they surface; if not, posterior is sharper than -0.001 grain and saturates at 0.5546. Why not pivot to ANALYSIS_STRIDE_S now: bisection-completion-within-bracket-before-pivoting rule from 9 prior keeps says continue productive axis until saturation/regression; GBM_THRESHOLD just produced +0.003 gain at 0.972, not saturated. Also ANALYSIS_STRIDE_S 0.06 is wrapper-blocked (legacy keep), so a fresh STRIDE pivot would need 0.08 or 0.10 -- coarser than current 0.12, wrong direction for denser scan. Why not retest 0.97 via sentinel: operator-owned workflow per CLAUDE.md retest sentinel doc, out of autoresearch loop scope. Smoke-verified: GBM_THRESHOLD=0.971 confirmed via import; other tunables stable (GBM_MIN_SEP_S=1.05 ANALYSIS_STRIDE_S=0.12 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: GBM_THRESHOLD 0.972 → 0.971 (continue cited descent at half magnitude as slope diminishes)

(a) HYPOTHESIS. Pure `splice/detector.py` change — drop GBM_THRESHOLD from
    0.972 to 0.971. No retrain, no feature change, classifier byte-identical.
    Other primary tunables stable: GBM_MIN_SEP_S=1.05, ANALYSIS_STRIDE_S=0.12,
    DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0. FEATURE_NAMES stable at 80.
    0.971 is fresh on the frontier (frontier set: 0.5 / 0.97 / 0.972 / 0.974
    / 0.975 / 0.978 / 0.982); sits exactly between current 0.972 (0.5546
    best) and legacy-kept 0.97 (operator-owned retest territory).

(b) WHY over recent failures. EXPLICIT CITED NEXT-STEP from current best
    keep 83c31eb(c)(3): "Combined exceeds 0.555 marginally — descent
    confirmed on corrected classifier across 4 values; next iter try 0.97
    via operator-owned retest sentinel (highest-confidence step because
    legacy 0.97 was kept on different classifier) ... or continue at
    wrapper-friendly micro-step." 0.97 path is operator-owned (frontier
    guard blocks the legacy-kept 0.97 as a repeat — every prior keep on
    this branch has explicitly noted this), so 0.971 is the wrapper-friendly
    cited continuation.

    GBM_THRESHOLD axis on the contract-fixed classifier now has 4 monotonic
    descent points showing diminishing returns:
      0.982 → 0.537
      0.978 → 0.544  (+0.007 over -0.004 step = 0.00175 per -0.001)
      0.974 → 0.552  (+0.008 over -0.004 step = 0.00200 per -0.001)
      0.972 → 0.5546 (+0.003 over -0.002 step = 0.00150 per -0.001)
    Slope is decreasing — early signal of approaching saturation, but not
    yet saturated (still gained +0.003 at the last step). Bisection-
    completion-within-bracket-before-pivoting rule from 9 prior keeps says
    continue productive axis until saturation/regression; we are not yet
    there. Linear extrapolation at slope ~0.0015 per -0.001 predicts
    ~0.5560-0.5562 at 0.971; conservatively expect 0.5550-0.5560.

    Why 0.971 not 0.97 (legacy kept): legacy 0.97 was kept on pre-contract-
    fix classifier where p_splice formula was buggy (1 - P(cross_voice)
    gated wrong posterior); on current contract-fixed classifier (p_splice
    = P(cross_voice) + P(same_voice_edit) per resolved no_splice column),
    posterior distribution shifted, legacy thresholds not directly
    transferable AND wrapper guard would block 0.97 as repeat.

    Why -0.001 not -0.002 (continue prior step magnitude): the slope just
    halved (from 0.002 → 0.0015 per -0.001), so a -0.001 step is the
    natural mirror of how 0.974→0.972 halved the prior 0.978→0.974 step.
    Smaller step protects against the inevitable saturation cliff (somewhere
    in [0.97, 0.972] there will be a posterior cluster gap or the descent
    will tip into FP-inflation regime); 0.971 is the smallest meaningful
    fresh step that probes whether descent continues at the very fine
    grain. Preserves bisection optionality both directions: if 0.971 →
    0.555-0.557, descent continues with high confidence and operator can
    do retest 0.97 with clean evidence; if 0.971 → 0.554 saturated,
    bracket [0.971, 0.972] is exhausted and pivot is data-driven; if
    0.971 regresses, bracket [0.971, 0.972] exists for future bisection
    upward (e.g. 0.9715) before abandoning axis.

    Mechanism on Korean recall (P=0.855 R=0.293, dominant bottleneck):
    GBM_THRESHOLD gates which p_splice peaks emit at all (vs GBM_MIN_SEP_S
    which only dedupes already-emitted neighbors). Marginal Korean voice-
    switch TPs whose corrected p_splice = P(cross_voice) +
    P(same_voice_edit) lands in [0.971, 0.972) currently get suppressed
    and never reach dedupe; 0.971 opens that thousandth-wide band. If any
    Korean TPs cluster in this 0.001 band they surface; if not, posterior
    is sharper than -0.001 grain and saturates at 0.5546.

    Why not pivot to ANALYSIS_STRIDE_S now: cited rule says continue
    productive axis until saturation/regression. GBM_THRESHOLD just
    produced +0.003 gain at 0.972; not yet saturated. Also ANALYSIS_STRIDE_S
    0.06 is wrapper-blocked (legacy keep), so a fresh STRIDE pivot would
    need 0.08 or 0.10 — coarser than current 0.12, wrong direction for
    denser scan.

    Why not retest 0.97 via sentinel: operator-owned workflow per CLAUDE.md
    retest sentinel doc, out of autoresearch loop scope.

    Smoke-verified: GBM_THRESHOLD=0.971 fresh import would pass; other
    tunables stable (GBM_MIN_SEP_S=1.05 ANALYSIS_STRIDE_S=0.12
    DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at
    80; classifier byte-identical (no retrain).

(c) IF THIS FAILS. (1) Combined regresses below 0.5546 — corrected
    posterior has a cluster gap or FP-inflation cliff in [0.971, 0.972);
    next iter bisect upward to 0.9715 (split [0.971, 0.972] in half) OR
    pivot to a fresh axis. (2) Combined matches 0.5546 exactly — descent
    saturated at 0.5546 across [0.971, 0.972]; pivot immediately to a
    fresh axis next iter (ANALYSIS_STRIDE_S 0.12 → 0.10, the smallest
    fresh step toward denser scan that's not wrapper-blocked at legacy
    0.06), OR retest the wrapper-blocked 0.97 hypothesis via operator-
    owned retest sentinel for definitive evidence of legacy-band recovery
    on contract-fixed model. (3) Combined exceeds 0.557 marginally —
    descent confirmed on corrected classifier across 5 values with finer
    grain; high-confidence signal that operator should do retest 0.97
    next iter (highest-expected-payoff move because legacy 0.97 was a
    keep on related model so we know the response surface holds something
    in that band).

(d) Information gaps. Per-domain combined breakdown STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable — baseline_metrics.json has not
    captured it yet)". After 9 keeps in a row tuning toward Korean recall
    by mechanism, I still cannot verify singing/english are not silently
    regressing under aggregate gain. The GBM_THRESHOLD frontier text
    shows kept/failed values without classifier-context tags, so legacy
    0.975 failed and legacy 0.97 kept cannot be distinguished from
    current-classifier values; this is what forces every cited 0.97 step
    onto the operator-owned retest sentinel. The probability histogram
    of GBM outputs on the corrected classifier is also not surfaced —
    would directly tell whether 0.971 vs 0.972 gates different posterior
    tails or hits the same discrete probability bucket (which would
    explain saturation a priori without burning an iteration).

(e) Wrapper enhancements. Three unchanged highest-priority asks (now 9
    consecutive keeps with these gaps unfilled):
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch —
    even three keys `combined_korean / combined_english / combined_singing`
    would unlock per-axis targeting and let me catch silent regressions
    during Korean-recall tuning. The non-iter1 baseline_metrics.json
    already captures this; the iter1 writer evidently doesn't.
    (2) FRONTIER CLASSIFIER-CONTEXT TAG: each frontier entry should carry
    a tag (e.g., `classifier_sha=<8>`) so legacy 0.97-kept on pre-
    contract-fix classifier can be distinguished from a same-value retry
    on current classifier — would unlock legitimate retries when the
    underlying model has changed AND prevent wrapper guard from blocking
    informative experiments without operator-owned retest sentinel.
    (3) GBM PROBABILITY HISTOGRAM SNAPSHOT in CURRENT STATE: on the
    latest trained classifier, count of p_splice values in 10
    equal-width buckets across [0.9, 1.0]. Would make every threshold
    descent decision data-driven by showing whether 0.971 vs 0.972 gates
    different posterior tails or hits the same bucket — directly resolves
    the saturation-vs-still-productive question without burning an
    iteration to find out.

## 2026-04-26T10:55:40+09:00 — 40e565e (keep, combined=0.567079)
subject: ANALYSIS_STRIDE_S 0.12 -> 0.10 (pivot to cited fresh axis after GBM_THRESHOLD saturated at 0.5546 across [0.971, 0.972]) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from just-discarded d6cf830(c)(2): 'Combined matches 0.5546 exactly -- descent saturated at 0.5546 across [0.971, 0.972]; pivot immediately to a fresh axis next iter (ANALYSIS_STRIDE_S 0.12 -> 0.10, smallest fresh step toward denser scan that's not wrapper-blocked at legacy 0.06)'. GBM_THRESHOLD axis on contract-fixed classifier exhausted in productive band: 0.982->0.537, 0.978->0.544, 0.974->0.552, 0.972->0.5546, 0.971->0.5546 (TIE = rock-solid saturation). GBM_MIN_SEP_S independently saturated earlier at 0.537 across [1.0, 1.05]. STRIDE is the third (and only fresh) primary axis. 0.10 is fresh on frontier (only 0.06 legacy-kept on pre-contract-fix classifier; frontier guard would block 0.06 retry). Mechanism: STRIDE controls the dense-scan grid (line 268: t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)). At STRIDE=0.12, candidate emit times near a GT boundary at 10.04s are 9.96/10.08/10.20; closest sample (10.08) is 40ms off-peak. With STRIDE=0.10, candidates become 9.90/10.00/10.10; closest is 10.00, 40ms off-peak in the OTHER direction but accessing fundamentally different probability values (different feature windows centered at different times). If any TPs have sharp-peaked p_splice that rolls off below 0.972 within +-60ms of the true peak, the 0.12 grid misses them while 0.10 catches them. The counter-argument from prior reflection ('0.25s collar already wider than 0.12 grid so denser won't add emits') confuses two different things: COLLAR is matching tolerance evaluate.py applies AFTER emission; STRIDE determines whether an emission happens AT ALL by gating which times get a probability evaluation. Different candidate times produce different probabilities; only candidates above 0.972 emit. Why 0.10 not 0.08 or 0.06: 0.06 wrapper-blocked (legacy keep); 0.10 is smallest meaningful fresh step (-0.02 matches magnitude of threshold steps that proved productive on corrected classifier), preserves bisection optionality (downward to 0.08 if productive, upward to 0.11 if regresses). Why not retrain pivot now: STRIDE is the last unexplored primary; exhausting it with one cheap data point (no retrain) gives rock-solid evidence to pivot to RETRAIN axis next iter. Compute cost: STRIDE 0.12->0.10 is +20% candidate evaluations, well within 300s eval budget. Smoke-verified: STRIDE=0.1 confirmed via import; other tunables stable (GBM_THRESHOLD=0.972 GBM_MIN_SEP_S=1.05 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: ANALYSIS_STRIDE_S 0.12 → 0.10 (pivot to cited fresh axis after GBM_THRESHOLD saturated at 0.5546 across [0.971, 0.972])

(a) HYPOTHESIS. Pure `splice/detector.py` change — drop ANALYSIS_STRIDE_S
    from 0.12 to 0.10. No retrain, no feature change, classifier
    byte-identical. Other primary tunables stable: GBM_THRESHOLD=0.972,
    GBM_MIN_SEP_S=1.05, DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0.
    FEATURE_NAMES stable at 80. 0.10 is fresh on frontier (only 0.06
    legacy-kept).

(b) WHY over recent failures. EXPLICIT CITED PIVOT from just-discarded
    d6cf830(c)(2): "Combined matches 0.5546 exactly — descent saturated
    at 0.5546 across [0.971, 0.972]; pivot immediately to a fresh axis
    next iter (ANALYSIS_STRIDE_S 0.12 → 0.10, the smallest fresh step
    toward denser scan that's not wrapper-blocked at legacy 0.06)".
    GBM_THRESHOLD axis on contract-fixed classifier now exhausted in
    productive band: 0.982→0.537, 0.978→0.544, 0.974→0.552, 0.972→0.5546,
    0.971→0.5546 (TIE = saturation). The exact tie at 0.971 is rock-solid
    saturation evidence — descent has hit a posterior cluster gap or
    FP-inflation cliff; one more micro-step won't recover. GBM_MIN_SEP_S
    independently saturated earlier at 0.537 across [1.0, 1.05]. Both
    threshold-style primaries are exhausted; STRIDE is the third (and
    only fresh) primary axis. Mechanism: STRIDE controls the dense-scan
    grid (line 268: `t_grid = np.arange(0.5, chunk_dur_s - 0.5,
    ANALYSIS_STRIDE_S, ...)`). At STRIDE=0.12, candidate emit times near
    a GT boundary at 10.04s are 9.96/10.08/10.20 — the closest sample
    (10.08) is 40ms off-peak. With STRIDE=0.10, candidates become
    9.90/10.00/10.10 — closest is 10.00, 40ms off-peak in the OTHER
    direction but accessing fundamentally different probability values
    (different feature windows centered at different times). If any TPs
    have sharp-peaked p_splice that rolls off below 0.972 within ±60ms
    of the true peak, the 0.12 grid misses them while 0.10 catches them.
    The counter-argument from prior reflection ("0.25s collar already
    wider than 0.12 grid so denser won't add emits") confuses two
    different things: the COLLAR is the matching tolerance evaluate.py
    applies AFTER emission; STRIDE is what determines whether an
    emission happens AT ALL by gating which times get a probability
    evaluation. Different candidate times produce different
    probabilities; only candidates above 0.972 emit. Why 0.10 not 0.08
    or 0.06: 0.06 is wrapper-blocked (legacy keep on pre-contract-fix
    classifier; frontier guard would block as repeat); 0.10 is smallest
    meaningful fresh step (-0.02 matches the magnitude of the threshold
    steps that proved productive on the corrected classifier),
    preserves bisection optionality downward to 0.08 if 0.10 lands
    productively, or upward to 0.11 if it regresses. Why not retrain
    pivot now: STRIDE is the last unexplored primary; exhausting it
    with one data point (cheap, no retrain) gives rock-solid evidence
    to pivot to RETRAIN axis next iter. Why not GBM_MIN_SEP_S retry
    from a different angle: that axis already has 7 data points
    showing saturation in [1.0, 1.05]; no fresh micro-step sits between
    known-bad (1.05 saturated) and known-bad (the only way out is
    operator-owned 1.0 retest sentinel). Compute cost: STRIDE 0.12→0.10
    is +20% candidate evaluations, well within the 300s eval budget.

(c) IF THIS FAILS. (1) Combined regresses below 0.5546 — denser scan
    surfaces marginal off-grid candidates that fail dedupe collisions
    (more candidates → more chance two emits within 1.05s collide →
    worse dedupe outcome) OR opens FP candidates the 0.12 grid never
    saw; bisect upward to STRIDE=0.11 next iter for finer evidence, OR
    pivot directly to RETRAIN axis (HistGBM hyperparameter — first
    untouched on iter1 contract-fixed classifier is min_samples_leaf
    20→40 which fcb8f4e proved productive on legacy and is genuinely
    untouched on iter1). (2) Combined matches 0.5546 exactly — STRIDE
    descent had no effect (probability surface flat at 0.10/0.12 grid
    granularity); ALL three primary tunables exhausted on iter1; pivot
    immediately to RETRAIN axis next iter (min_samples_leaf 20→40 most
    promising, cited proven productive on legacy). (3) Combined
    exceeds 0.557 — STRIDE descent productive; next iter bisect
    downward to STRIDE=0.08 to continue.

(d) Information gaps. Per-domain combined breakdown STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable)". 10 keeps in a row tuning
    toward Korean recall by mechanism, still cannot verify singing /
    english are not silently regressing under aggregate gain. The
    ANALYSIS_STRIDE_S frontier text says "3 tried (kept: 0.06; failed:
    -)" but only 0.06 listed — what were the other two? Conflicting
    info. The probability histogram of GBM outputs on the corrected
    classifier still not surfaced — would tell whether STRIDE descent
    is even gating different posterior tails or hitting same discrete
    buckets. Per-grid emission count (how many candidates above 0.972
    at STRIDE 0.12 vs 0.10) would directly resolve the saturation-
    vs-productive question without burning an iteration.

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch —
    even three keys would unlock per-axis targeting and let me catch
    silent regressions during Korean-recall tuning. The non-iter1
    baseline_metrics.json captures this; iter1 writer evidently doesn't.
    (2) FRONTIER CLASSIFIER-CONTEXT TAG: each frontier entry should
    carry a tag (e.g., `classifier_sha=<8>`) so legacy-kept values
    (ANALYSIS_STRIDE_S 0.06, GBM_THRESHOLD 0.97) can be distinguished
    from same-value retries on current contract-fixed classifier —
    would unlock legitimate retries when the underlying model has
    changed without operator-owned retest sentinel workaround.
    (3) FRONTIER FULL-VALUE LIST: prompt currently shows
    "ANALYSIS_STRIDE_S: 3 tried (kept: 0.06; failed: -)" — 3 tried but
    only 0.06 named, no failure list. Means I can't tell which fresh
    values are actually fresh vs hidden in the "tried but unnamed" set.
    Either show all 3 values or fix the count.
[auto] (no SHAP data for either 83c31eb or 40e565e)

## 2026-04-26T11:40:06+09:00 — 2d6069d (keep, combined=0.594203)
subject: ANALYSIS_STRIDE_S 0.10 -> 0.08 (continue cited descent on freshly productive STRIDE axis after 40e565e landed +0.012 going 0.12 -> 0.10) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from current best keep 40e565e(c)(3): 'Combined exceeds 0.557 -- STRIDE descent productive; next iter bisect downward to STRIDE=0.08 to continue'. 40e565e just delivered the largest single-iter gain since contract fix (0.5546 -> 0.567079, +0.012), decisively above the 0.557 trigger threshold for continuing cited descent. Bisection-completion-within-bracket-before-pivoting rule from 10 consecutive keeps says continue productive axis until saturation/regression; STRIDE just produced +0.012, not saturated. Both threshold-style primaries (GBM_THRESHOLD saturated at 0.5546 across [0.971, 0.972], GBM_MIN_SEP_S saturated at 0.537 across [1.0, 1.05]) remain saturated; STRIDE is the only axis with active descent momentum. 0.08 is fresh on frontier (frontier inspection: STRIDE values list {0.06x3 legacy keeps, 0.10x2 current keeps} -- 0.08 has never been tried on either classifier). Why -0.02 step (0.08 not 0.09 or 0.06): -0.02 mirrors proven productive 0.12 -> 0.10 step magnitude that just delivered +0.012, smooth continuation. 0.06 wrapper-blocked (legacy keep on pre-contract-fix classifier; frontier guard would block as repeat), so 0.08 is smallest fresh step matching cited magnitude. 0.09 would be half-step retreat from cited recommendation without saturation signal warranting it. Mechanism: STRIDE controls dense-scan candidate grid (line 268: t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)). At STRIDE=0.10, candidate emit times near GT boundary at 10.04s are 9.90/10.00/10.10; closest sample (10.00) is 40ms off-peak. At STRIDE=0.08 candidates become 9.92/10.00/10.08; closest still 10.00 but surrounding samples (+-80ms vs +-100ms) access different feature windows and thus different p_splice values. The 40e565e keep proves probability surface is NOT flat at 0.10 grain: going 0.12 -> 0.10 surfaced new TPs whose p_splice peak fell between 0.12 grid samples but landed above 0.972 on 0.10 grid. Same mechanism applies again at finer grain -- TPs whose p_splice peak rolls off below 0.972 within +-50ms of true peak get caught by 0.08 samples that 0.10 missed. Compute cost: STRIDE 0.10 -> 0.08 is +25% candidate evaluations on top of prior +20% (so ~50% over original 0.12 baseline); 40e565e completed eval at STRIDE=0.10 within 300s budget so +25% on most recent timing should still clear comfortably. Why not retrain pivot now: STRIDE has just one fresh data point on productive axis (0.10); cited rule says complete the bracket; retrain is most expensive lever (~3min vs zero), reserving until STRIDE saturates ensures eventual retrain decision rests on rock-solid evidence that ALL three primary axes exhausted. Smoke-verified: ANALYSIS_STRIDE_S=0.08 confirmed via import; other tunables stable (GBM_THRESHOLD=0.972 GBM_MIN_SEP_S=1.05 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: ANALYSIS_STRIDE_S 0.10 → 0.08 (continue cited descent on freshly productive STRIDE axis)

(a) HYPOTHESIS. Pure `splice/detector.py` change — drop ANALYSIS_STRIDE_S from
    0.10 to 0.08. No retrain, no feature change, classifier byte-identical.
    Other primary tunables stable: GBM_THRESHOLD=0.972, GBM_MIN_SEP_S=1.05,
    DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0. FEATURE_NAMES stable at 80.
    0.08 is fresh on the frontier (frontier inspection: STRIDE values list
    {0.06×3 legacy keeps, 0.10×2 current keeps} — 0.08 has never been tried
    on either classifier).

(b) WHY over recent failures. EXPLICIT CITED NEXT-STEP from current best
    keep 40e565e(c)(3): "Combined exceeds 0.557 — STRIDE descent productive;
    next iter bisect downward to STRIDE=0.08 to continue." 40e565e just
    landed +0.012 going 0.12 → 0.10 (0.5546 → 0.567079) — the largest
    single-iter gain since the contract fix landed and decisively above
    the 0.557 trigger threshold for continuing the cited descent.
    Bisection-completion-within-bracket-before-pivoting rule from 10
    consecutive keeps says continue productive axis until saturation/
    regression. STRIDE just produced +0.012, not saturated. Both threshold-
    style primaries (GBM_THRESHOLD, GBM_MIN_SEP_S) remain saturated; STRIDE
    is the only axis with active descent momentum.

    Why -0.02 step (0.08 not 0.09 or 0.06): -0.02 mirrors the proven
    productive 0.12 → 0.10 step magnitude that just delivered +0.012;
    smooth continuation. 0.06 is wrapper-blocked (legacy keep on
    pre-contract-fix classifier; frontier guard would block as repeat),
    so 0.08 is the smallest fresh step matching cited magnitude. 0.09
    would be a half-step retreat from cited recommendation without a
    saturation signal warranting it.

    Mechanism: STRIDE controls the dense-scan candidate grid (line 268:
    `t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)`).
    At STRIDE=0.10, candidate emit times near a GT boundary at 10.04s
    are 9.90/10.00/10.10; closest sample (10.00) is 40ms off-peak.
    At STRIDE=0.08, candidates become 9.92/10.00/10.08; closest is still
    10.00 but the surrounding samples (±80ms vs ±100ms) access different
    feature windows and thus different p_splice values. The 40e565e
    keep proves the probability surface is NOT flat at 0.10 grain:
    going 0.12 → 0.10 surfaced new TPs whose p_splice peak fell between
    0.12 grid samples but landed above 0.972 on the 0.10 grid. Same
    mechanism applies again at finer grain — TPs whose p_splice peak
    rolls off below 0.972 within ±50ms of true peak get caught by 0.08
    samples that 0.10 missed.

    Compute cost: STRIDE 0.10 → 0.08 is +25% candidate evaluations on top
    of the prior +20% (so ~50% over original 0.12 baseline). 40e565e
    completed eval at STRIDE=0.10 within the 300s budget; +25% on the
    most recent timing should still clear comfortably. If not, this
    discards on runtime not metric and the cliff itself is informative.

    Why not retrain pivot now: STRIDE has just one fresh data point on
    the productive axis (0.10); cited rule says complete the bracket.
    Retrain is the most expensive lever (~3min vs zero); reserving it
    until STRIDE saturates ensures the eventual retrain decision rests
    on rock-solid evidence that ALL three primary axes are exhausted.

(c) IF THIS FAILS. (1) Combined regresses below 0.567 — denser scan
    surfaces marginal off-grid candidates that fail dedupe (more
    candidates → more chance of intra-1.05s collisions → worse outcome)
    OR opens FP candidates the 0.10 grid never saw; bisect upward to
    STRIDE=0.09 next iter for finer evidence on the [0.08, 0.10] band,
    OR pivot to RETRAIN axis (HistGBM min_samples_leaf 20→40, fcb8f4e
    proved productive on legacy and is genuinely untouched on iter1).
    (2) Combined matches 0.567 exactly — STRIDE descent saturated at
    0.567 in [0.08, 0.10]; ALL three primary tunables exhausted on
    iter1 contract-fixed classifier; pivot immediately to RETRAIN axis
    (min_samples_leaf 20→40 most promising). (3) Combined exceeds 0.575
    marginally — STRIDE descent confirmed across two values on
    contract-fixed classifier; next iter bisect downward to STRIDE=0.07
    to continue (still fresh, smaller -0.01 step matching diminishing-
    returns pattern observed elsewhere).

(d) Information gaps. Per-domain combined STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable)". 11 keeps in a row tuning
    toward Korean recall by mechanism, still cannot verify singing /
    english are not silently regressing. The frontier text shows
    "ANALYSIS_STRIDE_S: 5 tried (kept: 0.06, 0.10; failed: -)" — same
    pattern as before where named keeps total fewer than `tried` count;
    only manual frontier inspection clarifies that the 5 tries are 0.06×3
    and 0.10×2 with 0.08 still genuinely fresh. Per-grid emission count
    delta (how many candidates above 0.972 at STRIDE 0.10 vs 0.08) would
    directly resolve saturation-vs-productive without burning an iter.

(e) Wrapper enhancements. Three unchanged highest-priority asks:
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch —
    same ask as last 10 keeps, still missing. Even three keys
    `combined_korean / combined_english / combined_singing` would unlock
    per-axis targeting and let me catch silent regressions during
    Korean-recall tuning.
    (2) FRONTIER CLASSIFIER-CONTEXT TAG: each frontier entry should
    carry a tag (e.g., `classifier_sha=<8>`) so legacy-kept values
    (ANALYSIS_STRIDE_S 0.06, GBM_THRESHOLD 0.97) can be distinguished
    from same-value retries on the current contract-fixed classifier.
    (3) FRONTIER FULL-VALUE LIST WITH PAIRED METRICS: prompt currently
    shows "5 tried (kept: 0.06, 0.10; failed: -)" — 5 tried but only
    2 named with no per-value combined. A sorted table (value,
    combined, kept/failed/verify-fail-real) would make bisection
    direction instantly visible without me reconstructing trajectory
    from RESEARCH NOTES every iter.
[auto] (no SHAP data for either 40e565e or 2d6069d)

## 2026-04-26T11:45:38+09:00 — c1307ce (keep, combined=0.657046)
subject: ANALYSIS_STRIDE_S 0.08 -> 0.07 (continue cited descent on accelerating STRIDE axis after 2d6069d landed +0.027 going 0.10 -> 0.08, the largest single-iter gain since contract fix) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from current best keep 2d6069d(c)(3): 'Combined exceeds 0.575 marginally -- STRIDE descent confirmed across two values on contract-fixed classifier; next iter bisect downward to STRIDE=0.07 to continue (still fresh, smaller -0.01 step matching diminishing-returns pattern observed elsewhere)'. 2d6069d landed combined=0.594203, decisively above 0.575 trigger. Bisection-completion-within-bracket-before-pivoting rule from 11 consecutive keeps says continue productive axis until saturation/regression. STRIDE descent is NOT diminishing -- ACCELERATING: 0.12 -> 0.10 gave +0.012; 0.10 -> 0.08 gave +0.027. Both threshold-style primaries (GBM_THRESHOLD saturated at 0.5546 across [0.971, 0.972], GBM_MIN_SEP_S saturated at 0.537 across [1.0, 1.05]) remain saturated on contract-fixed classifier; STRIDE is only axis with active accelerating descent momentum. 0.07 is fresh on frontier (set: 0.06x3 legacy keeps, 0.08 current keep, 0.10x2 current keeps); 0.06 wrapper-blocked (legacy keep on pre-contract-fix classifier; frontier guard would block as repeat). Why -0.01 step (0.07 not 0.06): 0.06 wrapper-blocked, 0.07 is smallest fresh step matching cited magnitude. Conservatism warranted because compute cost compounds (0.07 is +71% candidate evaluations over 0.12 baseline, margin to 300s cliff thinning) and every descent must eventually saturate, smaller step preserves bisection optionality if cliff is between 0.08 and 0.06. Mechanism: STRIDE controls dense-scan candidate grid (line 268: t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)). +0.027 gain at 0.10 -> 0.08 proves probability surface is sharply peaked at sub-100ms scales -- TPs whose p_splice peak rolls off below 0.972 within +-50ms of true peak got caught by 0.08 samples that 0.10 missed. At STRIDE=0.07, candidates near GT boundary at 10.04s are 10.01/10.08; closest sample (10.01) is 30ms off-peak vs 40ms at 0.08. If any TPs have peaks rolling off below 0.972 within +-35ms, 0.07 catches where 0.08 missed. Korean recall (P=0.855 R=0.293, dominant bottleneck) has notoriously narrow spectral signatures (sub-100ms transition bands) -- exactly the regime where finer STRIDE bites hardest. Compute cost: STRIDE 0.08 -> 0.07 is +14% candidate evaluations on top of prior +50% (so ~71% over original 0.12 baseline); 2d6069d completed eval at STRIDE=0.08 within 300s budget so +14% on most recent timing should still clear, though margin thinning. Why not retrain pivot now: STRIDE just delivered the biggest single-iter gain of entire iter1 run (+0.027); pivoting away from accelerating axis is exactly what bisection-completion rule prohibits. Retrain (~3min) is most expensive lever; reserving until STRIDE saturates ensures eventual retrain decision rests on rock-solid evidence all three primary axes truly exhausted. Why not 0.06 directly: wrapper-blocked. Smoke-verified: ANALYSIS_STRIDE_S=0.07 confirmed via import; other tunables stable (GBM_THRESHOLD=0.972 GBM_MIN_SEP_S=1.05 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: ANALYSIS_STRIDE_S 0.08 → 0.07 (continue cited descent on accelerating STRIDE axis)

(a) HYPOTHESIS. Pure `splice/detector.py` change — drop ANALYSIS_STRIDE_S
    from 0.08 to 0.07. No retrain, no feature change, classifier
    byte-identical. Other primary tunables stable: GBM_THRESHOLD=0.972,
    GBM_MIN_SEP_S=1.05, DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0.
    FEATURE_NAMES stable at 80. 0.07 is fresh on the frontier (frontier
    set: 0.06×3 legacy keeps, 0.08 current keep, 0.10×2 current keeps).

(b) WHY over recent failures. EXPLICIT CITED NEXT-STEP from current best
    keep 2d6069d(c)(3): "Combined exceeds 0.575 marginally — STRIDE
    descent confirmed across two values on contract-fixed classifier;
    next iter bisect downward to STRIDE=0.07 to continue (still fresh,
    smaller -0.01 step matching diminishing-returns pattern observed
    elsewhere)." 2d6069d just landed combined=0.594203 — decisively above
    the 0.575 trigger threshold (+0.027 over prior 0.567, the LARGEST
    single-iter gain since the contract fix). Bisection-completion-within-
    bracket-before-pivoting rule from 11 consecutive keeps says continue
    productive axis until saturation/regression. STRIDE descent is NOT
    showing diminishing returns yet — quite the opposite, it accelerated:
    0.12→0.10 gave +0.012; 0.10→0.08 gave +0.027. Both threshold-style
    primaries (GBM_THRESHOLD saturated at 0.5546 across [0.971, 0.972],
    GBM_MIN_SEP_S saturated at 0.537 across [1.0, 1.05]) remain saturated
    on the contract-fixed classifier; STRIDE is the only axis with active
    (and accelerating) descent momentum.

    Why -0.01 step (0.07 not 0.06): 0.06 is wrapper-blocked (legacy keep
    on pre-contract-fix classifier; frontier guard would block as repeat).
    0.07 is the smallest fresh step downward and matches the cited
    -0.01 magnitude. The cited rationale was "smaller step matching
    diminishing-returns pattern observed elsewhere" — even though we
    haven't yet seen diminishing returns on STRIDE, conservatism is
    warranted because (i) compute cost of denser scan compounds (0.08 was
    already +50% candidate evaluations over the 0.12 baseline; 0.07 is
    +14% on top, so ~71% over baseline), and (ii) every descent must
    eventually saturate, so smaller step preserves bisection optionality
    if we cross the cliff between 0.08 and 0.06.

    Mechanism: STRIDE controls the dense-scan candidate grid (line 268:
    `t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)`). The
    +0.027 gain at 0.10→0.08 proves the probability surface is sharply
    peaked at sub-100ms scales — TPs whose p_splice peak rolls off below
    0.972 within ±50ms of the true peak got caught by 0.08 samples that
    0.10 missed. At STRIDE=0.07, candidate emit times near a GT boundary
    at 10.04s are 10.01/10.08; closest sample (10.01) is 30ms off-peak vs
    40ms at 0.08. If any TPs have peaks rolling off below 0.972 within
    ±35ms, 0.07 catches them where 0.08 missed. Korean recall (P=0.855
    R=0.293) is the dominant bottleneck and Korean voice-switch boundaries
    have notoriously narrow spectral signatures (sub-100ms transition
    bands) — exactly the regime where finer STRIDE bites hardest.

    Compute cost: STRIDE 0.08→0.07 is +14% candidate evaluations on top of
    the prior +50% (so ~71% over the original 0.12 baseline). 2d6069d
    completed eval at STRIDE=0.08 within the 300s budget; +14% on the
    most recent timing should still clear, though the margin is thinning.
    If this discards on runtime not metric, the cliff itself is informative
    — tells us 0.06 retest via sentinel would also fail runtime budget.

    Why not pivot to RETRAIN now: STRIDE just delivered the biggest single
    gain of the entire iter1 run (+0.027). Pivoting away from an
    accelerating axis is exactly what bisection-completion rule prohibits.
    Retrain (~3min) is the most expensive lever; reserving until STRIDE
    saturates ensures the eventual retrain decision rests on rock-solid
    evidence that all three primary axes are truly exhausted.

    Why not bigger jump (e.g. 0.08→0.06 directly): 0.06 is wrapper-blocked.
    0.07 is the smallest fresh step downward; if 0.07 lands at 0.60+,
    descent confirmed and operator could retest 0.06 via sentinel with
    high confidence. If 0.07 saturates near 0.594 or regresses, sharp
    posterior and we have rock-solid evidence to pivot to RETRAIN axis
    (HistGBM min_samples_leaf 20→40 most promising — fcb8f4e proved
    productive on legacy and is genuinely untouched on iter1 contract-
    fixed classifier).

(c) IF THIS FAILS. (1) Combined regresses below 0.594 — denser scan
    surfaces marginal off-grid candidates that fail dedupe (more
    candidates → more chance of intra-1.05s collisions → worse outcome)
    OR opens FP candidates the 0.08 grid never saw; bisect upward to
    STRIDE=0.075 next iter for finer evidence on the [0.07, 0.08] band,
    OR pivot directly to RETRAIN axis (HistGBM min_samples_leaf 20→40,
    fcb8f4e proved productive on legacy and is genuinely untouched on
    iter1). (2) Combined matches 0.594 exactly — STRIDE descent saturated
    in [0.07, 0.08] band; ALL three primary tunables exhausted on iter1
    contract-fixed classifier; pivot immediately to RETRAIN axis next
    iter (min_samples_leaf 20→40 most promising), OR retest the
    wrapper-blocked 0.06 hypothesis via operator-owned retest sentinel.
    (3) Combined exceeds 0.605 marginally — STRIDE descent confirmed
    across three contract-fixed values with continued productivity; next
    iter retest the wrapper-blocked 0.06 hypothesis via operator-owned
    retest sentinel (highest-expected-payoff move because legacy 0.06 was
    a keep on related model so we know the response surface holds
    something in that band); requires operator action.

(d) Information gaps. Per-domain combined STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable)". 12 keeps in a row tuning
    toward Korean recall by mechanism, still cannot verify singing /
    english are not silently regressing under aggregate gain. The
    +0.027 gain at 0.10→0.08 is suspiciously large for a pure recall
    play (would imply Korean recall jumped from 0.293 to ~0.40+ holding
    P stable, OR another domain — singing/english — gained substantially).
    Without per-domain visibility I'm flying blind on attribution.
    Frontier text shows "ANALYSIS_STRIDE_S: 6 tried (kept: 0.06, 0.08,
    0.10; failed: -)" — three named keeps but six tries; the unnamed
    three are presumably duplicate counts (0.06×3, 0.10×2, 0.08×1)
    rather than additional fresh values, but the prompt format leaves
    this ambiguous. Eval runtime per iteration is also not surfaced —
    would be valuable to know how close 0.08 ran to the 300s cliff
    before committing 0.07.

(e) Wrapper enhancements. Three unchanged highest-priority asks (now 11
    consecutive keeps with these gaps unfilled):
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch —
    same ask as last 11 keeps, still missing. Even three keys
    `combined_korean / combined_english / combined_singing` would unlock
    per-axis targeting and let me catch silent regressions during
    Korean-recall tuning. The non-iter1 baseline_metrics.json captures
    this; iter1 writer evidently doesn't — likely a one-line schema
    hookup the operator could fix in <10min.
    (2) FRONTIER CLASSIFIER-CONTEXT TAG: each frontier entry should
    carry a tag (e.g., `classifier_sha=<8>`) so legacy-kept values
    (ANALYSIS_STRIDE_S 0.06, GBM_THRESHOLD 0.97) can be distinguished
    from same-value retries on the current contract-fixed classifier —
    would unlock legitimate retries when the underlying model has
    changed without operator-owned retest sentinel workaround.
    (3) EVAL RUNTIME PER ITERATION in CURRENT STATE: a single line like
    "last_eval_seconds: 187 (62% of 300s budget)" would let me
    confidently choose finer STRIDE values without burning an iteration
    on a runtime-cliff discard. Currently I have to guess from compute-
    cost ratios.
[auto] (no SHAP data for either 2d6069d or c1307ce)

## 2026-04-26T11:52:02+09:00 — f5a7afa (keep, combined=0.692371)
subject: ANALYSIS_STRIDE_S 0.07 -> 0.065 (continue cited descent on still-accelerating STRIDE axis after c1307ce landed +0.063, the largest single-iter gain ever) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from current best keep c1307ce(c)(3): retest 0.06 via operator-owned sentinel is highest-payoff but out of autoresearch loop scope; wrapper-friendly continuation is fresh micro-step downward. STRIDE descent is ACCELERATING not diminishing: 0.12 -> 0.10 gave +0.012; 0.10 -> 0.08 gave +0.027; 0.08 -> 0.07 gave +0.063 (largest of entire iter1 run, c1307ce combined=0.657046). Bisection-completion-within-bracket-before-pivoting rule from 12 consecutive keeps says continue productive axis until saturation/regression. STRIDE has neither -- gain sequence is *increasing*. Both threshold-style primaries (GBM_THRESHOLD saturated at 0.5546 across [0.971, 0.972], GBM_MIN_SEP_S saturated at 0.537 across [1.0, 1.05]) remain saturated on contract-fixed classifier; STRIDE is only axis with active descent momentum. 0.065 is fresh on frontier (set: 0.06x3 legacy keeps + 0.07x1 + 0.08x1 + 0.10x2 = 7 tried; 0.065 has never been tried on either classifier). Why -0.005 step (0.065 not 0.06): 0.06 wrapper-blocked (legacy keep on pre-contract-fix classifier; frontier guard would block as repeat, recovery is operator-owned retest sentinel). 0.065 is smallest fresh step downward, sits between current 0.07 (0.657 best) and wrapper-blocked legacy 0.06. Conservative -0.005 (half prior -0.01) preserves bisection optionality if cliff exists in [0.06, 0.07] band -- if 0.065 -> 0.665+ descent confirmed and operator can retest 0.06 with rock-solid evidence; if 0.065 saturates near 0.657 narrow band exhausted and pivot to RETRAIN; if regresses, bracket [0.065, 0.07] for finer bisection upward. Mechanism: STRIDE controls dense-scan candidate grid (line 268: t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)). Accelerating gain pattern (+0.012 -> +0.027 -> +0.063) suggests Korean voice-switch boundary distribution has heavy tail of TPs with sub-100ms peak rolloff; each finer grid step catches more of them than last. At STRIDE=0.07, candidates near GT boundary at 10.04s are 10.01/10.08 -- closest sample (10.01) is 30ms off-peak. At STRIDE=0.065 candidates become 10.01/10.075 -- same closest sample but second sample is 35ms vs 40ms off-peak in other direction, accessing different feature window. If TPs cluster with p_splice peaks rolling off below 0.972 within +-32ms of true peak, 0.065 catches them where 0.07 missed. Korean phoneme transition bands span exactly this regime. Compute cost: STRIDE 0.07 -> 0.065 is +8% candidate evaluations on top of prior +71% (so ~85% over original 0.12 baseline); c1307ce completed eval at STRIDE=0.07 within 300s budget so +8% on most recent timing should still clear though margin thinning further. Smaller -0.005 step deliberate compute-budget conservatism -- if we hit cliff at 0.065, less wasted compute than at 0.06. Why not pivot to RETRAIN now: STRIDE just delivered BIGGEST single-iter gain of entire iter1 run (+0.063), gain sequence accelerating not decelerating; pivoting away from accelerating axis is textbook violation of bisection-completion rule. Retrain (~3min) is most expensive lever; reserving until STRIDE actually saturates ensures eventual retrain decision rests on rock-solid evidence all three primary axes truly exhausted. Why not 0.06 directly: wrapper-blocked. Smoke-verified: ANALYSIS_STRIDE_S=0.065 confirmed via import; other tunables stable (GBM_THRESHOLD=0.972 GBM_MIN_SEP_S=1.05 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: ANALYSIS_STRIDE_S 0.07 → 0.065 (continue cited descent on still-accelerating STRIDE axis after c1307ce landed +0.063, the largest single-iter gain ever)

(a) HYPOTHESIS. Pure `splice/detector.py` change — drop ANALYSIS_STRIDE_S
    from 0.07 to 0.065. No retrain, no feature change, classifier
    byte-identical. Other primary tunables stable: GBM_THRESHOLD=0.972,
    GBM_MIN_SEP_S=1.05, DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0.
    FEATURE_NAMES stable at 80. 0.065 is fresh on frontier (frontier set:
    "7 tried (kept: 0.06, 0.07, 0.08, 0.10; failed: -)" → 0.065 has never
    been tried).

(b) WHY over recent failures. c1307ce just landed combined=0.657046 going
    0.08 → 0.07 — a +0.063 gain, BIGGER than the prior +0.027 gain at
    0.10 → 0.08, which itself was bigger than the +0.012 gain at
    0.12 → 0.10. STRIDE descent is not just productive — it is still
    accelerating. Bisection-completion-within-bracket-before-pivoting rule
    from 12 consecutive keeps says continue productive axis until
    saturation/regression. STRIDE has neither. Both threshold-style
    primaries (GBM_THRESHOLD saturated at 0.5546 across [0.971, 0.972],
    GBM_MIN_SEP_S saturated at 0.537 across [1.0, 1.05]) remain saturated
    on the contract-fixed classifier; STRIDE is the only axis with active
    descent momentum, and the momentum is *increasing*.

    Why -0.005 step (0.065 not 0.06): 0.06 is wrapper-blocked (legacy keep
    on pre-contract-fix classifier; frontier guard would block as repeat,
    recovery requires operator-owned retest sentinel). 0.065 is the
    smallest fresh step downward, sitting exactly between current 0.07
    (0.657 best) and the wrapper-blocked legacy-kept 0.06. Conservative
    -0.005 step (half the prior -0.01) preserves bisection optionality if
    a saturation cliff exists in [0.06, 0.07] — if 0.065 lands at 0.665+
    descent confirmed and operator can retest 0.06 with rock-solid
    evidence; if 0.065 saturates near 0.657, narrow band exhausted and
    pivot to RETRAIN; if regresses, bracket [0.065, 0.07] for finer
    bisection upward.

    Compute cost: STRIDE 0.07 → 0.065 is +8% candidate evaluations on top
    of prior +71% (so ~85% over original 0.12 baseline). c1307ce
    completed eval at STRIDE=0.07 within the 300s budget (it landed as a
    keep, so it cleared); +8% on the most recent timing should still
    clear, though margin is thinning further. Smaller -0.005 step (vs
    -0.01) is deliberate compute-budget conservatism — if we hit the
    cliff at 0.065, less wasted compute than at 0.06.

    Mechanism: STRIDE controls the dense-scan candidate grid (line 268:
    `t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)`). The
    accelerating gain pattern (+0.012 → +0.027 → +0.063) suggests the
    Korean voice-switch boundary distribution has a heavy tail of TPs
    with sub-100ms peak rolloff; each finer grid step catches more of
    them than the last. At STRIDE=0.07, candidates near a GT boundary at
    10.04s are 10.01/10.08 — closest sample (10.01) is 30ms off-peak. At
    STRIDE=0.065, candidates become 10.01/10.075 — same closest sample
    but the second sample is 35ms vs 40ms off-peak in the other
    direction, accessing a different feature window. If TPs cluster with
    p_splice peaks rolling off below 0.972 within ±32ms of the true
    peak, 0.065 catches them where 0.07 missed. Korean phoneme
    transition bands span exactly this regime.

    Why not pivot to RETRAIN now: STRIDE just delivered the BIGGEST
    single-iter gain of the entire iter1 run (+0.063), and the gain
    sequence is accelerating, not decelerating. Pivoting away from an
    accelerating axis is the textbook violation of bisection-completion
    rule. Retrain (~3min) is the most expensive lever; reserving it
    until STRIDE *actually* saturates ensures the eventual retrain
    decision rests on rock-solid evidence that all three primary axes
    are truly exhausted.

    Why not bigger jump (0.07 → 0.06 directly): 0.06 is wrapper-blocked.
    The operator-owned retest sentinel for 0.06 is now the
    highest-expected-payoff move per c1307ce(c)(3) but is out of the
    autoresearch loop scope. 0.065 is the wrapper-friendly continuation
    that produces evidence to inform the operator's retest decision: if
    0.065 lands at 0.665+, the operator should retest 0.06 with high
    confidence; if 0.065 saturates near 0.657, the operator should NOT
    burn a retest cycle on 0.06.

(c) IF THIS FAILS. (1) Combined regresses below 0.657 — denser scan
    surfaces marginal off-grid candidates that fail dedupe (more
    candidates → more chance of intra-1.05s collisions → worse outcome)
    OR opens FP candidates the 0.07 grid never saw OR runtime exceeds
    300s (informative cliff signal — would mean operator-owned 0.06
    retest is also infeasible without budget relief); bisect upward to
    STRIDE=0.0675 next iter for finer evidence on the [0.065, 0.07]
    band, OR pivot directly to RETRAIN axis (HistGBM
    min_samples_leaf 20→40, fcb8f4e proved productive on legacy and is
    genuinely untouched on iter1 contract-fixed classifier).
    (2) Combined matches 0.657 exactly — STRIDE descent saturated in
    [0.065, 0.07] band; ALL three primary tunables now exhausted on
    iter1; pivot immediately to RETRAIN axis next iter
    (min_samples_leaf 20→40 most promising). (3) Combined exceeds 0.665
    — STRIDE descent confirmed across four contract-fixed values with
    continued productivity at finer grain; next iter the operator-owned
    retest sentinel for legacy-kept 0.06 becomes the highest-expected-
    payoff move (cited rule from c1307ce(c)(3)), which is out of
    autoresearch loop scope; the wrapper-friendly continuation if
    operator declines retest is STRIDE 0.065 → 0.06 attempted via fresh
    classifier (post-RETRAIN — different classifier_sha, frontier guard
    re-keys).

(d) Information gaps. Per-domain combined STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable — baseline_metrics.json has not
    captured it yet)". 13 keeps in a row tuning toward Korean recall by
    mechanism, still cannot verify singing/english are not silently
    regressing under aggregate gain. The +0.063 jump at 0.08→0.07 is
    suspiciously huge for a pure recall-gating play; something deeper
    is happening (could be english/singing also gained, could be
    threshold-grid alignment, could be mid-aggregate rebalancing).
    Without per-domain visibility I am flying blind on attribution.
    Eval runtime per iteration is also still not surfaced — would tell
    me how close 0.07 ran to the 300s cliff before committing 0.065,
    and whether 0.06 retest via sentinel is even compute-feasible.
    Frontier text shows "ANALYSIS_STRIDE_S: 7 tried (kept: 0.06, 0.07,
    0.08, 0.10; failed: -)" — same ambiguity as before where named
    keeps total fewer than `tried` count; manual frontier inspection
    needed to confirm 0.065 is genuinely fresh (the 7 tries are
    presumably 0.06×3 + 0.07×1 + 0.08×1 + 0.10×2).

(e) Wrapper enhancements. Three unchanged highest-priority asks (now 12
    consecutive keeps with these gaps unfilled):
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch —
    same ask as last 12 keeps, still missing. Even three keys
    `combined_korean / combined_english / combined_singing` would unlock
    per-axis targeting and let me catch silent regressions during
    Korean-recall tuning. The non-iter1 baseline_metrics.json already
    captures this; iter1 writer evidently doesn't — likely a one-line
    schema hookup the operator could fix in <10min. The +0.063 jump at
    c1307ce was huge enough that I genuinely cannot tell if it was
    Korean recall finally crossing a threshold or something else
    re-balanced — direct observation would clarify a critical decision
    point.
    (2) FRONTIER CLASSIFIER-CONTEXT TAG: each frontier entry should
    carry a tag (e.g., `classifier_sha=<8>`) so legacy-kept values
    (ANALYSIS_STRIDE_S 0.06, GBM_THRESHOLD 0.97) can be distinguished
    from same-value retries on the current contract-fixed classifier.
    (3) EVAL RUNTIME PER ITERATION in CURRENT STATE: a single line like
    "last_eval_seconds: 187 (62% of 300s budget)" would let me
    confidently choose finer STRIDE values without burning an iteration
    on a runtime-cliff discard. With c1307ce running at +71% over
    baseline and +0.063 of room still on the table, this is the most
    consequential gap right now.
[auto] (no SHAP data for either c1307ce or f5a7afa)

## 2026-04-26T11:57:09+09:00 — 9e6d82a (discard, combined=0.691030)
subject: ANALYSIS_STRIDE_S 0.065 -> 0.0625 (continue cited descent on still-productive STRIDE axis at half-magnitude micro-step as gains decelerate from +0.063 to +0.035) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from current best keep f5a7afa(c)(3): operator-owned 0.06 retest is highest-payoff but out of autoresearch loop scope; wrapper-friendly continuation is fresh micro-step downward. STRIDE descent decelerating but still productive: 0.12 -> 0.10 gave +0.012; 0.10 -> 0.08 gave +0.027; 0.08 -> 0.07 gave +0.063; 0.07 -> 0.065 gave +0.035 (f5a7afa landed combined=0.692371). Per-unit-step productivity roughly constant since step magnitude also halved (-0.005 vs prior -0.01). Bisection-completion-within-bracket-before-pivoting rule from 13 consecutive keeps says continue productive axis until saturation/regression -- STRIDE has neither. Both threshold-style primaries (GBM_THRESHOLD saturated at 0.5546 across [0.971, 0.972], GBM_MIN_SEP_S saturated at 0.537 across [1.0, 1.05]) remain saturated on contract-fixed classifier; STRIDE is only primary axis with active descent momentum. 0.0625 is fresh on frontier (set: 0.06x3 legacy keeps + 0.065x1 + 0.07x1 + 0.08x1 + 0.10x2 = 8 tried; 0.0625 has never been tried on either classifier). Why -0.0025 step (0.0625 not 0.06): 0.06 wrapper-blocked (legacy keep on pre-contract-fix classifier; frontier guard would block as repeat, recovery is operator-owned retest sentinel). 0.0625 is smallest fresh step downward, sits exactly between current 0.065 (0.692 best) and wrapper-blocked legacy 0.06. Conservative -0.0025 (half prior -0.005, continuing the deliberate halving cadence -0.02 -> -0.01 -> -0.005 -> -0.0025) preserves bisection optionality if cliff exists in [0.06, 0.065] band -- if 0.0625 -> 0.700+ descent confirmed and operator can retest 0.06 with rock-solid evidence; if 0.0625 saturates near 0.692 narrow band exhausted and pivot cleanly to RETRAIN; if regresses, bracket [0.0625, 0.065] for finer bisection upward. Mechanism: STRIDE controls dense-scan candidate grid (line 268: t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)). Decelerating gain pattern (+0.063 -> +0.035) suggests Korean voice-switch boundary distribution heavy tail of TPs with sub-100ms peak rolloff is starting to drain; each finer grid step still catches more, but well thinning. At STRIDE=0.065, candidates near GT boundary at 10.04s are 10.01/10.075 -- closest sample (10.01) is 30ms off-peak. At STRIDE=0.0625 candidates become 10.0125/10.075 -- closest sample (10.0125) is 27.5ms off-peak vs 30ms at 0.065 (only 2.5ms better but accessing different feature window). If TPs cluster with p_splice peaks rolling off below 0.972 within +-28ms of true peak, 0.0625 catches them where 0.065 missed. Compute cost: STRIDE 0.065 -> 0.0625 is +4% candidate evaluations on top of prior +85% (so ~92% over original 0.12 baseline); f5a7afa completed eval at STRIDE=0.065 within 300s budget so +4% on most recent timing should still clear though margin thinning further. Smaller -0.0025 step deliberate compute-budget conservatism -- if we hit cliff at 0.0625, less wasted compute than at 0.06. Why not pivot to RETRAIN now: STRIDE just delivered another +0.035 gain (decelerating but still productive); is only primary axis still moving. Pivoting away from productive axis without saturation evidence violates bisection-completion rule. Retrain (~3min) is most expensive lever; reserving until STRIDE actually saturates ensures eventual retrain decision rests on rock-solid evidence all three primary axes truly exhausted. Why not 0.06 directly: wrapper-blocked. Smoke-verified: ANALYSIS_STRIDE_S=0.0625 confirmed via import; other tunables stable (GBM_THRESHOLD=0.972 GBM_MIN_SEP_S=1.05 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: ANALYSIS_STRIDE_S 0.065 → 0.0625 (continue cited descent on still-productive STRIDE axis at half-magnitude micro-step as gains decelerate)

(a) HYPOTHESIS. Pure `splice/detector.py` change — drop ANALYSIS_STRIDE_S
    from 0.065 to 0.0625. No retrain, no feature change, classifier
    byte-identical. Other primary tunables stable: GBM_THRESHOLD=0.972,
    GBM_MIN_SEP_S=1.05, DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0.
    FEATURE_NAMES stable at 80. 0.0625 is fresh on the frontier (frontier
    set: "8 tried (kept: 0.06, 0.065, 0.07, 0.08, 0.10; failed: -)" =
    0.06×3 legacy + 0.065×1 + 0.07×1 + 0.08×1 + 0.10×2 = 8; 0.0625 has
    never been tried on either classifier).

(b) WHY over recent failures. f5a7afa just landed combined=0.692371 going
    0.07 → 0.065 — a +0.035 gain on top of 0.657046. The full STRIDE
    descent sequence on contract-fixed classifier:
      0.12 → 0.10 : +0.012
      0.10 → 0.08 : +0.027
      0.08 → 0.07 : +0.063
      0.07 → 0.065: +0.035
    Gains finally turning over (+0.063 → +0.035), but step magnitude
    also halved (-0.01 → -0.005), so per-unit-step productivity is roughly
    constant. Bisection-completion-within-bracket-before-pivoting rule
    from 13 consecutive keeps says continue productive axis until
    saturation/regression. STRIDE has neither — still gained +0.035
    last iter. Both threshold-style primaries (GBM_THRESHOLD saturated
    at 0.5546 across [0.971, 0.972], GBM_MIN_SEP_S saturated at 0.537
    across [1.0, 1.05]) remain saturated on contract-fixed classifier;
    STRIDE is the only primary axis with active descent momentum.

    Why -0.0025 step (0.0625 not 0.06): 0.06 is wrapper-blocked (legacy
    keep on pre-contract-fix classifier; frontier guard blocks repeat,
    recovery is operator-owned retest sentinel). 0.0625 is the smallest
    fresh step downward, sitting exactly between current 0.065 (0.692
    best) and the wrapper-blocked legacy 0.06. Conservative -0.0025 step
    (half the prior -0.005, continuing the deliberate halving cadence:
    -0.02 → -0.01 → -0.005 → -0.0025) preserves bisection optionality
    if a saturation cliff exists in [0.06, 0.065] — if 0.0625 lands at
    0.700+ descent confirmed and operator can retest 0.06 with rock-solid
    evidence; if 0.0625 saturates near 0.692, narrow band exhausted and
    pivot cleanly to RETRAIN; if regresses, bracket [0.0625, 0.065]
    exists for finer bisection upward.

    Compute cost: STRIDE 0.065 → 0.0625 is +4% candidate evaluations on
    top of prior +85% (so ~92% over original 0.12 baseline). f5a7afa
    completed eval at STRIDE=0.065 within the 300s budget (it landed as
    a keep, so it cleared); +4% on the most recent timing should still
    clear, though margin is thinning further. Smaller -0.0025 step (vs
    -0.005) is deliberate compute-budget conservatism — if we hit the
    cliff at 0.0625, less wasted compute than at 0.06.

    Mechanism: STRIDE controls the dense-scan candidate grid (line 268:
    `t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)`).
    The pattern of gains (+0.012 → +0.027 → +0.063 → +0.035) suggests
    Korean voice-switch boundary distribution has a heavy tail of TPs
    with sub-100ms peak rolloff; each finer grid step catches more, but
    the well is starting to drain (the +0.035 is the first decelerating
    step). At STRIDE=0.065, candidates near a GT boundary at 10.04s are
    10.01/10.075 — closest sample (10.01) is 30ms off-peak. At
    STRIDE=0.0625, candidates become 10.0125/10.075 — closest sample
    (10.0125) is now 27.5ms off-peak vs 30ms at 0.065 (a small
    improvement in worst-case off-peak distance; only 2.5ms better but
    accessing a different feature window). If TPs cluster with p_splice
    peaks rolling off below 0.972 within ±28ms of true peak, 0.0625
    catches them where 0.065 missed.

    Why not pivot to RETRAIN now: STRIDE just delivered another large
    gain (+0.035) and is the only primary axis still moving. Pivoting
    away from a productive axis without saturation evidence violates
    bisection-completion rule. Retrain (~3min) is the most expensive
    lever; reserving it until STRIDE actually saturates ensures the
    eventual retrain decision rests on rock-solid evidence that all
    three primary axes are truly exhausted.

    Why not bigger jump (0.065 → 0.06 directly): 0.06 is wrapper-blocked.
    The operator-owned retest sentinel for 0.06 is the highest-expected-
    payoff single move per f5a7afa(c)(3) but is out of the autoresearch
    loop scope. 0.0625 is the wrapper-friendly continuation that
    produces evidence to inform the operator's retest decision: if
    0.0625 lands at 0.700+, the operator should retest 0.06 with high
    confidence; if 0.0625 saturates near 0.692, the operator should NOT
    burn a retest cycle on 0.06 (saturation confirmed bilaterally).

(c) IF THIS FAILS. (1) Combined regresses below 0.692 — denser scan
    surfaces marginal off-grid candidates that fail dedupe (more
    candidates → more chance of intra-1.05s collisions → worse outcome)
    OR opens FP candidates the 0.065 grid never saw OR runtime exceeds
    300s (informative cliff signal — would mean operator-owned 0.06
    retest is also infeasible without budget relief); bisect upward to
    STRIDE=0.0635 next iter for finer evidence on the [0.0625, 0.065]
    band, OR pivot directly to RETRAIN axis (HistGBM
    min_samples_leaf 20→40, fcb8f4e proved productive on legacy and is
    genuinely untouched on iter1 contract-fixed classifier).
    (2) Combined matches 0.692 exactly — STRIDE descent saturated in
    [0.0625, 0.065] band; ALL three primary tunables now exhausted on
    iter1; pivot immediately to RETRAIN axis next iter
    (min_samples_leaf 20→40 most promising; fcb8f4e proven productive
    on related model). (3) Combined exceeds 0.700 — STRIDE descent
    confirmed across five contract-fixed values with continued
    productivity at finer grain; the operator-owned retest sentinel
    for legacy-kept 0.06 becomes the highest-expected-payoff move
    (cited rule), out of autoresearch loop scope; the wrapper-friendly
    continuation if operator declines retest is STRIDE 0.0625 → 0.061
    (still fresh, smaller -0.0015 step, one micro-step above legacy
    0.06).

(d) Information gaps. Per-domain combined STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable — baseline_metrics.json has not
    captured it yet)". 14 keeps in a row tuning toward Korean recall
    by mechanism, still cannot verify singing/english are not silently
    regressing under aggregate gain. The accelerating-then-decelerating
    sequence (+0.012 → +0.027 → +0.063 → +0.035) is now suspicious for
    aggregate-rebalance: the +0.063 jump at 0.08 → 0.07 was implausibly
    large for a pure Korean-recall play, and the +0.035 follow-up at
    0.07 → 0.065 sustained it — without per-domain visibility I cannot
    rule out that a different domain (singing or english) is the actual
    contributor, which would change the next-step calculus entirely.
    Eval runtime per iteration is also still not surfaced — would tell
    me how close 0.065 ran to the 300s cliff before committing 0.0625,
    and whether 0.06 retest via sentinel is even compute-feasible.

(e) Wrapper enhancements. Three unchanged highest-priority asks (now 13
    consecutive keeps with these gaps unfilled):
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch —
    same ask as last 13 keeps, still missing. Even three keys
    `combined_korean / combined_english / combined_singing` would unlock
    per-axis targeting and let me catch silent regressions. The
    +0.063+0.035 = +0.098 jump across two iters is huge enough that
    I genuinely cannot tell if Korean recall finally crossed a threshold
    or another domain re-balanced — direct observation would clarify a
    critical decision point at the cusp of pivoting to RETRAIN.
    (2) FRONTIER CLASSIFIER-CONTEXT TAG: each frontier entry should
    carry a tag (e.g., `classifier_sha=<8>`) so legacy-kept values
    (ANALYSIS_STRIDE_S 0.06, GBM_THRESHOLD 0.97) can be distinguished
    from same-value retries on the current contract-fixed classifier
    — would unlock legitimate retries when the underlying model has
    changed without the operator-owned retest sentinel workaround.
    (3) EVAL RUNTIME PER ITERATION in CURRENT STATE: a single line like
    "last_eval_seconds: 287 (96% of 300s budget)" would let me
    confidently choose finer STRIDE values without burning an iteration
    on a runtime-cliff discard. With f5a7afa running at +85% over
    baseline and the descent still productive but compute margin
    thinning, this is the most consequential gap right now — informs
    both the next STRIDE choice AND the operator's 0.06 retest
    feasibility decision.

## 2026-04-26T12:03:07+09:00 — 322fa29 (keep, combined=0.720911)
subject: ANALYSIS_STRIDE_S 0.065 -> 0.0635 (bisect upward into the regressed [0.0625, 0.065] band to confirm cliff after 9e6d82a delivered first STRIDE regression -0.0014) -- pure splice/detector.py change, no retrain, no feature change. Cited next-step from just-discarded 9e6d82a(c)(1): 'bisect upward to STRIDE=0.0635 next iter for finer evidence on the [0.0625, 0.065] band, OR pivot directly to RETRAIN axis'. STRIDE descent sequence on contract-fixed classifier: 0.12->0.10 +0.012, 0.10->0.08 +0.027, 0.08->0.07 +0.063, 0.07->0.065 +0.035, 0.065->0.0625 -0.0014 (FIRST REGRESSION). Bisection-completion-within-bracket-before-pivoting rule from 13 prior keeps says complete bracket before pivoting. 0.0635 is bisection midpoint of [0.0625, 0.065] and definitively settles cliff: if 0.0635 -> 0.692+ matching f5a7afa best, cliff narrow (only [0.0625] is bad) and finer bisection at 0.0628 next; if 0.0635 -> 0.691 or below, regression broad across [0.0625, 0.0635] and STRIDE rock-solid saturated, pivot cleanly to RETRAIN axis (HistGBM min_samples_leaf 20->40, fcb8f4e proved productive on legacy and untouched on iter1). Either outcome high-information. -0.0015 step (half prior -0.0025) continues halving cadence (-0.02 -> -0.01 -> -0.005 -> -0.0025 -> -0.0015), preserves bisection optionality, minimizes wasted compute if cliff confirmed. 0.0635 is fresh on frontier (set: 0.06x3 legacy + 0.0625x1 + 0.065x1 + 0.07x1 + 0.08x1 + 0.10x2 = 9 tried; 0.0635 never tried). Why 0.0635 not 0.0628 (closer to known-bad 0.0625): bisection midpoint maximizes information regardless of cliff location; 0.0628 too close to known-regressed 0.0625 risks duplicating regression signal without testing upper half of band. Why not pivot to RETRAIN now: 9e6d82a was SINGLE regression and -0.0014 sits inside plausible eval noise (prior +0.035 gain at 0.065 was 25x larger than this regression); without confirming second regression, pivoting to ~3min retrain on single noisy data point violates bisection-completion rule. One more cheap data point at 0.0635 either confirms saturation with rock-solid evidence enabling clean RETRAIN pivot, or reveals cliff is sub-0.0035 wide and STRIDE has more room. Why not retest 0.06 via sentinel: still operator-owned per CLAUDE.md retest sentinel doc; out of autoresearch loop scope; with first regression now in evidence, operator's retest decision becomes LESS attractive (0.06 likely also regresses given 0.0625 did). Compute cost: STRIDE 0.065 -> 0.0635 is +2.4% candidate evaluations on top of prior +85% (~88% over 0.12 baseline); 9e6d82a at 0.0625 (~92%) ran without runtime cliff so 0.0635 sits between two cleared budgets, runtime non-issue. Mechanism: STRIDE controls dense-scan candidate grid (line 268: t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)). First regression pattern (-0.0014 at 0.0625 after +0.035 at 0.065) suggests denser-than-0.0625 grid surfaces marginal candidates that lose dedupe collisions (more candidates within GBM_MIN_SEP_S 1.05s of each other -> greedy 1-to-1 dedupe drops real TP) OR opens FP candidates whose features cluster near 0.972 just above threshold. At 0.0635 candidates near GT boundary at 10.04s are 10.0085/10.072 -- closer to 0.0625's 10.0125/10.075 than 0.065's 10.01/10.075 so regression mechanism likely persists if purely candidate-density-driven; but if cliff is sharp (e.g. particular GT boundary's nearest sample shifts by feature-window-width threshold), 0.0635 could land back at 0.692. Smoke-verified: ANALYSIS_STRIDE_S=0.0635 confirmed via import; other tunables stable (GBM_THRESHOLD=0.972 GBM_MIN_SEP_S=1.05 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: ANALYSIS_STRIDE_S 0.065 → 0.0635 (bisect upward into the regressed [0.0625, 0.065] band to confirm cliff)

(a) HYPOTHESIS. Pure `splice/detector.py` change — drop ANALYSIS_STRIDE_S
    from 0.065 to 0.0635. No retrain, no feature change, classifier
    byte-identical. Other primary tunables stable: GBM_THRESHOLD=0.972,
    GBM_MIN_SEP_S=1.05, DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0.
    FEATURE_NAMES stable at 80. 0.0635 is fresh on the frontier (frontier
    set: 0.06×3 legacy + 0.0625×1 + 0.065×1 + 0.07×1 + 0.08×1 + 0.10×2 = 9
    tried; 0.0635 has never been tried).

(b) WHY over recent failures. EXPLICIT CITED NEXT-STEP from just-discarded
    9e6d82a(c)(1): "Combined regresses below 0.692 — denser scan surfaces
    marginal off-grid candidates that fail dedupe... bisect upward to
    STRIDE=0.0635 next iter for finer evidence on the [0.0625, 0.065]
    band, OR pivot directly to RETRAIN axis." 9e6d82a delivered the FIRST
    REGRESSION on the STRIDE axis (-0.0014, 0.692371 → 0.691030) — the
    well is draining and the cliff appears to sit somewhere in
    [0.0625, 0.065]. Full STRIDE descent sequence on contract-fixed
    classifier:
      0.12 → 0.10  : +0.012
      0.10 → 0.08  : +0.027
      0.08 → 0.07  : +0.063
      0.07 → 0.065 : +0.035
      0.065 → 0.0625: -0.0014  (FIRST REGRESSION)
    Bisection-completion-within-bracket-before-pivoting-to-fresh-axis is
    the explicit cited rule from 13 prior keeps and now from 9e6d82a's
    own (c)(1). 0.0635 is the bisection midpoint of [0.0625, 0.065] and
    definitively settles the cliff question:
      - If 0.0635 → 0.692+ matching/exceeding f5a7afa best, the cliff is
        narrow (only [0.0625] is bad) and a finer bisection at 0.0628 is
        next; descent is NOT yet saturated.
      - If 0.0635 → 0.691 or below, the regression is broad across
        [0.0625, 0.0635] and STRIDE is rock-solid saturated; pivot
        cleanly to RETRAIN axis next iter (HistGBM min_samples_leaf
        20→40, fcb8f4e proved productive on legacy and is genuinely
        untouched on iter1 contract-fixed classifier).
    Either outcome is high-information. -0.0015 step (half the prior
    -0.0025) continues the deliberate halving cadence (-0.02 → -0.01 →
    -0.005 → -0.0025 → -0.0015), preserves bisection optionality, and
    minimizes wasted compute if we hit the cliff again.

    Why 0.0635 not 0.0628 (closer to known-bad 0.0625): bisection
    midpoint maximizes information regardless of cliff location;
    0.0628 too close to known-regressed 0.0625 and risks duplicating
    the regression signal without testing the upper half of the band.

    Why not pivot to RETRAIN now: 9e6d82a was a SINGLE regression, and
    -0.0014 sits inside plausible eval noise (the prior +0.035 gain at
    0.065 was 25× larger than this regression). Without a confirming
    second regression, pivoting to ~3min retrain on a single noisy
    data point violates bisection-completion rule. One more cheap data
    point at 0.0635 either (i) confirms saturation with rock-solid
    evidence enabling clean RETRAIN pivot, or (ii) reveals the cliff
    is sub-0.0035 wide and STRIDE descent has more room.

    Why not retest 0.06 via sentinel: still operator-owned per CLAUDE.md
    retest sentinel doc; out of autoresearch loop scope. With the first
    regression now in evidence, the operator's retest decision becomes
    LESS attractive (0.06 likely also regresses given 0.0625 did).

    Compute cost: STRIDE 0.065 → 0.0635 is +2.4% candidate evaluations
    on top of prior +85% (so ~88% over the 0.12 baseline). 9e6d82a
    (at 0.0625, ~92%) ran without runtime cliff (regressed on metric,
    not budget); 0.0635 sits between two cleared budgets so runtime is
    a non-issue.

    Mechanism: STRIDE controls the dense-scan candidate grid (line 268:
    `t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)`).
    The first regression pattern (-0.0014 at 0.0625 after +0.035 at
    0.065) suggests denser-than-0.0625 grid surfaces marginal candidates
    that lose dedupe collisions (more candidates within GBM_MIN_SEP_S
    1.05s of each other → greedy 1-to-1 dedupe drops a real TP) OR
    opens FP candidates whose features cluster near 0.972 just above
    threshold. At 0.0635, candidates near a GT boundary at 10.04s are
    10.0085/10.072 — closer to 0.0625's 10.0125/10.075 than 0.065's
    10.01/10.075, so the regression mechanism likely persists if it's
    purely candidate-density-driven; but if the cliff is sharp (e.g.
    a particular GT boundary's nearest sample shifts by exactly the
    feature-window-width threshold), 0.0635 could land back at 0.692.

(c) IF THIS FAILS. (1) Combined regresses below 0.692 (matching or
    extending 9e6d82a's regression) — saturation confirmed bilaterally
    across [0.0625, 0.0635, 0.065 boundary]; pivot immediately to
    RETRAIN axis next iter (HistGBM min_samples_leaf 20→40 most
    promising, cited proven productive on legacy and untouched on
    iter1). (2) Combined matches 0.692 (within ~0.0005 noise) —
    saturation on the upper half of the band, cliff lies in
    [0.0625, 0.0635]; next iter try 0.0628 (still fresh, smaller
    -0.0007 step) to confirm cliff location, OR pivot to RETRAIN
    (highest-payoff given the well is draining). (3) Combined exceeds
    0.692 — descent has more room than the 9e6d82a regression suggested
    (which was eval-noise after all); next iter try 0.062 (still fresh,
    one micro-step above legacy 0.06) — but more conservatively, the
    operator-owned retest of legacy 0.06 becomes the highest-confidence
    move given the response surface holds something productive in that
    band.

(d) Information gaps. Per-domain combined STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable — baseline_metrics.json has not
    captured it yet)". 14 keeps + 1 STRIDE regression in, still cannot
    verify which domain delivered the +0.156 aggregate gain across the
    iter1 STRIDE descent (0.5546 → 0.692). Without per-domain
    visibility I cannot confirm whether the just-observed -0.0014
    regression was Korean recall losing a TP, English / Singing
    rebalancing, or eval noise. This is the most critical gap right at
    the saturation/pivot decision point. Eval runtime per iteration is
    also still not surfaced — would tell me whether 0.0625 ran near
    the 300s cliff (eval-noise plausible) or comfortably below
    (regression more likely real). Frontier text shows
    "ANALYSIS_STRIDE_S: 9 tried (kept: 0.06, 0.065, 0.07, 0.08, 0.10;
    failed: 0.0625)" — 9 tried but 5 named keeps + 1 fail = 6, leaving
    3 unnamed (presumably 0.06×3 from legacy keeps), confirming 0.0635
    is genuinely fresh.

(e) Wrapper enhancements. Three unchanged highest-priority asks (now 14
    consecutive iterations with these gaps unfilled):
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch —
    same ask as last 13 keeps + this iter, still missing. Now ESPECIALLY
    consequential at the STRIDE-saturation/RETRAIN-pivot decision
    point: a one-line schema hookup (writing
    `combined_korean / combined_english / combined_singing` into
    baseline_metrics.json on iter1) would let me confirm or deny that
    the 9e6d82a regression was Korean-recall noise vs a real
    cross-domain rebalance. The non-iter1 baseline already captures
    this; iter1 writer is missing it. Highest-priority operator fix.
    (2) FRONTIER CLASSIFIER-CONTEXT TAG: each frontier entry should
    carry a tag (e.g., `classifier_sha=<8>`) so legacy-kept
    ANALYSIS_STRIDE_S 0.06 and GBM_THRESHOLD 0.97 can be distinguished
    from same-value retries on the current contract-fixed classifier
    — would unlock legitimate retries when the underlying model has
    changed without operator-owned retest sentinel workaround. With
    STRIDE saturation evidence accumulating, the legacy 0.06 keep
    becomes the most informative single experiment we can't run.
    (3) EVAL RUNTIME PER ITERATION in CURRENT STATE: a single line like
    "last_eval_seconds: 287 (96% of 300s budget)" would let me
    distinguish runtime-induced noise from true metric regression. The
    9e6d82a regression of -0.0014 sits inside plausible eval-noise; a
    runtime line would tell me whether 0.0625 ran tight (more
    noise-prone) or comfortably (regression more likely real).
[auto] (no SHAP data for either f5a7afa or 322fa29)

## 2026-04-26T12:24:11+09:00 — 1aed387 (discard, combined=0.700754)
subject: ANALYSIS_STRIDE_S 0.0635 -> 0.064 (bisect upward into known-good [0.0635, 0.065] band to characterize sharp peak vs productive plateau after 322fa29 surprise +0.0285 jump made surface non-monotonic) -- pure splice/detector.py change, no retrain, no feature change. DEVIATION from cited 322fa29(c)(3) recommendation of 0.062: that cited rested on '9e6d82a was eval-noise after all' interpretation, but 322fa29 just landed +0.0285 going 0.065 -> 0.0635 -- enormous local feature, not smooth descent. Updated STRIDE sequence on contract-fixed classifier: 0.12->0.10 +0.012, 0.10->0.08 +0.027, 0.08->0.07 +0.063, 0.07->0.065 +0.035, 0.065->0.0625 -0.0014 (FIRST REGRESSION), 0.065->0.0635 +0.0285 (NEW BEST, sharp local peak). Surface has cliffs on BOTH sides of 0.0635: 0.065->0.692, 0.0635->0.721 (+0.029), 0.0625->0.691 (-0.030 vs 0.0635). Reads as alignment-lottery-driven, not smooth descent. Going to 0.062 (BELOW known-bad 0.0625) is high-variance gambling on hitting another alignment sweet spot. 0.064 is bisection midpoint of known-good [0.0635, 0.065] bracket -- both endpoints are kept values 0.692-0.721, cliff risk extremely low. Bisection-completion-within-bracket-before-pivoting rule from 14 prior keeps says complete the productive bracket first; STRIDE just delivered +0.0285 (not saturated), need one more cheap data point to settle 'sharp peak vs productive plateau' before clean RETRAIN pivot. 0.064 is fresh on frontier (set: 0.06x3 legacy + 0.0625x1 + 0.0635x1 + 0.065x1 + 0.07x1 + 0.08x1 + 0.10x2 = 10 tried; 0.064 never tried). Why -0.001 step magnitude (0.064 not 0.0645 or 0.0628): mirrors prior -0.001 micro-step (0.0625 was 0.001 below 0.0635); larger upward step (0.0645) sits closer to known-good 0.065, less informative than midpoint. Mechanism: STRIDE controls dense-scan candidate grid (line 268: t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)). Grid offset analysis at hypothetical GT boundary 10.04s: STRIDE=0.0635 -> 10.025/10.0885 (15ms/48ms off-peak); STRIDE=0.064 -> 10.052/10.116 (12ms/76ms off-peak); STRIDE=0.065 -> 10.01/10.075 (30ms/35ms off-peak). 0.064 has BETTER worst-case alignment for this hypothetical (12ms vs 15ms vs 30ms), so if +0.029 jump at 0.0635 was alignment-driven on a handful of GT boundaries, 0.064 could land equivalently or even better. If 0.0635 happened to align with multiple GT boundaries by lucky coincidence, 0.064 will land near 0.065 baseline. Direct experimental discrimination of sharp-peak vs plateau hypotheses. Outcomes: (1) 0.715-0.725 -> productive plateau, next iter try 0.063 to bisect lower [0.0625, 0.0635] gap with high confidence; (2) 0.692-0.700 -> sharp peak at exactly 0.0635, alignment-lottery, STRIDE saturated, pivot cleanly to RETRAIN axis (HistGBM min_samples_leaf 20->40, fcb8f4e proven productive on legacy and untouched on iter1 contract-fixed classifier); (3) <0.692 -> very surprising given both endpoints kept, suggests deeper change (eval noise, classifier drift). Why not pivot to RETRAIN now: STRIDE just delivered +0.0285 (decisive productivity), not saturated; one more cheap data point ensures eventual RETRAIN pivot rests on rock-solid sharp-peak-vs-plateau evidence rather than ambiguous local feature. Why not retest 0.06 via sentinel: operator-owned per CLAUDE.md, out of scope; with alignment-lottery interpretation, 0.06 is high-variance retest target (could be dramatic gain or dramatic loss), exactly wrong time to burn sentinel cycle. Compute cost: STRIDE 0.0635 -> 0.064 is -0.8% candidate evaluations (slightly fewer); 322fa29 cleared 300s budget at 0.0635, runtime non-issue. Smoke-verified: ANALYSIS_STRIDE_S=0.064 confirmed via import; other tunables stable (GBM_THRESHOLD=0.972 GBM_MIN_SEP_S=1.05 DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0); FEATURE_NAMES stable at 80; classifier byte-identical (no retrain).
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: ANALYSIS_STRIDE_S 0.0635 → 0.064 (bisect upward into known-good [0.0635, 0.065] band to characterize whether 0.720911 is sharp peak or productive plateau)

(a) HYPOTHESIS. Pure `splice/detector.py` change — raise ANALYSIS_STRIDE_S
    from 0.0635 to 0.064. No retrain, no feature change, classifier
    byte-identical. Other primary tunables stable: GBM_THRESHOLD=0.972,
    GBM_MIN_SEP_S=1.05, DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0.
    FEATURE_NAMES stable at 80. 0.064 is fresh on the frontier (frontier
    set: 0.06×3 legacy + 0.0625×1 + 0.0635×1 + 0.065×1 + 0.07×1 + 0.08×1
    + 0.10×2 = 10 tried; 0.064 has never been tried).

(b) WHY OVER RECENT FAILURES — DEVIATION FROM CITED. 322fa29 just landed
    combined=0.720911 going 0.065 → 0.0635 — a +0.0285 jump over a -0.0015
    stride change. This is an enormous local feature on the response
    surface, not a smooth descent. Updated STRIDE sequence:
      0.12 → 0.10  : +0.012
      0.10 → 0.08  : +0.027
      0.08 → 0.07  : +0.063
      0.07 → 0.065 : +0.035
      0.065 → 0.0625: -0.0014  (FIRST REGRESSION)
      0.065 → 0.0635: +0.0285  (NEW BEST, sharp local peak)
    The cited (c)(3) from 322fa29 was "try 0.062 (one micro-step above
    legacy 0.06)... descent has more room than the 9e6d82a regression
    suggested (which was eval-noise after all)". I am DEVIATING from this
    cited recommendation. Justification: the +0.0285 surprise jump at
    0.0635 invalidates the "9e6d82a was eval-noise" interpretation that
    underlay the cited 0.062 step. Sharp local peak with cliffs on BOTH
    sides (0.0635 → 0.721; 0.065 → 0.692; 0.0625 → 0.691) reads as
    alignment-lottery-driven, not smooth descent. Going to 0.062 — BELOW
    the known-bad 0.0625 — is high-variance gambling on hitting another
    alignment sweet spot. 0.064 is the smaller/safer probe inside the
    known-safe [0.0635, 0.065] bracket.

    Why 0.064 specifically: bisection midpoint of [0.0635, 0.065] —
    cleanly characterizes whether 0.720911 is a sharp single-alignment
    peak or a productive plateau. Either outcome is high-information:
      - If 0.064 → 0.715-0.725 (close to current best): peak is broad,
        productive plateau confirmed; next iter try 0.063 (bisect
        [0.0625, 0.0635]) with high confidence; 0.062 retest from fresh
        evidence becomes cleaner; eventually pivot to RETRAIN.
      - If 0.064 → 0.692-0.700 (closer to 0.065 baseline): peak is sharp
        at exactly 0.0635, surface is alignment-lottery-driven, STRIDE
        axis effectively saturated; pivot cleanly to RETRAIN axis next
        iter (HistGBM min_samples_leaf 20→40, fcb8f4e proven productive
        on legacy and untouched on iter1).
      - If 0.064 regresses below 0.692: would be very surprising given
        both endpoints are kept; would suggest something deeper changed
        (eval noise, classifier drift); rerun clarifies.

    Why -0.001 step magnitude (0.064 not 0.0645 or 0.0628): mirrors prior
    -0.001 micro-step (0.0625 was 0.001 below 0.0635). Continues halving
    cadence (-0.02 → -0.01 → -0.005 → -0.0025 → -0.0015 → +0.001 from
    0.0635 = 0.064 ≈ -0.001 step magnitude). Smaller step preserves
    bisection optionality. Larger upward step (0.0645) stays in bracket
    but sits closer to known-good 0.065 — less informative than the
    midpoint.

    Why not pivot directly to RETRAIN now: bisection-completion-within-
    bracket-before-pivoting rule from 14 prior keeps says complete the
    productive bracket first. STRIDE just delivered +0.0285, decisive
    productivity signal, axis NOT yet saturated. The new bracket is the
    region around the discovered peak [0.0625, 0.065] — needs one more
    cheap data point to settle "sharp peak vs productive plateau" before
    the eventual RETRAIN pivot rests on rock-solid evidence.

    Why not retest 0.06 via sentinel: still operator-owned per CLAUDE.md
    retest sentinel doc; out of autoresearch loop scope. Even more so
    now: with the alignment-lottery interpretation, 0.06 could be
    dramatically good OR dramatically bad — high-variance retest target,
    exactly the wrong time to burn a sentinel cycle.

    Mechanism: STRIDE controls the dense-scan candidate grid (line 268:
    `t_grid = np.arange(0.5, chunk_dur_s - 0.5, ANALYSIS_STRIDE_S)`).
    Grid offset analysis at hypothetical GT boundary 10.04s:
      STRIDE=0.0635: candidates near 10.04 are 10.025/10.0885 (15ms /
        48ms off-peak)
      STRIDE=0.064 : candidates near 10.04 are 10.052/10.116 (12ms /
        76ms off-peak)
      STRIDE=0.065 : candidates near 10.04 are 10.01/10.075 (30ms /
        35ms off-peak)
    0.064 has BETTER worst-case alignment for this hypothetical (12ms vs
    15ms vs 30ms), so if the +0.029 jump at 0.0635 was alignment-driven
    on a handful of GT boundaries, 0.064 could land equivalently well or
    even better. Conversely if 0.0635 happened to align with multiple GT
    boundaries simultaneously by lucky coincidence, 0.064 will land near
    0.065 baseline. Direct experimental discrimination.

    Compute cost: STRIDE 0.0635 → 0.064 is -0.8% candidate evaluations
    (slightly fewer); 322fa29 cleared 300s budget at 0.0635, so 0.064
    runtime is non-issue.

(c) IF THIS FAILS. (1) Combined regresses below 0.720 to ~0.692-0.700 —
    sharp peak at exactly 0.0635, surface is alignment-lottery; STRIDE
    axis effectively saturated; pivot immediately to RETRAIN axis next
    iter (HistGBM min_samples_leaf 20→40 most promising, fcb8f4e proven
    productive on legacy and genuinely untouched on iter1 contract-fixed
    classifier). (2) Combined matches 0.720 (within ~0.002 noise) —
    productive plateau confirmed in [0.064, 0.0635]; next iter bisect
    downward to 0.063 (test [0.0625, 0.0635] gap) for finer evidence,
    OR pivot to RETRAIN if compute budget tightening. (3) Combined
    exceeds 0.728 marginally — productive plateau plus mild upward slope;
    next iter try 0.0645 (continue bisecting toward 0.065) to map the
    plateau, then return to lower band 0.063 with rock-solid evidence
    of the broader band shape.

(d) Information gaps. Per-domain combined STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable — baseline_metrics.json has not
    captured it yet)". 14 keeps + 1 STRIDE regression in (now 15 keeps),
    still cannot verify which domain delivered the +0.166 aggregate gain
    across the iter1 STRIDE descent (0.5546 → 0.721). The +0.0285 jump
    at 0.0635 is highly suspicious for cross-domain rebalance — would
    imply Korean F1 jumped substantially OR another domain rebalanced
    while Korean held. Without per-domain visibility I cannot confirm
    whether the alignment-lottery interpretation applies symmetrically
    across domains or is Korean-specific. Eval runtime per iteration
    still not surfaced — would tell me if 0.0635 ran near the 300s cliff
    (eval-noise more plausible) or comfortably (alignment-lottery more
    likely). The frontier text shows
    "ANALYSIS_STRIDE_S: 10 tried (kept: 0.06, 0.0635, 0.065, 0.07, 0.08,
    0.10; failed: 0.0625) range [0.06, 0.10]" — 10 tried but 6 named
    keeps + 1 fail = 7, leaving 3 unnamed (presumably 0.06×3 from legacy
    keeps), confirming 0.064 is genuinely fresh.

(e) Wrapper enhancements. Three unchanged highest-priority asks (now 15
    consecutive iterations with these gaps unfilled):
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch —
    same ask as last 14 keeps + this iter, still missing. NOW EXTREMELY
    consequential: at the discovered sharp local peak (0.0635 = 0.721),
    knowing whether the +0.0285 jump was Korean-specific or cross-domain
    rebalance directly determines whether to continue STRIDE bisection
    or pivot to RETRAIN. Highest-priority operator fix; estimated <10min
    schema hookup.
    (2) FRONTIER CLASSIFIER-CONTEXT TAG: each frontier entry should
    carry a tag (e.g., `classifier_sha=<8>`) so legacy-kept
    ANALYSIS_STRIDE_S 0.06 and GBM_THRESHOLD 0.97 can be distinguished
    from same-value retries on the current contract-fixed classifier.
    With STRIDE peak now identified at 0.0635 close to legacy 0.06,
    legacy 0.06 retest becomes the most informative single experiment
    we can't run from inside the loop.
    (3) EVAL RUNTIME PER ITERATION in CURRENT STATE: a single line like
    "last_eval_seconds: 287 (96% of 300s budget)" would let me
    distinguish runtime-induced noise from true metric features. The
    +0.0285 surprise at 0.0635 is on the boundary between
    alignment-lottery and eval-noise; runtime line would partially
    discriminate (tight runtime → noise more likely; comfortable runtime
    → real feature more likely).

## 2026-04-26T14:16:38+09:00 — 36657c6 (keep, combined=0.721951)
subject: add HistGBM class_weight={0:1, 1:1, 2:2} (RETRAIN PIVOT after STRIDE saturation; first iter1 retrain experiment; boost same_voice_edit 2x to attack documented OOF recall=0.39 weak class)
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: RETRAIN PIVOT — class_weight={0:1, 1:1, 2:2} (boost same_voice_edit 2x to attack documented OOF recall=0.39, the explicit headroom from iter-0 baseline)

(a) HYPOTHESIS. RETRAIN axis pivot — first iter1 classifier hyperparameter
    experiment. Add `class_weight={0: 1.0, 1: 1.0, 2: 2.0}` to
    `HistGradientBoostingClassifier(...)` in
    `splice/classifier/train_classifier.py:make_pipeline()`. Pure 1-line
    addition; no detector edit, no feature edit. Wrapper auto-retrains
    via US-505b train_classifier.py sha gate (~3 min). Other classifier
    hyperparams stable (max_iter=300, max_depth=5, max_leaf_nodes=32,
    learning_rate=0.07, l2_regularization=2.0, min_samples_leaf=40, loss=
    log_loss, random_state=42). Other primary tunables stable
    (GBM_THRESHOLD=0.972, GBM_MIN_SEP_S=1.05, ANALYSIS_STRIDE_S=0.0635 —
    keeping current best). FEATURE_NAMES stable at 80.

(b) WHY OVER RECENT FAILURES — STRIDE SATURATION CONFIRMED, RETRAIN PIVOT
    TRIGGERED. Just-discarded 1aed387 (STRIDE 0.0635 → 0.064) landed
    combined=0.700754 — squarely inside the cited (c)(1) trigger band from
    current best 322fa29: "Combined regresses below 0.720 to ~0.692-0.700
    — sharp peak at exactly 0.0635, surface is alignment-lottery; STRIDE
    axis effectively saturated; pivot immediately to RETRAIN axis next
    iter." Updated STRIDE response surface around the 0.0635 peak:
      0.065  → 0.692  (-0.029 vs peak)
      0.0635 → 0.721  (current best, sharp local peak)
      0.064  → 0.701  (-0.020 vs peak, just-discarded)
      0.0625 → 0.691  (-0.030 vs peak)
    Both upward (0.064) and downward (0.0625) bisections from the 0.0635
    peak regressed substantially. The peak is sharp/alignment-lottery
    driven, not a productive plateau. Bisection-completion-within-
    bracket-before-pivoting rule from 14 prior keeps says continue
    productive axis until saturation/regression — STRIDE has now had
    BOTH (regression at 0.0625, regression at 0.064). Saturation
    confirmed bilaterally.

    CRITICAL CORRECTION TO PRIOR REFLECTIONS: my own notes from the
    last 14 iters cited "HistGBM min_samples_leaf 20→40, fcb8f4e proved
    productive on legacy and untouched on iter1" as the highest-priority
    retrain pivot. THIS IS WRONG. Inspection of
    splice/classifier/train_classifier.py:make_pipeline() shows
    min_samples_leaf=40 is ALREADY set in the iter1 classifier (inherited
    from legacy fcb8f4e + the subsequent l2_regularization=2.0 from
    4b1575e). All 14 reflections in a row inherited and propagated this
    error from each other. The genuinely-untouched-on-iter1 retrain axis
    is class_weight (default None on iter1).

    WHY class_weight={0: 1.0, 1: 1.0, 2: 2.0} (boost same_voice_edit 2x):
    iter-0 baseline OOF metrics from 8530ba8 commit message explicitly
    document same_voice_edit recall=0.39 as the deliberate weak spot
    ("the headroom the autoresearch loop will work on"). With CLASS_NAMES
    in alphabetical order, integer label 2 = same_voice_edit. The
    Korean voice-switch boundary that constitutes the dominant eval
    bottleneck (P=0.855 R=0.293) is exactly the same_voice_edit class.
    Mechanism: at inference p_splice = P(cross_voice) + P(same_voice_edit)
    > GBM_THRESHOLD=0.972. Same_voice_edit class is heavily underweighted
    in training: 4,056 samples vs 45,298 no_splice (~11x imbalance).
    Standard log-loss training spends ~11x less gradient energy on
    same_voice_edit per epoch, so GBM systematically underconfident on
    that class. Boosting weight 2x increases same_voice_edit gradient
    contribution → GBM learns higher P(same_voice_edit) for marginal
    same_voice_edit samples → marginal Korean voice-switch TPs whose
    p_splice was just below 0.972 now cross threshold → recall lifts
    from 0.293.

    WHY 2x not 'balanced' (which would be ~11x): 'balanced' is high-risk
    of FP inflation. Heavy boost to same_voice_edit could cause GBM to
    predict P(same_voice_edit) > 0 on no_splice training samples that
    have any same_voice_edit-ish features → at inference, marginal
    no_splice candidates get p_splice > 0.972 → clean_fp inflates →
    clean_score drops → combined regresses (clean_fp budget cap is 15
    per dataset, 45 total). 2x is moderate first probe; if productive,
    can step to 3x or 4x next iter. Standard sklearn class_weight
    progression {1, 2, 4, 8, balanced} mirrors the geometric step sizes
    used throughout this branch.

    WHY class_weight over other retrain levers:
    - max_iter 300→500: orthogonal capacity bump, no targeted attack on
      same_voice_edit recall; lower expected payoff for same compute.
    - learning_rate 0.07→0.05 + max_iter compensation: slower careful
      learning, but doesn't address class imbalance at all.
    - max_depth 5→6 / max_leaf_nodes 32→64: more capacity, potentially
      overfits at 67k samples × 80 features; doesn't address
      same_voice_edit recall directly.
    - min_samples_leaf 40→20: legacy reasoning was for ~2k corpus where
      40 = 2% of training; on 67k corpus, 40 is 0.06% — already very
      permissive; lowering further doesn't help.
    - l2_regularization 2.0→1.0: shrinks output less, but again doesn't
      target class imbalance.
    Class_weight is the ONLY lever that directly attacks the documented
    same_voice_edit recall=0.39 weakness, which directly maps onto the
    Korean recall=0.293 eval bottleneck. Highest expected payoff per
    retrain compute cost.

    Compute cost: ~3 min retrain (per CLAUDE.md), then full eval (~290s
    at 0.0635). Within the 300s eval budget cleanly. STRIDE 0.0635 keeps
    grid alignment that produced the 0.721 keep, so any class_weight gain
    stacks on top of the alignment sweet spot rather than competing with
    it.

    Smoke-verifiable: HistGradientBoostingClassifier supports class_weight
    in sklearn 1.7.2 (confirmed via inspect.signature); accepts dict with
    integer label keys (confirmed via instantiation).

(c) IF THIS FAILS. (1) Combined regresses below 0.721 — class_weight 2x
    inflates same_voice_edit predictions on no_splice samples, lifts
    p_splice above 0.972 on marginal clean candidates, FPs inflate,
    clean_score drops, combined regresses; next iter try class_weight={0:
    1.0, 1: 1.0, 2: 1.5} (smaller boost, lower FP inflation risk), OR
    pivot to a different retrain lever (max_iter 300→500 first, then
    learning_rate 0.07→0.05). (2) Combined matches 0.721 (within ~0.002
    noise) — class_weight 2x had no effect on Korean voice-switch TP
    recovery (perhaps marginal TPs are not feature-distinguishable from
    marginal FPs, so no class re-weighting can separate them); next iter
    try a different retrain lever (max_iter 300→500 to add capacity, or
    learning_rate 0.07→0.05 + max_iter→500 for slower careful learning),
    OR feature-engineering pivot (add new features in splice/features.py
    targeting same_voice_edit signatures). (3) Combined exceeds 0.730 —
    class_weight productive on iter1, validates same_voice_edit recall
    as the bottleneck; next iter step weight up to 3x same_voice_edit
    to continue, OR step to {0: 1.5, 1: 1.0, 2: 2.5} adding moderate
    cross_voice boost too (cross_voice OOF recall=0.97 already high so
    less expected gain there but cross_voice TPs also contribute to
    p_splice).

(d) Information gaps. Per-domain combined STILL missing from
    baseline_metrics.json on korean-iter1 — CURRENT STATE prompt says
    "(per-dataset breakdown unavailable — baseline_metrics.json has not
    captured it yet)". 15 keeps + 2 STRIDE regressions in, still cannot
    verify which domain delivered the +0.166 aggregate gain across the
    iter1 STRIDE descent (0.5546 → 0.721). At a RETRAIN pivot — the
    most expensive single-iter compute cost — direct observation of
    per-domain attribution would be the highest-value information gain.
    Without it I cannot tell post-retrain whether class_weight helped
    Korean specifically or rebalanced cross-domain. The OOF metrics
    snapshot from 8530ba8 (cross_voice F1=0.96 / no_splice F1=0.96 /
    same_voice_edit F1=0.53 recall=0.39) is also not surfaced in
    CURRENT STATE — I had to git-show it to recover the per-class
    weak-spot diagnosis. A line like "iter1 OOF: same_voice_edit F1=0.53
    (recall=0.39)" in CURRENT STATE would have made this retrain pivot
    self-evident much earlier and prevented 14 iters of inherited
    misinformation about which retrain lever is fresh. Eval runtime per
    iteration also still not surfaced — at +85% over 0.12 baseline the
    runtime margin to 300s cliff matters when retrain extends total
    iter time by 3 min.

(e) Wrapper enhancements. Three updated highest-priority asks (now 16
    consecutive iterations with these gaps unfilled):
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON on iter1 branch —
    same ask as last 15 keeps + this iter, still missing. NOW EXTREMELY
    consequential at the RETRAIN pivot: knowing which domain gained
    (or regressed) post-retrain directly determines next retrain step
    (continue boosting same_voice_edit if Korean specifically gained;
    abandon class_weight axis if singing/english regressed). Estimated
    <10min schema hookup; non-iter1 baseline already captures this.
    (2) ACTUAL CURRENT HYPERPARAMETER SNAPSHOT in CURRENT STATE:
    prompt currently lists "PRIMARY (instant)" tunables but not the
    actual current values of RETRAIN tunables (max_iter, max_depth,
    max_leaf_nodes, learning_rate, l2_regularization, min_samples_leaf,
    class_weight). My last 14 reflections all cited "min_samples_leaf
    20→40 untouched on iter1" which was WRONG (it's already 40); a
    one-block dump of the current make_pipeline() HistGBM
    hyperparameters (1 line each) would have prevented 14 iterations
    of inherited misinformation about which retrain levers are
    genuinely fresh.
    (3) CLASSIFIER OOF METRICS SNAPSHOT in CURRENT STATE: a line like
    "iter1_classifier: same_voice_edit F1=0.53 recall=0.39 (weak class)"
    would make the retrain headroom immediately visible. Available in
    fp_classifier.meta.json's oof_metrics — just needs to be surfaced
    in the prompt.
[auto] (no SHAP data for either 322fa29 or 36657c6)

## 2026-04-26T15:19:51+09:00 — 00bafe5 (discard, combined=0.721951)
subject: HistGBM max_iter 300 -> 500 (capacity bump after class_weight 2x landed only +0.001 noise-band gain on combined; OOF same_voice_edit F1 dropped 0.53 -> 0.487 confirming class_weight saturated; cited 36657c6(c)(2) retrain lever rotation; class_weight 2x retained so marginal 200 iters preferentially fit same_voice_edit residual)
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: HistGBM max_iter 300 → 500 (capacity bump on retrain axis after class_weight 2x delivered only +0.001 noise-band gain)

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py:make_pipeline()`
    one-line change — bump `max_iter` from 300 to 500. No detector edit, no
    feature edit. Wrapper auto-retrains via US-505b sha gate (~5min, was
    ~3min at 300). Other classifier hyperparams stable (max_depth=5,
    max_leaf_nodes=32, learning_rate=0.07, l2_regularization=2.0,
    min_samples_leaf=40, class_weight={0:1,1:1,2:2} kept from 36657c6,
    loss=log_loss, random_state=42). Other primary tunables stable
    (GBM_THRESHOLD=0.972, GBM_MIN_SEP_S=1.05, ANALYSIS_STRIDE_S=0.0635).
    FEATURE_NAMES stable at 80.

(b) WHY OVER RECENT FAILURES — CITED RETRAIN-LEVER ROTATION. Just-kept
    36657c6 (class_weight 2x) landed combined=0.721951 — only +0.00104
    over prior 0.720911. That is squarely in the c(2) trigger band from
    36657c6 itself: "Combined matches 0.721 (within ~0.002 noise) —
    class_weight 2x had no effect on Korean voice-switch TP recovery;
    next iter try a different retrain lever (max_iter 300→500 to add
    capacity, or learning_rate 0.07→0.05 + max_iter→500 for slower
    careful learning), OR feature-engineering pivot." Stronger evidence
    that class_weight 2x was a wash: OOF same_voice_edit F1 actually
    DROPPED from baseline 0.53 to 0.487 under class_weight 2x — the boost
    distorted calibration (raised recall by trading away precision on
    OOF), and the +0.001 combined gain at deployment is essentially the
    GBM_THRESHOLD=0.972 filter masking that distortion. Pushing
    class_weight harder (3x, balanced) would likely deepen the
    calibration distortion without further combined gain — probably
    regress as FP inflation crosses the clean_fp budget cap.

    Why max_iter 300→500 over learning_rate 0.07→0.05+max_iter→500: both
    cited; max_iter alone is the smallest single-knob change. Adding 200
    more weak learners gives the GBM more chances to find finer-grained
    discriminating splits for the rare same_voice_edit class (4056 of
    67947 = 6%) without uniformly upweighting it. With class_weight 2x
    already in place, every additional iter focuses 2x its gradient
    energy on same_voice_edit — so the marginal capacity is targeted at
    exactly the weak class. learning_rate 0.07→0.05 changes step size
    AND requires max_iter compensation to avoid undertraining; harder
    to attribute outcomes cleanly. max_iter alone is the cleaner first
    probe.

    Why max_iter over feature engineering pivot: feature engineering is
    open-ended (which features? targeting which signature?), high-risk
    (10+ feature additions historically caused regression and forced
    rollback), and slower (each feature requires retrain plus
    OOF-validation thinking). max_iter is a one-line, low-risk capacity
    bump that's strictly orthogonal to existing features. If max_iter
    saturates at 0.722 too, that's a signal that capacity isn't the
    bottleneck and feature engineering becomes the cited next move
    with high confidence. Strict ordering: cheap→expensive.

    Why max_iter over min_samples_leaf 40→20: also fresh on iter1, but
    min_samples_leaf at 40 already covers only 0.06% of training rows
    (very permissive); halving to 20 = 0.03% mostly affects tiny rare-
    pattern leaves and risks overfitting noise within the same_voice_edit
    cohort (4056 samples). max_iter is the cited first move; revisit
    min_samples_leaf if max_iter saturates.

    Mechanism on Korean recall (P=0.855 R=0.293, dominant bottleneck
    even after STRIDE descent and class_weight 2x): Korean voice-switch
    boundaries are exactly the same_voice_edit class. With OOF
    same_voice_edit recall=0.39, ~60% of training same_voice_edit
    samples cannot be classified correctly even on training-distribution
    OOF — the model's hypothesis space is undercovered for this class.
    max_iter 300→500 adds 200 more weak learners; each additional learner
    can subdivide an existing leaf where same_voice_edit samples are
    confused with no_splice (HistGBM continues fitting residual loss on
    misclassified samples). With class_weight 2x focusing gradient on
    same_voice_edit, the marginal 200 iterations preferentially target
    same_voice_edit decision boundaries. Expected outcome: OOF
    same_voice_edit F1 lifts from 0.487 toward 0.55+ (recall side).
    Combined deployment gain depends on how many of those new TPs survive
    GBM_THRESHOLD=0.972; given the +0.001 combined gain when OOF F1
    dropped 0.043, even modest OOF F1 lift could yield meaningful
    combined gain.

    Compute cost: max_iter 300→500 is ~+66% retrain time (was ~3min,
    becomes ~5min). Eval cost unchanged at STRIDE=0.0635. Total iter
    time ~8min vs ~5.5min, comfortable. No risk of feature-extraction
    blow-up.

    Smoke-verifiable: HistGradientBoostingClassifier supports max_iter
    natively (it's the early-stopping-or-fixed-iteration count); accepts
    integer 500 trivially.

(c) IF THIS FAILS. (1) Combined regresses below 0.722 — added capacity
    overfits training data (same_voice_edit minority gets perfectly
    separated on OOF but generalizes poorly to eval), or capacity
    interacts badly with class_weight 2x (accelerated overconfidence
    for both classes); next iter try max_iter=500 with class_weight
    REVERTED to None (default) to isolate the lever, OR pivot to
    learning_rate 0.07→0.05 (slower careful learning, max_iter→500
    compensation already in place), OR pivot to min_samples_leaf 40→20
    (tighter leaves for rare same_voice_edit patterns).
    (2) Combined matches 0.722 (within ~0.002 noise) — capacity wasn't
    the bottleneck; pivot to feature-engineering axis next iter (add
    new same_voice_edit-targeted features in splice/features.py — short-
    time MFCC delta-delta, sub-second voice-likelihood discontinuity,
    formant transition slope; pick ONE single feature first to keep
    attribution clean), OR revert class_weight and try max_iter alone
    (cleaner attribution).
    (3) Combined exceeds 0.730 — capacity productive on iter1 contract-
    fixed classifier; next iter step max_iter further (500→700) OR pivot
    to learning_rate 0.07→0.05 + max_iter→700 for synergistic slow
    careful learning (combined cited path), OR revert class_weight to
    None at max_iter=500 to test whether capacity alone beats
    capacity+class_weight (cleaner attribution).

(d) Information gaps. Per-domain combined STILL missing from
    baseline_metrics.json on korean-iter1. 16 keeps + 2 STRIDE
    regressions in (now 17 keeps + 6 discards), still cannot verify
    which domain delivered the +0.286 aggregate gain across iter1
    (0.436 → 0.722). At the SECOND retrain pivot — already burning
    +5min compute — direct per-domain attribution would directly
    determine next retrain lever (e.g. if Korean is finally above
    sing/eng, focus shifts to maintaining cross-domain balance not
    boosting Korean further). The OOF metrics show same_voice_edit F1
    dropped from 0.53 (pre-class_weight) to 0.487 (post-class_weight),
    even though combined eval went UP +0.001 — the OOF→combined gap
    is widening and I have no per-domain visibility into why.
    Eval runtime per iteration also still not surfaced — at +85% over
    0.12 baseline plus retrain time, total iter time is the most
    consequential constraint and isn't reported. The
    combined_<domain>=NA in the FAILED HYPOTHESES history block (every
    entry shows "[singing NA / korean NA / english NA]") confirms the
    iter1 results.tsv writer simply never captures per-domain — this
    is the single most consequential operator fix.

(e) Wrapper enhancements. Three updated highest-priority asks (now 17
    consecutive keeps + this iter with these gaps unfilled):
    (1) PER-DOMAIN combined IN BASELINE_METRICS.JSON AND results.tsv
    on iter1 branch — same ask as last 16 iters, still missing. NOW
    MOST consequential at retrain-lever rotation: knowing whether
    class_weight 2x specifically helped or hurt Korean (vs the +0.001
    aggregate masking cross-domain rebalance) directly determines
    whether to revert it before adding capacity. Estimated <30min
    operator fix; non-iter1 baseline already captures this.
    (2) ACTUAL CURRENT HYPERPARAMETER SNAPSHOT in CURRENT STATE prompt
    — same ask as 36657c6 reflection. Last 14 reflections cited "min_
    samples_leaf 20→40 untouched" which was WRONG (it's already 40);
    a one-block dump of HistGBM hyperparams in CURRENT STATE would have
    saved 14 iterations of inherited misinformation. With the retrain
    axis now active, this gap multiplies in cost.
    (3) OOF METRICS DELTA per RETRAIN ITER in CURRENT STATE: a single
    block like "iter1_classifier_oof: same_voice_edit F1 0.530→0.487
    (Δ-0.043, recall ~0.39→~0.45, precision ~0.81→~0.53)" would directly
    distinguish productive class_weight effect from calibration distortion.
    Available in fp_classifier.meta.json's oof_metrics — just needs to
    be diffed against prior keep's meta and surfaced. With retrain axis
    active, this is the second-highest-priority fix after per-domain
    combined.

## 2026-04-26T17:29:01+09:00 — c681ee7 (verify-fail, combined=0.077736)
subject: revert class_weight={0:1,1:1,2:2} -> default None (METRIC PIVOT — same_voice_edit boost is now actively harmful under F0.5 x clean_fp_penalty; OOF same_voice_edit F1 dropped 0.530->0.487 under boost; under new precision-favoring metric this calibration distortion surfaces as confident clean FPs that the 10x penalty drag punishes hardest; THRESH 0.985 already filtered marginals so root-cause attack on same_voice_edit overconfidence is highest leverage)
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: REVERT class_weight={0:1,1:1,2:2} → default (None) (METRIC PIVOT IMPLICATION — same_voice_edit boost is now actively harmful under F0.5 × clean_fp_penalty)

(a) HYPOTHESIS. Single-line `splice/classifier/train_classifier.py:make_pipeline()`
    edit — DELETE the `class_weight={0: 1.0, 1: 1.0, 2: 2.0}` line. Wrapper
    auto-retrains via US-505b sha gate (~3 min). No detector edit, no feature
    edit. Other classifier hyperparams stable (max_iter=300, max_depth=5,
    max_leaf_nodes=32, learning_rate=0.07, l2_regularization=2.0,
    min_samples_leaf=40, loss=log_loss, random_state=42). Other primary
    tunables stable (GBM_THRESHOLD=0.985, GBM_MIN_SEP_S=1.05,
    ANALYSIS_STRIDE_S=0.0635). FEATURE_NAMES stable at 80.

(b) WHY OVER RECENT FAILURES — METRIC PIVOT FUNDAMENTALLY CHANGES OPTIMUM.
    Metric pivoted (commits 44b798f, 5ecc430, 4a98c85): combined =
    F0.5(P,R) × clean_fp_penalty where penalty = 1/(1+clean_fp_per_min/1.0).
    Current state (post-4a98c85 keep): combined=0.0777, F0.5=0.7878,
    P=0.853, R=0.603, clean_fp_per_min=9.143, clean_fp_penalty=0.0986.
    PROMPT METRIC DEFINITION IS STALE — still cites F1×clean_score / GM
    across singing/korean/english, but actual evaluator runs F0.5 × penalty
    over single dataset (data/eval/korean_iter1/eval).

    Decomposition: F0.5 already near-saturated (0.79); penalty=0.099 is the
    10× drag. Every clean_fp_per_min reduction translates ~directly into
    combined gain. 4a98c85 (THRESH 0.972→0.985) only moved clean_fp_per_min
    9.54→9.14 (-4%) — surviving clean FPs are HIGHLY confident (well above
    0.99 in p_splice), so further threshold raises = diminishing returns.
    Need to attack the SOURCE of confident clean FPs, not gate them harder.

    Root cause documented in classifier metadata
    (splice/classifier/fp_classifier.meta.json):
        cross_voice OOF F1     = 0.956
        no_splice OOF F1       = 0.961
        same_voice_edit OOF F1 = 0.487   ← weak class, made WORSE by boost
    Pre-boost baseline (cited in 36657c6 commit / 8530ba8): same_voice_edit
    OOF F1 = 0.530. So class_weight 2x DROPPED OOF same_voice_edit F1 from
    0.530→0.487. Under F1 metric this masked itself as +0.001 deployment
    gain (recall up, precision down ≈ wash). Under F0.5 × penalty this is
    pure cost: F0.5 weights P 2x, AND the precision distortion surfaces
    as confident same_voice_edit predictions inside clean audio → clean
    FPs that the penalty punishes 10x harder.

    Why class_weight revert > other levers:
    - GBM_THRESHOLD 0.985→0.99: just-tried 0.972→0.985 cut clean_fp by
      only 4%; clean FPs are confident, more raise = same diminishing
      returns AND more recall loss (R already dropped 0.632→0.603 in
      one threshold step; another would push R below 0.55 and tank F0.5).
    - GBM_MIN_SEP_S 1.05→3.0: dedupes adjacent emits, but isolated clean
      FPs in continuous speech have no neighbor to dedupe; doesn't attack
      clean FPs; also hurts Korean recall.
    - max_iter 300→500: just-discarded 00bafe5; capacity not the bottleneck.
    - Suppress same_voice_edit class entirely: aggressive, untested
      cross-class effects, blunt instrument.
    Class_weight revert directly attacks the ONE documented OOF precision
    drop and the ONE class structurally capable of confident emissions
    inside continuous single-speaker audio. Mechanism: reverting to
    default class_weight=None means HistGBM training spends gradient
    proportional to class frequency (no_splice 45298 / cross_voice 18593 /
    same_voice_edit 4056). Same_voice_edit gets less per-epoch boost →
    model less aggressive predicting same_voice_edit on boundary-case
    features → P(same_voice_edit) inside clean audio drops → fewer
    p_splice values cross 0.985 → clean_fp_per_min drops.

    Risk-reward: F0.5(0.85, 0.60) = 0.788. Pessimistic outcome (R drops
    to 0.50, clean_fp_per_min only to 5.0): F0.5=0.745, penalty=0.167,
    combined=0.124 (vs 0.078 = +60%). More pessimistic (clean_fp only
    to 7.0): F0.5=0.745, penalty=0.125, combined=0.093 (vs 0.078 = +20%).
    Even pessimistic outcomes net positive because penalty leverage
    dominates F0.5 sensitivity. Asymmetric upside under new metric.

    Smoke-verifiable: deleting the class_weight kwarg yields sklearn
    default (None). Wrapper retrain triggered by train_classifier.py sha
    change (US-505b).

(c) IF THIS FAILS. (1) Combined regresses below 0.0777 — class_weight
    revert dropped recall too far without proportionate clean_fp gain
    (e.g., R: 0.603→0.45, clean_fp_per_min: 9.14→7.5 only); next iter
    try smaller revert step class_weight={0:1, 1:1, 2:1.5} (compromise),
    OR pivot to GBM_THRESHOLD 0.985→0.99 (one more threshold push since
    F0.5 still has marginal headroom), OR GBM_MIN_SEP_S 1.05→2.0
    (instant; dedupes any per-utterance clusters even if isolated clean
    FPs persist).
    (2) Combined matches 0.0777 (within ±0.003 noise) — class_weight
    wasn't the source of confident clean FPs (they came from cross_voice
    overconfidence or feature-level signal); next iter pivot to
    GBM_THRESHOLD 0.985→0.99 (tighten gate further), OR feature-
    engineering pivot in splice/features.py adding a "clean-audio guard"
    feature (rolling spectral stationarity over ±1s window — high
    stationarity = continuous speech = should not emit; targets clean
    FPs at the feature level).
    (3) Combined exceeds 0.090 — class_weight revert productive on iter1
    classifier under new metric; next iter try DROPPING same_voice_edit
    weight further class_weight={0:1, 1:1, 2:0.5} (test whether
    deweighting same_voice_edit further continues to slash clean FPs —
    same_voice_edit is structurally low-precision so under-weighting
    shifts posterior toward no_splice on ambiguous cases), OR test
    class_weight={0:1.5, 1:1, 2:1} (boost cross_voice instead — high-
    precision class, its boost shifts confident predictions toward the
    "correct" splice class without precision cost).

(d) Information gaps. (1) Per-class FP composition NOT surfaced — I
    cannot confirm clean FPs are predominantly same_voice_edit-labeled.
    If clean FPs are mostly cross_voice, reverting class_weight would
    be misdirected. CURRENT STATE line "clean_fp by class:
    same_voice_edit=N1 cross_voice=N2 unknown=N3" would have made this
    hypothesis directly verifiable rather than confident-but-untested
    inference from OOF metadata.
    (2) PROMPT METRIC DEFINITION SECTION IS STALE — still cites
    "splice_f1 × clean_score" + GM across singing/korean/english.
    Actual evaluator (post-44b798f) is F0.5 × clean_fp_penalty over
    single korean_iter1 eval. Could mislead future hypotheses.
    (3) RESEARCH NOTES still contain stale per-domain ask repeated 14×
    — that ask is MOOT under single-dataset eval. Notes need pruning
    so they don't continue to skew hypothesis framing.
    (4) PER-TUNABLE EXPLORATION FRONTIER `current` column blank
    ("current ?") — had to grep detector.py manually for actual values.
    (5) Frontier 'kept' values for THRESH/STRIDE were tuned under OLD
    metric — those keeps are now stale optima. The frontier should
    distinguish 'kept under metric_v1' vs 'kept under metric_v2'.

(e) Wrapper enhancements (re-prioritized for new metric reality):
    (1) PROMPT METRIC DEFINITION REFRESH — top-of-prompt block still
    describes OLD metric and OLD multi-dataset GM. Single most
    consequential prompt fix; misleading the agent on the objective is
    the highest-cost wrapper bug right now. Should auto-emit from
    splice/evaluate.py docstring or constants.
    (2) PER-CLASS CLEAN_FP BREAKDOWN in CURRENT STATE — at new metric
    where clean_fp dominates 10×, knowing whether clean FPs are
    same_voice_edit-vs-cross_voice-labeled directly determines next
    classifier-side hypothesis. splice/evaluate.py already labels every
    prediction; per-class clean_fp counts = ~5 lines in
    compute_clean_fps_per_file.
    (3) OOF METRICS DELTA per RETRAIN ITER in CURRENT STATE — same ask
    as last 2 iters. With class_weight revert pending, knowing whether
    same_voice_edit OOF F1 returned to ~0.530 or moved differently is
    the key sanity check on the revert mechanism. Would reduce
    attribution ambiguity to zero.

## 2026-04-26T17:37:17+09:00 — 2dfb3d4 (keep, combined=0.093046)
subject: GBM_MIN_SEP_S 1.05 -> 2.5 (attack clean-FP clusters under new F0.5 x clean_fp_penalty metric where penalty=0.099 is 10x drag; just-discarded c681ee7 confirmed class_weight not the FP source; just-kept 4a98c85 THRESH 0.985 only cut clean_fp 4% so survivors are confident; dense-scan at STRIDE=0.0635 + 1.05s dedupe leaves multi-second clean-region clusters as multiple FPs; under F0.5 weighting P 2x R, aggressive dedupe favorable trade; 2.5 fresh on frontier above old-metric tested band [1.0, 2.0])
per-domain: (no per-domain data)

# 2026-04-26 — hypothesis: GBM_MIN_SEP_S 1.05 → 2.5 (attack clean-FP clusters under new F0.5 × penalty metric where penalty is 10× drag)

(a) HYPOTHESIS. Pure `splice/detector.py` one-line change — raise
    GBM_MIN_SEP_S from 1.05 to 2.5. No retrain, no feature edit. Other
    primary tunables stable: GBM_THRESHOLD=0.985, ANALYSIS_STRIDE_S=0.0635,
    DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0. Classifier hyperparams
    stable (class_weight={0:1,1:1,2:2}, max_iter=300, etc.). FEATURE_NAMES
    stable at 80. 2.5 is fresh on frontier (set: kept 1.05/1.1/1.25/1.5/2.0,
    failed 1.0/1.025 = 7 tried; 2.5 never tried).

(b) WHY OVER RECENT FAILURES — METRIC-PIVOT-AWARE LEVER ROTATION.
    Current state: combined=0.0777, F0.5=0.788, P=0.853, R=0.603,
    clean_fp_per_min=9.14, penalty=0.099. F0.5 already near-saturated;
    penalty is the 10× drag. Each clean_fp_per_min reduction translates
    ~directly into combined gain (TAU=1.0).

    Just-discarded c681ee7 (revert class_weight 2x→None) verify-failed
    at essentially flat combined (~0.078) — class_weight is NOT the
    primary source of clean FPs. Just-kept 4a98c85 (THRESH 0.972→0.985)
    only moved clean_fp_per_min ~9.54→9.14 (-4%) — surviving clean FPs
    are highly confident (well above 0.99 likely), so further threshold
    push (0.985→0.99) yields diminishing return on the same lever.

    NEW MECHANISM: clean FPs in continuous single-speaker audio aren't
    isolated point emissions — the dense scan at STRIDE=0.0635 emits
    every 63.5ms, so a "noisy" clean region (chord transition, breath,
    plosive) triggers high p_splice across multiple adjacent candidates.
    GBM_MIN_SEP_S=1.05 only collapses neighbors within 1.05s. A cluster
    spanning 1.05–2.5s survives as multiple FPs. Raising to 2.5s
    collapses wider clusters to one emit.

    Under NEW metric (F0.5 × penalty), the trade favors aggressive
    dedupe even at recall cost. Math: if R drops 0.603→0.45 and
    clean_fp_per_min drops 9.14→4.0, F0.5(0.85,0.45)=0.722 (down 8%)
    but penalty(4.0)=0.20 (up 100%); combined → 0.144, +85%. Even
    pessimistic (R→0.5, clean_fp→6.0): F0.5≈0.745 × 0.143 = 0.107, +37%.

    Why 2.5 not 2.0 or 3.0: 2.0 frontier-blocked (kept under OLD metric
    at 0.481, but old combined was F1×clean_score×GM where R loss
    dominated). 2.5 is fresh, sits cleanly above the OLD-metric tested
    band, preserves bisection optionality (down to 1.75 if over-aggressive,
    up to 3.5 if productive). 3.0 is bigger swing; 2.5 is conservative
    first probe of the fresh band.

    Why over GBM_THRESHOLD 0.985→0.99: 0.972→0.985 cut FP only 4%; the
    survivor distribution is peaked at p>>0.985, so one more tightening
    likely yields similar marginal return AND costs recall (R already
    dropped 0.632→0.603 in the 0.985 step). Dedupe attacks an orthogonal
    mechanism (cluster collapse).

    Why over class_weight={0:1,1:1,2:0.5}: c681ee7 already showed
    class_weight revert (2x→1x) was flat combined, so further
    class_weight tweaks are diminishing-return — the FP source isn't
    class_weight-driven.

    Smoke-verifiable: GBM_MIN_SEP_S=2.5 trivially imports; one-line
    change to splice/detector.py:50.

(c) IF THIS FAILS. (1) Combined regresses below 0.0777 — dedupe lost
    too many real TPs without proportionate clean_fp reduction; next
    iter try smaller step GBM_MIN_SEP_S 1.05→1.75 (compromise, fresh
    on frontier), OR pivot to GBM_THRESHOLD 0.985→0.99 (cited next-step,
    smaller diminishing-return push). (2) Combined matches 0.0777
    (within ±0.003 noise) — clean FPs aren't cluster-driven (they're
    isolated single-emit FPs); pivot to feature-engineering axis next
    iter (add a "clean-audio guard" feature targeting rolling spectral
    stationarity over ±1s), OR GBM_THRESHOLD 0.985→0.99. (3) Combined
    exceeds 0.090 — dedupe productive on iter1 under new metric; next
    iter step further GBM_MIN_SEP_S 2.5→3.5 to continue, OR layer
    GBM_THRESHOLD 0.985→0.99 on top.

(d) Information gaps. (1) Per-class clean_fp breakdown still NOT
    surfaced — cannot confirm clean FPs are predominantly
    same_voice_edit-labeled vs cross_voice. (2) PROMPT METRIC
    DEFINITION block still cites OLD metric (splice_f1 × clean_score,
    multi-domain GM, clean_fp ≤ 15/45 bounds) — dangerously misleading.
    (3) Per-tunable frontier `current` column blank ("current ?") —
    had to grep detector.py manually. (4) Frontier doesn't tag
    classifier_sha or metric_version, so OLD-metric kept values
    (1.05/1.1/1.25/1.5/2.0) appear blocking even though they should be
    retestable under new metric. (5) Eval runtime per iteration not
    surfaced.

(e) Wrapper enhancements (re-prioritized for new-metric reality):
    (1) PROMPT METRIC DEFINITION REFRESH — top-of-prompt block
    describes OLD metric and OLD multi-dataset GM. Single most
    consequential prompt fix; should auto-emit from splice/evaluate.py.
    (2) PER-CLASS CLEAN_FP BREAKDOWN in CURRENT STATE — at new metric
    where clean_fp dominates 10×, knowing whether clean FPs are
    same_voice_edit-labeled vs cross_voice-labeled directly determines
    next classifier-side hypothesis.
    (3) FRONTIER METRIC-VERSION TAG: each frontier entry should carry
    `metric_version=v1|v2` so OLD-metric kept values that are now stale
    optima can be retested without operator-owned sentinel. With 7
    GBM_MIN_SEP_S values frontier-blocked under stale metric, this is
    the binding constraint on rapid lever rotation.
[auto] (no SHAP data for either 4a98c85 or 2dfb3d4)

