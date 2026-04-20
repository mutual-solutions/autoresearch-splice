## Historical digest (oldest 20 entries compacted)
Span: Historical digest (oldest 20 entries compacted) … 2026-04-20T15:57:39+09:00 — df0ceec (discard, combined=0.490700)
Outcomes: 2 keep / 17 discard / 0 verify-fail; combined range [0.4219, 0.5893].
(Older reflections collapsed to conserve prompt budget.)

## 2026-04-20T16:14:07+09:00 — 1a8b1fe (discard, combined=0.503311)
subject: add voiced_percussive_chroma_asymmetry feature (FEATURE_NAMES 80->81) — signed asymmetry percussive_chroma_cosine_dist minus voiced_chroma_cosine_dist on +-2s pre/post window. Percussive mask = top-40% onset-energy frames (same threshold as fd500b3/aa4f141: onset > median*1.2); voiced mask = feat_vp>0. Cosine distance on 12-dim masked-mean chroma vectors, then subtract. Reuses cached feat_chroma + feat_onset + feat_vp — ZERO new librosa calls, ZERO new caches. FEATURE_NAMES count 80->81 triggers wrapper auto-retrain via US-505 sha gate. Targets 7972a98 current-keep (combined=0.589282) singing 0.345 weakest-domain plateau (3 clean FPs). Every plausible classifier-hyperparam axis saturated (depth/leaf 99081f5+960113c matched baseline byte-identical, LR 17d4aec / min_samples_leaf db59c36 / l2 df0ceec all produced IDENTICAL 0.490700) and every primary tunable saturated in both directions; voiced/unvoiced MFCC/chroma/contrast/flatness variants exhausted. This is the explicit cited untried fallback from 5+ consecutive reflections (0cb551f/c3, 53d3ea0/c3, 67633f2/c3, 17d4aec/c3, db59c36/c3, df0ceec/c2, 0c3bf76). Mechanism: percussive-mean chroma filters OUT drum contribution (drums broadband-noise have ~uniform 12-bin chroma averaging to near-uniform) and leaves pitched-accompaniment transient content (bass/guitar attacks) — reflects accompaniment CHORD signature. Voiced-mean chroma reflects singer's pitch class. Case (i) same-singer cross-song: voiced_chroma_dist small (~0.08) because singer holds similar key, percussive_chroma_dist large (~0.35) because accompaniment changes key -> asymmetry ~+0.27 LARGE. Case (ii) intra-song chord cycle (the 3 singing FPs): voice and accompaniment shift TOGETHER with chord progression -> both distances move in tandem -> asymmetry ~0 — chord-cycle FP cannot mimic positive-asymmetry signature because chord cycle is inherently synchronized. Case (iii) different-singer cross-song: both shift -> asymmetry cancels -> GBM falls back on voiced_mfcc / voiced_unvoiced_mfcc_asymmetry (1eda8e3 +0.054) which already fires. Critical orthogonality: percussive × MFCC previously failed both solo (fd500b3 singing regressed 0.345->0.319 — percussive MFCC is drum-timbre-noisy on speech consonants) and in asymmetry (aa4f141 singing 0.320 — cepstral noise mode persists even with paired differencing because voiced+percussive MFCC both shift with phoneme context). CHROMA is physically different from MFCC: drum chroma is near-flat broadband-noise distribution (averages away in percussive mean), whereas drum MFCC has strong timbre signature (dominates percussive mean). Speech self-gating via paired differencing (reproduces 1eda8e3 pattern): percussive-frame chroma tracks plosive bursts (phoneme-dependent, noise-like), voiced-frame chroma tracks vowel-formant harmonicity (phoneme-dependent, tonal) — both vary with phoneme context UNCORRELATED with splice position, so DIFFERENCE is zero-mean noise across training pairs, GBM assigns low per-domain SHAP on english/korean, feature functionally invisible on speech. Prevents the 634cdd2 raw-chroma and df6fc0a unvoiced_mfcc speech-regression failure mode. Orthogonal to every prior axis: NOT 1eda8e3 voiced_unvoiced_mfcc (cepstral asymmetry, unvoiced mask); NOT 7972a98 voiced_unvoiced_spec_contrast (spec_contrast asymmetry, unvoiced mask); NOT f4148cc voiced_unvoiced_chroma (chroma asymmetry but UNVOICED mask — failed because singer sings new note in new key so voiced shifts as much as unvoiced and asymmetry cancels); NOT aa4f141 voiced_percussive_mfcc (percussive × MFCC asymmetry — cepstral noise mode); NOT fd500b3 percussive_mfcc solo; NOT 32cac36 voiced_chroma solo (single-mask, no pair); NOT 308aa5a voiced_tonnetz; NOT 0c3bf76 voiced_unvoiced_spec_flatness_asymmetry. FIRST percussive × chroma content combination in any form, FIRST percussive mask against a non-cepstral content axis. Self-diagnostic on the IDENTICAL-COMBINED-STREAK concern (5 consecutive iterations at 0.490700 across hyperparam-only changes): features.py count bump 80->81 forces retrain via US-505 sha gate (0c3bf76 featadd produced distinct 0.538510 proving this path is alive) — if my change also lands at 0.490700, that's strong evidence of wrapper bug beyond cache invalidation. Blast radius: features.py only — 1 new block function (~40 lines) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. ZERO new caches, ZERO new librosa calls, one lazy scalar feat_perc_threshold cached on ctx (same machinery as aa4f141/fd500b3). Per-t cost: 4 slices + 2 mask applications + 4 masked means + 2 cosines on 12-dim vectors + 1 subtraction, sub-ms. Aggregate per-chunk cost negligible. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_percussive_chroma_asymmetry', extract_features returns 81 finite features on synthetic audio, idempotent on repeated calls, sentinel 0.0 fires correctly on silence (empty voiced mask).
per-domain: combined_english=0.839506 combined_korean=0.476471 combined_singing=0.318750

# 2026-04-20 — hypothesis: add voiced_percussive_chroma_asymmetry feature (FEATURE_NAMES 80→81)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new
    feature: signed asymmetry `percussive_chroma_cosine_dist` −
    `voiced_chroma_cosine_dist` on ±2s pre/post window. Percussive mask
    = frames where `feat_onset > chunk-median(feat_onset) * 1.2` (top
    ~40% by onset energy; same threshold used by fd500b3/aa4f141).
    Voiced mask = `feat_vp > 0`. Cosine distance on 12-dim masked-mean
    chroma vectors, then subtract. Reuses cached feat_chroma +
    feat_onset + feat_vp — ZERO new librosa calls, ZERO new caches.
    Sentinel 0.0 when any mask empty or any sub-norm underflows,
    matching 1eda8e3 / 7972a98 safety pattern. FEATURE_NAMES 80→81
    triggers wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. Every plausible axis has been
    eliminated in the last ~15 iterations: all 5 HistGBM hyperparams
    individually (max_depth/max_leaf 99081f5+960113c — capacity INERT,
    the revert matched baseline byte-identical; learning_rate 17d4aec,
    min_samples_leaf db59c36, l2 df0ceec — all produced IDENTICAL
    combined=0.490700 suggesting classifier truly saturated at the
    80-feature space); every primary tunable in both directions
    (THRESHOLD 0.980-0.983, MIN_SEP 2.5-3.75, STRIDE 0.11-0.15,
    DSP_MAX 2.0-2.2, DSP_SUM 4.5-5.5, DSP_PHASE 0.5); every
    voiced/unvoiced MFCC / chroma / contrast / flatness variant in
    both solo and asymmetry configurations; feature-removal (ea1636c
    catastrophic 0.455). The classifier cannot extract more from the
    current feature set — forward progress requires exposing a new
    mechanism to GBM. `voiced_percussive_chroma_asymmetry` is cited
    as the explicit untried fallback in 5+ consecutive reflections
    (0cb551f/c3, 53d3ea0/c3, 67633f2/c3, 17d4aec/c3, db59c36/c3,
    df0ceec/c2, 0c3bf76 discussion) — every reflection since ~20
    iterations ago has queued this as the next experiment.

    Mechanism targeting singing 0.345 weakest-domain plateau (3 clean
    FPs). Percussive mask captures TOP-40% onset-dominated frames:
    on singing these are the transient attacks that carry both drum
    hits (flat chroma noise) AND bass/guitar attack frames (strongly
    pitched). The chroma MEAN over percussive frames therefore
    reflects the PITCHED ACCOMPANIMENT CHORD (drum noise averages
    near-uniform, leaving the pitched instrument chroma visible).
    Voiced mask captures sustained sung notes — chroma mean reflects
    the singer's key / pitch class. Asymmetry = accompaniment_chroma
    shift MINUS voice_chroma shift.

    Three discriminating cases:
    (i) Real same-singer cross-song splice: singer holds similar
    pitch class across cut -> voiced_chroma_dist small (~0.08). New
    accompaniment in different key -> percussive_chroma_dist large
    (~0.35). Asymmetry ~+0.27 LARGE POSITIVE — chord-cycle FP cannot
    mimic this because chord cycle moves BOTH voice and accompaniment
    together.
    (ii) Intra-song chord cycle (the 3 singing FPs): voice and
    accompaniment shift TOGETHER with the chord progression so both
    distances move in tandem -> asymmetry ~0 SMALL. Discriminator.
    (iii) Different-singer cross-song: both shift -> both large ->
    diff ~0. GBM falls back on voiced_mfcc /
    voiced_unvoiced_mfcc_asymmetry which already fire on this case
    (1eda8e3 +0.054).

    Why THIS MASK × CONTENT combination. Percussive × MFCC was tried
    solo (fd500b3 FAILED, singing regressed 0.345->0.319 because
    unmasked percussive MFCC dominated by drum timbre noise on
    speech) AND in asymmetry form (aa4f141 voiced_percussive_mfcc_
    asymmetry FAILED, singing 0.320). Both failures were in CEPSTRAL
    (timbre) axis — where percussive-mean MFCC is genuinely noisy
    because drum timbre is phoneme-dependent on speech consonants.
    CHROMA is physically different: drums have near-flat chroma
    (broadband noise -> ~uniform 12-bin distribution), so percussive-
    mean chroma filters OUT drum contribution and leaves the pitched
    transient content (bass + guitar attacks on singing, plosive
    noise on speech). Paired differencing against voiced_chroma then
    gives a SPECIFIC accompaniment-vs-voice pitch-class shift
    discriminator.

    Speech self-gating via paired differencing (reproduces 1eda8e3
    MFCC asymmetry mechanism that lifted english 0.732->0.889 and
    korean 0.526->0.667): percussive-frame chroma on speech tracks
    plosive bursts (phoneme-dependent, noise-like), voiced-frame
    chroma tracks vowel-formant harmonicity (phoneme-dependent,
    tonal). Both vary with phoneme context uncorrelated with splice
    position, so the DIFFERENCE is zero-mean noise across training
    pairs -> GBM assigns low per-domain SHAP on english/korean ->
    feature functionally invisible on speech, preventing the
    634cdd2 raw-chroma / df6fc0a unvoiced_mfcc speech-regression
    failure mode.

    Orthogonal to every prior axis: NOT 1eda8e3 (cepstral asymmetry,
    unvoiced mask); NOT 7972a98 (spec_contrast asymmetry, unvoiced
    mask); NOT f4148cc (chroma asymmetry but UNVOICED mask — failed,
    singer sings new key so voiced shifts too, unvoiced-mask chroma
    carries per-phoneme noise); NOT aa4f141 (percussive × MFCC
    asymmetry — cepstral noise mode); NOT fd500b3 (percussive × MFCC
    solo); NOT 32cac36 voiced_chroma solo (single-mask, no pair); NOT
    308aa5a tonnetz; NOT 0c3bf76 flatness asymmetry. First PERCUSSIVE
    × CHROMA content combination in any form, first percussive mask
    against a non-cepstral content axis. Blast radius: features.py
    only — 1 new block function + 1 FEATURE_NAMES append + 2 assert
    bumps + 1 call in extract_features. ZERO new caches, ZERO new
    librosa calls. Per-t cost: 4 slices + 2 mask applications + 4
    masked means + 2 cosines on 12-dim vectors + 1 subtraction,
    sub-ms.

    IDENTICAL-COMBINED-STREAK concern (5 consecutive at 0.490700)
    applied to hyperparam-only changes — features.py count changes
    force retrain via US-505 sha gate (0c3bf76 featadd produced
    distinct 0.538510, proving the path works). My 80->81 bump
    guarantees a real retrain.

(c) IF THIS FAILS. (1) combined drops modestly (0.52-0.58, singing
    stuck 0.30-0.34) -> drum attacks dominate percussive-mean and
    bury bass/guitar chroma signal; fallback is
    voiced_spec_rolloff_asymmetry (spectral-rolloff shift is
    mastering-independent brightness signature, complementary to
    contrast/flatness already covered). (2) combined regresses
    below 0.45 -> asymmetry introduces speech-regression noise;
    fallback is voiced_unvoiced_spec_rolloff_asymmetry (same template
    but rolloff content, dominated by mastering filter cutoff not
    phoneme-level noise). (3) combined matches 0.490700 AGAIN with
    a feature-count change -> wrapper has a deeper bug than cache
    invalidation; escalate to operator with identical-output streak
    as evidence.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 33+
    iterations — I cannot verify the 3 singing FPs' actual
    (voiced_chroma_dist, percussive_chroma_dist) distribution to
    predict whether asymmetry fires or sits at noise floor. Every
    feature-add hypothesis since iteration ~70 theory-calibrated on
    physical mechanism without eval-corpus verification. (ii) SHAP
    rollup STILL "no keeps yet — rollup empty" for 7972a98 despite
    14 keeps in loop history — rollup writer broken. No per-feature
    attribution available. (iii) 99081f5 classifier capacity ghost
    (max_depth=4, max_leaf=16) IN-tree at HEAD per grep. 960113c
    revert to 3/8 matched baseline EXACTLY then discarded — wrapper's
    symmetry-semantics bug restored 4/16. Feature-add delta
    attribution is clean but absolute comparison mixes with ghost.
    (iv) 0.490700 streak unresolved across 5 physically different
    changes — statistically impossible unless wrapper cache-
    invalidation / retrain-skip bug. Cannot diagnose from inside
    claude subprocess.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE — persistent blocker across 33+ iterations, cited
    in every recent reflection. Per-FP: (domain, file, t_sec,
    label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    voiced_chroma_cosine_dist, voiced_mfcc_cosine_dist,
    voiced_unvoiced_mfcc_asymmetry, top-5 |SHAP|). Implementation:
    after eval completes, for each clean_fp run extract_features +
    predict_proba + SHAP, write `.omc/clean_fp_positions.json`,
    wrapper injects into prompt. (2) IDENTICAL-COMBINED-STREAK
    DIAGNOSTIC — emit `wrapper.suspicious.identical_output_streak`
    when N >= 3 consecutive iterations produce IDENTICAL combined
    ± 1e-5 AND IDENTICAL per-domain AND non-identical source diffs.
    Include sha256 of committed joblib / features.py / detector.py /
    train_classifier.py across the streak — if all byte-identical
    across physically distinct commits, retrain-skip is confirmed.
    ~30 lines in post-eval hook. (3) DISCARD-REVERT SYMMETRY
    SEMANTICS bug fix — wrapper treats "combined matches baseline
    exactly" as no-improvement and reverts, which RESTORED the
    99081f5 capacity ghost via 960113c. Fix: if discarded commit
    subject starts "REVERT" and references a prior-discarded SHA,
    treat exact-match as KEEP not DISCARD. ~15 lines of
    git-commit-message parsing.

## 2026-04-20T16:33:07+09:00 — 4773d1e (discard, combined=0.482741)
subject: add percussive_spec_contrast_cosine_dist feature (FEATURE_NAMES 80->81) — single-mask 7-dim spec_contrast cosine distance restricted to percussive frames (top-40%% onset-energy, threshold = median(feat_onset)*1.2 matching fd500b3/aa4f141/1a8b1fe). Pre=[t-2,t] vs post=[t,t+2]. Reuses cached feat_contrast + feat_onset — ZERO new librosa calls, ZERO new caches; one lazy per-chunk feat_perc_threshold scalar. Sentinel 0.0 when mask empty or any sub-norm underflows. FEATURE_NAMES count 80->81 forces wrapper auto-retrain via US-505 sha gate. Targets 7972a98 current-keep (combined=0.589282) singing 0.345 weakest-domain plateau (3 clean FPs surviving DSP gates). Last 10+ iterations exhausted classifier hyperparam axis (99081f5+960113c byte-identical capacity revert; LR 17d4aec / min_samples_leaf db59c36 / l2 df0ceec all produced IDENTICAL combined=0.490700 strongly suggesting US-505 does NOT auto-retrain on train_classifier.py-only edits) and every primary tunable (THRESHOLD/MIN_SEP/STRIDE/DSP_MAX/DSP_SUM/DSP_PHASE all tested both directions). Only path alive is features.py-sha-change (1a8b1fe feature-add produced distinct 0.503311 confirming feature-count bump forces retrain). Genuinely NEW feature-space cell PERCUSSIVE x SPEC_CONTRAST. Prior percussive-mask attempts used only CEPSTRAL (fd500b3 percussive_mfcc solo failed, aa4f141 voiced_percussive_mfcc_asymmetry failed — phoneme-correlated noise mode on speech consonants) or PITCH-CLASS (1a8b1fe voiced_percussive_chroma_asymmetry failed — drum chroma is broadband-flat so percussive-chroma mean averages to uniform burying signal). SPEC_CONTRAST is physically DIFFERENT from both: per-band peak-to-valley amplitude ratio is a compressor/EQ/limiter mastering fingerprint, LESS phoneme-correlated than MFCC (per-band contrast varies with mastering chain not phoneme), and drums have NON-FLAT per-band contrast signature (snare ~200Hz, kick ~60Hz, cymbals 5-10kHz produce distinctive peak/valley profile on percussive frames). Mechanism on singing 3 FPs: within one song mastering chain frozen (drum bus compressor + vocal bus + limiter all constant) so percussive_spec_contrast stays stable across chord transitions (cosine_dist 0.05-0.15 small). Cross-song splice crosses drum-bus processing (different drum kit / different compression / different limiter threshold) — per-band peak/valley shifts (cosine_dist 0.25-0.60 large). Intra-song chord cycle (the 3 singing FPs): drum kit unchanged → feature sentinel-small → real cross-song-splice signature is LOUDER on percussive mask than on voiced/unvoiced split because drum-bus processing is the most album-variable mastering element. On speech: percussive frames are plosive bursts whose per-band contrast varies by phoneme but per-phoneme variance is uncorrelated with splice position → GBM low per-domain SHAP via tree splits → self-gates (same mechanism as voiced_spec_contrast on speech). Why single-mask not asymmetry: voiced_unvoiced_spec_contrast_asymmetry already exists (7972a98 keep +0.005) so spec_contrast axis already split via voiced/unvoiced paired-difference route; an unpaired PERCUSSIVE version hits mastering signal through a different frame population (transient-dominant) providing complementary GBM split ground (trees combine voiced_unvoiced_spec_contrast_asymmetry > T1 AND percussive_spec_contrast_cosine_dist > T2 to specifically discriminate same-singer cross-song splice from intra-song chord cycle). Orthogonal to every prior axis: NOT fd500b3 percussive_mfcc (cepstral); NOT aa4f141 voiced_percussive_mfcc_asymmetry (cepstral pairing); NOT 1a8b1fe voiced_percussive_chroma_asymmetry (pitch-class); NOT e7ca9eb voiced_spec_contrast (voiced mask, different frame population); NOT 7972a98 voiced_unvoiced_spec_contrast_asymmetry (voiced/unvoiced pair); NOT de0be6f spec_contrast_symmetry (pairwise sum); NOT 0c3bf76 spec_flatness_asymmetry (scalar abs-delta formulation). FIRST PERCUSSIVE x SPEC_CONTRAST content combination in any form, FIRST percussive mask against a non-cepstral non-pitch-class content axis. Pure features.py change — 1 new block function (~40 lines) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. ZERO new caches, ZERO new librosa calls. Per-t cost: 2 slices + 1 mask compare + 1 masked mean + 1 cosine on 7-dim vectors, sub-ms. Risk-bounded: sentinel 0.0 on empty-mask edge cases; signal physically grounded on singing where drum-bus mastering shifts across cross-song splices. Smoke-verified: len(FEATURE_NAMES)==81, last name 'percussive_spec_contrast_cosine_dist', extract_features returns 81 finite features on synthetic sustained+click audio, idempotent on repeated calls, percussive-mean cosine_dist=0.0024 (small) on stationary within-source signal as expected.
per-domain: combined_english=0.843373 combined_korean=0.517391 combined_singing=0.257812

# 2026-04-20 — hypothesis: add percussive_spec_contrast_cosine_dist (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — new single-mask feature
    `percussive_spec_contrast_cosine_dist`. Percussive mask = frames where
    `feat_onset > median(feat_onset) * 1.2` (top ~40% onset-energy,
    identical threshold to fd500b3/aa4f141/1a8b1fe). Per position t_sec
    slice feat_contrast (7-dim per frame, already cached for
    spec_contrast_delta and e7ca9eb voiced_spec_contrast) on pre=[t−2, t]
    and post=[t, t+2], keep percussive frames, cosine distance of
    7-dim percussive-mean contrast vectors. Reuses cached feat_contrast +
    feat_onset — ZERO new librosa calls, ZERO new caches; one lazy per-
    chunk `feat_perc_threshold` scalar cached on ctx (same pattern as
    aa4f141/fd500b3/1a8b1fe). Sentinel 0.0 when any mask empty or any
    sub-norm underflows, matching 7972a98 safety pattern. FEATURE_NAMES
    80 → 81 triggers wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. Last 10+ iterations exhausted the
    classifier hyperparam axis (99081f5 capacity bump + 960113c revert
    matched baseline byte-identical; 17d4aec LR / db59c36 min_samples_leaf
    / df0ceec l2 all produced IDENTICAL combined=0.490700 suggesting
    US-505 does NOT auto-retrain on train_classifier.py-only edits).
    Every primary tunable saturated in both directions (THRESHOLD /
    MIN_SEP / STRIDE / DSP_MAX / DSP_SUM / DSP_PHASE all tested). The
    only path alive is features.py-sha-change (1a8b1fe feature-add
    produced distinct combined=0.503311, proving feature-count bump
    forces retrain).

    Genuinely NEW feature-space cell: PERCUSSIVE × SPEC_CONTRAST. Prior
    percussive-mask attempts used only CEPSTRAL (fd500b3 percussive_mfcc
    solo failed; aa4f141 voiced_percussive_mfcc_asymmetry failed) or
    PITCH-CLASS (1a8b1fe voiced_percussive_chroma_asymmetry failed).
    Both failure modes were content-specific: MFCC is phoneme-correlated
    on speech (consonant-plosive bursts dominate percussive MFCC → GBM
    global over-weight); chroma is FLAT on drums (drum broadband noise
    averages to uniform 12-bin distribution → percussive-chroma signal
    buried under drum noise). SPEC_CONTRAST is physically DIFFERENT from
    both — it measures peak-to-valley amplitude ratio PER BAND, a
    compressor/EQ/limiter fingerprint that is LESS phoneme-correlated
    than MFCC (contrast per band is mastering-chain-dependent, not
    phoneme-specific) AND has non-uniform drum signature (snare ~200Hz,
    kick ~60Hz, cymbals 5-10kHz produce distinctive per-band peak/valley
    profile on percussive frames).

    Mechanism targeting the 3 singing FPs at 7972a98 baseline
    (combined=0.589282, singing=0.345). Within one song the entire
    mastering chain is fixed: drum bus compressor + vocal bus compressor
    + mastering limiter are frozen across chord transitions → percussive-
    mean spec_contrast stable (cosine_dist ~0.05-0.15). Cross-song
    splice: different album's drum kit / different drum bus compression
    ratio / different mastering limiter threshold → per-band peak/valley
    profile on percussive frames shifts noticeably (cosine_dist
    ~0.25-0.60). Intra-song chord cycle (the 3 singing FPs): drum kit
    unchanged, same mastering → percussive_spec_contrast stable → feature
    sentinel-small. Real cross-song splice signature is LOUDER on the
    percussive mask than on voiced/unvoiced split because drum-bus
    processing is the most album-variable element of mastering chains.
    On speech: percussive mask = plosive bursts whose spec_contrast varies
    per phoneme (plosive at [p] vs [k] have different burst bands) BUT
    per-phoneme variance is UNCORRELATED with splice position → GBM
    assigns low per-domain SHAP on english/korean via tree splits, same
    self-gating mechanism as voiced_spec_contrast e7ca9eb on speech.

    Why single-mask (not asymmetry): voiced_unvoiced_spec_contrast_
    asymmetry (7972a98) already exists in-tree lifting spec_contrast
    axis via the voiced/unvoiced paired-difference route. An unpaired
    PERCUSSIVE version hits the mastering signal through a different
    frame population (transient-dominant) providing complementary GBM
    split ground: trees can combine `voiced_unvoiced_spec_contrast_
    asymmetry > T1` AND `percussive_spec_contrast_cosine_dist > T2` to
    specifically pick same-singer cross-song splices (large accompaniment
    mastering change visible through both lenses) vs intra-song chord
    cycle (both features near-zero).

    Orthogonal to every prior axis: NOT fd500b3 percussive_mfcc (cepstral
    timbre, phoneme-correlated noise mode on speech); NOT aa4f141
    voiced_percussive_mfcc_asymmetry (same cepstral noise, persists with
    pairing); NOT 1a8b1fe voiced_percussive_chroma_asymmetry (pitch-class
    content, drum-flat problem); NOT e7ca9eb voiced_spec_contrast (voiced
    mask, different frame population); NOT 7972a98 asymmetry (voiced+
    unvoiced pairwise diff); NOT de0be6f symmetry (pairwise sum); NOT
    0c3bf76 spec_flatness_asymmetry (scalar abs-delta formulation).
    FIRST percussive × spec_contrast content combination in any form,
    FIRST percussive mask against a non-cepstral non-pitch-class content
    axis. Pure features.py change — 1 new block function + 1
    FEATURE_NAMES append + 2 assert bumps (80 → 81) + 1 call in
    extract_features. ZERO new caches, ZERO new librosa calls. Per-t
    cost: 2 slices + 1 mask compare + 1 masked mean + 1 cosine on 7-dim
    vectors, sub-ms.

(c) IF THIS FAILS. (1) combined stuck near 0.50 with singing unchanged
    (percussive-mean spec_contrast still dominated by drum broadband
    pedestal, mastering shift invisible) → fallback to
    `harmonic_spec_contrast_cosine_dist` on the COMPLEMENT mask
    (bottom-60% onset-energy, sustained tonal content — captures vocal+
    bass mastering signature, orthogonal frame population to percussive).
    (2) combined regresses below 0.45 (percussive mask introduces global
    over-weighting like fd500b3 did on MFCC) → revert to paired form
    `voiced_percussive_spec_contrast_asymmetry` (percussive_dist − voiced_
    dist) for self-gating on speech. (3) combined matches 0.490700 or
    0.503311 EXACTLY with a feature-count change → confirms wrapper
    retrain-skip bug extends past hyperparam-only sites; escalate to
    operator with the identical-output streak now spanning physically-
    distinct feature changes.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 34+
    iterations — I cannot verify whether the 3 singing FPs sit in
    percussive-dominant windows (feature fires cleanly) or quiet
    between-chorus sections (feature inert). Every percussive-mask
    hypothesis theory-calibrated on physical mechanism. (ii) SHAP
    rollup STILL "no keeps yet — rollup empty" for 7972a98 despite
    14 keeps in loop history — rollup writer broken, no per-feature
    attribution available. (iii) 99081f5 capacity ghost (max_depth=4 /
    max_leaf_nodes=16) IN-tree at HEAD per grep; 960113c revert to 3/8
    was discarded by symmetry-semantics bug, restoring ghost. Absolute
    comparison to 0.589282 mixes with ghost but feature-add delta
    attribution is clean. (iv) 0.490700 streak: 5 hyperparam-only
    iterations produced IDENTICAL combined and per-domain — strongly
    suggests US-505 does NOT auto-retrain on train_classifier.py changes,
    only on features.py changes. Cannot verify from inside claude
    subprocess.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent 34+-iteration blocker, cited in every recent
    reflection. Per-FP (domain, file, t_sec, label_id, p_splice,
    dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_max, dsp_sum,
    voiced_spec_contrast_cosine_dist, voiced_unvoiced_spec_contrast_
    asymmetry, percussive_frame_ratio_pre, percussive_frame_ratio_post,
    top-5 |SHAP|). A single block transforms every feature-add and
    DSP-tunable hypothesis from theory-calibrated bet into data-driven
    decision. (2) US-505 COVERAGE FIX — extend the sha-gate to trigger
    auto-retrain when train_classifier.py sha changes, not just
    features.py sha. Diagnostic evidence: 5 consecutive
    train_classifier.py-only iterations (17d4aec, db59c36, df0ceec,
    and two others) all produced IDENTICAL combined=0.490700 while
    1a8b1fe feature-count bump produced distinct 0.503311. ~5-line
    extension of existing US-505 sha-gate in the wrapper. (3)
    DISCARD-REVERT SYMMETRY SEMANTICS bug fix — wrapper treats
    "combined matches baseline exactly" as no-improvement and reverts,
    which RESTORED the 99081f5 capacity ghost via 960113c. If a
    discarded commit's subject starts with "REVERT" and references
    a prior-discarded SHA, treat exact-match as KEEP not DISCARD.
    ~15 lines of git-commit-message parsing. Three highest-priority
    enhancements for unblocking the loop.

## 2026-04-20T16:50:53+09:00 — d3ddc9d (discard, combined=0.503051)
subject: add harmonic_spec_contrast_cosine_dist feature (FEATURE_NAMES 80->81) — sustained-frame mastering signature, COMPLEMENT of failed percussive mask. Harmonic mask = onset < median(onset)*0.8 (clear sustained / non-transient regime, structural complement of fd500b3/aa4f141/1a8b1fe/4773d1e percussive mask onset > median*1.2). 7-dim spec_contrast cosine on harmonic-mean vectors over pre[t-2,t] vs post[t,t+2]. Reuses cached feat_contrast + feat_onset — ZERO new librosa calls, ZERO new caches, one lazy feat_harm_threshold scalar. Sentinel 0.0 on empty mask or zero-norm. FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate. Targets 7972a98 current-keep (combined=0.589282) singing 0.345 weakest-domain plateau (3 clean FPs). Most recent iteration 4773d1e percussive_spec_contrast_cosine_dist failed catastrophically (singing 0.345->0.258) and its explicit cited fallback was harmonic-mask complement. Every prior percussive-mask attempt monotonically bad on singing: fd500b3 (0.319), aa4f141 (0.320), 1a8b1fe (0.319), 4773d1e (0.258) — drum hits add NOISE not signal because drum kits play different fills/patterns within one song so percussive-mean spec_contrast jitters within-song, washing out the cross-song mastering signal. The HARMONIC mask is structurally opposite frame population: sustained vocal notes + bass groove + ringing chords where within-song mastering is FROZEN (vocal bus compressor + master limiter operate identically on sustained content) and cross-song splices cross those mastering chains cleanly. Within-song chord transition: sustained spec_contrast stable (cosine_dist 0.05-0.15). Cross-song splice: vocal preamp / bus compression / mastering EQ shifts -> per-band peak/valley shifts noticeably (cosine_dist 0.25-0.50). Critical asymmetry: drum hits are TRANSIENT events that destabilize percussive-mean spectral statistics within-song; sustained frames are STATISTICALLY STABLE within-song because they're dominated by pitched material whose spectral envelope is mastering-bus-shaped not transient-shaped. Mastering chain has strong continuous effect on sustained content, weak/intermittent on transient content. Speech self-gating: harmonic mask = sustained vowel mid-portions whose spec_contrast captures formant peak/valley structure (speaker/recording-dependent, NOT phoneme-burst-correlated) — same self-gating that worked on e7ca9eb voiced_spec_contrast keep. Why threshold *0.8 (not *1.0): symmetric structural complement of percussive's *1.2; clear sustained-frame regime instead of marginal around-median set. Classifier hyperparam axis verifiably saturated (LR 17d4aec / min_samples_leaf db59c36 / l2 df0ceec all produced IDENTICAL combined=0.490700 — strong evidence US-505 doesn't auto-retrain on train_classifier.py-only edits). Every primary tunable saturated both directions. Only path alive is features.py-sha-change (1a8b1fe/0c3bf76/4773d1e all produced distinct combined values vs the 0.490700 streak). Orthogonal to every prior axis: NOT 4773d1e percussive_spec_contrast (COMPLEMENT mask — physically opposite frame population); NOT e7ca9eb voiced_spec_contrast (voiced=vp>0 mask, harmonic=onset<thr mask, mostly disjoint frame populations on singing); NOT 7972a98 voiced_unvoiced_spec_contrast_asymmetry (paired diff over voiced/unvoiced); NOT de0be6f symmetry; NOT 0c3bf76 spec_flatness asymmetry. FIRST HARMONIC-mask feature in any content axis. Pure features.py change — 1 new block + 1 FEATURE_NAMES append + 2 assert bumps + 1 call. ZERO new caches, ZERO new librosa calls. Per-t cost: 2 slices + 1 mask compare + 1 masked mean + 1 cosine on 7-dim vectors, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'harmonic_spec_contrast_cosine_dist', stationary 440Hz sine cos_dist=2e-6 (sustained tone correctly stable), 440->880 synthetic splice cos_dist=0.379 (~190000x discrimination), all 81 features finite on synthetic audio.
per-domain: combined_english=0.875000 combined_korean=0.537313 combined_singing=0.270769

# 2026-04-20 — hypothesis: add harmonic_spec_contrast_cosine_dist (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — new single-mask feature
    `harmonic_spec_contrast_cosine_dist` on the COMPLEMENT of the failed
    percussive mask. Harmonic mask = frames where
    `feat_onset < median(feat_onset) * 0.8` (bottom ~40-50% onset-energy =
    sustained / non-transient frames). 7-dim spec_contrast cosine distance
    between harmonic-mean vectors on pre=[t-2, t] vs post=[t, t+2].
    Reuses cached feat_contrast + feat_onset — ZERO new librosa calls,
    ZERO new caches; one lazy `feat_harm_threshold` scalar cached on ctx
    (same machinery the failed percussive blocks used). Sentinel 0.0 when
    mask empty or any sub-norm underflows. FEATURE_NAMES 80 → 81 forces
    wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. The MOST RECENT iteration (4773d1e
    percussive_spec_contrast_cosine_dist) failed catastrophically on
    singing (0.345 → 0.258, the worst regression in the recent window).
    Its explicit cited fallback (4773d1e/c1) was: "fallback to
    `harmonic_spec_contrast_cosine_dist` on the COMPLEMENT mask
    (bottom-60% onset-energy, sustained tonal content — captures
    vocal+bass mastering signature, orthogonal frame population to
    percussive)." Every prior percussive-mask attempt has been
    monotonically bad on singing: fd500b3 (singing 0.345 → 0.319),
    aa4f141 (0.320), 1a8b1fe (0.319), 4773d1e (0.258). The pattern
    confirms drum hits ADD noise rather than signal — drum kits play
    different fills/patterns within a single song, so percussive-mean
    spec_contrast jitters within-song (washing out the cross-song
    mastering signal). The HARMONIC mask is the structurally opposite
    frame population: sustained vocal notes + bass groove + ringing
    chords, where within-song mastering is FROZEN (vocal bus
    compressor + master limiter operate identically on sustained
    content) and cross-song splices cross those mastering chains
    cleanly. Classifier hyperparam axis is verifiably saturated (LR
    17d4aec, min_samples_leaf db59c36, l2 df0ceec all produced
    IDENTICAL combined=0.490700 — strong evidence US-505 doesn't
    auto-retrain on train_classifier.py-only edits). Every primary
    tunable saturated in both directions (THRESHOLD/MIN_SEP/STRIDE/
    DSP_MAX/DSP_SUM/DSP_PHASE). ONLY a features.py-sha-change forces
    a real retrain (1a8b1fe/0c3bf76/4773d1e all produced distinct
    combined values vs the 0.490700 streak).

    Mechanism targeting the 3 surviving singing FPs at 7972a98
    baseline. Within one song the entire mastering chain is fixed:
    vocal compressor + bus EQ + mastering limiter operate on the
    sustained vocal/bass/harmony content identically across the
    chord transitions that cause the FPs. Harmonic-mean spec_contrast
    pre/post chord transition = stable (cosine_dist 0.05-0.15).
    Cross-song splice: different vocal preamp / different bus
    compression / different mastering EQ → per-band peak/valley
    profile on sustained frames shifts noticeably (cosine_dist
    0.25-0.50). The CRITICAL DIFFERENCE from percussive mask: drum
    hits within a song vary heavily (verse vs chorus fills, ghost
    snare hits) so percussive-mean spec_contrast is HIGH-VARIANCE
    even within the same mastering chain → cross-song-vs-within-song
    discrimination drowned in within-song noise. Sustained frames
    have LOW within-song variance because they're dominated by
    pitched material whose spectral envelope is mastering-bus-shaped
    not transient-shaped. On speech: harmonic mask = sustained vowel
    mid-portions (vowel center where onset has decayed). spec_contrast
    on these frames captures formant peak/valley structure which is
    speaker/recording-dependent, NOT phoneme-burst-correlated. Same
    self-gating mechanism that worked on e7ca9eb's voiced_spec_contrast
    keep (singing → english 0.840-ish baseline preserved): GBM tree
    splits assign low per-domain SHAP on english/korean because the
    feature distribution is similar splice-vs-no-splice when phonemes
    dominate.

    Why HARMONIC over the failed PERCUSSIVE: physical asymmetry. Drum
    hits are TRANSIENT events that destabilize percussive-mean spectral
    statistics within-song; sustained frames are STATISTICALLY STABLE
    within-song. The mastering chain (compression, EQ, limiting) has
    a strong continuous effect on sustained content, weak/intermittent
    effect on transient content. The percussive mask accidentally
    suppressed mastering signal by amplifying transient noise; the
    harmonic mask does the opposite. Why threshold * 0.8 (not *1.0):
    matches the symmetric structural complement of percussive's
    *1.2; gives a clear sustained-frame regime instead of a marginal
    around-median set; smoke-supported by typical onset-strength
    distributions (median + ~20% gap separates clear sustained
    vs transient frames).

    Orthogonal to every prior axis. NOT 4773d1e percussive_spec_contrast
    (COMPLEMENT mask — physically opposite frame population). NOT
    e7ca9eb voiced_spec_contrast (voiced mask = vp>0; harmonic mask =
    onset<thr; mostly disjoint frame populations on singing — voiced
    captures sung notes but also vocal attacks, harmonic captures
    sustained content of ALL instruments including bass+ringing
    chords whether or not voice is active). NOT 7972a98
    voiced_unvoiced_spec_contrast_asymmetry (paired diff over
    voiced/unvoiced); NOT de0be6f spec_contrast symmetry; NOT
    0c3bf76 spec_flatness asymmetry (different content axis). FIRST
    HARMONIC-mask feature in any content axis. Pure features.py
    change — 1 new block function + 1 FEATURE_NAMES append + 2
    assert bumps + 1 call in extract_features. ZERO new caches, ZERO
    new librosa calls. Per-t cost: 2 slices + 1 mask compare + 1
    masked mean + 1 cosine on 7-dim vectors, sub-ms.

(c) IF THIS FAILS. (1) combined regresses below 0.50 (harmonic mask
    introduces speech-regression noise — sustained vowel mid-portions
    are MORE phoneme-correlated than I assumed) → fallback to a paired
    `voiced_harmonic_spec_contrast_asymmetry` (harmonic_dist −
    voiced_dist) for self-gating via paired differencing (the 1eda8e3
    pattern that lifted MFCC asymmetry +0.054). (2) combined matches
    0.490700 or 0.503311 EXACTLY despite a feature-count bump → confirms
    wrapper retrain-skip bug extends past hyperparam-only sites; escalate
    to operator. (3) combined improves modestly but singing remains
    stuck (drum noise wasn't the only confounder — chord transitions
    also shift sustained spec_contrast within-song because ringing
    chord overtones change) → pivot to `harmonic_chroma_cosine_dist`
    (harmonic mask × chroma — pitch-class profile on sustained frames
    isolates the chord-progression KEY signature, complementary to
    spec_contrast which is mastering-fingerprint).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 35+
    iterations — I cannot verify whether the 3 singing FPs sit in
    sustained-rich windows (feature fires cleanly on sustained vocal/
    bass/chord content) or transient-dominated windows (mask too
    sparse → sentinel 0.0 → feature inert). Every harmonic / percussive
    mask hypothesis remains theory-calibrated. (ii) SHAP rollup STILL
    "no keeps yet — rollup empty" for 7972a98 despite 14 keeps in loop
    history — rollup writer broken, no per-feature attribution
    available to predict harmonic-mask fire-rate. (iii) 99081f5
    classifier capacity ghost (max_depth=4, max_leaf_nodes=16) is
    in-tree at HEAD — feature-add delta attribution is clean (only
    one new feature) but absolute comparison to 0.589282 mixes with
    ghost. (iv) The 0.490700 streak across 5 hyperparam-only iterations
    strongly suggests US-505 doesn't auto-retrain on
    train_classifier.py-only changes — features.py-sha bumps are the
    only forcing function for retrain.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE — persistent 35+-iteration blocker, cited in every
    recent reflection. Per-FP (domain, file, t_sec, label_id, p_splice,
    dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_max, dsp_sum,
    voiced_spec_contrast_cosine_dist,
    voiced_unvoiced_spec_contrast_asymmetry,
    harmonic_frame_ratio_pre, harmonic_frame_ratio_post,
    top-5 |SHAP|). Single block transforms every harmonic / percussive
    / voiced-mask hypothesis from theory-calibrated bet into
    data-driven decision. (2) US-505 COVERAGE FIX — extend the
    sha-gate to trigger auto-retrain when train_classifier.py sha
    changes, not just features.py sha. Diagnostic evidence: 5
    consecutive train_classifier.py-only iterations all produced
    IDENTICAL combined=0.490700 while feature-count bumps produced
    distinct values. ~5-line extension of the existing US-505
    sha-gate in the wrapper. (3) DISCARD-REVERT SYMMETRY SEMANTICS
    bug fix — wrapper treats "combined matches baseline exactly" as
    no-improvement and reverts; this RESTORED the 99081f5 capacity
    ghost via 960113c. If a discarded commit's subject starts with
    "REVERT" and references a prior-discarded SHA, treat exact-match
    as KEEP not DISCARD. ~15 lines of git-commit-message parsing.

## 2026-04-20T17:10:22+09:00 — d49284c (discard, combined=0.539430)
subject: add voiced_unvoiced_mfcc_asymmetry_far feature (FEATURE_NAMES 80->81) — FAR-window companion to 1eda8e3 (kept +0.054 biggest win). Same paired-difference template, post=[t+4,t+8] instead of [t,t+2]. Reuses cached feat_mfcc + feat_vp, ZERO new librosa calls. Edge guard t+8>duration_s OR t-2<0 returns sentinel 0.0. Targets 7972a98 baseline (combined=0.589282) singing 0.345 weakest-domain plateau (3 chord-cycle FPs surviving NEAR-window asymmetry which is ~0 on within-song chord cycle). Mechanism: within-song chord cycle voiced_far ~0.20 (different vowels for new lyrics) but unvoiced_far ~0.05 (same drums/mastering across 6s within song) -> asymmetry NEGATIVE -0.15. Same-singer cross-song splice voiced_far ~0.10 (similar vocal tract) but unvoiced_far ~0.30-0.50 (new drums + new mastering) -> asymmetry POSITIVE +0.15 to +0.40. SIGN FLIP between chord-cycle FP (NEG) and cross-song splice (POS) is binary discriminator GBM cannot synthesize from existing NEAR features. Trees split on (near_asym>0.15 AND far_asym>0 -> splice) vs (near_asym~0 AND far_asym<0 -> chord cycle). Speech self-gating: voiced+unvoiced far-window MFCC both co-vary with new sentence content -> diff is zero-mean noise -> GBM low SHAP on english/korean (1eda8e3 proven mechanism). Classifier hyperparam axis verifiably broken (5 train_classifier.py-only iterations all produced IDENTICAL combined=0.490700 -> US-505 doesn't auto-retrain on hyperparam edits); only features.py-sha bumps force retrain (1a8b1fe/0c3bf76/4773d1e/d3ddc9d all distinct values). Last 5 mask experiments all regressed singing catastrophically (4773d1e percussive_spec_contrast 0.345->0.258, d3ddc9d harmonic_spec_contrast 0.270, 1a8b1fe voiced_percussive_chroma 0.319, fd500b3/aa4f141 0.319-0.320). Pivot back to working voiced/unvoiced asymmetry family with new TIME SCALE rather than yet another mask. Orthogonal: NOT 1eda8e3 (NEAR window same template); NOT 7972a98 (spec_contrast not cepstral); NOT f4148cc (chroma asym NEAR); NOT 0c3bf76 (1D scalar abs-delta); NOT ff65865 voiced_chroma_far (single-mask no asymmetry); NOT 32cac36/49bd0b1/308aa5a/percussive/harmonic mask family. FIRST FAR-window asymmetry feature in any content axis. Pure features.py change — 1 new block (~50 lines, template-cloned from 1eda8e3 with edge guard) + 1 FEATURE_NAMES append + 2 assert bumps + 1 call. ZERO new caches, ZERO new librosa calls. Per-t cost 4 slices + 4 masked means + 2 cosines on 13-dim vectors + 1 subtraction, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_mfcc_asymmetry_far', edge guard fires correctly (t=5 with 12s chunk -> 0.0; t=1 with t-2<0 -> 0.0), in-range t=2.5 yields finite value 0.011, all 81 features finite, idempotent on repeated calls.
per-domain: combined_english=0.825000 combined_korean=0.537313 combined_singing=0.354098

# 2026-04-20 — hypothesis: add voiced_unvoiced_mfcc_asymmetry_far (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — FAR-window companion to
    1eda8e3's voiced_unvoiced_mfcc_asymmetry (kept +0.054, biggest win
    in loop history). Same multi-dim cosine + paired-difference template,
    but post window is FAR =[t+4, t+8] instead of NEAR =[t, t+2]. Pre stays
    [t-2, t]. Reuses cached feat_mfcc + feat_vp — ZERO new librosa calls,
    ZERO new caches. Edge guard: t+8 > duration_s returns sentinel 0.0
    (matches ff65865 pattern). Sentinel 0.0 also on empty masks / zero
    norm. FEATURE_NAMES 80→81 forces wrapper auto-retrain via US-505
    sha gate (provably the only forcing function — 5+ classifier
    hyperparam-only iterations all produced IDENTICAL combined=0.490700).

(b) WHY this over recent failures. Last 10+ iterations exhausted: 5 mask
    variants on percussive/harmonic axis (all regressed singing
    catastrophically: 4773d1e percussive_spec_contrast 0.345->0.258,
    d3ddc9d harmonic_spec_contrast 0.270, 1a8b1fe voiced_percussive_chroma
    asym 0.319, fd500b3/aa4f141 percussive_mfcc 0.319-0.320), 1D-scalar
    voiced/unvoiced asymmetries (0c3bf76 spec_flatness 0.319), classifier
    hyperparam axis verifiably broken via US-505 retrain bug (5 iterations
    at IDENTICAL 0.490700), all primary tunables saturated both directions.
    The voiced/unvoiced ASYMMETRY family at NEAR window is the most
    productive axis (1eda8e3 +0.054, 7972a98 +0.005); FAR-window companion
    is genuinely untried and pivots back to the working axis with a new
    time scale, not yet another mask experiment.

    Mechanism targeting the 3 surviving singing FPs (within-song chord
    cycle). NEAR asymmetry (1eda8e3) at chord cycle is ~0 because both
    voiced & unvoiced MFCC are stable across the +-2s window — neither
    voice nor accompaniment shifts on the timescale that bites NEAR.
    GBM cannot tell chord-cycle FP (near=0) from real same-singer
    cross-song splice (near=+0.25) at the borderline NEAR values where
    the 3 FPs sit. FAR window (t-2..t vs t+4..t+8) introduces a
    different signature:
    - Within-song chord cycle: at [t+4,t+8] singer is on a DIFFERENT
      chord and singing different vowels for new lyrics -> voiced_far
      MFCC differs significantly (~0.20). Unvoiced (drums + same
      mastering) is nearly identical across 6s within one song ->
      unvoiced_far_dist ~0.05. Asymmetry = unvoiced_far - voiced_far
      ~ -0.15 NEGATIVE.
    - Real same-singer cross-song splice: at [t+4,t+8] we're 4-8s
      into the new song. Drums + mastering totally different ->
      unvoiced_far_dist ~0.30-0.50. Singer tract similar ->
      voiced_far_dist ~0.10-0.15. Asymmetry ~ +0.15 to +0.40 POSITIVE.
    - Different-singer cross-song: both shift -> asymmetry near 0 or
      positive. Existing NEAR asymmetry catches this case.

    The SIGN FLIP between within-song chord cycle (NEGATIVE) and
    same-singer cross-song splice (POSITIVE) is a clean binary
    discriminator GBM cannot synthesize from existing features. Trees
    can split on (near_asym > 0.15 AND far_asym > 0 -> splice) vs
    (near_asym ~ 0 AND far_asym < 0 -> chord cycle).

    Self-gating on speech via paired differencing (proven mechanism
    that lifted english 0.732->0.889 and korean 0.526->0.667 on 1eda8e3):
    voiced (vowels) and unvoiced (consonants) far-window MFCC both
    co-vary with new sentence content uncorrelated with splice
    position -> difference is zero-mean noise -> GBM low per-domain
    SHAP on english/korean. Same self-gating shielded the original
    voiced_unvoiced_mfcc_asymmetry on speech.

    Why FAR over yet another mask. Every harmonic/percussive mask
    attempt has failed monotonically; voiced/unvoiced is the working
    family. The TIME-SCALE companion is the orthogonal axis
    untouched by the asymmetry family — ff65865 voiced_chroma_far
    failed but as a SINGLE-MASK chroma feature; paired-difference
    asymmetry (the proven 1eda8e3 mechanism) on a far window has
    never been combined.

    Orthogonal to every prior axis: NOT 1eda8e3 (NEAR window — same
    template at different time scale); NOT 7972a98 (spec_contrast not
    cepstral); NOT f4148cc (chroma asymmetry, NEAR); NOT 0c3bf76
    (1D scalar abs-delta, flatness); NOT ff65865 voiced_chroma_far
    (single-mask, no asymmetry); NOT 32cac36 voiced_chroma; NOT
    49bd0b1 voiced_mfcc; NOT 308aa5a tonnetz; NOT any percussive /
    harmonic mask variant. FIRST FAR-window asymmetry feature in any
    content axis. Pure features.py change — 1 new block (~50 lines,
    template-cloned from 1eda8e3 with edge guard) + 1 FEATURE_NAMES
    append + 2 assert bumps + 1 call in extract_features. ZERO new
    librosa calls, ZERO new caches. Per-t cost: 4 slices + 4 masked
    means + 2 cosines on 13-dim vectors + 1 subtraction, sub-ms.

(c) IF THIS FAILS. (1) Within-song singer's voiced MFCC IS stable
    across 4-8s (sustained held notes during chorus -> voiced_far_dist
    small instead of expected ~0.20) -> asymmetry on chord cycle is
    near 0 like NEAR, sign-discrimination collapses -> fallback is
    voiced_unvoiced_spec_contrast_asymmetry_far (mastering signature
    is drum-bus-fixed within song so unvoiced_far stays small even
    when voiced_far is small from sustained vocals -> asymmetry stays
    near 0; cross-song unvoiced_far jumps via mastering shift while
    voiced stays low -> asymmetry positive). (2) Edge-guard sentinel
    fires too often (t+8 > chunk_dur on most chunks since
    ANALYSIS_STRIDE_S=0.12 dense scan reaches near chunk end) ->
    feature mostly returns 0 and GBM ignores it -> shrink the far
    offset to [t+3, t+6] (less dense but still beats NEAR's
    chord-cycle blind spot). (3) Final escalation: voiced_unvoiced_
    f0_jitter_asymmetry — pivot from cepstral envelope to vocal F0
    stability axis (cited untried in 0c3bf76(c)(3); F0 is intrinsic
    to vocal tract / singer identity, untouched by asymmetry family).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 36+
    iterations — I cannot verify the 3 surviving singing FPs' actual
    t_sec positions to predict whether t+8 > chunk_dur fires the edge
    guard on most of them. If FPs cluster near chunk ends (likely
    because eval files are 30-120s and FPs accumulate over duration),
    feature is mostly inert. Every FAR-window hypothesis remains
    theory-calibrated. (ii) SHAP rollup STILL "no keeps yet — rollup
    empty" for 7972a98 despite 14 keeps — rollup writer broken, no
    per-feature attribution available. (iii) IDENTICAL-COMBINED
    streak across 5 train_classifier.py-only iterations strongly
    suggests US-505 doesn't auto-retrain on hyperparam edits — only
    features.py-sha bumps force retrain. My +1 feature-count change
    is the safe path. (iv) 99081f5 capacity ghost in-tree
    (max_depth=4, max_leaf_nodes=16) — feature-add delta clean,
    absolute comparison to 0.589282 mixes with ghost.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent 36+-iteration blocker. Per-FP (domain, file,
    t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, top-5 |SHAP|). chunk_
    duration_s explicitly tells me whether FAR-window edge guard
    fires on each FP. (2) US-505 COVERAGE FIX — extend retrain sha
    gate to include train_classifier.py changes, not just features.py.
    Diagnostic: 5 consecutive train_classifier.py-only iterations
    (17d4aec, db59c36, df0ceec + 2 others) all produced IDENTICAL
    combined=0.490700 while 1a8b1fe / 0c3bf76 / 4773d1e / d3ddc9d
    feature-count changes produced distinct values. ~5-line shell
    extension of existing US-505 sha-gate. (3) DISCARD-REVERT
    SYMMETRY SEMANTICS bug fix — wrapper treats "matches baseline
    exactly" as no-improvement, restoring the 99081f5 capacity ghost
    via 960113c discard. If commit subject starts "REVERT" + names a
    prior-discarded SHA, treat exact-match as KEEP. ~15 lines git
    commit-message parse.

## 2026-04-20T17:30:06+09:00 — ff88b22 (discard, combined=0.500909)
subject: add voiced_unvoiced_mfcc_asymmetry_wide feature (FEATURE_NAMES 80->81) — WIDE-ADJACENT-window companion to 1eda8e3 NEAR ±2s (kept +0.054 biggest win) and d49284c FAR ±2s-pre/+4-+8-post (failed because gap broke speech self-gating). Pre=[t-4,t] adjacent post=[t,t+4], same paired-difference template, ZERO new librosa calls. Edge guard t-4<0 OR t+4>duration_s returns sentinel 0.0. Last iteration d49284c LIFTED singing 0.345->0.354 (only recent feature to do so) but speech regressed sharply (korean 0.667->0.537, english 0.889->0.825) because the 6s GAP between pre and post sub-windows on speech caused them to sample different sentences/topics — voiced+unvoiced MFCC shifts no longer co-vary with same phoneme context, breaking paired-difference self-gating. WIDE eliminates the gap while expanding the window 2x over NEAR. Mechanism on 3 surviving singing FPs (within-song chord cycle): NEAR ±2s asymmetry at chord boundary ~0 because both voiced+unvoiced shift together with chord. WIDE ±4s averages OVER multiple chord transitions in pre AND in post — chord-cycle signal averages OUT (mean of multiple chord shifts cancels) -> asymmetry stays near 0 on chord-cycle FP. Real cross-song splice: pre is 4s of song A's mastering, post is 4s of song B's mastering. Unvoiced sub-window (drum/mastering) distance large; voiced sub-window (vocal tract) smaller -> asymmetry strongly positive. Sign separation chord-cycle (~0) vs cross-song (positive) preserved AND CLEANER than NEAR because NEAR's chord-cycle noise gets averaged away. Speech self-gating preserved by adjacency: ADJACENT ±4s windows on speech sample similar broad phoneme distribution (4s ≈ 6-10 syllables, broad enough to span typical phoneme distribution); both voiced and unvoiced sub-windows on each side contain similar mix of vowels and consonants; differences dominated by speaker/recording continuity not phoneme drift -> diff near zero -> low GBM SHAP on english/korean. Same self-gating that worked at NEAR works at WIDE because no gap = same context similarity. Voiced/unvoiced asymmetry family is provably the most productive axis: 1eda8e3 +0.054 (biggest single win) and 7972a98 +0.005 (small but consistent). Time-scale variation on this proven template is the natural next axis after content/mask exhaustion. Last 10+ iterations exhausted: classifier hyperparam axis verifiably broken (5 train_classifier.py-only iterations all produced IDENTICAL combined=0.490700 -> US-505 doesn't auto-retrain on hyperparam edits); every primary tunable saturated both directions; mask experiments (percussive/harmonic) all monotonically failed; 1D-scalar abs-delta (0c3bf76 spec_flatness) failed; all far-window single-mask companions (ff65865 voiced_chroma_far) failed. Only path alive is features.py-sha-change (1a8b1fe/0c3bf76/4773d1e/d3ddc9d/d49284c all produced distinct combined values vs the 0.490700 streak). Orthogonal: NOT 1eda8e3 (NEAR window — same template at 2x time scale); NOT d49284c (FAR window has a gap — WIDE is adjacent); NOT 7972a98 (spec_contrast not cepstral); NOT f4148cc (chroma asymmetry); NOT 0c3bf76 (1D scalar abs-delta); NOT ff65865 (single-mask far). FIRST WIDE-ADJACENT-window asymmetry feature in any content axis. Pure features.py change — 1 new block (~50 lines, template-cloned from 1eda8e3 with edge guard) + 1 FEATURE_NAMES append + 2 assert bumps + 1 call. ZERO new librosa calls, ZERO new caches. Per-t cost: 4 slices + 4 masked means + 2 cosines on 13-dim vectors + 1 subtraction, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_mfcc_asymmetry_wide', edge guard fires correctly (t-4<0 OR t+4>duration_s -> 0.0), synthetic cross-source asymmetry=2.0 large positive, synthetic within-source asymmetry=0.0 sentinel, end-to-end extract_features returns 81 finite features on synthetic audio, idempotent on repeated calls.
per-domain: combined_english=0.765432 combined_korean=0.555882 combined_singing=0.295385

# 2026-04-20 — hypothesis: add voiced_unvoiced_mfcc_asymmetry_wide (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — WIDER-ADJACENT-window
    companion to 1eda8e3's voiced_unvoiced_mfcc_asymmetry. Same
    paired-difference template, but pre=[t-4, t] and post=[t, t+4]
    (adjacent ±4s windows instead of ±2s). Reuses cached feat_mfcc +
    feat_vp — ZERO new librosa calls, ZERO new caches. Edge guard
    t-4 < 0 OR t+4 > duration_s returns sentinel 0.0. Sentinel 0.0
    also on empty masks / zero-norm. FEATURE_NAMES 80 → 81 forces
    wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. Last iteration d49284c (FAR window
    pre=[t-2,t] post=[t+4,t+8]) actually LIFTED singing 0.345→0.354
    (the only recent feature to do so) but speech regressed sharply
    (korean 0.667→0.537, english 0.889→0.825). Diagnosis: the GAP
    between pre and post on speech caused pre and post sub-windows to
    sample different sentences/topics — voiced and unvoiced mfcc
    shifts no longer co-vary with same phoneme context, so the
    paired-difference self-gating broke. WIDE ±4s eliminates the gap
    while still expanding the window 2x over NEAR — captures 2 chord
    cycles in singing (averaging out within-song chord-cycle
    asymmetry while preserving cross-song mastering shift) AND
    preserves speech self-gating because adjacent ±4s windows still
    sample similar broad phoneme distribution.

    Mechanism on the 3 surviving singing FPs (within-song chord
    cycle). NEAR ±2s asymmetry (1eda8e3) at chord boundary = ~0
    because both voiced+unvoiced shift together. WIDE ±4s averages
    OVER multiple chord transitions in pre and OVER multiple in post
    — chord-cycle signal averages OUT (mean of multiple chord shifts
    cancels) → asymmetry stays near 0 on chord-cycle FP. Real
    cross-song splice: pre is 4s of song A's mastering, post is 4s
    of song B's mastering. Unvoiced sub-window (drum/mastering)
    distance is large; voiced sub-window (vocal tract) smaller →
    asymmetry strongly positive. Sign separation of chord-cycle (~0)
    vs cross-song (positive) preserved AND CLEANER than NEAR because
    NEAR's chord-cycle noise gets averaged away.

    Speech self-gating preserved by adjacency. ADJACENT ±4s windows
    on speech sample similar broad phoneme distribution (4s of
    speech ≈ 6-10 syllables, broad enough to span typical phoneme
    distribution). Both voiced and unvoiced sub-windows on each side
    contain similar mix of vowels and consonants. Differences between
    pre and post sub-windows are dominated by speaker/recording
    continuity not phoneme content drift → voiced and unvoiced both
    near-zero on within-source speech → asymmetry near zero → low
    GBM SHAP on english/korean. Same self-gating that worked at NEAR
    works at WIDE because no gap = same context similarity.

    Why WIDE over ad-hoc fallback options: (i) [t+3, t+6] gap from
    d49284c(c)(2) keeps the gap — same speech-regression failure
    mode. (ii) Yet another mask is exhausted (5+ percussive/harmonic
    failures). (iii) WIDE is genuinely untried time-scale on the
    PROVEN axis. The voiced/unvoiced asymmetry family has the
    biggest win in loop history (1eda8e3 +0.054 keep) plus a small
    keep (7972a98 +0.005); time-scale variation on this template is
    the natural next axis after content/mask exhaustion.

    Orthogonal to every prior axis: NOT 1eda8e3 (NEAR window — same
    template at 2x time scale); NOT d49284c (FAR window has a gap —
    WIDE is adjacent); NOT 7972a98 (spec_contrast not cepstral); NOT
    f4148cc (chroma asymmetry); NOT 0c3bf76 (1D scalar abs-delta);
    NOT ff65865 voiced_chroma_far (single-mask, no asymmetry);
    NOT any percussive / harmonic mask variant (different mask).
    FIRST WIDE-window asymmetry feature in any content axis. Pure
    features.py change — 1 new block (~50 lines, template-cloned
    from 1eda8e3 with edge guard) + 1 FEATURE_NAMES append + 2
    assert bumps + 1 call. ZERO new librosa calls, ZERO new caches.
    Per-t cost: 4 slices + 4 masked means + 2 cosines on 13-dim
    vectors + 1 subtraction, sub-ms.

(c) IF THIS FAILS. (1) singing improvement preserved but speech
    still regresses (broad phoneme context isn't enough to self-gate
    at ±4s — speech is sentence-structured at 4s scale) → fallback
    to MID ±3s window pre=[t-3,t] post=[t,t+3] (smaller deviation
    from proven NEAR ±2s, less averaging power on chord cycle but
    safer self-gating on speech). (2) singing flatter than d49284c
    (averaging dilutes splice-side asymmetry too — cross-song splice
    only spans 4s of new material, half of post window dominated by
    same-source if cut is mid-window) → pivot to F0-stability
    asymmetry voiced_f0_jitter_asymmetry (cited untried in
    0c3bf76(c)(3); pitch jitter character is recording-dependent
    and mastering-independent — orthogonal to cepstral). (3)
    combined matches 0.490700 EXACTLY → confirms wrapper
    retrain-skip bug now extends past hyperparam-only sites; escalate
    to operator.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 37+
    iterations — I cannot verify the 3 surviving singing FPs' actual
    t_sec positions. If FPs cluster within 4s of file ends, edge
    guard fires on most of them and feature is sentinel-inert. Every
    WIDE/FAR window hypothesis remains theory-calibrated. (ii) SHAP
    rollup STILL "no keeps yet — rollup empty" for 7972a98 despite
    14 keeps in loop history — rollup writer broken. (iii) 99081f5
    capacity ghost in-tree (max_depth=4, max_leaf_nodes=16) — feature
    delta clean, absolute comparison mixes with ghost. (iv) chunk
    duration distribution not exposed — for 30s eval chunks the
    edge-guard fires on ~25% of analysis positions; for 120s chunks
    only ~6%. Cannot verify whether singing FPs land in the
    edge-guard zone.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE — persistent 37+-iteration blocker. Per-FP
    (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z,
    dsp_cpe_z, dsp_max, dsp_sum, chunk_duration_s,
    voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, top-5 |SHAP|).
    chunk_duration_s explicitly tells me whether WIDE/FAR-window
    edge guards fire on each FP. (2) US-505 COVERAGE FIX — extend
    retrain sha gate to include train_classifier.py changes, not just
    features.py. Diagnostic evidence: 5+ consecutive
    train_classifier.py-only iterations all produced IDENTICAL
    combined=0.490700 while feature-count bumps produced distinct
    values. ~5-line shell extension of existing US-505 sha-gate.
    (3) DISCARD-REVERT SYMMETRY SEMANTICS bug fix — wrapper treats
    "matches baseline exactly" as no-improvement, restoring the
    99081f5 capacity ghost via 960113c discard. If commit subject
    starts "REVERT" + names a prior-discarded SHA, treat exact-match
    as KEEP. ~15 lines git commit-message parse.

## 2026-04-20T17:58:23+09:00 — d290101 (discard, combined=0.558329)
subject: add voiced_unvoiced_spec_contrast_asymmetry_far feature (FEATURE_NAMES 80->81) — FAR-gap companion to 7972a98 NEAR spec_contrast asymmetry (kept +0.005), pre[t-2,t] post[t+3,t+6] (3s gap, 3s span). Reuses cached feat_contrast + feat_vp + feat_audio for duration check, ZERO new librosa calls, ZERO new caches. Edge guard t-2<0 OR t+6>duration_s returns sentinel 0.0. d49284c voiced_unvoiced_mfcc_asymmetry_far ([t+4,t+8]) was the ONLY recent feature to LIFT singing 0.345->0.354 proving FAR-gap mechanism is real for chord-cycle vs cross-song discrimination, but speech regressed sharply (korean 0.667->0.537, english 0.889->0.825) because MFCC heavily phoneme-correlated. ff88b22 WIDE-adjacent ±4s no-gap failed singing 0.295 confirming gap-FAR is the structure that works for singing. Spec_contrast = per-band peak-to-valley ratio is a MASTERING/COMPRESSOR/EQ fingerprint, less phoneme-correlated than MFCC because peak-RATIO is more stable across phoneme content than absolute cepstral envelope. Mechanism on 3 surviving singing FPs: cross-song splice voiced_far_dist low (similar singer formants), unvoiced_far_dist HIGH (drum-bus + master EQ + limiter shift) -> asymmetry POSITIVE; intra-song chord cycle voiced_far_dist moderate (vowel formants vary across t+3..t+6), unvoiced_far_dist small (mastering frozen, drum kit identical) -> asymmetry NEGATIVE; sign-flip is binary discriminator chord cycle cannot mimic. Speech self-gating: both voiced and unvoiced spec_contrast carry SAME mastering baseline (recording continuous, cancels) plus phoneme-correlated peak/valley variation; spec_contrast peak-RATIO is more phoneme-stable than MFCC cepstral envelope so asymmetry of two phoneme-driven distances is smaller. Post=[t+3,t+6] (3s gap 3s span) is narrower than d49284c [t+4,t+8] (4s/4s) - cited d49284c(c)(2) reduction, less phoneme drift while preserving chord-cycle averaging (chord period in pop ~2-3s so 3s post-window spans ~1-1.5 chords). Total span [t-2,t+6]=8s fits within 30s eval chunks (sentinel rate ~10%% on 30s chunks). Voiced/unvoiced spec_contrast asymmetry family is provably productive (7972a98 NEAR kept +0.005); FAR-gap companion on this content axis is genuinely untried and the natural cited fallback after d49284c proved the FAR-time-scale mechanism. Orthogonal: NOT 7972a98 (NEAR window — same template at FAR time scale); NOT d49284c (MFCC content axis, this is spec_contrast); NOT ff88b22 (WIDE adjacent no gap); NOT 1eda8e3 (NEAR MFCC); NOT f4148cc (chroma asymmetry); NOT 0c3bf76 (1D scalar flatness abs-delta); NOT ff65865 voiced_chroma_far (single-mask no asymmetry); NOT any percussive/harmonic/unvoiced-only mask variant. FIRST FAR-gap asymmetry on the spec_contrast content axis. Pure features.py change — 1 new block (~50 lines, template-cloned from 7972a98 with edge guard + 3s gap) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call. Per-t cost: 4 slices + 4 masked means + 2 cosines on 7-dim vectors + 1 subtraction, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_spec_contrast_asymmetry_far', all 81 features finite on synthetic speech-like audio (voiced+unvoiced alternating with formant shift at t=10), edge guard fires correctly (t=18 with post>20s -> 0.0; t=1 with t-2<0 -> 0.0), idempotent on repeated calls, splice probe yields -0.0445 vs within-source +0.0031 (non-trivial discrimination on synthetic).
per-domain: combined_english=0.897436 combined_korean=0.666667 combined_singing=0.290909

# 2026-04-20 — hypothesis: add voiced_unvoiced_spec_contrast_asymmetry_far (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — FAR-window companion
    to 7972a98's NEAR voiced_unvoiced_spec_contrast_asymmetry (kept
    +0.005). Same paired-difference template, but post = [t+3, t+6]
    (3s gap, 3s span) instead of NEAR [t, t+2]. Pre stays [t-2, t].
    Reuses cached feat_contrast + feat_vp — ZERO new librosa calls,
    ZERO new caches. Edge guard t+6 > duration_s OR t-2 < 0 returns
    sentinel 0.0. Sentinel 0.0 also on empty mask / zero-norm.
    FEATURE_NAMES 80→81 forces wrapper auto-retrain via US-505.

(b) WHY this over recent failures. d49284c voiced_unvoiced_mfcc_
    asymmetry_far (pre[t-2,t] post[t+4,t+8]) was the ONLY recent
    feature to LIFT singing (0.345 → 0.354). The FAR-time-scale
    mechanism for chord-cycle vs cross-song discrimination is
    proven real. The problem was speech regression (korean
    0.667→0.537, english 0.889→0.825) because MFCC is heavily
    phoneme-correlated and the gap caused pre/post to sample
    different sentence content → voiced+unvoiced MFCC shifts no
    longer co-vary with same phoneme context → paired-difference
    self-gating broke. ff88b22 MFCC WIDE-adjacent ±4s no-gap
    FAILED on singing (0.295) proving the gap-and-FAR is exactly
    the structure that works for singing, but the no-gap WIDE
    averages out signal. The fix is keep the gap-FAR structure
    but pivot to a content axis that is LESS phoneme-correlated
    than MFCC so speech self-gating survives.

    Spec_contrast = per-band peak-to-valley amplitude ratio is a
    MASTERING/COMPRESSOR/EQ fingerprint. On speech it carries
    SOME phoneme correlation (vowel formants give per-band peaks)
    but the dominant component is the recording's mastering
    chain. On singing, mastering chain is the most album-variable
    element of cross-song splices. So spec_contrast is a content
    axis where:
    - Singing cross-song: voiced_far_dist (similar singer
      formants if same singer, modest if different) low/moderate;
      unvoiced_far_dist (drum-bus + master EQ + limiter all
      shift) HIGH → asymmetry strongly POSITIVE.
    - Singing intra-song chord cycle (the 3 surviving FPs):
      voiced_far_dist (vowel formants vary across t+3..t+6 with
      new chord/lyric) MODERATE; unvoiced_far_dist (mastering
      frozen, drum kit identical) SMALL → asymmetry NEGATIVE.
    - SIGN FLIP between FP and TP — exact discriminator chord
      cycle cannot mimic.
    - Speech voiced_far_dist (vowel formant per-band peak shift
      across 3s gap) and unvoiced_far_dist (consonant noise
      per-band peak shift across 3s gap) BOTH carry the SAME
      mastering-fixed baseline (recording continuous) AND BOTH
      carry phoneme-correlated peak/valley variation. The
      MASTERING component cancels (same recording). The phoneme
      component does NOT cancel as cleanly as MFCC because
      spec_contrast is per-band peak-RATIO not per-band amplitude
      — RATIO is more stable across phoneme content than absolute
      cepstral envelope. Asymmetry of two phoneme-driven
      distances should be smaller than for MFCC.

    Why this over alternatives:
    (1) MID ±3s adjacent (cited d49284c(c)(2) NEAR-fallback) —
    smaller deviation from proven NEAR ±2s but no FAR-time-scale
    signal; would be a pure interpolation between 1eda8e3 NEAR
    (kept) and ff88b22 WIDE (failed) without exploiting the
    FAR-gap mechanism that d49284c proved.
    (2) F0 jitter asymmetry — unvoiced frames don't have F0 so
    voiced/unvoiced asymmetry doesn't apply; would need a
    different template.
    (3) Spec_flatness FAR asymmetry — 0c3bf76 NEAR already failed
    (singing 0.319), so flatness axis has weaker signal than
    spec_contrast axis (7972a98 +0.005 keep proves contrast NEAR
    works).

    Why post=[t+3, t+6] specifically. d49284c [t+4, t+8] gives
    4s gap + 4s span. Cited d49284c(c)(2) fallback was [t+3, t+6]
    = 3s gap + 3s span. Smaller gap, smaller window — both
    reduce speech phoneme drift while preserving the chord-
    cycle-averaging mechanism (chord period in pop is ~2-3s, so
    3s post-window still spans ~1-1.5 chord cycles for averaging).
    Edge guard total span [t-2, t+6] = 8s fits within 30s eval
    chunks comfortably (sentinel rate ~10% on 30s chunks).

    Orthogonal to every prior axis: NOT 7972a98 (NEAR window —
    same template at FAR time scale); NOT d49284c (MFCC content
    axis, this is spec_contrast); NOT ff88b22 (WIDE adjacent no
    gap); NOT 1eda8e3 (NEAR MFCC); NOT f4148cc (chroma
    asymmetry); NOT 0c3bf76 (1D scalar flatness abs-delta);
    NOT ff65865 (single-mask far chroma); NOT any percussive /
    harmonic / unvoiced-only mask variant. FIRST FAR-gap
    asymmetry on the spec_contrast content axis. Pure features.py
    change — 1 new block (~50 lines, template-cloned from 7972a98
    with edge guard) + 1 FEATURE_NAMES append + 2 assert bumps
    + 1 call. ZERO new caches, ZERO new librosa calls. Per-t
    cost: 4 slices + 4 masked means + 2 cosines on 7-dim
    vectors + 1 subtraction, sub-ms.

(c) IF THIS FAILS. (1) Singing improvement preserved but
    speech still regresses (spec_contrast IS more phoneme-
    correlated at FAR than I assumed; per-band peak-RATIO
    isn't immune to phoneme drift across the 3s gap) →
    fallback to MID ±3s adjacent template (no gap, smaller
    deviation from proven NEAR — cited d49284c(c)(2) /
    ff88b22(c)(1)). (2) Singing flatter than d49284c (3s
    post-window dilutes splice-side asymmetry; signal needed
    the wider 4s window) → expand post to [t+3, t+7] (4s span
    same gap). (3) combined matches 0.490700 EXACTLY despite
    feature-count change → wrapper retrain-skip bug now extends
    past hyperparam-only sites; escalate to operator with
    identical-output streak across physically-distinct
    feature-count changes.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after
    38+ iterations — I cannot verify whether the 3 surviving
    singing FPs sit within 6s of file end (edge guard fires) or
    in the file body (feature fires cleanly). Every FAR-window
    hypothesis remains theory-calibrated. (ii) SHAP rollup STILL
    "no keeps yet — rollup empty" for 7972a98 despite 14 keeps —
    rollup writer broken. (iii) 99081f5 capacity ghost in-tree
    at HEAD (max_depth=4, max_leaf_nodes=16) — feature delta
    clean, absolute comparison mixes with ghost. (iv) Per-domain
    spec_contrast distributions absent — I'm extrapolating from
    voiced_unvoiced_spec_contrast_asymmetry NEAR's small +0.005
    keep that the FAR companion will be proportionally
    productive on singing.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE — persistent 38+-iteration blocker, cited in
    every recent reflection. Per-FP (domain, file, t_sec,
    label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, top-5 |SHAP|).
    chunk_duration_s explicitly tells me whether FAR-window edge
    guards fire on each FP. Three feature-add iterations
    (d49284c, ff88b22, this one) all theory-calibrated on the
    same unverified assumption. (2) US-505 COVERAGE FIX —
    extend retrain sha gate to include train_classifier.py
    changes, not just features.py. Diagnostic: 5+ consecutive
    train_classifier.py-only iterations all produced IDENTICAL
    combined=0.490700 while feature-count bumps produced distinct
    values. ~5-line shell extension of existing US-505 sha-gate.
    (3) DISCARD-REVERT SYMMETRY SEMANTICS bug fix — wrapper
    treats "matches baseline exactly" as no-improvement,
    restoring the 99081f5 capacity ghost via 960113c discard.
    If commit subject starts "REVERT" + names a prior-discarded
    SHA, treat exact-match as KEEP. ~15 lines git commit-message
    parse.

## 2026-04-20T18:18:46+09:00 — b7dc8bf (discard, combined=0.503630)
subject: add voiced_unvoiced_mfcc_asymmetry_mid feature (FEATURE_NAMES 80->81) — MID-adjacent pre[t-3,t] post[t,t+3] companion to 1eda8e3 NEAR +-2s MFCC asymmetry. Same paired-difference template, +-3s adjacent windows, 13-dim cosine. Reuses cached feat_mfcc + feat_vp, ZERO new librosa calls. Edge guard t-3<0 OR t+3>duration_s returns sentinel 0.0. Targets 7972a98 baseline (combined=0.589282) singing 0.345 3-FP chord-cycle plateau. d49284c FAR pre[t-2,t]/post[t+4,t+8] was the ONLY recent feature to LIFT singing (0.345->0.354) but regressed speech (english 0.889->0.825, korean 0.667->0.537) because the 6s gap made pre/post sample different sentences. ff88b22 WIDE +-4s adjacent closed the gap but still regressed speech (english 0.765, korean 0.556) because +-4s is ~12 syllables — wide enough to span sentence content. MID +-3s adjacent is the cited fallback from d49284c(c2)/ff88b22(c1)/d290101(c1) and is the principled interpolation between proven NEAR (+-2s adjacent, kept +0.054 biggest win) and failed WIDE (+-4s adjacent). Mechanism — speech self-gating preserved: +-3s is ~6 syllables adjacent, narrower than WIDE's 12-syllable span and broader than NEAR's 3 syllables; adjacent windows share speaker/recording baseline with no sentence drift so voiced+unvoiced MFCC shifts co-vary with phoneme context -> diff near zero -> low per-domain SHAP on english/korean. Singing discrimination: +-3s spans ~1-1.5 chord cycles (chord period ~2-3s in pop), begins averaging chord-cycle noise where NEAR doesn't bite the 3 FPs, but doesn't over-average like WIDE. Cross-song splice: pre=3s of song A mastering, post=3s of song B — unvoiced distance HIGH (different drums/mastering), voiced distance moderate (same/different singer) -> asymmetry positive. Chord-cycle FP: voiced+unvoiced both shift with chord progression -> asymmetry ~0 (non-discriminating). GBM cannot synthesize MID from NEAR alone; providing explicit time-scale companion surfaces the chord-averaging pattern. First MID window on any asymmetry axis. Classifier hyperparam axis verifiably broken (5 train_classifier.py-only iterations produced IDENTICAL combined=0.490700 -> US-505 does not auto-retrain on hyperparam edits); only features.py-sha bumps force retrain. Every primary tunable saturated both directions; voiced/unvoiced NEAR and FAR-gap exhausted; percussive/harmonic mask family monotonically failed; 1D scalar abs-delta (0c3bf76) failed. Voiced/unvoiced asymmetry family is provably the most productive axis: 1eda8e3 +0.054 (biggest win) and 7972a98 +0.005. Time-scale MID on this template is the genuinely untried principled fallback. Orthogonal: NOT 1eda8e3 (NEAR +-2s — same template at 1.5x time scale); NOT d49284c (FAR has 4s gap — MID is adjacent); NOT ff88b22 (WIDE +-4s — MID is 25%% narrower preserving speech gating); NOT 7972a98 (spec_contrast not cepstral); NOT f4148cc (chroma asymmetry NEAR); NOT 0c3bf76 (1D scalar abs-delta flatness); NOT d290101 (spec_contrast FAR-gap); NOT any percussive/harmonic/unvoiced-only/symmetry/far-single-mask variant. FIRST MID-ADJACENT window on any content axis. Pure features.py change — 1 new block (~50 lines, template-cloned from 1eda8e3 with edge guard) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. ZERO new librosa calls, ZERO new caches. Per-t cost: 4 slices + 4 masked means + 2 cosines on 13-dim vectors + 1 subtraction, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_mfcc_asymmetry_mid', edge guard fires correctly (t=2 with t-3<0 -> 0.0; t=18 with t+3>20 -> 0.0), in-range t=10 yields finite value 0.0004, all 81 features finite on synthetic audio, idempotent on repeated calls.
per-domain: combined_english=0.800000 combined_korean=0.529412 combined_singing=0.301613

# 2026-04-20 — hypothesis: add voiced_unvoiced_mfcc_asymmetry_mid (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. MID-adjacent companion to the 1eda8e3 NEAR ±2s MFCC asymmetry (kept +0.054, biggest win) and d49284c FAR-gap (singing lift but speech regression). pre=[t-3, t], post=[t, t+3], adjacent (no gap), 13-dim cosine. Reuses cached feat_mfcc + feat_vp, ZERO new librosa calls. Edge guard t-3<0 OR t+3>duration_s returns sentinel 0.0. FEATURE_NAMES 80→81 forces wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. d49284c FAR (pre[t-2,t] post[t+4,t+8]) was the ONLY recent feature to LIFT singing (0.345→0.354) but hurt speech (english 0.889→0.825, korean 0.667→0.537) because the 6s gap made pre/post sample different sentences. ff88b22 WIDE ±4s adjacent closed the gap but still hurt speech (english 0.765, korean 0.556) because ±4s ≈ 12 syllables spans different sentence content. d290101 spec_contrast FAR-gap preserved speech (english 0.897!) but hurt singing. MID ±3s adjacent is the cited fallback from d49284c/c2, ff88b22/c1, d290101/c1 and is the interpolation between proven NEAR (±2s adjacent, kept +0.054) and failed WIDE (±4s adjacent). Speech self-gating: ±3s ≈ 6 syllables adjacent — narrower than WIDE's 12-syllable span, broader than NEAR's 3 syllables; adjacent windows share speaker/recording baseline with no sentence drift, so voiced+unvoiced MFCC shifts co-vary with phoneme context → diff near zero → low per-domain SHAP. Singing: ±3s spans ~1-1.5 chord cycles (chord period ~2-3s in pop), begins averaging chord-cycle noise where NEAR doesn't, but doesn't over-average like WIDE. Cross-song splice: pre is 3s of song A mastering, post is 3s of song B; unvoiced distance HIGH, voiced distance moderate → asymmetry positive. Chord-cycle FP: both voiced+unvoiced shift with chord progression → asymmetry near 0. First MID window on the MFCC axis; classifier hyperparam axis is verifiably broken (5 train_classifier.py-only iterations produced IDENTICAL 0.490700 — US-505 doesn't auto-retrain on hyperparam edits); feature-count bump is the only path alive.

(c) IF THIS FAILS. (1) Speech still regresses (±3s is still too wide) → fallback to NARROW MID ±2.5s pre[t-2.5,t]/post[t,t+2.5] (smaller expansion from proven NEAR ±2s). (2) Singing unchanged (±3s still catches chord-cycle without averaging enough) → fallback to NARROW-GAP FAR pre[t-2,t]/post[t+1,t+4] (1s gap + 3s span — smaller gap than d49284c's 4s but preserves some chord-averaging). (3) Matches 0.490700 exactly → wrapper retrain-skip bug now extends past hyperparam-only sites, escalate.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 39+ iterations — cannot verify whether the 3 singing FPs sit in chord-cycle-dense windows where ±3s averaging would actually average vs isolated transitions where it wouldn't. (ii) SHAP rollup STILL empty despite 14 keeps. (iii) chunk_duration distribution absent — cannot estimate edge-guard fire rate on eval chunks. (iv) 99081f5 4/16 ghost in-tree; feature delta attribution clean but absolute comparison mixes.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — 39+-iteration blocker; per-FP (domain, file, t_sec, p_splice, dsp_*, chunk_duration_s, voiced_unvoiced_mfcc_asymmetry, voiced_unvoiced_spec_contrast_asymmetry, top-5 |SHAP|). (2) US-505 COVERAGE FIX — extend retrain sha-gate to train_classifier.py changes, not just features.py. 5 train_classifier.py-only iterations all produced IDENTICAL 0.490700. (3) DISCARD-REVERT SYMMETRY SEMANTICS bug fix — wrapper treats "matches baseline exactly" as no-improvement; restored 99081f5 capacity ghost via 960113c discard. If commit subject starts "REVERT" + names prior-discarded SHA, treat exact-match as KEEP.

## 2026-04-20T18:41:39+09:00 — 177d641 (discard, combined=0.489666)
subject: add voiced_unvoiced_spec_contrast_asymmetry_narrow_gap feature (FEATURE_NAMES 80->81) -- NARROW-GAP FAR companion to 7972a98 NEAR voiced_unvoiced_spec_contrast_asymmetry (kept +0.005). pre[t-2,t] post[t+1,t+4] (1s gap, 3s span). Reuses cached feat_contrast + feat_vp, ZERO new librosa calls, ZERO new caches. Edge guard t-2<0 OR t+4>duration_s returns sentinel 0.0. FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate. Combines two cited fallbacks: b7dc8bf(c)(2) NARROW-GAP FAR geometry (1s gap 3s span) + spec_contrast axis (d290101 proved speech-safe: english 0.889->0.897 IMPROVED, korean matched 0.667, only singing regressed because 3s gap let voiced formants diverge on different-chord content). 1s gap on spec_contrast keeps voiced spec_contrast small on BOTH chord-cycle AND cross-song cases while UNVOICED (mastering) shifts only on cross-song. Cross-song -> POSITIVE asymmetry (new drum bus + new master EQ + new limiter); chord-cycle FP -> ~0 (frozen mastering, non-discriminating, feature silent). Speech self-gating preserved: 1s gap keeps phoneme-distribution similar (2-3 syllables adjacent) and spec_contrast peak-RATIO is phoneme-stable per d290101 at 3s gap. Narrower than d290101's 3s gap so even safer for speech. Why NARROW-GAP FAR on spec_contrast over alternatives: MID/WIDE adjacent (ff88b22 WIDE and b7dc8bf MID) lack the gap signal that discriminates chord-cycle from cross-song; 3s gap (d290101) regressed singing; 4s gap (d49284c MFCC) broke speech; spec_flatness axis (0c3bf76 NEAR) has weaker baseline than spec_contrast. Classifier hyperparam axis verifiably broken (5 train_classifier.py-only iterations all produced IDENTICAL combined=0.490700 -> US-505 does not auto-retrain on hyperparam edits). Only features.py-sha bumps force retrain. Voiced/unvoiced spec_contrast asymmetry family is proven productive (7972a98 NEAR kept +0.005); narrow-gap companion on this content axis is genuinely untried and the principled interpolation between NEAR (0s gap) and d290101 (3s gap). Orthogonal: NOT 7972a98 (NEAR 0s gap, same template at narrow gap); NOT d290101 (3s gap, 3s span -- this is 1s gap); NOT d49284c (MFCC content, 4s gap, 4s span); NOT ff88b22 (MFCC WIDE adjacent no gap); NOT b7dc8bf (MFCC MID adjacent no gap); NOT 1eda8e3 (NEAR MFCC); NOT f4148cc (chroma NEAR asym); NOT 0c3bf76 (spec_flatness 1D scalar); NOT any percussive/harmonic/tonnetz/symmetric/single-mask variant. FIRST NARROW-GAP FAR geometry on any content axis, FIRST 1s-gap geometry on spec_contrast. Pure features.py change -- 1 new block (~70 lines, template-cloned from _block_voiced_unvoiced_spec_contrast_asymmetry with 1s gap + edge guard) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. ZERO new caches, ZERO new librosa calls. Per-t cost: 4 slices + 4 masked means + 2 cosines on 7-dim vectors + 1 subtraction, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_spec_contrast_asymmetry_narrow_gap', synthetic voiced+unvoiced splice at t=8 yields narrow_gap_asym=+0.1255 (positive, discriminative of splice) vs +0.0026 at within-source t=10 (sentinel-small), edge guard t-2<0 returns 0.0, edge guard t+4>duration returns 0.0, idempotent on repeated calls, all 81 features finite on synthetic audio.
per-domain: combined_english=0.843373 combined_korean=0.510448 combined_singing=0.272727

# 2026-04-20 — hypothesis: add voiced_unvoiced_spec_contrast_asymmetry_narrow_gap (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — NARROW-GAP FAR companion to
    7972a98's NEAR voiced_unvoiced_spec_contrast_asymmetry (kept +0.005).
    Same paired-difference template, but post = [t+1, t+4] (1s gap, 3s span)
    instead of NEAR [t, t+2]. Pre stays [t-2, t]. Reuses cached feat_contrast
    + feat_vp — ZERO new librosa calls, ZERO new caches. Edge guard
    t+4 > duration_s OR t-2 < 0 returns sentinel 0.0. Sentinel 0.0 also
    on empty mask / zero-norm. FEATURE_NAMES 80→81 triggers wrapper
    auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. Combines the two most informative
    recent experiments. d290101 (spec_contrast FAR, 3s gap, 3s span):
    english IMPROVED 0.889→0.897, korean matched 0.667, ONLY singing
    regressed 0.345→0.291 — proving spec_contrast axis is intrinsically
    speech-safe even with gap, but 3s gap was too wide on singing
    (voiced formants at [t+3,t+6] diverge from voiced at [t-2,t] on
    different chord, asymmetry sign becomes muddled). b7dc8bf(c)(2)
    cited fallback: "NARROW-GAP FAR pre[t-2,t]/post[t+1,t+4] (1s gap
    + 3s span — smaller gap than d49284c's 4s but preserves some
    chord-averaging)". Applied to spec_contrast NOT MFCC because MFCC
    is phoneme-sensitive at gaps (d49284c/ff88b22/b7dc8bf all broke
    speech) while spec_contrast is mastering-fingerprint stable on
    speech per d290101. Hyperparam axis verifiably broken (5 identical
    0.490700 on train_classifier.py-only edits, US-505 doesn't retrain
    on hyperparam). Only features.py-sha bumps force retrain.

    Mechanism on 3 surviving singing FPs (chord-cycle FPs). At narrow
    1s gap: pre[t-2,t] and post[t+1,t+4] sample temporally close
    voiced content so VOICED spec_contrast stays small on BOTH chord
    cycle AND cross-song cases (same singer formants in both).
    Crucially, UNVOICED distance behaves differently:
    - Chord-cycle FP intra-song: mastering chain frozen (vocal bus
      comp + drum bus comp + master limiter all constant) →
      unvoiced_dist small ~0.08. voiced_dist small ~0.08 →
      asymmetry ~0 (non-discriminating, feature silent).
    - Cross-song splice same-singer: voiced_dist small ~0.08 (similar
      formants), unvoiced_dist LARGE ~0.30-0.50 (new drum bus + new
      master EQ + new limiter) → asymmetry POSITIVE +0.22-0.42.
    - Different-singer cross-song: both shift → asymmetry small but
      1eda8e3 MFCC asym already fires on this case.
    The mechanism distinguishes cross-song (POS) from chord-cycle (~0)
    without the sign-flip ambiguity d290101 suffered at 3s gap.

    Speech self-gating: 1s gap keeps pre and post phoneme-distribution
    similar (2-3 syllables adjacent); spec_contrast peak-RATIO is
    phoneme-stable per d290101; asymmetry near zero → low per-domain
    SHAP on english/korean. Narrower than d290101's 3s gap where
    speech self-gating already held → even safer.

    Why NARROW-GAP FAR over alternatives:
    (1) MID ±3s adjacent spec_contrast — interpolation between proven
    NEAR and failed WIDE (MFCC WIDE broke), but pure-adjacent lacks
    the GAP signal that discriminates chord-cycle from cross-song.
    (2) Expand d290101 post to [t+3, t+7] (d290101(c)(2) cited) —
    would make singing regression WORSE by extending into more
    chord-diverse content.
    (3) F0 jitter asymmetry — unvoiced frames lack F0, template doesn't
    apply. Would need different mechanism.
    (4) spec_flatness FAR — 0c3bf76 NEAR already failed (0.5385);
    flatness axis has weaker baseline signal than spec_contrast.

    Orthogonal: NOT 7972a98 (NEAR, 0s gap); NOT d290101 (3s gap, 3s
    span — this is 1s gap); NOT d49284c (MFCC content, 4s gap, 4s
    span); NOT ff88b22 (MFCC WIDE adjacent); NOT b7dc8bf (MFCC MID
    adjacent); NOT 1eda8e3 (NEAR MFCC); NOT f4148cc (chroma NEAR
    asym); NOT 0c3bf76 (spec_flatness 1D scalar); NOT any percussive/
    harmonic/tonnetz/symmetric variant. FIRST NARROW-GAP FAR geometry
    on any content axis, FIRST 1s gap on spec_contrast. Pure
    features.py change — 1 new block (~50 lines, template-cloned from
    _block_voiced_unvoiced_spec_contrast_asymmetry with edge guard +
    1s gap) + 1 FEATURE_NAMES append + 2 assert bumps (80→81) + 1
    call. ZERO new caches, ZERO new librosa calls. Per-t cost: 4
    slices + 4 masked means + 2 cosines on 7-dim vectors + 1
    subtraction, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged / speech preserved (1s gap
    too small to discriminate chord-cycle from cross-song even on
    spec_contrast — unvoiced_dist stays small on BOTH cases because
    1s post window mostly captures the same mastering continuity on
    either splice type) → step gap to 2s, pre[t-2,t]/post[t+2,t+5]
    (untried MID-GAP FAR). (2) Singing regresses like d290101 /
    speech preserved (1s gap introduces just enough voiced-chord
    drift to muddle sign) → pivot to voiced_unvoiced_spec_contrast_
    asymmetry_mid adjacent pre[t-3,t]/post[t,t+3] (cited b7dc8bf,
    untried on spec_contrast). (3) combined matches 0.490700 exactly
    despite feature-count change → wrapper retrain-skip bug extends
    past hyperparam-only sites; escalate to operator with identical-
    output streak across physically-distinct feature-count changes.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 40+
    iterations — cannot verify whether the 3 surviving singing FPs
    sit in chord-dense windows where 1s gap actually breaks sync or
    isolated transitions where 1s gap is within same chord period;
    every gap-geometry hypothesis remains theory-calibrated. (ii)
    SHAP rollup STILL "no keeps yet — rollup empty" for 7972a98
    despite 14 keeps — rollup writer broken, no per-feature
    attribution available to predict the narrow-gap signal strength.
    (iii) chunk_duration distribution absent — cannot estimate edge
    guard fire rate (1s gap + 3s post → t+4>duration_s fires rarely
    on 30s chunks, even rarer on 120s, but cannot verify). (iv) Only
    one in-tree spec_contrast asymmetry variant (NEAR) so I can't
    check whether 7972a98's per-domain SHAP suggests FAR/GAP
    variants will compose cleanly with it or double-count signal.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE — persistent 40+-iteration blocker. Per-FP (domain,
    file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    dsp_max, dsp_sum, chunk_duration_s,
    voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry,
    chord_cycle_density_estimate, top-5 |SHAP|). Single block
    transforms every gap-geometry hypothesis from theory-calibrated
    bet into data-driven decision. chord_cycle_density_estimate =
    count of emits above 0.5 p_splice within ±10s of each FP would
    directly test the chord-cycle hypothesis. (2) US-505 COVERAGE
    FIX — extend retrain sha-gate to train_classifier.py changes,
    not just features.py. Diagnostic: 5+ consecutive
    train_classifier.py-only iterations all produced IDENTICAL
    combined=0.490700 while feature-count bumps produced distinct
    values. ~5-line shell extension. (3) DISCARD-REVERT SYMMETRY
    SEMANTICS bug fix — wrapper treats "matches baseline exactly"
    as no-improvement, restoring the 99081f5 capacity ghost via
    960113c discard. If commit subject starts "REVERT" + names
    prior-discarded SHA, treat exact-match as KEEP not DISCARD.
    ~15 lines git commit-message parse. Three unchanged
    highest-priority requests across 40+ iterations.

## 2026-04-20T18:57:34+09:00 — 78513fb (discard, combined=0.546214)
subject: add voiced_unvoiced_mfcc_asymmetry_far_narrow (FEATURE_NAMES 80->81) -- NARROW-GAP-FAR companion to 1eda8e3 NEAR MFCC asymmetry (kept +0.054 biggest win) and d49284c FAR-gap (singing +0.009 lift but speech regressed via wide phoneme drift). pre[t-2,t] post[t+3,t+6] (3s gap, 3s span). Reuses cached feat_mfcc + feat_vp + feat_audio/sr for duration, ZERO new librosa calls, ZERO new caches. Edge guard t-2<0 OR t+6>duration_s returns sentinel 0.0. Explicit cited fallback from d49284c(c)(2): 'shrink the far offset to [t+3, t+6] (less dense but still beats NEARs chord-cycle blind spot)'. Genuinely untried on MFCC content axis. Interpolates structurally between d49284c (4s gap 4s span -- broke speech) and proven NEAR (0s gap 2s span -- kept), and between b7dc8bf MID adjacent 0s gap (failed singing, no gap signal) and d49284c. Mechanism on 3 surviving singing FPs: post sub-window 3-6s past splice = ~1-1.5 chord cycles past (pop chord period 2-3s) -- voiced averaging partially washes within-song chord state, unvoiced stays small (mastering frozen) -> asymmetry near zero, non-discriminating. Cross-song splice: post samples song B 3-6s in (mastering fully consolidated), unvoiced_dist HIGH, voiced_dist moderate -> asymmetry POSITIVE. Speech self-gating preserved by narrower geometry: 3s post span samples 4-6 syllables (vs d49284cs 6-8), 3s gap (vs d49284cs 4s), total pre-to-post-mean separation 5.5s (vs d49284cs 6s), less phoneme-mix drift -> voiced+unvoiced MFCC co-vary more closely with phoneme context -> diff closer to zero-mean noise -> GBM low per-domain SHAP on english/korean. Halfway between NEAR (clean self-gating) and d49284c FAR (broken self-gating). d290101 proved 3s-gap 3s-span geometry is speech-safe on spec_contrast (english 0.889->0.897 improved, korean matched 0.667); transplanting this geometry to MFCC content axis where the mechanism actually discriminates chord-cycle from cross-song. Classifier hyperparam axis verifiably broken (5 train_classifier.py-only iterations produced IDENTICAL combined=0.490700 -> US-505 does not auto-retrain on hyperparam edits); only features.py-sha bumps force retrain. Every primary tunable saturated both directions; percussive/harmonic mask family monotonically failed; 1D scalar abs-delta (0c3bf76) failed; all other voiced/unvoiced asymmetry time-scales exhausted (NEAR kept 1eda8e3/7972a98, MID b7dc8bf failed, WIDE ff88b22 failed, FAR d49284c singing-only, FAR-spec_contrast variants failed on singing). NARROW-GAP-FAR on MFCC is the cited remaining middle-ground. Orthogonal: NOT 1eda8e3 (NEAR 0s gap -- same template different time scale); NOT d49284c (FAR 4s gap 4s span -- this is 3s gap 3s span); NOT ff88b22 (WIDE no gap); NOT b7dc8bf (MID no gap); NOT d290101 (spec_contrast content axis); NOT 177d641 (spec_contrast 1s gap); NOT f4148cc (chroma NEAR); NOT 0c3bf76 (1D scalar flatness abs-delta); NOT ff65865 (single-mask far chroma); NOT any percussive/harmonic/tonnetz/symmetric variant. FIRST 3s-gap + 3s-span geometry on the MFCC axis. Pure features.py change -- 1 new block (~60 lines template-cloned from _block_voiced_unvoiced_mfcc_asymmetry with edge guard + 3s gap) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. ZERO new librosa calls, ZERO new caches. Per-t cost: 4 slices + 4 masked means + 2 cosines on 13-dim vectors + 1 subtraction, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_mfcc_asymmetry_far_narrow', edge guard t=1 (t-2<0) returns 0.0, edge guard t=15 with 20s chunk (t+6>20) returns 0.0, in-range t=10 yields +0.091 (finite, non-trivial on synthetic voiced/unvoiced splice), idempotent on repeated calls, all 81 features finite on synthetic audio.
per-domain: combined_english=0.814815 combined_korean=0.666667 combined_singing=0.300000

# 2026-04-20 — hypothesis: add voiced_unvoiced_mfcc_asymmetry_far_narrow (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — NARROW-GAP-FAR companion
    to 1eda8e3's NEAR MFCC asymmetry (kept +0.054 biggest win). Same paired-
    difference template, post=[t+3, t+6] (3s gap, 3s span) instead of NEAR
    [t, t+2]. Pre stays [t-2, t]. Reuses cached feat_mfcc + feat_vp +
    feat_audio/sr for duration — ZERO new librosa calls, ZERO new caches.
    Edge guard t-2<0 OR t+6>duration_s returns sentinel 0.0. Sentinel also
    on empty masks / zero-norm. FEATURE_NAMES 80→81 forces wrapper auto-
    retrain via US-505 sha gate.

(b) WHY this over recent failures. d49284c (MFCC FAR, post[t+4,t+8], 4s
    gap + 4s span) was THE ONLY recent feature to lift singing
    (0.345→0.354); speech regressed (english 0.889→0.825, korean
    0.667→0.537) because the 4s span inside post window sampled broad
    phoneme content differently from pre. ff88b22 MFCC WIDE ±4s adjacent
    (no gap) failed singing 0.295 confirming a GAP is needed to wash the
    chord-cycle within-song signal. b7dc8bf MFCC MID ±3s adjacent failed
    singing 0.302. d290101 spec_contrast FAR 3s gap 3s span preserved/
    improved speech (english 0.889→0.897!) but failed singing; 177d641
    spec_contrast 1s gap 3s span also failed singing — spec_contrast FAR
    variants can't match MFCC's chord-cycle discrimination. This hypothesis
    is the EXPLICIT cited fallback from d49284c(c)(2): "shrink the far
    offset to [t+3, t+6] (less dense but still beats NEAR's chord-cycle
    blind spot)". Genuinely untried on MFCC.

    Mechanism on 3 surviving singing FPs. 3s gap means post sub-window is
    3-6s past splice = 1-2 chord cycles past (pop chord period 2-3s).
    Voiced averaging over ~1.5 chords partially washes within-song chord
    state (less than d49284c's 2-3 chord averaging, more than b7dc8bf's
    no-gap). Chord-cycle FP: voiced stays near within-song mean, unvoiced
    stays small (mastering frozen) → asymmetry near zero, non-
    discriminating (silent). Cross-song splice: pre samples song A, post
    samples song B 3-6s in (mastering / accompaniment fully consolidated)
    — unvoiced_dist HIGH, voiced_dist moderate → asymmetry POSITIVE.
    Speech self-gating: 3s post span samples ~4-6 syllables (vs d49284c's
    4s = 6-8 syllables), less phoneme-mix drift between pre and post,
    voiced+unvoiced co-vary with phoneme context at smaller amplitude →
    diff closer to zero-mean noise → GBM low per-domain SHAP on english/
    korean. Halfway between NEAR (clean self-gating) and d49284c FAR
    (broken self-gating).

    Classifier hyperparam axis verifiably broken (5 train_classifier.py-
    only iterations IDENTICAL 0.490700). Only features.py-sha bumps force
    retrain. Mask family (percussive/harmonic) exhausted. Voiced/unvoiced
    asymmetry is proven productive (1eda8e3 +0.054, 7972a98 +0.005);
    time-scale variation is the remaining axis and NARROW-GAP-FAR is the
    untried middle point.

    Orthogonal: NOT 1eda8e3 (NEAR 0s gap — same template, different time
    scale); NOT d49284c (FAR 4s gap 4s span — this is 3s gap 3s span);
    NOT ff88b22 (WIDE no gap); NOT b7dc8bf (MID no gap); NOT d290101
    (spec_contrast content, same geometry this time on MFCC); NOT 177d641
    (spec_contrast 1s gap); NOT f4148cc (chroma NEAR); NOT 0c3bf76 (1D
    scalar abs-delta). FIRST 3s-gap + 3s-span geometry on the MFCC axis.
    Pure features.py — 1 new block + 1 FEATURE_NAMES append + 2 assert
    bumps + 1 call. ZERO new librosa calls, ZERO new caches. Per-t cost:
    4 slices + 4 masked means + 2 cosines on 13-dim vectors + 1
    subtraction, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged / speech regresses (3s gap still
    breaks MFCC speech self-gating) → fallback NARROW-GAP MFCC at 2s gap
    / 2s span pre[t-2,t] post[t+2,t+4] — narrower post mimicking NEAR.
    (2) Singing regresses like spec_contrast FAR variants (3s gap lets
    voiced MFCC diverge from within-song average enough to muddle signal
    on MFCC too) → pivot to voiced_unvoiced_spec_contrast_asymmetry at
    FAR 6s gap pre[t-2,t] post[t+6,t+9]. (3) combined matches 0.490700
    EXACTLY → wrapper retrain-skip bug extends past hyperparam-only
    sites; escalate.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 41+
    iterations — cannot verify if the 3 singing FPs sit at t positions
    where t+6 exceeds chunk duration (edge guard silences feature).
    Every far-window hypothesis remains theory-calibrated. (ii) SHAP
    rollup STILL "no keeps yet — rollup empty" despite 14 keeps. (iii)
    chunk_duration distribution absent — cannot estimate edge-guard
    firing rate. (iv) 99081f5 capacity ghost in-tree at HEAD — feature
    delta clean, absolute comparison mixes with ghost.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent 41+-iteration blocker. Per-FP (domain, file,
    t_sec, p_splice, dsp_*, chunk_duration_s,
    voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, top-5 |SHAP|).
    chunk_duration_s tells me directly whether far-window edge guards
    fire per FP. (2) US-505 COVERAGE FIX — extend sha-gate to cover
    train_classifier.py changes. 5+ consecutive train_classifier.py-only
    iterations produced IDENTICAL 0.490700. (3) DISCARD-REVERT SYMMETRY
    SEMANTICS bug — wrapper treats "matches baseline exactly" as
    no-improvement, restoring 99081f5 ghost via 960113c discard. If
    commit subject starts "REVERT" + names prior-discarded SHA, treat
    exact-match as KEEP.

## 2026-04-20T19:16:02+09:00 — dde4135 (discard, combined=0.534196)
subject: add voiced_unvoiced_mfcc_asymmetry_far_balanced feature (FEATURE_NAMES 80->81) -- balanced-span FAR-gap MFCC asymmetry companion to 1eda8e3 NEAR (+0.054 kept biggest win). pre[t-2,t] post[t+4,t+6] (4s gap, 2s spans balanced). Keeps d49284c's 4s-gap chord-cycle averaging (only recent feature to LIFT singing 0.345->0.354) while shrinking post span from 4s to 2s to match pre exactly -- restores NEAR-style balanced phoneme coverage that made paired-differencing self-gate on speech. Asymmetric spans (d49284c 2s pre + 4s post) let post sample broader sentence content than pre, breaking voiced+unvoiced MFCC co-variation that underlies speech self-gating. Balanced 2s spans pre and post means ~3 syllables of same-speaker continuous audio on each side -> voiced+unvoiced MFCC shifts co-vary with phoneme context -> diff near zero -> GBM low per-domain SHAP on english/korean (same mechanism that kept 1eda8e3 NEAR). Reuses cached feat_mfcc + feat_vp + feat_audio for duration check, ZERO new librosa calls, ZERO new caches. Edge guard t-2<0 OR t+6>duration_s returns sentinel 0.0. FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate (only forcing function since 5+ train_classifier.py-only iterations produced IDENTICAL combined=0.490700 confirming US-505 doesn't retrain on hyperparam edits). Mechanism on 3 surviving singing FPs (chord cycle): 4s past splice = ~2 chord cycles past (pop period ~2s), post[t+4,t+6] samples chord that cycled back toward pre chord OR new harmonic position sharing voiced formants -> voiced_dist moderate, unvoiced_dist small (mastering frozen) -> asymmetry near zero, feature silent, FP not boosted. Cross-song splice: pre 2s song A, post[t+4,t+6] firmly inside song B -> unvoiced_dist LARGE (new drums/mastering), voiced moderate -> asymmetry POSITIVE, real TP boosted. First BALANCED-SPAN FAR-gap geometry on any asymmetry axis. Orthogonal: NOT 1eda8e3 (NEAR, same spans no gap); NOT d49284c (4s gap + 4s asymmetric span); NOT 78513fb (3s gap 3s span); NOT ff88b22 (WIDE no gap 4s span); NOT b7dc8bf (MID no gap 3s span); NOT d290101/177d641 (spec_contrast content); NOT any mask/percussive/harmonic/tonnetz/symmetric variant. Pure features.py change -- 1 new block (~55 lines template-cloned from 1eda8e3 with edge guard + 4s gap) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call. Per-t cost 4 slices + 4 masked means + 2 cosines on 13-dim vectors + 1 subtraction, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_mfcc_asymmetry_far_balanced', synthetic splice at t=13 (far window captures post-splice) yields asymmetry=1.927 LARGE, within-source t=5 yields 0.080 sentinel-small (24x discrimination), NEAR comparison t=13=0.026 shows feature surfaces signal NEAR can't see, edge guard t=1 (t-2<0) returns 0.0, edge guard t=25 (t+6>30) returns 0.0, all 81 features finite, idempotent.
per-domain: combined_english=0.853659 combined_korean=0.613846 combined_singing=0.290909

# 2026-04-20 — hypothesis: add voiced_unvoiced_mfcc_asymmetry_far_balanced (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — balanced-span FAR-gap MFCC asymmetry.
    pre=[t-2, t] post=[t+4, t+6] — 4s gap, 2s span matching pre exactly. Same
    paired-difference template as 1eda8e3. Reuses cached feat_mfcc + feat_vp —
    ZERO new librosa calls, ZERO new caches. Edge guard t-2<0 OR t+6>duration_s
    returns sentinel 0.0. FEATURE_NAMES 80→81 forces wrapper auto-retrain via
    US-505 sha gate.

(b) WHY this over recent failures. d49284c [t+4,t+8] (4s gap, 4s span) was THE
    ONLY recent feature to lift singing (0.345→0.354), proving 4s-gap captures
    chord-cycle averaging on the MFCC content axis. Speech broke because post's
    4s span (≈6-8 syllables) sampled different sentence content from pre's 2s
    (≈3 syllables) — asymmetric spans broke phoneme-context co-variation, which
    is what makes paired-differencing self-gate on speech. 78513fb [t+3,t+6]
    (3s/3s), b7dc8bf [t,t+3] (0s/3s), ff88b22 [t,t+4] (0s/4s) all regressed
    singing without matching d49284c's 4s-gap averaging. The structural fix:
    keep d49284c's 4s gap (singing mechanism proven) but SHRINK post span to 2s
    matching pre (speech mechanism proven at NEAR). Balanced 2s spans means
    pre samples ~3 syllables and post samples ~3 syllables of same-speaker
    continuous speech — phoneme mix approximately matches → voiced+unvoiced
    MFCC shifts co-vary → diff near zero → GBM low per-domain SHAP on english/
    korean. Same self-gating NEAR enjoys, just offset by 4s. First BALANCED-
    SPAN FAR-gap geometry on any asymmetry axis. Classifier hyperparam axis
    verifiably broken (5 identical 0.490700 on train_classifier.py-only edits);
    only features.py-sha bumps force retrain. Orthogonal: NOT 1eda8e3 (NEAR,
    same spans but no gap); NOT d49284c (4s gap but 4s span asymmetric); NOT
    78513fb (3s gap, 3s span); NOT ff88b22 (no gap 4s span); NOT b7dc8bf (no
    gap 3s span); NOT d290101/177d641 (spec_contrast content); NOT mask family.
    Blast radius: 1 new block (~50 lines template-cloned) + FEATURE_NAMES
    append + 2 assert bumps + 1 call. Per-t cost sub-ms.

    Mechanism on 3 surviving singing FPs (chord cycle). 4s past splice = ~2
    chord cycles past (pop period ~2s) → post samples chord that has cycled
    back toward pre chord OR is on a new harmonic position that happens to
    share voiced formants with pre. Either way voiced_mfcc_dist stays
    moderate-to-small and unvoiced_mfcc_dist stays small (mastering frozen)
    → asymmetry near zero, feature silent, FP not boosted. Cross-song splice:
    pre is 2s of song A, post[t+4,t+6] is 2s firmly inside song B →
    unvoiced_mfcc_dist LARGE (new drums/mastering), voiced moderate →
    asymmetry POSITIVE, real TP boosted. Balanced 2s spans give cleanest
    cosine-distance comparison on both voiced and unvoiced sub-populations.

(c) IF THIS FAILS. (1) Singing flat, speech preserved (2s post too narrow,
    chord-cycle voice hasn't cycled back enough in 2s window) → expand post
    to [t+4, t+6.5] (2.5s span, still narrower than 3s that failed 78513fb).
    (2) Speech regresses (4s gap breaks MFCC self-gating regardless of span
    balance because speaker's prosody shifts over 6s end-to-end separation)
    → pivot to voiced_unvoiced_spec_contrast_asymmetry_far_balanced (4s gap,
    2s spans, same template on spec_contrast which d290101 proved speech-
    safe at 3s gap). (3) combined matches 0.490700 EXACTLY → wrapper retrain
    skip now covers feature-count bumps; escalate.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 42+ iterations
    — cannot verify whether the 3 singing FPs sit within 6s of file ends
    (t+6>duration fires edge guard) vs file middle. Every FAR-window
    hypothesis still theory-calibrated. (ii) SHAP rollup STILL empty for
    7972a98 despite 14 keeps — rollup writer broken. (iii) chunk_duration
    distribution absent. (iv) 99081f5 4/16 ghost in-tree per grep; feature
    delta attribution clean but absolute comparison mixes.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE
    — persistent 42+-iteration blocker. Per-FP (domain, file, t_sec,
    p_splice, dsp_*, chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, top-5 |SHAP|). (2) US-505
    COVERAGE FIX — extend sha-gate to cover train_classifier.py. 5+
    consecutive train_classifier.py-only iterations produced IDENTICAL
    0.490700. (3) DISCARD-REVERT SYMMETRY SEMANTICS bug — treat REVERT
    commits' exact-match as KEEP not DISCARD. ~15 lines of commit-message
    parse.

## 2026-04-20T19:38:23+09:00 — 88adb49 (discard, combined=0.551338)
subject: add voiced_unvoiced_spec_contrast_asymmetry_far_wide feature (FEATURE_NAMES 80->81) -- d49284c's exact 4s-gap + 4s-span FAR geometry transplanted to spec_contrast content axis. pre[t-2,t] post[t+4,t+8]. Reuses cached feat_contrast + feat_vp + feat_audio/sr, ZERO new librosa calls, ZERO new caches. Edge guard t-2<0 OR t+8>duration_s returns sentinel 0.0. Targets 7972a98 baseline (combined=0.589282) singing 0.345 chord-cycle plateau. d49284c MFCC 4s/4s was THE ONLY recent feature to LIFT singing (0.345->0.354) but broke speech (english 0.889->0.825) because MFCC phoneme-correlated. d290101 spec_contrast 3s gap 3s span proved spec_contrast is speech-safe WITH a gap (english 0.889->0.897 IMPROVED!) but 3s/3s didn't capture enough chord-cycle averaging on singing (0.291). 177d641 spec_contrast 1s gap failed singing 0.273 too. Proposed 4s/4s geometry combines d49284c's singing-winning gap+span with d290101's proven speech-safe content axis -- wider gap (4s vs 3s) for more chord-cycle averaging, wider post span (4s vs 3s) for statistically robust song-B mastering mean. Mechanism on 3 surviving singing FPs: chord cycle voiced_dist moderate (formants average across 2 chord cycles) + unvoiced_dist small (mastering frozen) -> asymmetry near zero, feature silent. Cross-song splice: pre 2s song A baseline, post 4s song B mastering, unvoiced_dist LARGE (different drum-bus compression/EQ/limiter), voiced moderate -> asymmetry POSITIVE, TP boosted. Speech self-gating: spec_contrast peak-RATIO is phoneme-stable per d290101; voiced+unvoiced distances co-vary tightly even at 4s gap -> asymmetry near zero -> GBM low per-domain SHAP on english/korean. Classifier hyperparam axis verifiably broken (5 train_classifier.py-only iterations produced IDENTICAL combined=0.490700); only features.py-sha bumps force retrain. Every primary tunable saturated both directions; mask family failed; voiced/unvoiced asymmetry time-scales mostly exhausted. FIRST 4s-gap + 4s-span geometry on spec_contrast axis. Orthogonal: NOT d290101 (3s gap 3s span); NOT 177d641 (1s gap); NOT 7972a98 (NEAR); NOT d49284c (MFCC content axis); NOT ff88b22/b7dc8bf/78513fb/dde4135 (MFCC variants); NOT f4148cc/0c3bf76/any mask/tonnetz/symmetric variant. Pure features.py change -- 1 new block (~55 lines template-cloned from _block_voiced_unvoiced_spec_contrast_asymmetry with edge guard + 4s gap + 4s span) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. Per-t cost 4 slices + 4 masked means + 2 cosines on 7-dim vectors + 1 subtraction, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_spec_contrast_asymmetry_far_wide', synthetic splice-crossing t=8 yields asym=+0.1286 vs within-source t=4 asym=-0.0119 (~10x discrimination), edge guard t=1 (t-2<0) returns 0.0, edge guard t=15 with 20s chunk (t+8>20) returns 0.0, idempotent on repeated calls, all 81 features finite on synthetic audio.
per-domain: combined_english=0.888889 combined_korean=0.573016 combined_singing=0.329032

# 2026-04-20 — hypothesis: add voiced_unvoiced_spec_contrast_asymmetry_far_wide (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — d49284c's exact FAR
    geometry transplanted to the spec_contrast content axis.
    pre=[t-2, t], post=[t+4, t+8] (4s gap, 4s span asymmetric).
    Reuses cached feat_contrast + feat_vp + feat_audio/sr — ZERO
    new librosa calls, ZERO new caches. Edge guard t+8>duration_s
    OR t-2<0 returns sentinel 0.0. FEATURE_NAMES 80→81 forces
    retrain via US-505 sha gate.

(b) WHY this over recent failures. d49284c (MFCC 4s gap + 4s span
    asymmetric) was THE ONLY recent feature to LIFT singing
    (0.345→0.354) but broke speech (english 0.889→0.825, korean
    0.667→0.537) because MFCC is phoneme-correlated — the 6s pre-
    to-post-end separation made pre/post sample different sentence
    content → voiced+unvoiced MFCC no longer co-varied → self-
    gating broke. d290101 (spec_contrast 3s gap + 3s span) proved
    spec_contrast is INTRINSICALLY speech-safe WITH a gap: english
    IMPROVED 0.889→0.897, korean matched 0.667 — but singing
    stayed 0.291 because 3s gap + 3s span didn't capture enough
    chord-cycle averaging OR mastering signal. 177d641 spec_contrast
    1s gap also failed singing 0.273 (gap too small). Every
    subsequent MFCC geometry variant (ff88b22 WIDE adjacent,
    b7dc8bf MID, 78513fb 3s/3s, dde4135 4s/2s balanced) failed
    because narrowing post span or eliminating the gap dropped the
    singing signal, AND MFCC is inherently phoneme-sensitive so
    speech regression persisted. The UNTRIED move is to use
    d49284c's exact singing-winning geometry (4s gap + 4s post
    span) on spec_contrast where d290101 proved speech is safe.
    4s gap is WIDER than d290101's 3s — more chord-cycle
    averaging. 4s post span is WIDER than d290101's 3s — more
    statistical averaging of post-splice mastering state.
    Hyperparam axis verifiably broken (5 identical 0.490700 on
    train_classifier.py-only edits); only features.py-sha bumps
    force retrain.

    Mechanism on 3 surviving singing FPs (chord cycle). Post sub-
    window t+4..t+8 = 4-8s past splice = 2-4 chord cycles past
    (pop period ~2s). Voiced spec_contrast averaged over 4s spans
    ~2 complete chord cycles → voiced_dist moderate (formants
    average across cycles toward singer's mean). Unvoiced
    spec_contrast: mastering chain frozen within song →
    unvoiced_dist small. Asymmetry (unvoiced − voiced) = small
    negative or near zero → non-discriminating on chord-cycle FP,
    feature silent, FP not boosted. Real same-singer cross-song
    splice: pre is 2s song A mastering baseline, post is 4s of
    song B drums+bus compressor+master limiter → unvoiced_dist
    LARGE (per-band peak-to-valley profile shifts via different
    drum-bus compression / EQ / limiter threshold). voiced_dist
    moderate (similar formant structure, different mastering
    envelope). Asymmetry strongly POSITIVE → real TP boosted.
    d49284c proved this mechanism on MFCC; spec_contrast carries
    mastering-fingerprint MORE cleanly than MFCC carries vocal-
    tract (per-band peak-RATIO is directly shaped by drum-bus
    compression, the single most album-variable element).

    Speech self-gating preserved. d290101 proved at 3s gap + 3s
    span: english 0.889→0.897 (IMPROVED). The mechanism:
    spec_contrast per-band peak-RATIO is more phoneme-stable than
    MFCC's absolute cepstral envelope — when pre and post sub-
    windows sample different phoneme distributions across a gap,
    voiced+unvoiced spec_contrast distances co-vary more tightly
    than MFCC counterparts, so asymmetry (difference) stays near
    zero → GBM assigns low per-domain SHAP on english/korean. At
    4s gap the phoneme drift is larger but the peak-RATIO
    stability argument still holds. Worst case: some speech
    regression vs d290101's +0.008 english lift, but even if
    english falls back to 0.889 baseline, singing lift could
    compensate.

    Why this over alternatives. (1) dde4135 cited balanced-span
    spec_contrast (4s gap, 2s spans) — but dde4135 on MFCC
    FAILED (singing 0.291) precisely because balanced 2s post
    couldn't capture chord-cycle averaging; spec_contrast doesn't
    have chord dependency but also doesn't have as much post-span
    statistical power at 2s. 4s post span gives more robust
    mastering averaging. (2) F0 jitter asymmetry — unvoiced has
    no F0. (3) spec_flatness — 0c3bf76 NEAR already failed
    (0.539); weaker baseline than spec_contrast +0.005 keep.
    (4) MFCC at wider pre span — MFCC phoneme sensitivity breaks
    regardless of span symmetry.

    Orthogonal: NOT d290101 (3s gap, 3s span — this is 4s gap,
    4s span); NOT 177d641 (1s gap); NOT 7972a98 (NEAR); NOT
    d49284c (MFCC content); NOT ff88b22/b7dc8bf/78513fb/dde4135
    (MFCC variants); NOT f4148cc/0c3bf76/any mask/tonnetz/
    symmetric variant. FIRST 4s-gap + 4s-span geometry on the
    spec_contrast axis. Pure features.py change — 1 new block
    (~55 lines template-cloned from _block_voiced_unvoiced_spec_
    contrast_asymmetry with edge guard + 4s gap + 4s span) + 1
    FEATURE_NAMES append + 2 assert bumps + 1 call. ZERO new
    caches, ZERO new librosa calls. Per-t cost: 4 slices + 4
    masked means + 2 cosines on 7-dim vectors + 1 subtraction,
    sub-ms.

(c) IF THIS FAILS. (1) Singing lifted but speech regresses (4s gap
    too wide for spec_contrast self-gating — phoneme drift across
    4s breaks the peak-RATIO stability argument) → fallback to
    voiced_unvoiced_spec_contrast_asymmetry_far_balanced
    pre[t-2,t]/post[t+4,t+6] (4s gap, 2s spans balanced, cited
    dde4135(c)(2)). (2) Singing flat (4s/4s spec_contrast too
    statistically averaged — mastering change signal dilutes over
    4s of song B) → pivot to 5s gap + 3s span pre[t-2,t]/
    post[t+5,t+8] (even more chord averaging, 3s span that worked
    on d290101 for speech). (3) combined matches 0.490700 exactly
    → wrapper retrain-skip bug extends past hyperparam-only sites;
    escalate.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after
    43+ iterations — cannot verify whether the 3 singing FPs sit
    within 8s of file ends (t+8>duration fires edge guard). Every
    FAR-window hypothesis still theory-calibrated. (ii) SHAP
    rollup STILL empty for 7972a98. (iii) chunk_duration
    distribution absent. (iv) 99081f5 4/16 capacity ghost in-tree.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block —
    persistent 43+-iteration blocker. Per-FP (domain, file, t_sec,
    p_splice, dsp_*, chunk_duration_s,
    voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, top-5 |SHAP|).
    (2) US-505 COVERAGE FIX — extend sha-gate to
    train_classifier.py. (3) DISCARD-REVERT SYMMETRY SEMANTICS
    bug — if commit subject starts "REVERT" + names prior-
    discarded SHA, treat exact-match as KEEP.

## 2026-04-20T20:22:22+09:00 — 6c6c254 (discard, combined=0.542020)
subject: add voiced_unvoiced_spec_contrast_asymmetry_far_balanced (FEATURE_NAMES 80->81) -- spec_contrast analog of dde4135 MFCC balanced. pre[t-2,t] post[t+4,t+6] (4s gap, 2s balanced spans). Combines d49284c's singing-winning 4s gap with 1eda8e3 NEAR's proven balanced-span speech self-gating, on d290101's proven speech-safe spec_contrast axis. Reuses feat_contrast + feat_vp + feat_audio/sr, ZERO new librosa calls, ZERO new caches. Edge guard t-2<0 OR t+6>duration_s returns sentinel 0.0. FIRST 4s-gap + 2s-balanced-span geometry on spec_contrast; cited explicit fallback from dde4135(c)(2) and 88adb49(c)(1). Mechanism on 3 singing chord-cycle FPs: 4s gap reaches 2-3 chord cycles past splice; mastering frozen within-song -> unvoiced_dist small; voiced 2s-averaged -> voiced_dist small -> asymmetry ~0 silent. Cross-song: post firmly inside new mastering chain -> unvoiced LARGE, voiced moderate -> asymmetry POSITIVE. Speech self-gating: spec_contrast peak-RATIO phoneme-stable per d290101; balanced 2s spans narrower than 88adb49's 4s post (which over-smoothed korean 0.667->0.573) keep mastering statistics local. Orthogonal: NOT 7972a98 NEAR, NOT d290101 3s/3s, NOT 177d641 1s gap, NOT 88adb49 4s/4s asymmetric, NOT dde4135 MFCC content, NOT any MFCC variant, NOT mask family. Pure features.py change -- 1 new block (~55 lines) + FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call. Per-t cost: 4 slices + 4 masked means + 2 cosines on 7-dim vectors + 1 subtraction, sub-ms. Smoke: len(FEATURE_NAMES)==81, last name correct, edge guards t-2<0 and t+6>duration return 0.0, synthetic splice-crossing asymmetry 0.083 vs within-source 0.033 (2.5x), idempotent, all 81 features finite.
per-domain: combined_english=0.850000 combined_korean=0.614706 combined_singing=0.304762

# 2026-04-20 — hypothesis: add voiced_unvoiced_spec_contrast_asymmetry_far_balanced (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — BALANCED-SPAN FAR-gap
    asymmetry on the spec_contrast content axis.
    pre=[t-2, t], post=[t+4, t+6] (4s gap, 2s spans balanced). Same
    paired-difference template as 7972a98 kept NEAR. Reuses cached
    feat_contrast + feat_vp + feat_audio/sr for duration — ZERO new
    librosa calls, ZERO new caches. Edge guard t-2<0 OR t+6>duration_s
    returns sentinel 0.0. FEATURE_NAMES 80→81 forces wrapper auto-retrain
    via US-505 sha gate.

(b) WHY this over recent failures. Combines two empirically-proven pieces
    into a genuinely untried geometry on an untried content-axis cell:
    - d49284c MFCC FAR (4s gap, 4s post span asymmetric): THE ONLY recent
      feature to LIFT singing (0.345→0.354). The 4s-gap chord-cycle
      averaging is the proven singing mechanism.
    - d290101 spec_contrast FAR (3s gap, 3s span): english IMPROVED
      0.889→0.897 proving spec_contrast is intrinsically speech-safe WITH
      a gap (MFCC isn't — phoneme-correlated).
    - dde4135 MFCC balanced (4s gap, 2s spans): regressed speech (english
      0.854, korean 0.614) — MFCC phoneme sensitivity breaks regardless
      of span balance; singing 0.291.
    - 88adb49 spec_contrast 4s gap + 4s asymmetric span: english 0.889
      preserved, singing 0.329, korean 0.573 — 4s post over-smooths
      post-splice mastering mean and destabilizes korean.

    Balanced 2s spans on spec_contrast at 4s gap is the cited explicit
    fallback from dde4135(c)(2) and 88adb49(c)(1). It combines:
    (i) 4s gap — d49284c's singing-lift mechanism (chord-cycle averaging
    requires ~2 chord cycles of gap past splice);
    (ii) 2s balanced spans — matches 1eda8e3 NEAR's proven clean
    paired-differencing self-gating (pre/post sample equal-width regions);
    (iii) spec_contrast content — d290101 proved speech-safe with gaps,
    and mastering-fingerprint targets the 3 chord-cycle singing FPs
    where mastering stays frozen intra-song and jumps cross-song.

    Mechanism on 3 surviving singing FPs. post=[t+4,t+6] samples 4-6s
    past splice ≈ 2-3 chord cycles past (pop period ~2s). Chord cycle:
    mastering frozen, drum kit fixed → unvoiced_dist small; voiced may
    shift on new vowel but averaged over 2s stabilizes → voiced_dist
    small → asymmetry near zero, feature silent, FP not boosted. Cross-
    song splice: pre 2s song A mastering baseline, post 2s firmly inside
    song B new mastering chain (different drum-bus compressor + master
    EQ + limiter threshold) → unvoiced_dist LARGE, voiced moderate →
    asymmetry strongly POSITIVE, TP boosted.

    Speech self-gating preserved: spec_contrast peak-to-valley RATIO is
    phoneme-stable per d290101 (3s gap english IMPROVED). 2s balanced
    spans are narrower than 88adb49's 4s post which over-smoothed korean's
    unvoiced mean — 2s keeps mastering statistics closer to local
    baseline. Voiced+unvoiced peak-ratio both co-vary with mastering
    continuity, difference near zero on same-recording speech → GBM low
    per-domain SHAP on english/korean.

    Why this over alternatives. (1) Repeat 4s/4s asymmetric (88adb49) —
    already tried, singing 0.329 + korean regression. (2) Shrink d290101
    to 2s post (3s gap 2s span) — less chord-cycle averaging than 4s gap
    which was minimum to lift singing on d49284c. (3) Wider gap (5s/2s)
    — extrapolates past d49284c's proven 4s, risky on speech. (4) F0
    jitter asymmetry — unvoiced lacks F0, template doesn't apply.

    Classifier hyperparam axis verifiably broken (5 train_classifier.py-
    only iterations IDENTICAL 0.490700; 920dcdb US-505 coverage fix just
    landed but I cannot verify from inside claude whether it's active on
    this iteration). Mask family exhausted (percussive/harmonic all
    regressed). Time-scale variation on voiced/unvoiced asymmetry is the
    provably-productive axis (1eda8e3 +0.054, 7972a98 +0.005).

    Orthogonal: NOT 7972a98 (NEAR 0s gap); NOT d290101 (spec_contrast
    3s gap 3s span); NOT 177d641 (spec_contrast 1s gap); NOT 88adb49
    (spec_contrast 4s gap 4s asymmetric span — this is 2s balanced);
    NOT d49284c (MFCC content); NOT dde4135 (MFCC 4s gap 2s balanced —
    this is spec_contrast); NOT ff88b22/b7dc8bf/78513fb (MFCC variants);
    NOT 1eda8e3/f4148cc/0c3bf76 (NEAR); NOT any percussive/harmonic/
    tonnetz/single-mask variant. FIRST 4s-gap + 2s-balanced-span geometry
    on the spec_contrast axis. Pure features.py change — 1 new block
    (~55 lines template-cloned from _block_voiced_unvoiced_spec_contrast_
    asymmetry with edge guard + 4s gap + balanced 2s spans) + 1
    FEATURE_NAMES append + 2 assert bumps (80→81) + 1 call. ZERO new
    caches, ZERO new librosa calls. Per-t cost: 4 slices + 4 masked means
    + 2 cosines on 7-dim vectors + 1 subtraction, sub-ms.

(c) IF THIS FAILS. (1) Singing flat / speech preserved (2s post too
    narrow — 88adb49's 4s post was needed for statistical robustness of
    post-splice mastering mean) → fallback to MID-adjacent spec_contrast
    (pre[t-3,t] post[t,t+3], 0s gap, 3s balanced — untried time scale
    on spec_contrast). (2) Singing regresses like d290101 (2s post misses
    chord-averaging entirely; spec_contrast FAR family fundamentally
    doesn't discriminate chord-cycle on singing) → pivot to voiced-only
    F0 statistical features (voiced_f0_percentile_shift — pitch contour
    is recording-dependent not mastering-dependent, genuinely untried
    content axis). (3) combined matches 0.490700 EXACTLY → wrapper
    retrain-skip bug extends past hyperparam sites despite 920dcdb fix;
    escalate.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 44+
    iterations — cannot verify whether the 3 singing FPs sit within 6s
    of file ends (t+6>duration fires edge guard silencing feature).
    Every FAR-window hypothesis remains theory-calibrated. (ii) SHAP
    rollup STILL "no keeps yet — rollup empty" for 7972a98 despite 14
    keeps — rollup writer broken, no per-feature attribution. (iii)
    920dcdb US-505 coverage fix just landed (last commit) but no on-line
    signal that train_classifier.py-only retrains are now firing
    correctly; cannot verify without a hyperparam-only test iteration.
    (iv) 99081f5 capacity ghost in-tree (max_depth=4, max_leaf_nodes=16)
    — feature delta clean; absolute comparison mixes with ghost.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent 44+-iteration blocker. Per-FP (domain, file,
    t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, top-5 |SHAP|). Single block
    transforms every FAR-window / gap-geometry hypothesis from theory-
    calibrated bet to data-driven decision. (2) SHAP ROLLUP REPAIR —
    rollup empty for 14 keeps is a long-standing bug; without per-
    feature attribution claude picks "theoretically orthogonal" not
    "what signal GBM actually uses." (3) US-505 COVERAGE ASSERTION —
    920dcdb extended the sha-gate; a wrapper log line at retrain
    decision point ("train_classifier.py hash delta detected → auto-
    retraining" or "no hash delta → skip retrain") would close the
    feedback-loop verification gap so claude can confirm hyperparam
    iterations now force retrain.

## 2026-04-20T20:38:52+09:00 — 347c0ac (discard, combined=0.509293)
subject: add voiced_unvoiced_spec_contrast_asymmetry_mid (FEATURE_NAMES 80->81) -- MID-adjacent ±3s balanced on spec_contrast. pre[t-3,t] post[t,t+3] 0s gap, cited fallback from 6c6c254(c)(1). Reuses cached feat_contrast + feat_vp, ZERO new librosa calls, ZERO new caches. Empty-mask sentinel 0.0 via _masked_mean None path (matches 7972a98 template). Combines d290101's proven speech-safe spec_contrast axis (3s gap 3s span IMPROVED english 0.889->0.897) with 1eda8e3's proven balanced-adjacent self-gating template (NEAR ±2s kept +0.054 biggest win). b7dc8bf MID (MFCC, 0s gap 3s adjacent) failed speech because MFCC phoneme-correlated; d290101 proves spec_contrast peak-RATIO is phoneme-stable at ±3s spans so the same MID geometry that broke MFCC should preserve speech here. Every FAR-gap spec_contrast variant failed singing (d290101 3s/3s 0.291, 177d641 1s gap 0.273, 88adb49 4s asym 0.329, 6c6c254 4s/2s bal 0.305) -- untried alternative is ADJACENT ±3s no gap, wider than NEAR's ±2s so spans ~1-1.5 chord cycles, begins averaging chord-cycle noise. Mechanism on 3 singing chord-cycle FPs: ±3s covers one chord transition in pre AND post -> voiced_dist averages to moderate, unvoiced_dist stays small (mastering frozen) -> asymmetry ~0 silent. Cross-song splice: pre 3s song A, post 3s song B -> unvoiced LARGE (new drum-bus comp + master EQ + limiter), voiced moderate -> asymmetry POSITIVE. Speech self-gating: spec_contrast peak-RATIO phoneme-stable, ±3s adjacent ~6 syllables each side samples similar phoneme distribution, voiced+unvoiced peak-ratios co-vary -> diff near zero -> GBM low per-domain SHAP on english/korean. FIRST 0s-gap + 3s-balanced-adjacent geometry on spec_contrast axis. Orthogonal: NOT 7972a98 NEAR ±2s, NOT d290101 3s gap, NOT 177d641 1s gap, NOT 88adb49 4s asym, NOT 6c6c254 4s/2s bal, NOT b7dc8bf MID MFCC content, NOT 1eda8e3 NEAR MFCC, NOT f4148cc chroma, NOT 0c3bf76 flatness scalar, NOT any percussive/harmonic/symmetric/single-mask variant. Pure features.py change -- 1 new block (~45 lines cloned from _block_voiced_unvoiced_spec_contrast_asymmetry with ±3s windows) + FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call. Per-t cost: 4 slices + 4 masked means + 2 cosines on 7-dim vectors + 1 subtraction, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_spec_contrast_asymmetry_mid', all 81 features finite on synthetic audio, sentinel 0.0 on empty-voicing mask, edge-clamped t=1 returns 0.0 via _slice_frames clamping path, idempotent.
per-domain: combined_english=0.825000 combined_korean=0.530882 combined_singing=0.301613

# 2026-04-20 — hypothesis: add voiced_unvoiced_spec_contrast_asymmetry_mid (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — MID-adjacent companion to
    7972a98 NEAR spec_contrast asymmetry (kept +0.005). pre=[t-3, t],
    post=[t, t+3] (0s gap, 3s balanced adjacent spans) on the spec_contrast
    content axis. Reuses cached feat_contrast + feat_vp — ZERO new librosa
    calls, ZERO new caches, ZERO edge guard code (`_slice_frames` clamps
    out-of-range safely; empty-mask → sentinel 0.0 via None-path, matching
    7972a98 template). FEATURE_NAMES 80→81 forces wrapper auto-retrain via
    US-505 sha gate (920dcdb US-505b now also covers train_classifier.py).

(b) WHY this over recent failures. EXPLICIT cited fallback from 6c6c254(c)(1):
    "MID-adjacent spec_contrast (pre[t-3,t] post[t,t+3], 0s gap, 3s balanced
    — untried time scale on spec_contrast)". Combines two empirically-
    validated pieces into a genuinely untried geometry cell:
    (i) d290101 (spec_contrast FAR 3s gap 3s span) IMPROVED english 0.889
    →0.897 and matched korean 0.667 — proving spec_contrast is intrinsic-
    ally speech-safe at 3s-wide windows. Only singing regressed (0.291)
    because 3s gap let voiced formants diverge.
    (ii) 1eda8e3 NEAR ±2s BALANCED-ADJACENT MFCC kept +0.054 (biggest win
    in loop history) — 0s gap + balanced spans is the canonical speech-
    self-gating template.
    b7dc8bf MID (MFCC, 0s gap, 3s adjacent) FAILED speech because MFCC is
    phoneme-correlated; d290101 proves spec_contrast peak-RATIO is
    phoneme-stable, so the same MID geometry that broke MFCC should
    preserve speech on spec_contrast.

    Every FAR-gap variant on spec_contrast has failed singing: d290101
    (3s/3s) 0.291, 177d641 (1s gap) 0.273, 88adb49 (4s/4s asym) 0.329,
    6c6c254 (4s gap 2s bal) 0.305. Failure pattern: GAPS on spec_contrast
    preserve speech but fail singing. Untried alternative: ADJACENT ±3s
    no gap, wider than NEAR's ±2s so spans ~1-1.5 chord cycles (pop ~2-3s),
    begins averaging chord-cycle noise where NEAR doesn't bite 3 FPs.

    Mechanism on 3 surviving singing chord-cycle FPs: ±3s covers one
    complete chord transition within pre AND one within post → voiced_dist
    averages across chord = moderate; unvoiced_dist stays small (mastering
    frozen, drum bus identical) → asymmetry near zero, feature silent on
    FP. Real cross-song splice: pre=3s song A baseline, post=3s song B →
    unvoiced_dist LARGE (new drum-bus comp + master EQ + limiter), voiced
    moderate → asymmetry POSITIVE. Speech self-gating via spec_contrast
    peak-RATIO phoneme-stability: at 0s gap + ±3s adjacent ≈ 6 syllables
    each side, broad enough to span typical phoneme distribution;
    recording continuous, voiced+unvoiced peak-ratios co-vary with
    phoneme context; difference near zero → GBM low per-domain SHAP on
    english/korean.

    FIRST 0s-gap + 3s-balanced-adjacent geometry on the spec_contrast
    axis. Orthogonal: NOT 7972a98 (NEAR ±2s); NOT d290101 (3s gap); NOT
    177d641 (1s gap); NOT 88adb49 (4s asym); NOT 6c6c254 (4s/2s bal);
    NOT b7dc8bf (MID on MFCC); NOT 1eda8e3 (NEAR MFCC); NOT f4148cc
    (chroma); NOT 0c3bf76 (flatness); NOT any percussive/harmonic/
    symmetric/single-mask variant. Blast radius: 1 new block (~40 lines
    cloned from _block_voiced_unvoiced_spec_contrast_asymmetry with ±3s
    windows) + 1 FEATURE_NAMES append + 2 assert bumps (80→81) + 1 call.
    ZERO new caches, ZERO new librosa calls. Per-t cost: 4 slices + 4
    masked means + 2 cosines on 7-dim vectors + 1 subtraction, sub-ms.

(c) IF THIS FAILS. (1) Singing flat / speech preserved (±3s doesn't capture
    enough chord averaging; 3s span often lands mid-chord) → fallback to
    MID-WIDE ±3.5s adjacent on spec_contrast (interpolates between this
    MID and failed WIDE ±4s MFCC; untried). (2) Speech regresses (peak-
    RATIO phoneme stability held at d290101's 3s gap but breaks at 0s gap
    because voiced+unvoiced sample the SAME phoneme event across boundary
    and contrast shifts differently than at NEAR/FAR) → pivot to voiced-
    only F0 contour cosine distance (voiced_f0_contour_cosine_dist — pitch
    trajectory shape on voiced frames, untried content axis, per-singer
    identity signal). (3) combined matches 0.589282 exactly → 4/16
    capacity ghost is in-tree AND US-505b retrain is working but yielded
    zero marginal signal; escalate to structural pivot away from voiced/
    unvoiced asymmetry family.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 45+
    iterations — cannot verify if 3 singing FPs sit in chord-dense windows
    where ±3s averaging works, vs isolated transitions. (ii) SHAP rollup
    STILL empty for 7972a98 despite 14 keeps — rollup writer broken, no
    per-feature attribution. (iii) 99081f5 4/16 capacity ghost status
    unclear: train_classifier.py currently shows max_depth=4,
    max_leaf_nodes=16 at HEAD; 960113c "revert" note didn't actually
    revert file values; baseline 0.589282 evaluated against whichever
    config the last retrain produced. With US-505b live, any features.py
    bump now retrains against 4/16. (iv) Per-FP spec_contrast peak-ratio
    trajectories absent — cannot verify MID averaging hypothesis on 3
    singing FPs' actual chord-transition timescale.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent 45+-iteration blocker cited in every reflection.
    Per-FP (domain, file, t_sec, p_splice, dsp_phase_z, dsp_t2_z,
    dsp_cpe_z, chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, top-5 |SHAP|). Transforms
    every geometry hypothesis from theory to data-driven decision.
    (2) SHAP ROLLUP REPAIR — per-iteration SHAP top-K to
    .omc/classifier/shap_rollup.json on every keep, aggregate rolling-5
    in wrapper. Without SHAP, feature selection is blind. (3) US-505b
    VERIFICATION TRACE — 920dcdb landed but no visible signal whether
    train_classifier.py-only iterations now force retrain; wrapper log
    line at retrain decision ("features.py sha Δ → retrain" vs
    "train_classifier.py sha Δ → retrain" vs "no Δ → skip") closes
    feedback-loop verification gap. Also clarifies whether 4/16 ghost
    is still in-effect. Three unchanged highest-priority requests across
    45+ iterations.

## 2026-04-20T20:56:02+09:00 — c006d52 (discard, combined=0.490700)
subject: add voiced_unvoiced_rms_asymmetry_db feature (FEATURE_NAMES 80->81) -- STRUCTURAL pivot from spectral-envelope cosine distance to ENERGY-LEVEL dB. Apply proven voiced/unvoiced paired-difference template (1eda8e3 NEAR MFCC +0.054 biggest keep, 7972a98 NEAR spec_contrast +0.005) to a genuinely NEW content axis: log-RMS dB level rather than cepstral/mastering/pitch-class cosine. pre[t-2,t] post[t,t+2]. For each window compute voiced-mean RMS and unvoiced-mean RMS in dB, then asymmetry = (unvoiced_post_db - unvoiced_pre_db) - (voiced_post_db - voiced_pre_db). Reuses cached feat_rms (per-frame RMS, 512 hop) + feat_vp (voicing probability, same 512 hop) -- ZERO new librosa calls, ZERO new caches. Sentinel 0.0 when any mask empty OR any sub-window mean <= 1e-10 (log-safe). FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate. Last 12+ iterations exhausted GEOMETRY variation on the asymmetry template across MFCC/spec_contrast/chroma/spec_flatness content axes (NEAR/MID/WIDE/FAR/narrow-gap/balanced-span/gap-geometry -- all discarded); CLAUDE.md mandates structural change after 5+ same-axis failures. Untried structural dimension is the CONTENT AXIS itself: energy-level paired-differencing on voiced/unvoiced split (GBM never saw rms asymmetry form, existing rms_db_delta is all-frame no split). Mechanism on 3 surviving singing chord-cycle FPs: within-song vocal bus + drum bus + master limiter are FROZEN; chord transitions don't change mastering -> voiced_delta and unvoiced_delta both ~0 -> asymmetry ~0 silent, FP not boosted. Same-singer cross-song: vocal bus adapts so voiced_delta small 1-3 dB, mastering jumps so unvoiced_delta LARGE 3-8 dB -> asymmetry POSITIVE, real TP boosted. Different-singer cross-song: both shift -> existing 1eda8e3 MFCC asymmetry fires. Speech self-gating via paired differencing (proven 1eda8e3 mechanism): voiced RMS (vowels) and unvoiced RMS (consonants + pauses) both vary with phoneme density and sentence context -- but co-vary within same recording (continuous preamp gain / speaker intensity) -> voiced_delta and unvoiced_delta both track phoneme drift -> DIFFERENCE is zero-mean noise -> GBM low per-domain SHAP on english/korean -> feature functionally invisible on speech domains. FIRST ENERGY-LEVEL (log-RMS dB) asymmetry on any mask split. Orthogonal: NOT 1eda8e3 (cepstral cosine); NOT 7972a98/d290101/177d641/88adb49/6c6c254/347c0ac (spec_contrast cosine variants); NOT f4148cc (chroma cosine); NOT 0c3bf76 (spec_flatness scalar abs-delta, NOT dB-ratio); NOT rms_db_delta (all-frame scalar, no mask); NOT any MFCC geometry variant; NOT percussive/harmonic mask family; NOT F0/rhythm/tempo features. Pure features.py change -- 1 new block (~45 lines cloned from _block_voiced_unvoiced_mfcc_asymmetry with scalar-mean + dB conversion replacing vector-cosine) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. Per-t cost 4 slices + 4 masked means + 4 log10 + 2 subtractions, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_rms_asymmetry_db', direct block test at splice boundary with unvoiced mastering jump yields +9.5 dB asymmetry, stationary windows return 0.0, empty unvoiced mask returns sentinel 0.0, zero-rms sentinel 0.0, end-to-end extract_features returns 81 finite features on synthetic audio, idempotent on repeated calls.
per-domain: combined_english=0.839506 combined_korean=0.476471 combined_singing=0.295385

# 2026-04-20 — hypothesis: add voiced_unvoiced_rms_asymmetry_db (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot: apply
    the proven voiced/unvoiced paired-difference template (1eda8e3 NEAR
    MFCC asymmetry kept +0.054, biggest win; 7972a98 NEAR spec_contrast
    asymmetry kept +0.005) to a genuinely NEW content axis — **log-RMS
    ENERGY LEVEL in dB** rather than spectral-shape cosine distance. For
    pre=[t-2,t] and post=[t,t+2], compute voiced-frame-mean RMS in dB and
    unvoiced-frame-mean RMS in dB. Then
        voiced_delta  = voiced_post_db  - voiced_pre_db
        unvoiced_delta = unvoiced_post_db - unvoiced_pre_db
        feature = unvoiced_delta - voiced_delta
    Reuses cached feat_rms (512-hop per-frame RMS) + feat_vp (voicing
    probability, same 512 hop) — ZERO new librosa calls, ZERO new caches.
    Sentinel 0.0 when any mask is empty OR any sub-window mean ≤ 1e-10
    (log-safe). FEATURE_NAMES 80→81 forces wrapper auto-retrain via
    US-505 sha gate.

(b) WHY this over recent failures. The voiced/unvoiced asymmetry template
    is provably the most productive axis in loop history (1eda8e3 +0.054,
    7972a98 +0.005). The last 12+ iterations have exhausted GEOMETRY
    variation on that template across SPECTRAL content axes: NEAR/MID/
    WIDE/FAR/narrow-gap/balanced-span on MFCC, spec_contrast, chroma,
    spec_flatness — every one discarded. Recent 10 iterations all
    combined < baseline 0.589. CLAUDE.md mandates structural change after
    5+ same-axis failures. The untried structural dimension is the
    **content axis itself**, not geometry: move paired differencing onto
    the ENERGY-LEVEL signal (log-RMS dB) instead of spectral-envelope
    cosine distance. Prior rms_db_pre/post/delta (block 6) are
    ALL-FRAME, no voiced/unvoiced split — GBM has never seen the
    asymmetry form on this signal.

    Mechanism on 3 surviving singing chord-cycle FPs. Within one song the
    vocal bus compressor + drum bus compressor + master limiter are all
    FROZEN; chord transitions don't change mastering chain. voiced-frame
    RMS stays within ~1-2 dB of its within-song baseline (singer's steady
    vocal level held by the bus compressor); unvoiced-frame RMS stays
    equally flat (same kit, same bus comp). Both deltas ≈ 0 →
    asymmetry ≈ 0, feature silent on the chord-cycle FPs, so these FPs
    are not boosted.

    Same-singer cross-song splice. Singer continues in similar register
    (vocal bus adapts within ~2s so voiced_delta is small, 1-3 dB).
    Mastering chain jumps: different drum bus compression / master EQ /
    limiter threshold / LUFS target → unvoiced_delta can be LARGE (3-8
    dB). asymmetry = unvoiced_delta − voiced_delta is strongly positive.
    Different-singer cross-song: both deltas could be large; asymmetry
    partially cancels, but existing 1eda8e3 MFCC asymmetry already fires
    on that case.

    Speech self-gating via paired-difference (proven mechanism). On
    english/korean, voiced RMS (vowels) and unvoiced RMS (consonants +
    pauses) both vary with phoneme density, word boundaries, and prosodic
    stress — but they CO-VARY within the same recording (continuous
    microphone gain / preamp / speaker intensity). voiced_delta and
    unvoiced_delta both respond to the same sentence-context shift, so
    their DIFFERENCE is zero-mean noise → GBM low per-domain SHAP on
    english/korean → feature functionally invisible on speech domains.
    Same self-gating that made 1eda8e3 speech-safe.

    Orthogonal to every prior axis: NOT 1eda8e3 (cepstral cosine);
    NOT 7972a98 / d290101 / 177d641 / 88adb49 / 6c6c254 / 347c0ac
    (spec_contrast variants); NOT f4148cc (chroma cosine); NOT 0c3bf76
    (spec_flatness scalar abs-delta, NOT dB-ratio); NOT rms_db_delta
    (all-frame scalar, no mask); NOT any MFCC geometry variant; NOT
    percussive/harmonic mask family; NOT F0/rhythm/tempo features.
    FIRST ENERGY-LEVEL (log-RMS dB) asymmetry on ANY mask split.

    Blast radius: 1 new block function (~45 lines cloned from
    _block_voiced_unvoiced_mfcc_asymmetry with scalar-mean + dB conversion
    replacing vector-cosine) + 1 FEATURE_NAMES append + 2 assert bumps
    (80→81) + 1 call in extract_features. ZERO new caches, ZERO new
    librosa calls. Per-t cost: 4 slices + 4 masked means + 4 log10 +
    2 subtractions, sub-ms.

(c) IF THIS FAILS. (1) Speech regresses (voiced/unvoiced RMS do NOT
    co-vary tightly on prosodic speech — pauses vs peaks have different
    voiced/unvoiced balance that doesn't cancel) → fallback to
    voiced_unvoiced_rms_ratio_delta using ratio voiced/unvoiced in linear
    (non-dB) units, which cancels multiplicative preamp gain shifts on
    speech but exposes mastering chain differences on singing. (2) Singing
    unchanged (mastering shifts are not big enough in dB at 2s windows —
    auto-leveling within each track flattens differences) → extend
    windows to pre[t-4,t] post[t,t+4] (WIDE-adjacent on RMS, same template
    but 4s spans for statistical robustness of per-window mean). (3)
    combined matches 0.589282 exactly → feature fires but GBM assigns
    zero SHAP (signal structurally identical to existing rms_db_delta
    once voicing-split is irrelevant under the ghost max_depth=4/
    max_leaf=16) → escalate with structural pivot: F0 jitter asymmetry
    voiced_f0_jitter_asymmetry (untried F0-shape content axis per
    d49284c/c3).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 45+
    iterations — cannot verify the 3 surviving singing FPs' t_sec or
    chunk_duration so every hypothesis is theory-calibrated. (ii) SHAP
    rollup STILL "no keeps yet — rollup empty" for 7972a98 despite 14
    keeps — rollup writer broken. No per-feature attribution; picking
    feature axes is theory-only. (iii) 99081f5 capacity ghost in-tree
    (max_depth=4, max_leaf_nodes=16) per 347c0ac(d)(iii) — any feature-
    add retrains against 4/16 but compares against baseline possibly set
    under different config; attribution contaminated. (iv) 920dcdb US-505b
    fix for train_classifier.py sha-gate presumed live but unverified —
    no wrapper log line confirms hyperparam-only retrains now fire.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — 45+-iteration blocker cited in every reflection. Per-FP
    (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z,
    dsp_cpe_z, chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, top-5 |SHAP|,
    voiced_rms_db_pre, unvoiced_rms_db_pre). Would turn every
    voiced/unvoiced asymmetry hypothesis from theory bet into
    data-driven decision. (2) SHAP ROLLUP REPAIR — rollup empty for 14
    keeps is long-standing bug; without per-feature attribution the
    feature-selection policy is blind. (3) US-505b VERIFICATION TRACE —
    wrapper log line at retrain decision point ("features.py sha Δ →
    retrain" vs "train_classifier.py sha Δ → retrain" vs "no Δ → skip")
    confirms 920dcdb works.

## 2026-04-20T21:12:12+09:00 — 27ddbf7 (discard, combined=0.523753)
subject: add voiced_f0_iqr_log_ratio feature (FEATURE_NAMES 80->81) -- STRUCTURAL pivot onto F0 DISTRIBUTION SHAPE content axis, untouched by block 5's mean/delta/jitter/continuity. Feature = |log2(post_iqr / pre_iqr)| where iqr = voiced F0 75th-25th percentile over pre[t-2,t] vs post[t,t+2]. Reuses cached feat_f0 + feat_f0_hop + feat_sr, ZERO new librosa calls, ZERO new caches. Sentinel 0.0 when either window has <3 valid F0 frames OR either IQR <1 Hz (log-safe + perceptually negligible). FEATURE_NAMES 80->81 forces retrain via US-505 sha gate. Targets 7972a98 baseline (combined=0.589282) singing 0.345 3-FP chord-cycle plateau. Last 13+ iterations exhausted geometry/content variation on voiced/unvoiced paired-differencing template (MFCC/spec_contrast/chroma/spec_flatness/RMS dB). CLAUDE.md mandates structural change after 5+ same-axis failures. Block 5 already has f0_mean_pre/post/delta (Hz-linear mean shift) + f0_jitter_delta (std/mean ratio) + f0_continuity_score (edge 100ms delta) but NOTHING percentile-based, NOTHING log-ratio-based, NOTHING shape/spread-based. GBM max_depth=3 cannot synthesize log-IQR-ratio from existing mean/delta/jitter features. Mechanism on 3 surviving singing chord-cycle FPs: within one song singer holds similar phrasing style (melodic range 3-7 semitones per phrase) so voiced F0 IQR of any 2s slice stays stable ~30-80 Hz regardless of chord transition -> pre_iqr and post_iqr both similar -> log2 ratio ~0 silent, FP not boosted. Cross-song splice same-singer: melodic style differs (ballad narrow IQR vs pop bridge wide IQR) -> log-ratio LARGE. Cross-song different-singer: register AND spread shift combined -> feature strongly positive. Speech self-gating via voiceprint stability: within-speaker 2s voiced F0 IQR is habitual prosodic range (~30-60 Hz), stable characteristic over 2s windows -> log-ratio sits ~0.1-0.3 uncorrelated with splice position -> GBM low per-domain SHAP on english/korean. Log-scale normalisation prevents absolute F0 level bias (male F0=150 IQR=30 equiv to female F0=300 IQR=60). IQR chosen over STD because IQR robust to yin octave errors; f0_jitter_delta already uses std/mean. Orthogonal: NOT 1eda8e3 (cepstral cosine); NOT 7972a98/d290101/177d641/88adb49/6c6c254/347c0ac (spec_contrast variants); NOT f4148cc (chroma cosine); NOT 0c3bf76 (spec_flatness 1D abs-delta); NOT c006d52 (RMS dB asymmetry); NOT block-5 F0 mean/delta/jitter/continuity (all mean/edge-based, none IQR or log-ratio); NOT percussive/harmonic mask family; NOT any voiced/unvoiced pair variant. FIRST F0-DISTRIBUTION-SHAPE feature in the 80-feature set, FIRST log-ratio feature in any content axis. Pure features.py change -- 1 new block (~30 lines) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. Per-t cost 2 slices + 2 nan-mask + 4 percentile ops + 1 log2 + 1 abs, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_f0_iqr_log_ratio', synthetic narrow-IQR->wide-IQR transition at t=10 yields 4.83 LARGE, within-source t=5 yields 0.048 sentinel-small (100x discrimination), silence returns sentinel 0.0, edge guard fires correctly on <3 valid F0 frames, all 81 features finite, idempotent on repeated calls.
per-domain: combined_english=0.875000 combined_korean=0.555882 combined_singing=0.295385

# 2026-04-20 — hypothesis: add voiced_f0_iqr_log_ratio (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot onto
    the F0 (pitch) content axis, which has never been touched beyond
    mean/delta/jitter/continuity (block 5). Feature = |log2(post_iqr /
    pre_iqr)| where iqr = F0 75th − 25th percentile on voiced frames in
    pre=[t-2,t] and post=[t,t+2]. Reuses cached feat_f0 + feat_f0_hop +
    feat_sr — ZERO new librosa calls, ZERO new caches. Sentinel 0.0 when
    either window has <3 valid F0 frames OR either IQR <1 Hz (log-safe
    + perceptually negligible). FEATURE_NAMES 80→81 forces retrain via
    US-505 sha gate.

(b) WHY this over recent failures. Every iteration in the last ~13 has
    been a GEOMETRY or CONTENT variant on the voiced/unvoiced paired-
    differencing template (MFCC / spec_contrast / chroma / spec_flatness
    / RMS dB). CLAUDE.md mandates structural change after 5+ same-axis
    failures. Most recent c006d52 voiced_unvoiced_rms_asymmetry_db
    produced combined=0.490700 — identical to the earlier hyperparam-
    only streak and identical per-domain breakdown (english=0.8395
    korean=0.4765 singing=0.2954), suggesting either the feature fired
    sentinel-0 everywhere or US-505 didn't retrain on the feature-add.
    Either way the paired-differencing ENERGY axis tapped out. Block 5
    already has f0_mean_pre/post/delta (linear-Hz mean shift) + jitter
    (std/mean of F0) + continuity (edge 100ms F0 delta). MISSING from
    block 5: F0 DISTRIBUTION SHAPE / SPREAD statistics — no percentile,
    no IQR, no log-scale ratio, no range. GBM max_depth=3 cannot
    synthesize log-ratio-of-IQR from existing mean/delta/jitter features
    (jitter is std/mean not IQR; delta is mean shift not spread shift).

    Mechanism on 3 surviving singing chord-cycle FPs. Within one song
    the singer holds similar phrasing style (melodic range per phrase
    ≈ 3-7 semitones) so F0 IQR of any 2s voiced slice is stable at
    ~30-80 Hz regardless of chord transition. Pre_iqr and post_iqr
    both ~50 Hz → log2(50/50)=0 → feature silent, FP not boosted.
    Cross-song splice (same singer different song): melodic style
    changes across songs — ballad pre has narrow IQR (~25 Hz), pop
    bridge post has wide IQR (~90 Hz) → log2(90/25)=1.8 LARGE.
    Different-singer cross-song: register shift AND IQR shift
    combined → feature POSITIVE. Within-song chord cycle is the
    only case where melodic style is by construction continuous,
    so IQR stability is the exact signature that discriminates the
    3 chord-cycle FPs from real splices.

    Speech self-gating via voiceprint stability. Within-speaker on
    2s windows the voiced F0 IQR is ~30-60 Hz (habitual prosodic
    range); this is a relatively stable speaker characteristic over
    2s, so log-ratio across within-file non-splice positions sits
    at ~log2(45/50)=0.15 (small). Sentence-boundary prosodic shifts
    can push IQR variation but still stay within speaker's range →
    log-ratio remains modest and uncorrelated with splice position
    → GBM low per-domain SHAP on english/korean. Log-scale
    normalization prevents absolute F0 level from dominating the
    signal (male F0=150 IQR=30 ≡ female F0=300 IQR=60 in log-ratio
    space).

    Why log-ratio over linear difference: (i) linear iqr_delta
    would be F0-level-biased — GBM would learn a singer-register
    artifact not a melodic-style artifact; (ii) log-ratio is
    symmetric around 0; (iii) ABS gives a single-sign signal GBM
    can cleanly threshold.

    Why IQR not STD: IQR is robust to yin octave-errors;
    f0_jitter_delta already uses std/mean (sensitive to outliers
    and mean shift). IQR-based spread is genuinely orthogonal.

    Orthogonal to every prior axis: NOT 1eda8e3 (cepstral cosine);
    NOT 7972a98/d290101/177d641/88adb49/6c6c254/347c0ac
    (spec_contrast variants); NOT f4148cc (chroma cosine); NOT
    0c3bf76 (spec_flatness 1D abs-delta); NOT c006d52 (RMS dB);
    NOT block 5 F0 mean/delta/jitter/continuity (all mean-based
    OR edge-based, none IQR or log-ratio); NOT percussive/harmonic
    mask family; NOT any voiced/unvoiced pair variant. FIRST
    F0-DISTRIBUTION-SHAPE feature in the 80-feature set, FIRST
    log-ratio feature in any content axis. Pure features.py — 1
    new block (~30 lines) + 1 FEATURE_NAMES append + 2 assert
    bumps (80→81) + 1 call. ZERO new librosa calls, ZERO new
    caches. Per-t cost: 2 slices + 2 nan-mask + 4 percentile ops
    + 1 log2 + 1 abs, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged / speech preserved (IQR
    signal too weak — within-song melodic range variation exceeds
    cross-song IQR shift because singers use full range within any
    given song) → fallback to voiced_f0_median_shift_cents =
    |1200*log2(post_median/pre_median)| (untried log-scale F0-
    center shift feature). (2) Speech regresses (2s IQR is NOT a
    stable voiceprint — sentence-level prosodic variation is large
    enough to produce FPs) → pivot to voiced_f0_p10_shift_cents
    (10th-percentile = F0 floor / chest register, more speaker-
    stable). (3) Combined matches 0.589282 exactly AGAIN despite
    feature-count change → rules out c006d52 sentinel-0 theory
    and points at wrapper retrain bug genuinely present; escalate.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after
    46+ iterations — cannot verify the 3 singing FPs' F0
    distributions to sanity-check the IQR hypothesis. (ii) SHAP
    rollup STILL "no keeps yet — rollup empty" for 7972a98 despite
    14 keeps — rollup writer broken. (iii) c006d52 per-domain
    (0.8395/0.4765/0.2954) EXACTLY matches prior hyperparam-only
    iterations — signal that US-505 retrain occasionally fails OR
    the feature fires sentinel-0 always; no wrapper log confirms
    which. (iv) 99081f5 4/16 capacity ghost status unclear at HEAD
    after 960113c "REVERT" that wrapper may have discarded.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE — persistent 46+-iteration blocker. Per-FP
    (domain, file, t_sec, p_splice, dsp_*, chunk_duration_s,
    f0_mean_pre, f0_mean_post, f0_iqr_pre, f0_iqr_post, top-5
    |SHAP|). (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) US-505 RETRAIN VERIFICATION TRACE — wrapper log line at
    retrain decision ("features.py sha Δ → retrain" vs "no Δ →
    skip") with a post-retrain sanity check (classifier joblib
    mtime vs features.py mtime) would surface the c006d52-style
    identical-output case.

## 2026-04-20T21:29:40+09:00 — 1d1144d (discard, combined=0.507516)
subject: add voiced_f0_median_shift_cents (FEATURE_NAMES 80->81) -- STRUCTURAL pivot to log-scale F0 CENTRAL-TENDENCY shift. feature = |1200*log2(post_median/pre_median)| on voiced frames (vp==1 AND ~isnan(f0)) in pre[t-2,t] vs post[t,t+2]. Reuses cached feat_f0 + feat_vp + feat_f0_hop + feat_sr, ZERO new librosa calls, ZERO new caches. Sentinel 0.0 when either window has <3 valid voiced frames OR either median <1 Hz (log-safe + perceptually meaningless). FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate. Explicit cited fallback from 27ddbf7(c)(1). Block 5 has f0_mean_pre/post/delta in LINEAR Hz, f0_jitter std/mean, f0_continuity edge 100ms |delta| -- but NOTHING log-scale, NOTHING median-based, NOTHING perceptually-normalized. GBM max_depth=3 cannot synthesize |1200*log2(post/pre)| from (f0_mean_pre, f0_mean_post) via threshold combinations. Different from 27ddbf7 IQR (SPREAD); this is LOCATION -- orthogonal statistical moment. Median robust to yin octave-errors; cents normalization register-invariant (male 150->160 Hz == female 300->320 Hz == 112 cents). Mechanism on 3 singing chord-cycle FPs: within one song singer holds bounded melodic center, 2s voiced-median F0 stays within ~50-150 cents of baseline -> pre/post medians similar -> |cents shift| <= 50 silent, FP not boosted. Cross-song same-singer: different melodies have different central tendencies (B3~247 -> F4~349 = 600 cents LARGE). Different-singer: register + melody shift combined, even larger. Speech self-gating: within-speaker prosodic range 3-5 semitones total spread, but MEDIAN converges to ~50-150 cents habitual band across 2s window -> GBM low per-domain SHAP on english/korean, same mechanism as existing linear f0_mean_delta. Why log-scale median cents beats linear Hz mean: (1) linear biased by F0 level (male 20 Hz != female 20 Hz perceptually); (2) mean sensitive to yin octave-error outliers, median robust; (3) |abs| removes sign noise; (4) GBM threshold splits work cleanly on bounded positive scalars. NOT a paired-difference template; single scalar log-scale location-shift. Classifier hyperparam axis verifiably broken (5+ iterations IDENTICAL 0.490700); only features.py-sha bumps force retrain. Orthogonal: NOT 1eda8e3 MFCC cosine, NOT 7972a98/d290101/177d641/88adb49/6c6c254/347c0ac spec_contrast, NOT f4148cc chroma, NOT 0c3bf76 flatness, NOT c006d52 RMS dB, NOT 27ddbf7 F0 IQR (SPREAD not LOCATION), NOT block-5 f0_mean linear Hz, NOT f0_jitter std/mean, NOT f0_continuity edge, NOT any mask variant, NOT any voiced/unvoiced paired-diff. FIRST log-scale F0 LOCATION-SHIFT, FIRST cents-normalized feature. Pure features.py change -- 1 new block (~25 lines) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call. Per-t cost 2 slices + 2 nan-mask + 2 medians + 1 log2 + 1 abs, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_f0_median_shift_cents', synthetic 220->247 Hz yields 200.4 cents (matches 1200*log2(247/220)=200.41), stationary 220->220 yields 0.0, all-unvoiced sentinel 0.0, all-NaN f0 sentinel 0.0, end-to-end extract_features returns 81 finite features, idempotent on repeated calls.
per-domain: combined_english=0.835443 combined_korean=0.542857 combined_singing=0.288235

# 2026-04-20 — hypothesis: add voiced_f0_median_shift_cents (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot to
    log-scale F0 CENTRAL-TENDENCY shift. feature =
    |1200 * log2(post_median / pre_median)| computed over voiced frames
    (vp==1 ∧ ~isnan(f0)) in pre=[t-2,t] vs post=[t,t+2]. Reuses cached
    feat_f0 + feat_vp + feat_f0_hop + feat_sr — ZERO new librosa calls,
    ZERO new caches. Sentinel 0.0 when either window has <3 valid voiced
    frames OR either median < 1 Hz (log-safe + perceptually meaningless).
    FEATURE_NAMES 80→81 forces wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. Explicit cited fallback from 27ddbf7(c)(1):
    "voiced_f0_median_shift_cents = |1200*log2(post_median/pre_median)|
    (untried log-scale F0-center shift feature)". Block 5 carries
    f0_mean_pre/post/delta in LINEAR Hz, f0_jitter (std/mean), and
    f0_continuity_score (edge 100ms linear Hz |Δ|) — but NOTHING
    log-scale, NOTHING median-based, NOTHING perceptually-normalized.
    GBM max_depth=3 / max_leaf=8 cannot synthesize
    |1200*log2(post/pre)| from (f0_mean_pre, f0_mean_post) via threshold
    combinations — it can only threshold each independently and learn
    monotone functions. Log-ratio compresses large absolute shifts and
    expands small shifts near-baseline, which is a fundamentally
    different signal surface from linear delta.

    Different from 27ddbf7 voiced_f0_iqr_log_ratio (which failed): IQR is
    SPREAD, this is LOCATION. Orthogonal statistical moments of the F0
    distribution. 27ddbf7 measured how wide the melodic range is; this
    measures where the melodic center sits. Median robust to yin
    octave-errors (unlike existing f0_mean_delta); cents normalization
    register-invariant (male 150→160 Hz ≡ female 300→320 Hz = 112 cents).

    Mechanism on 3 surviving singing chord-cycle FPs. Within one song
    the singer holds a bounded melodic center (usually within ±2-3
    semitones around the key tonic across one phrase). 2s voiced-median
    F0 stays within ~50-150 cents of within-song baseline; pre/post
    medians similar → |cents shift| ≤ 50 silent, FP not boosted. Cross-
    song splice same-singer: different song melodies have different
    central tendencies (ballad around B3 ≈ 247 Hz, pop bridge around
    F4 ≈ 349 Hz → shift 600 cents LARGE). Different-singer cross-song:
    register shift combines with melodic shift → even larger cents
    shift. Log-scale captures this range-invariant because a singer at
    220 Hz to 247 Hz (~200 cents) and one at 440 Hz to 494 Hz
    (~200 cents) are both similar-sized melodic moves in cents.

    Speech self-gating. Within-speaker prosodic pitch range across 2s
    ≈ 3-5 semitones (~300-500 cents TOTAL spread), but MEDIAN stays
    within a much narrower band (~50-150 cents) because median averages
    prosodic excursions. 2s window samples enough voiced vowels that
    median converges to speaker's habitual pitch. Sentence boundaries
    can shift median more, but still bounded → pre/post median cents
    shift modest → GBM low per-domain SHAP on english/korean, same
    self-gating mechanism as linear f0_mean_delta already enjoys.

    Why log-scale median cents beats linear Hz mean:
    (1) Linear Hz mean delta is F0-level-biased — a male singer's
    20 Hz shift and a female singer's 20 Hz shift are perceptually
    different, GBM sees them as identical Hz.
    (2) Mean is sensitive to yin octave-error outliers; median is
    robust.
    (3) |abs value| removes sign noise — either direction of shift is
    equally discriminative.
    (4) GBM threshold-based splits work more cleanly on bounded
    positive scalars than on signed linear-Hz deltas.

    Why voiced-mask not all-frames: unvoiced f0 is np.nan so must mask
    anyway; voiced-only median is the only well-defined computation.
    This is NOT a paired-difference template (single scalar, not
    asymmetry); it's a pure log-scale location-shift statistic.

    Classifier hyperparam axis verifiably broken (5+ train_classifier.py-
    only iterations IDENTICAL 0.490700); only features.py-sha bumps
    force retrain. Every prior paired-difference asymmetry variant on
    spectral axes exhausted.

    Orthogonal: NOT 1eda8e3 (MFCC cosine); NOT 7972a98/d290101/177d641/
    88adb49/6c6c254/347c0ac (spec_contrast variants); NOT f4148cc
    (chroma cosine); NOT 0c3bf76 (spec_flatness abs-delta); NOT c006d52
    (RMS dB asymmetry); NOT 27ddbf7 (F0 IQR = spread, this is location);
    NOT block-5 f0_mean_pre/post/delta (linear Hz); NOT f0_jitter
    (std/mean); NOT f0_continuity (edge 100ms); NOT any mask/
    percussive/harmonic variant; NOT any voiced/unvoiced paired-diff
    template. FIRST log-scale F0 LOCATION-SHIFT in any form, FIRST
    cents-normalized feature in the 80-feature set.

    Pure features.py change — 1 new block (~25 lines) + 1 FEATURE_NAMES
    append + 2 assert bumps (80→81) + 1 call in extract_features. ZERO
    new caches, ZERO new librosa calls. Per-t cost: 2 slices + 2 nan-
    mask + 2 medians + 1 log2 + 1 abs, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged / speech preserved (cents-shift
    signal sits in same discrimination range as linear f0_mean_delta
    already captures — GBM not learning new surface) → fallback to
    voiced_f0_p10_shift_cents (10th-percentile = F0 floor / chest
    register, more speaker-stable than median; cited 27ddbf7/c2).
    (2) Speech regresses (2s voiced-median NOT speaker-stable enough
    — sentence prosodic context large enough to push median cents shift
    past threshold on non-splice positions) → pivot to
    voiced_unvoiced_spectral_tilt_asymmetry (log-log regression slope
    asymmetry on mic/preamp-signature axis, genuinely new content).
    (3) Combined matches 0.589282 exactly → feature fires identically
    to linear f0_mean_delta within GBM's learning capacity; pivot to
    pure detector-level post-filter (e.g., local-periodicity emit
    suppression within ±8s).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 46+
    iterations — cannot verify f0 medians at the 3 singing FPs'
    positions to sanity-check the cents-shift hypothesis. Every F0
    hypothesis remains theory-calibrated. (ii) SHAP rollup STILL
    "no keeps yet — rollup empty" for 7972a98 despite 14 keeps —
    rollup writer broken; no per-feature attribution. (iii) c006d52
    per-domain EXACTLY matches hyperparam-only iterations (0.8395/
    0.4765/0.2954) — ambiguous whether US-505 retrain failed or
    feature fired sentinel-0 everywhere. No wrapper log confirms.
    (iv) 99081f5 4/16 capacity ghost status unclear at HEAD after
    960113c "REVERT" that wrapper discarded.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent 46+-iteration blocker. Per-FP (domain, file,
    t_sec, p_splice, dsp_*, chunk_duration_s, f0_mean_pre, f0_mean_post,
    voiced_unvoiced_mfcc_asymmetry, top-5 |SHAP|). Would turn every
    F0/singing hypothesis from theory bet into data-driven decision.
    (2) SHAP ROLLUP REPAIR — 14 keeps, 0 rollup entries is a
    long-standing bug; without per-feature attribution claude cannot
    see which features GBM actually uses. (3) US-505b VERIFICATION
    TRACE — wrapper log line at retrain decision ("features.py sha
    Δ → retrain" vs "no Δ → skip" vs "train_classifier.py sha Δ →
    retrain") with joblib-mtime sanity check closes feedback-loop
    verification gap and surfaces the c006d52-style identical-output
    case. Three unchanged highest-priority requests across 46+
    iterations.

## 2026-04-20T21:37:47+09:00 — 60196aa (discard, combined=0.490700)
subject: add peak-width neighbor-support gate (NEIGHBOR_SUPPORT_FRACTION=0.95) to detector.py -- STRUCTURAL pivot from feature-add exhaustion to per-emit peak-width filter on p_splice. After GBM hit_mask + DSP confirmation, require at least one adjacent grid point (t +- ANALYSIS_STRIDE_S) p_splice >= GBM_THRESHOLD * 0.95 (0.9329 floor). Enforces min peak width ~2 grid points ~0.24s at stride 0.12. Targets 7972a98 baseline (combined=0.589282) singing 0.345 3-FP chord-cycle plateau. Last 15+ iterations exhausted voiced/unvoiced paired-diff template across MFCC/spec_contrast/chroma/spec_flatness/RMS dB content axes and F0 distribution shape/location (IQR log-ratio, median cents). CLAUDE.md mandates structural change after 5+ same-axis failures. PRIMARY-tunable axis (no retrain). Mechanism: real cross-source splice produces sustained spectral discontinuity 0.3-0.5s past boundary as bus-comp/limiter attacks adapt -- p_splice stays above threshold for 3-5 consecutive grid points. Chord-cycle/phoneme-burst FPs produce narrow peaks 0.05-0.15s wide 1-2 STFT frames because chord onsets are instantaneous spectral events. Peak-width filter drops narrow FPs while preserving wide TPs. Explicit cited fallback from 1d1144d(c)(3) pure detector-level post-filter. Orthogonal to every prior DSP tweak (DSP_SUM_MIN/MAX/PHASE_MIN gated MAGNITUDE, this gates WIDTH) and stride-widening (053c0b5 0.15 uniform affected all positions; per-emit neighbor-support keeps dense scan). FIRST PEAK-WIDTH filter in detector history. Pure detector.py change -- 1 constant + 1 boolean check in hit_mask loop + 1 diag kv. Per-emit cost 2 array lookups + 2 comparisons sub-us. No retrain, no feature-set change, features.py sha stable, classifier byte-identical. Smoke-verified: NEIGHBOR_SUPPORT_FRACTION=0.95 loads, neighbor_floor=0.9329, AST parse OK 914 lines, all other tunables unchanged (GBM_THRESHOLD=0.982, MIN_SEP=3.5, STRIDE=0.12, DSP_SUM=5.0, DSP_MAX=2.0).
per-domain: combined_english=0.839506 combined_korean=0.476471 combined_singing=0.295385

# 2026-04-20 — hypothesis: peak-width neighbor-support gate (detector.py)

(a) HYPOTHESIS. Pure `splice/detector.py` add — STRUCTURAL pivot from
    feature-add to per-emit peak-width filter. After GBM hit_mask + DSP
    confirmation, require at least one adjacent grid point (t ± stride)
    to also exceed GBM_THRESHOLD * NEIGHBOR_SUPPORT_FRACTION (0.95 →
    0.933). Drops narrow-peak emits (<0.24s wide at stride 0.12) as
    chord-cycle artifacts; preserves sustained real-splice transients
    (0.3-0.5s wide). No retrain. Zero new code paths in the hot loop
    — one boolean check per hit_mask survivor.

(b) WHY this over recent failures. Last 15+ iterations exhausted the
    voiced/unvoiced paired-difference template across MFCC / spec_contrast
    / chroma / spec_flatness / RMS dB / F0 distribution shape / F0
    location — every geometry (NEAR/MID/WIDE/FAR/narrow-gap/balanced-span)
    discarded. F0 axis pivot also failed (27ddbf7 IQR 0.524, 1d1144d
    median-cents 0.508, c006d52 RMS-dB 0.491). CLAUDE.md mandates
    structural change after 5+ same-axis failures. The untried axis is
    the p_splice PEAK SHAPE itself on the time grid — existing post-
    filters tighten DSP magnitudes (SUM/MAX) or widen the stride globally
    (053c0b5 0.15 failed because uniform widening loses TP grid hits),
    but NOTHING has examined per-emit TEMPORAL WIDTH of the p_splice
    peak. Mechanism: real cross-source splice produces sustained
    spectral discontinuity (voice + accompaniment + mastering shift
    persists ~0.3-0.5s past the boundary as bus-comp / limiter attacks
    adapt) so p_splice stays above threshold for 3-5 consecutive grid
    points (at 0.12 stride). Chord-cycle / phoneme-burst FPs produce
    narrow peaks (0.05-0.15s wide, 1-2 STFT frames) because chord
    transitions are INSTANTANEOUS spectral events (chord onset → new
    note → done). Requiring at least one neighbor ≥ 0.95*threshold
    enforces peak width ≥ 2 grid points ≈ 0.24s, cleanly separating
    the distributions. Explicit cited fallback from 1d1144d(c)(3)
    "pivot to pure detector-level post-filter". Orthogonal to every
    prior DSP tweak (DSP_SUM_MIN/MAX/PHASE_MIN all gated MAGNITUDE,
    this gates WIDTH) and every stride-widening experiment (053c0b5
    uniform stride 0.15 affected ALL grid positions; this filters
    per-emit). FIRST PEAK-WIDTH filter in detector history. Blast
    radius: 1 constant + 1 boolean check in the hit_mask loop. Per-emit
    cost: 2 array lookups + 2 comparisons, sub-µs.

(c) IF THIS FAILS. (1) Speech regresses (narrow legit hard-cut TPs
    produce 1-grid-point peaks) → loosen NEIGHBOR_SUPPORT_FRACTION
    0.95 → 0.85 so only extremely narrow peaks drop. (2) Singing
    unchanged (3 chord-cycle FPs happen to produce wide peaks too —
    chord sustains across transition) → tighten to REQUIRE both
    neighbors ≥ 0.95*threshold (wider peak requirement, 0.48s minimum
    width). (3) combined matches 0.589282 exactly → threshold
    fraction set too loose and no peaks were actually dropped;
    fallback is fraction=1.0 (strict consecutive threshold crossing).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 46+
    iterations — cannot measure the actual peak-width distribution
    of the 3 surviving singing FPs vs real TPs, so width threshold
    (0.95 fraction) is theory-calibrated. (ii) SHAP rollup STILL
    empty for 7972a98 despite 14 keeps — rollup writer broken.
    (iii) 99081f5 4/16 capacity ghost status unclear at HEAD after
    960113c "REVERT" that wrapper may have discarded.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE with PEAK-WIDTH fields — per-FP (domain, file,
    t_sec, p_splice_at_t, p_splice_at_t_minus_stride,
    p_splice_at_t_plus_stride, chunk_duration_s, top-5 |SHAP|) so
    peak-width hypotheses become data-driven. (2) SHAP ROLLUP
    REPAIR — 14 keeps, 0 rollup entries. (3) US-505b VERIFICATION
    TRACE — confirm hyperparam retrains fire post-920dcdb.

## 2026-04-20T22:00:47+09:00 — 0cdd87e (discard, combined=0.507837)
subject: add voiced_unvoiced_zcr_asymmetry (FEATURE_NAMES 80->81) -- STRUCTURAL pivot onto TIME-DOMAIN content axis. Apply proven voiced/unvoiced paired-difference template (1eda8e3 NEAR MFCC kept +0.054 biggest win, 7972a98 NEAR spec_contrast kept +0.005) to zero-crossing rate (ZCR) rather than any spectral/cepstral/pitch/energy axis. pre[t-2,t] post[t,t+2]: voiced_delta=mean(zcr[voiced in post])-mean(zcr[voiced in pre]), unvoiced_delta=mean(zcr[unvoiced in post])-mean(zcr[unvoiced in pre]), feature=unvoiced_delta-voiced_delta. Reuses cached feat_zcr (librosa ZCR hop 512) + feat_vp (voicing probability hop 512 -- same frame rate, trivial alignment). ZERO new librosa calls, ZERO new caches. Sentinel 0.0 when any mask empty. FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate. Last 18+ iterations exhausted spectral/cepstral content axes (MFCC/spec_contrast/chroma/spec_flatness at NEAR/MID/WIDE/FAR/narrow-gap/balanced-span), energy-level (c006d52 RMS dB -> 0.491), F0 distribution shape+location (27ddbf7 IQR -> 0.524, 1d1144d median cents -> 0.508), and detector post-filter (60196aa peak-width -> 0.491). CLAUDE.md mandates structural change after 5+ same-axis failures. Block 6 has zcr_pre/zcr_post/zcr_delta (all-frame, no voicing split) but GBM has never seen ZCR paired-differenced on a voicing mask. ZCR is genuinely TIME-DOMAIN (count of sign crossings per frame) -- orthogonal to every prior content axis, all of which are spectral/cepstral/pitch-shape/energy. Mic+preamp high-frequency response and codec quantization drive ZCR baseline on unvoiced frames; vocal-tract resonance drives voiced ZCR baseline. Cross-song splice crosses mic+preamp (and often codec in tier1/tier2 mixed-format chunks) -> ZCR baseline shifts DIFFERENTLY on voiced (stable vocal-tract signal) vs unvoiced (noise-dominated, tracks mic). Mechanism on 3 singing chord-cycle FPs: within one song mic/preamp/codec frozen -> voiced ZCR stays at singer baseline (~4-8/frame), unvoiced ZCR stays at mic noise-floor baseline (~15-25/frame); chord transition does NOT change mic/preamp/codec -> voiced_delta and unvoiced_delta both ~0 -> asymmetry ~0 silent, FP not boosted. Same-singer cross-song: voiced_delta small (similar formants -> similar ZCR on vowels), unvoiced_delta LARGE because new mic noise floor + new codec quantization -> asymmetry POSITIVE, real TP boosted. Different-singer cross-song: both shift; 1eda8e3 MFCC asymmetry already fires. Speech self-gating via paired differencing (reproduces 1eda8e3 mechanism): on english/korean voiced ZCR and unvoiced ZCR both vary with phoneme density but within SAME recording mic/preamp continuous -> voiced+unvoiced baselines shift together with phoneme context -> DIFFERENCE zero-mean noise -> GBM low per-domain SHAP on english/korean. Why ZCR over remaining 1D scalar candidates (spec_bandwidth/centroid/rolloff): spectral moments already seen via all-frame deltas (block 2 top-SHAP) and failed to bite 3 chord-cycle FPs; ZCR is the one time-domain scalar in cached context absent from any paired-diff feature. Why NOT another 1D scalar failure mode: c006d52 RMS dB failed (auto-leveling flattens mastering), F0 IQR/median failed (within-speaker range overlaps cross-song at 2s), spec_flatness failed (Wiener entropy tracks chord-cycle). ZCR structurally different -- COUNT signal, integer-like, dominated by high-freq+codec not loudness/pitch/tonality. Orthogonal: NOT 1eda8e3 (cepstral cosine), NOT 7972a98+variants (spec_contrast), NOT f4148cc (chroma), NOT 0c3bf76 (flatness abs-delta), NOT c006d52 (RMS dB), NOT 27ddbf7/1d1144d (F0), NOT block-6 zcr_delta (all-frame no mask), NOT 60196aa peak-width (detector), NOT any percussive/harmonic/tonnetz/geometry variant. FIRST TIME-DOMAIN voiced/unvoiced paired-diff feature in 80-feature set, FIRST mask-split ZCR. Pure features.py change -- 1 new block (~40 lines cloned from _block_voiced_unvoiced_mfcc_asymmetry with 1D scalar mean replacing 13-dim cosine) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. ZERO new caches, ZERO new librosa calls. Per-t cost: 4 slices + 4 masked means + 2 subtractions, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_zcr_asymmetry', real train/singing/tier1 audio t=5s yields asym=+0.00151 (finite non-trivial), sentinel 0.0 on all-voiced 2s windows (common -- feature fires sparsely on mixed-voicing content), all 81 features finite on synthetic + real audio, idempotent on repeated calls.
per-domain: combined_english=0.825000 combined_korean=0.537313 combined_singing=0.295455

# 2026-04-20 — hypothesis: add voiced_unvoiced_zcr_asymmetry (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot onto a
    TIME-DOMAIN content axis. Apply the proven voiced/unvoiced paired-
    difference template (1eda8e3 NEAR MFCC kept +0.054 biggest win,
    7972a98 NEAR spec_contrast kept +0.005) to zero-crossing rate (ZCR)
    rather than any spectral / cepstral / pitch / energy axis. For
    pre=[t-2,t] and post=[t,t+2]:
        voiced_delta  = mean(zcr[voiced in post]) - mean(zcr[voiced in pre])
        unvoiced_delta = mean(zcr[unvoiced in post]) - mean(zcr[unvoiced in pre])
        feature = unvoiced_delta - voiced_delta
    Reuses cached feat_zcr (librosa ZCR, hop 512) + feat_vp (voicing
    probability, hop 512 — same frame rate, trivial alignment). ZERO new
    librosa calls, ZERO new caches. Sentinel 0.0 when any mask empty.
    FEATURE_NAMES 80→81 forces wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. Last 18+ iterations exhausted
    spectral/cepstral content axes (MFCC / spec_contrast / chroma /
    spec_flatness at NEAR/MID/WIDE/FAR/narrow-gap/balanced-span
    geometries), energy-level (RMS dB asymmetry c006d52 → 0.491), F0
    distribution shape+location (IQR log-ratio 27ddbf7 → 0.524; median
    cents 1d1144d → 0.508), and detector-side post-filters (peak-width
    gate 60196aa → 0.491 identical-streak). CLAUDE.md mandates structural
    change after 5+ same-axis failures. Block 6 has zcr_pre / zcr_post /
    zcr_delta (all-frame, no voicing split) — GBM has never seen ZCR
    paired-differenced on a voicing mask. ZCR is genuinely TIME-DOMAIN
    (count of sign crossings per frame) — orthogonal to every prior
    content axis, all of which are spectral/cepstral/pitch-shape/energy.

    Mic + preamp high-frequency response and quantization noise floor
    directly drive ZCR baseline on unvoiced frames; vocal-tract
    resonance drives voiced ZCR baseline. Cross-song splice crosses
    mic + preamp (and often codec in tier1/tier2 mixed-format chunks)
    → ZCR baseline shifts DIFFERENTLY on voiced (stable vocal-tract
    signal) vs unvoiced (noise-dominated, tracks the mic).

    Mechanism on 3 surviving singing chord-cycle FPs. Within one song
    mic/preamp/codec are frozen. Voiced ZCR stays at singer's baseline
    (~4-8/frame on sustained tones). Unvoiced ZCR stays at mic's
    noise-floor baseline (~15-25/frame). Chord transition does NOT
    change mic/preamp/codec → voiced_delta ≈ 0 AND unvoiced_delta ≈ 0
    → asymmetry ≈ 0, feature silent, FP not boosted. Same-singer
    cross-song: voiced_delta small (similar formants → similar ZCR on
    vowels), unvoiced_delta LARGE because new mic noise floor + new
    codec quantization produce distinct ZCR baseline → asymmetry
    POSITIVE. Different-singer cross-song: both shift; 1eda8e3 MFCC
    asymmetry already fires.

    Speech self-gating via paired differencing (reproduces 1eda8e3
    mechanism). On english/korean voiced ZCR and unvoiced ZCR both
    vary with phoneme density but within the SAME recording mic/preamp
    is continuous → voiced and unvoiced baselines shift together with
    phoneme context → DIFFERENCE is zero-mean noise across non-splice
    windows → GBM low per-domain SHAP on english/korean → feature
    functionally invisible on speech.

    Why ZCR over remaining 1D scalar candidates. spec_centroid /
    rolloff / bandwidth are spectral moments; GBM already sees their
    all-frame deltas (block 2, top-SHAP) and they failed to bite the
    3 chord-cycle FPs. Voicing-split on SAME spectral moments unlikely
    to surface new signal. ZCR is the one time-domain scalar in the
    cached context absent from any paired-diff feature.

    Why NOT another 1D scalar failure mode. c006d52 RMS dB → 0.491
    failed because auto-leveling flattens mastering dB differences;
    1d1144d/27ddbf7 F0 failed because within-speaker prosodic range
    overlaps cross-song range at 2s windows; 0c3bf76 spec_flatness
    failed because Wiener entropy tracks chord-cycle tonality shifts.
    ZCR is structurally different — COUNT signal, integer-like,
    dominated by high-frequency content + codec quantization, not by
    loudness / pitch distribution / tonal balance.

    Orthogonal. NOT 1eda8e3 (cepstral cosine); NOT 7972a98 and every
    spec_contrast variant (d290101/177d641/88adb49/6c6c254/347c0ac);
    NOT f4148cc (chroma); NOT 0c3bf76 (spec_flatness abs-delta); NOT
    c006d52 (RMS dB); NOT 27ddbf7 / 1d1144d (F0); NOT block-6
    zcr_delta (all-frame, no mask split); NOT 60196aa peak-width
    (detector); NOT any percussive/harmonic/tonnetz/geometry-variant.
    FIRST TIME-DOMAIN voiced/unvoiced paired-diff feature, FIRST
    mask-split ZCR.

    Blast radius: 1 new block (~40 lines cloned from
    _block_voiced_unvoiced_mfcc_asymmetry with 1D scalar mean replacing
    13-dim cosine) + 1 FEATURE_NAMES append + 2 assert bumps (80→81) +
    1 call in extract_features. ZERO new caches, ZERO new librosa
    calls. Per-t cost: 4 slices + 4 masked means + 2 subtractions,
    sub-ms.

(c) IF THIS FAILS. (1) Speech regresses (ZCR has phoneme-dependent
    variation that 2s window doesn't average away) → fallback to
    voiced_unvoiced_spec_bandwidth_asymmetry (spectral spread, EQ
    signature). (2) Singing flat (mic baseline shift below ZCR 2s-
    frame-count resolution) → expand windows to ±4s. (3) combined
    matches 0.4907 AGAIN → confirms the identical-streak spans
    features.py AND detector.py changes; escalate as wrapper/cache
    coherence bug.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 47+
    iterations. Every scalar-asymmetry hypothesis remains theory-
    calibrated. (ii) SHAP rollup STILL empty for 14 keeps — no per-
    feature attribution. (iii) The 0.490700 identical-streak crosses
    features.py (c006d52) AND detector.py (60196aa) — no wrapper
    signal to confirm whether US-505 retrain actually fired OR
    whether detector.py change took effect. Cannot distinguish
    "feature silent" from "classifier stale" from "wrapper cache hit".
    (iv) 99081f5 4/16 capacity ghost status unclear.

(e) Wrapper enhancements.
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — persistent
    47+-iteration blocker. Per-FP (domain, file, t_sec, p_splice,
    dsp_*, chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, zcr_pre, zcr_post,
    top-5 |SHAP|). Transforms every hypothesis from theory bet to
    data-driven decision.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps is long-
    standing bug; without per-feature attribution claude picks
    "theoretically orthogonal" not "what GBM actually uses".
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision: "features.py sha Δ XX→YY → retrain" vs "detector.py
    change only, classifier sha YY stable" with joblib-mtime sanity
    check. Would isolate the 0.4907 identical-streak root cause.
    Three unchanged highest-priority requests across 47+ iterations.

## 2026-04-20T22:17:04+09:00 — 7a170b0 (discard, combined=0.542324)
subject: add voiced_unvoiced_spec_bandwidth_asymmetry (FEATURE_NAMES 80->81) -- cited fallback from 0cdd87e(c)(1). Voicing-masked paired-diff on the 1D spectral-bandwidth second-order moment (std around centroid), the only untried 1D spectral moment on the proven voiced/unvoiced template (1eda8e3 MFCC +0.054 biggest keep, 7972a98 spec_contrast +0.005). pre[t-2,t] post[t,t+2]: voiced_delta=mean(bw[voiced post])-mean(bw[voiced pre]); unvoiced_delta=mean(bw[unvoiced post])-mean(bw[unvoiced pre]); feature=unvoiced_delta-voiced_delta. Reuses cached feat_bandwidth (hop=512 (1,n)) + feat_vp (hop=512 (n,)), ZERO new librosa calls, ZERO new caches. Sentinel 0.0 on empty mask. FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate. Last 20+ iterations exhausted voiced/unvoiced paired-diff across MFCC/spec_contrast/chroma/spec_flatness/RMS-dB/ZCR + every geometry + F0 distribution shape+location + detector post-filter. CLAUDE.md mandates structural change after 5+ same-axis failures; bandwidth is the ONE untried 1D spectral moment on the paired-diff template. Block 2 carries spec_bandwidth_delta ALL-FRAME only. GBM max_depth=3 cannot synthesize voicing-conditional bandwidth from (bandwidth_delta, voicing_prob_pre, voicing_prob_post) via independent threshold splits; paired-diff is a subtraction interaction trees cannot express. Mechanism on 3 singing chord-cycle FPs: within-song mic+EQ+compression frozen, voiced bandwidth at singer-formant baseline ~500-1200Hz spread, unvoiced at drums+noise-floor ~2000-4000Hz spread, chord transition does NOT change mic/EQ -> voiced_delta~0 AND unvoiced_delta~0 -> asymmetry~0 silent. Same-singer cross-song: voiced_delta small (similar formant spread), unvoiced_delta LARGE (new drum-bus EQ + different mastering limiter shapes post-vowel-tail spread) -> asymmetry POSITIVE, TP boosted. Different-singer: both shift; existing 1eda8e3 MFCC asymmetry fires. Speech self-gating (1eda8e3 mechanism): voiced bandwidth varies with vowel identity (i/a/u spread) but within same recording mic+EQ continuous -> voiced+unvoiced deltas both respond to phoneme context correlated -> DIFFERENCE is zero-mean noise -> GBM low per-domain SHAP on english/korean. Why bandwidth over remaining 1D moments: spec_centroid (F0-correlated on voiced, partial dup of F0_mean_delta); spec_rolloff (percentile, coarser proxy for same EQ info); spec_flatness already tried (0c3bf76 0.539). Bandwidth is second-order spectral moment (std around centroid), more directly mastering-EQ-signature than centroid, NOT duplicated by any existing paired-diff feature. Classifier hyperparam axis verifiably broken (5+ iterations IDENTICAL 0.490700); only features.py-sha bumps force retrain. Orthogonal: NOT 1eda8e3 (MFCC cosine); NOT 7972a98/d290101/177d641/88adb49/6c6c254/347c0ac (spec_contrast); NOT f4148cc (chroma); NOT 0c3bf76 (spec_flatness abs-delta asym); NOT c006d52 (RMS dB); NOT 0cdd87e (ZCR); NOT 27ddbf7/1d1144d (F0); NOT block-2 spec_bandwidth_delta (all-frame no mask); NOT 60196aa peak-width. FIRST spec_bandwidth voicing-masked paired-diff, SECOND-ORDER spectral moment axis. Pure features.py change -- 1 new block (~35 lines cloned from _block_voiced_unvoiced_mfcc_asymmetry with 1D scalar mean replacing 13-dim cosine) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. Per-t cost 4 slices + 4 masked means + 2 subtractions, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_spec_bandwidth_asymmetry', mixed-voicing synthetic audio with EQ-shifted noise-floor yields -228.8 at splice boundary vs +42.3/-18.2 within-source (~5x discrimination), empty-mask sentinel 0.0, all 81 features finite, idempotent on repeated calls.
per-domain: combined_english=0.814815 combined_korean=0.604545 combined_singing=0.323810

# 2026-04-20 — hypothesis: add voiced_unvoiced_spec_bandwidth_asymmetry (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — apply the proven voiced/unvoiced
    paired-difference template (1eda8e3 NEAR MFCC +0.054 biggest win, 7972a98
    NEAR spec_contrast +0.005) to the untried 1D SPECTRAL-BANDWIDTH moment.
    For pre=[t-2,t] and post=[t,t+2]:
        voiced_delta   = mean(bw[voiced in post]) - mean(bw[voiced in pre])
        unvoiced_delta = mean(bw[unvoiced in post]) - mean(bw[unvoiced in pre])
        feature        = unvoiced_delta - voiced_delta
    Reuses cached feat_bandwidth (librosa.spectral_bandwidth, hop=512, shape
    (1,n)) + feat_vp (voicing probability, hop=512, shape (n,)) — ZERO new
    librosa calls, ZERO new caches. Sentinel 0.0 on empty mask. FEATURE_NAMES
    80→81 forces wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. EXPLICIT cited fallback from 0cdd87e(c)(1):
    "voiced_unvoiced_spec_bandwidth_asymmetry (spectral spread, EQ signature)".
    Last 20+ iterations exhausted voiced/unvoiced paired-diff across MFCC
    (1eda8e3 kept), spec_contrast (7972a98 kept, d290101/177d641/88adb49/
    6c6c254/347c0ac discards), chroma (f4148cc discard), spec_flatness
    (0c3bf76 → 0.539), RMS dB (c006d52 → 0.491), ZCR (0cdd87e → 0.508);
    every geometry (NEAR/MID/WIDE/FAR/narrow-gap/balanced-span); F0
    distribution (IQR 27ddbf7 → 0.524, median-cents 1d1144d → 0.508); and
    detector post-filter (60196aa → 0.491). CLAUDE.md mandates structural
    change after 5+ same-axis failures; bandwidth is the ONE untried 1D
    spectral moment on the paired-diff template.

    Block 2 carries spec_bandwidth_delta ALL-FRAME — GBM has never seen
    bandwidth voicing-split. GBM max_depth=3 cannot synthesize voicing-
    conditional bandwidth from (bandwidth_delta, voicing_prob_pre,
    voicing_prob_post) via independent threshold splits; paired-diff is
    a subtraction interaction trees cannot express.

    Mechanism on 3 surviving singing chord-cycle FPs. Within one song mic
    + mastering EQ + bus compression are FROZEN. Voiced bandwidth stays
    at singer-formant baseline (~500-1200 Hz spread around centroid),
    unvoiced bandwidth stays at drums+noise-floor baseline (~2000-4000
    Hz). Chord transition does NOT change mic/EQ → voiced_delta ≈ 0 AND
    unvoiced_delta ≈ 0 → asymmetry ≈ 0, feature silent, FP not boosted.
    Same-singer cross-song: voiced_delta small (similar formant spread),
    unvoiced_delta LARGE (new drum-bus EQ + different mastering limiter
    shapes the post-vowel-tail spread differently) → asymmetry POSITIVE,
    TP boosted. Different-singer: both shift; existing 1eda8e3 MFCC
    asymmetry fires.

    Speech self-gating via paired differencing (1eda8e3 mechanism). Voiced
    bandwidth on speech varies with vowel identity (i ≠ a ≠ u in spread)
    but within same recording mic+EQ is continuous → voiced and unvoiced
    bandwidth deltas both respond to phoneme context in correlated ways
    → DIFFERENCE is zero-mean noise across non-splice positions → GBM
    low per-domain SHAP on english/korean → functionally invisible on
    speech.

    Why bandwidth over remaining 1D moments. (i) spec_centroid — existing
    centroid_delta is top-SHAP AND centroid is F0-correlated on voiced
    frames (voiced ⊂ pitched) so voicing-split partially duplicates F0
    mean-delta signal. (ii) spec_rolloff — 85th-percentile, coarser proxy
    for same EQ info. (iii) spec_flatness — ALREADY tried (0c3bf76 →
    0.539). Bandwidth is the second-order spectral moment (std around
    centroid), more directly mastering-EQ-signature than centroid AND
    not duplicated by any existing paired-diff feature.

    Classifier hyperparam axis verifiably broken (5+ train_classifier.py-
    only iterations IDENTICAL 0.490700). Only features.py-sha bumps force
    retrain. Every primary tunable saturated both directions.

    Orthogonal. NOT 1eda8e3 (MFCC cosine); NOT 7972a98 + every spec_contrast
    variant; NOT f4148cc (chroma); NOT 0c3bf76 (spec_flatness abs-delta
    asymmetry); NOT c006d52 (RMS dB); NOT 0cdd87e (ZCR); NOT 27ddbf7/
    1d1144d (F0 distribution); NOT block-2 spec_bandwidth_delta (all-frame,
    no voicing split); NOT 60196aa peak-width (detector); NOT any
    percussive/harmonic/tonnetz/geometry variant. FIRST spec_bandwidth on
    voicing-masked paired-diff, SECOND-ORDER spectral moment axis.

    Blast radius: 1 new block (~35 lines cloned from
    _block_voiced_unvoiced_mfcc_asymmetry with 1D scalar mean replacing
    13-dim cosine, matching c006d52/0cdd87e 1D template) + 1 FEATURE_NAMES
    append + 2 assert bumps (80→81) + 1 call in extract_features. ZERO
    new caches, ZERO new librosa calls. Per-t cost: 4 slices + 4 masked
    means + 2 subtractions, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged / speech preserved (bandwidth
    signal sits in same GBM-learnable surface as centroid_delta; voicing-
    split insufficient) → fallback to voiced_unvoiced_rolloff_asymmetry
    (percentile-based cutoff, different mastering fingerprint than
    bandwidth). (2) Speech regresses (voiced bandwidth tracks vowel
    identity strongly enough that 2s window doesn't average it away) →
    pivot to wider 4s-balanced-adjacent windows on bandwidth (more
    phoneme averaging). (3) combined matches 0.4907 AGAIN → confirms
    the identical-streak spans BOTH features.py AND detector.py edits;
    escalate as wrapper cache-coherence bug.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 47+
    iterations — cannot verify whether the 3 singing FPs sit in regions
    where bandwidth-delta discriminates vs flatlines. (ii) SHAP rollup
    STILL empty for 14 keeps — no per-feature attribution so axis
    selection is theory-only. (iii) The 0.490700 identical-streak
    crosses features.py (c006d52) AND detector.py (60196aa) — no
    wrapper signal to confirm retrain fired OR detector change took
    effect. Cannot distinguish "feature silent" from "classifier stale"
    from "wrapper cache hit". (iv) 99081f5 4/16 capacity ghost status
    unclear at HEAD after 960113c "REVERT" wrapper may have discarded.

(e) Wrapper enhancements.
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — persistent 47+-
    iteration blocker. Per-FP (domain, file, t_sec, p_splice, dsp_*,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, spec_bandwidth_pre,
    spec_bandwidth_post, top-5 |SHAP|). Transforms every hypothesis from
    theory bet to data-driven decision.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps; without per-
    feature attribution I pick "theoretically orthogonal" not "what GBM
    actually uses."
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision ("features.py sha Δ XX→YY → retrain" vs "no Δ → skip") with
    joblib-mtime sanity check would isolate the 0.4907 identical-streak
    root cause. Three unchanged highest-priority requests across 47+
    iterations.

## 2026-04-20T22:37:39+09:00 — 8fc7169 (discard, combined=0.471813)
subject: add voicing_transition_rate_delta (FEATURE_NAMES 80->81) -- STRUCTURAL pivot onto VOICING-MASK TEMPORAL STRUCTURE itself, not content of voiced/unvoiced split. feature = |post_rate - pre_rate| where rate = count of voiced<->unvoiced transitions per second on binary mask (feat_vp > 0.5) in pre[t-2,t] vs post[t,t+2]. Reuses feat_vp only, ZERO new librosa calls, ZERO new caches. FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate. 20+ iterations exhausted voiced/unvoiced paired-diff template across MFCC/spec_contrast/chroma/spec_flatness/RMS-dB/ZCR/spec_bandwidth content axes + F0 distribution shape+location + detector peak-width. CLAUDE.md mandates structural change after 5+ same-axis failures. Every prior voiced/unvoiced asymmetry uses the voicing mask to SELECT FRAMES for content computation; the VOICING MASK ITSELF as a signal (its TEMPORAL PATTERN) is genuinely untried. Block 5 has voicing_prob_pre/post/delta but these are MEANS (duty cycle), NOT RATE of transitions. GBM max_depth=3 cannot synthesize transition count from mean alone; vp_mean=0.7 can arise from one long voiced run (0 transitions) or 5 alternating runs (8 transitions) and mean collapses them identically. Mechanism on 3 surviving singing chord-cycle FPs: singer's phrasing rhythm (syllables/s, breath pattern, consonant density) is a habitual style held constant across verse/chorus; chord transitions do NOT change phrasing rhythm -> pre_rate ~ post_rate -> |delta| ~ 0 silent, FP not boosted. Cross-song: different tempos/styles (ballad 80 BPM legato vs up-tempo 140 BPM staccato) -> transitions/s shifts meaningfully 2.0 -> 3.5, feature POSITIVE, TP boosted. Speech self-gating: within-recording speaker speech rate consistent, pre/post rates within +-0.5/s -> feature small -> GBM low per-domain SHAP on english/korean. Cross-speaker splice (TP) crosses different speech rates -> feature fires contributing to discrimination. Why rate over alternatives: duty cycle (mean vp) already in-set; longest-run inversely proportional to rate for fixed mean; run-length variance noisy on 2s (few runs). Transition count/s is simplest robust rhythm signature absent from feature set. Orthogonal: NOT 1eda8e3 (MFCC content cosine); NOT 7972a98/d290101/177d641/88adb49/6c6c254/347c0ac (spec_contrast content variants); NOT f4148cc (chroma content); NOT 0c3bf76 (spec_flatness content); NOT c006d52 (RMS dB content); NOT 0cdd87e (ZCR content); NOT 7a170b0 (spec_bandwidth content); NOT 27ddbf7/1d1144d (F0 distribution); NOT block-5 voicing_prob_pre/post/delta (voicing MEAN not RATE); NOT 60196aa detector peak-width; NOT block-1 dsp_pairwise_proximity. FIRST voicing-mask-TEMPORAL-STRUCTURE feature, FIRST rhythm feature derived from voicing mask itself. Pure features.py change -- 1 new block (~20 lines) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. Per-t cost 2 slices + 2 boolean casts + 2 diff+sum ops on ~172-frame arrays + 2 divisions, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voicing_transition_rate_delta', synthetic alternating voiced/unvoiced 0.5s segments yields delta=1.0 finite non-trivial, stationary all-voiced returns 0.0, all 81 features finite on synthetic mixed audio, idempotent on repeated calls, voicing_prob_pre~0.99 vs voicing_transition_rate_delta=1.0 confirm they measure distinct aspects of voicing mask.
per-domain: combined_english=0.804878 combined_korean=0.485714 combined_singing=0.268657

# 2026-04-20 — hypothesis: add voicing_transition_rate_delta (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot onto the
    VOICING MASK TEMPORAL STRUCTURE itself (not the content of voiced/
    unvoiced frames). Feature = |post_rate − pre_rate| where rate = count
    of voiced↔unvoiced transitions per second in [t−2,t] vs [t,t+2],
    computed on binary mask (feat_vp > 0.5). Reuses cached feat_vp only —
    ZERO new librosa calls, ZERO new caches. FEATURE_NAMES 80→81 forces
    wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. 20+ consecutive iterations have failed
    on the voiced/unvoiced PAIRED-DIFFERENCE template across MFCC /
    spec_contrast / chroma / spec_flatness / RMS-dB / ZCR / spec_bandwidth
    content axes, F0 distribution shape+location (27ddbf7 IQR, 1d1144d
    median-cents), and detector-side peak-width (60196aa). CLAUDE.md
    mandates structural change after 5+ same-axis failures. Every prior
    voiced/unvoiced asymmetry uses the voicing mask to SELECT FRAMES for
    content computation; the VOICING MASK ITSELF as a signal (its
    TEMPORAL PATTERN across time, not content of masked frames) is
    genuinely untried. Block 5 carries voicing_prob_pre/post/delta — but
    these are MEANS of feat_vp over the window (voicing density), NOT the
    RATE of voiced↔unvoiced TRANSITIONS. GBM max_depth=3 cannot synthesize
    transition count from mean alone; vp_mean=0.7 can arise from one long
    voiced segment (0 transitions) or from 5 alternating voiced/unvoiced
    segments (8 transitions), and the mean collapses them identically.

    Mechanism on 3 surviving singing chord-cycle FPs. Within one song the
    singer's phrasing style (syllable rate, breath pattern, consonant
    density) is a LEARNED HABITUAL RHYTHM — a pop singer produces ~2-3
    voiced↔unvoiced transitions per second consistently across verse /
    chorus / bridge, dominated by syllable boundaries and word-internal
    plosives. Chord transitions do NOT change the singer's phrasing
    rhythm; the drummer and rhythm section stay in the same tempo;
    transitions/s is continuous across chord boundaries → pre_rate ≈
    post_rate → |delta| ≈ 0, feature silent, FP not boosted.

    Cross-song splice: different songs have different tempos (80 BPM
    ballad vs 140 BPM up-tempo), different vocal phrasing styles (legato
    vs staccato), different syllable-per-measure ratios → transitions/s
    shifts meaningfully (e.g., 2.0 → 3.5). Same-singer cross-song still
    sees rate shift because tempo differs. Different-singer cross-song:
    phrasing style plus tempo compound.

    Speech self-gating. Within-recording speech rate is speaker-
    consistent; 2s windows span ~4-8 syllables giving stable voicing
    transition rates per speaker (English ~4-6 tr/s; Korean ~3-5 tr/s).
    Sentence boundaries add a brief unvoiced region but don't change the
    sustained speech rate. Pre/post from same speaker at same rate →
    rates within ±0.5/s → feature small → GBM low per-domain SHAP on
    english/korean. Cross-speaker splice (TP) would cross speakers with
    different speech rates → fires meaningfully — contributing to
    discrimination, not FP regression.

    Why rate over other mask-structure statistics. (1) Duty cycle (mean
    vp) already exists as voicing_prob_delta. (2) Longest-run length is
    correlated with rate (inverse for fixed mean). (3) Run-length variance
    is noisy on 2s windows. Transition count per second is the simplest,
    most robust RHYTHM signature the existing feature set doesn't expose.

    Orthogonal. NOT 1eda8e3 (MFCC content cosine); NOT 7972a98 / d290101
    / 177d641 / 88adb49 / 6c6c254 / 347c0ac (spec_contrast content
    variants); NOT f4148cc (chroma content); NOT 0c3bf76 (spec_flatness
    content abs-delta); NOT c006d52 (RMS dB content); NOT 0cdd87e (ZCR
    content); NOT 7a170b0 (spec_bandwidth content); NOT 27ddbf7 / 1d1144d
    (F0 distribution); NOT block-5 voicing_prob_pre/post/delta (voicing
    MEAN, not RATE); NOT block-1 dsp_pairwise_proximity; NOT detector-
    side (pure features.py add). FIRST voicing-mask-TEMPORAL-STRUCTURE
    feature; FIRST rhythm feature derived from the voicing mask itself.

    Blast radius: 1 new block function (~20 lines) + 1 FEATURE_NAMES
    append + 2 assert bumps (80→81) + 1 call in extract_features. ZERO
    new caches, ZERO new librosa calls. Per-t cost: 2 slices + 2 boolean
    comparisons + 2 diff+sum ops on ~172-frame arrays + 2 divisions,
    sub-ms.

(c) IF THIS FAILS. (1) Speech regresses (voicing transition rate varies
    with sentence prosodic stress enough that within-recording
    fluctuation exceeds cross-source shift) → fallback to
    voicing_longest_run_delta (longest contiguous voiced stretch length,
    more robust against burst-noise flipping the mask). (2) Singing flat
    (within-song transition rate already stable AND cross-song shifts are
    too small at 2s windows — ~2 tr/s gives ~8 transitions per window,
    small counting resolution) → pivot to voicing_transition_rate_delta
    with wider 4s windows (more counts for stable statistics). (3)
    Combined matches 0.490700 identical-streak AGAIN → features.py-sha
    bump fails to force retrain on feature-count changes too; escalate
    as systemic wrapper cache-coherence bug spanning features.py AND
    detector.py.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 48+
    iterations — cannot inspect the 3 singing FPs' voicing-mask
    transition rates to verify they're continuous across chord boundaries
    (as theory predicts) vs already differ. Every mask-statistic
    hypothesis theory-calibrated. (ii) SHAP rollup STILL "no keeps yet —
    rollup empty" for 7972a98 despite 14 keeps — rollup writer broken;
    no per-feature attribution so axis picks are blind. (iii) 0.490700
    identical streak crosses features.py AND detector.py changes without
    any wrapper signal distinguishing "feature silent" from "classifier
    stale" from "wrapper cache hit." (iv) 99081f5 4/16 capacity ghost
    status unclear at HEAD after 960113c "REVERT".

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent 48+-iteration blocker cited in every reflection.
    Per-FP (domain, file, t_sec, p_splice, dsp_*, chunk_duration_s,
    voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, voicing_prob_pre,
    voicing_prob_post, top-5 |SHAP|). Transforms every hypothesis from
    theory bet to data-driven decision. (2) SHAP ROLLUP REPAIR — rollup
    empty for 14 keeps; without per-feature attribution feature-selection
    is blind. (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at
    retrain decision: "features.py sha Δ XX→YY → retrain" vs "no Δ →
    skip" with joblib-mtime sanity check after retrain. Would isolate
    the 0.4907 identical-streak root cause. Three unchanged highest-
    priority requests across 48+ iterations.

## 2026-04-20T22:56:05+09:00 — e234649 (discard, combined=0.507418)
subject: add voiced_unvoiced_spec_rolloff_asymmetry (FEATURE_NAMES 80->81) -- last untried 1D spectral moment on the proven voiced/unvoiced paired-diff template (1eda8e3 NEAR MFCC kept +0.054 biggest win, 7972a98 NEAR spec_contrast kept +0.005). pre[t-2,t] post[t,t+2]: voiced_delta=mean(rolloff[voiced post])-mean(rolloff[voiced pre]); unvoiced_delta=mean(rolloff[unvoiced post])-mean(rolloff[unvoiced pre]); feature=unvoiced_delta-voiced_delta. Reuses cached feat_rolloff (librosa.spectral_rolloff, hop 512, shape (1,n)) + feat_vp, ZERO new librosa calls, ZERO new caches. Sentinel 0.0 on any empty mask. FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate. Cited fallback from 7a170b0(c)(1). Last 22+ iterations exhausted voiced/unvoiced paired-diff across MFCC (kept), spec_contrast (kept), chroma (f4148cc), spec_flatness (0c3bf76 -> 0.539), RMS dB (c006d52 -> 0.491), ZCR (0cdd87e -> 0.508), spec_bandwidth (7a170b0 -> 0.542), every geometry (NEAR/MID/WIDE/FAR/narrow-gap/balanced-span), F0 distribution (IQR 27ddbf7, median-cents 1d1144d), voicing-mask temporal structure (8fc7169 -> 0.472), and two detector post-filters (60196aa peak-width -> 0.491). spec_rolloff is the ONE remaining 1D spectral descriptor on this template. Block 2 carries spec_rolloff_delta all-frame only; GBM has never seen rolloff voicing-split. GBM max_depth=3 cannot synthesize voicing-conditional rolloff from (rolloff_delta, voicing_prob_pre, voicing_prob_post) via independent threshold splits; paired-diff is a subtraction interaction trees cannot express. Why rolloff distinct from bandwidth/centroid/flatness: centroid=1st moment (F0-correlated); bandwidth=2nd moment (std); flatness=Wiener entropy (already tried 0c3bf76). Rolloff=85th-percentile HF cutoff directly measures mastering chain LP behavior (master limiter + EQ + codec; Opus rolls off ~20 kHz, MP3-128 ~16 kHz). Percentile statistic robust to spectral-shape outliers that drive bandwidth jitter. Mechanism on 3 singing chord-cycle FPs: within-song limiter+EQ+codec frozen, voiced rolloff at vowel HF baseline ~3-5 kHz, unvoiced rolloff at drum-bus HF baseline ~7-10 kHz; chord transition doesn't change limiter/codec/drum kit -> voiced_delta~0 AND unvoiced_delta~0 -> asymmetry~0 silent, FP not boosted. Same-singer cross-song: voiced_delta small (similar vowel HF), unvoiced_delta LARGE (new drum kit + new cymbal + possibly new codec, rolloff shift 500-2000 Hz) -> asymmetry POSITIVE, TP boosted. Different-singer cross-song: both shift; existing 1eda8e3 MFCC asymmetry fires. Speech self-gating via paired differencing (1eda8e3 mechanism): voiced rolloff on speech varies with vowel identity (/i,u/ lower than /a/); unvoiced rolloff varies with consonant type (fricatives ~8 kHz, plosives moderate); but within same recording mic+preamp+codec constant so voiced+unvoiced deltas both track phoneme context correlated -> DIFFERENCE is zero-mean noise -> GBM low per-domain SHAP on english/korean. Orthogonal: NOT 1eda8e3 (MFCC 13-dim cosine); NOT 7972a98 + every spec_contrast variant (7-dim cosine); NOT f4148cc (chroma 12-dim cosine); NOT 0c3bf76 (spec_flatness Wiener entropy); NOT c006d52 (RMS dB energy); NOT 0cdd87e (ZCR time-domain); NOT 7a170b0 (spec_bandwidth 2nd-moment std -- this is 85th-percentile); NOT 27ddbf7/1d1144d (F0 distribution); NOT block-2 spec_rolloff_delta (all-frame no mask); NOT 8fc7169 voicing_transition_rate_delta (mask structure); NOT 60196aa peak-width (detector); NOT any percussive/harmonic/tonnetz/geometry variant. FIRST percentile-based 1D spectral moment on voicing-masked paired-diff template. Pure features.py change -- 1 new block (~40 lines cloned from _block_voiced_unvoiced_spec_contrast_asymmetry with 1D scalar mean replacing 7-dim cosine) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call. Per-t cost 4 slices + 4 masked means + 2 subtractions, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_spec_rolloff_asymmetry', synthetic mixed voiced+noise audio yields finite -91.39 on noise region vs 0.0 sentinel on pure-voiced regions (expected -- synthetic pure-tone lacks unvoiced frames so mask-empty path fires), all 81 features finite, idempotent on repeated calls, edge-clamped t=1 returns sentinel via empty-mask path.
per-domain: combined_english=0.835443 combined_korean=0.529412 combined_singing=0.295385

# 2026-04-20 — hypothesis: add voiced_unvoiced_spec_rolloff_asymmetry (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — the last untried 1D spectral
    moment on the proven voiced/unvoiced paired-diff template (1eda8e3 NEAR MFCC
    kept +0.054 biggest win, 7972a98 NEAR spec_contrast kept +0.005). For
    pre=[t-2,t] and post=[t,t+2]:
        voiced_delta   = mean(rolloff[voiced post]) - mean(rolloff[voiced pre])
        unvoiced_delta = mean(rolloff[unvoiced post]) - mean(rolloff[unvoiced pre])
        feature        = unvoiced_delta - voiced_delta
    Reuses cached feat_rolloff (librosa.spectral_rolloff, hop 512, shape (1,n))
    + feat_vp (voicing probability, hop 512). ZERO new librosa calls, ZERO
    new caches. Sentinel 0.0 on any empty mask. FEATURE_NAMES 80→81 forces
    retrain via US-505 sha gate.

(b) WHY this over recent failures. EXPLICIT cited fallback from 7a170b0(c)(1):
    "voiced_unvoiced_rolloff_asymmetry (percentile-based cutoff, different
    mastering fingerprint than bandwidth)". Last 22+ iterations exhausted
    voiced/unvoiced paired-diff across MFCC (kept), spec_contrast (kept),
    chroma (f4148cc discard), spec_flatness (0c3bf76 → 0.539), RMS dB
    (c006d52 → 0.491), ZCR (0cdd87e → 0.508), spec_bandwidth (7a170b0 →
    0.542), every geometry (NEAR/MID/WIDE/FAR/narrow-gap/balanced-span), F0
    distribution (IQR 27ddbf7 → 0.524, median-cents 1d1144d → 0.508),
    voicing-mask temporal structure (voicing_transition_rate_delta 8fc7169
    → 0.472), and two detector post-filters (60196aa peak-width → 0.491).
    CLAUDE.md mandates structural change after 5+ same-axis failures;
    spec_rolloff is the ONE remaining 1D spectral descriptor on this
    template. Block 2 carries spec_rolloff_delta ALL-FRAME only — GBM has
    never seen rolloff voicing-split. GBM max_depth=3 cannot synthesize
    voicing-conditional rolloff from (rolloff_delta, voicing_prob_pre,
    voicing_prob_post) via independent threshold splits; paired-diff is a
    subtraction interaction trees cannot express.

    Why rolloff is distinct from bandwidth/centroid/flatness. Centroid is
    the 1st spectral moment (center of mass) — F0-correlated on voiced
    frames, partially duplicates f0_mean_delta. Bandwidth is the 2nd moment
    (std around centroid). Flatness is Wiener entropy (already tried
    0c3bf76). ROLLOFF is the 85th-percentile frequency CUTOFF — directly
    measures the mastering chain's high-frequency rolloff characteristic
    (low-pass behavior imposed by master limiter + EQ + ADC reconstruction
    filter). Different limiters and codecs have different HF rolloff
    profiles (Opus rolls off ~20 kHz, MP3-128 ~16 kHz, WAV ~sr/2). Rolloff
    is a percentile statistic, robust to spectral-shape outliers that drive
    bandwidth jitter.

    Mechanism on 3 surviving singing chord-cycle FPs. Within one song,
    master limiter + EQ + codec are FROZEN. Voiced rolloff sits at singer's
    vowel HF characteristic (~3-5 kHz — F4 formant and harmonics dominate
    rolloff_85); unvoiced rolloff sits at drum-bus HF characteristic
    (~7-10 kHz — cymbal / high-hat / reverb tail). Chord transition does
    NOT change limiter / codec / drum kit → voiced_delta ≈ 0 AND
    unvoiced_delta ≈ 0 → asymmetry ≈ 0, feature silent, FP not boosted.
    Same-singer cross-song: voiced_delta small (similar vowel HF across
    songs), unvoiced_delta LARGE (new drum kit + new cymbal signature +
    possibly new codec → different 85th-percentile cutoff, shifts by
    500-2000 Hz) → asymmetry POSITIVE, TP boosted. Different-singer
    cross-song: both shift; existing 1eda8e3 MFCC asymmetry fires.

    Speech self-gating via paired differencing (1eda8e3 mechanism).
    Voiced rolloff on speech varies with vowel identity (high vowels
    /i,u/ have lower rolloff than /a/ because formants concentrate
    lower); unvoiced rolloff on speech varies with consonant type
    (fricatives /s,f/ have high rolloff ~8 kHz; plosives /p,t,k/ have
    moderate rolloff). But within same recording microphone + preamp +
    codec constant → voiced and unvoiced rolloff shifts both track
    phoneme context in CORRELATED ways → DIFFERENCE is zero-mean noise
    across non-splice windows → GBM low per-domain SHAP on
    english/korean → functionally invisible on speech.

    Classifier hyperparam axis verifiably broken (5+ train_classifier.py-
    only iterations IDENTICAL 0.490700). Only features.py-sha bumps force
    retrain consistently. Every primary tunable saturated both directions.

    Orthogonal. NOT 1eda8e3 (13-dim MFCC cosine); NOT 7972a98 + every
    spec_contrast variant (d290101/177d641/88adb49/6c6c254/347c0ac, 7-dim
    cosine); NOT f4148cc (chroma 12-dim cosine); NOT 0c3bf76 (spec_flatness
    abs-delta Wiener entropy, not percentile); NOT c006d52 (log-RMS dB
    energy level, not frequency); NOT 0cdd87e (ZCR time-domain count); NOT
    7a170b0 (spec_bandwidth 2nd-moment std — this is 85th-percentile); NOT
    27ddbf7 / 1d1144d (F0 distribution); NOT block-2 spec_rolloff_delta
    (all-frame, no voicing split); NOT 8fc7169 voicing_transition_rate_delta
    (mask structure, not rolloff content); NOT 60196aa peak-width (detector);
    NOT any percussive/harmonic/tonnetz/geometry variant. FIRST percentile-
    based 1D spectral moment on voicing-masked paired-diff template.

    Blast radius: 1 new block (~35 lines cloned from
    _block_voiced_unvoiced_spec_contrast_asymmetry with 1D scalar mean
    replacing 7-dim cosine, matching c006d52/0cdd87e/7a170b0 template) +
    1 FEATURE_NAMES append + 2 assert bumps (80→81) + 1 call. ZERO new
    caches, ZERO new librosa calls. Per-t cost: 4 slices + 4 masked means
    + 2 subtractions, sub-ms.

(c) IF THIS FAILS. (1) Singing flat / speech preserved (rolloff signal
    sits in same surface as spec_centroid_delta; voicing-split produces
    partial duplicate info GBM already has through block-2 deltas) →
    fallback to voiced_unvoiced_spec_flux_asymmetry (frame-to-frame
    spectral change, genuinely different from moments; feat_flux isn't
    cached — would require librosa call but tiny). (2) Speech regresses
    (voiced rolloff tracks vowel formant strongly enough that 2s window
    doesn't average it away) → pivot to voiced_unvoiced_rolloff_LOG_ratio
    = log2(post_u_mean / pre_u_mean) - log2(post_v_mean / pre_v_mean) —
    log-scale normalisation for register/codec invariance. (3) combined
    matches 0.4907 AGAIN → confirms the identical-streak spans multiple
    features.py adds; escalate as systemic wrapper cache-coherence bug.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 48+
    iterations — cannot verify rolloff values at the 3 singing FPs to
    confirm whether unvoiced rolloff is in fact stable cross-chord-cycle
    (as theory predicts) vs already discriminating. Every 1D-scalar
    asymmetry hypothesis remains theory-calibrated. (ii) SHAP rollup STILL
    empty for 14 keeps despite 48+ iterations of explicit reflection —
    rollup writer broken; without per-feature attribution the axis pick
    is theory-only. Cannot verify whether 1eda8e3/7972a98 are even in
    the top features GBM uses. (iii) 0.490700 identical-streak crosses
    features.py AND detector.py changes; no wrapper signal distinguishes
    "feature silent" from "classifier stale" from "wrapper cache hit".
    (iv) 99081f5 4/16 capacity ghost status unclear at HEAD after 960113c
    "REVERT" that wrapper may have discarded.

(e) Wrapper enhancements.
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — persistent 48+-
    iteration blocker cited in every reflection. Per-FP (domain, file,
    t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, chunk_duration_s,
    voiced_unvoiced_mfcc_asymmetry, voiced_unvoiced_spec_contrast_asymmetry,
    spec_rolloff_delta, top-5 |SHAP|). Transforms every paired-diff
    hypothesis from theory bet to data-driven decision.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps is long-standing
    bug; emit per-iteration SHAP top-K to `.omc/classifier/shap_rollup.json`
    on every keep; aggregate rolling-5 in wrapper.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision ("features.py sha Δ XX→YY → retrain" vs "no Δ → skip") with
    joblib-mtime sanity check would isolate the 0.4907 identical-streak
    root cause. Three unchanged highest-priority requests across 48+
    iterations.

## 2026-04-20T23:13:14+09:00 — 9064eec (discard, combined=0.544129)
subject: add mfcc_cross_intra_contrast (FEATURE_NAMES 80->81) -- SECOND-ORDER MFCC consistency feature. cross_dist - 0.5*(intra_pre + intra_post) over +-4s span (clipped to [t-4,t+4]; edge guard t-4<0 or t+4>duration -> 0.0). intra_pre=cos_dist(mean_mfcc(t-4,t-2), mean_mfcc(t-2,t)); intra_post=cos_dist(mean_mfcc(t,t+2), mean_mfcc(t+2,t+4)); cross=cos_dist(mean_mfcc(t-4,t), mean_mfcc(t,t+4)). Reuses cached feat_mfcc, ZERO new librosa calls, ZERO new caches. FEATURE_NAMES 80->81 forces auto-retrain via US-505 sha gate. 25+ iterations exhausted the 1D voiced/unvoiced paired-diff template on every content axis (MFCC/spec_contrast/chroma/spec_flatness/RMS-dB/ZCR/spec_bandwidth/spec_rolloff), every geometry (NEAR/MID/WIDE/FAR/narrow-gap/balanced-span), F0 distribution (IQR, median-cents), voicing-mask temporal structure (transition-rate), detector post-filter (peak-width) -- CLAUDE.md mandates structural change after 5+ same-axis failures. Untried dimension is SECOND-ORDER: comparing cross-boundary distance to intra-side self-distances on the same span. Every prior feature computes ONE distance (or one paired subtraction of masked distances on the SAME window). This feature computes THREE distances on distinct time windows and combines them. GBM max_depth=4/max_leaf=16 cannot synthesize via threshold splits on existing block-2 MFCC deltas because those are all single-cross-boundary. Mechanism on 3 singing chord-cycle FPs: singer+mastering+drum-bus continuous within one song so MFCC drifts steadily both BEFORE and AFTER t; intra_pre ~0.10 (one chord transition each side), intra_post ~0.10, cross averaging 2-3 chords bounded ~0.10 -> feature ~0 silent, FP not boosted. Real cross-song splice: intra_pre small ~0.04 (song A self-consistent), intra_post small ~0.04 (song B self-consistent), cross LARGE ~0.30 (A->B mastering+drum+singer jump) -> feature +0.26 STRONGLY POSITIVE, TP boosted. Sign-and-magnitude separation chord cycle (~0) vs real splice (+0.25+) is binary discriminator. Speech self-gating: 4s pre samples ~6-8 syllables -> intra_pre 0.08-0.15 phoneme drift; 4s post same -> intra_post 0.08-0.15; 4s-mean vs 4s-mean cross smooths phoneme variation -> cross 0.08-0.15 tracking intra -> feature ~0 across non-splice positions on english/korean. Speech splices TP handled by existing 1eda8e3 voiced_unvoiced_mfcc_asymmetry (+0.054 biggest keep). Why subtraction (not ratio): ratio amplifies near-zero-intra noise; subtraction zero-mean-balanced; 0.5*(intra_pre+intra_post) average smoother than max-over-intras. Why MFCC axis: 1eda8e3 proved MFCC carries singer/instrument timbre continuity signal (+0.054 biggest keep); intra-side baseline captures within-song timbre drift precisely where chord-cycle FPs sit. Orthogonal: NOT 1eda8e3 / any voiced_unvoiced_mfcc geometry (all 1st-order voicing-masked cross-boundary); NOT block-2 MFCC deltas (1st-order single distance no intra baseline); NOT 7972a98/spec_contrast variants; NOT f4148cc chroma; NOT 0c3bf76 spec_flatness; NOT c006d52 RMS dB; NOT 0cdd87e ZCR; NOT 7a170b0 spec_bandwidth; NOT e234649 spec_rolloff; NOT 27ddbf7/1d1144d F0; NOT 8fc7169 voicing transition rate; NOT 60196aa peak-width detector. FIRST SECOND-ORDER consistency feature in the 80-feature set. Pure features.py change -- 1 new block (~55 lines) + FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call. Per-t cost: 6 slices + 6 means on 13-dim vectors + 3 cosines + 2 subtractions + 1 average, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'mfcc_cross_intra_contrast', synthetic A/B splice at t=10 yields 1.68 LARGE, within-source t=5 yields -8e-5 sentinel-near-zero (~20000x discrimination), edge guard t=1 (t-4<0) returns 0.0, edge guard t=17 with 20s audio (t+4>20) returns 0.0, all 81 features finite, idempotent on repeated calls.
per-domain: combined_english=0.860759 combined_korean=0.623881 combined_singing=0.300000

# 2026-04-20 — hypothesis: add mfcc_cross_intra_contrast (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot to a
    SECOND-ORDER MFCC consistency feature that distinguishes real cross-
    source splice from intra-song chord cycle by comparing the cross-
    boundary distance to the intra-side self-distances on the SAME
    ±4s span. For windows clipped to [t-4, t+4]:
        intra_pre  = cos_dist(mean_mfcc(t-4, t-2), mean_mfcc(t-2, t))
        intra_post = cos_dist(mean_mfcc(t,   t+2), mean_mfcc(t+2, t+4))
        cross      = cos_dist(mean_mfcc(t-4, t),   mean_mfcc(t,   t+4))
        feature    = cross - 0.5 * (intra_pre + intra_post)
    Reuses cached feat_mfcc — ZERO new librosa calls, ZERO new caches.
    Edge guard t-4<0 OR t+4>duration_s returns sentinel 0.0.
    FEATURE_NAMES 80→81 forces wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. 25+ iterations exhausted the 1D
    voiced/unvoiced paired-diff template on every content axis: MFCC
    cosine (1eda8e3 kept, every geometry variant discarded), spec_contrast
    (7972a98 kept, NEAR/MID/WIDE/FAR/narrow-gap/balanced-span all
    discarded), chroma (f4148cc), spec_flatness (0c3bf76 → 0.539), RMS
    dB (c006d52 → 0.491), ZCR (0cdd87e → 0.508), spec_bandwidth (7a170b0
    → 0.542), spec_rolloff (e234649 → 0.507). F0 distribution shape and
    location (27ddbf7 IQR → 0.524, 1d1144d median-cents → 0.508).
    Voicing-mask temporal structure (8fc7169 transition-rate → 0.472).
    Detector post-filter (60196aa peak-width → 0.491). CLAUDE.md
    mandates structural change after 5+ same-axis failures; the
    1st-order cross-boundary family is structurally exhausted.

    The untried dimension is SECOND-ORDER — not another content axis
    or another geometry, but a comparison between cross-boundary
    distance and the intra-side self-distances. Every prior feature
    computes ONE distance (or one paired subtraction of distances on
    different masks computed on the SAME window). This feature
    computes THREE distances on distinct time windows and combines
    them. GBM max_depth=4 / max_leaf_nodes=16 (ghost still in-tree)
    cannot synthesize this subtraction via threshold splits on
    existing block-2 MFCC deltas because those deltas are all computed
    over a SINGLE ±(2s) cross-boundary pair — no feature in the
    80-feature set carries an intra-side baseline.

    Mechanism on 3 surviving singing chord-cycle FPs. Within one song
    the singer/mastering/drum-bus is continuous; chord transitions
    drift the MFCC steadily both BEFORE and AFTER t. intra_pre captures
    one chord transition in [t-4, t] ≈ 0.08-0.12 cosine dist.
    intra_post captures another chord transition in [t, t+4] ≈ 0.08-
    0.12. cross distance over longer 4s means averages over 2-3 chords
    each side within the same song ≈ 0.08-0.15 (not amplified because
    within-song mean is bounded). Feature ≈ 0.10 − 0.10 ≈ 0 or slightly
    negative, silent → FP NOT boosted.

    Real cross-song splice. Pre-side 4s from song A is self-consistent
    (intra_pre ≈ 0.03-0.05). Post-side 4s from song B is self-
    consistent (intra_post ≈ 0.03-0.05). cross captures A → B jump
    dominated by different mastering chain + different drum kit +
    possibly different singer formants → 0.25-0.45. Feature ≈ 0.30 −
    0.04 = +0.26 STRONGLY POSITIVE, real TP boosted. Sign-and-magnitude
    separation of chord cycle (≈0) vs real splice (+0.25+) is a binary
    discriminator GBM cannot synthesize from the existing feature set
    because every existing MFCC feature mixes intra-side and cross-
    side in the same single distance.

    Speech self-gating. On english/korean, pre-side 4s samples ~6-8
    syllables with phoneme drift → intra_pre ≈ 0.08-0.15. Post-side
    4s samples next ~6-8 syllables with phoneme drift → intra_post ≈
    0.08-0.15. cross computes 4s-mean vs 4s-mean — longer averaging
    smooths phoneme variation so cross ≈ 0.08-0.15 as well (tracks
    intra at same scale). Feature ≈ 0 across non-splice positions →
    GBM low per-domain SHAP on english/korean. Speech splices (TP)
    remain handled by existing voiced_unvoiced_mfcc_asymmetry (1eda8e3
    +0.054 biggest keep) which is designed for speech.

    Why subtraction (not ratio): cross/intra ratio amplifies near-
    zero-intra noise. Subtraction is zero-mean-balanced and GBM handles
    signed scalars cleanly via threshold splits. Using 0.5*(intra_pre+
    intra_post) vs max-over-intras: average smoother, less sensitive
    to single-sided chord transitions that coincidentally land large
    on one intra window.

    Why MFCC content axis: 1eda8e3 proved MFCC carries the singer/
    instrument timbre continuity signal (+0.054 biggest keep); the
    intra-side baseline on MFCC captures within-song timbre drift
    precisely where chord-cycle FPs sit. Transplanting this second-
    order template to MFCC leverages the proven productive axis rather
    than exploring yet another content space.

    Classifier hyperparam axis verifiably broken (5+ train_classifier.py-
    only iterations IDENTICAL 0.490700). Only features.py-sha bumps
    force retrain. Every primary tunable saturated both directions.

    Orthogonal. NOT 1eda8e3 / any voiced_unvoiced_mfcc geometry variant
    (all 1st-order voicing-masked cross-boundary); NOT block-2 MFCC
    deltas (1st-order single distance, no intra baseline); NOT 7972a98
    + every spec_contrast variant; NOT f4148cc chroma; NOT 0c3bf76
    spec_flatness; NOT c006d52 RMS dB; NOT 0cdd87e ZCR; NOT 7a170b0
    spec_bandwidth; NOT e234649 spec_rolloff; NOT 27ddbf7 / 1d1144d
    F0 distribution; NOT 8fc7169 voicing transition rate; NOT 60196aa
    peak-width detector; NOT any percussive/harmonic/tonnetz mask or
    single-mask variant. FIRST SECOND-ORDER consistency feature in the
    80-feature set: first feature computing a function of MULTIPLE
    distance measurements across distinct time windows.

    Blast radius. Pure features.py change — 1 new block (~35 lines)
    + 1 FEATURE_NAMES append + 2 assert bumps (80→81) + 1 call in
    extract_features. ZERO new caches, ZERO new librosa calls. Per-t
    cost: 6 slices + 6 means on 13-dim vectors + 3 cosines + 2
    subtractions + 1 average, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged (intra-side baseline too
    noisy at 2s half-windows because 2s spans only ~1 chord → intra
    distance indistinguishable from cross when chord cycles dominate)
    → fallback to same formula on spec_contrast axis (d290101 proved
    speech-safe at 3s spans; intra-side baseline on spec_contrast
    mastering-fingerprint cancels chord-cycle drift more cleanly).
    (2) Speech regresses (cross > intra on speech because phoneme
    transitions are correlated within short stretches, so 4s vs 4s
    cross-means carry sentence-level shift beyond 2s intra-shifts) →
    normalized ratio cross / (0.1 + max(intra_pre, intra_post)) capped
    at 10, compressed signal less sensitive to speech prosody. (3)
    combined matches 0.490700 again → features.py sha-bump no longer
    forces retrain (coverage bug regressed); escalate.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 49+
    iterations — cannot verify whether the 3 singing FPs' t_sec sits
    in a chord-dense region where intra_pre/intra_post actually
    dominate cross_dist, or whether they sit near file ends where the
    ±4s edge guard fires and feature silences. (ii) SHAP rollup STILL
    empty for 14 keeps — no per-feature attribution; cannot verify
    whether 1eda8e3/7972a98 are even in top features GBM actually uses.
    (iii) 99081f5 4/16 capacity ghost STILL at HEAD (max_depth=4,
    max_leaf_nodes=16 per grep); 960113c "REVERT" commit note did not
    change the file; baseline 0.589282 was set against whichever config
    was live at retrain time. Attribution against baseline contaminated.
    (iv) No wrapper log signal confirms features.py sha bumps actually
    trigger retrain vs cache hit; 0.490700 identical-streak spanned
    multiple features.py edits.

(e) Wrapper enhancements.
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — persistent
    49+-iteration blocker. Per-FP (domain, file, t_sec, p_splice,
    dsp_phase_z, dsp_t2_z, dsp_cpe_z, chunk_duration_s,
    voiced_unvoiced_mfcc_asymmetry, voiced_unvoiced_spec_contrast_
    asymmetry, mfcc_delta_01..13 norm, top-5 |SHAP|). Would turn every
    2nd-order-consistency hypothesis into a data-driven decision.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps; without per-
    feature attribution the axis pick is theory-only. Cannot verify
    which features GBM uses or which are redundant with existing
    block-2 deltas vs genuinely orthogonal.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision ("features.py sha Δ XX→YY → retrain" vs "no Δ → skip")
    with joblib-mtime sanity check post-retrain would isolate the
    0.4907 identical-streak root cause. Three unchanged highest-
    priority requests across 49+ iterations.

## 2026-04-20T23:28:05+09:00 — 26a3687 (discard, combined=0.493112)
subject: add spec_contrast_cross_intra_contrast (FEATURE_NAMES 80->81) -- SECOND-ORDER spec_contrast consistency feature, cited fallback from 9064eec(c)(1). cross_dist - 0.5*(intra_pre + intra_post) over +-4s span clipped to [t-4,t+4]; edge guard t-4<0 or t+4>duration -> 0.0. intra_pre=cos_dist(mean_contrast(t-4,t-2), mean_contrast(t-2,t)); intra_post=cos_dist(mean_contrast(t,t+2), mean_contrast(t+2,t+4)); cross=cos_dist(mean_contrast(t-4,t), mean_contrast(t,t+4)). Transplants 9064eec MFCC second-order template onto 7-dim spec_contrast (mastering-fingerprint axis). Reuses cached feat_contrast, ZERO new librosa calls, ZERO new caches. FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate. 9064eec MFCC version produced 0.544 (closer to baseline than every 1st-order failure in 20+ iterations) but speech regressed (english 0.889->0.861, korean 0.667->0.624) because MFCC cepstral envelope is phoneme-correlated so 4s-mean cross overshoots 2s-sub-mean intra. spec_contrast is the cited content-axis swap: (i) d290101 proved spec_contrast intrinsically speech-safe (english 0.889->0.897 at 3s spans, peak-RATIO phoneme-stable); (ii) 7972a98 NEAR asymmetry kept +0.005 proving singing mastering-fingerprint signal; (iii) intra-side on within-song +-4s is dominated by FROZEN mastering chain (drum-bus comp + master EQ + limiter) whose per-band peak/valley signature does NOT shift with chord cycles -> intra_pre~intra_post~cross~0.02-0.08 -> feature~0 silent on chord-cycle FP. Cross-song splice: intra small per side (self-consistent mastering), cross LARGE (A->B mastering jump 0.20-0.45) -> feature +0.25+ STRONGLY POSITIVE. Sign-and-magnitude separation chord-cycle (~0) vs real splice (+0.25+) is binary discriminator GBM cannot synthesize from existing feature set because no existing spec_contrast feature carries intra-side baseline (block-2 all-frame scalar; 7972a98 voicing-masked single-boundary; e7ca9eb single-mask single-boundary). Speech self-gating: spec_contrast peak-RATIO phoneme-stable so intra and cross co-vary in similar 0.04-0.08 range on english/korean -> subtraction zeros CLEANER than 9064eec MFCC where cepstral envelope drift made cross overshoot intra. CLAUDE.md mandates structural change after 5+ same-axis failures; 26+ iterations exhausted 1st-order paired-diff. 2nd-order on speech-safe axis is the cited next move. Orthogonal: NOT 9064eec (MFCC 13-dim content -- this is 7-dim spec_contrast); NOT 7972a98 (voicing-masked paired-diff, no intra baseline); NOT d290101/177d641/88adb49/6c6c254/347c0ac (first-order voicing-masked spec_contrast); NOT 1eda8e3 / any voiced_unvoiced_mfcc variant; NOT block-2 spec_contrast_delta (1D all-frame); NOT e7ca9eb voiced_spec_contrast (single-mask no intra); NOT any 1D-scalar / F0 / ZCR / bandwidth / rolloff / flatness / chroma variant; NOT 60196aa detector peak-width. FIRST spec_contrast second-order consistency feature, complementary to 9064eec MFCC second-order. Pure features.py change -- 1 new block (~50 lines) + FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call. Per-t cost: 6 slices + 6 means on 7-dim vectors + 3 cosines + 2 subtractions + 1 average, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'spec_contrast_cross_intra_contrast', synthetic A(440Hz sine)/B(220Hz saw) splice at t=10 yields 0.129408 LARGE, within-A t=5 yields -0.000001 sentinel (~100000x discrimination), edge guard t=1 (t-4<0) returns 0.0, edge guard t=17 with 20s audio (t+4>20) returns 0.0, all 81 features finite, idempotent on repeated calls.
per-domain: combined_english=0.783750 combined_korean=0.517808 combined_singing=0.295455

# 2026-04-20 — hypothesis: add spec_contrast_cross_intra_contrast (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — transplant 9064eec's
    SECOND-ORDER consistency template (cross vs intra distances on
    ±4s span, 2s sub-windows) from MFCC onto spec_contrast. 7-dim
    per-band peak-to-valley cosines instead of 13-dim cepstral
    cosines. For windows clipped to [t-4, t+4]:
        intra_pre  = cos_dist(mean_contrast(t-4,t-2), mean_contrast(t-2,t))
        intra_post = cos_dist(mean_contrast(t,t+2),   mean_contrast(t+2,t+4))
        cross      = cos_dist(mean_contrast(t-4,t),   mean_contrast(t,t+4))
        feature    = cross - 0.5*(intra_pre + intra_post)
    Reuses cached feat_contrast — ZERO new librosa calls, ZERO new
    caches. Edge guard t-4<0 OR t+4>duration returns sentinel 0.0.
    FEATURE_NAMES 80→81 forces wrapper auto-retrain via US-505 sha
    gate.

(b) WHY this over recent failures. Explicit cited fallback from
    9064eec(c)(1): "fallback to same formula on spec_contrast axis
    (d290101 proved speech-safe at 3s spans; intra-side baseline on
    spec_contrast mastering-fingerprint cancels chord-cycle drift
    more cleanly)". 9064eec MFCC second-order produced combined=
    0.544 — CLOSER to baseline than every 1st-order failure in the
    last 20+ iterations — but speech regressed (english 0.889 →
    0.861, korean 0.667 → 0.624) because MFCC is phoneme-correlated
    and the 4s-mean-vs-4s-mean cross picked up cross-sentence
    phoneme drift. On singing it produced 0.300 (below 0.345
    baseline) because the 4s-sub-windows averaged ~2 chord cycles
    on BOTH intra AND cross, so the subtraction zeroed on chord-
    cycle FPs AND on real splices.

    Three empirical facts make spec_contrast the right content axis
    for second-order:
    (i) d290101 FAR-gap spec_contrast IMPROVED english 0.889 → 0.897
    at 3s spans proving the axis is intrinsically speech-safe —
    peak-RATIO is phoneme-stable where MFCC cepstral envelope is
    not.
    (ii) 7972a98 NEAR spec_contrast asymmetry (current keep +0.005)
    proves the mastering-fingerprint signal exists on singing.
    (iii) Intra-side spec_contrast on within-song ±4s is dominated
    by the FROZEN mastering chain (drum-bus comp + master EQ +
    limiter) whose per-band peak/valley signature does not shift
    with chord cycles — so intra_pre and intra_post on a within-
    song chord-cycle FP should sit NEAR ZERO (mastering frozen →
    ≈same 7-dim vector across every 2s sub-window).

    Mechanism on 3 surviving singing chord-cycle FPs. Master limiter
    + EQ + drum-bus compression are FROZEN within one song. 2s-mean
    spec_contrast on mixed-voicing frames is dominated by drum bus
    + master limiter (broadband compression signature) which are
    chord-invariant. intra_pre ≈ intra_post ≈ cross ≈ 0.02-0.08
    (per-band peak/valley signature held). feature ≈ 0.05 − 0.05 ≈
    0, silent, FP NOT boosted.

    Real cross-song splice. Pre-side [t-4, t] reflects song A
    mastering (intra_pre ≈ 0.02-0.04 self-consistent). Post-side
    [t, t+4] reflects song B mastering (intra_post ≈ 0.02-0.04
    self-consistent). cross captures A→B mastering jump dominated
    by different compressor ratio + different limiter threshold +
    different EQ curve → 0.20-0.45. feature ≈ 0.30 − 0.03 = +0.27
    STRONGLY POSITIVE, TP boosted. Sign-and-magnitude separation
    chord-cycle (≈0) vs real splice (+0.25+) is a binary
    discriminator GBM cannot synthesize from the existing feature
    set because no existing spec_contrast feature carries an intra-
    side baseline (block-2 spec_contrast_delta is 1D all-frame;
    7972a98 paired-diff is voicing-masked single-boundary; e7ca9eb
    voiced_spec_contrast is single-mask single-boundary).

    Speech self-gating. spec_contrast peak-RATIO is phoneme-stable
    (d290101 mechanism). On english/korean, 4s pre samples ~6-8
    syllables, 4s post samples next ~6-8 syllables. intra_pre ≈
    0.04-0.08 (within-recording peak-ratio drift across phoneme
    mix). intra_post ≈ 0.04-0.08. cross ≈ 0.04-0.08 (longer 4s-
    mean averaging smooths phoneme variation to same magnitude as
    intra). feature ≈ 0 across non-splice positions → GBM low per-
    domain SHAP on english/korean. Critically, because spec_
    contrast intra-values on speech SIT AT SIMILAR MAGNITUDE as
    cross (both driven by phoneme-ratio drift, not by cepstral
    formant shape), the subtraction zeros on speech much more
    cleanly than 9064eec MFCC where cross tended to over-shoot
    intra due to cepstral envelope drift.

    Why ±4s span retained. Same template as 9064eec for clean A/B
    on the content-axis swap. ±3s would produce 1.5s sub-windows
    too narrow for stable 7-dim vector means. ±6s would break
    speech self-gating (spans two sentences).

    Orthogonal. NOT 9064eec (MFCC 13-dim content, this is 7-dim
    spec_contrast mastering-fingerprint); NOT 7972a98 (voicing-
    masked paired-diff single boundary, no intra baseline); NOT
    d290101/177d641/88adb49/6c6c254/347c0ac (all first-order
    voicing-masked on spec_contrast); NOT 1eda8e3 / any voiced_
    unvoiced_mfcc geometry; NOT block-2 spec_contrast_delta (1D
    all-frame); NOT e7ca9eb voiced_spec_contrast (single-mask
    single-boundary, no intra baseline); NOT any 1D-scalar / F0 /
    ZCR / bandwidth / rolloff / flatness / chroma variant; NOT
    60196aa detector peak-width. FIRST spec_contrast second-order
    consistency feature — complementary to 9064eec MFCC second-
    order on the proven speech-safe axis.

    Blast radius. Pure features.py change — 1 new block function
    (~45 lines cloned from 9064eec template but on 7-dim contrast
    vector with no voicing mask) + 1 FEATURE_NAMES append + 2
    assert bumps (80 → 81) + 1 call in extract_features. ZERO new
    caches, ZERO new librosa calls. Per-t cost: 6 slices + 6 means
    on 7-dim vectors + 3 cosines + 2 subtractions + 1 average,
    sub-ms.

(c) IF THIS FAILS. (1) Speech regresses (cross > intra on speech
    because 4s-sub-window phoneme mix differs enough from 2s-sub-
    window mix to break self-gating on spec_contrast too) →
    NORMALIZED fallback cross / (0.1 + max(intra_pre, intra_post))
    capped at 10. (2) Singing unchanged (chord-cycle FPs actually
    drive intra_pre/post HIGH because chord transitions shift per-
    band peak/valley on the voiced side — mastering-frozen
    hypothesis wrong) → narrow window to ±3s / 1.5s sub-windows
    (d290101 proved 3s span speech-safe). (3) combined matches
    0.589 exactly OR lands at 0.4907 identical-streak → wrapper
    retrain gate broken across features.py sha bump; escalate as
    systemic cache-coherence bug.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after
    49+ iterations — cannot verify whether the 3 singing FPs sit
    in regions where intra-side spec_contrast is in fact flat
    (mastering-frozen hypothesis) or whether they carry intra-
    drift already. Every 2nd-order hypothesis remains theory-
    calibrated. (ii) SHAP rollup STILL empty for 14 keeps. (iii)
    99081f5 4/16 capacity ghost STILL at HEAD (grep
    train_classifier.py: max_depth=4, max_leaf_nodes=16); baseline
    0.589282 set against whichever config was live at retrain.
    (iv) 0.4907 identical-streak root cause still unknown.

(e) Wrapper enhancements. Three unchanged highest-priority
    requests across 49+ iterations:
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — per-FP
    (domain, file, t_sec, p_splice, dsp_phase_z, dsp_t2_z,
    dsp_cpe_z, chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, spec_contrast_delta,
    top-5 |SHAP|). Would turn every 2nd-order-consistency
    hypothesis into a data-driven decision.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps; without
    per-feature attribution I pick "theoretically orthogonal" not
    "what GBM actually uses." Emit per-iteration SHAP top-K to
    .omc/classifier/shap_rollup.json on every keep; aggregate
    rolling-5 in wrapper.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at
    retrain decision ("features.py sha Δ XX→YY → retrain" vs
    "no Δ → skip") with joblib-mtime sanity check post-retrain.
    Would isolate the 0.4907 identical-streak root cause.

## 2026-04-20T23:47:24+09:00 — c5040d7 (discard, combined=0.512673)
subject: add voiced_mfcc_cross_intra_contrast (FEATURE_NAMES 80->81) -- voicing-masked SECOND-ORDER MFCC consistency feature fusing two proven mechanisms: voicing-mask (1eda8e3 asym kept +0.054 biggest win, 49bd0b1 voiced_mfcc kept +0.028) + 2nd-order cross-intra template (9064eec all-frame MFCC 0.544 closest-to-baseline in 20+ recent failures, but speech regressed english 0.889->0.861 due to unvoiced consonant/silence MFCC heterogeneity pushing cross > intra on speech). pre/post +-4s clipped to [t-4,t+4]: intra_pre=cos_dist(voiced_mean_mfcc(t-4,t-2), voiced_mean_mfcc(t-2,t)); intra_post=cos_dist(voiced_mean_mfcc(t,t+2), voiced_mean_mfcc(t+2,t+4)); cross=cos_dist(voiced_mean_mfcc(t-4,t), voiced_mean_mfcc(t,t+4)); feature=cross-0.5*(intra_pre+intra_post). Voiced mask uses feat_vp>0 on feat_mfcc, ZERO new librosa calls, ZERO new caches. Edge guard t-4<0 OR t+4>duration_s OR any mask empty OR any norm underflow -> sentinel 0.0. FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate. CLAUDE.md mandates structural change after 5+ same-axis failures; 26+ iterations exhausted 1st-order voiced/unvoiced paired-diff across every content axis (MFCC/spec_contrast/chroma/flatness/RMS-dB/ZCR/bandwidth/rolloff) + every geometry (NEAR/MID/WIDE/FAR/narrow-gap/balanced-span) + F0 distribution (IQR/median-cents) + voicing-mask temporal structure + detector peak-width; the untried move inside the 2nd-order template is NOT another content axis or geometry but COMBINING 2nd-order with the voicing mask that has always been the productive speech-self-gating mechanism. Mechanism on 3 singing chord-cycle FPs: voiced MFCC drifts smoothly with lyric/vowel progression on both sides of t within one song; intra_pre captures ~1 chord boundary on voiced vowels ~0.04-0.08; intra_post similar; cross 4s-vs-4s voiced-mean still within-song same singer ~0.04-0.08; feature ~0 silent, FP not boosted. Real cross-song splice: pre 4s song A voiced-consistent (intra_pre ~0.03-0.05), post 4s song B voiced-consistent (intra_post ~0.03-0.05), cross captures vocal-tract/mastering shift on voiced frames ~0.10-0.30; feature +0.16 POSITIVE. Speech self-gating: voiced MFCC on speech is vowel-identity-driven; 2s samples ~4-6 vowels giving stable per-speaker centroid; 4s samples ~8-12 vowels giving SAME centroid; intra~intra~cross on speech -> feature ~0 across non-splice positions -> GBM low per-domain SHAP on english/korean -> feature functionally invisible on speech. Same mechanism that made 49bd0b1 voiced_mfcc speech-safe. Orthogonal: NOT 9064eec mfcc_cross_intra_contrast (all-frame no mask); NOT 26a3687 spec_contrast_cross_intra_contrast (different content axis); NOT 1eda8e3 / any voiced_unvoiced_mfcc geometry variant (1st-order paired-diff no intra baseline); NOT 49bd0b1 voiced_mfcc (1st-order single distance no intra); NOT any 1D-scalar asymmetry (RMS/ZCR/bandwidth/rolloff/flatness); NOT F0 distribution; NOT voicing_transition_rate_delta; NOT detector post-filter. FIRST voicing-masked second-order feature in the 80-feature set. Pure features.py change -- 1 new block (~65 lines fusing _block_voiced_mfcc mask logic with 9064eec intra/cross structure) + FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. Per-t cost 6 slices + 6 voiced-masked means on 13-dim vectors + 3 cosines + 2 subtractions + 1 average, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_mfcc_cross_intra_contrast', synthetic A(440Hz sine)/B(220Hz saw) splice at t=10 yields +0.372 LARGE, within-source t=5 yields -0.0002 sentinel-small (~1900x discrimination), edge guard t=1 (t-4<0) returns 0.0, edge guard t=17 with 20s audio (t+4>20) returns 0.0, all 81 features finite, idempotent on repeated calls.
per-domain: combined_english=0.774074 combined_korean=0.555385 combined_singing=0.313433

# 2026-04-20 — hypothesis: add voiced_mfcc_cross_intra_contrast (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot combining
    two proven mechanisms into one: the voicing-mask (1eda8e3 voiced_unvoiced
    asymmetry kept +0.054 biggest win; 49bd0b1 voiced_mfcc kept +0.028) plus
    the second-order cross-intra template (9064eec all-frame MFCC version
    was the CLOSEST-to-baseline recent hypothesis at 0.544 — every other
    1st-order failure sat at 0.49-0.52). For pre/post ±4s clipped to
    [t-4, t+4]:
        intra_pre  = cos_dist(voiced_mean_mfcc(t-4,t-2), voiced_mean_mfcc(t-2,t))
        intra_post = cos_dist(voiced_mean_mfcc(t,t+2),   voiced_mean_mfcc(t+2,t+4))
        cross      = cos_dist(voiced_mean_mfcc(t-4,t),   voiced_mean_mfcc(t,t+4))
        feature    = cross - 0.5 * (intra_pre + intra_post)
    Voiced mean uses vp>0 mask on existing feat_mfcc+feat_vp — ZERO new
    librosa calls, ZERO new caches. Edge guard t-4<0 OR t+4>duration_s OR
    any masked sub-window empty OR any norm underflow → sentinel 0.0.
    FEATURE_NAMES 80→81 forces wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. 9064eec (all-frame MFCC 2nd-order) hit
    combined=0.544 — the best recent near-baseline result in the 20+-
    iteration valley — but speech regressed (english 0.889→0.861, korean
    0.667→0.624) because unvoiced MFCC frames (consonants /s,f,p,t/ +
    silence) have wildly varying cepstra, so on speech the 4s-mean cross
    overshot 2s-sub-mean intra. Its cited fallback (c)(1) was normalized
    ratio and (c)(2) narrower windows. 26a3687 content-axis swap to
    spec_contrast didn't close the gap (0.493, singing 0.295, english
    0.784 collapsed harder than MFCC did). CLAUDE.md mandates structural
    change after 5+ same-axis failures; the untried move inside the
    2nd-order template is NOT another content axis or geometry but
    **combining the 2nd-order template with the voicing mask** that
    has always been the productive speech-self-gating mechanism on this
    project (1eda8e3 +0.054, 49bd0b1 +0.028).

    Mechanism on 3 surviving singing chord-cycle FPs. Singer's vocal
    tract + bus compression are continuous within one song; voiced MFCC
    drifts smoothly across chord transitions (vowels shift formants with
    lyric progression but not with chord). intra_pre captures ~1 chord
    boundary on voiced vowels → 0.04-0.08. intra_post similar. cross
    averages 4s of voiced vowels pre vs 4s of voiced vowels post — still
    within-song, same singer → 0.04-0.08. feature ≈ 0.06 − 0.06 ≈ 0,
    silent, FP NOT boosted.

    Real cross-song splice (different-singer OR same-singer different
    recording). Pre-side 4s of song A voiced vowels is self-consistent
    (intra_pre ≈ 0.03-0.05). Post-side 4s of song B voiced vowels is
    self-consistent (intra_post ≈ 0.03-0.05). cross captures vocal-tract
    + mastering shift on voiced frames between A and B → 0.10-0.30.
    feature ≈ 0.20 − 0.04 = +0.16 POSITIVE. Smaller than 9064eec all-
    frame on cross-song (which captured drum/mastering shift too) but
    cleaner separation from chord-cycle FP because intra is now
    genuinely small within-song on voiced frames.

    Speech self-gating via voicing mask. The 9064eec failure mode —
    cross overshoots intra on english/korean due to unvoiced-frame
    heterogeneity — is neutralized by masking to voiced. Voiced MFCC on
    speech is driven by vowel identity; 2s voiced-mean samples ~4-6
    vowels giving a stable per-speaker centroid; 4s voiced-mean samples
    ~8-12 vowels giving the SAME centroid (same speaker). So intra_pre
    ≈ intra_post ≈ cross on speech → feature ≈ 0 across non-splice
    positions → GBM low per-domain SHAP on english/korean → feature
    functionally invisible on speech domains. Same mechanism that made
    49bd0b1 voiced_mfcc speech-safe.

    Orthogonal. NOT 9064eec mfcc_cross_intra_contrast (all-frame, no
    voicing mask — this is voicing-masked); NOT 26a3687
    spec_contrast_cross_intra_contrast (different content axis); NOT
    1eda8e3 voiced_unvoiced_mfcc_asymmetry or any geometry variant
    (1st-order paired-diff, no intra baseline); NOT 49bd0b1 voiced_mfcc
    (1st-order single distance, no intra baseline); NOT any 1D-scalar
    asymmetry (RMS/ZCR/bandwidth/rolloff/flatness); NOT F0 distribution;
    NOT voicing_transition_rate_delta; NOT detector post-filter. FIRST
    voicing-masked second-order feature in the 80-feature set.

    Blast radius: 1 new block (~55 lines, fusion of _block_voiced_mfcc
    mask logic with 9064eec-style intra/cross structure) + 1
    FEATURE_NAMES append + 2 assert bumps (80→81) + 1 call in
    extract_features. ZERO new caches, ZERO new librosa calls. Per-t
    cost: 6 slices + 6 voiced-masked means on 13-dim vectors + 3
    cosines + 2 subtractions + 1 average, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged / speech preserved (voiced
    MFCC 2s vs 4s drift matches cross on chord cycles — intra-baseline
    cancels the chord-cycle signal too aggressively because voiced
    vowel content shifts with lyrics even within one song) → fallback
    to voiced_chroma_cross_intra_contrast (chroma is key-signature,
    stable within song's key even across lyric phrases). (2) Speech
    regresses (voiced vowel identity varies enough across 4s spans
    that cross > intra on speech too) → narrow template to ±3s / 1.5s
    sub-windows. (3) combined matches 0.589282 exactly → feature
    fires but GBM assigns ~zero SHAP (redundant with existing
    voiced_mfcc_cosine_dist); pivot to per-side asymmetry of 2nd-order
    signals.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 50+
    iterations — cannot verify whether the 3 singing FPs sit in
    regions where voiced-mean MFCC intra drift is actually small
    (mechanism assumption) or whether lyric-driven vowel turnover
    already dominates intra. (ii) SHAP rollup STILL empty for 14
    keeps — cannot verify whether 9064eec's 0.544 hit truly came
    from 2nd-order signal or from some unrelated splits. (iii) 0.4907
    identical-streak root cause still unknown. (iv) 99081f5 4/16
    capacity ghost still at HEAD per prior reflections.

(e) Wrapper enhancements. Three unchanged highest-priority requests
    across 50+ iterations:
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — per-FP
    (domain, file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, voiced_mfcc_cosine_dist,
    voicing_prob_pre, voicing_prob_post, top-5 |SHAP|). Would turn
    every voicing-masked hypothesis into a data-driven decision.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps; without per-
    feature attribution I pick "theoretically orthogonal" not "what
    GBM actually uses."
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision ("features.py sha Δ XX→YY → retrain" vs "no Δ → skip")
    with joblib-mtime sanity check post-retrain. Would isolate the
    0.4907 identical-streak root cause.

## 2026-04-21T00:02:45+09:00 — 8170784 (discard, combined=0.494106)
subject: add voiced_chroma_cross_intra_contrast (FEATURE_NAMES 80->81) -- voicing-masked SECOND-ORDER consistency feature on CHROMA (pitch-class / key-signature) axis. Explicit cited fallback from c5040d7(c)(1). For +-4s span clipped to [t-4,t+4]: intra_pre=cos_dist(voiced_mean_chroma(t-4,t-2), voiced_mean_chroma(t-2,t)); intra_post=cos_dist(voiced_mean_chroma(t,t+2), voiced_mean_chroma(t+2,t+4)); cross=cos_dist(voiced_mean_chroma(t-4,t), voiced_mean_chroma(t,t+4)); feature=cross-0.5*(intra_pre+intra_post). Voiced mask uses feat_vp>0 on feat_chroma; ZERO new librosa calls, ZERO new caches. Edge guard t-4<0 OR t+4>duration_s OR any mask empty OR any norm underflow -> sentinel 0.0. FEATURE_NAMES 80->81 forces wrapper auto-retrain via US-505 sha gate. 26+ iterations exhausted 1st-order voiced/unvoiced paired-diff across every content axis; three 2nd-order variants tried: 9064eec (all-frame MFCC 0.544 closest-to-baseline but speech regressed), 26a3687 (all-frame spec_contrast 0.493), c5040d7 (voicing-masked MFCC 0.513 but english 0.774 regressed via cepstral drift). Untried axis is CHROMA 2nd-order with voicing mask -- fundamentally different signal: chroma is 12-dim pitch-class probability, within one song's key every 2s voiced-mean chroma slice converges to SAME key-signature vector because every chord shares 3-5 of 12 pitch classes -> intra_pre~intra_post~cross~0.02-0.08 tiny -> feature~0 silent on chord-cycle FP, NOT boosted. Cross-song splice crosses keys: intra small per side (song A/B key-consistent), cross LARGE (A-key vs B-key 0.25-0.60) -> feature +0.2 to +0.5 STRONGLY POSITIVE. Sign-and-magnitude separation chord-cycle (~0) vs real splice (+0.25+) is binary discriminator GBM cannot synthesize from existing chroma features (32cac36 voiced_chroma_cosine_dist is 1st-order no intra; f4148cc asymmetry 1st-order no intra). Speech self-gating: voiced-chroma on speech is per-vowel prosodic pitch-class noise that averages to near-uniform over any window >=1s -> voiced-chroma intra+cross both sit at ~0.08-0.15 on english/korean -> feature~0 across non-splice -> GBM low per-domain SHAP; same mechanism that made 32cac36 voiced_chroma a keep. Orthogonal: NOT 9064eec (all-frame MFCC no mask); NOT 26a3687 (all-frame spec_contrast no mask); NOT c5040d7 (voicing-masked MFCC 13-dim cepstral -- this is 12-dim pitch-class); NOT f4148cc voiced_unvoiced_chroma_asymmetry (1st-order no intra); NOT 32cac36 voiced_chroma (1st-order no intra); NOT block-5 voicing features; NOT any 1D scalar / F0 / ZCR / bandwidth / rolloff / flatness / tonnetz variant; NOT 60196aa detector peak-width. FIRST voicing-masked 2nd-order on chroma axis, FIRST key-signature-axis 2nd-order feature in 80-feature set. Pure features.py change -- 1 new block (~55 lines fusing _block_voiced_chroma mask logic with 9064eec intra/cross structure on 12-dim chroma) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call. Per-t cost 6 slices + 6 voiced-masked means on 12-dim vectors + 3 cosines + 2 subtractions + 1 average, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_chroma_cross_intra_contrast', synthetic A(220/277/330 Hz)/B(175/208/262 Hz) key-shift splice at t=10 yields +0.421 LARGE, within-A t=5 yields -6e-8 sentinel-near-zero (~7M x discrimination), edge guard t=2 (t-4<0) returns 0.0, edge guard t=18 with 20s audio (t+4>20) returns 0.0, all 81 features finite, idempotent on repeated calls.
per-domain: combined_english=0.850000 combined_korean=0.485714 combined_singing=0.292187

# 2026-04-20 — hypothesis: add voiced_chroma_cross_intra_contrast (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — voicing-masked 2nd-order
    consistency feature on CHROMA (pitch-class / key-signature content
    axis). Explicit cited fallback from c5040d7(c)(1). For ±4s span
    clipped to [t-4, t+4]:
        intra_pre  = cos_dist(voiced_mean_chroma(t-4,t-2), voiced_mean_chroma(t-2,t))
        intra_post = cos_dist(voiced_mean_chroma(t,t+2),   voiced_mean_chroma(t+2,t+4))
        cross      = cos_dist(voiced_mean_chroma(t-4,t),   voiced_mean_chroma(t,t+4))
        feature    = cross − 0.5 * (intra_pre + intra_post)
    Reuses cached feat_chroma + feat_vp — ZERO new librosa calls, ZERO
    new caches. Edge guard t-4<0 OR t+4>duration_s OR any masked sub-
    window empty OR any norm underflow → sentinel 0.0. FEATURE_NAMES
    80→81 forces wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. 26+ iterations exhausted 1st-order
    voiced/unvoiced paired-diff across every content axis. Three
    2nd-order variants tried: 9064eec (all-frame MFCC → 0.544 closest-
    to-baseline but speech regressed), 26a3687 (all-frame spec_contrast
    → 0.493), c5040d7 (voicing-masked MFCC → 0.513, singing 0.313,
    english 0.774). Untried axis is CHROMA for 2nd-order with voicing
    mask — fundamentally different signal from MFCC/spec_contrast:
    chroma = 12-dim pitch-class probability. Within one song's key the
    voiced-mean chroma at any ≥2s slice converges to the SAME key-
    signature vector because every chord shares 3-5 of 12 pitch classes
    → intra_pre, intra_post, cross all tiny → feature ≈ 0 silent →
    chord-cycle FP NOT boosted. Real cross-song splice: different keys
    → intra small per side (self-consistent key), cross LARGE (A-key
    vs B-key cosine distance 0.25-0.60) → feature +0.2 to +0.5
    STRONGLY POSITIVE. Sign-and-magnitude separation chord-cycle (≈0)
    vs real splice (+0.25+) is a binary discriminator GBM cannot
    synthesize from existing chroma features (32cac36
    voiced_chroma_cosine_dist is 1st-order no intra; f4148cc asymmetry
    is 1st-order masked no intra).

    Why chroma 2nd-order succeeds where MFCC 2nd-order (c5040d7)
    regressed speech. Voiced MFCC carries vowel identity so 4s-mean
    samples different centroid than 2s-mean; cross overshoots intra on
    speech due to cepstral drift. Voiced chroma on speech is per-vowel
    prosodic pitch-class noise that averages to near-uniform over any
    window ≥ 1s (vowels span multiple pitches in natural prosody). So
    voiced-chroma intra and cross both sit near-uniform on english/
    korean → feature ≈ 0 across non-splice positions → GBM low per-
    domain SHAP → feature functionally invisible on speech. Same self-
    gating mechanism that made 32cac36 voiced_chroma a KEEP despite
    chroma's notorious speech noise.

    Orthogonal. NOT 9064eec (all-frame MFCC, no mask); NOT 26a3687
    (all-frame spec_contrast, no mask); NOT c5040d7 (voicing-masked
    MFCC cepstral, 13-dim); NOT f4148cc voiced_unvoiced_chroma_
    asymmetry (1st-order, no intra); NOT 32cac36 voiced_chroma (1st-
    order, no intra); NOT any voiced_unvoiced_mfcc / spec_contrast /
    F0 / ZCR / bandwidth / rolloff variant; NOT 60196aa peak-width.
    FIRST voicing-masked 2nd-order feature on chroma axis; FIRST
    key-signature-axis 2nd-order feature in the 80-feature set.

    Blast radius. Pure features.py change — 1 new block (~50 lines,
    _block_voiced_chroma mask + 9064eec-style intra/cross on 12-dim
    chroma) + 1 FEATURE_NAMES append + 2 assert bumps (80→81) + 1
    call. ZERO new caches, ZERO new librosa calls. Per-t cost: 6
    slices + 6 voiced-masked means on 12-dim + 3 cosines + 2 subs
    + 1 avg, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged (2s voiced-chroma sub-windows
    don't converge to key-signature, intra dominates) → wider ±6s /
    3s sub-windows. (2) Speech regresses (voiced-chroma cross>intra
    on 4s vs 2s scales due to prosody-driven pitch-class drift) →
    normalize by per-window chroma entropy. (3) combined matches
    0.589282 exactly → feature redundant with 32cac36 per GBM SHAP;
    pivot to 1st-order WIDE chroma pre[t-4,t] post[t,t+4] single
    cosine (no intra baseline).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 50+
    iterations — cannot verify voiced-chroma drift at the 3 singing
    FPs. (ii) SHAP rollup STILL empty for 14 keeps — no per-feature
    attribution. (iii) 0.4907 identical-streak root cause unknown.
    (iv) 99081f5 capacity-ghost status unclear at HEAD.

(e) Wrapper enhancements (3 unchanged highest-priority requests
    across 50+ iterations):
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE per-FP
    (domain, file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry,
    voiced_chroma_cosine_dist, voiced_mfcc_cosine_dist, top-5 |SHAP|).
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log at retrain decision
    ("features.py sha Δ XX→YY → retrain" vs "no Δ → skip") with
    joblib-mtime sanity check.

## 2026-04-21T00:10:35+09:00 — d4d35b1 (discard, combined=0.286699)
subject: p_splice local-background-margin filter (drop wide plateaus in detector)
per-domain: combined_english=0.301887 combined_korean=0.321429 combined_singing=0.242857

# 2026-04-21 — hypothesis: p_splice local-background-margin filter (detector.py)

(a) HYPOTHESIS. Pure `splice/detector.py` add — DETECTOR-side structural
    filter that drops candidates whose p_splice is not a LOCAL ANOMALY
    against its ±3s neighborhood. After GBM hit_mask + DSP confirm, for
    each surviving candidate at grid index i compute
        local_median = median(p_splice[i-K : i+K+1])   # K = 3s / stride = 25 frames
        if p_splice[i] - local_median < 0.15: drop
    Dense p_splice is already computed per chunk; add one
    `scipy.ndimage.median_filter(p_splice, size=2*K+1, mode='nearest')`
    call per chunk plus one comparison per hit. ZERO new features, ZERO
    retrain (classifier sha stable, features.py sha stable). Primary
    tunable — instant.

(b) WHY this over recent failures. 55+ iterations exhausted:
    voiced/unvoiced 1st-order paired-diff on every content axis
    (MFCC / spec_contrast / chroma / spec_flatness / RMS-dB / ZCR /
    spec_bandwidth / spec_rolloff), every geometry (NEAR/MID/WIDE/FAR/
    narrow-gap/balanced-span), F0 distribution shape+location, voicing-
    mask temporal structure, and 2nd-order cross-intra consistency on
    4 content axes (MFCC / spec_contrast / voiced-MFCC / voiced-chroma).
    Prior detector-side attempt 60196aa peak-width neighbor-support
    (require adjacent grid points to ALSO exceed 0.95*threshold → enforces
    WIDTH ≥2 grid points) → 0.491 discarded. My proposal is the
    structural INVERSE: require neighbors to be LOWER (enforce SPIKE /
    ANOMALY, not width).

    Mechanism on 3 surviving singing chord-cycle FPs. Within one song,
    chord transitions happen every 2-3s and each produces modest spectral
    shifts that GBM scores high on spec_*_delta + f0_mean_delta. The dense
    p_splice curve sits on an ELEVATED FLOOR across a 10-15s chord-cycle
    section (~0.92-0.98 baseline) with individual chord boundaries producing
    plateaus at ~0.98-0.99. The surviving FP is the MAX within its 3.5s
    dedupe window — but the ENTIRE 6s neighborhood also sits at 0.95+.
    local_median ≈ 0.96, margin = p_splice[i] − 0.96 ≈ 0.02-0.03, which
    is FAR below 0.15 → DROPPED. Real cross-source splice: genuinely
    anomalous single grid point; surrounding p_splice on either side sits
    in the within-song regime (0.1-0.5 typical). local_median ≈ 0.3-0.5,
    margin ≈ 0.5-0.7 ≫ 0.15 → KEPT.

    Why this SUCCEEDS where 60196aa FAILED. Peak-width required adjacent
    support ≥0.95*threshold → KEEPS wide plateaus (the chord-cycle failure
    mode) AND DROPS narrow TPs. Wrong sign for the 3 singing FPs — they
    ARE wide plateaus so peak-width kept them. Background-subtraction
    KEEPS narrow spikes (TPs) and DROPS wide plateaus (chord-cycle FPs).
    Sign flipped; addresses the actual failure mode. Speech self-gating
    is automatic: speech p_splice curves have isolated phoneme-burst
    spikes surrounded by low (0.1-0.4) background — margin huge — so
    speech TPs keep firing.

    Why 0.15 margin. Chord-cycle plateau gaps between candidates are
    ~0.02-0.05. Real-splice margins should be 0.3-0.7. 0.15 sits in
    the gap — aggressive enough to bite all 3 FPs but leaves real
    splice margins untouched.

    Why ±3s window. MIN_SEP_S=3.5, so local window must be wider than
    dedupe to capture elevated plateau. 3s gives a 6s total window
    covering 2-3 chord transitions. Narrower = too few samples; wider
    = cross-chunk-boundary issues since chunks are 60s with 30s overlap.

    Orthogonal. NOT 60196aa peak-width (same detector axis but OPPOSITE
    sign: they required neighbors HIGH, I require neighbors LOWER); NOT
    any feature-space hypothesis; NOT DSP magnitude gates; NOT MIN_SEP_S
    dedupe. FIRST local-background-subtraction filter in detector history.
    Zero retrain, zero feature change.

    Blast radius. detector.py only — 2 new constants + 1 import addition
    (median_filter) + 1 function call per chunk + 1 comparison per hit
    + 1 diag kv. Per-chunk cost: median_filter size=51 on ~500-element
    array ≈ 25k ops, sub-ms. No retrain — classifier sha stable.

(c) IF THIS FAILS. (1) All singing FPs dropped but english/korean
    TPs also dropped → domain-gate filter behind singing heuristic.
    (2) Singing unchanged (p_splice floor is 0.3-0.5 not 0.95 as
    theorized) → widen window to ±5s or lower margin to 0.05.
    (3) combined matches 0.589282 exactly → filter never fired because
    dense p_splice is near-binary → pivot to RANK-based filter (only
    emit top-K p_splice per chunk).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 55+
    iterations — cannot inspect actual p_splice distribution around
    the 3 singing FPs to verify elevated-floor hypothesis. Every detector-
    side hypothesis theory-calibrated. (ii) SHAP rollup STILL empty for
    14 keeps. (iii) Wrapper does not log dense p_splice histograms or
    local-median stats — cannot distinguish "filter never fired" from
    "filter fired but didn't change any emit". (iv) GBM output calibration
    unknown — if p_splice clusters at 0.99+, margin-based filter is
    meaningless. (v) 99081f5 4/16 capacity ghost status unclear.

(e) Wrapper enhancements. Three unchanged highest-priority requests
    across 55+ iterations:
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE with dense-
    p_splice fields — per-FP (domain, file, t_sec, p_splice, p_splice_
    local_median_3s, p_splice_max_in_10s_window, dsp_phase_z, dsp_t2_z,
    dsp_cpe_z, chunk_duration_s, top-5 |SHAP|). Would turn every
    detector-side p_splice-distribution hypothesis into data-driven
    decision.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision with joblib-mtime sanity check.

## 2026-04-21T00:28:07+09:00 — 3bcec76 (discard, combined=0.495614)
subject: add unvoiced_spec_contrast_cross_intra_contrast (FEATURE_NAMES 80->81)
per-domain: combined_english=0.888889 combined_korean=0.547826 combined_singing=0.250000

# 2026-04-21 — hypothesis: add unvoiced_spec_contrast_cross_intra_contrast (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — UNVOICED-mask 2nd-order
    cross-intra consistency on spec_contrast. Every prior 2nd-order
    variant used all-frame (9064eec MFCC → 0.544, 26a3687 spec_contrast
    → 0.493) or VOICED mask (c5040d7 MFCC → 0.513, 8170784 chroma →
    0.494). UNVOICED mask on 2nd-order is the genuinely untried
    combination. For ±4s span clipped to [t-4, t+4]:
        intra_pre  = cos_dist(unvoiced_mean_contrast(t-4,t-2), unvoiced_mean_contrast(t-2,t))
        intra_post = cos_dist(unvoiced_mean_contrast(t,t+2),   unvoiced_mean_contrast(t+2,t+4))
        cross      = cos_dist(unvoiced_mean_contrast(t-4,t),   unvoiced_mean_contrast(t,t+4))
        feature    = cross − 0.5 * (intra_pre + intra_post)
    Unvoiced mask = vp<0.5 on cached feat_contrast + feat_vp. ZERO new
    librosa calls, ZERO new caches. Edge guard (t-4<0 or t+4>duration
    or any sub-window unvoiced-mask empty or any norm underflow) → 0.0.
    FEATURE_NAMES 80→81 forces retrain via US-505 sha gate.

(b) WHY this over recent failures. d4d35b1 (detector p_splice margin
    filter) regressed all three domains (0.287) — catastrophic signal
    the detector-side axis is dangerous without FP position data.
    8170784 voiced_chroma 2nd-order landed 0.494 explicitly citing
    fallback to wider windows or ratio form, NOT mask inversion. All
    four prior 2nd-order attempts used voiced or all-frame masks; the
    unvoiced-mask companion remains untried across every content axis.
    spec_contrast is the correct content choice: (i) d290101 proved
    spec_contrast peak-RATIO is phoneme-stable (english improved
    0.889→0.897 at 3s gaps) so unvoiced spec_contrast on speech does
    not carry the cepstral envelope drift that broke c5040d7 voiced
    MFCC. (ii) 7972a98 spec_contrast asymmetry remains a keep — axis
    is singing-productive. (iii) 26a3687 all-frame spec_contrast
    2nd-order regressed because voiced+unvoiced mixing on the cross
    window overshot the voiced-dominated intra sub-windows; masking
    to unvoiced-only equalizes both.

    Mechanism on 3 surviving singing chord-cycle FPs. Master bus
    compressor + limiter + drum-bus EQ + reverb tail are FROZEN within
    one song. Unvoiced frames are dominated by drums / cymbals / decay
    tail / ambient noise — all mastering-shaped, chord-invariant. 2s
    unvoiced-mean spec_contrast sits at stable per-band peak/valley
    baseline across every 2s slice in the chord cycle. intra_pre ≈
    intra_post ≈ cross ≈ 0.02-0.06. feature ≈ 0.04 − 0.04 ≈ 0 silent,
    FP NOT boosted. Real cross-song splice: pre 4s unvoiced reflects
    song A mastering (intra_pre ≈ 0.02-0.04 self-consistent), post 4s
    unvoiced reflects song B (intra_post ≈ 0.02-0.04 self-consistent),
    cross captures A→B mastering jump (different drum kit + different
    limiter + different master EQ → per-band peak/valley shifts
    0.20-0.45). feature ≈ 0.30 − 0.03 = +0.27 STRONGLY POSITIVE.

    Speech self-gating. Unvoiced in speech = fricatives + plosives +
    silence. spec_contrast peak-RATIO is phoneme-stable per d290101
    so per-phoneme variation is small compared to register/recording
    continuity. 2s unvoiced-mean across ~3-5 consonants + silence
    averages to a within-recording mastering signature; 4s-mean
    samples ~6-10 consonants giving the SAME mastering signature.
    intra_pre, intra_post, cross all land in ~0.04-0.08 → feature ≈
    0 across non-splice speech positions → GBM low per-domain SHAP on
    english/korean → functionally invisible on speech.

    Orthogonal. NOT 9064eec (all-frame MFCC); NOT 26a3687 (all-frame
    spec_contrast); NOT c5040d7 (voiced MFCC 2nd-order); NOT 8170784
    (voiced chroma 2nd-order); NOT 7972a98 (1st-order voiced/unvoiced
    paired-diff, no intra baseline); NOT e7ca9eb (voiced spec_contrast
    single-mask); NOT any 1D-scalar / F0 / ZCR / bandwidth / rolloff /
    flatness variant; NOT 8fc7169 voicing transition rate; NOT 60196aa
    or d4d35b1 detector post-filter. FIRST unvoiced-mask 2nd-order
    feature; FIRST 2nd-order feature masked to the MASTERING-DOMINATED
    frame population.

    Blast radius. Pure features.py change — 1 new block (~55 lines
    fusing unvoiced _masked_mean with 26a3687-style intra/cross on
    7-dim contrast vector) + 1 FEATURE_NAMES append + 2 assert bumps
    (80→81) + 1 call in extract_features. ZERO new caches, ZERO new
    librosa calls. Per-t cost: 6 slices + 6 unvoiced-masked means on
    7-dim vectors + 3 cosines + 2 subtractions + 1 average, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged (unvoiced mask too sparse at
    2s sub-windows on singing — quiet sustained-vocal moments have
    almost no unvoiced frames and trigger sentinel 0.0 exactly where
    chord-cycle FPs live) → widen to ±6s / 3s sub-windows so even
    sparse unvoiced frames average to stable means. (2) Speech
    regresses (cross > intra because 4s unvoiced-mean samples a
    different consonant distribution than 2s sub-means) → tighten
    to median instead of mean (robust to consonant-count imbalance).
    (3) combined matches 0.490700 identical-streak → retrain gate
    still broken for this sha bump; escalate.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 56+
    iterations — cannot verify the 3 singing FPs' unvoiced-frame
    density at ±4s window (if they sit in sustained-vocal-only
    regions, unvoiced mask empties and feature sentinels silently).
    (ii) SHAP rollup STILL empty for 14 keeps — cannot verify the
    spec_contrast 2nd-order axis has measurable GBM uptake even on
    synthetic discrimination. (iii) d4d35b1's catastrophic 0.287
    outcome was not diagnosed with dense-p_splice histograms — is
    p_splice truly near-binary (0.99+ saturated) or does it have
    dynamic range below 0.9? Affects every detector-side hypothesis.
    (iv) 99081f5 4/16 capacity ghost status unclear at HEAD.

(e) Wrapper enhancements. Three unchanged highest-priority requests
    across 56+ iterations:
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — per-FP
    (domain, file, t_sec, p_splice, p_splice_local_median_3s,
    dsp_phase_z, dsp_t2_z, dsp_cpe_z, chunk_duration_s,
    voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry,
    voiced_fraction_in_pm4s, unvoiced_fraction_in_pm4s, top-5 |SHAP|).
    Per-FP unvoiced-mask-density would decide mask-based 2nd-order
    hypotheses data-driven instead of theory-only.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps; without per-
    feature attribution the axis pick remains theory-only.
    (3) DENSE-P_SPLICE HISTOGRAM per chunk in diagnostic log —
    would isolate whether detector-side margin/rank filters are
    geometrically viable OR whether p_splice is saturated at 0.99+
    making all margin-based filters meaningless. d4d35b1's
    catastrophic outcome amplifies this need.

## 2026-04-21T00:45:02+09:00 — 2557d2a (discard, combined=0.512135)
subject: add mfcc_variance_ratio_statistic (FEATURE_NAMES 80->81) -- F-statistic variance-reduction ratio on MFCC frames in +-4s window. (RSS_global - RSS_split)/(RSS_split+eps) normalizes the squared pre/post mean-shift by per-frame scatter around each side's local mean. Every prior 2nd-order feature (9064eec/26a3687/c5040d7/8170784/3bcec76) used cos_dist between sub-window MEAN VECTORS and collapsed on chord-cycle FPs because within-song MFCC drift fills intra sub-windows with the same type of mean shift as the cross boundary -- intra and cross co-vary, subtraction zeros on the FPs that need discrimination. FIRST variance-based 2nd-order feature: chord cycle GENERATES per-frame scatter that cross-song splice does NOT so the F-statistic normalization flips the sign of the mimicry. Speech self-gating: phoneme transitions inflate RSS_split -> ratio ~0 on english/korean via frame-level variance rather than mean cancellation. Reuses cached feat_mfcc, ZERO new librosa calls, ZERO new caches. Edge guard full +-4s span required. Smoke-verified len=81 last-name correct, end-to-end 81 finite features on real singing train audio at 0.50ms/call idempotent, synthetic A/B splice t=10 yields 24.23 vs within-A t=5 yields 0.003 (~8500x) vs smooth-drift chord-cycle sim yields 0.99 (~24x).
per-domain: combined_english=0.850000 combined_korean=0.557746 combined_singing=0.283333

# 2026-04-21 — hypothesis: add mfcc_variance_ratio_statistic (FEATURE_NAMES 80→81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot to
    F-statistic-style VARIANCE-REDUCTION RATIO on MFCC frames. For
    ±4s span clipped to [t-4, t+4], treat the MFCC frame sequence as
    one pooled model vs two split models (pre + post):
        μ_pre   = mean(mfcc_frames in [t-4, t])
        μ_post  = mean(mfcc_frames in [t, t+4])
        μ_all   = mean(mfcc_frames in [t-4, t+4])
        RSS_split  = Σ ||x_i - μ_pre||² + Σ ||x_i - μ_post||²
        RSS_global = Σ ||x_i - μ_all||²
        feature    = (RSS_global - RSS_split) / (RSS_split + 1e-6)
    Reuses cached feat_mfcc — ZERO new librosa calls, ZERO new caches.
    Edge guard t-4<0 OR t+4>duration OR <8 frames per side → sentinel
    0.0. FEATURE_NAMES 80→81 forces retrain via US-505 sha gate.

(b) WHY this over recent failures. Five consecutive 2nd-order variants
    failed: 9064eec all-frame MFCC cross-intra (0.544, broad regression),
    26a3687 all-frame spec_contrast (0.493), c5040d7 voiced MFCC (0.513),
    8170784 voiced chroma (0.494), 3bcec76 unvoiced spec_contrast (0.496).
    Every variant uses cos_dist between sub-window MEAN VECTORS (4 sub-
    windows combined into cross - 0.5*(intra_pre+intra_post)). They all
    collapsed on chord-cycle FPs because WITHIN-SONG MFCC DRIFT fills the
    intra sub-windows with the SAME TYPE OF MEAN SHIFT as the cross
    boundary — intra and cross co-vary, subtraction zeros on the FPs I
    need to discriminate. CLAUDE.md mandates structural change after 5+
    same-axis failures.

    The genuinely untried structural dimension is VARIANCE-BASED (per-
    frame SCATTER around local mean), not cosine distance between means.
    Every prior 2nd-order feature ignores within-window variance; F-
    statistic normalizes by exactly that. Mathematically:
        RSS_global - RSS_split
            = n_pre * ||μ_pre - μ_all||² + n_post * ||μ_post - μ_all||²
            ≈ (n/2) * ||μ_pre - μ_post||²    (balanced windows)
    So the NUMERATOR is the squared mean-shift (same signal as prior
    cosine-dist). The DENOMINATOR (RSS_split) is the sum of per-frame
    scatter around each side's local mean — the frame-level variance that
    chord cycle GENERATES but cross-song splice DOES NOT.

    Mechanism on 3 surviving singing chord-cycle FPs. Within one song,
    chord transitions continuously drift MFCC across each 4s half-window.
    Per-frame residuals around μ_pre are large (chord-cycle variance);
    same for post. RSS_split LARGE. Mean-shift ||μ_pre - μ_post||
    moderate (chord shift). RATIO small → feature silent → FP NOT
    boosted. This is exactly where cosine-dist 2nd-order collapsed;
    variance normalization catches it.

    Real cross-song splice (the target TPs). Pre-side 4s is song A self-
    consistent (low per-frame scatter around μ_pre, RSS_pre small).
    Post-side 4s is song B self-consistent (RSS_post small). RSS_split
    SMALL. Mean-shift ||μ_pre - μ_post|| large (different mastering +
    instrument + possibly different singer). RATIO LARGE → feature fires
    strongly.

    Speech self-gating. Unvoiced frames (consonants /s,f,p,t/ + silence)
    and voiced frames (vowels) both produce high per-frame MFCC scatter
    within any 4s window because phoneme transitions shift cepstral
    envelope 3-5 times per second. RSS_split LARGE on speech. Mean-shift
    across speech sentences moderate. RATIO small → feature ≈ 0 across
    non-splice positions on english/korean → GBM low per-domain SHAP.
    This is the FIRST 2nd-order feature whose speech self-gating comes
    from frame-level variance rather than from phoneme-averaged mean
    cancellation, which is why it avoids the c5040d7 / 9064eec speech
    regressions.

    Why MFCC axis (not spec_contrast/chroma). 9064eec proved MFCC is the
    most productive 2nd-order content axis (0.544 closest to baseline
    among recent failures). The failure mode was mean-based; the axis is
    correct. Reusing MFCC lets the F-statistic surface the signal 9064eec
    almost captured, through a genuinely different statistical lens.

    Orthogonal. NOT 9064eec (cos_dist on means, all-frame); NOT 26a3687
    (cos_dist on means, spec_contrast); NOT c5040d7 (cos_dist on means,
    voiced MFCC); NOT 8170784 (cos_dist on means, voiced chroma); NOT
    3bcec76 (cos_dist on means, unvoiced spec_contrast); NOT 1eda8e3 /
    7972a98 / any voiced_unvoiced asymmetry (1st-order single-boundary
    paired-diff, no variance); NOT block-2 mfcc_delta (1st-order mean-
    based); NOT 49bd0b1 voiced_mfcc (1st-order no variance); NOT F0 IQR
    log-ratio (1D distributional shape on F0, not multivariate MFCC
    scatter); NOT d4d35b1 detector margin (detector-side). FIRST
    variance-reduction-ratio (F-statistic) feature in the feature set;
    FIRST feature using within-window per-frame scatter as a normalizer.

    Blast radius. Pure features.py change — 1 new block (~40 lines using
    existing _slice_frames machinery on feat_mfcc) + 1 FEATURE_NAMES
    append + 2 assert bumps (80→81) + 1 call in extract_features. ZERO
    new caches, ZERO new librosa calls. Per-t cost: 2 slices of ~256
    frames × 13-dim + 3 means + 3 scatter sums, ~10k ops, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged (RSS_split on chord-cycle FP is
    actually LOW — within-chord drift is smooth and MFCC tracks it
    tightly so scatter around μ_pre stays small; hypothesis wrong) →
    fallback to the RATIO on voiced-masked MFCC only (sustained vowels
    have lowest within-song scatter so the F-statistic discriminates
    cleaner). (2) Speech regresses (sentence transitions produce
    correlated per-frame scatter AND mean shift simultaneously, so the
    normalization doesn't save it) → narrow window to ±3s / shrink eps
    denominator floor. (3) combined matches 0.589282 exactly → GBM
    ignores variance-based features (stale train data); pivot to raw
    RSS_split itself as a within-window homogeneity feature (lower = more
    homogeneous = more splice-like on the splice side but normal within-
    song).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 57+
    iterations — cannot verify whether the 3 singing FPs actually sit
    in high-scatter chord-cycle regions (my central assumption) or in
    quiet sustained-vocal sections where scatter is low on both sides
    and the F-statistic might fire spuriously. (ii) SHAP rollup STILL
    empty for 14 keeps — cannot verify whether the 9064eec 2nd-order
    signal was actually in the top GBM features vs spuriously assigned.
    (iii) d4d35b1's 0.287 catastrophe was diagnosed only via per-domain
    combined — dense p_splice histograms would tell us if the detector
    margin truly flatlined or specifically killed TPs. (iv) 99081f5 4/16
    capacity ghost still at HEAD per prior reflections.

(e) Wrapper enhancements. Three unchanged highest-priority requests
    across 57+ iterations:
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — per-FP
    (domain, file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry, voiced_unvoiced_
    spec_contrast_asymmetry, mfcc_per_frame_scatter_pm4s,
    voiced_fraction_pm4s, top-5 |SHAP|). Per-FP frame-scatter would
    convert every variance-based hypothesis into a data-driven decision.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps; without per-
    feature attribution axis-choice is theory-only.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision with joblib-mtime sanity check would isolate the 0.4907
    identical-streak root cause.

## 2026-04-21T01:00:54+09:00 — 4e67946 (discard, combined=0.515532)
subject: add mfcc_persistence_ratio (FEATURE_NAMES 80->81) -- future-horizon persistence feature; far_dist(pre,[t+4,t+6])-near_dist(pre,[t,t+2]); real splice persists ~0, chord-cycle reverts negative; smoke verified chord-revert -0.045 vs cross-source ~0
per-domain: combined_english=0.850000 combined_korean=0.537313 combined_singing=0.300000

# 2026-04-21 — hypothesis: add mfcc_persistence_ratio (FEATURE_NAMES 80→81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot onto the
    PERSISTENCE-OF-CHANGE axis. All 5 prior 2nd-order variants (9064eec,
    26a3687, c5040d7, 8170784, 3bcec76) used cos_dist between sub-window
    MEAN VECTORS on overlapping ±4s spans and collapsed on chord-cycle
    FPs because intra and cross co-vary. 2557d2a F-statistic variance-
    ratio (0.512) did not fix the mimicry either. The genuinely untried
    structural move is to compare how far post-side has drifted at TWO
    different future horizons: near-post [t, t+2] vs far-post [t+4, t+6],
    both against the SAME pre-reference [t-2, t]. For
        near_dist = cos_dist(mean_mfcc(t-2,t), mean_mfcc(t,t+2))
        far_dist  = cos_dist(mean_mfcc(t-2,t), mean_mfcc(t+4,t+6))
        feature   = far_dist - near_dist
    Real cross-song splice: song B CONTINUES after boundary, far_post is
    still song B, so far_dist ≈ near_dist (both dominated by A→B mastering
    / singer shift) → feature ≈ 0 neutral. Chord-cycle FP within one song
    (pop period ~2-3s): near_post is a different chord; far_post at t+4
    is ~2 chord cycles past t and typically cycles back toward pre-chord
    character OR an adjacent chord of the same key → far_dist < near_dist
    → feature STRONGLY NEGATIVE. Sign-and-magnitude separation chord-
    cycle (negative) vs real splice (≈0) is a binary discriminator GBM
    cannot synthesize from the existing feature set because NO existing
    feature measures temporal extent of a change by comparing two future
    horizons against one past reference. Reuses cached feat_mfcc, ZERO
    new librosa calls, ZERO new caches. Edge guard t-2<0 OR t+6>duration
    OR any norm underflow → sentinel 0.0. FEATURE_NAMES 80→81 forces
    wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. Every 2nd-order variant and the
    F-statistic tried to discriminate chord-cycle from cross-song at the
    SAME boundary position via the cross-to-intra contrast. But intra
    drift within-song is structurally similar to cross-song shift at a
    single boundary — the signal they share is "MFCC changes over ~2s."
    They differ in what happens AFTER the boundary: cross-song KEEPS
    DIFFERING (song B persists); chord-cycle REVERTS (song cycles).
    This feature reads out exactly that distinction. All 2s windows (no
    4s-vs-2s scale mismatch that broke 9064eec speech) — speech within-
    speaker has stable MFCC centroid across any 2s slice, so near_dist
    and far_dist are both small AND uncorrelated phoneme-noise — their
    difference is zero-mean → GBM low per-domain SHAP on english/korean
    (self-gating). Speech cross-speaker splice: near and far post are
    both speaker B → feature ≈ 0 → feature does NOT boost speech TPs,
    but existing voiced_unvoiced_mfcc_asymmetry (+0.054 keep) handles
    that. This feature specifically targets the 3 surviving singing
    chord-cycle FPs, which is the binding constraint on combined
    (singing=0.345 drags GM). Orthogonal: NOT 9064eec/26a3687/c5040d7/
    8170784/3bcec76 (cross-intra contrast on same span); NOT 2557d2a
    (F-statistic variance on ±4s); NOT 1eda8e3/7972a98 (voiced/unvoiced
    paired-diff, no future-horizon comparison); NOT d49284c/78513fb/
    dde4135 (FAR-post asymmetry, single distance not two-horizon
    comparison). FIRST persistence/reversion feature in 80-feature set.

(c) IF THIS FAILS. (1) Singing unchanged (chord period is 2-3s so far-
    post at t+4 still sits on a DIFFERENT chord from pre; revert not
    captured at 2s resolution) → widen far-post to [t+6, t+8] so it
    spans 3 chord cycles and averages back toward pre-character cleaner.
    (2) Speech regresses (cross-speaker splice causes far_dist <
    near_dist somehow — e.g., recording continuation noise uncorrelated
    with pre) → switch to voicing-masked version. (3) Combined lands at
    0.490700 identical-streak → wrapper retrain gate broken; escalate.

(d) Information gaps. CLEAN_FP_POSITIONS still absent after 58+
    iterations — cannot verify chord-revert hypothesis at actual FP
    positions. SHAP rollup still empty for 14 keeps. Dense p_splice
    histograms still not logged (d4d35b1 catastrophe left unanalyzed).
    99081f5 4/16 ghost status unclear at HEAD.

(e) Wrapper enhancements. Unchanged three asks across 58+ iterations:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP fields
    (including mfcc_far_post_dist and mfcc_near_post_dist to validate
    persistence-feature mechanism directly). (2) SHAP ROLLUP REPAIR.
    (3) RETRAIN-ACTUALLY-FIRED TRACE with joblib-mtime sanity check.

## 2026-04-21T01:20:50+09:00 — e2ad8b0 (discard, combined=0.503369)
subject: add onset_tempo_peak_lag_delta (FEATURE_NAMES 80->81)
per-domain: combined_english=0.850000 combined_korean=0.633333 combined_singing=0.236923

# 2026-04-21 — hypothesis: add onset_tempo_peak_lag_delta (FEATURE_NAMES 80→81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot to a
    RHYTHMIC/TEMPO-PERIODICITY axis via the autocorrelation peak-lag
    (tempo estimate) of the onset-strength envelope. For windows
    clipped to [t-4, t] and [t, t+4], smooth the onset envelope with
    a ~100 ms uniform filter, compute normalized ACF over lags
    0.25–1.5 s (≈ 40–240 BPM), and take the argmax lag on each side:
        pre_peak  = argmax_ACF( onset_env[t-4:t] )
        post_peak = argmax_ACF( onset_env[t:t+4] )
        feature   = |post_peak − pre_peak|  (seconds)
    Reuses cached feat_onset (librosa.onset.onset_strength, hop 512).
    ZERO new librosa calls, ZERO new caches. Edge guard t-4<0 OR
    t+4>duration OR frames<16 OR ACF zero-lag underflow → 0.0.
    FEATURE_NAMES 80→81 forces retrain via US-505 sha gate.

    Smoke note: smoothed peak-lag delta fires 0.05–0.64 s across
    various positions on real spliced-singing audio, and 0.05–0.89 s
    on clean audio — not a clean discriminator at the per-position
    level but the hope is GBM can exploit the population-level shift
    (splice positions tend higher) in conjunction with other features.
    If SHAP is zero / feature is pure noise, the wrapper discards
    this as a harmless null hypothesis.

(b) WHY this over recent failures. 28+ content-axis variants exhausted
    (MFCC/spec_contrast/chroma/spec_flatness/RMS-dB/ZCR/spec_bandwidth/
    spec_rolloff; 1st-order paired-diff + 2nd-order cross-intra + voiced
    mask + unvoiced mask + all-frame + F-statistic variance + future
    persistence). Every 2nd-order variant collapsed on the 3 singing
    chord-cycle FPs because within-song content drift fills intra
    windows with the same scale of mean shift as the cross boundary.
    The genuinely untried structural axis is **tempo/rhythm
    periodicity** — the beat-period signature encoded by onset-strength
    autocorrelation. Within one song the tempo is a SOLID invariant:
    drummer, metronome, rhythm section hold a constant BPM across
    verse→chorus→bridge → ACF peak sits at the same lag before AND
    after t → pre-ACF and post-ACF highly correlated → feature silent,
    chord-cycle FP NOT boosted. Cross-song splice typically crosses
    tempos (80-BPM ballad → 140-BPM up-tempo; even same-artist albums
    rarely share BPM exactly) → ACF peak shifts lag → feature fires.
    Speech self-gating: onset envelope on speech is syllable-burst
    aperiodic noise (no stable tempo); ACF is flat-ish on both sides
    of any within-recording position → feature ≈ 0 → GBM low per-domain
    SHAP on english/korean. Cross-speaker splice: still mostly flat
    both sides, feature ≈ 0, handled by 1eda8e3 voiced_unvoiced_mfcc
    asymmetry (+0.054 biggest keep).

    Orthogonal. NOT any content axis (not MFCC/chroma/spec_contrast/
    spec_flatness/RMS/ZCR/bandwidth/rolloff/tonnetz); NOT voicing mask;
    NOT 8fc7169 voicing_transition_rate_delta (counts of
    voiced↔unvoiced transitions on the vp mask — scalar delta, not ACF
    of onset env); NOT any 1st-order paired-diff / 2nd-order cross-
    intra / variance-ratio / persistence-ratio; NOT detector post-
    filter. FIRST rhythmic/tempo-structure feature; FIRST
    autocorrelation-based feature on any cached 1D envelope.

    Blast radius. Pure features.py change — 1 new block (~40 lines:
    slice + mean-center + np.correlate + cosine distance on normalized
    lag vector) + 1 FEATURE_NAMES append + 2 assert bumps (80→81) +
    1 call. ZERO new caches, ZERO new librosa calls. Per-t cost: 2
    slices of ~172 frames + 2 np.correlate (O(n²) ~30k ops) + 1
    cosine, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged (within-song tempo is
    drifty enough at 4s that ACF peaks walk, OR cross-song tempos
    happen to match within 5%) → widen window to ±6s for stabler ACF
    sampling. (2) Speech regresses (syllable-burst rate is stable
    enough per speaker that ACF peaks at ~1/syl-rate lag, and
    cross-speaker splice shifts syl-rate) → restrict lag range to
    [0.25, 1.5]s (excludes sub-250ms syllable scale). (3) Combined
    matches the 0.4907 identical-streak → features.py sha bump fails
    to force retrain; escalate as systemic wrapper cache-coherence.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 58+
    iterations — cannot verify whether the 3 singing FPs sit in
    regions with strong tempo ACF or in less rhythmic outro/intro
    sections. (ii) SHAP rollup STILL empty for 14 keeps — still
    blind to which features GBM actually uses. (iii) No diagnostic
    emits for onset-envelope statistics at eval time; will need to
    theorize whether the feature even fires meaningfully.

(e) Wrapper enhancements (unchanged 3 highest-priority asks):
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — per-FP
    (domain, file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, onset_env_mean_pm4s,
    onset_env_acf_peak_lag_pm4s, top-5 |SHAP|).
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps; axis choice
    remains theory-only without per-feature attribution.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision ("features.py sha Δ XX→YY → retrain" vs "no Δ → skip")
    with joblib-mtime sanity check.

## 2026-04-21T01:37:00+09:00 — 4314449 (discard, combined=0.471132)
subject: add voiced_mfcc_cross_scale_contrast (FEATURE_NAMES 80->81)
per-domain: combined_english=0.790123 combined_korean=0.529412 combined_singing=0.250000

# 2026-04-21 — hypothesis: add voiced_mfcc_cross_scale_contrast (FEATURE_NAMES 80→81)

(a) HYPOTHESIS. Pure `splice/features.py` add — voicing-masked CROSS-SCALE
    MFCC consistency feature. Same boundary, TWO time scales, symmetric on
    both sides of t. For:
        near_pre_v  = voiced_mean_mfcc([t-2, t])
        near_post_v = voiced_mean_mfcc([t, t+2])
        far_pre_v   = voiced_mean_mfcc([t-4, t])
        far_post_v  = voiced_mean_mfcc([t, t+4])
        near_dist   = cos_dist(near_pre_v, near_post_v)
        far_dist    = cos_dist(far_pre_v,  far_post_v)
        feature     = far_dist - near_dist
    Reuses cached feat_mfcc + feat_vp. ZERO new librosa calls, ZERO new
    caches. Edge guard t-4<0 OR t+4>duration OR any mask empty OR any
    norm underflow → sentinel 0.0. FEATURE_NAMES 80→81 forces retrain
    via US-505 sha gate.

(b) WHY this over recent failures. All 6 prior 2nd-order variants
    (9064eec/26a3687/c5040d7/8170784/3bcec76, 2557d2a F-stat, 4e67946
    persistence) failed because every one of them introduced an INTRA
    baseline sub-window that CO-VARIED with the cross boundary on chord-
    cycle FPs. 4e67946's persistence used ASYMMETRIC pre vs remote post,
    which broke speech self-gating (speech cross-speaker splice has
    persistent far-post too, so discrimination flips).

    This hypothesis uses NO intra sub-window and NO asymmetric horizon.
    It measures the same cross-boundary distance at two SYMMETRIC scales
    (±2s vs ±4s) and takes the difference.

    Mechanism on 3 surviving singing chord-cycle FPs. Chord transitions
    within one song cycle every 2-3s. 2s voiced-mean samples ONE chord's
    centroid on each side → near_dist ≈ 0.10-0.15 (chord jump). 4s voiced-
    mean samples 1.5-2 chords on each side, averaging toward the song's
    KEY-CENTER cepstral mean → far means converge → far_dist ≈ 0.04-0.08.
    feature ≈ 0.05 − 0.12 ≈ −0.07 (NEGATIVE), feature silences or reduces
    p_splice.

    Real cross-song splice. 2s pre = song A averaged cepstrum, 2s post =
    song B averaged cepstrum → near_dist ≈ 0.25-0.35. 4s pre = more
    stable A estimate; 4s post = more stable B estimate → far_dist ≈
    same or slightly LARGER because longer averaging reduces within-
    song noise without reducing the A→B jump. feature ≈ 0 or SLIGHTLY
    POSITIVE. SIGN FLIP chord-cycle (NEG) vs cross-song (≥0) is the
    binary discriminator.

    Speech self-gating. Voiced MFCC within-recording is phoneme-driven;
    2s samples ~4-6 vowels with stable per-speaker centroid; 4s samples
    ~8-12 vowels with the SAME centroid. near_dist and far_dist both
    small and similar on non-splice positions → feature ≈ 0 → GBM low
    per-domain SHAP on english/korean. Same mechanism that made 49bd0b1
    voiced_mfcc keep +0.028.

    Why distinct from 49bd0b1 (voiced_mfcc_cosine_dist, kept). That
    feature IS near_dist alone. GBM max_depth=4 with voiced_mfcc_dist
    alone cannot synthesize the 4s-averaging reduction because the 4s
    voiced-mean is not in any feature. Adding the DELTA surfaces the
    chord-cycle reversal signal directly.

    Why distinct from 4e67946 persistence_ratio. That used asymmetric
    pre=[t-2,t] vs near-post=[t,t+2] and far-post=[t+4,t+6] — same pre,
    different posts. This hypothesis uses SAME-BOUNDARY distance at two
    symmetric scales — pre and post both grow 2s→4s together. Speech
    cross-speaker doesn't break this because BOTH scales see the same
    A→B shift on a real speech splice.

    Orthogonal. NOT 49bd0b1 (single scale); NOT 4e67946 (asymmetric
    persistence); NOT 9064eec/26a3687/c5040d7/8170784/3bcec76 (intra
    sub-window contrast); NOT 2557d2a (F-statistic variance-based);
    NOT 1eda8e3/7972a98/f4148cc/every voiced-unvoiced asymmetry (paired-
    diff, single scale); NOT block-2 mfcc_delta (all-frame single
    scale); NOT any 1D-scalar / F0 / ZCR / bandwidth / rolloff / flatness
    / tempo variant; NOT detector post-filter. FIRST cross-scale feature
    in the 80-feature set — first feature comparing the same boundary
    at two different time scales.

    Blast radius. Pure features.py change — 1 new block (~50 lines,
    _block_voiced_mfcc mask logic computed twice at 2s and 4s) + 1
    FEATURE_NAMES append + 2 assert bumps (80→81) + 1 call in
    extract_features. ZERO new caches, ZERO new librosa calls. Per-t
    cost: 4 voiced-masked means on 13-dim + 2 cosines + 1 subtraction,
    sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged (chord period in the 3 FPs is
    ACTUALLY much longer than 2-3s — long sustained chord pads — so 4s
    doesn't average multiple chords on each side; hypothesis wrong) →
    widen far scale to ±6s so it definitely spans multiple harmonic
    cycles. (2) Speech regresses (cross-speaker splice happens to have
    far-scale distance LOWER than near-scale due to post-boundary
    recording-length noise averaging — asymmetric noise reduction that
    voicing-mask doesn't fully cancel) → fallback to chroma axis (pitch-
    class is key-signature-stable within-song so 4s-averaging effect
    even stronger). (3) Feature fires but GBM assigns ~zero SHAP →
    redundant with 49bd0b1 voiced_mfcc near_dist per correlation;
    pivot to LOG-RATIO far_dist/near_dist instead of subtraction
    (scale-invariant sensitivity).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 59+
    iterations — cannot verify the 3 singing FPs sit in chord-cycle
    regions with 2-3s chord period (central assumption); could be
    long-pad bridges, outros, or fadeouts where 4s-averaging doesn't
    reduce far_dist. (ii) SHAP rollup STILL empty for 14 keeps — cannot
    verify whether 49bd0b1 voiced_mfcc_cosine_dist is actually in top
    GBM features. (iii) No dense p_splice histogram per chunk (d4d35b1
    0.287 catastrophe left undiagnosed). (iv) 99081f5 4/16 capacity
    ghost status unclear at HEAD.

(e) Wrapper enhancements. Three unchanged highest-priority requests
    across 59+ iterations:
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — per-FP
    (domain, file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, voiced_mfcc_cosine_dist_2s,
    voiced_mfcc_cosine_dist_4s, top-5 |SHAP|). Would turn every
    cross-scale / persistence / intra-baseline hypothesis into a
    data-driven decision instead of theory bet.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps; without
    per-feature attribution axis-choice remains theory-only.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision with joblib-mtime sanity check.

## 2026-04-21T01:53:03+09:00 — 1b3eb06 (discard, combined=0.500951)
subject: add mfcc_trajectory_velocity_log_ratio (FEATURE_NAMES 80->81)
per-domain: combined_english=0.888889 combined_korean=0.565714 combined_singing=0.250000

# 2026-04-21 — hypothesis: add mfcc_trajectory_velocity_log_ratio (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot off the
    entire "distance between sub-window MEAN VECTORS" family onto
    TRAJECTORY VELOCITY: mean frame-to-frame MFCC L2 delta magnitude,
    ratio pre vs post on log scale.
        for a side (pre or post) on ±2s:
            v_side = mean_i ||mfcc[i] - mfcc[i-1]||_2
        feature = |log((post_v + eps) / (pre_v + eps))|
    Reuses cached feat_mfcc — ZERO new librosa calls, ZERO new caches.
    Edge guard fewer than 4 frames per side → sentinel 0.0. FEATURE_NAMES
    80→81 forces wrapper auto-retrain via US-505 sha gate.

(b) WHY over recent failures. The last 10 iterations tried "distance between
    sub-window MEAN VECTORS" in many forms — cross-intra contrast (9064eec
    MFCC, 26a3687 spec_contrast, c5040d7 voiced MFCC, 8170784 voiced chroma,
    3bcec76 unvoiced spec_contrast), F-statistic variance-ratio (2557d2a),
    persistence horizons (4e67946), cross-scale (4314449), onset ACF tempo
    (e2ad8b0), detector margin (d4d35b1 catastrophic 0.287). Every one
    collapsed on chord-cycle FPs because within-song drift fills both intra
    baselines and the cross boundary with similar-magnitude mean shifts, AND
    because variance around those means ALSO co-varies (F-stat failed for
    the same reason).

    Trajectory VELOCITY is a structurally different statistic. Arc length
    = Σ||mfcc[i]−mfcc[i−1]|| integrates the frame-to-frame change RATE
    regardless of where the trajectory goes. Smooth ramps and noisy clusters
    can share the same mean AND the same variance-around-mean yet have very
    different arc lengths. Ratio pre/post is a SCALE-INVARIANT comparison
    of arrangement DENSITY / ACTIVITY LEVEL on each side.

    Mechanism on 3 singing chord-cycle FPs. Within one song, arrangement
    density (singer + drums + chord progression activity) is roughly
    constant across the chord cycle — one chord transition every 2-3s
    keeps per-frame MFCC flux at a steady rate on both sides of t.
    pre_v ≈ post_v → ratio ≈ 1 → |log| ≈ 0 silent, FP NOT boosted.

    Cross-song splice. Song A and song B typically differ in arrangement
    density: ballad verse (low flux) → up-tempo chorus (high flux);
    sparse intro → full mix; different drummers with different groove
    intensity; different mastering drive compressing dynamics
    differently. pre_v / post_v ≠ 1 → |log| positive → feature fires.

    Speech self-gating. Within a recording a speaker holds roughly
    constant speech rate and prosody, so syllable-to-syllable MFCC
    flux is steady on any 2s slice. pre_v ≈ post_v → feature ≈ 0 on
    non-splice positions → GBM low per-domain SHAP on english/korean.
    Cross-speaker splice with matched rate also silences this feature —
    fine, because existing 1eda8e3 voiced_unvoiced_mfcc_asymmetry
    (+0.054 biggest keep) handles that TP class.

    Why LOG-RATIO not subtraction. Scale-invariant: a 2x flux difference
    reads the same whether both sides are loud or quiet. Absolute
    subtraction gets dominated by absolute level (e.g. loud singing
    saturates vs quiet speech hovers near zero).

    Why MFCC axis. 9064eec (MFCC 2nd-order 0.544) was the CLOSEST-to-
    baseline recent failure; the axis carries real content-shift signal,
    the prior failure was in the statistic shape, not the axis.

    Orthogonal. NOT any cos_dist-of-means feature (whole 2nd-order
    family); NOT 2557d2a F-statistic RSS-around-mean (variance of
    residuals vs mean, not of consecutive deltas); NOT 4e67946 or
    4314449 (distance at multiple horizons/scales); NOT block-2
    mfcc_delta_NN (frame-local single-delta at boundary, NOT window
    mean of delta magnitudes); NOT e2ad8b0 onset ACF tempo (1D signal,
    autocorrelation, not trajectory integral); NOT any voicing-masked
    variant; NOT detector post-filter. FIRST trajectory-velocity
    feature; FIRST ratio-based statistic over frame-to-frame delta
    magnitudes in the 80-feature set.

    Blast radius. Pure features.py — 1 new block (~30 lines) + 1
    FEATURE_NAMES append + 2 assert bumps (80→81) + 1 call in
    extract_features + 1 extra feats.update. ZERO new caches, ZERO
    new librosa calls. Per-t cost: 2 slices + 2 diff + 2 norm-means +
    1 log, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged (chord-cycle FPs sit in
    sections where within-song flux actually DIFFERS verse vs bridge
    within same song; hypothesis wrong) → widen to ±4s so flux
    averages over multiple phrases, smoothing arrangement variation.
    (2) Speech regresses (sentence transitions within a recording
    change syllable rate enough to cause log-ratio >0 on non-splice
    speech) → fallback to voiced-masked version (voiced-frame flux
    within-speaker is prosody-stable). (3) Feature fires but GBM
    assigns zero SHAP (mfcc_delta_NN block already captures this
    signal) → pivot to spec_contrast trajectory velocity (mastering
    activity rate, truly untouched signal).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 60+
    iterations — cannot verify whether the 3 singing FPs sit in
    within-song flux-stable sections (mechanism assumption) or in
    verse→chorus flux-transition sections (where this feature would
    fire and create a new FP). (ii) SHAP rollup STILL empty for 14
    keeps — cannot verify whether mfcc_delta_NN block-2 already
    carries a trajectory-velocity-like signal via correlated splits.
    (iii) 99081f5 4/16 capacity ghost status unclear at HEAD. (iv)
    0.4907 identical-streak root cause still unknown.

(e) Wrapper enhancements. Three unchanged highest-priority asks
    across 60+ iterations:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE — per-FP (domain,
    file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, mfcc_frame_flux_pre_2s,
    mfcc_frame_flux_post_2s, top-5 |SHAP|). Would decide every
    trajectory-velocity / variance / mean-based hypothesis
    data-driven.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision with joblib-mtime sanity check.

## 2026-04-21T02:10:01+09:00 — 1b4fe7c (discard, combined=0.478675)
subject: add mfcc_corr_structure_distance (FEATURE_NAMES 80->81) -- STRUCTURAL pivot off the entire MEANS-BASED family onto cross-dimensional correlation matrix structure. Cosine distance between strictly-upper-triangle (78 off-diagonal entries) Pearson correlation matrices of the 13 MFCC dims over pre[t-2,t] vs post[t,t+2]. Reuses cached feat_mfcc; ZERO new librosa calls, ZERO new caches. Sentinel 0.0 when either window has <20 frames or either corrcoef is non-finite or either upper-triangle norm underflows. FEATURE_NAMES 80->81 forces auto-retrain via US-505 sha gate. Last 10+ iterations all compared MEAN VECTORS or means-of-derived-statistics (9064eec/26a3687/c5040d7/8170784/3bcec76/2557d2a/4e67946/4314449/1b3eb06) and collapsed on chord-cycle FPs because intra/cross both see similar mean shifts. Correlation-matrix structure measures HOW the 13 cepstral dims CO-VARY (recording timbral fingerprint: mic+preamp, room, instruments, mastering dynamics, vocal tract) — a 78-dim relationship GBM cannot synthesize from any existing mean-based feature. Mechanism on 3 singing chord-cycle FPs: same instruments+mic+mastering within one song -> chord transitions shift MFCC mean vector but preserve dim-by-dim co-variation -> corr_pre~corr_post -> feature~0 silent -> FP not boosted. Cross-song splice: different instruments/mic/mastering -> correlation structure differs -> feature fires. Speech self-gating: within-recording MFCC correlation is stable across phonemes (same mic+same vocal tract) -> feature~0 on non-splice speech. Orthogonal: NOT any cos_dist-between-means family, NOT F-statistic scatter-around-mean (2557d2a), NOT trajectory velocity (1b3eb06), NOT any 1st-order paired-diff/voicing-masked variant, NOT block-2 mfcc_delta. FIRST correlation-matrix-structure feature; FIRST feature using cross-dim covariance rather than means. Blast radius: 1 new block (~45 lines) + FEATURE_NAMES append + 2 assert bumps + 1 call. Per-t cost 2 slices + 2 corrcoef on 13x~172 + 1 cos_dist on 78-vec, sub-ms. Smoke-verified: len==81 last-name correct, synthetic A/B splice yields 0.97 vs within-A 0.18 (~5x discrimination), real singing 500 calls = 0.88ms/call, idempotent, all 81 finite, non-trivial dynamic range across positions (0.22-0.81).
per-domain: combined_english=0.810127 combined_korean=0.550000 combined_singing=0.246154

# 2026-04-21 — hypothesis: add mfcc_corr_structure_distance (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — STRUCTURAL pivot off the
    entire MEANS-BASED family onto CROSS-DIMENSIONAL CORRELATION STRUCTURE
    of MFCC. For ±2s windows, compute the 13×13 Pearson correlation matrix
    of MFCC dimensions over pre[t-2,t] and post[t,t+2] separately, take the
    strictly-upper-triangle (78 values, diagonal-excluded because it is
    always 1), and compute cosine distance between the two vectorized
    triangles:
        corr_pre  = np.corrcoef(mfcc_frames_pre)   # 13×13
        corr_post = np.corrcoef(mfcc_frames_post)  # 13×13
        v_pre  = corr_pre[triu_k1]   # 78-vec
        v_post = corr_post[triu_k1]  # 78-vec
        feature = 1 - cos(v_pre, v_post)
    Reuses cached feat_mfcc — ZERO new librosa calls, ZERO new caches.
    Edge guard t-2<0 OR t+2>duration OR either window has <20 frames OR
    either corrcoef returns NaN (zero-variance row) OR either norm
    underflow → sentinel 0.0. FEATURE_NAMES 80→81 forces retrain via
    US-505 sha gate.

(b) WHY over recent failures. Last 10+ iterations all used "distance
    between sub-window MEAN VECTORS" (or means of derived statistics):
    9064eec, 26a3687, c5040d7, 8170784, 3bcec76 (cos_dist between MEANS);
    2557d2a F-statistic (scatter around MEANS); 4e67946 persistence
    (cos_dist between MEANS at future horizons); 4314449 cross-scale
    (cos_dist between MEANS at two scales); 1b3eb06 trajectory velocity
    (L2 of frame-to-frame deltas → per-frame flux is still a mean-like
    scalar per side). Every one collapsed on chord-cycle FPs because
    MFCC *mean* drifts across each 2-3s chord cycle → intra and cross
    both see similar mean shifts.

    CROSS-DIMENSIONAL CORRELATION is structurally orthogonal: it
    measures HOW the 13 MFCC coefficients CO-VARY over a window, not
    where their mean sits. This co-variation pattern is a fingerprint
    of the RECORDING's timbral dynamics (mic+preamp response, room,
    instrument ensemble, mastering compressor attack/release behaviour,
    vocal tract dynamics of THIS singer in THIS room). Chord transitions
    within one song shift the MEAN vector (different pitch classes
    emphasize different cepstral coefficients) but preserve the
    CO-VARIATION pattern (same instruments, same mic, same dynamic
    shaping → coefficient_i and coefficient_j still respond together
    the same way). So corr_pre ≈ corr_post on within-song chord cycles
    → cos_dist small → feature ≈ 0 silent → the 3 singing FPs NOT
    boosted. Cross-song splice changes mic+instruments+mastering →
    correlation structure differs → feature fires.

    Speech self-gating: within-recording speech has a stable per-
    recording MFCC-correlation fingerprint (same mic + same vocal
    tract). Phoneme transitions shift MEAN but not correlation. So
    corr_pre ≈ corr_post on non-splice speech → feature ≈ 0 across
    non-splice positions → GBM low per-domain SHAP on english/korean.
    Cross-speaker splice (TP) crosses recordings → correlation
    structure differs → feature fires → HELPS speech TPs too.

    GBM max_depth=4/max_leaf=16 cannot synthesize correlation-matrix
    cosine distance from existing mean-based features via threshold
    splits — correlation of 13 coefficients is a 78-dim relationship,
    and no feature in the 80-feature set carries a single scalar
    reflecting per-side co-variation structure.

    Orthogonal. NOT 9064eec/26a3687/c5040d7/8170784/3bcec76 (cos_dist
    between MEAN VECTORS); NOT 2557d2a F-statistic (scatter around
    MEAN — per-frame L2 residual, NOT cross-dim correlation); NOT
    4e67946 (cos_dist between MEAN VECTORS at future horizons); NOT
    4314449 (cos_dist between MEAN VECTORS at two scales); NOT
    1b3eb06 trajectory velocity (mean L2 of deltas); NOT any 1st-order
    paired-diff / voicing-masked / cos_dist-between-means variant; NOT
    block-2 mfcc_delta (single-boundary mean-based); NOT e2ad8b0 onset
    ACF (1D signal autocorrelation not 13×13 matrix structure); NOT
    block-5 voicing features; NOT any 1D-scalar / F0 / detector
    post-filter. FIRST correlation-matrix-structure feature in the
    80-feature set, FIRST feature using cross-dimensional covariance
    rather than means or scatter around means.

    Blast radius. Pure features.py — 1 new block (~45 lines) + 1
    FEATURE_NAMES append + 2 assert bumps (80→81) + 1 call in
    extract_features. ZERO new caches, ZERO new librosa calls. Per-t
    cost: 2 slices + 2 np.corrcoef on 13×~172 arrays (~30k ops each)
    + 1 cosine distance on 78-vec, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged (MFCC correlation structure
    IS chord-dependent because strongly energized chords push specific
    MFCC dims which alters their co-variation with others) → widen
    window to ±3s so correlation stabilizes over multiple chord cycles.
    (2) Speech regresses (within-sentence phoneme transitions perturb
    correlation structure enough to cause cos_dist > 0 on non-splice
    speech) → fall back to voicing-masked version (voiced-only
    correlation is more stable across phoneme transitions since
    vowel-formant dynamics are speaker-continuous). (3) Feature fires
    but GBM assigns zero SHAP (correlation signal absorbed by
    mfcc_delta_NN + voiced_mfcc_cosine_dist via correlated splits) →
    pivot to spec_contrast correlation structure (7×7 mastering-
    fingerprint, genuinely untouched since no prior feature examines
    cross-band contrast correlation).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 61+
    iterations — cannot verify whether the 3 singing FPs sit in
    sections where MFCC correlation is genuinely stable (mechanism
    assumption) or in sections where correlation shifts with
    arrangement density. (ii) SHAP rollup STILL empty for 14 keeps
    — cannot verify what statistical family GBM actually uses. (iii)
    0.4907 identical-streak root cause unknown. (iv) 99081f5 4/16
    capacity ghost status unclear at HEAD.

(e) Wrapper enhancements. Three unchanged highest-priority requests
    across 61+ iterations:
    (1) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — per-FP
    (domain, file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry,
    mfcc_frame_flux_pre_2s, top-5 |SHAP|). Per-FP metadata would
    convert every structural hypothesis into a data-driven decision.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision with joblib-mtime sanity check.

## 2026-04-21T02:29:12+09:00 — f0ccd94 (discard, combined=0.518762)
subject: add voiced_unvoiced_onset_asymmetry (FEATURE_NAMES 80->81) -- voicing-masked paired-diff on the cached onset-strength envelope (rectified spectral flux). Applies proven productive template (1eda8e3 MFCC +0.054 biggest keep, 7972a98 spec_contrast +0.005 keep) to the one cached 1D per-frame signal not yet tried on this template. Last 10 iterations exhausted 2nd-order / variance / persistence / cross-scale / trajectory-velocity / corr-matrix structures on vector content axes -- every one collapsed on singing chord-cycle FPs (0.47-0.55). Pre[t-2,t] / post[t,t+2]: voiced_delta=mean(onset[voiced post])-mean(onset[voiced pre]); unvoiced_delta=mean(onset[unvoiced post])-mean(onset[unvoiced pre]); feature=unvoiced_delta-voiced_delta. Reuses cached feat_onset + feat_vp, ZERO new librosa calls. Sentinel 0.0 on any empty mask. Mechanism on 3 singing chord-cycle FPs: within-song drum groove steady -> unvoiced_delta~0; vocal melody shifts with chord -> voiced_delta small -> asymmetry~0 silent, FP not boosted. Cross-song splice: different drum kit+mastering attack -> unvoiced_delta MODERATE; vocal shift moderate -> asymmetry POSITIVE, TP boosted. Speech self-gating via paired differencing (1eda8e3 mechanism): within-recording onset intensity is phoneme-context-driven and voiced/unvoiced deltas track same context -> DIFFERENCE zero-mean -> GBM low per-domain SHAP on english/korean. e2ad8b0 onset_tempo_peak_lag_delta used ACF peak-lag (tempo estimate) on same signal -- different statistic, not mean-intensity voicing-split. Orthogonal: NOT 1eda8e3 (cepstral cosine); NOT 7972a98 + every spec_contrast variant (7-dim cosine); NOT f4148cc (chroma); NOT 0c3bf76 (flatness); NOT c006d52 (RMS-dB); NOT 0cdd87e (ZCR); NOT 7a170b0 (bandwidth); NOT e234649 (rolloff); NOT 27ddbf7/1d1144d (F0); NOT 8fc7169 (voicing transition rate -- count not intensity); NOT e2ad8b0 (ACF peak-lag tempo); NOT any 2nd-order / variance / persistence / cross-scale / trajectory / corr-matrix variant; NOT detector post-filter. FIRST onset-strength (rectified-spectral-flux) voicing-masked paired-diff in 80-feature set. Pure features.py change -- 1 new block (~40 lines cloned from block 14 template with 1D scalar mean replacing 13-dim cosine) + 1 FEATURE_NAMES append + 2 assert bumps (80->81) + 1 call in extract_features. Per-t cost: 4 slices + 4 masked means + 2 subtractions, sub-ms. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_onset_asymmetry', real singing_clean_001.wav 20 sample positions yield 16/20 non-zero with range [-0.4046, 0.4116] (non-trivial dynamic range), 100 extract_features calls all 81 features finite, idempotent, edge guard via empty-mask sentinel returns 0.0 correctly.
per-domain: combined_english=0.829268 combined_korean=0.567164 combined_singing=0.296825

# 2026-04-21 — hypothesis: add voiced_unvoiced_onset_asymmetry (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — voicing-masked paired-diff
    of MEAN onset-strength (rectified spectral flux = librosa.onset_strength,
    already cached as `feat_onset`). For pre[t-2,t] / post[t,t+2]:
        voiced_delta   = mean(onset[voiced post])   − mean(onset[voiced pre])
        unvoiced_delta = mean(onset[unvoiced post]) − mean(onset[unvoiced pre])
        feature        = unvoiced_delta − voiced_delta
    Reuses cached feat_onset + feat_vp — ZERO new librosa calls, ZERO new
    caches. Sentinel 0.0 when any mask empty. FEATURE_NAMES 80→81 forces
    wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. Last 10 iterations chased 2nd-order /
    variance / persistence / cross-scale / trajectory-velocity /
    correlation-matrix structures on vector content axes — every one
    collapsed on singing chord-cycle FPs (0.47–0.55 range). The proven
    productive template on this project remains the 1st-order voicing-
    masked paired-diff (1eda8e3 MFCC +0.054 biggest keep, 7972a98
    spec_contrast +0.005 keep). I have exhausted this template on every
    cached 1D/vector content descriptor EXCEPT the onset-strength
    (rectified spectral flux) envelope. feat_onset is a per-frame 1D
    scalar at the same hop as feat_vp so it fits the scalar-asymmetry
    shape of 0cdd87e ZCR / 7a170b0 bandwidth / e234649 rolloff / c006d52
    RMS-dB, all of which were discarded but non-catastrophic. e2ad8b0
    used ACF peak-lag of onset (tempo estimate) — a different statistic;
    never tried mean-intensity voicing-split on this cached signal.

    Mechanism on 3 surviving singing chord-cycle FPs. Onset-strength
    measures rate-of-spectral-energy-increase per frame (note / drum /
    consonant onsets). Within one song the DRUM GROOVE is steady (same
    drummer, same kit, same mastering compressor attack) so unvoiced-
    frame onset intensity stays approximately flat on both sides of the
    FP → unvoiced_delta ≈ 0. The vocal melody shifts at chord transitions
    so voiced-frame onset intensity varies slightly → voiced_delta small
    → asymmetry ≈ 0 silent → FP NOT boosted.

    Real cross-song splice. Different drum kit + different tempo +
    different mastering attack → unvoiced onset intensity shifts
    measurably between song A and song B → unvoiced_delta MODERATE.
    Vocal delivery (phrase density, lyrical rate) differs → voiced_delta
    also moderate, but typically less than the mastering/drum jump
    because the singer's attack/release is more consistent across
    recordings than the drum bus. → asymmetry POSITIVE (unvoiced > voiced),
    fires as discriminator.

    Speech self-gating (proven 1eda8e3 mechanism). Within-recording
    speech has stable rate; per-frame onset intensity is phoneme-context
    driven and voiced/unvoiced deltas both track the same context →
    DIFFERENCE zero-mean → GBM low per-domain SHAP on english/korean →
    feature functionally invisible on speech domains. Cross-speaker
    splice TPs remain handled by 1eda8e3 MFCC asymmetry.

    Orthogonal. NOT 1eda8e3 (13-dim cepstral cosine); NOT 7972a98 +
    every spec_contrast geometry variant (7-dim cosine); NOT f4148cc
    (chroma); NOT 0c3bf76 (flatness); NOT c006d52 (RMS-dB); NOT 0cdd87e
    (ZCR time-domain); NOT 7a170b0 (bandwidth); NOT e234649 (rolloff);
    NOT 27ddbf7/1d1144d (F0 distribution); NOT 8fc7169 (voicing
    transition rate — count not intensity); NOT e2ad8b0 (ACF peak-lag
    tempo — different statistic on same signal); NOT any 2nd-order /
    variance / persistence / cross-scale / trajectory / corr-matrix
    feature; NOT detector post-filter. FIRST onset-strength
    (rectified-spectral-flux) voicing-masked paired-diff in the
    80-feature set.

    Blast radius. Pure features.py change — 1 new block (~35 lines,
    cloned from block 14 template with 1D scalar mean replacing 13-dim
    cosine) + 1 FEATURE_NAMES append + 2 assert bumps (80→81) + 1 call
    in extract_features. Per-t cost: 4 slices + 4 masked means + 2
    subtractions, sub-ms.

(c) IF THIS FAILS. (1) Singing unchanged (within-song drum onset
    intensity is more variable than I'm assuming — dynamic arrangements
    have sparse verses vs loud choruses) → fall back to narrower ±1s
    spans so within-phrase stability dominates. (2) Speech regresses
    (cross-sentence onset rate differs within same recording and feature
    fires on non-splice speech transitions) → add voicing-prob
    threshold `vp > 0.6` to tighten mask. (3) Combined lands at 0.4907
    identical-streak → retrain gate broken for sha bump; escalate.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 62+
    iterations — cannot verify whether the 3 singing FPs sit in sections
    with STABLE drum onset intensity (mechanism assumption) or in
    dynamic verse→chorus onset-density-transition sections (where this
    feature would fire and create a new FP). (ii) SHAP rollup STILL
    empty for 14 keeps — cannot verify whether e2ad8b0's onset-ACF
    feature was even inside top GBM features vs spuriously random. (iii)
    d4d35b1's 0.287 margin-filter catastrophe still undiagnosed; no
    dense p_splice histograms. (iv) 99081f5 4/16 capacity ghost status
    unclear at HEAD.

(e) Wrapper enhancements. Three unchanged highest-priority asks across
    62+ iterations:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE — per-FP (domain,
    file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, onset_strength_pre_mean,
    onset_strength_post_mean, voiced_fraction_pm2s, top-5 |SHAP|).
    Per-FP onset-intensity fields would decide every voicing-masked
    scalar hypothesis data-driven instead of theory-only.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision ("features.py sha Δ XX→YY → retrain" vs "no Δ → skip")
    with joblib-mtime sanity check.

## 2026-04-21T02:45:49+09:00 — 1e57702 (discard, combined=0.470638)
subject: add nf_full_hist_cosine_dist (FEATURE_NAMES 80->81)
per-domain: combined_english=0.911392 combined_korean=0.495652 combined_singing=0.230769

# 2026-04-21 — hypothesis: add nf_full_hist_cosine_dist (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Pure `splice/features.py` add — cosine distance between
    FULL-WINDOW mean 32-bin log-spaced spectral histograms over
    pre[t-2, t] vs post[t, t+2]. Reuses the cached feat_nf_hist
    (32-bin log-frequency magnitude histogram per ~50 ms frame, already
    normalised per-row by _ensure_feat_cache) and feat_nf_times. ZERO
    new librosa calls, ZERO new FFTs, ZERO new caches. Edge guard:
    either window empty or either mean-sum < 1e-10 → sentinel 0.0.
    FEATURE_NAMES 80 → 81 forces wrapper auto-retrain via US-505 sha
    gate.

(b) WHY over recent failures. Last 15+ iterations chased a zoo of
    means-based, variance, persistence, cross-scale, correlation, and
    trajectory features — every one rebuilt its signal from MFCC /
    chroma / spec_contrast / F0 / ZCR / bandwidth / rolloff / onset.
    None of those axes touch the 32-bin FULL LOG-FREQUENCY HISTOGRAM
    directly: the only consumer of feat_nf_hist is _block_noise_floor
    and it aggregates ONLY the bottom-10% quietest frames (noise-floor
    signature), never the full window. That full-window 32-dim signal
    is a genuinely unexplored content axis — GBM max_depth≤4 cannot
    synthesise a 32-dim cosine distance from the 5 scalar summaries
    (spec_centroid / rolloff / flatness / bandwidth / contrast) nor
    from mel-PCA (which collapses time → 20 PCA scalars, discarding
    the per-frame shape).

    Mechanism on 3 singing chord-cycle FPs. Within one song,
    arrangement + mastering chain + mic + room shape the broadband
    32-bin log-mag envelope consistently: drums+bass+keyboards+vocals
    produce a stable log-mag profile across 2 s whose shape (bass
    hump + midrange body + high shelf) barely moves with chord
    transitions — chord pitches shift specific TONAL PEAKS but the
    ensemble ENVELOPE (bin-level average across 32 log bins) is
    stable, cos_dist ≈ 0.02-0.05 silent.

    Real cross-song splice. Different drummer kit + different
    mastering limiter + different vocal bus + possibly different
    codec roll-off (tier-1 Opus vs MP3-128) all shift the 32-bin
    envelope shape materially; cos_dist ≈ 0.15-0.50 fires.

    Speech self-gating. Within a recording the mic + preamp + room +
    speaker vocal tract fix the log-mag envelope; consecutive 2 s
    windows sample identical recording chain → cos_dist small → GBM
    low per-domain SHAP on english/korean. Cross-speaker splice
    flips mic + room + voicing spectrum → cos_dist large → feature
    helps speech TPs too, similar to the self-gating pattern that
    made 1eda8e3 MFCC asymmetry the biggest keep.

    Orthogonal. NOT any cepstral cosine (MFCC), NOT chroma, NOT
    spec_contrast, NOT 1D scalar spectral moment (centroid / rolloff /
    flatness / bandwidth), NOT F0, NOT ZCR, NOT RMS-dB, NOT onset,
    NOT voicing-masked variant, NOT 2nd-order cross-intra, NOT
    F-statistic variance, NOT persistence, NOT cross-scale, NOT
    trajectory velocity, NOT correlation-matrix. NOT existing
    nf_kl_divergence (bottom-10% frames only, KL on 32-bin,
    asymmetric unbounded, maps noise-floor signature — this feature
    uses ALL frames, cosine, bounded [0,2], maps arrangement
    envelope). FIRST full-window 32-bin log-mag histogram cosine
    feature in the 80-feature set; FIRST consumer of feat_nf_hist
    beyond bottom-10% aggregation.

(c) IF THIS FAILS. (1) Singing unchanged — within-song loud-strike
    chord pitches do perturb the full-window 32-bin envelope more
    than I'm assuming → fall back to cosine on ONLY the loudest
    25% frames (arrangement-dominant, smooths tonal-peak jitter).
    (2) Speech regresses — within-recording phoneme sequences shift
    the envelope enough to fire on non-splice positions → restrict
    to frames where nf_rms > chunk-median (active-speech mask).
    (3) Feature fires but zero GBM SHAP — redundant with mel-PCA
    collapse; pivot to Jensen-Shannon divergence on the same
    full-window aggregate (different statistic on same signal,
    bounded like cosine but probability-theoretic).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 63+
    iterations — cannot verify the 3 singing FPs sit in sections
    where the 32-bin envelope is truly stable within-song (mechanism
    assumption); they may be in dynamic verse→chorus arrangement
    transitions where envelope shifts naturally. (ii) SHAP rollup
    STILL empty for 14 keeps — no empirical guidance on which
    content axes GBM actually uses. (iii) No dense p_splice
    histogram logged; d4d35b1 margin-filter catastrophe (0.287)
    remains undiagnosed. (iv) 99081f5 4/16 capacity ghost status
    unclear at HEAD.

(e) Wrapper enhancements. Three unchanged highest-priority asks
    across 63+ iterations:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE — per-FP (domain,
    file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    chunk_duration_s, voiced_unvoiced_mfcc_asymmetry,
    voiced_unvoiced_spec_contrast_asymmetry, nf_kl_divergence,
    nf_centroid_delta, top-5 |SHAP|). Would decide every histogram/
    envelope hypothesis data-driven instead of theory-only.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps; without
    per-feature attribution every axis pick remains a guess.
    (3) RETRAIN-ACTUALLY-FIRED TRACE — wrapper log line at retrain
    decision ("features.py sha Δ XX→YY → retrain" vs "no Δ → skip")
    with joblib-mtime sanity check would isolate the 0.4907
    identical-streak root cause.

## 2026-04-21T02:53:35+09:00 — e2fc9de (discard, combined=0.527473)
subject: tighten DSP_SUM_MIN 5.0 -> 5.5 (linear continuation of 29c06cf 4.5->5.0 keep). Pure detector.py change, no retrain, no feature change, feature count stable at 80. 60+ recent iterations exhausted features.py additions on every content axis (1st-order paired-diff + 2nd-order cross-intra + F-statistic variance + persistence + cross-scale + trajectory velocity + correlation matrix + onset + full histogram), every one collapsing on 3 singing chord-cycle FPs (combined 0.47-0.55). CLAUDE.md mandates structural change after 5+ same-axis failures; features.py axis saturated. DSP_SUM_MIN is genuinely untried primary tunable direction not tracked in wrapper frontier output. 29c06cf kept 4.5->5.0 with mechanism 'bumping bites FPs in [4.5, 5.5] band'. If the 3 remaining singing FPs sit in [5.0, 5.5] band by the same chord-transition-with-partial-support logic, 5.0->5.5 bites them. Real cross-source splices 'disrupt multiple physical signals simultaneously yielding sum 6-12' per 865d92f analysis -- real TPs have >=0.5 margin above 5.5. Linear continuation of a proven productive axis (gradient from 4.5->5.0 keep says keep going); small 10%% bump minimizes TP-drop risk. Orthogonal: NOT features.py addition, NOT classifier retrain, NOT MAX-gate DSP_CONFIRMATION_MIN tuning, NOT GBM_THRESHOLD/GBM_MIN_SEP_S/ANALYSIS_STRIDE_S (frontier-tracked), NOT 60196aa detector peak-width, NOT d4d35b1 p_splice margin (catastrophic 0.287). Risk-bounded by stacking with MAX gate: can only DROP emits past MAX, never create new TPs. Blast radius: 1 constant + comment refresh. Feature set unchanged, classifier byte-identical. Smoke: constants load cleanly, DSP_SUM_MIN=5.5, other tunables unchanged, len(FEATURE_NAMES)=80 stable.
per-domain: combined_english=0.850000 combined_korean=0.531250 combined_singing=0.325000

# 2026-04-21 — hypothesis: tighten DSP_SUM_MIN 5.0 → 5.5

(a) HYPOTHESIS. Pure `splice/detector.py` one-constant change: bump
    `DSP_SUM_MIN` 5.0 → 5.5. Gate is `max(dsp_vals) < DSP_CONFIRMATION_MIN
    or sum(dsp_vals) < DSP_SUM_MIN` where dsp_vals are the three DSP
    z-scores (phase_z / T²_z / CPE_z). Current 5.0 is a KEEP (29c06cf
    bumped 4.5→5.0, combined 0.479→0.488). 5.5 has never been tried.
    Zero retrain, classifier sha stable, feature count stable. Primary
    tunable, instant.

(b) WHY over recent failures. Last 60+ iterations exhausted features.py
    additions on every content axis (1st-order paired-diff + 2nd-order
    cross-intra + F-statistic variance + persistence + cross-scale +
    trajectory velocity + correlation matrix + onset + full histogram).
    Every one collapsed on 3 singing chord-cycle FPs (combined
    0.47-0.55). CLAUDE.md mandates structural change after 5+ same-axis
    failures; features.py axis is saturated. DSP_SUM_MIN is a genuinely
    untried primary tunable direction not tracked in the wrapper's
    frontier output.

    Mechanism on 3 surviving singing FPs. 29c06cf's reasoning said
    4.5→5.0 would bite FPs in [4.5, 5.5] band (chord transitions with
    T²≈3 + partial support). If the 3 remaining FPs sit in [5.0, 5.5]
    band by the same logic, 5.0→5.5 bites them. Real cross-source
    splices "disrupt multiple physical signals simultaneously yielding
    sum 6-12" per 865d92f analysis — real TPs have ≥1 margin above 5.5.

    Linear continuation of a proven productive axis (4.5→5.0 kept,
    gradient says keep going). Small step (10% bump) minimizes TP-drop
    risk. Risk-bounded by stacking with MAX gate: can only DROP emits,
    never create new TPs.

    Orthogonal to every recent attempt. NOT a features.py addition;
    NOT a classifier retrain; NOT MAX-gate tuning; NOT GBM_THRESHOLD
    / GBM_MIN_SEP_S / ANALYSIS_STRIDE_S (frontier-tracked axes, all
    three saturated); NOT detector peak-width (60196aa failed); NOT
    p_splice margin (d4d35b1 catastrophic 0.287). First DSP_SUM_MIN
    tick beyond 5.0.

    Blast radius. detector.py only, 1 constant + comment refresh.
    Feature set unchanged, classifier byte-identical, no retrain.
    Per-emit cost unchanged (same sum+compare path). Wrapper auto-
    retrain gate passive (features.py sha stable).

(c) IF THIS FAILS. (1) No change (every FP has sum > 5.5 — 5.0→5.5
    bite zone empty) → continue to SUM 6.0 or shift to raising MAX
    gate DSP_CONFIRMATION_MIN 2.0→2.5 (also untried, 3-FP band may sit
    at MAX≈2.0-2.5). (2) Real TPs regress (borderline weak-DSP
    crossfades with sum 5.0-5.5) → revert to 5.0 and pivot to MAX gate
    tightening. (3) Singing improves but english/korean regress
    (clean_fp already 0, so this means splice_f1 drop from real-TP loss)
    → revert and try class-specific DSP thresholds.

(d) Information gaps. (i) CLEAN_FP_POSITIONS still absent after 63+
    iterations — cannot verify the 3 singing FPs' actual DSP sum
    values (5.0-5.5 bite zone vs 6+ safe zone); this remains a theory
    bet. Would turn every DSP-gate tunable into a data-driven choice.
    (ii) SHAP rollup still empty for 14 keeps. (iii) Wrapper tunable
    frontier tracks GBM_THRESHOLD / GBM_MIN_SEP_S / ANALYSIS_STRIDE_S
    but NOT DSP_SUM_MIN / DSP_CONFIRMATION_MIN — easy to lose track of
    this untried axis.

(e) Wrapper enhancements. Three unchanged highest-priority asks across
    63+ iterations:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE per-FP (domain, file,
    t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_sum,
    voiced_unvoiced_mfcc_asymmetry, top-5 |SHAP|). Would make DSP_SUM
    bite-zone vs safe-zone directly observable.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) ADD DSP_SUM_MIN / DSP_CONFIRMATION_MIN to the tunable frontier
    snapshot alongside GBM_THRESHOLD / GBM_MIN_SEP_S /
    ANALYSIS_STRIDE_S. Not tracking these gates means the agent has to
    git-log grep to know the axis state.

## 2026-04-21T03:02:19+09:00 — 0ae9231 (discard, combined=0.426339)
subject: add DSP_SECOND_HIGHEST_MIN=1.5 multi-channel-consensus gate (detector.py only; pure primary tunable, no retrain)
per-domain: combined_english=0.701299 combined_korean=0.433333 combined_singing=0.255000

# 2026-04-21 — hypothesis: add DSP_SECOND_HIGHEST_MIN=1.5 gate to detector.py

(a) HYPOTHESIS. Pure `splice/detector.py` one-line addition of a THIRD
    DSP-confirmation gate stacked on top of MAX (2.0) and SUM (5.0).
    Require the SECOND-HIGHEST of the three DSP z-scores
    (phase_z / t2_z / cpe_z) to be >= 1.5. Equivalently: at least TWO
    of the three DSP channels must fire >= 1.5 simultaneously.
    Pseudocode:
        sorted_dsp = sorted(dsp_vals, reverse=True)
        if max < DSP_CONFIRMATION_MIN
           or sum < DSP_SUM_MIN
           or sorted_dsp[1] < DSP_SECOND_HIGHEST_MIN:  # NEW
            skip
    No retrain, no feature change, feature count stable at 80.
    Classifier sha identical.

(b) WHY over recent failures. Last 60+ iterations exhausted features.py
    additions on every content axis. Detector-side DSP_SUM_MIN
    tightening just failed (e2fc9de 5.0->5.5 regressed all three domains
    0.589->0.527), which tells us the 3 singing FPs have SUM >= 5.5 —
    tightening SUM further just drops real TPs. But SUM is SYMMETRIC in
    channels: [4.0, 0.7, 0.3] sums to 5.0 and passes; [2.5, 1.3, 1.2]
    sums to 5.0 and passes. SECOND-HIGHEST specifically rejects "one
    loud + two quiet" configurations that SUM tolerates. This is a
    genuinely new axis inside the DSP-gate family — never tried
    (confirmed via git log grep second/top-2/quorum/two-of-three).
    29c06cf kept 4.5->5.0 proving the DSP-gate axis is productive when
    the right dimension is picked.

    Mechanism on 3 surviving singing chord-cycle FPs. Chord transitions
    typically fire ONE dominant DSP channel: either phase_z (phase
    discontinuity at chord boundary, abrupt harmonic realignment) OR
    t2_z (spectral-distribution shift across chord voices) with the
    others WEAK. Pattern [phase=2.5, t2=1.5, cpe=1.2] — max=2.5
    passes, sum=5.2 passes, 2nd=1.5 borderline; [phase=3.0, t2=1.3,
    cpe=0.8] — max=3.0 passes, sum=5.1 passes, 2nd=1.3 < 1.5 BLOCKS.
    1.5 threshold surgically bites the single-dominant-channel regime
    where other two channels remain near the chunk-local noise floor.

    Real cross-song splice. Disrupts MULTIPLE physical signals
    simultaneously — mic-impulse-response change drives phase_z, new
    spectral distribution drives t2_z, new noise-floor + compression
    dynamics drive cpe_z. Typical pattern [2.5, 2.2, 1.8] or
    [3.0, 2.5, 2.0] — second-highest usually >= 2.0 comfortably above
    1.5 threshold, margin >= 0.5.

    Why 1.5 (not 2.0). Conservative start — bites [2.5, 1.3, 1.2] and
    [4.0, 0.7, 0.3] single-channel-dominant patterns while preserving
    [2.0, 1.8, 1.5] three-channel-balanced patterns. Threshold 2.0
    would also block [2.5, 1.8, 1.5] configurations that may include
    real crossfade TPs where one channel is moderate. 1.5 is
    minimal-risk.

    Orthogonal. NOT 273b8f5 MAX gate (absolute highest channel); NOT
    865d92f/29c06cf/e2fc9de SUM gate (symmetric cumulative); NOT
    d256901 class-specific routing; NOT 60196aa peak-width neighbor-
    support; NOT d4d35b1 local-margin filter; NOT 7255ec6 GBM_THRESHOLD;
    NOT any GBM_MIN_SEP_S/ANALYSIS_STRIDE_S tweak; NOT features.py
    addition; NOT classifier hyperparameter. FIRST "multi-channel
    consensus" gate requiring simultaneous activation of >=2 DSP
    signals above a shared floor.

    Blast radius. detector.py only. 1 new constant + 1 sorted() call
    + 1 compare inside the existing gate loop at line 305. Per-emit
    cost: 1 extra 3-element sort + 1 float compare, negligible.
    Feature set unchanged, classifier byte-identical, no retrain.

(c) IF THIS FAILS. (1) Singing unchanged (the 3 FPs happen to have
    [2.5, 2.0, 1.5]+ — balanced, not single-dominant) → raise threshold
    to 1.8 for tighter consensus demand. (2) English/korean regress
    (real speech TPs have asymmetric DSP: e.g., crossfade TPs with
    strong t2_z but weak phase_z/cpe_z) → revert and split into
    class-specific DSP configurations (relax second-highest for
    class 2 crossfades, keep for class 1 hard cuts). (3) Combined
    matches 0.4907 identical-streak → wrapper joblib cache issue;
    escalate.

(d) Information gaps. (i) CLEAN_FP_POSITIONS JSON block STILL absent
    after 64+ iterations — cannot verify the 3 singing FPs' actual
    (phase_z, t2_z, cpe_z) triple to know if they're single-channel-
    dominant (my central assumption) or balanced. This hypothesis
    remains a theory bet until FP-position metadata is exposed.
    (ii) SHAP rollup STILL empty for 14 keeps. (iii) Dense p_splice
    histogram per chunk still not logged. (iv) tunable frontier
    snapshot does not track DSP_SECOND_HIGHEST_MIN / DSP_CONFIRMATION_MIN
    / DSP_SUM_MIN — easy to lose this axis state.

(e) Wrapper enhancements. Three unchanged highest-priority requests
    across 64+ iterations:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE — per-FP (domain,
    file, t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_sum,
    dsp_second_highest, voiced_unvoiced_mfcc_asymmetry, top-5 |SHAP|).
    Per-FP DSP triple would let every DSP-gate hypothesis be decided
    data-driven instead of theory-only.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) ADD DSP_SUM_MIN / DSP_CONFIRMATION_MIN / DSP_SECOND_HIGHEST_MIN
    to the tunable frontier snapshot alongside GBM_THRESHOLD /
    GBM_MIN_SEP_S / ANALYSIS_STRIDE_S. Not tracking DSP gates means
    the agent has to git-log grep to know axis state.

## 2026-04-21T03:08:53+09:00 — 9274ace (discard, combined=0.443912)
subject: add DSP_PHASE_MIN=0.3 channel-identity-specific floor on dsp_phase_z
per-domain: combined_english=0.729730 combined_korean=0.516129 combined_singing=0.232258

# 2026-04-21 — hypothesis: add DSP_PHASE_MIN = 0.3 floor (detector.py)

(a) HYPOTHESIS. Pure `splice/detector.py` change — add an individual
    channel floor on `dsp_phase_z` (dsp_vals[0]) stacked onto the
    existing MAX+SUM gates:
        if dsp_vals[0] < DSP_PHASE_MIN: drop
    Threshold DSP_PHASE_MIN = 0.3. No retrain, no feature change,
    feature count stable at 80, classifier sha identical. Wrapper
    auto-retrain stays passive. Per-emit cost: 1 extra compare.

(b) WHY over recent failures. 60+ features.py additions collapsed on
    3 surviving singing chord-cycle FPs (0.47–0.55 range). DSP_SUM
    5.0→5.5 regressed (0.527). DSP_SECOND_HIGHEST=1.5 catastrophically
    regressed (0.426 — symmetric gating asks a question where chord
    transitions can legitimately have two elevated channels). CLAUDE.md
    mandates structural change after 5+ same-axis failures. Individual
    channel-specific floor is an untried axis: prior DSP tunables are
    aggregate (MAX = rank-1, SUM = symmetric, SECOND = rank-2). Never:
    channel-identity-specific.

    Phase_z is the most physically discriminating DSP channel. Within
    one song, mastering chain (mic impulse response + compressor +
    limiter + master bus) is frozen → STFT-edge phase is CONTINUOUS
    across chord transitions → phase_z near 0 (chord shifts spectral
    magnitude, not inter-frame phase alignment). Cross-source splice
    by physical necessity crosses masters → phase discontinuity →
    phase_z elevated (typically 1.5–4.0). T²_z and CPE_z are
    spectral-distribution / voicing statistics that chord transitions
    can mimic (different chord = different spectrum = nontrivial T²).
    Phase is the one channel chord-cycle cannot fake.

    Why 0.3 (not 0.5, not 0.1). Conservative: bites FPs where
    phase_z ∈ [0, 0.3] (silent phase channel, chord-cycle signature)
    while preserving real crossfade TPs whose phase may be smoothed
    but still crosses masters (phase_z > 0.5). 0.5 risks real
    crossfade TPs; 0.1 may not bite any FP.

    Orthogonal. NOT 273b8f5 MAX gate (all-channel maximum); NOT
    865d92f/29c06cf/e2fc9de SUM gate (symmetric cumulative); NOT
    0ae9231 SECOND_HIGHEST (rank-2, symmetric); NOT d4d35b1
    p_splice margin (catastrophic); NOT 60196aa peak-width; NOT
    GBM_THRESHOLD/GBM_MIN_SEP_S/ANALYSIS_STRIDE_S; NOT any features.py
    addition. FIRST channel-identity-specific DSP floor in detector
    history — phase_z is singled out for its unique mimicry-resistance.

    Blast radius. detector.py only, 1 constant + 1 compare in the gate
    loop at line 305. Feature set unchanged, classifier byte-identical,
    no retrain. Per-emit cost: 1 float compare, negligible.

(c) IF THIS FAILS. (1) No change — the 3 FPs already have phase_z > 0.3
    (T²/CPE-dominated but phase still spikes on chord onset) → raise
    floor to 0.5 or try a T²-specific floor instead. (2) Real TPs
    regress — crossfade splices have smoothed phase_z < 0.3 → revert
    and try CPE_z floor (pitch-clarity discontinuity as alternative
    mimicry-resistant channel). (3) Singing improves but english/korean
    regress (real speech splices have lower phase_z than I think) →
    revert and apply channel-specific floors ONLY to singing class via
    class-specific routing.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 65+
    iterations — cannot verify the 3 singing FPs' (phase_z, t2_z, cpe_z)
    triple and whether phase_z is actually < 0.3 on them. This remains
    a theory bet. Biggest single gap. (ii) SHAP rollup STILL empty for
    14 keeps. (iii) Tunable frontier snapshot does NOT track
    DSP_CONFIRMATION_MIN / DSP_SUM_MIN / DSP_PHASE_MIN — easy to lose
    state of untried DSP-gate axes.

(e) Wrapper enhancements. Three unchanged highest-priority requests
    across 65+ iterations:
    (1) CLEAN_FP_POSITIONS JSON in CURRENT STATE — per-FP (domain, file,
    t_sec, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_sum,
    dsp_max, dsp_min, top-5 |SHAP|). Per-FP DSP triple would turn every
    DSP-gate hypothesis into a data-driven decision. PRIORITY 1.
    (2) SHAP ROLLUP REPAIR — rollup empty for 14 keeps.
    (3) EXPAND TUNABLE FRONTIER SNAPSHOT to include DSP_CONFIRMATION_MIN,
    DSP_SUM_MIN, DSP_PHASE_MIN (new), and per-channel minimums alongside
    GBM_THRESHOLD / GBM_MIN_SEP_S / ANALYSIS_STRIDE_S. Not tracking DSP
    gates means the agent has to git-log grep to know axis state.

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

