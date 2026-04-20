## Historical digest (oldest 20 entries compacted)
Span: Historical digest (oldest 20 entries compacted) … 2026-04-20T03:35:28+09:00 — 308aa5a (discard, combined=0.522259)
Outcomes: 5 keep / 10 discard / 2 verify-fail; combined range [0.4486, 0.5323].
(Older reflections collapsed to conserve prompt budget.)

## 2026-04-20T03:50:25+09:00 — df6fc0a (discard, combined=0.522818)
subject: add unvoiced_mfcc_cosine_dist feature (FEATURE_NAMES 77->78) — MIRROR of 49bd0b1's voiced-mask MFCC on the complementary frame set. Per position t_sec slice feat_vp + feat_mfcc on pre=[t-2,t] and post=[t,t+2]; keep frames where vp==0.0 (UNVOICED); return cosine distance of unvoiced-only mean 13-dim MFCC vectors. Sentinel 0.0 when either window has zero unvoiced frames or zero-norm vectors, matching 49bd0b1/32cac36 safety pattern. Zero new caches, zero new librosa calls — pure slice + mask flip + mean + cosine, sub-ms per t. Targets 32cac36 current-keep singing 0.341 weakest-domain plateau where voiced-mask features (49bd0b1 voiced_mfcc + 32cac36 voiced_chroma) have saturated the timbre-continuity axis but are actively REDUCING singing recall because real singing splices are often SAME-SINGER-DIFFERENT-SONG and voiced_mfcc stays similar across them. Accompaniment (drums/bass/mastering) is what cross-song splices consistently disrupt, and unvoiced frames in singing carry exactly that signal — between-phrase breath intakes, sibilants, vocal-gap windows where accompaniment dominates. Same-song accompaniment stable giving cosine distance ~0.05-0.15; cross-song accompaniment shifts giving ~0.30-0.60. Self-gating on speech comes free from the mask flip: unvoiced speech frames = consonants (plosive/fricative/silence) whose MFCC varies radically per phoneme regardless of splice -> high-variance noise -> GBM assigns low per-domain SHAP on english/korean. Orthogonal to every prior axis: NOT 49bd0b1 (INVERSE mask — complementary frame set; voiced captures vocal-tract, unvoiced captures accompaniment); NOT 32cac36 voiced_chroma (pitch-class not cepstral); NOT 308aa5a voiced_tonnetz; NOT ff65865 voiced_chroma_far (different time scale); NOT remote-half MFCC 438080/411473 (long half-chunk averages dominated by file-mean character — this uses LOCAL ±2s UNVOICED-RESTRICTED mean). First feature targeting accompaniment continuity directly. Pure features.py change; FEATURE_NAMES count gate triggers wrapper auto-retrain via US-505. Smoke: import + assert len(FEATURE_NAMES)==78 + last name 'unvoiced_mfcc_cosine_dist' + pure-sine/noise synthetic confirms sentinel 0.0 on zero-unvoiced-frame pre window (correct mask logic).
per-domain: combined_english=0.814815 combined_korean=0.555385 combined_singing=0.315789

# 2026-04-20 — hypothesis: unvoiced_mfcc_cosine_dist feature (FEATURE_NAMES 77→78)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE feature
    `unvoiced_mfcc_cosine_dist`, the MIRROR of 49bd0b1's voiced-mask MFCC.
    Per position t_sec: slice `feat_vp` and `feat_mfcc` on pre=[t−2, t]
    and post=[t, t+2]; keep frames where `vp == 0.0` (UNVOICED); return
    cosine distance of unvoiced-only mean 13-dim MFCC vectors. Sentinel
    0.0 when either window has zero unvoiced frames or zero-norm vectors,
    matching 49bd0b1/32cac36 safety pattern. Zero new caches, zero new
    librosa calls — pure slice + mask flip + mean + cosine, sub-ms per t.
    FEATURE_NAMES 77→78 triggers wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. 32cac36 (current keep, 0.532) has
    singing STILL weakest at 0.341. Post-32cac36 feature attempts on the
    voiced-mask axis have stalled: ff65865 voiced_chroma_far (english
    regressed 0.840→0.684 — far-window speech-vowel chroma is noisy,
    breaks self-gating); 308aa5a voiced_tonnetz (english 0.850 good,
    but singing stuck 0.341 and korean regressed slightly). Root cause
    of the singing plateau: 49bd0b1's voiced_mfcc signal measures
    VOCAL-TRACT continuity — real singing splices are often
    SAME-SINGER-DIFFERENT-SONG (common in evaluation corpora), so
    voiced_mfcc stays similar across such splices and LOWERS p_splice
    on real TPs. Voiced-mask features have saturated the "timbre
    continuity" axis for english/korean (speech splices are
    different-speaker) but are actively REDUCING recall on singing.
    What singing splices DO disrupt consistently is ACCOMPANIMENT
    (drums/bass/production) — and unvoiced frames carry exactly that
    signal. In a singing chunk, unvoiced frames are gaps between
    vocal phrases, breath intakes, sibilants — windows where
    accompaniment dominates. Within one song the accompaniment is
    stable (same drum kit, same mastering, same bass line) → unvoiced
    MFCC cosine distance ~0.05-0.15. Cross-song splice: different
    drum kit / different mastering / different tempo bass → unvoiced
    MFCC cosine distance ~0.30-0.60.

    Self-gating on speech is strong and comes free from the mask flip:
    unvoiced speech frames = consonants (plosive/fricative/silence)
    are EXTREMELY phoneme-dependent. Consonant MFCC varies radically
    across [s, p, t, k, silence] regardless of splice, so
    unvoiced_mfcc distance is high-variance noise on speech domains.
    GBM tree splits will assign low per-domain SHAP on english/korean
    (same mechanism that made 49bd0b1 invisible on singing), leaving
    the feature functionally active only on singing.

    Orthogonal to every prior axis: NOT 49bd0b1 (INVERSE mask —
    complementary frame set; voiced captures vocal-tract, unvoiced
    captures accompaniment); NOT 32cac36 (pitch-class not cepstral);
    NOT 308aa5a tonnetz; NOT ff65865 (far-window, different time
    scale); NOT remote-half MFCC (438080/411473 — used long
    half-chunk averages dominated by file-mean character; this uses
    LOCAL ±2s UNVOICED-RESTRICTED mean). First feature targeting
    accompaniment continuity directly. Blast radius: features.py
    only — one new block + FEATURE_NAMES append + assert bump. Per-t
    cost: 2 slices + mask flip + mean + cosine on 13-dim vectors,
    sub-ms. Classifier sha auto-invalidates so wrapper retrains
    (~3 min). Risk-bounded: zero-unvoiced sentinel = 0.0 prevents
    NaN; mask-flip guarantees complementary-set behavior so singing
    TPs that voiced_mfcc suppresses (same singer) get RAISED
    p_splice via unvoiced_mfcc firing on accompaniment change — the
    two features work in tandem without stepping on each other.

(c) IF THIS FAILS. (1) If singing has too few unvoiced frames in
    ±2s windows (continuous-vocal performances), combine with an
    RMS-floor mask — a frame counts as "accompaniment-carrying" if
    unvoiced OR has below-median RMS (vocal-quiet moments). (2) If
    GBM assigns ~0 SHAP on singing (unvoiced MFCC noisy even on
    accompaniment), replace MFCC with chroma — unvoiced chroma
    cosine distance captures accompaniment HARMONY content. (3)
    Final escalation: combine voiced AND unvoiced into a single
    `accompaniment_mfcc_asymmetry` feature =
    `unvoiced_mfcc_cosine_dist − voiced_mfcc_cosine_dist`, encoding
    "accompaniment changed more than voice" (positive =
    same-singer-different-song splice; negative = intra-song
    phoneme variation).

(d) Information gaps. (i) Per-position CLEAN_FP_POSITIONS still not
    in CURRENT STATE after 16+ consecutive hypotheses — cannot verify
    whether the 2 surviving singing FPs have unvoiced frames in
    both pre/post ±2s windows (continuous-vocal phrases → empty mask
    → feature is no-op). (ii) SHAP rollup reports "no keeps yet —
    rollup empty" so the theoretical claim "voiced_mfcc is
    over-suppressing same-singer splice TPs" is untestable. (iii)
    Per-file unvoiced-frame-fraction distribution per domain is
    unknown — on singing it's theoretical that accompaniment-only
    windows exist within ±2s of real splice positions.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE — persistent blocker for 16+ hypotheses; per-FP
    (domain, file, t_sec, label_id, p_splice, voiced_mfcc_dist,
    vp_pre_frac, vp_post_frac, top-5 |SHAP| features with values)
    would flip mask design from theory to data. (2)
    `scripts/feature_oof_preview.py --add <feature_fn>` that trains
    once and reports per-domain OOF-F1 delta — turns "is this
    feature worth a 3-min retrain" into a numeric. (3)
    LOST_TP_POSITIONS between consecutive keeps — per-file list of
    real-splice positions detected at commit N-1 but lost at commit
    N. Would directly confirm/deny the "voiced_mfcc suppresses
    same-singer singing TPs" hypothesis driving THIS iteration.

## 2026-04-20T09:52:06+09:00 — 1eda8e3 (keep, combined=0.586244)
subject: add voiced_unvoiced_mfcc_asymmetry feature (FEATURE_NAMES 78->79) — signed asymmetry unvoiced_mfcc_cosine_dist minus voiced_mfcc_cosine_dist on ±2s pre/post window. Positive = accompaniment changed more than voice (same-singer cross-song splice signature); ~0 on intra-song chord cycle (both low) and different-singer cross-song (both high, cancel). Reuses cached feat_mfcc + feat_vp — ZERO new librosa calls, ZERO new caches. Per-t cost is 4 slices + 4 masked means + 2 cosines on 13-dim vectors, sub-ms. Sentinel 0.0 when either mask is empty or either sub-norm underflows, matching 49bd0b1/32cac36 safety pattern. Targets 32cac36 current-keep (0.532261) singing 0.341 weakest-domain plateau where five consecutive voiced-mask single-mask variants have failed: ff65865 voiced_chroma_far (english 0.840->0.684), 308aa5a voiced_tonnetz (stuck 0.522), df6fc0a unvoiced_mfcc (singing REGRESSED 0.341->0.316), e7ca9eb voiced_spec_contrast (singing REGRESSED 0.341->0.316). Common failure mode: single-mask distance features with signal too noisy on at least one domain and GBM globally mis-weights via tree splits. The df6fc0a unvoiced_mfcc fail is directly instructive — unvoiced MFCC carries the accompaniment-change signal on singing (drums/bass/mastering dominate unvoiced gaps + sibilants) BUT its INDEPENDENT value varies wildly on speech consonants [s,p,t,k,silence], so GBM over-weighted it globally and singing tanked. The fix is NOT to reintroduce unvoiced_mfcc as an independent feature. It is to add the DIFFERENCE against voiced_mfcc as a SINGLE feature — PAIRED DIFFERENCING cancels phoneme-correlated noise on speech (both voiced and unvoiced speech MFCC vary together with phoneme context / speaker identity / recording conditions so their difference is ~zero-mean noise), while preserving singing's same-singer-cross-song signature (voiced~0.05 unvoiced~0.30 asymmetry~+0.25). Intra-song chord cycle: voiced~0.05 unvoiced~0.10 asymmetry~+0.05 SMALL. Explicit escalation from df6fc0a(c)(3): "combine voiced AND unvoiced into a single accompaniment_mfcc_asymmetry feature = unvoiced_mfcc_cosine_dist - voiced_mfcc_cosine_dist". GBM max_depth=3 CANNOT synthesize this interaction from voiced_mfcc alone post-df6fc0a revert (unvoiced_mfcc isn't in feature set); providing the subtraction surfaces the pattern directly. Orthogonal to every prior axis: NOT 49bd0b1 voiced_mfcc (single-mask no differential); NOT df6fc0a unvoiced_mfcc (single-mask, tanked singing via GBM global over-weighting); NOT 32cac36 voiced_chroma (pitch-class not cepstral); NOT 308aa5a tonnetz; NOT ff65865 far-window; NOT e7ca9eb voiced_spec_contrast (spectral-contrast not cepstral AND not asymmetric). First ASYMMETRIC/DIFFERENTIAL feature in the voiced-mask axis. Pure features.py change; FEATURE_NAMES count gate triggers wrapper auto-retrain via US-505 sha gate. Smoke-verified: len(FEATURE_NAMES)==79, last name 'voiced_unvoiced_mfcc_asymmetry', synthetic unvoiced-shifted yields +0.39 positive, voiced-shifted yields -0.13 negative, empty-mask yields sentinel 0.0, end-to-end extract_features returns 79 finite features on synthetic audio.
per-domain: combined_english=0.888889 combined_korean=0.666667 combined_singing=0.340000

# 2026-04-20 — hypothesis: voiced_unvoiced_mfcc_asymmetry feature (FEATURE_NAMES 78→79)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new feature
    `voiced_unvoiced_mfcc_asymmetry`. Per position t_sec compute in a single
    block: (1) voiced_mfcc_dist = cosine distance between VOICED-mean 13-dim
    MFCC on pre=[t−2, t] vs post=[t, t+2]; (2) unvoiced_mfcc_dist = same on
    UNVOICED frames (vp==0.0); (3) return `unvoiced_mfcc_dist −
    voiced_mfcc_dist` (SIGNED asymmetry; positive = accompaniment changed
    more than voice). Reuses feat_mfcc + feat_vp (ZERO new librosa calls,
    ZERO new caches). Sentinel 0.0 when either mask is empty or either
    sub-norm underflows, matching 49bd0b1 / 32cac36 safety pattern.
    FEATURE_NAMES 78→79 triggers wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. Five consecutive voiced-mask variants on
    cached features have failed since 32cac36 kept voiced_chroma: ff65865
    voiced_chroma_far (english 0.840→0.684), 308aa5a voiced_tonnetz (stuck
    0.522), df6fc0a unvoiced_mfcc (singing REGRESSED 0.341→0.316), e7ca9eb
    voiced_spec_contrast (singing 0.341→0.316 REGRESSED again). Common
    failure mode: SINGLE-MASK distance features where the signal is noisy
    on at least one domain and GBM globally mis-weights it via tree splits.
    The df6fc0a unvoiced_mfcc fail is directly instructive: unvoiced MFCC
    carries the exact accompaniment-change signal I want on singing
    (drums/bass/mastering live in unvoiced gaps and sibilants) BUT its
    INDEPENDENT value varies wildly on speech consonants [s, p, t, k,
    silence], so GBM over-weighted it globally and singing tanked.  The
    fix is NOT to reintroduce unvoiced_mfcc as an independent feature. It
    is to add the DIFFERENCE against voiced_mfcc as a SINGLE feature.
    PAIRED DIFFERENCING cancels most phoneme-correlated noise on speech —
    both voiced and unvoiced speech MFCC vary TOGETHER with phoneme
    context / speaker identity / recording conditions, so their difference
    is ~zero-mean noise. On singing the same-singer-cross-song pattern
    STAYS: voice timbre same (voiced_mfcc ~0.05) BUT accompaniment shifts
    (unvoiced_mfcc ~0.30), so asymmetry ~+0.25 LARGE POSITIVE; chord
    cycle-back intra-song: voiced ~0.05, unvoiced ~0.10, asymmetry ~+0.05
    SMALL. Explicit cited escalation from df6fc0a(c)(3): "combine voiced
    AND unvoiced into a single accompaniment_mfcc_asymmetry feature =
    unvoiced_mfcc_cosine_dist − voiced_mfcc_cosine_dist, encoding
    'accompaniment changed more than voice' (positive = same-singer-
    different-song splice; negative = intra-song phoneme variation)."
    GBM max_depth=3 CANNOT synthesize this interaction from voiced_mfcc
    alone (the tree would need to split on unvoiced_mfcc which isn't in
    the post-df6fc0a feature set) — providing the subtraction surfaces
    the pattern directly. Orthogonal to every prior axis: NOT 49bd0b1
    voiced_mfcc (single-mask, no differential); NOT df6fc0a unvoiced_mfcc
    (single-mask, tanked singing via GBM global over-weighting); NOT
    32cac36 voiced_chroma (pitch-class not cepstral); NOT 308aa5a tonnetz;
    NOT ff65865 far-window; NOT e7ca9eb voiced_spec_contrast (spectral
    contrast not cepstral AND not asymmetric). First ASYMMETRIC /
    DIFFERENTIAL feature in the voiced-mask axis — genuinely new mechanism.
    Blast radius: features.py only — 1 block + 1 FEATURE_NAMES append +
    assert bump. Per-t cost: 4 slices + 4 masked means + 2 cosines on
    13-dim vectors, sub-ms. Zero new caches. Risk-bounded: both
    sub-distances have proven sentinel-0.0 safety; the subtraction
    retains 0.0 under any sentinel; paired differencing is inherently
    self-cancelling on domains where both masks carry similar
    phoneme-correlated noise.

(c) IF THIS FAILS. (1) If GBM still assigns high SHAP on english/korean
    (paired differencing doesn't fully cancel because the voiced and
    unvoiced frame subsets hit DIFFERENT 40-ms frames so phoneme noise
    doesn't perfectly co-vary), CLIP to nonnegative:
    `max(0, unvoiced_dist − voiced_dist)` — accompaniment-change-only,
    zero on speech where the sign of the diff is random-noise. (2) If
    singing TPs still regress (real DIFFERENT-singer cross-song splices
    have voiced≈0.3 and unvoiced≈0.3 → asymmetry ≈ 0 so the feature
    goes silent there too, while voiced_mfcc alone would have fired),
    replace with a RATIO: `unvoiced_dist / max(voiced_dist, 0.05)` —
    the ratio stays >1 even when both distances are large, as long as
    unvoiced shifts more. (3) Final escalation: add a second feature
    `voiced_unvoiced_chroma_asymmetry` (same mechanism on the chroma
    axis) so GBM sees the accompaniment-vs-voice asymmetry
    simultaneously in cepstral and pitch-class space — redundant enough
    to bite through either feature's per-domain noise.

(d) Information gaps. (i) CLEAN_FP_POSITIONS still absent after 17+
    iterations — I cannot verify whether 32cac36's 2 surviving singing
    FPs have BOTH voiced and unvoiced frames in ±2s windows. Continuous
    sung phrases may have zero unvoiced frames so the asymmetry
    collapses to sentinel 0.0 on those FPs and the feature is a no-op
    for exactly the singing population I'm targeting. (ii) SHAP rollup
    still "no keeps yet — rollup empty" — the theoretical claim
    "unvoiced_mfcc tanked singing via GBM global over-weighting" is
    unverifiable against the actual SHAP distribution on df6fc0a. (iii)
    The in-tree features.py has 78 features (voiced_spec_contrast from
    e7ca9eb kept in code even though listed as a RECENT FAILED
    HYPOTHESIS and absent from baseline 32cac36 at 77 features) — the
    wrapper appears to have failed to revert features.py on the
    voiced_spec_contrast discard. My asymmetry adds as the 79th; the
    wrapper retrains on 79 features, so eval is internally consistent,
    but attribution of the combined delta will mix my asymmetry's lift
    with voiced_spec_contrast's drag, confounding the signal.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent blocker for 17+ consecutive iterations; per-FP
    (domain, file, t_sec, label_id, p_splice, vp_pre_frac, vp_post_frac,
    voiced_mfcc_dist, unvoiced_mfcc_dist, top-5 |SHAP| features with
    values). Every mask-based feature hypothesis has been calibrated
    from theory; this single block flips the entire loop to data-driven.
    (2) `scripts/feature_oof_preview.py --add <feature_fn>` that
    retrains once and reports per-domain OOF-F1 delta vs current —
    turns "is this feature worth a 3-min retrain cycle" from a bet into
    a numeric preview. (3) DISCARD-REVERT SYNC auditor: the wrapper's
    discard path should diff features.py vs the baseline-sha features.py
    and ABORT / restore if they differ, so a discarded hypothesis's
    code can never silently persist into the next iteration's classifier.
    Currently e7ca9eb's voiced_spec_contrast block is still in-tree
    despite the feature being listed as a failure, which confounds
    every subsequent hypothesis.
[auto] (no SHAP data for either 32cac36 or 1eda8e3)

## 2026-04-20T10:06:22+09:00 — f4148cc (discard, combined=0.582332)
subject: add voiced_unvoiced_chroma_asymmetry feature (FEATURE_NAMES 79->80) — pitch-class companion to 1eda8e3's voiced_unvoiced_mfcc_asymmetry. Signed asymmetry unvoiced_chroma_cosine_dist minus voiced_chroma_cosine_dist on +-2s pre/post window. Positive = accompaniment key/chord shifted more than voice (same-singer cross-song splice signature on the HARMONIC axis, orthogonal to the MFCC/timbre asymmetry which leaves chord-cycle intra-song undiscriminated). ~0 on intra-song chord cycle (both voiced and unvoiced chroma shift together with progression) and different-singer cross-song (both high and cancel). Reuses cached feat_chroma (32cac36) + feat_vp — ZERO new librosa calls, ZERO new caches. Per-t cost 4 slices + 4 masked means + 2 cosines on 12-dim vectors, sub-ms. Sentinel 0.0 when either mask empty or either sub-norm underflows, matching 1eda8e3/49bd0b1/32cac36 safety pattern. Targets 1eda8e3 current-keep (0.586244) singing 0.340 weakest-domain plateau where MFCC asymmetry lifted english 0.840->0.889 and korean 0.526->0.667 spectacularly but singing stayed FLAT. The MFCC asymmetry mechanism worked on speech domains (different speakers across splice = different voiced+unvoiced MFCC) but not on singing where real cross-song splices often land on a held note (voiced_mfcc_dist low, unvoiced_mfcc_dist high -> asymmetry positive, BUT this case is rare enough or GBM assigns SHAP modestly). Chroma asymmetry attacks the HARMONIC axis directly: same-singer holding note E across a cut voiced_chroma_dist ~0.05, accompaniment shifted to new key unvoiced_chroma_dist ~0.30-0.50, asymmetry ~+0.25 LARGE. Intra-song chord cycle: voiced and unvoiced chroma BOTH shift together with chord progression (harmony applies to whole mix) so asymmetry ~0, NOT confused with splice. Paired differencing on speech: voiced and unvoiced chroma co-vary with prosodic phoneme context so difference is zero-mean noise, GBM assigns low per-domain SHAP on english/korean (neutralizes the 634cdd2 raw-chroma verify-fail mode). Explicit escalation from 1eda8e3(c)(3): 'add voiced_unvoiced_chroma_asymmetry on the chroma axis so GBM sees accompaniment-vs-voice asymmetry simultaneously in cepstral and pitch-class space — redundant enough to bite through either feature's per-domain noise'. GBM max_depth=3 CANNOT synthesize this interaction from voiced_chroma_cosine_dist alone (would need unvoiced_chroma which isn't in feature set); providing subtraction surfaces pattern directly. Orthogonal to every prior axis: NOT 1eda8e3 (cepstral not pitch-class); NOT 32cac36 voiced_chroma (single-mask no differential); NOT 634cdd2 raw chroma_cosine_dist (no mask no asymmetry); NOT 68004ca tonality-gated chroma (external max/mean ratio); NOT 9fe41d1 chroma persistence_far (4-8s post-filter); NOT ff65865 voiced_chroma_far (far window); NOT 308aa5a tonnetz (linear projection); NOT df6fc0a unvoiced_mfcc (cepstral single-mask); NOT e7ca9eb voiced_spec_contrast (mastering signature not pitch-class). First ASYMMETRIC/DIFFERENTIAL feature in the pitch-class axis. Pure features.py change; FEATURE_NAMES count gate triggers wrapper auto-retrain via US-505 sha gate. Smoke-verified: len(FEATURE_NAMES)==80, last name 'voiced_unvoiced_chroma_asymmetry', end-to-end extract_features returns 80 finite features on synthetic audio, sentinel 0.0 triggers correctly when voicing-mask empty (pure synthetic signals).
per-domain: combined_english=0.888889 combined_korean=0.696970 combined_singing=0.318750

# 2026-04-20 — hypothesis: voiced_unvoiced_chroma_asymmetry feature (FEATURE_NAMES 79→80)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new feature
    `voiced_unvoiced_chroma_asymmetry`, the PITCH-CLASS companion to the
    just-kept `voiced_unvoiced_mfcc_asymmetry` (1eda8e3). Per position t_sec
    compute (1) voiced_chroma_dist = cosine distance between VOICED-mean
    12-dim chroma on pre=[t−2, t] vs post=[t, t+2]; (2) unvoiced_chroma_dist
    = same on UNVOICED frames (vp==0.0); (3) return
    `unvoiced_chroma_dist − voiced_chroma_dist` (SIGNED asymmetry; positive
    = accompaniment key/chord shifted more than voice). Reuses feat_chroma
    + feat_vp (ZERO new librosa calls, ZERO new caches). Sentinel 0.0 when
    either mask is empty or either sub-norm underflows, matching
    1eda8e3/49bd0b1/32cac36 safety pattern. FEATURE_NAMES 79→80 triggers
    wrapper auto-retrain via US-505 sha gate.

(b) WHY this over the current state. 1eda8e3 (current keep, combined=
    0.586244) MFCC-space asymmetry lifted english 0.840→0.889 and korean
    0.526→0.667 spectacularly but singing stayed FLAT at 0.340 (was 0.341)
    — still the weakest domain by a wide margin. The asymmetry mechanism
    worked; it just didn't bite on singing. Explicit cited escalation from
    1eda8e3(c)(3): "add a second feature voiced_unvoiced_chroma_asymmetry
    (same mechanism on the chroma axis) so GBM sees the accompaniment-vs-
    voice asymmetry simultaneously in cepstral and pitch-class space —
    redundant enough to bite through either feature's per-domain noise."
    Mechanism on singing: real cross-song splice — voice pitch class
    preserved if singer holds a note across cut (voiced_chroma_dist low
    ~0.05) but accompaniment shifts to new key/chord progression
    (unvoiced_chroma_dist high ~0.30-0.50) → asymmetry +0.25 LARGE
    POSITIVE (same sign as MFCC asymmetry but on the ORTHOGONAL harmonic
    axis). Intra-song chord cycle — both voiced and unvoiced chroma shift
    together with the chord progression (harmony applies to everything in
    the mix) → asymmetry ~0 SMALL. MFCC asymmetry cannot see this case
    because MFCC is timbre — chord cycle leaves timbre stable on BOTH
    masks. Chroma asymmetry sees it clearly as a signed value. Paired
    differencing on speech: both voiced and unvoiced chroma vary TOGETHER
    with phoneme / prosody context (consonant and vowel land in similar
    pitch-class regions within a speaker's prosodic window), so the
    difference is ~zero-mean noise → GBM assigns low per-domain SHAP on
    english/korean (the 634cdd2 verify-fail failure mode for raw-chroma
    is directly neutralized by paired-differencing cancellation).
    Orthogonal to every prior axis: NOT 1eda8e3 (cepstral, not pitch-
    class); NOT 32cac36 voiced_chroma (single-mask, no differential); NOT
    634cdd2 raw chroma_cosine_dist (no mask, no asymmetry); NOT 68004ca
    tonality-gated chroma (external max/mean ratio gate); NOT 9fe41d1
    chroma persistence_far (post-filter, 4-8s window); NOT ff65865
    voiced_chroma_far (far window); NOT 308aa5a tonnetz (linear
    projection); NOT df6fc0a unvoiced_mfcc (single-mask cepstral, not
    asymmetric). Blast radius: features.py only — 1 new block + 1
    FEATURE_NAMES append + 2 assert bumps. Zero new caches. Per-t cost:
    4 slices + 4 masked means + 2 cosines on 12-dim vectors, sub-ms.
    Risk-bounded: sentinel-0.0 safety matches proven 1eda8e3 pattern;
    paired differencing is inherently self-cancelling on speech.

(c) IF THIS FAILS. (1) If singing has too few unvoiced frames in ±2s
    around real splices (continuous sung phrases → empty unvoiced mask
    → feature is no-op on exactly the singing population I'm targeting),
    combine with an RMS-floor mask (frame counts as "accompaniment-
    carrying" if unvoiced OR below-median RMS). (2) If GBM assigns ~0
    SHAP (paired differencing cancels too much signal, or the 79-feature
    baseline already fills the harmonic-accompaniment niche via
    voiced_chroma_cosine_dist), replace with a RATIO: `unvoiced_chroma_dist
    / max(voiced_chroma_dist, 0.05)` — preserves magnitude when both
    distances are large but unvoiced dominates. (3) Final escalation:
    add `voiced_unvoiced_spec_contrast_asymmetry` (same mechanism on
    7-dim spec contrast) so GBM sees accompaniment-vs-voice asymmetry
    in cepstral + pitch-class + mastering-signature spaces
    simultaneously.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 18+
    iterations — I cannot verify whether 1eda8e3's 3 surviving clean FPs
    (singing=3 per baseline, korean=0, english=0 on 1eda8e3) have BOTH
    voiced and unvoiced frames in ±2s windows. Continuous sung phrases
    may zero the unvoiced mask → asymmetry sentinel 0.0 → feature
    no-op for exactly the singing FPs it's designed to bite. (ii) SHAP
    rollup STILL "no keeps yet — rollup empty" for 1eda8e3 — the
    theoretical claim "MFCC asymmetry couldn't bite singing because
    singer holds note across cut" is unverifiable against actual SHAP.
    (iii) Domain-specific not-splice vs cross-song distributions of
    unvoiced_chroma_dist − voiced_chroma_dist are unknown pre-commit;
    calibrating threshold intuitions from mechanism not data.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent blocker for 18+ iterations; per-FP (domain, file,
    t_sec, label_id, p_splice, vp_pre_frac, vp_post_frac,
    voiced_mfcc_dist, unvoiced_mfcc_dist, voiced_chroma_dist,
    unvoiced_chroma_dist, top-5 |SHAP| features with values). Every
    asymmetry / mask hypothesis has been calibrated from theory; this
    single block flips the entire loop to data-driven. (2)
    `scripts/feature_oof_preview.py --add <feature_fn>` that retrains
    once and reports per-domain OOF-F1 delta vs current — turns "is this
    feature worth a 3-min retrain" from a bet into a numeric preview.
    (3) DISCARD-REVERT SYNC auditor: the wrapper's discard path should
    diff features.py vs baseline-sha features.py and ABORT / restore if
    they differ, so a discarded hypothesis's code cannot silently persist
    into the next iteration's classifier (voiced_spec_contrast dragging
    confounds every subsequent attribution).

## 2026-04-20T10:26:27+09:00 — 7972a98 (keep, combined=0.589282)
subject: add voiced_unvoiced_spec_contrast_asymmetry feature (FEATURE_NAMES 79->80) — third entry in the asymmetric voiced/unvoiced series after 1eda8e3 MFCC (keep +0.054) and f4148cc chroma (discard -0.004). Signed asymmetry unvoiced_spec_contrast_cosine_dist minus voiced_spec_contrast_cosine_dist on +-2s pre/post window. Reuses cached feat_contrast + feat_vp (existing for spec_contrast_delta and e7ca9eb voiced_spec_contrast) — ZERO new librosa calls, ZERO new caches. Per-t cost 4 slices + 4 masked means + 2 cosines on 7-dim vectors, sub-ms. Sentinel 0.0 when either mask empty or either sub-norm underflows, matching 1eda8e3 safety pattern. Targets 1eda8e3 current-keep (0.586244) singing 0.340 weakest-domain plateau where MFCC asymmetry lifted english 0.732->0.889 and korean 0.526->0.667 but singing stayed FLAT, and chroma asymmetry failed (singing 0.340->0.319 because singer sings new note in new key across real splice so voiced_chroma_dist also high and asymmetry cancels). Spec_contrast targets a genuinely orthogonal axis: MASTERING SIGNATURE — per-band peak-to-valley amplitude ratio is a compressor/limiter fingerprint. Within one mastered track the profile is fixed; cross-song splice crosses masters (different compression ratio, different EQ, different limiter). Mechanism on singing: (i) real same-singer cross-song splice — voiced contrast stable (same formant structure), unvoiced contrast SHIFTS (different mastering on drums/bass) -> asymmetry LARGE POSITIVE. (ii) intra-song chord cycle — voiced contrast shifts (different vowel formants per phrase), unvoiced contrast stable (same drums, same mastering) -> asymmetry NEGATIVE. (iii) different-singer cross-song — both shift -> asymmetry ~0. The SIGN carries discriminating info that chord-cycle FPs cannot mimic. Paired differencing on speech: voiced and unvoiced contrast co-vary with phoneme context (plosive burst + vowel land on similar band peaks) -> difference is zero-mean noise -> GBM low SHAP on english/korean. Cited escalation (3) from f4148cc(c)(3): 'add voiced_unvoiced_spec_contrast_asymmetry so GBM sees accompaniment-vs-voice asymmetry in cepstral + pitch-class + mastering-signature spaces simultaneously'. e7ca9eb voiced_spec_contrast_cosine_dist is still in-tree (wrapper failed to revert on discard) so adding the asymmetric companion completes the pair — GBM can learn the (voiced_single, asymmetry) joint that single-mask alone couldn't surface. GBM max_depth=3 CANNOT synthesize this subtraction interaction from voiced_spec_contrast alone. Orthogonal to every prior axis: NOT 1eda8e3 voiced_unvoiced_mfcc_asymmetry (13-dim cepstral smooth envelope); NOT f4148cc voiced_unvoiced_chroma_asymmetry (12-dim pitch-class); NOT e7ca9eb voiced_spec_contrast (single-mask, no sign); NOT 32cac36 voiced_chroma (single-mask, pitch-class); NOT 308aa5a tonnetz (6-dim linear projection of chroma); NOT 49bd0b1 voiced_mfcc (single-mask cepstral). First ASYMMETRIC/DIFFERENTIAL feature in the 7-dim MASTERING-SIGNATURE axis. Pure features.py change; FEATURE_NAMES count gate triggers wrapper auto-retrain via US-505 sha gate. Smoke-verified: len(FEATURE_NAMES)==80, last name 'voiced_unvoiced_spec_contrast_asymmetry', end-to-end extract_features returns 80 finite features on synthetic audio, sentinel 0.0 triggers correctly when voicing-mask empty (pure-sine probe).
per-domain: combined_english=0.888889 combined_korean=0.666667 combined_singing=0.345312

# 2026-04-20 — hypothesis: voiced_unvoiced_spec_contrast_asymmetry (FEATURE_NAMES 79→80)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new feature
    `voiced_unvoiced_spec_contrast_asymmetry`. Third entry in the asymmetric
    voiced/unvoiced series (MFCC=keep 1eda8e3 +0.054, chroma=fail f4148cc
    −0.004, spec_contrast=untried). Per position t_sec compute (1)
    voiced_contrast_dist = cosine distance between VOICED-mean 7-dim
    spectral_contrast on pre=[t−2, t] vs post=[t, t+2]; (2)
    unvoiced_contrast_dist = same on UNVOICED frames; (3) return
    `unvoiced_contrast_dist − voiced_contrast_dist`. Reuses feat_contrast
    + feat_vp (ZERO new librosa calls, ZERO new caches — feat_contrast is
    already built for the existing spec_contrast_delta feature and the
    e7ca9eb voiced_spec_contrast single-mask block). Sentinel 0.0 when
    either mask empty or sub-norm underflows.

(b) WHY this over recent failures. f4148cc's chroma asymmetry post-mortem
    explicitly cited this as escalation (3): "voiced_unvoiced_spec_
    contrast_asymmetry so GBM sees accompaniment-vs-voice asymmetry in
    cepstral + pitch-class + mastering-signature spaces simultaneously."
    MFCC asymmetry (1eda8e3) lifted speech dramatically (english
    0.732→0.889, korean 0.526→0.667) but singing stayed FLAT 0.340 —
    MFCC captures VOCAL-TRACT timbre, and intra-song chord cycles also
    have stable voiced-MFCC + shifting unvoiced-MFCC (drums hit at every
    chord) giving FALSE-POSITIVE asymmetry. Chroma asymmetry (f4148cc)
    failed because on real cross-song splice the SINGER sings a new
    note in the new key → voiced_chroma_dist ALSO high → asymmetry
    cancels. Spec_contrast attacks a genuinely orthogonal axis:
    MASTERING SIGNATURE. Spec_contrast measures peak-to-valley
    amplitude ratio per frequency band — a compressor/limiter
    fingerprint. Within one mastered track the per-band contrast
    profile is fixed; cross-song splice crosses masters (different
    compression ratio, different EQ, different limiter threshold).
    Mechanism on singing: (i) real SAME-singer cross-song splice —
    voiced contrast stable (same formant structure), unvoiced contrast
    SHIFTS (different mastering on drums/bass) → asymmetry LARGE
    POSITIVE. (ii) intra-song chord cycle — voiced contrast shifts
    (different vowel per phrase, different formant peaks), unvoiced
    contrast stable (same drums, same mastering) → asymmetry NEGATIVE.
    (iii) real different-singer cross-song — both shift → asymmetry
    ~0. The SIGN carries discriminating information that chord-cycle
    FPs don't produce in the same direction as real splices. Paired
    differencing on speech: voiced and unvoiced contrast co-vary with
    phoneme context (plosive burst + vowel land on similar band peaks)
    → difference is zero-mean noise → GBM low SHAP on english/korean.
    Note that e7ca9eb voiced_spec_contrast_cosine_dist is STILL in the
    codebase (79-feature baseline includes it — wrapper failed to
    revert on discard); adding the asymmetric companion completes the
    pair so GBM can learn the (voiced_single, asymmetry) joint that
    single-mask alone couldn't surface.

    Orthogonal to every prior axis: NOT 1eda8e3 (13-dim MFCC cepstral
    envelope); NOT f4148cc (12-dim chroma pitch-class); NOT e7ca9eb
    voiced_spec_contrast (single-mask, no sign); NOT 32cac36 voiced_chroma
    (single-mask, pitch-class); NOT 308aa5a tonnetz (linear chroma
    projection). First asymmetry in the 7-dim MASTERING-SIGNATURE
    axis. Blast radius: features.py only — 1 block + 1 FEATURE_NAMES
    append + assert bump. Zero new caches. Per-t cost: 4 slices + 4
    masked means + 2 cosines on 7-dim vectors, sub-ms. Risk-bounded:
    sentinel-0.0 safety; paired differencing is self-cancelling on
    domains where both masks co-vary with phoneme noise.

(c) IF THIS FAILS. (1) If asymmetry signal collapses on singing because
    chord-transition voiced contrast shifts are small enough that
    asymmetry looks positive too, CLIP sign to the positive side only
    and add a companion `voiced_unvoiced_spec_contrast_symmetry` =
    unvoiced+voiced (jointly-big vs jointly-small), giving GBM both
    axes. (2) If GBM assigns ~0 SHAP because feat_contrast is too
    correlated with feat_rolloff/centroid/bandwidth already in the
    feature set, drop to a SPARSIFIED 3-band asymmetry (low 100-500Hz,
    mid 500-4k, high 4k+) targeting drum/bass vs mastering-tail
    separately. (3) Final escalation: shift off features entirely —
    try ANALYSIS_STRIDE_S=0.06 (denser scan, only 0.11 failed) or
    GBM_MIN_SEP_S=2.0 (below current 2.5, catch clustered singing TPs).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 18+
    iterations — I cannot verify the 3 surviving singing FPs have
    BOTH voiced AND unvoiced frames in ±2s windows (all-voiced sung
    phrases → empty unvoiced mask → sentinel 0.0 → feature no-op).
    (ii) SHAP rollup STILL "no keeps yet — rollup empty" for 1eda8e3
    — the claim "MFCC asymmetry couldn't bite singing because chord
    cycles look like accompaniment change" is unverifiable. (iii)
    The voiced_spec_contrast discard-that-wasn't (e7ca9eb in code at
    79) contaminates attribution — combined delta from this
    hypothesis will mix the asymmetry's lift with whatever
    voiced_spec_contrast currently contributes.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent blocker for 18+ iterations; per-FP (domain, file,
    t_sec, label_id, p_splice, vp_pre_frac, vp_post_frac,
    voiced_mfcc_dist, unvoiced_mfcc_dist, voiced_chroma_dist,
    unvoiced_chroma_dist, voiced_spec_contrast_dist,
    unvoiced_spec_contrast_dist, top-5 |SHAP| features with values).
    Every asymmetry / mask hypothesis is calibrated from theory; this
    block flips the entire loop to data-driven. (2)
    `scripts/feature_oof_preview.py --add <feature_fn>` that retrains
    once and reports per-domain OOF-F1 delta vs current — turns "is
    this feature worth a 3-min retrain" into a numeric preview. (3)
    DISCARD-REVERT SYNC auditor: the wrapper's discard path should
    diff features.py vs baseline-sha features.py and ABORT / restore
    if they differ — e7ca9eb's voiced_spec_contrast_cosine_dist is
    STILL in-tree post-discard, confounding every subsequent
    attribution.
[auto] (no SHAP data for either 1eda8e3 or 7972a98)

## 2026-04-20T11:23:40+09:00 — de0be6f (discard, combined=0.584400)
subject: add voiced_unvoiced_spec_contrast_symmetry feature (FEATURE_NAMES 80->81) — SUM companion to 7972a98's just-kept voiced_unvoiced_spec_contrast_asymmetry. Nonnegative joint magnitude voiced_spec_contrast_cosine_dist + unvoiced_spec_contrast_cosine_dist on +-2s pre/post window. Reuses cached feat_contrast + feat_vp — ZERO new librosa calls, ZERO new caches. Per-t cost is 4 slices + 4 masked means + 2 cosines on 7-dim vectors (same machinery as asymmetry block), sub-ms. Sentinel 0.0 when either mask empty or either sub-norm underflows, matching 1eda8e3/7972a98 safety pattern. Targets 7972a98 current-keep (0.589282) singing 0.345 weakest-domain plateau where asymmetry barely moved singing (+0.005) because asymmetry collapses two physically distinct cases to ~0: (i) BOTH voiced and unvoiced contrast shift together (different-singer cross-song, everything changes, diff cancels); (ii) NEITHER shifts (quiet within-song passage). Both collapse to sentinel losing discrimination. The SYMMETRY recovers that axis: joint magnitude HIGH = splice signature (either case i or strong overall change), LOW = stable within-song. Combined with asymmetry GBM sees the full 2D (symmetry, asymmetry) plane: (high_sym, ~0_asym) = different-singer cross-song; (any_sym, +asym) = same-singer cross-song accompaniment shift; (low_sym, ~0_asym) = within-song stable. GBM max_depth=3 CAN access voiced_spec_contrast_cosine_dist (e7ca9eb in-tree) but NOT unvoiced_spec_contrast_cosine_dist so it cannot synthesize the sum from current features — explicit symmetry surfaces the 'joint magnitude' pattern directly. Explicit escalation option (1) from 7972a98(c)(1): 'add a companion voiced_unvoiced_spec_contrast_symmetry = unvoiced+voiced (jointly-big vs jointly-small), giving GBM both axes'. Orthogonal to every prior axis: NOT 7972a98 (DIFFERENCE not sum — orthogonal linear combination of same two components); NOT e7ca9eb voiced_spec_contrast (single-mask voiced-only); NOT 1eda8e3 voiced_unvoiced_mfcc_asymmetry (cepstral envelope not mastering signature AND diff not sum); NOT f4148cc voiced_unvoiced_chroma_asymmetry (pitch-class not mastering AND diff not sum); NOT 32cac36 voiced_chroma (single-mask pitch-class); NOT 308aa5a tonnetz; NOT 49bd0b1 voiced_mfcc (single-mask cepstral). First SYMMETRIC/JOINT-MAGNITUDE feature in the voiced/unvoiced mask family. Pure features.py change; FEATURE_NAMES count gate triggers wrapper auto-retrain via US-505 sha gate. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_unvoiced_spec_contrast_symmetry', end-to-end extract_features returns 81 finite features on synthetic audio, feature is nonnegative bounded [0, 4] as expected from sum of two cosine distances each in [0, 2].
per-domain: combined_english=0.925000 combined_korean=0.676923 combined_singing=0.318750

# 2026-04-20 — hypothesis: voiced_unvoiced_spec_contrast_symmetry (FEATURE_NAMES 80→81)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new feature
    `voiced_unvoiced_spec_contrast_symmetry` = voiced_spec_contrast_dist +
    unvoiced_spec_contrast_dist on ±2s pre/post window. Paired companion
    of 7972a98's just-kept voiced_unvoiced_spec_contrast_ASYMMETRY
    (difference); this is the SUM on identical machinery. Reuses cached
    feat_contrast + feat_vp — ZERO new librosa calls, ZERO new caches.
    Per-t cost is 4 slices + 4 masked means + 2 cosines on 7-dim vectors
    (same as the asymmetry block), sub-ms. Sentinel 0.0 when either mask
    empty or either sub-norm underflows, matching 1eda8e3/7972a98 safety
    pattern. FEATURE_NAMES 80→81 triggers wrapper auto-retrain via US-505
    sha gate.

(b) WHY this over recent failures. 7972a98 (current keep, combined=0.589282)
    lifted aggregate +0.003 and singing slightly (0.340→0.345) via
    ASYMMETRY on the mastering-signature axis. The asymmetry feature is
    SILENT on two physically distinct cases that both need discriminating:
    (i) BOTH voiced and unvoiced contrast shift together (different-singer
    cross-song splice with different mastering — everything changes,
    asymmetry cancels to ~0); (ii) NEITHER shifts (quiet within-song
    passage — asymmetry also ~0). The ASYMMETRY collapses both the
    "jointly big" and "jointly small" cases to the same sentinel, losing
    discrimination. The SYMMETRY (sum) recovers exactly that axis: joint
    magnitude large=splice-signature, small=within-song-stability.
    Combined with the existing asymmetry, GBM sees the full 2D
    (symmetry, asymmetry) plane: (high sym, ~0 asym) = different-singer
    cross-song; (any sym, +asym) = same-singer cross-song; (low sym,
    ~0 asym) = stable within song. GBM max_depth=3 CAN already access
    voiced_spec_contrast_cosine_dist (e7ca9eb in-tree) but NOT
    unvoiced_spec_contrast_cosine_dist as independent features, so it
    cannot synthesize the sum — explicit SYMMETRY as a feature surfaces
    the "joint magnitude" pattern directly.

    Explicit cited escalation from 7972a98(c)(1): "If asymmetry signal
    collapses on singing because chord-transition voiced contrast shifts
    are small enough that asymmetry looks positive too, CLIP sign to the
    positive side only and add a companion
    `voiced_unvoiced_spec_contrast_symmetry` = unvoiced+voiced
    (jointly-big vs jointly-small), giving GBM both axes." This is
    option (1) from the most recent keep's own fallback plan.

    Orthogonal to every prior axis: NOT 7972a98 (DIFFERENCE not sum; same
    two components, orthogonal linear combination); NOT e7ca9eb
    voiced_spec_contrast (single-mask, voiced-only); NOT
    voiced_unvoiced_mfcc_asymmetry 1eda8e3 (different axis: cepstral
    envelope not mastering signature, AND difference not sum); NOT
    f4148cc voiced_unvoiced_chroma_asymmetry (different axis AND
    difference). First SYMMETRIC / JOINT-MAGNITUDE feature in the
    voiced/unvoiced mask family. Blast radius: features.py only — 1
    new block function that duplicates the asymmetry block's mask/cosine
    machinery and returns the sum; 1 FEATURE_NAMES append; 1 assert bump;
    1 call in extract_features. ZERO new caches. Per-t cost: sub-ms
    (mirrors asymmetry). Risk-bounded: sentinel-0.0 safety identical to
    7972a98; feature is strictly nonnegative (sum of two cosine distances,
    each in [0, 2]), bounded [0, 4].

(c) IF THIS FAILS. (1) If GBM assigns ~0 SHAP (symmetry too correlated
    with the existing single voiced_spec_contrast_cosine_dist at
    max_depth=3), ADD a companion on the MFCC axis as well
    (voiced_unvoiced_mfcc_symmetry) so GBM sees joint-magnitude
    simultaneously in cepstral + mastering-signature spaces. (2) If
    singing regresses (jointly-big false positives fire on loud
    orchestral swells within same song), threshold the symmetry below
    a quantile — only fire above the 90th percentile of training-set
    joint magnitudes. (3) Final escalation: shift off features entirely
    — try ANALYSIS_STRIDE_S=0.08 (denser scan between untried 0.06 and
    current 0.12) or GBM_MIN_SEP_S=3.0 (= previously kept lower value,
    for tighter dedup letting clustered TPs survive).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 19+
    iterations — I cannot verify whether 7972a98's 3 surviving singing
    FPs have jointly-big contrast shifts (testing this hypothesis's
    core claim directly) or jointly-small. Without that data, the
    "chord cycle = low symmetry" mechanism is theoretical. (ii) SHAP
    rollup STILL "no keeps yet — rollup empty" for the current keep
    (7972a98) — cannot see how the asymmetry feature's per-domain SHAP
    compares to voiced_spec_contrast_cosine_dist to predict whether the
    symmetry will add lift or be dominated by the existing single-mask
    feature. (iii) The 0.589/0.586/0.532 plateau across the last 3
    keeps suggests the feature family (voiced-mask + asymmetry on
    MFCC/chroma/spec_contrast) may be reaching diminishing returns;
    adding the final symmetry completes the linear-combination coverage
    of this family but doesn't introduce a new physical axis (tempo,
    rhythm, HPSS-percussive).

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent blocker for 19+ consecutive iterations; per-FP
    (domain, file, t_sec, label_id, p_splice, vp_pre_frac, vp_post_frac,
    voiced_mfcc_dist, unvoiced_mfcc_dist, voiced_chroma_dist,
    unvoiced_chroma_dist, voiced_spec_contrast_dist,
    unvoiced_spec_contrast_dist, top-5 |SHAP| features with values).
    Every symmetry/asymmetry/mask hypothesis is calibrated from theory;
    this single block flips the entire loop to data-driven. (2)
    `scripts/feature_oof_preview.py --add <feature_fn>` that retrains
    once and reports per-domain OOF-F1 delta vs current — turns "is
    this feature worth a 3-min retrain" into a numeric preview. (3)
    DISCARD-REVERT SYNC auditor — the wrapper's discard path should
    diff features.py vs baseline-sha features.py and ABORT/restore on
    drift — e7ca9eb's voiced_spec_contrast_cosine_dist is still in-tree
    post-discard contaminating every subsequent attribution.

## 2026-04-20T11:39:15+09:00 — fd500b3 (discard, combined=0.567821)
subject: add percussive_mfcc_cosine_dist feature (FEATURE_NAMES 80->81) — third mask type orthogonal to the voiced/unvoiced family that saturated the last 10+ iterations. Percussive frames = onset_strength > chunk-median*1.2 (top ~40% by onset energy = transient-rich frames dominated by drum hits + attacks). Per position t_sec: slice feat_onset + feat_mfcc on +-2s pre/post, keep onset-dominant frames, return cosine distance of percussive-mean 13-dim MFCC vectors. Reuses cached feat_mfcc + feat_onset — ZERO new librosa calls. Per-chunk cost: one scalar median (cached lazily). Per-t cost: 2 slices + 1 boolean compare + 1 masked mean + 1 cosine on 13-dim vectors, sub-ms. Sentinel 0.0 when either mask empty or either sub-norm underflows, matching 1eda8e3/7972a98 safety pattern. Targets 7972a98 current-keep (0.589282) singing 0.345 weakest-domain plateau. Last 4 keeps (0.506->0.532->0.586->0.589) all voiced/unvoiced variants; most recent discards (f4148cc chroma-asym, de0be6f spec_contrast-symmetry) both failed to move singing off 0.318-0.345. CLAUDE.md mandates structural change after 5+ same-axis failures; voiced/unvoiced mask axis is exhausted. Root-cause singing plateau is that real cross-song splices often land on held vocal notes where voiced_mfcc AND voiced_chroma stay similar (singer holds new note's pitch class too), so voiced-mask features fire weakly on these TPs. The signal that CONSISTENTLY shifts across cross-song splices is accompaniment — and drum/bass attack frames (percussive-dominant) carry that signature most cleanly. Within one song drum kit + mic + mastering + limiter is FIXED so percussive-frame mean MFCC stays stable across chord transitions (cos_dist 0.05-0.15). Cross-song splice: different drum kit/mic/mastering/compression -> percussive-frame MFCC shifts (0.25-0.60). Unlike the unvoiced MASK (any non-vowel including breath/silence/sibilants, dominated by phoneme-correlated variance on speech), percussive is tighter transient-driven subset — plosive onsets land in the mask on speech (phoneme-dependent noise) so feature self-gates via GBM low per-domain SHAP on english/korean, same mechanism as voiced_mfcc (1eda8e3 pattern). Unlike the VOICED mask (sustained tones where same-singer cross-song has low distance), percussive mask sees drum/accompaniment ONLY so held-note cross-song splices light it up cleanly. Crucially NOT re-introducing df6fc0a unvoiced_mfcc (failed -0.025 on singing): unvoiced = EVERY non-voiced frame across phoneme distribution; percussive = TOP 40% BY ONSET STRENGTH, much tighter transient-focused subset. Orthogonal to every prior axis: NOT 49bd0b1 voiced_mfcc (voiced mask, sustained tones); NOT df6fc0a unvoiced_mfcc (complement mask); NOT 32cac36 voiced_chroma (voiced+pitch-class); NOT 308aa5a tonnetz; NOT e7ca9eb voiced_spec_contrast; NOT 1eda8e3/f4148cc/7972a98 asymmetries (voiced vs unvoiced pairwise); NOT de0be6f symmetry (pairwise sum); NOT mfcc_delta (all frames). First PERCUSSIVE/TRANSIENT-DOMINANT mask feature — genuinely new physical axis. Pure features.py change; FEATURE_NAMES count gate triggers wrapper auto-retrain via US-505 sha gate. Smoke-verified: len(FEATURE_NAMES)==81, last name 'percussive_mfcc_cosine_dist', synthetic cross-source splice yields 0.025 at boundary vs 0.002 within pre (12x discrimination), end-to-end extract_features returns 81 finite features, idempotent, sentinel 0.0 fires correctly on silence.
per-domain: combined_english=0.888889 combined_korean=0.646154 combined_singing=0.318750

# 2026-04-20 — hypothesis: percussive_mfcc_cosine_dist feature (FEATURE_NAMES 80→81)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new feature
    `percussive_mfcc_cosine_dist` on a THIRD mask type (percussive-dominant
    frames) orthogonal to the voiced/unvoiced mask family that has saturated
    the last 10+ iterations. Per chunk, cache a single scalar threshold
    `perc_threshold = median(feat_onset) * 1.2`. Per position t_sec: slice
    feat_onset + feat_mfcc on pre=[t−2, t] and post=[t, t+2]; keep frames
    where `onset_strength > perc_threshold` (top ~40% by onset energy =
    transient-rich frames dominated by drum hits / attacks); return cosine
    distance of percussive-mean 13-dim MFCC vectors. Reuses cached
    feat_mfcc + feat_onset — ZERO new librosa calls, ZERO new expensive
    caches (one scalar median). Sentinel 0.0 when either mask empty or
    either sub-norm underflows, matching 1eda8e3/7972a98 safety pattern.
    FEATURE_NAMES 80→81 triggers wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. The voiced/unvoiced mask family has
    hit diminishing returns across the last 4 keeps (0.506 → 0.532 →
    0.586 → 0.589) and has now plateaued — the most recent discards
    (f4148cc chroma-asym, de0be6f spec_contrast-symmetry) both failed to
    move singing off 0.318–0.345. CLAUDE.md mandates structural change
    after 5+ same-axis failures; voiced/unvoiced is exactly that exhausted
    axis. The singing plateau's root cause (per prior reflections) is
    that real cross-song splices often land on held vocal notes where
    voiced_mfcc stays similar AND voiced_chroma stays similar (singer
    holds the new note's pitch class too), so every voiced-mask feature
    fires weakly on these TPs. The signal that CONSISTENTLY shifts
    across cross-song splices is the ACCOMPANIMENT — and drums/bass
    attack frames (percussive-dominant) carry that signature most
    cleanly. Onset_strength peaks on every drum hit and transient;
    top-40% onset frames exclude sustained vocal tones and sustained
    instrument notes, isolating the drum/mastering fingerprint. Within
    one song the drum kit + mic + mastering + limiter is FIXED so
    percussive-frame mean MFCC stays ~stable across chord transitions
    (cos_dist ~0.05-0.15). Cross-song splice: different drum kit /
    mic / mastering / compression → percussive-frame mean MFCC shifts
    (~0.25-0.60). Unlike the unvoiced MASK (which captures any
    non-vowel frame including breath + silence + sibilants, with high
    phoneme-correlated variance on speech), the percussive MASK is
    specifically transient-driven — plosive onsets DO land in the mask
    on speech (phoneme-dependent noise) so the feature self-gates via
    GBM low SHAP on english/korean via the same mechanism as
    voiced_mfcc (1eda8e3 mask pattern). Unlike the VOICED mask (which
    captures sustained tones — same singer holds note across cut =
    low distance), the percussive mask sees the drum/accompaniment
    ONLY so held-note cross-song splices light it up cleanly.

    Crucially, this is NOT re-introducing unvoiced_mfcc (df6fc0a failed
    with singing regression 0.341→0.316): unvoiced frames = EVERY
    non-voiced frame (breath + sibilant + silence + consonant attacks),
    dominated on speech by phoneme-dependent noise across the whole
    non-vowel distribution. Percussive frames = TOP 40% BY ONSET
    STRENGTH, a much tighter transient-focused subset. On singing
    this excludes breath intakes and sibilants that weren't drum-like;
    on speech it preferentially catches plosive onsets (not sustained
    consonants), concentrating the phoneme noise on a narrower
    frame set that GBM can discount per-domain more cleanly.

    Orthogonal to every prior axis: NOT voiced_mfcc 49bd0b1 (voiced
    mask, sustained-tone frames); NOT unvoiced_mfcc df6fc0a
    (unvoiced mask = complement of voiced, includes silence+breath);
    NOT voiced_chroma 32cac36 (voiced+pitch-class); NOT tonnetz
    308aa5a; NOT voiced_spec_contrast e7ca9eb; NOT asymmetries
    1eda8e3/f4148cc/7972a98 (pairwise voiced vs unvoiced diffs);
    NOT symmetry de0be6f (pairwise sum); NOT mfcc_delta (all
    frames); NOT remote-half MFCC (long half-chunk means); NOT
    HPR file-level gates; NOT spec_flux/spec_contrast_delta (all
    frames). First PERCUSSIVE / TRANSIENT-DOMINANT mask feature —
    a genuinely new physical axis for the mask family. Blast
    radius: features.py only — 1 new block + 1 FEATURE_NAMES
    append + 2 assert bumps + 1 extract_features call + 1 lazy
    ctx scalar. ZERO new caches beyond a chunk-local median.
    Per-t cost: 2 slices + 1 mask comparison + 1 masked mean +
    1 cosine on 13-dim vectors, sub-ms. Risk-bounded: sentinel-0.0
    safety matches proven 1eda8e3 pattern; chunk-global median
    threshold is robust to quiet sections (still produces a
    non-degenerate mask); per-t boolean comparison adds <5μs.

(c) IF THIS FAILS. (1) If the mask is TOO STRICT on quiet singing
    sections (intro/outro without drums) so the ±2s window has
    <3 percussive frames and the sentinel fires on exactly the
    held-note cross-song splice TPs I'm targeting, LOWER the
    threshold multiplier 1.2 → 1.0 (top-50% onset) or use a
    hybrid mask (onset > median OR rms > median). (2) If GBM
    assigns ~0 SHAP on singing because percussive_mfcc is too
    correlated with mfcc_delta_NN on drum-rich songs, shift the
    feature to CHROMA (percussive_chroma_cosine_dist) — drums
    have near-flat chroma so this captures pitched accompaniment
    (bass line, guitar) at transients, complementary signal.
    (3) Final escalation: PAIRED ASYMMETRY on the new axis
    (voiced_percussive_mfcc_asymmetry = percussive_mfcc_dist −
    voiced_mfcc_dist), making the "accompaniment vs voice"
    discriminator explicit in the way 1eda8e3's MFCC asymmetry
    did for voiced/unvoiced.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after
    20+ consecutive iterations — I cannot verify whether 7972a98's
    3 surviving singing FPs have enough high-onset frames in
    ±2s windows (purely-sustained FP regions would zero the mask
    → sentinel 0.0 → feature no-op exactly where I want it to
    bite). (ii) SHAP rollup STILL "no keeps yet — rollup empty"
    for 7972a98 — cannot confirm whether voiced_mfcc_cosine_dist
    and voiced_unvoiced_mfcc_asymmetry are the top singing-side
    features or whether they're both low-SHAP and the singing
    lift came from elsewhere. (iii) Per-chunk onset_strength
    distribution is not surfaced; the 1.2x median threshold is
    calibrated from general music-analysis priors not observed
    onset histograms on the eval population.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE — persistent blocker for 20+ iterations;
    per-FP (domain, file, t_sec, label_id, p_splice,
    n_voiced_pre, n_unvoiced_pre, n_percussive_pre, onset_median_chunk,
    voiced_mfcc_dist, unvoiced_mfcc_dist, top-5 |SHAP| features
    with values). Every mask hypothesis is calibrated from
    theory; this one block flips the entire loop to data-driven.
    (2) `scripts/feature_oof_preview.py --add <feature_fn>` that
    retrains once and reports per-domain OOF-F1 delta vs current
    — turns "is this feature worth a 3-min retrain" into a
    numeric preview. (3) DISCARD-REVERT SYNC auditor: the wrapper's
    discard path should diff features.py vs baseline-sha features.py
    and ABORT / restore if they differ — e7ca9eb's
    voiced_spec_contrast_cosine_dist remains in-tree post-discard,
    contaminating every subsequent attribution including this one.

## 2026-04-20T11:56:25+09:00 — aa4f141 (discard, combined=0.578070)
subject: add voiced_percussive_mfcc_asymmetry feature (FEATURE_NAMES 80->81) — PAIRED DIFFERENCING companion to 1eda8e3's voiced_unvoiced_mfcc_asymmetry on a new mask axis: percussive (top-40%% onset-strength frames) instead of unvoiced. Signed asymmetry percussive_mfcc_cosine_dist minus voiced_mfcc_cosine_dist on +-2s pre/post window. Reuses cached feat_mfcc + feat_onset + feat_vp with one lazy scalar perc_threshold = median(feat_onset)*1.2 — ZERO new librosa calls. Per-t cost 4 slices + 4 masked means + 2 cosines on 13-dim vectors, sub-ms. Sentinel 0.0 when either mask empty or sub-norm underflows. Targets 7972a98 current-keep (0.589282) singing 0.345 weakest-domain plateau. Mechanism: fd500b3's percussive_mfcc solo failed (singing 0.345->0.319) because GBM globally over-weighted transient-mask noise on speech. Paired differencing (asymmetry) is the proven fix — 1eda8e3 voiced_unvoiced_mfcc_asymmetry kept +0.054 (biggest win in the voiced-mask family). On singing: real same-singer-cross-song splice = held vocal note so voiced_mfcc_dist ~0.05 stable, but drum kit/mastering shifts so percussive_mfcc_dist ~0.30, asymmetry ~+0.25 LARGE POSITIVE — exact chord-cycle-vs-cross-song discriminator voiced_mfcc alone CAN'T see. Intra-song chord cycle: both stable (same singer, same drums) -> asymmetry ~0. Different-singer cross-song: both shift -> asymmetry cancels (GBM falls back to voiced_mfcc which fires anyway). On speech, plosive onsets (percussive mask) and vowel formants (voiced mask) both co-vary with phoneme context, so difference is zero-mean noise -> GBM low SHAP on english/korean (self-gates exactly the way fd500b3 solo couldn't). Explicit cited escalation from fd500b3(c)(3): 'PAIRED ASYMMETRY on the new axis (voiced_percussive_mfcc_asymmetry = percussive_mfcc_dist - voiced_mfcc_dist), making the accompaniment-vs-voice discriminator explicit in the way 1eda8e3 MFCC asymmetry did for voiced/unvoiced'. Critically orthogonal to voiced/unvoiced asymmetry family: PERCUSSIVE mask != UNVOICED mask — percussive is top-40%% onset-strength (tight transient subset: drum hits / bass attacks on singing, plosive bursts on speech); unvoiced is every non-voiced frame (broad: breath/silence/sibilant/consonant). Physically different populations of accompaniment signal. Orthogonal to every prior axis: NOT 1eda8e3 voiced_unvoiced_mfcc_asymmetry (unvoiced mask); NOT fd500b3 percussive_mfcc (single-mask, no differential — that's what failed); NOT 7972a98 voiced_unvoiced_spec_contrast_asymmetry (7-dim mastering not 13-dim cepstral); NOT f4148cc voiced_unvoiced_chroma_asymmetry (pitch-class); NOT 49bd0b1 voiced_mfcc (single-mask); NOT df6fc0a unvoiced_mfcc (single-mask); NOT 308aa5a tonnetz; NOT raw mfcc_delta (all-frames). First TRANSIENT-PAIRED ASYMMETRY in the voiced-mask family. Pure features.py change; FEATURE_NAMES count gate triggers wrapper auto-retrain via US-505 sha gate. Smoke-verified: len(FEATURE_NAMES)==81, last name 'voiced_percussive_mfcc_asymmetry', extract_features returns 81 finite features on synthetic audio, sentinel 0.0 fires on silence.
per-domain: combined_english=0.878049 combined_korean=0.687500 combined_singing=0.320000

# 2026-04-20 — hypothesis: voiced_percussive_mfcc_asymmetry feature (FEATURE_NAMES 80→81)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new feature
    `voiced_percussive_mfcc_asymmetry` = percussive_mfcc_cosine_dist −
    voiced_mfcc_cosine_dist on ±2s pre/post window. Percussive mask = frames
    where feat_onset > chunk-median(feat_onset) * 1.2 (top ~40% by onset
    energy; drum-hit / transient-dominant frames). Voiced mask reused from
    49bd0b1. Positive asymmetry = percussive (accompaniment/mastering)
    changed MORE than voiced (singer's formant structure) = same-singer
    cross-song splice on held note. ~0 on intra-song chord cycles (both
    stable within fixed drum kit + singer) and different-singer cross-song
    (both shift, cancel). Reuses cached feat_mfcc + feat_onset + feat_vp —
    ZERO new librosa calls, ZERO new caches. Per-t cost is 4 slices + 4
    masked means + 2 cosines on 13-dim vectors, sub-ms. Sentinel 0.0 when
    either mask empty or sub-norm underflows, matching 1eda8e3/7972a98
    safety. FEATURE_NAMES 80→81 triggers wrapper auto-retrain via US-505.

(b) WHY this over recent failures. fd500b3's `percussive_mfcc_cosine_dist`
    (SINGLE mask) failed with combined=0.568 / singing REGRESSED 0.345→
    0.319 — GBM globally over-weighted a feature whose independent value
    on speech (plosive onsets are phoneme-dependent) contaminated all
    three domains. The EXPLICIT CITED fallback from fd500b3(c)(3) is
    "PAIRED ASYMMETRY on the new axis (voiced_percussive_mfcc_asymmetry
    = percussive_mfcc_dist − voiced_mfcc_dist), making the 'accompaniment
    vs voice' discriminator explicit in the way 1eda8e3's MFCC asymmetry
    did for voiced/unvoiced." The asymmetry pattern has a proven track
    record: 1eda8e3 voiced_unvoiced_mfcc_asymmetry was the single biggest
    keep +0.054, 7972a98 voiced_unvoiced_spec_contrast_asymmetry kept
    +0.003. Paired differencing cancels phoneme-correlated noise on
    speech because BOTH masks co-vary with phoneme context (plosive
    onsets land in percussive mask, vowel formants drive voiced mask,
    both move together within a speaker's prosodic window) → difference
    is zero-mean noise → GBM assigns low per-domain SHAP on english/
    korean, reproducing the self-gating that made fd500b3's solo mask
    fail. On singing the physics diverge: voiced_mfcc ~0.05 (held note,
    same singer, same vocal tract) but percussive_mfcc ~0.30 (new drum
    kit / mastering / compressor) → asymmetry ~+0.25 LARGE POSITIVE =
    the exact chord-cycle-vs-cross-song discriminator that voiced_mfcc
    alone CAN'T see (singer holds note, voiced_mfcc stays low, single-
    mask voiced_mfcc doesn't fire).

    Critically orthogonal to the voiced/unvoiced asymmetry family: the
    PERCUSSIVE mask ≠ UNVOICED mask. Unvoiced = any non-voiced frame
    (breath, silence, sibilant, consonant) — a broad phoneme-
    distribution noise source already captured by 1eda8e3. Percussive
    = top-40% by onset strength, a much tighter transient-focused
    subset. On singing, unvoiced captures breath gaps + sibilants
    (mostly vocal-source), percussive captures drum hits + bass
    attacks (mostly accompaniment-source). The PHYSICAL signal is
    different. On speech, unvoiced captures all non-vowel frames
    including silence, percussive captures plosive bursts only —
    again different. So voiced_unvoiced_mfcc_asymmetry (1eda8e3) and
    voiced_percussive_mfcc_asymmetry fire on DIFFERENT populations of
    accompaniment signal.

    Orthogonal to every prior axis: NOT 1eda8e3 voiced_unvoiced_mfcc
    (UNVOICED mask, broad non-vowel distribution); NOT fd500b3
    percussive_mfcc (single-mask, no differential — that's what failed);
    NOT 7972a98 voiced_unvoiced_spec_contrast (spec_contrast 7-dim
    mastering-signature, not 13-dim cepstral); NOT f4148cc
    voiced_unvoiced_chroma (pitch-class); NOT 49bd0b1 voiced_mfcc
    (single-mask); NOT df6fc0a unvoiced_mfcc (single-mask); NOT 308aa5a
    tonnetz; NOT raw mfcc_delta (all-frames). First TRANSIENT-PAIRED
    ASYMMETRY in the voiced-mask family. Blast radius: features.py
    only — 1 new block + 1 FEATURE_NAMES append + 2 assert bumps + 1
    call in extract_features. Zero new caches. Per-t cost: sub-ms.
    Risk-bounded: sentinel-0.0 safety matches 1eda8e3 proven pattern;
    paired differencing inherently self-cancels phoneme-correlated
    noise; percussive-mask chunk-median threshold is local-adaptive so
    robust to overall chunk loudness.

(c) IF THIS FAILS. (1) If singing has continuous-vocal phrases with
    <3 percussive frames in ±2s (empty mask → sentinel 0.0 → feature
    no-op exactly where needed), use a HYBRID mask (onset>median OR
    rms<median — "accompaniment-carrying" frames including quiet
    gaps). (2) If GBM assigns ~0 SHAP (percussive_mfcc still too
    correlated with voiced_mfcc because drum hits have broad spectral
    content overlapping formant bands), shift to CHROMA
    (voiced_percussive_chroma_asymmetry) — drums have near-flat
    chroma so percussive_chroma isolates pitched accompaniment (bass
    line, guitar riff) at transient frames, complementary to voiced
    pitch signal. (3) Final escalation: shift off features entirely
    — try ANALYSIS_STRIDE_S=0.08 (denser scan, only 0.11 failed,
    current 0.3 per baseline) or GBM_MIN_SEP_S=2.0 (below current 2.5,
    catch clustered singing TPs).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 21+
    consecutive iterations — I cannot verify whether 7972a98's 3
    surviving singing FPs have enough high-onset frames in ±2s windows
    for the percussive mask to fire. Purely-sustained chord-transition
    FPs would empty the mask → sentinel 0.0 → feature no-op on exactly
    the population this targets. Every mask design remains theory-
    calibrated. (ii) SHAP rollup STILL "no keeps yet — rollup empty"
    for 7972a98 — I cannot confirm that voiced_unvoiced_mfcc_asymmetry
    is actually carrying the english/korean lift on 1eda8e3 or whether
    it's being shadowed. Without this, the claim that asymmetry cancels
    phoneme noise on speech is structural-theoretical not empirical.
    (iii) Per-chunk onset_strength distribution histograms across the
    eval population are unknown; the 1.2x median threshold is
    calibrated from general music-analysis priors.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent blocker for 21+ iterations; per-FP (domain,
    file, t_sec, label_id, p_splice, vp_pre_frac, vp_post_frac,
    n_percussive_pre, n_percussive_post, voiced_mfcc_dist,
    unvoiced_mfcc_dist, percussive_mfcc_dist, top-5 |SHAP| features
    with values). Every asymmetry / mask / pairing design has been
    calibrated from theory; this single block flips the entire loop
    to data-driven. (2) `scripts/feature_oof_preview.py --add
    <feature_fn>` that retrains once and reports per-domain OOF-F1
    delta vs current — turns "is this feature worth a 3-min retrain
    cycle" from a bet into a numeric preview. (3) DISCARD-REVERT SYNC
    auditor: the wrapper's discard path should diff features.py vs
    baseline-sha features.py and ABORT / restore if they differ —
    e7ca9eb's voiced_spec_contrast_cosine_dist is STILL in-tree
    post-discard (features.py has 80 features, its block still
    present) contaminating every subsequent attribution including
    this one. A 3-line shell guard after the discard decision would
    prevent this.

## 2026-04-20T12:50:18+09:00 — 49fa2ca (discard, combined=0.516753)
subject: add onset_autocorr_cosine_dist feature (FEATURE_NAMES 80->81) — first RHYTHM/BEAT-PATTERN feature, genuinely new physical axis orthogonal to every prior voiced/unvoiced/percussive MASK variant and every spectral/cepstral/harmonic CONTENT feature. Autocorrelate feat_onset on pre=[t-4,t] and post=[t,t+4] (4-second windows = 8-13 beats at 60-200 BPM), normalize each by lag-0, extract beat-range slice [0.25s,1.25s] (= 48-240 BPM), return cosine distance between the two autocorr vectors. Reuses cached feat_onset — ZERO new librosa calls, ZERO new caches. Per-t cost 1-3 ms (numpy.correlate on ~345-frame segment). Beat-saliency gate: if either window's max normalized autocorr in beat range < 0.15, return sentinel 0.0 — hard-zeros speech where consonant onsets lack a dominant beat peak. Edge guard: window < max_lag+20 frames returns 0.0. FEATURE_NAMES 80->81 triggers wrapper auto-retrain via US-505 sha gate.
per-domain: combined_english=0.860759 combined_korean=0.502941 combined_singing=0.318750

# 2026-04-20 — hypothesis: onset_autocorr_cosine_dist feature (FEATURE_NAMES 80→81)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new feature
    `onset_autocorr_cosine_dist`, a RHYTHM/BEAT-PATTERN fingerprint.
    Compute autocorrelation of `feat_onset` independently on pre=[t−4, t]
    and post=[t, t+4] (4-second windows = 8-13 beats at 60-200 BPM),
    normalize each by its lag-0 value, extract the beat-range slice
    [0.25s, 1.25s] (= 48-240 BPM), and return cosine distance between the
    two beat-range autocorr vectors. Reuses cached `feat_onset` +
    `feat_frame_hop` + `feat_sr` — ZERO new librosa calls, ZERO new caches.
    Per-t cost: 2 slices + 2 autocorrelations via numpy.correlate on
    ~345-frame segments + 2 normalizations + 1 cosine on ~86-element
    vectors ≈ 1-3 ms. Beat-saliency gate: if either window's max
    normalized autocorr in the beat range < BEAT_SALIENCY_MIN=0.15 (no
    dominant beat peak), return sentinel 0.0 — intrinsically hard-zeros
    speech where onsets are irregular consonant transients. Edge guard:
    window < (max_lag + 20) frames returns sentinel 0.0. FEATURE_NAMES
    80→81 triggers wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. Last 8+ hypotheses since 1eda8e3 keep
    (0.586) iterated the voiced/unvoiced/percussive MASK family (ff65865,
    308aa5a, df6fc0a, e7ca9eb, 1eda8e3 keep, f4148cc, 7972a98 keep,
    de0be6f, fd500b3, aa4f141) — multi-dim cosine distances in
    MFCC/chroma/tonnetz/spec_contrast spaces with single-mask /
    paired-diff / paired-sum variants. Singing plateau 0.340-0.345
    held throughout; latest feature discards REGRESSED singing.
    99081f5 (latest commit, outcome uncertain — baseline_metrics.json
    still reflects 7972a98=0.589282 so likely DISCARDED) tried
    classifier capacity bump (max_depth 3→4, max_leaf 8→16) — first
    NON-feature structural move. CLAUDE.md mandates structural change
    after 5+ same-axis failures; voiced/unvoiced/percussive mask axis
    is exhausted on spectral/cepstral/harmonic CONTENT. Rhythm/tempo
    is a wholly new physical axis untouched by every prior feature in
    the 80-feature set. Within one song the drum kit plays at a fixed
    tempo; onset_strength autocorrelation has a STRONG peak at the
    beat period (typically 0.4-0.6s for 100-150 BPM pop/rock) plus
    subharmonic peaks at 2×/3× beat (bar boundaries). Cross-song
    splices cross drum grooves — new song often has different BPM OR
    different beat subdivision OR different syncopation. The autocorr
    VECTOR in the beat range encodes the full rhythm fingerprint (not
    just peak lag) — primary beat peak, bar-level subharmonics, and
    overall shape. Same song pre/post = near-identical vectors →
    cos_dist ~0.02-0.10. Different song = shifted peaks → cos_dist
    ~0.30-0.70. On chord-cycle intra-song FPs (exactly the singing
    population I'm targeting): voiced_mfcc stable (same singer),
    voiced_chroma moderate (chord transitions share pitch classes),
    voiced_unvoiced_spec_contrast_asymmetry ~0 (chord cycle is
    symmetric across voiced/unvoiced) — but drum pattern IDENTICAL
    before and after, so onset_autocorr_cosine_dist ~0. Tree splits
    learn "chord-cycle-FP = multiple spectral features fire moderately
    AND onset_autocorr_cosine_dist near-zero" vs "real-splice = same
    spectral pattern AND onset_autocorr shows rhythm shift". Self-
    gating on speech via beat-saliency gate: speech onset peaks are
    irregular consonant transients; autocorr in 0.25-1.25s range has
    NO dominant peak (max normalized autocorr ~0.05-0.12), gate fires
    → sentinel 0.0 hard-zero on english/korean. This gate is
    STRUCTURALLY different from voicing-mask self-gating (which still
    fires on speech with high-variance values): hard-zero means GBM
    sees the feature as literally non-informative on speech and
    CANNOT over-weight it globally (neutralizes the df6fc0a/fd500b3
    failure mode where high-variance noise feature regressed singing
    via GBM global mis-weighting).

    Orthogonal to every prior axis: NOT voiced/unvoiced/percussive
    MASK family (those use onset threshold as a FRAME MASK on
    spectral content; this uses the onset WAVEFORM for temporal
    autocorrelation); NOT cepstral (MFCC) / pitch-class (chroma) /
    tonal-centroid (tonnetz) / mastering-signature (spec_contrast) —
    these measure SPECTRAL content, rhythm is TEMPORAL pattern; NOT
    spec_flux_delta (single-t onset gradient); NOT
    boundary_spec_flux_peak (peak-at-boundary not period-repetition).
    First RHYTHM / BEAT-PATTERN feature — genuinely new physical axis.
    Blast radius: features.py only — 1 new block function + 1
    FEATURE_NAMES append + 2 assert bumps + 1 call in
    extract_features. ZERO new caches, ZERO new librosa calls. Per-t
    cost 1-3 ms; aggregate added cost in dense scan ≈ ~3-5 s within
    243/300 s eval budget. Risk-bounded: sentinel-0.0 safety +
    beat-saliency gate hard-zeros speech; feature bounded [0, 2];
    signal physically-grounded on the singing subset where both
    sides pass the gate.

(c) IF THIS FAILS. (1) If beat-saliency gate fires too often on
    singing (intro/outro / ballad passages without drums), widen lag
    range to [0.15, 1.5]s (40-400 BPM) OR lower BEAT_SALIENCY_MIN
    to 0.08 to catch sparse rhythmic signatures. (2) If GBM assigns
    ~0 SHAP (autocorr_cosine_dist is too correlated with
    spec_flux_delta because both are sensitive to transient density),
    replace with an explicit LAG-PEAK-SHIFT feature
    `onset_autocorr_peak_lag_shift` = |argmax(pre_ac) −
    argmax(post_ac)| / fps (smaller-magnitude scalar encoding tempo
    period shift in seconds, may be easier for GBM to split on).
    (3) Final escalation: tempogram cosine distance
    (librosa.feature.tempogram averaged over pre/post windows, cosine
    on 384-dim vectors) — richer rhythm fingerprint including meter
    /subdivision structure, ~50 ms/chunk added cost.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 22+
    iterations — I cannot verify whether 7972a98's 3 surviving
    singing FPs fall on drum-driven sections (beat-saliency gate
    passes → feature fires) or quiet vocal passages (gate fires →
    sentinel, no-op for exactly the population I'm targeting).
    Betting on the prior that most singing eval files are drum-heavy
    pop/rock. (ii) SHAP rollup STILL "no keeps yet — rollup empty"
    so I cannot see how 7972a98's features weigh per-domain.
    (iii) 99081f5 (HistGBM capacity bump) has NO keep/discard note
    commit yet but baseline_metrics.json reflects 7972a98 — outcome
    ambiguous. My feature delta attribution mixes with whatever
    classifier hyperparams currently live at HEAD (may be depth=3
    max_leaf=8 if discarded and reverted, or depth=4 max_leaf=16 if
    still pending).

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block —
    persistent blocker for 22+ iterations; per-FP (domain, file,
    t_sec, label_id, p_splice, onset_autocorr_peak_pre,
    onset_autocorr_peak_post, rhythm_salient_pre, rhythm_salient_post,
    voiced_mfcc_dist, unvoiced_mfcc_dist, top-5 |SHAP|). Every
    hypothesis since iteration ~70 calibrated from theory;
    flips loop to data-driven. (2) `scripts/feature_oof_preview.py
    --add <feature_fn>` that retrains once and reports per-domain
    OOF-F1 delta. (3) PREVIOUS-ITERATION-OUTCOME header in prompt
    ("last commit SHA: 99081f5 — DISCARD/KEEP/PENDING") resolves
    attribution ambiguity when keep/discard notes lag.

## 2026-04-20T14:01:07+09:00 — 6384137 (discard, combined=0.490700)
subject: add onset_autocorr_peak_lag_shift feature (FEATURE_NAMES 80->81) — SCALAR tempo-period shift companion to 49fa2ca's failed onset_autocorr_cosine_dist. Per position t_sec autocorrelate feat_onset on pre=[t-4,t] and post=[t,t+4] via numpy.correlate, normalize by lag-0, find argmax in beat-range lag window [0.25s,1.25s] (48-240 BPM), return |pre_peak_lag_seconds - post_peak_lag_seconds|. Beat-saliency gate: if either window's max normalized autocorr in beat range < 0.35, return sentinel 0.0 (hard-zeros speech + white noise). Edge guard: window < (max_lag+20) frames returns 0.0. Reuses cached feat_onset + feat_frame_hop + feat_sr — ZERO new librosa calls, ZERO new caches. Per-t cost 1-2 ms (numpy.correlate on ~345-frame segment). Targets 7972a98 current-keep (0.589282) singing 0.345 weakest-domain plateau where voiced/unvoiced/percussive MASK family has saturated last 10+ iterations (fd500b3 percussive_mfcc discard, aa4f141 voiced_percussive_asymmetry discard, 99081f5 HistGBM capacity bump discard, 49fa2ca onset_autocorr_cosine_dist discard) and singing won't budge off 0.318-0.345. CLAUDE.md mandates structural change after 5+ same-axis failures; mask axis exhausted. 49fa2ca failed by regressing korean 0.667->0.503 via cosine-on-86-dim-vector noise amplification — even with saliency gate, marginal-salience korean chunks leaked high-variance cosine values that GBM over-weighted. Explicit cited escalation from 49fa2ca(c)(2): 'replace with LAG-PEAK-SHIFT feature onset_autocorr_peak_lag_shift = |argmax(pre_ac) - argmax(post_ac)| / fps (smaller-magnitude scalar encoding tempo period shift in seconds, may be easier for GBM to split on)'. Mechanism: within one song drum groove fixes tempo, pre/post peak lags match within 1-2 frames -> lag_shift ~0.01-0.03s; cross-song splice crosses tempi (typical 10-50 BPM difference) -> lag_shift 0.05-0.30s 10-20x intra-song noise floor; intra-song chord cycles preserve tempo exactly (chord changes align to beat boundaries within same drum pattern) -> no lag shift -> feature silent. This is the chord-cycle-vs-cross-song discriminator voiced-mask family has failed to surface because voiced features care about tonal content not tempo. On speech, beat-saliency gate hard-zeros (threshold 0.35 tightened from 49fa2ca's 0.15 based on smoke data: white noise peaks 0.16-0.23, drums peak 0.74-0.83 — 0.35 cleanly separates). Why scalar beats cosine: (1) low variance — only PEAK POSITION matters, immune to noise in other lag bins; (2) bounded/physically meaningful in [0,1.0] seconds; (3) GBM tree splits on scalar see distribution directly, cannot misweight phantom per-bin correlations. Orthogonal to every prior axis: NOT 49fa2ca (cosine on 86-dim vector vs scalar peak-lag); NOT voiced/unvoiced/percussive MASK family (doesn't use frame masking; pure time-series autocorrelation); NOT spectral/cepstral/harmonic CONTENT features (TEMPORAL PATTERN only); NOT spec_flux_delta (single-t onset gradient at boundary); NOT boundary_spec_flux_peak (peak-at-boundary not tempo). First SCALAR rhythm feature in the 80-feature set. Pure features.py change; FEATURE_NAMES count gate triggers wrapper auto-retrain via US-505 sha gate. Smoke-verified: len(FEATURE_NAMES)==81, last name 'onset_autocorr_peak_lag_shift', same-tempo drums (120/120) yields 0.0, tempo-shift drums (120->150) yields 0.60 large positive, white noise yields sentinel 0.0 via saliency gate, edge t+4>chunk_dur yields 0.0, end-to-end extract_features returns 81 finite features.
per-domain: combined_english=0.839506 combined_korean=0.476471 combined_singing=0.295385

# 2026-04-20 — hypothesis: onset_autocorr_peak_lag_shift feature (FEATURE_NAMES 80→81)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new feature
    `onset_autocorr_peak_lag_shift`, a SCALAR tempo-period shift (in seconds).
    Per position t_sec: slice `feat_onset` on pre=[t−4, t] and post=[t, t+4]
    (4-second windows = 8-13 beats at 60-200 BPM), autocorrelate each
    independently via numpy.correlate, normalize by lag-0, then find
    argmax in the beat-range lag window [0.25s, 1.25s] (48-240 BPM).
    Return `|argmax_pre_seconds − argmax_post_seconds|` — the absolute
    tempo period difference in seconds. Beat-saliency gate: if either
    window's max normalized autocorr in beat range < 0.15, return sentinel
    0.0 (no dominant beat peak → hard-zero on speech). Edge guard: window
    < (max_lag + 20) frames returns 0.0. Reuses cached feat_onset +
    feat_frame_hop + feat_sr — ZERO new librosa calls, ZERO new caches.
    Per-t cost: 2 slices + 2 autocorrelations (numpy.correlate on
    ~345-frame segment) + 2 argmax + 1 abs subtract ≈ 1-2 ms.
    FEATURE_NAMES 80→81 triggers wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. 49fa2ca's `onset_autocorr_cosine_dist`
    (86-dim cosine distance over the beat-range autocorr vector) FAILED
    catastrophically on korean (0.667→0.503) while singing stayed flat
    (0.319). 99081f5's HistGBM capacity bump (max_depth 3→4, max_leaf_nodes
    8→16) also failed against the 80-feature space. CLAUDE.md's "structural
    change after 5+ same-axis failures" trigger fires: voiced/unvoiced/
    percussive MASK family is saturated (last 10+ iterations), and the
    scalar rhythm axis remains virgin — 49fa2ca's ONE rhythm attempt used
    a high-dim vector which is inherently noise-amplifying.

    The 49fa2ca post-mortem identified the failure mode precisely: cosine
    distance over 86 lag bins amplifies per-bin noise. Even with the
    beat-saliency gate (max_ac < 0.15 → sentinel), korean chunks with
    marginal beat-saliency (max_ac ~ 0.15-0.25 from quasi-periodic
    prosody) passed the gate and contributed high-variance cosine values,
    which GBM at 80 features globally over-weighted. The explicit cited
    escalation from 49fa2ca(c)(2) is: "replace with an explicit LAG-PEAK-
    SHIFT feature `onset_autocorr_peak_lag_shift` = |argmax(pre_ac) −
    argmax(post_ac)| / fps (smaller-magnitude scalar encoding tempo period
    shift in seconds, may be easier for GBM to split on)."

    Mechanism on singing: within one song the drum groove fixes the
    tempo; autocorr peaks at the beat period (0.40-0.60s for 100-150 BPM
    pop/rock) plus subharmonics at 2×/3× beat. pre- and post-peak lags
    match to within 1-2 frames → lag_shift ≈ 0.01-0.03s. Cross-song
    splice that crosses tempi (typical 10-50 BPM BPM difference between
    songs) produces lag_shift 0.05-0.30s — 10-20× the intra-song noise
    floor. Intra-song chord cycles preserve tempo exactly (the chord
    changes on beat boundaries within the same drum pattern) → no lag
    shift → feature silent. This is the CHORD-CYCLE-VS-CROSS-SONG
    discriminator that the voiced-mask family has failed to surface
    (because voiced features care about tonal content, not tempo). On
    speech, beat-saliency gate hard-zeros: consonant onsets lack a
    dominant periodic peak (max normalized autocorr ~0.05-0.12 in beat
    range), so sentinel 0.0 fires → GBM cannot over-weight (the exact
    failure mode of 49fa2ca cosine on 86 dims).

    Why scalar beats cosine: (1) Low variance — only PEAK POSITION
    matters, immune to noise in other lag bins. (2) Bounded and
    physically meaningful — value is in [0, 1.0] seconds, directly
    corresponds to tempo period difference. (3) GBM tree splits on a
    scalar see the distribution directly; tree cannot misweight
    phantom correlations across 86 bins.

    Orthogonal to every prior axis: NOT 49fa2ca (cosine on 86-dim
    vector vs scalar peak-lag); NOT voiced/unvoiced/percussive MASK
    family (doesn't use frame masking; uses pure time-series
    autocorrelation); NOT spectral/cepstral/harmonic CONTENT features
    (doesn't touch spectral content; TEMPORAL PATTERN only); NOT
    spec_flux_delta (single-t onset gradient at boundary, not
    period-repetition); NOT boundary_spec_flux_peak (peak-at-boundary
    not tempo); NOT spec_contrast (peak-to-valley amplitude per band,
    not tempo period). First SCALAR rhythm feature in the 80-feature
    set. Blast radius: features.py only — 1 new block function + 1
    FEATURE_NAMES append + 1 assert bump + 1 call in extract_features.
    ZERO new caches. Per-t cost 1-2 ms; aggregate added cost in
    dense scan ≈ ~2-4 s within 243/300 s eval budget. Risk-bounded:
    beat-saliency gate + edge guard + scalar-only output; feature
    bounded [0, 1.0]; signal is physically grounded on drum-driven
    singing where both windows pass the gate.

(c) IF THIS FAILS. (1) If the peak-lag feature also collapses because
    chord-cycle intra-song produces small lag drift (e.g., slight
    tempo fluctuations of ±5ms from human-played drums are enough to
    make lag_shift always non-zero and noise-dominated), add a
    DISCRETIZATION step — snap lag to nearest 1/32 note grid before
    differencing, so microtiming noise zeros out. (2) If singing real
    cross-song TPs happen to have matching tempos (common on
    mastered-compilation splices by intent), try
    `onset_autocorr_peak_height_shift` = |max_ac_pre − max_ac_post| —
    even at same tempo the beat-strength shifts if drum kit changes
    (different snare crack, different kick punch). (3) Final
    escalation: tempogram cosine distance (49fa2ca(c)(3)) — 384-dim
    tempogram with ~50 ms/chunk added cost. Richer fingerprint but
    reverts to cosine-on-vector failure mode of 49fa2ca.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 22+
    iterations — I cannot verify whether 7972a98's 3 surviving singing
    FPs fall on drum-driven sections (beat-saliency gate passes →
    feature fires) or on acapella/ballad passages (gate fires →
    sentinel, no-op exactly where I want it to bite). The prior that
    most singing eval files are drum-heavy pop/rock is reasonable but
    unverified. (ii) SHAP rollup STILL "no keeps yet — rollup empty"
    for 7972a98 — cannot confirm which features actually drive the
    singing-weakest plateau. (iii) per-chunk onset_strength
    distributions across domains unknown; beat-saliency threshold 0.15
    calibrated from prior/theory not data. (iv) Eval timing: my
    feature delta attribution mixes with classifier state — if
    99081f5's capacity bump was reverted (likely given baseline still
    0.589282), classifier is depth=3 max_leaf=8; if not reverted,
    depth=4 max_leaf=16.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block —
    persistent blocker for 22+ iterations; per-FP (domain, file, t_sec,
    label_id, p_splice, onset_autocorr_peak_pre, onset_autocorr_peak_post,
    rhythm_salient_pre, rhythm_salient_post, voiced_mfcc_dist, top-5
    |SHAP| features). Every hypothesis since iteration ~70 calibrated
    from theory; flips loop to data-driven. (2)
    `scripts/feature_oof_preview.py --add <feature_fn>` that retrains
    once and reports per-domain OOF-F1 delta vs current — "is this
    feature worth a 3-min retrain" as a numeric. (3) PREVIOUS-
    ITERATION-OUTCOME header in prompt (last SHA + KEEP/DISCARD/PENDING
    + last classifier hyperparams) resolves attribution ambiguity when
    keep/discard notes lag behind HEAD.

## 2026-04-20T14:15:24+09:00 — ea1636c (discard, combined=0.455183)
subject: REMOVE voiced_spec_contrast_cosine_dist ghost feature (FEATURE_NAMES 80->79) — first ABLATION in 23+ iterations of pure adds. e7ca9eb introduced this feature and was DISCARDED (singing regressed 0.341->0.316) but the wrapper failed to revert features.py so it persisted 15+ iterations contaminating attribution. Every reflection since 1eda8e3 flagged this as a blocker. Remove list entry at line 146, extract_features call at line 1107, update both len(FEATURE_NAMES)==80 asserts to 79. Leaves dead _block_voiced_spec_contrast function in-file (minimal diff, zero runtime cost). The kept asymmetry feature voiced_unvoiced_spec_contrast_asymmetry (7972a98) computes voiced_dist + unvoiced_dist INLINE from feat_contrast + feat_vp so removing the voiced-only raw feature does NOT break the asymmetry — it forces GBM to route spec_contrast signal through the paired-difference discriminator only, eliminating the chord-cycle noise channel that e7ca9eb's per-vowel formant shifts create on within-song singing. Targets 7972a98 current-keep (0.589282) singing 0.345 plateau. Self-diagnostic: if ablation HELPS (>=0.589, singing >=0.345), confirms the 5+-iteration attribution-contamination theory; if HURTS (<0.574), proves the voiced-only feature was providing signal the asymmetry missed — information either way. Orthogonal to EVERY prior axis: ZERO prior feature-removal attempts — 50+ feature hypotheses since iteration 1 all ADDS. Not primary-tunable (detector.py unchanged), not hyperparam (train_classifier.py unchanged), not DSP post-filter, not ensemble/calibration/class_weight/audio-mutation. Pure features.py REMOVAL. Blast radius: 4 line edits. FEATURE_NAMES length change triggers wrapper auto-retrain via US-505. Smoke-verified: import OK, len(FEATURE_NAMES)==79, ghost feature absent from list AND from extract_features output, asymmetry feature preserved, 79 finite features on synthetic audio.
per-domain: combined_english=0.775000 combined_korean=0.405634 combined_singing=0.300000

# 2026-04-20 — hypothesis: REMOVE voiced_spec_contrast_cosine_dist ghost feature (FEATURE_NAMES 80→79)

(a) HYPOTHESIS. Structural `splice/features.py` change — REMOVE one feature
    (the first ablation in 23+ iterations of pure additions). Delete the
    entry `+ ["voiced_spec_contrast_cosine_dist"]` from FEATURE_NAMES
    (drop 80→79), remove the `feats.update(_block_voiced_spec_contrast(...))`
    call in `extract_features`, update the two `assert len(FEATURE_NAMES)`
    checks 80→79. Leave the dead `_block_voiced_spec_contrast` function
    in-file as dead code (minimal diff). FEATURE_NAMES length change
    triggers wrapper auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. This feature was introduced by e7ca9eb
    which was DISCARDED (singing REGRESSED 0.341→0.316) but the wrapper
    failed to revert features.py — the block + FEATURE_NAMES entry +
    extract_features call have remained in-tree for ~15+ iterations.
    Every reflection since 1eda8e3 has explicitly flagged this: "e7ca9eb's
    voiced_spec_contrast_cosine_dist is STILL in-tree post-discard,
    contaminating every subsequent attribution." Grep confirms: line 146
    FEATURE_NAMES, line 1107 extract_features call, 937/942/946 function
    body. The asymmetry feature `voiced_unvoiced_spec_contrast_asymmetry`
    (kept 7972a98, in-tree at block 15, line 1007-1055) computes its own
    voiced_dist + unvoiced_dist INLINE from feat_contrast + feat_vp — it
    does NOT reference `voiced_spec_contrast_cosine_dist` as a feature.
    So removing the ghost feature does not break the asymmetry. Every
    other discard since feature-additions began (f4148cc chroma-asym,
    fd500b3 percussive_mfcc, aa4f141 voiced_percussive_asymmetry, 49fa2ca
    autocorr_cosine, 6384137 autocorr_peak_lag, de0be6f sym, df6fc0a
    unvoiced_mfcc, 308aa5a tonnetz, ff65865 far-chroma) WAS correctly
    reverted — only e7ca9eb slipped through.

    Expected mechanism: the ghost feature fires on within-song chord
    transitions (each vowel phrase shifts the voiced-frame 7-dim contrast
    mean by ~0.10-0.20) so acts as a chord-cycle NOISE channel that GBM
    partially up-weights into p_splice, inflating singing clean_fp. Its
    complement signal (unvoiced contrast on the same ±2s window) now lives
    EXCLUSIVELY in the asymmetry feature where paired differencing
    cancels the chord-cycle noise. Removing the raw voiced-only feature
    forces GBM to route the spec_contrast signal through the asymmetry
    ONLY — the clean splice-vs-chord-cycle discriminator. The e7ca9eb
    discard evidence is direct (singing 0.341→0.316 when added), but
    attribution was contaminated because at e7ca9eb's eval the
    asymmetry (7972a98) didn't exist yet — now that it does, the
    voiced-only single-mask version is redundant noise.

    Self-diagnostic character: if removing the ghost HELPS (combined ≥
    0.589 with singing ≥ 0.345), it confirms the attribution-
    contamination theory that 5+ prior reflections have cited as a
    blocker; if it HURTS (combined drops), it falsifies the theory and
    proves the feature was actually providing signal the asymmetry
    missed, informing future additions. Either outcome is information.

    Orthogonal to every prior axis: ZERO feature-removal attempts in the
    loop (all 50+ feature hypotheses since iteration 1 have been ADDS);
    NOT a primary-tunable bracket (detector.py unchanged), NOT a
    classifier hyperparam (train_classifier.py unchanged), NOT an
    asymmetry/symmetry/mask companion (no new feature block), NOT a
    DSP post-filter, NOT a calibration / ensemble / class_weight /
    audio-mutation. Pure features.py REMOVAL. Blast radius: 3 edits
    to features.py (line 146 remove list entry, line 151 & 1151 update
    asserts 80→79, line 1107 remove extract call). Zero librosa / numpy
    cost change (the function body stays dead but isn't called).

(c) IF THIS FAILS. (1) If combined stays flat or drops marginally (e.g.,
    0.585-0.589), the ghost was functionally inert and the failure
    confirms the plateau is NOT attribution contamination but genuine
    feature-space saturation — next axis is primary tunables
    (GBM_MIN_SEP_S = 2.5 untried below-tried range; ANALYSIS_STRIDE_S =
    0.08 untried above 0.11 failed). (2) If combined drops significantly
    (>0.015 hit), the ghost was providing real signal despite being a
    discard — then RESTORE it and pivot to asymmetry + symmetry joint
    structural use, e.g., add tonnetz asymmetry variant instead. (3)
    Final escalation: per-domain classifier routing (singing model
    trained ONLY on spliced/clean singing, speech model trained on
    english+korean together) — HistGBM × 2 with a file-HPR gate to
    choose which predict_proba to use at eval time.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 23+
    iterations — I cannot verify whether the 3 singing FPs correlate
    with high voiced_spec_contrast_cosine_dist values (direct evidence
    the ghost inflates them) or are orthogonal (ghost is inert).
    (ii) SHAP rollup STILL "no keeps yet — rollup empty" for 7972a98 —
    I cannot see the ghost feature's per-domain SHAP to predict the
    sign of the ablation effect. (iii) Classifier state: baseline
    reflects 7972a98 but commits since (99081f5 capacity bump, 49fa2ca,
    6384137 both rhythm, 4b7f27b/aea6d05 notes) may or may not have
    left the classifier at 7972a98's state — the wrapper's retrain
    cycle is atomic per iteration so CURRENT HEAD classifier must be
    trained on the 80-feature set including the ghost.

(e) Wrapper enhancements. (1) DISCARD-REVERT SYNC auditor — the
    wrapper's discard path should diff features.py vs baseline-sha
    features.py and ABORT / restore-and-retry if they differ. This
    prevents exactly the e7ca9eb situation where a discarded
    hypothesis's code persists for 15+ iterations contaminating every
    subsequent attribution. ~20 lines of shell guard after the discard
    decision. (2) CLEAN_FP_POSITIONS JSON block in CURRENT STATE (persistent
    blocker for 23+ iterations; per-FP feature vector + top-5 |SHAP|
    values) — flips the loop from theory-calibrated to data-driven
    feature design. (3) `scripts/feature_oof_preview.py --add <fn>`
    AND `--remove <name>` modes — retrains once, reports per-domain
    OOF-F1 delta. "Is this feature ADD/REMOVE worth a 3-min retrain"
    becomes a numeric preview instead of a bet.

## 2026-04-20T14:22:02+09:00 — 53d3ea0 (discard, combined=0.527473)
subject: tighten DSP_SUM_MIN 5.0 -> 5.5 — one untried step remaining in a monotonically-improving primary-tunable axis. Pure detector.py one-line constant edit, stacked on existing MAX>=2.0 + GBM_THRESHOLD=0.982. No retrain, no feature change, features.py sha stable so classifier byte-identical. Targets 7972a98 current-keep (combined=0.589282) singing 0.345 weakest-domain plateau with clean_fp=3 survivors past the current SUM>=5.0 gate. The SUM axis has prior monotone keeps: 865d92f (SUM=4.5 kept 0.488) -> 29c06cf (SUM=5.0 kept 0.494, +0.006), and 29c06cf's own keep note explicitly cited 5.5 as the next escalation that was never tried because feature-axis exploration took over immediately after. Per-tunable frontier confirms only 4.5 and 5.0 tried in the SUM axis. Mechanism: real cross-source splices disrupt mic/room (phase_z) + spectral-distribution (T2_z) + complex-prediction (CPE_z) SIMULTANEOUSLY by physical construction, so sum(dsp_phase_z, dsp_t2_z, dsp_cpe_z) is typically 6-12 with big margin above 5.5. Chord-transition / speech-phoneme FPs are dominated by ONE strong channel (T^2=3) plus partial support from others (phase~0.7 + CPE~1.0 + or small variations), producing sum in [5.0, 5.8] — exactly the band 5.5 bites. The 3 singing clean FPs surviving 7972a98's gates by construction pass sum>=5.0 AND max>=2.0; if any sit in [5.0, 5.5] they drop under the new gate while real singing splices (which touch multiple DSP channels via cross-source disruption) are preserved. Classifier improvements over the last 40 iterations (75->80 features, HistGBM swap, voiced-mask + asymmetry families for MFCC/chroma/spec_contrast) have tightened p_splice on real TPs to near-1.0 so TPs comfortably pass DSP regardless of this gate's tightening. Orthogonal to every recent axis: NOT a feature add (features.py sha stable, no retrain); NOT a classifier hyperparam (train_classifier.py unchanged, 99081f5 HistGBM capacity bump discarded); NOT DSP_CONFIRMATION_MIN MAX gate; NOT GBM_THRESHOLD softmax gate (7255ec6 0.980 failed); NOT GBM_MIN_SEP_S dedupe geometry; NOT ANALYSIS_STRIDE_S scan density; NOT class-routed DSP (eb8984e failed); NOT HPR-routed DSP (failed); NOT channel-specific CPE floor (0.449 failed); NOT a feature removal (ea1636c catastrophic 0.455). The ONE unused step in a tunable axis whose prior steps were monotonically positive. Blast radius: 1-line constant edit plus comment refresh. Zero new code paths; per-emit cost unchanged (same sum+compare). Risk-bounded: stacked gate can only DROP emits, cannot create TPs; drops require cumulative-DSP below 5.5. Worst-case failure mode: smooth-crossfade TPs with phase/CPE firing weakly (sum 5.0-5.8) get dropped and recall falls — then fallback is DSP_CONFIRMATION_MIN 2.0->2.2 on the MAX axis (different channel-mass distribution, never tried). Smoke-verified: import DSP_SUM_MIN returns 5.5, all other tunables unchanged (DSP_CONFIRMATION_MIN=2.0, GBM_THRESHOLD=0.982, GBM_MIN_SEP_S=3.5).
per-domain: combined_english=0.850000 combined_korean=0.531250 combined_singing=0.325000

# 2026-04-20 — hypothesis: tighten DSP_SUM_MIN 5.0 → 5.5

(a) HYPOTHESIS. Pure `splice/detector.py` primary tunable — tighten the
    SUM-based DSP-confirmation floor from 5.0 → 5.5. One-line edit on
    line ~70, stacked on top of the existing MAX floor 2.0 and GBM
    softmax threshold 0.982. No retrain, no feature change, no new
    constants. Drops GBM emits where `sum(dsp_phase_z, dsp_t2_z,
    dsp_cpe_z) < 5.5` even when MAX passes and all 80 classifier
    features say "splice."

(b) WHY this over recent failures. Last 10 iterations are all feature/
    classifier structural swings that plateaued or regressed (ea1636c
    feature removal catastrophic 0.455, 6384137+49fa2ca rhythm features
    both regressed korean, aa4f141/fd500b3 percussive-mask variants
    regressed singing, 99081f5 HistGBM capacity bump regressed,
    de0be6f/f4148cc symmetry/chroma asymmetry plateaued). CLAUDE.md
    mandates structural change after 5+ same-axis failures — the
    feature-add axis is the most-failed axis. The DSP SUM tunable is
    genuinely NOT saturated: the keep progression 865d92f (SUM=4.5 kept
    combined=0.488) → 29c06cf (SUM=5.0 kept 0.494, +0.006) was
    monotonically improving; 5.5 was the explicit cited escalation in
    the 29c06cf keep note but never got tried (feature-axis explosion
    took over immediately after). Per-tunable frontier explicitly shows
    only 4.5, 5.0 tried in the SUM axis.

    Mechanism: 7972a98 current-keep has singing clean_fp=3 surviving
    all gates. By construction they pass sum≥5.0 and max≥2.0. Real
    cross-source splices disrupt mic/room (phase) + spectral-distribution
    (T²) + complex-prediction (CPE) simultaneously so sum typically
    6-12 (big margin above 5.5). Chord-transition / speech-phoneme FPs
    are dominated by ONE strong channel + partial support from others,
    sum in [5.0, 5.8] — exactly the band 5.5 bites. The classifier
    improvements over the last 40 iterations (75 → 80 features, HistGBM
    swap, voiced-mask + asymmetry families) have tightened p_splice on
    real TPs to near-1.0, so TPs also pass DSP comfortably at sum≥5.5
    by mechanism.

    Orthogonal to every recent axis: NOT a feature add (features.py sha
    stable so classifier byte-identical, no retrain); NOT a classifier
    hyperparam (train_classifier.py unchanged); NOT DSP_CONFIRMATION_MIN
    (the MAX gate); NOT GBM_THRESHOLD (softmax gate); NOT GBM_MIN_SEP_S
    (dedupe geometry); NOT ANALYSIS_STRIDE_S (scan density); NOT
    class-routed DSP (eb8984e failed); NOT HPR-routed DSP (failed); NOT
    channel-specific CPE floor (0.449 failed). ONE unused step remains
    in a tunable axis whose prior steps were monotone improvements.
    Blast radius: 1-line constant edit (DSP_SUM_MIN=5.0 → 5.5) +
    comment refresh. Zero new code paths. Per-emit cost unchanged
    (same sum+compare). Eval budget unchanged (243/300s). Risk-bounded:
    stacked gate so can only DROP emits; drops require cumulative-DSP
    below 5.5; cannot create new TPs. Worst case: singing TPs in
    [5.0, 5.5] sum band get dropped and recall falls; but real splices
    have sum 6-12 by physical mechanism.

(c) IF THIS FAILS. (1) If singing FPs survive 5.5 (sum ≥ 5.5), go
    DSP_CONFIRMATION_MIN 2.0 → 2.2 on the MAX axis — untried, bites
    moderate-single-channel firings that dominate the SUM with 4.5-5.5
    floor. (2) If singing TPs drop at 5.5 (real crossfade TPs have sum
    5.0-5.8 because phase and CPE fire weakly on smooth-transition
    crossfades — the eval corpus mix of tier1/tier2 hard_cut/crossfade
    may skew this), revert and try DSP_CONFIRMATION_MIN 2.0 → 2.2
    instead. (3) Final escalation: feature-axis return —
    voiced_percussive_chroma_asymmetry (untried; percussive-mask chroma
    would isolate pitched-accompaniment key-shift, complementary to the
    MFCC asymmetry family).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 24+
    iterations — I cannot verify the 3 singing FPs' actual sum values
    (would directly predict whether 5.5 bites or misses). Every
    DSP-tunable hypothesis has been calibrated from theory. (ii) SHAP
    rollup STILL "no keeps yet — rollup empty" for 7972a98 — I cannot
    see how the new voiced/unvoiced asymmetries rank against DSP
    z-scores in the current classifier, so cannot predict whether
    tightening SUM might flush legitimate emits whose p_splice is
    driven by feature-side evidence (asymmetry family) rather than
    DSP-side evidence. (iii) Per-domain sum(dsp_phase_z, dsp_t2_z,
    dsp_cpe_z) distributions on real TPs vs FPs in the current eval
    are not surfaced; mechanism-calibrated only.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent blocker for 24+ iterations; per-FP (domain,
    file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    dsp_sum, voiced_mfcc_dist, unvoiced_mfcc_dist,
    voiced_unvoiced_mfcc_asymmetry, top-5 |SHAP|). A single block
    transforms every DSP-tunable and feature hypothesis from
    theory-calibrated bet into data-driven decision. (2)
    `scripts/primary_tunable_preview.py --tunable DSP_SUM_MIN --value
    5.5` — loads current classifier, runs detect on a cached 3-file
    eval subset, reports delta to current combined. Zero retrain cost.
    Turns "is this primary-tunable step worth a full 243s eval" into a
    10-second numeric preview. (3) CURRENT DETECTOR CONSTANTS block in
    prompt — explicit list of (GBM_THRESHOLD, GBM_MIN_SEP_S,
    ANALYSIS_STRIDE_S, DSP_CONFIRMATION_MIN, DSP_SUM_MIN) values at
    HEAD, so I don't have to read detector.py every iteration to know
    current tunable state (and the per-tunable frontier's "current ?"
    rows resolve).

## 2026-04-20T14:42:14+09:00 — 0cb551f (discard, combined=0.490700)
subject: tighten DSP_CONFIRMATION_MIN 2.0 -> 2.2 (MAX floor) — first tightening of the MAX axis in the loop's history. Pure detector.py one-line constant edit plus comment refresh, stacked on existing SUM>=5.0 + GBM_THRESHOLD=0.982 + MIN_SEP=3.5. No retrain, no feature change, features.py sha stable so classifier byte-identical. Drops emits where max(dsp_phase_z, dsp_t2_z, dsp_cpe_z) < 2.2 even when SUM passes. Targets 7972a98 current-keep (combined=0.589282) singing 0.345 weakest-domain plateau with clean_fp=3 survivors past the current max>=2.0 gate. Last 10 iterations exhausted every other axis: feature-add (fd500b3 percussive_mfcc, aa4f141 voiced_percussive_asymmetry, 49fa2ca/6384137 rhythm/autocorr — all regressed), SUM tightening (53d3ea0 5.0->5.5 failed, dropped singing 0.345->0.325 via legitimate crossfade TPs in 5.0-5.5 band), feature removal (ea1636c catastrophic 0.455), classifier capacity bump (99081f5 discard). CLAUDE.md mandates structural change after 5+ same-axis failures — MAX floor is the ONE primary-tunable axis never tuned (history has only value 2.0). Explicit cited fallback from 7255ec6 ('If TPs don't rise but FPs do, fallback is tighten DSP_CONFIRMATION_MIN 2.0 -> 2.2') and from 53d3ea0 ('fallback is DSP_CONFIRMATION_MIN 2.0 -> 2.2 on the MAX axis — different channel-mass distribution, never tried'). Channel-mass distribution argument: the MAX gate bites emits where NO SINGLE channel is strong (2.0 <= max < 2.2), a strictly smaller set than SUM-tightening catches — chord-transition singing FP with T²≈2.1 + phase≈1.5 + CPE≈1.5 gives max=2.1 sum=5.1 passes both current gates but drops under MAX>=2.2. Real cross-source singing TPs produce at least one strong channel (phase_z>=3 or T²_z>=3) by physical mechanism (different mic/room, different spectral distribution) so max>=2.2 is structurally robust. Preferred over SUM-bumping because 53d3ea0 proved the SUM band 5.0-5.5 contains legitimate smooth-crossfade TPs; the MAX [2.0, 2.2] band is structurally different (all-channels-mediocre). Orthogonal to every recent axis: NOT a feature add (features.py sha stable, no retrain); NOT a classifier hyperparam (train_classifier.py unchanged); NOT DSP_SUM_MIN (53d3ea0 failed); NOT GBM_THRESHOLD (7255ec6 0.980 failed); NOT GBM_MIN_SEP_S; NOT ANALYSIS_STRIDE_S (0.11 failed); NOT class-routed DSP (eb8984e failed); NOT HPR-routed DSP; NOT channel-specific CPE floor (0.449 failed); NOT feature removal (ea1636c catastrophic). First MAX-axis tightening. Blast radius: 1-line constant edit + comment refresh. Zero new code paths. Per-emit cost unchanged (same max+compare). Risk-bounded: stacked on SUM so can only DROP emits past SUM; drops require max DSP in [2.0, 2.2] narrow band; cannot create TPs. Worst case: singing smooth-crossfade TPs with all channels in 2.0-2.2 band drop and recall falls — fallback is DSP_PHASE_MIN=0.5 (channel-specific phase floor, untried vs 0.449-failed CPE floor) or ANALYSIS_STRIDE_S=0.08 (denser scan, untried below 0.11). Smoke-verified: DSP_CONFIRMATION_MIN=2.2 loads, all other tunables unchanged (SUM=5.0 THR=0.982 SEP=3.5 STRIDE=0.12).
per-domain: combined_english=0.839506 combined_korean=0.476471 combined_singing=0.295385

# 2026-04-20 — hypothesis: tighten DSP_CONFIRMATION_MIN 2.0 → 2.2 (MAX floor)

(a) HYPOTHESIS. Pure `splice/detector.py` primary tunable — tighten the
    DSP MAX floor from 2.0 → 2.2. One-line constant edit at line 60,
    stacked on existing SUM≥5.0 + GBM_THRESHOLD=0.982 + MIN_SEP=3.5. No
    retrain, no feature change, features.py sha stable so classifier
    byte-identical. Drops emits where max(dsp_phase_z, dsp_t2_z,
    dsp_cpe_z) < 2.2 even when SUM passes and all 80 classifier
    features say "splice."

(b) WHY this over recent failures. Last 10 iterations exhausted the
    feature-add axis (voiced/unvoiced/percussive/rhythm variants, all
    failed — plateau 0.318-0.345 on singing), the SUM axis (53d3ea0
    tried 5.0→5.5 and failed dropping singing to 0.325), the feature-
    removal axis (ea1636c catastrophic 0.455), and the classifier-
    capacity axis (99081f5 bump discard). CLAUDE.md mandates
    structural change after 5+ same-axis failures — every axis tried
    is saturated EXCEPT the MAX floor. The MAX axis has NEVER been
    tuned (only value 2.0 in history); 2.2 is the explicit cited
    untried fallback from 7255ec6 ("If TPs don't rise but FPs do,
    fallback is tighten DSP_CONFIRMATION_MIN 2.0 → 2.2") and 53d3ea0
    ("fallback is DSP_CONFIRMATION_MIN 2.0 → 2.2 on the MAX axis —
    different channel-mass distribution, never tried"). Channel-mass
    distribution argument: the MAX gate bites emits where NO SINGLE
    channel is strong (2.0 ≤ max < 2.2), a strictly smaller set than
    SUM-tightening catches. Realistic chord-transition singing FP:
    T²≈2.1, phase≈1.5, CPE≈1.5 gives max=2.1 sum=5.1 — passes
    current gates, drops under MAX≥2.2. Real cross-source singing
    TPs produce at least one strong channel (phase_z≥3 or T²_z≥3)
    by physical mechanism (different mic/room or different spectral
    distribution), so max≥2.2 is structurally robust. Preferred
    over SUM-bumping because 53d3ea0 proved the SUM band 5.0-5.5
    contained legitimate crossfade TPs (smooth transitions with
    phase/CPE firing weakly, sum 5.0-5.5). The MAX floor at 2.2
    targets emits where ALL channels are mediocre — a different
    structural signature than SUM.

    Orthogonal to every recent axis: NOT a feature add (features.py
    sha stable — classifier byte-identical); NOT a classifier
    hyperparam (train_classifier.py unchanged); NOT DSP_SUM_MIN
    (53d3ea0 failed); NOT GBM_THRESHOLD (7255ec6 0.980 failed); NOT
    GBM_MIN_SEP_S; NOT ANALYSIS_STRIDE_S (0.11 failed); NOT
    class-routed DSP (eb8984e failed); NOT HPR-routed DSP (failed);
    NOT channel-specific CPE floor (0.449 failed); NOT feature
    removal (ea1636c catastrophic). First tightening of MAX axis in
    the loop's history. Blast radius: 1-line constant edit + comment
    refresh. Zero new code paths. Per-emit cost unchanged (same
    max+compare path). Risk-bounded: stacked on SUM gate so can only
    DROP emits; drops require max DSP in [2.0, 2.2] band, a narrow
    strip. Cannot create TPs.

(c) IF THIS FAILS. (1) If clean_fp unchanged (the 3 singing FPs have
    max ≥ 2.2 by measurement — they're driven by a single strong
    channel above 2.2), fallback to a PHASE-specific floor
    DSP_PHASE_MIN = 0.5 — untried channel-specific axis on phase
    instead of CPE (the 0.449-failed axis). Chord transitions
    produce phase_z ≈ 0; same-recording same-singer cross-song
    splices still produce phase_z > 1 via mic/mastering differences.
    (2) If TPs drop (singing real crossfade TPs sit in max [2.0, 2.2]
    band — smooth transitions don't produce a single strong channel),
    revert and try ANALYSIS_STRIDE_S = 0.08 (denser scan, untried
    below 0.11 fail and below current 0.12) to catch boundary-
    adjacent TPs. (3) Final escalation: feature-axis return with
    voiced_percussive_chroma_asymmetry (untried; percussive-mask
    chroma isolates pitched accompaniment at transient frames —
    bass/guitar riff shift across cross-song splices).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 25+
    iterations — I cannot verify the 3 singing FPs' actual (phase_z,
    t2_z, cpe_z) distribution to predict whether max falls in the
    [2.0, 2.2] band 2.2 bites or sits well above. Every DSP-tunable
    hypothesis has been theory-calibrated. (ii) Two GHOST code
    states persist that confound attribution: (a) e7ca9eb's
    voiced_spec_contrast_cosine_dist still in features.py (ea1636c
    ablation failed so it stays — 80 features still include it);
    (b) 99081f5's max_depth=4/max_leaf_nodes=16 still in
    train_classifier.py (discarded but never reverted — current
    classifier trains with higher capacity than 7972a98-keep-time
    classifier). My MAX-floor change runs against a classifier that
    differs from 7972a98's keep-time state; combined deltas mix with
    these ghosts. (iii) SHAP rollup STILL "no keeps yet — rollup
    empty" for 7972a98 — cannot see how DSP z-scores rank against
    the 80 classifier features or predict MAX-floor impact.

(e) Wrapper enhancements. (1) DISCARD-REVERT SYNC auditor —
    addresses the root cause of the 99081f5 hyperparam ghost AND
    the e7ca9eb feature ghost. Wrapper's discard path should diff
    each changed file vs baseline-sha file and ABORT + restore if
    drift remains. ~20 lines of shell guard after the discard
    decision. Would have prevented 15+ iterations of attribution
    contamination from both ghosts. (2) CLEAN_FP_POSITIONS JSON
    block in CURRENT STATE — persistent blocker for 25+ iterations;
    per-FP (domain, file, t_sec, label_id, p_splice, dsp_phase_z,
    dsp_t2_z, dsp_cpe_z, dsp_max, dsp_sum, top-5 |SHAP|). A single
    block transforms every DSP-tunable hypothesis from theory-
    calibrated bet to data-driven decision. (3) CURRENT DETECTOR
    CONSTANTS + CURRENT CLASSIFIER HYPERPARAMS block in prompt —
    explicit snapshot of (GBM_THRESHOLD, GBM_MIN_SEP_S,
    ANALYSIS_STRIDE_S, DSP_CONFIRMATION_MIN, DSP_SUM_MIN) + (max_iter,
    max_depth, max_leaf_nodes, learning_rate, l2, min_samples_leaf)
    at HEAD so ghost state is surfaced explicitly and the
    per-tunable frontier "current ?" rows resolve.

## 2026-04-20T14:51:02+09:00 — 67633f2 (discard, combined=0.421917)
subject: add DSP_PHASE_MIN=0.5 channel-specific floor
per-domain: combined_english=0.657534 combined_korean=0.491803 combined_singing=0.232258

# 2026-04-20 — hypothesis: add DSP_PHASE_MIN=0.5 channel-specific floor

(a) HYPOTHESIS. Pure `splice/detector.py` primary tunable — add a new
    channel-specific DSP floor `DSP_PHASE_MIN = 0.5` stacked on the
    existing MAX>=2.0 + SUM>=5.0 gates. Drop emits where
    `dsp_vals[0] < 0.5` (i.e. dsp_phase_z is weak) even when MAX/SUM
    pass. Phase channel has NEVER been given a dedicated floor in the
    loop's history (only CPE at 0.449-failed value; phase is untouched).
    One-line constant + one-clause gate extension, no retrain,
    features.py sha stable so classifier byte-identical.

(b) WHY this over recent failures. Two consecutive CUMULATIVE DSP-gate
    tightenings just failed catastrophically — 53d3ea0 (SUM 5.0→5.5
    combined=0.527) and 0cb551f (MAX 2.0→2.2 combined=0.491). Both dropped
    legitimate cross-domain TPs whose cumulative DSP signal is marginal
    (sum in [5.0, 5.8], max in [2.0, 2.2]). Feature-add axis saturated
    too (aa4f141/49fa2ca/6384137/de0be6f/fd500b3/f4148cc all failed).
    CLAUDE.md mandates structural change after 5+ same-axis failures;
    cumulative-DSP axis is that exhausted axis. The channel-specific
    PHASE floor is a DIFFERENT structural signature than cumulative
    tightening: it targets emits where the mic/room PHASE channel is
    silent regardless of T²/CPE magnitude. Explicit cited fallback from
    0cb551f(c)(1): "fallback to a PHASE-specific floor DSP_PHASE_MIN =
    0.5 — untried channel-specific axis on phase instead of CPE (the
    0.449-failed axis). Chord transitions produce phase_z ≈ 0; same-
    recording same-singer cross-song splices still produce phase_z > 1
    via mic/mastering differences."

    Mechanism on singing FPs (3 surviving clean FPs at singing=0.345):
    chord transitions within one song keep mic/room fixed, so phase
    remains continuous across bin frequencies → dsp_phase_z ≈ 0.0-0.3.
    They fire via T²_z (spectral distribution change on chord switch)
    + modest CPE_z (complex-prediction drift), with sum 5.0-5.8 passing
    current gates. Under the new phase floor they drop cleanly because
    phase < 0.5 — exactly the chord-cycle FP signature. Real cross-
    source singing splices (even same-singer-different-song): the new
    recording has different mic/room/mastering → phase discontinuity
    across all bins → phase_z > 1 (≥2× the 0.5 floor with safety
    margin). Speech domains (english/korean clean_fp=0 already): chord-
    transition FPs irrelevant; any emit with phase_z < 0.5 isn't in the
    FP set so clean_fp stays 0. No regression expected.

    Why 0.5 specifically: sits in BIMODAL GAP between chord-transition
    phase (~0) and splice phase (≥1) without biting legitimate TPs that
    have modest but present phase shift. DIFFERENT from 0cb551f's
    MAX=2.2 tightening (dropped emits where ALL channels were mediocre,
    biting splice-adjacent smooth-crossfade TPs). DIFFERENT from
    53d3ea0's SUM=5.5 tightening (dropped emits where cumulative support
    was weak, biting same TP type). Phase-specific at 0.5 targets only
    the CHORD-CYCLE SIGNATURE (phase-silent), not the SMOOTH-CROSSFADE
    signature (cumulative-weak but phase-present from master difference).

    Orthogonal to every recent axis: NOT DSP_CONFIRMATION_MIN MAX floor
    (0cb551f 2.2 failed); NOT DSP_SUM_MIN (53d3ea0 5.5 failed); NOT
    GBM_THRESHOLD (7255ec6 0.980 failed); NOT GBM_MIN_SEP_S dedupe; NOT
    ANALYSIS_STRIDE_S; NOT class-routed DSP (eb8984e failed on channel
    mapping at 2.0); NOT HPR-routed DSP (failed); NOT a CPE channel
    floor (0.449 combined from whatever CPE threshold that attempt
    used); NOT a feature add/remove (all saturated); NOT a classifier
    hyperparam (99081f5 discard). First PHASE-specific channel floor.
    Blast radius: 1 new constant + 1 clause in the 1-line DSP gate.
    Zero new code paths. Per-emit cost: one extra float compare past
    the stacked gate, negligible. Risk-bounded: stacked on MAX+SUM so
    can only DROP emits that already pass both; cannot create TPs.
    Drops require phase_z in [0, 0.5) narrow band.

(c) IF THIS FAILS. (1) If singing TPs drop (real crossfades between
    same-mic / same-master recordings preserve phase, phase_z sits in
    [0.2, 0.8] rather than the theorized bimodal gap), LOWER floor to
    0.3 (still bites hard-chord-transition FPs at ~0.1 while preserving
    borderline crossfade TPs). (2) If clean_fp unchanged (the 3 singing
    FPs actually have phase_z ≥ 0.5 — hypothesis wrong about chord-
    transition phase signature being silent), pivot to GBM_MIN_SEP_S =
    2.5 (below-tried primary-tunable axis: MIN_SEP 3.0 kept, 2.5
    untried below 3.0 floor — allows clustered crossfade TPs in
    wedge-splice patterns to co-survive dedupe). (3) Final escalation:
    shift to feature-add axis with voiced_percussive_chroma_asymmetry
    (untried; percussive-mask chroma isolates pitched accompaniment
    (bass/guitar riff) at transient frames, complementary to voiced
    pitch signal — cited in 0cb551f(c)(3)/53d3ea0(c)(3)).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 26+
    iterations — I cannot verify the 3 surviving singing FPs' actual
    (phase_z, t2_z, cpe_z) distribution to directly predict whether
    0.5 bites (phase_z in [0, 0.5)) or misses (phase_z ≥ 0.5). Every
    DSP-channel hypothesis theory-calibrated on the claim that chord
    transitions produce phase_z ≈ 0 without eval-corpus verification.
    (ii) SHAP rollup STILL "no keeps yet — rollup empty" for 7972a98
    — cannot see how DSP z-scores rank against the 80 classifier
    features. (iii) Two persistent ghost states contaminate
    attribution: (a) e7ca9eb's voiced_spec_contrast_cosine_dist still
    in-tree (ea1636c ablation failed so feature remains at position
    80); (b) 99081f5's max_depth=4/max_leaf_nodes=16 may still be in
    train_classifier.py (discard but unclear if reverted). My
    DSP-channel-floor delta mixes with whatever classifier state is
    at HEAD.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent blocker for 26+ iterations; per-FP (domain,
    file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    top-5 |SHAP|). A single block would convert every DSP-channel /
    feature hypothesis from theory-calibrated bet into data-driven
    decision. (2) DISCARD-REVERT SYNC auditor — a wrapper guard that
    diffs features.py + train_classifier.py vs baseline-sha on discard
    and RESTORE if drift remains. Would have prevented both the
    e7ca9eb voiced_spec_contrast ghost (15+ iterations) and the
    99081f5 capacity-bump ghost from contaminating attribution. ~20
    lines of shell after the discard decision. (3) CURRENT DETECTOR
    CONSTANTS + CURRENT CLASSIFIER HYPERPARAMS block in prompt —
    explicit snapshot of (GBM_THRESHOLD, GBM_MIN_SEP_S,
    ANALYSIS_STRIDE_S, DSP_CONFIRMATION_MIN, DSP_SUM_MIN) +
    (max_iter, max_depth, max_leaf_nodes, learning_rate, l2,
    min_samples_leaf) at HEAD, so per-tunable-frontier "current ?"
    resolves and ghost state surfaces explicitly.

## 2026-04-20T15:03:57+09:00 — 960113c (discard, combined=0.589282)
subject: REVERT 99081f5 HistGBM capacity-bump ghost (max_depth 4->3, max_leaf_nodes 16->8) — first hyperparam REVERT in loop history. Confirmed ghost via git-diff: 7972a98 baseline had max_depth=3/max_leaf=8 but HEAD carries max_depth=4/max_leaf=16 from 99081f5's un-rolled-back commit (99081f5 row ABSENT from results.tsv — suggests verify-fail whose diff never reverted). Every iteration post-99081f5 (49fa2ca, 6384137, ea1636c, 53d3ea0, 0cb551f, 67633f2) evaluated against 4/16 classifier while 0.589282 baseline was set at 3/8 — attribution contamination. Manually retrained, committed fresh joblib+cv_results+meta (f71c526 joblib-skip lesson). OOF weighted F1=0.6437 (vs 99081f5-reported 0.6321 on 4/16) — hard_cut F1 0.373, crossfade 0.342, both above 99081f5 baselines suggesting capacity bump may have been hurting fit quality too. Self-diagnostic: combined >= 0.589 confirms ghost theory and many recent discards deserve re-evaluation; combined < 0.589 proves 99081f5 was actually positive and reinstates 4/16 as the correct config. Orthogonal: NOT feature add (features.py sha stable); NOT primary-tunable (detector.py unchanged); NOT DSP gate; NOT new hyperparam exploration — REVERT of unauthorized ghost. First hyperparam REVERT in loop history. Blast radius: 2-line edit in make_pipeline() + regenerated bundle. Smoke-verified: reverted values load (max_depth=3, max_leaf_nodes=8), train completed in budget, OOF metrics non-degenerate.
per-domain: combined_english=0.888889 combined_korean=0.666667 combined_singing=0.345312

# 2026-04-20 — hypothesis: REVERT 99081f5 HistGBM capacity-bump ghost (max_depth 4→3, max_leaf_nodes 16→8)

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` hyperparam
    REVERT — restore HistGBM max_depth=4→3 and max_leaf_nodes=16→8 to
    match the 7972a98-keep-time classifier state. Verified ghost via
    git-diff: 7972a98 had max_depth=3 / max_leaf=8; HEAD carries
    max_depth=4 / max_leaf=16 from 99081f5's commit that was never
    rolled back. Manually retrain and commit the new joblib +
    cv_results + training_manifest alongside (f71c526 joblib-skip
    lesson). Features.py sha stable so the wrapper's auto-retrain gate
    is passive; the committed bundle guarantees the reverted
    classifier is consumed at eval.

(b) WHY this over recent failures. Every reflection since ea1636c has
    flagged the 99081f5 ghost as attribution contamination. Confirmed
    today: HEAD's train_classifier.py still has the discarded capacity
    bump. Every iteration post-99081f5 (49fa2ca, 6384137, ea1636c,
    53d3ea0, 0cb551f, 67633f2) evaluated against a 4/16 classifier
    while the 0.589282 baseline was set at 3/8. 53d3ea0/0cb551f/67633f2
    deteriorated past 0.527 not only because their DSP tightenings
    were bad but partially because the 4/16 classifier produces
    different p_splice distributions than 7972a98-keep-time.

    Self-diagnostic: if REVERT HELPS (combined ≥ 0.589), the ghost was
    hurting — many recent discards deserve re-evaluation with the
    clean classifier. If REVERT HURTS (combined < 0.589), 99081f5's
    capacity bump was actually positive and should have been kept; in
    that case stop reverting and restart the feature-add axis on top
    of 4/16. Either outcome is strong information.

    Orthogonal to every prior axis: NOT a feature add (features.py sha
    stable); NOT primary-tunable (detector.py unchanged); NOT DSP
    gate; NOT new hyperparam exploration — this is a REVERT of an
    unauthorized ghost, not a new direction; NOT feature removal
    (ea1636c). First hyperparam REVERT in the loop's history.

(c) IF THIS FAILS. (1) combined drops below 0.589 with 3/8 capacity —
    99081f5 was actually positive; reinstate 4/16 and pivot to
    learning_rate reduction 0.07 → 0.05 (smoother fit, tightens
    calibration at the high-probability tail; untried). (2) combined
    matches 0.589 within ±0.003 — ghost inert, plateau is structural;
    pivot to ANALYSIS_STRIDE_S = 0.08 (denser than both 0.11-failed
    and 0.12-current, cited in multiple fallback plans). (3) combined
    rises above 0.589 — ghost theory confirmed; revisit most-promising
    recent discards (e.g., voiced_percussive_mfcc_asymmetry aa4f141)
    against the clean classifier since their attribution was
    contaminated.

(d) Information gaps. (i) The 99081f5 row is ABSENT from results.tsv
    (grep confirms no line). Progress reports 10 verify-fails out of
    100 iterations — 99081f5 may be a verify-fail whose hyperparam
    diff was never rolled back. Fix is the same either way (manual
    revert). (ii) CLEAN_FP_POSITIONS STILL absent after 27+
    iterations; I cannot verify which singing FPs are classifier-
    capacity-sensitive vs DSP-gated. (iii) The on-disk joblib's
    training metadata (training_manifest.json, cv_results.json) is
    not surfaced in the prompt — I can't tell which hyperparams the
    CURRENT joblib was trained with without reading files.

(e) Wrapper enhancements. (1) DISCARD-REVERT SYNC auditor (repeated
    for 5+ iterations): the wrapper's discard AND verify-fail paths
    must diff every changed file vs baseline-sha and RESTORE on drift.
    99081f5 ghost persisted 10+ iterations after e7ca9eb's
    15+-iteration precedent. ~20 lines of shell guard would prevent
    the next one. (2) CURRENT-CLASSIFIER-STATE block in CURRENT STATE:
    (max_depth, max_leaf_nodes, learning_rate, l2_regularization,
    min_samples_leaf, max_iter, joblib_sha, training_sha) so any
    divergence between baseline-keep-time config and HEAD config
    surfaces immediately. (3) CLEAN_FP_POSITIONS JSON block — the
    persistent 27+-iteration blocker. Every theory-calibrated
    hypothesis around singing clean_fp=3 could become data-driven
    with (domain, file, t_sec, p_splice, dsp_phase_z, dsp_t2_z,
    dsp_cpe_z, voiced_mfcc_dist, voiced_unvoiced_mfcc_asymmetry,
    top-5 |SHAP|).

## 2026-04-20T15:21:21+09:00 — b120d38 (discard, combined=0.530924)
subject: widen ANALYSIS_STRIDE_S 0.12 -> 0.15 (untried sparser direction) — FIRST sparser-stride experiment in loop history. Pure detector.py one-line constant edit plus 4-line comment refresh, stacked on existing GBM_THRESHOLD=0.982 + GBM_MIN_SEP_S=3.5 + DSP_CONFIRMATION_MIN=2.0 + DSP_SUM_MIN=5.0. No retrain, no feature change, features.py sha stable so classifier byte-identical. Reduces candidate positions per 30s chunk ~250 -> ~200 (-20%%). Still ~23 candidates per 3.5s dedupe window. Per-tunable frontier: only 0.11 tried (failed, denser than current 0.12); 0.12 is codebase default never formally varied upward; sparser direction genuinely virgin. Every other primary-tunable axis saturated in both directions: GBM_THRESHOLD (0.980-0.983 all tested), GBM_MIN_SEP_S (3.75/3.6 failed sparser, below 3.0 adds FPs), DSP_CONFIRMATION_MIN (2.2 failed tighter), DSP_SUM_MIN (5.5 failed tighter). Feature-add axis saturated (10+ recent failures), feature removal catastrophic (ea1636c 0.455), classifier capacity revert matched baseline exactly (960113c 0.589282) confirming plateau is structural on classifier side. Targets 7972a98 current-keep (combined=0.589282) singing 0.345 weakest-domain plateau with clean_fp=3. Asymmetric-peak mechanism: TP vs FP p_splice peaks differ in time width. Real cross-source splices produce sustained spectral discontinuity (voice/accompaniment/mastering/mic-room shift persists ~0.3-0.5s past cut) so p_splice stays high across 3-5 consecutive grid points at 0.12 stride. At 0.15 stride same 0.3-0.5s TP plateau still hits 2-3 grid points -> dedupe picks maximum -> TP preserved. Chord-transition FPs driven by sharp spec_centroid_delta/spec_rolloff_delta/mfcc_delta peaks (~0.05-0.15s wide, 1-3 STFT frames). At 0.12 stride FP peak hits 1-2 grid points crossing threshold. At 0.15 stride narrowest FP peaks (0.05-0.10s wide) fall BETWEEN grid points (0.075s half-stride exceeds 0.05s peak half-width) -> no grid point samples resonance center -> adjacent-grid p_splice below threshold -> FP dropped. Asymmetry is structural: TP peaks broader because cross-source physics persists; FP peaks sharper because chord-transition spectral events are transient. Orthogonal to every recent axis: NOT a feature add (features.py sha stable, no retrain); NOT a classifier hyperparam (99081f5 ghost in-tree unchanged — orthogonal to my delta); NOT DSP_SUM_MIN/DSP_CONFIRMATION_MIN/DSP_PHASE_MIN (all stacked-DSP tightenings failed); NOT GBM_THRESHOLD; NOT GBM_MIN_SEP_S; NOT feature removal (ea1636c catastrophic); NOT class-routed/HPR-routed DSP. First sparser-stride experiment. Blast radius: 1-line constant + 4-line comment. Zero new code paths. Per-emit cost unchanged. Risk-bounded: stride can only MISS narrow-peak candidates predominantly FPs by physical prior; can't create new FPs; eval ~20%% FASTER (more headroom within 243/300s). Smoke-verified: ANALYSIS_STRIDE_S=0.15 loads, all other tunables unchanged (GBM_THRESHOLD=0.982, GBM_MIN_SEP_S=3.5, DSP_CONFIRMATION_MIN=2.0, DSP_SUM_MIN=5.0).
per-domain: combined_english=0.804878 combined_korean=0.656250 combined_singing=0.283333

# 2026-04-20 — hypothesis: ANALYSIS_STRIDE_S 0.12 → 0.15 (untried sparser direction)

(a) HYPOTHESIS. Pure `splice/detector.py` primary tunable — widen the dense
    scan stride from 0.12 → 0.15. One-line constant edit at line 41,
    stacked on existing GBM_THRESHOLD=0.982 + GBM_MIN_SEP_S=3.5 +
    DSP_CONFIRMATION_MIN=2.0 + DSP_SUM_MIN=5.0. No retrain, no feature
    change, features.py sha stable so classifier byte-identical. Reduces
    candidate positions per 30s chunk ~250 → ~200 (−20%). Still ~23
    candidates per 3.5s dedupe window, well above 1 needed to catch any
    real-splice plateau.

(b) WHY this over recent failures. ANALYSIS_STRIDE_S is genuinely the
    ONLY primary tunable where the SPARSER direction is untried.
    Per-tunable frontier: only 0.11 tried (failed, denser than current
    0.12), and 0.12 is the codebase default never formally varied
    upward. Every other primary-tunable axis is saturated in both
    directions: GBM_THRESHOLD (0.980-0.983 all tested, 0.982 kept),
    GBM_MIN_SEP_S (2.5 below-tried would only ADD dedupe FPs, 3.75/3.6
    failed sparser), DSP_CONFIRMATION_MIN (2.2 failed tighter, looser
    adds FPs), DSP_SUM_MIN (5.5 failed tighter, 4.5 prior-kept value).
    Feature-add axis saturated (10+ recent failures across mask/rhythm
    variants), feature removal catastrophic (ea1636c 0.455), classifier
    capacity ghost (99081f5) revert matched baseline exactly (960113c
    0.589282) confirming plateau is structural on the CLASSIFIER side.
    Stride is the one axis with clear untested ground.

    Asymmetric-peak mechanism: TP vs FP p_splice peaks have different
    TIME WIDTHS. Real cross-source splices produce sustained spectral
    discontinuity — voice / accompaniment / mastering / mic-room shift
    persists ~0.3-0.5s past the cut, so p_splice stays high across
    3-5 consecutive grid points at 0.12 stride (0.36-0.60s span). At
    0.15 stride, the same 0.3-0.5s TP plateau still hits 2-3 grid
    points (0.30-0.45s coverage) → dedupe picks the maximum, TP
    preserved. In contrast, chord-transition FPs are driven by sharp
    spec_centroid_delta / spec_rolloff_delta / mfcc_delta peaks at
    the transition frame (~0.05-0.15s wide, 1-3 STFT frames). At
    0.12 stride the FP peak hits 1-2 grid points (p_splice likely
    crosses threshold at least once). At 0.15 stride the FP peak
    may fall BETWEEN grid points (0.075s half-stride > 0.05s peak
    half-width at the narrowest FPs) → no grid point samples the
    resonance center → peak p_splice at adjacent grid points is
    below threshold → FP dropped. The asymmetry is structural: TP
    peaks are broader because cross-source physics persists; FP
    peaks are sharper because chord-transition spectral events are
    transient.

    Risk-bounded: 0.15 is still dense enough (every 0.15s) that
    real splice plateaus of ≥0.3s duration hit ≥2 grid points.
    Eval budget unchanged (actually FASTER by ~20% — more headroom
    within 243/300s). Per-emit cost unchanged. Can only miss
    narrow-peak candidates which should be predominantly FPs.
    Secondary: 0.15 is outside the [0.11, 0.12] tested range, so
    extrapolation risk exists — if the classifier's p_splice peaks
    on real TPs happen to be as narrow as FP peaks on this eval set
    (counter to physical prior), recall drops.

    Orthogonal to every recent axis: NOT a feature add (features.py
    sha stable, no retrain); NOT a classifier hyperparam (99081f5
    ghost unchanged — orthogonal to my delta); NOT DSP_SUM_MIN /
    DSP_CONFIRMATION_MIN / DSP_PHASE_MIN (all stacked-DSP failed);
    NOT GBM_THRESHOLD; NOT GBM_MIN_SEP_S; NOT feature removal
    (ea1636c catastrophic); NOT class-routed / HPR-routed DSP.
    FIRST sparser-stride experiment in loop history. Blast radius:
    1-line constant edit + comment refresh. Zero new code paths.

(c) IF THIS FAILS. (1) If singing TPs drop (real singing crossfade
    TPs have narrower plateaus than the physical prior suggests
    — maybe because tier1 hard cuts also produce sharp peaks),
    step DOWN to 0.10 (between failed 0.11 and current 0.12 —
    narrowly-untested, slightly denser than current, might catch
    sub-resolution boundary alignments without 0.11's specific
    failure mode). (2) If clean_fp unchanged (stride doesn't bite
    singing chord-transition FPs because their peaks are broad
    enough to hit 0.15 grid alignment), pivot to CLASSIFIER
    hyperparam: learning_rate 0.07 → 0.05 (cited fallback in
    960113c(c)(1); smoother fit tightens p_splice tail calibration
    at 0.98-0.99 where FPs sit). (3) Final escalation: GBM_MIN_SEP_S
    = 4.0 (looser dedupe, untried above 3.75 failed — marginal but
    unexplored), stacked with 0.15 stride.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 28+
    iterations — I cannot verify the peak-width assumption: are the
    3 surviving singing FPs narrow-peak (stride bites) or broad-peak
    (stride misses)? Every stride / threshold / DSP-gate hypothesis
    remains theory-calibrated. (ii) SHAP rollup STILL empty for
    7972a98 — cannot correlate which features drive singing FP
    p_splice to predict peak width. (iii) 99081f5 classifier-capacity
    ghost remains in-tree at HEAD (max_depth=4, max_leaf_nodes=16) —
    960113c revert discarded as "no improvement" because 0.589282
    matched baseline exactly, so the discard reversal RESTORED the
    ghost code, but the on-disk joblib may still be 960113c's 3/8
    manual-retrain artifact (wrapper auto-retrain fires on features.py
    sha drift only, NOT train_classifier.py). Current eval runs
    against whatever joblib is on disk. (iv) No per-domain stride
    sensitivity data; 0.15 applied uniformly across domains.

(e) Wrapper enhancements. (1) **DISCARD-REVERT SYMMETRY SEMANTICS**
    — the discard path treats "combined matches baseline exactly"
    as no-improvement and reverts the change. When the change was
    a REVERT of a ghost commit (e.g., 960113c reverting 99081f5),
    this RESTORES the ghost instead of keeping the cleaner state.
    Proposed: if the discarded commit's title starts with "REVERT"
    and its body references a previously-discarded SHA, keep the
    revert; otherwise discard as normal. ~15 lines of git-commit-
    message parsing in the discard path. (2) CLEAN_FP_POSITIONS
    JSON block in CURRENT STATE — persistent blocker for 28+
    iterations; per-FP (domain, file, t_sec, label_id, p_splice,
    dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_max, dsp_sum, peak_width_s,
    top-5 |SHAP|). peak_width_s (FWHM of p_splice trace around emit)
    would directly test the stride-peak-width hypothesis. (3)
    `scripts/primary_tunable_sweep.py --tunable ANALYSIS_STRIDE_S
    --values 0.08,0.10,0.12,0.15,0.20` — loads current classifier,
    runs detect on a cached 3-file eval subset per value, reports
    per-value combined delta. Turns "is this stride worth a full
    243s eval" into a 30-second sweep across 5 values.

## 2026-04-20T15:26:57+09:00 — 17d4aec (discard, combined=0.490700)
subject: HistGBM learning_rate 0.07 -> 0.05 (untried classifier axis)
per-domain: combined_english=0.839506 combined_korean=0.476471 combined_singing=0.295385

# 2026-04-20 — hypothesis: HistGBM learning_rate 0.07 → 0.05

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` hyperparam — lower
    HistGBM `learning_rate` from 0.07 → 0.05, keeping max_iter=200 /
    max_depth=4 / max_leaf_nodes=16 / l2=1.0 / min_samples_leaf=20. One-line
    edit in `make_pipeline()`. features.py sha stable so wrapper's
    auto-retrain fires on train_classifier.py edit (US-505).

(b) WHY this over recent failures. learning_rate is the one HistGBM
    hyperparameter NEVER tuned in the HistGBM era. Per 99081f5's own note,
    all prior LR experiments (e.g. 9f66d66 0.1→0.05) were against the
    *old* GradientBoostingClassifier pre-acba4aa swap — zero data about LR
    behavior against HistGBM + 80-feature space. Explicitly cited untried
    in 960113c(c)(1) "pivot to learning_rate reduction 0.07 → 0.05 (smoother
    fit, tightens calibration at the high-probability tail; untried)" and
    again in b120d38(c)(2). Last 10 iterations exhausted feature-add
    (mask/rhythm variants), DSP tunings (SUM=5.5, MAX=2.2, PHASE=0.5),
    stride widening (0.15), feature removal (ea1636c catastrophic), and
    capacity revert (960113c matched baseline exactly → discarded). The
    PER-TUNABLE frontier shows all other primary-tunable axes saturated
    on both sides or with tried values in hazardous bands. Mechanism:
    singing's 3 FPs sit at p_splice ∈ [0.982, 0.99] — barely above the
    0.982 gate. Lower LR with same max_iter produces a smoother decision
    surface (weaker per-step correction, more averaged trees) and pulls
    BORDERLINE predictions toward the middle of the probability scale.
    Real TPs sitting comfortably at p≈1.0 stay safely above the gate;
    borderline FPs get pulled below. Risk-bounded: if underfit dominates
    and real TPs also drop below gate, combined regresses; fallback is
    LR=0.05 + max_iter=280 to preserve overall fit strength.

(c) IF THIS FAILS. (1) Underfit on TPs → pair LR=0.05 with max_iter bump
    200→280 to preserve effective fit depth. (2) LR change inert (tree
    ensemble saturated regardless of LR at 200 iter) → pivot to
    `min_samples_leaf` 20→40 (untried regularization axis) to smooth
    leaf-level variance. (3) Final escalation: feature-axis return with
    `voiced_percussive_chroma_asymmetry` (cited untried 3+ times; drums
    have flat chroma so percussive-mask chroma isolates pitched
    accompaniment at transient frames — bass/guitar riff cross-song
    shift, complementary to the MFCC asymmetry family which is saturated).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent (29+ iterations);
    I cannot confirm the 3 singing FPs actually live at p ∈ [0.982, 0.99]
    rather than higher. If they're at p≈0.99 already, LR smoothing won't
    reach them. (ii) SHAP rollup empty — cannot predict LR's effect on
    per-domain p_splice distributions. (iii) 99081f5 ghost state is
    currently IN-tree (max_depth=4, max_leaf_nodes=16 per grep) so my LR
    change stacks on the capacity bump, not the 7972a98 keep-time
    classifier. Attribution of LR delta is clean (only LR changes), but
    absolute combined comparison mixes with the ghost.

(e) Wrapper enhancements. (1) **DISCARD-REVERT SYNC auditor** — persistent
    ask for 6+ iterations. Ghost state cycles (ea1636c feature ghost,
    99081f5 hyperparam ghost, 960113c discarded revert re-instated the
    ghost) continue to contaminate attribution. ~20-line shell guard
    after the discard decision that diffs features.py +
    train_classifier.py vs baseline-sha and aborts / restores on drift.
    (2) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — persistent
    blocker for 29+ iterations. (3) CURRENT CLASSIFIER HYPERPARAMS
    snapshot in prompt (learning_rate, max_depth, max_leaf_nodes,
    max_iter, l2, min_samples_leaf at HEAD) so I don't have to grep
    train_classifier.py every iteration to confirm ghost state.

## 2026-04-20T15:32:38+09:00 — db59c36 (discard, combined=0.490700)
subject: HistGBM min_samples_leaf 20 -> 40 (untried regularization axis)
per-domain: combined_english=0.839506 combined_korean=0.476471 combined_singing=0.295385

# 2026-04-20 — hypothesis: HistGBM min_samples_leaf 20 → 40 (untried regularization axis)

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` one-line change —
    raise HistGBM `min_samples_leaf` 20 → 40, keeping max_iter=200 /
    max_depth=4 / max_leaf_nodes=16 / learning_rate=0.07 /
    l2_regularization=1.0. features.py sha stable so wrapper's auto-retrain
    fires on train_classifier.py edit (US-505). min_samples_leaf has NEVER
    been tuned in the loop's history — every keep/discard used value=20.

(b) WHY this over recent failures. Last 14 iterations exhausted every other
    axis against the 7972a98 baseline (combined=0.589282, singing=0.345,
    3 clean FPs): feature-add (aa4f141/fd500b3/49fa2ca/6384137 all
    regressed), feature removal (ea1636c catastrophic 0.455), DSP gates
    (53d3ea0 SUM=5.5 / 0cb551f MAX=2.2 / 67633f2 PHASE=0.5 all failed),
    stride (b120d38 0.15 failed, denser direction untouched), classifier
    capacity (99081f5 bump in-tree as ghost, 960113c revert matched
    baseline EXACTLY → proves capacity is saturated on this feature
    space), learning_rate (17d4aec 0.05 failed). CLAUDE.md mandates
    structural change after 5+ same-axis failures; every classifier
    axis except min_samples_leaf + max_iter is saturated.
    min_samples_leaf is the explicit cited untried fallback from
    17d4aec(c)(2): "pivot to min_samples_leaf 20→40 (untried
    regularization axis) to smooth leaf-level variance."

    Mechanism targeting the 3 singing FPs: min_samples_leaf is a LEAF-
    LEVEL regularizer requiring every terminal leaf to be supported by
    ≥N training samples before a split is made. Doubling 20→40 forces
    GBM to coalesce small-support leaves into larger ones, smoothing
    the decision surface specifically where training data is sparse.
    Real cross-source splice TPs sit in regions of feature space
    densely covered by training (cross-domain augmentation, pitch-shift
    aug, large class-2/class-1 populations) so their leaves have
    hundreds of supporting samples — p_splice stays ≈1.0 regardless
    of whether the minimum is 20 or 40. The 3 singing FPs at
    p_splice ∈ [0.982, ~1.0] are a different population: chord-
    transition configurations on pop/rock vocals are RARE in training
    (the augmentation targets splice-edge physics, not chord-cycle
    within-song physics), so their decision path likely terminates in
    small-sample leaves that over-fit the training positives around
    them. Raising min_samples_leaf to 40 forces these micro-leaves to
    merge with larger neighbor leaves dominated by clean-label
    training data → p_splice on chord-cycle points gets pulled down
    toward 0.95, below the 0.982 gate → FPs drop while TPs hold.

    This is exactly the classifier-side remedy the 960113c revert
    probed but couldn't deliver: 960113c reduced capacity globally
    (max_depth 4→3, max_leaf 16→8) and produced byte-identical
    combined (0.589282) because the classifier is ALREADY finding the
    same decision function at lower capacity. Capacity isn't the
    bottleneck; leaf-support regularization is a different knob that
    specifically penalizes small training-support leaves without
    globally shrinking the decision surface.

    Why 40 specifically: doubles support requirement (still allows
    granular splits — training set has ~1360 rows in class-2, 20
    samples/leaf allowed ~68 effective leaves, 40 allows ~34, a
    meaningful but not draconian tightening). Larger jumps (60, 80)
    risk wholesale under-fitting; 40 is the natural first step cited
    in the reflection.

    Orthogonal to every recent axis: NOT feature add/remove
    (features.py sha stable); NOT primary-tunable (detector.py
    unchanged); NOT learning_rate (17d4aec); NOT max_depth/
    max_leaf_nodes (99081f5 / 960113c revert); NOT DSP gate. First
    min_samples_leaf tuning in the loop's history. Blast radius:
    1-line edit (20 → 40). Retrain cost absorbed in wrapper's
    normal cycle (~3 min). Risk-bounded: min_samples_leaf is a
    well-understood regularizer; worst case under-fits and combined
    regresses below baseline, fallback is clean (revert to 20 and
    try max_iter=280 from 17d4aec(c)(1)).

(c) IF THIS FAILS. (1) Under-fit — 40 is too coarse, legitimate splice
    boundaries get smoothed away → step down to min_samples_leaf=30
    (finer regularization) OR pair min_samples_leaf=40 with max_iter
    200→280 to restore effective capacity. (2) Inert — decision paths
    already terminate in well-supported leaves → pivot to
    l2_regularization 1.0 → 2.0 (untried shrinkage axis on the weight
    side rather than leaf-support side). (3) Final escalation: feature-
    axis return with voiced_percussive_chroma_asymmetry (cited untried
    4+ times; drums have flat chroma so percussive-mask chroma isolates
    pitched accompaniment at transient frames — bass/guitar riff
    cross-song shift, complementary to MFCC asymmetry).

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 30+
    iterations — I cannot verify whether the 3 singing FPs sit in
    small-support leaves (min_samples_leaf bites cleanly) or well-
    supported leaves (regularization misses). Every classifier-side
    hypothesis remains theory-calibrated on the claim that FP leaves
    are smaller than TP leaves. (ii) SHAP rollup STILL "no keeps yet
    — rollup empty" for 7972a98 despite 4+ keeps in the last 50
    iterations — the rollup pipeline appears broken. (iii) Leaf-level
    statistics are NOT exposed in the training bundle; I don't know
    the current distribution of leaf sample counts (e.g., are most
    leaves already well above the 20 floor, making 40 inert?).

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in CURRENT
    STATE — persistent blocker for 30+ iterations; per-FP (domain,
    file, t_sec, label_id, p_splice, leaf_sample_count_at_prediction,
    top-5 |SHAP|). Leaf sample count at prediction would directly
    test this iteration's mechanism hypothesis. (2) SHAP rollup
    pipeline DEBUG — "no keeps yet" persists across 4+ recent keeps.
    Diagnose why the rollup writer isn't capturing them. (3) CURRENT
    CLASSIFIER HYPERPARAMS + OOF METRICS block in prompt: (max_depth,
    max_leaf_nodes, learning_rate, l2_regularization, min_samples_leaf,
    max_iter, OOF_weighted_F1, OOF_per_class_F1) at HEAD. Resolves
    the persistent "current ?" rows in the per-tunable frontier AND
    surfaces the classifier state without me grep-ing every iteration.

## 2026-04-20T15:51:24+09:00 — 0c3bf76 (discard, combined=0.538510)
subject: add voiced_unvoiced_spec_flatness_asymmetry feature (FEATURE_NAMES 80->81)
per-domain: combined_english=0.853659 combined_korean=0.573913 combined_singing=0.318750

# 2026-04-20 — hypothesis: add voiced_unvoiced_spec_flatness_asymmetry (FEATURE_NAMES 80 → 81)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new
    feature, a voicing-masked SIGNED ABS-DELTA asymmetry on the
    spec_flatness (Wiener-entropy) axis. For t_sec compute the voiced-
    masked mean spec_flatness on pre=[t−2,t] and post=[t,t+2] and the
    unvoiced-masked mean on the same windows; return
    `unvoiced_abs_delta − voiced_abs_delta` where each abs_delta is
    `|post_mean − pre_mean|`. Reuses cached feat_flatness (already
    present; spec_flatness_delta at index 21 consumes it) + feat_vp —
    ZERO new librosa calls, ZERO new caches. Per-t cost 4 slices + 4
    masked scalar means + 2 abs-diffs + 1 subtraction, sub-ms. Sentinel
    0.0 when any voiced/unvoiced mask is empty. Since spec_flatness is
    1D per frame, cosine-distance formulation (used for contrast/MFCC/
    chroma asymmetries) doesn't apply — abs-delta-of-means is the
    natural scalar analogue and preserves the mastering-fingerprint
    mechanism. FEATURE_NAMES 80 → 81 triggers wrapper auto-retrain via
    US-505 sha gate.

(b) WHY this over recent failures. Last 10 iterations exhausted the
    remaining classifier axes: learning_rate (17d4aec failed),
    min_samples_leaf (db59c36 failed), capacity revert (960113c matched
    baseline EXACTLY 0.589282 — proves classifier is inert at this
    feature space: max_depth=3/max_leaf=8 and max_depth=4/max_leaf=16
    produce byte-identical emits). Every primary tunable is also
    saturated (GBM_THRESHOLD / MIN_SEP / stride / DSP gates all tested).
    With the classifier axis provably saturated AND primary tunables
    saturated, the only remaining lever is STRUCTURAL feature work.

    Pick this axis because the voiced/unvoiced ASYMMETRY family has
    the best hit rate in the loop: voiced_unvoiced_mfcc_asymmetry
    (1eda8e3 kept +0.054 — biggest win), voiced_unvoiced_spec_contrast
    _asymmetry (7972a98 kept +0.005 tiny but consistent), chroma-
    asymmetry failed (singer sings new key → voiced_chroma shifts as
    much as unvoiced → asymmetry cancels). Spec flatness is a
    genuinely different "mastering fingerprint" descriptor: Wiener
    entropy (geometric_mean / arithmetic_mean of magnitudes per
    frame) is dominated by NOISE FLOOR and TONAL CONCENTRATION, while
    spec_contrast is dominated by PEAK-TO-VALLEY amplitude ratio per
    band. Different compressors affect these differently — a limiter
    that hits peaks lowers contrast but barely touches flatness; a
    compressor that raises the noise floor (bus compression, tape
    saturation) lifts flatness without changing contrast shape.

    Mechanism targeting the 3 singing FPs. Within one song the entire
    mastering chain is fixed: the vocal bus has one compressor
    profile, the drum bus another, both frozen for the whole track.
    So voiced-frame flatness (vocal noisiness — breath modulation +
    vocal noise floor) is stable across chord transitions (|delta| ≈
    0.002-0.008), and unvoiced-frame flatness (accompaniment
    noisiness — drum kit noise, room bleed, mastering compressor
    output) is ALSO stable (|delta| ≈ 0.003-0.010). Both stable →
    asymmetry ≈ 0. Real same-singer-cross-song splice: singer's vocal
    production is continuous (same larynx / same mic preamp if
    recording sessions overlap), so voiced |delta| stays low
    (~0.005). But the accompaniment track and its mastering chain
    SHIFT (different drum kit / different compression / different
    noise floor), so unvoiced |delta| jumps to 0.02-0.06. Result:
    asymmetry ≈ +0.02-0.05 POSITIVE. Different-singer-cross-song:
    both voiced AND unvoiced flatness shift → both |delta| large →
    asymmetry small (MFCC asymmetry catches this case, so redundancy
    is already covered).

    Why THIS over any feature add. The 7972a98 spec_contrast
    asymmetry was the last kept feature (+0.005) — the axis clearly
    has signal, and flatness is the companion spectral descriptor
    never added. Spec_contrast and spec_flatness are mathematically
    independent (contrast is peak-valley per band, flatness is
    geometric/arithmetic-mean ratio over the whole spectrum). GBM
    max_depth=3/4 can split on each independently; providing both
    lets tree splits learn joint conditions like "contrast_asymmetry
    > 0.15 AND flatness_asymmetry > 0.02" — a stronger mastering-
    change signature than either alone. Speech self-gating via
    paired differencing: voiced and unvoiced speech flatness both
    shift with phoneme context (vowel = tonal low-flatness,
    fricative = noise high-flatness), shifts are uncorrelated with
    splice position, so |delta| values are zero-mean noise that GBM
    learns to ignore via low per-domain SHAP — same safety mechanism
    that neutralized voiced_unvoiced_mfcc_asymmetry on speech.

    Orthogonal to every prior axis: NOT 1eda8e3 voiced_unvoiced_mfcc
    (cepstral envelope, 13-dim cosine); NOT 7972a98
    voiced_unvoiced_spec_contrast (peak-valley per band, 7-dim
    cosine); NOT f4148cc voiced_unvoiced_chroma (pitch-class);
    NOT 49bd0b1 voiced_mfcc (single-mask, no differential); NOT
    32cac36 voiced_chroma (single-mask); NOT 308aa5a tonnetz
    (linear projection of chroma); NOT fd500b3 percussive_mfcc
    (different mask); NOT 49fa2ca/6384137 onset autocorrelation
    (rhythm/tempo); NOT the raw spec_flatness_delta at index 21
    (all-frame, no mask, no voiced/unvoiced split). First voicing-
    masked feature on the spec_flatness (Wiener-entropy) axis, and
    first SCALAR-valued asymmetry (abs-delta-of-means instead of
    cosine-distance-of-vectors). Blast radius: features.py only —
    1 new block function + 1 FEATURE_NAMES append + 2 assert
    bumps + 1 call in extract_features + 1 print update in
    __main__ self-test. ZERO new caches, ZERO new librosa calls.
    Per-t cost sub-ms; aggregate per-chunk cost <1s. Risk-bounded:
    sentinel 0.0 on empty-mask edge cases; signal is physically
    grounded on singing where mastering signatures shift across
    cross-song splices.

(c) IF THIS FAILS. (1) If the abs-delta-of-means formulation loses
    signal (intra-song chord cycle has |voiced_delta| ≈ 0.005 but
    cross-song has |voiced_delta| ≈ 0.008 too — small signal
    drowned by cross-chunk flatness variance), replace with SIGNED
    delta-of-means `(unvoiced_post − unvoiced_pre) − (voiced_post −
    voiced_pre)` to preserve DIRECTION of shift (up vs down) —
    the GBM can then split on sign. (2) If flatness is dominated
    by file-level noise-floor not transition-local, replace with
    ratio formulation `voiced_unvoiced_spec_flatness_ratio_asymmetry`
    = ratio of post/pre within each mask — self-normalizes file-
    level level. (3) Final escalation: an F0-stability asymmetry
    `voiced_unvoiced_f0_jitter_asymmetry` combining pitch-stability
    with accompaniment-level continuity — pivot to loudness/pitch
    domain rather than spectral-tonal.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 31+
    iterations — I cannot verify whether the 3 singing FPs actually
    sit at boundaries with stable voiced flatness (the specific
    pattern this feature targets), or at spectral-content-matched
    positions where flatness is uninformative. Every feature-add
    hypothesis since iteration ~70 has been theory-calibrated on
    physical mechanism without eval-corpus verification. (ii) SHAP
    rollup STILL "no keeps yet — rollup empty" for 7972a98 despite
    7972a98 being the current keep — the rollup pipeline appears
    broken, so I cannot see which features currently drive
    per-domain p_splice. (iii) 99081f5 classifier-capacity ghost
    (max_depth=4/max_leaf=16) is IN-tree at HEAD per grep; 960113c
    revert to 3/8 was discarded because it matched baseline
    exactly → wrapper's symmetry-semantics bug RESTORED the ghost.
    My feature-add delta is attributed against a classifier that
    differs from 7972a98-keep-time 3/8 state. (iv) Four recent
    iterations (17d4aec, db59c36, 0cb551f, 6384137) all produced
    IDENTICAL combined=0.490700 with IDENTICAL per-domain —
    statistically near-impossible unless (a) all four changes
    converge on the same emit set (suspicious given very different
    change types), OR (b) wrapper cache invalidation / retrain-skip
    bug. Cannot investigate from here.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE — persistent blocker for 31+ iterations; per-FP
    (domain, file, t_sec, label_id, p_splice, dsp_phase_z, dsp_t2_z,
    dsp_cpe_z, voiced_mean_spec_flatness_pre,
    voiced_mean_spec_flatness_post, unvoiced_mean_spec_flatness_pre,
    unvoiced_mean_spec_flatness_post, top-5 |SHAP|). Every
    spec_flatness / voiced-mask hypothesis becomes data-driven
    instead of theory-calibrated. (2) DISCARD-REVERT SYMMETRY
    SEMANTICS bug fix (persistent ask) — wrapper's discard path
    should DETECT revert commits (commit message starts with
    "REVERT" AND references a prior-discarded SHA) and treat
    "matched baseline exactly" as SUCCESS-KEEP rather than
    no-improvement-discard. Would have kept 960113c's 3/8 revert
    and eliminated the 99081f5 ghost. ~15 lines git-commit-message
    parsing. (3) IDENTICAL-COMBINED ALERT — if N consecutive
    iterations produce IDENTICAL combined within ±1e-5 AND
    IDENTICAL per-domain, emit a WARN event
    `wrapper.suspicious.identical_output_streak`. Last 4
    iterations (17d4aec, db59c36, 0cb551f, 6384137) all hit
    combined=0.490700 exactly; this alert would flag potential
    cache-invalidation bugs to the operator.

## 2026-04-20T15:57:39+09:00 — df0ceec (discard, combined=0.490700)
subject: HistGBM l2_regularization 1.0 -> 2.0 (final untried classifier axis)
per-domain: combined_english=0.839506 combined_korean=0.476471 combined_singing=0.295385

# 2026-04-20 — hypothesis: HistGBM l2_regularization 1.0 → 2.0 (final untried classifier axis)

(a) HYPOTHESIS. Pure `splice/classifier/train_classifier.py` one-line change
    — raise HistGBM `l2_regularization` 1.0 → 2.0, keeping max_iter=200 /
    max_depth=4 / max_leaf_nodes=16 / learning_rate=0.07 /
    min_samples_leaf=20. l2_regularization applies an L2 penalty on LEAF
    WEIGHTS (prediction values, not structure). Doubling shrinks leaf
    outputs toward 0 → predicted log-odds pulled toward the class prior →
    borderline p_splice values (currently in [0.982, 1.0]) concentrate
    slightly lower. features.py sha stable so the wrapper retrains via
    the US-505 train_classifier.py-change path.

(b) WHY this over recent failures. l2_regularization is the FINAL
    untried HistGBM hyperparameter axis. Per db59c36(c)(2) explicit
    cited fallback: "pivot to l2_regularization 1.0 → 2.0 (untried
    shrinkage axis on the weight side rather than leaf-support side)".
    Last 10+ iterations exhausted every other classifier knob:
    max_depth/max_leaf (99081f5 bump + 960113c revert matched baseline
    — STRUCTURE-axis saturated), learning_rate (17d4aec 0.05 failed —
    GLOBAL-SHRINKAGE axis saturated), min_samples_leaf (db59c36 20→40
    failed — LEAF-SUPPORT axis saturated). l2_regularization operates
    on a physically distinct axis: it penalizes large leaf weights
    directly during fit. Different from learning_rate (which shrinks
    tree CONTRIBUTIONS globally without changing the fit objective)
    and different from min_samples_leaf (which restricts split CREATION
    by sample count but doesn't constrain the leaf weight magnitude).

    Mechanism targeting the 3 singing FPs at p_splice ∈ [0.982, ~1.0]:
    doubling λ from 1.0 to 2.0 adds penalty 2·Σ(w²) per tree to the
    loss. Trees that currently produce high-magnitude leaves on
    chord-transition training-neighbor regions (where training
    positives sit densely around the chord-cycle feature space) will
    shrink those leaves more, producing LOWER predicted log-odds for
    those same chord-transition configurations at eval. Real cross-
    source splice TPs sit in regions with broader training-positive
    support (cross-domain augmentation spreads them across feature
    space), so each tree's contribution is averaged over many training
    samples — the MEAN leaf weight is less affected by doubling λ
    because the unregularized fit was already close to the l2-shrunk
    optimum. Net: borderline singing FPs (chord-cycle noise) pulled
    slightly below 0.982 gate; real TPs at p≈1.0 survive.

    Why 2.0 specifically: doubling is the natural first step in a
    multiplicative regularization sweep. λ=10+ would risk wholesale
    under-fitting. Orthogonal to every recent axis: NOT feature add
    (features.py sha stable); NOT primary-tunable (detector.py
    unchanged); NOT DSP gate; NOT learning_rate (17d4aec); NOT max_depth/
    max_leaf_nodes (99081f5/960113c); NOT min_samples_leaf (db59c36).
    First l2_regularization tuning in the loop's history. Blast radius:
    1-line edit (1.0 → 2.0). Retrain cost absorbed in the wrapper's
    normal 3-min cycle.

(c) IF THIS FAILS. (1) Under-fit on borderline TPs — raising λ pulls
    real TPs below gate too → step to max_iter 200 → 280 (untried,
    more trees compensate for per-tree shrinkage). (2) Inert —
    l2_regularization has minimal effect at 80-feature HistGBM because
    early stopping already constrains fit → pivot to
    voiced_percussive_chroma_asymmetry (cited untried 4+ times;
    percussive-mask chroma isolates pitched accompaniment at transient
    frames — bass/guitar riff cross-song key shift, genuinely new
    mask/content combination). (3) Final escalation: ensemble of 2
    HistGBMs with different l2 values (1.0 + 2.0) via soft-voting;
    different from prior same-seed ensemble (failed 0.473) because
    diversification via regularization axis not random state.

(d) Information gaps. (i) CLEAN_FP_POSITIONS STILL absent after 32+
    iterations — I cannot verify whether the 3 singing FPs sit at
    p ∈ [0.982, 0.99] (where l2 shrinkage would bite) or p ≈ 1.0
    (where l2 shrinkage has negligible effect). Every classifier-side
    hypothesis remains theory-calibrated. (ii) SHAP rollup STILL "no
    keeps yet — rollup empty" for 7972a98 despite 4+ keeps — rollup
    pipeline appears broken. (iii) Persistent IDENTICAL-COMBINED
    STREAK concern: 4 iterations (17d4aec LR, db59c36 leaf_size,
    0cb551f MAX floor, 6384137 rhythm feature) all produced EXACTLY
    combined=0.490700, english=0.839506, korean=0.476471,
    singing=0.295385. Statistically near-impossible unless wrapper
    cache/retrain bug masks true attribution. If my l2 change also
    lands at 0.490700, that's strong evidence of a wrapper-side bug.
    (iv) 99081f5 capacity ghost still IN-tree at HEAD (max_depth=4 /
    max_leaf_nodes=16) — my l2 change runs against the 4/16 base,
    not the 7972a98-keep-time 3/8 base. Absolute comparison to
    0.589282 mixes with the ghost, but the l2 delta attribution is
    clean (only l2 changes).

(e) Wrapper enhancements. (1) **IDENTICAL-COMBINED-STREAK DIAGNOSTIC**
    — now persistent across 2 reflections. Emit
    `wrapper.suspicious.identical_output_streak` when N ≥ 3
    consecutive iterations produce IDENTICAL combined ± 1e-5 AND
    IDENTICAL per-domain. Include predicted-emit-count comparison
    across the streak — if emit sets are byte-identical across
    physically different changes, retrain/cache invalidation is
    broken. ~30 lines of Python in the wrapper's post-eval hook.
    (2) CLEAN_FP_POSITIONS JSON block in CURRENT STATE — persistent
    blocker for 32+ iterations. Without it every DSP / classifier /
    feature hypothesis around singing FPs is a theory-calibrated bet.
    Per-FP: (domain, file, t_sec, label_id, p_splice, dsp_phase_z,
    dsp_t2_z, dsp_cpe_z, top-5 |SHAP|, leaf_sample_count). (3)
    DISCARD-REVERT SYNC auditor — the 99081f5 ghost persists at HEAD
    (verified: max_depth=4, max_leaf_nodes=16 still in
    make_pipeline()) after 960113c's revert was itself discarded by
    the wrapper's symmetry-semantics bug. ~20 lines of shell guard
    parsing commit subjects for "REVERT" prefix + prior-SHA reference,
    treating such matches as KEEP not DISCARD when combined matches
    baseline.

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

