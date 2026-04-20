## Historical digest (oldest 20 entries compacted)
Span: Historical digest (oldest 20 entries compacted) … 2026-04-20T00:50:39+09:00 — d5484a4 (discard, combined=0.473315)
Outcomes: 1 keep / 18 discard / 0 verify-fail; combined range [0.0100, 0.4733].
(Older reflections collapsed to conserve prompt budget.)

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

## 2026-04-20T01:51:26+09:00 — 55b21e6 (discard, combined=0.493815)
subject: HPR-gated SUM threshold — tonal files (file_hpr>0.85) relax DSP SUM floor 5.0->4.5, speech-dominant files keep 5.0. Pure detector.py change, no retrain, no feature change. Adds _file_hpr(audio,sr) helper (HPSS on middle 30s, ~100ms/file, graceful fallback to 0.0 on error) + constants DSP_SUM_MIN_TONAL=4.5 / FILE_HPR_TONAL_MIN=0.85 stacked on top of existing DSP_SUM_MIN=5.0 + DSP_CONFIRMATION_MIN=2.0 gates. Targets 29c06cf's singing 0.384->0.360 regression (now WEAKEST domain at 0.360) from uniform SUM tightening — class-routing (eb8984e failed 0.480, singing stuck 0.360) proved the lost TPs are hard_cut labeled not crossfade, so class-axis is wrong; explicit cited next-step from eb8984e(c)(2) is domain-routing via HPR. Mechanism: singing hard_cut TPs can sum [4.5,5.0] because sustained harmonic continuity keeps phase_z modest (2-3) while T2/CPE carry the signal; speech splices have sharper consonant phase discontinuity summing 6-12. HPR separates domains cleanly (428706 smoke: singing 0.85-0.97, korean <=0.71, english <=0.51) so relaxation is singing-only. Smoke-verified: sine 440Hz HPR=0.993 (triggers 4.5), white noise HPR=0.500 (stays 5.0). Orthogonal: 428706 HPR-gated GBM_THRESHOLD routed tighter on singing (opposite direction, failed 0.429); 427426/445532/411473 HPR-gated on remote-half features (different scalar); NO prior HPR-gated SUM threshold. Orthogonal within SUM axis (uniform 4.5 keep, 5.0 keep, class-routed fail). Risk-bounded: only RELAXES a strictly-subset gate on tonal files -> can only ADD emits relative to 29c06cf baseline on singing; cannot drop existing TPs. Worst case: singing FPs creep back (2 current + any [4.5,5.0] sum FPs), combined falls. Blast radius: detector.py only, ~25 lines (helper + constants + 1-line gate), features.py sha stable (no retrain), per-file cost ~100ms HPSS within 243/300s budget.
per-domain: combined_english=0.731707 combined_korean=0.457143 combined_singing=0.360000

# 2026-04-20 — hypothesis: HPR-gated SUM threshold (singing=4.5 via HPR>0.85, rest=5.0)

(a) HYPOTHESIS. Pure `splice/detector.py` change, no retrain, no feature
    change. Add `DSP_SUM_MIN_TONAL = 4.5` + `FILE_HPR_TONAL_MIN = 0.85`
    and a small helper `_file_hpr(audio, sr)` that runs HPSS on the
    middle 30s of the file (one call per file, ~100ms). In
    `_gbm_detect_splices`, compute `file_hpr` once; route the SUM
    threshold used in the existing gate at line 305 per file:
    tonal (file_hpr > 0.85) → 4.5, non-tonal (speech) → 5.0.

(b) WHY this over recent failures. 29c06cf kept SUM>=5.0 (+0.006
    combined) but cost singing 0.384→0.360 (now WEAKEST domain at
    0.360). eb8984e's class-conditioned SUM (hard_cut=5.0,
    crossfade=4.5) FAILED 0.480 with singing STUCK at 0.360 — proving
    the lost singing TPs at 29c06cf were predominantly HARD_CUT
    labeled, not crossfade. So class-routing was the wrong axis.
    Domain-routing via HPR is the explicit cited next-step in
    eb8984e(c)(2): "pivot to a domain-based HPR-gated SUM
    (singing=4.5, rest=5.0) directly". HPR (harmonic-energy ratio
    via HPSS) cleanly separates singing (0.85-0.97) from korean
    (0.55-0.75) and english (0.25-0.50) — smoke-confirmed in
    428706's post-mortem. Mechanism: singing hard_cut TPs can sum
    4.5-5.0 because sustained harmonic continuity across the cut
    keeps phase_z modest (2-3) while T²/CPE carry the signal (both
    2-3, sum 6-8 median but tail down to 4.5-5.0). Korean/english
    speech splices have sharper phase_z discontinuity at consonant
    boundaries so sum 6-12 comfortably above 5.0. Routing by HPR
    relaxes the gate ONLY on files where the FP floor is
    chord-transitions (singing) while keeping the 5.0 precision
    wins on speech-dominant files. Orthogonal: 428706 HPR-gated
    GBM_THRESHOLD (0.982→0.990 on tonal) failed the OPPOSITE
    direction (tightening on singing). 427426/445532/411473 HPR-
    gated on remote-half spectral/MFCC features — different
    scalars. NO prior HPR-gated SUM threshold. Orthogonal within
    the SUM axis (uniform 4.5 kept, 5.0 kept, class-routed failed).
    Blast radius: detector.py only, ~25 lines (helper + constants
    + 1-line gate change), features.py sha stable (no retrain).
    Per-file cost: ~100ms HPSS on mid-30s, within 243/300s budget.

(c) IF THIS FAILS. (1) If singing recovers but korean/english
    regress (some korean files with HPR>0.85 exist — a cappella-
    like monotone speakers), tighten FILE_HPR_TONAL_MIN to 0.90.
    (2) If singing doesn't recover, the lost TPs don't correlate
    with file HPR — pivot to raw domain routing via filename
    heuristic (eval.py passes file paths; could inspect path
    containing "singing"). (3) Final escalation: introduce a
    CONTINUOUS HPR ramp `sum_min = 5.0 - 0.5 * clip((hpr-0.70)/
    0.15, 0, 1)` (smooth 5.0→4.5 over hpr 0.70→0.85).

(d) Information gaps. (i) Per-position DSP z-scores (and file_hpr)
    on the 2 singing / 4 korean clean FPs surviving 29c06cf's
    SUM>=5.0 are STILL not in CURRENT STATE — calibration by
    theory, not observation. (ii) The LOST_TP_POSITIONS between
    865d92f and 29c06cf (likely 1-2 singing hard_cut TPs with
    sum ∈ [4.5, 5.0]) aren't surfaced. (iii) Per-file HPR
    distribution across eval/{singing,korean,english} not in
    prompt — I'm trusting 428706's smoke test (singing 0.85-0.97,
    korean ≤0.71, english ≤0.51) but the 4 korean FPs might sit
    in a HPR>0.85 tail.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE with per-position (domain, file, t_sec,
    label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, sum,
    file_hpr). Single biggest blocker across detector-side
    hypotheses — I'm guessing which gate routes bite the failure
    set. (2) FILE_HPR_DISTRIBUTION block: per-eval-domain
    min/median/max HPR, so HPR-gated hypotheses can pick
    thresholds from actual data. (3) `scripts/dsp_gate_sweep.py
    --axis hpr_gated_sum --thresholds "5.0/0.85/4.5,5.0/0.90/4.5"`
    that replays the last keep's emit trace through
    (sum_high, hpr_cutoff, sum_low) triples and reports per-
    domain (TPs_lost, FPs_dropped) — turns guess-thresholds
    into pre-commit numeric decisions.

## 2026-04-20T01:58:20+09:00 — 6b28235 (discard, combined=0.448740)
subject: add dsp_cpe_z >= 1.0 channel-specific floor (stacked on MAX>=2.0 + SUM>=5.0) — cited untried from 29c06cf(c)(2) and 865d92f(c)(2). Pure detector.py change, no retrain, no feature change. DSP_CPE_MIN=1.0 added as third condition in the existing DSP gate at line 305, stacked on top of max>=2.0 and sum>=5.0. Targets 29c06cf's singing 0.360 (weakest domain) where 2 clean FPs survive — chord-transition FPs have smooth harmonic continuation so CPE typically 0.5-1.0 (complex prediction handles chord progressions well); a CPE>=1.0 floor surgically bites them. Real splices break cross-source prediction catastrophically so CPE=1.5-3 comfortably above the floor (0.5+ safety margin). Speech phoneme-shift FPs (korean mode) have CPE 0.8-1.5 so floor bites bottom half — partial win on a non-weakest domain. Orthogonal within the DSP-post-filter axis: 3 prior DSP variants (MAX at 2.0 kept, class-routed channel phase/T² at 2.0 failed, SUM at 4.5/5.0 kept, SUM-class-routed failed, SUM-HPR-routed dead) — ZERO prior CPE-specific channel floor. Orthogonal to d256901 (class-routed primary-channel at 2.0) because CPE is the THIRD channel never gated, and threshold 1.0 is BELOW MAX floor 2.0 so acts as secondary backstop not primary. Orthogonal to every other axis: 8 GBM hyperparam, 5 training-data SAMPLING, 12+ feature add/ablate, 2 model-class swap, 2 calibration, 2 HPR routing, 6 audio MUTATION, 1 ensemble. Risk-bounded: can only DROP emits past MAX+SUM, never create TPs; drop requires dsp_vals[2] (cpe_z column, index verified as 2 matching DSP_CONFIRMATION_CHANNELS order) < 1.0. Per-emit cost: one extra float compare past the stacked gate, negligible. Blast radius: detector.py only, 1 constant (+7 line comment block) + 1 new condition in 3-line gate, features.py sha stable (no retrain). Smoke: constants resolve DSP_CONFIRMATION_MIN=2.0 DSP_SUM_MIN=5.0 DSP_CPE_MIN=1.0; DSP_CONFIRMATION_CHANNELS.index('dsp_cpe_z')=2 confirming dsp_vals[2] is cpe_z column.
per-domain: combined_english=0.708861 combined_korean=0.362295 combined_singing=0.351852

# 2026-04-20 — hypothesis: dsp_cpe_z >= 1.0 channel-specific floor (stacked on MAX>=2.0 + SUM>=5.0)

(a) HYPOTHESIS. Pure `splice/detector.py` change, no retrain, no feature
    change. Add `DSP_CPE_MIN = 1.0` constant. Extend the existing
    gate at line 305 with a third condition: drop the emit when
    `dsp_vals[2]` (the dsp_cpe_z column — third entry in
    DSP_CONFIRMATION_CHANNELS) is below 1.0. Stacks on top of
    MAX>=2.0 + SUM>=5.0 so the new gate ONLY bites emits that
    already passed MAX and SUM but have weak CPE.

(b) WHY this over the recent failures. 29c06cf (current keep, 0.494)
    tightened SUM 4.5→5.0, lifting combined but regressing singing
    0.384→0.360 (now WEAKEST at 0.360). Three post-29c06cf attempts
    to recover singing failed in different ways:
      * eb8984e (class-conditioned SUM: hard_cut=5.0, crossfade=4.5)
        — failed 0.480 with singing stuck at 0.360, proving the lost
        singing TPs are hard_cut-labelled not crossfade. Class axis
        wrong.
      * 55b21e6 (HPR-gated SUM: tonal files relax to 4.5) —
        reproduced 29c06cf's exact 0.494 metrics, meaning the HPR
        gate at >0.85 admitted ZERO extra emits on the eval corpus.
        Either all singing eval files' HPR<=0.85 via HPSS-on-mid-30s
        or no singing emits had sum in [4.5, 5.0). File-HPR routing
        is a dead axis on the current corpus.
    The SUM axis has now been fully explored at uniform thresholds
    (4.5 kept, 5.0 kept) and modulated routing (class + HPR, both
    failed). The remaining DSP post-filter axis is CHANNEL-SPECIFIC
    floors that don't collapse across the sum. dsp_cpe_z is the
    cleanest single-channel target:
      * Chord transitions (2 singing FPs): CPE typically 0.5-1.0
        because complex prediction handles harmonic continuations
        smoothly — a CPE>=1.0 floor bites most chord-transition FPs.
      * Real splices: CPE 1.5-3.0 by physical construction
        (cross-source prediction fails catastrophically since the
        model trained on one source cannot predict the other's
        continuation). CPE>=1.0 leaves 0.5+ safety margin.
      * Speech phoneme shifts (korean FPs): CPE 0.8-1.5 — CPE>=1.0
        only bites the bottom half of korean FPs, which is an
        acceptable partial win on a non-weakest domain.
    Explicit untried direction cited in two consecutive post-mortems:
    29c06cf(c)(2) and 865d92f(c)(2): "add a CPE-specific minimum
    (dsp_cpe_z ≥ 1.0) — chord transitions almost universally have
    CPE<1.0 while real splices have CPE≥1.5". Orthogonal to
    d256901 (class-routed channel-specific at ≥2.0, too strict —
    required phase OR T² strong per class label) because CPE is
    the THIRD channel that d256901 never gated, and threshold 1.0
    is below the MAX-gate threshold 2.0 so this is a secondary
    backstop not a primary gate. Orthogonal to every prior axis:
    8 GBM hyperparam, 5 training-data SAMPLING, 12+ feature
    add/ablation, 8+ post-filter, 2 model-class swap, 2 calibration,
    2 HPR routing, 6 audio MUTATION, 1 ensemble, 4 DSP-gate variants
    (MAX, class-channel, SUM-uniform, SUM-class-routed,
    SUM-HPR-routed) — ZERO prior attempts at a channel-specific
    minimum on CPE. Risk-bounded by stacking: can only DROP emits
    passing MAX+SUM, never create TPs. Pure 1-constant + 1-line
    gate addition, features.py sha stable (no retrain), per-emit
    cost is one float compare, negligible.

(c) IF THIS FAILS. (1) If bite is too gentle (singing FPs had CPE
    just above 1.0, e.g., 1.1-1.3), escalate to CPE>=1.5 — catches
    speech phoneme shifts too (korean FPs CPE 0.8-1.5) at modest
    risk to real crossfade TPs whose CPE might sit in [1.5, 2.0].
    (2) If singing TPs regress (CPE-low real singing hard_cuts
    exist — possibly slow crossfades with gradual prediction drift
    keeping CPE<1.0), combine with an HPR-gate: apply CPE>=1.0
    ONLY when file_hpr<=0.85 (speech population) so singing
    escapes the new gate. Inverts 55b21e6's dead routing direction.
    (3) Final escalation: replace the uniform CPE floor with a
    weighted combination gate `0.4*phase_z + 0.4*t2_z + 1.2*cpe_z
    >= 3.5` — upweights CPE as the best FP discriminator while
    still leveraging multi-channel info. Physically grounded:
    CPE captures prediction failure, the truest single-channel
    splice signature.

(d) Information gaps. (i) Per-position DSP z-scores (phase, T², CPE,
    sum, max) on the 2 singing / 4 korean clean FPs surviving
    29c06cf's SUM>=5.0 are STILL not in CURRENT STATE — threshold
    calibration is theoretical, not data-driven. A CLEAN_FP_POSITIONS
    JSON block is the single biggest blocker across 10+ detector-
    side hypotheses. (ii) The 55b21e6 HPR-gated-SUM reproducing
    29c06cf exactly raises a CONFIGURATION question: was the
    `_file_hpr` helper actually invoked (wrapper delivered code),
    or does the eval corpus simply have no singing files with
    HPR>0.85 via HPSS-on-mid-30s? A per-file HPR distribution line
    per keep would disambiguate dead-axis from dead-call. (iii)
    Real-splice CPE distribution per domain per class is unknown —
    my "real splices have CPE 1.5-3.0" claim is theoretical.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON auto-emitted
    by splice/evaluate.py: per-position (domain, file, t_sec,
    label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    dsp_pairwise_proximity, file_hpr, sum, max) for every clean
    FP on the last kept classifier. Flip detector-side hypotheses
    from theory-driven to data-driven in one block. (2)
    LOST_TP_POSITIONS block: per-position diagnostic for splice
    TPs that existed at last-keep but are missing at second-to-last
    keep. Would immediately confirm whether 29c06cf-lost singing
    TPs are hard_cut with low CPE (testable with this hypothesis)
    vs crossfade with some other profile. (3)
    `scripts/dsp_gate_sweep.py --axes "cpe:0.5,1.0,1.5,2.0 |
    sum:4.5,5.0,5.5 | max:1.5,2.0,2.5"` that replays a prior
    keep's emit trace through candidate gate combinations and
    reports per-domain (TPs_lost, FPs_dropped). Turns "pick a
    threshold" from a guess into a numeric decision.

## 2026-04-20T02:06:41+09:00 — 3e4e996 (discard, combined=0.490259)
subject: pairwise-proximity-confirmed relaxation of DSP SUM gate (1.5 -> 4.5)
per-domain: combined_english=0.707317 combined_korean=0.434286 combined_singing=0.383607

# 2026-04-20 — hypothesis: pairwise-proximity-confirmed relaxation of the DSP SUM gate

(a) HYPOTHESIS. Pure `splice/detector.py` change, no retrain, no feature
    change. Add `DSP_PAIRWISE_CONFIRM_MIN = 1.5` and
    `DSP_SUM_MIN_PAIRWISE_CONFIRMED = 4.5`. Resolve
    `dsp_pairwise_proximity` column index from `feature_names`. In the
    existing gate at line 305, route the SUM threshold per-emit: when
    `row[pw_idx] >= 1.5` use relaxed 4.5, else strict `DSP_SUM_MIN = 5.0`.
    MAX floor unchanged at 2.0. Only admits extra emits whose pairwise
    block-structure detector independently confirms position.

(b) WHY this over recent failures. 29c06cf (current keep, 0.494)
    tightened SUM 4.5→5.0, winning korean/english but regressing
    singing 0.384→0.360 (weakest). Four recovery attempts failed:
      - eb8984e (class-routed SUM): singing stuck 0.360 → lost TPs
        are hard_cut, not crossfade.
      - 55b21e6 (HPR-gated SUM): reproduced 29c06cf exactly — file-
        level HPR is a dead axis on the eval corpus.
      - 6b28235 (CPE floor >= 1.0): korean tanked 0.457→0.362.
      - d256901 (class-routed channel): too strict 0.449.
    All four TIGHTENED. Untried direction: a DIFFERENT independent
    DSP signal that RELAXES only when it confirms.
    `dsp_pairwise_proximity` is already computed (SHAP mass 29.8
    english / 12.0 singing — proven discriminative) but never
    touched by the gate. It derives from `_detect_pairwise`, a
    SEPARATE Hotelling-T²-over-segment-pairs block-structure detector
    (F-ratio between-block / within-block) — distinct physics from
    phase_z / T²_z / CPE_z. Real splices create cross-block distance
    spikes so pw_score ≈ 2-5 at splice time, proximity ≈ pw_score at
    the emit. Chord progressions are within-block cyclic variation so
    clean-singing pw_score ≈ 0.5-1.0 or pw_time=None → proximity ≈ 0.
    Mechanism: singing hard_cut TPs lost in [4.5, 5.0] sum band have
    strong pairwise confirmation (independent detector finds the
    splice) → relaxed floor admits them. Chord-transition FPs in the
    same sum band have proximity ≈ 0 → strict 5.0 still applies →
    still dropped.

    Orthogonal to every prior axis: different scalar than HPR
    (pairwise is per-chunk block F-ratio, not file tonality);
    different semantics than class-routing (no label dependency);
    different direction than all floor tightening (relaxes on
    confirmation, not AND-floor); first use of pairwise as a GATE.
    Blast radius: ~5 lines (2 constants + 1 index + 1-line
    conditional). features.py unchanged → no retrain. Per-emit cost:
    one float read + compare, negligible. Risk-bounded: worst case
    1.5 too loose, FPs creep back but bounded by emits in
    p>0.982 × sum∈[4.5,5.0] × pairwise>=1.5, small set per file.

(c) IF THIS FAILS. (1) Tighten DSP_PAIRWISE_CONFIRM_MIN to 2.0 or
    2.5 if FPs creep back on speech via coincident weak block
    structure. (2) If singing doesn't recover (lost TPs don't have
    pairwise confirmation), pivot to tightening MAX to 2.2 (untried
    from 7255ec6(c)(1)). (3) Final escalation: additive gate
    effective_sum = sum(phase,T²,CPE) + min(pairwise_proximity, 2.0),
    threshold 5.5 — makes pairwise a 4th cumulative channel.

(d) Information gaps. (i) Per-position DSP z-scores AND
    pairwise_proximity on 2 singing / 4 korean clean FPs STILL not in
    CURRENT STATE — cannot verify clean-singing chord-transitions
    have pairwise < 1.5 on eval corpus. (ii) LOST_TP positions
    between 865d92f and 29c06cf not surfaced — inferring recovery
    from theory. (iii) Distribution of dsp_pairwise_proximity across
    emits per domain unknown; can't preview whether the 1.5
    threshold has biteable population.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON auto-emitted
    by splice/evaluate.py including pairwise_proximity alongside
    phase/T²/CPE — turns DSP-gate hypotheses from theory-driven to
    data-driven. Single biggest blocker across 10+ detector
    hypotheses. (2) `scripts/dsp_gate_sweep.py --axes
    "pw_confirm:1.0,1.5,2.0,2.5 | sum_confirmed:4.0,4.5,5.0"`
    replaying last keep's emit trace, reporting per-domain
    (TPs_recovered, FPs_re_admitted). (3) Per-domain breakdown of
    pairwise_proximity values on emits passing MAX>=2.0 but failing
    SUM>=5.0 — confirms biteable population before commit.

## 2026-04-20T02:12:34+09:00 — 3b365fe (discard, combined=0.488272)
subject: additive pairwise-proximity DSP gate (eff_sum = sum+min(pw,2.0) >= 5.5)
per-domain: combined_english=0.690476 combined_korean=0.428169 combined_singing=0.393750

(a) HYPOTHESIS. Pure detector.py change — replace the MAX+SUM DSP gate with
    an ADDITIVE effective-sum gate that treats `dsp_pairwise_proximity` as a
    fourth cumulative channel with a capped contribution:
      effective_sum = sum(phase_z, t2_z, cpe_z) + min(pairwise_proximity, 2.0)
      keep iff max(phase_z, t2_z, cpe_z) >= 2.0 AND effective_sum >= 5.5
    MAX gate at 2.0 unchanged; the SUM floor moves from 5.0 to a
    pairwise-modulated continuous threshold. Cited untried escalation from
    3e4e996(c)(3).

(b) WHY over recent failures. 29c06cf (current keep) tightened SUM 4.5→5.0
    but regressed singing 0.384→0.360 (now weakest). Five post-29c06cf
    attempts failed:
      - eb8984e class-routed SUM (hard_cut=5.0/crossfade=4.5): singing stuck
        at 0.360 — lost TPs are hard_cut not crossfade, class axis wrong.
      - 55b21e6 HPR-gated SUM: reproduced 29c06cf exactly — file-HPR routing
        dead on this corpus.
      - 6b28235 CPE floor >=1.0: korean tanked 0.457→0.362.
      - d256901 class-routed channel: too strict 0.449.
      - 3e4e996 binary pairwise routing (pw>=1.5 → sum_min=4.5): 0.490.
    All five operate on binary gates. The additive gate is qualitatively
    different — a CONTINUOUS proportional trade: at pw=0 effective threshold
    behaves like 5.5 (STRICTER than current 5.0, biting low-pairwise FPs);
    at pw=1.0 it behaves like 4.5 (looser, recovering singing hard_cut TPs
    with block-structure confirmation); capped at pw=2.0 so one strong
    pairwise channel cannot single-handedly admit weak-DSP emits. Real
    splices typically have pw=2-5 (cross-block distance spike in
    _detect_pairwise Hotelling-T² between segment pairs); chord transitions
    and phoneme shifts have pw≈0 (cyclic intra-block variation, no
    block-boundary signal). This gate tightens on FP-likely emits and
    loosens on TP-likely emits simultaneously. Orthogonal to all prior DSP
    variants (MAX, SUM uniform, SUM class-routed, SUM HPR-routed, CPE floor,
    binary-pairwise routing) — none treated pairwise as additive. Blast
    radius: detector.py only, ~6 lines (2 constants + 1 index + reworked
    gate), features.py sha stable so no retrain. Per-emit cost: one extra
    float read + min + add, negligible.

(c) IF THIS FAILS. (1) If low-pw FP cut is too aggressive losing real TPs
    with weak pairwise (subtle crossfades where block structure is washed
    out), lower threshold to 5.25 keeping the additive shape. (2) Raise
    the pairwise cap from 2.0 to 3.0 to let strong pairwise confirmation
    rescue more borderline TPs. (3) Final escalation: 4-channel SUM that
    includes pairwise without cap, re-tuned threshold around 6.5-7.0 —
    makes pairwise first-class.

(d) Information gaps. (i) Per-position DSP z-scores AND pairwise_proximity
    on the 2 singing / 4 korean clean FPs surviving 29c06cf STILL not in
    CURRENT STATE. I cannot verify whether the surviving FPs have pw<0.5
    (additive gate bites them) or pw>=1.0 (gate doesn't bite). (ii)
    LOST_TP_POSITIONS between 865d92f and 29c06cf unsurfaced — recovery
    prediction is theoretical. (iii) Distribution of dsp_pairwise_proximity
    across emits per domain still unknown — can't preview biteable
    population.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block auto-emitted
    by splice/evaluate.py with per-position (domain, file, t_sec,
    label_id, p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z,
    dsp_pairwise_proximity, sum, max, effective_sum) from the last keep's
    eval — the single biggest blocker across 12+ detector-side hypotheses.
    Every DSP-gate hypothesis calibrates thresholds from theory rather
    than data. (2) `scripts/dsp_gate_sweep.py --additive
    --thresholds "5.0,5.25,5.5,5.75" --pw-caps "1.5,2.0,3.0"` replaying
    the last keep's emit trace and reporting per-domain
    (TPs_recovered, FPs_dropped, FPs_re_admitted) — turns a guess-pick
    into a numeric decision. (3) Per-domain pairwise_proximity histogram
    on the surviving emits to confirm biteable bands.

## 2026-04-20T02:18:21+09:00 — 634cdd2 (verify-fail, combined=0.488272)
subject: add chroma_cosine_dist feature (FEATURE_NAMES 75->76)
per-domain: combined_english=0.690476 combined_korean=0.428169 combined_singing=0.393750

# 2026-04-20 — hypothesis: add chroma pre/post cosine DISTANCE feature

(a) HYPOTHESIS. Structural change to `splice/features.py` — add ONE new
    feature `chroma_cosine_dist` (FEATURE_NAMES 75→76). At each analysis
    position t_sec, compute the mean 12-dim `librosa.feature.chroma_stft`
    vector on pre-window [t−2, t] and post-window [t, t+2], then store
    the cosine DISTANCE (1 − cos_sim). Precompute the full-chunk chroma
    matrix inside `_ensure_feat_cache` (one librosa call per chunk,
    ~30–80 ms) and slice via the existing `_slice_frames` helper to
    keep per-t cost negligible. Wrapper auto-retrains via the
    feature-sha gate (US-505).

(b) WHY this direction over recent failures. The last 7 iterations —
    eb8984e, 55b21e6, 6b28235, 3e4e996, 3b365fe, plus the 273b8f5-
    865d92f-29c06cf DSP-gate keeps — have ALL operated on variations of
    the DSP SUM/MAX/CPE/pairwise floor. Singing is now the weakest
    domain at 0.360 (it was 0.388 pre-29c06cf) and the DSP-gate axis
    has stopped finding new precision without dropping real-splice
    recall. CLAUDE.md explicitly directs a structural change after 5+
    consecutive same-axis failures. Chroma is genuinely untried:
    history shows remote-half MFCC (timbre), spec_rolloff_far_delta
    (companion time-scale), file_hpr (tonality), per-bin phase
    residual, but ZERO chroma-based features. Chroma is physically
    motivated for the weakest-domain failure mode: singing chord
    transitions preserve KEY (C-major→A-minor still shares [C,E,G,A]
    pitch classes → small cosine distance ≤0.2), whereas cross-song
    splices change key entirely (different scale/mode → 0.3–0.7).
    GBM's current top-SHAP features
    (spec_rolloff_delta/spec_centroid_delta/spec_bandwidth_delta)
    measure spectral SHAPE shift, not pitch-class continuity, so
    chord transitions with a 2000-Hz rolloff jump all look like
    splices. Chroma is the scalar that says "the tonal content DIDN'T
    shift to a new key" even when the spectral shape jumped — exactly
    the chord-transition-FP discriminator. Orthogonal to every prior
    axis: not DSP (physical-signal z-scores), not spectral-delta
    (magnitude spectrum statistics), not MFCC (cepstral timbre), not
    f0 (single-pitch), not ENF/codec/mel-PCA. Blast radius: +1
    feature to FEATURE_NAMES, +1 block, +1 cache key in
    _ensure_feat_cache. Classifier sha auto-invalidates so wrapper
    retrains (~3 min). Per-t cost: slice 12×n_frames mean + 12-dim
    cosine, sub-ms. Per-chunk cost: ~30–80 ms for chroma_stft on
    60 s.

(c) IF THIS FAILS. Three paths. (1) If GBM assigns ~0 SHAP to the new
    feature (didn't help), replace with a PCP-based feature that
    weights the chroma vector by voiced-frame energy so silence/
    instrumental gaps don't dilute the key signature. (2) If it helps
    singing but hurts speech (chroma on speech is noise), HPR-gate
    the feature: zero it out when file_hpr<0.5. (3) Final escalation:
    add `chroma_persistence_far` — cosine distance between t-window
    chroma and [t+4, t+8] chroma, capturing "cross-source tonal
    content persists far after the cut" which discriminates a real
    source-change (far chroma stays different) from a chord cycle-
    back (far chroma returns to original).

(d) Information gaps. (i) Per-position feature values on the 2 singing
    / 4 korean clean FPs are STILL not in CURRENT STATE — the same
    blocker flagged in 10+ consecutive reflections. I am picking
    chroma vs. other candidate features from physical theory not
    observed failure values. (ii) Per-domain OOF weighted-F1 with vs
    without the candidate feature would let me preview whether chroma
    even has discriminative lift over the 75 existing features; no
    such per-feature OOF-lift oracle exists pre-commit. (iii)
    Training-data chunk chroma distributions per class (not_splice vs
    hard_cut vs crossfade) aren't surfaced — I can't predict whether
    training rows exhibit the theoretical signal separation.

(e) Wrapper enhancements. (1) `scripts/feature_oof_preview.py --add
    chroma_cosine_dist` that retrains once and reports per-domain OOF
    F1 delta vs current classifier, turning "is chroma worth adding"
    from a bet into a pre-commit numeric. (2) `CLEAN_FP_POSITIONS`
    JSON block in CURRENT STATE — SUPER-persistent blocker; per
    position (domain, file, t_sec, label_id, p_splice, dsp_phase_z,
    dsp_t2_z, dsp_cpe_z, top-5 most-influential features by absolute
    SHAP). Every feature-add hypothesis is guessing which scalar
    discriminates the failure set. (3)
    `scripts/feature_class_separability.py --feature FEATURE_NAME`
    that computes Welch's t statistic between not_splice and
    (hard_cut+crossfade) training rows for a candidate feature, so I
    can rank candidate features by training-time separability before
    committing to a retrain cycle.


---
## [auto-diagnosis]
diagnose: no-traceback
tsv: RESULTS_TSV: combined=<REDACTED> combined_mean=<REDACTED> combined_min=<REDACTED> clean_fp=6 n_datasets=3 combined_english=<REDACTED> clean_fp_english=0 combined_korean=<REDACTED> clean_fp_korean=4 combined_singing=<REDACTED> clean_fp_singing=2
note: eval log has no Python traceback; likely a normal discard. If combined collapsed below 0.05, check wrapper log for PIPELINE_FAILURE lines.

## 2026-04-20T02:28:04+09:00 — 9fe41d1 (discard, combined=0.493815)
subject: chroma persistence_far post-filter (drop emits where pre/far chroma cosine dist <0.10 on tonal content) — cited next-step from 634cdd2(c)(3). Pure detector.py change, no retrain. Compute librosa.feature.chroma_stft once per chunk in _build_chunk_context (~50ms/60s chunk, norm=None). After existing MAX>=2.0 + SUM>=5.0 DSP gate, for each surviving emit at t compute mean chroma over pre [t-2, t] and far_post [t+4, t+8]. Per-emit tonality gate: max(pre_chroma)/mean(pre_chroma) > CHROMA_TONALITY_MIN=3.0 (intrinsically routes filter away from speech — speech windows have flat chroma ratio 1.0-1.5; smoke-verified white-noise=1.11). When tonal, drop emit if cosine distance < CHROMA_FAR_DIST_MIN=0.10. Edge guard: skip gate when t<2 or t+8>chunk_dur. Targets 29c06cf's singing 0.360 (weakest domain) where 2 chord-transition FPs survive — chord progressions cycle back to root key within 4-8s in pop/rock so far_dist ~0.02-0.10; real cross-source splices land in different key/scale that PERSISTS far after cut so far_dist 0.30-0.70. Smoke-verified: pure 440Hz sine pre-tonality=8.44, far_dist=2e-6 (would drop, correct). 440->622Hz tritone shift pre-tonality=8.47, far_dist=1.0 (would keep, correct). White-noise far_dist not evaluated because tonality 1.11 < 3.0 short-circuits gate. Orthogonal to every prior axis: not DSP z-score (chroma is harmonic-class-12 PCP); not class-routed; not file-HPR-routed (per-emit local tonality, not file-level); not the 634cdd2 chroma_cosine_dist FEATURE (different time scale FAR vs LOCAL ±2s, different mechanism POST-FILTER vs learned GBM feature, different normalization norm=None vs default 'inf'); not pairwise (chroma is harmonic-content, pairwise is segment-distance block-structure). Risk-bounded by stacking on MAX+SUM gates and tonality pre-gate: can only DROP emits past DSP that have tonal content AND chord-cycle pattern; cannot create TPs. Cannot regress english (FP=0; the only risk is dropping a TP, but english files are speech with chroma tonality <2.0 so gate never fires). Per-emit cost: 12-dim mean + cosine, sub-ms. Per-chunk cost: ~50ms chroma_stft x ~3 chunks/file x 60 files = ~9s within 243/300s budget. Blast radius: detector.py only, ~80 lines (helpers + 6 constants + ctx field + emit-loop integration); features.py sha stable so classifier byte-identical (no retrain).
per-domain: combined_english=0.731707 combined_korean=0.457143 combined_singing=0.360000

# 2026-04-20 — hypothesis: chroma PERSISTENCE_FAR post-filter (cited next-step from 634cdd2(c)(3))

(a) HYPOTHESIS. Pure `splice/detector.py` change, no retrain, no feature
    change. Compute `librosa.feature.chroma_stft(norm=None)` once per
    chunk in `_build_chunk_context` (~50ms / 60s chunk). After the
    existing MAX>=2.0 + SUM>=5.0 DSP gate, for each surviving emit
    compute mean chroma vector on the pre window [t-2, t] and the
    FAR-post window [t+4, t+8]. If pre window's tonality is
    sufficient (`max(pre_chroma)/mean(pre_chroma) > CHROMA_TONALITY_MIN
    = 3.0` — intrinsic per-emit gate that excludes speech) AND the
    cosine distance between pre and far chroma vectors is below
    `CHROMA_FAR_DIST_MIN=0.10`, drop the emit. Edge-guard: skip the
    gate when t<2 or t+8 > chunk_dur (insufficient context).

(b) WHY this over recent failures. (i) CLAUDE.md mandate: 8+
    consecutive DSP-post-filter variants since 29c06cf have all failed
    to lift combined past 0.494. The DSP-gate axis is exhausted. (ii)
    634cdd2 added `chroma_cosine_dist` as a LOCAL pre/post (t±2s)
    FEATURE for GBM to learn — verify-fail at 0.488272 with singing
    IMPROVING 0.360→0.394 but english regressing 0.732→0.690. The
    chroma signal demonstrably helps singing; the failure mode was
    GBM globally over-weighting a feature that's noise on speech.
    (iii) Persistence_far is genuinely orthogonal to the failed local
    chroma feature on TWO axes simultaneously: TIME SCALE (4-8s far
    vs ±2s local) and MECHANISM (deterministic POST-FILTER vs learned
    GBM feature). The far window discriminates real source-change
    (new key persists) from chord cycle-back (chord progression
    returns to root within ~4s in pop/rock). (iv) Per-emit tonality
    gate (`max(pre_chroma)/mean(pre_chroma) > 3.0` with norm=None)
    prevents the speech-noise-chroma failure mode that hurt english
    in the feature attempt — speech windows have flat chroma (ratio
    1.0-1.5) so the gate doesn't fire on them, preserving english /
    korean recall. Singing chord transitions have ratio 5-12 and
    chord-cycle far-distance 0.02-0.10, exactly the band 0.10 cuts.
    (v) Risk-bounded by stacking on MAX+SUM gates and the tonality
    pre-gate: can only DROP emits that pass DSP AND have tonal
    content AND cycle back chromatically. Cannot create TPs.

    Orthogonal to every prior axis: not DSP (chroma is not a DSP
    z-score), not class-routed, not HPR-routed (chroma tonality is
    per-emit not file-level), not GBM-feature (POST-FILTER is
    deterministic), not the chroma_cosine_dist feature (different
    time scale and mechanism), not pairwise (chroma is harmonic-
    content not block-structure). Blast radius: detector.py only,
    ~30 lines (helper + 2 constants + ctx field + gate); features.py
    sha stable so classifier byte-identical (no retrain). Per-emit
    cost: 12-dim vector mean + cosine distance, sub-ms. Per-chunk
    cost: ~50ms chroma_stft × ~3 chunks/file × 60 files = ~9s within
    243/300s eval budget headroom.

(c) IF THIS FAILS. (1) If singing TPs drop (real singing crossfades
    have weak persistence_far because the new tonal content takes
    >8s to establish), LOWER threshold to 0.05 (looser bite). (2) If
    bite is too gentle (singing FPs surviving have far chroma
    distance ~0.12-0.20), RAISE threshold to 0.15 — catches deeper
    chord cycles at modest TP risk. (3) Final escalation: stack BOTH
    persistence_far AND local chroma POST-FILTERS — drop if EITHER
    LOCAL pre/post chroma_dist < 0.05 (immediate same-key) OR FAR
    persistence_dist < 0.10 (chord cycle-back).

(d) Information gaps. (i) Per-position chroma values on the 2 singing
    / 4 korean clean FPs surviving 29c06cf STILL not in CURRENT STATE
    — calibrating chroma threshold from theory not data. The chroma
    persistence_far distribution on real-splice TPs vs chord-
    transition FPs is theoretical. (ii) The 634cdd2 chroma feature
    attempt used librosa default norm='inf' (column max=1) which
    differs from norm=None needed for max/mean tonality ratio — without
    per-position chroma values I can't verify the tonality gate
    threshold 3.0 actually separates singing music from speech.
    (iii) ANALYSIS_WINDOW_S=60 with 30s overlap means edge-effect
    emits at chunk boundaries lose the gate entirely; how many emits
    is that? unknown.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE: per-position (domain, file, t_sec, label_id,
    p_splice, dsp_phase_z, dsp_t2_z, dsp_cpe_z, dsp_pairwise_proximity,
    chroma_pre, chroma_far, chroma_local_dist, chroma_far_dist,
    file_hpr) for every clean FP on the last keep. THE persistent
    blocker — every detector-side hypothesis for 12+ iterations is
    calibrating thresholds from theory rather than observed values.
    (2) `scripts/chroma_probe.py --keep-sha <sha> --thresholds
    "0.05,0.10,0.15"` that replays the last keep's emit trace through
    a candidate chroma POST-FILTER and reports per-domain (TPs_lost,
    FPs_dropped). Turns "0.05 vs 0.10 vs 0.15" from a guess into a
    numeric pre-commit decision. (3) Chunk-edge emit count per file:
    how many emits land in [chunk_dur - 8, chunk_dur] where
    persistence_far gates can't apply.

## 2026-04-20T02:32:40+09:00 — 68004ca (verify-fail, combined=0.493815)
subject: add tonality-gated chroma_local_dist_tonal feature (75->76)
per-domain: combined_english=0.731707 combined_korean=0.457143 combined_singing=0.360000

# 2026-04-20 — hypothesis: tonality-GATED chroma local cosine-distance FEATURE (FEATURE_NAMES 75→76)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new
    feature `chroma_local_dist_tonal`. Compute `librosa.feature.chroma_stft
    (norm=None)` once per chunk (~50ms / 60s chunk), cache as `feat_chroma`
    with hop 512. Per-position: slice mean 12-dim chroma over pre=[t−2, t]
    and post=[t, t+2]; compute pre-window tonality ratio
    `max(pre_chroma)/mean(pre_chroma)`. If ratio ≥ 3.0 (tonal content)
    return cosine distance between pre and post; else return 0.0 (sentinel).
    FEATURE_NAMES 75→76 so the wrapper auto-retrains via US-505 sha gate.

(b) WHY this over recent failures. (i) CLAUDE.md explicitly directs a
    structural change after 5+ consecutive same-axis failures — the DSP-
    gate axis has burned 9 attempts since 29c06cf without lifting combined
    past 0.494, and four of the most recent (55b21e6 HPR-gated SUM, 9fe41d1
    chroma persistence_far post-filter, 29c06cf itself, plus this keep)
    produced the IDENTICAL 0.493815 / singing=0.360 / korean=0.457 /
    english=0.732 triple — the detector-side signal has flatlined. (ii)
    634cdd2 is a directly-relevant data point: UNGATED chroma_cosine_dist
    as a feature lifted SINGING 0.360→0.394 (the target) but REGRESSED
    english 0.732→0.690 and korean 0.457→0.428, because librosa's default
    `norm='inf'` (column-max) destroys tonality structure and GBM globally
    over-weighted a feature that carries signal on tonal content but pure
    noise on speech. The signal for singing is REAL — the problem was
    collateral damage on speech domains, not feature validity. (iii)
    Tonality-gating addresses the failure mode directly: speech windows
    (flat chroma, max/mean ~1.0-1.5) get a ZERO sentinel, so GBM cannot
    base any split on chroma for speech — the feature is functionally
    invisible on english/korean. Singing chord transitions (max/mean
    5-12, same-key cycle-back distance 0.02-0.15) and real cross-song
    splices (max/mean 5-12, cross-key distance 0.3-0.7) both produce
    nonzero values whose distribution GBM can learn. norm=None preserves
    magnitude so the tonality ratio is computable. (iv) Orthogonal to
    every prior axis: not DSP (physical z-scores), not spectral delta
    (magnitude shape), not MFCC (cepstral timbre), not f0/voicing, not
    ENF/codec/mel-PCA, NOT ungated chroma_cosine_dist (634cdd2 — different
    normalization AND no gating), NOT chroma persistence_far post-filter
    (9fe41d1 — deterministic post-filter on 4-8s window, different time
    scale, different mechanism). Blast radius: +1 feature to FEATURE_NAMES,
    +1 cache key, +1 per-position block; classifier sha auto-invalidates
    so wrapper retrains (~3 min). Per-t cost: 12-dim mean + cosine, sub-ms.
    Per-chunk cost: one chroma_stft call ~30-80ms, absorbed within
    243/300s budget.

(c) IF THIS FAILS. (1) If GBM assigns ~0 SHAP to the feature (gating was
    TOO aggressive and zeroed-out too many borderline tonal windows),
    LOWER the tonality threshold to 2.0, letting korean voiced-vowel
    windows contribute. (2) If speech domains still regress (the feature
    still leaks signal into english/korean via some tonal windows — short
    sung prompts, emphatic vowels), raise the tonality threshold to 4.0
    AND require file-level voicing_prob > 0.5 as a secondary gate. (3)
    Final escalation: replace the single gated feature with a PAIR —
    `chroma_local_dist_tonal` (as proposed) PLUS `chroma_far_dist_tonal`
    (pre=[t−2, t] vs far=[t+4, t+8]) — the latter captures "new tonal
    content PERSISTS post-cut" which discriminates real source-change
    from chord cycle-back.

(d) Information gaps. (i) Per-position CLEAN_FP_POSITIONS (domain, file,
    t_sec, chroma_pre, chroma_post, tonality_ratio, cos_dist) on the 2
    singing / 4 korean FPs STILL not in CURRENT STATE — tonality threshold
    3.0 is calibrated from theory (white-noise 1.1, pure sine 8.4) not
    observed failure values. (ii) Training-chunk chroma distribution per
    class (not_splice / hard_cut / crossfade) not surfaced — cannot
    preview whether training rows exhibit the theoretical gated
    separability. (iii) The recurrence of EXACT 0.493815 across 55b21e6,
    9fe41d1, and 29c06cf raises a pipeline question: are eval discards
    sometimes byte-reusing a cached detector output? A CLASSIFIER_SHA or
    DETECTOR_EMIT_COUNT line in RESULTS_TSV would disambiguate "gate
    didn't fire" from "no change was applied".

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block auto-emitted
    by `splice/evaluate.py` — persistent blocker across 12+ detector
    hypotheses. Per-FP (domain, file, t_sec, label_id, p_splice, TOP-5
    feature values with SHAP > 100 on that domain) would flip every
    subsequent hypothesis from theory-driven to data-driven. (2)
    `scripts/feature_oof_preview.py --add <feature_fn>` that trains once
    and reports per-domain OOF-F1 delta vs current classifier — turns
    "is this feature worth a retrain" from a bet into a numeric. (3)
    `DETECTOR_EMIT_TRACE` JSONL alongside results.tsv with (domain, file,
    t_sec, p_splice, pre_dsp_max, pre_dsp_sum, survives_gates) — would
    let post-mortems distinguish "filter dropped no emits" from "filter
    dropped emits but all were true splices".


---
## [auto-diagnosis]
diagnose: no-traceback
tsv: RESULTS_TSV: combined=<REDACTED> combined_mean=<REDACTED> combined_min=<REDACTED> clean_fp=6 n_datasets=3 combined_english=<REDACTED> clean_fp_english=0 combined_korean=<REDACTED> clean_fp_korean=4 combined_singing=<REDACTED> clean_fp_singing=2
note: eval log has no Python traceback; likely a normal discard. If combined collapsed below 0.05, check wrapper log for PIPELINE_FAILURE lines.

## 2026-04-20T02:48:27+09:00 — 49bd0b1 (keep, combined=0.505589)
subject: add voiced_mfcc_cosine_dist feature (FEATURE_NAMES 75->76) — voiced-frame-only MFCC cosine distance between pre[t-2,t] and post[t,t+2]. Reuses cached feat_mfcc + feat_vp; per-t cost is two slices + masked mean + cosine, sub-ms. Sentinel 0.0 when either window has zero voiced frames. Targets 29c06cf's singing 0.360 weakest-domain plateau where 9 consecutive DSP-post-filter variants since 29c06cf failed (3 reproducing exact 0.493815) and CLAUDE.md mandates structural change after 5+ same-axis failures. Mechanism: existing mfcc_delta_NN averages over ALL frames so chord-transition spectral envelope shifts trigger it on FPs; restricting to voiced frames isolates vocal-tract / instrument timbre — same singer across a chord boundary keeps voiced MFCC similar (cosine_dist ~0.05), real cross-source splice shifts it (~0.30+). On speech, voiced=vowels and pre/post may have different vowels regardless of splice → high-variance noise; GBM learns low SHAP per-domain via tree splits so feature is functionally invisible on english/korean. Self-gates via voiced flag requirement so won't regress speech (the 634cdd2/68004ca chroma verify-fail mode required external tonality-gate which lacked clean signal). Orthogonal to every prior axis: not DSP, not spectral delta, not chroma (cepstral not pitch-class), not pairwise, NOT remote-half MFCC (438080/411473 used long half-chunk averages dominated by file-mean character; this uses LOCAL +-2s VOICED-RESTRICTED mean), NOT raw mfcc_delta (different masking — voiced-only vs all-frames). Pure features.py change; FEATURE_NAMES count gate triggers wrapper auto-retrain via US-505. Smoke: import + assert len(FEATURE_NAMES)==76 + last name 'voiced_mfcc_cosine_dist'.
per-domain: combined_english=0.835443 combined_korean=0.498462 combined_singing=0.310345

# 2026-04-20 — hypothesis: voiced-frame-only MFCC cosine-distance feature (FEATURE_NAMES 75→76)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new
    feature `voiced_mfcc_cosine_dist`. Per position t_sec, restrict the
    MFCC mean to frames where `vp==1.0` (precomputed voiced flag in
    `feat_vp`) inside pre=[t−2, t] and post=[t, t+2], then return cosine
    distance between the two 13-dim voiced-mean vectors. If either window
    has zero voiced frames, return 0.0 sentinel. Reuses cached `feat_mfcc`
    + `feat_vp`; per-t cost is two slices + masked mean + cosine, sub-ms.
    FEATURE_NAMES 75→76 so wrapper auto-retrains via US-505 sha gate.

(b) WHY this over recent failures. CLAUDE.md mandates STRUCTURAL after
    5+ same-axis failures — the DSP-post-filter axis has burned 9
    attempts since 29c06cf, and three recent attempts (55b21e6 HPR-gated
    SUM, 9fe41d1 chroma persistence_far post-filter, 68004ca tonality-
    gated chroma feature) reproduced the IDENTICAL 0.493815 / singing=
    0.360 / korean=0.457 / english=0.732 triple — gates simply don't
    fire on the eval emit population. Existing `mfcc_delta_NN` (block 2)
    averages MFCC over ALL frames in pre/post — silences, breaths,
    consonants, sustains all blended. On singing chord transitions, the
    RAW pre/post MFCC mean shifts because the spectral envelope shifts
    even without singer change → mfcc_delta fires on FPs. Restricting
    to VOICED frames isolates the VOCAL-TRACT / INSTRUMENT TIMBRE
    signature: same singer, same room → voiced MFCC stays similar
    across a chord boundary (same vowel formants, same vibrato
    envelope, same mic coloration), cosine_dist ~0.05. Real cross-
    source splice: different singer / different mic → voiced MFCC
    shifts, cosine_dist ~0.30+. On speech (korean/english), voiced =
    vowels, and pre/post may have different vowels regardless of
    splice → feature value high-variance noise; GBM learns low SHAP
    on speech domains via tree splits, so feature is functionally
    invisible on those. Orthogonal to every prior axis: not DSP, not
    spectral delta, not chroma (cepstral not pitch-class), not pairwise,
    not HPR-routed, NOT remote-half MFCC (438080/411473 used long
    half-chunk averages dominated by file-mean character; this uses
    LOCAL ±2s VOICED-RESTRICTED mean), NOT raw mfcc_delta (different
    masking — voiced-only vs all-frames). Risk-bounded: zero-voiced
    sentinel = 0.0 prevents NaN; verify-fail history (634cdd2/68004ca
    chroma) was on a feature that LACKED voicing-conditioning, so
    speech regressed; this feature SELF-GATES via the voiced flag
    requirement.

(c) IF THIS FAILS. (1) If GBM assigns ~0 SHAP (voiced-frame mask too
    strict — drops too many frames in vibrato/staccato singing),
    relax to ALL frames where rms > 0.5*median (energy-floor mask
    instead of f0-validity mask). (2) If it lifts singing but
    regresses speech (false discriminator on different vowels pre/
    post), add a SECOND feature `voiced_frame_overlap` = fraction
    of frames voiced in pre+post; GBM learns the JOINT (high voiced
    overlap → trust the cosine_dist; low overlap → ignore). (3)
    Final escalation: replace cosine with WASSERSTEIN-1 distance
    between per-frame voiced MFCC distributions (captures shape
    not just mean — same singer with strong vibrato has wider
    distribution than monotone, distance discriminates beyond mean
    shift).

(d) Information gaps. (i) Per-position CLEAN_FP_POSITIONS (domain,
    file, t_sec, vp_pre_count, vp_post_count, voiced_mfcc_pre,
    voiced_mfcc_post, cosine_dist) STILL not in CURRENT STATE — same
    persistent blocker for 13+ detector-side hypotheses. I'm
    calibrating the voiced-mask mechanism from physical theory
    rather than observed FP voicing patterns. (ii) The exact-
    triple reproduction of 0.493815 across 3+ recent commits
    raises a wrapper question: are these hypotheses actually
    landing in the eval-time classifier? A DETECTOR_BUNDLE_SHA
    line in RESULTS_TSV would let me distinguish "no-bite" from
    "no-load". (iii) Per-domain weighted-F1 OOF lift from adding
    a candidate feature is not previewable pre-commit; I'm betting
    a 3-min retrain on theory.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block in
    CURRENT STATE with per-FP (domain, file, t_sec, label_id,
    p_splice, top-5 features by |SHAP|, voiced-frame counts in
    pre/post windows). Single biggest blocker across 13+ hypotheses
    — flips threshold/mask/feature design from theory-driven to
    data-driven. (2) `scripts/feature_oof_preview.py --add
    <feature_fn>` retrains once and reports per-domain OOF-F1
    delta vs current — turns "is this feature worth a retrain"
    from a bet into a numeric. (3) DETECTOR_BUNDLE_SHA + EMIT_COUNT
    line in RESULTS_TSV / wrapper log — disambiguates the
    "exact-triple reproduction" mystery (is the gate not firing,
    or is the wrapper byte-reusing a stale joblib?).
[auto] (no SHAP data for either 29c06cf or 49bd0b1)

## 2026-04-20T03:04:41+09:00 — 32cac36 (keep, combined=0.532261)
subject: add voiced_chroma_cosine_dist feature (FEATURE_NAMES 76->77) — voiced-frame-only chroma (pitch-class profile) cosine distance between pre[t-2,t] and post[t,t+2]. Reuses cached feat_vp plus a new per-chunk feat_chroma (one librosa.feature.chroma_stft call, hop/fft same as MFCC). Per-t cost is two slices + voiced-mask mean + cosine, sub-ms. Sentinel 0.0 when either window has zero voiced frames, matching 49bd0b1 pattern. Targets singing 0.310 weakest-domain regression caused by the just-kept voiced_mfcc_cosine_dist (49bd0b1 lifted english 0.732->0.835 and korean 0.457->0.498 via SPEAKER/INSTRUMENT-timbre continuity signal but dropped singing 0.360->0.310 because same-singer chord transitions keep voiced MFCC stable and the feature lowers p_splice on real crossfade TPs too). Mechanism: chroma is the canonical MIR pitch-class profile — within a song's key chord transitions share 3-5 of 12 pitch classes so voiced-chroma cosine distance is small (0.05-0.20), cross-song splices change key entirely giving 0.30-0.70. 634cdd2 proved ungated chroma_cosine_dist lifts singing 0.360->0.394 — the signal is real — but raw chroma regressed english/korean because GBM globally over-weighted a feature that is per-vowel prosodic noise on speech. Voicing-mask self-gating mirrors 49bd0b1: on speech voiced=vowels whose chroma swings with prosody regardless of splice -> high-variance noise -> GBM assigns low per-domain SHAP via tree splits, feature functionally invisible. On singing voiced=sustained sung notes whose pitch class IS the key signature -> GBM learns a clean discriminator. Orthogonal to every prior axis: NOT 634cdd2 (no mask), NOT 68004ca tonality-gated chroma (external max/mean ratio, verify-fail), NOT 9fe41d1 chroma persistence_far (post-filter, 4-8s window), NOT 49bd0b1 voiced_mfcc (cepstral not pitch-class — smoke confirms on 440Hz->C-chord: chroma dist=0.865 vs mfcc dist=0.014, orthogonal signals). Pure features.py change; FEATURE_NAMES count gate triggers wrapper auto-retrain via US-505. Smoke: import + assert len(FEATURE_NAMES)==77 + last name 'voiced_chroma_cosine_dist' + synthetic chord-shift yields dist=0.865.
per-domain: combined_english=0.839506 combined_korean=0.526154 combined_singing=0.341379

# 2026-04-20 — hypothesis: voiced-frame-only CHROMA cosine distance feature (FEATURE_NAMES 76→77)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE feature
    `voiced_chroma_cosine_dist` parallel to the just-kept
    `voiced_mfcc_cosine_dist` but on the pitch-class axis instead of
    cepstral. Precompute `librosa.feature.chroma_stft(y=audio, sr,
    hop_length=512, n_fft=2048)` once per chunk (~30–60 ms). Per
    position: slice voiced-flag (`feat_vp`) and chroma frames on
    pre=[t−2, t] and post=[t, t+2]; keep only frames where `vp==1.0`;
    return cosine distance of voiced-only mean chroma vectors. Sentinel
    0.0 when either window has zero voiced frames. FEATURE_NAMES
    76→77 triggers wrapper auto-retrain via US-505 sha gate.

(b) WHY this over the last keep. 49bd0b1 (current keep, 0.506)
    lifted english 0.732→0.835 and korean 0.457→0.498 but REGRESSED
    singing 0.360→0.310 — now the weakest domain by a wide margin.
    Voiced-mask over MFCC picked up SPEAKER/INSTRUMENT timbre
    continuity which is exactly what speech FPs needed (different
    speakers across a splice shift MFCC; same speaker across a pause
    doesn't); on singing, same-singer chord transitions keep voiced
    MFCC stable so the feature lowers p_splice on real crossfades
    too. Singing needs a signal tied to MUSICAL CONTENT (key/mode)
    not vocal identity. Chroma is the canonical MIR pitch-class
    profile: within a song's key most chords share 3–5 of 12 pitch
    classes so voiced-chroma cosine distance across a chord
    transition is small (0.05–0.20); cross-song splices change key
    (different scale/mode) giving 0.30–0.70. 634cdd2 already proved
    raw chroma_cosine_dist lifts singing 0.360→0.394 — the signal
    is real — but ungated chroma regressed english/korean because
    speech windows carry pitch-class noise that GBM globally
    over-weighted. Self-gating via the voicing mask mirrors the
    49bd0b1 success pattern: on speech voiced=vowels whose
    per-vowel chroma content swings with prosody regardless of
    splice → high-variance noise → GBM assigns low per-domain SHAP
    via tree splits, feature functionally invisible on
    english/korean. On singing voiced=sustained sung notes whose
    pitch class IS the key signature → GBM learns a clean
    discriminator. Orthogonal to 634cdd2 (no mask, all frames) and
    68004ca (per-position max/mean tonality gate, verify-fail) and
    9fe41d1 (post-filter not feature, 4–8s far window not ±2s
    local) and 49bd0b1 (voiced-frame mask on MFCC, cepstral not
    pitch-class). Blast radius: +1 feature + 1 cache key + 1 block
    in features.py; per-t cost ~1 ms (mean + cosine on 12-dim
    vectors); per-chunk cost ~30–60 ms chroma_stft within the
    243/300 s eval budget. Risk-bounded: zero-voiced sentinel = 0.0
    matches the 49bd0b1 pattern so no NaN leak.

(c) IF THIS FAILS. (1) If GBM assigns ~0 SHAP (voiced mask too
    permissive on speech, so the feature does become an english
    regressor and GBM compromises by ignoring it), replace the
    voicing mask with an HPSS-HARMONIC mask — keep only frames
    where harmonic_energy / (harmonic+percussive) > 0.7, which on
    speech yields very few frames → stronger self-gate. (2) If it
    lifts singing but regresses speech, add a second feature
    `voiced_chroma_dist_far` (pre=[t−2, t] vs far=[t+4, t+8])
    so GBM sees a joint (near, far) signature — chord cycle-backs
    have high near distance but low far distance (key returns);
    real cross-song has both high. (3) Final escalation: use
    KL-divergence between voiced chroma distributions
    (L1-normalized 12-dim PMFs) instead of cosine — KL weights
    single dominant pitch classes more strongly which better
    captures modal scale changes.

(d) Information gaps. (i) Per-position CLEAN_FP_POSITIONS still
    not in CURRENT STATE — 14+ consecutive hypotheses calibrated
    feature/mask/threshold choices from theory not observed
    values. I can't verify whether the 2 singing FPs on 49bd0b1
    have vp_pre/vp_post > 0 (if they land on instrumental-only
    passages the voiced mask is empty and the feature is a no-op
    for them). (ii) Rolling SHAP rollup reported as "no keeps yet
    — rollup empty" for the current keep, so I can't see which
    features 49bd0b1 actually relied on per domain — the singing
    regression cause is theoretical. (iii) Per-domain OOF weighted
    F1 with-vs-without candidate features is not previewable
    pre-commit so I'm betting a 3-min retrain on chroma vs HPSS-
    harmonic-mask vs KL-on-voiced-chroma without empirical
    ranking.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block —
    persistent blocker for 14+ hypotheses; per-FP (domain, file,
    t_sec, label_id, p_splice, vp_pre_count, vp_post_count,
    dsp_phase_z, dsp_t2_z, dsp_cpe_z, top-5 |SHAP| features with
    values) would flip mask design from theory to data. (2)
    `scripts/feature_oof_preview.py --add <feature_fn>` that
    trains once and reports per-domain OOF-F1 delta vs current
    — turns "is this feature worth a 3 min retrain" into a
    numeric. (3) Per-keep SHAP rollup populated immediately on
    keep commit (the "no keeps yet — rollup empty" marker means
    I have no per-domain SHAP for 49bd0b1, the freshest keep
    and the one whose regression I'm trying to understand).
[auto] (no SHAP data for either 49bd0b1 or 32cac36)

## 2026-04-20T03:19:30+09:00 — ff65865 (discard, combined=0.490508)
subject: add voiced_chroma_far_cosine_dist feature (FEATURE_NAMES 77->78) — FAR-window companion to 32cac36's voiced_chroma_cosine_dist. Same mechanism (voiced-mask restricted mean 12-dim chroma + cosine distance) but pre=[t-2,t] vs post_far=[t+4,t+8] so GBM sees the (near, far) TIME-SCALE JOINT. Reuses cached feat_chroma + feat_vp — zero new librosa calls. Edge guard: t+8 > chunk_dur returns sentinel 0.0 (matches 9fe41d1 pattern). Sentinel 0.0 also on zero-voiced-frame windows and zero-norm vectors, matching 49bd0b1/32cac36 safety pattern. Pure features.py change; FEATURE_NAMES count bump triggers wrapper auto-retrain via US-505 sha gate. Targets 32cac36 current-keep (combined=0.532) where singing remains weakest at 0.341. Mechanism: chord cycle-backs within a song's 4-bar progression (I-vi-IV-V etc.) produce high NEAR chroma distance (near feature fires ~0.3-0.5 at each chord transition) BUT the progression returns to the root chord within 4-8s so FAR-window chroma comes back to the same key signature (far_dist ~0.05-0.15). Real cross-song splices land in a different key/mode that PERSISTS 4-8s after the cut (far_dist ~0.3-0.7). GBM with only the near feature cannot distinguish these; with the (near, far) joint, tree splits learn "near>0.25 AND far<0.15 -> chord cycle (NOT splice)" vs "near>0.25 AND far>0.25 -> real cross-song (splice)". Voicing-mask self-gating mirrors 49bd0b1/32cac36: on speech voiced=vowels whose far-window chroma swings with prosody regardless of splice -> high-variance noise -> GBM assigns low per-domain SHAP, feature functionally invisible on english/korean. On singing voiced=sustained sung notes whose pitch class IS the key signature -> clean discriminator. Cited option (2) from 32cac36(c)(2): "add a second feature voiced_chroma_dist_far (pre=[t-2,t] vs far=[t+4,t+8]) so GBM sees a joint (near, far) signature". Orthogonal to every prior axis: NOT 634cdd2 (no mask, +/-2s local), NOT 68004ca (external tonality-ratio gate, +/-2s local), NOT 9fe41d1 (POST-FILTER not feature, no voicing mask, default-norm chroma with tonality pre-gate), NOT 32cac36 (+/-2s LOCAL window, same mechanism DIFFERENT TIME SCALE), NOT 49bd0b1 (cepstral not pitch-class), NOT spec_rolloff_far_delta (spectral not pitch-class). First time-scale-paired companion feature in the voiced-chroma axis. Blast radius: features.py only — one new block + FEATURE_NAMES append. Per-t cost: 2 slices + voiced-mask mean + cosine on 12-dim vectors, sub-ms. Risk-bounded: edge-guard sentinel + voiced-empty sentinel + zero-norm sentinel prevent NaN; self-gating via voicing mask reproduces the 32cac36 safety on speech domains. Smoke: len(FEATURE_NAMES)==78, last name 'voiced_chroma_far_cosine_dist', tritone-shift chroma yields dist=1.0, chord-cycle-back yields dist=0.0, t+8>chunk_dur edge guard returns 0.0, unvoiced windows return 0.0.
per-domain: combined_english=0.683544 combined_korean=0.495652 combined_singing=0.348333

# 2026-04-20 — hypothesis: voiced_chroma_far_cosine_dist feature (FEATURE_NAMES 77→78)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE feature
    `voiced_chroma_far_cosine_dist`. Mechanism identical to 32cac36's
    `voiced_chroma_cosine_dist` but on a FAR post-window: pre=[t−2, t]
    vs post_far=[t+4, t+8], voiced-mask restricted, cosine distance of
    voiced-only mean 12-dim chroma vectors. Reuses existing `feat_chroma`
    and `feat_vp` caches — zero new librosa calls. Edge guard: if
    `t_sec + 8.0 > chunk_dur` return sentinel 0.0 (matches 9fe41d1
    pattern). Sentinel 0.0 when either window has zero voiced frames,
    matching 49bd0b1/32cac36. FEATURE_NAMES 77→78 triggers wrapper
    auto-retrain via US-505 sha gate.

(b) WHY this over the current state. 32cac36 (current keep, 0.532261)
    lifted combined from 0.506→0.532 by adding voiced_chroma at ±2s,
    partially recovering singing from 0.310→0.341 but singing is STILL
    weakest. The remaining singing gap is chord cycle-backs — within a
    song's 4-bar chord progression (I–vi–IV–V etc.), chord transitions
    produce high LOCAL chroma distance (near feature fires ~0.3–0.5)
    but the chord cycle returns to root within 4–8s so FAR-window
    chroma comes back to the same key signature. Real cross-song
    splices land in a DIFFERENT key/mode that PERSISTS far after the
    cut. GBM with only the near feature cannot distinguish these two;
    with the (near, far) joint, tree splits can learn
    "near_dist>0.25 AND far_dist<0.15 → chord cycle (NOT splice)" vs
    "near_dist>0.25 AND far_dist>0.25 → real cross-song (splice)".

    Explicit cited next-step in 32cac36(c)(2): "add a second feature
    `voiced_chroma_dist_far` (pre=[t−2, t] vs far=[t+4, t+8]) so GBM
    sees a joint (near, far) signature — chord cycle-backs have high
    near distance but low far distance (key returns); real cross-song
    has both high." Voicing-mask self-gating mirrors the 49bd0b1 /
    32cac36 pattern: on speech voiced=vowels whose far-window chroma
    swings with prosody → high-variance noise → GBM assigns low
    per-domain SHAP, feature functionally invisible on english /
    korean. On singing voiced=sustained sung notes whose pitch class
    IS the key signature → clean discriminator.

    Orthogonal to every prior axis: NOT 634cdd2 (no mask, ±2s local),
    NOT 68004ca (external tonality-ratio gate, ±2s local, verify-fail),
    NOT 9fe41d1 (deterministic POST-FILTER not feature, no voicing
    mask, default-norm chroma with tonality pre-gate), NOT 32cac36
    (±2s LOCAL window, same mechanism DIFFERENT TIME SCALE), NOT
    49bd0b1 (cepstral not pitch-class), NOT spec_rolloff_far_delta
    (spectral not pitch-class). The distinguishing axis here is
    TIME SCALE paired with the existing near feature — NO prior
    feature has given GBM a near/far chroma pair.

    Blast radius: features.py only — one new block + one cache-key
    reuse + FEATURE_NAMES append. Zero new librosa calls (reuses
    feat_chroma from 32cac36). Per-t cost: 2 slices + voiced-mask
    mean + cosine distance on 12-dim vectors, sub-ms. Classifier
    sha auto-invalidates so wrapper retrains (~3 min). Risk-bounded:
    zero-voiced sentinel=0.0 and edge-guard sentinel=0.0 prevent
    NaN; self-gating via voicing mask reproduces the 32cac36 safety
    on speech domains.

(c) IF THIS FAILS. (1) If singing TPs regress (real cross-song
    splices sometimes have weak far-window pitch-class persistence
    because the new song's vocal starts mid-phrase with unvoiced
    consonants), widen to symmetric pre_far=[t−8, t−4] vs
    post_far=[t+4, t+8] — more robust to asymmetric voicing.
    (2) If GBM assigns ~0 SHAP (joint interaction not learned at
    max_depth=3 with n_estimators=200), raise max_depth to 4 or
    n_estimators to 300 to give boosting deeper joint-feature
    capacity. (3) Final escalation: replace cosine with
    jensen-shannon divergence between pitch-class histograms
    (normalize per column then JS on voiced mean) — JS emphasizes
    modal-scale shifts (3+ pitch classes differ) over single-
    chord shifts (1–2 pitch classes differ).

(d) Information gaps. (i) Per-position CLEAN_FP_POSITIONS still not
    in CURRENT STATE — 15+ consecutive hypotheses calibrated from
    theory not observed values. I cannot verify whether 32cac36's
    2 surviving singing FPs have t+8 < chunk_dur (edge guard may
    make the feature a no-op for them). (ii) SHAP rollup reports
    "no keeps yet — rollup empty" for both recent keeps (49bd0b1
    and 32cac36) — I cannot see which features 32cac36 actually
    relied on per domain, so the chord-cycle-back hypothesis is
    theoretical. (iii) Training-chunk voiced_chroma_dist distributions
    per class (not_splice vs hard_cut vs crossfade) not surfaced;
    cannot preview whether training rows exhibit the theoretical
    near/far joint separability.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block —
    persistent blocker for 15+ hypotheses. Per-FP (domain, file,
    t_sec, label_id, p_splice, chunk_dur, voiced_chroma_dist_near,
    voiced_mfcc_dist, top-5 |SHAP| features with values) would
    flip feature/mask design from theory to data. (2)
    `scripts/feature_oof_preview.py --add <feature_fn>` that trains
    once and reports per-domain OOF-F1 delta — turns "is this
    feature worth a 3-min retrain" into a numeric. (3) Per-keep
    SHAP rollup populated IMMEDIATELY on keep commit. The
    "no keeps yet — rollup empty" marker means I'm forming hypotheses
    with zero per-domain SHAP data on the classifier I'm trying to
    improve — the biggest information gap in the entire autoresearch
    loop.

## 2026-04-20T03:35:28+09:00 — 308aa5a (discard, combined=0.522259)
subject: add voiced_tonnetz_cosine_dist feature (FEATURE_NAMES 77->78) — voiced-frame-only tonnetz (6-dim tonal-centroid projection of chroma) cosine distance between pre[t-2,t] and post[t,t+2]. Reuses cached feat_chroma (32cac36); adds one feat_tonnetz cache via librosa.feature.tonnetz(chroma=feat_chroma) — a ~0.4ms 6x12 matmul, zero new librosa STFT calls. Per-t cost is 2 slices + voiced-mask mean + cosine on 6-dim vectors, sub-ms. Sentinel 0.0 when either window has zero voiced frames, matching 49bd0b1/32cac36 safety pattern. Targets 32cac36's singing 0.341 weakest-domain plateau: 32cac36 proved voiced-chroma near-window works but singing still drags; ff65865 tried FAR-window voiced chroma and regressed english 0.840->0.684 (far-window speech-vowel chroma is noisy, breaks self-gating), ruling out time-scale companions on the voiced-chroma axis. Tonnetz is a 6-dim LINEAR projection of chroma onto perfect-fifths / major-thirds / minor-thirds axes — measures HARMONIC RELATIONSHIP not raw pitch-class overlap. Within a song's key chord progressions (I-IV-V-I) stay tight in tonnetz space; cross-song splices to a different key move across it. Smoke-verified on synthetic sine chords: C->G (in-key fifth) tonnetz_dist=0.95 vs chroma_dist=0.52; C->F# (out-of-key tritone) tonnetz_dist=1.82 vs chroma_dist=0.68. Tonnetz has ~50%% better in-key-vs-out-of-key discrimination ratio (1.92x vs chroma's 1.30x). GBM max_depth=3 cannot learn the 12->6 linear projection itself; providing tonnetz as an explicit feature gives the geometric embedding directly. Near window (+/-2s) preserves 32cac36 self-gating on speech: voiced=vowels whose per-vowel tonnetz swings with prosody -> high-variance noise -> GBM assigns low per-domain SHAP on english/korean, feature functionally invisible; on singing voiced=sustained notes whose tonnetz IS the key signature -> clean discriminator. Orthogonal to every prior axis: NOT 634cdd2 (raw chroma no mask), NOT 68004ca (external tonality-ratio gate, verify-fail), NOT 9fe41d1 (chroma persistence_far POST-FILTER 4-8s), NOT 32cac36 (voiced chroma — pitch-class not tonal-centroid; tonnetz geometrically distinct from 12-dim PCP), NOT ff65865 (far window), NOT 49bd0b1 (cepstral). Pure features.py change; FEATURE_NAMES count gate triggers wrapper auto-retrain via US-505. Smoke: len(FEATURE_NAMES)==78, last name 'voiced_tonnetz_cosine_dist', synthetic tritone-shift at t=4s yields dist=1.82, synthetic noise yields sentinel 0.0.
per-domain: combined_english=0.850000 combined_korean=0.490909 combined_singing=0.341379

# 2026-04-20 — hypothesis: voiced_tonnetz_cosine_dist feature (FEATURE_NAMES 77→78)

(a) HYPOTHESIS. Structural `splice/features.py` change — add ONE new
    feature `voiced_tonnetz_cosine_dist`. Precompute
    `librosa.feature.tonnetz(chroma=feat_chroma)` once per chunk
    (reuses the chroma cache added in 32cac36 — zero new librosa
    calls beyond a ~0.4 ms 6×12 matmul); cache as `feat_tonnetz`.
    Per position t_sec: slice voiced-flag `feat_vp` and tonnetz
    frames on pre=[t−2, t] and post=[t, t+2]; keep only frames where
    `vp==1.0`; return cosine distance of voiced-only mean 6-dim
    tonnetz vectors. Sentinel 0.0 when either window has zero
    voiced frames or zero-norm vectors, matching the 49bd0b1 /
    32cac36 safety pattern. FEATURE_NAMES 77→78 triggers wrapper
    auto-retrain via US-505 sha gate.

(b) WHY this over recent failures. 32cac36 (current keep, 0.532)
    proved voiced-chroma near-window lifts singing but singing
    REMAINS weakest at 0.341. ff65865 (discard, 0.490) tried a
    FAR-window voiced chroma companion and REGRESSED english
    0.840→0.684 — voicing-mask self-gating fails on speech at
    far windows because speech far-vowel chroma is noisy. That
    rules out further time-scale companions on the voiced-
    chroma axis. Tonnetz is a 6-dim LINEAR projection of
    chroma onto the tonal-centroid space (perfect-fifths axis,
    major-thirds axis, minor-thirds axis) — functionally it
    measures HARMONIC RELATIONSHIP not raw pitch-class overlap.
    Within a song's key chord progressions (I-IV-V-I) stay in
    a tight tonnetz region because every chord is harmonically
    related to the tonic. Cross-song splices that land in a
    different key move across tonnetz space. Smoke-verified on
    synthetic sine chords: C→G (in-key fifth) tonnetz_dist=0.95
    vs chroma_dist=0.52; C→F# (out-of-key tritone) tonnetz_dist
    =1.83 vs chroma_dist=0.68. Tonnetz has ~50% BETTER in-key-
    vs-out-of-key discrimination ratio (1.92x vs chroma's
    1.30x). GBM max_depth=3 cannot learn the linear 12→6
    projection itself; providing tonnetz as an explicit feature
    gives the geometric embedding directly. Near window (±2s)
    preserves 32cac36 self-gating success on speech: voiced=
    vowels whose per-vowel tonnetz swings with prosody
    regardless of splice → high-variance noise → GBM assigns
    low per-domain SHAP on english/korean, feature functionally
    invisible; on singing voiced=sustained sung notes whose
    tonnetz IS the key signature → clean discriminator.
    Orthogonal to every prior axis: NOT 634cdd2 (raw chroma no
    mask), NOT 68004ca (chroma with external tonality-ratio
    gate, verify-fail), NOT 9fe41d1 (chroma persistence_far
    POST-FILTER 4-8s), NOT 32cac36 (voiced chroma — pitch-class
    not tonal-centroid; tonnetz is a 6-dim harmonic-function
    projection, geometrically distinct from 12-dim PCP), NOT
    ff65865 (far window), NOT 49bd0b1 (cepstral). Blast radius:
    features.py only — one new block + one cache-key +
    FEATURE_NAMES append + assert bump. Per-t cost: 2 slices +
    voiced-mask mean + cosine on 6-dim vectors, sub-ms.
    Per-chunk cost: one 6×12 matmul on cached chroma (~0.4 ms),
    absorbed within 243/300 s eval budget. Risk-bounded: self-
    gating via voicing mask reproduces 32cac36 safety on speech;
    zero-voiced sentinel prevents NaN.

(c) IF THIS FAILS. (1) If GBM assigns ~0 SHAP (tonnetz-of-
    chroma_stft too correlated with raw chroma for max_depth=3
    to find additional splits), compute tonnetz from chroma_cqt
    instead (logarithmic frequency bin centers give cleaner
    tonal-centroid separation at cost of ~30 ms/chunk). (2) If
    it lifts singing but regresses speech (near-window speech
    tonnetz turns out to carry splice signal too), add an
    HPSS-HARMONIC secondary mask requiring harmonic_energy >
    0.7·(harm+perc) on contributing frames — singing voice
    frames pass, speech voiced-vowel frames often fail. (3)
    Final escalation: replace cosine with jensen-shannon
    divergence on L1-normalized tonnetz PMFs — emphasizes
    single-axis shifts (tonic modulation) over uniform spread.

(d) Information gaps. (i) CLEAN_FP_POSITIONS still absent from
    CURRENT STATE after 15+ consecutive hypotheses. I cannot
    verify whether 32cac36's 2 surviving singing FPs have
    voiced frames in both pre and post ±2s windows (if
    instrumental-only section, the new feature is a no-op for
    them). (ii) SHAP rollup reports "no keeps yet — rollup
    empty" for 49bd0b1 and 32cac36, so I cannot confirm which
    features 32cac36 actually leans on per domain — the
    chord-cycle-back hypothesis is theoretical. (iii) Training-
    chunk voiced_tonnetz distributions per class unknown;
    cannot preview separability pre-retrain.

(e) Wrapper enhancements. (1) CLEAN_FP_POSITIONS JSON block
    — persistent blocker for 15+ hypotheses; per-FP (domain,
    file, t_sec, label_id, p_splice, vp_pre_count, vp_post_count,
    voiced_chroma_dist, voiced_mfcc_dist, top-5 |SHAP| features
    with values) would flip feature design from theory to data.
    (2) `scripts/feature_oof_preview.py --add <feature_fn>`
    that trains once and reports per-domain OOF-F1 delta vs
    current — turns "is this feature worth a 3-min retrain"
    into a numeric. (3) Per-keep SHAP rollup populated
    IMMEDIATELY on keep commit — current "no keeps yet — rollup
    empty" marker means I'm forming hypotheses with zero
    per-domain SHAP data on the classifier I'm trying to
    improve, the biggest information gap in the loop.

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

