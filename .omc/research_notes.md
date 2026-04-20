## Historical digest (oldest 20 entries compacted)
Span: 2026-04-18T14:55:19+09:00 — 4f40e1d (verify-fail, combined=0.463218) … 2026-04-18T18:42:11+09:00 — d47c0e2 (discard, combined=0.010000)
Outcomes: 0 keep / 11 discard / 7 verify-fail; combined range [0.0100, 0.4632].
(Older reflections collapsed to conserve prompt budget.)

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

