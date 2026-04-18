#!/usr/bin/env bash
# Perpetual autoresearch wrapper. Runs Karpathy-style experiment loop in tmux.
# Usage: ./run_autoresearch.sh start|stop|status
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION="autoresearch"
STOP_FILE="$PROJECT_DIR/.omc/autoresearch-stop"
LOG_FILE="$PROJECT_DIR/.omc/autoresearch.log"
RESULTS="$PROJECT_DIR/results.tsv"
MAX_CONSECUTIVE_DISCARDS=50

_eval_cleanup() {
    # Shred (best-effort) and remove the decrypted eval tree. Called on
    # every loop exit path via the EXIT/HUP/INT/TERM trap so plaintext
    # eval audio never outlives the loop process.
    if [ -n "${EVAL_TMP_PARENT:-}" ] && [ -d "$EVAL_TMP_PARENT" ]; then
        rm -rf "$EVAL_TMP_PARENT" 2>/dev/null || true
        echo "$(date -Iseconds) Eval cleanup: removed $EVAL_TMP_PARENT" >> "$LOG_FILE"
    fi
}

# US-509: per-phase iteration timing. Uses bash $SECONDS (O(1), <1ms).
# Phases: claude, retrain, eval, verify, note, total. Values are seconds.
# `-` = phase skipped this iteration. Log lines use distinct prefixes
# (`PHASE` + `ITER_SUMMARY`) so scripts/phase_stats.py can parse cleanly.
_phase_start() {
    eval "_PHASE_$1_T0=$SECONDS"
}
_phase_end() {
    local name="$1"
    local t0_var="_PHASE_${name}_T0"
    local t0="${!t0_var:-$SECONDS}"
    local elapsed=$((SECONDS - t0))
    echo "$(date -Iseconds) PHASE ${name}: ${elapsed}s" >> "$LOG_FILE"
    eval "ITER_${name}_S=$elapsed"
}
_phase_skip() {
    eval "ITER_$1_S=-"
}
_reset_iter_timers() {
    local p
    for p in total claude retrain eval verify note; do
        eval "ITER_${p}_S=-"
    done
}
_iter_summary() {
    local sha="$1"
    local status="$2"
    echo "$(date -Iseconds) ITER_SUMMARY iter=${sha} status=${status}" \
         "total=${ITER_total_S:-?} claude=${ITER_claude_S:-?}" \
         "retrain=${ITER_retrain_S:-?} eval=${ITER_eval_S:-?}" \
         "verify=${ITER_verify_S:-?} note=${ITER_note_S:-?}" \
         >> "$LOG_FILE"
}

# Paths that MUST survive every hypothesis rollback. Wrapper and
# infrastructure — NOT the claude-editable hypothesis surface
# (detector.py / features.py / classifier joblib+meta). Without
# guarding, any in-flight edit to these files is wiped by the next
# `git reset --hard` on a discard. Root cause of losses earlier in
# this session (the portable-timeout fix, features.py cache code).
_GUARD_PATHS=(
    run_autoresearch.sh
    scripts/
    .gitignore
    .omc/prd.json
    progress.txt
    CLAUDE.md
)

_guard_stash_push() {
    if git diff --quiet -- "${_GUARD_PATHS[@]}" 2>/dev/null \
       && git diff --cached --quiet -- "${_GUARD_PATHS[@]}" 2>/dev/null \
       && [ -z "$(git ls-files --others --exclude-standard -- "${_GUARD_PATHS[@]}" 2>/dev/null)" ]; then
        return 1
    fi
    git stash push --include-untracked --quiet \
        -m "autoresearch-guard-$$-$RANDOM" \
        -- "${_GUARD_PATHS[@]}" 2>> "$LOG_FILE" || return 1
    return 0
}

_guard_stash_pop() {
    local top_label
    top_label=$(git stash list -1 2>/dev/null | grep -oE 'autoresearch-guard-[0-9]+-[0-9]+' | head -1)
    [ -z "$top_label" ] && return 0
    if ! git stash pop --quiet 2>>"$LOG_FILE"; then
        echo "$(date -Iseconds) WARN: guard stash pop conflict — stash $top_label preserved; recover with 'git stash list / apply'" >> "$LOG_FILE"
        return 1
    fi
    return 0
}

# Thin wrapper around `git reset --hard`. The ONLY function that should
# call `git reset --hard` inside the autoresearch loop.
_guarded_reset() {
    local target="$1"
    local stashed=0
    _guard_stash_push && stashed=1 || stashed=0
    git reset --hard "$target" >> "$LOG_FILE" 2>&1
    local rc=$?
    [ $stashed -eq 1 ] && _guard_stash_pop
    return $rc
}

# Append an entry to .omc/research_notes.md and commit it as `note:`
# AFTER any keep/discard tree mutation (reset for discard, baseline
# commit for keep). Commit order keeps notes out of the hypothesis
# reset path: note lives on top of either the reset-to state (discard)
# or the baseline commit (keep). Orphan detectors ignore `note:`
# subjects because they only match `hypothesis:`.
_append_note() {
    local status="$1"        # keep | discard | verify-fail
    local short_sha="$2"
    local subject="$3"       # stripped of `hypothesis: ` prefix
    local notes="$PROJECT_DIR/.omc/research_notes.md"
    local refl="$PROJECT_DIR/.omc/last_reflection.md"
    local eval_log="$PROJECT_DIR/.omc/last_eval.log"

    local tsv_line=""
    if [ -f "$eval_log" ]; then
        tsv_line=$(grep -E "^RESULTS_TSV: " "$eval_log" | tail -1)
    fi
    local combined per_domain_line
    combined=$(printf '%s' "$tsv_line" | grep -oE "\\bcombined=[0-9.]+" | head -1 | cut -d= -f2)
    combined="${combined:-NA}"
    per_domain_line=$(printf '%s' "$tsv_line" \
        | grep -oE "\\bcombined_(singing|korean|english)=[0-9.]+" \
        | paste -sd' ' -)
    per_domain_line="${per_domain_line:-(no per-domain data)}"

    local reflection="(no reflection — claude did not write .omc/last_reflection.md this iteration)"
    if [ -f "$refl" ] && [ -s "$refl" ]; then
        reflection=$(cat "$refl")
    fi

    {
        printf '## %s — %s (%s, combined=%s)\n' \
            "$(date -Iseconds)" "$short_sha" "$status" "$combined"
        printf 'subject: %s\n' "$subject"
        printf 'per-domain: %s\n' "$per_domain_line"
        printf '\n%s\n\n' "$reflection"
    } >> "$notes"

    # Consume the reflection so the next iteration's "(no reflection)"
    # fallback actually fires if claude forgets.
    : > "$refl" 2>/dev/null || true

    if ! git diff --quiet "$notes" 2>/dev/null || [ -n "$(git ls-files --others --exclude-standard "$notes")" ]; then
        git add "$notes"
        git commit -m "note: ${status} ${short_sha}" >> "$LOG_FILE" 2>&1 || true
    fi

    # Compact if we've grown past the threshold.
    uv run python "$PROJECT_DIR/scripts/notebook_digest.py" >> "$LOG_FILE" 2>&1 || true
    if ! git diff --quiet "$notes" 2>/dev/null; then
        git add "$notes"
        git commit -m "note: digest compaction" >> "$LOG_FILE" 2>&1 || true
    fi
}

# Flag `hypothesis:` commits in the last N that lack a paired `baseline:`
# in the next 3 commits AND are missing from results.tsv. Warn-only:
# auto-reverting deep history would risk losing human-authored
# maintenance commits layered on top (baseline-restore, refactors, docs).
_detect_deep_orphans() {
    local depth="${1:-20}"
    local results_tsv="$PROJECT_DIR/results.tsv"
    local branch_range="HEAD~${depth}..HEAD"
    git rev-parse "HEAD~${depth}" >/dev/null 2>&1 || branch_range="HEAD"

    # hypothesis: commits older than the earliest TSV row predate the
    # tracker — flagging them is noise, not signal.
    local cutoff_sha=""
    if [ -f "$results_tsv" ]; then
        cutoff_sha=$(awk -F'\t' 'NR>1 && $1 != "" && $1 != "NA" {print $1; exit}' "$results_tsv")
    fi

    local count=0
    while IFS='|' read -r sha subject; do
        case "$subject" in
            hypothesis:*) ;;
            *) continue ;;
        esac
        if [ -n "$cutoff_sha" ] \
           && ! git merge-base --is-ancestor "$cutoff_sha" "$sha" 2>/dev/null; then
            continue
        fi
        if git log --format=%s --reverse "${sha}..HEAD" 2>/dev/null \
             | head -3 | grep -q "^baseline:"; then
            continue
        fi
        if [ -f "$results_tsv" ] \
           && grep -q "^${sha:0:7}	" "$results_tsv" 2>/dev/null; then
            continue
        fi
        local msg="$(date -Iseconds) DEEP-ORPHAN: ${sha:0:7} \"${subject}\" — no paired baseline: in next 3 commits, no row in results.tsv"
        echo "$msg" >&2
        echo "$msg" >> "$LOG_FILE"
        count=$((count + 1))
    done < <(git log --format='%H|%s' "$branch_range" 2>/dev/null)

    if [ $count -gt 0 ]; then
        echo "$(date -Iseconds) DEEP-ORPHAN: found $count suspect commit(s) in last $depth; review with 'git log --oneline -$depth'" >> "$LOG_FILE"
    fi
    return 0
}

run_loop() {
    cd "$PROJECT_DIR"
    rm -f "$STOP_FILE"
    consecutive_discards=0
    rate_limit_backoff=300  # start at 5 min, double each time, cap at 5 hours

    # Combined crash-log + eval-cleanup trap. Shred ordering matters: we
    # cleanup AFTER logging the crash so the cleanup failure (if any)
    # doesn't eat the crash signal.
    trap '_trap_ec=$?; echo "$(date -Iseconds) CRASH: loop terminated (signal $_trap_ec)" >> "$LOG_FILE"; _eval_cleanup' EXIT HUP INT TERM

    echo "$(date -Iseconds) Autoresearch loop started" >> "$LOG_FILE"

    # Orphan-hypothesis detection. If a prior run crashed mid-iteration,
    # HEAD can be an unverified `hypothesis: ...` commit sitting on top of
    # the last verified keep. Its effect is implicitly baked into whatever
    # future iterations build on. Either verify it or reset — we reset
    # because the wrapper can't retroactively re-run verify_agent against
    # a mutated baseline. A human can `git cherry-pick` the orphan back
    # if they want to revisit it.
    orphan_subject=$(git log -1 --format=%s 2>/dev/null)
    if echo "$orphan_subject" | grep -q "^hypothesis:"; then
        echo "$(date -Iseconds) ORPHAN hypothesis detected at HEAD: $(git log -1 --format=%h). Resetting to HEAD~1 for clean baseline." >> "$LOG_FILE"
        _guarded_reset HEAD~1
    fi

    _detect_deep_orphans 20

    # ---- Eval dataset isolation ----------------------------------------
    # The plaintext data/eval/ tree does NOT exist on disk — it's been
    # encrypted into data/eval.tar.gz.enc and shredded (see
    # scripts/eval_crypto.py). Decrypt once per loop into a fresh /tmp
    # dir, export OMC_EVAL_DATA_ROOT for evaluate.py + preflight +
    # verify_agent (dataset_registry resolves eval_path against this at
    # import). The env var is explicitly UNSET in the claude subprocess's
    # env via `env -u` below, so the claude-driven iteration cannot
    # locate the decrypted tree; only this shell and its direct
    # evaluate.py children see it.
    if [ ! -f "$PROJECT_DIR/data/eval.tar.gz.enc" ]; then
        echo "$(date -Iseconds) ERROR: data/eval.tar.gz.enc missing. Run: uv run python scripts/eval_crypto.py setup" >> "$LOG_FILE"
        echo "ERROR: data/eval.tar.gz.enc missing. Run: uv run python scripts/eval_crypto.py setup" >&2
        exit 1
    fi
    _decrypt_out=$(uv run python "$PROJECT_DIR/scripts/eval_crypto.py" decrypt --keep 2>&1)
    _decrypt_rc=$?
    if [ $_decrypt_rc -ne 0 ]; then
        echo "$(date -Iseconds) ERROR: eval decrypt failed (rc=$_decrypt_rc): $_decrypt_out" >> "$LOG_FILE"
        echo "ERROR: eval decrypt failed. See $LOG_FILE" >&2
        exit 1
    fi
    EVAL_TMP_ROOT=$(echo "$_decrypt_out" | tail -1)
    EVAL_TMP_PARENT="$(dirname "$EVAL_TMP_ROOT")"
    export OMC_EVAL_DATA_ROOT="$EVAL_TMP_ROOT"
    echo "$(date -Iseconds) Eval decrypted to $EVAL_TMP_ROOT (OMC_EVAL_DATA_ROOT set for wrapper children)" >> "$LOG_FILE"

    # US-504: prune stale feature_cache subdirs. Each features.py sha
    # gets its own subdir; when the sha changes, the old one becomes
    # dead weight (~1.5 GB per dir on this dataset). Keep only the dir
    # matching the current features.py sha.
    _CUR_FEAT_SHA="$(git hash-object "$PROJECT_DIR/features.py" 2>/dev/null || echo "")"
    if [ -d "$PROJECT_DIR/.omc/feature_cache" ] && [ -n "$_CUR_FEAT_SHA" ]; then
        for d in "$PROJECT_DIR/.omc/feature_cache"/*/; do
            [ -d "$d" ] || continue
            base="$(basename "$d")"
            if [ "$base" != "$_CUR_FEAT_SHA" ]; then
                echo "$(date -Iseconds) Pruning stale feature_cache dir $base (current is $_CUR_FEAT_SHA)" >> "$LOG_FILE"
                rm -rf "$d"
            fi
        done
    fi

    while true; do
        # Stop signal check
        if [ -f "$STOP_FILE" ]; then
            echo "$(date -Iseconds) Stop signal received. Exiting." >> "$LOG_FILE"
            break
        fi

        # Circuit breaker
        if [ "$consecutive_discards" -ge "$MAX_CONSECUTIVE_DISCARDS" ]; then
            echo "$(date -Iseconds) Circuit breaker: $MAX_CONSECUTIVE_DISCARDS consecutive discards. Exiting." >> "$LOG_FILE"
            break
        fi

        # Run one iteration via claude
        echo "$(date -Iseconds) Starting iteration (consecutive discards: $consecutive_discards)" >> "$LOG_FILE"
        _reset_iter_timers
        _phase_start total

        # ---- Build prompt context from durable artifacts ---------------
        # results.tsv columns (post-parser-fix):
        #   1 commit          8 combined_korean
        #   2 combined        9 combined_english
        #   3 combined_mean  10 clean_fp_singing
        #   4 combined_min   11 clean_fp_korean
        #   5 clean_fp       12 clean_fp_english
        #   6 n_datasets     13 status
        #   7 combined_singing  14 description
        recent_failures=""
        recent_keeps=""
        iter_summary=""
        if [ -f "$RESULTS" ]; then
            # Last 30 discards/verify-fails WITH per-dataset breakdown so
            # claude sees which domain the failure came from, not just the
            # aggregate. NA rows (parse failures) are dropped.
            recent_failures=$(awk -F'\t' '
                NR==1 {next}
                ($13 == "discard" || $13 == "verify-fail") && $2 != "NA" {
                    printf "  - combined=%s [singing %s / korean %s / english %s]: %s\n", \
                           $2, $7, $8, $9, $14
                }' "$RESULTS" | tail -30)

            # Top 5 keeps by combined — lets claude see what has worked,
            # not only what has failed.
            recent_keeps=$(awk -F'\t' '
                NR==1 {next}
                $13 == "keep" && $2 != "NA" {
                    printf "%s\t  + combined=%s [singing %s / korean %s / english %s]: %s\n", \
                           $2, $2, $7, $8, $9, $14
                }' "$RESULTS" | sort -rn -k1,1 -t$'\t' | cut -f2- | head -5)

            iter_summary=$(awk -F'\t' '
                NR==1 {next}
                { total++
                  if ($13 == "keep") keeps++
                  else if ($13 == "discard") discards++
                  else if ($13 == "verify-fail") vfails++ }
                END {
                  printf "%d iterations (keeps=%d, discards=%d, verify-fail=%d)", \
                         total+0, keeps+0, discards+0, vfails+0
                }' "$RESULTS")
        fi

        # Authoritative current state from baseline_metrics.json: the
        # aggregate GM plus each domain's individual combined so claude
        # can spot the weakest domain to target.
        current_best=$(python3 -c "import json; print(json.load(open('.omc/coordination/baseline_metrics.json')).get('combined', 0))" 2>/dev/null || echo "0")
        per_domain_state=$(python3 -c "
import json
d = json.load(open('.omc/coordination/baseline_metrics.json'))
per = d.get('per_dataset_combined', {})
fp = d.get('per_dataset_clean_fp', {})
if not per:
    print('  (per-dataset breakdown unavailable — baseline_metrics.json has not captured it yet)')
else:
    for ds in sorted(per):
        print(f'    {ds:<8}  combined={per[ds]:.6f}  clean_fp={fp.get(ds, \"?\")}')
" 2>/dev/null)

        # SHAP rollup (US-500): biases claude's feature-engineering
        # hypotheses toward slots actually driving keeps, not guesses.
        uv run python "$PROJECT_DIR/scripts/shap_rollup.py" --keeps 5 \
            >/dev/null 2>>"$LOG_FILE" || true

        # Per-tunable exploration frontier (US-506).
        uv run python "$PROJECT_DIR/scripts/tunable_frontier.py" \
            --output "$PROJECT_DIR/.omc/tunable_frontier.txt" \
            >/dev/null 2>>"$LOG_FILE" || true
        tunable_frontier_block=""
        if [ -f "$PROJECT_DIR/.omc/tunable_frontier.txt" ]; then
            tunable_frontier_block=$(cat "$PROJECT_DIR/.omc/tunable_frontier.txt")
        fi

        # Research notes tail (US-503): inject last 10 entries verbatim.
        research_notes_block=""
        if [ -f "$PROJECT_DIR/.omc/research_notes.md" ]; then
            research_notes_block=$(awk '
                /^## / { cnt++; starts[cnt]=NR }
                { buf[NR]=$0 }
                END {
                    start = (cnt > 10) ? starts[cnt-9] : 1
                    for (i = start; i <= NR; i++) print buf[i]
                }
            ' "$PROJECT_DIR/.omc/research_notes.md")
        fi
        if [ -z "$research_notes_block" ]; then
            research_notes_block="  (research notebook empty — this is the first iteration with notes)"
        fi
        shap_rollup_block=$(python3 -c "
import json, os
p = os.path.join('$PROJECT_DIR', '.omc/shap_rollup.json')
try:
    d = json.load(open(p))
except Exception:
    print('  (shap rollup unavailable)')
    raise SystemExit(0)
per = d.get('per_domain', {})
keeps = d.get('keeps_considered', [])
if not per:
    print('  (no keeps yet — rollup empty)')
    raise SystemExit(0)
print(f\"  rolled up from {len(keeps)} keep(s): {', '.join(keeps) if keeps else '(none)'}\")
for dom in sorted(per):
    rows = per[dom][:6]
    names = ', '.join(f\"{r['name']}({r['sum_abs_shap']:.1f})\" for r in rows)
    print(f'    {dom:<8}  {names}')
" 2>/dev/null)

        head_before=$(git rev-parse HEAD)

        # Inverted flow: claude forms a hypothesis, edits code, commits,
        # and exits. The wrapper — NOT claude — runs evaluate.py. This
        # keeps the eval corpus (decrypted under $OMC_EVAL_DATA_ROOT)
        # out of the claude subprocess's reach. `env -u` strips the env
        # var before exec'ing claude so dataset_registry in the claude
        # subprocess resolves the default (missing) data/eval/ paths.
        _phase_start claude
        iteration_output=$(env -u OMC_EVAL_DATA_ROOT -u OMC_FEATURE_CACHE_DIR -u OMC_FEATURES_PY_SHA claude -p "You are forming ONE hypothesis for the audio splice detection project.

==== METRIC DEFINITION (what 'combined' measures) =========================
evaluate.py iterates dataset_registry.DATASETS (singing / korean / english)
and for each dataset computes:
    splice_f1   = harmonic_mean(precision, recall) over spliced files
    clean_score = 1 - clean_fp / n_clean_files          (clamped to [0,1])
    dataset_combined = splice_f1 * clean_score
    combined    = geometric_mean_with_floor(per_ds, floor=0.01)
The GEOMETRIC MEAN is dominated by the WEAKEST domain. Bounds:
clean_fp_<domain> <= 15 per dataset, total clean_fp <= 45.

==== CURRENT STATE (authoritative — baseline_metrics.json) ================
combined (aggregate GM): ${current_best}
per-domain:
${per_domain_state}

TOP PREDICTIVE FEATURES (rolling sum_|shap| over last 5 keeps, per domain):
${shap_rollup_block}

PER-TUNABLE EXPLORATION FRONTIER (what's been tried on each axis):
${tunable_frontier_block}

RESEARCH NOTES (last 10 entries, chronological, latest at bottom — your own prior reflections):
${research_notes_block}

${iter_summary:+Progress: $iter_summary}

==== ARCHITECTURE & TUNABLE SURFACE =======================================
detect_splices runs a GBM-first dense scan.
  PRIMARY (instant) — detector.py:
      GBM_THRESHOLD         P(splice)>thr is an emit (higher = fewer FP)
      GBM_MIN_SEP_S         dedupe distance for adjacent emits
      ANALYSIS_STRIDE_S     dense-scan stride (smaller = denser, slower)
  RETRAIN (~3 min) — .omc/classifier/train_classifier.py:
      GradientBoostingClassifier hyperparams (n_estimators, max_depth,
      learning_rate, subsample).
  Feature engineering: features.py (extend FEATURE_NAMES). Requires retrain.
  DO NOT tune ml_config.py — legacy path, zero effect on combined.
  DO NOT tune _detect_phase/_detect_crossfade/_detect_cpe/_detect_pairwise
  internal thresholds — DSP fallback path, not exercised by evaluate.py.

==== HISTORY ==============================================================
RECENT FAILED HYPOTHESES (last 30; per-domain combined in brackets):
${recent_failures:-  (none yet)}

TOP-5 KEEPS (by combined):
${recent_keeps:-  (none yet)}

Read-only artifacts for deeper context:
  - results.tsv                               full iteration ledger
  - .omc/autoresearch.log                     wrapper/verify narrative
  - .omc/classifier/versions.json             keep-only version history
  - git log --oneline --grep='hypothesis:\|baseline:' | head -40

==== ACCESS RULES =========================================================
- The plaintext eval corpus (data/eval/) is NOT on disk and is NOT
  accessible to you. Do not attempt to read it or infer its location.
  The wrapper will run evaluate.py against an encrypted-then-decrypted
  copy after you commit.
- Do NOT run evaluate.py — the wrapper does this and parses the result.
- Do NOT write to results.tsv or .omc/coordination/baseline_metrics.json
  — the wrapper owns both.
- Do NOT edit evaluate.py, .omc/coordination/manifest.json,
  .omc/coordination/preflight.py — protected.

==== ONE ITERATION ========================================================
0. Read the RESEARCH NOTES above — your prior reflections about what you
   tried, why, and what to try next. If 5+ recent entries all failed on
   the same tunable axis, seriously consider a structural change
   (features.py / train_classifier.py) instead of another tweak.
1. Write 5-8 lines to .omc/last_reflection.md (the wrapper will capture
   this into the permanent research notebook regardless of keep/discard):
     (a) what hypothesis you're about to try
     (b) WHY this direction over the recent failures
     (c) what you'd try next if this one fails
     (d) In your perspective, is there any information that is NOT given
         or is conflicting in this prompt that is degrading your
         performance? Be specific — name the gap or the contradiction.
         If nothing is missing, write \"(no gaps noted)\".
     (e) From YOUR perspective as the claude agent performing autoresearch:
         what design improvements / tool enhancements / feature additions
         to the wrapper, prompt, or available scripts would accelerate
         your research? Be concrete (a function signature, a new prompt
         block, a new wrapper subcommand). If nothing comes to mind,
         write \"(no enhancements noted)\". The human operator reads these
         periodically to decide what to build next.
2. Read baseline_metrics.json and any history you need. Target the
   WEAKEST domain (GM is dragged down by it).
3. Form a hypothesis. Prefer PRIMARY tunables (instant). Touch RETRAIN
   tunables only when primary feels exhausted.
   CRITICAL: Do NOT repeat a hypothesis from RECENT FAILED HYPOTHESES.
4. Edit detector.py / features.py / train_classifier.py with the smallest
   viable change. The wrapper auto-retrains when features.py changes —
   you do NOT need to manually run train_classifier.py (US-505).
5. git add <touched files> ; git commit -m \"hypothesis: <one-line description>\"
6. Print RESULT:ready and exit. The wrapper will run evaluate.py, compare
   to current_best, and decide keep vs discard.
   If you made NO change (unusual — only when every plausible hypothesis
   is ruled out by recent failures), print RESULT:skip and exit without
   committing.

Do NOT loop. Execute exactly ONE iteration and exit." \
            --allowedTools "Bash Edit Read Write Grep Glob" \
            --add-dir "$PROJECT_DIR" \
            2>&1)

        # Check if claude itself failed (rate limit, token exhausted, crash)
        claude_exit=$?
        _phase_end claude

        # Append iteration output to log AFTER claude finishes (immune to git reset)
        echo "$iteration_output" >> "$LOG_FILE"
        echo "$iteration_output" | tail -5

        last_output="$iteration_output"

        if echo "$last_output" | grep -qiE "rate.limit|usage.limit|credit|quota|429|overloaded|capacity"; then
            minutes=$((rate_limit_backoff / 60))
            echo "$(date -Iseconds) API limit detected. Exponential backoff: ${minutes}m..." >> "$LOG_FILE"
            sleep "$rate_limit_backoff"
            rate_limit_backoff=$((rate_limit_backoff * 2))
            [ "$rate_limit_backoff" -gt 18000 ] && rate_limit_backoff=18000
            continue
        fi

        # Successful claude run — reset backoff
        rate_limit_backoff=300

        if [ $claude_exit -ne 0 ] && ! echo "$last_output" | grep -q "RESULT:"; then
            echo "$(date -Iseconds) Claude failed (exit $claude_exit). Backing off 60s." >> "$LOG_FILE"
            sleep 60
            continue
        fi

        last_status=$(echo "$last_output" | grep -o 'RESULT:[a-z-]*' | tail -1 | cut -d: -f2 || echo "unknown")

        # Auto-log every iteration to results.tsv. The contract: every metric
        # column is either the value extracted from `.omc/last_eval.log`'s
        # single `RESULTS_TSV:` line, or the literal string "NA" when that
        # line is missing / malformed. We NEVER use 0 as a parse-failure
        # sentinel because 0 is also a legitimate combined value (e.g. when
        # a hypothesis pushes clean_fp past the bound and combined clamps to
        # 0). Conflating "couldn't parse" with "real zero" is how the prior
        # wrapper silently corrupted the TSV for weeks.
        _tsv_field() {
            # Usage: _tsv_field <line> <key>  ->  stdout = value or "NA"
            local line="$1"
            local key="$2"
            local val
            val=$(printf '%s' "$line" | grep -oE "${key}=[^ ]+" | head -1 | cut -d= -f2)
            if [ -z "$val" ]; then
                echo "NA"
            else
                echo "$val"
            fi
        }

        log_to_results_tsv() {
            local status="$1"
            local commit="${2:-$(git log -1 --format=%h 2>/dev/null)}"
            local subject="${3:-$(git log -1 --format=%s 2>/dev/null | sed 's/^hypothesis: //' | tr '\t' ' ')}"
            local eval_log="$PROJECT_DIR/.omc/last_eval.log"
            local tsv_line=""
            if [ -f "$eval_log" ]; then
                # Take the LAST RESULTS_TSV line — defensive against multiple
                # eval runs within one iteration.
                tsv_line=$(grep -E "^RESULTS_TSV: " "$eval_log" | tail -1)
            fi

            if [ -z "$tsv_line" ]; then
                echo "$(date -Iseconds) WARN: no RESULTS_TSV in $eval_log; logging NA row" >> "$LOG_FILE"
            fi

            local combined=$(_tsv_field "$tsv_line" combined)
            local combined_mean=$(_tsv_field "$tsv_line" combined_mean)
            local combined_min=$(_tsv_field "$tsv_line" combined_min)
            local clean_fp=$(_tsv_field "$tsv_line" clean_fp)
            local n_datasets=$(_tsv_field "$tsv_line" n_datasets)
            local combined_singing=$(_tsv_field "$tsv_line" combined_singing)
            local combined_korean=$(_tsv_field "$tsv_line" combined_korean)
            local combined_english=$(_tsv_field "$tsv_line" combined_english)
            local clean_fp_singing=$(_tsv_field "$tsv_line" clean_fp_singing)
            local clean_fp_korean=$(_tsv_field "$tsv_line" clean_fp_korean)
            local clean_fp_english=$(_tsv_field "$tsv_line" clean_fp_english)

            if [ ! -s "$RESULTS" ]; then
                printf 'commit\tcombined\tcombined_mean\tcombined_min\tclean_fp\tn_datasets\tcombined_singing\tcombined_korean\tcombined_english\tclean_fp_singing\tclean_fp_korean\tclean_fp_english\tstatus\tdescription\n' > "$RESULTS"
            fi

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$commit" "$combined" "$combined_mean" "$combined_min" "$clean_fp" "$n_datasets" \
                "$combined_singing" "$combined_korean" "$combined_english" \
                "$clean_fp_singing" "$clean_fp_korean" "$clean_fp_english" \
                "$status" "$subject" \
                >> "$RESULTS"
        }

        head_after=$(git rev-parse HEAD)

        if [ "$last_status" = "skip" ]; then
            echo "$(date -Iseconds) Claude reported RESULT:skip — no commit. Moving on." >> "$LOG_FILE"
            sleep 5
            continue
        fi

        if [ "$head_before" = "$head_after" ]; then
            echo "$(date -Iseconds) Claude made no commit this iteration (status=$last_status). Treating as skip." >> "$LOG_FILE"
            sleep 5
            continue
        fi

        # Short-form SHA for results.tsv consistency; the column mixes
        # 7-char (from legacy %h path) and 40-char (raw rev-parse) if we
        # don't normalize here.
        hypothesis_commit=$(git log -1 --format=%h "$head_after")
        hypothesis_subject=$(git log -1 --format=%s "$head_after" | sed 's/^hypothesis: //' | tr '\t' ' ')

        # US-505: classifier staleness gate. If features.py has drifted
        # since the classifier was last trained, auto-retrain. Removes a
        # silent-skew bug class where claude edits features.py but
        # forgets to retrain; evaluate.py would then use a classifier
        # whose feature dims or semantics mismatch the detector.
        _features_sha_now=$(git hash-object "$PROJECT_DIR/features.py" 2>/dev/null || echo "")
        _features_sha_trained=$(python3 -c "
import json, sys
try:
    d = json.load(open('.omc/classifier/fp_classifier.meta.json'))
    print(d.get('features_py_sha') or '')
except Exception:
    print('')
" 2>/dev/null)
        if [ -n "$_features_sha_now" ] && [ "$_features_sha_now" != "$_features_sha_trained" ]; then
            echo "$(date -Iseconds) AUTO-RETRAIN: features.py drift (was $_features_sha_trained, now $_features_sha_now)" >> "$LOG_FILE"
            # Portable 300s timeout: GNU `timeout` or brew's `gtimeout`
            # when present, else unguarded. macOS ships neither by default.
            if command -v timeout >/dev/null 2>&1; then
                _retrain_cmd=(timeout 300 uv run python .omc/classifier/train_classifier.py)
            elif command -v gtimeout >/dev/null 2>&1; then
                _retrain_cmd=(gtimeout 300 uv run python .omc/classifier/train_classifier.py)
            else
                _retrain_cmd=(uv run python .omc/classifier/train_classifier.py)
            fi
            _phase_start retrain
            set +e
            "${_retrain_cmd[@]}" >> "$LOG_FILE" 2>&1
            _retrain_rc=$?
            set -e
            _phase_end retrain
            if [ $_retrain_rc -ne 0 ]; then
                echo "$(date -Iseconds) AUTO-RETRAIN FAILED (rc=$_retrain_rc). Treating as verify-fail." >> "$LOG_FILE"
                uv run python .omc/coordination/verify_agent.py --diagnose >> "$LOG_FILE" 2>&1 || true
                log_to_results_tsv "verify-fail" "$hypothesis_commit" "$hypothesis_subject"
                _guarded_reset "$head_before"
                _phase_start note; _append_note "verify-fail" "$hypothesis_commit" "$hypothesis_subject"; _phase_end note
                _phase_end total
                _iter_summary "$hypothesis_commit" "verify-fail"
                consecutive_discards=$((consecutive_discards + 1))
                continue
            fi
            # Stage the refreshed model so a later discard's reset
            # doesn't revert the joblib to a features-mismatched version.
            # AMEND-SAFETY: only amend if HEAD is still the hypothesis
            # commit. If the user (or anything else) landed a commit on
            # top of the hypothesis in the meantime, amending would fold
            # claude's retrain artifacts into that user commit — which a
            # later discard reset would then wipe, orphaning the user's
            # work. Detected bug: my own wrapper-prompt commit got
            # amended into a hypothesis and then lost on discard (reflog
            # 2026-04-18). When this guard fires, commit the joblib+meta
            # as a separate commit instead.
            git add .omc/classifier/fp_classifier.joblib \
                    .omc/classifier/fp_classifier.meta.json >> "$LOG_FILE" 2>&1 || true
            _head_subj=$(git log -1 --format=%s 2>/dev/null)
            case "$_head_subj" in
                hypothesis:*)
                    git commit --amend --no-edit >> "$LOG_FILE" 2>&1 || true
                    ;;
                *)
                    echo "$(date -Iseconds) WARN: HEAD is not a hypothesis commit ('$_head_subj'). Not amending; committing retrain artifacts as separate commit." >> "$LOG_FILE"
                    git commit -m "retrain: auto-refresh classifier for features.py sha $_features_sha_now" >> "$LOG_FILE" 2>&1 || true
                    ;;
            esac
            # Refresh hypothesis_commit since amend changed the SHA.
            head_after=$(git rev-parse HEAD)
            hypothesis_commit=$(git log -1 --format=%h "$head_after")
        else
            _phase_skip retrain
        fi

        # Wrapper runs evaluate.py — claude never sees the decrypted
        # eval tree. OMC_EVAL_DATA_ROOT is already exported in this
        # shell, so evaluate.py + its subprocess children (preflight,
        # verify_agent) inherit it.
        echo "$(date -Iseconds) Running evaluate.py on hypothesis $(git log -1 --format=%h "$hypothesis_commit")" >> "$LOG_FILE"
        # US-504: feature cache env vars. Invalidated automatically when
        # features.py sha changes (new sha → new cache subdir).
        export OMC_FEATURE_CACHE_DIR="$PROJECT_DIR/.omc/feature_cache"
        export OMC_FEATURES_PY_SHA="$(git hash-object "$PROJECT_DIR/features.py" 2>/dev/null || echo unknown)"
        set +e
        _phase_start eval
        uv run python evaluate.py --shap > "$PROJECT_DIR/.omc/last_eval.log" 2>&1
        eval_exit=$?
        set -e
        _phase_end eval

        if [ $eval_exit -ne 0 ]; then
            echo "$(date -Iseconds) evaluate.py exited $eval_exit. Reverting hypothesis (last 20 lines of eval log):" >> "$LOG_FILE"
            tail -20 "$PROJECT_DIR/.omc/last_eval.log" >> "$LOG_FILE" 2>&1 || true
            uv run python .omc/coordination/verify_agent.py --diagnose >> "$LOG_FILE" 2>&1 || true
            log_to_results_tsv "verify-fail" "$hypothesis_commit" "$hypothesis_subject"
            _guarded_reset "$head_before"
            _phase_start note; _append_note "verify-fail" "$hypothesis_commit" "$hypothesis_subject"; _phase_end note
            _phase_skip verify
            _phase_end total
            _iter_summary "$hypothesis_commit" "verify-fail"
            consecutive_discards=$((consecutive_discards + 1))
            continue
        fi

        # Parse combined from the deterministic RESULTS_TSV line.
        reported=$(grep -E "^RESULTS_TSV: " "$PROJECT_DIR/.omc/last_eval.log" | tail -1 | grep -oE "\bcombined=[0-9.]+" | head -1 | cut -d= -f2)
        if [ -z "$reported" ]; then
            echo "$(date -Iseconds) Could not parse combined from RESULTS_TSV. Reverting." >> "$LOG_FILE"
            uv run python .omc/coordination/verify_agent.py --diagnose >> "$LOG_FILE" 2>&1 || true
            log_to_results_tsv "verify-fail" "$hypothesis_commit" "$hypothesis_subject"
            _guarded_reset "$head_before"
            _phase_start note; _append_note "verify-fail" "$hypothesis_commit" "$hypothesis_subject"; _phase_end note
            _phase_skip verify
            _phase_end total
            _iter_summary "$hypothesis_commit" "verify-fail"
            consecutive_discards=$((consecutive_discards + 1))
            continue
        fi

        strictly_better=$(python3 -c "print(1 if float('$reported') > float('$current_best') + 1e-9 else 0)" 2>/dev/null || echo 0)

        if [ "$strictly_better" != "1" ]; then
            echo "$(date -Iseconds) DISCARD: combined=$reported not > current_best=$current_best" >> "$LOG_FILE"
            log_to_results_tsv "discard" "$hypothesis_commit" "$hypothesis_subject"
            _guarded_reset "$head_before"
            _phase_start note; _append_note "discard" "$hypothesis_commit" "$hypothesis_subject"; _phase_end note
            _phase_skip verify
            _phase_end total
            _iter_summary "$hypothesis_commit" "discard"
            consecutive_discards=$((consecutive_discards + 1))
            continue
        fi

        # Strict improvement — run verify_agent for structural checks.
        echo "$(date -Iseconds) Strict improvement (combined=$reported > $current_best). Running verify_agent..." >> "$LOG_FILE"
        _phase_start verify
        set +e
        verify_output=$(uv run python .omc/coordination/verify_agent.py \
            --agent-name autoresearch \
            --reported-combined "$reported" 2>&1)
        verify_exit=$?
        set -e
        _phase_end verify
        echo "$verify_output" >> "$LOG_FILE"

        if [ $verify_exit -ne 0 ]; then
            log_to_results_tsv "verify-fail" "$hypothesis_commit" "$hypothesis_subject"
            _guarded_reset "$head_before"
            _phase_start note; _append_note "verify-fail" "$hypothesis_commit" "$hypothesis_subject"; _phase_end note
            _phase_end total
            _iter_summary "$hypothesis_commit" "verify-fail"
            consecutive_discards=$((consecutive_discards + 1))
            echo "$(date -Iseconds) VERIFY-FAIL: reverting (discards: $consecutive_discards/$MAX_CONSECUTIVE_DISCARDS)" >> "$LOG_FILE"
            continue
        fi

        # VERIFIED KEEP.
        consecutive_discards=0
        echo "$(date -Iseconds) VERIFIED KEEP (combined=$reported, prev best=$current_best)" >> "$LOG_FILE"
        log_to_results_tsv "keep" "$hypothesis_commit" "$hypothesis_subject"

        VERSION=$(($(git tag -l 'detector-v*' 2>/dev/null | wc -l) + 1))
        git tag "detector-v$VERSION"
        SNAP="$PROJECT_DIR/.omc/classifier/detector_v${VERSION}.py"
        cp "$PROJECT_DIR/detector.py" "$SNAP"

        # Update baseline_metrics.json from the RESULTS_TSV line on disk.
        python3 - "$PROJECT_DIR" <<'PYEOF'
import json, os, subprocess, sys, re, datetime
proj = sys.argv[1]
bf = os.path.join(proj, '.omc/coordination/baseline_metrics.json')
eval_log = os.path.join(proj, '.omc/last_eval.log')
try:
    data = json.load(open(bf))
except Exception:
    data = {}

tsv = ""
try:
    with open(eval_log) as f:
        for line in f:
            if line.startswith("RESULTS_TSV:"):
                tsv = line.strip()
except FileNotFoundError:
    pass

def pull(key: str):
    m = re.search(rf"\b{re.escape(key)}=([^\s]+)", tsv)
    return m.group(1) if m else None

combined = pull("combined")
if combined is not None:
    data["combined"] = float(combined)
for k in ("combined_mean", "combined_min"):
    v = pull(k)
    if v is not None:
        data[k] = float(v)

per_ds = {}
for k in list(re.findall(r"\bcombined_([A-Za-z_]+)=", tsv)):
    if k in ("mean", "min"): continue
    v = pull(f"combined_{k}")
    if v is not None:
        per_ds[k] = float(v)
if per_ds:
    data["per_dataset_combined"] = per_ds

per_ds_fp = {}
for k in list(re.findall(r"\bclean_fp_([A-Za-z_]+)=", tsv)):
    v = pull(f"clean_fp_{k}")
    if v is not None:
        per_ds_fp[k] = int(v)
if per_ds_fp:
    data["per_dataset_clean_fp"] = per_ds_fp
total_fp = pull("clean_fp")
if total_fp is not None:
    data["clean_fp_total"] = int(total_fp)

data["git_sha"] = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"], cwd=proj, text=True
).strip()
data["timestamp"] = datetime.datetime.now(datetime.timezone.utc).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)
json.dump(data, open(bf, "w"), indent=2)
PYEOF

        if ! git diff --quiet .omc/coordination/baseline_metrics.json; then
            git add .omc/coordination/baseline_metrics.json
            git commit -m "baseline: combined=$reported after keep $(git log -1 --format=%h "$hypothesis_commit")" >> "$LOG_FILE" 2>&1
        fi

        python3 -c "
import json, subprocess, datetime, os
vf = os.path.join('$PROJECT_DIR', '.omc/classifier/versions.json')
try:
    data = json.load(open(vf))
except:
    data = {'versions': [], 'latest': 0, 'production': 0}
sha = subprocess.run(['git','rev-parse','--short','HEAD'], capture_output=True, text=True).stdout.strip()
data['versions'].append({
    'version': $VERSION, 'git_tag': 'detector-v$VERSION', 'git_sha': sha,
    'detector_snapshot': 'detector_v${VERSION}.py',
    'classifier': 'classifier_v${VERSION}.joblib',
    'combined_dsp': float('$reported'), 'combined_full': None,
    'timestamp': datetime.datetime.now().isoformat()
})
data['latest'] = $VERSION
json.dump(data, open(vf, 'w'), indent=2)
"

        # SHAP shift sentinel (US-507): classify the new keep as
        # STRUCTURAL or LOCAL so the next iteration's research note
        # picks up the verdict in its reflection preamble.
        shap_shift_line=$(uv run python "$PROJECT_DIR/scripts/shap_shift.py" 2>/dev/null | head -1)
        if [ -n "$shap_shift_line" ]; then
            echo "$(date -Iseconds) $shap_shift_line" >> "$LOG_FILE"
            echo "[auto] $shap_shift_line" >> "$PROJECT_DIR/.omc/last_reflection.md"
        fi

        _phase_start note; _append_note "keep" "$hypothesis_commit" "$hypothesis_subject"; _phase_end note
        _phase_end total
        _iter_summary "$hypothesis_commit" "keep"

        # Brief pause between iterations
        sleep 2
    done

    trap - EXIT HUP INT TERM  # clear trap on clean exit
    echo "$(date -Iseconds) Autoresearch loop ended" >> "$LOG_FILE"
}

run_loop_with_restart() {
    # Auto-restart on crash unless stop signal exists
    while true; do
        run_loop
        if [ -f "$STOP_FILE" ]; then
            echo "$(date -Iseconds) Clean shutdown (stop signal)." >> "$LOG_FILE"
            break
        fi
        echo "$(date -Iseconds) Loop exited unexpectedly. Restarting in 60s..." >> "$LOG_FILE"
        sleep 60
    done
}

case "${1:-help}" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "Autoresearch already running. Use 'stop' or 'status'."
            exit 1
        fi
        rm -f "$STOP_FILE"
        tmux new-session -d -s "$SESSION" "bash $0 _loop_restart"
        echo "Autoresearch started in tmux session '$SESSION'."
        echo "  View:   tmux attach -t $SESSION"
        echo "  Stop:   $0 stop"
        echo "  Status: $0 status"
        ;;
    stop)
        touch "$STOP_FILE"
        echo "Stop signal sent. Loop will exit after current iteration."
        ;;
    status)
        # Fast — no subprocesses, no evaluate.py, just file reads
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "🟢 RUNNING"
        else
            echo "🔴 STOPPED"
            [ -f "$STOP_FILE" ] && echo "⏸  Stop signal pending (will be cleared on next start)"
            # Warn if loop stopped after recent code changes
            last_loop_end=$(grep "Autoresearch loop ended" "$LOG_FILE" 2>/dev/null | tail -1 | cut -dT -f1-2 | head -c19)
            last_commit=$(git log -1 --format=%ci -- detector.py evaluate.py ml_eval.py run_autoresearch.sh 2>/dev/null | head -c19)
            if [ -n "$last_loop_end" ] && [ -n "$last_commit" ] && [[ "$last_commit" > "$last_loop_end" ]]; then
                echo "⚠️  Code changed after loop stopped — run '$0 start' to pick up changes"
            fi
        fi
        if [ -f "$RESULTS" ]; then
            total=$(($(wc -l < "$RESULTS") - 1))
            keeps=$(grep -c $'	keep\t' "$RESULTS" 2>/dev/null || true)
            discards=$(grep -c $'	discard\t' "$RESULTS" 2>/dev/null || true)
            vfails=$(grep -c "verify-fail" "$RESULTS" 2>/dev/null || true)
            keeps=${keeps:-0}; discards=${discards:-0}; vfails=${vfails:-0}
            echo "Experiments: $total (keep: $keeps, discard: $discards, verify-fail: $vfails)"
            # cols 13=status, 14=description in the post-parser-fix schema.
            echo "Last: $(tail -1 "$RESULTS" | cut -f13,14)"
        fi
        if [ -f "$PROJECT_DIR/.omc/classifier/versions.json" ]; then
            latest=$(python3 -c "import json; print(json.load(open('$PROJECT_DIR/.omc/classifier/versions.json'))['latest'])" 2>/dev/null || echo "?")
            echo "Version: detector-v$latest"
        fi
        if [ -f "$LOG_FILE" ]; then
            last_log=$(tail -1 "$LOG_FILE")
            echo "Log: $last_log"
            # Show backoff state if rate limited
            echo "$last_log" | grep -qi "backoff" && echo "⚠️  Rate limited — backing off"
        fi
        ;;
    dashboard)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            state="RUNNING"
        elif [ -f "$STOP_FILE" ]; then
            state="STOPPED (stop signal pending)"
        else
            state="STOPPED"
        fi
        python3 "$PROJECT_DIR/scripts/dashboard.py" --state "$state"
        ;;
    rollback)
        N="${2:?Usage: $0 rollback <version>}"
        SNAP="$PROJECT_DIR/.omc/classifier/detector_v${N}.py"
        CLAS="$PROJECT_DIR/.omc/classifier/classifier_v${N}.joblib"
        if [ ! -f "$SNAP" ]; then echo "Detector v$N not found"; exit 1; fi
        cp "$SNAP" "$PROJECT_DIR/detector.py"
        if [ -f "$CLAS" ]; then
            cp "$CLAS" "$PROJECT_DIR/.omc/classifier/fp_classifier.joblib"
            echo "Restored detector v$N + classifier v$N"
        else
            echo "Restored detector v$N (no classifier for this version)"
        fi
        ;;
    _loop)
        run_loop
        ;;
    _loop_restart)
        run_loop_with_restart
        ;;
    *)
        echo "Usage: $0 {start|stop|status|dashboard|rollback <version>}"
        exit 1
        ;;
esac
