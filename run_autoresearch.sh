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

run_loop() {
    cd "$PROJECT_DIR"
    rm -f "$STOP_FILE"
    consecutive_discards=0
    rate_limit_backoff=300  # start at 5 min, double each time, cap at 5 hours

    # Log crashes — if the process dies unexpectedly, record it
    trap 'echo "$(date -Iseconds) CRASH: loop terminated unexpectedly (signal $?)" >> "$LOG_FILE"' EXIT HUP INT TERM

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
        git reset --hard HEAD~1 >> "$LOG_FILE" 2>&1
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

        iteration_output=$(claude -p "You are running autoresearch on the audio splice detection project.

==== METRIC DEFINITION (what 'combined' actually measures) ================
evaluate.py iterates every entry in dataset_registry.DATASETS (currently
singing / korean / english) and for each dataset computes:
    splice_f1   = harmonic_mean(precision, recall) over spliced files
    clean_score = 1 - clean_fp / n_clean_files          (clamped to [0,1])
    dataset_combined = splice_f1 * clean_score
Then the aggregate:
    combined = geometric_mean_with_floor(per_dataset_combined, floor=0.01)
              = exp(mean(log(max(v, 0.01)) for v in per_ds_values))
Because it's a geometric mean: the WEAKEST domain dominates the aggregate.
Bounds: clean_fp_<domain> <= 15 per dataset, total clean_fp <= 45 across
all 3 domains. Exceeding either fails verify_agent's Clean FP bound check.

==== CURRENT STATE (authoritative — from baseline_metrics.json) ===========
combined (aggregate GM): ${current_best}
per-domain:
${per_domain_state}

${iter_summary:+Progress: $iter_summary}

==== ARCHITECTURE & TUNABLE SURFACE =======================================
detect_splices runs a GBM-first dense scan.
  PRIMARY tunables (instant, no retrain) — detector.py:
      GBM_THRESHOLD         P(splice)>thr is an emit; higher = fewer FP
      GBM_MIN_SEP_S         dedupe distance for adjacent emits
      ANALYSIS_STRIDE_S     dense-scan stride (smaller = denser, slower)
  RETRAIN-required tunables (~3 min retrain) — .omc/classifier/train_classifier.py:
      make_pipeline() GradientBoostingClassifier hyperparams
      (n_estimators, max_depth, learning_rate, subsample).
  Feature engineering: features.py (extend FEATURE_NAMES). Requires retrain.
  DO NOT tune ml_config.py — its params drive a legacy path that no longer
  runs. Edits have ZERO effect on combined.
  DO NOT tune the _detect_phase/_detect_crossfade/_detect_cpe/_detect_pairwise
  internal thresholds — they're only called in the DSP-fallback path which
  is not exercised by evaluate.py.

==== HISTORY ==============================================================

RECENT FAILED HYPOTHESES (last 30; per-domain combined in brackets):
${recent_failures:-  (none yet)}

TOP-5 KEEPS (by combined — what has actually worked):
${recent_keeps:-  (none yet)}

For deeper history use the Read tool on these READ-ONLY artifacts:
  - results.tsv                                  full iteration ledger
  - .omc/autoresearch.log                        wrapper/verify narrative
  - .omc/classifier/versions.json                keep-only version history
  - git log --oneline --grep='hypothesis:\|baseline:' | head -40

==== ONE EXPERIMENT ITERATION =============================================
1. Read baseline_metrics.json and any history artifacts you need. Decide
   which domain most needs improvement (check per-domain table above —
   the GM is dragged down by the WEAKEST domain, so target it).
2. Form a hypothesis to improve combined. Prefer PRIMARY tunables
   (instant loop). Touch RETRAIN tunables only when primary feels exhausted.

   CRITICAL: Do NOT repeat a hypothesis from RECENT FAILED HYPOTHESES.
3. Edit detector.py (or features.py / train_classifier.py if retraining)
   with the smallest viable change. If retraining, also run
   train_classifier.py in this step AND stage the refreshed model with:
     git add .omc/classifier/fp_classifier.joblib .omc/classifier/fp_classifier.meta.json
   (without staging the joblib, a later iteration's reset reverts it and
   train_classifier.py's hyperparams to different commits — silent skew.)
4. git add <touched files> ; git commit -m \"hypothesis: <one-line description>\"
5. Run (REDIRECT REQUIRED — the wrapper parses the file, not your narrative):
   uv run python evaluate.py --shap 2>&1 | tee .omc/last_eval.log
6. Parse combined from the LAST 'combined:' line in the output.
7. CURRENT BEST (authoritative): ${current_best}
   If combined > ${current_best} (strictly):   print RESULT:keep-pending
   If combined <= ${current_best}:             git reset --hard HEAD~1 ; print RESULT:discard
8. Do NOT run verify_agent.py yourself — the wrapper handles it.
9. Do NOT write to results.tsv — the wrapper handles it.
10. Do NOT delete or rename .omc/last_eval.log — the wrapper reads it.
    It is in .gitignore and will never enter a commit.

Do NOT loop. Execute exactly ONE iteration and exit." \
            --allowedTools "Bash Edit Read Write Grep Glob" \
            --add-dir "$PROJECT_DIR" \
            2>&1)

        # Check if claude itself failed (rate limit, token exhausted, crash)
        claude_exit=$?

        # Append iteration output to log AFTER claude finishes (immune to git reset)
        echo "$iteration_output" >> "$LOG_FILE"
        echo "$iteration_output" | tail -5

        last_output="$iteration_output"

        if echo "$last_output" | grep -qiE "rate.limit|usage.limit|credit|quota|429|overloaded|capacity"; then
            minutes=$((rate_limit_backoff / 60))
            echo "$(date -Iseconds) API limit detected. Exponential backoff: ${minutes}m..." >> "$LOG_FILE"
            sleep "$rate_limit_backoff"
            # Double backoff, cap at 5 hours (18000s)
            rate_limit_backoff=$((rate_limit_backoff * 2))
            [ "$rate_limit_backoff" -gt 18000 ] && rate_limit_backoff=18000
            continue
        fi

        # Successful claude run — reset backoff
        rate_limit_backoff=300

        if [ $claude_exit -ne 0 ] && ! echo "$last_output" | grep -q "RESULT:"; then
            echo "$(date -Iseconds) Claude failed (exit $claude_exit). Backing off 60 seconds..." >> "$LOG_FILE"
            sleep 60
            continue
        fi

        # Parse result from claude output
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

        case "$last_status" in
            keep-pending)
                # STRUCTURAL VERIFICATION: shell runs verify_agent.py directly (not Claude)
                echo "$(date -Iseconds) Keep pending — running structural verification..." >> "$LOG_FILE"

                # Extract reported combined from Claude's output.
                # Match "combined: 0.X", "combined=0.X", "Combined dropped to 0.X", "combined score of 0.X", etc.
                reported=$(echo "$last_output" | grep -oiE '\bcombined[^0-9\n]{0,30}[0-9]+\.[0-9]+' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "0")

                # Wrapper-side strict-improvement gate. Claude's own keep/discard
                # decision can be wrong when the prompt-provided current_best
                # lags a recent baseline update, so we re-check here.
                strictly_better=$(python3 -c "print(1 if float('$reported') > float('$current_best') + 1e-9 else 0)" 2>/dev/null || echo 0)
                if [ "$strictly_better" != "1" ]; then
                    echo "$(date -Iseconds) NO-OP KEEP REJECTED: reported=$reported not > current_best=$current_best" >> "$LOG_FILE"
                    log_to_results_tsv "verify-fail"
                    git reset --hard HEAD~1 >> "$LOG_FILE" 2>&1
                    consecutive_discards=$((consecutive_discards + 1))
                    # Brief pause then continue to next iteration
                    sleep 2
                    continue
                fi

                # verify_agent.py enforces a 240s timeout on evaluate.py internally.
                # Capture exit code separately — `|| true` would mask LOW confidence failures.
                set +e
                verify_output=$(uv run python .omc/coordination/verify_agent.py \
                    --agent-name autoresearch \
                    --reported-combined "$reported" 2>&1)
                verify_exit=$?
                set -e
                echo "$verify_output" >> "$LOG_FILE"

                if [ $verify_exit -eq 0 ]; then
                    consecutive_discards=0
                    echo "$(date -Iseconds) VERIFIED KEEP (combined=$reported, prev best=$current_best)" >> "$LOG_FILE"
                    log_to_results_tsv "keep"

                    # Version management
                    VERSION=$(($(git tag -l 'detector-v*' 2>/dev/null | wc -l) + 1))
                    git tag "detector-v$VERSION"
                    SNAP="$PROJECT_DIR/.omc/classifier/detector_v${VERSION}.py"
                    cp "$PROJECT_DIR/detector.py" "$SNAP"

                    # Update baseline_metrics.json so the NEXT iteration sees
                    # the new current_best. Written from RESULTS_TSV on disk
                    # (the authoritative source) so per-dataset fields are
                    # captured too — not just the aggregate `combined`.
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

                    # Durability: commit the baseline update as its own
                    # commit so subsequent `git reset --hard HEAD~1` on a
                    # discard reverts only the failed hypothesis, not the
                    # latest keep's baseline. Without this commit step the
                    # wrapper's baseline write lives as an unstaged change
                    # and gets wiped by the next discard's reset.
                    if ! git diff --quiet .omc/coordination/baseline_metrics.json; then
                        git add .omc/coordination/baseline_metrics.json
                        git commit -m "baseline: combined=$reported after keep $(git log -1 --format=%h)" >> "$LOG_FILE" 2>&1
                    fi

                    # Update versions.json
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
                else
                    # Verification failed — log BEFORE reverting (otherwise git HEAD changes)
                    log_to_results_tsv "verify-fail"
                    git reset --hard HEAD~1 >> "$LOG_FILE" 2>&1
                    consecutive_discards=$((consecutive_discards + 1))
                    echo "$(date -Iseconds) VERIFY-FAIL: reverting (discards: $consecutive_discards/$MAX_CONSECUTIVE_DISCARDS)" >> "$LOG_FILE"
                fi
                ;;
            keep)
                # Legacy: if Claude outputs "keep" instead of "keep-pending", still verify
                echo "$(date -Iseconds) Keep (legacy) — running structural verification..." >> "$LOG_FILE"
                reported=$(echo "$last_output" | grep -oE 'combined[: ]+[0-9]+\.[0-9]+' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "0")
                verify_output=$(uv run python .omc/coordination/verify_agent.py \
                    --agent-name autoresearch --reported-combined "$reported" 2>&1)
                verify_exit=$?
                echo "$verify_output" >> "$LOG_FILE"
                if [ $verify_exit -eq 0 ]; then
                    consecutive_discards=0
                    echo "$(date -Iseconds) VERIFIED KEEP" >> "$LOG_FILE"

                    # Version management
                    VERSION=$(($(git tag -l 'detector-v*' 2>/dev/null | wc -l) + 1))
                    git tag "detector-v$VERSION"
                    SNAP="$PROJECT_DIR/.omc/classifier/detector_v${VERSION}.py"
                    cp "$PROJECT_DIR/detector.py" "$SNAP"

                    # Update versions.json
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

                    # Classifier training now happens in-loop via evaluate.py --shap
                else
                    git reset --hard HEAD~1 >> "$LOG_FILE" 2>&1
                    consecutive_discards=$((consecutive_discards + 1))
                    echo "$(date -Iseconds) VERIFY-FAIL (legacy keep)" >> "$LOG_FILE"
                fi
                ;;
            discard)
                consecutive_discards=$((consecutive_discards + 1))
                echo "$(date -Iseconds) Iteration: discard (discards: $consecutive_discards/$MAX_CONSECUTIVE_DISCARDS)" >> "$LOG_FILE"
                # Recover reverted hypothesis from reflog (agent already did git reset)
                discard_commit=$(git rev-parse --short "HEAD@{1}" 2>/dev/null || echo "")
                discard_subject=$(git log -1 --format=%s "HEAD@{1}" 2>/dev/null | sed 's/^hypothesis: //' | tr '\t' ' ' || echo "")
                log_to_results_tsv "discard" "$discard_commit" "$discard_subject"
                ;;
            unknown)
                echo "$(date -Iseconds) Iteration: unknown output (not counting toward circuit breaker)" >> "$LOG_FILE"
                sleep 30
                ;;
        esac

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
            echo "Last: $(tail -1 "$RESULTS" | cut -f9,10)"
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
        echo "Usage: $0 {start|stop|status|rollback <version>}"
        exit 1
        ;;
esac
