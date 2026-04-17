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

        # Build "recent failed hypotheses" summary from results.tsv last 30 discards.
        # Injected into prompt so agent doesn't repeat experiments that already failed.
        recent_failures=""
        if [ -f "$RESULTS" ]; then
            recent_failures=$(awk -F'\t' '$9 == "discard" || $9 == "verify-fail" {printf "  - combined=%s: %s\n", $2, $10}' "$RESULTS" | tail -30)
        fi

        # Current best = baseline_metrics.json.combined. Use this (not the
        # possibly-stale tail of results.tsv) as the keep/discard threshold.
        current_best=$(python3 -c "import json; print(json.load(open('.omc/coordination/baseline_metrics.json'))['combined'])" 2>/dev/null || echo "0")

        iteration_output=$(claude -p "You are running autoresearch on the audio splice detection project.

ARCHITECTURE NOTE (2026-04-18): detect_splices now runs GBM-first dense scan.
  - PRIMARY tunables (instant, no retrain) — all live in detector.py:
      GBM_THRESHOLD         current 0.985  — P(splice)>thr is an emit; higher = fewer FP
      GBM_MIN_SEP_S         current 2.5    — dedupe distance for adjacent emits
      ANALYSIS_STRIDE_S     current 0.2    — dense-scan stride (smaller = denser, slower)
  - RETRAIN-required tunables (slower, ~3-5 min each) — .omc/classifier/train_classifier.py:
      make_pipeline() GradientBoostingClassifier hyperparams (n_estimators, max_depth,
      learning_rate, subsample). After edits run: uv run python .omc/classifier/train_classifier.py
  - Feature engineering: edit features.py (extend FEATURE_NAMES). Requires retrain.
  - DO NOT tune ml_config.py — its params only drive the LEGACY binary OOF path
    which is DEAD when the multi-class bundle is loaded. Edits have ZERO effect on combined.
  - DO NOT tune the _detect_phase / _detect_crossfade / _detect_cpe / _detect_pairwise
    internal thresholds — those functions are only called in the DSP-fallback path
    (classifier missing) which is not exercised by evaluate.py on this machine.

Read program.md for the overall goal, then execute exactly ONE experiment iteration:
1. Check git state and read .omc/coordination/baseline_metrics.json for context.
2. Form a hypothesis to improve combined. Prefer PRIMARY tunables (instant loop).
   Touch RETRAIN tunables only when the primary knob space feels exhausted.

   CRITICAL: Do NOT repeat hypotheses that have already been tried and failed.
   RECENT FAILED HYPOTHESES (last 30):
${recent_failures:-  (none yet)}

   If your idea matches any of the above, pick a DIFFERENT one.
3. Edit detector.py (or features.py / train_classifier.py if retraining) with the
   smallest viable change. If retraining, also run train_classifier.py in this step.
4. git commit -m \"hypothesis: <one-line description>\"
5. Run: uv run python evaluate.py --with-classifier
6. Parse combined score from the LAST 'combined:' line in the output.
7. CURRENT BEST (authoritative, from baseline_metrics.json): ${current_best}
   If combined > ${current_best} (strictly greater):    print RESULT:keep-pending
   If combined <= ${current_best}:                      git reset --hard HEAD~1 ; print RESULT:discard
   Do NOT rely on results.tsv for the 'previous best' — it lags the baseline
   file after successful keeps.
8. Do NOT run verify_agent.py yourself — the wrapper handles it.
9. Do NOT write to results.tsv — the wrapper handles it.

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

        # Auto-log every iteration to results.tsv (agent-independent reliability).
        # Called by each case handler with status + optional commit override (for discards).
        log_to_results_tsv() {
            local status="$1"
            local commit="${2:-$(git log -1 --format=%h 2>/dev/null)}"
            local subject="${3:-$(git log -1 --format=%s 2>/dev/null | sed 's/^hypothesis: //' | tr '\t' ' ')}"
            local combined=$(echo "$last_output" | grep -oE 'combined[: ]+[0-9]+\.[0-9]+' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "0")
            local splice_f1=$(echo "$last_output" | grep -oE 'splice_f1:\s*[0-9]+\.[0-9]+' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "0")
            local clean_score=$(echo "$last_output" | grep -oE 'clean_score:\s*[0-9]+\.[0-9]+' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "0")
            local precision=$(echo "$last_output" | grep -oE 'precision:\s*[0-9]+\.[0-9]+' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "0")
            local recall=$(echo "$last_output" | grep -oE 'recall:\s*[0-9]+\.[0-9]+' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "0")
            local fp_rate=$(echo "$last_output" | grep -oE 'fp_rate:\s*[0-9]+\.[0-9]+' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "0")
            local clean_fp=$(echo "$last_output" | grep -oE 'dsp_clean_fp:\s*[0-9]+' | tail -1 | grep -oE '[0-9]+' || echo "0")

            if [ ! -s "$RESULTS" ]; then
                printf 'commit\tcombined\tsplice_f1\tclean_score\tprecision\trecall\tfp_rate\tclean_fp\tstatus\tdescription\n' > "$RESULTS"
            fi

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$commit" "$combined" "$splice_f1" "$clean_score" "$precision" "$recall" "$fp_rate" "$clean_fp" "$status" "$subject" \
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

                # verify_agent.py has its own 120s timeout for evaluate.py internally.
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

                    # Update baseline_metrics.json so the NEXT iteration sees the
                    # new current_best. Without this, claude would compare against
                    # a stale baseline and accept no-op commits as "improvements".
                    python3 -c "
import json, os
bf = os.path.join('$PROJECT_DIR', '.omc/coordination/baseline_metrics.json')
try:
    data = json.load(open(bf))
except Exception:
    data = {}
data['combined'] = float('$reported')
data['git_sha'] = __import__('subprocess').run(['git','rev-parse','--short','HEAD'], capture_output=True, text=True).stdout.strip()
data['timestamp'] = __import__('datetime').datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
data.setdefault('note', '')
json.dump(data, open(bf, 'w'), indent=2)
"

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

                    # Classifier training now happens in-loop via evaluate.py --with-classifier
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
