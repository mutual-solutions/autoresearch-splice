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

        iteration_output=$(claude -p "You are running autoresearch on the audio splice detection project.

Read program.md for full instructions, then execute exactly ONE experiment iteration:
1. Check git state and read detector.py, ml_config.py, and results.tsv
2. Form a hypothesis to improve combined score. You can:
   - Tune DSP parameters in detector.py (thresholds, algorithms, features)
   - Tune ML classifier parameters in ml_config.py (n_estimators, max_depth, OOF_THRESHOLD, etc.)
   - Combine both DSP and ML changes in one hypothesis
3. Edit detector.py and/or ml_config.py with the smallest viable change
4. git commit -m \"hypothesis: <description>\"
5. Run: uv run python evaluate.py --with-classifier
6. Parse combined score from the LAST 'combined:' line in output (this is combined_full when classifier runs)
7. If combined > previous best: print RESULT:keep-pending
8. If combined <= previous best: git reset --hard HEAD~1, print RESULT:discard
9. Do NOT run verify_agent.py yourself. The shell wrapper handles verification.

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

        case "$last_status" in
            keep-pending)
                # STRUCTURAL VERIFICATION: shell runs verify_agent.py directly (not Claude)
                echo "$(date -Iseconds) Keep pending — running structural verification..." >> "$LOG_FILE"

                # Extract reported combined from Claude's output
                reported=$(echo "$last_output" | grep -oE 'combined[: ]+[0-9]+\.[0-9]+' | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "0")

                verify_output=$(timeout 300 uv run python .omc/coordination/verify_agent.py \
                    --agent-name autoresearch \
                    --reported-combined "$reported" 2>&1) || true
                verify_exit=$?

                if [ $verify_exit -eq 124 ]; then
                    echo "$(date -Iseconds) VERIFY-TIMEOUT: verification took >300s, treating as fail" >> "$LOG_FILE"
                    verify_exit=1
                fi
                echo "$verify_output" >> "$LOG_FILE"

                if [ $verify_exit -eq 0 ]; then
                    consecutive_discards=0
                    echo "$(date -Iseconds) VERIFIED KEEP (combined=$reported)" >> "$LOG_FILE"

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
                    # Verification failed — revert
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
