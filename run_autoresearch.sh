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

        claude -p "You are running autoresearch on the audio splice detection project.

Read program.md for full instructions, then execute exactly ONE experiment iteration:
1. Check git state and read detector.py and results.tsv
2. Form a hypothesis (one specific DSP idea to improve combined score)
3. Edit detector.py with the smallest viable change
4. git commit -m \"hypothesis: <description>\"
5. Run: uv run python prepare.py
6. Parse combined score from output
7. If combined > previous best: run uv run python .omc/coordination/verify_agent.py --agent-name autoresearch --reported-combined <score>
   - If verify PASSES: keep commit, log to results.tsv as 'keep'
   - If verify FAILS: git reset --hard HEAD~1, log to results.tsv as 'verify-fail'
8. If combined <= previous best: git reset --hard HEAD~1, log to results.tsv as 'discard'
9. Print exactly one line at the end: RESULT:<status> where status is keep, discard, or verify-fail

Do NOT loop. Execute exactly ONE iteration and exit." \
            --allowedTools "Bash Edit Read Write Grep Glob" \
            --add-dir "$PROJECT_DIR" \
            2>&1 | tee -a "$LOG_FILE" | tail -5

        # Check if claude itself failed (rate limit, token exhausted, crash)
        claude_exit=$?
        last_output=$(tail -50 "$LOG_FILE")

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
            keep)
                consecutive_discards=0
                echo "$(date -Iseconds) Iteration: KEEP (reset discard counter)" >> "$LOG_FILE"
                ;;
            discard|verify-fail)
                consecutive_discards=$((consecutive_discards + 1))
                echo "$(date -Iseconds) Iteration: $last_status (discards: $consecutive_discards/$MAX_CONSECUTIVE_DISCARDS)" >> "$LOG_FILE"
                ;;
            unknown)
                # Unknown = claude didn't produce RESULT line, don't count as experiment discard
                echo "$(date -Iseconds) Iteration: unknown output (not counting toward circuit breaker)" >> "$LOG_FILE"
                sleep 30
                ;;
        esac

        # Brief pause between iterations
        sleep 2
    done

    echo "$(date -Iseconds) Autoresearch loop ended" >> "$LOG_FILE"
}

case "${1:-help}" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "Autoresearch already running. Use 'stop' or 'status'."
            exit 1
        fi
        rm -f "$STOP_FILE"
        tmux new-session -d -s "$SESSION" "bash $0 _loop"
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
        # Fast — no subprocesses, no prepare.py, just file reads
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "🟢 RUNNING"
        else
            echo "🔴 STOPPED"
        fi
        [ -f "$STOP_FILE" ] && echo "⏸  Stop signal pending"
        if [ -f "$RESULTS" ]; then
            total=$(($(wc -l < "$RESULTS") - 1))
            keeps=$(grep -c $'	keep\t' "$RESULTS" 2>/dev/null || true)
            discards=$(grep -c $'	discard\t' "$RESULTS" 2>/dev/null || true)
            vfails=$(grep -c "verify-fail" "$RESULTS" 2>/dev/null || true)
            keeps=${keeps:-0}; discards=${discards:-0}; vfails=${vfails:-0}
            echo "Experiments: $total (keep: $keeps, discard: $discards, verify-fail: $vfails)"
            echo "Last: $(tail -1 "$RESULTS" | cut -f9,10)"
        fi
        if [ -f "$LOG_FILE" ]; then
            last_log=$(tail -1 "$LOG_FILE")
            echo "Log: $last_log"
            # Show backoff state if rate limited
            echo "$last_log" | grep -qi "backoff" && echo "⚠️  Rate limited — backing off"
        fi
        ;;
    _loop)
        run_loop
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
