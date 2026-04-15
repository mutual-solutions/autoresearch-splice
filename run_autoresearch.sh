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

        # Parse result from claude output
        last_status=$(tail -20 "$LOG_FILE" | grep -o 'RESULT:[a-z-]*' | tail -1 | cut -d: -f2 || echo "unknown")

        case "$last_status" in
            keep)
                consecutive_discards=0
                echo "$(date -Iseconds) Iteration: KEEP (reset discard counter)" >> "$LOG_FILE"
                ;;
            discard|verify-fail|unknown)
                consecutive_discards=$((consecutive_discards + 1))
                echo "$(date -Iseconds) Iteration: $last_status (discards: $consecutive_discards/$MAX_CONSECUTIVE_DISCARDS)" >> "$LOG_FILE"
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
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "STATUS: RUNNING"
        else
            echo "STATUS: STOPPED"
        fi
        if [ -f "$RESULTS" ]; then
            echo "Last result: $(tail -1 "$RESULTS")"
            echo "Total experiments: $(($(wc -l < "$RESULTS") - 1))"
            keeps=$(grep -c "keep" "$RESULTS" 2>/dev/null || echo 0)
            echo "Keeps: $keeps"
        fi
        if [ -f "$LOG_FILE" ]; then
            echo "Last log: $(tail -1 "$LOG_FILE")"
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
