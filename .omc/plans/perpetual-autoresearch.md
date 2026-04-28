# Plan: Perpetual Autoresearch Wrapper

**Complexity:** LOW (3 files)
**Date:** 2026-04-15

## Context

The autoresearch-splice project runs a Karpathy-style autonomous research loop (program.md) where Claude Code iterates: hypothesis -> edit detector.py -> evaluate -> keep/revert. Currently this is run manually. This plan wraps it in a tmux-managed shell script with verify-on-keep gating and a file-based stop toggle.

### Existing assets
- `program.md` — loop instructions for Claude Code (immutable by agents, editable by human)
- `.omc/coordination/verify_agent.py` — 4-check verification (metric rerun, git diff audit, anomaly, preflight). Exits 0 on HIGH/MEDIUM confidence, 1 on LOW.
- `results.tsv` — tab-separated experiment log (columns: commit, combined, splice_f1, clean_score, precision, recall, fp_rate, clean_fp, status, description)
- `CLAUDE.md` — agent rules referencing verify protocol

### Key design decisions
- verify_agent.py already handles the full verification pipeline; the wrapper just needs to invoke it and act on exit code
- The loop prompt fed to `claude` CLI should reference program.md (Claude reads it each iteration)
- Circuit breaker is a simple counter in the shell script, not a separate file

## Guardrails

**Must have:**
- `start` launches in tmux so terminal is recoverable
- `stop` is non-blocking (touch file, loop checks it)
- verify-on-keep: every "keep" decision re-runs verify_agent.py before actually keeping
- Circuit breaker at 50 consecutive discards
- All verify failures logged to results.tsv as "verify-fail" status

**Must NOT have:**
- No daemon/systemd/launchd. Just tmux.
- No new Python files. Shell script only for the wrapper.
- No changes to verify_agent.py, prepare.py, or preflight.py

## Task Flow

### Step 1: Create `run_autoresearch.sh`

**File:** `run_autoresearch.sh`

Shell script with 3 subcommands:

**`start`:**
- Checks tmux session "autoresearch" doesn't already exist
- Removes `.omc/autoresearch-stop` if present
- Creates tmux session "autoresearch" running the loop
- The loop: invokes `claude` with a prompt that instructs it to read program.md and run the experiment loop, with these additions woven into the prompt:
  - After each "keep" decision, run: `uv run python .omc/coordination/verify_agent.py --agent-name autoresearch --reported-combined <score>`
  - If verify exits non-zero: `git reset --hard HEAD~1`, log "verify-fail" to results.tsv, increment discard counter
  - Check `[ -f .omc/autoresearch-stop ]` at the start of each iteration; exit cleanly if present
  - Track consecutive discards; exit with message if >= 50

**`stop`:**
- `touch .omc/autoresearch-stop`
- Print message: "Stop signal sent. Loop will exit within 1 iteration."

**`status`:**
- `tmux has-session -t autoresearch 2>/dev/null` to check if running
- `tail -1 results.tsv` to show last experiment
- Count of keep/discard/verify-fail from results.tsv

**Acceptance criteria:**
- [ ] `./run_autoresearch.sh start` creates tmux session, `tmux ls` shows "autoresearch"
- [ ] `./run_autoresearch.sh stop` creates `.omc/autoresearch-stop`
- [ ] `./run_autoresearch.sh status` prints session state + last result
- [ ] Script is executable (`chmod +x`)

### Step 2: Update `program.md` — add verify-on-keep gate section

**File:** `program.md`

Add a new section after the "Experiment loop" section (after step 10):

```markdown
## Verify-on-keep gate

After step 9 (keep decision), before advancing:

1. Run: `uv run python .omc/coordination/verify_agent.py --agent-name autoresearch --reported-combined <combined_score>`
2. If exit code is 0 (HIGH or MEDIUM confidence): keep the commit, advance branch.
3. If exit code is non-zero (LOW confidence): treat as discard — `git reset --hard HEAD~1`, log status as `verify-fail` in results.tsv.

## Stop signal

At the start of each iteration, check: `[ -f .omc/autoresearch-stop ]`
If the file exists, exit the loop cleanly. The human creates this file via `./run_autoresearch.sh stop`.
```

**Acceptance criteria:**
- [ ] program.md contains verify-on-keep section referencing verify_agent.py with correct CLI args
- [ ] program.md contains stop signal check instruction
- [ ] Existing sections are unchanged

### Step 3: Document `.omc/autoresearch-stop` convention

This is not a separate file to create -- it is the convention used by steps 1 and 2. Verify it works end-to-end:

**Acceptance criteria:**
- [ ] `./run_autoresearch.sh stop` creates `.omc/autoresearch-stop`
- [ ] `./run_autoresearch.sh start` removes `.omc/autoresearch-stop` on launch
- [ ] `.gitignore` includes `.omc/autoresearch-stop` (or `.omc/` is already ignored)

## Success Criteria

1. `./run_autoresearch.sh start` launches a tmux session running the autoresearch loop via `claude` CLI
2. The loop applies verify_agent.py on every keep, reverting on LOW confidence
3. `./run_autoresearch.sh stop` halts the loop within one iteration without killing the process
4. Circuit breaker stops execution after 50 consecutive discards
5. All outcomes (keep, discard, verify-fail, crash) are logged to results.tsv
