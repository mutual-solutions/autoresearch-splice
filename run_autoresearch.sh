#!/usr/bin/env bash
# Perpetual autoresearch wrapper. Runs Karpathy-style experiment loop in tmux.
# Usage: ./run_autoresearch.sh start|stop|status
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
SESSION="autoresearch"
STOP_FILE="$PROJECT_DIR/.omc/autoresearch-stop"
# US-515 phase 2: structured events go to the unified JSONL log (via `_log`).
# Freeform subprocess stdout/stderr goes to the child-stderr sidecar so it
# does not corrupt the JSONL stream. Both live under .omc/logs/ (gitignored).
CHILD_STDERR_LOG="$PROJECT_DIR/.omc/logs/child-stderr.log"
mkdir -p "$PROJECT_DIR/.omc/logs"
RESULTS="$PROJECT_DIR/results.tsv"
MAX_CONSECUTIVE_DISCARDS=50
# US-518: disk-safety constants. Tune here if SSD capacity changes.
FEATURE_CACHE_CAP_MB=8192    # 8 GB absolute cap on .omc/feature_cache/
DISK_MIN_FREE_KB=$((5 * 1024 * 1024))   # 5 GB — hard exit at loop start
DISK_ITER_MIN_FREE_KB=$((3 * 1024 * 1024)) # 3 GB — soft stop mid-iteration

# Emit a structured JSONL event via .omc/coordination/log_cli.py (US-515
# phase 2). Signature:
#     _log LEVEL SUBSYSTEM EVENT [key=value ...]
# Reserved key=value flags: `claude_visible=true`, `oracle_sensitive=true`.
# Failures are suppressed — logging must never block the loop.
_log() {
    uv run python "$PROJECT_DIR/autoresearch/log_cli.py" "$@" 2>>"$CHILD_STDERR_LOG" || true
}

_eval_cleanup() {
    # Shred (best-effort) and remove the decrypted eval tree. Called on
    # every loop exit path via the EXIT/HUP/INT/TERM trap so plaintext
    # eval audio never outlives the loop process.
    if [ -n "${EVAL_TMP_PARENT:-}" ] && [ -d "$EVAL_TMP_PARENT" ]; then
        rm -rf "$EVAL_TMP_PARENT" 2>/dev/null || true
        _log INFO wrapper eval.cleanup parent="$EVAL_TMP_PARENT"
    fi
}

# US-518: feature_cache LRU prune with absolute GB cap.
# Evicts oldest sha dirs first; never evicts the current $OMC_FEATURES_PY_SHA.
# Called at loop start (replaces US-504 block) and after each keep-path.
_prune_feature_cache_gb() {
    local cache_dir="$PROJECT_DIR/.omc/feature_cache"
    [ -d "$cache_dir" ] || return 0
    local cur_sha="${OMC_FEATURES_PY_SHA:-}"
    if [ -z "$cur_sha" ]; then
        cur_sha="$(git hash-object "$PROJECT_DIR/splice/features.py" 2>/dev/null || echo "")"
    fi

    # Sum total size in MB.
    local total_mb
    total_mb=$(du -sm "$cache_dir" 2>/dev/null | awk '{print $1}')
    total_mb="${total_mb:-0}"

    if [ "$total_mb" -le "$FEATURE_CACHE_CAP_MB" ]; then
        return 0
    fi

    _log INFO wrapper feature_cache.prune.start total_mb="$total_mb" cap_mb="$FEATURE_CACHE_CAP_MB"

    # Evict oldest sha dirs first (LRU by mtime ascending).
    # Use stat -f on macOS, fallback to ls -t + reverse for portability.
    # No local array (bash 3.2 compat); iterate via while-read pipeline.
    while IFS= read -r _prune_d; do
        [ -d "$_prune_d" ] || continue
        local _prune_base
        _prune_base="$(basename "$_prune_d")"
        # Never evict the currently-active sha.
        [ "$_prune_base" = "$cur_sha" ] && continue
        local _prune_dir_mb
        _prune_dir_mb=$(du -sm "$_prune_d" 2>/dev/null | awk '{print $1}')
        _prune_dir_mb="${_prune_dir_mb:-0}"
        _log INFO wrapper feature_cache.prune.lru \
            stale="$_prune_base" current="$cur_sha" dir_mb="$_prune_dir_mb"
        rm -rf "$_prune_d"
        total_mb=$((total_mb - _prune_dir_mb))
        if [ "$total_mb" -le "$FEATURE_CACHE_CAP_MB" ]; then
            break
        fi
    done < <(
        # macOS stat + sort gives mtime-ascending order (oldest first).
        find "$cache_dir" -mindepth 1 -maxdepth 1 -type d \
            | xargs stat -f '%m %N' 2>/dev/null \
            | sort -n \
            | awk '{print $2}' \
        || find "$cache_dir" -mindepth 1 -maxdepth 1 -type d
    )

    local final_mb
    final_mb=$(du -sm "$cache_dir" 2>/dev/null | awk '{print $1}')
    _log INFO wrapper feature_cache.prune.done total_mb="${final_mb:-0}" cap_mb="$FEATURE_CACHE_CAP_MB"
}

# US-518: sweep stale /tmp/.ar-eval-* dirs left by ungraceful exits.
# Dirs older than 2 hours are safe to remove (active evals use fresh dirs).
_cleanup_stale_tmp_eval() {
    local count=0
    local freed_mb=0
    local _cst_root _cst_d _cst_mb _cst_resolved
    # macOS: /tmp is a symlink to /private/tmp; use -L (follow symlinks) with
    # find. Also sweep TMPDIR (/var/folders/.../T/) for the per-user sandbox.
    # Avoid local arrays (bash 3.2 compat) — iterate roots explicitly.
    for _cst_root in "/tmp" "${TMPDIR:-}"; do
        [ -n "$_cst_root" ] || continue
        [ -d "$_cst_root" ] || continue
        while IFS= read -r _cst_d; do
            [ -d "$_cst_d" ] || continue
            _cst_mb=$(du -sm "$_cst_d" 2>/dev/null | awk '{print $1}')
            _cst_mb="${_cst_mb:-0}"
            _log INFO wrapper tmp_eval.sweep path="$_cst_d" mb="$_cst_mb"
            rm -rf "$_cst_d"
            count=$((count + 1))
            freed_mb=$((freed_mb + _cst_mb))
        done < <(find -L "$_cst_root" -maxdepth 1 -name '.ar-eval-*' -type d -mmin +120 2>/dev/null)
    done

    _log INFO wrapper tmp_eval.sweep_summary count="$count" freed_mb="$freed_mb"
}

# US-518: disk-space gate. At loop start: hard-exit if below DISK_MIN_FREE_KB.
# Mid-iteration: soft-stop via STOP_FILE if below DISK_ITER_MIN_FREE_KB.
# $1 = "hard" (default) or "soft"
_check_disk_space() {
    local mode="${1:-hard}"
    local avail_kb
    avail_kb=$(df -k "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
    avail_kb="${avail_kb:-0}"

    if [ "$mode" = "hard" ]; then
        if [ "$avail_kb" -lt "$DISK_MIN_FREE_KB" ]; then
            _log CRITICAL wrapper preflight.disk_critical \
                avail_kb="$avail_kb" threshold_kb="$DISK_MIN_FREE_KB" \
                suggested_action="rm -rf .omc/feature_cache/<stale-sha> and clean /tmp/.ar-eval-*"
            echo "ERROR: Disk space critical — ${avail_kb} KB free, need ${DISK_MIN_FREE_KB} KB." >&2
            echo "  Reclaim space: du -sm .omc/feature_cache/* && rm -rf .omc/feature_cache/<stale-sha>" >&2
            echo "  Then: find /tmp -maxdepth 1 -name '.ar-eval-*' -type d -exec rm -rf {} +" >&2
            exit 1
        fi
    else
        if [ "$avail_kb" -lt "$DISK_ITER_MIN_FREE_KB" ]; then
            _log CRITICAL wrapper disk.low \
                avail_kb="$avail_kb" threshold_kb="$DISK_ITER_MIN_FREE_KB" \
                action="soft-halt via STOP_FILE"
            touch "$STOP_FILE"
        fi
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
    _log INFO wrapper phase.finished phase="$name" elapsed_s="$elapsed"
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
    # Field names are load-bearing: scripts/phase_stats.py reads them off
    # this event verbatim. `-` = phase skipped; `?` = never-set (bug).
    _log INFO wrapper iteration.phase \
        iter="${sha}" status="${status}" \
        total="${ITER_total_S:-?}" claude="${ITER_claude_S:-?}" \
        retrain="${ITER_retrain_S:-?}" eval="${ITER_eval_S:-?}" \
        verify="${ITER_verify_S:-?}" note="${ITER_note_S:-?}"
}

# Paths that MUST survive every hypothesis rollback. Wrapper and
# infrastructure — NOT the claude-editable hypothesis surface
# (splice/detector.py / splice/features.py / classifier joblib+meta). Without
# guarding, any in-flight edit to these files is wiped by the next
# `git reset --hard` on a discard. Root cause of losses earlier in
# this session (the portable-timeout fix, splice/features.py cache code).
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
        -- "${_GUARD_PATHS[@]}" 2>>"$CHILD_STDERR_LOG" || return 1
    return 0
}

_guard_stash_pop() {
    local top_label
    top_label=$(git stash list -1 2>/dev/null | grep -oE 'autoresearch-guard-[0-9]+-[0-9]+' | head -1)
    [ -z "$top_label" ] && return 0
    if ! git stash pop --quiet 2>>"$CHILD_STDERR_LOG"; then
        _log WARN wrapper guard.stash_conflict stash="$top_label" \
            recover="git stash list / apply"
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
    git reset --hard "$target" >>"$CHILD_STDERR_LOG" 2>&1
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
        git commit -m "note: ${status} ${short_sha}" >>"$CHILD_STDERR_LOG" 2>&1 || true
    fi
    _log INFO wrapper note.appended status="$status" sha="$short_sha" \
        combined="$combined" claude_visible=true

    # Compact if we've grown past the threshold.
    uv run python "$PROJECT_DIR/scripts/notebook_digest.py" >>"$CHILD_STDERR_LOG" 2>&1 \
        || _log ERROR pipeline failure script=notebook_digest.py rc="$?"
    if ! git diff --quiet "$notes" 2>/dev/null; then
        git add "$notes"
        git commit -m "note: digest compaction" >>"$CHILD_STDERR_LOG" 2>&1 || true
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
        # `head -3 | grep -q` closes stdin after the first match or at line 3;
        # upstream `git log` can SIGPIPE under pipefail. Isolate in a subshell.
        if (set +o pipefail; git log --format=%s --reverse "${sha}..HEAD" 2>/dev/null \
             | head -3 | grep -q "^baseline:"); then
            continue
        fi
        if [ -f "$results_tsv" ] \
           && grep -q "^${sha:0:7}	" "$results_tsv" 2>/dev/null; then
            continue
        fi
        local short="${sha:0:7}"
        echo "DEEP-ORPHAN: ${short} \"${subject}\" — no paired baseline: in next 3 commits, no row in results.tsv" >&2
        _log WARN wrapper orphan.deep sha="$short" subject="$subject"
        count=$((count + 1))
    done < <(git log --format='%H|%s' "$branch_range" 2>/dev/null)

    if [ $count -gt 0 ]; then
        _log WARN wrapper orphan.deep.summary count="$count" depth="$depth" \
            hint="git log --oneline -$depth"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Phase 1 of US-514 retest: internal wrapper verbs. Additive helpers so the
# retest subcommand can reuse the loop's keep-path semantics without lifting
# the logic into Python. `_do_keep_path` performs tag + baseline refresh +
# baseline commit + versions.json append + note append, same as the loop
# keep branch. `_do_ensure_classifier_fresh` performs the sha-drift gate
# and retrain. Both are invoked from the dispatcher via the `_keep_path`
# and `_ensure_classifier_fresh` verbs. The LOOP keep body (lines in the
# 807-919 band, post-edit) and the loop staleness gate are NOT edited in
# phase 1 — commit-message byte-identity against historical loop keeps is
# preserved when `--retest-origin` is absent.
# ---------------------------------------------------------------------------

_do_keep_path() {
    # Positional args:
    #   $1 = hypothesis_commit  (short sha of the commit we are accepting)
    #   $2 = reported           (combined score as a string)
    #   $3 = current_best       (prior baseline combined — used in log msg)
    #   $4 = hypothesis_subject (commit subject minus `hypothesis: ` prefix)
    #   [--retest-origin <orig_sha>] optional tail args for retest provenance
    local hypothesis_commit="$1"
    local reported="$2"
    local current_best="$3"
    local hypothesis_subject="$4"
    shift 4 || true
    local retest_origin=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --retest-origin)
                retest_origin="${2:-}"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    local baseline_suffix="after keep $(git log -1 --format=%h "$hypothesis_commit")"
    local note_status="keep"
    local note_subject="$hypothesis_subject"
    if [ -n "$retest_origin" ]; then
        baseline_suffix="after retest-recovery $retest_origin"
        note_status="keep-retest"
        note_subject="$hypothesis_subject (retest-recovery of $retest_origin)"
    fi

    _log INFO wrapper keep_path.start hypothesis="$hypothesis_commit" \
        reported="$reported" prev="$current_best" \
        retest_origin="${retest_origin:-none}"

    local VERSION
    VERSION=$(($(git tag -l 'detector-v*' 2>/dev/null | wc -l) + 1))
    git tag "detector-v$VERSION"
    local SNAP="$PROJECT_DIR/.omc/classifier/detector_v${VERSION}.py"
    cp "$PROJECT_DIR/splice/detector.py" "$SNAP"

    # Update baseline_metrics.json from the RESULTS_TSV line on disk.
    python3 - "$PROJECT_DIR" <<'PYEOF'
import json, os, subprocess, sys, re, datetime
proj = sys.argv[1]
bf = os.path.join(proj, 'autoresearch/baseline_metrics.json')
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

    if ! git diff --quiet autoresearch/baseline_metrics.json; then
        git add autoresearch/baseline_metrics.json
        git commit -m "baseline: combined=$reported $baseline_suffix" \
            >>"$CHILD_STDERR_LOG" 2>&1
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
    set +e
    local shap_shift_line
    shap_shift_line=$(uv run python "$PROJECT_DIR/scripts/shap_shift.py" 2>>"$CHILD_STDERR_LOG" | head -1)
    local shift_rc=$?
    set -e
    if [ $shift_rc -ne 0 ]; then
        _log ERROR pipeline failure script=shap_shift.py rc="$shift_rc"
    fi
    if [ -n "$shap_shift_line" ]; then
        _log INFO wrapper keep_path.shap_shift line="$shap_shift_line"
        echo "[auto] $shap_shift_line" >> "$PROJECT_DIR/.omc/last_reflection.md"
    fi

    _phase_start note
    _append_note "$note_status" "$hypothesis_commit" "$note_subject"
    _phase_end note

    _log INFO wrapper keep_path.done version="$VERSION"
}

_do_ensure_classifier_fresh() {
    # Standalone classifier staleness gate for the dispatcher verb (retest
    # + any future caller). Returns 0 when features.py sha matches the
    # trained classifier's meta sha OR when a retrain+stage succeeded.
    # Returns 1 when retrain failed. Does NOT amend onto any caller commit
    # — that amend-safety branch is loop-specific and stays in run_loop.
    # When this function retrains successfully AND a `hypothesis:` HEAD is
    # present, callers (the loop) own commit policy; this function only
    # stages the artifacts so the caller can amend or commit separately.
    # US-505b: gate on BOTH features.py AND train_classifier.py shas so
    # hyperparam-only edits (no features.py change) also trigger retrain.
    local features_sha_now features_sha_trained train_sha_now train_sha_trained
    features_sha_now=$(git hash-object "$PROJECT_DIR/splice/features.py" 2>/dev/null || echo "")
    train_sha_now=$(git hash-object "$PROJECT_DIR/splice/classifier/train_classifier.py" 2>/dev/null || echo "")
    read -r features_sha_trained train_sha_trained <<<"$(python3 -c "
import json
try:
    d = json.load(open('splice/classifier/fp_classifier.meta.json'))
    print(d.get('features_py_sha') or '_', d.get('train_classifier_py_sha') or '_')
except Exception:
    print('_ _')
" 2>/dev/null)"
    if [ -z "$features_sha_now" ] || [ -z "$train_sha_now" ]; then
        _log INFO wrapper retrain.fresh features_sha="$features_sha_now" train_sha="$train_sha_now" reason=unhashable
        return 0
    fi
    if [ "$features_sha_now" = "$features_sha_trained" ] && [ "$train_sha_now" = "$train_sha_trained" ]; then
        _log INFO wrapper retrain.fresh features_sha="$features_sha_now" train_sha="$train_sha_now"
        return 0
    fi
    _log INFO wrapper retrain.drift \
        features_was="$features_sha_trained" features_now="$features_sha_now" \
        train_was="$train_sha_trained" train_now="$train_sha_now"
    local retrain_cmd=()
    if command -v timeout >/dev/null 2>&1; then
        retrain_cmd=(timeout 300 uv run python splice/classifier/train_classifier.py)
    elif command -v gtimeout >/dev/null 2>&1; then
        retrain_cmd=(gtimeout 300 uv run python splice/classifier/train_classifier.py)
    else
        retrain_cmd=(uv run python splice/classifier/train_classifier.py)
    fi
    set +e
    "${retrain_cmd[@]}" >>"$CHILD_STDERR_LOG" 2>&1
    local rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        _log ERROR wrapper retrain.failed rc="$rc" caller=ensure_classifier_fresh
        return 1
    fi
    git add splice/classifier/fp_classifier.joblib \
            splice/classifier/fp_classifier.meta.json >>"$CHILD_STDERR_LOG" 2>&1 || true
    return 0
}

# Retest sentinel guard: refuse to start the loop while a retest is
# between main-tree cherry-pick and baseline commit. See
# .omc/plans/ralplan-retest-discards.md (step 7).
_RETEST_SENTINEL_PATH="$PROJECT_DIR/.omc/retest-in-progress"
_refuse_if_retest_sentinel() {
    if [ -f "$_RETEST_SENTINEL_PATH" ]; then
        cat >&2 <<SENTINEL_EOF
ERROR: .omc/retest-in-progress sentinel found — a retest crashed mid-recovery.
Inspect: .omc/retest-report.md and \`git log --oneline -10\` on main.
Recover: verify the HEAD commit (cherry-pick may be partial), then
         rm -f .omc/retest-in-progress
SENTINEL_EOF
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# US-517: supervisor_agent.py --maintain helpers
# ---------------------------------------------------------------------------

_maybe_crash_maintain() {
    # Invoked after each --diagnose site. Checks the disable sentinel,
    # runs --maintain --trigger=crash, and halts the loop on exit code 1.
    if [ -f "$PROJECT_DIR/.omc/maintainer-disabled" ]; then
        return 0
    fi
    set +e
    PYTHONPATH="$PROJECT_DIR" uv run python autoresearch/supervisor_agent.py --maintain --trigger=crash \
        >>"$CHILD_STDERR_LOG" 2>&1
    _maintain_rc=$?
    set -e
    if [ "$_maintain_rc" -eq 1 ]; then
        touch "$STOP_FILE"
        _log WARN wrapper maintain.halt trigger=crash
    elif [ "$_maintain_rc" -ge 2 ]; then
        _log ERROR wrapper maintain.crash rc="$_maintain_rc"
    fi
}

_maybe_periodic_maintain() {
    # Invoked after each iteration summary. Fires on multiples of 10 iterations.
    if [ -f "$PROJECT_DIR/.omc/maintainer-disabled" ]; then
        return 0
    fi
    _maintain_iter_count=$(( $(wc -l < "$RESULTS" 2>/dev/null || echo 1) - 1 ))
    if [ "$_maintain_iter_count" -gt 0 ] && [ $((_maintain_iter_count % 10)) -eq 0 ]; then
        set +e
        PYTHONPATH="$PROJECT_DIR" uv run python autoresearch/supervisor_agent.py --maintain --trigger=periodic \
            >>"$CHILD_STDERR_LOG" 2>&1
        _maintain_rc=$?
        set -e
        if [ "$_maintain_rc" -eq 1 ]; then
            touch "$STOP_FILE"
            _log WARN wrapper maintain.halt trigger=periodic
        elif [ "$_maintain_rc" -ge 2 ]; then
            _log ERROR wrapper maintain.crash rc="$_maintain_rc"
        fi
    fi
}

run_loop() {
    cd "$PROJECT_DIR"
    rm -f "$STOP_FILE"
    consecutive_discards=0
    rate_limit_backoff=300  # start at 5 min, double each time, cap at 5 hours

    # Combined crash-log + eval-cleanup trap. Shred ordering matters: we
    # cleanup AFTER logging the crash so the cleanup failure (if any)
    # doesn't eat the crash signal. `_log` is a Python-subprocess call; on
    # clean signals (HUP/INT/TERM) the shell has time to run it. On SIGKILL
    # the trap does not fire at all, so there is no asymmetry to preserve.
    trap '_trap_ec=$?; _log CRITICAL wrapper loop.crash signal=$_trap_ec; _eval_cleanup' \
        EXIT HUP INT TERM

    _log INFO wrapper loop.started

    # Orphan-hypothesis detection. If a prior run crashed mid-iteration,
    # HEAD can be an unverified `hypothesis: ...` commit. US-518 smarter
    # handling: if the orphan changed detector.py/features.py/train_classifier.py
    # (real hypothesis), proceed as current — the next iteration will
    # evaluate or replace it naturally. If only metadata changed (safe to
    # lose), reset as before.
    orphan_subject=$(git log -1 --format=%s 2>/dev/null)
    if echo "$orphan_subject" | grep -q "^hypothesis:"; then
        orphan_sha=$(git log -1 --format=%h)
        if git diff --quiet HEAD~1 HEAD -- \
                splice/detector.py splice/features.py \
                splice/classifier/train_classifier.py 2>/dev/null; then
            # Metadata-only orphan — safe to reset, no hypothesis code lost.
            _log WARN wrapper orphan.hypothesis sha="$orphan_sha" \
                subject="$orphan_subject" action="reset (metadata-only)"
            _guarded_reset HEAD~1
        else
            # Real hypothesis with code changes — proceed as current so the
            # next iteration evaluates or supersedes it without data loss.
            _log WARN wrapper orphan.hypothesis sha="$orphan_sha" \
                subject="$orphan_subject" action="proceed (code changes present)"
        fi
    fi

    _detect_deep_orphans 20

    # ---- Eval dataset isolation ----------------------------------------
    # The plaintext data/eval/ tree does NOT exist on disk — it's been
    # encrypted into data/eval.tar.gz.enc and shredded (see
    # scripts/eval_crypto.py). Decrypt once per loop into a fresh /tmp
    # dir, export OMC_EVAL_DATA_ROOT for evaluate.py + preflight +
    # supervisor_agent (dataset_registry resolves eval_path against this at
    # import). The env var is explicitly UNSET in the claude subprocess's
    # env via `env -u` below, so the claude-driven iteration cannot
    # locate the decrypted tree; only this shell and its direct
    # evaluate.py children see it.
    if [ ! -f "$PROJECT_DIR/data/eval.tar.gz.enc" ]; then
        _log ERROR wrapper eval.setup_missing file=data/eval.tar.gz.enc \
            remedy="uv run python scripts/eval_crypto.py setup"
        echo "ERROR: data/eval.tar.gz.enc missing. Run: uv run python scripts/eval_crypto.py setup" >&2
        exit 1
    fi
    _decrypt_out=$(uv run python "$PROJECT_DIR/scripts/eval_crypto.py" decrypt --keep 2>&1)
    _decrypt_rc=$?
    if [ $_decrypt_rc -ne 0 ]; then
        _log ERROR wrapper eval.decrypt_failed rc="$_decrypt_rc" output="$_decrypt_out"
        echo "ERROR: eval decrypt failed. See .omc/logs/child-stderr.log" >&2
        exit 1
    fi
    EVAL_TMP_ROOT=$(echo "$_decrypt_out" | tail -1)
    EVAL_TMP_PARENT="$(dirname "$EVAL_TMP_ROOT")"
    export OMC_EVAL_DATA_ROOT="$EVAL_TMP_ROOT"
    _log INFO wrapper eval.decrypted path="$EVAL_TMP_ROOT"

    # US-518: sweep stale /tmp eval dirs, then enforce feature_cache cap.
    _cleanup_stale_tmp_eval
    _prune_feature_cache_gb
    # Preflight disk gate — hard-exit if less than 5 GB free.
    _check_disk_space hard

    while true; do
        # Stop signal check
        if [ -f "$STOP_FILE" ]; then
            _log INFO wrapper stop_signal
            break
        fi

        # Circuit breaker
        if [ "$consecutive_discards" -ge "$MAX_CONSECUTIVE_DISCARDS" ]; then
            _log WARN wrapper circuit_break consecutive_discards="$MAX_CONSECUTIVE_DISCARDS"
            break
        fi

        # US-518: mid-iteration disk checkpoint (post-claude slot handled below).
        _check_disk_space soft
        [ -f "$STOP_FILE" ] && { _log INFO wrapper stop_signal; break; }

        # US-518b: after a discard streak, OMC_FEATURES_PY_SHA points at a sha
        # _guarded_reset already purged from features.py; unset so the prune
        # guard (L47-50) falls back to git hash-object on the reset file.
        unset OMC_FEATURES_PY_SHA
        _prune_feature_cache_gb

        # Run one iteration via claude
        _log INFO wrapper iteration.start consecutive_discards="$consecutive_discards"
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
            # NOTE: `head -5` closes stdin early; under `set -euo pipefail`
            # that SIGPIPEs upstream `cut` → exit 141 → loop.crash. Disable
            # pipefail in the subshell so the head-based truncation can't
            # propagate as an error.
            recent_keeps=$(
                set +o pipefail
                awk -F'\t' '
                    NR==1 {next}
                    $13 == "keep" && $2 != "NA" {
                        printf "%s\t  + combined=%s [singing %s / korean %s / english %s]: %s\n", \
                               $2, $2, $7, $8, $9, $14
                    }' "$RESULTS" | sort -rn -k1,1 -t$'\t' | cut -f2- | head -5
            )

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
        current_best=$(python3 -c "import json; print(json.load(open('autoresearch/baseline_metrics.json')).get('combined', 0))" 2>/dev/null || echo "0")
        per_domain_state=$(python3 -c "
import json
d = json.load(open('autoresearch/baseline_metrics.json'))
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
            >/dev/null 2>>"$CHILD_STDERR_LOG" \
            || _log ERROR pipeline failure script=shap_rollup.py rc="$?"

        # Per-tunable exploration frontier (US-506). US-515 phase 2:
        # script emits `tunable.frontier.snapshot` to the unified log AND
        # prints the block to stdout, which we capture directly into the
        # prompt. No intermediate .omc/tunable_frontier.txt artifact.
        set +e
        tunable_frontier_block=$(
            uv run python "$PROJECT_DIR/scripts/tunable_frontier.py" 2>>"$CHILD_STDERR_LOG"
        )
        _tunable_rc=$?
        set -e
        if [ $_tunable_rc -ne 0 ]; then
            _log ERROR pipeline failure script=tunable_frontier.py rc="$_tunable_rc"
            tunable_frontier_block="  (tunable_frontier.py failed — see child-stderr sidecar)"
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
splice/evaluate.py iterates splice.dataset_registry.DATASETS (singing / korean / english)
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
  PRIMARY (instant) — splice/detector.py:
      GBM_THRESHOLD         P(splice)>thr is an emit (higher = fewer FP)
      GBM_MIN_SEP_S         dedupe distance for adjacent emits
      ANALYSIS_STRIDE_S     dense-scan stride (smaller = denser, slower)
  RETRAIN (~3 min) — splice/classifier/train_classifier.py:
      GradientBoostingClassifier hyperparams (n_estimators, max_depth,
      learning_rate, subsample).
  Feature engineering: splice/features.py (extend FEATURE_NAMES). Requires retrain.
  DO NOT tune ml_config.py — legacy path, zero effect on combined.
  DO NOT tune _detect_phase/_detect_crossfade/_detect_cpe/_detect_pairwise
  internal thresholds — DSP fallback path, not exercised by splice/evaluate.py.

==== HISTORY ==============================================================
RECENT FAILED HYPOTHESES (last 30; per-domain combined in brackets):
${recent_failures:-  (none yet)}

TOP-5 KEEPS (by combined):
${recent_keeps:-  (none yet)}

Read-only artifacts for deeper context:
  - results.tsv                               full iteration ledger
  - .omc/logs/autoresearch.jsonl              wrapper/verify structured log
  - .omc/classifier/versions.json             keep-only version history
  - git log --oneline --grep='hypothesis:\|baseline:' | head -40

==== ACCESS RULES =========================================================
- The plaintext eval corpus (data/eval/) is NOT on disk and is NOT
  accessible to you. Do not attempt to read it or infer its location.
  The wrapper will run splice/evaluate.py against an encrypted-then-decrypted
  copy after you commit.
- Do NOT run splice/evaluate.py — the wrapper does this and parses the result.
- Do NOT write to results.tsv or autoresearch/baseline_metrics.json
  — the wrapper owns both.
- Do NOT edit splice/evaluate.py, autoresearch/manifest.json,
  autoresearch/preflight.py — protected.

==== ONE ITERATION ========================================================
0. Read the RESEARCH NOTES above — your prior reflections about what you
   tried, why, and what to try next. If 5+ recent entries all failed on
   the same tunable axis, seriously consider a structural change
   (splice/features.py / splice/classifier/train_classifier.py) instead of another tweak.
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
4. Edit splice/detector.py / splice/features.py / splice/classifier/train_classifier.py with the smallest
   viable change. The wrapper auto-retrains when splice/features.py changes —
   you do NOT need to manually run train_classifier.py (US-505).
5. git add <touched files> ; git commit -m \"hypothesis: <one-line description>\"
6. Print RESULT:ready and exit. The wrapper will run splice/evaluate.py, compare
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

        # Iteration stdout is multi-line and can be thousands of chars. It
        # goes to the sidecar verbatim; the JSONL log gets a compact event
        # with line count + tail so reviewers can navigate without loading
        # the whole transcript.
        {
            printf '=== iteration_output %s ===\n' "$(date -Iseconds)"
            printf '%s\n' "$iteration_output"
            printf '=== end iteration_output ===\n'
        } >>"$CHILD_STDERR_LOG"
        _iter_lines=$(printf '%s\n' "$iteration_output" | wc -l | tr -d ' ')
        _iter_tail=$(printf '%s\n' "$iteration_output" | tail -1)
        _log INFO wrapper claude.output lines="$_iter_lines" tail="$_iter_tail"
        echo "$iteration_output" | tail -5

        last_output="$iteration_output"

        if echo "$last_output" | grep -qiE "rate.limit|usage.limit|credit|quota|429|overloaded|capacity"; then
            minutes=$((rate_limit_backoff / 60))
            _log WARN wrapper rate_limit.backoff minutes="$minutes" \
                seconds="$rate_limit_backoff"
            sleep "$rate_limit_backoff"
            rate_limit_backoff=$((rate_limit_backoff * 2))
            [ "$rate_limit_backoff" -gt 18000 ] && rate_limit_backoff=18000
            continue
        fi

        # Successful claude run — reset backoff
        rate_limit_backoff=300

        if [ $claude_exit -ne 0 ] && ! echo "$last_output" | grep -q "RESULT:"; then
            _log ERROR wrapper claude.failed exit="$claude_exit" backoff_s=60
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
                _log WARN wrapper results_tsv.na eval_log="$eval_log"
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
            _log INFO wrapper claude.skip reason="RESULT:skip"
            sleep 5
            continue
        fi

        if [ "$head_before" = "$head_after" ]; then
            _log WARN wrapper claude.no_commit status="$last_status"
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
        # US-505b: also gate on train_classifier.py sha so hyperparam-only
        # edits trigger retrain — closes the silent-skip bug where 5
        # consecutive hyperparam iterations produced identical
        # combined=0.490700 because the gate missed the source of change.
        _features_sha_now=$(git hash-object "$PROJECT_DIR/splice/features.py" 2>/dev/null || echo "")
        _train_sha_now=$(git hash-object "$PROJECT_DIR/splice/classifier/train_classifier.py" 2>/dev/null || echo "")
        read -r _features_sha_trained _train_sha_trained <<<"$(python3 -c "
import json
try:
    d = json.load(open('splice/classifier/fp_classifier.meta.json'))
    print(d.get('features_py_sha') or '_', d.get('train_classifier_py_sha') or '_')
except Exception:
    print('_ _')
" 2>/dev/null)"
        if { [ -n "$_features_sha_now" ] && [ "$_features_sha_now" != "$_features_sha_trained" ]; } \
            || { [ -n "$_train_sha_now" ] && [ "$_train_sha_now" != "$_train_sha_trained" ]; }; then
            _log INFO wrapper retrain.auto.start \
                features_was="$_features_sha_trained" features_now="$_features_sha_now" \
                train_was="$_train_sha_trained" train_now="$_train_sha_now" \
                caller=loop
            # Portable 300s timeout: GNU `timeout` or brew's `gtimeout`
            # when present, else unguarded. macOS ships neither by default.
            if command -v timeout >/dev/null 2>&1; then
                _retrain_cmd=(timeout 300 uv run python splice/classifier/train_classifier.py)
            elif command -v gtimeout >/dev/null 2>&1; then
                _retrain_cmd=(gtimeout 300 uv run python splice/classifier/train_classifier.py)
            else
                _retrain_cmd=(uv run python splice/classifier/train_classifier.py)
            fi
            _phase_start retrain
            set +e
            "${_retrain_cmd[@]}" >>"$CHILD_STDERR_LOG" 2>&1
            _retrain_rc=$?
            set -e
            _phase_end retrain
            # US-518: post-retrain disk checkpoint (before eval starts).
            _check_disk_space soft
            if [ $_retrain_rc -ne 0 ]; then
                _log ERROR wrapper retrain.auto.failed rc="$_retrain_rc" \
                    followup="verify-fail"
                uv run python autoresearch/supervisor_agent.py --diagnose \
                    >>"$CHILD_STDERR_LOG" 2>&1 || true
                _maybe_crash_maintain
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
            git add splice/classifier/fp_classifier.joblib \
                    splice/classifier/fp_classifier.meta.json \
                    >>"$CHILD_STDERR_LOG" 2>&1 || true
            _head_subj=$(git log -1 --format=%s 2>/dev/null)
            case "$_head_subj" in
                hypothesis:*)
                    git commit --amend --no-edit >>"$CHILD_STDERR_LOG" 2>&1 || true
                    ;;
                *)
                    _log WARN wrapper retrain.amend_warning \
                        head_subject="$_head_subj" \
                        action="separate retrain commit"
                    git commit -m "retrain: auto-refresh classifier (features.py=$_features_sha_now, train_classifier.py=$_train_sha_now)" \
                        >>"$CHILD_STDERR_LOG" 2>&1 || true
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
        # supervisor_agent) inherit it.
        _log INFO wrapper eval.start sha="$(git log -1 --format=%h "$hypothesis_commit")"
        # US-504: feature cache env vars. Invalidated automatically when
        # features.py sha changes (new sha → new cache subdir).
        export OMC_FEATURE_CACHE_DIR="$PROJECT_DIR/.omc/feature_cache"
        export OMC_FEATURES_PY_SHA="$(git hash-object "$PROJECT_DIR/splice/features.py" 2>/dev/null || echo unknown)"
        set +e
        _phase_start eval
        # .omc/last_eval.log is evaluate.py's transient stdout capture
        # for the iteration. _append_note greps RESULTS_TSV off it; the
        # wrapper parses `reported` from the same line. Phase 2 keeps the
        # file — it's a per-iteration scratch, not a log — and mirrors
        # the stream into the child-stderr sidecar for the unified audit
        # trail. Carve-out tracked in CLAUDE.md (phase 3a).
        uv run python splice/evaluate.py --shap 2>&1 | tee "$PROJECT_DIR/.omc/last_eval.log" \
            >>"$CHILD_STDERR_LOG"
        eval_exit="${PIPESTATUS[0]}"
        set -e
        _phase_end eval
        # US-518: post-eval disk checkpoint (before note commit / next iteration).
        _check_disk_space soft

        if [ $eval_exit -ne 0 ]; then
            _log ERROR wrapper eval.crash exit="$eval_exit" \
                tail="$(tail -1 "$PROJECT_DIR/.omc/last_eval.log" 2>/dev/null)"
            uv run python autoresearch/supervisor_agent.py --diagnose \
                >>"$CHILD_STDERR_LOG" 2>&1 || true
            _maybe_crash_maintain
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
        # `head -1` truncates upstream grep output; isolate pipefail so a
        # SIGPIPE on the inner grep can't propagate as loop.crash.
        reported=$(
            set +o pipefail
            grep -E "^RESULTS_TSV: " "$PROJECT_DIR/.omc/last_eval.log" | tail -1 | grep -oE "\bcombined=[0-9.]+" | head -1 | cut -d= -f2
        )
        if [ -z "$reported" ]; then
            _log ERROR wrapper eval.parse_fail reason="no_combined_in_results_tsv"
            uv run python autoresearch/supervisor_agent.py --diagnose \
                >>"$CHILD_STDERR_LOG" 2>&1 || true
            _maybe_crash_maintain
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
            _log INFO wrapper discard combined="$reported" prev="$current_best"
            # US-510: catastrophic-floor discard (combined ≤ 0.05) invokes
            # --diagnose so per-domain ERROR lines (e.g. classifier-interface
            # breaks that collapse the aggregate to the GM floor 0.01)
            # reach claude via .omc/last_reflection.md. Normal discards
            # (0.45 < 0.47) are routine research signal; skip diagnose.
            _catastrophic=$(python3 -c "print(1 if float('$reported') <= 0.05 else 0)" 2>/dev/null || echo 0)
            if [ "$_catastrophic" = "1" ]; then
                _log WARN wrapper discard.catastrophic combined="$reported" \
                    floor=0.01 action=diagnose
                uv run python autoresearch/supervisor_agent.py --diagnose \
                    >>"$CHILD_STDERR_LOG" 2>&1 \
                    || _log ERROR pipeline failure "script=supervisor_agent.py --diagnose" rc="$?"
                _maybe_crash_maintain
            fi
            # ralph-monitor breadcrumb instrumentation — tracks which command
            # in the discard handler is the last to complete before any crash.
            # Find the breadcrumb just before loop.crash in autoresearch.jsonl
            # to know where the failing command is. Remove these once the bug
            # is identified and fixed.
            _log INFO wrapper bp.discard step=before_log_to_results_tsv
            log_to_results_tsv "discard" "$hypothesis_commit" "$hypothesis_subject"
            _log INFO wrapper bp.discard step=before_guarded_reset
            _guarded_reset "$head_before"
            _log INFO wrapper bp.discard step=before_append_note
            _phase_start note; _append_note "discard" "$hypothesis_commit" "$hypothesis_subject"; _phase_end note
            _log INFO wrapper bp.discard step=before_phase_skip
            _phase_skip verify
            _phase_end total
            _log INFO wrapper bp.discard step=before_iter_summary
            _iter_summary "$hypothesis_commit" "discard"
            _log INFO wrapper bp.discard step=before_counter_reset
            # US-517: reset crash counter on non-catastrophic discard (combined > 0.05)
            if [ "$_catastrophic" != "1" ]; then
                echo 0 > "$PROJECT_DIR/.omc/supervisor-crash-counter.txt"
            fi
            _log INFO wrapper bp.discard step=before_periodic_maintain
            _maybe_periodic_maintain
            _log INFO wrapper bp.discard step=before_continue
            consecutive_discards=$((consecutive_discards + 1))
            continue
        fi

        # Strict improvement — run supervisor_agent for structural checks.
        _log INFO wrapper verify.start combined="$reported" prev="$current_best"
        _phase_start verify
        set +e
        # US-511: export OMC_HEAD_BEFORE so supervisor_agent's diff audit
        # catches committed protected-file edits (invisible to working-
        # tree / staged diffs after the agent's commit).
        verify_output=$(OMC_HEAD_BEFORE="$head_before" \
            uv run python autoresearch/supervisor_agent.py --verify \
            --agent-name autoresearch \
            --reported-combined "$reported" 2>&1)
        verify_exit=$?
        set -e
        _phase_end verify
        printf '%s\n' "$verify_output" >>"$CHILD_STDERR_LOG"

        if [ $verify_exit -ne 0 ]; then
            log_to_results_tsv "verify-fail" "$hypothesis_commit" "$hypothesis_subject"
            _guarded_reset "$head_before"
            _phase_start note; _append_note "verify-fail" "$hypothesis_commit" "$hypothesis_subject"; _phase_end note
            _phase_end total
            _iter_summary "$hypothesis_commit" "verify-fail"
            consecutive_discards=$((consecutive_discards + 1))
            _log WARN wrapper verify_fail consecutive="$consecutive_discards" \
                cap="$MAX_CONSECUTIVE_DISCARDS"
            continue
        fi

        # VERIFIED KEEP.
        consecutive_discards=0
        _log INFO wrapper keep_path.verified combined="$reported" prev="$current_best"
        log_to_results_tsv "keep" "$hypothesis_commit" "$hypothesis_subject"

        # US-514 phase 1: delegate tag + baseline_metrics.json refresh +
        # baseline commit + versions.json append + shap-shift + note
        # append to _do_keep_path. Commit messages are byte-identical to
        # the pre-phase-1 inline block because --retest-origin is absent
        # in the loop call path.
        _do_keep_path "$hypothesis_commit" "$reported" "$current_best" "$hypothesis_subject"
        # US-518: prune feature_cache after each keep (new classifier sha may have landed).
        _prune_feature_cache_gb

        # US-517: reset crash counter on successful keep
        echo 0 > "$PROJECT_DIR/.omc/supervisor-crash-counter.txt"

        _phase_end total
        _iter_summary "$hypothesis_commit" "keep"
        _maybe_periodic_maintain

        # Brief pause between iterations
        sleep 2
    done

    trap - EXIT HUP INT TERM  # clear trap on clean exit
    _log INFO wrapper loop.ended
}

run_loop_with_restart() {
    # Auto-restart on crash unless stop signal exists
    while true; do
        run_loop
        if [ -f "$STOP_FILE" ]; then
            _log INFO wrapper loop.clean_shutdown reason=stop_signal
            break
        fi
        _log WARN wrapper loop.unexpected_exit backoff_s=60
        sleep 60
    done
}

case "${1:-help}" in
    start)
        _refuse_if_retest_sentinel || exit 1
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
        STATUS_LOG="$PROJECT_DIR/.omc/logs/autoresearch.jsonl"
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "🟢 RUNNING"
        else
            echo "🔴 STOPPED"
            [ -f "$STOP_FILE" ] && echo "⏸  Stop signal pending (will be cleared on next start)"
            # Warn if loop stopped after recent code changes. JSONL records
            # are chronologically appended; pull the last loop.ended event
            # by scanning tail-backwards with python (avoids a full pass).
            last_loop_end=$(python3 - <<PY 2>/dev/null
import json, os
p = os.path.join("$PROJECT_DIR", ".omc/logs/autoresearch.jsonl")
last = ""
try:
    with open(p) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("subsystem") == "wrapper" and rec.get("event") == "loop.ended":
                last = rec.get("ts", "")
except FileNotFoundError:
    pass
print(last[:19])
PY
)
            last_commit=$(git log -1 --format=%ci -- splice/detector.py splice/evaluate.py splice/ml_eval.py run_autoresearch.sh 2>/dev/null | head -c19)
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
            echo "Version: splice/detector-v$latest"
        fi
        if [ -f "$STATUS_LOG" ]; then
            last_log=$(tail -1 "$STATUS_LOG")
            echo "Log: $last_log"
            # Show backoff state if rate limited
            echo "$last_log" | grep -qi "rate_limit\|backoff" && echo "⚠️  Rate limited — backing off"
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
        cp "$SNAP" "$PROJECT_DIR/splice/detector.py"
        if [ -f "$CLAS" ]; then
            cp "$CLAS" "$PROJECT_DIR/splice/classifier/fp_classifier.joblib"
            echo "Restored splice/detector v$N + classifier v$N"
        else
            echo "Restored splice/detector v$N (no classifier for this version)"
        fi
        ;;
    _loop)
        _refuse_if_retest_sentinel || exit 1
        run_loop
        ;;
    _loop_restart)
        _refuse_if_retest_sentinel || exit 1
        run_loop_with_restart
        ;;
    _keep_path)
        # Internal verb for US-514 retest. Args: <hypothesis_sha> <reported_combined> <current_best> <subject> [--retest-origin <sha>]
        shift
        _do_keep_path "$@"
        ;;
    _ensure_classifier_fresh)
        # Internal verb for US-514 retest — standalone classifier staleness gate.
        _do_ensure_classifier_fresh
        ;;
    selftest)
        # US-518: in-process smoke tests for disk-safety guards. No side effects
        # on real feature_cache or loop state. Exits 0 if all pass, 1 otherwise.
        _selftest_pass=0
        _selftest_fail=0

        echo "=== US-518 selftest ==="

        # ---- Test A: feature_cache LRU prune --------------------------------
        # Inline prune logic (mirrors _prune_feature_cache_gb) with a test dir
        # and a small cap so eviction fires without touching the real cache.
        _st_cache="$PROJECT_DIR/.omc/feature_cache_selftest_$$"
        mkdir -p "$_st_cache/sha_old" "$_st_cache/sha_mid" "$_st_cache/sha_cur"
        dd if=/dev/zero of="$_st_cache/sha_old/data" bs=1M count=2 2>/dev/null
        dd if=/dev/zero of="$_st_cache/sha_mid/data" bs=1M count=2 2>/dev/null
        dd if=/dev/zero of="$_st_cache/sha_cur/data" bs=1M count=2 2>/dev/null
        # Make sha_old clearly older via touch (mtime 2h ago).
        touch -t "$(date -v-2H +%Y%m%d%H%M 2>/dev/null || date -d '2 hours ago' +%Y%m%d%H%M 2>/dev/null || echo 202001010000)" \
            "$_st_cache/sha_old" "$_st_cache/sha_old/data" 2>/dev/null || true
        _st_cap=5   # 5 MB cap — the 3 dirs (~6 MB) will exceed it
        _st_cur="sha_cur"
        _st_total=$(du -sm "$_st_cache" 2>/dev/null | awk '{print $1}')
        _st_evicted=0
        if [ "${_st_total:-0}" -gt "$_st_cap" ]; then
            while IFS= read -r _st_d; do
                [ -d "$_st_d" ] || continue
                _st_b="$(basename "$_st_d")"
                [ "$_st_b" = "$_st_cur" ] && continue
                rm -rf "$_st_d"
                _st_evicted=$((_st_evicted + 1))
                _st_total=$(du -sm "$_st_cache" 2>/dev/null | awk '{print $1}')
                [ "${_st_total:-0}" -le "$_st_cap" ] && break
            done < <(
                find "$_st_cache" -mindepth 1 -maxdepth 1 -type d \
                    | xargs stat -f '%m %N' 2>/dev/null \
                    | sort -n | awk '{print $2}' \
                || find "$_st_cache" -mindepth 1 -maxdepth 1 -type d
            )
        fi
        if [ -d "$_st_cache/sha_cur" ] && [ "$_st_evicted" -gt 0 ]; then
            echo "  PASS: Test A — cache prune evicted $_st_evicted dir(s); sha_cur preserved"
            _selftest_pass=$((_selftest_pass + 1))
        else
            echo "  FAIL: Test A — cache prune: evicted=$_st_evicted sha_cur=$([ -d "$_st_cache/sha_cur" ] && echo ok || echo missing)"
            _selftest_fail=$((_selftest_fail + 1))
        fi
        rm -rf "$_st_cache"

        # ---- Test B: /tmp stale eval dir sweep ------------------------------
        _st_tmp_old="/tmp/.ar-eval-selftest-old-$$"
        _st_tmp_new="/tmp/.ar-eval-selftest-new-$$"
        mkdir -p "$_st_tmp_old" "$_st_tmp_new"
        # Make old dir's mtime 3 hours ago.
        touch -t "$(date -v-3H +%Y%m%d%H%M 2>/dev/null || date -d '3 hours ago' +%Y%m%d%H%M 2>/dev/null || echo 202001010000)" \
            "$_st_tmp_old" 2>/dev/null || true
        _cleanup_stale_tmp_eval >/dev/null 2>&1
        if [ ! -d "$_st_tmp_old" ] && [ -d "$_st_tmp_new" ]; then
            echo "  PASS: Test B — stale dir removed, fresh dir preserved"
            _selftest_pass=$((_selftest_pass + 1))
        else
            echo "  FAIL: Test B — old_exists=$([ -d "$_st_tmp_old" ] && echo yes || echo no) new_exists=$([ -d "$_st_tmp_new" ] && echo yes || echo no)"
            _selftest_fail=$((_selftest_fail + 1))
        fi
        rm -rf "$_st_tmp_old" "$_st_tmp_new" 2>/dev/null || true

        # ---- Test C: preflight df gate --------------------------------------
        # Override DISK_MIN_FREE_KB to something astronomically large so the
        # check fires on any real system, then call _check_disk_space in a
        # subshell so exit 1 doesn't kill the selftest.
        _orig_disk_min=$DISK_MIN_FREE_KB
        _st_disk_exit=0
        # set +e so the subshell's exit 1 doesn't abort the selftest.
        set +e
        (DISK_MIN_FREE_KB=$((999 * 1024 * 1024)); _check_disk_space hard) 2>/dev/null
        _st_disk_exit=$?
        set -e
        DISK_MIN_FREE_KB=$_orig_disk_min
        if [ "$_st_disk_exit" -ne 0 ]; then
            echo "  PASS: Test C — preflight df gate fired (exit $_st_disk_exit)"
            _selftest_pass=$((_selftest_pass + 1))
        else
            echo "  FAIL: Test C — preflight df gate did not exit non-zero"
            _selftest_fail=$((_selftest_fail + 1))
        fi

        echo "=== selftest complete: ${_selftest_pass} passed, ${_selftest_fail} failed ==="
        [ "$_selftest_fail" -eq 0 ] && exit 0 || exit 1
        ;;
    *)
        echo "Usage: $0 {start|stop|status|dashboard|rollback <version>|selftest}"
        exit 1
        ;;
esac
