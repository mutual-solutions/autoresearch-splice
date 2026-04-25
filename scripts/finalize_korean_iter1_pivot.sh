#!/usr/bin/env bash
# Finalize the korean-iter1 pivot once the external drive is reconnected.
#
# Prereqs:
#   1. /Volumes/HIKSEMI/ is mounted with korean-iter-1-delivery.tar present.
#   2. autoresearch-splice repo at HEAD = df62969 (or later) on
#      branch autoresearch/korean-iter1.
#   3. Sentinel .omc/korean-iter1-regen-in-progress exists.
#
# Sequence (all 6 steps run unattended; the operator should review the
# pre-flight banner and confirm before each phase):
#   A. Pre-flight checks
#   B. Extract tarball to /Volumes/HIKSEMI/korean-iter-1-extracted/  (~5 min)
#   C. Regenerate corpus from extracted dir (workers=8, ~12 min)
#   D. Verify determinism (10 sample byte-diff)
#   E. Update autoresearch/manifest.json with real ground_truth_sha256_prefix
#      + expected file counts; commit as MIGRATE-PROTECTED
#   F. Train 3-class classifier (~10 min)
#   G. Smoke iter via splice/evaluate.py + atomic baseline_metrics.json write
#      + supervisor --verify gate
#   H. Remove sentinel on happy path; report final state
#
# Operator next step after this script succeeds:
#   Remove .omc/autoresearch-stop, then ./run_autoresearch.sh start

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$PWD"
TARBALL="/Volumes/HIKSEMI/korean-iter-1-delivery.tar"
EXTRACT_DIR="/Volumes/HIKSEMI/korean-iter-1-delivery"
OUTPUT_ROOT="$REPO_ROOT/data/eval/korean_iter1"
SENTINEL="$REPO_ROOT/.omc/korean-iter1-regen-in-progress"
# NOTE: use `env KEY=VAL ...` not bare `KEY=VAL ...` so that variable
# expansion of $PYTHON_RUN word-splits cleanly under bash without `eval`.
# The bare-KEY=VAL form requires bash to recognize the prefix at parse
# time, but variable expansion happens AFTER parse, so without eval the
# entire expansion is treated as one command name and fails. With `env`,
# the leading word is just `env` (a real command) which handles the env-
# var assignment itself — no eval, no multi-line argument re-tokenization.
PYTHON_RUN="env PYTHONPATH=$REPO_ROOT uv run python"

red() { printf "\033[31m%s\033[0m\n" "$*"; }
grn() { printf "\033[32m%s\033[0m\n" "$*"; }
ylw() { printf "\033[33m%s\033[0m\n" "$*"; }
banner() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }

# ------- A. PRE-FLIGHT --------
banner "A. Pre-flight"

[[ -f "$TARBALL" ]] || { red "FAIL: $TARBALL not found. Reconnect HIKSEMI drive."; exit 1; }
[[ -f "$SENTINEL" ]] || { red "FAIL: sentinel $SENTINEL missing. Are you in the right tree state?"; exit 1; }
[[ "$(git rev-parse --abbrev-ref HEAD)" == "autoresearch/korean-iter1" ]] || {
  red "FAIL: not on autoresearch/korean-iter1 branch."; exit 1; }
git tag -l iter1-anchor | grep -q iter1-anchor || {
  red "FAIL: iter1-anchor tag missing."; exit 1; }

grn "Pre-flight OK: tarball present, sentinel up, branch + tag correct."

# ------- B. EXTRACT --------
if [[ -d "$EXTRACT_DIR" ]] && [[ -f "$EXTRACT_DIR/manifest.json" ]]; then
  ylw "B. SKIP extraction: $EXTRACT_DIR already exists with manifest."
else
  banner "B. Extract tarball (sequential read, ~5 min on USB)"
  rm -rf "$EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR"
  tar -xf "$TARBALL" -C "$EXTRACT_DIR/"
  N=$(find "$EXTRACT_DIR" -type f | wc -l)
  grn "Extracted $N files to $EXTRACT_DIR"
fi

# ------- C. REGENERATE --------
banner "C. Regenerate corpus (workers=8, ~12 min)"
rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
$PYTHON_RUN scripts/regenerate_korean_iter1.py --regenerate \
  --workers 8 --tarball "$EXTRACT_DIR" --output-root "$OUTPUT_ROOT" \
  2>&1 | tee .omc/regen-fulllog.txt | tail -3

NTOTAL=$(find "$OUTPUT_ROOT" -name '*.opus' | wc -l)
NTRAIN=$(find "$OUTPUT_ROOT/train" -name '*.opus' 2>/dev/null | wc -l)
NEVAL=$(find "$OUTPUT_ROOT/eval" -name '*.opus' 2>/dev/null | wc -l)
NTEST=$(find "$OUTPUT_ROOT/test" -name '*.opus' 2>/dev/null | wc -l)
grn "Regen wrote $NTOTAL .opus files (train=$NTRAIN, eval=$NEVAL, test=$NTEST)"

[[ -f "$OUTPUT_ROOT/eval/ground_truth.json" ]] || { red "FAIL: eval/ground_truth.json missing."; exit 1; }

# ------- D. VERIFY DETERMINISM --------
banner "D. Verify determinism (10 random samples byte-diff)"
$PYTHON_RUN scripts/regenerate_korean_iter1.py --verify-determinism \
  --tarball "$EXTRACT_DIR" --output-root "$OUTPUT_ROOT" --sample 10 \
  2>&1 | tee -a .omc/regen-fulllog.txt | tail -3
grn "Determinism receipt at .omc/korean-iter1-determinism-receipt.json"

# ------- E. UPDATE MANIFEST SHA --------
banner "E. Update autoresearch/manifest.json with real ground_truth SHA + counts"
GT_SHA=$($PYTHON_RUN -c "
import hashlib, sys
print(hashlib.sha256(open('$OUTPUT_ROOT/eval/ground_truth.json','rb').read()).hexdigest()[:12])
")
grn "ground_truth.json SHA-256 prefix = $GT_SHA"

$PYTHON_RUN -c "
import json
m = json.load(open('autoresearch/manifest.json'))
m['ground_truth_sha256_prefix'] = '$GT_SHA'
m['expected_counts'] = {'_train_files': $NTRAIN, '_eval_files': $NEVAL, '_test_files': $NTEST}
open('autoresearch/manifest.json','w').write(json.dumps(m, indent=2) + '\n')
"
git add autoresearch/manifest.json
# --allow-empty so re-runs against an already-correct manifest don't trip set -e.
git commit --allow-empty -m "MIGRATE-PROTECTED autoresearch/manifest.json: korean-iter1 real GT SHA + counts (post-regen)"

# ------- F. TRAIN CLASSIFIER --------
banner "F. Train 3-class GBM classifier (~10 min)"
$PYTHON_RUN splice/classifier/train_classifier.py 2>&1 | tee .omc/train-classifier.log | tail -10
[[ -f splice/classifier/fp_classifier.joblib ]] || { red "FAIL: classifier bundle not produced."; exit 1; }
grn "Classifier bundle written: $(ls -la splice/classifier/fp_classifier.joblib)"
git add splice/classifier/fp_classifier.joblib splice/classifier/fp_classifier.meta.json
git commit -m "korean-iter1: train iter-0 3-class classifier (cross_voice / no_splice / same_voice_edit)"

# ------- G. SMOKE ITERATION + BASELINE WRITE + VERIFY --------
banner "G. Smoke iteration + atomic baseline write + supervisor verify"
$PYTHON_RUN splice/evaluate.py 2>&1 | tee .omc/last_eval.log | tail -3
COMBINED=$(grep '^combined:' .omc/last_eval.log | tail -1 | awk '{print $2}')
grn "Smoke iter combined = $COMBINED"

$PYTHON_RUN -c "
import json, tempfile, os, subprocess, datetime, hashlib
from pathlib import Path

# Parse RESULTS_TSV from .omc/last_eval.log
log_text = Path('.omc/last_eval.log').read_text()
tsv_line = next(l for l in log_text.splitlines() if l.startswith('RESULTS_TSV:'))
fields = dict(t.split('=', 1) for t in tsv_line.replace('RESULTS_TSV:', '').strip().split())

git_sha = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
det_sha = ''
det_path = Path('.omc/korean-iter1-determinism-receipt.json')
if det_path.exists():
    det_sha = hashlib.sha256(det_path.read_bytes()).hexdigest()[:12]

baseline = {
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    'git_sha': git_sha,
    'phase': 'korean_iter1_pivot_initial',
    'combined': float(fields.get('combined', 0)),
    'precision': float(fields.get('precision', 0)),
    'recall': float(fields.get('recall', 0)),
    'n_files': int(fields.get('n_files', 0)),
    'collar_ms': 250,
    'splice_class_breakdown': {
        'cross_voice_f1': float(fields.get('cross_voice_f1', 0)),
        'same_voice_edit_f1': float(fields.get('same_voice_edit_f1', 0)),
    },
    'voices_holdout': {
        'test': ['DaeBuHo', 'Kanna'],
        'eval': ['Sunwoo', 'Joon'],
        'train': ['ChloeCha', 'DangchanYeo', 'Donghyun', 'Eunha', 'Minho', 'Minwoo', 'Ondo'],
    },
    'regen_determinism_receipt_sha': det_sha,
    'unknown_label_count': int(fields.get('unknown_label_count', 0)),
}

# Atomic write via tempfile + os.replace
with tempfile.NamedTemporaryFile('w', dir='autoresearch', delete=False) as tf:
    tf.write(json.dumps(baseline, indent=2) + '\n')
    tmppath = tf.name
os.replace(tmppath, 'autoresearch/baseline_metrics.json')
print('baseline_metrics.json written atomically:', baseline['combined'])
"
git add autoresearch/baseline_metrics.json
git commit -m "MIGRATE-PROTECTED autoresearch/baseline_metrics.json: korean-iter1 iter-0 baseline (atomic from real eval)"

# Re-tag iter1-anchor at the FINAL maintainer commit (the baseline_metrics.json
# write). Per plan Item 6, this tag must equal HEAD just before run_autoresearch.sh
# start so a first-iteration discard's _guarded_reset target is the iter-0
# baseline, not an earlier maintainer commit.
git tag -d iter1-anchor 2>/dev/null || true
git tag -a iter1-anchor HEAD -m "Safe first-iteration discard target — iter-0 baseline (post-finalize)"
grn "iter1-anchor re-tagged at HEAD ($(git rev-parse iter1-anchor | cut -c1-12))"

# Supervisor verify
$PYTHON_RUN autoresearch/supervisor_agent.py --verify \
  --agent-name maintainer-pivot --reported-combined "$COMBINED" \
  || { red "FAIL: supervisor --verify failed."; exit 1; }
grn "Supervisor verify PASSED."

# ------- H. CLEANUP --------
banner "H. Cleanup"

rm -f "$SENTINEL"
grn "Sentinel removed."

ylw "Final commit count on korean-iter1:"
git log --oneline | head -15

echo ""
grn "================================================"
grn "korean-iter1 pivot finalized successfully."
grn "================================================"
echo ""
echo "Operator next steps:"
echo "  1. Verify .omc/korean-iter1-regen-in-progress is gone: ls $SENTINEL"
echo "  2. Remove autoresearch-stop sentinel: rm -f .omc/autoresearch-stop"
echo "  3. Start the loop: ./run_autoresearch.sh start"
echo ""
echo "First-iteration baseline_metrics.json combined = $COMBINED"
echo "iter1-anchor tag = $(git rev-parse iter1-anchor | cut -c1-12)"
echo "HEAD            = $(git rev-parse HEAD | cut -c1-12)"
