#!/usr/bin/env python3
"""One-shot operator dashboard for the autoresearch loop (PRD US-502).

Pure read-side: consults results.tsv, baseline_metrics.json,
shap_rollup.json, versions.json, and `git log` for the last-keep
timestamp. No evaluate.py, no expensive subprocesses. The shell
wrapper passes the tmux state as --state so we don't shell out a
second time for `tmux has-session`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results.tsv"
BASELINE = REPO / ".omc" / "coordination" / "baseline_metrics.json"
SHAP_ROLLUP = REPO / ".omc" / "shap_rollup.json"
VERSIONS = REPO / ".omc" / "classifier" / "versions.json"

# US-515: recent-events augmentation via the unified JSONL log. Phase 2
# drops the legacy shim so this reads `.omc/logs/autoresearch.jsonl`
# exclusively. The dashboard's primary surface (results.tsv + baseline +
# shap_rollup + versions + git log) is unchanged; iter_events() layers on.
sys.path.insert(0, str(REPO / "scripts"))
from log_reader import iter_events  # noqa: E402


def _recent_pipeline_failures(limit: int = 3) -> list[dict]:
    """Return the last `limit` pipeline.failure events from the unified log."""
    events = list(iter_events(event="pipeline.failure"))
    return events[-limit:] if events else []


def _pipeline_failure_block() -> list[str]:
    fails = _recent_pipeline_failures(limit=3)
    if not fails:
        return []
    lines = [f"recent pipeline failures ({len(fails)}):"]
    for rec in fails:
        ts = rec.get("ts", "?")[:19]
        detail = rec.get("detail") or rec.get("script") or rec.get("event", "?")
        rc = rec.get("rc")
        if rc is not None and "rc=" not in str(detail):
            detail = f"{detail} (rc={rc})"
        lines.append(f"  [{ts}] {detail}")
    return lines


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _experiments_summary() -> tuple[str, str | None]:
    """Return (summary line, last_keep_sha | None)."""
    if not RESULTS.exists():
        return "(no experiments yet)", None
    total = keep = disc = vfail = 0
    last_keep_sha: str | None = None
    with open(RESULTS) as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13:
                continue
            total += 1
            status = parts[12]
            if status == "keep":
                keep += 1
                if parts[0] and parts[0] != "NA":
                    last_keep_sha = parts[0]
            elif status == "discard":
                disc += 1
            elif status == "verify-fail":
                vfail += 1
    return (f"experiments: {total} total — keep {keep} / discard {disc} / verify-fail {vfail}",
            last_keep_sha)


def _last_keep_age(sha: str | None) -> str:
    if not sha:
        return ""
    short = sha[:7]
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%cr", sha],
            cwd=REPO, text=True, stderr=subprocess.DEVNULL, timeout=2,
        ).strip()
        return f" (last keep {out}, commit {short})"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return f" (last keep commit {short})"


def _baseline_block() -> list[str]:
    b = _read_json(BASELINE)
    if not b:
        return ["baseline: (no baseline_metrics.json yet)"]
    combined = b.get("combined")
    per = b.get("per_dataset_combined") or {}
    fp = b.get("per_dataset_clean_fp") or {}
    lines = []
    combined_s = f"{combined:.6f}" if isinstance(combined, (int, float)) else "?"
    lines.append(f"baseline combined: {combined_s}")
    if per:
        lines.append("per-domain:")
        for ds in sorted(per):
            v = per[ds]
            v_s = f"{v:.6f}" if isinstance(v, (int, float)) else "?"
            lines.append(f"  {ds:<8}  combined={v_s}  clean_fp={fp.get(ds, '?')}")
    return lines


def _rollup_block() -> list[str]:
    r = _read_json(SHAP_ROLLUP)
    per = r.get("per_domain") or {}
    if not per:
        return ["top features: (no rollup yet)"]
    lines = [f"top features (last {len(r.get('keeps_considered', []))} keeps):"]
    for dom in sorted(per):
        top = per[dom][:3]
        names = ", ".join(f"{row['name']}({row['sum_abs_shap']:.1f})" for row in top)
        lines.append(f"  {dom:<8}  {names}")
    return lines


def _version_line() -> str | None:
    v = _read_json(VERSIONS)
    latest = v.get("latest")
    return f"detector version: detector-v{latest}" if latest else None


def _eval_tmp_line() -> str | None:
    """Best-effort report on the currently-decrypted eval tree.

    The wrapper's own tmp path isn't exported beyond its own shell, so
    scan /var/folders for recent .ar-eval-* dirs (the mkdtemp prefix in
    scripts/eval_crypto.py). Report size only; not the path — printing
    it would defeat the isolation property.
    """
    tmpdir_parent = Path(os.environ.get("TMPDIR", "/tmp"))
    try:
        cands = [p for p in tmpdir_parent.iterdir()
                 if p.is_dir() and p.name.startswith(".ar-eval-")]
    except OSError:
        return None
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    most_recent = cands[0]
    try:
        size_mb = sum(
            f.stat().st_size
            for f in most_recent.rglob("*") if f.is_file()
        ) / (1024 * 1024)
    except OSError:
        return None
    return f"decrypted eval tmp: {size_mb:.0f}MB (wrapper-owned, path withheld)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default="STOPPED",
                    help="Loop state passed in by the shell (RUNNING / STOPPED / ...)")
    args = ap.parse_args()

    lines: list[str] = []
    lines.append("=== autoresearch dashboard ===")
    lines.append(f"state: {args.state}")

    exp_summary, last_sha = _experiments_summary()
    lines.append(exp_summary + _last_keep_age(last_sha))
    lines.extend(_baseline_block())
    lines.extend(_rollup_block())
    v = _version_line()
    if v:
        lines.append(v)
    tmp = _eval_tmp_line()
    if tmp:
        lines.append(tmp)
    lines.extend(_pipeline_failure_block())
    lines.append(f"generated_at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
