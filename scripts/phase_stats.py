#!/usr/bin/env python3
"""Per-phase iteration timing stats for the autoresearch loop (US-509).

Parses `.omc/autoresearch.log` for `ITER_SUMMARY` lines emitted by
run_autoresearch.sh and prints a compact table of median + p95 per phase
across the trailing N iterations.

Phases tracked: claude (hypothesis formation), retrain, eval, verify
(verify_agent structural checks), note (_append_note), total.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / ".omc" / "autoresearch.log"

PHASES = ["total", "claude", "retrain", "eval", "verify", "note"]
ITER_RE = re.compile(
    r"ITER_SUMMARY\s+iter=(\S+)\s+status=(\S+)\s+"
    r"total=(\S+)\s+claude=(\S+)\s+retrain=(\S+)\s+"
    r"eval=(\S+)\s+verify=(\S+)\s+note=(\S+)"
)


def _parse(n: int) -> list[dict]:
    if not LOG.exists():
        return []
    rows: list[dict] = []
    with open(LOG) as f:
        for line in f:
            m = ITER_RE.search(line)
            if not m:
                continue
            iter_sha, status, total, claude, retrain, eval_, verify, note = m.groups()
            rows.append({
                "iter": iter_sha,
                "status": status,
                "total": total,
                "claude": claude,
                "retrain": retrain,
                "eval": eval_,
                "verify": verify,
                "note": note,
            })
    return rows[-n:]


def _nums(rows: list[dict], key: str) -> list[int]:
    out: list[int] = []
    for r in rows:
        v = r[key]
        if v in ("-", "?", ""):
            continue
        try:
            out.append(int(v))
        except ValueError:
            continue
    return out


def _median(xs: list[int]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _p95(xs: list[int]) -> int | None:
    if not xs:
        return None
    s = sorted(xs)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=20,
                    help="Trailing iterations to evaluate (default: 20)")
    args = ap.parse_args()

    rows = _parse(args.window)

    if not rows:
        print("(no iteration summaries yet)")
        return 0

    status_counts = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    print(f"phase_stats: trailing {len(rows)} iterations "
          f"({', '.join(f'{k}={v}' for k,v in sorted(status_counts.items()))})")
    print(f"{'phase':<10}{'n':>5}{'median':>10}{'p95':>10}")
    for phase in PHASES:
        nums = _nums(rows, phase)
        if not nums:
            print(f"  {phase:<8}{'-':>5}{'-':>10}{'-':>10}")
            continue
        med = _median(nums)
        p95 = _p95(nums)
        print(f"  {phase:<8}{len(nums):>5}{med:>9}s{p95:>9}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
