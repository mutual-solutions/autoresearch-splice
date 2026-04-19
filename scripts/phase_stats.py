#!/usr/bin/env python3
"""Per-phase iteration timing stats for the autoresearch loop (US-509).

Consumes `wrapper.iteration.phase` events via `scripts.log_reader.iter_events()`
from `.omc/logs/autoresearch.jsonl`. Phase 2 dropped the legacy-shim
synthesizer; the wrapper emits the event natively via `_iter_summary()`.

Phases tracked: claude (hypothesis formation), retrain, eval, verify
(verify_agent structural checks), note (_append_note), total.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from log_reader import iter_events  # noqa: E402

PHASES = ["total", "claude", "retrain", "eval", "verify", "note"]


def _parse(n: int, jsonl_path: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    for rec in iter_events(
        subsystem="wrapper", event="iteration.phase", jsonl_path=jsonl_path
    ):
        rows.append({k: rec.get(k, "-") for k in
                     ("iter", "status", "total", "claude", "retrain", "eval",
                      "verify", "note")})
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
    ap.add_argument("--log", type=Path, default=None,
                    help="JSONL log path override (test hook; default "
                         "resolves to .omc/logs/autoresearch.jsonl)")
    args = ap.parse_args()

    rows = _parse(args.window, jsonl_path=args.log)

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
