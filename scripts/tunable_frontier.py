#!/usr/bin/env python3
"""Per-tunable exploration frontier for the autoresearch prompt (PRD US-506).

Parses `hypothesis: <TUNABLE> <old>-><new> ...` subjects from results.tsv
joined with their keep/discard status, and emits per-tunable summary
lines so claude can see which values have been tried on each axis.
Supported tunables: GBM_THRESHOLD, GBM_MIN_SEP_S, ANALYSIS_STRIDE_S.

Output format (one line per tunable):
  GBM_THRESHOLD: 7 tried (kept: 0.983, 0.982; failed: 0.980, 0.981, 0.98)
                 range [0.978, 0.985]  current 0.983
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_TSV = REPO / "results.tsv"
DETECTOR = REPO / "detector.py"
DEFAULT_OUTPUT = REPO / ".omc" / "tunable_frontier.txt"

TUNABLES = ["GBM_THRESHOLD", "GBM_MIN_SEP_S", "ANALYSIS_STRIDE_S"]


def _current_value(name: str) -> str | None:
    try:
        src = DETECTOR.read_text()
    except OSError:
        return None
    m = re.search(rf"^{re.escape(name)}\s*=\s*([0-9.]+)", src, re.MULTILINE)
    return m.group(1) if m else None


def _parse_subject(subject: str) -> list[tuple[str, str]]:
    """Return every (tunable, new_value) occurrence in the subject string.

    Handles both `hypothesis: NAME 0.98->0.99 ...` and compound subjects
    like `hypothesis: NAME_A 1->2 and NAME_B 3->4 ...`.
    """
    pairs: list[tuple[str, str]] = []
    for name in TUNABLES:
        for m in re.finditer(
            rf"\b{re.escape(name)}\s+[0-9.]+\s*->\s*([0-9.]+)", subject,
        ):
            pairs.append((name, m.group(1)))
    return pairs


def build_frontier() -> dict[str, dict]:
    """Aggregate all (name, new_value, status) triples from results.tsv."""
    per: dict[str, dict[str, set | list]] = {
        name: {"kept": set(), "failed": set(), "values": []}
        for name in TUNABLES
    }
    if not RESULTS_TSV.exists():
        return per
    with open(RESULTS_TSV) as f:
        header = f.readline()
        if not header:
            return per
        cols = header.rstrip("\n").split("\t")
        try:
            status_idx = cols.index("status")
            desc_idx = cols.index("description")
        except ValueError:
            return per
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(status_idx, desc_idx):
                continue
            status = parts[status_idx]
            if status not in ("keep", "discard", "verify-fail"):
                continue
            subj = parts[desc_idx]
            for name, new_val in _parse_subject(subj):
                bucket = "kept" if status == "keep" else "failed"
                per[name][bucket].add(new_val)
                per[name]["values"].append(new_val)
    return per


def format_frontier(per: dict[str, dict]) -> list[str]:
    if not any(per[name]["values"] for name in TUNABLES):
        return ["  (no tunable history yet)"]

    lines: list[str] = []
    for name in TUNABLES:
        b = per[name]
        if not b["values"]:
            cur = _current_value(name) or "?"
            lines.append(f"  {name}: 0 tried  current {cur}")
            continue
        kept = sorted(b["kept"], key=lambda v: float(v))
        failed = sorted(b["failed"], key=lambda v: float(v))
        all_vals = sorted({*b["kept"], *b["failed"]}, key=lambda v: float(v))
        rng = f"[{all_vals[0]}, {all_vals[-1]}]" if all_vals else "-"
        cur = _current_value(name) or "?"
        kept_s = ", ".join(kept) if kept else "-"
        failed_s = ", ".join(failed) if failed else "-"
        lines.append(
            f"  {name}: {len(b['values'])} tried "
            f"(kept: {kept_s}; failed: {failed_s}) "
            f"range {rng}  current {cur}"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Output path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--stdout", action="store_true",
                    help="Print to stdout instead of writing to --output")
    args = ap.parse_args()

    lines = format_frontier(build_frontier())
    blob = "\n".join(lines)
    if args.stdout:
        print(blob)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(blob + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
