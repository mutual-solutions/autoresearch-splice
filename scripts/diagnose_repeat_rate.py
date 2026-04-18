#!/usr/bin/env python3
"""Repeat-rate metric for the US-508 Phase 1 ship/no-ship decision.

Reads results.tsv + .omc/research_notes.md. For verify-fail rows in the
trailing `--window` iterations, extract (exception_class, top_frame) from
the matching research-notes entry's `[auto-diagnosis]` block and count the
fraction whose (exception_class, top_frame) pair matches another verify-fail
within `--proximity` iterations.

Repeat rate definition:
    repeat_rate = repeated_failures / total_verify_fails_in_window

Phase 2 trigger:
    If post-ship trailing-20 repeat_rate has NOT dropped ≥50% from baseline,
    Phase 2 (session-resume + fix-on-top) is authorized.

Pre-ship baseline is captured before shipping Phase 1; post-ship value
is what this script measures after 20 iterations have run against the
patched wrapper.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results.tsv"
NOTES = REPO / ".omc" / "research_notes.md"


def _recent_verify_fails(window: int) -> list[tuple[int, str, str]]:
    """Return list of (iter_index, short_sha, description) for verify-fail rows
    in the trailing `window` iterations. iter_index is the row's 1-based position
    from the start of results.tsv (so proximity comparisons are chronologically
    meaningful).
    """
    if not RESULTS.exists():
        return []
    rows: list[tuple[int, str, str]] = []
    with open(RESULTS) as f:
        header = f.readline()
        if not header:
            return []
        cols = header.rstrip("\n").split("\t")
        try:
            sha_idx = cols.index("commit")
            status_idx = cols.index("status")
            desc_idx = cols.index("description")
        except ValueError:
            return []
        all_rows: list[tuple[int, str, str, str]] = []
        for i, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(sha_idx, status_idx, desc_idx):
                continue
            all_rows.append((i, parts[sha_idx], parts[status_idx], parts[desc_idx]))
        # Trailing window over ALL rows, then filter to verify-fail.
        tail = all_rows[-window:]
        for i, sha, status, desc in tail:
            if status == "verify-fail":
                rows.append((i, sha, desc))
    return rows


def _parse_diagnosis_blocks(text: str) -> list[tuple[str, str]]:
    """Return [(exception_class, top_frame)] extracted from every
    `[auto-diagnosis]` block in the research_notes text, in chronological
    order of appearance."""
    out: list[tuple[str, str]] = []
    # [auto-diagnosis] block pattern (open-ended until next --- or end).
    for m in re.finditer(
        r"\[auto-diagnosis\]\s*\n(.*?)(?=\n---\n|\n## |\Z)",
        text, re.DOTALL,
    ):
        block = m.group(1)
        exc = None
        top_frame = None
        exc_m = re.search(r"exception:\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", block)
        if exc_m:
            exc = exc_m.group(1)
        # top_frames: next non-blank line after the header.
        tf_m = re.search(r"top_frames:\s*\n\s+(\S.*?)(?:\n|$)", block)
        if tf_m:
            top_frame = tf_m.group(1).strip()
        if exc and top_frame:
            out.append((exc, top_frame))
    return out


def compute_repeat_rate(window: int, proximity: int) -> dict:
    verify_fails = _recent_verify_fails(window)
    if not verify_fails:
        return {
            "repeat_rate": None,
            "total_verify_fails": 0,
            "repeats": 0,
            "breakdown": {},
            "reason": "no verify-fails in trailing window",
        }

    notes_text = NOTES.read_text() if NOTES.exists() else ""
    diagnoses = _parse_diagnosis_blocks(notes_text)

    # Align verify-fails with diagnosis blocks. Research notes are
    # chronological; trailing N verify-fails → last N diagnoses (best effort).
    aligned: list[tuple[int, str, str]] = []  # (iter_index, exc_class, top_frame)
    n = min(len(verify_fails), len(diagnoses))
    for (iter_idx, _sha, _desc), (exc, frame) in zip(verify_fails[-n:], diagnoses[-n:]):
        aligned.append((iter_idx, exc, frame))

    # Count repeats: fingerprint (exc, frame) matches another aligned
    # verify-fail within `proximity` iterations.
    repeats = 0
    key_offenders: Counter[tuple[str, str]] = Counter()
    for i, (idx_a, exc_a, frame_a) in enumerate(aligned):
        for idx_b, exc_b, frame_b in aligned[:i]:
            if exc_a == exc_b and frame_a == frame_b and abs(idx_a - idx_b) <= proximity:
                repeats += 1
                key_offenders[(exc_a, frame_a)] += 1
                break  # count each verify-fail at most once

    total = len(aligned)
    rate = repeats / total if total else None
    return {
        "repeat_rate": rate,
        "total_verify_fails": len(verify_fails),
        "aligned_with_diagnosis": total,
        "repeats": repeats,
        "breakdown": dict(key_offenders),
        "window": window,
        "proximity": proximity,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=20,
                    help="Trailing iterations to evaluate (default: 20)")
    ap.add_argument("--proximity", type=int, default=5,
                    help="Max distance in iterations to count as a repeat (default: 5)")
    args = ap.parse_args()

    r = compute_repeat_rate(args.window, args.proximity)

    if r["repeat_rate"] is None:
        print(f"repeat_rate: N/A ({r.get('reason', 'no aligned verify-fails')})")
        print(f"total_verify_fails: {r['total_verify_fails']}")
        return 0

    print(f"repeat_rate: {r['repeat_rate']:.3f}")
    print(f"window: {r['window']}  proximity: {r['proximity']}")
    print(f"total_verify_fails: {r['total_verify_fails']}  "
          f"aligned_with_diagnosis: {r['aligned_with_diagnosis']}  "
          f"repeats: {r['repeats']}")
    if r["breakdown"]:
        print("repeat offenders:")
        for (exc, frame), n in sorted(r["breakdown"].items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {n}x  {exc} @ {frame}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
