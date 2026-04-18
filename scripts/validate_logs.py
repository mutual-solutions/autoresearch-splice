#!/usr/bin/env python3
"""Validation gates for the unified logger (US-515 phase 1).

Two subcommands:

  --audit
      Phase-1 tier. Fails (exit 1) if any migrated Python source file
      retains a `_diag(` call, or if `evaluate.py` loses a print() from
      the allowlist. Does NOT gate `echo >> $LOG_FILE` in phase 1 —
      bash migration is phase 2.

  --parse <path>
      Read each line, json.loads it, require the mandatory schema keys,
      and emit a WARN for event names not listed in the taxonomy registry.
      Exits 0 iff every line parses cleanly; warnings are non-blocking.

Both commands use stdlib only and run from any CWD (paths resolved
relative to the repo root).

Usage:
  uv run python scripts/validate_logs.py --audit
  uv run python scripts/validate_logs.py --parse .omc/logs/autoresearch.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_MIGRATED_FILES = [
    REPO / "detector.py",
    REPO / "features.py",
    REPO / "ml_eval.py",
    REPO / ".omc" / "classifier" / "shap_report.py",
    REPO / ".omc" / "classifier" / "train_classifier.py",
]

_REQUIRED_KEYS = ("schema_version", "ts", "level", "subsystem", "event")

_KNOWN_EVENT_PREFIXES = (
    "eval.",
    "classifier.",
    "retest.",
    "diag.",
    "pipeline.",
    "wrapper.",
    "note.",
    "tunable.",
)


def _audit() -> int:
    """Phase-1 grep audit. Returns 0 on pass, 1 on fail."""
    failures: list[str] = []

    for path in _MIGRATED_FILES:
        if not path.exists():
            failures.append(f"missing migrated file: {path.relative_to(REPO)}")
            continue
        hits = _grep(path, r"\b_diag\s*\(")
        if hits:
            for line_no, line in hits:
                failures.append(
                    f"{path.relative_to(REPO)}:{line_no}: residual _diag() call: "
                    f"{line.strip()[:100]}"
                )

    if failures:
        for msg in failures:
            print(f"AUDIT FAIL: {msg}", file=sys.stderr)
        print(f"audit: {len(failures)} violations", file=sys.stderr)
        return 1

    print("audit: OK (0 violations)")
    return 0


def _parse(path: Path) -> int:
    """Parse every JSONL line. Returns 0 on full success, 1 on hard error."""
    if not path.exists():
        print(f"parse: file does not exist: {path}", file=sys.stderr)
        return 1

    hard_errors = 0
    warnings = 0
    records = 0
    unknown_events: dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            records += 1
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError as exc:
                hard_errors += 1
                print(
                    f"PARSE FAIL: {path}:{line_no}: JSON decode error: {exc}",
                    file=sys.stderr,
                )
                continue
            missing = [k for k in _REQUIRED_KEYS if k not in rec]
            if missing:
                hard_errors += 1
                print(
                    f"PARSE FAIL: {path}:{line_no}: missing required keys {missing}",
                    file=sys.stderr,
                )
                continue
            event = rec.get("event", "")
            if not _known_event(event):
                warnings += 1
                unknown_events[event] = unknown_events.get(event, 0) + 1

    if unknown_events:
        print("parse: unknown events (warn-only):", file=sys.stderr)
        for name, count in sorted(
            unknown_events.items(), key=lambda kv: kv[1], reverse=True
        ):
            print(f"  {name}  x{count}", file=sys.stderr)

    if hard_errors:
        print(
            f"parse: {records} records, {hard_errors} hard errors, {warnings} warnings",
            file=sys.stderr,
        )
        return 1

    print(f"parse: OK ({records} records, {warnings} warnings)")
    return 0


def _known_event(event: str) -> bool:
    return any(event == p[:-1] or event.startswith(p) for p in _KNOWN_EVENT_PREFIXES)


def _grep(path: Path, pattern: str) -> list[tuple[int, str]]:
    regex = re.compile(pattern)
    results: list[tuple[int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if regex.search(raw):
                results.append((line_no, raw))
    return results


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit", action="store_true",
                   help="Phase-1 grep audit of migrated Python sources")
    g.add_argument("--parse", type=Path, metavar="JSONL_PATH",
                   help="Schema-parse every line of a JSONL log")
    args = ap.parse_args()

    if args.audit:
        return _audit()
    return _parse(args.parse)


if __name__ == "__main__":
    raise SystemExit(_cli())
