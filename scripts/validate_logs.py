#!/usr/bin/env python3
"""Validation gates for the unified logger (US-515 phases 1 + 2).

Two subcommands:

  --audit
      Grep audit. Phase-1 tier: fails if any migrated Python source
      retains a `_diag(` call. Phase-2 tier: fails if `run_autoresearch.sh`
      contains any `echo ... >> $LOG_FILE` site (the wrapper must emit
      via `_log` exclusively). Reports both tiers in one pass.

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

# Structured Python emitters embed their namespace in the event name
# (e.g. `eval.metrics.splice`, `diag.gbm.dedupe`). These prefixes recognize
# the phase-1 Python taxonomy.
_KNOWN_EVENT_PREFIXES = (
    "eval.",
    "classifier.",
    "retest.",
    "diag.",
    "pipeline.",
    "wrapper.",
    "note.",
    "tunable.",
    "notebook.",
)

# Phase-2 bash emitters and phase-1 legacy synthesizers use bare event
# names scoped by subsystem (e.g. subsystem=`wrapper` event=`iteration.phase`).
# Any event under these subsystems is taxonomy-clean.
_KNOWN_SUBSYSTEMS = frozenset(
    {
        "wrapper",
        "pipeline",
        "notebook.digest",
        "tunable.frontier",
    }
)


_WRAPPER = REPO / "run_autoresearch.sh"

# Phase-2 bash audit: the wrapper must not retain any legacy echo-to-LOG_FILE
# site. The regex matches `echo ... >> $LOG_FILE` with or without quoting,
# and allows any intervening text so multi-line / trailing-pipe variants
# still fail the gate. A narrow false-positive: the same token inside a
# here-doc would trip the audit; none exists today, and a future one is
# acceptable churn for the simplicity of a single grep.
_BASH_LEGACY_RE = r'^[^#]*\becho\b.*>>\s*"?\$LOG_FILE"?'


def _audit() -> int:
    """Phase-1 (python) + phase-2 (bash) grep audit. 0 = pass, 1 = fail."""
    failures: list[str] = []

    # Phase-1 tier: residual `_diag(` in migrated Python sources.
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

    # Phase-2 tier: residual `echo ... >> $LOG_FILE` in the bash wrapper.
    if _WRAPPER.exists():
        hits = _grep(_WRAPPER, _BASH_LEGACY_RE)
        for line_no, line in hits:
            failures.append(
                f"{_WRAPPER.relative_to(REPO)}:{line_no}: residual $LOG_FILE write: "
                f"{line.strip()[:100]}"
            )
    else:
        failures.append(f"missing wrapper: {_WRAPPER.relative_to(REPO)}")

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
            subsystem = rec.get("subsystem", "")
            if not _known_event(event, subsystem):
                warnings += 1
                unknown_events[f"{subsystem}:{event}"] = (
                    unknown_events.get(f"{subsystem}:{event}", 0) + 1
                )

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


def _known_event(event: str, subsystem: str = "") -> bool:
    if any(event == p[:-1] or event.startswith(p) for p in _KNOWN_EVENT_PREFIXES):
        return True
    return subsystem in _KNOWN_SUBSYSTEMS


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
