#!/usr/bin/env python3
"""Phase-1 dual-format reader for the unified log (US-515).

Produces a unified iterator over events from two sources:

- `.omc/logs/autoresearch.jsonl` (native JSONL, emitted by Python callers)
- `.omc/autoresearch.log` (legacy shell-emitted text; phase-2 deletes this)

The shim synthesizes structured events from well-known legacy prefixes so
log-consuming readers (phase_stats.py, dashboard.py) can move to this API
without waiting for bash migration.

**SCHEDULED REMOVAL MARKER:** When phase 2 migrates the wrapper, the
`_iter_legacy_events` branch and the `legacy_path` parameter are DELETED.
Search for this marker to find the removal locus.

Test-time override: if `OMC_LOG_OVERRIDE` is set, that path is used for
JSONL in place of the default. Callers can also pass explicit paths.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from heapq import merge
from pathlib import Path
from typing import Iterable, Iterator, Optional

_DEFAULT_JSONL = Path(".omc/logs/autoresearch.jsonl")
_DEFAULT_LEGACY = Path(".omc/autoresearch.log")

_ISO_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:?\d{2}|Z)?)\s+")

_ITER_SUMMARY_RE = re.compile(
    r"ITER_SUMMARY\s+iter=(?P<iter>\S+)\s+status=(?P<status>\S+)\s+"
    r"total=(?P<total>\S+)\s+claude=(?P<claude>\S+)\s+retrain=(?P<retrain>\S+)\s+"
    r"eval=(?P<eval>\S+)\s+verify=(?P<verify>\S+)\s+note=(?P<note>\S+)"
)
_PHASE_RE = re.compile(r"PHASE:\s*(?P<phase>\S+)\s+(?P<action>start|end|skip)"
                       r"(?:\s+(?P<rest>.*))?$")
_PIPELINE_FAIL_RE = re.compile(r"PIPELINE_FAILURE:\s*(?P<detail>.*)$")
_STARTING_RE = re.compile(r"Starting iteration \(consecutive discards:\s*(\d+)\)")
_ITERATION_RESULT_RE = re.compile(
    r"Iteration:\s+(?P<status>\S+)(?:\s+\(discards:\s+(?P<d>\d+)/(?P<cap>\d+)\))?"
)


def _jsonl_path() -> Path:
    override = os.environ.get("OMC_LOG_OVERRIDE")
    return Path(override) if override else _DEFAULT_JSONL


def iter_events(
    *,
    subsystem: Optional[str] = None,
    event: Optional[str] = None,
    level: Optional[str] = None,
    since: Optional[str] = None,
    jsonl_path: Optional[Path] = None,
    legacy_path: Optional[Path] = None,
) -> Iterator[dict]:
    """Yield matching events from both JSONL and legacy log, merged by `ts`.

    Filters (all optional, combined with AND):
      - subsystem: exact match OR dotted prefix match ("wrapper" matches "wrapper.iteration").
      - event:     exact match OR dotted prefix match.
      - level:     exact match (e.g. "INFO").
      - since:     ISO-8601 string; only events with ts > since.

    Unknown fields are preserved on each record. The `ts` key is always
    present after normalization.
    """
    jp = Path(jsonl_path) if jsonl_path is not None else _jsonl_path()
    lp = Path(legacy_path) if legacy_path is not None else _DEFAULT_LEGACY

    jsonl_iter = _iter_jsonl(jp) if jp.exists() else iter(())
    legacy_iter = _iter_legacy_events(lp) if lp.exists() else iter(())

    combined = merge(jsonl_iter, legacy_iter, key=lambda r: r.get("ts", ""))

    for rec in combined:
        if subsystem is not None and not _dotted_prefix_match(rec.get("subsystem"), subsystem):
            continue
        if event is not None and rec.get("event") != event:
            continue
        if level is not None and rec.get("level") != level:
            continue
        if since is not None and rec.get("ts", "") <= since:
            continue
        yield rec


def _dotted_prefix_match(value: Optional[str], wanted: str) -> bool:
    """Used for subsystem filters. `wrapper` matches `wrapper.iteration`.

    Event filtering is exact-match (see caller) so `iteration.phase` does
    NOT catch `iteration.phase.step`. Callers wanting prefix semantics on
    events should filter manually or use two calls.
    """
    if not value:
        return False
    return value == wanted or value.startswith(wanted + ".")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if "ts" not in rec:
                rec["ts"] = ""
            yield rec


def _iter_legacy_events(path: Path) -> Iterator[dict]:
    """Synthesize structured events from legacy shell-prefixed log lines.

    Covers the minimum set that phase 1 consumers need:

        ITER_SUMMARY iter=... status=... total=... ...  -> wrapper.iteration.phase
        PHASE: <name> start|end|skip [rest]             -> wrapper.iteration.phase.step
        PIPELINE_FAILURE: <detail>                      -> pipeline.failure
        Starting iteration (consecutive discards: N)    -> wrapper.iteration.start
        Iteration: <status> (discards: d/cap)           -> wrapper.iteration.summary

    SCHEDULED REMOVAL MARKER: delete this function in phase 2.
    """
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            ts_match = _ISO_PREFIX.match(line)
            ts = ts_match.group(1) if ts_match else ""
            body = line[ts_match.end():] if ts_match else line

            m = _ITER_SUMMARY_RE.search(body)
            if m:
                yield {
                    "schema_version": 1,
                    "ts": ts,
                    "level": "INFO",
                    "subsystem": "wrapper",
                    "event": "iteration.phase",
                    "source": "legacy",
                    **m.groupdict(),
                }
                continue

            m = _PHASE_RE.match(body)
            if m:
                yield {
                    "schema_version": 1,
                    "ts": ts,
                    "level": "INFO",
                    "subsystem": "wrapper",
                    "event": "iteration.phase.step",
                    "source": "legacy",
                    "phase": m.group("phase"),
                    "action": m.group("action"),
                    "detail": (m.group("rest") or "").strip(),
                }
                continue

            m = _PIPELINE_FAIL_RE.match(body)
            if m:
                yield {
                    "schema_version": 1,
                    "ts": ts,
                    "level": "CRITICAL",
                    "subsystem": "wrapper",
                    "event": "pipeline.failure",
                    "source": "legacy",
                    "detail": m.group("detail").strip(),
                }
                continue

            m = _STARTING_RE.search(body)
            if m:
                yield {
                    "schema_version": 1,
                    "ts": ts,
                    "level": "INFO",
                    "subsystem": "wrapper",
                    "event": "iteration.start",
                    "source": "legacy",
                    "consecutive_discards": int(m.group(1)),
                }
                continue

            m = _ITERATION_RESULT_RE.search(body)
            if m:
                yield {
                    "schema_version": 1,
                    "ts": ts,
                    "level": "INFO",
                    "subsystem": "wrapper",
                    "event": "iteration.summary",
                    "source": "legacy",
                    "status": m.group("status"),
                    **({"discards": int(m.group("d")), "cap": int(m.group("cap"))}
                       if m.group("d") else {}),
                }


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsystem")
    ap.add_argument("--event")
    ap.add_argument("--level")
    ap.add_argument("--since")
    ap.add_argument("--jsonl", type=Path)
    ap.add_argument("--legacy", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    count = 0
    for rec in iter_events(
        subsystem=args.subsystem,
        event=args.event,
        level=args.level,
        since=args.since,
        jsonl_path=args.jsonl,
        legacy_path=args.legacy,
    ):
        print(json.dumps(rec, default=str))
        count += 1
        if args.limit and count >= args.limit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
