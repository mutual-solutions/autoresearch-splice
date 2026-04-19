#!/usr/bin/env python3
"""Native JSONL reader for the unified log (US-515 phase 2).

Yields events from `.omc/logs/autoresearch.jsonl`. Phase 1 shipped with a
dual-format shim that also parsed legacy shell-prefixed lines from
`.omc/autoresearch.log`; phase 2 migrated the wrapper so the shim is
gone. No more merge-sort; no more legacy path.

Test-time override: if `OMC_LOG_OVERRIDE` is set, that path replaces the
default. Callers can also pass an explicit `jsonl_path` argument.

Filtering semantics (kept from phase 1 for reader callers):
- subsystem: exact match OR dotted prefix ("wrapper" matches
  "wrapper.iteration").
- event: exact match only.
- level: exact match.
- since: ISO-8601 timestamp; STRICT less-than is dropped so a caller
  capturing `since_ts = now_iso()` before spawning a subprocess still
  sees that subprocess's first event.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Optional

_DEFAULT_JSONL = Path(".omc/logs/autoresearch.jsonl")


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
) -> Iterator[dict]:
    """Yield matching events from the unified JSONL log."""
    jp = Path(jsonl_path) if jsonl_path is not None else _jsonl_path()
    if not jp.exists():
        return

    with open(jp, "r", encoding="utf-8") as f:
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
            if subsystem is not None and not _dotted_prefix_match(rec.get("subsystem"), subsystem):
                continue
            if event is not None and rec.get("event") != event:
                continue
            if level is not None and rec.get("level") != level:
                continue
            if since is not None and rec.get("ts", "") < since:
                continue
            yield rec


def _dotted_prefix_match(value: Optional[str], wanted: str) -> bool:
    if not value:
        return False
    return value == wanted or value.startswith(wanted + ".")


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsystem")
    ap.add_argument("--event")
    ap.add_argument("--level")
    ap.add_argument("--since")
    ap.add_argument("--jsonl", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    count = 0
    for rec in iter_events(
        subsystem=args.subsystem,
        event=args.event,
        level=args.level,
        since=args.since,
        jsonl_path=args.jsonl,
    ):
        print(json.dumps(rec, default=str))
        count += 1
        if args.limit and count >= args.limit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
