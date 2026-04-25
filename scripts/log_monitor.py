#!/usr/bin/env python3
"""Pretty-print live monitor for .omc/logs/autoresearch.jsonl (US-515).

Tails the unified JSONL log and formats each record as a scannable line:

    HH:MM:SS  LEVEL  subsystem             event                    kv pairs

Colors (ANSI, tty-only):
  DEBUG  dim-gray   INFO  default   WARN  yellow
  ERROR  red        CRITICAL  bold red

Rotation-safe: if the inode changes (logger.py rolls the file after 50 MB)
or the file shrinks, the monitor re-opens at offset 0.

Usage
-----
  uv run python scripts/log_monitor.py              # tail -f the live log
  uv run python scripts/log_monitor.py --no-follow  # dump existing, exit
  uv run python scripts/log_monitor.py --level WARN --subsystem wrapper
  uv run python scripts/log_monitor.py --max-kv 80  # per-value truncation

Flags:
  --level LEVEL      Only show records at >= this level. Default: DEBUG.
  --subsystem S      Dotted-prefix filter (e.g. `wrapper` matches
                     `wrapper.iteration`).
  --event E          Exact-match event filter.
  --since ISO        Skip records with ts < this value.
  --max-kv N         Truncate individual kv values to N chars (default 120).
  --no-follow        Dump once and exit instead of tailing.
  --path PATH        Alternate JSONL path (default: .omc/logs/autoresearch.jsonl).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(".omc/logs/autoresearch.jsonl")
_LEVELS = ("DEBUG", "INFO", "WARN", "ERROR", "CRITICAL")
_LEVEL_RANK = {name: i for i, name in enumerate(_LEVELS)}

_ANSI = {
    "reset": "\x1b[0m",
    "dim": "\x1b[2m",
    "yellow": "\x1b[33m",
    "red": "\x1b[31m",
    "bold_red": "\x1b[1;31m",
    "cyan": "\x1b[36m",
    "gray": "\x1b[90m",
    "green": "\x1b[32m",
}

_LEVEL_COLOR = {
    "DEBUG": _ANSI["gray"],
    "INFO": "",
    "WARN": _ANSI["yellow"],
    "ERROR": _ANSI["red"],
    "CRITICAL": _ANSI["bold_red"],
}


def _colorize(text: str, color: str, enabled: bool) -> str:
    if not enabled or not color:
        return text
    return f"{color}{text}{_ANSI['reset']}"


def _short_ts(ts: str) -> str:
    # "2026-04-19T07:07:53.671245+00:00" -> "07:07:53"
    if len(ts) >= 19 and ts[10] == "T":
        return ts[11:19]
    return ts[:8] if ts else "--:--:--"


def _trunc(value: Any, limit: int) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    s = s.replace("\n", "\\n").replace("\t", " ")
    if len(s) > limit:
        return s[: limit - 1] + "…"
    return s


def _format_kv(rec: dict, max_kv: int) -> str:
    skip = {"schema_version", "ts", "level", "subsystem", "event", "claude_visible"}
    parts = []
    for k, v in rec.items():
        if k in skip:
            continue
        parts.append(f"{k}={_trunc(v, max_kv)}")
    return "  ".join(parts)


def _format_record(rec: dict, *, color: bool, max_kv: int) -> str:
    level = rec.get("level", "INFO")
    subsystem = rec.get("subsystem", "?")
    event = rec.get("event", "?")
    cv = rec.get("claude_visible", False)

    ts_s = _colorize(_short_ts(rec.get("ts", "")), _ANSI["gray"], color)
    level_s = _colorize(f"{level:<8}", _LEVEL_COLOR.get(level, ""), color)
    subsystem_s = _colorize(f"{subsystem:<20}", _ANSI["cyan"], color)
    event_s = f"{event:<28}"
    cv_marker = _colorize("👁 ", _ANSI["green"], color) if cv else "  "
    kv_s = _format_kv(rec, max_kv)
    return f"{ts_s}  {level_s}{cv_marker}{subsystem_s} {event_s} {kv_s}".rstrip()


def _filter_pass(
    rec: dict,
    *,
    min_level: str,
    subsystem: str | None,
    event: str | None,
    since: str | None,
) -> bool:
    lvl = rec.get("level", "INFO")
    if _LEVEL_RANK.get(lvl, 0) < _LEVEL_RANK.get(min_level, 0):
        return False
    if subsystem is not None:
        sub = rec.get("subsystem", "")
        if not (sub == subsystem or sub.startswith(subsystem + ".")):
            return False
    if event is not None and rec.get("event") != event:
        return False
    if since is not None and rec.get("ts", "") < since:
        return False
    return True


def _emit(line: str, rec: dict, out, *, color: bool, max_kv: int, **filters) -> None:
    if not _filter_pass(rec, **filters):
        return
    out.write(_format_record(rec, color=color, max_kv=max_kv) + "\n")
    out.flush()


def _stat_tuple(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_size)


def _dump_once(path: Path, out, *, color: bool, max_kv: int, **filters) -> int:
    if not path.exists():
        print(f"log_monitor: no such file: {path}", file=sys.stderr)
        return 1
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            _emit(raw, rec, out, color=color, max_kv=max_kv, **filters)
    return 0


def _follow(path: Path, out, *, color: bool, max_kv: int, **filters) -> int:
    # Print existing records, then tail for new ones. Rotation-safe: if
    # the inode changes (logger.py rolled after 50 MB) or the size shrinks
    # (truncation), we reopen at offset 0.
    _dump_once(path, out, color=color, max_kv=max_kv, **filters)

    last_stat = _stat_tuple(path)
    offset = path.stat().st_size if path.exists() else 0

    while True:
        time.sleep(0.5)
        cur = _stat_tuple(path)
        if cur is None:
            # File vanished (unusual — wait for reappearance).
            continue
        cur_ino, cur_size = cur
        if last_stat is None or cur_ino != last_stat[0] or cur_size < offset:
            # Rotation or truncation: start over from the beginning of
            # whatever file currently sits at `path`.
            offset = 0
        last_stat = cur

        if cur_size <= offset:
            continue

        with open(path, "r", encoding="utf-8") as f:
            f.seek(offset)
            for raw in f:
                if not raw.endswith("\n"):
                    # Partial line — leave for the next tick so we never
                    # emit half an event.
                    break
                offset += len(raw.encode("utf-8"))
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                _emit(stripped, rec, out, color=color, max_kv=max_kv, **filters)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=_DEFAULT_PATH)
    ap.add_argument("--level", default="DEBUG", choices=_LEVELS)
    ap.add_argument("--subsystem")
    ap.add_argument("--event")
    ap.add_argument("--since")
    ap.add_argument("--max-kv", type=int, default=120,
                    help="Truncate kv values to this many chars")
    ap.add_argument("--no-follow", action="store_true",
                    help="Dump existing records and exit (default: tail -f)")
    ap.add_argument("--no-color", action="store_true",
                    help="Disable ANSI colors even on a tty")
    args = ap.parse_args()

    color = sys.stdout.isatty() and not args.no_color
    filters = dict(
        min_level=args.level,
        subsystem=args.subsystem,
        event=args.event,
        since=args.since,
    )

    try:
        if args.no_follow:
            return _dump_once(
                args.path, sys.stdout, color=color, max_kv=args.max_kv, **filters
            )
        return _follow(
            args.path, sys.stdout, color=color, max_kv=args.max_kv, **filters
        )
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        # `log_monitor ... | head` closes stdout early; clean exit.
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
