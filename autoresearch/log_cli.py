"""Thin CLI wrapper over `logger.emit()` for the bash wrapper (US-515 phase 2).

Usage
-----
    python .omc/coordination/log_cli.py LEVEL SUBSYSTEM EVENT [key=value ...]

Every `key=value` after EVENT becomes a string kwarg on `logger.emit()`.
Two keys get special treatment (stripped from the kwargs before emit):

    claude_visible=true|false     -> forwarded as the explicit flag
    oracle_sensitive=true|false   -> forwarded as the explicit flag

Values are always strings — bash has no int/float types worth encoding
for this surface. Per-site sites that actually need numeric payloads
either embed the number in a string message or call `logger.emit()`
directly from Python.

Exit code is 0 on success. On argparse error the process exits 2. A
`logger.emit()` raise is reraised with exit code 1 (surfaces rotation
or IO failures to the shell caller, which redirects stderr to the
child-stderr sidecar).

Latency note
------------
Python startup is ~200-400ms per invocation on this machine. With
~15-25 `_log` calls per iteration and 5-15 minute iteration wall time,
the per-iteration overhead is 0.3-3% — within the 5% phase-2 budget.
If this regresses (>5% for 10 consecutive iterations), the fallback
per ralplan §2-Follow-ups #6 is a socket daemon or named pipe.
"""

from __future__ import annotations

import os
import sys

from autoresearch.logger import get_logger


_FLAG_KEYS = frozenset({"claude_visible", "oracle_sensitive"})


def _coerce_flag(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(
            "usage: log_cli.py LEVEL SUBSYSTEM EVENT [key=value ...]",
            file=sys.stderr,
        )
        return 2
    level, subsystem, event = argv[1], argv[2], argv[3]

    claude_visible = False
    oracle_sensitive = False
    kv: dict[str, str] = {}
    for raw in argv[4:]:
        if "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        if key == "claude_visible":
            claude_visible = _coerce_flag(value)
        elif key == "oracle_sensitive":
            oracle_sensitive = _coerce_flag(value)
        else:
            kv[key] = value

    get_logger(subsystem).emit(
        level,
        event,
        claude_visible=claude_visible,
        oracle_sensitive=oracle_sensitive,
        **kv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
