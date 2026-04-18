#!/usr/bin/env python3
"""Research-notes compaction (US-503 helper).

Reads `.omc/research_notes.md`. When total entries exceed MAX_ENTRIES,
the oldest SUMMARIZE_OLDEST entries are collapsed into one '## Historical
digest (entries 1..N)' block (keep/discard counts + representative
combined range), preserving chronological narrative for recent entries.

Entry delimiter is a `## ` header at start-of-line. Wrapper appends
`## <timestamp> — <sha> (<status>, combined=<v>)\n...` blocks.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTES = REPO / ".omc" / "research_notes.md"
MAX_ENTRIES = 50
SUMMARIZE_OLDEST = 20


def _split_entries(text: str) -> list[str]:
    """Return entry-blocks. First element may be preamble (no `## ` header)."""
    if not text.strip():
        return []
    parts = re.split(r"(?m)^## ", text)
    out: list[str] = []
    if parts and parts[0]:
        out.append(parts[0])
    out.extend("## " + p for p in parts[1:] if p)
    return out


def _summarize(entries: list[str]) -> str:
    keeps = discards = vfails = 0
    combined_vals: list[float] = []
    first_header = last_header = ""
    for e in entries:
        first_line = e.split("\n", 1)[0]
        if not first_header:
            first_header = first_line
        last_header = first_line
        if "(keep," in first_line:
            keeps += 1
        elif "(discard," in first_line:
            discards += 1
        elif "(verify-fail," in first_line:
            vfails += 1
        m = re.search(r"combined=([0-9.]+)", first_line)
        if m:
            try:
                combined_vals.append(float(m.group(1)))
            except ValueError:
                pass

    rng = (f"combined range [{min(combined_vals):.4f}, {max(combined_vals):.4f}]"
           if combined_vals else "no combined values recorded")
    span = f"{first_header.replace('## ', '').strip()} … {last_header.replace('## ', '').strip()}"
    return (
        f"## Historical digest (oldest {len(entries)} entries compacted)\n"
        f"Span: {span}\n"
        f"Outcomes: {keeps} keep / {discards} discard / {vfails} verify-fail; {rng}.\n"
        f"(Older reflections collapsed to conserve prompt budget.)\n\n"
    )


def digest() -> tuple[bool, int]:
    """Return (did_compact, new_entry_count). Rewrites NOTES in place."""
    if not NOTES.exists():
        return False, 0
    text = NOTES.read_text()
    entries = _split_entries(text)
    content_entries = [e for e in entries if e.startswith("## ")]
    if len(content_entries) <= MAX_ENTRIES:
        return False, len(content_entries)

    to_summarize = content_entries[:SUMMARIZE_OLDEST]
    kept = content_entries[SUMMARIZE_OLDEST:]
    summary = _summarize(to_summarize)
    NOTES.write_text(summary + "".join(kept))
    return True, len(kept) + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true",
                    help="Print entry count without compacting.")
    args = ap.parse_args()
    if args.status:
        if not NOTES.exists():
            print("notebook: (not yet created)")
            return 0
        entries = [e for e in _split_entries(NOTES.read_text()) if e.startswith("## ")]
        print(f"notebook: {len(entries)} entries (threshold: {MAX_ENTRIES})")
        return 0
    did, count = digest()
    msg = "compacted" if did else "no compaction needed"
    print(f"notebook_digest: {msg}; entries now {count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
