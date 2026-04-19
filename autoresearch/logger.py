"""Unified structured JSONL logger for autoresearch-splice (US-515 phase 1).

One writer per surface: every Python caller uses this module. The wrapper
(run_autoresearch.sh) emits via `_log` helper which shells out to
`autoresearch/log_cli.py` (US-515 phase 2 complete).

IMPORT CONVENTION
-----------------
The `autoresearch` package resolves via `PYTHONPATH=$PROJECT_DIR` (set
by `run_autoresearch.sh`'s `export PYTHONPATH` line, inherited by all
subprocesses). Direct callers use:

    from autoresearch.logger import get_logger

    log = get_logger("detector.gbm")
    log.emit("INFO", "diag.gbm.dedupe", before=120, after=80)

Operator scripts invoked outside the wrapper must set PYTHONPATH
manually: `PYTHONPATH=$PWD uv run python scripts/dashboard.py`.

DISABLED MODE
-------------
Set `OMC_LOGGER_DISABLED=1` to make `emit()` a no-op. Useful for unit tests
that want to import caller modules without a real log sink. The US-515
smoke fixture deliberately does NOT set this — redaction/parse gates need
real emission.

EVENT TAXONOMY
--------------
Canonical event names (dotted) per subsystem:

    eval.*
        eval.run                -- evaluator started / finished wrapping
        eval.run.complete       -- final elapsed_s
        eval.crash              -- unhandled exception in evaluate.py
        eval.dataset.result     -- per-dataset combined / clean_fp rollup
        eval.dataset.error      -- per-dataset "ERROR" carrier (wrapper contract)
        eval.metrics.splice     -- splice_f1/clean_score/combined for one
                                   evaluate() run (per data_dir)
        eval.metrics.clean      -- precision/recall/fp_rate + TP/FP/FN counts
                                   for one evaluate() run (per data_dir)
        eval.fp.distribution    -- FP distribution breakdown
        eval.crossfade.breakdown-- per-crossfade-type metrics
        eval.loc_accuracy       -- localization accuracy rollup
        eval.opus32k.metrics    -- opus-32k per-dir metrics
        eval.opus32k.aggregate  -- opus-32k aggregate across dirs
        eval.aggregate          -- cross-dataset aggregate. `mode` field
                                   discriminates fields present:
                                     mode="multi"         -> splice_f1,
                                       clean_score, combined, precision,
                                       recall, tp/fp/fn/clean_fp
                                     mode="cross-dataset" -> combined,
                                       combined_mean, combined_min,
                                       clean_fp, per_dataset*
                                   Readers querying by `event==eval.aggregate`
                                   MUST branch on `mode` (or use .get()).
        eval.single.combined    -- single-dataset combined line
        eval.input.error        -- input file / arg / path error

    classifier.*
        classifier.retrain.trigger   -- wrapper decided to retrain
        classifier.retrain.complete  -- retrain succeeded
        classifier.retrain.failed    -- retrain raised

    retest.*     (auto-sets oracle_sensitive=True for retest.diagnose.*)
        retest.candidate.begin       -- entering a retest worktree
        retest.candidate.end         -- leaving it
        retest.recovery.committed    -- recovered hypothesis committed
        retest.diagnose.traceback    -- diag w/ traceback (REDACTED)
        retest.diagnose.per_domain_error -- silent per-domain failure (REDACTED)

    diag.*       (mechanical migration of former detector._diag calls)
        diag.<component>.<reason>    -- component = gbm | slider | pairwise |
                                        hotelling | refine | zscore |
                                        robust_zscore | ... (detector)
        diag.features.features.*     -- features.py
        diag.ml.eval.*               -- ml_eval.py
        diag.shap.report.*           -- shap_report.py
        diag.classifier.train.*      -- train_classifier.py

    pipeline.*
        pipeline.failure             -- catastrophic wrapper exit

    RESERVED for phase 2 (wrapper migration):
        wrapper.iteration.start / .summary / .phase
        note.appended
        tunable.frontier.snapshot

ORACLE REDACTION
----------------
`_ORACLE_DENYLIST_RE` strips `combined=...`, `combined_<domain>=...`,
`splice_f1=...`, `clean_score=...` from STRING kv values when either:

- the caller passes `oracle_sensitive=True`, OR
- the event name starts with `retest.diagnose.` (auto-enabled)

**Redaction is top-level and string-only by design.** Numeric kwargs
(`combined=0.47` as a float) flow through untouched. Do NOT pass raw
metric floats on a path that could reach claude — either format them
into the string message or omit them. Nested dicts/lists aren't
recursed either; keep oracle-sensitive payloads flat and string-valued.

Default is NO redaction: wrapper-owned events carry real metrics. This
preserves the US-510 claude/oracle isolation contract for diagnose paths.

SCHEMA
------
    {
      "schema_version": 1,
      "ts": "<ISO-8601 UTC>",
      "level": "DEBUG|INFO|WARN|ERROR|CRITICAL",
      "subsystem": "<dotted>",
      "event": "<dotted>",
      "claude_visible": false,
      ...kv
    }

DESTINATION
-----------
`.omc/logs/autoresearch.jsonl`. Rotated when size exceeds 50 MB; retains
the last 10 rolls as `.omc/logs/autoresearch.<iso-date>.jsonl`. Rotation
is guarded by `fcntl.flock(LOCK_EX)` so concurrent subprocesses cannot
interleave rolled and live writes.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_DEFAULT_LOG_PATH = Path(".omc/logs/autoresearch.jsonl")
MAX_SIZE_BYTES = 50 * 1024 * 1024
MAX_ROLLS = 10

_ORACLE_DENYLIST_RE = re.compile(
    r"\b(combined(?:_[a-z]+)?|splice_f1|clean_score)\s*=\s*[0-9.]+"
)
_VALID_LEVELS = frozenset({"DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"})


def _log_path() -> Path:
    """Resolve the active log destination.

    Honors `OMC_LOG_OVERRIDE` (used by tests and the smoke fixture). Returns
    an absolute-or-relative `Path`; caller is responsible for mkdir.
    """
    override = os.environ.get("OMC_LOG_OVERRIDE")
    return Path(override) if override else _DEFAULT_LOG_PATH


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _ORACLE_DENYLIST_RE.sub(r"\1=<REDACTED>", value)
    return value


class Logger:
    """Subsystem-bound logger. Instances are cheap; get_logger caches."""

    def __init__(self, subsystem: str) -> None:
        self.subsystem = subsystem

    def emit(
        self,
        level: str,
        event: str,
        *,
        oracle_sensitive: bool = False,
        claude_visible: bool = False,
        **kv: Any,
    ) -> None:
        if os.environ.get("OMC_LOGGER_DISABLED") == "1":
            return
        if level not in _VALID_LEVELS:
            level = "INFO"
        if event.startswith("retest.diagnose."):
            oracle_sensitive = True
        if oracle_sensitive:
            kv = {k: _redact(v) for k, v in kv.items()}

        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": SCHEMA_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "subsystem": self.subsystem,
            "event": event,
            "claude_visible": claude_visible,
            **kv,
        }
        line = json.dumps(record, default=str) + "\n"

        with open(path, "a", encoding="utf-8") as guard:
            fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
            try:
                rotated = False
                if path.stat().st_size + len(line) > MAX_SIZE_BYTES:
                    self._rotate_locked(path)
                    rotated = True
                if rotated:
                    # `guard`'s fd points at the renamed inode after
                    # rotation. Writing to it would land in the rolled
                    # archive instead of the live log. Open a new fd on
                    # `path` for the record. LOCK_EX on `guard` continues
                    # to serialize concurrent writers until this block
                    # exits — no need to re-flock, since rotation is
                    # rare and the live-log write is append-atomic.
                    with open(path, "a", encoding="utf-8") as live:
                        live.write(line)
                        live.flush()
                else:
                    guard.write(line)
                    guard.flush()
            finally:
                fcntl.flock(guard.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _rotate_locked(path: Path) -> None:
        """Rename the live log and prune older rolls. Caller holds LOCK_EX."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rolled = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
        try:
            path.rename(rolled)
        except FileNotFoundError:
            return
        siblings = sorted(
            path.parent.glob(f"{path.stem}.*{path.suffix}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in siblings[MAX_ROLLS:]:
            try:
                stale.unlink()
            except FileNotFoundError:
                pass


_CACHE: dict[str, Logger] = {}


def get_logger(subsystem: str) -> Logger:
    """Return (or create+cache) the Logger bound to the given subsystem."""
    inst = _CACHE.get(subsystem)
    if inst is None:
        inst = Logger(subsystem)
        _CACHE[subsystem] = inst
    return inst


def render_claude_visible_events(
    *,
    since_ts: str | None = None,
    jsonl_path: Path | str | None = None,
    target: Path | str = Path(".omc/research_notes.md"),
) -> int:
    """Append Markdown blocks for claude_visible events to `target`.

    Preserves the US-503 contract: the wrapper/claude sees Markdown, never
    JSONL. Returns the count of events rendered. Target parent dir is
    created lazily; target file is APPENDED to, not overwritten.
    """
    path = Path(jsonl_path) if jsonl_path is not None else _log_path()
    target = Path(target)
    if not path.exists():
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "r", encoding="utf-8") as src, open(
        target, "a", encoding="utf-8"
    ) as dst:
        for raw in src:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not rec.get("claude_visible"):
                continue
            if since_ts is not None and rec.get("ts", "") <= since_ts:
                continue
            dst.write(_render_markdown(rec))
            count += 1
    return count


def _render_markdown(rec: dict[str, Any]) -> str:
    ts = rec.get("ts", "")
    level = rec.get("level", "INFO")
    subsystem = rec.get("subsystem", "?")
    event = rec.get("event", "?")
    extras = {
        k: v
        for k, v in rec.items()
        if k
        not in {
            "schema_version",
            "ts",
            "level",
            "subsystem",
            "event",
            "claude_visible",
        }
    }
    head = f"### [{ts}] {level} {subsystem} :: {event}\n"
    if not extras:
        return head + "\n"
    body_lines = [f"- **{k}**: {v}" for k, v in extras.items()]
    return head + "\n".join(body_lines) + "\n\n"
