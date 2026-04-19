# Deep Interview Spec: US-515 — Unified logging (single schema, writer, file, taxonomy)

## Metadata
- Interview ID: unified-logging-2026-04-18
- Rounds: 3
- Final Ambiguity: 18%
- Type: brownfield
- Status: PASSED (≤ 20%)
- **Fresh-start policy:** old log files are DELETED, not archived. Git preserves history if anyone needs to look back.

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|---|---|---|---|
| Goal | 0.80 | 0.35 | 0.28 |
| Constraint | 0.85 | 0.25 | 0.21 |
| Success | 0.90 | 0.25 | 0.23 |
| Context | 0.85 | 0.15 | 0.13 |
| **Total** | | | **0.82** |
| **Ambiguity** | | | **18%** |

## Goal
Unify logging across the autoresearch-splice repo along **all four axes simultaneously** so the current mess (ad-hoc `echo`, bare `print`, `_diag()`, per-subsystem prefixes like `PIPELINE_FAILURE:`, `PHASE:`, `ITER_SUMMARY:`, `CATASTROPHIC-DISCARD:`) collapses into one writer, one schema, one file, and one event taxonomy.

**The four axes (all selected by user):**
1. **Single schema** — every log entry is a structured JSONL record parseable by a single parser
2. **Single writer module** — `.omc/coordination/logger.py` + matching bash function `_log` own all emission; no caller writes to log files directly
3. **Single log file** — `.omc/logs/autoresearch.jsonl` (append-only, rotation-aware). Legacy files (`.omc/autoresearch.log`, `.omc/last_eval.log`, `.omc/research_notes.md`, `.omc/last_reflection.md`, `.omc/tunable_frontier.txt`, `.omc/shap_rollup.json`, `.omc/retest-report.md`) DELETED from the working tree during migration. Git retains history for any post-hoc recovery need.
4. **Consistent levels + event taxonomy** — DEBUG / INFO / WARN / ERROR / CRITICAL, and a named event namespace (wrapper.iteration.start, eval.run, eval.crash, classifier.retrain, retest.candidate, etc.)

## Constraints
- **evaluate.py gets a one-time unprotected edit exemption.** Operator (session claude, this maintainer) migrates `print()` calls to the unified logger in a single commit. Protected status restored afterward. Verify_agent's diff audit does NOT fire on maintainer commits (it only runs during autoresearch hypothesis iterations), so no wrapper collision.
- **Oracle redaction preserved.** The unified logger includes the oracle-denylist regex from US-510 (`combined=`, `splice_f1=`, `clean_score=`, `combined_<domain>=`). Any event passing through the logger that would have these tokens gets them redacted if `subsystem == "retest.diagnose"` or if the event is explicitly tagged `oracle_sensitive: true`. Default is no redaction (wrapper-owned events carry real metrics).
- **Claude subprocess isolation preserved.** Claude (autoresearch iteration worker) only sees events through `.omc/research_notes.md` (US-503 plumbing). A new renderer `_log_to_research_notes` converts the subset of events claude should see (iteration outcomes + research notes + diagnostics) into the existing Markdown format so claude's prompt injection pipeline is unchanged.
- **Old log files DELETED, not preserved.** User explicit: "I don't need / want the previous logs. Delete them, it's okay as long as it's recoverable from git." Migration commit removes the listed legacy files in one atomic sweep.
- **Python and shell callers must both emit schema-compatible lines.** Python uses `logger.py`. Shell uses `_log <level> <event> <kv...>` which writes the same JSONL shape.
- **Zero legacy emitters after migration.** Grep-audit gates: `grep -rn 'print(' [tracked .py]`, `grep -rn 'echo .*>>.*LOG_FILE' run_autoresearch.sh`, `grep -rn '_diag(' [tracked .py]` all return 0 in the migrated codebase (except in the logger module itself, the evaluate.py pre-migration reference if kept, and test fixtures).
- **All readers rewritten in the same PR.** dashboard.py, phase_stats.py, diagnose_repeat_rate.py, shap_shift.py, tunable_frontier.py, notebook_digest.py — all consume the unified schema via a shared reader helper, not ad-hoc grep.

## Non-Goals
- NOT preserving backwards readability of old log formats. Fresh start; git keeps the memory.
- NOT introducing external dependencies (no `structlog`, no `loguru`). Pure stdlib + a 100-200 line logger module.
- NOT unifying classifier training's sklearn `print` verbosity (those are sklearn internals; leave).
- NOT moving files across folders in this PR (US-516 does the reorg separately).

## Acceptance Criteria
- [ ] `.omc/coordination/logger.py` exists with: `Logger` class / `get_logger(subsystem)` factory; `emit(level, event, **kv)` writes one JSONL record; schema = `{ts, level, subsystem, event, ...kv}` with ISO-8601 `ts` and a `schema_version: 1` field on every record.
- [ ] Wrapper-side: `run_autoresearch.sh` has a `_log <level> <subsystem> <event> <k=v k=v ...>` function that writes the identical JSONL record shape. All existing `echo "$(date -Iseconds) ... >> $LOG_FILE"` call sites migrated to `_log`.
- [ ] Python-side: `print()` / `_diag()` call sites in detector.py, features.py, scripts/*.py migrated. `evaluate.py` gets its one-time migration in the same commit. `_diag` either replaced entirely or reimplemented inside logger.py as a thin compatibility wrapper during migration and then removed.
- [ ] Event taxonomy documented in `.omc/coordination/logger.py` module docstring: canonical event names per subsystem, with brief descriptions. Examples: `wrapper.iteration.start`, `wrapper.iteration.summary`, `eval.run`, `eval.crash`, `classifier.retrain.trigger`, `classifier.retrain.complete`, `retest.candidate.begin`, `retest.recovery.committed`, `pipeline.failure`, `diag.shap.no_feature_importances`.
- [ ] Level policy: DEBUG (verbose progress), INFO (normal events), WARN (degraded behavior, recoverable), ERROR (failed iteration, recoverable), CRITICAL (loop-exit-worthy or unrecoverable).
- [ ] Oracle-redaction integrated: logger.emit applies denylist regex on string kv values when `oracle_sensitive=True` is passed (default False). `retest.diagnose.*` events always set it True.
- [ ] Claude-visible renderer: `.omc/research_notes.md` continues to receive Markdown entries (US-503 contract preserved). Implementation: a background tail of the JSONL log filters events where `claude_visible=True` and appends human-readable Markdown blocks to `research_notes.md`. `last_reflection.md` likewise remains the claude-written reflection surface; not affected by this PR.
- [ ] Old files deleted in the migration commit: `.omc/autoresearch.log`, `.omc/last_eval.log`, `.omc/last_reflection.md`, `.omc/research_notes.md`, `.omc/tunable_frontier.txt`, `.omc/shap_rollup.json`, `.omc/retest-report.md`. `.gitignore` entries updated so the new `.omc/logs/` path is the only logger destination.
- [ ] New destination: `.omc/logs/autoresearch.jsonl` (primary). Log rotation: when the file exceeds 50 MB, roll to `.omc/logs/autoresearch.<iso-date>.jsonl`; keep last 10 rolls. `.omc/logs/` gitignored.
- [ ] All readers rewritten: dashboard.py, phase_stats.py, diagnose_repeat_rate.py, shap_shift.py (consumes SHAP sidecars, not logs — confirm scope), tunable_frontier.py, notebook_digest.py. Shared helper `scripts/log_reader.py` provides `iter_events(subsystem=None, event=None, level=None, since=None) → Iterator[dict]` so readers filter without re-parsing each time.
- [ ] Grep-audit gates: `scripts/validate_logs.py --audit` exits 0 iff zero tracked-file violations of the legacy patterns (bare print in non-logger code, bare `echo >> $LOG_FILE`, `_diag(` usage). Used as a pre-commit check going forward.
- [ ] Schema parser gate: `scripts/validate_logs.py --parse .omc/logs/autoresearch.jsonl` parses every line without error and reports any unknown event names against the taxonomy (warning, not blocking).
- [ ] End-to-end: the wrapper runs one full iteration post-migration (US-514 --retest self-test counts; or a clean-eval discard iteration in the live loop). Every observable log line is JSONL, every event maps to a taxonomy entry, no legacy emitter fires.

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|---|---|---|
| "Unified" has one meaning | Ambiguous between schema / writer / file / levels | User selected ALL FOUR |
| Old logs must be preserved | Archival complexity + parser-shim cost | User: DELETE them; git keeps history |
| evaluate.py cannot be touched | "Full retrofit" collides with PROTECTED status | One-time unprotected edit exemption; maintainer-authored, verify_agent doesn't fire on maintainer commits |
| Research notes stay Markdown | Schema-only approach loses claude's human-readable context | Dual surface: JSONL is canonical; a renderer tail re-emits claude-visible events to research_notes.md to preserve US-503 contract |

## Technical Context
Key call sites enumerated this session:
- `run_autoresearch.sh`: `PHASE`, `ITER_SUMMARY`, `PIPELINE_FAILURE`, `DEEP-ORPHAN`, `AUTO-RETRAIN`, `CATASTROPHIC-DISCARD`, `Starting iteration`, `Eval decrypted`, `Eval cleanup`, plus `echo $(date -Iseconds) ...` narrative
- `detector.py`, `features.py`, `ml_eval.py`: `_diag(level, subsystem, event, **kv)` — already semi-structured; natural migration target
- `evaluate.py` (PROTECTED, one-time exemption): `print(f"combined_{ds.id}: ...")` etc.
- `scripts/*.py`: `print()` for CLI output; stays for CLI tools (their stdout is a user interface, not a log)
- `verify_agent.py`: mixed `print()` (report output) + would benefit from structured events for retest-report.md (already markdown, but internal progress should be JSONL-logged)

Schema sketch:
```json
{"schema_version":1,"ts":"2026-04-18T19:42:31+09:00","level":"INFO","subsystem":"wrapper","event":"iteration.start","iter":37,"discards_since_last_keep":4}
{"schema_version":1,"ts":"2026-04-18T19:43:12+09:00","level":"WARN","subsystem":"shap","event":"no_feature_importances","classifier_type":"HistGradientBoostingClassifier","fallback":"uniform_weights"}
{"schema_version":1,"ts":"2026-04-18T19:44:05+09:00","level":"ERROR","subsystem":"eval","event":"crash","exit_code":1,"exception":"AttributeError","message":"'HistGradientBoostingClassifier' object has no attribute 'feature_importances_'","oracle_sensitive":true}
```

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|---|---|---|---|
| LogEvent | core | schema_version, ts, level, subsystem, event, ...kv | emitted by any caller; consumed by any reader |
| Logger | supporting | subsystem, minimum_level, sink_path | produces LogEvent; bound to a subsystem |
| EventTaxonomy | supporting | name_registry, description, level_policy | governs LogEvent.event values |
| LogReader | supporting | iter(filter_spec) → Iterator[LogEvent] | consumes LogEvent stream |
| ResearchNoteRenderer | supporting | filter=claude_visible | converts subset of LogEvents to Markdown |

## Interview Transcript
<details>
<summary>3 rounds</summary>

### Round 1 — Unification axis
**Q:** What does 'unified logging' primarily mean to you? (multi-select)
**A:** All four: schema + writer module + single file + levels/taxonomy.
**Ambiguity:** 37% → next: Success Criteria

### Round 2 — Done criterion
**Q:** How do we know the unification is done?
**A:** Full retrofit + removal of old emitters.
**Ambiguity:** 23% → next: Constraints (collision with protected evaluate.py)

### Round 3 — Protected evaluate.py
**Q:** How should the full retrofit handle evaluate.py's bare print()s?
**A:** One-time unprotected edit exemption.
**Post-answer user clarification:** "I don't need / want the previous logs. Delete them, it's okay as long as it's recoverable from git. Let's start fresh." This eliminates parse-shim + archival complexity.
**Ambiguity:** 18% → threshold met.
</details>
