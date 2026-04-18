# RALPLAN — US-508: `verify_agent --diagnose` (Phase 1) + session-resume (Phase 2, deferred)

**Plan ID:** `ralplan-verify-agent-resume`
**Source spec:** `/Users/yejunjang/Projects/mutual/autoresearch-splice/.omc/specs/deep-interview-verify-agent-resume.md`
**Date:** 2026-04-18 (revised after Architect ITERATE verdict)
**Mode:** consensus (SHORT)
**Status:** Phase 1 ready for Critic review; Phase 2 queued as follow-up US

---

## Context

When the autoresearch loop's post-hypothesis evaluation crashes (one of three paths in `run_autoresearch.sh` — eval exit != 0, no `RESULTS_TSV` parseable, or US-505 retrain rc != 0), we currently fire `_guarded_reset` and move on. Claude never sees *why* its hypothesis crashed, so the same mistake recurs.

This plan now ships in **two phases**. **Phase 1** (this plan) adds an additive `verify_agent.py --diagnose` subcommand that reads `.omc/last_eval.log`, redacts oracle signals, classifies the exception, and writes a 3-8 line note. The note is picked up by existing `_append_note` plumbing and appended to `.omc/research_notes.md`, so the **next** iteration's claude sees the crash diagnosis via the current note-injection path. No session-resume, no fix-on-top, no re-run of evaluate.py. **Phase 2** (deferred follow-up US, conditionally triggered) layers session-resume + fix-on-top + single re-run.

**Autoresearch loop status:** currently RUNNING in tmux `autoresearch`. Phase 1 implementation requires a brief stop window.

---

## Architect Asks — Resolution Map

| # | Ask | Resolution | Where documented |
|---|-----|------------|-------------------|
| 1 | Phase-1/Phase-2 split vs. ship-both | **Accepted (1a).** Phase 1 = passive note only. Phase 2 = session-resume, queued as follow-up US, triggered only if repeat-rate reduction < 50% after 20 iterations. | §"Phase Decision" below + ADR |
| 2 | `fix:` subject regex | **Deferred to Phase 2** (no wrapper-side `fix:` detection exists in Phase 1). Locked choice: `^fix[:(]` case-sensitive. | Phase 2 stub below + ADR Follow-ups |
| 3 | 2nd-eval crash overwriting first diagnosis | **Deferred to Phase 2.** Locked resolution: option (b) — `.omc/diagnose-log.jsonl` append-only trail; `.omc/last_reflection.md` stays single-write-per-iteration. | Phase 2 stub below + ADR Follow-ups |
| 4 | Probe script asserts creation-time success | **Deferred to Phase 2.** When implemented: probe verifies (i) `--session-id <uuid>` at creation exits 0, (ii) `-r <same-uuid>` second call sees prior context. | Phase 2 stub below + ADR Follow-ups |
| 5 | Denylist vs whitelist redaction | **Chosen (b): denylist + pinned unit test + residual-risk acknowledgment.** Phase 1 scope. Unit test asserts canonical oracle patterns ARE stripped AND canonical `AssertionError` lines containing floats are PRESERVED. | §"Principles" #2 + Step 3 acceptance |

---

## Phase Decision (Architect ask #1)

### Why accept the split

The deep-interview framing ("resume the previous agent") expressed a **preference for preserving claude's reasoning thread**. The Architect's counter is sharper than it first looked: in Phase 1, the *information* reaching claude on the next iteration (classified exception + top frames + redacted message) is identical to what the Phase-2 resumed prompt would contain. What Phase 2 adds is:

1. Preservation of the original hypothesis-forming tool-use trace inside claude's own session.
2. A same-iteration fix-on-top so the hypothesis isn't discarded before claude can correct it.

Neither is load-bearing for the primary failure mode this plan targets (same-exception-class recurrence). If the diagnosis note reduces repeat-rate by ≥50% alone, Phase 2 is gold-plating on a problem already solved. If it doesn't, Phase 2 is strictly justified — the metric itself decides.

### Why the user's intent is still honored

"Keep the door open" ≠ "ship today." Phase 1 writes the diagnosis in a machine-readable format that a Phase 2 implementation consumes verbatim (`.omc/last_reflection.md` contents + classification fields); no Phase 1 work is thrown away. The ADR explicitly records Phase 2 as a committed follow-up gated on the metric, not on a vague "later."

### Scope impact

Phase 1 shrinks from ~7 steps / ~300 lines of wrapper+python diff to **3 steps / ~40-50 lines of python + 2-3 lines in the wrapper + the metrics script**. No probe script. No session-id capture. No `_diagnose_and_resume` helper. No `_rerun_eval_once` helper. No new `verify-fail-postfix` status.

---

## RALPLAN-DR Summary (Phase 1)

### Principles (5)

1. **Additive, not replacement.** `--diagnose` is a new subcommand. The existing 5-check strict-improvement path is untouched. The three crash sites gain one new line (a `--diagnose` invocation) before the existing `_append_note` call.
2. **Oracle isolation via denylist + pinned test + acknowledged residual risk.** Redact `combined=`, `combined_<dom>=`, `splice_f1=`, `clean_score=` before anything reaches disk. Unit test (Architect ask #5, option b) pins the redaction regex: canonical oracle patterns ARE stripped, canonical `AssertionError: x=1.5 not in (0, 1)` lines are PRESERVED. **Residual risk acknowledged:** a future protocol change that introduces a new oracle key (e.g. `per_dataset_score_singing=`) would leak until the denylist is widened. Owner: whoever adds the new oracle key must update the denylist in the same PR.
3. **Passive information flow.** Phase 1 does not modify claude's invocation, does not capture a session-id, does not invoke claude twice per hypothesis. The diagnosis reaches claude through the same `_append_note` → `.omc/research_notes.md` → next-iteration prompt-prefix plumbing that already exists.
4. **Single reset primitive.** Existing `_guarded_reset` is untouched. No new reset sites.
5. **Metric-gated escalation.** Phase 2 ships only if the success-metric script (Step 3) shows the passive note achieved <50% repeat-rate reduction after 20 iterations. Decision criterion is numeric, not vibes.

### Decision Drivers (top 3)

1. **Minimize test surface on the primary path.** The Architect's observation — that Phase 2's 2x test surface (session-resume + fallback) is worth paying only when the simpler path has been measured and found wanting — is sound.
2. **Zero regression on clean iterations.** If evaluate.py succeeds, nothing new fires. Same constraint as the original plan; now trivially satisfied because Phase 1 only adds a single conditional python invocation on the crash paths.
3. **Diagnose speed (<1s).** Rereading the existing log is trivial; re-running evaluate.py would take 3+ min and risk re-crashing.

### Viable Options (Phase 1)

#### Option C+ — Passive diagnosis note, consumed by next iteration (**CHOSEN**)

`verify_agent.py --diagnose` reads `.omc/last_eval.log`, redacts, classifies, writes to `.omc/last_reflection.md`. On the three crash paths, wrapper invokes `--diagnose` *before* `_append_note`, so `_append_note` appends the diagnosis-bearing reflection into `.omc/research_notes.md` exactly like a normal iteration's reflection. Next iteration's claude sees it through the existing note-injection prompt prefix.

- Pros: one code path; ~40-50 lines python + ~3 lines wrapper; no CLI probe; no session-id; no composite-prompt fallback; no `fix:` commits to reason about; no hypothesis+fix revert pair; oracle redaction has one write site to guard.
- Cons: diagnosis arrives one iteration later; claude's original tool-use reasoning trace on the crashed hypothesis is lost (same as current behavior — nothing worsened).

#### Option A — Session-resume primary, re-prime fallback (**DEFERRED to Phase 2**)

Preserved verbatim for when Phase 2 ships. See "Phase 2 Stub" section below.

### Invalidation Rationale

- **Option B** (always re-prime with composite prompt, no session resume) remains invalidated by spec Goal if and when Phase 2 ships. For Phase 1, the question doesn't arise (no resume at all).
- The earlier invalidation of "Option C" in the prior plan revision was predicated on the spec's acceptance criterion requiring a `fix:` commit. That criterion is **scoped to Phase 2** going forward; Phase 1's acceptance criteria are the strict subset listed below. The spec is not being weakened — it is being **phased**, with Phase 2 completing it.

---

## Work Objectives (Phase 1)

1. Add `verify_agent.py --diagnose` subcommand (traceback parse + classify + redact + write note, <1s).
2. Wire the three crash paths in `run_autoresearch.sh` to invoke `--diagnose` before the existing `_append_note` call. No other wrapper changes.
3. Ship the success-metric script (`scripts/diagnose_repeat_rate.py`) for baseline capture and post-ship measurement. Its output decides whether Phase 2 ships.

---

## Guardrails (Phase 1)

### Must Have
- `--diagnose` completes in <1s (no evaluate.py re-run).
- Redaction happens *before* the note is written to disk (one write, already clean). Unit test pins the behavior.
- Clean-iteration path unaffected — no new subprocess spawns on success branch, no new env vars.
- Unit test asserts canonical `AssertionError: x=1.5 not in (0, 1)` lines are PRESERVED intact (Architect ask #5).

### Must NOT Have
- No modification to `evaluate.py`, `program.md`, `.omc/coordination/manifest.json`, `.omc/coordination/preflight.py`, or `data/eval/**` / `data/test/**`.
- No modification to the existing 5 structural checks in `verify_agent.py`.
- No session-id capture in Phase 1.
- No new claude invocation in Phase 1 (neither `claude -r` nor a re-primed `claude -p`).
- No oracle-signal leakage: redacted text must not contain `combined=<float>`, `combined_<dom>=<float>`, `splice_f1=<float>`, or `clean_score=<float>` substrings.
- No new `git reset --hard` call sites. No new result-status values in results.tsv.

---

## Task Flow (Phase 1)

```
hypothesis commit → evaluate.py → [exit 0 AND RESULTS_TSV ok] ──→ normal keep/discard (unchanged)
                                    │
                                    └─ crash (3 paths)
                                         │
                                         ▼
                                  verify_agent.py --diagnose
                                  (reads .omc/last_eval.log, redacts, classifies,
                                   writes 3-8 line note to .omc/last_reflection.md)
                                         │
                                         ▼
                                  _guarded_reset "$head_before"  (unchanged)
                                         │
                                         ▼
                                  _append_note "verify-fail" ...  (unchanged — now consumes
                                                                    diagnosis-bearing reflection)
                                         │
                                         ▼
                                  next iteration's claude sees diagnosis in
                                  .omc/research_notes.md prompt prefix
```

---

## Ordered Implementation Steps (Phase 1)

### Step 1 — Stop the running loop and snapshot baseline

**Files:** none modified. Tmux session `autoresearch` stopped.
**Delta:** 0 lines.
**Actions:**
- `./run_autoresearch.sh stop`
- Confirm tmux session ended, `.omc/autoresearch-stop` present.
- Capture current `results.tsv` tail (last 20 iterations) — baseline input to Step 3's repeat-rate measurement.

**Acceptance:** `tmux ls | grep autoresearch` returns empty. `results.tsv` has ≥20 rows for baseline.

---

### Step 2 — Add `verify_agent.py --diagnose` subcommand

**Files:** `/Users/yejunjang/Projects/mutual/autoresearch-splice/.omc/coordination/verify_agent.py`.
**Delta:** +90 / -5 lines (new `run_diagnose()` function, new argparse subcommand, small `main()` dispatch refactor, embedded `--self-test`).
**Actions:**
- Add `argparse` subparsers: default path stays as today (strict-improvement verify); new `--diagnose` subcommand takes no required args.
- **Critic-fix (APPEND-not-OVERWRITE):** `.omc/last_reflection.md` is written by the claude subprocess at the start of its iteration (hypothesis reflection — "what I'm trying / why / next if this fails"). If `--diagnose` overwrites this file, claude's own reasoning context is lost before `_append_note` can capture it into `research_notes.md`. `run_diagnose` MUST open the output in append mode (`"a"`) and prefix its classification block with `\n\n---\n## [auto-diagnosis]\n` so the two-voice entry reads cleanly. Add acceptance check: after a simulated crash, `.omc/last_reflection.md` contains BOTH claude's original reflection AND the diagnosis block, in that order.
- Implement `run_diagnose(log_path=".omc/last_eval.log", out_path=".omc/last_reflection.md")`:
  1. Read log file. If missing: write a 2-line `diagnose: log-missing` stub and exit 0.
  2. Scan for the **last** occurrence of `Traceback (most recent call last):`. Capture from that line through the exception line (non-indented line after the frames).
  3. Extract: exception class (first token before `:` on exception line), exception message (rest of line), top-3 frame `file:line` tuples from `File "..."` lines.
  4. Classify into one of: `KeyError-missing-ctx`, `SyntaxError`, `ImportError`, `NameError`, `AssertionError`, `Other`. `KeyError-missing-ctx` = `KeyError` where the missing key name (captured from the message) does NOT appear in `features.py` FEATURE_NAMES.
  5. **Redaction (Architect ask #5, option b):** apply denylist regex `re.sub(r'\b(combined(?:_[a-z]+)?|splice_f1|clean_score)\s*=\s*[0-9.]+', r'\1=<REDACTED>', text)` to the captured traceback text. Run redaction BEFORE writing.
  6. Write 3-8 lines to `.omc/last_reflection.md` **in APPEND mode**, prefixed with `\n\n---\n## [auto-diagnosis]\n`:
     ```
     diagnose: <ClassifiedCategory>
     exception: <ExceptionClass>: <redacted message>
     top_frames:
       <file:line>
       <file:line>
       <file:line>
     note: (1-2 line suggested action derived from classification)
     ```
  7. Print single-line summary to stdout, exit 0.
- Add embedded sanity check behind `if __name__ == "__main__" and "--self-test" in sys.argv`, containing **two synthetic cases**:
  - **Case A (oracle-stripping):** synthetic log contains `combined=0.47 splice_f1=0.33 clean_score=0.91 combined_singing=0.55` inside a traceback → assert output file contains `<REDACTED>` for each and no `=<digit>` occurrences on any of the four keys.
  - **Case B (legitimate-float preservation, Architect ask #5):** synthetic log contains `AssertionError: x=1.5 not in (0, 1)` and `ValueError: threshold=0.72 outside [0, 1]` inside tracebacks → assert the `=1.5`, `=0.72` floats are **preserved verbatim** in the output. This is the pinned anti-regression for the denylist.
- Timing: wrap `run_diagnose` body with `t0 = time.monotonic()`; log elapsed; if >1.0s, print warning to stderr (soft budget).

**Acceptance:**
- `uv run python .omc/coordination/verify_agent.py --diagnose` runs against a crashed log and produces 3-8 line output at `.omc/last_reflection.md`.
- `uv run python .omc/coordination/verify_agent.py --self-test` exits 0. Both Case A and Case B pass.
- Output file contains no `combined=<float>`, `combined_<dom>=<float>`, `splice_f1=<float>`, or `clean_score=<float>` substrings across Case A synthetic input.
- Wall time <1s on a 1 MB eval log.

---

### Step 3 — Wire the three crash paths + ship success-metric script

**Files:**
- `/Users/yejunjang/Projects/mutual/autoresearch-splice/run_autoresearch.sh`
- new `/Users/yejunjang/Projects/mutual/autoresearch-splice/scripts/diagnose_repeat_rate.py`

**Delta:** wrapper +3 / -0; script +80 / -0.

**Actions (wrapper):**
- At each of the three existing crash sites in `run_autoresearch.sh`, insert ONE line immediately before the existing `_guarded_reset`/`_append_note` pair:
  ```bash
  uv run python .omc/coordination/verify_agent.py --diagnose >> "$LOG_FILE" 2>&1 || true
  ```
  Sites (line numbers approximate per current HEAD):
  - ~615 (retrain_fail, before the existing `_guarded_reset "$head_before"` + `_append_note "verify-fail"`).
  - ~647-651 (eval_exit != 0, between the `tail -20 ... >> "$LOG_FILE"` and `_guarded_reset "$head_before"`).
  - ~659-662 (parse_fail, between the `reported=...` parse attempt failing and `_guarded_reset "$head_before"`).
- The `|| true` guards ensure a `--diagnose` bug never blocks the existing reset path. If `--diagnose` fails, the log shows it and the iteration proceeds exactly as today.
- No other wrapper edits. No helper functions. No new env vars. No new result-status values.

**Actions (success-metric script):**
- Script reads `results.tsv` and the last 20 entries of `.omc/research_notes.md`.
- For each `verify-fail` row in the trailing 20-iteration window, extract the exception class + top-frame `file:line` from the matching `.omc/research_notes.md` entry (the `_append_note` consumption of `.omc/last_reflection.md` preserves diagnosis there as plain text).
- Compute: *repeat rate* = fraction of verify-fails whose (exception_class, top_frame) pair matches another verify-fail within 5 iterations.
- CLI: `uv run python scripts/diagnose_repeat_rate.py [--window 20] [--proximity 5]`. Output: `repeat_rate: <float>`, a breakdown by exception class, and a list of repeat offenders.
- Record baseline value pre-ship (captured from results.tsv at Step 1). **Phase 2 trigger criterion:** if post-ship trailing-20 repeat-rate has *not* dropped ≥50% relative to baseline, Phase 2 is authorized to ship.

**Acceptance:**
- `grep -c 'verify_agent.py --diagnose' run_autoresearch.sh` returns 3 (one per crash site). `grep -c 'git reset --hard' run_autoresearch.sh` returns 1 (the single existing line inside `_guarded_reset`).
- Clean-iteration byte diff: the success branch (eval exit 0, RESULTS_TSV parse ok) has NO changed lines — the 3 new lines sit strictly inside the three crash branches.
- `scripts/diagnose_repeat_rate.py` runs against current `results.tsv`, prints a numeric repeat rate + breakdown. Baseline value recorded below in "Measurement Record" section.
- Restart loop: `rm -f .omc/autoresearch-stop && ./run_autoresearch.sh start`. Tail `.omc/autoresearch.log` until the next natural crash (or inject a synthetic one). Confirm: `--diagnose` fires, `.omc/last_reflection.md` contains a 3-8 line classification note, `.omc/research_notes.md` appended via existing `_append_note` path, next iteration's claude prompt prefix contains the diagnosis.

---

## Success Criteria (Phase 1, derived from spec)

- [ ] `verify_agent.py --diagnose` subcommand exists. Reads `.omc/last_eval.log`, finds last `Traceback (most recent call last):` block, extracts exception class + message + top-3 frame `file:line`, classifies into the 6-class taxonomy, writes 3-8 line note to `.omc/last_reflection.md`. Completes in <1s.
- [ ] Oracle-leak defense: `--diagnose` redacts the 4 named substrings before write. **Unit test pins two cases:** (A) oracle patterns stripped, (B) `AssertionError`/`ValueError` lines with legitimate floats preserved verbatim (Architect ask #5, option b).
- [ ] On the 3 crash paths: wrapper invokes `verify_agent.py --diagnose` before the existing `_guarded_reset` + `_append_note` pair. No other wrapper changes.
- [ ] `_append_note` picks up diagnosis from `.omc/last_reflection.md` automatically (existing US-503 plumbing).
- [ ] Success metric instrumented: `scripts/diagnose_repeat_rate.py` computes trailing-20-iteration repeat rate. Baseline captured pre-ship. **Phase 2 gate:** trigger only if repeat-rate has not dropped ≥50% within 20 iterations post-ship.
- [ ] All wrapper reset paths continue to use `_guarded_reset`. No new `git reset --hard` call sites. No new result-status values.
- [ ] Regression: clean-eval iteration is byte-identical (no extra calls, no slowdown). `--diagnose` only fires on crash paths.

**Deferred to Phase 2** (full spec coverage):
- Session-id capture + resume (`claude -r $SID -p "<diagnosis>"`).
- `fix:` commit on top of hypothesis, one re-run of evaluate.py.
- `verify-fail-postfix` status for post-fix failures.
- CLI probe script.
- `--session-id`/`-r` composability verification.

---

## Measurement Record

*(Updated during Step 1 + Step 3.)*

```
Baseline (pre-Phase-1-ship):
  Date: TBD
  Trailing-20 verify-fail count: TBD
  Trailing-20 repeat-rate: TBD

Post-Phase-1-ship (20 iterations after restart):
  Date: TBD
  Trailing-20 verify-fail count: TBD
  Trailing-20 repeat-rate: TBD
  Delta vs baseline: TBD
  Phase 2 trigger fired: TBD (yes if delta < 50% reduction, else no)
```

---

## Phase 2 Stub — Session-resume (deferred; conditional ship)

**Trigger:** Phase 1 measurement shows <50% repeat-rate reduction after 20 iterations.

**Scope when it ships:** Option A from the original plan revision (session-resume primary, re-prime fallback). The Phase 2 plan will re-open as a separate ralplan document and will address the Architect's asks #2, #3, #4 as follows — locked choices below, not re-opened for debate:

- **Ask #2 — `fix:` subject regex:** case-sensitive `^fix[:(]`. Matches `fix:` and `fix(scope):`. Rejects `Fix:`, `fix !`, non-colon-non-paren variants. Rationale: conventional-commit spec is lowercase; aligning with that spec avoids future linter friction.
- **Ask #3 — 2nd-eval crash overwriting first diagnosis:** option (b). Introduce append-only `.omc/diagnose-log.jsonl` with one record per `--diagnose` invocation (timestamp, iteration id, classification, top frames, redacted message). `.omc/last_reflection.md` remains single-writer-per-iteration (overwritten each call). Repeat-rate script reads from `.jsonl` going forward so both first and second crashes are counted independently.
- **Ask #4 — Probe must assert creation-time success:** the probe script must verify (i) `claude --session-id <uuid> -p "..."` exits 0 (creation-time acceptance, not silently ignored), AND (ii) `claude -r <same-uuid> -p "..."` sees prior context (resume works). Both assertions fail the probe if either fails. If probe fails, Phase 2 ships with re-prime-only path (no `--session-id` flag).
- **Loop-invariant:** `_rerun_eval_once` must NOT invoke `_diagnose_and_resume` recursively (one resume per hypothesis, hard cap).

Implementation artifacts (wrapper helpers, crash-site rewrites, probe script, unit tests for `fix:` regex and `.jsonl` append) get planned in the Phase 2 ralplan if/when triggered.

---

## ADR — Phase-split diagnosis-first, session-resume deferred behind a metric

### Decision
Phase 1: ship `verify_agent.py --diagnose` as a passive diagnosis writer whose output reaches claude via the existing `_append_note` → `.omc/research_notes.md` plumbing. Three one-line insertions in `run_autoresearch.sh` on the crash sites. One success-metric script. No session-id, no CLI probe, no resume, no fix-on-top.

Phase 2: session-resume + fix-on-top + single re-run of evaluate.py. **Conditionally shipped** iff Phase 1's repeat-rate metric shows <50% reduction after 20 iterations.

### Drivers
1. Architect's observation that Phase 1's test surface is ~1/2 of Phase 2's, and Phase 1 alone may resolve the primary failure mode (same-exception-class recurrence), making Phase 2 gold-plating in the good case.
2. User's deep-interview intent ("keep the door open") is satisfied by **sequencing with a locked trigger criterion**, not by shipping both paths together. No Phase 1 work is thrown away; the diagnosis note's schema is what Phase 2 will consume verbatim.
3. Oracle isolation, clean-iteration zero-regression, and `_guarded_reset`-only reset semantics are trivially preserved when the crash-path delta is 3 lines.

### Alternatives considered
- **Ship Phase 1 + Phase 2 together (original plan):** rejected per Architect ITERATE. 2x test surface without measurement evidence justifying it.
- **Ship Phase 2 only (skip Phase 1):** rejected — Phase 1 is a strict prerequisite for the success metric that gates Phase 2.
- **Ship Phase 1 and commit unconditionally to Phase 2:** rejected — erodes the metric-gate discipline. If Phase 1 solves the problem, Phase 2 should not ship.
- **Widen denylist to match `<word>=<float>` generally (Architect ask #5a):** rejected — too aggressive, strips legitimate floats in `AssertionError`/`ValueError` messages that carry bug-finding information.
- **Whitelist redaction (Architect ask #5c):** rejected — loses debuggability for numeric-bounds bugs. The denylist + pinned test combo balances safety and diagnostic value.

### Why chosen
Phase-split with a numeric gate respects both the user's intent (preservation pathway is not abandoned, merely sequenced) and the Architect's asks (smaller Phase 1 surface, deferred complexity behind evidence). The denylist + pinned anti-regression unit test (Case A + Case B) chooses Architect ask #5 option (b) explicitly and documents residual risk on Principle 2.

### Consequences
- **Positive:** Phase 1 ship is ~3 steps, ~40-50 python lines + ~3 wrapper lines; review surface is small; unit test pins the redaction behavior; no probe script; no new result statuses; clean iteration byte-identical. Measurement answers the "do we need Phase 2" question on empirical ground, not speculation.
- **Negative:** Phase 1 delays the diagnosis delivery by exactly one iteration (next claude sees the note, not same claude). If Phase 2 is eventually triggered, the wrapper gets a second edit pass — non-zero rework.
- **Neutral:** `.omc/last_reflection.md` now has two possible writers on crash paths (claude's own reflection, or `--diagnose`). Per-iteration, only one fires (crash → `--diagnose`; clean → claude's reflection path is unchanged). Overwrite semantics are deterministic.

### Follow-ups
1. **Phase 2 gate (primary):** after 20 post-ship iterations, run `scripts/diagnose_repeat_rate.py` and compare to baseline. If reduction <50%, open Phase 2 ralplan with locked decisions from "Phase 2 Stub" above. If reduction ≥50%, archive Phase 2 stub as "not needed."
2. **Residual oracle-leak risk (Principle 2):** when any future US introduces a new oracle key (e.g. `per_dataset_score_singing=`), the owning PR must widen the Step 2 denylist regex in the same change.
3. **Classification taxonomy:** if repeat-rate shows <20% reduction, the 6-class taxonomy may be too coarse — revisit regardless of whether Phase 2 is triggered.
4. **Redaction unit test drift:** if `evaluate.py` introduces new oracle print formats, Step 2's Case A must be extended. Link this ADR from the Step 2 unit test comment block so future editors find the rationale.

---

## Out-of-Scope (Phase 1, Explicit)

- Rewriting the existing 5 structural checks in `verify_agent.py`.
- Pre-eval phase gates (syntax/import/feature-consistency).
- Session-id capture (Phase 2).
- `claude -r` / `claude --session-id` CLI usage (Phase 2).
- `fix:` commits on top of hypothesis (Phase 2).
- Re-running evaluate.py after a crash (Phase 2).
- `verify-fail-postfix` result status (Phase 2).
- Probe script for CLI session behavior (Phase 2).
- Multi-iteration retry budgets.
- User-gated retry / manual triage interleaving.
- Changes to `evaluate.py`, `program.md`, `data/eval/**`, `data/test/**`, `.omc/coordination/manifest.json`, `.omc/coordination/preflight.py` (protected).

---

## File Reference (absolute paths)

Phase 1 (this plan, ships now):
- `/Users/yejunjang/Projects/mutual/autoresearch-splice/.omc/coordination/verify_agent.py` — add `--diagnose` subcommand + `--self-test` with Case A + Case B.
- `/Users/yejunjang/Projects/mutual/autoresearch-splice/run_autoresearch.sh` — 3 one-line `--diagnose` invocations at existing crash sites (lines ~615, ~647-651, ~659-662 per current HEAD).
- `/Users/yejunjang/Projects/mutual/autoresearch-splice/scripts/diagnose_repeat_rate.py` — new success-metric script; reads `results.tsv` + `.omc/research_notes.md`.
- `/Users/yejunjang/Projects/mutual/autoresearch-splice/.omc/last_eval.log` — read-only input to `--diagnose`.
- `/Users/yejunjang/Projects/mutual/autoresearch-splice/.omc/last_reflection.md` — write target of `--diagnose`; consumed by `_append_note` (unchanged).

Phase 2 (deferred, conditional ship):
- `/Users/yejunjang/Projects/mutual/autoresearch-splice/scripts/probe_claude_session.sh` — temporary probe.
- `/Users/yejunjang/Projects/mutual/autoresearch-splice/.omc/diagnose-log.jsonl` — append-only trail for 2nd-crash visibility (Architect ask #3).
- Further `run_autoresearch.sh` edits (session-id capture, `_diagnose_and_resume` helper, `_rerun_eval_once` helper, `verify-fail-postfix` status).
