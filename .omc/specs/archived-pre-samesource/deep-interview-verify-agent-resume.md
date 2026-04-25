# Deep Interview Spec: verify_agent --diagnose + session-resume for crashed hypotheses

## Metadata
- Interview ID: verify-agent-resume-2026-04-18
- Rounds: 2
- Final Ambiguity Score: 16%
- Type: brownfield
- Generated: 2026-04-18
- Threshold: 20%
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|---|---|---|---|
| Goal Clarity | 0.85 | 0.35 | 0.30 |
| Constraint Clarity | 0.80 | 0.25 | 0.20 |
| Success Criteria | 0.85 | 0.25 | 0.21 |
| Context Clarity | 0.85 | 0.15 | 0.13 |
| **Total Clarity** | | | **0.84** |
| **Ambiguity** | | | **16%** |

## Goal
When the autoresearch loop's post-hypothesis evaluation crashes (evaluate.py exits non-zero,
RESULTS_TSV fails to parse, or the auto-retrain from US-505 crashes), the wrapper invokes
`verify_agent.py --diagnose` to extract + classify the error from `.omc/last_eval.log`, then
RESUMES the same claude session that formed the hypothesis with the diagnosis as additional
context. Claude's resumed call sees its own prior reasoning plus the bug report and commits a
targeted `fix:` on top. The wrapper re-runs evaluate.py once; if clean, the normal keep/discard
path proceeds; if still broken, verify-fail + `_guarded_reset` reverts hypothesis+fix together.

## Constraints
- Session resume uses the claude CLI's `-r <session-id>` / `--session-id` flags (verify support
  at implementation time; fallback is re-priming with prior prompt + commit diff + diagnosis).
- Retry budget is exactly ONE resume per hypothesis. If the fix still fails eval, verify-fail
  and move on — no infinite fix loops.
- `verify_agent.py --diagnose` MUST NOT re-run evaluate.py. It only reads the existing log
  (sub-second operation). Re-running would defeat the speed benefit AND risk the same crash.
- Oracle isolation must be preserved through the resume: the diagnosis sent to claude includes
  the traceback tail but NOT per-domain combined values or scoring intermediates. Redaction
  happens inside `--diagnose` before the note is written.
- `_guarded_reset` (commit e2fe680) stays unchanged — it already handles hypothesis+fix rollback
  correctly because both commits sit on top of `$head_before`.
- The feature does NOT replace the existing strict-improvement `verify_agent` call. That path
  is untouched.

## Non-Goals
- Do NOT run verify_agent after every eval (only after crashes). The user explicitly rejected
  this: "after every eval seems like too much - just do better error handling and in case it
  errors call the verify agent".
- Do NOT spawn a separate "debug claude" subprocess with a fresh context. The user's key
  insight: "resume the previous agent ... I would want that if I were the agent". A fresh
  subprocess would lose the hypothesis-forming reasoning thread.
- Do NOT attempt multi-round fix iterations. One resume, then give up.
- Do NOT implement pre-eval phase gates (syntax/import/feature-consistency static checks).
  Discussed and deferred — session resume is the cleaner primitive.
- Do NOT user-gate the retry. The loop must stay autonomous per the session's directive.

## Acceptance Criteria
- [ ] `verify_agent.py --diagnose` subcommand exists. Reads `.omc/last_eval.log`, finds the last
      Python `Traceback (most recent call last):` block, extracts the exception class + message +
      top-3 frame file:line, classifies into {KeyError-missing-ctx, SyntaxError, ImportError,
      NameError, AssertionError, Other}, writes a 3-8 line note to
      `.omc/last_reflection.md`. Completes in <1s.
- [ ] Oracle-leak defense: `--diagnose` redacts any `combined=<float>`, `combined_<domain>=<float>`,
      `splice_f1=<float>`, or `clean_score=<float>` substrings from the traceback text before
      writing. Unit-style sanity run confirms redaction.
- [ ] `run_autoresearch.sh` captures a stable session-id per hypothesis iteration
      (`uuidgen` or `python -c "import uuid; print(uuid.uuid4())"`) and passes it to
      `claude -p --session-id $SID` (or the supported equivalent; fallback: re-prime).
- [ ] On the 3 crash paths (eval exit != 0, no RESULTS_TSV, retrain rc != 0): wrapper invokes
      `verify_agent.py --diagnose`, then resumes `claude -r $SID -p "<diagnosis + fix directive>"`
      with the diagnosis. Claude is instructed: "Fix the bug, commit as `fix:`, exit. Do NOT
      change the detector parameters you already set."
- [ ] After the resumed claude commits a `fix:`, wrapper re-runs evaluate.py ONCE.
      On success → normal keep/discard routing. On failure → `log_to_results_tsv verify-fail`,
      `_guarded_reset "$head_before"` (reverts both the hypothesis AND the fix commits), next
      iteration.
- [ ] If the resume fails (no session-id support, CLI error, or claude produces no commit):
      fallback to the current behavior — verify-fail + `_guarded_reset` immediately. Log the
      fallback reason for triage.
- [ ] Session-id is NOT persisted across iterations (fresh uuid per hypothesis). It's only
      valid for the one resume-on-crash scenario.
- [ ] `_append_note` picks up the diagnosis from `.omc/last_reflection.md` automatically
      (existing plumbing from US-503). Research notes accumulate diagnoses for visibility.
- [ ] Success metric instrumented: `scripts/diagnose_repeat_rate.py` computes, for a trailing
      20-iteration window in results.tsv, the fraction of verify-fails whose exception class +
      top-frame file:line matches a prior verify-fail within 5 iterations. Baseline before
      ship; target ≥50% reduction within 20 iterations after ship.
- [ ] All new wrapper reset paths continue to use `_guarded_reset` (the fix from commit e2fe680).
      No new `git reset --hard` call sites.
- [ ] Regression: an iteration where evaluate.py succeeds cleanly is unaffected (no extra calls,
      no slowdown). `--diagnose` only fires on crash paths.

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|---|---|---|
| Calling verify_agent on every eval is worth the cost | Too expensive (~3 min per call) | Only call on crash paths. Strict-improvement verify_agent stays. |
| Passive note-in-research-notes is sufficient | Loses broken-but-good structural ideas when claude "moves on" | Use session resume: claude keeps its reasoning + sees the bug → targeted fix |
| We need a separate "debug subprocess" | Loses claude's prior reasoning thread | Resume existing session instead of spawning new one |
| Traceback text is safe to forward | Could contain partial oracle signal (mid-eval combined values) | Redact `combined=*`, `splice_f1=*`, `clean_score=*` patterns in --diagnose |
| Retry loops are unbounded risk | Could thrash forever | One resume max; 2nd failure = verify-fail, discard, move on |

## Technical Context
Wrapper: `run_autoresearch.sh` (post-US-503..507, commit e2fe680 "_guarded_reset"). The three
crash paths to hook are at lines 612-618 (eval_exit != 0), 645-651 (no RESULTS_TSV parseable),
and 658-662 (retrain rc != 0, US-505).

Existing verify_agent: `.omc/coordination/verify_agent.py` already has 5 structural checks. The
`--diagnose` subcommand is an additive code path; existing HIGH/MEDIUM/LOW confidence flow is
untouched.

Claude CLI session flags: need verification at implementation — the CLI supports `-c, --continue`
and `-r, --resume [sessionId]`. For headless scripted resume with a pre-chosen id, either
`--session-id <uuid>` at creation + `-r <uuid>` at resume, or parse the session id from the
first call's output + reuse. Fallback if neither works: re-prime claude with a composite prompt
containing the original hypothesis context + the diagnosis (same information content, loses
only claude's internal tool-use trace).

Guard infrastructure: `_guarded_reset` (commit e2fe680) already protects wrapper/scripts edits
during resets. No changes needed for this feature; `_GUARD_PATHS` already covers
`run_autoresearch.sh` and `scripts/`.

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|---|---|---|---|
| HypothesisSession | core | session_id (uuid), hypothesis_commit_sha, claude_exit_code | owned by one Iteration; resumed on crash |
| DiagnosisNote | supporting | exception_class, file_line, traceback_tail, redacted_text | produced by --diagnose; consumed by resumed HypothesisSession; archived into ResearchNote |
| CrashPath | supporting | type ∈ {eval_exit, parse_fail, retrain_fail}, trigger_line | emits DiagnosisNote |
| Iteration | core | head_before, head_after, outcome | contains one HypothesisSession; may have one FixCommit |
| FixCommit | supporting | sha, parent=hypothesis_commit | lives on top of HypothesisSession's commit; reverted together via _guarded_reset on 2nd failure |

## Ontology Convergence
| Round | Entities | New | Changed | Stable | Stability |
|---|---|---|---|---|---|
| 1 | HypothesisSession, DiagnosisNote, Iteration | 3 | 0 | 0 | N/A |
| 2 | HypothesisSession, DiagnosisNote, CrashPath, Iteration, FixCommit | 2 | 0 | 3 | 60% |

Entity set grew as resume semantics were explicitly defined. Stability ratio moderate — expected
since the core entity (Session) was nameless in Round 1. Full convergence will come from the
ralplan pass.

## Interview Transcript
<details>
<summary>Full Q&A (2 rounds)</summary>

### Round 1
**Q:** How do we know --diagnose is actually helping? Pick the operational signal that would
convince you the feature is worth keeping (or rolling back if missing).
**A:** Lower verify-fail repeat rate.
**Ambiguity:** 26% (Goal: 0.75, Constraints: 0.55, Criteria: 0.85, Context: 0.85)

### Round 2
**Q:** After --diagnose writes the note, how does the loop use it? (Options: passive note /
active fix-and-retry / user-gated retry)
**A:** "Can we just resume the agent with additional context about the evaluator's bug?
I would want that if I were the agent."
**Resolution:** Session-resume approach adopted. Neither of the three presented options
captured it — the user's framing is architecturally superior because it preserves claude's
own reasoning thread.
**Ambiguity:** 16% (Goal: 0.85, Constraints: 0.80, Criteria: 0.85, Context: 0.85) — threshold met.
</details>
