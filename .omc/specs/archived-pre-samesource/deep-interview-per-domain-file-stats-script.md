# deep-interview: per-domain-file-stats-script

_Auto-drafted by supervisor_agent.py --maintain on 2026-04-19T16:04Z_
_Source: enhancement-backlog.md entry `per-domain-file-stats-script` (request_count=3, risk=low (new script in scripts/; no existing code paths changed), category=new-script)_

---

## Goal

<!-- Paste the excerpt below and refine into a concrete goal statement -->

"A `scripts/per_domain_file_stats.py` emitting mean HPR/flatness/voicing per domain on EVAL (not training) so file-level gating thresholds come from the evaluation distribution."

## Constraints

<!-- From backlog notes -->

Walk `data/eval/{singing,korean,english}/{tier1,tier2,clean}/`, compute per-file HPR (via librosa.decompose.hpss), flatness, voicing; report per-domain mean + quantile distribution. Enables the agent to design file-level routing gates using eval-corpus statistics instead of training-set heuristics. Pure read-only analysis tool; safe to auto-implement.

## Non-Goals

<!-- Fill in -->

## Acceptance Criteria

<!-- Fill in -->

## Open Questions

<!-- Fill in -->
