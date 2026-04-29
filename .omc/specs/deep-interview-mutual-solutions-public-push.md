# Deep Interview Spec: Public push to mutual-solutions/autoresearch-splice

## Metadata
- Interview ID: di-20260428-mutual-solutions-public-push
- Rounds: 4
- Final Ambiguity Score: 10%
- Type: brownfield
- Generated: 2026-04-28
- Threshold: 20%
- Status: PASSED (under threshold at round 4)

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.95 | 0.35 | 0.333 |
| Constraint Clarity | 0.85 | 0.25 | 0.213 |
| Success Criteria | 0.90 | 0.25 | 0.225 |
| Context Clarity (brownfield) | 0.85 | 0.15 | 0.128 |
| **Total Clarity** | | | **0.898** |
| **Ambiguity** | | | **0.102 (10%)** |

## Goal
Publish this repo as `mutual-solutions/autoresearch-splice` (public) such that an outsider who clones the repo can read the README, follow `DATA.md`, install the env via `uv`, and run the full autoresearch loop end-to-end on the audio-splice problem. The repo presents itself as an **independent fork of `karpathy/autoresearch` with visible attribution** — Karpathy's history is preserved, his README is referenced (not deleted), and the top-level README is rewritten to describe the audio-splice work that actually lives in the tree. All local + remote-tracking branches get pushed to the new remote.

## Constraints
- **Upstream relationship**: derivative-work / fork-with-attribution. Karpathy's history is preserved in full. His README content stays referenced (e.g., as `README-upstream.md` or in an `ACKNOWLEDGEMENTS` / `CREDITS` section). License chosen must be compatible with upstream (MIT default — upstream-permissive).
- **History scope**: no rebases, no squashes, no rewrites. The 506 autoresearch-journal commits (`note: discard X`, `hypothesis: …`, `baseline: …`) stay as-is — the journal IS part of the artifact for the reproducible-research framing.
- **Cleanup level: Light** (per round-4 scope choice):
  - README rewrite (audio-splice oriented, with upstream attribution)
  - Add `LICENSE` (MIT, copyright holder = the user)
  - Add `DATA.md` (outsider data-acquisition story: source pools, regen flow, encrypted blobs explained)
  - Scrub the 9 `/Users/<name>` absolute-path leaks in tracked `.omc/plans/*.md` and `.omc/evaluate_integration.md` (replace with relative paths or `<project-root>` placeholder)
- **Branches to push**: all 5 — `master`, `autoresearch/apr15`, `autoresearch/korean-iter1` (3 local) + `agenthub`, `exp/H100/mar8` (2 currently remote-tracking-only from upstream `origin`; need to be checked out or pushed via `git push mutual refs/remotes/origin/<branch>:refs/heads/<branch>` so they exist on the new remote).
- **Visibility**: public.
- **Pre-push state**: working tree clean before `git push -u`. Autoresearch loop must be gracefully stopped (`./run_autoresearch.sh stop` + wait for current iteration) before any tracked-file edits — per project memory rule on never force-killing iterations.
- **Secrets**: tracked tree has been scanned (no API keys / tokens / authtokens). The encrypted blobs (`data/eval.tar.gz.enc`, `data/test.tar.gz.enc`) are gitignored and stay out.
- **Ngrok tunnel**: still live serving `~/Desktop/{eval,train}.zip` — out-of-scope for this spec, untouched by it.

## Non-Goals
- Sanitizing tracked `.omc/research_notes.md` or `.omc/plans/*.md` content (Medium scope, not chosen).
- Smoke-testing the loop on a fresh checkout (Heavy scope, not chosen) — the README will document the path; we're not verifying it actually runs.
- Removing the 31 already-tracked `detector_v1.py … detector_v31.py` snapshots (Heavy scope, not chosen).
- Rewriting / squashing the 506 autoresearch-journal commits.
- Sanitizing `CLAUDE.md` content (it's tracked and contains audio-splice agent rules — kept as-is for reproducibility).
- Adding GitHub Actions / CI / docs site / contribution guide.

## Acceptance Criteria
- [ ] Loop is gracefully stopped before any file edit (`.omc/autoresearch-stop` sentinel present, current iteration finished).
- [ ] `LICENSE` exists at the repo root, MIT, copyright `<year> <name>`.
- [ ] `README.md` describes the audio-splice work (project, results, quickstart pointer to DATA.md), with a clear "Acknowledgements / Upstream" section linking to `karpathy/autoresearch` and explaining the fork relationship. Karpathy's original README content is preserved either inline (clearly demarcated) or relocated to `README-upstream.md`.
- [ ] `DATA.md` exists at the repo root and documents:
  - source-pool acquisition (zeroth-korean, LibriSpeech dev-clean, singing WAVs) with links + commands;
  - the `data_synth/regenerate_datasets.py` flow that produces eval/train/test splits;
  - what `data/eval.tar.gz.enc` and `data/test.tar.gz.enc` are for (research-integrity prevention, not security secret) and how an outsider can either decrypt them, regenerate equivalent corpora, or skip the held-out gate.
- [ ] Zero `/Users/<name>` strings remain in tracked text files (verified by `git ls-files | xargs grep -l '/Users/<name>'` → empty).
- [ ] All 4 changes (README, LICENSE, DATA.md, path-scrubs) committed in a coherent commit (or small commit series) with `Co-Authored-By: Claude` trailer.
- [ ] `gh repo create mutual-solutions/autoresearch-splice --public --source=. --remote=mutual` succeeds.
- [ ] `git push -u mutual` pushes the 5 branches (master, autoresearch/apr15, autoresearch/korean-iter1, agenthub, exp/H100/mar8). Verified via `gh repo view mutual-solutions/autoresearch-splice --json defaultBranchRef,refs`.
- [ ] No secrets land on the public remote (post-push grep verifying).

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| "Clean up" = generic polish | Round 1 audience question | Audience = strangers reproducing the loop end-to-end — defines a concrete coherence bar |
| The repo IS the upstream nanochat training scaffold | Round 2 upstream question | The tree is a fork that pivoted away from nanochat to audio-splice — README rewrite required |
| All 506 journal commits should be squashed | Round 2 upstream question | Journal is preserved — it's part of the artifact for "reproducible research drop" framing |
| Reader just needs to read | Round 3 reader-journey question | Reader needs to RUN the full loop — DATA.md, env setup, classifier weights all become required |
| Cleanup should aggressively sanitize internal `.omc/` | Round 4 scope question | Light scope only — preserve `.omc/research_notes.md` + plans as-is (they're part of what makes the journal valuable) |
| License is up for debate | Round 4 default | MIT, upstream-compatible, default unless user objects |

## Technical Context
- **Repo**: `/Users/<name>/Projects/mutual/autoresearch-splice` on branch `autoresearch/korean-iter1`, HEAD `0a81909` (chore commit just landed).
- **Upstream**: `origin = https://github.com/karpathy/autoresearch.git` (read-only for this user — `mutantQ` GitHub identity has no write access there).
- **New remote**: `mutual = https://github.com/mutual-solutions/autoresearch-splice.git` (to be created via `gh repo create`).
- **Branches**:
  - `master` (upstream Karpathy work, ~unchanged)
  - `autoresearch/apr15` (older work branch)
  - `autoresearch/korean-iter1` (current, 506 commits ahead of master)
  - `agenthub`, `exp/H100/mar8` (currently only refs/remotes/origin/* — need explicit handling)
- **Working tree**: clean (post `0a81909`); `progress.txt` will churn again as the loop heartbeat resumes.
- **Loop state**: 🟢 RUNNING in tmux session `autoresearch` — must `./run_autoresearch.sh stop` before this work, resume after push (or leave stopped if user is done iterating).
- **Tracked-file PII / leakage scan**:
  - Secrets: clean ✓
  - `/Users/<name>` absolute paths: 9 files (`.omc/evaluate_integration.md`, 8 of `.omc/plans/*.md`)
  - Non-user-author emails in history: pre-fork upstream contributors (Karpathy + others) — preserved by design
- **Ngrok tunnel**: `https://decoratively-conidial-sadye.ngrok-free.dev` → `localhost:8765` serving `~/Desktop/{eval,train}.zip`. Out-of-scope for this spec.

## Ontology (Key Entities, final round)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| Repo | core | branches, history, working tree | hosted on `mutual-solutions` remote |
| Audio-splice project | core domain | detector, classifier, evaluator, journal | the actual subject of the public repo |
| Reader | persona | reproducer | runs the full loop end-to-end |
| Reproduction journey | scenario | clone → uv → data → run | drives README + DATA.md scope |
| README | artifact | top-level Markdown | rewritten for audio-splice |
| LICENSE | artifact | MIT text | new file at root |
| DATA.md | artifact | data acquisition Markdown | new file at root |
| Upstream README (Karpathy) | artifact | nanochat narrative | preserved as `README-upstream.md` or appended |
| Attribution / CREDITS | artifact | acknowledgements section | inside README |
| License compatibility | constraint | upstream-permissive requirement | bounds license choice (MIT) |
| CLAUDE.md | artifact | agent rules | tracked, kept as-is |
| Branches | artifact set | 5 refs | all pushed to mutual remote |
| Commit history | artifact | 506 journal commits + upstream | preserved, not rewritten |
| `.omc/` internal notes | artifact set | research_notes, plans, specs | tracked subset preserved (light editorial pass NOT in scope) |
| Trained classifier artifact | artifact | `splice/classifier/fp_classifier.joblib` (2.6 MB) | shipped in tree |
| Environment setup | artifact | `pyproject.toml`, `uv.lock`, `.python-version` | drives `uv sync` flow |
| Loop wrapper | artifact | `run_autoresearch.sh` | entry point |
| Data acquisition flow | scenario | sources → regen → splits | documented in DATA.md |
| Reproducibility goal | objective | fresh-clone → runnable loop | the central acceptance criterion |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability |
|-------|-------------|-----|---------|--------|-----------|
| 1 | 9 | 9 | - | - | N/A (baseline) |
| 2 | 12 | 3 | 0 | 9 | 75% |
| 3 | 18 | 6 | 0 | 12 | 67% (supporting detail; healthy) |
| 4 | 18 | 0 | 0 | 18 | 100% (converged) |

## Interview Transcript

<details>
<summary>Full Q&A (4 rounds)</summary>

### Round 1 — Goal Clarity
**Q:** Who is this public repo for, and what's its primary purpose?
**A:** Reproducible research drop.
**Ambiguity:** 61% (Goal 0.55, Constraints 0.20, Criteria 0.20, Context 0.65)

### Round 2 — Constraint Clarity
**Q:** How should the new repo relate to the upstream `karpathy/autoresearch` fork?
**A:** Independent fork w/ attribution.
**Ambiguity:** 44% (Goal 0.70, Constraints 0.55, Criteria 0.30, Context 0.70)

### Round 3 — Success Criteria
**Q:** What should a stranger be able to do end-to-end after cloning the repo?
**A:** Read + reproduce full loop (highest bar).
**Ambiguity:** 20.2% (Goal 0.90, Constraints 0.60, Criteria 0.85, Context 0.80)

### Round 4 — Constraints + scope-of-action
**Q:** Pick the cleanup scope (Light / Medium / Heavy / Surgical). License defaulted to MIT.
**A:** Light (Recommended).
**Ambiguity:** 10% (Goal 0.95, Constraints 0.85, Criteria 0.90, Context 0.85). **Threshold met.**

</details>
