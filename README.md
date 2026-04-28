# autoresearch-splice

**Autonomous research for audio splice detection using classical signal
processing.** A Korean-language same-source splice detector grown by an LLM
agent through ~500 commits of hypothesis → train → eval → keep/discard
iterations against a 250 ms collar boundary-F1 metric.

This repo is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
that pivoted off the nanochat LLM-training scaffold and used the same loop
structure to drive a fundamentally different problem: detecting where
spoken-audio recordings have been cut and re-stitched, with no neural
networks and a CPU-only scipy/librosa/numpy stack so every detection is
explainable.

> The original `autoresearch` README — Karpathy's nanochat narrative — is
> preserved verbatim at [`README-upstream.md`](./README-upstream.md). It
> describes the agent-driven research scaffold this fork inherits.

## What this project does

Given an `<conv_id>.opus` recording of one or more Korean speakers, the
detector returns `list[(time_s, label)]` where each entry is a predicted
splice boundary with a 3-class label:

- `cross_voice` — splice between two different speakers
- `same_voice_edit` — same-speaker word-level cut
- `unknown` — boundary suspected but class evidence is weak

A boundary is correct if it lands within ±250 ms of a ground-truth boundary
and the label matches. The agent optimizes a precision-weighted aggregate:

```
combined = F0.5(precision, recall) × clean_fp_penalty
clean_fp_penalty = 1 / (1 + clean_fp_per_min / 1.0)
```

`clean_fp_per_min` counts unmatched predictions in clean (non-spliced)
sections. The formulation pushes the loop toward a forensic-audit
detector: high-precision, low false-alarm in clean speech.

## How it works

The loop is a thin bash wrapper around three modifiable Python files plus
one immutable evaluator:

- **`splice/evaluate.py`** *(protected — only the human edits this)* —
  evaluation oracle. Reads the eval split, runs `splice/detector.py` on
  each file, computes boundary-F1 with 250 ms collar + clean-FP penalty,
  prints metrics. Wrapper greps the last `combined: <f>` line for the
  keep/discard decision.
- **`splice/program.md`** *(protected)* — the prompt context the agent
  reads at the top of every iteration. Explains the metric, constraints
  (no neural networks, CPU-only, every detection must be explainable),
  and the file map.
- **`splice/detector.py`** — the splice detector. Returns `list[(t, label)]`.
  Edited freely by the agent for tunable thresholds (`GBM_THRESHOLD`,
  `GBM_MIN_SEP_S`, isolation gates) and sliding-window geometry.
- **`splice/features.py`** — feature extractor (~80 hand-crafted spectral
  + DSP features) feeding the GBM. Edited to add new features; requires
  retrain afterwards.
- **`splice/classifier/train_classifier.py`** — 3-class HistGradientBoosting
  trainer with paired negative sampling. Hyperparameters are agent-tunable;
  retraining writes a fresh `fp_classifier.joblib`.
- **`run_autoresearch.sh`** — the loop. Spawns the agent, applies its
  hypothesis, retrains if needed, runs eval, decides keep / discard /
  verify-fail, writes a journal commit (`hypothesis: …`, `note: discard …`,
  `baseline: …`), repeats.

The interesting bit isn't the detector — it's that **no human wrote the
detector**. The agent built it incrementally over ~500 commits, each one
a single hypothesis (e.g., "raise GBM_MIN_SEP_S 5.5 → 6.0", "add
boundary_onset_chunk_relative_ratio FE") with a paragraph of mechanism,
risk, and prior-art reasoning. The git log IS the research narrative.

## Quick start

**Requirements:** macOS or Linux, Python 3.10+, [uv](https://docs.astral.sh/uv/),
~10 GB free disk for source pools + decrypted corpora, Touch ID for the
held-out test gate (macOS only).

```bash
# 1. Install dependencies
uv sync

# 2. Get the data — read DATA.md for the full story.
#    Short version: download zeroth-korean + LibriSpeech + singing source pools,
#    then run the regen script to produce the splits.
PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --regenerate

# 3. (Optional) encrypt the eval and held-out test splits to mirror the
#    isolation model the loop uses. Without encryption, the loop still
#    works; the encryption is research-integrity hygiene to prevent the
#    LLM subprocess from accidentally seeing oracle data.
PYTHONPATH=$PWD uv run python scripts/eval_crypto.py setup
PYTHONPATH=$PWD uv run python scripts/test_crypto.py setup    # Touch ID prompt

# 4. Run the loop
./run_autoresearch.sh start
./run_autoresearch.sh status
tmux attach -t autoresearch    # to watch
./run_autoresearch.sh stop     # graceful — finishes current iteration first
```

Per-iteration commits land on the current branch. Successful keeps update
`autoresearch/baseline_metrics.json`; discards become `note: …` commits
that record the failed hypothesis. Browse `git log` to read the research.

## Project structure

```
autoresearch-splice/
├── README.md                   ← this file
├── README-upstream.md          ← Karpathy's original autoresearch README (preserved)
├── DATA.md                     ← outsider data-acquisition guide
├── LICENSE                     ← MIT (this fork) + upstream-MIT acknowledgement
├── splice/                     ← the audio-splice application
│   ├── evaluate.py             ← protected metric oracle
│   ├── program.md              ← protected agent prompt
│   ├── detector.py             ← agent edits this
│   ├── features.py             ← agent edits this
│   ├── classifier/             ← agent edits training hyperparams
│   ├── dataset_registry.py     ← single source of truth for split paths
│   └── tests/                  ← unit tests for the evaluator and trainer
├── autoresearch/               ← reusable runtime harness (loop, logger, supervisor)
│   ├── supervisor_agent.py     ← verifier + maintainer; runs after every iteration
│   ├── logger.py               ← unified structured logging (US-515)
│   ├── preflight.py            ← dataset integrity check
│   └── manifest.json           ← protected dataset manifest
├── scripts/                    ← operator tools: dashboard, regen, crypto, test_eval, …
├── data/                       ← (gitignored) eval/train/test corpora, encrypted blobs
├── run_autoresearch.sh         ← the loop wrapper
└── pyproject.toml              ← uv-managed dependencies
```

`splice/` is the application; `autoresearch/` is reusable scaffolding —
swapping `splice/` for a different domain detector should produce a
working agent-driven research loop for that domain.

## Constraints (non-negotiable)

- **No neural networks.** No torch / TF / sklearn MLPs. HistGradientBoosting
  is the only allowed classifier. Detector code is hand-crafted DSP +
  statistical tests.
- **CPU only.** No GPU calls anywhere.
- **Every detection must be explainable** — attributable to a specific
  statistical test or spectral anomaly.
- **240 s eval budget** on a laptop CPU; **1200 s held-out test budget**.
- **Voice-pair holdout (fixed):** test = {DaeBuHo, Kanna}; eval = {Sunwoo,
  Joon}; train = remaining 7 voices. The agent never sees test voices
  during the optimization loop.
- **`splice/evaluate.py` and `splice/program.md` are protected** — only
  the human edits them. The verifier audits diffs and reverts agent
  modifications to these files automatically.

## Research history

Run `git log --oneline autoresearch/korean-iter1` for the full journal
of ~500 hypothesis / discard / keep commits. Each `hypothesis:` commit
contains a paragraph-long mechanism explanation; each `note: discard`
records a failed iteration with a brief reason. The autoresearch
notebook compaction also lives in `.omc/research_notes.md` as the
agent's compressed memory across sessions.

Current Korean-iter1 baseline: `combined = 0.151237` at SHA `c35ba8e`
(metric-v2 first keep). See `autoresearch/baseline_metrics.json` for
the full metric breakdown.

## Acknowledgements

This repository is a fork of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the
loop structure, the `program.md` agent-prompt convention, and the keep /
discard / verify-fail journal pattern are all upstream ideas applied
here to a different problem. Karpathy's original README is at
[`README-upstream.md`](./README-upstream.md). Thanks to Andrej for the
scaffolding and the framing.

## License

MIT — see [`LICENSE`](./LICENSE). Compatible with the upstream MIT.
