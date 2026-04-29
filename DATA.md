# Data acquisition guide

This document covers how to get the audio corpora the loop needs. The
generated splits are **not in git** (1.4+ GB regenerable bytes) — only
ground-truth manifests + encrypted blobs ship with the repo. Use this
guide to either decrypt the shipped blobs (if you have the keys) or
regenerate equivalent corpora from publicly downloadable sources.

## Layout (post 2026-04-29 flatten)

```
data/
├── eval/korean_iter1/         ← what the loop optimizes against (~200 MB plaintext)
├── train/korean_iter1/        ← classifier training set (~380 MB plaintext)
├── test/korean_iter1/         ← held-out, never seen by the loop (~770 MB plaintext)
├── eval.tar.gz.enc            ← encrypted eval (AES-256-GCM, ~195 MB on disk)
├── test.tar.gz.enc            ← encrypted held-out test (Touch-ID-gated, ~1.0 GB)
└── sources/                   ← raw audio pools (~20 GB; only needed for regen)
    ├── singing_wav/
    ├── zeroth-korean/
    └── LibriSpeech/dev-clean/
```

## Paths to the data

### Path A — regenerate from public sources (works for any outsider)

The `scripts/regenerate_korean_iter1.py` script deterministically produces
all three splits from a Korean source corpus. Source pools:

1. **Zeroth-Korean (~20 GB, 115 speakers).** Public Korean read-speech corpus.
   Place under `data/sources/zeroth-korean/`. The repo expects the
   `korean-iter-1-delivery.tar` form; if you have the upstream Zeroth
   release instead, see `scripts/regenerate_korean_iter1.py --verify-source`
   for the structure check.
2. **LibriSpeech `dev-clean` (~340 MB, 40 speakers).** English read-speech.
   Used as the "english" dormant-dataset source pool — not consumed by the
   active iter1 loop, only by future cross-domain experiments. Download:
   <https://www.openslr.org/12>.
3. **Singing WAVs (~varies, 101 files).** Mix of public-domain singing
   recordings. Used by the dormant `singing` dataset. Not required for
   iter1 reproduction.

**Regen command** (Korean iter1 only — sufficient to reproduce the loop):

```bash
# Verify the source pool has the expected speakers + structure
PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --verify-source

# Regenerate train + eval + test with deterministic file-hash seeds.
# Takes ~30–60 min on a laptop CPU.
PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --regenerate

# Verify determinism: re-run on a sample, compare per-byte against the
# previous output. Useful before committing baseline shifts.
PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --verify-determinism
```

The augmentation chain (synthetic exp-decay RIR + pink noise + Opus 32 kbps
codec roundtrip) is baked into the corpus on disk — the loop does NOT
re-augment per iteration. Outputs:

- `data/eval/korean_iter1/conversation_*.opus` + matching `.json` boundary files
- `data/train/korean_iter1/conversation_*.opus` + matching `.json`
- `data/test/korean_iter1/conversation_*.opus` + matching `.json`
- `data/{eval,train,test}/korean_iter1/ground_truth.json` per split (aggregate manifest)
- `data/{eval,train,test}/korean_iter1/_manifest.jsonl` (per-conv regen telemetry)

### Path B — decrypt the shipped blobs (requires keys)

The repo ships `data/eval.tar.gz.enc` and `data/test.tar.gz.enc`. They are
**encrypted at rest** — not as security secrets, but for **research
integrity**: the loop runs Claude as a subprocess, and we don't want
Claude accidentally reading the eval data while forming hypotheses.

The keys live in macOS Keychain on the original author's machine. **For
strangers, this path is not available**; regenerate from sources (Path A)
instead. For future maintainers with key access:

```bash
# Eval: same-user keychain, no biometric prompt (loop decrypts 12×/hour)
PYTHONPATH=$PWD uv run python scripts/eval_crypto.py decrypt --keep
# → prints /tmp/.ar-eval-XXXXXX/eval/korean_iter1/  (use as OMC_EVAL_DATA_ROOT)

# Held-out test: Touch-ID-gated (harder threat model)
PYTHONPATH=$PWD uv run python scripts/test_crypto.py decrypt --keep
# → prints /tmp/.ar-test-XXXXXX/test/korean_iter1/  (Touch ID prompt fires)
```

The wrapper `run_autoresearch.sh` handles decrypt automatically per
iteration — operators only need to manually decrypt for retest workflows
(see `CLAUDE.md` § Retest).

### Path C — skip the test gate

If you only want to run the optimization loop (eval-set iterations) and
don't need the held-out overfitting check: regen everything anyway (Path
A produces all three splits in one pass), then just delete `data/test/`
or leave it alone — the loop's `keep` / `discard` decisions don't depend
on the test gate. The wrapper fires `scripts/test_eval.py` every 10 keeps
as an overfitting alarm only; it never blocks a keep.

To suppress the test_eval fire entirely, comment out its invocation in
`run_autoresearch.sh` (search for `test_eval` in the wrapper).

## After data is in place

Run `autoresearch/preflight.py` to verify file counts, ground-truth
hashes, and audio byte-shas:

```bash
PYTHONPATH=$PWD uv run python autoresearch/preflight.py
```

Expected output: `All checks passed.` Mismatches are flagged as ERROR
(missing files, hash drift) or WARNING (writable dirs that should be
chmod 555 for protection).

## Encryption setup (for new operators)

If you regenerated the corpora yourself and want the same isolation
posture as the original repo:

```bash
# One-time: generate fresh keys, encrypt the splits, shred plaintext
PYTHONPATH=$PWD uv run python scripts/eval_crypto.py setup
PYTHONPATH=$PWD uv run python scripts/test_crypto.py setup    # Touch ID prompt fires once

# Subsequent re-encrypts (after regen): same key, no prompt for eval, Touch ID for test
PYTHONPATH=$PWD uv run python scripts/eval_crypto.py encrypt
PYTHONPATH=$PWD uv run python scripts/test_crypto.py encrypt
```

The Swift helper `scripts/test_unlock` is the biometric gate for the test
key; `test_crypto.py` shells out to it. Recompile from
`scripts/test_unlock.swift` if the binary doesn't run on your platform:

```bash
swiftc -O scripts/test_unlock.swift -o scripts/test_unlock
```

Linux operators: the keychain integration is macOS-only. For Linux,
either skip encryption (Path A produces plaintext that the loop can read
directly), or substitute equivalent OS keystore + biometric flows.

## Voice-pair holdout (fixed across all paths)

Regardless of how you produced the data, the speaker assignment is fixed:

| Split | Voices | Files | Purpose |
|-------|--------|-------|---------|
| `test` | DaeBuHo, Kanna | ~4278 | Held-out overfitting alarm; never seen during optimization |
| `eval` | Sunwoo, Joon | ~1030 | Loop's primary metric target |
| `train` | ChloeCha, DangchanYeo, Donghyun, Eunha, Minho, Minwoo, Ondo | ~2052 | Classifier training |

These assignments live in `autoresearch/manifest.json` and
`splice/dataset_registry.py`. The loop's verifier asserts them on every
iteration.
