#!/usr/bin/env python3
"""Corpus regenerator for korean-iter1 (Step 3 skeleton).

Reads /Volumes/HIKSEMI/korean-iter-1-delivery.tar, assigns conversations to
train/eval/test by voice-pair holdout, injects same-voice word-cuts at
silence-bounded positions, applies a deterministic augmentation chain
(pink noise + synthetic RIR + Opus 32k roundtrip), and writes per-conversation
audio + ground-truth JSON under data/eval/korean_iter1/{train,eval,test}/.

Only --verify-source is implemented in this skeleton (Step 3).
The full pipeline is Step 6.

Usage:
    PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --verify-source
    PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --regenerate
    PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --smoke 5
    PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --resume
    PYTHONPATH=$PWD uv run python scripts/regenerate_korean_iter1.py --verify-determinism

Events emitted (taxonomy whitelist for this script):
    regen.verify_source.start
    regen.verify_source.diversity_table
    regen.verify_source.summary
    regen.verify_source.error
    regen.regenerate.start
    regen.regenerate.conv.done
    regen.regenerate.complete
    regen.smoke.start
    regen.smoke.complete
    regen.determinism.start
    regen.determinism.result
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal
import soundfile as sf

# autoresearch.logger resolves via PYTHONPATH=$PWD
from autoresearch.logger import get_logger

log = get_logger("regen")

_DEFAULT_TARBALL = Path("/Volumes/HIKSEMI/korean-iter-1-delivery.tar")
_DEFAULT_OUTPUT_ROOT = Path("data/eval/korean_iter1")
# test = M middle-aged gyeongsang + F young seoul (max acoustic diversity)
# eval = M middle-aged seoul + M young seoul (kept from spec; eval gender-balanced via train mix)
# train = remaining 7 voices: 4 M (Minwoo, Donghyun, Minho, Ondo) + 3 F (ChloeCha, DangchanYeo, Eunha)
# Original spec defaults {DaeBuHo, Ondo, Sunwoo, Joon} all-male; corrected at Step 3 verify-source.
_DEFAULT_VOICES_TEST = "DaeBuHo,Kanna"
_DEFAULT_VOICES_EVAL = "Sunwoo,Joon"
_DEFAULT_SEED_BASE = 0


# ---------------------------------------------------------------------------
# Fully-implemented helpers (used by --verify-source)
# ---------------------------------------------------------------------------

def read_tarball_manifest(tarball: Path) -> dict:
    """Read manifest.json from the tarball root.

    Opens the tarball with tarfile, locates the top-level manifest.json
    member, and returns its parsed contents.  The expected manifest shape is:

        {
          "generated_at": "<ISO-8601>",
          "iteration": 1,
          "language": "korean",
          "total_conversations": 7360,
          "total_duration_ms": <int>,
          "format_split": {"wav": N, "flac": N, ...},
          "files": [
            {
              "id": "<uuid>",
              "idx": <int>,
              "format": "wav"|"flac"|...,
              "audio_file": "<path in tar>",
              "transcript_file": "<path in tar>",
              "topic": "<str>",
              "scenario": "<str>",
              "ambience": "<str>",
              "duration_ms": <int>,
              "speakers": [
                {
                  "speaker_id": "<str>",
                  "voice_name": "<str>",
                  "gender": "M"|"F",
                  "age": <int>,
                  "accent": "<str>"
                }, ...
              ],
              "turn_count": <int>
            }, ...
          ]
        }

    Raises FileNotFoundError if the tarball does not exist.
    Raises KeyError if manifest.json is not found inside the tarball.
    """
    if not tarball.exists():
        raise FileNotFoundError(f"Tarball not found: {tarball}")

    with tarfile.open(tarball, "r:*") as tf:
        # Accept manifest.json at any depth (e.g. "delivery/manifest.json")
        # but prefer the shortest path.
        candidates = [
            m for m in tf.getmembers()
            if m.name.endswith("manifest.json") and not m.isdir()
        ]
        if not candidates:
            raise KeyError("manifest.json not found in tarball")
        candidates.sort(key=lambda m: len(m.name))
        member = candidates[0]

        fobj = tf.extractfile(member)
        if fobj is None:
            raise KeyError(f"Cannot extract {member.name}")
        return json.load(fobj)


def voice_diversity_check(
    manifest: dict,
    voices_test: set[str],
    voices_eval: set[str],
) -> bool:
    """Print the voice diversity table and validate gender coverage.

    Walks manifest["files"], builds a per-voice catalog:
        {voice_name: {gender, age, accent, file_count, mean_duration_s}}

    Prints a markdown-style table with columns:
        voice_name | gender | age | accent | appearances | mean_dur_s | holdout

    Returns True if voices_test ∪ voices_eval spans both genders (at least one
    M and one F across the held-out set).  Returns False and logs ERROR
    otherwise.

    Also emits regen.verify_source.diversity_table with the per-voice catalog.
    """
    catalog: dict[str, dict[str, Any]] = {}

    for file_entry in manifest.get("files", []):
        dur_s = file_entry.get("duration_ms", 0) / 1000.0
        for spk in file_entry.get("speakers", []):
            vname: str = spk.get("voice_name", "unknown")
            if vname not in catalog:
                catalog[vname] = {
                    "gender": spk.get("gender", "?"),
                    "age": spk.get("age", 0),
                    "accent": spk.get("accent", "?"),
                    "file_count": 0,
                    "total_duration_s": 0.0,
                }
            catalog[vname]["file_count"] += 1
            catalog[vname]["total_duration_s"] += dur_s

    def _holdout_label(vname: str) -> str:
        if vname in voices_test:
            return "test"
        if vname in voices_eval:
            return "eval"
        return "train"

    # Print markdown table
    header = (
        f"{'voice_name':<20} {'gender':<8} {'age':<5} {'accent':<16} "
        f"{'appearances':<13} {'mean_dur_s':<12} {'holdout':<8}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for vname, info in sorted(catalog.items()):
        mean_dur = (
            info["total_duration_s"] / info["file_count"]
            if info["file_count"] > 0
            else 0.0
        )
        row = (
            f"{vname:<20} {info['gender']:<8} {info['age']:<5} "
            f"{info['accent']:<16} {info['file_count']:<13} "
            f"{mean_dur:<12.1f} {_holdout_label(vname):<8}"
        )
        print(row)
    print(separator)

    # Emit structured event
    catalog_kv: dict[str, Any] = {}
    for vname, info in catalog.items():
        mean_dur = (
            info["total_duration_s"] / info["file_count"]
            if info["file_count"] > 0
            else 0.0
        )
        catalog_kv[f"voice_{vname}_gender"] = info["gender"]
        catalog_kv[f"voice_{vname}_file_count"] = info["file_count"]
        catalog_kv[f"voice_{vname}_mean_dur_s"] = round(mean_dur, 2)
        catalog_kv[f"voice_{vname}_holdout"] = _holdout_label(vname)

    log.emit(
        "INFO",
        "regen.verify_source.diversity_table",
        voice_count=len(catalog),
        **catalog_kv,
    )

    # Gender coverage check over held-out voices
    held_out = voices_test | voices_eval
    held_out_genders = {
        catalog[v]["gender"]
        for v in held_out
        if v in catalog
    }
    # Manifest gender strings are "male"/"female", not "M"/"F".
    both_genders_present = {"male", "female"}.issubset(held_out_genders)

    if not both_genders_present:
        log.emit(
            "ERROR",
            "regen.verify_source.error",
            reason="held-out voices do not span both genders",
            held_out_voices=str(held_out),
            held_out_genders=str(held_out_genders),
        )
        print(
            f"\n[ERROR] held-out voices {held_out} cover genders {held_out_genders}; "
            "both M and F are required."
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Stub helpers (Step 6 implementation targets)
# ---------------------------------------------------------------------------

def assign_split(
    speaker_voice_names: list[str],
    voices_test: set[str],
    voices_eval: set[str],
) -> str:
    """Return 'train' | 'eval' | 'test'.

    A conversation belongs to test if ANY of its speakers' voices is in
    voices_test.  Same logic for eval.  Otherwise train.  Test takes
    precedence over eval (strictest holdout).
    """
    ...


def choose_edit_positions(
    word_alignment: list[dict],
    min_silence_ms: int = 120,
    n_edits: int = 1,
    rng: np.random.Generator = None,
) -> list[dict]:
    """Pick n_edits words from word_alignment at silence-bounded positions.

    Each picked word must have silence >= min_silence_ms BOTH before AND after
    it.  Returns a list of dicts, each containing the chosen word and its
    surrounding gap info.  n_edits is clamped to the number of qualifying
    words.  Picks are drawn without replacement using rng.
    """
    ...


def edit_audio(
    audio: np.ndarray,
    sr: int,
    edits: list[dict],
    turn_start_ms: int,
    turn_end_ms: int,
) -> tuple[np.ndarray, list[float]]:
    """Cut out the audio span for each edit and rejoin at the silence boundary.

    Returns (edited_audio, joined_times_seconds) where joined_times are the
    timestamps (in seconds, within the EDITED audio) where the cuts were joined.
    Edits are applied in reverse chronological order so earlier indices remain
    valid after each cut.
    """
    ...


def build_rir(t60_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic exponential-decay RIR.

    Models a shoebox room impulse response as a windowed exponential decay.
    t60_s is the -60 dB decay time in seconds.  Returns a 1-D float32 array
    normalised so its L2 norm equals 1.
    """
    ...


def build_pink_noise(
    n_samples: int, sr: int, rng: np.random.Generator
) -> np.ndarray:
    """Pink-filtered white noise, bandpass 100-6000 Hz, normalised to unit std.

    Uses a 1/f shaping filter approximated via scipy.signal IIR design, then
    a bandpass filter to limit the frequency range.  Returns a float32 array
    of length n_samples.
    """
    ...


def augment(
    audio: np.ndarray,
    sr: int,
    snr_db: float,
    t60_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """RIR convolution + pink noise injection at the requested SNR.

    Pipeline:
      1. build_rir -> convolve with audio (mode='full', trimmed to len(audio))
      2. build_pink_noise -> scale to achieve snr_db relative to reverbed audio
      3. add noise
      4. peak-normalise output to 0.95

    Returns a float32 array of the same length as audio.
    """
    ...


def opus_roundtrip(
    audio: np.ndarray,
    sr: int,
    bitrate: str = "32k",
) -> np.ndarray:
    """Encode-decode through libopus via ffmpeg pipes.

    Pipes raw PCM float32 into ffmpeg, encodes to Opus at the given bitrate,
    decodes back to raw PCM float32, and returns the decoded audio.  Uses
    subprocess with pipes; requires ffmpeg on PATH or at /opt/homebrew/bin/ffmpeg.
    """
    ...


def write_per_conv_outputs(
    out_dir: Path,
    conv_id: str,
    audio: np.ndarray,
    sr: int,
    boundaries: list[dict],
    augment_meta: dict,
) -> str:
    """Write {conv_id}.opus + {conv_id}.json to out_dir.

    The JSON ground-truth file records:
        {
          "conv_id": str,
          "boundaries": [{time_s, label}, ...],
          "augment_meta": {...},
          "sha256": "<hex>"
        }

    Returns the sha256 hex digest of the written .opus file.
    """
    ...


def synthesize_corpus_ground_truth(split_dir: Path) -> Path:
    """Aggregate per-conversation JSON files into a single ground_truth.json.

    Reads every {conv_id}.json under split_dir, merges into a dict keyed by
    conv_id with a list of {time_s, label} dicts, and writes
    split_dir/ground_truth.json.  Used by autoresearch/preflight.py.

    Format:
        {conv_id: [{time_s: float, label: str}, ...], ...}

    Returns the path to the written ground_truth.json.
    """
    ...


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_verify_source(args: argparse.Namespace) -> int:
    """Implement --verify-source: manifest read, diversity table, summary."""
    tarball = Path(args.tarball)
    voices_test: set[str] = set(args.voices_test.split(","))
    voices_eval: set[str] = set(args.voices_eval.split(","))

    log.emit(
        "INFO",
        "regen.verify_source.start",
        tarball=str(tarball),
        voices_test=str(voices_test),
        voices_eval=str(voices_eval),
    )

    print(f"Reading tarball manifest from: {tarball}")
    manifest = read_tarball_manifest(tarball)

    print(f"\n=== Voice Diversity Table ===")
    ok = voice_diversity_check(manifest, voices_test, voices_eval)

    # Summary statistics
    files = manifest.get("files", [])
    total_convs = manifest.get("total_conversations", len(files))
    total_dur_ms = manifest.get("total_duration_ms", sum(
        f.get("duration_ms", 0) for f in files
    ))
    total_dur_h = total_dur_ms / 3_600_000.0

    format_split: dict[str, int] = manifest.get("format_split", {})
    if not format_split:
        for f in files:
            fmt = f.get("format", "unknown")
            format_split[fmt] = format_split.get(fmt, 0) + 1

    ambience_dist: dict[str, int] = {}
    speaker_count_dist: dict[int, int] = {}
    for f in files:
        amb = f.get("ambience", "unknown")
        ambience_dist[amb] = ambience_dist.get(amb, 0) + 1
        n_spk = len(f.get("speakers", []))
        speaker_count_dist[n_spk] = speaker_count_dist.get(n_spk, 0) + 1

    print(f"\n=== Corpus Summary ===")
    print(f"  total_conversations : {total_convs}")
    print(f"  total_duration      : {total_dur_h:.2f} h ({total_dur_ms:,} ms)")
    print(f"  iteration           : {manifest.get('iteration', '?')}")
    print(f"  language            : {manifest.get('language', '?')}")
    print(f"  generated_at        : {manifest.get('generated_at', '?')}")
    print(f"\n  Format split:")
    for fmt, cnt in sorted(format_split.items()):
        print(f"    {fmt:<10}: {cnt}")
    print(f"\n  Ambience distribution:")
    for amb, cnt in sorted(ambience_dist.items(), key=lambda x: -x[1]):
        print(f"    {amb:<20}: {cnt}")
    print(f"\n  Speaker-count distribution:")
    for n_spk, cnt in sorted(speaker_count_dist.items()):
        print(f"    {n_spk} speaker(s)  : {cnt}")

    log.emit(
        "INFO",
        "regen.verify_source.summary",
        total_conversations=total_convs,
        total_duration_ms=total_dur_ms,
        format_split=str(format_split),
        ambience_count=len(ambience_dist),
        speaker_count_variants=len(speaker_count_dist),
    )

    if not ok:
        print("\n[FAIL] voice_diversity_check failed — held-out voices do not span both genders.")
        return 2

    print("\n[OK] voice_diversity_check passed.")
    return 0


def _cmd_regenerate(args: argparse.Namespace) -> int:
    """Stub for --regenerate (Step 6)."""
    print("not implemented in skeleton", file=sys.stderr)
    return 2


def _cmd_verify_determinism(args: argparse.Namespace) -> int:
    """Stub for --verify-determinism (Step 6)."""
    print("not implemented in skeleton", file=sys.stderr)
    return 2


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Stub for --smoke N (Step 6)."""
    print("not implemented in skeleton", file=sys.stderr)
    return 2


def _cmd_resume(args: argparse.Namespace) -> int:
    """Stub for --resume (Step 6)."""
    print("not implemented in skeleton", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="regenerate_korean_iter1.py",
        description="korean-iter1 corpus regenerator (skeleton, Step 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mutually-exclusive subcommands (--flag style)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--verify-source",
        action="store_true",
        help="Read tarball manifest, print voice diversity table, validate gender coverage.",
    )
    mode.add_argument(
        "--regenerate",
        action="store_true",
        help="Full pipeline (not implemented in skeleton).",
    )
    mode.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Re-run augmentation on 10 random files and byte-diff against on-disk version (stub).",
    )
    mode.add_argument(
        "--smoke",
        metavar="N",
        type=int,
        help="Regenerate N conversations end-to-end for development (stub).",
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already on disk (stub).",
    )

    # Common options
    p.add_argument(
        "--tarball",
        default=str(_DEFAULT_TARBALL),
        help=f"Path to the source tarball. Default: {_DEFAULT_TARBALL}",
    )
    p.add_argument(
        "--output-root",
        default=str(_DEFAULT_OUTPUT_ROOT),
        dest="output_root",
        help=f"Destination root directory. Default: {_DEFAULT_OUTPUT_ROOT}",
    )
    p.add_argument(
        "--voices-test",
        default=_DEFAULT_VOICES_TEST,
        dest="voices_test",
        help=(
            "Comma-separated voice names held out for test. "
            f"Default: {_DEFAULT_VOICES_TEST}"
        ),
    )
    p.add_argument(
        "--voices-eval",
        default=_DEFAULT_VOICES_EVAL,
        dest="voices_eval",
        help=(
            "Comma-separated voice names held out for eval. "
            f"Default: {_DEFAULT_VOICES_EVAL}"
        ),
    )
    p.add_argument(
        "--seed-base",
        default=_DEFAULT_SEED_BASE,
        type=int,
        dest="seed_base",
        help=(
            "Base random seed. Per-file seed = (seed_base ^ hash(file_id)) & 0xFFFFFFFF. "
            f"Default: {_DEFAULT_SEED_BASE}"
        ),
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verify_source:
        sys.exit(_cmd_verify_source(args))
    elif args.regenerate:
        sys.exit(_cmd_regenerate(args))
    elif args.verify_determinism:
        sys.exit(_cmd_verify_determinism(args))
    elif args.smoke is not None:
        sys.exit(_cmd_smoke(args))
    elif args.resume:
        sys.exit(_cmd_resume(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
