#!/usr/bin/env python3
"""Corpus regenerator for korean-iter1 (Step 6 implementation).

Reads /Volumes/HIKSEMI/korean-iter-1-delivery.tar, assigns conversations to
train/eval/test by voice-pair holdout, injects same-voice word-cuts at
silence-bounded positions, applies a deterministic augmentation chain
(synthetic RIR + pink noise + Opus 32k roundtrip), and writes per-conversation
audio + ground-truth JSON under data/eval/korean_iter1/{train,eval,test}/.

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
    regen.regenerate.conv.skip
    regen.regenerate.conv.error
    regen.regenerate.complete
    regen.regenerate.summary
    regen.smoke.start
    regen.smoke.complete
    regen.determinism.start
    regen.determinism.sample
    regen.determinism.result
    regen.synthesize_gt.start
    regen.synthesize_gt.done
    regen.augment.warn_clip
    regen.augment.warn_duration_drift
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
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

# ffmpeg binary from .omc/korean-iter1-env-fingerprint.json
_FFMPEG_BIN = "/opt/homebrew/bin/ffmpeg"
if not Path(_FFMPEG_BIN).exists():  # graceful fallback for non-default installs
    _FFMPEG_BIN = shutil.which("ffmpeg") or _FFMPEG_BIN

_TARGET_SR = 44100
_MIN_SILENCE_MS = 120


# ---------------------------------------------------------------------------
# Fully-implemented helpers (used by --verify-source)
# ---------------------------------------------------------------------------

def read_tarball_manifest(source: Path) -> dict:
    """Read manifest.json from a corpus source.

    `source` may be either a tarball file (.tar) OR an extracted directory
    (the directory must contain `manifest.json` at its root or one level deep).
    The directory mode bypasses tar seek-contention and is recommended for
    multi-worker regen runs from external/USB storage.

    Raises FileNotFoundError if the source does not exist.
    Raises KeyError if manifest.json is not found.
    """
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if source.is_dir():
        # Search for manifest.json at root then one level deep.
        candidates = list(source.glob("manifest.json")) + list(source.glob("*/manifest.json"))
        if not candidates:
            raise KeyError(f"manifest.json not found under {source}")
        candidates.sort(key=lambda p: len(str(p)))
        return json.loads(candidates[0].read_text())

    with tarfile.open(source, "r:*") as tf:
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


def _source_root_for_member(source: Path, member_name: str) -> Path:
    """Resolve a tarball member name to its on-disk path under an extracted source.

    Tarball members like 'audio/conversation_00088.wav' map to
    `<source>/audio/conversation_00088.wav` if extracted at root, OR to
    `<source>/<top>/audio/conversation_00088.wav` if there's a single
    top-level dir. This helper handles both layouts.
    """
    direct = source / member_name
    if direct.exists():
        return direct
    # Try one level deep (extractor may have preserved a top-level dir)
    for top in source.iterdir():
        if top.is_dir():
            candidate = top / member_name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Member {member_name} not found under {source}")


def voice_diversity_check(
    manifest: dict,
    voices_test: set[str],
    voices_eval: set[str],
) -> bool:
    """Print the voice diversity table and validate gender coverage."""
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

    held_out = voices_test | voices_eval
    held_out_genders = {
        catalog[v]["gender"]
        for v in held_out
        if v in catalog
    }
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
# Step 6 helpers (the regenerator pipeline)
# ---------------------------------------------------------------------------

def assign_split(
    speaker_voice_names: list[str],
    voices_test: set[str],
    voices_eval: set[str],
) -> str:
    """Return 'train' | 'eval' | 'test'.

    Test takes precedence over eval (strictest holdout). A conversation
    belongs to test if ANY of its speakers' voices is in voices_test;
    otherwise eval if any is in voices_eval; otherwise train.
    """
    voices = set(speaker_voice_names)
    if voices & voices_test:
        return "test"
    if voices & voices_eval:
        return "eval"
    return "train"


def _file_seed(file_id: str, seed_base: int) -> int:
    """Per-file deterministic seed.

    seed = (seed_base ^ first 4 bytes of sha256(file_id)) & 0xFFFFFFFF
    """
    digest = hashlib.sha256(file_id.encode("utf-8")).digest()[:4]
    return (seed_base ^ int.from_bytes(digest, "big")) & 0xFFFFFFFF


def _flatten_word_alignment(turns: list[dict]) -> list[dict]:
    """Flatten transcript turns into a list of words with absolute timestamps.

    BUG FIX (2026-04-25): the iter-1 manifest's word_alignment start_ms /
    end_ms fields are ABSOLUTE timestamps (i.e., already offset within the
    full audio), NOT relative-to-turn. Earlier code added turn.start_ms,
    double-counting and shifting cuts ~turn_start_ms forward — landing some
    cuts in inter-turn silences (cutting nothing meaningful) and others
    inside unrelated words (e.g. mid-syllable "지난번" → "지번"). Verified
    by comparing turn 5's first word_alignment.start_ms (21688) against
    its turn.start_ms (21478) — they're nearly equal, confirming absolute.

    Each returned dict carries:
        word, start_ms, end_ms, turn_idx, prev_end_ms, next_start_ms
        (all in ABSOLUTE-audio-time milliseconds).
    """
    out: list[dict] = []
    for turn in turns:
        wa = turn.get("word_alignment") or []
        turn_start = int(turn.get("start_ms", 0))
        turn_end = int(turn.get("end_ms", 0))
        for j, w in enumerate(wa):
            ws = int(w.get("start_ms", 0))
            we = int(w.get("end_ms", 0))
            prev_end = (
                int(wa[j - 1].get("end_ms", 0))
                if j > 0 else turn_start
            )
            next_start = (
                int(wa[j + 1].get("start_ms", 0))
                if j + 1 < len(wa) else turn_end
            )
            out.append({
                "word": w.get("word", ""),
                "start_ms": ws,
                "end_ms": we,
                "turn_idx": int(turn.get("idx", 0)),
                "prev_end_ms": prev_end,
                "next_start_ms": next_start,
            })
    return out


def choose_edit_positions(
    word_alignment: list[dict],
    min_silence_ms: int = _MIN_SILENCE_MS,
    n_edits: int = 1,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Pick n_edits words from word_alignment at silence-bounded positions.

    `word_alignment` is the flattened list produced by `_flatten_word_alignment`
    OR a list with explicit `prev_end_ms` / `next_start_ms` keys.

    Each picked word has gap_before_ms >= min_silence_ms AND
    gap_after_ms >= min_silence_ms. Returns [] if fewer than n_edits eligible
    words exist. Picks are drawn without replacement using rng.
    """
    if rng is None:
        rng = np.random.default_rng()

    eligible_idx: list[int] = []
    for i, w in enumerate(word_alignment):
        gap_before = int(w["start_ms"]) - int(w["prev_end_ms"])
        gap_after = int(w["next_start_ms"]) - int(w["end_ms"])
        if gap_before >= min_silence_ms and gap_after >= min_silence_ms:
            eligible_idx.append(i)

    if len(eligible_idx) < n_edits:
        return []

    chosen_idx = rng.choice(
        np.array(eligible_idx, dtype=np.int64), size=n_edits, replace=False
    )
    chosen = [word_alignment[i] for i in chosen_idx.tolist()]
    chosen.sort(key=lambda w: int(w["start_ms"]))
    return chosen


def edit_audio(
    audio: np.ndarray,
    sr: int,
    edits: list[dict],
) -> tuple[np.ndarray, list[tuple[float, str]]]:
    """Cut out the audio span for each edit; return edited audio + GT joins.

    Edits are processed earliest-first. For each cut at original `start_ms`,
    the post-cut join position is `start_ms - cumulative_shift_ms`. After
    the cut, `cumulative_shift_ms += (end_ms - start_ms)`.

    Returns (edited_audio, [(post_cut_time_s, "same_voice_edit"), ...]).
    """
    if not edits:
        return audio, []

    edits_sorted = sorted(edits, key=lambda w: int(w["start_ms"]))

    # Build the kept-segment slices in original-sample coordinates, then
    # concatenate. Track join timestamps in post-cut seconds.
    keeps: list[tuple[int, int]] = []
    joins_post_s: list[tuple[float, str]] = []
    cursor = 0  # in original samples
    cumulative_shift_ms = 0
    for w in edits_sorted:
        start_ms = int(w["start_ms"])
        end_ms = int(w["end_ms"])
        start_sample = int(round(start_ms / 1000.0 * sr))
        end_sample = int(round(end_ms / 1000.0 * sr))
        # Clamp into bounds
        start_sample = max(0, min(start_sample, audio.shape[0]))
        end_sample = max(start_sample, min(end_sample, audio.shape[0]))
        if start_sample > cursor:
            keeps.append((cursor, start_sample))
        post_cut_ms = start_ms - cumulative_shift_ms
        joins_post_s.append((post_cut_ms / 1000.0, "same_voice_edit"))
        cumulative_shift_ms += (end_ms - start_ms)
        cursor = end_sample
    if cursor < audio.shape[0]:
        keeps.append((cursor, audio.shape[0]))

    if not keeps:
        edited = audio[:0].copy()
    else:
        edited = np.concatenate([audio[a:b] for a, b in keeps])
    return edited, joins_post_s


def _shift_at(orig_time_s: float, edits: list[dict]) -> float:
    """Cumulative duration (s) removed at original times <= orig_time_s.

    Used to remap cross-voice GT timestamps from original to post-cut
    coordinates. Each edit removes (end_ms - start_ms); edits whose cut
    fully precedes `orig_time_s` (start_ms < orig_time_ms) contribute
    their full duration. Edits straddling `orig_time_s` are unusual since
    cross-voice splices are at silence-bounded turn boundaries — we still
    handle the case by taking max(0, orig_time_ms - start_ms) for the
    straddling edit.
    """
    orig_time_ms = orig_time_s * 1000.0
    shift_ms = 0.0
    for w in edits:
        s = int(w["start_ms"])
        e = int(w["end_ms"])
        if e <= orig_time_ms:
            shift_ms += (e - s)
        elif s < orig_time_ms < e:
            shift_ms += (orig_time_ms - s)
        else:
            break
    return shift_ms / 1000.0


def build_rir(t60_s: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic exponential-decay RIR.

    decay = exp(-6 * arange(n) / n)  (so the envelope hits ~ -60 dB at sample n)
    rir = randn(n) * decay
    rir[0] += 1.0
    Returns the unit-max-normalized impulse response.
    """
    n = max(1, int(t60_s * sr))
    decay = np.exp(-6.0 * np.arange(n) / n)
    rir = rng.standard_normal(n).astype(np.float64) * decay
    rir[0] += 1.0
    peak = float(np.max(np.abs(rir)))
    if peak > 0:
        rir = rir / peak
    return rir.astype(np.float32)


def build_pink_noise(
    n_samples: int, sr: int, rng: np.random.Generator
) -> np.ndarray:
    """Bandpass-filtered white noise (100-6000 Hz), normalized to unit std.

    A 2nd-order Butterworth bandpass acting on white noise. (The original
    spec calls this "pink" — strict 1/f shaping is approximated by the
    bandpass low cutoff suppressing the very low end and the high cutoff
    rolling off above 6 kHz.)
    """
    n_samples = max(1, int(n_samples))
    white = rng.standard_normal(n_samples).astype(np.float64)
    sos = scipy.signal.butter(2, [100, 6000], btype="bandpass", fs=sr, output="sos")
    filtered = scipy.signal.sosfilt(sos, white)
    std = float(np.std(filtered))
    if std > 0:
        filtered = filtered / std
    return filtered.astype(np.float32)


def augment(
    audio: np.ndarray,
    sr: int,
    snr_db: float,
    t60_s: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """RIR convolution + pink noise injection at the requested SNR.

    Returns (augmented_audio, meta) where meta tracks invariant outcomes
    (n_clipped_samples, rms_db, peak_db).
    """
    audio_f = audio.astype(np.float64, copy=False)

    rir = build_rir(t60_s, sr, rng)
    reverbed = scipy.signal.fftconvolve(audio_f, rir.astype(np.float64), mode="full")
    reverbed = reverbed[: audio_f.shape[0]]

    signal_rms = float(np.sqrt(np.mean(reverbed * reverbed) + 1e-20))
    noise = build_pink_noise(reverbed.shape[0], sr, rng).astype(np.float64)
    noise_std = float(np.std(noise))
    if noise_std <= 0:
        noise_scaled = np.zeros_like(noise)
    else:
        target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        noise_scaled = noise * (target_noise_rms / noise_std)

    noisy = reverbed + noise_scaled

    # Invariants
    if not np.isfinite(noisy).all():
        # Replace NaN/Inf with zeros and warn
        n_bad = int((~np.isfinite(noisy)).sum())
        log.emit(
            "WARN",
            "regen.augment.warn_clip",
            reason="non_finite_samples",
            n_bad=n_bad,
        )
        noisy = np.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)

    n_clipped = int((np.abs(noisy) >= 0.999).sum())
    if n_clipped > 0:
        peak = float(np.max(np.abs(noisy)))
        log.emit(
            "WARN",
            "regen.augment.warn_clip",
            reason="pre_norm_clip",
            n_clipped_samples=n_clipped,
            peak=peak,
        )

    # Peak-normalize to 0.95 final
    peak = float(np.max(np.abs(noisy)))
    if peak > 0:
        noisy = noisy * (0.95 / peak)

    rms = float(np.sqrt(np.mean(noisy * noisy) + 1e-20))
    rms_db = 20.0 * np.log10(rms + 1e-20)
    peak_db = 20.0 * np.log10(float(np.max(np.abs(noisy))) + 1e-20)

    meta = {
        "snr_db": float(snr_db),
        "t60_s": float(t60_s),
        "codec": "opus_32k",
        "n_clipped_samples": n_clipped,
        "rms_db": round(rms_db, 2),
        "peak_db": round(peak_db, 2),
    }
    return noisy.astype(np.float32), meta


# --- OGG page rewriter (for deterministic Opus output) ---
# ffmpeg's Ogg muxer assigns a random 32-bit bitstream serial number per
# encode session (libavformat ogg_init -> av_get_random_seed). Two runs of
# the same PCM input thus produce different bytes at offset 14-17 of every
# page (and a different page CRC at offset 22-25). To get byte-identical
# encodes (required by --verify-determinism), we walk every Ogg page and
# (a) overwrite the serial number with a fixed value and
# (b) recompute the page CRC.
# The Ogg page CRC uses polynomial 0x04C11DB7, MSB-first, no reflection.
def _ogg_crc_table() -> list[int]:
    table: list[int] = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            if r & 0x80000000:
                r = (r << 1) ^ 0x04C11DB7
            else:
                r <<= 1
            r &= 0xFFFFFFFF
        table.append(r)
    return table


_OGG_CRC_TABLE = _ogg_crc_table()


def _ogg_crc(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = ((crc << 8) ^ _OGG_CRC_TABLE[((crc >> 24) ^ b) & 0xFF]) & 0xFFFFFFFF
    return crc


def _normalize_ogg_serial(data: bytes, serial: int = 0) -> bytes:
    """Rewrite every Ogg page's serial number to `serial` and fix the CRC.

    Returns a new byte string of identical length. Non-Ogg-page bytes (none
    in valid input) are passed through untouched.
    """
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        if data[i:i + 4] != b"OggS":
            # Should not happen for well-formed Ogg; pass through.
            out.append(data[i])
            i += 1
            continue
        # Header is 27 bytes + n_segs segment lengths + sum(segments) body.
        if i + 27 > n:
            out.extend(data[i:])
            break
        n_segs = data[i + 26]
        seg_table_end = i + 27 + n_segs
        if seg_table_end > n:
            out.extend(data[i:])
            break
        body_len = sum(data[i + 27:seg_table_end])
        page_len = 27 + n_segs + body_len
        if i + page_len > n:
            out.extend(data[i:])
            break
        page = bytearray(data[i:i + page_len])
        page[14:18] = serial.to_bytes(4, "little")
        page[22:26] = b"\x00\x00\x00\x00"
        crc = _ogg_crc(bytes(page))
        page[22:26] = crc.to_bytes(4, "little")
        out.extend(page)
        i += page_len
    return bytes(out)


def opus_roundtrip(
    audio: np.ndarray,
    sr: int,
    bitrate: str = "32k",
) -> tuple[np.ndarray, bytes]:
    """Encode-decode through libopus via ffmpeg pipes.

    Returns (decoded_float32, encoded_ogg_opus_bytes).
    The encoded bytes are what we write to disk as `<conv_id>.opus`. They are
    post-processed to use a stable Ogg bitstream serial number so two runs of
    the same input produce byte-identical output.
    """
    pcm_int16 = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm_int16 * 32767.0).astype(np.int16)
    pcm_bytes = pcm_int16.tobytes()

    enc_cmd = [
        _FFMPEG_BIN,
        "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(sr), "-ac", "1",
        "-i", "pipe:0",
        "-c:a", "libopus", "-b:a", bitrate,
        "-f", "ogg", "pipe:1",
    ]
    enc = subprocess.run(
        enc_cmd, input=pcm_bytes, capture_output=True, check=False
    )
    if enc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg opus encode failed (rc={enc.returncode}): "
            f"{enc.stderr.decode('utf-8', 'replace')[:400]}"
        )
    encoded = _normalize_ogg_serial(enc.stdout, serial=0)

    dec_cmd = [
        _FFMPEG_BIN,
        "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le", "-ar", str(sr), "-ac", "1", "pipe:1",
    ]
    dec = subprocess.run(
        dec_cmd, input=encoded, capture_output=True, check=False
    )
    if dec.returncode != 0:
        raise RuntimeError(
            f"ffmpeg opus decode failed (rc={dec.returncode}): "
            f"{dec.stderr.decode('utf-8', 'replace')[:400]}"
        )
    decoded_int16 = np.frombuffer(dec.stdout, dtype=np.int16)
    decoded = decoded_int16.astype(np.float32) / 32768.0

    dur_in = audio.shape[0] / float(sr)
    dur_out = decoded.shape[0] / float(sr)
    ratio = dur_out / dur_in if dur_in > 0 else 0.0
    if not (0.95 <= ratio <= 1.05):
        log.emit(
            "ERROR",
            "regen.augment.warn_duration_drift",
            duration_in_s=round(dur_in, 4),
            duration_out_s=round(dur_out, 4),
            ratio=round(ratio, 4),
        )
        raise RuntimeError(
            f"opus_duration_drift: in={dur_in:.3f}s out={dur_out:.3f}s ratio={ratio:.3f}"
        )

    return decoded, encoded


def write_per_conv_outputs(
    out_dir: Path,
    conv_id: str,
    audio: np.ndarray,
    sr: int,
    boundaries: list[dict],
    augment_meta: dict,
    split: str,
    voice_holdout: dict[str, list[str]],
    augment_seed: int,
    encoded_opus_bytes: bytes,
) -> str:
    """Write {conv_id}.opus + {conv_id}.json + append _manifest.jsonl row.

    The .opus file is the EXACT bytes from the encoder (no re-encode).
    Returns the sha256 hex digest of the written .opus file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    opus_path = out_dir / f"{conv_id}.opus"
    json_path = out_dir / f"{conv_id}.json"
    manifest_path = out_dir / "_manifest.jsonl"

    # Atomic-ish opus write: write to .tmp then rename
    tmp_opus = opus_path.with_suffix(".opus.tmp")
    tmp_opus.write_bytes(encoded_opus_bytes)
    os.replace(tmp_opus, opus_path)

    audio_sha = hashlib.sha256(encoded_opus_bytes).hexdigest()

    duration_s = float(audio.shape[0]) / float(sr)
    json_payload = {
        "conv_id": conv_id,
        "split": split,
        "duration_s": round(duration_s, 4),
        "sr": int(sr),
        "boundaries": [
            {"time_s": round(float(b["time_s"]), 4), "label": b["label"]}
            for b in boundaries
        ],
        "augment_meta": augment_meta,
    }
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2))
    os.replace(tmp_json, json_path)

    manifest_row = {
        "conv_id": conv_id,
        "split": split,
        "voice_holdout": voice_holdout,
        "augment_seed": int(augment_seed),
        "n_boundaries": len(boundaries),
        "audio_sha256": audio_sha,
    }
    # Append-only — supplementary writers may interleave; the file is
    # rewritten on resume to dedupe (see `--resume` and `_rewrite_manifest`).
    with open(manifest_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

    return audio_sha


def synthesize_corpus_ground_truth(split_dir: Path) -> Path:
    """Aggregate per-conversation .json files into ground_truth.json.

    Walks every <conv_id>.json under split_dir, merges into a dict keyed by
    conv_id with a list of {time_s, label} dicts, and writes
    `split_dir/ground_truth.json`. Returns the written path.
    """
    log.emit("INFO", "regen.synthesize_gt.start", split_dir=str(split_dir))

    out_path = split_dir / "ground_truth.json"
    aggregate: dict[str, list[dict]] = {}

    for jp in sorted(split_dir.glob("*.json")):
        if jp.name == "ground_truth.json":
            continue
        try:
            data = json.loads(jp.read_text())
        except json.JSONDecodeError:
            continue
        conv_id = data.get("conv_id") or jp.stem
        boundaries = data.get("boundaries", []) or []
        # Re-emit with stable schema; preserve only the two keys preflight needs.
        aggregate[conv_id] = [
            {"time_s": float(b["time_s"]), "label": str(b["label"])}
            for b in boundaries
        ]

    # Stable key order so the SHA prefix is deterministic.
    ordered = {k: aggregate[k] for k in sorted(aggregate)}
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2))
    os.replace(tmp_path, out_path)

    log.emit(
        "INFO",
        "regen.synthesize_gt.done",
        split_dir=str(split_dir),
        n_files=len(ordered),
        out_path=str(out_path),
    )
    return out_path


# ---------------------------------------------------------------------------
# Per-conversation worker
# ---------------------------------------------------------------------------

def _decode_audio_bytes(
    raw: bytes, fmt: str, target_sr: int = _TARGET_SR
) -> np.ndarray:
    """Decode in-memory audio bytes to float32 mono at target_sr."""
    if fmt == "wav":
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            # Use librosa lazily to avoid the import cost when not needed
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32)
    # mp3 / flac / opus / ogg paths -> librosa.load via temp file
    import librosa
    # Write bytes to a temp pipe via soundfile for non-MP3, else fall back to ffmpeg pipe
    if fmt in {"flac", "ogg"}:
        try:
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            return audio.astype(np.float32)
        except Exception:
            pass
    # Default: shell to ffmpeg for robust mp3/opus decoding to int16 PCM
    cmd = [
        _FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le", "-ar", str(target_sr), "-ac", "1", "pipe:1",
    ]
    proc = subprocess.run(cmd, input=raw, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed for fmt={fmt}: "
            f"{proc.stderr.decode('utf-8', 'replace')[:200]}"
        )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


def _process_one(
    file_entry: dict,
    *,
    tarball: Path,
    output_root: Path,
    voices_test: set[str],
    voices_eval: set[str],
    seed_base: int,
    spot_check_count: int = 0,
    spot_check_dir: Path | None = None,
    spot_check_state: dict[str, int] | None = None,
) -> dict:
    """Process a single conversation. Returns a result dict for aggregation.

    Result keys: conv_id, split, ok, error, augment_meta, n_boundaries, audio_sha256.
    """
    conv_id = file_entry["id"]
    fmt = file_entry.get("format", "wav")
    speakers = file_entry.get("speakers", []) or []
    voice_names = [s.get("voice_name", "") for s in speakers]
    split = assign_split(voice_names, voices_test, voices_eval)
    split_dir = output_root / split

    file_seed = _file_seed(conv_id, seed_base)
    rng = np.random.default_rng(file_seed)

    try:
        if tarball.is_dir():
            # Directory mode: read extracted files directly (no tar seeks).
            audio_path = _source_root_for_member(tarball, file_entry["audio_file"])
            transcript_path = _source_root_for_member(tarball, file_entry["transcript_file"])
            audio_raw = audio_path.read_bytes()
            transcript = json.loads(transcript_path.read_text())
        else:
            with tarfile.open(tarball, "r:*") as tf:
                audio_member = tf.getmember(file_entry["audio_file"])
                transcript_member = tf.getmember(file_entry["transcript_file"])
                audio_fobj = tf.extractfile(audio_member)
                tx_fobj = tf.extractfile(transcript_member)
                if audio_fobj is None or tx_fobj is None:
                    raise RuntimeError("tarfile.extractfile returned None")
                audio_raw = audio_fobj.read()
                transcript = json.load(tx_fobj)
    except Exception as exc:
        log.emit(
            "ERROR", "regen.regenerate.conv.error",
            conv_id=conv_id, stage="extract", error=str(exc),
        )
        return {"conv_id": conv_id, "split": split, "ok": False,
                "error": f"extract:{exc}"}

    try:
        audio = _decode_audio_bytes(audio_raw, fmt, target_sr=_TARGET_SR)
        sr = _TARGET_SR
    except Exception as exc:
        log.emit(
            "ERROR", "regen.regenerate.conv.error",
            conv_id=conv_id, stage="decode", error=str(exc),
        )
        return {"conv_id": conv_id, "split": split, "ok": False,
                "error": f"decode:{exc}"}

    turns = transcript.get("turns", []) or []
    flat_words = _flatten_word_alignment(turns)
    n_edits = int(rng.integers(1, 4))  # 1, 2, or 3
    chosen = choose_edit_positions(
        flat_words, min_silence_ms=_MIN_SILENCE_MS, n_edits=n_edits, rng=rng
    )
    if not chosen and flat_words:
        log.emit(
            "WARN", "regen.regenerate.conv.skip",
            conv_id=conv_id, reason="no_eligible_edit_positions",
            n_words=len(flat_words),
        )

    # Cross-voice GT in ORIGINAL coords: turn_idx > 0 -> turns[k]["start_ms"]
    cross_voice_orig_s: list[float] = []
    for k, t in enumerate(turns):
        if k > 0:
            cross_voice_orig_s.append(int(t.get("start_ms", 0)) / 1000.0)

    edited_audio, same_voice_joins = edit_audio(audio, sr, chosen)

    # Remap cross-voice points to post-cut coordinates
    cross_voice_post_s: list[tuple[float, str]] = []
    for orig_t in cross_voice_orig_s:
        post_t = orig_t - _shift_at(orig_t, chosen)
        if post_t < 0:
            continue
        cross_voice_post_s.append((post_t, "cross_voice"))

    # SNR + T60 sample
    snr_db = float(np.clip(rng.normal(22.0, 4.0), 10.0, 35.0))
    t60_s = float(rng.uniform(0.2, 0.6))

    try:
        augmented, augment_meta = augment(edited_audio, sr, snr_db, t60_s, rng)
        decoded, encoded = opus_roundtrip(augmented, sr, bitrate="32k")
    except Exception as exc:
        log.emit(
            "ERROR", "regen.regenerate.conv.error",
            conv_id=conv_id, stage="augment_or_opus", error=str(exc),
        )
        return {"conv_id": conv_id, "split": split, "ok": False,
                "error": f"augment:{exc}"}

    # Combine + sort GT by time
    boundaries_combined = [
        {"time_s": float(t), "label": label}
        for t, label in (cross_voice_post_s + same_voice_joins)
    ]
    boundaries_combined.sort(key=lambda b: b["time_s"])

    # Spot check (raw WAV for operator listening)
    if (
        spot_check_dir is not None
        and spot_check_state is not None
        and spot_check_state.get("emitted", 0) < spot_check_count
    ):
        spot_check_dir.mkdir(parents=True, exist_ok=True)
        wav_path = spot_check_dir / f"{conv_id}.wav"
        opus_copy = spot_check_dir / f"{conv_id}.opus"
        sf.write(str(wav_path), decoded, sr, subtype="PCM_16")
        opus_copy.write_bytes(encoded)
        spot_check_state["emitted"] = spot_check_state.get("emitted", 0) + 1

    voice_holdout = {
        "test": sorted(voices_test),
        "eval": sorted(voices_eval),
    }

    try:
        audio_sha = write_per_conv_outputs(
            out_dir=split_dir,
            conv_id=conv_id,
            audio=decoded,
            sr=sr,
            boundaries=boundaries_combined,
            augment_meta=augment_meta,
            split=split,
            voice_holdout=voice_holdout,
            augment_seed=file_seed,
            encoded_opus_bytes=encoded,
        )
    except Exception as exc:
        log.emit(
            "ERROR", "regen.regenerate.conv.error",
            conv_id=conv_id, stage="write", error=str(exc),
        )
        return {"conv_id": conv_id, "split": split, "ok": False,
                "error": f"write:{exc}"}

    return {
        "conv_id": conv_id,
        "split": split,
        "ok": True,
        "error": "",
        "augment_meta": augment_meta,
        "n_boundaries": len(boundaries_combined),
        "audio_sha256": audio_sha,
    }


# Worker entry point for multiprocessing.Pool — top-level for picklability.
def _pool_worker(payload: tuple) -> dict:
    file_entry, kwargs = payload
    # spot_check_state is per-process (not aggregated across workers); avoid
    # spot-check emission inside pool workers — caller only enables it for
    # workers=1.
    return _process_one(file_entry, **kwargs)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _existing_manifest_rows(split_dir: Path) -> dict[str, dict]:
    """Read existing _manifest.jsonl and index rows by conv_id (last row wins)."""
    manifest_path = split_dir / "_manifest.jsonl"
    rows: dict[str, dict] = {}
    if not manifest_path.exists():
        return rows
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = row.get("conv_id")
            if cid:
                rows[cid] = row
    return rows


def _is_complete(conv_id: str, split_dir: Path, manifest_rows: dict[str, dict]) -> bool:
    """Resume eligibility: opus + json + manifest row all present and consistent."""
    opus_path = split_dir / f"{conv_id}.opus"
    json_path = split_dir / f"{conv_id}.json"
    if not opus_path.exists() or not json_path.exists():
        return False
    row = manifest_rows.get(conv_id)
    if not row:
        return False
    expected_sha = row.get("audio_sha256", "")
    actual_sha = hashlib.sha256(opus_path.read_bytes()).hexdigest()
    return expected_sha == actual_sha


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


def _run_pipeline(
    args: argparse.Namespace,
    *,
    file_entries: list[dict],
    output_root: Path,
    workers: int,
    resume: bool,
    spot_check_count: int = 0,
) -> dict:
    """Shared driver for --regenerate and --smoke.

    Returns aggregate stats (per-split file counts, error count, augment summary).
    """
    output_root.mkdir(parents=True, exist_ok=True)
    voices_test: set[str] = set(args.voices_test.split(","))
    voices_eval: set[str] = set(args.voices_eval.split(","))
    seed_base = int(args.seed_base)

    spot_check_dir = Path(".omc/regen-spot-check") if spot_check_count > 0 else None
    spot_check_state: dict[str, int] = {"emitted": 0}

    # Resume filter
    if resume:
        # Index existing manifest rows per split
        per_split_rows: dict[str, dict[str, dict]] = {}
        skip_count = 0
        kept: list[dict] = []
        for fe in file_entries:
            voice_names = [s.get("voice_name", "") for s in fe.get("speakers", [])]
            split = assign_split(voice_names, voices_test, voices_eval)
            split_dir = output_root / split
            if split not in per_split_rows:
                per_split_rows[split] = _existing_manifest_rows(split_dir)
            if _is_complete(fe["id"], split_dir, per_split_rows[split]):
                skip_count += 1
                log.emit(
                    "INFO", "regen.regenerate.conv.skip",
                    conv_id=fe["id"], split=split, reason="already_complete",
                )
            else:
                kept.append(fe)
        print(f"[resume] skipped {skip_count} complete files; processing {len(kept)}.")
        file_entries = kept

    total = len(file_entries)
    per_split_count: dict[str, int] = {"train": 0, "eval": 0, "test": 0}
    error_count = 0
    snr_vals: list[float] = []
    t60_vals: list[float] = []
    max_clipped = 0
    splits_seen: set[str] = set()
    start_t = time.time()

    process_kwargs = {
        "tarball": Path(args.tarball),
        "output_root": output_root,
        "voices_test": voices_test,
        "voices_eval": voices_eval,
        "seed_base": seed_base,
    }

    def _consume(result: dict) -> None:
        nonlocal error_count, max_clipped
        if not result.get("ok"):
            error_count += 1
            return
        per_split_count[result["split"]] = per_split_count.get(result["split"], 0) + 1
        splits_seen.add(result["split"])
        am = result.get("augment_meta") or {}
        if "snr_db" in am:
            snr_vals.append(float(am["snr_db"]))
        if "t60_s" in am:
            t60_vals.append(float(am["t60_s"]))
        nclip = int(am.get("n_clipped_samples", 0))
        if nclip > max_clipped:
            max_clipped = nclip

    if workers <= 1:
        # Serial path — supports spot-check emission.
        for i, fe in enumerate(file_entries, start=1):
            res = _process_one(
                fe,
                **process_kwargs,
                spot_check_count=spot_check_count,
                spot_check_dir=spot_check_dir,
                spot_check_state=spot_check_state,
            )
            _consume(res)
            if i % 100 == 0 or i == total:
                elapsed = time.time() - start_t
                rate = i / elapsed if elapsed > 0 else 0.0
                est_remaining = (total - i) / rate if rate > 0 else 0.0
                log.emit(
                    "INFO", "regen.regenerate.conv.done",
                    progress_pct=round(100.0 * i / max(total, 1), 2),
                    conv_count=i,
                    elapsed_s=round(elapsed, 2),
                    est_remaining_s=round(est_remaining, 2),
                )
    else:
        payloads = [(fe, process_kwargs) for fe in file_entries]
        i = 0
        # `spawn` start method is safer (esp. on macOS) and avoids fork+tarfile pitfalls
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for res in pool.imap_unordered(_pool_worker, payloads, chunksize=1):
                _consume(res)
                i += 1
                if i % 100 == 0 or i == total:
                    elapsed = time.time() - start_t
                    rate = i / elapsed if elapsed > 0 else 0.0
                    est_remaining = (total - i) / rate if rate > 0 else 0.0
                    log.emit(
                        "INFO", "regen.regenerate.conv.done",
                        progress_pct=round(100.0 * i / max(total, 1), 2),
                        conv_count=i,
                        elapsed_s=round(elapsed, 2),
                        est_remaining_s=round(est_remaining, 2),
                    )

    # Synthesize per-split ground_truth.json for every split that received files.
    for split in sorted(splits_seen):
        synthesize_corpus_ground_truth(output_root / split)

    summary = {
        "total_processed": total,
        "errors": error_count,
        "per_split": per_split_count,
        "mean_snr_db": round(float(np.mean(snr_vals)), 3) if snr_vals else None,
        "mean_t60_s": round(float(np.mean(t60_vals)), 3) if t60_vals else None,
        "max_clipped_samples": max_clipped,
    }
    return summary


def _cmd_regenerate(args: argparse.Namespace) -> int:
    """Full corpus regeneration."""
    tarball = Path(args.tarball)
    output_root = Path(args.output_root)
    workers = max(1, min(int(getattr(args, "workers", 1) or 1), 8))
    resume = bool(getattr(args, "resume_in_regen", False))
    spot_check_count = int(getattr(args, "emit_spot_check", 0) or 0)
    if spot_check_count > 0 and workers != 1:
        print(
            f"[note] --emit-spot-check requires workers=1 (got {workers}); forcing serial.",
            file=sys.stderr,
        )
        workers = 1

    manifest = read_tarball_manifest(tarball)
    files = manifest.get("files", [])
    voices_test: set[str] = set(args.voices_test.split(","))
    voices_eval: set[str] = set(args.voices_eval.split(","))

    log.emit(
        "INFO", "regen.regenerate.start",
        total_files=len(files),
        voices_test=str(sorted(voices_test)),
        voices_eval=str(sorted(voices_eval)),
        workers=workers,
        spot_check_count=spot_check_count,
    )

    summary = _run_pipeline(
        args,
        file_entries=files,
        output_root=output_root,
        workers=workers,
        resume=resume,
        spot_check_count=spot_check_count,
    )

    log.emit(
        "INFO", "regen.regenerate.complete",
        total_processed=summary["total_processed"],
        errors=summary["errors"],
        train_count=summary["per_split"].get("train", 0),
        eval_count=summary["per_split"].get("eval", 0),
        test_count=summary["per_split"].get("test", 0),
    )
    log.emit(
        "INFO", "regen.regenerate.summary",
        mean_snr_db=summary["mean_snr_db"],
        mean_t60_s=summary["mean_t60_s"],
        max_clipped_samples=summary["max_clipped_samples"],
        errors=summary["errors"],
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["errors"] == 0 else 1


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Regenerate the first N conversations from the manifest."""
    n = int(args.smoke)
    tarball = Path(args.tarball)
    # Smoke-specific default output location
    if args.output_root == str(_DEFAULT_OUTPUT_ROOT):
        output_root = Path("/tmp/korean-iter1-smoke")
    else:
        output_root = Path(args.output_root)
    # Smoke: clear previous output for a clean re-run of the gate
    if output_root.exists():
        for split in ("train", "eval", "test"):
            split_dir = output_root / split
            if split_dir.exists():
                for f in split_dir.iterdir():
                    if f.is_file():
                        f.unlink()

    workers = max(1, min(int(getattr(args, "workers", 1) or 1), 8))

    manifest = read_tarball_manifest(tarball)
    files = manifest.get("files", [])[:n]

    log.emit(
        "INFO", "regen.smoke.start",
        n_requested=n, n_available=len(files),
        output_root=str(output_root), workers=workers,
    )

    # Borrow regenerate logic
    voices_test: set[str] = set(args.voices_test.split(","))
    voices_eval: set[str] = set(args.voices_eval.split(","))
    log.emit(
        "INFO", "regen.regenerate.start",
        total_files=len(files),
        voices_test=str(sorted(voices_test)),
        voices_eval=str(sorted(voices_eval)),
        workers=workers,
        smoke=True,
    )
    summary = _run_pipeline(
        args,
        file_entries=files,
        output_root=output_root,
        workers=workers,
        resume=False,
        spot_check_count=0,
    )
    log.emit(
        "INFO", "regen.regenerate.complete",
        total_processed=summary["total_processed"],
        errors=summary["errors"],
        train_count=summary["per_split"].get("train", 0),
        eval_count=summary["per_split"].get("eval", 0),
        test_count=summary["per_split"].get("test", 0),
        smoke=True,
    )
    log.emit(
        "INFO", "regen.smoke.complete",
        n=len(files), output_root=str(output_root),
        errors=summary["errors"],
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["errors"] == 0 else 1


def _cmd_resume(args: argparse.Namespace) -> int:
    """Re-run --regenerate but skip already-complete files."""
    setattr(args, "resume_in_regen", True)
    if not hasattr(args, "workers") or args.workers is None:
        args.workers = 1
    if not hasattr(args, "emit_spot_check"):
        args.emit_spot_check = 0
    return _cmd_regenerate(args)


def _cmd_verify_determinism(args: argparse.Namespace) -> int:
    """Re-run augmentation on N random files; byte-diff against on-disk Opus."""
    sample = int(getattr(args, "sample", 0) or 10)
    tarball = Path(args.tarball)
    output_root = Path(args.output_root)
    voices_test: set[str] = set(args.voices_test.split(","))
    voices_eval: set[str] = set(args.voices_eval.split(","))
    seed_base = int(args.seed_base)

    log.emit(
        "INFO", "regen.determinism.start",
        sample_size=sample, output_root=str(output_root),
    )

    # Collect on-disk conv_ids with their split
    on_disk: list[tuple[str, str]] = []  # (conv_id, split)
    for split in ("train", "eval", "test"):
        sd = output_root / split
        if not sd.exists():
            continue
        for op in sd.glob("*.opus"):
            on_disk.append((op.stem, split))
    if not on_disk:
        log.emit("ERROR", "regen.determinism.result",
                 reason="no_on_disk_files", matches=0, mismatches=0,
                 sample_size=sample)
        print("[FAIL] no on-disk files found under", output_root)
        return 1

    rnd = random.Random(seed_base)
    rnd.shuffle(on_disk)
    chosen = on_disk[:sample]

    manifest = read_tarball_manifest(tarball)
    by_id = {f["id"]: f for f in manifest.get("files", [])}

    matches = 0
    mismatches = 0
    details: list[dict] = []

    for conv_id, split in chosen:
        if conv_id not in by_id:
            mismatches += 1
            details.append({"conv_id": conv_id, "split": split,
                            "result": "missing_in_manifest"})
            continue
        # Re-run pipeline IN-MEMORY (do not overwrite outputs); compare encoded bytes
        result_bytes = _regenerate_in_memory(
            by_id[conv_id], tarball, voices_test, voices_eval, seed_base
        )
        on_disk_bytes = (output_root / split / f"{conv_id}.opus").read_bytes()
        on_disk_sha = hashlib.sha256(on_disk_bytes).hexdigest()
        regen_sha = hashlib.sha256(result_bytes).hexdigest()
        ok = on_disk_sha == regen_sha
        if ok:
            matches += 1
        else:
            mismatches += 1
        details.append({
            "conv_id": conv_id, "split": split,
            "result": "match" if ok else "mismatch",
            "on_disk_sha": on_disk_sha[:16], "regen_sha": regen_sha[:16],
        })
        log.emit(
            "INFO", "regen.determinism.sample",
            conv_id=conv_id, split=split,
            result="match" if ok else "mismatch",
            on_disk_sha=on_disk_sha[:16], regen_sha=regen_sha[:16],
        )

    receipt = {
        "sample_size": len(chosen),
        "matches": matches,
        "mismatches": mismatches,
        "seed_base": seed_base,
        "output_root": str(output_root),
        "details": details,
    }
    Path(".omc").mkdir(parents=True, exist_ok=True)
    Path(".omc/korean-iter1-determinism-receipt.json").write_text(
        json.dumps(receipt, indent=2)
    )

    log.emit(
        "INFO", "regen.determinism.result",
        matches=matches, mismatches=mismatches, sample_size=len(chosen),
    )
    print(json.dumps(
        {"matches": matches, "mismatches": mismatches,
         "sample_size": len(chosen),
         "receipt": ".omc/korean-iter1-determinism-receipt.json"},
        indent=2,
    ))
    return 0 if mismatches == 0 else 1


def _regenerate_in_memory(
    file_entry: dict,
    tarball: Path,
    voices_test: set[str],
    voices_eval: set[str],
    seed_base: int,
) -> bytes:
    """Re-run the augmentation+opus pipeline for one file, return encoded bytes.

    Mirrors `_process_one` but without writing anything. Used by --verify-determinism.
    """
    conv_id = file_entry["id"]
    fmt = file_entry.get("format", "wav")
    file_seed = _file_seed(conv_id, seed_base)
    rng = np.random.default_rng(file_seed)

    if tarball.is_dir():
        audio_path = _source_root_for_member(tarball, file_entry["audio_file"])
        transcript_path = _source_root_for_member(tarball, file_entry["transcript_file"])
        audio_raw = audio_path.read_bytes()
        transcript = json.loads(transcript_path.read_text())
    else:
        with tarfile.open(tarball, "r:*") as tf:
            audio_member = tf.getmember(file_entry["audio_file"])
            transcript_member = tf.getmember(file_entry["transcript_file"])
            audio_raw = tf.extractfile(audio_member).read()
            transcript = json.load(tf.extractfile(transcript_member))

    audio = _decode_audio_bytes(audio_raw, fmt, target_sr=_TARGET_SR)
    sr = _TARGET_SR
    turns = transcript.get("turns", []) or []
    flat_words = _flatten_word_alignment(turns)
    n_edits = int(rng.integers(1, 4))
    chosen = choose_edit_positions(
        flat_words, min_silence_ms=_MIN_SILENCE_MS, n_edits=n_edits, rng=rng
    )
    edited_audio, _ = edit_audio(audio, sr, chosen)
    snr_db = float(np.clip(rng.normal(22.0, 4.0), 10.0, 35.0))
    t60_s = float(rng.uniform(0.2, 0.6))
    augmented, _ = augment(edited_audio, sr, snr_db, t60_s, rng)
    _, encoded = opus_roundtrip(augmented, sr, bitrate="32k")
    return encoded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="regenerate_korean_iter1.py",
        description="korean-iter1 corpus regenerator (Step 6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--verify-source",
        action="store_true",
        help="Read tarball manifest, print voice diversity table, validate gender coverage.",
    )
    mode.add_argument(
        "--regenerate",
        action="store_true",
        help="Full pipeline: process every conversation in the manifest.",
    )
    mode.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Re-run augmentation on N random on-disk files and byte-diff (sample with --sample).",
    )
    mode.add_argument(
        "--smoke",
        metavar="N",
        type=int,
        help="Regenerate the first N conversations end-to-end (default output: /tmp/korean-iter1-smoke).",
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Like --regenerate, but skip files already complete on disk.",
    )

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
        "--voices-test", default=_DEFAULT_VOICES_TEST, dest="voices_test",
        help=f"Comma-separated voice names held out for test. Default: {_DEFAULT_VOICES_TEST}",
    )
    p.add_argument(
        "--voices-eval", default=_DEFAULT_VOICES_EVAL, dest="voices_eval",
        help=f"Comma-separated voice names held out for eval. Default: {_DEFAULT_VOICES_EVAL}",
    )
    p.add_argument(
        "--seed-base", default=_DEFAULT_SEED_BASE, type=int, dest="seed_base",
        help="Base random seed (per-file = seed_base ^ hash(file_id)).",
    )
    p.add_argument(
        "--workers", default=1, type=int,
        help="Worker processes (1-8). Default: 1.",
    )
    p.add_argument(
        "--emit-spot-check", default=0, type=int, dest="emit_spot_check",
        help="With --regenerate, write the first N files as raw WAV under .omc/regen-spot-check/.",
    )
    p.add_argument(
        "--sample", default=10, type=int,
        help="With --verify-determinism, number of on-disk files to re-run.",
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
