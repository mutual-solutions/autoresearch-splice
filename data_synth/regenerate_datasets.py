#!/usr/bin/env python3
"""Unified eval / train / test dataset generator for autoresearch-splice.

Splits source audio into three disjoint pools per domain (train / eval / test)
and generates 60 files per split with matching 20 tier1 + 20 tier2 + 20 clean
layout. Per-cell duration mix and encoding mix are hardcoded so every split
has identical structure except for the duration envelope (test is longer).

Usage:
  python data_synth/regenerate_datasets.py --domain singing --split eval
  python data_synth/regenerate_datasets.py --domain all --split all    # 9 combos

Output layout:
  data/eval/<domain>/                    eval (iterated by evaluate.py)
  data/train/<domain>/                   train (consumed by train_classifier.py)
  data/test/<domain>/                    held-out test (20-min budget)
  data/sources/                          raw audio pools (input)

Each output dir contains tier1/ tier2/ clean/ and ground_truth.json.

Audio files are encoded as WAV / FLAC / OGG-Opus / MP3-128 per the
per-cell encoding plan; the encoding is recorded in ground_truth.json and
filenames carry the extension so downstream soundfile.read picks the
right decoder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
from splice_boundary import find_splice_point  # type: ignore

# ---------------------------------------------------------------------------
# Shape configuration (per split)
# ---------------------------------------------------------------------------

# 10 files per tier × regime cell. Position in the list determines duration
# and encoding. Keep the two lists the same length so the zip is lossless.
EVAL_TRAIN_DURATION_S = [30, 30, 30, 60, 60, 60, 60, 90, 90, 120]
TEST_DURATION_S       = [60, 120, 180, 180, 180, 240, 240, 240, 300, 300]
ENCODING_PLAN         = ["wav", "wav", "wav", "wav", "wav", "wav",
                         "flac", "flac", "opus", "mp3"]
assert len(EVAL_TRAIN_DURATION_S) == len(ENCODING_PLAN) == 10
assert len(TEST_DURATION_S) == len(ENCODING_PLAN) == 10

CROSSFADE_MS = [10, 10, 10, 50, 50, 50, 100, 100, 200, 200]  # 10 per cell
assert len(CROSSFADE_MS) == 10

REGIMES = ["random", "quiet_matched"]  # 10 files each

SPLICE_FRAC_RANGE = (0.30, 0.70)  # splice lands in 30-70% of the file

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
ENCODING_EXT = {"wav": "wav", "flac": "flac", "opus": "opus", "mp3": "mp3"}


# ---------------------------------------------------------------------------
# Domain source pool: deterministic 3-way split over a sortable identifier.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourcePool:
    domain: str
    split: str
    sr: int
    # List of (source_id, loader_fn) — loader_fn(needed_samples) -> np.ndarray
    entries: list


def _sorted_singing_sources() -> list[tuple[str, Path]]:
    base = _ROOT / "data" / "sources" / "singing_wav"
    return [(p.stem, p) for p in sorted(base.glob("*.wav"))]


def _sorted_korean_speakers() -> list[tuple[str, list[Path]]]:
    """Return [(speaker_id, [flac_paths...])], sorted by speaker_id across
    the full zeroth-korean corpus (train_data_01 ∪ test_data_01). We treat
    the corpus as a single pool and split it by our own deterministic slice.
    """
    speakers: dict[str, list[Path]] = {}
    for base in [_ROOT / "data" / "sources" / "zeroth-korean" / "train_data_01",
                 _ROOT / "data" / "sources" / "zeroth-korean" / "test_data_01"]:
        if not base.exists():
            continue
        for flac in base.rglob("*.flac"):
            spk = flac.name.split("_")[0]
            speakers.setdefault(spk, []).append(flac)
    return [(spk, sorted(paths)) for spk, paths in sorted(speakers.items())]


def _sorted_english_speakers() -> list[tuple[str, list[Path]]]:
    base = _ROOT / "data" / "sources" / "LibriSpeech" / "dev-clean"
    speakers: dict[str, list[Path]] = {}
    if base.exists():
        for spk_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            paths = sorted(spk_dir.rglob("*.flac"))
            if paths:
                speakers[spk_dir.name] = paths
    return list(speakers.items())


def _slice_pool(items: list, split: str, sizes: tuple[int, int, int]):
    """Return the subset of `items` for the given split. sizes=(train,eval,test)."""
    n_train, n_eval, n_test = sizes
    if split == "train":
        return items[:n_train]
    if split == "eval":
        return items[n_train:n_train + n_eval]
    if split == "test":
        return items[n_train + n_eval:n_train + n_eval + n_test]
    raise ValueError(f"unknown split {split!r}")


def build_source_pool(domain: str, split: str) -> SourcePool:
    if domain == "singing":
        items = _sorted_singing_sources()
        n_total = len(items)
        if n_total < 101:
            print(f"WARN: expected ≥101 singing sources, got {n_total}", file=sys.stderr)
        sizes = (41, 40, 20)  # train / eval / test
        picked = _slice_pool(items, split, sizes)
        entries = [(sid, _make_wav_loader(p, 44100)) for sid, p in picked]
        return SourcePool("singing", split, 44100, entries)

    if domain == "korean":
        items = _sorted_korean_speakers()
        sizes = (60, 40, 15)
        picked = _slice_pool(items, split, sizes)
        entries = [(spk, _make_flac_loader(paths, 16000)) for spk, paths in picked]
        return SourcePool("korean", split, 16000, entries)

    if domain == "english":
        items = _sorted_english_speakers()
        sizes = (20, 12, 8)
        picked = _slice_pool(items, split, sizes)
        entries = [(spk, _make_flac_loader(paths, 16000)) for spk, paths in picked]
        return SourcePool("english", split, 16000, entries)

    raise ValueError(f"unknown domain {domain!r}")


def _make_wav_loader(path: Path, expect_sr: int):
    def load(min_samples: int) -> np.ndarray:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != expect_sr:
            raise ValueError(f"{path}: sr={sr}, expected {expect_sr}")
        if len(audio) < min_samples:
            raise ValueError(f"{path}: {len(audio)} samples < required {min_samples}")
        return audio
    return load


def _make_flac_loader(paths: list[Path], expect_sr: int):
    """Concatenate FLACs until target length is reached."""
    def load(min_samples: int) -> np.ndarray:
        chunks = []
        total = 0
        for fp in paths:
            a, sr = sf.read(str(fp), dtype="float32", always_2d=False)
            if a.ndim == 2:
                a = a.mean(axis=1)
            if sr != expect_sr:
                raise ValueError(f"{fp}: sr={sr}, expected {expect_sr}")
            chunks.append(a)
            total += len(a)
            if total >= min_samples:
                break
        if total < min_samples:
            raise ValueError(
                f"speaker source too short: {total} < {min_samples} samples"
            )
        return np.concatenate(chunks)[:min_samples + int(0.1 * expect_sr)]  # small pad
    return load


# ---------------------------------------------------------------------------
# Splicing + encoding
# ---------------------------------------------------------------------------

def linear_crossfade(seg_a: np.ndarray, seg_b: np.ndarray,
                     fade_samples: int) -> np.ndarray:
    fade_samples = min(fade_samples, len(seg_a), len(seg_b))
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    body_a  = seg_a[:-fade_samples]
    overlap = seg_a[-fade_samples:] * fade_out + seg_b[:fade_samples] * fade_in
    tail_b  = seg_b[fade_samples:]
    return np.concatenate([body_a, overlap, tail_b])


def encode_file(wav_array: np.ndarray, sr: int, encoding: str, out_path: Path) -> None:
    """Write wav_array to disk in the requested encoding. Requires ffmpeg
    for opus / mp3 encodings since libsndfile's encoders for those depend
    on libsndfile build options.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if encoding == "wav":
        sf.write(str(out_path), wav_array.astype(np.float32), sr, subtype="PCM_16")
        return
    if encoding == "flac":
        sf.write(str(out_path), wav_array.astype(np.float32), sr, subtype="PCM_16", format="FLAC")
        return

    # opus / mp3 go through ffmpeg — write a temp WAV first
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), wav_array.astype(np.float32), sr, subtype="PCM_16")
        if encoding == "opus":
            args = [FFMPEG, "-y", "-i", str(tmp_path),
                    "-c:a", "libopus", "-b:a", "48k",
                    "-vbr", "on", "-application", "audio",
                    str(out_path)]
        elif encoding == "mp3":
            args = [FFMPEG, "-y", "-i", str(tmp_path),
                    "-c:a", "libmp3lame", "-b:a", "128k",
                    str(out_path)]
        else:
            raise ValueError(f"unknown encoding {encoding!r}")
        r = subprocess.run(args, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {encoding}: "
                               f"{r.stderr.decode(errors='ignore')[:300]}")
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Per-file generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileSpec:
    filename: str
    duration_s: int
    encoding: str       # wav / flac / opus / mp3
    tier: int           # 0 clean, 1 hard cut, 2 crossfade
    regime: str         # random / quiet_matched (only meaningful for spliced)
    crossfade_ms: int   # 0 for clean / hard cut
    slot_idx: int       # position within cell (0..9) for reproducibility


def plan_cell(prefix: str, tier: int, regime: str, durations: list[int]) -> list[FileSpec]:
    """Build 10 FileSpecs for one cell (tier × regime or clean)."""
    specs = []
    for slot, (dur, enc) in enumerate(zip(durations, ENCODING_PLAN)):
        # Cell index is the global slot inside the tier: regime=random → 0..9,
        # quiet_matched → 10..19. Clean uses the same slot indexing with a
        # "clean" regime label.
        regime_offset = 0 if regime == "random" or regime == "clean" else 10
        idx_in_tier = slot + regime_offset
        ext = ENCODING_EXT[enc]
        filename = f"{prefix}_{idx_in_tier + 1:03d}.{ext}"
        cf_ms = CROSSFADE_MS[slot] if tier == 2 else 0
        specs.append(FileSpec(
            filename=filename,
            duration_s=dur,
            encoding=enc,
            tier=tier,
            regime=regime,
            crossfade_ms=cf_ms,
            slot_idx=slot,
        ))
    return specs


def plan_split(split: str) -> list[FileSpec]:
    """Return 60 FileSpecs for a full split (20 t1 + 20 t2 + 20 clean)."""
    durations = EVAL_TRAIN_DURATION_S if split != "test" else TEST_DURATION_S
    specs: list[FileSpec] = []
    for tier, prefix in [(1, "splice_t1"), (2, "splice_t2")]:
        for regime in REGIMES:
            specs.extend(plan_cell(prefix, tier, regime, durations))
    # Clean: 20 files, using a single "clean" regime label (not tied to splice
    # energy modes). Duration/encoding plan shared across the 20 slots.
    for regime_label, regime_offset in [("clean_a", 0), ("clean_b", 10)]:
        for slot, (dur, enc) in enumerate(zip(durations, ENCODING_PLAN)):
            ext = ENCODING_EXT[enc]
            idx = slot + regime_offset
            specs.append(FileSpec(
                filename=f"clean_{idx + 1:03d}.{ext}",
                duration_s=dur,
                encoding=enc,
                tier=0,
                regime=regime_label,  # purely for internal bookkeeping
                crossfade_ms=0,
                slot_idx=slot,
            ))
    assert len(specs) == 60, len(specs)
    return specs


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

OUTPUT_DIRS = {
    ("singing", "eval"):  _ROOT / "data" / "eval" / "singing",
    ("korean",  "eval"):  _ROOT / "data" / "eval" / "korean",
    ("english", "eval"):  _ROOT / "data" / "eval" / "english",
    ("singing", "train"): _ROOT / "data" / "train" / "singing",
    ("korean",  "train"): _ROOT / "data" / "train" / "korean",
    ("english", "train"): _ROOT / "data" / "train" / "english",
    ("singing", "test"):  _ROOT / "data" / "test" / "singing",
    ("korean",  "test"):  _ROOT / "data" / "test" / "korean",
    ("english", "test"):  _ROOT / "data" / "test" / "english",
}


def regenerate(domain: str, split: str, seed: int = 42) -> None:
    out_dir = OUTPUT_DIRS[(domain, split)]
    pool = build_source_pool(domain, split)
    if len(pool.entries) < 2:
        raise RuntimeError(
            f"{domain}/{split}: need ≥2 source entries, got {len(pool.entries)}"
        )
    print(f"\n[{domain}/{split}] output={out_dir}  sr={pool.sr}  sources={len(pool.entries)}")

    # Clean the existing dir (tier1/tier2/clean/ground_truth.json only)
    if out_dir.exists():
        for sub in ("tier1", "tier2", "clean"):
            p = out_dir / sub
            if p.exists():
                shutil.rmtree(p)
        gt_path = out_dir / "ground_truth.json"
        if gt_path.exists():
            gt_path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    rs = np.random.RandomState(seed + hash((domain, split)) % 1_000_000)
    specs = plan_split(split)
    ground_truth: dict = {}

    # Precompute padding margin: ensure generators can land splices within
    # [SPLICE_MIN_FRAC, SPLICE_MAX_FRAC] of the longest file in the split.
    max_dur_needed = max(s.duration_s for s in specs)
    needed_samples = int(max_dur_needed * pool.sr) + pool.sr  # +1s safety

    # Simple loader cache: load each source entry only once per invocation.
    loaded: dict[int, np.ndarray] = {}
    def _load(idx: int, min_samples: int) -> np.ndarray:
        if idx in loaded and len(loaded[idx]) >= min_samples:
            return loaded[idx]
        arr = pool.entries[idx][1](min_samples)
        loaded[idx] = arr
        return arr

    def _pair(min_samples: int, exclude: set[int]) -> tuple[int, int, np.ndarray, np.ndarray]:
        idxs = [i for i in range(len(pool.entries)) if i not in exclude]
        rs.shuffle(idxs)
        a = b = None
        i_a = i_b = -1
        for cand in idxs:
            try:
                arr = _load(cand, min_samples)
            except ValueError:
                continue
            if a is None:
                a, i_a = arr, cand
            elif b is None:
                b, i_b = arr, cand
                break
        if a is None or b is None:
            raise RuntimeError(f"could not find 2 sources with ≥{min_samples} samples")
        return i_a, i_b, a, b

    # ---- Spliced files (tier1 + tier2) ----
    for spec in [s for s in specs if s.tier in (1, 2)]:
        target_samples = int(spec.duration_s * pool.sr)
        # Margin for crossfade length + safety
        fade_samples = int(pool.sr * spec.crossfade_ms / 1000)
        margin = fade_samples + int(0.5 * pool.sr)

        splice_samples = -1
        info = {}
        for try_idx in range(15):
            try:
                i_a, i_b, full_a, full_b = _pair(target_samples, exclude=set())
            except RuntimeError as e:
                raise RuntimeError(f"{spec.filename}: {e}")
            # Pick a contiguous target_samples slice of each
            start_a = int(rs.randint(0, len(full_a) - target_samples + 1))
            start_b = int(rs.randint(0, len(full_b) - target_samples + 1))
            seg_a = full_a[start_a:start_a + target_samples]
            seg_b = full_b[start_b:start_b + target_samples]
            try:
                splice_samples, info = find_splice_point(
                    seg_a, seg_b, pool.sr, rs,
                    mode=spec.regime,
                    splice_range=SPLICE_FRAC_RANGE,
                    margin_samples=margin,
                )
                break
            except ValueError:
                continue
        if splice_samples < 0:
            raise RuntimeError(f"{spec.filename}: regime {spec.regime!r} unsatisfiable after 15 tries")

        if spec.tier == 1:
            spliced = np.concatenate([seg_a[:splice_samples], seg_b[splice_samples:]])
        else:
            seg_a_cut = seg_a[:splice_samples]
            seg_b_cut = seg_b[splice_samples:]
            spliced = linear_crossfade(seg_a_cut, seg_b_cut, fade_samples)

        # Length-normalize
        spliced = spliced[:target_samples]
        if len(spliced) < target_samples:
            spliced = np.pad(spliced, (0, target_samples - len(spliced)))

        out_path = out_dir / f"tier{spec.tier}" / spec.filename
        encode_file(spliced.astype(np.float32), pool.sr, spec.encoding, out_path)

        splice_time_sec = splice_samples / pool.sr
        src_a = pool.entries[i_a][0]
        src_b = pool.entries[i_b][0]
        ground_truth[spec.filename] = {
            "path": f"tier{spec.tier}/{spec.filename}",
            "spliced": True,
            "tier": spec.tier,
            "crossfade_ms": spec.crossfade_ms,
            "splice_time_sec": round(splice_time_sec, 6),
            "duration_s": spec.duration_s,
            "encoding": spec.encoding,
            "boundary_energy": spec.regime,
            "boundary_rms_db_a": round(info["rms_db_a"], 2),
            "boundary_rms_db_b": round(info["rms_db_b"], 2),
            "source_a": src_a,
            "source_b": src_b,
            "n_candidates": info.get("n_candidates", -1),
        }
        print(f"  {spec.filename}: splice@{splice_time_sec:.2f}s  "
              f"[{spec.regime}]  cf={spec.crossfade_ms}ms  enc={spec.encoding}")

    # ---- Clean files (one source each) ----
    for spec in [s for s in specs if s.tier == 0]:
        target_samples = int(spec.duration_s * pool.sr)
        # Rotate through sources deterministically so no single source is
        # over-used; we don't need to exclude anything from the spliced pool
        # since there's no detection-time leakage concern for clean files.
        idxs = list(range(len(pool.entries)))
        rs.shuffle(idxs)
        picked = None
        for cand in idxs:
            try:
                a = _load(cand, target_samples)
                picked = cand
                break
            except ValueError:
                continue
        if picked is None:
            raise RuntimeError(f"{spec.filename}: no source entry ≥{target_samples} samples")
        src_audio = _load(picked, target_samples)
        start = int(rs.randint(0, len(src_audio) - target_samples + 1))
        clean = src_audio[start:start + target_samples]

        out_path = out_dir / "clean" / spec.filename
        encode_file(clean.astype(np.float32), pool.sr, spec.encoding, out_path)

        src_id = pool.entries[picked][0]
        ground_truth[spec.filename] = {
            "path": f"clean/{spec.filename}",
            "spliced": False,
            "tier": 0,
            "duration_s": spec.duration_s,
            "encoding": spec.encoding,
            "source": src_id,
            "start_sec": round(start / pool.sr, 6),
        }
        print(f"  {spec.filename}: clean  dur={spec.duration_s}s  enc={spec.encoding}")

    gt_path = out_dir / "ground_truth.json"
    with gt_path.open("w") as f:
        json.dump(ground_truth, f, indent=2, sort_keys=True)
    print(f"  wrote {gt_path}  ({len(ground_truth)} entries)")


# ---------------------------------------------------------------------------

DOMAINS = ("singing", "korean", "english")
SPLITS = ("train", "eval", "test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=(*DOMAINS, "all"), required=True)
    ap.add_argument("--split", choices=(*SPLITS, "all"), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-test-encrypt", action="store_true",
        help="Do NOT auto-encrypt data/test/ after generating the test "
             "split. Default is to invoke scripts/test_crypto.py encrypt "
             "(requires Touch ID) so the plaintext never sits on disk.",
    )
    args = ap.parse_args()

    doms = DOMAINS if args.domain == "all" else (args.domain,)
    splits = SPLITS if args.split == "all" else (args.split,)
    generated_splits: set = set()
    for d in doms:
        for s in splits:
            regenerate(d, s, seed=args.seed)
            generated_splits.add(s)

    # Auto-encrypt test split when we just regenerated it. Plaintext
    # data/test/ must never linger on disk — any downstream script could
    # accidentally leak it into autoresearch's evaluation loop.
    if "test" in generated_splits and not args.skip_test_encrypt:
        import subprocess
        from pathlib import Path
        script = Path(__file__).resolve().parents[1] / "scripts" / "test_crypto.py"
        blob = Path(__file__).resolve().parents[1] / "data" / "test.tar.gz.enc"
        mode = "setup" if not blob.exists() else "encrypt"
        print(f"\n[post-regen] Encrypting data/test/ via {script.name} {mode}")
        print("             (Touch ID prompt expected)")
        r = subprocess.run(
            ["uv", "run", "python", str(script), mode],
            check=False,
        )
        if r.returncode != 0:
            print(f"[post-regen] test encryption FAILED (exit {r.returncode}). "
                  "data/test/ was left in plaintext — encrypt it manually before "
                  "starting autoresearch.")


if __name__ == "__main__":
    main()
