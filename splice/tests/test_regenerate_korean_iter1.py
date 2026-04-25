"""Unit tests for scripts/regenerate_korean_iter1.py.

These tests cover the pure helpers (assign_split, choose_edit_positions,
edit_audio, build_rir, build_pink_noise, augment, opus_roundtrip,
synthesize_corpus_ground_truth). They do NOT touch the 20 GB tarball.

Imports resolve via PYTHONPATH=$PWD; tests run via:
    PYTHONPATH=$PWD uv run pytest splice/tests/test_regenerate_korean_iter1.py -v
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

# Disable the unified logger sink for tests; we don't need real JSONL output.
import os as _os
_os.environ.setdefault("OMC_LOGGER_DISABLED", "1")

# Load the regenerator module by file path (it lives under scripts/, not a package).
_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "regenerate_korean_iter1.py"
_spec = importlib.util.spec_from_file_location("_regen_korean_iter1", _SCRIPT)
regen = importlib.util.module_from_spec(_spec)
sys.modules["_regen_korean_iter1"] = regen
_spec.loader.exec_module(regen)


# ---------------------------------------------------------------------------
# assign_split
# ---------------------------------------------------------------------------

def test_assign_split_test_takes_precedence() -> None:
    """A speaker in BOTH test and eval falls into test (strictest holdout)."""
    out = regen.assign_split(
        ["DaeBuHo", "Sunwoo"],
        voices_test={"DaeBuHo", "Sunwoo"},
        voices_eval={"Sunwoo", "Joon"},
    )
    assert out == "test"


def test_assign_split_train_default() -> None:
    """No held-out voices -> train."""
    out = regen.assign_split(
        ["Minwoo", "ChloeCha"],
        voices_test={"DaeBuHo", "Kanna"},
        voices_eval={"Sunwoo", "Joon"},
    )
    assert out == "train"


def test_assign_split_eval_when_only_eval_voice_present() -> None:
    out = regen.assign_split(
        ["Sunwoo", "Minwoo"],
        voices_test={"DaeBuHo", "Kanna"},
        voices_eval={"Sunwoo", "Joon"},
    )
    assert out == "eval"


# ---------------------------------------------------------------------------
# choose_edit_positions
# ---------------------------------------------------------------------------

def _mk_word(start_ms: int, end_ms: int, prev_end_ms: int, next_start_ms: int,
             word: str = "x", turn_idx: int = 0) -> dict:
    return {
        "word": word, "start_ms": start_ms, "end_ms": end_ms,
        "prev_end_ms": prev_end_ms, "next_start_ms": next_start_ms,
        "turn_idx": turn_idx,
    }


def test_choose_edit_positions_respects_silence_padding() -> None:
    """All chosen words must have >= 120 ms silence before AND after."""
    words = [
        _mk_word(0, 100, 0, 110),         # gap_after = 10 ms (FAIL)
        _mk_word(200, 300, 110, 500),     # gap_before = 90 ms (FAIL)
        _mk_word(700, 800, 500, 950),     # gap_before=200, gap_after=150 (PASS)
        _mk_word(1100, 1200, 950, 1400),  # gap_before=150, gap_after=200 (PASS)
        _mk_word(1700, 1800, 1400, 1820), # gap_after=20 (FAIL)
        _mk_word(2000, 2100, 1820, 2300), # gap_before=180, gap_after=200 (PASS)
    ]
    rng = np.random.default_rng(42)
    chosen = regen.choose_edit_positions(words, min_silence_ms=120, n_edits=3, rng=rng)
    assert len(chosen) == 3
    for w in chosen:
        assert int(w["start_ms"]) - int(w["prev_end_ms"]) >= 120
        assert int(w["next_start_ms"]) - int(w["end_ms"]) >= 120


def test_choose_edit_positions_deterministic_with_seed() -> None:
    words = [
        _mk_word(i * 1000, i * 1000 + 200, max(0, i * 1000 - 200), (i + 1) * 1000)
        for i in range(20)
    ]
    chosen_a = regen.choose_edit_positions(
        words, min_silence_ms=120, n_edits=3, rng=np.random.default_rng(7)
    )
    chosen_b = regen.choose_edit_positions(
        words, min_silence_ms=120, n_edits=3, rng=np.random.default_rng(7)
    )
    assert [w["start_ms"] for w in chosen_a] == [w["start_ms"] for w in chosen_b]


def test_choose_edit_positions_returns_empty_when_no_eligible() -> None:
    """All words tightly packed -> no eligible positions -> []."""
    words = [
        _mk_word(0, 100, 0, 110),
        _mk_word(110, 200, 100, 220),
        _mk_word(220, 320, 200, 330),
    ]
    out = regen.choose_edit_positions(
        words, min_silence_ms=120, n_edits=1, rng=np.random.default_rng(0)
    )
    assert out == []


def test_choose_edit_positions_empty_when_fewer_eligible_than_requested() -> None:
    words = [
        _mk_word(700, 800, 500, 950),  # only 1 eligible
        _mk_word(1700, 1800, 1400, 1820),  # 20 ms after-gap (fail)
    ]
    out = regen.choose_edit_positions(
        words, min_silence_ms=120, n_edits=3, rng=np.random.default_rng(0)
    )
    assert out == []


# ---------------------------------------------------------------------------
# edit_audio
# ---------------------------------------------------------------------------

def test_edit_audio_concatenates_correctly() -> None:
    """1-s sine, 2 cuts -> output duration = 1.0 - sum(cut durations); joins correct."""
    sr = 44100
    audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    # Cut 100 ms at t=300 ms and 200 ms at t=600 ms (total removed = 300 ms)
    edits = [
        {"word": "a", "start_ms": 300, "end_ms": 400,
         "prev_end_ms": 0, "next_start_ms": 600, "turn_idx": 0},
        {"word": "b", "start_ms": 600, "end_ms": 800,
         "prev_end_ms": 400, "next_start_ms": 1000, "turn_idx": 0},
    ]
    edited, joins = regen.edit_audio(audio, sr, edits)
    expected_len = sr - int(round(0.1 * sr)) - int(round(0.2 * sr))
    assert abs(edited.shape[0] - expected_len) <= 1, (
        f"expected {expected_len}, got {edited.shape[0]}"
    )
    # Join positions: first cut at t=300 ms (no shift yet), second cut shifted by 100 ms.
    assert len(joins) == 2
    assert joins[0][1] == "same_voice_edit"
    assert abs(joins[0][0] - 0.300) < 1e-3
    assert abs(joins[1][0] - 0.500) < 1e-3  # 600 - 100 ms shift


def test_edit_audio_empty_edits_returns_unchanged() -> None:
    sr = 44100
    audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    edited, joins = regen.edit_audio(audio, sr, [])
    assert joins == []
    np.testing.assert_array_equal(edited, audio)


# ---------------------------------------------------------------------------
# build_rir
# ---------------------------------------------------------------------------

def test_build_rir_t60_decay() -> None:
    """RIR envelope at sample n should sit ~ -60 dB below peak (within ±6 dB)."""
    sr = 44100
    t60 = 0.4
    rir = regen.build_rir(t60, sr, np.random.default_rng(0))
    n = int(t60 * sr)
    assert rir.shape[0] == n
    peak = float(np.max(np.abs(rir)))
    # Compare a small window near the END of the RIR (the decay floor) against peak.
    tail = float(np.sqrt(np.mean(rir[-max(8, n // 100):] ** 2)))
    db_drop = 20.0 * np.log10(tail / peak + 1e-20)
    # Envelope is exp(-6 * t/n) -> ~ -52 dB at sample n; allow generous tolerance.
    assert -75 < db_drop < -30, f"db_drop={db_drop:.1f} dB outside tolerance"


# ---------------------------------------------------------------------------
# build_pink_noise
# ---------------------------------------------------------------------------

def test_build_pink_noise_bandpass() -> None:
    """>70% of energy lives in the [100 Hz, 6 kHz] band."""
    sr = 44100
    n = sr * 2  # 2 s
    noise = regen.build_pink_noise(n, sr, np.random.default_rng(123))
    # Power spectrum
    fft = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    psd = np.abs(fft) ** 2
    in_band = ((freqs >= 100.0) & (freqs <= 6000.0))
    energy_in = float(psd[in_band].sum())
    energy_total = float(psd.sum())
    frac = energy_in / max(energy_total, 1e-20)
    assert frac > 0.70, f"only {frac:.3f} of energy in [100, 6000] Hz band"


# ---------------------------------------------------------------------------
# augment
# ---------------------------------------------------------------------------

def test_augment_no_nan_no_clip() -> None:
    sr = 44100
    audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32) * 0.5
    out, meta = regen.augment(audio, sr, snr_db=20.0, t60_s=0.4,
                              rng=np.random.default_rng(0))
    assert out.shape == audio.shape
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) < 0.999
    assert meta["codec"] == "opus_32k"


# ---------------------------------------------------------------------------
# opus_roundtrip
# ---------------------------------------------------------------------------

def test_opus_roundtrip_duration_preserved() -> None:
    """Encoded-then-decoded length must be within 5% of input."""
    sr = 44100
    audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32) * 0.5
    decoded, encoded = regen.opus_roundtrip(audio, sr, bitrate="32k")
    ratio = decoded.shape[0] / audio.shape[0]
    assert 0.95 <= ratio <= 1.05, f"duration ratio {ratio:.3f} out of bounds"
    assert isinstance(encoded, bytes) and len(encoded) > 0


def test_opus_roundtrip_byte_identical_across_calls() -> None:
    """Two encodes of the same input produce byte-identical encoded bytes.

    ffmpeg's Ogg muxer normally injects a random per-session bitstream serial
    number; the script's _normalize_ogg_serial post-processing pins it to 0
    so --verify-determinism passes.
    """
    sr = 44100
    audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32) * 0.5
    _, e1 = regen.opus_roundtrip(audio, sr, bitrate="32k")
    _, e2 = regen.opus_roundtrip(audio, sr, bitrate="32k")
    assert e1 == e2, "opus_roundtrip must produce byte-identical output across runs"


def test_opus_roundtrip_signal_correlation() -> None:
    """Decoded signal must correlate >= 0.7 with the input on tonal content."""
    sr = 44100
    audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32) * 0.5
    decoded, _ = regen.opus_roundtrip(audio, sr, bitrate="32k")
    # Truncate to common length
    n = min(audio.shape[0], decoded.shape[0])
    a = audio[:n] - audio[:n].mean()
    b = decoded[:n] - decoded[:n].mean()
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))
    assert corr >= 0.7, f"correlation {corr:.3f} < 0.7"


# ---------------------------------------------------------------------------
# synthesize_corpus_ground_truth
# ---------------------------------------------------------------------------

def test_synthesize_corpus_ground_truth_aggregates(tmp_path: Path) -> None:
    """3 fake per-conv JSONs -> ground_truth.json has 3 keys with matching lists."""
    split_dir = tmp_path / "eval"
    split_dir.mkdir()
    fake = {
        "conversation_00001": [{"time_s": 1.0, "label": "cross_voice"},
                               {"time_s": 2.5, "label": "same_voice_edit"}],
        "conversation_00002": [{"time_s": 0.5, "label": "cross_voice"}],
        "conversation_00003": [],
    }
    for cid, b in fake.items():
        (split_dir / f"{cid}.json").write_text(json.dumps({
            "conv_id": cid,
            "split": "eval",
            "duration_s": 10.0,
            "sr": 44100,
            "boundaries": b,
            "augment_meta": {"snr_db": 22.0, "t60_s": 0.4, "codec": "opus_32k"},
        }))
    out = regen.synthesize_corpus_ground_truth(split_dir)
    assert out == split_dir / "ground_truth.json"
    loaded = json.loads(out.read_text())
    assert set(loaded.keys()) == set(fake.keys())
    for cid, expected in fake.items():
        actual = loaded[cid]
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert abs(a["time_s"] - e["time_s"]) < 1e-9
            assert a["label"] == e["label"]


# ---------------------------------------------------------------------------
# Per-file seed determinism (sanity)
# ---------------------------------------------------------------------------

def test_file_seed_deterministic() -> None:
    a = regen._file_seed("conversation_00001", seed_base=0)
    b = regen._file_seed("conversation_00001", seed_base=0)
    assert a == b
    c = regen._file_seed("conversation_00002", seed_base=0)
    assert a != c


def test_file_seed_in_uint32_range() -> None:
    seed = regen._file_seed("conversation_99999", seed_base=12345)
    assert 0 <= seed <= 0xFFFFFFFF


# Regression test for the 2026-04-25 word_alignment absolute-vs-relative bug.
# The iter-1 manifest's word_alignment fields are ABSOLUTE timestamps, not
# relative-to-turn. Earlier code added turn.start_ms, double-counting and
# producing out-of-bounds positions. This test guards against regression.
def test_flatten_word_alignment_treats_timestamps_as_absolute():
    from scripts.regenerate_korean_iter1 import _flatten_word_alignment
    # Synthetic transcript: 2 turns, word_alignment is in absolute audio time.
    turns = [
        {
            "idx": 0,
            "start_ms": 0,
            "end_ms": 2000,
            "word_alignment": [
                {"word": "hello", "start_ms": 100, "end_ms": 500},
                {"word": "world", "start_ms": 800, "end_ms": 1500},
            ],
        },
        {
            "idx": 1,
            "start_ms": 2500,    # turn 2 starts at 2.5 sec (post-gap)
            "end_ms": 5000,
            "word_alignment": [
                # ABSOLUTE positions within full audio:
                {"word": "foo", "start_ms": 2700, "end_ms": 3200},
                {"word": "bar", "start_ms": 3500, "end_ms": 4500},
            ],
        },
    ]
    flat = _flatten_word_alignment(turns)
    assert len(flat) == 4
    # The pre-fix bug would have produced 2700 + 2500 = 5200 here.
    assert flat[2]["word"] == "foo"
    assert flat[2]["start_ms"] == 2700, \
        f"expected absolute 2700; got {flat[2]['start_ms']} (regression of the +turn_start bug)"
    assert flat[2]["end_ms"] == 3200
    assert flat[3]["word"] == "bar"
    assert flat[3]["start_ms"] == 3500
    # prev_end for turn-2's first word should be the turn boundary, not turn-1's last word
    assert flat[2]["prev_end_ms"] == 2500  # turn 2's start_ms
    # next_start for the last word in a turn should be the turn end
    assert flat[3]["next_start_ms"] == 5000  # turn 2's end_ms
