"""Paired negative sampling for the GBM splice-detector training.

Hypothesis (the friend's): the current `_sample_negatives` in
train_classifier.py samples negatives uniformly from the conversation,
≥1.0s from any boundary. That works for ratio targets but lets the
classifier learn per-file confounds — Opus codec phase, RIR T60, pink
noise SNR, ambience layer — that are roughly constant within a file
but vary file-to-file. Every positive from File A shares those
confounds with every negative from File A → the model can use them as
splice/no-splice signal without ever looking at the splice itself.

Fix: for each positive, sample its paired negative(s) from the SAME
same-speaker turn(s) in the SAME conversation. Speaker, room tone,
ambience, codec, RIR, pink-noise level all stay constant across the
(positive, negative) pair. The classifier is forced to discriminate
splice-specific signal in identical acoustic context.

Speaker inference
-----------------
The regenerated per-conversation JSON on disk does NOT preserve the
source's per-turn (speaker, start_ms, end_ms) metadata, so we infer
turns from `cross_voice` boundaries:

- Source pipeline (verified by dataset author across 7,360 convs):
  strict A→B→A→B alternation. Zero exceptions. Each turn is a single
  ElevenLabs TTS call.
- `cross_voice` boundaries split audio into segments whose parity
  (index % 2) maps to speaker A/B within that conversation.
- `same_voice_edit` boundaries are INTERIOR cuts within a single
  speaker's turn — they do not change speaker.

Identity-of-speaker across conversations is not needed for this fix.

Usage
-----
Drop-in replacement for `_sample_negatives(conv_id, audio_dur_s,
boundary_times, n_neg)` in train_classifier.py:

    from splice.classifier.paired_sampling import sample_negatives_paired

    neg_times = sample_negatives_paired(
        conv_id, audio_dur_s, boundaries, pos_times, n_neg,
    )

`boundaries` must be the FULL list of `{"time_s", "label"}` dicts
(label needed to filter cross_voice from same_voice_edit).
`pos_times` is the candidate positive timestamps that this batch of
negatives is being paired AGAINST.
"""
from __future__ import annotations

import hashlib
import random
from typing import Sequence


# Tuned to korean-iter1 boundary density (median spacing ~3-5s):
DEFAULT_MIN_BUFFER_S = 0.5   # exclude this much around every boundary
DEFAULT_MIN_OFFSET_S = 0.3   # paired negative must be at least this far from its positive
_MIN_INTERVAL_S = 0.05       # drop safe intervals shorter than this


def _rng(*key_parts) -> random.Random:
    """Deterministic per-key RNG. Mirrors train_classifier._rng so the
    paired-sampling output is reproducible across runs.
    """
    h = hashlib.sha256(repr(key_parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _build_segments(
    cv_times: Sequence[float], audio_dur_s: float,
) -> list[tuple[float, float]]:
    """Split audio into segments at cross_voice boundaries.
    Segment i has parity i % 2 → speaker A (even) / speaker B (odd).
    """
    endpoints = [0.0] + sorted(cv_times) + [float(audio_dur_s)]
    return list(zip(endpoints[:-1], endpoints[1:]))


def _parity_of(t: float, segments: list[tuple[float, float]]) -> int:
    """Return parity (0/1 = speaker A/B) of the segment containing t.
    Cross_voice positives sit AT a segment boundary; we deterministically
    assign them to the LEFT segment (the one ending at t).
    """
    for i, (a, b) in enumerate(segments):
        if a <= t < b:
            return i % 2
    # t is at audio_dur_s exactly — last segment
    return (len(segments) - 1) % 2


def _safe_intervals(
    parity: int,
    segments: list[tuple[float, float]],
    boundary_times: list[float],
    exclude_t: float,
    min_buffer_s: float,
    min_offset_s: float,
) -> list[tuple[float, float]]:
    """Return list of (start, end) intervals inside same-parity segments,
    with ±min_buffer_s carved out around every boundary and ±min_offset_s
    around the paired positive `exclude_t`.
    """
    same_segs = [s for i, s in enumerate(segments) if i % 2 == parity]
    intervals: list[tuple[float, float]] = []
    # Sorted exclusion points — boundary radius differs from positive radius
    excl = sorted({*boundary_times, exclude_t})

    for a, b in same_segs:
        inner_a, inner_b = a + min_buffer_s, b - min_buffer_s
        if inner_a >= inner_b:
            continue  # turn too short to host any safe interior
        cursor = inner_a
        for x in excl:
            radius = min_offset_s if x == exclude_t else min_buffer_s
            lo, hi = x - radius, x + radius
            if lo > cursor:
                intervals.append((cursor, min(lo, inner_b)))
            cursor = max(cursor, hi)
            if cursor >= inner_b:
                break
        if cursor < inner_b:
            intervals.append((cursor, inner_b))

    return [(a, b) for a, b in intervals if (b - a) >= _MIN_INTERVAL_S]


def _sample_one_paired(
    pos_t: float,
    parity: int,
    segments: list[tuple[float, float]],
    boundary_times: list[float],
    min_buffer_s: float,
    min_offset_s: float,
    rng: random.Random,
) -> float | None:
    """Sample ONE negative paired with `pos_t`. Returns None if no safe
    same-speaker interior exists at any buffer level.
    """
    # Try progressively looser buffers — boundaries are dense (median ~3-5s
    # spacing), so a positive in a short turn may have zero safe interior
    # at the strict buffer.
    for buffer in (min_buffer_s, min_buffer_s / 2, 0.0):
        intervals = _safe_intervals(
            parity, segments, boundary_times, pos_t,
            buffer, min_offset_s,
        )
        if not intervals:
            continue
        # Length-weighted uniform draw across union of intervals.
        lengths = [b - a for a, b in intervals]
        total = sum(lengths)
        u = rng.uniform(0.0, total)
        acc = 0.0
        for (a, b), L in zip(intervals, lengths):
            acc += L
            if u <= acc:
                return rng.uniform(a, b)
    return None


def sample_negatives_paired(
    conv_id: str,
    audio_dur_s: float,
    boundaries: list[dict],
    pos_times: Sequence[float],
    n_neg: int,
    min_buffer_s: float = DEFAULT_MIN_BUFFER_S,
    min_offset_s: float = DEFAULT_MIN_OFFSET_S,
) -> list[float]:
    """Sample `n_neg` negatives, each paired to a positive in `pos_times`.

    Cycles through positives if n_neg > len(pos_times) (e.g., NEG_RATIO=2
    yields 2 paired negatives per positive). Each replicate uses a fresh
    RNG seed so the two paired negatives for the same positive are
    independent.

    Falls back to uniform-random over the whole conversation (the OLD
    behavior) ONLY if no positives exist (cannot pair to nothing) OR
    if every per-positive pairing attempt at every buffer level returns
    None (degenerate conversation).

    Parameters
    ----------
    conv_id : str
        For deterministic seeding. Same input → same output.
    audio_dur_s : float
        Total audio duration in seconds.
    boundaries : list[dict]
        Full ground-truth list of `{"time_s": float, "label": str}`.
        Labels in {"cross_voice", "same_voice_edit"}.
    pos_times : Sequence[float]
        Positive timestamps to pair against.
    n_neg : int
        Total negatives to return.
    min_buffer_s : float
        Distance from any boundary that a negative must maintain.
    min_offset_s : float
        Distance from the paired positive that a negative must maintain.
    """
    if not pos_times:
        # No positives in this conversation → cannot pair. Fall back.
        from splice.classifier.train_classifier import _sample_negatives
        boundary_times = [b["time_s"] for b in boundaries]
        return _sample_negatives(conv_id, audio_dur_s, boundary_times, n_neg)

    cv_times = [b["time_s"] for b in boundaries if b.get("label") == "cross_voice"]
    boundary_times = [b["time_s"] for b in boundaries]
    segments = _build_segments(cv_times, audio_dur_s)

    results: list[float] = []
    n_pos = len(pos_times)
    fallback_count = 0

    for i in range(n_neg):
        pos_idx = i % n_pos
        replicate = i // n_pos
        pos_t = float(pos_times[pos_idx])
        parity = _parity_of(pos_t, segments)
        rng = _rng(conv_id, pos_t, replicate)
        sampled = _sample_one_paired(
            pos_t, parity, segments, boundary_times,
            min_buffer_s, min_offset_s, rng,
        )
        if sampled is None:
            fallback_count += 1
            # Last-resort fallback: uniform random in conversation.
            from splice.classifier.train_classifier import _sample_negatives
            sampled = _sample_negatives(
                conv_id, audio_dur_s, boundary_times, 1,
            )[0]
        results.append(float(sampled))

    return results
