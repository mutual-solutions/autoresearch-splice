#!/usr/bin/env python3
"""Per-domain file-level feature distribution probe for the EVAL corpus (US-519).

Walks data/eval/{singing,korean,english}/{tier1,tier2,clean}/ and computes
three scalars per audio file:
  - hpr       : harmonic power ratio, harmonic/(harmonic+percussive) on STFT
  - flatness  : mean spectral flatness across frames
  - voicing   : fraction of YIN pitch frames with finite (voiced) f0

Reports per-domain mean + {min, p25, median, p75, max, n} distribution to
stdout.  Use --json for machine-readable output suitable for wrapper prompt
injection.

This is a pure read-only operator tool.  It does not touch evaluate.py,
program.md, or any protected file.

Usage:
  uv run python scripts/per_domain_file_stats.py [--data-dir DIR]
      [--domains singing,korean,english] [--tiers tier1,tier2,clean]
      [--json] [--sample N]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent

SCALARS = ("hpr", "flatness", "voicing")
ALL_DOMAINS = ("singing", "korean", "english")
ALL_TIERS = ("tier1", "tier2", "clean")
AUDIO_EXTS = {".wav", ".flac", ".opus", ".mp3"}


def file_stats(wav_path: Path) -> dict:
    """Compute HPR, spectral flatness, and voicing fraction for one audio file.

    Returns a dict with keys hpr, flatness, voicing, or {"skip": True, "reason": ...}.
    """
    try:
        audio, sr = sf.read(str(wav_path))
    except Exception as exc:
        return {"skip": True, "reason": f"read_error: {exc}"}

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if audio.size < sr:  # < 1 second
        return {"skip": True, "reason": "too_short"}

    # Silence warnings from librosa (e.g. PySoundFile deprecation notices)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # HPR: harmonic / (harmonic + percussive) on magnitude STFT
        S = np.abs(librosa.stft(audio))
        h, p = librosa.decompose.hpss(S)
        hpr = float(h.sum() / (h.sum() + p.sum() + 1e-10))

        # Spectral flatness (geometric-mean / arithmetic-mean ratio per frame, then mean)
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=S + 1e-10)))

        # Voicing fraction via YIN pitch tracker; voiced = finite f0
        f0 = librosa.yin(audio, fmin=50, fmax=600, sr=sr)
        voicing = float(np.isfinite(f0).mean()) if len(f0) else 0.0

    return {"hpr": hpr, "flatness": flatness, "voicing": voicing}


def _collect(
    data_dir: Path,
    domains: list[str],
    tiers: list[str],
    sample: int | None,
) -> dict[str, dict[str, list[float]]]:
    """Walk the eval tree and return {domain: {scalar: [values...]}}."""
    results: dict[str, dict[str, list[float]]] = {
        d: {s: [] for s in SCALARS} for d in domains
    }

    for domain in domains:
        file_count = 0
        for tier in tiers:
            tier_dir = data_dir / domain / tier
            if not tier_dir.exists():
                print(
                    f"WARN: {tier_dir} not found — skipping"
                    " (eval corpus may not be decrypted)",
                    file=sys.stderr,
                )
                continue

            audio_files = sorted(
                f for f in tier_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS
            )

            if sample is not None:
                # Apply --sample cap per domain (not per tier) by tracking total
                remaining = sample - file_count
                if remaining <= 0:
                    break
                audio_files = audio_files[:remaining]

            for path in audio_files:
                stats = file_stats(path)
                if stats.get("skip"):
                    print(
                        f"WARN: skipping {path.name}: {stats['reason']}",
                        file=sys.stderr,
                    )
                    continue
                for scalar in SCALARS:
                    results[domain][scalar].append(stats[scalar])
                file_count += 1

    return results


def _distribution(values: list[float]) -> dict:
    """Return {n, mean, min, p25, median, p75, max} for a list of floats."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "min": None, "p25": None,
                "median": None, "p75": None, "max": None}
    arr = np.array(values, dtype=float)
    return {
        "n": n,
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def _build_report(
    collected: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict]]:
    """Build {domain: {scalar: distribution_dict}} report."""
    report: dict[str, dict[str, dict]] = {}
    for domain, scalars in collected.items():
        report[domain] = {}
        for scalar, values in scalars.items():
            report[domain][scalar] = _distribution(values)
    return report


def _print_table(report: dict[str, dict[str, dict]]) -> None:
    """Print a human-readable plain-text table to stdout."""
    header = f"{'domain':<10} {'scalar':<10} {'n':>5}  {'mean':>6}  {'min':>6}  {'p25':>6}  {'median':>6}  {'p75':>6}  {'max':>6}"
    print(header)
    print("-" * len(header))

    for domain in sorted(report):
        for scalar in SCALARS:
            d = report[domain].get(scalar, {})
            n = d.get("n", 0)
            if n == 0:
                print(f"{'domain':<10} {scalar:<10} {'0':>5}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}")
                continue
            print(
                f"{domain:<10} {scalar:<10} {n:>5}"
                f"  {d['mean']:>6.3f}  {d['min']:>6.3f}"
                f"  {d['p25']:>6.3f}  {d['median']:>6.3f}"
                f"  {d['p75']:>6.3f}  {d['max']:>6.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-domain EVAL-corpus distribution of HPR, spectral flatness, and voicing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        default=str(REPO / "data" / "eval"),
        help="Root of the eval corpus (default: data/eval relative to repo root).",
    )
    parser.add_argument(
        "--domains",
        default=",".join(ALL_DOMAINS),
        help="Comma-separated list of domains to analyse (default: singing,korean,english).",
    )
    parser.add_argument(
        "--tiers",
        default=",".join(ALL_TIERS),
        help="Comma-separated list of tiers to include (default: tier1,tier2,clean).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Emit JSON instead of plain-text table.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N files per domain (across all tiers) for quick probes.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(
            f"WARN: data dir {data_dir} does not exist — "
            "eval corpus may not be decrypted. Nothing to report.",
            file=sys.stderr,
        )
        return 0

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    invalid_domains = [d for d in domains if d not in ALL_DOMAINS]
    if invalid_domains:
        print(f"ERROR: unknown domains: {invalid_domains}", file=sys.stderr)
        return 1

    invalid_tiers = [t for t in tiers if t not in ALL_TIERS]
    if invalid_tiers:
        print(f"ERROR: unknown tiers: {invalid_tiers}", file=sys.stderr)
        return 1

    collected = _collect(data_dir, domains, tiers, args.sample)
    report = _build_report(collected)

    any_data = any(
        report[d][s]["n"] > 0
        for d in report
        for s in SCALARS
        if s in report[d]
    )
    if not any_data:
        print(
            "WARN: no audio files found — eval corpus may not be decrypted.",
            file=sys.stderr,
        )
        return 0

    if args.emit_json:
        json.dump(report, sys.stdout, indent=2)
        print()  # trailing newline
    else:
        _print_table(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
