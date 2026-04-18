#!/usr/bin/env python3
"""Roll up SHAP feature importance across recent keeps → `.omc/shap_rollup.json`.

Reads `reports/<short_sha>/<domain>/*.json` sidecars (written by
ml_eval's SHAP exporter) for each recent `keep` in `results.tsv`, sums
|shap| per feature per domain, and emits the top K. The autoresearch
wrapper injects this into claude's prompt so feature-engineering
hypotheses are grounded in what's actually driving keeps.

Output schema (locked by PRD US-500):
{"generated_at": iso8601, "keeps_considered": [sha, ...],
 "per_domain": {dom: [{name, sum_abs_shap, mean_value,
                       n_detections, rank}, ...]}}
Empty when no keeps exist — the wrapper block handles that case.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO / "reports"
RESULTS_TSV = REPO / "results.tsv"
DEFAULT_OUTPUT = REPO / ".omc" / "shap_rollup.json"
DEFAULT_KEEPS = 5
TOP_K = 10


def recent_keep_shas(n: int) -> list[str]:
    """Return the last N keep-status short SHAs from results.tsv (oldest→newest)."""
    if n <= 0 or not RESULTS_TSV.exists():
        return []
    keeps: list[str] = []
    with open(RESULTS_TSV) as f:
        header = f.readline()
        if not header:
            return []
        # Locate columns by header — survives future TSV schema widening.
        cols = header.rstrip("\n").split("\t")
        try:
            sha_idx = cols.index("commit")
            status_idx = cols.index("status")
        except ValueError:
            return []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(sha_idx, status_idx):
                continue
            if parts[status_idx] == "keep" and parts[sha_idx] and parts[sha_idx] != "NA":
                keeps.append(parts[sha_idx])
    return keeps[-n:]


def iter_sidecars_for_commit(sha: str):
    """Yield (domain, json_path) for every SHAP sidecar under reports/<sha>/<domain>/*.json.

    Skips the pre-split `reports/<sha>/spliced/` layout — those commits
    predate per-domain bucketing and can't be attributed to a domain.
    """
    root = REPORTS_DIR / sha
    if not root.is_dir():
        return
    for domain_dir in sorted(root.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name == "spliced":
            continue
        domain = domain_dir.name
        for json_path in sorted(domain_dir.glob("*.json")):
            yield domain, json_path


def aggregate_per_domain(
    shas: list[str],
) -> dict[str, list[dict]]:
    """Sum |shap| per (domain, feature_name) across all sidecars for the given commits."""
    agg: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(
        lambda: {"sum_abs_shap": 0.0, "value_sum": 0.0, "n": 0}
    ))
    for sha in shas:
        for domain, json_path in iter_sidecars_for_commit(sha):
            try:
                data = json.loads(json_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            top = data.get("top_features") or []
            for feat in top:
                name = feat.get("name")
                shap = feat.get("shap")
                value = feat.get("value")
                if name is None or shap is None:
                    continue
                try:
                    abs_shap = abs(float(shap))
                    value_f = float(value) if value is not None else 0.0
                except (TypeError, ValueError):
                    continue
                slot = agg[domain][name]
                slot["sum_abs_shap"] += abs_shap
                slot["value_sum"] += value_f
                slot["n"] += 1

    out: dict[str, list[dict]] = {}
    for domain, feats in agg.items():
        ranked = sorted(
            feats.items(),
            key=lambda kv: kv[1]["sum_abs_shap"],
            reverse=True,
        )[:TOP_K]
        out[domain] = [
            {
                "name": name,
                "sum_abs_shap": round(slot["sum_abs_shap"], 4),
                "mean_value": round(slot["value_sum"] / slot["n"], 4) if slot["n"] else 0.0,
                "n_detections": slot["n"],
                "rank": i + 1,
            }
            for i, (name, slot) in enumerate(ranked)
        ]
    return out


def build_rollup(n_keeps: int) -> dict:
    shas = recent_keep_shas(n_keeps)
    per_domain = aggregate_per_domain(shas)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "keeps_considered": shas,
        "per_domain": per_domain,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keeps", type=int, default=DEFAULT_KEEPS,
                    help=f"Number of recent keeps to aggregate over (default: {DEFAULT_KEEPS})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Output path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--stdout", action="store_true",
                    help="Print to stdout instead of writing to --output")
    args = ap.parse_args()

    rollup = build_rollup(args.keeps)

    blob = json.dumps(rollup, indent=2, sort_keys=False)
    if args.stdout:
        print(blob)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(blob + "\n")
        n_doms = len(rollup["per_domain"])
        n_keeps = len(rollup["keeps_considered"])
        print(f"shap_rollup: wrote {args.output} ({n_doms} domain(s), {n_keeps} keep(s))",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
