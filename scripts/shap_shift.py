#!/usr/bin/env python3
"""SHAP shift sentinel: classify the latest keep as STRUCTURAL or LOCAL (US-507).

Reads the last 2 keep-status commits from results.tsv, computes top-3
features by summed |shap| per domain for each, and emits a Jaccard
similarity based classification:

  - Jaccard(top3_prev, top3_new) < 0.5 for ANY domain  → STRUCTURAL SHIFT
  - else                                                → LOCAL KEEP

The wrapper pipes the one-line verdict into the next iteration's
research-notes reflection header so claude sees the classification.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO / "reports"
RESULTS_TSV = REPO / "results.tsv"
TOP_K = 3
JACCARD_STRUCTURAL_THRESHOLD = 0.5


def _recent_keeps(n: int = 2) -> list[str]:
    if not RESULTS_TSV.exists():
        return []
    keeps: list[str] = []
    with open(RESULTS_TSV) as f:
        header = f.readline()
        if not header:
            return []
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
            if parts[status_idx] == "keep" and parts[sha_idx] not in ("", "NA"):
                keeps.append(parts[sha_idx])
    return keeps[-n:]


def _top_features_for_commit(sha: str) -> dict[str, set[str]]:
    """Return {domain: {top-K feature names}} for a single keep commit."""
    per_domain_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    root = REPORTS_DIR / sha
    if not root.is_dir():
        return {}
    for domain_dir in root.iterdir():
        if not domain_dir.is_dir() or domain_dir.name == "spliced":
            continue
        for p in domain_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for feat in data.get("top_features") or []:
                name = feat.get("name")
                shap = feat.get("shap")
                if name is None or shap is None:
                    continue
                try:
                    per_domain_sums[domain_dir.name][name] += abs(float(shap))
                except (TypeError, ValueError):
                    continue
    return {
        dom: set(name for name, _ in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K])
        for dom, d in per_domain_sums.items()
    }


def classify(prev_sha: str, new_sha: str) -> str:
    prev = _top_features_for_commit(prev_sha)
    new = _top_features_for_commit(new_sha)
    domains = sorted(set(prev) | set(new))
    if not domains:
        return f"(no SHAP data for either {prev_sha[:7]} or {new_sha[:7]})"

    min_jaccard = 1.0
    shift_details: list[str] = []
    for dom in domains:
        a, b = prev.get(dom, set()), new.get(dom, set())
        if not a or not b:
            continue
        j = len(a & b) / len(a | b) if (a | b) else 1.0
        if j < JACCARD_STRUCTURAL_THRESHOLD:
            shift_details.append(
                f"{dom} top-3 changed from {sorted(a)} to {sorted(b)}"
            )
        min_jaccard = min(min_jaccard, j)

    if shift_details:
        return (f"STRUCTURAL SHIFT at keep {new_sha[:7]} (min Jaccard {min_jaccard:.2f}): "
                + "; ".join(shift_details))
    return f"LOCAL KEEP at {new_sha[:7]}: top-3 features stable across all domains"


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    keeps = _recent_keeps(n=2)
    if len(keeps) < 2:
        print("(not enough keep history to compute shift)")
        return 0
    print(classify(keeps[-2], keeps[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
