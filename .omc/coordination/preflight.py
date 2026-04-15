#!/usr/bin/env python3
"""Pre-flight checks for autoresearch-splice agent coordination.

Run before evaluation to verify dataset integrity.
Usage: uv run python .omc/coordination/preflight.py
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MANIFEST_PATH = SCRIPT_DIR / "manifest.json"


def load_manifest() -> dict:
    """Load manifest.json from the coordination directory."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def get_agent_id() -> str:
    """Generate agent ID from branch name + PID."""
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        branch = "unknown"
    return f"{branch}_{os.getpid()}"


def check_data_integrity(manifest: dict) -> list[str]:
    """Verify file counts and ground truth hash against manifest."""
    errors = []
    data_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent / manifest["dataset"]

    if not data_dir.exists():
        errors.append(f"Dataset directory not found: {data_dir}")
        return errors

    # Check file counts per subdir
    expected = manifest["expected_counts"]
    for subdir, expected_count in expected.items():
        if subdir.startswith("_"):
            continue  # skip notes
        if not isinstance(expected_count, int):
            continue
        subdir_path = data_dir / subdir
        if not subdir_path.exists():
            errors.append(f"Missing subdir: {subdir_path}")
            continue
        actual = len([f for f in subdir_path.iterdir()
                      if f.is_file() and f.name != ".DS_Store"])
        if actual != expected_count:
            errors.append(f"{subdir}: expected {expected_count} files, found {actual}")

    # Check ground truth hash
    gt_path = data_dir / "ground_truth.json"
    if not gt_path.exists():
        errors.append(f"ground_truth.json not found at {gt_path}")
    else:
        sha = hashlib.sha256(gt_path.read_bytes()).hexdigest()[:12]
        expected_sha = manifest.get("ground_truth_sha256_prefix", "")
        if sha != expected_sha:
            errors.append(f"ground_truth.json hash mismatch: got {sha}, expected {expected_sha}")

    # Check total file count
    total_expected = manifest.get("total_files")
    if total_expected:
        actual_total = sum(
            1 for f in data_dir.rglob("*")
            if f.is_file() and f.name != ".DS_Store"
        )
        if actual_total != total_expected:
            errors.append(f"Total files: expected {total_expected}, found {actual_total}")

    return errors


def check_permissions(data_dir: str) -> list[str]:
    """Check if data directory is read-only. Returns warnings (not errors)."""
    warnings = []
    path = Path(data_dir)
    resolved = path.resolve()
    if os.access(str(resolved), os.W_OK):
        warnings.append(f"WARNING: {resolved} is writable. Run chmod 555 for protection.")
    return warnings


def run_preflight() -> bool:
    """Run all pre-flight checks. Returns True if all critical checks pass."""
    # Skip if env var set
    if os.environ.get("SKIP_PREFLIGHT") == "1":
        print("  Preflight: SKIP (SKIP_PREFLIGHT=1)")
        return True

    agent_id = get_agent_id()
    print(f"  Preflight [{agent_id}]:")

    try:
        manifest = load_manifest()
    except FileNotFoundError as e:
        print(f"    FAIL: {e}")
        return False

    # Critical: data integrity
    errors = check_data_integrity(manifest)
    if errors:
        for e in errors:
            print(f"    FAIL: {e}")
        return False
    print(f"    Data integrity: PASS ({manifest.get('total_files', '?')} files)")

    # Advisory: permissions
    data_dir = manifest["dataset"]
    warnings = check_permissions(data_dir)
    for w in warnings:
        print(f"    {w}")
    if not warnings:
        print("    Permissions: PASS (read-only)")

    print("    All checks passed.")
    return True


if __name__ == "__main__":
    ok = run_preflight()
    sys.exit(0 if ok else 1)
