#!/usr/bin/env python3
"""Pre-flight checks for autoresearch-splice agent coordination.

Run before evaluation to verify dataset integrity.
Usage: PYTHONPATH=$PWD uv run python autoresearch/preflight.py
"""

import hashlib
import json
import os
import random
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


def _resolve_data_dir(manifest_dataset: str) -> Path:
    """Manifest records the dataset as `data/eval/<id>` relative to the repo.
    When the wrapper has decrypted eval into a tmp dir and exported
    OMC_EVAL_DATA_ROOT, rewrite `data/eval/<id>` → `<tmp>/<id>` so preflight
    checks the live tree instead of the (missing) plaintext path.
    """
    # Post-US-516: preflight.py lives at `autoresearch/preflight.py`, ONE
    # level below the repo root. Previously at `.omc/coordination/preflight.py`
    # (two levels), where the extra `.parent` was correct.
    repo_root = SCRIPT_DIR.parent
    override = os.environ.get("OMC_EVAL_DATA_ROOT")
    prefix = "data/eval/"
    if override and manifest_dataset.startswith(prefix):
        return Path(override) / manifest_dataset[len(prefix):]
    return repo_root / manifest_dataset


def check_data_integrity(manifest: dict) -> list[str]:
    """Verify file counts and ground truth hash against manifest."""
    errors = []
    data_dir = _resolve_data_dir(manifest["dataset"])

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


def check_env_fingerprint(repo_root: Path) -> dict:
    """Compare live env versions against .omc/korean-iter1-env-fingerprint.json.

    Returns a result dict with 'status' key: 'ok', 'warn', or 'skip'.
    Emits preflight.env_fingerprint.{ok,warn,skip} events.
    """
    from autoresearch.logger import get_logger
    log = get_logger("preflight")

    fp_path = repo_root / ".omc" / "korean-iter1-env-fingerprint.json"
    if not fp_path.exists():
        log.emit("INFO", "preflight.env_fingerprint.skip", reason="no fingerprint file")
        return {"status": "skip", "reason": "no fingerprint file"}

    try:
        fp = json.loads(fp_path.read_text())
    except Exception as exc:
        log.emit("WARN", "preflight.env_fingerprint.skip", reason=f"parse error: {exc}")
        return {"status": "skip", "reason": f"parse error: {exc}"}

    import librosa  # type: ignore
    import numpy  # type: ignore
    import soundfile  # type: ignore
    import scipy  # type: ignore

    live = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "librosa": librosa.__version__,
        "soundfile": soundfile.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }
    expected = {"python_version": fp["python_version"], **fp["deps"]}
    diffs = {k: (expected[k], live[k]) for k in expected if expected.get(k) != live.get(k)}

    if diffs:
        log.emit("WARN", "preflight.env_fingerprint.warn", diffs=str(diffs),
                 msg="env diverges from fingerprint; regen may not be byte-deterministic")
        return {"status": "warn", "diffs": diffs,
                "msg": "env diverges from fingerprint; regen may not be byte-deterministic"}
    log.emit("INFO", "preflight.env_fingerprint.ok")
    return {"status": "ok"}


def check_audio_sha_spot_sample(eval_dir: Path, sample_size: int = 5, seed: int = 0) -> dict:
    """Spot-check 5 random audio files from _manifest.jsonl against their SHA-256.

    Returns a result dict with 'status' key: 'ok', 'fail', or 'skip'.
    Emits preflight.audio_sha_spot.{ok,fail,skip} events.
    """
    from autoresearch.logger import get_logger
    log = get_logger("preflight")

    manifest_path = eval_dir / "_manifest.jsonl"
    if not manifest_path.exists():
        log.emit("INFO", "preflight.audio_sha_spot.skip", reason="no _manifest.jsonl")
        return {"status": "skip", "reason": "no _manifest.jsonl"}

    try:
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    except Exception as exc:
        log.emit("WARN", "preflight.audio_sha_spot.skip", reason=f"parse error: {exc}")
        return {"status": "skip", "reason": f"parse error: {exc}"}

    if not rows:
        log.emit("INFO", "preflight.audio_sha_spot.skip", reason="empty manifest")
        return {"status": "skip", "reason": "empty manifest"}

    rng = random.Random(seed)
    sample = rng.sample(rows, min(sample_size, len(rows)))

    mismatches = []
    for row in sample:
        opus_path = eval_dir / f"{row['conv_id']}.opus"
        if not opus_path.exists():
            mismatches.append({"conv_id": row["conv_id"], "reason": "file missing"})
            continue
        live_sha = hashlib.sha256(opus_path.read_bytes()).hexdigest()
        expected_sha = row.get("audio_sha256")
        if live_sha != expected_sha:
            mismatches.append({"conv_id": row["conv_id"], "expected": expected_sha, "got": live_sha})

    if mismatches:
        log.emit("ERROR", "preflight.audio_sha_spot.fail",
                 mismatches=str(mismatches), sample_size=len(sample))
        return {"status": "fail", "mismatches": mismatches, "sample_size": len(sample)}
    log.emit("INFO", "preflight.audio_sha_spot.ok", sample_size=len(sample))
    return {"status": "ok", "sample_size": len(sample)}


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
    data_ok = not errors
    if errors:
        for e in errors:
            print(f"    FAIL: {e}")
    else:
        print(f"    Data integrity: PASS ({manifest.get('total_files', '?')} files)")

    # Advisory: permissions (only meaningful when data dir exists)
    data_dir = str(_resolve_data_dir(manifest["dataset"]))
    if data_ok:
        warnings = check_permissions(data_dir)
        for w in warnings:
            print(f"    {w}")
        if not warnings:
            print("    Permissions: PASS (read-only)")

    # Advisory: env fingerprint check (WARN only, never fails preflight)
    repo_root = SCRIPT_DIR.parent
    fp_result = check_env_fingerprint(repo_root)
    if fp_result["status"] == "warn":
        print(f"    Env fingerprint: WARN — {fp_result.get('msg', '')} diffs={fp_result.get('diffs', {})}")
    elif fp_result["status"] == "ok":
        print("    Env fingerprint: PASS (matches)")
    else:
        print(f"    Env fingerprint: SKIP ({fp_result.get('reason', '')})")

    # Critical: audio SHA spot check (FAIL if mismatches found)
    resolved_eval_dir = _resolve_data_dir(manifest["dataset"])
    sha_result = check_audio_sha_spot_sample(resolved_eval_dir)
    if sha_result["status"] == "fail":
        print(f"    Audio SHA spot check: FAIL — {sha_result['mismatches']}")
        return False
    elif sha_result["status"] == "ok":
        print(f"    Audio SHA spot check: PASS ({sha_result.get('sample_size', '?')} files sampled)")
    else:
        print(f"    Audio SHA spot check: SKIP ({sha_result.get('reason', '')})")

    if not data_ok:
        return False

    print("    All checks passed.")
    return True


if __name__ == "__main__":
    ok = run_preflight()
    sys.exit(0 if ok else 1)
