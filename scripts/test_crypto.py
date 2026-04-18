#!/usr/bin/env python3
"""Encrypt / decrypt the held-out test dataset with a Keychain-stored,
Touch-ID-gated AES-256-GCM key.

The key lives in macOS Keychain via scripts/test_unlock (Swift). This
script never holds the key on disk; it reads it on demand through the
Swift helper, which triggers the Touch ID prompt. Uses:

  setup       generate a fresh 32-byte key, store it (Touch ID prompt once
              at set time), tar+gzip+encrypt data/test/ → data/test.tar.gz.enc,
              shred the plaintext tree.
  encrypt     re-encrypt data/test/ → data/test.tar.gz.enc using the
              existing Keychain key (Touch ID prompt to read key).
              Intended to run after regenerate_datasets.py --split test.
  decrypt     decrypt data/test.tar.gz.enc → /tmp/.ar-test-XXXXXX/ with
              0700 perms. Prints the tmp dir path on stdout so callers
              can pipe it into --data-dir or env. Touch ID prompt fires.
  status      inspect whether the encrypted blob exists, whether the
              plaintext tree is present, and whether the Keychain item
              is readable (without actually decrypting).

The decrypt path registers an atexit handler so the tmp dir is shredded
on normal exit; callers that want persistence across script lifetimes
should use `--keep` and handle cleanup themselves.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

REPO = Path(__file__).resolve().parent.parent
PLAINTEXT_DIR = REPO / "data" / "test"
ENCRYPTED_BLOB = REPO / "data" / "test.tar.gz.enc"
UNLOCK_HELPER = REPO / "scripts" / "test_unlock"

# GCM nonce is 12 bytes (recommended). Key is 32 bytes (AES-256).
NONCE_LEN = 12
KEY_LEN = 32

# Blob layout: version(1) + nonce(12) + ciphertext(var) + tag(16, embedded).
# `cryptography` AESGCM.encrypt returns ciphertext || tag, so we write
# version || nonce || (ciphertext || tag). Version byte lets us rotate
# formats later without a flag day.
BLOB_VERSION = b"\x01"


# --- helpers ---------------------------------------------------------------


def _die(msg: str, code: int = 1) -> None:
    print(f"test_crypto: {msg}", file=sys.stderr)
    sys.exit(code)


def _check_helper() -> None:
    if not UNLOCK_HELPER.exists():
        _die(
            f"{UNLOCK_HELPER} not found. Compile it first:\n"
            "  swiftc scripts/test_unlock.swift "
            "-framework LocalAuthentication -framework Security "
            "-o scripts/test_unlock"
        )
    if not os.access(UNLOCK_HELPER, os.X_OK):
        _die(f"{UNLOCK_HELPER} is not executable (chmod +x it)")


def _store_key(key: bytes) -> None:
    """Push key into Keychain via the Swift helper. Prompts Touch ID once."""
    b64 = base64.b64encode(key).decode()
    r = subprocess.run(
        [str(UNLOCK_HELPER), "--set-key"],
        input=b64, text=True, capture_output=True,
    )
    if r.returncode != 0:
        _die(f"test_unlock --set-key failed: {r.stderr.strip()}")


def _fetch_key() -> bytes:
    """Pull key from Keychain via the Swift helper. Prompts Touch ID
    every invocation (fresh LAContext per process).
    """
    _check_helper()
    r = subprocess.run(
        [str(UNLOCK_HELPER)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        _die(f"test_unlock failed: {r.stderr.strip() or 'user cancel?'}")
    try:
        return base64.b64decode(r.stdout.strip(), validate=True)
    except Exception as e:
        _die(f"test_unlock returned invalid base64: {e}")


def _shred_dir(path: Path) -> None:
    """Best-effort shred — overwrite files then unlink. Not guaranteed on
    APFS (CoW may preserve blocks); mostly defensive against trivial
    inspection of free blocks.
    """
    if not path.exists():
        return
    for p in path.rglob("*"):
        if p.is_file():
            try:
                size = p.stat().st_size
                with open(p, "r+b") as f:
                    f.write(secrets.token_bytes(min(size, 1 << 20)))
                    f.flush()
                    os.fsync(f.fileno())
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


# --- tarball / crypto ------------------------------------------------------


def _tar_gz_stream(src_dir: Path) -> bytes:
    """Pack src_dir into a gzipped tarball, return the bytes in memory.
    The full dataset is ~1.1 GB / ~350 MB gzipped so in-memory is fine.
    """
    import io

    buf = io.BytesIO()
    # Use deterministic mode-stripping: encrypt blob stays stable across
    # filesystems that disagree on permission bits.
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as tar:
        for entry in sorted(src_dir.rglob("*")):
            if entry.is_symlink():
                continue
            rel = entry.relative_to(src_dir.parent)
            tar.add(str(entry), arcname=str(rel), recursive=False)
    return buf.getvalue()


def _extract_tar_gz(data: bytes, dest_root: Path) -> None:
    import io

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(path=dest_root)


def _encrypt(key: bytes, plaintext: bytes) -> bytes:
    aes = AESGCM(key)
    nonce = secrets.token_bytes(NONCE_LEN)
    ct = aes.encrypt(nonce, plaintext, associated_data=None)
    return BLOB_VERSION + nonce + ct


def _decrypt(key: bytes, blob: bytes) -> bytes:
    if not blob.startswith(BLOB_VERSION):
        _die(f"blob format version mismatch (got {blob[:1]!r}, expected {BLOB_VERSION!r})")
    nonce = blob[1 : 1 + NONCE_LEN]
    ct = blob[1 + NONCE_LEN :]
    aes = AESGCM(key)
    try:
        return aes.decrypt(nonce, ct, associated_data=None)
    except Exception as e:
        _die(f"decryption failed (wrong key or tampered blob): {e}")


# --- commands --------------------------------------------------------------


def cmd_setup(args) -> None:
    if not PLAINTEXT_DIR.exists():
        _die(f"{PLAINTEXT_DIR} not found. Regenerate with "
             "`uv run python data_synth/regenerate_datasets.py --domain all --split test`")
    _check_helper()
    print("Generating fresh 32-byte AES-256 key…")
    key = secrets.token_bytes(KEY_LEN)
    print("Storing key in macOS Keychain (Touch ID prompt on this step)…")
    _store_key(key)

    print("Tarring + gzipping data/test/…")
    plaintext = _tar_gz_stream(PLAINTEXT_DIR)
    print(f"  plaintext size: {len(plaintext):,} bytes "
          f"({len(plaintext)/1024/1024:.1f} MiB)")

    print("Encrypting with AES-256-GCM…")
    blob = _encrypt(key, plaintext)
    ENCRYPTED_BLOB.parent.mkdir(parents=True, exist_ok=True)
    tmp = ENCRYPTED_BLOB.with_suffix(".enc.tmp")
    tmp.write_bytes(blob)
    tmp.replace(ENCRYPTED_BLOB)
    print(f"  wrote {ENCRYPTED_BLOB} ({len(blob):,} bytes)")

    if args.keep_plaintext:
        print("  (--keep-plaintext: leaving data/test/ in place)")
    else:
        print("Shredding plaintext data/test/…")
        _shred_dir(PLAINTEXT_DIR)
        print(f"  removed {PLAINTEXT_DIR}")

    # Zero-out the key reference we still hold in this process.
    del key
    del blob


def cmd_encrypt(args) -> None:
    """Re-encrypt data/test/ with the existing Keychain key. Used post-regen."""
    if not PLAINTEXT_DIR.exists():
        _die(f"{PLAINTEXT_DIR} not found — nothing to encrypt")
    _check_helper()

    print("Fetching key from Keychain (Touch ID prompt)…")
    key = _fetch_key()

    print("Tarring + gzipping data/test/…")
    plaintext = _tar_gz_stream(PLAINTEXT_DIR)
    print(f"  plaintext size: {len(plaintext):,} bytes")

    print("Encrypting…")
    blob = _encrypt(key, plaintext)
    tmp = ENCRYPTED_BLOB.with_suffix(".enc.tmp")
    tmp.write_bytes(blob)
    tmp.replace(ENCRYPTED_BLOB)
    print(f"  wrote {ENCRYPTED_BLOB}")

    if not args.keep_plaintext:
        print("Shredding plaintext data/test/…")
        _shred_dir(PLAINTEXT_DIR)

    del key
    del blob


def cmd_decrypt(args) -> None:
    if not ENCRYPTED_BLOB.exists():
        _die(f"{ENCRYPTED_BLOB} not found")
    _check_helper()

    print("Fetching key from Keychain (Touch ID prompt)…", file=sys.stderr)
    key = _fetch_key()

    blob = ENCRYPTED_BLOB.read_bytes()
    print(f"Decrypting {len(blob):,} bytes…", file=sys.stderr)
    plaintext = _decrypt(key, blob)

    # Mint an ephemeral dir with restrictive perms.
    root = Path(tempfile.mkdtemp(prefix=".ar-test-"))
    os.chmod(root, 0o700)

    if not args.keep:
        # Register cleanup on any exit path.
        def _cleanup():
            _shred_dir(root)
        atexit.register(_cleanup)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: sys.exit(130))

    _extract_tar_gz(plaintext, root)
    # Extracted layout: <root>/test/<domain>/tier{1,2}/clean/...
    # Print the inner "test" dir so callers can use it directly as
    # a set of dataset roots.
    test_root = root / "test"
    if not test_root.exists():
        # Tarball was packed from data/, so top-level is "test"
        candidates = [c for c in root.iterdir() if c.is_dir()]
        if len(candidates) == 1:
            test_root = candidates[0]
        else:
            _die(f"unexpected tarball layout under {root}")
    print(str(test_root))  # stdout — the path consumers use
    del key
    del plaintext
    del blob

    if args.keep:
        # Don't cleanup; caller is responsible.
        pass
    else:
        # Wait for caller to signal done, or keep running while they
        # consume the path. For now, just hold the dir alive for N
        # seconds if --hold is set; otherwise exit (atexit fires).
        if args.hold > 0:
            import time
            time.sleep(args.hold)


def cmd_status(args) -> None:
    print(f"plaintext dir   : {PLAINTEXT_DIR}  "
          f"({'PRESENT' if PLAINTEXT_DIR.exists() else 'absent'})")
    print(f"encrypted blob  : {ENCRYPTED_BLOB}  "
          f"({'PRESENT (' + str(ENCRYPTED_BLOB.stat().st_size) + ' bytes)' if ENCRYPTED_BLOB.exists() else 'absent'})")
    print(f"unlock helper   : {UNLOCK_HELPER}  "
          f"({'executable' if (UNLOCK_HELPER.exists() and os.access(UNLOCK_HELPER, os.X_OK)) else 'missing'})")


# --- entry -----------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)

    s_setup = sp.add_parser("setup",
        help="first-time encryption: generate key, encrypt data/test/, "
             "shred plaintext")
    s_setup.add_argument("--keep-plaintext", action="store_true",
        help="don't shred data/test/ after encrypting (debug only)")
    s_setup.set_defaults(func=cmd_setup)

    s_enc = sp.add_parser("encrypt",
        help="re-encrypt data/test/ (post regen-datasets) using the "
             "existing Keychain key")
    s_enc.add_argument("--keep-plaintext", action="store_true",
        help="don't shred data/test/ after encrypting (debug only)")
    s_enc.set_defaults(func=cmd_encrypt)

    s_dec = sp.add_parser("decrypt",
        help="decrypt to a fresh /tmp/.ar-test-* dir, print path to stdout")
    s_dec.add_argument("--keep", action="store_true",
        help="don't auto-shred on exit (caller is responsible)")
    s_dec.add_argument("--hold", type=int, default=0,
        help="keep process alive for N seconds so the tmp dir persists")
    s_dec.set_defaults(func=cmd_decrypt)

    s_st = sp.add_parser("status", help="inspect blob / plaintext / helper state")
    s_st.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
