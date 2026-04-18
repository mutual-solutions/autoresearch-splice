#!/usr/bin/env python3
"""Encrypt / decrypt the eval dataset with a session-keychain-stored key.

Isolates `data/eval/` from autoresearch's claude subprocess. The encrypted
blob is `data/eval.tar.gz.enc`; the AES-256-GCM key is stored in the
macOS login keychain as a generic password, accessible to any same-user
process that knows the SERVICE name. No biometric gate — the wrapper
decrypts once per iteration (or at loop start) and shreds on exit.

Why no biometric: the threat is *accidental* reads by automation that
doesn't know the key exists, not *targeted* bypass by a human with
shell access. Biometric-per-iteration would be 12+ prompts/hour during
autoresearch. A biometric-gated helper lives at scripts/test_unlock for
the test split, which has a harder threat model.

Commands:
  setup       generate a fresh 32-byte key, store in Keychain via
              `security add-generic-password -A`, tar+gzip+encrypt
              data/eval/ → data/eval.tar.gz.enc, shred plaintext.
  encrypt     re-encrypt with existing key (post regenerate_datasets).
  decrypt     decrypt to a fresh /tmp/.ar-eval-* dir (0700), print the
              root path on stdout. Caller owns cleanup (--keep) OR
              wrapper passes --hold N to keep dir alive for N seconds.
  status      inspect blob / plaintext / keychain entry.
  erase-key   wipe the key from keychain (for rotation or decommissioning).

Format matches test_crypto: version(1) || nonce(12) || ciphertext||tag.
Key lives at keychain item SERVICE='autoresearch-eval-aes-key'.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import io
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
PLAINTEXT_DIR = REPO / "data" / "eval"
ENCRYPTED_BLOB = REPO / "data" / "eval.tar.gz.enc"

SERVICE = "autoresearch-eval-aes-key"
ACCOUNT = "default"

NONCE_LEN = 12
KEY_LEN = 32
BLOB_VERSION = b"\x01"


# --- helpers ---------------------------------------------------------------


def _die(msg: str, code: int = 1) -> None:
    print(f"eval_crypto: {msg}", file=sys.stderr)
    sys.exit(code)


def _store_key(key: bytes) -> None:
    """Push into login keychain via `security add-generic-password -A`.
    `-A` = any app can read without prompt; `-U` = update if exists.
    """
    # Wipe prior item so attribute updates land.
    subprocess.run(
        ["security", "delete-generic-password",
         "-a", ACCOUNT, "-s", SERVICE],
        capture_output=True,
    )
    b64 = base64.b64encode(key).decode()
    r = subprocess.run(
        ["security", "add-generic-password",
         "-a", ACCOUNT,
         "-s", SERVICE,
         "-w", b64,
         "-A",  # any app, no password prompt
         "-U",  # update if exists
        ],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        _die(f"`security add-generic-password` failed: {r.stderr.strip()}")


def _fetch_key() -> bytes:
    r = subprocess.run(
        ["security", "find-generic-password",
         "-a", ACCOUNT, "-s", SERVICE, "-w"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        _die(f"keychain fetch failed (is the key registered via `setup`?): "
             f"{r.stderr.strip()}")
    try:
        return base64.b64decode(r.stdout.strip(), validate=True)
    except Exception as e:
        _die(f"keychain returned invalid base64: {e}")


def _shred_dir(path: Path) -> None:
    """Best-effort overwrite then unlink. APFS CoW caveats apply."""
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


def _tar_gz_bytes(src_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as tar:
        for entry in sorted(src_dir.rglob("*")):
            if entry.is_symlink():
                continue
            rel = entry.relative_to(src_dir.parent)
            tar.add(str(entry), arcname=str(rel), recursive=False)
    return buf.getvalue()


def _extract(data: bytes, dest: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(path=dest)


def _encrypt(key: bytes, plaintext: bytes) -> bytes:
    aes = AESGCM(key)
    nonce = secrets.token_bytes(NONCE_LEN)
    return BLOB_VERSION + nonce + aes.encrypt(nonce, plaintext, None)


def _decrypt(key: bytes, blob: bytes) -> bytes:
    if not blob.startswith(BLOB_VERSION):
        _die(f"blob version mismatch: got {blob[:1]!r}")
    nonce = blob[1 : 1 + NONCE_LEN]
    ct = blob[1 + NONCE_LEN :]
    try:
        return AESGCM(key).decrypt(nonce, ct, None)
    except Exception as e:
        _die(f"decrypt failed (wrong key or tampered): {e}")


# --- commands --------------------------------------------------------------


def cmd_setup(args) -> None:
    if not PLAINTEXT_DIR.exists():
        _die(f"{PLAINTEXT_DIR} not found")
    print("Generating 32-byte AES-256 key…")
    key = secrets.token_bytes(KEY_LEN)
    print(f"Storing in Keychain ({SERVICE}/{ACCOUNT}) — no prompt expected…")
    _store_key(key)

    print("Tar+gzip data/eval/…")
    plaintext = _tar_gz_bytes(PLAINTEXT_DIR)
    print(f"  plaintext: {len(plaintext):,} bytes")

    print("Encrypt with AES-256-GCM…")
    blob = _encrypt(key, plaintext)
    tmp = ENCRYPTED_BLOB.with_suffix(".enc.tmp")
    tmp.write_bytes(blob)
    tmp.replace(ENCRYPTED_BLOB)
    print(f"  wrote {ENCRYPTED_BLOB} ({len(blob):,} bytes)")

    if args.keep_plaintext:
        print("  (--keep-plaintext: data/eval/ left in place)")
    else:
        print("Shredding data/eval/…")
        _shred_dir(PLAINTEXT_DIR)
        print(f"  removed {PLAINTEXT_DIR}")

    del key; del blob


def cmd_encrypt(args) -> None:
    if not PLAINTEXT_DIR.exists():
        _die(f"{PLAINTEXT_DIR} not found — nothing to encrypt")
    key = _fetch_key()
    print("Tar+gzip data/eval/…")
    plaintext = _tar_gz_bytes(PLAINTEXT_DIR)
    print(f"  plaintext: {len(plaintext):,} bytes")
    blob = _encrypt(key, plaintext)
    tmp = ENCRYPTED_BLOB.with_suffix(".enc.tmp")
    tmp.write_bytes(blob)
    tmp.replace(ENCRYPTED_BLOB)
    print(f"  wrote {ENCRYPTED_BLOB}")
    if not args.keep_plaintext:
        _shred_dir(PLAINTEXT_DIR)
    del key; del blob


def cmd_decrypt(args) -> None:
    if not ENCRYPTED_BLOB.exists():
        _die(f"{ENCRYPTED_BLOB} not found")
    key = _fetch_key()
    blob = ENCRYPTED_BLOB.read_bytes()
    plaintext = _decrypt(key, blob)

    root = Path(tempfile.mkdtemp(prefix=".ar-eval-"))
    os.chmod(root, 0o700)

    if not args.keep:
        def _cleanup():
            _shred_dir(root)
        atexit.register(_cleanup)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: sys.exit(130))

    _extract(plaintext, root)
    eval_root = root / "eval"
    if not eval_root.exists():
        candidates = [c for c in root.iterdir() if c.is_dir()]
        if len(candidates) == 1:
            eval_root = candidates[0]
        else:
            _die(f"unexpected tarball layout under {root}")
    print(str(eval_root))
    del key; del plaintext; del blob

    if args.keep and args.hold > 0:
        import time
        time.sleep(args.hold)


def cmd_status(args) -> None:
    print(f"plaintext dir : {PLAINTEXT_DIR} "
          f"({'PRESENT' if PLAINTEXT_DIR.exists() else 'absent'})")
    print(f"encrypted blob: {ENCRYPTED_BLOB} "
          f"({'PRESENT (' + str(ENCRYPTED_BLOB.stat().st_size) + ' bytes)' if ENCRYPTED_BLOB.exists() else 'absent'})")
    r = subprocess.run(
        ["security", "find-generic-password", "-a", ACCOUNT, "-s", SERVICE],
        capture_output=True, text=True,
    )
    key_present = r.returncode == 0
    print(f"keychain entry: {SERVICE} ({'REGISTERED' if key_present else 'missing'})")


def cmd_erase_key(args) -> None:
    r = subprocess.run(
        ["security", "delete-generic-password",
         "-a", ACCOUNT, "-s", SERVICE],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print(f"erased keychain item {SERVICE}/{ACCOUNT}")
    else:
        print(f"no keychain item to erase (or delete failed): {r.stderr.strip()}")


# --- entry -----------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("setup")
    s.add_argument("--keep-plaintext", action="store_true")
    s.set_defaults(func=cmd_setup)

    s = sp.add_parser("encrypt")
    s.add_argument("--keep-plaintext", action="store_true")
    s.set_defaults(func=cmd_encrypt)

    s = sp.add_parser("decrypt")
    s.add_argument("--keep", action="store_true")
    s.add_argument("--hold", type=int, default=0)
    s.set_defaults(func=cmd_decrypt)

    s = sp.add_parser("status")
    s.set_defaults(func=cmd_status)

    s = sp.add_parser("erase-key")
    s.set_defaults(func=cmd_erase_key)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
