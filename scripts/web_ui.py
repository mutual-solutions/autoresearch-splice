"""Web UI for the splice detector.

Run:
    PYTHONPATH=$PWD uv run python scripts/web_ui.py

Env vars:
    PORT   Override default port 8000.
    HOST   Override default 127.0.0.1.

Routes:
    GET  /            Upload form.
    POST /upload      Decode audio, run detect_splices, render result.
    GET  /audio/<id>  Serve uploaded audio for the <audio> player.

Caveats:
    * Audio held in-memory only (per-process dict, TTL 1h).
    * splice.detector is reloaded per request to track mid-run mutations
      from the autoresearch loop. First request after a detector change
      may take longer due to module re-init.
    * No auth, no HTTPS - designed for localhost research use only.
"""
from __future__ import annotations

import base64
import importlib
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

# splice/detector.py contains a lazy `from features import ...` that only
# resolves when `splice/` is the script directory (sys.path[0] = splice/).
# That's true when the autoresearch wrapper runs `python splice/evaluate.py`
# but NOT when this Flask app reloads the detector module from scripts/.
# Add splice/ to sys.path BEFORE first import so the lazy import resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "splice"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from flask import Flask, abort, render_template, request, send_file  # noqa: E402
from scipy.signal import stft  # noqa: E402

import librosa  # noqa: E402

import splice.detector as splice_detector  # noqa: E402

# Templates live at the repo root, not under scripts/templates/. Set explicitly
# so Flask doesn't look for scripts/templates/ relative to this file.
app = Flask(__name__, template_folder=str(_REPO_ROOT / "templates"))

# In-memory upload store: upload_id -> (audio_bytes, original_filename, mime_type, upload_ts)
_UPLOADS: dict[str, tuple[bytes, str, str, float]] = {}
_TTL_SECONDS = 3600

_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".opus": "audio/ogg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}

_LABEL_COLOR = {
    "cross_voice": "red",
    "same_voice_edit": "gold",
    "unknown": "gray",
}

_META_PATH = Path(__file__).resolve().parent.parent / "splice" / "classifier" / "fp_classifier.meta.json"


def _mime_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return _MIME_BY_EXT.get(ext, "application/octet-stream")


def _sweep_uploads() -> None:
    """Drop entries older than _TTL_SECONDS from the upload store."""
    now = time.time()
    stale = [k for k, v in _UPLOADS.items() if now - v[3] > _TTL_SECONDS]
    for k in stale:
        del _UPLOADS[k]


def _decode_audio(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes to mono float32 at 44.1k. librosa first, soundfile fallback."""
    target_sr = 44100
    try:
        audio, sr = librosa.load(io.BytesIO(raw), sr=target_sr, mono=True)
        return audio.astype(np.float32, copy=False), int(sr)
    except Exception:
        audio, sr = sf.read(io.BytesIO(raw), always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio, int(sr)


def _load_meta() -> dict:
    """Read classifier metadata fresh per request."""
    try:
        raw = _META_PATH.read_text()
        meta_full = json.loads(raw)
        return {
            "version": meta_full.get("version"),
            "classes_": meta_full.get("classes_"),
            "n_features": len(meta_full.get("feature_names", [])),
            "oof_metrics": meta_full.get("oof_metrics"),
        }
    except Exception:
        return {"error": "no metadata"}


def _render_spectrogram(
    audio: np.ndarray,
    sr: int,
    boundaries: list[dict],
    title: str,
) -> str:
    """Render two-panel waveform + spectrogram with boundary lines. Returns data URI."""
    n_fft = 2048
    hop = 512
    duration_s = len(audio) / sr if sr > 0 else 0.0

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(4, 1, hspace=0.3)
    ax_wave = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[1:4, 0], sharex=ax_wave)

    t_wave = np.arange(len(audio)) / sr if sr > 0 else np.arange(len(audio))
    ax_wave.plot(t_wave, audio, linewidth=0.5, color="black")
    ax_wave.set_ylabel("amp")
    ax_wave.set_xlim(0, duration_s)
    ax_wave.tick_params(labelbottom=False)

    f, t, Z = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Z) + 1e-10
    db = 20.0 * np.log10(mag / np.max(mag))
    extent = (t[0], t[-1], f[0] / 1000.0, f[-1] / 1000.0)
    ax_spec.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=-80,
        vmax=-10,
    )
    ax_spec.set_ylabel("kHz")
    ax_spec.set_xlabel("time (s)")
    ax_spec.set_ylim(0, sr / 2 / 1000.0)

    for b in boundaries:
        color = b["color"]
        for ax in (ax_wave, ax_spec):
            ax.axvline(b["time_s"], color=color, linewidth=1.2, alpha=0.85)

    fig.suptitle(title, fontsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    _sweep_uploads()

    file_storage = request.files.get("audio")
    if file_storage is None or not file_storage.filename:
        abort(400, "no audio file in upload")

    original_filename = file_storage.filename
    raw = file_storage.read()
    if not raw:
        abort(400, "empty audio upload")

    mime_type = _mime_for(original_filename)
    upload_id = uuid.uuid4().hex[:16]
    _UPLOADS[upload_id] = (raw, original_filename, mime_type, time.time())

    error: str | None = None
    boundaries: list[dict] = []
    audio = np.zeros(0, dtype=np.float32)
    sr = 44100
    spectrogram_data_uri = ""

    try:
        audio, sr = _decode_audio(raw)
        importlib.reload(splice_detector)
        raw_boundaries = splice_detector.detect_splices(audio, sr)
        for time_s, label in raw_boundaries:
            color = _LABEL_COLOR.get(label, "gray")
            boundaries.append({"time_s": float(time_s), "label": str(label), "color": color})
        boundaries.sort(key=lambda b: b["time_s"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    duration_s = float(len(audio) / sr) if sr > 0 and len(audio) > 0 else 0.0
    n_boundaries = len(boundaries)
    title = f"{original_filename} - {duration_s:.2f}s, sr={sr}, {n_boundaries} boundaries"

    if len(audio) > 0:
        try:
            spectrogram_data_uri = _render_spectrogram(audio, sr, boundaries, title)
        except Exception as exc:
            if error is None:
                error = f"spectrogram render failed: {type(exc).__name__}: {exc}"

    class_counts = {"cross_voice": 0, "same_voice_edit": 0, "unknown": 0}
    for b in boundaries:
        class_counts[b["label"]] = class_counts.get(b["label"], 0) + 1

    boundaries_json = json.dumps(
        [{"time_s": b["time_s"], "label": b["label"]} for b in boundaries],
        indent=3,
    )

    meta_dict = _load_meta()

    return render_template(
        "result.html",
        original_filename=original_filename,
        upload_id=upload_id,
        audio_url=f"/audio/{upload_id}",
        audio_mime_type=mime_type,
        duration_s=duration_s,
        sample_rate=sr,
        boundaries=boundaries,
        n_boundaries=n_boundaries,
        class_counts=class_counts,
        spectrogram_data_uri=spectrogram_data_uri,
        meta_dict=meta_dict,
        boundaries_json=boundaries_json,
        error=error,
    )


@app.route("/audio/<upload_id>", methods=["GET"])
def serve_audio(upload_id: str):
    entry = _UPLOADS.get(upload_id)
    if entry is None:
        abort(404, "audio expired or unknown")
    raw, original_filename, mime_type, _ = entry
    return send_file(
        io.BytesIO(raw),
        mimetype=mime_type,
        as_attachment=False,
        download_name=original_filename,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=False)
