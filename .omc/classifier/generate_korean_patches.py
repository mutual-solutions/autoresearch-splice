"""
Generate spectrogram patches from Korean speech splice data.

Adapts generate_patches.py for 16kHz Korean audio:
  - Adjusts n_fft and hop_length proportionally to sample rate
    so mel patches have comparable time/frequency resolution
  - Saves to patches_korean/ directory
  - Prefixes file names in manifest with "kr_" to avoid collisions
"""

import json
import os
import sys
import numpy as np
import soundfile as sf
import librosa

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

_snap = os.environ.get("DETECTOR_SNAPSHOT")
if _snap:
    import importlib.util
    spec = importlib.util.spec_from_file_location("detector", _snap)
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)
    detect_splices = detector.detect_splices
else:
    from detector import detect_splices

PATCH_HALF_S = 2.0      # +-2s around detection
N_MELS = 128
TOLERANCE_S = 1.0
TARGET_FRAMES = 200      # Pad/crop to this width

# Reference sample rate and FFT params (from singing data)
REF_SR = 44100
REF_HOP = 512
REF_NFFT = 2048


def get_fft_params(sr):
    """Scale FFT parameters proportionally to sample rate so time resolution matches."""
    ratio = sr / REF_SR
    hop = max(1, int(REF_HOP * ratio))
    n_fft = max(hop, int(REF_NFFT * ratio))
    # Round n_fft up to next power of 2 for efficiency
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))
    return n_fft, hop


def extract_mel_patch(audio, sr, center_s):
    """Extract a mel spectrogram patch centered at center_s."""
    n_fft, hop_length = get_fft_params(sr)

    center_sample = int(center_s * sr)
    half_samples = int(PATCH_HALF_S * sr)
    start = max(0, center_sample - half_samples)
    end = min(len(audio), center_sample + half_samples)

    segment = audio[start:end]
    if len(segment) < sr // 2:  # Too short
        return None

    S = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_mels=N_MELS,
        n_fft=n_fft, hop_length=hop_length
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize to [0, 1]
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

    # Pad or crop to TARGET_FRAMES
    if S_db.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
    elif S_db.shape[1] > TARGET_FRAMES:
        S_db = S_db[:, :TARGET_FRAMES]

    return S_db  # (128, TARGET_FRAMES)


def is_tp(det_time, gt_times, tolerance=TOLERANCE_S):
    """Check if a detection is a true positive."""
    for gt in gt_times:
        if abs(det_time - gt) <= tolerance:
            return True
    return False


def generate_augmented_patches(audio, sr, center_s, label, n_aug=3):
    """Generate original + augmented patches with shifted centers."""
    patches = []
    labels = []

    patch = extract_mel_patch(audio, sr, center_s)
    if patch is not None:
        patches.append(patch)
        labels.append(label)

    for shift in np.linspace(-0.3, 0.3, n_aug):
        shifted_center = center_s + shift
        if shifted_center < 0 or shifted_center > len(audio) / sr:
            continue
        patch = extract_mel_patch(audio, sr, shifted_center)
        if patch is not None:
            patches.append(patch)
            labels.append(label)

    return patches, labels


def main():
    korean_data_dir = os.path.join(
        os.path.expanduser("~"),
        "Projects", "mutual", "audio-splice-detector", "data", "korean-splice"
    )
    gt_path = os.path.join(korean_data_dir, "ground_truth.json")
    output_dir = os.path.join(PROJECT_ROOT, ".omc", "classifier", "patches_korean")
    os.makedirs(output_dir, exist_ok=True)

    with open(gt_path) as f:
        ground_truth = json.load(f)

    # Check sample rate of first file to confirm 16kHz
    first_entry = next(iter(ground_truth.values()))
    first_path = os.path.join(korean_data_dir, first_entry["path"])
    info = sf.info(first_path)
    print(f"Korean audio sample rate: {info.samplerate} Hz")
    n_fft, hop = get_fft_params(info.samplerate)
    print(f"Using n_fft={n_fft}, hop_length={hop} (scaled from {REF_NFFT}/{REF_HOP} @ {REF_SR}Hz)")

    all_patches = []
    all_labels = []
    manifest = []

    for name, entry in ground_truth.items():
        wav_path = os.path.join(korean_data_dir, entry["path"])
        if not os.path.exists(wav_path):
            print(f"  SKIP {name}: not found")
            continue

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        gt_times = []
        if entry.get("spliced", False):
            st = entry.get("splice_time_sec")
            if isinstance(st, list):
                gt_times = [float(t) for t in st]
            elif st is not None:
                gt_times = [float(st)]

        # Run detector
        try:
            det_times = detect_splices(audio, sr)
            det_times = sorted(float(t) for t in det_times)
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            continue

        # Prefix file names with "kr_" to avoid collision with singing data
        kr_name = f"kr_{name}"

        # Label each detection
        for det_t in det_times:
            label = 1 if is_tp(det_t, gt_times) else 0
            patches, labels = generate_augmented_patches(audio, sr, det_t, label, n_aug=3)
            for p, l in zip(patches, labels):
                all_patches.append(p)
                all_labels.append(l)
                manifest.append({
                    "file": kr_name,
                    "center_s": float(det_t),
                    "label": l,
                    "source": "detection",
                    "domain": "korean_speech"
                })

        # Clean files: random negatives
        if not entry.get("spliced", False):
            duration = len(audio) / sr
            rng = np.random.RandomState(hash(name) % (2**31))
            n_neg = 3
            for _ in range(n_neg):
                t = rng.uniform(PATCH_HALF_S, duration - PATCH_HALF_S)
                patches, labels = generate_augmented_patches(audio, sr, t, 0, n_aug=2)
                for p, l in zip(patches, labels):
                    all_patches.append(p)
                    all_labels.append(l)
                    manifest.append({
                        "file": kr_name,
                        "center_s": float(t),
                        "label": 0,
                        "source": "clean_random",
                        "domain": "korean_speech"
                    })

        # Spliced files: GT patches if missed by detector
        if entry.get("spliced", False):
            for gt_t in gt_times:
                already_covered = any(abs(gt_t - d) < TOLERANCE_S for d in det_times)
                if not already_covered:
                    patches, labels = generate_augmented_patches(audio, sr, gt_t, 1, n_aug=3)
                    for p, l in zip(patches, labels):
                        all_patches.append(p)
                        all_labels.append(l)
                        manifest.append({
                            "file": kr_name,
                            "center_s": float(gt_t),
                            "label": 1,
                            "source": "gt_missed",
                            "domain": "korean_speech"
                        })

            # Random negatives from spliced files
            duration = len(audio) / sr
            rng = np.random.RandomState(hash(name) % (2**31) + 1)
            for _ in range(2):
                t = rng.uniform(PATCH_HALF_S, duration - PATCH_HALF_S)
                if not any(abs(t - gt) < 3.0 for gt in gt_times):
                    patches, labels = generate_augmented_patches(audio, sr, t, 0, n_aug=2)
                    for p, l in zip(patches, labels):
                        all_patches.append(p)
                        all_labels.append(l)
                        manifest.append({
                            "file": kr_name,
                            "center_s": float(t),
                            "label": 0,
                            "source": "spliced_random_neg",
                            "domain": "korean_speech"
                        })

        print(f"  {name}: dets={len(det_times)}, patches so far={len(all_patches)}")

    # Save
    all_patches = np.array(all_patches)
    all_labels = np.array(all_labels)

    np.save(os.path.join(output_dir, "patches.npy"), all_patches)
    np.save(os.path.join(output_dir, "labels.npy"), all_labels)
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    n_tp = int(all_labels.sum())
    n_fp = len(all_labels) - n_tp
    print(f"\nDone: {len(all_patches)} patches, {n_tp} TP, {n_fp} FP/neg")
    print(f"Patch shape: {all_patches.shape}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
