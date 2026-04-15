"""
Merge singing and Korean speech patches into a combined dataset for training.

Reads from:
  - patches/       (singing data)
  - patches_korean/ (Korean speech data)

Writes to:
  - patches_combined/patches.npy
  - patches_combined/labels.npy
  - patches_combined/manifest.json
"""

import json
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFIER_DIR = os.path.join(PROJECT_ROOT, ".omc", "classifier")


def main():
    singing_dir = os.path.join(CLASSIFIER_DIR, "patches")
    korean_dir = os.path.join(CLASSIFIER_DIR, "patches_korean")
    combined_dir = os.path.join(CLASSIFIER_DIR, "patches_combined")
    os.makedirs(combined_dir, exist_ok=True)

    all_patches = []
    all_labels = []
    all_manifest = []

    for name, patch_dir in [("singing", singing_dir), ("korean", korean_dir)]:
        patches_path = os.path.join(patch_dir, "patches.npy")
        labels_path = os.path.join(patch_dir, "labels.npy")
        manifest_path = os.path.join(patch_dir, "manifest.json")

        if not os.path.exists(patches_path):
            print(f"  SKIP {name}: {patches_path} not found")
            continue

        patches = np.load(patches_path)
        labels = np.load(labels_path)
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Add domain tag to singing manifest entries if missing
        for entry in manifest:
            if "domain" not in entry:
                entry["domain"] = "singing"

        all_patches.append(patches)
        all_labels.append(labels)
        all_manifest.extend(manifest)

        n_tp = int(labels.sum())
        n_fp = len(labels) - n_tp
        print(f"  {name}: {len(patches)} patches ({n_tp} TP, {n_fp} FP/neg), shape={patches.shape}")

    if not all_patches:
        print("ERROR: no patches found")
        return

    combined_patches = np.concatenate(all_patches, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    np.save(os.path.join(combined_dir, "patches.npy"), combined_patches)
    np.save(os.path.join(combined_dir, "labels.npy"), combined_labels)
    with open(os.path.join(combined_dir, "manifest.json"), "w") as f:
        json.dump(all_manifest, f, indent=2)

    n_tp = int(combined_labels.sum())
    n_fp = len(combined_labels) - n_tp
    print(f"\nCombined: {len(combined_patches)} patches ({n_tp} TP, {n_fp} FP/neg)")
    print(f"Shape: {combined_patches.shape}")
    print(f"Saved to {combined_dir}")


if __name__ == "__main__":
    main()
