"""
ML classifier configuration — agent-editable.

The autoresearch agent can tune these parameters to improve combined_full.
ml_eval.py reads from this file at eval time.
"""

# --- Patch extraction (must match generate_patches.py) ---
PATCH_HALF_S = 2.0        # seconds of audio on each side of detection
N_MELS = 128              # mel spectrogram bins
HOP_LENGTH = 512          # STFT hop length
N_FFT = 2048              # STFT window size
TARGET_FRAMES = 200       # time frames per patch

# --- Classifier pipeline ---
N_ESTIMATORS = 300        # GradientBoosting trees
MAX_DEPTH = 3             # max tree depth (shallower for better generalization on small data)
LEARNING_RATE = 0.1       # boosting learning rate
SUBSAMPLE = 0.5           # fraction of samples per tree (stochastic regularization)
MAX_FEATURES = "sqrt"     # features per split (stochastic feature regularization)
PCA_COMPONENTS = 50       # PCA dimensionality reduction (capped by data)
RANDOM_STATE = 42

# --- Evaluation ---
OOF_THRESHOLD = 0.45      # OOF probability threshold for keeping detections
N_FOLDS = 5               # GroupKFold CV splits
DSP_FP_BOUND = 15         # max DSP false positives on clean files (0 = combined)

# --- Data ---
USE_PREGENERATED_PATCHES = True   # load patches_combined/ as additional training data
