"""Central configuration: paths and constants.

Point DATA_DIR at your local folder of .mat files (P1_pre_training.mat,
P1_pre_test.mat, ... — same naming convention as the original MATLAB
MainScript.m). This repo never expects the data to be committed; data/ is
git-ignored.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

SUBJECTS = ["P1", "P2", "P3"]
PHASES = ["pre", "post"]

# 16-channel montage, from the original README (international 10/20 system)
CHANNELS = [
    "FC5", "FC1", "FCz", "FC2", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP1", "CP2", "CP6",
]

# Motor-imagery band. The original preprocessing.m used a broad 1-40 Hz
# bandpass for all features; this is narrowed to the mu+beta range that
# ERD/ERS actually lives in, matching the published Sebastian-Romagosa
# et al. methodology (8-30 Hz) that this repo's CSP+LDA baseline is meant
# to replicate.
BANDPASS_LOW = 8.0
BANDPASS_HIGH = 30.0

# Trial window relative to trigger onset, in seconds. Matches the original
# preprocessing.m (i + 3.5*fs : i + 5*fs - 1), i.e. the early-to-mid
# feedback phase of the trial (see Paradigm.png: feedback runs ~3.5s-8s).
TRIAL_TMIN = 3.5
TRIAL_TMAX = 5.0

# CSP baseline: 4 spatial filters, matching the published methodology
# (Mueller-Gerking et al. recommendation used in Sebastian-Romagosa et al.)
CSP_N_COMPONENTS = 4

# Filter Bank CSP: same 9 sub-bands as the original filterbank.m (4-8, 8-12,
# ... 36-40 Hz) — but this time every band is actually used (the original
# new_main.m sliced to band 1 only and discarded the rest, see README).
FBCSP_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]
FBCSP_N_COMPONENTS = 4  # per band
FBCSP_N_SELECT = 8      # features kept after mutual-info selection (of 9 bands x 4 = 36 total)

# Broadband range trials are extracted at before per-band filtering for
# FBCSP (must cover every sub-band above with margin for the filter's
# transition band).
BROADBAND_LOW = 3.0
BROADBAND_HIGH = 42.0

RANDOM_STATE = 0
