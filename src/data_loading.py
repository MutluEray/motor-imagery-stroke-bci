"""Loading the original .mat session files.

Each file (e.g. P1_pre_training.mat) contains, per the original README:
  fs   - sampling rate in Hz
  y    - EEG data, samples x channels
  trig - trigger channel: +1 = imagine left hand, -1 = imagine right hand,
         0 = rest/inter-trial interval
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io


def load_session(mat_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (y, trig, fs) exactly as stored in the original .mat file.

    y: (n_samples, n_channels)
    trig: (n_samples,) — +1 left, -1 right, 0 rest
    fs: sampling rate in Hz
    """
    mat = scipy.io.loadmat(str(mat_path))
    y = np.asarray(mat["y"], dtype=np.float64)
    trig = np.asarray(mat["trig"], dtype=np.float64).ravel()
    fs = float(np.asarray(mat["fs"]).ravel()[0])
    return y, trig, fs


def session_file_path(data_dir: Path, subject: str, phase: str, split: str) -> Path:
    """Reconstructs the original naming convention: P1_pre_training.mat,
    P1_pre_test.mat, etc. split is 'training' or 'test'."""
    return data_dir / f"{subject}_{phase}_{split}.mat"
