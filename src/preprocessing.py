"""Preprocessing: common average reference, mu+beta bandpass, trial
extraction.

Two deliberate changes from the original preprocessing.m:
  1. Bandpass narrowed to 8-30 Hz (mu+beta), not 1-40 Hz broadband — see
     config.py docstring for why.
  2. This is the ONLY preprocessing step; band-specific feature engineering
     (the original features.m) is superseded entirely by CSP / Riemannian
     covariance methods, which operate directly on these bandpassed trials.
"""
from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt

import config


def bandpass_filter(data: np.ndarray, fs: float, low: float = None, high: float = None,
                     order: int = 4) -> np.ndarray:
    """data: (n_samples, n_channels). Zero-phase Butterworth bandpass,
    applied per channel."""
    low = config.BANDPASS_LOW if low is None else low
    high = config.BANDPASS_HIGH if high is None else high
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, data, axis=0)


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """data: (n_samples, n_channels)."""
    return data - data.mean(axis=1, keepdims=True)


def extract_trials(y: np.ndarray, trig: np.ndarray, fs: float,
                    tmin: float = None, tmax: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Ports preprocessing.m's trial extraction exactly: a trial starts at
    each 0 -> +-1 trigger transition, and the window [tmin, tmax] seconds
    after that transition is extracted.

    Returns (right_trials, left_trials), each (n_trials, n_channels, n_samples).
    """
    tmin = config.TRIAL_TMIN if tmin is None else tmin
    tmax = config.TRIAL_TMAX if tmax is None else tmax
    start_offset = int(tmin * fs)
    end_offset = int(tmax * fs)
    win_len = end_offset - start_offset

    right_trials, left_trials = [], []
    for i in range(1, len(trig)):
        if trig[i] == -1 and trig[i - 1] == 0:  # right
            seg = y[i + start_offset: i + end_offset, :]
            if seg.shape[0] == win_len:
                right_trials.append(seg.T)  # -> (n_channels, n_samples)
        elif trig[i] == 1 and trig[i - 1] == 0:  # left
            seg = y[i + start_offset: i + end_offset, :]
            if seg.shape[0] == win_len:
                left_trials.append(seg.T)

    right_trials = np.stack(right_trials) if right_trials else np.empty((0, y.shape[1], win_len))
    left_trials = np.stack(left_trials) if left_trials else np.empty((0, y.shape[1], win_len))
    return right_trials, left_trials


def bandpass_filter_trials(X: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Same as bandpass_filter but for already-extracted trials, shaped
    (n_trials, n_channels, n_samples) — filters along the time axis (-1).
    Used by FilterBankCSP to carve broadband trials into sub-bands."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, X, axis=-1)


def preprocess_session(y: np.ndarray, trig: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Full pipeline for one session file, for the single-band CSP+LDA and
    Riemannian pipelines: CAR -> 8-30Hz bandpass -> trial extraction.
    Returns (right_trials, left_trials)."""
    y = common_average_reference(y)
    y = bandpass_filter(y, fs)
    return extract_trials(y, trig, fs)


def preprocess_session_broadband(y: np.ndarray, trig: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """For FBCSP: CAR -> wide 3-42Hz bandpass (just wide enough to cover
    every FBCSP sub-band's transition edges) -> trial extraction. The
    per-band filtering into the 9 FBCSP sub-bands happens later, inside
    FilterBankCSP.transform, on these broadband trials — NOT here. This
    mirrors the original filterbank.m concept faithfully, unlike
    new_main.m which sliced to a single band before CSP ever ran."""
    y = common_average_reference(y)
    y = bandpass_filter(y, fs, low=config.BROADBAND_LOW, high=config.BROADBAND_HIGH)
    return extract_trials(y, trig, fs)
