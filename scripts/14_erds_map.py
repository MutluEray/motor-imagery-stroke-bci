"""ERDS maps (event-related spectral perturbation) — the field-standard way
to show ERD/ERS, per Pfurtscheller & Lopes da Silva (1999) and Graimann et
al. (2002), implemented via MNE's built-in TFR machinery (same approach as
MNE's own "Compute and visualize ERDS maps" tutorial).

Unlike the earlier single-band (mu+beta) line plot, this shows the full
2-35Hz spectrum evolving over time at once — you see which frequencies
desync/sync and when, not just one preset band's average. Convention:
red = ERD (desync), blue = ERS (sync) — note this is the OPPOSITE color
mapping from the earlier line plot script.

NOTE: this uses mne.time_frequency.tfr_multitaper and AverageTFR.plot(),
whose exact keyword arguments (vmin/vmax vs. vlim) have changed slightly
across MNE versions. If you hit a TypeError on the .plot() call, try
swapping vmin=-1, vmax=1 for vlim=(-1, 1) or vice versa, and send me the
traceback.

    python3 scripts/14_erds_map.py P1 post
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import tfr_multitaper

import config
from src.data_loading import load_session, session_file_path
from src.preprocessing import bandpass_filter, common_average_reference, extract_trials

FMIN, FMAX = 2, 35
TMIN, TMAX = 0.0, 8.0
BASELINE = (0.0, 2.0)  # pre-cue rest period
PLOT_CHANNELS = ["C3", "C4"]


def build_tfr(trials, info, freqs, n_cycles):
    n_trials = trials.shape[0]
    events = np.column_stack([np.arange(n_trials), np.zeros(n_trials, dtype=int), np.ones(n_trials, dtype=int)])
    epochs = mne.EpochsArray(trials, info, events=events, tmin=TMIN, verbose=False)
    tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                          return_itc=False, average=True, decim=2, verbose=False)
    tfr.apply_baseline(BASELINE, mode="percent")
    return tfr


def main(subject: str, phase: str):
    path = session_file_path(config.DATA_DIR, subject, phase, "training")
    y, trig, fs = load_session(path)
    y = common_average_reference(y)
    y = bandpass_filter(y, fs, low=1.0, high=40.0)  # wide enough to cover the 2-35Hz map range
    right, left = extract_trials(y, trig, fs, tmin=TMIN, tmax=TMAX)

    info = mne.create_info(ch_names=config.CHANNELS, sfreq=fs, ch_types="eeg")
    info.set_montage("standard_1020")

    freqs = np.arange(FMIN, FMAX + 1, 1)
    n_cycles = freqs  # standard choice: n_cycles scales with frequency

    tfr_right = build_tfr(right, info, freqs, n_cycles)
    tfr_left = build_tfr(left, info, freqs, n_cycles)

    for ch in PLOT_CHANNELS:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, (label, tfr) in zip(axes, [("right-hand imagery", tfr_right), ("left-hand imagery", tfr_left)]):
            tfr.plot(picks=[ch], baseline=None, mode=None, axes=ax, colorbar=(ax is axes[-1]),
                     cmap="RdBu_r", vlim=(-1, 1), show=False)
            ax.set_title(f"{ch} \u2014 {label}", fontsize=10)
            ax.set_xlabel("Time (s)")
        fig.tight_layout()

        out_path = config.FIGURES_DIR / f"erds_map_{ch}_{subject}_{phase}.png"
        fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
        print(f"Saved {out_path}")

    print("\nLook for: red (ERD) concentrated in the 8-30Hz mu+beta band after t=2s, deeper/earlier "
          "at C3 for right-hand imagery and at C4 for left-hand imagery. Any blue (ERS) patches "
          "outside that band are less relevant to the classification story.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/14_erds_map.py <subject> <phase>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
