"""ERD/ERS band-power time course at C3 vs C4 — shows mu+beta power
desynchronizing over the trial, separately for right- and left-hand
imagery. Real motor imagery should show contralateral desynchronization:
C3 (left hemisphere) desyncs more for RIGHT-hand imagery, C4 desyncs more
for LEFT-hand imagery.

Trigger timing (from the original paradigm): t=0 is the attention sound
(trial start, before the patient knows which hand to imagine); t=2s is the
cue (arrow + spoken word revealing left/right); t=3.5-8s is the feedback
phase (avatar + FES respond to the live classifier). The plot starts at
t=2s (the cue) rather than t=0, since nothing hand-specific can be present
in the signal before the patient knows which hand to imagine.

    python3 scripts/09_erd_ers.py P1 post
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

import config
from src.data_loading import load_session, session_file_path
from src.preprocessing import common_average_reference, bandpass_filter, extract_trials

TMIN, TMAX = 0.0, 8.0       # extract the full trial (need the 0-2s baseline for normalization)
PLOT_TMIN = 2.0              # but only plot from the cue onward — see docstring
BASELINE_TMIN, BASELINE_TMAX = 0.0, 2.0  # pre-cue rest period, used only to normalize
WINDOW_SEC, STEP_SEC = 0.5, 0.1


def sliding_power(trials, fs):
    win = int(WINDOW_SEC * fs)
    step = int(STEP_SEC * fs)
    n_samples = trials.shape[-1]
    n_windows = (n_samples - win) // step + 1
    power = np.empty((trials.shape[0], n_windows, trials.shape[1]))
    for w in range(n_windows):
        seg = trials[:, :, w * step: w * step + win]
        power[:, w, :] = (seg ** 2).mean(axis=-1)
    times = TMIN + (np.arange(n_windows) * step + win / 2) / fs
    return power, times


def main(subject: str, phase: str):
    path = session_file_path(config.DATA_DIR, subject, phase, "training")
    y, trig, fs = load_session(path)
    y = common_average_reference(y)
    y = bandpass_filter(y, fs)  # 8-30Hz mu+beta
    right, left = extract_trials(y, trig, fs, tmin=TMIN, tmax=TMAX)

    c3_idx = config.CHANNELS.index("C3")
    c4_idx = config.CHANNELS.index("C4")

    power_right, times = sliding_power(right, fs)
    power_left, _ = sliding_power(left, fs)
    baseline_mask = (times >= BASELINE_TMIN) & (times <= BASELINE_TMAX)
    plot_mask = times >= PLOT_TMIN

    def erd_curve(power, ch_idx):
        baseline = power[:, baseline_mask, ch_idx].mean()
        curve = power[:, :, ch_idx].mean(axis=0)
        return (curve - baseline) / baseline * 100

    c3_right = erd_curve(power_right, c3_idx)[plot_mask]
    c3_left = erd_curve(power_left, c3_idx)[plot_mask]
    c4_right = erd_curve(power_right, c4_idx)[plot_mask]
    c4_left = erd_curve(power_left, c4_idx)[plot_mask]
    times_plot = times[plot_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    ax1.plot(times_plot, c3_right, label="right-hand imagery", color="#0f766e")
    ax1.plot(times_plot, c3_left, label="left-hand imagery", color="#dc2626")
    ax1.set_title("C3 (left hemisphere)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("ERD/ERS (%, vs. pre-cue baseline)")
    ax1.axhline(0, color="#94a3b8", linewidth=0.8)
    ax1.axvline(3.5, color="#94a3b8", linestyle=":", linewidth=0.8, label="feedback phase begins")
    ax1.legend(fontsize=8)

    ax2.plot(times_plot, c4_right, label="right-hand imagery", color="#0f766e")
    ax2.plot(times_plot, c4_left, label="left-hand imagery", color="#dc2626")
    ax2.set_title("C4 (right hemisphere)")
    ax2.set_xlabel("Time (s)")
    ax2.axhline(0, color="#94a3b8", linewidth=0.8)
    ax2.axvline(3.5, color="#94a3b8", linestyle=":", linewidth=0.8)

    fig.tight_layout()

    out_path = config.FIGURES_DIR / f"erd_ers_{subject}_{phase}.png"
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/09_erd_ers.py <subject> <phase>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
