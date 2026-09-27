"""Item 6: animated GIF — scalp ERD%/ERS% topomap evolving across the
trial, right-hand vs left-hand imagery side by side. Requires Pillow
(usually already installed with matplotlib).

    python3 scripts/12_topomap_animation.py P1 post
"""
import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import mne
import numpy as np
from PIL import Image

import config
from src.data_loading import load_session, session_file_path
from src.preprocessing import common_average_reference, bandpass_filter, extract_trials

TMIN, TMAX = 0.0, 8.0
BASELINE_TMIN, BASELINE_TMAX = 0.0, 2.0
WINDOW_SEC, STEP_SEC = 0.5, 0.25  # coarser than the line-plot script -> fewer, smoother GIF frames


def sliding_power_all_channels(trials, fs):
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
    y = bandpass_filter(y, fs)  # 8-30Hz
    right, left = extract_trials(y, trig, fs, tmin=TMIN, tmax=TMAX)

    info = mne.create_info(ch_names=config.CHANNELS, sfreq=fs, ch_types="eeg")
    info.set_montage("standard_1020")

    power_right, times = sliding_power_all_channels(right, fs)
    power_left, _ = sliding_power_all_channels(left, fs)
    baseline_mask = (times >= BASELINE_TMIN) & (times <= BASELINE_TMAX)

    def erd(power):
        baseline = power[:, baseline_mask, :].mean(axis=(0, 1))  # per-channel baseline
        avg = power.mean(axis=0)  # (n_windows, n_channels)
        return (avg - baseline) / baseline * 100

    erd_right = erd(power_right)  # (n_windows, n_channels)
    erd_left = erd(power_left)

    vlim = np.percentile(np.abs(np.concatenate([erd_right, erd_left])), 98)

    frames = []
    for i, t in enumerate(times):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
        mne.viz.plot_topomap(erd_right[i], info, axes=axes[0], show=False,
                              vlim=(-vlim, vlim), cmap="RdBu_r")
        axes[0].set_title("Right-hand imagery", fontsize=10)
        mne.viz.plot_topomap(erd_left[i], info, axes=axes[1], show=False,
                              vlim=(-vlim, vlim), cmap="RdBu_r")
        axes[1].set_title("Left-hand imagery", fontsize=10)
        fig.suptitle(f"t = {t:.1f}s", fontsize=11)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    out_path = config.FIGURES_DIR / f"topomap_animation_{subject}_{phase}.gif"
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"Saved {out_path} ({len(frames)} frames)")
    print("Blue = desync (ERD), red = sync (ERS). Watch for blue spreading toward C4 during "
          "left-hand imagery and toward C3 during right-hand imagery as t increases.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/12_topomap_animation.py <subject> <phase>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
