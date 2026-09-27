"""Shows the actual CSP feature the classifier uses — not a raw anatomical
channel, but the specific spatial combination of all 16 channels that CSP
found to best separate the two classes. This should show much cleaner
separation than C3/C4 alone, because that's literally what CSP is
optimized to produce (raw channels can't show it because the
discriminative signal is spread across a spatial combination, not sitting
in any one channel).

CSP is fit once on the winning classification window (3.5-8.0s), then its
fixed spatial filters are applied to short sliding sub-windows across the
whole trial (using the model's own .transform()) to get a time-resolved
view of the same log-variance feature the LDA actually classifies on.

    python3 scripts/15_csp_feature_view.py P1 post
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from mne.decoding import CSP

import config
from src.data_loading import load_session, session_file_path
from src.preprocessing import bandpass_filter, common_average_reference, extract_trials

FIT_TMIN, FIT_TMAX = 3.5, 8.0     # the winning classification window — CSP is FIT here
VIEW_TMIN, VIEW_TMAX = 2.0, 8.0   # wider window for the time-resolved view, from the cue onward
WINDOW_SEC, STEP_SEC = 1.0, 0.25  # longer than the earlier plots — variance needs enough samples


def sliding_csp_features(trials, csp, fs):
    """trials: (n_trials, n_channels, n_samples). Returns (n_trials, n_windows,
    n_components) log-variance features from the model's own .transform()."""
    win = int(WINDOW_SEC * fs)
    step = int(STEP_SEC * fs)
    n_samples = trials.shape[-1]
    n_windows = (n_samples - win) // step + 1
    n_trials = trials.shape[0]

    feats = np.empty((n_trials, n_windows, config.CSP_N_COMPONENTS))
    for w in range(n_windows):
        seg = trials[:, :, w * step: w * step + win]
        feats[:, w, :] = csp.transform(seg)
    times = VIEW_TMIN + (np.arange(n_windows) * step + win / 2) / fs
    return feats, times


def main(subject: str, phase: str):
    fit_path = session_file_path(config.DATA_DIR, subject, phase, "training")
    y_fit_raw, trig_fit, fs = load_session(fit_path)
    y_fit_raw = common_average_reference(y_fit_raw)
    y_fit_raw = bandpass_filter(y_fit_raw, fs)  # 8-30Hz

    # Fit CSP on the actual winning window, same as the real classifier
    right_fit, left_fit = extract_trials(y_fit_raw, trig_fit, fs, tmin=FIT_TMIN, tmax=FIT_TMAX)
    X_fit = np.concatenate([right_fit, left_fit], axis=0)
    y_fit = np.concatenate([np.zeros(len(right_fit)), np.ones(len(left_fit))])
    csp = CSP(n_components=config.CSP_N_COMPONENTS, reg="ledoit_wolf", log=True, norm_trace=False)
    csp.fit(X_fit, y_fit)

    # Time-resolved view on the HELD-OUT TEST file — CSP filters were fit on
    # training only, so any separation shown here is genuine generalization,
    # not just what CSP was optimized to show on its own training data.
    view_path = session_file_path(config.DATA_DIR, subject, phase, "test")
    y_view_raw, trig_view, fs_view = load_session(view_path)
    y_view_raw = common_average_reference(y_view_raw)
    y_view_raw = bandpass_filter(y_view_raw, fs_view)
    right_view, left_view = extract_trials(y_view_raw, trig_view, fs_view, tmin=VIEW_TMIN, tmax=VIEW_TMAX)
    feats_right, times = sliding_csp_features(right_view, csp, fs_view)
    feats_left, _ = sliding_csp_features(left_view, csp, fs_view)

    # The two most discriminative components: index 0 (max right-class variance)
    # and the last one (max left-class variance) — standard CSP ordering.
    components = {
        "CSP component 1 (right-optimized)": 0,
        f"CSP component {config.CSP_N_COMPONENTS} (left-optimized)": config.CSP_N_COMPONENTS - 1,
    }

    fig, axes = plt.subplots(1, len(components), figsize=(11, 4))
    for ax, (label, idx) in zip(axes, components.items()):
        curve_right = feats_right[:, :, idx].mean(axis=0)
        curve_left = feats_left[:, :, idx].mean(axis=0)

        ax.plot(times, curve_right, color="#0f766e", label="right-hand imagery")
        ax.plot(times, curve_left, color="#dc2626", label="left-hand imagery")
        ax.axvline(3.5, color="#94a3b8", linestyle=":", linewidth=0.8, label="feedback phase begins")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("CSP log-variance (the model's actual feature)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = config.FIGURES_DIR / f"csp_feature_view_{subject}_{phase}.png"
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Saved {out_path}")
    print("This version is computed on the HELD-OUT TEST file (CSP filters were fit on "
          "training only) — separation shown here is genuine generalization, not just "
          "what CSP was optimized to show on its own training data.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/15_csp_feature_view.py <subject> <phase>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
