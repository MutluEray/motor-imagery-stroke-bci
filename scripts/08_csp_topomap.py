"""CSP spatial pattern topomaps — shows what the model actually learned as
scalp maps. For real motor imagery, left-hand imagery should show
right-hemisphere (C4-centered) activation and vice versa (contralateral
control) — this is the sanity/credibility check as much as it's a figure.

Uses the winning config from the final run: wide window (3.5-8.0s),
narrowband 8-30Hz, CSP+LDA.

    python scripts/08_csp_topomap.py P1 post
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP

import config
from src.data_loading import load_session, session_file_path
from src.preprocessing import common_average_reference, bandpass_filter, extract_trials

TMIN, TMAX = 3.5, 8.0


def main(subject: str, phase: str):
    path = session_file_path(config.DATA_DIR, subject, phase, "training")
    y, trig, fs = load_session(path)
    y = common_average_reference(y)
    y = bandpass_filter(y, fs)  # 8-30Hz
    right, left = extract_trials(y, trig, fs, tmin=TMIN, tmax=TMAX)

    import numpy as np
    X = np.concatenate([right, left], axis=0)
    y_labels = np.concatenate([np.zeros(len(right)), np.ones(len(left))])

    info = mne.create_info(ch_names=config.CHANNELS, sfreq=fs, ch_types="eeg")
    info.set_montage("standard_1020")

    csp = CSP(n_components=config.CSP_N_COMPONENTS, reg="ledoit_wolf", log=True, norm_trace=False)
    csp.fit(X, y_labels)

    fig = csp.plot_patterns(info, ch_type="eeg", components=list(range(config.CSP_N_COMPONENTS)),
                             show=False)
    fig.suptitle(f"{subject}_{phase} — CSP spatial patterns (right=0 vs left=1)", fontsize=11)

    out_path = config.FIGURES_DIR / f"csp_topomap_{subject}_{phase}.png"
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Saved {out_path}")
    print("Look for: patterns concentrated over C3/C4 area (contralateral to the imagined hand) "
          "rather than diffuse/frontal — that's the sign CSP found real motor-imagery signal, "
          "not noise or artifact.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/08_csp_topomap.py <subject> <phase>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
