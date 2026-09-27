"""Idea 2: does widening the trial window help CSP+LDA?

Original window: 3.5-5.0s post-trigger (1.5s, matches preprocessing.m).
This compares it against wider windows that cover more of the feedback
phase (paradigm runs cue->2s, feedback ~3.5s->8s relax).

    python scripts/03_wide_window.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from src.data_loading import load_session, session_file_path
from src.preprocessing import common_average_reference, bandpass_filter, extract_trials
from src.modeling import build_csp_lda
from src.evaluation import compute_metrics

WINDOWS = {
    "narrow_3.5-5.0s (original)": (3.5, 5.0),
    "wide_3.5-8.0s":               (3.5, 8.0),
    "full_2.0-8.0s":               (2.0, 8.0),
}


def load_and_window(subject, phase, split, tmin, tmax):
    path = session_file_path(config.DATA_DIR, subject, phase, split)
    y, trig, fs = load_session(path)
    y = common_average_reference(y)
    y = bandpass_filter(y, fs)
    right, left = extract_trials(y, trig, fs, tmin=tmin, tmax=tmax)
    X = np.concatenate([right, left], axis=0)
    y_labels = np.concatenate([np.zeros(len(right)), np.ones(len(left))])
    return X, y_labels, fs


def main():
    rows = []
    for window_name, (tmin, tmax) in WINDOWS.items():
        for subject in config.SUBJECTS:
            for phase in config.PHASES:
                X_train, y_train, fs = load_and_window(subject, phase, "training", tmin, tmax)
                X_test, y_test, _ = load_and_window(subject, phase, "test", tmin, tmax)

                model = build_csp_lda(fs)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = compute_metrics(y_test, y_pred)["Accuracy"]
                rows.append({"Window": window_name, "Subject": subject, "Phase": phase, "Accuracy": acc})

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["Subject", "Phase"], columns="Window", values="Accuracy") * 100
    print(pivot.round(1))
    print("\nMean by window:")
    print((pivot.mean() ).round(1))

    out_path = config.RESULTS_DIR / "window_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
