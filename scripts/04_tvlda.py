"""Idea 1: TVLDA — the actual published method that beat CSP+LDA by
+10.6 points on average in the original hackathon. Uses the WIDE trial
window (3.5-8.0s, the whole feedback phase) since TVLDA's entire point is
summing evidence across time, unlike CSP+LDA's single fixed window.

    python scripts/04_tvlda.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from src.data_loading import load_session, session_file_path
from src.preprocessing import common_average_reference, bandpass_filter, extract_trials
from src.tvlda import TVLDA
from src.evaluation import compute_metrics

TMIN, TMAX = 3.5, 8.0  # full feedback phase, not the narrow original window

PUBLISHED_BENCHMARKS = {
    ("P1", "pre"):  {"CSP+LDA": 79.7, "PCA+TVLDA": 100.0},
    ("P1", "post"): {"CSP+LDA": 68.4, "PCA+TVLDA": 72.4},
    ("P2", "pre"):  {"CSP+LDA": 77.1, "PCA+TVLDA": 92.9},
    ("P2", "post"): {"CSP+LDA": 93.9, "PCA+TVLDA": 97.0},
    ("P3", "pre"):  {"CSP+LDA": 96.1, "PCA+TVLDA": 97.4},
    ("P3", "post"): {"CSP+LDA": 74.4, "PCA+TVLDA": 93.6},
}


def load_wide(subject, phase, split):
    path = session_file_path(config.DATA_DIR, subject, phase, split)
    y, trig, fs = load_session(path)
    y = common_average_reference(y)
    y = bandpass_filter(y, fs)
    right, left = extract_trials(y, trig, fs, tmin=TMIN, tmax=TMAX)
    X = np.concatenate([right, left], axis=0)
    y_labels = np.concatenate([np.zeros(len(right)), np.ones(len(left))])
    return X, y_labels, fs


def main():
    rows = []
    for subject in config.SUBJECTS:
        for phase in config.PHASES:
            X_train, y_train, fs = load_wide(subject, phase, "training")
            X_test, y_test, _ = load_wide(subject, phase, "test")

            model = TVLDA(fs=fs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = compute_metrics(y_test, y_pred)["Accuracy"] * 100

            bench = PUBLISHED_BENCHMARKS[(subject, phase)]
            rows.append({"Subject": subject, "Phase": phase, "TVLDA_ours": acc,
                         "CSP+LDA_paper": bench["CSP+LDA"], "TVLDA_paper": bench["PCA+TVLDA"]})
            print(f"{subject}_{phase}: our TVLDA={acc:.1f}%   "
                  f"paper CSP+LDA={bench['CSP+LDA']:.1f}%   paper TVLDA={bench['PCA+TVLDA']:.1f}%")

    df = pd.DataFrame(rows)
    print(f"\nMean: our TVLDA={df['TVLDA_ours'].mean():.1f}%   "
          f"paper CSP+LDA={df['CSP+LDA_paper'].mean():.1f}%   paper TVLDA={df['TVLDA_paper'].mean():.1f}%")

    out_path = config.RESULTS_DIR / "tvlda_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
