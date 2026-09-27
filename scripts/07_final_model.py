"""Final combined run: wide window (3.5-8.0s) fed into all four models,
plus their ensemble. This is the two wins from the last round (window,
ensemble) stacked together — the likely candidate for the final
showcase numbers.

    python scripts/07_final_model.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from src.data_loading import load_session, session_file_path
from src.preprocessing import common_average_reference, bandpass_filter, extract_trials
from src.modeling import build_csp_lda, build_riemann_tangent_space, build_riemann_mdm, build_fbcsp_lda
from src.evaluation import compute_metrics

TMIN, TMAX = 3.5, 8.0

PUBLISHED_BENCHMARKS = {
    ("P1", "pre"):  {"CSP+LDA": 79.7, "PCA+TVLDA": 100.0},
    ("P1", "post"): {"CSP+LDA": 68.4, "PCA+TVLDA": 72.4},
    ("P2", "pre"):  {"CSP+LDA": 77.1, "PCA+TVLDA": 92.9},
    ("P2", "post"): {"CSP+LDA": 93.9, "PCA+TVLDA": 97.0},
    ("P3", "pre"):  {"CSP+LDA": 96.1, "PCA+TVLDA": 97.4},
    ("P3", "post"): {"CSP+LDA": 74.4, "PCA+TVLDA": 93.6},
}


def load_wide(subject, phase, split, broadband=False):
    path = session_file_path(config.DATA_DIR, subject, phase, split)
    y, trig, fs = load_session(path)
    y = common_average_reference(y)
    low, high = (config.BROADBAND_LOW, config.BROADBAND_HIGH) if broadband else (None, None)
    y = bandpass_filter(y, fs, low=low, high=high)
    right, left = extract_trials(y, trig, fs, tmin=TMIN, tmax=TMAX)
    X = np.concatenate([right, left], axis=0)
    y_labels = np.concatenate([np.zeros(len(right)), np.ones(len(left))])
    return X, y_labels, fs


def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


MODEL_BUILDERS = {
    "csp_lda": (build_csp_lda, False),
    "riemann_ts_lr": (build_riemann_tangent_space, False),
    "riemann_mdm": (build_riemann_mdm, False),
    "fbcsp_lda": (build_fbcsp_lda, True),
}


def main():
    rows = []
    for subject in config.SUBJECTS:
        for phase in config.PHASES:
            probas_test, individual_acc, fs_used = {}, {}, None
            y_test_ref = None

            for name, (builder, broadband) in MODEL_BUILDERS.items():
                X_train, y_train, fs = load_wide(subject, phase, "training", broadband)
                X_test, y_test, _ = load_wide(subject, phase, "test", broadband)
                y_test_ref = y_test
                fs_used = fs

                model = builder(fs)
                model.fit(X_train, y_train)
                probas_test[name] = get_proba(model, X_test)
                pred = model.predict(X_test)
                individual_acc[name] = compute_metrics(y_test, pred)["Accuracy"] * 100

            avg_proba = np.mean(list(probas_test.values()), axis=0)
            ensemble_pred = (avg_proba >= 0.5).astype(float)
            ensemble_acc = compute_metrics(y_test_ref, ensemble_pred)["Accuracy"] * 100

            bench = PUBLISHED_BENCHMARKS[(subject, phase)]
            row = {"Subject": subject, "Phase": phase, "Ensemble": ensemble_acc,
                   **individual_acc, "CSP+LDA_paper": bench["CSP+LDA"], "TVLDA_paper": bench["PCA+TVLDA"]}
            rows.append(row)
            indiv_str = "  ".join(f"{k}={v:.1f}%" for k, v in individual_acc.items())
            print(f"{subject}_{phase}: ensemble={ensemble_acc:.1f}%   ({indiv_str})   "
                  f"|  paper CSP+LDA={bench['CSP+LDA']:.1f}%  paper TVLDA={bench['PCA+TVLDA']:.1f}%")

    df = pd.DataFrame(rows)
    print(f"\nMean ensemble (wide window): {df['Ensemble'].mean():.1f}%")
    for name in MODEL_BUILDERS:
        print(f"Mean {name} (wide window): {df[name].mean():.1f}%")
    print(f"Mean paper CSP+LDA: {df['CSP+LDA_paper'].mean():.1f}%")
    print(f"Mean paper TVLDA: {df['TVLDA_paper'].mean():.1f}%")

    out_path = config.RESULTS_DIR / "final_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
