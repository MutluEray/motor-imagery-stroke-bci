"""Idea 3: light hyperparameter tuning for CSP+LDA — CSP n_components and
LDA shrinkage, both currently fixed defaults never tuned at all.

Uses 5-fold CV WITHIN the 80 training trials only (never touches the test
set) to pick a config per subject/phase, then evaluates once on test.

    python scripts/05_hyperparam_search.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import config
from src.evaluation import load_subject_phase, compute_metrics

GRID = [
    {"n_components": nc, "shrinkage": sh}
    for nc in [2, 4, 6, 8]
    for sh in [None, "auto"]
]


def build(n_components, shrinkage):
    return Pipeline([
        ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr" if shrinkage else "svd", shrinkage=shrinkage)),
    ])


def main():
    rows = []
    for subject in config.SUBJECTS:
        for phase in config.PHASES:
            X_train, y_train, X_test, y_test, fs = load_subject_phase(config.DATA_DIR, subject, phase, "narrow")

            best_score, best_params = -1, None
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
            for params in GRID:
                model = build(**params)
                try:
                    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
                    mean_score = scores.mean()
                except Exception as e:
                    mean_score = -1  # some (n_components, shrinkage) combos may be numerically unstable
                if mean_score > best_score:
                    best_score, best_params = mean_score, params

            model = build(**best_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_acc = compute_metrics(y_test, y_pred)["Accuracy"] * 100

            rows.append({"Subject": subject, "Phase": phase, **best_params,
                         "CV_Accuracy": best_score * 100, "Test_Accuracy": test_acc})
            print(f"{subject}_{phase}: best={best_params}  cv_acc={best_score*100:.1f}%  "
                  f"test_acc={test_acc:.1f}%")

    df = pd.DataFrame(rows)
    print(f"\nMean test accuracy (tuned): {df['Test_Accuracy'].mean():.1f}%")
    print("(compare against the default csp_lda mean of 86.2% from the earlier run)")

    out_path = config.RESULTS_DIR / "hyperparam_search.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
