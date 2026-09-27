"""Idea 4: simple ensemble of the four existing models (csp_lda, fbcsp_lda,
riemann_ts_lr, riemann_mdm) — average their predicted probabilities and
threshold at 0.5. Cheapest option, unlikely to close a large gap but worth
a data point.

    python scripts/06_ensemble.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from src.evaluation import load_subject_phase, compute_metrics
from src.modeling import MODELS, MODEL_BAND_MODE


def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))  # sigmoid fallback


def main():
    rows = []
    for subject in config.SUBJECTS:
        for phase in config.PHASES:
            probas_test = {}
            y_test_ref = None
            individual_acc = {}

            for name, builder in MODELS.items():
                band_mode = MODEL_BAND_MODE[name]
                X_train, y_train, X_test, y_test, fs = load_subject_phase(
                    config.DATA_DIR, subject, phase, band_mode)
                y_test_ref = y_test

                model = builder(fs)
                model.fit(X_train, y_train)
                probas_test[name] = get_proba(model, X_test)
                pred = model.predict(X_test)
                individual_acc[name] = compute_metrics(y_test, pred)["Accuracy"] * 100

            avg_proba = np.mean(list(probas_test.values()), axis=0)
            ensemble_pred = (avg_proba >= 0.5).astype(float)
            ensemble_acc = compute_metrics(y_test_ref, ensemble_pred)["Accuracy"] * 100

            row = {"Subject": subject, "Phase": phase, "Ensemble": ensemble_acc, **individual_acc}
            rows.append(row)
            indiv_str = "  ".join(f"{k}={v:.1f}%" for k, v in individual_acc.items())
            print(f"{subject}_{phase}: ensemble={ensemble_acc:.1f}%   ({indiv_str})")

    df = pd.DataFrame(rows)
    print(f"\nMean ensemble accuracy: {df['Ensemble'].mean():.1f}%")
    for name in MODELS:
        print(f"Mean {name}: {df[name].mean():.1f}%")

    out_path = config.RESULTS_DIR / "ensemble_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
