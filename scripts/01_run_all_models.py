"""Runs CSP+LDA and both Riemannian variants across every subject x phase,
saves a combined results table, and prints a comparison against the
literature benchmarks already established in overview.pdf (CSP+LDA and
PCA+TVLDA accuracies from the original hackathon).

    python scripts/01_run_all_models.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from src.evaluation import run_all
from src.modeling import MODELS, MODEL_BAND_MODE

# From overview.pdf — the original hackathon's own benchmark numbers.
# Index: (subject, phase) -> accuracy in %
PUBLISHED_BENCHMARKS = {
    ("P1", "pre"):  {"CSP+LDA": 79.7, "PCA+TVLDA": 100.0},
    ("P1", "post"): {"CSP+LDA": 68.4, "PCA+TVLDA": 72.4},
    ("P2", "pre"):  {"CSP+LDA": 77.1, "PCA+TVLDA": 92.9},
    ("P2", "post"): {"CSP+LDA": 93.9, "PCA+TVLDA": 97.0},
    ("P3", "pre"):  {"CSP+LDA": 96.1, "PCA+TVLDA": 97.4},
    ("P3", "post"): {"CSP+LDA": 74.4, "PCA+TVLDA": 93.6},
}


def main():
    all_results = []
    for model_name, builder in MODELS.items():
        band_mode = MODEL_BAND_MODE[model_name]
        df = run_all(config.DATA_DIR, model_name, builder, band_mode=band_mode)
        all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)
    out_path = config.RESULTS_DIR / "model_comparison.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")

    # Pivot for a clean side-by-side view, with published benchmarks alongside
    pivot = results.pivot_table(index=["Subject", "Phase"], columns="Model", values="Accuracy") * 100
    for (subject, phase), row in pivot.iterrows():
        bench = PUBLISHED_BENCHMARKS.get((subject, phase), {})
        row_str = "  ".join(f"{m}={v:.1f}%" for m, v in row.items())
        bench_str = "  ".join(f"{k}(paper)={v:.1f}%" for k, v in bench.items())
        print(f"{subject}_{phase}: {row_str}   |   {bench_str}")


if __name__ == "__main__":
    main()
