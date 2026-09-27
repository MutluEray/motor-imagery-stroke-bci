"""Item 1: the main showcase bar chart — best of our models vs. both
published benchmarks, per session. Reads the CSV already saved by
07_final_model.py, no data reload needed.

    python3 scripts/10_results_bar_chart.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

OUR_MODEL_COLS = ["Ensemble", "fbcsp_lda", "csp_lda", "riemann_ts_lr", "riemann_mdm"]


def main():
    df = pd.read_csv(config.RESULTS_DIR / "final_comparison.csv")
    df["Session"] = df["Subject"] + "_" + df["Phase"]
    df["Ours_best"] = df[OUR_MODEL_COLS].max(axis=1)

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, df["Ours_best"], width, label="Ours (best model)", color="#0f766e")
    ax.bar(x, df["CSP+LDA_paper"], width, label="Paper CSP+LDA", color="#94a3b8")
    ax.bar(x + width, df["TVLDA_paper"], width, label="Paper TVLDA", color="#1e293b")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Session"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    out_path = config.FIGURES_DIR / "results_comparison.png"
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"Mean: ours={df['Ours_best'].mean():.1f}%  paper CSP+LDA={df['CSP+LDA_paper'].mean():.1f}%  "
          f"paper TVLDA={df['TVLDA_paper'].mean():.1f}%")


if __name__ == "__main__":
    main()
