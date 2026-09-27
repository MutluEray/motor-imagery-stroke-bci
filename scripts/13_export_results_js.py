"""Prints the final_comparison.csv contents as a ready-to-paste JavaScript
array for the interactive comparison widget (interactive-comparison.html).
Copy the printed block into that file's DATA constant.

    python3 scripts/13_export_results_js.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config

COLS = ["Subject", "Phase", "Ensemble", "fbcsp_lda", "csp_lda", "riemann_ts_lr", "riemann_mdm",
        "CSP+LDA_paper", "TVLDA_paper"]
JS_KEYS = ["subject", "phase", "ensemble", "fbcsp", "csp_lda", "riemann_ts", "riemann_mdm",
           "paper_csp", "paper_tvlda"]


def main():
    df = pd.read_csv(config.RESULTS_DIR / "final_comparison.csv")
    print("const DATA = [")
    for _, row in df.iterrows():
        parts = []
        for col, key in zip(COLS, JS_KEYS):
            val = row[col]
            if key in ("subject", "phase"):
                parts.append(f'{key}: "{val}"')
            else:
                parts.append(f"{key}: {round(float(val), 1)}")
        print("  { " + ", ".join(parts) + " },")
    print("];")


if __name__ == "__main__":
    main()
