"""Item 2: confusion matrix for FBCSP (the strongest single model overall),
wide window, one session.

    python3 scripts/11_confusion_matrix.py P1 post
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import config
from src.data_loading import load_session, session_file_path
from src.preprocessing import common_average_reference, bandpass_filter, extract_trials
from src.modeling import build_fbcsp_lda

TMIN, TMAX = 3.5, 8.0
LABELS = ["Right", "Left"]


def load_broadband(subject, phase, split):
    path = session_file_path(config.DATA_DIR, subject, phase, split)
    y, trig, fs = load_session(path)
    y = common_average_reference(y)
    y = bandpass_filter(y, fs, low=config.BROADBAND_LOW, high=config.BROADBAND_HIGH)
    right, left = extract_trials(y, trig, fs, tmin=TMIN, tmax=TMAX)
    X = np.concatenate([right, left], axis=0)
    y_labels = np.concatenate([np.zeros(len(right)), np.ones(len(left))])
    return X, y_labels, fs


def main(subject: str, phase: str):
    X_train, y_train, fs = load_broadband(subject, phase, "training")
    X_test, y_test, _ = load_broadband(subject, phase, "test")

    model = build_fbcsp_lda(fs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap="BuGn", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABELS)
    ax.set_yticks([0, 1]); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "#0f172a"
            ax.text(j, i, f"{val*100:.0f}%\n(n={cm[i,j]})", ha="center", va="center", color=color, fontsize=11)
    fig.tight_layout()

    out_path = config.FIGURES_DIR / f"confusion_matrix_{subject}_{phase}.png"
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/11_confusion_matrix.py <subject> <phase>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
