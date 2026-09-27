"""Diagnostics for one subject x phase — use this to sanity-check any
result that looks surprisingly good or bad (e.g. P1_post's CSP+LDA beating
even the published TVLDA number).

Prints: trigger counts and class balance in each file, per-class accuracy
(not just overall accuracy — a lopsided confusion matrix can hide behind a
good-looking overall number), and the confusion matrix itself.

    python scripts/02_diagnose_session.py P1 post
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import confusion_matrix

import config
from src.data_loading import load_session, session_file_path
from src.evaluation import compute_metrics
from src.modeling import build_csp_lda
from src.preprocessing import preprocess_session


def describe_triggers(trig: np.ndarray, label: str):
    n_right = int(np.sum((trig[1:] == -1) & (trig[:-1] == 0)))
    n_left = int(np.sum((trig[1:] == 1) & (trig[:-1] == 0)))
    print(f"  {label}: {n_right} right-triggers, {n_left} left-triggers "
          f"(imbalance: {abs(n_right - n_left)})")


def main(subject: str, phase: str):
    train_path = session_file_path(config.DATA_DIR, subject, phase, "training")
    test_path = session_file_path(config.DATA_DIR, subject, phase, "test")

    y_tr, trig_tr, fs_tr = load_session(train_path)
    y_te, trig_te, fs_te = load_session(test_path)

    print(f"=== {subject}_{phase} ===")
    print(f"fs: train={fs_tr}, test={fs_te}")
    print(f"y shape: train={y_tr.shape}, test={y_te.shape}")
    print("\nTrigger counts (raw, before any windowing/rejection):")
    describe_triggers(trig_tr, "train")
    describe_triggers(trig_te, "test")

    right_tr, left_tr = preprocess_session(y_tr, trig_tr, fs_tr)
    right_te, left_te = preprocess_session(y_te, trig_te, fs_te)
    print(f"\nAfter preprocessing + windowing (trials with a full window only):")
    print(f"  train: {len(right_tr)} right, {len(left_tr)} left")
    print(f"  test:  {len(right_te)} right, {len(left_te)} left")

    X_train = np.concatenate([right_tr, left_tr], axis=0)
    y_train = np.concatenate([np.zeros(len(right_tr)), np.ones(len(left_tr))])
    X_test = np.concatenate([right_te, left_te], axis=0)
    y_test = np.concatenate([np.zeros(len(right_te)), np.ones(len(left_te))])

    model = build_csp_lda(fs_tr)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = compute_metrics(y_test, y_pred, y_score)
    print(f"\nOverall accuracy: {metrics['Accuracy']:.3f}")
    print(f"Right-class (0) recall (Spec): {metrics['Spec']:.3f}   "
          f"Left-class (1) recall (Sens): {metrics['Sens']:.3f}")
    if abs(metrics["Sens"] - metrics["Spec"]) > 0.3:
        print("  ^ WARNING: large gap between per-class recall — overall accuracy may be "
              "hiding a model that's mostly just predicting one class well and the other poorly.")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("\nConfusion matrix (rows=true, cols=pred), 0=right, 1=left:")
    print(cm)

    print(f"\nTrain class balance: {int((y_train==0).sum())} right / {int((y_train==1).sum())} left")
    print(f"Test class balance:  {int((y_test==0).sum())} right / {int((y_test==1).sum())} left")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/02_diagnose_session.py <subject> <phase>")
        print("Example: python scripts/02_diagnose_session.py P1 post")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
