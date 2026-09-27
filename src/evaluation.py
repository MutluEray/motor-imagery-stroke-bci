"""Per-subject, per-phase evaluation — mirrors MainScript.m's loop
structure exactly (train on the *_training.mat file, test on the
*_test.mat file for each subject x phase), so results are directly
comparable to both the original MATLAB numbers and the paper's table.

compute_metrics() ports RFsimple.m / SVMlinear.m's metric block
(TP/TN/FP/FN, Sens, Spec, BalAcc, Precision, Recall, F1, AUC) to Python.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from src.data_loading import load_session, session_file_path
from src.preprocessing import preprocess_session, preprocess_session_broadband


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> dict:
    """y_true, y_pred in {0, 1}. y_score: predicted probability of class 1,
    for AUC (optional)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    bal_acc = (sens + spec) / 2
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = sens
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) and not np.isnan(precision) else float("nan"))
    acc = (tp + tn) / len(y_true)

    metrics = {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": acc, "Sens": sens, "Spec": spec, "BalAcc": bal_acc,
        "Precision": precision, "Recall": recall, "F1Score": f1,
    }
    if y_score is not None:
        try:
            metrics["AUC"] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics["AUC"] = float("nan")
    return metrics


def load_subject_phase(data_dir: Path, subject: str, phase: str, band_mode: str = "narrow"):
    """Loads + preprocesses the training and test sessions for one
    subject x phase, returns (X_train, y_train, X_test, y_test, fs) with
    y in {0=right, 1=left}, matching MainScript.m's convention.

    band_mode: 'narrow' (single 8-30Hz band, for csp_lda/riemann) or
    'broadband' (wide band, for fbcsp_lda to subdivide itself).
    """
    train_path = session_file_path(data_dir, subject, phase, "training")
    test_path = session_file_path(data_dir, subject, phase, "test")

    y_tr, trig_tr, fs_tr = load_session(train_path)
    y_te, trig_te, fs_te = load_session(test_path)

    preprocess = preprocess_session if band_mode == "narrow" else preprocess_session_broadband
    right_tr, left_tr = preprocess(y_tr, trig_tr, fs_tr)
    right_te, left_te = preprocess(y_te, trig_te, fs_te)

    X_train = np.concatenate([right_tr, left_tr], axis=0)
    y_train = np.concatenate([np.zeros(len(right_tr)), np.ones(len(left_tr))])

    X_test = np.concatenate([right_te, left_te], axis=0)
    y_test = np.concatenate([np.zeros(len(right_te)), np.ones(len(left_te))])

    return X_train, y_train, X_test, y_test, fs_tr


def run_all(data_dir: Path, model_name: str, model_builder, band_mode: str = "narrow") -> pd.DataFrame:
    """Runs one model across every subject x phase, mirroring
    MainScript.m's ResultsRF / ResultsSVM tables."""
    rows = []
    for subject in config.SUBJECTS:
        for phase in config.PHASES:
            X_train, y_train, X_test, y_test, fs = load_subject_phase(data_dir, subject, phase, band_mode)

            model = model_builder(fs)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = None
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)

            metrics = compute_metrics(y_test, y_pred, y_score)
            metrics.update({"Subject": subject, "Phase": phase, "Model": model_name,
                             "NTrainTrials": len(y_train), "NTestTrials": len(y_test)})
            rows.append(metrics)
            print(f"{model_name:15s} {subject}_{phase}: "
                  f"acc={metrics['Accuracy']:.3f}  n_train={len(y_train)}  n_test={len(y_test)}")

    return pd.DataFrame(rows)
