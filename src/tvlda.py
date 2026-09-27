"""Time-Variant LDA (Gruenwald et al., 2019), adapted for this dataset.

Core idea from the paper: instead of picking one fixed time point to
classify a trial (what CSP+LDA effectively does with its single window),
train an LDA at every time point across the trial and sum their scores
(Eq. 29 in the paper: z = sum_n p[n]). This uses the whole trial's
temporal evolution instead of throwing away everything outside one slice.

Adaptations from the original (documented honestly, not a literal port):
  - The paper used log-power in the 50-300Hz high-gamma band from
    invasive ECoG, with a 50ms sliding window. We don't have high-gamma
    or ECoG; this uses log-variance (a bandpower proxy) in the 8-30Hz
    mu+beta band from scalp EEG, with a longer window (configurable,
    default 250ms) since 50ms is too short a window to estimate variance
    reliably at 256Hz scalp EEG SNR.
  - The paper's TVLDA covariance model is diagonal in projected p-space by
    construction (Eq. 24). Here each per-timepoint LDA uses shrinkage
    (sklearn's `shrinkage="auto"`) instead, since with only 80 trials vs
    16 channels the raw per-timepoint covariance is severely
    underdetermined — shrinkage serves the same "keep it invertible and
    stable" purpose as their regularization.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class TVLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, fs=None, window_sec=0.25, step_sec=0.25, shrinkage="auto"):
        self.fs = fs
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.shrinkage = shrinkage

    def _bandpower_features(self, X):
        """X: (n_trials, n_channels, n_samples) -> (n_trials, n_windows, n_channels)
        log-variance in sliding windows across the trial."""
        win = max(2, int(self.window_sec * self.fs))
        step = max(1, int(self.step_sec * self.fs))
        n_samples = X.shape[-1]
        n_windows = max(1, (n_samples - win) // step + 1)

        feats = np.empty((X.shape[0], n_windows, X.shape[1]))
        for w in range(n_windows):
            seg = X[:, :, w * step: w * step + win]
            feats[:, w, :] = np.log(seg.var(axis=-1) + 1e-12)
        return feats

    def fit(self, X, y):
        feats = self._bandpower_features(np.asarray(X))
        self.n_windows_ = feats.shape[1]
        self.ldas_ = []
        for w in range(self.n_windows_):
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.shrinkage)
            lda.fit(feats[:, w, :], y)
            self.ldas_.append(lda)
        self.classes_ = self.ldas_[0].classes_
        return self

    def decision_function(self, X):
        feats = self._bandpower_features(np.asarray(X))
        scores = np.zeros(feats.shape[0])
        for w, lda in enumerate(self.ldas_):
            scores += lda.decision_function(feats[:, w, :])
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        scores = self.decision_function(X) / max(self.n_windows_, 1)
        p1 = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - p1, p1])
