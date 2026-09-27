"""Two classification pipelines:

1. CSP+LDA — replicates the published Sebastian-Romagosa et al. method
   (4 CSP components, LDA), for direct comparison against that paper's
   accuracies and against the original MATLAB attempt.
2. Riemannian geometry (covariance -> tangent space -> logistic regression,
   and covariance -> MDM) — the modern standard for small-trial-count
   MI-BCI, proposed as the upgrade path over CSP.

Both take trials shaped (n_trials, n_channels, n_samples) and labels in
{0, 1} (0 = right, 1 = left, matching MainScript.m's convention).
"""
from mne.decoding import CSP
import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
from src.preprocessing import bandpass_filter_trials


def build_csp_lda(fs: float = None) -> Pipeline:
    """The baseline: matches the published methodology this repo is meant
    to replicate (4 CSP components + LDA).

    reg="ledoit_wolf": common average referencing (applied in
    preprocessing.py, matching the original MATLAB pipeline) makes the
    channel covariance matrix rank-deficient by construction. Without
    shrinkage regularization here, CSP's generalized eigenvalue solve can
    fail outright ("leading minor is not positive definite") rather than
    just performing poorly — this isn't a tuning nicety, it's needed for
    the fit to run at all on CAR'd data.
    """
    return Pipeline([
        ("csp", CSP(n_components=config.CSP_N_COMPONENTS, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def build_riemann_tangent_space(fs: float = None) -> Pipeline:
    """Covariance -> tangent space projection -> logistic regression.
    Generally the strongest, most sample-efficient option for small-trial
    MI-BCI datasets like this one."""
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("lr", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
    ])


def build_riemann_mdm(fs: float = None) -> Pipeline:
    """Covariance -> Minimum Distance to Mean. No downstream classifier to
    tune at all — classifies by geodesic distance to each class's mean
    covariance. Simplest possible Riemannian baseline."""
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("mdm", MDM(metric="riemann")),
    ])


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Filter Bank CSP (Ang et al., 2008): fits a separate CSP per sub-band
    on broadband trials, concatenates the resulting log-variance features
    across all bands.

    This is the properly-wired-up version of what the original
    filterbank.m / new_main.m attempted — that code built a 9-band
    filterbank but then sliced to `resultRT(:,:,1)` (band 1 = 4-8Hz theta
    only) before CSP ever ran, so 8 of the 9 bands were computed and
    immediately discarded. Here every band actually contributes features.

    Input X: (n_trials, n_channels, n_samples) BROADBAND trials (i.e. from
    preprocess_session_broadband, not the narrowband preprocess_session).
    """

    def __init__(self, bands=None, fs=None, n_components=None, reg="ledoit_wolf"):
        self.bands = bands if bands is not None else config.FBCSP_BANDS
        self.fs = fs
        self.n_components = n_components if n_components is not None else config.FBCSP_N_COMPONENTS
        self.reg = reg

    def fit(self, X, y):
        if self.fs is None:
            raise ValueError("FilterBankCSP requires fs (sampling rate) to be set.")
        self.csps_ = []
        for low, high in self.bands:
            Xb = bandpass_filter_trials(X, self.fs, low, high)
            csp = CSP(n_components=self.n_components, reg=self.reg, log=True, norm_trace=False)
            csp.fit(Xb, y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        feats = []
        for (low, high), csp in zip(self.bands, self.csps_):
            Xb = bandpass_filter_trials(X, self.fs, low, high)
            feats.append(csp.transform(Xb))
        return np.concatenate(feats, axis=1)


def build_fbcsp_lda(fs: float) -> Pipeline:
    """FBCSP -> mutual-info feature selection -> LDA. fs must be passed
    explicitly (unlike the other pipelines) since band-splitting needs the
    real sampling rate, and this repo doesn't assume a fixed one.

    Feature selection (SelectKBest w/ mutual_info_classif) is the modern
    stand-in for the original FBCSP paper's MIBIF step — 9 bands x 4
    components = 36 raw features from only 80 training trials is a lot of
    dimensionality to hand an LDA unfiltered.
    """
    return Pipeline([
        ("fbcsp", FilterBankCSP(fs=fs)),
        ("select", SelectKBest(score_func=mutual_info_classif, k=config.FBCSP_N_SELECT)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


MODELS = {
    "csp_lda": build_csp_lda,
    "riemann_ts_lr": build_riemann_tangent_space,
    "riemann_mdm": build_riemann_mdm,
    "fbcsp_lda": build_fbcsp_lda,
}

# Which trial preprocessing each model needs: 'narrow' = single 8-30Hz band
# (preprocess_session), 'broadband' = wide band for FBCSP to subdivide
# itself (preprocess_session_broadband). See preprocessing.py.
MODEL_BAND_MODE = {
    "csp_lda": "narrow",
    "riemann_ts_lr": "narrow",
    "riemann_mdm": "narrow",
    "fbcsp_lda": "broadband",
}
