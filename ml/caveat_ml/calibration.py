"""Probability calibration + reliability reporting.

We calibrate raw logistic-regression probabilities with isotonic regression, then
report Brier score and a reliability curve so confidence numbers mean what they
say. (The conformal layer in conformal.py is fitted on a *separate* slice so its
coverage guarantee stays valid.)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Fit P(resistant) -> calibrated P(resistant)."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(probs, dtype=float), np.asarray(labels, dtype=float))
    return iso


def apply_isotonic(iso: IsotonicRegression, probs: np.ndarray) -> np.ndarray:
    return np.clip(iso.transform(np.asarray(probs, dtype=float)), 0.0, 1.0)


def brier(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    if len(np.unique(labels)) < 2:
        # brier_score_loss needs both classes present; fall back to MSE.
        return float(np.mean((probs - labels) ** 2))
    return float(brier_score_loss(labels, probs))


def reliability(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Dict[str, List[float]]:
    """Reliability curve points for a calibration plot."""
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if len(np.unique(labels)) < 2:
        return {"prob_pred": [], "prob_true": [], "note": ["single-class calibration set"]}
    try:
        prob_true, prob_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="quantile")
    except Exception:  # pragma: no cover - degenerate bins
        return {"prob_pred": [], "prob_true": []}
    return {"prob_pred": [float(x) for x in prob_pred], "prob_true": [float(x) for x in prob_true]}
