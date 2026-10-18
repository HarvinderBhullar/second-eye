"""Calibration and operating points, all fit on val.

* Platt scaling: p_cal = sigmoid(a * logit(p) + b), fit by logistic regression on val.
* Youden threshold: argmax over val thresholds of sensitivity + specificity - 1.
* Sensitivity threshold: the highest threshold whose val sensitivity is still >= 0.90
  (any lower threshold also reaches 0.90; the highest keeps the most specificity).
Thresholds are on the calibrated probability.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Platt:
    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = a, b

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Platt":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a * logit(np.asarray(p)) + self.b)))

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b}


def youden_threshold(p: np.ndarray, y: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def sensitivity_threshold(p: np.ndarray, y: np.ndarray, target: float = 0.90) -> float:
    fpr, tpr, thr = roc_curve(y, p)        # thresholds descending, tpr ascending
    return float(thr[np.argmax(tpr >= target)])


def sens_spec(p: np.ndarray, y: np.ndarray, t: float) -> tuple[float, float]:
    pred = p >= t
    return float(pred[y == 1].mean()), float((~pred[y == 0]).mean())


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of p(fracture), equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def fit(p_val: np.ndarray, y_val: np.ndarray) -> dict:
    """-> {"platt": {a, b}, "threshold_youden", "threshold_sens90"} fit on val."""
    platt = Platt().fit(p_val, y_val)
    pc = platt(p_val)
    return {"platt": platt.to_dict(), "threshold_youden": youden_threshold(pc, y_val),
            "threshold_sens90": sensitivity_threshold(pc, y_val, 0.90)}
