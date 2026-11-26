from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.size == 0:
        raise ValueError("Порожній набір для оцінювання")
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def expected_calibration_error(correct, confidence, n_bins: int = 10) -> float:
    correct, confidence = np.asarray(correct, float), np.asarray(confidence, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return float(ece)


def threshold_report(y_true, y_pred, confidence, threshold: float) -> dict:
    y_true, y_pred, confidence = np.asarray(y_true), np.asarray(y_pred), np.asarray(confidence, float)
    auto = confidence >= threshold
    return {
        "threshold": threshold,
        "coverage": float(auto.mean()),
        "accuracy_auto": float(accuracy_score(y_true[auto], y_pred[auto])) if auto.any() else float("nan"),
        "macro_f1_auto": macro_f1(y_true[auto], y_pred[auto]) if auto.any() else float("nan"),
    }


def classification_summary(y_true, y_pred, confidence=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if confidence is not None:
        out["ece"] = expected_calibration_error(np.asarray(y_true) == np.asarray(y_pred), confidence)
    return out
