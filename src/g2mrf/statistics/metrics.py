from __future__ import annotations

import numpy as np


def aggregate_r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    if y.shape != pred.shape:
        raise ValueError("shape mismatch")
    yy = y.reshape(-1)
    pp = pred.reshape(-1)
    denom = float(np.sum((yy - yy.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((yy - pp) ** 2) / denom)


def incremental_r2(y: np.ndarray, pred_full: np.ndarray, pred_base: np.ndarray) -> float:
    return aggregate_r2(y, pred_full) - aggregate_r2(y, pred_base)


def nrmse(y: np.ndarray, pred: np.ndarray) -> float:
    """Radii are already height-normalized, so RMSE is nRMSE in fractions of height."""
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def participant_mse(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    return np.mean((y - pred) ** 2, axis=1)
