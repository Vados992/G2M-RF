from __future__ import annotations

import numpy as np


def _paired_arrays(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    if y.shape != pred.shape:
        raise ValueError("shape mismatch")
    if y.size == 0:
        raise ValueError("metric inputs must be non-empty")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
        raise ValueError("metric inputs contain NaN/Inf")
    return y, pred


def aggregate_r2(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = _paired_arrays(y, pred)
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
    y, pred = _paired_arrays(y, pred)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def participant_mse(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    y, pred = _paired_arrays(y, pred)
    if y.ndim < 2:
        raise ValueError("participant_mse expects participant x feature arrays")
    return np.mean((y - pred) ** 2, axis=tuple(range(1, y.ndim)))
