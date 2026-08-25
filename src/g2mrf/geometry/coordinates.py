from __future__ import annotations

import numpy as np


def normalize_by_height(xy: np.ndarray, height: np.ndarray | float) -> np.ndarray:
    xy = np.asarray(xy, float)
    h = np.asarray(height, float)
    if np.any(h <= 0):
        raise ValueError("height must be positive")
    if h.ndim == 0:
        return xy / h
    shape = (h.shape[0],) + (1,) * (xy.ndim - 1)
    return xy / h.reshape(shape)


def cartesian_to_polar(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, float)
    if xy.shape[-1] != 2:
        raise ValueError("last dimension must be x,y")
    x, y = xy[..., 0], xy[..., 1]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return r, theta
