from __future__ import annotations

import numpy as np


def linear_kernel(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, float)
    Y = X if Y is None else np.asarray(Y, float)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("feature dimensions differ")
    return (X @ Y.T) / X.shape[1]


def trace_normalize(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, float)
    s = np.trace(K) / K.shape[0]
    if s <= 0:
        raise ValueError("kernel trace must be positive")
    return K / s


def composite_kernel(kernels: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if len(kernels) != len(weights) or not kernels:
        raise ValueError("kernels and weights mismatch")
    w = np.asarray(weights, float)
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    out = np.zeros_like(np.asarray(kernels[0], float))
    for wi, Ki in zip(w, kernels):
        out += wi * trace_normalize(Ki)
    return out


def projection_matrix(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C = np.asarray(C, float)
    P = C @ np.linalg.pinv(C.T @ C) @ C.T
    M = np.eye(C.shape[0]) - P
    return P, M


def residualized_kernel(K: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Corrected penalized-FWL kernel: K_r = M_C K M_C."""
    _, M = projection_matrix(C)
    return M @ np.asarray(K, float) @ M
