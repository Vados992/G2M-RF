from __future__ import annotations

import numpy as np


def _require_finite_matrix(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def linear_kernel(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    X = _require_finite_matrix("X", X)
    Y = X if Y is None else _require_finite_matrix("Y", Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("feature dimensions differ")
    return (X @ Y.T) / X.shape[1]


def trace_normalize(K: np.ndarray) -> np.ndarray:
    K = _require_finite_matrix("K", K)
    if K.shape[0] != K.shape[1]:
        raise ValueError("kernel must be square")
    s = float(np.trace(K) / K.shape[0])
    if not np.isfinite(s) or s <= 0:
        raise ValueError("kernel trace must be positive")
    return K / s


def composite_kernel(kernels: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if len(kernels) != len(weights) or not kernels:
        raise ValueError("kernels and weights mismatch")
    w = np.asarray(weights, float)
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be finite, nonnegative and sum to one")
    normalized = [trace_normalize(K) for K in kernels]
    shape = normalized[0].shape
    if any(K.shape != shape for K in normalized[1:]):
        raise ValueError("all kernels must have identical shape")
    out = np.zeros_like(normalized[0])
    for wi, Ki in zip(w, normalized):
        out += wi * Ki
    return out


def projection_matrix(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C = np.asarray(C, float)
    if C.ndim != 2 or C.shape[0] == 0:
        raise ValueError("C must be a non-empty N x Q matrix")
    if not np.all(np.isfinite(C)):
        raise ValueError("C contains NaN/Inf")
    P = C @ np.linalg.pinv(C.T @ C) @ C.T
    M = np.eye(C.shape[0]) - P
    return P, M


def residualized_kernel(K: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Corrected penalized-FWL kernel: K_r = M_C K M_C."""
    K = _require_finite_matrix("K", K)
    if K.shape[0] != K.shape[1]:
        raise ValueError("kernel must be square")
    _, M = projection_matrix(C)
    if M.shape[0] != K.shape[0]:
        raise ValueError("kernel and covariate sample counts differ")
    return M @ K @ M
