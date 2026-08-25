from __future__ import annotations

import numpy as np

from .metrics import incremental_r2


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return tuple(np.quantile(v, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist())


def _validate_bootstrap_inputs(B: int, *arrays: np.ndarray) -> list[np.ndarray]:
    if not isinstance(B, (int, np.integer)) or B <= 0:
        raise ValueError("B must be a positive integer")
    converted = [np.asarray(a, float) for a in arrays]
    if not converted or converted[0].shape[0] == 0:
        raise ValueError("bootstrap inputs must contain participants")
    shape = converted[0].shape
    if any(a.shape != shape for a in converted[1:]):
        raise ValueError("bootstrap arrays must have identical shapes")
    return converted


def bootstrap_incremental_r2(
    y: np.ndarray,
    pred_full: np.ndarray,
    pred_base: np.ndarray,
    B: int = 1000,
    seed: int = 1601001,
) -> np.ndarray:
    y, pred_full, pred_base = _validate_bootstrap_inputs(B, y, pred_full, pred_base)
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    out = np.empty(B, float)
    for b in range(B):
        idx = rng.integers(0, n, n)
        out[b] = incremental_r2(y[idx], pred_full[idx], pred_base[idx])
    return out


def bootstrap_eta(
    y: np.ndarray,
    pred_cov: np.ndarray,
    pred_direct: np.ndarray,
    pred_mgc: np.ndarray,
    B: int = 1000,
    seed: int = 1601001,
) -> np.ndarray:
    y, pred_cov, pred_direct, pred_mgc = _validate_bootstrap_inputs(
        B,
        y,
        pred_cov,
        pred_direct,
        pred_mgc,
    )
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        dd = incremental_r2(y[idx], pred_direct[idx], pred_cov[idx])
        dm = incremental_r2(y[idx], pred_mgc[idx], pred_cov[idx])
        if dd > 0:
            vals.append(dm / dd)
    return np.asarray(vals, float)


def bootstrap_one_sided_p(samples: np.ndarray, null: float = 0.0) -> float:
    """Practical pre-data bootstrap evidence score; confirmatory deployments may substitute a locked null-bootstrap."""
    if not np.isfinite(null):
        raise ValueError("null must be finite")
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if not s.size:
        return 1.0
    return float((1.0 + np.sum(s <= null)) / (s.size + 1.0))
