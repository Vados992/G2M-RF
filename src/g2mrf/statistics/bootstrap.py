from __future__ import annotations

import numpy as np
from .metrics import incremental_r2


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return tuple(np.quantile(v, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist())


def bootstrap_incremental_r2(
    y: np.ndarray,
    pred_full: np.ndarray,
    pred_base: np.ndarray,
    B: int = 1000,
    seed: int = 1601001,
) -> np.ndarray:
    y = np.asarray(y, float)
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
    y = np.asarray(y, float)
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
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if not s.size:
        return 1.0
    return float((1.0 + np.sum(s <= null)) / (s.size + 1.0))
