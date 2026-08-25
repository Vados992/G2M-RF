from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from .models import bounds, canonical_phase, physical_envelope, radius, B_PHI, PHI


@dataclass
class FitResult:
    model: str
    params: np.ndarray
    mse: float
    success: bool
    boundary_distance: float


def _mse(model: str, theta: np.ndarray, r: np.ndarray, params: np.ndarray) -> float:
    if not physical_envelope(model, params):
        return math.inf
    pred = radius(model, theta, params)
    if not np.all(np.isfinite(pred)):
        return math.inf
    return float(np.mean((r - pred) ** 2))


def _boundary_distance(model: str, params: np.ndarray) -> float:
    bs = bounds(model)
    d = []
    for x, (lo, hi) in zip(params, bs):
        if model.upper() == "M5" and len(params) == 2 and (lo, hi) == (-1.0, 1.0):
            continue
        d.extend([x - lo, hi - x])
    return float(min(d)) if d else math.inf


def fit_m4_profile(theta: np.ndarray, r: np.ndarray) -> FitResult:
    """Fast exact-profile fit for M4 under radial least squares."""
    theta = np.asarray(theta, float)
    r = np.asarray(r, float)

    def objective_b(b: float) -> float:
        z = np.exp(b * theta)
        a = float(np.dot(r, z) / np.dot(z, z))
        a = float(np.clip(a, 0.01, 0.75))
        return _mse("M4", theta, r, np.array([a, b]))

    res = minimize_scalar(objective_b, bounds=(-2.0, 2.0), method="bounded", options={"xatol": 1e-12})
    b = float(res.x)
    z = np.exp(b * theta)
    a = float(np.clip(np.dot(r, z) / np.dot(z, z), 0.01, 0.75))
    p = np.array([a, b])
    return FitResult("M4", p, _mse("M4", theta, r, p), bool(res.success and physical_envelope("M4", p)), _boundary_distance("M4", p))


def fit_model(model: str, theta: np.ndarray, r: np.ndarray, seed: int = 1601001) -> FitResult:
    model = model.upper()
    theta = np.asarray(theta, float)
    r = np.asarray(r, float)
    if theta.shape != r.shape or theta.ndim != 1:
        raise ValueError("theta and r must be same-length 1-D arrays")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(r)):
        raise ValueError("NaN/Inf blocked before fitting")

    if model == "M0":
        R = float(np.clip(np.mean(r), 0.05, 0.75))
        p = np.array([R])
        return FitResult(model, p, _mse(model, theta, r, p), True, _boundary_distance(model, p))
    if model == "M4":
        return fit_m4_profile(theta, r)
    if model == "M5":
        candidates = []
        for s in (-1.0, 1.0):
            z = np.exp(s * B_PHI * theta)
            a = float(np.clip(np.dot(r, z) / np.dot(z, z), 0.01, 0.75 / PHI))
            p = np.array([a, s])
            candidates.append(FitResult(model, p, _mse(model, theta, r, p), physical_envelope(model, p), _boundary_distance(model, p)))
        return min(candidates, key=lambda x: x.mse)

    bnds = bounds(model)
    fun = lambda x: _mse(model, theta, r, np.asarray(x, float))
    de = differential_evolution(fun, bnds, seed=seed, polish=False, updating="immediate", workers=1, tol=1e-9)
    local = minimize(fun, de.x, method="SLSQP", bounds=bnds, options={"ftol": 1e-12, "maxiter": 1000})
    x = np.asarray(local.x if local.success and local.fun <= de.fun else de.x, float)
    if model in {"M1", "M2", "M3"}:
        x[-1] = canonical_phase(float(x[-1]))
    ok = bool(np.isfinite(fun(x)) and physical_envelope(model, x))
    return FitResult(model, x, fun(x), ok, _boundary_distance(model, x))


def fit_population_m4(theta: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Fit M4 independently to every participant without genome access."""
    radii = np.asarray(radii, float)
    out = np.empty((radii.shape[0], 2), dtype=float)
    for i in range(radii.shape[0]):
        res = fit_m4_profile(theta, radii[i])
        if not res.success:
            raise RuntimeError(f"M4 fit failed for participant {i}")
        out[i] = res.params
    return out


def reconstruct_m4(theta: np.ndarray, params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, float)
    a = params[:, [0]]
    b = params[:, [1]]
    return a * np.exp(b * np.asarray(theta, float)[None, :])
