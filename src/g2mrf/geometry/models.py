from __future__ import annotations

import math
import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0
B_PHI = 2.0 * math.log(PHI) / math.pi
THETA_MIN = -math.pi / 2.0
THETA_MAX = math.pi / 2.0
R_MAX = 0.75


def canonical_phase(delta: float) -> float:
    return float((delta + math.pi) % (2.0 * math.pi) - math.pi)


def radius(model: str, theta: np.ndarray, params: np.ndarray | list[float]) -> np.ndarray:
    t = np.asarray(theta, dtype=float)
    p = np.asarray(params, dtype=float)
    m = model.upper()
    if m == "M0":
        return np.full_like(t, p[0], dtype=float)
    if m == "M1":
        R, delta = p
        return 0.5 * R * (1.0 + np.cos(t - delta))
    if m == "M2":
        q, e, delta = p
        return q / (1.0 + e * np.cos(t - delta))
    if m == "M3":
        q, delta = p
        return q / (1.0 + np.cos(t - delta))
    if m == "M4":
        a, b = p
        return a * np.exp(b * t)
    if m == "M5":
        a, s = p
        s = 1.0 if s >= 0 else -1.0
        return a * np.exp(s * B_PHI * t)
    raise ValueError(f"Unknown model {model}")


def bounds(model: str) -> list[tuple[float, float]]:
    m = model.upper()
    return {
        "M0": [(0.05, 0.75)],
        "M1": [(0.05, 0.75), (-math.pi, math.pi)],
        "M2": [(0.01, 0.75), (0.0, 0.95), (-math.pi, math.pi)],
        "M3": [(0.01, 0.75), (-math.pi, math.pi)],
        "M4": [(0.01, 0.75), (-2.0, 2.0)],
        "M5": [(0.01, 0.75), (-1.0, 1.0)],
    }[m]


def _critical_angles(delta: float) -> list[float]:
    out = [THETA_MIN, THETA_MAX]
    for k in range(-2, 3):
        for base in (delta + 2 * math.pi * k, delta + math.pi + 2 * math.pi * k):
            if THETA_MIN <= base <= THETA_MAX:
                out.append(base)
    return out


def physical_envelope(model: str, params: np.ndarray | list[float], tol: float = 1e-12) -> bool:
    """Analytic/candidate-extrema feasibility check on theta in [-pi/2, pi/2]."""
    p = np.asarray(params, dtype=float)
    m = model.upper()
    if not np.all(np.isfinite(p)):
        return False
    try:
        if m == "M0":
            vals = np.array([p[0]])
        elif m == "M1":
            R, delta = p
            if not (0.05 <= R <= 0.75):
                return False
            vals = radius(m, np.asarray(_critical_angles(delta)), p)
        elif m == "M2":
            q, e, delta = p
            if not (0.01 <= q <= 0.75 and 0.0 <= e <= 0.95):
                return False
            ang = np.asarray(_critical_angles(delta))
            den = 1.0 + e * np.cos(ang - delta)
            if np.min(den) <= tol:
                return False
            vals = q / den
        elif m == "M3":
            q, delta = p
            if not (0.01 <= q <= 0.75):
                return False
            ang = np.asarray(_critical_angles(delta))
            den = 1.0 + np.cos(ang - delta)
            if np.min(den) <= tol:
                return False
            vals = q / den
        elif m == "M4":
            a, b = p
            if not (0.01 <= a <= 0.75 and -2.0 <= b <= 2.0):
                return False
            vals = radius(m, np.array([THETA_MIN, THETA_MAX]), p)
        elif m == "M5":
            a, s = p
            if not (0.01 <= a <= 0.75):
                return False
            s = 1.0 if s >= 0 else -1.0
            vals = radius(m, np.array([THETA_MIN, THETA_MAX]), [a, s])
        else:
            return False
    except (FloatingPointError, OverflowError, ValueError):
        return False
    return bool(np.all(np.isfinite(vals)) and np.min(vals) >= -tol and np.max(vals) <= R_MAX + tol)
