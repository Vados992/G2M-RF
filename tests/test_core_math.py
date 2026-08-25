import math
import numpy as np

from g2mrf.geometry.models import PHI, B_PHI, radius, physical_envelope
from g2mrf.geometry.fit import fit_model
from g2mrf.geometry.coordinates import normalize_by_height
from g2mrf.genomics.kernels import linear_kernel, residualized_kernel
from g2mrf.genomics.krr import penalized_fwl_ridge
from g2mrf.planning import daetwyler_r2, required_n


def test_golden_spiral_constant():
    assert abs(math.exp(B_PHI * math.pi / 2) - PHI) < 1e-12
    assert abs(0.75 / PHI - 0.4635254915624211) < 1e-12


def test_scale_invariance():
    xy = np.array([[20.0, 30.0], [10.0, 40.0]])
    a = normalize_by_height(xy, 180.0)
    b = normalize_by_height(3.7 * xy, 3.7 * 180.0)
    assert np.allclose(a, b)


def test_m0_analytic_optimum():
    r = np.array([0.2, 0.3, 0.4, 0.5])
    t = np.linspace(-1, 1, len(r))
    fit = fit_model("M0", t, r)
    assert fit.success
    assert abs(fit.params[0] - r.mean()) < 1e-12


def test_cardioid_physical_range():
    t = np.linspace(-math.pi / 2, math.pi / 2, 1001)
    p = np.array([0.6, 1.1])
    rr = radius("M1", t, p)
    assert rr.min() >= -1e-12
    assert rr.max() <= 0.6 + 1e-12
    assert physical_envelope("M1", p)


def test_log_spiral_phase_identity():
    t = np.linspace(-1.5, 1.5, 50)
    a, b, d = 0.3, 0.4, 0.7
    left = a * np.exp(b * (t - d))
    right = (a * np.exp(-b * d)) * np.exp(b * t)
    assert np.max(np.abs(left - right)) < 1e-14


def test_ellipse_denominator_positive():
    t = np.linspace(-math.pi / 2, math.pi / 2, 200)
    den = 1 + 0.95 * np.cos(t - 0.2)
    assert den.min() > 0


def test_parabola_singularity_rejected():
    assert not physical_envelope("M3", [0.2, -math.pi / 2])


def test_kernel_psd():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 20))
    K = linear_kernel(X)
    ev = np.linalg.eigvalsh(K)
    assert ev.min() > -1e-10
    assert np.allclose(K, K.T)


def test_residualized_kernel_orthogonal_to_covariates():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 12))
    C = np.column_stack([np.ones(30), rng.normal(size=30)])
    Kr = residualized_kernel(linear_kernel(X), C)
    assert np.max(np.abs(C.T @ Kr)) < 1e-9
    assert np.linalg.eigvalsh((Kr + Kr.T) / 2).min() > -1e-9


def test_penalized_fwl_counterexample_corrected():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 4.0])
    C = np.empty((3, 0))
    beta, _ = penalized_fwl_ridge(X, y, C, lambda_=1.0)
    assert abs(float(beta[0, 0]) - 4.0 / 3.0) < 1e-12


def test_planning_known_values():
    assert abs(daetwyler_r2(20_000, 0.35, 75_000) - 0.029878048780487803) < 1e-12
    assert abs(required_n(0.05, 0.2, 100_000) - 166_666.66666666663) < 1e-8
