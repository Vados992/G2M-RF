import math

import numpy as np
import pytest

from g2mrf.genomics.kernels import (
    composite_kernel,
    linear_kernel,
    residualized_kernel,
    trace_normalize,
)
from g2mrf.genomics.krr import penalized_fwl_ridge
from g2mrf.geometry.coordinates import cartesian_to_polar, normalize_by_height
from g2mrf.geometry.fit import fit_m4_profile
from g2mrf.geometry.models import B_PHI, PHI, canonical_phase, physical_envelope, radius
from g2mrf.planning import daetwyler_r2, required_n


@pytest.mark.parametrize("h2,me", [(0.2, 100_000), (0.35, 75_000), (0.5, 50_000)])
def test_planning_r2_is_monotone_and_below_heritability(h2, me):
    ns = [20_000, 50_000, 100_000, 200_000, 500_000]
    vals = [daetwyler_r2(n, h2, me) for n in ns]
    assert vals == sorted(vals)
    assert all(0 < x < h2 for x in vals)


@pytest.mark.parametrize(
    "target,h2,me",
    [(0.01, 0.2, 100_000), (0.02, 0.35, 75_000), (0.05, 0.5, 50_000)],
)
def test_required_n_inverts_planning_equation(target, h2, me):
    n = required_n(target, h2, me)
    assert daetwyler_r2(n, h2, me) == pytest.approx(target, rel=1e-12, abs=1e-12)


def test_golden_spiral_ratio_identity_both_orientations():
    assert math.exp(B_PHI * math.pi / 2) == pytest.approx(PHI, rel=1e-14)
    assert math.exp(-B_PHI * math.pi / 2) == pytest.approx(1 / PHI, rel=1e-14)


@pytest.mark.parametrize("delta", [-9 * math.pi, -3.4, 0.0, 4.2, 11 * math.pi])
def test_canonical_phase_is_periodic_and_bounded(delta):
    a = canonical_phase(delta)
    b = canonical_phase(delta + 8 * math.pi)
    assert -math.pi <= a < math.pi
    assert a == pytest.approx(b, abs=1e-12)


@pytest.mark.parametrize(
    "model,params",
    [
        ("M0", [0.3]),
        ("M1", [0.4, 0.1]),
        ("M2", [0.2, 0.3, -0.2]),
        ("M4", [0.25, 0.15]),
        ("M5", [0.25, 1.0]),
    ],
)
def test_representative_models_are_finite_inside_envelope(model, params):
    theta = np.linspace(-math.pi / 2, math.pi / 2, 101)
    assert physical_envelope(model, params)
    rr = radius(model, theta, params)
    assert np.all(np.isfinite(rr))
    assert np.all(rr >= -1e-12)
    assert np.all(rr <= 0.75 + 1e-12)


def test_m4_profile_recovers_noiseless_parameters():
    theta = np.linspace(-math.pi / 2, math.pi / 2, 31)
    expected = np.array([0.27, -0.31])
    r = radius("M4", theta, expected)
    fit = fit_m4_profile(theta, r)
    assert fit.success
    assert fit.mse < 1e-18
    assert np.allclose(fit.params, expected, atol=1e-7)


def test_height_normalization_vector_broadcasting():
    xy = np.array([[[180.0, 90.0]], [[100.0, 50.0]]])
    h = np.array([180.0, 100.0])
    out = normalize_by_height(xy, h)
    assert np.allclose(out[:, 0, 0], [1.0, 1.0])
    assert np.allclose(out[:, 0, 1], [0.5, 0.5])


def test_cartesian_to_polar_known_axes():
    xy = np.array([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
    r, theta = cartesian_to_polar(xy)
    assert np.allclose(r, [1.0, 2.0, 3.0])
    assert np.allclose(theta, [0.0, math.pi / 2, math.pi])


def test_trace_normalization_sets_mean_diagonal_to_one():
    X = np.arange(60, dtype=float).reshape(12, 5) / 10
    K = trace_normalize(linear_kernel(X))
    assert np.trace(K) / K.shape[0] == pytest.approx(1.0, abs=1e-12)


def test_composite_kernel_is_convex_combination_after_trace_normalization():
    rng = np.random.default_rng(44)
    K1 = linear_kernel(rng.normal(size=(20, 7)))
    K2 = linear_kernel(rng.normal(size=(20, 5)))
    out = composite_kernel([K1, K2], [0.25, 0.75])
    expected = 0.25 * trace_normalize(K1) + 0.75 * trace_normalize(K2)
    assert np.allclose(out, expected)
    assert np.linalg.eigvalsh((out + out.T) / 2).min() > -1e-10


def test_residualized_kernel_annihilates_covariate_space():
    rng = np.random.default_rng(45)
    X = rng.normal(size=(40, 12))
    C = np.column_stack([np.ones(40), rng.normal(size=40), rng.normal(size=40)])
    Kr = residualized_kernel(linear_kernel(X), C)
    assert np.linalg.norm(C.T @ Kr, ord=np.inf) < 1e-8
    assert np.linalg.norm(Kr @ C, ord=np.inf) < 1e-8


def test_penalized_fwl_matches_joint_ridge_normal_equations():
    rng = np.random.default_rng(46)
    n, p, q = 80, 8, 3
    X = rng.normal(size=(n, p))
    C0 = rng.normal(size=(n, q))
    C = np.column_stack([np.ones(n), C0])
    y = rng.normal(size=(n, 2))
    lam = 0.7

    beta, gamma = penalized_fwl_ridge(X, y, C0, lambda_=lam)
    A = np.block(
        [
            [X.T @ X + lam * np.eye(p), X.T @ C],
            [C.T @ X, C.T @ C],
        ]
    )
    rhs = np.vstack([X.T @ y, C.T @ y])
    sol = np.linalg.solve(A, rhs)
    assert np.allclose(beta, sol[:p], atol=1e-9)
    assert np.allclose(gamma, sol[p:], atol=1e-9)
