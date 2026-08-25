import numpy as np
import pytest

from g2mrf.genomics.krr import (
    ExactLinearKRR,
    FixedEffectKRR,
    NystromLinearKRR,
    make_genomic_regressor,
)
from g2mrf.genomics.kernels import linear_kernel


def test_auto_solver_selects_expected_backend():
    assert isinstance(make_genomic_regressor(100, 1.0, "auto", exact_threshold=200), ExactLinearKRR)
    assert isinstance(make_genomic_regressor(300, 1.0, "auto", exact_threshold=200), NystromLinearKRR)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_solver_rejects_invalid_regularization(bad):
    with pytest.raises(ValueError, match="lambda_"):
        make_genomic_regressor(100, bad)


def test_fixed_effect_component_remains_unpenalized():
    rng = np.random.default_rng(70)
    X = rng.normal(size=(80, 20))
    C = rng.normal(size=(80, 2))
    gamma = np.array([1.2, -0.7])
    y = 2.5 + C @ gamma
    model = ExactLinearKRR(lambda_=1e4).fit(X, y, C)
    pred = model.predict(X, C)
    assert np.max(np.abs(pred - y)) < 1e-8


def test_exact_and_full_rank_nystrom_are_numerically_close():
    rng = np.random.default_rng(71)
    X = rng.normal(size=(45, 12))
    C = rng.normal(size=(45, 2))
    y = X[:, :4] @ np.array([0.5, -0.4, 0.2, 0.1]) + 0.3 * C[:, 0] - 0.2 * C[:, 1]
    exact = ExactLinearKRR(lambda_=0.7).fit(X, y, C)
    approx = NystromLinearKRR(lambda_=0.7, n_components=X.shape[0], seed=11).fit(X, y, C)
    assert np.allclose(exact.predict(X, C), approx.predict(X, C), atol=1e-7, rtol=1e-7)


def test_fixed_effect_krr_rejects_nonsymmetric_training_kernel():
    K = np.array([[1.0, 0.2], [0.1, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        FixedEffectKRR().fit(K, np.array([1.0, 2.0]), np.zeros((2, 1)))


def test_fixed_effect_krr_rejects_prediction_shape_mismatch():
    X = np.eye(5)
    K = linear_kernel(X)
    model = FixedEffectKRR().fit(K, np.arange(5.0), np.zeros((5, 1)))
    with pytest.raises(ValueError, match="prediction shape mismatch"):
        model.predict(np.ones((2, 4)), np.zeros((2, 1)))


def test_nystrom_rejects_zero_rank_landmark_kernel():
    X = np.zeros((20, 5))
    y = np.arange(20.0)
    C = np.zeros((20, 1))
    with pytest.raises(ValueError, match="no positive eigenvalues"):
        NystromLinearKRR(lambda_=1.0, n_components=10).fit(X, y, C)


def test_solver_factory_rejects_nonpositive_sizes():
    with pytest.raises(ValueError, match="n_train"):
        make_genomic_regressor(0, 1.0)
    with pytest.raises(ValueError, match="exact_threshold"):
        make_genomic_regressor(10, 1.0, exact_threshold=0)
    with pytest.raises(ValueError, match="nystrom_components"):
        make_genomic_regressor(10, 1.0, nystrom_components=0)
