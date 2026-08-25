import numpy as np
import pytest

from g2mrf.statistics.bootstrap import (
    bootstrap_eta,
    bootstrap_incremental_r2,
    bootstrap_one_sided_p,
    percentile_ci,
)
from g2mrf.statistics.metrics import aggregate_r2, nrmse, participant_mse


def test_percentile_ci_known_quantiles():
    values = np.arange(101.0)
    lo, hi = percentile_ci(values, alpha=0.10)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(95.0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_percentile_ci_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        percentile_ci(np.arange(10.0), alpha=alpha)


@pytest.mark.parametrize("B", [0, -1, 1.5])
def test_bootstrap_rejects_invalid_replicate_count(B):
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="B"):
        bootstrap_incremental_r2(y, y, np.zeros_like(y), B=B)


def test_bootstrap_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        bootstrap_incremental_r2(np.zeros(10), np.zeros(9), np.zeros(10), B=10)


def test_bootstrap_eta_returns_empty_when_direct_increment_never_positive():
    y = np.linspace(-1, 1, 40)
    base = y.copy()
    worse = np.zeros_like(y)
    vals = bootstrap_eta(y, base, worse, worse, B=20, seed=4)
    assert vals.size == 0


def test_one_sided_p_has_plus_one_correction():
    samples = np.array([1.0, 2.0, 3.0, 4.0])
    assert bootstrap_one_sided_p(samples, null=0.0) == pytest.approx(1.0 / 5.0)


def test_one_sided_p_rejects_nonfinite_null():
    with pytest.raises(ValueError, match="null"):
        bootstrap_one_sided_p(np.ones(5), null=float("nan"))


def test_constant_target_r2_is_undefined():
    y = np.ones(20)
    assert np.isnan(aggregate_r2(y, y))


def test_nrmse_known_value():
    y = np.array([[0.0, 1.0], [0.0, 1.0]])
    pred = np.array([[0.0, 0.0], [0.0, 0.0]])
    assert nrmse(y, pred) == pytest.approx(np.sqrt(0.5))


def test_participant_mse_reduces_all_feature_axes():
    y = np.zeros((2, 2, 2))
    pred = np.array([np.ones((2, 2)), 2 * np.ones((2, 2))])
    assert np.allclose(participant_mse(y, pred), [1.0, 4.0])


def test_metric_rejects_nonfinite_inputs():
    with pytest.raises(ValueError, match="NaN/Inf"):
        nrmse(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
