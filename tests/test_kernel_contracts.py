import numpy as np
import pytest

from g2mrf.genomics.kernels import (
    composite_kernel,
    linear_kernel,
    projection_matrix,
    residualized_kernel,
    trace_normalize,
)


def test_linear_kernel_rejects_empty_feature_axis():
    with pytest.raises(ValueError, match="non-empty 2-D"):
        linear_kernel(np.empty((5, 0)))


def test_linear_kernel_rejects_nonfinite_values():
    X = np.ones((4, 3))
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        linear_kernel(X)


def test_trace_normalize_rejects_nonsquare_matrix():
    with pytest.raises(ValueError, match="square"):
        trace_normalize(np.ones((3, 2)))


def test_composite_kernel_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="identical shape"):
        composite_kernel([np.eye(3), np.eye(4)], [0.5, 0.5])


def test_composite_kernel_rejects_nonfinite_weight():
    with pytest.raises(ValueError, match="finite"):
        composite_kernel([np.eye(3), np.eye(3)], [float("nan"), 1.0])


def test_projection_matrix_is_symmetric_and_idempotent():
    rng = np.random.default_rng(80)
    C = np.column_stack([np.ones(30), rng.normal(size=30), rng.normal(size=30)])
    P, M = projection_matrix(C)
    assert np.allclose(P, P.T, atol=1e-10)
    assert np.allclose(M, M.T, atol=1e-10)
    assert np.allclose(P @ P, P, atol=1e-9)
    assert np.allclose(M @ M, M, atol=1e-9)
    assert np.allclose(P @ M, 0.0, atol=1e-9)


def test_projection_matrix_rejects_nonfinite_covariates():
    C = np.ones((5, 2))
    C[0, 1] = np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        projection_matrix(C)


def test_residualized_kernel_rejects_sample_count_mismatch():
    with pytest.raises(ValueError, match="sample counts"):
        residualized_kernel(np.eye(5), np.ones((4, 1)))
