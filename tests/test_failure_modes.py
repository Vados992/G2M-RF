import math

import numpy as np
import pytest

from g2mrf.data.synthetic import make_synthetic_dataset
from g2mrf.genomics.kernels import composite_kernel, linear_kernel, trace_normalize
from g2mrf.genomics.krr import ExactLinearKRR, NystromLinearKRR, make_genomic_regressor
from g2mrf.genomics.preprocessing import GenotypeStandardizer
from g2mrf.geometry.coordinates import cartesian_to_polar, normalize_by_height
from g2mrf.geometry.fit import fit_model
from g2mrf.geometry.models import physical_envelope, radius
from g2mrf.pipeline import run_confirmatory
from g2mrf.planning import daetwyler_r2, required_n
from g2mrf.statistics.metrics import aggregate_r2


@pytest.mark.parametrize(
    "args",
    [(0, 0.2, 100_000), (-1, 0.2, 100_000), (100, 0.0, 100_000), (100, 1.1, 100_000), (100, 0.2, 0)],
)
def test_planning_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        daetwyler_r2(*args)


@pytest.mark.parametrize("target,h2,me", [(0.0, 0.2, 1), (0.2, 0.2, 1), (0.3, 0.2, 1), (0.1, 1.1, 1), (0.1, 0.2, 0)])
def test_required_n_rejects_invalid_inputs(target, h2, me):
    with pytest.raises(ValueError):
        required_n(target, h2, me)


def test_standardizer_rejects_all_monomorphic_variants():
    G = np.ones((20, 4))
    with pytest.raises(ValueError, match="No polymorphic variants remain"):
        GenotypeStandardizer().fit(G)


def test_standardizer_imputes_missing_values_to_finite_output():
    G = np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 2.0], [1.0, 1.0]])
    X = GenotypeStandardizer().fit_transform(G)
    assert np.all(np.isfinite(X))


def test_standardizer_transform_before_fit_fails():
    with pytest.raises(RuntimeError, match="fit first"):
        GenotypeStandardizer().transform(np.zeros((2, 2)))


def test_fit_model_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same-length"):
        fit_model("M4", np.array([0.0, 1.0]), np.array([0.2]))


def test_fit_model_rejects_nan_input():
    with pytest.raises(ValueError, match="NaN/Inf"):
        fit_model("M4", np.array([0.0, 1.0]), np.array([0.2, np.nan]))


def test_unknown_geometry_model_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        radius("MX", np.array([0.0]), [0.2])
    assert not physical_envelope("MX", [0.2])


def test_parabola_singularity_is_rejected():
    assert not physical_envelope("M3", [0.2, -math.pi / 2])


def test_height_normalization_rejects_nonpositive_height():
    with pytest.raises(ValueError, match="positive"):
        normalize_by_height(np.ones((2, 2)), 0.0)


def test_polar_conversion_rejects_wrong_last_dimension():
    with pytest.raises(ValueError, match="last dimension"):
        cartesian_to_polar(np.ones((3, 3)))


def test_kernel_rejects_feature_dimension_mismatch():
    with pytest.raises(ValueError, match="feature dimensions differ"):
        linear_kernel(np.ones((4, 2)), np.ones((5, 3)))


def test_trace_normalize_rejects_zero_trace():
    with pytest.raises(ValueError, match="trace"):
        trace_normalize(np.zeros((4, 4)))


@pytest.mark.parametrize("weights", [[0.5], [0.8, 0.8], [-0.1, 1.1]])
def test_composite_kernel_rejects_invalid_weights(weights):
    kernels = [np.eye(3), np.eye(3)]
    with pytest.raises(ValueError):
        composite_kernel(kernels, weights)


def test_exact_predict_before_fit_fails():
    with pytest.raises(RuntimeError, match="fit first"):
        ExactLinearKRR().predict(np.ones((2, 2)), np.zeros((2, 1)))


def test_nystrom_predict_before_fit_fails():
    with pytest.raises(RuntimeError, match="fit first"):
        NystromLinearKRR().predict(np.ones((2, 2)), np.zeros((2, 1)))


def test_invalid_solver_name_rejected():
    with pytest.raises(ValueError, match="auto/exact/nystrom"):
        make_genomic_regressor(100, 1.0, solver="invalid")


def test_aggregate_r2_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        aggregate_r2(np.zeros((3, 2)), np.zeros((3, 1)))


def test_pipeline_rejects_too_small_split():
    bundle = make_synthetic_dataset(n=40, variants=30, landmarks=15)
    with pytest.raises(ValueError, match="split too small"):
        run_confirmatory(bundle)
