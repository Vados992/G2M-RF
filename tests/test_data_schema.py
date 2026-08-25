import json

import numpy as np
import pytest

from g2mrf.data.io import DataBundle
from g2mrf.data.synthetic import make_synthetic_dataset
from g2mrf.pipeline import write_report


def test_valid_synthetic_bundle_passes_schema_validation():
    bundle = make_synthetic_dataset(n=100, variants=30, landmarks=15, seed=90)
    assert bundle.validate() is bundle


def test_duplicate_sample_ids_rejected():
    bundle = make_synthetic_dataset(n=100, variants=30, landmarks=15, seed=91)
    bundle.sample_ids[1] = bundle.sample_ids[0]
    with pytest.raises(ValueError, match="unique"):
        bundle.validate()


def test_family_count_mismatch_rejected():
    bundle = make_synthetic_dataset(n=100, variants=30, landmarks=15, seed=92)
    bundle.family_ids = bundle.family_ids[:-1]
    with pytest.raises(ValueError, match="family_ids"):
        bundle.validate()


def test_genotype_sample_count_mismatch_rejected():
    bundle = make_synthetic_dataset(n=100, variants=30, landmarks=15, seed=93)
    bundle.G = bundle.G[:-1]
    with pytest.raises(ValueError, match="N x M"):
        bundle.validate()


def test_landmark_angle_count_mismatch_rejected():
    bundle = make_synthetic_dataset(n=100, variants=30, landmarks=15, seed=94)
    bundle.angles = bundle.angles[:-1]
    with pytest.raises(ValueError, match="angles"):
        bundle.validate()


def test_nonfinite_covariate_rejected():
    bundle = make_synthetic_dataset(n=100, variants=30, landmarks=15, seed=95)
    bundle.C[0, 0] = np.nan
    with pytest.raises(ValueError, match="covariates"):
        bundle.validate()


def test_negative_or_nonfinite_radii_rejected():
    bundle = make_synthetic_dataset(n=100, variants=30, landmarks=15, seed=96)
    bundle.radii[0, 0] = -0.1
    with pytest.raises(ValueError, match="radii"):
        bundle.validate()
    bundle = make_synthetic_dataset(n=100, variants=30, landmarks=15, seed=97)
    bundle.radii[0, 0] = np.inf
    with pytest.raises(ValueError, match="radii"):
        bundle.validate()


def test_strict_json_writer_replaces_nonfinite_values_with_null(tmp_path):
    report = {"x": float("nan"), "y": float("inf"), "z": np.float64(1.25)}
    path = write_report(report, tmp_path / "report.json")
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    parsed = json.loads(raw)
    assert parsed == {"x": None, "y": None, "z": 1.25}


def test_manual_bundle_requires_nonempty_sample_ids():
    bundle = DataBundle(
        sample_ids=np.array([]),
        family_ids=np.array([]),
        G=np.empty((0, 1)),
        C=np.empty((0, 1)),
        radii=np.empty((0, 1)),
        angles=np.array([0.0]),
        landmark_classes=np.array(["class_1"]),
    )
    with pytest.raises(ValueError, match="sample_ids"):
        bundle.validate()
