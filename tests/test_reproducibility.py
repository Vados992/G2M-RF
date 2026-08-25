import json

import numpy as np

from g2mrf.config import RunConfig
from g2mrf.data.io import load_npz, save_npz
from g2mrf.data.synthetic import make_synthetic_dataset
from g2mrf.genomics.krr import NystromLinearKRR
from g2mrf.pipeline import run_confirmatory, write_report
from g2mrf.splitting import family_aware_split
from g2mrf.statistics.bootstrap import bootstrap_incremental_r2


def test_synthetic_dataset_same_seed_is_identical():
    a = make_synthetic_dataset(n=120, variants=40, landmarks=15, seed=100)
    b = make_synthetic_dataset(n=120, variants=40, landmarks=15, seed=100)
    for name in ["sample_ids", "family_ids", "G", "C", "radii", "angles", "landmark_classes"]:
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_synthetic_dataset_different_seed_changes_genotypes():
    a = make_synthetic_dataset(n=120, variants=40, landmarks=15, seed=100)
    b = make_synthetic_dataset(n=120, variants=40, landmarks=15, seed=101)
    assert not np.array_equal(a.G, b.G)


def test_split_is_reproducible_for_same_salt_and_changes_for_new_salt():
    b = make_synthetic_dataset(n=400, variants=30, landmarks=15, seed=120)
    a = family_aware_split(b.family_ids, salt="A")
    c = family_aware_split(b.family_ids, salt="A")
    d = family_aware_split(b.family_ids, salt="B")
    assert np.array_equal(a, c)
    assert np.any(a != d)


def test_bootstrap_seed_controls_output():
    rng = np.random.default_rng(8)
    y = rng.normal(size=(100, 3))
    base = np.zeros_like(y)
    full = y + rng.normal(scale=0.2, size=y.shape)
    a = bootstrap_incremental_r2(y, full, base, B=80, seed=5)
    b = bootstrap_incremental_r2(y, full, base, B=80, seed=5)
    c = bootstrap_incremental_r2(y, full, base, B=80, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_nystrom_same_seed_produces_identical_predictions():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(120, 30))
    C = rng.normal(size=(120, 2))
    y = X[:, :4].sum(axis=1) + C[:, 0]
    a = NystromLinearKRR(lambda_=0.2, n_components=40, seed=77).fit(X, y, C)
    b = NystromLinearKRR(lambda_=0.2, n_components=40, seed=77).fit(X, y, C)
    assert np.array_equal(a.landmark_X_, b.landmark_X_)
    assert np.allclose(a.predict(X, C), b.predict(X, C), atol=1e-12)


def test_npz_roundtrip_is_lossless(tmp_path):
    bundle = make_synthetic_dataset(n=100, variants=30, landmarks=15, seed=130)
    path = tmp_path / "bundle.npz"
    save_npz(bundle, path)
    restored = load_npz(path)
    for name in ["sample_ids", "family_ids", "G", "C", "radii", "angles", "landmark_classes"]:
        assert np.array_equal(getattr(bundle, name), getattr(restored, name))


def test_report_serialization_is_stable(tmp_path):
    cfg = RunConfig()
    cfg.model.bootstrap = 20
    cfg.model.solver = "nystrom"
    cfg.model.nystrom_components = 60
    bundle = make_synthetic_dataset(n=420, variants=120, landmarks=20, seed=cfg.model.seed)
    a = run_confirmatory(bundle, cfg)
    b = run_confirmatory(bundle, cfg)
    assert a["decision"] == b["decision"]
    assert a["metrics"] == b["metrics"]
    p1 = write_report(a, tmp_path / "a.json")
    p2 = write_report(b, tmp_path / "b.json")
    assert json.loads(p1.read_text()) == json.loads(p2.read_text())
