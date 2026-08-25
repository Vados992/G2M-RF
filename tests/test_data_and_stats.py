import numpy as np

from g2mrf.data.synthetic import make_synthetic_dataset
from g2mrf.genomics.preprocessing import GenotypeStandardizer
from g2mrf.splitting import family_aware_split
from g2mrf.statistics.bootstrap import bootstrap_incremental_r2
from g2mrf.statistics.multiplicity import holm_adjust


def test_standardizer_uses_train_frequencies_only():
    Gtr = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 1], [2, 1, 1]], float)
    Gte = np.array([[2, 2, 1], [2, 2, 1]], float)
    gs = GenotypeStandardizer().fit(Gtr)
    Xte = gs.transform(Gte)
    assert Xte.shape[1] == 2
    assert np.all(np.isfinite(Xte))


def test_family_split_deterministic_and_no_family_crossing():
    b = make_synthetic_dataset(n=300, variants=50, landmarks=15)
    a = family_aware_split(b.family_ids)
    c = family_aware_split(b.family_ids)
    assert np.array_equal(a, c)
    for fam in np.unique(b.family_ids):
        assert len(np.unique(a[b.family_ids == fam])) == 1


def test_bootstrap_deterministic():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(80, 4))
    base = np.zeros_like(y)
    full = y + rng.normal(scale=0.1, size=y.shape)
    a = bootstrap_incremental_r2(y, full, base, B=50, seed=9)
    b = bootstrap_incremental_r2(y, full, base, B=50, seed=9)
    assert np.array_equal(a, b)


def test_holm_known_ordering():
    p = np.array([0.01, 0.04, 0.03])
    adj = holm_adjust(p)
    assert np.allclose(adj, [0.03, 0.06, 0.06])
