import numpy as np

from g2mrf.genomics.krr import ExactLinearKRR, NystromLinearKRR


def test_exact_krr_multioutput_predicts_signal():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(100, 30))
    C = rng.normal(size=(100, 2))
    B = rng.normal(size=(30, 2)) * 0.1
    y = X @ B + C @ np.array([[0.3, -0.2], [0.1, 0.2]]) + rng.normal(scale=0.02, size=(100, 2))
    model = ExactLinearKRR(lambda_=0.05).fit(X, y, C)
    pred = model.predict(X, C)
    assert pred.shape == y.shape
    assert np.mean((pred - y) ** 2) < np.var(y) * 0.2


def test_nystrom_krr_runs_and_is_finite():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(180, 50))
    C = rng.normal(size=(180, 2))
    y = X[:, :3].sum(axis=1) + 0.2 * C[:, 0] + rng.normal(scale=0.1, size=180)
    model = NystromLinearKRR(lambda_=0.1, n_components=60, seed=3).fit(X, y, C)
    pred = model.predict(X[:10], C[:10])
    assert pred.shape == (10,)
    assert np.all(np.isfinite(pred))
