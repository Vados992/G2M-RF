import numpy as np

import g2mrf.genomics.krr as krr_module
from g2mrf.genomics.krr import NystromLinearKRR, make_genomic_regressor


def test_large_auto_configuration_selects_nystrom():
    model = make_genomic_regressor(
        n_train=250_000,
        lambda_=1.0,
        solver="auto",
        exact_threshold=2_500,
        nystrom_components=512,
    )
    assert isinstance(model, NystromLinearKRR)
    assert model.n_components == 512


def test_nystrom_never_forms_full_n_by_n_training_kernel(monkeypatch):
    rng = np.random.default_rng(81)
    n = 180
    components = 24
    X = rng.normal(size=(n, 40))
    C = rng.normal(size=(n, 2))
    y = X[:, :5].sum(axis=1) + 0.1 * C[:, 0]

    original = krr_module.linear_kernel
    output_shapes = []

    def recording_kernel(A, B=None):
        out = original(A, B)
        output_shapes.append(out.shape)
        return out

    monkeypatch.setattr(krr_module, "linear_kernel", recording_kernel)
    model = NystromLinearKRR(lambda_=0.5, n_components=components, seed=7)
    model.fit(X, y, C)
    model.predict(X[:20], C[:20])

    assert (n, n) not in output_shapes
    assert (components, components) in output_shapes
    assert (n, components) in output_shapes
    assert max(r * c for r, c in output_shapes) <= n * components


def test_nystrom_storage_is_bounded_by_landmark_count():
    rng = np.random.default_rng(82)
    X = rng.normal(size=(150, 25))
    C = rng.normal(size=(150, 2))
    y = rng.normal(size=150)
    model = NystromLinearKRR(lambda_=1.0, n_components=20, seed=8).fit(X, y, C)
    assert model.landmark_X_.shape == (20, 25)
    assert model.transform_.shape[0] == 20
    assert model.beta_.shape[0] <= 20
