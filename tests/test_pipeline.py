from g2mrf.config import RunConfig
from g2mrf.data.synthetic import make_synthetic_dataset
from g2mrf.pipeline import run_confirmatory


def test_end_to_end_smoke():
    cfg = RunConfig()
    cfg.model.bootstrap = 30
    cfg.model.solver = "nystrom"
    cfg.model.nystrom_components = 80
    bundle = make_synthetic_dataset(n=420, variants=180, landmarks=20, seed=cfg.model.seed)
    report = run_confirmatory(bundle, cfg)
    assert report["framework"] == "G2M-RF v2.0"
    assert report["metrics"]["n_train"] > 200
    assert report["metrics"]["n_external"] > 20
    assert len(report["decision"]["gates"]) == 5
