import json

import numpy as np

from g2mrf.cli import main
from g2mrf.config import RunConfig
from g2mrf.data.synthetic import make_synthetic_dataset
from g2mrf.pipeline import run_confirmatory, write_report


def test_cli_plan_smoke(capsys):
    main(["plan", "--h2", "0.35", "--me", "75000", "--n", "20000", "50000", "--target", "0.05"])
    out = capsys.readouterr().out
    assert "N_TRAIN" in out
    assert "20000" in out
    assert "required N" in out


def test_cli_generate_and_run_roundtrip(tmp_path):
    data = tmp_path / "synthetic.npz"
    report = tmp_path / "report.json"
    main(["generate", "--n", "420", "--variants", "100", "--landmarks", "20", "--seed", "17", "--out", str(data)])
    assert data.exists()
    main(["run", "--data", str(data), "--out", str(report)])
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["framework"] == "G2M-RF v2.0"
    assert len(payload["decision"]["gates"]) == 5


def test_exact_pipeline_integration(tmp_path):
    cfg = RunConfig()
    cfg.model.solver = "exact"
    cfg.model.bootstrap = 20
    bundle = make_synthetic_dataset(n=420, variants=80, landmarks=20, seed=31)
    report = run_confirmatory(bundle, cfg)
    assert report["metrics"]["n_train"] > report["metrics"]["n_external"]
    assert np.isfinite(report["metrics"]["nrmse_direct"])
    out = write_report(report, tmp_path / "exact.json")
    assert out.exists()


def test_nystrom_pipeline_integration():
    cfg = RunConfig()
    cfg.model.solver = "nystrom"
    cfg.model.nystrom_components = 50
    cfg.model.bootstrap = 20
    bundle = make_synthetic_dataset(n=420, variants=100, landmarks=20, seed=32)
    report = run_confirmatory(bundle, cfg)
    assert np.isfinite(report["metrics"]["r2_direct"])
    assert report["metrics"]["k"] <= cfg.gates.max_k


def test_auto_solver_small_cohort_matches_exact_decision():
    bundle = make_synthetic_dataset(n=420, variants=80, landmarks=20, seed=33)
    exact = RunConfig()
    exact.model.solver = "exact"
    exact.model.bootstrap = 15
    auto = RunConfig()
    auto.model.solver = "auto"
    auto.model.exact_threshold = 10_000
    auto.model.bootstrap = 15
    a = run_confirmatory(bundle, exact)
    b = run_confirmatory(bundle, auto)
    assert a["decision"] == b["decision"]
    for key in ["r2_direct", "r2_mgc", "r2_pca", "nrmse_direct"]:
        assert a["metrics"][key] == b["metrics"][key]
