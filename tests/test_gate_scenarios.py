import math

import pytest

from g2mrf.config import GateConfig
from g2mrf.gates import evaluate_gates


def passing_metrics(**overrides):
    metrics = {
        "dR2_direct": 0.20,
        "dR2_direct_ci_low": 0.10,
        "g1_class_pass_count": 5,
        "k": 2,
        "g2_theta_pass_count": 2,
        "theta_multivariate_positive": True,
        "nrmse_mgc": 0.02,
        "nrmse_cov": 0.05,
        "nrmse_pca": 0.04,
        "eta": 0.95,
        "eta_ci_low": 0.85,
    }
    metrics.update(overrides)
    return metrics


def test_all_gate_boundaries_pass_when_inclusive():
    cfg = GateConfig()
    m = passing_metrics(
        dR2_direct=cfg.g5_direct_floor,
        dR2_direct_ci_low=1e-12,
        g1_class_pass_count=3,
        k=2,
        g2_theta_pass_count=1,
        nrmse_mgc=0.02,
        nrmse_cov=0.03,
        nrmse_pca=0.03,
        eta=cfg.g5_eta,
        eta_ci_low=cfg.g5_eta_ci_lower,
    )
    report = evaluate_gates(m, cfg)
    assert report.strong_support


@pytest.mark.parametrize(
    ("overrides", "label"),
    [
        ({"dR2_direct": 0.019}, "NO GENOME-WIDE GEOMETRY SUPPORT"),
        ({"dR2_direct_ci_low": 0.0}, "NO GENOME-WIDE GEOMETRY SUPPORT"),
        ({"g1_class_pass_count": 2}, "NO GENOME-WIDE GEOMETRY SUPPORT"),
        ({"g2_theta_pass_count": 0}, "GEOMETRY SIGNAL WITHOUT MGC LATENT SUPPORT"),
        ({"theta_multivariate_positive": False}, "GEOMETRY SIGNAL WITHOUT MGC LATENT SUPPORT"),
        ({"nrmse_mgc": 0.041}, "MGC BOTTLENECK NOT USEFUL"),
        ({"nrmse_mgc": 0.031, "nrmse_pca": 0.04}, "MGC NOT PRIVILEGED VS PCA"),
        ({"eta": 0.89}, "MGC SIGNAL LOSS TOO LARGE"),
        ({"eta_ci_low": 0.74}, "MGC SIGNAL LOSS TOO LARGE"),
        ({"k": 4, "g2_theta_pass_count": 4}, "MGC SIGNAL LOSS TOO LARGE"),
    ],
)
def test_each_gate_failure_has_expected_label(overrides, label):
    report = evaluate_gates(passing_metrics(**overrides), GateConfig())
    assert not report.strong_support
    assert report.label == label


def test_external_transfer_failure_has_distinct_label():
    report = evaluate_gates(passing_metrics(), GateConfig(), external_pass=False)
    assert not report.strong_support
    assert report.label == "NOT TRANSPORTABLE"
    assert not report.gates[-1].passed


@pytest.mark.parametrize(("qc", "no_leakage"), [(False, True), (True, False), (False, False)])
def test_qc_or_leakage_failure_invalidates_report(qc, no_leakage):
    report = evaluate_gates(passing_metrics(), GateConfig(), qc=qc, no_leakage=no_leakage)
    assert not report.strong_support
    assert report.label == "INVALID"


def test_g2_uses_ceiling_half_of_k():
    m = passing_metrics(k=3, g2_theta_pass_count=1)
    report = evaluate_gates(m, GateConfig())
    assert report.gates[1].metrics["needed"] == 2
    assert not report.gates[1].passed


def test_nonfinite_eta_fails_g5():
    report = evaluate_gates(passing_metrics(eta=math.nan), GateConfig())
    assert not report.gates[-1].passed
    assert report.label == "MGC SIGNAL LOSS TOO LARGE"


def test_nonpositive_k_is_rejected():
    with pytest.raises(ValueError, match="k must be positive"):
        evaluate_gates(passing_metrics(k=0), GateConfig())
