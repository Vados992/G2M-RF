from g2mrf.config import GateConfig
from g2mrf.gates import evaluate_gates


def test_gate_chain_strong_support():
    m = {
        "dR2_direct": 0.20, "dR2_direct_ci_low": 0.10, "g1_class_pass_count": 5,
        "k": 2, "g2_theta_pass_count": 2, "theta_multivariate_positive": True,
        "nrmse_mgc": 0.02, "nrmse_cov": 0.05, "nrmse_pca": 0.04,
        "eta": 0.95, "eta_ci_low": 0.85,
    }
    r = evaluate_gates(m, GateConfig())
    assert r.strong_support
    assert r.label == "STRONG G2M-RF SUPPORT"


def test_gate_chain_stops_at_g1():
    m = {
        "dR2_direct": 0.005, "dR2_direct_ci_low": -0.01, "g1_class_pass_count": 0,
        "k": 2, "g2_theta_pass_count": 2, "theta_multivariate_positive": True,
        "nrmse_mgc": 0.02, "nrmse_cov": 0.05, "nrmse_pca": 0.04,
        "eta": 10.0, "eta_ci_low": 9.0,
    }
    r = evaluate_gates(m, GateConfig())
    assert not r.strong_support
    assert r.label == "NO GENOME-WIDE GEOMETRY SUPPORT"
