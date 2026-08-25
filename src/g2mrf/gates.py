from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import GateConfig

_NUMERIC_ATOL = 1e-12


def _geq(value: float, threshold: float) -> bool:
    """Numerically stable inclusive comparison for pre-registered decimal thresholds."""
    return bool(math.isfinite(value) and value >= threshold - _NUMERIC_ATOL)


@dataclass
class GateResult:
    name: str
    passed: bool
    reason: str
    metrics: dict

    def to_dict(self):
        return asdict(self)


@dataclass
class DecisionReport:
    gates: list[GateResult]
    strong_support: bool
    label: str

    def to_dict(self):
        return {
            "gates": [g.to_dict() for g in self.gates],
            "strong_support": self.strong_support,
            "label": self.label,
        }


def evaluate_gates(
    metrics: dict,
    cfg: GateConfig,
    external_pass: bool = True,
    qc: bool = True,
    no_leakage: bool = True,
) -> DecisionReport:
    class_ok = metrics.get("g1_class_pass_count", 0) >= 3
    d_direct = float(metrics["dR2_direct"])
    d_direct_ci_low = float(metrics["dR2_direct_ci_low"])
    g1 = _geq(d_direct, cfg.g1_delta_r2) and d_direct_ci_low > 0.0 and class_ok
    G1 = GateResult(
        "G1",
        g1,
        "external genome-wide geometry signal",
        {
            "dR2_direct": d_direct,
            "ci_low": d_direct_ci_low,
            "class_pass_count": metrics.get("g1_class_pass_count", 0),
        },
    )

    k = int(metrics["k"])
    if k <= 0:
        raise ValueError("k must be positive")
    needed = math.ceil(k / 2)
    g2 = (
        int(metrics.get("g2_theta_pass_count", 0)) >= needed
        and metrics.get("theta_multivariate_positive", False)
    )
    G2 = GateResult(
        "G2",
        g2,
        "genome predicts preregistered morphology parameters",
        {"pass_count": metrics.get("g2_theta_pass_count", 0), "needed": needed},
    )

    nrmse_mgc = float(metrics["nrmse_mgc"])
    nrmse_cov = float(metrics["nrmse_cov"])
    g3 = _geq(nrmse_cov - nrmse_mgc, cfg.g3_nrmse_gain)
    G3 = GateResult(
        "G3",
        g3,
        "MGC bottleneck improves geometry over covariates",
        {"nrmse_mgc": nrmse_mgc, "nrmse_cov": nrmse_cov},
    )

    nrmse_pca = float(metrics["nrmse_pca"])
    g4 = _geq(nrmse_pca - nrmse_mgc, cfg.g4_nrmse_gain)
    G4 = GateResult(
        "G4",
        g4,
        "MGC is privileged versus matched-dimensional PCA",
        {"nrmse_mgc": nrmse_mgc, "nrmse_pca": nrmse_pca},
    )

    eta = float(metrics.get("eta", float("nan")))
    eta_low = float(metrics.get("eta_ci_low", float("nan")))
    g5_core = (
        _geq(d_direct, cfg.g5_direct_floor)
        and _geq(eta, cfg.g5_eta)
        and _geq(eta_low, cfg.g5_eta_ci_lower)
        and k <= cfg.max_k
    )
    g5 = g5_core and external_pass
    G5 = GateResult(
        "G5",
        g5,
        "genomic-morphological compression",
        {
            "dR2_direct": d_direct,
            "eta": eta,
            "eta_ci_low": eta_low,
            "k": k,
            "external_pass": external_pass,
        },
    )

    gates = [G1, G2, G3, G4, G5]
    strong = bool(all(g.passed for g in gates) and qc and no_leakage)
    if not qc or not no_leakage:
        label = "INVALID"
    elif not G1.passed:
        label = "NO GENOME-WIDE GEOMETRY SUPPORT"
    elif not G2.passed:
        label = "GEOMETRY SIGNAL WITHOUT MGC LATENT SUPPORT"
    elif not G3.passed:
        label = "MGC BOTTLENECK NOT USEFUL"
    elif not G4.passed:
        label = "MGC NOT PRIVILEGED VS PCA"
    elif not external_pass:
        label = "NOT TRANSPORTABLE"
    elif not g5_core:
        label = "MGC SIGNAL LOSS TOO LARGE"
    else:
        label = "STRONG G2M-RF SUPPORT"
    return DecisionReport(gates, strong, label)
