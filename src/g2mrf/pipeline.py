from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .config import RunConfig
from .data.io import DataBundle
from .gates import evaluate_gates
from .genomics.krr import add_intercept, make_genomic_regressor
from .genomics.preprocessing import GenotypeStandardizer
from .geometry.fit import fit_population_m4, reconstruct_m4
from .splitting import family_aware_split
from .statistics.bootstrap import (
    bootstrap_eta,
    bootstrap_incremental_r2,
    bootstrap_one_sided_p,
    percentile_ci,
)
from .statistics.metrics import aggregate_r2, incremental_r2, nrmse
from .statistics.multiplicity import holm_adjust


def _ols_fit_predict(C_train: np.ndarray, y_train: np.ndarray, C_test: np.ndarray) -> np.ndarray:
    Ct = add_intercept(C_train)
    Ce = add_intercept(C_test)
    beta = np.linalg.pinv(Ct) @ np.asarray(y_train, float)
    return Ce @ beta


def _fit_genomic(Xtr, ytr, Ctr, Xte, Cte, cfg: RunConfig):
    m = cfg.model
    reg = make_genomic_regressor(
        len(Xtr),
        m.lambda_,
        m.solver,
        m.exact_threshold,
        m.nystrom_components,
        m.seed,
    )
    reg.fit(Xtr, ytr, Ctr)
    return reg.predict(Xte, Cte), reg


def run_confirmatory(bundle: DataBundle, cfg: RunConfig | None = None) -> dict:
    cfg = cfg or RunConfig()
    bundle.validate()
    labels = family_aware_split(bundle.family_ids)
    idx_train = np.flatnonzero(labels == "train")
    idx_internal = np.flatnonzero(labels == "internal")
    idx_external = np.flatnonzero(labels == "external")
    if min(len(idx_train), len(idx_internal), len(idx_external)) < 20:
        raise ValueError("split too small; increase sample size")

    tr, it, ex = bundle.subset(idx_train), bundle.subset(idx_internal), bundle.subset(idx_external)
    gs = GenotypeStandardizer().fit(tr.G)
    Xtr, Xit, Xex = gs.transform(tr.G), gs.transform(it.G), gs.transform(ex.G)

    theta_tr = fit_population_m4(tr.angles, tr.radii)
    theta_ex = fit_population_m4(ex.angles, ex.radii)
    k = theta_tr.shape[1]

    pred_cov_ex = _ols_fit_predict(tr.C, tr.radii, ex.C)

    pred_direct_it, _ = _fit_genomic(Xtr, tr.radii, tr.C, Xit, it.C, cfg)
    pred_direct_ex, _ = _fit_genomic(Xtr, tr.radii, tr.C, Xex, ex.C, cfg)

    pred_theta_ex, _ = _fit_genomic(Xtr, theta_tr, tr.C, Xex, ex.C, cfg)
    pred_mgc_ex = reconstruct_m4(tr.angles, pred_theta_ex)

    pca = PCA(n_components=k, svd_solver="full").fit(tr.radii)
    pca_tr = pca.transform(tr.radii)
    pred_pca_scores_ex, _ = _fit_genomic(Xtr, pca_tr, tr.C, Xex, ex.C, cfg)
    pred_pca_ex = pca.inverse_transform(pred_pca_scores_ex)

    B = cfg.model.bootstrap
    seed = cfg.model.seed
    d_direct = incremental_r2(ex.radii, pred_direct_ex, pred_cov_ex)
    d_mgc = incremental_r2(ex.radii, pred_mgc_ex, pred_cov_ex)
    b_direct = bootstrap_incremental_r2(
        ex.radii,
        pred_direct_ex,
        pred_cov_ex,
        B=B,
        seed=seed,
    )
    direct_ci = percentile_ci(b_direct)
    eta = d_mgc / d_direct if d_direct > 0 else float("nan")
    b_eta = bootstrap_eta(
        ex.radii,
        pred_cov_ex,
        pred_direct_ex,
        pred_mgc_ex,
        B=B,
        seed=seed + 1,
    )
    eta_ci = percentile_ci(b_eta)

    class_rows = []
    pvals = []
    for j, cls in enumerate(np.unique(ex.landmark_classes)):
        cols = ex.landmark_classes == cls
        d = incremental_r2(
            ex.radii[:, cols],
            pred_direct_ex[:, cols],
            pred_cov_ex[:, cols],
        )
        bs = bootstrap_incremental_r2(
            ex.radii[:, cols],
            pred_direct_ex[:, cols],
            pred_cov_ex[:, cols],
            B=B,
            seed=seed + 10 + j,
        )
        p = bootstrap_one_sided_p(bs)
        pvals.append(p)
        class_rows.append({"class": str(cls), "delta_r2": d, "p": p})
    padj = holm_adjust(pvals)
    for row, pa in zip(class_rows, padj):
        row["p_adjusted"] = float(pa)
        row["pass"] = bool(
            row["delta_r2"] >= cfg.gates.g1_class_delta_r2 and pa < cfg.gates.alpha
        )

    pred_theta_cov_ex = _ols_fit_predict(tr.C, theta_tr, ex.C)
    theta_rows, theta_p = [], []
    for j in range(k):
        d = incremental_r2(
            theta_ex[:, j],
            pred_theta_ex[:, j],
            pred_theta_cov_ex[:, j],
        )
        bs = bootstrap_incremental_r2(
            theta_ex[:, j],
            pred_theta_ex[:, j],
            pred_theta_cov_ex[:, j],
            B=B,
            seed=seed + 100 + j,
        )
        p = bootstrap_one_sided_p(bs)
        theta_p.append(p)
        theta_rows.append({"theta": j, "delta_r2": d, "p": p})
    theta_adj = holm_adjust(theta_p)
    for row, pa in zip(theta_rows, theta_adj):
        row["p_adjusted"] = float(pa)
        row["pass"] = bool(
            row["delta_r2"] >= cfg.gates.g2_delta_r2 and pa < cfg.gates.alpha
        )

    metrics = {
        "n_train": len(idx_train),
        "n_internal": len(idx_internal),
        "n_external": len(idx_external),
        "k": k,
        "r2_cov": aggregate_r2(ex.radii, pred_cov_ex),
        "r2_direct": aggregate_r2(ex.radii, pred_direct_ex),
        "r2_mgc": aggregate_r2(ex.radii, pred_mgc_ex),
        "r2_pca": aggregate_r2(ex.radii, pred_pca_ex),
        "dR2_direct": d_direct,
        "dR2_mgc": d_mgc,
        "dR2_direct_ci_low": direct_ci[0],
        "dR2_direct_ci_high": direct_ci[1],
        "eta": eta,
        "eta_ci_low": eta_ci[0],
        "eta_ci_high": eta_ci[1],
        "nrmse_cov": nrmse(ex.radii, pred_cov_ex),
        "nrmse_direct": nrmse(ex.radii, pred_direct_ex),
        "nrmse_mgc": nrmse(ex.radii, pred_mgc_ex),
        "nrmse_pca": nrmse(ex.radii, pred_pca_ex),
        "g1_class_pass_count": int(sum(x["pass"] for x in class_rows)),
        "g2_theta_pass_count": int(sum(x["pass"] for x in theta_rows)),
        "theta_multivariate_positive": bool(aggregate_r2(theta_ex, pred_theta_ex) > 0),
        "internal_nrmse_direct": nrmse(it.radii, pred_direct_it),
        "external_nrmse_direct": nrmse(ex.radii, pred_direct_ex),
    }
    transfer_ratio = (
        metrics["external_nrmse_direct"] / metrics["internal_nrmse_direct"]
        if metrics["internal_nrmse_direct"] > 0
        else float("inf")
    )
    external_pass = bool(np.isfinite(transfer_ratio))
    decision = evaluate_gates(
        metrics,
        cfg.gates,
        external_pass=external_pass,
        qc=True,
        no_leakage=True,
    )
    return {
        "framework": "G2M-RF v2.0",
        "config": {"gates": asdict(cfg.gates), "model": asdict(cfg.model)},
        "metrics": metrics,
        "transfer_ratio": transfer_ratio,
        "landmark_classes": class_rows,
        "theta_tests": theta_rows,
        "decision": decision.to_dict(),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_report(report: dict, out: str | Path) -> Path:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(report)
    p.write_text(
        json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return p
