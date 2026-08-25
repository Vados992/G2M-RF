from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GenotypeStandardizer:
    p_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    keep_: np.ndarray | None = None

    def fit(self, G: np.ndarray) -> "GenotypeStandardizer":
        G = np.asarray(G, float)
        if G.ndim != 2 or G.shape[0] == 0 or G.shape[1] == 0:
            raise ValueError("G must be a non-empty n x m matrix")
        finite = np.isfinite(G)
        if np.any(finite & ((G < 0.0) | (G > 2.0))):
            raise ValueError("genotype dosages must lie in [0, 2] or be NaN")
        p = np.nanmean(G, axis=0) / 2.0
        scale = np.sqrt(2.0 * p * (1.0 - p))
        observed_sd = np.nanstd(G, axis=0)
        keep = np.isfinite(p) & (scale > 1e-8) & (observed_sd > 1e-8) & (p > 0.0) & (p < 1.0)
        if not np.any(keep):
            raise ValueError("No polymorphic variants remain")
        self.p_ = p[keep]
        self.scale_ = scale[keep]
        self.keep_ = keep
        return self

    def transform(self, G: np.ndarray) -> np.ndarray:
        if self.p_ is None or self.keep_ is None or self.scale_ is None:
            raise RuntimeError("fit first")
        G = np.asarray(G, float)
        if G.ndim != 2 or G.shape[1] != self.keep_.size:
            raise ValueError("G must be 2-D with the same variant count used at fit")
        finite = np.isfinite(G)
        if np.any(finite & ((G < 0.0) | (G > 2.0))):
            raise ValueError("genotype dosages must lie in [0, 2] or be NaN")
        G = G[:, self.keep_].copy()
        means = 2.0 * self.p_
        miss = ~np.isfinite(G)
        if miss.any():
            G[miss] = np.broadcast_to(means, G.shape)[miss]
        return (G - means) / self.scale_

    def fit_transform(self, G: np.ndarray) -> np.ndarray:
        return self.fit(G).transform(G)
