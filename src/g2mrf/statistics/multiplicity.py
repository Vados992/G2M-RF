from __future__ import annotations

import numpy as np


def holm_adjust(pvalues: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, float)
    m = p.size
    order = np.argsort(p)
    adj_sorted = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj_sorted[rank] = running
    out = np.empty(m, float)
    out[order] = adj_sorted
    return out
