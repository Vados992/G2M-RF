from __future__ import annotations

import hashlib
import numpy as np


def _u01(text: str) -> float:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def family_aware_split(family_ids: np.ndarray, salt: str = "g2mrf-v2", train=0.70, internal=0.15):
    """Assign complete family/kinship components deterministically to train/internal/external."""
    fam = np.asarray(family_ids).astype(str)
    labels = np.empty(fam.shape[0], dtype="U8")
    for f in np.unique(fam):
        u = _u01(f"{salt}:{f}")
        lab = "train" if u < train else ("internal" if u < train + internal else "external")
        labels[fam == f] = lab
    return labels
