from __future__ import annotations

import hashlib

import numpy as np


def _u01(text: str) -> float:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def family_aware_split(
    family_ids: np.ndarray,
    salt: str = "g2mrf-v2",
    train: float = 0.70,
    internal: float = 0.15,
) -> np.ndarray:
    """Assign complete family/kinship components deterministically to train/internal/external."""
    fam = np.asarray(family_ids)
    if fam.ndim != 1 or fam.size == 0:
        raise ValueError("family_ids must be a non-empty 1-D array")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")
    if not (0.0 < train < 1.0 and 0.0 < internal < 1.0 and train + internal < 1.0):
        raise ValueError("require 0 < train, internal and train + internal < 1")
    fam = fam.astype(str)
    if np.any(np.char.str_len(fam) == 0):
        raise ValueError("family_ids must not contain empty identifiers")

    labels = np.empty(fam.shape[0], dtype="U8")
    for f in np.unique(fam):
        u = _u01(f"{salt}:{f}")
        lab = "train" if u < train else ("internal" if u < train + internal else "external")
        labels[fam == f] = lab
    return labels
