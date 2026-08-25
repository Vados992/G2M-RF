from __future__ import annotations

import numpy as np
from .io import DataBundle


def make_synthetic_dataset(n: int = 720, variants: int = 500, landmarks: int = 25, seed: int = 1601001) -> DataBundle:
    if landmarks < 15:
        raise ValueError("use at least 15 landmarks so five classes are represented")
    rng = np.random.default_rng(seed)
    maf = rng.uniform(0.05, 0.45, variants)
    G = rng.binomial(2, maf, size=(n, variants)).astype(float)
    age = rng.normal(0, 1, n)
    sex = rng.integers(0, 2, n).astype(float)
    C = np.column_stack([age, sex])

    causal1 = rng.choice(variants, min(40, variants), replace=False)
    causal2 = rng.choice(variants, min(40, variants), replace=False)
    z = (G - 2 * maf) / np.sqrt(2 * maf * (1 - maf))
    s1 = z[:, causal1].mean(axis=1) * np.sqrt(len(causal1))
    s2 = z[:, causal2].mean(axis=1) * np.sqrt(len(causal2))
    a = 0.27 + 0.035 * s1 + 0.010 * sex + rng.normal(0, 0.010, n)
    b = 0.10 + 0.20 * s2 + 0.030 * age + rng.normal(0, 0.045, n)
    b = np.clip(b, -0.55, 0.55)
    a = np.clip(a, 0.10, 0.38)
    a_cap = 0.70 * np.exp(-np.abs(b) * (np.pi / 2.0))
    a = np.minimum(a, a_cap)

    angles = np.linspace(-np.pi / 2, np.pi / 2, landmarks)
    radii = a[:, None] * np.exp(b[:, None] * angles[None, :])
    radii += rng.normal(0, 0.006, size=radii.shape)
    radii = np.clip(radii, 0.001, 0.749)

    classes = np.asarray([f"class_{i % 5 + 1}" for i in range(landmarks)])
    sample_ids = np.asarray([f"S{i:06d}" for i in range(n)])
    family_ids = np.asarray([f"F{i // 2:06d}" if i % 10 < 2 else f"Fsolo{i:06d}" for i in range(n)])
    return DataBundle(sample_ids, family_ids, G, C, radii, angles, classes)
