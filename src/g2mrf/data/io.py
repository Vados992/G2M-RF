from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DataBundle:
    sample_ids: np.ndarray
    family_ids: np.ndarray
    G: np.ndarray
    C: np.ndarray
    radii: np.ndarray
    angles: np.ndarray
    landmark_classes: np.ndarray

    def validate(self) -> "DataBundle":
        self.sample_ids = np.asarray(self.sample_ids)
        self.family_ids = np.asarray(self.family_ids)
        self.G = np.asarray(self.G, float)
        self.C = np.asarray(self.C, float)
        self.radii = np.asarray(self.radii, float)
        self.angles = np.asarray(self.angles, float)
        self.landmark_classes = np.asarray(self.landmark_classes)

        if self.sample_ids.ndim != 1 or self.sample_ids.size == 0:
            raise ValueError("sample_ids must be a non-empty 1-D array")
        n = self.sample_ids.size
        if np.unique(self.sample_ids.astype(str)).size != n:
            raise ValueError("sample_ids must be unique")
        if self.family_ids.ndim != 1 or self.family_ids.size != n:
            raise ValueError("family_ids must be 1-D and match sample count")
        if self.G.ndim != 2 or self.G.shape[0] != n or self.G.shape[1] == 0:
            raise ValueError("G must be a non-empty N x M matrix")
        if self.C.ndim != 2 or self.C.shape[0] != n:
            raise ValueError("C must be an N x Q matrix")
        if self.radii.ndim != 2 or self.radii.shape[0] != n or self.radii.shape[1] == 0:
            raise ValueError("radii must be a non-empty N x L matrix")
        landmarks = self.radii.shape[1]
        if self.angles.ndim != 1 or self.angles.size != landmarks:
            raise ValueError("angles must be 1-D and match landmark count")
        if self.landmark_classes.ndim != 1 or self.landmark_classes.size != landmarks:
            raise ValueError("landmark_classes must match landmark count")
        if not np.all(np.isfinite(self.C)):
            raise ValueError("covariates contain NaN/Inf")
        if not np.all(np.isfinite(self.radii)) or np.any(self.radii < 0):
            raise ValueError("radii must be finite and nonnegative")
        if not np.all(np.isfinite(self.angles)):
            raise ValueError("angles contain NaN/Inf")
        return self

    def subset(self, idx: np.ndarray) -> "DataBundle":
        idx = np.asarray(idx)
        return DataBundle(
            self.sample_ids[idx],
            self.family_ids[idx],
            self.G[idx],
            self.C[idx],
            self.radii[idx],
            self.angles.copy(),
            self.landmark_classes.copy(),
        )


def save_npz(bundle: DataBundle, path: str | Path) -> None:
    bundle.validate()
    np.savez_compressed(
        path,
        sample_ids=bundle.sample_ids,
        family_ids=bundle.family_ids,
        G=bundle.G,
        C=bundle.C,
        radii=bundle.radii,
        angles=bundle.angles,
        landmark_classes=bundle.landmark_classes,
    )


def load_npz(path: str | Path) -> DataBundle:
    with np.load(path, allow_pickle=False) as d:
        bundle = DataBundle(
            d["sample_ids"],
            d["family_ids"],
            d["G"],
            d["C"],
            d["radii"],
            d["angles"],
            d["landmark_classes"],
        )
    return bundle.validate()
