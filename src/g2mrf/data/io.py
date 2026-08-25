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

    def subset(self, idx: np.ndarray) -> "DataBundle":
        idx = np.asarray(idx)
        return DataBundle(self.sample_ids[idx], self.family_ids[idx], self.G[idx], self.C[idx], self.radii[idx], self.angles.copy(), self.landmark_classes.copy())


def save_npz(bundle: DataBundle, path: str | Path) -> None:
    np.savez_compressed(path, sample_ids=bundle.sample_ids, family_ids=bundle.family_ids, G=bundle.G, C=bundle.C, radii=bundle.radii, angles=bundle.angles, landmark_classes=bundle.landmark_classes)


def load_npz(path: str | Path) -> DataBundle:
    d = np.load(path, allow_pickle=False)
    return DataBundle(d["sample_ids"], d["family_ids"], d["G"], d["C"], d["radii"], d["angles"], d["landmark_classes"])
