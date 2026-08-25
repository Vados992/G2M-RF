from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class GateConfig:
    g1_delta_r2: float = 0.02
    g1_class_delta_r2: float = 0.01
    g2_delta_r2: float = 0.01
    g3_nrmse_gain: float = 0.01
    g4_nrmse_gain: float = 0.01
    g5_eta: float = 0.90
    g5_eta_ci_lower: float = 0.75
    g5_direct_floor: float = 0.05
    alpha: float = 0.05
    max_k: int = 3


@dataclass
class ModelConfig:
    lambda_: float = 1.0
    solver: str = "auto"  # auto | exact | nystrom
    exact_threshold: int = 2500
    nystrom_components: int = 512
    pca_k: int = 2
    bootstrap: int = 300
    seed: int = 1601001


@dataclass
class RunConfig:
    gates: GateConfig = field(default_factory=GateConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            gates=GateConfig(**raw.get("gates", {})),
            model=ModelConfig(**raw.get("model", {})),
        )
