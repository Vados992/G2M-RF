from __future__ import annotations


def daetwyler_r2(n_train: int, h2: float, me: float) -> float:
    if n_train <= 0 or not (0 < h2 <= 1) or me <= 0:
        raise ValueError("invalid planning parameters")
    return float(h2 / (1.0 + me / (n_train * h2)))


def required_n(target_r2: float, h2: float, me: float) -> float:
    if not (0 < target_r2 < h2 <= 1) or me <= 0:
        raise ValueError("require 0 < target_r2 < h2 <= 1")
    return float(target_r2 * me / (h2 * (h2 - target_r2)))


def planning_grid(ns=(20_000, 50_000, 100_000, 200_000, 500_000), h2=0.35, me=75_000.0):
    return [{"n_train": int(n), "r2_pred": daetwyler_r2(int(n), h2, me)} for n in ns]
