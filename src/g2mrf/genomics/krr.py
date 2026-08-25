from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve

from .kernels import linear_kernel


def _validate_lambda(lambda_: float) -> float:
    value = float(lambda_)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("lambda_ must be finite and strictly positive")
    return value


def _finite_2d(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def _response(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, float)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2 or out.shape[0] == 0:
        raise ValueError("y must be a non-empty vector or 2-D response matrix")
    if not np.all(np.isfinite(out)):
        raise ValueError("y contains NaN/Inf")
    return out


def add_intercept(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, float)
    if C.ndim == 1:
        C = C[:, None]
    if C.ndim != 2:
        raise ValueError("C must be a vector or 2-D matrix")
    if not np.all(np.isfinite(C)):
        raise ValueError("C contains NaN/Inf")
    if C.size == 0:
        return np.ones((C.shape[0], 1))
    if np.allclose(C[:, 0], 1.0):
        return C
    return np.column_stack([np.ones(C.shape[0]), C])


@dataclass
class FixedEffectKRR:
    lambda_: float = 1.0
    alpha_: np.ndarray | None = None
    gamma_: np.ndarray | None = None

    def fit(self, K: np.ndarray, y: np.ndarray, C: np.ndarray) -> "FixedEffectKRR":
        self.lambda_ = _validate_lambda(self.lambda_)
        K = _finite_2d("K", K)
        y = _response(y)
        C = add_intercept(C)
        n = K.shape[0]
        if K.shape != (n, n) or y.shape[0] != n or C.shape[0] != n:
            raise ValueError("shape mismatch")
        if not np.allclose(K, K.T, rtol=1e-10, atol=1e-12):
            raise ValueError("training kernel must be symmetric")
        A = np.block(
            [
                [K + self.lambda_ * np.eye(n), C],
                [C.T, np.zeros((C.shape[1], C.shape[1]))],
            ]
        )
        rhs = np.vstack([y, np.zeros((C.shape[1], y.shape[1]))])
        try:
            sol = solve(A, rhs, assume_a="sym")
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        self.alpha_ = sol[:n]
        self.gamma_ = sol[n:]
        return self

    def predict(self, K_cross: np.ndarray, C: np.ndarray) -> np.ndarray:
        if self.alpha_ is None or self.gamma_ is None:
            raise RuntimeError("fit first")
        K_cross = _finite_2d("K_cross", K_cross)
        C = add_intercept(C)
        if K_cross.shape[1] != self.alpha_.shape[0] or C.shape[0] != K_cross.shape[0]:
            raise ValueError("prediction shape mismatch")
        if C.shape[1] != self.gamma_.shape[0]:
            raise ValueError("prediction covariate dimension mismatch")
        pred = K_cross @ self.alpha_ + C @ self.gamma_
        return pred[:, 0] if pred.shape[1] == 1 else pred


@dataclass
class ExactLinearKRR:
    lambda_: float = 1.0
    model_: FixedEffectKRR | None = None
    X_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, C: np.ndarray) -> "ExactLinearKRR":
        self.lambda_ = _validate_lambda(self.lambda_)
        self.X_train_ = _finite_2d("X", X)
        K = linear_kernel(self.X_train_)
        self.model_ = FixedEffectKRR(self.lambda_).fit(K, y, C)
        return self

    def predict(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.X_train_ is None:
            raise RuntimeError("fit first")
        X = _finite_2d("X", X)
        return self.model_.predict(linear_kernel(X, self.X_train_), C)


@dataclass
class NystromLinearKRR:
    lambda_: float = 1.0
    n_components: int = 512
    seed: int = 1601001
    landmark_X_: np.ndarray | None = None
    transform_: np.ndarray | None = None
    beta_: np.ndarray | None = None
    gamma_: np.ndarray | None = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        if self.landmark_X_ is None or self.transform_ is None:
            raise RuntimeError("fit first")
        K_nm = linear_kernel(_finite_2d("X", X), self.landmark_X_)
        return K_nm @ self.transform_

    def fit(self, X: np.ndarray, y: np.ndarray, C: np.ndarray) -> "NystromLinearKRR":
        self.lambda_ = _validate_lambda(self.lambda_)
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        X = _finite_2d("X", X)
        y = _response(y)
        C = add_intercept(C)
        if y.shape[0] != X.shape[0] or C.shape[0] != X.shape[0]:
            raise ValueError("shape mismatch")
        rng = np.random.default_rng(self.seed)
        m = min(int(self.n_components), X.shape[0])
        idx = np.sort(rng.choice(X.shape[0], m, replace=False))
        self.landmark_X_ = X[idx].copy()
        K_mm = linear_kernel(self.landmark_X_)
        d, V = np.linalg.eigh((K_mm + K_mm.T) / 2.0)
        keep = d > max(1e-10, float(d.max()) * 1e-10)
        if not np.any(keep):
            raise ValueError("Nyström landmark kernel has no positive eigenvalues")
        self.transform_ = V[:, keep] @ np.diag(1.0 / np.sqrt(d[keep]))
        Z = self._features(X)

        cpinv = np.linalg.pinv(C)
        Zr = Z - C @ (cpinv @ Z)
        yr = y - C @ (cpinv @ y)
        system = Zr.T @ Zr + self.lambda_ * np.eye(Zr.shape[1])
        self.beta_ = np.linalg.solve(system, Zr.T @ yr)
        self.gamma_ = np.linalg.pinv(C.T @ C) @ C.T @ (y - Z @ self.beta_)
        return self

    def predict(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        if self.beta_ is None or self.gamma_ is None:
            raise RuntimeError("fit first")
        Z = self._features(X)
        C = add_intercept(C)
        if C.shape[0] != Z.shape[0] or C.shape[1] != self.gamma_.shape[0]:
            raise ValueError("prediction covariate dimension mismatch")
        pred = Z @ self.beta_ + C @ self.gamma_
        return pred[:, 0] if pred.shape[1] == 1 else pred


def penalized_fwl_ridge(
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    lambda_: float = 1.0,
):
    """Correct penalized FWL: residualize both y and X, then recover unpenalized covariates."""
    lambda_ = _validate_lambda(lambda_)
    X = _finite_2d("X", X)
    y = _response(y)
    C = add_intercept(C)
    if y.shape[0] != X.shape[0] or C.shape[0] != X.shape[0]:
        raise ValueError("shape mismatch")
    P = C @ np.linalg.pinv(C.T @ C) @ C.T
    M = np.eye(C.shape[0]) - P
    Xr = M @ X
    yr = M @ y
    beta = np.linalg.solve(
        Xr.T @ Xr + lambda_ * np.eye(Xr.shape[1]),
        Xr.T @ yr,
    )
    gamma = np.linalg.pinv(C.T @ C) @ C.T @ (y - X @ beta)
    return beta, gamma


def make_genomic_regressor(
    n_train: int,
    lambda_: float,
    solver: str = "auto",
    exact_threshold: int = 2500,
    nystrom_components: int = 512,
    seed: int = 1601001,
):
    if solver not in {"auto", "exact", "nystrom"}:
        raise ValueError("solver must be auto/exact/nystrom")
    if not isinstance(n_train, (int, np.integer)) or n_train <= 0:
        raise ValueError("n_train must be a positive integer")
    _validate_lambda(lambda_)
    if not isinstance(exact_threshold, (int, np.integer)) or exact_threshold <= 0:
        raise ValueError("exact_threshold must be a positive integer")
    if not isinstance(nystrom_components, (int, np.integer)) or nystrom_components <= 0:
        raise ValueError("nystrom_components must be a positive integer")
    chosen = (
        "exact"
        if solver == "exact" or (solver == "auto" and n_train <= exact_threshold)
        else "nystrom"
    )
    if chosen == "exact":
        return ExactLinearKRR(lambda_=lambda_)
    return NystromLinearKRR(
        lambda_=lambda_,
        n_components=nystrom_components,
        seed=seed,
    )
