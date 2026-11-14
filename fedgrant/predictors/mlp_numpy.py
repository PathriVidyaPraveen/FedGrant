"""P-MLP, NumPy float32 inference from models/mlp_prod.npz (Sections 4.3, 4.6).

48 -> 64 -> 64 -> 1 with ReLU, sigmoid output, on the standardized 48-feature
vector of PredInput. No PyTorch dependency. observe() is a no-op: every input
P-MLP needs is already in x48 (built from ClientState by policy/features.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from fedgrant.ml.export import load_prod
from fedgrant.predictors.base import PredInput, RoundCtx


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """sigmoid(x) = 0.5 * (1 + tanh(x / 2)): no overflow for any x, keeps the dtype of x."""
    y = np.multiply(x, 0.5, dtype=x.dtype)
    np.tanh(y, out=y)
    y += 1.0
    y *= 0.5
    return y


def standardize(x: np.ndarray, mu: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(x - mu) / s computed in float64 and rounded once to float32 (same as training)."""
    return ((np.asarray(x, np.float64) - mu) / s).astype(np.float32)


def _weights(src: str | Path | Mapping[str, np.ndarray], model: str) -> Mapping[str, np.ndarray]:
    d = load_prod(src) if isinstance(src, (str, Path)) else src
    if str(d["model"]) != model:
        raise ValueError(f"expected a {model!r} export, got {str(d['model'])!r}")
    return d


class PMLPNumpy:
    """P-MLP predictor (implements Predictor)."""

    name = "mlp"

    def __init__(self, npz: str | Path | Mapping[str, np.ndarray]) -> None:
        d = _weights(npz, "mlp")
        self.w1t = np.ascontiguousarray(d["w1"].T, dtype=np.float32)       # [48, 64]
        self.b1 = np.asarray(d["b1"], np.float32)
        self.w2t = np.ascontiguousarray(d["w2"].T, dtype=np.float32)       # [64, 64]
        self.b2 = np.asarray(d["b2"], np.float32)
        self.w3t = np.ascontiguousarray(d["w3"].T, dtype=np.float32)       # [64, 1]
        self.b3 = np.asarray(d["b3"], np.float32)
        self.mu = np.asarray(d["mu_x48"], np.float64)
        self.s = np.asarray(d["s_x48"], np.float64)
        self.seed = int(d["seed"]) if "seed" in d else -1
        # Warm up NumPy/BLAS once here: the first matmul of a process can take tens
        # of ms and would trip the 2 ms wall-clock fallback (Section 9) in round 1.
        stable_sigmoid(self.logits(np.zeros((2, 48), np.float32)))

    def reset_round(self, ctx: RoundCtx) -> None:
        """No state."""

    def observe(self, client: int, y: int, t_ns: int, obs: np.ndarray) -> None:
        """No-op: P-MLP reads only x48."""

    def logits(self, x48: np.ndarray) -> np.ndarray:
        """Logits float32 [n] for raw (unstandardized) x48 [n, 48]."""
        x = standardize(x48, self.mu, self.s)
        a = np.maximum(x @ self.w1t + self.b1, 0.0)
        a = np.maximum(a @ self.w2t + self.b2, 0.0)
        return (a @ self.w3t + self.b3)[:, 0]

    def predict(self, inp: PredInput) -> np.ndarray:
        """p_hat float32 [inp.n] in [0, 1]."""
        if inp.n == 0:
            return np.zeros(0, np.float32)
        return stable_sigmoid(self.logits(inp.x48))