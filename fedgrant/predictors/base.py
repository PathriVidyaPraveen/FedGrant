"""Predictor interface (Section 6.1).

Identical in the simulator and the gateway daemon. A predictor estimates
p_hat_i(t) ~= Pr[y_g = 1 | x_i(t)], Eq. (9), for the label of Eq. (8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from fedgrant.policy.constants import CTX_SLICE, N_CTX, N_X48


@dataclass(frozen=True, slots=True)
class RoundCtx:
    """Per-round context passed to Predictor.reset_round.

    round_id: round number r (from 1, Section 8).
    D_ns: deadline D in ns, relative to round start (Section 3.1).
    N: number of clients. K: blocks per update.
    """

    round_id: int
    D_ns: int
    N: int
    K: int

    def __post_init__(self) -> None:
        for name in ("round_id", "D_ns", "N", "K"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise TypeError(f"RoundCtx.{name} must be an int, got {type(v).__name__}")
            object.__setattr__(self, name, int(v))
        if self.round_id < 1:
            raise ValueError(f"round_id must be >= 1, got {self.round_id}")
        if self.D_ns <= 0:
            raise ValueError(f"D_ns must be > 0, got {self.D_ns}")
        if self.N < 1:
            raise ValueError(f"N must be >= 1, got {self.N}")
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")


@dataclass(frozen=True, slots=True)
class PredInput:
    """Predictor input at time t, built by policy/features.py (Section 6.1).

    client_ids: int32 [n]. x48: float32 [n, 48], the P-MLP features (Section 4.3).
    ctx: float32 [n, 16], equal to x48[:, 32:48] (features 33..48, Section 4.2).
    t_ns: prediction time, ns relative to round start.

    Construct with PredInput.from_x48 to get ctx as a view of x48.
    """

    client_ids: np.ndarray
    x48: np.ndarray
    ctx: np.ndarray
    t_ns: int

    def __post_init__(self) -> None:
        ids, x48, ctx = self.client_ids, self.x48, self.ctx
        if not isinstance(ids, np.ndarray) or ids.dtype != np.int32 or ids.ndim != 1:
            raise TypeError("client_ids must be a 1-D int32 ndarray")
        n = ids.shape[0]
        if not isinstance(x48, np.ndarray) or x48.dtype != np.float32 or x48.shape != (n, N_X48):
            raise TypeError(f"x48 must be a float32 ndarray of shape ({n}, {N_X48})")
        if not isinstance(ctx, np.ndarray) or ctx.dtype != np.float32 or ctx.shape != (n, N_CTX):
            raise TypeError(f"ctx must be a float32 ndarray of shape ({n}, {N_CTX})")
        if not np.array_equal(ctx, x48[:, CTX_SLICE], equal_nan=True):
            raise ValueError("ctx must equal x48[:, 32:48] (Section 6.1)")
        t = self.t_ns
        if isinstance(t, bool) or not isinstance(t, (int, np.integer)):
            raise TypeError("t_ns must be an int (ns)")
        object.__setattr__(self, "t_ns", int(t))

    @classmethod
    def from_x48(cls, client_ids: np.ndarray, x48: np.ndarray, t_ns: int) -> "PredInput":
        """Build a PredInput whose ctx is the view x48[:, 32:48]."""
        ids = np.ascontiguousarray(client_ids, dtype=np.int32)
        x = np.ascontiguousarray(x48, dtype=np.float32)
        return cls(client_ids=ids, x48=x, ctx=x[:, CTX_SLICE], t_ns=t_ns)

    @property
    def n(self) -> int:
        """Number of clients in this request."""
        return int(self.client_ids.shape[0])


@runtime_checkable
class Predictor(Protocol):
    """Delivery predictor (Section 6.1; implementations in Section 4.6)."""

    def reset_round(self, ctx: RoundCtx) -> None:
        """Called at ROUND_START. Estimator state persists across rounds (Table 2)."""
        ...

    def observe(self, client: int, y: int, t_ns: int, obs: np.ndarray) -> None:
        """Record one resolution event of `client`: y in {0,1}, obs float32 [8] (Section 4.2)."""
        ...

    def predict(self, inp: PredInput) -> np.ndarray:
        """Return p_hat as float32 [inp.n], every value in [0, 1] (Eq. 9)."""
        ...


__all__ = ["RoundCtx", "PredInput", "Predictor"]