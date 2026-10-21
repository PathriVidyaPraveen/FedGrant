"""P-EWMA, the non-DL baseline predictor (Section 4.6).

p_hat_i <- (1 - 0.1) p_hat_i + 0.1 y at every resolution of client i; initial 0.9;
persistent across rounds.
"""

from __future__ import annotations

import numpy as np

from fedgrant.core.estimators import check_y
from fedgrant.predictors.base import PredInput, RoundCtx

PEWMA_WEIGHT: float = 0.1
PEWMA_INIT: float = 0.9


class PEWMA:
    """Per-client EWMA of the outcome y (implements Predictor)."""

    name = "ewma"

    def __init__(self, weight: float = PEWMA_WEIGHT, init: float = PEWMA_INIT) -> None:
        if not 0.0 < weight <= 1.0 or not 0.0 <= init <= 1.0:
            raise ValueError("need 0 < weight <= 1 and 0 <= init <= 1")
        self.weight = float(weight)
        self.init = float(init)
        self._p: dict[int, float] = {}

    def reset_round(self, ctx: RoundCtx) -> None:
        """No per-round state: the estimate persists across rounds."""

    def observe(self, client: int, y: int, t_ns: int, obs: np.ndarray) -> None:
        """p <- (1 - w) p + w y."""
        c, yy = int(client), check_y(y)
        self._p[c] = (1.0 - self.weight) * self._p.get(c, self.init) + self.weight * yy

    def value(self, client: int) -> float:
        """Current estimate of one client (init if never observed)."""
        return self._p.get(int(client), self.init)

    def predict(self, inp: PredInput) -> np.ndarray:
        """Stored p_hat for each requested client, float32 [n]."""
        get, init = self._p.get, self.init
        return np.fromiter((get(int(c), init) for c in inp.client_ids), dtype=np.float32, count=inp.n)