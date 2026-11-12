"""Gilbert-Elliott 2-state Markov channel loss model (Sections 3.5, 5.4)."""

from __future__ import annotations

import numpy as np


class GilbertElliottModel:
    """2-state Markov chain modeling bursty wireless frame loss.

    States:
        0: GOOD - Frame loss probability is 0.
        1: BAD  - Frame loss probability is 1.

    Transitions (netem loss gemodel p r 100% 0%):
        p: Probability of transitioning from GOOD to BAD.
        r: Probability of transitioning from BAD to GOOD.

    Stationary distribution:
        pi = p / (p + r) (average loss rate).
        mean burst length = 1 / r.
    """

    # Section 5.4 scenario parameters
    L1_PARAMS = {"p": 0.05, "r": 0.95}       # Bernoulli 5%
    L2_PARAMS = {"p": 0.006579, "r": 0.125}   # GE pi=5%, burst 8
    L3_PARAMS = {"p": 0.001645, "r": 0.03125} # GE pi=5%, burst 32

    def __init__(self, p: float, r: float, seed: int | None = None) -> None:
        if not (0.0 <= p <= 1.0) or not (0.0 <= r <= 1.0):
            raise ValueError(f"Transition probabilities p and r must be in [0, 1], got p={p}, r={r}")
        self.p = float(p)
        self.r = float(r)
        self.state: int = 0  # Start in GOOD state
        self.rng: np.random.Generator = np.random.default_rng(seed)

    @property
    def pi(self) -> float:
        """Theoretical average loss rate pi = p / (p + r)."""
        if self.p + self.r == 0.0:
            return 0.0
        return self.p / (self.p + self.r)

    @property
    def mean_burst_length(self) -> float:
        """Theoretical mean consecutive bad frames 1 / r."""
        if self.r == 0.0:
            return float("inf")
        return 1.0 / self.r

    def should_drop(self) -> bool:
        """Advance channel state by one frame step and return True if frame is dropped.

        The chain advances once per packet presented to the link (Section 3.5).
        """
        if self.state == 0:
            if self.rng.random() < self.p:
                self.state = 1
        else:
            if self.rng.random() < self.r:
                self.state = 0

        return self.state == 1

    def prob_next_good(self) -> float:
        """Pr[next state = GOOD | current state] for P-ORACLE (Section 4.6)."""
        return (1.0 - self.p) if self.state == 0 else self.r

    def reset(self, seed: int | None = None) -> None:
        """Reset channel state to GOOD and reseed RNG if provided."""
        self.state = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    @classmethod
    def for_scenario(cls, scenario: str, seed: int | None = None) -> GilbertElliottModel:
        """Factory for scenarios L1, L2, L3 (and L4/L5 which use L2 parameters)."""
        s = scenario.upper()
        if s == "L1":
            return cls(**cls.L1_PARAMS, seed=seed)
        elif s in ("L2", "L4", "L5", "X2"):
            return cls(**cls.L2_PARAMS, seed=seed)
        elif s == "L3":
            return cls(**cls.L3_PARAMS, seed=seed)
        raise ValueError(f"Unknown loss scenario '{scenario}'; must be L1, L2, L3, L4, L5, or X2")

    @classmethod
    def for_loss_burst(cls, pi: float, mean_burst_length: float, seed: int | None = None) -> GilbertElliottModel:
        """Build a GE chain from stationary loss rate and mean bad-state run length.

        For stationary loss rate ``pi = p / (p + r)`` and mean burst length
        ``1 / r``, the corresponding transitions are ``r = 1 / burst`` and
        ``p = pi * r / (1 - pi)``.
        """
        loss = float(pi)
        burst = float(mean_burst_length)
        if not np.isfinite(loss) or not 0.0 <= loss < 1.0:
            raise ValueError("pi must be finite and in [0, 1)")
        if not np.isfinite(burst) or burst < 1.0:
            raise ValueError("mean_burst_length must be finite and >= 1")
        r = 1.0 / burst
        p = loss * r / (1.0 - loss)
        return cls(p=p, r=r, seed=seed)


__all__ = ["GilbertElliottModel"]
