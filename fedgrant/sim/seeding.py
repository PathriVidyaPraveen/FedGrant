"""Reproducible random seed derivation for round-based network simulations."""

from __future__ import annotations

import numpy as np


def network_round_seed(master_seed: int, round_id: int) -> int:
    """Return a deterministic seed unique to a (master seed, round) pair.

    Each simulation round creates child streams for traces, losses, readiness,
    exploration, and cross traffic from this seed. Reusing the master seed on
    every round repeats those streams; this derivation preserves whole-run
    reproducibility while changing the streams between rounds.
    """
    if round_id < 1:
        raise ValueError("round_id must be >= 1")
    return int(np.random.SeedSequence([int(master_seed), int(round_id) - 1]).generate_state(1)[0])


__all__ = ["network_round_seed"]
