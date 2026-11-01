"""Deterministic synthetic updates with prescribed block-energy profiles."""

from __future__ import annotations

import numpy as np

from fedgrant.encoder.blocks import n_blocks
from fedgrant.policy.constants import D_MODEL, L


def synthetic_update(energies: np.ndarray, seed: int = 0, *, d: int = D_MODEL, block_length: int = L) -> np.ndarray:
    """Build a float32 update whose unquantized block energies match ``energies``.

    Values are distributed over each block with a seeded random unit vector;
    the short final block uses only its real coordinates.
    """
    energy = np.asarray(energies, dtype=np.float64)
    K = n_blocks(d, block_length)
    if energy.shape != (K,) or not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError(f"energies must be a finite nonnegative vector of shape ({K},)")
    rng = np.random.default_rng(seed)
    out = np.zeros(d, dtype=np.float32)
    for block in range(K):
        start = block * block_length
        stop = min(d, start + block_length)
        vector = rng.standard_normal(stop - start)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            vector[0] = 1.0
            norm = 1.0
        out[start:stop] = (vector * (np.sqrt(energy[block]) / norm)).astype(np.float32)
    return out


def profile_bank(seed: int = 0, *, d: int = D_MODEL, block_length: int = L) -> np.ndarray:
    """Return a small deterministic bank of descending synthetic energy profiles."""
    K = n_blocks(d, block_length)
    base = np.linspace(1.0, 0.01, K, dtype=np.float32)
    rng = np.random.default_rng(seed)
    # A per-coordinate multiplier can invert nearby entries and produce an
    # invalid READY. Scale each descending profile uniformly instead.
    scales = rng.uniform(0.95, 1.05, size=(8, 1)).astype(np.float32)
    return (scales * base[None, :]).astype(np.float32)


__all__ = ["synthetic_update", "profile_bank"]
