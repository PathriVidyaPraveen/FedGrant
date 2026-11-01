"""Persist block-energy observations for the value-policy bank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ValueBank:
    """Round-indexed sorted encoder energies and the corresponding client weights."""

    def __init__(self, client_weights: np.ndarray, num_blocks: int) -> None:
        weights = np.asarray(client_weights, dtype=np.float32)
        if weights.ndim != 1 or weights.size == 0 or not np.isfinite(weights).all():
            raise ValueError("client_weights must be a non-empty finite 1-D array")
        if np.any(weights < 0) or not np.isclose(weights.sum(dtype=np.float64), 1.0):
            raise ValueError("client_weights must be non-negative and sum to 1")
        self.client_weights = weights.copy()
        self.num_blocks = int(num_blocks)
        self._energies: list[np.ndarray] = []
        self._round_ids: list[int] = []

    def add_round(self, round_id: int, energies_sorted: np.ndarray) -> None:
        """Record [clients, blocks] sorted block energies from the adjusted updates."""
        e = np.asarray(energies_sorted, dtype=np.float32)
        expected = (self.client_weights.size, self.num_blocks)
        if e.shape != expected or not np.isfinite(e).all() or np.any(e < 0):
            raise ValueError(f"energies_sorted must be finite/non-negative with shape {expected}")
        if np.any(np.diff(e, axis=1) > 0):
            raise ValueError("each client's block energies must be sorted descending")
        if self._round_ids and round_id != self._round_ids[-1] + 1:
            raise ValueError("bank rounds must be consecutive")
        self._round_ids.append(int(round_id))
        self._energies.append(e.copy())

    @property
    def energies_sorted(self) -> np.ndarray:
        if not self._energies:
            return np.empty((0, self.client_weights.size, self.num_blocks), dtype=np.float32)
        return np.stack(self._energies)

    def save(
        self,
        path: str | Path,
        seed: int,
        num_rounds: int,
        dataset_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write a compressed NPZ with explicit array names and run metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            energies_sorted=self.energies_sorted,
            client_weights=self.client_weights,
            round_ids=np.asarray(self._round_ids, dtype=np.int32),
            seed=np.asarray(seed, dtype=np.int64),
            num_rounds=np.asarray(num_rounds, dtype=np.int32),
            dataset_metadata_json=np.asarray(
                json.dumps(dataset_metadata or {}, sort_keys=True)
            ),
        )
        return path


__all__ = ["ValueBank"]
