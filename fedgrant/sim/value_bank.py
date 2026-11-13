"""Validated access to the FULL run's client weights and block-energy profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fedgrant.policy.constants import K as K_DEFAULT


@dataclass(frozen=True, slots=True)
class ValueBankInputs:
    energies_sorted: np.ndarray
    client_weights: np.ndarray
    sha256: str
    provenance_sha256: str

    def energies_for_round(self, round_id: int) -> np.ndarray:
        """Return the matching FULL bank profile for one-based round IDs."""
        if not 1 <= round_id <= self.energies_sorted.shape[0]:
            raise ValueError(
                f"round_id {round_id} is outside the value bank range "
                f"[1, {self.energies_sorted.shape[0]}]"
            )
        return self.energies_sorted[round_id - 1].copy()

    def energies_for_run_round(self, master_seed: int, round_id: int) -> np.ndarray:
        """Draw one independent uniform FULL profile index per client and round."""
        if round_id < 1:
            raise ValueError("round_id must be >= 1")
        rng = np.random.default_rng(
            np.random.SeedSequence([int(master_seed), int(round_id) - 1, 1])
        )
        indices = rng.integers(0, self.energies_sorted.shape[0], size=self.client_weights.size)
        return self.energies_sorted[indices, np.arange(self.client_weights.size)].copy()


def load_value_bank(path: str | Path, num_clients: int) -> ValueBankInputs:
    bank_path = Path(path)
    if not bank_path.is_file():
        raise FileNotFoundError(f"FULL value bank not found: {bank_path}")
    digest = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    with np.load(bank_path, allow_pickle=False) as bank:
        required = {
            "energies_sorted", "client_weights", "round_ids", "seed",
            "num_rounds", "dataset_metadata_json",
        }
        missing = required.difference(bank.files)
        if missing:
            raise ValueError(f"value bank missing arrays: {sorted(missing)}")
        energies = np.asarray(bank["energies_sorted"], dtype=np.float32)
        weights = np.asarray(bank["client_weights"], dtype=np.float32)
        round_ids = np.asarray(bank["round_ids"], dtype=np.int64)
        num_rounds = int(bank["num_rounds"].item())
        bank_seed = int(bank["seed"].item())
        try:
            metadata = json.loads(str(bank["dataset_metadata_json"].item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("value bank dataset metadata is invalid JSON") from exc

    if energies.ndim != 3 or energies.shape[1:] != (num_clients, K_DEFAULT):
        raise ValueError(
            f"value bank energies must have shape [rounds,{num_clients},{K_DEFAULT}], "
            f"got {energies.shape}"
        )
    if energies.shape[0] != num_rounds or num_rounds < 1:
        raise ValueError("value bank num_rounds does not match its energy profiles")
    if not np.isfinite(energies).all() or np.any(energies < 0):
        raise ValueError("value bank energies must be finite and non-negative")
    if np.any(np.diff(energies, axis=2) > 0):
        raise ValueError("value bank energies must be sorted descending per client")
    if not np.array_equal(round_ids, np.arange(1, num_rounds + 1)):
        raise ValueError("value bank round_ids must be consecutive and start at 1")
    if weights.shape != (num_clients,) or not np.isfinite(weights).all():
        raise ValueError("value bank must have one finite weight per client")
    if np.any(weights < 0) or not np.isclose(weights.sum(dtype=np.float64), 1.0):
        raise ValueError("value bank client weights must be non-negative and sum to 1")
    provenance = str(metadata.get("provenance_sha256", "")) if isinstance(metadata, dict) else ""
    if not provenance or metadata.get("source") == "unverified":
        raise ValueError("value bank has no verified dataset provenance")
    provenance_path = Path(str(metadata.get("provenance_path", "")))
    if not provenance_path.is_file():
        provenance_path = bank_path.parent.parent / "femnist" / "reproduced" / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"value-bank provenance file not found: {provenance_path}")
    if hashlib.sha256(provenance_path.read_bytes()).hexdigest() != provenance:
        raise ValueError("value-bank provenance hash does not match its provenance file")
    manifest_hash = str(metadata.get("client_manifest_sha256", ""))
    manifest_path = Path(str(metadata.get("client_manifest", "")))
    if manifest_hash:
        if not manifest_path.is_file():
            manifest_path = bank_path.parent.parent / "femnist" / "reproduced" / "clients" / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"value-bank client manifest not found: {manifest_path}")
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_hash:
            raise ValueError("value-bank client manifest hash does not match")
    if bank_seed != 0:
        raise ValueError(f"expected seed-0 FULL value bank, got seed {bank_seed}")
    return ValueBankInputs(energies, weights, digest, provenance)


__all__ = ["ValueBankInputs", "load_value_bank"]
