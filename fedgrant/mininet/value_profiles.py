"""Client updates for network-only Mininet runs from the real value-profile bank.

Specification Section 5.2: network-only runs draw, for each (round, client), a
FULL round index uniformly from 1-300 and use that client's stored, sorted
energy profile. ``ValueBankInputs.energies_for_run_round`` implements the draw.

A network-only client needs a DATA payload whose block energies equal that
profile. The payload is built with ``profile_update.synthetic_update`` (block j
carries the j-th stored energy) and the payload seed agreed in the runner's
integration contract (docs/MININET_INTEGRATION_CONTRACT.md, Section 3):
``SeedSequence([master_seed, round_id - 1, 2, client_id]).generate_state(1)[0]``.
The seed only fixes the direction of each block's values; block energies, and
therefore the values the gateway schedules on, do not depend on it.

There is deliberately no synthetic-bank fallback: ``from_bank`` raises when
the real bank, its provenance or its client count is unavailable. Fixture
banks may be injected only by tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fedgrant.mininet.profile_update import synthetic_update
from fedgrant.policy.constants import D_MODEL, L as L_DEFAULT
from fedgrant.sim.value_bank import ValueBankInputs, load_value_bank

PAYLOAD_SEED_STREAM = 2  # contract Section 3: SeedSequence([seed, r - 1, 2, i])


class ValueProfileError(RuntimeError):
    """The real value-profile bank cannot serve this client."""


def payload_seed(master_seed: int, client_id: int, round_id: int) -> int:
    if round_id < 1:
        raise ValueError("round_id must be >= 1")
    return int(np.random.SeedSequence(
        [int(master_seed), int(round_id) - 1, PAYLOAD_SEED_STREAM, int(client_id)]
    ).generate_state(1)[0])


@dataclass(frozen=True)
class BankUpdateProvider:
    """``update_provider(round_id) -> float32[d]`` for one network-only client."""

    bank: ValueBankInputs
    master_seed: int
    client_id: int
    d: int = D_MODEL
    L: int = L_DEFAULT
    source_path: str | None = None

    def __post_init__(self) -> None:
        n = self.bank.client_weights.size
        if isinstance(self.client_id, bool) or not 0 <= int(self.client_id) < n:
            raise ValueProfileError(f"client_id {self.client_id} is outside the bank's {n} clients")
        if isinstance(self.master_seed, bool) or int(self.master_seed) < 0:
            raise ValueProfileError("master_seed must be a non-negative integer")

    @classmethod
    def from_bank(cls, path: str | Path, num_clients: int, master_seed: int, client_id: int,
                  *, d: int = D_MODEL, L: int = L_DEFAULT) -> BankUpdateProvider:
        try:
            bank = load_value_bank(path, num_clients)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueProfileError(
                f"real FEMNIST value bank unavailable for N={num_clients}: {exc}"
            ) from exc
        return cls(bank, int(master_seed), int(client_id), d, L, str(path))

    @property
    def weight(self) -> float:
        return float(self.bank.client_weights[self.client_id])

    def energies(self, round_id: int) -> np.ndarray:
        """This client's stored, descending FULL energy profile for the round."""
        return self.bank.energies_for_run_round(self.master_seed, round_id)[self.client_id]

    def __call__(self, round_id: int) -> np.ndarray:
        return synthetic_update(
            self.energies(round_id),
            seed=payload_seed(self.master_seed, self.client_id, round_id),
            d=self.d,
            block_length=self.L,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "femnist_value_bank",
            "bank_path": self.source_path,
            "bank_sha256": self.bank.sha256,
            "provenance_sha256": self.bank.provenance_sha256,
            "client_id": int(self.client_id),
            "master_seed": int(self.master_seed),
            "profile_draw": "energies_for_run_round(master_seed, round_id)[client_id]",
            "payload": "synthetic_update; block j <- j-th stored energy; "
                       "seed = SeedSequence([master_seed, round_id-1, 2, client_id]).generate_state(1)[0]",
        }


__all__ = ["BankUpdateProvider", "ValueProfileError", "payload_seed"]
