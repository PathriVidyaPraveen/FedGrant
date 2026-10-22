"""Socket-independent FedGrant client state and message payload preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedgrant.common.constants import TransferMode
from fedgrant.encoder.blocks import EncodedUpdate, block_payload_f16, encode, n_blocks
from fedgrant.encoder.ef import ErrorFeedback, indices_from_bitmap
from fedgrant.policy.constants import D_MODEL, L as L_DEFAULT


class ClientCoreError(ValueError):
    """Invalid client configuration, protocol event, or state transition."""


@dataclass(frozen=True, slots=True)
class ClientRound:
    round_id: int
    deadline_ns: int
    encoded: EncodedUpdate
    u: np.ndarray

    @property
    def K(self) -> int:
        return self.encoded.K

    @property
    def L(self) -> int:
        return self.encoded.L


class ClientCore:
    """Prepare READY/DATA contents and apply ROUND_END without owning sockets."""

    def __init__(
        self,
        client_id: int,
        *,
        d: int = D_MODEL,
        L: int = L_DEFAULT,
        error_feedback: bool = True,
    ) -> None:
        if isinstance(client_id, bool) or not isinstance(client_id, int) or not 0 <= client_id <= 0xFFFF:
            raise ClientCoreError("client_id must be an integer in [0, 65535]")
        if isinstance(d, bool) or not isinstance(d, int) or d < 1:
            raise ClientCoreError("d must be a positive integer")
        if isinstance(L, bool) or not isinstance(L, int) or not 1 <= L <= 0xFFFF:
            raise ClientCoreError("L must be an integer in [1, 65535]")
        if not isinstance(error_feedback, bool):
            raise ClientCoreError("error_feedback must be a bool")
        self.client_id = client_id
        self.error_feedback = ErrorFeedback(d, L, enabled=error_feedback)
        self.round: ClientRound | None = None
        self.last_completed_round_id: int | None = None
        self._grant_positions: dict[int, int] = {}

    def start_round(
        self,
        *,
        round_id: int,
        start_ns: int,
        deadline_ns: int,
        K: int,
        L: int,
        mode: TransferMode | int,
        delta: np.ndarray,
    ) -> tuple[float, ...]:
        """Prepare encoded update and return READY energies in send order."""
        if self.round is not None:
            raise ClientCoreError("cannot start a round while another round is active")
        self._uint(round_id, "round_id", 0xFFFFFFFF, minimum=1)
        self._uint(start_ns, "start_ns", 0xFFFFFFFFFFFFFFFF)
        self._uint(deadline_ns, "deadline_ns", 0xFFFFFFFFFFFFFFFF)
        self._uint(K, "K", 0xFFFF, minimum=1)
        self._uint(L, "L", 0xFFFF, minimum=1)
        if deadline_ns <= start_ns:
            raise ClientCoreError("deadline_ns must be after ROUND_START timestamp")
        if round_id <= (self.last_completed_round_id or 0):
            raise ClientCoreError("round_id must increase monotonically")
        try:
            transfer_mode = TransferMode(mode)
        except (TypeError, ValueError) as exc:
            raise ClientCoreError("unsupported transfer mode") from exc
        if transfer_mode is not TransferMode.GRANT:
            raise ClientCoreError("UDP client core supports GRANT mode only")
        if L != self.error_feedback.L or K != n_blocks(self.error_feedback.d, L):
            raise ClientCoreError("round K/L do not match the configured update dimensions")

        u = self.error_feedback.prepare(delta)
        encoded = encode(u, L)
        if encoded.K != K:
            raise ClientCoreError("encoded update does not match round K")
        prepared_round = ClientRound(round_id, deadline_ns, encoded, u.copy())
        energies = tuple(float(value) for value in encoded.energies_sorted)
        self.round = prepared_round
        self._grant_positions.clear()
        return energies

    def data_for_grant(self, round_id: int, grant_id: int, send_pos: int) -> dict[str, object]:
        """Build DATA body only for the exact position requested by a valid grant."""
        current = self._require_round(round_id)
        self._uint(grant_id, "grant_id", 0xFFFFFFFF, minimum=1)
        self._uint(send_pos, "send_pos", 0xFFFF, minimum=1)
        if send_pos > current.K:
            raise ClientCoreError("send_pos is outside this round")
        if grant_id in self._grant_positions:
            raise ClientCoreError("duplicate grant_id")
        block_idx = current.encoded.block_at(send_pos)
        payload, n_coords = block_payload_f16(current.encoded, block_idx)
        self._grant_positions[grant_id] = send_pos
        return {
            "grant_id": grant_id,
            "send_pos": send_pos,
            "block_idx": block_idx,
            "n_coords": n_coords,
            "payload": tuple(float(value) for value in payload),
        }

    def finish_round(self, round_id: int, bitmap: bytes | bytearray | memoryview) -> np.ndarray:
        """Apply the delivered bitmap once and update persistent residual state."""
        current = self._require_round(round_id)
        try:
            indices_from_bitmap(bitmap, current.K)
            reconstructed = self.error_feedback.finish_round(current.u, bitmap)
        except (TypeError, ValueError) as exc:
            raise ClientCoreError(f"invalid ROUND_END bitmap: {exc}") from exc
        self.last_completed_round_id = current.round_id
        self.round = None
        self._grant_positions.clear()
        return reconstructed

    def finish_round_timeout(self, round_id: int) -> np.ndarray:
        """Apply the specification's all-undelivered fallback after D + 200 ms."""
        current = self._require_round(round_id)
        reconstructed = self.error_feedback.on_round_end_timeout(current.u)
        self.last_completed_round_id = current.round_id
        self.round = None
        self._grant_positions.clear()
        return reconstructed

    def _require_round(self, round_id: int) -> ClientRound:
        if self.round is None or self.round.round_id != round_id:
            raise ClientCoreError("message does not match the active round")
        return self.round

    @staticmethod
    def _uint(value: object, name: str, maximum: int, *, minimum: int = 0) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ClientCoreError(f"{name} must be an integer in [{minimum}, {maximum}]")


__all__ = ["ClientCore", "ClientCoreError", "ClientRound"]
