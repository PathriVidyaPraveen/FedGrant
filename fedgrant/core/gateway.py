"""Gateway-core coordination boundary for the FedGrant protocol.

This module owns round lifecycle, protocol-level grant bookkeeping, and the
boundary checks that must happen before ``ClientState.mark_delivered``. Policy,
pacing, predictor, timeout-estimator, feature, and aggregation implementations
are dependencies: this module defines only the contracts it consumes.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from numbers import Real
from typing import Any, Protocol, runtime_checkable

import numpy as np

from fedgrant.common.constants import DATA_GRANT_ID, L as L_DEFAULT, MessageType, TransferMode
from fedgrant.common.wire import WireProtocolError, decode_message
from fedgrant.encoder.ef import bitmap_from_indices
from fedgrant.policy.constants import K as K_DEFAULT
from fedgrant.policy.features import build_pred_input
from fedgrant.policy.types import CandidateView, Selector
from fedgrant.predictors.base import Predictor, RoundCtx
from fedgrant.predictors.fallback import FallbackPredictor
from fedgrant.core.state import ClientState, OutstandingGrant, Status, values_from_energies


class GatewayError(ValueError):
    """Raised for invalid gateway configuration or event arguments."""


class GatewayPhase(IntEnum):
    IDLE = 0
    ACTIVE = 1
    DEADLINE = 2


@dataclass(frozen=True, slots=True)
class RoundContext:
    """Immutable active-round values shared with injected dependencies."""

    round_id: int
    start_ns: int
    deadline_ns: int
    K: int
    L: int
    rho_us: int
    mode: TransferMode
    rho_us_by_client: Mapping[int, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SendRoundStart:
    client_id: int
    round_id: int
    t_send_ns: int
    deadline_ns: int
    K: int
    L: int
    rho_us: int
    mode: TransferMode


@dataclass(frozen=True, slots=True)
class SendGrant:
    client_id: int
    grant_id: int
    send_pos: int


@dataclass(frozen=True, slots=True)
class SendRoundEnd:
    client_id: int
    round_id: int
    bitmap: bytes


@dataclass(frozen=True, slots=True)
class EmitAggregate:
    aggregate: Any
    round_id: int = field(default=0, compare=False)
    client_n_coords: Mapping[int, Mapping[int, int]] = field(default_factory=dict, compare=False)


GatewayAction = SendRoundStart | SendGrant | SendRoundEnd | EmitAggregate


@runtime_checkable
class EligibilityProvider(Protocol):
    """Build the Section 3.9 eligible-client view; implementation is external."""

    def candidates(
        self,
        clients: Mapping[int, ClientState],
        now_ns: int,
        deadline_ns: int,
        retransmit: bool,
    ) -> CandidateView:
        ...


@runtime_checkable
class PacingProvider(Protocol):
    """Apply the Section 3.8 pacing condition; implementation is external."""

    def allow(self, client_id: int, p_hat: float, now_ns: int, deadline_ns: int) -> bool:
        ...


@runtime_checkable
class PredictionProvider(Protocol):
    """Supply the already-implemented predictor result for one client."""

    def probability(self, client_id: int, now_ns: int) -> float:
        ...


@runtime_checkable
class TimeoutProvider(Protocol):
    """Supply an absolute grant timeout timestamp; estimator logic is external."""

    def timeout_ns(self, client_id: int, issue_ns: int) -> int:
        ...


@runtime_checkable
class EstimatorHooks(Protocol):
    """Lifecycle hooks for Praveen-owned estimator/feature implementations."""

    def reset_round(self, ctx: RoundCtx) -> None:
        ...

    def resolve(
        self,
        client_id: int,
        grant: OutstandingGrant,
        delivered: bool,
        t_ns: int,
    ) -> None:
        ...

    def on_tick(self, t_ns: int) -> None:
        ...

    def on_deadline(self, t_ns: int) -> None:
        ...


@runtime_checkable
class AggregateProvider(Protocol):
    """Aggregate from clients' authoritative payload/count maps.

    For each client, ``delivered_payloads[block_idx]`` is a float32 vector of
    negotiated length L and ``delivered_n_coords[block_idx]`` is its true
    coordinate count. Missing blocks contribute zeros; payload arrays include
    protocol padding and consumers must honor the stored count.
    """

    def aggregate(self, clients: Mapping[int, ClientState]) -> Any:
        ...


@runtime_checkable
class ValueIntegrityChecker(Protocol):
    """Check one DATA payload against externally maintained energy metadata."""

    def valid(
        self,
        client_id: int,
        block_idx: int,
        payload: np.ndarray,
        reported_energies: np.ndarray,
        n_coords: int,
    ) -> bool:
        ...


@dataclass(slots=True)
class GatewayStats:
    """Counters for boundary decisions; transport/logging owns persistence."""

    invalid_events: int = 0
    rejected_ready: int = 0
    duplicate_ready: int = 0
    rejected_data: int = 0
    accepted_data: int = 0
    duplicate_data: int = 0
    late_data: int = 0
    late_after_deadline: int = 0
    energy_violations: dict[int, int] = field(default_factory=dict)
    missing_dependencies: int = 0
    rejected_round_start: int = 0
    bad_client_id: int = 0
    late_after_timeout: int = 0


class GatewayCore:
    """Round and grant state machine at the gateway/application boundary.

    ``on_round_start``, ``on_ready``, ``on_data``, ``on_tick``, and
    ``on_deadline`` correspond to the five GatewayCore methods in Section 6.2.
    The caller is responsible for decoding datagrams and sending returned
    actions. No socket, scheduler policy, estimator, or aggregation algorithm
    is implemented here.
    """

    def __init__(
        self,
        clients: Mapping[int, ClientState] | None = None,
        *,
        selector: Selector | None = None,
        eligibility: EligibilityProvider | None = None,
        pacing: PacingProvider | None = None,
        prediction: PredictionProvider | None = None,
        timeout: TimeoutProvider | None = None,
        estimator: EstimatorHooks | None = None,
        predictor: Predictor | None = None,
        aggregate: AggregateProvider | None = None,
        value_integrity: ValueIntegrityChecker | None = None,
        view: Any | None = None,
        retransmit: bool = True,
    ) -> None:
        self.phase = GatewayPhase.IDLE
        self.clients: dict[int, ClientState] = {}
        for client_id, state in (clients or {}).items():
            self.register_client(client_id, state)
        self.selector = selector
        self.eligibility = eligibility
        self.pacing = pacing
        self.prediction = prediction
        self.timeout = timeout
        self.estimator = estimator
        self.predictor = (
            None
            if predictor is None
            else predictor if isinstance(predictor, FallbackPredictor) else FallbackPredictor(predictor)
        )
        self.aggregate = aggregate
        self.value_integrity = value_integrity
        self.view = view
        if not isinstance(retransmit, bool):
            raise TypeError("retransmit must be a bool")
        self.retransmit = retransmit
        self.round: RoundContext | None = None
        self.last_round_id: int | None = None
        self.next_grant_id = 1
        self.energies: dict[int, np.ndarray] = {}
        self.delivered_blocks: dict[int, set[int]] = {}
        self.resolved_grants: dict[int, dict[int, OutstandingGrant]] = {}
        self.timed_out_grants: dict[int, set[int]] = {}
        self.round_min_value: float | None = None
        self.stats = GatewayStats()

    def register_client(self, client_id: int, state: ClientState) -> None:
        """Attach one connected client state before or between rounds."""
        if isinstance(client_id, bool) or not isinstance(client_id, int) or client_id < 0:
            raise GatewayError("client_id must be a non-negative int")
        if not isinstance(state, ClientState) or state.client_id != client_id:
            raise GatewayError("client state must be a ClientState with the same client_id")
        if self.phase is GatewayPhase.ACTIVE:
            raise GatewayError("cannot register a client during an active round")
        self.clients[client_id] = state

    def unregister_client(self, client_id: int) -> None:
        if self.phase is GatewayPhase.ACTIVE:
            raise GatewayError("cannot unregister a client during an active round")
        self.clients.pop(client_id, None)

    def on_round_start(
        self,
        t_ns: int,
        *,
        round_id: int,
        deadline_ns: int | None = None,
        K: int = K_DEFAULT,
        L: int = L_DEFAULT,
        rho_us: int | float | Mapping[int, int | float] = 0,
        mode: TransferMode = TransferMode.GRANT,
        duration_ns: int | None = None,
    ) -> list[GatewayAction]:
        """Reset per-round state and notify injected estimator/predictor hooks."""
        if self.phase is GatewayPhase.ACTIVE:
            raise GatewayError("a round is already active")
        t_ns = self._uint(t_ns, "t_ns")
        round_id = self._uint(round_id, "round_id", minimum=1, maximum=0xFFFFFFFF)
        if self.last_round_id is not None and round_id <= self.last_round_id:
            raise GatewayError("round_id must increase monotonically")
        if (
            self.phase is GatewayPhase.DEADLINE
            and self.round is not None
            and t_ns < self.round.deadline_ns
        ):
            raise GatewayError("t_ns cannot precede the completed round deadline")
        if any(
            state.round_start_ns is not None and t_ns <= state.round_start_ns
            for state in self.clients.values()
        ):
            raise GatewayError("t_ns must increase across every client's rounds")
        if duration_ns is not None:
            duration_ns = self._uint(duration_ns, "duration_ns", minimum=1)
            deadline_ns = t_ns + duration_ns
            if deadline_ns > 0xFFFFFFFFFFFFFFFF:
                raise GatewayError("deadline_ns exceeds u64")
        elif deadline_ns is not None:
            deadline_ns = self._uint(deadline_ns, "deadline_ns")
        else:
            raise GatewayError("one of deadline_ns or duration_ns is required")
        if deadline_ns <= t_ns:
            raise GatewayError("deadline_ns must be after t_ns")
        K = self._uint(K, "K", minimum=1, maximum=0xFFFF)
        L = self._uint(L, "L", minimum=1, maximum=0xFFFF)
        rho_by_client = self._normalize_rho_us(rho_us)
        scalar_rho_us = next(iter(rho_by_client.values()), 0)
        try:
            mode = TransferMode(mode)
        except (TypeError, ValueError) as exc:
            raise GatewayError("unsupported transfer mode") from exc
        ctx = RoundContext(round_id, t_ns, deadline_ns, K, L, scalar_rho_us, mode, rho_by_client)
        predictor_ctx = RoundCtx(round_id, deadline_ns - t_ns, len(self.clients), K)
        if self.estimator is not None:
            self.estimator.reset_round(predictor_ctx)
        if self.predictor is not None:
            self.predictor.reset_round(predictor_ctx)
        for state in self.clients.values():
            state.reset_round(K, round_start_ns=t_ns)
        self.round = ctx
        self.last_round_id = round_id
        self.phase = GatewayPhase.ACTIVE
        self.next_grant_id = 1
        self.energies.clear()
        self.delivered_blocks = {client_id: set() for client_id in self.clients}
        self.resolved_grants = {client_id: {} for client_id in self.clients}
        self.timed_out_grants = {client_id: set() for client_id in self.clients}
        self.round_min_value = None
        self._reset_view_arrivals()
        return [
            SendRoundStart(
                client_id,
                round_id,
                t_ns,
                deadline_ns,
                K,
                L,
                rho_by_client[client_id],
                mode,
            )
            for client_id in sorted(self.clients)
        ]

    def on_round_start_message(
        self,
        data: bytes | bytearray | memoryview,
        *,
        block_length: int = L_DEFAULT,
    ) -> list[GatewayAction]:
        """Decode and process one protocol ``ROUND_START`` message.

        Transport owns reception and transmission; this entry point only
        bridges the validated wire representation to the round lifecycle.
        """
        try:
            message = decode_message(data, block_length=block_length)
        except WireProtocolError:
            self._reject("ROUND_START")
            return []
        if message.message_type is not MessageType.ROUND_START:
            self._reject("ROUND_START")
            return []
        if message.client_id not in self.clients:
            self._reject("ROUND_START", bad_client_id=True)
            return []
        body = message.body
        try:
            return self.on_round_start(
                message.t_send_ns,
                round_id=message.round_id,
                deadline_ns=body["deadline_ns"],
                K=body["K"],
                L=body["L"],
                rho_us=body["rho_us"],
                mode=body["mode"],
            )
        except GatewayError:
            self._reject("ROUND_START")
            return []

    def on_ready_message(
        self,
        data: bytes | bytearray | memoryview,
        *,
        block_length: int = L_DEFAULT,
        received_at_ns: int | None = None,
    ) -> list[GatewayAction]:
        """Decode READY; network callers should pass gateway receive time.

        The header timestamp is the sender's CLOCK_MONOTONIC value and is only
        used as a compatibility fallback when no receive timestamp is supplied.
        """
        try:
            message = decode_message(data, block_length=block_length)
        except WireProtocolError:
            self._reject("READY")
            return []
        if message.message_type is not MessageType.READY:
            self._reject("READY")
            return []
        if message.client_id not in self.clients:
            self._reject("READY", bad_client_id=True)
            return []
        if self.round is None or message.round_id != self.round.round_id:
            self._reject("READY")
            return []
        event_ns = message.t_send_ns if received_at_ns is None else received_at_ns
        return self.on_ready(message.client_id, message.body["energy"], event_ns)

    def on_data_message(
        self,
        data: bytes | bytearray | memoryview,
        *,
        block_length: int | None = None,
        received_at_ns: int | None = None,
    ) -> list[GatewayAction]:
        """Decode and process DATA; network callers should pass gateway receive time.

        The header timestamp is the sender's CLOCK_MONOTONIC value and is only
        used as a compatibility fallback when no receive timestamp is supplied.
        """
        try:
            if block_length is None:
                decode_length = self.round.L if self.round else L_DEFAULT
            else:
                decode_length = block_length
            message = decode_message(data, block_length=decode_length)
        except WireProtocolError:
            self._reject("DATA")
            return []
        if message.message_type is not MessageType.DATA:
            self._reject("DATA")
            return []
        if message.client_id not in self.clients:
            self._reject("DATA", bad_client_id=True)
            return []
        if self.round is None or message.round_id != self.round.round_id:
            self._reject("DATA")
            return []
        body = message.body
        return self.on_data(
            message.client_id,
            body["grant_id"],
            body["send_pos"],
            body["block_idx"],
            body["payload"],
            message.t_send_ns if received_at_ns is None else received_at_ns,
            n_coords=body["n_coords"],
        )

    def on_ready(self, client_id: int, energies: Sequence[float] | np.ndarray, t_ns: int) -> list[GatewayAction]:
        """Validate READY and install values, then ask injected policy to pump."""
        if not self._event_allowed(t_ns):
            self._reject("READY")
            return []
        if not self._known_client(client_id):
            self._reject("READY", bad_client_id=True)
            return []
        state = self.clients[client_id]
        if state.ready:
            self.stats.duplicate_ready += 1
            return []
        values = np.asarray(energies, dtype=np.float32)
        if values.shape != (self.round.K,) or not np.all(np.isfinite(values)) or np.any(values < 0):
            self._reject("READY")
            return []
        if values.size > 1 and np.any(np.diff(values) > 0):
            self._reject("READY")
            return []
        try:
            state.set_ready(values_from_energies(values, state.w))
        except (TypeError, ValueError):
            self._reject("READY")
            return []
        self.energies[client_id] = values.copy()
        minimum = float(np.min(state.val))
        self.round_min_value = minimum if self.round_min_value is None else min(self.round_min_value, minimum)
        return self._pump(t_ns)

    def on_data(
        self,
        client_id: int,
        grant_id: int,
        send_pos: int,
        block_idx: int,
        payload: Sequence[float] | np.ndarray,
        t_ns: int,
        n_coords: int | None = None,
    ) -> list[GatewayAction]:
        """Validate DATA mode/grant ownership before mutating ClientState."""
        if self.round is None:
            self._reject("DATA")
            return []
        if not self._known_client(client_id):
            self._reject("DATA", bad_client_id=True)
            return []
        if not self._valid_uint(t_ns, 0xFFFFFFFFFFFFFFFF):
            self._reject("DATA")
            return []
        if t_ns > self.round.deadline_ns:
            self.stats.late_after_deadline += 1
            return []
        if self.phase is not GatewayPhase.ACTIVE:
            self._reject("DATA")
            return []
        if not self._valid_time(t_ns, allow_at_deadline=True) or not self._valid_uint(grant_id, 0xFFFFFFFF) or not self._valid_uint(send_pos, 0xFFFF) or not self._valid_uint(block_idx, 0xFFFF):
            self._reject("DATA")
            return []
        state = self.clients[client_id]
        if not 1 <= send_pos <= state.K or not 0 <= block_idx < state.K:
            self._reject("DATA")
            return []
        if isinstance(payload, (str, bytes, bytearray, memoryview)) or not hasattr(payload, "__iter__"):
            self._reject("DATA")
            return []
        try:
            payload_values = tuple(payload)
        except (TypeError, ValueError):
            self._reject("DATA")
            return []
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in payload_values):
            self._reject("DATA")
            return []
        data = np.asarray(payload_values, dtype=np.float32)
        if data.shape != (self.round.L,) or not np.all(np.isfinite(data)):
            self._reject("DATA")
            return []
        if n_coords is None:
            n_coords = self.round.L
        if not self._valid_uint(n_coords, self.round.L) or n_coords < 1:
            self._reject("DATA")
            return []

        already_delivered = state.status[send_pos - 1] == Status.DELIVERED
        grant: OutstandingGrant | None = None
        late_after_timeout = False
        grant_is_outstanding = False
        if self.round.mode is TransferMode.GRANT:
            grant = state.outstanding.get(grant_id)
            grant_is_outstanding = grant is not None
            if grant is None:
                grant = self.resolved_grants.get(client_id, {}).get(grant_id)
                late_after_timeout = (
                    grant is not None
                    and grant_id in self.timed_out_grants.get(client_id, set())
                )
            if grant is None or grant.pos != send_pos:
                self._reject("DATA")
                return []
            if state.status[send_pos - 1] not in (Status.OUTSTANDING, Status.LOST, Status.DELIVERED):
                self._reject("DATA")
                return []
            if already_delivered:
                self.stats.duplicate_data += 1
                if not grant_is_outstanding:
                    if late_after_timeout:
                        self.stats.late_data += 1
                        self.stats.late_after_timeout += 1
                    return []
        elif self.round.mode in (TransferMode.PUSH, TransferMode.PLAN):
            if grant_id != DATA_GRANT_ID:
                self._reject("DATA")
                return []
            if already_delivered:
                self.stats.duplicate_data += 1
                return []
        else:
            self._reject("DATA")
            return []

        if self.round.mode is TransferMode.GRANT:
            active_grant = state.outstanding.get(grant_id)
            if active_grant is not None and t_ns > active_grant.timeout_ns:
                self._resolve_timeout(client_id, state, grant_id, active_grant, t_ns)
                grant = active_grant
                late_after_timeout = True

        if not already_delivered and self.value_integrity is not None:
            if not self.value_integrity.valid(client_id, block_idx, data, self.energies.get(client_id, np.empty(0, np.float32)), n_coords):
                count = self.stats.energy_violations.get(client_id, 0) + 1
                self.stats.energy_violations[client_id] = count
                if count >= 3 and state.val.size and self.round_min_value is not None:
                    remaining = state.status != Status.DELIVERED
                    state.val[remaining] = self.round_min_value

        if not already_delivered:
            first = state.mark_delivered(send_pos)
            if not first:
                self.stats.duplicate_data += 1
                return []
            self.delivered_blocks.setdefault(client_id, set()).add(block_idx)
            state.delivered_payloads[block_idx] = data.copy()
            state.delivered_n_coords[block_idx] = n_coords
            self.stats.accepted_data += 1
        if late_after_timeout:
            self.stats.late_data += 1
            self.stats.late_after_timeout += 1
        if grant is not None:
            state.outstanding.pop(grant_id, None)
            self.resolved_grants.setdefault(client_id, {})[grant_id] = grant
            if self.estimator is not None and not late_after_timeout:
                self.estimator.resolve(client_id, grant, True, t_ns)
        return self._pump(t_ns)

    def on_tick(self, t_ns: int) -> list[GatewayAction]:
        """Resolve expired grants, notify hooks, and request further grants."""
        if self.round is None or self.phase is not GatewayPhase.ACTIVE:
            return []
        if not self._valid_time(t_ns, allow_after_deadline=True):
            self._reject("TICK")
            return []
        if t_ns >= self.round.deadline_ns:
            return self.on_deadline(self.round.deadline_ns)
        for client_id, state in self.clients.items():
            expired = [grant_id for grant_id, grant in state.outstanding.items() if grant.timeout_ns <= t_ns]
            for grant_id in expired:
                grant = state.outstanding[grant_id]
                self._resolve_timeout(client_id, state, grant_id, grant, t_ns)
        if self.estimator is not None:
            self.estimator.on_tick(t_ns)
        return self._pump(t_ns)

    def on_deadline(self, t_ns: int) -> list[GatewayAction]:
        """Stop granting, resolve outstanding grants as losses, and emit round-end actions."""
        if self.round is None:
            return []
        if self.phase is GatewayPhase.DEADLINE:
            return []
        if not self._valid_time(t_ns, allow_after_deadline=True):
            self._reject("DEADLINE")
            return []
        if t_ns < self.round.deadline_ns:
            self._reject("DEADLINE")
            return []
        deadline_ns = self.round.deadline_ns
        actions: list[GatewayAction] = []
        pending_resolutions: list[tuple[int, OutstandingGrant]] = []
        for client_id, state in self.clients.items():
            for grant_id, grant in list(state.outstanding.items()):
                self._resolve_timeout(
                    client_id,
                    state,
                    grant_id,
                    grant,
                    deadline_ns,
                    notify_estimator=False,
                )
                pending_resolutions.append((client_id, grant))
            state.update_starve()
            actions.append(
                SendRoundEnd(
                    client_id,
                    self.round.round_id,
                    bitmap_from_indices(self.delivered_blocks.get(client_id, ()), state.K),
                )
            )
        self.phase = GatewayPhase.DEADLINE
        if self.estimator is not None:
            for client_id, grant in pending_resolutions:
                self.estimator.resolve(client_id, grant, False, deadline_ns)
            self.estimator.on_deadline(deadline_ns)
        if self.aggregate is not None:
            actions.append(
                EmitAggregate(
                    self.aggregate.aggregate(self.clients),
                    round_id=self.round.round_id,
                    client_n_coords={
                        client_id: dict(state.delivered_n_coords)
                        for client_id, state in self.clients.items()
                    },
                )
            )
        return actions

    def _resolve_timeout(
        self,
        client_id: int,
        state: ClientState,
        grant_id: int,
        grant: OutstandingGrant,
        t_ns: int,
        *,
        notify_estimator: bool = True,
    ) -> None:
        state.outstanding.pop(grant_id, None)
        state.mark_timeout(grant.pos)
        self.resolved_grants.setdefault(client_id, {})[grant_id] = grant
        self.timed_out_grants.setdefault(client_id, set()).add(grant_id)
        if notify_estimator and self.estimator is not None:
            self.estimator.resolve(client_id, grant, False, t_ns)

    def _pump(self, t_ns: int) -> list[GatewayAction]:
        if self.round is None or self.phase is not GatewayPhase.ACTIVE:
            return []
        if t_ns >= self.round.deadline_ns:
            return []
        if self.round.mode is not TransferMode.GRANT:
            return []
        if (
            self.selector is None
            or self.eligibility is None
            or self.pacing is None
            or self.timeout is None
            or (self.prediction is None and (self.predictor is None or self.view is None))
        ):
            self.stats.missing_dependencies += 1
            return []
        actions: list[GatewayAction] = []
        while True:
            candidates = self.eligibility.candidates(self.clients, t_ns, self.round.deadline_ns, self.retransmit)
            if len(candidates) == 0:
                break
            client_id = self.selector.select(candidates)
            if client_id not in set(int(value) for value in candidates.client_id):
                raise GatewayError("selector returned a client outside CandidateView")
            if self.prediction is not None:
                p_hat = float(self.prediction.probability(client_id, t_ns))
            else:
                p_hat = self._predict_probability(client_id, t_ns)
            if not np.isfinite(p_hat) or not 0.0 <= p_hat <= 1.0:
                raise GatewayError("prediction probability must be finite and in [0, 1]")
            if not self.pacing.allow(client_id, p_hat, t_ns, self.round.deadline_ns):
                break
            state = self.clients[client_id]
            send_pos = state.next_pos(self.retransmit)
            if send_pos > state.K:
                break
            timeout_ns = self.timeout.timeout_ns(client_id, t_ns)
            timeout_ns = self._uint(timeout_ns, "timeout_ns")
            if timeout_ns <= t_ns:
                raise GatewayError("timeout_ns must be after grant issue time")
            timeout_ns = min(timeout_ns, self.round.deadline_ns)
            grant_id = self.next_grant_id
            self.next_grant_id += 1
            state.take_position(send_pos, self.retransmit)
            state.outstanding[grant_id] = OutstandingGrant(send_pos, t_ns, p_hat, timeout_ns)
            actions.append(SendGrant(client_id, grant_id, send_pos))
        return actions

    def _predict_probability(self, client_id: int, t_ns: int) -> float:
        """Run the configured Predictor on the selected client before mutation.

        The public ``PredictionProvider`` remains available for callers that
        already own feature construction.  When only a Predictor is supplied,
        this bridge builds the standard 48-feature input from the injected
        GatewayView.  FallbackPredictor owns exception, non-finite, and timeout
        handling; this method only validates the returned scalar shape.
        """
        if self.round is None or self.predictor is None or self.view is None:
            raise GatewayError("predictor and feature view are required")
        feature_view = copy.copy(self.view)
        feature_view.t_ns = t_ns - self.round.start_ns
        feature_view.D_ns = self.round.deadline_ns - self.round.start_ns
        inp = build_pred_input([self.clients[client_id]], feature_view)
        output = np.asarray(self.predictor.predict(inp))
        if output.shape != (1,):
            raise GatewayError("predictor must return one probability for one client")
        return float(output[0])

    def _event_allowed(self, t_ns: int) -> bool:
        return self.round is not None and self.phase is GatewayPhase.ACTIVE and self._valid_time(t_ns)

    def _valid_time(
        self,
        t_ns: int,
        *,
        allow_after_deadline: bool = False,
        allow_at_deadline: bool = False,
    ) -> bool:
        if not self._valid_uint(t_ns, 0xFFFFFFFFFFFFFFFF) or self.round is None:
            return False
        if allow_after_deadline:
            return self.round.start_ns <= t_ns
        if allow_at_deadline:
            return self.round.start_ns <= t_ns <= self.round.deadline_ns
        return self.round.start_ns <= t_ns < self.round.deadline_ns

    def _reject(self, kind: str, *, bad_client_id: bool = False) -> None:
        self.stats.invalid_events += 1
        if bad_client_id:
            self.stats.bad_client_id += 1
        if kind == "READY":
            self.stats.rejected_ready += 1
        elif kind == "ROUND_START":
            self.stats.rejected_round_start += 1
        elif kind == "DATA":
            self.stats.rejected_data += 1

    def _reset_view_arrivals(self) -> None:
        arrivals = getattr(self.view, "arrivals", None)
        if arrivals is None:
            return
        if isinstance(arrivals, np.ndarray):
            arrivals[...] = 0
        elif hasattr(arrivals, "clear"):
            arrivals.clear()

    def _normalize_rho_us(self, rho_us: int | float | Mapping[int, int | float]) -> dict[int, int]:
        values = rho_us if isinstance(rho_us, Mapping) else {client_id: rho_us for client_id in self.clients}
        result: dict[int, int] = {}
        for client_id in self.clients:
            raw = values.get(client_id, 0)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not np.isfinite(raw):
                raise GatewayError("rho_us values must be finite numbers")
            floored = int(np.floor(float(raw)))
            if not 0 <= floored <= 0xFFFFFFFF:
                raise GatewayError("rho_us values must be in [0, 4294967295]")
            result[client_id] = floored
        return result

    @staticmethod
    def _valid_uint(value: Any, maximum: int) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= maximum

    def _known_client(self, client_id: Any) -> bool:
        return self._valid_uint(client_id, 0xFFFF) and client_id in self.clients

    @staticmethod
    def _uint(value: Any, name: str, *, minimum: int = 0, maximum: int = 0xFFFFFFFFFFFFFFFF) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            raise GatewayError(f"{name} must be an integer in [{minimum}, {maximum}]")
        return value


__all__ = [
    "AggregateProvider",
    "EligibilityProvider",
    "EmitAggregate",
    "EstimatorHooks",
    "GatewayAction",
    "GatewayCore",
    "GatewayError",
    "GatewayPhase",
    "GatewayStats",
    "MessageType",
    "PacingProvider",
    "PredictionProvider",
    "RoundContext",
    "SendRoundStart",
    "SendGrant",
    "SendRoundEnd",
    "TimeoutProvider",
    "ValueIntegrityChecker",
]
