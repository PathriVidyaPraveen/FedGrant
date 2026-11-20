"""B3 PLAN-then-send baseline (Section 10)."""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Awaitable, cast

import numpy as np

from fedgrant.baselines.b2_udp import (
    B2ClientCore,
    B2ClientTransfer,
    B2ProtocolError,
    B2RoundAccumulator,
    B2RoundResult,
    B2TransferError,
    B2UdpClient,
    apply_delivery_result,
    block_payload_f16,
    decode_data_packet,
    encode_data_packet,
)
from fedgrant.baselines.b3_pacing import (
    data_frame_size_bytes,
    paced_blocks_sent,
    pacing_send_time_ns,
)
from fedgrant.common.constants import MessageType, TransferMode
from fedgrant.common.wire import WireProtocolError, decode_message, encode_message
from fedgrant.core.estimators import RateWindow, rtt_update
from fedgrant.core.state import ClientState
from fedgrant.encoder.blocks import values as block_values
from fedgrant.encoder.ef import bitmap_from_indices, indices_from_bitmap
from fedgrant.policy.constants import C as DEFAULT_C
from fedgrant.policy.constants import L as L_DEFAULT
from fedgrant.policy.plan import (
    DeliveryRatioEWMA,
    plan_allocate,
    plan_budget,
    plan_cap_offset_ns,
    plan_caps,
)

log = logging.getLogger(__name__)
_NANOSECONDS_PER_SECOND = 1_000_000_000
ROUND_END_GRACE_NS = 200_000_000  # Section 9: ROUND_END fallback after D
# The spec defines the EWMA weight but no prior; keep an unobserved client conservative.
B3_DELIVERY_RATIO_INIT = 0.1


class B3PlanError(ValueError):
    """Invalid B3 event, plan, or round transition."""


@dataclass(frozen=True, slots=True)
class B3ClientPlan:
    client_id: int
    blocks: int
    rate_bps: int
    sent_ns: int


@dataclass(frozen=True, slots=True)
class B3RoundResult:
    round_id: int
    start_ns: int
    deadline_ns: int
    plan_ns: int  # actual plan time: the last READY's arrival, or plan_cap_ns at the latest
    budget: int
    median_rtt_ns: float | None
    caps: Mapping[int, int]
    plans: Mapping[int, B3ClientPlan]
    rtt_samples_ns: Mapping[int, int]
    round_result: B2RoundResult
    plan_cap_ns: int | None = None
    ready_clients: int = 0

    @property
    def planned_early(self) -> bool:
        """True when every READY arrived and the plan was made before the cap."""
        return self.plan_cap_ns is not None and self.plan_ns < self.plan_cap_ns

    @property
    def completed_at_deadline(self) -> bool:
        return self.round_result.completed_at_deadline

    def aggregate(self, aggregator) -> np.ndarray:
        return self.round_result.aggregate(aggregator)


class B3GatewayCore:
    """Deterministic B3 plan and DATA state, independent of socket scheduling.

    Plan time (CHANGELOG 2026-11-21): the plan is made when the last expected
    READY arrives, and at ``plan_cap_ns`` = start + 0.1 D + 105 ms at the latest.
    READYs received at the cap are included; later ones are rejected for the
    round. ``plan_ns``/``plan_offset_ns`` hold the cap until the plan is made and
    the actual plan time afterwards; the budget M and the caps c_i use the actual
    plan time. Callers plan at ``plan_cap_ns`` unless ``ready_complete`` lets
    them plan earlier.
    """

    def __init__(
        self,
        *,
        round_id: int,
        start_ns: int,
        deadline_ns: int,
        d: int,
        client_weights: Mapping[int, float],
        L: int = L_DEFAULT,
        bottleneck_bps: int = DEFAULT_C,
        client_states: Mapping[int, ClientState] | None = None,
        delivery_ewma: DeliveryRatioEWMA | None = None,
        delivery_ratio_init: float = B3_DELIVERY_RATIO_INIT,
    ) -> None:
        if isinstance(d, bool) or not isinstance(d, int) or d < 1:
            raise B3PlanError("d must be a positive integer")
        if isinstance(L, bool) or not isinstance(L, int) or L < 1:
            raise B3PlanError("L must be a positive integer")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or not 1 <= round_id <= 0xFFFFFFFF:
            raise B3PlanError("round_id must be in [1, 2^32-1]")
        if isinstance(start_ns, bool) or not isinstance(start_ns, int) or start_ns < 0:
            raise B3PlanError("start_ns must be a non-negative integer")
        if isinstance(deadline_ns, bool) or not isinstance(deadline_ns, int) or deadline_ns <= start_ns:
            raise B3PlanError("deadline_ns must be after start_ns")
        if isinstance(bottleneck_bps, bool) or not isinstance(bottleneck_bps, int) or bottleneck_bps < 1:
            raise B3PlanError("bottleneck_bps must be a positive integer")
        delivery_ratio_init = _validate_delivery_ratio_init(delivery_ratio_init)
        self.round_id = round_id
        self.start_ns = start_ns
        self.deadline_ns = deadline_ns
        self.duration_ns = deadline_ns - start_ns
        self.plan_cap_offset_ns = plan_cap_offset_ns(self.duration_ns)
        self.plan_cap_ns = start_ns + self.plan_cap_offset_ns
        # Latest plan time until compute_plans() runs, then the actual plan time.
        self.plan_offset_ns = self.plan_cap_offset_ns
        self.plan_ns = self.plan_cap_ns
        self.d = d
        self.L = L
        self.K = (d + L - 1) // L
        self.bottleneck_bps = bottleneck_bps
        self.client_ids = tuple(sorted(client_weights))
        if not self.client_ids:
            raise B3PlanError("at least one client is required")
        self.weights = {cid: float(client_weights[cid]) for cid in self.client_ids}
        self.accumulator = B2RoundAccumulator(
            round_id=round_id,
            client_ids=self.client_ids,
            start_ns=start_ns,
            deadline_ns=deadline_ns,
            d=d,
            L=L,
            client_weights=self.weights,
        )
        if client_states is not None and set(client_states) != set(self.client_ids):
            raise B3PlanError("client_states must contain exactly the configured clients")
        self.states: dict[int, ClientState] = dict(client_states or {})
        self.windows: dict[int, RateWindow] = {}
        for cid in self.client_ids:
            state = self.states.get(cid)
            if state is None:
                state = ClientState(cid, self.weights[cid], self.K)
                self.states[cid] = state
            elif state.client_id != cid or state.w != self.weights[cid]:
                raise B3PlanError(f"persistent state for client {cid} has incompatible identity/weight")
            state.reset_round(self.K, round_start_ns=start_ns)
            self.windows[cid] = RateWindow(start_ns, outstanding=0)
        self.delivery_ewma = (
            delivery_ewma
            if delivery_ewma is not None
            else DeliveryRatioEWMA(init=delivery_ratio_init)
        )
        for cid in self.client_ids:
            _validate_delivery_ratio_init(self.delivery_ewma.get(cid))
        self.ready_times_ns: dict[int, int] = {}
        self.plans: dict[int, B3ClientPlan] = {}
        self._plan_sent: set[int] = set()
        self.caps: dict[int, int] = {}
        self.budget: int | None = None
        self.median_rtt_ns: float | None = None
        self.rtt_samples_ns: dict[int, int] = {}
        # (send_pos, arrival_ns) of the first accepted DATA that produced each
        # client's rtt_samples_ns entry; kept for raw logs, value unchanged.
        self.rtt_first_data: dict[int, tuple[int, int]] = {}
        self.plan_srtt_ns: dict[int, int] = {}
        self.transmission_start_ns: dict[int, int] = {}
        self._plan_computed = False
        self._finished = False
        self.rejected_events = 0

    def accept_ready(self, packet: bytes, received_ns: int) -> None:
        try:
            message = decode_message(packet)
        except WireProtocolError as exc:
            self.rejected_events += 1
            raise B3PlanError(f"invalid READY: {exc}") from exc
        if message.message_type is not MessageType.READY:
            self.rejected_events += 1
            raise B3PlanError("B3 gateway expected READY")
        if message.round_id != self.round_id or message.client_id not in self.states:
            self.rejected_events += 1
            raise B3PlanError("READY has a stale round or unknown client")
        if received_ns < self.start_ns or received_ns > self.plan_cap_ns or self._plan_computed:
            self.rejected_events += 1
            raise B3PlanError("READY arrived outside the round start-to-plan window")
        cid = message.client_id
        if cid in self.ready_times_ns:
            self.rejected_events += 1
            raise B3PlanError("duplicate READY")
        energies = np.asarray(message.body["energy"], dtype=np.float32)
        if message.body["K"] != self.K:
            self.rejected_events += 1
            raise B3PlanError("READY K does not match the round")
        values = block_values(energies, self.weights[cid])
        try:
            self.states[cid].set_ready(values)
        except (TypeError, ValueError) as exc:
            self.rejected_events += 1
            raise B3PlanError(f"invalid READY values: {exc}") from exc
        self.ready_times_ns[cid] = received_ns

    @property
    def ready_complete(self) -> bool:
        """Every configured client's READY has arrived: the plan may be made now."""
        return len(self.ready_times_ns) == len(self.client_ids)

    @property
    def plan_computed(self) -> bool:
        return self._plan_computed

    def compute_plans(self, now_ns: int) -> dict[int, B3ClientPlan]:
        if self._plan_computed:
            raise B3PlanError("PLAN has already been computed for this round")
        if now_ns < self.plan_cap_ns and not self.ready_complete:
            raise B3PlanError("cannot compute PLAN before every READY or the plan cap")
        if self.ready_times_ns and now_ns < max(self.ready_times_ns.values()):
            raise B3PlanError("cannot compute PLAN before the READYs it uses")
        if now_ns >= self.deadline_ns:
            raise B3PlanError("cannot compute PLAN at or after the deadline")
        # Budget and caps use the actual plan time (CHANGELOG 2026-11-21).
        self.plan_ns = now_ns
        self.plan_offset_ns = now_ns - self.start_ns

        for cid in self.client_ids:
            self._advance_rate_window(cid, self.plan_ns, backlogged=False)
        ready_ids = tuple(cid for cid in self.client_ids if self.ready_times_ns.get(cid, self.plan_ns + 1) <= self.plan_ns)
        if ready_ids:
            self.median_rtt_ns = float(statistics.median(self.states[cid].srtt_ns for cid in ready_ids))
            self.budget = plan_budget(
                self.bottleneck_bps,
                self.duration_ns,
                self.plan_offset_ns,
                self.median_rtt_ns,
            )
            self.caps = {
                cid: plan_caps(
                    self.delivery_ewma.get(cid),
                    self.states[cid].bhat_bps,
                    self.duration_ns,
                    self.plan_offset_ns,
                    self.states[cid].srtt_ns,
                )
                for cid in ready_ids
            }
            values = {cid: self.states[cid].val for cid in ready_ids}
            allocation = plan_allocate(values, self.caps, self.budget)
        else:
            self.median_rtt_ns = None
            self.budget = 0
            self.caps = {}
            allocation = {}

        plans: dict[int, B3ClientPlan] = {}
        for cid in self.client_ids:
            rate = _rate_to_wire(self.states[cid].bhat_bps)
            plans[cid] = B3ClientPlan(
                client_id=cid,
                blocks=allocation.get(cid, 0),
                rate_bps=rate,
                sent_ns=0,
            )
        self.plans = plans
        self._plan_computed = True
        return plans.copy()

    def mark_plan_sent(self, client_id: int, sent_ns: int) -> B3ClientPlan:
        if not self._plan_computed or client_id not in self.plans:
            raise B3PlanError("cannot send PLAN before allocation")
        if sent_ns < self.plan_ns or sent_ns >= self.deadline_ns:
            raise B3PlanError("PLAN send time must be within [t_plan, deadline)")
        if client_id in self._plan_sent:
            raise B3PlanError("PLAN was already sent for this client")
        current = self.plans[client_id]
        srtt_ns = self.states[client_id].srtt_ns
        fallback_start_ns = sent_ns + srtt_ns // 2
        if fallback_start_ns > 0xFFFFFFFFFFFFFFFF:
            raise B3PlanError("PLAN fallback transmission start exceeds u64 range")
        plan = B3ClientPlan(client_id, current.blocks, current.rate_bps, sent_ns)
        self.plans[client_id] = plan
        self.plan_srtt_ns[client_id] = srtt_ns
        self.transmission_start_ns[client_id] = fallback_start_ns
        self._plan_sent.add(client_id)
        return plan

    def accept_data(self, packet: bytes, received_ns: int) -> None:
        if self._finished or not self.accumulator._active:
            self.rejected_events += 1
            raise B3PlanError("B3 round is already closed")
        if not isinstance(received_ns, int) or isinstance(received_ns, bool):
            self.rejected_events += 1
            raise B3PlanError("DATA arrival time must be an integer nanosecond timestamp")
        if not self._plan_computed:
            self.rejected_events += 1
            raise B3PlanError("DATA arrived before PLAN")
        try:
            message = decode_message(packet, block_length=self.L)
        except WireProtocolError as exc:
            self.rejected_events += 1
            raise B3PlanError(f"invalid DATA: {exc}") from exc
        if message.message_type is not MessageType.DATA:
            self.rejected_events += 1
            raise B3PlanError("B3 gateway expected DATA")
        cid = message.client_id
        if cid not in self._plan_sent:
            self.rejected_events += 1
            raise B3PlanError("DATA arrived before this client's PLAN")
        if message.body["send_pos"] > self.plans[cid].blocks:
            self.rejected_events += 1
            raise B3PlanError("DATA send_pos exceeds the client's PLAN")
        if received_ns < self.plans[cid].sent_ns or received_ns > self.deadline_ns:
            self.rejected_events += 1
            raise B3PlanError("DATA arrived outside the PLAN-to-deadline window")
        try:
            _round, _client, send_pos, _block, _count, _payload = decode_data_packet(packet, self.L)
            candidate_start_ns = (
                received_ns
                - self.plan_srtt_ns[cid] // 2
                - pacing_send_time_ns(
                    send_pos,
                    t_start_ns=0,
                    k=self.plans[cid].blocks,
                    rate_bps=self.plans[cid].rate_bps,
                    block_size_bytes=data_frame_size_bytes(self.L),
                )
            )
            self.accumulator.accept_packet(packet, received_ns)
        except (B2ProtocolError, ValueError) as exc:
            self.rejected_events += 1
            raise B3PlanError(f"invalid B3 DATA: {exc}") from exc

        self.transmission_start_ns[cid] = max(
            self.plans[cid].sent_ns,
            min(self.transmission_start_ns[cid], candidate_start_ns),
        )
        if cid not in self.rtt_samples_ns:
            sample = received_ns - self.plans[cid].sent_ns
            rtt_update(self.states[cid], sample)
            self.rtt_samples_ns[cid] = sample
            self.rtt_first_data[cid] = (int(send_pos), int(received_ns))
        self._advance_rate_window(cid, received_ns, backlogged=None)
        self.windows[cid].add_delivery(
            received_ns,
            bits=8 * data_frame_size_bytes(self.L),
        )

    def finish(self, completed_ns: int) -> B3RoundResult:
        if self._finished:
            raise B3PlanError("B3 round has already finished")
        if not self._plan_computed:
            raise B3PlanError("cannot finish before PLAN computation")
        if completed_ns < self.deadline_ns:
            raise B3PlanError("cannot finish a B3 round before its deadline")
        for cid in self.client_ids:
            self._advance_rate_window(cid, self.deadline_ns, backlogged=None)
            planned = self.plans[cid].blocks
            delivered = self.accumulator.clients[cid].delivered
            self.delivery_ewma.update(cid, len(delivered), planned)
        self.accumulator.stop(completed_ns, "deadline")
        round_result = self.accumulator.result()
        self._finished = True
        return B3RoundResult(
            round_id=self.round_id,
            start_ns=self.start_ns,
            deadline_ns=self.deadline_ns,
            plan_ns=self.plan_ns,
            budget=self.budget or 0,
            median_rtt_ns=self.median_rtt_ns,
            caps=self.caps.copy(),
            plans=self.plans.copy(),
            rtt_samples_ns=self.rtt_samples_ns.copy(),
            round_result=round_result,
            plan_cap_ns=self.plan_cap_ns,
            ready_clients=len(self.ready_times_ns),
        )

    def _advance_rate_window(
        self,
        client_id: int,
        time_ns: int,
        *,
        backlogged: bool | None,
    ) -> int:
        window = self.windows[client_id]
        state = self.states[client_id]
        closed = 0
        explicit_backlogged = backlogged
        while window.t_end <= time_ns:
            window_backlogged = explicit_backlogged
            if window_backlogged is None:
                plan = self.plans.get(client_id)
                start_ns = self.transmission_start_ns.get(client_id)
                window_backlogged = bool(
                    plan is not None
                    and start_ns is not None
                    and start_ns <= window.t_start
                    and paced_blocks_sent(
                        window.t_end,
                        t_start_ns=start_ns,
                        k=plan.blocks,
                        rate_bps=plan.rate_bps,
                        block_size_bytes=data_frame_size_bytes(self.L),
                    )
                    < plan.blocks
                )
            window.close(state, backlogged=window_backlogged)
            closed += 1
        return closed


class B3UdpGateway(asyncio.DatagramProtocol):
    """UDP control/data adapter for B3's standard ROUND_START, READY, PLAN and DATA."""

    def __init__(
        self,
        *,
        client_weights: Mapping[int, float],
        endpoints: Mapping[int, tuple[str, int]],
        d: int,
        duration_ns: int,
        L: int = L_DEFAULT,
        bottleneck_bps: int = DEFAULT_C,
        delivery_ratio_init: float = B3_DELIVERY_RATIO_INIT,
        host: str = "0.0.0.0",
        port: int = 9102,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if set(endpoints) != set(client_weights):
            raise ValueError("endpoints must contain exactly the configured clients")
        if isinstance(duration_ns, bool) or not isinstance(duration_ns, int) or duration_ns < 10:
            raise ValueError("duration_ns must be at least 10 ns")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 0xFFFF:
            raise ValueError("port must be in [0, 65535]")
        self.client_weights = dict(client_weights)
        self.endpoints = {cid: _endpoint(endpoint) for cid, endpoint in endpoints.items()}
        self.d = d
        self.duration_ns = duration_ns
        self.L = L
        self.bottleneck_bps = bottleneck_bps
        self.delivery_ewma = DeliveryRatioEWMA(
            init=_validate_delivery_ratio_init(delivery_ratio_init)
        )
        self.host = host
        self.port = port
        self.clock_ns = clock_ns
        self.transport: asyncio.DatagramTransport | None = None
        self.core: B3GatewayCore | None = None
        self.client_states: dict[int, ClientState] = {}
        self._plan_task: asyncio.Task[None] | None = None
        self._deadline_task: asyncio.Task[None] | None = None
        self._plan_error: Exception | None = None
        self._closed_event = asyncio.Event()
        self.rejected_datagrams = 0
        self.plan_send_attempts = 0
        self.plan_local_handoffs = 0

    async def bind(self) -> B3UdpGateway:
        if self.transport is not None:
            raise B3PlanError("B3 gateway is already bound")
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: self, local_addr=(self.host, self.port))
        return self

    @property
    def address(self) -> tuple[str, int]:
        if self.transport is None:
            raise B3PlanError("B3 gateway is not bound")
        address = self.transport.get_extra_info("sockname")
        if not isinstance(address, tuple) or len(address) < 2:
            raise B3PlanError("B3 gateway socket has no bound address")
        return str(address[0]), int(address[1])

    def set_endpoint(self, client_id: int, endpoint: tuple[str, int]) -> None:
        if self.core is not None:
            raise B3PlanError("cannot change client endpoints during a B3 round")
        if client_id not in self.client_weights:
            raise B3PlanError("endpoint client_id is not registered")
        self.endpoints[client_id] = _endpoint(endpoint)

    def start_round(self, round_id: int, *, rho_us: Mapping[int, int] | None = None) -> B3GatewayCore:
        """Send ROUND_START (mode PLAN) to every client and schedule t_plan and D.

        ``rho_us`` gives each client's readiness delay (Section 3.1); the
        default 0 keeps the previous behaviour.
        """
        if self.transport is None:
            raise B3PlanError("B3 gateway is not bound")
        rho_by_client = {cid: 0 for cid in self.endpoints}
        if rho_us is not None:
            if set(rho_us) != set(self.endpoints):
                raise B3PlanError("rho_us must contain exactly the configured clients")
            for cid, value in rho_us.items():
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
                    raise B3PlanError("rho_us values must be u32 integers")
            rho_by_client = dict(rho_us)
        if self.core is not None and not self.core._finished:
            raise B3PlanError("previous B3 round has not finished")
        if self.core is not None and round_id <= self.core.round_id:
            raise B3PlanError("round_id must increase monotonically")
        start_ns = self.clock_ns()
        deadline_ns = start_ns + self.duration_ns
        K = (self.d + self.L - 1) // self.L
        if not self.client_states:
            self.client_states = {
                cid: ClientState(cid, float(self.client_weights[cid]), K)
                for cid in self.client_weights
            }
        self.core = B3GatewayCore(
            round_id=round_id,
            start_ns=start_ns,
            deadline_ns=deadline_ns,
            d=self.d,
            L=self.L,
            client_weights=self.client_weights,
            bottleneck_bps=self.bottleneck_bps,
            client_states=self.client_states,
            delivery_ewma=self.delivery_ewma,
        )
        self.plan_send_attempts = 0
        self.plan_local_handoffs = 0
        self._closed_event.clear()
        self._plan_error = None
        for cid, endpoint in self.endpoints.items():
            self._send_message(
                MessageType.ROUND_START,
                client_id=cid,
                t_send_ns=start_ns,
                body={
                    "deadline_ns": deadline_ns,
                    "K": self.core.K,
                    "L": self.L,
                    "rho_us": rho_by_client[cid],
                    "mode": int(TransferMode.PLAN),
                },
            )
        self._plan_task = asyncio.create_task(self._plan_at_deadline(), name="b3-plan")
        self._plan_task.add_done_callback(self._plan_task_done)
        self._deadline_task = asyncio.create_task(self._close_at_deadline(), name="b3-deadline")
        return self.core

    def datagram_received(
        self,
        data: bytes,
        addr: tuple[str, int] | tuple[str, int, int, int],
    ) -> None:
        self.receive_datagram(data, addr, self.clock_ns())

    def receive_datagram(
        self,
        data: bytes,
        addr: tuple[str, int] | tuple[str, int, int, int],
        received_ns: int,
    ) -> None:
        """Process one datagram stamped once at receipt.

        Callers that also log the datagram pass the same ``received_ns``, so the
        logged arrival time is exactly the time used for the RTT sample.
        """
        core = self.core
        if core is None:
            self.rejected_datagrams += 1
            return
        try:
            message = decode_message(data, block_length=self.L)
            if message.message_type not in (MessageType.READY, MessageType.DATA):
                raise B3PlanError("unexpected B3 client message type")
            if self.endpoints.get(message.client_id) != _endpoint(addr[:2]):
                raise B3PlanError("B3 datagram came from an unexpected endpoint")
            if message.message_type is MessageType.READY:
                core.accept_ready(data, received_ns)
            else:
                core.accept_data(data, received_ns)
        except (B3PlanError, B2ProtocolError, WireProtocolError, ValueError, TypeError):
            self.rejected_datagrams += 1
            log.debug("rejecting invalid B3 datagram", exc_info=True)
            return
        if core.ready_complete and not core.plan_computed:
            # Early close: the last expected READY is in, so plan now instead of
            # waiting for the cap (CHANGELOG 2026-11-21).
            try:
                self._send_plans()
            except Exception as exc:  # noqa: BLE001 - surfaced by wait(), like the plan task
                self._plan_error = exc
                log.error("B3 early PLAN failed", exc_info=True)
                self._closed_event.set()

    def error_received(self, exc: Exception) -> None:
        log.error("B3 UDP gateway socket error: %r", exc)

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.DatagramTransport, transport)

    def connection_lost(self, exc: Exception | None) -> None:
        self.transport = None
        self._closed_event.set()

    def send_round_end(self, result: B3RoundResult) -> None:
        """Send ROUND_END with each client's delivered-block bitmap (Section 8)."""
        core = self.core
        if core is None or result.round_id != core.round_id:
            raise B3PlanError("ROUND_END must match the current B3 round")
        for cid in self.endpoints:
            delivered = result.round_result.client_results[cid].delivered_indices
            self._send_message(
                MessageType.ROUND_END,
                client_id=cid,
                t_send_ns=self.clock_ns(),
                body={
                    "K": core.K,
                    "n_delivered": len(delivered),
                    "bitmap": bitmap_from_indices(delivered, core.K),
                },
            )

    async def wait(self) -> B3RoundResult:
        if self.core is None or self._deadline_task is None:
            raise B3PlanError("B3 round has not started")
        await self._closed_event.wait()
        if self._plan_error is not None:
            raise B3PlanError(f"B3 PLAN task failed: {self._plan_error}") from self._plan_error
        if self._plan_task is not None:
            await self._plan_task
        return self.core.finish(self.core.deadline_ns)

    async def close(self) -> None:
        for task in (self._plan_task, self._deadline_task):
            if task is not None and task is not asyncio.current_task() and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self._plan_task = None
        self._deadline_task = None
        transport, self.transport = self.transport, None
        if transport is not None:
            transport.close()
        self._closed_event.set()

    async def _plan_at_deadline(self) -> None:
        core = self.core
        if core is None:
            return
        await _sleep_until(core.plan_cap_ns, self.clock_ns)
        if core is self.core and not core.plan_computed:
            self._send_plans()

    def _send_plans(self) -> None:
        """Compute the round's plans now and send one PLAN per client."""
        core = self.core
        if core is None or core.plan_computed:
            return
        now_ns = self.clock_ns()
        if now_ns >= core.deadline_ns:
            # Only reachable when the cap is D - 1 ns (D < 116.7 ms): the budget is
            # zero, so plan with no time left and send nothing that would arrive after D.
            core.compute_plans(core.deadline_ns - 1)
            return
        plans = core.compute_plans(now_ns)
        for cid, plan in plans.items():
            sent_ns = self.clock_ns()
            self._send_message(
                MessageType.PLAN,
                client_id=cid,
                t_send_ns=sent_ns,
                body={"k": plan.blocks, "rate_bps": plan.rate_bps},
            )
            core.mark_plan_sent(cid, sent_ns)

    async def _close_at_deadline(self) -> None:
        core = self.core
        if core is None:
            return
        await _sleep_until(core.deadline_ns, self.clock_ns)
        core.accumulator.stop(self.clock_ns(), "deadline")
        self._closed_event.set()

    def _plan_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._plan_error = error
            log.error("B3 PLAN scheduling failed", exc_info=(type(error), error, error.__traceback__))
            self._closed_event.set()

    def _send_message(
        self,
        message_type: MessageType,
        *,
        client_id: int,
        t_send_ns: int,
        body: Mapping[str, object],
    ) -> None:
        if self.transport is None:
            raise B3PlanError("B3 gateway transport is closed")
        core = self.core
        if core is None:
            raise B3PlanError("B3 round has not started")
        packet = encode_message(
            message_type,
            round_id=core.round_id,
            client_id=client_id,
            t_send_ns=t_send_ns,
            body=body,
            block_length=self.L,
        )
        try:
            if message_type is MessageType.PLAN:
                self.plan_send_attempts += 1
            self.transport.sendto(packet, self.endpoints[client_id])
        except OSError as exc:
            raise B3PlanError(f"failed to send {message_type.name}: {exc}") from exc
        if message_type is MessageType.PLAN:
            self.plan_local_handoffs += 1


@dataclass(frozen=True, slots=True)
class B3ClientRoundRecord:
    """Immutable outcome of one finished client round, published exactly once.

    ``transfer`` is None only when no PLAN was received. Otherwise it reflects
    every DATA that was actually handed to the socket: on completion it is the
    task's own result; on cancellation at round end or on a failed send it is
    the progress recorded up to that point.
    """

    round_id: int
    delivered_indices: tuple[int, ...]
    finished_by: str  # "round_end" | "round_end_timeout"
    transfer: B2ClientTransfer | None
    transfer_status: str  # "no_plan" | "completed" | "cancelled" | "failed"
    transfer_error: str | None
    ready_sent_ns: int | None
    plan_start_ns: int | None

    @property
    def packets_sent(self) -> int:
        return 0 if self.transfer is None else self.transfer.packets_sent


@dataclass(slots=True)
class _TransferProgress:
    """Sends recorded as they succeed, so partial transfers keep exact counts."""

    client_id: int
    round_id: int
    sent_indices: list[int] = field(default_factory=list)
    send_times_ns: list[int] = field(default_factory=list)
    bytes_sent: int = 0
    plan_start_ns: int | None = None

    def snapshot(self, completed_ns: int, stop_reason: str) -> B2ClientTransfer:
        return B2ClientTransfer(
            client_id=self.client_id,
            round_id=self.round_id,
            sent_indices=tuple(self.sent_indices),
            send_times_ns=tuple(self.send_times_ns),
            application_bytes_sent=self.bytes_sent,
            completed_ns=completed_ns,
            stop_reason=stop_reason,
        )


class B3UdpClient(asyncio.DatagramProtocol):
    """Wait for standard PLAN, then send its first planned energy-order blocks."""

    def __init__(
        self,
        core: B2ClientCore,
        *,
        gateway_addr: tuple[str, int],
        update_provider: Callable[[int], np.ndarray],
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not isinstance(core, B2ClientCore):
            raise TypeError("core must be a B2ClientCore")
        if not callable(update_provider):
            raise TypeError("update_provider must be callable")
        self.core = core
        self.gateway_addr = _endpoint(gateway_addr)
        self.update_provider = update_provider
        self.clock_ns = clock_ns
        self.sender = B2UdpClient(
            core,
            gateway_addr=self.gateway_addr,
            clock_ns=clock_ns,
            sleep=sleep,
        )
        self.transport: asyncio.DatagramTransport | None = None
        self._round_start_event = asyncio.Event()
        self._plan_event = asyncio.Event()
        self._plan_task: asyncio.Task[B2ClientTransfer] | None = None
        self._deadline_ns: int | None = None
        self._round_id: int | None = None
        self._plan_received = False
        self.last_transfer: B2ClientTransfer | None = None
        self.plan_start_ns: int | None = None
        self.rejected_datagrams = 0
        self._ready_handle: asyncio.TimerHandle | None = None
        self._round_end_handle: asyncio.TimerHandle | None = None
        self.last_finished_round_id: int | None = None
        self.last_delivered_indices: tuple[int, ...] = ()
        self.round_end_timeouts = 0
        self.ready_sent_ns: int | None = None
        self._plan_progress: _TransferProgress | None = None
        # Finished rounds, in completion order; each round is published once.
        self.round_records: dict[int, B3ClientRoundRecord] = {}
        self.last_round_record: B3ClientRoundRecord | None = None
        self._finished_rounds: asyncio.Queue[B3ClientRoundRecord] = asyncio.Queue()
        self._settling: set[asyncio.Task[B2ClientTransfer]] = set()

    @classmethod
    async def bind(
        cls,
        core: B2ClientCore,
        *,
        gateway_addr: tuple[str, int],
        local_addr: tuple[str, int] = ("127.0.0.1", 0),
        update_provider: Callable[[int], np.ndarray],
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> B3UdpClient:
        client = cls(
            core,
            gateway_addr=gateway_addr,
            update_provider=update_provider,
            clock_ns=clock_ns,
            sleep=sleep,
        )
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: client, local_addr=local_addr)
        return client

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.DatagramTransport, transport)
        self.sender.transport = self.transport

    def connection_lost(self, exc: Exception | None) -> None:
        self.transport = None
        self.sender.connection_lost(exc)

    def error_received(self, exc: Exception) -> None:
        self.sender.error_received(exc)

    def datagram_received(
        self,
        data: bytes,
        addr: tuple[str, int] | tuple[str, int, int, int],
    ) -> None:
        if _endpoint(addr[:2]) != self.gateway_addr:
            self.rejected_datagrams += 1
            return
        try:
            message = decode_message(
                data,
                block_length=self.core.error_feedback.L,
            )
            if message.client_id != self.core.client_id:
                raise B3PlanError("gateway message has the wrong client ID")
            if message.message_type is MessageType.ROUND_START:
                self._handle_round_start(message)
            elif message.message_type is MessageType.PLAN:
                self._handle_plan(message)
            elif message.message_type is MessageType.ROUND_END:
                self._handle_round_end(message)
            else:
                raise B3PlanError("unexpected B3 gateway message type")
        except (B3PlanError, WireProtocolError, B2TransferError, ValueError, TypeError):
            self.rejected_datagrams += 1
            log.debug("rejecting invalid B3 gateway datagram", exc_info=True)

    async def wait_for_plan(self) -> B2ClientTransfer:
        if self.core.round is None or self._deadline_ns is None:
            raise B3PlanError("client has no active B3 round")
        remaining_ns = self._deadline_ns - self.clock_ns()
        if remaining_ns <= 0:
            raise B3PlanError("PLAN was not received before the deadline")
        try:
            await asyncio.wait_for(
                self._plan_event.wait(),
                timeout=remaining_ns / _NANOSECONDS_PER_SECOND,
            )
        except asyncio.TimeoutError as exc:
            raise B3PlanError("PLAN was not received before the deadline") from exc
        task = self._plan_task
        if task is not None:
            # Wait for the task without letting its cancellation (at round end)
            # cancel this caller.
            await asyncio.wait({task})
            if task.cancelled():
                raise B3PlanError("PLAN transfer was cancelled at round end")
            self.last_transfer = task.result()  # re-raises a failed send
        if self.last_transfer is None:
            raise B3PlanError("PLAN task completed without a transfer result")
        return self.last_transfer

    async def wait_for_round_start(self, timeout: float = 5.0) -> int:
        """Wait for ROUND_START and return its round ID, bounded for launch failures."""
        if (
            self._round_id is not None
            and self.core.last_completed_round_id is not None
            and self._round_id <= self.core.last_completed_round_id
        ):
            self._round_start_event.clear()
        await asyncio.wait_for(self._round_start_event.wait(), timeout=timeout)
        if self._round_id is None:
            raise B3PlanError("ROUND_START event completed without a round ID")
        return self._round_id

    async def wait_for_round_record(self, timeout: float | None = None) -> B3ClientRoundRecord:
        """Next finished round (ROUND_END or 200 ms fallback) with its settled transfer."""
        return await asyncio.wait_for(self._finished_rounds.get(), timeout=timeout)

    async def wait_for_round_end(self, timeout: float | None = None) -> int:
        """Wait until ROUND_END (or the 200 ms post-deadline fallback) finishes a round."""
        return (await self.wait_for_round_record(timeout)).round_id

    async def close(self) -> None:
        for handle in (self._ready_handle, self._round_end_handle):
            if handle is not None:
                handle.cancel()
        self._ready_handle = self._round_end_handle = None
        tasks = set(self._settling)
        if self._plan_task is not None:
            tasks.add(self._plan_task)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._plan_task = None
        self._settling.clear()
        transport, self.transport = self.transport, None
        self.sender.transport = None
        if transport is not None:
            transport.close()

    def _handle_round_start(self, message) -> None:
        received_ns = self.clock_ns()  # rho_i is measured from ROUND_START receipt
        if self.core.round is not None:
            raise B3PlanError("ROUND_START arrived while a B3 round is active")
        if message.body["mode"] != int(TransferMode.PLAN):
            raise B3PlanError("B3 client requires ROUND_START mode PLAN")
        if message.body["L"] != self.core.error_feedback.L:
            raise B3PlanError("ROUND_START L does not match the B3 client")
        expected_k = (self.core.error_feedback.d + self.core.error_feedback.L - 1) // self.core.error_feedback.L
        if message.body["K"] != expected_k:
            raise B3PlanError("ROUND_START K does not match the B3 client")
        encoded = self.core.start_round(
            message.round_id,
            message.body["deadline_ns"],
            self.update_provider(message.round_id),
        )
        energies = tuple(float(value) for value in encoded.energies_sorted)
        self._round_id = message.round_id
        self._deadline_ns = message.body["deadline_ns"]
        self._plan_received = False
        self._round_start_event.clear()
        self._round_start_event.set()
        self._plan_event.clear()
        self.last_transfer = None
        self.plan_start_ns = None
        self.ready_sent_ns = None
        # A previous round's transfer settles through its own done-callback and
        # its own progress object; it can no longer touch this round's state.
        self._plan_task = None
        self._plan_progress = None
        round_id = message.round_id
        loop = asyncio.get_running_loop()
        if self._round_end_handle is not None:
            self._round_end_handle.cancel()
        # Section 9: without ROUND_END within 200 ms after D, every block is undelivered.
        fallback_s = max(0.0, (self._deadline_ns + ROUND_END_GRACE_NS - self.clock_ns()) / _NANOSECONDS_PER_SECOND)
        self._round_end_handle = loop.call_later(fallback_s, self._round_end_timeout, round_id)
        rho_us = int(message.body["rho_us"])
        if rho_us == 0:
            self._send_ready(round_id, energies)
        else:
            # Section 2.1: on ROUND_START, wait the readiness delay rho_i, then
            # READY. Time spent preparing the update counts toward rho_i.
            remaining_ns = received_ns + rho_us * 1_000 - self.clock_ns()
            self._ready_handle = loop.call_later(
                max(0.0, remaining_ns / _NANOSECONDS_PER_SECOND), self._send_ready, round_id, energies
            )

    def _send_ready(self, round_id: int, energies: tuple[float, ...]) -> None:
        self._ready_handle = None
        if self.core.round is None or self.core.round.round_id != round_id:
            return
        self.ready_sent_ns = self.clock_ns()
        self._send(
            encode_message(
                MessageType.READY,
                round_id=round_id,
                client_id=self.core.client_id,
                t_send_ns=self.ready_sent_ns,
                body={"K": len(energies), "energy": energies},
                block_length=self.core.error_feedback.L,
            )
        )

    def _handle_round_end(self, message) -> None:
        current = self.core.round
        if current is None or message.round_id != current.round_id:
            raise B3PlanError("ROUND_END does not match the active B3 round")
        if message.body["K"] != current.encoded.K:
            raise B3PlanError("ROUND_END K does not match the active B3 round")
        delivered = tuple(int(i) for i in indices_from_bitmap(message.body["bitmap"], current.encoded.K))
        if len(delivered) != message.body["n_delivered"]:
            raise B3PlanError("ROUND_END n_delivered disagrees with its bitmap")
        self._finish_round(message.round_id, delivered, "round_end")

    def _round_end_timeout(self, round_id: int) -> None:
        self._round_end_handle = None
        if self.core.round is not None and self.core.round.round_id == round_id:
            self.round_end_timeouts += 1
            self._finish_round(round_id, (), "round_end_timeout")

    def _finish_round(self, round_id: int, delivered: tuple[int, ...], finished_by: str) -> None:
        """End the round now; publish its record once the PLAN transfer settles.

        Error feedback is applied immediately and never waits for the transfer.
        A transfer still running at round end is cancelled (the round is over,
        so further DATA would be late); every send it completed is already in
        its progress object, so counts stay exact. The record is published from
        the task's done-callback, exactly once, whatever the task's outcome.
        """
        if self.core.round is None or self.core.round.round_id != round_id:
            return  # already finished (e.g. by an out-of-band delivery result)
        for handle in (self._ready_handle, self._round_end_handle):
            if handle is not None:
                handle.cancel()
        self._ready_handle = self._round_end_handle = None
        self.core.finish_round(round_id, delivered)
        self.last_finished_round_id = round_id
        self.last_delivered_indices = delivered
        task, progress = self._plan_task, self._plan_progress
        ready_sent_ns = self.ready_sent_ns
        if task is None or progress is None or progress.round_id != round_id:
            self._publish(B3ClientRoundRecord(round_id, delivered, finished_by, None, "no_plan", None,
                                              ready_sent_ns, None))
            return

        def settle(done: asyncio.Task[B2ClientTransfer]) -> None:
            self._settling.discard(done)
            transfer, status, error = self._settled_transfer(done, progress)
            self._publish(B3ClientRoundRecord(round_id, delivered, finished_by, transfer, status, error,
                                              ready_sent_ns, progress.plan_start_ns))

        if task.done():
            settle(task)
        else:
            self._settling.add(task)
            task.add_done_callback(settle)
            task.cancel()

    def _settled_transfer(
        self, task: asyncio.Task[B2ClientTransfer], progress: _TransferProgress
    ) -> tuple[B2ClientTransfer, str, str | None]:
        if task.cancelled():
            return progress.snapshot(self.clock_ns(), "round_end"), "cancelled", None
        error = task.exception()  # also marks the exception as retrieved
        if error is not None:
            return progress.snapshot(self.clock_ns(), "send_failed"), "failed", f"{type(error).__name__}: {error}"
        return task.result(), "completed", None

    def _publish(self, record: B3ClientRoundRecord) -> None:
        if record.round_id in self.round_records:
            raise B3PlanError(f"round {record.round_id} was already finalized")
        self.round_records[record.round_id] = record
        self.last_round_record = record
        if record.transfer is not None and record.round_id == self._round_id:
            # Never let a late-settling record overwrite a newer round's field.
            self.last_transfer = record.transfer
        self._finished_rounds.put_nowait(record)

    def _handle_plan(self, message) -> None:
        if (
            self.core.round is None
            or message.round_id != self._round_id
            or message.round_id != self.core.round.round_id
        ):
            raise B3PlanError("PLAN does not match the active B3 round")
        if self._plan_received:
            raise B3PlanError("duplicate PLAN")
        blocks = message.body["k"]
        rate = message.body["rate_bps"]
        if blocks > self.core.round.encoded.K:
            raise B3PlanError("PLAN k exceeds K")
        if blocks and rate == 0:
            raise B3PlanError("positive PLAN k requires a positive rate")
        if self._deadline_ns is None or self.clock_ns() >= self._deadline_ns:
            raise B3PlanError("PLAN arrived at or after the deadline")
        self._plan_received = True
        self.plan_start_ns = None
        self._plan_event.set()
        progress = _TransferProgress(self.core.client_id, message.round_id)
        self._plan_progress = progress
        self._plan_task = asyncio.create_task(
            self._send_plan_blocks(message.round_id, blocks, rate, progress),
            name=f"b3-client-{self.core.client_id}-send",
        )

    def _set_plan_start(self, progress: _TransferProgress, t_ns: int) -> None:
        progress.plan_start_ns = t_ns
        if progress.round_id == self._round_id:  # never write into a newer round
            self.plan_start_ns = t_ns

    async def _send_plan_blocks(
        self,
        round_id: int,
        block_count: int,
        rate_bps: int,
        progress: _TransferProgress | None = None,
    ) -> B2ClientTransfer:
        progress = progress if progress is not None else _TransferProgress(self.core.client_id, round_id)
        deadline_ns = self._deadline_ns
        current = self.core.round
        if deadline_ns is None or current is None:
            raise B3PlanError("PLAN send has no active prepared round")
        t_start_ns = self.clock_ns()
        self._set_plan_start(progress, t_start_ns)
        if block_count == 0:
            return progress.snapshot(self.clock_ns(), "all_blocks_sent")
        if rate_bps <= 0:
            raise B3PlanError("positive PLAN k requires a positive rate")
        transport = self.sender.transport
        if transport is None:
            raise B2TransferError("B3 UDP client is not bound")
        if self.sender._sending:
            raise B2TransferError("B3 UDP client is already sending a round")

        self.sender._sending = True
        # Shared with the round's record: each successful send is appended
        # immediately (no await in between), so cancellation keeps exact counts.
        sent_indices = progress.sent_indices
        send_times = progress.send_times_ns
        size = data_frame_size_bytes(self.core.error_feedback.L)
        stop_reason = "all_blocks_sent"
        try:
            while len(sent_indices) < block_count:
                self.sender._raise_transport_error()
                now_ns = self.clock_ns()
                if now_ns >= deadline_ns:
                    stop_reason = "deadline"
                    break
                if not send_times:
                    t_start_ns = now_ns
                    self._set_plan_start(progress, t_start_ns)
                send_pos = len(sent_indices) + 1
                sent_by_now = paced_blocks_sent(
                    now_ns,
                    t_start_ns=t_start_ns,
                    k=block_count,
                    rate_bps=rate_bps,
                    block_size_bytes=size,
                )
                target_ns = pacing_send_time_ns(
                    send_pos,
                    t_start_ns=t_start_ns,
                    k=block_count,
                    rate_bps=rate_bps,
                    block_size_bytes=size,
                )
                if send_times:
                    target_ns = max(
                        target_ns,
                        pacing_send_time_ns(
                            2,
                            t_start_ns=send_times[-1],
                            k=max(block_count, 2),
                            rate_bps=rate_bps,
                            block_size_bytes=size,
                        ),
                    )
                if sent_by_now < send_pos or now_ns < target_ns:
                    wait_ns = min(target_ns, deadline_ns) - now_ns
                    if wait_ns <= 0:
                        continue
                    await self.sender._sleep(wait_ns / _NANOSECONDS_PER_SECOND)
                    continue

                block_idx = int(current.encoded.order[send_pos - 1])
                payload, n_coords = block_payload_f16(current.encoded, block_idx)
                packet = encode_data_packet(
                    round_id,
                    self.core.client_id,
                    block_idx,
                    n_coords,
                    payload,
                    send_pos=send_pos,
                    t_send_ns=now_ns,
                )
                try:
                    transport.sendto(packet, self.gateway_addr)
                except OSError as exc:
                    raise B2TransferError(f"B3 UDP send failed: {exc}") from exc
                sent_indices.append(block_idx)
                send_times.append(now_ns)
                progress.bytes_sent += len(packet)

            self.sender._raise_transport_error()
            return progress.snapshot(self.clock_ns(), stop_reason)
        finally:
            self.sender._sending = False

    def _send(self, packet: bytes) -> None:
        if self.transport is None:
            raise B3PlanError("B3 client transport is closed")
        try:
            self.transport.sendto(packet, self.gateway_addr)
        except OSError as exc:
            raise B3PlanError(f"B3 UDP send failed: {exc}") from exc


def _validate_delivery_ratio_init(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("delivery_ratio_init must be finite and in [0, 1]")
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("delivery_ratio_init must be finite and in [0, 1]") from exc
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("delivery_ratio_init must be finite and in [0, 1]")
    return ratio


def _rate_to_wire(rate_bps: float) -> int:
    """Round to nearest whole bit/s (ties to even) for PLAN's u32 field."""
    if isinstance(rate_bps, bool):
        raise B3PlanError("estimated rate must be finite and non-negative")
    try:
        rate = float(rate_bps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise B3PlanError("estimated rate must be finite and non-negative") from exc
    if not math.isfinite(rate) or rate < 0:
        raise B3PlanError("estimated rate must be finite and non-negative")
    rounded = round(rate)
    if rounded > 0xFFFFFFFF:
        raise B3PlanError("rounded rate_bps is outside the PLAN wire range")
    return rounded


def _endpoint(endpoint: tuple[object, ...]) -> tuple[str, int]:
    if (
        not isinstance(endpoint, tuple)
        or len(endpoint) < 2
        or not isinstance(endpoint[0], str)
        or isinstance(endpoint[1], bool)
        or not isinstance(endpoint[1], int)
        or not 0 <= endpoint[1] <= 0xFFFF
    ):
        raise ValueError("UDP endpoint must be a (host, port) tuple")
    return endpoint[0], endpoint[1]


async def _sleep_until(target_ns: int, clock_ns: Callable[[], int]) -> None:
    while True:
        remaining_ns = target_ns - clock_ns()
        if remaining_ns <= 0:
            return
        await asyncio.sleep(remaining_ns / _NANOSECONDS_PER_SECOND)


__all__ = [
    "B3ClientPlan",
    "B3GatewayCore",
    "B3PlanError",
    "B3RoundResult",
    "B3UdpClient",
    "B3UdpGateway",
    "apply_delivery_result",
]
