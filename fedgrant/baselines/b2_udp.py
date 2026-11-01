"""B2 UDP push baseline (Section 10), using standard sentinel DATA messages."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np

from fedgrant.common.constants import DATA_GRANT_ID, MessageType
from fedgrant.common.wire import WireProtocolError, decode_message, encode_message
from fedgrant.encoder.blocks import EncodedUpdate, block_payload_f16, encode, n_blocks
from fedgrant.encoder.ef import ErrorFeedback, bitmap_from_indices
from fedgrant.policy.constants import L as L_DEFAULT

log = logging.getLogger(__name__)

B2_PORT = 9101
B2_MAX_DATAGRAM_BYTES = 1472
# b_bar_i accounts for the complete Ethernet + IPv4 + UDP frame.
B2_WIRE_OVERHEAD_BYTES = 42
_NANOSECONDS_PER_SECOND = 1_000_000_000
_MAX_BLOCK_LENGTH = (B2_MAX_DATAGRAM_BYTES - 32) // 2


class B2ProtocolError(ValueError):
    """Malformed or invalid B2 UDP packet."""


class B2TransferError(RuntimeError):
    """B2 client or gateway transfer failed."""


def encode_data_packet(
    round_id: int,
    client_id: int,
    block_idx: int,
    n_coords: int,
    payload: np.ndarray,
    *,
    send_pos: int = 1,
    t_send_ns: int = 0,
) -> bytes:
    """Encode one standard PUSH/PLAN DATA message for the independent B2 sender."""
    _validate_block_length(int(payload.size) if isinstance(payload, np.ndarray) else 1)
    if isinstance(n_coords, bool) or not isinstance(n_coords, (int, np.integer)):
        raise B2ProtocolError("n_coords must be an integer")
    if not isinstance(payload, np.ndarray) or payload.dtype != np.float16 or payload.ndim != 1:
        raise B2ProtocolError("payload must be a one-dimensional float16 ndarray")
    L = int(payload.size)
    _validate_block_length(L)
    if not 1 <= int(n_coords) <= L:
        raise B2ProtocolError("n_coords must be in [1, L]")
    if not np.isfinite(payload).all():
        raise B2ProtocolError("payload values must be finite")
    if np.any(payload[int(n_coords) :] != 0):
        raise B2ProtocolError("coordinates beyond n_coords must be zero padded")
    try:
        return encode_message(
            MessageType.DATA,
            round_id=round_id,
            client_id=client_id,
            t_send_ns=t_send_ns,
            block_length=L,
            body={
                "grant_id": DATA_GRANT_ID,
                "send_pos": send_pos,
                "block_idx": block_idx,
                "n_coords": int(n_coords),
                "payload": tuple(float(value) for value in payload),
            },
        )
    except WireProtocolError as exc:
        raise B2ProtocolError(str(exc)) from exc


def decode_data_packet(
    packet: bytes | bytearray | memoryview,
    L: int,
) -> tuple[int, int, int, int, int, np.ndarray]:
    """Decode and validate exactly one fixed-size B2 datagram."""
    _validate_block_length(L)
    try:
        message = decode_message(packet, block_length=L)
    except WireProtocolError as exc:
        raise B2ProtocolError(str(exc)) from exc
    expected_size = 32 + 2 * L
    if len(packet) != expected_size or len(packet) > B2_MAX_DATAGRAM_BYTES:
        raise B2ProtocolError(f"B2 DATA datagram must be exactly {expected_size} bytes")
    if message.message_type is not MessageType.DATA:
        raise B2ProtocolError("B2 packet must be a DATA message")
    if message.body["grant_id"] != DATA_GRANT_ID:
        raise B2ProtocolError("B2 DATA must use the PUSH/PLAN grant sentinel")
    send_pos = message.body["send_pos"]
    block_idx = message.body["block_idx"]
    n_coords = message.body["n_coords"]
    if not 1 <= n_coords <= L:
        raise B2ProtocolError("n_coords must be in [1, L]")
    payload = np.asarray(message.body["payload"], dtype=np.float16)
    if not np.isfinite(payload).all():
        raise B2ProtocolError("payload values must be finite")
    if np.any(payload[n_coords:] != 0):
        raise B2ProtocolError("coordinates beyond n_coords must be zero padded")
    return message.round_id, message.client_id, send_pos, block_idx, n_coords, payload


@dataclass(frozen=True, slots=True)
class B2ClientRound:
    round_id: int
    deadline_ns: int
    encoded: EncodedUpdate
    u: np.ndarray


class B2ClientCore:
    """Prepare B2 updates and finalize error feedback from gateway delivery data."""

    def __init__(self, client_id: int, *, d: int, L: int = L_DEFAULT) -> None:
        _require_uint(client_id, "client_id", 0xFFFF)
        _require_uint(d, "d", 0xFFFFFFFF, minimum=1)
        _validate_block_length(L)
        self.client_id = client_id
        self.error_feedback = ErrorFeedback(d, L)
        self.round: B2ClientRound | None = None
        self.last_completed_round_id = 0

    def start_round(self, round_id: int, deadline_ns: int, delta: np.ndarray) -> EncodedUpdate:
        if self.round is not None:
            raise B2TransferError("cannot start B2 round while another round is active")
        _require_uint(round_id, "round_id", 0xFFFFFFFF, minimum=1)
        _require_uint(deadline_ns, "deadline_ns", 0xFFFFFFFFFFFFFFFF, minimum=1)
        if round_id <= self.last_completed_round_id:
            raise B2TransferError("round_id must increase monotonically")
        u = self.error_feedback.prepare(delta)
        encoded = encode(u, self.error_feedback.L)
        self.round = B2ClientRound(round_id, deadline_ns, encoded, u.copy())
        return encoded

    def finish_round(self, round_id: int, delivered_indices: Iterable[int]) -> np.ndarray:
        current = self._require_round(round_id)
        indices = tuple(delivered_indices)
        if len(set(indices)) != len(indices):
            raise B2TransferError("delivered block indices must be unique")
        bitmap = bitmap_from_indices(indices, current.encoded.K)
        reconstructed = self.error_feedback.finish_round(current.u, bitmap)
        self.last_completed_round_id = round_id
        self.round = None
        return reconstructed

    def _require_round(self, round_id: int) -> B2ClientRound:
        if self.round is None or self.round.round_id != round_id:
            raise B2TransferError("round result does not match the active B2 round")
        return self.round


@dataclass(frozen=True, slots=True)
class B2ClientTransfer:
    client_id: int
    round_id: int
    sent_indices: tuple[int, ...]
    send_times_ns: tuple[int, ...]
    application_bytes_sent: int
    completed_ns: int
    stop_reason: str

    @property
    def packets_sent(self) -> int:
        return len(self.sent_indices)


class B2UdpClient(asyncio.DatagramProtocol):
    """Send every encoded block once in energy order, paced by ``rate_bps``."""

    def __init__(
        self,
        core: B2ClientCore,
        *,
        gateway_addr: tuple[str, int],
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not isinstance(core, B2ClientCore):
            raise TypeError("core must be a B2ClientCore")
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self.core = core
        self.gateway_addr = _normalise_endpoint(gateway_addr)
        self.clock_ns = clock_ns
        self._sleep = sleep
        self.transport: asyncio.DatagramTransport | None = None
        self._sending = False
        self._transport_error: Exception | None = None

    @classmethod
    async def bind(
        cls,
        core: B2ClientCore,
        *,
        gateway_addr: tuple[str, int],
        local_addr: tuple[str, int] = ("0.0.0.0", 0),
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> B2UdpClient:
        client = cls(core, gateway_addr=gateway_addr, clock_ns=clock_ns)
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: client, local_addr=local_addr)
        return client

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.DatagramTransport, transport)

    def connection_lost(self, exc: Exception | None) -> None:
        self.transport = None
        if self._sending:
            self._transport_error = exc or ConnectionError(
                "B2 UDP transport closed during a round"
            )

    def error_received(self, exc: Exception) -> None:
        self._transport_error = exc
        log.warning("B2 UDP client socket error: %r", exc)

    async def send_round(
        self,
        *,
        round_id: int,
        deadline_ns: int,
        delta: np.ndarray,
        rate_bps: float,
    ) -> B2ClientTransfer:
        rate = _validate_rate(rate_bps)
        if self.transport is None:
            raise B2TransferError("B2 UDP client is not bound")
        if self._sending:
            raise B2TransferError("B2 UDP client is already sending a round")
        encoded = self.core.start_round(round_id, deadline_ns, delta)
        return await self._send_encoded(
            round_id,
            deadline_ns,
            encoded,
            rate,
            encoded.K,
        )

    async def send_prepared(
        self,
        *,
        round_id: int,
        deadline_ns: int,
        block_count: int,
        rate_bps: float,
    ) -> B2ClientTransfer:
        """Send the first ``block_count`` positions of an already-prepared round."""
        if self.transport is None:
            raise B2TransferError("B2 UDP client is not bound")
        if self._sending:
            raise B2TransferError("B2 UDP client is already sending a round")
        rate = _validate_rate(rate_bps)
        current = self.core._require_round(round_id)
        if current.deadline_ns != deadline_ns:
            raise B2TransferError("deadline does not match the prepared B2 round")
        _require_uint(block_count, "block_count", current.encoded.K)
        return await self._send_encoded(
            round_id,
            deadline_ns,
            current.encoded,
            rate,
            block_count,
        )

    async def _send_encoded(
        self,
        round_id: int,
        deadline_ns: int,
        encoded: EncodedUpdate,
        rate: float,
        block_count: int,
    ) -> B2ClientTransfer:
        self._sending = True
        sent_indices: list[int] = []
        send_times: list[int] = []
        bytes_sent = 0
        next_send_ns = self.clock_ns()
        stop_reason = "all_blocks_sent"
        try:
            for block_idx_value in encoded.order[:block_count]:
                now_ns = self.clock_ns()
                target_ns = max(next_send_ns, now_ns)
                wait_ns = target_ns - now_ns
                if wait_ns > 0:
                    remaining_ns = deadline_ns - now_ns
                    if remaining_ns <= 0:
                        stop_reason = "deadline"
                        break
                    await self._sleep(
                        min(wait_ns, remaining_ns) / _NANOSECONDS_PER_SECOND
                    )
                now_ns = self.clock_ns()
                self._raise_transport_error()
                if now_ns >= deadline_ns:
                    stop_reason = "deadline"
                    break
                block_idx = int(block_idx_value)
                payload, n_coords = block_payload_f16(encoded, block_idx)
                packet = encode_data_packet(
                    round_id,
                    self.core.client_id,
                    block_idx,
                    n_coords,
                    payload,
                    send_pos=len(sent_indices) + 1,
                    t_send_ns=now_ns,
                )
                try:
                    self.transport.sendto(packet, self.gateway_addr)
                except OSError as exc:
                    raise B2TransferError(f"B2 UDP send failed: {exc}") from exc
                sent_indices.append(block_idx)
                send_times.append(now_ns)
                bytes_sent += len(packet)
                serialization_ns = math.ceil(
                    (len(packet) + B2_WIRE_OVERHEAD_BYTES)
                    * 8
                    * _NANOSECONDS_PER_SECOND
                    / rate
                )
                next_send_ns = now_ns + serialization_ns
            self._raise_transport_error()
            return B2ClientTransfer(
                client_id=self.core.client_id,
                round_id=round_id,
                sent_indices=tuple(sent_indices),
                send_times_ns=tuple(send_times),
                application_bytes_sent=bytes_sent,
                completed_ns=self.clock_ns(),
                stop_reason=stop_reason,
            )
        finally:
            self._sending = False

    async def close(self) -> None:
        transport, self.transport = self.transport, None
        if transport is not None:
            transport.close()

    def _raise_transport_error(self) -> None:
        if self._transport_error is not None:
            raise B2TransferError(f"B2 UDP socket error: {self._transport_error}") from self._transport_error


@dataclass(slots=True)
class _ClientReceived:
    delivered: dict[int, tuple[int, np.ndarray]] = field(default_factory=dict)
    receive_order: list[int] = field(default_factory=list)
    send_positions: set[int] = field(default_factory=set)
    bytes_received: int = 0


@dataclass(frozen=True, slots=True)
class B2ClientRoundResult:
    client_id: int
    delivered_indices: tuple[int, ...]
    receive_order: tuple[int, ...]
    delivered_bitmap: bytes
    n_coords: Mapping[int, int]
    decoded_update: np.ndarray
    packets_received: int
    bytes_received: int


class Aggregator(Protocol):
    def __call__(self, updates: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray: ...


class B2RoundAccumulator:
    """Validate B2 packets and create zero-filled client updates for one round."""

    def __init__(
        self,
        *,
        round_id: int,
        client_ids: Sequence[int],
        start_ns: int,
        deadline_ns: int,
        d: int,
        L: int = L_DEFAULT,
        client_weights: Mapping[int, float],
    ) -> None:
        _require_uint(round_id, "round_id", 0xFFFFFFFF, minimum=1)
        _require_uint(start_ns, "start_ns", 0xFFFFFFFFFFFFFFFF)
        _require_uint(deadline_ns, "deadline_ns", 0xFFFFFFFFFFFFFFFF, minimum=1)
        _require_uint(d, "d", 0xFFFFFFFF, minimum=1)
        _validate_block_length(L)
        if deadline_ns <= start_ns:
            raise ValueError("deadline_ns must be after start_ns")
        ids = tuple(_require_uint(cid, "client_id", 0xFFFF) for cid in client_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("client_ids must be a non-empty sequence of unique IDs")
        if set(client_weights) != set(ids):
            raise ValueError("client_weights must contain exactly the configured clients")
        weights = np.asarray([client_weights[cid] for cid in ids], dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("client weights must be finite, non-negative, and sum to 1")
        self.round_id = round_id
        self.client_ids = ids
        self.start_ns = start_ns
        self.deadline_ns = deadline_ns
        self.d = d
        self.L = L
        self.K = n_blocks(d, L)
        if self.K > 0x10000:
            raise ValueError("B2 block count exceeds the u16 block index space")
        self.client_weights = {cid: float(client_weights[cid]) for cid in ids}
        self.clients = {cid: _ClientReceived() for cid in ids}
        self.rejected_malformed_packets = 0
        self.rejected_unknown_clients = 0
        self.rejected_rounds = 0
        self.rejected_early_packets = 0
        self.rejected_late_packets = 0
        self.rejected_duplicates = 0
        self._active = True
        self.close_reason: str | None = None
        self.completed_ns: int | None = None

    def accept_packet(self, packet: bytes, received_ns: int) -> None:
        try:
            round_id, client_id, send_pos, block_idx, n_coords, payload = decode_data_packet(packet, self.L)
        except B2ProtocolError:
            self.rejected_malformed_packets += 1
            raise
        if round_id != self.round_id:
            self.rejected_rounds += 1
            raise B2ProtocolError("stale or mismatched B2 round")
        if client_id not in self.clients:
            self.rejected_unknown_clients += 1
            raise B2ProtocolError("unknown B2 client ID")
        if received_ns < self.start_ns:
            self.rejected_early_packets += 1
            raise B2ProtocolError("B2 block arrived before the round start")
        if not self._active or received_ns > self.deadline_ns:
            self.rejected_late_packets += 1
            raise B2ProtocolError("B2 block arrived after the round deadline")
        if block_idx >= self.K:
            self.rejected_malformed_packets += 1
            raise B2ProtocolError("block_idx is outside the configured update")
        if send_pos > self.K:
            self.rejected_malformed_packets += 1
            raise B2ProtocolError("send_pos is outside the configured update")
        expected_n = min(self.L, self.d - block_idx * self.L)
        if n_coords != expected_n:
            self.rejected_malformed_packets += 1
            raise B2ProtocolError("n_coords does not match the original block")
        state = self.clients[client_id]
        if block_idx in state.delivered:
            self.rejected_duplicates += 1
            raise B2ProtocolError("duplicate B2 block")
        if send_pos in state.send_positions:
            self.rejected_duplicates += 1
            raise B2ProtocolError("duplicate B2 send position")
        state.delivered[block_idx] = (n_coords, payload)
        state.receive_order.append(block_idx)
        state.send_positions.add(send_pos)
        state.bytes_received += len(packet)

    def stop(self, completed_ns: int, reason: str = "deadline") -> None:
        if not self._active:
            return
        self._active = False
        self.completed_ns = completed_ns
        self.close_reason = reason

    def result(self) -> B2RoundResult:
        if self._active:
            raise B2TransferError("cannot build B2 result before the round closes")
        client_results: dict[int, B2ClientRoundResult] = {}
        for client_id in self.client_ids:
            state = self.clients[client_id]
            indices = tuple(sorted(state.delivered))
            update = np.zeros(self.d, dtype=np.float32)
            n_coords: dict[int, int] = {}
            for block_idx, (count, payload) in state.delivered.items():
                start = block_idx * self.L
                update[start : start + count] = payload[:count].astype(np.float32)
                n_coords[block_idx] = count
            client_results[client_id] = B2ClientRoundResult(
                client_id=client_id,
                delivered_indices=indices,
                receive_order=tuple(state.receive_order),
                delivered_bitmap=bitmap_from_indices(indices, self.K),
                n_coords=n_coords,
                decoded_update=update,
                packets_received=len(indices),
                bytes_received=state.bytes_received,
            )
        return B2RoundResult(
            round_id=self.round_id,
            start_ns=self.start_ns,
            deadline_ns=self.deadline_ns,
            completed_ns=self.completed_ns if self.completed_ns is not None else self.deadline_ns,
            client_results=client_results,
            client_weights=self.client_weights.copy(),
            L=self.L,
            d=self.d,
            close_reason=self.close_reason or "round_stopped",
            rejected_malformed_packets=self.rejected_malformed_packets,
            rejected_unknown_clients=self.rejected_unknown_clients,
            rejected_rounds=self.rejected_rounds,
            rejected_early_packets=self.rejected_early_packets,
            rejected_late_packets=self.rejected_late_packets,
            rejected_duplicates=self.rejected_duplicates,
        )


@dataclass(frozen=True, slots=True)
class B2RoundResult:
    round_id: int
    start_ns: int
    deadline_ns: int
    completed_ns: int
    client_results: Mapping[int, B2ClientRoundResult]
    client_weights: Mapping[int, float]
    L: int
    d: int
    close_reason: str
    rejected_malformed_packets: int
    rejected_unknown_clients: int
    rejected_rounds: int
    rejected_early_packets: int
    rejected_late_packets: int
    rejected_duplicates: int

    @property
    def K(self) -> int:
        return n_blocks(self.d, self.L)

    @property
    def delivered_blocks(self) -> int:
        return sum(result.packets_received for result in self.client_results.values())

    @property
    def completed_at_deadline(self) -> bool:
        return self.close_reason == "deadline"

    @property
    def client_n_coords(self) -> dict[int, dict[int, int]]:
        return {
            client_id: dict(result.n_coords)
            for client_id, result in self.client_results.items()
        }

    def aggregate(self, aggregator: Aggregator) -> np.ndarray:
        """Delegate zero-filled client updates to the existing aggregation provider."""
        if not callable(aggregator):
            raise TypeError("aggregator must be callable")
        client_ids = tuple(sorted(self.client_results))
        updates = [self.client_results[cid].decoded_update.copy() for cid in client_ids]
        weights = np.asarray([self.client_weights[cid] for cid in client_ids], dtype=np.float32)
        aggregate = aggregator(updates, weights)
        if not isinstance(aggregate, np.ndarray) or aggregate.dtype != np.float32 or aggregate.shape != (self.d,):
            raise B2TransferError("aggregator must return a float32 vector with the model dimension")
        if not np.isfinite(aggregate).all():
            raise B2TransferError("aggregator returned non-finite values")
        return aggregate


def apply_delivery_result(core: B2ClientCore, result: B2RoundResult) -> np.ndarray:
    """Finalize this client's residual with the gateway's actual delivery set."""
    if not isinstance(core, B2ClientCore) or not isinstance(result, B2RoundResult):
        raise TypeError("core and result must be B2ClientCore and B2RoundResult instances")
    if core.round is None or core.round.round_id != result.round_id:
        raise B2TransferError("gateway result does not match the client's active round")
    client_result = result.client_results.get(core.client_id)
    if client_result is None:
        raise B2TransferError("gateway result does not include this client")
    return core.finish_round(result.round_id, client_result.delivered_indices)


class B2UdpGateway(asyncio.DatagramProtocol):
    """Receive B2 datagrams until D and reject any packet received after D."""

    def __init__(
        self,
        accumulator: B2RoundAccumulator,
        *,
        host: str = "0.0.0.0",
        port: int = B2_PORT,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(accumulator, B2RoundAccumulator):
            raise TypeError("accumulator must be a B2RoundAccumulator")
        _require_uint(port, "port", 0xFFFF)
        self.accumulator = accumulator
        self.host = host
        self.port = port
        self.clock_ns = clock_ns
        self.transport: asyncio.DatagramTransport | None = None
        self._deadline_task: asyncio.Task[None] | None = None
        self._closed_event = asyncio.Event()

    async def start(self) -> B2UdpGateway:
        if self.transport is not None:
            raise B2TransferError("B2 gateway is already started")
        if not self.accumulator._active:
            raise B2TransferError("B2 round is already closed")
        if self.accumulator.deadline_ns <= self.clock_ns():
            raise B2TransferError("B2 deadline must be in the future")
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: self,
            local_addr=(self.host, self.port),
        )
        self._deadline_task = asyncio.create_task(self._close_at_deadline())
        return self

    @property
    def address(self) -> tuple[str, int]:
        if self.transport is None:
            raise B2TransferError("B2 gateway is not started")
        address = self.transport.get_extra_info("sockname")
        if not isinstance(address, tuple) or len(address) < 2:
            raise B2TransferError("B2 gateway socket has no bound address")
        return str(address[0]), int(address[1])

    def datagram_received(
        self,
        data: bytes,
        addr: tuple[str, int] | tuple[str, int, int, int],
    ) -> None:
        del addr
        try:
            self.accumulator.accept_packet(data, self.clock_ns())
        except B2ProtocolError:
            log.debug("rejecting invalid B2 datagram", exc_info=True)

    def error_received(self, exc: Exception) -> None:
        log.error("B2 UDP gateway socket error: %r", exc)

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.DatagramTransport, transport)

    def connection_lost(self, exc: Exception | None) -> None:
        self.transport = None
        if self.accumulator._active:
            self.accumulator.stop(self.clock_ns(), "socket_closed")
        self._closed_event.set()

    async def wait(self) -> B2RoundResult:
        if self._deadline_task is None:
            raise B2TransferError("B2 gateway is not started")
        await self._closed_event.wait()
        return self.accumulator.result()

    async def close(self) -> None:
        if self.accumulator._active:
            self.accumulator.stop(self.clock_ns(), "gateway_shutdown")
        self._closed_event.set()
        task, self._deadline_task = self._deadline_task, None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        transport, self.transport = self.transport, None
        if transport is not None:
            transport.close()

    async def _close_at_deadline(self) -> None:
        remaining_ns = self.accumulator.deadline_ns - self.clock_ns()
        if remaining_ns > 0:
            await asyncio.sleep(remaining_ns / _NANOSECONDS_PER_SECOND)
        while self.clock_ns() < self.accumulator.deadline_ns:
            await asyncio.sleep(
                (self.accumulator.deadline_ns - self.clock_ns()) / _NANOSECONDS_PER_SECOND
            )
        self.accumulator.stop(self.clock_ns(), "deadline")
        self._closed_event.set()


def _validate_block_length(L: int) -> None:
    _require_uint(L, "L", 0xFFFF, minimum=1)
    if L > _MAX_BLOCK_LENGTH:
        raise ValueError(
            f"L={L} exceeds the B2 safe UDP payload limit of {_MAX_BLOCK_LENGTH} coordinates"
        )


def _validate_rate(rate_bps: float) -> float:
    if isinstance(rate_bps, bool):
        raise ValueError("rate_bps must be finite and positive")
    try:
        rate = float(rate_bps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rate_bps must be finite and positive") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate_bps must be finite and positive")
    return rate


def _require_uint(
    value: object,
    name: str,
    maximum: int,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _normalise_endpoint(endpoint: tuple[str, int]) -> tuple[str, int]:
    if (
        not isinstance(endpoint, tuple)
        or len(endpoint) != 2
        or not isinstance(endpoint[0], str)
        or isinstance(endpoint[1], bool)
        or not isinstance(endpoint[1], int)
        or not 0 <= endpoint[1] <= 0xFFFF
    ):
        raise ValueError("UDP endpoint must be a (host, port) tuple")
    return endpoint


__all__ = [
    "B2ClientCore",
    "B2ClientRoundResult",
    "B2ClientTransfer",
    "B2ProtocolError",
    "B2RoundAccumulator",
    "B2RoundResult",
    "B2TransferError",
    "B2UdpClient",
    "B2UdpGateway",
    "apply_delivery_result",
    "decode_data_packet",
    "encode_data_packet",
]
