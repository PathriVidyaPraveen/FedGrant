"""B1 TCP baseline (Section 10).

The TCP connection carries a B1-only identity preamble followed by fixed-size
block records from Section 8: ``block_idx:u16, n_coords:u16, payload:f16*L``.
The preamble does not alter the six-message UDP protocol. No congestion-control
socket option is set, so Linux's configured default (CUBIC in the specified
Mininet environment) remains in effect.
"""

from __future__ import annotations

import asyncio
import struct
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from fedgrant.encoder.blocks import EncodedUpdate, block_payload_f16, encode, n_blocks
from fedgrant.encoder.ef import ErrorFeedback, bitmap_from_indices
from fedgrant.policy.constants import L as L_DEFAULT

B1_PORT = 9100
_HELLO = struct.Struct("!4sIH")
_HELLO_MAGIC = b"FGB1"
_BLOCK_PREFIX = struct.Struct("!HH")


class B1ProtocolError(ValueError):
    """Malformed B1 handshake or block frame."""


class B1TransferError(RuntimeError):
    """B1 TCP transport could not complete its bounded transfer."""


def encode_hello(round_id: int, client_id: int) -> bytes:
    """Encode the B1-only connection identity preamble."""
    _require_uint(round_id, "round_id", 0xFFFFFFFF, minimum=1)
    _require_uint(client_id, "client_id", 0xFFFF)
    return _HELLO.pack(_HELLO_MAGIC, round_id, client_id)


def decode_hello(frame: bytes | bytearray | memoryview) -> tuple[int, int]:
    """Decode exactly one B1 identity preamble."""
    raw = bytes(frame)
    if len(raw) != _HELLO.size:
        raise B1ProtocolError("B1 hello has an invalid length")
    magic, round_id, client_id = _HELLO.unpack(raw)
    if magic != _HELLO_MAGIC:
        raise B1ProtocolError("invalid B1 hello magic")
    if round_id < 1:
        raise B1ProtocolError("round_id must start at 1")
    return round_id, client_id


def encode_block_frame(block_idx: int, n_coords: int, payload: np.ndarray) -> bytes:
    """Encode a spec-format B1 block record with big-endian IEEE binary16 values."""
    _require_uint(block_idx, "block_idx", 0xFFFF)
    if isinstance(n_coords, bool) or not isinstance(n_coords, (int, np.integer)):
        raise B1ProtocolError("n_coords must be an integer")
    if not isinstance(payload, np.ndarray) or payload.dtype != np.float16 or payload.ndim != 1:
        raise B1ProtocolError("payload must be a one-dimensional float16 ndarray")
    if payload.size < 1 or payload.size > 0xFFFF:
        raise B1ProtocolError("payload length must be in [1, 65535]")
    if not 1 <= int(n_coords) <= payload.size:
        raise B1ProtocolError("n_coords must be in [1, payload length]")
    if not np.isfinite(payload).all():
        raise B1ProtocolError("payload values must be finite")
    if np.any(payload[int(n_coords) :] != 0):
        raise B1ProtocolError("coordinates beyond n_coords must be zero padded")
    return _BLOCK_PREFIX.pack(block_idx, int(n_coords)) + payload.astype(">f2", copy=False).tobytes()


def decode_block_frame(frame: bytes | bytearray | memoryview, L: int) -> tuple[int, int, np.ndarray]:
    """Decode exactly one fixed-size B1 block record."""
    _require_uint(L, "L", 0xFFFF, minimum=1)
    raw = bytes(frame)
    expected = _BLOCK_PREFIX.size + 2 * L
    if len(raw) != expected:
        raise B1ProtocolError(f"block frame must be exactly {expected} bytes")
    block_idx, n_coords = _BLOCK_PREFIX.unpack_from(raw)
    if not 1 <= n_coords <= L:
        raise B1ProtocolError("n_coords must be in [1, L]")
    payload = np.frombuffer(raw, dtype=">f2", count=L, offset=_BLOCK_PREFIX.size).astype(np.float16)
    if not np.isfinite(payload).all():
        raise B1ProtocolError("payload values must be finite")
    if np.any(payload[n_coords:] != 0):
        raise B1ProtocolError("coordinates beyond n_coords must be zero padded")
    return block_idx, n_coords, payload


@dataclass(frozen=True, slots=True)
class B1ClientRound:
    round_id: int
    deadline_ns: int
    encoded: EncodedUpdate
    u: np.ndarray


class B1ClientCore:
    """Prepare B1 updates and update persistent error feedback from gateway results."""

    def __init__(self, client_id: int, *, d: int, L: int = L_DEFAULT) -> None:
        _require_uint(client_id, "client_id", 0xFFFF)
        _require_uint(d, "d", 0xFFFFFFFF, minimum=1)
        _require_uint(L, "L", 0xFFFF, minimum=1)
        self.client_id = client_id
        self.error_feedback = ErrorFeedback(d, L)
        self.round: B1ClientRound | None = None
        self.last_completed_round_id = 0

    def start_round(self, round_id: int, deadline_ns: int, delta: np.ndarray) -> EncodedUpdate:
        if self.round is not None:
            raise B1TransferError("cannot start B1 round while another round is active")
        _require_uint(round_id, "round_id", 0xFFFFFFFF, minimum=1)
        _require_uint(deadline_ns, "deadline_ns", 0xFFFFFFFFFFFFFFFF, minimum=1)
        if round_id <= self.last_completed_round_id:
            raise B1TransferError("round_id must increase monotonically")
        u = self.error_feedback.prepare(delta)
        encoded = encode(u, self.error_feedback.L)
        self.round = B1ClientRound(round_id, deadline_ns, encoded, u.copy())
        return encoded

    def finish_round(self, round_id: int, delivered_indices: Iterable[int]) -> np.ndarray:
        """Apply the exact gateway delivery set once; no delivery ACK is invented."""
        current = self._require_round(round_id)
        indices = tuple(delivered_indices)
        if len(set(indices)) != len(indices):
            raise B1TransferError("delivered block indices must be unique")
        bitmap = bitmap_from_indices(indices, current.encoded.K)
        reconstructed = self.error_feedback.finish_round(current.u, bitmap)
        self.last_completed_round_id = round_id
        self.round = None
        return reconstructed

    def _require_round(self, round_id: int) -> B1ClientRound:
        if self.round is None or self.round.round_id != round_id:
            raise B1TransferError("round result does not match the active B1 round")
        return self.round


@dataclass(frozen=True, slots=True)
class B1ClientTransfer:
    """Client-side transfer counters count application bytes queued to asyncio's writer."""

    client_id: int
    round_id: int
    blocks_written: int
    application_bytes_written: int
    connected: bool
    connection_error: str | None
    close_reason: str
    completed_ns: int
    residual_finalized: bool


class B1TcpClient:
    """Stream all energy-ordered B1 blocks once over one TCP connection."""

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        port: int = B1_PORT,
    ) -> None:
        self.clock_ns = clock_ns
        _require_uint(port, "port", 0xFFFF, minimum=1)
        self.port = port

    async def send_round(
        self,
        core: B1ClientCore,
        *,
        gateway_host: str,
        round_id: int,
        deadline_ns: int,
        delta: np.ndarray,
    ) -> B1ClientTransfer:
        encoded = core.start_round(round_id, deadline_ns, delta)
        blocks_written = 0
        bytes_written = 0
        connected = False
        connection_error: str | None = None
        close_reason = "deadline"
        writer: asyncio.StreamWriter | None = None
        try:
            remaining = self._remaining_s(deadline_ns)
            if remaining <= 0:
                raise asyncio.TimeoutError("deadline passed before connection")
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(gateway_host, self.port), timeout=remaining
            )
            connected = True
            hello = encode_hello(round_id, core.client_id)
            writer.write(hello)
            bytes_written += len(hello)
            await asyncio.wait_for(writer.drain(), timeout=self._remaining_s(deadline_ns))

            for block_idx in encoded.order:
                if self.clock_ns() >= deadline_ns:
                    close_reason = "deadline_before_next_block"
                    break
                payload, n_coords = block_payload_f16(encoded, int(block_idx))
                frame = encode_block_frame(int(block_idx), n_coords, payload)
                writer.write(frame)
                bytes_written += len(frame)
                blocks_written += 1
                await asyncio.wait_for(writer.drain(), timeout=self._remaining_s(deadline_ns))
            else:
                close_reason = "all_blocks_written"

            # Keep the connection open: the gateway, not the client, closes it at D.
            remaining = self._remaining_s(deadline_ns)
            if remaining > 0:
                try:
                    await asyncio.wait_for(reader.read(1), timeout=remaining)
                except asyncio.TimeoutError:
                    close_reason = "local_deadline"
                else:
                    close_reason = "gateway_closed"
            else:
                close_reason = "local_deadline"
        except (OSError, asyncio.TimeoutError) as exc:
            connection_error = str(exc)
            if not connected:
                close_reason = "connection_failed"
            elif self.clock_ns() >= deadline_ns or isinstance(exc, asyncio.TimeoutError):
                close_reason = "deadline_during_write"
            else:
                close_reason = "connection_error"
        finally:
            residual_finalized = False
            if not connected and core.round is not None and core.round.round_id == round_id:
                core.finish_round(round_id, ())
                residual_finalized = True
            if writer is not None:
                await _close_stream_writer(writer)
        return B1ClientTransfer(
            core.client_id,
            round_id,
            blocks_written,
            bytes_written,
            connected,
            connection_error,
            close_reason,
            self.clock_ns(),
            residual_finalized,
        )

    def _remaining_s(self, deadline_ns: int) -> float:
        return max(0.0, (deadline_ns - self.clock_ns()) / 1e9)


@dataclass(slots=True)
class _ClientReceived:
    connected: bool = False
    delivered: dict[int, tuple[int, np.ndarray]] = field(default_factory=dict)
    receive_order: list[int] = field(default_factory=list)
    incomplete_frames: int = 0
    rejected_frames: int = 0
    bytes_received: int = 0
    last_block_ns: int | None = None
    close_reason: str = "no_connection_by_deadline"


class B1RoundAccumulator:
    """Validate and decode completed block frames for one deadline-bounded round."""

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
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        _require_uint(round_id, "round_id", 0xFFFFFFFF, minimum=1)
        _require_uint(start_ns, "start_ns", 0xFFFFFFFFFFFFFFFF)
        _require_uint(deadline_ns, "deadline_ns", 0xFFFFFFFFFFFFFFFF, minimum=1)
        if deadline_ns <= start_ns:
            raise ValueError("deadline_ns must be after start_ns")
        _require_uint(d, "d", 0xFFFFFFFF, minimum=1)
        _require_uint(L, "L", 0xFFFF, minimum=1)
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
            raise ValueError("B1 block count exceeds the u16 block index space")
        self.client_weights = {cid: float(client_weights[cid]) for cid in ids}
        self.clock_ns = clock_ns
        self.clients = {cid: _ClientReceived() for cid in ids}
        self.rejected_connections = 0
        self._active = True
        self.close_reason: str | None = None
        self.stopped_ns: int | None = None

    def mark_connected(self, client_id: int, round_id: int) -> None:
        if not self._active or self.clock_ns() > self.deadline_ns:
            raise B1ProtocolError("B1 round is no longer accepting connections")
        if round_id != self.round_id:
            raise B1ProtocolError("stale or mismatched B1 round")
        if client_id not in self.clients:
            raise B1ProtocolError("unknown B1 client ID")
        state = self.clients[client_id]
        if state.connected:
            raise B1ProtocolError("duplicate B1 client connection")
        state.connected = True
        state.close_reason = "open"

    def accept_frame(self, client_id: int, frame: bytes, received_ns: int) -> None:
        if client_id not in self.clients:
            self.rejected_connections += 1
            raise B1ProtocolError("unknown B1 client ID")
        state = self.clients[client_id]
        state.bytes_received += len(frame)
        if not self._active or received_ns > self.deadline_ns:
            state.rejected_frames += 1
            raise B1ProtocolError("B1 block arrived after the round deadline")
        if not state.connected:
            state.rejected_frames += 1
            raise B1ProtocolError("B1 client has not completed its handshake")
        try:
            block_idx, n_coords, payload = decode_block_frame(frame, self.L)
            if block_idx >= self.K:
                raise B1ProtocolError("block_idx is outside the configured update")
            expected_n = min(self.L, self.d - block_idx * self.L)
            if n_coords != expected_n:
                raise B1ProtocolError("n_coords does not match the original block")
            if block_idx in state.delivered:
                raise B1ProtocolError("duplicate B1 block")
        except B1ProtocolError:
            state.rejected_frames += 1
            raise
        state.delivered[block_idx] = (n_coords, payload.copy())
        state.receive_order.append(block_idx)
        state.last_block_ns = received_ns

    def mark_incomplete(self, client_id: int, partial_bytes: int, *, at_deadline: bool) -> None:
        if client_id in self.clients and partial_bytes:
            state = self.clients[client_id]
            state.incomplete_frames += 1
            state.bytes_received += partial_bytes
            state.close_reason = (
                "incomplete_frame_at_deadline" if at_deadline else "incomplete_frame_peer_closed"
            )

    def close_client(self, client_id: int, reason: str) -> None:
        if client_id in self.clients:
            state = self.clients[client_id]
            if state.close_reason == "open":
                state.close_reason = reason

    def close_all(self, reason: str) -> None:
        for state in self.clients.values():
            if state.connected and state.close_reason == "open":
                state.close_reason = reason

    def stop(self, reason: str = "round_stopped") -> None:
        self._active = False
        self.close_reason = reason
        self.stopped_ns = self.clock_ns()

    def result(self, completed_ns: int | None = None) -> "B1RoundResult":
        if self._active:
            raise B1TransferError("cannot build B1 result before the deadline closes the round")
        if completed_ns is None:
            completed_ns = self.stopped_ns if self.stopped_ns is not None else self.clock_ns()
        client_results: dict[int, B1ClientRoundResult] = {}
        for client_id in self.client_ids:
            state = self.clients[client_id]
            indices = tuple(sorted(state.delivered))
            update = np.zeros(self.d, dtype=np.float32)
            n_coords: dict[int, int] = {}
            for block_idx, (count, payload) in state.delivered.items():
                start = block_idx * self.L
                update[start : start + count] = payload[:count].astype(np.float32)
                n_coords[block_idx] = count
            client_results[client_id] = B1ClientRoundResult(
                client_id=client_id,
                delivered_indices=indices,
                receive_order=tuple(state.receive_order),
                delivered_bitmap=bitmap_from_indices(indices, self.K),
                n_coords=n_coords,
                decoded_update=update,
                blocks_received=len(indices),
                incomplete_frames=state.incomplete_frames,
                rejected_frames=state.rejected_frames,
                bytes_received=state.bytes_received,
                connected=state.connected,
                close_reason=state.close_reason,
                last_block_ns=state.last_block_ns,
            )
        return B1RoundResult(
            round_id=self.round_id,
            start_ns=self.start_ns,
            deadline_ns=self.deadline_ns,
            completed_ns=completed_ns,
            client_results=client_results,
            client_weights=self.client_weights.copy(),
            rejected_connections=self.rejected_connections,
            L=self.L,
            d=self.d,
            close_reason=self.close_reason or "round_stopped",
            completed_at_deadline=self.close_reason == "gateway_deadline",
        )


@dataclass(frozen=True, slots=True)
class B1ClientRoundResult:
    client_id: int
    delivered_indices: tuple[int, ...]
    receive_order: tuple[int, ...]
    delivered_bitmap: bytes
    n_coords: Mapping[int, int]
    decoded_update: np.ndarray
    blocks_received: int
    incomplete_frames: int
    rejected_frames: int
    bytes_received: int
    connected: bool
    close_reason: str
    last_block_ns: int | None


class Aggregator(Protocol):
    def __call__(self, updates: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class B1RoundResult:
    round_id: int
    start_ns: int
    deadline_ns: int
    completed_ns: int
    client_results: Mapping[int, B1ClientRoundResult]
    client_weights: Mapping[int, float]
    rejected_connections: int
    L: int
    d: int
    close_reason: str
    completed_at_deadline: bool

    @property
    def K(self) -> int:
        return n_blocks(self.d, self.L)

    @property
    def delivered_blocks(self) -> int:
        return sum(result.blocks_received for result in self.client_results.values())

    @property
    def incomplete_blocks(self) -> int:
        return sum(result.incomplete_frames for result in self.client_results.values())

    @property
    def rejected_blocks(self) -> int:
        return sum(result.rejected_frames for result in self.client_results.values())

    @property
    def connection_failures(self) -> int:
        return sum(not result.connected for result in self.client_results.values())

    @property
    def application_bytes_received(self) -> int:
        return sum(result.bytes_received for result in self.client_results.values())

    @property
    def goodput_bps(self) -> float:
        return self.goodput_for_duration(self.deadline_ns - self.start_ns)

    def goodput_for_duration(self, duration_ns: int) -> float:
        if duration_ns <= 0:
            raise ValueError("duration_ns must be positive")
        # Section 12 counts transmitted FP16 payload bits, including zero padding.
        payload_bytes = self.delivered_blocks * self.L * 2
        return payload_bytes * 8e9 / duration_ns

    @property
    def client_n_coords(self) -> dict[int, dict[int, int]]:
        return {
            client_id: dict(result.n_coords)
            for client_id, result in self.client_results.items()
        }

    def energy_ratio(self, block_energies: Mapping[int, np.ndarray]) -> float:
        """Compute Section 3.1 eta using original-index energies retained by clients."""
        from fedgrant.analysis.metrics import compute_eta

        if set(block_energies) != set(self.client_results):
            raise ValueError("block_energies must contain exactly the configured clients")
        energy_rows = []
        delivered_rows = []
        for client_id in sorted(self.client_results):
            row = np.asarray(block_energies[client_id], dtype=np.float64)
            if row.shape != (self.K,) or not np.isfinite(row).all() or np.any(row < 0):
                raise ValueError(f"client {client_id} block energies must be finite and have shape ({self.K},)")
            energy_rows.append(row)
            delivered_rows.append(self.client_results[client_id].delivered_indices)
        weights = [self.client_weights[cid] for cid in sorted(self.client_results)]
        return compute_eta(weights, np.stack(energy_rows), delivered_rows, K=self.K)

    def aggregate(self, aggregator: Aggregator) -> np.ndarray:
        """Delegate decoded zero-filled client updates to the existing FL aggregator."""
        if not callable(aggregator):
            raise TypeError("aggregator must be callable")
        client_ids = tuple(sorted(self.client_results))
        updates = [self.client_results[cid].decoded_update.copy() for cid in client_ids]
        weights = np.asarray([self.client_weights[cid] for cid in client_ids], dtype=np.float32)
        aggregate = aggregator(updates, weights)
        if not isinstance(aggregate, np.ndarray) or aggregate.dtype != np.float32 or aggregate.shape != (self.d,):
            raise B1TransferError("aggregator must return a float32 vector with the model dimension")
        if not np.isfinite(aggregate).all():
            raise B1TransferError("aggregator returned non-finite values")
        return aggregate

    def client_records(self) -> list[dict[str, object]]:
        """Return per-client fields needed by B1 logging without altering runs.csv."""
        return [
            {
                "round_id": self.round_id,
                "start_ns": self.start_ns,
                "deadline_ns": self.deadline_ns,
                "completed_ns": self.completed_ns,
                "client_id": cid,
                "blocks_received": result.blocks_received,
                "receive_order": result.receive_order,
                "incomplete_blocks": result.incomplete_frames,
                "rejected_blocks": result.rejected_frames,
                "delivered_indices": result.delivered_indices,
                "delivered_bitmap_hex": result.delivered_bitmap.hex(),
                "n_coords": dict(result.n_coords),
                "application_bytes_received": result.bytes_received,
                "connected": result.connected,
                "close_reason": result.close_reason,
                "last_block_ns": result.last_block_ns,
                "round_close_reason": self.close_reason,
                "completed_at_deadline": self.completed_at_deadline,
                "goodput_bps": self.goodput_bps,
                "round_delivered_blocks": self.delivered_blocks,
                "round_incomplete_frames": self.incomplete_blocks,
                "round_rejected_frames": self.rejected_blocks,
                "round_rejected_connections": self.rejected_connections,
                "round_connection_failures": self.connection_failures,
                "round_application_bytes_received": self.application_bytes_received,
            }
            for cid, result in sorted(self.client_results.items())
        ]


def apply_delivery_result(core: B1ClientCore, result: B1RoundResult) -> np.ndarray:
    """Apply this client's gateway delivery set through the orchestration boundary."""
    if not isinstance(core, B1ClientCore) or not isinstance(result, B1RoundResult):
        raise TypeError("core and result must be B1ClientCore and B1RoundResult instances")
    if core.round is None or core.round.round_id != result.round_id:
        raise B1TransferError("gateway result does not match the client's active round")
    client_result = result.client_results.get(core.client_id)
    if client_result is None:
        raise B1TransferError("gateway result does not include this client")
    return core.finish_round(result.round_id, client_result.delivered_indices)


class B1TcpGateway:
    """Receive B1 streams until D, then close every active client connection."""

    def __init__(self, accumulator: B1RoundAccumulator, *, host: str = "0.0.0.0", port: int = B1_PORT) -> None:
        self.accumulator = accumulator
        self.host = host
        _require_uint(port, "port", 0xFFFF)
        self.port = port
        self.server: asyncio.AbstractServer | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        self._handlers: set[asyncio.Task[object]] = set()
        self._deadline_task: asyncio.Task[None] | None = None
        self._deadline_event = asyncio.Event()
        self._closing = False

    async def start(self) -> "B1TcpGateway":
        if self.server is not None:
            raise B1TransferError("B1 gateway is already started")
        if self.accumulator.deadline_ns <= self.accumulator.clock_ns():
            raise B1TransferError("B1 deadline must be in the future")
        self.server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
            reuse_address=True,
        )
        self._deadline_task = asyncio.create_task(self._close_at_deadline())
        return self

    @property
    def address(self) -> tuple[str, int]:
        if self.server is None or not self.server.sockets:
            raise B1TransferError("B1 gateway is not started")
        host, port = self.server.sockets[0].getsockname()[:2]
        return str(host), int(port)

    async def wait(self) -> B1RoundResult:
        # The server is released at D, so "started" is tracked by the deadline
        # task: waiting after the deadline must return the result, not fail.
        if self._deadline_task is None:
            raise B1TransferError("B1 gateway is not started")
        await self._deadline_event.wait()
        await asyncio.gather(*tuple(self._handlers), return_exceptions=True)
        return self.accumulator.result()

    async def close(self) -> None:
        await self._close_now("gateway_shutdown")
        if self._deadline_task is not None and not self._deadline_task.done():
            self._deadline_task.cancel()
            await asyncio.gather(self._deadline_task, return_exceptions=True)
        await asyncio.gather(*tuple(self._handlers), return_exceptions=True)

    async def _close_at_deadline(self) -> None:
        remaining_ns = self.accumulator.deadline_ns - self.accumulator.clock_ns()
        if remaining_ns > 0:
            await asyncio.sleep(remaining_ns / 1e9)
        await self._close_now("gateway_deadline")

    async def _close_now(self, reason: str) -> None:
        if self._closing:
            return
        self._closing = True
        self.accumulator.stop(reason)
        self.accumulator.close_all(reason)
        server = self.server
        if self.server is not None:
            self.server.close()
            self.server = None
        for writer in tuple(self._writers):
            writer.close()
        self._writers.clear()
        if server is not None:
            await server.wait_closed()
        self._deadline_event.set()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        self._writers.add(writer)
        client_id: int | None = None
        try:
            if not self.accumulator._active or self.accumulator.clock_ns() > self.accumulator.deadline_ns:
                return
            try:
                hello = await reader.readexactly(_HELLO.size)
            except asyncio.IncompleteReadError:
                self.accumulator.rejected_connections += 1
                return
            round_id, client_id = decode_hello(hello)
            self.accumulator.mark_connected(client_id, round_id)
            state = self.accumulator.clients[client_id]
            state.bytes_received += len(hello)
            frame_size = _BLOCK_PREFIX.size + 2 * self.accumulator.L
            while self.accumulator._active:
                try:
                    frame = await reader.readexactly(frame_size)
                except asyncio.IncompleteReadError as exc:
                    if exc.partial:
                        self.accumulator.mark_incomplete(
                            client_id,
                            len(exc.partial),
                            at_deadline=(
                                not self.accumulator._active
                                or self.accumulator.clock_ns() >= self.accumulator.deadline_ns
                            ),
                        )
                    else:
                        reason = "gateway_deadline" if not self.accumulator._active else "peer_closed"
                        self.accumulator.close_client(client_id, reason)
                    return
                received_ns = self.accumulator.clock_ns()
                try:
                    self.accumulator.accept_frame(client_id, frame, received_ns)
                except B1ProtocolError:
                    return
        except B1ProtocolError:
            self.accumulator.rejected_connections += 1
            if client_id is not None:
                self.accumulator.close_client(client_id, "invalid_connection")
        finally:
            self._writers.discard(writer)
            await _close_stream_writer(writer)
            if task is not None:
                self._handlers.discard(task)


def _require_uint(value: object, name: str, maximum: int, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise B1ProtocolError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise B1ProtocolError(f"{name} must be in [{minimum}, {maximum}]")
    return result


async def _close_stream_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=0.1)
    except asyncio.TimeoutError:
        writer.transport.abort()
    except OSError:
        pass


def write_metrics_csv(
    path: str | Path,
    result: B1RoundResult,
    client_transfers: Mapping[int, B1ClientTransfer] | None = None,
) -> None:
    """Write per-client B1 delivery metrics as JSON-valued CSV fields."""
    import csv
    import json

    records = result.client_records()
    if not records:
        raise ValueError("B1 result contains no configured clients")
    transfers = client_transfers or {}
    if set(transfers) - set(result.client_results):
        raise ValueError("client transfer metrics include an unknown client")
    for record in records:
        transfer = transfers.get(int(record["client_id"]))
        record.update(
            {
                "blocks_written": transfer.blocks_written if transfer else "",
                "application_bytes_written": transfer.application_bytes_written if transfer else "",
                "client_connected": transfer.connected if transfer else "",
                "client_connection_error": transfer.connection_error if transfer else "",
                "client_close_reason": transfer.close_reason if transfer else "",
                "client_completed_ns": transfer.completed_ns if transfer else "",
                "residual_finalized": transfer.residual_finalized if transfer else "",
            }
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(records[0])
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, tuple, list))
                    else value
                    for key, value in record.items()
                }
            )


__all__ = [
    "B1_PORT",
    "B1ClientCore",
    "B1ClientRoundResult",
    "B1ClientTransfer",
    "B1ProtocolError",
    "B1RoundAccumulator",
    "B1RoundResult",
    "B1TcpClient",
    "B1TcpGateway",
    "B1TransferError",
    "apply_delivery_result",
    "decode_block_frame",
    "decode_hello",
    "encode_block_frame",
    "encode_hello",
    "write_metrics_csv",
]
