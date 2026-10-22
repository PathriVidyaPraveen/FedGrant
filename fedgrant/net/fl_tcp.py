"""Bounded TCP transport for model broadcast and completed aggregates.

This transport is separate from FedGrant's six-message UDP protocol. Each
connection carries one length-prefixed JSON envelope and one NPY array.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import struct
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fedgrant.common.model import ArrayPayloadError, deserialize_array, serialize_array

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
MAX_TRANSFER_BYTES = 64 * 1024 * 1024
_LENGTH = struct.Struct("!I")
_HEADER_LENGTH = struct.Struct("!I")
_MAX_HEADER_BYTES = 64 * 1024


class TransferError(RuntimeError):
    """Raised when an FL TCP transfer is incomplete, invalid, or rejected."""


class ModelSource(Protocol):
    def current_model(self) -> tuple[int, np.ndarray]:
        """Return a stable snapshot of the current model version and numeric array."""


class ModelMailbox:
    """Hold the latest model delivered for an upcoming client round."""

    def __init__(self, expected_shape: tuple[int, ...] | None = None) -> None:
        self.expected_shape = expected_shape
        self._current: tuple[int, int, np.ndarray] | None = None

    def install(self, round_id: int, model_version: int, model: np.ndarray) -> None:
        _require_uint(round_id, "round_id", 0xFFFFFFFF, minimum=1)
        _require_uint(model_version, "model_version", 0xFFFFFFFFFFFFFFFF)
        if self.expected_shape is not None and model.shape != self.expected_shape:
            raise TransferError(
                f"model shape {model.shape} does not match expected {self.expected_shape}"
            )
        current = self._current
        if current is not None:
            current_round, current_version, current_model = current
            if round_id == current_round:
                if model_version == current_version and np.array_equal(model, current_model):
                    return
                raise TransferError("conflicting model payload for the same round")
            if round_id < current_round:
                raise TransferError("stale model round")
        self._current = (round_id, model_version, model.copy())

    def model_for_round(self, round_id: int) -> np.ndarray:
        _require_uint(round_id, "round_id", 0xFFFFFFFF, minimum=1)
        if self._current is None or self._current[0] != round_id:
            raise TransferError(f"model for round {round_id} has not been received")
        return self._current[2].copy()


class ModelBackedUpdateProvider:
    """Adapt the received global model to the existing UdpClient update callback."""

    def __init__(
        self,
        mailbox: ModelMailbox,
        update: Callable[[int, np.ndarray], np.ndarray],
    ) -> None:
        if not callable(update):
            raise TypeError("update must be callable")
        self.mailbox = mailbox
        self.update = update

    def __call__(self, round_id: int) -> np.ndarray:
        return self.update(round_id, self.mailbox.model_for_round(round_id))


@dataclass(frozen=True, slots=True)
class AggregateEnvelope:
    """Server-facing aggregate plus the gateway's round and delivery metadata."""

    round_id: int
    aggregate: np.ndarray
    client_n_coords: Mapping[int, Mapping[int, int]]

    @property
    def client_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.client_n_coords))


class AggregateReceiver(Protocol):
    async def receive_aggregate(self, envelope: AggregateEnvelope) -> None:
        """Apply or durably accept one completed round aggregate."""


class TcpModelReceiver:
    """TCP endpoint that validates and installs models into a client mailbox."""

    def __init__(
        self,
        mailbox: ModelMailbox,
        *,
        max_transfer_bytes: int = MAX_TRANSFER_BYTES,
        timeout: float = 30.0,
    ) -> None:
        self.mailbox = mailbox
        self.max_transfer_bytes = _validate_limit(max_transfer_bytes)
        _validate_timeout(timeout)
        self.timeout = timeout
        self.server: asyncio.AbstractServer | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> "TcpModelReceiver":
        if self.server is not None:
            raise TransferError("model receiver is already started")
        self.server = await asyncio.start_server(self._handle, host, port)
        return self

    @property
    def address(self) -> tuple[str, int]:
        if self.server is None or not self.server.sockets:
            raise TransferError("model receiver is not started")
        host, port = self.server.sockets[0].getsockname()[:2]
        return str(host), int(port)

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        round_id: int | None = None
        try:
            header, payload = await asyncio.wait_for(
                _read_frame(reader, self.max_transfer_bytes), self.timeout
            )
            if set(header) != {
                "version",
                "kind",
                "round_id",
                "model_version",
                "payload_bytes",
                "payload_sha256",
            }:
                raise TransferError("model envelope has missing or unexpected fields")
            _validate_header(header, "MODEL")
            round_id = _require_uint(header["round_id"], "round_id", 0xFFFFFFFF, minimum=1)
            model_version = _require_uint(
                header["model_version"], "model_version", 0xFFFFFFFFFFFFFFFF
            )
            digest = _validate_payload_digest(header, payload)
            model = await asyncio.to_thread(deserialize_array, payload)
            self.mailbox.install(round_id, model_version, model)
            await _write_frame(
                writer,
                {
                    "version": PROTOCOL_VERSION,
                    "kind": "ACK",
                    "round_id": round_id,
                    "model_version": model_version,
                    "payload_sha256": digest,
                    "status": "ok",
                    "payload_bytes": 0,
                },
                b"",
                self.max_transfer_bytes,
            )
        except (TransferError, ArrayPayloadError, TypeError, ValueError, OSError, asyncio.TimeoutError) as exc:
            log.warning("rejecting model transfer: %s", exc)
            await _write_error_ack(writer, round_id, str(exc), self.max_transfer_bytes)
        finally:
            await _close_writer(writer)


class TcpAggregateReceiver:
    """Receive aggregate transfers and suppress duplicate application per round."""

    def __init__(
        self,
        receiver: AggregateReceiver | Callable[[AggregateEnvelope], Awaitable[None] | None],
        *,
        max_transfer_bytes: int = MAX_TRANSFER_BYTES,
        timeout: float = 30.0,
    ) -> None:
        if not callable(receiver) and not callable(getattr(receiver, "receive_aggregate", None)):
            raise TypeError("receiver must implement receive_aggregate(envelope)")
        self.receiver = receiver
        self.max_transfer_bytes = _validate_limit(max_transfer_bytes)
        _validate_timeout(timeout)
        self.timeout = timeout
        self.server: asyncio.AbstractServer | None = None
        self._lock = asyncio.Lock()
        self._last_round_id = 0
        self._last_digest: bytes | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> "TcpAggregateReceiver":
        if self.server is not None:
            raise TransferError("aggregate receiver is already started")
        self.server = await asyncio.start_server(self._handle, host, port)
        return self

    @property
    def address(self) -> tuple[str, int]:
        if self.server is None or not self.server.sockets:
            raise TransferError("aggregate receiver is not started")
        host, port = self.server.sockets[0].getsockname()[:2]
        return str(host), int(port)

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        round_id: int | None = None
        try:
            header, payload = await asyncio.wait_for(
                _read_frame(reader, self.max_transfer_bytes), self.timeout
            )
            required = {
                "version",
                "kind",
                "round_id",
                "client_n_coords",
                "payload_bytes",
                "payload_sha256",
            }
            if set(header) != required:
                raise TransferError("aggregate envelope has missing or unexpected fields")
            _validate_header(header, "AGGREGATE")
            round_id = _require_uint(header["round_id"], "round_id", 0xFFFFFFFF, minimum=1)
            metadata = _decode_client_metadata(header["client_n_coords"])
            digest = _validate_payload_digest(header, payload)
            aggregate = await asyncio.to_thread(deserialize_array, payload)
            if aggregate.dtype != np.dtype(np.float32) or aggregate.ndim != 1:
                raise TransferError("aggregate must be a 1-D float32 vector")
            transfer_digest = hashlib.sha256(_canonical_json(header) + payload).digest()
            async with self._lock:
                if round_id < self._last_round_id:
                    raise TransferError("stale aggregate round")
                if round_id == self._last_round_id:
                    if transfer_digest != self._last_digest:
                        raise TransferError("conflicting duplicate aggregate for the same round")
                else:
                    envelope = AggregateEnvelope(round_id, aggregate, metadata)
                    callback = getattr(self.receiver, "receive_aggregate", self.receiver)
                    result = callback(envelope)
                    if inspect.isawaitable(result):
                        await asyncio.wait_for(result, self.timeout)
                    self._last_round_id = round_id
                    self._last_digest = transfer_digest
            await _write_frame(
                writer,
                {
                    "version": PROTOCOL_VERSION,
                    "kind": "ACK",
                    "round_id": round_id,
                    "payload_sha256": digest,
                    "status": "ok",
                    "payload_bytes": 0,
                },
                b"",
                self.max_transfer_bytes,
            )
        except (TransferError, ArrayPayloadError, TypeError, ValueError, OSError, asyncio.TimeoutError) as exc:
            log.warning("rejecting aggregate transfer: %s", exc)
            await _write_error_ack(writer, round_id, str(exc), self.max_transfer_bytes)
        except Exception:
            log.exception("aggregate receiver failed")
            await _write_error_ack(writer, round_id, "aggregate receiver failed", self.max_transfer_bytes)
        finally:
            await _close_writer(writer)


class TcpAggregateSender:
    """Async gateway callback that sends one completed aggregate and its metadata."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: float = 10.0,
        max_transfer_bytes: int = MAX_TRANSFER_BYTES,
    ) -> None:
        self.host, self.port = _validate_endpoint(host, port)
        _validate_timeout(timeout)
        self.timeout = timeout
        self.max_transfer_bytes = _validate_limit(max_transfer_bytes)

    async def send_aggregate(
        self,
        round_id: int,
        aggregate: np.ndarray,
        client_n_coords: Mapping[int, Mapping[int, int]],
    ) -> None:
        _require_uint(round_id, "round_id", 0xFFFFFFFF, minimum=1)
        if not isinstance(aggregate, np.ndarray) or aggregate.dtype != np.float32 or aggregate.ndim != 1:
            raise TransferError("aggregate must be a 1-D float32 vector")
        _validate_timeout(self.timeout)
        _validate_array_size(aggregate, self.max_transfer_bytes)
        metadata = _encode_client_metadata(client_n_coords)
        payload = await asyncio.to_thread(_serialize_snapshot, aggregate, self.max_transfer_bytes)
        header = {
            "version": PROTOCOL_VERSION,
            "kind": "AGGREGATE",
            "round_id": round_id,
            "client_n_coords": metadata,
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
        for attempt in range(2):
            try:
                await _send_and_check_ack(
                    self.host,
                    self.port,
                    header,
                    payload,
                    self.timeout,
                    self.max_transfer_bytes,
                )
                return
            except TransferError:
                if attempt == 1:
                    raise


async def send_model(
    host: str,
    port: int,
    *,
    round_id: int,
    model_version: int,
    model: np.ndarray,
    timeout: float = 10.0,
    max_transfer_bytes: int = MAX_TRANSFER_BYTES,
) -> None:
    """Send one model and return only after the client mailbox acknowledges it."""
    host, port = _validate_endpoint(host, port)
    _require_uint(round_id, "round_id", 0xFFFFFFFF, minimum=1)
    _require_uint(model_version, "model_version", 0xFFFFFFFFFFFFFFFF)
    limit = _validate_limit(max_transfer_bytes)
    _validate_timeout(timeout)
    payload = await asyncio.to_thread(_serialize_snapshot, model, limit)
    await _send_model_payload(host, port, round_id, model_version, payload, timeout, limit)


async def _send_model_payload(
    host: str,
    port: int,
    round_id: int,
    model_version: int,
    payload: bytes,
    timeout: float,
    max_transfer_bytes: int,
) -> None:
    await _send_and_check_ack(
        host,
        port,
        {
            "version": PROTOCOL_VERSION,
            "kind": "MODEL",
            "round_id": round_id,
            "model_version": model_version,
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        },
        payload,
        timeout,
        max_transfer_bytes,
    )


async def broadcast_model(
    source: ModelSource,
    endpoints: Sequence[tuple[str, int]],
    *,
    round_id: int,
    timeout: float = 10.0,
    max_transfer_bytes: int = MAX_TRANSFER_BYTES,
) -> int:
    """Send the model to every client; await this before starting the gateway round."""
    if not callable(getattr(source, "current_model", None)):
        raise TypeError("source must implement current_model()")
    _require_uint(round_id, "round_id", 0xFFFFFFFF, minimum=1)
    model_version, model = source.current_model()
    _require_uint(model_version, "model_version", 0xFFFFFFFFFFFFFFFF)
    if not endpoints:
        raise TransferError("model broadcast requires at least one client endpoint")
    _validate_timeout(timeout)
    limit = _validate_limit(max_transfer_bytes)
    payload = await asyncio.to_thread(_serialize_snapshot, model, limit)
    transfers = [
        asyncio.create_task(
            _send_model_payload(
                host,
                port,
                round_id,
                model_version,
                payload,
                timeout,
                limit,
            )
        )
        for host, port in endpoints
    ]
    results = await asyncio.gather(*transfers, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return model_version


async def _send_and_check_ack(
    host: str,
    port: int,
    header: Mapping[str, object],
    payload: bytes,
    timeout: float,
    max_transfer_bytes: int,
) -> None:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except (OSError, asyncio.TimeoutError) as exc:
        raise TransferError(f"could not connect to {host}:{port}: {exc}") from exc
    try:
        await asyncio.wait_for(_write_frame(writer, header, payload, max_transfer_bytes), timeout)
        ack, ack_payload = await asyncio.wait_for(_read_frame(reader, max_transfer_bytes), timeout)
        status = ack.get("status")
        _require_uint(ack.get("version"), "ack version", 0xFF)
        if ack.get("version") != PROTOCOL_VERSION:
            raise TransferError("unsupported transfer acknowledgement version")
        ack_round_id = _require_uint(ack.get("round_id"), "ack round_id", 0xFFFFFFFF)
        if ack_round_id != header["round_id"]:
            raise TransferError("acknowledgement refers to a different round")
        ack_payload_bytes = _require_uint(ack.get("payload_bytes"), "ack payload_bytes", 0xFFFFFFFF)
        if status == "ok":
            expected_fields = {"version", "kind", "round_id", "status", "payload_bytes", "payload_sha256"}
            if header.get("kind") == "MODEL":
                expected_fields.add("model_version")
        else:
            expected_fields = {"version", "kind", "round_id", "status", "error", "payload_bytes"}
        if (
            set(ack) != expected_fields
            or ack_payload
            or ack.get("kind") != "ACK"
            or ack_payload_bytes != 0
        ):
            raise TransferError("invalid transfer acknowledgement")
        if status != "ok":
            error = ack.get("error")
            if not isinstance(error, str):
                raise TransferError("invalid transfer error acknowledgement")
            raise TransferError(error or "transfer rejected")
        if not isinstance(ack.get("payload_sha256"), str) or ack["payload_sha256"] != header.get("payload_sha256"):
            raise TransferError("acknowledgement refers to a different payload")
        if header.get("kind") == "MODEL":
            ack_model_version = _require_uint(
                ack.get("model_version"), "ack model_version", 0xFFFFFFFFFFFFFFFF
            )
            if ack_model_version != header.get("model_version"):
                raise TransferError("acknowledgement refers to a different model version")
    except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError) as exc:
        raise TransferError(f"transfer to {host}:{port} failed: {exc}") from exc
    finally:
        await _close_writer(writer)


async def _read_frame(
    reader: asyncio.StreamReader, max_transfer_bytes: int
) -> tuple[dict[str, object], bytes]:
    try:
        (frame_length,) = _LENGTH.unpack(await reader.readexactly(_LENGTH.size))
        if frame_length < _HEADER_LENGTH.size or frame_length > max_transfer_bytes + _MAX_HEADER_BYTES + 4:
            raise TransferError("frame length is outside the permitted bound")
        frame = await reader.readexactly(frame_length)
    except asyncio.IncompleteReadError as exc:
        raise TransferError("truncated TCP frame") from exc
    (header_length,) = _HEADER_LENGTH.unpack_from(frame)
    if header_length < 2 or header_length > _MAX_HEADER_BYTES or 4 + header_length > len(frame):
        raise TransferError("invalid TCP frame header length")
    try:
        header = json.loads(
            frame[4 : 4 + header_length],
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TransferError) as exc:
        raise TransferError("invalid JSON transfer header") from exc
    if not isinstance(header, dict):
        raise TransferError("transfer header must be a JSON object")
    payload = frame[4 + header_length :]
    if header.get("payload_bytes") != len(payload) or len(payload) > max_transfer_bytes:
        raise TransferError("payload length does not match its envelope")
    return header, payload


async def _write_frame(
    writer: asyncio.StreamWriter,
    header: Mapping[str, object],
    payload: bytes,
    max_transfer_bytes: int,
) -> None:
    if len(payload) > max_transfer_bytes:
        raise TransferError("payload exceeds the configured transfer limit")
    header_bytes = _canonical_json(header)
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise TransferError("transfer header exceeds the configured limit")
    body_length = _HEADER_LENGTH.size + len(header_bytes) + len(payload)
    if body_length > 0xFFFFFFFF:
        raise TransferError("transfer frame exceeds u32 length")
    writer.write(
        _LENGTH.pack(body_length)
        + _HEADER_LENGTH.pack(len(header_bytes))
        + header_bytes
    )
    await writer.drain()
    for offset in range(0, len(payload), 64 * 1024):
        writer.write(payload[offset : offset + 64 * 1024])
        await writer.drain()


async def _write_error_ack(
    writer: asyncio.StreamWriter,
    round_id: int | None,
    message: str,
    max_transfer_bytes: int,
) -> None:
    try:
        await _write_frame(
            writer,
            {
                "version": PROTOCOL_VERSION,
                "kind": "ACK",
                "round_id": round_id or 0,
                "status": "error",
                "error": message[:512],
                "payload_bytes": 0,
            },
            b"",
            max_transfer_bytes,
        )
    except (OSError, TransferError):
        log.debug("could not return transfer error acknowledgement", exc_info=True)


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        log.debug("TCP peer closed during connection teardown", exc_info=True)


def _validate_header(header: Mapping[str, object], kind: str) -> None:
    version = _require_uint(header.get("version"), "version", 0xFF)
    if version != PROTOCOL_VERSION or header.get("kind") != kind:
        raise TransferError("unsupported transfer version or message kind")
    _require_uint(header.get("payload_bytes"), "payload_bytes", 0xFFFFFFFF)


def _validate_payload_digest(header: Mapping[str, object], payload: bytes) -> str:
    digest = header.get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise TransferError("payload_sha256 must be a 64-character SHA-256 hex digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise TransferError("payload_sha256 is not hexadecimal") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if digest != actual:
        raise TransferError("payload checksum mismatch")
    return actual


def _encode_client_metadata(
    client_n_coords: Mapping[int, Mapping[int, int]],
) -> dict[str, dict[str, int]]:
    if not isinstance(client_n_coords, Mapping):
        raise TransferError("client coordinate metadata must be a mapping")
    encoded: dict[str, dict[str, int]] = {}
    for client_id, counts in client_n_coords.items():
        _require_uint(client_id, "client_id", 0xFFFF)
        if not isinstance(counts, Mapping):
            raise TransferError("per-client coordinate metadata must be a mapping")
        encoded[str(client_id)] = {}
        for block_idx, count in counts.items():
            _require_uint(block_idx, "block_idx", 0xFFFF)
            _require_uint(count, "n_coords", 0xFFFF, minimum=1)
            encoded[str(client_id)][str(block_idx)] = count
    return encoded


def _decode_client_metadata(value: object) -> dict[int, dict[int, int]]:
    if not isinstance(value, dict):
        raise TransferError("client coordinate metadata must be a JSON object")
    decoded: dict[int, dict[int, int]] = {}
    for client, counts in value.items():
        if not isinstance(client, str) or not client.isdecimal():
            raise TransferError("client metadata keys must be decimal client IDs")
        client_id = int(client)
        _require_uint(client_id, "client_id", 0xFFFF)
        if client_id in decoded or not isinstance(counts, dict):
            raise TransferError("duplicate client ID or invalid coordinate metadata")
        decoded[client_id] = {}
        for block, count in counts.items():
            if not isinstance(block, str) or not block.isdecimal():
                raise TransferError("block metadata keys must be decimal block indexes")
            block_idx = int(block)
            _require_uint(block_idx, "block_idx", 0xFFFF)
            if block_idx in decoded[client_id]:
                raise TransferError("duplicate block index in coordinate metadata")
            _require_uint(count, "n_coords", 0xFFFF, minimum=1)
            decoded[client_id][block_idx] = count
    return decoded


def _canonical_json(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TransferError(f"invalid transfer metadata: {exc}") from exc


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TransferError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise TransferError(f"invalid JSON constant: {value}")


def _require_uint(value: object, name: str, maximum: int, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise TransferError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _validate_limit(value: int) -> int:
    return _require_uint(value, "max_transfer_bytes", 0xFFFFFFFF, minimum=1)


def _validate_endpoint(host: str, port: int) -> tuple[str, int]:
    if not isinstance(host, str) or not host:
        raise ValueError("host must be a non-empty string")
    _require_uint(port, "port", 65535)
    return host, port


def _validate_timeout(timeout: float) -> None:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")


def _validate_array_size(array: np.ndarray, limit: int) -> None:
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "fiu" or array.dtype.hasobject:
        raise TransferError("array payload must be a real numeric ndarray")
    if array.nbytes > limit:
        raise TransferError("array payload exceeds the configured transfer limit")


def _serialize_snapshot(array: np.ndarray, limit: int) -> bytes:
    _validate_array_size(array, limit)
    snapshot = np.array(array, copy=True, order="K")
    return serialize_array(snapshot)


__all__ = [
    "AggregateEnvelope",
    "AggregateReceiver",
    "MAX_TRANSFER_BYTES",
    "ModelBackedUpdateProvider",
    "ModelMailbox",
    "ModelSource",
    "PROTOCOL_VERSION",
    "TcpAggregateReceiver",
    "TcpAggregateSender",
    "TcpModelReceiver",
    "TransferError",
    "broadcast_model",
    "send_model",
]
