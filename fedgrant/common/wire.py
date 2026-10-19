"""Declarative FedGrant message and header field definitions (Section 8).

This module intentionally defines protocol shape only. Binary packing,
unpacking, and validation of received datagrams belong to the wire codec task.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from fedgrant.common.constants import (
    COMMON_HEADER_BYTES,
    COMMON_HEADER_FORMAT,
    DATA_GRANT_ID,
    L as DEFAULT_L,
    MAGIC,
    MessageType,
    PROTOCOL_VERSION,
    TransferMode,
)

Scalar = Literal["u8", "u16", "u32", "u64", "f16", "f32", "bytes"]


@dataclass(frozen=True, slots=True)
class FieldDefinition:
    """One ordered wire field; ``count`` may name a protocol-dependent length."""

    name: str
    scalar: Scalar
    count: int | str = 1
    reserved: bool = False
    offset: int | None = None
    meaning: str = ""


@dataclass(frozen=True, slots=True)
class MessageDefinition:
    """A message identifier, direction, and ordered body fields."""

    message_type: MessageType
    direction: str
    body_fields: tuple[FieldDefinition, ...]
    body_fixed_bytes: int
    body_variable_bytes: str | None = None


class WireProtocolError(ValueError):
    """Raised when a FedGrant datagram or message mapping is malformed.

    Encoders reject missing, unexpected, out-of-range, and non-finite values.
    Decoders reject malformed headers, reserved fields, counts, payload lengths,
    and invalid float values before returning a message.
    """


@dataclass(frozen=True, slots=True)
class DecodedMessage:
    """A validated common header and message body returned by :func:`decode_message`."""

    message_type: MessageType
    round_id: int
    client_id: int
    t_send_ns: int
    body: Mapping[str, Any]


COMMON_HEADER_FIELDS: tuple[FieldDefinition, ...] = (
    FieldDefinition("magic", "u16", offset=0, meaning="0x4647 ('FG')"),
    FieldDefinition("version", "u8", offset=2, meaning="1"),
    FieldDefinition("type", "u8", offset=3, meaning="MessageType identifier"),
    FieldDefinition("round_id", "u32", offset=4, meaning="Round number, starting at 1"),
    FieldDefinition("client_id", "u16", offset=8, meaning="Client identifier"),
    FieldDefinition("flags", "u16", offset=10, reserved=True, meaning="Must be zero"),
    FieldDefinition("t_send_ns", "u64", offset=12, meaning="Sender CLOCK_MONOTONIC time in ns"),
)


MESSAGE_DEFINITIONS: dict[MessageType, MessageDefinition] = {
    MessageType.ROUND_START: MessageDefinition(
        MessageType.ROUND_START,
        "gateway-to-client",
        (
            FieldDefinition("deadline_ns", "u64"),
            FieldDefinition("K", "u16"),
            FieldDefinition("L", "u16"),
            FieldDefinition("rho_us", "u32"),
            FieldDefinition("mode", "u8", meaning="TransferMode identifier"),
            FieldDefinition("pad", "u8", count=3, reserved=True, meaning="All zero"),
        ),
        body_fixed_bytes=20,
    ),
    MessageType.READY: MessageDefinition(
        MessageType.READY,
        "client-to-gateway",
        (
            FieldDefinition("K", "u16"),
            FieldDefinition("pad", "u16", reserved=True, meaning="Must be zero"),
            FieldDefinition("energy", "f32", count="K", meaning="Energy in send-position order"),
        ),
        body_fixed_bytes=4,
        body_variable_bytes="4*K",
    ),
    MessageType.GRANT: MessageDefinition(
        MessageType.GRANT,
        "gateway-to-client",
        (
            FieldDefinition("grant_id", "u32", meaning="Per-round ID starting at 1"),
            FieldDefinition("send_pos", "u16", meaning="1-based send position"),
            FieldDefinition("pad", "u16", reserved=True, meaning="Must be zero"),
        ),
        body_fixed_bytes=8,
    ),
    MessageType.DATA: MessageDefinition(
        MessageType.DATA,
        "client-to-gateway",
        (
            FieldDefinition("grant_id", "u32", meaning="0xFFFFFFFF in push/plan mode"),
            FieldDefinition("send_pos", "u16", meaning="1-based send position"),
            FieldDefinition("block_idx", "u16", meaning="0-based block index"),
            FieldDefinition("n_coords", "u16", meaning="True coordinate count in this block"),
            FieldDefinition("pad", "u16", reserved=True, meaning="Must be zero"),
            FieldDefinition("payload", "f16", count="L", meaning="Big-endian IEEE-754 binary16 coordinates"),
        ),
        body_fixed_bytes=12,
        body_variable_bytes="2*L",
    ),
    MessageType.ROUND_END: MessageDefinition(
        MessageType.ROUND_END,
        "gateway-to-client",
        (
            FieldDefinition("K", "u16"),
            FieldDefinition("n_delivered", "u16"),
            FieldDefinition("bitmap", "bytes", count="ceil(K/8)", meaning="Block-index bits, least-significant bit first"),
        ),
        body_fixed_bytes=4,
        body_variable_bytes="ceil(K/8)",
    ),
    MessageType.PLAN: MessageDefinition(
        MessageType.PLAN,
        "gateway-to-client (B3 only)",
        (
            FieldDefinition("k", "u16", meaning="Number of planned blocks"),
            FieldDefinition("pad", "u16", reserved=True, meaning="Must be zero"),
            FieldDefinition("rate_bps", "u32"),
        ),
        body_fixed_bytes=8,
    ),
}


_SCALAR_FORMATS: dict[Scalar, str] = {
    "u8": "B",
    "u16": "H",
    "u32": "I",
    "u64": "Q",
    "f16": "e",
    "f32": "f",
}
_SCALAR_BYTES: dict[Scalar, int] = {
    "u8": 1,
    "u16": 2,
    "u32": 4,
    "u64": 8,
    "f16": 2,
    "f32": 4,
    "bytes": 1,
}
_UINT_MAX: dict[Scalar, int] = {
    "u8": 0xFF,
    "u16": 0xFFFF,
    "u32": 0xFFFFFFFF,
    "u64": 0xFFFFFFFFFFFFFFFF,
}


def _require_uint(name: str, value: Any, scalar: Scalar, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WireProtocolError(f"{name} must be an unsigned {scalar}")
    maximum = _UINT_MAX[scalar]
    if not minimum <= value <= maximum:
        raise WireProtocolError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _require_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WireProtocolError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise WireProtocolError(f"{name} must be finite")
    return value


def _is_value_sequence(value: Any) -> bool:
    """Accept sized numeric containers, including NumPy arrays, but not text/bytes."""
    return (
        not isinstance(value, (str, bytes, bytearray, memoryview, Mapping))
        and hasattr(value, "__len__")
        and hasattr(value, "__iter__")
    )


def _require_block_length(block_length: Any) -> int:
    return _require_uint("block_length", block_length, "u16", minimum=1)


def _body_required_fields(definition: MessageDefinition) -> set[str]:
    return {field.name for field in definition.body_fields if not field.reserved}


def _require_body(definition: MessageDefinition, body: Mapping[str, Any]) -> None:
    if not isinstance(body, Mapping):
        raise WireProtocolError("body must be a mapping")
    required = _body_required_fields(definition)
    missing = required - set(body)
    extra = set(body) - required
    if missing:
        raise WireProtocolError(f"missing body fields: {sorted(missing)}")
    if extra:
        raise WireProtocolError(f"unexpected body fields: {sorted(extra)}")


def _body_count(field: FieldDefinition, body: Mapping[str, Any], block_length: int) -> int:
    if isinstance(field.count, int):
        return field.count
    if field.count == "K":
        return _require_uint("K", body["K"], "u16", minimum=1)
    if field.count == "L":
        return block_length
    if field.count == "ceil(K/8)":
        K = _require_uint("K", body["K"], "u16", minimum=1)
        return (K + 7) // 8
    raise AssertionError(f"unsupported schema count {field.count!r}")


def _encode_scalar(field: FieldDefinition, value: Any, count: int) -> bytes:
    if field.scalar == "bytes":
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise WireProtocolError(f"{field.name} must be bytes-like")
        raw = bytes(value)
        if len(raw) != count:
            raise WireProtocolError(f"{field.name} must be exactly {count} bytes")
        return raw
    if field.scalar in _UINT_MAX:
        if count == 1:
            values = (_require_uint(field.name, value, field.scalar),)
        else:
            if not _is_value_sequence(value):
                raise WireProtocolError(f"{field.name} must be a sequence of {count} values")
            if len(value) != count:
                raise WireProtocolError(f"{field.name} must contain exactly {count} values")
            values = tuple(_require_uint(field.name, item, field.scalar) for item in value)
        return struct.pack("!" + str(count) + _SCALAR_FORMATS[field.scalar], *values)
    if not _is_value_sequence(value):
        raise WireProtocolError(f"{field.name} must be a sequence of {count} values")
    if len(value) != count:
        raise WireProtocolError(f"{field.name} must contain exactly {count} values")
    values = tuple(_require_finite(field.name, item) for item in value)
    try:
        return struct.pack("!" + str(count) + _SCALAR_FORMATS[field.scalar], *values)
    except (OverflowError, struct.error) as exc:
        raise WireProtocolError(f"{field.name} contains a value not representable as {field.scalar}") from exc


def _validate_body_values(message_type: MessageType, body: Mapping[str, Any], block_length: int) -> None:
    if message_type is MessageType.ROUND_START:
        _require_uint("deadline_ns", body["deadline_ns"], "u64")
        _require_uint("K", body["K"], "u16", minimum=1)
        _require_uint("L", body["L"], "u16", minimum=1)
        try:
            TransferMode(_require_uint("mode", body["mode"], "u8"))
        except ValueError as exc:
            raise WireProtocolError("mode is not a supported transfer mode") from exc
    elif message_type is MessageType.READY:
        K = _require_uint("K", body["K"], "u16", minimum=1)
        energy = body["energy"]
        if not _is_value_sequence(energy):
            raise WireProtocolError("energy must be a sequence")
        if len(energy) != K:
            raise WireProtocolError("energy count must equal K")
        values = tuple(_require_finite("energy", value) for value in energy)
        if any(value < 0.0 for value in values):
            raise WireProtocolError("energy values must be non-negative")
        if any(right > left for left, right in zip(values, values[1:])):
            raise WireProtocolError("energy values must be non-increasing in send order")
    elif message_type is MessageType.GRANT:
        _require_uint("grant_id", body["grant_id"], "u32", minimum=1)
        _require_uint("send_pos", body["send_pos"], "u16", minimum=1)
    elif message_type is MessageType.DATA:
        grant_id = _require_uint("grant_id", body["grant_id"], "u32")
        if grant_id == 0:
            raise WireProtocolError("grant_id must start at 1 or be the push/plan sentinel")
        _require_uint("send_pos", body["send_pos"], "u16", minimum=1)
        _require_uint("block_idx", body["block_idx"], "u16")
        _require_uint("n_coords", body["n_coords"], "u16", minimum=1)
        if body["n_coords"] > block_length:
            raise WireProtocolError("n_coords cannot exceed block_length")
        payload = body["payload"]
        if not _is_value_sequence(payload):
            raise WireProtocolError("payload must be a sequence")
        if len(payload) != block_length:
            raise WireProtocolError("payload count must equal block_length")
        for value in payload:
            _require_finite("payload", value)
    elif message_type is MessageType.ROUND_END:
        K = _require_uint("K", body["K"], "u16", minimum=1)
        n_delivered = _require_uint("n_delivered", body["n_delivered"], "u16")
        if n_delivered > K:
            raise WireProtocolError("n_delivered cannot exceed K")
        bitmap = body["bitmap"]
        if not isinstance(bitmap, (bytes, bytearray, memoryview)):
            raise WireProtocolError("bitmap must be bytes-like")
        raw = bytes(bitmap)
        if len(raw) != (K + 7) // 8:
            raise WireProtocolError("bitmap length does not match K")
        if K % 8 and raw[-1] & ~((1 << (K % 8)) - 1):
            raise WireProtocolError("bitmap has non-zero padding bits")
        if sum(byte.bit_count() for byte in raw) != n_delivered:
            raise WireProtocolError("n_delivered must equal the bitmap population count")
    elif message_type is MessageType.PLAN:
        _require_uint("k", body["k"], "u16")
        _require_uint("rate_bps", body["rate_bps"], "u32")


def encode_message(
    message_type: MessageType | int,
    *,
    round_id: int,
    client_id: int,
    t_send_ns: int,
    body: Mapping[str, Any],
    block_length: int = DEFAULT_L,
) -> bytes:
    """Encode one complete, validated FedGrant UDP payload.

    The encoder supplies fixed magic/version and zeroes every reserved field.
    ``body`` contains exactly the non-reserved fields from ``MESSAGE_DEFINITIONS``.
    ``block_length`` is the negotiated ``L`` used only by the DATA payload shape;
    it defaults to the project's fixed value of 700.
    """
    try:
        message_type = MessageType(message_type)
    except (TypeError, ValueError) as exc:
        raise WireProtocolError("message_type is not supported") from exc
    definition = MESSAGE_DEFINITIONS[message_type]
    block_length = _require_block_length(block_length)
    _require_body(definition, body)
    round_id = _require_uint("round_id", round_id, "u32", minimum=1)
    client_id = _require_uint("client_id", client_id, "u16")
    t_send_ns = _require_uint("t_send_ns", t_send_ns, "u64")
    _validate_body_values(message_type, body, block_length)

    chunks = [struct.pack(COMMON_HEADER_FORMAT, MAGIC, PROTOCOL_VERSION, int(message_type), round_id, client_id, 0, t_send_ns)]
    for field in definition.body_fields:
        count = _body_count(field, body, block_length)
        if field.reserved:
            chunks.append(_encode_scalar(field, 0 if count == 1 else (0,) * count, count))
        else:
            chunks.append(_encode_scalar(field, body[field.name], count))
    encoded = b"".join(chunks)
    expected_length = COMMON_HEADER_BYTES + definition.body_fixed_bytes
    if definition.body_variable_bytes is not None:
        variable_field = definition.body_fields[-1]
        expected_length += _body_count(variable_field, body, block_length) * _SCALAR_BYTES[variable_field.scalar]
    if len(encoded) != expected_length:
        raise AssertionError("wire schema and encoder length disagree")
    return encoded


def _decode_values(data: bytes, offset: int, field: FieldDefinition, count: int) -> tuple[Any, int]:
    size = count * _SCALAR_BYTES[field.scalar]
    end = offset + size
    if end > len(data):
        raise WireProtocolError(f"truncated {field.name} field")
    if field.scalar == "bytes":
        return data[offset:end], end
    try:
        values = struct.unpack_from("!" + str(count) + _SCALAR_FORMATS[field.scalar], data, offset)
    except struct.error as exc:
        raise WireProtocolError(f"malformed {field.name} field") from exc
    # Numeric vectors stay tuples even when their count is one. This keeps the
    # decoded representation independent of negotiated K/L values.
    return (values[0] if field.scalar in _UINT_MAX and count == 1 else values), end


def decode_message(data: bytes | bytearray | memoryview, *, block_length: int = DEFAULT_L) -> DecodedMessage:
    """Decode and validate one complete FedGrant UDP payload.

    No trailing data is accepted. The resulting body is an immutable mapping;
    repeated float fields are tuples. ``WireProtocolError`` is raised for every
    malformed or unsupported input instead of leaking ``struct`` exceptions.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise WireProtocolError("datagram must be bytes-like")
    data = bytes(data)
    block_length = _require_block_length(block_length)
    if len(data) < COMMON_HEADER_BYTES:
        raise WireProtocolError("datagram is shorter than the common header")
    try:
        magic, version, raw_type, round_id, client_id, flags, t_send_ns = struct.unpack_from(COMMON_HEADER_FORMAT, data)
    except struct.error as exc:
        raise WireProtocolError("malformed common header") from exc
    if magic != MAGIC:
        raise WireProtocolError("invalid protocol magic")
    if version != PROTOCOL_VERSION:
        raise WireProtocolError("unsupported protocol version")
    try:
        message_type = MessageType(raw_type)
    except ValueError as exc:
        raise WireProtocolError("unsupported message type") from exc
    if round_id < 1:
        raise WireProtocolError("round_id must start at 1")
    if flags != 0:
        raise WireProtocolError("reserved header flags must be zero")

    definition = MESSAGE_DEFINITIONS[message_type]
    offset = COMMON_HEADER_BYTES
    body: dict[str, Any] = {}
    for field in definition.body_fields:
        if isinstance(field.count, str) and field.count in {"K", "ceil(K/8)"} and "K" not in body:
            raise AssertionError("schema places K-dependent field before K")
        count = _body_count(field, body, block_length)
        value, offset = _decode_values(data, offset, field, count)
        if field.reserved:
            values = (value,) if count == 1 else value
            if any(item != 0 for item in values):
                raise WireProtocolError(f"reserved {field.name} field must be zero")
        else:
            body[field.name] = value
    if offset != len(data):
        raise WireProtocolError("unexpected trailing bytes")
    _validate_body_values(message_type, body, block_length)
    return DecodedMessage(message_type, round_id, client_id, t_send_ns, MappingProxyType(body))


__all__ = [
    "COMMON_HEADER_BYTES",
    "COMMON_HEADER_FIELDS",
    "COMMON_HEADER_FORMAT",
    "FieldDefinition",
    "MESSAGE_DEFINITIONS",
    "MessageDefinition",
]
