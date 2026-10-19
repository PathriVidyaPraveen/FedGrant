"""Protocol definition invariants from Section 8 of the FedGrant specification."""

import math

import pytest

from fedgrant.common.constants import (
    COMMON_HEADER_BYTES,
    COMMON_HEADER_FORMAT,
    BYTE_ORDER,
    DATA_GRANT_ID,
    INTEGER_FIELDS_UNSIGNED,
    MAGIC,
    MessageType,
    PROTOCOL_VERSION,
    TransferMode,
)
from fedgrant.common.wire import (
    COMMON_HEADER_FIELDS,
    MESSAGE_DEFINITIONS,
    WireProtocolError,
    decode_message,
    encode_message,
)


def test_message_identifiers_and_protocol_constants() -> None:
    assert [int(t) for t in MessageType] == [1, 2, 3, 4, 5, 6]
    assert MAGIC == 0x4647
    assert PROTOCOL_VERSION == 1
    assert COMMON_HEADER_FORMAT == "!HBBIHHQ"
    assert COMMON_HEADER_BYTES == 20
    assert BYTE_ORDER == "big" and INTEGER_FIELDS_UNSIGNED is True
    assert DATA_GRANT_ID == 0xFFFFFFFF
    assert [int(mode) for mode in TransferMode] == [0, 1, 2, 3]


def test_common_header_layout() -> None:
    assert [(f.name, f.scalar, f.offset, f.reserved) for f in COMMON_HEADER_FIELDS] == [
        ("magic", "u16", 0, False),
        ("version", "u8", 2, False),
        ("type", "u8", 3, False),
        ("round_id", "u32", 4, False),
        ("client_id", "u16", 8, False),
        ("flags", "u16", 10, True),
        ("t_send_ns", "u64", 12, False),
    ]


def test_all_six_message_body_definitions() -> None:
    assert set(MESSAGE_DEFINITIONS) == set(MessageType)
    expected = {
        MessageType.ROUND_START: (
            "gateway-to-client", 20,
            [("deadline_ns", "u64", 1), ("K", "u16", 1), ("L", "u16", 1),
             ("rho_us", "u32", 1), ("mode", "u8", 1), ("pad", "u8", 3)],
        ),
        MessageType.READY: (
            "client-to-gateway", 4,
            [("K", "u16", 1), ("pad", "u16", 1), ("energy", "f32", "K")],
        ),
        MessageType.GRANT: (
            "gateway-to-client", 8,
            [("grant_id", "u32", 1), ("send_pos", "u16", 1), ("pad", "u16", 1)],
        ),
        MessageType.DATA: (
            "client-to-gateway", 12,
            [("grant_id", "u32", 1), ("send_pos", "u16", 1), ("block_idx", "u16", 1),
             ("n_coords", "u16", 1), ("pad", "u16", 1), ("payload", "f16", "L")],
        ),
        MessageType.ROUND_END: (
            "gateway-to-client", 4,
            [("K", "u16", 1), ("n_delivered", "u16", 1), ("bitmap", "bytes", "ceil(K/8)")],
        ),
        MessageType.PLAN: (
            "gateway-to-client (B3 only)", 8,
            [("k", "u16", 1), ("pad", "u16", 1), ("rate_bps", "u32", 1)],
        ),
    }
    for message_type, (direction, fixed_bytes, fields) in expected.items():
        spec = MESSAGE_DEFINITIONS[message_type]
        assert spec.direction == direction
        assert spec.body_fixed_bytes == fixed_bytes
        assert [(f.name, f.scalar, f.count) for f in spec.body_fields] == fields
        assert all(not f.reserved or f.name == "pad" for f in spec.body_fields)


def test_variable_message_payload_lengths_and_reserved_fields() -> None:
    assert MESSAGE_DEFINITIONS[MessageType.READY].body_variable_bytes == "4*K"
    assert MESSAGE_DEFINITIONS[MessageType.DATA].body_variable_bytes == "2*L"
    assert MESSAGE_DEFINITIONS[MessageType.ROUND_END].body_variable_bytes == "ceil(K/8)"
    for spec in MESSAGE_DEFINITIONS.values():
        for field in spec.body_fields:
            if field.name == "pad":
                assert field.reserved


def test_specified_udp_payload_lengths_at_default_block_count() -> None:
    from fedgrant.policy.constants import K, L

    assert COMMON_HEADER_BYTES + MESSAGE_DEFINITIONS[MessageType.ROUND_START].body_fixed_bytes == 40
    assert COMMON_HEADER_BYTES + 4 + 4 * K == 24 + 4 * K
    assert COMMON_HEADER_BYTES + MESSAGE_DEFINITIONS[MessageType.GRANT].body_fixed_bytes == 28
    assert COMMON_HEADER_BYTES + 12 + 2 * L == 1432
    assert COMMON_HEADER_BYTES + 4 + (K + 7) // 8 == 46
    assert COMMON_HEADER_BYTES + MESSAGE_DEFINITIONS[MessageType.PLAN].body_fixed_bytes == 28


_HEADER = dict(round_id=7, client_id=2, t_send_ns=123_456_789)
_BODIES = {
    MessageType.ROUND_START: {"deadline_ns": 987_654_321, "K": 3, "L": 4, "rho_us": 25, "mode": 2},
    MessageType.READY: {"K": 3, "energy": (4.0, 1.5, 0.0)},
    MessageType.GRANT: {"grant_id": 8, "send_pos": 3},
    MessageType.DATA: {
        "grant_id": DATA_GRANT_ID,
        "send_pos": 3,
        "block_idx": 2,
        "n_coords": 4,
        "payload": (0.0, -2.0, 1.0, 65504.0),
    },
    MessageType.ROUND_END: {"K": 3, "n_delivered": 2, "bitmap": b"\x05"},
    MessageType.PLAN: {"k": 3, "rate_bps": 4_000_000},
}


@pytest.mark.parametrize("message_type", list(MessageType))
def test_all_messages_round_trip(message_type: MessageType) -> None:
    packet = encode_message(message_type, body=_BODIES[message_type], block_length=4, **_HEADER)
    decoded = decode_message(packet, block_length=4)
    assert decoded.message_type is message_type
    assert (decoded.round_id, decoded.client_id, decoded.t_send_ns) == (7, 2, 123_456_789)
    expected = _BODIES[message_type]
    for name, value in expected.items():
        actual = decoded.body[name]
        if isinstance(value, tuple):
            assert actual == value
        else:
            assert actual == value


def test_round_start_exact_network_order_bytes() -> None:
    packet = encode_message(
        MessageType.ROUND_START,
        round_id=0x01020304,
        client_id=0x0506,
        t_send_ns=0x0708090A0B0C0D0E,
        body={"deadline_ns": 0x1011121314151617, "K": 0x1819, "L": 0x1A1B, "rho_us": 0x1C1D1E1F, "mode": 2},
    )
    assert packet == bytes.fromhex(
        "4647010101020304050600000708090a0b0c0d0e"
        "101112131415161718191a1b1c1d1e1f02000000"
    )


def test_ready_data_and_round_end_exact_variable_layouts() -> None:
    ready = encode_message(MessageType.READY, body={"K": 2, "energy": (1.0, 0.5)}, **_HEADER)
    assert ready[20:] == bytes.fromhex("000200003f8000003f000000")

    data = encode_message(
        MessageType.DATA,
        body={"grant_id": 1, "send_pos": 2, "block_idx": 3, "n_coords": 2, "payload": (1.0, -2.0)},
        block_length=2,
        **_HEADER,
    )
    assert data[20:] == bytes.fromhex("0000000100020003000200003c00c000")

    round_end = encode_message(
        MessageType.ROUND_END,
        body={"K": 9, "n_delivered": 2, "bitmap": bytes.fromhex("0101")},
        **_HEADER,
    )
    assert round_end[20:] == bytes.fromhex("000900020101")

    grant = encode_message(MessageType.GRANT, body={"grant_id": 0x01020304, "send_pos": 0x0506}, **_HEADER)
    assert grant[20:] == bytes.fromhex("0102030405060000")
    plan = encode_message(MessageType.PLAN, body={"k": 0, "rate_bps": 0x01020304}, **_HEADER)
    assert plan[20:] == bytes.fromhex("0000000001020304")


def test_float_representations_and_zero_are_preserved() -> None:
    ready = encode_message(MessageType.READY, body={"K": 3, "energy": (1.0, 0.5, 0.0)}, **_HEADER)
    assert ready[-12:] == bytes.fromhex("3f8000003f00000000000000")
    assert decode_message(ready).body["energy"] == (1.0, 0.5, 0.0)

    data = encode_message(
        MessageType.DATA,
        body={"grant_id": 1, "send_pos": 1, "block_idx": 0, "n_coords": 3, "payload": (0.0, 1.0, -2.0)},
        block_length=3,
        **_HEADER,
    )
    assert data[-6:] == bytes.fromhex("00003c00c000")
    assert decode_message(data, block_length=3).body["payload"] == (0.0, 1.0, -2.0)


@pytest.mark.parametrize(
    ("message_type", "body", "field"),
    [
        (MessageType.ROUND_START, {"deadline_ns": 0, "K": 1, "L": 1, "rho_us": 0, "mode": 9}, "mode"),
        (MessageType.GRANT, {"grant_id": 0, "send_pos": 1}, "grant_id"),
        (MessageType.GRANT, {"grant_id": 1, "send_pos": 0}, "send_pos"),
        (MessageType.PLAN, {"k": -1, "rate_bps": 1}, "k"),
        (MessageType.PLAN, {"k": 1, "rate_bps": 1 << 32}, "rate_bps"),
        (MessageType.READY, {"K": 1, "energy": (math.nan,)}, "energy"),
        (MessageType.READY, {"K": 1, "energy": (-1.0,)}, "energy"),
    ],
)
def test_encoder_rejects_invalid_integer_and_energy_values(
    message_type: MessageType, body: dict, field: str
) -> None:
    with pytest.raises(WireProtocolError, match=field):
        encode_message(message_type, body=body, **_HEADER)


def test_encoder_rejects_missing_extra_and_unrepresentable_values() -> None:
    with pytest.raises(WireProtocolError, match="mapping"):
        encode_message(MessageType.GRANT, body=[], **_HEADER)  # type: ignore[arg-type]
    with pytest.raises(WireProtocolError, match="missing"):
        encode_message(MessageType.GRANT, body={"grant_id": 1}, **_HEADER)
    with pytest.raises(WireProtocolError, match="unexpected"):
        encode_message(MessageType.GRANT, body={"grant_id": 1, "send_pos": 1, "pad": 0}, **_HEADER)
    with pytest.raises(WireProtocolError, match="representable"):
        encode_message(
            MessageType.DATA,
            body={"grant_id": 1, "send_pos": 1, "block_idx": 0, "n_coords": 1, "payload": (65520.0,)},
            block_length=1,
            **_HEADER,
        )


def test_decoder_rejects_malformed_common_header_and_reserved_fields() -> None:
    packet = bytearray(encode_message(MessageType.GRANT, body=_BODIES[MessageType.GRANT], **_HEADER))
    for offset, value, text in ((0, 0, "magic"), (2, 2, "version"), (3, 99, "type"), (10, 1, "flags"), (26, 1, "reserved")):
        bad = packet.copy()
        bad[offset] = value
        with pytest.raises(WireProtocolError, match=text):
            decode_message(bad)
    with pytest.raises(WireProtocolError, match="shorter"):
        decode_message(b"FG")


def test_decoder_rejects_truncation_counts_nonfinite_values_and_trailing_bytes() -> None:
    ready = encode_message(MessageType.READY, body={"K": 2, "energy": (2.0, 1.0)}, **_HEADER)
    with pytest.raises(WireProtocolError, match="truncated"):
        decode_message(ready[:-1])
    with pytest.raises(WireProtocolError, match="trailing"):
        decode_message(ready + b"\x00")

    bad_count = bytearray(ready)
    bad_count[21] = 3  # K changes from 2 to 3 without a third f32.
    with pytest.raises(WireProtocolError, match="truncated"):
        decode_message(bad_count)

    bad_nan = bytearray(ready)
    bad_nan[24:28] = bytes.fromhex("7fc00000")
    with pytest.raises(WireProtocolError, match="finite"):
        decode_message(bad_nan)

    data = bytearray(
        encode_message(
            MessageType.DATA,
            body={"grant_id": 1, "send_pos": 1, "block_idx": 0, "n_coords": 1, "payload": (1.0,)},
            block_length=1,
            **_HEADER,
        )
    )
    data[-2:] = bytes.fromhex("7e00")
    with pytest.raises(WireProtocolError, match="finite"):
        decode_message(data, block_length=1)


def test_round_end_bitmap_invariants_are_checked() -> None:
    with pytest.raises(WireProtocolError, match="population"):
        encode_message(
            MessageType.ROUND_END,
            body={"K": 9, "n_delivered": 1, "bitmap": bytes.fromhex("0300")},
            **_HEADER,
        )
    with pytest.raises(WireProtocolError, match="padding"):
        encode_message(
            MessageType.ROUND_END,
            body={"K": 9, "n_delivered": 2, "bitmap": bytes.fromhex("0102")},
            **_HEADER,
        )
