from __future__ import annotations

import asyncio
import numpy as np
import pytest

from fedgrant.common.constants import DATA_GRANT_ID, MessageType
from fedgrant.common.wire import decode_message, encode_message
from fedgrant.baselines.b2_udp import (
    B2ClientCore,
    B2ProtocolError,
    B2RoundAccumulator,
    B2TransferError,
    B2UdpClient,
    apply_delivery_result,
    decode_data_packet,
    encode_data_packet,
)
from fedgrant.encoder.blocks import block_payload_f16, encode


def _packet(round_id: int = 3, client_id: int = 7, block_idx: int = 0, n_coords: int = 2):
    return encode_data_packet(
        round_id,
        client_id,
        block_idx,
        n_coords,
        np.asarray([1.5, 0.0], dtype=np.float16),
    )


def _accumulator(*, d: int = 5, L: int = 2) -> B2RoundAccumulator:
    return B2RoundAccumulator(
        round_id=3,
        client_ids=(7, 9),
        start_ns=100,
        deadline_ns=1_000,
        d=d,
        L=L,
        client_weights={7: 0.25, 9: 0.75},
    )


def test_packet_round_trip_preserves_identity_index_count_and_fp16() -> None:
    values = np.asarray([1.5, -2.25, 0.0], dtype=np.float16)
    packet = encode_data_packet(9, 513, 4, 2, values, send_pos=3, t_send_ns=123)
    message = decode_message(packet, block_length=3)
    assert message.message_type is MessageType.DATA
    assert (message.round_id, message.client_id, message.t_send_ns) == (9, 513, 123)
    assert message.body["grant_id"] == DATA_GRANT_ID
    assert (message.body["send_pos"], message.body["block_idx"], message.body["n_coords"]) == (3, 4, 2)
    assert message.body["payload"] == (1.5, -2.25, 0.0)
    round_id, client_id, send_pos, block_idx, n_coords, payload = decode_data_packet(packet, 3)
    assert (round_id, client_id, send_pos, block_idx, n_coords) == (9, 513, 3, 4, 2)
    assert payload.dtype == np.float16
    np.testing.assert_array_equal(payload, values)


@pytest.mark.parametrize(
    "packet,L",
    [
        (b"", 2),
        (_packet()[:-1], 2),
    ],
)
def test_malformed_datagrams_are_rejected(packet: bytes, L: int) -> None:
    with pytest.raises(B2ProtocolError):
        decode_data_packet(packet, L)


def test_b2_rejects_non_data_and_non_push_data_messages() -> None:
    ready = encode_message(
        MessageType.READY,
        round_id=3,
        client_id=7,
        t_send_ns=0,
        body={"K": 1, "energy": (1.0,)},
    )
    with pytest.raises(B2ProtocolError, match="DATA"):
        decode_data_packet(ready, 2)
    ordinary_data = encode_message(
        MessageType.DATA,
        round_id=3,
        client_id=7,
        t_send_ns=0,
        block_length=2,
        body={
            "grant_id": 1,
            "send_pos": 1,
            "block_idx": 0,
            "n_coords": 2,
            "payload": (1.5, 0.0),
        },
    )
    with pytest.raises(B2ProtocolError, match="sentinel"):
        decode_data_packet(ordinary_data, 2)


def test_packet_limit_and_encoder_payload_validation() -> None:
    with pytest.raises(ValueError, match="safe UDP payload"):
        B2ClientCore(1, d=730, L=730)
    with pytest.raises(B2ProtocolError):
        encode_data_packet(1, 1, 0, 1, np.asarray([1.0], dtype=np.float32))
    with pytest.raises(B2ProtocolError):
        encode_data_packet(1, 1, 0, 1, np.asarray([1.0, 2.0], dtype=np.float16))


class _FakeClock:
    def __init__(self, now_ns: int) -> None:
        self.now_ns = now_ns
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now_ns += int(round(seconds * 1_000_000_000))


class _FakeTransport:
    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self.packets: list[tuple[int, bytes]] = []

    def sendto(self, packet: bytes, addr: tuple[str, int]) -> None:
        assert addr == ("127.0.0.1", 9101)
        self.packets.append((self.clock.now_ns, packet))

    def close(self) -> None:
        pass


def test_client_sends_energy_order_once_at_configured_rate_without_grants() -> None:
    async def scenario() -> None:
        clock = _FakeClock(100)
        core = B2ClientCore(7, d=5, L=2)
        client = B2UdpClient(
            core,
            gateway_addr=("127.0.0.1", 9101),
            clock_ns=lambda: clock.now_ns,
            sleep=clock.sleep,
        )
        transport = _FakeTransport(clock)
        client.connection_made(transport)  # type: ignore[arg-type]
        delta = np.asarray([1.0, 1.0, 4.0, 0.0, 2.0], dtype=np.float32)
        encoded = encode(delta, 2)

        transfer = await client.send_round(
            round_id=3,
            deadline_ns=2_000_000_000,
            delta=delta,
            rate_bps=4_800.0,
        )

        expected_order = tuple(int(idx) for idx in encoded.order)
        assert transfer.sent_indices == expected_order
        assert transfer.packets_sent == encoded.K
        assert len({idx for idx in transfer.sent_indices}) == encoded.K
        assert [timestamp for timestamp, _ in transport.packets] == list(transfer.send_times_ns)
        packet_data = [decode_message(packet, block_length=2) for _, packet in transport.packets]
        assert tuple(entry.body["block_idx"] for entry in packet_data) == expected_order
        assert [entry.body["send_pos"] for entry in packet_data] == [1, 2, 3]
        assert [entry.body["grant_id"] for entry in packet_data] == [DATA_GRANT_ID] * encoded.K
        packet_interval_ns = (
            (len(transport.packets[0][1]) + 42) * 8 * 1_000_000_000 // 4_800
        )
        assert all(
            second - first >= packet_interval_ns
            for first, second in zip(transfer.send_times_ns, transfer.send_times_ns[1:])
        )
        assert len(clock.sleeps) == encoded.K - 1
        assert transfer.stop_reason == "all_blocks_sent"

    asyncio.run(scenario())


def test_client_stops_at_deadline_and_does_not_retransmit() -> None:
    async def scenario() -> None:
        clock = _FakeClock(0)
        core = B2ClientCore(7, d=6, L=2)
        client = B2UdpClient(
            core,
            gateway_addr=("127.0.0.1", 9101),
            clock_ns=lambda: clock.now_ns,
            sleep=clock.sleep,
        )
        transport = _FakeTransport(clock)
        client.connection_made(transport)  # type: ignore[arg-type]
        # 18-byte datagrams plus 42 wire bytes at 4,800 bit/s take 100 ms each.
        transfer = await client.send_round(
            round_id=3,
            deadline_ns=150_000_000,
            delta=np.asarray([3.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            rate_bps=4_800.0,
        )
        assert transfer.packets_sent == 2
        assert transfer.stop_reason == "deadline"
        assert transfer.completed_ns == 150_000_000
        assert all(timestamp < 150_000_000 for timestamp, _ in transport.packets)
        assert len({decode_data_packet(packet, 2)[3] for _, packet in transport.packets}) == 2

    asyncio.run(scenario())


def test_gateway_rejects_bad_identity_indices_duplicates_and_late_packets() -> None:
    accumulator = _accumulator(d=3)
    valid = _packet()
    with pytest.raises(B2ProtocolError, match="round"):
        accumulator.accept_packet(_packet(round_id=2), 200)
    with pytest.raises(B2ProtocolError, match="unknown"):
        accumulator.accept_packet(_packet(client_id=8), 200)
    with pytest.raises(B2ProtocolError, match="before the round start"):
        accumulator.accept_packet(_packet(client_id=9), 99)
    with pytest.raises(B2ProtocolError, match="outside"):
        accumulator.accept_packet(_packet(block_idx=2), 200)
    with pytest.raises(B2ProtocolError, match="n_coords"):
        accumulator.accept_packet(
            encode_data_packet(3, 7, 1, 2, np.asarray([1.0, 0.0], dtype=np.float16)),
            200,
        )
    accumulator.accept_packet(valid, 200)
    with pytest.raises(B2ProtocolError, match="duplicate"):
        accumulator.accept_packet(valid, 201)
    with pytest.raises(B2ProtocolError, match="deadline"):
        accumulator.accept_packet(_packet(client_id=9), 1_001)
    accumulator.stop(1_100)
    result = accumulator.result()
    assert result.rejected_rounds == 1
    assert result.rejected_unknown_clients == 1
    assert result.rejected_early_packets == 1
    assert result.rejected_malformed_packets == 2
    assert result.rejected_duplicates == 1
    assert result.rejected_late_packets == 1


def test_result_zero_fills_missing_blocks_and_aggregates_multiple_clients() -> None:
    accumulator = _accumulator(d=3)
    accumulator.accept_packet(
        encode_data_packet(3, 7, 1, 1, np.asarray([5.0, 0.0], dtype=np.float16)),
        500,
    )
    accumulator.stop(1_000)
    result = accumulator.result()
    assert result.client_results[7].delivered_indices == (1,)
    assert result.client_results[7].delivered_bitmap == b"\x02"
    np.testing.assert_array_equal(result.client_results[7].decoded_update, [0.0, 0.0, 5.0])
    assert result.client_results[9].delivered_indices == ()
    assert result.client_results[9].delivered_bitmap == b"\x00"
    np.testing.assert_array_equal(result.client_results[9].decoded_update, [0.0, 0.0, 0.0])
    assert result.client_n_coords == {7: {1: 1}, 9: {}}

    aggregate = result.aggregate(
        lambda updates, weights: (
            weights[0] * updates[0] + weights[1] * updates[1]
        ).astype(np.float32)
    )
    np.testing.assert_array_equal(aggregate, [0.0, 0.0, 1.25])


def test_delivery_result_finalizes_error_feedback_once_and_carries_to_next_round() -> None:
    core = B2ClientCore(7, d=3, L=2)
    delta = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    encoded = core.start_round(3, 1_000, delta)
    accumulator = _accumulator(d=3)
    idx = int(encoded.order[0])
    payload, count = block_payload_f16(encoded, idx)
    accumulator.accept_packet(encode_data_packet(3, 7, idx, count, payload), 500)
    accumulator.stop(1_000)
    result = accumulator.result()

    apply_delivery_result(core, result)
    residual = core.error_feedback.e.copy()
    assert np.count_nonzero(residual) > 0
    with pytest.raises(B2TransferError, match="active round"):
        apply_delivery_result(core, result)
    core.start_round(4, 2_000, np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(core.round.u, residual)
