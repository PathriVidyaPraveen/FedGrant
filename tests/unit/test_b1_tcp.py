from __future__ import annotations

import asyncio
import socket
import struct
import time

import numpy as np
import pytest

from fedgrant.baselines.b1_tcp import (
    B1ClientCore,
    B1ProtocolError,
    B1RoundAccumulator,
    B1TcpClient,
    B1TransferError,
    apply_delivery_result,
    decode_block_frame,
    decode_hello,
    encode_block_frame,
    encode_hello,
)
from fedgrant.encoder.blocks import block_payload_f16


def _accumulator(*, d: int = 5, L: int = 2, now: int = 100) -> B1RoundAccumulator:
    return B1RoundAccumulator(
        round_id=3,
        client_ids=(7, 9),
        start_ns=50,
        deadline_ns=200,
        d=d,
        L=L,
        client_weights={7: 0.25, 9: 0.75},
        clock_ns=lambda: now,
    )


def _frame(block_idx: int, n_coords: int, values: tuple[float, ...]) -> bytes:
    return encode_block_frame(block_idx, n_coords, np.asarray(values, dtype=np.float16))


def test_b1_hello_round_trip_and_validation() -> None:
    encoded = encode_hello(12, 513)
    assert len(encoded) == 10
    assert decode_hello(encoded) == (12, 513)
    with pytest.raises(B1ProtocolError):
        decode_hello(encoded[:-1])
    with pytest.raises(B1ProtocolError):
        decode_hello(b"NOPE" + encoded[4:])


def test_block_frame_preserves_metadata_and_big_endian_fp16() -> None:
    frame = _frame(2, 1, (4.5, 0.0))
    assert frame[:4] == struct.pack("!HH", 2, 1)
    assert frame[4:] == np.asarray([4.5, 0.0], dtype=">f2").tobytes()
    block_idx, n_coords, payload = decode_block_frame(frame, 2)
    assert (block_idx, n_coords) == (2, 1)
    assert payload.dtype == np.float16
    np.testing.assert_array_equal(payload, [4.5, 0.0])


@pytest.mark.parametrize(
    "frame,L",
    [
        (b"", 2),
        (b"\x00\x00\x00\x00" + b"\x00\x00", 2),
        (struct.pack("!HH2e", 0, 0, 0.0, 0.0), 2),
        (struct.pack("!HH2e", 0, 3, 0.0, 0.0), 2),
        (struct.pack("!HH2e", 0, 1, 1.0, 2.0), 2),
        (struct.pack("!HH2e", 0, 2, float("inf"), 0.0), 2),
    ],
)
def test_malformed_block_frames_are_rejected(frame: bytes, L: int) -> None:
    with pytest.raises(B1ProtocolError):
        decode_block_frame(frame, L)


def test_encoder_rejects_wrong_payload_type_or_shape() -> None:
    with pytest.raises(B1ProtocolError):
        encode_block_frame(0, 1, [1.0, 0.0])  # type: ignore[arg-type]
    with pytest.raises(B1ProtocolError):
        encode_block_frame(0, 1, np.asarray([[1.0, 0.0]], dtype=np.float16))
    with pytest.raises(B1ProtocolError):
        encode_block_frame(0, 1, np.asarray([1.0, 2.0], dtype=np.float16))


def test_partial_and_coalesced_tcp_reads_are_framed_by_fixed_size() -> None:
    async def scenario() -> None:
        first = _frame(0, 2, (1.0, 2.0))
        second = _frame(1, 2, (3.0, 4.0))
        reader = asyncio.StreamReader()
        task = asyncio.create_task(reader.readexactly(len(first)))
        reader.feed_data(first[:3])
        await asyncio.sleep(0)
        assert not task.done()
        reader.feed_data(first[3:] + second)
        assert await task == first
        assert await reader.readexactly(len(second)) == second

        truncated = asyncio.StreamReader()
        truncated.feed_data(first[:4])
        truncated.feed_eof()
        with pytest.raises(asyncio.IncompleteReadError):
            await truncated.readexactly(len(first))

    asyncio.run(scenario())


def test_accumulator_accepts_complete_frames_only_by_deadline() -> None:
    accumulator = _accumulator()
    accumulator.mark_connected(7, 3)
    accumulator.accept_frame(7, _frame(0, 2, (1.0, 2.0)), 200)
    with pytest.raises(B1ProtocolError, match="after the round deadline"):
        accumulator.accept_frame(7, _frame(1, 2, (3.0, 4.0)), 201)
    accumulator.stop()
    result = accumulator.result(250)
    assert result.client_results[7].delivered_indices == (0,)
    assert result.client_results[7].decoded_update.tolist() == [1.0, 2.0, 0.0, 0.0, 0.0]
    assert result.client_results[7].rejected_frames == 1


def test_accumulator_validates_identity_duplicates_and_true_final_count() -> None:
    accumulator = _accumulator(d=3)
    with pytest.raises(B1ProtocolError, match="round"):
        accumulator.mark_connected(7, 2)
    with pytest.raises(B1ProtocolError, match="unknown"):
        accumulator.mark_connected(8, 3)
    accumulator.mark_connected(7, 3)
    with pytest.raises(B1ProtocolError, match="duplicate B1 client"):
        accumulator.mark_connected(7, 3)

    valid_final = _frame(1, 1, (5.0, 0.0))
    accumulator.accept_frame(7, valid_final, 150)
    with pytest.raises(B1ProtocolError, match="n_coords"):
        accumulator.accept_frame(7, _frame(0, 1, (6.0, 0.0)), 151)
    with pytest.raises(B1ProtocolError, match="duplicate B1 block"):
        accumulator.accept_frame(7, valid_final, 152)
    accumulator.stop()
    result = accumulator.result(220)
    assert result.client_results[7].n_coords == {1: 1}
    assert result.client_results[7].decoded_update.tolist() == [0.0, 0.0, 5.0]
    assert result.client_results[9].connected is False
    assert result.connection_failures == 1


def test_incomplete_frame_is_not_delivered_and_is_counted() -> None:
    accumulator = _accumulator()
    accumulator.mark_connected(7, 3)
    accumulator.accept_frame(7, _frame(0, 2, (1.0, 2.0)), 150)
    accumulator.mark_incomplete(7, 3, at_deadline=True)
    accumulator.stop()
    result = accumulator.result(250)
    assert result.client_results[7].delivered_indices == (0,)
    assert result.client_results[7].incomplete_frames == 1
    assert result.incomplete_blocks == 1


def test_aggregate_boundary_gets_zero_filled_updates_and_client_weights() -> None:
    accumulator = _accumulator(d=3)
    accumulator.mark_connected(7, 3)
    accumulator.accept_frame(7, _frame(1, 1, (8.0, 0.0)), 150)
    accumulator.stop()
    result = accumulator.result(220)
    seen: list[tuple[list[np.ndarray], np.ndarray]] = []

    def aggregate(updates: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
        seen.append((updates, weights.copy()))
        return (weights[0] * updates[0] + weights[1] * updates[1]).astype(np.float32)

    aggregate_result = result.aggregate(aggregate)
    assert result.client_n_coords == {7: {1: 1}, 9: {}}
    assert seen[0][0][0].tolist() == [0.0, 0.0, 8.0]
    assert seen[0][0][1].tolist() == [0.0, 0.0, 0.0]
    np.testing.assert_array_equal(seen[0][1], [0.25, 0.75])
    np.testing.assert_array_equal(aggregate_result, [0.0, 0.0, 2.0])
    assert result.goodput_for_duration(100_000_000) == 320.0
    ratio = result.energy_ratio(
        {
            7: np.asarray([1.0, 3.0], dtype=np.float64),
            9: np.asarray([1.0, 1.0], dtype=np.float64),
        }
    )
    assert ratio == pytest.approx((0.25**2 * 3.0) / (0.25**2 * 4.0 + 0.75**2 * 2.0))


def test_error_feedback_uses_gateway_delivery_set_once_and_persists() -> None:
    core = B1ClientCore(7, d=3, L=2)
    delta = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    encoded = core.start_round(3, 1_000, delta)
    accumulator = B1RoundAccumulator(
        round_id=3,
        client_ids=(7,),
        start_ns=100,
        deadline_ns=900,
        d=3,
        L=2,
        client_weights={7: 1.0},
        clock_ns=lambda: 100,
    )
    accumulator.mark_connected(7, 3)
    idx = int(encoded.order[0])
    payload, n_coords = block_payload_f16(encoded, idx)
    accumulator.accept_frame(7, encode_block_frame(idx, n_coords, payload), 500)
    accumulator.stop()
    result = accumulator.result(1_000)

    expected_hat = np.zeros(3, dtype=np.float32)
    expected_hat[idx * 2 : idx * 2 + n_coords] = payload[:n_coords].astype(np.float32)
    applied = apply_delivery_result(core, result)
    assert applied.shape == (3,)
    expected_residual = core.error_feedback.e.copy()
    np.testing.assert_array_equal(expected_residual, delta - expected_hat)
    with pytest.raises(B1TransferError, match="active round"):
        apply_delivery_result(core, result)

    core.start_round(4, 2_000, np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(core.round.u, expected_residual)


def test_connection_failure_is_observable() -> None:
    async def scenario() -> None:
        reservation = socket.socket()
        reservation.bind(("127.0.0.1", 0))
        unused_port = reservation.getsockname()[1]
        reservation.close()
        core = B1ClientCore(2, d=3, L=2)
        client = B1TcpClient(port=unused_port)
        result = await client.send_round(
            core,
            gateway_host="127.0.0.1",
            round_id=1,
            deadline_ns=time.monotonic_ns() + 300_000_000,
            delta=np.zeros(3, dtype=np.float32),
        )
        assert result.connected is False
        assert result.connection_error
        assert result.close_reason == "connection_failed"
        assert result.residual_finalized

    asyncio.run(scenario())
