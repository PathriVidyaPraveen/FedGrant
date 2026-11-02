from __future__ import annotations

import asyncio
import time

import numpy as np

from fedgrant.baselines.b1_tcp import (
    B1ClientCore,
    B1RoundAccumulator,
    B1TcpClient,
    B1TcpGateway,
    apply_delivery_result,
    encode_block_frame,
    encode_hello,
    write_metrics_csv,
)
from fedgrant.encoder.blocks import block_payload_f16


def test_real_tcp_b1_loopback_deadline_delivery_residuals_and_aggregation(tmp_path) -> None:
    async def scenario() -> None:
        round_id, d, L = 1, 5, 2
        duration_ns = 250_000_000
        start_ns = time.monotonic_ns()
        deadline_ns = start_ns + duration_ns
        cores = {
            cid: B1ClientCore(cid, d=d, L=L)
            for cid in (0, 1, 2)
        }
        updates = {
            0: np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            1: np.asarray([0.0, 0.0, 8.0, 0.0, 0.0], dtype=np.float32),
            2: np.asarray([9.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        cores[1].start_round(round_id, deadline_ns, updates[1])
        cores[2].start_round(round_id, deadline_ns, updates[2])

        accumulator = B1RoundAccumulator(
            round_id=round_id,
            client_ids=(0, 1, 2),
            start_ns=start_ns,
            deadline_ns=deadline_ns,
            d=d,
            L=L,
            client_weights={0: 0.2, 1: 0.3, 2: 0.5},
        )
        gateway = await B1TcpGateway(accumulator, host="127.0.0.1", port=0).start()
        client_task = asyncio.create_task(
            B1TcpClient(port=gateway.address[1]).send_round(
                cores[0],
                gateway_host=gateway.address[0],
                round_id=round_id,
                deadline_ns=deadline_ns,
                delta=updates[0],
            )
        )

        writers = []
        readers = []
        try:
            await asyncio.sleep(0.01)
            for cid in (1, 2):
                reader, writer = await asyncio.open_connection(*gateway.address)
                readers.append(reader)
                writers.append(writer)
                writer.write(encode_hello(round_id, cid))
                await writer.drain()

            encoded_one = cores[1].round.encoded
            first_idx = int(encoded_one.order[0])
            first_payload, first_count = block_payload_f16(encoded_one, first_idx)
            first_frame = encode_block_frame(first_idx, first_count, first_payload)
            writers[0].write(first_frame)
            await writers[0].drain()

            encoded_two = cores[2].round.encoded
            partial_idx = int(encoded_two.order[0])
            partial_payload, partial_count = block_payload_f16(encoded_two, partial_idx)
            partial_frame = encode_block_frame(partial_idx, partial_count, partial_payload)
            writers[1].write(partial_frame[:3])
            await writers[1].drain()

            result = await asyncio.wait_for(gateway.wait(), timeout=1.0)
            client_transfer = await asyncio.wait_for(client_task, timeout=1.0)
            assert client_transfer.connected
            assert client_transfer.blocks_written == 3
            assert client_transfer.close_reason in {"gateway_closed", "local_deadline"}
            assert client_transfer.application_bytes_written == 10 + 3 * (4 + 2 * L)

            assert result.client_results[0].delivered_indices == (0, 1, 2)
            assert result.client_results[0].receive_order == tuple(cores[0].round.encoded.order)
            assert result.client_results[0].close_reason == "gateway_deadline"
            assert result.client_results[1].delivered_indices == (first_idx,)
            assert result.client_results[1].receive_order == (first_idx,)
            assert result.client_results[2].delivered_indices == ()
            assert result.client_results[2].incomplete_frames == 1
            assert result.client_results[2].close_reason == "incomplete_frame_at_deadline"
            assert result.connection_failures == 0
            assert result.delivered_blocks == 4
            assert result.client_results[1].n_coords[first_idx] == int(
                cores[1].round.encoded.n_coords[first_idx]
            )
            assert len(result.client_results[0].delivered_bitmap) == 1
            eof_results = await asyncio.gather(*(reader.read() for reader in readers))
            assert eof_results == [b"", b""]

            # The orchestration boundary supplies the actual gateway bitmap. B1
            # itself has no specified delivery-report or acknowledgement message.
            for core in cores.values():
                apply_delivery_result(core, result)
            assert np.count_nonzero(cores[0].error_feedback.e) == 0
            assert np.count_nonzero(cores[2].error_feedback.e) > 0

            aggregate = result.aggregate(
                lambda client_updates, weights: sum(
                    (weight * update for weight, update in zip(weights, client_updates, strict=True)),
                    start=np.zeros(d, dtype=np.float32),
                ).astype(np.float32)
            )
            expected = sum(
                (result.client_weights[cid] * result.client_results[cid].decoded_update
                 for cid in sorted(result.client_results)),
                start=np.zeros(d, dtype=np.float32),
            )
            np.testing.assert_array_equal(aggregate, expected)
            assert result.client_n_coords[2] == {}
            assert result.completed_at_deadline
            assert result.close_reason == "gateway_deadline"
            assert result.goodput_for_duration(duration_ns) > 0

            csv_path = tmp_path / "b1-client-metrics.csv"
            write_metrics_csv(csv_path, result, {0: client_transfer})
            rows = csv_path.read_text(encoding="utf-8").splitlines()
            assert len(rows) == 4
            assert "delivered_bitmap_hex" in rows[0]
            assert "application_bytes_written" in rows[0]
        finally:
            for writer in writers:
                writer.close()
                await writer.wait_closed()
            if not client_task.done():
                client_task.cancel()
                await asyncio.gather(client_task, return_exceptions=True)
            await gateway.close()

    asyncio.run(scenario())
